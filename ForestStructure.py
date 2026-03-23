"""
Forest Structure Raster Extraction  (windowed-read, multi-resolution version)
===============================================================================
Supports variable grid cell sizes (16 m, 32 m, 48 m …).
The surrounding window is always a FIXED PHYSICAL SIZE:
    SURR_RADIUS_M  metres on each side of the cell centroid
    (default 3 * 16 = 48 m  →  7x7 pixels at 16 m resolution)

For rasters with a different pixel size (e.g. 20 m for 2009/2011) the window
pixel count is recomputed automatically from the physical radius.

Processing logic:
  - VolumeBirch (output) = VolumeBirch_raw + VolumeOtherBroadleave
  - Per-year: Prop_Pine, Prop_Birch, Prop_Spruce, ForestType
      ForestType  1=Birch-dominated  2=Pine-dominated
                  3=Spruce-dominated  4=Mixed-forest  (threshold 70 %)
                  NaN = total volume 0 or nodata
  - Surrounding: NaN-aware mean within SURR_RADIUS_M of each centroid
      -> at least 1 valid pixel -> mean of valid pixels
      -> all nodata -> NaN
      Same Prop_ / ForestType_ computed for surrounding window too.

Output columns (179 total):
  Cell_ID | centroid_x | centroid_y
  <var>_<year>  x88  (11 vars x 8 years, cell value)
  Surrounding_<var>_<year>  x88  (surrounding mean)

Dependencies:
    pip install geopandas rasterio pyogrio scipy numpy pandas tqdm
"""

import os
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.transform import rowcol
from scipy.ndimage import uniform_filter
from tqdm import tqdm

# ================================================================
# Configuration
# ================================================================

GRID_PATH  = r"D:\Data Finland\Disturbance\TestNK20\GridOutput\grid_16m_from_clip.gpkg"
GRID_LAYER = "grid_16m"
OUTPUT_CSV = r"D:\Data Finland\Disturbance\TestNK20\GridOutput\forest_structure_extracted.csv"

YEARS               = [2009, 2011, 2013, 2015, 2017, 2019, 2021, 2023]
DOMINANCE_THRESHOLD = 0.70

# Grid cell size (metres) -------------------------------------------
# Controls how many raster pixels are averaged for the CELL value:
#   GRID_CELL_M=16, raster=16 m  ->  1x1  (single centroid pixel)
#   GRID_CELL_M=32, raster=16 m  ->  2x2  (4 pixels averaged)
#   GRID_CELL_M=48, raster=16 m  ->  3x3  (9 pixels averaged)
# The SURROUNDING window is unaffected by this parameter.
GRID_CELL_M = 16      # <-- change to 32 or 48 when using a larger grid

# ── Surrounding window definition ────────────────────────────
# Always expressed as a physical distance in metres.
# 3 * 16 m = 48 m on each side  →  total 7x7 pixels at 16 m resolution.
# If the raster pixel size differs (e.g. 20 m), the pixel count is
# recomputed automatically so the physical coverage stays the same.
BASE_PIXEL_M  = 16    # reference pixel size (metres) – the grid's native resolution
SURR_HALF_PIX = 3     # half-window in BASE_PIXEL_M units (3 → radius = 48 m)
SURR_RADIUS_M = SURR_HALF_PIX * BASE_PIXEL_M   # 48 m  (do not change this line)

RASTER_TEMPLATES = {
    "BasalArea":             r"D:\Data Finland\Forest Structure\Basal Area\StandBasalArea_{year}.tif",
    "Age":                   r"D:\Data Finland\Forest Structure\Age\StandAge_{year}.tif",
    "MeanDiameter":          r"D:\Data Finland\Forest Structure\MeanDiameter\StandMeanDiameter_{year}.tif",
    "MeanHeight":            r"D:\Data Finland\Forest Structure\MeanHeight\StandMeanHeight_{year}.tif",
    "VolumePine":            r"D:\Data Finland\Forest Structure\VolumeSpecies\VolumePine_{year}.tif",
    "VolumeBirch_raw":       r"D:\Data Finland\Forest Structure\VolumeSpecies\VolumeBirch_{year}.tif",
    "VolumeOtherBroadleave": r"D:\Data Finland\Forest Structure\VolumeSpecies\VolumeOtherBroadleaves_{year}.tif",
    "VolumeSpruce":          r"D:\Data Finland\Forest Structure\VolumeSpecies\VolumeSpruce_{year}.tif",
}

# ================================================================
# Step 1 – Read grid
# ================================================================

print("=" * 60)
print("Step 1: Read grid file")
print("=" * 60)

grid = gpd.read_file(GRID_PATH, layer=GRID_LAYER, engine="pyogrio")
n_cells = len(grid)
print(f"  Grid cells : {n_cells:,}")
print(f"  Grid CRS   : {grid.crs}")

if "Cell_ID" not in grid.columns:
    raise ValueError(f"'Cell_ID' column not found. Available: {list(grid.columns)}")

cell_ids  = grid["Cell_ID"].values
centroids = grid.geometry.centroid

# Reference raster: get CRS
def find_first_raster():
    for tmpl in RASTER_TEMPLATES.values():
        p = tmpl.format(year=YEARS[0])
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No raster files found – check path configuration.")

ref_path = find_first_raster()
with rasterio.open(ref_path) as src:
    raster_crs = src.crs
    full_shape = (src.height, src.width)

print(f"  Raster CRS        : {raster_crs}")
print(f"  Full raster shape : {full_shape[0]:,} x {full_shape[1]:,}  "
      f"(~{full_shape[0]*full_shape[1]*4/1e9:.1f} GB if fully loaded)")
print(f"\n  Surrounding radius: {SURR_RADIUS_M} m each side  "
      f"(= {SURR_HALF_PIX} x {BASE_PIXEL_M} m pixels at base resolution)")

# Reproject centroids if needed
if grid.crs != raster_crs:
    print("  [INFO] CRS mismatch – reprojecting centroids ...")
    pts = centroids.to_crs(raster_crs)
else:
    pts = centroids

px_vals = pts.x.values
py_vals = pts.y.values

# Bounding box padded by SURR_RADIUS_M so edge cells get full windows
bbox       = pts.total_bounds     # [minx, miny, maxx, maxy]
win_left   = bbox[0] - SURR_RADIUS_M
win_bottom = bbox[1] - SURR_RADIUS_M
win_right  = bbox[2] + SURR_RADIUS_M
win_top    = bbox[3] + SURR_RADIUS_M

print(f"\n  Padded bounding box:")
print(f"    X: {win_left:.1f} -> {win_right:.1f}")
print(f"    Y: {win_bottom:.1f} -> {win_top:.1f}")

# ================================================================
# Step 2 – Extraction function
# ================================================================

def extract(raster_path: str):
    """
    1. Build a window from the padded bounding box using the raster's own transform.
    2. Compute the surrounding filter size in pixels from the physical radius
       (SURR_RADIUS_M) and this raster's actual pixel size.
       -> 16 m raster: window_size = 2*3+1 = 7
       -> 20 m raster: window_size = 2*ceil(48/20)+1 = 2*3+1 = 7 (covers >= 48 m)
       -> 32 m raster: window_size = 2*ceil(48/32)+1 = 2*2+1 = 5
       -> 48 m raster: window_size = 2*ceil(48/48)+1 = 2*1+1 = 3
    3. Recompute row/col indices using the window's own transform.
    4. Return cell_vals and surr_vals (NaN-aware mean).
    """
    with rasterio.open(raster_path) as src:
        pixel_m = abs(src.transform.a)   # actual pixel size of this raster

        # Surrounding half-window in pixels for this raster's resolution
        half_pix    = math.ceil(SURR_RADIUS_M / pixel_m)
        window_size = 2 * half_pix + 1   # always odd

        # Build and integer-round the read window
        win = from_bounds(win_left, win_bottom, win_right, win_top,
                          transform=src.transform)
        win = win.intersection(Window(0, 0, src.width, src.height))
        col_off  = int(round(win.col_off))
        row_off  = int(round(win.row_off))
        w_width  = int(round(win.width))
        w_height = int(round(win.height))
        win = Window(col_off, row_off, w_width, w_height)

        win_transform = src.window_transform(win)
        arr = src.read(1, window=win).astype(np.float32)
        nd  = src.nodata

    # Nodata mask
    nd_mask = (arr == nd) if nd is not None else np.zeros(arr.shape, dtype=bool)
    nd_mask |= np.isnan(arr)
    arr[nd_mask] = np.nan

    # NaN-aware surrounding mean (sum / count decomposition)
    #   window_size is adapted to this raster's pixel size so the physical
    #   coverage always matches SURR_RADIUS_M metres on each side.
    w2     = float(window_size ** 2)
    filled = np.where(nd_mask, 0.0, arr)
    valid  = (~nd_mask).astype(np.float32)

    sum_arr = uniform_filter(filled, size=window_size, mode="constant", cval=0.0) * w2
    cnt_arr = uniform_filter(valid,  size=window_size, mode="constant", cval=0.0) * w2

    with np.errstate(invalid="ignore", divide="ignore"):
        surr = np.where(cnt_arr > 0, sum_arr / cnt_arr, np.nan).astype(np.float32)

    # Cell mean: average all raster pixels within the grid cell footprint.
    # cell_win_size = round(GRID_CELL_M / pixel_m)
#     GRID_CELL_M=16, pixel=16 m -> 1x1 (centroid pixel only)
#     GRID_CELL_M=32, pixel=16 m -> 2x2 (4 pixels)
#     GRID_CELL_M=48, pixel=16 m -> 3x3 (9 pixels)
    cell_win_size = max(1, round(GRID_CELL_M / pixel_m))
    if cell_win_size == 1:
        cell_arr = arr   # no filtering needed, single pixel
    else:
        cw2      = float(cell_win_size ** 2)
        c_sum    = uniform_filter(filled, size=cell_win_size, mode="constant", cval=0.0) * cw2
        c_cnt    = uniform_filter(valid,  size=cell_win_size, mode="constant", cval=0.0) * cw2
        with np.errstate(invalid="ignore", divide="ignore"):
            cell_arr = np.where(c_cnt > 0, c_sum / c_cnt, np.nan).astype(np.float32)

    # Row/col indices for this raster's window transform
    local_rows, local_cols = rowcol(win_transform, px_vals, py_vals)
    local_rows = np.asarray(local_rows, dtype=np.int32)
    local_cols = np.asarray(local_cols, dtype=np.int32)

    in_b = ((local_rows >= 0) & (local_rows < w_height) &
            (local_cols >= 0) & (local_cols < w_width))
    r = np.clip(local_rows, 0, w_height - 1)
    c = np.clip(local_cols, 0, w_width  - 1)

    cell_v = cell_arr[r, c].copy()
    surr_v = surr    [r, c].copy()
    cell_v[~in_b] = np.nan
    surr_v[~in_b] = np.nan
    return cell_v, surr_v, pixel_m, window_size, cell_win_size

# ================================================================
# Step 3 – Extract all rasters
# ================================================================

print("\n" + "=" * 60)
print("Step 2: Extract raster values")
print("=" * 60)

data          = {}
missing_files = []
_empty        = lambda: np.full(n_cells, np.nan, dtype=np.float32)

total_tasks = len(RASTER_TEMPLATES) * len(YEARS)
with tqdm(total=total_tasks, unit="file") as pbar:
    for var_name, tmpl in RASTER_TEMPLATES.items():
        for year in YEARS:
            path = tmpl.format(year=year)
            pbar.set_description(f"{var_name}_{year}")
            if not os.path.exists(path):
                print(f"\n  [WARN] File not found: {path}")
                missing_files.append(path)
                data[(var_name, year, "cell")] = _empty()
                data[(var_name, year, "surr")] = _empty()
            else:
                cv, sv, pix_m, win_sz, cwin_sz = extract(path)
                data[(var_name, year, "cell")] = cv
                data[(var_name, year, "surr")] = sv
                tqdm.write(f"  {var_name}_{year}: pixel={pix_m:.0f} m  "
                           f"cell window={cwin_sz}x{cwin_sz}  "
                           f"surr window={win_sz}x{win_sz} "
                           f"(covers {(win_sz//2)*pix_m:.0f} m each side)")
            pbar.update(1)

# ================================================================
# Step 4 – Derived variables
# ================================================================

print("\n" + "=" * 60)
print("Step 3: Compute derived variables")
print("=" * 60)

def get(var, year, kind):
    return data.get((var, year, kind), _empty())

def nan_add(a, b):
    both_nan = np.isnan(a) & np.isnan(b)
    result   = np.nansum(np.stack([a, b], axis=0), axis=0).astype(np.float32)
    result[both_nan] = np.nan
    return result

def props_and_foresttype(pine, birch, spruce):
    total = pine + birch + spruce
    with np.errstate(invalid="ignore", divide="ignore"):
        pp = np.where(total > 0, pine   / total, np.nan).astype(np.float32)
        pb = np.where(total > 0, birch  / total, np.nan).astype(np.float32)
        ps = np.where(total > 0, spruce / total, np.nan).astype(np.float32)
    ft = np.where(np.isnan(total), np.nan,
                  np.full(n_cells, 4.0, dtype=np.float32))
    ft = np.where(pp > DOMINANCE_THRESHOLD, 2.0, ft)
    ft = np.where(pb > DOMINANCE_THRESHOLD, 1.0, ft)
    ft = np.where(ps > DOMINANCE_THRESHOLD, 3.0, ft)
    return pp, pb, ps, ft.astype(np.float32)

for year in YEARS:
    for kind in ("cell", "surr"):
        birch  = nan_add(get("VolumeBirch_raw", year, kind),
                         get("VolumeOtherBroadleave", year, kind))
        pine   = get("VolumePine",   year, kind)
        spruce = get("VolumeSpruce", year, kind)
        pp, pb, ps, ft = props_and_foresttype(pine, birch, spruce)
        data[("VolumeBirch", year, kind)] = birch
        data[("Prop_Pine",   year, kind)] = pp
        data[("Prop_Birch",  year, kind)] = pb
        data[("Prop_Spruce", year, kind)] = ps
        data[("ForestType",  year, kind)] = ft

for year in YEARS:
    for kind in ("cell", "surr"):
        data.pop(("VolumeBirch_raw",       year, kind), None)
        data.pop(("VolumeOtherBroadleave", year, kind), None)

print("  Done.")

# ================================================================
# Step 5 – Assemble DataFrame
# ================================================================

print("\n" + "=" * 60)
print("Step 4: Assemble output table")
print("=" * 60)

ORDERED_VARS = [
    "BasalArea", "Age", "MeanDiameter", "MeanHeight",
    "VolumePine", "VolumeBirch", "VolumeSpruce",
    "Prop_Pine", "Prop_Birch", "Prop_Spruce", "ForestType",
]

out = {
    "Cell_ID"    : cell_ids,
    "centroid_x" : px_vals,
    "centroid_y" : py_vals,
}
for year in YEARS:
    for var in ORDERED_VARS:
        out[f"{var}_{year}"] = data.get((var, year, "cell"), _empty())
for year in YEARS:
    for var in ORDERED_VARS:
        out[f"Surrounding_{var}_{year}"] = data.get((var, year, "surr"), _empty())

result_df = pd.DataFrame(out, index=grid.index)
n_cols = len(result_df.columns)
print(f"  Rows    : {len(result_df):,}")
print(f"  Columns : {n_cols}")

# ================================================================
# Step 6 – Write CSV
# ================================================================

print("\n" + "=" * 60)
print("Step 5: Write CSV")
print("=" * 60)

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
result_df.to_csv(OUTPUT_CSV, index=False, float_format="%.4f")

print(f"  Done!  {OUTPUT_CSV}")
print(f"  Rows: {len(result_df):,}  |  Columns: {n_cols}")

if missing_files:
    print(f"\n  WARNING – {len(missing_files)} file(s) not found (columns set to NaN):")
    for f in missing_files:
        print(f"    - {f}")

print("\nColumn preview (first 25):")
for c in list(result_df.columns)[:25]:
    print(f"  {c}")
print("  ...")
