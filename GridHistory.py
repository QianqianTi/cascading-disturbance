"""
Grid Event History Extractor  –  v3
=====================================
Changes vs v2
  - Date parsing: explicit format list tried in order; garbled / unparseable
    dates are coerced to NaT and the corresponding Finland rows are DROPPED
    before the spatial join (only reports with a valid year are kept).
  - Year sanity filter: years outside [1980, 2030] are treated as garbled
    and dropped (adjust YEAR_MIN / YEAR_MAX if needed).
  - Centroid fix: centroid is computed in the grid's native projected CRS,
    then the single point layer is reprojected to EPSG:4326 for lat/long.
    This silences the "geometry in geographic CRS" warning and gives correct
    coordinates.
  - dayfirst warning: explicit format list replaces the ambiguous dayfirst flag.

Output columns
--------------
  Cell_ID             Unique identifier (from grid)
  lat / long          Centroid coordinates (WGS84 degrees)
  Report_ID           Cell_ID + "_" + sequential number (chronological)
  Report_Start_Year   Year of the PREVIOUS report (NA for first)
  Report_End_Year     Year of this report
  Wind / BB / Frost / Snow / Drought / Fire / Flood /
  Nutritional / Fungal / Insect / Animal / Harvest / Wiping
                      Binary 0/1 per disturbance type
  No_Damage           Binary 0/1  (qualifier is null)
  Stand_ID            Stand identifier from Finland polygon
  Stand_ID_area       Geometric area of the Finland polygon (m²)
  Polygon_Prop        % of the grid cell covered by the polygon
  Time_Interval       Report_End_Year − Report_Start_Year  (NA for first)
  Management_type     Value of cuttingrealizationpractice (numeric)
  Management_Plan     0 = first report in this cell, 1 = subsequent report
"""

import os
import time
import warnings
import geopandas as gpd
import pandas as pd
import numpy as np

# Progress bar: use tqdm if available, otherwise plain print
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    print("  (tqdm not installed – plain progress messages will be used)")

def _ts():
    """Return elapsed time since script start as a short string."""
    return f"[+{time.time() - _T0:.0f}s]"

_T0 = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# 1. Settings
# ─────────────────────────────────────────────────────────────────────────────

GRID_GPKG  = r"D:\Data Finland\Disturbance\TestNK20\GridOutput\grid_16m.gpkg"
GRID_LAYER = "grid_16m"

FINLAND_GPKG  = r"E:\dataprepare\Data\MKI_Finland.gpkg"
FINLAND_LAYER = "Finland"

OUTPUT_DIR = r"D:\Data Finland\Disturbance\TestNK20\GridEventHistory"

DATE_FIELD      = "standarrivaldate"
QUALIFIER_FIELD = "forestdamagequalifier"
STAND_ID_FIELD  = "Stand_ID"
MGMT_FIELD      = "cuttingrealizationpractice"

# Acceptable year range — rows outside this window are treated as garbled
YEAR_MIN = 1980
YEAR_MAX = 2030

# ─────────────────────────────────────────────────────────────────────────────
# 2. Disturbance code map  (name → set of qualifier codes)
# ─────────────────────────────────────────────────────────────────────────────

DISTURBANCE_MAP = {
    "Wind"        : {1504},
    "BB"          : {1602, 1603, 1606, 1613},
    "Frost"       : {1500},
    "Snow"        : {1501},
    "Drought"     : {1502},
    "Fire"        : {1503},
    "Flood"       : {1508},
    "Nutritional" : {1530, 1531, 1533, 1549},
    "Fungal"      : {1550, 1556, 1557, 1558},
    "Insect"      : {1600, 1601, 1608},
    "Animal"      : {1650, 1655, 1656},
    "Harvest"     : {1752},
    "Wiping"      : {1753},
}

ALL_KNOWN_CODES = {c for codes in DISTURBANCE_MAP.values() for c in codes}
DIST_COLS       = list(DISTURBANCE_MAP.keys()) + ["No_Damage"]

def parse_qualifier(val):
    """Return list of integer codes, or [] if null / unparseable.

    Handles:
      - NaN / None                 → []
      - Integer  1504              → [1504]
      - Float    1504.0            → [1504]   ← common in GeoPackage numeric fields
      - String   "1504"            → [1504]
      - String   "1504.0"          → [1504]
      - Multi    "1504,1602"       → [1504, 1602]
      - Multi    "1504.0;1602.0"   → [1504, 1602]
    """
    if pd.isna(val):
        return []
    # Numeric types (int, float, numpy scalar) — fast path
    try:
        fval = float(val)
        if fval == int(fval):          # e.g. 1504.0 → 1504
            return [int(fval)]
        return []                      # fractional float → ignore
    except (TypeError, ValueError):
        pass
    # String path — may be comma / semicolon separated
    codes = []
    for token in str(val).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            fval = float(token)
            if fval == int(fval):
                codes.append(int(fval))
        except ValueError:
            pass                       # truly garbled token — skip
    return codes

def has_code(codes, code_set):
    return 1 if any(c in code_set for c in codes) else 0

def make_flags(codes):
    """Return dict of binary flags for all disturbance types + No_Damage."""
    flags = {col: has_code(codes, cs) for col, cs in DISTURBANCE_MAP.items()}
    flags["No_Damage"] = 1 if not codes else 0
    return flags

# ─────────────────────────────────────────────────────────────────────────────
# 3. Robust date parser
# ─────────────────────────────────────────────────────────────────────────────

# Try formats in order; return NaT if none succeed
_DATE_FORMATS = [
    "%Y-%m-%d",    # ISO  2018-07-15
    "%d.%m.%Y",    # FI   15.07.2018
    "%d/%m/%Y",    # EU   15/07/2018
    "%m/%d/%Y",    # US   07/15/2018
    "%Y%m%d",      # compact 20180715
]

def robust_parse_date(series: pd.Series) -> pd.Series:
    """
    Try each format in _DATE_FORMATS in sequence.
    Values that match no format end up as NaT.
    Emits no warnings.
    """
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    remaining_mask = result.isna()  # all True initially

    for fmt in _DATE_FORMATS:
        if not remaining_mask.any():
            break
        subset = series[remaining_mask]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(subset, format=fmt, errors="coerce")
        good = parsed.notna()
        result[remaining_mask & good.reindex(series.index, fill_value=False)] = \
            parsed[good].values

        # Recompute remaining mask
        remaining_mask = result.isna()

    return result

# ─────────────────────────────────────────────────────────────────────────────
# 4. Read & prepare Finland event layer
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Reading Finland event layer ...")
finland = gpd.read_file(FINLAND_GPKG, layer=FINLAND_LAYER)
finland = finland[finland.geometry.notnull() & ~finland.geometry.is_empty].copy()
print(f"  {_ts()} Finland polygons (raw)   : {len(finland):,}")

# ── Parse dates robustly ──────────────────────────────────────────────────────
finland[DATE_FIELD] = robust_parse_date(finland[DATE_FIELD].astype(str))
finland["Event_Year"] = finland[DATE_FIELD].dt.year

n_nat  = finland["Event_Year"].isna().sum()
n_bad  = ((finland["Event_Year"] < YEAR_MIN) | (finland["Event_Year"] > YEAR_MAX)).sum()
print(f"  {_ts()} Rows with unparseable date : {n_nat:,}")
print(f"  {_ts()} Rows with out-of-range year ({YEAR_MIN}–{YEAR_MAX}): {n_bad:,}")

# Drop garbled rows (unparseable OR out-of-range year)
finland = finland[
    finland["Event_Year"].notna() &
    (finland["Event_Year"] >= YEAR_MIN) &
    (finland["Event_Year"] <= YEAR_MAX)
].copy()
print(f"  {_ts()} Finland polygons (clean)  : {len(finland):,}")

# ── Binary disturbance flags ──────────────────────────────────────────────────
print(f"  {_ts()} Computing disturbance flags ...")
finland["_codes"] = finland[QUALIFIER_FIELD].apply(parse_qualifier)
_flags_df = finland["_codes"].apply(make_flags).apply(pd.Series)
finland = pd.concat([finland, _flags_df], axis=1)

# Stand polygon area in projected CRS
finland["_stand_area"] = finland.geometry.area

# Slim down — include all disturbance flag columns
_keep = (
    [DATE_FIELD, "Event_Year"]
    + DIST_COLS
    + ["_stand_area", MGMT_FIELD, "geometry"]
)
if STAND_ID_FIELD in finland.columns:
    _keep.insert(-1, STAND_ID_FIELD)
else:
    print(f"  WARNING: '{STAND_ID_FIELD}' not found – Stand_ID will be NA")

events_slim = finland[[c for c in _keep if c in finland.columns]].copy()
events_slim = events_slim.reset_index(drop=True)
events_slim.index.name = "_fin_idx"

# ─────────────────────────────────────────────────────────────────────────────
# 5. Read & prepare grid layer
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{_ts()} Reading grid layer ...")
grid = gpd.read_file(GRID_GPKG, layer=GRID_LAYER)
grid = grid[grid.geometry.notnull() & ~grid.geometry.is_empty].copy()
print(f"  {_ts()} Grid cells: {len(grid):,}")

if "Cell_ID" not in grid.columns:
    grid["Cell_ID"] = range(len(grid))

# Reproject grid to Finland CRS if needed (must be projected for area/centroid)
if grid.crs != events_slim.crs:
    print(f"  Reprojecting grid {grid.crs} → {events_slim.crs}")
    grid = grid.to_crs(events_slim.crs)

# ── Centroid in projected CRS, then convert to WGS84 ─────────────────────────
# Compute centroids while still in projected CRS (avoids the geographic-CRS warning)
centroids_proj = grid.geometry.centroid                          # GeoSeries, projected
centroids_gdf  = gpd.GeoDataFrame(geometry=centroids_proj, crs=grid.crs)
centroids_wgs  = centroids_gdf.to_crs(epsg=4326)
grid["lat"]    = centroids_wgs.geometry.y
grid["long"]   = centroids_wgs.geometry.x

grid["_cell_area"] = grid.geometry.area

grid_sub = grid[["Cell_ID", "lat", "long", "_cell_area", "geometry"]].copy()

# ─────────────────────────────────────────────────────────────────────────────
# 6. Spatial join
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{_ts()} Spatial join (grid ← Finland events) ...")
events_for_join = events_slim.reset_index()   # _fin_idx becomes a regular column

joined = gpd.sjoin(
    grid_sub,
    events_for_join,
    how="left",
    predicate="intersects",
)
print(f"  {_ts()} Joined rows: {len(joined):,}")

# Map Stand_ID_area via _fin_idx
stand_area_map = events_slim["_stand_area"].to_dict()
joined["Stand_ID_area"] = joined["_fin_idx"].map(stand_area_map)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Polygon_Prop  –  vectorised overlay
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{_ts()} Computing polygon coverage proportions via spatial overlay ...")

events_for_ov = events_for_join[["_fin_idx", "geometry"]].copy()
print(f"  ({len(grid_sub):,} grid cells × {len(events_for_ov):,} Finland polygons – may take a few minutes)")
ov = gpd.overlay(
    grid_sub[["Cell_ID", "_cell_area", "geometry"]],
    events_for_ov,
    how="intersection",
    keep_geom_type=False,
)
print(f"  {_ts()} Overlay done – {len(ov):,} intersection patches")
ov["_inter_area"] = ov.geometry.area
ov["_poly_prop"]  = (ov["_inter_area"] / ov["_cell_area"] * 100).round(4)

prop_lookup = ov.set_index(["Cell_ID", "_fin_idx"])["_poly_prop"].to_dict()
print(f"  {_ts()} Proportion lookup built ({len(prop_lookup):,} entries)")

def _get_prop(row):
    fin_idx = row["_fin_idx"]
    if pd.isna(fin_idx):
        return 0.0
    return prop_lookup.get((row["Cell_ID"], int(fin_idx)), 0.0)

print(f"  {_ts()} Mapping Polygon_Prop to {len(joined):,} rows ...")
joined["Polygon_Prop"] = joined.apply(_get_prop, axis=1)
print(f"  {_ts()} Polygon_Prop done")

# ─────────────────────────────────────────────────────────────────────────────
# 7b. Deduplicate: within each (Cell_ID, Event_Year), keep only ONE
#     No_Damage row.  Disturbed rows are always kept as-is.
# ─────────────────────────────────────────────────────────────────────────────

n_before = len(joined)

# Mark No_Damage rows (qualifier is null → codes = [] → No_Damage flag = 1)
# We rely on the already-computed No_Damage column carried from events_slim.
_is_nodmg = joined["No_Damage"] == 1

# For No_Damage rows: drop duplicates on (Cell_ID, Event_Year), keep first.
# For disturbed rows: keep all.
nodmg_deduped = (
    joined[_is_nodmg]
    .drop_duplicates(subset=["Cell_ID", "Event_Year"], keep="first")
)
disturbed = joined[~_is_nodmg]

joined = pd.concat([disturbed, nodmg_deduped], ignore_index=True)
n_after = len(joined)
print(f"  {_ts()} No_Damage dedup: {n_before:,} → {n_after:,} rows "
      f"(removed {n_before - n_after:,} duplicate no-damage rows)")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Build chronological sequence per Cell_ID  –  fully vectorised
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{_ts()} Building per-cell chronological sequences (vectorised) ...")

DIST_COLS = list(DISTURBANCE_MAP.keys()) + ["No_Damage"]

sort_cols = [c for c in ["Cell_ID", "Event_Year", DATE_FIELD] if c in joined.columns]
df = joined.sort_values(sort_cols, na_position="first").reset_index(drop=True)

# ── seq number and Management_Plan ───────────────────────────────────────────
df["_seq"]          = df.groupby("Cell_ID").cumcount() + 1
print(f"  {_ts()} seq numbers assigned")
df["Management_Plan"] = (df["_seq"] > 1).astype(int)
df["Report_ID"]     = df["Cell_ID"].astype(str) + "_" + df["_seq"].astype(str)

# ── Report_Start_Year and Time_Interval via "previous distinct year" ──────────
print(f"  {_ts()} Computing start-year / interval ...")
# Within each cell (sorted by year), rows sharing the same year all inherit the
# same prev_distinct_year = the last year seen before that year block started.
#
#   step 1: shift Event_Year within group → previous row's year
#   step 2: keep the shifted value only where the year actually changed
#           (or at the first row of a cell); NaN elsewhere
#   step 3: forward-fill within cell → every row now carries the correct prev year

df["_year_shifted"]     = df.groupby("Cell_ID")["Event_Year"].shift(1)
df["_year_changed"]     = (df["Event_Year"] != df["_year_shifted"]) | (df["_seq"] == 1)
df["_prev_year_marker"] = df["_year_shifted"].where(df["_year_changed"])
df["_prev_distinct_yr"] = (
    df.groupby("Cell_ID")["_prev_year_marker"]
      .transform(lambda s: s.ffill())
)

df["Report_Start_Year"] = df["_prev_distinct_yr"]
df["Time_Interval"]     = (df["Event_Year"] - df["_prev_distinct_yr"]).where(
    df["_prev_distinct_yr"].notna()
)

# Rename / select final columns
rename_map = {
    "Event_Year"   : "Report_End_Year",
    STAND_ID_FIELD : "Stand_ID",
    MGMT_FIELD     : "Management_type",
}
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

# Drop internal helper columns
drop_cols = [c for c in df.columns if c.startswith("_")]
df = df.drop(columns=drop_cols, errors="ignore")
print(f"  {_ts()} Sequences done – {len(df):,} rows, {df['Cell_ID'].nunique():,} cells")

records = df  # DataFrame, not list of dicts

# ─────────────────────────────────────────────────────────────────────────────
# 9. Assemble & save
# ─────────────────────────────────────────────────────────────────────────────

result = records  # already a DataFrame

out_cols = (
    ["Cell_ID", "lat", "long",
     "Report_ID", "Report_Start_Year", "Report_End_Year"]
    + DIST_COLS
    + ["Stand_ID", "Stand_ID_area", "Polygon_Prop",
       "Time_Interval", "Management_type", "Management_Plan"]
)
result = result[[c for c in out_cols if c in result.columns]]

out_csv = os.path.join(OUTPUT_DIR, f"{GRID_LAYER}_event_history.csv")
print(f"\n{_ts()} Saving CSV → {out_csv} ...")
result.to_csv(out_csv, index=False, encoding="utf-8-sig")

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'─'*55}")
print(f"  Total elapsed            : {time.time()-_T0:.1f} s")
print(f"  Grid cells processed     : {result['Cell_ID'].nunique():,}")
print(f"  Total report rows        : {len(result):,}")
for col in DIST_COLS:
    if col in result.columns:
        print(f"  {col:<16}: {result[col].sum():,}")
print(f"  Management_Plan = 1  : {(result['Management_Plan'] == 1).sum():,}")
print(f"  Saved → {out_csv}")
print("Done!")