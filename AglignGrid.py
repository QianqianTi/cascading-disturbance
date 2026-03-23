import os
import time

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import geometry_mask
from shapely.geometry import box


# =========================================================
# 1. Input
# =========================================================
vector_path = r"E:\dataprepare\Data\MKI_Finland.gpkg"
vector_layer = "Finland"

raster_path = r"E:\dataprepare\Data\ForestType_2021.tif"

out_folder = r"D:\Data Finland\Disturbance\TestNK20\GridOutput"
os.makedirs(out_folder, exist_ok=True)

# Grid resolution in meters: 16 / 32 / 48 / 64 ...
target_grid_size = 16

clipped_raster_path = os.path.join(out_folder, f"ForestType_2021_clip_{target_grid_size}m.tif")
grid_path = os.path.join(out_folder, f"grid_{target_grid_size}m.gpkg")
grid_layer_name = f"grid_{target_grid_size}m"

progress_step_polygons = 20000
save_driver = "GPKG"


# =========================================================
# 2. Helper functions
# =========================================================
def print_block(title):
    print("\n" + "=" * 70)
    print(title)


def elapsed_msg(start_time):
    return f"{time.time() - start_time:.1f}s"


def read_and_prepare_vector(vector_path, vector_layer, target_crs=None):
    t = time.time()
    print_block("Step 1: Reading study area vector")

    gdf = gpd.read_file(vector_path, layer=vector_layer)
    print(f"Loaded polygons: {len(gdf):,}")

    if gdf.empty:
        raise ValueError("Input vector layer is empty.")

    if target_crs is not None and gdf.crs != target_crs:
        print("Reprojecting vector to match raster CRS...")
        gdf = gdf.to_crs(target_crs)

    dissolved = gdf.dissolve()
    print("Vector dissolved to single mask geometry.")
    print(f"Finished in {elapsed_msg(t)}")
    return dissolved


def clip_raster_by_vector(raster_path, study_area_gdf, clipped_raster_path):
    t = time.time()
    print_block("Step 2: Clipping raster by study area")

    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(
            src,
            study_area_gdf.geometry,
            crop=True,
            all_touched=True  
        )

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width":  out_image.shape[2],
            "transform": out_transform
        })

    with rasterio.open(clipped_raster_path, "w", **out_meta) as dst:
        dst.write(out_image)

    print(f"Clipped raster saved to: {clipped_raster_path}")
    print(f"Clip finished in {elapsed_msg(t)}")


def get_valid_mask(data, nodata):
    """Return a boolean mask where True = pixel has a real data value."""
    if nodata is None:
        if np.issubdtype(data.dtype, np.floating):
            return ~np.isnan(data)
        return np.ones(data.shape, dtype=bool)

    if np.issubdtype(data.dtype, np.floating) and np.isnan(nodata):
        return ~np.isnan(data)

    return data != nodata


def build_grid_from_clipped_raster(clipped_raster_path, target_grid_size,
                                    study_area_gdf,
                                    progress_step=20000):
    """
    Build an aggregated grid aligned to the clipped raster extent.

    A grid cell is kept whenever at least one pixel inside the block falls
    within the study area vector boundary — regardless of whether the
    underlying raster pixel carries a valid (non-NoData) value.

    Output GeoDataFrame columns:
        Cell_ID    – sequential integer ID
        GridSize   – grid cell size in metres
        ValidPix   – pixels with real raster values  (may be 0 for NoData areas)
        InsidePix  – pixels that fall inside the study area boundary (always >= 1)
    """
    t = time.time()
    print_block("Step 3: Reading clipped raster")

    with rasterio.open(clipped_raster_path) as src:
        data      = src.read(1)
        transform = src.transform
        crs       = src.crs
        nodata    = src.nodata
        width     = src.width
        height    = src.height
        res_x, res_y = src.res
        res_y = abs(res_y)

    print(f"Raster size  : {width:,} cols x {height:,} rows")
    print(f"Base resolution: {res_x} x {res_y} m")
    print(f"Target grid size: {target_grid_size} m")
    print(f"NoData value : {nodata}")
    print(f"Finished in {elapsed_msg(t)}")

    if target_grid_size % int(res_x) != 0:
        raise ValueError(
            f"target_grid_size={target_grid_size} is not an integer multiple "
            f"of raster resolution {res_x}."
        )

    factor = int(round(target_grid_size / res_x))
    print(f"Aggregation factor: {factor} x {factor} pixels per block")

    # ------------------------------------------------------------------
    # Build study-area mask from vector boundary
    #   geometry_mask returns True  = OUTSIDE the vector shapes
    #   ~geometry_mask returns True = INSIDE  the vector shapes
    # ------------------------------------------------------------------
    print("\nRasterising study area boundary to pixel mask ...")
    t_mask = time.time()
    inside_mask = ~geometry_mask(
        study_area_gdf.geometry,
        transform=transform,
        invert=False,
        out_shape=(height, width),
        all_touched=True 
    )
    print(f"Inside-mask built in {elapsed_msg(t_mask)}")

    # Valid-data mask (used for statistics only, NOT for filtering)
    valid_mask = get_valid_mask(data, nodata)

    # ------------------------------------------------------------------
    # Build aggregated grid blocks
    # ------------------------------------------------------------------
    t_grid = time.time()
    print_block("Step 4: Building aggregated aligned grid")

    n_block_rows  = height // factor
    n_block_cols  = width  // factor
    total_blocks  = n_block_rows * n_block_cols

    print(f"Full block rows : {n_block_rows:,}")
    print(f"Full block cols : {n_block_cols:,}")
    print(f"Total candidate blocks: {total_blocks:,}")
    print("Note: Incomplete edge blocks are discarded.")

    geoms               = []
    cell_ids            = []
    valid_pixel_counts  = []
    inside_pixel_counts = []

    kept      = 0
    processed = 0

    for br in range(n_block_rows):
        row_start = br * factor
        row_end   = row_start + factor

        for bc in range(n_block_cols):
            col_start = bc * factor
            col_end   = col_start + factor

            # Keep block if ANY pixel within it lies inside the study area
            block_inside = inside_mask[row_start:row_end, col_start:col_end]
            n_inside = int(block_inside.sum())

            if n_inside > 0:
                x_left,  y_top    = transform * (col_start, row_start)
                x_right  = x_left  + factor * res_x
                y_bottom = y_top   - factor * res_y

                geoms.append(box(x_left, y_bottom, x_right, y_top))
                kept += 1
                cell_ids.append(kept)

                block_valid = valid_mask[row_start:row_end, col_start:col_end]
                valid_pixel_counts.append(int(block_valid.sum()))
                inside_pixel_counts.append(n_inside)

            processed += 1

            if processed % progress_step == 0 or processed == total_blocks:
                pct = processed / total_blocks * 100
                print(
                    f"  Progress: {processed:,}/{total_blocks:,} "
                    f"({pct:.2f}%) | kept: {kept:,} | elapsed: {elapsed_msg(t_grid)}"
                )

    grid_gdf = gpd.GeoDataFrame(
        {
            "Cell_ID":   cell_ids,
            "GridSize":  [target_grid_size] * kept,
            "ValidPix":  valid_pixel_counts,    # real raster values; can be 0
            "InsidePix": inside_pixel_counts,   # pixels inside study area; >= 1
        },
        geometry=geoms,
        crs=crs
    )

    print(f"\nGrid building finished in {elapsed_msg(t_grid)}")
    print(f"Final grid cells kept: {len(grid_gdf):,}")
    return grid_gdf


def save_grid(grid_gdf, grid_path, layer_name, driver="GPKG"):
    t = time.time()
    print_block("Step 5: Saving grid")

    grid_gdf.to_file(grid_path, layer=layer_name, driver=driver)

    print(f"Saved to   : {grid_path}")
    print(f"Layer name : {layer_name}")
    print(f"Grid cells : {len(grid_gdf):,}")
    print(f"Finished in {elapsed_msg(t)}")


# =========================================================
# 3. Main
# =========================================================
def main():
    total_start = time.time()
    print_block("START")

    # --- Pre-flight checks ---
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        raster_res = src.res[0]
        print(f"Raster CRS        : {raster_crs}")
        print(f"Raster resolution : {src.res}")

    if target_grid_size % int(raster_res) != 0:
        raise ValueError(
            f"target_grid_size ({target_grid_size} m) must be an integer "
            f"multiple of raster resolution ({raster_res} m)."
        )

    # --- Step 1: Load and dissolve study area ---
    study_area = read_and_prepare_vector(
        vector_path=vector_path,
        vector_layer=vector_layer,
        target_crs=raster_crs
    )

    # --- Step 2: Clip raster to study area ---
    clip_raster_by_vector(
        raster_path=raster_path,
        study_area_gdf=study_area,
        clipped_raster_path=clipped_raster_path
    )

    # --- Steps 3-4: Build grid (vector-boundary-based filtering) ---
    grid_gdf = build_grid_from_clipped_raster(
        clipped_raster_path=clipped_raster_path,
        target_grid_size=target_grid_size,
        study_area_gdf=study_area,          # <-- drives keep/discard logic
        progress_step=progress_step_polygons
    )

    # --- Step 5: Save ---
    save_grid(
        grid_gdf=grid_gdf,
        grid_path=grid_path,
        layer_name=grid_layer_name,
        driver=save_driver
    )

    print_block("ALL FINISHED")
    print(f"Total elapsed : {time.time() - total_start:.1f} s")
    print(f"Final grid cells: {len(grid_gdf):,}")


if __name__ == "__main__":
    main()