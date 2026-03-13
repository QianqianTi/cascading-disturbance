import os
import pandas as pd
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats

# =========================
# 1. Paths
# =========================
grid_gpkg = r"D:\Data Finland\Disturbance\TestNK\WBB_valid_grids.gpkg"
out_folder = r"D:\Data Finland\Disturbance\TestNK\BasalArea"
os.makedirs(out_folder, exist_ok=True)

# =========================
# 2. Grid layers
# =========================
grid_layers = {
    "15m": "grid_15m_valid",
    "20m": "grid_20m_valid",
    "25m": "grid_25m_valid",
    "30m": "grid_30m_valid",
    "40m": "grid_40m_valid",
    "50m": "grid_50m_valid"
}

# =========================
# 3. BA rasters
# =========================
ba_rasters = {
    2009: r"D:\Data Finland\Forest Structure\Basal Area\StandBasalArea_2009.tif",
    2011: r"D:\Data Finland\Forest Structure\Basal Area\StandBasalArea_2011.tif",
    2013: r"D:\Data Finland\Forest Structure\Basal Area\StandBasalArea_2013.tif",
    2015: r"D:\Data Finland\Forest Structure\Basal Area\StandBasalArea_2015.tif",
    2017: r"D:\Data Finland\Forest Structure\Basal Area\StandBasalArea_2017.tif",
    2019: r"D:\Data Finland\Forest Structure\Basal Area\StandBasalArea_2019.tif",
    2021: r"D:\Data Finland\Forest Structure\Basal Area\StandBasalArea_2021.tif",
    2023: r"D:\Data Finland\Forest Structure\Basal Area\StandBasalArea_2023.tif"
}

# =========================
# 4. Check raster files
# =========================
for year, raster_path in ba_rasters.items():
    if not os.path.exists(raster_path):
        raise FileNotFoundError(f"Raster not found: {raster_path}")

# =========================
# 5. Process each grid layer
# =========================
for scale, layer_name in grid_layers.items():
    print("=" * 70)
    print(f"Processing layer: {layer_name}")

    # 读取网格
    gdf = gpd.read_file(grid_gpkg, layer=layer_name)

    if gdf.empty:
        print(f"Layer {layer_name} is empty, skipped.")
        continue

    if "Grid_ID" not in gdf.columns:
        raise ValueError(f"'Grid_ID' not found in layer {layer_name}")

    # 去掉空几何
    gdf = gdf[gdf.geometry.notnull()].copy()

    print(f"Grid count: {len(gdf)}")
    print(f"Grid CRS: {gdf.crs}")

    # 结果表：只保留 Grid_ID
    result = gdf[["Grid_ID"]].copy()

    # 逐年提取
    for year, raster_path in ba_rasters.items():
        print(f"  Extracting BA_{year} ...")

        with rasterio.open(raster_path) as src:
            raster_crs = src.crs
            raster_nodata = src.nodata

        print(f"    Raster CRS: {raster_crs}")
        print(f"    Raster nodata: {raster_nodata}")

        # CRS 对齐
        if gdf.crs != raster_crs:
            gdf_proj = gdf.to_crs(raster_crs)
        else:
            gdf_proj = gdf.copy()

        # 关键：all_touched=True
        # 只要 polygon 和像元有一点接触就统计
        zs = zonal_stats(
            gdf_proj,
            raster_path,
            stats=["count", "mean"],
            all_touched=True,
            nodata=raster_nodata,
            geojson_out=False
        )

        result[f"BAcount_{year}"] = [item["count"] for item in zs]
        result[f"BA_{year}"] = [item["mean"] for item in zs]

        non_null_mean = pd.Series(result[f"BA_{year}"]).notna().sum()
        positive_count = (pd.Series(result[f"BAcount_{year}"]).fillna(0) > 0).sum()

        print(f"    Grids with touched valid pixels: {positive_count}/{len(result)}")
        print(f"    Grids with BA mean extracted:   {non_null_mean}/{len(result)}")

    # 排序
    result = result.sort_values("Grid_ID").reset_index(drop=True)

    # 输出
    out_csv = os.path.join(out_folder, f"BA_{scale}.csv")
    result.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"Saved to: {out_csv}")

print("=" * 70)
print("All BA tables finished.")