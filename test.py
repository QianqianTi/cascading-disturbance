import geopandas as gpd

gpkg_path = r"D:\Data Finland\Disturbance\TestNK\WBB_valid_grids.gpkg"

layers = [
    "grid_15m_valid",
    "grid_20m_valid",
    "grid_25m_valid",
    "grid_30m_valid",
    "grid_40m_valid",
    "grid_50m_valid"
]

for layer_name in layers:
    print(f"Processing {layer_name}...")
    
    gdf = gpd.read_file(gpkg_path, layer=layer_name)
    
    if len(gdf) == 0:
        print(f"  {layer_name} is empty, skipped.")
        continue

    # 如果原来有 Grid_ID，就先备份
    if "Grid_ID" in gdf.columns and "Old_ID" not in gdf.columns:
        gdf["Old_ID"] = gdf["Grid_ID"]

    # 重新编号，从 1 开始
    gdf["Grid_ID"] = range(1, len(gdf) + 1)

    # 保存回原图层
    gdf.to_file(gpkg_path, layer=layer_name, driver="GPKG")

    print(f"  Done: {len(gdf)} features renumbered from 1.")

print("All layers finished.")