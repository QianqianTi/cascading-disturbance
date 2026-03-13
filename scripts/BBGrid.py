import os
import pandas as pd
import geopandas as gpd

# =========================
# 1. Paths
# =========================
candidate_gpkg = r"D:\Data Finland\Disturbance\TestNK\grid_WBB_candidate.gpkg"
damage_gpkg = r"D:\Data Finland\Disturbance\TestNK\MKI_Pohjois-Karjala\MKI_Pohjois-Karjala.gpkg"
damage_layer = "NK_BBDamage"

sizes = [15, 20, 25, 30, 40, 50]

# =========================
# 2. Read BB damage layer
# =========================
print("Reading BB damage layer...")
bb_gdf = gpd.read_file(damage_gpkg, layer=damage_layer)

print("BB damage rows:", len(bb_gdf))
print("BB damage columns:", list(bb_gdf.columns))

# -------------------------
# Parse date field
# -------------------------
date_field = "standarrivaldate"

if date_field in bb_gdf.columns:
    bb_gdf[date_field] = pd.to_datetime(
        bb_gdf[date_field],
        format="%d/%m/%Y",
        errors="coerce"
    )
    bb_gdf["BBYear"] = bb_gdf[date_field].dt.year
else:
    raise ValueError(f"Field '{date_field}' not found in NK_BBDamage layer.")

# 去掉没有几何的记录
bb_gdf = bb_gdf[~bb_gdf.geometry.isna()].copy()

# =========================
# 3. Loop through each grid size
# =========================
for size in sizes:
    print("\n" + "=" * 60)
    print(f"Processing {size}m grid...")

    grid_layer = f"grid_{size}m_WBB_candidate"
    out_layer = f"grid_{size}m_with_BBinfo"

    # -------------------------
    # Read candidate grid
    # -------------------------
    grid_gdf = gpd.read_file(candidate_gpkg, layer=grid_layer)
    grid_gdf = grid_gdf[~grid_gdf.geometry.isna()].copy()

    print(f"Grid rows: {len(grid_gdf)}")

    # -------------------------
    # CRS alignment
    # -------------------------
    if grid_gdf.crs != bb_gdf.crs:
        print(f"Reprojecting BB damage from {bb_gdf.crs} to {grid_gdf.crs}")
        bb_use = bb_gdf.to_crs(grid_gdf.crs)
    else:
        bb_use = bb_gdf.copy()

    # -------------------------
    # Spatial join
    # Keep every matching BB record
    # -------------------------
    joined = gpd.sjoin(
        grid_gdf,
        bb_use,
        how="inner",
        predicate="intersects"
    )

    print(f"Matched rows: {len(joined)}")

    # 去掉 sjoin 产生的索引字段
    for col in ["index_right", "index_left"]:
        if col in joined.columns:
            joined = joined.drop(columns=[col])

    # 排序
    sort_cols = []
    if "Grid_ID" in joined.columns:
        sort_cols.append("Grid_ID")
    if "BBYear" in joined.columns:
        sort_cols.append("BBYear")

    if sort_cols:
        joined = joined.sort_values(sort_cols).reset_index(drop=True)

    # -------------------------
    # Write to GPKG
    # -------------------------
    joined.to_file(candidate_gpkg, layer=out_layer, driver="GPKG")

    print(f"Saved layer: {out_layer}")

print("\nAll done.")
