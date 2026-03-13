import sqlite3
import pandas as pd
import geopandas as gpd

# =========================
# 1. Paths and settings
# =========================
valid_gpkg = r"D:\Data Finland\Disturbance\TestNK\WBB_valid_grids.gpkg"
candidate_gpkg = r"D:\Data Finland\Disturbance\TestNK\grid_WBB_candidate.gpkg"

sizes = [15, 20, 25, 30, 40, 50]
buffer_dist = 500  # meters

# =========================
# 2. Loop through scales
# =========================
for size in sizes:
    print("\n" + "=" * 60)
    print(f"Processing {size}m ...")

    valid_layer = f"grid_{size}m_valid"
    bb_layer = f"grid_{size}m_with_BBinfo"

    # -------------------------
    # Read data
    # -------------------------
    print("Reading valid grids...")
    valid_gdf = gpd.read_file(valid_gpkg, layer=valid_layer)
    valid_gdf = valid_gdf[~valid_gdf.geometry.isna()].copy()

    print("Reading BB info layer...")
    bb_gdf = gpd.read_file(candidate_gpkg, layer=bb_layer)
    bb_gdf = bb_gdf[~bb_gdf.geometry.isna()].copy()

    print("Valid rows:", len(valid_gdf))
    print("BB rows:", len(bb_gdf))

    # -------------------------
    # Check fields
    # -------------------------
    required_valid = ["Grid_ID", "BByear"]
    required_bb = ["Grid_ID", "BBYear"]

    for col in required_valid:
        if col not in valid_gdf.columns:
            raise ValueError(f"[{valid_layer}] Missing field: {col}")

    for col in required_bb:
        if col not in bb_gdf.columns:
            raise ValueError(f"[{bb_layer}] Missing field: {col}")

    # -------------------------
    # Numeric conversion
    # -------------------------
    valid_gdf["Grid_ID"] = pd.to_numeric(valid_gdf["Grid_ID"], errors="coerce")
    valid_gdf["BByear"] = pd.to_numeric(valid_gdf["BByear"], errors="coerce")

    bb_gdf["Grid_ID"] = pd.to_numeric(bb_gdf["Grid_ID"], errors="coerce")
    bb_gdf["BBYear"] = pd.to_numeric(bb_gdf["BBYear"], errors="coerce")

    valid_gdf = valid_gdf.dropna(subset=["Grid_ID", "BByear"]).copy()
    bb_gdf = bb_gdf.dropna(subset=["Grid_ID", "BBYear"]).copy()

    valid_gdf["Grid_ID"] = valid_gdf["Grid_ID"].astype(int)
    valid_gdf["BByear"] = valid_gdf["BByear"].astype(int)

    bb_gdf["Grid_ID"] = bb_gdf["Grid_ID"].astype(int)
    bb_gdf["BBYear"] = bb_gdf["BBYear"].astype(int)

    # -------------------------
    # CRS alignment
    # -------------------------
    if valid_gdf.crs != bb_gdf.crs:
        print(f"Reprojecting BB layer from {bb_gdf.crs} to {valid_gdf.crs}")
        bb_gdf = bb_gdf.to_crs(valid_gdf.crs)

    # -------------------------
    # Prepare geometry
    # -------------------------
    valid_poly = valid_gdf.copy()

    valid_buf = valid_gdf.copy()
    valid_buf["geometry"] = valid_buf.geometry.centroid.buffer(buffer_dist)
    valid_buf = valid_buf[["Grid_ID", "BByear", "geometry"]].copy()

    bb_pts = bb_gdf.copy()
    bb_pts["geometry"] = bb_pts.geometry.centroid
    bb_pts = bb_pts.rename(columns={"Grid_ID": "BB_Grid_ID"})

    # -------------------------
    # Spatial join
    # -------------------------
    joined = gpd.sjoin(
        bb_pts[["BB_Grid_ID", "BBYear", "geometry"]],
        valid_buf,
        how="inner",
        predicate="within"
    )

    print("Matched point-buffer pairs:", len(joined))

    # -------------------------
    # Remove self matches
    # -------------------------
    joined = joined[joined["BB_Grid_ID"] != joined["Grid_ID"]].copy()
    print("After removing self-matches:", len(joined))

    # -------------------------
    # Time filter
    # -------------------------
    joined = joined[joined["BBYear"] < joined["BByear"]].copy()
    print("After time filtering (BBYear < BByear):", len(joined))

    # -------------------------
    # Count unique neighboring BB grids
    # -------------------------
    count_df = (
        joined.groupby("Grid_ID")["BB_Grid_ID"]
        .nunique()
        .reset_index()
        .rename(columns={"BB_Grid_ID": "BBN_500m_pre"})
    )

    print("Count preview:")
    print(count_df.head())

    # -------------------------
    # Join back to valid layer
    # -------------------------
    valid_out = valid_poly.copy()

    if "BBN_500m_pre" in valid_out.columns:
        valid_out = valid_out.drop(columns=["BBN_500m_pre"])

    valid_out = valid_out.merge(count_df, on="Grid_ID", how="left")
    valid_out["BBN_500m_pre"] = valid_out["BBN_500m_pre"].fillna(0).astype(int)

    print("Joined preview:")
    print(valid_out[["Grid_ID", "BByear", "BBN_500m_pre"]].head(10))

    # -------------------------
    # Overwrite original valid layer
    # -------------------------
    conn = sqlite3.connect(valid_gpkg)
    cur = conn.cursor()

    cur.execute("DELETE FROM gpkg_contents WHERE table_name = ?", (valid_layer,))
    cur.execute("DELETE FROM gpkg_geometry_columns WHERE table_name = ?", (valid_layer,))
    cur.execute("DELETE FROM gpkg_extensions WHERE table_name = ?", (valid_layer,))
    cur.execute(f'DROP TABLE IF EXISTS "{valid_layer}"')

    conn.commit()
    conn.close()

    valid_out.to_file(valid_gpkg, layer=valid_layer, driver="GPKG")

    print(f"Done: {valid_layer}")

print("\nAll scales finished.")
