import re
import sqlite3
import pandas as pd
import geopandas as gpd

# =========================
# 1. Scales
# =========================
sizes = [20, 25, 30, 40, 50]

# =========================
# 2. Helper: extract event order
# =========================
def extract_event_order(event_id):
    if pd.isna(event_id):
        return None
    m = re.search(r'_(\d+)$', str(event_id))
    if m:
        return int(m.group(1))
    return None

# =========================
# 3. Batch process
# =========================
for size in sizes:
    print("\n" + "=" * 60)
    print(f"Processing grid size: {size} m")

    csv_path = rf"D:\Data Finland\Disturbance\TestNK\GridInformation\grid_{size}m_WBB_event_timeline_with_BA.csv"
    gpkg_path = r"D:\Data Finland\Disturbance\TestNK\WBB_valid_grids.gpkg"
    layer_name = f"grid_{size}m_valid"

    # -------------------------
    # Read CSV
    # -------------------------
    df = pd.read_csv(csv_path)

    for col in ["Grid_ID", "EndYear", "BB", "Wind"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 统一使用 EndYear
    df["EventYear"] = df["EndYear"]

    # 提取事件顺序
    df["event_order"] = df["Event_ID"].apply(extract_event_order)

    # 清洗
    df = df[~df["Grid_ID"].isna()].copy()
    df["Grid_ID"] = df["Grid_ID"].astype(int)

    df = df[~df["EventYear"].isna()].copy()
    df["EventYear"] = df["EventYear"].astype(int)

    print(f"CSV rows after cleaning: {len(df)}")

    # -------------------------
    # Extract windyear and BByear
    # -------------------------
    results = []

    for grid_id, sub in df.groupby("Grid_ID"):
        sub = sub.copy()

        # 优先按事件顺序排，再按年份排
        sub = sub.sort_values(
            by=["event_order", "EventYear"],
            na_position="last"
        ).reset_index(drop=True)

        pairs = []

        # 遇到 BB，就往前找最近一次 Wind
        for i in range(len(sub)):
            row = sub.iloc[i]

            if row.get("BB", 0) == 1:
                prev = sub.iloc[:i]
                prev_wind = prev[prev["Wind"] == 1]

                if len(prev_wind) > 0:
                    last_wind = prev_wind.iloc[-1]
                    windyear = int(last_wind["EventYear"])
                    BByear = int(row["EventYear"])
                    pairs.append((windyear, BByear))

        if pairs:
            # 保留最后一组
            windyear, BByear = pairs[-1]
        else:
            windyear, BByear = None, None

        results.append({
            "Grid_ID": int(grid_id),
            "windyear": windyear,
            "BByear": BByear
        })

    pair_df = pd.DataFrame(results)
    pair_df["windyear"] = pd.to_numeric(pair_df["windyear"], errors="coerce")
    pair_df["BByear"] = pd.to_numeric(pair_df["BByear"], errors="coerce")
    pair_df["WBB_interval"] = pair_df["BByear"] - pair_df["windyear"]

    print("Extracted pair preview:")
    print(pair_df.head())

    # -------------------------
    # Read GPKG layer
    # -------------------------
    gdf = gpd.read_file(gpkg_path, layer=layer_name)

    gdf["Grid_ID"] = pd.to_numeric(gdf["Grid_ID"], errors="coerce")
    gdf = gdf[~gdf["Grid_ID"].isna()].copy()
    gdf["Grid_ID"] = gdf["Grid_ID"].astype(int)

    # -------------------------
    # Join
    # -------------------------
    for col in ["windyear", "BByear", "WBB_interval"]:
        if col in gdf.columns:
            gdf = gdf.drop(columns=[col])

    gdf_out = gdf.merge(pair_df, on="Grid_ID", how="left")

    print("Joined preview:")
    print(gdf_out[["Grid_ID", "windyear", "BByear", "WBB_interval"]].head())

    # -------------------------
    # Overwrite original layer
    # -------------------------
    conn = sqlite3.connect(gpkg_path)
    cur = conn.cursor()

    cur.execute("DELETE FROM gpkg_contents WHERE table_name = ?", (layer_name,))
    cur.execute("DELETE FROM gpkg_geometry_columns WHERE table_name = ?", (layer_name,))
    cur.execute("DELETE FROM gpkg_extensions WHERE table_name = ?", (layer_name,))
    cur.execute(f'DROP TABLE IF EXISTS "{layer_name}"')

    conn.commit()
    conn.close()

    gdf_out.to_file(gpkg_path, layer=layer_name, driver="GPKG")

    print(f"Done: {layer_name}")

print("\nAll finished.")
