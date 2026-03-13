import geopandas as gpd
import pandas as pd
import os

base_dir = r"D:\Data Finland\Disturbance\Forest owners data\MKI_Pohjois-Karjala"
grid_sizes = [15, 20, 25, 30, 40, 50]

# 所有原始网格都在这一个 gpkg 里
grid_gpkg = os.path.join(base_dir, "grid_WBB.gpkg")

# 输出总 gpkg
out_gpkg = os.path.join(base_dir, "WBB_valid_grids_all_scales.gpkg")

date_field = "standarrivaldate"

for size in grid_sizes:
    print(f"\nProcessing {size}m ...")

    csv_path = os.path.join(base_dir, f"grid_{size}m_WBB_events.csv")
    grid_layer = f"grid_{size}m_WBB"
    out_layer = f"grid_{size}m_valid"

    if not os.path.exists(csv_path):
        print(f"Event CSV not found: {csv_path}")
        continue

    if not os.path.exists(grid_gpkg):
        print(f"Grid GPKG not found: {grid_gpkg}")
        break

    # ------------------------------
    # 读取事件表
    # ------------------------------
    df = pd.read_csv(csv_path)
    df[date_field] = pd.to_datetime(df[date_field], errors="coerce")
    df = df.dropna(subset=[date_field]).copy()
    df = df.drop_duplicates(subset=["Grid_ID", date_field, "Wind", "BarkBeetle"])
    df = df.sort_values(["Grid_ID", date_field]).reset_index(drop=True)

    # ------------------------------
    # 定义有效格子：同时有 Wind 和 BB
    # 如果你后面有更严格的 valid IDs，可替换这里
    # ------------------------------
    summary = df.groupby("Grid_ID").agg(
        Has_Wind=("Wind", "max"),
        Has_BB=("BarkBeetle", "max")
    ).reset_index()

    valid_ids = summary.loc[
        (summary["Has_Wind"] == 1) & (summary["Has_BB"] == 1),
        "Grid_ID"
    ].tolist()

    print("Valid Grid_ID count:", len(valid_ids))

    # ------------------------------
    # 读取对应尺度图层
    # ------------------------------
    grid = gpd.read_file(grid_gpkg, layer=grid_layer)

    if "Grid_ID" not in grid.columns:
        print(f"'Grid_ID' not found in layer {grid_layer}")
        continue

    grid_valid = grid[grid["Grid_ID"].isin(valid_ids)].copy()

    print("Selected grid rows:", len(grid_valid))

    # ------------------------------
    # 保存到新的 gpkg
    # ------------------------------
    grid_valid.to_file(out_gpkg, layer=out_layer, driver="GPKG")
    print(f"Saved layer: {out_layer}")

print("\nAll done.")
print("Output GPKG:", out_gpkg)