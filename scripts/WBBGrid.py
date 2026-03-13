import geopandas as gpd
import pandas as pd
import numpy as np

# =====================================
# 1. 路径与参数
# =====================================
event_gpkg = r"D:\Data Finland\Disturbance\TestNK\MKI_Pohjois-Karjala\MKI_Pohjois-Karjala.gpkg"
grid_gpkg  = r"D:\Data Finland\Disturbance\TestNK\grid_multiscale_WBB_candidate.gpkg"
out_gpkg   = r"D:\Data Finland\Disturbance\TestNK\grid_multiscale_WBB_valid_onlyGridID.gpkg"
summary_csv = r"D:\Data Finland\Disturbance\TestNK\grid_multiscale_WBB_valid_onlyGridID_summary.csv"

wind_layer  = "NK_WINDDamage"
bb_layer    = "NK_BBDamage"
other_layer = "NK_OtherDamage"

grid_layers = [
    "grid_15m_WBB_candidate",
    "grid_20m_WBB_candidate",
    "grid_25m_WBB_candidate",
    "grid_30m_WBB_candidate",
    "grid_40m_WBB_candidate",
    "grid_50m_WBB_candidate"
]

DATE_FIELD = "standarrivaldate"
STRICT_BETWEEN = True


# =====================================
# 2. 工具函数
# =====================================
def clean_events(gdf, date_field, name):
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    gdf[date_field] = pd.to_datetime(gdf[date_field], errors="coerce")
    gdf = gdf[gdf[date_field].notna()].copy()
    print(f"{name}: {len(gdf)} valid features with valid dates")
    return gdf

def summarize_dates(x):
    vals = sorted(set(pd.to_datetime(v) for v in x if pd.notna(v)))
    return vals

def find_first_valid_wind_bb_sequence(wind_dates, bb_dates, other_dates, strict_between=True):
    wind_dates = sorted(set(pd.to_datetime(d) for d in wind_dates))
    bb_dates = sorted(set(pd.to_datetime(d) for d in bb_dates))
    other_dates = sorted(set(pd.to_datetime(d) for d in other_dates))

    for wd in wind_dates:
        later_bbs = [bd for bd in bb_dates if bd > wd]
        for bd in later_bbs:
            if strict_between:
                between_other = [od for od in other_dates if wd < od < bd]
            else:
                between_other = [od for od in other_dates if wd <= od <= bd]

            if len(between_other) == 0:
                return wd, bd
    return None, None


# =====================================
# 3. 读取事件图层
# =====================================
wind = gpd.read_file(event_gpkg, layer=wind_layer)
bb = gpd.read_file(event_gpkg, layer=bb_layer)
other = gpd.read_file(event_gpkg, layer=other_layer)

wind = clean_events(wind, DATE_FIELD, "WIND")
bb = clean_events(bb, DATE_FIELD, "BB")
other = clean_events(other, DATE_FIELD, "OTHER")

target_crs = wind.crs
bb = bb.to_crs(target_crs)
other = other.to_crs(target_crs)

summary_records = []

# =====================================
# 4. 逐个网格尺度处理
# =====================================
for grid_layer in grid_layers:
    print("\n" + "=" * 60)
    print(f"Processing grid layer: {grid_layer}")
    print("=" * 60)

    grid = gpd.read_file(grid_gpkg, layer=grid_layer)
    grid = grid[grid.geometry.notnull() & ~grid.geometry.is_empty].copy()
    grid = grid.to_crs(target_crs)

    if "Grid_ID" not in grid.columns:
        grid["Grid_ID"] = np.arange(1, len(grid) + 1)

    print(f"Grid count before filtering: {len(grid)}")

    # 空间连接
    wind_join = gpd.sjoin(
        grid[["Grid_ID", "geometry"]],
        wind[[DATE_FIELD, "geometry"]],
        how="left",
        predicate="intersects"
    )

    bb_join = gpd.sjoin(
        grid[["Grid_ID", "geometry"]],
        bb[[DATE_FIELD, "geometry"]],
        how="left",
        predicate="intersects"
    )

    other_join = gpd.sjoin(
        grid[["Grid_ID", "geometry"]],
        other[[DATE_FIELD, "geometry"]],
        how="left",
        predicate="intersects"
    )

    # 按 Grid_ID 汇总日期
    wind_dates = (
        wind_join.dropna(subset=[DATE_FIELD])
        .groupby("Grid_ID")[DATE_FIELD]
        .apply(summarize_dates)
        .reset_index()
        .rename(columns={DATE_FIELD: "Wind_Dates"})
    )

    bb_dates = (
        bb_join.dropna(subset=[DATE_FIELD])
        .groupby("Grid_ID")[DATE_FIELD]
        .apply(summarize_dates)
        .reset_index()
        .rename(columns={DATE_FIELD: "BB_Dates"})
    )

    other_dates = (
        other_join.dropna(subset=[DATE_FIELD])
        .groupby("Grid_ID")[DATE_FIELD]
        .apply(summarize_dates)
        .reset_index()
        .rename(columns={DATE_FIELD: "Other_Dates"})
    )

    result = grid.merge(wind_dates, on="Grid_ID", how="left")
    result = result.merge(bb_dates, on="Grid_ID", how="left")
    result = result.merge(other_dates, on="Grid_ID", how="left")

    result["Wind_Dates"] = result["Wind_Dates"].apply(lambda x: x if isinstance(x, list) else [])
    result["BB_Dates"] = result["BB_Dates"].apply(lambda x: x if isinstance(x, list) else [])
    result["Other_Dates"] = result["Other_Dates"].apply(lambda x: x if isinstance(x, list) else [])

    # 判断有效序列
    seq_records = []
    for _, row in result.iterrows():
        wd, bd = find_first_valid_wind_bb_sequence(
            row["Wind_Dates"],
            row["BB_Dates"],
            row["Other_Dates"],
            strict_between=STRICT_BETWEEN
        )
        seq_records.append((wd, bd))

    result[["Wind_Date", "BB_Date"]] = pd.DataFrame(seq_records, index=result.index)

    # 只保留有效网格
    filtered = result[result["Wind_Date"].notna() & result["BB_Date"].notna()].copy()

    # 最终输出只保留 Grid_ID + geometry
    filtered_out = filtered[["Grid_ID", "geometry"]].copy()

    out_layer = grid_layer.replace("_candidate", "_WindBB_valid")
    filtered_out.to_file(out_gpkg, layer=out_layer, driver="GPKG")

    print(f"Valid grid count: {len(filtered_out)}")
    print(f"Saved layer: {out_layer}")

    summary_records.append({
        "Grid_Layer": grid_layer,
        "Output_Layer": out_layer,
        "Input_Grid_Count": len(grid),
        "Valid_Grid_Count": len(filtered_out)
    })

# =====================================
# 5. 输出汇总表
# =====================================
summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

print("\nDone!")
print(f"Filtered GPKG saved to: {out_gpkg}")
print(f"Summary CSV saved to: {summary_csv}")
print(summary_df)
