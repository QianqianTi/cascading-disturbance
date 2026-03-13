import geopandas as gpd
import pandas as pd
import numpy as np

# =====================================
# 1. 路径与图层名
# =====================================
gpkg_path = r"D:\Data Finland\Disturbance\TestNK\MKI_Pohjois-Karjala\MKI_Pohjois-Karjala.gpkg"

valid_grid_gpkg = r"D:\Data Finland\Disturbance\TestNK\grid_multiscale_WBB_valid_onlyGridID.gpkg"
valid_grid_layer = "grid_25m_WBB_WindBB_valid"   # 改成你要用的尺度

report_layer = "forestusedeclaration"            # 如果实际是 main.forestusedeclaration 就改这里

out_gpkg = r"D:\Data Finland\Disturbance\TestNK\forestusedeclaration_matched_to_valid_grid.gpkg"
out_layer = "forestusedeclaration_bestgrid"

# 报告唯一ID字段
# 如果这个字段不存在，代码会自动生成 Report_ID
REPORT_ID_FIELD = "objectid"


# =====================================
# 2. 读取数据
# =====================================
grid = gpd.read_file(valid_grid_gpkg, layer=valid_grid_layer)
report = gpd.read_file(gpkg_path, layer=report_layer)

# 清理空几何
grid = grid[grid.geometry.notnull() & ~grid.geometry.is_empty].copy()
report = report[report.geometry.notnull() & ~report.geometry.is_empty].copy()

# 统一坐标系
report = report.to_crs(grid.crs)

print(f"Valid grid count: {len(grid)}")
print(f"Report polygon count: {len(report)}")

# =====================================
# 3. 准备唯一报告ID
# =====================================
if REPORT_ID_FIELD not in report.columns:
    print(f"Field '{REPORT_ID_FIELD}' not found. Creating Report_ID from row index.")
    report = report.reset_index(drop=True).copy()
    report["Report_ID"] = np.arange(1, len(report) + 1)
    REPORT_ID_FIELD = "Report_ID"

# 只保留必要字段可减少内存
# 这里保留全部字段也可以；如果数据很大，可只挑常用字段
report_cols = list(report.columns)

# =====================================
# 4. 先做空间相交筛选（只保留相交的报告）
# =====================================
# 用 sjoin 先找哪些报告和网格相交
joined = gpd.sjoin(
    report,
    grid[["Grid_ID", "geometry"]],
    how="inner",
    predicate="intersects"
)

print(f"Intersected report-grid pairs: {len(joined)}")

# 如果没有相交结果，直接退出
if len(joined) == 0:
    raise ValueError("No intersecting report-grid pairs found.")

# =====================================
# 5. 计算真实交叠面积
# =====================================
# 为了算交叠面积，需要把对应 grid geometry 合并回来
grid_geom = grid[["Grid_ID", "geometry"]].rename(columns={"geometry": "grid_geometry"})
joined = joined.merge(grid_geom, on="Grid_ID", how="left")

# 计算交叠几何和面积
joined["intersection_geom"] = joined.apply(
    lambda row: row.geometry.intersection(row["grid_geometry"]),
    axis=1
)
joined["intersect_area"] = joined["intersection_geom"].area

# 去掉面积为0的情况（理论上 intersects 可能有边界接触）
joined = joined[joined["intersect_area"] > 0].copy()

print(f"Pairs with positive overlap area: {len(joined)}")

if len(joined) == 0:
    raise ValueError("No positive-area intersections found.")

# =====================================
# 6. 每个报告只保留交叠面积最大的那个网格
# =====================================
joined = joined.sort_values(
    by=[REPORT_ID_FIELD, "intersect_area"],
    ascending=[True, False]
).copy()

best_match = joined.drop_duplicates(subset=[REPORT_ID_FIELD], keep="first").copy()

print(f"Best-matched reports retained: {len(best_match)}")

# =====================================
# 7. 输出结果
# =====================================
# 只保留原报告属性 + Grid_ID + intersect_area + geometry
keep_cols = [c for c in report_cols if c in best_match.columns and c != "geometry"]
keep_cols += ["Grid_ID", "intersect_area", "geometry"]

best_out = best_match[keep_cols].copy()
best_out = gpd.GeoDataFrame(best_out, geometry="geometry", crs=best_match.crs)

# 保存
best_out.to_file(out_gpkg, layer=out_layer, driver="GPKG")

print("Done!")
print(f"Saved to: {out_gpkg}")
print(f"Layer: {out_layer}")