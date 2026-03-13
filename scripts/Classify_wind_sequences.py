import geopandas as gpd # type: ignore

# -----------------------------
# 1. 路径
# -----------------------------
gpkg_path = r"D:\Data Finland\Disturbance\Forest owners data\MKI_Pohjois-Karjala\declarations_wind_intersect.gpkg"
gpkg_layer = "declarations_wind_intersect"

wbb_path = r"D:\Data Finland\Disturbance\MKI_Disturbance_WIND_SNOW_INSECTS\WBBboth.shp"

out_path = r"D:\Data Finland\Disturbance\Forest owners data\MKI_Pohjois-Karjala\WBB_sequences.gpkg"
out_layer = "wind_intersect_WBBonly"

# -----------------------------
# 2. 读取数据
# -----------------------------
wind_intersect = gpd.read_file(gpkg_path, layer=gpkg_layer)
wbb = gpd.read_file(wbb_path)

print("wind_intersect:", len(wind_intersect))
print("WBBboth:", len(wbb))

# -----------------------------
# 3. 坐标系统统一
# -----------------------------
if wind_intersect.crs != wbb.crs:
    wbb = wbb.to_crs(wind_intersect.crs)

# -----------------------------
# 4. 空间筛选：只保留与 WBBboth 相交的要素
# -----------------------------
# 先把 WBB 合并成一个整体几何，提升筛选效率
wbb_union = wbb.union_all()

filtered = wind_intersect[wind_intersect.geometry.intersects(wbb_union)].copy()

print("Filtered features:", len(filtered))

# -----------------------------
# 5. 保存输出
# -----------------------------
filtered.to_file(out_path, layer=out_layer, driver="GPKG")

print("Done.")
print("Output saved to:", out_path)
print("Layer name:", out_layer)
