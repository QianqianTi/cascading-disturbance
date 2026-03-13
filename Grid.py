import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box

# =========================
# 1. 输入
# =========================
gpkg_path = r"D:\Data Finland\Disturbance\TestNK\MKI_Pohjois-Karjala\MKI_Pohjois-Karjala.gpkg"

wind_layer = "NK_WINDDamage"
bb_layer   = "NK_BBDamage"

out_path = r"D:\Data Finland\Disturbance\TestNK\grid_multiscale_WBB_candidate.gpkg"

cell_sizes = [15, 20, 25, 30, 40, 50]

# =========================
# 2. 读取图层
# =========================
wind = gpd.read_file(gpkg_path, layer=wind_layer)
bb   = gpd.read_file(gpkg_path, layer=bb_layer)

wind = wind[wind.geometry.notnull() & ~wind.geometry.is_empty].copy()
bb   = bb[bb.geometry.notnull() & ~bb.geometry.is_empty].copy()

events = pd.concat([wind, bb], ignore_index=True)
events = gpd.GeoDataFrame(events, geometry="geometry", crs=wind.crs)

print(f"Total event features: {len(events)}")
print(f"CRS: {events.crs}")

# =========================
# 3. 单个 polygon 生成局部网格
# =========================
def generate_grid_for_polygon(geom, cell_size):
    minx, miny, maxx, maxy = geom.bounds

    # 对齐到网格
    minx = np.floor(minx / cell_size) * cell_size
    miny = np.floor(miny / cell_size) * cell_size
    maxx = np.ceil(maxx / cell_size) * cell_size
    maxy = np.ceil(maxy / cell_size) * cell_size

    xs = np.arange(minx, maxx, cell_size)
    ys = np.arange(miny, maxy, cell_size)

    cells = []
    for x in xs:
        for y in ys:
            cell = box(x, y, x + cell_size, y + cell_size)
            if cell.intersects(geom):
                cells.append(cell)

    return cells

# =========================
# 4. 逐个尺度生成候选网格
# =========================
for cell_size in cell_sizes:
    print("\n" + "=" * 50)
    print(f"Processing grid size: {cell_size} x {cell_size}")
    print("=" * 50)

    all_cells = []

    for i, geom in enumerate(events.geometry, start=1):
        if geom is None or geom.is_empty:
            continue

        if i % 200 == 0:
            print(f"  Feature {i}/{len(events)}")

        cells = generate_grid_for_polygon(geom, cell_size)
        all_cells.extend(cells)

    print(f"Raw cell count ({cell_size}m): {len(all_cells)}")

    # 转为 GeoDataFrame
    grid = gpd.GeoDataFrame(geometry=all_cells, crs=events.crs)

    # 用 WKB 去重
    grid["wkb"] = grid.geometry.apply(lambda g: g.wkb)
    grid = grid.drop_duplicates(subset="wkb").copy()
    grid = grid.drop(columns="wkb").reset_index(drop=True)

    # 添加 Grid_ID 和 Cell_Size
    grid["Grid_ID"] = np.arange(1, len(grid) + 1)
    grid["Cell_Size"] = cell_size

    print(f"Unique cell count ({cell_size}m): {len(grid)}")

    # 图层名称
    out_layer = f"grid_{cell_size}m_WBB_candidate"

    # 输出
    grid.to_file(out_path, layer=out_layer, driver="GPKG")
    print(f"Saved layer: {out_layer}")

print("\nDone! All grid layers have been saved.")
print(f"Output GPKG: {out_path}")