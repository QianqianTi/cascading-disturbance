import geopandas as gpd
import pandas as pd

# -----------------------------
# 1. Settings
# -----------------------------
wbb_gpkg = r"D:\Data Finland\Disturbance\TestNK\MKI_Pohjois-Karjala\MKI_Pohjois-Karjala.gpkg"
grid_gpkg = r"D:\Data Finland\Disturbance\TestNK\WBB_valid_grids.gpkg"
out_dir   = r"D:\Data Finland\Disturbance\TestNK\MKI_Pohjois-Karjala"

cell_sizes = [15, 20, 25, 30, 40, 50]

WIND_CODES = {1504}
BB_CODES   = {1602, 1603, 1606, 1613}

# -----------------------------
# 2. read WBB data and prepare event attributes
# -----------------------------
wbb = gpd.read_file(wbb_gpkg, layer="WBB_Cleaned")
print("WBB features:", len(wbb))

wbb["Event_Year"] = pd.to_datetime(wbb["standarrivaldate"], errors="coerce").dt.year

# forestdamagequalifier null → nodamage → Wind=0, BarkBeetle=0
# time from standarrivaldate
wbb["Wind"]       = wbb["forestdamagequalifier"].apply(
    lambda x: 1 if not pd.isna(x) and int(x) in WIND_CODES else 0)
wbb["BarkBeetle"] = wbb["forestdamagequalifier"].apply(
    lambda x: 1 if not pd.isna(x) and int(x) in BB_CODES  else 0)

wbb_slim = wbb[["standarrivaldate", "Event_Year", "Wind", "BarkBeetle", "geometry"]].copy()

# -----------------------------
# 3. save CSV
# -----------------------------
for cell_size in cell_sizes:
    layer_name = f"grid_{cell_size}m_valid"
    print(f"\nProcessing {layer_name} ...")

    grid = gpd.read_file(grid_gpkg, layer=layer_name)

    joined = gpd.sjoin(
        grid[["Grid_ID", "geometry"]],
        wbb_slim,
        how="left",
        predicate="intersects"
    )

    result = joined[["Grid_ID", "standarrivaldate", "Event_Year", "Wind", "BarkBeetle"]].copy()
    result = result.sort_values(["Grid_ID", "Event_Year"]).reset_index(drop=True)

    out_csv = f"{out_dir}\\grid_{cell_size}m_WBB_events.csv"
    result.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"  Rows: {len(result)}  →  {out_csv}")

print("\nAll done!")