import geopandas as gpd
import pandas as pd
import networkx as nx

# =========================
# 1. Settings
# =========================
input_gpkg = r"D:\Data Finland\Disturbance\TestNK\MKI_Pohjois-Karjala\MKI_Pohjois-Karjala.gpkg"
input_layer = "WBB"

output_gpkg = input_gpkg
output_layer = "WBB_cleaned"

date_field = "standarrivaldate"
damage_field = "forestdamagequalifier"

WIND_CODES = {1504}
BB_CODES   = {1602, 1603, 1606, 1613}

# =========================
# 2. Read data
# =========================
gdf = gpd.read_file(input_gpkg, layer=input_layer)

print("Original rows:", len(gdf))
print("Columns:", list(gdf.columns))

# 日期处理
gdf[date_field] = pd.to_datetime(gdf[date_field], errors="coerce")
gdf["Event_Year"] = gdf[date_field].dt.year

# =========================
# 3. Define damage group
#    Wind / BarkBeetle / Other_or_None
# =========================
def classify_damage(x):
    if pd.isna(x):
        return "Other_or_None"
    try:
        x = int(x)
    except Exception:
        return "Other_or_None"

    if x in WIND_CODES:
        return "Wind"
    elif x in BB_CODES:
        return "BarkBeetle"
    else:
        return "Other_or_None"

gdf["Damage_Group"] = gdf[damage_field].apply(classify_damage)

# 只保留有年份的数据参与清洗
to_clean = gdf[gdf["Event_Year"].notna()].copy()

# 没有日期的记录无法判断“同一年”，只能单独保留
no_year = gdf[gdf["Event_Year"].isna()].copy()

print("Rows to clean (with valid year):", len(to_clean))
print("Rows without year (kept separately):", len(no_year))

# =========================
# 4. Cleaning function
#    same Damage_Group + same year + intersect => merge
# =========================
def merge_intersecting_records(sub_gdf, year, group_name):
    sub_gdf = sub_gdf.copy().reset_index(drop=True)

    if len(sub_gdf) == 0:
        return []

    sindex = sub_gdf.sindex

    G = nx.Graph()
    G.add_nodes_from(sub_gdf.index)

    for i, geom in enumerate(sub_gdf.geometry):
        if geom is None or geom.is_empty:
            continue

        candidate_idx = list(sindex.intersection(geom.bounds))
        candidates = sub_gdf.iloc[candidate_idx]

        for j in candidates.index:
            if i >= j:
                continue
            other_geom = sub_gdf.loc[j, "geometry"]
            if other_geom is None or other_geom.is_empty:
                continue
            if geom.intersects(other_geom):
                G.add_edge(i, j)

    results = []

    components = list(nx.connected_components(G))

    for comp_id, comp in enumerate(components, start=1):
        comp = list(comp)
        cluster = sub_gdf.loc[comp].copy()

        # 合并几何
        merged_geom = cluster.geometry.union_all()

        # 最早日期
        earliest_date = cluster[date_field].min()

        # 用最早日期那条记录做属性模板
        first_row = cluster.sort_values(date_field).iloc[0].copy()
        row_dict = first_row.drop(labels="geometry").to_dict()

        row_dict[date_field] = earliest_date
        row_dict["Event_Year"] = year
        row_dict["Damage_Group"] = group_name
        row_dict["Cluster_ID"] = f"{group_name}_{year}_{comp_id}"
        row_dict["Merged_Count"] = len(cluster)
        row_dict["geometry"] = merged_geom

        results.append(row_dict)

    return results

# =========================
# 5. Clean by Damage_Group + Event_Year
# =========================
results = []

for (group_name, year), sub in to_clean.groupby(["Damage_Group", "Event_Year"]):
    print(f"Processing {group_name} - {year} ... rows = {len(sub)}")
    merged_rows = merge_intersecting_records(sub, year, group_name)
    results.extend(merged_rows)

cleaned_events = gpd.GeoDataFrame(results, geometry="geometry", crs=gdf.crs)

print("Cleaned rows:", len(cleaned_events))

# =========================
# 6. Add records without valid year back
# =========================
if len(no_year) > 0:
    no_year = no_year.copy()
    no_year["Cluster_ID"] = pd.NA
    no_year["Merged_Count"] = 1

    # 补齐字段
    for col in cleaned_events.columns:
        if col not in no_year.columns:
            no_year[col] = pd.NA

    for col in no_year.columns:
        if col not in cleaned_events.columns:
            cleaned_events[col] = pd.NA

    no_year = no_year[cleaned_events.columns]

    final_gdf = pd.concat([cleaned_events, no_year], ignore_index=True)
    final_gdf = gpd.GeoDataFrame(final_gdf, geometry="geometry", crs=gdf.crs)
else:
    final_gdf = cleaned_events.copy()

print("Final rows:", len(final_gdf))

# =========================
# 7. Clean date/time fields before export
# =========================
for col in final_gdf.columns:
    col_lower = col.lower()
    if any(key in col_lower for key in ["date", "time"]):
        final_gdf[col] = pd.to_datetime(final_gdf[col], errors="coerce")
        try:
            final_gdf[col] = final_gdf[col].dt.tz_localize(None)
        except Exception:
            pass

print("\nField dtypes before export:")
print(final_gdf.dtypes)

# =========================
# 8. Export
# =========================
final_gdf.to_file(output_gpkg, layer=output_layer, driver="GPKG")

print("Done.")
print("Saved to:", output_gpkg)
print("Layer:", output_layer)
