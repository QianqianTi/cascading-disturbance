"""
Find all forest use declaration polygons that spatially intersect with wind damage polygons.

Input:
  - Wind damage layer:       main.NK_WINDDamage          (already filtered)
  - Declaration layer:       main.forestusedeclaration

Output:
  - GeoPackage with all declaration polygons intersecting any wind damage polygon
"""

import geopandas as gpd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
GPKG = r"D:\Data Finland\Disturbance\Forest owners data\MKI_Pohjois-Karjala\MKI_Pohjois-Karjala.gpkg"
OUT  = r"D:\Data Finland\Disturbance\Forest owners data\MKI_Pohjois-Karjala\declarations_wind_intersect.gpkg"

WIND_LAYER = "NK_WINDDamage"
DECL_LAYER = "forestusedeclaration"

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading wind damage layer ...")
wind = gpd.read_file(GPKG, layer=WIND_LAYER)
print(f"  Wind damage polygons: {len(wind)}")

print("Loading forest use declaration layer ...")
decl = gpd.read_file(GPKG, layer=DECL_LAYER)
print(f"  Declaration polygons: {len(decl)}")

# ── Make sure CRS matches ──────────────────────────────────────────────────────
if wind.crs != decl.crs:
    print(f"CRS mismatch! Reprojecting wind layer from {wind.crs} to {decl.crs}")
    wind = wind.to_crs(decl.crs)

# ── Spatial join (intersects) ──────────────────────────────────────────────────
# sjoin returns only declaration rows that touch / overlap any wind polygon.
# We use 'intersects' predicate which covers overlap, touch, and containment.
print("Running spatial join ...")
joined = gpd.sjoin(
    decl,
    wind[["geometry"]],   # only need geometry from wind layer
    how="inner",
    predicate="intersects"
)

# Drop the extra index column added by sjoin and remove duplicate declaration rows
# (a single declaration polygon might intersect multiple wind polygons)
result = joined.drop(columns=["index_right"]).drop_duplicates(
    subset=decl.index.name or joined.index.name
)

# Reset index cleanly
result = result.reset_index(drop=True)

print(f"  Declaration polygons intersecting wind damage: {len(result)}")

# ── Save output ────────────────────────────────────────────────────────────────
print(f"Saving to: {OUT}")
result.to_file(OUT, driver="GPKG", layer="declarations_wind_intersect")
print("Done!")