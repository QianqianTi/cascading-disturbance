"""
Merge Forest Structure + Event History → Before/After Variables
================================================================
Year matching logic (based on Report_End_Year):

  Even year (e.g. 2018):
    Structure_before_year = Report_End_Year - 1   (e.g. 2017)
    Structure_after_year  = Report_End_Year + 1   (e.g. 2019)

  Odd year (e.g. 2017):
    Structure_before_year = Report_End_Year - 2   (e.g. 2015)
    Structure_after_year  = Report_End_Year        (e.g. 2017)

  lag_before_years = Report_End_Year - Structure_before_year
  lag_after_years  = Structure_after_year - Report_End_Year

All extraction is fully vectorised (no row-wise apply).
"""

import os
import time
import numpy as np
import pandas as pd

_T0 = time.time()
def ts(): return f"[+{time.time()-_T0:.0f}s]"

# ================================================================
# 0. Paths
# ================================================================

STRUCTURE_CSV = r"D:\Data Finland\Disturbance\TestNK20\GridOutput\forest_structure_extracted.csv"
EVENT_CSV     = r"D:\Data Finland\Disturbance\TestNK20\GridEventHistory\grid_16m_event_history.csv"
OUTPUT_CSV    = r"D:\Data Finland\Disturbance\TestNK20\GridEventHistory\grid_16m_event_with_structure.csv"

AVAILABLE_YEARS = [2009, 2011, 2013, 2015, 2017, 2019, 2021, 2023]

CONTINUOUS_VARS  = [
    "BasalArea", "Age", "MeanDiameter", "MeanHeight",
    "VolumePine", "VolumeBirch", "VolumeSpruce",
    "Prop_Pine", "Prop_Birch", "Prop_Spruce",
]
CATEGORICAL_VARS = ["ForestType"]
ALL_VARS         = CONTINUOUS_VARS + CATEGORICAL_VARS

PREFIXES = ["", "Surrounding_"]   # "" = cell value,  "Surrounding_" = window mean

# ================================================================
# 1. Read inputs
# ================================================================

print(f"{ts()} Reading structure table ...")
struct = pd.read_csv(STRUCTURE_CSV)
print(f"  Rows: {len(struct):,}   Cols: {len(struct.columns)}")

print(f"{ts()} Reading event history table ...")
events = pd.read_csv(EVENT_CSV)
print(f"  Rows: {len(events):,}   Cols: {len(events.columns)}")

# ================================================================
# 2. Merge on Cell_ID
# ================================================================

print(f"{ts()} Merging on Cell_ID ...")
df = events.merge(struct, on="Cell_ID", how="left")
print(f"  Merged rows: {len(df):,}")

# ================================================================
# 3. Compute Structure_before_year / Structure_after_year
#    Fully vectorised — no apply()
# ================================================================

print(f"{ts()} Computing structure match years ...")

end = df["Report_End_Year"]
is_even = (end % 2 == 0)

avail_set = set(AVAILABLE_YEARS)

before = np.where(is_even, end - 1, end - 2)
after  = np.where(is_even, end + 1, end    )

# Mask out years outside available raster years (use float so NaN is representable)
df["Structure_before_year"] = np.where(np.isin(before, list(avail_set)),
                                        before, np.nan).astype(float)
df["Structure_after_year"]  = np.where(np.isin(after,  list(avail_set)),
                                        after,  np.nan).astype(float)

# ================================================================
# 4. Lag columns  (always positive by construction)
# ================================================================

df["lag_before_years"] = (
    df["Report_End_Year"] - df["Structure_before_year"]
).where(df["Structure_before_year"].notna())

df["lag_after_years"] = (
    df["Structure_after_year"] - df["Report_End_Year"]
).where(df["Structure_after_year"].notna())

# Sanity check
print(f"  before_year valid : {df['Structure_before_year'].notna().sum():,}")
print(f"  after_year  valid : {df['Structure_after_year'].notna().sum():,}")
print(f"  lag_before  range : {df['lag_before_years'].min():.0f} – {df['lag_before_years'].max():.0f} yrs")
print(f"  lag_after   range : {df['lag_after_years'].min():.0f} – {df['lag_after_years'].max():.0f} yrs")

# ================================================================
# 5. Vectorised before/after extraction + change
#
# For each year Y in AVAILABLE_YEARS, build one boolean mask and
# do a single column assignment.  No row-wise looping at all.
# Total masks: 8 years × 11 vars × 2 prefixes × 2 sides = 352
# ================================================================

print(f"{ts()} Extracting before/after variable values ...")

for prefix in PREFIXES:
    for var in ALL_VARS:
        before_col = f"{prefix}{var}_before"
        after_col  = f"{prefix}{var}_after"
        change_col = f"{prefix}{var}_change"

        df[before_col] = np.nan
        df[after_col]  = np.nan

        for yr in AVAILABLE_YEARS:
            src_col = f"{prefix}{var}_{yr}"
            if src_col not in df.columns:
                continue

            mask_b = df["Structure_before_year"] == yr
            if mask_b.any():
                df.loc[mask_b, before_col] = df.loc[mask_b, src_col].values

            mask_a = df["Structure_after_year"] == yr
            if mask_a.any():
                df.loc[mask_a, after_col] = df.loc[mask_a, src_col].values

        if var in CONTINUOUS_VARS:
            valid = df[before_col].notna() & df[after_col].notna()
            df[change_col] = np.where(valid,
                                      df[after_col] - df[before_col],
                                      np.nan)

print(f"{ts()} Extraction done.")

# ================================================================
# 6. Assemble final column order
# ================================================================

event_base_cols = [
    "Cell_ID", "lat", "long",
    "Report_ID", "Report_Start_Year", "Report_End_Year",
    "Wind", "BB", "Frost", "Snow", "Drought", "Fire", "Flood",
    "Nutritional", "Fungal", "Insect", "Animal", "Harvest", "Wiping",
    "No_Damage",
    "Stand_ID", "Stand_ID_area", "Polygon_Prop",
    "Time_Interval", "Management_type", "Management_Plan",
]
event_base_cols = [c for c in event_base_cols if c in df.columns]

meta_cols = [
    "Structure_before_year", "Structure_after_year",
    "lag_before_years", "lag_after_years",
]

var_cols = []
for prefix in PREFIXES:
    for var in ALL_VARS:
        var_cols.append(f"{prefix}{var}_before")
        var_cols.append(f"{prefix}{var}_after")
        if var in CONTINUOUS_VARS:
            var_cols.append(f"{prefix}{var}_change")

final_cols = [c for c in event_base_cols + meta_cols + var_cols if c in df.columns]
result = df[final_cols].copy()

# ================================================================
# 7. Save
# ================================================================

print(f"{ts()} Saving CSV ...")
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig", float_format="%.4f")

# ================================================================
# 8. Summary
# ================================================================

elapsed = time.time() - _T0
print(f"\n{'='*55}")
print(f"  Total elapsed      : {elapsed:.1f} s")
print(f"  Output rows        : {len(result):,}")
print(f"  Output columns     : {len(result.columns)}")
print(f"  Cells covered      : {result['Cell_ID'].nunique():,}")
print(f"\n  Year matching examples (first 5 unique Report_End_Year):")
sample = (
    df[["Report_End_Year", "Structure_before_year",
        "Structure_after_year", "lag_before_years", "lag_after_years"]]
    .drop_duplicates("Report_End_Year")
    .sort_values("Report_End_Year")
    .head(5)
)
print(sample.to_string(index=False))
print(f"\n  Saved → {OUTPUT_CSV}")
print("Done!")
