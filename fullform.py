"""
Merge Forest Structure + Event History → Before/After Variables  (vectorised)
==============================================================================
Key change vs v1: eliminates all df.apply() row-by-row calls.
Before/After extraction is done via year-grouped boolean masks → ~100x faster.
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
OUTPUT_CSV    = r"D:\Data Finland\Disturbance\TestNK20\GridEventHistory\grid_16m_event.csv"

AVAILABLE_YEARS = [2009, 2011, 2013, 2015, 2017, 2019, 2021, 2023]

CONTINUOUS_VARS  = [
    "BasalArea", "Age", "MeanDiameter", "MeanHeight",
    "VolumePine", "VolumeBirch", "VolumeSpruce",
    "Prop_Pine", "Prop_Birch", "Prop_Spruce",
]
CATEGORICAL_VARS = ["ForestType"]
ALL_VARS         = CONTINUOUS_VARS + CATEGORICAL_VARS

# "" = cell value,  "Surrounding_" = surrounding window mean
PREFIXES = ["", "Surrounding_"]

# ================================================================
# 1. Year-matching helpers
# ================================================================

def match_start_year(year):
    if pd.isna(year): return np.nan
    year = int(year)
    if year % 2 == 1:
        return year if year in AVAILABLE_YEARS else np.nan
    prev = year - 1
    return prev if prev in AVAILABLE_YEARS else np.nan

def match_end_year(end_year, start_var_year):
    if pd.isna(end_year): return np.nan
    end_year = int(end_year)
    if end_year % 2 == 1:
        return end_year if end_year in AVAILABLE_YEARS else np.nan
    prev, nxt = end_year - 1, end_year + 1
    if prev in AVAILABLE_YEARS:
        if pd.notna(start_var_year) and prev == int(start_var_year):
            return nxt if nxt in AVAILABLE_YEARS else prev
        return prev
    return np.nan

def previous_available_year(year):
    if pd.isna(year): return np.nan
    candidates = [y for y in AVAILABLE_YEARS if y < int(year)]
    return max(candidates) if candidates else np.nan

# ================================================================
# 2. Read inputs
# ================================================================

print(f"{ts()} Reading structure table ...")
struct = pd.read_csv(STRUCTURE_CSV)
print(f"  Rows: {len(struct):,}   Cols: {len(struct.columns)}")

print(f"{ts()} Reading event history table ...")
events = pd.read_csv(EVENT_CSV)
print(f"  Rows: {len(events):,}   Cols: {len(events.columns)}")

# ================================================================
# 3. Merge on Cell_ID
# ================================================================

print(f"{ts()} Merging on Cell_ID ...")
df = events.merge(struct, on="Cell_ID", how="left")
print(f"  Merged rows: {len(df):,}")

# ================================================================
# 4. Compute Structure_before_year / Structure_after_year (vectorised)
# ================================================================

print(f"{ts()} Computing structure match years ...")

# --- before_year: vectorised over unique Report_Start_Year values ---
unique_start = pd.Series(df["Report_Start_Year"].unique())
start_map = {y: match_start_year(y) for y in unique_start}
df["Structure_before_year"] = df["Report_Start_Year"].map(start_map)

# --- after_year: depends on both Report_End_Year and before_year ---
# Group by (Report_End_Year, Structure_before_year) combos — far fewer than rows
combo_keys = df[["Report_End_Year", "Structure_before_year"]].drop_duplicates()
combo_keys["Structure_after_year"] = combo_keys.apply(
    lambda r: match_end_year(r["Report_End_Year"], r["Structure_before_year"]),
    axis=1
)
df = df.merge(combo_keys, on=["Report_End_Year", "Structure_before_year"], how="left")

# --- Special first-report rule: start NaN → before = previous(after) ---
first_mask = df["Report_Start_Year"].isna() & df["Structure_before_year"].isna()
unique_after_for_first = df.loc[first_mask, "Structure_after_year"].unique()
prev_map = {y: previous_available_year(y) for y in unique_after_for_first}
df.loc[first_mask, "Structure_before_year"] = \
    df.loc[first_mask, "Structure_after_year"].map(prev_map)

# ================================================================
# 5. Lag columns (vectorised)
# ================================================================

df["lag_before_years"] = np.where(
    df["Structure_before_year"].notna(),
    df["Report_End_Year"] - df["Structure_before_year"],
    np.nan
)
df["lag_after_years"] = np.where(
    df["Structure_after_year"].notna(),
    df["Structure_after_year"] - df["Report_End_Year"],
    np.nan
)

# ================================================================
# 6. Vectorised before/after extraction
# ================================================================
#
# Key idea: instead of looping over rows, loop over the small set of
# possible year values (max 8).  For each year Y, build a boolean mask
# "rows whose Structure_before_year == Y", then do a single column
# assignment from the corresponding structure column.
# Total operations: 8 years × 11 vars × 2 prefixes × 2 sides = 352 masks
# vs. 2 M rows × 88 apply calls in the slow version.
# ================================================================

print(f"{ts()} Extracting before/after variable values (vectorised) ...")

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

            # before
            mask_b = df["Structure_before_year"] == yr
            if mask_b.any():
                df.loc[mask_b, before_col] = df.loc[mask_b, src_col].values

            # after
            mask_a = df["Structure_after_year"] == yr
            if mask_a.any():
                df.loc[mask_a, after_col] = df.loc[mask_a, src_col].values

        # Change for continuous variables only
        if var in CONTINUOUS_VARS:
            valid = df[before_col].notna() & df[after_col].notna()
            df[change_col] = np.where(valid,
                                      df[after_col] - df[before_col],
                                      np.nan)

print(f"{ts()} Extraction done.")

# ================================================================
# 7. Assemble final column order
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
# 8. Save
# ================================================================

print(f"{ts()} Saving CSV ...")
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig", float_format="%.4f")

# ================================================================
# 9. Summary
# ================================================================

elapsed = time.time() - _T0
print(f"\n{'='*55}")
print(f"  Total elapsed      : {elapsed:.1f} s")
print(f"  Output rows        : {len(result):,}")
print(f"  Output columns     : {len(result.columns)}")
print(f"  Cells covered      : {result['Cell_ID'].nunique():,}")
print(f"\n  Structure year coverage:")
print(f"    before_year valid : {result['Structure_before_year'].notna().sum():,}")
print(f"    after_year  valid : {result['Structure_after_year'].notna().sum():,}")
print(f"\n  Lag stats (years):")
for col in ["lag_before_years", "lag_after_years"]:
    s = result[col].dropna()
    if len(s):
        print(f"    {col}: mean={s.mean():.1f}  min={s.min():.0f}  max={s.max():.0f}")
print(f"\n  Saved → {OUTPUT_CSV}")
print("Done!")