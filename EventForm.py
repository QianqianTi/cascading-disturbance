import os
import pandas as pd
import numpy as np

# =========================================================
# 1. Paths
# =========================================================
event_folder = r"D:\Data Finland\Disturbance\TestNK\GridInformation"
ba_folder = r"D:\Data Finland\Disturbance\TestNK\BasalArea"
species_folder = r"D:\Data Finland\Disturbance\TestNK\Species"

# Grid sizes to process
sizes = [15, 20, 25, 30, 40, 50]

# Available BA years
available_years = [2009, 2011, 2013, 2015, 2017, 2019, 2021, 2023]

# =========================================================
# 2. Year matching functions
# =========================================================
def match_start_year(year):
    """
    Match StartYear to the nearest available BA year.

    Rule:
    - If the year is odd, use the same year.
    - If the year is even, use the previous year.
    """
    if pd.isna(year):
        return np.nan

    year = int(year)

    if year % 2 == 1:
        return year if year in available_years else np.nan
    else:
        prev_year = year - 1
        return prev_year if prev_year in available_years else np.nan


def match_end_year(end_year, start_ba_year):
    """
    Match EndYear to the nearest available BA year.

    Rule:
    - If the year is odd, use the same year.
    - If the year is even, prefer the previous year.
    - If the previous year is the same as Start_BA_Year,
      then use the next available year instead if possible.
    """
    if pd.isna(end_year):
        return np.nan

    end_year = int(end_year)

    # Odd year: use itself
    if end_year % 2 == 1:
        return end_year if end_year in available_years else np.nan

    # Even year: prefer previous year
    prev_year = end_year - 1
    next_year = end_year + 1

    if prev_year in available_years:
        if pd.notna(start_ba_year) and prev_year == int(start_ba_year):
            if next_year in available_years:
                return next_year
            else:
                return prev_year
        else:
            return prev_year

    return np.nan


def previous_available_ba_year(year):
    """
    Return the previous available BA year before the given BA year.

    Examples:
    2013 -> 2011
    2011 -> 2009
    2009 -> NaN
    """
    if pd.isna(year):
        return np.nan

    year = int(year)
    prev_candidates = [y for y in available_years if y < year]

    if len(prev_candidates) == 0:
        return np.nan

    return max(prev_candidates)


def extract_ba_value(row, year_col):
    """
    Extract BA value from the corresponding BA_YEAR column.

    Example:
    if Start_BA_Year = 2013, then extract BA_2013
    """
    year = row[year_col]
    if pd.isna(year):
        return np.nan

    col_name = f"BA_{int(year)}"
    if col_name in row.index:
        return row[col_name]

    return np.nan


def extract_species_value(row, year_col, prefix):
    """
    Extract species-related value from the corresponding year column.

    Example:
    if End_BA_Year = 2013:
      prefix='Species'      -> Species_2013
      prefix='Speciescount' -> Speciescount_2013
    """
    year = row[year_col]
    if pd.isna(year):
        return np.nan

    col_name = f"{prefix}_{int(year)}"
    if col_name in row.index:
        return row[col_name]

    return np.nan


def read_table_auto(file_base):
    """
    Automatically read .csv or .xlsx file.
    file_base should be path without extension.
    """
    csv_path = file_base + ".csv"
    xlsx_path = file_base + ".xlsx"

    if os.path.exists(csv_path):
        return pd.read_csv(csv_path), csv_path
    elif os.path.exists(xlsx_path):
        return pd.read_excel(xlsx_path), xlsx_path
    else:
        return None, None


# =========================================================
# 3. Process each grid size
# =========================================================
for size in sizes:
    print("=" * 70)
    print(f"Processing {size} m ...")

    input_csv = os.path.join(event_folder, f"grid_{size}m_WBB_events.csv")
    ba_csv = os.path.join(ba_folder, f"BA_{size}m.csv")
    species_base = os.path.join(species_folder, f"Species_{size}m")

    output_csv = os.path.join(
        event_folder,
        f"grid_{size}m_WBB_event_timeline_with_BA_species.csv"
    )

    # -----------------------------------------------------
    # Check event / BA files
    # -----------------------------------------------------
    if not os.path.exists(input_csv):
        print(f"  Event file not found, skipped: {input_csv}")
        continue

    if not os.path.exists(ba_csv):
        print(f"  BA file not found, skipped: {ba_csv}")
        continue

    # -----------------------------------------------------
    # Read input data
    # -----------------------------------------------------
    df = pd.read_csv(input_csv)
    ba_df = pd.read_csv(ba_csv)
    species_df, species_file = read_table_auto(species_base)

    print(f"  Event rows: {len(df)}")
    print(f"  BA rows: {len(ba_df)}")

    if species_df is None:
        print(f"  Species file not found, skipped: {species_base}.csv / .xlsx")
        continue
    else:
        print(f"  Species rows: {len(species_df)}")
        print(f"  Species file used: {species_file}")

    if "Grid_ID" not in df.columns:
        print("  'Grid_ID' not found in event file, skipped.")
        continue

    if "Grid_ID" not in ba_df.columns:
        print("  'Grid_ID' not found in BA file, skipped.")
        continue

    if "Grid_ID" not in species_df.columns:
        print("  'Grid_ID' not found in species file, skipped.")
        continue

    # -----------------------------------------------------
    # Handle event year
    # -----------------------------------------------------
    if "Event_Year" in df.columns:
        df["Event_Year"] = pd.to_numeric(df["Event_Year"], errors="coerce")
    else:
        if "standarrivaldate" not in df.columns:
            print("  Neither 'Event_Year' nor 'standarrivaldate' found, skipped.")
            continue

        df["standarrivaldate"] = pd.to_datetime(df["standarrivaldate"], errors="coerce")
        df["Event_Year"] = df["standarrivaldate"].dt.year

    df = df[df["Event_Year"].notna()].copy()
    df["Event_Year"] = df["Event_Year"].astype(int)

    if "standarrivaldate" in df.columns:
        df["standarrivaldate"] = pd.to_datetime(df["standarrivaldate"], errors="coerce")

    # -----------------------------------------------------
    # Standardize BarkBeetle / Wind fields
    # -----------------------------------------------------
    if "BarkBeetle" in df.columns:
        df["BB"] = pd.to_numeric(df["BarkBeetle"], errors="coerce").fillna(0).astype(int)
    elif "BB" not in df.columns:
        df["BB"] = 0

    if "Wind" in df.columns:
        df["Wind"] = pd.to_numeric(df["Wind"], errors="coerce").fillna(0).astype(int)
    else:
        df["Wind"] = 0

    # -----------------------------------------------------
    # Sort events within each Grid_ID
    # -----------------------------------------------------
    sort_cols = ["Grid_ID", "Event_Year"]
    if "standarrivaldate" in df.columns:
        sort_cols.append("standarrivaldate")

    df = df.sort_values(sort_cols).reset_index(drop=True)

    # -----------------------------------------------------
    # Create event sequence ID within each grid
    # -----------------------------------------------------
    df["event_order"] = df.groupby("Grid_ID").cumcount() + 1
    df["Event_ID"] = df["Grid_ID"].astype(str) + "_" + df["event_order"].astype(str)

    # -----------------------------------------------------
    # Build backward-looking event interval table
    # -----------------------------------------------------
    df["EndYear"] = df["Event_Year"]
    df["StartYear"] = df.groupby("Grid_ID")["Event_Year"].shift(1)
    df["Interval_time"] = df["EndYear"] - df["StartYear"]
    df.loc[df["StartYear"].isna(), "Interval_time"] = np.nan

    # -----------------------------------------------------
    # Match BA years
    # -----------------------------------------------------
    df["Start_BA_Year"] = df["StartYear"].apply(match_start_year)

    df["End_BA_Year"] = df.apply(
        lambda row: match_end_year(row["EndYear"], row["Start_BA_Year"]),
        axis=1
    )

    # -----------------------------------------------------
    # Special rule:
    # For any event with missing StartYear and available End_BA_Year,
    # assign Start_BA_Year as the previous available BA year
    # -----------------------------------------------------
    special_mask = (
        (df["StartYear"].isna()) &
        (df["Start_BA_Year"].isna()) &
        (df["End_BA_Year"].notna())
    )

    df.loc[special_mask, "Start_BA_Year"] = df.loc[special_mask, "End_BA_Year"].apply(
        previous_available_ba_year
    )

    # -----------------------------------------------------
    # Merge BA table by Grid_ID
    # -----------------------------------------------------
    merged = df.merge(ba_df, on="Grid_ID", how="left")

    # -----------------------------------------------------
    # Merge Species table by Grid_ID
    # -----------------------------------------------------
    merged = merged.merge(species_df, on="Grid_ID", how="left")

    # -----------------------------------------------------
    # Extract BA_start and BA_end
    # -----------------------------------------------------
    merged["BA_start"] = merged.apply(lambda row: extract_ba_value(row, "Start_BA_Year"), axis=1)
    merged["BA_end"] = merged.apply(lambda row: extract_ba_value(row, "End_BA_Year"), axis=1)

    # -----------------------------------------------------
    # Extract end-point species information
    # -----------------------------------------------------
    merged["Speciescount_end"] = merged.apply(
        lambda row: extract_species_value(row, "End_BA_Year", "Speciescount"),
        axis=1
    )

    merged["Species_end"] = merged.apply(
        lambda row: extract_species_value(row, "End_BA_Year", "Species"),
        axis=1
    )

    # -----------------------------------------------------
    # Fill missing BA values with 0
    # -----------------------------------------------------
    merged["BA_start"] = merged["BA_start"].fillna(0)
    merged["BA_end"] = merged["BA_end"].fillna(0)

    # Speciescount 缺失可填 0
    merged["Speciescount_end"] = pd.to_numeric(
        merged["Speciescount_end"], errors="coerce"
    ).fillna(0)

    # Species_end 保持为空或字符串
    # 如果你想空值填 0，可以改成：
    # merged["Species_end"] = merged["Species_end"].fillna(0)

    # -----------------------------------------------------
    # Calculate BA change
    # -----------------------------------------------------
    merged["Delta_BA"] = merged["BA_end"] - merged["BA_start"]

    # -----------------------------------------------------
    # Select final output columns
    # -----------------------------------------------------
    output_df = merged[
        [
            "Grid_ID",
            "Event_ID",
            "StartYear",
            "EndYear",
            "Start_BA_Year",
            "End_BA_Year",
            "BA_start",
            "BA_end",
            "Delta_BA",
            "Speciescount_end",
            "Species_end",
            "BB",
            "Wind",
            "Interval_time",
        ]
    ].copy()

    # -----------------------------------------------------
    # Convert selected columns to nullable integer type
    # -----------------------------------------------------
    for col in [
        "StartYear",
        "EndYear",
        "Start_BA_Year",
        "End_BA_Year",
        "BB",
        "Wind",
        "Interval_time",
        "Speciescount_end",
    ]:
        output_df[col] = pd.to_numeric(output_df[col], errors="coerce").astype("Int64")

    # Species_end 也转成数值型（因为你的编码像 1/2/3/4）
    output_df["Species_end"] = pd.to_numeric(output_df["Species_end"], errors="coerce").astype("Int64")

    # -----------------------------------------------------
    # Save result
    # -----------------------------------------------------
    output_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"  Saved to: {output_csv}")
    print(output_df.head(10))

print("=" * 70)
print("All sizes finished.")