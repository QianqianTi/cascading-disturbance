import pandas as pd
import os
import glob

# =========================
# 1.input_dir = r"D:\Data Finland\Disturbance\Forest owners data\MKI_Pohjois-Karjala"
# =========================
input_dir = r"D:\Data Finland\Disturbance\Forest owners data\MKI_Pohjois-Karjala"

# find all event files
event_files = glob.glob(os.path.join(input_dir, "grid_*m_WBB_events.csv"))

summary_stats = []

for file in sorted(event_files):
    fname = os.path.basename(file)
    grid_size = fname.split("_")[1].replace("m", "")   # get 15, 20, 25 ...

    print(f"Processing {fname} ...")

    df = pd.read_csv(file)

    # -------------------------
    # 2. time conversion, deduplication, sorting
    # -------------------------
    df["standarrivaldate"] = pd.to_datetime(df["standarrivaldate"], errors="coerce")
    df = df.dropna(subset=["standarrivaldate"]).copy()

    # romove duplicates (same grid, same date, same event type)
    df = df.drop_duplicates(subset=["Grid_ID", "standarrivaldate", "Wind", "BarkBeetle"])

    # sort by Grid_ID and date
    df = df.sort_values(["Grid_ID", "standarrivaldate"]).copy()

    # -------------------------
    # 3. build summary
    # -------------------------
    results = []

    for grid_id, g in df.groupby("Grid_ID"):
        g = g.sort_values("standarrivaldate").copy()

        wind_dates = g.loc[g["Wind"] == 1, "standarrivaldate"]
        bb_dates = g.loc[g["BarkBeetle"] == 1, "standarrivaldate"]

        has_wind = int(not wind_dates.empty)
        has_bb = int(not bb_dates.empty)

        first_wind = wind_dates.min() if has_wind else pd.NaT

        first_bb_after_wind = pd.NaT
        interval_days = pd.NA
        interval_years = pd.NA
        seq = 0

        if has_wind and has_bb:
            bb_after = bb_dates[bb_dates > first_wind]
            if not bb_after.empty:
                first_bb_after_wind = bb_after.min()
                interval_days = (first_bb_after_wind - first_wind).days
                interval_years = interval_days / 365.25
                seq = 1

        results.append({
            "Grid_ID": grid_id,
            "Grid_Size": int(grid_size),
            "Has_Wind": has_wind,
            "Has_BB": has_bb,
            "First_Wind_Date": first_wind,
            "First_BB_Date": first_bb_after_wind,
            "WindBB_Sequence": seq,
            "Interval_Days": interval_days,
            "Interval_Years": interval_years
        })

    summary_df = pd.DataFrame(results)

    # save summary table
    summary_file = os.path.join(input_dir, f"grid_{grid_size}m_WBB_summary.csv")
    summary_df.to_csv(summary_file, index=False)

    # -------------------------
    # 4. summarize statistics for this grid size
    # -------------------------
    total_grids = summary_df["Grid_ID"].nunique()
    wind_grids = summary_df["Has_Wind"].sum()
    bb_grids = summary_df["Has_BB"].sum()
    windbb_grids = summary_df["WindBB_Sequence"].sum()

    ratio_wbb_in_wind = windbb_grids / wind_grids if wind_grids > 0 else pd.NA

    valid_intervals = summary_df.loc[summary_df["WindBB_Sequence"] == 1, "Interval_Years"].dropna()

    median_interval = valid_intervals.median() if len(valid_intervals) > 0 else pd.NA
    mean_interval = valid_intervals.mean() if len(valid_intervals) > 0 else pd.NA
    q1_interval = valid_intervals.quantile(0.25) if len(valid_intervals) > 0 else pd.NA
    q3_interval = valid_intervals.quantile(0.75) if len(valid_intervals) > 0 else pd.NA

    summary_stats.append({
        "Grid_Size": int(grid_size),
        "Total_Grids": total_grids,
        "Wind_Grids": wind_grids,
        "BB_Grids": bb_grids,
        "WindBB_Grids": windbb_grids,
        "WindBB_Ratio_in_Wind": ratio_wbb_in_wind,
        "Median_Interval_Years": median_interval,
        "Mean_Interval_Years": mean_interval,
        "Q1_Interval_Years": q1_interval,
        "Q3_Interval_Years": q3_interval
    })

# =========================
# 5. output comparison table
# =========================
comparison_df = pd.DataFrame(summary_stats).sort_values("Grid_Size")
comparison_file = os.path.join(input_dir, "grid_size_comparison_WBB.csv")
comparison_df.to_csv(comparison_file, index=False)

print("\nDone.")
print(comparison_df)
print("\nComparison table saved to:")
print(comparison_file)