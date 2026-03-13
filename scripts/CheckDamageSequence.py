import pandas as pd

# =========================
# 1. Read file
# =========================
file_path = r"D:\Data Finland\Disturbance\TestNK\GridInformation\grid_20m_WBB_event_timeline_with_BA.csv"
df = pd.read_csv(file_path)

# =========================
# 2. Column names
# =========================
grid_id_col = "Grid_ID"
event_id_col = "Event_ID"
start_col = "StartYear"
end_col = "EndYear"
wind_col = "Wind"
bb_col = "BB"
interval_col = "Interval_time"

# =========================
# 3. Clean data
# =========================
df[wind_col] = pd.to_numeric(df[wind_col], errors="coerce").fillna(0).astype(int)
df[bb_col] = pd.to_numeric(df[bb_col], errors="coerce").fillna(0).astype(int)
df[end_col] = pd.to_numeric(df[end_col], errors="coerce")
if interval_col in df.columns:
    df[interval_col] = pd.to_numeric(df[interval_col], errors="coerce")

# Event_ID like 1_1, 1_2, 1_3
def get_event_order(x):
    try:
        return int(str(x).split("_")[1])
    except:
        return 9999

df["Event_Order"] = df[event_id_col].apply(get_event_order)

# =========================
# 4. Convert each row into event label
# =========================
def row_to_event(row):
    w = row[wind_col]
    b = row[bb_col]

    if w == 1 and b == 0:
        return "Wind"
    elif w == 0 and b == 1:
        return "BB"
    elif w == 1 and b == 1:
        return "Wind&BB"
    elif w == 0 and b == 0:
        return "No disturbance"
    else:
        return "Unknown"

df["Event_Label"] = df.apply(row_to_event, axis=1)

# =========================
# 5. Build full sequence for each Grid_ID
# =========================
grid_sequences = []

for grid_id, group in df.groupby(grid_id_col):
    group = group.sort_values(["Event_Order", end_col])

    events = group["Event_Label"].tolist()

    # 如果整个网格只有 No disturbance，也保留
    if len(events) == 0:
        sequence = "No record"
    else:
        sequence = " -> ".join(events)

    grid_sequences.append({
        "Grid_ID": grid_id,
        "Sequence_Full": sequence,
        "Step_Count": len(events)
    })

grid_seq_df = pd.DataFrame(grid_sequences)

# =========================
# 6. Count how many grids share the same sequence
# =========================
sequence_summary = (
    grid_seq_df.groupby("Sequence_Full")["Grid_ID"]
    .nunique()
    .reset_index(name="Grid_Count")
    .sort_values(["Grid_Count", "Sequence_Full"], ascending=[False, True])
)

total_grids = grid_seq_df["Grid_ID"].nunique()
sequence_summary["Percentage"] = (sequence_summary["Grid_Count"] / total_grids * 100).round(2)

# =========================
# 7. Print results
# =========================
print("\n===== Sequence for each Grid (with no-disturbance intervals) =====")
print(grid_seq_df.head(20))

print("\n===== Sequence Summary =====")
print(sequence_summary)

# =========================
# 8. Save results
# =========================
out_path = r"D:\Data Finland\Disturbance\TestNK\GridInformation\grid_20m_sequence_summary_with_nodisturbance.xlsx"

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    grid_seq_df.to_excel(writer, sheet_name="Grid_Sequences_Full", index=False)
    sequence_summary.to_excel(writer, sheet_name="Sequence_Summary_Full", index=False)

print(f"\nSaved to:\n{out_path}")
