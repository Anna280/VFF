import pandas as pd
import numpy as np

def compress_consecutive_id(df, id_col="id", n_observations_in_row=5):
    blocks = []
    current_block = None
    for _, row in df.iterrows():
        if current_block is None:
            current_block = {
                "id": row[id_col],
                "Team": row["Team"],
                "start_time": row["time"],
                "end_time": row["time"],
                "count": 1
            }
        elif row[id_col] == current_block["id"]:
            current_block["end_time"] = row["time"]
            current_block["count"] += 1
        else:
            if current_block["count"] >= n_observations_in_row:
                blocks.append(current_block)
            current_block = {
                "id": row[id_col],
                "Team": row["Team"],
                "start_time": row["time"],
                "end_time": row["time"],
                "count": 1
            }
    if current_block and current_block["count"] >= n_observations_in_row:
        blocks.append(current_block)
    return pd.DataFrame(blocks)

def build_pass_events(blocks_df, rank_df, uncertainty_col="uncertainty_index"):
    blocks_df = blocks_df.sort_values("start_time").reset_index(drop=True)
    events = []
    for i in range(len(blocks_df) - 1):
        if blocks_df.loc[i, "Team"] != blocks_df.loc[i+1, "Team"]:
            continue
        start_time = blocks_df.loc[i, "start_time"]
        end_time = blocks_df.loc[i+1, "start_time"]
        subset = rank_df[(rank_df["time"] >= start_time) & (rank_df["time"] <= end_time)]
        uncertainty_value = subset[uncertainty_col].mean() if not subset.empty else np.nan
        events.append({
            "Start Time [s]": start_time,
            "End Time [s]": end_time,
            "From": blocks_df.loc[i, "id"],
            "To": blocks_df.loc[i+1, "id"],
            "uncertainty": uncertainty_value,
            "Team": blocks_df.loc[i, "Team"]
        })
    return pd.DataFrame(events)
