import numpy as np
import pandas as pd


def make_ball_features(ball_dataframe, window_size = 5,acceleration_threshold = 10):

    ball = ball_dataframe.dropna()
    
    # Calculate disfferences to compute speed
    ball['dx'] = ball['ball_x'].diff()
    ball['dy'] = ball['ball_y'].diff()
    ball['dt'] = ball['time'].diff()

    ball['speed'] = np.sqrt(ball['dx']**2 + ball['dy']**2) / ball['dt']

    ball['acceleration'] = ball["speed"].diff() / ball['time'].diff()

    ball.drop(columns=['dx', 'dy', 'dt'], inplace=True)

    # Create a new column for the moving average smoothed acceleration
    ball['smoothed_acceleration'] = ball['acceleration'].rolling(window=window_size, center=True).mean()
    ball['smoothed_acceleration_observed'] = [1 if abs(x) >= acceleration_threshold else 0 for x in ball['acceleration']]
    
    return ball


def filter_players_and_ball(ball_and_player_dataframe,distance_to_ball = 2):

    ball_and_player_dataframe = ball_and_player_dataframe.dropna()

    # Compute the Euclidean distance from the player (x, y) to the ball (ball_x, ball_y)
    ball_and_player_dataframe["distance_to_ball"] = np.sqrt((ball_and_player_dataframe["x"] - ball_and_player_dataframe["ball_x"])**2 +
                                            (ball_and_player_dataframe["y"] - ball_and_player_dataframe["ball_y"])**2)

    ball_and_player_dataframe["distance_rank"] = ball_and_player_dataframe.groupby("time")["distance_to_ball"].rank(method="min")
    threshold = 3.0

    ball_and_player_dataframe["uncertainty_index"] = ball_and_player_dataframe.groupby("time")["distance_to_ball"].transform(
        lambda x: (x <= threshold).sum()
    )
    
    filtered_on_distance= ball_and_player_dataframe[ball_and_player_dataframe["distance_to_ball"] < distance_to_ball]
    filtered_on_distance_and_acc= filtered_on_distance[filtered_on_distance["smoothed_acceleration_observed"] == 1]
    return filtered_on_distance_and_acc








def resolve_ties_by_team(df):
    unique_times = df["time"].unique()
    resolved = []
    for i, t in enumerate(unique_times):
        candidates = df[df["time"] == t]
        if len(candidates) == 1:
            resolved.append(candidates.iloc[0])
        else:
            # If we have a previous candidate, use its team.
            if resolved:
                prev_team = resolved[-1]["Team"]
            else:
                prev_team = None

            # Look at next unique time (if exists)
            if i < len(unique_times) - 1:
                next_time = unique_times[i+1]
                next_candidates = df[df["time"] == next_time]
                next_team = next_candidates.iloc[0]["Team"] if len(next_candidates) > 0 else None
            else:
                next_team = None

            chosen = None
            # 1) Try matching both prev_team & next_team.
            if prev_team and next_team:
                both = candidates[(candidates["Team"] == prev_team) & (candidates["Team"] == next_team)]
                if len(both) == 1:
                    chosen = both.iloc[0]
            # 2) If not, try matching prev_team.
            if chosen is None and prev_team:
                match_prev = candidates[candidates["Team"] == prev_team]
                if len(match_prev) == 1:
                    chosen = match_prev.iloc[0]
            # 3) If still not, try matching next_team.
            if chosen is None and next_team:
                match_next = candidates[candidates["Team"] == next_team]
                if len(match_next) == 1:
                    chosen = match_next.iloc[0]
            # 4) Fallback: choose the first candidate.
            if chosen is None:
                chosen = candidates.iloc[0]
            resolved.append(chosen)
    return pd.DataFrame(resolved).reset_index(drop=True)

def compress_consecutive_id(df, n_observations_in_row = 5):
    blocks = []
    current_block = None
    for _, row in df.iterrows():
        if current_block is None:
            # Start a new block with count 1.
            current_block = {
                "player_num": row["player_num"],
                "Team": row["Team"],
                "start_time": row["time"],
                "end_time": row["time"],
                "count": 1
            }
        else:
            if row["player_num"] == current_block["player_num"]:
                current_block["end_time"] = row["time"]
                current_block["count"] += 1
            else:
                # Only add the block if it has at least 3 observations.
                if current_block["count"] >= n_observations_in_row:
                    blocks.append(current_block)
                # Start a new block for the new id.
                current_block = {
                    "player_num": row["player_num"],
                    "Team": row["Team"],
                    "start_time": row["time"],
                    "end_time": row["time"],
                    "count": 1
                }
    if current_block and current_block["count"] >= n_observations_in_row:
        blocks.append(current_block)
    return pd.DataFrame(blocks)

def build_pass_events(blocks_df, filtered_dataframe, uncertainty_col="uncertainty_index"):
    blocks_df = blocks_df.sort_values("start_time").reset_index(drop=True)
    events = []
    for i in range(len(blocks_df) - 1):
        # Only create a pass event if both blocks are on the same team.
        if blocks_df.loc[i, "Team"] != blocks_df.loc[i+1, "Team"]:
            continue
        start_time = blocks_df.loc[i, "start_time"]
        end_time = blocks_df.loc[i+1, "start_time"]
        # Filter rows from filtered_dataframe with times between start_time and end_time.
        subset = filtered_dataframe[(filtered_dataframe["time"] >= start_time) & (filtered_dataframe["time"] <= end_time)]
        uncertainty_value = subset[uncertainty_col].mean() if not subset.empty else np.nan
        events.append({
            "Start Time [s]": start_time,
            "End Time [s]": end_time,
            "From": blocks_df.loc[i, "player_num"],
            "To": blocks_df.loc[i+1, "player_num"],
            "uncertainty": uncertainty_value,
            "Team": blocks_df.loc[i, "Team"]
        })
    return pd.DataFrame(events)


if __name__ == "__main__":
    viborg_players = pd.read_csv("/Users/annadaugaard/Desktop/VFF/VFF_analytics_src/data/02_preprocessed/viborg_players_gps_23-02-24.csv")
    ball = pd.read_csv("/Users/annadaugaard/Desktop/VFF/VFF_analytics_src/data/02_preprocessed/viborg_ball_gps_23-02-24.csv")
    ball = make_ball_features(ball)
    
    # Your dictionary defining home team ids FINDE NICE MÅDE AT GØRE DETTE PÅ
    away = {"away":["72cc31c2","9f93778d","d498635e","8dad1822"]}
    teams_dict = {"home": [16, 24, 18, 8, 12, 13, 11, 10, 23, 28, 2]}
    
    
    viborg_players['Team'] = viborg_players['player_num'].apply(lambda x: 'home' if x in teams_dict["home"] else 'away')
    viborg_players.loc[viborg_players['player_id'].str[:8].isin(away["away"]), 'Team'] = 'away'
    
    # Get the first period of the match
    viborg_players = viborg_players[viborg_players["period"] == 1]
    ball = ball[ball["period"] == 1]
    ball_and_player_dataframe = viborg_players.merge(ball, on="time", how="left")
    
    
    ball_and_player_dataframe.to_csv("/Users/annadaugaard/Desktop/VFF/VFF_analytics_src/data/02_preprocessed/decision_model_data/preprocessed_data_for_decision_making_old_old_old.csv")
   
    
    
    filtered_dataframe = filter_players_and_ball(ball_and_player_dataframe)
    
    # 1) Resolve ties in your DataFrame.
    df_resolved = resolve_ties_by_team(filtered_dataframe)

    df_blocks = compress_consecutive_id(df_resolved,n_observations_in_row = 5)

    df_passes = build_pass_events(df_blocks, filtered_dataframe, uncertainty_col="uncertainty_index")
    print(len(df_passes))
    df_filtered = df_passes[(df_passes["End Time [s]"] - df_passes["Start Time [s]"]) <= 10]
    print(len(df_filtered))   
    df_filtered.to_csv("/Users/annadaugaard/Desktop/VFF/VFF_analytics_src/data/03_model_data/test_for_streamit_old_old_old.csv")