from utils.filtering import make_ball_features_viborgFF
from utils.helpers import resolve_ties_by_team
from utils.build_passes import compress_consecutive_id, build_pass_events
import pandas as pd
import numpy as np
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def filter_players_and_ball(ball_and_player_dataframe, distance_to_ball=2):
    ball_and_player_dataframe = ball_and_player_dataframe.dropna()
    
    ball_and_player_dataframe["distance_to_ball"] = np.sqrt(
        (ball_and_player_dataframe["x"] - ball_and_player_dataframe["ball_x"])**2 +
        (ball_and_player_dataframe["y"] - ball_and_player_dataframe["ball_y"])**2
    )

    ball_and_player_dataframe["distance_rank"] = ball_and_player_dataframe.groupby("time")["distance_to_ball"].rank(method="min")
    threshold = 3.0
    ball_and_player_dataframe["uncertainty_index"] = ball_and_player_dataframe.groupby("time")["distance_to_ball"].transform(
        lambda x: (x <= threshold).sum()
    )

    filtered_on_distance = ball_and_player_dataframe[ball_and_player_dataframe["distance_to_ball"] < distance_to_ball]
    filtered_on_distance_and_acc = filtered_on_distance[filtered_on_distance["smoothed_acceleration_observed"] == 1]

    log(f"Filtered players near ball with high acceleration: {len(filtered_on_distance_and_acc)} rows")
    return filtered_on_distance_and_acc

# Start script
log("Loading data...")

viborg_players = pd.read_csv("/Users/annadaugaard/Desktop/VFF/VFF_analytics_src/data/02_preprocessed/viborg_players_gps_23-02-24.csv")
ball = pd.read_csv("/Users/annadaugaard/Desktop/VFF/VFF_analytics_src/data/02_preprocessed/viborg_ball_gps_23-02-24.csv")

log("Preprocessing ball data...")
ball = make_ball_features_viborgFF(ball, acceleration_threshold=10)

log("Tagging teams...")
away = {"away": ["72cc31c2", "9f93778d", "d498635e", "8dad1822"]}
teams_dict = {"home": [16, 24, 18, 8, 12, 13, 11, 10, 23, 28, 2]}

viborg_players['Team'] = viborg_players['player_num'].apply(lambda x: 'home' if x in teams_dict["home"] else 'away')
viborg_players.loc[viborg_players['player_id'].str[:8].isin(away["away"]), 'Team'] = 'away'

log("Filtering to first period...")
viborg_players = viborg_players[viborg_players["period"] == 1]
ball = ball[ball["period"] == 1]

log("Merging player and ball data...")
merged = viborg_players.merge(ball, on="time", how="left").dropna()
output_path= "/Users/annadaugaard/Desktop/VFF/VFF_analytics_src/data/03_model_data/preprocessed_data_for_decision_making.csv"
merged.to_csv(output_path)
log(f"Merged dataset shape: {merged.shape}")
log(f"Saved final pass events to {output_path}")

log("Filtering players near the ball during high acceleration...")
filtered_dataframe = filter_players_and_ball(merged)

log("Resolving tie cases...")
df_resolved = resolve_ties_by_team(filtered_dataframe)
log(f"Resolved rows: {len(df_resolved)}")

log("Compressing consecutive player IDs...")
df_blocks = compress_consecutive_id(df_resolved, id_col="player_num", n_observations_in_row=5)
log(f"Identified blocks: {len(df_blocks)}")

log("Building pass events...")
df_passes = build_pass_events(df_blocks, filtered_dataframe, uncertainty_col="uncertainty_index")
log(f"Pass events before duration filter: {len(df_passes)}")

df_filtered = df_passes[(df_passes["End Time [s]"] - df_passes["Start Time [s]"]) <= 10]
log(f"Pass events after duration filter: {len(df_filtered)}")

output_path = "/Users/annadaugaard/Desktop/VFF/VFF_analytics_src/data/03_model_data/test_for_streamit_test_test.csv"
df_filtered.to_csv(output_path, index=False)
log(f"Saved final pass events to {output_path}")

