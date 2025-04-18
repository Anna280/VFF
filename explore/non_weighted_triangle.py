import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import random

# --- Load and Preprocess Data ---
pass_data = pd.read_csv("/Users/annadaugaard/Desktop/VFF/VFF_analytics_src/data/03_model_data/test_for_streamit.csv")
gps_data = pd.read_csv("/Users/annadaugaard/Desktop/VFF/preprocessed_data_for_decision_making.csv", index_col=0)
gps_data_home = gps_data[gps_data["Team"] == "home"]
annotations = pd.read_csv("/Users/annadaugaard/Desktop/VFF/explore/spiller_24_renskrevet.csv",sep=";")
annotations["player_num"] = annotations["player_num"].astype(int)

# Step 1: subset pass_data to home team only
pass_data_home = pass_data[pass_data["Team"] == "home"].reset_index(drop=True)
pass_data_home = pass_data_home[pass_data_home["uncertainty"] < 2].reset_index(drop=True)

# Columns to keep from gps_data
gps_columns = ["time", "player_id", "player_num", "x", "y", "spd", "ball_x", "ball_y", "Team", "acceleration", "smoothed_acceleration"]

# Prepare an empty DataFrame for extracted data
extracted_gps_data = pd.DataFrame(columns=gps_columns + ["pass_event_id"])

# Steps 2 & 3: Extract GPS data for each pass event
for idx, row in pass_data_home.iterrows():
    from_player = row["From"]
    start_time = row["Start Time [s]"]
    end_time = row["End Time [s]"]

    gps_subset = gps_data_home[
        (gps_data_home["player_num"] == from_player) &
        (gps_data_home["time"] >= start_time) &
        (gps_data_home["time"] <= end_time)
    ][gps_columns].copy()

    gps_subset["pass_event_id"] = idx
    extracted_gps_data = pd.concat([extracted_gps_data, gps_subset], ignore_index=True)

extracted_gps_data["distance_to_ball"] = np.hypot(
    extracted_gps_data["x"] - extracted_gps_data["ball_x"],
    extracted_gps_data["y"] - extracted_gps_data["ball_y"]
)

low_distance = extracted_gps_data[extracted_gps_data["distance_to_ball"] < 2]
df = low_distance.copy()
df['player_chunk'] = (df['player_num'] != df['player_num'].shift()).cumsum()
max_acceleration_indices = df.groupby('player_chunk')['acceleration'].idxmax() - 10
binary_list = [0] * len(extracted_gps_data)

for idx in max_acceleration_indices:
    if 0 <= idx < len(extracted_gps_data):
        binary_list[idx] = 1

extracted_gps_data["decision_making_point"] = binary_list

# --- Overlap Area Calculation ---
def calculate_overlap_area(target_x, target_y, x_ball, y_ball, away_players, angle_degrees, circle_radius):
    dx, dy = target_x - x_ball, target_y - y_ball
    height = np.hypot(dx, dy)
    if height == 0:
        return 0.0

    angle = np.arctan2(dy, dx)
    half_angle = np.radians(angle_degrees / 2)
    base = 2 * height * np.tan(half_angle)

    left = (target_x + (base / 2) * np.cos(angle + np.pi/2),
            target_y + (base / 2) * np.sin(angle + np.pi/2))
    right = (target_x + (base / 2) * np.cos(angle - np.pi/2),
             target_y + (base / 2) * np.sin(angle - np.pi/2))

    triangle = Polygon([left, right, (x_ball, y_ball)])
    if not triangle.is_valid or triangle.area == 0:
        return 0.0

    total_overlap_area = sum(
        triangle.intersection(Point(x, y).buffer(circle_radius)).area
        for x, y in zip(away_players["x"], away_players["y"])
    )

    return total_overlap_area

# --- Custom Scoring ---
def custom_score(overlap_area, distance_to_ball, ball_direction_x, alpha=1.0, beta=0.01, gamma=0.1):
    direction_bonus = gamma if ball_direction_x > 0 else -gamma
    penalty = alpha * overlap_area + beta * (-distance_to_ball)
    return np.exp(-penalty + direction_bonus)

# --- Filter Decision Points ---
decision_points = extracted_gps_data[extracted_gps_data["decision_making_point"] == 1].reset_index()
decision_points = decision_points[decision_points["player_num"]==24]
# --- Sampling-Based Parameter Search ---
best_accuracy = 0
best_params = {}
results_summary = []

n_samples = 200

for _ in range(n_samples):
    alpha = random.uniform(0.1, 0.9)
    beta = random.uniform(0.001, 0.05)
    gamma = random.uniform(0.4, 1)
    angle = random.uniform(20, 60)
    fixed_radius = random.choice([2, 3, 4, 5, 6])

    results = []
    for idx, row in decision_points.iterrows():
        current_time = row["time"]
        current_player_num = row["player_num"]
        x_ball, y_ball = row["ball_x"], row["ball_y"]

        players_at_time = gps_data[(gps_data["time"] == current_time) & 
                                   (gps_data["player_num"] != current_player_num)]

        home_players = players_at_time[players_at_time["Team"] == "home"]
        away_players = players_at_time[players_at_time["Team"] == "away"]

        for _, home_player in home_players.iterrows():
            ball_direction_x = x_ball - home_player["x"]
            distance_to_ball = np.hypot(home_player["x"] - x_ball, home_player["y"] - y_ball)

            overlap_area = calculate_overlap_area(
                target_x=home_player["x"],
                target_y=home_player["y"],
                x_ball=x_ball,
                y_ball=y_ball,
                away_players=away_players,
                angle_degrees=angle,
                circle_radius=fixed_radius
            )

            score = custom_score(overlap_area, distance_to_ball, ball_direction_x, alpha, beta, gamma)

            results.append({
                "timestamp": current_time,
                "reference_player_num": current_player_num,
                "target_player_num": home_player["player_num"],
                "score": score,
                "pass_event_id": row["pass_event_id"]
            })

    results_df = pd.DataFrame(results)

    results_df_wide = results_df.pivot_table(
        index=['timestamp', "pass_event_id", 'reference_player_num'],
        columns='target_player_num',
        values='score'
    ).reset_index()

    results_df_wide.columns = [
        f'score_to_player_{int(col)}' if isinstance(col, (int, float)) else col
        for col in results_df_wide.columns
    ]

    score_columns = [col for col in results_df_wide.columns if 'score_to_player' in col]
    results_df_wide['max_score'] = results_df_wide[score_columns].max(axis=1)
    results_df_wide['max_score_player'] = results_df_wide[score_columns].idxmax(axis=1)
    results_df_wide['max_score_player'] = results_df_wide['max_score_player'].str.extract('score_to_player_(\d+)').astype(int)

    only_24 = results_df_wide[results_df_wide["reference_player_num"] == 24]
    only_24.reset_index(drop=True, inplace=True)
    only_24.index = only_24.index + 1
    df_filtered = only_24.drop(index=[3,6,17,20,25,30,34,40,42,44,45,47])
    df_filtered.index = df_filtered.index.astype(int)

    merged_df = annotations.merge(df_filtered, left_on='Picture_id', right_index=True)
    matching_rows2 = merged_df[merged_df["max_score_player"] == merged_df["player_num"]]
    accuracy = len(matching_rows2) / len(annotations["Picture_id"].unique())

    results_summary.append({
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'angle': angle,
        'radius': fixed_radius,
        'accuracy': accuracy
    })

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'angle': angle, 'radius': fixed_radius}

# --- Print Final Result ---
print("Best Parameters:", best_params)
print("Best Accuracy:", best_accuracy)

# --- Optional: Summary DataFrame ---
results_df_summary = pd.DataFrame(results_summary)
results_df_summary.sort_values("accuracy", ascending=False, inplace=True)
results_df_summary.to_csv("tuning_results_sampled.csv", index=False)
print(results_df_summary)
