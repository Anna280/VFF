import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# --- Load and Preprocess Data ---
pass_data = pd.read_csv("/Users/annadaugaard/Desktop/VFF/VFF_analytics_src/data/03_model_data/test_for_streamit.csv")
gps_data = pd.read_csv("/Users/annadaugaard/Desktop/VFF/preprocessed_data_for_decision_making.csv", index_col=0)
gps_data_home = gps_data[gps_data["Team"] == "home"]
annotations = pd.read_csv("/Users/annadaugaard/Desktop/VFF/explore/spiller_24_renskrevet.csv", sep=";")
annotations["player_num"] = annotations["player_num"].astype(int)

pass_data_home = pass_data[pass_data["Team"] == "home"]
pass_data_home = pass_data_home[pass_data_home["uncertainty"] < 2].reset_index(drop=True)

gps_columns = ["time", "player_id", "player_num", "x", "y", "spd", "ball_x", "ball_y", "Team", "acceleration", "smoothed_acceleration"]
extracted_gps_data = pd.DataFrame(columns=gps_columns + ["pass_event_id"])

for idx, row in tqdm(pass_data_home.iterrows(), total=len(pass_data_home), desc="Extracting GPS data"):
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
decision_points = extracted_gps_data[extracted_gps_data["decision_making_point"] == 1].reset_index()

# --- Overlap Area Calculation ---
def calculate_overlap_area(target_x, target_y, x_ball, y_ball, away_players, angle_degrees=40, circle_radius=2):
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

    return sum(
        triangle.intersection(Point(x, y).buffer(circle_radius)).area
        for x, y in zip(away_players["x"], away_players["y"])
    )

# --- Custom Scoring ---
def custom_score(overlap_area, distance_to_ball, ball_direction_x, alpha=1.0, beta=0.01, gamma=0.1):
    direction_bonus = gamma if ball_direction_x > 0 else -gamma
    penalty = alpha * overlap_area + beta * (-distance_to_ball)
    return np.exp(-penalty + direction_bonus)

# --- Parameter Evaluation Function ---
def evaluate_param_combination(params):
    alpha, beta, gamma, angle = params
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

            if distance_to_ball < 15:
                circle_radius = 2
            elif distance_to_ball <= 25:
                circle_radius = 3
            else:
                circle_radius = 6

            overlap_area = calculate_overlap_area(
                target_x=home_player["x"],
                target_y=home_player["y"],
                x_ball=x_ball,
                y_ball=y_ball,
                away_players=away_players,
                angle_degrees=angle,
                circle_radius=circle_radius
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

    only_24 = results_df_wide[results_df_wide["reference_player_num"] == 24].reset_index(drop=True)
    only_24.index = only_24.index + 1
    df_filtered = only_24.drop(index=[3, 6, 17, 20, 25, 30, 34, 40, 42, 44, 45, 47])
    df_filtered.index = df_filtered.index.astype(int)

    merged_df = annotations.merge(df_filtered, left_on='Picture_id', right_index=True)
    matching_rows2 = merged_df[merged_df["max_score_player"] == merged_df["player_num"]]
    accuracy = len(matching_rows2) / len(annotations["Picture_id"].unique())

    return {
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'angle': angle,
        'accuracy': accuracy
    }

# --- Run Sampling in Parallel ---
n_samples = 100
param_samples = [
    (random.uniform(0.1, 1.0), random.uniform(0.001, 0.05), random.uniform(0.1, 1), random.uniform(20, 60))
    for _ in range(n_samples)
]

#param_samples = [(0.21378698795024154,0.04521767482654141,0.961862342433836,28.41389086074131)] 


results_summary = []
with ThreadPoolExecutor() as executor:
    for result in tqdm(executor.map(evaluate_param_combination, param_samples), total=n_samples, desc="Parameter Search"):
        results_summary.append(result)

# --- Final Result ---
results_df_summary = pd.DataFrame(results_summary)
results_df_summary.sort_values("accuracy", ascending=False, inplace=True)
best_row = results_df_summary.iloc[0]

print("\nBest Parameters:", best_row.drop("accuracy").to_dict())
print("Best Accuracy:", best_row["accuracy"])

results_df_summary.to_csv("tuning_results_sampled_static_radius_parallel.csv", index=False)
