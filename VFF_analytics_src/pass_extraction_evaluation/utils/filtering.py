import pandas as pd
import numpy as np


def filter_pass_events_by_timestamp_coverage(pass_df, time_df, step=0.04, missing_threshold=0.30):
    available_times = set(np.round(time_df["time"].values, 2))
    filtered_rows = []
    for _, row in pass_df.iterrows():
        start = row["Start Time [s]"]
        end = row["End Time [s]"]
        expected_times = np.round(np.arange(start, end + step / 2, step), 2)
        expected_count = len(expected_times)
        found_count = sum(1 for t in expected_times if t in available_times)
        missing_fraction = 1 - (found_count / expected_count) if expected_count > 0 else 0
        if missing_fraction <= missing_threshold:
            filtered_rows.append(row)
    return pd.DataFrame(filtered_rows)

def preprocess_ball_and_events(ball_path, event_data_path, filter_missing_threshold=0.0):
    """
    Preprocess the ball and event data CSV files into a cleaned DataFrame of filtered pass events.

    Parameters:
        ball_path (str): Path to the ball tracking CSV file.
        event_data_path (str): Path to the raw event data CSV file.
        filter_missing_threshold (float): Threshold for filtering passes based on timestamp coverage.

    Returns:
        ball (pd.DataFrame): Preprocessed ball tracking data with speed and acceleration.
        event_data_passes (pd.DataFrame): Preprocessed pass events.
        filtered_pass_df (pd.DataFrame): Passes filtered by timestamp coverage.
    """

    # Load ball data and drop NaNs
    ball = pd.read_csv(ball_path, index_col=0).dropna()

    # Calculate velocity and acceleration
    ball['dx'] = ball['ball_x'].diff()
    ball['dy'] = ball['ball_y'].diff()
    ball['dt'] = ball['time'].diff()

    ball['speed'] = np.sqrt(ball['dx'] ** 2 + ball['dy'] ** 2) / ball['dt']
    ball['acceleration'] = ball['speed'].diff() / ball['time'].diff()

    # Filter unrealistic speeds
    ball = ball[ball['speed'] <= 36]

    # Drop intermediate columns
    ball.drop(columns=['dx', 'dy', 'dt'], inplace=True)

    # Load event data and filter to passes
    event_data = pd.read_csv(event_data_path)
    event_data_passes = event_data[event_data["Type"] == "PASS"].copy()

    # Clean and transform player and coordinate data
    event_data_passes["From"] = event_data_passes["From"].astype(str).str.replace("Player", "", regex=True).astype(int)
    event_data_passes["To"] = event_data_passes["To"].astype(str).str.replace("Player", "", regex=True).astype(int)
    event_data_passes["Start X"] = event_data_passes["Start X"].astype(float) * 106
    event_data_passes["End X"] = event_data_passes["End X"].astype(float) * 106
    event_data_passes["Start Y"] = event_data_passes["Start Y"].astype(float) * 68
    event_data_passes["End Y"] = event_data_passes["End Y"].astype(float) * 68

    # Apply filtering by timestamp coverage
    filtered_pass_df = filter_pass_events_by_timestamp_coverage(
        event_data_passes,
        ball,
        step=0.04,
        missing_threshold=filter_missing_threshold
    )

    return ball, event_data_passes, filtered_pass_df


def make_ball_features_viborgFF(ball, window_size=5, acceleration_threshold=10):
    ball = ball.dropna()
    ball['dx'] = ball['ball_x'].diff()
    ball['dy'] = ball['ball_y'].diff()
    ball['dt'] = ball['time'].diff()
    ball['speed'] = np.sqrt(ball['dx']**2 + ball['dy']**2) / ball['dt']
    ball['acceleration'] = ball["speed"].diff() / ball['time'].diff()
    ball.drop(columns=['dx', 'dy', 'dt'], inplace=True)
    ball['smoothed_acceleration'] = ball['acceleration'].rolling(window=window_size, center=True).mean()
    ball['smoothed_acceleration_observed'] = [1 if abs(x) >= acceleration_threshold else 0 for x in ball['acceleration']]
    return ball


def compute_player_ball_uncertainty(ball_df, players_path, window_size=5, accel_threshold=5.0, proximity_threshold=3.0):
    """
    Smooths ball acceleration, merges with player data, and computes proximity-based uncertainty index.

    Parameters:
        ball_df (pd.DataFrame): Preprocessed ball tracking data with 'time', 'ball_x', 'ball_y', 'acceleration'.
        players_path (str): Path to player tracking CSV file.
        window_size (int): Window size for moving average smoothing of acceleration.
        accel_threshold (float): Threshold above which smoothed acceleration is considered significant.
        proximity_threshold (float): Distance (in meters) below which a player is considered "close" to the ball.

    Returns:
        players_and_ball (pd.DataFrame): Merged player-ball data with distance, rank, and uncertainty index.
        rank_1_index (pd.DataFrame): Players ranked #1 in proximity to ball at times of high acceleration.
    """

    # Smooth the ball acceleration
    ball_df = ball_df.copy()
    ball_df['smoothed_acceleration'] = ball_df['acceleration'].rolling(window=window_size, center=True).mean()
    ball_df['smoothed_acceleration_observed'] = ball_df['smoothed_acceleration'].apply(
        lambda x: 1 if abs(x) >= accel_threshold else 0
    )

    # Load player data and merge on time
    players = pd.read_csv(players_path, index_col=0)
    players_and_ball = players.merge(ball_df, on="time", how="left").dropna()

    # Compute distance to ball
    players_and_ball["distance_to_ball"] = np.sqrt(
        (players_and_ball["x"] - players_and_ball["ball_x"])**2 +
        (players_and_ball["y"] - players_and_ball["ball_y"])**2
    )

    # Rank players by distance to the ball (1 = closest)
    players_and_ball["distance_rank"] = players_and_ball.groupby("time")["distance_to_ball"].rank(method="min")

    # Count how many players are within the proximity threshold at each timestamp
    players_and_ball["uncertainty_index"] = players_and_ball.groupby("time")["distance_to_ball"].transform(
        lambda x: (x <= proximity_threshold).sum()
    )

    # Subset: Closest player at times of high ball acceleration
    rank_1 = players_and_ball[players_and_ball["distance_rank"] == 1]
    rank_1_index = rank_1[rank_1["smoothed_acceleration_observed"] == 1]

    return players_and_ball, rank_1_index
