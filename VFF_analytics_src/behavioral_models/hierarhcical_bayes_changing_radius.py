import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from tqdm import tqdm # Use standard version
import logging
import warnings
import math
from scipy.special import softmax # For numpy softmax in accuracy check

# --- Configuration ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning) # Suppress some pandas/pytensor warnings
warnings.filterwarnings("ignore", category=UserWarning) # Suppress some PyMC warnings

# --- File Paths (Both are now used for training) ---
gps_data_filepath = "/Users/annadaugaard/Desktop/VFF/VFF_analytics_src/data/02_preprocessed/decision_model_data/preprocessed_data_for_decision_making.csv"
annotation_filepath_p24 = "/Users/annadaugaard/Desktop/VFF/VFF_analytics_src/data/02_preprocessed/decision_model_data/player_24.csv" # Player 24 data
annotation_filepath_p2 = "/Users/annadaugaard/Desktop/VFF/VFF_analytics_src/data/02_preprocessed/decision_model_data/player_2.csv"    # Player 2 data

# --- MCMC Settings ---
N_DRAWS = 5000
N_TUNE = 1000
N_CHAINS = 3
TARGET_ACCEPT = 0.9 # Might need adjustment for hierarchical models

# --- Load GPS Data ---
logger.info("Loading GPS data...")
try:
    gps_data = pd.read_csv(gps_data_filepath, index_col=0)
    gps_data['time'] = pd.to_numeric(gps_data['time'], errors='coerce')
    gps_data.dropna(subset=['time'], inplace=True)
    logger.info(f"GPS data loaded successfully. Total points: {len(gps_data)}")
except FileNotFoundError as e:
    logger.error(f"GPS data file not found: {e}")
    exit()
except Exception as e:
    logger.error(f"Error loading GPS data: {e}")
    exit()

# --- Function to Load and Prepare Annotations ---
def load_and_prepare_annotations(filepath, player_identifier, file_description=""):
    """Loads, prepares, and assigns a unique player ID to annotations."""
    logger.info(f"Loading and preparing {file_description} annotations for player {player_identifier} from: {filepath}")
    try:
        annotations = pd.read_csv(filepath, index_col=0)
        required_cols = ['player_num', 'timestamp', 'reference_player_num']
        if not all(col in annotations.columns for col in required_cols):
            raise ValueError(f"Annotations CSV must contain columns: {required_cols}")

        annotations["player_num"] = annotations["player_num"].astype(int)
        annotations["timestamp"] = pd.to_numeric(annotations["timestamp"], errors='coerce')
        annotations["reference_player_num"] = annotations["reference_player_num"].astype(int)
        annotations.dropna(subset=["timestamp"], inplace=True)

        annotations = annotations.rename(columns={
            "player_num": "actual_target_player_num",
            "timestamp": "decision_timestamp",
            "reference_player_num": "passer_player_num"
        })
        annotations['annotation_row_id'] = annotations.index # Keep track of original row
        annotations['player_id'] = player_identifier # Assign the player ID

        logger.info(f"{file_description} annotations prepared. Rows: {len(annotations)}")
        if annotations.empty:
            logger.warning(f"{file_description} annotations file is empty or resulted in empty data.")
        return annotations

    except FileNotFoundError:
        logger.error(f"{file_description} annotations file not found: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error loading/preparing {file_description} annotations: {e}")
        return None

# --- Load and Combine Annotations for Training ---
annotations_p24 = load_and_prepare_annotations(annotation_filepath_p24, player_identifier=0, file_description="Player 24")
annotations_p2 = load_and_prepare_annotations(annotation_filepath_p2, player_identifier=1, file_description="Player 2")

if annotations_p24 is None or annotations_p2 is None:
    logger.error("Failed to load one or both annotation files. Exiting.")
    exit()

# Combine annotations, ensuring player_id is preserved
annotations_combined = pd.concat([annotations_p24, annotations_p2], ignore_index=True)

if annotations_combined.empty:
    logger.error("Combined annotations are empty. Cannot proceed.")
    exit()

player_ids = annotations_combined['player_id'].unique()
n_players = len(player_ids)
logger.info(f"Combined annotations for {n_players} players. Total decisions: {len(annotations_combined)}")

# --- Helper Functions (Keep implementations, but update signature if needed) ---
def calculate_overlap_area(target_x, target_y, x_ball, y_ball, away_players, angle_degrees=28.41389086074131, circle_radius=2):
    # (Implementation remains the same)
    dx, dy = target_x - x_ball, target_y - y_ball
    height = np.hypot(dx, dy)
    if height < 1e-6: return 0.0
    trajectory_angle = np.arctan2(dy, dx)
    half_angle_radians = np.radians(angle_degrees / 2)
    if abs(np.cos(half_angle_radians)) < 1e-9: return 0.0
    base_width = 2 * height * np.tan(half_angle_radians)
    perp_vec_right = np.array([np.sin(trajectory_angle), -np.cos(trajectory_angle)])
    perp_vec_left = np.array([-np.sin(trajectory_angle), np.cos(trajectory_angle)])
    left_x, left_y = np.array([target_x, target_y]) + perp_vec_left * (base_width / 2)
    right_x, right_y = np.array([target_x, target_y]) + perp_vec_right * (base_width / 2)
    points = [(left_x, left_y), (right_x, right_y), (x_ball, y_ball)]
    try:
        triangle = Polygon(points)
        if not triangle.is_valid or triangle.area < 1e-6: return 0.0
    except Exception: return 0.0
    total_overlap_area = sum(
        triangle.intersection(Point(away_x, away_y).buffer(circle_radius)).area
        for away_x, away_y in zip(away_players["x"], away_players["y"]) if pd.notna(away_x) and pd.notna(away_y)
    )
    return total_overlap_area

def mirror_y_if_below_passer(passer_y, target_x, target_y):
    """Mirrors the target's Y-coordinate over the passer's Y-line if below it."""
    mirrored_y = target_y if target_y >= passer_y else passer_y + (passer_y - target_y)
    return (target_x, mirrored_y)

def angle_from_x_axis_after_mirroring(passer_x, passer_y, target_x, target_y):
    """Calculates the angle from the X-axis reference after Y-mirroring."""
    # Mirror the target's Y if needed
    target_x, mirrored_y = mirror_y_if_below_passer(passer_y, target_x, target_y)

    dx = target_x - passer_x
    dy = mirrored_y - passer_y
    angle_rad = math.atan2(-dx, dy)  # Flip dx to make +Y forward (0°)
    angle_deg = math.degrees(angle_rad)
    return angle_rad


#def angle_relative_to_forward(passer_x, passer_y, target_x, target_y): 
#   """Calculates the angle of the pass relative to the positive y-axis."""
#    delta_x = target_x - passer_x
#    delta_y = target_y - passer_y
#    angle_radians = np.arctan2(delta_x, delta_y) # Angle in radians, -pi to pi
#    return angle_radians




# PyTensor compatible utility function
# PyTensor compatible utility function (MODIFIED for generalized logistic angle boost)
def calculate_score_utility_pt(overlap_area, distance_to_ball, pass_angle, alpha, beta, k_direction, L=-3.0, K=3.0, x0=0.0, v=1.0):
    """
    Calculates the utility of a pass choice using a generalized logistic
    function for the angle-based boost.

    Args:
        overlap_area (tensor): Area of overlap with opponents.
        distance_to_ball (tensor): Distance from passer to target.
        pass_angle (tensor): Angle of the pass relative to forward.
        alpha (tensor): Player-specific sensitivity to overlap.
        beta (tensor): Player-specific sensitivity to distance.
        k_direction (tensor): Player-specific growth rate (sensitivity to angle).
        L (float): Lower asymptote of the logistic function.
        K (float): Upper asymptote of the logistic function.
        x0 (float): Center of the logistic function (default 0 for angle).
        v (float): Parameter affecting the asymmetry (default 1 for standard logistic).

    Returns:
        tensor: The utility score for the pass choice.
    """
    logistic_exponent = -k_direction * (pass_angle - x0)
    generalized_logistic = L + (K - L) / (1 + pt.exp(logistic_exponent))**(1/v)
    penalty = -alpha * overlap_area + beta * (distance_to_ball)
    utility = penalty + generalized_logistic
    return utility

# Standard Python version for accuracy calculation (MODIFIED for generalized logistic angle boost)
def calculate_score_utility_np(overlap, distance, angle, alpha_val, beta_val, k_direction_val, L=-3.0, K=3.0, x0=0.0, v=1.0):
    logistic_exponent = -k_direction_val * (angle - x0)
    generalized_logistic = L + (K - L) / (1 + np.exp(logistic_exponent))**(1/v)
    penalty = alpha_val * overlap + beta_val * (-distance)
    utility = -penalty + generalized_logistic
    return utility
# --- Function to Pre-calculate Features (Modified) ---
# --- Function to Pre-calculate Features (Modified to include angle) ---
def preprocess_annotations_for_hierarchical_model(annotations_df, gps_df):
    """Prepares data for the hierarchical model, including player_id and pass angle."""
    logger.info(f"Pre-calculating features for hierarchical model...")
    model_input_list = []
    required_gps_cols = ["time", "player_num", "x", "y", "Team", "ball_x", "ball_y"]

    if not all(col in gps_df.columns for col in required_gps_cols):
        missing = [col for col in required_gps_cols if col not in gps_df.columns]
        logger.error(f"GPS data is missing required columns: {missing}")
        return []

    for idx, ann_row in tqdm(annotations_df.iterrows(), total=len(annotations_df), desc="Processing combined annotations"):
        ann_row_id = ann_row["annotation_row_id"]
        decision_time = ann_row["decision_timestamp"]
        passer_num = ann_row["passer_player_num"]
        actual_target = ann_row["actual_target_player_num"]
        player_id = ann_row["player_id"] # Get the assigned player ID

        time_diff = np.abs(gps_df['time'] - decision_time)
        if time_diff.empty: continue
        closest_time_idx = time_diff.idxmin()
        closest_time = gps_df.loc[closest_time_idx, 'time']

        players_now = gps_df[gps_df["time"] == closest_time].set_index('player_num')

        if passer_num not in players_now.index: continue
        passer_state = players_now.loc[passer_num]
        x_passer = passer_state.get("x", np.nan)
        y_passer = passer_state.get("y", np.nan)
        x_ball = passer_state.get("ball_x", np.nan)
        y_ball = passer_state.get("ball_y", np.nan)
        if pd.isna(x_ball) or pd.isna(y_ball) or pd.isna(x_passer) or pd.isna(y_passer): continue

        # Make sure passer belongs to the 'home' team for consistency
        if passer_state.get("Team") != "home":
            continue # Skip if passer isn't marked as home team at that instant

        targets_df = players_now[(players_now["Team"] == "home") & (players_now.index != passer_num)]
        opponents_df = players_now[players_now["Team"] == "away"]
        if targets_df.empty: continue

        features_list = []
        target_nums = []
        chosen_idx = -1

        for i, (t_num, t_row) in enumerate(targets_df.iterrows()):
            t_x = t_row.get("x", np.nan)
            t_y = t_row.get("y", np.nan)
            if pd.isna(t_x) or pd.isna(t_y): continue

            dist = np.hypot(t_x - x_ball, t_y - y_ball)
            
        
            if dist < 15:
                circle_radius = 2
            elif dist <= 25:
                circle_radius = 3
            else:
                circle_radius = 6
                
            direct = x_ball - t_x # Direction relative to target (positive if ball is 'right' of target)
            overlap = calculate_overlap_area(t_x, t_y, x_ball, y_ball, opponents_df, circle_radius=circle_radius)
            angle = angle_from_x_axis_after_mirroring(x_passer, y_passer, t_x, t_y) # Calculate the angle

            features_list.append({"overlap": overlap, "distance": dist, "direction": direct, "angle": angle}) # Include angle
            target_nums.append(t_num)
            if t_num == actual_target: chosen_idx = i

        if chosen_idx != -1 and len(features_list) > 1: # Need at least 2 choices for Categorical/Softmax
            event_data = {
                "annotation_row_id": ann_row_id,
                "player_id": player_id, # Add player ID
                "n_choices": len(features_list),
                "features_overlap": np.array([f["overlap"] for f in features_list], dtype=np.float64),
                "features_distance": np.array([f["distance"] for f in features_list], dtype=np.float64),
                "features_direction": np.array([f["direction"] for f in features_list], dtype=np.float64),
                "features_angle": np.array([f["angle"] for f in features_list], dtype=np.float64), # Include angle in event data
                "chosen_target_index": chosen_idx,
                "target_player_numbers": np.array(target_nums)
            }
            model_input_list.append(event_data)

    logger.info(f"Pre-calculation complete. Usable decision events: {len(model_input_list)}")
    return model_input_list

# --- Pre-calculate Features for Combined Data ---
model_input_data = preprocess_annotations_for_hierarchical_model(annotations_combined, gps_data)

if not model_input_data:
    logger.error("No usable training data available for the hierarchical Bayesian model.")
    exit()

## --- Prepare Data for PyMC Model ---
# Extract data into numpy arrays for easier indexing in the model
all_player_ids = np.array([d["player_id"] for d in model_input_data], dtype=int)
all_n_choices = np.array([d["n_choices"] for d in model_input_data], dtype=int)
all_chosen_indices = np.array([d["chosen_target_index"] for d in model_input_data], dtype=int)
# Pad feature arrays so they can be stacked into single tensors
max_choices = all_n_choices.max()
n_events = len(model_input_data)

padded_overlaps = np.full((n_events, max_choices), np.nan, dtype=np.float64)
padded_distances = np.full((n_events, max_choices), np.nan, dtype=np.float64)
padded_directions = np.full((n_events, max_choices), np.nan, dtype=np.float64)
# --- ADD THIS LINE ---
padded_angles = np.full((n_events, max_choices), np.nan, dtype=np.float64) # Initialize padded_angles

for i, event in enumerate(model_input_data):
    n = event["n_choices"]
    padded_overlaps[i, :n] = event["features_overlap"]
    padded_distances[i, :n] = event["features_distance"]
    padded_directions[i, :n] = event["features_direction"]
    # --- ADD THIS LINE ---
    padded_angles[i, :n] = event["features_angle"] # Populate padded_angles
    
# --- Build Hierarchical PyMC Model ---
logger.info("Building hierarchical PyMC model with new group-level priors for alpha (truly corrected dims)...")
coords = {
    "player": player_ids, # Unique IDs [0, 1]
    "event": np.arange(n_events) # Index for each decision event
}
with pm.Model(coords=coords) as hierarchical_pass_choice_model:
    
    # --- Hyperpriors (Group Level) ---
    mu_alpha = pm.Beta("mu_alpha", alpha=1.0, beta=1.0)
    kappa_alpha_unbounded = pm.HalfNormal("kappa_alpha_unbounded", sigma=1.0)
    kappa_alpha = pm.Deterministic("kappa_alpha", pt.abs(kappa_alpha_unbounded) + 1)

    mu_beta = pm.Normal("mu_beta", mu=1.0, sigma=1.0)
    sigma_beta_unbounded = pm.HalfNormal("sigma_beta_unbounded", sigma=1.0)
    sigma_beta = sigma_beta_unbounded

    # --- Hyperpriors for Direction Sensitivity (Growth Rate k) ---
    mu_k_direction = pm.Normal("mu_k_direction", mu=0.0, sigma=1.0) # Prior for mean sensitivity
    sigma_k_direction = pm.HalfNormal("sigma_k_direction", sigma=1.0) # Prior for std dev of sensitivity
    sigma_k_direction = sigma_k_direction #pm.Deterministic("sigma_k_direction", pt.abs(sigma_k_direction) + 1)


    # --- Player-Specific Parameters ---
    a_alpha = mu_alpha * kappa_alpha
    b_alpha = (1 - mu_alpha) * kappa_alpha
    alpha = pm.Beta("alpha", alpha=a_alpha, beta=b_alpha, dims="player")

    a_beta = mu_beta * sigma_beta
    b_beta = (1 - mu_beta) * sigma_beta
    beta = pm.Beta("beta", alpha=a_beta, beta=b_beta, dims="player")

    k_direction = pm.Normal("k_direction", mu=mu_k_direction, sigma=sigma_k_direction, dims="player")
    #k_direction = pm.Deterministic("k_direction", mu_k_direction, dims="player")

    # --- Data Mapping ---
    event_player_id = pm.Data("event_player_id", all_player_ids, dims="event", mutable=False)
    overlap_data = pm.Data("overlap_data", padded_overlaps, dims=("event", "choice_option"), mutable=False)
    distance_data = pm.Data("distance_data", padded_distances, dims=("event", "choice_option"), mutable=False)
    angle_data = pm.Data("angle_data", padded_angles, dims=("event", "choice_option"), mutable=False) # Contains the angle
    n_choices_data = pm.Data("n_choices_data", all_n_choices, dims="event", mutable=False)
    observed_choice_data = pm.Data("observed_choice_data", all_chosen_indices, dims="event", mutable=False)

    # --- Likelihood ---
    alpha_event = alpha[event_player_id]
    beta_event = beta[event_player_id]
    k_direction_event = k_direction[event_player_id] # Use the sensitivity parameter

    utility = calculate_score_utility_pt(
        overlap_data, distance_data, angle_data,
        alpha_event[:, None], beta_event[:, None], k_direction_event[:, None]
    )

    choice_indices = pt.arange(max_choices)
    mask = choice_indices[None, :] < n_choices_data[:, None]
    masked_utility = pt.where(mask, utility, -np.inf)

    probabilities = pt.special.softmax(masked_utility, axis=1)

    likelihood = pm.Categorical("likelihood", p=probabilities, observed=observed_choice_data, dims="event")

logger.info("Hierarchical PyMC model built with generalized logistic angle-based direction sensitivity.")

# --- Run MCMC Sampler ---
logger.info(f"Starting MCMC sampling ({N_CHAINS} chains, {N_DRAWS} draws, {N_TUNE} tuning)...")
with hierarchical_pass_choice_model:
    idata = pm.sample(
        draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS,
        target_accept=0.9,
        random_seed=RANDOM_SEED, cores=1
    )
logger.info("MCMC sampling complete.")

# --- Analyze Posterior Results ---
logger.info("Analyzing posterior distributions...")
divergences = idata.sample_stats["diverging"].sum().item()
logger.info(f"Number of divergences: {divergences}")
if divergences > 0:
    logger.warning(f"Divergences encountered ({divergences}). Results may be unreliable. Consider increasing target_accept, reparameterizing, or using stronger priors.")

# Summarize group-level (hyper) parameters and player-level parameters
# NEW Line (around 367)
var_names_summary = ["mu_alpha", "kappa_alpha", "mu_beta", "sigma_beta_unbounded", "mu_k_direction", "sigma_k_direction", "alpha", "beta", "k_direction"]
summary = az.summary(idata, var_names=var_names_summary, hdi_prob=0.94)
print("\nPosterior Summary Statistics:")
print(summary)

# --- Function to Calculate Accuracy with Player-Specific Denominators ---
def calculate_accuracy_hierarchical_fixed_denominator(eval_data, idata, player_id_map, fixed_denominators, description=""):
    """Calculates accuracy using player-specific posterior mean parameters and fixed denominators.""" # Updated docstring
    logger.info(f"Calculating accuracy using posterior means for {description} data...")
    if not eval_data:
        logger.warning(f"No data provided for {description} accuracy calculation.")
        return 0.0, 0, 0

    # Extract posterior means for player-specific parameters
    try:
        mean_alpha_player = idata.posterior["alpha"].mean(dim=("chain", "draw")).values
        mean_beta_player = idata.posterior["beta"].mean(dim=("chain", "draw")).values
        # --- CHANGE HERE ---
        mean_k_direction_player = idata.posterior["k_direction"].mean(dim=("chain", "draw")).values # Changed from mean_gamma_player = idata.posterior["gamma"]...
    except Exception as e:
        logger.error(f"Could not extract posterior means for player parameters: {e}")
        # Ensure all required keys exist before attempting extraction
        required_keys = {"alpha", "beta", "k_direction"}
        missing_keys = required_keys - set(idata.posterior.keys())
        if missing_keys:
             logger.error(f"Missing required keys in posterior: {missing_keys}")
        return 0.0, 0, 0

    correct_predictions = 0
    # Calculate total based on fixed denominators if possible, otherwise count events
    # This logic seems complex - simpler to count actual events processed?
    processed_events = 0
    denominator_sum = 0 # Sum of relevant fixed denominators

    player_event_counts = {}
    for event_row_data in eval_data:
        player_id = event_row_data["player_id"]
        player_event_counts[player_id] = player_event_counts.get(player_id, 0) + 1

        # Check if player_id is valid
        if player_id >= len(mean_alpha_player):
             logger.error(f"Player ID {player_id} out of bounds for posterior means in {description}.")
             continue # Skip this event

        # Get the posterior mean parameters for this specific player
        alpha_val = mean_alpha_player[player_id]
        beta_val = mean_beta_player[player_id]
        # --- CHANGE HERE ---
        k_direction_val = mean_k_direction_player[player_id] # Changed from gamma_val = mean_gamma_player...

        # Calculate utilities using the standard numpy function
        # --- CORRECT ARGUMENTS HERE ---
        utilities = calculate_score_utility_np(
            event_row_data["features_overlap"],
            event_row_data["features_distance"],
            event_row_data["features_angle"],     # Pass angle as the third argument
            alpha_val,
            beta_val,
            k_direction_val            # Pass k_direction_val as the sixth argument
        )

        # Calculate probabilities using softmax - ensure utilities are not all -inf or NaN
        if np.all(np.isinf(utilities)) or np.all(np.isnan(utilities)):
             logger.warning(f"Skipping event due to invalid utilities: {utilities}")
             continue # Skip prediction if utilities are unusable

        probabilities = softmax(utilities)

        # Predict the choice by finding the index with the highest probability
        predicted_index = np.argmax(probabilities)
        processed_events += 1 # Count events actually processed

        if predicted_index == event_row_data["chosen_target_index"]:
            correct_predictions += 1

    # Determine the denominator based on fixed values for the players present in the data
    total_denominator = sum(fixed_denominators.get(p_id, 0) for p_id in player_event_counts.keys())

    if total_denominator == 0 and processed_events > 0:
        logger.warning(f"Total fixed denominator is 0 for {description}, but {processed_events} events were processed. Accuracy calculation might be misleading.")

    accuracy = correct_predictions / total_denominator if total_denominator > 0 else 0.0
    logger.info(f"Accuracy for {description} (Posterior Mean): {accuracy:.4f} ({correct_predictions} correct out of {total_denominator} potential based on fixed denominators, {processed_events} events processed)")
    return accuracy, correct_predictions, total_denominator

def evaluate_posterior_accuracy_per_player(eval_data, idata, n_samples=100, description=""):
    logger.info(f"Evaluating posterior predictive accuracy over all samples for {description} split by player...")
    posterior = idata.posterior.stack(sample=("chain", "draw"))
    indices = np.arange(posterior.sample.size)

    # Group data by player
    player_groups = {}
    for event in eval_data:
        player_id = event.get("player_id", 0)
        player_groups.setdefault(player_id, []).append(event)

    for player_id, events in player_groups.items():
        all_accuracies = []
        param_records = []

        best_accuracy = 0.0
        best_index = None
        best_params = {}
        best_outputs = []

        for idx, sample_idx in enumerate(indices):
            # Get parameter values at this sample index (by position)
            alpha_sample = posterior["alpha"].isel(sample=sample_idx).values
            beta_sample = posterior["beta"].isel(sample=sample_idx).values
            gamma_sample = posterior["k_direction"].isel(sample=sample_idx).values

            # Retrieve (chain, draw) tuple for this sample index
            chain, draw = tuple(posterior.sample[sample_idx].item())

            correct = 0
            temp_outputs = []

            for event in events:
                alpha_val = alpha_sample[player_id]
                beta_val = beta_sample[player_id]
                gamma_val = gamma_sample[player_id]

                scores = calculate_score_utility_np(
                    event["features_overlap"],
                    event["features_distance"],
                    event["features_direction"],
                    alpha_val, beta_val, gamma_val
                )
                chosen = event["chosen_target_index"]
                probabilities = softmax(scores)
                prediction = np.argmax(probabilities)

                target_numbers = event.get("target_player_numbers", [])
                actual_target_player_num = target_numbers[chosen] if chosen < len(target_numbers) else None
                predicted_player_num = target_numbers[prediction] if prediction < len(target_numbers) else None

                if prediction == chosen:
                    correct += 1

                temp_outputs.append({
                    "scores": scores.tolist(),
                    "chosen_target_index": chosen,
                    "predicted_index": int(prediction),
                    "target_player_numbers": target_numbers,
                    "actual_target_player_num": actual_target_player_num,
                    "predicted_player_num": predicted_player_num
                })

            denominator = 36 if player_id == 0 else 27
            acc = correct / denominator
            all_accuracies.append(acc)

            param_records.append({
                "sample_position": sample_idx,
                "chain": int(chain),
                "draw": int(draw),
                "alpha": alpha_sample[player_id],
                "beta": beta_sample[player_id],
                "gamma": gamma_sample[player_id],
                "accuracy": acc
            })

            if acc > best_accuracy:
                best_accuracy = acc
                best_index = sample_idx
                best_params = {
                    "alpha": alpha_sample,
                    "beta": beta_sample,
                    "gamma": gamma_sample
                }
                best_outputs = temp_outputs

        # Save best outputs to CSV
        df_outputs = pd.DataFrame(best_outputs)
       # df_outputs.to_csv(f"/Users/annadaugaard/Desktop/VFF/change_radius_results_csv/player_{player_id}_posterior_best_sample_change_r.csv", index=False)

        # Save full posterior parameter values and accuracies
        df_params = pd.DataFrame(param_records)
       # df_params.to_csv(f"/Users/annadaugaard/Desktop/VFF/change_radius_results_csv/player_{player_id}_posterior_parameters_change_r.csv", index=False)

        df_accs = df_params[["sample_position", "accuracy"]]
       # df_accs.to_csv(f"/Users/annadaugaard/Desktop/VFF/change_radius_results_csv/player_{player_id}_posterior_accuracies_change_r.csv", index=False)

        # Print summary
        mean_acc = np.mean(all_accuracies)
        std_acc = np.std(all_accuracies)

        print(f"\n[Player {player_id}] Posterior Predictive Accuracy (Bayesian): {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"[Player {player_id}] Best Posterior Sample Accuracy: {best_accuracy:.4f} (Sample index: {best_index})")
        print(f"[Player {player_id}] Best Parameter Combination:")
        print(f"  alpha = {best_params['alpha'][player_id]:.4f}, beta = {best_params['beta'][player_id]:.4f}, k_direction = {best_params['gamma'][player_id]:.4f}")


if 'idata' in locals() and idata is not None and idata.posterior is not None:

    player_id_mapping = {0: "Player 24", 1: "Player 2"} # Map ID back to name
    fixed_training_denominators = {0: 36, 1: 27} # Fixed denominators for training

    # --- Evaluate on the combined training data ---
    acc_train_combined, correct_train_combined, total_train_combined = calculate_accuracy_hierarchical_fixed_denominator(
        model_input_data, idata, player_id_mapping, fixed_training_denominators, description= "Combined TRAINING"
    )
    print(f"\nAccuracy on Combined TRAINING Data: {acc_train_combined:.4f} ({correct_train_combined}/{total_train_combined} potential hits)")

    # --- Evaluate accuracy per player on the training data ---
    for p_id, p_name in player_id_mapping.items():
        player_train_data = [d for d in model_input_data if d["player_id"] == p_id]
        if player_train_data:
            acc_player_train, correct_player_train, denominator_player_train = calculate_accuracy_hierarchical_fixed_denominator(
                player_train_data, idata, player_id_mapping, fixed_training_denominators, f"TRAINING - {p_name}"
            )
            print(f"Accuracy on TRAINING Data for {p_name}: {acc_player_train:.4f} ({correct_player_train}/{denominator_player_train} potential hits)")
        else:
            print(f"No training data found for {p_name}.")
# --- Posterior Predictive Accuracy Evaluation ---

    # Run posterior predictive accuracy evaluation split per player
    evaluate_posterior_accuracy_per_player(model_input_data, idata, n_samples=100, description="Combined TRAINING")

    logger.info("Script finished.")


else:
    logger.warning("MCMC results (idata) not found. Skipping accuracy calculation.")
    print("\nCould not calculate accuracy (MCMC results not available).")
# --- Plotting Posterior Distributions ---
try:
    if 'idata' in locals() and idata is not None:
        logger.info("Generating plots...")

        # Plot group-level parameters
        az.plot_trace(idata, var_names=["mu_alpha", "kappa_alpha", "mu_beta", "sigma_beta_unbounded", "mu_k_direction", "sigma_k_direction"], combined=True,divergences=None)
        plt.suptitle("Group-Level Parameter Traces and Posteriors", y=1.02)
        plt.tight_layout()
        plt.show()

        # Plot player-level parameters
        az.plot_trace(idata, var_names=["alpha", "beta", "k_direction"], combined=True,divergences=None)
        plt.suptitle("Player-Level Parameter Traces and Posteriors", y=1.02)
        plt.tight_layout()
        plt.show()

        # Forest plot is good for comparing player parameters
        az.plot_forest(idata, var_names=["alpha", "beta", "k_direction"], combined=True, hdi_prob=0.94)
        plt.suptitle("Player-Level Parameter Forest Plot (Mean and 94% HDI)", y=1.02)
        plt.tight_layout()
        plt.show()

    else:
        logger.info("Skipping plot generation (no MCMC data).")
except Exception as e:
    logger.error(f"Error during plotting: {e}")
    
