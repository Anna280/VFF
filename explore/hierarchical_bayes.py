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
from scipy.special import softmax # For numpy softmax in accuracy check

# --- Configuration ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning) # Suppress some pandas/pytensor warnings
warnings.filterwarnings("ignore", category=UserWarning) # Suppress some PyMC warnings

# --- File Paths (Both are now used for training) ---
gps_data_filepath = "/Users/annadaugaard/Desktop/VFF/preprocessed_data_for_decision_making.csv"
annotation_filepath_p24 = "/Users/annadaugaard/Desktop/VFF/explore/player_24.csv" # Player 24 data
annotation_filepath_p2 = "/Users/annadaugaard/Desktop/VFF/explore/player_2.csv"    # Player 2 data

# --- MCMC Settings ---
N_DRAWS = 2000
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

# PyTensor compatible utility function
def calculate_score_utility_pt(overlap_area, distance_to_ball, ball_direction_x, alpha, beta, gamma):
    direction_bonus = pt.switch(ball_direction_x > 0.0, gamma, -gamma)
    # Note: Utility = -Penalty + Bonus. High utility is good.
    # Penalty increases with overlap (alpha*overlap)
    # Penalty increases as distance decreases (beta * (-distance)) -> High distance means less penalty
    penalty = -alpha * overlap_area + beta * (distance_to_ball)
    utility = penalty + direction_bonus
    return utility

# Standard Python version for accuracy calculation
def calculate_score_utility_np(overlap, distance, direction, alpha_val, beta_val, gamma_val):
    if isinstance(direction, np.ndarray):
        direction_bonus = np.where(direction > 0.0, gamma_val, -gamma_val)
    else:
        direction_bonus = gamma_val if direction > 0.0 else -gamma_val
    penalty = alpha_val * overlap + beta_val * (-distance)
    utility = -penalty + direction_bonus
    return utility

# --- Function to Pre-calculate Features (Modified) ---
def preprocess_annotations_for_hierarchical_model(annotations_df, gps_df):
    """Prepares data for the hierarchical model, including player_id."""
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
        x_ball = passer_state.get("ball_x", np.nan)
        y_ball = passer_state.get("ball_y", np.nan)
        if pd.isna(x_ball) or pd.isna(y_ball): continue

        # Make sure passer belongs to the 'home' team for consistency
        if passer_state.get("Team") != "home":
            # logger.warning(f"Passer {passer_num} at time {closest_time} is not 'home'. Skipping event {ann_row_id}.")
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
            direct = x_ball - t_x # Direction relative to target (positive if ball is 'right' of target)
            overlap = calculate_overlap_area(t_x, t_y, x_ball, y_ball, opponents_df)

            features_list.append({"overlap": overlap, "distance": dist, "direction": direct})
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
                "chosen_target_index": chosen_idx,
                "target_player_numbers": np.array(target_nums)
            }
            model_input_list.append(event_data)
        # else:
            # if chosen_idx == -1:
            #     logger.debug(f"Event {ann_row_id}: Actual target {actual_target} not found in potential targets.")
            # if len(features_list) <= 1:
            #     logger.debug(f"Event {ann_row_id}: Only {len(features_list)} valid target(s) found, need >1 for choice model.")


    logger.info(f"Pre-calculation complete. Usable decision events: {len(model_input_list)}")
    return model_input_list

# --- Pre-calculate Features for Combined Data ---
model_input_data = preprocess_annotations_for_hierarchical_model(annotations_combined, gps_data)

if not model_input_data:
    logger.error("No usable training data available for the hierarchical Bayesian model.")
    exit()

# --- Prepare Data for PyMC Model ---
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

for i, event in enumerate(model_input_data):
    n = event["n_choices"]
    padded_overlaps[i, :n] = event["features_overlap"]
    padded_distances[i, :n] = event["features_distance"]
    padded_directions[i, :n] = event["features_direction"]
# --- Build Hierarchical PyMC Model ---
logger.info("Building hierarchical PyMC model with new group-level priors for alpha (truly corrected dims)...")
coords = {
    "player": player_ids, # Unique IDs [0, 1]
    "event": np.arange(n_events) # Index for each decision event
}
with pm.Model(coords=coords) as hierarchical_pass_choice_model:
    
    # --- Hyperpriors (Group Level) - Widened Priors ---
    # Mean for alpha (wider Beta)
    mu_alpha = pm.Beta("mu_alpha", alpha=1.0, beta=1.0)  # Uniform on [0, 1]

    # Concentration for alpha (wider HalfNormal)
    kappa_alpha = pm.HalfNormal("kappa_alpha", sigma=5.0)  ### truncated at 1 for the halfnormal      uniform from 1 to 100

    mu_beta = pm.Beta("mu_beta", alpha=1.0, beta=1.0)
    #mu_beta = pm.Deterministic("mu_beta", -1.0 + 2.0 * raw_mu_beta)

    # Standard deviation for beta (wider HalfNormal)
    sigma_beta = pm.HalfNormal("sigma_beta", sigma=1.0) ### truncated at 1 for the halfnormal 

    # Mean for gamma (transformed Beta to [-5, 5])
    mu_gamma = pm.Beta("mu_gamma", alpha=2.0, beta=2.0) #### distribution giving flip -1 and 1 
    
    
    ### Prediction standpoint on the gamma variable -> increasing predictive accurcacy. desire principled formulazation, 
    # but withoput more constraint it consistently favors "bad" passes. otherwise generalized logistic. x is direction -> more directly forward, more positve reward.
    
    # prior of sensitivity -> gamma distribution, half-normal or causchy. 
    
    # semko sensitivity of backwards passings, modelling liberalness with regard to backwards passing.
    
    #mu_gamma = pm.Deterministic("mu_gamma", -5.0 + 10.0 * raw_mu_gamma)

    # Standard deviation for gamma (wider HalfNormal)
    sigma_gamma = pm.HalfNormal("sigma_gamma", sigma=3.0)

    # --- Player-Specific Parameters for alpha --- ### a and b dont have priors 
    a_alpha = mu_alpha * kappa_alpha
    b_alpha = (1 - mu_alpha) * kappa_alpha
    alpha = pm.Beta("alpha", alpha=a_alpha, beta=b_alpha, dims="player")
    
    ## the same as alpha for the beta parameter AND gamma

    # Player-Specific Parameters for beta and gamma (Non-centered Parameterization)
    offset_beta = pm.Normal("offset_beta", mu=0, sigma=1, dims="player") 
    beta = pm.Deterministic("beta", mu_beta + offset_beta * sigma_beta, dims="player")
    
# softmax -> choice consistensy, given model, data what is conssitent logially to do. think about choice rules..

    offset_gamma = pm.Normal("offset_gamma", mu=0, sigma=1, dims="player")
    gamma = pm.Deterministic("gamma", mu_gamma + offset_gamma * sigma_gamma, dims="player")

    # --- Data Mapping ---
    event_player_id = pm.Data("event_player_id", all_player_ids, dims="event", mutable=False)
    overlap_data = pm.Data("overlap_data", padded_overlaps, dims=("event", "choice_option"), mutable=False)
    distance_data = pm.Data("distance_data", padded_distances, dims=("event", "choice_option"), mutable=False)
    direction_data = pm.Data("direction_data", padded_directions, dims=("event", "choice_option"), mutable=False)
    n_choices_data = pm.Data("n_choices_data", all_n_choices, dims="event", mutable=False)
    observed_choice_data = pm.Data("observed_choice_data", all_chosen_indices, dims="event", mutable=False)

    # --- Likelihood ---
    alpha_event = alpha[event_player_id]
    beta_event = beta[event_player_id]
    gamma_event = gamma[event_player_id]

    utility = calculate_score_utility_pt(
        overlap_data, distance_data, direction_data,
        alpha_event[:, None], beta_event[:, None], gamma_event[:, None]
    )

    choice_indices = pt.arange(max_choices)
    mask = choice_indices[None, :] < n_choices_data[:, None]
    masked_utility = pt.where(mask, utility, -np.inf)

    probabilities = pt.special.softmax(masked_utility, axis=1)

    likelihood = pm.Categorical("likelihood", p=probabilities, observed=observed_choice_data, dims="event")

logger.info("Hierarchical PyMC model built with updated Beta priors for mu_beta and mu_gamma.")

# --- Run MCMC Sampler ---
logger.info(f"Starting MCMC sampling ({N_CHAINS} chains, {N_DRAWS} draws, {N_TUNE} tuning)...")
with hierarchical_pass_choice_model:
    idata = pm.sample(
        draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS,
        target_accept=0.9,
        random_seed=RANDOM_SEED, cores=1
    )
logger.info("MCMC sampling complete.")

# inverting the pyramid, interestewd in modelleing decision making of specific players in relation to tactics. 

# keep relevant to choice of passing and model 

# rather than top down and bottom up -> judgement (encoding of decision landscape) and decision making (model
# 
# #habits and excplicit choices -> train good habits. 

# perception in the background, a lot of information of not perceptual stuff 


# adding speed later -> does the model have any time sensitive processes. r
# r esource requriiring models with time required processes in linear time. iris van rooij
# behavioral modelling rather than cognitive modelling 

# --- Analyze Posterior Results ---
logger.info("Analyzing posterior distributions...")
divergences = idata.sample_stats["diverging"].sum().item()
logger.info(f"Number of divergences: {divergences}")
if divergences > 0:
    logger.warning(f"Divergences encountered ({divergences}). Results may be unreliable. Consider increasing target_accept, reparameterizing, or using stronger priors.")

# Summarize group-level (hyper) parameters and player-level parameters
var_names_summary = ["mu_alpha", "kappa_alpha", "mu_beta", "sigma_beta", "mu_gamma", "sigma_gamma", "alpha", "beta", "gamma"]
summary = az.summary(idata, var_names=var_names_summary, hdi_prob=0.94)
print("\nPosterior Summary Statistics:")
print(summary)

# --- Function to Calculate Accuracy with Player-Specific Denominators ---
def calculate_accuracy_hierarchical_fixed_denominator(eval_data, idata, player_id_map, fixed_denominators, description=""):
    """Calculates accuracy using player-specific posterior mean parameters and fixed denominators for training data."""
    logger.info(f"Calculating accuracy for {description} data...")
    if not eval_data:
        logger.warning(f"No data provided for {description} accuracy calculation.")
        return 0.0, 0, 0

    # Extract posterior means for player-specific parameters
    try:
        mean_alpha_player = idata.posterior["alpha"].mean(dim=("chain", "draw")).values
        mean_beta_player = idata.posterior["beta"].mean(dim=("chain", "draw")).values
        mean_gamma_player = idata.posterior["gamma"].mean(dim=("chain", "draw")).values
    except Exception as e:
        logger.error(f"Could not extract posterior means for player parameters: {e}")
        return 0.0, 0, 0

    correct_predictions = 0
    total_events_in_data = len(eval_data)
    denominator = total_events_in_data  # Default to the number of events in the data

    for event_row_data in eval_data:
        player_id = event_row_data["player_id"]

        # Get the posterior mean parameters for this specific player
        alpha_val = mean_alpha_player[player_id]
        beta_val = mean_beta_player[player_id]
        gamma_val = mean_gamma_player[player_id]

        # Calculate utilities using the standard numpy function
        utilities = calculate_score_utility_np(
            event_row_data["features_overlap"],
            event_row_data["features_distance"],
            event_row_data["features_direction"],
            alpha_val, beta_val, gamma_val
        )

        # Calculate probabilities using softmax
        probabilities = softmax(utilities)

        # Predict the choice by finding the index with the highest probability
        predicted_index = np.argmax(probabilities)

        if predicted_index == event_row_data["chosen_target_index"]:
            correct_predictions += 1

    if description.startswith("TRAINING"):
        # Determine the fixed denominator based on the player
        for p_id, p_name in player_id_map.items():
            if p_id in fixed_denominators and any(event["player_id"] == p_id for event in eval_data):
                denominator = fixed_denominators[p_id]
                break

    accuracy = correct_predictions / denominator if denominator > 0 else 0.0
    logger.info(f"Accuracy for {description}: {accuracy:.4f} ({correct_predictions}/{denominator} potential hits)")
    return accuracy, correct_predictions, denominator

# --- Calculate Accuracy using Player-Specific Posterior Mean Parameters ---
logger.info("Calculating accuracy using player-specific posterior mean parameters with fixed denominators for training...")



# --- Posterior Predictive Accuracy Evaluation ---
def evaluate_posterior_accuracy_per_player(eval_data, idata, n_samples=100, description=""):
    logger.info(f"Evaluating posterior predictive accuracy over {n_samples} samples for {description} split by player...")
    posterior = idata.posterior.stack(sample=("chain", "draw"))
    indices = np.random.choice(posterior.sample.size, size=n_samples, replace=False)

    # Group data by player
    player_groups = {}
    for event in eval_data:
        player_id = event.get("player_id", 0)
        player_groups.setdefault(player_id, []).append(event)

    for player_id, events in player_groups.items():
        all_accuracies = []
        best_accuracy = 0.0
        best_index = None
        best_params = {}

        for idx in indices:
            sample_idx = posterior.sample.values[idx]
            alpha_sample = posterior["alpha"].sel(sample=sample_idx).values
            beta_sample = posterior["beta"].sel(sample=sample_idx).values
            gamma_sample = posterior["gamma"].sel(sample=sample_idx).values

            correct = 0
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
                if np.argmax(scores) == event["chosen_target_index"]:
                    correct += 1
            denominator = 36 if player_id == 0 else 27
            acc = correct / denominator
            all_accuracies.append(acc)

            if acc > best_accuracy:
                best_accuracy = acc
                best_index = sample_idx
                best_params = {
                    "alpha": alpha_sample,
                    "beta": beta_sample,
                    "gamma": gamma_sample
                }

        mean_acc = np.mean(all_accuracies)
        std_acc = np.std(all_accuracies)

        print(f"\n[Player {player_id}] Posterior Predictive Accuracy (Bayesian): {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"[Player {player_id}] Best Posterior Sample Accuracy: {best_accuracy:.4f} (Sample index: {best_index})")
        print(f"[Player {player_id}] Best Parameter Combination:")
        print(f"  alpha = {best_params['alpha'][player_id]:.4f}, beta = {best_params['beta'][player_id]:.4f}, gamma = {best_params['gamma'][player_id]:.4f}")


if 'idata' in locals() and idata is not None and idata.posterior is not None:

    player_id_mapping = {0: "Player 24", 1: "Player 2"} # Map ID back to name
    fixed_training_denominators = {0: 36, 1: 27} # Fixed denominators for training

    # --- Evaluate on the combined training data ---
    acc_train_combined, correct_train_combined, total_train_combined = calculate_accuracy_hierarchical_fixed_denominator(
        model_input_data, idata, player_id_mapping, fixed_training_denominators, "Combined TRAINING"
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
        az.plot_trace(idata, var_names=["mu_alpha", "kappa_alpha", "mu_beta", "sigma_beta", "mu_gamma", "sigma_gamma"], combined=True)
        plt.suptitle("Group-Level Parameter Traces and Posteriors", y=1.02)
        plt.tight_layout()
        plt.show()

        # Plot player-level parameters
        az.plot_trace(idata, var_names=["alpha", "beta", "gamma"], combined=True)
        plt.suptitle("Player-Level Parameter Traces and Posteriors", y=1.02)
        plt.tight_layout()
        plt.show()

        # Forest plot is good for comparing player parameters
        az.plot_forest(idata, var_names=["alpha", "beta", "gamma"], combined=True, hdi_prob=0.94)
        plt.suptitle("Player-Level Parameter Forest Plot (Mean and 94% HDI)", y=1.02)
        plt.tight_layout()
        plt.show()

    else:
        logger.info("Skipping plot generation (no MCMC data).")
except Exception as e:
    logger.error(f"Error during plotting: {e}")
    
