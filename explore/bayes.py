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

# --- Configuration ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning) # Suppress some pandas/pytensor warnings

# --- File Paths (Update these) ---
gps_data_filepath = "/Users/annadaugaard/Desktop/VFF/preprocessed_data_for_decision_making.csv"
# *** Annotation file used for TRAINING the model ***
annotation_filepath_train = "/Users/annadaugaard/Desktop/VFF/explore/player_24.csv" # <<< CHANGE PATH (e.g., player 24 data)
# *** Annotation file used for TESTING the model ***
annotation_filepath_test = "/Users/annadaugaard/Desktop/VFF/explore/player_2.csv"      # <<< CHANGE PATH (the new player 2 data)

# --- MCMC Settings ---
N_DRAWS = 1000
N_TUNE = 1000
N_CHAINS = 2
TARGET_ACCEPT = 0.85

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
def load_and_prepare_annotations(filepath, file_description=""):
    logger.info(f"Loading and preparing {file_description} annotations from: {filepath}")
    try:
        # Load (adjust sep if needed)
        annotations = pd.read_csv(filepath, index_col=0)

        required_cols = ['player_num', 'timestamp', 'reference_player_num']
        if not all(col in annotations.columns for col in required_cols):
             raise ValueError(f"Annotations CSV must contain columns: {required_cols}")

        # Convert types
        annotations["player_num"] = annotations["player_num"].astype(int)
        annotations["timestamp"] = pd.to_numeric(annotations["timestamp"], errors='coerce')
        annotations["reference_player_num"] = annotations["reference_player_num"].astype(int)
        annotations.dropna(subset=["timestamp"], inplace=True)

        # Rename columns
        annotations = annotations.rename(columns={
            "player_num": "actual_target_player_num",
            "timestamp": "decision_timestamp",
            "reference_player_num": "passer_player_num"
        })
        annotations['annotation_row_id'] = annotations.index # Keep track of original row

        logger.info(f"{file_description} annotations prepared. Rows: {len(annotations)}")
        if annotations.empty:
            logger.warning(f"{file_description} annotations file is empty or resulted in empty data.")
        return annotations

    except FileNotFoundError:
        logger.error(f"{file_description} annotations file not found: {filepath}")
        return None # Return None to indicate failure
    except Exception as e:
        logger.error(f"Error loading/preparing {file_description} annotations: {e}")
        return None

# --- Load Train and Test Annotations ---
annotations_train = load_and_prepare_annotations(annotation_filepath_train, "TRAINING")
annotations_test = load_and_prepare_annotations(annotation_filepath_test, "TEST")

if annotations_train is None or annotations_train.empty:
    logger.error("Cannot proceed without valid training annotations.")
    exit()
if annotations_test is None:
    logger.warning("Test annotations not loaded. Will only report training accuracy.")
    # Allow script to continue to train, but testing won't happen


# --- Helper Functions (Keep implementations) ---
def calculate_overlap_area(target_x, target_y, x_ball, y_ball, away_players, angle_degrees=28.41389086074131, circle_radius=2):
    # ... (Keep the implementation exactly as before) ...
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
        triangle = Polygon(points);
        if not triangle.is_valid or triangle.area < 1e-6: return 0.0
    except Exception: return 0.0
    total_overlap_area = sum(
        triangle.intersection(Point(away_x, away_y).buffer(circle_radius)).area
        for away_x, away_y in zip(away_players["x"], away_players["y"]) if pd.notna(away_x) and pd.notna(away_y)
    )
    return total_overlap_area

# PyTensor compatible version for MCMC model
def calculate_score_utility_pt(overlap_area, distance_to_ball, ball_direction_x, alpha, beta, gamma):
    direction_bonus = pt.switch(ball_direction_x > 0.0, gamma, -gamma)
    penalty = alpha * overlap_area + beta * (-distance_to_ball)
    utility = -penalty + direction_bonus
    return utility

# Standard Python version for accuracy calculation after fitting
def calculate_score_utility_np(overlap, distance, direction, alpha_val, beta_val, gamma_val):
    if isinstance(direction, np.ndarray):
         direction_bonus = np.where(direction > 0.0, gamma_val, -gamma_val)
    else: direction_bonus = gamma_val if direction > 0.0 else -gamma_val
    penalty = alpha_val * overlap + beta_val * (-distance)
    utility = -penalty + direction_bonus
    return utility

# --- Function to Pre-calculate Features ---
def preprocess_annotations_for_model(annotations_df, gps_df, description=""):
    logger.info(f"Pre-calculating features for {description} data...")
    model_input_list = []
    required_gps_cols = ["time", "player_num", "x", "y", "Team", "ball_x", "ball_y"] # Check if ball pos is always present

    # Check if gps_df has required columns
    if not all(col in gps_df.columns for col in required_gps_cols):
        missing = [col for col in required_gps_cols if col not in gps_df.columns]
        logger.error(f"GPS data is missing required columns: {missing}")
        return [] # Return empty list on critical error

    for idx, ann_row in tqdm(annotations_df.iterrows(), total=len(annotations_df), desc=f"Processing {description} annotations"):
        ann_row_id = ann_row["annotation_row_id"]
        decision_time = ann_row["decision_timestamp"]
        passer_num = ann_row["passer_player_num"]
        actual_target = ann_row["actual_target_player_num"]

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
            direct = x_ball - t_x
            overlap = calculate_overlap_area(t_x, t_y, x_ball, y_ball, opponents_df)

            features_list.append({"overlap": overlap, "distance": dist, "direction": direct})
            target_nums.append(t_num)
            if t_num == actual_target: chosen_idx = i

        if chosen_idx != -1 and len(features_list) > 0:
            event_data = {
                "annotation_row_id": ann_row_id, "n_choices": len(features_list),
                "features_overlap": np.array([f["overlap"] for f in features_list]),
                "features_distance": np.array([f["distance"] for f in features_list]),
                "features_direction": np.array([f["direction"] for f in features_list]),
                "chosen_target_index": chosen_idx, "target_player_numbers": np.array(target_nums)
            }
            model_input_list.append(event_data)

    logger.info(f"Pre-calculation for {description} complete. Usable rows: {len(model_input_list)}")
    return model_input_list

# --- Pre-calculate for Training Data ---
model_input_data_train = preprocess_annotations_for_model(annotations_train, gps_data, "TRAINING")

if not model_input_data_train:
    logger.error("No usable training data available for the Bayesian model.")
    exit()

# --- Build PyMC Model ---
logger.info("Building PyMC model...")
coords = {"annotation_row": np.arange(len(model_input_data_train))}

with pm.Model(coords=coords) as pass_choice_model:
    # Priors
    alpha = pm.Uniform("alpha", lower=0.0, upper=1.0)
    beta = pm.Uniform("beta", lower=0.0, upper=0.5)
    gamma = pm.Uniform("gamma", lower=-2.0, upper=2.0) # Adjust range if needed

    # Likelihood
    logp_sum = 0.0
    for i in range(len(model_input_data_train)): # Use TRAINING data here
        event = model_input_data_train[i]
        n_choices = event["n_choices"]; chosen_idx = event["chosen_target_index"]
        if n_choices <= 0: continue

        overlap = pt.as_tensor_variable(event["features_overlap"])
        distance = pt.as_tensor_variable(event["features_distance"])
        direction = pt.as_tensor_variable(event["features_direction"])

        # Use PyTensor compatible utility function
        utility = calculate_score_utility_pt(overlap, distance, direction, alpha, beta, gamma)
        log_prob_chosen = utility[chosen_idx] - pm.logsumexp(utility)
        logp_sum += log_prob_chosen

    pm.Potential("likelihood", logp_sum)
logger.info("PyMC model built.")

# --- Run MCMC Sampler ---
logger.info(f"Starting MCMC sampling ({N_CHAINS} chains, {N_DRAWS} draws, {N_TUNE} tuning)...")
with pass_choice_model:
    idata = pm.sample(
        draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS,
        target_accept=TARGET_ACCEPT, random_seed=RANDOM_SEED, cores=1
    )
logger.info("MCMC sampling complete.")

# --- Analyze Posterior Results ---
logger.info("Analyzing posterior distributions...")
divergences = idata.sample_stats["diverging"].sum().item()
logger.info(f"Number of divergences: {divergences}")
if divergences > 0:
     logger.warning("Divergences encountered. Results may be unreliable.")

summary = az.summary(idata, var_names=["alpha", "beta", "gamma"], hdi_prob=0.94)
print("\nPosterior Summary Statistics (from Training Data):")
print(summary)

# --- Function to Calculate Accuracy ---
def calculate_accuracy_with_params(eval_data, alpha_val, beta_val, gamma_val, unique_events, description=""):
    logger.info(f"Calculating accuracy for {description} data...")
    if not eval_data: # Check if the input list is empty
        logger.warning(f"No data provided for {description} accuracy calculation.")
        return 0.0, 0, 0 # Return 0 accuracy and counts

    correct_predictions = 0
    total_rows_evaluated = len(eval_data)

    for event_row_data in eval_data:
        # Use the standard Python scoring function
        scores = calculate_score_utility_np(
            event_row_data["features_overlap"],
            event_row_data["features_distance"],
            event_row_data["features_direction"],
            alpha_val, beta_val, gamma_val
        )
        predicted_index = np.argmax(scores)
        if predicted_index == event_row_data["chosen_target_index"]:
            correct_predictions += 1

    accuracy = correct_predictions / unique_events if unique_events > 0 else 0.0
    logger.info(f"Accuracy for {description}: {accuracy:.4f} ({correct_predictions}/{unique_events} rows)")
    return accuracy, correct_predictions, unique_events

# --- Calculate Accuracy using Posterior Mean Parameters ---
logger.info("Calculating accuracy using posterior mean parameters...")
if 'idata' in locals() and idata is not None and idata.posterior is not None:
    mean_alpha = idata.posterior["alpha"].mean().item()
    mean_beta = idata.posterior["beta"].mean().item()
    mean_gamma = idata.posterior["gamma"].mean().item()
    logger.info(f"Posterior Mean Parameters: alpha={mean_alpha:.4f}, beta={mean_beta:.4f}, gamma={mean_gamma:.4f}")

    # --- Evaluate on TRAINING Data ---
    acc_train, correct_train, total_train = calculate_accuracy_with_params(
        model_input_data_train, mean_alpha, mean_beta, mean_gamma, 36,"TRAINING"
    )
    print(f"\nAccuracy on TRAINING Data (Player 24): {acc_train:.4f}")

    # --- Evaluate on TEST Data ---
    if annotations_test is not None and not annotations_test.empty:
        model_input_data_test = preprocess_annotations_for_model(annotations_test, gps_data, "TEST")
        if model_input_data_test: # Check if preprocessing yielded usable test data
            acc_test, correct_test, total_test = calculate_accuracy_with_params(
                model_input_data_test, mean_alpha, mean_beta, mean_gamma, 27, "TEST"
            )
            print(f"Accuracy on TEST Data (Player 2): {acc_test:.4f}")
        else:
            print("\nCould not calculate accuracy on TEST data (no usable rows after preprocessing).")
    else:
        print("\nSkipping TEST data accuracy calculation (test annotations not loaded or empty).")

else:
    logger.warning("MCMC results (idata) not found. Skipping accuracy calculation.")
    print("\nCould not calculate accuracy (MCMC results not available).")


### [Original script remains unchanged until the end of the posterior mean accuracy evaluation] ###

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
            alpha_sample = posterior["alpha"].sel(sample=sample_idx).values.item()
            beta_sample = posterior["beta"].sel(sample=sample_idx).values.item()
            gamma_sample = posterior["gamma"].sel(sample=sample_idx).values.item()

            correct = 0
            for event in events:
                scores = calculate_score_utility_np(
                    event["features_overlap"],
                    event["features_distance"],
                    event["features_direction"],
                    alpha_sample, beta_sample, gamma_sample
                )
                if np.argmax(scores) == event["chosen_target_index"]:
                    correct += 1

            acc = correct / len(events)
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
        print("[Player {player_id}] Best Parameter Combination:")
        print(f"  alpha = {best_params['alpha']:.4f}, beta = {best_params['beta']:.4f}, gamma = {best_params['gamma']:.4f}")

# Run posterior predictive accuracy evaluation split per player
evaluate_posterior_accuracy_per_player(model_input_data_train, idata, n_samples=100, description="TRAINING")

logger.info("Script finished.")
# --- Plotting Posterior Distributions ---
# (Keep this section the same)
try:
    if 'idata' in locals() and idata is not None:
        logger.info("Generating plots...")
        axes = az.plot_trace(idata, var_names=["alpha", "beta", "gamma"], combined=True)
        plt.suptitle("Parameter Trace Plots and Posteriors (from Training Data)", y=1.02)
        plt.tight_layout()
        plt.show()
        ax_posterior = az.plot_posterior(idata, var_names=["alpha", "beta", "gamma"])
        plt.suptitle("Posterior Distributions with HDI (from Training Data)", y=1.02)
        plt.tight_layout()
        plt.show()
    else:
        logger.info("Skipping plot generation (no MCMC data).")
except Exception as e:
    logger.error(f"Error during plotting: {e}")




logger.info("Script finished.")