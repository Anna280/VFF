import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

def generate_player_decision_accuracy(pass_data,gps_data, real_player2_events, annotations2,player_custom_events,subject):
    pass_data = pd.read_csv(pass_data)
    gps_data=  pd.read_csv(gps_data, index_col=0)
    annotations2 = pd.read_csv(annotations2,sep=";")
    real_player2_events = pd.read_csv(real_player2_events)
    
    gps_data_home = gps_data[gps_data["Team"]== "home"]
    # Step 1: subset pass_data to home team only
    pass_data_home = pass_data[pass_data["Team"] == "home"].reset_index(drop=True)
    pass_data_home = pass_data_home[pass_data_home["uncertainty"] < 2].reset_index(drop=True)
    # Columns to keep from gps_data

    gps_columns = ["time", "player_id", "player_num", "x", "y", "spd", "ball_x","ball_y","Team","acceleration", "smoothed_acceleration"]

    # Prepare an empty DataFrame for extracted data
    extracted_gps_data = pd.DataFrame(columns=gps_columns + ["pass_event_id"])

    # Steps 2 & 3: Extract GPS data for each pass event
    for idx, row in pass_data_home.iterrows():
        from_player = row["From"]
        start_time = row["Start Time [s]"]
        end_time = row["End Time [s]"]
        
        # Extract relevant gps data
        gps_subset = gps_data_home[
            (gps_data_home["player_num"] == from_player) &
            (gps_data_home["time"] >= start_time) &
            (gps_data_home["time"] <= end_time)
        ][gps_columns].copy()

        # Add pass event identifier
        gps_subset["pass_event_id"] = idx
        
        # Concatenate results
        extracted_gps_data = pd.concat([extracted_gps_data, gps_subset], ignore_index=True)

    extracted_gps_data["distance_to_ball"] = np.hypot(
        extracted_gps_data["x"] - extracted_gps_data["ball_x"],
        extracted_gps_data["y"] - extracted_gps_data["ball_y"]
    )
    low_distance = extracted_gps_data[extracted_gps_data["distance_to_ball"] < 2]
    # Assuming df is your DataFrame
    df = low_distance.copy()
    # Identify chunks by consecutive player_num
    df['player_chunk'] = (df['player_num'] != df['player_num'].shift()).cumsum()
    # Get index of max acceleration per chunk
    max_acceleration_indices = df.groupby('player_chunk')['acceleration'].idxmax() - 10
    # Initialize a list of zeros with length 2500
    binary_list = [0] * len(extracted_gps_data)
    # Set positions from max_acceleration_indices to 1, ensuring indices within bounds
    for idx in max_acceleration_indices:
        if 0 <= idx < len(extracted_gps_data):
            binary_list[idx] = 1
    # Verify the result (optional)
    extracted_gps_data["decision_making_point"] = binary_list
    # Iterate over each row in extracted_gps_data with decision-making points

    decision_points = extracted_gps_data[extracted_gps_data["decision_making_point"] == 1].reset_index()

    decision_points = decision_points[decision_points["player_num"]==subject] ## PLAYER NUM NEEDS TO BE AN INPUT VARIABLE
    decision_points.reset_index(drop=True, inplace=True)
    decision_points.index = decision_points.index + 1
    df_filtered_player = decision_points.drop(index=player_custom_events[0]) ## THE INDEXES NEED TO BE AN INPUT VARIABLE

    pass_data_home = real_player2_events[real_player2_events["Team"] == "home"].reset_index(drop=True)
    pass_data_home = pass_data_home[pass_data_home["uncertainty"] < 2].reset_index(drop=True)


    annotations2 = annotations2.drop(index=player_custom_events[1])
    annotations2["player_num"] = annotations2["player_num"].astype(int)
    annotations2.index = annotations2.index +1 

    merged_df = df_filtered_player.merge(pass_data_home, left_on='pass_event_id', right_index=True)
    merged_df = merged_df.drop(merged_df.columns[0], axis = 1)
    merged_df["Picture_id"] = merged_df.index
    final_df = annotations2.merge(merged_df, on="Picture_id", how="left")

    new_df = final_df[['Picture_id', 'player_num_x' ,'To','Type']]
    new_df.index = new_df.index +1

    matching_rows2 = new_df[new_df["player_num_x"] == new_df["To"]]

    # Count occurrences per Type
    annotation_counts = annotations2["Type"].value_counts().rename("Annotated")
    prediction_counts = matching_rows2["Type"].value_counts().rename("Model Predicted")

    # Combine into one DataFrame for plotting
    comparison_df_player_2_decision = pd.concat([annotation_counts, prediction_counts], axis=1).fillna(0)

    comparison_df_player_2_decision["Accuracy (%)"] = (comparison_df_player_2_decision["Model Predicted"] / comparison_df_player_2_decision["Annotated"]) * 100
    
    return comparison_df_player_2_decision

def generate_plot_of_model_vs_player_accuracy(model_output, comparison_df_player_decision,annotations):

    # Align by row — they match line-by-line
    model_output = model_output.copy()
    model_output["Type"] = annotations["Type"]  # Copy the pass type from annotations

    # Compute hit (correct prediction)
    model_output["hit"] = model_output["actual_target_player_num"] == model_output["predicted_player_num"]


    accuracy_by_type = (
        model_output.groupby("Type")["hit"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "Eval Accuracy"})
        .reset_index()
    )
    accuracy_by_type["Eval Accuracy (%)"] = accuracy_by_type["Eval Accuracy"] * 100
    

    # Merge both into a single plot table
    all_types = ['HFM', 'LB', 'LINK', 'SWITCH']

    plot_df = pd.DataFrame({
        "Type": all_types,
        "Eval Accuracy (%)": accuracy_by_type.set_index("Type").reindex(all_types)["Eval Accuracy (%)"],
        "Accuracy (%)": comparison_df_player_decision.reindex(all_types)["Accuracy (%)"]
    }).fillna(0)
    print(plot_df)

    type_counts = annotations["Type"].value_counts().to_dict()
    # Step 2: Add count info to your plot_df
    plot_df["Annotation Count"] = plot_df["Type"].map(type_counts).fillna(0).astype(int)


    # Prepare values
    x = np.arange(len(plot_df))
    width = 0.35
    eval_scores = plot_df["Eval Accuracy (%)"]
    bib_scores = plot_df["Accuracy (%)"]

    # Define colors (optional)
    colors = ["#f1a226", "#006a5d"]

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width/2, eval_scores, width, label="Model Decisions", color=colors[0])
    bars2 = plt.bar(x + width/2, bib_scores, width, label="Player 2's Decisions", color=colors[1])

    # Add text labels above bars
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{height:.0f}%", ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{height:.0f}%", ha='center', va='bottom', fontsize=9)

    # Style tweaks
    xtick_labels = [f"{t} (n={c})" for t, c in zip(plot_df["Type"], plot_df["Annotation Count"])]
    plt.xticks(x, xtick_labels, rotation=30, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Prediction Accuracy Relative To Ideal Choices Player 2", fontsize=14, fontweight="bold")
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(frameon=False, fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_scores_for_index(df, index):
    row = df.iloc[index]
    scores = row["scores"]
    chosen_idx = row["chosen_target_index"]
    predicted_idx = row["predicted_index"]

    # Parse player numbers safely
    raw_labels = [12, 11,  8, 28, 23, 18, 24, 10, 16, 13]
    if isinstance(raw_labels, str):
        target_labels = [int(x) for x in raw_labels.strip("[]").split()]
    else:
        target_labels = list(range(len(scores)))

    # Define bar colors
    colors = []
    for i in range(len(scores)):
        if i == chosen_idx and i == predicted_idx:
            colors.append("blue")
        elif i == chosen_idx:
            colors.append("#006a5d")
        elif i == predicted_idx:
            colors.append("red")
        else:
            colors.append("gray")

    # Define color legend mapping
    color_legend = {
        "blue": "Correct & predicted",
        "#006a5d": "Correct target",
        "red": "Model prediction",
        "gray": "Other candidates"
    }

    # Create legend handles from color_legend
    #legend_patches = [Patch(color=color, label=label) for color, label in color_legend.items()]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(scores)), scores, color=colors)
    plt.xticks(range(len(scores)),  target_labels)
    plt.ylabel("Score")
    plt.xlabel("Player Number")
    plt.title(f"Score Across Potential Recievers case {index}")
   # plt.grid(axis='y', linestyle='--', alpha=0.5)
    #plt.legend(handles=legend_patches, frameon=False)
    plt.tight_layout()
    plt.show()
    
def plot_posterior_density_parameter_highest_accuracy(csv_path, accuracy_threshold,id):
    df = pd.read_csv(csv_path)
    # Step 1: Filter rows where accuracy starts with "0.55"
    filtered = df[df["accuracy"].astype(str).str.startswith(accuracy_threshold)]

    # Step 2: Compute mean of all parameter values
    param_cols = ["alpha", "beta", "k"]
    param_means = df[param_cols].mean()

    # Step 3: Compute Euclidean distance to mean
    def distance_to_mean(row, means):
        return np.linalg.norm(row[param_cols].values - means.values)

    filtered["distance_to_mean"] = filtered.apply(lambda row: distance_to_mean(row, param_means), axis=1)

    # Step 4: Get the most balanced high-accuracy sample
    highlight_row = filtered.loc[filtered["distance_to_mean"].idxmin()]

    # Step 5: Plot parameter densities + highlight selected sample
    colors = {"alpha": "#1b9e77", "beta": "#d95f02", "k": "#7570b3"}

    fig, axes = plt.subplots(3, 1, figsize=(15, 9))
    for i, param in enumerate(param_cols):
        ax = axes[i]
        
        # Density plot
        sns.kdeplot(df[param], fill=True, ax=ax, color=colors[param])

        # 94% HDI
        hdi_bounds = az.hdi(df[param].values, hdi_prob=0.94)
        ax.axvspan(hdi_bounds[0], hdi_bounds[1], color=colors[param], alpha=0.2, label="94% HDI")

        # Highlight "balanced best" sample
        ax.axvline(highlight_row[param], color=colors[param], linestyle="--", linewidth=2,
                label=f"Highlighted: {highlight_row[param]:.4f}")

        # Labels
        ax.set_title(f"Density of {param}")
        ax.set_xlabel(param)
        ax.set_ylabel("Density")
        ax.legend()

    # Final layout
    plt.suptitle(f"Player {id}Highest Accuracy Parameter Sample and 94% HDI intervals", fontsize=20, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
def compute_top_k_accuracy(df, k=3):
    correct_in_top_k = 0
    for _, row in df.iterrows():
        scores = row["scores"]
        chosen = row["chosen_target_index"]
        
        # Get indices of top-k scores
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        if chosen in top_k_indices:
            correct_in_top_k += 1

    return correct_in_top_k / len(df)
