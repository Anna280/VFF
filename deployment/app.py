import streamlit as st
import pandas as pd
import os
import shutil
import numpy as np

from utils.video_utils import create_clips
from utils.network_utils import plot_pass_network_for_player
from utils.file_utils import create_zip_for_clips

st.title("Generate ball pass events")

st.markdown("""
**Choose a "from" player and "to" player, then select the zone combination available in that subset**  
""")

# Set paths for your main video and output folder inside the container
video_file = os.getenv("VIDEO_FILE", "/app/raw_video/video.mp4")
output_folder = "/app/clip_output"

# Load event data from CSV inside the container
csv_path = os.getenv("CSV_FILE", "/app/tracking/test_for_streamit.csv")

if 'event_data_passes' not in st.session_state:
    event_data_passes = pd.read_csv(csv_path)
    st.session_state.event_data_passes = event_data_passes
else:
    event_data_passes = st.session_state.event_data_passes

# --- Team Filter ---
if "Team" in event_data_passes.columns:
    team_filter = st.radio("Filter passes by team:", options=["All", "home", "away"], index=0)
    if team_filter != "All":
        event_data_passes = event_data_passes[event_data_passes["Team"] == team_filter]

# Create select boxes for "From" and "To" players
unique_from_players = sorted(event_data_passes["From"].unique())
unique_to_players = sorted(event_data_passes["To"].unique())

from_player = st.selectbox("Select From Player", unique_from_players,
                           index=unique_from_players.index(20) if 20 in unique_from_players else 0)
to_options = ["All"] + unique_to_players
to_player = st.selectbox("Select To Player (or choose All)", to_options, index=0)

# Filter dataset
if to_player == "All":
    player_subset = event_data_passes[event_data_passes["From"] == from_player]
else:
    player_subset = event_data_passes[(event_data_passes["From"] == from_player) &
                                      (event_data_passes["To"] == to_player)]

if player_subset.empty:
    st.write("No passes available for the selected player combination.")
else:
    player_subset = player_subset.copy()
    player_subset["zone_combo"] = player_subset.apply(
        lambda row: f"{row['start_x_zone']} | {row['start_y_zone']} -> {row['end_x_zone']} | {row['end_y_zone']}",
        axis=1
    )
    
    unique_zone_combos = sorted(player_subset["zone_combo"].unique())
    selected_zone_combo = st.selectbox("Select Zone Combination", ["All"] + unique_zone_combos, index=0)

st.subheader("Generate Clips")
if st.button("Generate Clips"):
    if player_subset.empty:
        st.write("No passes available for the selected player combination.")
    else:
        final_subset = player_subset.copy()
        if selected_zone_combo != "All":
            final_subset = final_subset[final_subset["zone_combo"] == selected_zone_combo]
        
        if final_subset.empty:
            st.write("No passes found for the selected zone combination.")
        else:
            final_subset = final_subset.sort_values(by="uncertainty", ascending=True)
            start_times = [x + 322.0 for x in final_subset["Start Time [s]"]]
            end_times = [x + 330.0 for x in final_subset["End Time [s]"]]

            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            
            try:
                created_clips = create_clips(video_file, start_times, end_times, output_folder)
                st.write("Clips generated:")
                with st.expander("Show/Hide Clips", expanded=True):
                    for i, clip in enumerate(created_clips):
                        pass_row = final_subset.iloc[i]
                        st.write(f"**Clip {i+1}: From {pass_row['From']} to {pass_row['To']}** (Avg. amount of players close to ball during pass:  {pass_row['uncertainty']:.2f})")
                        st.video(clip)
                        
                zip_buffer = create_zip_for_clips(output_folder)
                st.download_button(
                    label="**Export Clips as ZIP**",
                    data=zip_buffer,
                    file_name="clips.zip",
                    mime="application/zip"
                )
            except Exception as e:
                st.error(f"An error occurred during clip extraction: {e}")

if to_player == "All":
    st.subheader("Pass Network Visualization")
    fig = plot_pass_network_for_player(event_data_passes, from_player)
    if fig is not None:
        st.pyplot(fig)
