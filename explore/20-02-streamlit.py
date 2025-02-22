import streamlit as st
import pandas as pd
import os
import subprocess
import networkx as nx
import matplotlib.pyplot as plt
import zipfile
import io
import shutil

# --- Video Clip Extraction Functions ---

def ffmpeg_extract_subclip_no_data(video_path, start_time, end_time, output_file):
    """
    Extracts a subclip from video_path between start_time and end_time (in seconds)
    and writes it to output_file, disabling data streams.
    """
    duration = end_time - start_time
    cmd = [
        "ffmpeg",
        "-y",                   # overwrite output file if exists
        "-ss", str(start_time), # start time
        "-i", video_path,       # input file
        "-t", str(duration),    # duration of the clip
        "-map", "0",            # include all streams...
        "-dn",                  # ...but disable data streams (e.g., timecode)
        "-c", "copy",           # copy codecs (no re-encoding)
        output_file
    ]
    subprocess.run(cmd, check=True)

def create_clips(video_path, start_times, end_times, output_folder):
    """
    Create video clips from a source video given lists of start and end times.
    Returns a list of created clip file paths.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    created_files = []
    for i, (start, end) in enumerate(zip(start_times, end_times)):
        output_file = os.path.join(output_folder, f"clip_{i+1}.mp4")
        ffmpeg_extract_subclip_no_data(video_path, start, end, output_file)
        created_files.append(output_file)
    return created_files

# --- Pass Network Plot Function for a Single Player ---

def plot_pass_network_for_player(df_passes, player):
    """
    Creates a pass network plot for a given player.
    Only passes where the selected player is the 'From' player are used.
    Nodes represent players and edges are weighted (with numbers) by pass counts.
    
    Returns:
        matplotlib.figure.Figure: The figure containing the network plot.
    """
    # Filter passes for the given player
    player_passes = df_passes[df_passes["From"] == player]
    if player_passes.empty:
        st.write(f"No passes found for player {player} to plot a network.")
        return None

    # Create a directed graph and count passes from the player
    G = nx.DiGraph()
    pass_counts = player_passes.groupby(["From", "To"]).size().reset_index(name="pass_count")
    players = set(pass_counts["From"]).union(set(pass_counts["To"]))
    G.add_nodes_from(players)
    for _, row in pass_counts.iterrows():
        G.add_edge(row["From"], row["To"], weight=row["pass_count"])

    # Use a spring layout for a prettier visualization
    pos = nx.spring_layout(G, seed=42)
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="skyblue", edgecolors="black")
    nx.draw_networkx_edges(G, pos, width=[w for w in edge_weights], alpha=0.7, edge_color="gray")
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    
    # Draw edge labels (pass counts) in red
    edge_labels = {(u, v): G[u][v]["weight"] for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    
    plt.title(f"Pass Network for Player {player}")
    fig = plt.gcf()
    plt.close(fig)
    return fig

# --- Utility Function to Create a Zip Archive of the Clips Folder ---

def create_zip_for_clips(folder):
    """
    Zips all files in the given folder and returns a BytesIO buffer.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            zipf.write(file_path, arcname=filename)
    buffer.seek(0)
    return buffer

# --- Streamlit App UI ---

st.title("Clip Generator and Viewer")

st.markdown("""
**Choose a "from" player and "to" player**  
""")

# Set paths for your main video and output folder
video_file = "SL-R16_AAB-VFF_TACTICAL_241124.mp4"  # Your video file path
output_folder = "video_clips"

# Load your event data (filtering for passes) from your CSV
if 'event_data_passes' not in st.session_state:
    event_data_passes = pd.read_csv("/Users/annadaugaard/Desktop/VFF/explore/test_for_streamit.csv")
    st.session_state.event_data_passes = event_data_passes
else:
    event_data_passes = st.session_state.event_data_passes

# --- Team Filter ---
# If your event_data_passes DataFrame has a "Team" column, add a filter option.
if "Team" in event_data_passes.columns:
    team_filter = st.radio("Filter passes by team:", options=["All", "home", "away"], index=0)
    if team_filter != "All":
        event_data_passes = event_data_passes[event_data_passes["Team"] == team_filter]

# Create select boxes for "From" and "To" players
unique_from_players = sorted(event_data_passes["From"].unique())
unique_to_players = sorted(event_data_passes["To"].unique())

from_player = st.selectbox("Select From Player", unique_from_players, index=unique_from_players.index(20) if 20 in unique_from_players else 0)
# For "To" selection, add an "All" option at the top.
to_options = ["All"] + unique_to_players
to_player = st.selectbox("Select To Player (or choose All)", to_options, index=0)

# Button to generate clips
st.subheader("Generate Clips")
if st.button("Generate Clips"):
    # Delete previous clips in the local folder if they exist
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    # Filter passes based on the selection
    if to_player == "All":
        pair_passes = event_data_passes[event_data_passes["From"] == from_player]
    else:
        pair_passes = event_data_passes[
            (event_data_passes["From"] == from_player) &
            (event_data_passes["To"] == to_player)
        ]
        
    if pair_passes.empty:
        st.write(f"No passes found for {from_player} with To: {to_player}.")
    else:
        # Adjust timestamps with offsets: +322.0 for start, +330.0 for end
        start_times = [x + 322.0 for x in pair_passes["Start Time [s]"]]
        end_times = [x + 330.0 for x in pair_passes["End Time [s]"]]
        try:
            created_clips = create_clips(video_file, start_times, end_times, output_folder)
            st.write("Clips generated:")
            with st.expander("Show/Hide Clips", expanded=True):
                # Display each clip with a description showing the From and To players
                for i, clip in enumerate(created_clips):
                    pass_row = pair_passes.iloc[i]
                    from_id = pass_row["From"]
                    to_id = pass_row["To"]
                    st.video(clip)
                    st.write(f"**Clip {i+1}: From {from_id} to {to_id}**")
                         
            # Provide an option to export the clips folder as a zip file
            zip_buffer = create_zip_for_clips(output_folder)
            st.download_button(
                label="**Export Clips as ZIP**",
                data=zip_buffer,
                file_name="clips.zip",
                mime="application/zip"
            )
        except subprocess.CalledProcessError as e:
            st.error(f"An error occurred during clip extraction: {e}")
            
# Display the network visualization at the beginning if "All" is selected
if to_player == "All":
    st.subheader("Pass Network Visualization")
    fig = plot_pass_network_for_player(event_data_passes, from_player)
    if fig is not None:
        st.pyplot(fig)
