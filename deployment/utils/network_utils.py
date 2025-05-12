import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

def plot_pass_network_for_player(df_passes, player):
    """
    Creates a pass network plot for a given player.
    Only passes where the selected player is the 'From' player are used.
    Returns a matplotlib.figure.Figure.
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
