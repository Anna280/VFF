import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_density(data, column, title="Density Plot", color="skyblue"):
    """Plot a KDE density plot for a specified column."""
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=data, x=column, fill=True, color=color)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_acceleration_with_events(ball, event_data, start_index=20, end_index=120, event_range=(2, 4),
                                   start_color="blue", end_color="darkviolet"):
    """Plot acceleration with markers for start and end of passes."""
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Plot acceleration
    ax1.plot(
        ball['time'][start_index:end_index],
        ball['acceleration'][start_index:end_index],
        label='Original Acceleration',
        marker='o',
        linestyle='--',
        alpha=0.6,
        linewidth=2,
        markersize=4,
        color='dodgerblue'
    )
    
    ax1.set_xlabel("Time (s)", fontsize=14)
    ax1.set_ylabel("Acceleration (m/s²)", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Extract event times
    start_times = event_data['Start Time [s]'][event_range[0]:event_range[1]]
    end_times = event_data['End Time [s]'][event_range[0]:event_range[1]]
    
    # Interpolate y-values
    accel_time = ball['time'][start_index:end_index]
    accel_values = ball['acceleration'][start_index:end_index]
    start_y = np.interp(start_times, accel_time, accel_values)
    end_y = np.interp(end_times, accel_time, accel_values)

    ax1.scatter(start_times, start_y, color=start_color, label='Start Time', s=60, zorder=5)
    ax1.scatter(end_times, end_y, color=end_color, label='End Time', s=60, zorder=5)

    for x, y in zip(start_times, start_y):
        ax1.text(x - 0.11, y - 60.5, 'kicked', rotation=45, color=start_color, fontsize=10)

    for x, y in zip(end_times, end_y):
        ax1.text(x, y + 20.5, 'received', rotation=45, color=end_color, fontsize=10)

    plt.title("Ball Acceleration with Kick and Receive Markers", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_smoothed_acceleration(ball, start_index=190, end_index=300, window_size=5, color="dodgerblue"):
    """Plot original and smoothed acceleration using a moving average."""
    ball = ball.copy()
    ball['smoothed_acceleration'] = ball['acceleration'].rolling(window=window_size, center=True).mean()
    
    plt.figure(figsize=(14, 7))
    plt.plot(
        ball['time'][start_index:end_index],
        ball['acceleration'][start_index:end_index],
        label='Original Acceleration',
        linestyle='--',
        alpha=0.5,
        linewidth=2,
        color='gray'
    )
    plt.plot(
        ball['time'][start_index:end_index],
        ball['smoothed_acceleration'][start_index:end_index],
        label=f'Smoothed (Moving Avg, window={window_size})',
        linewidth=2.5,
        color=color
    )

    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Acceleration (m/s²)", fontsize=14)
    plt.title("Acceleration Smoothing with Moving Average", fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.show()
