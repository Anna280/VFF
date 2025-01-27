import psycopg2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

""" DB_PARAMS = {
    "dbname": "bold_tracking",
    "user": "admin",
    "password": "admin",
    "host": "localhost",
    "port": "5432"
}
 """
 
DB_PARAMS = {
    "dbname": "bold_tracking",
    "user": "admin",
    "password": "CrazySecure",
    "host": "kaspersvendsen.dk",
    "port": "5432"
}
def fetch_tracking_data():
    """Fetch ball tracking data from the database."""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    cur.execute("""
        SELECT x, y, spd
        FROM ball_tracking 
        WHERE x IS NOT NULL 
        AND y IS NOT NULL 
        AND spd IS NOT NULL;
    """)
    data = cur.fetchall()
    cur.close()
    conn.close()

    return pd.DataFrame(data, columns=['x', 'y', 'spd'])

def find_field_boundaries(df, outlier_threshold=2):
    """Find the field boundaries using statistical analysis."""
    # Remove outliers using IQR method
    Q1_x = df['x'].quantile(0.25)
    Q3_x = df['x'].quantile(0.75)
    IQR_x = Q3_x - Q1_x
    
    Q1_y = df['y'].quantile(0.25)
    Q3_y = df['y'].quantile(0.75)
    IQR_y = Q3_y - Q1_y
    
    df_filtered = df[
        (df['x'] >= Q1_x - outlier_threshold * IQR_x) &
        (df['x'] <= Q3_x + outlier_threshold * IQR_x) &
        (df['y'] >= Q1_y - outlier_threshold * IQR_y) &
        (df['y'] <= Q3_y + outlier_threshold * IQR_y)
    ]
    
    # Find the extreme points
    x_min = df_filtered['x'].quantile(0.01)
    x_max = df_filtered['x'].quantile(0.99)
    y_min = df_filtered['y'].quantile(0.01)
    y_max = df_filtered['y'].quantile(0.99)
    
    return x_min, x_max, y_min, y_max

def find_stationary_points(df, speed_threshold=0.5):
    """Find points where the ball was stationary."""
    return df[df['spd'] < speed_threshold]

def find_corners(boundaries, stationary_df, corner_radius=5.0):
    """Find corners using boundary intersections and stationary points."""
    x_min, x_max, y_min, y_max = boundaries
    corners = []
    confidences = []
    
    # Define corner regions
    corner_regions = [
        (x_min, y_min),  # Bottom left
        (x_max, y_min),  # Bottom right
        (x_max, y_max),  # Top right
        (x_min, y_max)   # Top left
    ]
    
    for x_ref, y_ref in corner_regions:
        # Find stationary points near this corner
        corner_points = stationary_df[
            (abs(stationary_df['x'] - x_ref) < corner_radius) &
            (abs(stationary_df['y'] - y_ref) < corner_radius)
        ]
        
        if not corner_points.empty:
            # Use density-weighted average for corner position
            weights = 1 / (1 + np.sqrt((corner_points['x'] - x_ref)**2 + 
                                     (corner_points['y'] - y_ref)**2))
            corner_x = np.average(corner_points['x'], weights=weights)
            corner_y = np.average(corner_points['y'], weights=weights)
            
            # Calculate confidence based on number of points and their distribution
            n_points = len(corner_points)
            point_spread = np.std(weights)
            confidence = min(n_points / 20.0 * (1 - point_spread), 1.0)
            
            corners.append([corner_x, corner_y])
            confidences.append(confidence)
        else:
            # If no stationary points found, use the boundary intersection
            corners.append([x_ref, y_ref])
            confidences.append(0.5)  # Lower confidence for boundary-only corners
    
    return np.array(corners), np.array(confidences)

def validate_field(corners):
    """Validate field dimensions and shape."""
    if len(corners) != 4:
        return False, "Incorrect number of corners"
    
    # Calculate side lengths
    sides = []
    for i in range(4):
        next_i = (i + 1) % 4
        side_length = np.sqrt(
            (corners[next_i][0] - corners[i][0])**2 +
            (corners[next_i][1] - corners[i][1])**2
        )
        sides.append(side_length)
    
    # Check if opposite sides are similar in length
    side_pairs_ratio = min(sides[0]/sides[2], sides[2]/sides[0])
    if side_pairs_ratio < 0.95:
        return False, "Opposite sides are not equal"
    
    # Calculate and validate aspect ratio
    width = min(max(sides[0], sides[2]), max(sides[1], sides[3]))
    length = max(max(sides[0], sides[2]), max(sides[1], sides[3]))
    ratio = length / width
    
    if not (1.3 <= ratio <= 1.7):
        return False, f"Invalid aspect ratio: {ratio:.2f}"
    
    return True, f"Valid field (aspect ratio: {ratio:.2f})"

def plot_field_analysis(df, corners, confidences, boundaries):
    """Plot the field analysis."""
    plt.figure(figsize=(15, 10))
    
    # Plot density map of all positions
    xy = np.vstack([df['x'], df['y']])
    z = gaussian_kde(xy)(xy)
    plt.scatter(df['x'], df['y'], c=z, s=1, alpha=0.1, cmap='Greys')
    
    # Plot field boundaries
    x_min, x_max, y_min, y_max = boundaries
    plt.axvline(x=x_min, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=x_max, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=y_min, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=y_max, color='gray', linestyle='--', alpha=0.5)
    
    # Plot corners
    for i, (corner, conf) in enumerate(zip(corners, confidences)):
        plt.scatter(corner[0], corner[1], 
                   color=plt.cm.RdYlGn(conf), 
                   s=200, marker='X', 
                   label=f'Corner {i+1} (conf: {conf:.2f})')
    
    # Connect corners
    corners_cycle = np.vstack([corners, corners[0]])
    plt.plot(corners_cycle[:, 0], corners_cycle[:, 1], 'r-', label='Field boundary')
    
    plt.grid(True)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Football Field Analysis')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def main():
    # Fetch data
    df = fetch_tracking_data()
    if df.empty:
        print("No data found in the database.")
        return
    
    # Find field boundaries
    boundaries = find_field_boundaries(df)
    
    # Find stationary points
    stationary_df = find_stationary_points(df)
    
    # Detect corners
    corners, confidences = find_corners(boundaries, stationary_df)
    
    # Validate field
    is_valid, validation_message = validate_field(corners)
    
    # Print results
    print("\nField Analysis Results:")
    print(f"Number of corners detected: {len(corners)}")
    print(f"Validation: {validation_message}")
    
    print("\nCorner Positions:")
    for i, (corner, conf) in enumerate(zip(corners, confidences), 1):
        print(f"Corner {i}: ({corner[0]:.2f}, {corner[1]:.2f}) - Confidence: {conf:.2f}")
    
    # Plot results
    plot_field_analysis(df, corners, confidences, boundaries)

if __name__ == "__main__":
    main()