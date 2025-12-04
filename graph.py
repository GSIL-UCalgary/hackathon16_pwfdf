import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

EARTH_RADIUS_KM = 6371.0 # Earth's radius in kilometers for Haversine conversion

def create_connectivity_graph(df, output_path, k, max_dist):
    """
    Loads fire data, calculates pairwise Haversine distances, identifies k nearest 
    neighbors (up to a max distance), and generates a CSV file for QGIS visualization.
    """

    # 1. Prepare Spatial Data
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        print("\nFATAL ERROR: The script requires 'Latitude' and 'Longitude' columns.")
        return

    coords_deg = df[['Latitude', 'Longitude']].astype(float).values
    fire_ids = df['ID'].tolist()
    storm_dates = df['StormDate'].values

    # Convert degrees to radians for Haversine calculation
    coords_rad = np.radians(coords_deg)
    
    # Extract Lat/Lon from the radian array
    latitudes = coords_rad[:, 0]
    longitudes = coords_rad[:, 1]

    # 2. Calculate Pairwise Haversine Distance Matrix using NumPy
    print("Calculating pairwise Haversine distances (Great-Circle distance)...")
    
    # Calculate difference in latitudes (dlat) and longitudes (dlon)
    dlat = latitudes[:, np.newaxis] - latitudes
    dlon = longitudes[:, np.newaxis] - longitudes

    # Haversine formula core part (a)
    a = np.sin(dlat / 2.0)**2 + np.cos(latitudes[:, np.newaxis]) * np.cos(latitudes) * np.sin(dlon / 2.0)**2

    # Central angle (c)
    c = 2 * np.arcsin(np.sqrt(a))

    # Distance matrix in kilometers
    distance_matrix = c * EARTH_RADIUS_KM
    
    print(f"Distances calculated in Kilometers (R={EARTH_RADIUS_KM} km).")

    # 3. Find the K Nearest Neighbors (KNN)
    # np.argsort returns the indices that would sort an array.
    sorted_indices = np.argsort(distance_matrix, axis=1)
    # Also get the distances corresponding to the sorted indices
    sorted_distances = np.take_along_axis(distance_matrix, sorted_indices, axis=1)

    # Exclude distance to self (index 0)
    neighbor_indices = sorted_indices[:, 1:k+1]
    neighbor_distances = sorted_distances[:, 1:k+1]

    # 4. Generate Edge List with Coordinates (New, Performant Format)
    edge_data = []
    
    print(f"Generating optimized edge list for max {k} neighbors within {max_dist} km...")
    
    # Get the coordinate arrays for easy lookup
    lon_coords = df['Longitude'].values
    lat_coords = df['Latitude'].values

    for i, source_id in enumerate(fire_ids):
        # Coordinates of the source node
        source_lon = lon_coords[i]
        source_lat = lat_coords[i]
        source_storm_date = storm_dates[i]

        # Iterate through the k nearest neighbors for this source node
        for j, neighbor_idx in enumerate(neighbor_indices[i]):
            current_distance = neighbor_distances[i, j]
            target_storm_date = storm_dates[neighbor_idx]

            if current_distance <= max_dist and source_storm_date == target_storm_date:
                target_id = fire_ids[neighbor_idx]
                target_lon = lon_coords[neighbor_idx]
                target_lat = lat_coords[neighbor_idx]
                
                # Append the full edge definition: Source, Target, and both coordinates
                edge_data.append({
                    'Source_ID': source_id,
                    'Target_ID': target_id,
                    'Source_Lon': source_lon,
                    'Source_Lat': source_lat,
                    'Target_Lon': target_lon,
                    'Target_Lat': target_lat,
                    'Distance': current_distance,
                    'StormDate': source_storm_date,
                })
            # else: the neighbor is too far, so we skip this connection

    # 5. Create DataFrame and Save to CSV
    edge_df = pd.DataFrame(edge_data)

    # Calculate the new average metric
    total_connections = len(edge_df)
    total_sources = len(df)
    avg_connections_per_source = total_connections / total_sources if total_sources > 0 else 0

    print(edge_df.head())
    # Save the resulting connectivity graph file
    edge_df.to_csv(output_path, index=False)
    print(f"\nSuccessfully generated performant edge list with {len(edge_df)} edges.")
    print(f"Output saved to: {output_path}")
    print(f"Average connections per source ID (accounting for distance cap): {avg_connections_per_source:.2f}")

    return edge_df
