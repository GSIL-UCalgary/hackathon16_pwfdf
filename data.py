import os
import sciencebasepy
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Proj, transform, Transformer

from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
import geopandas as gpd
from shapely.geometry import Point
from sklearn.impute import SimpleImputer

from graph import create_connectivity_graph

# Downloads all of the hazard assessments to dir
def download_pwfdf_collection():
    dir = 'data/collection'
    sb = sciencebasepy.SbSession()
    id = '6818f950d4be0208bc3e0165' #Post-Wildfire Debris-Flow Hazard Assessment (PWFDF) Collection
    item = sb.get_item(id)

    print(f"Title: {item.get('title', 'No title')}")
    print(f"Summary: {item.get('summary', 'No summary')}")
    print(f"Item URL: https://www.sciencebase.gov/catalog/item/{id}")

    child_ids = sb.get_child_ids(id)

    path = Path(dir)
    path.mkdir(exist_ok=True)

    for child_id in child_ids:
        child_item = sb.get_item(child_id)
        child_title = child_item.get('title', 'No title')
        child_path = Path(dir + '/' + child_title)
        if not (os.path.isdir(child_path)):
            print(f"Downloading: {child_title} to {child_path}")
            child_path.mkdir(exist_ok=True)
            sb.get_item_files(child_item, child_path)

class PWFDF_Entry:
    def __init__(self, d):
        self.d = d
        self.utm_x = self.d['UTM_X']
        self.utm_y = self.d['UTM_Y']
        self.zone = self.d['UTM_Zone']

        utm_proj = Proj(proj='utm', zone=self.zone, ellps='WGS84', datum='WGS84', south=False)
        wgs84_proj = Proj(proj='latlong', ellps='WGS84', datum='WGS84')
        self.transformer = Transformer.from_proj(utm_proj, wgs84_proj, always_xy=True)
        self.transformer_albers = Transformer.from_crs(
            f"EPSG:326{self.zone:02d}",  # UTM Zone in WGS84 (North)
            "EPSG:5070",                  # NAD83 / Conus Albers
            always_xy=True
        )

    def coordinates_wgs84(self):
        return self.transformer.transform(self.utm_x, self.utm_y)
    
    def coordinates_5070(self):
        return self.transformer_albers.transform(self.utm_x, self.utm_y)
    
    def bounds(self, buffer_km):
        buffer_m = buffer_km * 1000 # Convert buffer from km to meters

        # Calculate bounds in UTM
        utm_west  = self.utm_x - buffer_m
        utm_south = self.utm_y - buffer_m
        utm_east  = self.utm_x + buffer_m
        utm_north = self.utm_y + buffer_m

        west, south =  self.transformer.transform(utm_west, utm_south)
        east, north =  self.transformer.transform(utm_east, utm_north)

        return (west, south, east, north)

    def __getitem__(self, key):
        return self.d[key]

def get_usgs_mask(df, duration):
    T = df['PropHM23'].values
    F = df['dNBR/1000'].values
    S = df['KF'].values
    if duration == '15min':
        R = df['Acc015_mm'].values
    elif duration == '30min':
        R = df['Acc030_mm'].values
    else:  # 60min
        R = df['Acc060_mm'].values
    mask = ~(np.isnan(T) | np.isnan(F) | np.isnan(S) | np.isnan(R))
    #mask = ~df.isnull().any(axis=1).values
    return mask

def fill_nan(df):
    #
    # Filling in NaN values
    #
    # Strategy 1: Remove rows with critical missing values
    critical_features = ['UTM_X', 'UTM_Y']
    df = df.dropna(subset=critical_features)

    # Strategy 2: Fill storm-related features with 0 (assuming no storm = 0)
    storm_zero_features = ['StormDur_H', 'StormAccum_mm', 'StormAvgI_mm/h', 'Peak_I15_mm/h', 'Peak_I30_mm/h', 'Peak_I60_mm/h', 'Acc015_mm', 'Acc030_mm', 'Acc060_mm']
    df[storm_zero_features] = df[storm_zero_features].fillna(0)

    # Strategy 3: Median imputation for remaining features
    # First, identify which features still have NaN values
    remaining_features = df.select_dtypes(include=[np.number]).columns.tolist()
    remaining_features = [f for f in remaining_features if f not in critical_features + storm_zero_features]

    # Only impute features that actually have NaN values
    features_with_nan = [f for f in remaining_features if df[f].isna().any()]

    if features_with_nan:
        print(f"Imputing {len(features_with_nan)} features with median:")
        print(f"  {features_with_nan}")
        
        imputer = SimpleImputer(strategy='median')
        df[features_with_nan] = imputer.fit_transform(df[features_with_nan])
        
        # Verify imputation worked
        print(f"Remaining NaN values: {df[features_with_nan].isna().sum().sum()}")
    else:
        print("No remaining features need imputation")

    return df

def normalize(df, scaler=None):
    all_features = list(df.columns)
    features_to_include = [
        'GaugeDist_m',
        'StormDur_H',
        'StormAccum_mm',
        'StormAvgI_mm/h',
        'Peak_I15_mm/h',
        'Peak_I30_mm/h',
        'Peak_I60_mm/h',
        'ContributingArea_km2',
        'Acc015_mm',
        'Acc030_mm',
        'Acc060_mm',
        'KF_Acc015'
    ]

    normalize_features = [f for f in features_to_include if f in all_features]

    # Assuming 'scaler' is None initially (for training set) or an existing scaler (for test set)
    if len(normalize_features) != 0:
        if scaler is None:
            scaler = StandardScaler()
            # Fit and transform ONLY the selected columns
            df[normalize_features] = scaler.fit_transform(df[normalize_features])
            print(f"\nNormalized {len(normalize_features)} features (fitted new scaler)")
        else:
            # Use provided scaler (for test set)
            # Transform ONLY the selected columns
            df[normalize_features] = scaler.transform(df[normalize_features])
            print(f"\nNormalized {len(normalize_features)} features (using provided scaler)")
            
        return df, scaler
    else:
        return df, None

all_features = None

class PWFDF_Data:
    # original data
    og_path = 'data/ofr20161106_appx-1.xlsx'
    og_sheet_name = 'Appendix1_ModelData'
    
    # updated data
    path = 'data/data.csv'

    graph_path = 'data/graph.csv'

    def __init__(self, use_cached=False):

        # load from original data spreadsheet
        if (not use_cached) or (not os.path.exists(self.path)):
            print(f"Loading Excel ({self.og_path})")
            df = pd.read_excel(self.og_path, sheet_name=self.og_sheet_name)

            #
            # Modifying the dataset
            #

            # Turns the string ids into number ids ('bck' -> 0)
            self.encoders = {}
            fire_cols = ['Fire Name', 'Fire_ID', 'Fire_SegID', 'State']
            for col in fire_cols:
                le = LabelEncoder()
                le.fit(df[col].astype(str).unique()) 
                self.encoders[col] = le
            
            for col in fire_cols:
                if col in df.columns and col in self.encoders:
                    le = self.encoders[col]
                    df[col] = le.transform(df[col].astype(str))

            # Adding columns
            df['ID'] = df.index + 1 # Add ID
            df['Missing_Data'] = df.isna().any(axis=1).astype(int) # Identify missing data
            df['Latitude'] = None
            df['Longitude'] = None
            df['KF_Acc015'] = df['KF'] * df['Acc015_mm']

            for zone in df['UTM_Zone'].unique():
                # Build a transformer for this zone
                transformer = Transformer.from_crs(
                    f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs",
                    "EPSG:4326",
                    always_xy=True
                )
                
                mask = df['UTM_Zone'] == zone
                xs = df.loc[mask, 'UTM_X'].values
                ys = df.loc[mask, 'UTM_Y'].values
                
                lons, lats = transformer.transform(xs, ys)
                
                df.loc[mask, 'Longitude'] = lons
                df.loc[mask, 'Latitude'] = lats

            # new training test split?
            

            print("Missing Data:")
            print(f"Total: {df['Missing_Data'].sum()} / {len(df)}")
            train_df = df[df['Database'] == 'Training']
            test_df = df[df['Database'] == 'Test']
            print(f"Training: {train_df['Missing_Data'].sum()} / {len(train_df)}")
            print(f"Test: {test_df['Missing_Data'].sum()} / {len(test_df)}")

            print(f"Saving CSV ({self.path})")
            df.to_csv(self.path, index=False)

            edge_df = create_connectivity_graph(df, self.graph_path, k=3, max_dist=4.0)

        else:
            df = pd.read_csv(self.path)
            edge_df = pd.read_csv(self.graph_path)

        self.df = df 
        self.graph_df = edge_df

        global all_features 
        all_features = list(df.columns) 

        # features that shouldn't be included in the input
        all_features.remove('Response')
        all_features.remove('Database')
        all_features.remove('StormDate')
        all_features.remove('StormStart')
        all_features.remove('StormEnd')

    def prepare_data_usgs(self, features, split='Training', duration='15min', scaler=None, max_neighbors=5):
        """
        Prepares data for graph-based model, including filtering, normalization,
        and structuring features into a [N, K, F] tensor.
        
        self.df and self.graph_df are expected to be available.
        
        Args:
            features (list): List of column names to use as features (F).
            split (str): 'Training' or 'Test'.
            duration (str): Filter for 'get_usgs_mask'.
            scaler (StandardScaler, optional): Pre-fitted scaler.
            max_neighbors (int): Maximum number of neighbors to include (K-1).
            
        Returns:
            tuple: (X_graph, y, scaler) where X_graph has shape [N, max_neighbors + 1, F].
        """
        df = self.df
        graph_df = self.graph_df 
        
        # 1. Filtering and Cleaning
        df = df[df['Database'] == split].copy()
        mask = get_usgs_mask(df, duration) # Assuming this function exists
        df = fill_nan(df)                   # Assuming this function exists

        if split == 'Test':
            df = df[mask].copy() 

        df, scaler = normalize(df, scaler)

        X_matrix = df[features].values
        y = df['Response'].values
        
        N = len(df)                  # Number of nodes in the current split (e.g., 1550)
        F = len(features)            # Number of features (e.g., 16)
        K = max_neighbors + 1        # Number of nodes per sample (e.g., 5 + 1 = 6)
        
        # Initialize the output feature tensor X_graph [N, K, F]
        X_graph = np.zeros((N, K, F))
        
        # Map 'ID' to row index for fast feature lookup
        node_ids = df['ID'].tolist()
        id_to_index = {id_val: i for i, id_val in enumerate(node_ids)}
        
        # 4. Create Graph Part (Stacking Node and Neighbor Features)
        print(f"\nStructuring data into graph tensor [N={N}, K={K}, F={F}]...")
        
        for i, current_id in enumerate(node_ids):
            # a. Node of Interest (Index 0 in the K dimension)
            # This is the feature vector of the main node (X_i)
            X_graph[i, 0, :] = X_matrix[i, :]

            # b. Find Neighbors using graph_df
            # Edges starting from the current node, sorted by distance (closest first)
            neighbors_df = graph_df[graph_df['Source_ID'] == current_id].sort_values(by='Distance')
            
            # Get the IDs of the closest neighbors, up to max_neighbors
            neighbor_ids = neighbors_df['Target_ID'].head(max_neighbors).tolist()
            
            neighbor_slot_idx = 1
            
            for neighbor_id in neighbor_ids:
                # Check if the neighbor is also present in the current split's data (node_ids)
                if neighbor_id in id_to_index:
                    neighbor_data_idx = id_to_index[neighbor_id]
                    
                    # Assign neighbor features to the next available slot (1 to max_neighbors)
                    X_graph[i, neighbor_slot_idx, :] = X_matrix[neighbor_data_idx, :]
                    neighbor_slot_idx += 1
                
                # Stop if we've filled all neighbor slots (K-1)
                if neighbor_slot_idx > max_neighbors:
                    break
                    
        # --- NEW IMPUTATION LOGIC: Fill remaining slots ---
        
        # Determine the feature vector to copy for the remaining empty slots:
        
        # Case 1: At least one neighbor was found (neighbor_slot_idx > 1). 
        # Copy the features of the nearest neighbor (slot 1).
        if neighbor_slot_idx > 1:
            imputation_features = X_graph[i, 1, :]
            
        # Case 2: No neighbors were found (neighbor_slot_idx is still 1). 
        # Copy the target node's features (slot 0).
        else:
            imputation_features = X_graph[i, 0, :]
        
        # Fill the remaining empty slots (from neighbor_slot_idx up to K-1)
        # K is the exclusive end index (e.g., K=6).
        # If neighbor_slot_idx=1, it fills slots 1, 2, 3, 4, 5.
        # If neighbor_slot_idx=4, it fills slots 4, 5.
        X_graph[i, neighbor_slot_idx:K, :] = imputation_features
        
        # --- END OF NEW LOGIC ---

        print(f"Final feature tensor shape: {X_graph.shape}")
        
        return X_graph, y, scaler

def export_to_shapefile(pwfdf_data, output_path='data/pwfdf_points.shp', crs='EPSG:4326'):
    geometries = []
    records = []
    
    for i in range(len(pwfdf_data.df)):
        entry = pwfdf_data.get(i)
        
        if crs == 'EPSG:4326':
            lon, lat = entry.coordinates_wgs84()
            point = Point(lon, lat)
        elif crs == 'EPSG:5070':
            x, y = entry.coordinates_5070()
            point = Point(x, y)
        else:
            # Use original UTM coordinates
            point = Point(entry.utm_x, entry.utm_y)
        
        geometries.append(point)
        records.append(entry.d)
    
    gdf = gpd.GeoDataFrame(records, geometry=geometries, crs=crs)
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gdf.to_file(output_path)
    print(f"Shapefile saved to: {output_path}")
    print(f"Total points: {len(gdf)}")
    print(f"CRS: {crs}")
    
    return gdf

features = [
    'UTM_X', 'UTM_Y', 
    'Fire_ID', 'Fire_SegID',
    'GaugeDist_m', 
    'StormDur_H', 'StormAccum_mm', 'StormAvgI_mm/h', 
    'Peak_I15_mm/h', 'Peak_I30_mm/h', 'Peak_I60_mm/h',
    'ContributingArea_km2', 
    'PropHM23', 'dNBR/1000', 'KF', 'Acc015_mm', 
    'Acc030_mm', 'Acc060_mm'
]

# Example usage:
if __name__ == "__main__":
    data = PWFDF_Data()
    X_train_full, y_train_full, scaler = data.prepare_data_usgs(features, split='Training')
    print(X_train_full[0][0])
    print(X_train_full[0][1])
    #prep_data('data/ofr20161106_appx-1.xlsx', 'Appendix1_ModelData', 'data/data.csv')
    #gdf_wgs84 = export_to_shapefile(pwfdf, 'data/pwfdf_wgs84.shp', crs='EPSG:4326')
    #gdf_albers = export_to_shapefile(pwfdf, 'data/pwfdf_albers.shp', crs='EPSG:5070')
    
    #print("\nDataFrame Info:")
    #print(gdf_wgs84.head())
    #print(f"\nBounds: {gdf_wgs84.total_bounds}")

