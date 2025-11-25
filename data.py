import os
import sciencebasepy
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Proj, transform, Transformer

from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
import geopandas as gpd
from shapely.geometry import Point

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

class PWFDF_Data:
    path = 'data/ofr20161106_appx-1.xlsx'
    sheet_name = 'Appendix1_ModelData'

    def __init__(self):
        self.df = pd.read_excel(self.path, sheet_name=self.sheet_name)

    def get(self, i):
        return PWFDF_Entry(self.df.iloc[i].to_dict())
    
    def get_data_target(self):
        numerical_features = [
            'UTM_X', 'UTM_Y', 'GaugeDist_m', 'StormDur_H', 'StormAccum_mm',
            'StormAvgI_mm/h', 'Peak_I15_mm/h', 'Peak_I30_mm/h', 'Peak_I60_mm/h',
            'ContributingArea_km2', 'PropHM23', 'dNBR/1000', 'KF',
            'Acc015_mm', 'Acc030_mm', 'Acc060_mm'
        ]

        categorical_features = ['State', 'UTM_Zone']
        label_encoders = {}
        for col in categorical_features:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col + '_encoded'] = le.fit_transform(self.df[col].astype(str))
                label_encoders[col] = le

        self.df[numerical_features] = self.df[numerical_features].fillna(self.df[numerical_features].mean()) # handle missing values
        feature_cols = numerical_features + [col + '_encoded' for col in categorical_features if col in self.df.columns] # combine features
        
        data = self.df[feature_cols].values
        target = self.df['Response'].values

        #normalize
        scaler = StandardScaler() 
        data = scaler.fit_transform(data)

        return data, target

    
    def prepare_data(self, split='Training'):
        """
        Prepare data directly from PWFDF_Data using Database column
        Returns the full feature matrix and labels
        
        Args:
            pwfdf_data: PWFDF_Data object
            split: 'Training' or 'Test'
        """
        df = self.df
        
        # Filter by Database column
        df = df[df['Database'] == split].copy()
        
        # Get all numerical features
        numerical_features = [
            'UTM_X', 'UTM_Y', 'GaugeDist_m', 'StormDur_H', 'StormAccum_mm',
            'StormAvgI_mm/h', 'Peak_I15_mm/h', 'Peak_I30_mm/h', 'Peak_I60_mm/h',
            'ContributingArea_km2', 'PropHM23', 'dNBR/1000', 'KF',
            'Acc015_mm', 'Acc030_mm', 'Acc060_mm'
        ]
        
        X = df[numerical_features].values
        y = df['Response'].values

        # Remove rows with ANY missing values
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]

        # Fill missing values with 0 instead of removing rows
        #X = np.nan_to_num(X, nan=0.0)
        #y = np.nan_to_num(y, nan=0.0)
        
        #scaler = StandardScaler() 
        #X = scaler.fit_transform(X)

        return X, y

    def prepare_data_with_normalization(self, split='Training', normalize=True):
        """
        Prepare data with proper normalization
        """
        df = self.df
        
        # Filter by Database column
        df = df[df['Database'] == split].copy()
        
        # Get all numerical features
        numerical_features = [
            'UTM_X', 'UTM_Y', 'GaugeDist_m', 'StormDur_H', 'StormAccum_mm',
            'StormAvgI_mm/h', 'Peak_I15_mm/h', 'Peak_I30_mm/h', 'Peak_I60_mm/h',
            'ContributingArea_km2', 'PropHM23', 'dNBR/1000', 'KF',
            'Acc015_mm', 'Acc030_mm', 'Acc060_mm'
        ]
        
        X = df[numerical_features].values
        y = df['Response'].values

        # Remove rows with ANY missing values
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        if normalize:
            # Different normalization strategies for different feature types
            X_normalized = self._smart_normalize(X, numerical_features)
            return X_normalized, y
        
        return X, y

    def _smart_normalize(self, X, feature_names):
        """
        Apply different normalization strategies based on feature characteristics
        """
        feature_indices = {name: i for i, name in enumerate(feature_names)}
        X_normalized = X.copy().astype(np.float32)
        
        # Group 1: UTM coordinates - use RobustScaler (less sensitive to outliers)
        utm_features = ['UTM_X', 'UTM_Y']
        utm_indices = [feature_indices[f] for f in utm_features]
        if utm_indices:
            utm_scaler = RobustScaler()
            X_normalized[:, utm_indices] = utm_scaler.fit_transform(X[:, utm_indices])
        
        # Group 2: Distance and area features - StandardScaler
        spatial_features = ['GaugeDist_m', 'ContributingArea_km2']
        spatial_indices = [feature_indices[f] for f in spatial_features]
        if spatial_indices:
            spatial_scaler = StandardScaler()
            X_normalized[:, spatial_indices] = spatial_scaler.fit_transform(X[:, spatial_indices])
        
        # Group 3: Rainfall intensity and accumulation - StandardScaler
        rainfall_features = [
            'StormDur_H', 'StormAccum_mm', 'StormAvgI_mm/h', 
            'Peak_I15_mm/h', 'Peak_I30_mm/h', 'Peak_I60_mm/h',
            'Acc015_mm', 'Acc030_mm', 'Acc060_mm'
        ]
        rainfall_indices = [feature_indices[f] for f in rainfall_features]
        if rainfall_indices:
            rainfall_scaler = StandardScaler()
            X_normalized[:, rainfall_indices] = rainfall_scaler.fit_transform(X[:, rainfall_indices])
        
        # Group 4: Proportional features (already 0-1 range) - Minimal scaling
        proportional_features = ['PropHM23', 'KF']
        proportional_indices = [feature_indices[f] for f in proportional_features]
        if proportional_indices:
            # Just center them around 0
            prop_scaler = StandardScaler(with_std=False)  # Only center, don't scale
            X_normalized[:, proportional_indices] = prop_scaler.fit_transform(X[:, proportional_indices])
        
        # Group 5: dNBR - StandardScaler (already divided by 1000)
        dNBR_index = feature_indices['dNBR/1000']
        dNBR_scaler = StandardScaler()
        X_normalized[:, dNBR_index] = dNBR_scaler.fit_transform(X[:, dNBR_index:dNBR_index+1]).flatten()
        
        return X_normalized

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

# Example usage:
if __name__ == "__main__":
    pwfdf = PWFDF_Data()
    
    gdf_wgs84 = export_to_shapefile(pwfdf, 'data/pwfdf_wgs84.shp', crs='EPSG:4326')
    #gdf_albers = export_to_shapefile(pwfdf, 'data/pwfdf_albers.shp', crs='EPSG:5070')
    
    print("\nDataFrame Info:")
    print(gdf_wgs84.head())
    print(f"\nBounds: {gdf_wgs84.total_bounds}")