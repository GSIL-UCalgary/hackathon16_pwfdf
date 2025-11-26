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

    def prepare_data_usgs(self, features, split='Training', duration='15min', scaler=None):
        df = self.df
        df = df[df['Database'] == split].copy()

        #        
        # Creating the USGS MASK
        #
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

        X = df[features].values
        y = df['Response'].values


        if split == 'Test':
            X = X[mask]
            y = y[mask]

        # Normalize
        if True:
            to_skip = ['UTM_X', 'UTM_Y', 'PropHM23', 'dNBR/1000', 'KF', 'Acc015_mm', 'Acc030_mm', 'Acc060_mm']
            present = [f for f in to_skip if f in features]
            skip_indices = [features.index(f) for f in present]
            normalize_indices = [i for i in range(len(features)) if i not in skip_indices]
            
            if len(normalize_indices) != 0:
                if scaler is None:
                    scaler = StandardScaler()
                    X[:, normalize_indices] = scaler.fit_transform(X[:, normalize_indices])
                    print(f"\nNormalized {len(normalize_indices)} features (fitted new scaler)")
                else:
                    # Use provided scaler (for test set)
                    X[:, normalize_indices] = scaler.transform(X[:, normalize_indices])
                    print(f"\nNormalized {len(normalize_indices)} features (using provided scaler)")
                
                return X, y, scaler
        
        return X, y, None

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
