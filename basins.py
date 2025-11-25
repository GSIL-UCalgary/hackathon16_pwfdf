import json
import os
import ee
import py3dep
import requests
from rasterio.io import MemoryFile
from datetime import datetime, timedelta
import geopandas as gpd
from shapely.geometry import shape

# Initialize Earth Engine (run ee.Authenticate() first time)
ee.Initialize(project='ee-quinnledingham')

def download_watershed(out_file, geometry, crs='EPSG:4326'):
    """Download DEM for given geometry"""
    if not os.path.isfile(out_file):
        print(f"Downloading DEM to {out_file}...")
        
        # Convert geometry to shapely if needed
        if isinstance(geometry, dict):
            geom = shape(geometry)
        else:
            geom = geometry
            
        # Get bounds in the original CRS
        bounds = geom.bounds  # (minx, miny, maxx, maxy)
        print(f"Bounds (lon/lat): {bounds}")
        
        # Download DEM using the geometry directly
        dem = py3dep.get_dem(geom, resolution=30, crs=crs)
        
        # Reproject to EPSG:5070 (or keep original CRS)
        dem = dem.rio.reproject("EPSG:5070")
        dem.rio.to_raster(out_file)
        print(f"DEM saved to {out_file}")
    else:
        print(f"DEM already downloaded ({out_file})")

def get_satellite_image(geometry, start_date, end_date, out_file, crs='EPSG:4326'):
    """Download Landsat image for given geometry and date range"""
    if os.path.isfile(out_file):
        print(f"Image already downloaded ({out_file})")
        return
    
    print(f"Downloading Landsat image from {start_date} to {end_date}...")
    
    # Convert geometry to shapely if needed
    if isinstance(geometry, dict):
        geom = shape(geometry)
    else:
        geom = geometry
    
    bounds = geom.bounds  # (minx, miny, maxx, maxy)
    
    # Create Earth Engine geometry
    region = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]])
    
    # Use Landsat 7 or 8 depending on date
    image = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2") \
        .filterBounds(region) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUD_COVER', 30))
    
    bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']  # NIR, SWIR1, SWIR2
    rgb = image.median().select(bands).clip(region)
    
    url = rgb.getDownloadURL({
        'scale': 30,
        'region': region,
        'format': 'GEO_TIFF'
    })
    
    response = requests.get(url)
    with open(out_file, 'wb') as f:
        f.write(response.content)
    print(f"Image saved to {out_file}")

def process_fire_features(geojson_file, output_dir='./data/fire_data'):
    """Process all features in GeoJSON file and download data"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read GeoJSON file using geopandas to preserve CRS
    gdf = gpd.read_file(geojson_file)
    print(f"GeoJSON CRS: {gdf.crs}")
    print(f"Total features in file: {len(gdf)}\n")
    
    # Ensure we're working in WGS84 (EPSG:4326) for consistency
    if gdf.crs != 'EPSG:4326':
        print(f"Reprojecting from {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs('EPSG:4326')
    
    # Group by Fire_ID to organize files in folders
    grouped = gdf.groupby('Fire_ID')
    print(f"Found {len(grouped)} unique Fire_IDs\n")
    
    for fire_id, group_df in grouped:
        # Create folder for this Fire_ID
        safe_fire_id = str(fire_id).replace('/', '_').replace(' ', '_')
        fire_folder = os.path.join(output_dir, safe_fire_id)
        os.makedirs(fire_folder, exist_ok=True)
        
        # Get fire information from first row (should be same for all in group)
        first_row = group_df.iloc[0]
        fire_name = first_row.get('Fire Name', 'unknown')
        fire_start = first_row.get('FireStart')
        fire_end = first_row.get('FireEnd')
        
        print(f"\n{'='*60}")
        print(f"Fire ID: {fire_id} - {fire_name}")
        print(f"Number of basin polygons: {len(group_df)}")
        print(f"Fire dates: {fire_start} to {fire_end}")
        print(f"Output folder: {fire_folder}")
        
        # Process each basin polygon for this fire
        for idx, row in group_df.iterrows():
            geometry = row.geometry
            
            # Use Index or PtIndex to identify individual basins
            basin_id = row.get('Fire_SegID', idx)
            
            print(f"\n  Processing basin {basin_id}...")
            print(f"    Geometry type: {geometry.geom_type}")
            print(f"    Bounds: {geometry.bounds}")
            
            # Create filenames for this specific basin
            dem_file = os.path.join(fire_folder, f"{basin_id}_dem.tif")
            before_file = os.path.join(fire_folder, f"{basin_id}_before.tif")
            after_file = os.path.join(fire_folder, f"{basin_id}_after.tif")
            
            try:
                # Download DEM for this basin
                download_watershed(dem_file, geometry, crs='EPSG:4326')
                
                # Download before-fire image (60 days before fire start)
                if fire_start:
                    # Handle different date formats
                    fire_start_str = str(fire_start).replace('Z', '').split('T')[0]
                    fire_start_dt = datetime.fromisoformat(fire_start_str)
                    before_start = (fire_start_dt - timedelta(days=120)).strftime('%Y-%m-%d')
                    before_end = (fire_start_dt - timedelta(days=1)).strftime('%Y-%m-%d')
                    get_satellite_image(geometry, before_start, before_end, before_file)
                
                # Download after-fire image (60 days after fire end)
                if fire_end:
                    # Handle different date formats
                    fire_end_str = str(fire_end).replace('Z', '').split('T')[0]
                    fire_end_dt = datetime.fromisoformat(fire_end_str)
                    after_start = (fire_end_dt + timedelta(days=1)).strftime('%Y-%m-%d')
                    after_end = (fire_end_dt + timedelta(days=120)).strftime('%Y-%m-%d')
                    get_satellite_image(geometry, after_start, after_end, after_file)
                    
            except Exception as e:
                print(f"    Error processing basin {basin_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n{'='*60}")
    print(f"Processing complete! Files saved to {output_dir}/")

# Run the script
if __name__ == "__main__":
    # Replace with your GeoJSON file path
    geojson_file = './data/staley_basins.geojson'
    
    # Process all features
    process_fire_features(geojson_file)