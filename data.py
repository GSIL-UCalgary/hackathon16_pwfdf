import pandas as pd
from pyproj import Proj, transform, Transformer
import elevation

import rasterio
from rasterio.plot import show
from rasterio.warp import reproject, Resampling, calculate_default_transform

import matplotlib.pyplot as plt
from pysheds.grid import Grid
import numpy as np
import requests
import ee
import geetools
from geetools import tools

ee.Authenticate()
ee.Initialize(project='ee-quinnledingham')

def plot_catchment(grid, clipped_catch):
    fig, ax = plt.subplots(figsize=(8,6))
    fig.patch.set_alpha(0)
    plt.grid('on', zorder=0)
    im = ax.imshow(np.where(clipped_catch, clipped_catch, np.nan), extent=grid.extent, zorder=1, cmap='Greys_r')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Delineated Catchment', size=14)

def plot_catchment_dist(grid, dist):
    fig, ax = plt.subplots(figsize=(8,6))
    fig.patch.set_alpha(0)
    plt.grid('on', zorder=0)
    im = ax.imshow(dist, extent=grid.extent, zorder=2, cmap='cubehelix_r')
    plt.colorbar(im, ax=ax, label='Distance to outlet (cells)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Flow Distance', size=14)

def delineate_watershed(dem_file, pour_point_coords):
    grid = Grid.from_raster(dem_file)
    dem = grid.read_raster(dem_file)

    pit_filled_dem = grid.fill_pits(dem)
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    inflated_dem = grid.resolve_flats(flooded_dem)

    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    flow_dir  = grid.flowdir(inflated_dem, dirmap=dirmap)
    flow_acc = grid.accumulation(flow_dir, dirmap=dirmap)

    threshold = 100
    branches = grid.extract_river_network(flow_dir, flow_acc > threshold, dirmap=dirmap)

    x, y = pour_point_coords
    x_snap, y_snap = grid.snap_to_mask(flow_acc > 10, pour_point_coords) # Snap pour point to high accumulation cell
    catch = grid.catchment(x=x_snap, y=y_snap, fdir=flow_dir, dirmap=dirmap, xytype='coordinate')

    dist = grid.distance_to_outlet(x=x_snap, y=y_snap, fdir=flow_dir, dirmap=dirmap, xytype='coordinate')

    # plot DEM
    print("PLOTTING DEM")
    fig, ax = plt.subplots(figsize=(8,8))
    fig.patch.set_alpha(0)
    plt.imshow(dem, extent=grid.extent, cmap='terrain', zorder=1)
    plt.colorbar(label='Elevation (m)')
    plt.grid(zorder=0)
    #im = ax.imshow(np.where(clipped_catch, clipped_catch, np.nan), extent=grid.extent, zorder=1, cmap='Greys_r')
    im = ax.imshow(dist, extent=grid.extent, zorder=2, cmap='cubehelix_r')
    plt.title('Digital elevation map', size=14)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    #print(f"x: {x}, y: {y}")
    plt.plot(x, y, 'ro', markersize=1, zorder=2)  # 'ro' = red circle
    plt.plot(x_snap, y_snap, 'go', markersize=1, zorder=2)  # 'ro' = red circle
    plt.tight_layout()
    plt.show()

def get_watershed_from_usgs_api(longitude: float, latitude: float):
    try:
        # USGS StreamStats API endpoint
        url_format = "https://test.streamstats.usgs.gov/streamstatsservices/watershed{0}?rcode={1}&xlocation={2}&ylocation={3}&crs={4}&includeparameters={5}&includeflowtypes={6}&includefeatures={7}&simplify={8}"
        api_url = url_format.format(
            '.geojson',          # {0} - format extension
            "CA",                # {1} - region code
            longitude,           # {2} - x location
            latitude,            # {3} - y location
            "4326",              # {4} - coordinate reference system
            str("true").lower(), # {5} - include parameters
            str("false").lower(),# {6} - include flow types
            str("true").lower(), # {7} - include features
            str("true").lower()  # {8} - simplify geometry
        )
        response = requests.get(api_url, timeout=120)
        response.raise_for_status()

        data = response.json()

        if 'featurecollection' in data and len(data['featurecollection']) > 0:
            watershed_features = data['featurecollection'][0]
            return watershed_features
        else:
            print("No watershed found for the given coordinates")
            return None

    except Exception as e:
        print(f"Error querying USGS API: {e}")
        return None

class PWFDF_Entry:
    def __init__(self, d):
        self.d = d
        self.utm_x = self.d['UTM_X']
        self.utm_y = self.d['UTM_Y']
        self.zone = self.d['UTM_Zone']

        utm_proj = Proj(proj='utm', zone=self.zone, ellps='WGS84', datum='WGS84', south=False)
        wgs84_proj = Proj(proj='latlong', ellps='WGS84', datum='WGS84')
        self.transformer = Transformer.from_proj(utm_proj, wgs84_proj, always_xy=True)

    def coordinates_wgs84(self):
        return self.transformer.transform(self.utm_x, self.utm_y)
    
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
    
def plot_dem(output_file, x, y):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_alpha(0)
    
    with rasterio.open(output_file) as src:
        target_crs = src.crs
        target_bounds = src.bounds
        target_shape = src.shape
        target_transform = src.transform
        show(src, ax=ax, cmap='terrain', zorder=1)

    plot_satellite_image(ax, '/home/quinn/pwfdf/data/satellite.tif', target_crs, target_bounds, target_shape, target_transform)

    plt.colorbar(ax.images[0], label='Elevation (m)')

    ax.plot(x, y, 'ro', markersize=10, zorder=3)
    plt.grid(zorder=0)
    plt.title('DEM with Satellite Overlay', size=14)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.show()

def plot_satellite_image(ax, file, target_crs, target_bounds, target_shape, target_transform):
    with rasterio.open(file) as src:
        raster = src.read([1, 2, 3])

        print(f"\nImage metadata:")
        print(f"  Shape: {raster.shape}")
        print(f"  CRS: {src.crs}")
        print(f"  Bounds: {src.bounds}")

        reprojected = np.zeros((3, target_shape[0], target_shape[1]), dtype=raster.dtype)
        for i in range(3):
            reproject(
                source=raster[i],
                destination=reprojected[i],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear
            )

        raster_plot = np.transpose(reprojected, (1, 2, 0))
        raster_plot = np.clip(raster_plot / 3000, 0, 1)
        extent = [target_bounds.left, target_bounds.right, target_bounds.bottom, target_bounds.top]
        ax.imshow(raster_plot, extent=extent, zorder=2, alpha=0.5)

def get_satellite_image(file, bounds, start_date, end_date):
    region = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]])

    image = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2") \
        .filterBounds(region) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUD_COVER', 20))
    
    bands = ['SR_B3', 'SR_B2', 'SR_B1', 'SR_B4']  # RGB
    rgb = image.median().select(bands).clip(region)
    url = rgb.getDownloadURL({
        'scale': 10,
        'region': region,
        'format': 'GEO_TIFF'
    })

    response = requests.get(url)
    with open(file, 'wb') as f:
        f.write(response.content)

from datetime import datetime, timedelta

def get_fire_images(fire_year, bounds):
    print("FIRE")
    fire_date = f"{fire_year}-07-15"
    fire_dt = datetime.strptime(fire_date, "%Y-%m-%d")

    pre_fire_end = fire_dt - timedelta(days=30)  
    pre_fire_start = pre_fire_end - timedelta(days=90)
    post_fire_start = fire_dt + timedelta(days=30)
    post_fire_end = post_fire_start + timedelta(days=150)

    get_satellite_image('/home/quinn/pwfdf/data/before.tif', bounds, pre_fire_start, pre_fire_end)
    get_satellite_image('/home/quinn/pwfdf/data/after.tif', bounds, post_fire_start, post_fire_end)


def main():
    data = PWFDF_Data()
    i = 2
    entry = data.get(i)

    print(f"Using entry: {i}, fire name: {entry['Fire Name']}, seg id: {entry['Fire_SegID']}")
    bounds = entry.bounds(5)
    output_file = '/home/quinn/pwfdf/data/dem.tif'
    elevation.clip(bounds=bounds, output=output_file, product='SRTM1')

    #get_satellite_image('/home/quinn/pwfdf/data/satellite.tif', bounds)
    get_fire_images(entry['Year'], bounds)

    #x, y = entry.coordinates_wgs84()
    #plot_dem(output_file, x, y)

    #delineate_watershed(output_file, entry.coordinates_wgs84())

if __name__ == '__main__':
    main()