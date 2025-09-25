import pandas as pd
from pyproj import Proj, transform, Transformer
import elevation
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from pysheds.grid import Grid
import numpy as np

def utm_to_geographic_bounds(utm_x, utm_y, utm_zone, hemisphere='north', buffer_km=5, analysis_type='fire'):

    # Suggested buffer sizes based on analysis type
    buffer_suggestions = {
        'fire': {
            'small': 2,      # Small fire < 1000 acres
            'medium': 5,     # Medium fire 1000-10000 acres
            'large': 10,     # Large fire > 10000 acres
            'mega': 25       # Mega fire > 100000 acres
        },
        'small_basin': 5,    # < 100 km²
        'large_basin': 15,   # > 100 km²
        'watershed': 25,     # Full watershed analysis
        'erosion': 3,        # Post-fire erosion modeling
        'hydrology': 10      # Hydrological modeling
    }

    # Create UTM projection
    if hemisphere.lower() == 'north':
        utm_proj = Proj(proj='utm', zone=utm_zone, ellps='WGS84', datum='WGS84')
    else:
        utm_proj = Proj(proj='utm', zone=utm_zone, ellps='WGS84', datum='WGS84', south=True)

    wgs84_proj = Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    transformer = Transformer.from_proj(utm_proj, wgs84_proj, always_xy=True)
    buffer_m = buffer_km * 1000 # Convert buffer from km to meters

    # Calculate bounds in UTM
    utm_west = utm_x - buffer_m
    utm_east = utm_x + buffer_m
    utm_south = utm_y - buffer_m
    utm_north = utm_y + buffer_m

    # Convert corners to geographic coordinates
    west, south = transformer.transform(utm_west, utm_south)
    east, north = transformer.transform(utm_east, utm_north)

    # Get suggested buffer
    if analysis_type in buffer_suggestions:
        if isinstance(buffer_suggestions[analysis_type], dict):
            suggested = buffer_suggestions[analysis_type]['medium']  # default to medium
        else:
            suggested = buffer_suggestions[analysis_type]
    else:
        suggested = 5  # default

    return (west, south, east, north), suggested

def delineate_watershed(dem_file, pour_point_coords):
    grid = Grid.from_raster(dem_file, data_name='dem')
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
    x_snap, y_snap = grid.snap_to_mask(flow_acc > 1000, (x, y)) # Snap pour point to high accumulation cell
    catch = grid.catchment(x=x_snap, y=y_snap, fdir=flow_dir, dirmap=dirmap, xytype='coordinate')

    grid.clip_to(catch)
    clipped_catch = grid.view(catch)

    fig, ax = plt.subplots(figsize=(8,6))
    fig.patch.set_alpha(0)
    plt.imshow(dem, extent=grid.extent, cmap='terrain', zorder=1)
    plt.colorbar(label='Elevation (m)')
    plt.grid(zorder=0)
    plt.title('Digital elevation map', size=14)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()

    # Plot the catchment
    fig, ax = plt.subplots(figsize=(8,6))
    fig.patch.set_alpha(0)
    plt.grid('on', zorder=0)
    im = ax.imshow(np.where(clipped_catch, clipped_catch, np.nan), extent=grid.extent, zorder=1, cmap='Greys_r')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Delineated Catchment', size=14)

    dist = grid.distance_to_outlet(x=x_snap, y=y_snap, fdir=flow_dir, dirmap=dirmap, xytype='coordinate')

    fig, ax = plt.subplots(figsize=(8,6))
    fig.patch.set_alpha(0)
    plt.grid('on', zorder=0)
    im = ax.imshow(dist, extent=grid.extent, zorder=2, cmap='cubehelix_r')
    plt.colorbar(im, ax=ax, label='Distance to outlet (cells)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Flow Distance', size=14)

def main():
    data_path = 'data/ofr20161106_appx-1.xlsx'

    df = pd.read_excel(data_path, sheet_name='Appendix1_ModelData')
    print(df.head())

    x = df['UTM_X'].astype(float)
    y = df['UTM_Y'].astype(float)
    zone = df['UTM_Zone'].astype(int)

    print(f"x: {x[0]}, y: {y[0]}, zone: {zone[0]}")
    bounds, suggested_buffer = utm_to_geographic_bounds(x[0], y[0], zone[0], 'north', 5, 'fire')

    output_file = '/home/quinn/pwfdf/data/dem.tif'
    elevation.clip(bounds=bounds, output=output_file, product='SRTM1')

    #with rasterio.open(output_file) as src:
    #    show(src, cmap='terrain', title=f'Fire/Basin DEM - {5}km buffer')
    delineate_watershed(output_file, (bounds[2], bounds[3]))

if __name__ == '__main__':
    main()
