import geopandas as gpd
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import bounds, Window, from_bounds
from tqdm import tqdm
from osgeo import gdal
from collections import Counter
import os
from itertools import product
from tqdm.contrib.concurrent import process_map
import math
import warnings

warnings.filterwarnings('ignore')
gdal.UseExceptions()

# Constants
threshold = 0.6
min_obs = 1
no_data_value = np.nan

def gdal_progress_callback(complete, message, data):
    """Callback function to show progress during GDAL operations."""
    if data:
        data.update(int(complete * 100) - data.n)
        if complete == 1:
            data.close()
    return 1

def reproject_to_common_crs(file_path, target_crs):
    """Reproject a raster file to the target CRS."""
    with rasterio.open(file_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(file_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest
                )
    print(f"Reprojected {file_path} to match CRS {target_crs}.")

def get_pixelwise_mean(yearly_raster_arrays, yearly_mask_arrays, threshold=0.5, nodata_val=np.nan, min_obs = 0):
    # Find the minimum shape across all raster and mask arrays
    min_height = min(arr.shape[0] for arr in yearly_raster_arrays + yearly_mask_arrays)
    min_width = min(arr.shape[1] for arr in yearly_raster_arrays + yearly_mask_arrays)
    
    # Initialize a stacked array with the shape adjusted to the minimum common size
    stacked = np.zeros((len(yearly_raster_arrays), min_height, min_width))
    
    for i in range(len(yearly_raster_arrays)):
        # Crop arrays to the minimum common shape
        raster_array = yearly_raster_arrays[i][:min_height, :min_width]
        mask_array = yearly_mask_arrays[i][:min_height, :min_width]
        
        # Apply mask and assign to the stacked array
        stacked[i, ...] = np.where(mask_array != 0, raster_array, nodata_val)
    
    # Compute the mean of the stacked array
    mean_data = np.nanmean(stacked, axis=0)
    observations = np.sum(~np.isnan(stacked), axis=0)  # Per-pixel count of observations

    presence = mean_data >= threshold
    valid_pixels = observations > min_obs  # Boolean mask of valid pixels

    # Apply the threshold to generate the output data
    output_data = valid_pixels #np.logical_and(presence, valid_pixels)
    
    return output_data


def get_chunk_windows(raster_fps, mask_fp, output_dir, grid_size):
    n_rows, n_cols = grid_size
    params = []
    with rasterio.open(raster_fps[0]) as raster:
        chunk_width, chunk_height = math.ceil(raster.width / n_cols), math.ceil(raster.height / n_rows)
        for i, j in product(range(n_rows), range(n_cols)):
            chunk_bounds = bounds(Window(chunk_width * j, chunk_height * i, chunk_width, chunk_height), raster.transform)
            params.append([raster_fps, mask_fp, chunk_bounds,
                           f"{output_dir}/{os.path.basename(raster_fps[0])}_{chunk_width * j}_{chunk_height * i}.tif"])
    return params

def process_chunk(params):
    raster_fps, mask_fps, window_bounds, out_fp = params
    yearly_raster_arrays, yearly_mask_arrays = [], []
    for raster_fp in raster_fps:
        with rasterio.open(raster_fp) as src:
            profile = src.profile
            window = from_bounds(*window_bounds, transform=src.transform)
            raster_array = src.read(1, window=window)
            yearly_raster_arrays.append(raster_array)
            window_transform = src.window_transform(window)

    for mask_fp in mask_fps:
        with rasterio.open(mask_fp) as src:
            window = from_bounds(*window_bounds, transform=src.transform)
            mask_array = src.read_masks(1, window=window, out_shape=raster_array.shape)
            yearly_mask_arrays.append(mask_array)


    # Calculate pixel-wise mean and apply threshold
    out_array = get_pixelwise_mean(yearly_raster_arrays, yearly_mask_arrays, threshold=threshold, min_obs=min_obs)

    # Convert output data to uint8 and replace NaNs with 0
    out_array = np.nan_to_num(out_array, nan=0).astype(np.uint8)

    profile.update({
        "driver": "GTiff",
        "compress": "LZW",
        "transform": window_transform,

        "height": out_array.shape[0],
        "width": out_array.shape[1],
        "dtype": np.uint8,   # Set dtype to uint8
    })

    with rasterio.open(out_fp, "w", **profile) as dst:
        dst.write(out_array, 1)
    return out_fp

def process_raster_parallel_windowed(raster_fps, mask_fps, output_fp, grid_size=(4, 4), num_workers=16):
    params = get_chunk_windows(raster_fps, mask_fps, os.path.dirname(output_fp), grid_size)
    chunk_fps = process_map(process_chunk, params, desc="Processing chunks", max_workers=num_workers)
    gdal.BuildVRT(f"/vsimem/merged.vrt", chunk_fps)
    options = dict(
        creationOptions=["TILED=YES", "BIGTIFF=IF_SAFER", "COMPRESS=LZW", "NUM_THREADS=ALL_CPUS"],
        callback=gdal_progress_callback, callback_data=tqdm(desc="Merging", total=100)
    )
    gdal.Translate(output_fp, f"/vsimem/merged.vrt", format="GTiff", **options)
    [os.remove(f) for f in chunk_fps if os.path.exists(f)]

# Load grid data and file IDs
grid = gpd.read_file("/isipd/projects/p_planetdw/data/auxilliary/carl_na_new (copy).gpkg")
file_ids = sorted(grid['id'].tolist())

for file_id in tqdm(file_ids, desc="Processing file IDs"):
    years = ['2019', '2021', '2023']
    pred_fps, mask_fps = [], []

    for year in years:
        folder_path = f'/isipd/projects/p_planetdw/data/dw_detection/PlanetScope/results/{year}/rasters/det_{year}_{file_id}_with_dem.tif'
        mask_path = f'/isipd/projects/p_planetdw/data/dw_detection/PlanetScope/images/{year}/ps_PSScene4Band_{year}_{file_id}_hist_composite.jp2'
        if os.path.exists(folder_path) and os.path.exists(mask_path):
            pred_fps.append(folder_path)
            mask_fps.append(mask_path)

    # Skip file if no valid rasters are found
    if not pred_fps:
        print(f"Skipping file ID {file_id}: No valid rasters found.")
        continue

    # Check if the output already exists, and skip if it does
    output_fp = f'/isipd/projects/p_planetdw/data/dw_detection/PlanetScope/results/valid_obs/rasters/observations_valid_{file_id}.tif'
    if os.path.exists(output_fp):
        print(f"Skipping file ID {file_id}: Output already exists.")
        continue

    # Check the shapes of the files and remove mismatches (iterating in reverse to avoid out-of-range errors)
    shapes = [rasterio.open(fp).shape for fp in pred_fps]
    shape_counts = Counter(shapes)
    most_common_shape = shape_counts.most_common(1)[0][0]

    for i in range(len(shapes) - 1, -1, -1):  # Reverse iteration over the list
        if shapes[i] != most_common_shape:
            pred_fps.pop(i)
            mask_fps.pop(i)

    # Reproject rasters and masks to a common CRS
    crss = [rasterio.open(fp).crs for fp in pred_fps + mask_fps]
    crs_counts = Counter(crss)
    most_common_crs = crs_counts.most_common(1)[0][0]

    for i, crs in enumerate(crss):
        if crs != most_common_crs:
            reproject_to_common_crs(pred_fps[i] if i < len(pred_fps) else mask_fps[i - len(pred_fps)], most_common_crs)

    # Process the raster files in parallel
    process_raster_parallel_windowed(pred_fps, mask_fps, output_fp, grid_size=(8, 8), num_workers=16)

print('Finished processing all files')
