import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import bounds, from_bounds, Window
from tqdm import tqdm
from osgeo import gdal
import os
from itertools import product
from collections import Counter
from tqdm.contrib.concurrent import process_map
import math
import warnings

warnings.filterwarnings('ignore')
gdal.UseExceptions()


def gdal_progress_callback(complete, message, data):
    if data:
        data.update(int(complete * 100) - data.n)
        if complete == 1:
            data.close()
    return 1


def get_pixelwise_mean(yearly_raster_arrays, yearly_mask_arrays, threshold, config):
    """Compute the per-pixel mean across multiple years with a given threshold."""
    min_height = min(arr.shape[0] for arr in yearly_raster_arrays + yearly_mask_arrays)
    min_width = min(arr.shape[1] for arr in yearly_raster_arrays + yearly_mask_arrays)

    stacked = np.zeros((len(yearly_raster_arrays), min_height, min_width))

    for i in range(len(yearly_raster_arrays)):
        raster_array = yearly_raster_arrays[i][:min_height, :min_width]
        mask_array = yearly_mask_arrays[i][:min_height, :min_width]
        stacked[i, ...] = np.where(mask_array != 0, raster_array, config.no_data_value)

    mean_data = np.nanmean(stacked, axis=0)
    observations = np.sum(~np.isnan(stacked), axis=0)

    presence = mean_data >= threshold
    valid_pixels = observations > config.min_obs

    return valid_pixels


def get_chunk_windows(raster_fps, mask_fp, output_dir, config):
    """Divide raster into smaller windows for parallel processing."""
    n_rows, n_cols = config.grid_size
    params = []
    with rasterio.open(raster_fps[0]) as raster:
        chunk_width, chunk_height = math.ceil(raster.width / n_cols), math.ceil(raster.height / n_rows)
        for i, j in product(range(n_rows), range(n_cols)):
            chunk_bounds = bounds(Window(chunk_width * j, chunk_height * i, chunk_width, chunk_height), raster.transform)
            params.append([raster_fps, mask_fp, chunk_bounds,
                           f"{output_dir}/{os.path.basename(raster_fps[0])}_{chunk_width * j}_{chunk_height * i}.tif"])
    return params


def process_chunk(params):
    """Process a single raster chunk."""
    raster_fps, mask_fps, window_bounds, out_fp, threshold, config = params
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

    out_array = get_pixelwise_mean(yearly_raster_arrays, yearly_mask_arrays, threshold, config)
    out_array = np.nan_to_num(out_array, nan=0).astype(np.uint8)

    profile.update({
        "driver": "GTiff",
        "compress": "LZW",
        "transform": window_transform,
        "height": out_array.shape[0],
        "width": out_array.shape[1],
        "dtype": np.uint8,
    })

    with rasterio.open(out_fp, "w", **profile) as dst:
        dst.write(out_array, 1)
    
    return out_fp


def process_raster_parallel_windowed(raster_fps, mask_fps, output_fp, threshold, config):
    """Process raster files in parallel, split into chunks."""
    params = get_chunk_windows(raster_fps, mask_fps, os.path.dirname(output_fp), config)
    params = [(p[0], p[1], p[2], p[3], threshold, config) for p in params]
    
    chunk_fps = process_map(process_chunk, params, desc=f"Processing chunks (Threshold: {threshold})",
                            max_workers=config.num_workers)

    gdal.BuildVRT(f"/vsimem/merged.vrt", chunk_fps)
    options = dict(
        creationOptions=["TILED=YES", "BIGTIFF=IF_SAFER", "COMPRESS=LZW", "NUM_THREADS=ALL_CPUS"],
        callback=gdal_progress_callback, callback_data=tqdm(desc=f"Merging (Threshold: {threshold})", total=100)
    )
    gdal.Translate(output_fp, f"/vsimem/merged.vrt", format="GTiff", **options)

    [os.remove(f) for f in chunk_fps if os.path.exists(f)]


def aggregate_all(config):
    """Aggregate multi-year predictions into a single observation file for each probability threshold."""
    print("\nStarting aggregation for multiple years...\n")

    # Load grid data
    grid = gpd.read_file(config.grid_file)
    file_ids = sorted(grid['id'].tolist())

    for threshold in config.thresholds:
        prob_folder = os.path.join(config.predictions_base_dir, f"prob{int(threshold * 100):02d}/rasters")
        os.makedirs(prob_folder, exist_ok=True)

        print(f"\nProcessing for threshold {threshold}, saving to {prob_folder}\n")

        for file_id in tqdm(file_ids, desc=f"Processing file IDs (Threshold {threshold})"):
            pred_fps, mask_fps = [], []

            for year in config.prediction_years:
                pred_folder = f"{config.predictions_base_dir}/{year}_{config.run_name}/rasters"
                mask_folder = f"/isipd/projects/p_planetdw/data/dw_detection/PlanetScope/images/{year}"

                pred_fp = os.path.join(pred_folder, f"det_{year}_{file_id}_with_dem.tif")
                mask_fp = os.path.join(mask_folder, f"ps_PSScene4Band_{year}_{file_id}_hist_composite.jp2")

                if os.path.exists(pred_fp) and os.path.exists(mask_fp):
                    pred_fps.append(pred_fp)
                    mask_fps.append(mask_fp)

            if not pred_fps:
                print(f"Skipping file ID {file_id}: No valid rasters found.")
                continue

            output_fp = os.path.join(prob_folder, f"observations_valid_{file_id}.tif")
            if os.path.exists(output_fp):
                print(f"Skipping file ID {file_id}: Output already exists.")
                continue

            process_raster_parallel_windowed(pred_fps, mask_fps, output_fp, threshold, config)

    print("\nFinished processing all probability thresholds\n")
