import os
import glob
import rasterio
import traceback
import numpy as np
from tqdm import tqdm
import multiprocessing
from osgeo import gdal
from rasterio.mask import mask
from rasterio.features import shapes
from rasterio.warp import transform_geom
from shapely.geometry import box, shape, MultiPolygon
# from shapely.errors import ShapelyDeprecationWarning
import warnings

# warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
gdal.SetConfigOption("GDAL_TIFF_INTERAL_MASK", "YES")


def histogram_match(input_img, ref_img, nodata_val=0):
    """
    Match the input image to the reference image using histogram equalisation.
    No-data values are ignored during the histogram and cumulative distribution function calculation.
    """

    # Initialise result with zero, which is our no-data value
    out_img = np.zeros(input_img.shape)

    # Match histograms independently per band
    for band in range(input_img.shape[0]):

        # Convert source and ref to numpy masked arrays to handle no-data values correctly
        input_band = np.ma.masked_array(input_img[band, ...], mask=(input_img[band, ...] == nodata_val))
        ref_img[np.isnan(ref_img)] = nodata_val
        ref_band = np.ma.masked_array(ref_img[band, ...], mask=(ref_img[band, ...] == nodata_val))

        # Get unique pixel values, their counts and indices
        input_values, input_idxs, input_counts = np.unique(input_band.ravel(), return_counts=True, return_inverse=True)
        ref_values, ref_counts = np.unique(ref_band.ravel(), return_counts=True)

        # Remove counts/values of no-data pixels
        input_counts = input_counts[~input_values.mask]
        ref_counts = ref_counts[~ref_values.mask]
        ref_values = ref_values[~ref_values.mask]

        if len(ref_counts) == 0:
            raise ValueError("Reference basemap has only nodata for this scene")

        # Get cumulative distribution functions for input and ref images, normalised to total number of pixels
        input_cdf = np.cumsum(input_counts)
        input_cdf = input_cdf / input_cdf[-1]
        ref_cdf = np.cumsum(ref_counts)
        ref_cdf = ref_cdf / ref_cdf[-1]

        # Linearly interpolate ref values between input and ref cumulative distribution functions
        interpolated_vals = np.interp(input_cdf, ref_cdf, ref_values)
        interpolated_vals = np.append(interpolated_vals, nodata_val)  # no-data values are in the last bin of input_idxs

        # Map interpolated values to pixels and reshape back to band shape
        out_img[band, ...] = interpolated_vals[input_idxs].reshape(input_band.shape)

    return out_img


def process_scene(params):
    """
    Process one mosaic scene.
    Search for a reference basemap containing this scene, then cut out the corresponding scene in the basemap and
    use it as the reference for histogram equalisation. The new histogram matched scene is written to a temp file.
    !! For scenes that intersect multiple basemaps, only the first basemap is currently used. In rare cases this could
    be a problem, so we save these to a log file.
    """
    scene_fp, basemap_fp, scene_pattern, temp_dir, dstSRS = params
    scene_name = os.path.basename(scene_fp).split("scene_fp")[0]
    order = scene_fp.split("/")[-3]
    error_msg = ""
    out_fp = os.path.join(temp_dir, f"{scene_name}_{order}_histmatched_reproj.tif")
    # if os.path.exists(out_fp):
    #     return

    try:
        with rasterio.open(scene_fp) as scene:

            scene_profile = scene.profile
            scene_colinterp = scene.colorinterp

            # Get cutline of valid scene data by polygonising the nodata mask
            valid_scene_bounds = []

            for coordinates, value in shapes(scene.read_masks(), transform=scene.transform):
                if value != 0:
                    # import ipdb; ipdb.set_trace()
                    # Create shape and transform to target CRS
                    # valid_scene_bounds.append(shape(transform_geom(scene.crs, "EPSG:4326", coordinates)))
                    valid_scene_bounds.append(shape(transform_geom(scene.crs, dstSRS if dstSRS is not None else scene.crs, coordinates)))
            scene_cutline = MultiPolygon(valid_scene_bounds)

            # Ensure scene is covered by basemap bounds
            with rasterio.open(basemap_fp) as bm:
                basemap_bounds = (box(*bm.bounds))
                
            if not basemap_bounds.intersects(scene_cutline):
                msg = f"Scene cutline not in basemap bounds for basemap {os.path.basename(basemap_fp)}"
                raise Exception(msg)

            try:
                # Extract scene from basemap
                # import ipdb; ipdb.set_trace()
                with rasterio.open(basemap_fp) as basemap:
                    basemap_scene, _ = mask(basemap, scene_cutline, all_touched=True, crop=True)

                # Match histogram of scene to basemap extract histogram
                transformed_scene = histogram_match(scene.read(), basemap_scene, 0)

            except ValueError:
                error_msg = f"\n!! Scene not covered - using basemap average to match scene {scene_fp}"

                # If scene is nodata in basemap, try matching with the histogram of the entire basemap
                with rasterio.open(basemap_fp) as basemap:
                    transformed_scene = histogram_match(scene.read(), basemap.read(), 0)

        # Write histogram matched scene to memory raster
        # import ipdb; ipdb.set_trace()    # file not found error: need a specific path
        matched_scene_fp = f"/vsimem/{scene_name}_WHISTOF_{os.path.basename(basemap_fp)}.tif"
        # matched_scene_fp = f"/home/rscph/siyu/temp_dir/{scene_name}_WHISTOF_{os.path.basename(basemap_fp)}.tif"
        with rasterio.open(matched_scene_fp, 'w', **scene_profile) as out:
            out.write(transformed_scene.astype(np.uint16))
            out.colorinterp = scene_colinterp
        # Re-project to output CRS in temp folder
        # gdal.Warp(out_fp, matched_scene_fp)
        gdal.Warp(out_fp, matched_scene_fp, dstSRS=dstSRS if dstSRS is not None else scene.crs, xRes=3, yRes=3, resampleAlg=gdal.GRA_CubicSpline, setColorInterpretation=True) # dstSRS="EPSG:4326",
        # os.system(f'rm {matched_scene_fp}')

    except Exception as err:

        # import ipdb; ipdb.set_trace()

        traceback.print_exc()
        print(f"\nWarning! Error on matching scene {scene_fp}: ", err)
        error_msg = str(err)

        # For scenes where matching failed, instead of having gaps we include the original scene, only reprojected
        out_fp = os.path.join(temp_dir, f"{scene_name}_{order}_reproj.tif")
        gdal.Warp(out_fp, scene_fp, dstSRS=dstSRS if dstSRS is not None else scene.crs, xRes=3, yRes=3, resampleAlg=gdal.GRA_CubicSpline, setColorInterpretation=True)
        # gdal.Warp(out_fp, scene_fp) # dstSRS="EPSG:4326",
        # os.system(f'rm {matched_scene_fp}')

    return out_fp, error_msg


def gdal_progress_callback(complete, _, data):
    if data:
        data.update(int(complete * 100) - data.n)
        if complete == 1:
            data.close()
    return 1


def create_histmatch_mosaic(raw_scenes_dir, output_dir, ref_basemaps_dir, scene_pattern, rename=False, overwrite=True,
                            temp_dir="temp", num_threads=32, max_gdal_cache_gb=16, basemap_filetype=".tif", log=print, 
                            scenes_list=None, ref_basemap_file=None, dstSRS=None):

    gdal.SetCacheMax(int(max_gdal_cache_gb * 1e9))
    gdal.SetConfigOption('CPL_LOG', '/dev/null')
    gdal.UseExceptions()

    tile_name = os.path.basename(raw_scenes_dir)
    out_path = os.path.join(output_dir, f"{tile_name}_composite_lshm.tif")  # jp2
    # log(f"\nCreating mosaic for tile {tile_name}\n")

    try:
        # Check if mosaic already exists
        if os.path.exists(out_path):
            if not overwrite and not rename:
                raise ValueError(f"!! rename=False and overwrite=False but mosaic already exists at {tile_name}")
            elif rename:
                os.rename(out_path, out_path.replace(".jp2", "_previous.jp2"))

        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
        
        # if direct input scenes, ignore the finding from raw scene dir
        if scenes_list:
            scene_fps = scenes_list

        # Get planet scene file paths
        else:
            scene_fps = []
            for root, folders, files in os.walk(raw_scenes_dir):
                for file in [f for f in files if scene_pattern in f]:
                    scene_fp = os.path.join(root, file)
                    try:
                        # try to open the scene to ensure it's not corrupt
                        rasterio.open(scene_fp)
                        scene_fps.append(scene_fp)
                    except rasterio.RasterioIOError:
                        log(f"Warning! Skipping corrupt scene {scene_fp}")
            scene_fps = sorted(scene_fps)


        # Check if no scenes available
        if len(scene_fps) == 0:
            raise ValueError(f"!! No raw scenes found for tile {tile_name}")

        if ref_basemap_file:
            basemap_fp = ref_basemap_file
        # Get basemap for this tile
        else:
            try:
                tile = "_".join(raw_scenes_dir.split("_")[-6:-4])
                basemap_fp = glob.glob(fr"{ref_basemaps_dir.rstrip('/')}/**/*{tile}*{basemap_filetype}", recursive=True)[0]
            except Exception:
                raise ValueError(f"!! No basemap found for tile {tile_name}") from None

        # Start worker pool to process scenes in parallel
        processed_scene_fps = []
        params = [[scene_fp, basemap_fp, scene_pattern, temp_dir, dstSRS] for scene_fp in scene_fps]
        n_success, n_missing_basemap, n_other_error = 0, 0, 0

        # import ipdb; ipdb.set_trace()
        
        # for i in params:
        #     process_scene(i)

        multiprocessing.set_start_method("spawn", force=True)
        with multiprocessing.Pool(processes=num_threads) as pool:

            # Wrap process enumeration in tqdm for a total progress bar
            with tqdm(total=len(scene_fps), desc=f"Matching scenes for {tile_name}", position=0, leave=True) as pb:
                for _, result in enumerate(pool.imap_unordered(process_scene, params, chunksize=1)):
                    pb.update()
                    if result:
                        out_fp, err_msg = result
                        processed_scene_fps.append(out_fp)
                        if err_msg == "":
                            n_success += 1
                        elif ("Scene cutline not in" in err_msg) or ("Scene not covered" in err_msg):
                            n_missing_basemap += 1
                        else:
                            n_other_error += 1

        # # Check for missing transformed scenes
        # if len(processed_scene_fps) != len(scene_fps):
        #     log(f"Warning! Only {len(processed_scene_fps)}/{len(scene_fps)} scenes processed for tile {tile_name}")

        # # Merge all scenes into one big mosaic
        # gdal.BuildVRT(f"/vsimem/{tile_name}_merged.vrt", sorted(processed_scene_fps))

        # options = dict(
        #     format="GTiff",
        #     callback=gdal_progress_callback,
        #     callback_data=tqdm(total=100, position=0, leave=True, desc=f"Merging mosaic for {tile_name}")
        # )
        # gdal.Translate(out_path, f"/vsimem/{tile_name}_merged.vrt", **options)


        # options = dict(
        #     format="JP2OpenJPEG",
        #     creationOptions=["QUALITY=80"],
        #     callback=gdal_progress_callback,
        #     callback_data=tqdm(total=100, position=0, leave=True, desc=f"Merging mosaic for {tile_name}")
        # )
        # gdal.Translate(out_path, f"/vsimem/{tile_name}_merged.vrt", **options)

        # # Clean up temp files
        # for fp in processed_scene_fps:
        #     if os.path.exists(fp):
        #         os.remove(fp)

        return out_path, n_missing_basemap, n_other_error

    except Exception as err:
        # If mosaic creation failed and we had a previous mosaic, revert to that one
        if os.path.exists(out_path.replace(".jp2", "_previous.jp2")) and not os.path.exists(out_path):
            os.rename(out_path.replace(".jp2", "_previous.jp2"), out_path)

        raise Exception(f"Error on creating mosaic for tile {raw_scenes_dir}: {err}")
