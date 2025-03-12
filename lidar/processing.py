import os
import time
import json
import glob
import pdal
import laspy
import numpy as np
import subprocess
from datetime import timedelta
from tqdm import tqdm
from scipy.spatial import KDTree
import rasterio
import shutil 


def check_resolution(las_file, resolution, method="sampling", num_samples=10000):
    """
    Checks if the DSM resolution is appropriate based on point cloud density.

    Parameters:
        las_file (str): Path to the LAS/LAZ file.
        resolution (float): Desired DSM resolution.
        method (str): "sampling" (nearest neighbor) or "density" (Poisson estimate).
        num_samples (int): Number of random samples for nearest neighbor method.

    Returns:
        float: Estimated average point spacing.
        bool: Whether the resolution is appropriate.
    """
    with laspy.open(las_file) as file:
        point_cloud = file.read()
        points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).T

    if len(points) == 0:
        raise ValueError(f"Point cloud {las_file} is empty.")

    if method == "sampling":
        num_samples = min(num_samples, len(points))
        sampled_points = points[np.random.choice(len(points), num_samples, replace=False)]
        tree = KDTree(points)
        distances, _ = tree.query(sampled_points, k=2)
        avg_distance = np.mean(distances[:, 1])  # Ignore self-distance

    elif method == "density":
        bbox_volume = np.prod(points.max(axis=0) - points.min(axis=0))
        density = len(points) / bbox_volume if bbox_volume > 0 else float('inf')
        avg_distance = (1 / density) ** (1 / 3)

    else:
        raise ValueError("Invalid method. Choose 'sampling' or 'density'.")

    #if avg_distance > resolution:
        #print(f"Warning: DSM resolution ({resolution}m) is finer than average point spacing ({avg_distance:.3f}m). "
              #f"This may cause interpolation gaps.")

    return avg_distance, avg_distance <= resolution


def check_classification_exists(las_file):
    """
    Checks if a LAS/LAZ file contains classification information.

    Returns:
        bool: True if classification exists, False otherwise.
    """
    try:
        with laspy.open(las_file) as file:
            return "classification" in file.header.point_format.dimension_names
    except Exception as e:
        print(f"Error checking classification for {las_file}: {e}")
        return False  # Assume no classification if an error occurs


def get_bounding_box(las_file):
    """Retrieves the bounding box (extent) of a LAS/LAZ file."""
    with laspy.open(las_file) as file:
        point_cloud = file.read()
        min_x, min_y = np.min(np.vstack((point_cloud.x, point_cloud.y)), axis=1)
        max_x, max_y = np.max(np.vstack((point_cloud.x, point_cloud.y)), axis=1)
    return min_x, max_x, min_y, max_y


def generate_dsm(input_folder, output_folder, run_name, method, resolution, fill_gaps=True):
    # Define the final output folder and ensure it exists.
    final_output_folder = os.path.join(output_folder, run_name, 'DSM')
    os.makedirs(final_output_folder, exist_ok=True)
    
    # Create a temporary folder for intermediate outputs.
    temp_folder = os.path.join(final_output_folder, "temp")
    os.makedirs(temp_folder, exist_ok=True)
    
    #print("\nStarting DSM generation...")
    start_time = time.time()
    
    las_files = glob.glob(os.path.join(input_folder, run_name, "*.las")) + \
                glob.glob(os.path.join(input_folder, run_name, "*.laz"))
    
    if not las_files:
        print("No LAS/LAZ files found. Exiting DSM generation.")
        return
    
    for las_file in tqdm(las_files, desc="Processing DSMs", unit="file"):
        try:
            base_name = os.path.splitext(os.path.basename(las_file))[0]
            # Use the temporary folder for intermediate files.
            temp_dsm_path = os.path.join(temp_folder, f"{base_name}_dsm.tif")
            temp_filled_dsm_path = os.path.join(temp_folder, f"{base_name}_dsm_filled.tif")
            # Final DSM will be saved directly in the final output folder.
            final_dsm_path = os.path.join(final_output_folder, f"{base_name}_DSM.tif")
            
            # Check resolution suitability.
            avg_spacing, is_resolution_ok = check_resolution(las_file, resolution, method)
            if not is_resolution_ok:
                print(f"Warning: DSM resolution ({resolution}m) is finer than average point spacing ({avg_spacing:.3f}m).")
                print("Consider increasing the resolution to avoid interpolation gaps.")
            
            # Check if classification exists (if needed).
            has_classification = check_classification_exists(las_file)
            
            # Define the PDAL pipeline using the temporary DSM path.
            pipeline = [
                {"type": "readers.las", "filename": las_file},
                {"type": "filters.ferry", "dimensions": "Z=>Elevation"},
                {
                    "type": "filters.range",
                    "limits": "Classification[0:0]"  # Use all points for initial DSM
                },
                {
                    "type": "writers.gdal",
                    "filename": temp_dsm_path,
                    "resolution": avg_spacing,
                    "output_type": "max",
                    "nodata": -9999,
                    "gdalopts": "COMPRESS=LZW"
                }
            ]
            
            # Run PDAL pipeline.
            pdal.pipeline.Pipeline(json.dumps(pipeline)).execute()
            #print(f"DSM saved (temp): {temp_dsm_path}")
            
            # Fill gaps using GDAL if enabled.
            if fill_gaps:
                subprocess.run([
                    "gdal_fillnodata.py",
                    "-md", "10",
                    "-si", "2",
                    temp_dsm_path,
                    temp_filled_dsm_path
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                #print(f"Filled DSM saved (temp): {temp_filled_dsm_path}")
                # Move the gap-filled DSM to the final output folder.
                os.replace(temp_filled_dsm_path, final_dsm_path)
                #print(f"Final DSM saved as: {final_dsm_path}")
            else:
                # If not filling gaps, move the initial DSM.
                os.rename(temp_dsm_path, final_dsm_path)
                #print(f"Final DSM saved as: {final_dsm_path}")
                
        except Exception as e:
            print(f"Error processing {las_file}: {e}")
    
    # Delete the temporary folder after processing all files.
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
        #print("Temporary files have been deleted.")
    
    elapsed_time = timedelta(seconds=int(time.time() - start_time))
    print(f"\nDSM generation completed in {elapsed_time}.")


def generate_dtm(input_folder, output_folder, run_name, method, rigidness, iterations, resolution, fill_gaps=True):
    """
    Generates a DTM from LAS/LAZ files using the cloth simulation filtering (CSF) method.
    
    Parameters:
        input_folder (str): The base folder containing input point cloud files.
        output_folder (str): The folder where the output will be saved.
        run_name (str): The subfolder (or run identifier) within input_folder.
        method (str): The ground filtering method to use (e.g., 'cloth').
        resolution (float): The output grid resolution in meters.
        fill_gaps (bool): Whether to run gap-filling on the output DTM.
    """
    # Define the final output folder and ensure it exists.
    final_output_folder = os.path.join(output_folder, run_name, 'DTM')
    os.makedirs(final_output_folder, exist_ok=True)
    
    # Create a temporary folder for intermediate outputs.
    temp_folder = os.path.join(final_output_folder, "temp")
    os.makedirs(temp_folder, exist_ok=True)
    
    print("\nStarting DTM generation")
    start_time = time.time()
    
    # Collect all LAS/LAZ files.
    las_files = glob.glob(os.path.join(input_folder, run_name, "*.las")) + \
                glob.glob(os.path.join(input_folder, run_name, "*.laz"))
    
    if not las_files:
        print("No LAS/LAZ files found. Exiting DTM generation.")
        return
    
    for las_file in tqdm(las_files, desc="Processing DTMs", unit="file"):
        try:
            base_name = os.path.splitext(os.path.basename(las_file))[0]
            # Temporary DTM paths.
            temp_DTM_path = os.path.join(temp_folder, f"{base_name}_DTM.tif")
            temp_filled_DTM_path = os.path.join(temp_folder, f"{base_name}_DTM_filled.tif")
            # Final DTM will be saved directly in the final output folder.
            final_DTM_path = os.path.join(final_output_folder, f"{base_name}_DTM.tif")

            avg_spacing, is_resolution_ok = check_resolution(las_file, resolution, method)
            resolution = avg_spacing
            # Define the PDAL pipeline using the cloth simulation filter.
            pipeline = [
                {"type": "readers.las", "filename": las_file},
                # Apply the CSF filter to classify ground points.
                {"type": "filters.csf",
                 "resolution": resolution,  # Adjust based on your dataset
                 "rigidness": rigidness,                  # Typical value; modify if needed
                 "iterations": iterations                # Number of iterations for the simulation
                },
                {"type": "filters.ferry", "dimensions": "Z=>Elevation"},
                # Filter only ground points (assuming CSF sets ground points to classification 2)
                {"type": "filters.range", "limits": "Classification[2:2]"},
                {"type": "writers.gdal",
                 "filename": temp_DTM_path,
                 "resolution": resolution,
                 "output_type": "mean",
                 "nodata": -9999,
                 "gdalopts": "COMPRESS=LZW"}
            ]
            
            # Run the PDAL pipeline.
            pdal.pipeline.Pipeline(json.dumps(pipeline)).execute()
            
            # Fill gaps using GDAL if enabled.
            if fill_gaps:
                subprocess.run([
                    "gdal_fillnodata.py",
                    "-md", "10",
                    "-si", "2",
                    temp_DTM_path,
                    temp_filled_DTM_path
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # Move the gap-filled DTM to the final output folder.
                os.replace(temp_filled_DTM_path, final_DTM_path)
            else:
                # Move the initial DTM.
                os.rename(temp_DTM_path, final_DTM_path)
                
        except Exception as e:
            print(f"Error processing {las_file}: {e}")
    
    # Clean up the temporary folder after processing.
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
    
    elapsed_time = timedelta(seconds=int(time.time() - start_time))
    print(f"\nDTM generation completed in {elapsed_time}.")


def generate_chm(input_folder, output_folder, run_name):
    """
    Generates Canopy Height Models (CHM) for all corresponding DSM and DTM files in the given folders.

    Parameters:
        input_folder (str): Base folder containing DSM and DTM subfolders.
        output_folder (str): Base folder where CHM outputs will be saved.
        run_name (str): Name of the subfolder (same for DSM, DTM, and CHM).

    Returns:
        None
    """
    dsm_folder = os.path.join(input_folder, run_name, "DSM")
    DTM_folder = os.path.join(input_folder, run_name, "DTM")
    chm_folder = os.path.join(output_folder, run_name, "CHM")

    # Ensure output folder exists
    os.makedirs(chm_folder, exist_ok=True)

    # Find all DSM files
    dsm_files = glob.glob(os.path.join(dsm_folder, "*.tif"))

    print("\nStarting CHM generation")
    start_time = time.time()
    

    if not dsm_files:
        print(f"No DSM files found in {dsm_folder}. Exiting CHM generation.")
        return

    for dsm_path in tqdm(dsm_files, desc="Processing CHMs", unit="file"):
        try:
            # Extract base name (without extension)
            base_name = os.path.splitext(os.path.basename(dsm_path))[0]
            
            # Find the corresponding DTM file
            base_name = os.path.splitext(os.path.basename(dsm_path))[0].replace("_DSM", "")
            DTM_path = os.path.join(DTM_folder, f"{base_name}_DTM.tif")
            chm_output_path = os.path.join(chm_folder, f"{base_name}_CHM.tif")

            if not os.path.exists(DTM_path):
                print(f"Skipping {base_name}: Corresponding DTM not found.")
                continue

            # Open DSM and DTM rasters
            with rasterio.open(dsm_path) as dsm_src, rasterio.open(DTM_path) as DTM_src:
                # Read the raster data
                dsm = dsm_src.read(1)
                dtm = DTM_src.read(1)
                
                # Ensure they have the same shape
                if dsm.shape != dtm.shape:
                    print(f"Skipping {base_name}: DSM and DTM raster sizes do not match.")
                    continue

                # Calculate CHM by subtracting DTM from DSM
                chm = dsm - dtm

                # Handle NoData values
                chm[dsm == dsm_src.nodata] = dsm_src.nodata
                chm[dtm == DTM_src.nodata] = DTM_src.nodata

                # Define output metadata
                chm_meta = dsm_src.meta.copy()
                chm_meta.update(dtype=rasterio.float32)

                # Save the CHM raster
                with rasterio.open(chm_output_path, "w", **chm_meta) as chm_dst:
                    chm_dst.write(chm.astype(rasterio.float32), 1)

                

        except Exception as e:
            print(f"Error processing {base_name}: {e}")

    elapsed_time = timedelta(seconds=int(time.time() - start_time))
    print(f"\n CHM generation completed in {elapsed_time}.")


def process_all(config):
    """
    Runs DSM generation using cleaned LAS files.

    Reads from: `config.preprocessed_dir`
    Saves to: `config.results_dir / run_name / DSM/`
    """
    print('Starting Processing ...')

    start_time = time.time()

    if config.create_DSM:
        print("\n========== Starting DSM Generation ==========")
        generate_dsm(
            input_folder=config.preprocessed_dir,
            output_folder=config.results_dir,
            run_name=config.run_name,
            resolution=config.resolution,
            fill_gaps=config.fill_gaps, 
            method=config.point_density_method
        )
    
    

    if config.create_DEM:
        print("\n========== Starting DEM Generation ==========")
        generate_dtm(
            input_folder=config.preprocessed_dir,
            output_folder=config.results_dir,
            run_name=config.run_name,
            resolution=config.resolution,
            fill_gaps=config.fill_gaps,
            method=config.point_density_method, 
            rigidness = config.rigidness,
            iterations = config.iterations
        )

    if config.create_CHM:
        print("\n========== Starting CHM Generation ==========")
        generate_chm(
            input_folder=config.results_dir,
            output_folder=config.results_dir,
            run_name=config.run_name
        )

    elapsed_time = timedelta(seconds=int(time.time() - start_time))
    print(f"\n DEM generation completed in {elapsed_time}.\n")
