import os
import time
import json
import glob
import pdal
import laspy
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull, QhullError
from shapely.geometry import mapping
from tqdm import tqdm
from datetime import timedelta


def extract_footprint_batch(input_folder, output_folder):
    """Extracts footprints (convex hulls) from LAS/LAZ files and saves them in a specified folder."""
    
    os.makedirs(output_folder, exist_ok=True)
    
    print("\nStarting footprint extraction...")
    start = time.time()

    laz_files = glob.glob(os.path.join(input_folder, "*.laz")) + glob.glob(os.path.join(input_folder, "*.las"))

    if not laz_files:
        print("No LAS/LAZ files found in the input directory. Exiting.")
        return

    for laz_file in tqdm(laz_files, desc="Processing footprints", unit="file"):
        try:
            with laspy.open(laz_file) as file:
                point_cloud = file.read()
                x, y = point_cloud.x, point_cloud.y
                las_crs = file.header.parse_crs()
                crs = las_crs.to_epsg() if las_crs and las_crs.to_epsg() else "EPSG:4326"

            unique_points = np.unique(np.vstack((x, y)).T, axis=0)
            try:
                hull = ConvexHull(unique_points)
                footprint = Polygon(unique_points[hull.vertices])
            except QhullError:
                print(f"Warning: Convex Hull computation failed for {laz_file}. Creating an empty polygon.")
                footprint = Polygon()

            gdf = gpd.GeoDataFrame({'geometry': [footprint]}, crs=crs)
            output_path = os.path.join(output_folder, os.path.splitext(os.path.basename(laz_file))[0] + ".gpkg")
            gdf.to_file(output_path, driver="GPKG")

        except Exception as e:
            print(f"Error processing {laz_file}: {e}")

    print(f"Footprint extraction completed in {timedelta(seconds=int(time.time() - start))}.")

def is_utm_crs(las_file):
    """Checks if a LAS/LAZ file is in a UTM projection."""
    with laspy.open(las_file) as file:
        crs = file.header.parse_crs()
        crs_epsg = crs.to_epsg() if crs else None

    if crs_epsg is None:
        print(f"Warning: No CRS found in {las_file}. Assuming it needs reprojection.")
        return False

    return 32600 <= crs_epsg <= 32799  # UTM zones are EPSG:32600-32660 (Northern) & EPSG:32700-32760 (Southern)


def get_utm_epsg(las_file):
    """Detects the best UTM EPSG code based on the LAS file's longitude."""
    with laspy.open(las_file) as file:
        point_cloud = file.read()
        avg_longitude = np.mean(point_cloud.x)

    utm_zone = int((avg_longitude + 180) / 6) + 1
    is_northern = np.mean(point_cloud.y) >= 0  # Check if the point cloud is in the northern hemisphere
    epsg_code = 32600 + utm_zone if is_northern else 32700 + utm_zone

    print(f"Detected UTM Zone: {utm_zone}, EPSG: {epsg_code}")
    return epsg_code


def reproject_las(input_las, output_las):
    """Reprojects a LAS/LAZ file to UTM if it's not already in a UTM CRS."""
    if is_utm_crs(input_las):
        print(f"Skipping reprojection for {input_las}: Already in UTM.")
        return input_las  # Return original file if already projected

    target_epsg = get_utm_epsg(input_las)

    pipeline = [
        {"type": "readers.las", "filename": input_las},
        {"type": "filters.reprojection", "out_srs": f"EPSG:{target_epsg}"},
        {"type": "writers.las", "filename": output_las}
    ]

    print(f"Reprojecting {input_las} to EPSG:{target_epsg} -> {output_las}")
    pdal.pipeline.Pipeline(json.dumps(pipeline)).execute()
    return output_las  # Return new file path

def match_footprints(target_footprint_dir, las_footprint_dir, las_file_dir):
    """Matches target footprints with LAS files and returns a dictionary mapping target area names to LAS file paths."""
    
    os.makedirs(las_footprint_dir, exist_ok=True)
    
    print("\nStarting footprint matching...")
    start = time.time()

    if not os.listdir(las_footprint_dir):
        print("No footprint files found. Generating footprints first.")
        extract_footprint_batch(las_file_dir, las_footprint_dir)

    target_footprints = [os.path.join(target_footprint_dir, f) for f in os.listdir(target_footprint_dir) if f.endswith(".gpkg")]
    las_footprints = [os.path.join(las_footprint_dir, f) for f in os.listdir(las_footprint_dir) if f.endswith(".gpkg")]

    target_dict = {}

    for target_fp in target_footprints:
        target_gdf = gpd.read_file(target_fp)
        target_name = os.path.basename(target_fp)
        las_paths = []

        for las_fp in las_footprints:
            las_gdf = gpd.read_file(las_fp)
            if target_gdf.crs != las_gdf.crs:
                las_gdf = las_gdf.to_crs(target_gdf.crs)

            joined = gpd.sjoin(las_gdf, target_gdf, predicate="intersects")
            if not joined.empty:
                las_name = os.path.splitext(os.path.basename(las_fp))[0] + ".las"
                las_path = os.path.join(las_file_dir, las_name)
                if os.path.exists(las_path):
                    las_paths.append(las_path)

        target_dict[target_name] = las_paths

        # Print the number of LAS files found for each target area
        print(f"Target area: {target_name}, LAS files found: {len(las_paths)}")

    print(f"Footprint matching completed in {timedelta(seconds=int(time.time() - start))}. Found {len(target_dict)} target areas.")
    return target_dict


def get_las_header(las_file):
    """Extracts scale, offset, and CRS from an input LAS file."""
    with laspy.open(las_file) as las:
        header = las.header
        scale = header.scales
        offset = header.offsets
        crs = header.parse_crs()
        crs_epsg = crs.to_epsg() if crs else 4979  # Default to EPSG:4979 if unknown
    return scale, offset, crs_epsg


def merge_and_clean_las(las_dict, preprocessed_dir, run_name, target_footprint_dir, sor_knn, sor_multiplier):
    """
    Merges LAS files using PDAL while maintaining original header settings (scale, offset, CRS).
    Handles overlapping files by preserving header information from the first file.
    Clips merged LAS files to the target area's boundary.
    Removes outliers using Statistical Outlier Removal (SOR) and excludes points with classification 7.
    Reprojects the final output to UTM before writing.
    """
    run_merged_dir = os.path.join(preprocessed_dir, run_name)
    os.makedirs(run_merged_dir, exist_ok=True)
    
    print("\nMerging, cleaning, clipping, and reprojecting LAS files using PDAL...")
    start = time.time()
    
    for target_fp, las_files in tqdm(las_dict.items(), desc="Processing target areas", unit="area"):
        if not las_files:
            print(f"No valid LAS files for {target_fp}. Skipping.")
            continue
            
        # Get header information from the first file
        ref_scale, ref_offset, ref_crs = get_las_header(las_files[0])
        
        # Load the corresponding footprint file
        footprint_path = os.path.join(target_footprint_dir, target_fp if target_fp.endswith('.gpkg') else f"{target_fp}.gpkg")
        if not os.path.exists(footprint_path):
            print(f"Footprint file {footprint_path} not found. Skipping clip.")
            continue
        
        gdf = gpd.read_file(footprint_path)
        
        # Reproject geometry to match the point cloud CRS
        gdf = gdf.to_crs(epsg=ref_crs)
        target_geom = gdf["geometry"].iloc[0]
        
        bbox = mapping(target_geom)  # Convert to GeoJSON format
        
        # Create PDAL pipeline
        pipeline = [{"type": "readers.las", "filename": las_files[0]}]
        
        # Add merge readers for remaining files
        for las_file in las_files[1:]:
            pipeline.append({"type": "readers.las", "filename": las_file})
            
        # Add merge filter
        pipeline.append({"type": "filters.merge"})
        
        # Add cropping filter (clip by bounding box)
        pipeline.append({
            "type": "filters.crop",
            "polygon": json.dumps(bbox)
        })
        
        # Add outlier filtering (Statistical Outlier Removal - SOR)
        pipeline.append({
            "type": "filters.outlier",
            "method": "statistical",
            "mean_k": sor_knn,
            "multiplier": sor_multiplier
        })
        
        # Add classification filter to exclude points with classification 7 (noise)
        pipeline.append({
            "type": "filters.range",
            "limits": "Classification![7:7]"
        })
        
        # Temporary output file before reprojection
        temp_output_file = os.path.join(run_merged_dir, f"{os.path.splitext(target_fp)[0]}_temp.las")
        pipeline.append({
            "type": "writers.las",
            "filename": temp_output_file,
            "scale_x": str(ref_scale[0]),
            "scale_y": str(ref_scale[1]),
            "scale_z": str(ref_scale[2]),
            "offset_x": str(ref_offset[0]),
            "offset_y": str(ref_offset[1]),
            "offset_z": str(ref_offset[2]),
            "a_srs": f"EPSG:{ref_crs}"
        })

        # Execute initial processing pipeline
        try:
            pdal.pipeline.Pipeline(json.dumps(pipeline)).execute()
            print(f"Successfully processed {target_fp} (pre-reprojection)")
        except Exception as e:
            print(f"Error processing {target_fp}: {str(e)}")
            continue
        
        # **Reproject the final output**
        final_output_file = os.path.join(run_merged_dir, f"{os.path.splitext(target_fp)[0]}.las")
        reprojected_file = reproject_las(temp_output_file, final_output_file)

        print(f"Final output saved: {reprojected_file}")

    print(f"\nProcessing completed in {str(timedelta(seconds=time.time() - start)).split('.')[0]}.")



def preprocess_all(conf):
    """Runs full preprocessing pipeline for LAS data."""

    global config
    config = conf

    print("\n========== Starting Preprocessing ==========")
    start = time.time()

    run_name = config.run_name

    # Ensure required directories exist
    os.makedirs(os.path.join(config.preprocessed_dir, run_name), exist_ok=True)
    os.makedirs(os.path.join(config.results_dir, run_name), exist_ok=True)

    # Step 1: Match footprints with LAS files
    print("\n--- Matching footprints to LAS files ---")
    target_dict = match_footprints(
        target_footprint_dir=config.target_area_dir, 
        las_footprint_dir=config.las_footprints_dir, 
        las_file_dir=config.las_files_dir,
        #run_name=run_name
    )

    # Step 2: Merge and Clean LAS files using Statistical Outlier Removal (SOR)
    print("\n--- Merging and Cleaning LAS files ---")
    merge_and_clean_las(
        target_footprint_dir=config.target_area_dir,
        las_dict=target_dict, 
        preprocessed_dir=config.preprocessed_dir, 
        sor_knn=config.knn,  # Adjust based on density
        sor_multiplier=config.multiplier,  # Adjust based on noise level
        run_name=run_name
    )

    print(f"\nPreprocessing completed in {str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n")