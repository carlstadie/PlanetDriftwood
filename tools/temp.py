from pathlib import Path
import shutil
import re
import logging
from tqdm import tqdm  # Import progress bar library
from datetime import datetime, timezone
import pandas as pd

def setup_logging(log_file):
    """Set up logging to overwrite the log file each run."""
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(message)s", filemode='w')

def read_mta_list(mta_list_path):
    """Read the MTA zone list and return a dictionary mapping filenames to zones."""
    file_zone_map = {}
    with mta_list_path.open('r') as f:
        for line in f:
            parts = line.strip().split()       
            if len(parts) == 2:
                file_zone_map[parts[0]] = parts[1]
    return file_zone_map

def extract_time(filename):
    """Extracts time from filename (formatted as Txxxxx) and converts it to HH:MM:SS format."""
    match = re.search(r"T(\d{6})", filename)
    if match:
        time_str = match.group(1)
        return datetime.strptime(time_str, "%H%M%S")
    return None

def time_difference(target, reference):
    """Computes absolute time difference in seconds."""
    return abs((target - reference).total_seconds())

def find_closest_las(date_time, folder, folder_cache):
    """Find the closest matching .las file based on date and time."""
    date, time = date_time.split('_')
    full_date = "20" + date  # Convert 230710 → 20230710
    target_time = datetime.strptime(time, "%H%M%S")

    if folder not in folder_cache:
        folder_cache[folder] = list(folder.glob("*.las"))
    
    files = folder_cache[folder]
    candidate_files = [f for f in files if full_date in f.name]

    if not candidate_files:
        return None

    closest_file = min(candidate_files, key=lambda f: time_difference(target_time, extract_time(f.name)) if extract_time(f.name) else float('inf'))
    time_diff = time_difference(target_time, extract_time(closest_file.name)) if closest_file else None
    
    return closest_file, time_diff

def find_tiff_pair(las_file, tiff_folder):
    """Find the corresponding .tif file for a given .las file."""
    tiff_filename = las_file.name.replace(".las", "_summary.tif")
    tiff_file = tiff_folder / tiff_filename
    return tiff_file if tiff_file.exists() else None

def process_mta_files(mta_list_path, output_folder_tiff, output_folder_las, log_file, zone_folders):
    """Main function to process MTA files and match them with LAS and TIFF files."""
    # Setup directories
    output_folder_tiff.mkdir(parents=True, exist_ok=True)
    output_folder_las.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(log_file)

    # Read MTA list
    file_zone_map = read_mta_list(mta_list_path)

    # Tracking statistics
    matched_files = []
    unmatched_files = []
    match_details = []
    folder_cache = {}

    # Process each file with a progress bar
    for base_filename, zone in tqdm(file_zone_map.items(), desc="Processing files", unit="file"):
        matched = False
        
        if zone in zone_folders:
            las_folder = zone_folders[zone][1]
            tiff_folder = zone_folders[zone][0]

            if las_folder.exists():
                matched_las, time_diff = find_closest_las(base_filename, las_folder, folder_cache)
                if matched_las:
                    shutil.copy(matched_las, output_folder_las)
                    matched_files.append((base_filename, zone, matched_las))

                    # Find and copy corresponding .tif file
                    matched_tiff = find_tiff_pair(matched_las, tiff_folder)
                    if matched_tiff:
                        shutil.copy(matched_tiff, output_folder_tiff)

                    match_details.append(f"{base_filename} -> {matched_las.name} (Time diff: {str(datetime.fromtimestamp(time_diff, tz=timezone.utc).strftime('%H:%M:%S'))})")
                    matched = True

        if not matched:
            unmatched_files.append(base_filename)

    # Summary log
    with log_file.open('w') as log:
        log.write(f"Summary:\n")
        log.write(f"Total files in list: {len(file_zone_map)}\n")
        log.write(f"Files found: {len(matched_files)}/{len(file_zone_map)}\n")
        log.write(f"Unmatched files: {len(unmatched_files)}\n\n")

        log.write("Matched files:\n")
        for entry in matched_files:
            log.write(f"{entry[0]} | Zone {entry[1]} | {entry[2]}\n")

        log.write("\nUnmatched files:\n")
        for entry in unmatched_files:
            log.write(f"{entry}\n")

        log.write("\nMatch details (Time differences):\n")
        for detail in match_details:
            log.write(f"{detail}\n")

    logging.info("Processing complete.")

# Example usage
if __name__ == "__main__":
    #Define file paths
    mta_list_path = Path("/response/Restricted_Airborne/Q680_lidar/Canada_2023_Perma-X/02_Processed/MTA_ZONE_list.txt")
    output_folder_tiff = Path("/isipd/projects/p_planetdw/example_dem/output/tiff")
    output_folder_las = Path("/isipd/projects/p_planetdw/example_dem/output/las")
    log_file = Path("/isipd/projects/p_planetdw/example_dem/match_log.txt")

    # Define zone folders for TIFF and LAS files
    zone_folders = {
    "2": [Path("/response/Restricted_Airborne/Q680_lidar/Canada_2023_Perma-X/02_Processed/03_product-dem/ALS_mta2"), 
          Path("/response/Restricted_Airborne/Q680_lidar/Canada_2023_Perma-X/02_Processed/02_product-als/mta2_las")],
    "3": [Path("/response/Restricted_Airborne/Q680_lidar/Canada_2023_Perma-X/02_Processed/03_product-dem/ALS_mta3"), 
          Path("/response/Restricted_Airborne/Q680_lidar/Canada_2023_Perma-X/02_Processed/02_product-als/mta3_las")]
    }

    # Run the processing function
    process_mta_files(mta_list_path, output_folder_tiff, output_folder_las, log_file, zone_folders)
