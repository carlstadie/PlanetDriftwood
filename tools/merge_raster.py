import rasterio
from rasterio.merge import merge
import glob
import os
import re
import numpy as np

# Define the directory containing the raster files
raster_dir = r"/isipd/projects/p_planetdw/data/mosaicing/arcticdem/ArcticDEM"  # Update to your directory
output_dir = r"/isipd/projects/p_planetdw/data/mosaicing/arcticdem/DEM"       # Update to your output directory

# Regular expression to extract the ID (N54W164 pattern) from the filename
pattern = re.compile(r".*(N\d{2}W\d{3}).*")

# Dictionary to hold rasters by ID
raster_groups = {}

# Find all raster files in the directory (assuming .tif files)
raster_files = glob.glob(os.path.join(raster_dir, "*.tif"))

# Group rasters by ID
for raster in raster_files:
    match = pattern.match(os.path.basename(raster))
    if match:
        raster_id = match.group(1)
        if raster_id not in raster_groups:
            raster_groups[raster_id] = []
        raster_groups[raster_id].append(raster)

# Function to merge rasters with the same ID
def merge_rasters(rasters, output_path):
    # Open all raster files before merging them
    src_files_to_mosaic = [rasterio.open(fp) for fp in rasters]
    
    #print(f"Merging the following rasters: {rasters}")  # Print raster names being processed
    
    try:
        # Merge the rasters with 'max' method for overlapping areas
        mosaic, out_trans = merge(src_files_to_mosaic, method='max')
        
        # Post-processing
        mosaic = np.maximum(mosaic, 0.1)
        mosaic = mosaic*10
        mosaic[np.isnan(mosaic)] = 0
        mosaic = mosaic.astype(np.uint16)

        # Copy the metadata from the first raster
        out_meta = src_files_to_mosaic[0].meta.copy()

        # Update the metadata for JPEG 2000 output
        out_meta.update({
            "driver": "JP2OpenJPEG",  # Set the driver to JPEG 2000
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "dtype": 'uint16'  # Ensures the data type is preserved
        })

        # Write the merged file as JP2
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)
        print(f"Successfully merged and saved: {output_path}")  # Print success message
    
    except Exception as e:
        print(f"Error while merging rasters: {rasters}")  # Log the rasters causing the issue
        print(f"Exception: {e}")  # Print the exception message
    
    finally:
        # Close all opened raster files
        for src in src_files_to_mosaic:
            src.close()

# Process each group of rasters by ID
for raster_id, rasters in raster_groups.items():
    print(f"Processing raster group ID: {raster_id}")  # Print current raster group ID
    # Define output file name
    output_file = os.path.join(output_dir, f"{raster_id}_DEM.jp2")
    # Merge and save the rasters
    merge_rasters(rasters, output_file)

print("Merging complete!")
