import os
import rasterio
from rasterio.plot import reshape_as_image
import numpy as np
import cv2
from tqdm import tqdm  # Import tqdm for the progress bar

# Function to process each GeoTIFF file
def process_geotiff(input_path, output_path, pixels_to_erode=6):
    # Read the GeoTIFF file
    with rasterio.open(input_path) as src:
        image = src.read(1)  # Read the first band (grayscale assumption)
        profile = src.profile  # Keep the metadata for writing back

    # Define a kernel for morphological operations
    kernel_size = (pixels_to_erode * 2 + 1)  # Kernel size for erosion/dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Apply morphological opening to remove small artifacts
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # Apply morphological closing to fill small holes in the deposits
    closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)

    # Update profile to match data
    profile.update(dtype=rasterio.uint8, count=1)

    # Save the resulting image back as GeoTIFF
    try:
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(closed_image.astype(rasterio.uint8), 1)  # Ensure data type matches
        print(f"Closed GeoTIFF image saved to {output_path}")
    except PermissionError as e:
        print(f"Permission denied for {output_path}: {e}")
    except rasterio.errors.RasterioIOError as e:
        print(f"Rasterio IO error for {output_path}: {e}")

# Directory containing the GeoTIFF files
input_folder = '/isipd/projects/p_planetdw/data/dw_detection/PlanetScope/results/all/rasters'
output_folder = '/isipd/projects/p_planetdw/data/dw_detection/PlanetScope/results/larger_prob05_morph_filtered_disk7/rasters'

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get all .tif files in the input folder
input_files = [f for f in os.listdir(input_folder) if f.endswith(".tif")]

# Iterate over each .tif file in the input folder with tqdm progress bar
for filename in tqdm(input_files, desc="Processing files", unit="file"):
    input_image_path = os.path.join(input_folder, filename)
    output_image_path = os.path.join(output_folder, f"closed_{filename}")
    
    # Check if the output file already exists
    if os.path.exists(output_image_path):
        print(f"Skipping {filename} as output already exists.")
        continue
    
    process_geotiff(input_image_path, output_image_path)
