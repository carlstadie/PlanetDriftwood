import os
import numpy as np
import rasterio
from rasterio.enums import Resampling

def calculate_ndvi(red_band, nir_band):
    """
    Calculate NDVI using the formula: (NIR - Red) / (NIR + Red)
    """
    np.seterr(divide='ignore', invalid='ignore')  # Ignore division by zero
    ndvi = (nir_band.astype(float) - red_band.astype(float)) / (nir_band + red_band)
    return ndvi

def scale_ndvi_to_uint16(ndvi):
    """
    Scale NDVI values to fit into uint16 range:
    - Clip NDVI values to the range [-10000, 10000]
    - Add 10000 to shift the range from [-10000, 10000] to [0, 20000]
    """
    # Clip NDVI values to the range [-10000, 10000]
    ndvi_clipped = np.clip(ndvi * 10000, -10000, 10000)

    # Shift values by adding 10000 to move them to the uint16 range (0 to 20000)
    ndvi_shifted = ndvi_clipped + 10000
    
    # Convert to uint16
    return ndvi_shifted.astype(np.uint16)

def process_raster(filepath, output_folder):
    """
    Process a single raster file, calculate NDVI, and append NDVI with the original bands
    """
    print(f"Processing file: {filepath}")
    
    try:
        # Create the output file path
        ndvi_output_path = os.path.join(output_folder, f"ndvi_{os.path.basename(filepath)}")
        
        # Check if the output file already exists, skip if it does
        if os.path.exists(ndvi_output_path):
            print(f"File {ndvi_output_path} already processed. Skipping.")
            return

        with rasterio.open(filepath) as dataset:
            # Read the original bands 1, 2, 3 (Red), and 4 (NIR) as uint16
            blue = dataset.read(1).astype(np.uint16)  # Blue band (Band 1)
            green = dataset.read(2).astype(np.uint16)  # Green band (Band 2)
            red = dataset.read(3).astype(np.uint16)  # Red band (Band 3)
            nir = dataset.read(4).astype(np.uint16)  # NIR band (Band 4)

            # Calculate NDVI
            ndvi = calculate_ndvi(red, nir)

            # Scale NDVI to fit uint16 data type by clipping and shifting
            scaled_ndvi = scale_ndvi_to_uint16(ndvi)

            # Copy metadata and update for 5 bands (bands 1-4 as uint16, band 5 as uint16)
            metadata = dataset.meta.copy()
            metadata.update(
                dtype='uint16',  # Set data type to uint16 for all bands
                count=5  # Number of bands (4 original + 1 NDVI)
            )

            # Write the original bands and scaled NDVI to a new file
            with rasterio.open(ndvi_output_path, 'w', **metadata) as dst:
                # Write the first four bands (1-4) as uint16
                dst.write(blue, 1)
                dst.write(green, 2)
                dst.write(red, 3)
                dst.write(nir, 4)

                # Write the scaled NDVI as band 5 in uint16
                dst.write(scaled_ndvi, 5)

            print(f"NDVI saved to {ndvi_output_path}")
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def process_folder(folder_path, output_folder):
    """
    Process all .jp2 raster files in the given folder to calculate NDVI and append to original bands
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Check if input folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Input folder '{folder_path}' does not exist.")
        return
    
    # List all files in the folder
    file_count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.jp2'):
            filepath = os.path.join(folder_path, filename)
            process_raster(filepath, output_folder)
            file_count += 1
    
    if file_count == 0:
        print(f"No .jp2 files found in the directory: {folder_path}")

# Set the folder paths
input_folder = "/isipd/projects/p_planetdw/data/dw_detection/PlanetScope/images/2019"  # Change this to your folder
output_folder = "/isipd/projects/p_planetdw/data/dw_detection/PlanetScope/images/new_2019_ndvi"   # Change this to your desired output folder

# Run the NDVI calculation for all .jp2 rasters in the folder
process_folder(input_folder, output_folder)
