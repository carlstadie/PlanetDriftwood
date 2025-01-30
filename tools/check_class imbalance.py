import os
import rasterio
import numpy as np

def count_ones_and_zeros_in_band5(folder_path):
    count_ones = 0
    count_zeros = 0
    
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if it's a file and a raster image
        if os.path.isfile(file_path) and file_path.endswith('.tif'):
            # Open the image using rasterio
            with rasterio.open(file_path) as src:
                # Read the fifth band
                if src.count >= 5:
                    band5 = src.read(5)  # Read the 5th band
                    count_ones += np.sum(band5 == 1)
                    count_zeros += np.sum(band5 == 0)
                else:
                    print(f"Image {filename} does not have 5 bands.")
    
    return count_ones, count_zeros

# Define the folder path containing the images
folder_path = r"N:\isipd\projects\p_planetdw\data\dw_detection\PlanetScope\training_data\training_data_additional_positives"


# Call the function and get the results
ones, zeros = count_ones_and_zeros_in_band5(folder_path)

print(f"Total count of 1s in band 5: {ones}")
print(f"Total count of 0s in band 5: {zeros}")
