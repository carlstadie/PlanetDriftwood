import os
import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
from tqdm import tqdm  # Progress bar

def calculate_average_elevation_below_ndvi_threshold(img_data, dem_data, ndvi_band=4, threshold=10000): #shifted ndvi by 10000 so 10000 is 0
    """
    Calculate the average elevation of all DEM pixels where the NDVI (in the specified band) is below a threshold.
    """
    # Extract the NDVI band (band 5, index 4 in 0-based indexing)
    ndvi = img_data[ndvi_band]
    
    # Mask where NDVI is smaller than the threshold (e.g., < 0)
    mask = ndvi < threshold
    
    # Extract corresponding DEM values for those masked NDVI values
    dem_values_below_threshold = dem_data[mask]
    
    # Calculate the average elevation where NDVI < threshold
    if dem_values_below_threshold.size > 0:
        avg_elevation = np.mean(dem_values_below_threshold)
    else:
        avg_elevation = 0  # Handle cases where no pixels are below the threshold
    
    return avg_elevation

def get_raster_id(filename, file_type):
    """
    Extract ID from raster file name based on the file type (DEM or IMG).
    For DEM files, the ID is at the start (before the first '_').
    For IMG files, the ID is just before the last '_' in the filename.
    """
    if file_type == 'DEM':
        return filename.split('_')[0]  # ID is before the first underscore in DEM file names.
    elif file_type == 'IMG':
        return filename.split('_')[-3]  # ID is the third element from the end in IMG file names.

def reproject_dem_to_img_crs(dem_data, dem_transform, dem_crs, img_crs, img_transform, img_shape):
    """
    Reprojects DEM data to the CRS of the IMG raster and resamples to the target shape.
    """
    reprojected_dem = np.empty(img_shape, dtype=np.float32)

    reproject(
        source=dem_data,
        destination=reprojected_dem,
        src_transform=dem_transform,
        src_crs=dem_crs,
        dst_transform=img_transform,
        dst_crs=img_crs,
        resampling=Resampling.bilinear
    )

    return reprojected_dem

def add_dem_to_img(img_folder, dem_folder, output_folder):
    """Add DEM band to matching IMG rasters and mask DEM values where IMG has no data (0).
       Also set DEM values > 65000 (og no data in DEM) to 0, and include a progress bar for processing.
       Skips files that have already been processed.
    """
    
    error_ids = []  # List to keep track of image IDs where errors occur

    # Get the list of rasters in each folder
    img_files = {get_raster_id(f, 'IMG'): os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.jp2')}
    dem_files = {get_raster_id(f, 'DEM'): os.path.join(dem_folder, f) for f in os.listdir(dem_folder) if f.endswith('.jp2')}

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through img files and match with dem files by ID
    for img_id, img_path in tqdm(img_files.items(), desc="Processing IMG files"):
        # Define the output path
        output_path = os.path.join(output_folder, f'{img_id}_with_dem.jp2')

        # Skip if output file already exists
        if os.path.exists(output_path):
            print(f"Skipping {img_id}, output file already exists.")
            continue

        if img_id in dem_files:
            dem_path = dem_files[img_id]
            print(f"Attempting to open DEM file: {dem_path}")

            try:
                # Open the img raster (JP2)
                with rasterio.open(img_path) as img_raster:
                    img_data = img_raster.read()  # Reading all bands of the image
                    img_profile = img_raster.profile
                    img_crs = img_raster.crs
                    img_transform = img_raster.transform
                    img_shape = img_data.shape[1:]  # The shape of one band (height, width)

                    # Open the dem raster (JP2)
                    try:
                        with rasterio.open(dem_path) as dem_raster:
                            dem_data = dem_raster.read(1)  # Reading the DEM data
                            dem_crs = dem_raster.crs
                            dem_transform = dem_raster.transform

                            # Check if CRS matches, if not reproject DEM to IMG CRS
                            if dem_crs != img_crs:
                                print(f"Reprojecting DEM {dem_path} to match IMG {img_path} CRS")
                                dem_data = reproject_dem_to_img_crs(
                                    dem_data, dem_transform, dem_crs, img_crs, img_transform, img_shape
                                )

                            # Check if DEM and IMG have the same shape, if not, resample DEM to match IMG shape
                            if dem_data.shape != img_shape:
                                print(f"Resampling DEM {dem_path} to match IMG {img_path} dimensions")
                                dem_data = reproject_dem_to_img_crs(
                                    dem_data, dem_transform, dem_crs, img_crs, img_transform, img_shape
                                )

                            # Mask DEM values where IMG has no data (0)
                            dem_data = np.where(img_data[0] == 0, 0, dem_data)

                            #avg_elevation = calculate_average_elevation_below_ndvi_threshold(img_data, dem_data, ndvi_band=4, threshold=10000)
                            #print(f"Average elevation of water areas: {avg_elevation}")

                            # Subtract the average elevation from the DEM
                            #dem_data -= avg_elevation

                            # Stack the 5 bands of IMG data with the reprojected DEM data as the last (6th) band
                            new_bands = np.vstack([img_data, dem_data[np.newaxis, ...]])

                            # Update the profile for the output (increase count to 6 bands)
                            img_profile.update(count=new_bands.shape[0], driver='JP2OpenJPEG')

                            # Write the output file
                            with rasterio.open(output_path, 'w', **img_profile) as dst:
                                dst.write(new_bands)
                                print(f'Successfully wrote {output_path}')

                    except Exception as e:
                        print(f"Failed to open DEM file {dem_path}: {e}")
                        error_ids.append(img_id)  # Log the failed image ID
                        continue  # Skip to the next image

            except Exception as e:
                print(f"Failed to open image file {img_path}: {e}")
                error_ids.append(img_id)  # Log the failed image ID
                continue  # Skip to the next image
        else:
            print(f"No DEM found for {img_id}")

    # Print the IDs of images that encountered errors
    if error_ids:
        print("\nErrors occurred for the following image IDs:")
        for error_id in error_ids:
            print(error_id)
    else:
        print("\nNo errors occurred.")

# Define folder paths
img_folder = "/isipd/projects/p_planetdw/data/dw_detection/PlanetScope/images/2023"
dem_folder = "/isipd/projects/p_planetdw/data/mosaicing/arcticdem/DEM"
output_folder = "/isipd/projects/p_planetdw/data/dw_detection/PlanetScope/images/2023_dem"

# Run the function
add_dem_to_img(img_folder, dem_folder, output_folder)
