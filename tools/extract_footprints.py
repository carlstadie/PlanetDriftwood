import os
import geopandas as gpd
import rasterio
from rasterio.features import shapes

# Define input and output directories
folder_path = '/home/rscph/Desktop/images/2019'
output_folder = '/home/rscph/Desktop/mosaic_footprints/2019'

# Get a list of all .jp2 files in the folder
jp2_files = [f for f in os.listdir(folder_path) if f.endswith('.jp2')]

# Get a list of all already processed files in the output folder
processed_files = [f.replace('_footprints', '').replace('.gpkg', '') for f in os.listdir(output_folder) if
                   f.endswith('.gpkg')]

# Iterate over each JP2 file
for jp2_file in jp2_files:
    jp2_path = os.path.join(folder_path, jp2_file)

    # Check if the file has already been processed
    if jp2_file in processed_files:
        print(f"Skipping {jp2_file}, already processed.")
        continue

    with rasterio.open(jp2_path) as dataset:
        print('Processing:', jp2_file)

        # Read the mask from the specified band (band 1 in this case)
        mask = dataset.read_masks(1)

        # Extract CRS
        crs = dataset.crs

        # Convert raster to vector shapes
        print('Converting raster to shapes...')
        vectorized_shapes = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) in enumerate(shapes(mask, mask=None, transform=dataset.transform))
        )

        # Create a GeoDataFrame from the vectorized shapes
        print('Creating GeoDataFrame...')
        gdf = gpd.GeoDataFrame.from_features(vectorized_shapes, crs=crs)

        # Define output path including the CRS for clarity
        output_file = os.path.join(output_folder, f'footprints_{jp2_file}_{crs.to_string()}.gpkg')

        # Export to GeoPackage
        print('Exporting to:', output_file)
        gdf.to_file(output_file, layer='countries', driver="GPKG")

        print(f"Finished processing {jp2_file} and saved valid shapes.")
