import os
import shutil
import geopandas as gpd
import random
from tqdm import tqdm  

# Define paths
grid_path = r'p:/cstadie/dw_data/utm_mosaic/grids/00_processing_grid.gpkg'
file_type = '.jp2'
out_dir = r'p:/cstadie/dw_data/utm_mosaic/mosaic/param_test_images'
source_base_path = r'p:/cstadie/dw_data/utm_mosaic/mosaic'

# Read grid
grid = gpd.read_file(grid_path)

# Define parameters
start_year = 2019
end_year = 2024
step = 2

def sample_id(group):
    return group.sample(n=2)

# Sample random images
random_images = grid.groupby('ZONE').apply(sample_id).reset_index(drop=True)
grid_ids = random_images['id'].tolist()

years = range(start_year, end_year, step)
years_list = list(years)

# Clear the output directory
if os.path.exists(out_dir):
    for filename in os.listdir(out_dir):
        file_path = os.path.join(out_dir, filename)
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)
        else:
            os.remove(file_path)
else:
    os.makedirs(out_dir)

# Initialize tqdm progress bar
print("Starting file copying process...")
for grid_id in tqdm(grid_ids, desc='Copying Files', unit='file'):
    year = random.choice(years_list)
    image_name = f'ps_PSScene4Band_{year}_{grid_id}_hist_composite'
    image_path = os.path.join(source_base_path, str(year), f'{image_name}{file_type}')
    destination_path = os.path.join(out_dir, f'{image_name}{file_type}')

    print(f'Checking source path: {image_path}')
    print(f'Checking destination path: {destination_path}')
    
    if os.path.exists(image_path):
        shutil.copy2(image_path, destination_path)
        tqdm.write(f'Finished copying {image_name} to {destination_path}')
    else:
        tqdm.write(f'File not found: {image_path}')

print("Finished copying process.")