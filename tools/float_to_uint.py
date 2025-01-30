import os 
import glob
import rasterio
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

out_dir = '/isipd/projects/p_planetdw/data/dw_detection/PlanetScope/results/allbool/rasters'

def convert_and_save_tiff(file):
    out_file = os.path.join(out_dir, os.path.basename(file))
    with rasterio.open(file) as src:
        image = src.read(1)
        image = np.nan_to_num(image, nan=0)
        image = image.astype(np.uint8)
        
        out_meta = src.meta.copy()
        out_meta.update({
            'dtype': np.uint8,
            'compress': 'lzw'
        })
        
        with rasterio.open(out_file, 'w', **out_meta) as dst:
            dst.write(image, 1)

input_dir = '/isipd/projects/p_planetdw/data/dw_detection/PlanetScope/results/all/rasters'
files = glob.glob(os.path.join(input_dir, '*.tif'))

with Pool(processes=32) as pool:
    with tqdm(total=len(files), desc=f"Building COGs", position=0, leave=True) as pb:
        for _, result in enumerate(pool.imap_unordered(convert_and_save_tiff, files, chunksize=1)):
            pb.update()