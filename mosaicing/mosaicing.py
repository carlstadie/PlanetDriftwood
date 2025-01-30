import os 
import glob 
from tqdm import tqdm
from collections import Counter

import rasterio
import signal
from contextlib import contextmanager
from osgeo import gdal
import pandas as pd
import geopandas as gpd 
from shapely.geometry import shape, Point, MultiPoint, Polygon, MultiPolygon, LineString, MultiLineString, \
    GeometryCollection, mapping as shape2json, box

from histmatch import create_histmatch_mosaic

pd.options.mode.chained_assignment = None


class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def get_planet_files(id, planet_dir):

    planet_files = []
    row, col = id.split(',')
    if len(row) >= 3:
        fn_pattern = '*' + ('-' if '-' in row else '0') + '0' + (row[-3:].replace('-','0')) + \
                '_' + ('-' if '-' in col else '0') + '00' + ('0'+col[-1] if len(col)==1 else col[-2:].replace('-','0')) + '*super.gpkg'    
    else:
        fn_pattern = '*' + ('-' if '-' in row else '0') + '00' + ('0'+row[-1] if len(row)==1 else row[-2:].replace('-','0')) + \
                '_' + ('-' if '-' in col else '0') + '00' + ('0'+col[-1] if len(col)==1 else col[-2:].replace('-','0')) + '*super.gpkg'
    # import ipdb; ipdb.set_trace()
    fn = glob.glob(os.path.join(planet_dir, fn_pattern))
    if len(fn) == 1:
        planet_files.append(fn[0])
    return planet_files


def get_intersection(old_footprints, new_footprint):

    for index, alist in enumerate(old_footprints):
        if index == 0:
            gdf = gpd.read_file(alist)
            gdf = gdf[['id', 'geometry']]
        else:
            gdf_new = gpd.read_file(alist)
            gdf_new = gdf_new[['id', 'geometry']]
            gdf = gdf.append(gdf_new)
    gdf.reset_index(inplace=True, drop=True)

    overlay = gpd.overlay(gdf, new_footprint)
    return overlay


def clip_mosaic(out_fn, in_fn, boundary_shp):
    options = dict(
        format="GTiff",
        multithread=True,
        cutlineDSName=boundary_shp,
        resampleAlg=gdal.GRA_NearestNeighbour,  # GRA_NearestNeighbour,GRA_NearestNeighbour
        cropToCutline=True,
        warpMemoryLimit=1000000000,
        creationOptions=['BIGTIFF=IF_SAFER', 'NUM_THREADS=ALL_CPUS'],
    )
    print('Clipping...')
    out_fn = out_fn.replace('.jp2', '.tif')
    gdal.Warp(out_fn, in_fn, **options)

    options = dict(
        format="JP2OpenJPEG",
        creationOptions=["QUALITY=80", "NUM_THREADS=ALL_CPUS"],
        callback_data=tqdm(total=1, position=0, leave=True, desc=f"Merging mosaic for {out_fn}")
    )
    print('Compressing...')
    gdal.Translate(out_fn.replace('.tif', '.jp2'), out_fn, **options)
    os.system(f"rm {out_fn}")


def clean_temp_fd(temp_fd):
    # Clean up temp files
    flist = os.listdir(temp_fd)
    temp_files = [os.path.join(temp_fd, i) for i in flist]
    for fp in temp_files:
        if os.path.exists(fp):
            os.system(f'rm -rf {fp}')
            #os.remove(fp)


def mosaic_img(in_fd, out_file):
    in_fns = [os.path.join(in_fd, i) for i in os.listdir(in_fd)]
    gdal.BuildVRT('/vsimem/merged.vrt', in_fns)

    options = dict(
        format="GTiff",
        creationOptions=["NUM_THREADS=ALL_CPUS"],
        callback_data=tqdm(total=100, position=0, leave=True, desc=f"Merging tiles...")
    )
    gdal.Translate(out_file, '/vsimem/merged.vrt', **options)


def generate_new_mosaics():
    '''generate new mosaics based on the new grid'''

    year = '2021'
    utm_zone = '20'
    utm_grid = '/media/rscph/274b2768-5356-4929-90a9-91f99c20da4c/dw_data/utm_mosaic/grids/carl_na_new.gpkg'
    raw_footprints = glob.glob(f'/media/rscph/274b2768-5356-4929-90a9-91f99c20da4c/dw_data/utm_mosaic/grids/AOI_utm{utm_zone}_{year}*.gpkg')[0]
    raw_dir = '/media/rscph/274b2768-5356-4929-90a9-91f99c20da4c/Planet_images_raw'
    temp_hist_dir = f'/media/rscph/274b2768-5356-4929-90a9-91f99c20da4c/dw_data/utm_mosaic/temp_hist_dir/{utm_zone}_{year}'
    temp_dir = f'/media/rscph/274b2768-5356-4929-90a9-91f99c20da4c/dw_data/utm_mosaic/temp_dir/{utm_zone}_{year}'
    out_dir = f'/media/rscph/274b2768-5356-4929-90a9-91f99c20da4c/dw_data/utm_mosaic/mosaic/{year}/{utm_zone}'

    histmatch_refimages_basedir = '/media/rscph/274b2768-5356-4929-90a9-91f99c20da4c/dw_data/utm_mosaic/utmbasemap'
    scene_pattern = "_3B_AnalyticMS_SR_clip.tif"

    gdf_rf = gpd.read_file(raw_footprints)
    gdf_rf = gdf_rf[['id', 'geometry']]
    gdf_rf.columns = ['id_rf', 'geometry']
    gdf_utm = gpd.read_file(utm_grid, bbox=gdf_rf.geometry)
    gdf_utm = gdf_utm[['id', 'geometry']]
    gdf_utm.columns = ['id_utm', 'geometry']
    
    gdf_intersects = gpd.overlay(gdf_rf, gdf_utm, how='intersection').groupby('id_utm')
    fails = []
    for id_utm, group in tqdm(gdf_intersects, total=len(gdf_intersects)):
        ps_name = 'ps_PSScene4Band' + '_' + year + '_' + id_utm + '_hist_composite.tif'
        out_temp = os.path.join(temp_dir, ps_name)
        out_mosaic = os.path.join(out_dir, ps_name)
        new_footprint = gdf_utm[gdf_utm.id_utm == id_utm]
        
        if os.path.isfile(out_mosaic.replace('.tif', '.jp2')):
            continue
        
        raw_scene_patterns = group.id_rf.to_list()
        raw_scene_paths = []
        for i in raw_scene_patterns:
            # raw_scene_paths.extend(glob.glob(os.path.join('/mnt/data/planet_raw_download/images_N61W165', i + scene_pattern)))
            raw_scene_paths.extend(glob.glob(os.path.join(raw_dir, year, '*', '*', 'PSScene', i + scene_pattern)))
        for raw_scene_path in raw_scene_paths:
            try:
                rasterio.open(raw_scene_path)
            except:
                print(f'corrupted file {raw_scene_path}')
                raw_scene_paths.remove(raw_scene_path)

        if len(raw_scene_paths) != 0:
            print(f'find {len(raw_scene_paths)} files to mosaic to new grid {id_utm}')
            hist_ref = glob.glob(os.path.join(histmatch_refimages_basedir, '*' + f'{id_utm}' + '*.tif'))
            if len(hist_ref) == 0:
                print(f'no ref img for {id_utm}')
                continue
            else:
                hist_ref = hist_ref[0]
            
            
            projection_stats = []
            for i in raw_scene_paths:
                try:
                    projection_stats.append(gdal.Open(i).GetProjection().split('"')[-2])
                except Exception as err:
                    print(f"\nError! Error on getting projection {i}: ", err)
                    continue

            dstSRS = "EPSG:" + Counter(projection_stats).most_common()[0][0]
            print(f"Matching to:", dstSRS)  

            # import ipdb; ipdb.set_trace()

            try:
                with time_limit(600):
        
                    new_mosaic_fp, n_missing, n_error = create_histmatch_mosaic(ps_name, temp_dir, histmatch_refimages_basedir,
                                                                            scene_pattern, False, True, temp_hist_dir,
                                                                            32, 16, '.tif', print, raw_scene_paths, hist_ref, dstSRS)
            except:
                print(f'histmatch failed for {ps_name}')
                fails.append(id_utm)
                continue

            mosaic_img(temp_hist_dir, out_temp)
            clean_temp_fd(temp_hist_dir)

            new_footprint = new_footprint.to_crs(dstSRS)
            new_footprint.to_file(os.path.join(temp_dir, f'temp_{id_utm}.gpkg'), driver='GPKG')
            clip_mosaic(out_mosaic, out_temp, os.path.join(temp_dir, f'temp_{id_utm}.gpkg'))
            clean_temp_fd(temp_dir)

    print(f'processing of AOI_utm{utm_zone}_{year} finished with fails: ', fails)
    return  



if __name__ == '__main__':
    
    generate_new_mosaics()
