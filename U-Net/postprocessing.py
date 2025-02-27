import os
import math
import glob
import time
import multiprocessing
from datetime import timedelta

import geopandas as gpd
import rasterio
from tqdm import tqdm
from osgeo import ogr, gdal
from shapely.geometry import shape, Polygon
from rasterio.windows import Window
from rasterio.features import shapes

# Disable warnings
gdal.UseExceptions()


def create_vector_vrt(vrt_out_fp, layer_fps, out_layer_name="trees", pbar=False):
    """Create an OGR virtual vector file for merging multiple vector files."""
    if len(layer_fps) == 0:
        return print(f"Warning! Attempt to create empty VRT file, skipping: {vrt_out_fp}")

    xml = f'<OGRVRTDataSource>\n' \
          f'    <OGRVRTUnionLayer name="{out_layer_name}">\n'
    for layer_fp in tqdm(layer_fps, desc="Creating VRT", disable=not pbar):
        shapefile = ogr.Open(layer_fp)
        layer = shapefile.GetLayer()
        relative_path = layer_fp.replace(f"{os.path.join(os.path.dirname(vrt_out_fp), '')}", "")
        xml += f'        <OGRVRTLayer name="{os.path.basename(layer_fp).split(".")[0]}">\n' \
               f'            <SrcDataSource relativeToVRT="1">{relative_path}</SrcDataSource>\n' \
               f'            <SrcLayer>{layer.GetName()}</SrcLayer>\n' \
               f'            <GeometryType>wkb{ogr.GeometryTypeToName(layer.GetGeomType())}</GeometryType>\n' \
               f'        </OGRVRTLayer>\n'
    xml += '    </OGRVRTUnionLayer>\n' \
           '</OGRVRTDataSource>\n'
    with open(vrt_out_fp, "w") as file:
        file.write(xml)


def polygonize_chunk(params):
    """Polygonize a single raster chunk."""
    raster_fp, out_fp, window = params
    polygons = []
    with rasterio.open(raster_fp) as src:
        raster_crs = src.crs
        for feature, _ in shapes(src.read(window=window), src.read(window=window), 4,
                                 src.window_transform(window)):
            polygons.append(shape(feature))
    if len(polygons) > 0:
        gpd.GeoDataFrame({"geometry": polygons}).to_file(out_fp, driver="GPKG", crs=raster_crs, layer="trees")
        return out_fp


def create_polygons(raster_dir, polygons_basedir, config):
    """Polygonize the raster predictions into vector files."""
    raster_fps = glob.glob(f"{raster_dir}/*.tif")
    for raster_fp in tqdm(raster_fps, desc="Polygonizing raster predictions"):

        prediction_name = os.path.splitext(os.path.basename(raster_fp))[0]
        polygons_dir = os.path.join(polygons_basedir, prediction_name)
        if os.path.exists(polygons_dir):
            print(f"Skipping, already processed {polygons_dir}")
            continue
        os.makedirs(os.path.join(polygons_dir, "vrtdata"), exist_ok=True)

        chunk_windows = []
        n_rows, n_cols = config.postproc_gridsize
        with rasterio.open(raster_fp) as raster:
            width, height = math.ceil(raster.width / n_cols), math.ceil(raster.height / n_rows)
        for i in range(n_rows):
            for j in range(n_cols):
                out_fp = os.path.join(polygons_dir, "vrtdata", f"{prediction_name}_{width * j}_{height * i}.gpkg")
                chunk_windows.append([raster_fp, out_fp, Window(width * j, height * i, width, height)])

        polygon_fps = []
        with multiprocessing.Pool(processes=config.postproc_workers) as pool:
            with tqdm(total=len(chunk_windows), desc="Polygonizing raster chunks", position=1, leave=False) as pbar:
                for out_fp in pool.imap_unordered(polygonize_chunk, chunk_windows):
                    if out_fp:
                        polygon_fps.append(out_fp)
                    pbar.update()

        create_vector_vrt(os.path.join(polygons_dir, f"polygons_{prediction_name}.vrt"), polygon_fps)


def dissolve_polygons(input_fp, output_fp, min_area_threshold):
    """Dissolve overlapping polygons, sum their areas, and filter by minimum area."""
    print("\nDissolving polygons...\n")

    polygons = gpd.read_file(input_fp)
    dissolved = polygons.dissolve(by=None, aggfunc={'area': 'sum'})
    dissolved_filtered = dissolved[dissolved['area'] > min_area_threshold]

    dissolved_filtered.to_file(output_fp, driver="GPKG", layer="trees")
    print(f"Dissolved polygons saved to {output_fp}")

    return output_fp


def filter_polygons(input_fp, output_fp, water_mask_fp, roads_fp, water_overlap_threshold):
    """Filter polygons by water overlap and road intersection and compute shape features."""
    print("\nFiltering extracted polygons...\n")

    # Load data
    intersection = gpd.read_file(input_fp)
    water_mask = gpd.read_file(water_mask_fp)
    roads = gpd.read_file(roads_fp)

    # Ensure CRS alignment
    water_mask = water_mask.to_crs(intersection.crs)
    roads = roads.to_crs(intersection.crs)

    # Calculate overlap with water mask and filter by roads
    print("Processing features...")
    intersection['water_overlap_area'] = intersection.geometry.apply(
        lambda geom: water_mask[water_mask.intersects(geom)].intersection(geom).area.sum()
    )
    intersection['overlap_percentage'] = intersection['water_overlap_area'] / intersection.geometry.area

    # Check intersection with roads
    intersection['road_intersects'] = intersection.geometry.apply(
        lambda geom: roads.intersects(geom).any()
    )

    # Filter polygons: Keep those with ≤ water_overlap_threshold and that don't intersect roads
    intersection_filtered = intersection[
        (intersection['overlap_percentage'] <= water_overlap_threshold) & (~intersection['road_intersects'])
    ]

    # Save results
    intersection_filtered.to_file(output_fp, driver="GPKG", layer="trees")
    print(f"Filtered polygons saved to {output_fp}")

    return output_fp


def postprocess_all(config):
    """Run full postprocessing pipeline."""
    
    print("Starting postprocessing for all probability thresholds.")
    start = time.time()

    for threshold in config.threshold_list:
        prob_folder = os.path.join(config.predictions_base_dir, f"prob{int(threshold * 100):02d}")
        if not os.path.exists(prob_folder):
            print(f"Skipping {prob_folder}, folder does not exist.")
            continue

        print(f"\nProcessing postprocessing for threshold {threshold} at {prob_folder}\n")

        rasters_dir = os.path.join(prob_folder, "rasters")
        polygons_dir = os.path.join(prob_folder, "polygons")
        intersected_dir = os.path.join(config.predictions_base_dir, "intersected_polygons")

        os.makedirs(intersected_dir, exist_ok=True)

        # Polygonization
        if config.create_polygons:
            create_polygons(rasters_dir, polygons_dir, config)

        # Extract intersecting polygons
        intersecting_fp = os.path.join(intersected_dir, "intersecting_polygons.gpkg")
        intersecting_fp = dissolve_polygons(polygons_dir, intersecting_fp, config.min_area_threshold)

        if not intersecting_fp:
            continue

        # Apply final filtering
        final_polygons_fp = os.path.join(intersected_dir, "final_polygons.gpkg")
        final_polygons_fp = filter_polygons(
            intersecting_fp, final_polygons_fp, config.water_mask_fp, config.roads_fp, config.water_overlap_threshold
        )

    print(f"\nPostprocessing completed in {str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n")
