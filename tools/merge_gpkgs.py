import geopandas as gpd
import pandas as pd
import os
import shapely
from shapely.geometry import shape
from shapely.ops import transform
from tqdm import tqdm


def find_geopackages(directory):
    """ Recursively find all .gpkg files within the given directory. """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".gpkg"):
                yield os.path.join(root, file)


def convert_to_2d(geometry):
    """ Convert 3D geometries to 2D by dropping the Z coordinate. """
    if geometry.is_empty:
        return geometry
    elif hasattr(geometry, 'geoms'):
        # It's a GeometryCollection or similar
        return shapely.geometry.GeometryCollection([convert_to_2d(geom) for geom in geometry.geoms])
    else:
        # It's a single geometry (Point, LineString, Polygon, etc.)
        return transform(lambda x, y, z=None: (x, y), geometry)


def process_geopackage(file_path):
    """ Load a GeoPackage, reproject to EPSG 32612, compute the area of each feature, and return the GeoDataFrame. """
    gdf = gpd.read_file(file_path)

    # Convert all geometries to 2D
    gdf['geometry'] = gdf['geometry'].apply(convert_to_2d)

    # Ensure the GeoDataFrame has a valid CRS before transforming
    if gdf.crs is None:
        raise ValueError(f"The file {file_path} does not have a CRS defined.")

    # Simplify geometries to handle any potential issues
    gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.01, preserve_topology=True)

    # Reproject to EPSG 32612
    gdf = gdf.to_crs(epsg=32612)
    gdf['area'] = gdf.geometry.area  # Compute area
    return gdf


def main(directory, output_file):
    all_data = []

    # Find all GeoPackages and process each one with a progress bar
    gpkg_files = list(find_geopackages(directory))
    for gpkg_file in tqdm(gpkg_files, desc="Processing GeoPackages"):
        try:
            gdf = process_geopackage(gpkg_file)
            all_data.append(gdf)
        except Exception as e:
            print(f"Error processing {gpkg_file}: {e}")

    if all_data:
        # Concatenate all GeoDataFrames
        combined_gdf = gpd.GeoDataFrame(pd.concat(all_data, ignore_index=True))

        # Save to a new GeoPackage file
        combined_gdf.to_file(output_file, driver='GPKG')
        print(f"Output saved to {output_file}")
    else:
        print("No GeoPackage files found or no valid data to process.")


if __name__ == "__main__":
    directory = '/isipd/projects/p_planetdw/data/dw_detection/PlanetScope/results/valid_obs/polygons'  # Path to the directory containing GeoPackages
    output_file = '/isipd/projects/p_planetdw/data/outputs/valid_obs.gpkg'  # Path where the merged output should be saved
    main(directory, output_file)
