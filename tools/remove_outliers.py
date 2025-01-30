import geopandas as gpd
from tqdm import tqdm
from shapely.geometry import MultiPolygon, Polygon

location_shape_filter = False

output_path = "/isipd/projects/p_planetdw/data/outputs/dw_prob0506_filtered_maxwater03_presenceFalse_v2.gpkg"

# Paths to the GeoPackage files
gdf1_path = "/isipd/projects/p_planetdw/data/outputs/dw_all_prob_05_aoi.gpkg"
water_mask_path = "/isipd/projects/p_planetdw/data/auxilliary/WaterMask/water.gpkg"
roads_path = "/isipd/projects/p_planetdw/data/auxilliary/roads.gpkg"
presence_path = '/isipd/projects/p_planetdw/data/outputs/larger_prob06.gpkg'

# Load the GeoDataFrames
print("Loading GeoDataFrames...")
intersection = gpd.read_file(gdf1_path)
water_mask = gpd.read_file(water_mask_path)
roads = gpd.read_file(roads_path)
presence = gpd.read_file(presence_path)
print("GeoDataFrames loaded successfully.")

if location_shape_filter:
    print('Extracting polygons from 0.5 prob based on presence from 0.6 prob...')
    intersection = gpd.sjoin(intersection, presence, how="inner", predicate="intersects")
    intersection = intersection[intersection.columns]

# Check and align CRS
if water_mask.crs.is_geographic:
    print("Reprojecting water mask to a projected CRS...")
    water_mask = water_mask.to_crs(epsg=32606)
    print("Reprojection completed.")
else:
    print("Water mask is already in a projected CRS.")

intersection = intersection.to_crs(water_mask.crs)
roads = roads.to_crs(water_mask.crs)
print("CRS alignment completed.")

# Filter invalid geometries
intersection = intersection[intersection.is_valid]
water_mask = water_mask[water_mask.is_valid]
roads = roads[roads.is_valid]

# Build spatial indexes
print("Building spatial indexes...")
water_mask_sindex = water_mask.sindex
roads_sindex = roads.sindex
print("Spatial indexes built.")

# Calculate overlap with water mask and filter by roads
print("Processing features...")
intersection['water_overlap_area'] = intersection.geometry.apply(
    lambda geom: water_mask.iloc[list(water_mask_sindex.intersection(geom.bounds))]
    .intersection(geom)
    .area.sum()
)
intersection['overlap_percentage'] = intersection['water_overlap_area'] / intersection.geometry.area

# Check intersection with roads
intersection['road_intersects'] = intersection.geometry.apply(
    lambda geom: roads.iloc[list(roads_sindex.intersection(geom.bounds))].intersects(geom).any()
)

# Filter based on overlap and road intersections
intersection_filtered = intersection[
    (intersection['overlap_percentage'] <= 0.3) & (~intersection['road_intersects'])
]
print("Feature processing completed.")

# Compute features
def compute_features(gdf):
    gdf = gdf.copy()  # Ensure a deep copy
    gdf.loc[:, 'area'] = gdf.geometry.area
    gdf.loc[:, 'perimeter'] = gdf.geometry.length
    gdf.loc[:, 'compactness'] = (4 * 3.1416 * gdf['area']) / (gdf['perimeter'] ** 2)
    gdf.loc[:, 'bounding_box_area'] = gdf.geometry.bounds.apply(
        lambda bounds: (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]), axis=1
    )
    gdf.loc[:, 'aspect_ratio'] = gdf.geometry.bounds.apply(
        lambda bounds: (bounds[2] - bounds[0]) / (bounds[3] - bounds[1])
        if (bounds[3] - bounds[1]) != 0 else 0, axis=1
    )
    gdf.loc[:, 'convex_hull_area'] = gdf.geometry.convex_hull.area
    gdf.loc[:, 'solidity'] = gdf['area'] / gdf['convex_hull_area']
    gdf.loc[:, 'vertices_per_area'] = gdf.geometry.apply(
        lambda geom: len(geom.exterior.coords) if geom.type == "Polygon" else 0
    ) / gdf['area']
    gdf.loc[:, 'num_holes'] = gdf.geometry.apply(
        lambda geom: len(geom.interiors) if isinstance(geom, Polygon) else sum(len(poly.interiors) for poly in geom.geoms)
    )
    return gdf

# Apply feature computation
print("Computing features...")
gdf_test = compute_features(intersection_filtered)

# Filter inliers based on number of holes
gdf_inliers = gdf_test[gdf_test['num_holes'] <= 3]

# Save results
print("Saving filtered GeoDataFrame...")
gdf_inliers.to_file(output_path, driver="GPKG")
print(f"Filtered GeoDataFrame saved to: {output_path}")
