import ee
import geopandas as gpd

output_file = '/isipd/projects/p_planetdw/data/outputs/larger_prob06_morph_filtered_disk7_waterfiltered.gpkg'

# Authenticate using the browser flow
ee.Authenticate(auth_mode='notebook')

# Initialize
ee.Initialize(project='landsatsentinelmosaics')

gdf = gpd.read_file('/isipd/projects/p_planetdw/data/outputs/larger_prob06_morph_filtered_disk7.gpkg')

def filter_remove_water(gdf, threshold=0.2):

    

    # convert gdf to ee FC
    rts_ee = ee.FeatureCollection(gdf.__geo_interface__)
    # load gee layer
    esri_lulc2020 = ee.ImageCollection('projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m')
    # filter to rts footprint
    filtered = esri_lulc2020.filterBounds(rts_ee).mosaic()
    # get binary water mask
    water_layer = filtered.eq(1)
    data_mask = water_layer.mask().eq(0)
    # create final water mask : either water or nodata (assumed that no data is over the sea)
    water_mask = water_layer.unmask().Or(data_mask)
    # reduce regions and get value
    reduced = ee.Image.reduceRegions(water_mask, rts_ee, reducer=ee.Reducer.mean(), scale=10)
    # convert to gdf
    gdf_out = ee.data.computeFeatures({'expression': reduced, 'fileFormat': 'GEOPANDAS_GEODATAFRAME'})
    # filter to no water
    gdf_filtered = gdf.loc[gdf_out.query(f'mean <= {threshold}').index]

    return gdf_filtered

filtered = filter_remove_water(gdf, threshold=0.2)

filtered.to_file(output_file, driver='GPKG')