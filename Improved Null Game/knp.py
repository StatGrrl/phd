"""
Module docstring
"""

import geopandas as gpd

# EPSG:4326, WGS 84 World
default_crs = {'init': 'epsg:4326'}

# EPSG:2054, Hartebeesthoek94, WGS 84 South Africa 30-32 degrees E
proj_crs = {'init': 'epsg:2054'}

boundary = gpd.read_file('knp/shapefiles/bndry_kruger.shp')

# geographic public
roads_all = gpd.read_file('knp/shapefiles/infra_roads.shp')
roads_all = roads_all.to_crs(default_crs)
roads_main = gpd.read_file('knp/shapefiles/roads_tarred_public.shp')
rivers_all = gpd.read_file('knp/shapefiles/rivers_all.shp')
rivers_main = gpd.read_file('knp/shapefiles/rivers_main.shp')
dams = gpd.read_file('knp/shapefiles/water_dams_geo.shp')
dams_col = dams.shape[1]
dams = gpd.sjoin(dams, boundary)
dams = dams.iloc[:, 0:dams_col]
geology = gpd.read_file('knp/shapefiles/Geology.shp')

trees_lt = ['LE03', 'LE04', 'LE07', 'PA01', 'PA03', 'PA04', 'PA05', 'PA06',
            'SA02', 'SA03', 'SA04', 'SP01']
trees_index = [x in trees_lt for x in geology['LT_ID']]
trees = geology[trees_index]

mountains_lt = ['BU01', 'KL01', 'KL02', 'LE07', 'PA01', 'PA03', 'PA04', 'PA06',
                'PH04', 'SP01', 'SP02', 'SP04']
mountains_index = [x in mountains_lt for x in geology['LT_ID']]
mountains = geology[mountains_index]

# park public
zone1 = gpd.read_file('knp/shapefiles/CPZ.shp')
zone1['Zone'] = 'CPZ'
zone2 = gpd.read_file('knp/shapefiles/JPZ.shp')
zone2['Zone'] = 'JPZ'
zone3 = gpd.read_file('knp/shapefiles/IPZ.shp')
zone3['Zone'] = 'IPZ'
zones = zone1.append(zone2)
zones = zones.append(zone3)
camps_all = gpd.read_file('knp/shapefiles/camps_ESRI.shp')
camps_main = gpd.read_file('knp/shapefiles/camps_main.shp')
picnic_spots = gpd.read_file('knp/shapefiles/picnic_spots.shp')
gates = gpd.read_file('knp/shapefiles/gates_public.shp')
springs = gpd.read_file('knp/shapefiles/springs.shp')
water_holes = gpd.read_file('knp/shapefiles/water_holes.shp')
drinking_troughs = gpd.read_file('knp/shapefiles/drinking_trough.shp', crs=default_crs)

springs['Type'] = 'Spring'
water = springs.iloc[:, [1, 2, 11, 12]]
water_holes['Type'] = 'Water hole'
water = water.append(water_holes.iloc[:, [1, 2, 10, 11]])
drinking_troughs['Type'] = 'Drinking trough'
water = water.append(drinking_troughs.iloc[:, [1, 2, 10, 11]])
water_col = water.shape[1]
water = gpd.sjoin(water, boundary)
water = water.iloc[:, 0:water_col]
