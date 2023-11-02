"""
Module docstring
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# EPSG:4326, WGS 84 World
default_crs ='epsg:4326'

# EPSG:2054, Hartebeesthoek94, WGS 84 South Africa 30-32 degrees E
proj_crs = 'epsg:2054'

folder = 'shapefiles/knp/'

boundary = gpd.read_file('zip://' + folder + 'boundary_kruger.zip', crs=default_crs)  # polygon
boundary = boundary.to_crs(default_crs)

# geographic public
rivers_main = gpd.read_file('zip://' + folder + 'rivers_main.zip', crs=default_crs)  # lines
roads_main = gpd.read_file('zip://' + folder + 'roads_tarred_public.zip', crs=default_crs)  # lines
dams = gpd.read_file('zip://' + folder + 'water_dams_geo.zip', crs=default_crs)  # points
dams_col = dams.shape[1]
dams = gpd.sjoin(dams, boundary)
dams = dams.iloc[:, 0:dams_col]
geology = gpd.read_file('zip://' + folder + 'geology.zip', crs=default_crs)  # polygons

trees_lt = ['LE03', 'LE04', 'LE07', 'PA01', 'PA03', 'PA04', 'PA05', 'PA06',
            'SA02', 'SA03', 'SA04', 'SP01']
trees_index = [x in trees_lt for x in geology['LT_ID']]
trees = geology[trees_index]

mountains_lt = ['BU01', 'KL01', 'KL02', 'LE07', 'PA01', 'PA03', 'PA04', 'PA06',
                'PH04', 'SP01', 'SP02', 'SP04']
mountains_index = [x in mountains_lt for x in geology['LT_ID']]
mountains = geology[mountains_index]

lebombo_lt = ['KL01', 'KL02', 'SP01', 'SP02', 'SP04']
lebombo = mountains[mountains['LT_ID'].isin(lebombo_lt)]

# park public
camps_main = gpd.read_file('zip://' + folder + 'camps_main.zip', crs=default_crs)  # points
picnic_spots = gpd.read_file('zip://' + folder + 'picnic_spots.zip', crs=default_crs)  # points
gates = gpd.read_file('zip://' + folder + 'gates_public.zip', crs=default_crs)  # points
springs = gpd.read_file('zip://' + folder + 'springs.zip', crs=default_crs)  # points
springs = springs.to_crs(default_crs)
water_holes = gpd.read_file('zip://' + folder + 'water_holes.zip', crs=default_crs)  # points
water_holes = water_holes.to_crs(default_crs)
drinking_troughs = gpd.read_file('zip://' + folder + 'drinking_troughs.zip', crs=default_crs)  # points
drinking_troughs = drinking_troughs.to_crs(default_crs)

springs['Type'] = 'Spring'
water_holes['Type'] = 'Water hole'
drinking_troughs['Type'] = 'Drinking trough'
water = pd.concat([springs.iloc[:, [1, 2, 11, 12]], water_holes.iloc[:, [1, 2, 10, 11]], 
                   drinking_troughs.iloc[:, [1, 2, 10, 11]]], axis=0)
water_col = water.shape[1]
water = gpd.sjoin(water, boundary)
water = water.iloc[:, 0:water_col]

landscapes = gpd.read_file('zip://' + folder + 'landscapes.zip', crs=default_crs)  # polygons

# white rhino landscape preferences
prefer = [3, 11, 13]
prefer_index = [x in prefer for x in landscapes['LSCAP_ID']]
w_rhino_prefer = landscapes[prefer_index]
avoid = [4, 23, 25, 26, 28, 32, 33]
avoid_index = [x in avoid for x in landscapes['LSCAP_ID']]
w_rhino_avoid = landscapes[avoid_index]

landscapes['White Rhino'] = 'Neutral'
landscapes.loc[prefer_index, 'White Rhino'] = 'Prefer'
landscapes.loc[avoid_index, 'White Rhino'] = 'Avoid'

# color maps
import numpy as np
import matplotlib.colors as colors
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
greys = truncate_colormap(plt.get_cmap('Greys_r'), 0.3, 0.7)

# plot regions
x_adj = -0.15
y_adj = -0.05
river_bounds = ['Limpopo', 'Sabie', 'Crocodile', 'Olifants', 'Nsikazi', 'Luvuvhu']
rivers = rivers_main[rivers_main['RIVERNAME'].isin(river_bounds)]

# plot white rhino landscape preferences
fig, ax = plt.subplots(1, figsize=(3, 6))
landscapes.plot(ax=ax, column='White Rhino', legend=False, cmap='RdYlGn')
boundary.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5)
ax.annotate('X', xy=(31.27, -25.25), color='black', size=8, weight='bold')
ax.xaxis.set_tick_params(labelbottom=False)
ax.yaxis.set_tick_params(labelleft=False)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig('knp/white_rhino_landscape.pdf', dpi=1200, bbox_inches='tight')

# main vegetation types
ne = ['KL01', 'KL02', 'LE01', 'LE02', 'LE03', 'LE04', 'LE05', 'LE06', 'NW01', 'NW02', 'PA03', 'PA04', 'PA05', 'PA06']
nw = ['BU01', 'BU02', 'BU03', 'LE07', 'PA01', 'PA02', 'PH01', 'PH02', 'PH03', 'PH04', 'PH05', 'PH06', 'PH07', 'PH08', 'PH09', 'PH10', 'PH11', 'PH12']
se = ['SA01', 'SA02', 'SA03', 'SA04', 'SA06', 'SP01', 'SP02', 'SP03', 'SP04', 'VU01']
sw = ['MA01', 'MA02', 'SA05', 'SK01', 'SK02', 'SK03', 'SK04', 'SK05', 'SK06', 'SK07', 'SK08', 'SK09', 'SK10', 'SK11']

ne_index = [x in ne for x in geology['LT_ID']]
nw_index = [x in nw for x in geology['LT_ID']]
se_index = [x in se for x in geology['LT_ID']]
sw_index = [x in sw for x in geology['LT_ID']]
ne = geology[ne_index]
nw = geology[nw_index]
se = geology[se_index]
sw = geology[sw_index]
ne['region'] = 'NE'
nw['region'] = 'NW'
se['region'] = 'SE'
sw['region'] = 'SW'
geology_regions = pd.concat([ne, nw, se, sw], axis=0)
geology_regions = geology_regions.dissolve(by='region')
geology_regions['reg'] = ['NE', 'NW', 'SE', 'SW']
centroids = geology_regions.centroid

xadj = -0.05
yadj = -0.1
fig, ax = plt.subplots(1, figsize=(3, 6))
geology_regions.plot(ax=ax, column='reg', legend=False, cmap=greys)
boundary.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5)
for i in range(len(centroids)):
    ax.annotate(str(centroids.index[i]), xy=(centroids.iloc[i].coords[0][0] + xadj, centroids.iloc[i].coords[0][1] + yadj))
ax.xaxis.set_tick_params(labelbottom=False)
ax.yaxis.set_tick_params(labelleft=False)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig('knp/veg_main.pdf', dpi=1200, bbox_inches='tight')
plt.close()

xadj = 0
yadj = 0
centroids = landscapes.centroid
fig, ax = plt.subplots(1, figsize=(3, 6))
landscapes.plot(ax=ax, column='ALLIANCES', legend=False)
for i in range(len(centroids)):
    ax.annotate(str(landscapes['ALLIANCES_'][i]), xy=(centroids.iloc[i].coords[0][0] + xadj, centroids.iloc[i].coords[0][1] + yadj), fontsize=6)
ax.xaxis.set_tick_params(labelbottom=False)
ax.yaxis.set_tick_params(labelleft=False)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig('knp/kruger_landscape.pdf', dpi=1200, bbox_inches='tight')
