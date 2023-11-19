from framework import *
import knp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.colors as colors
import contextily as ctx
import warnings

warnings.simplefilter(action='ignore')


# color maps
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


greens = truncate_colormap(plt.get_cmap('Greens'), 0.3, 0.9)
blues = truncate_colormap(plt.get_cmap('Blues'), 0.3, 0.9)
reds = truncate_colormap(plt.get_cmap('Reds'), 0.3, 0.9)
oranges = truncate_colormap(plt.get_cmap('Oranges'), 0.3, 0.9)
purples = truncate_colormap(plt.get_cmap('Purples'), 0.3, 0.9)


# EXAMPLE 1

# Kruger National Park
kruger_subarea = np.array([31.3, -24.75, 32.05, -24.35])
kruger_park = Park('Kruger National Park', knp.camps_main, knp.picnic_spots, knp.gates, knp.water,
                   subarea=kruger_subarea)
subarea_poly = bounds_to_polygon(kruger_subarea, kruger_park.default_crs)

grid_square = kruger_park.grid
grid_park = kruger_park.bound_grid
grid_border = kruger_park.border_cells
grid_edge = kruger_park.edge_cells

# Poacher stay within predefined area
poacher_area = Point(31.6, -24.5).buffer(0.3)
poacher_df = pd.DataFrame({'Type': 'Poacher home', 'Area': poacher_area}, index=[0])
poacher_home = gpd.GeoDataFrame(poacher_df, geometry='Area', crs=kruger_park.default_crs)
poacher0a = Poacher('poacher0a', kruger_park, 'full', 'random', within=[poacher_home])
poacher_within = poacher0a.allowed_cells

# Poacher stay out of cells with geographical obstacles
out = ['rivers', 'dams', 'mountains', 'trees']
poacher0b = Poacher('poacher0b', kruger_park, 'full', 'random', out=out)
poacher_out = poacher0b.allowed_cells

# random poacher, within and out areas
poacher0 = Poacher('poacher0', kruger_park, 'full', 'random', out=out, within=[poacher_home])

# Poacher dislikes
poacher1 = Poacher('poacher1', kruger_park, 'full', 'strategic', dislike={'roads': 5000})
poacher_roads = poacher1.allowed_cells
poacher2 = Poacher('poacher2', kruger_park, 'full', 'strategic', dislike={'camps': 8000, 'picnic_spots': 8000})
poacher_camps = poacher2.allowed_cells
poacher3 = Poacher('poacher4', kruger_park, 'full', 'strategic', dislike={'gates': 10000})
poacher_gates = poacher3.allowed_cells
poacher4 = Poacher('poacher4', kruger_park, 'full', 'strategic',
                   dislike={'roads': 5000, 'camps': 8000, 'picnic_spots': 8000, 'gates': 10000})
poacher_dislikes = poacher4.allowed_cells

# Poacher likes
poacher6 = Poacher('poacher6', kruger_park, 'full', 'strategic', like={'border': 30000})
poacher_border = poacher6.allowed_cells
poacher7 = Poacher('poacher7', kruger_park, 'full', 'strategic', like={'dams': 25000})
poacher_dams = poacher7.allowed_cells
poacher8 = Poacher('poacher8', kruger_park, 'full', 'strategic', like={'water': 20000})
poacher_water = poacher8.allowed_cells
poacher9 = Poacher('poacher9', kruger_park, 'full', 'strategic',
                   like={'border': 30000, 'dams': 25000, 'water': 20000})
poacher_likes = poacher9.allowed_cells

# Poacher all allowed cells and their probabilities
poacher10 = Poacher('poacher10', kruger_park, 'bound', 'strategic', within=[poacher_home],
                    out=['rivers', 'dams', 'mountains', 'trees'],
                    dislike={'roads': 5000, 'camps': 8000, 'picnic_spots': 8000, 'gates': 10000},
                    like={'border': 30000, 'dams': 25000, 'water': 20000})
poacher_cells = poacher10.allowed_cells

rhino_rand = Wildlife('rhino_rand', kruger_park, 'bound', 'random', out=out)
rhino_strat = Wildlife('rhino_strat', kruger_park, 'bound', 'strategic', out=out,
                       dislike={'camps': 2000},
                       like={'dams': 10000, 'rivers': 10000, 'water': 5000})
ranger_rand = Ranger('ranger_rand', kruger_park, 'full', 'random', out=out)

# PLOTS

# Plot map of park
fig0, ax = plt.subplots(1, figsize=(10, 10))
kruger_park.boundary.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5)
kruger_park.trees.plot(ax=ax, facecolor='olive', edgecolor='none', alpha=0.3, label='dense trees')
kruger_park.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5, label='steep mountains')
kruger_park.roads.plot(ax=ax, color='black', linewidth=1, linestyle='dashed', alpha=0.6, label='tarred roads')
kruger_park.rivers.plot(ax=ax, color='blue', linewidth=0.7, linestyle='solid', alpha=0.6, label='main rivers')
kruger_park.dams.plot(ax=ax, edgecolor='none', color='black', marker='o', markersize=15, label='dams')
kruger_park.water.plot(ax=ax, edgecolor='none', color='royalblue', marker='D', markersize=15, alpha=0.5,
                       label='fountains, water holes & drinking troughs')
kruger_park.camps.plot(ax=ax, edgecolor='none', color='indigo', marker='p', markersize=25, label='main camps')
kruger_park.picnic_spots.plot(ax=ax, edgecolor='none', color='darkgreen', marker='^', markersize=20,
                              label='picnic spots')
kruger_park.gates.plot(ax=ax, edgecolor='none', color='crimson', marker='s', markersize=15, label='public gates')
left, right = plt.xlim()
ax.set_xlim(left, right + 1.25)
# ax.axis('off')
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
leg_item = [Patch(edgecolor='none', facecolor='olive', alpha=0.3, label='dense trees'),
            Patch(edgecolor='none', facecolor='brown', alpha=0.5, label='steep mountains'),
            Line2D([0], [0], color='black', linewidth=1, linestyle='dashed', alpha=0.6, label='tarred roads'),
            Line2D([0], [0], color='blue', linewidth=0.7, linestyle='solid', alpha=0.6, label='main rivers'),
            Line2D([0], [0], color='white', markerfacecolor='black', marker='o', markersize=6, label='dams'),
            Line2D([0], [0], color='white', markerfacecolor='royalblue', marker='D', markersize=6, alpha=0.5,
                   label='water holes, fountains\n& drinking troughs'),
            Line2D([0], [0], color='white', markerfacecolor='indigo', marker='p', markersize=8, label='main camps'),
            Line2D([0], [0], color='white', markerfacecolor='darkgreen', marker='^', markersize=7.5,
                   label='picnic spots'),
            Line2D([0], [0], color='white', markerfacecolor='crimson', marker='s', markersize=6, label='public gates')]
ax.legend(handles=leg_item, loc='upper right', frameon=False, fontsize='x-large')
plt.savefig('knp/thesis/4_1_2_park.pdf', dpi=1200, bbox_inches='tight')

# Plot subarea
fig1, ax = plt.subplots(1, figsize=(10, 10))
subarea_poly.plot(ax=ax, color='cyan', alpha=0.5, edgecolor='black', linewidth=2, linestyle='dashed')
kruger_park.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
ctx.add_basemap(ax, url=ctx.tile_providers.ST_TERRAIN, crs=kruger_park.default_crs, zoom=12)
# ax.axis('off')
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
plt.savefig('knp/thesis/4_1_2_subarea.pdf', dpi=600, bbox_inches='tight')

x_lim = (kruger_subarea[0] - 0.04, kruger_subarea[2] + 0.04)
y_lim = (kruger_subarea[1] - 0.04, kruger_subarea[3] + 0.04)

# Plot grid with geo features
fig, ax = plt.subplots(1, figsize=(15, 10))
kruger_park.grid.plot(ax=ax, facecolor='white', edgecolor='black', linewidth=0.5)
kruger_park.edge_cells.plot(ax=ax, facecolor='magenta', alpha=0.5)
kruger_park.border_cells.plot(ax=ax, facecolor='yellow', alpha=0.5)
kruger_park.trees.plot(ax=ax, facecolor='olive', edgecolor='none', alpha=0.5, label='trees')
kruger_park.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5, label='mountains')
kruger_park.rivers.plot(ax=ax, color='blue', linewidth=1.5, linestyle='solid', alpha=0.6, label='rivers')
kruger_park.dams.plot(ax=ax, color='black', marker='o', markersize=40, label='dams')
kruger_park.water.plot(ax=ax, color='royalblue', marker='D', markersize=40,
                       label='fountains, water holes & drinking troughs', alpha=0.5)
kruger_park.roads.plot(ax=ax, color='black', linewidth=2, linestyle='dashed', alpha=0.6, label='roads')
kruger_park.camps.plot(ax=ax, color='indigo', marker='p', markersize=50, label='camps')
kruger_park.picnic_spots.plot(ax=ax, color='darkgreen', marker='^', markersize=45, label='picnic spots')
kruger_park.gates.plot(ax=ax, color='crimson', marker='s', markersize=40, label='gates')
kruger_park.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
# ax.axis('off')
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
plt.savefig('knp/thesis/4_1_2_grid.pdf', dpi=1200, bbox_inches='tight')

# Plot grid with cell numbers
centroids = ranger_rand.allowed_cells['Centroids']
fig, ax = plt.subplots(1, figsize=(45, 30))
kruger_park.grid.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5)
kruger_park.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
kruger_park.edge_cells.plot(ax=ax, facecolor='magenta', alpha=0.5)
kruger_park.border_cells.plot(ax=ax, facecolor='yellow', alpha=0.5)
kruger_park.trees.plot(ax=ax, facecolor='olive', edgecolor='none', alpha=0.3, label='dense trees')
kruger_park.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5, label='steep mountains')
kruger_park.roads.plot(ax=ax, color='black', linewidth=2, linestyle='dashed', alpha=0.6, label='tarred roads')
kruger_park.rivers.plot(ax=ax, color='blue', linewidth=2, linestyle='solid', alpha=0.6, label='main rivers')
kruger_park.dams.plot(ax=ax, edgecolor='none', color='black', marker='o', markersize=180, label='dams')
kruger_park.water.plot(ax=ax, edgecolor='none', color='royalblue', marker='D', markersize=180, alpha=0.5,
                       label='fountains, water holes & drinking troughs')
kruger_park.camps.plot(ax=ax, edgecolor='none', color='indigo', marker='p', markersize=300, label='main camps')
kruger_park.picnic_spots.plot(ax=ax, edgecolor='none', color='darkgreen', marker='^', markersize=240,
                              label='picnic spots')
kruger_park.gates.plot(ax=ax, edgecolor='none', color='crimson', marker='s', markersize=180, label='public gates')
adj = -0.003
for i in range(len(centroids)):
    ax.annotate(str(centroids.index[i]), xy=(centroids.iloc[i].coords[0][0] + adj, centroids.iloc[i].coords[0][1]))
ax.axis('off')
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
plt.savefig('knp/eg1_cell_numbers.pdf', dpi=1200, bbox_inches='tight')

# Plot geographical obstacles and allowed cells
fig, ax = plt.subplots(1, figsize=(15, 10))
poacher_out.plot(ax=ax, color='grey', edgecolor='black', linewidth=0.5, alpha=0.3)
# kruger_park.edge_cells.plot(ax=ax, facecolor='magenta', alpha=0.5)
# kruger_park.border_cells.plot(ax=ax, facecolor='yellow', alpha=0.5)
kruger_park.trees.plot(ax=ax, facecolor='olive', edgecolor='none', alpha=0.5, label='trees')
kruger_park.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5, label='mountains')
kruger_park.rivers.plot(ax=ax, color='blue', linewidth=1.5, linestyle='solid', alpha=0.6, label='rivers')
kruger_park.dams.plot(ax=ax, color='black', marker='o', markersize=40, label='dams')
# kruger_park.water.plot(ax=ax, color='royalblue', marker='D', markersize=40,
#                        label='fountains, water holes & drinking troughs', alpha=0.5)
# kruger_park.roads.plot(ax=ax, color='black', linewidth=2, linestyle='dashed', alpha=0.6, label='roads')
# kruger_park.camps.plot(ax=ax, color='indigo', marker='p', markersize=50, label='camps')
# kruger_park.picnic_spots.plot(ax=ax, color='darkgreen', marker='^', markersize=45, label='picnic spots')
# kruger_park.gates.plot(ax=ax, color='crimson', marker='s', markersize=40, label='gates')
kruger_park.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
#subarea_poly.plot(ax=ax, facecolor='none', edgecolor='darkcyan', linewidth=2, linestyle='dashed')
# ax.axis('off')
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
plt.savefig('knp/thesis/4_1_2_area_out.pdf', dpi=1200, bbox_inches='tight')

# Plot predefined home area and allowed cells
fig2, ax = plt.subplots(1, figsize=(15, 10))
poacher_home.plot(ax=ax, facecolor='pink', edgecolor='hotpink', alpha=0.5, linewidth=2, linestyle='dotted')
poacher_within.plot(ax=ax, color='grey', edgecolor='black', linewidth=0.5, alpha=0.8)
kruger_park.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
subarea_poly.plot(ax=ax, facecolor='none', edgecolor='darkcyan', linewidth=2, linestyle='dashed')
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
# ax.axis('off')
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
plt.savefig('knp/thesis/4_1_2_area_within.pdf', dpi=1200, bbox_inches='tight')

# All dislikes
x_lim = (kruger_subarea[0], kruger_subarea[2])
y_lim = (kruger_subarea[1], kruger_subarea[3])

poacher_roads_max = poacher_roads[poacher_roads['Select Prob'] == 0.5]
poacher_roads_max = poacher_roads_max.dissolve(by='Select Prob', as_index=False)
poacher_roads_max = pd.concat([poacher_roads_max, poacher_roads[poacher_roads['Select Prob'] != 0.5]])

fig, ax = plt.subplots(1, figsize=(12, 10))
poacher_roads.plot(ax=ax, column='Select Prob', cmap=oranges, edgecolor='black', linewidth=0.2)
kruger_park.roads.plot(ax=ax, color='black', linewidth=2, linestyle='dashed', label='roads')
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
# ax.axis('off')
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
plt.savefig('knp/thesis/4_1_3_dislikes_roads.pdf', dpi=1200, bbox_inches='tight')

fig, ax = plt.subplots(1, figsize=(12, 10))
poacher_camps.plot(ax=ax, column='Select Prob', cmap=oranges, edgecolor='black', linewidth=0.2)
kruger_park.camps.plot(ax=ax, color='indigo', marker='p', markersize=50, label='camps')
kruger_park.picnic_spots.plot(ax=ax, color='darkgreen', marker='^', markersize=45, label='picnic spots')
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
# ax.axis('off')
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
plt.savefig('knp/thesis/4_1_3_dislikes_camps.pdf', dpi=1200, bbox_inches='tight')

fig, ax = plt.subplots(1, figsize=(12, 10))
poacher_gates.plot(ax=ax, column='Select Prob', cmap=oranges, edgecolor='black', linewidth=0.2)
kruger_park.gates.plot(ax=ax, color='crimson', marker='s', markersize=40, label='gates')
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
# ax.axis('off')
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
plt.savefig('knp/thesis/4_1_3_dislikes_gates.pdf', dpi=1200, bbox_inches='tight')

fig4, ax = plt.subplots(1, figsize=(12, 10))
poacher_dislikes.plot(ax=ax, column='Select Prob', cmap=oranges, edgecolor='black', linewidth=0.2)
kruger_park.roads.plot(ax=ax, color='black', linewidth=2, linestyle='dashed', label='roads')
kruger_park.camps.plot(ax=ax, color='indigo', marker='p', markersize=50, label='camps')
kruger_park.picnic_spots.plot(ax=ax, color='darkgreen', marker='^', markersize=45, label='picnic spots')
kruger_park.gates.plot(ax=ax, color='crimson', marker='s', markersize=40, label='gates')
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
# ax.axis('off')
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
plt.savefig('knp/thesis/4_1_3_dislikes_all.pdf', dpi=1200, bbox_inches='tight')

# All likes
fig, ax = plt.subplots(1, figsize=(12, 10))
poacher_border.plot(ax=ax, column='Select Prob', cmap=purples, edgecolor='black', linewidth=0.2)
kruger_park.border_line.plot(ax=ax, linewidth=1, linestyle='solid', color='red')
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
# ax.axis('off')
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
plt.savefig('knp/thesis/4_1_3_likes_border.pdf', dpi=1200, bbox_inches='tight')

fig, ax = plt.subplots(1, figsize=(12, 10))
poacher_dams.plot(ax=ax, column='Select Prob', cmap=purples, edgecolor='black', linewidth=0.2)
kruger_park.dams.plot(ax=ax, color='black', marker='o', markersize=40, label='dams')
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
# ax.axis('off')
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
plt.savefig('knp/thesis/4_1_3_likes_dams.pdf', dpi=1200, bbox_inches='tight')

fig, ax = plt.subplots(1, figsize=(12, 10))
poacher_water.plot(ax=ax, column='Select Prob', cmap=purples, edgecolor='black', linewidth=0.2)
kruger_park.water.plot(ax=ax, color='royalblue', marker='D', markersize=40,
                       label='fountains, water holes & drinking troughs')
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
# ax.axis('off')
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
plt.savefig('knp/thesis/4_1_3_likes_water.pdf', dpi=1200, bbox_inches='tight')

fig5, ax = plt.subplots(1, figsize=(12, 10))
poacher_likes.plot(ax=ax, column='Select Prob', cmap=purples, edgecolor='black', linewidth=0.2)
kruger_park.border_line.plot(ax=ax, linewidth=1, linestyle='solid', color='red')
kruger_park.dams.plot(ax=ax, color='black', marker='o', markersize=40, label='dams')
kruger_park.water.plot(ax=ax, color='royalblue', marker='D', markersize=40,
                       label='fountains, water holes & drinking troughs')
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
# ax.axis('off')
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
plt.savefig('knp/thesis/4_1_3_likes_all.pdf', dpi=1200, bbox_inches='tight')

# Poacher cells within, out and selection probabilities for all likes and dislikes
x_lim = (kruger_subarea[0] - 0.04, kruger_subarea[2] + 0.04)
y_lim = (kruger_subarea[1] - 0.04, kruger_subarea[3] + 0.04)

fig13, ax = plt.subplots(1, figsize=(15, 10))
poacher_cells.plot(ax=ax, column='Select Prob', cmap='Reds', edgecolor='black', linewidth=0.2)
kruger_park.trees.plot(ax=ax, facecolor='olive', edgecolor='none', alpha=0.5, label='trees')
kruger_park.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5, label='mountains')
kruger_park.roads.plot(ax=ax, color='black', linewidth=2, linestyle='dashed', alpha=0.6, label='roads')
kruger_park.rivers.plot(ax=ax, color='blue', linewidth=1.5, linestyle='solid', alpha=0.6, label='rivers')
kruger_park.dams.plot(ax=ax, color='black', marker='o', markersize=40, label='dams')
kruger_park.water.plot(ax=ax, color='royalblue', marker='D', markersize=40,
                       label='fountains, water holes & drinking troughs', alpha=0.5)
kruger_park.camps.plot(ax=ax, color='indigo', marker='p', markersize=50, label='camps')
kruger_park.picnic_spots.plot(ax=ax, color='darkgreen', marker='^', markersize=45, label='picnic spots')
kruger_park.gates.plot(ax=ax, color='crimson', marker='s', markersize=40, label='gates')
kruger_park.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
subarea_poly.plot(ax=ax, facecolor='none', edgecolor='darkcyan', linewidth=2, linestyle='dashed')
poacher_home.plot(ax=ax, facecolor='none', edgecolor='hotpink', alpha=0.5, linewidth=2, linestyle='dotted')
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
# ax.axis('off')
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
plt.savefig('knp/thesis/4_1_3_selection_weights.pdf', dpi=1200, bbox_inches='tight')

# Rhino allowed cells and selection probabilities for all likes and dislikes
x_lim = (kruger_subarea[0] - 0.04, kruger_subarea[2] + 0.04)
y_lim = (kruger_subarea[1] - 0.04, kruger_subarea[3] + 0.04)

fig, ax = plt.subplots(1, figsize=(15, 10))
rhino_strat.allowed_cells.plot(ax=ax, column='Select Prob', cmap='Greens', edgecolor='black', linewidth=0.5)
kruger_park.trees.plot(ax=ax, facecolor='olive', edgecolor='none', alpha=0.5, label='trees')
kruger_park.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5, label='mountains')
kruger_park.rivers.plot(ax=ax, color='blue', linewidth=1.5, linestyle='solid', alpha=0.6, label='rivers')
kruger_park.dams.plot(ax=ax, color='black', marker='o', markersize=40, label='dams')
kruger_park.water.plot(ax=ax, color='royalblue', marker='D', markersize=40,
                       label='fountains, water holes & drinking troughs', alpha=0.5)
kruger_park.roads.plot(ax=ax, color='black', linewidth=2, linestyle='dashed', alpha=0.6, label='roads')
kruger_park.camps.plot(ax=ax, color='indigo', marker='p', markersize=50, label='camps')
kruger_park.picnic_spots.plot(ax=ax, color='darkgreen', marker='^', markersize=45, label='picnic spots')
kruger_park.gates.plot(ax=ax, color='crimson', marker='s', markersize=40, label='gates')
kruger_park.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
subarea_poly.plot(ax=ax, facecolor='none', edgecolor='cyan', linewidth=2, linestyle='dashed')
poacher_home.plot(ax=ax, facecolor='none', edgecolor='black', alpha=0.5, linewidth=1, linestyle='dotted')
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
# ax.axis('off')
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
plt.savefig('knp/thesis/4_1_4_rhino_selection_weights.pdf', dpi=1200, bbox_inches='tight')


# Trajectories

x_lim = (kruger_subarea[0] - 0.02, kruger_subarea[2] - 0.02)
y_lim = (kruger_subarea[1], kruger_subarea[3])

rhino_start = 1804
ranger_start = 423
poacher_start = 1755
rhino_dest = 1275
ranger_dest = 1511
poacher_dest = 1572

rhino_rand.start_cell = rhino_rand.allowed_cells.loc[rhino_start, ]
rhino_strat.start_cell = rhino_strat.allowed_cells.loc[rhino_start, ]
ranger_rand.start_cell = ranger_rand.allowed_cells.loc[ranger_start, ]
poacher0.start_cell = poacher0.allowed_cells.loc[poacher_start, ]
poacher10.start_cell = poacher10.allowed_cells.loc[poacher_start, ]
rhino_rand.heading_cell = rhino_rand.allowed_cells.loc[rhino_dest, ]
rhino_strat.heading_cell = rhino_strat.allowed_cells.loc[rhino_dest, ]
ranger_rand.heading_cell = ranger_rand.allowed_cells.loc[ranger_dest, ]
poacher0.heading_cell = poacher0.allowed_cells.loc[poacher_dest, ]
poacher10.heading_cell = poacher10.allowed_cells.loc[poacher_dest, ]

# seed_list = np.random.randint(101, 666666, 500)
# seed_list = [8754, 15106, 35228, 35833, 51967, 151210, 157517, 169435, 172741, 174541, 193550, 209601, 215680, 227084,
#              238059, 250418, 267783, 270565, 271370, 273751, 305681, 317283, 339602, 350517, 367540, 379217, 380687,
#              392985, 405433, 422553, 426060, 432383, 433234, 441986, 448841, 448960, 451103, 452718, 471711, 492652,
#              499957, 500856, 501330, 520441, 533414, 571066, 572071, 600689, 625132]
seed_list = [250418]
for seed in seed_list:
    game1 = Game('geo', rhino_rand, ranger_rand, poacher0, end_moves=300, games_pm=1, months=1, seed=seed,
                 rtn_moves=True, rtn_traj=True)
    game2 = Game('poacher', rhino_rand, ranger_rand, poacher10, end_moves=300, games_pm=1, months=1, seed=seed,
                 rtn_moves=True, rtn_traj=True)
    game3 = Game('animal', rhino_strat, ranger_rand, poacher10, end_moves=300, games_pm=1, months=1, seed=seed,
                 rtn_moves=True, rtn_traj=True)
    game_list = [game1, game2, game3]
    i = 1
    for game in game_list:
        i += 1
        sim = sim_games(game)
        traj = sim['trajectories'][0].trajectories
        gdf = traj[2].df
        leave = gdf[gdf['Leave'] > 0]
        capture = gdf[gdf['Capture'] > 0]
        poach = gdf[gdf['Poach'] > 0]
        fig, ax = plt.subplots(1, figsize=(15, 10))
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        # ax.axis('off')
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        #ctx.add_basemap(ax, url=ctx.tile_providers.OSM_A, crs=kruger_park.default_crs, zoom=12, zorder=0)
        poacher0.allowed_cells.plot(ax=ax, color='none', edgecolor='grey', linewidth=0.3, zorder=1)
        traj[0].plot(ax=ax, column='Trajectory', cmap=greens, linewidth=3, label='rhino', zorder=2)
        traj[1].plot(ax=ax, column='Trajectory', cmap=blues, linewidth=3, label='ranger', zorder=3)
        traj[2].plot(ax=ax, column='Trajectory', cmap=reds, linewidth=3, label='poacher', zorder=4)
        rhino_rand.allowed_cells.loc[[rhino_start], ].plot(ax=ax, color='green', zorder=5)
        ranger_rand.allowed_cells.loc[[ranger_start], ].plot(ax=ax, color='cornflowerblue', zorder=5)
        poacher0.allowed_cells.loc[[poacher_start], ].plot(ax=ax, color='red', zorder=5)
        rhino_rand.allowed_cells.loc[[rhino_start], ].set_geometry('Centroids').plot(ax=ax, color='black', markersize=80,
                                                                                     marker='$W$', zorder=6)
        ranger_rand.allowed_cells.loc[[ranger_start], ].set_geometry('Centroids').plot(ax=ax, color='black', markersize=80,
                                                                                       marker='$R$', zorder=6)
        poacher0.allowed_cells.loc[[poacher_start], ].set_geometry('Centroids').plot(ax=ax, color='black', markersize=80,
                                                                                     marker='$P$', zorder=6)
        if len(capture) > 0:
            capture.plot(ax=ax, color='black', marker='*', markersize=250, zorder=7, edgecolor='white')
        if len(leave) > 0:
            leave.plot(ax=ax, color='black', marker='P', markersize=200, zorder=7, edgecolor='white')
        if len(poach) > 0:
            poach.plot(ax=ax, color='black', marker='X', markersize=200, zorder=7, edgecolor='white')
        kruger_park.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
        # kruger_park.trees.plot(ax=ax, facecolor='olive', edgecolor='none', alpha=0.5, label='trees')
        # kruger_park.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5, label='mountains')
        # kruger_park.rivers.plot(ax=ax, color='blue', linewidth=1, linestyle='solid', alpha=0.6, label='rivers')
        # kruger_park.dams.plot(ax=ax, color='black', marker='o', markersize=40, label='dams')
        # kruger_park.water.plot(ax=ax, color='royalblue', marker='D', markersize=40,
        #                        label='fountains, water holes & drinking troughs', alpha=0.5)
        kruger_park.roads.plot(ax=ax, color='black', linewidth=1, linestyle='dashed', alpha=0.6, label='roads')
        kruger_park.camps.plot(ax=ax, color='indigo', marker='p', markersize=50, label='camps')
        kruger_park.picnic_spots.plot(ax=ax, color='darkgreen', marker='^', markersize=45, label='picnic spots')
        kruger_park.gates.plot(ax=ax, color='crimson', marker='s', markersize=40, label='gates')

        #plt.savefig('knp/traj/_eg1_' + str(seed) + game.name + '.pdf', dpi=600, bbox_inches='tight')
        plt.savefig('knp/thesis/4_1_' + str(i) + '_traj.pdf', dpi=1200, bbox_inches='tight')

ranger_rand.sampling = 'max'
poacher0.sampling = 'max'
poacher10.sampling = 'max'
ranger_rand.stay_in_cell = False
poacher0.stay_in_cell = False
poacher10.stay_in_cell = False

seed_list = [250418]
for seed in seed_list:
    game1 = Game('geo', rhino_rand, ranger_rand, poacher0, end_moves=300, games_pm=1, months=1, seed=seed,
                 rtn_moves=True, rtn_traj=True)
    game2 = Game('poacher', rhino_rand, ranger_rand, poacher10, end_moves=300, games_pm=1, months=1, seed=seed,
                 rtn_moves=True, rtn_traj=True)
    game3 = Game('animal', rhino_strat, ranger_rand, poacher10, end_moves=300, games_pm=1, months=1, seed=seed,
                 rtn_moves=True, rtn_traj=True)
    game_list = [game1, game2, game3]
    i = 1
    for game in game_list:
        i += 1
        sim = sim_games(game)
        traj = sim['trajectories'][0].trajectories
        gdf = traj[2].df
        leave = gdf[gdf['Leave'] > 0]
        capture = gdf[gdf['Capture'] > 0]
        poach = gdf[gdf['Poach'] > 0]
        fig, ax = plt.subplots(1, figsize=(15, 10))
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        # ax.axis('off')
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        #ctx.add_basemap(ax, url=ctx.tile_providers.OSM_A, crs=kruger_park.default_crs, zoom=12, zorder=0)
        poacher0.allowed_cells.plot(ax=ax, color='none', edgecolor='grey', linewidth=0.3, zorder=1)
        traj[0].plot(ax=ax, column='Trajectory', cmap=greens, linewidth=3, label='rhino', zorder=2)
        traj[1].plot(ax=ax, column='Trajectory', cmap=blues, linewidth=3, label='ranger', zorder=3)
        traj[2].plot(ax=ax, column='Trajectory', cmap=reds, linewidth=3, label='poacher', zorder=4)
        rhino_rand.allowed_cells.loc[[rhino_start], ].plot(ax=ax, color='green', zorder=5)
        ranger_rand.allowed_cells.loc[[ranger_start], ].plot(ax=ax, color='cornflowerblue', zorder=5)
        poacher0.allowed_cells.loc[[poacher_start], ].plot(ax=ax, color='red', zorder=5)
        rhino_rand.allowed_cells.loc[[rhino_start], ].set_geometry('Centroids').plot(ax=ax, color='black', markersize=80,
                                                                                     marker='$W$', zorder=6)
        ranger_rand.allowed_cells.loc[[ranger_start], ].set_geometry('Centroids').plot(ax=ax, color='black', markersize=80,
                                                                                       marker='$R$', zorder=6)
        poacher0.allowed_cells.loc[[poacher_start], ].set_geometry('Centroids').plot(ax=ax, color='black', markersize=80,
                                                                                     marker='$P$', zorder=6)
        if len(capture) > 0:
            capture.plot(ax=ax, color='black', marker='*', markersize=250, zorder=7, edgecolor='white')
        if len(leave) > 0:
            leave.plot(ax=ax, color='black', marker='P', markersize=200, zorder=7, edgecolor='white')
        if len(poach) > 0:
            poach.plot(ax=ax, color='black', marker='X', markersize=200, zorder=7, edgecolor='white')
        kruger_park.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
        # kruger_park.trees.plot(ax=ax, facecolor='olive', edgecolor='none', alpha=0.5, label='trees')
        # kruger_park.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5, label='mountains')
        # kruger_park.rivers.plot(ax=ax, color='blue', linewidth=1, linestyle='solid', alpha=0.6, label='rivers')
        # kruger_park.dams.plot(ax=ax, color='black', marker='o', markersize=40, label='dams')
        # kruger_park.water.plot(ax=ax, color='royalblue', marker='D', markersize=40,
        #                        label='fountains, water holes & drinking troughs', alpha=0.5)
        kruger_park.roads.plot(ax=ax, color='black', linewidth=1, linestyle='dashed', alpha=0.6, label='roads')
        kruger_park.camps.plot(ax=ax, color='indigo', marker='p', markersize=50, label='camps')
        kruger_park.picnic_spots.plot(ax=ax, color='darkgreen', marker='^', markersize=45, label='picnic spots')
        kruger_park.gates.plot(ax=ax, color='crimson', marker='s', markersize=40, label='gates')
        plt.savefig('knp/thesis/4_1_' + str(i) + '_traj_max.pdf', dpi=1200, bbox_inches='tight')
