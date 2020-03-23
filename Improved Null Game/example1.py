from framework import *
import knp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import contextily as ctx

# EXAMPLE 1

# Kruger National Park
kruger_park = Park('Kruger National Park', knp.camps_main, knp.picnic_spots, knp.gates, knp.water)

# Create grid of subarea
kruger_subarea = np.array([31.3, -24.75, 32.05, -24.35])
subarea_poly = bounds_to_polygon(kruger_subarea, kruger_park.default_crs)
kruger_grid = kruger_park.grid(1000, 1000, kruger_subarea)
grid_square = kruger_grid['square']
grid_park = kruger_grid['park']
grid_border = kruger_grid['border']
grid_edge = kruger_grid['edge']

# Poacher stay within predefined area
poacher_area = Point(31.6, -24.5).buffer(0.3)
poacher_df = pd.DataFrame({'Type': 'Poacher home', 'Area': poacher_area}, index=[0])
poacher_home = gpd.GeoDataFrame(poacher_df, geometry='Area', crs=kruger_park.default_crs)
poacher0a = Poacher(1, 'random', within=[poacher_home])
poacher_within = poacher0a.allowed_cells(kruger_park, kruger_grid, 'square')

# Poacher stay out of cells with geographical obstacles
poacher0b = Poacher(1, 'random', out=['rivers', 'dams', 'mountains', 'trees'])
poacher_out = poacher0b.allowed_cells(kruger_park, kruger_grid, 'square')

# Poacher dislikes
poacher1 = Poacher(1, 'structured', dislike={'roads': 5000})
poacher_roads = poacher1.allowed_cells(kruger_park, kruger_grid, 'square')
poacher2 = Poacher(1, 'structured', dislike={'camps': 8000})
poacher_camps = poacher2.allowed_cells(kruger_park, kruger_grid, 'square')
poacher3 = Poacher(1, 'structured', dislike={'picnic_spots': 8000})
poacher_picnics = poacher3.allowed_cells(kruger_park, kruger_grid, 'square')
poacher4 = Poacher(1, 'structured', dislike={'gates': 10000})
poacher_gates = poacher4.allowed_cells(kruger_park, kruger_grid, 'square')
poacher5 = Poacher(1, 'structured', dislike={'roads': 5000, 'camps': 8000,
                                             'picnic_spots': 8000, 'gates': 10000})
poacher_dislikes = poacher5.allowed_cells(kruger_park, kruger_grid, 'square')

# Poacher likes
poacher6 = Poacher(1, 'structured', like={'border': 30000})
poacher_border = poacher6.allowed_cells(kruger_park, kruger_grid, 'square')
poacher7 = Poacher(1, 'structured', like={'dams': 25000})
poacher_dams = poacher7.allowed_cells(kruger_park, kruger_grid, 'square')
poacher8 = Poacher(1, 'structured', like={'water': 20000})
poacher_water = poacher8.allowed_cells(kruger_park, kruger_grid, 'square')
poacher9 = Poacher(1, 'structured', like={'border': 30000, 'dams': 25000, 'water': 20000})
poacher_likes = poacher9.allowed_cells(kruger_park, kruger_grid, 'square')

poacher_pref = pd.DataFrame(np.full(grid_square.shape[0], 1), index=grid_square.index, columns=['Baseline'])
poacher_pref['Dislike roads'] = poacher_roads['cells']['Select Prob']
poacher_pref['Dislike camps'] = poacher_camps['cells']['Select Prob']
poacher_pref['Dislike picnic'] = poacher_picnics['cells']['Select Prob']
poacher_pref['Dislike gates'] = poacher_gates['cells']['Select Prob']
poacher_pref['Dislike all'] = poacher_dislikes['cells']['Select Prob']
poacher_pref['Like border'] = poacher_border['cells']['Select Prob']
poacher_pref['Like dams'] = poacher_dams['cells']['Select Prob']
poacher_pref['Like water'] = poacher_water['cells']['Select Prob']
poacher_pref['Like all'] = poacher_likes['cells']['Select Prob']
poacher_pref.to_excel('knp/likes_dislikes.xlsx')

# Poacher all allowed cells and their probabilities
poacher10 = Poacher(1, 'structured', within=[poacher_home],
                    out=['rivers', 'dams', 'mountains', 'trees'],
                    dislike={'roads': 5000, 'camps': 8000, 'picnic_spots': 8000, 'gates': 10000},
                    like={'border': 30000, 'dams': 10000, 'water': 10000})
poacher_cells = poacher10.allowed_cells(kruger_park, kruger_grid, 'square')

fig, ax = plt.subplots(1, figsize=(15, 10))
poacher_cells['cells']['Select Prob'].sort_values().plot(ax=ax, use_index=False)
plt.savefig('knp/select_prob.png')

# PLOTS

# Plot map of park
fig0, ax = plt.subplots(1, figsize=(10, 10))
kruger_park.boundary.plot(ax=ax, color='none', edgecolor='black', linewidth=1)
kruger_park.trees.plot(ax=ax, facecolor='lightgreen', edgecolor='none', alpha=0.5, label='trees')
kruger_park.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5, label='mountains')
kruger_park.roads.plot(ax=ax, color='black', linewidth=1, linestyle='dashed', alpha=0.6, label='roads')
kruger_park.rivers.plot(ax=ax, color='blue', linewidth=1, linestyle='solid', alpha=0.6, label='rivers')
kruger_park.dams.plot(ax=ax, edgecolor='none', color='royalblue', marker='o', markersize=15, alpha=0.5,
                      label='dams')
kruger_park.water.plot(ax=ax, edgecolor='none', color='lightblue', marker='o', markersize=15, alpha=0.5,
                       label='fountains, water holes & drinking troughs')
kruger_park.camps.plot(ax=ax, edgecolor='none', color='darkorange', marker='^', markersize=20, label='camps')
kruger_park.picnic_spots.plot(ax=ax, edgecolor='none', color='darkgreen', marker='^', markersize=20, label='picnic spots')
kruger_park.gates.plot(ax=ax, edgecolor='none', color='red', marker='s', markersize=15, label='gates')
left, right = plt.xlim()
ax.set_xlim(left, right + 1.25)
ax.axis('off')
leg_item = [Patch(edgecolor='none', facecolor='lightgreen', alpha=0.5, label='trees'),
            Patch(edgecolor='none', facecolor='brown', alpha=0.5, label='mountains'),
            Line2D([0], [0], color='black', linewidth=1, linestyle='dashed', alpha=0.6, label='roads'),
            Line2D([0], [0], color='blue', linewidth=1, linestyle='solid', alpha=0.6, label='rivers'),
            Line2D([0], [0], color='white', markerfacecolor='royalblue', marker='o', markersize=6, alpha=0.5,
                   label='dams'),
            Line2D([0], [0], color='white', markerfacecolor='lightblue', marker='o', markersize=6, alpha=0.5,
                   label='fountains, water holes & drinking troughs'),
            Line2D([0], [0], color='white', markerfacecolor='darkorange', marker='^', markersize=7.5, label='camps'),
            Line2D([0], [0], color='white', markerfacecolor='darkgreen', marker='^', markersize=7.5,
                   label='picnic spots'),
            Line2D([0], [0], color='white', markerfacecolor='red', marker='s', markersize=6, label='gates')]
ax.legend(handles=leg_item, loc='upper right', frameon=False)
plt.savefig('knp/eg1_a_park.png', bbox_inches='tight')

# Plot grid of subarea
fig1a, ax = plt.subplots(1, figsize=(10, 10))
subarea_poly.plot(ax=ax, color='cyan', alpha=0.5)
kruger_park.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
ctx.add_basemap(ax, url=ctx.tile_providers.ST_TERRAIN, crs=kruger_park.default_crs, zoom=12)
ax.axis('off')
plt.savefig('knp/eg1_b_subarea.png', bbox_inches='tight')

x_lim = (kruger_subarea[0] - 0.03, kruger_subarea[2] + 0.03)
y_lim = (kruger_subarea[1] - 0.03, kruger_subarea[3] + 0.03)
fig1b, ax = plt.subplots(1, figsize=(15, 10))
kruger_park.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
kruger_park.trees.plot(ax=ax, facecolor='lightgreen', edgecolor='none', alpha=0.5, label='trees')
kruger_park.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5, label='mountains')
kruger_park.roads.plot(ax=ax, color='black', linewidth=1, linestyle='dashed', alpha=0.6, label='roads')
kruger_park.rivers.plot(ax=ax, color='blue', linewidth=1, linestyle='solid', alpha=0.6, label='rivers')
kruger_park.dams.plot(ax=ax, color='royalblue', marker='o', markersize=40, label='dams')
kruger_park.water.plot(ax=ax, color='lightblue', marker='o', markersize=40,
                       label='fountains, water holes & drinking troughs')
kruger_park.camps.plot(ax=ax, color='darkorange', marker='^', markersize=45, label='camps')
kruger_park.picnic_spots.plot(ax=ax, color='darkgreen', marker='^', markersize=45, label='picnic spots')
kruger_park.gates.plot(ax=ax, color='red', marker='s', markersize=40, label='gates')
grid_square.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5, alpha=0.7)
grid_border.plot(ax=ax, facecolor='yellow', alpha=0.5, label='border cells')
grid_edge.plot(ax=ax, facecolor='magenta', alpha=0.5, label='edge cells')
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.axis('off')
plt.savefig('knp/eg1_c_grid.png', bbox_inches='tight')

# Plot predefined home area and allowed cells
fig3, ax = plt.subplots(1, figsize=(15, 10))
subarea_poly.plot(ax=ax, facecolor='cyan', edgecolor='none', alpha=0.5)
poacher_home.plot(ax=ax, facecolor='pink', edgecolor='none', alpha=0.5)
kruger_park.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
poacher_within['cells'].plot(ax=ax, color='grey', edgecolor='black', linewidth=0.5, alpha=0.3)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.axis('off')
plt.savefig('knp/eg1_d_area_within.png', bbox_inches='tight')


# Plot geographical obstacles and allowed cells
fig2, ax = plt.subplots(1, figsize=(15, 10))
kruger_park.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
kruger_park.trees.plot(ax=ax, facecolor='lightgreen', edgecolor='none', alpha=0.5, label='trees')
kruger_park.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5, label='mountains')
kruger_park.roads.plot(ax=ax, color='black', linewidth=1, linestyle='dashed', label='roads')
kruger_park.rivers.plot(ax=ax, color='blue', linewidth=1, linestyle='solid', label='rivers')
kruger_park.dams.plot(ax=ax, color='royalblue', marker='o', markersize=40, label='dams')
poacher_out['cells'].plot(ax=ax, color='grey', edgecolor='black', linewidth=0.5, alpha=0.3)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.axis('off')
plt.savefig('knp/eg1_e_area_out.png', bbox_inches='tight')

# Plot probability heat maps of dislikes
# Roads 5km away
fig4, ax = plt.subplots(1, figsize=(15, 10))
poacher_roads['cells'].plot(ax=ax, column='Select Prob', cmap='Reds')
kruger_park.roads.plot(ax=ax, linewidth=1, linestyle='dashed', color='black')
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.axis('off')
plt.savefig('knp/eg1_f_dislikes_roads.png', bbox_inches='tight')

# Camps 8km away
fig5, ax = plt.subplots(1, figsize=(15, 10))
poacher_camps['cells'].plot(ax=ax, column='Select Prob', cmap='Reds')
kruger_park.camps.plot(ax=ax, edgecolor='none', color='darkorange', marker='^', markersize=45)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.axis('off')
plt.savefig('knp/eg1_g_dislikes_camps.png', bbox_inches='tight')

# Picnic Spots 8km away
fig6, ax = plt.subplots(1, figsize=(15, 10))
poacher_picnics['cells'].plot(ax=ax, column='Select Prob', cmap='Reds')
kruger_park.picnic_spots.plot(ax=ax, edgecolor='none', color='darkgreen', marker='^', markersize=45)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.axis('off')
plt.savefig('knp/eg1_h_dislikes_picnic.png', bbox_inches='tight')

# Gates 10km away
fig7, ax = plt.subplots(1, figsize=(15, 10))
poacher_gates['cells'].plot(ax=ax, column='Select Prob', cmap='Reds')
kruger_park.gates.plot(ax=ax, edgecolor='none', color='red', marker='s', markersize=40)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.axis('off')
plt.savefig('knp/eg1_i_dislikes_gate.png', bbox_inches='tight')

# All dislikes
fig8, ax = plt.subplots(1, figsize=(15, 10))
poacher_dislikes['cells'].plot(ax=ax, column='Select Prob', cmap='Reds')
kruger_park.roads.plot(ax=ax, linewidth=1, linestyle='dashed', color='black')
kruger_park.camps.plot(ax=ax, edgecolor='none', color='darkorange', marker='^', markersize=45)
kruger_park.picnic_spots.plot(ax=ax, edgecolor='none', color='darkgreen', marker='^', markersize=45)
kruger_park.gates.plot(ax=ax, edgecolor='none', color='red', marker='s', markersize=40)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.axis('off')
plt.savefig('knp/eg1_j_dislikes_all.png', bbox_inches='tight')

# Plot probability heat maps of likes
# Border 30km near
fig9, ax = plt.subplots(1, figsize=(15, 10))
poacher_border['cells'].plot(ax=ax, column='Select Prob', cmap='Greens')
kruger_park.border_line.plot(ax=ax, linewidth=1, linestyle='solid', color='red')
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.axis('off')
plt.savefig('knp/eg1_k_likes_border.png', bbox_inches='tight')

# Dams 25km near
fig10, ax = plt.subplots(1, figsize=(15, 10))
poacher_dams['cells'].plot(ax=ax, column='Select Prob', cmap='Greens')
kruger_park.dams.plot(ax=ax, color='royalblue', marker='o', markersize=40)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.axis('off')
plt.savefig('knp/eg1_l_likes_dams.png', bbox_inches='tight')

# Water 20km near
fig11, ax = plt.subplots(1, figsize=(15, 10))
poacher_water['cells'].plot(ax=ax, column='Select Prob', cmap='Greens')
kruger_park.water.plot(ax=ax, color='lightblue', marker='o', markersize=40)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.axis('off')
plt.savefig('knp/eg1_m_likes_water.png', bbox_inches='tight')

# All likes
fig12, ax = plt.subplots(1, figsize=(15, 10))
poacher_likes['cells'].plot(ax=ax, column='Select Prob', cmap='Greens')
kruger_park.border_line.plot(ax=ax, linewidth=1, linestyle='solid', color='red')
kruger_park.dams.plot(ax=ax, color='royalblue', marker='o', markersize=40)
kruger_park.water.plot(ax=ax, color='lightblue', marker='o', markersize=40)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.axis('off')
plt.savefig('knp/eg1_n_likes_all.png', bbox_inches='tight')

# Poacher cells within, out and selection probabilities for all likes and dislikes
fig13, ax = plt.subplots(1, figsize=(15, 10))
poacher_cells['cells'].plot(ax=ax, column='Select Prob', cmap='Purples',
                            edgecolor='black', linewidth=0.5)
kruger_park.trees.plot(ax=ax, facecolor='lightgreen', edgecolor='none', alpha=0.5, label='trees')
kruger_park.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5, label='mountains')
kruger_park.roads.plot(ax=ax, color='black', linewidth=1, linestyle='dashed', alpha=0.6, label='roads')
kruger_park.rivers.plot(ax=ax, color='blue', linewidth=1, linestyle='solid', alpha=0.6, label='rivers')
kruger_park.dams.plot(ax=ax, color='royalblue', marker='o', markersize=40, label='dams')
kruger_park.water.plot(ax=ax, color='lightblue', marker='o', markersize=40)
kruger_park.camps.plot(ax=ax, edgecolor='none', color='darkorange', marker='^', markersize=45)
kruger_park.picnic_spots.plot(ax=ax, edgecolor='none', color='darkgreen', marker='^', markersize=45)
kruger_park.gates.plot(ax=ax, edgecolor='none', color='red', marker='s', markersize=40)
kruger_park.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
subarea_poly.plot(ax=ax, color='none', edgecolor='cyan')
poacher_home.plot(ax=ax, color='none', edgecolor='pink')
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.axis('off')
plt.savefig('knp/eg1_o_area_dislikes_likes.png', bbox_inches='tight')
