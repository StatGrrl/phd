from framework import *
import knp
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
from datetime import datetime
import dill
import warnings
warnings.simplefilter(action='ignore')

# Example 2
park = Park('Kruger National Park', knp.camps_main, knp.picnic_spots, knp.gates, knp.water)
subarea = np.array([31.12, -25.3, 31.65, -25])
x_lim = (subarea[0] + 0.03, subarea[2] + 0.03)
y_lim = (subarea[1] + 0.03, subarea[3] + 0.03)

grid = park.grid(1500, 1500, subarea)
print('number grid cells: ', grid['square'].shape[0])

out = ['rivers', 'dams', 'mountains', 'trees']

rhino_rand = Wildlife(1, 'random', out=out, stay_in_cell=True)
rhino_rand_cells = rhino_rand.allowed_cells(park, grid, 'park')
rhino_rand_cell_num = rhino_rand_cells['cells'].shape[0]
print('number random rhino cells: ', rhino_rand_cell_num)

rhino_struct = Wildlife(1, 'structured', out=out, dislike={'camps': 1000},
                        like={'dams': 10000, 'water': 10000}, stay_in_cell=True)
rhino_struct_cells = rhino_struct.allowed_cells(park, grid, 'park')
rhino_struct_cell_num = rhino_struct_cells['cells'].shape[0]
print('number structured rhino cells: ', rhino_struct_cell_num)

ranger_rand = Ranger(1, 'random', out=out)
ranger_rand_cells = ranger_rand.allowed_cells(park, grid, 'park')
ranger_rand_cell_num = ranger_rand_cells['cells'].shape[0]
print('number random ranger cells: ', ranger_rand_cell_num)

ranger_game = Ranger(1, 'game', out=out, like={'dams': 10000, 'water': 10000},
                     geo_util_fctr=50, arrest_util=50)
ranger_game_cells = ranger_game.allowed_cells(park, grid, 'park')
ranger_game_cell_num = ranger_game_cells['cells'].shape[0]
print('number game ranger cells: ', ranger_rand_cell_num)

poacher_rand = Poacher(1, 'random', out=out)
poacher_rand_cells = poacher_rand.allowed_cells(park, grid, 'park')
poacher_rand_cell_num = poacher_rand_cells['cells'].shape[0]
print('number random poacher cells: ', poacher_rand_cell_num)

poacher_struct = Poacher(1, 'structured', out=out,
                         dislike={'roads': 3000, 'camps': 2000, 'gates': 5000},
                         like={'border': 20000}, entry_type='border')
poacher_struct_cells = poacher_struct.allowed_cells(park, grid, 'park')
poacher_struct_cell_num = poacher_struct_cells['cells'].shape[0]
print('number structured poacher cells: ', poacher_struct_cell_num)

poacher_game = Poacher(1, 'game', out=out,
                       dislike={'roads': 3000, 'camps': 2000, 'gates': 5000},
                       like={'border': 20000}, entry_type='border', geo_util_fctr=120, arrest_util=150)
poacher_game_cells = poacher_game.allowed_cells(park, grid, 'park')
poacher_game_cell_num = poacher_game_cells['cells'].shape[0]
print('number game poacher cells: ', poacher_game_cell_num, '\n')

# single games for plots

end = 200  # 200 x 1.5km = 300km ~ 3 days with short stops to rest and eat (+/- 5km/h -> 100km in 20h)
gpm = 10  # 3 x 10 = 30 days

t0 = datetime.now()
print(t0, '\nRunning single game for random rhino, random ranger and random poacher')
game1 = one_game(grid, rhino_rand, rhino_rand_cells, ranger_rand, ranger_rand_cells,
                 poacher_rand, poacher_rand_cells, end_moves=end)
t1 = datetime.now()
print('Finished process, time taken: ', t1-t0, '\n')
game1_traj = game1['trajectories'].trajectories
print(game1['paths'].groupby(['Player']).agg({'Trajectory': 'max'}), '\n')
print(game1['totals'], '/n')

t0 = datetime.now()
print(t0, '\nRunning single game for structured rhino, random ranger and structured poacher')
game2 = one_game(grid, rhino_struct, rhino_struct_cells, ranger_rand, ranger_rand_cells,
                 poacher_struct, poacher_struct_cells, end_moves=end)
t1 = datetime.now()
print('Finished process, time taken: ', t1-t0, '\n')
game2_traj = game2['trajectories'].trajectories
print(game2['paths'].groupby(['Player']).agg({'Trajectory': 'max'}), '\n')
print(game2['totals'], '/n')

t0 = datetime.now()
print(t0, '\nRunning single game for structured rhino, game ranger and structured poacher')
game3 = one_game(grid, rhino_struct, rhino_struct_cells, ranger_game, ranger_game_cells,
                 poacher_struct, poacher_struct_cells, end_moves=end)
t1 = datetime.now()
print('Finished process, time taken: ', t1-t0, '\n')
game3_traj = game3['trajectories'].trajectories
print(game3['paths'].groupby(['Player']).agg({'Trajectory': 'max'}), '\n')
print(game3['totals'], '/n')

t0 = datetime.now()
print(t0, '\nRunning single game for structured rhino, game ranger and game poacher')
game4 = one_game(grid, rhino_struct, rhino_struct_cells, ranger_game, ranger_game_cells,
                 poacher_game, poacher_game_cells, end_moves=end)
t1 = datetime.now()
print('Finished process, time taken: ', t1-t0, '\n')
game4_traj = game4['trajectories'].trajectories
print(game4['paths'].groupby(['Player']).agg({'Trajectory': 'max'}), '\n')
print(game4['totals'], '/n')

# simulations
set_seed = 123456

t0 = datetime.now()
print(t0, '\nRunning simulations for random rhino, random ranger and random poacher')
sim1 = simulate(grid, rhino_rand, rhino_rand_cells, ranger_rand, ranger_rand_cells,
                poacher_rand, poacher_rand_cells, set_seed=set_seed, end_moves=end, games_pm=gpm)
t1 = datetime.now()
print('Finished process, time taken: ', t1-t0, '\n')

t0 = datetime.now()
print(t0, '\nRunning simulations for structured rhino, random ranger and structured poacher')
sim2 = simulate(grid, rhino_struct, rhino_struct_cells, ranger_rand, ranger_rand_cells,
                poacher_struct, poacher_struct_cells, set_seed=set_seed, end_moves=end, games_pm=gpm)
t1 = datetime.now()
print('Finished process, time taken: ', t1-t0, '\n')
#
t0 = datetime.now()
print(t0, '\nRunning simulations for structured rhino, game ranger and structured poacher')
sim3 = simulate(grid, rhino_struct, rhino_struct_cells, ranger_game, ranger_game_cells,
                poacher_struct, poacher_struct_cells, set_seed=set_seed, end_moves=end, games_pm=gpm)
t1 = datetime.now()
print('Finished process, time taken: ', t1-t0, '\n')

t0 = datetime.now()
print(t0, '\nRunning simulations for structured rhino, game ranger and game poacher')
sim4 = simulate(grid, rhino_struct, rhino_struct_cells, ranger_game, ranger_game_cells,
                poacher_game, poacher_game_cells, set_seed=set_seed, end_moves=end, games_pm=gpm)
t1 = datetime.now()
print('Finished process, time taken: ', t1-t0, '\n')

# PLOTS

# Subarea
fig0, ax = plt.subplots(1, figsize=(10, 15))
bounds_to_polygon(subarea, park.default_crs).plot(ax=ax, color='cyan', alpha=0.5)
park.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
ctx.add_basemap(ax, url=ctx.tile_providers.ST_TERRAIN, crs=park.default_crs, zoom=11)
ax.axis('off')
plt.savefig('knp/eg2_a_subarea.png', bbox_inches='tight')

# Plot random rhino cells
fig1, ax = plt.subplots(1, figsize=(15, 10))
rhino_rand_cells['cells'].plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5)
park.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
park.trees.plot(ax=ax, facecolor='lightgreen', edgecolor='none', alpha=0.5)
park.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5)
park.rivers.plot(ax=ax, color='blue', linewidth=1, linestyle='solid', alpha=0.6)
park.dams.plot(ax=ax, color='royalblue', marker='o', markersize=40)
park.water.plot(ax=ax, edgecolor='none', color='lightblue', marker='o', markersize=40)
park.roads.plot(ax=ax, color='black', linewidth=1, linestyle='dashed', alpha=0.6)
park.camps.plot(ax=ax, edgecolor='none', color='darkorange', marker='^', markersize=45)
park.picnic_spots.plot(ax=ax, edgecolor='none', color='darkgreen', marker='^', markersize=45)
park.gates.plot(ax=ax, edgecolor='none', color='red', marker='s', markersize=40)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.axis('off')
plt.savefig('knp/eg2_b_rhino_rand_cells.png', bbox_inches='tight')

# Plot structured rhino cells
fig2, ax = plt.subplots(1, figsize=(15, 10))
rhino_struct_cells['cells'].plot(ax=ax, column='Select Prob', cmap='Purples',
                                 edgecolor='black', linewidth=0.5)
park.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
park.trees.plot(ax=ax, facecolor='lightgreen', edgecolor='none', alpha=0.5)
park.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5)
park.rivers.plot(ax=ax, color='blue', linewidth=1, linestyle='solid', alpha=0.6)
park.dams.plot(ax=ax, color='royalblue', marker='o', markersize=40)
park.water.plot(ax=ax, edgecolor='none', color='lightblue', marker='o', markersize=40)
park.roads.plot(ax=ax, color='black', linewidth=1, linestyle='dashed', alpha=0.6)
park.camps.plot(ax=ax, edgecolor='none', color='darkorange', marker='^', markersize=45)
park.picnic_spots.plot(ax=ax, edgecolor='none', color='darkgreen', marker='^', markersize=45)
park.gates.plot(ax=ax, edgecolor='none', color='red', marker='s', markersize=40)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.axis('off')
plt.savefig('knp/eg2_c_rhino_struct_cells.png', bbox_inches='tight')

# Plot random ranger cells
fig3, ax = plt.subplots(1, figsize=(15, 10))
ranger_rand_cells['cells'].plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5)
park.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
park.trees.plot(ax=ax, facecolor='lightgreen', edgecolor='none', alpha=0.5)
park.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5)
park.rivers.plot(ax=ax, color='blue', linewidth=1, linestyle='solid', alpha=0.6)
park.dams.plot(ax=ax, color='royalblue', marker='o', markersize=40)
park.water.plot(ax=ax, edgecolor='none', color='lightblue', marker='o', markersize=40)
park.roads.plot(ax=ax, color='black', linewidth=1, linestyle='dashed', alpha=0.6)
park.camps.plot(ax=ax, edgecolor='none', color='darkorange', marker='^', markersize=45)
park.picnic_spots.plot(ax=ax, edgecolor='none', color='darkgreen', marker='^', markersize=45)
park.gates.plot(ax=ax, edgecolor='none', color='red', marker='s', markersize=40)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.axis('off')
plt.savefig('knp/eg2_d_ranger_rand_cells.png', bbox_inches='tight')

# Plot game ranger cells
fig4, ax = plt.subplots(1, figsize=(15, 10))
ranger_game_cells['cells'].plot(ax=ax, column='Utility', cmap='Purples',
                                edgecolor='black', linewidth=0.5)
park.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
park.trees.plot(ax=ax, facecolor='lightgreen', edgecolor='none', alpha=0.5)
park.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5)
park.rivers.plot(ax=ax, color='blue', linewidth=1, linestyle='solid', alpha=0.6)
park.dams.plot(ax=ax, color='royalblue', marker='o', markersize=40)
park.water.plot(ax=ax, edgecolor='none', color='lightblue', marker='o', markersize=40)
park.roads.plot(ax=ax, color='black', linewidth=1, linestyle='dashed', alpha=0.6)
park.camps.plot(ax=ax, edgecolor='none', color='darkorange', marker='^', markersize=45)
park.picnic_spots.plot(ax=ax, edgecolor='none', color='darkgreen', marker='^', markersize=45)
park.gates.plot(ax=ax, edgecolor='none', color='red', marker='s', markersize=40)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.axis('off')
plt.savefig('knp/eg2_e_ranger_game_cells.png', bbox_inches='tight')

# Plot random poacher cells
fig5, ax = plt.subplots(1, figsize=(15, 10))
poacher_rand_cells['cells'].plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5)
park.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
park.trees.plot(ax=ax, facecolor='lightgreen', edgecolor='none', alpha=0.5)
park.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5)
park.rivers.plot(ax=ax, color='blue', linewidth=1, linestyle='solid', alpha=0.6)
park.dams.plot(ax=ax, color='royalblue', marker='o', markersize=40)
park.water.plot(ax=ax, edgecolor='none', color='lightblue', marker='o', markersize=40)
park.roads.plot(ax=ax, color='black', linewidth=1, linestyle='dashed', alpha=0.6)
park.camps.plot(ax=ax, edgecolor='none', color='darkorange', marker='^', markersize=45)
park.picnic_spots.plot(ax=ax, edgecolor='none', color='darkgreen', marker='^', markersize=45)
park.gates.plot(ax=ax, edgecolor='none', color='red', marker='s', markersize=40)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.axis('off')
plt.savefig('knp/eg2_f_poacher_rand_cells.png', bbox_inches='tight')

# Plot structured poacher cells
fig6, ax = plt.subplots(1, figsize=(15, 10))
poacher_struct_cells['cells'].plot(ax=ax, column='Select Prob', cmap='Purples',
                                   edgecolor='black', linewidth=0.5)
park.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
park.trees.plot(ax=ax, facecolor='lightgreen', edgecolor='none', alpha=0.5)
park.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5)
park.rivers.plot(ax=ax, color='blue', linewidth=1, linestyle='solid', alpha=0.6)
park.dams.plot(ax=ax, color='royalblue', marker='o', markersize=40)
park.water.plot(ax=ax, edgecolor='none', color='lightblue', marker='o', markersize=40)
park.roads.plot(ax=ax, color='black', linewidth=1, linestyle='dashed', alpha=0.6)
park.camps.plot(ax=ax, edgecolor='none', color='darkorange', marker='^', markersize=45)
park.picnic_spots.plot(ax=ax, edgecolor='none', color='darkgreen', marker='^', markersize=45)
park.gates.plot(ax=ax, edgecolor='none', color='red', marker='s', markersize=40)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.axis('off')
plt.savefig('knp/eg2_g_poacher_struct_cells.png', bbox_inches='tight')

# Plot game poacher cells
fig7, ax = plt.subplots(1, figsize=(15, 10))
poacher_game_cells['cells'].plot(ax=ax, column='Utility', cmap='Purples',
                                 edgecolor='black', linewidth=0.5)
park.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
park.trees.plot(ax=ax, facecolor='lightgreen', edgecolor='none', alpha=0.5)
park.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5)
park.rivers.plot(ax=ax, color='blue', linewidth=1, linestyle='solid', alpha=0.6)
park.dams.plot(ax=ax, color='royalblue', marker='o', markersize=40)
park.water.plot(ax=ax, edgecolor='none', color='lightblue', marker='o', markersize=40)
park.roads.plot(ax=ax, color='black', linewidth=1, linestyle='dashed', alpha=0.6)
park.camps.plot(ax=ax, edgecolor='none', color='darkorange', marker='^', markersize=45)
park.picnic_spots.plot(ax=ax, edgecolor='none', color='darkgreen', marker='^', markersize=45)
park.gates.plot(ax=ax, edgecolor='none', color='red', marker='s', markersize=40)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.axis('off')
plt.savefig('knp/eg2_h_poacher_game_cells.png', bbox_inches='tight')

# Plot trajectories for game1
fig8, ax = plt.subplots(1, figsize=(15, 10))
rhino_rand_cells['cells'].plot(ax=ax, color='none', edgecolor='black', linewidth=0.3)
game1_traj[0].plot(ax=ax, column='Trajectory', cmap='Greens_r', linewidth=5, label='rhino')
game1_traj[1].plot(ax=ax, column='Trajectory', cmap='Blues_r', linewidth=5, label='ranger')
game1_traj[2].plot(ax=ax, column='Trajectory', cmap='Reds_r', linewidth=5, label='poacher')
ctx.add_basemap(ax, url=ctx.tile_providers.OSM_A, crs=park.default_crs, zoom=12)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.axis('off')
plt.savefig('knp/eg2_i_game1_paths.png', bbox_inches='tight')

# Plot trajectories for game2
fig9, ax = plt.subplots(1, figsize=(15, 10))
rhino_rand_cells['cells'].plot(ax=ax, color='none', edgecolor='black', linewidth=0.3)
game2_traj[0].plot(ax=ax, column='Trajectory', cmap='Greens_r', linewidth=5, label='rhino')
game2_traj[1].plot(ax=ax, column='Trajectory', cmap='Blues_r', linewidth=5, label='ranger')
game2_traj[2].plot(ax=ax, column='Trajectory', cmap='Reds_r', linewidth=5, label='poacher')
ctx.add_basemap(ax, url=ctx.tile_providers.OSM_A, crs=park.default_crs, zoom=12)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.axis('off')
plt.savefig('knp/eg2_j_game2_paths.png', bbox_inches='tight')

# Plot trajectories for game3
fig10, ax = plt.subplots(1, figsize=(15, 10))
rhino_rand_cells['cells'].plot(ax=ax, color='none', edgecolor='black', linewidth=0.3)
game3_traj[0].plot(ax=ax, column='Trajectory', cmap='Greens_r', linewidth=5, label='rhino')
game3_traj[1].plot(ax=ax, column='Trajectory', cmap='Blues_r', linewidth=5, label='ranger')
game3_traj[2].plot(ax=ax, column='Trajectory', cmap='Reds_r', linewidth=5, label='poacher')
ctx.add_basemap(ax, url=ctx.tile_providers.OSM_A, crs=park.default_crs, zoom=12)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.axis('off')
plt.savefig('knp/eg2_k_game3_paths.png', bbox_inches='tight')

# Plot trajectories for game4
fig11, ax = plt.subplots(1, figsize=(15, 10))
rhino_rand_cells['cells'].plot(ax=ax, color='none', edgecolor='black', linewidth=0.3)
game4_traj[0].plot(ax=ax, column='Trajectory', cmap='Greens_r', linewidth=5, label='rhino')
game4_traj[1].plot(ax=ax, column='Trajectory', cmap='Blues_r', linewidth=5, label='ranger')
game4_traj[2].plot(ax=ax, column='Trajectory', cmap='Reds_r', linewidth=5, label='poacher')
ctx.add_basemap(ax, url=ctx.tile_providers.OSM_A, crs=park.default_crs, zoom=12)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.axis('off')
plt.savefig('knp/eg2_l_game4_paths.png', bbox_inches='tight')

# save workspace
filename = 'example2-sim.pkl'
dill.dump_session(filename)

# and to load the session again:
# dill.load_session(filename)
