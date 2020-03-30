from framework import *
import knp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import contextily as ctx
from datetime import datetime
import concurrent.futures
import dill
import warnings

warnings.simplefilter(action='ignore')

# Example 2
subarea = np.array([31.12, -25.3, 31.65, -25])
x_lim = (subarea[0] + 0.03, subarea[2] + 0.03)
y_lim = (subarea[1] + 0.03, subarea[3] + 0.03)
park = Park('Kruger National Park', knp.camps_main, knp.picnic_spots, knp.gates, knp.water, subarea=subarea,
            x_len=1500, y_len=1500)
grid = park.grid
print('number grid cells: ', len(grid))

out = ['rivers', 'dams', 'mountains', 'trees']

rhino_rand = Wildlife('rhino_rand', park, 'bound', 'random', out=out)
rhino_rand_cells = rhino_rand.allowed_cells
print('number allowed cells: ', len(rhino_rand_cells), '\n')

rhino_strat = Wildlife('rhino_strat', park, 'bound', 'strategic', out=out, dislike={'camps': 1000},
                       like={'dams': 10000, 'water': 10000})
rhino_strat_cells = rhino_strat.allowed_cells

ranger_rand = Ranger('ranger_rand', park, 'bound', 'random', out=out)
ranger_rand_cells = ranger_rand.allowed_cells

ranger_game = Ranger('ranger_game', park, 'bound', 'game', out=out, like={'dams': 10000, 'water': 10000},
                     geo_util_fctr=80, arrest_util=100)
ranger_game_cells = ranger_game.allowed_cells

poacher_rand = Poacher('poacher_rand', park, 'bound', 'random', out=out)
poacher_rand_cells = poacher_rand.allowed_cells

poacher_strat = Poacher('poacher_strat', park, 'bound', 'strategic', out=out,
                        dislike={'roads': 3000, 'camps': 2000, 'gates': 5000},
                        like={'border': 30000, 'dams': 15000, 'water': 15000})
poacher_strat_cells = poacher_strat.allowed_cells

poacher_game = Poacher('poacher_game', park, 'bound', 'game', out=out,
                       dislike={'roads': 3000, 'camps': 2000, 'gates': 5000},
                       like={'border': 30000, 'dams': 15000, 'water': 15000},
                       geo_util_fctr=50, arrest_util=150)
poacher_game_cells = poacher_game.allowed_cells

# simulation experiments
end = 200  # 200 x 1.5km = 300km ~ 1 days with short stops to rest and eat (+/- 5km/h -> 100km in 20h)

# single game for plots
gpm = 1
months = 1

rhino_start = rhino_rand.start()
ranger_start = ranger_rand.start()
poacher_start = poacher_rand.start()

rhino_1 = rhino_rand
rhino_1.start_cell = rhino_start
rhino_2 = rhino_strat
rhino_2.start_cell = rhino_start
ranger_1 = ranger_rand
ranger_1.start_cell = ranger_start
ranger_2 = ranger_game
ranger_2.start_cell = ranger_start
poacher_1 = poacher_rand
poacher_1.start_cell = poacher_start
poacher_2 = poacher_strat
poacher_2.start_cell = poacher_start
poacher_3 = poacher_game
poacher_3.start_cell = poacher_start

game1 = Game('game1', rhino_1, ranger_1, poacher_1, end_moves=end, games_pm=gpm, months=months)
game2 = Game('game2', rhino_2, ranger_1, poacher_2, end_moves=end, games_pm=gpm, months=months)
game3 = Game('game3', rhino_2, ranger_2, poacher_2, end_moves=end, games_pm=gpm, months=months,
             game_type='spne')
game4 = Game('game4', rhino_2, ranger_2, poacher_3, end_moves=end, games_pm=gpm, months=months,
             game_type='spne')

game1_sim = sim_game(game1)
game2_sim = sim_game(game2)
game3_sim = sim_game(game3)
game4_sim = sim_game(game4)

game1_traj = game1_sim['trajectories'][0].trajectories
game2_traj = game2_sim['trajectories'][0].trajectories
game3_traj = game3_sim['trajectories'][0].trajectories
game4_traj = game4_sim['trajectories'][0].trajectories

game1_gdf = game1_traj[2].df
game1_leave = game1_gdf[game1_gdf['Leave'] > 0]
game1_capture = game1_gdf[game1_gdf['Capture'] > 0]
game1_poach = game1_gdf[game1_gdf['Poach'] > 0]

game2_gdf = game2_traj[2].df
game2_leave = game2_gdf[game2_gdf['Leave'] > 0]
game2_capture = game2_gdf[game2_gdf['Capture'] > 0]
game2_poach = game2_gdf[game2_gdf['Poach'] > 0]

game3_gdf = game3_traj[2].df
game3_leave = game3_gdf[game3_gdf['Leave'] > 0]
game3_capture = game3_gdf[game3_gdf['Capture'] > 0]
game3_poach = game3_gdf[game3_gdf['Poach'] > 0]

game4_gdf = game4_traj[2].df
game4_leave = game4_gdf[game4_gdf['Leave'] > 0]
game4_capture = game4_gdf[game4_gdf['Capture'] > 0]
game4_poach = game4_gdf[game4_gdf['Poach'] > 0]

# simulations
seed = 123456
gpm = 10  # 3 x 10 = 30 days
months = 1000

game1 = Game('game1', rhino_rand, ranger_rand, poacher_rand, seed=seed, end_moves=end, games_pm=gpm, months=months)
game2 = Game('game2', rhino_strat, ranger_rand, poacher_strat, seed=seed, end_moves=end, games_pm=gpm, months=months)
game3 = Game('game3', rhino_strat, ranger_game, poacher_strat, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='spne')
game4 = Game('game4', rhino_strat, ranger_game, poacher_game, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='spne')

t0 = datetime.now()
print(t0, '\nStarted running simulations')
with concurrent.futures.ProcessPoolExecutor() as executor:
    args = [game1, game2, game3, game4]
    results = executor.map(sim_game, args)
    sim = []
    for result in results:
        sim.append(result)
t1 = datetime.now()
print('Finished process, time taken: ', t1-t0, '\n')

sim1 = sim[0]
sim2 = sim[1]
sim3 = sim[2]
sim4 = sim[3]

ave = pd.DataFrame(sim1['ave'], columns=['sim1'])
ave['sim2'] = sim2['ave']
ave['sim3'] = sim3['ave']
ave['sim4'] = sim4['ave']

# save workspace
filename = 'example2.pkl'
dill.dump_session(filename)

# PLOTS


# color maps
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


greens = truncate_colormap(plt.get_cmap('Greens'), 0.3, 0.9)
blues = truncate_colormap(plt.get_cmap('Blues'), 0.3, 0.9)
reds = truncate_colormap(plt.get_cmap('Reds'), 0.3, 0.9)

# Subarea
fig0, ax = plt.subplots(1, figsize=(10, 15))
bounds_to_polygon(subarea, park.default_crs).plot(ax=ax, color='cyan', alpha=0.5)
park.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
ctx.add_basemap(ax, url=ctx.tile_providers.ST_TERRAIN, crs=park.default_crs, zoom=11)
ax.axis('off')
plt.savefig('knp/eg2_a_subarea.png', bbox_inches='tight')

# Plot random rhino cells
fig1, ax = plt.subplots(1, figsize=(15, 10))
rhino_rand_cells.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5)
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

# Plot strategic rhino cells
fig2, ax = plt.subplots(1, figsize=(15, 10))
rhino_strat_cells.plot(ax=ax, column='Select Prob', cmap='Purples', edgecolor='black', linewidth=0.5)
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
plt.savefig('knp/eg2_c_rhino_strat_cells.png', bbox_inches='tight')

# Plot random ranger cells
fig3, ax = plt.subplots(1, figsize=(15, 10))
ranger_rand_cells.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5)
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
ranger_game_cells.plot(ax=ax, column='Utility', cmap='Purples', edgecolor='black', linewidth=0.5)
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
poacher_rand_cells.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5)
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

# Plot strategic poacher cells
fig6, ax = plt.subplots(1, figsize=(15, 10))
poacher_strat_cells.plot(ax=ax, column='Select Prob', cmap='Purples', edgecolor='black', linewidth=0.5)
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
plt.savefig('knp/eg2_g_poacher_strat_cells.png', bbox_inches='tight')

# Plot game poacher cells
fig7, ax = plt.subplots(1, figsize=(15, 10))
poacher_game_cells.plot(ax=ax, column='Utility', cmap='Purples', edgecolor='black', linewidth=0.5)
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
rhino_rand_cells.plot(ax=ax, color='none', edgecolor='black', linewidth=0.3, zorder=1)
game1_traj[0].plot(ax=ax, column='Trajectory', cmap=greens, linewidth=5, label='rhino', zorder=2)
game1_traj[1].plot(ax=ax, column='Trajectory', cmap=blues, linewidth=5, label='ranger', zorder=3)
game1_traj[2].plot(ax=ax, column='Trajectory', cmap=reds, linewidth=5, label='poacher', zorder=4)
game1_capture.plot(ax=ax, color='black', marker='*', markersize=100, zorder=5)
game1_leave.plot(ax=ax, color='black', marker='P', markersize=100, zorder=5)
game1_poach.plot(ax=ax, color='black', marker='X', markersize=100, zorder=5)
ctx.add_basemap(ax, url=ctx.tile_providers.OSM_A, crs=park.default_crs, zoom=12)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.axis('off')
plt.savefig('knp/eg2_i_game1_paths.png', bbox_inches='tight')

# Plot trajectories for game2
fig9, ax = plt.subplots(1, figsize=(15, 10))
rhino_rand_cells.plot(ax=ax, color='none', edgecolor='black', linewidth=0.3, zorder=1)
game2_traj[0].plot(ax=ax, column='Trajectory', cmap=greens, linewidth=5, label='rhino', zorder=2)
game2_traj[1].plot(ax=ax, column='Trajectory', cmap=blues, linewidth=5, label='ranger', zorder=3)
game2_traj[2].plot(ax=ax, column='Trajectory', cmap=reds, linewidth=5, label='poacher', zorder=4)
game2_capture.plot(ax=ax, color='black', marker='*', markersize=100, zorder=5)
game2_leave.plot(ax=ax, color='black', marker='P', markersize=100, zorder=5)
game2_poach.plot(ax=ax, color='black', marker='X', markersize=100, zorder=5)
ctx.add_basemap(ax, url=ctx.tile_providers.OSM_A, crs=park.default_crs, zoom=12)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.axis('off')
plt.savefig('knp/eg2_j_game2_paths.png', bbox_inches='tight')

# Plot trajectories for game3
fig10, ax = plt.subplots(1, figsize=(15, 10))
rhino_rand_cells.plot(ax=ax, color='none', edgecolor='black', linewidth=0.3, zorder=1)
game3_traj[0].plot(ax=ax, column='Trajectory', cmap=greens, linewidth=5, label='rhino', zorder=2)
game3_traj[1].plot(ax=ax, column='Trajectory', cmap=blues, linewidth=5, label='ranger', zorder=3)
game3_traj[2].plot(ax=ax, column='Trajectory', cmap=reds, linewidth=5, label='poacher', zorder=4)
game3_capture.plot(ax=ax, color='black', marker='*', markersize=100, zorder=5)
game3_leave.plot(ax=ax, color='black', marker='P', markersize=100, zorder=5)
game3_poach.plot(ax=ax, color='black', marker='X', markersize=100, zorder=5)
ctx.add_basemap(ax, url=ctx.tile_providers.OSM_A, crs=park.default_crs, zoom=12)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.axis('off')
plt.savefig('knp/eg2_k_game3_paths.png', bbox_inches='tight')

# Plot trajectories for game4
fig11, ax = plt.subplots(1, figsize=(15, 10))
rhino_rand_cells.plot(ax=ax, color='none', edgecolor='black', linewidth=0.3, zorder=1)
game4_traj[0].plot(ax=ax, column='Trajectory', cmap=greens, linewidth=5, label='rhino', zorder=2)
game4_traj[1].plot(ax=ax, column='Trajectory', cmap=blues, linewidth=5, label='ranger', zorder=3)
game4_traj[2].plot(ax=ax, column='Trajectory', cmap=reds, linewidth=5, label='poacher', zorder=4)
game4_capture.plot(ax=ax, color='black', marker='*', markersize=100, zorder=5)
game4_leave.plot(ax=ax, color='black', marker='P', markersize=100, zorder=5)
game4_poach.plot(ax=ax, color='black', marker='X', markersize=100, zorder=5)
ctx.add_basemap(ax, url=ctx.tile_providers.OSM_A, crs=park.default_crs, zoom=12)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.axis('off')
plt.savefig('knp/eg2_l_game4_paths.png', bbox_inches='tight')

# and to load the session again:
# dill.load_session(filename)
