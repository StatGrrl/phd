import os
os.chdir('C:/Users/lisak/OneDrive/PhD2.0/Coding/Simulations/')

from framework import *
import knp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import concurrent.futures
import dill
import matplotlib.colors as colors
import contextily as ctx
import warnings
import random

sns.set(style="dark")

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

# load the session again
filename = '2023_spne.pkl'
dill.load_session(filename)

# Subarea and park
subarea = np.array([31.12, -25.3, 31.65, -25])
park = Park('Kruger National Park', knp.camps_main, knp.picnic_spots, knp.gates, knp.water, subarea=subarea,
            x_len=1500, y_len=1500)
print(f'number grid cells: {len(park.grid)}')

# rhinos
out = ['rivers', 'dams', 'mountains', 'trees']
rhino_rand = Wildlife('rhino_rand', park, 'bound', 'random', out=out)
rhino_strat = Wildlife('rhino_strat', park, 'bound', 'strategic', out=out,
                       dislike={'camps': 1000, 'picnic_spots': 1000, 'gates': 1000},
                       like={'rivers': 8000, 'dams': 8000, 'water': 8000})
print('number allowed cells: ', len(rhino_strat.allowed_cells), '\n')

# Simulate rhino sightings
boundary = park.boundary.to_crs(park.proj_crs)
kruger_area = boundary.area[0] / 1e+6
print('Park area (sqkm): ', kruger_area, '\n')

rhino_cells = gpd.GeoSeries(rhino_rand.allowed_cells.unary_union, crs=park.default_crs)
rhino_cells = rhino_cells.to_crs(park.proj_crs)
rhino_area = rhino_cells.area[0] / 1e+6
print('Rhino area (sqkm): ', rhino_area, '\n')

subarea_perc = rhino_area / kruger_area
print(f'Area % {subarea_perc * 100}')
wr_max = subarea_perc * 2809 *1.5
print(f'Appr. Number White Rhino: {wr_max}')

rhino_density = rhino_strat.allowed_cells[['Centroids', 'Select Prob']]
rhino_density = rhino_density.sort_values(by=['Select Prob'], ascending=False)
rhino_density['TOTAL'] = 0
rhino_density['CALVES'] = 0
total_col = len(rhino_density.columns) - 2
random.seed(56789)
rhino_counts = np.random.choice([1, 2, 3, 4, 5, 6, 7], len(rhino_density), replace=True,
                                p=[0.268, 0.568, 0.130, 0.021, 0.01, 0.002, 0.001] )
rhino_sum = 0
cell = 0
while rhino_sum < int(wr_max):
    rhino_density.iloc[cell, total_col] = rhino_counts[cell]
    rhino_sum += rhino_counts[cell]
    cell += 1

rhino_density = rhino_density.set_geometry('Centroids')
rhino_totals = rhino_density['TOTAL']

# rhino_density.to_csv('knp/rhino_density.csv')
rhino_density.to_file('2023_new/shapefiles/rhino_density.shp')

# Subarea and park
subarea = np.array([31.12, -25.3, 31.65, -25])
park = Park('Kruger National Park', knp.camps_main, knp.picnic_spots, knp.gates, knp.water, subarea=subarea,
            x_len=1500, y_len=1500, wildlife_sightings=rhino_density)

# rhinos
out = ['rivers', 'dams', 'mountains', 'trees']
rhino_rand = Wildlife('rhino_rand', park, 'bound', 'random', out=out)
rhino_strat = Wildlife('rhino_strat', park, 'bound', 'strategic', out=out,
                       dislike={'camps': 1000, 'picnic_spots': 1000, 'gates': 1000},
                       like={'rivers': 8000, 'dams': 8000, 'water': 8000})

# rangers
ranger_rand = Ranger('ranger_rand', park, 'bound', 'random', out=out)
ranger_game = Ranger('ranger_game', park, 'bound', 'game', out=out,
                     like={'rivers': 8000, 'dams': 8000, 'water': 8000},
                     geo_util_fctr=2, arrest_util=50, wild_save_fctr=6, wild_calve_fctr=0)

# poachers
poacher_rand = Poacher('poacher_rand', park, 'bound', 'random', out=out)
poacher_strat = Poacher('poacher_strat', park, 'bound', 'strategic', out=out,
                       dislike={'camps': 1000, 'gates': 1000, 'picnic_spots': 1000},
                       like={'roads': 2500, 'border': 10000, 'rivers': 8000, 'dams': 8000, 'water': 8000},
                        geo_util_fctr=2, arrest_util=50, wild_save_fctr=6, wild_calve_fctr=0)
poacher_game = Poacher('poacher_game', park, 'bound', 'game', out=out,
                       dislike={'camps': 1000, 'gates': 1000, 'picnic_spots': 1000},
                       like={'roads': 2500, 'border': 10000, 'rivers': 8000, 'dams': 8000, 'water': 8000},
                       geo_util_fctr=2, arrest_util=50, wild_save_fctr=6, wild_calve_fctr=0)

# simulations
seed = 123456
end = 50 # 1 day - 18hr movement at 4km/hr
gpm = 30  
months = 500

game1 = Game('GAME1', rhino_rand, ranger_rand, poacher_rand, seed=seed, end_moves=end, games_pm=gpm,
             months=months)  # , rtn_traj=True, rtn_moves=True)
game2 = Game('GAME2', rhino_strat, ranger_rand, poacher_strat, seed=seed, end_moves=end, games_pm=gpm,
             months=months)  # , rtn_traj=True, rtn_moves=True)
game3 = Game('GAME3', rhino_strat, ranger_game, poacher_strat, seed=seed, end_moves=end, games_pm=gpm,
             months=months, game_type='spne', leader = 'ranger')  # , rtn_traj=True, rtn_moves=True)
game4 = Game('GAME4', rhino_strat, ranger_game, poacher_game, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='spne', leader = 'ranger')  # , rtn_traj=True, rtn_moves=True)
game5 = Game('GAME5', rhino_strat, ranger_game, poacher_strat, seed=seed, end_moves=end, games_pm=gpm,
             months=months, game_type='spne', leader='poacher')  # , rtn_traj=True, rtn_moves=True)
game6 = Game('GAME6', rhino_strat, ranger_game, poacher_game, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='spne', leader='poacher')  # , rtn_traj=True, rtn_moves=True)

t0 = datetime.now()
print(t0, '\nStarted running simulations')
with concurrent.futures.ProcessPoolExecutor() as executor:
    args = [game1, game2, game3, game4, game5, game6]
    results = executor.map(sim_games, args)
    sim = []
    for result in results:
        sim.append(result)
t1 = datetime.now()
print(t1, '\nFinished process, time taken: ', t1-t0, '\n')

# save workspace
del executor, result, results
dill.dump_session(filename)

# evaluation
game_eval = sim_eval(sim)

with pd.ExcelWriter('2023_new/2023_spne.xlsx', engine='xlsxwriter') as writer:
    game_eval['months'].to_excel(writer, sheet_name='totals')
    game_eval['metrics'].to_excel(writer, sheet_name='metrics')
    game_eval['descrip'].to_excel(writer, sheet_name='descrip')
    game_eval['med_pval'].to_excel(writer, sheet_name='med_pval')

# boxplots of measures
sim = pd.read_csv('2023_new/2023_spne_metrics.csv')
g = sns.catplot(x='Poach Freq per Day', y='Simulation', data=sim, kind='box', height=5, aspect=1.5)
g.set_axis_labels("Poach Freq per Day", "")
plt.savefig('2023_new/box_poach_freq.pdf', dpi=1200)

g = sns.catplot(x='Arrest Freq per Day', y='Simulation', data=sim, kind='box', height=5, aspect=1.5)
g.set_axis_labels("Arrest Freq per Day", "")
plt.savefig('2023_new/box_arrest_freq.pdf', dpi=1200)

g = sns.catplot(x='Ave Moves for Arrests', y='Simulation', data=sim, kind='box', height=5, aspect=1.5)
g.set_axis_labels("Ave Moves for Arrests", "")
plt.savefig('2023_new/box_ave_moves.pdf', dpi=1200)

g = sns.catplot(x='Ave Distance for Non Arrests', y='Simulation', data=sim, kind='box', height=5, aspect=1.5)
g.set_axis_labels("Ave Distance for Non Arrests (m)", "")
plt.savefig('2023_new/box_ave_distance.pdf', dpi=1200)

# plot rhino sightings
rhino_select_tot = sum(rhino_strat.allowed_cells['Select Prob'])
rhino_strat.allowed_cells['Select Norm'] = rhino_strat.allowed_cells['Select Prob'] / rhino_select_tot

x_lim = (subarea[0], subarea[2] + 0.03)
y_lim = (subarea[1] - 0.03, subarea[3] + 0.03)

fig, ax = plt.subplots(1, figsize=(9, 6))
rhino_strat.allowed_cells.plot(ax=ax, column='Select Norm', cmap='Greens', edgecolor='none', 
                               legend=True, legend_kwds={'label': 'Normalized Selection Weights'})
adj = -0.004
for i in range(len(rhino_density)):
    ax.annotate(str(rhino_density.iloc[i, total_col]), xy=(rhino_density.geometry.iloc[i].coords[0][0] + adj,
                                                           rhino_density.geometry.iloc[i].coords[0][1] + adj), color='black')
park.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
park.trees.plot(ax=ax, facecolor='olive', edgecolor='none', alpha=0.5, label='trees')
park.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5, label='mountains')
park.roads.plot(ax=ax, color='black', linewidth=2, linestyle='dashed', alpha=0.6, label='roads')
park.rivers.plot(ax=ax, color='blue', linewidth=1.5, linestyle='solid', alpha=0.6, label='rivers')
park.dams.plot(ax=ax, color='indigo', marker='o', markersize=40, label='dams', alpha=0.5)
park.water.plot(ax=ax, color='royalblue', marker='D', markersize=40,
                label='fountains, water holes & drinking troughs', alpha=0.5)
park.camps.plot(ax=ax, color='black', marker='p', markersize=50, label='camps')
park.picnic_spots.plot(ax=ax, color='darkgreen', marker='^', markersize=45, label='picnic spots')
park.gates.plot(ax=ax, color='crimson', marker='s', markersize=40, label='gates')
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
# ax.set_title('Rhino Density Estimates and Selection Weights', fontsize=16)
plt.savefig('2023_new/rhino_density.pdf', dpi=1200, bbox_inches='tight')


# plot geographic utility
game4_sol = game4.game_solution()
game6_sol = game6.game_solution()

ranger_lead = game4_sol['ranger_strat']
ranger_follow = game6_sol['ranger_strat']
poacher_lead = game6_sol['poacher_strat']
poacher_follow = game4_sol['poacher_strat']

x_lim = (subarea[0], subarea[2] + 0.03)
y_lim = (subarea[1] - 0.03, subarea[3] + 0.03)

fig, ax = plt.subplots(1, figsize=(9, 6))
poacher_game.allowed_cells.plot(ax=ax, column='Utility', cmap='Reds', edgecolor='none', 
                               legend=True, legend_kwds={'label': 'Geographic Utility'})
park.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
park.trees.plot(ax=ax, facecolor='olive', edgecolor='none', alpha=0.5, label='trees')
park.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5, label='mountains')
park.roads.plot(ax=ax, color='black', linewidth=2, linestyle='dashed', alpha=0.4, label='roads')
park.rivers.plot(ax=ax, color='blue', linewidth=1.5, linestyle='solid', alpha=0.6, label='rivers')
park.dams.plot(ax=ax, color='indigo', marker='o', markersize=40, label='dams', alpha=0.5)
park.water.plot(ax=ax, color='royalblue', marker='D', markersize=40,
                label='fountains, water holes & drinking troughs', alpha=0.5)
park.camps.plot(ax=ax, color='black', marker='p', markersize=50, label='camps')
park.picnic_spots.plot(ax=ax, color='darkgreen', marker='^', markersize=45, label='picnic spots')
park.gates.plot(ax=ax, color='crimson', marker='s', markersize=40, label='gates')
poacher_game.allowed_cells.loc[[poacher_lead], ].set_geometry('Centroids').plot(ax=ax, color='white', marker='$L$',
                                                                              markersize=50)
poacher_game.allowed_cells.loc[[poacher_follow], ].set_geometry('Centroids').plot(ax=ax, color='white', marker='$F$',
                                                                              markersize=50)
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
# ax.set_title('Poacher Geographic Utility and Optimal Cells', fontsize=16)
plt.savefig('2023_new/poacher_utility.pdf', dpi=1200, bbox_inches='tight')

fig, ax = plt.subplots(1, figsize=(9, 6))
ranger_game.allowed_cells.plot(ax=ax, column='Utility', cmap='Blues', edgecolor='none', 
                               legend=True, legend_kwds={'label': 'Geographic Utility'})
park.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
park.trees.plot(ax=ax, facecolor='olive', edgecolor='none', alpha=0.5, label='trees')
park.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5, label='mountains')
park.roads.plot(ax=ax, color='black', linewidth=2, linestyle='dashed', alpha=0.4, label='roads')
park.rivers.plot(ax=ax, color='blue', linewidth=1.5, linestyle='solid', alpha=0.6, label='rivers')
park.dams.plot(ax=ax, color='indigo', marker='o', markersize=40, label='dams', alpha=0.5)
park.water.plot(ax=ax, color='royalblue', marker='D', markersize=40,
                label='fountains, water holes & drinking troughs', alpha=0.5)
park.camps.plot(ax=ax, color='black', marker='p', markersize=50, label='camps')
park.picnic_spots.plot(ax=ax, color='darkgreen', marker='^', markersize=45, label='picnic spots')
park.gates.plot(ax=ax, color='crimson', marker='s', markersize=40, label='gates')
ranger_game.allowed_cells.loc[[ranger_lead], ].set_geometry('Centroids').plot(ax=ax, color='white', marker='$L$',
                                                                              markersize=50)
ranger_game.allowed_cells.loc[[ranger_follow], ].set_geometry('Centroids').plot(ax=ax, color='white', marker='$F$',
                                                                              markersize=50)
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
# ax.set_title('Ranger Geographic Utility and Optimal Cells', fontsize=16)
plt.savefig('2023_new/ranger_utility.pdf', dpi=1200, bbox_inches='tight')
