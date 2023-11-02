import os
os.chdir('C:/Users/lisak/OneDrive/PhD2.0/Coding/Simulations/')

# load modules
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
sns.set(style="white")

# Subarea and park
subarea = np.array([31.12, -25.3, 31.65, -25])
park = Park('Kruger National Park', knp.camps_main, knp.picnic_spots, knp.gates, knp.water, subarea=subarea,
            x_len=1500, y_len=1500)
print(f'number grid cells: {len(park.grid)}')

# smaller subarea
subarea_small = np.array([31.189, -25.190, 31.337, -25.055])
park_small = Park('Kruger National Park', knp.camps_main, knp.picnic_spots, knp.gates, knp.water, subarea=subarea_small,
            x_len=1500, y_len=1500)
print(f'number grid cells: {len(park_small.grid)}')

subarea_poly = bounds_to_polygon(subarea_small, park.default_crs)
small_cells = gpd.GeoSeries(park_small.grid.unary_union, crs=park.default_crs)
small_cells = small_cells.to_crs(park_small.proj_crs)
park_small_area = small_cells.area[0] / 1e+6
print('Grid area (sqkm): ', park_small_area, '\n')

# rhino density
rhino_density = gpd.read_file('2023_new/shapefiles/rhino_density.shp')
rhino_cells = park_small.grid.sjoin(rhino_density, how='inner', op='intersects')
rhino_cells = rhino_cells[['Centroids', 'TOTAL', 'CALVES']]
rhino_cells = rhino_cells.set_geometry('Centroids')
park_small = Park('Kruger National Park', knp.camps_main, knp.picnic_spots, knp.gates, knp.water, subarea=subarea_small,
            x_len=1500, y_len=1500, wildlife_sightings=rhino_cells)

# rhinos
out = ['rivers', 'dams']
rhino_rand = Wildlife('rhino_rand', park_small, 'bound', 'random', out=out)
rhino_strat = Wildlife('rhino_strat', park_small, 'bound', 'strategic', out=out,
                       dislike={'camps': 1000, 'picnic_spots': 1000, 'gates': 1000},
                       like={'dams': 10000, 'water': 10000})
print('number allowed cells: ', len(rhino_strat.allowed_cells), '\n')

# rangers
ranger_rand = Ranger('ranger_rand', park_small, 'bound', 'random', out=out)
ranger_game = Ranger('ranger_game', park_small, 'bound', 'game', out=out,
                     like={'rivers': 8000, 'dams': 8000, 'water': 8000},
                     geo_util_fctr=2, arrest_util=50, wild_save_fctr=6, wild_calve_fctr=0, path_type='stay')

# poachers
poacher_rand = Poacher('poacher_rand', park_small, 'bound', 'random', out=out, entry_type='border')
poacher_strat = Poacher('poacher_strat', park_small, 'bound', 'strategic', out=out,
                       dislike={'camps': 1000, 'gates': 1000, 'picnic_spots': 1000},
                       like={'roads': 2500, 'border': 10000, 'rivers': 8000, 'dams': 8000, 'water': 8000},
                        geo_util_fctr=2, arrest_util=50, wild_save_fctr=6, wild_calve_fctr=0, entry_type='border')
poacher_game = Poacher('poacher_game', park_small, 'bound', 'game', out=out,
                       dislike={'camps': 1000, 'gates': 1000, 'picnic_spots': 1000},
                       like={'roads': 2500, 'border': 10000, 'rivers': 8000, 'dams': 8000, 'water': 8000},
                       geo_util_fctr=2, arrest_util=50, wild_save_fctr=6, wild_calve_fctr=0, entry_type='border')

# games
seed = 1024
end = 50 # 1 day - 18hr movement at 4km/hr
gpm = 30
months = 500

game1 = Game('GAME1', rhino_rand, ranger_rand, poacher_rand, seed=seed, end_moves=end, games_pm=gpm,
             months=months)  
game2 = Game('GAME2', rhino_strat, ranger_game, poacher_game, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='maximin')
game3 = Game('GAME3', rhino_strat, ranger_game, poacher_game, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='spne', leader = 'ranger')
game4 = Game('GAME4', rhino_strat, ranger_game, poacher_game, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='spne', leader='poacher') 
game5 = Game('GAME5', rhino_strat, ranger_game, poacher_game, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='nash') 
game6 = Game('GAME6', rhino_strat, ranger_game, poacher_game, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='dobss', leader='ranger') 
game7 = Game('GAME7', rhino_strat, ranger_game, poacher_game, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='ssg_follower', leader='poacher')
game8 = Game('GAME8', rhino_strat, ranger_game, poacher_game, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='ssg_follower', leader='poacher', ssg_high=1)

# against intelligent poacher
game9 = Game('GAME9', rhino_strat, ranger_rand, poacher_strat, seed=seed, end_moves=end, games_pm=gpm,
             months=months) 
game10 = Game('GAME10', rhino_strat, ranger_game, poacher_strat, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='maximin')
game11 = Game('GAME11', rhino_strat, ranger_game, poacher_strat, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='spne', leader = 'ranger')
game12 = Game('GAME12', rhino_strat, ranger_game, poacher_strat, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='spne', leader='poacher') 
game13 = Game('GAME13', rhino_strat, ranger_game, poacher_strat, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='nash') 
game14 = Game('GAME14', rhino_strat, ranger_game, poacher_strat, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='dobss', leader='ranger') 
game15 = Game('GAME15', rhino_strat, ranger_game, poacher_strat, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='ssg_follower', leader='poacher')
game16 = Game('GAME16', rhino_strat, ranger_game, poacher_strat, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='ssg_follower', leader='poacher', ssg_high=1)

# ssg follower
x1 = game7.mixed()['poacher']
u1 = game7.payoffs['poacher']
u2 = game7.payoffs['ranger']
up = [1, 1/2, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8, 1/9, 1/10, 0.08, 0.05]
util = np.zeros((len(up), 3))
cells = list()
for i in range(len(up)):
    game7.ssg_high = up[i]
    game7_sol = game7.game_solution()
    util[i, 0] = up[i]
    util[i, 1] = game7_sol['ranger_util']
    util[i, 2] = game7_sol['poacher_util']
    cells.append(np.where(game7_sol['ranger_strat'] >0)[0])
    print(game7_sol['ranger_strat'])
np.round(util, 3)

# set upper bound for follower game
game7.ssg_high = 0.08

# run simulations
t0 = datetime.now()
print(t0, '\nStarted running simulations')
with concurrent.futures.ProcessPoolExecutor() as executor:
    args = [game1, game2, game3, game4, game5, game6, game7, game8, game9, game10, game11, game12, game13, game14, game15, game16]
    results = executor.map(sim_games, args)
    sim = []
    for result in results:
        sim.append(result)
t1 = datetime.now()
print(t1, '\nFinished process, time taken: ', t1-t0, '\n')

# save workspace
del executor, result, results
filename = '2023_mixed_border_stay.pkl'
dill.dump_session(filename)

# evaluation
game_eval = sim_eval(sim)

with pd.ExcelWriter('2023_new/2023_mixed_border_stay.xlsx', engine='xlsxwriter') as writer:
    game_eval['games'].to_excel(writer, sheet_name='games')
    game_eval['months'].to_excel(writer, sheet_name='totals')
    game_eval['metrics'].to_excel(writer, sheet_name='metrics')
    game_eval['descrip'].to_excel(writer, sheet_name='descrip')
    game_eval['med_pval'].to_excel(writer, sheet_name='med_pval')


 ### PLOTS

# calculated fields
rhino_select_tot = sum(rhino_strat.allowed_cells['Select Prob'])
rhino_strat.allowed_cells['Select Norm'] = rhino_strat.allowed_cells['Select Prob'] / rhino_select_tot
animal_col = rhino_strat.allowed_cells.columns.get_loc('Total Adults') 
ranger_select_tot = sum(ranger_game.allowed_cells['Select Prob'])
ranger_game.allowed_cells['Select Norm'] = ranger_game.allowed_cells['Select Prob'] / ranger_select_tot
poacher_select_tot = sum(poacher_game.allowed_cells['Select Prob'])
poacher_game.allowed_cells['Select Norm'] = poacher_game.allowed_cells['Select Prob'] / poacher_select_tot

# game solutions
game2_sol = game2.game_solution()
game3_sol = game3.game_solution()
game4_sol = game4.game_solution()
game5_sol = game5.game_solution()
game6_sol = game6.game_solution()
game7_sol = game7.game_solution()
game8_sol = game8.game_solution()

ranger_game.allowed_cells['Maximin'] = list(game2_sol['ranger_strat'])
poacher_game.allowed_cells['Maximin'] = list(game2_sol['poacher_strat'])
ranger_lead = game3_sol['ranger_strat']
poacher_follow = game3_sol['poacher_strat']
ranger_follow = game4_sol['ranger_strat']
poacher_lead = game4_sol['poacher_strat']
ranger_game.allowed_cells['Nash'] = list(game5_sol['ranger_strat'])
poacher_game.allowed_cells['Nash'] = list(game5_sol['poacher_strat'])
dobss_poacher = list(game6_sol['poacher_strat']).index(1)
ranger_game.allowed_cells['DOBSS'] = game6_sol['ranger_strat']
ranger_game.allowed_cells['Follower'] = game7_sol['ranger_strat']
follower_optimal = list(game8_sol['ranger_strat']).index(1)

# print game solutions
print('Maximin')
round(ranger_game.allowed_cells['Maximin'][ranger_game.allowed_cells['Maximin']>0] , 3)
game2_sol['ranger_util']
round(poacher_game.allowed_cells['Maximin'][poacher_game.allowed_cells['Maximin']>0] , 3)
game2_sol['poacher_util']

print('SPNE leader')
game3_sol

print('SPNE follower')
game4_sol

print('Nash')
round(ranger_game.allowed_cells['Nash'][ranger_game.allowed_cells['Nash']>0] , 3)
game5_sol['ranger_util']
round(poacher_game.allowed_cells['Nash'][poacher_game.allowed_cells['Nash']>0] , 3)
game5_sol['poacher_util']

print('DOBSS')
round(ranger_game.allowed_cells['DOBSS'][ranger_game.allowed_cells['DOBSS']>0] , 3)
game6_sol['ranger_util']
dobss_poacher
game6_sol['poacher_util']

print('SSG follower')
ranger_game.allowed_cells['Follower'][ranger_game.allowed_cells['Follower']>0] 
game7_sol['ranger_util']
game7_sol['poacher_util']

 # plot small subarea on big grid
x_lim = (subarea[0], subarea[2] + 0.03)
y_lim = (subarea[1] - 0.03, subarea[3] + 0.03)
fig, ax = plt.subplots(1, figsize=(9, 6))
park.grid.plot(ax=ax, facecolor='white', edgecolor='black', linewidth=0.5)
subarea_poly.plot(ax=ax, facecolor='cyan', alpha=0.5, edgecolor='none')
park.border_cells.plot(ax=ax, facecolor='yellow', alpha=0.5)
park.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
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
plt.savefig('2023_new/grid.pdf', dpi=1200, bbox_inches='tight')

# plot small grid and cell numbers
x_lim = (subarea_small[0] - 0.01, subarea_small[2] + 0.01)
y_lim = (subarea_small[1] - 0.01, subarea_small[3] + 0.01)
centroids = ranger_game.allowed_cells['Centroids']
adj = -0.003
fig, ax = plt.subplots(1, figsize=(9, 6))
park_small.grid.plot(ax=ax, facecolor='white', edgecolor='black', linewidth=0.5)
park_small.edge_cells.plot(ax=ax, facecolor='magenta', alpha=0.5)
park_small.border_cells.plot(ax=ax, facecolor='yellow', alpha=0.5)
park_small.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
park_small.roads.plot(ax=ax, color='black', linewidth=2, linestyle='dashed', alpha=0.6, label='roads')
park_small.rivers.plot(ax=ax, color='blue', linewidth=1.5, linestyle='solid', alpha=0.6, label='rivers')
park_small.dams.plot(ax=ax, color='indigo', marker='o', markersize=40, label='dams', alpha=0.5)
park_small.water.plot(ax=ax, color='royalblue', marker='D', markersize=40,
                label='fountains, water holes & drinking troughs', alpha=0.5)
park_small.camps.plot(ax=ax, color='black', marker='p', markersize=50, label='camps')
park_small.picnic_spots.plot(ax=ax, color='darkgreen', marker='^', markersize=45, label='picnic spots')
park_small.gates.plot(ax=ax, color='crimson', marker='s', markersize=40, label='gates')
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
for i in range(len(centroids)):
    ax.annotate(str(centroids.index[i]), xy=(centroids.iloc[i].coords[0][0] + adj, centroids.iloc[i].coords[0][1] + adj))
plt.savefig('2023_new/grid_small.pdf', dpi=1200, bbox_inches='tight')
plt.close()

# plot rhino selection weights
sns.set(style="dark")
x_lim = (subarea_small[0] - 0.01, subarea_small[2] + 0.01)
y_lim = (subarea_small[1] - 0.03, subarea_small[3] + 0.03)
fig, ax = plt.subplots(1, figsize=(9, 6))
rhino_strat.allowed_cells.plot(ax=ax, column='Select Norm', cmap='Greens', edgecolor='none', 
                               legend=True, legend_kwds={'label': 'Normalised Selection Weights'})
adj = -0.002
for i in range(len(rhino_strat.allowed_cells)):
    ax.annotate(str(round(rhino_strat.allowed_cells.iloc[i, animal_col])), 
                        xy=(rhino_strat.allowed_cells['Centroids'].iloc[i].coords[0][0] + adj,
                        rhino_strat.allowed_cells['Centroids'].iloc[i].coords[0][1] + adj), color='black')
park_small.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
park_small.trees.plot(ax=ax, facecolor='olive', edgecolor='none', alpha=0.5, label='trees')
park_small.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5, label='mountains')
park_small.roads.plot(ax=ax, color='black', linewidth=2, linestyle='dashed', alpha=0.6, label='roads')
park_small.rivers.plot(ax=ax, color='blue', linewidth=1.5, linestyle='solid', alpha=0.6, label='rivers')
park_small.dams.plot(ax=ax, color='indigo', marker='o', markersize=40, label='dams', alpha=0.5)
park_small.water.plot(ax=ax, color='royalblue', marker='D', markersize=40,
                label='fountains, water holes & drinking troughs', alpha=0.5)
park_small.camps.plot(ax=ax, color='black', marker='p', markersize=50, label='camps')
park_small.picnic_spots.plot(ax=ax, color='darkgreen', marker='^', markersize=45, label='picnic spots')
park_small.gates.plot(ax=ax, color='crimson', marker='s', markersize=40, label='gates')
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
plt.savefig('2023_new/rhino_density.pdf', dpi=1200, bbox_inches='tight')

# plot ranger selection weights
fig, ax = plt.subplots(1, figsize=(9, 6))
ranger_game.allowed_cells.plot(ax=ax, column='Select Norm', cmap='Blues', edgecolor='none', 
                               legend=True, legend_kwds={'label': 'Normalised Selection Weights'})
park_small.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
park_small.trees.plot(ax=ax, facecolor='olive', edgecolor='none', alpha=0.5, label='trees')
park_small.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5, label='mountains')
park_small.roads.plot(ax=ax, color='black', linewidth=2, linestyle='dashed', alpha=0.6, label='roads')
park_small.rivers.plot(ax=ax, color='blue', linewidth=1.5, linestyle='solid', alpha=0.6, label='rivers')
park_small.dams.plot(ax=ax, color='indigo', marker='o', markersize=40, label='dams', alpha=0.5)
park_small.water.plot(ax=ax, color='royalblue', marker='D', markersize=40,
                label='fountains, water holes & drinking troughs', alpha=0.5)
park_small.camps.plot(ax=ax, color='black', marker='p', markersize=50, label='camps')
park_small.picnic_spots.plot(ax=ax, color='darkgreen', marker='^', markersize=45, label='picnic spots')
park_small.gates.plot(ax=ax, color='crimson', marker='s', markersize=40, label='gates')
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
plt.savefig('2023_new/ranger_weights.pdf', dpi=1200, bbox_inches='tight')

# plot poacher selection weights
fig, ax = plt.subplots(1, figsize=(9, 6))
poacher_game.allowed_cells.plot(ax=ax, column='Select Norm', cmap='Reds', edgecolor='none', 
                               legend=True, legend_kwds={'label': 'Normalised Selection Weights'})
park_small.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
park_small.trees.plot(ax=ax, facecolor='olive', edgecolor='none', alpha=0.5, label='trees')
park_small.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5, label='mountains')
park_small.roads.plot(ax=ax, color='black', linewidth=2, linestyle='dashed', alpha=0.6, label='roads')
park_small.rivers.plot(ax=ax, color='blue', linewidth=1.5, linestyle='solid', alpha=0.6, label='rivers')
park_small.dams.plot(ax=ax, color='indigo', marker='o', markersize=40, label='dams', alpha=0.5)
park_small.water.plot(ax=ax, color='royalblue', marker='D', markersize=40,
                label='fountains, water holes & drinking troughs', alpha=0.5)
park_small.camps.plot(ax=ax, color='black', marker='p', markersize=50, label='camps')
park_small.picnic_spots.plot(ax=ax, color='darkgreen', marker='^', markersize=45, label='picnic spots')
park_small.gates.plot(ax=ax, color='crimson', marker='s', markersize=40, label='gates')
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
plt.savefig('2023_new/poacher_weights.pdf', dpi=1200, bbox_inches='tight')

# plot poacher geographic utility
x_lim = (subarea_small[0] - 0.01, subarea_small[2] + 0.01)
y_lim = (subarea_small[1] - 0.01, subarea_small[3] + 0.01)
fig, ax = plt.subplots(1, figsize=(9, 6))
poacher_game.allowed_cells.plot(ax=ax, column='Utility', cmap='Reds', edgecolor='none', 
                               legend=True, legend_kwds={'label': 'Geographic Utility'})
park_small.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
park_small.trees.plot(ax=ax, facecolor='olive', edgecolor='none', alpha=0.5, label='trees')
park_small.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5, label='mountains')
park_small.roads.plot(ax=ax, color='black', linewidth=2, linestyle='dashed', alpha=0.4, label='roads')
park_small.rivers.plot(ax=ax, color='blue', linewidth=1.5, linestyle='solid', alpha=0.6, label='rivers')
park_small.dams.plot(ax=ax, color='indigo', marker='o', markersize=40, label='dams', alpha=0.5)
park_small.water.plot(ax=ax, color='royalblue', marker='D', markersize=40,
                label='fountains, water holes & drinking troughs', alpha=0.5)
park_small.camps.plot(ax=ax, color='black', marker='p', markersize=50, label='camps')
park_small.picnic_spots.plot(ax=ax, color='darkgreen', marker='^', markersize=45, label='picnic spots')
park_small.gates.plot(ax=ax, color='crimson', marker='s', markersize=40, label='gates')
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
plt.savefig('2023_new/poacher_utility.pdf', dpi=1200, bbox_inches='tight')

# plot ranger geographic utility
fig, ax = plt.subplots(1, figsize=(9, 6))
ranger_game.allowed_cells.plot(ax=ax, column='Utility', cmap='Blues', edgecolor='none', 
                               legend=True, legend_kwds={'label': 'Geographic Utility'})
park_small.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
park_small.trees.plot(ax=ax, facecolor='olive', edgecolor='none', alpha=0.5, label='trees')
park_small.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5, label='mountains')
park_small.roads.plot(ax=ax, color='black', linewidth=2, linestyle='dashed', alpha=0.4, label='roads')
park_small.rivers.plot(ax=ax, color='blue', linewidth=1.5, linestyle='solid', alpha=0.6, label='rivers')
park_small.dams.plot(ax=ax, color='indigo', marker='o', markersize=40, label='dams', alpha=0.5)
park_small.water.plot(ax=ax, color='royalblue', marker='D', markersize=40,
                label='fountains, water holes & drinking troughs', alpha=0.5)
park_small.camps.plot(ax=ax, color='black', marker='p', markersize=50, label='camps')
park_small.picnic_spots.plot(ax=ax, color='darkgreen', marker='^', markersize=45, label='picnic spots')
park_small.gates.plot(ax=ax, color='crimson', marker='s', markersize=40, label='gates')
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
plt.savefig('2023_new/ranger_utility.pdf', dpi=1200, bbox_inches='tight')

# plot ranger mixed strategy
fig, ax = plt.subplots(1, figsize=(9, 6))
ranger_game.allowed_cells.plot(ax=ax, column='Follower', cmap='Blues', edgecolor='none', 
                    legend=True, legend_kwds={'label': 'Mixed Strategy'})
ranger_game.allowed_cells.loc[[follower_optimal], ].set_geometry('Centroids').plot(ax=ax, color='white', marker='$O$',
                                                                              markersize=50)
park_small.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
park_small.trees.plot(ax=ax, facecolor='olive', edgecolor='none', alpha=0.5, label='trees')
park_small.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5, label='mountains')
park_small.roads.plot(ax=ax, color='black', linewidth=2, linestyle='dashed', alpha=0.6, label='roads')
park_small.rivers.plot(ax=ax, color='blue', linewidth=1.5, linestyle='solid', alpha=0.6, label='rivers')
park_small.dams.plot(ax=ax, color='indigo', marker='o', markersize=40, label='dams', alpha=0.5)
park_small.water.plot(ax=ax, color='royalblue', marker='D', markersize=40,
                label='fountains, water holes & drinking troughs', alpha=0.5)
park_small.camps.plot(ax=ax, color='black', marker='p', markersize=50, label='camps')
park_small.picnic_spots.plot(ax=ax, color='darkgreen', marker='^', markersize=45, label='picnic spots')
park_small.gates.plot(ax=ax, color='crimson', marker='s', markersize=40, label='gates')
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
plt.savefig('2023_new/ranger_mixed.pdf', dpi=1200, bbox_inches='tight')

# plot dobss
fig, ax = plt.subplots(1, figsize=(9, 6))
ranger_game.allowed_cells.plot(ax=ax, column='DOBSS', cmap='Blues', edgecolor='none', 
                    legend=True, legend_kwds={'label': 'Mixed Strategy'})
poacher_game.allowed_cells.loc[[dobss_poacher], ].set_geometry('Centroids').plot(ax=ax, color='black', marker='$P$',
                                                                              markersize=50)
park_small.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
park_small.trees.plot(ax=ax, facecolor='olive', edgecolor='none', alpha=0.5, label='trees')
park_small.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5, label='mountains')
park_small.roads.plot(ax=ax, color='black', linewidth=2, linestyle='dashed', alpha=0.6, label='roads')
park_small.rivers.plot(ax=ax, color='blue', linewidth=1.5, linestyle='solid', alpha=0.6, label='rivers')
park_small.dams.plot(ax=ax, color='indigo', marker='o', markersize=40, label='dams', alpha=0.5)
park_small.water.plot(ax=ax, color='royalblue', marker='D', markersize=40,
                label='fountains, water holes & drinking troughs', alpha=0.5)
park_small.camps.plot(ax=ax, color='black', marker='p', markersize=50, label='camps')
park_small.picnic_spots.plot(ax=ax, color='darkgreen', marker='^', markersize=45, label='picnic spots')
park_small.gates.plot(ax=ax, color='crimson', marker='s', markersize=40, label='gates')
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
plt.savefig('2023_new/dobss.pdf', dpi=1200, bbox_inches='tight')

# plot ranger Nash mixed strategy
fig, ax = plt.subplots(1, figsize=(9, 6))
ranger_game.allowed_cells.plot(ax=ax, column='Nash', cmap='Blues', edgecolor='none', 
                    legend=True, legend_kwds={'label': 'Mixed Strategy'})
park_small.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
park_small.trees.plot(ax=ax, facecolor='olive', edgecolor='none', alpha=0.5, label='trees')
park_small.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5, label='mountains')
park_small.roads.plot(ax=ax, color='black', linewidth=2, linestyle='dashed', alpha=0.6, label='roads')
park_small.rivers.plot(ax=ax, color='blue', linewidth=1.5, linestyle='solid', alpha=0.6, label='rivers')
park_small.dams.plot(ax=ax, color='indigo', marker='o', markersize=40, label='dams', alpha=0.5)
park_small.water.plot(ax=ax, color='royalblue', marker='D', markersize=40,
                label='fountains, water holes & drinking troughs', alpha=0.5)
park_small.camps.plot(ax=ax, color='black', marker='p', markersize=50, label='camps')
park_small.picnic_spots.plot(ax=ax, color='darkgreen', marker='^', markersize=45, label='picnic spots')
park_small.gates.plot(ax=ax, color='crimson', marker='s', markersize=40, label='gates')
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
plt.savefig('2023_new/ranger_nash.pdf', dpi=1200, bbox_inches='tight')

# plot poacher Nash mixed strategy
fig, ax = plt.subplots(1, figsize=(9, 6))
poacher_game.allowed_cells.plot(ax=ax, column='Nash', cmap='Reds', edgecolor='none', 
                    legend=True, legend_kwds={'label': 'Mixed Strategy'})
park_small.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
park_small.trees.plot(ax=ax, facecolor='olive', edgecolor='none', alpha=0.5, label='trees')
park_small.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5, label='mountains')
park_small.roads.plot(ax=ax, color='black', linewidth=2, linestyle='dashed', alpha=0.6, label='roads')
park_small.rivers.plot(ax=ax, color='blue', linewidth=1.5, linestyle='solid', alpha=0.6, label='rivers')
park_small.dams.plot(ax=ax, color='indigo', marker='o', markersize=40, label='dams', alpha=0.5)
park_small.water.plot(ax=ax, color='royalblue', marker='D', markersize=40,
                label='fountains, water holes & drinking troughs', alpha=0.5)
park_small.camps.plot(ax=ax, color='black', marker='p', markersize=50, label='camps')
park_small.picnic_spots.plot(ax=ax, color='darkgreen', marker='^', markersize=45, label='picnic spots')
park_small.gates.plot(ax=ax, color='crimson', marker='s', markersize=40, label='gates')
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
plt.savefig('2023_new/poacher_nash.pdf', dpi=1200, bbox_inches='tight')

# plot ranger maximin mixed strategy
fig, ax = plt.subplots(1, figsize=(9, 6))
ranger_game.allowed_cells.plot(ax=ax, column='Maximin', cmap='Blues', edgecolor='none', 
                    legend=True, legend_kwds={'label': 'Mixed Strategy'})
park_small.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
park_small.trees.plot(ax=ax, facecolor='olive', edgecolor='none', alpha=0.5, label='trees')
park_small.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5, label='mountains')
park_small.roads.plot(ax=ax, color='black', linewidth=2, linestyle='dashed', alpha=0.6, label='roads')
park_small.rivers.plot(ax=ax, color='blue', linewidth=1.5, linestyle='solid', alpha=0.6, label='rivers')
park_small.dams.plot(ax=ax, color='indigo', marker='o', markersize=40, label='dams', alpha=0.5)
park_small.water.plot(ax=ax, color='royalblue', marker='D', markersize=40,
                label='fountains, water holes & drinking troughs', alpha=0.5)
park_small.camps.plot(ax=ax, color='black', marker='p', markersize=50, label='camps')
park_small.picnic_spots.plot(ax=ax, color='darkgreen', marker='^', markersize=45, label='picnic spots')
park_small.gates.plot(ax=ax, color='crimson', marker='s', markersize=40, label='gates')
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
plt.savefig('2023_new/ranger_maximin.pdf', dpi=1200, bbox_inches='tight')

# plot poacher maximin mixed strategy
fig, ax = plt.subplots(1, figsize=(9, 6))
poacher_game.allowed_cells.plot(ax=ax, column='Maximin', cmap='Reds', edgecolor='none', 
                    legend=True, legend_kwds={'label': 'Mixed Strategy'})
park_small.boundary.plot(ax=ax, color='none', edgecolor='red', linewidth=1)
park_small.trees.plot(ax=ax, facecolor='olive', edgecolor='none', alpha=0.5, label='trees')
park_small.mountains.plot(ax=ax, facecolor='brown', edgecolor='none', alpha=0.5, label='mountains')
park_small.roads.plot(ax=ax, color='black', linewidth=2, linestyle='dashed', alpha=0.6, label='roads')
park_small.rivers.plot(ax=ax, color='blue', linewidth=1.5, linestyle='solid', alpha=0.6, label='rivers')
park_small.dams.plot(ax=ax, color='indigo', marker='o', markersize=40, label='dams', alpha=0.5)
park_small.water.plot(ax=ax, color='royalblue', marker='D', markersize=40,
                label='fountains, water holes & drinking troughs', alpha=0.5)
park_small.camps.plot(ax=ax, color='black', marker='p', markersize=50, label='camps')
park_small.picnic_spots.plot(ax=ax, color='darkgreen', marker='^', markersize=45, label='picnic spots')
park_small.gates.plot(ax=ax, color='crimson', marker='s', markersize=40, label='gates')
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
plt.savefig('2023_new/poacher_maximin.pdf', dpi=1200, bbox_inches='tight')


# boxplots of measures
sns.set(style="dark")
sim_metrics = pd.read_csv('2023_new/2023_mixed_border_metrics.csv')

g = sns.catplot(x='Poach Freq per Day', y='Simulation', hue='Game Strategy', 
                row='Strategy Execution', col='Poacher Movement',
                data=sim_metrics, kind='box', height=4, aspect=1.5, legend_out=False,
                width=0.6, dodge=False, sharex=True, sharey=False)
g.set_axis_labels("Frequency per Day", "")
g.set_titles("")
sns.move_legend(g, loc="upper center", ncol=3, title=None, frameon=False)
plt.savefig('2023_new/mixed_box_poach_freq.pdf', dpi=1200)
plt.close()

g = sns.catplot(x='Arrest Freq per Day', y='Simulation', hue='Game Strategy', 
                row='Strategy Execution', col='Poacher Movement',
                data=sim_metrics, kind='box', height=4, aspect=1.5, legend_out=False,
                width=0.6, dodge=False, sharex=False, sharey=False)
g.set_axis_labels("Frequency per Day", "")
g.set_titles("")
sns.move_legend(g, loc="upper center", ncol=3, title=None, frameon=False)
plt.savefig('2023_new/mixed_box_arrest_freq.pdf', dpi=1200)
plt.close()

g = sns.catplot(x='Ave Moves for Arrests', y='Simulation', hue='Game Strategy', 
                row='Strategy Execution', col='Poacher Movement',
                data=sim_metrics, kind='box', height=4, aspect=1.5, legend_out=False,
                width=0.6, dodge=False, sharex=True, sharey=False)
g.set_axis_labels("Ave Moves for Arrests", "")
g.set_titles("")
sns.move_legend(g, loc="upper center", ncol=3, title=None, frameon=False)
plt.savefig('2023_new/mixed_box_ave_moves.pdf', dpi=1200)
plt.close()

g = sns.catplot(x='Ave Distance for Non Arrests', y='Simulation', hue='Game Strategy', 
                row='Strategy Execution', col='Poacher Movement',
                data=sim_metrics, kind='box', height=4, aspect=1.5, legend_out=False,
                width=0.6, dodge=False, sharex=True, sharey=False)
g.set_axis_labels("Ave Distance for Non Arrests", "")
g.set_titles("")
sns.move_legend(g, loc="upper center", ncol=3, title=None, frameon=False)
plt.savefig('2023_new/mixed_box_ave_distance.pdf', dpi=1200)
plt.close()

# plot utility
sim_utility = sim_metrics[sim_metrics['Simulation'] == 'GAME2']
sim_utility = pd.concat([sim_utility, sim_metrics[sim_metrics['Game Strategy'] == 'Mixed']], ignore_index=True)
sim_utility = sim_utility[sim_utility['Strategy Execution'] == 'Continue']
sim_utility = sim_utility[sim_utility['Poacher Movement'] == 'Game']
sim_utility = sim_utility.melt(id_vars=['Simulation', 'Strategy Execution', 'Game Strategy', 'Month'],
                 value_vars=['Ave Ranger Utility', 'Ave Poacher Utility'])

g = sns.catplot(x='value', y='Simulation', row='variable',
                hue='Game Strategy', legend_out=False,
                data=sim_utility, kind='box', height=3, aspect=1.5, width=0.6, 
                dodge=False, sharex=True, sharey=True, linewidth=0.5, fliersize=1)
g.set_axis_labels("Ave Utility", "")
g.set_titles("")
sns.move_legend(g, loc="upper center", ncol=3, title=None, frameon=False)
plt.savefig('2023_new/mixed_box_utility.pdf', dpi=1200)
plt.close()

# plot sampling of mixed-strategy
sampling = pd.read_csv('2023_new/2023_mixed_border_sampling.csv')

g = sns.catplot(x='Cell Number', y='Probability', hue='Strategy', col='Game Player', col_wrap=3,
                data=sampling, kind='bar', height=3, aspect=1.2, legend_out=False,
                width=0.6, sharex=False, sharey=True, linewidth=0)
g.set_axis_labels("Cell Number", "Probability")
g.set_titles("")
sns.move_legend(g, loc="upper center", ncol=2, title=None, frameon=False)
plt.savefig('2023_new/mixed_sampling.pdf', dpi=1200)
plt.close()
