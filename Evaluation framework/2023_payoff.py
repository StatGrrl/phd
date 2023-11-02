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
sns.set(style="dark")

# Subarea and park
subarea = np.array([31.189, -25.190, 31.337, -25.055])
park = Park('Kruger National Park', knp.camps_main, knp.picnic_spots, knp.gates, knp.water, subarea=subarea,
            x_len=1500, y_len=1500)

# rhino density
rhino_density = gpd.read_file('2023_new/shapefiles/rhino_density.shp')
rhino_cells = park.grid.sjoin(rhino_density, how='inner', op='intersects')
rhino_cells = rhino_cells[['Centroids', 'TOTAL', 'CALVES']]
rhino_cells = rhino_cells.set_geometry('Centroids')
park = Park('Kruger National Park', knp.camps_main, knp.picnic_spots, knp.gates, knp.water, subarea=subarea,
            x_len=1500, y_len=1500, wildlife_sightings=rhino_cells)

# rhinos
out = ['rivers', 'dams']
rhino_strat = Wildlife('rhino_strat', park, 'bound', 'strategic', out=out,
                       dislike={'camps': 1000, 'picnic_spots': 1000, 'gates': 1000},
                       like={'dams': 10000, 'water': 10000})

# rangers
ranger_1 = Ranger('ranger_1', park, 'bound', 'game', out=out,
                     like={'rivers': 8000, 'dams': 8000, 'water': 8000},
                     geo_util_fctr=5, arrest_util=5, wild_save_fctr=5, wild_calve_fctr=0, path_type='stay')
ranger_2 = Ranger('ranger_2', park, 'bound', 'game', out=out,
                     like={'rivers': 8000, 'dams': 8000, 'water': 8000},
                     geo_util_fctr=50, arrest_util=5, wild_save_fctr=5, wild_calve_fctr=0, path_type='stay')
ranger_3 = Ranger('ranger_3', park, 'bound', 'game', out=out,
                     like={'rivers': 8000, 'dams': 8000, 'water': 8000},
                     geo_util_fctr=5, arrest_util=5, wild_save_fctr=50, wild_calve_fctr=0, path_type='stay')
ranger_4 = Ranger('ranger_4', park, 'bound', 'game', out=out,
                     like={'rivers': 8000, 'dams': 8000, 'water': 8000},
                     geo_util_fctr=5, arrest_util=50, wild_save_fctr=5, wild_calve_fctr=0, path_type='stay')
ranger_5 = Ranger('ranger_5', park, 'bound', 'game', out=out,
                     like={'rivers': 8000, 'dams': 8000, 'water': 8000},
                     geo_util_fctr=5, arrest_util=50, wild_save_fctr=50, wild_calve_fctr=0, path_type='stay')

# poachers
poacher_1 = Poacher('poacher_1', park, 'bound', 'game', out=out,
                       dislike={'camps': 1000, 'gates': 1000, 'picnic_spots': 1000},
                       like={'roads': 2500, 'border': 10000, 'rivers': 8000, 'dams': 8000, 'water': 8000},
                       geo_util_fctr=5, arrest_util=5, wild_save_fctr=5, wild_calve_fctr=0, entry_type='edge')
poacher_2 = Poacher('poacher_2', park, 'bound', 'game', out=out,
                       dislike={'camps': 1000, 'gates': 1000, 'picnic_spots': 1000},
                       like={'roads': 2500, 'border': 10000, 'rivers': 8000, 'dams': 8000, 'water': 8000},
                       geo_util_fctr=50, arrest_util=5, wild_save_fctr=5, wild_calve_fctr=0, entry_type='edge')
poacher_3 = Poacher('poacher_3', park, 'bound', 'game', out=out,
                       dislike={'camps': 1000, 'gates': 1000, 'picnic_spots': 1000},
                       like={'roads': 2500, 'border': 10000, 'rivers': 8000, 'dams': 8000, 'water': 8000},
                       geo_util_fctr=5, arrest_util=5, wild_save_fctr=50, wild_calve_fctr=0, entry_type='edge')
poacher_4 = Poacher('poacher_4', park, 'bound', 'game', out=out,
                       dislike={'camps': 1000, 'gates': 1000, 'picnic_spots': 1000},
                       like={'roads': 2500, 'border': 10000, 'rivers': 8000, 'dams': 8000, 'water': 8000},
                       geo_util_fctr=5, arrest_util=50, wild_save_fctr=5, wild_calve_fctr=0, entry_type='edge')
poacher_5 = Poacher('poacher_5', park, 'bound', 'game', out=out,
                       dislike={'camps': 1000, 'gates': 1000, 'picnic_spots': 1000},
                       like={'roads': 2500, 'border': 10000, 'rivers': 8000, 'dams': 8000, 'water': 8000},
                       geo_util_fctr=5, arrest_util=50, wild_save_fctr=50, wild_calve_fctr=0, entry_type='edge')

# intelligent poacher
poacher_strat_1 = Poacher('poacher_1', park, 'bound', 'strategic', out=out,
                       dislike={'camps': 1000, 'gates': 1000, 'picnic_spots': 1000},
                       like={'roads': 2500, 'border': 10000, 'rivers': 8000, 'dams': 8000, 'water': 8000},
                       geo_util_fctr=5, arrest_util=5, wild_save_fctr=5, wild_calve_fctr=0, entry_type='edge')
poacher_strat_2 = Poacher('poacher_2', park, 'bound', 'strategic', out=out,
                       dislike={'camps': 1000, 'gates': 1000, 'picnic_spots': 1000},
                       like={'roads': 2500, 'border': 10000, 'rivers': 8000, 'dams': 8000, 'water': 8000},
                       geo_util_fctr=50, arrest_util=5, wild_save_fctr=5, wild_calve_fctr=0, entry_type='edge')
poacher_strat_3 = Poacher('poacher_3', park, 'bound', 'strategic', out=out,
                       dislike={'camps': 1000, 'gates': 1000, 'picnic_spots': 1000},
                       like={'roads': 2500, 'border': 10000, 'rivers': 8000, 'dams': 8000, 'water': 8000},
                       geo_util_fctr=5, arrest_util=5, wild_save_fctr=50, wild_calve_fctr=0, entry_type='edge')
poacher_strat_4 = Poacher('poacher_4', park, 'bound', 'strategic', out=out,
                       dislike={'camps': 1000, 'gates': 1000, 'picnic_spots': 1000},
                       like={'roads': 2500, 'border': 10000, 'rivers': 8000, 'dams': 8000, 'water': 8000},
                       geo_util_fctr=5, arrest_util=50, wild_save_fctr=5, wild_calve_fctr=0, entry_type='edge')
poacher_strat_5 = Poacher('poacher_5', park, 'bound', 'strategic', out=out,
                       dislike={'camps': 1000, 'gates': 1000, 'picnic_spots': 1000},
                       like={'roads': 2500, 'border': 10000, 'rivers': 8000, 'dams': 8000, 'water': 8000},
                       geo_util_fctr=5, arrest_util=50, wild_save_fctr=50, wild_calve_fctr=0, entry_type='edge')

# games
seed = 123456789
end = 50 # 1 day - 18hr movement at 4km/hr
gpm = 30
months = 500

# ssg follower games
game1 = Game('GAME1', rhino_strat, ranger_1, poacher_1, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='ssg_follower', leader='poacher', ssg_high=0.175) 
game2 = Game('GAME2', rhino_strat, ranger_2, poacher_2, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='ssg_follower', leader='poacher', ssg_high=0.125) 
game3 = Game('GAME3', rhino_strat, ranger_3, poacher_3, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='ssg_follower', leader='poacher', ssg_high=0.2) 
game4 = Game('GAME4', rhino_strat, ranger_4, poacher_4, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='ssg_follower', leader='poacher', ssg_high=0.325) 
game5 = Game('GAME5', rhino_strat, ranger_5, poacher_5, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='ssg_follower', leader='poacher', ssg_high=0.25) 

# dobss games
game6 = Game('GAME6', rhino_strat, ranger_1, poacher_1, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='dobss', leader='ranger') 
game7 = Game('GAME7', rhino_strat, ranger_2, poacher_2, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='dobss', leader='ranger') 
game8 = Game('GAME8', rhino_strat, ranger_3, poacher_3, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='dobss', leader='ranger') 
game9 = Game('GAME9', rhino_strat, ranger_4, poacher_4, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='dobss', leader='ranger') 
game10 = Game('GAME10', rhino_strat, ranger_5, poacher_5, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='dobss', leader='ranger') 

# intelligent poacher games
game1_strat = Game('GAME1', rhino_strat, ranger_1, poacher_strat_1, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='ssg_follower', leader='poacher', ssg_high=0.175) 
game2_strat = Game('GAME2', rhino_strat, ranger_2, poacher_strat_2, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='ssg_follower', leader='poacher', ssg_high=0.125)
game3_strat = Game('GAME3', rhino_strat, ranger_3, poacher_strat_3, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='ssg_follower', leader='poacher', ssg_high=0.2)
game4_strat = Game('GAME4', rhino_strat, ranger_4, poacher_strat_4, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='ssg_follower', leader='poacher', ssg_high=0.325)
game5_strat = Game('GAME5', rhino_strat, ranger_5, poacher_strat_5, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='ssg_follower', leader='poacher', ssg_high=0.25)
game6_strat = Game('GAME6', rhino_strat, ranger_1, poacher_strat_1, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='dobss', leader='ranger') 
game7_strat = Game('GAME7', rhino_strat, ranger_2, poacher_strat_2, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='dobss', leader='ranger') 
game8_strat = Game('GAME8', rhino_strat, ranger_3, poacher_strat_3, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='dobss', leader='ranger') 
game9_strat = Game('GAME9', rhino_strat, ranger_4, poacher_strat_4, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='dobss', leader='ranger') 
game10_strat = Game('GAME10', rhino_strat, ranger_5, poacher_strat_5, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='dobss', leader='ranger') 

# run simulations game ranger vs game poacher
t0 = datetime.now()
print(t0, '\nStarted running simulations')
with concurrent.futures.ProcessPoolExecutor() as executor:
    #args = [game1, game2, game3, game4, game5, game6, game7, game8, game9, game10]
    args = [game1, game2, game4]
    results = executor.map(sim_games, args)
    sim = []
    for result in results:
        sim.append(result)
t1 = datetime.now()
print(t1, '\nFinished process, time taken: ', t1-t0, '\n')
del executor, result, results

# run simulations game ranger vs intelligent poacher
t0 = datetime.now()
print(t0, '\nStarted running simulations')
with concurrent.futures.ProcessPoolExecutor() as executor:
    #args2 = [game1_strat, game2_strat, game3_strat, game4_strat, game5_strat, 
    #        game6_strat, game7_strat, game8_strat, game9_strat, game10_strat]
    args2 = [game1_strat, game2_strat, game4_strat]
    results = executor.map(sim_games, args2)
    sim2 = []
    for result in results:
        sim2.append(result)
t1 = datetime.now()
print(t1, '\nFinished process, time taken: ', t1-t0, '\n')

# save workspace
del executor, result, results
filename = '2023_payoffs_edge.pkl'
#dill.dump_session(filename)

# evaluation
game_eval = sim_eval(sim)
game_eval2 = sim_eval(sim2)

with pd.ExcelWriter('2023_new/2023_payoffs_edge_game_xtra.xlsx', engine='xlsxwriter') as writer:
    game_eval['games'].to_excel(writer, sheet_name='games')
    game_eval['months'].to_excel(writer, sheet_name='totals')
    game_eval['metrics'].to_excel(writer, sheet_name='metrics')
    game_eval['descrip'].to_excel(writer, sheet_name='descrip')
    game_eval['med_pval'].to_excel(writer, sheet_name='med_pval')

with pd.ExcelWriter('2023_new/2023_payoffs_edge_intel_xtra.xlsx', engine='xlsxwriter') as writer:
    game_eval2['games'].to_excel(writer, sheet_name='games')
    game_eval2['months'].to_excel(writer, sheet_name='totals')
    game_eval2['metrics'].to_excel(writer, sheet_name='metrics')
    game_eval2['descrip'].to_excel(writer, sheet_name='descrip')
    game_eval2['med_pval'].to_excel(writer, sheet_name='med_pval')

# follower utility
def follower_util(game):
    up = [0.05, 0.08, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 
          0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575,
          0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825,
          0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1]
    util = np.zeros((len(up), 3))
    strat = list()
    for i in range(len(up)):
        game.ssg_high = up[i]
        game_sol = game.game_solution()
        util[i, 0] = up[i]
        util[i, 1] = game_sol['ranger_util']
        util[i, 2] = game_sol['poacher_util']
        strat.append(game_sol['ranger_strat'])
    return util, strat

game1_util = follower_util(game1)
game2_util = follower_util(game2)
game3_util = follower_util(game3)
game4_util = follower_util(game4)
game5_util = follower_util(game5)

# boxplots of measures
sim_metrics = pd.read_csv('2023_new/2023_payoffs_edge_metrics.csv')
sim_metrics = sim_metrics.melt(id_vars=['Simulation', 'Poacher Movement', 'Payoff Preference', 'Ranger Strategy', 'Month'],
                 value_vars=['Ave Ranger Utility', 'Ave Poacher Utility', 'Poach Freq per Day', 'Arrest Freq per Day'])

vars = ['Ave Ranger Utility', 'Ave Poacher Utility']
sim_utility = sim_metrics[sim_metrics['variable'] == 'Ave Ranger Utility']
sim_utility = pd.concat([sim_utility, sim_metrics[sim_metrics['variable'] == 'Ave Poacher Utility']])
g = sns.catplot(x='value', y='Payoff Preference', col='variable',
                row='Poacher Movement', hue='Ranger Strategy', legend_out=False,
                data=sim_utility, kind='box', height=3, aspect=1, width=0.5, 
                dodge=True, sharex=True, sharey=True, linewidth=0.5, fliersize=2,
                col_order=vars, row_order=['Game', 'Intelligent'])
g.set_axis_labels("Ave Utility", "")
g.set_titles("")
sns.move_legend(g, loc="upper center", ncol=2, title=None, frameon=False)
plt.savefig('2023_new/payoffs_box_utility.pdf', dpi=1200)
plt.close()

vars = ['Poach Freq per Day', 'Arrest Freq per Day']
sim_frequency = sim_metrics[sim_metrics['variable'] == 'Poach Freq per Day']
sim_frequency = pd.concat([sim_frequency, sim_metrics[sim_metrics['variable'] == 'Arrest Freq per Day']])
g = sns.catplot(x='value', y='Payoff Preference', col='variable',
                row='Poacher Movement', hue='Ranger Strategy', legend_out=False,
                data=sim_frequency, kind='box', height=3, aspect=1, width=0.5, 
                dodge=True, sharex='col', sharey=True, linewidth=0.5, fliersize=2,
                col_order=vars, row_order=['Game', 'Intelligent'])
g.set_axis_labels("Frequency per Day", "")
g.set_titles("")
sns.move_legend(g, loc="upper center", ncol=2, title=None, frameon=False)
plt.savefig('2023_new/payoffs_box_freq.pdf', dpi=1200)
plt.close()

# game solutions
follower = [game1, game2, game3, game4, game5]
follower_ranger_util = []
follower_poacher_util = []
follower_optimal = []
ranger_cells = ranger_1.allowed_cells.copy()
for game in follower:
    game_sol = game.game_solution()
    follower_ranger_util.append(game_sol['ranger_util'])
    follower_poacher_util.append(game_sol['poacher_util'])
    ranger_cells[game.name] = game_sol['ranger_strat']
    tmp_upper = game.ssg_high
    game.ssg_high = 1
    game_sol = game.game_solution()
    follower_optimal.append(list(game_sol['ranger_strat']).index(1))
    game.ssg_high = tmp_upper

dobss = [game6, game7, game8, game9, game10]
dobss_ranger_util = []
dobss_poacher_util = []
dobss_poacher_cell = []
poacher_cells = poacher_strat_1.allowed_cells.copy()
for game in dobss:
    game_sol = game.game_solution()
    dobss_ranger_util.append(game_sol['ranger_util'])
    dobss_poacher_util.append(game_sol['poacher_util'])
    ranger_cells[game.name] = game_sol['ranger_strat']
    dobss_poacher_cell.append(list(game_sol['poacher_strat']).index(1))

utility = pd.DataFrame({'payoff_structure': [1,2,3,4,5],
                        'follower_poacher': follower_poacher_util, 'follower_ranger': follower_ranger_util,
                        'follower_optimal': follower_optimal, 'dobss_cell': dobss_poacher_cell,
                        'dobss_poacher': dobss_poacher_util, 'dobss_ranger': dobss_ranger_util})

# plot game solutions
games = follower + dobss
vmin = min([ranger_cells[game.name].min() for game in games])
vmax = max([ranger_cells[game.name].max() for game in games])
x_lim = (subarea[0] + 0.01, subarea[2] + 0.01)
y_lim = (subarea[1] - 0.01, subarea[3] + 0.01)

fig, axs = plt.subplots(2, 5, figsize=(9, 4.5), sharex=True, sharey=True, constrained_layout=True)
for i in range(5):
    ranger_cells.plot(ax=axs[0, i], column=follower[i].name, cmap='Blues', edgecolor='none', vmin=vmin, vmax=vmax)
    ranger_cells.loc[[follower_optimal[i]], ].set_geometry('Centroids').plot(ax=axs[0, i], color='black', marker='$O$', markersize=25)
    ranger_cells.plot(ax=axs[1, i], column=dobss[i].name, cmap='Blues', edgecolor='none', vmin=vmin, vmax=vmax)
    poacher_cells.loc[[dobss_poacher_cell[i]], ].set_geometry('Centroids').plot(ax=axs[1, i], color='black', marker='$P$', markersize=25)
    for j in range(2):
            park.boundary.plot(ax=axs[j, i], color='none', edgecolor='red', linewidth=1)
            park.roads.plot(ax=axs[j, i], color='black', linewidth=1, linestyle='dashed', alpha=0.6)
            park.rivers.plot(ax=axs[j, i], color='blue', linewidth=1, linestyle='solid', alpha=0.6)
            park.dams.plot(ax=axs[j, i], color='indigo', marker='o', markersize=10, alpha=0.5)
            park.water.plot(ax=axs[j, i], color='royalblue', marker='D', markersize=10, alpha=0.5)
            park.camps.plot(ax=axs[j, i], color='black', marker='p', markersize=12)
            park.picnic_spots.plot(ax=axs[j, i], color='darkgreen', marker='^', markersize=12)
            park.gates.plot(ax=axs[j, i], color='crimson', marker='s', markersize=10)
            axs[j, i].xaxis.set_tick_params(labelsize=10)
            axs[j, i].yaxis.set_tick_params(labelsize=10)
            axs[j, i].set_xlim(x_lim)
            axs[j, i].set_ylim(y_lim)
fig.supxlabel('Longitude', fontsize=12)
fig.supylabel('Latitude', fontsize=12)
patch_col = axs[0,2].collections[0]
cb = fig.colorbar(patch_col, ax=axs, location='top', shrink=0.95, aspect=50)
cb.set_label(label='Probability', size=12)
plt.savefig('2023_new/game_solutions.pdf', dpi=1200, bbox_inches='tight')
plt.close()