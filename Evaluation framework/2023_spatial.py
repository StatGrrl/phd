import os
os.chdir('C:/Users/lisak/OneDrive/PhD2.0/Coding/Simulations/')

from framework import *
from datetime import datetime
import concurrent.futures
import dill
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="dark")

# rhino densities
rhino_density = gpd.read_file('2023_new/shapefiles/rhino_density.shp')

# Subarea and park
subarea = np.array([31.12, -25.3, 31.65, -25])
park = Park('Kruger National Park', knp.camps_main, knp.picnic_spots, knp.gates, knp.water, subarea=subarea,
            x_len=1500, y_len=1500, wildlife_sightings=rhino_density)

# exclude cells with obstacles
out = ['rivers', 'dams', 'mountains', 'trees']

# rhinos
rhino_strat = Wildlife('rhino_strat', park, 'bound', 'strategic', out=out,
                       dislike={'camps': 1000, 'picnic_spots': 1000, 'gates': 1000},
                       like={'rivers': 8000, 'dams': 8000, 'water': 8000})

# rangers
ranger_rand = Ranger('ranger_rand', park, 'bound', 'random', out=out)
ranger_game = Ranger('ranger_game', park, 'bound', 'game', out=out,
                     like={'rivers': 8000, 'dams': 8000, 'water': 8000},
                     geo_util_fctr=2, arrest_util=50, wild_save_fctr=6, wild_calve_fctr=0)

# poachers
poacher_strat = Poacher('poacher_strat', park, 'bound', 'strategic', out=out,
                       dislike={'camps': 1000, 'gates': 1000, 'picnic_spots': 1000},
                       like={'roads': 2500, 'border': 10000, 'rivers': 8000, 'dams': 8000, 'water': 8000},
                        geo_util_fctr=2, arrest_util=50, wild_save_fctr=6, wild_calve_fctr=0)
poacher_game = Poacher('poacher_game', park, 'bound', 'game', out=out,
                       dislike={'camps': 1000, 'gates': 1000, 'picnic_spots': 1000},
                       like={'roads': 2500, 'border': 10000, 'rivers': 8000, 'dams': 8000, 'water': 8000},
                       geo_util_fctr=2, arrest_util=50, wild_save_fctr=6, wild_calve_fctr=0)

# Poacher probability and distances
poacher_cells = poacher_strat.allowed_cells.copy()
poacher_cells['geometry'] = poacher_cells['Map Cells']
poacher_cells = poacher_cells.set_geometry('geometry')
poacher_cells = poacher_cells.to_crs(park.proj_crs)
poacher_cells['Poacher Prob'] = poacher_cells['Select Prob'] / sum(poacher_cells['Select Prob']) # normalize
roads_series = park.roads.geometry.to_crs(park.proj_crs)
poacher_cells['Roads'] = [roads_series.distance(x).min() for x in poacher_cells['Centroids Proj']]
camps_series = park.camps.geometry.to_crs(park.proj_crs)
poacher_cells['Camps'] = [camps_series.distance(x).min() for x in poacher_cells['Centroids Proj']]
picnic_spots_series = park.picnic_spots.geometry.to_crs(park.proj_crs)
poacher_cells['Picnic'] = [picnic_spots_series.distance(x).min() for x in poacher_cells['Centroids Proj']]
gates_series = park.gates.geometry.to_crs(park.proj_crs)
poacher_cells['Gates'] = [gates_series.distance(x).min() for x in poacher_cells['Centroids Proj']]
border_series = park.border_line.geometry.to_crs(park.proj_crs)
poacher_cells['Border'] = [border_series.distance(x).min() for x in poacher_cells['Centroids Proj']]
rivers_series = park.rivers.geometry.to_crs(park.proj_crs)
poacher_cells['Rivers'] = [rivers_series.distance(x).min() for x in poacher_cells['Centroids Proj']]
dams_series = park.dams.geometry.to_crs(park.proj_crs)
poacher_cells['Dams'] = [dams_series.distance(x).min() for x in poacher_cells['Centroids Proj']]
water_series = park.water.geometry.to_crs(park.proj_crs)
poacher_cells['Water'] = [water_series.distance(x).min() for x in poacher_cells['Centroids Proj']]
poacher_cells['Rhinos'] = poacher_cells['Total Adults']
poacher_cells = poacher_cells.iloc[:, 11:22]

# save shapefiles
grid_gps = poacher_cells.to_crs(park.default_crs)
grid_gps.to_file('2023_new/shapefiles/grid_gps.shp', driver='ESRI Shapefile')
grid_proj = poacher_cells
grid_proj.to_file('2023_new/shapefiles/grid_proj.shp', driver='ESRI Shapefile')

# load grid in R to estimate poacher probability, save as ppm_fit.shp

# load ppm fitted intensity
ppm_fit = gpd.read_file('2023_new/shapefiles/ppm_fit.shp')
ppm_fit['Centroids'] = ppm_fit.centroid
ppm_fit = ppm_fit.set_geometry('Centroids')
ppm_fit = ppm_fit.to_crs(park.default_crs)
ppm_fit['Probability'] = ppm_fit['layer'] / ppm_fit['layer'].sum()
ppm_fit = ppm_fit[['Centroids', 'Probability']]
poacher_cells = poacher_cells.to_crs(park.default_crs)
ppm_fit_cells = poacher_cells.copy()
ppm_fit_cells = ppm_fit_cells.sjoin(ppm_fit, how='inner', predicate='intersects')

# poachers using spatial fitted probability
poacher_strat_spat = Poacher('poacher_strat_spat', park, 'bound', 'strategic', out=out,
                       dislike={'camps': 1000, 'gates': 1000, 'picnic_spots': 1000},
                       like={'roads': 2500, 'border': 10000, 'rivers': 8000, 'dams': 8000, 'water': 8000},
                        geo_util_fctr=2, arrest_util=50, wild_save_fctr=6, wild_calve_fctr=0)
poacher_strat_spat.allowed_cells['Select Prob'] = ppm_fit_cells['Probability']
poacher_game_spat = Poacher('poacher_game_spat', park, 'bound', 'game', out=out,
                       dislike={'camps': 1000, 'gates': 1000, 'picnic_spots': 1000},
                       like={'roads': 2500, 'border': 10000, 'rivers': 8000, 'dams': 8000, 'water': 8000},
                       geo_util_fctr=2, arrest_util=50, wild_save_fctr=6, wild_calve_fctr=0)
poacher_game_spat.allowed_cells['Select Prob'] = ppm_fit_cells['Probability']

# simulation parameters
seed = 123456789
end = 50 # 1 day - 18hr movement at 4km/hr
gpm = 30  
months = 500

# games
game1 = Game('GAME1', rhino_strat, ranger_rand, poacher_strat, seed=seed, end_moves=end, games_pm=gpm,
             months=months) 
game2 = Game('GAME2', rhino_strat, ranger_rand, poacher_strat_spat, seed=seed, end_moves=end, games_pm=gpm,
             months=months) 
game3 = Game('GAME3', rhino_strat, ranger_game, poacher_strat, seed=seed, end_moves=end, games_pm=gpm,
             months=months, game_type='ssg_follower', leader='poacher', ssg_high=0.0125)
game4 = Game('GAME4', rhino_strat, ranger_game, poacher_strat_spat, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='ssg_follower', leader='poacher', ssg_high=0.0125)
game5 = Game('GAME5', rhino_strat, ranger_game, poacher_game, seed=seed, end_moves=end, games_pm=gpm,
             months=months, game_type='ssg_follower', leader='poacher', ssg_high=0.0125)
game6 = Game('GAME6', rhino_strat, ranger_game, poacher_game_spat, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='ssg_follower', leader='poacher', ssg_high=0.0125)

# simulations
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
filename = '2023_spatial.pkl'
dill.dump_session(filename)

# evaluation
game_eval = sim_eval(sim)

with pd.ExcelWriter('2023_new/2023_spatial.xlsx', engine='xlsxwriter') as writer:
    game_eval['months'].to_excel(writer, sheet_name='totals')
    game_eval['metrics'].to_excel(writer, sheet_name='metrics')
    game_eval['descrip'].to_excel(writer, sheet_name='descrip')
    game_eval['med_pval'].to_excel(writer, sheet_name='med_pval')

# boxplots of measures
sim_metrics = pd.read_csv('2023_new/2023_spatial_metrics.csv')
sim_metrics = sim_metrics.melt(id_vars=['Simulation', 'Poacher Prob', 'Month'], 
                               value_vars=['Poach Freq per Day', 'Arrest Freq per Day'])

g = sns.catplot(x='value', y='Simulation', hue='Poacher Prob', col='variable', col_wrap=2,
                data=sim_metrics, kind='box', width=0.6, height=4, aspect=1.5, legend_out=False, 
                dodge=False, sharex=False, sharey=True)
g.set_axis_labels("Frequency per Day", "")
g.set_titles("")
sns.move_legend(g, loc="upper center", ncol=3, title=None, frameon=False)
plt.savefig('2023_new/spatial_box_freq.pdf', dpi=1200)
plt.close()

# plot calculated and fitted poacher probability
vmin = min(poacher_cells['Poacher Prob'].min(), poacher_strat_spat.allowed_cells['Select Prob'].min())
vmax = max(poacher_cells['Poacher Prob'].max(), poacher_strat_spat.allowed_cells['Select Prob'].max())
x_lim = (subarea[0] - 0.03, subarea[2] + 0.03)
y_lim = (subarea[1] - 0.03, subarea[3] + 0.03)

fig, axs = plt.subplots(1, 2, figsize=(9, 4.5), sharex=True, sharey=True, constrained_layout=True)
poacher_cells.plot(ax=axs[0], column='Poacher Prob', cmap='Reds', edgecolor='none', vmin=vmin, vmax=vmax)
poacher_strat_spat.allowed_cells.plot(ax=axs[1], column='Select Prob', cmap='Reds', edgecolor='none', vmin=vmin, vmax=vmax)
for i in range(2):
    park.boundary.plot(ax=axs[i], color='none', edgecolor='red', linewidth=1)
    park.trees.plot(ax=axs[i], facecolor='olive', edgecolor='none', alpha=0.5)
    park.mountains.plot(ax=axs[i], facecolor='brown', edgecolor='none', alpha=0.5)
    park.roads.plot(ax=axs[i], color='black', linewidth=2, linestyle='dashed', alpha=0.6)
    park.rivers.plot(ax=axs[i], color='blue', linewidth=1.5, linestyle='solid', alpha=0.6)
    park.dams.plot(ax=axs[i], color='indigo', marker='o', markersize=20, alpha=0.5)
    park.water.plot(ax=axs[i], color='royalblue', marker='D', markersize=20, alpha=0.5)
    park.camps.plot(ax=axs[i], color='black', marker='p', markersize=25)
    park.picnic_spots.plot(ax=axs[i], color='darkgreen', marker='^', markersize=25)
    park.gates.plot(ax=axs[i], color='crimson', marker='s', markersize=20)
    axs[i].xaxis.set_tick_params(labelsize=10)
    axs[i].yaxis.set_tick_params(labelsize=10)
    axs[i].set_xlim(x_lim)
    axs[i].set_ylim(y_lim)
fig.supxlabel('Longitude', fontsize=12)
fig.supylabel('Latitude', fontsize=12)
patch_col = axs[0].collections[0]
cb = fig.colorbar(patch_col, ax=axs, location='top', shrink=0.95, aspect=50)
cb.set_label(label='Probability', size=12)
plt.savefig('2023_new/poacher_probability.pdf', dpi=1200, bbox_inches='tight')
plt.close()

# plot ranger mixed strategies
game5_sol = game5.game_solution()
game6_sol = game6.game_solution()
ranger_cells = ranger_game.allowed_cells.copy()
ranger_cells['Mixed Strategy'] = game5_sol['ranger_strat']
ranger_cells['Mixed Strategy Spatial'] = game6_sol['ranger_strat']
game5_opt = Game('GAME5', rhino_strat, ranger_game, poacher_game, seed=seed, end_moves=end, games_pm=gpm,
             months=months, game_type='ssg_follower', leader='poacher', ssg_high=1)
game6_opt = Game('GAME6', rhino_strat, ranger_game, poacher_game_spat, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='ssg_follower', leader='poacher', ssg_high=1)
game5_opt_sol = game5_opt.game_solution()
game6_opt_sol = game6_opt.game_solution()
game5_opt_cell = list(game5_opt_sol['ranger_strat']).index(1)
game6_opt_cell = list(game6_opt_sol['ranger_strat']).index(1)

game5_opt_cell
game5_opt_sol['ranger_util']
game5_opt_sol['poacher_util'] 
np.where(game5_sol['ranger_strat']>0)[0]
game5_sol['ranger_util']
game5_sol['poacher_util'] 
game6_opt_cell 
game6_opt_sol['ranger_util']  
game6_opt_sol['poacher_util'] 
np.where(game6_sol['ranger_strat']>0)[0]
game6_sol['ranger_util']                 
game6_sol['poacher_util'] 

ranger_cells.to_csv('2023_new/ranger_spatial_strat.csv', index=True)

x_lim = (subarea[0] - 0.03, subarea[2] + 0.03)
y_lim = (subarea[1] - 0.03, subarea[3] + 0.03)
fig, axs = plt.subplots(1, 2, figsize=(9, 4.5), sharex=True, sharey=True, constrained_layout=True)
ranger_cells.plot(ax=axs[0], column='Mixed Strategy', cmap='Blues', edgecolor='none')
ranger_cells.plot(ax=axs[1], column='Mixed Strategy Spatial', cmap='Blues', edgecolor='none')
for i in range(2):
    park.boundary.plot(ax=axs[i], color='none', edgecolor='red', linewidth=1)
    park.trees.plot(ax=axs[i], facecolor='olive', edgecolor='none', alpha=0.5)
    park.mountains.plot(ax=axs[i], facecolor='brown', edgecolor='none', alpha=0.5)
    park.roads.plot(ax=axs[i], color='black', linewidth=2, linestyle='dashed', alpha=0.6)
    park.rivers.plot(ax=axs[i], color='blue', linewidth=1.5, linestyle='solid', alpha=0.6)
    park.dams.plot(ax=axs[i], color='indigo', marker='o', markersize=20, alpha=0.5)
    park.water.plot(ax=axs[i], color='royalblue', marker='D', markersize=20, alpha=0.5)
    park.camps.plot(ax=axs[i], color='black', marker='p', markersize=25)
    park.picnic_spots.plot(ax=axs[i], color='darkgreen', marker='^', markersize=25)
    park.gates.plot(ax=axs[i], color='crimson', marker='s', markersize=20)
    ranger_cells.loc[[game5_opt_cell], ].set_geometry('Centroids').plot(ax=axs[i], color='white', marker='$O$', markersize=25)
    axs[i].xaxis.set_tick_params(labelsize=10)
    axs[i].yaxis.set_tick_params(labelsize=10)
    axs[i].set_xlim(x_lim)
    axs[i].set_ylim(y_lim)
fig.supxlabel('Longitude', fontsize=12)
fig.supylabel('Latitude', fontsize=12)
patch_col = axs[0].collections[0]
cb = fig.colorbar(patch_col, ax=axs, location='top', shrink=0.95, aspect=50)
cb.set_label(label='Mixed Strategy', size=12)
plt.savefig('2023_new/ranger_mixed.pdf', dpi=1200, bbox_inches='tight')

# follower big  sim
up = [0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065, 0.007, 0.0075, 0.008, 0.0085, 0.009, 0.0095, 0.01,
      0.0125, 0.015, 0.0175, 0.02, 0.0225, 0.025, 0.0275, 0.03, 0.0325, 0.035, 0.0375, 0.04]
cal_games = []
fit_games = []
for bound in up:
    cal_games.append(Game('GAME Cal ' + str(bound), rhino_strat, ranger_game, poacher_strat, seed=seed, end_moves=end, games_pm=gpm,
             months=months, game_type='ssg_follower', leader='poacher', ssg_high=bound))
    fit_games.append(Game('GAME Fit ' + str(bound), rhino_strat, ranger_game, poacher_strat_spat, seed=seed, end_moves=end, games_pm=gpm,
             months=months, game_type='ssg_follower', leader='poacher', ssg_high=bound))

comb = pd.DataFrame(columns = ['Prob', 'Cal Cells', 'Cal Ranger Util', 'Cal Poacher Util',
                               'Fit Cells', 'Fit Ranger Util', 'Fit Poacher Util'])    
for i in range(len(up)):
    prob = up[i]
    cal = cal_games[i].game_solution()
    cal_cells = str(np.where(cal['ranger_strat']>0)[0])
    cal_ranger_util = cal['ranger_util']
    cal_poacher_util = cal['poacher_util']
    fit = fit_games[i].game_solution()
    fit_cells = str(np.where(fit['ranger_strat']>0)[0])
    fit_ranger_util = fit['ranger_util']
    fit_poacher_util = fit['poacher_util']
    row = pd.DataFrame([[prob, cal_cells, cal_ranger_util, cal_poacher_util, fit_cells, fit_ranger_util, fit_poacher_util]],
                       columns = ['Prob', 'Cal Cells', 'Cal Ranger Util', 'Cal Poacher Util',
                                  'Fit Cells', 'Fit Ranger Util', 'Fit Poacher Util'])
    comb = pd.concat([comb, row], ignore_index=True)
comb.to_csv('2023_new/follower_spatial_util.csv', index=False)

up = [0.0065, 0.0075, 0.008, 0.009, 0.0095, 0.0125, 0.015, 0.0175, 0.02, 0.0225, 0.025, 0.0275, 0.03, 0.0375]
ranger_cells = ranger_game.allowed_cells.copy()
for bound in up:
    cal_game = Game('GAME Cal ' + str(bound), rhino_strat, ranger_game, poacher_strat, seed=seed, end_moves=end, games_pm=gpm,
             months=months, game_type='ssg_follower', leader='poacher', ssg_high=bound)
    cal_sol = cal_game.game_solution()
    ranger_cells['Cal ' + str(bound)] = cal_sol['ranger_strat']
    fit_game = Game('GAME Fit ' + str(bound), rhino_strat, ranger_game, poacher_strat_spat, seed=seed, end_moves=end, games_pm=gpm,
                months=months, game_type='ssg_follower', leader='poacher', ssg_high=bound)
    fit_sol = fit_game.game_solution()
    ranger_cells['Fit ' + str(bound)] = fit_sol['ranger_strat']
ranger_cells['Calculated'] = poacher_strat.allowed_cells['Select Prob']
ranger_cells['Fitted'] = poacher_strat_spat.allowed_cells['Select Prob']
ranger_cells.to_csv('2023_new/follower_spatial_strat.csv', index=True)

# follower small sim
up = [0.0065, 0.0125, 0.0175, 0.0225]
games = []
for bound in up:
    games.append(Game('GAME Cal ' + str(bound), rhino_strat, ranger_game, poacher_strat, seed=seed, end_moves=end, games_pm=gpm,
             months=months, game_type='ssg_follower', leader='poacher', ssg_high=bound))
    games.append(Game('GAME Fit ' + str(bound), rhino_strat, ranger_game, poacher_strat_spat, seed=seed, end_moves=end, games_pm=gpm,
             months=months, game_type='ssg_follower', leader='poacher', ssg_high=bound))

# run simulations
t0 = datetime.now()
print(t0, '\nStarted running simulations')
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = executor.map(sim_games, games)
    sim = []
    for result in results:
        sim.append(result)
t1 = datetime.now()
game_time = t1-t0
print(t1, '\nFinished process, time taken: ', game_time, '\n')

# save workspace
del executor, result, results
filename = '2023_follower_spatial_small.pkl'
dill.dump_session(filename)

# evaluation
game_eval = sim_eval(sim)

with pd.ExcelWriter('2023_new/2023_follower_spatial_small.xlsx', engine='xlsxwriter') as writer:
    game_eval['games'].to_excel(writer, sheet_name='games')
    game_eval['months'].to_excel(writer, sheet_name='totals')
    game_eval['metrics'].to_excel(writer, sheet_name='metrics')
    game_eval['descrip'].to_excel(writer, sheet_name='descrip')
    game_eval['med_pval'].to_excel(writer, sheet_name='med_pval')
