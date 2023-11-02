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
ranger_stay = Ranger('ranger_game', park, 'bound', 'game', out=out,
                     like={'rivers': 8000, 'dams': 8000, 'water': 8000},
                     geo_util_fctr=2, arrest_util=50, wild_save_fctr=6, wild_calve_fctr=0, path_type='stay')
ranger_continue = Ranger('ranger_game', park, 'bound', 'game', out=out,
                     like={'rivers': 8000, 'dams': 8000, 'water': 8000},
                     geo_util_fctr=2, arrest_util=50, wild_save_fctr=6, wild_calve_fctr=0, path_type='continue')

# poachers
poacher_game = Poacher('poacher_game', park, 'bound', 'game', out=out,
                       dislike={'camps': 1000, 'gates': 1000, 'picnic_spots': 1000},
                       like={'roads': 2500, 'border': 10000, 'rivers': 8000, 'dams': 8000, 'water': 8000},
                       geo_util_fctr=2, arrest_util=50, wild_save_fctr=6, wild_calve_fctr=0, entry_type='edge')
poacher_strat_edge = Poacher('poacher_game', park, 'bound', 'strategic', out=out,
                       dislike={'camps': 1000, 'gates': 1000, 'picnic_spots': 1000},
                       like={'roads': 2500, 'border': 10000, 'rivers': 8000, 'dams': 8000, 'water': 8000},
                       geo_util_fctr=2, arrest_util=50, wild_save_fctr=6, wild_calve_fctr=0, entry_type='edge')
poacher_strat_border = Poacher('poacher_game', park, 'bound', 'strategic', out=out,
                       dislike={'camps': 1000, 'gates': 1000, 'picnic_spots': 1000},
                       like={'roads': 2500, 'border': 10000, 'rivers': 8000, 'dams': 8000, 'water': 8000},
                       geo_util_fctr=2, arrest_util=50, wild_save_fctr=6, wild_calve_fctr=0, entry_type='border')

# games
seed = 1024
end = 50 # 1 day - 18hr movement at 4km/hr
gpm = 30
months = 500

bound = [0.02, 0.04, 0.06, 0.08, 0.1, 1/9, 0.125, 1/7, 1/6, 0.2, 0.25, 1/3, 0.5, 1]
edge_games = list()
border_games = list()
for count, value in enumerate(bound):
    edge_games.append(Game('GAME' + str(count), rhino_strat, ranger_stay, poacher_strat_edge, seed=seed, end_moves=end, 
                           games_pm=gpm, months=months, game_type='ssg_follower', leader='poacher', ssg_high=value))
    border_games.append(Game('GAME' + str(count), rhino_strat, ranger_stay, poacher_strat_border, seed=seed, end_moves=end, 
                           games_pm=gpm, months=months, game_type='ssg_follower', leader='poacher', ssg_high=value))
    
# run edge games
t0 = datetime.now()
print(t0, '\nStarted running simulations')
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = executor.map(sim_games, edge_games)
    sim1 = []
    for result in results:
        sim1.append(result)
t1 = datetime.now()
edge_time = t1-t0
print(t1, '\nFinished process, time taken: ', edge_time, '\n')

# save workspace
del executor, result, results
filename = '2023_follower_stay_intelligent_edge.pkl'
dill.dump_session(filename)

# evaluation
game_eval1 = sim_eval(sim1)

with pd.ExcelWriter('2023_new/2023_follower_edge.xlsx', engine='xlsxwriter') as writer:
    game_eval1['games'].to_excel(writer, sheet_name='games')
    game_eval1['months'].to_excel(writer, sheet_name='totals')
    game_eval1['metrics'].to_excel(writer, sheet_name='metrics')
    game_eval1['descrip'].to_excel(writer, sheet_name='descrip')
    game_eval1['med_pval'].to_excel(writer, sheet_name='med_pval')

# run border games
t0 = datetime.now()
print(t0, '\nStarted running simulations')
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = executor.map(sim_games, border_games)
    sim2 = []
    for result in results:
        sim2.append(result)
t1 = datetime.now()
border_time = t1-t0
print(t1, '\nFinished process, time taken: ', border_time, '\n')

# save workspace
del executor, result, results
filename = '2023_follower_stay_intelligent_border.pkl'
dill.dump_session(filename)

# evaluation
game_eval2 = sim_eval(sim2)

with pd.ExcelWriter('2023_new/2023_follower_border.xlsx', engine='xlsxwriter') as writer:
    game_eval2['games'].to_excel(writer, sheet_name='games')
    game_eval2['months'].to_excel(writer, sheet_name='totals')
    game_eval2['metrics'].to_excel(writer, sheet_name='metrics')
    game_eval2['descrip'].to_excel(writer, sheet_name='descrip')
    game_eval2['med_pval'].to_excel(writer, sheet_name='med_pval')


# EXAMPLE

game = Game('GAME', rhino_strat, ranger_stay, poacher_game, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='ssg_follower', leader='poacher', ssg_high=1) 

util = np.zeros((len(bound), 3))
cells = list()
for count, value in enumerate(bound):
    game.ssg_high = value
    game_sol = game.game_solution()
    util[count, 0] = value
    util[count, 1] = game_sol['ranger_util']
    util[count, 2] = game_sol['poacher_util']
    cells.append(game_sol['ranger_strat'])
np.round(util, 3)

# get dobss for poacher leader
u1 = np.transpose(game.payoffs['poacher'])
u2 = np.transpose(game.payoffs['ranger'])
dobss_poacher = gt.dobss(u1, u2)

# assume poachers play dobss strategy, get ranger follower strategy
x1 = dobss_poacher[0]
u1 = game.payoffs['poacher']
u2 = game.payoffs['ranger']
m = u2.shape[0]
util = np.matmul(u2, x1)
max_util = util.max()
support = np.where(np.round(util, 3) == np.round(max_util, 3))[0]

comb = pd.DataFrame(columns = ['Cells', 'Prob', 'Ranger Util', 'Poacher Util'])
for i in range(1, len(support)+1):
    for j in itertools.combinations(support, i):
        cells = j
        prob = 1/i
        x2 = np.zeros(m)
        x2[list(cells)] = prob
        exp_util = gt.exp_util(x1, u1, x2, u2, 'poacher')
        row = pd.DataFrame([[str(cells), prob, exp_util[3], exp_util[1]]],
                           columns = ['Cells', 'Prob', 'Ranger Util', 'Poacher Util'])
        comb = pd.concat([comb, row], ignore_index=True)
comb.to_csv('2023_new/ranger_follower.csv', index=False)

# follower game
follower_pure = game.game_solution()

# follower suboptimal mixed strategy setting upper bound
support = np.where(follower_pure['util']>20)
bound = 1/len(support[0])
game.ssg_high = bound
follower = game.game_solution()
