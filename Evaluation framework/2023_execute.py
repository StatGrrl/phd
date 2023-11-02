import os
os.chdir('C:/Users/lisak/OneDrive/PhD2.0/Coding/Simulations/')

# load previous session
import dill
filename = '2023_spne.pkl'
dill.load_session(filename)
del adj, animal_col, args, ax, boundary, cell, end, fig, game1, game2, game3, game4, game5, game6
del game4_sol, gpm, i, kruger_area, months, poacher_follow, poacher_lead, poacher_rand, poacher_strat
del ranger_follow, ranger_lead, ranger_game, ranger_rand, rhino_rand, rhino_area, rhino_cells, rhino_counts
del rhino_select_tot, rhino_sum, rhino_totals, sim, subarea_perc, t0, t1, total_col, wr_max, x_lim, y_lim

sns.set(style="dark")

# totally random start
ranger_random = Ranger('random', park, 'bound', 'game', out=out,
                    like={'rivers': 8000, 'dams': 8000, 'water': 8000},
                    geo_util_fctr=2, arrest_util=50, wild_save_fctr=6, wild_calve_fctr=0)

# start at camp and continue back and forth between camp and destination
start = ranger_random.allowed_cells.loc[325, ]
ranger_continue = Ranger('continue', park, 'bound', 'game', out=out, start_cell=start,
                    like={'rivers': 8000, 'dams': 8000, 'water': 8000},
                    geo_util_fctr=2, arrest_util=50, wild_save_fctr=6, wild_calve_fctr=0)

# start at camp and stay at destination
ranger_end = Ranger('end', park, 'bound', 'game', out=out, start_cell=start, path_type='end',
                    like={'rivers': 8000, 'dams': 8000, 'water': 8000},
                    geo_util_fctr=2, arrest_util=50, wild_save_fctr=6, wild_calve_fctr=0)

# start and stay at destination
destination = ranger_random.allowed_cells.loc[game6_sol['ranger_strat'], ]
ranger_stay = Ranger('stay', park, 'bound', 'game', out=out, start_cell=destination, path_type='stay',
                    like={'rivers': 8000, 'dams': 8000, 'water': 8000},
                    geo_util_fctr=2, arrest_util=50, wild_save_fctr=6, wild_calve_fctr=0)

# simulations
seed = 123456
end = 50 # 1 day - 18hr movement at 4km/hr
gpm = 30  
months = 500

game7 = Game('GAME7', rhino_strat, ranger_random, poacher_game, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='spne', leader='poacher', rtn_moves=True, gamesol=game6_sol)
game8 = Game('GAME8', rhino_strat, ranger_continue, poacher_game, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='spne', leader='poacher', rtn_moves=True, gamesol=game6_sol)
game9 = Game('GAME9', rhino_strat, ranger_end, poacher_game, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='spne', leader='poacher', rtn_moves=True, gamesol=game6_sol)
game10 = Game('GAME10', rhino_strat, ranger_stay, poacher_game, seed=seed, end_moves=end, games_pm=gpm, months=months,
             game_type='spne', leader='poacher', rtn_moves=True, gamesol=game6_sol)

t0 = datetime.now()
print(t0, '\nStarted running simulations')
with concurrent.futures.ProcessPoolExecutor() as executor:
    args = [game7, game8, game9, game10]
    results = executor.map(sim_games, args)
    sim = []
    for result in results:
        sim.append(result)
t1 = datetime.now()
print(t1, '\nFinished process, time taken: ', t1-t0, '\n')

# save workspace
del executor, result, results
work_file = '2023_execute.pkl'
dill.dump_session(work_file)

# evaluation
game_eval = sim_eval(sim)
with pd.ExcelWriter('2023_new/2023_execute.xlsx', engine='xlsxwriter') as writer:
    game_eval['months'].to_excel(writer, sheet_name='totals')
    game_eval['metrics'].to_excel(writer, sheet_name='metrics')
    game_eval['descrip'].to_excel(writer, sheet_name='descrip')
    game_eval['med_pval'].to_excel(writer, sheet_name='med_pval')

with pd.ExcelWriter('2023_new/2023_execute_moves.xlsx', engine='xlsxwriter') as writer:
    for s in range(4):
        pd.concat(sim[s]['moves']).to_excel(writer, sheet_name=sim[s]['name'])

# boxplots of measures
sim = pd.read_csv('2023_new/2023_execute_metrics.csv')
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
