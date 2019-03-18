# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 18:15:46 2019

@author: Lisa
"""
# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt    
import random
import nasheq

# function to get all cells in the perimeter of a specific cell
def cell_per(vec, step=1, max_size=10, type_per='step'):
    per_cells = []
    if (step==0):
        per_cells = [vec]
    else:
        for i in range(1, step+1):
            if i == 1:
                curr_per = [[vec[0]-i, vec[1]-i],
                [vec[0]-i, vec[1]],
                [vec[0]-i, vec[1]+i],
                [vec[0], vec[1]-i],
                [vec[0], vec[1]+i],
                [vec[0]+i, vec[1]-i],
                [vec[0]+i, vec[1]],
                [vec[0]+i, vec[1]+i]]
                curr_per = [list(x) for x in curr_per if all(y>=0 for y in x)]
                curr_per = [list(x) for x in curr_per if all(y<max_size for y in x)]
                curr_per.append(vec)
            else:
                next_per = []
                for j in range(len(curr_per)):
                    next_per.extend(cell_per(curr_per[j], 1, max_size, 'step'))
                curr_per = [list(x) for x in set(tuple(x) for x in next_per)]
            per_cells.extend(curr_per)
            per_cells = [list(x) for x in set(tuple(x) for x in per_cells)]
        if type_per == 'step' and vec in per_cells:
            per_cells.remove(vec)
    return per_cells

# function to get all cells in the edge of the grid
def grid_edge(max_size=10):
    edge_cells = []
    for i in range(max_size):
        for j in range(max_size):
            if i in(0, max_size-1) or j in(0, max_size-1):
                edge_cells.append([i,j])
    return edge_cells

# function to show cells in the grid
def make_grid(vec=None, step=1, max_size=10, type_per='step'):
    grid = np.zeros(shape=(max_size, max_size))
    if not vec is None:
        cells = cell_per(vec, step, max_size, type_per)
        grid[tuple(vec)] = 0.5
    else:
        cells = grid_edge(max_size)
    for x in cells:
        grid[tuple(x)] = 1
    return grid

# function to show grid subplots
def grid_plot(grids, xlab, file_name=''):
    nrows = 1
    ncols = len(grids)
    max_size = len(grids[0])
    fig = plt.figure()
    for i in range(ncols):
        ax = fig.add_subplot(nrows, ncols, (i+1))
        ax.set_xticks(np.arange(0.5, (max_size-0.5), 1), minor=True)
        ax.set_yticks(np.arange(0.5, (max_size-0.5), 1), minor=True)
        ax.grid(b=True, which='minor', lw=1, ls='solid')
        plt.xlabel(xlab[i])
        plt.tick_params(which='major', length=0)
        ax.imshow(grids[i], cmap='Blues', interpolation='nearest', origin='lower')
    fig.savefig(file_name, dpi=200)

# function to return euclidean distance between two vectors
def distance(vec1, vec2):
    return ((vec1[0]-vec2[0])**2 + (vec1[1]-vec2[1])**2)**(0.5)

# function to select next cell going towards a heading
def heading_cell(heading, cell, grid_size, method, cov=[[0.1, 0], [0, 0.1]]):
    step = cell_per(cell, 1, grid_size, 'step')
    if heading in(step) or heading == cell:
        select_step = heading
    else:
        dist_cell = distance(heading, cell)
        if method == 'norm':
            norm_coord = [grid_size, grid_size]
            dist_norm = distance(norm_coord, [0,0])
            while norm_coord not in(step) or dist_norm >= dist_cell:
                norm_bv = np.random.multivariate_normal(cell, cov, 1)
                norm_coord = [round(abs(norm_bv[0][0])), round(abs(norm_bv[0][1]))]
                dist_norm = distance(heading, norm_coord)
            select_step = norm_coord
        else:
            dist = [distance(heading, x) for x in step]
            ind = [dist.index(x) for x in dist if x < dist_cell]
            dist = [dist[x] for x in ind]
            step = [step[x] for x in ind]
            dist_inv = [1/x for x in dist]
            dist_inv_tot = sum(dist_inv)
            prob = [(step[x], dist_inv[x]/dist_inv_tot) for x in range(len(step))]
            prob_struct = np.array(prob, dtype=[('step',list),('prob',float)])
            prob_sort = np.sort(prob_struct, order='prob')
            if method=='max':
                select_step = prob_sort['step'][len(prob_sort)-1]
            else:
                prob_cumm = [0]
                for i in range(len(step)):
                    prob_cumm.append(prob_cumm[i] + prob_sort['prob'][i])
                prob_cumm.pop(0)
                select_prob = random.random()
                select_step = prob_sort['step'][np.array(prob_cumm) > select_prob][0]
    return select_step

# function to get solution for pure stackelberg game
def stackel_pure(grid_size, rhino_cell, rhino2_cell, leader):
    # get strategies
    strat_list = []
    for i in range(grid_size):
        for j in range(grid_size):
            strat_list.append([i, j])
    # get payoffs
    strat_len = len(strat_list)
    rhino_loc = strat_list.index(rhino_cell)
    rhino2_loc = strat_list.index(rhino2_cell)
    ranger_payoff = np.zeros(shape=(strat_len, strat_len))
    poacher_payoff = np.zeros(shape=(strat_len, strat_len))
    for i in range(strat_len):
        for j in range(strat_len):
            if i == rhino_loc or i == rhino2_loc:
                ranger_payoff[i,j] += 1
            if j == rhino_loc or j == rhino2_loc:
                poacher_payoff[i,j] += 1
            if i == j:
                ranger_payoff[i,j] += 1
                poacher_payoff[i,j] = -1
    # get subgame perfect nash equilibrium
    equilibrium = nasheq.spne(strat_list, strat_list, ranger_payoff, poacher_payoff, 'p1')    
    return {'ranger_cell':equilibrium[0][0], 'poacher_cell':equilibrium[0][1],
            'ranger_payoff':equilibrium[1][0], 'poacher_payoff':equilibrium[1][1]}

# function for one game
def one_game(game_type='Null', method=None, 
         step_size=1, view_size=0, grid_size=100, min_moves=None, 
         rhino_cell=None, rhino2_cell=None, ranger_cell=None, poacher_cell=None,
         start_end=None, ranger_heading=None, poacher_heading=None, ranger_end=None, poacher_end=None, #heading only
         ranger_prev_moves=0, poacher_prev_moves=0, seed=None): #avoid cyclic only
    
    '''
    game_type = 'Null'
    game_type = 'Realistic Null'
    game_type = 'Avoid Cyclic', method in('rand', 'fixed')
    game_type = 'Heading', method in('max', 'prob', 'norm'), start_end in('same', 'diff')
    game_type = 'SPNE', method in('max', 'prob', 'norm'), start_end in('same', 'diff')
    '''
    
    # set seed
    if not seed is None:
        random.seed(seed)

    # define edge of grid
    edge_cells = grid_edge(grid_size)
    
    # start cells
    if rhino_cell is None:
        rhino_cell = [random.randint(0,grid_size-1), random.randint(0, grid_size-1)]
    if rhino2_cell is None:
        rhino2_cell = [random.randint(0,grid_size-1), random.randint(0, grid_size-1)]        
    if ranger_cell is None:
        ranger_cell = edge_cells[random.randint(0,len(edge_cells)-1)]
    if poacher_cell is None:
        poacher_cell = edge_cells[random.randint(0,len(edge_cells)-1)]
            
    # Heading game only
    # heading for ranger and poacher, start cell and end cell
    if game_type == 'Heading' or game_type == 'SPNE':
        if ranger_heading is None:
            ranger_heading = [random.randint(0,grid_size-1), random.randint(0, grid_size-1)]
        if poacher_heading is None:
            poacher_heading = [random.randint(0,grid_size-1), random.randint(0, grid_size-1)]
        ranger_start = ranger_cell
        poacher_start = poacher_cell
        # end cell same as start cell or different random edge cell
        if start_end == 'same':
            ranger_end = ranger_cell
            poacher_end = poacher_cell
        else:
            if ranger_end is None:
                ranger_end = edge_cells[random.randint(0,len(edge_cells)-1)]
            if poacher_end is None:
                poacher_end = edge_cells[random.randint(0,len(edge_cells)-1)]
                 
    # record path of rhino, ranger and poacher
    ind = ['Moves', 'PoachEvents', 'CaptureEvents', 
           'LeaveBefore', 'LeaveAfter', 
           'CatchBefore', 'CatchAfter']
    results_game = pd.Series(0, ind)
    
    rhino_path = [rhino_cell]
    rhino2_path = [rhino2_cell]
    ranger_path = [ranger_cell]
    poacher_path = [poacher_cell]
    end = 0
    
    # select minimum moves of poacher and keep track of number of moves
    if min_moves is None:
        min_moves = random.randint(10,30)

    # Avoid cyclic game only    
    # choose for how many moves ranger or poacher can't return to previous cells
    if game_type == 'Avoid Cyclic':
        if method == 'rand':
            ranger_moves_avoid = random.randint(5,10)
            poacher_moves_avoid = random.randint(5,10)
        else:
            ranger_moves_avoid = ranger_prev_moves
            poacher_moves_avoid = poacher_prev_moves

    while end == 0:

        # Rhino moves
        # Null game: only 1 rhino and rhino cannot move after it has been poached
        if game_type == 'Null' and results_game['PoachEvents'] == 0:
            rhino_step = cell_per(rhino_cell, step_size, grid_size, 'step')   # possible cells
            rhino_cell = rhino_step[random.randint(0,len(rhino_step)-1)]      # next cell
            rhino_path.append(rhino_cell)                                     # record path
        # Other games: allow for rhino to stay in current cell and the herds keeps moving after a poach
        if game_type != 'Null':
            rhino_step = cell_per(rhino_cell, step_size, grid_size, 'step')   # herd 1 possible moves
            rhino_step.append(rhino_cell)                                     # herd 1 include current cell
            rhino_cell = rhino_step[random.randint(0,len(rhino_step)-1)]      # herd 1 next cell
            rhino_path.append(rhino_cell)                                     # herd 1 record path
            rhino2_step = cell_per(rhino2_cell, step_size, grid_size, 'step') # herd 2 possible moves
            rhino2_step.append(rhino2_cell)                                   # herd 2 include current cell
            rhino2_cell = rhino2_step[random.randint(0,len(rhino2_step)-1)]   # herd 2 next cell
            rhino2_path.append(rhino2_cell)                                   # herd 2 record path

        # Ranger and poacher possible cells
        ranger_step = cell_per(ranger_cell, step_size, grid_size, 'step')
        poacher_step = cell_per(poacher_cell, step_size, grid_size, 'step')
        
        # Avoid cyclic game only         
        # remove previous cells from list of possible steps
        if game_type == 'Avoid Cyclic':
            ranger_remove = ranger_path[-ranger_moves_avoid:]
            while all(x in ranger_remove for x in ranger_step):
                ranger_moves_avoid -= 1
                ranger_remove = ranger_path[-ranger_moves_avoid:]
            ranger_step = [x for x in ranger_step if x not in ranger_remove]
            poacher_remove = poacher_path[-poacher_moves_avoid:]
            while all(x in poacher_remove for x in poacher_step):
                poacher_moves_avoid -= 1
                poacher_remove = poacher_path[-poacher_moves_avoid:]
            poacher_step = [x for x in poacher_step if x not in poacher_remove]
    
        # Ranger and poacher next cell
        if game_type == 'Heading' or game_type == 'SPNE':
            # if current cell is the heading cell then change heading, start and end cells
            if ranger_cell == ranger_heading:
                    ranger_heading = ranger_end
                    if ranger_end == ranger_start:
                        ranger_end = ranger_cell
                    else: 
                        ranger_end = ranger_start
                    ranger_start = ranger_cell
            if poacher_cell == poacher_heading:
                    poacher_heading = poacher_end
                    if poacher_end == poacher_start:
                        poacher_end = poacher_cell
                    else: 
                        poacher_end = poacher_start
                    poacher_start = poacher_cell
            # ranger and poacher select the next cell 
            ranger_cell = heading_cell(ranger_heading, ranger_cell, grid_size, method)
            poacher_cell = heading_cell(poacher_heading, poacher_cell, grid_size, method)
        else:
            ranger_cell = ranger_step[random.randint(0,len(ranger_step)-1)]
            poacher_cell = poacher_step[random.randint(0,len(poacher_step)-1)]

        # Record paths of ranger and poacher
        ranger_path.append(ranger_cell)
        poacher_path.append(poacher_cell)

        # Check for poaching event
        # Null game: rhino dead after 1st poaching event
        # other games: rhino herds can keep moving after a poaching event
        if rhino_cell == poacher_cell:
            if game_type == 'Null' and results_game['PoachEvents'] == 0:
                results_game['PoachEvents'] = results_game['PoachEvents'] + 1
            if game_type != 'Null':
                results_game['PoachEvents'] = results_game['PoachEvents'] + 1
        if rhino2_cell == poacher_cell: 
            results_game['PoachEvents'] = results_game['PoachEvents'] + 1

        # Check for capture event
        # Ranger viewing cells 0 for null game, random for other games
        if game_type == 'Null':
            view_size = 0
        elif view_size is None:
            view_size = random.randint(0,3)
        ranger_view = cell_per(ranger_cell, view_size, grid_size, 'view')
        if poacher_cell in ranger_view:
            end = 1
            results_game['CaptureEvents']  = 1
            if results_game['PoachEvents'] > 0:
                results_game['CatchAfter']  = 1
                event = 'Catch After Poach'
            else:
                results_game['CatchBefore']  = 1
                event = 'Catch Before Poach'
        
        # Check for leaving event
        if poacher_cell in edge_cells:
            if results_game['PoachEvents']  > 0:
                results_game['LeaveAfter']  = 1
                event = 'Leave After Poach'
                end = 1
            else:
                if game_type == 'Null':
                    results_game['LeaveBefore']  = 1
                    event = 'Leave Before Poach'
                    end = 1
                if game_type != 'Null' and results_game['Moves'] > min_moves:
                    results_game['LeaveBefore']  = 1
                    event = 'Leave Before Poach'
                    end = 1
        results_game['Moves'] = results_game['Moves'] + 1

    # end of while loop - game ends - return results and paths
    return {'results':results_game, 'rhino':rhino_path, 'rhino2':rhino2_path, 
            'ranger':ranger_path, 'poacher':poacher_path,
            'game':game_type, 'size':grid_size, 'event':event, 'poachings':results_game['PoachEvents'], 
            'tot_moves': results_game['Moves'], 'method':method, 'start_end':start_end,
            'ranger_prev_moves':ranger_prev_moves, 'poacher_prev_moves':poacher_prev_moves}
    
# function to plot paths for one game
def plot_path(game_result):
    fig = plt.figure()
    plt.plot()
    plt.xlim(-1, game_result['size'])
    plt.ylim(-1, game_result['size'])
    plt.title(game_result['game'] + ' Game')
    rhino_points = plt.Polygon(game_result['rhino'], closed=None, fill=None, edgecolor='g', label='rhino')
    if game_result['game'] != 'Null':
        rhino2_points = plt.Polygon(game_result['rhino2'], closed=None, fill=None, edgecolor='y', label='rhino2')
    ranger_points = plt.Polygon(game_result['ranger'], closed=None, fill=None, edgecolor='b', label='ranger')
    poacher_points = plt.Polygon(game_result['poacher'], closed=None, fill=None, edgecolor='r', label='poacher')
    plt.gca().add_patch(rhino_points)
    if game_result['game'] != 'Null':
        plt.gca().add_patch(rhino2_points)
    plt.gca().add_patch(ranger_points)
    plt.gca().add_patch(poacher_points)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1.025), loc='upper left', handletextpad=1.5, borderpad=1.05)
    plt.figtext(0.95, 0.5, 'Game Ending:', style='italic')
    plt.figtext(0.95, 0.45, game_result['event'])
    plt.figtext(0.95, 0.4, str(game_result['tot_moves']) + ' Moves')
    plt.figtext(0.95, 0.35, str(game_result['poachings']) + ' Poachings')
    filename = 'paths_' + game_result['game']
    if game_result['game'] == 'Avoid Cyclic':
        plt.figtext(0.95, 0.25, 'Game Conditions:', style='italic')
        if game_result['method'] == 'rand':
            plt.figtext(0.95, 0.2, 'Randomize Over')
            plt.figtext(0.95, 0.15, '5-10 Prev Moves')
        else:
            plt.figtext(0.95, 0.2, 'Fixed')
            plt.figtext(0.95, 0.15, str(game_result['ranger_prev_moves']) + ' Prev Moves')
            filename = filename + '_' + str(game_result['ranger_prev_moves'])
    if game_result['game'] == 'Heading' or game_result['game'] == 'SPNE':
        plt.figtext(0.94, 0.25, 'Game Conditions:', style='italic')
        if game_result['method'] == 'rand':
            plt.figtext(0.94, 0.2, 'Probability Sampling')
        elif game_result['method'] == 'max':
            plt.figtext(0.94, 0.2, 'Maximum Probability')
        else:
            plt.figtext(0.94, 0.2, 'Bivariate Normal')
        if game_result['start_end'] == 'same':
            plt.figtext(0.94, 0.15, 'Same Start & End')
        else:
            plt.figtext(0.94, 0.15, 'Different Start & End')
        filename = filename + '_' + game_result['method'] + '_' + game_result['start_end']
    filename = filename + '.png'
    fig.savefig(filename, dpi=200, bbox_extra_artists=(lgd,), bbox_inches='tight')

# function for simulations
def sim(months=1000, games_pm=10, seed=123, game_type='Null', method=None, 
        step_size=1, view_size=0, grid_size=100, min_moves=None, start_end=None, 
        rhino_start=None, rhino2_start=None, ranger_start=None, poacher_start=None,
        ranger_heading=None, poacher_heading=None, ranger_end=None, poacher_end=None,
        ranger_prev_moves=0, poacher_prev_moves=0):    

    # ITERATIONS

    # set seed
    if not seed is None:
        random.seed(seed)
    
    # define edge of grid
    edge_cells = grid_edge(grid_size)   
    
    # create empty data frame for results
    ind_dat = [range(1, months+1), range(1, games_pm+1)]
    ind = pd.MultiIndex.from_product(ind_dat, names=['Month', 'Game'])
    col = ['Moves', 'PoachEvents', 'CaptureEvents', 
           'LeaveBefore', 'LeaveAfter', 
           'CatchBefore', 'CatchAfter']
    results = pd.DataFrame(0.0, index=ind, columns=col)
    
    # starting cells same for all iterations
    if rhino_start is None:
        rhino_cell = [random.randint(0,grid_size-1), random.randint(0, grid_size-1)]
    else:
        rhino_cell = rhino_start
    if rhino2_start is None:
        rhino2_cell = [random.randint(0,grid_size-1), random.randint(0, grid_size-1)]
    else:
        rhino2_cell = rhino2_start
    if ranger_start is None:
        ranger_cell = edge_cells[random.randint(0,len(edge_cells)-1)]
    else:
        ranger_cell = ranger_start
    if poacher_start is None:
        poacher_cell = edge_cells[random.randint(0,len(edge_cells)-1)]
    else:
        poacher_cell = poacher_start
    
    # month iterations
    for month in range(1, months+1):
        
        # calculate game solution for the month
        if game_type == 'SPNE':
            game_sol = stackel_pure(grid_size, rhino_cell, rhino2_cell, 'p1')
            ranger_heading = game_sol['ranger_cell']
            poacher_heading = game_sol['poacher_cell']        

        # game iterations
        for game in range(1, games_pm+1):
            game_res = one_game(game_type, method, step_size, view_size, grid_size, 
                                min_moves, rhino_cell, rhino2_cell, ranger_cell, 
                                poacher_cell, start_end, ranger_heading, 
                                poacher_heading, ranger_end, poacher_end,
                                ranger_prev_moves, poacher_prev_moves)
            results.loc[month, game] = game_res['results']
        # end of game
    # end of month
    
    # results
    results.applymap("{0:.3f}".format)
    sum_per_month = results.groupby(by='Month').sum()
    ave_per_month = sum_per_month.mean()
    return {'all':results, 'months':sum_per_month, 'ave':ave_per_month}