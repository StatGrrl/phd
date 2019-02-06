# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 15:45:57 2016

Plot path of one game for each method. 
Choose same starting points of rhinos, ranger and poacher for each method.

@author: Lisa
"""

# import packages
import numpy as np
import random
import matplotlib.pyplot as plt

# function for cell perimeters
def cell_per(vec, step, max_size, type_per):
    per = []
    if (step==0):
        per = [vec]
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
            per.extend(curr_per)
            per = [list(x) for x in set(tuple(x) for x in per)]
        if type_per == 'step' and vec in per:
            per.remove(vec)
    return per

# function to return euclidean distance between two vectors
def distance(vec1, vec2):
    return ((vec1[0]-vec2[0])**2 + (vec1[1]-vec2[1])**2)**(0.5)

# function to select next cell going towards a heading
def heading_cell(heading, cell, grid_size, method, cov=[[0.1, 0], [0, 0.1]]):
    step = cell_per(cell, 1, grid_size, 'step')
    if heading in(step):
        select_step = heading
    else:
        if method == 'norm':
            dist_cell = distance(heading, cell)
            norm_coord = [grid_size, grid_size]
            dist_norm = distance(norm_coord, [0,0])
            while norm_coord not in(step) or dist_norm >= dist_cell:
                norm_bv = np.random.multivariate_normal(cell, cov, 1)
                norm_coord = [round(abs(norm_bv[0][0])), round(abs(norm_bv[0][1]))]
                dist_norm = distance(heading, norm_coord)
            select_step = norm_coord
        else:
            dist = [distance(heading, x) for x in step]
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

# function for one game - path returned
def one_game(game_type, grid_size, rhino_cell, rhino2_cell, ranger_cell, poacher_cell, 
             seed=123, method='null', prev_moves=0, start_end='null'):
    random.seed(seed)
    # step size
    step_size = 1

    # define edge of grid
    edge_cells = []
    for i in range(grid_size):
        for j in range(grid_size):
            if i in(0, grid_size-1) or j in(0, grid_size-1):
                edge_cells.append([i,j])
    
    # Heading game only
    # heading for ranger and poacher, start cell and end cell
    if game_type == 'Heading':
        ranger_heading = [random.randint(0,grid_size-1), random.randint(0, grid_size-1)]
        poacher_heading = [random.randint(0,grid_size-1), random.randint(0, grid_size-1)]
        ranger_start = ranger_cell
        poacher_start = poacher_cell
        # end cell same as start cell or different random edge cell
        if start_end == 'same':
            ranger_end = ranger_cell
            poacher_end = poacher_cell
        else:
            ranger_end = edge_cells[random.randint(0,len(edge_cells)-1)]
            poacher_end = edge_cells[random.randint(0,len(edge_cells)-1)]
                 
    # record path of rhino, ranger and poacher
    rhino_path = [rhino_cell]
    rhino2_path = [rhino2_cell]
    ranger_path = [ranger_cell]
    poacher_path = [poacher_cell]
    poach = 0
    end = 0
    
    # select minimum moves of poacher and keep track of number of moves
    min_moves = random.randint(10,30)
    move = 0

    # Avoid cyclic game only    
    # choose for how many moves ranger or poacher can't return to previous cells
    if game_type == 'Avoid Cyclic':
        if method == 'rand':
            ranger_moves_avoid = random.randint(5,10)
            poacher_moves_avoid = random.randint(5,10)
        else:
            ranger_moves_avoid = prev_moves
            poacher_moves_avoid = prev_moves

    while end == 0:

        # Rhino moves
        # Null game: only 1 rhino and rhino cannot move after it has been poached
        if game_type == 'Null' and poach == 0:
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
        if game_type == 'Heading':
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
            # ranger and poacher select the next cell with probability which is
            # inversely proportional to the distance that the cell has from the heading cell
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
            if game_type == 'Null' and poach == 0:
                poach = poach + 1
            if game_type != 'Null':
                poach = poach + 1
        if rhino2_cell == poacher_cell: 
            poach = poach + 1

        # Check for capture event
        # Ranger viewing cells 0 for null game, random for other games
        if game_type == 'Null':
            view_size = 0
        else:
            view_size = random.randint(0,3)
        ranger_view = cell_per(ranger_cell, view_size, grid_size, 'view')
        if poacher_cell in ranger_view:
            end = 1
            if poach > 0:
                event = 'Catch After Poach'
            else:
                event = 'Catch Before Poach'
        
        # Check for leaving event
        if poacher_cell in edge_cells:
            if poach > 0:
                event = 'Leave After Poach'
                end = 1
            else:
                if game_type == 'Null':
                    event = 'Leave Before Poach'
                    end = 1
                if game_type != 'Null' and move > min_moves:
                    event = 'Leave Before Poach'
                    end = 1
        move = move + 1

        
    # end of while loop - game ends - return end event, no poaches and paths
    return {'game':game_type, 'size':grid_size, 'event':event, 'poachings':poach, 'tot_moves':move,
    'method':method, 'prev_moves':prev_moves, 'start_end':start_end, 
    'rhino':rhino_path, 'rhino2':rhino2_path, 'ranger':ranger_path, 'poacher':poacher_path}

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
    filename = game_result['game'] + '_paths'
    if game_result['game'] == 'Avoid Cyclic':
        plt.figtext(0.95, 0.25, 'Game Conditions:', style='italic')
        if game_result['method'] == 'rand':
            plt.figtext(0.95, 0.2, 'Randomize Over')
            plt.figtext(0.95, 0.15, '5-10 Prev Moves')
        else:
            plt.figtext(0.95, 0.2, 'Fixed')
            plt.figtext(0.95, 0.15, str(game_result['prev_moves']) + ' Prev Moves')
            filename = filename + '_' + str(game_result['prev_moves'])
    if game_result['game'] == 'Heading':
        plt.figtext(0.95, 0.25, 'Game Conditions:', style='italic')
        if game_result['method'] == 'rand':
            plt.figtext(0.95, 0.2, 'Probability Sampling')
        elif game_result['method'] == 'max':
            plt.figtext(0.95, 0.2, 'Maximum Probability')
        else:
            plt.figtext(0.95, 0.2, 'Bivariate Normal')
        if game_result['start_end'] == 'same':
            plt.figtext(0.95, 0.15, 'Same Start & End')
        else:
            plt.figtext(0.95, 0.15, 'Different Start & End')
        filename = filename + '_' + game_result['method'] + '_' + game_result['start_end']
    filename = filename + '.png'
    fig.savefig(filename, dpi=200, bbox_extra_artists=(lgd,), bbox_inches='tight')

# declare constants
grid_size = 100
rhino_start = [24,74]
rhino2_start = [63,44]
ranger_start = [0,52]
poacher_start = [30,99]
prev_moves = 20
seed = 62152626

# null game
null = one_game('Null', grid_size, rhino_start, rhino2_start, 
                ranger_start, poacher_start, seed)
plot_path(null)

# more realistic null game
null_real = one_game('Realistic Null', grid_size, rhino_start, rhino2_start, 
                ranger_start, poacher_start, seed)
plot_path(null_real)

# avoid cyclic movement by staying out of previous cells - randomly choose number of moves 5-10
cyclic = one_game('Avoid Cyclic', grid_size, rhino_start, rhino2_start, 
                  ranger_start, poacher_start, seed, 'rand')
plot_path(cyclic)

# avoid cyclic movement by staying out of previous cells - specify number of moves
cyclic_fixed = one_game('Avoid Cyclic', grid_size, rhino_start, rhino2_start, 
                        ranger_start, poacher_start, seed, 'null', prev_moves)
plot_path(cyclic_fixed)

# Heading games, Max Same, Max Diff, Rand Same, Rand Diff, Norm Same, Norm Diff
head_rand_same = one_game('Heading', grid_size, rhino_start, rhino2_start, 
                          ranger_start, poacher_start, seed, 'rand', 0, 'same')
plot_path(head_rand_same)

head_rand_diff = one_game('Heading', grid_size, rhino_start, rhino2_start, 
                          ranger_start, poacher_start, seed, 'rand', 0, 'diff')
plot_path(head_rand_diff)

head_max_same = one_game('Heading', grid_size, rhino_start, rhino2_start, 
                         ranger_start, poacher_start, seed, 'max', 0, 'same')
plot_path(head_max_same)

head_max_diff = one_game('Heading', grid_size, rhino_start, rhino2_start, 
                         ranger_start, poacher_start, seed, 'max', 0, 'diff')
plot_path(head_max_diff)

head_norm_same = one_game('Heading', grid_size, rhino_start, rhino2_start, 
                         ranger_start, poacher_start, seed, 'norm', 0, 'same')
plot_path(head_norm_same)

head_norm_diff = one_game('Heading', grid_size, rhino_start, rhino2_start, 
                         ranger_start, poacher_start, seed, 'norm', 0, 'diff')
plot_path(head_norm_diff)

