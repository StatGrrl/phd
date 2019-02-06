# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 22:14:42 2016

@author: Lisa
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:22:00 2016

@author: Lisa

Smoother path for the null game
Avoid previous cells for a certain number of moves
"""

# define function for cell perimeters
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
    
# import packages
import numpy as np
import random
import os
import time
import matplotlib.pyplot as plt
import pandas as pd

# declare constants
seed = 198
grid_size = 100
step_size = 1
### view_size now randomized
### view_size = 0
month_games = 10
reps = 1000

# set seed
random.seed(seed)

# create grid
# grid = np.zeros(shape=(grid_size,grid_size))

# define edge of grid
edge_cells = []
for i in range(grid_size):
    for j in range(grid_size):
        if i in(0, grid_size-1) or j in(0, grid_size-1):
            edge_cells.append([i,j])
#np.savetxt('edge_cells.csv', edge_cells, delimiter=",", fmt='%d')

# remove files which are appended
### also second rhino herd
files = ['poach_per_game.csv','steps_per_game.csv','rhino_path.txt','rhino2_path.txt','ranger_path.txt','poacher_path.txt']
for fn in files:
    os.remove(fn) if os.path.exists(fn) else None

# ITERATIONS
results = np.zeros(reps, 
                   dtype={'names':['leave_after', 'leave_before', 
                   'catch_after', 'catch_before', 'events'], 
                   'formats':['i1','i1','i1','i1','i2']})

for itr in range(reps):
    start_time = time.time()

    # games in one month
    leave_after_poach = 0
    leave_before_poach = 0
    catch_after_poach = 0
    catch_before_poach = 0
    poach_events = 0
    steps_to_end = []
    poach_per_game = []
    
    for game in range(month_games):
    
        # starting cells for rhino, ranger and poacher
        rhino_cell = [random.randint(0,grid_size-1), random.randint(0, grid_size-1)]
        ### second herd of rhino        
        rhino2_cell = [random.randint(0,grid_size-1), random.randint(0, grid_size-1)]
        ranger_cell = edge_cells[random.randint(0,len(edge_cells)-1)]
        poacher_cell = edge_cells[random.randint(0,len(edge_cells)-1)]
        
        # record path of rhino, ranger and poacher
        rhino_path = [rhino_cell]
        ### second herd of rhino
        rhino2_path = [rhino2_cell]
        ranger_path = [ranger_cell]
        poacher_path = [poacher_cell]
        
        # one game
        poach = 0
        end = 0
        
        ### select minimum moves of poacher and keep track of number of moves
        min_moves = random.randint(10,30)
        move = 1

        # choose for how many moves ranger or poacher can't return to previous cells
        ranger_moves_avoid = random.randint(5,10)
        poacher_moves_avoid = random.randint(5,10)
                
        while end == 0:
            # rhino, ranger and poacher possible step cells
            ### allow for rhino to stay in current cell and the herd keeps moving after a poach
            ### if poach == 0:
            rhino_step = cell_per(rhino_cell, step_size, grid_size, 'step')
            rhino_step.append(rhino_cell)
            ### second herd of rhino
            rhino2_step = cell_per(rhino2_cell, step_size, grid_size, 'step')
            rhino2_step.append(rhino2_cell)
            ranger_step = cell_per(ranger_cell, step_size, grid_size, 'step')
            poacher_step = cell_per(poacher_cell, step_size, grid_size, 'step')
            
            # remove previous cells from list of possible steps
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
            
            # next cell for rhino, ranger and poacher
            ### rhino herd can keep moving after a poaching event
            ### if poach == 0:
            rhino_cell = rhino_step[random.randint(0,len(rhino_step)-1)]
            ### second herd of rhino
            rhino2_cell = rhino2_step[random.randint(0,len(rhino2_step)-1)]
            ranger_cell = ranger_step[random.randint(0,len(ranger_step)-1)]
            poacher_cell = poacher_step[random.randint(0,len(poacher_step)-1)]
        
            # record path of rhino, ranger and poacher
            ### rhino herd can keep moving after a poaching event
            ### if poach == 0:
            rhino_path.append(rhino_cell)
            ### second herd of rhino
            rhino2_path.append(rhino2_cell)
            ranger_path.append(ranger_cell)
            poacher_path.append(poacher_cell)
        
            # ranger viewing cells
            ### randomize view size of ranger
            view_size = random.randint(0,3)
            ranger_view = cell_per(ranger_cell, view_size, grid_size, 'view')
        
            # check for poaching event
            ### rhino herd can keep moving after a poaching event
            if rhino_cell == poacher_cell: ### and poach == 0:
                poach = poach + 1
                poach_events = poach_events + 1
            ### also check second herd of rhino
            if rhino2_cell == poacher_cell: ### and poach == 0:
                poach = poach + 1
                poach_events = poach_events + 1
            
            # check for capture event
            if poacher_cell in ranger_view:
                if poach > 0:
                    catch_after_poach = catch_after_poach + 1
                else:
                    catch_before_poach = catch_before_poach + 1
                    # save one path for example plot                    
                    rhino_plot = rhino_path
                    ### second herd of rhino
                    rhino2_plot = rhino2_path
                    ranger_plot = ranger_path
                    poacher_plot = poacher_path
                end = 1
            
            # check for leaving event
            ### impose minimum moves limitation of poacher
            if poacher_cell in edge_cells:
                if poach > 0:
                    leave_after_poach = leave_after_poach + 1
                    end = 1
                if poach == 0 and move > min_moves:
                    leave_before_poach = leave_before_poach + 1
                    end = 1
            move = move + 1
            
            # end of while loop - game ends

        # write paths to file
        with open('rhino_path.txt', 'a') as f:
            f.write(str(rhino_path)+'\n')
        ### second herd of rhino
        with open('rhino2_path.txt', 'a') as f:
            f.write(str(rhino2_path)+'\n')
        with open('ranger_path.txt', 'a') as f:
            f.write(str(ranger_path)+'\n')
        with open('poacher_path.txt', 'a') as f:
            f.write(str(poacher_path)+'\n')

        # update arrays
        steps_to_end.append(len(poacher_path))
        poach_per_game.append(poach)

        # end of for loop - month ends
    
    # write extra info files
    with open('steps_per_game.csv', 'a') as f:
        f.write(str(steps_to_end)[1:-1]+'\n')
    with open('poach_per_game.csv', 'a') as f:
        f.write(str(poach_per_game)[1:-1]+'\n')
    
    # record results
    results[itr] = (leave_after_poach, leave_before_poach, catch_after_poach, catch_before_poach, poach_events)

    # track time for iteration    
    print('iter %d --- %1.5f seconds ---' % (itr, time.time() - start_time))

    # end for loop - iterations end

# write results to file
    with open('results.csv','w') as f:
        for name in results.dtype.names:
            f.write(name + ',')
        f.write('\n')
        for row in results:
            for item in row:
                f.write(repr(item)+',')
            f.write('\n')

# figure 1: plot frequency for game endings
fig1 = plt.figure()
plt.subplot(2,2,1)
plt.ylabel('Poacher Left')
plt.xticks(())
plt.xlim(0, month_games)
plt.ylim(0, reps)
plt.hist(results['leave_before'])
plt.subplot(2,2,2)
plt.xticks(())
plt.yticks(())
plt.xlim(0, month_games)
plt.ylim(0, reps)
plt.hist(results['leave_after'])
plt.subplot(2,2,3)
plt.xlabel('Before Poaching')
plt.ylabel('Poacher Caught')
plt.xlim(0, month_games)
plt.ylim(0, reps)
plt.hist(results['catch_before'])
plt.subplot(2,2,4)
plt.xlabel('After Poaching')
plt.yticks(())
plt.xlim(0, month_games)
plt.ylim(0, reps)
plt.hist(results['catch_after'])
fig1.savefig('game_ends.png', dpi=200)

# figure 2: plot frequency of poaching events per month
fig2 = plt.figure()
plt.plot()
plt.xlim(0, month_games)
plt.ylim(0, reps)
plt.xlabel('Poaching events per month')
plt.ylabel('Frequency')
plt.hist(results['events'])
fig2.savefig('poach_events.png', dpi=200)

# figure 3: plot example paths for one game
fig3 = plt.figure()
plt.plot()
plt.xlim(0, grid_size)
plt.ylim(0, grid_size)
rhino_points = plt.Polygon(rhino_plot, closed=None, fill=None, edgecolor='g', label='rhino')
ranger_points = plt.Polygon(ranger_plot, closed=None, fill=None, edgecolor='b', label='ranger')
poacher_points = plt.Polygon(poacher_plot, closed=None, fill=None, edgecolor='r', label='poacher')
### second herd of rhino
rhino2_points = plt.Polygon(rhino2_plot, closed=None, fill=None, edgecolor='y', label='rhino2')
plt.gca().add_patch(rhino2_points)
plt.gca().add_patch(rhino_points)
plt.gca().add_patch(ranger_points)
plt.gca().add_patch(poacher_points)
plt.legend()
fig3.savefig('paths.png', dpi=200)

# table 1: Averages
ave = {'Before Poaching' : [np.mean(results['leave_before']), np.mean(results['catch_before'])], 
'After Poaching' : [np.mean(results['leave_after']),np.mean(results['catch_after'])]}
table_ave = pd.DataFrame(data=ave,index=['Poacher Left','Poacher Caught'])
table_ave
