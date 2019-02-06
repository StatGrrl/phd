# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 07:02:00 2016

@author: Lisa

Building on the null model
Smoother path by giving ranger and poacher a heading
Next cell chosen using bivariate normal distribution
That cell is selected if the euclidean distance from the heading is less 
than the distance between the heading and the current cell
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

# euclidean distance
def distance(vec1, vec2):
    return ((vec1[0]-vec2[0])**2 + (vec1[1]-vec2[1])**2)**(0.5)

def heading_norm_cell(heading, cell, grid_size=100, sigma2=0.1, rho=0):
    import numpy as np
    step = cell_per(cell, 1, grid_size, 'step')
    if heading in(step):
        select_step = heading
    else:
        dist_cell = distance(heading, cell)
        norm_coord = [grid_size, grid_size]
        dist_norm = distance(norm_coord, [0,0])
        while norm_coord not in(step) or dist_norm >= dist_cell:
            norm_bv = np.random.multivariate_normal(cell, [[sigma2, rho],[rho, sigma2]], 1)
            norm_coord = [round(abs(norm_bv[0][0])), round(abs(norm_bv[0][1]))]
            dist_norm = distance(heading, norm_coord)
        select_step = norm_coord
    return select_step
    
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
start_end = 'diff'

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
        
        # heading for ranger and poacher, start cell and end cell
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
        
        while end == 0:
            # rhino possible step cells
            ### allow for rhino to stay in current cell and the herd keeps moving after a poach
            ### if poach == 0:
            rhino_step = cell_per(rhino_cell, step_size, grid_size, 'step')
            rhino_step.append(rhino_cell)
            ### second herd of rhino
            rhino2_step = cell_per(rhino2_cell, step_size, grid_size, 'step')
            rhino2_step.append(rhino2_cell)
            
            # next cell for rhino, ranger and poacher
            ### rhino herd can keep moving after a poaching event
            ### if poach == 0:
            rhino_cell = rhino_step[random.randint(0,len(rhino_step)-1)]
            ### second herd of rhino
            rhino2_cell = rhino2_step[random.randint(0,len(rhino2_step)-1)]
            
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
            ranger_cell = heading_norm_cell(ranger_heading, ranger_cell, grid_size)
            poacher_cell = heading_norm_cell(poacher_heading, poacher_cell, grid_size)
        
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

# table 1: Averages
ave = {'Before Poaching' : [np.mean(results['leave_before']), np.mean(results['catch_before'])], 
'After Poaching' : [np.mean(results['leave_after']),np.mean(results['catch_after'])]}
table_ave = pd.DataFrame(data=ave,index=['Poacher Left','Poacher Caught'])
table_ave
