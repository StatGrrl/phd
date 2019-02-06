# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 10:44:00 2016

@author: Lisa

A null game is constructed to serve as a baseline for comparison purposes. The game
is intended to emulate a completely random scenario under the most basic framework. 
Any conditions that are imposed are thought to satisfy realistic situations. The
following assumptions are made:
• There is one rhino, one poacher and one ranger
• A square grid is used to imitate a map of the area
• The size of each cell within the grid represents an area which one ranger patrols
• The rhino starts in a random cell on the grid
• The poacher and the ranger must enter the area in a border cell (figure 6)
• The rhino, poacher and ranger move randomly into an adjacent cell, they cannot stay in the 
  same cell and they do not learn (figure 4)
• If the poacher is in the same cell as the rhino then the rhino is poached and it
  stops moving
• If the poacher is in the same cell as the ranger then the poacher is captured and
  the game ends
• If the poacher enters a border cell then he leaves the area and the game ends

The simulation is performed in a way that depicts the average behaviour for one
month. A grid size of 100 x 100 cells was used. One game follows the routes of the
rhino, poacher and ranger until the poacher either leaves or is captured. If a rhino
is poached, the event is recorded. We ssume that 10 such games occur within one
month. For each month the following measures are recorded:
number of poaching events
• number of times the poacher leaves before a poaching event
• number of times the poacher leaves after a poaching event
• number of times the poacher is captured before a poaching event
• number of times the poacher is captured after a poaching event

The monthly cycle is simulated for 1000 repititions. The averages and frequencies
of each event are depicted in table 1 and figures 1 and 2. An example of the
routes followed in one game is shown in figure 3

For this game, the detection area of the ranger includes only the cell in which he is 
(figure 5 with view size = 0)
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
view_size = 0
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
np.savetxt('edge_cells.csv', edge_cells, delimiter=",", fmt='%d')

# remove files which are appended
files = ['poach_per_game.csv','steps_per_game.csv','rhino_path.txt','ranger_path.txt','poacher_path.txt']
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
        ranger_cell = edge_cells[random.randint(0,len(edge_cells)-1)]
        poacher_cell = edge_cells[random.randint(0,len(edge_cells)-1)]
        
        # record path of rhino, ranger and poacher
        rhino_path = [rhino_cell]
        ranger_path = [ranger_cell]
        poacher_path = [poacher_cell]
        
        # one game
        poach = 0
        end = 0
        
        while end == 0:
            # rhino, ranger and poacher possible step cells
            if poach == 0:
                rhino_step = cell_per(rhino_cell, step_size, grid_size, 'step')
            ranger_step = cell_per(ranger_cell, step_size, grid_size, 'step')
            poacher_step = cell_per(poacher_cell, step_size, grid_size, 'step')
            
            # next cell for rhino, ranger and poacher
            if poach == 0:
                rhino_cell = rhino_step[random.randint(0,len(rhino_step)-1)]
            ranger_cell = ranger_step[random.randint(0,len(ranger_step)-1)]
            poacher_cell = poacher_step[random.randint(0,len(poacher_step)-1)]
        
            # record path of rhino, ranger and poacher
            if poach == 0:
                rhino_path.append(rhino_cell)
            ranger_path.append(ranger_cell)
            poacher_path.append(poacher_cell)
        
            # ranger viewing cells
            ranger_view = cell_per(ranger_cell, view_size, grid_size, 'view')
        
            # check for poaching event
            if rhino_cell == poacher_cell and poach == 0:
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
                    ranger_plot = ranger_path
                    poacher_plot = poacher_path
                end = 1
            
            # check for leaving event
            if poacher_cell in edge_cells:
                if poach > 0:
                    leave_after_poach = leave_after_poach + 1
                else:
                    leave_before_poach = leave_before_poach + 1
                end = 1
            
            # end of while loop - game ends

        # write paths to file
        with open('rhino_path.txt', 'a') as f:
            f.write(str(rhino_path)+'\n')
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
plt.gca().add_patch(rhino_points)
plt.gca().add_patch(ranger_points)
plt.gca().add_patch(poacher_points)
plt.legend()
fig3.savefig('paths.png', dpi=200)

# plot example step and view perimeters
def make_grid(vec, step, max_size, type_per):
    per = cell_per(vec, step, max_size, type_per)
    grid = np.zeros(shape=(max_size, max_size))
    grid[tuple(vec)] = 0.5
    for x in per:
        grid[tuple(x)] = 1
    return grid

grid1 = make_grid([3,3], 1, 10, 'step')
grid2 = make_grid([0,3], 1, 10, 'step')
grid3 = make_grid([0,0], 1, 10, 'step')
grid4 = make_grid([4,4], 0, 10, 'view')
grid5 = make_grid([4,4], 1, 10, 'view')
grid6 = make_grid([4,4], 2, 10, 'view')
grid7 = make_grid([4,4], 3, 10, 'view')

# figure 4: step perimeters
fig4 = plt.figure()
ax = fig4.add_subplot(1,3,1)
ax.set_xticks(np.arange(0.5,9.5,1), minor=True)
ax.set_yticks(np.arange(0.5,9.5,1), minor=True)
ax.grid(b=True, which='minor', lw=1, ls='solid')
plt.xlabel('inside cell')
plt.tick_params(which='major', length=0)
ax.imshow(grid1, cmap='Blues', interpolation='nearest', origin='lower')
ax = fig4.add_subplot(1,3,2)
ax.set_xticks(np.arange(0.5,9.5,1), minor=True)
ax.set_yticks(np.arange(0.5,9.5,1), minor=True)
ax.grid(b=True, which='minor', lw=1, ls='solid')
plt.xlabel('edge cell')
plt.tick_params(which='major', length=0)
ax.imshow(grid2, cmap='Blues', interpolation='nearest', origin='lower')
ax = fig4.add_subplot(1,3,3)
ax.set_xticks(np.arange(0.5,9.5,1), minor=True)
ax.set_yticks(np.arange(0.5,9.5,1), minor=True)
ax.grid(b=True, which='minor', lw=1, ls='solid')
plt.xlabel('corner cell')
ax.imshow(grid3, cmap='Blues', interpolation='nearest', origin='lower')
plt.tick_params(which='major', length=0)
fig4.savefig('step_perimeter.png', dpi=200)

# figure 5: view perimeters
fig5 = plt.figure()
ax = fig5.add_subplot(1,4,1)
ax.set_xticks(np.arange(0.5,9.5,1), minor=True)
ax.set_yticks(np.arange(0.5,9.5,1), minor=True)
ax.grid(b=True, which='minor', lw=1, ls='solid')
plt.xlabel('view size = 0')
plt.tick_params(which='major', length=0)
plt.annotate('X', xy=(3.5, 3.5))
ax.imshow(grid4, cmap='Blues', interpolation='nearest', origin='lower')
ax = fig5.add_subplot(1,4,2)
ax.set_xticks(np.arange(0.5,9.5,1), minor=True)
ax.set_yticks(np.arange(0.5,9.5,1), minor=True)
ax.grid(b=True, which='minor', lw=1, ls='solid')
plt.xlabel('view size = 1')
plt.tick_params(which='major', length=0)
plt.annotate('X', xy=(3.5, 3.5))
ax.imshow(grid5, cmap='Blues', interpolation='nearest', origin='lower')
ax = fig5.add_subplot(1,4,3)
ax.set_xticks(np.arange(0.5,9.5,1), minor=True)
ax.set_yticks(np.arange(0.5,9.5,1), minor=True)
ax.grid(b=True, which='minor', lw=1, ls='solid')
plt.xlabel('view size = 2')
plt.tick_params(which='major', length=0)
plt.annotate('X', xy=(3.5, 3.5))
ax.imshow(grid6, cmap='Blues', interpolation='nearest', origin='lower')
ax = fig5.add_subplot(1,4,4)
ax.set_xticks(np.arange(0.5,9.5,1), minor=True)
ax.set_yticks(np.arange(0.5,9.5,1), minor=True)
ax.grid(b=True, which='minor', lw=1, ls='solid')
plt.xlabel('view size = 3')
plt.tick_params(which='major', length=0)
plt.annotate('X', xy=(3.5, 3.5))
ax.imshow(grid7, cmap='Blues', interpolation='nearest', origin='lower')
fig5.savefig('view_perimeter.png', dpi=200)

# figure 6: plot example edge cells
grid8 = np.zeros(shape=(10, 10))
edge = []
for i in range(10):
    for j in range(10):
        if i in(0, 9) or j in(0, 9):
            edge.append([i,j])
for x in edge:
    grid8[tuple(x)] = 1

fig6 = plt.figure()
ax = fig6.add_subplot(1,1,1)
ax.set_xticks(np.arange(0.5,9.5,1), minor=True)
ax.set_yticks(np.arange(0.5,9.5,1), minor=True)
ax.grid(b=True, which='minor', lw=1, ls='solid')
plt.xlabel('border cells')
plt.tick_params(which='major', length=0)
ax.imshow(grid8, cmap='Blues', interpolation='nearest', origin='lower')
fig6.savefig('edge_cells.png', dpi=200)

# table 1: Averages
ave = {'Before Poaching' : [np.mean(results['leave_before']), np.mean(results['catch_before'])], 
'After Poaching' : [np.mean(results['leave_after']),np.mean(results['catch_after'])]}
table_ave = pd.DataFrame(data=ave,index=['Poacher Left','Poacher Caught'])
table_ave
