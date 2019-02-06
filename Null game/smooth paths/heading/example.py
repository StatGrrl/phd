# -*- coding: utf-8 -*-
"""
Created on Fri Apr 01 02:52:01 2016

@author: Lisa
"""
import numpy as np
import matplotlib.pyplot as plt

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
   
def make_grid(vec, step, max_size, type_per):
    per = cell_per(vec, step, max_size, type_per)
    grid = np.zeros(shape=(max_size, max_size))
    grid[tuple(vec)] = 0.5
    for x in per:
        grid[tuple(x)] = 1
    return grid

start = [0,0]
heading = [2,3]
end = [4,1]
grid = make_grid(start, 1, 5, 'step')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xticks(np.arange(0.5,9.5,1), minor=True)
ax.set_yticks(np.arange(0.5,9.5,1), minor=True)
ax.grid(b=True, which='minor', lw=1, ls='solid')
plt.tick_params(which='major', length=0)
ax.imshow(grid, cmap='Blues', interpolation='nearest', origin='lower')
ax.text(start[0]-0.1, start[1], 'S', size=12, weight='bold')
ax.text(heading[0]-0.1, heading[1], 'H', size=12, weight='bold')
#probabilities
ax.text(0-0.25, 1, '0.32', size=12, weight='bold', color='white')
ax.text(1-0.25, 0, '0.28', size=12, weight='bold', color='white')
ax.text(1-0.25, 1, '0.40', size=12, weight='bold', color='white')
fig.savefig('ex_heading.png', dpi=200)

grid = np.zeros(shape=(5, 5))

fig2 = plt.figure()
ax = fig2.add_subplot(1,2,1)
ax.set_xticks(np.arange(0.5,9.5,1), minor=True)
ax.set_yticks(np.arange(0.5,9.5,1), minor=True)
ax.grid(b=True, which='minor', lw=1, ls='solid')
plt.tick_params(which='major', length=0)
ax.imshow(grid, cmap='Blues', interpolation='nearest', origin='lower')
ax.text(start[0]-0.1, start[1], 'S', size=12, weight='bold')
ax.text(heading[0]-0.1, heading[1], 'H', size=12, weight='bold')
ax.annotate('', xy=(start[0]+0.2, start[1]+0.2), xytext=(heading[0]-0.1, heading[1]-0.1), 
            arrowprops=dict(arrowstyle="<->", linewidth = 1.))
plt.xlabel('Same Start & End')
ax = fig2.add_subplot(1,2,2)
ax.set_xticks(np.arange(0.5,9.5,1), minor=True)
ax.set_yticks(np.arange(0.5,9.5,1), minor=True)
ax.grid(b=True, which='minor', lw=1, ls='solid')
plt.tick_params(which='major', length=0)
ax.imshow(grid, cmap='Blues', interpolation='nearest', origin='lower')
ax.text(start[0]-0.1, start[1], 'S', size=12, weight='bold')
ax.text(heading[0]-0.1, heading[1], 'H', size=12, weight='bold')
ax.annotate('', xy=(start[0]+0.2, start[1]+0.2), xytext=(heading[0]-0.1, heading[1]-0.1), 
            arrowprops=dict(arrowstyle="<-", linewidth = 1.))
ax.annotate('', xy=(heading[0]+0.1, heading[1]-0.1), xytext=(end[0]-0.1, end[1]-0.2), 
            arrowprops=dict(arrowstyle="<-", linewidth = 1.))
ax.annotate('', xy=(end[0]-0.2, end[1]-0.2), xytext=(start[0]+0.2, start[1]+0.2), 
            arrowprops=dict(arrowstyle="<-", linewidth = 1.))
ax.text(end[0]-0.1, end[1], 'E', size=12, weight='bold')
plt.xlabel('Different Start & End')
fig2.savefig('ex_start_end.png', dpi=200)


