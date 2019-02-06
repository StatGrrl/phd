# -*- coding: utf-8 -*-
"""
Created on Mon May 23 15:41:30 2016

@author: Lisa
"""

import numpy as np

# function to calculate surrounding cells
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

cell = [2,2]
per = set(tuple(x) for x in cell_per(cell, 1, 4, 'step'))
per_list = [list(x) for x in per]
per_len = len(per_list)

mean = cell   # mean of bivariate normals
rho = 0   # covariance of bivariate normals
size = 1000 # size of bivariate normal samples

names = [str(x) for x in per_list]
names.insert(0, 'Cell ' + str(cell))
names.insert(0, 'Var')
names.append('Other')
formats = ['i1'] * (len(per_list) + 2)
formats.insert(0, 'a8')
sigma2_vary = np.arange(0.01, 0.26, 0.01)   # variance of bivariate normals
sigma2_test = np.zeros(len(sigma2_vary), dtype={'names':names,'formats':formats})
i = 0
for sigma2 in sigma2_vary:
    normals = np.random.multivariate_normal(mean, [[sigma2, rho],[rho, sigma2]], size)
    normals_rounded_set = set(tuple(x) for x in [[round(abs(x[0])), round(abs(x[1]))] for x in normals])
    sigma2_test[i][0] = str(sigma2)
    if tuple(cell) in normals_rounded_set:
        sigma2_test[i][1] = 1
    for j in range(per_len):
        if tuple(per_list[j]) in normals_rounded_set:
            sigma2_test[i][j+2] = 1
    sigma2_test[i][per_len+2] = len(normals_rounded_set - per - {tuple(cell)})
    i += 1

# write results to file
with open('variance_test.csv','w') as f:
    for name in sigma2_test.dtype.names:
        f.write("'" + name + "',")
    f.write('\n')
    for row in sigma2_test:
        for item in row:
            f.write(repr(item)+',')
        f.write('\n')

grid_size = 100
cell_vary = []
for i in range(grid_size):
    for j in range(grid_size):
        cell_vary.append([i,j])
sigma2 = 0.1
sigma2_test2 = []
for cell in cell_vary:
    per = set(tuple(x) for x in cell_per(cell, 1, grid_size, 'step'))
    normals = np.random.multivariate_normal(cell, [[sigma2, rho],[rho, sigma2]], size)
    normals_rounded_set = set(tuple(x) for x in [[round(abs(x[0])), round(abs(x[1]))] for x in normals])
    sigma2_test2.append([str(cell), len((per | {tuple(cell)}) - normals_rounded_set), 
                         len(normals_rounded_set - (per | {tuple(cell)}))])

# write results to file
with open('variance_test2.csv','w') as f:
    f.write('Cell, Per Cells not in Normals, Normals not in Per Cells\n')
    for row in sigma2_test2:
        for item in row:
            f.write(repr(item)+',')
        f.write('\n')

