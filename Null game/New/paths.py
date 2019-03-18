# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 15:00:52 2019

@author: Lisa
"""
from functions import *

# declare constants
grid_size = 10
rhino_start = [3,3]
rhino2_start = [7,7]
ranger_start = [0,1]
poacher_start = [8,9]
prev_moves = 20
seed = 2562547

# null game
null_game = one_game('Null', grid_size=grid_size, rhino_cell=rhino_start, 
                rhino2_cell=rhino2_start, ranger_cell=ranger_start, 
                poacher_cell=poacher_start, seed=seed)
plot_path(null_game)

# more realistic null game
null_real_game = one_game('Realistic Null', grid_size=grid_size, rhino_cell=rhino_start, 
                     rhino2_cell=rhino2_start, ranger_cell=ranger_start, 
                     poacher_cell=poacher_start, seed=seed)
plot_path(null_real_game)

# avoid cyclic movement by staying out of previous cells - randomly choose number of moves 5-10
cyclic_rand_game = one_game('Avoid Cyclic', method='rand', grid_size=grid_size, 
                  rhino_cell=rhino_start, rhino2_cell=rhino2_start, 
                  ranger_cell=ranger_start, poacher_cell=poacher_start, seed=seed)
plot_path(cyclic_rand_game)

# avoid cyclic movement by staying out of previous cells - specify number of moves
cyclic_fixed_game = one_game('Avoid Cyclic', method='fixed', grid_size=grid_size, 
                        rhino_cell=rhino_start, rhino2_cell=rhino2_start, 
                        ranger_cell=ranger_start, poacher_cell=poacher_start, seed=seed, 
                        ranger_prev_moves=prev_moves, poacher_prev_moves=prev_moves)
plot_path(cyclic_fixed_game)

# Heading games, Max Same, Max Diff, Rand Same, Rand Diff, Norm Same, Norm Diff
head_rand_same_game = one_game('Heading', method='rand', start_end='same', grid_size=grid_size, 
                          rhino_cell=rhino_start, rhino2_cell=rhino2_start, 
                          ranger_cell=ranger_start, poacher_cell=poacher_start, seed=seed)
plot_path(head_rand_same_game)

head_rand_diff_game = one_game('Heading', method='rand', start_end='diff', grid_size=grid_size, 
                          rhino_cell=rhino_start, rhino2_cell=rhino2_start, 
                          ranger_cell=ranger_start, poacher_cell=poacher_start, seed=seed)
plot_path(head_rand_diff_game)

head_max_same_game = one_game('Heading', method='max', start_end='same', grid_size=grid_size, 
                         rhino_cell=rhino_start, rhino2_cell=rhino2_start, 
                         ranger_cell=ranger_start, poacher_cell=poacher_start, seed=seed)
plot_path(head_max_same_game)

head_max_diff_game = one_game('Heading', method='max', start_end='diff', grid_size=grid_size, 
                         rhino_cell=rhino_start, rhino2_cell=rhino2_start, 
                         ranger_cell=ranger_start, poacher_cell=poacher_start, seed=seed)
plot_path(head_max_diff_game)

head_norm_same_game = one_game('Heading', method='norm', start_end='same', grid_size=grid_size, 
                          rhino_cell=rhino_start, rhino2_cell=rhino2_start, 
                          ranger_cell=ranger_start, poacher_cell=poacher_start, seed=seed)
plot_path(head_norm_same_game)

head_norm_diff_game = one_game('Heading', method='norm', start_end='diff', grid_size=grid_size, 
                          rhino_cell=rhino_start, rhino2_cell=rhino2_start, 
                          ranger_cell=ranger_start, poacher_cell=poacher_start, seed=seed)
plot_path(head_norm_diff_game)

# Stackelberg pure strategy game

game_sol = stackel_pure(grid_size, rhino_start, rhino2_start, 'p1')
ranger_heading = game_sol['ranger_cell']
poacher_heading = game_sol['poacher_cell']        

spne_max_same_game = one_game('SPNE', method='max', start_end='same', grid_size=grid_size, 
                         rhino_cell=rhino_start, rhino2_cell=rhino2_start, 
                         ranger_cell=ranger_start, poacher_cell=poacher_start,
                         ranger_heading=ranger_heading, poacher_heading=poacher_heading,
                         seed=seed)
plot_path(spne_max_same_game)

spne_max_diff_game = one_game('SPNE', method='max', start_end='diff', grid_size=grid_size, 
                         rhino_cell=rhino_start, rhino2_cell=rhino2_start, 
                         ranger_cell=ranger_start, poacher_cell=poacher_start, 
                         ranger_heading=ranger_heading, poacher_heading=poacher_heading,
                         seed=seed)
plot_path(spne_max_diff_game)

spne_rand_same_game = one_game('SPNE', method='rand', start_end='same', grid_size=grid_size, 
                         rhino_cell=rhino_start, rhino2_cell=rhino2_start, 
                         ranger_cell=ranger_start, poacher_cell=poacher_start,
                         ranger_heading=ranger_heading, poacher_heading=poacher_heading,
                         seed=seed)
plot_path(spne_rand_same_game)

spne_rand_diff_game = one_game('SPNE', method='rand', start_end='diff', grid_size=grid_size, 
                         rhino_cell=rhino_start, rhino2_cell=rhino2_start, 
                         ranger_cell=ranger_start, poacher_cell=poacher_start, 
                         ranger_heading=ranger_heading, poacher_heading=poacher_heading,
                         seed=seed)
plot_path(spne_rand_diff_game)

spne_norm_same_game = one_game('SPNE', method='norm', start_end='same', grid_size=grid_size, 
                         rhino_cell=rhino_start, rhino2_cell=rhino2_start, 
                         ranger_cell=ranger_start, poacher_cell=poacher_start,
                         ranger_heading=ranger_heading, poacher_heading=poacher_heading,
                         seed=seed)
plot_path(spne_norm_same_game)

spne_norm_diff_game = one_game('SPNE', method='norm', start_end='diff', grid_size=grid_size, 
                         rhino_cell=rhino_start, rhino2_cell=rhino2_start, 
                         ranger_cell=ranger_start, poacher_cell=poacher_start, 
                         ranger_heading=ranger_heading, poacher_heading=poacher_heading,
                         seed=seed)
plot_path(spne_norm_diff_game)
