# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 15:51:48 2019

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

# Null games
null_sim = sim(game_type='Null', method=None, grid_size=grid_size,
           rhino_start=rhino_start, rhino2_start=rhino2_start, 
           ranger_start=ranger_start, poacher_start=poacher_start, seed=seed)

null_real_sim = sim(game_type='Realistic Null', method=None, grid_size=grid_size, 
                rhino_start=rhino_start, rhino2_start=rhino2_start, 
                ranger_start=ranger_start, poacher_start=poacher_start, seed=seed)

# Avoid cyclic games
cyclic_rand_sim = sim(game_type='Avoid Cyclic', method='rand', grid_size=grid_size, 
                  rhino_start=rhino_start, rhino2_start=rhino2_start, 
                  ranger_start=ranger_start, poacher_start=poacher_start, seed=seed)

cyclic_fixed_sim = sim(game_type='Avoid Cyclic', method='fixed', grid_size=grid_size, 
                   rhino_start=rhino_start, rhino2_start=rhino2_start, 
                   ranger_start=ranger_start, poacher_start=poacher_start, seed=seed,
                   ranger_prev_moves=prev_moves, poacher_prev_moves=prev_moves)

# Heading games
heading_max_same_sim = sim(game_type='Heading', method='max', start_end='same', 
                        grid_size=grid_size, rhino_start=rhino_start, 
                        rhino2_start=rhino2_start, ranger_start=ranger_start, 
                        poacher_start=poacher_start, seed=seed)

heading_max_diff_sim = sim(game_type='Heading', method='max', start_end='diff', 
                        grid_size=grid_size, rhino_start=rhino_start, 
                        rhino2_start=rhino2_start, ranger_start=ranger_start, 
                        poacher_start=poacher_start, seed=seed)

heading_rand_same_sim = sim(game_type='Heading', method='rand', start_end='same', 
                        grid_size=grid_size, rhino_start=rhino_start, 
                        rhino2_start=rhino2_start, ranger_start=ranger_start, 
                        poacher_start=poacher_start, seed=seed)

heading_rand_diff_sim = sim(game_type='Heading', method='rand', start_end='diff', 
                        grid_size=grid_size, rhino_start=rhino_start, 
                        rhino2_start=rhino2_start, ranger_start=ranger_start, 
                        poacher_start=poacher_start, seed=seed)

heading_norm_same_sim = sim(game_type='Heading', method='norm', start_end='same', 
                        grid_size=grid_size, rhino_start=rhino_start, 
                        rhino2_start=rhino2_start, ranger_start=ranger_start, 
                        poacher_start=poacher_start, seed=seed)

heading_norm_diff_sim = sim(game_type='Heading', method='norm', start_end='diff', 
                        grid_size=grid_size, rhino_start=rhino_start, 
                        rhino2_start=rhino2_start, ranger_start=ranger_start, 
                        poacher_start=poacher_start, seed=seed)

# Stackelberg games
spne_max_same_sim = sim(game_type='SPNE', method='max', start_end='same', 
                    grid_size=grid_size, rhino_start=rhino_start, 
                    rhino2_start=rhino2_start, ranger_start=ranger_start, 
                    poacher_start=poacher_start, seed=seed)

spne_max_diff_sim = sim(game_type='SPNE', method='max', start_end='diff', 
                    grid_size=grid_size, rhino_start=rhino_start, 
                    rhino2_start=rhino2_start, ranger_start=ranger_start, 
                    poacher_start=poacher_start, seed=seed)

spne_rand_same_sim = sim(game_type='SPNE', method='rand', start_end='same', 
                    grid_size=grid_size, rhino_start=rhino_start, 
                    rhino2_start=rhino2_start, ranger_start=ranger_start, 
                    poacher_start=poacher_start, seed=seed)

spne_rand_diff_sim = sim(game_type='SPNE', method='rand', start_end='diff', 
                    grid_size=grid_size, rhino_start=rhino_start, 
                    rhino2_start=rhino2_start, ranger_start=ranger_start, 
                    poacher_start=poacher_start, seed=seed)

spne_norm_same_sim = sim(game_type='SPNE', method='norm', start_end='same', 
                    grid_size=grid_size, rhino_start=rhino_start, 
                    rhino2_start=rhino2_start, ranger_start=ranger_start, 
                    poacher_start=poacher_start, seed=seed)

spne_norm_diff_sim = sim(game_type='SPNE', method='norm', start_end='diff', 
                    grid_size=grid_size, rhino_start=rhino_start, 
                    rhino2_start=rhino2_start, ranger_start=ranger_start, 
                    poacher_start=poacher_start, seed=seed)

# average per month for all games
ave = pd.DataFrame({'Null': null_sim['ave'], 'Realistic Null': null_real_sim['ave'],
              'Avoid Cyclic, random' : cyclic_rand_sim['ave'],
              'Avoid Cyclic, fixed' : cyclic_fixed_sim['ave'],
              'Heading, max, same' : heading_max_same_sim['ave'],
              'Heading, max, diff' : heading_max_diff_sim['ave'],
              'Heading, rand, same' : heading_rand_same_sim['ave'],
              'Heading, rand, diff' : heading_rand_diff_sim['ave'],
              'Heading, norm, same' : heading_norm_same_sim['ave'],
              'Heading, norm, diff' : heading_norm_diff_sim['ave'],
              'SPNE, max, same' : spne_max_same_sim['ave'],
              'SPNE, max, diff' : spne_max_diff_sim['ave'],
              'SPNE, rand, same' : spne_rand_same_sim['ave'],
              'SPNE, rand, diff' : spne_rand_diff_sim['ave'],
              'SPNE, norm, same' : spne_norm_same_sim['ave'],
              'SPNE, norm, diff' : spne_norm_diff_sim['ave']})

ave.to_csv('ave_per_month.csv')