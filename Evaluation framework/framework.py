"""
Module docstring:
TO DO:
* docstrings!
* change Park class to get geographical features from Google Earth Engine
* plot method for Park class
* print method for all classes
"""
import knp
import gametheory as gt
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import movingpandas as mpd
from shapely.geometry import Polygon, Point, LineString
import random
from datetime import datetime
from functools import lru_cache
import itertools
from scipy import stats


def coords_to_points(coords, crs):
    return gpd.GeoSeries([Point(x) for x in coords], index=[0, 1, 2, 3], crs=crs)


def coords_to_line(coords, crs, append_first=False):
    if append_first:
        coords.append(coords[0])
    return gpd.GeoSeries(LineString(coords), index=[0], crs=crs)


def coords_to_polygon(coords, crs):
    return gpd.GeoSeries(Polygon(coords), index=[0], crs=crs)


def bounds_to_coords(bounds):
    bounds_x_min = bounds[0]
    bounds_y_min = bounds[1]
    bounds_x_max = bounds[2]
    bounds_y_max = bounds[3]
    return [(bounds_x_min, bounds_y_min), (bounds_x_min, bounds_y_max),
            (bounds_x_max, bounds_y_max), (bounds_x_max, bounds_y_min)]


def bounds_to_points(bounds, crs):
    coords = bounds_to_coords(bounds)
    return coords_to_points(coords, crs)


def bounds_to_line(bounds, crs):
    coords = bounds_to_coords(bounds)
    return coords_to_line(coords, crs)


def bounds_to_polygon(bounds, crs):
    coords = bounds_to_coords(bounds)
    return coords_to_polygon(coords, crs)


def line_to_polygon(gdf):
    def line_lambda(line):
        return line.coords

    crs = gdf.crs
    coords = gdf.apply(lambda x: line_lambda(x['geometry']), axis=1)
    return coords_to_polygon(coords[0], crs)


def polygon_to_line(gdf):
    def poly_lambda(poly):
        return poly.exterior.coords

    crs = gdf.crs
    coords = gdf.apply(lambda x: poly_lambda(x['geometry']), axis=1)
    return coords_to_line(coords[0], crs)


class Park:
    """Game park with grid overlay, default crs is EPSG:4326 WGS 84 World """

    def __init__(self, name, camps=None, picnic_spots=None, gates=None, water=None, wildlife_sightings=None,
                 subarea=None, x_len=1000, y_len=1000):
        self.name = name  # name of game park
        # self.boundary = zagis.parks[zagis.parks['Name'] == self.name]
        # GIS properties - for now use knp, fix to get from gis using park name
        self.default_crs = knp.default_crs
        self.proj_crs = knp.proj_crs
        self.boundary = knp.boundary
        self.border_line = gpd.GeoDataFrame(geometry=polygon_to_line(self.boundary), crs=self.default_crs)
        self.trees = knp.trees  # polygons
        self.mountains = knp.mountains  # polygons
        self.roads = knp.roads_main  # lines
        self.rivers = knp.rivers_main  # lines
        self.dams = knp.dams  # points
        # Park places
        self.camps = camps  # camping spots
        self.picnic_spots = picnic_spots  # picnic spots and landmarks
        self.gates = gates  # entrance gates
        self.water = water  # fountains, water holes and drinking troughs
        self.features = {'border': self.border_line, 'trees': self.trees, 'mountains': self.mountains,
                         'roads': self.roads, 'rivers': self.rivers, 'dams': self.dams, 'water': self.water,
                         'camps': self.camps, 'picnic_spots': self.picnic_spots, 'gates': self.gates}
        self.wildlife_sightings = wildlife_sightings
        self.subarea = subarea
        self.x_len = x_len
        self.y_len = y_len

    @property
    @lru_cache()
    def grid(self):
        boundary_proj = self.boundary.to_crs(self.proj_crs)
        if self.subarea is None:
            bounds = boundary_proj.total_bounds
        else:
            bounds_gs = bounds_to_points(self.subarea, crs=self.default_crs)  # (lon, lat)
            bounds_gs = bounds_gs.to_crs(self.proj_crs)  # (y, x)
            bounds = np.array([bounds_gs[0].x, bounds_gs[0].y, bounds_gs[2].x, bounds_gs[2].y])
        x_min = bounds[2]
        y_min = bounds[3]
        x_max = bounds[0]
        y_max = bounds[1]
        grid_cells = pd.DataFrame(columns=['Area Number', 'Grid x', 'Grid y', 'Grid Index', 'Map Cells'])
        cell_num = 1
        df_index = 0
        y_index = 0
        y = y_min
        while y < y_max:
            x_index = 0
            x = x_min
            y_mid = y + self.y_len
            while x < x_max:
                x_mid = x + self.x_len
                grid_cells.loc[df_index, 'Area Number'] = cell_num
                grid_cells.loc[df_index, 'Grid x'] = x_index
                grid_cells.loc[df_index, 'Grid y'] = y_index
                grid_cells.loc[df_index, 'Grid Index'] = [x_index, y_index]
                grid_cells.loc[df_index, 'Map Cells'] = \
                    Polygon([(x, y), (x, y_mid), (x_mid, y_mid), (x_mid, y)])
                cell_num += 1
                df_index += 1
                x_index += 1
                x = x_mid
            y_index += 1
            y = y_mid
        grid_proj = gpd.GeoDataFrame(grid_cells, geometry='Map Cells', crs=self.proj_crs)
        grid = grid_proj.to_crs(self.default_crs)
        grid['Map Proj'] = grid_proj.geometry
        grid['Centroids Proj'] = grid_proj.centroid
        grid['Centroids'] = grid['Centroids Proj'].to_crs(self.default_crs)
        grid['Select Prob'] = 1
        grid['Utility'] = 0
        grid = grid.set_index('Area Number')
        # wildlife sightings
        if self.wildlife_sightings is not None:
            grid['Total Adults'] = 0
            grid['Total Calves'] = 0
            wildlife_sightings = self.wildlife_sightings
            grid_wildlife = gpd.sjoin(grid, wildlife_sightings, how='inner', predicate='intersects')
            grid_wildlife = grid_wildlife.set_geometry('Map Cells')
            #grid_wildlife = grid_wildlife.dissolve(by='Area Number', aggfunc=sum)
            grid['Total Adults'] = grid_wildlife['TOTAL']
            no_wildlife = grid['Total Adults'].isna()
            grid.loc[no_wildlife, 'Total Adults'] = 0
            grid['Total Calves'] = grid_wildlife['CALVES']
            no_calves = grid['Total Calves'].isna()
            grid.loc[no_calves, 'Total Calves'] = 0
        return grid

    @property
    @lru_cache()
    def bound_grid(self):
        grid = self.grid
        grid_col = len(grid.columns)
        park_grid = gpd.sjoin(grid, self.boundary, predicate='intersects', how='inner')
        park_grid = park_grid.iloc[:, 0:grid_col]
        return park_grid

    @property
    @lru_cache()
    def border_cells(self):
        grid = self.grid
        grid_col = len(grid.columns)
        border = gpd.sjoin(grid, self.border_line, predicate='intersects', how='inner')
        border = border.iloc[:, 0:grid_col]
        return border

    @property
    @lru_cache()
    def edge_cells(self):
        grid = self.grid
        x_max = max(grid['Grid x'])
        y_max = max(grid['Grid y'])
        edge_cells = []
        for i in range(x_max + 1):
            for j in range(y_max + 1):
                if i in (0, x_max) or j in (0, y_max):
                    edge_cells.append([i, j])
        edge_index = [x in list(edge_cells) for x in grid['Grid Index']]
        edge = grid[edge_index]
        return edge


class Player:

    def __init__(self, name, park, grid_type, move_type, path_type='continue', sampling='prob',  within=None, out=None,
                 dislike=None, like=None, start_cell=None, heading_cell=None, step_size=1, view_size=0,
                 stay_in_cell=True, geo_util_fctr=1, arrest_util=10, wild_save_fctr=1, wild_calve_fctr=1):
        self.name = name
        self.park = park  # Park class instance
        self.grid_type = grid_type  # 'full' / 'bound'
        self.move_type = move_type  # 'random' / 'strategic' / 'game'
        self.path_type = path_type  # 'stay' / 'end' / 'continue'
        self.sampling = sampling  # 'prob' / 'max'
        # within - polygon area to stay within, user input list of GeoDataFrames
        self.within = [] if within is None else within
        # GIS and park area options: 'roads', 'rivers', 'dams', 'mountains', 'trees', 'camps',
        #                            'picnic_spots', 'gates', 'water'
        # out - places can't enter
        # eg. out = ['rivers', 'dams', 'mountains', 'trees']
        self.out = [] if out is None else out
        # for strategic / game movement - places that deter: distance (m) to stay away
        # eg. dislike={'roads': 500, 'camps': 500, 'picnic_spots': 500, 'gates': 500}
        self.dislike = {} if dislike is None else dislike
        # for strategic / game movement - places that attract: distance (m) to stay near
        # eg. like={'water': 5000, 'dams': 8000}
        self.like = {} if like is None else like
        self.start_cell = start_cell
        self.heading_cell = heading_cell
        self.step_size = step_size
        self.view_size = view_size
        self.stay_in_cell = stay_in_cell
        self.geo_util_fctr = geo_util_fctr
        self.arrest_util = arrest_util
        self.wild_save_fctr = wild_save_fctr
        self.wild_calve_fctr = wild_calve_fctr

    @property
    @lru_cache()
    def allowed_cells(self):
        # grid_type: 'square', 'park'
        if self.grid_type == 'full':
            grid_cells = self.park.grid
        else:
            grid_cells = self.park.bound_grid
        within = self.within
        out = self.out
        dislike = self.dislike
        like = self.like
        grid_col = len(grid_cells.columns)
        if len(within) > 0:
            grid_bound = within.pop(0)
            if len(within) > 0:
                for gdf in within:
                    grid_bound = gpd.overlay(grid_bound, gdf, how='union')
            grid_bound = gpd.sjoin(grid_cells, grid_bound, predicate='intersects', how='inner')
            if grid_bound.shape[0] != 0:
                grid_cells = grid_bound.iloc[:, 0:grid_col]
        if len(out) != 0:
            for ftr in out:
                if self.park.features[ftr] is not None:
                    grid_exclude = gpd.sjoin(grid_cells, self.park.features[ftr], predicate='intersects', how='inner')
                    if grid_exclude.shape[0] != 0:
                        grid_cells = gpd.overlay(grid_cells, grid_exclude, how='difference')
                        grid_cells = grid_cells.iloc[:, 0:grid_col]
        grid_cells = grid_cells.to_crs(self.park.proj_crs)
        grid_centroids = grid_cells.centroid
        if self.move_type in ['strategic', 'game']:
            col_prob = grid_cells.columns.get_loc('Select Prob')
            col_util = grid_cells.columns.get_loc('Utility')
            if len(dislike) != 0:
                for ftr, ftr_val in dislike.items():
                    if self.park.features[ftr] is not None:
                        ftr_series = self.park.features[ftr].geometry.to_crs(self.park.proj_crs)
                        min_dist = [ftr_series.distance(x).min() for x in grid_centroids]
                        # if min_dist < ftr_val, decrease probability / utility as min_dist decreases
                        for i in range(len(min_dist)):
                            percent = (ftr_val - min_dist[i]) / ftr_val
                            # game player moves strategically too
                            ftr_prob = 0.5 * (1 - percent) if min_dist[i] < ftr_val else 0.5
                            grid_cells.iloc[i, col_prob] = grid_cells.iloc[i, col_prob] * ftr_prob
                            # if only one player has move_type=game, utility still needed for other player
                            ftr_util = 0 - percent * self.geo_util_fctr if min_dist[i] < ftr_val else 0
                            grid_cells.iloc[i, col_util] = grid_cells.iloc[i, col_util] + ftr_util
            if len(like) != 0:
                for ftr, ftr_val in like.items():
                    if self.park.features[ftr] is not None:
                        ftr_series = self.park.features[ftr].geometry.to_crs(self.park.proj_crs)
                        min_dist = [ftr_series.distance(x).min() for x in grid_centroids]
                        for i in range(len(min_dist)):
                            percent = (ftr_val - min_dist[i]) / ftr_val
                            ftr_prob = 0.5 * (1 + percent) if min_dist[i] < ftr_val else 0.5
                            grid_cells.iloc[i, col_prob] = grid_cells.iloc[i, col_prob] * ftr_prob
                            ftr_util = 0 + percent * self.geo_util_fctr if min_dist[i] < ftr_val else 0
                            grid_cells.iloc[i, col_util] = grid_cells.iloc[i, col_util] + ftr_util
        grid_cells = grid_cells.to_crs(self.park.default_crs)
        return grid_cells

    def start(self):
        if self.start_cell is not None:
            start_cell = self.start_cell
            if 'Time' in start_cell.index:
                start_cell = start_cell.drop('Time')
        else:
            start_cell = None
            len_cells = len(self.allowed_cells)
            perim = []
            while len(perim) < 3:
                rand_cell = random.randint(0, len_cells - 1)
                start_cell = self.allowed_cells.iloc[rand_cell, :]
                perim = self.perim_grid(start_cell, 'step')
        start_time = pd.Series(datetime.now(), index=['Time'], name=start_cell.name)
        start_cell = pd.concat([start_cell, start_time])
        return start_cell

    def heading(self):
        if self.heading_cell is not None:
            heading_cell = self.heading_cell
        else:
            heading_cell = None
            len_cells = len(self.allowed_cells)
            perim = []
            while len(perim) < 3:
                rand_cell = random.randint(0, len_cells - 1)
                heading_cell = self.allowed_cells.iloc[rand_cell, :]
                perim = self.perim_grid(heading_cell, 'step')
        return heading_cell

    def perim_grid(self, curr_cell, perim_type):
        def cell_perim(cell_index, step, view, p_type):
            # perim_type: 'step' / 'view'
            incr = step if p_type == 'step' else view
            curr_perim = []
            perim_cells = []
            if incr == 0:
                perim_cells = [cell_index]
            else:
                for i in range(1, incr + 1):
                    if i == 1:
                        curr_perim = [[cell_index[0] - i, cell_index[1] - i], [cell_index[0] - i, cell_index[1]],
                                      [cell_index[0] - i, cell_index[1] + i], [cell_index[0], cell_index[1] - i],
                                      [cell_index[0], cell_index[1] + i], [cell_index[0] + i, cell_index[1] - i],
                                      [cell_index[0] + i, cell_index[1]], [cell_index[0] + i, cell_index[1] + i]]
                    else:
                        next_perim = []
                        for j in range(len(curr_perim)):
                            next_perim.extend(cell_perim(curr_perim[j], 1, 1, p_type))
                        curr_perim = [list(x) for x in set(tuple(x) for x in next_perim)]
                    perim_cells.extend(curr_perim)
                    perim_cells = [list(x) for x in set(tuple(x) for x in perim_cells)]
                if p_type == 'step' and cell_index in perim_cells:
                    perim_cells.remove(cell_index)
            return perim_cells
        curr_cell_index = curr_cell['Grid Index']
        perim_index = cell_perim(curr_cell_index, self.step_size, self.view_size, perim_type)
        if self.stay_in_cell:
            if curr_cell_index not in perim_index:
                perim_index.append(curr_cell_index)
        perim_grid_index = [x in perim_index for x in self.allowed_cells['Grid Index']]
        perim = self.allowed_cells[perim_grid_index]
        return perim

    def movement(self, curr_cell, heading_cell):
        # sampling - 'prob' / 'max'
        sampling = self.sampling
        if heading_cell.name == curr_cell.name:
            select_step = curr_cell
        else:
            curr_perim = self.perim_grid(curr_cell, 'step')
            if heading_cell.name in curr_perim.index:
                select_step = heading_cell
            else:
                heading_centroid = heading_cell['Centroids Proj']
                perim_centroids = curr_perim.set_geometry('Centroids Proj')
                perim_centroids.crs = self.park.proj_crs
                cell_prob = list(curr_perim['Select Prob'])
                dist = [x.distance(heading_centroid) for x in perim_centroids.geometry]
                dist_inv = [1 / x for x in dist]
                dist_inv_tot = sum(dist_inv)
                if sampling == 'max':  # maximum probability
                    prob = [x / dist_inv_tot for x in dist_inv]
                    tot_prob = [prob[i] * cell_prob[i] for i in range(len(prob))]
                    max_prob_ind = tot_prob.index(max(tot_prob))
                    select_step = curr_perim.iloc[max_prob_ind, :]
                else:  # probability sampling
                    prob = [(list(curr_perim.index)[x],
                             dist_inv[x] / dist_inv_tot * cell_prob[x]) for x in range(len(curr_perim))]
                    prob_struct = np.array(prob, dtype=[('perim_ind', int), ('tot_prob', float)])
                    prob_sort = np.sort(prob_struct, order='tot_prob')
                    prob_cumm = [0]
                    for i in range(len(prob_sort)):
                        prob_cumm.append(prob_cumm[i] + prob_sort['tot_prob'][i])
                    prob_cumm.pop(0)
                    max_prob = prob_cumm[-1]
                    prec = -int(math.log10(abs(max_prob))) + 1
                    max_prob = round(max_prob + float('0.' + '0' * prec + '5'), prec)
                    select_prob = random.uniform(0, max_prob)
                    select_ind = np.array(prob_cumm) > select_prob
                    if not select_ind.any():
                        select = prob_sort['perim_ind'][-1]
                    else:
                        select = prob_sort['perim_ind'][select_ind][0]
                    select_step = curr_perim.loc[select, :]
            if 'Time' in select_step.index:
                select_step = select_step.drop('Time')
            cell_time = pd.Series(datetime.now(), index=['Time'], name=select_step.name)
            select_step = pd.concat([select_step, cell_time])
        return select_step


class Wildlife(Player):

    def __init__(self, name, park, grid_type, move_type, path_type='continue', sampling='prob',  within=None, out=None,
                 dislike=None, like=None, start_cell=None, heading_cell=None, step_size=1, view_size=0,
                 stay_in_cell=True, home_range=None, census=None):
        super().__init__(name, park, grid_type, move_type, path_type, sampling,  within, out, dislike, like, start_cell,
                         heading_cell, step_size, view_size, stay_in_cell)
        # within - 'home_range' / 'census' / GeoDataFrame
        self.within = [] if within is None else within
        self.home_range = home_range
        self.census = census
        if len(self.within) > 0:
            if 'home_range' in within:
                self.within.remove('home_range')
                if home_range is not None:
                    self.within.append(home_range)
            if 'census' in within:
                self.within.remove('census')
                if census is not None:
                    self.within.append(census)


class Ranger(Player):

    def __init__(self, name, park, grid_type, move_type, path_type='continue', sampling='prob',  within=None, out=None,
                 dislike=None, like=None, start_cell=None, heading_cell=None, step_size=1, view_size=0,
                 stay_in_cell=True, ranger_home=None, geo_util_fctr=1, arrest_util=10,
                 wild_save_fctr=1, wild_calve_fctr=1):
        super().__init__(name, park, grid_type, move_type, path_type, sampling, within, out, dislike, like, start_cell,
                         heading_cell, step_size, view_size, stay_in_cell, geo_util_fctr, arrest_util,
                         wild_save_fctr, wild_calve_fctr)
        self.ranger_home = ranger_home

    def start(self):
        if self.start_cell is not None:
            start_cell = self.start_cell
            if 'Time' in start_cell.index:
                start_cell = start_cell.drop('Time')
        else:
            start_cell = None
            ranger_home = self.ranger_home
            if ranger_home is None:
                cells = self.allowed_cells
            else:
                cols = len(self.allowed_cells.columns)
                cells = gpd.sjoin(self.allowed_cells, ranger_home, predicate='intersects', how='inner')
                cells = cells.iloc[:, 0:cols]
                if len(cells) < 2:
                    cells = self.allowed_cells
            len_cells = len(cells)
            perim = []
            while len(perim) < 3:
                rand_cell = random.randint(0, len_cells - 1)
                start_cell = cells.iloc[rand_cell, :]
                perim = self.perim_grid(start_cell, 'step')
        start_time = pd.Series(datetime.now(), index=['Time'], name=start_cell.name)
        start_cell = pd.concat([start_cell, start_time])
        return start_cell


class Poacher(Player):

    def __init__(self, name, park, grid_type, move_type, path_type='continue', sampling='prob',  within=None, out=None,
                 dislike=None, like=None, start_cell=None, heading_cell=None, step_size=1, view_size=0,
                 stay_in_cell=True, entry_type='border', geo_util_fctr=1, arrest_util=10,
                 wild_save_fctr=1, wild_calve_fctr=1):
        super().__init__(name, park, grid_type, move_type, path_type, sampling, within, out, dislike, like, start_cell,
                         heading_cell, step_size, view_size, stay_in_cell, geo_util_fctr, arrest_util,
                         wild_save_fctr, wild_calve_fctr)
        self.entry_type = entry_type

    def start(self):
        if self.start_cell is not None:
            start_cell = self.start_cell
            if 'Time' in start_cell.index:
                start_cell = start_cell.drop('Time')
        else:
            start_cell = None
            col_names = list(self.allowed_cells.columns)
            end_col = self.allowed_cells.columns.get_loc('Total Calves')
            col_geo = col_names.index('Centroids')
            col_names_new = list(self.allowed_cells.columns + '_left')
            col_names_new[col_geo] = 'Centroids'
            if end_col + 1 < len(col_names):
                col_names_new[(end_col + 1):len(col_names)] = col_names[(end_col + 1):len(col_names)]
            entry_cells = self.park.border_cells if self.entry_type == 'border' else self.park.edge_cells
            entry_cells = entry_cells.set_geometry('Centroids')
            cells = self.allowed_cells.sjoin(entry_cells, how='inner', predicate='intersects')
            cells = cells.loc[:, col_names_new]
            col_dict = {col_names_new[i]: col_names[i] for i in range(len(col_names))}
            cells = cells.rename(columns=col_dict).set_geometry('Map Cells')
            len_cells = len(cells)
            perim = []
            while len(perim) < 3:
                rand_cell = random.randint(0, len_cells - 1)
                start_cell = cells.iloc[rand_cell, :]
                perim = self.perim_grid(start_cell, 'step')
        start_time = pd.Series(datetime.now(), index=['Time'], name=start_cell.name)
        start_cell = pd.concat([start_cell, start_time])
        return start_cell


class Game:

    def __init__(self, name, wildlife, ranger, poacher, leader='ranger', game_type=None, same_start=False, seed=None,
                 end_moves=100, games_pm=30, months=1000, rtn_moves=False, rtn_traj=False, ssg_low=0., ssg_high=1, dobssM = 1000,
                 gamesol=None):
        self.name = name
        self.wildlife = wildlife
        self.ranger = ranger
        self.poacher = poacher
        self.leader = leader # set to 'ranger' for 'dobss', set to 'poacher' for 'ssg_follower'
        self.game_type = game_type # 'None' / 'spne' / 'ssg_follower' / 'dobss' / 'nash' / 'maximin'
        self.same_start = same_start
        self.seed = seed
        self.end_moves = end_moves
        self.games_pm = games_pm
        self.months = months
        self.rtn_moves = rtn_moves
        self.rtn_traj = rtn_traj
        self.ssg_low = ssg_low
        self.ssg_high = ssg_high
        self.dobssM = dobssM
        self.gamesol = gamesol

    @property
    @lru_cache()
    def strategies(self):
        if self.game_type is None:
            return None
        else:
            r_strat = self.ranger.allowed_cells.index
            p_strat = self.poacher.allowed_cells.index
        return {'ranger': r_strat, 'poacher': p_strat}

    @property
    @lru_cache()
    def payoffs(self):
        if self.game_type is None:
            return None
        else:
            wildlife_sighting = False if self.ranger.park.wildlife_sightings is None else True
            r_index = self.ranger.allowed_cells.index
            p_index = self.poacher.allowed_cells.index
            r_payoff = np.zeros(shape=(len(r_index), len(p_index)))
            p_payoff = np.zeros(shape=(len(r_index), len(p_index)))
            r_mtrx = 0
            for r_area in r_index:
                p_mtrx = 0
                for p_area in p_index:
                    r_utility = self.ranger.allowed_cells.loc[r_area, 'Utility']
                    p_utility = self.poacher.allowed_cells.loc[p_area, 'Utility']
                    if self.ranger.view_size == 0:
                        capture = True if p_area == r_area else False
                    else:
                        ranger_curr = self.ranger.allowed_cells.loc[r_area, :]
                        ranger_view_perim = self.ranger.perim_grid(ranger_curr, 'view')
                        perim_area = list(ranger_view_perim.index)
                        capture = True if p_area in perim_area else False
                    if capture:
                        r_utility = r_utility + self.ranger.arrest_util
                        p_utility = p_utility - self.poacher.arrest_util
                    if wildlife_sighting:
                        r_wildlife = self.ranger.allowed_cells.loc[r_area, 'Total Adults']
                        r_calves = self.ranger.allowed_cells.loc[r_area, 'Total Calves']
                        p_wildlife = self.poacher.allowed_cells.loc[p_area, 'Total Adults']
                        p_calves = self.poacher.allowed_cells.loc[p_area, 'Total Calves']
                        r_utility = r_utility + self.ranger.wild_save_fctr * r_wildlife
                        r_utility = r_utility + self.ranger.wild_calve_fctr * r_calves
                        if not capture:
                            r_utility = r_utility - self.ranger.wild_save_fctr * p_wildlife
                            r_utility = r_utility - self.ranger.wild_calve_fctr * p_calves
                            p_utility = p_utility + self.poacher.wild_save_fctr * p_wildlife
                            p_utility = p_utility + self.poacher.wild_calve_fctr * p_calves
                    r_payoff[r_mtrx, p_mtrx] = r_utility
                    p_payoff[r_mtrx, p_mtrx] = p_utility
                    p_mtrx += 1
                r_mtrx += 1
            return {'ranger': r_payoff, 'poacher': p_payoff}

    def mixed(self):
        ranger_select = self.ranger.allowed_cells['Select Prob']
        poacher_select = self.poacher.allowed_cells['Select Prob']
        ranger_total = sum(ranger_select)
        poacher_total = sum(poacher_select)
        ranger_mixed = np.zeros(len(ranger_select))
        poacher_mixed = np.zeros(len(poacher_select))
        for i in range(len(ranger_select)):
            ranger_mixed[i] = ranger_select.iloc[i] / ranger_total
        for j in range(len(poacher_select)):
            poacher_mixed[j] = poacher_select.iloc[j] / poacher_total
        return {'ranger': ranger_mixed, 'poacher': poacher_mixed}

    def game_solution(self): 
        if self.game_type is None:
            return None
        if self.game_type == 'spne':
            leader = 'p1' if self.leader == 'ranger' else 'p2'
            sol = gt.spne(self.strategies['ranger'], self.strategies['poacher'], self.payoffs['ranger'],
                          self.payoffs['poacher'], leader)
            return {'ranger_strat': sol[0][0], 'ranger_util': sol[1][0], 'poacher_strat': sol[0][1],
                    'poacher_util': sol[1][1]}
        if self.game_type == 'ssg_follower':
            sol = gt.ssg_follower(np.round(self.mixed()['poacher'],3), self.payoffs['ranger'], low=self.ssg_low, up=self.ssg_high)
            util = gt.exp_util(self.mixed()['poacher'], self.payoffs['poacher'], sol[0], self.payoffs['ranger'], self.leader)
            return {'ranger_strat': util[2], 'ranger_util': util[3], 'poacher_strat': util[0], 'poacher_util': util[1], 'feasible': sol[2], 'optimal': sol[3], 'util': sol[4]}
        if self.game_type == 'dobss':
            sol = gt.dobss(self.payoffs['ranger'], self.payoffs['poacher'], m_const=self.dobssM)
            return {'ranger_strat': sol[0], 'ranger_util': sol[1], 'poacher_strat': sol[2], 'poacher_util': sol[3]}
        if self.game_type == 'nash':
            sol = gt.nash(self.strategies['ranger'], self.strategies['poacher'], self.payoffs['ranger'], self.payoffs['poacher'])
            return {'ranger_strat': sol[0], 'ranger_util': sol[1], 'poacher_strat': sol[2], 'poacher_util': sol[3]}
        if self.game_type == 'maximin':
            if self.ranger.move_type == 'game':
                ranger_maximin = gt.maximin(self.payoffs['ranger'], 'ranger')
                ranger_mix = ranger_maximin[0]
            else:
                ranger_mix = self.mixed()['ranger']
            if self.poacher.move_type == 'game':
                poacher_maximin = gt.maximin(self.payoffs['poacher'], 'poacher')
                poacher_mix = poacher_maximin[0]
            else:
                poacher_mix = self.mixed()['poacher']
            util = gt.exp_util(ranger_mix, self.payoffs['ranger'], poacher_mix, self.payoffs['poacher'], 'ranger')
            return {'ranger_strat': ranger_mix, 'ranger_util': util[1], 'poacher_strat': poacher_mix, 'poacher_util': util[3]}
        
    def game_sol_instance(self, game_sol): 
        if self.game_type is None:
            return None
        else:
            # if game_solution returns mixed strategy, this method selects random strategy from the distribution
            if self.game_type == 'spne':
                r_strat = self.ranger.allowed_cells.loc[game_sol['ranger_strat'], :]
                r_util = game_sol['ranger_util']
                p_strat = self.poacher.allowed_cells.loc[game_sol['poacher_strat'], :]
                p_util = game_sol['poacher_util']
            else:
                # probability sampling
                r_prob = [(self.strategies['ranger'][x], game_sol['ranger_strat'][x])
                          for x in range(len(game_sol['ranger_strat']))]
                prob_struct = np.array(r_prob, dtype=[('strategy', int), ('probability', float)])
                prob_sort = np.sort(prob_struct, order='probability')
                prob_cumm = [0]
                for i in range(len(prob_sort)):
                    prob_cumm.append(prob_cumm[i] + prob_sort['probability'][i])
                prob_cumm.pop(0)
                select_prob = random.uniform(0, 1.0)
                select_ind = np.array(prob_cumm) > select_prob
                if not select_ind.any():
                    r_strat = prob_sort['strategy'][-1]
                else:
                    r_strat = prob_sort['strategy'][select_ind][0]
                r_vec = np.zeros((len(self.ranger.allowed_cells), 1))
                r_vec[r_strat][0] = 1
                r_strat = self.ranger.allowed_cells.loc[r_strat, ]
                if self.game_type == 'dobss':
                    p_vec = game_sol['poacher_strat']
                    p_strat = self.poacher.allowed_cells[p_vec.astype(bool)].squeeze()
                else:
                    p_prob = [(self.strategies['poacher'][x], game_sol['poacher_strat'][x])
                            for x in range(len(game_sol['poacher_strat']))]
                    prob_struct = np.array(p_prob, dtype=[('strategy', int), ('probability', float)])
                    prob_sort = np.sort(prob_struct, order='probability')
                    prob_cumm = [0]
                    for i in range(len(prob_sort)):
                        prob_cumm.append(prob_cumm[i] + prob_sort['probability'][i])
                    prob_cumm.pop(0)
                    select_prob = random.uniform(0, 1.0)
                    select_ind = np.array(prob_cumm) > select_prob
                    if not select_ind.any():
                        p_strat = prob_sort['strategy'][-1]
                    else:
                        p_strat = prob_sort['strategy'][select_ind][0]
                    p_vec = np.zeros((len(self.poacher.allowed_cells), 1))
                    p_vec[p_strat][0] = 1
                    p_strat = self.poacher.allowed_cells.loc[p_strat, ]

                # expected utility
                if self.leader == 'ranger':
                    util = gt.exp_util(r_vec, self.payoffs['ranger'], p_vec, self.payoffs['poacher'], 'ranger')
                    r_util = util[1]
                    p_util = util[3]
                else:
                    util = gt.exp_util(p_vec, self.payoffs['poacher'], r_vec, self.payoffs['ranger'], 'poacher')
                    r_util = util[3]
                    p_util = util[1]
            return {'ranger_strat': r_strat, 'ranger_util': r_util, 'poacher_strat': p_strat, 'poacher_util': p_util}


def sim_movement(game):
    move = 0
    columns = ['Wildlife Current', 'Wildlife Start', 'Wildlife Heading', 'Wildlife Toward', 'Wildlife Reach Start',
               'Wildlife Reach Heading', 'Ranger Current', 'Ranger Start', 'Ranger Heading', 'Ranger Toward',
               'Ranger Reach Start', 'Ranger Reach Heading', 'Poacher Current', 'Poacher Start',
               'Poacher Heading', 'Poacher Toward', 'Poacher Reach Start', 'Poacher Reach Heading',
               'Poach Cell', 'Poach Events', 'Capture Cell', 'Capture Events', 'Leave Cell', 'Leave Events',
               'Leave Before', 'Leave After', 'Capture Before', 'Capture After', 'Distance', 'Moves to Capture']
    ind = list(range(1, game.end_moves + 1))
    moves = pd.DataFrame(0.0, index=ind, columns=columns)
    moves.index.name = 'Move'

    path_columns = list(game.wildlife.park.grid.columns)
    path_columns.extend(['Time', 'Trajectory', 'PlayerID', 'Player', 'Move', 'Leave', 'Capture', 'Poach'])

    wildlife_start = game.wildlife.start()
    wildlife_heading = game.wildlife.heading()
    wildlife_traj = 1
    wildlife_toward = wildlife_heading
    wildlife_curr = wildlife_start
    if game.rtn_traj:
        wildlife_path = pd.DataFrame(columns=path_columns)
        wildlife_add = pd.Series([wildlife_traj, 1, 'Wildlife', move, 0, 0, 0],
                                 index=['Trajectory', 'PlayerID', 'Player', 'Move', 'Leave', 'Capture', 'Poach'],
                                 name=wildlife_curr.name)
        wildlife_add = pd.concat([wildlife_curr, wildlife_add])
        wildlife_path = pd.concat([wildlife_path, wildlife_add])
    else:
        wildlife_path = None

    ranger_start = game.ranger.start()
    ranger_heading = game.ranger.heading()
    ranger_traj = 1
    ranger_toward = ranger_heading
    ranger_curr = ranger_start
    if game.rtn_traj:
        ranger_path = pd.DataFrame(columns=path_columns)
        ranger_add = pd.Series([ranger_traj, 2, 'Ranger', move, 0, 0, 0],
                               index=['Trajectory', 'PlayerID', 'Player', 'Move', 'Leave', 'Capture', 'Poach'],
                               name=ranger_curr.name)
        ranger_add = pd.concat([ranger_curr, ranger_add])
        ranger_path = pd.concat([ranger_path, ranger_add])
    else:
        ranger_path = None

    poacher_start = game.poacher.start()
    poacher_heading = game.poacher.heading()
    poacher_traj = 1
    poacher_toward = poacher_heading
    poacher_curr = poacher_start
    poacher_path = pd.DataFrame(columns=path_columns)
    poacher_add = pd.Series([poacher_traj, 3, 'Poacher', move, 0, 0, 0],
                            index=['Trajectory', 'PlayerID', 'Player', 'Move', 'Leave', 'Capture', 'Poach'],
                            name=poacher_curr.name)
    poacher_add = pd.concat([poacher_curr, poacher_add])
    poacher_path = pd.concat([poacher_path, poacher_add])

    while move < game.end_moves:
        move += 1

        if game.wildlife.path_type != 'stay':
            wildlife_curr = game.wildlife.movement(wildlife_curr, wildlife_toward)
        if game.rtn_traj:
            wildlife_add = pd.Series([wildlife_traj, 1, 'Wildlife', move, 0, 0, 0],
                                     index=['Trajectory', 'PlayerID', 'Player', 'Move', 'Leave', 'Capture', 'Poach'],
                                     name=wildlife_curr.name)
            wildlife_add = pd.concat([wildlife_curr, wildlife_add])
            wildlife_path = pd.concat([wildlife_path, wildlife_add])
        moves.loc[move, 'Wildlife Current'] = wildlife_curr.name
        moves.loc[move, 'Wildlife Start'] = wildlife_start.name
        moves.loc[move, 'Wildlife Heading'] = wildlife_heading.name
        moves.loc[move, 'Wildlife Toward'] = wildlife_toward.name

        if wildlife_curr.name == wildlife_toward.name:
            if wildlife_curr.name == wildlife_heading.name:
                if game.wildlife.path_type == 'end':
                    game.wildlife.path_type = 'stay'
                moves.loc[move, 'Wildlife Reach Heading'] = 1
                wildlife_toward = wildlife_start
            if wildlife_curr.name == wildlife_start.name:
                moves.loc[move, 'Wildlife Reach Start'] = 1
                wildlife_toward = wildlife_heading
            if game.wildlife.path_type != 'stay':
                wildlife_traj += 1

        if game.ranger.path_type != 'stay':
            ranger_curr = game.ranger.movement(ranger_curr, ranger_toward)
        if game.rtn_traj:
            ranger_add = pd.Series([ranger_traj, 2, 'Ranger', move, 0, 0, 0],
                                   index=['Trajectory', 'PlayerID', 'Player', 'Move', 'Leave', 'Capture', 'Poach'],
                                   name=ranger_curr.name)
            ranger_add = pd.concat([ranger_curr, ranger_add])
            ranger_path = pd.concat([ranger_path, ranger_add])
        moves.loc[move, 'Ranger Current'] = ranger_curr.name
        moves.loc[move, 'Ranger Start'] = ranger_start.name
        moves.loc[move, 'Ranger Heading'] = ranger_heading.name
        moves.loc[move, 'Ranger Toward'] = ranger_toward.name

        ranger_view = game.ranger.perim_grid(ranger_curr, 'view')
        if ranger_curr.name == ranger_toward.name:
            if ranger_curr.name == ranger_heading.name:
                if game.ranger.path_type == 'end':
                    game.ranger.path_type = 'stay'
                moves.loc[move, 'Ranger Reach Heading'] = 1
                ranger_toward = ranger_start
            if ranger_curr.name == ranger_start.name:
                moves.loc[move, 'Ranger Reach Start'] = 1
                ranger_toward = ranger_heading
            if game.ranger.path_type != 'stay':
                ranger_traj += 1

        if game.poacher.path_type != 'stay':
            poacher_curr = game.poacher.movement(poacher_curr, poacher_toward)
        poacher_add = pd.Series([poacher_traj, 3, 'Poacher', move, 0, 0, 0],
                                index=['Trajectory', 'PlayerID', 'Player', 'Move', 'Leave', 'Capture', 'Poach'],
                                name=poacher_curr.name)
        poacher_add = pd.concat([poacher_curr, poacher_add])
        poacher_path = pd.concat([poacher_path, poacher_add])
        poacher_view = game.poacher.perim_grid(poacher_curr, 'view')
        moves.loc[move, 'Poacher Start'] = poacher_start.name
        moves.loc[move, 'Poacher Current'] = poacher_curr.name
        moves.loc[move, 'Poacher Heading'] = poacher_heading.name
        moves.loc[move, 'Poacher Toward'] = poacher_toward.name

        poacher_twd_heading = poacher_toward.name == poacher_heading.name
        poacher_twd_start = poacher_toward.name == poacher_start.name
        poacher_curr_heading = poacher_curr.name == poacher_heading.name
        poacher_curr_start = poacher_curr.name == poacher_start.name
        poacher_traj_ind = poacher_path['Trajectory'] == poacher_traj
        poacher_moves_traj = list(poacher_path.loc[poacher_traj_ind, 'Move'])
        if poacher_twd_start:
            poacher_traj_ind = poacher_path['Trajectory'] == (poacher_traj - 1)
            poacher_moves_traj.extend(list(poacher_path.loc[poacher_traj_ind, 'Move']))
        if 0 in poacher_moves_traj:
            poacher_moves_traj.remove(0)
        poaches_traj = sum(moves.loc[poacher_moves_traj, 'Poach Events'])

        if poacher_curr.name == poacher_toward.name:
            if poacher_curr.name == poacher_heading.name:
                if game.poacher.path_type == 'end':
                    game.poacher.path_type = 'stay'
                moves.loc[move, 'Poacher Reach Heading'] = 1
                poacher_toward = poacher_start
            if poacher_curr.name == poacher_start.name:
                moves.loc[move, 'Poacher Reach Start'] = 1
                poacher_toward = poacher_heading
            if game.poacher.path_type != 'stay':
                poacher_traj += 1

        moves.loc[move, 'Distance'] = ranger_curr['Centroids Proj'].distance(poacher_curr['Centroids Proj'])

        if wildlife_curr.name in poacher_view.index:
            if poaches_traj == 0 and not poacher_curr_heading:
                moves.loc[move, 'Poach Cell'] = wildlife_curr.name
                moves.loc[move, 'Poach Events'] = 1
                if game.rtn_traj:
                    wildlife_path.loc[wildlife_path['Move'] == move, 'Poach'] = 1
                    ranger_path.loc[ranger_path['Move'] == move, 'Poach'] = 1
                poacher_path.loc[poacher_path['Move'] == move, 'Poach'] = 1
                if poacher_twd_heading:
                    poacher_toward = poacher_start
                    poacher_traj += 1
                if game.poacher.move_type == 'strategic':
                    poacher_heading = poacher_curr

        if poacher_twd_start and poacher_curr_start:
            moves.loc[move, 'Leave Cell'] = poacher_curr.name
            moves.loc[move, 'Leave Events'] = 1
            if game.rtn_traj:
                wildlife_path.loc[wildlife_path['Move'] == move, 'Leave'] = 1
                ranger_path.loc[ranger_path['Move'] == move, 'Leave'] = 1
            poacher_path.loc[poacher_path['Move'] == move, 'Leave'] = 1
            if poaches_traj > 0:
                moves.loc[move, 'Leave After'] = 1
            else:
                moves.loc[move, 'Leave Before'] = 1

        if poacher_curr.name in ranger_view.index:
            moves.loc[move, 'Moves to Capture'] = move
            moves.loc[move, 'Capture Cell'] = poacher_curr.name
            moves.loc[move, 'Capture Events'] = 1
            if game.rtn_traj:
                wildlife_path.loc[wildlife_path['Move'] == move, 'Capture'] = 1
                ranger_path.loc[ranger_path['Move'] == move, 'Capture'] = 1
            poacher_path.loc[poacher_path['Move'] == move, 'Capture'] = 1
            if poaches_traj > 0:
                moves.loc[move, 'Capture After'] = 1
            else:
                moves.loc[move, 'Capture Before'] = 1
            if game.poacher.move_type == 'strategic':
                poacher_start = game.poacher.start()
            poacher_curr = poacher_start
            poacher_toward = poacher_heading
            poacher_traj += 1
        else:
            moves.loc[move, 'Moves to Capture'] = 0

    sum_cols = ['Wildlife Reach Start', 'Wildlife Reach Heading', 'Ranger Reach Start', 'Ranger Reach Heading',
                'Poacher Reach Start', 'Poacher Reach Heading', 'Poach Events', 'Capture Events', 'Leave Events',
                'Leave Before', 'Leave After', 'Capture Before', 'Capture After', 'Distance', 'Moves to Capture']
    res_sum = (moves.loc[:, sum_cols]).sum(axis=0)
    res_sum['Moves to Capture'] = max(moves['Moves to Capture'])
    res_sum['Distance'] = min(moves['Distance'])  # closest ranger got to poacher
    results = {'totals': res_sum}
    if game.rtn_moves:
        results['moves'] = moves
    if game.rtn_traj:
        def_crs = game.wildlife.park.grid.crs
        paths = pd.concat([wildlife_path, ranger_path, poacher_path])
        paths = paths.rename(columns={'Time': 't', 'Centroids': 'geometry'})
        paths = gpd.GeoDataFrame(paths, crs=def_crs, geometry='geometry')
        traj_coll = mpd.TrajectoryCollection(paths.set_index('t'), 'PlayerID')
        results['trajectories'] = traj_coll
    return results


def sim_games(game):
    if game.seed is not None:
        random.seed(game.seed)
    ind_dat = [range(1, (game.months + 1)), range(1, (game.games_pm + 1))]
    ind = pd.MultiIndex.from_product(ind_dat, names=['Month', 'Game'])
    columns = ['Wildlife Reach Start', 'Wildlife Reach Heading', 'Ranger Reach Start', 'Ranger Reach Heading',
               'Poacher Reach Start', 'Poacher Reach Heading', 'Poach Events', 'Capture Events', 'Leave Events',
               'Capture Before', 'Capture After', 'Leave Before', 'Leave After', 'Distance', 'Moves to Capture']
    games = pd.DataFrame(0.0, index=ind, columns=columns)
    moves = []
    traj = []
    ranger_util = []
    poacher_util = []
    ranger_optimal = []
    poacher_optimal = []

    # calculate the game solution for the month
    game_month = None
    if game.game_type is not None:
        if game.gamesol is None:
            game_month = game.game_solution()
        else:
            game_month = game.gamesol

    # month iterations
    for m in range(1, (game.months + 1)):
        # game iterations
        for g in range(1, (game.games_pm + 1)):
            # get game solution instance
            if game.game_type is not None:
                game_instance = game.game_sol_instance(game_month)
                if game.ranger.move_type == 'game':
                    game.poacher.heading_cell = game_instance['poacher_strat']
                    poacher_optimal.append(game_instance['poacher_strat'].name)
                    poacher_util.append(game_instance['poacher_util'])
                    game.ranger.heading_cell = game_instance['ranger_strat']
                    ranger_optimal.append(game_instance['ranger_strat'].name)
                    ranger_util.append(game_instance['ranger_util'])
                    if game.ranger.path_type == 'stay':
                        game.ranger.start_cell = game.ranger.heading_cell
                if game.same_start:
                    game.wildlife.start_cell = game.wildlife.start()
                    game.ranger.start_cell = game.ranger.start()
                    game.poacher.start_cell = game.poacher.start()
            else:
                game_instance = None
                poacher_util.append(0)
                ranger_util.append(0)
                ranger_optimal.append(0)
                poacher_optimal.append(0)

            # run single game
            single_game = sim_movement(game)
            games.loc[m, g] = single_game['totals']
            if game.rtn_moves:
                moves.append(single_game['moves'])
            if game.rtn_traj:
                traj.append(single_game['trajectories'])

    games.applymap("{0:.3f}".format)
    games['Distance'] = [x/1000 for x in games['Distance']]
    games['Games No Capture'] = [1 if x == 0 else 0 for x in games['Capture Events']]
    games['Ranger Utility'] = ranger_util
    games['Poacher Utility'] = poacher_util
    games['Utility Diff'] = games['Ranger Utility'] - games['Poacher Utility']
    games['Ranger Win'] = [1 if x > y else 0 for x, y in zip(ranger_util, poacher_util)]
    games['Ranger Optimal'] = ranger_optimal
    games['Poacher Optimal'] = poacher_optimal
    sum_per_month = games.groupby(by='Month').sum()
    metrics = pd.DataFrame({'Poach Freq per Day': sum_per_month['Poach Events'] / 30,
                            'Capture Freq per Day': sum_per_month['Capture Events'] / 30,
                            'Ave Ranger Utility': sum_per_month['Ranger Utility'] / game.games_pm,
                            'Ave Poacher Utility': sum_per_month['Poacher Utility'] / game.games_pm,
                            'Ave Utility Diff': sum_per_month['Utility Diff'] / game.games_pm,
                            'Ave Ranger Win': sum_per_month['Ranger Win'] / game.games_pm})
    metrics['Ave Moves for Captures'] = [x/y if y >0  else np.nan for x, y in zip(sum_per_month['Moves to Capture'], sum_per_month['Capture Events'])]
    metrics['Ave Distance for Non Captures'] = [x/y if y > 0 else np.nan for x, y in zip(sum_per_month['Distance'],sum_per_month['Games No Capture'])]

    descrip = pd.DataFrame(index=metrics.columns, columns=['Min', 'Max', 'Var', 'Mean', 'SE Mean', 'Median', 'SE Med'])
    descrip['Min'] = np.min(metrics, axis=0)
    descrip['Max'] = np.max(metrics, axis=0)
    descrip['Var'] = np.var(metrics, axis=0)
    weights = pd.DataFrame({'Poach Freq per Day': 1, 'Capture Freq per Day': 1,
                            'Ave Ranger Utility': 1, 'Ave Poacher Utility': 1, 
                            'Ave Utility Diff': 1, 'Ave Ranger Win': 1,
                            'Ave Moves for Captures': sum_per_month['Capture Events'],
                            'Ave Distance for Non Captures': sum_per_month['Games No Capture']})
    descrip['Mean'] = np.ma.average(np.ma.masked_array(metrics, np.isnan(metrics)), axis=0, weights=weights)
    descrip['Median'] = np.nanmedian(metrics, axis=0)
    col_nan = metrics.isna().any()
    for i in range(len(descrip)):
        x = metrics.iloc[:, [i]].dropna() if col_nan[i] else metrics.iloc[:, [i]]
        x = x.values.flatten()
        descrip.loc[descrip.index[i], 'SE Mean'] = boot_se(x, 1000, 'mean')
        descrip.loc[descrip.index[i], 'SE Med'] = boot_se(x, 1000, 'median')
    results = {'name': game.name,
               'games': games,
               'months': sum_per_month,
               'metrics': metrics,
               'descrip': descrip}
    if game.rtn_moves:  
        results['moves'] = moves
    if game.rtn_traj:
        results['trajectories'] = traj
    return results


# Bootstrap Standard Errors
def boot_se(x, b, stat='mean'):
    n = len(x)
    boot_sample = np.random.choice(x, (n, b))
    boot_stat = np.median(boot_sample, axis=0) if stat == 'median' else np.mean(boot_sample, axis=0)
    return np.std(boot_stat, ddof=1)


# Evaluation incl Mood's median test
def sim_eval(sim_list):
    # Combine results into dataframes
    sim_names = []
    sim_games = []
    sim_months = []
    sim_metrics = []
    sim_descrip = []
    for i in range(len(sim_list)):
        sim_names.append(sim_list[i]['name'])
        sim_list[i]['games']['Simulation'] = sim_list[i]['name']
        sim_games.append(sim_list[i]['games'])
        sim_list[i]['months']['Simulation'] = sim_list[i]['name']
        sim_months.append(sim_list[i]['months'])
        sim_list[i]['metrics']['Simulation'] = sim_list[i]['name']
        sim_metrics.append(sim_list[i]['metrics'])
        sim_list[i]['descrip']['Simulation'] = sim_list[i]['name']
        sim_descrip.append(sim_list[i]['descrip'])
    sim_games = pd.concat(sim_games)    
    sim_months = pd.concat(sim_months)
    sim_metrics = pd.concat(sim_metrics)
    sim_descrip = pd.concat(sim_descrip)

    # Mood's Median Test
    comb = list(itertools.combinations(sim_names, 2))
    med_pval = pd.DataFrame(comb)
    metrics = list(sim_metrics.columns)
    del metrics[-1]
    for i in range(len(med_pval)):
        for j in metrics:
            x = sim_metrics.loc[sim_metrics['Simulation'] == med_pval.loc[i, 0], j].dropna()
            y = sim_metrics.loc[sim_metrics['Simulation'] == med_pval.loc[i, 1], j].dropna()
            tot = pd.concat([x, y])
            grand_med = np.median(tot)
            if all([a <= grand_med for a in x]) :
                med_pval.loc[i, j] = np.nan
            else:
                med_pval.loc[i, j] = stats.median_test(x, y)[1]
    return {'games': sim_games, 'months': sim_months, 'metrics': sim_metrics, 'descrip': sim_descrip, 'med_pval': med_pval}
