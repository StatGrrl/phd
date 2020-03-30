"""
Module docstring:
TO DO:
* docstrings!
* change Park class to get geographical features from Google Earth Engine
* plot method for Park class
* print method for all classes
"""
import knp
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point, LineString
import random
from datetime import datetime
import gametheory as gt
import movingpandas as mpd
from functools import lru_cache


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
        self.border_line = gpd.GeoDataFrame(geometry=polygon_to_line(self.boundary),
                                            crs=self.default_crs)
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
        grid['Centroids'] = grid.centroid
        grid['Centroids Proj'] = grid_proj.centroid
        grid['Select Prob'] = 1
        grid['Utility'] = 0
        # wildlife sightings
        if self.wildlife_sightings is not None:
            grid['Total Adults'] = 0
            grid['Total Calves'] = 0
            wildlife_sightings = self.wildlife_sightings
            grid_wildlife = gpd.sjoin(grid, wildlife_sightings, how='inner', op='intersects')
            grid_wildlife = grid_wildlife.dissolve(by='Area Number', aggfunc=sum)
            grid = grid.set_index('Area Number')
            grid['Total Adults'] = grid_wildlife['TOTAL']
            no_wildlife = grid['Total Adults'].isna()
            grid.loc[no_wildlife, 'Total Adults'] = 0
            grid['Total Calves'] = grid_wildlife['CALVES']
            no_calves = grid['Total Calves'].isna()
            grid.loc[no_calves, 'Total Calves'] = 0
        else:
            grid = grid.set_index('Area Number')
        return grid

    @property
    @lru_cache()
    def bound_grid(self):
        grid = self.grid
        grid_col = len(grid.columns)
        park_grid = gpd.sjoin(grid, self.boundary, op='intersects', how='inner')
        park_grid = park_grid.iloc[:, 0:grid_col]
        return park_grid

    @property
    @lru_cache()
    def border_cells(self):
        grid = self.grid
        grid_col = len(grid.columns)
        border = gpd.sjoin(grid, self.border_line, op='intersects', how='inner')
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

    def __init__(self, name, park, park_type, move_type, within=None, out=None, dislike=None, like=None,
                 start_cell=None, heading_cell=None, step_size=1, view_size=0, stay_in_cell=True,
                 geo_util_fctr=1, arrest_util=10, wild_save_fctr=1, wild_calve_fctr=1):
        self.name = name
        self.park = park  # Park class instance
        self.park_type = park_type  # 'full' / 'bound'
        self.move_type = move_type  # simulated movement - 'random' / 'strategic' / 'game'
        # within - polygon area to stay within, user input list of GeoDataFrames
        self.within = [] if within is None else within
        # GIS and park area options: 'roads', 'rivers', 'dams', 'mountains', 'trees', 'camps',
        #                            'picnic_spots', 'gates', 'water'
        # out - places can't enter
        # eg. out = ['rivers', 'dams', 'mountains', 'trees']
        self.out = [] if out is None else out
        # for random / game movement - places that deter: distance (m) to stay away
        # eg. dislike={'roads': 500, 'camps': 500, 'picnic_spots': 500, 'gates': 500}
        self.dislike = {} if dislike is None else dislike
        # for random / game movement - places that attract: distance (m) to stay near
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

    def add_within(self, within_item):
        if within_item not in self.within:
            self.within.append(within_item)

    def remove_within(self, within_item):
        if within_item in self.within:
            self.within.remove(within_item)

    def add_out(self, out_item):
        if out_item not in self.out:
            self.out.append(out_item)

    def remove_out(self, out_item):
        if out_item in self.out:
            self.out.remove(out_item)

    def update_dislike(self, dislike_key_val):
        self.dislike.update(dislike_key_val)

    def remove_dislike(self, dislike_key):
        if dislike_key in self.dislike:
            del self.dislike[dislike_key]

    def update_like(self, like_key_val):
        self.like.update(like_key_val)

    def remove_like(self, like_key):
        if like_key in self.like:
            del self.like[like_key]

    @property
    @lru_cache()
    def allowed_cells(self):
        # grid_type: 'square', 'park'
        if self.park_type == 'full':
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
            grid_bound = gpd.sjoin(grid_cells, grid_bound, op='intersects', how='inner')
            if grid_bound.shape[0] != 0:
                grid_cells = grid_bound.iloc[:, 0:grid_col]
        if len(out) != 0:
            for ftr in out:
                if self.park.features[ftr] is not None:
                    grid_exclude = gpd.sjoin(grid_cells, self.park.features[ftr], op='intersects', how='inner')
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
                            ftr_util = 0 - percent * self.geo_util_fctr if min_dist[i] < ftr_val else 0.5
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
                            ftr_util = 0 + percent * self.geo_util_fctr if min_dist[i] < ftr_val else 0.5
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
        start_cell = start_cell.append(start_time)
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

    def movement(self, curr_cell, heading_cell, sampling='prob'):
        # sampling - 'prob' / 'max'
        curr_perim = self.perim_grid(curr_cell, 'step')
        if heading_cell.name in curr_perim.index or heading_cell.name == curr_cell.name:
            select_step = heading_cell
            if 'Time' in select_step.index:
                select_step = select_step.drop('Time')
        else:
            heading_centroid = heading_cell['Centroids Proj']
            perim_centroids = curr_perim.set_geometry('Centroids Proj')
            perim_centroids.crs = self.park.proj_crs
            cell_prob = list(curr_perim['Select Prob'])
            dist = [x.distance(heading_centroid) for x in perim_centroids.geometry]
            dist_inv = [1 / x for x in dist]
            dist_inv_tot = sum(dist_inv)
            if sampling == 'max':
                # maximum probability
                prob = [x / dist_inv_tot for x in dist_inv]
                tot_prob = [prob[i] * cell_prob[i] for i in range(len(prob))]
                max_prob_ind = tot_prob.index(max(tot_prob))
                select_step = curr_perim.iloc[max_prob_ind, :]
            else:
                # probability sampling
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
        cell_time = pd.Series(datetime.now(), index=['Time'], name=select_step.name)
        select_step = select_step.append(cell_time)
        return select_step


class Wildlife(Player):

    def __init__(self, name, park, park_type, move_type, within=None, out=None, dislike=None, like=None, start_cell=None,
                 heading_cell=None, step_size=1, view_size=0, stay_in_cell=True, home_range=None, census=None):
        super().__init__(name, park, park_type, move_type, within, out, dislike, like, start_cell,
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

    def __init__(self, name, park, park_type, move_type, within=None, out=None, dislike=None, like=None, start_cell=None,
                 heading_cell=None, step_size=1, view_size=0, stay_in_cell=True, ranger_home=None,
                 geo_util_fctr=1, arrest_util=10, wild_save_fctr=1, wild_calve_fctr=1):
        super().__init__(name, park, park_type, move_type, within, out, dislike, like, start_cell,
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
                cells = gpd.sjoin(self.allowed_cells, ranger_home, op='intersects', how='inner')
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
        start_cell = start_cell.append(start_time)
        return start_cell


class Poacher(Player):

    def __init__(self, name, park, park_type, move_type, within=None, out=None, dislike=None, like=None, start_cell=None,
                 heading_cell=None, step_size=1, view_size=0, stay_in_cell=True, entry_type='border',
                 geo_util_fctr=1, arrest_util=10, wild_save_fctr=1, wild_calve_fctr=1):
        super().__init__(name, park, park_type, move_type, within, out, dislike, like, start_cell,
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
            col_geo = col_names.index('Map Cells')
            col_names_new = list(self.allowed_cells.columns + '_left')
            col_names_new[col_geo] = 'Map Cells'
            entry_cells = self.park.border_cells if self.entry_type == 'border' else self.park.edge_cells
            cells = gpd.sjoin(self.allowed_cells, entry_cells, how='inner', op='intersects')
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
        start_cell = start_cell.append(start_time)
        return start_cell


class Game:

    def __init__(self, name, wildlife, ranger, poacher, leader='ranger', game_type=None, same_start=False, seed=None,
                 end_moves=100, games_pm=30, months=1000):
        self.name = name
        self.wildlife = wildlife
        self.ranger = ranger
        self.poacher = poacher
        self.leader = leader
        self.game_type = game_type
        self.same_start = same_start
        self.seed = seed
        self.end_moves = end_moves
        self.games_pm = games_pm
        self.months = months

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

                        p_utility = p_utility + self.poacher.wild_save_fctr * p_wildlife
                        p_utility = p_utility + self.poacher.wild_calve_fctr * p_calves
                        if not capture:
                            r_utility = r_utility - self.ranger.wild_save_fctr * p_wildlife
                            r_utility = r_utility - self.ranger.wild_calve_fctr * p_calves

                    r_payoff[r_mtrx, p_mtrx] = r_utility
                    p_payoff[r_mtrx, p_mtrx] = p_utility
                    p_mtrx += 1
                r_mtrx += 1
            return {'ranger': r_payoff, 'poacher': p_payoff}

    def game_solution(self):
        if self.game_type is None:
            return None
        if self.game_type == 'spne':
            leader = 'p1' if self.leader == 'ranger' else 'p2'
            sol = gt.spne(self.strategies['ranger'], self.strategies['poacher'],
                          self.payoffs['ranger'], self.payoffs['poacher'], leader)
            return {'ranger': sol['p1_optimal'], 'poacher': sol['p2_optimal']}

    def game_sol_instance(self, game_sol):
        if self.game_type is None:
            return None
        else:
            # if game_solution returns mixed strategy, this method selects random strategy from the distribution
            if self.game_type == 'spne':
                r_strat = game_sol['ranger']
                p_strat = game_sol['poacher']
            else:
                # probability sampling
                r_prob = [(self.strategies['ranger'][x], game_sol['ranger'][x])
                          for x in range(len(game_sol['ranger']))]
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

                p_prob = [(self.strategies['poacher'][x], game_sol['poacher'][x])
                          for x in range(len(game_sol['poacher']))]
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

            return {'ranger': r_strat, 'poacher': p_strat}


def sim_movement(game):
    move = 0
    columns = ['Wildlife Current', 'Wildlife Start', 'Wildlife Heading', 'Wildlife Toward', 'Wildlife Reach Start',
               'Wildlife Reach Heading', 'Ranger Current', 'Ranger Start', 'Ranger Heading', 'Ranger Toward',
               'Ranger Reach Start', 'Ranger Reach Heading', 'Poacher Current', 'Poacher Start',
               'Poacher Heading', 'Poacher Toward', 'Poacher Reach Start', 'Poacher Reach Heading',
               'Poach Cell', 'Poach Events', 'Capture Cell', 'Capture Events', 'Leave Cell', 'Leave Events',
               'Leave Before', 'Leave After', 'Capture Before', 'Capture After']
    ind = list(range(1, game.end_moves + 1))
    results = pd.DataFrame(0.0, index=ind, columns=columns)
    results.index.name = 'Move'

    path_columns = list(game.wildlife.park.grid.columns)
    path_columns.extend(['Time', 'Trajectory', 'PlayerID', 'Player', 'Move', 'Leave', 'Capture', 'Poach'])

    wildlife_start = game.wildlife.start()
    wildlife_heading = game.wildlife.heading()
    wildlife_traj = 1
    wildlife_toward = wildlife_heading
    wildlife_curr = wildlife_start
    wildlife_path = pd.DataFrame(columns=path_columns)
    wildlife_add = pd.Series([wildlife_traj, 1, 'Wildlife', move, 0, 0, 0],
                             index=['Trajectory', 'PlayerID', 'Player', 'Move', 'Leave', 'Capture', 'Poach'],
                             name=wildlife_curr.name)
    wildlife_add = wildlife_curr.append(wildlife_add)
    wildlife_path = wildlife_path.append(wildlife_add)

    ranger_start = game.ranger.start()
    ranger_heading = game.ranger.heading()
    ranger_traj = 1
    ranger_toward = ranger_heading
    ranger_curr = ranger_start
    ranger_path = pd.DataFrame(columns=path_columns)
    ranger_add = pd.Series([ranger_traj, 2, 'Ranger', move, 0, 0, 0],
                           index=['Trajectory', 'PlayerID', 'Player', 'Move', 'Leave', 'Capture', 'Poach'],
                           name=ranger_curr.name)
    ranger_add = ranger_curr.append(ranger_add)
    ranger_path = ranger_path.append(ranger_add)

    poacher_start = game.poacher.start()
    poacher_heading = game.poacher.heading()
    poacher_traj = 1
    poacher_toward = poacher_heading
    poacher_curr = poacher_start
    poacher_path = pd.DataFrame(columns=path_columns)
    poacher_add = pd.Series([poacher_traj, 3, 'Poacher', move, 0, 0, 0],
                            index=['Trajectory', 'PlayerID', 'Player', 'Move', 'Leave', 'Capture', 'Poach'],
                            name=poacher_curr.name)
    poacher_add = poacher_curr.append(poacher_add)
    poacher_path = poacher_path.append(poacher_add)

    while move < game.end_moves:
        move += 1

        wildlife_curr = game.wildlife.movement(wildlife_curr, wildlife_toward)
        wildlife_add = pd.Series([wildlife_traj, 1, 'Wildlife', move, 0, 0, 0],
                                 index=['Trajectory', 'PlayerID', 'Player', 'Move', 'Leave', 'Capture', 'Poach'],
                                 name=wildlife_curr.name)
        wildlife_add = wildlife_curr.append(wildlife_add)
        wildlife_path = wildlife_path.append(wildlife_add)
        results.loc[move, 'Wildlife Current'] = wildlife_curr.name
        results.loc[move, 'Wildlife Start'] = wildlife_start.name
        results.loc[move, 'Wildlife Heading'] = wildlife_heading.name
        results.loc[move, 'Wildlife Toward'] = wildlife_toward.name

        if wildlife_curr.name == wildlife_toward.name:
            if wildlife_curr.name == wildlife_heading.name:
                results.loc[move, 'Wildlife Reach Heading'] = 1
                wildlife_toward = wildlife_start
            if wildlife_curr.name == wildlife_start.name:
                results.loc[move, 'Wildlife Reach Start'] = 1
                wildlife_toward = wildlife_heading
            wildlife_traj += 1

        ranger_curr = game.ranger.movement(ranger_curr, ranger_toward)
        ranger_add = pd.Series([ranger_traj, 2, 'Ranger', move, 0, 0, 0],
                               index=['Trajectory', 'PlayerID', 'Player', 'Move', 'Leave', 'Capture', 'Poach'],
                               name=ranger_curr.name)
        ranger_add = ranger_curr.append(ranger_add)
        ranger_path = ranger_path.append(ranger_add)
        results.loc[move, 'Ranger Current'] = ranger_curr.name
        results.loc[move, 'Ranger Start'] = ranger_start.name
        results.loc[move, 'Ranger Heading'] = ranger_heading.name
        results.loc[move, 'Ranger Toward'] = ranger_toward.name

        ranger_view = game.ranger.perim_grid(ranger_curr, 'view')
        if ranger_curr.name == ranger_toward.name:
            if ranger_curr.name == ranger_heading.name:
                results.loc[move, 'Ranger Reach Heading'] = 1
                ranger_toward = ranger_start
            if ranger_curr.name == ranger_start.name:
                results.loc[move, 'Ranger Reach Start'] = 1
                ranger_toward = ranger_heading
            ranger_traj += 1

        poacher_curr = game.poacher.movement(poacher_curr, poacher_toward)
        poacher_add = pd.Series([poacher_traj, 3, 'Poacher', move, 0, 0, 0],
                                index=['Trajectory', 'PlayerID', 'Player', 'Move', 'Leave', 'Capture', 'Poach'],
                                name=poacher_curr.name)
        poacher_add = poacher_curr.append(poacher_add)
        poacher_path = poacher_path.append(poacher_add)
        poacher_view = game.poacher.perim_grid(poacher_curr, 'view')
        results.loc[move, 'Poacher Start'] = poacher_start.name
        results.loc[move, 'Poacher Current'] = poacher_curr.name
        results.loc[move, 'Poacher Heading'] = poacher_heading.name
        results.loc[move, 'Poacher Toward'] = poacher_toward.name

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
        poaches_traj = sum(results.loc[poacher_moves_traj, 'Poach Events'])

        if poacher_curr.name == poacher_toward.name:
            if poacher_curr.name == poacher_heading.name:
                results.loc[move, 'Poacher Reach Heading'] = 1
                poacher_toward = poacher_start
            if poacher_curr.name == poacher_start.name:
                results.loc[move, 'Poacher Reach Start'] = 1
                poacher_toward = poacher_heading
            poacher_traj += 1

        if wildlife_curr.name in poacher_view.index:
            if poaches_traj == 0 and not poacher_curr_heading:
                results.loc[move, 'Poach Cell'] = wildlife_curr.name
                results.loc[move, 'Poach Events'] = 1
                wildlife_path.loc[wildlife_path['Move'] == move, 'Poach'] = 1
                ranger_path.loc[ranger_path['Move'] == move, 'Poach'] = 1
                poacher_path.loc[poacher_path['Move'] == move, 'Poach'] = 1
                if poacher_twd_heading:
                    poacher_toward = poacher_start
                    poacher_traj += 1
                if game.poacher.move_type == 'strategic':
                    poacher_heading = poacher_curr

        if poacher_twd_start and poacher_curr_start:
            results.loc[move, 'Leave Cell'] = poacher_curr.name
            results.loc[move, 'Leave Events'] = 1
            wildlife_path.loc[wildlife_path['Move'] == move, 'Leave'] = 1
            ranger_path.loc[ranger_path['Move'] == move, 'Leave'] = 1
            poacher_path.loc[poacher_path['Move'] == move, 'Leave'] = 1
            if poaches_traj > 0:
                results.loc[move, 'Leave After'] = 1
            else:
                results.loc[move, 'Leave Before'] = 1

        if poacher_curr.name in ranger_view.index:
            results.loc[move, 'Capture Cell'] = poacher_curr.name
            results.loc[move, 'Capture Events'] = 1
            wildlife_path.loc[wildlife_path['Move'] == move, 'Capture'] = 1
            ranger_path.loc[ranger_path['Move'] == move, 'Capture'] = 1
            poacher_path.loc[poacher_path['Move'] == move, 'Capture'] = 1
            if poaches_traj > 0:
                results.loc[move, 'Capture After'] = 1
            else:
                results.loc[move, 'Capture Before'] = 1
            if game.poacher.move_type == 'strategic':
                poacher_start = game.poacher.start()
            poacher_curr = poacher_start
            poacher_toward = poacher_heading
            poacher_traj += 1

    sum_cols = ['Wildlife Reach Start', 'Wildlife Reach Heading', 'Ranger Reach Start', 'Ranger Reach Heading',
                'Poacher Reach Start', 'Poacher Reach Heading', 'Poach Events', 'Capture Events', 'Leave Events',
                'Leave Before', 'Leave After', 'Capture Before', 'Capture After']
    res_sum = (results.loc[:, sum_cols]).sum(axis=0)
    def_crs = game.wildlife.park.grid.crs
    paths = pd.concat([wildlife_path, ranger_path, poacher_path])
    paths = paths.rename(columns={'Time': 't', 'Centroids': 'geometry'})
    paths = gpd.GeoDataFrame(paths, crs=def_crs, geometry='geometry')
    traj_coll = mpd.TrajectoryCollection(paths.set_index('t'), 'PlayerID')

    return {'totals': res_sum, 'moves': results, 'trajectories': traj_coll}


def sim_game(game):
    if game.seed is not None:
        random.seed(game.seed)
    ind_dat = [range(1, (game.months + 1)), range(1, (game.games_pm + 1))]
    ind = pd.MultiIndex.from_product(ind_dat, names=['Month', 'Game'])
    columns = ['Wildlife Reach Start', 'Wildlife Reach Heading', 'Ranger Reach Start', 'Ranger Reach Heading',
               'Poacher Reach Start', 'Poacher Reach Heading', 'Poach Events', 'Capture Events', 'Leave Events',
               'Capture Before', 'Capture After', 'Leave Before', 'Leave After']
    results = pd.DataFrame(0.0, index=ind, columns=columns)
    moves = []
    trajectories = []

    # month iterations
    for m in range(1, (game.months + 1)):
        if game.same_start:
            game.wildlife.start_cell = game.wildlife.start()
            game.ranger.start_cell = game.ranger.start()
            game.poacher.start_cell = game.poacher.start()

        # calculate the game solution for the month
        game_month = None
        if game.game_type is not None:
            game_month = game.game_solution()

        # game iterations
        for g in range(1, (game.games_pm + 1)):
            # get game solution instance
            if game.game_type is not None:
                game_strat = game.game_sol_instance(game_month)
                if game.ranger.movement == 'game':
                    game.ranger.heading_cell = game_strat['ranger']
                if game.poacher.movement == 'game':
                    game.poacher.heading_cell = game_strat['poacher']
            single_game = sim_movement(game)
            results.loc[m, g] = single_game['totals']
            moves.append(single_game['moves'])
            trajectories.append(single_game['trajectories'])

    results.applymap("{0:.3f}".format)
    sum_per_month = results.groupby(by='Month').sum()
    ave_per_month = sum_per_month.mean()
    return {'all': results, 'sum': sum_per_month, 'ave': ave_per_month, 'moves': moves, 'trajectories': trajectories}
