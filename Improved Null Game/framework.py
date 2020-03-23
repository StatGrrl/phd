"""
Module docstring:
TO DO:
* docstrings!
* change Park class to get geographical features from Google Earth Engine
* plot method for Park class
"""
# import za_gis
import knp
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point, LineString
import random
from datetime import datetime
import gametheory as gt
import movingpandas as mpd


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

    def __init__(self, name, camps=None, picnic_spots=None, gates=None, water=None, rhino_sightings=None):
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
        self.rhino_sightings = rhino_sightings

    def grid(self, x_cell_length, y_cell_length, subarea=None):
        """
        Calculate grid for simulations and game to overlay on map of the park.
        Change to projected crs to calculate distance in meters
        :param x_cell_length: horizontal length of cells in meters
        :param y_cell_length: vertical length of cells in meters
        :param subarea: bounds of smaller area of park to analyse
        :return: GeoDataFrame containing cell numbers, array indices of the grid and map geometry
        """
        boundary = self.boundary
        boundary = boundary.to_crs(self.proj_crs)
        if subarea is None:
            bounds = boundary.total_bounds
        else:
            bounds_gs = bounds_to_points(subarea, crs=self.default_crs)  # (lon, lat)
            bounds_gs = bounds_gs.to_crs(self.proj_crs)  # (y, x)
            bounds = np.array([bounds_gs[0].x, bounds_gs[0].y, bounds_gs[2].x, bounds_gs[2].y])
        x_min = bounds[2]
        y_min = bounds[3]
        x_max = bounds[0]
        y_max = bounds[1]
        grid_cells = pd.DataFrame(columns=['Area Number', 'Grid x', 'Grid y',
                                           'Grid Index', 'Map Cells'])
        cell_num = 1
        df_index = 0
        y_index = 0
        y = y_min
        while y < y_max:
            x_index = 0
            x = x_min
            y_mid = y + y_cell_length
            while x < x_max:
                x_mid = x + x_cell_length
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
        grid_proj['Select Prob'] = 1
        grid_proj['Utility'] = 0
        # rhino sightings
        if self.rhino_sightings is not None:
            grid_proj['Total Rhino'] = 0
            grid_proj['Total Calves'] = 0
            rhino_sightings = self.rhino_sightings.to_crs(self.proj_crs)
            grid_rhino = gpd.sjoin(grid_proj, rhino_sightings, how='inner', op='intersects')
            grid_rhino = grid_rhino.dissolve(by='Area Number', aggfunc=sum)
            grid_proj = grid_proj.set_index('Area Number')
            grid_proj['Total Rhino'] = grid_rhino['TOTAL']
            no_rhino = grid_proj['Total Rhino'].isna()
            grid_proj.loc[no_rhino, 'Total Rhino'] = 0
            grid_proj['Total Calves'] = grid_rhino['CALVES']
            no_calves = grid_proj['Total Calves'].isna()
            grid_proj.loc[no_calves, 'Total Calves'] = 0
        else:
            grid_proj = grid_proj.set_index('Area Number')
        grid_col = grid_proj.shape[1]
        park_proj = gpd.sjoin(grid_proj, boundary, op='intersects', how='inner')
        park_proj = park_proj.iloc[:, 0:grid_col]
        border_line = self.border_line.to_crs(self.proj_crs)
        border_proj = gpd.sjoin(grid_proj, border_line, op='intersects', how='inner')
        border_proj = border_proj.iloc[:, 0:grid_col]
        x_max = max(grid_proj['Grid x'])
        y_max = max(grid_proj['Grid y'])
        edge_cells = []
        for i in range(x_max + 1):
            for j in range(y_max + 1):
                if i in (0, x_max) or j in (0, y_max):
                    edge_cells.append([i, j])
        edge_index = [x in list(edge_cells) for x in grid_proj['Grid Index']]
        edge_proj = grid_proj[edge_index]
        grid = grid_proj.to_crs(self.default_crs)
        park = park_proj.to_crs(self.default_crs)
        border = border_proj.to_crs(self.default_crs)
        edge = edge_proj.to_crs(self.default_crs)
        return {'square': grid, 'square_proj': grid_proj, 'park': park, 'park_proj': park_proj,
                'edge': edge, 'edge_proj': edge_proj, 'border': border, 'border_proj': border_proj}


class Player:

    def __init__(self, number, movement, within=None, out=None, dislike=None, like=None,
                 start_cell=None, heading_cell=None, step_size=1, view_size=0, stay_in_cell=False,
                 geo_util_fctr=1, arrest_util=10, wild_save_fctr=1, wild_calve_fctr=1):
        self.number = number
        self.movement = movement  # simulated movement - 'random' / 'structured' / 'game'
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

    def allowed_cells(self, park, grid, grid_type):
        # grid_type: 'square', 'park'
        within = self.within
        out = self.out
        dislike = self.dislike
        like = self.like
        grid_cells = grid[grid_type]
        grid_col = grid_cells.shape[1]
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
                if park.features[ftr] is not None:
                    grid_exclude = gpd.sjoin(grid_cells, park.features[ftr], op='intersects', how='inner')
                    if grid_exclude.shape[0] != 0:
                        grid_cells = gpd.overlay(grid_cells, grid_exclude, how='difference')
                        grid_cells = grid_cells.iloc[:, 0:grid_col]
        grid_cells = grid_cells.to_crs(park.proj_crs)
        grid_centroids = grid_cells.centroid
        grid_cells['Centroids'] = grid_centroids
        if self.movement in ['structured', 'game']:
            col_prob = grid_cells.columns.get_loc('Select Prob')
            col_util = grid_cells.columns.get_loc('Utility')
            if len(dislike) != 0:
                for ftr, ftr_val in dislike.items():
                    if park.features[ftr] is not None:
                        ftr_series = park.features[ftr].geometry.to_crs(park.proj_crs)
                        min_dist = [ftr_series.distance(x).min() for x in grid_centroids]
                        # if min_dist < ftr_val, decrease probability / utility as min_dist decreases
                        for i in range(len(min_dist)):
                            percent = (ftr_val - min_dist[i]) / ftr_val
                            # if self.movement == 'structured':
                            ftr_prob = 0.5 * (1 - percent) if min_dist[i] < ftr_val else 0.5
                            grid_cells.iloc[i, col_prob] = grid_cells.iloc[i, col_prob] * ftr_prob
                            # if self.movement == 'game':
                            ftr_util = 0 - percent * self.geo_util_fctr if min_dist[i] < ftr_val else 0.5
                            grid_cells.iloc[i, col_util] = grid_cells.iloc[i, col_util] + ftr_util
            if len(like) != 0:
                for ftr, ftr_val in like.items():
                    if park.features[ftr] is not None:
                        ftr_series = park.features[ftr].geometry.to_crs(park.proj_crs)
                        min_dist = [ftr_series.distance(x).min() for x in grid_centroids]
                        for i in range(len(min_dist)):
                            percent = (ftr_val - min_dist[i]) / ftr_val
                            # if self.movement == 'structured':
                            ftr_prob = 0.5 * (1 + percent) if min_dist[i] < ftr_val else 0.5
                            grid_cells.iloc[i, col_prob] = grid_cells.iloc[i, col_prob] * ftr_prob
                            # if self.movement == 'game':
                            ftr_util = 0 + percent * self.geo_util_fctr if min_dist[i] < ftr_val else 0.5
                            grid_cells.iloc[i, col_util] = grid_cells.iloc[i, col_util] + ftr_util
        grid_cells_def_crs = grid_cells.to_crs(park.default_crs)
        grid_cells_def_crs['Centroids'] = grid_cells_def_crs.centroid
        return {'cells': grid_cells_def_crs, 'cells_proj': grid_cells}

    def heading(self, player_cells):
        if self.heading_cell is not None:
            heading_cell = self.heading_cell['cells']
            heading_cell_proj = self.heading_cell['cells_proj']
            heading = {'cells': heading_cell, 'cells_proj': heading_cell_proj}
        else:
            heading = None
            len_cells = player_cells['cells'].shape[0]
            perim = {'cells': [], 'cells_proj': []}
            while len(perim['cells']) < 3:
                rand_cell = random.randint(0, len_cells - 1)
                heading_cell = player_cells['cells'].iloc[rand_cell, :]
                heading_cell_proj = player_cells['cells_proj'].iloc[rand_cell, :]
                heading = {'cells': heading_cell, 'cells_proj': heading_cell_proj}
                perim = perim_grid(heading, self.step_size, self.view_size, 'step', player_cells)
        return heading

    def moving(self, curr_cell, player_cells, heading_cell):
        step_size = self.step_size
        perim_type = 'view' if self.stay_in_cell else 'step'
        curr_perim = perim_grid(curr_cell, step_size, step_size, perim_type, player_cells)
        if heading_cell['cells']['Grid Index'] in list(curr_perim['cells']['Grid Index']) or \
                heading_cell['cells']['Grid Index'] == curr_cell['cells']['Grid Index']:
            select_step = heading_cell['cells']
            select_step_proj = heading_cell['cells_proj']
        else:
            heading_centroid = heading_cell['cells_proj']['Map Cells'].centroid
            perim_centroids = curr_perim['cells_proj'].centroid
            cell_prob = list(curr_perim['cells_proj']['Select Prob'])
            dist = [heading_centroid.distance(x) for x in perim_centroids]
            dist_inv = [1 / x for x in dist]
            dist_inv_tot = sum(dist_inv)

            # maximum probability
            # prob = [x / dist_inv_tot for x in dist_inv]
            # tot_prob = [prob[i] * cell_prob[i] for i in range(len(curr_perim['cells_proj']))]
            # max_prob_ind = tot_prob.index(max(tot_prob))
            # select_step = curr_perim['cells'].iloc[max_prob_ind, :]
            # select_step_proj = curr_perim['cells_proj'].iloc[max_prob_ind, :]

            # probability sampling
            prob = [(list(curr_perim['cells'].index)[x],
                     dist_inv[x] / dist_inv_tot * cell_prob[x]) for x in range(len(dist))]
            prob_struct = np.array(prob, dtype=[('perim_ind', int), ('tot_prob', float)])
            prob_sort = np.sort(prob_struct, order='tot_prob')
            prob_cumm = [0]
            for i in range(len(prob_sort)):
                prob_cumm.append(prob_cumm[i] + prob_sort['tot_prob'][i])
            prob_cumm.pop(0)
            select_prob = random.random()
            select_ind = np.array(prob_cumm) > select_prob
            if not select_ind.any():
                select = prob_sort['perim_ind'][len(prob_sort) - 1]
            else:
                select = prob_sort['perim_ind'][0]
            select_step = curr_perim['cells'].loc[select, :]
            select_step_proj = curr_perim['cells_proj'].loc[select, :]
            cell_time = pd.Series(datetime.now(), index=['Time'], name=select_step.name)
            select_step = select_step.append(cell_time)
            select_step_proj = select_step_proj.append(cell_time)
        return {'cells': select_step, 'cells_proj': select_step_proj}


class Wildlife(Player):

    def __init__(self, number, movement, within=None, out=None, dislike=None, like=None, start_cell=None,
                 heading_cell=None, step_size=1, view_size=0, stay_in_cell=True, home_range=None, census=None):
        super().__init__(number, movement, within, out, dislike, like, start_cell,
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

    def start(self, player_cells):
        if self.start_cell is not None:
            start_cell = self.start_cell['cells']
            start_cell_proj = self.start_cell['cells_proj']
            start = {'cells': start_cell, 'cells_proj': start_cell_proj}
        else:
            start = None
            len_cells = player_cells['cells'].shape[0]
            perim = {'cells': [], 'cells_proj': []}
            while len(perim['cells']) < 3:
                rand_cell = random.randint(0, len_cells - 1)
                start_cell = player_cells['cells'].iloc[rand_cell, :]
                start_cell_proj = player_cells['cells_proj'].iloc[rand_cell, :]
                start_time = pd.Series(datetime.now(), index=['Time'], name=start_cell.name)
                start_cell = start_cell.append(start_time)
                start_cell_proj = start_cell_proj.append(start_time)
                start = {'cells': start_cell, 'cells_proj': start_cell_proj}
                perim = perim_grid(start, self.step_size, self.view_size, 'step', player_cells)
        return start


class Ranger(Player):

    def __init__(self, number, movement, within=None, out=None, dislike=None, like=None, start_cell=None,
                 heading_cell=None, step_size=1, view_size=0, stay_in_cell=False, ranger_home=None,
                 geo_util_fctr=1, arrest_util=10, wild_save_fctr=1, wild_calve_fctr=1):
        super().__init__(number, movement, within, out, dislike, like, start_cell,
                         heading_cell, step_size, view_size, stay_in_cell, geo_util_fctr, arrest_util,
                         wild_save_fctr, wild_calve_fctr)
        self.ranger_home = ranger_home

    def start(self, player_cells):
        if self.start_cell is not None:
            start_cell = self.start_cell['cells']
            start_cell_proj = self.start_cell['cells_proj']
            start = {'cells': start_cell, 'cells_proj': start_cell_proj}
        else:
            start = None
            ranger_home = self.ranger_home
            if ranger_home is None:
                cells = player_cells['cells']
                cells_proj = player_cells['cells_proj']
            else:
                cols = player_cells['cells'].shape[1]
                proj_crs = player_cells['cells_proj'].crs
                cells = gpd.sjoin(player_cells['cells'], ranger_home, op='intersects', how='inner')
                cells = cells.iloc[:, 0:cols]
                if cells.shape[0] < 2:
                    cells = player_cells['cells']
                cells_proj = cells.to_crs(proj_crs)
            len_cells = cells.shape[0]
            perim = {'cells': [], 'cells_proj': []}
            while len(perim['cells']) < 3:
                rand_cell = random.randint(0, len_cells - 1)
                start_cell = cells.iloc[rand_cell, :]
                start_cell_proj = cells_proj.iloc[rand_cell, :]
                start_time = pd.Series(datetime.now(), index=['Time'], name=start_cell.name)
                start_cell = start_cell.append(start_time)
                start_cell_proj = start_cell_proj.append(start_time)
                start = {'cells': start_cell, 'cells_proj': start_cell_proj}
                perim = perim_grid(start, self.step_size, self.view_size, 'step', player_cells)
        return start


class Poacher(Player):

    def __init__(self, number, movement, within=None, out=None, dislike=None, like=None, start_cell=None,
                 heading_cell=None, step_size=1, view_size=0, stay_in_cell=False, entry_type='border',
                 geo_util_fctr=1, arrest_util=10, wild_save_fctr=1, wild_calve_fctr=1):
        super().__init__(number, movement, within, out, dislike, like, start_cell,
                         heading_cell, step_size, view_size, stay_in_cell, geo_util_fctr, arrest_util,
                         wild_save_fctr, wild_calve_fctr)
        self.entry_type = entry_type

    def start(self, player_cells, grid):
        if self.start_cell is not None:
            start_cell = self.start_cell['s']
            start_cell_proj = self.start_cell['cells_proj']
            start = {'cells': start_cell, 'cells_proj': start_cell_proj}
        else:
            start = None
            col_names = list(player_cells['cells'].columns)
            col_geo = col_names.index('Map Cells')
            col_names_new = list(player_cells['cells'].columns + '_1')
            col_names_new[col_geo] = 'geometry'
            col_cent = col_names_new.index('Centroids_1')
            col_names_new[col_cent] = 'Centroids'
            proj_crs = player_cells['cells_proj'].crs
            entry_type = self.entry_type
            cells = gpd.overlay(player_cells['cells'], grid[entry_type], how='intersection')
            cells = cells.loc[:, col_names_new]
            col_names.remove('Centroids')
            col_names_new.remove('Centroids')
            col_dict = {col_names_new[i]: col_names[i] for i in range(len(col_names))}
            cells = cells.rename(columns=col_dict).set_geometry('Map Cells')
            cells_proj = cells.to_crs(proj_crs)
            len_cells = cells.shape[0]
            perim = {'cells': [], 'cells_proj': []}
            while len(perim['cells']) < 3:
                rand_cell = random.randint(0, len_cells - 1)
                start_cell = cells.iloc[rand_cell, :]
                start_cell_proj = cells_proj.iloc[rand_cell, :]
                start_time = pd.Series(datetime.now(), index=['Time'], name=start_cell.name)
                start_cell = start_cell.append(start_time)
                start_cell_proj = start_cell_proj.append(start_time)
                start = {'cells': start_cell, 'cells_proj': start_cell_proj}
                perim = perim_grid(start, self.step_size, self.view_size, 'step', player_cells)
        return start


def perim_grid(curr_cell, step_size, view_size, perim_type, player_cells):
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

    curr_cell_index = curr_cell['cells']['Grid Index']
    perim_index = cell_perim(curr_cell_index, step_size, view_size, perim_type)
    perim_grid_index = [x in perim_index for x in player_cells['cells']['Grid Index']]
    perim = player_cells['cells'][perim_grid_index]
    perim_proj = player_cells['cells_proj'][perim_grid_index]
    return {'cells': perim, 'cells_proj': perim_proj}


def pure_stcklbrg(ranger, ranger_cells, poacher, poacher_cells, leader='p1'):
    # get strategies
    ranger_index = ranger_cells['cells'].index
    poacher_index = poacher_cells['cells'].index
    ranger_strategies = [str(x) for x in ranger_index]
    poacher_strategies = [str(x) for x in poacher_index]

    # get payoffs
    r_payoff = np.zeros(shape=(len(ranger_index), len(poacher_index)))
    p_payoff = np.zeros(shape=(len(ranger_index), len(poacher_index)))
    r_mtrx = 0
    for r_area in ranger_index:
        p_mtrx = 0
        for p_area in poacher_index:
            r_utility = ranger_cells['cells'].loc[r_area, 'Utility']
            p_utility = poacher_cells['cells'].loc[p_area, 'Utility']
            if ranger.view_size == 0:
                capture = True if p_area == r_area else False
            else:
                ranger_curr_cell = ranger_cells['cells'].loc[r_area, :]
                ranger_curr_cell_proj = ranger_cells['cells_proj'].loc[r_area, :]
                ranger_curr = {'cells': ranger_curr_cell, 'cells_proj': ranger_curr_cell_proj}
                ranger_view_perim = perim_grid(ranger_curr, None, ranger.view_size,
                                               'view', ranger_cells)
                perim_area = [x for x in ranger_view_perim['cells'].index]
                capture = True if p_area in perim_area else False
            if capture:
                r_utility = r_utility + ranger.arrest_util
                p_utility = p_utility - poacher.arrest_util
            if 'Total Rhino' in ranger_cells['cells'].columns:
                r_rhino = ranger_cells['cells'].loc[r_area, 'Total Rhino']
                r_calves = ranger_cells['cells'].loc[r_area, 'Total Calves']
                p_rhino = poacher_cells['cells'].loc[p_area, 'Total Rhino']
                p_calves = poacher_cells['cells'].loc[p_area, 'Total Calves']

                r_utility = r_utility + ranger.wild_save_fctr * r_rhino
                r_utility = r_utility + ranger.wild_calve_fctr * r_calves

                p_utility = p_utility + poacher.wild_save_fctr * p_rhino
                p_utility = p_utility + poacher.wild_calve_fctr * p_calves
                if not capture:
                    r_utility = r_utility - ranger.wild_save_fctr * p_rhino
                    r_utility = r_utility - ranger.wild_calve_fctr * p_calves

            r_payoff[r_mtrx, p_mtrx] = r_utility
            p_payoff[r_mtrx, p_mtrx] = p_utility
            p_mtrx += 1
        r_mtrx += 1

    # get subgame perfect nash equilibrium
    equilibrium = gt.spne(ranger_strategies, poacher_strategies, r_payoff, p_payoff, leader)

    ranger_opt = ranger_cells['cells'].loc[int(equilibrium[0][0]), ]
    ranger_opt_proj = ranger_cells['cells_proj'].loc[int(equilibrium[0][0]), ]
    ranger_optimal = {'cells': ranger_opt, 'cells_proj': ranger_opt_proj}
    poacher_opt = poacher_cells['cells'].loc[int(equilibrium[0][1], ), ]
    poacher_opt_proj = poacher_cells['cells_proj'].loc[int(equilibrium[0][1], ), ]
    poacher_optimal = {'cells': poacher_opt, 'cells_proj': poacher_opt_proj}

    return {'ranger strategies': ranger_strategies, 'poacher strategies': poacher_strategies,
            'ranger payoff matrix': r_payoff, 'poacher payoff matrix': p_payoff,
            'ranger optimal': ranger_optimal, 'poacher optimal': poacher_optimal,
            'ranger expected payoff': equilibrium[1][0], 'poacher expected payoff': equilibrium[1][1]}


def one_game(grid, rhino, rhino_cells, ranger, ranger_cells, poacher, poacher_cells, end_moves=100):

    move = 0

    columns = ['Rhino Current', 'Rhino Start', 'Rhino Heading', 'Rhino Toward', 'Rhino Reach Start',
               'Rhino Reach Heading', 'Ranger Current', 'Ranger Start', 'Ranger Heading', 'Ranger Toward',
               'Ranger Reach Start', 'Ranger Reach Heading', 'Poacher Current', 'Poacher Start',
               'Poacher Heading', 'Poacher Toward', 'Poacher Reach Start', 'Poacher Reach Heading',
               'Poach Cell', 'Poach Events', 'Capture Cell', 'Capture Events',
               'Leave Before', 'Leave After', 'Catch Before', 'Catch After']
    ind = list(range(1, end_moves + 1))
    results = pd.DataFrame(0.0, index=ind, columns=columns)
    results.index.name = 'Move'

    path_columns = list(grid['square'].columns)
    path_columns.extend(['Time', 'Trajectory', 'Player', 'Moves'])

    if rhino.start_cell is not None:
        rhino_start = rhino.start_cell
    else:
        rhino_start = rhino.start(rhino_cells)
    if rhino.heading_cell is not None:
        rhino_heading = rhino.heading_cell
    else:
        rhino_heading = rhino.heading(rhino_cells)
    rhino_traj = 1
    rhino_toward = rhino_heading
    rhino_curr = rhino_start
    rhino_path = pd.DataFrame(columns=path_columns)
    rhino_add = pd.Series([rhino_traj, 1, 'Rhino', move], index=['Trajectory', 'PlayerID', 'Player', 'Moves'],
                          name=rhino_curr['cells'].name)
    rhino_add = rhino_curr['cells'].append(rhino_add)
    rhino_path = rhino_path.append(rhino_add)

    if ranger.start_cell is not None:
        ranger_start = ranger.start_cell
    else:
        ranger_start = ranger.start(ranger_cells)
    if ranger.heading_cell is not None:
        ranger_heading = ranger.heading_cell
    else:
        ranger_heading = ranger.heading(ranger_cells)
    ranger_traj = 1
    ranger_toward = ranger_heading
    ranger_curr = ranger_start
    ranger_path = pd.DataFrame(columns=path_columns)
    ranger_add = pd.Series([ranger_traj, 2, 'Ranger', move], index=['Trajectory', 'PlayerID', 'Player', 'Moves'],
                           name=ranger_curr['cells'].name)
    ranger_add = ranger_curr['cells'].append(ranger_add)
    ranger_path = ranger_path.append(ranger_add)

    if poacher.start_cell is not None:
        poacher_start = poacher.start_cell
    else:
        poacher_start = poacher.start(poacher_cells, grid)
    if poacher.heading_cell is not None:
        poacher_heading = poacher.heading_cell
    else:
        poacher_heading = poacher.heading(poacher_cells)
    poacher_traj = 1
    poacher_toward = poacher_heading
    poacher_curr = poacher_start
    poacher_path = pd.DataFrame(columns=path_columns)
    poacher_add = pd.Series([poacher_traj, 3, 'Poacher', move], index=['Trajectory', 'PlayerID', 'Player', 'Moves'],
                            name=poacher_curr['cells'].name)
    poacher_add = poacher_curr['cells'].append(poacher_add)
    poacher_path = poacher_path.append(poacher_add)

    while move < end_moves:
        move += 1

        rhino_curr = rhino.moving(rhino_curr, rhino_cells, rhino_toward)
        rhino_add = pd.Series([rhino_traj, 1, 'Rhino', move],
                              index=['Trajectory', 'PlayerID', 'Player', 'Moves'],
                              name=rhino_curr['cells'].name)
        rhino_add = rhino_curr['cells'].append(rhino_add)
        rhino_path = rhino_path.append(rhino_add)
        if rhino_curr['cells'].name == rhino_toward['cells'].name:
            if rhino_curr['cells'].name == rhino_heading['cells'].name:
                results.loc[move, 'Rhino Reach Heading'] = 1
                rhino_toward = rhino_start
            if rhino_curr['cells'].name == rhino_start['cells'].name:
                results.loc[move, 'Rhino Reach Start'] = 1
                rhino_toward = rhino_heading
            rhino_traj += 1
        results.loc[move, 'Rhino Current'] = rhino_curr['cells'].name
        results.loc[move, 'Rhino Start'] = rhino_start['cells'].name
        results.loc[move, 'Rhino Heading'] = rhino_heading['cells'].name
        results.loc[move, 'Rhino Toward'] = rhino_toward['cells'].name

        ranger_curr = ranger.moving(ranger_curr, ranger_cells, ranger_toward)
        ranger_add = pd.Series([ranger_traj, 2, 'Ranger', move],
                               index=['Trajectory', 'PlayerID', 'Player', 'Moves'],
                               name=ranger_curr['cells'].name)
        ranger_add = ranger_curr['cells'].append(ranger_add)
        ranger_path = ranger_path.append(ranger_add)
        ranger_view = perim_grid(ranger_curr, ranger.step_size, ranger.view_size,
                                 'view', ranger_cells)
        if ranger_curr['cells'].name == ranger_toward['cells'].name:
            if ranger_curr['cells'].name == ranger_heading['cells'].name:
                results.loc[move, 'Ranger Reach Heading'] = 1
                ranger_toward = ranger_start
            if ranger_curr['cells'].name == ranger_start['cells'].name:
                results.loc[move, 'Ranger Reach Start'] = 1
                ranger_toward = ranger_heading
            ranger_traj += 1
        results.loc[move, 'Ranger Current'] = ranger_curr['cells'].name
        results.loc[move, 'Ranger Start'] = ranger_start['cells'].name
        results.loc[move, 'Ranger Heading'] = ranger_heading['cells'].name
        results.loc[move, 'Ranger Toward'] = ranger_toward['cells'].name

        poacher_curr = poacher.moving(poacher_curr, poacher_cells, poacher_toward)
        poacher_add = pd.Series([poacher_traj, 3, 'Poacher', move],
                                index=['Trajectory', 'PlayerID', 'Player', 'Moves'],
                                name=poacher_curr['cells'].name)
        poacher_add = poacher_curr['cells'].append(poacher_add)
        poacher_path = poacher_path.append(poacher_add)
        poacher_view = perim_grid(poacher_curr, poacher.step_size, poacher.view_size,
                                  'view', poacher_cells)
        poacher_moves_traj = []
        if poacher_toward['cells'].name == poacher_start['cells'].name:
            poacher_moves_traj = list(poacher_path.loc[poacher_path['Trajectory'] == (poacher_traj - 1), 'Moves'])
            poacher_moves_traj.extend(list(poacher_path.loc[poacher_path['Trajectory'] == poacher_traj, 'Moves']))
            if 0 in poacher_moves_traj:
                poacher_moves_traj.remove(0)
        if poacher_curr['cells'].name == poacher_toward['cells'].name:
            if poacher_curr['cells'].name == poacher_heading['cells'].name:
                results.loc[move, 'Poacher Reach Heading'] = 1
                poacher_toward = poacher_start
            if poacher_curr['cells'].name == poacher_start['cells'].name:
                results.loc[move, 'Poacher Reach Start'] = 1
                poacher_toward = poacher_heading
            poacher_traj += 1
        if rhino_curr['cells'].name in poacher_view['cells'].index:
            poaches_traj = sum(results.loc[poacher_moves_traj, 'Poach Events'])
            if poaches_traj == 0:
                results.loc[move, 'Poach Cell'] = rhino_curr['cells'].name
                results.loc[move, 'Poach Events'] = 1
                if poacher_toward['cells'].name == poacher_heading['cells'].name:
                    poacher_toward = poacher_start
                    poacher_traj += 1
                if poacher.movement == 'structured':
                    poacher_heading = poacher_curr
        if poacher_curr['cells'].name == poacher_start['cells'].name:
            poaches_traj = sum(results.loc[poacher_moves_traj, 'Poach Events'])
            if poaches_traj > 0:
                results.loc[move, 'Leave After'] = 1
            else:
                results.loc[move, 'Leave Before'] = 1
        if poacher_curr['cells'].name in ranger_view['cells'].index:
            results.loc[move, 'Capture Cell'] = poacher_curr['cells'].name
            results.loc[move, 'Capture Events'] = 1
            poaches_traj = sum(results.loc[poacher_moves_traj, 'Poach Events'])
            if poaches_traj > 0:
                results.loc[move, 'Capture After'] = 1
            else:
                results.loc[move, 'Capture Before'] = 1
            if poacher.movement == 'structured':
                poacher_start = poacher.start(poacher_cells, grid)
            poacher_curr = poacher_start
            poacher_toward = poacher_heading
            poacher_traj += 1
        results.loc[move, 'Poacher Start'] = poacher_start['cells'].name
        results.loc[move, 'Poacher Current'] = poacher_curr['cells'].name
        results.loc[move, 'Poacher Heading'] = poacher_heading['cells'].name
        results.loc[move, 'Poacher Toward'] = poacher_toward['cells'].name

    sum_cols = ['Rhino Reach Start', 'Rhino Reach Heading', 'Ranger Reach Start', 'Ranger Reach Heading',
                'Poacher Reach Start', 'Poacher Reach Heading', 'Poach Events', 'Capture Events',
                'Leave Before', 'Leave After', 'Catch Before', 'Catch After']
    res_sum = (results.loc[:, sum_cols]).sum(axis=0)
    def_crs = grid['square'].crs
    paths = pd.concat([rhino_path, ranger_path, poacher_path])
    paths = paths.rename(columns={'Time': 't', 'Centroids': 'geometry'})
    paths = gpd.GeoDataFrame(paths, crs=def_crs, geometry='geometry')
    traj_coll = mpd.TrajectoryCollection(paths.set_index('t'), 'PlayerID')

    return {'results': results, 'totals': res_sum, 'paths': paths, 'trajectories': traj_coll}


def simulate(grid, rhino, rhino_cells, ranger, ranger_cells, poacher, poacher_cells, end_moves=100,
             set_seed=None, months=1000, games_pm=30, same_start=False):

    if set_seed is not None:
        random.seed(set_seed)

    ind_dat = [range(1, months + 1), range(1, games_pm + 1)]
    ind = pd.MultiIndex.from_product(ind_dat, names=['Month', 'Game'])
    columns = ['Rhino Reach Start', 'Rhino Reach Heading', 'Ranger Reach Start', 'Ranger Reach Heading',
               'Poacher Reach Start', 'Poacher Reach Heading', 'Poach Events', 'Capture Events',
               'Leave Before', 'Leave After', 'Catch Before', 'Catch After']
    results = pd.DataFrame(0.0, index=ind, columns=columns)
    trajectories = []

    # month iterations
    for month in range(1, months + 1):

        if same_start:
            rhino.start_cell = rhino.start(rhino_cells)
            ranger.start_cell = ranger.start(ranger_cells)
            poacher.start_cell = poacher.start(poacher_cells, grid)

        if ranger.movement == 'game' or poacher.movement == 'game':
            game_sol = pure_stcklbrg(ranger, ranger_cells, poacher, poacher_cells, 'p1')
            if ranger.movement == 'game':
                ranger.heading_cell = game_sol['ranger optimal']
            if poacher.movement == 'game':
                poacher.heading_cell = game_sol['poacher optimal']

        # game iterations
        for game in range(1, games_pm):
            single_game = one_game(grid, rhino, rhino_cells, ranger, ranger_cells,
                                   poacher, poacher_cells, end_moves)
            results.loc[month, game] = single_game['totals']
            trajectories.append(single_game['trajectories'])

    results.applymap("{0:.3f}".format)
    sum_per_month = results.groupby(by='Month').sum()
    ave_per_month = sum_per_month.mean()
    return {'all': results, 'months': sum_per_month, 'ave': ave_per_month, 'trajectories': trajectories}
