# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:07:07 2014

@author: muahah

For performance
"""


import pdb
from ..core import Points, Profile, ScalarField, VectorField, make_unit,\
    ARRAYTYPES, NUMBERTYPES, STRINGTYPES, TemporalScalarFields,\
    TemporalVectorFields
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
from scipy import optimize
import warnings
import sets


def velocityfield_to_vf(velocityfield):
    """
    Create a VF object from a VelocityField object.
    """
    # extracting data
    vx = velocityfield.get_comp('Vx', raw=True).data
    vy = velocityfield.get_comp('Vy', raw=True).data
    ind_x, ind_y = velocityfield.get_axes()
    mask = velocityfield.get_comp('mask', raw=True)
    theta = velocityfield.get_comp('theta', raw=True)
    time = velocityfield.get_comp('time')
    # TODO : add the following when fill will be optimized
    #THETA.fill()
    theta = theta.data
    # using VF methods to get cp position
    vf = VF(vx, vy, ind_x, ind_y, mask, theta, time)
    return vf


class VF(object):

    def __init__(self, vx, vy, axe_x, axe_y, mask, theta, time):
        self.vx = np.array(vx)
        self.vy = np.array(vy)
        self.mask = np.array(mask)
        self.theta = np.array(theta)
        self.time = time
        self._min_win_size = 3
        self.shape = self.vx.shape
        self.axe_x = axe_x
        self.axe_y = axe_y

    def export_to_velocityfield(self):
        """
        Return the field as VelocityField object.
        """
        tmp_vf = VelocityField()
        tmp_vf.import_from_arrays(self.axe_x, self.axe_y, self.vx, self.vy,
                                  self.time)
        return tmp_vf

    def get_cp_position(self):
        """
        Return critical points positions and their associated PBI.
        (PBI : Poincarre_Bendixson indice)
        Positions are returned in axis unities (axe_x and axe_y) and are
        always at the center of 4 points (maximum accuracy of this algorithm
        is limited by the field spatial resolution).


        Returns
        -------
        pos : 2xN array
            position (x, y) of the detected critical points.
        pbis : 1xN array
            PBI (1 indicate a node, -1 a saddle point)
        """
        delta_x = self.axe_x[1] - self.axe_x[0]
        delta_y = self.axe_y[1] - self.axe_y[0]
        positions = []
        pbis = []
        # first splitting
        grid_x, grid_y = self._find_cut_positions()
        # If there is nothing or we really don't have luck...
        # just a split in the middle to see
        if grid_x is None:
            len_x = self.shape[1]
            len_y = self.shape[0]
            pool = self._split_the_field([0, np.round(len_x/2.), len_x],
                                         [0, np.round(len_y/2.), len_y])
        else:
            pool = self._split_the_field(grid_x, grid_y)
        # loop on the pool (funny no ?)
        while True:
            # if the pool is empty we have finish !
            if len(pool) == 0:
                break
            # get a field
            tmp_vf = pool[0]
            nmb_struct = tmp_vf._check_struct_number()
            # check if there is something in it
            if nmb_struct == (0, 0):
                pass
            # if there is only one critical point and the field is as
            # small as possible, we store the cp position, the end !
            elif nmb_struct == (1, 1)\
                    and np.all(self.shape[0] < (2*self._min_win_size,
                                                2*self._min_win_size)):
                cp_pos = tmp_vf._get_poi_position()
                positions.append((tmp_vf.axe_x[cp_pos[0]] + delta_x/2.,
                                  tmp_vf.axe_y[cp_pos[1]] + delta_y/2.))
                pbis.append(tmp_vf.pbi_x[-1])
            # if there is only one critical point in it, we cut around
            # this point
            elif nmb_struct == (1, 1):
                pass
            # Else, we split again the field
            else:
                tmp_grid_x, tmp_grid_y = tmp_vf._find_cut_positions()
                # if there is possible cutting, we do it
                if len(tmp_grid_x) != 2 or len(tmp_grid_y) != 2:
                    tmp_pool = tmp_vf._split_the_field(tmp_grid_x, tmp_grid_y)
                    pool = np.append(pool, tmp_pool[:])
                #else we just delete the field
            pool = np.delete(pool, 0)
        return positions, pbis

    def get_pbi(self, direction):
        """
        Return the PBI along the given axe.
        """
        theta = self.theta.copy()
        if direction == 2:
            theta = np.rot90(theta, 3)
        pbi = np.zeros(theta.shape[1])
        for i in np.arange(theta.shape[1]):
            # getting and concatening profiles
            thetas_border = np.concatenate((theta[::-1, 0],
                                            theta[0, 0:i],
                                            theta[:, i],
                                            theta[-1, i:0:-1]),
                                           axis=0)
            delta_thetas = np.concatenate((thetas_border[1::]
                                           - thetas_border[0:-1],
                                           thetas_border[0:1]
                                           - thetas_border[-1::]), axis=0)
            # particular points treatment
            delta_thetas[delta_thetas > np.pi] -= 2*np.pi
            delta_thetas[delta_thetas < -np.pi] += 2*np.pi
            # Stockage
            pbi[i] = np.sum(delta_thetas)/(2.*np.pi)
        if direction == 2:
            pbi = pbi[::-1]
        return np.array(np.round(pbi))

    def get_field_around_pt(self, xy, nmb_lc):
        """
        Return a field around the point given by x and y.
        nm_lc give the number of line and column to take around the point.
        """
        # getting data
        delta_x = self.axe_x[1] - self.axe_x[0]
        delta_y = self.axe_y[1] - self.axe_y[0]
        nmb_lc_2 = np.ceil(nmb_lc/2.)
        # getting bornes
        x_min = xy[0] - nmb_lc_2*delta_x
        x_max = xy[0] + nmb_lc_2*delta_x
        y_min = xy[1] - nmb_lc_2*delta_y
        y_max = xy[1] + nmb_lc_2*delta_y
        # getting masks
        mask_x = np.logical_and(self.axe_x > x_min, self.axe_x < x_max)
        mask_y = np.logical_and(self.axe_y > y_min, self.axe_y < y_max)
        # trimming original field
        vx = self.vx[mask_y, :]
        vx = vx[:, mask_x]
        vy = self.vy[mask_y, :]
        vy = vy[:, mask_x]
        axe_x = self.axe_x[mask_x]
        axe_y = self.axe_y[mask_y]
        mask = self.mask[mask_y, :]
        mask = mask[:, mask_x]
        theta = self.theta[mask_y, :]
        mask = theta[:, mask_x]
        time = self.time
        if len(axe_x) == 0 or len(axe_y) == 0:
            raise ValueError()
        return VF(vx, vy, axe_x, axe_y, mask, theta, time)

    def _find_cut_positions(self):
        """
        Return the position along x and y where the field has to be cut to
        isolate critical points.
        Return '(None, None)' if there is no possible cut position.
        """
        self._calc_pbi()
        # Getting point of interest position
        poi_x = self.pbi_x[1::] - self.pbi_x[0:-1]
        poi_y = self.pbi_y[1::] - self.pbi_y[0:-1]
        ind_x = np.where(poi_x != 0)[0]
        #ind_x = np.concatenate(([0], ind_x, [len(self.pbi_x) - 1]))
        ind_y = np.where(poi_y != 0)[0]
        #ind_y = np.concatenate(([0], ind_y, [len(self.pbi_y) - 1]))
        # If there is only one or none distinct structure
        if ind_x.shape[0] <= 1 and ind_y.shape[0] <= 1:
            return None, None
        # finding all possible cutting positions
        dist_x = np.abs(ind_x[1::] - ind_x[0:-1])
        dist_y = np.abs(ind_y[1::] - ind_y[0:-1])
        cut_pos_x = ind_x[0:-1] + np.ceil(dist_x/2)
        cut_pos_y = ind_y[0:-1] + np.ceil(dist_y/2)
        # eliminating cutting creating too small fields
        cut_pos_x = cut_pos_x[dist_x > self._min_win_size]
        cut_pos_y = cut_pos_y[dist_y > self._min_win_size]
        # adding trimming and the border (allow to get good result even if we
        # have to cut on a critical point line or column.
        if len(ind_x) == 0:
            trim_x_1 = []
        elif ind_x[0] > self._min_win_size:
            trim_x_1 = [ind_x[0] - self._min_win_size + 1]
        else:
            trim_x_1 = []
        if len(ind_x) == 0:
            trim_x_2 = []
        elif len(self.pbi_x) - ind_x[-1] > self._min_win_size:
            trim_x_2 = [ind_x[-1] + self._min_win_size - 1]
        else:
            trim_x_2 = []
        if len(ind_y) == 0:
            trim_y_1 = []
        elif ind_y[0] > self._min_win_size:
            trim_y_1 = [ind_y[0] - self._min_win_size + 1]
        else:
            trim_y_1 = []
        if len(ind_y) == 0:
            trim_y_2 = []
        elif len(self.pbi_y) - ind_y[-1] > self._min_win_size:
            trim_y_2 = [ind_y[-1] + self._min_win_size - 1]
        else:
            trim_y_2 = []
        # adding borders
        grid_x = np.concatenate(([0], trim_x_1, cut_pos_x + 1, trim_x_2,
                                 [len(self.pbi_x)]))
        grid_y = np.concatenate(([0], trim_y_1, cut_pos_y + 1, trim_y_2,
                                 [len(self.pbi_y)]))
        return grid_x, grid_y

    def _split_the_field(self, grid_x, grid_y):
        """
        Return a set of fields, resulting of cutting the field at the
        positions given by grid_x and grid_y.
        Resulting fields are a little bigger than if we do a basic cutting,
        in order to not lose cp.
        """
        fields = []
        for i in np.arange(len(grid_x) - 1):
            for j in np.arange(len(grid_y) - 1):
                if grid_x[i] == 0:
                    slic_x = slice(grid_x[i], grid_x[i + 1] + 1)
                else:
                    slic_x = slice(grid_x[i] - 1, grid_x[i + 1] + 1)
                if grid_y[j] == 0:
                    slic_y = slice(grid_y[j], grid_y[j + 1] + 1)
                else:
                    slic_y = slice(grid_y[j] - 1, grid_y[j + 1] + 1)
                vx_tmp = self.vx[slic_y, slic_x]
                vy_tmp = self.vy[slic_y, slic_x]
                mask_tmp = self.mask[slic_y, slic_x]
                theta_tmp = self.theta[slic_y, slic_x]
                time_tmp = self.time
                axe_x_tmp = self.axe_x[slic_x]
                axe_y_tmp = self.axe_y[slic_y]
                vf_tmp = VF(vx_tmp, vy_tmp, axe_x_tmp, axe_y_tmp,
                            mask_tmp, theta_tmp, time_tmp)
                fields.append(vf_tmp)
        return fields

    def _calc_pbi(self):
        """
        Compute the pbi along the two axis (if not already computed,
        and store them.
        """
        try:
            self.pbi_x
            self.pbi_y
        except AttributeError:
            self.pbi_x = self.get_pbi(direction=1)
            self.pbi_y = self.get_pbi(direction=2)

    def _get_poi_position(self):
        """
        On a field with only one structure detected by PBI, return the
        position of this structure (position is given in indice).
        """
        self._calc_pbi()
        poi_x = self.pbi_x[1::] - self.pbi_x[0:-1]
        poi_y = self.pbi_y[1::] - self.pbi_y[0:-1]
        ind_x = np.where(poi_x != 0)[0]
        ind_y = np.where(poi_y != 0)[0]
        if len(ind_x) > 1 or len(ind_y) > 1:
            raise StandardError()
        if len(ind_x) == 0 and len(ind_y) == 0:
            raise StandardError("empty field")
        return ind_x[0], ind_y[0]

    def _check_struct_number(self):
        """
        Return the possible number of structures in the field along each axis.
        """
        self._calc_pbi()
        # finding points of interest (where pbi change)
        poi_x = self.pbi_x[1::] - self.pbi_x[0:-1]
        poi_y = self.pbi_y[1::] - self.pbi_y[0:-1]
        # computing number of structures
        num_x = len(poi_x[poi_x != 0])
        num_y = len(poi_y[poi_y != 0])
        return num_x, num_y


### Critical points detection algorithm ###
def get_cp_traj(TVFS, epsilon=None, kind='crit'):
    """
    For a set of velocity field (TemporalVectorFields object), return the
    trajectory of critical points.
    If the number of points returned is low, you should smooth or filter your
    field (POD filtering for example).

    Parameters
    ----------
    TVFS : TemporalVectorFields object
        .
    epsilon : float, optional
        Maximum length between two consecutive points in trajectory.
        (default is Inf), extremely usefull to put a correct value here.
    kind : string, optional
        If 'pbi', return cp position given by PBI algorithm.
        If 'crit' (default), return cp position given by PBI + criterion.
        (more accurate, but slower).

    Returns
    -------
    focus_traj : tuple of Points objects
        Rotative focus trajectories
    focus_c_traj : tuple of Points objects
        Contrarotative focus trajectories
    nodes_i_traj : tuple of Points objects
        In nodes trajectories
    nodes_o_traj : tuple of Points objects
        Out nodes trajectories
    saddle_pts : tuple of Points objects
        Saddle points trajectories
    pbi_p : tuple of Points objects
        Points get by PBI algorithm (with pbi=1)
    pbi_m : tuple of Points objects
        Points get by PBI algorithm (with pbi=-1)
    """
    # check parameters coherence
    if not isinstance(TVFS, TemporalVectorFields):
        raise TypeError("'TVFS' must be a TemporalVectorFields")
    if epsilon is not None:
        if not isinstance(epsilon, NUMBERTYPES):
            raise TypeError("'epsilon' must be a positive real")
        if epsilon < 0:
            raise ValueError("'epsilon' must be a positive real")
    if not isinstance(kind, STRINGTYPES):
        raise TypeError("'kind' must be a string")
    # TODO : Add warning when masked values are present (or not ?)
    # create storage arrays
    focus = []
    focus_c = []
    nodes_i = []
    nodes_o = []
    saddles = []
    cp_pbi_p = []
    cp_pbi_m = []
    # getting first critical points for all fields
    for field in TVFS.fields:
        if kind == 'crit':
            foc, foc_c, nod_i, nod_o, sadd, pbi = get_cp_crit(field)
            pbi_p, pbi_m = [], []
            if len(pbi) != 0:
                for pbi_pt in pbi:
                    if pbi_pt.pbi == 1:
                        pbi_p.append(pbi_pt)
                    else:
                        pbi_m.append(pbi_pt)
        elif kind == 'pbi':
            pos, pbis = get_cp_pbi(field)
            foc, foc_c, nod_i, nod_o, sadd = [], [], [], [], []
            pbi_p, pbi_m = [], []
            for i, pt in enumerate(pos):
                tmp_pt = Points([pt], [field.time])
                tmp_pt.pbi = pbis[i]
                if pbis[i] == 1:
                    pbi_p.append(tmp_pt)
                else:
                    pbi_m.append(tmp_pt)

        else:
            raise ValueError()
        # Concatenate result of each field in bigger entities
        if len(foc) != 0:
            focus += foc
        if len(foc_c) != 0:
            focus_c += foc_c
        if len(nod_i) != 0:
            nodes_i += nod_i
        if len(nod_o) != 0:
            nodes_o += nod_o
        if len(sadd) != 0:
            saddles += sadd
        if len(pbi_p) != 0:
            cp_pbi_p += pbi_p
        if len(pbi_m) != 0:
            cp_pbi_m += pbi_m
    # getting times
    times = TVFS.get_comp('time')
    # getting critical points trajectory for eachkind of point
    if len(focus) != 0:
        focus_traj = get_cp_time_evolution(focus, times, epsilon)
    else:
        focus_traj = []
    if len(focus_c) != 0:
        focus_c_traj = get_cp_time_evolution(focus_c, times, epsilon)
    else:
        focus_c_traj = []
    if len(nodes_i) != 0:
        nodes_i_traj = get_cp_time_evolution(nodes_i, times, epsilon)
    else:
        nodes_i_traj = []
    if len(nodes_o) != 0:
        nodes_o_traj = get_cp_time_evolution(nodes_o, times, epsilon)
    else:
        nodes_o_traj = []
    if len(saddles) != 0:
        saddles_traj = get_cp_time_evolution(saddles, times, epsilon)
    else:
        saddles_traj = []
    if len(cp_pbi_p) != 0:
        cp_pbip_traj = get_cp_time_evolution(cp_pbi_p, times, epsilon)
    else:
        cp_pbip_traj = []
    if len(cp_pbi_m) != 0:
        cp_pbim_traj = get_cp_time_evolution(cp_pbi_m, times, epsilon)
    else:
        cp_pbim_traj = []
    # returning result
    return focus_traj, focus_c_traj, nodes_i_traj, nodes_o_traj, saddles_traj,\
        cp_pbim_traj, cp_pbip_traj


def get_cp_time_evolution(points, times=None, epsilon=None):
    """
    Compute the temporal evolution of each vortex centers from a set of points
    at different times. (Points objects must each contain only one point and
    time must be specified in 'v' argument of points).

    Parameters:
    -----------
    points : tuple of Points objects.
        .
    times : array of numbers
        Times. If 'None' (default), only times represented by at least one
        point are taken into account (can create wring link between points).
    epsilon : number, optional
        Maximal distance between two successive points.
        default value is Inf.
    """
    if len(points) == 0:
        return None
    if not isinstance(points, ARRAYTYPES):
        raise TypeError("'points' must be an array of Points objects")
    if times is not None:
        if not isinstance(times, ARRAYTYPES):
            raise TypeError("'times' must be an array of numbers")
    if not isinstance(points[0], Points):
        raise TypeError("'points' must be an array of Points objects")

    # local class to store the point field
    class PointField(object):
        """
        Class representing an orthogonal set of points, defined by a position
        and a time.
        """
        def __init__(self, pts_tupl, times):
            if not isinstance(pts_tupl, ARRAYTYPES):
                raise TypeError("'pts' must be a tuple of Point objects")
            for pt in pts_tupl:
                if not isinstance(pt, Points):
                    raise TypeError("'pts' must be a tuple of Point objects")
                if not len(pt) == len(pt.v):
                    raise StandardError("v has not the same dimension as xy")
            # if some Points objects contains more than one point, we decompose
            # them
            for i in np.arange(len(pts_tupl)-1, -1, -1):
                if len(pts_tupl[i]) != 1:
                    pts_tupl[i:i+1] = pts_tupl[i].decompose()
            self.points = []
            # possible times determination
            if times is None:
                times = []
                for pt in pts_tupl:
                    times.append(pt.v[0])
                times = list(sets.Set(times))
                times.sort()
                self.times = times
            # Sorting points by times
            for time in times:
                self.times = times
                tmp_points = []
                for pt in pts_tupl:
                    if pt.v[0] == time:
                        tmp_points.append(Point(pt.xy[0, 0], pt.xy[0, 1],
                                                pt.v[0]))
                self.points.append(tmp_points)
            self.unit_x = pts_tupl[0].unit_x
            self.unit_y = pts_tupl[0].unit_y
            self.unit_v = pts_tupl[0].unit_v
            self.time_step = np.size(self.points, 0)

        def make_point_useless(self, i, j):
            """
            Make a point of the field useless (None).
            """
            if not isinstance(i, int) or not isinstance(j, int):
                raise TypeError()
            self.points[i][j] = None

        def get_points_at_time(self, time):
            """
            Return all the points for a given time.
            """
            if not isinstance(time, int):
                raise TypeError()
            return self.points[time]

    # local class line to store vortex center evolution line
    class Line(object):
        """
        Class representing a line, defined by a set of ordened points.
        """

        def __init__(self, epsilon):
            self.points = []
            self.epsilon = epsilon

        def add_point(self, pts):
            """
            Add a new point to the line.
            """
            if not isinstance(pts, Point):
                raise TypeError("'pts' must be a Point object")
            self.points.append(pts)

        def remove_point(self, ind):
            """
            Remove the designated point of the line.
            """
            if not isinstance(ind, int):
                raise TypeError("'ind' must be an integer")
            if ind < 0 or ind > len(self):
                raise ValueError("'ind' is out of range")
            self.points.pop(ind)

        def choose_starting_point(self, PF):
            """
            Choose in the given field, a new new starting point for a line.
            """
            if not isinstance(PF, PointField):
                raise TypeError()
            start_i = None
            start_j = None
            for i in np.arange(PF.time_step):
                for j in np.arange(len(PF.points[i])):
                    if PF.points[i][j] is not None:
                        start_i = i
                        start_j = j
                        break
                if start_i is not None:
                    break
            if start_i is None:
                return None, None
            self.points.append(PF.points[start_i][start_j])
            PF.make_point_useless(start_i, start_j)
            return start_i, start_j

        def choose_next_point(self, ext_pt_tupl):
            """
            Get the next point of the line (closest one).
            """
            if not isinstance(ext_pt_tupl, ARRAYTYPES):
                raise TypeError()
            if len(ext_pt_tupl) == 0:
                return None
            if ext_pt_tupl[0] is not None:
                if not isinstance(ext_pt_tupl[0], Point):
                    raise TypeError()
            if len(self.points) == 0:
                raise Warning("there is no starting points")
            # nearest point choice
            dist = []
            for ext_pt in ext_pt_tupl:
                dist.append(distance(ext_pt, self.points[-1]))
            ind_min = np.argmin(dist)
            # if all the points are 'useless', the line should stop
            if dist[ind_min] == 1e99:
                return None
            # else, we check that the point is not too far (using epsilon)
            if epsilon is not None:
                if dist[ind_min] > self.epsilon**2:
                    return None
            # and if not we add the new point
            self.points.append(ext_pt_tupl[ind_min])
            return ind_min

        def export_to_Points(self, PF):
            """
            Export the current line to a Points object.
            """
            xy = []
            v = []
            for pt in self.points:
                xy.append([pt.x, pt.y])
                v.append(pt.v)
            points = Points(xy, v, PF.unit_x, PF.unit_y, PF.unit_v)
            return points

    # local class point to store one point
    class Point(object):
        """
        Class representing a point with a value on it.
        """
        def __init__(self, x, y, v):
            self.x = x
            self.y = y
            self.v = v

    # local function distance to compute distance between Point objects
    def distance(pts1, pts2):
        """
        Compute the distance between two points.
        """
        if pts2 is None or pts1 is None:
            return 1e99
        else:
            return (pts2.x - pts1.x)**2 + (pts2.y - pts1.y)**2
    # Getting the vortex centers trajectory
    PF = PointField(points, times)
    if len(PF.points) == 0:
        return []
    points_f = []
    while True:
        line = Line(epsilon)
        start_i, start_j = line.choose_starting_point(PF)
        if start_i is None:
            break
        for i in np.arange(start_i + 1, PF.time_step):
            j = line.choose_next_point(PF.get_points_at_time(i))
            if j is None:
                break
            PF.make_point_useless(i, j)
        points_f.append(line.export_to_Points(PF))
    return points_f


def get_cp_pbi(velocityfield):
    """
    For a VelocityField object, return the critical points positions and their
    PBI (Poincarre Bendixson indice)

    Parameters
    ----------
    velocityfield : a VelocityField object.
        .
    Returns
    -------
    pos : 2xN array
        position (x, y) of the detected critical points.
    pbis : 1xN array
        PBI (1 indicate a node, -1 a saddle point)
    """
    # checking parameters coherence
    if not isinstance(velocityfield, VelocityField):
        raise TypeError("'velocityfield' must be a VelocityField")
    # using VF methods to get cp position
    field = velocityfield_to_vf(velocityfield)
    return field.get_cp_position()


def get_cp_crit(velocityfield):
    """
    For a VelocityField object, return the position of critical points.
    This algorithm use the PBI algorithm, then a method based on criterion to
    give more accurate results.

    Parameters
    ----------
    velocityfield : VelocityField object
        .

    Returns
    -------
    focus, focus_c, nodes_i, nodes_o, saddles : tuple of points
        Found points of each type.
    cp_pbi : tuple of points
        Points too close to a wall to use criterion method.
    """
    if not isinstance(velocityfield, VelocityField):
        raise TypeError("'VF' must be a VelocityField")
    ### Getting pbi cp position and fields around ###
    VF_field = velocityfield_to_vf(velocityfield)
    cp_positions, pbis = VF_field.get_cp_position()
    # creating velocityfields around critical points
    # and transforming into VelocityField objects
    VF_tupl = []
    cp_pbi = []
    for i, cp_pos in enumerate(cp_positions):
        tmp_vf = VF_field.get_field_around_pt(cp_pos, 5)
        tmp_vf = tmp_vf.export_to_velocityfield()
        # trating small fields
        axe_x, axe_y = tmp_vf.get_axes()
        if len(axe_x) < 6 or len(axe_y) < 6:
            pt = Points([cp_pos], [VF_field.time])
            pt.pbi = pbis[i]
            cp_pbi.append(pt)
        else:
            tmp_vf.PBI = pbis[i]
            VF_tupl.append(tmp_vf)
    ### Sorting by critical points type ###
    VF_focus = []
    VF_nodes = []
    VF_saddle = []
    for VF in VF_tupl:
        # node or focus
        if VF.PBI == 1:
            # checking if node or focus
            VF.gamma1 = get_gamma(VF, radius=1.9, ind=True)
            VF.kappa1 = get_kappa(VF, radius=1.9, ind=True)
            max_gam = np.max([abs(VF.gamma1.get_max()),
                             abs(VF.gamma1.get_min())])
            max_kap = np.max([abs(VF.kappa1.get_max()),
                             abs(VF.kappa1.get_min())])
            if max_gam > max_kap:
                VF_focus.append(VF)
            else:
                VF_nodes.append(VF)
        # saddle point
        elif VF.PBI == -1:
            VF_saddle.append(VF)
    ### Computing saddle points positions ###
    saddles = []
    if len(VF_saddle) != 0:
        for VF in VF_saddle:
            tmp_sigma = get_sigma(VF, 1.9, ind=True)
            pts = _min_detection(tmp_sigma)
            if pts is not None:
                if pts.xy[0][0] is not None and len(pts) == 1:
                    pts.v = [VF.time]
                    pts.unit_v = VF.unit_time
                    saddles.append(pts)
    ### Computing focus positions (rotatives and contrarotatives) ###
    focus = []
    focus_c = []
    if len(VF_focus) != 0:
        for VF in VF_focus:
            tmp_gam = VF.gamma1
            min_gam = tmp_gam.get_min()
            max_gam = tmp_gam.get_max()
            # rotative vortex
            if abs(max_gam) > abs(min_gam):
                pts = _min_detection(-1.*tmp_gam)
                if pts is not None:
                    if pts.xy[0][0] is not None and len(pts) == 1:
                        pts.v = [VF.time]
                        pts.unit_v = VF.unit_time
                        focus.append(pts)
            # contrarotative vortex
            else:
                pts = _min_detection(tmp_gam)
                if pts is not None:
                    if pts.xy[0][0] is not None and len(pts) == 1:
                        pts.v = [VF.time]
                        pts.unit_v = VF.unit_time
                        focus_c.append(pts)
    ### Computing nodes points positions (in or out) ###
    nodes_i = []
    nodes_o = []
    if len(VF_nodes) != 0:
        for VF in VF_nodes:
            tmp_kap = VF.kappa1
            min_kap = tmp_kap.get_min()
            max_kap = tmp_kap.get_max()
            # out nodes
            if abs(max_kap) > abs(min_kap):
                pts = _min_detection(-1.*tmp_kap)
                if pts is not None:
                    if pts.xy[0][0] is not None and len(pts) == 1:
                        pts.v = [VF.time]
                        pts.unit_v = VF.unit_time
                        nodes_o.append(pts)
            # in nodes
            else:
                pts = _min_detection(tmp_kap)
                if pts is not None:
                    if pts.xy[0][0] is not None and len(pts) == 1:
                        pts.v = [VF.time]
                        pts.unit_v = VF.unit_time
                        nodes_i.append(pts)
    return focus, focus_c, nodes_i, nodes_o, saddles, cp_pbi


def _min_detection(SF):
    """
    Only for use in 'get_cp_crit'.
    """
    # interpolation on the field
    axe_x, axe_y = SF.get_axes()
    values = SF.get_comp('values', raw=True).data
    try:
        interp = RectBivariateSpline(axe_y, axe_x, values, s=0, ky=2, kx=2)
    except:
        pdb.set_trace()
    # extended field (resolution x100)
    x = np.linspace(axe_x[0], axe_x[-1], 100)
    y = np.linspace(axe_y[0], axe_y[-1], 100)
    values = interp(y, x)
    ind_min = np.argmin(values)
    ind_y, ind_x = np.unravel_index(ind_min, values.shape)
    pos = (x[ind_x], y[ind_y])
    return Points([pos])


def _gaussian_fit(SF):
    """
    Only for use in 'get_cp_crit'.
    """
    # gaussian fitting
    def gaussian(height, center_x, center_y, width_x, width_y):
        """Returns a gaussian function with the given parameters"""
        width_x = float(width_x)
        width_y = float(width_y)
        return lambda x, y: height*np.exp(
            -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

    def moments(data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution by calculating its
        moments """
        total = data.sum()
        X, Y = np.indices(data.shape)
        x = (X*data).sum()/total
        y = (Y*data).sum()/total
        col = data[:, int(y)]
        width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
        row = data[int(x), :]
        width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
        height = data.max()
        return height, x, y, width_x, width_y

    def fitgaussian(data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution found by a fit"""
        params = moments(data)
        errorfunction = lambda p: np.ravel(gaussian(*p)
                                           (*np.indices(data.shape)) -
                                           data)
        p, success = optimize.leastsq(errorfunction, params)
        return p
    axe_x, axe_y = SF.get_axes()
    values = SF.get_comp('values')
    params = fitgaussian(values)
    delta_x = axe_x[1] - axe_x[0]
    delta_y = axe_y[1] - axe_y[0]
    x = SF.axe_x[0] + delta_x*params[1]
    y = SF.axe_y[0] + delta_y*params[2]
    return Points([(x, y)])


### Separation point detection algorithm ###
def get_separation_position(obj, wall_direction, wall_position,
                            interval=None):
    """
    Compute and return the separation points position.
    Separation points position is computed by searching zero streamwise
    velocities on surrounding field lines (4 of them) and by interpolating at
    the wanted 'wall_position'.
    'interval' must include separation points on the 4 nearest field line.

    Parameters
    ----------
    obj : ScalarField, VectorField, VelocityField or TemporalVelocityField
        If 'VectorField' or 'VelocityField', wall_direction is used to
        determine the interesting velocity component.
    wall_direction : integer
        1 for a wall at a given value of x,
        2 for a wall at a given value of y.
    wall_position : number
        Position of the wall.
    interval : 2x1 array of numbers, optional
        Optional interval in which search for the detachment points.

    """
    # checking parameters coherence
    if not isinstance(obj, (ScalarField, VectorField, VelocityField,
                            TemporalVectorFields)):
        raise TypeError("Unknown type for 'obj'")
    if not isinstance(wall_direction, NUMBERTYPES):
        raise TypeError("'wall_direction' must be a number")
    if wall_direction != 1 and wall_direction != 2:
        raise ValueError("'wall_direction' must be 1 or 2")
    if not isinstance(wall_position, NUMBERTYPES):
        raise ValueError("'wall_position' must be a number")
    axe_x, axe_y = obj.get_axes()
    if interval is None:
        if wall_direction == 2:
            interval = [np.min(axe_x), np.max(axe_x)]
        else:
            interval = [np.min(axe_y), np.max(axe_y)]
    if not isinstance(interval, ARRAYTYPES):
        raise TypeError("'interval' must be a array")
    # checking 'obj' type
    if isinstance(obj, ScalarField):
        V = obj.get_comp('values', raw=True)
        if wall_direction == 1:
            axe = axe_x
        else:
            axe = axe_y
    elif isinstance(obj, VectorField):
        if wall_direction == 1:
            V = obj.get_comp('Vy', raw=True)
            axe = axe_x
        else:
            V = obj.get_comp('Vx', raw=True)
            axe = axe_y
    elif isinstance(obj, VelocityField):
        if wall_direction == 1:
            V = obj.get_comp('Vy', raw=True)
            axe = axe_x
        else:
            V = obj.get_comp('Vx', raw=True)
            axe = axe_y
    elif isinstance(obj, TemporalVectorFields):
        pts = []
        times = obj.get_comp('time')
        if wall_direction == 1:
            _, unit_axe = obj.get_axe_units()
        else:
            unit_axe, _ = obj.get_axe_units()
        for field in obj:
            pts.append(get_separation_position(field,
                                               wall_direction=wall_direction,
                                               wall_position=wall_position,
                                               interval=interval))
        return Profile(times, pts, unit_x=obj.get_comp('unit_time'),
                       unit_y=unit_axe)
    else:
        raise ValueError("Unknown type for 'obj'")
    # Getting separation position ( We get separation points on adjacents
    # lines, and we interpolate.)
    #    Getting lines around wall
    nmb_lines = 4
    if wall_position < axe[0]:
        lines_pos = axe[0:nmb_lines]
    elif wall_position > axe[-1]:
        lines_pos = axe[-nmb_lines-1:-1]
    else:
        inds = V.get_indice_on_axe(wall_direction, wall_position)
        if len(inds) == 1:
            inds = [inds[0], inds[0] + 1]
        if inds[0] - nmb_lines/2 < 0:
            lines_pos = axe[0:nmb_lines]
        elif inds[-1] + nmb_lines/2 > len(axe):
            lines_pos = axe[-nmb_lines-1:-1]
        else:
            lines_pos = axe[inds[0] - nmb_lines/2:inds[1] + nmb_lines/2]
    #    Getting separation points on surrounding lines
    seps = np.array([])
    for lp in lines_pos:
        # extraction one line
        tmp_profile, _ = V.get_profile(wall_direction, lp)
        # getting the velocity sign changment on the line
        values = tmp_profile.get_interpolated_value(y=0)
        values = np.array(values)
        # masking with 'interval'
        values = values[np.logical_and(values > interval[0],
                                       values < interval[1])]
        seps = np.append(seps, np.mean(values))
    # deleting lines where no separation points were found
    if np.any(np.isnan(seps)):
        warnings.warn("I can't find a separation points on one (or more)"
                      " line(s). You may want to change 'interval' values")
        seps = seps[~np.isnan(seps)]
        lines_pos = lines_pos[~np.isnan(seps)]
    interp = UnivariateSpline(lines_pos, seps, k=1)
    return interp(wall_position)


### Critical lines detection algorithm ###

def get_critical_line(VF, source_point, direction, kol='stream',
                      delta=1, fit='None', order=2):
    """
    Return a parametric curve fitting the virtual streamlines expanding from
    the 'source_point' critical point on the 'VF' field.

    Parameters
    ----------
    VF : VelocityField object
        Base field for streamline
    source_point : Point object
        Source critical point.
    direction : integer
        Direction in which the streamline should go.
        (0 for x axis, 1 for y axis)
    kol : string
        Kind of line to use (can be 'stream' for streamlines (default)
        or 'track' for tracklines).
    delta : integer, optional
        Range (in axis step) in where searching for expanding streamlines.
        (Default is 1)
    fit : string, optional
        Kind of fitting. Can be 'polynomial' or 'ellipse' or 'none'.
    order : integer, optional
        Order for the polynomial fitting (Default is 2).

    Returns
    -------
    If 'None' :
        return a Points object representing the curve.
    If 'polynomial' :
        return the polynomial coefficient in a tuple.
        (warning, these coefficient are to apply on the axis given by
        'direction')
    If 'ellipse' :
        return 'radii' (ellipse semi-radii),
        'center' (ellipse center) and
        'angle' (angle between the semi-axis)
    """
    # check parameters coherence
    if not isinstance(VF, VelocityField):
        raise TypeError("'VF' must be a VelocityField object")
    if not isinstance(source_point, Points):
        raise TypeError("'source_point' must be a Point object")
    if not isinstance(direction, int):
        raise TypeError("'direction' must be an integer")
    if not direction in [0, 1]:
        raise ValueError("'direction' must be 0 or 1")
    if not isinstance(delta, int):
        raise TypeError("'delta' must be an integer")
    if delta < 0:
        raise ValueError("'delta' must be positive")
    if not isinstance(fit, STRINGTYPES):
        raise TypeError("'fit' must be a string")
    if not isinstance(order, int):
        raise TypeError("'order' must be an integer")
    if order < 0:
        raise ValueError("'order' must be positive")
    # get initial position and axis steps
    x_pt = source_point.xy[0, 0]
    y_pt = source_point.xy[0, 1]
    axe_x, axe_y = VF.get_axes()
    dx = axe_x[1] - axe_x[0]
    dy = axe_y[1] - axe_y[0]
    # get around positions
    # TODO : may be optimize
    if direction == 0:
        xs = [x_pt - dx*delta, x_pt + dx*delta]
        ys = [y_pt]  # [y_pt - dy, y_pt, y_pt + dy]
    else:
        xs = [x_pt]  # [x_pt - dx, x_pt, x_pt + dx]
        ys = [y_pt - dy*delta, y_pt + dy*delta]
    xs, ys = np.meshgrid(xs, ys)
    pts = zip(xs.flatten(), ys.flatten())
    # get stream or track lines on around positions
    if kol == 'stream':
        lines = VF.get_streamlines(pts)
    elif kol == 'track':
        lines = VF.get_tracklines(pts)
    else:
        raise ValueError("Unknown value for 'kol' (see documentation)")
    # remove streamline near the critical point
    for i in np.arange(len(lines)):
        if direction == 0:
            lines[i] = lines[i].cut(interv_x=[x_pt - 3*delta*dx,
                                              x_pt + 3*delta*dx])
        else:
            lines[i] = lines[i].cut(interv_y=[y_pt - 3*delta*dy,
                                              y_pt + 3*delta*dy])
    # concatenating streamlines
    line_t = lines[0]
    for sl in lines[1::]:
        line_t += sl
    # fitting
    if fit == 'none':
        return line_t
    elif fit == 'polynomial':
        if direction == 1:
            return line_t.reverse().fit(fit, order=order)
        else:
            return line_t.fit(fit, order=order)
    elif fit == 'ellipse':
        return line_t.fit(fit)
    else:
        raise ValueError("Unknown kind of fitting")


### Criterion computation ###
def get_sigma(vectorfield, radius=None, ind=False, mask=None, raw=False):
    """
    Return the sigma scalar field, reprensenting the homogeneity of the
    VectorField. Values of 1 mean homogeneous velocity field  and 0 mean
    heterogeneous velocity field. Heterogeneous velocity zone are
    representative of zone with critical points.
    In details, Sigma is calculated as the variance of the 'ecart' between
    velocity angles of points surrounding the point of interest.

    Parameters
    ----------
    vectorfield : VectorField object
    radius : number, optionnal
        The radius used to choose the zone where to compute
        sigma for each point. If not mentionned, a value is choosen in
        order to have about 8 points in the circle. It allow to get good
        result, without big computation cost.
    ind : boolean
        If 'True', radius is expressed on number of vectors.
        If 'False' (default), radius is expressed on axis unit.
    mask : array of boolean, optionnal
        Has to be an array of the same size of the vector field object,
        gamma will be compute only where mask is 'False'.
    raw : boolean, optional
        If 'False' (default), a ScalarField is returned,
        if 'True', an array is returned.

    Returns
    -------
    Sigma : ScalarField
        Sigma scalar field
    """
    # parameters constitency
    if not isinstance(vectorfield, VectorField):
        raise TypeError("'vectorfield' must be a VectorField object")
    if radius is None:
        radius = 1.9
        ind = True
    if not isinstance(radius, NUMBERTYPES):
        raise TypeError("'radius' must be a number")
    if not isinstance(ind, bool):
        raise TypeError("'ind' must be a boolean")
    if mask is None:
        mask = np.zeros(vectorfield.get_dim())
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'zone' must be an array of boolean")
    else:
        mask = np.array(mask)
    # Getting neighbouring points motif
    axe_x, axe_y = vectorfield.get_axes()
    indcentral = [int(len(axe_x)/2.), int(len(axe_y)/2.)]
    if ind:
        motif = vectorfield.get_points_around(indcentral, radius, ind)
        motif = motif - indcentral
    else:
        ptcentral = [axe_x[indcentral[0]], axe_y[indcentral[1]]]
        motif = vectorfield.get_points_around(ptcentral, radius, ind)
        motif = motif - indcentral
    nmbpts = len(motif)
    # Getting vector angles
    theta = vectorfield.get_comp('theta', raw=True).data
    # récupération du masque du champ de vitesse
    mask = np.logical_or(mask, vectorfield.get_comp('mask', raw=True))
    # Ajout au masque des valeurs sur le bord
    if ind:
        indx = np.arange(len(axe_x))
        indy = np.arange(len(axe_y))
        border_x = np.logical_or(indx <= indx[0] + (radius - 1),
                                 indx >= indx[-1] - (radius - 1))
        border_y = np.logical_or(indy <= indy[0] + (radius - 1),
                                 indy >= indy[-1] - (radius - 1))
        border_x, border_y = np.meshgrid(border_x, border_y)
        mask1 = np.logical_or(border_x, border_y)
    else:
        delta = (axe_x[1] - axe_x[0] + axe_y[1] - axe_y[0])/2
        border_x = np.logical_or(axe_x <= axe_x[0] + (radius - delta),
                                 axe_x >= axe_x[-1] - (radius - delta))
        border_y = np.logical_or(axe_y <= axe_y[0] + (radius - delta),
                                 axe_y >= axe_y[-1] - (radius - delta))
        border_x, border_y = np.meshgrid(border_x, border_y)
        mask1 = np.logical_or(border_x, border_y)
    # calcul de delta moyen
    deltamoy = 2.*np.pi/(nmbpts)
    # boucle sur les points
    sigmas = np.zeros(vectorfield.get_dim())
    mask2 = np.zeros(mask.shape)
    for inds, pos, _ in vectorfield:
        # On ne fait rien si la valeur est masquée
        if mask[inds[1], inds[0]]:
            continue
        # stop if on border
        if mask1[inds[1], inds[0]]:
            continue
        # stop if one surrounding point is masked
        skip = [mask[i[1], i[0]] for i in motif + inds]
        if np.any(skip):
            mask2[inds[1], inds[0]] = True
            continue
        # Extraction des thetas interessants
        theta_nei = np.zeros((len(motif),))
        i = 0
        for indx, indy in motif + inds:
            theta_nei[i] = theta[indy, indx]
            i += 1
        # Tri des thetas et calcul de deltas
        theta_nei.sort()
        delta = np.zeros((len(theta_nei) + 1,))
        for i in np.arange(len(theta_nei) - 1):
            delta[i] = theta_nei[i+1] - theta_nei[i]
        delta[-1] = np.pi*2 - (theta_nei[-1] - theta_nei[0])
        # calcul de sigma
        sigmas[inds[1], inds[0]] = np.var(delta)
    # calcul (analytique) du sigma max
    sigma_max = ((np.pi*2 - deltamoy)**2
                 + (nmbpts - 1)*(0 - deltamoy)**2)/nmbpts
    # normalisation analytique
    sigmas /= sigma_max
    # masking
    mask = np.logical_or(mask, mask1)
    mask = np.logical_or(mask, mask2)
    sigmas = np.ma.masked_array(sigmas, mask)
    if raw:
        return sigmas
    else:
        sigma_sf = ScalarField()
        unit_x, unit_y = vectorfield.get_axe_units()
        sigma_sf.import_from_arrays(axe_x, axe_y, sigmas,
                                    unit_x=unit_x, unit_y=unit_y,
                                    unit_values=make_unit(""))
        return sigma_sf


def get_gamma(vectorfield, radius=None, ind=False, kind='gamma1', mask=None,
              raw=False):
    """
    Return the gamma scalar field. Gamma criterion is used in
    vortex analysis.
    The fonction recognize if the field is ortogonal, and use an
    apropriate algorithm.

    Parameters
    ----------
    vectorfield : VectorField object
    radius : number, optionnal
        The radius used to choose the zone where to compute
        gamma for each point. If not mentionned, a value is choosen in
        ordre to have about 8 points in the circle. It allow to get good
        result, without big computation cost.
    ind : boolean
        If 'True', radius is expressed on number of vectors.
        If 'False' (default), radius is expressed on axis unit.
    kind : string
        If 'gamma1' (default), compute gamma1 criterion.
        If 'gamma2', compute gamma2 criterion (with relative velocities).
    mask : array of boolean, optionnal
        Has to be an array of the same size of the vector field object,
        gamma will be compute only where mask is 'False'.
    raw : boolean, optional
        If 'False' (default), a ScalarField is returned,
        if 'True', an array is returned.
    """
    ### Checking parameters coherence ###
    if not isinstance(vectorfield, VectorField):
        raise TypeError("'vectorfield' must be a VectorField object")
    if radius is None:
        radius = 1.9
        ind = True
    if not isinstance(radius, NUMBERTYPES):
        raise TypeError("'radius' must be a number")
    if not isinstance(ind, bool):
        raise TypeError("'ind' must be a boolean")
    axe_x, axe_y = vectorfield.get_axes()
    delta = (axe_x[1] - axe_x[0] + axe_y[1] - axe_y[0])/2
    if not isinstance(kind, STRINGTYPES):
        raise TypeError("'kind' must be a string")
    if not kind in ['gamma1', 'gamma2']:
        raise ValueError("Unkown value for kind")
    if mask is None:
        mask = np.zeros(vectorfield.get_dim())
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'zone' must be an array of boolean")
    else:
        mask = np.array(mask)
    ### Importing data from vectorfield (velocity, axis and mask) ###
    Vx = vectorfield.get_comp('Vx', raw=True).data
    Vy = vectorfield.get_comp('Vy', raw=True).data
    mask = np.logical_or(mask, vectorfield.get_comp('mask', raw=True))
    norm_v = vectorfield.get_comp('magnitude', raw=True)
    ### Compute motif and motif angles on an arbitrary point ###
    axe_x, axe_y = vectorfield.get_axes()
    indcentral = [int(len(axe_x)/2.), int(len(axe_y)/2.)]
    if ind:
        motif = vectorfield.get_points_around(indcentral, radius, ind)
        motif = motif - indcentral
    else:
        ptcentral = [axe_x[indcentral[0]], axe_y[indcentral[1]]]
        motif = vectorfield.get_points_around(ptcentral, radius, ind)
        motif = motif - indcentral
    nmbpts = len(motif)
    # getting the vectors between center and neighbouring
    deltax = axe_x[1] - axe_x[0]
    deltay = axe_y[1] - axe_y[0]
    vector_a_x = np.zeros(motif.shape[0])
    vector_a_y = np.zeros(motif.shape[0])
    for i, indaround in enumerate(motif):
        vector_a_x[i] = indaround[0]*deltax
        vector_a_y[i] = indaround[1]*deltay
    norm_vect_a = (vector_a_x**2 + vector_a_y**2)**(.5)
    ### Generating masks ###
    # creating surrounding masked point zone mask
    mask_surr = np.zeros(mask.shape)
    inds_masked = np.transpose(np.where(mask))
    for ind_masked in inds_masked:
        for i, j in motif + ind_masked:
            # continue if outside the field
            if i < 0 or j < 0 or i >= mask_surr.shape[0]\
                    or j >= mask_surr.shape[1]:
                continue
            mask_surr[i, j] = True
    # creating near-border zone mask
    if ind:
        indx = np.arange(len(axe_x))
        indy = np.arange(len(axe_y))
        border_x = np.logical_or(indx <= indx[0] + (int(radius) - 1),
                                 indx >= indx[-1] - (int(radius) - 1))
        border_y = np.logical_or(indy <= indy[0] + (int(radius) - 1),
                                 indy >= indy[-1] - (int(radius) - 1))
        border_x, border_y = np.meshgrid(border_x, border_y)
        mask_border = np.logical_or(border_x, border_y)
    else:
        delta = (axe_x[1] - axe_x[0] + axe_y[1] - axe_y[0])/2
        border_x = np.logical_or(axe_x <= axe_x[0] + (radius - delta),
                                 axe_x >= axe_x[-1] - (radius - delta))
        border_y = np.logical_or(axe_y <= axe_y[0] + (radius - delta),
                                 axe_y >= axe_y[-1] - (radius - delta))
        border_x, border_y = np.meshgrid(border_x, border_y)
        mask_border = np.logical_or(border_x, border_y)
    ### Loop on points ###
    gammas = np.zeros(vectorfield.get_dim())
    for inds, pos, _ in vectorfield:
        ind_x = inds[0]
        ind_y = inds[1]
        # stop if masked or on border or with a masked surrouinding point
        if mask[ind_y, ind_x] or mask_surr[ind_y, ind_x]\
                or mask_border[ind_y, ind_x]:
            continue
        # getting neighbour points
        indsaround = motif + inds
        # If necessary, compute mean velocity on points (gamma2)
        v_mean = [0., 0.]
        if kind == 'gamma2':
            for indaround in indsaround:
                v_mean[0] += Vx[ind_y, ind_x]
                v_mean[1] += Vy[ind_y, ind_x]
            v_mean[0] /= nmbpts
            v_mean[1] /= nmbpts
        ### Loop on neighbouring points ###
        gamma = 0.
        for i, indaround in enumerate(indsaround):
            inda_x = indaround[0]
            inda_y = indaround[1]
            # getting vectors for scalar product
            vector_b_x = Vx[inda_y, inda_x] - v_mean[0]
            vector_b_y = Vy[inda_y, inda_x] - v_mean[1]
            if kind == 'gamma1':
                denom = norm_v[inda_y, inda_x]*norm_vect_a[i]
            else:
                denom = (vector_b_x**2 + vector_b_y**2)**.5*norm_vect_a[i]
            # getting scalar product
            if denom != 0:
                gamma += (vector_a_x[i]*vector_b_y
                          - vector_a_y[i]*vector_b_x)/denom
        # storing computed gamma value
        gammas[ind_y, ind_x] = gamma/nmbpts
    ### Applying masks ###
    mask = np.logical_or(mask, mask_border)
    mask = np.logical_or(mask, mask_surr)
    gammas = np.ma.masked_array(gammas, mask)
    ### Creating gamma ScalarField ###
    if raw:
        return gammas
    else:
        gamma_sf = ScalarField()
        unit_x, unit_y = vectorfield.get_axe_units()
        gamma_sf.import_from_arrays(axe_x, axe_y, gammas,
                                    unit_x=unit_x, unit_y=unit_y,
                                    unit_values=make_unit(''))
        return gamma_sf


def get_kappa(vectorfield, radius=None, ind=False, kind='kappa1', mask=None,
              raw=False):
    """
    Return the kappa scalar field. Kappa criterion is used in
    vortex analysis.
    The fonction recognize if the field is ortogonal, and use an
    apropriate algorithm.

    Parameters
    ----------
    vectorfield : VectorField object
    radius : number, optionnal
        The radius used to choose the zone where to compute
        kappa for each point. If not mentionned, a value is choosen in
        ordre to have about 8 points in the circle. It allow to get good
        result, without big computation cost.
    ind : boolean
        If 'True', radius is expressed on number of vectors.
        If 'False' (default), radius is expressed on axis unit.
    kind : string
        If 'kappa1' (default), compute kappa1 criterion.
        If 'kappa2', compute kappa2 criterion (with relative velocities).
    mask : array of boolean, optionnal
        Has to be an array of the same size of the vector field object,
        kappa will be compute only where mask is 'False'.
    raw : boolean, optional
        If 'False' (default), a ScalarField is returned,
        if 'True', an array is returned.
    """
    ### Checking parameters coherence ###
    if not isinstance(vectorfield, VectorField):
        raise TypeError("'vectorfield' must be a VectorField object")
    if radius is None:
        radius = 1.9
        ind = True
    if not isinstance(radius, NUMBERTYPES):
        raise TypeError("'radius' must be a number")
    if not isinstance(ind, bool):
        raise TypeError("'ind' must be a boolean")
    axe_x, axe_y = vectorfield.get_axes()
    delta = (axe_x[1] - axe_x[0] + axe_y[1] - axe_y[0])/2
    if not isinstance(kind, STRINGTYPES):
        raise TypeError("'kind' must be a string")
    if not kind in ['kappa1', 'kappa2']:
        raise ValueError("Unkown value for kind")
    if mask is None:
        mask = np.zeros(vectorfield.get_dim())
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'zone' must be an array of boolean")
    else:
        mask = np.array(mask)
    ### Importing data from vectorfield (velocity, axis and mask) ###
    Vx = vectorfield.get_comp('Vx', raw=True).data
    Vy = vectorfield.get_comp('Vy', raw=True).data
    mask = np.logical_or(mask, vectorfield.get_comp('mask', raw=True))
    norm_v = vectorfield.get_comp('magnitude', raw=True)
    ### Compute motif and motif angles on an arbitrary point ###
    axe_x, axe_y = vectorfield.get_axes()
    indcentral = [int(len(axe_x)/2.), int(len(axe_y)/2.)]
    if ind:
        motif = vectorfield.get_points_around(indcentral, radius, ind)
        motif = motif - indcentral
    else:
        ptcentral = [axe_x[indcentral[0]], axe_y[indcentral[1]]]
        motif = vectorfield.get_points_around(ptcentral, radius, ind)
        motif = motif - indcentral
    nmbpts = len(motif)
    # getting the vectors between center and neighbouring
    deltax = axe_x[1] - axe_x[0]
    deltay = axe_y[1] - axe_y[0]
    vector_a_x = np.zeros(motif.shape[0])
    vector_a_y = np.zeros(motif.shape[0])
    for i, indaround in enumerate(motif):
        vector_a_x[i] = indaround[0]*deltax
        vector_a_y[i] = indaround[1]*deltay
    norm_vect_a = (vector_a_x**2 + vector_a_y**2)**(.5)
    ### Generating masks ###
    # creating surrounding masked point zone mask
    mask_surr = np.zeros(mask.shape)
    inds_masked = np.transpose(np.where(mask))
    for ind_masked in inds_masked:
        for i, j in motif + ind_masked:
            # continue if outside the field
            if i < 0 or j < 0 or i >= mask_surr.shape[0]\
                    or j >= mask_surr.shape[1]:
                continue
            mask_surr[i, j] = True
    # creating near-border zone mask
    if ind:
        indx = np.arange(len(axe_x))
        indy = np.arange(len(axe_y))
        border_x = np.logical_or(indx <= indx[0] + (int(radius) - 1),
                                 indx >= indx[-1] - (int(radius) - 1))
        border_y = np.logical_or(indy <= indy[0] + (int(radius) - 1),
                                 indy >= indy[-1] - (int(radius) - 1))
        border_x, border_y = np.meshgrid(border_x, border_y)
        mask_border = np.logical_or(border_x, border_y)
    else:
        delta = (axe_x[1] - axe_x[0] + axe_y[1] - axe_y[0])/2
        border_x = np.logical_or(axe_x <= axe_x[0] + (radius - delta),
                                 axe_x >= axe_x[-1] - (radius - delta))
        border_y = np.logical_or(axe_y <= axe_y[0] + (radius - delta),
                                 axe_y >= axe_y[-1] - (radius - delta))
        border_x, border_y = np.meshgrid(border_x, border_y)
        mask_border = np.logical_or(border_x, border_y)
    ### Loop on points ###
    kappas = np.zeros(vectorfield.get_dim())
    for inds, pos, _ in vectorfield:
        ind_x = inds[0]
        ind_y = inds[1]
        # stop if masked or on border or with a masked surrouinding point
        if mask[ind_y, ind_x] or mask_surr[ind_y, ind_x]\
                or mask_border[ind_y, ind_x]:
            continue
        # getting neighbour points
        indsaround = motif + np.array(inds)
        # If necessary, compute mean velocity on points (kappa2)
        v_mean = [0., 0.]
        if kind == 'kappa2':
            nmbpts = len(indsaround)
            for indaround in indsaround:
                v_mean[0] += Vx[ind_y, ind_x]
                v_mean[1] += Vy[ind_y, ind_x]
            v_mean[0] /= nmbpts
            v_mean[1] /= nmbpts
        ### Loop on neighbouring points ###
        kappa = 0.
        nmbpts = len(indsaround)
        for i, indaround in enumerate(indsaround):
            inda_x = indaround[0]
            inda_y = indaround[1]
            # getting vectors for scalar product
            vector_b_x = Vx[inda_y, inda_x] - v_mean[0]
            vector_b_y = Vy[inda_y, inda_x] - v_mean[1]
            if kind == 'kappa1':
                denom = norm_v[inda_y, inda_x]*norm_vect_a[i]
            else:
                denom = (vector_b_x**2 + vector_b_y**2)**.5*norm_vect_a[i]
            # getting scalar product
            if denom != 0:
                kappa += (vector_a_x[i]*vector_b_x
                          + vector_a_y[i]*vector_b_y)/denom
        # storing computed kappa value
        kappas[ind_y, ind_x] = kappa/nmbpts
    ### Applying masks ###
    mask = np.logical_or(mask, mask_border)
    mask = np.logical_or(mask, mask_surr)
    kappas = np.ma.masked_array(kappas, mask)
    ### Creating kappa ScalarField ###
    if raw:
        return kappas
    else:
        kappa_sf = ScalarField()
        unit_x, unit_y = vectorfield.get_axe_units()
        kappa_sf.import_from_arrays(axe_x, axe_y, kappas,
                                    unit_x=unit_x, unit_y=unit_y,
                                    unit_values=make_unit(''))
        return kappa_sf


def get_q_criterion(vectorfield, mask=None, raw=False):
    """
    Return the normalized scalar field of the 2D Q criterion .
    Define as "1/2*(sig**2 - S**2)" , with "sig" the deformation tensor,
    norm and "S" the rate of rotation tensor norm.

    Parameters
    ----------
    vectorfield : VectorField object
    mask : array of boolean, optional
        Has to be an array of the same size of the vector field object,
        iota2 will be compute only where zone is 'False'.
    raw : boolean, optional
        If 'False' (default), a ScalarField is returned,
        if 'True', an array is returned.
    """
    if not isinstance(vectorfield, VectorField):
        raise TypeError("'vectorfield' must be a VectorField object")
    if mask is None:
        mask = np.zeros(vectorfield.get_dim())
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'mask' must be an array of boolean")
    else:
        mask = np.array(mask)
    axe_x, axe_y = vectorfield.get_axes()
    Vx = vectorfield.get_comp('Vx', raw=True)
    Vy = vectorfield.get_comp('Vy', raw=True)
    #calcul des gradients
    Exy, Exx = np.gradient(Vx)
    Eyy, Eyx = np.gradient(Vy)
    # calcul de 'sig**2' et 'S**2'
    sig = 2.*(Exy - Eyx)**2
    S = (2.*Exx)**2 + 2.*(Eyx + Exy)**2 + (2.*Eyy)**2
    # calcul de Qcrit
    qcrit = 1./2*(sig - S)
    qcrit = np.ma.masked_array(qcrit, mask)
    qcrit = qcrit/np.abs(qcrit).max()
    if raw:
        return qcrit
    else:
        q_sf = ScalarField()
        q_sf.import_from_arrays(axe_x, axe_y, qcrit,
                                unit_x=vectorfield.comp_x.unit_x,
                                unit_y=vectorfield.comp_x.unit_y)
        return q_sf


def get_iota(vectorfield, mask=None, raw=False):
    """
    Return the iota scalar field. iota criterion is used in
    vortex analysis.
    The fonction is only usable on orthogonal fields.
    Warning : This function is minimum at the saddle point center, and
    maximum around this point.

    Parameters
    ----------
    vectorfield : VectorField object
    mask : array of boolean, optionnal
        Has to be an array of the same size of the vector field object,
        iota2 will be compute only where zone is 'False'.
    raw : boolean, optional
        If 'False' (default), a ScalarField is returned,
        if 'True', an array is returned.
    """
    if not isinstance(vectorfield, VectorField):
        raise TypeError("'vectorfield' must be a VectorField object")
    if mask is None:
        mask = np.zeros(vectorfield.get_dim())
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'mask' must be an array of boolean")
    else:
        mask = np.array(mask)
    axe_x, axe_y = vectorfield.get_axes()
    # récupération de theta et de son masque
    theta = vectorfield.get_comp('theta', raw=True)
    mask = np.logical_or(mask, theta.mask)
    theta = theta.data
    # récupération des dx et dy
    dx = np.abs(axe_x[0] - axe_x[2])
    dy = np.abs(axe_y[0] - axe_y[2])
    # boucle sur les points
    grad_theta_m = np.zeros(vectorfield.get_dim())
    maskf = mask.copy()
    for inds, _, _ in vectorfield:
        # arrete si on est sur un bords ou sur une valeurs masquée
        if (inds[1] == 0 or inds[0] == 0 or inds[1] == len(axe_y)-1
                or inds[0] == len(axe_x)-1):
            maskf[inds[1], inds[0]] = True
            continue
        if np.any(mask[inds[1]-1:inds[1]+2, inds[0]-1:inds[0]+2]):
            maskf[inds[1], inds[0]] = True
            continue
        # calcul de theta_m autours
        theta_m = [[theta[inds[1] - 1, inds[0] - 1] + 3./4*np.pi,
                   theta[inds[1], inds[0] - 1] + np.pi/2,
                   theta[inds[1] + 1, inds[0] - 1] + 1./4*np.pi],
                  [theta[inds[1] - 1, inds[0]] + np.pi,
                   0.,
                   theta[inds[1] + 1, inds[0]]],
                  [theta[inds[1] - 1, inds[0] + 1] - 3./4*np.pi,
                   theta[inds[1], inds[0] + 1] - np.pi/2,
                   theta[inds[1] + 1, inds[0] + 1] - np.pi/4]]
        theta_m = np.mod(theta_m, np.pi*2)
        grad_x = (theta_m[2, 1] - theta_m[0, 1])/dx
        grad_y = (theta_m[1, 2] - theta_m[1, 0])/dy
        # calcul de gradthetaM
        grad_theta_m[inds[1], inds[0]] = (grad_x**2 + grad_y**2)**(1./2)
    # application du masque
    grad_theta_m = np.ma.masked_array(grad_theta_m, maskf)
    if raw:
        return grad_theta_m
    else:
        iota_sf = ScalarField()
        unit_x, unit_y = vectorfield.get_axe_units()
        iota_sf.import_from_arrays(axe_x, axe_y, grad_theta_m,
                                   unit_x=unit_x, unit_y=unit_y,
                                   unit_values=make_unit(''))
        return iota_sf
