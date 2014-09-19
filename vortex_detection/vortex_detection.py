# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:07:07 2014

@author: muahah

For performance
"""


import pdb
from ..core import Points, OrientedPoints, Profile, ScalarField, VectorField,\
    make_unit, ARRAYTYPES, NUMBERTYPES, STRINGTYPES, TemporalScalarFields,\
    TemporalVectorFields, ShapeError
from ..field_treatment import get_streamlines, get_gradients
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
from scipy import optimize
from scipy import linalg
import warnings
import sets
import scipy.ndimage.measurements as msr
import unum
import copy


def velocityfield_to_vf(vectorfield, time):
    """
    Create a VF object from a VectorField object.
    """
    # extracting data
    vx = np.transpose(vectorfield.comp_x)
    vy = np.transpose(vectorfield.comp_y)
    ind_x, ind_y = vectorfield.axe_x, vectorfield.axe_y
    mask = np.transpose(vectorfield.mask)
    theta = np.transpose(vectorfield.theta)
    # TODO : add the following when fill will be optimized
    #THETA.fill()
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
        Return the field as VectorField object.
        """
        tmp_vf = VectorField()
        vx = np.transpose(self.vx)
        vy = np.transpose(self.vy)
        mask = np.transpose(self.mask)
        tmp_vf.import_from_arrays(self.axe_x, self.axe_y, vx, vy, mask)
        return tmp_vf

    def get_cp_position(self, window_size=2):
        """
        Return critical points positions and their associated PBI.
        (PBI : Poincarre_Bendixson indice)
        Positions are returned in axis unities (axe_x and axe_y) and are
        always at the center of 4 points (maximum accuracy of this algorithm
        is limited by the field spatial resolution).

        Parameters
        ----------
        window_size : integer, optional
            Minimal window size for PBI detection.
            Smaller window size allow detection where points are dense.
            Default is finnest possible (2).

        Returns
        -------
        pos : 2xN array
            position (x, y) of the detected critical points.
        pbis : 1xN array
            PBI (1 indicate a node, -1 a saddle point)
        """
        if not isinstance(window_size, int):
            raise TypeError()
        if window_size < 2:
            raise ValueError()
        delta_x = self.axe_x[1] - self.axe_x[0]
        delta_y = self.axe_y[1] - self.axe_y[0]
        positions = []
        pbis = []
        grid_x = np.append(np.arange(0, self.shape[0] - window_size + 1,
                                     window_size), self.shape[0])
        grid_y = np.append(np.arange(0, self.shape[1] - window_size + 1,
                                     window_size), self.shape[1])
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
                pool = np.delete(pool, 0)
            # if there is only one critical point and the field is as
            # small as possible, we store the cp position, the end !
            elif nmb_struct == (1, 1):
                cp_pos = tmp_vf._get_poi_position()
                positions.append((tmp_vf.axe_x[cp_pos[0]] + delta_x/2.,
                                  tmp_vf.axe_y[cp_pos[1]] + delta_y/2.))
                pbis.append(tmp_vf.pbi_x[-1])
                pool = np.delete(pool, 0)
            # if the cp density is too high, bu the pbi is valid
            elif tmp_vf.pbi_x[-1] in [-1, 1]:
                cp_pos = [np.mean(tmp_vf.axe_x), np.mean(tmp_vf.axe_y)]
                positions.append(cp_pos)
                pbis.append(tmp_vf.pbi_x[-1])
                pool = np.delete(pool, 0)
            # if the cp density is too high and the pbi is invalid
            else:
                pool = np.delete(pool, 0)
        # removing doublons
        positions = np.array(positions)
        pbis = np.array(pbis)
        ind_pos = _argunique(positions)
        positions = positions[ind_pos]
        pbis = pbis[ind_pos]
        # returning
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
        theta = theta[:, mask_x]
        time = self.time
        if len(axe_x) == 0 or len(axe_y) == 0:
            raise ValueError()
        return VF(vx, vy, axe_x, axe_y, mask, theta, time)

    def _find_arbitrary_cut_positions(self):
        """
        Return the position along x and y where the field can be cut.
        (at middle positions along each axes)
        """
        if np.any(self.shape <= 4):
            return None, None
        len_x = self.shape[1]
        len_y = self.shape[0]
        grid_x = [0, np.round(len_x/2.), len_x]
        grid_y = [0, np.round(len_y/2.), len_y]
        return grid_x, grid_y

    def _find_cut_positions(self):
        """
        Return the position along x and y where the field has to be cut to
        isolate critical points.
        Return '(None, None)' if there is no possible cut position.
        """
        raise Exception("WARNING : may be inusable with actual "
                        "'_split_the_field'")
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
        if len(grid_x) == 2 and len(grid_y) == 2:
            return None, None
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
            if grid_x[i] == 0:
                slic_x = slice(grid_x[i], grid_x[i + 1] + 1)
            else:
                slic_x = slice(grid_x[i] - 1, grid_x[i + 1] + 1)
            for j in np.arange(len(grid_y) - 1):
                if grid_y[j] == 0:
                    slic_y = slice(grid_y[j], grid_y[j + 1] + 1)
                else:
                    slic_y = slice(grid_y[j] - 1, grid_y[j + 1] + 1)
                vx_tmp = self.vx[slic_x, slic_y]
                vy_tmp = self.vy[slic_x, slic_y]
                mask_tmp = self.mask[slic_x, slic_y]
                theta_tmp = self.theta[slic_x, slic_y]
                time_tmp = self.time
                axe_x_tmp = self.axe_x[slic_y]
                axe_y_tmp = self.axe_y[slic_x]
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


class CritPoints(object):
    """
    Class representing a set of critical point associated to a VectorField.

    Parameters
    ----------
    unit_time : string or Unit object
        Unity for the time.
    """
    ### Operators ###
    def __init__(self,  unit_time='s'):
        # check parameters
        self.foc = np.array([])
        self.foc_c = np.array([])
        self.node_i = np.array([])
        self.node_o = np.array([])
        self.sadd = np.array([])
        self.pbi_m = np.array([])
        self.pbi_p = np.array([])
        self.times = np.array([])
        self.unit_time = unit_time
        self.unit_x = make_unit('')
        self.unit_y = make_unit('')
        self.colors = ['r', 'b', 'y', 'm', 'g', 'w', 'k']

    def __add__(self, obj):
        if isinstance(obj, CritPoints):
            if not self.unit_time == obj.unit_time:
                raise ValueError()
            if not self.unit_x == obj.unit_x:
                raise ValueError()
            if not self.unit_y == obj.unit_y:
                raise ValueError()
            tmp_CP = CritPoints(unit_time=self.unit_time)
            tmp_CP.foc = np.append(self.foc, obj.foc)
            tmp_CP.foc_c = np.append(self.foc_c, obj.foc_c)
            tmp_CP.node_i = np.append(self.node_i, obj.node_i)
            tmp_CP.node_o = np.append(self.node_o, obj.node_o)
            tmp_CP.sadd = np.append(self.sadd, obj.sadd)
            tmp_CP.pbi_m = np.append(self.pbi_m, obj.pbi_m)
            tmp_CP.pbi_p = np.append(self.pbi_p, obj.pbi_p)
            tmp_CP.times = np.append(self.times, obj.times)
            tmp_CP._sort_by_time()
            # test if there is double
            for i, t in enumerate(tmp_CP.times[0:-1]):
                if t == tmp_CP.times[i+1]:
                    raise ValueError()
            # returning
            return tmp_CP
        else:
            raise TypeError()

    ### Attributes ###
    @property
    def unit_time(self):
        return self.__unit_time

    @unit_time.setter
    def unit_time(self, new_unit_time):
        if isinstance(new_unit_time, STRINGTYPES):
            new_unit_time = make_unit(new_unit_time)
        if not isinstance(new_unit_time, unum.Unum):
            raise TypeError()
        self.__unit_time = new_unit_time

    ### Properties ###
    @property
    def iter(self):
        return [self.foc, self.foc_c, self.node_i, self.node_o, self.sadd,
                self.pbi_m, self.pbi_p]

    @property
    def iter_traj(self):
        return [self.foc_traj, self.foc_c_traj, self.node_i_traj,
                self.node_o_traj, self.sadd_traj]

    ### Watchers ###
    def copy(self):
        """
        Return a copy of the CritPoints object.
        """
        return copy.deepcopy(self)

    def compute_traj(self, epsilon=None):
        """
        Compute cp trajectory from cp positions.

        Parameters
        ----------
        epsilon : number, optional
            Maximal distance between two successive points.
            default value is Inf.
        """
        # get points together with v as time
        focs = np.array([])
        for i, pts in enumerate(self.foc):
            if len(pts) == 0:
                continue
            pts.v = [self.times[i]]*len(pts)
            focs = np.append(focs, pts.decompose())
        focs_c = np.array([])
        for i, pts in enumerate(self.foc_c):
            if len(pts) == 0:
                continue
            pts.v = [self.times[i]]*len(pts)
            focs_c = np.append(focs_c, pts.decompose())
        nodes_i = np.array([])
        for i, pts in enumerate(self.node_i):
            if len(pts) == 0:
                continue
            pts.v = [self.times[i]]*len(pts)
            nodes_i = np.append(nodes_i, pts.decompose())
        nodes_o = np.array([])
        for i, pts in enumerate(self.node_o):
            if len(pts) == 0:
                continue
            pts.v = [self.times[i]]*len(pts)
            nodes_o = np.append(nodes_o, pts.decompose())
        sadds = np.array([])
        for i, pts in enumerate(self.sadd):
            if len(pts) == 0:
                continue
            pts.v = [self.times[i]]*len(pts)
            sadds = np.append(sadds, pts.decompose())
        # getting trajectories
        self.foc_traj = self._get_cp_time_evolution(focs, times=self.times,
                                                    epsilon=epsilon)
        self.foc_c_traj = self._get_cp_time_evolution(focs_c, times=self.times,
                                                      epsilon=epsilon)
        self.node_i_traj = self._get_cp_time_evolution(nodes_i,
                                                       times=self.times,
                                                       epsilon=epsilon)
        self.node_o_traj = self._get_cp_time_evolution(nodes_o,
                                                       times=self.times,
                                                       epsilon=epsilon)
        self.sadd_traj = self._get_cp_time_evolution(sadds, times=self.times,
                                                     epsilon=epsilon)

    ### Modifiers ###
    def add_point(self, foc=None, foc_c=None, node_i=None, node_o=None,
                  sadd=None, pbi_m=None, pbi_p=None, time=None):
        """
        Add a new point to the CritPoints object.

        Parameters
        ----------
        foc, foc_c, node_i, node_o, sadd, pbi_m, pbi_p : Points objects
            Representing the critical points at this time.
        time : number
            Time.
        """
        # check parameters
        for pt in [foc, foc_c, node_i, node_o, sadd, pbi_m, pbi_p]:
            if pt is None:
                continue
            if not isinstance(pt, Points):
                raise TypeError()
            pt.v = np.array([])
        if not isinstance(time, NUMBERTYPES):
            raise ValueError()
        if np.any(self.times == time):
            raise ValueError()
        # first point
        if len(self.times) == 0:
            if foc is not None:
                self.unit_x = foc.unit_x
                self.unit_y = foc.unit_y
            elif pbi_m is not None:
                self.unit_x = pbi_m.unit_x
                self.unit_y = pbi_m.unit_y
        # other ones
        self.foc = np.append(self.foc, foc)
        self.foc_c = np.append(self.foc_c, foc_c)
        self.node_i = np.append(self.node_i, node_i)
        self.node_o = np.append(self.node_o, node_o)
        self.sadd = np.append(self.sadd, sadd)
        self.pbi_m = np.append(self.pbi_m, pbi_m)
        self.pbi_p = np.append(self.pbi_p, pbi_p)
        self.times = np.append(self.times, time)
        self._sort_by_time()

    def remove_point(self, time=None, indice=None):
        """
        Remove some critical points.

        Parameters
        ----------
        time : number, optional
            If specified, critical points associated to this time are
            removed.
        indice : integer, optional
            If specified, critical points associated to this indice are
            removed.
        """
        # check parameters
        if time is None and indice is None:
            raise Exception()
        if time is not None and indice is not None:
            raise Exception()
        # remove by time
        if time is not None:
            indice = self._get_indice_from_time(time)
        # remove by indice
        self.foc = np.delete(self.foc, indice)
        self.foc_c = np.delete(self.foc_c, indice)
        self.node_i = np.delete(self.node_i, indice)
        self.node_o = np.delete(self.node_o, indice)
        self.sadd = np.delete(self.sadd, indice)
        self.pbi_m = np.delete(self.pbi_m, indice)
        self.pbi_p = np.delete(self.pbi_p, indice)
        self.times = np.delete(self.times, indice)

    ### Private ###
    def _sort_by_time(self):
        """
        Sort the cp by increasing times.
        """
        indsort = np.argsort(self.times)
        for pt in self.iter:
            pt[:] = pt[indsort]
        self.times[:] = self.times[indsort]

    def _get_indice_from_time(self, time):
        """
        Return the indice associated to the given number.
        """
        # check parameters
        if not isinstance(time, NUMBERTYPES):
            raise TypeError()
        # get indice
        indice = np.argwhere(self.times == time)
        if indice.shape[0] == 0:
            raise ValueError()
        indice = indice[0][0]
        return indice

    @staticmethod
    def _get_cp_time_evolution(points, times=None, epsilon=None):
        """
        Compute the temporal evolution of each vortex centers from a set of
        points at different times. (Points objects must each contain only one
        and point time must be specified in 'v' argument of points).

        Parameters:
        -----------
        points : tuple of Points objects.
            .
        times : array of numbers
            Times. If 'None' (default), only times represented by at least one
            point are taken into account
            (can create wrong link between points).
        epsilon : number, optional
            Maximal distance between two successive points.
            default value is Inf.

        Returns
        -------
        traj : tuple of Points object
            .
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
            Class representing an orthogonal set of points, defined by a
            position and a time.
            """
            def __init__(self, pts_tupl, times):
                if not isinstance(pts_tupl, ARRAYTYPES):
                    raise TypeError("'pts' must be a tuple of Point objects")
                for pt in pts_tupl:
                    if not isinstance(pt, Points):
                        raise TypeError("'pts' must be a tuple of Point"
                                        "objects")
                    if not len(pt) == len(pt.v):
                        raise StandardError("v has not the same dimension as "
                                            "xy")
                # if some Points objects contains more than one point,
                # we decompose them
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

    ### Displayers ###
    def display(self, time=None, indice=None, field=None, **kw):
        """
        Display some critical points.

        Parameters
        ----------
        time : number, optional
            If specified, critical points associated to this time are
            displayed.
        indice : integer, optional
            If specified, critical points associated to this indice are
            displayed.
        field : VectorField object, optional
            If specified, critical points are displayed on the given field.
            critical lines are also computed and displayed
        """
        # check parameters
        if time is None and indice is None:
            if len(self.times) == 1:
                time = self.times[0]
            else:
                raise ValueError()
        if time is not None and indice is not None:
            raise ValueError()
        if time is not None:
            indice = self._get_indice_from_time(time)
        if field is not None:
            if not isinstance(field, VectorField):
                raise TypeError()
        # Set the color
        if 'color' in kw.keys():
            colors = [kw.pop('color')]*len(self.colors)
        else:
            colors = self.colors
        # loop on the points types
        for i, pt in enumerate(self.iter):
            if pt[indice] is None:
                continue
            pt[indice].display(kind='plot', marker='o', color=colors[i],
                               linestyle='none', **kw)
        # display the critical lines
        if field is not None and len(self.sadd[indice].xy) != 0:
            streams = self.sadd[indice]\
                .get_streamlines_from_orientations(field,
                reverse_direction=[True, False])
            for stream in streams:
                stream.display(kind='plot', color=colors[4])

    def display_traj(self, kind='default', **kw):
        """
        Display the stored trajectories.

        Parameters
        ----------
        kind : string
            If 'default', trajectories are plotted in a 2-dimensional plane.
            If 'x', x position of cp are plotted against time.
            If 'y', y position of cp are plotted against time.
        kw : dict, optional
            Arguments passed to plot.
        """
        # check if some trajectories are computed
        try:
            self.foc_traj
        except AttributeError:
            raise StandardError("you must compute trajectories before "
                                "displaying them")
        # display
        if 'color' in kw.keys():
            colors = [kw.pop('color')]*len(self.colors)
        else:
            colors = self.colors
        if kind == 'default':
            for i, trajs in enumerate(self.iter_traj):
                color = colors[i]
                if trajs is None:
                    continue
                for traj in trajs:
                    traj.display(kind='plot', marker='o', color=color)
            plt.xlabel('x {}'.format(self.unit_x.strUnit()))
            plt.xlabel('y {}'.format(self.unit_y.strUnit()))
        elif kind == 'x':
            for i, trajs in enumerate(self.iter_traj):
                color = colors[i]
                if trajs is None:
                    continue
                for traj in trajs:
                    plt.plot(traj.xy[:, 0], traj.v[:], 'o-', color=color)
            plt.xlabel('x {}'.format(self.unit_x.strUnit()))
            plt.ylabel('time {}'.format(self.unit_time.strUnit()))
        elif kind == 'y':
            for i, trajs in enumerate(self.iter_traj):
                color = colors[i]
                if trajs is None:
                    continue
                for traj in trajs:
                    plt.plot(traj.xy[:, 1], traj.v[:], 'o-', color=color)
            plt.ylabel('time {}'.format(self.unit_time.strUnit()))
            plt.xlabel('y {}'.format(self.unit_y.strUnit()))
        else:
            raise StandardError()

    def display_animate(self, TF, **kw):
        """
        Display an interactive windows with the velocity fields from TF and the
        critical points.

        TF : TemporalFields
            .
        kw : dict, optional
            Additional arguments for 'TF.display()'
        """
        return TF.display(suppl_display=self.update_cp, **kw)

    def __update_cp(self, ind):
        self.display(indice=ind, field=TF.fields[ind])

#    def display_traj(self, kind='default'):
#        """
#        Display the stored trajectories.
#
#        Parameters
#        ----------
#        kind : string
#            If 'default', trajectories are plotted in a 2-dimensional plane.
#            If 'x', x position of cp are plotted against time.
#            If 'y', y position of cp are plotted against time.
#        """
#        # check if some trajectories are computed
#        try:
#            self.foc_traj
#        except AttributeError:
#            raise StandardError("you must compute trajectories before "
#                                "displaying them")
#        # display
#        if kind == 'default':
#            for i, trajs in enumerate(self.iter_traj):
#                color = self.colors[i]
#                if trajs is None:
#                    continue
#                for traj in trajs:
#                    traj.display(kind='plot', marker='o', color=color)
#        elif kind == 'x':
#            for i, trajs in enumerate(self.iter_traj):
#                color = self.colors[i]
#                if trajs is None:
#                    continue
#                for traj in trajs:
#                    plt.plot(traj.xy[:, 0], traj.v[:], 'o-', color=color)
#        elif kind == 'y':
#            for i, trajs in enumerate(self.iter_traj):
#                color = self.colors[i]
#                if trajs is None:
#                    continue
#                for traj in trajs:
#                    plt.plot(traj.xy[:, 1], traj.v[:], 'o-', color=color)
#        else:
#            raise StandardError()


### Vortex properties ###
def get_vortex_radius(VF, vort_center, gamma2_radius=None, output_center=False,
                      output_unit=False):
    """
    Return the radius of the given vortex.

    Use the criterion |gamma2| > 2/pi. The returned radius is an average value
    if the vortex zone is not circular.

    Parameters:
    -----------
    VF : vectorfield object
        Velocity field on which compute gamma2.
    vort_center : 2x1 array
        Approximate position of the vortex center.
    gamma2_radius : number, optional
        Radius needed to compute gamma2.
    output_center : boolean, optional
        If 'True', return the associated vortex center, computed using center
        of mass algorythm.
    output_unit ; boolean, optional
        If 'True', return the associated unit.

    Returns :
    ---------
    radius : number
        Average radius of the vortex. If no vortex is found, 0 is returned.
    center : 2x1 array of numbers
        If 'output_center' is 'True', contain the newly computed vortex center.
    unit_radius : Unit object
        Radius unity
    """
    # getting data
    gamma2 = get_gamma(VF, radius=gamma2_radius, ind=False, kind='gamma2',
                       raw=True)
    ind_x = VF.get_indice_on_axe(1, vort_center[0], nearest=True)
    ind_y = VF.get_indice_on_axe(2, vort_center[1], nearest=True)
    dx = VF.axe_x[1] - VF.axe_x[0]
    dy = VF.axe_y[1] - VF.axe_y[0]
    # find vortex zones adn label them
    vort = np.abs(gamma2) > 2/np.pi
    vort, nmb_vort = msr.label(vort)
    # get wanted zone label
    lab = vort[ind_x, ind_y]
    # if we are outside a zone
    if lab == 0:
        if output_center:
            return 0, vort_center
        else:
            return 0
    # else, we compute the radius
    area = dx*dy*np.sum(vort == lab)
    radius = np.sqrt(area/np.pi)
    # optional computed center
    if output_center:
        if gamma2[ind_x, ind_y] > 0:
            pond = gamma2
        else:
            pond = -gamma2
        center = np.array(msr.center_of_mass(pond, vort == lab))
        center[0] = VF.axe_x[0] + center[0]*dx
        center[1] = VF.axe_y[0] + center[1]*dy
    # optional computed unit
    if output_unit:
        unit_radius = VF.unit_x
    # return
    if not output_unit and not output_center:
        return radius
    elif output_unit and not output_center:
        return radius, unit_radius
    elif not output_unit and output_center:
        return radius, center
    else:
        return radius, center, unit_radius


def get_vortex_radius_time_evolution(TVFS, traj, gamma2_radius=None,
                                     output_center=False):
    """
    Return the radius evolution in time for the given vortex center trajectory.

    Use the criterion |gamma2| > 2/pi. The returned radius is an average value
    if the vortex zone is not circular.

    Parameters:
    -----------
    TVFS : TemporalField object
        Velocity field on which compute gamma2.
    traj : Points object
        Trajectory of the vortex.
    gamma2_radius : number, optional
        Radius needed to compute gamma2.
    output_center : boolean, optional
        If 'True', return a Points object with associated vortex centers,
        computed using center of mass algorythm.

    Returns :
    ---------
    radius : Profile object
        Average radius of the vortex. If no vortex is found, 0 is returned.
    center : Points object
        If 'output_center' is 'True', contain the newly computed vortex center.
    """
    radii = np.empty((len(traj.xy),))
    centers = Points(unit_x=TVFS.unit_x, unit_y=TVFS.unit_y,
                     unit_v=TVFS.unit_times)
    # computing with vortex center
    if output_center:
        for i, pt in enumerate(traj):
            # getting time and associated velocity field
            time = traj.v[i]
            field = TVFS.fields[TVFS.times == time][0]
            # getting radius and center
            rad, cent = get_vortex_radius(field, traj.xy[i],
                                          gamma2_radius=gamma2_radius,
                                          output_center=True)
            radii[i] = rad
            centers.add(cent, time)
    # computing without vortex centers
    else:
        for i, pt in enumerate(traj):
            # getting time and associated velocity field
            time = pt.v
            field = TVFS.fields[TVFS.times == time][0]
            # getting radius
            radii[i] = get_vortex_radius(field, traj.xy[i],
                                         gamma2_radius=gamma2_radius,
                                         output_center=False)
    # returning
    radii_prof = Profile(traj.v, radii, mask=False, unit_x=TVFS.unit_times,
                         unit_y=TVFS.unit_x)
    if output_center:
        return radii_prof, centers
    else:
        return radii_prof


def get_vortex_circulation(VF, vort_center, epsilon=0.1, output_unit=False):
    """
    Return the circulation of the given vortex.

    $\Gamma = \int_S \omega dS$
    avec : $S$ : surface su vortex ($| \omega | > \epsilon$)

    Recirculation is representative of the swirling strength.

    Warning : integral on complex domain is complex (you don't say?),
    here is just implemented a sum of accessible values on the domain.

    Parameters:
    -----------
    VF : vectorfield object
        Velocity field on which compute gamma2.
    vort_center : 2x1 array
        Approximate position of the vortex center.
    epsilon : float, optional
        seuil for the vorticity integral (default is 0.1).
    output_unit : boolean, optional
        If 'True', circulation unit is returned.

    Returns :
    ---------
    circ : float
        Vortex virculation.
    """
    # getting data
    ind_x = VF.get_indice_on_axe(1, vort_center[0], nearest=True)
    ind_y = VF.get_indice_on_axe(2, vort_center[1], nearest=True)
    dx = VF.axe_x[1] - VF.axe_x[0]
    dy = VF.axe_y[1] - VF.axe_y[0]
    import IMTreatment.field_treatment as imtft
    vort = imtft.get_vorticity(VF)
    # find omega > 0.1 zones and label them
    vort_zone = np.abs(vort.values) > epsilon
    vort_zone, nmb_zone = msr.label(vort_zone)
    # get wanted zone label
    lab = vort_zone[ind_x, ind_y]
    # if we are outside a zone
    if lab == 0:
        if output_unit:
            return 0., make_unit("")
        else:
            return 0.
    # else, we compute the circulation
    circ = np.sum(vort.values[vort_zone == lab])*dx*dy
    # if necessary, we compute the unit
    unit_circ = vort.unit_values*VF.unit_x*VF.unit_y
    circ *= unit_circ.asNumber()
    unit_circ /= unit_circ.asNumber()
    # returning
    plt.figure()
    vort.display()
    plt.plot(VF.axe_x[ind_x], VF.axe_y[ind_y], 'o')
    plt.figure()
    plt.imshow(vort_zone == lab, interpolation='nearest')
    if output_unit:
        return circ, unit_circ
    else:
        return circ


#### Critical points ###
#def get_cp_traj(TVFS, epsilon=None, kind='crit', window_size=4):
#    """
#    For a set of velocity field (TemporalVectorFields object), return the
#    trajectory of critical points.
#    If the number of points returned is low, you should smooth or filter your
#    field (POD filtering for example).
#
#    Parameters
#    ----------
#    TVFS : TemporalVectorFields object
#        .
#    epsilon : float, optional
#        Maximum length between two consecutive points in trajectory.
#        (default is Inf), extremely usefull to put a correct value here.
#    kind : string, optional
#        If 'pbi', return cp position given by PBI algorithm.
#        If 'crit' (default), return cp position given by PBI + criterion.
#        (more accurate, but slower).
#    window_size : integer, optional
#        Minimal window size for PBI detection.
#        Smaller window size allow detection where points are dense.
#        Default is 4 (smallest is 2).
#
#    Returns
#    -------
#    focus_traj : tuple of Points objects
#        Rotative focus trajectories
#    focus_c_traj : tuple of Points objects
#        Contrarotative focus trajectories
#    nodes_i_traj : tuple of Points objects
#        In nodes trajectories
#    nodes_o_traj : tuple of Points objects
#        Out nodes trajectories
#    saddle_pts : tuple of Points objects
#        Saddle points trajectories
#    pbi_p : tuple of Points objects
#        Points get by PBI algorithm (with pbi=1)
#    pbi_m : tuple of Points objects
#        Points get by PBI algorithm (with pbi=-1)
#    """
#    # check parameters coherence
#    if not isinstance(TVFS, TemporalVectorFields):
#        raise TypeError("'TVFS' must be a TemporalVectorFields")
#    if epsilon is not None:
#        if not isinstance(epsilon, NUMBERTYPES):
#            raise TypeError("'epsilon' must be a positive real")
#        if epsilon < 0:
#            raise ValueError("'epsilon' must be a positive real")
#    if not isinstance(kind, STRINGTYPES):
#        raise TypeError("'kind' must be a string")
#    # TODO : Add warning when masked values are present (or not ?)
#    # create storage arrays
#    focus = []
#    focus_c = []
#    nodes_i = []
#    nodes_o = []
#    saddles = []
#    cp_pbi_p = []
#    cp_pbi_m = []
#    # getting first critical points for all fields
#    times = TVFS.times
#    for i, field in enumerate(TVFS.fields):
#        if kind == 'crit':
#            foc, foc_c, nod_i, nod_o, sadd, pbi \
#                = get_cp_crit(field, times[i], window_size=window_size)
#            pbi_p = [pbi_pt for pbi_pt in pbi if pbi_pt.pbi == 1]
#            pbi_m = [pbi_pt for pbi_pt in pbi if pbi_pt.pbi == 0]
#        elif kind == 'pbi':
#            pos, pbis = get_cp_pbi(field, times[i], window_size=window_size)
#            foc, foc_c, nod_i, nod_o, sadd = [], [], [], [], []
#            pbi_p, pbi_m = [], []
#            for j, pt in enumerate(pos):
#                tmp_pt = Points([pt], [times[i]])
#                tmp_pt.pbi = pbis[j]
#                if pbis[j] == 1:
#                    pbi_p.append(tmp_pt)
#                else:
#                    pbi_m.append(tmp_pt)
#        else:
#            raise ValueError()
#        # Concatenate result of each field in bigger entities
#        if len(foc) != 0:
#            focus += foc
#        if len(foc_c) != 0:
#            focus_c += foc_c
#        if len(nod_i) != 0:
#            nodes_i += nod_i
#        if len(nod_o) != 0:
#            nodes_o += nod_o
#        if len(sadd) != 0:
#            saddles += sadd
#        if len(pbi_p) != 0:
#            cp_pbi_p += pbi_p
#        if len(pbi_m) != 0:
#            cp_pbi_m += pbi_m
#    # getting times
#    times = TVFS.times
#    # getting critical points trajectory for eachkind of point
#    if len(focus) != 0:
#        focus_traj = get_cp_time_evolution(focus, times, epsilon)
#    else:
#        focus_traj = []
#    if len(focus_c) != 0:
#        focus_c_traj = get_cp_time_evolution(focus_c, times, epsilon)
#    else:
#        focus_c_traj = []
#    if len(nodes_i) != 0:
#        nodes_i_traj = get_cp_time_evolution(nodes_i, times, epsilon)
#    else:
#        nodes_i_traj = []
#    if len(nodes_o) != 0:
#        nodes_o_traj = get_cp_time_evolution(nodes_o, times, epsilon)
#    else:
#        nodes_o_traj = []
#    if len(saddles) != 0:
#        saddles_traj = get_cp_time_evolution(saddles, times, epsilon)
#    else:
#        saddles_traj = []
#    if len(cp_pbi_p) != 0:
#        cp_pbip_traj = get_cp_time_evolution(cp_pbi_p, times, epsilon)
#    else:
#        cp_pbip_traj = []
#    if len(cp_pbi_m) != 0:
#        cp_pbim_traj = get_cp_time_evolution(cp_pbi_m, times, epsilon)
#    else:
#        cp_pbim_traj = []
#    # returning result
#    return focus_traj, focus_c_traj, nodes_i_traj, nodes_o_traj, \
#        saddles_traj,cp_pbim_traj, cp_pbip_traj

def get_cp_pbi_on_TVF(TVF, window_size=4):
    """
    For a TemporalVectorField object, return the critical points positions and
    their PBI (Poincarre Bendixson indice).

    Parameters
    ----------
    TVF : a TemporalVectorField object.
        .
    window_size : integer, optional
        Minimal window size for PBI detection.
        Smaller window size allow detection where points are dense.
        Default is 4 (smallest is 2).

    Returns
    -------
    pts : CritPoints object
        Containing all critical points position
    """
    cp = CritPoints(unit_time=TVF.unit_times)
    for i, field in enumerate(TVF.fields):
        cp += get_cp_pbi_on_VF(field, time=TVF.times[i],
                               unit_time=TVF.unit_times,
                               window_size=window_size)
    return cp


def get_cp_pbi_on_VF(vectorfield, time=0, unit_time=make_unit(""),
                     window_size=4):
    """
    For a VectorField object, return the critical points positions and their
    PBI (Poincarre Bendixson indice)

    Parameters
    ----------
    vectorfield : a VectorField object.
        .
    time : number, optional
        Time
    unit_time : units object, optional
        Time unit.
    window_size : integer, optional
        Minimal window size for PBI detection.
        Smaller window size allow detection where points are dense.
        Default is 4 (smallest is 2).

    Returns
    -------
    pts : CritPoints object
        Containing all critical points position
    """
    # checking parameters coherence
    if not isinstance(vectorfield, VectorField):
        raise TypeError("'vectorfield' must be a VectorField")
    # using VF methods to get cp position
    field = velocityfield_to_vf(vectorfield, time)
    pos, pbis = field.get_cp_position(window_size=window_size)
    pbi_m = Points()
    pbi_p = Points()
    for i, pbi in enumerate(pbis):
        if pbi == -1:
            pbi_m.add(pos[i])
        elif pbi == 1:
            pbi_p.add(pos[i])
        else:
            raise Exception()
    pts = CritPoints(unit_time=unit_time)
    pts.add_point(pbi_m=pbi_m, pbi_p=pbi_p, time=time)
    return pts


def get_cp_crit_on_TVF(TVF, window_size=4):
    """
    For a TemporalVectorField object, return the position of critical points.
    This algorithm use the PBI algorithm, then a method based on criterion to
    give more accurate results.

    Parameters
    ----------
    TVF : a TemporalVectorField object.
        .
    window_size : integer, optional
        Minimal window size for PBI detection.
        Smaller window size allow detection where points are dense.
        Default is 4 (smallest is 2).

    Returns
    -------
    pts : CritPoints object
        Containing all critical points
    """
    cp = CritPoints(unit_time=TVF.unit_times)
    for i, field in enumerate(TVF.fields):
        cp += get_cp_crit_on_VF(field, time=TVF.times[i],
                                unit_time=TVF.unit_times,
                                window_size=window_size)
    return cp


def get_cp_crit_on_VF(vectorfield, time=0, unit_time=make_unit(""),
                      window_size=4):
    """
    For a VectorField object, return the position of critical points.
    This algorithm use the PBI algorithm, then a method based on criterion to
    give more accurate results.

    Parameters
    ----------
    vectorfield : a VectorField object.
        .
    time : number, optional
        Time
    unit_time : units object, optional
        Time unit.
    window_size : integer, optional
        Minimal window size for PBI detection.
        Smaller window size allow detection where points are dense.
        Default is 4 (smallest is 2).

    Returns
    -------
    pts : CritPoints object
        Containing all critical points

    Notes
    -----
    In the CritPoints object, the saddles points (CritPoints.sadd) have
    associated principal orientations (CritPoints.sadd[0].orientations).
    These orientations are the eigenvectors of the velocity jacobian.
    """
    if not isinstance(vectorfield, VectorField):
        raise TypeError("'VF' must be a VectorField")
    ### Getting pbi cp position and fields around ###
    VF_field = velocityfield_to_vf(vectorfield, time)
    cp_positions, pbis = VF_field.get_cp_position(window_size=window_size)
    # creating velocityfields around critical points
    # and transforming into VectorField objects
    VF_tupl = []
    pbi_m = Points()
    pbi_p = Points()
    for i, cp_pos in enumerate(cp_positions):
        tmp_vf = VF_field.get_field_around_pt(cp_pos, 5)
        tmp_vf = tmp_vf.export_to_velocityfield()
        # treating small fields
        axe_x, axe_y = tmp_vf.axe_x, tmp_vf.axe_y
        if len(axe_x) < 6 or len(axe_y) < 6 or np.any(tmp_vf.mask):
            if pbis[i] == -1:
                pbi_m += Points([cp_pos])
            elif pbis[i] == 1:
                pbi_p += Points([cp_pos])
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
            max_gam = np.max([abs(VF.gamma1.max),
                             abs(VF.gamma1.min)])
            max_kap = np.max([abs(VF.kappa1.max),
                             abs(VF.kappa1.min)])
            if max_gam > max_kap:
                VF_focus.append(VF)
            else:
                VF_nodes.append(VF)
        # saddle point
        elif VF.PBI == -1:
            VF_saddle.append(VF)
    ### Computing saddle points positions and direction###
    saddles = Points()
    if len(VF_saddle) != 0:
        for VF in VF_saddle:
            tmp_dev = get_angle_deviation(VF, 1.9, ind=True)
            pts = _min_detection(1 - tmp_dev)
            if pts is not None:
                if pts.xy[0][0] is not None and len(pts) == 1:
                    saddles += pts
    # compute directions
    saddles_ori = []
    for sad in saddles.xy:
        saddles_ori.append(np.array(_get_saddle_orientations(vectorfield,
                                                             sad)))
    tmp_opts = OrientedPoints()
    tmp_opts.import_from_Points(saddles, saddles_ori)
    saddles = tmp_opts

    ### Computing focus positions (rotatives and contrarotatives) ###
    focus = Points()
    focus_c = Points()
    if len(VF_focus) != 0:
        for VF in VF_focus:
            tmp_gam = VF.gamma1
            min_gam = tmp_gam.min
            max_gam = tmp_gam.max
            # rotative vortex
            if abs(max_gam) > abs(min_gam):
                pts = _min_detection(-1.*tmp_gam)
                if pts is not None:
                    if pts.xy[0][0] is not None and len(pts) == 1:
                        focus += pts
            # contrarotative vortex
            else:
                pts = _min_detection(tmp_gam)
                if pts is not None:
                    if pts.xy[0][0] is not None and len(pts) == 1:
                        focus_c += pts
    ### Computing nodes points positions (in or out) ###
    nodes_i = Points()
    nodes_o = Points()
    if len(VF_nodes) != 0:
        for VF in VF_nodes:
            tmp_kap = VF.kappa1
            min_kap = tmp_kap.min
            max_kap = tmp_kap.max
            # out nodes
            if abs(max_kap) > abs(min_kap):
                pts = _min_detection(-1.*tmp_kap)
                if pts is not None:
                    if pts.xy[0][0] is not None and len(pts) == 1:
                        nodes_o += pts
            # in nodes
            else:
                pts = _min_detection(tmp_kap)
                if pts is not None:
                    if pts.xy[0][0] is not None and len(pts) == 1:
                        nodes_i += pts
    ### creating the CritPoints object for returning
    pts = CritPoints(unit_time=unit_time)
    pts.add_point(focus, focus_c, nodes_i, nodes_o, saddles,
                  pbi_m, pbi_p, time=time)
    return pts


def _min_detection(SF):
    """
    Only for use in 'get_cp_crit'.
    """
    # interpolation on the field
    if np.any(SF.mask):
        SF.crop_masked_border()
        if np.any(SF.mask):
            raise Exception("should not have masked values")
    axe_x, axe_y = SF.axe_x, SF.axe_y
    values = SF.values
    interp = RectBivariateSpline(axe_x, axe_y, values, s=0, ky=3, kx=3)
    # extended field (resolution x100)
    x = np.linspace(axe_x[0], axe_x[-1], 100)
    y = np.linspace(axe_y[0], axe_y[-1], 100)
    values = interp(x, y)
    ind_min = np.argmin(values)
    ind_x, ind_y = np.unravel_index(ind_min, values.shape)
    pos = (x[ind_x], y[ind_y])
    return Points([pos])


def _gaussian_fit(SF):
    """
    Only for use in 'get_cp_crit'.
    """
    # gaussian fitting
    def gaussian(height, center_x, center_y, width_x, width_y):
        """
        Returns a gaussian function with the given parameters
        """
        width_x = float(width_x)
        width_y = float(width_y)
        return lambda x, y: height*np.exp(
            -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

    def moments(data):
        """
        Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution by calculating its
        moments
        """
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
        """
        Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution found by a fit
        """
        params = moments(data)
        errorfunction = lambda p: np.ravel(gaussian(*p)
                                           (*np.indices(data.shape)) -
                                           data)
        p, success = optimize.leastsq(errorfunction, params)
        return p
    axe_x, axe_y = SF.axe_x, SF.axe_y
    values = SF.values
    params = fitgaussian(values)
    delta_x = axe_x[1] - axe_x[0]
    delta_y = axe_y[1] - axe_y[0]
    x = SF.axe_x[0] + delta_x*params[1]
    y = SF.axe_y[0] + delta_y*params[2]
    return Points([(x, y)])


def _get_saddle_orientations(vectorfield, pt):
    """
    Return the orientation of a saddle point.
    """
    # get data
    axe_x, axe_y = vectorfield.axe_x, vectorfield.axe_y
    dx = vectorfield.axe_x[1] - vectorfield.axe_x[0]
    dy = vectorfield.axe_y[1] - vectorfield.axe_y[0]
    Vx = vectorfield.comp_x
    Vy = vectorfield.comp_y
    # get surrounding gradients
    inds_x = vectorfield.get_indice_on_axe(1, pt[0], kind='bounds')
    inds_y = vectorfield.get_indice_on_axe(2, pt[1], kind='bounds')
    inds_x_2 = np.arange(-1, 3) + inds_x[0]
    inds_y_2 = np.arange(-1, 3) + inds_y[0]
    inds_X_2, inds_Y_2 = np.meshgrid(inds_x_2, inds_y_2)
    local_Vx = Vx[inds_X_2, inds_Y_2]
    local_Vy = Vy[inds_X_2, inds_Y_2]
    Vx_dx, Vx_dy = np.gradient(local_Vx.transpose(), dx, dy)
    Vy_dx, Vy_dy = np.gradient(local_Vy.transpose(), dx, dy)
    # get interpolated gradient at the point
    axe_x2 = axe_x[inds_x_2]
    axe_y2 = axe_y[inds_y_2]
    Vx_dx2 = RectBivariateSpline(axe_x2, axe_y2, Vx_dx, kx=2, ky=2, s=0)
    Vx_dx2 = Vx_dx2(pt[0], pt[1])[0][0]
    Vx_dy2 = RectBivariateSpline(axe_x2, axe_y2, Vx_dy, kx=2, ky=2, s=0)
    Vx_dy2 = Vx_dy2(pt[0], pt[1])[0][0]
    Vy_dx2 = RectBivariateSpline(axe_x2, axe_y2, Vy_dx, kx=2, ky=2, s=0)
    Vy_dx2 = Vy_dx2(pt[0], pt[1])[0][0]
    Vy_dy2 = RectBivariateSpline(axe_x2, axe_y2, Vy_dy, kx=2, ky=2, s=0)
    Vy_dy2 = Vy_dy2(pt[0], pt[1])[0][0]
    # get jacobian eignevalues
    jac = np.array([[Vx_dx2, Vx_dy2], [Vy_dx2, Vy_dy2]], subok=True)
    eigvals, eigvects = np.linalg.eig(jac)
    if eigvals[0] < eigvals[1]:
        orient1 = eigvects[:, 0]
        orient2 = eigvects[:, 1]
    else:
        orient1 = eigvects[:, 1]
        orient2 = eigvects[:, 0]
    return np.real(orient1), np.real(orient2)


### Separation point ###
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
    obj : ScalarField, VectorField, VectorField or TemporalVelocityField
        If 'VectorField' or 'VectorField', wall_direction is used to
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
    if not isinstance(obj, (ScalarField, VectorField, VectorField,
                            TemporalVectorFields)):
        raise TypeError("Unknown type for 'obj' : {}".format(type(obj)))
    if not isinstance(wall_direction, NUMBERTYPES):
        raise TypeError("'wall_direction' must be a number")
    if wall_direction != 1 and wall_direction != 2:
        raise ValueError("'wall_direction' must be 1 or 2")
    if not isinstance(wall_position, NUMBERTYPES):
        raise ValueError("'wall_position' must be a number")
    axe_x, axe_y = obj.axe_x, obj.axe_y
    if interval is None:
        if wall_direction == 2:
            interval = [np.min(axe_x), np.max(axe_x)]
        else:
            interval = [np.min(axe_y), np.max(axe_y)]
    if not isinstance(interval, ARRAYTYPES):
        raise TypeError("'interval' must be a array")
    # checking 'obj' type
    if isinstance(obj, ScalarField):
        # checking masked values
        if np.any(obj.mask):
            raise Warning("I can give weird results if masked values remains")
        V = obj.values_as_sf
        if wall_direction == 1:
            axe = axe_x
        else:
            axe = axe_y
    elif isinstance(obj, VectorField):
        if np.any(obj.mask):
            raise Warning("I can give weird results if masked values remains")
        if wall_direction == 1:
            V = obj.comp_y_as_sf
            axe = axe_x
        else:
            V = obj.comp_x_as_sf
            axe = axe_y
    elif isinstance(obj, TemporalVectorFields):
        if np.any(obj.fields[0].mask):
            raise Warning("I can give weird results if masked values remains")
        pts = []
        times = obj.times
        if wall_direction == 1:
            unit_axe = obj.unit_y
        else:
            unit_axe = obj.unit_x
        for field in obj.fields:
            pts.append(get_separation_position(field,
                                               wall_direction=wall_direction,
                                               wall_position=wall_position,
                                               interval=interval))
        return Profile(times, pts, unit_x=obj.unit_times,
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


### Critical lines ###
def get_critical_line(VF, source_point, direction, kol='stream',
                      delta=1, fit='none', order=2):
    """
    Return a parametric curve fitting the virtual streamlines expanding from
    the 'source_point' critical point on the 'VF' field.

    Parameters
    ----------
    VF : VectorField object
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
    if not isinstance(VF, VectorField):
        raise TypeError("'VF' must be a VectorField object")
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
    axe_x, axe_y = VF.axe_x, VF.axe_y
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
        lines = get_streamlines(VF, pts)
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


### Angle based criterion ###
def get_angle_deviation(vectorfield, radius=None, ind=False, mask=None,
                        raw=False, local_treatment='none', order=1):
    """
    Return the angle deviation field.

    Parameters
    ----------
    vectorfield : VectorField object
        .
    radius : number, optionnal
        The radius used to choose the zone where to compute
        for each field oint. If not mentionned, a value is choosen in
        ordre to have about 8 points in the circle. It allow to get good
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
    local_treatment : string, optional
        If 'None' (default), angles are taken directly from the velocity field
        If 'galilean_inv', angles are taken from localy averaged velocity field
        if 'local', angles are taken from velocity fields where the velocity of
        the central point is localy substracted.
    order : number, optional
        Order used to compute the deviation
        (default 1 for sum of differences, 2 for standart deviation (std)
        or more)
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
    axe_x, axe_y = vectorfield.axe_x, vectorfield.axe_y
    if mask is None:
        mask = np.zeros(vectorfield.shape)
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'zone' must be an array of boolean")
    else:
        mask = np.array(mask)
    if not isinstance(raw, bool):
        raise TypeError()
    if not isinstance(local_treatment, STRINGTYPES):
        raise TypeError()
    if not local_treatment in ['none', 'galilean_inv', 'local']:
        raise ValueError()
    if not isinstance(order, NUMBERTYPES):
        raise TypeError()
    if order < 1:
        raise ValueError()
    ### Getting data ###
    theta = vectorfield.theta
    mask, nmbpts, mask_dev, mask_border, mask_surr, motif =\
        _non_local_criterion_precomputation(vectorfield, mask, radius, ind,
                                            dev_pass=False)
    ### Computing criterion ###
    # creating reference dispersion functions
    best_fun = np.array([2.*np.pi/nmbpts]*nmbpts)
    worse_fun = np.array([0]*(nmbpts - 1) + [2.*np.pi])
    worse_value = (np.sum(np.abs(worse_fun - best_fun)**order))**(1./order)
    # Loop on points
    deviation = np.zeros(vectorfield.shape)
    for inds, pos, _ in vectorfield:
        ind_x = inds[0]
        ind_y = inds[1]
        # stop if masked or on border or with a masked surrouinding point
        if mask[ind_x, ind_y] or mask_surr[ind_x, ind_y]\
                or mask_border[ind_x, ind_y]:
            continue
        # getting neighbour points
        indsaround = motif + inds
        # getting neighbour angles  (galilean inv or not)
        if local_treatment == 'none':
            angles = theta[indsaround[:, 0], indsaround[:, 1]]
        elif local_treatment == 'galilean_inv':
            tmp_Vx = vectorfield.comp_x[indsaround[:, 0], indsaround[:, 1]]
            tmp_Vy = vectorfield.comp_y[indsaround[:, 0], indsaround[:, 1]]
            tmp_Vx -= np.mean(tmp_Vx)
            tmp_Vy -= np.mean(tmp_Vy)
            angles = _get_angles(tmp_Vx, tmp_Vy)
        elif local_treatment == 'local':
            tmp_Vx = vectorfield.comp_x[indsaround[:, 0], indsaround[:, 1]]
            tmp_Vy = vectorfield.comp_y[indsaround[:, 0], indsaround[:, 1]]
            tmp_Vx -= vectorfield.comp_x[ind_x, ind_y]
            tmp_Vy -= vectorfield.comp_x[ind_x, ind_y]
            angles = _get_angles(tmp_Vx, tmp_Vy)
        # getting neightbour angles repartition
        angles = np.sort(angles)
        d_angles = np.empty(angles.shape)
        d_angles[0:-1] = angles[1::] - angles[:-1:]
        d_angles[-1] = angles[0] + 2*np.pi - angles[-1]
        # getting neighbour angles deviation
        deviation[ind_x, ind_y] = (1 - (np.sum(np.abs(d_angles - best_fun)
                                               ** order))
                                   ** (1./order)/worse_value)
    ### Applying masks ###
    mask = np.logical_or(mask, mask_border)
    mask = np.logical_or(mask, mask_surr)
    ### Creating gamma ScalarField ###
    if raw:
        return np.ma.masked_array(deviation, mask)
    else:
        deviation_sf = ScalarField()
        unit_x, unit_y = vectorfield.unit_x, vectorfield.unit_y
        deviation_sf.import_from_arrays(axe_x, axe_y, deviation, mask,
                                        unit_x=unit_x, unit_y=unit_y,
                                        unit_values=make_unit(''))
        return deviation_sf


def get_gamma(vectorfield, radius=None, ind=False, kind='gamma1', mask=None,
              raw=False, dev_pass=True):
    """
    Return the gamma scalar field. Gamma criterion is used in
    vortex analysis.
    The fonction recognize if the field is ortogonal, and use an
    apropriate algorithm.

    Parameters
    ----------
    vectorfield : VectorField object
        .
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
        If 'gamma1b', compute gamma1 criterion with velocity corrector.
        (multiply with the mean velocity)
        If 'gamma2', compute gamma2 criterion (with relative velocities)
        If 'gamma2b', compute gamma2 criterion with a velocity corrector.
        (hide uniform velocity zone)
    mask : array of boolean, optionnal
        Has to be an array of the same size of the vector field object,
        gamma will be compute only where mask is 'False'.
    raw : boolean, optional
        If 'False' (default), a ScalarField is returned,
        if 'True', an array is returned.
    dev_pass : boolean, optional
        If 'True', the algorithm compute gamma criterion only where the
        velocity angles deviation is strong (faster if there is few points)
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
    axe_x, axe_y = vectorfield.axe_x, vectorfield.axe_y
    if not isinstance(kind, STRINGTYPES):
        raise TypeError("'kind' must be a string")
    if not kind in ['gamma1', 'gamma2', 'gamma1b', 'gamma2b']:
        raise ValueError("Unkown value for kind")
    if mask is None:
        mask = np.zeros(vectorfield.shape)
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'zone' must be an array of boolean")
    else:
        mask = np.array(mask)
    # getting data and masks
    Vx = vectorfield.comp_x
    Vy = vectorfield.comp_y
    norm_v = vectorfield.magnitude
    mask, nmbpts, mask_dev, mask_border, mask_surr, motif =\
        _non_local_criterion_precomputation(vectorfield, mask, radius, ind,
                                            dev_pass)
    # getting the vectors between center and neighbouring
    deltax = axe_x[1] - axe_x[0]
    deltay = axe_y[1] - axe_y[0]
    vector_a_x = np.zeros(motif.shape[0])
    vector_a_y = np.zeros(motif.shape[0])
    for i, indaround in enumerate(motif):
        vector_a_x[i] = indaround[0]*deltax
        vector_a_y[i] = indaround[1]*deltay
    norm_vect_a = (vector_a_x**2 + vector_a_y**2)**(.5)
    ### Loop on points ###
    gammas = np.zeros(vectorfield.shape)
    for inds, pos, _ in vectorfield:
        ind_x = inds[0]
        ind_y = inds[1]
        # stop if masked or on border or with a masked surrouinding point
        if mask[ind_x, ind_y] or mask_surr[ind_x, ind_y]\
                or mask_border[ind_x, ind_y] or mask_dev[ind_x, ind_y]:
            continue
        # getting neighbour points
        indsaround = motif + inds
        # If necessary, compute mean velocity on points (gamma2)
        v_mean = [0., 0.]
        v_mean2 = [0., 0.]
        fact = 1
        if kind in ['gamma1b', 'gamma2', 'gamma2b']:
            v_mean[0] = np.mean(Vx[indsaround[:, 0], indsaround[:, 1]])
            v_mean[1] = np.mean(Vy[indsaround[:, 0], indsaround[:, 1]])
        if kind in ['gamma2b']:
            v_mean2[0] = np.mean((Vx[indsaround[:, 0], indsaround[:, 1]]
                                 - v_mean[0])**2)
            v_mean2[1] = np.mean((Vy[indsaround[:, 0], indsaround[:, 1]]
                                 - v_mean[1])**2)
            fact = np.sqrt(v_mean2[0] + v_mean2[1]) / \
                np.sqrt(v_mean[0]**2 + v_mean[1]**2)
            if np.abs(fact) > 1:
                fact = 1.
        ### Loop on neighbouring points ###
        gamma = 0.
        for i, indaround in enumerate(indsaround):
            inda_x = indaround[0]
            inda_y = indaround[1]
            # getting vectors for scalar product
            if kind in ['gamma1', 'gamma1b']:
                vector_b_x = Vx[inda_x, inda_y]
                vector_b_y = Vy[inda_x, inda_y]
                denom = norm_v[inda_x, inda_y]*norm_vect_a[i]
                if denom != 0:
                    gamma += (vector_a_x[i]*vector_b_y
                              - vector_a_y[i]*vector_b_x)/denom
            elif kind in ['gamma2', 'gamma2b']:
                vector_b_x = Vx[inda_x, inda_y] - v_mean[0]
                vector_b_y = Vy[inda_x, inda_y] - v_mean[1]
                denom = (vector_b_x**2 + vector_b_y**2)**.5*norm_vect_a[i]
                if denom != 0:
                    gamma += (vector_a_x[i]*vector_b_y
                              - vector_a_y[i]*vector_b_x)/denom
        # adapting with factors
        if kind in ['gamma1', 'gamma2']:
            gamma = gamma/nmbpts
        elif kind == 'gamma1b':
            gamma = gamma/nmbpts*np.sqrt(v_mean[0]**2 + v_mean[1]**2)
        elif kind == 'gamma2b':
            gamma = gamma/nmbpts*fact
        # storing computed gamma value
        gammas[ind_x, ind_y] = gamma
    ### Applying masks ###
    mask = np.logical_or(mask, mask_border)
    mask = np.logical_or(mask, mask_surr)
    ### Creating gamma ScalarField ###
    if raw:
        return np.ma.masked_array(gammas, mask)
    else:
        gamma_sf = ScalarField()
        unit_x, unit_y = vectorfield.unit_x, vectorfield.unit_y
        gamma_sf.import_from_arrays(axe_x, axe_y, gammas, mask,
                                    unit_x=unit_x, unit_y=unit_y,
                                    unit_values=make_unit(''))
        return gamma_sf


def get_kappa(vectorfield, radius=None, ind=False, kind='kappa1', mask=None,
              raw=False, dev_pass=False):
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
    dev_pass : boolean, optional
        If 'True', the algorithm compute gamma criterion only where the
        velocity angles deviation is strong (faster if there is few points)
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
    axe_x, axe_y = vectorfield.axe_x, vectorfield.axe_y
    if not isinstance(kind, STRINGTYPES):
        raise TypeError("'kind' must be a string")
    if not kind in ['kappa1', 'kappa2']:
        raise ValueError("Unkown value for kind")
    if mask is None:
        mask = np.zeros(vectorfield.shape)
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'zone' must be an array of boolean")
    else:
        mask = np.array(mask)
    # getting data and masks
    Vx = vectorfield.comp_x
    Vy = vectorfield.comp_y
    norm_v = vectorfield.magnitude
    mask, nmbpts, mask_dev, mask_border, mask_surr, motif =\
        _non_local_criterion_precomputation(vectorfield, mask, radius, ind,
                                            dev_pass)
    # getting the vectors between center and neighbouring
    deltax = axe_x[1] - axe_x[0]
    deltay = axe_y[1] - axe_y[0]
    vector_a_x = np.zeros(motif.shape[0])
    vector_a_y = np.zeros(motif.shape[0])
    for i, indaround in enumerate(motif):
        vector_a_x[i] = indaround[0]*deltax
        vector_a_y[i] = indaround[1]*deltay
    norm_vect_a = (vector_a_x**2 + vector_a_y**2)**(.5)
    ### Loop on points ###
    kappas = np.zeros(vectorfield.shape)
    for inds, pos, _ in vectorfield:
        ind_x = inds[0]
        ind_y = inds[1]
        # stop if masked or on border or with a masked surrouinding point
        if mask[ind_x, ind_y] or mask_surr[ind_x, ind_y]\
                or mask_border[ind_x, ind_y] or mask_dev[ind_x, ind_y]:
            continue
        # getting neighbour points
        indsaround = motif + np.array(inds)
        # If necessary, compute mean velocity on points (kappa2)
        v_mean = [0., 0.]
        if kind == 'kappa2':
            nmbpts = len(indsaround)
            for indaround in indsaround:
                v_mean[0] += Vx[ind_x, ind_y]
                v_mean[1] += Vy[ind_x, ind_y]
            v_mean[0] /= nmbpts
            v_mean[1] /= nmbpts
        ### Loop on neighbouring points ###
        kappa = 0.
        nmbpts = len(indsaround)
        for i, indaround in enumerate(indsaround):
            inda_x = indaround[0]
            inda_y = indaround[1]
            # getting vectors for scalar product
            vector_b_x = Vx[inda_x, inda_y] - v_mean[0]
            vector_b_y = Vy[inda_x, inda_y] - v_mean[1]
            if kind == 'kappa1':
                denom = norm_v[inda_x, inda_y]*norm_vect_a[i]
            else:
                denom = (vector_b_x**2 + vector_b_y**2)**.5*norm_vect_a[i]
            # getting scalar product
            if denom != 0:
                kappa += (vector_a_x[i]*vector_b_x
                          + vector_a_y[i]*vector_b_y)/denom
        # storing computed kappa value
        kappas[ind_x, ind_y] = kappa/nmbpts
    ### Applying masks ###
    mask = np.logical_or(mask, mask_border)
    mask = np.logical_or(mask, mask_surr)
    ### Creating kappa ScalarField ###
    if raw:
        return np.ma.masked_array(kappas, mask)
    else:
        kappa_sf = ScalarField()
        unit_x, unit_y = vectorfield.unit_x, vectorfield.unit_y
        kappa_sf.import_from_arrays(axe_x, axe_y, kappas, mask,
                                    unit_x=unit_x, unit_y=unit_y,
                                    unit_values=make_unit(''))
        return kappa_sf


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
        mask = np.zeros(vectorfield.shape)
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'mask' must be an array of boolean")
    else:
        mask = np.array(mask)
    axe_x, axe_y = vectorfield.axe_x, vectorfield.axe_y
    # récupération de theta et de son masque
    theta = vectorfield.theta
    mask = np.logical_or(mask, vectorfield.mask)
    # récupération des dx et dy
    dx = np.abs(axe_x[0] - axe_x[2])
    dy = np.abs(axe_y[0] - axe_y[2])
    # boucle sur les points
    grad_theta_m = np.zeros(vectorfield.shape)
    maskf = mask.copy()
    for inds, _, _ in vectorfield:
        ind_x = inds[0]
        ind_y = inds[1]
        # arrete si on est sur un bords ou sur une valeurs masquée
        if (ind_x == 0 or ind_y == 0 or ind_y == len(axe_y)-1
                or ind_x == len(axe_x)-1):
            maskf[ind_x, ind_y] = True
            continue
        if np.any(mask[ind_x-1:ind_x+2, ind_y-1:ind_y+2]):
            maskf[ind_x, ind_y] = True
            continue
        # calcul de theta_m autours
        theta_m = [[theta[ind_x - 1, ind_y - 1] + 3./4*np.pi,
                   theta[ind_x, ind_y - 1] + np.pi/2,
                   theta[ind_x + 1, ind_y - 1] + 1./4*np.pi],
                  [theta[ind_x - 1, ind_y] + np.pi,
                   0.,
                   theta[ind_x + 1, ind_y]],
                  [theta[ind_x - 1, ind_y + 1] - 3./4*np.pi,
                   theta[ind_x, ind_y + 1] - np.pi/2,
                   theta[ind_x + 1, ind_y + 1] - np.pi/4]]
        theta_m = np.mod(theta_m, np.pi*2)
        ### TODO : Warning, ot really the formula from article !!!
        sin_theta = np.sin(theta_m)
        #cos_theta = np.cos(theta_m)
        grad_x = -(sin_theta[2, 1] - sin_theta[0, 1])/dx
        grad_y = -(sin_theta[1, 2] - sin_theta[1, 0])/dy
        # calcul de gradthetaM
        grad_theta_m[ind_x, ind_y] = (grad_x**2 + grad_y**2)**(1./2)
    # application du masque
    if raw:
        return np.ma.masked_array(grad_theta_m, maskf)
    else:
        iota_sf = ScalarField()
        unit_x, unit_y = vectorfield.unit_x, vectorfield.unit_y
        iota_sf.import_from_arrays(axe_x, axe_y, grad_theta_m, maskf,
                                   unit_x=unit_x, unit_y=unit_y,
                                   unit_values=make_unit(''))
        return iota_sf


def _non_local_criterion_precomputation(vectorfield, mask, radius, ind,
                                        dev_pass):
    """
    """
    ### Importing data from vectorfield (velocity, axis and mask) ###
    mask = np.logical_or(mask, vectorfield.mask)
    ### Compute motif and motif angles on an arbitrary point ###
    axe_x, axe_y = vectorfield.axe_x, vectorfield.axe_y
    indcentral = [int(len(axe_x)/2.), int(len(axe_y)/2.)]
    if ind:
        motif = vectorfield.get_points_around(indcentral, radius, ind)
        motif = motif - indcentral
    else:
        ptcentral = [axe_x[indcentral[0]], axe_y[indcentral[1]]]
        motif = vectorfield.get_points_around(ptcentral, radius, ind)
        motif = motif - indcentral
    nmbpts = len(motif)
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
        mask_border = np.transpose(np.logical_or(border_x, border_y))
    else:
        delta = (axe_x[1] - axe_x[0] + axe_y[1] - axe_y[0])/2
        border_x = np.logical_or(axe_x <= axe_x[0] + (radius - delta),
                                 axe_x >= axe_x[-1] - (radius - delta))
        border_y = np.logical_or(axe_y <= axe_y[0] + (radius - delta),
                                 axe_y >= axe_y[-1] - (radius - delta))
        border_x, border_y = np.meshgrid(border_x, border_y)
        mask_border = np.transpose(np.logical_or(border_x, border_y))
    # creating dev mask
    mask_dev = np.zeros(vectorfield.shape)
    if dev_pass:
        dev = get_angle_deviation(vectorfield, radius=radius, ind=ind,
                                  raw=True)
        mask_dev = dev < 0.1
    # returning
    return (mask, nmbpts, mask_dev, mask_border, mask_surr,
            motif)


def _get_angles(Vx, Vy, check=False):
    """
    Return the angles from velocities vectors.
    """
    if check:
        # check parameters
        if not isinstance(Vx, ARRAYTYPES):
            raise TypeError()
        Vx = np.array(Vx)
        if not Vx.ndim == 1:
            raise ValueError()
        if not isinstance(Vy, ARRAYTYPES):
            raise TypeError()
        Vy = np.array(Vy)
        if not Vy.ndim == 1:
            raise ValueError()
        if not Vx.shape == Vy.shape:
            raise ValueError()
    # get data
    norm = (Vx**2 + Vy**2)**(.5)
    # getting angle
    theta = np.arccos(Vx/norm)
    theta[Vy < 0] = 2*np.pi - theta[Vy < 0]
    return theta


### Tensor based criterion ###
def get_vorticity(vf, raw=False):
    """
    Return a scalar field with the z component of the vorticity.

    Parameters
    ----------
    vf : VectorField or Velocityfield
        Field on which compute shear stress
    raw : boolean, optional
        If 'True', return an arrays,
        if 'False' (default), return a ScalarField object.
    """
    if not isinstance(vf, VectorField):
        raise TypeError()
    tmp_vf = vf.copy()
    tmp_vf.fill(crop_border=True)
    axe_x, axe_y = tmp_vf.axe_x, tmp_vf.axe_y
    comp_x, comp_y = tmp_vf.comp_x, tmp_vf.comp_y
    mask = tmp_vf.mask
    dx = axe_x[1] - axe_x[0]
    dy = axe_y[1] - axe_y[0]
    _, Exy = np.gradient(comp_x, dx, dy)
    Eyx, _ = np.gradient(comp_y, dx, dy)
    vort = Eyx - Exy
    if raw:
        return vort
    else:
        unit_x, unit_y = tmp_vf.unit_x, tmp_vf.unit_y
        unit_values = vf.unit_values/vf.unit_x
        vort *= unit_values.asNumber()
        unit_values /= unit_values.asNumber()
        vort_sf = ScalarField()
        vort_sf.import_from_arrays(axe_x, axe_y, vort, mask=mask,
                                   unit_x=unit_x, unit_y=unit_y,
                                   unit_values=unit_values)
        return vort_sf


def get_swirling_strength(vf, raw=False):
    """
    Return a scalar field with the swirling strength
    (imaginary part of the eigenvalue of the velocity laplacian matrix)

    Parameters
    ----------
    vf : VectorField or Velocityfield
        Field on which compute shear stress
    raw : boolean, optional
        If 'True', return an arrays,
        if 'False' (default), return a ScalarField object.
    """
    if not isinstance(vf, VectorField):
        raise TypeError()
    tmp_vf = vf.copy()
    tmp_vf.fill(crop_border=True)
    # Getting gradients and axes
    axe_x, axe_y = tmp_vf.axe_x, tmp_vf.axe_y
    comp_x, comp_y = tmp_vf.comp_x, tmp_vf.comp_y
    mask = tmp_vf.mask
    dx = axe_x[1] - axe_x[0]
    dy = axe_y[1] - axe_y[0]
    du_dx, du_dy = np.gradient(comp_x, dx, dy)
    dv_dx, dv_dy = np.gradient(comp_y, dx, dy)
    # swirling stregnth matrix
    swst = np.zeros(tmp_vf.shape)
    # loop on  points
    for i in np.arange(len(axe_x)):
        for j in np.arange(len(axe_y)):
            if not mask[i, j]:
                lapl = [[du_dx[i, j], du_dy[i, j]],
                        [dv_dx[i, j], dv_dy[i, j]]]
                eigvals = np.linalg.eigvals(lapl)
                swst[i, j] = np.max(np.imag(eigvals))
    mask = np.logical_or(mask, np.isnan(swst))
    # creating ScalarField object
    if raw:
        return swst
    else:
        unit_x, unit_y = tmp_vf.unit_x, tmp_vf.unit_y
        # TODO: implémenter unité
        unit_values = ""
        tmp_sf = ScalarField()
        tmp_sf.import_from_arrays(axe_x, axe_y, swst, mask=mask,
                                  unit_x=unit_x, unit_y=unit_y,
                                  unit_values=unit_values)
        return tmp_sf


def get_q_criterion(vectorfield, mask=None, raw=False):
    """
    Return the scalar field of the 2D Q criterion .
    Define as "1/2*(R**2 - S**2)" , with "R" the deformation tensor,
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
        mask = np.zeros(vectorfield.shape)
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'mask' must be an array of boolean")
    else:
        mask = np.array(mask)
    axe_x, axe_y = vectorfield.axe_x, vectorfield.axe_y
    #calcul des gradients
    Exx, Exy, Eyx, Eyy = get_gradients(vectorfield, raw=True)
    # calcul de Qcrit
    norm_rot = 2.*(Exy - Eyx)**2
    norm_shear = (2.*Exx)**2 + (2.*Eyy)**2 + 2.*(Exy + Eyx)**2
    qcrit = .5*(norm_rot - norm_shear)
    unit_values = vectorfield.unit_values**2
    if raw:
        return np.ma.masked_array(qcrit, mask)
    else:
        q_sf = ScalarField()
        q_sf.import_from_arrays(axe_x, axe_y, qcrit, mask,
                                unit_x=vectorfield.unit_x,
                                unit_y=vectorfield.unit_y,
                                unit_values=unit_values)
        return q_sf


def get_delta_criterion(vectorfield, mask=None, raw=False):
    """
    Return the scalar field of the 2D Delta criterion .
    Define as "(Q/3)**3 + (R/2)**2" , with "Q" the Q criterion,
    and "R" the determinant of the jacobian matrice of the velocity.

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
        mask = np.zeros(vectorfield.shape)
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'mask' must be an array of boolean")
    else:
        mask = np.array(mask)
    axe_x, axe_y = vectorfield.axe_x, vectorfield.axe_y
    #calcul des gradients
    Exx, Exy, Eyx, Eyy = get_gradients(vectorfield, raw=True)
    # calcul de Q
    norm_rot = 2.*(Exy - Eyx)**2
    norm_shear = (2.*Exx)**2 + (2.*Eyy)**2 + 2.*(Exy + Eyx)**2
    Q = .5*(norm_rot - norm_shear)
    unit_values = vectorfield.unit_values**2
    # calcul de R
    R = np.zeros(Exx.shape)
    for i in np.arange(Exx.shape[0]):
        for j in np.arange(Exx.shape[1]):
            R[i, j] = np.linalg.det([[Exx[i, j], Exy[i, j]],
                                     [Eyx[i, j], Eyy[i, j]]])
    # calcul de Delta
    delta = (Q/3.)**3 + (R/2.)**2
    if raw:
        return np.ma.masked_array(delta, mask)
    else:
        delta_sf = ScalarField()
        delta_sf.import_from_arrays(axe_x, axe_y, delta, mask,
                                    unit_x=vectorfield.unit_x,
                                    unit_y=vectorfield.unit_y,
                                    unit_values=unit_values)
        return delta_sf


def get_lambda2(vectorfield, mask=None, raw=False):
    """
    Return the lambda2 scalar field. lambda2 criterion is used in
    vortex analysis.
    The fonction is only usable on orthogonal fields.

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
    # check parameter
    if not isinstance(vectorfield, VectorField):
        raise TypeError("'vectorfield' must be a VectorField object")
    if mask is None:
        mask = np.zeros(vectorfield.shape)
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'mask' must be an array of boolean")
    else:
        mask = np.array(mask)
    mask = np.logical_or(mask, vectorfield.mask)
    # getting velocity gradients
    Udx, Udy, Vdx, Vdy = get_gradients(vectorfield, raw=True)
    mask = np.logical_or(mask, Udx.mask)
    # creating returning matrix
    lambda2 = np.zeros(vectorfield.shape)
    # loop on points
    for ij, _, _ in vectorfield:
        i, j = ij
        # check if masked
        if mask[i, j]:
            continue
        # getting symmetric and antisymetric parts
        S = 1./2.*np.array([[2*Udx[i, j], Udy[i, j] + Vdx[i, j]],
                            [Vdx[i, j] + Udy[i, j], 2*Vdy[i, j]]])
        Omega = 1./2.*np.array([[0, Udy[i, j] - Vdx[i, j]],
                                [Vdx[i, j] - Udy[i, j], 0]])
        # getting S^2 + Omega^2
        M = np.dot(S, S) + np.dot(Omega, Omega)
        # getting second eigenvalue
        lambds = linalg.eig(M, left=False, right=False)
        l2 = np.min(lambds)
        # storing lambda2
        lambda2[i, j] = l2.real
    #returning
    if raw:
        return np.ma.masked_array(l2, mask)
    else:
        lambd_sf = ScalarField()
        axe_x, axe_y = vectorfield.axe_x, vectorfield.axe_y
        unit_x, unit_y = vectorfield.unit_x, vectorfield.unit_y
        lambd_sf.import_from_arrays(axe_x, axe_y, lambda2, mask,
                                    unit_x=unit_x, unit_y=unit_y,
                                    unit_values=make_unit(''))
        return lambd_sf

def get_residual_vorticity(vf, raw=False):
    """
    Return a scalar field with the residual of the vorticity.
    (see Kolar (2007)).

    Parameters
    ----------
    vf : VectorField or Velocityfield
        Field on which compute shear stress
    raw : boolean, optional
        If 'True', return an arrays,
        if 'False' (default), return a ScalarField object.
    """
    if not isinstance(vf, VectorField):
        raise TypeError()
    # getting data
    tmp_vf = vf.copy()
    tmp_vf.fill(crop_border=True)
    axe_x, axe_y = tmp_vf.axe_x, tmp_vf.axe_y
    comp_x, comp_y = tmp_vf.comp_x, tmp_vf.comp_y
    mask = tmp_vf.mask
    dx = axe_x[1] - axe_x[0]
    dy = axe_y[1] - axe_y[0]
    # getting gradients
    Exx, Exy = np.gradient(comp_x, dx, dy)
    Eyx, Eyy = np.gradient(comp_y, dx, dy)
    # getting principal rate of strain (s)
    s = np.sqrt(4*Exx**2 + (Exy + Eyx)**2)/2.
    # getting the vorticity-tensor component
    omega = (Eyx - Exy)/2.
    omega_abs = np.abs(omega)
    sign_omega = omega/omega_abs
    filt = s < omega_abs
    # getting the residual vorticity
    res_vort = np.zeros(vf.shape)
    res_vort[filt] = sign_omega[filt]*(omega_abs[filt] - s[filt])
    if raw:
        return res_vort
    else:
        unit_x, unit_y = tmp_vf.unit_x, tmp_vf.unit_y
        unit_values = vf.unit_values/vf.unit_x
        res_vort *= unit_values.asNumber()
        unit_values /= unit_values.asNumber()
        vort_sf = ScalarField()
        vort_sf.import_from_arrays(axe_x, axe_y, res_vort, mask=mask,
                                   unit_x=unit_x, unit_y=unit_y,
                                   unit_values=unit_values)
        return vort_sf

### Others ###
def _unique(a):
    ind_unique = _argunique(a)
    return a[ind_unique]

def _argunique(a):
    a = np.array(a)
    unique_ind = []
    for i, val in enumerate(a):
        uniq = True
        for ind in unique_ind:
            if np.all(val == a[ind]):
                uniq = False
                break
        if uniq:
            unique_ind.append(i)
    return unique_ind
