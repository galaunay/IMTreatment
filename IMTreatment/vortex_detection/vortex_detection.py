# -*- coding: utf-8 -*-
#!/bin/env python3

# Copyright (C) 2003-2007 Gaby Launay

# Author: Gaby Launay  <gaby.launay@tutanota.com>
# URL: https://framagit.org/gabylaunay/IMTreatment

# This file is part of IMTreatment.

# IMTreatment is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.

# IMTreatment is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from ..core import Points, OrientedPoints, Profile, ScalarField, VectorField,\
    TemporalScalarFields,\
    TemporalVectorFields
from ..utils import ProgressCounter, make_unit
from ..utils.types import ARRAYTYPES, NUMBERTYPES, STRINGTYPES
from ..vortex_criterions import get_kappa, get_gamma, get_iota, \
    get_residual_vorticity
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.interpolate import RectBivariateSpline
import warnings
import unum
import copy
from .. import plotlib as pplt

from multiprocess import Pool


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
    # THETA.fill()
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

    def get_cp_cell_position(self):
        """
        Return critical points cell positions and their associated PBI.
        (PBI : Poincarre_Bendixson indice)
        Positions are returned in axis unities (axe_x and axe_y) at the center
        of a cell.

        Returns
        -------
        pos : 2xN array
            position (x, y) of the detected critical points.
        type : 1xN array
            type of CP :

            ====   ===============
            ind    CP type
            ====   ===============
            0      saddle point
            1      unstable focus
            2      stable focus
            3      unstable node
            4      stable node
            ====   ===============
        """
        # get axe data
        delta_x = self.axe_x[1] - self.axe_x[0]
        delta_y = self.axe_y[1] - self.axe_y[0]
        vx_sup_0 = np.array(self.vx > 0, dtype=int)
        vy_sup_0 = np.array(self.vy > 0, dtype=int)
        # check if any masks
        if np.any(self.mask):
            mask = self.mask
            mask = np.logical_or(np.logical_or(mask[0:-1, 0:-1],
                                               mask[1::, 0:-1]),
                                 np.logical_or(mask[0:-1, 1::],
                                               mask[1::, 1::]))
        else:
            mask = np.zeros((self.mask.shape[0] - 1, self.mask.shape[1] - 1))
        # fast first check to see where 0-levelset pass
        signx = (vx_sup_0[0:-1, 0:-1] + vx_sup_0[1::, 0:-1] +
                 vx_sup_0[0:-1, 1::] + vx_sup_0[1::, 1::])
        signy = (vy_sup_0[0:-1, 0:-1] + vy_sup_0[1::, 0:-1] +
                 vy_sup_0[0:-1, 1::] + vy_sup_0[1::, 1::])
        lsx = np.logical_and(signx != 0, signx != 4)
        lsy = np.logical_and(signy != 0, signy != 4)
        filt_0ls = np.logical_and(lsx, lsy)
        filt = np.logical_and(np.logical_not(mask), filt_0ls)
        # loop on filtered cells
        positions = []
        cp_types = []
        I, J = np.meshgrid(np.arange(self.shape[0] - 1),
                           np.arange(self.shape[1] - 1), indexing='ij')
        Is, Js = I[filt], J[filt]
        for n in np.arange(len(Is)):
            i, j = Is[n], Js[n]
            # create tmp_vf object
            tmp_vx = self.vx[i:i + 2, j:j + 2]
            tmp_vy = self.vy[i:i + 2, j:j + 2]
            tmp_theta = self.theta[i:i + 2, j:j + 2]
            tmp_axe_x = self.axe_x[j:j + 2]
            tmp_axe_y = self.axe_y[i:i + 2]
            tmp_vf = VF(vx=tmp_vx, vy=tmp_vy, axe_x=tmp_axe_x,
                        axe_y=tmp_axe_y, mask=[[False, False],
                                               [False, False]],
                        theta=tmp_theta, time=self.time)
            # check struct number
            nmb_struct = tmp_vf._check_struct_number()
            # if there is nothing
            if nmb_struct == (0, 0):
                continue
            else:
                positions.append((tmp_vf.axe_x[0] + delta_x/2.,
                                  tmp_vf.axe_y[0] + delta_y/2.))
                # getting CP type
                if tmp_vf.pbi_x[1] == -1:
                    cp_types.append(0)
                else:
                    jac = _get_jacobian_matrix(tmp_vf.vx, tmp_vf.vy)
                    eigvals, eigvects = np.linalg.eig(jac)
                    tau = np.real(eigvals[0])
                    mu = np.sum(np.abs(np.imag(eigvals)))
                    if mu != 0:
                        if jac[1, 0] < 0:
                            cp_types.append(1)
                        else:
                            cp_types.append(2)
                    elif tau >= 0 and mu == 0:
                        cp_types.append(3)
                    elif tau < 0 and mu == 0:
                        cp_types.append(4)
                    else:
                        raise Exception()
        # returning
        return np.array(positions), np.array(cp_types)

    def get_cp_position(self, thread=1):
        """
        Return critical points positions and their associated PBI using
        bilinear interpolation.
        (PBI : Poincarre_Bendixson indice)
        Positions are returned in axis unities (axe_x and axe_y).

        Parameters
        ----------
        thread : integer or 'all'
            Number of thread to use (multiprocessing).

        Returns
        -------
        pos : 2xN array
            position (x, y) of the detected critical points.
        pbis : 1xN array
            PBI (1 indicate a node, -1 a saddle point)

        Note
        ----
        Using the work of :
        [1]F. Effenberger and D. Weiskopf, “Finding and classifying critical
        points of 2D vector fields: a cell-oriented approach using group
        theory,” Computing and Visualization in Science, vol. 13, no. 8,
        pp. 377–396, Dec. 2010.
        """
        # get the cell position
        positions, cp_types = self.get_cp_cell_position()
        axe_x = self.axe_x
        axe_y = self.axe_y
        Vx = self.vx
        Vy = self.vy
        mask = self.mask
        real_dx = axe_x[1] - axe_x[0]
        real_dy = axe_y[1] - axe_y[0]
        # import linear interpolation levelset solutions (computed with sympy)
        from .levelset_data import get_sol
        sol = get_sol()

        # function to get the position in a cell detected by
        #     get_cp_cell_position
        def get_cp_position_in_cell(pos):
            tmp_x = pos[0]
            tmp_y = pos[1]
            # get the cell indices
            ind_x = np.where(tmp_x < axe_x)[0][0] - 1
            ind_y = np.where(tmp_y < axe_y)[0][0] - 1
            # get data
            Vx_bl = Vx[ind_y:ind_y + 2, ind_x:ind_x + 2]
            Vy_bl = Vy[ind_y:ind_y + 2, ind_x:ind_x + 2]
            # check if masked values
            mask_bl = mask[ind_y:ind_y + 2, ind_x:ind_x + 2]
            if np.any(mask_bl):
                res_pos = pos
                return pos
            # check if null values
            nmb_zeros = np.sum(Vx_bl + Vy_bl == 0)
            if nmb_zeros == 1:
                inds = np.array(np.where(Vx_bl + Vy_bl == 0)).reshape(2)
                res_pos = np.array([axe_x[ind_x + inds[0]],
                                    axe_y[ind_y + inds[1]]])
                return res_pos
            elif nmb_zeros > 1:
                res_pos = pos
                return res_pos
            # check if zero values
            if np.any(Vx_bl == 0) or np.any(Vy_bl == 0):
                # if only one point
                filt = np.logical_and(Vx_bl == 0, Vy_bl == 0)
                if np.sum(filt) == 1:
                    tmp_inds = np.argwhere(filt)
                    return np.array([tmp_inds[0] + axe_x[ind_x],
                                     tmp_inds[1] + axe_y[ind_y]])
                # else raise warning
                warnings.warn("There is a point with zero velocity, it's"
                              "a particular case not implemented yet. "
                              "Skipping this cell (there will be missing CP).")
                return np.array([np.nan, np.nan])
            # solve to get the zero velocity point
            tmp_dic = {'Vx_1': Vx_bl[0, 0], 'Vx_2': Vx_bl[0, 1],
                       'Vx_3': Vx_bl[1, 0], 'Vx_4': Vx_bl[1, 1],
                       'Vy_1': Vy_bl[0, 0], 'Vy_2': Vy_bl[0, 1],
                       'Vy_3': Vy_bl[1, 0], 'Vy_4': Vy_bl[1, 1],
                       'dx': real_dx, 'dy': real_dy,
                       'sqrt': lambda x: x**.5}
            x_sols = []
            y_sols = []
            for j in np.arange(len(sol)):
                # if error such as 'Division by zero' occur, the zero velocity
                # point is at the cell center
                try:
                    x_sols.append(eval(sol[j][0], {"__builtins__": {}},
                                       tmp_dic))
                    y_sols.append(eval(sol[j][1], {"__builtins__": {}},
                                       tmp_dic))
                except RuntimeWarning:
                    print(("bad point : \n ({}, {})".format(tmp_x, tmp_y)))
                    x_sols.append(real_dx/2.)
                    y_sols.append(real_dy/2.)

            # delete the points outside the cell
            tmp_sol = []
            for j in np.arange(len(x_sols)):
                if np.iscomplex(x_sols[j]):
                    x_sols[j] = x_sols[j].as_real_imag()[0]
                if np.iscomplex(y_sols[j]):
                    y_sols[j] = y_sols[j].as_real_imag()[0]
                if (x_sols[j] < 0 or x_sols[j] > real_dx or
                        y_sols[j] < 0 or y_sols[j] > real_dy):
                    continue
                tmp_sol.append([x_sols[j], y_sols[j]])
            # if no more points
            if len(tmp_sol) == 0:
                res_pos = np.array([tmp_x, tmp_y])
            # if multiple points
            elif len(tmp_sol) != 1:
                tmp_sol = np.array(tmp_sol)
                tmp_sol = [t for t in tmp_sol if not np.any(np.isnan(t))]
                if not np.all(tmp_sol[0] == tmp_sol):
                    tmp_sol = [np.mean(tmp_sol, axis=0)]
                tmp_sol = tmp_sol[0]
                res_pos = np.array([tmp_sol[0] + axe_x[ind_x],
                                    tmp_sol[1] + axe_y[ind_y]])
            # if one point (ok case)
            else:
                tmp_sol = tmp_sol[0]
                res_pos = np.array([tmp_sol[0] + axe_x[ind_x],
                                    tmp_sol[1] + axe_y[ind_y]])
            return res_pos

        # use multiprocessing to get cp position on cell if asked
        if thread != 1:
            if thread == 'all':
                pool = Pool()
            else:
                pool = Pool(thread)
            new_positions = pool.map_async(get_cp_position_in_cell, positions)
            pool.close()
            pool.join()
            new_positions = new_positions.get()
        else:
            new_positions = [get_cp_position_in_cell(pos) for pos in positions]
        new_positions = np.array(new_positions)
        # clean Nan
        if len(new_positions) > 0:
            filt1d = ~np.sum(np.isnan(new_positions), axis=1, dtype=bool)
            cp_types = cp_types[filt1d]
            new_positions = new_positions[filt1d, :]
        # returning
        return new_positions, cp_types

    def get_pbi(self, direction):
        """
        Return the PBI along the given axe.
        """
        theta = self.theta.copy()
        if direction == 'y':
            theta = np.rot90(theta, 3)
        pbi = np.zeros(theta.shape[1])
        for i in np.arange(1, theta.shape[1]):
            # getting and concatening profiles
            thetas_border = np.concatenate((theta[::-1, 0],
                                            theta[0, 0:i],
                                            theta[:, i],
                                            theta[-1, i:0:-1]),
                                           axis=0)
            delta_thetas = np.concatenate((thetas_border[1::] -
                                           thetas_border[0:-1],
                                           thetas_border[0:1] -
                                           thetas_border[-1::]), axis=0)
            # particular points treatment
            delta_thetas[delta_thetas > np.pi] -= 2*np.pi
            delta_thetas[delta_thetas < -np.pi] += 2*np.pi
            # Stockage
            pbi[i] = np.sum(delta_thetas)/(2.*np.pi)
        if direction == 'y':
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
        # cropping original field
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

    # def _find_arbitrary_cut_positions(self):
    #     """
    #     Return the position along x and y where the field can be cut.
    #     (at middle positions along each axes)
    #     """
    #     if np.any(self.shape <= 4):
    #         return None, None
    #     len_x = self.shape[1]
    #     len_y = self.shape[0]
    #     grid_x = [0, np.round(len_x/2.), len_x]
    #     grid_y = [0, np.round(len_y/2.), len_y]
    #     return grid_x, grid_y

    def _calc_pbi(self):
        """
        Compute the pbi along the two axis (if not already computed,
        and store them.
        """
        try:
            self.pbi_x
            self.pbi_y
        except AttributeError:
            self.pbi_x = self.get_pbi(direction='x')
            self.pbi_y = self.get_pbi(direction='y')

    def _check_struct_number(self):
        """
        Return the possible number of structures in the field along each axis.
        """
        if self.shape == (2, 2):
            pbi_x = self.get_pbi(1)
            self.pbi_x = pbi_x
            self.pbi_y = pbi_x
            poi_x = pbi_x[1::] - pbi_x[0:-1]
            num_x = len(poi_x[poi_x != 0])
            return num_x, num_x
        else:
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
    def __init__(self, unit_time='s'):
        # check parameters
        self.foc = np.array([])
        self.foc_traj = None
        self.foc_c = np.array([])
        self.foc_c_traj = None
        self.node_i = np.array([])
        self.node_i_traj = None
        self.node_o = np.array([])
        self.node_o_traj = None
        self.sadd = np.array([])
        self.sadd_traj = None
        self.times = np.array([])
        self.unit_time = unit_time
        self.unit_x = make_unit('')
        self.unit_y = make_unit('')
        self.colors = ['#E24A33', '#348ABD', '#988ED5', '#777777',
                       '#FBC15E', '#8EBA42', '#FFB5B8']
        self.cp_types = ['foc', 'foc_c', 'node_i', 'node_o', 'sadd']
        self.current_epsilon = None

    def __add__(self, obj):
        if isinstance(obj, CritPoints):
            if len(obj.times) == 0:
                return self
            if len(self.times) == 0:
                return obj
            if not self.unit_time == obj.unit_time:
                raise ValueError()
            if not self.unit_x == obj.unit_x:
                raise ValueError()
            if not self.unit_y == obj.unit_y:
                raise ValueError()
            tmp_CP = CritPoints(unit_time=self.unit_time)
            tmp_CP.unit_x = self.unit_x
            tmp_CP.unit_y = self.unit_y
            tmp_CP.unit_time = self.unit_time
            tmp_CP.foc = np.append(self.foc, obj.foc)
            tmp_CP.foc_c = np.append(self.foc_c, obj.foc_c)
            tmp_CP.node_i = np.append(self.node_i, obj.node_i)
            tmp_CP.node_o = np.append(self.node_o, obj.node_o)
            tmp_CP.sadd = np.append(self.sadd, obj.sadd)
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

    def __eq__(self, obj):
        if not isinstance(obj, CritPoints):
            return False
        for attr in ['foc', 'foc_c', 'node_i', 'node_o', 'sadd']:
            if not np.all(obj.__getattribute__(attr) ==
                          self.__getattribute__(attr)):
                return False
            attr2 = '{}_traj'.format(attr)
            if not np.all(obj.__getattribute__(attr2) ==
                          self.__getattribute__(attr2)):
                return False
        for attr in ['unit_time', 'unit_x', 'unit_y']:
            if obj.__getattribute__(attr) != self.__getattribute__(attr):
                return False
        for attr in ['times', 'colors', 'cp_types']:
            if not np.all(obj.__getattribute__(attr) ==
                          self.__getattribute__(attr)):
                return False
        return True

    @property
    def unit_x(self):
        return self.__unit_x

    @unit_x.setter
    def unit_x(self, new_unit_x):
        if isinstance(new_unit_x, STRINGTYPES):
            new_unit_x = make_unit(new_unit_x)
        if not isinstance(new_unit_x, unum.Unum):
            raise TypeError()
        self.__unit_x = new_unit_x
#        for kind in self.iter:
#            for i in np.arange(len(kind)):
#                kind[i].unit_x = new_unit_x
#        for kind in self.iter_traj:
#            if kind is None:
#                continue
#            for i in np.arange(len(kind)):
#                kind[i].unit_x = new_unit_x

    @property
    def unit_y(self):
        return self.__unit_y

    @unit_y.setter
    def unit_y(self, new_unit_y):
        if isinstance(new_unit_y, STRINGTYPES):
            new_unit_y = make_unit(new_unit_y)
        if not isinstance(new_unit_y, unum.Unum):
            raise TypeError()
        self.__unit_y = new_unit_y
#        for kind in self.iter:
#            for i in np.arange(len(kind)):
#                kind[i].unit_y = new_unit_y
#        for kind in self.iter_traj:
#            if kind is None:
#                continue
#            for i in np.arange(len(kind)):
#                kind[i].unit_y = new_unit_y

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
#        for kind in self.iter:
#            for i in np.arange(len(kind)):
#                kind[i].unit_v = new_unit_time
#        for kind in self.iter_traj:
#            if kind is None:
#                continue
#            for i in np.arange(len(kind)):
#                kind[i].unit_v = new_unit_time

    @property
    def iter(self):
        return [self.foc, self.foc_c, self.node_i, self.node_o, self.sadd]

    @property
    def iter_traj(self):
        return [self.foc_traj, self.foc_c_traj, self.node_i_traj,
                self.node_o_traj, self.sadd_traj]

    def copy(self):
        """
        Return a copy of the CritPoints object.
        """
        return copy.deepcopy(self)

    def compute_traj(self, epsilon=None, close_traj=False):
        """
        Compute cp trajectory from cp positions.

        Parameters
        ----------
        epsilon : number, optional
            Maximal distance between two successive points.
            default value is Inf.
        close_traj : bool
            If 'True', try to close the trajectories (better to get the
            cp fusion position)
        """
        # check parameters
        if epsilon is None:
            epsilon = np.inf
        elif isinstance(epsilon, NUMBERTYPES):
            pass
        elif isinstance(epsilon, unum.Unum):
            fact = epsilon/self.unit_x
            unit_fact = fact.strUnit()
            if unit_fact != '[]':
                raise ValueError()
            epsilon = fact.asNumber()
        else:
            raise TypeError()
        self.current_epsilon = epsilon
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
                                                    epsilon=epsilon,
                                                    close_traj=close_traj)
        self.foc_c_traj = self._get_cp_time_evolution(focs_c, times=self.times,
                                                      epsilon=epsilon,
                                                      close_traj=close_traj)
        self.node_i_traj = self._get_cp_time_evolution(nodes_i,
                                                       times=self.times,
                                                       epsilon=epsilon,
                                                       close_traj=close_traj)
        self.node_o_traj = self._get_cp_time_evolution(nodes_o,
                                                       times=self.times,
                                                       epsilon=epsilon,
                                                       close_traj=close_traj)
        self.sadd_traj = self._get_cp_time_evolution(sadds, times=self.times,
                                                     epsilon=epsilon,
                                                     close_traj=close_traj)
#        # close the trajectories if asked
#        if close_traj:
#            # loop on trajectories
#            for i, traj_kind in enumerate(self.iter_traj):
#                for j, traj in enumerate(traj_kind):
#                    ending_time = traj.v[-1]
#                    ind_time = np.where(ending_time == self.times)[0][0]
#                    after_time = self.times[ind_time + 1]

    def get_points_density(self, kind, bw_method=None, resolution=100,
                           output_format='normalized'):
        """
        Return the presence density map for the given point type.

        Parameters
        ----------
        kind : string
            Type of critical point for the density map
            (can be 'foc', 'foc_c', 'sadd', 'node_i', 'node_o')
        bw_method : str, scalar or callable, optional
            The method used to calculate the estimator bandwidth.
            This can be 'scott', 'silverman', a scalar constant or
            a callable. If a scalar, this will be used as std
            (it should aprocimately be the size of the density
            node you want to see).
            If a callable, it should take a gaussian_kde instance as only
            parameter and return a scalar. If None (default), 'scott' is used.
            See Notes for more details.
        resolution : integer or 2x1 tuple of integers, optional
            Resolution for the resulting field.
            Can be a tuple in order to specify resolution along x and y.
        output_format : string, optional
            'normalized' (default) : give position probability (integral egal 1).
            'absolute' : sum of integral over all points density egal 1.
            'ponderated' : give position probability ponderated by the number
                           or points (integral egal number of points).
            'concentration' : give local concentration (in point per surface).
        """
        # getting wanted points
        if kind == 'foc':
            pts = self.foc
        elif kind == 'foc_c':
            pts = self.foc_c
        elif kind == 'sadd':
            pts = self.sadd
        elif kind == 'node_i':
            pts = self.node_i
        elif kind == 'node_o':
            pts = self.node_o
        else:
            raise ValueError()
        # concatenate
        tot = Points()
        for pt in pts:
            tot += pt
        # getting density map
            absolute = False
        if output_format == 'absolute':
            absolute = True
            output_format = 'normalized'
        dens = tot.get_points_density(bw_method=bw_method,
                                      resolution=resolution,
                                      output_format=output_format, raw=False)
        if absolute:
            nmb_pts = np.sum([len(pt.xy) for pt in pts])*1.
            nmb_tot_pts = np.sum([np.sum([len(pt.xy) for pt in ptsb])
                                  for ptsb in self.iter])*1.
            dens *= nmb_pts/nmb_tot_pts
        # returning
        return dens

    def display_traj_len_repartition(self):
        """
        display profiles with trajectories length repartition
        (usefull to choose the right epsilon)
        """
        # get traj length
        typ_lens = []
        for i, typ in enumerate(self.iter_traj):
            lens = []
            for traj in typ:
                lens.append(len(traj.xy))
            typ_lens.append(lens)
        # display
        fig = plt.figure()
        plt.hist(typ_lens, bins=np.arange(np.min(np.min(typ_lens)) - 0.5,
                                          np.max(np.max(typ_lens)) + 1.5, 1),
                 stacked=True, histtype='barstacked',
                 color=self.colors[0:5],
                 label=self.cp_types)
        plt.legend()
        plt.xlabel('Trajectories size')
        plt.ylabel('Number of trajectories')
        # returning
        return fig

    def get_traj_direction_changement(self, cp_type, direction,
                                      smoothing=None):
        """
        Return the position of trajectories direction changement.

        Parameters
        ----------
        cp_type : string in ['foc', 'foc_c', 'node_i', 'node_o', 'sadd']
            CP type to use
        direction: string in ['x', 'y']
            Direction along which look
        smoothing : number, optional
            Smoothing size (performed on value before gradient search).
        Returns
        -------
        chg_pts_1, chg_pts_2 : Points objects
            .
        """
        if cp_type not in self.cp_types:
            raise ValueError()
        # get trajectories
        trajs = np.array(self.iter_traj)[cp_type == np.array(self.cp_types)][0]
        # get changement points
        res_min = Points(unit_x=trajs[0].unit_x, unit_y=trajs[0].unit_y,
                         unit_v=trajs[0].unit_v)
        res_max = Points(unit_x=trajs[0].unit_x, unit_y=trajs[0].unit_y,
                         unit_v=trajs[0].unit_v)
        for traj in trajs:
            # skip too short trajs
            if len(traj) <= 2:
                continue
            # get data
            prof_x = Profile(traj.v, traj.xy[:, 0], unit_x=traj.unit_v,
                             unit_y=traj.unit_x)
            prof_y = Profile(traj.v, traj.xy[:, 1], unit_x=traj.unit_v,
                             unit_y=traj.unit_y)
            prof_t = Profile(traj.v, traj.v, unit_x=traj.unit_v,
                             unit_y=traj.unit_v)
            # smooth if necessary
            if smoothing is not None:
                prof_x.smooth(tos='gaussian', size=smoothing, inplace=True)
                prof_y.smooth(tos='gaussian', size=smoothing, inplace=True)
                prof_t.smooth(tos='gaussian', size=smoothing, inplace=True)
            # get directionnal data
            if direction == 'x':
                prof_a = prof_x
            elif direction == 'y':
                prof_a = prof_y
            else:
                raise ValueError()
            # get 0-gradient position
            ind_min, ind_max = prof_a.get_extrema_position(smoothing=None,
                                                           ind=True)

            x_min = prof_x.get_interpolated_values(x=ind_min, ind=True)
            y_min = prof_y.get_interpolated_values(x=ind_min, ind=True)
            t_min = prof_t.get_interpolated_values(x=ind_min, ind=True)
            x_min = np.array(x_min).flatten()
            y_min = np.array(y_min).flatten()
            t_min = np.array(t_min).flatten()
            x_max = prof_x.get_interpolated_values(x=ind_max, ind=True)
            y_max = prof_y.get_interpolated_values(x=ind_max, ind=True)
            t_max = prof_t.get_interpolated_values(x=ind_max, ind=True)
            x_max = np.array(x_max).flatten()
            y_max = np.array(y_max).flatten()
            t_max = np.array(t_max).flatten()
            for i in range(len(x_min)):
                res_min.add([x_min[i], y_min[i]], v=t_min[i])
            for i in range(len(x_max)):
                res_max.add([x_max[i], y_max[i]], v=t_max[i])
        # returning
        return res_min, res_max

    def break_trajectories(self, x=None, y=None, smooth_size=5, inplace=False):
        """
        Break the trajectories in pieces.
        Usefull to do before using 'get_mean_trajectory'

        Parameters
        ----------
        x, y : number
            Position of breaking
            If not specified, break the trajectories where dx/dv=0
        smooth_size : number
            Smoothing used for the dx/dv=0 detection
        """
        if inplace:
            tmp_cp = self
        else:
            tmp_cp = self.copy()
        # decompose each traj of each kind
        new_trajs = []
        for traj_type in tmp_cp.iter_traj:
            tmp_trajs = []
            for traj in traj_type:
                # if trajectory too short, just keep it
                if len(traj) <= 2:
                    tmp_trajs.append(traj)
                    continue
                # if x specified, find breaking indice
                if x is not None:
                    x_prof = traj.export_to_profile(axe_x='v', axe_y='x')
                    dv = x_prof.x[1] - x_prof.x[0]
                    ind_brk = x_prof.get_value_position(x, ind=False)
                elif y is not None:
                    y_prof = traj.export_to_profile(axe_x='v', axe_y='y')
                    dv = y_prof.x[1] - y_prof.x[0]
                    ind_brk = y_prof.get_value_position(y, ind=False)
                else:
                    x_prof = traj.export_to_profile(axe_x='v', axe_y='x')
                    dv = x_prof.x[1] - x_prof.x[0]
                    ind_brk, _ = x_prof.get_extrema_position(
                        smoothing=smooth_size)
                    # if nothing in ind_min
                    if len(ind_brk) == 0:
                        tmp_trajs.append(traj)
                        continue
                # else, break the trajectory
                ind_brk = np.concatenate(([-np.inf], ind_brk, [np.inf]))
                for i, ind in enumerate(ind_brk[0:-1]):
                    tmp_trajs.append(traj.crop(intervv=[ind_brk[i],
                                                        ind_brk[i+1] + dv],
                                     inplace=False))
            new_trajs.append(tmp_trajs)
        # store
        for i, type_name in enumerate(tmp_cp.cp_types):
            tmp_cp.__dict__['{}_traj'.format(type_name)] = new_trajs[i]
        # return
        if not inplace:
            return tmp_cp

    def get_mean_trajectory(self, cp_type, min_len=20,
                            min_nmb_to_avg=10, rel_diff_epsilon=0.03,
                            rel_len_epsilon=0.25, verbose=False):
        """
        Return a mean trajectory (based on a set of trajectories)

        Parameters
        ----------
        cp_type : string in ['foc', 'foc_c', 'node_i', 'node_o', 'sadd']
                  or a set of trajectories
            Trajectory type to average.
        min_len : number
            Ignore trajectories with length smaller.
        min_nmb_to_avg : number
            Minimum number of values necessary to make an average
        rel_diff_epsilon : number
            Maximum relative difference used to determine the different
            mean trajectories.
        rel_len_epsilon : number
            Maximum relative length difference used to determine the different
            mean trajectories.
        Returns
        -------
        mean_traj : MeanTrajectory object
            .
        skipped_traj : tuple of Points objects
            Skipped trajectories

        Notes
        -----
        -You may want to separate your lonf trajectories with
        'break_trajectories' before using this function
        -Checking trajectory resemblance ake into account length of the
         trajectories and integrale of their differences.
        """
        # check
        if cp_type not in ['foc', 'foc_c', 'node_i', 'node_o', 'sadd']:
            trajs = cp_type
        else:
            trajs = self.__dict__['{}_traj'.format(cp_type)]
        # loop for each mean trajectory
        mean_trajs = []
        used = np.zeros((len(trajs)), dtype=bool)
        nmb_mean_traj = 0
        nmb_too_short = 0
        nmb_too_diff = 0
        skipped_trajs = []
        while True:
            nmb_traj_used = 0
            av_x = None
            av_y = None
            av_t = None
            used_traj_inds = []
            used_trajs = []
            # concatenate all the trajectories
            # we make multiples loops to be sure that the result is not
            # dependant on the initial trajectory choice
            old_used = -1
            while True:
                # break if no modification on the run
                if np.sum(used) == old_used:
                    break
                else:
                    old_used = np.sum(used)
                for i, traj in enumerate(trajs):
                    # skip if already used
                    if used[i]:
                        continue
                    # remove from the set if trajectory length is too low
                    if len(traj) < min_len:
                        used[i] = True
                        nmb_too_short += 1
                        continue
                    # get the parameters evolution with time
                    tmp_x = traj.export_to_profile(axe_x="v", axe_y="x")
                    tmp_y = traj.export_to_profile(axe_x="v", axe_y="y")
                    tmp_t = traj.export_to_profile(axe_x="v", axe_y="v")
                    tmp_x.x -= tmp_x.x[0]
                    tmp_y.x -= tmp_y.x[0]
                    tmp_t.x -= tmp_t.x[0]
                    # store first trajectory (referential)
                    if av_x is None:
                        tmp_x_base = tmp_x.copy()
                        av_x = tmp_x.copy()
                        av_y = tmp_y.copy()
                        av_t = tmp_t.copy()
                        used[i] = True
                        used_traj_inds.append(i)
                        used_trajs.append(traj)
                        nmb_traj_used += 1
                        continue
                    # actualize reference
                    tmp_x_base = av_x.remove_doublons(method="average",
                                                      inplace=False)
                    tmp_y_base = av_y.remove_doublons(method="average",
                                                      inplace=False)
                    # if trajectories length are too different, skip
                    len_min, len_max = np.sort([len(tmp_x_base), len(tmp_x)])
                    diff = (len_max - len_min)/float(len_max)
                    if diff > rel_len_epsilon:
                        continue
                    # if trajectory is too different from the referential one,
                    # skip
                    tmp_conv_x = tmp_x.get_convolution_of_difference(
                        tmp_x_base,
                        normalized=True)
                    tmp_conv_y = tmp_y.get_convolution_of_difference(
                        tmp_y_base,
                        normalized=True)
                    tmp_conv = (tmp_conv_x*tmp_conv_y)**.5
                    if tmp_conv.min > rel_diff_epsilon:
                        continue
                    # else, shift the trajectory and add it to the set
                    shift = tmp_conv.x[min(list(range(len(tmp_conv))),
                                           key=lambda i: tmp_conv.y[i])]
                    dx = tmp_x.x[1] - tmp_x.x[0]
                    shift = np.floor(shift/dx)*dx   # not allow quarter-dx
                    tmp_x.x += shift
                    tmp_y.x += shift
                    tmp_t.x += shift
                    av_x.add_points(tmp_x)
                    av_y.add_points(tmp_y)
                    av_t.add_points(tmp_t)
                    used[i] = True
                    used_traj_inds.append(i)
                    used_trajs.append(traj)
                    nmb_traj_used += 1
            # if no remaining trajectories, end the While loop
            if av_x is None:
                break
            # averaging the set of trajectories on each time step
            tmp_av_x = av_x.remove_doublons(method="average",
                                            inplace=False)
            av_x_norm = (sum(av_x.x**2)/len(av_x))**.5
            # Get std for each points
            new_x = []
            new_y = []
            new_t = []
            assoc_real_times = []
            assoc_std_x = []
            assoc_std_y = []
            for t in tmp_av_x.x:
                filt = abs(av_x.x - t)/av_x_norm < 1e-6
                if np.sum(filt) < min_nmb_to_avg:
                    continue
                new_x.append(np.mean(av_x.y[filt]))
                new_y.append(np.mean(av_y.y[filt]))
                new_t.append(t)
                assoc_std_x.append(np.std(av_x.y[filt]))
                assoc_std_y.append(np.std(av_y.y[filt]))
                assoc_real_times.append(av_t.y[filt])
            # skipping if no remaining points
            if len(new_x) == 0:
                nmb_too_diff += nmb_traj_used
                for ind in used_traj_inds:
                    skipped_trajs.append(trajs[ind])
                continue
            nmb_mean_traj += 1
            # storing the mean trajectory
            xy = list(zip(new_x, new_y))
            mean_traj = MeanTrajectory(xy=xy, time=new_t,
                                       assoc_real_times=assoc_real_times,
                                       assoc_std_x=assoc_std_x,
                                       assoc_std_y=assoc_std_y,
                                       nmb_traj_used=nmb_traj_used,
                                       unit_x=self.unit_x,
                                       unit_y=self.unit_y,
                                       unit_times=self.unit_time, name='',
                                       base_trajs=used_trajs,)
            mean_traj.sort(ref='v', inplace=True)
            mean_trajs.append(mean_traj)
        # sort the mean_traj by decreasing number of used trajectories
        ind_sort = np.argsort([traj.nmb_traj_used for traj in mean_trajs])
        mean_trajs = [mean_trajs[ind] for ind in ind_sort[::-1]]
        # echo some stats
        if verbose:
            pad = len("{}".format(len(trajs)))
            print("+++++++++++++++++++++++++++++++++++++")
            print("+++ Mean trajectories computation +++")
            print("+++++++++++++++++++++++++++++++++++++")
            print(("+++ {:>{pad}} Trajectories in the set"
                  .format(len(trajs), pad=pad)))
            print(("+++ {:>{pad}} Mean Trajectories defined :"
                  .format(nmb_mean_traj, pad=pad)))
            for i, traj in enumerate(mean_trajs):
                print(("+++     {:>{pad}} trajectories used for MT{}"
                      .format(traj.nmb_traj_used, i, pad=pad)))
            print(("+++ {:>{pad}} Trajectories skipped"
                  .format(nmb_too_short + nmb_too_diff, pad=pad)))
            print(("+++     {:>{pad}} too short"
                   .format(nmb_too_short, pad=pad)))
            print(("+++     {:>{pad}} too different"
                   .format(nmb_too_diff, pad=pad)))
            # plot
            cycler = plt.rcParams['axes.prop_cycle']()
            colors = [cycler.next()['color'] for i in range(40)]
            fig, axs = plt.subplots(2, 1, sharex=True)
            plt.sca(axs[0])
            filt = [cp_type == 'foc', cp_type == 'foc_c', cp_type == 'node_i',
                    cp_type == 'node_o', cp_type == 'sadd']
            self.display_traj('x', filt=filt, color='k', marker=None)
            for i, mtraj in enumerate(mean_trajs):
                for traj in mtraj.base_trajs:
                    traj.display(kind='plot', axe_x='x', axe_y='v',
                                 color=colors[i])
            plt.sca(axs[1])
            for i, traj in enumerate(mean_trajs):
                traj.display(kind='plot', color=colors[i], marker='o',
                             label='{}'.format(i))
            for i, mtraj in enumerate(mean_trajs):
                for traj in mtraj.base_trajs:
                    traj.display(kind='plot', ls='none', marker='o', mfc='w',
                                 mec='k', zorder=0)
            plt.legend(loc=0)
            plt.tight_layout()

        # return the set of mean trajectories
        return mean_trajs, skipped_trajs

    def add_point(self, foc=None, foc_c=None, node_i=None,
                  node_o=None, sadd=None, time=None):
        """
        Add a new point to the CritPoints object.

        Parameters
        ----------
        foc, foc_c, node_i, node_o, sadd : Points objects
            Representing the critical points at this time.
        time : number
            Time.
        """
        # check parameters
        diff_pts = [foc, foc_c, node_i, node_o, sadd]
        for pt in diff_pts:
            if pt is not None:
                if not isinstance(pt, Points):
                    raise TypeError()
        if not isinstance(time, NUMBERTYPES):
            raise ValueError()
        if np.any(self.times == time):
            raise ValueError()
        # first point
        if len(self.times) == 0:
            for pt in diff_pts:
                if pt is not None:
                    self.unit_x = pt.unit_x
                    self.unit_y = pt.unit_y
                    break
        # set default values
        for i, pt in enumerate(diff_pts):
            if pt is None:
                diff_pts[i] = Points(unit_x=self.unit_x, unit_y=self.unit_y,
                                     unit_v=self.unit_time)
        # check units
        for pt in diff_pts:
            if pt.unit_x != self.unit_x:
                raise ValueError()
            if pt.unit_y != self.unit_y:
                raise ValueError()
        # store data
        self.foc = np.append(self.foc, diff_pts[0].copy())
        self.foc_c = np.append(self.foc_c, diff_pts[1].copy())
        self.node_i = np.append(self.node_i, diff_pts[2].copy())
        self.node_o = np.append(self.node_o, diff_pts[3].copy())
        self.sadd = np.append(self.sadd, diff_pts[4].copy())
        self.times = np.append(self.times, time)
        self._sort_by_time()
        # trajectories are obsolete
        self._remove_trajectories()

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
        self.times = np.delete(self.times, indice)
        # trajectories are obsolete
        self._remove_trajectories()

    def set_origin(self, x=None, y=None):
        if x is None and y is None:
            return None
        for pts in self.iter:
            for pt in pts:
                pt.set_origin(x=x, y=y)
        for trajs in self.iter_traj:
            for traj in trajs:
                traj.set_origin(x=x, y=y)

    def change_unit(self, axe, new_unit):
        """
        Change the unit of an axe.

        Parameters
        ----------
        axe : string
            'y' for changing the y axis unit
            'x' for changing the x axis unit
            'time' for changing the time unit
        new_unit : Unum.unit object or string
            The new unit.
        """
        if isinstance(new_unit, STRINGTYPES):
            new_unit = make_unit(new_unit)
        if not isinstance(new_unit, unum.Unum):
            raise TypeError()
        if not isinstance(axe, STRINGTYPES):
            raise TypeError()
        if axe in ['x', 'y']:
            for kind in self.iter:
                for pt in kind:
                    pt.change_unit(axe, new_unit)
            for kind in self.iter_traj:
                if kind is not None:
                    for pt in kind:
                        pt.change_unit(axe, new_unit)
            if axe == 'x':
                self.unit_x = new_unit
            else:
                self.unit_y = new_unit
        elif axe == 'time':
            for kind in self.iter:
                for pt in kind:
                    pt.change_unit('v', new_unit)
            for kind in self.iter_traj:
                for pt in kind:
                    pt.change_unit('v', new_unit)
            old_unit = self.unit_time
            new_unit = old_unit.asUnit(new_unit)
            fact = new_unit.asNumber()
            self.times *= fact
            self.unit_time = new_unit/fact
        else:
            raise ValueError()

    def scale(self, scalex=1., scaley=1., scalev=1., inplace=False):
        """
        Change the scale of the axis.

        Parameters
        ----------
        scalex, scaley, scalev : numbers or Unum objects
            scales along x, y and v
        inplace : boolean, optional
            If 'True', scaling is done in place, else, a new instance is
            returned.
        """
        # check params
        if not isinstance(scalex, NUMBERTYPES + (unum.Unum, )):
            raise TypeError()
        if not isinstance(scaley, NUMBERTYPES + (unum.Unum, )):
            raise TypeError()
        if not isinstance(scalev, NUMBERTYPES + (unum.Unum, )):
            raise TypeError()
        if not isinstance(inplace, bool):
            raise TypeError()
        if inplace:
            tmp_cp = self
        else:
            tmp_cp = self.copy()
        # scale self v
        new_unit = tmp_cp.unit_time*scalev
        fact = new_unit.asNumber()
        new_unit /= fact
        tmp_cp.times *= fact
        tmp_cp.unit_time = new_unit
        # scale self x
        new_unit = tmp_cp.unit_x*scalex
        fact = new_unit.asNumber()
        new_unit /= fact
        tmp_cp.unit_x = new_unit
        # scale self y
        new_unit = tmp_cp.unit_y*scaley
        fact = new_unit.asNumber()
        new_unit /= fact
        tmp_cp.unit_y = new_unit
        # loop to scale pts
        for pt_type in tmp_cp.iter:
            for i, pts in enumerate(pt_type):
                pt_type[i].scale(scalex=scalex, scaley=scaley, scalev=scalev,
                                 inplace=True)
        # loop to scale traj (if necessary)
        for traj in tmp_cp.iter_traj:
            if traj is None:
                continue
            for i, pts in enumerate(traj):
                traj[i].scale(scalex=scalex, scaley=scaley, scalev=scalev,
                              inplace=True)
        # returning
        if not inplace:
            return tmp_cp

    def crop(self, intervx=None, intervy=None, intervt=None, ind=False,
             inplace=False):
        """
        crop the point field.
        """
        # inplace
        if inplace:
            tmp_cp = self
        else:
            tmp_cp = self.copy()
        # temporal cropping
        if intervt is not None:
            if ind:
                filt = np.zeros(shape=(len(tmp_cp.times)), dtype=bool)
                filt[intervt[0]:intervt[1]] = True
            else:
                filt = np.logical_and(tmp_cp.times >= intervt[0],
                                      tmp_cp.times <= intervt[1])
            tmp_cp.times = tmp_cp.times[filt]
            tmp_cp.foc = tmp_cp.foc[filt]
            tmp_cp.foc_c = tmp_cp.foc_c[filt]
            tmp_cp.sadd = tmp_cp.sadd[filt]
            tmp_cp.node_i = tmp_cp.node_i[filt]
            tmp_cp.node_o = tmp_cp.node_o[filt]
            for traj_type in self.iter_traj:
                [traj.crop(intervv=intervt, ind=ind, inplace=True)
                 for traj in traj_type]
            self._remove_void_trajectories()
        # spatial cropping
        if intervx is not None or intervy is not None:
            for kind in tmp_cp.iter:
                for pts in kind:
                    pts.crop(intervx=intervx, intervy=intervy,
                             inplace=True)
            # make trajectories obsolete
            tmp_cp._remove_trajectories()
        # returning
        if not inplace:
            return tmp_cp

#    def trim_time(self, intervtime, ind=False, inplace=False):
#        """
#        Trim the points field
#        """
#        # inplace
#        if inplace:
#            tmp_cp = self
#        else:
#            tmp_cp = self.copy()
#        # trajectories are obsolete
#        tmp_cp._remove_trajectories()
#        # trim the things...
#        if ind:
#            filt = np.zeros(shape=(len(tmp_cp.times)), dtype=bool)
#            filt[intervtime[0]:intervtime[1]] = True
#        else:
#            filt = np.logical_and(tmp_cp.times >= intervtime[0],
#                                  tmp_cp.times <= intervtime[1])
#        tmp_cp.times = tmp_cp.times[filt]
#        tmp_cp.foc = tmp_cp.foc[filt]
#        tmp_cp.foc_c = tmp_cp.foc_c[filt]
#        tmp_cp.sadd = tmp_cp.sadd[filt]
#        tmp_cp.node_i = tmp_cp.node_i[filt]
#        tmp_cp.node_o = tmp_cp.node_o[filt]
#        # returning
#        if not inplace:
#            return tmp_cp

    def clean_traj(self, min_nmb_in_traj):
        """
        Remove some isolated points.

        Parameters
        ----------
        min_nmb_in_traj : integer
            Trajectories that have less than this amount of points are deleted.
            Associated points are also deleted.
        """
        if self.current_epsilon is None:
            raise Exception("You must calculate trajectories"
                            "before cleaning them")
        # clean trajectories
        trajs = ['foc_traj', 'foc_c_traj', 'node_i_traj', 'node_o_traj',
                 'sadd_traj']
        for j in range(len(trajs)):
            tmp_type_traj = self.__getattribute__(trajs[j])
            for i in np.arange(len(tmp_type_traj) - 1, -1, -1):
                traj = tmp_type_traj[i]
                if len(traj.xy) < min_nmb_in_traj:
                    tmp_type_traj = np.delete(tmp_type_traj, i)
            self.__setattr__(trajs[j], tmp_type_traj)
        # extend deleting to points
        self._traj_to_pts()

    def smooth_traj(self, tos='uniform', size=None):
        """
        Smooth the CP trajectories.

        Parameters
        ----------
        tos : string, optional
            Type of smoothing, can be 'uniform' (default) or 'gaussian'
            (See ndimage module documentation for more details)
        size : number, optional
            Size of the smoothing (is radius for 'uniform' and
            sigma for 'gaussian').
            Default is 3 for 'uniform' and 1 for 'gaussian'.
        """
        # loop on trajectories
        for typ in self.iter_traj:
            for traj in typ:
                traj.smooth(tos=tos, size=size, inplace=True)

    def topo_simplify(self, dist_min, kind='replacement'):
        """
        Simplify the topological points field.

        Parameters
        ----------
        dist_min : number
            Minimal distance between two points in the simplified field.
        kind : string, optional
            Algorithm used to simplify the field, can be 'replacement'(default)
            for iterative replacement with center of mass or 'only_delete' to
            eliminate only the first order associates.

        Returns
        -------
        simpl_CP : CritPoints object
            Simplified topological points field.
        """
        simpl_CP = CritPoints(unit_time=self.unit_time)
        for i in np.arange(len(self.times)):
            TP = TopoPoints()
            TP.import_from_CP(self, i)
            TP = TP.simplify(dist_min, kind=kind)
            xy = TP.xy[TP.types == 1]
            foc = Points(xy=xy, v=[self.times[i]]*len(xy),
                         unit_x=self.unit_x, unit_y=self.unit_y,
                         unit_v=self.unit_time)
            xy = TP.xy[TP.types == 2]
            foc_c = Points(xy=xy, v=[self.times[i]]*len(xy),
                           unit_x=self.unit_x, unit_y=self.unit_y,
                           unit_v=self.unit_time)
            xy = TP.xy[TP.types == 3]
            sadd = Points(xy=xy, v=[self.times[i]]*len(xy),
                          unit_x=self.unit_x, unit_y=self.unit_y,
                          unit_v=self.unit_time)
            xy = TP.xy[TP.types == 4]
            node_i = Points(xy=xy, v=[self.times[i]]*len(xy),
                            unit_x=self.unit_x, unit_y=self.unit_y,
                            unit_v=self.unit_time)
            xy = TP.xy[TP.types == 5]
            node_o = Points(xy=xy, v=[self.times[i]]*len(xy),
                            unit_x=self.unit_x, unit_y=self.unit_y,
                            unit_v=self.unit_time)
            simpl_CP.add_point(foc=foc, foc_c=foc_c, sadd=sadd, node_i=node_i,
                               node_o=node_o, time=self.times[i])
        return simpl_CP

    def refine_cp_position(self, cp_type, fields, inplace=True, verbose=True,
                           extrema='max'):
        """
        Refine the position of the critical points by putting them on the
        given scalar field extrema.

        Parameters
        ----------
        cp_type : string in ['foc', 'foc_c', 'sadd', 'node_i', 'node_o']
            Critical points type to refine.
        fields : TemporalScalarFields object
            fields where to search for extrema.
        extrema : string in ['min', 'max']
            If 'max', cp are displaced on field maxima, if 'min', cp are
            displaced on field minima.
        inplace : boolean
            .
        verbose : boolean
            .
        """
        # check
        if cp_type not in ['foc', 'foc_c', 'sadd', 'node_i', 'node_o']:
            raise ValueError()
        if not isinstance(fields, TemporalScalarFields):
            raise TypeError()
        if not np.all(self.times == fields.times):
            raise ValueError()
        times = self.times
        if not isinstance(inplace, bool):
            raise TypeError()
        # get cp pts
        if cp_type == 'foc':
            cp_pts = self.foc
        elif cp_type == 'foc_c':
            cp_pts = self.foc_c
        elif cp_type == 'sadd':
            cp_pts = self.sadd
        elif cp_type == 'node_i':
            cp_pts = self.node_i
        elif cp_type == 'node_o':
            cp_pts = self.node_o
        new_cp_pts = []
        # Loop on time
        PG = ProgressCounter(init_mess="Begin '{}' points refinment"
                             .format(cp_type),
                             nmb_max=len(cp_pts))
        for i in np.arange(len(times)):
            PG.print_progress()
            # check if there is point to refine
            if len(cp_pts[i]) == 0:
                new_cp_pts.append(cp_pts[i].copy())
                continue
            # getting pt and field associated with the current time iteration
            new_pt = cp_pts[i].copy()
            tmp_field = fields[fields.times == times[i]][0]
            # getting new cp position
            new_pt.xy = tmp_field.get_nearest_extrema(cp_pts[i].xy,
                                                      extrema=extrema)
            new_cp_pts.append(new_pt)
        # store new positions
        if inplace:
            tmp_traj = self
        else:
            tmp_traj = self.copy()
        if cp_type == 'foc':
            tmp_traj.foc = new_cp_pts
        elif cp_type == 'foc_c':
            tmp_traj.foc_c = new_cp_pts
        elif cp_type == 'sadd':
            tmp_traj.sadd = new_cp_pts
        elif cp_type == 'node_i':
            tmp_traj.node_i = new_cp_pts
        elif cp_type == 'node_o':
            tmp_traj.node_o = new_cp_pts
        # make trajectories obsolete
        tmp_traj._remove_trajectories()
        # return
        if not inplace:
            return tmp_traj

    def _remove_trajectories(self):
        """
        Delete the computed trajectories.
        This method is called when some points are modified (deleted or added).
        """
        self.current_epsilon = None
        for i in range(len(self.iter_traj)):
            self.iter_traj[i] = None

    def _remove_void_trajectories(self):
        """
        Delete the empty trajectories.
        """
        for traj_type in self.cp_types:
            traj_type = "{}_traj".format(traj_type)
            trajs = self.__getattribute__(traj_type)
            filt = np.array([len(traj) != 0 for traj in trajs])
            self.__setattr__(traj_type, trajs[filt])

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

    def _traj_to_pts(self):
        """
        Define new set of points based on the trajectories.
        """
        # loop on each point type
        for i, traj_type in enumerate(self.iter_traj):
            # concatenate
            tot = Points(unit_x=self.unit_x, unit_y=self.unit_y,
                         unit_v=self.unit_time)
            for traj in traj_type:
                tot += traj
            # replace exising points
            self.iter[i] = np.empty((len(self.times),), dtype=object)
            for j, time in enumerate(self.times):
                filt = tot.v == time
                self.iter[i][j] = Points(tot.xy[filt], v=tot.v[filt],
                                         unit_x=tot.unit_x,
                                         unit_y=tot.unit_y,
                                         unit_v=tot.unit_v)

    @staticmethod
    def _get_cp_time_evolution(points, times=None, epsilon=None,
                               close_traj=False):
        """
        Compute the temporal evolution of each critical point from a set of
        points at different times. (Points objects must each contain only one
        and point time must be specified in 'v' argument of points).

        Parameters
        ----------
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
            return []
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
            def __init__(self, pts_tupl, epsilon, times):
                if not isinstance(pts_tupl, ARRAYTYPES):
                    raise TypeError("'pts' must be a tuple of Point objects")
                for pt in pts_tupl:
                    if not isinstance(pt, Points):
                        raise TypeError("'pts' must be a tuple of Point"
                                        "objects")
                    if not len(pt) == len(pt.v):
                        raise Exception("v has not the same dimension as "
                                        "xy")
                # store the points in a more convenient way
                uns_xs = []
                uns_ys = []
                uns_times = []
                for pts in pts_tupl:
                    uns_xs += list(pts.xy[:, 0])
                    uns_ys += list(pts.xy[:, 1])
                    uns_times += list(pts.v)
                # sort them by times
                self.times = times
                self.epsilon = epsilon
                self.epsilon2 = float(epsilon**2)
                self.points = []
                for time in self.times:
                    tmp_points = [Point(uns_xs[i], uns_ys[i], time=time)
                                  for i in range(len(uns_times))
                                  if time == uns_times[i]]
                    self.points.append(tmp_points)

                # store in Point objects
                del uns_xs, uns_ys, uns_times
                #  get unities
                self.unit_x = pts_tupl[0].unit_x
                self.unit_y = pts_tupl[0].unit_y
                self.unit_v = pts_tupl[0].unit_v
                # prepare some storage for lines
                self.closed_lines = []
                self.open_lines = []
                self.curr_time = 0

            def get_points_at_time(self, time):
                """
                Return all the points for a given time.
                If 'init' is True, get the points from the initial points
                field.
                """
                return self.points[time]

            def get_first_lines(self):
                """
                Construct some lines for the first time step
                """
                for pt in self.get_points_at_time(0):
                    self.open_lines.append(Line(pt))

            def make_one_step(self):
                """
                Use the points from the next step to fill the Lines
                """
                # get points to add
                new_pts = self.get_points_at_time(self.curr_time + 1)
                # get distances matrix
                dist2_mat = np.empty(shape=(len(new_pts),
                                            len(self.open_lines)))
                for i in range(len(new_pts)):
                    pt = new_pts[i]
                    for j in range(len(self.open_lines)):
                        last_pt = self.open_lines[j].points[-1]
                        dist2_mat[i, j] = pt.dist2(last_pt)
                # track used pt and lines
                used_lines = np.zeros(len(self.open_lines), dtype=bool)
                used_pts = np.zeros(len(new_pts), dtype=bool)
                # check dist2 matrix size
                if dist2_mat.shape[0] != 0 and dist2_mat.shape[1] != 0:
                    usable_mat = True
                else:
                    usable_mat = False
                # loop over min values of dist2_mat
                while usable_mat:
                    indmin_pt, indmin_line \
                        = np.unravel_index(np.argmin(dist2_mat),
                                           dist2_mat.shape)
                    # if dist to big or nothing remaining, we stop
                    if (dist2_mat[indmin_pt, indmin_line] > epsilon or
                            dist2_mat[indmin_pt, indmin_line] == np.inf):
                        break
                    # append pt to Line
                    self.open_lines[indmin_line].add_point(new_pts[indmin_pt])
                    # remove pt and line from dist2_mat
                    dist2_mat[indmin_pt, :] = np.inf
                    dist2_mat[:, indmin_line] = np.inf
                    used_lines[indmin_line] = True
                    used_pts[indmin_pt] = True
                # close the lines that have not a new pt
                for i in range(len(used_lines))[::-1]:
                    if not used_lines[i]:
                        self.closed_lines.append(self.open_lines[i])
                        del self.open_lines[i]
                # create new line for remaining points
                new_pts = np.asarray(new_pts)
                for pt in new_pts[np.logical_not(used_pts)]:
                    self.open_lines.append(Line(init_pt=pt))
                # ready for the next step
                self.curr_time += 1

            def make_all_steps(self):
                """
                Put all points into the Lines.
                """
                # create the first lines
                self.get_first_lines()
                # make times steps
                for i in range(len(self.times) - 1):
                    self.make_one_step()
                # close all remaining Lines
                for line in self.open_lines:
                    self.closed_lines.append(line)
                self.open_lines = []

            def get_trajectories(self):
                """
                return the trajectories under the form of sorted Points objects
                """
                # compute Lines
                self.make_all_steps()
                # export Lines to Points objects
                pts = []
                for line in self.closed_lines:
                    pts.append(line.export_to_Points(unit_x=self.unit_x,
                                                     unit_y=self.unit_y,
                                                     unit_v=self.unit_v))
                # sort trajectories by length
                if len(pts) > 1:
                    lengths = [len(pt.xy) for pt in pts]
                    ind_sort = np.argsort(lengths)[::-1]
                    pts = np.asarray(pts)
                    pts = pts[ind_sort]
                return pts

            @staticmethod
            def get_closer_point(pt, pts):
                """
                """
                pts = np.asarray(pts)
                dist2 = [pt.dist2(opt) for opt in pts]
                ind_min = np.argmin(dist2)
                return ind_min, dist2[ind_min]

        # local class line to store vortex center evolution line
        class Line(object):
            """
            Class representing a line, defined by a set of ordened points.
            """

            def __init__(self, init_pt):
                self.points = [init_pt]

            def add_point(self, pts):
                """
                Add a new point to the line.
                """
                self.points.append(pts)

            def export_to_Points(self, unit_x, unit_y, unit_v):
                """
                Export the current line to a Points object.
                """
                xy = []
                v = []
                for pt in self.points:
                    xy.append([pt.x, pt.y])
                    v.append(pt.time)
                points = Points(xy, v=v, unit_x=unit_x, unit_y=unit_y,
                                unit_v=unit_v)
                return points

        # local class point to store one point
        class Point(object):
            """
            Class representing a point with a value on it.
            """
            def __init__(self, x, y, time):
                self.x = x
                self.y = y
                self.time = time

            def __repr__(self):
                return "{}, {}".format(self.x, self.y)

            def dist2(self, pt):
                return (self.x - pt.x)**2 + (self.y - pt.y)**2
        # Getting the vortex centers trajectory
        PF = PointField(points, epsilon, times)
        pts = PF.get_trajectories()
        return pts

    def display_arch(self, indice=None, time=None, field=None,
                     cpkw={}, lnkw={}):
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
                raise ValueError("You should specify a time or an indice")
        if time is not None and indice is not None:
            raise ValueError()
        if time is not None:
            indice = self._get_indice_from_time(time)
        elif indice is not None:
            if not isinstance(indice, int):
                raise TypeError()
        if field is not None:
            if not isinstance(field, VectorField):
                raise TypeError()
        # Set default params
        if 'color' in list(cpkw.keys()):
            colors = [cpkw.pop('color')]*len(self.colors)
        else:
            colors = self.colors
        if "marker" in list(cpkw.keys()):
            pass
        else:
            cpkw['marker'] = 'o'
        # display the critical lines
        if (field is not None and len(self.sadd[indice].xy) != 0 and
                isinstance(self.sadd[indice], OrientedPoints)):
            if 'color' not in list(lnkw.keys()):
                lnkw['color'] = colors[4]
            streams = self.sadd[indice]\
                .get_streamlines_from_orientations(
                    field, reverse_direction=[True, False],
                    interp='cubic')
            for stream in streams:
                stream._display(kind='plot', **lnkw)
        # loop on the points types
        for i, pt in enumerate(self.iter):
            if pt[indice] is None:
                continue
            pt[indice].display(kind='plot', color=colors[i],
                               linestyle='none', axe_x='x', axe_y='y', **cpkw)

    def display(self, fields=None, cpkw={}, lnkw={}, display_traj=False,
                buffer_size=100, **kwargs):
        """
        Display some critical points.

        Parameters
        ----------
        fields : TemporalVectorFields object, optional
            If specified, critical points are displayed on the given field.
            critical lines are also computed and displayed
        """
        # check parameters
        if fields is not None:
            if len(self.times) == 1:
                if isinstance(fields, TemporalVectorFields):
                    if len(fields.fields) != 1:
                        raise ValueError()
                else:
                    if not isinstance(fields, VectorField):
                        raise TypeError()
                fields = [fields]
            else:
                if not isinstance(fields, TemporalVectorFields):
                    raise TypeError()
                fields = fields.fields
        # Set default params
        if 'color' in list(cpkw.keys()):
            colors = [cpkw.pop('color')]*len(self.colors)
        else:
            colors = self.colors
        if "marker" in list(cpkw.keys()):
            pass
        else:
            cpkw['marker'] = 'o'
        dbs = []
        # If trajectories are also asked
        if display_traj:
            # check if trajectorie are computed
            if self.iter_traj[0] is None:
                raise Exception("You need to compute trajectory first,"
                                " with 'compute_trajectory' method")
            # TODO : Absolutely not optimized
            times = self.times
            for i, kind in enumerate(self.iter_traj):
                color = np.array(mpl.colors.colorConverter.to_rgb(colors[i]))
                color = color + (1 - color)*.5
                args = {'color': color, 'kind': 'plot', 'marker': None,
                        'lw': 3}
                for traj in kind:
                    if len(traj.xy) == 0:
                        print('pb here')
                        continue
                    xs = []
                    ys = []
                    x = traj.xy[:, 0]
                    y = traj.xy[:, 1]
                    for time in times:
                        if time in traj.v:
                            xs.append(x)
                            ys.append(y)
                        else:
                            xs.append([])
                            ys.append([])
                    dbs.append(pplt.Displayer(xs, ys, **args))
        # create lines if necessary
        if fields is not None and isinstance(self.sadd[0], OrientedPoints):
            # getting the streamlines
            for field, opts in zip(fields, self.sadd):
                if len(opts.xy) == 0:
                    continue
                v = opts.v[0]
                tmp_stream = opts.get_streamlines_from_orientations(
                    field, reverse_direction=[False, True], interp='cubic')
                if "color" in list(lnkw.keys()):
                    tmp_color = lnkw.pop('color')
                else:
                    tmp_color = colors[2]
                for st in tmp_stream:
                    data_streamx = [[]]*len(fields)
                    data_streamy = [[]]*len(fields)
                    ind_time = np.where(v == self.times)[0][0]
                    data_streamx[ind_time] = st.xy[:, 0]
                    data_streamy[ind_time] = st.xy[:, 1]
                    tmp_db = pplt.Displayer(data_streamx,
                                            data_streamy,
                                            color=tmp_color, kind='plot',
                                            **lnkw)
                    dbs.append(tmp_db)
        # display points
        for i, pt in enumerate(self.iter):
            # get data
            color = colors[i]
            x = [list(tmp_pt.xy[:, 0]) for tmp_pt in pt]
            y = [list(tmp_pt.xy[:, 1]) for tmp_pt in pt]
            # check if there is data
            if len(np.concatenate(x)) == 0:
                continue
            # default args
            args = {'mfc': color, 'kind': 'plot', 'mec': 'k',
                    'ls': 'none'}
            if 'kind' in list(cpkw.keys()):
                if cpkw['kind'] != 'plot':
                    args = {}
            args.update(cpkw)
            dbs.append(pplt.Displayer(x, y, buffer_size=buffer_size, **args))
        # plot
        if len(self.times) == 1:
            for db in dbs:
                db.draw(0, rescale=False)
        else:
            pplt.ButtonManager(dbs)
        return dbs

    def display_traj(self, data='default', filt=None, **kw):
        """
        Display the stored trajectories.

        Parameters
        ----------
        data : string
            If 'default', trajectories are plotted in a 2-dimensional plane.
            If 'x', x position of cp are plotted against time.
            If 'y', y position of cp are plotted against time.
        filt : array of boolean
            Filter on CP types.
        kw : dict, optional
            Arguments passed to plot.
        """
        # check if some trajectories are computed
        if self.current_epsilon is None:
            raise Exception("you must compute trajectories before "
                            "displaying them")
        if filt is None:
            filt = np.ones((5,), dtype=bool)
        if not isinstance(filt, ARRAYTYPES):
            raise TypeError()
        filt = np.array(filt, dtype=bool)
        # display
        if 'color' in list(kw.keys()):
            colors = [kw.pop('color')]*len(self.colors)
        else:
            colors = self.colors
        if 'marker' not in list(kw.keys()):
            kw['marker'] = 'o'
        if 'linestyle' not in list(kw.keys()) and "ls" not in list(kw.keys()):
            kw['linestyle'] = '-'
        if data == 'default':
            for i, trajs in enumerate(self.iter_traj):
                color = colors[i]
                if trajs is None or not filt[i]:
                    continue
                if 'kind' not in list(kw.keys()):
                    kw['kind'] = 'plot'
                for traj in trajs:
                    traj.display(color=color, **kw)
            plt.xlabel('x {}'.format(self.unit_x.strUnit()))
            plt.ylabel('y {}'.format(self.unit_y.strUnit()))
        elif data == 'x':
            for i, trajs in enumerate(self.iter_traj):
                color = colors[i]
                if trajs is None or not filt[i]:
                    continue
                for traj in trajs:
                    plt.plot(traj.xy[:, 0], traj.v[:], color=color,
                             **kw)
            plt.xlabel('x {}'.format(self.unit_x.strUnit()))
            plt.ylabel('time {}'.format(self.unit_time.strUnit()))
        elif data == 'y':
            for i, trajs in enumerate(self.iter_traj):
                color = colors[i]
                if trajs is None or not filt[i]:
                    continue
                for traj in trajs:
                    plt.plot(traj.v[:], traj.xy[:, 1], color=color,
                             **kw)
            plt.ylabel('time {}'.format(self.unit_time.strUnit()))
            plt.xlabel('y {}'.format(self.unit_y.strUnit()))
        else:
            raise Exception()

    def display_3D(self, xlabel='', ylabel='', zlabel='', title='',
                   **plotargs):
        """
        """
        # loop on the points types
        for i, kind in enumerate(self.iter):
            for pts in kind:
                if pts is None:
                    continue
                pts.display3D(kind='plot', marker='o', linestyle='none',
                              color=self.colors[i], **plotargs)

    def display_traj_3D(self, xlabel='', ylabel='', zlabel='', title='',
                        **plotargs):
        """
        """
        # loop on the points types
        for i, kind in enumerate(self.iter_traj):
            for pts in kind:
                if pts is None:
                    continue
                pts.display3D(kind='plot', marker=None, linestyle='-',
                              color=self.colors[i], **plotargs)

    def display_animate(self, TF, **kw):
        """
        Display an interactive windows with the velocity fields from TF and the
        critical points.

        TF : TemporalFields
            .
        kw : dict, optional
            Additional arguments for 'TF.display()'
        """
        return TF.display(suppl_display=self.__update_cp, **kw)

    def __update_cp(self, ind):
        self.display(indice=ind)


class TopoPoints(object):
    """
    Represent a topological points field.
    (only for topo simplification, in fact, should be merged with CritPoints)

    Parameters
    ----------
    xy : Nx2 array of numbers
        Points positions
    types : Nx1 array of integers
        Type code (1:focus, 2:focus_c, 3:saddle, 4:node_i, 5:node_o)
    """
    # TODO : +++ Implement orientation conservation and computation +++s
    def __init__(self):
        self.xy = []
        self.types = []
        self.pbis = []
        self.dim = 0
        self.dist2 = []

    def import_from_arrays(self, xy, types):
        # check parameters
        try:
            xy = np.array(xy, dtype=float)
            types = np.array(types, dtype=int)
        except ValueError:
            raise TypeError()
        if xy.ndim != 2:
            raise ValueError()
        if types.ndim != 1:
            raise ValueError()
        if xy.shape[1] != 2:
            raise ValueError()
        if not xy.shape[0] != types.shape:
            raise ValueError()
        # storing
        self.xy = xy
        self.types = types
        self.pbis = np.zeros(self.types.shape)
        self.pbis[self.types == 1] = 1
        self.pbis[self.types == 2] = 1
        self.pbis[self.types == 3] = -1
        self.pbis[self.types == 4] = 1
        self.pbis[self.types == 5] = 1
        self.dim = self.xy.shape[0]
        # get dist grid
        self.dist2 = self._get_dist2()

    def import_from_CP(self, CP_obj, wanted_ind):
        """
        Import from a CritPoints object, the critical points from the time
        associated with the given indice.
        """
        # check params
        if not isinstance(CP_obj, CritPoints):
            raise TypeError()
        if not isinstance(wanted_ind, int):
            raise TypeError()
        if wanted_ind > len(CP_obj.foc) - 1:
            raise ValueError()
        # get data
        xy = []
        types = []
        for i, typ in enumerate([CP_obj.foc, CP_obj.foc_c, CP_obj.sadd,
                                 CP_obj.node_i, CP_obj.node_o]):
            for pt in typ[wanted_ind].xy:
                xy.append(pt)
                types.append(i + 1)
        self.import_from_arrays(xy, types)

    def _get_dist2(self):
        # initialize array
        dist2 = np.zeros((self.dim, self.dim), dtype=float)
        dist2.fill(np.inf)
        # loop on points
        for i in np.arange(self.dim - 1):
            for j in np.arange(self.dim)[i + 1::]:
                dist2[i, j] = np.sum((self.xy[i] - self.xy[j])**2)
        return dist2

    def simplify(self, dist_min, kind='replacement'):
        """
        Simplify the topological points field.

        Parameters
        ----------
        dist_min : number
            Minimal distance between two points in the simplified field.
        kind : string, optional
            Algorithm used to simplify the field, can be 'replacement'(default)
            for iterative replacement with center of mass or 'only_delete' to
            eliminate only the first order associates.

        Returns
        -------
        simpl_TP : TopoPoints object
            Simplified topological points field.
        """
        # check params
        try:
            dist_min = float(dist_min)
        except ValueError:
            raise TypeError()
        if kind not in ['replacement', 'only_delete']:
            raise ValueError()
        # get dist2
        dist_min2 = dist_min**2
        dist2 = self.dist2.copy()
        xy = self.xy.copy()
        weight = np.ones(xy.shape, dtype=int)
        filt = np.ones(self.dim, dtype=bool)
        types = self.types.copy()
        types = [[typ] for typ in types]
        pbis = self.pbis.copy()
        # simplication loop
        while True:
            # get min position
            amin = np.argmin(dist2)
            ind_1 = int(amin/self.dim)
            ind_2 = amin % self.dim
            # check if we can stop simplification
            if dist2[ind_1, ind_2] > dist_min2:
                break
            # ignore the points and search others
            if kind == 'only_delete':
                if pbis[ind_1] + pbis[ind_2] == 0:
                    filt[ind_1] = False
                    filt[ind_2] = False
                    dist2[ind_1, :] = np.inf
                    dist2[:, ind_1] = np.inf
                    dist2[ind_2, :] = np.inf
                    dist2[:, ind_2] = np.inf
                else:
                    dist2[ind_1, ind_2] = np.inf
                    dist2[ind_2, ind_1] = np.inf
            # add a new point at the center of the two else
            elif kind == 'replacement':
                new_coord = ((xy[ind_1]*weight[ind_1] +
                              xy[ind_2]*weight[ind_2]) /
                             (weight[ind_1] + weight[ind_2]))
                # remove the pooints
                filt[ind_1] = False
                filt[ind_2] = False
                dist2[ind_1, :] = np.inf
                dist2[:, ind_1] = np.inf
                dist2[ind_2, :] = np.inf
                dist2[:, ind_2] = np.inf
                # compute new distance
                new_dists = np.zeros((self.dim), dtype=float)
                new_dists.fill(np.inf)
                for i in np.arange(self.dim):
                    if filt[i]:
                        new_dists[i] = np.sum((xy[i] - new_coord)**2)
                # store new distance and other properties
                filt[ind_1] = True
                xy[ind_1] = new_coord
                weight[ind_1] = weight[ind_1] + weight[ind_2]
                types[ind_1] = types[ind_1] + types[ind_2]
                pbis[ind_1] = pbis[ind_1] + pbis[ind_2]
                dist2[ind_1, :] = new_dists
                dist2[:, ind_1] = new_dists
        # remove useless lines
        xy = xy[filt]
        types = [types[i] for i in np.arange(len(types)) if filt[i]]
        pbis = pbis[filt]
        # remove the points if they are equivalent to uniform flow
        pbi_filt = pbis != 0
        xy = xy[pbi_filt]
        types = [types[i] for i in np.arange(len(types)) if pbi_filt[i]]
        pbis = pbis[pbi_filt]
        # try to get informations on equivalent points type
        new_types = []
        if 'replacement':
            for i, typ in enumerate(types):
                if len(typ) == 1:
                    new_types.append(typ[0])
                elif pbis[i] not in [-1, 1]:
                    print("simplification radius too big,"
                          " equivalent pbi too high")
                    new_types.append(0)
                elif pbis[i] == -1:
                    new_types.append(3)
                else:
                    nmb_1 = np.sum(np.array(typ) == 1)
                    nmb_2 = np.sum(np.array(typ) == 2)
                    nmb_4 = np.sum(np.array(typ) == 4)
                    nmb_5 = np.sum(np.array(typ) == 5)
                    ind_typ = np.argmax([nmb_1, nmb_2, 0, nmb_4, nmb_5])
                    new_types.append(ind_typ + 1)
            types = new_types
        # return
        new_tp = TopoPoints()
        new_tp.import_from_arrays(xy, types)
        return new_tp

    def NL_simplifiy(self, window_size):
        """
        Simplify the topological field using Non-local criterions.

        Parameters
        ----------
        window_size : number
            Window size used to compute non-local criterions.
        """

    def display(self):
        plt.figure()
        plt.scatter(self.xy[:, 0], self.xy[:, 1], c=self.types, vmax=6,
                    vmin=0, s=50)
        plt.colorbar()


class MeanTrajectory(Points):

    def __init__(self, xy=np.empty((0, 2), dtype=float), time=[],
                 assoc_real_times=[], assoc_std_x=[], assoc_std_y=[],
                 nmb_traj_used=0, base_trajs=[],
                 unit_x='', unit_y='',
                 unit_times='', name=''):
        """
        Representing an averaged trajecory
        (mean trajectory over a set of trajectories)

        Parameters
        ----------
        xy : nx2 array.
            Representing the coordinates of each point of the set (n points).
        time : n array, optional
            Representing time at each points.
        assoc_real_times : nxm array
            Representing the real times (of the original trajectories)
            associated with the mean trajectory points.
        assoc_std : nx1 array
            Representing the std at each mean trajectory points
        nmb_traj_used : number
            Number of trajectories used to average
        base_trajs : tuple of Points objects
            Trajectories used to compute this mean trajectory
        unit_x : Unit object, optional
            X unit.
        unit_y : Unit object, optional
            Y unit.
        unit_times : Unit object, optional
            time unit.
        name : string, optional
            Name of the points set
        """
        super(MeanTrajectory, self).__init__(xy=xy, v=time, unit_x=unit_x,
                                             unit_y=unit_y,
                                             unit_v=unit_times, name=name)
        # check
        if not isinstance(assoc_real_times, ARRAYTYPES):
            raise TypeError()
        if not len(assoc_real_times) == len(xy):
            raise ValueError()
        self.assoc_real_times = assoc_real_times
        self.assoc_std_x = np.array(assoc_std_x, dtype=float)
        self.assoc_std_y = np.array(assoc_std_y, dtype=float)
        self.nmb_traj_used = int(nmb_traj_used)
        self.base_trajs = base_trajs

    @property
    def time(self):
        return self.v

    @time.setter
    def time(self, values):
        self.v = values

    @property
    def unit_times(self):
        return self.unit_v

    @unit_times.setter
    def unit_times(self, unit):
        self.unit_v = unit

    @property
    def used_traj_number(self):
        return float(len(self.base_trajs))

    @property
    def used_pts_number(self):
        return float(np.sum([len(tr) for tr in self.base_trajs]))

    def crop(self, intervx=None, intervy=None, intervt=None, inplace=True,
             ind=False):
            """
            Crop the points cloud.

            Parameters
            ----------
            intervx : 2x1 tuple
                Interval on x axis
            intervy : 2x1 tuple
                Interval on y axis
            intervt : 2x1 tuple
                Interval on time

            Returns
            -------
            tmp_pts : Points object
                croped version of the point cloud.
            """
            if inplace:
                tmp_pts = self
            else:
                tmp_pts = self.copy()
            # Getting cropping mask
            if ind:
                if intervx is not None or intervy is not None:
                    raise ValueError()
                mask = np.ones(len(self.xy), dtype=bool)
                mask[intervt[0]:intervt[1]] = False
            else:
                mask = np.zeros(len(self.xy), dtype=bool)
                if intervx is not None:
                    out_zone = np.logical_or(tmp_pts.xy[:, 0] < intervx[0],
                                             tmp_pts.xy[:, 0] > intervx[1])
                    mask = np.logical_or(mask, out_zone)
                if intervy is not None:
                    out_zone = np.logical_or(tmp_pts.xy[:, 1] < intervy[0],
                                             tmp_pts.xy[:, 1] > intervy[1])
                    mask = np.logical_or(mask, out_zone)
                if intervt is not None and len(tmp_pts.v) != 0:
                    out_zone = np.logical_or(tmp_pts.v < intervt[0],
                                             tmp_pts.v > intervt[1])
                    mask = np.logical_or(mask, out_zone)
            # crop
            tmp_pts.assoc_real_times = [tmp_pts.assoc_real_times[i]
                                        for i in range(len(tmp_pts.
                                                           assoc_real_times))
                                        if not mask[i]]
            tmp_pts.assoc_std_x = tmp_pts.assoc_std_x[~mask]
            tmp_pts.assoc_std_y = tmp_pts.assoc_std_y[~mask]
            # crop mean_traj
            super(MeanTrajectory, tmp_pts).crop(intervx=intervx,
                                                intervy=intervy,
                                                intervv=intervt,
                                                inplace=True,
                                                ind=ind)
            # DO NOT crop base fields
            # return
            if not inplace:
                return tmp_pts

    def scale(self, scalex=1., scaley=1., scalet=1., inplace=False):
        if inplace:
            tmp_mt = self
        else:
            tmp_mt = self.copy()
        # scaling base trajectories
        for i in range(len(tmp_mt.base_trajs)):
            tmp_mt.base_trajs[i].scale(scalex=scalex, scaley=scaley,
                                       scalev=scalet, inplace=True)
        # scaling associated values
        if isinstance(scalex, unum.Unum):
            new_unit = scalex*self.unit_x
            tmp_scalex = new_unit.asNumber()
        else:
            tmp_scalex = scalex
        if isinstance(scaley, unum.Unum):
            new_unit = scaley*self.unit_y
            tmp_scaley = new_unit.asNumber()
        else:
            tmp_scaley = scaley
        if isinstance(scalet, unum.Unum):
            new_unit = scalet*self.unit_v
            tmp_scalet = new_unit.asNumber()
        else:
            tmp_scalet = scalet
        for i in range(len(tmp_mt.assoc_real_times)):
            tmp_mt.assoc_real_times[i] *= tmp_scalet
        tmp_mt.assoc_std_x = np.array(tmp_mt.assoc_std_x)*tmp_scalex
        tmp_mt.assoc_std_y = np.array(tmp_mt.assoc_std_y)*tmp_scaley
        # sclaing points attributes
        super(MeanTrajectory, tmp_mt).scale(scalex=scalex, scaley=scaley,
                                            scalev=scalet, inplace=True)
        # returning
        if not inplace:
            return tmp_mt

    def _display(self, *args, **kwargs):
        return super(MeanTrajectory, self)._display(*args, **kwargs)

    def display(self, *args, **kwargs):
        return super(MeanTrajectory, self).display(*args, **kwargs)

    def display_error_bars(self, kind='bar', **kwargs):
        """
        Display the error bar according to the trajectories used to compute the
        mean trajectory.

        Parameters
        ----------
        kind : string in ['bar', 'envelope']
            If 'bar', display error bar,
            if 'envelope', display envelope around trajectory
        """
        # get data
        if "axe_x" in list(kwargs.keys()):
            axe = kwargs.pop("axe_x")
            if axe == "x":
                axe_x = self.xy[:, 0]
                axe_x_err = self.assoc_std_x
            elif axe == "y":
                axe_x = self.xy[:, 1]
                axe_x_err = self.assoc_std_y
            elif axe in ["v", 'time']:
                axe_x = self.v
                axe_x_err = None
            else:
                raise ValueError()
        else:
            axe_x = self.xy[:, 0]
            axe_x_err = self.assoc_std_x
        if "axe_y" in list(kwargs.keys()):
            axe = kwargs.pop("axe_y")
            if axe == "x":
                axe_y = self.xy[:, 0]
                axe_y_err = self.assoc_std_x
            elif axe == "y":
                axe_y = self.xy[:, 1]
                axe_y_err = self.assoc_std_y
            elif axe in ["v", 'time']:
                axe_y = self.v
                axe_y_err = None
            else:
                raise ValueError()
        else:
            axe_y = self.xy[:, 1]
            axe_y_err = self.assoc_std_y
        # set default properties
        if kind == 'bar':
            if 'fmt' not in list(kwargs.keys()):
                kwargs['fmt'] = 'none'
            if 'color' in list(kwargs.keys()):
                kwargs['ecolor'] = kwargs.pop('color')
            if 'ecolor' not in list(kwargs.keys()):
                kwargs['ecolor'] = 'k'
            plt.errorbar(axe_x, axe_y, xerr=axe_x_err,
                         yerr=axe_y_err, **kwargs)
        elif kind == 'envelope':
            if 'alpha' in list(kwargs.keys()):
                alpha = kwargs.pop('alpha')
            else:
                alpha = np.inf
            pts = Points()
            for x, y, epsx, epsy in zip(axe_x, axe_y, axe_x_err, axe_y_err):
                pts.add([x + epsx, y + epsy])
                pts.add([x + epsx, y - epsy])
                pts.add([x - epsx, y + epsy])
                pts.add([x - epsx, y - epsy])
            env = pts.get_envelope(alpha=alpha)
            env.display(**kwargs)
        else:
            raise ValueError()

    def reconstruct_fields(self, TF):
        """
        Do a conditionnal averaging based on the CP positions.

        Parameters
        ----------
        TF : TemporalFields
            .
        """
        fin_tf = TF.__class__()
        for av_t, real_ts in zip(self.time, self.assoc_real_times):
            # get real times indices for this averaged time
            inds = []
            for real_t in real_ts:
                inds.append(np.where(TF.times == real_t)[0][0])
            # sum the fields on those inds to make conditionnal averaging
            tmp_field = TF.fields[inds[0]]
            for ind in inds[1::]:
                tmp_field += TF.fields[ind]
            tmp_field /= len(inds)
            # add this conditionnal averaged field to the set
            fin_tf.add_field(tmp_field, time=av_t, unit_times=self.unit_times)
        # return
        return fin_tf


def get_critical_points(obj, time=0, unit_time='', window_size=4,
                        kind='pbi', mirroring=None, mirror_interp='linear',
                        smoothing_size=0, verbose=False, thread=1):
    """
    For a VectorField of a TemporalVectorField object, return the critical
    points positions and informations on their type.

    Parameters
    ----------
    obj : VectorField or TemporalVectorFields objects
        Base vector field(s)
    time : number, optional
        if 'obj' is a VectorField, 'time' is the time associated to the field.
    unit_time : string or Unum.units object
        Unit for 'time'
    window_size : integer, optional
        Size of the interrogation windows for the computation of critical
        point position (defaut : 4).
    kind : string, optional
        Method used to compute the critical points position.
        can be : 'pbi_cell' for simple (fast) Poincarre-Bendixson sweep.
        'pbi' for PB sweep and bi-linear interpolation inside cells.
        'pbi_crit' for PB sweep and use of non-local criterions.
        'gam_vort' for gamma criterion extremum detection (only detect vortex).
    mirroring : array of numbers
        If specified, use mirroring to get the critical points on the
        eventual walls. should be an array of
        '[direction ('x' or 'y'), position]*N'.
    mirror_interp : string, optional
        Method used to fill the gap at the wall.
        ‘value’ : fill with the given value,
        ‘nearest’ : fill with the nearest value,
        ‘linear’ (default): fill using linear interpolation
        (Delaunay triangulation),
        ‘cubic’ : fill using cubic interpolation (Delaunay triangulation)
    smoothing_size : number, optional
        If specified, a gaussian smoothing of the wanted size is used before
        detecting the CP.
    verbose : boolean, optional
        If 'True', display message on CP detection advancement.
    thread : integer or 'all'
            Number of thread to use (multiprocessing).
            (Only implemented for 'kind = 'pbi'')
    Notes
    -----
    If the fields have masked values, saddle streamlines ar not computed.
    """
    # check parameters
    if not isinstance(time, NUMBERTYPES):
        raise TypeError()
    if not isinstance(unit_time, STRINGTYPES + (unum.Unum,)):
        raise TypeError()
    if not isinstance(window_size, NUMBERTYPES):
        raise TypeError()
    window_size = int(window_size)
    if not isinstance(kind, STRINGTYPES):
        raise TypeError()
    if kind not in ['pbi', 'pbi_crit', 'pbi_cell', 'gam_vort']:
        raise ValueError()
    if mirroring is not None:
        if not isinstance(mirroring, ARRAYTYPES):
            raise TypeError()
        if len(mirroring) == 0:
            mirroring = None
        else:
            if np.array(mirroring).ndim != 2:
                raise ValueError()
            if np.array(mirroring).shape[1] != 2:
                raise ValueError()
    if mirror_interp is not None:
        if not isinstance(mirror_interp, STRINGTYPES):
            raise TypeError()
        if mirror_interp not in ['linear', 'value', 'nearest', 'cubic']:
            raise ValueError()
    if not isinstance(smoothing_size, NUMBERTYPES):
        raise TypeError()
    if smoothing_size < 0:
        raise ValueError()
    if not isinstance(thread, int):
        if thread not in ['all']:
            raise ValueError()
    # check if mask (not fully supported yet)
    if np.any(obj.mask):
        raise ValueError("Should not have masked values")
    # if obj is a vector field
    if isinstance(obj, VectorField):
        # mirroring if necessary
        if mirroring is not None:
            tmp_vf = obj.copy()
            for direction, position in mirroring:
                if kind == 'pbi_crit':
                    tmp_vf.mirroring(direction, position,
                                     inds_to_mirror=window_size*2,
                                     inplace=True, interp=mirror_interp,
                                     value=[0, 0])
                else:
                    tmp_vf.mirroring(direction, position,
                                     inds_to_mirror=2,
                                     inplace=True, interp=mirror_interp,
                                     value=[0, 0])
        else:
            tmp_vf = obj
        # smoothing if necessary
        if smoothing_size != 0:
            tmp_vf.smooth(tos='gaussian', size=smoothing_size, inplace=True)
        # get cp positions
        if kind == 'pbi':
            res = _get_cp_pbi_on_VF(tmp_vf, time=time, unit_time=unit_time,
                                    window_size=window_size, thread=thread)
        elif kind == 'pbi_cell':
            res = _get_cp_cell_pbi_on_VF(tmp_vf, time=time,
                                         unit_time=unit_time,
                                         window_size=window_size)
        elif kind == 'pbi_crit':
            res = _get_cp_crit_on_VF(tmp_vf, time=time, unit_time=unit_time,
                                     window_size=window_size)
        elif kind == 'gam_vort':
            res = _get_gamma2_vortex_center_on_VF(tmp_vf, time=time,
                                                  unit_time=unit_time,
                                                  radius=window_size)
        else:
            raise ValueError
        # removing critical points outside of the field
        if mirroring is not None:
            x_median = (tmp_vf.axe_x[-1] + tmp_vf.axe_x[0])/2.
            y_median = (tmp_vf.axe_y[-1] + tmp_vf.axe_y[0])/2.
            intervx = np.array([-np.inf, np.inf])
            intervy = np.array([-np.inf, np.inf])
            axe_x, axe_y = tmp_vf.axe_x, tmp_vf.axe_y
            for direction, position in mirroring:
                if direction == 'x':
                    if position < x_median and position > intervx[0]:
                        ind = tmp_vf.get_indice_on_axe(1, position)[0]
                        intervx[0] = axe_x[ind - 1]
                    elif position < intervx[1]:
                        ind = tmp_vf.get_indice_on_axe(1, position)[1]
                        intervx[1] = axe_x[ind + 1]
                else:
                    if position < y_median and position > intervy[0]:
                        ind = tmp_vf.get_indice_on_axe(2, position)[0]
                        intervy[0] = axe_y[ind - 1]
                    elif position < intervy[1]:
                        ind = tmp_vf.get_indice_on_axe(2, position)[1]
                        intervy[1] = axe_y[ind + 1]
            res.crop(intervx=intervx, intervy=intervy, ind=False, inplace=True)
    # if obj is vector fields
    elif isinstance(obj, TemporalVectorFields):
        res = CritPoints(unit_time=obj.unit_times)
        PG = ProgressCounter(init_mess="Begin CP detection", end_mess='Done',
                             nmb_max=len(obj.fields),
                             name_things='fields')

        # function o ge cp on one field (used for multiprocessing)
        def get_cp_on_one_field(args):
            if verbose:
                PG.print_progress()
            field, time = args
            res = get_critical_points(field, time=time,
                                      unit_time=obj.unit_times,
                                      window_size=window_size,
                                      kind=kind, mirroring=mirroring,
                                      smoothing_size=smoothing_size,
                                      thread=1)
            return res
        # Mapping with multiprocess or not
        if thread == 1:
            for i in np.arange(len(obj.fields)):
                res += get_cp_on_one_field((obj.fields[i], obj.times[i]))
        else:
            if thread == 'all':
                pool = Pool()
            else:
                pool = Pool(thread)
            res = pool.map_async(get_cp_on_one_field, list(zip(obj.fields,
                                                               obj.times)))
            pool.close()
            pool.join()
            res = res.get()
        res = np.sum(res)
    else:
        raise TypeError()
    return res


def get_vortex_position(obj, criterion=get_residual_vorticity,
                        criterion_args={}, threshold=0.5,
                        rel=True):
    """
    Return the position of the vortex (according to the given criterion) on
    vector field(s).

    Parameters
    ----------
    vectorfield : VectorField or TemporalVectorFields object
        .
    criterion : function
        Criterion used to highlight vortex position. Should be a function,
        taking a VectorField object and returning a ScalarField object.
    criterion_args : dict
        Additional arguments to give to the criterion function
    threshold : number
        Threshold value determining the vortex zone.
    rel : Boolean
        If 'rel' is 'True' (default), 'threshold' is relative to
        the extremum values of the field.
        If 'rel' is 'False', 'threshold' is treated like an absolut
        values.
    """
    # vectorfield
    if isinstance(obj, VectorField):
        vort_c, vort = _get_vortex_position_on_VF(
            obj, criterion=criterion,
            criterion_args=criterion_args,
            threshold=threshold,
            rel=rel)
        cp = CritPoints(unit_time='')
        cp.add_point(foc=vort, foc_c=vort_c, time=0)
        return cp
    elif isinstance(obj, TemporalVectorFields):
        cp = CritPoints(unit_time=obj.unit_times)
        # loop on fields
        for i, field in enumerate(obj.fields):
            tmp_vort_c, tmp_vort = _get_vortex_position_on_VF(
                field,
                criterion=criterion,
                criterion_args=criterion_args,
                threshold=threshold,
                rel=rel)
            cp.add_point(foc=tmp_vort, foc_c=tmp_vort_c, time=obj.times[i])
        # returning
        return cp
    else:
        raise TypeError()


def _get_vortex_position_on_VF(vectorfield, criterion=get_residual_vorticity,
                               criterion_args={}, threshold=0.5, rel=True):
    """
    Return the vortex positions on a vector field, using the given criterion.

    Parameters
    ----------
    vectorfield : VectorField object
        .
    criterion : function
        Criterion used to highlight vortex position. Should be a function,
        taking a VectorField object and returning a ScalarField object.
    criterion_args : dict
        Additional arguments to give to the criterion function
    threshold : number
        Threshold value determining the vortex zone.
    rel : Boolean
        If 'rel' is 'True' (default), 'threshold' is relative to
        the extremum values of the field.
        If 'rel' is 'False', 'threshold' is treated like an absolut
        values.
    """
    # check
    if not isinstance(vectorfield, VectorField):
        raise TypeError()
    try:
        threshold = float(threshold)
    except:
        raise TypeError()
    if threshold < 0:
        raise ValueError()
    if not isinstance(rel, bool):
        raise TypeError()
    if rel and threshold > 1:
        raise ValueError()
    try:
        sf = criterion(vectorfield, **criterion_args)
        sf.crop_masked_border(inplace=True, hard=False)
    except:
        raise TypeError()
    if not isinstance(sf, ScalarField):
        raise TypeError()
    # get vortex positions
    val_max = np.max(np.abs(sf.values[~sf.mask]))
    if rel:
        threshold *= val_max
    # make sure the max value is superior to the threshold
    if val_max > threshold:
        bornes_n = [-val_max, -threshold]
        bornes_p = [threshold, val_max]
    else:
        bornes_n = [-threshold*1.1, -threshold]
        bornes_p = [threshold, threshold*1.1]
    vort = sf.get_zones_centers(bornes=bornes_n, rel=False, kind='ponderated')
    vort_c = sf.get_zones_centers(bornes=bornes_p, rel=False,
                                  kind='ponderated')
    return vort, vort_c


def _get_cp_pbi_on_VF(vectorfield, time=0, unit_time=make_unit(""),
                      window_size=4, thread=1):
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
    thread : integer or 'all'
            Number of thread to use (multiprocessing).
    Returns
    -------
    pts : CritPoints object
        Containing all critical points position

    """
    # checking parameters coherence
    if not isinstance(vectorfield, VectorField):
        raise TypeError("'vectorfield' must be a VectorField")
    if isinstance(unit_time, STRINGTYPES):
        unit_time = make_unit(unit_time)
    if np.any(vectorfield.mask):
        sadd_ori = False
    else:
        sadd_ori = True
    # using VF methods to get cp position and types
    field = velocityfield_to_vf(vectorfield, time)
    pos, cp_types = field.get_cp_position(thread=thread)
    if sadd_ori:
        sadd = OrientedPoints(unit_x=vectorfield.unit_x,
                              unit_y=vectorfield.unit_y,
                              unit_v=unit_time)
    else:
        sadd = Points(unit_x=vectorfield.unit_x,
                      unit_y=vectorfield.unit_y,
                      unit_v=unit_time)
    foc = Points(unit_x=vectorfield.unit_x,
                 unit_y=vectorfield.unit_y,
                 unit_v=unit_time)
    foc_c = Points(unit_x=vectorfield.unit_x,
                   unit_y=vectorfield.unit_y,
                   unit_v=unit_time)
    node_i = Points(unit_x=vectorfield.unit_x,
                    unit_y=vectorfield.unit_y,
                    unit_v=unit_time)
    node_o = Points(unit_x=vectorfield.unit_x,
                    unit_y=vectorfield.unit_y,
                    unit_v=unit_time)
    for i, t in enumerate(cp_types):
        if t == 0:
            if sadd_ori:
                tmp_pos = pos[i]
                ori = np.array(_get_saddle_orientations(vectorfield, tmp_pos))
                sadd.add(tmp_pos, orientations=ori, v=time)
            else:
                sadd.add(pos[i], v=time)
        elif t == 1:
            foc_c.add(pos[i], v=time)
        elif t == 2:
            foc.add(pos[i], v=time)
        elif t == 3:
            node_o.add(pos[i], v=time)
        elif t == 4:
            node_i.add(pos[i], v=time)
        else:
            raise Exception()
    # returning
    pts = CritPoints(unit_time=unit_time)
    pts.add_point(foc=foc, foc_c=foc_c, node_i=node_i, node_o=node_o,
                  sadd=sadd, time=time)
    return pts


def _get_cp_cell_pbi_on_VF(vectorfield, time=0, unit_time=make_unit(""),
                           window_size=1):
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
        Default is smallest (1).

    Returns
    -------
    pts : CritPoints object
        Containing all critical points position
    """
    # checking parameters coherence
    if not isinstance(vectorfield, VectorField):
        raise TypeError("'vectorfield' must be a VectorField")
    if isinstance(unit_time, STRINGTYPES):
        unit_time = make_unit(unit_time)
    if np.any(vectorfield.mask):
        sadd_ori = False
    else:
        sadd_ori = True
    # using VF methods to get cp position and types
    field = velocityfield_to_vf(vectorfield, time)
    pos, cp_types = field.get_cp_cell_position()
    if sadd_ori:
        sadd = OrientedPoints(unit_x=vectorfield.unit_x,
                              unit_y=vectorfield.unit_y,
                              unit_v=unit_time)
    else:
        sadd = Points(unit_x=vectorfield.unit_x,
                      unit_y=vectorfield.unit_y,
                      unit_v=unit_time)
    foc = Points(unit_x=vectorfield.unit_x,
                 unit_y=vectorfield.unit_y,
                 unit_v=unit_time)
    foc_c = Points(unit_x=vectorfield.unit_x,
                   unit_y=vectorfield.unit_y,
                   unit_v=unit_time)
    node_i = Points(unit_x=vectorfield.unit_x,
                    unit_y=vectorfield.unit_y,
                    unit_v=unit_time)
    node_o = Points(unit_x=vectorfield.unit_x,
                    unit_y=vectorfield.unit_y,
                    unit_v=unit_time)
    for i, t in enumerate(cp_types):
        if t == 0:
            if sadd_ori:
                tmp_pos = pos[i]
                ori = np.array(_get_saddle_orientations(vectorfield, tmp_pos))
                sadd.add(tmp_pos, orientations=ori)
            else:
                sadd.add(pos[i])
        elif t == 1:
            foc_c.add(pos[i])
        elif t == 2:
            foc.add(pos[i])
        elif t == 3:
            node_o.add(pos[i])
        elif t == 4:
            node_i.add(pos[i])
        else:
            raise Exception()
    # returning
    pts = CritPoints(unit_time=unit_time)
    pts.add_point(foc=foc, foc_c=foc_c, node_i=node_i, node_o=node_o,
                  sadd=sadd, time=time)
    return pts


def _get_cp_crit_on_VF(vectorfield, time=0, unit_time=make_unit(""),
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
    if isinstance(unit_time, STRINGTYPES):
        unit_time = make_unit(unit_time)
    if np.any(vectorfield.mask):
        sadd_ori = False
    else:
        sadd_ori = True
    # Getting pbi cp position and fields around ###
    VF_field = velocityfield_to_vf(vectorfield, time)
    cp_positions, cp_types = VF_field.get_cp_cell_position()
    radius = window_size/4.
    # creating velocityfields around critical points
    # and transforming into VectorField objects
    VF_tupl = []
    for i, cp_pos in enumerate(cp_positions):
        tmp_vf = VF_field.get_field_around_pt(cp_pos, window_size + 1)
        tmp_vf = tmp_vf.export_to_velocityfield()
        # treating small fields
        axe_x, axe_y = tmp_vf.axe_x, tmp_vf.axe_y
        if (len(axe_x) <= window_size + 1 or len(axe_y) <= window_size + 1 or
                np.any(tmp_vf.mask)):
            pass
        else:
            tmp_vf.PBI = int(cp_types[i] != 0)
            tmp_vf.assoc_ind = i
            VF_tupl.append(tmp_vf)
    # Sorting by critical points type ###
    VF_focus = []
    VF_nodes = []
    VF_saddle = []
    for VF in VF_tupl:
        # node or focus
        if VF.PBI == 1:
            # checking if node or focus
            VF.gamma1 = get_gamma(VF, radius=radius, ind=True)
            VF.kappa1 = get_kappa(VF, radius=radius, ind=True)
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
    # Computing focus positions (rotatives and contrarotatives) ###
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
                        cp_positions[VF.assoc_ind] = pts.xy[0]
                        cp_types[VF.assoc_ind] = 2
            # contrarotative vortex
            else:
                pts = _min_detection(tmp_gam)
                if pts is not None:
                    if pts.xy[0][0] is not None and len(pts) == 1:
                        cp_positions[VF.assoc_ind] = pts.xy[0]
                        cp_types[VF.assoc_ind] = 1
    # Computing nodes points positions (in or out) ###
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
                        cp_positions[VF.assoc_ind] = pts.xy[0]
                        cp_types[VF.assoc_ind] = 3
            # in nodes
            else:
                pts = _min_detection(tmp_kap)
                if pts is not None:
                    if pts.xy[0][0] is not None and len(pts) == 1:
                        cp_positions[VF.assoc_ind] = pts.xy[0]
                        cp_types[VF.assoc_ind] = 4
    # Computing saddle points positions and direction ###
    if len(VF_saddle) != 0:
        for VF in VF_saddle:
            tmp_iot = get_iota(VF, radius=radius, ind=True)
            pts = _min_detection(np.max(tmp_iot.values) - tmp_iot)
            if pts is not None:
                if pts.xy[0][0] is not None and len(pts) == 1:
                        cp_positions[VF.assoc_ind] = pts.xy[0]
                        cp_types[VF.assoc_ind] = 0
    # creating the CritPoints object for returning
    focus = Points(unit_x=vectorfield.unit_x,
                   unit_y=vectorfield.unit_y,
                   unit_v=unit_time)
    focus_c = Points(unit_x=vectorfield.unit_x,
                     unit_y=vectorfield.unit_y,
                     unit_v=unit_time)
    nodes_i = Points(unit_x=vectorfield.unit_x,
                     unit_y=vectorfield.unit_y,
                     unit_v=unit_time)
    nodes_o = Points(unit_x=vectorfield.unit_x,
                     unit_y=vectorfield.unit_y,
                     unit_v=unit_time)
    if sadd_ori:
        sadd = OrientedPoints(unit_x=vectorfield.unit_x,
                              unit_y=vectorfield.unit_y,
                              unit_v=unit_time)
    else:
        sadd = Points(unit_x=vectorfield.unit_x,
                      unit_y=vectorfield.unit_y,
                      unit_v=unit_time)
    for i, t in enumerate(cp_types):
        if t == 0:
            if sadd_ori:
                tmp_pos = cp_positions[i]
                ori = np.array(_get_saddle_orientations(vectorfield, tmp_pos))
                sadd.add(tmp_pos, orientations=ori)
            else:
                sadd.add(cp_positions[i])
        elif t == 1:
            focus_c.add(cp_positions[i])
        elif t == 2:
            focus.add(cp_positions[i])
        elif t == 3:
            nodes_o.add(cp_positions[i])
        elif t == 4:
            nodes_i.add(cp_positions[i])
        else:
            raise Exception()
    pts = CritPoints(unit_time=unit_time)
    pts.add_point(focus, focus_c, nodes_i, nodes_o, sadd, time=time)
    return pts


def _get_gamma2_vortex_center_on_VF(vectorfield, time=0,
                                    unit_time=make_unit(""),
                                    radius=4):
    """
    For a VectorField object, return the position of the vortex centers.
    This algorithm use extremum detection on gamma fields.

    Parameters
    ----------
    vectorfield : a VectorField object.
        .
    time : number, optional
        Time
    unit_time : units object, optional
        Time unit.
    radius : integer, optional
        Default is 4.

    Returns
    -------
    pts : CritPoints object
        Containing all critical points
    """
    if not isinstance(vectorfield, VectorField):
        raise TypeError("'VF' must be a VectorField")
    if isinstance(unit_time, STRINGTYPES):
        unit_time = make_unit(unit_time)
    # get gamma field
    gamma = get_gamma(vectorfield, radius=radius, ind=True, kind='gamma1')
    gamma.crop_masked_border(hard=True, inplace=True)
    # get zones centers
    centers_c = gamma.get_zones_centers(bornes=[-1., -2/np.pi], rel=False)
    centers_cv = gamma.get_zones_centers(bornes=[2/np.pi, 1.], rel=False)
    # return
    pts = CritPoints(unit_time=unit_time)
    pts.add_point(foc=centers_cv, foc_c=centers_c, time=time)
    return pts


def _min_detection(SF):
    """
    Only for use in 'get_cp_crit'.
    """
    # interpolation on the field
    if np.any(SF.mask):
        SF.crop_masked_border(inplace=True)
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
    return Points([pos], unit_x=SF.unit_x, unit_y=SF.unit_y)


#def _gaussian_fit(SF):
#    """
#    Only for use in 'get_cp_crit'.
#    """
#    # gaussian fitting
#    def gaussian(height, center_x, center_y, width_x, width_y):
#        """
#        Returns a gaussian function with the given parameters
#        """
#        width_x = float(width_x)
#        width_y = float(width_y)
#        return lambda x, y: height*np.exp(
#            -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)
#
#    def moments(data):
#        """
#        Returns (height, x, y, width_x, width_y)
#        the gaussian parameters of a 2D distribution by calculating its
#        moments
#        """
#        total = data.sum()
#        X, Y = np.indices(data.shape)
#        x = (X*data).sum()/total
#        y = (Y*data).sum()/total
#        col = data[:, int(y)]
#        width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
#        row = data[int(x), :]
#        width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
#        height = data.max()
#        return height, x, y, width_x, width_y
#
#    def fitgaussian(data):
#        """
#        Returns (height, x, y, width_x, width_y)
#        the gaussian parameters of a 2D distribution found by a fit
#        """
#        params = moments(data)
#        errorfunction = lambda p: np.ravel(gaussian(*p)
#                                           (*np.indices(data.shape)) -
#                                           data)
#        p, success = optimize.leastsq(errorfunction, params)
#        return p
#    axe_x, axe_y = SF.axe_x, SF.axe_y
#    values = SF.values
#    params = fitgaussian(values)
#    delta_x = axe_x[1] - axe_x[0]
#    delta_y = axe_y[1] - axe_y[0]
#    x = SF.axe_x[0] + delta_x*params[1]
#    y = SF.axe_y[0] + delta_y*params[2]
#    return Points([(x, y)])


def _get_saddle_orientations(vectorfield, pt):
    """
    Return the orientation of a saddle point.
    """
    # smooth to avoid wrong orientations
    # vectorfield = vectorfield.smooth(tos='gaussian', size=1, inplace=False)
    # get data
    axe_x, axe_y = vectorfield.axe_x, vectorfield.axe_y
    dx = vectorfield.axe_x[1] - vectorfield.axe_x[0]
    dy = vectorfield.axe_y[1] - vectorfield.axe_y[0]
    Vx = vectorfield.comp_x
    Vy = vectorfield.comp_y
    # get surrounding points
    inds_x = vectorfield.get_indice_on_axe(1, pt[0], kind='bounds')
    inds_y = vectorfield.get_indice_on_axe(2, pt[1], kind='bounds')
    inds_x_2 = np.arange(-1, 3) + inds_x[0]
    inds_y_2 = np.arange(-1, 3) + inds_y[0]
    # check if surrounding gradients are available
    if (np.any(inds_x_2 > len(axe_x) - 1) or np.any(inds_x_2 < 0) or
            np.any(inds_y_2 > len(axe_y) - 1) or np.any(inds_y_2 < 0)):
        return [[0, 0], [0, 0]]
    # get surounding gradients
    inds_X_2, inds_Y_2 = np.meshgrid(inds_x_2, inds_y_2)
    local_Vx = Vx[inds_X_2, inds_Y_2]
    local_Vy = Vy[inds_X_2, inds_Y_2]
    # get jacobian eignevalues
    jac = _get_jacobian_matrix(local_Vx, local_Vy, dx, dy, pt)
    eigvals, eigvects = np.linalg.eig(jac)
    if eigvals[1] < eigvals[0]:
        orient1 = np.real(eigvects[:, 0])
        orient2 = np.real(eigvects[:, 1])
    else:
        orient1 = np.real(eigvects[:, 1])
        orient2 = np.real(eigvects[:, 0])
    return np.real(orient1), np.real(orient2)


def _get_jacobian_matrix(Vx, Vy, dx=1., dy=1, pt=None):
    """
    Return the jacobian matrix at the center of 4 points.
    """
    # check params
    if not isinstance(Vx, ARRAYTYPES):
        raise TypeError()
    if not isinstance(Vy, ARRAYTYPES):
        raise TypeError()
    Vx = np.array(Vx)
    Vy = np.array(Vy)
    if not Vx.shape == Vy.shape:
        raise ValueError()
    # compute gradients
    Vx_dx, Vx_dy = np.gradient(Vx.transpose(), dx, dy)
    Vy_dx, Vy_dy = np.gradient(Vy.transpose(), dx, dy)
    axe_x = np.arange(0, Vx.shape[0]*dx, dx)
    axe_y = np.arange(0, Vx.shape[1]*dy, dy)
    if pt is None:
        pt = [np.mean(axe_x), np.mean(axe_y)]
    # get interpolated gradient at the point
    Vx_dx2 = np.mean(Vx_dx)
    Vx_dy2 = np.mean(Vx_dy)
    Vy_dx2 = np.mean(Vy_dx)
    Vy_dy2 = np.mean(Vy_dy)
    # # More robust way of doing it ??? ###
    # if Vx.shape[0] == 2 or Vx.shape[1] == 2:
    #     k = 1
    # else:
    #     k = 2
    # Vx_dx2 = RectBivariateSpline(axe_x, axe_y, Vx_dx, kx=k, ky=k, s=0)
    # Vx_dx2 = Vx_dx2(*pt)[0][0]
    # Vx_dy2 = RectBivariateSpline(axe_x, axe_y, Vx_dy, kx=k, ky=k, s=0)
    # Vx_dy2 = Vx_dy2(*pt)[0][0]
    # Vy_dx2 = RectBivariateSpline(axe_x, axe_y, Vy_dx, kx=k, ky=k, s=0)
    # Vy_dx2 = Vy_dx2(*pt)[0][0]
    # Vy_dy2 = RectBivariateSpline(axe_x, axe_y, Vy_dy, kx=k, ky=k, s=0)
    # Vy_dy2 = Vy_dy2(*pt)[0][0]
    ###
    # get jacobian eignevalues
    jac = np.array([[Vx_dx2, Vx_dy2], [Vy_dx2, Vy_dy2]], subok=True)
    return jac
