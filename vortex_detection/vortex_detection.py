# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:07:07 2014

@author: muahah

For performance
"""

import pdb
from ..core import Points, OrientedPoints, Profile, ScalarField, VectorField,\
    make_unit, ARRAYTYPES, NUMBERTYPES, STRINGTYPES, TemporalScalarFields,\
    TemporalVectorFields
from ..field_treatment import get_streamlines, get_gradients
from ..Tools import ProgressCounter
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
import sympy
import time as modtime


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
        signx = (vx_sup_0[0:-1, 0:-1] + vx_sup_0[1::, 0:-1]
                 + vx_sup_0[0:-1, 1::] + vx_sup_0[1::, 1::])
        signy = (vy_sup_0[0:-1, 0:-1] + vy_sup_0[1::, 0:-1]
                 + vy_sup_0[0:-1, 1::] + vy_sup_0[1::, 1::])
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

    def get_cp_position(self):
        """
        Return critical points positions and their associated PBI using
        bilinear interpolation.
        (PBI : Poincarre_Bendixson indice)
        Positions are returned in axis unities (axe_x and axe_y).

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
        real_dx = axe_x[1] - axe_x[0]
        real_dy = axe_y[1] - axe_y[0]

        # import linear interpolation levelset solutions (computed with sympy)
        from levelset_data import get_sol
        sol = get_sol()
        # for each position
        new_positions = np.zeros(positions.shape)
        for i, pos in enumerate(positions):
            tmp_x = pos[0]
            tmp_y = pos[1]
            # get the cell indices
            ind_x = np.where(tmp_x < axe_x)[0][0] - 1
            ind_y = np.where(tmp_y < axe_y)[0][0] - 1
            # get data
            Vx_bl = self.vx[ind_y:ind_y + 2, ind_x:ind_x + 2]
            Vy_bl = self.vy[ind_y:ind_y + 2, ind_x:ind_x + 2]
            mask = self.mask[ind_y:ind_y + 2, ind_x:ind_x + 2]
            if np.any(mask):
                new_positions[i] = pos
                continue
            # solve to get the zero velocity point
            tmp_dic = {'Vx_1': Vx_bl[0, 0], 'Vx_2': Vx_bl[0, 1],
                       'Vx_3': Vx_bl[1, 0], 'Vx_4': Vx_bl[1, 1],
                       'Vy_1': Vy_bl[0, 0], 'Vy_2': Vy_bl[0, 1],
                       'Vy_3': Vy_bl[1, 0], 'Vy_4': Vy_bl[1, 1],
                       'dx': real_dx, 'dy': real_dy,
                       'sqrt': lambda x: x**.5}
            x_sols = [eval(sol[j][0], {"__builtins__": {}}, tmp_dic)
                      for j in np.arange(len(sol))]
            y_sols = [eval(sol[j][1], {"__builtins__": {}}, tmp_dic)
                      for j in np.arange(len(sol))]
            # delete the points outside the cell
            tmp_sol = []
            for j in np.arange(len(x_sols)):
                if np.iscomplex(x_sols[j]):
                    x_sols[j] = x_sols[j].as_real_imag()[0]
                if np.iscomplex(y_sols[j]):
                    y_sols[j] = y_sols[j].as_real_imag()[0]
                if (x_sols[j] < 0 or x_sols[j] > real_dx
                        or y_sols[j] < 0 or y_sols[j] > real_dy):
                    continue
                tmp_sol.append([x_sols[j], y_sols[j]])
            # if no more points
            if len(tmp_sol) == 0:
                new_positions[i] = np.array([tmp_x, tmp_y])
            # if multiple points
            elif len(tmp_sol) != 1:
                tmp_sol = np.array(tmp_sol)
                if not np.all(tmp_sol[0] == tmp_sol):
                    raise Exception()
                tmp_sol = tmp_sol[0]
                new_positions[i] = np.array([tmp_sol[0] + axe_x[ind_x],
                                             tmp_sol[1] + axe_y[ind_y]])
            # if one point (ok case)
            else:
                tmp_sol = tmp_sol[0]
                new_positions[i] = np.array([tmp_sol[0] + axe_x[ind_x],
                                             tmp_sol[1] + axe_y[ind_y]])
        # returning
        return new_positions, cp_types

    def get_pbi(self, direction):
        """
        Return the PBI along the given axe.
        """
        theta = self.theta.copy()
        if direction == 2:
            theta = np.rot90(theta, 3)
        pbi = np.zeros(theta.shape[1])
        for i in np.arange(1, theta.shape[1]):
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
    ### Operators ###
    def __init__(self,  unit_time='s'):
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
        self.colors = ['r', 'b', 'y', 'm', 'g']
        self.cp_types = ['foc', 'foc_c', 'node_i', 'node_o', 'sadd']
        self.current_epsilon = None

    def __add__(self, obj):
        if isinstance(obj, CritPoints):
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

    ### Attributes ###
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
        for kind in self.iter:
            for i in np.arange(len(kind)):
                kind[i].unit_x = new_unit_x
        for kind in self.iter_traj:
            if kind is None:
                continue
            for i in np.arange(len(kind)):
                kind[i].unit_x = new_unit_x

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
        for kind in self.iter:
            for i in np.arange(len(kind)):
                kind[i].unit_y = new_unit_y
        for kind in self.iter_traj:
            if kind is None:
                continue
            for i in np.arange(len(kind)):
                kind[i].unit_y = new_unit_y

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
        for kind in self.iter:
            for i in np.arange(len(kind)):
                kind[i].unit_v = new_unit_time
        for kind in self.iter_traj:
            if kind is None:
                continue
            for i in np.arange(len(kind)):
                kind[i].unit_v = new_unit_time

    ### Properties ###
    @property
    def iter(self):
        return [self.foc, self.foc_c, self.node_i, self.node_o, self.sadd]

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
        # check parameters
        if epsilon is None:
            epsilon = np.inf
        if not isinstance(epsilon, NUMBERTYPES):
            raise ValueError()
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

    def get_points_density(self, kind, bw_method=None, resolution=100,
                           output_format=None):
        """
        Return the presence density map for the given point type.

        Parameters:
        -----------
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
            'normalized' (default) : give position probability
                                     (integral egal 1).
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
                 color=self.colors,
                 label=self.cp_types)
        plt.legend()
        plt.xlabel('Trajectories size')
        plt.ylabel('Number of trajectories')
        # returning
        return fig

    ### Modifiers ###
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
        self.current_epsilon = None

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
        self.current_epsilon = None

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

    def trim(self, intervx=None, intervy=None, inplace=False):
        """
        Trim the point field.
        """
        # default values
        if intervx is None:
            intervx = [-np.inf, np.inf]
        if intervy is None:
            intervy = [-np.inf, np.inf]
        # inplace
        if inplace:
            tmp_cp = self
        else:
            tmp_cp = self.copy()
        # trajectories are obsolete
        tmp_cp.current_epsilon = None
        # loop on points
        for kind in tmp_cp.iter:
            for time in kind:
                for i in np.arange(len(time.xy) - 1, -1, -1):
                    xy = time.xy[i]
                    if xy[0] < intervx[0] or xy[0] > intervx[1]\
                            or xy[1] < intervy[0] or xy[1] > intervy[1]:
                        time.remove(i)
        # returning
        if not inplace:
            return tmp_cp

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
        for type_traj in self.iter_traj:
            for i in np.arange(len(type_traj) - 1, -1, -1):
                traj = type_traj[i]
                if len(traj.xy) < min_nmb_in_traj:
                    del type_traj[i]
        # extend deleting to points
        self._traj_to_pts()

    def smooth_traj(self, tos='uniform', size=None):
        """
        Smooth the CP trjaectories.

        Parameters :
        ------------
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
    def _get_cp_time_evolution(points, times=None, epsilon=None):
        """
        Compute the temporal evolution of each critical point from a set of
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
                try:
                    i = int(i)
                    j = int(j)
                except ValueError:
                    raise TypeError("'i' and 'j' must be integers, not {}"
                                    " and {}"
                                    .format(type(i), type(j)))
                self.points[i][j] = None

            def get_points_at_time(self, time):
                """
                Return all the points for a given time.
                """
                try:
                    time = int(time)
                except ValueError:
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
        # sort by length
        points_f = np.asarray(points_f)
        lens = [len(pts) for pts in points_f]
        ind_sort = np.argsort(lens)
        points_f = points_f[ind_sort [::-1]]
        return list(points_f)

    ### Displayers ###
    def display(self, indice=None, time=None, field=None, cpkw={}, lnkw={}):
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
        if field is not None:
            if not isinstance(field, VectorField):
                raise TypeError()
        # Set the color
        if 'color' in cpkw.keys():
            colors = [cpkw.pop('color')]*len(self.colors)
        else:
            colors = self.colors
        # display the critical lines
        if (field is not None and len(self.sadd[indice].xy) != 0
                and isinstance(self.sadd[indice], OrientedPoints)):
            if 'color' not in lnkw.keys():
                lnkw['color'] = colors[4]
            streams = self.sadd[indice]\
                .get_streamlines_from_orientations(field,
                reverse_direction=[True, False], interp='cubic')
            for stream in streams:
                stream._display(kind='plot', **lnkw)
        # loop on the points types
        for i, pt in enumerate(self.iter):
            if pt[indice] is None:
                continue
            pt[indice].display(kind='plot', marker='o', color=colors[i],
                               linestyle='none', **cpkw)

    def display_traj(self, data='default', reverse=None, filt=None, **kw):
        """
        Display the stored trajectories.

        Parameters
        ----------
        data : string
            If 'default', trajectories are plotted in a 2-dimensional plane.
            If 'x', x position of cp are plotted against time.
            If 'y', y position of cp are plotted against time.
        reverse : boolean, optional
            If 'True', reverse the axis.
        filt : array of boolean
            Filter on CP types.
        kw : dict, optional
            Arguments passed to plot.
        """
        # check if some trajectories are computed
        if self.current_epsilon is None:
            raise StandardError("you must compute trajectories before "
                                "displaying them")
        if reverse is None:
            if data == 'x':
                reverse = False
            elif data == 'y':
                reverse = True
        if filt is None:
            filt = np.ones((5,), dtype=bool)
        if not isinstance(filt, ARRAYTYPES):
            raise TypeError()
        filt = np.array(filt)
        if not filt.dtype == bool:
            raise TypeError()
        # display
        if 'color' in kw.keys():
            colors = [kw.pop('color')]*len(self.colors)
        else:
            colors = self.colors
        if 'marker' not in kw.keys():
            kw['marker'] = 'o'
        if 'linestyle' not in kw.keys():
            kw['linestyle'] = '-'
        if data == 'default':
            for i, trajs in enumerate(self.iter_traj):
                color = colors[i]
                if trajs is None or not filt[i]:
                    continue
                for traj in trajs:
                    traj.display(color=color, kind='plot', reverse=reverse,
                                 **kw)
            if reverse:
                plt.ylabel('x {}'.format(self.unit_x.strUnit()))
                plt.xlabel('y {}'.format(self.unit_y.strUnit()))
            else:
                plt.xlabel('x {}'.format(self.unit_x.strUnit()))
                plt.ylabel('y {}'.format(self.unit_y.strUnit()))
        elif data == 'x':
            for i, trajs in enumerate(self.iter_traj):
                color = colors[i]
                if trajs is None or not filt[i]:
                    continue
                for traj in trajs:
                    if reverse:
                        plt.plot(traj.v[:], traj.xy[:, 0], color=color,
                                 **kw)
                    else:
                        plt.plot(traj.xy[:, 0], traj.v[:], color=color,
                                 **kw)
            if reverse:
                plt.ylabel('x {}'.format(self.unit_x.strUnit()))
                plt.xlabel('time {}'.format(self.unit_time.strUnit()))
            else:
                plt.xlabel('x {}'.format(self.unit_x.strUnit()))
                plt.ylabel('time {}'.format(self.unit_time.strUnit()))
        elif data == 'y':
            for i, trajs in enumerate(self.iter_traj):
                color = colors[i]
                if trajs is None or not filt[i]:
                    continue
                for traj in trajs:
                    if reverse:
                        plt.plot(traj.v[:], traj.xy[:, 1], color=color,
                                 **kw)
                    else:
                        plt.plot(traj.xy[:, 1], traj.v[:], color=color,
                                 **kw)
            if reverse:
                plt.xlabel('time {}'.format(self.unit_time.strUnit()))
                plt.ylabel('y {}'.format(self.unit_y.strUnit()))
            else:
                plt.ylabel('time {}'.format(self.unit_time.strUnit()))
                plt.xlabel('y {}'.format(self.unit_y.strUnit()))
        else:
            raise StandardError()

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
                new_coord = ((xy[ind_1]*weight[ind_1]
                              + xy[ind_2]*weight[ind_2])
                             / (weight[ind_1] + weight[ind_2]))
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
    gamma2 = get_gamma(VF, radius=gamma2_radius, ind=False, kind='gamma2')
    gamma2 = gamma2.smooth('gaussian').values
    vort = np.abs(gamma2) > 2/np.pi
    ind_x = VF.get_indice_on_axe(1, vort_center[0], kind='nearest')
    ind_y = VF.get_indice_on_axe(2, vort_center[1], kind='nearest')
    dx = VF.axe_x[1] - VF.axe_x[0]
    dy = VF.axe_y[1] - VF.axe_y[0]
    # find vortex zones and label them

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
                                     output_center=False, verbose=False):
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
    verbose : boolean
        .

    Returns :
    ---------
    radius : Profile object
        Average radius of the vortex. If no vortex is found, 0 is returned.
    center : Points object
        If 'output_center' is 'True', contain the newly computed vortex center.
    """
    radii = np.empty((len(traj.xy),))
    if verbose:
        pg = ProgressCounter("Begin vortex radii detection",
                             "Done", len(traj.xy), 'fields', perc_interv=1)
    # computing with vortex center
    if output_center:
        centers = Points(unit_x=TVFS.unit_x, unit_y=TVFS.unit_y,
                         unit_v=TVFS.unit_times)

        for i, pt in enumerate(traj):
            if verbose:
                pg.print_progress()
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
        for i, _ in enumerate(traj):
            if verbose:
                pg.print_progress()
            # getting time and associated velocity field
            time = traj.v[i]
            field = TVFS.fields[TVFS.times == time][0]
            # getting radius
            radii[i] = get_vortex_radius(field, traj.xy[i],
                                         gamma2_radius=gamma2_radius,
                                         output_center=False)
    # returning
    mask = radii == 0.
    radii_prof = Profile(traj.v, radii, mask=mask, unit_x=TVFS.unit_times,
                         unit_y=TVFS.unit_x)
    if output_center:
        return radii_prof, centers
    else:
        return radii_prof


def get_vortex_intensity(VF, vort_center, crit=None, output_unit=False,
                         use_gamma2=True):
    """
    Return the intensity of the given vortex.

    Parameters:
    -----------
    VF : vectorfield object
        Velocity field on which compute gamma2.
    vort_center : 2x1 array
        Approximate position of the vortex center.
    crit : function
        Function to inegrate on the vortex zone. should take a VectorField as
        argument and return a ScalarField. Default is 'get_residual_vorticity'.
    use_gamma2 : boolean, optional
        If 'True' (default), gamma2 is used to get the vortex area, and the
        criterion is integrated on this area. If 'False', returned intensity is
        directly the criterion intensity at the wanted point.
    output_unit ; boolean, optional
        If 'True', return the associated unit.

    Returns :
    ---------
    intens : number
        Intensity of the vortex. If no vortex is found, 0 is returned.
    """
    if crit is None:
        crit = get_residual_vorticity
    # getting data
        if use_gamma2:
            gamma2 = get_gamma(VF, ind=False, kind='gamma2', raw=True)
    ind_x = VF.get_indice_on_axe(1, vort_center[0], kind='nearest')
    ind_y = VF.get_indice_on_axe(2, vort_center[1], kind='nearest')
    dx = VF.axe_x[1] - VF.axe_x[0]
    dy = VF.axe_y[1] - VF.axe_y[0]
    # getting criterion field
    tmp_cf = crit(VF)
    if use_gamma2:
        # get vortex zone
        vort = np.abs(gamma2) > 2/np.pi
        vort, nmb_vort = msr.label(vort)
        # get wanted zone label
        lab = vort[ind_x, ind_y]
        # if we are outside a zone
        if lab == 0:
            tmp_int = 0
        else:
            tmp_int = np.sum(np.abs(tmp_cf.values[vort == lab]))*dx*dy
    else:
        tmp_int = tmp_cf.get_value(ind_x, ind_y, ind=True)
    if output_unit and use_gamma2:
        unit_int = tmp_cf.unit_values*tmp_cf.unit_x*tmp_cf.unit_y
        scale = unit_int.asNumber()
        unit_int /= scale
        tmp_int *= scale
        return tmp_int, unit_int
    elif output_unit:
        return tmp_int, tmp_cf.unit_values
    else:
        return tmp_int


def get_vortex_intensity_time_evolution(TVFS, traj, crit=None,
                                        use_gamma2=True, verbose=False):
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
    crit : function
        Function to inegrate on the vortex zone. should take a VectorField as
        argument and return a ScalarField. Default is 'get_residual_vorticity'.
    use_gamma2 : boolean, optional
        If 'True' (default), gamma2 is used to get the vortex area, and the
        criterion is integrated on this area. If 'False', returned intensity is
        directly the criterion intensity at the wanted point.
    verbose : boolean
        .

    Returns :
    ---------
    intensity : Profile object
        Average intensity of the vortex. If no vortex is found, 0 is returned.
    """
    intens = np.empty((len(traj.xy),))
    if verbose:
        pg = ProgressCounter("Begin vortex intensity detection",
                             "Done", len(traj.xy), 'fields', perc_interv=1)
    # loop on traj times
    for i, _ in enumerate(traj):
        if verbose:
            pg.print_progress()
        # getting time and associated velocity field
        time = traj.v[i]
        field = TVFS.fields[TVFS.times == time][0]
        # getting the wanted point
        wanted_xy = traj.xy[i, :]
        tmp_int, unit_int = get_vortex_intensity(field, wanted_xy, crit=crit,
                                                 output_unit=True,
                                                 use_gamma2=use_gamma2)
        intens[i] = tmp_int
    # returning
    mask = intens == 0.
    radii_prof = Profile(traj.v, intens, mask=mask, unit_x=TVFS.unit_times,
                         unit_y=unit_int)
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
    ind_x = VF.get_indice_on_axe(1, vort_center[0], kind='nearest')
    ind_y = VF.get_indice_on_axe(2, vort_center[1], kind='nearest')
    dx = VF.axe_x[1] - VF.axe_x[0]
    dy = VF.axe_y[1] - VF.axe_y[0]
    import IMTreatment.field_treatment as imtft
    vort = get_vorticity(VF)
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
    if output_unit:
        return circ, unit_circ
    else:
        return circ


def get_critical_points(obj, time=0, unit_time='', window_size=4,
                        kind='pbi', mirroring=None, mirror_interp='linear',
                        smoothing_size=0, verbose=False):
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
        '[direction (1 or 2), position]*N'.
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

    Notes
    -----
    If the fields have masked values, saddle streamlines ar not computed.
    """
    # check parameters
    if not isinstance(time, NUMBERTYPES):
        raise TypeError()
    if not isinstance(unit_time, STRINGTYPES + (unum.Unum,)):
        raise TypeError()
    if not isinstance(window_size,  NUMBERTYPES):
        raise TypeError()
    window_size = int(window_size)
    if not isinstance(kind, STRINGTYPES):
        raise TypeError()
    if kind not in ['pbi', 'pbi_crit', 'pbi_cell', 'gam_vort']:
        raise ValueError()
    if mirroring is not None:
        if not isinstance(mirroring, ARRAYTYPES):
            raise TypeError()
        mirroring = np.array(mirroring)
        if mirroring.ndim != 2:
            raise ValueError()
        if mirroring.shape[1] != 2:
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
    # if obj is a vector field
    if isinstance(obj, VectorField):
        # mirroring if necessary
        if mirroring is not None:
            tmp_vf = obj.copy()
            for direction, position in mirroring:
                if kind == 'pbi_crit':
                    tmp_vf.mirroring(int(direction), position,
                                     inds_to_mirror=window_size*2,
                                     inplace=True, interp=mirror_interp,
                                     value=[0, 0])
                else:
                    tmp_vf.mirroring(int(direction), position,
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
                                    window_size=window_size)
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
                if direction == 1:
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
            res.trim(intervx=intervx, intervy=intervy, inplace=True)
    # if obj is vector fields
    elif isinstance(obj, TemporalVectorFields):
        res = CritPoints(unit_time=obj.unit_times)
        res.unit_x = obj.unit_x
        res.unit_y = obj.unit_y
        res.unit_time = obj.unit_times
        if verbose:
            print("\n+++ Begin CP detection on {:.0f} fields"
                  .format(len(obj.fields)))
            t0 = modtime.time()
            df = int(np.round(len(obj.fields)/20.))
            field_pad = len(str(len(obj.fields)))
        for i, field in enumerate(obj.fields):
            res += get_critical_points(field, time=obj.times[i],
                                       unit_time=obj.unit_times,
                                       window_size=window_size,
                                       kind=kind, mirroring=mirroring,
                                       smoothing_size=smoothing_size)
            if verbose and df != 0 and len(obj.fields) != 1:
                if i % df == 0 or i == len(obj.fields) - 1:
                    ti = modtime.time()
                    if i == 0:
                        tf = '---'
                    else:
                        dt = (ti - t0)/i
                        tf = t0 + dt*(len(obj.fields))
                        tf = _format_time(tf - t0)
                    ti = _format_time(ti - t0)
                    print("+++    {:>3.0f} %    {:{field_pad}d}/{} fields    {}/{}"
                          .format(np.round(i*1./len(obj.fields)*100),
                                  i, len(obj.fields), ti, tf, field_pad=field_pad))
    else:
        raise TypeError()
    return res


def _get_cp_pbi_on_VF(vectorfield, time=0, unit_time=make_unit(""),
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
    if isinstance(unit_time, STRINGTYPES):
        unit_time = make_unit(unit_time)
    if np.any(vectorfield.mask):
        sadd_ori = False
    else:
        sadd_ori = True
    # using VF methods to get cp position and types
    field = velocityfield_to_vf(vectorfield, time)
    pos, cp_types = field.get_cp_position()
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
    ### Getting pbi cp position and fields around ###
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
        if (len(axe_x) <= window_size + 1 or len(axe_y) <= window_size + 1
                or np.any(tmp_vf.mask)):
            pass
        else:
            tmp_vf.PBI = int(cp_types[i] != 0)
            tmp_vf.assoc_ind = i
            VF_tupl.append(tmp_vf)
    ### Sorting by critical points type ###
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
    ### Computing focus positions (rotatives and contrarotatives) ###
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
    ### Computing nodes points positions (in or out) ###
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
    ### Computing saddle points positions and direction ###
    if len(VF_saddle) != 0:
        for VF in VF_saddle:
            tmp_iot = get_iota(VF, radius=radius, ind=True)
            pts = _min_detection(np.max(tmp_iot.values) - tmp_iot)
            if pts is not None:
                if pts.xy[0][0] is not None and len(pts) == 1:
                        cp_positions[VF.assoc_ind] = pts.xy[0]
                        cp_types[VF.assoc_ind] = 0
    ### creating the CritPoints object for returning
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
                tmp_pos = pos[i]
                ori = np.array(_get_saddle_orientations(vectorfield, tmp_pos))
                sadd.add(tmp_pos, orientations=ori)
            else:
                sadd.add(pos[i])
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
    gamma.crop_masked_border(hard=True)
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
    return Points([pos], unit_x=SF.unit_x, unit_y=SF.unit_y)


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
    jac = _get_jacobian_matrix(local_Vx, local_Vy, dx, dy)
    eigvals, eigvects = np.linalg.eig(jac)
    if eigvals[0] < eigvals[1]:
        orient1 = eigvects[:, 0]
        orient2 = eigvects[:, 1]
    else:
        orient1 = eigvects[:, 1]
        orient2 = eigvects[:, 0]
    return np.real(orient1), np.real(orient2)


def _get_jacobian_matrix(Vx, Vy, dx=1., dy=1.):
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
    if Vx.shape[0] == 2 or Vx.shape[1] == 2:
        k = 1
    else:
        k = 2
    # compute gradients
    Vx_dx, Vx_dy = np.gradient(Vx.transpose(), dx, dy)
    Vy_dx, Vy_dy = np.gradient(Vy.transpose(), dx, dy)
    axe_x = np.arange(0, Vx.shape[0]*dx, dx)
    axe_y = np.arange(0, Vx.shape[1]*dy, dy)
    # get interpolated gradient at the point
    Vx_dx2 = RectBivariateSpline(axe_x, axe_y, Vx_dx, kx=k, ky=k, s=0)
    Vx_dx2 = Vx_dx2(np.mean(axe_x), np.mean(axe_y))[0][0]
    Vx_dy2 = RectBivariateSpline(axe_x, axe_y, Vx_dy, kx=k, ky=k, s=0)
    Vx_dy2 = Vx_dy2(np.mean(axe_x), np.mean(axe_y))[0][0]
    Vy_dx2 = RectBivariateSpline(axe_x, axe_y, Vy_dx, kx=k, ky=k, s=0)
    Vy_dx2 = Vy_dx2(np.mean(axe_x), np.mean(axe_y))[0][0]
    Vy_dy2 = RectBivariateSpline(axe_x, axe_y, Vy_dy, kx=k, ky=k, s=0)
    Vy_dy2 = Vy_dy2(np.mean(axe_x), np.mean(axe_y))[0][0]
    # get jacobian eignevalues
    jac = np.array([[Vx_dx2, Vx_dy2], [Vy_dx2, Vy_dy2]], subok=True)
    return jac


def _format_time(second):
    second = int(second)
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    j, h = divmod(h, 24)
    repr_time = '{:d}s'.format(s)
    if m != 0:
        repr_time = '{:d}mn'.format(m) + repr_time
    if h != 0:
        repr_time = '{:d}h'.format(h) + repr_time
    if j != 0:
        repr_time = '{:d}j'.format(m) + repr_time
    return repr_time


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
    return float(interp(wall_position))


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
    if direction not in [0, 1]:
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
    if local_treatment not in ['none', 'galilean_inv', 'local']:
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
              raw=False, dev_pass=False):
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
        velocity angles deviation is strong (faster if there is few points).
        Work only with 'gamma1'
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
    if kind not in ['gamma1', 'gamma2', 'gamma1b', 'gamma2b']:
        raise ValueError("Unkown value for kind")
    if mask is None:
        mask = np.zeros(vectorfield.shape)
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'zone' must be an array of boolean")
    else:
        mask = np.array(mask)
    if kind in ['gamma2', 'gamma2b']:
        dev_pass = False
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


def get_NL_residual_vorticity(vectorfield, radius=None, ind=False, mask=None,
                              raw=False):
    """
    Return the residual vorticity computed with non-local gradients.

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
    axe_x, axe_y = vectorfield.axe_x, vectorfield.axe_y
    if mask is None:
        mask = np.zeros(vectorfield.shape)
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'zone' must be an array of boolean")
    else:
        mask = np.array(mask)
    # getting data and masks
    Vx = vectorfield.comp_x
    Vy = vectorfield.comp_y
    mask, nmbpts, mask_dev, mask_border, mask_surr, motif =\
        _non_local_criterion_precomputation(vectorfield, mask, radius, ind,
                                            dev_pass=False)
    ### Loop on points to get non-local gradients ###
    Exx = np.zeros(vectorfield.shape, dtype=float)
    Exy = np.zeros(vectorfield.shape, dtype=float)
    Eyx = np.zeros(vectorfield.shape, dtype=float)
    Eyy = np.zeros(vectorfield.shape, dtype=float)
    for inds, pos, _ in vectorfield:
        ind_x = inds[0]
        ind_y = inds[1]
        # stop if masked or on border or with a masked surrouinding point
        if mask[ind_x, ind_y] or mask_surr[ind_x, ind_y]\
                or mask_border[ind_x, ind_y] or mask_dev[ind_x, ind_y]:
            continue
        # getting neighbour points
        indsaround = motif + inds
        # non-local gradients computation by linear fitting
        ind_xs = indsaround[:, 0]
        ind_ys = indsaround[:, 1]
        Vxs = Vx[ind_xs, ind_ys]
        Vys = Vy[ind_xs, ind_ys]
        Exx[ind_x, ind_y], a_xx = np.polyfit(axe_x[ind_xs], Vxs, 1)
        Exy[ind_x, ind_y], a_xy = np.polyfit(axe_y[ind_ys], Vxs, 1)
        Eyx[ind_x, ind_y], _ = np.polyfit(axe_x[ind_xs], Vys, 1)
        Eyy[ind_x, ind_y], _ = np.polyfit(axe_y[ind_ys], Vys, 1)
    # getting principal rate of strain (s)
    s = np.sqrt(4*Exx**2 + (Exy + Eyx)**2)/2.
    # getting the vorticity-tensor component
    omega = (Eyx - Exy)/2.
    omega_abs = np.abs(omega)
    sign_omega = np.sign(omega)
    sign_omega[sign_omega == 0] = 1
    filt = s < omega_abs
    # getting the residual vorticity
    res_vort = np.zeros(vectorfield.shape)
    res_vort[filt] = sign_omega[filt]*(omega_abs[filt] - s[filt])
    ### Applying masks ###
    mask = np.logical_or(mask, mask_border)
    mask = np.logical_or(mask, mask_surr)
    ### Creating gamma ScalarField ###
    if raw:
        return np.ma.masked_array(res_vort, mask)
    else:
        gamma_sf = ScalarField()
        unit_values = vectorfield.unit_values/vectorfield.unit_x
        res_vort *= unit_values.asNumber()
        unit_values /= unit_values.asNumber()
        unit_x, unit_y = vectorfield.unit_x, vectorfield.unit_y
        gamma_sf.import_from_arrays(axe_x, axe_y, res_vort, mask,
                                    unit_x=unit_x, unit_y=unit_y,
                                    unit_values=unit_values)
        return gamma_sf


#def get_gamma_res(vectorfield, radius=None, ind=False, kind='gamma1', mask=None,
#                  raw=False, dev_pass=False):
#    """
#    Return the gamma scalar field. Gamma criterion is used in
#    vortex analysis.
#    The fonction recognize if the field is ortogonal, and use an
#    apropriate algorithm.
#
#    Parameters
#    ----------
#    vectorfield : VectorField object
#        .
#    radius : number, optionnal
#        The radius used to choose the zone where to compute
#        gamma for each point. If not mentionned, a value is choosen in
#        ordre to have about 8 points in the circle. It allow to get good
#        result, without big computation cost.
#    ind : boolean
#        If 'True', radius is expressed on number of vectors.
#        If 'False' (default), radius is expressed on axis unit.
#    kind : string
#        If 'gamma1' (default), compute gamma1 criterion.
#        If 'gamma2', compute gamma2 criterion (with relative velocities)
#    mask : array of boolean, optionnal
#        Has to be an array of the same size of the vector field object,
#        gamma will be compute only where mask is 'False'.
#    raw : boolean, optional
#        If 'False' (default), a ScalarField is returned,
#        if 'True', an array is returned.
#    dev_pass : boolean, optional
#        If 'True', the algorithm compute gamma criterion only where the
#        velocity angles deviation is strong (faster if there is few points).
#        Work only with 'gamma1'
#    """
#    ### Checking parameters coherence ###
#    if not isinstance(vectorfield, VectorField):
#        raise TypeError("'vectorfield' must be a VectorField object")
#    if radius is None:
#        radius = 1.9
#        ind = True
#    if not isinstance(radius, NUMBERTYPES):
#        raise TypeError("'radius' must be a number")
#    if not isinstance(ind, bool):
#        raise TypeError("'ind' must be a boolean")
#    axe_x, axe_y = vectorfield.axe_x, vectorfield.axe_y
#    if not isinstance(kind, STRINGTYPES):
#        raise TypeError("'kind' must be a string")
#    if kind not in ['gamma1', 'gamma2']:
#        raise ValueError("Unkown value for kind")
#    if mask is None:
#        mask = np.zeros(vectorfield.shape)
#    elif not isinstance(mask, ARRAYTYPES):
#        raise TypeError("'zone' must be an array of boolean")
#    else:
#        mask = np.array(mask)
#    if kind in ['gamma2', 'gamma2b']:
#        dev_pass = False
#    # getting data and masks
#    Vx = vectorfield.comp_x
#    Vy = vectorfield.comp_y
#    norm_v = vectorfield.magnitude
#    mask, nmbpts, mask_dev, mask_border, mask_surr, motif =\
#        _non_local_criterion_precomputation(vectorfield, mask, radius, ind,
#                                            dev_pass)
#    # getting the vectors between center and neighbouring
#    deltax = axe_x[1] - axe_x[0]
#    deltay = axe_y[1] - axe_y[0]
#    vector_a_x = np.zeros(motif.shape[0])
#    vector_a_y = np.zeros(motif.shape[0])
#    theta = []
#    for i, indaround in enumerate(motif):
#        vector_a_x[i] = indaround[0]*deltax
#        vector_a_y[i] = indaround[1]*deltay
#        theta.append(np.arctan(vector_a_y[i]/vector_a_x[i]))
#    theta = np.array(theta) % np.pi
#    norm_vect_a = (vector_a_x**2 + vector_a_y**2)**(.5)
#    ### Loop on points ###
#    gammas = np.zeros(vectorfield.shape)
#    for inds, pos, _ in vectorfield:
#        ind_x = inds[0]
#        ind_y = inds[1]
#        # stop if masked or on border or with a masked surrouinding point
#        if mask[ind_x, ind_y] or mask_surr[ind_x, ind_y]\
#                or mask_border[ind_x, ind_y] or mask_dev[ind_x, ind_y]:
#            continue
#        # getting neighbour points
#        indsaround = motif + inds
#        # If necessary, compute mean velocity on points (gamma2)
#        v_mean = [Vx[ind_x, ind_y], Vy[ind_x, ind_y]]
#        v_mean[0] = np.mean(Vx[indsaround[:, 0], indsaround[:, 1]])
#        v_mean[1] = np.mean(Vy[indsaround[:, 0], indsaround[:, 1]])
#
#        ### Loop on neighbouring points ###
#        loc_gammas = []
#        loc_weights = []
#        for i, indaround in enumerate(indsaround):
#            inda_x = indaround[0]
#            inda_y = indaround[1]
#            # getting vectors for scalar product
#            if kind in ['gamma1']:
#                vector_b_x = Vx[inda_x, inda_y]
#                vector_b_y = Vy[inda_x, inda_y]
#                denom = norm_v[inda_x, inda_y]*norm_vect_a[i]
#                if denom != 0:
#                    gamma += (vector_a_x[i]*vector_b_y
#                              - vector_a_y[i]*vector_b_x)/denom
#            elif kind in ['gamma2']:
#                vector_b_x = Vx[inda_x, inda_y] - v_mean[0]
#                vector_b_y = Vy[inda_x, inda_y] - v_mean[1]
#                denom = (vector_b_x**2 + vector_b_y**2)**.5*norm_vect_a[i]
#                if denom != 0:
#                    loc_weights.append((vector_b_x**2 + vector_b_y**2)**.5)
#                    loc_gammas.append((vector_a_x[i]*vector_b_y
#                                      - vector_a_y[i]*vector_b_x)/denom)
#
#        loc_weights = np.array(loc_weights)
#        loc_gammas = np.array(loc_gammas)
#        theta = np.array(theta)
#        weights = []
#        gams = []
#        for thet in theta:
#            weights.append(np.mean(loc_weights[theta == thet]))
#            gams.append(np.mean(loc_gammas[theta == thet]))
#        gamma = np.sum(loc_gammas)
#
#        if ind_x == 10 and ind_y == 50:
#            plt.figure()
#            plt.plot(weights, gams, 'o')
#            bug
#        # adapting with factors
#        if kind in ['gamma1', 'gamma2']:
#            gamma = gamma/nmbpts
#        # storing computed gamma value
#        gammas[ind_x, ind_y] = gamma
#    ### Applying masks ###
#    mask = np.logical_or(mask, mask_border)
#    mask = np.logical_or(mask, mask_surr)
#    ### Creating gamma ScalarField ###
#    if raw:
#        return np.ma.masked_array(gammas, mask)
#    else:
#        gamma_sf = ScalarField()
#        unit_x, unit_y = vectorfield.unit_x, vectorfield.unit_y
#        gamma_sf.import_from_arrays(axe_x, axe_y, gammas, mask,
#                                    unit_x=unit_x, unit_y=unit_y,
#                                    unit_values=make_unit(''))
#        return gamma_sf


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
    if kind not in ['kappa1', 'kappa2']:
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


def get_iota(vectorfield, mask=None, radius=None, ind=False, raw=False):
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
    radius : number, optionam
        If specified, the velocity field is smoothed with gaussian filter
        of the given radius before computing the vectors angles.
    ind : boolean, optional
        If 'True', radius is an indice number, if 'False', radius if in the
        field units (default).
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
    # smoothing if necessary and getting theta
    if radius is not None:
        if not ind:
            dx = axe_x[1] - axe_x[0]
            dy = axe_y[1] - axe_y[0]
            radius = radius/((dx + dy)/2.)
            ind = True
        tmp_vf = vectorfield.copy()
        tmp_vf.smooth(tos='gaussian', size=radius, inplace=True)
        theta = tmp_vf.theta
        mask = np.logical_or(mask, tmp_vf.mask)
    else:
        theta = vectorfield.theta
        mask = np.logical_or(mask, vectorfield.mask)
    ### calcul du gradients de theta
    # necesary steps to avoid big gradients by passing from 0 to 2*pi
    theta1 = theta.copy()
    theta2 = theta.copy()
    theta2[theta2 > np.pi] -= 2*np.pi
    theta1_x, theta1_y = np.gradient(theta1)
    theta2_x, theta2_y = np.gradient(theta2)
    filtx = np.abs(theta1_x) > np.abs(theta2_x)
    filty = np.abs(theta1_y) > np.abs(theta2_y)
    theta_x = theta1_x.copy()
    theta_x[filtx] = theta2_x[filtx]
    theta_y = theta1_y.copy()
    theta_y[filty] = theta2_y[filty]
    iota = 1/2.*np.sqrt(theta_x**2 + theta_y**2)
    # getting mask
    maskf = np.logical_or(vectorfield.mask, np.isnan(iota))
    # returning
    if raw:
        return np.ma.masked_array(iota, maskf)
    else:
        iota_sf = ScalarField()
        unit_x, unit_y = vectorfield.unit_x, vectorfield.unit_y
        iota_sf.import_from_arrays(axe_x, axe_y, iota, maskf,
                                   unit_x=unit_x, unit_y=unit_y,
                                   unit_values=make_unit(''))
        return iota_sf

def get_enstrophy(vectorfield, radius=None, ind=False, mask=None,
                  raw=False):
    """
    Return the enstriphy field.

    Parameters
    ----------
    vectorfield : VectorField object
        .
    radius : number, optionnal
        The radius used to choose the zone where to integrate
        enstrophy for each point. If not mentionned, a value is choosen in
        ordre to have about 8 points in the circle.
    ind : boolean
        If 'True', radius is expressed on number of vectors.
        If 'False' (default), radius is expressed on axis unit.
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
    axe_x, axe_y = vectorfield.axe_x, vectorfield.axe_y
    if mask is None:
        mask = np.zeros(vectorfield.shape)
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'mask' must be an array of boolean")
    else:
        mask = np.array(mask)
    # getting data and masks
    vort2 = get_vorticity(vectorfield, raw=False)**2
    unit_values = vort2.unit_values*vectorfield.unit_x*vectorfield.unit_y
    vort2 = vort2.values
    mask, nmbpts, mask_dev, mask_border, mask_surr, motif =\
        _non_local_criterion_precomputation(vectorfield, mask, radius, ind,
                                            dev_pass=False)
    dv = ((vectorfield.axe_x[1] - vectorfield.axe_x[0])
          * (vectorfield.axe_y[1] - vectorfield.axe_y[0]))
    ### Loop on points ###
    enstrophy = np.zeros(vectorfield.shape)
    for inds, pos, _ in vectorfield:
        ind_x = inds[0]
        ind_y = inds[1]
        # stop if masked or on border or with a masked surrouinding point
        if mask[ind_x, ind_y] or mask_surr[ind_x, ind_y]\
                or mask_border[ind_x, ind_y]:
            continue
        # getting neighbour points
        indsaround = motif + inds
        ### Loop on neighbouring points ###
        loc_enstrophy = 0.
        for i, indaround in enumerate(indsaround):
            loc_enstrophy += vort2[indaround[0], indaround[1]]
        # storing computed gamma value
        enstrophy[ind_x, ind_y] = loc_enstrophy*dv
    ### Applying masks ###
    mask = np.logical_or(mask, mask_border)
    mask = np.logical_or(mask, mask_surr)
    ### Creating gamma ScalarField ###
    if raw:
        return np.ma.masked_array(enstrophy, mask)
    else:
        enstrophy_sf = ScalarField()
        unit_x, unit_y = vectorfield.unit_x, vectorfield.unit_y
        scale = unit_values.asNumber()
        enstrophy *= scale
        unit_values = unit_values/scale
        enstrophy_sf.import_from_arrays(axe_x, axe_y, enstrophy, mask,
                                        unit_x=unit_x, unit_y=unit_y,
                                        unit_values=unit_values)
        return enstrophy_sf


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
        motif = np.delete(motif, len(motif)/2, axis=0)
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
    vf : VectorField or TemporalVectorfields
        Field(s) on which compute shear stress
    raw : boolean, optional
        If 'True', return an arrays,
        if 'False' (default), return a ScalarField object.

    Returns
    -------
    vort : ScalarField or TemporalScalarFields
        Vorticity field(s)
    """
    if isinstance(vf, VectorField):
        tmp_vf = vf.fill(inplace=False)
        axe_x, axe_y = tmp_vf.axe_x, tmp_vf.axe_y
        comp_x, comp_y = tmp_vf.comp_x, tmp_vf.comp_y
        mask = tmp_vf.mask
        dx = axe_x[1] - axe_x[0]
        dy = axe_y[1] - axe_y[0]
        _, Exy = np.gradient(comp_x, dx, dy)
        Eyx, _ = np.gradient(comp_y, dx, dy)
        vort = 1./2.*(Eyx - Exy)
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
    elif isinstance(vf, TemporalVectorFields):
        if raw:
            vort_tsf = np.empty((len(vf.fields), vf.shape[0], vf.shape[1]),
                                dtype=float)
            for i, field in enumerate(vf.fields):
                vort_tsf[i] = get_vorticity(field, raw=True)
        else:
            vort_tsf = TemporalScalarFields()
            for i, field in enumerate(vf.fields):
                tmp_vort = get_vorticity(field, raw=False)
                vort_tsf.add_field(tmp_vort, time=vf.times[i],
                                   unit_times=vf.unit_times)
        return vort_tsf
    else:
        raise TypeError()


def get_stokes_vorticity(vf, window_size=2, raw=False):
    """
    Return a scalar field with the z component of the vorticity using
    Stokes' theorem.

    Parameters
    ----------
    vf : VectorField or Velocityfield
        Field on which compute shear stress
    window_size : integer, optional
        Window size for stokes approximation of the vorticity.
    raw : boolean, optional
        If 'True', return an arrays,
        if 'False' (default), return a ScalarField object.

    Notes
    -----
    Seal et al., “Quantitative characteristics of a laminar,
    unsteady necklace vortex system at a rectangular block-flat plate
    juncture,” Journal of Fluid Mechanics, vol. 286, pp. 117–135, 1995.

    """
    # getting data
    axe_x, axe_y = vf.axe_x, vf.axe_y
    dx = axe_x[1] - axe_x[0]
    dy = axe_y[1] - axe_y[0]
    Vx = vf.comp_x
    Vy = vf.comp_y
    mask = vf.mask
    # creating new axis
    new_axe_x = np.arange(np.mean(axe_x[0:window_size]),
                          np.mean(axe_x[-window_size::] + dx*.9),
                          dx)
    new_axe_y = np.arange(np.mean(axe_y[0:window_size]),
                          np.mean(axe_y[-window_size::] + dy*.9),
                          dy)
    # Loop on field
    vort = np.zeros((len(new_axe_x), len(new_axe_y)))
    new_mask = np.zeros((len(new_axe_x), len(new_axe_y)), dtype=bool)
    for i in np.arange(len(axe_x) - window_size + 1):
        for j in np.arange(len(axe_y) - window_size + 1):
            # reinitialazing
            tmp_vort = 0.
            # checking masked values
            if np.any(mask[i:i + window_size, j:j + window_size]):
                new_mask[i, j] = True
                continue
            # summing over first border
            bord_vec = -Vy[i, j:j + window_size].copy()
            tmp_vort += np.trapz(bord_vec, dx=dy)
            # summing over second border
            bord_vec = Vy[i + window_size - 1, j:j + window_size].copy()
            tmp_vort += np.trapz(bord_vec, dx=dy)
            # summing over third border
            bord_vec = Vx[i:i + window_size, j].copy()
            tmp_vort += np.trapz(bord_vec, dx=dx)
            # summing over fourth border
            bord_vec = -Vx[i:i + window_size, j + window_size - 1].copy()
            tmp_vort += np.trapz(bord_vec, dx=dx)
            # adding coefficients
            tmp_vort *= 1./(dx*dy*window_size**2)
            # storing
            vort[i, j] = tmp_vort
    # returning
    if raw:
        return vort
    else:
        unit_values = vf.unit_values/vf.unit_x
        vort *= unit_values.asNumber()
        unit_values /= unit_values.asNumber()
        vort_sf = ScalarField()
        vort_sf.import_from_arrays(new_axe_x, new_axe_y, vort, mask=new_mask,
                                   unit_x=vf.unit_x, unit_y=vf.unit_y,
                                   unit_values=unit_values)
        return vort_sf


def get_swirling_strength(vf, raw=False):
    """
    Return a scalar field with the swirling strength
    (imaginary part of the eigenvalue of the velocity Jacobian)

    Parameters
    ----------
    vf : VectorField or Velocityfield
        Field on which compute shear stress
    raw : boolean, optional
        If 'True', return an arrays,
        if 'False' (default), return a ScalarField object.

    Notes
    -----
    Zhou, J., R. J. Adrian, S. Balachandar, et T. M. Kendall.
    « Mechanisms for generating coherent packets of hairpin vortices in
    channel flow ». Journal of Fluid Mechanics 387 (mai 1999): 353‑96.

    """
    if not isinstance(vf, VectorField):
        raise TypeError()
    tmp_vf = vf.copy()
    tmp_vf.fill()
    # Getting gradients and axes
    axe_x, axe_y = tmp_vf.axe_x, tmp_vf.axe_y
    mask = tmp_vf.mask
    du_dx, du_dy, dv_dx, dv_dy = get_gradients(vf, raw=True)
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


def get_improved_swirling_strength(vf, raw=False):
    """
    Return a scalar field with the improved swirling strength

    Parameters
    ----------
    vf : VectorField or Velocityfield
        Field on which compute shear stress
    raw : boolean, optional
        If 'True', return an arrays,
        if 'False' (default), return a ScalarField object.

    Notes
    -----
    Chakraborty, Pinaki, S. Balachandar, et Ronald J. Adrian.
    « On the Relationships between Local Vortex Identification Schemes ».
    Journal of Fluid Mechanics 535 (5 juillet 2005): 189‑214.

    """
    if not isinstance(vf, VectorField):
        raise TypeError()
    tmp_vf = vf.copy()
    tmp_vf.fill()
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
                lambcr = np.real(eigvals[0])
                lambci = np.abs(np.imag(eigvals[0]))
                if lambci == 0:
                    mask[i, j] = True
                swst[i, j] = lambcr/lambci
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
        Q criterion will be compute only where zone is 'False'.
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
    # calcul des gradients
    Exx, Exy, Eyx, Eyy = get_gradients(vectorfield, raw=True)
    # calcul de Qcrit
    norm_rot2 = 1/2.*(Exy - Eyx)**2
    norm_shear2 = (Exx**2 + 1./2.*(Exy + Eyx)**2 + Eyy**2)
    qcrit = .5*(norm_rot2) - norm_shear2
    unit_values = (vectorfield.unit_values/vectorfield.unit_x)**2
    scale = unit_values.asNumber()
    qcrit *= scale
    unit_values = unit_values/scale
    if raw:
        return np.ma.masked_array(qcrit, mask)
    else:
        q_sf = ScalarField()
        q_sf.import_from_arrays(axe_x, axe_y, qcrit, mask,
                                unit_x=vectorfield.unit_x,
                                unit_y=vectorfield.unit_y,
                                unit_values=unit_values)
        return q_sf


def get_Nk_criterion(vectorfield, mask=None, raw=False):
    """
    Return the scalar field of the 2D Nk criterion .
    Define as "||Omega||/||S||" , with "||Omega||" the rotation rate tensor
    norm and ||S|| the shear rate tensor norm.

    Parameters
    ----------
    vectorfield : VectorField object
    mask : array of boolean, optional
        Has to be an array of the same size of the vector field object,
        Nk criterion will be compute only where zone is 'False'.
    raw : boolean, optional
        If 'False' (default), a ScalarField is returned,
        if 'True', an array is returned.

    Notes
    -----
    See J. Jeong and F. Hussain, “On the identification of a vortex,” Journal
    of Fluid Mechanics, vol. 285, pp. 69–94, 1995.

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
    # calcul des gradients
    Exx, Exy, Eyx, Eyy = get_gradients(vectorfield, raw=True)
    # calcul de Nk
    norm_rot = 1./2.**.5*np.abs(Exy - Eyx)
    norm_shear = (Exx**2 + 1./2.*(Exy + Eyx)**2 + Eyy**2)**.5
    Nkcrit = norm_rot/norm_shear
    unit_values = make_unit('')
    if raw:
        return np.ma.masked_array(Nkcrit, mask)
    else:
        q_sf = ScalarField()
        q_sf.import_from_arrays(axe_x, axe_y, Nkcrit, mask,
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

    Note
    ----
    Negative values of Delta mean that the local streamline pattern is closed
    or spiraled.
    """
    if not isinstance(vectorfield, VectorField):
        raise TypeError("'vectorfield' must be a VectorField object")
    if mask is None:
        mask = np.zeros(vectorfield.shape, dtype=bool)
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'mask' must be an array of boolean")
    else:
        mask = np.array(mask, dtype=bool)
    axe_x, axe_y = vectorfield.axe_x, vectorfield.axe_y
    # calcul des gradients
    Exx, Exy, Eyx, Eyy = get_gradients(vectorfield, raw=True)
    # calcul de Q
    norm_rot2 = 1/2.*(Exy - Eyx)**2
    norm_shear2 = (Exx**2 + 1./2.*(Exy + Eyx)**2 + Eyy**2)
    Q = .5*(norm_rot2) - norm_shear2
    # calcul de R
    R = np.zeros(Exx.shape)
    for i in np.arange(Exx.shape[0]):
        for j in np.arange(Exx.shape[1]):
            Jac = [[Exx[i, j], Exy[i, j]], [Eyx[i, j], Eyy[i, j]]]
            eigval, eigvect = np.linalg.eig(Jac)
            if np.all(np.imag(eigval) == 0):
                mask[i, j] = True
                continue
            R[i, j] = -np.linalg.det(Jac)
    # calcul de Delta
    delta = (Q/3.)**3 + (R/2.)**2
    unit_values = ((vectorfield.unit_values/vectorfield.unit_x)**2)**3
    scale = unit_values.asNumber()
    delta *= scale
    unit_values = unit_values/scale
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
    Return the lambda2 scalar field. According to ... vortex are defined by
    zone of negative values of lambda2.
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
    for i in np.arange(lambda2.shape[0]):
        for j in np.arange(lambda2.shape[1]):
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
            l2 = np.min(np.real(lambds))
            # storing lambda2
            lambda2[i, j] = l2
    # returning
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
    if isinstance(vf, VectorField):
        # getting data
        tmp_vf = vf.copy()
        tmp_vf.fill(inplace=True, reduce_tri=True)
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
        sign_omega = np.zeros(omega.shape, dtype=int)
        sign_omega[omega_abs == 0] = 1.
        sign_omega[omega_abs != 0] = omega[omega_abs != 0]/omega_abs[omega_abs != 0]
        filt = s < omega_abs
        # getting the residual vorticity
        res_vort = np.zeros(tmp_vf.shape)
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
    elif isinstance(vf, TemporalVectorFields):
        if raw:
            tvf = []
            for field in vf.fields:
                tvf.append(get_residual_vorticity(field, raw=True))
        else:
            tvf = TemporalScalarFields()
            for i, field in enumerate(vf.fields):
                tvf.add_field(get_residual_vorticity(field, raw=False),
                              time=vf.times[i], unit_times=vf.unit_times)
        # return
        return tvf
    else:
        raise TypeError()


def get_shear_vorticity(vf, raw=False):
    """
    Return a scalar field with the shear vorticity.
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
    tmp_vf.crop_masked_border()
    tmp_vf.fill(inplace=True, reduce_tri=True)
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
    filt1 = s < omega_abs
    filt2 = np.logical_not(filt1)
    # getting the residual vorticity
    sh_vort = np.zeros(vf.shape)
    sh_vort[filt2] = omega[filt2]
    sh_vort[filt1] = sign_omega[filt1]*(s[filt1])
    if raw:
        return sh_vort
    else:
        unit_x, unit_y = tmp_vf.unit_x, tmp_vf.unit_y
        unit_values = vf.unit_values/vf.unit_x
        sh_vort *= unit_values.asNumber()
        unit_values /= unit_values.asNumber()
        vort_sf = ScalarField()
        vort_sf.import_from_arrays(axe_x, axe_y, sh_vort, mask=mask,
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
