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

from ... import VectorField, ScalarField
from ...utils.types import ARRAYTYPES, NUMBERTYPES, STRINGTYPES
import numpy as np
from scipy.interpolate import RectBivariateSpline


class FTLE(object):

    def __init__(self, vf, interp='linear'):
        # ceck params
        if not isinstance(vf, VectorField):
            raise TypeError()
        # store
        self.vf = vf
        self.axe_x = vf.axe_x
        self.axe_y = vf.axe_y
        self.dx = vf.axe_x[1] - vf.axe_x[0]
        self.dy = vf.axe_y[1] - vf.axe_y[0]
        self.Vx = vf.comp_x
        self.Vy = vf.comp_y
        if interp == 'linear':
            self.Vx_tor = RectBivariateSpline(self.axe_x, self.axe_y, self.Vx,
                                              kx=1, ky=1)
            self.Vy_tor = RectBivariateSpline(self.axe_x, self.axe_y, self.Vy,
                                              kx=1, ky=1)

        elif interp == 'cubic':
            self.Vx_tor = RectBivariateSpline(self.axe_x, self.axe_y, self.Vx,
                                              kx=3, ky=3)
            self.Vy_tor = RectBivariateSpline(self.axe_x, self.axe_y, self.Vy,
                                              kx=3, ky=3)

    def generate_point_field(self, res):
        """
        Generate a point field on the vector field.

        Parameters
        ----------
        res : number or 2x1 tuple of numbers
            Resolution. If a single number, resolution is applied on both axis,
            if a tuple, separated resolutions are applied on x and y axis.
        """
        # check
        if isinstance(res, NUMBERTYPES):
            res_x = res
            res_y = res
        elif isinstance(res, ARRAYTYPES):
            res = np.array(res)
            if res.shape != (2,):
                raise ValueError()
            res_x = res[0]
            res_y = res[1]
        else:
            raise TypeError()
        # create grid
        dx = (self.axe_x[-1] - self.axe_x[0])/(res_x + 1)
        dy = (self.axe_y[-1] - self.axe_y[0])/(res_y + 1)
        delta = (dx**2 + dy**2)**.5/2.
        pos_x = np.arange(self.axe_x[0] + delta/2., self.axe_x[-1], delta)
        pos_y = np.arange(self.axe_y[0] + delta/2., self.axe_y[-1], delta)
        # returning
        return pos_x, pos_y, dx, dy

    def get_pts_displacement(self, xy, dt, rel_err=1.e-5, dampl=.5):
        """
        Return the lagrangien displacement of a particule initialy at the
        position xy.
        Use Runge-Kutta algorithm with Fehlberg adaptative time step (RKF45).

        Parameters
        ----------
        xy : 2x1 tuple of number
            Point initial position
        dt : number
            displacement time interval
        rel_err : number
            relative maximum error for rk4 algorithm
        dampl : number
            Between 0 and 1, dampling for rk45 algorithm
        """
        # check
        if not isinstance(xy, ARRAYTYPES):
            raise TypeError()
        xy = np.array(xy)
        if not xy.shape == (2,):
            raise ValueError()
        if not isinstance(dt, NUMBERTYPES):
            raise TypeError()
        if dt == 0:
            raise ValueError()
        if dt < 0:
            reverse = True
            dt = -dt
        else:
            reverse = False
        if not isinstance(rel_err, NUMBERTYPES):
            raise TypeError()
        if rel_err <= 0:
            raise ValueError()
        if not isinstance(dampl, NUMBERTYPES):
            raise TypeError()
        if dampl < 0 or dampl > 1:
            raise ValueError()
        # velocity function
        if reverse:
            def fun(xy):
                return -np.array([self.Vx_tor(*xy)[0][0],
                                  self.Vy_tor(*xy)[0][0]])
        else:
            def fun(xy):
                return np.array([self.Vx_tor(*xy)[0][0],
                                 self.Vy_tor(*xy)[0][0]])

        #rk algo
        def rkf45(xy_init, t, rk_dt):
            k1 = fun(xy_init)*rk_dt
            k2 = fun(xy_init + k1/4.)*rk_dt
            k3 = fun(xy_init + 3./32.*k1 + 9./32.*k2)*rk_dt
            k4 = fun(xy_init + 1932./2197.*k1 - 7200./2197.*k2 +
                     7296./2197.*k3)*rk_dt
            k5 = fun(xy_init + 439./216.*k1 - 8*k2 + 3680./513.*k3 -
                     845./4104.*k4)*rk_dt
            k6 = fun(xy_init - 8./27.*k1 + 2.*k2 - 3544./2565.*k3 +
                     1859./4104.*k4 - 11./40.*k5)*rk_dt
            new_xy = (xy_init + 25./216.*k1 + 1408./2565.*k3 +
                      2197./4104.*k4 - 1./5.*k5)
            best_xy = (xy_init + 16./135.*k1 + 6656./12825.*k3 +
                       28561./56430.*k4 - 9./50.*k5 + 2./55.*k6)
            err = (np.linalg.norm(best_xy - new_xy) /
                   np.linalg.norm(new_xy - xy_init))
            # compute new adaptative rk_dt and return
            new_rk_dt = ((rel_err*rk_dt)/(2.*err))**.25
            new_rk_dt = (dampl*rk_dt + (1 - dampl)*new_rk_dt)
            t += rk_dt
            return new_xy, t, new_rk_dt

        # initiate rk_dt
        t = 0.
        rk_dt = self.dx/np.linalg.norm(fun(xy))
        # Time loop
        new_xy = xy
        t_end = False
        border = False
        while True:
            # stoping test
            if new_xy[0] < self.axe_x[0] or new_xy[0] > self.axe_x[-1]:
                border = True
                break
            if new_xy[1] < self.axe_y[0] or new_xy[1] > self.axe_y[-1]:
                border = True
                break
            if t_end:
                break
            if t + rk_dt > dt:
                t_end = True
                rk_dt = dt - t
            # rk adaptative step
            new_xy, t, rk_dt = rkf45(new_xy, t, rk_dt)
        return new_xy, border

    def get_displaced_point_field(self, res, dt, rel_err=1.e-5, dampl=.5):
        """
        Parameters
        ----------
        res : number or 2x1 tuple of numbers
            Resolution. If a single number, resolution is applied on both axis,
            if a tuple, separated resolutions are applied on x and y axis.
        dt : number
            displacement time interval
        rel_err : number
            relative maximum error for rk4 algorithm
        dampl : number
            Between 0 and 1, dampling for rk45 algorithm
        """
        axe_x, axe_y, dx, dy = self.generate_point_field(res=res)
        pos_X, pos_Y = np.meshgrid(axe_x, axe_y, indexing='ij')
        # get dispalced points
        new_pos_X = np.zeros(pos_X.shape)
        new_pos_Y = np.zeros(pos_X.shape)
        mask = np.zeros(pos_X.shape, dtype=bool)
        for i in np.arange(pos_X.shape[0]):
            for j in np.arange(pos_X.shape[1]):
                pt = (pos_X[i, j], pos_Y[i, j])
                tmp_pt, out = self.get_pts_displacement(pt, dt=dt,
                                                        rel_err=rel_err,
                                                        dampl=dampl)
                if out is True:
                    mask[i, j] = True
                new_pos_X[i, j] = tmp_pt[0]
                new_pos_Y[i, j] = tmp_pt[1]
        # returning
        if np.any(mask):
            new_pos_X = np.ma.masked_array(new_pos_X, mask)
            new_pos_Y = np.ma.masked_array(new_pos_Y, mask)
        return new_pos_X, new_pos_Y

    def get_FTLE(self, res, dt, rel_err=1.e-5, dampl=.5):
        """
        Parameters
        ----------
        res : number or 2x1 tuple of numbers
            Resolution. If a single number, resolution is applied on both axis,
            if a tuple, separated resolutions are applied on x and y axis.
        dt : number
            displacement time interval (can be negative for backward FTLE)
        rel_err : number
            relative maximum error for rk4 algorithm
        dampl : number
            Between 0 and 1, dampling for rk45 algorithm
        """
        # get points
        axe_x, axe_y, dx, dy = self.generate_point_field(res=res)
        pos_X, pos_Y = np.meshgrid(axe_x, axe_y, indexing='ij')
        # get dispalced points
        new_pos_X = np.zeros(pos_X.shape)
        new_pos_Y = np.zeros(pos_X.shape)
        mask = np.zeros(pos_X.shape, dtype=bool)
        for i in np.arange(pos_X.shape[0]):
            for j in np.arange(pos_X.shape[1]):
                pt = (pos_X[i, j], pos_Y[i, j])
                tmp_pt, out = self.get_pts_displacement(pt, dt=dt,
                                                        rel_err=rel_err,
                                                        dampl=dampl)
                if out is True:
                    mask[i, j] = True
                new_pos_X[i, j] = tmp_pt[0]
                new_pos_Y[i, j] = tmp_pt[1]
        # get flow map gradient
        new_pos_X = np.ma.masked_array(new_pos_X, mask)
        new_pos_Y = np.ma.masked_array(new_pos_Y, mask)
        pos_X_dx, pos_X_dy = np.gradient(new_pos_X, dx, dy)
        pos_Y_dx, pos_Y_dy = np.gradient(new_pos_Y, dx, dy)
        mask = np.logical_or(pos_X_dx.mask, pos_X_dy.mask)
        # get eigenvalues
        ftle = np.zeros(pos_X.shape)
        for i in np.arange(pos_X.shape[0]):
            for j in np.arange(pos_X.shape[1]):
                if mask[i, j]:
                    continue
                phi = np.array([[pos_X_dx[i, j], pos_X_dy[i, j]],
                                [pos_Y_dx[i, j], pos_Y_dy[i, j]]])
                Delta = np.dot(phi, np.transpose(phi))
                eigval, eigvect = np.linalg.eig(Delta)
                ftle[i, j] = np.max(np.real(eigval))
        # normalize ?
        ftle[~mask] = 1./np.abs(dt)*np.log(ftle[~mask]**.5)
        # return
        SF = ScalarField()
        SF.import_from_arrays(axe_x=axe_x, axe_y=axe_y, values=ftle,
                              mask=mask,
                              unit_x=self.vf.unit_x, unit_y=self.vf.unit_y)
        return SF
