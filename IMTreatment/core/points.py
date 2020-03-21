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

import matplotlib.pyplot as plt
from .. import plotlib as pplt
import numpy as np
import unum
import copy
from scipy import stats
from scipy import ndimage
from ..utils.types import ARRAYTYPES, INTEGERTYPES, NUMBERTYPES, STRINGTYPES
from ..utils import make_unit


class Points(object):

    def __init__(self, xy=np.empty((0, 2), dtype=float), v=[],
                 unit_x='', unit_y='', unit_v='', name=''):
        """
        Class representing a set of points.
        You can use 'make_unit' to provide unities.

        Parameters
        ----------
        xy : nx2 array.
            Representing the coordinates of each point of the set (n points).
        v : n array, optional
            Representing values attached at each points.
        unit_x : Unit object, optional
            X unit_y.
        unit_y : Unit object, optional
            Y unit_y.
        unit_v : Unit object, optional
            values unit_y.
        name : string, optional
            Name of the points set
        """
        self.__v = []
        if len(xy) == 0:
            xy = np.empty((0, 2), dtype=float)
        self.xy = xy
        self.v = v
        self.unit_v = unit_v
        self.unit_x = unit_x
        self.unit_y = unit_y
        self.name = name

    def __iter__(self):
        if self.v is None or len(self.v) == 0:
            for i in np.arange(len(self.xy)):
                yield self.xy[i]
        else:
            for i in np.arange(len(self.xy)):
                yield self.xy[i], self.v[i]

    def __len__(self):
        return self.xy.shape[0]

    def __add__(self, another):
        if isinstance(another, Points):
            # trivial additions
            if len(self.xy) == 0:
                return another.copy()
            elif len(another.xy) == 0:
                return self.copy()
            # checking unit systems
            if len(self.xy) != 0:
                try:
                    self.unit_x + another.unit_x
                    self.unit_y + another.unit_y
                    if self.v is not None and another.v is not None:
                        self.unit_v + another.unit_v
                except unum.IncompatibleUnitsError:
                    raise ValueError("Units system are not the same")
            else:
                self.unit_x = another.unit_x
                self.unit_y = another.unit_y
                if another.v is not None:
                    self.unit_v = another.unit_v
            # compacting coordinates
            if another.xy.shape == (0,):
                new_xy = self.xy
            elif self.xy.shape == (0,):
                xy = another.xy
                xy[:, 0] = xy[:, 0]*(self.unit_x/another.unit_x).asNumber()
                xy[:, 1] = xy[:, 1]*(self.unit_y/another.unit_y).asNumber()
                new_xy = xy
            elif another.xy.shape == (0,) and self.xy.shape == (0,):
                new_xy = np.array([[]])
            else:
                xy = another.xy
                xy[:, 0] = xy[:, 0]*(self.unit_x/another.unit_x).asNumber()
                xy[:, 1] = xy[:, 1]*(self.unit_y/another.unit_y).asNumber()
                new_xy = np.append(self.xy, xy, axis=0)
            # testing v presence
            v_presence = True
            if self.v is None and another.v is None:
                if len(self.xy) != 0:
                    v_presence = False
            elif self.v is not None and another.v is not None:
                    v_presence = True
            else:
                raise Exception()
            # compacting points and returning
            if v_presence:
                if self.v is None and another.v is None:
                    v = np.array([])
                elif self.v is None:
                    v = another.v*(self.unit_v/another.unit_v).asNumber()
                elif another.v is None:
                    v = self.v
                else:
                    v_tmp = another.v*(self.unit_v/another.unit_v).asNumber()
                    v = np.append(self.v, v_tmp)
                return Points(new_xy, v,
                              unit_x=self.unit_x,
                              unit_y=self.unit_y,
                              unit_v=self.unit_v)
            else:
                return Points(new_xy,
                              unit_x=self.unit_x,
                              unit_y=self.unit_y)
        else:
            raise Exception("You can't add {} to Points objects"
                            .format(type(another)))

    def __eq__(self, obj):
        if not isinstance(obj, Points):
            return False
        if not np.all(self.xy == obj.xy):
            return False
        if not np.all(self.v == obj.v):
            return False
        if not self.unit_x == obj.unit_x:
            return False
        if not self.unit_y == obj.unit_y:
            return False
        if not self.unit_v == obj.unit_v:
            return False
        return True

    @property
    def xy(self):
        return self.__xy

    @xy.setter
    def xy(self, values):
        values = np.asarray(values, dtype=float)
        if len(values != 0):
            if not values.ndim == 2:
                raise ValueError("ndim of xy is {} and should be 2"
                                 .format(values.ndim))
            if not values.shape[1] == 2:
                raise ValueError()
        self.__xy = values
        if len(values) != len(self.__v):
            self.__v = np.array([])

    @xy.deleter
    def xy(self):
        raise Exception("Nope, can't do that")

    @property
    def v(self):
        return self.__v

    @v.setter
    def v(self, values):
        if values is None:
            self__v = None
        else:
            values = np.array(values, subok=True)
            if not values.ndim == 1:
                raise ValueError()
            if not len(values) in [0, len(self.__xy)]:
                raise ValueError()
            self.__v = values

    @v.deleter
    def v(self):
        raise Exception("Nope, can't do that")

    @property
    def unit_x(self):
        return self.__unit_x

    @unit_x.setter
    def unit_x(self, unit):
        if isinstance(unit, unum.Unum):
            self.__unit_x = unit
        elif isinstance(unit, STRINGTYPES):
            try:
                self.__unit_x = make_unit(unit)
            except (ValueError, TypeError):
                raise Exception()
        else:
            raise Exception()

    @unit_x.deleter
    def unit_x(self):
        raise Exception("Nope, can't delete 'unit_x'")

    @property
    def unit_y(self):
        return self.__unit_y

    @unit_y.setter
    def unit_y(self, unit):
        if isinstance(unit, unum.Unum):
            self.__unit_y = unit
        elif isinstance(unit, STRINGTYPES):
            try:
                self.__unit_y = make_unit(unit)
            except (ValueError, TypeError):
                raise Exception()
        else:
            raise Exception()

    @unit_y.deleter
    def unit_y(self):
        raise Exception("Nope, can't delete 'unit_y'")

    @property
    def unit_v(self):
        return self.__unit_v

    @unit_v.setter
    def unit_v(self, unit):
        if isinstance(unit, unum.Unum):
            self.__unit_v = unit
        elif isinstance(unit, STRINGTYPES):
            try:
                self.__unit_v = make_unit(unit)
            except (ValueError, TypeError):
                raise Exception()
        else:
            try:
                self.__unit_v = make_unit(str(unit))
            except Exception as msg:
                print(msg)
                print(type(unit))
                print(unit)
                raise Exception()

    @unit_v.deleter
    def unit_v(self):
        raise Exception("Nope, can't delete 'unit_v'")

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        if isinstance(name, STRINGTYPES):
            self.__name = name
        else:
            raise Exception()

    @name.deleter
    def name(self):
        raise Exception("Nope, can't delete 'name'")

    def copy(self):
        """
        Return a copy of the Points object.
        """
        return copy.deepcopy(self)

    def get_points_density(self, bw_method=None, resolution=100,
                           output_format='normalized', raw=False):
        """
        Return a ScalarField with points density.

        Parameters
        ----------
        bw_method : str, scalar or callable, optional
            The method used to calculate the estimator bandwidth.
            This can be 'scott', 'silverman', a scalar constant or
            a callable. If a scalar, this will be used as percent of
            the data std. If a callable, it should take a gaussian_kde
            instance as only parameter and return a scalar.
            If None (default), 'scott' is used.
        resolution : integer, optional
            Resolution for the resulting field.
        output_format : string, optional
            'normalized' (default) : give position probability
                                     (integral egal 1).
            'ponderated' : give position probability ponderated by the number
                           or points (integral egal number of points).
            'concentration' : give local concentration (in point per surface).
        raw : boolean, optional
            If 'False' (default), return a ScalarField object,
            if 'True', return numpy array.

        Returns
        -------
        density : array, ScalarField object or None
            Return 'None' if there is not enough points in the cloud.
        """
        # checking points length
        if len(self.xy) < 2:
            return None
        # getting data
        min_x = np.min(self.xy[:, 0])
        max_x = np.max(self.xy[:, 0])
        min_y = np.min(self.xy[:, 1])
        max_y = np.max(self.xy[:, 1])
        # getting min, max values and resolution in each direction
        width_x = max_x - min_x
        width_y = max_y - min_y
        if width_x == 0 and width_y == 0:
            raise Exception()
        elif width_x == 0:
            min_x = min_x - width_y/2.
            max_x = max_x + width_y/2.
            width_x = width_y
            res_x = resolution
            res_y = resolution
        elif width_y == 0:
            min_y = min_y - width_x/2.
            max_y = max_y + width_x/2.
            width_y = width_x
            res_x = resolution
            res_y = resolution
        elif width_x > width_y:
            res_x = resolution
            res_y = int(np.round(resolution*width_y/width_x))
        else:
            res_y = resolution
            res_x = int(np.round(resolution*width_x/width_y))
        if res_x < 2 or res_y < 2:
            raise ValueError()
        # check potential singular covariance matrix situations
        if (np.all(self.xy[:, 0] == self.xy[0, 0]) or
                np.all(self.xy[:, 1] == self.xy[0, 1])):
            return None
        # get kernel using scipy
        std = np.mean([np.std(self.xy[:, 0]), np.std(self.xy[:, 1])])
        if isinstance(bw_method, NUMBERTYPES):
            if width_x > width_y:
                ad_len = width_y
            else:
                ad_len = width_x
            ad_bw_method = bw_method*std/ad_len
        else:
            ad_bw_method = bw_method
        kernel = stats.gaussian_kde(self.xy.transpose(),
                                    bw_method=ad_bw_method)
        # little adaptation to avoid streched density map
        if width_x > width_y:
            kernel.inv_cov[0, 0] = np.max([kernel.inv_cov])
        else:
            kernel.inv_cov[1, 1] = np.max([kernel.inv_cov])
        kernel.inv_cov[0, 1] *= 0
        kernel.inv_cov[1, 0] *= 0
        # creating grid
        if width_x > width_y:
            dx_border = kernel.factor*width_y/2.
            dy_border = dx_border
        else:
            dx_border = kernel.factor*width_x/2.
            dy_border = dx_border
        axe_x = np.linspace(min_x - dx_border, max_x + dx_border, res_x)
        axe_y = np.linspace(min_y - dy_border, max_y + dy_border, res_y)
        X, Y = np.meshgrid(axe_x, axe_y)
        X = X.flatten()
        Y = Y.flatten()
        positions = np.array([[X[i], Y[i]]
                              for i in np.arange(len(X))]).transpose()
        # estimating density
        values = kernel(positions)
        values = values.reshape((res_y, res_x)).transpose()
        # normalize (not normalized yet because of the modification of inv_cov)
        dx = axe_x[1] - axe_x[0]
        dy = axe_y[1] - axe_y[0]
        values /= np.sum(np.sum(values))*(dx)*(dy)
        # adapt to wanted output_format
        if output_format is None or output_format == "normalized":
            unit_values = make_unit('')
        elif output_format == 'ponderated':
            values = values*len(self.xy)
            unit_values = make_unit('')
        elif output_format == "percentage":
            values = values*100
            unit_values = make_unit('')
        elif output_format == "concentration":
            unit_values = 1/self.unit_x/self.unit_y
            values = values*len(self.xy)
        else:
            raise ValueError()
        # return
        if np.all(np.isnan(values)) or np.all(values == np.inf):
            return None
        if raw:
            return values
        else:
            from .scalarfield import ScalarField
            sf = ScalarField()
            sf.import_from_arrays(axe_x, axe_y, values, mask=False,
                                  unit_x=self.unit_x, unit_y=self.unit_y,
                                  unit_values=unit_values)
            return sf

    def get_points_density2(self, res, subres=None, raw=False,
                            ponderated=False):
        """
        Return a ScalarField with points density.

        Parameters
        ----------
        res : number or 2x1 array of numbers
            fdensity field number of subdivision.
            Can be the same number for both axis,  or one number per axis
            (need to give a tuple).
        raw : boolean, optional
            If 'False' (default), return a ScalarField object,
            if 'True', return numpy array.
        ponderated : boolean, optiona
            If 'True', values associated to points are used to ponderate the
            density field. Default is 'False'.
        subres : odd integer, optional
            If specified, a subgrid of resolution res*subres is used to
            make result more accurate.
        """
        # checking parameters
        if isinstance(res, int):
            res_x = res
            res_y = res
        elif isinstance(res, ARRAYTYPES):
            if len(res) != 2:
                raise ValueError()
            res_x = res[0]
            res_y = res[1]
        else:
            raise TypeError()
        if not isinstance(raw, bool):
            raise TypeError()
        if not isinstance(ponderated, bool):
            raise TypeError()
        if isinstance(subres, int) and subres > 0:
            subres = np.floor(subres/2)*2
            subres2 = (subres)/2
        elif subres is None:
            pass
        else:
            raise TypeError()
        # If we use a subgrid
        if subres is not None:
            # creating grid
            min_x = np.min(self.xy[:, 0])
            max_x = np.max(self.xy[:, 0])
            min_y = np.min(self.xy[:, 1])
            max_y = np.max(self.xy[:, 1])
            dx = (max_x - min_x)/(res_x)
            dy = (max_y - min_y)/(res_y)
            sub_dx = dx/subres
            sub_dy = dy/subres
            axe_x = np.arange(min_x - dx/2, max_x + dx/2 + sub_dx, sub_dx)
            axe_y = np.arange(min_y - dy/2, max_y + dy/2 + sub_dy, sub_dy)
            values = np.zeros((len(axe_x), len(axe_y)))
            # filling grid with density
            for i, pt in enumerate(self.xy):
                x = pt[0]
                y = pt[1]
                ind_x = np.argmin(np.abs(axe_x - x))
                ind_y = np.argmin(np.abs(axe_y - y))
                slic_x = slice(ind_x - subres2 + 1, ind_x + subres2)
                slic_y = slice(ind_y - subres2 + 1, ind_y + subres2)
                if ponderated:
                    values[slic_x, slic_y] += self.v[i]
                else:
                    values[slic_x, slic_y] += 1
            values /= (dx*dy)
            values = values[subres2:-subres2, subres2:-subres2]
            axe_x = axe_x[subres2:-subres2]
            axe_y = axe_y[subres2:-subres2]
        # if we do not use a subgrid
        else:
            # creating grid
            min_x = np.min(self.xy[:, 0])
            max_x = np.max(self.xy[:, 0])
            min_y = np.min(self.xy[:, 1])
            max_y = np.max(self.xy[:, 1])
            axe_x, dx = np.linspace(min_x, max_x, res_x, retstep=True)
            axe_y, dy = np.linspace(min_y, max_y, res_y, retstep=True)
            values = np.zeros((len(axe_x), len(axe_y)))
            # filling grid with density
            for i, pt in enumerate(self.xy):
                x = pt[0]
                y = pt[1]
                ind_x = np.argmin(np.abs(axe_x - x))
                ind_y = np.argmin(np.abs(axe_y - y))
                if ponderated:
                    values[ind_x, ind_y] += self.v[i]
                else:
                    values[ind_x, ind_y] += 1
            values /= (dx*dy)
        # return the field
        if raw:
            return values
        else:
            from .scalarfield import ScalarField
            sf = ScalarField()
            if ponderated:
                unit_values = self.unit_v/self.unit_x/self.unit_y
            else:
                unit_values = 1/self.unit_x/self.unit_y
            sf.import_from_arrays(axe_x, axe_y, values, mask=False,
                                  unit_x=self.unit_x, unit_y=self.unit_y,
                                  unit_values=unit_values)
            return sf

    def get_clusters(self, eps, min_samples=5):
        """
        Perform DBSCAN clustering from vector array or distance matrix.
        (see sklearn.cluster.DBSCAN)

        Notes
        ------
        DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
        Finds core samples of high density and expands clusters from them.
        Good for data which contains clusters of similar density.
        """
        try:
            import sklearn.cluster as clst
        except ImportError:
            raise Exception("You need to install `sklearn` to use this "
                            "functionnality")
        X = self.xy
        db = clst.DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = db.labels_
        uniq_labels = np.array(list(set(labels)))
        nmb_cluster = len(uniq_labels)
        if -1 in uniq_labels:
            nmb_cluster -= 1
        # create separated points objects
        points = []
        for lab in uniq_labels:
            filt = labels == lab
            if len(self.v) == len(self.xy):
                new_v = self.v[filt]
            else:
                new_v = self.v
            tmp_pts = Points(self.xy[filt], new_v, unit_x=self.unit_x,
                             unit_y=self.unit_y, unit_v=self.unit_v)
            points.append(tmp_pts)
        return points

    def get_envelope(self, alpha=None):
        """
        Return the convex or concave hull (if alpha specified) for the set of
        points.

        Parameters
        ----------
        alpha : number
            maximum distance between two points of the hull.

        Notes
        -----
        Credit to mlaloux
        (https://github.com/mlaloux/Python--alpha-shape_concave_hull)
        """
        # import shapely functions
        try:
            from shapely.geometry import mapping
            from shapely.geometry import MultiLineString
            from shapely.ops import polygonize, cascaded_union
        except ImportError:
            raise Exception("This functionnality need 'shapely' module")
        if alpha is None:
            alpha = np.inf
        # triangulate
        from scipy.spatial import Delaunay
        points = self.xy
        tri = Delaunay(points)
        # add and sort points
        edges = set()
        edge_points = []

        def add_edge(i, j):
            """Add a line between the i-th and j-th points,
            if not in the list already"""
            if (i, j) in edges or (j, i) in edges:
                return
            edges.add((i, j))
            edge_points.append(points[[i, j]])
        points = np.array(points)
        for ia, ib, ic in tri.vertices:
            pa = points[ia]
            pb = points[ib]
            pc = points[ic]
            # Lengths of sides of triangle
            a = np.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
            b = np.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
            c = np.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
            # Semiperimeter of triangle
            s = (a + b + c)/2.0
            # Area of triangle by Heron's formula
            area = np.sqrt(s*(s-a)*(s-b)*(s-c))
            circum_r = a*b*c/(4.0*area)
            # Here's the radius filter.
            #if circum_r < 1.0/alpha:
            if circum_r < alpha:
                add_edge(ia, ib)
                add_edge(ib, ic)
                add_edge(ic, ia)
        # concatenate polygons
        m = MultiLineString(edge_points)
        triangles = list(polygonize(m))
        hull = mapping(cascaded_union(triangles))['coordinates'][0]
        hull = np.array(hull, dtype=float)
        if hull.ndim == 3:
            hull = hull[0]
        # transform to Points object
        pt = Points(xy=hull, v=[], unit_x=self.unit_x,
                    unit_y=self.unit_y)
        return pt

    def get_velocity(self, incr=1, smooth=0, xaxis='time'):
        """
        Assuming that associated 'v' values are times for each points,
        compute the velocity of the trajectory.

        Parameters
        ----------
        incr : integer, optional
            Increment use to get used points (default is 1).
        smooth : number, optional
            Cut off frequency for the lowpass filter.
        xaxis : string, optional
            Value to put in the profile x axis, can be 'time' (default), 'x'
            or 'y'.

        Returns
        -------
        Vx : Profile object
            Profile of x velocity versus time.
        Vy : Profile object
            Profile of y velocity versus time.
        """
        if smooth < 0:
            raise ValueError()
        if xaxis not in ['time', 'x', 'y']:
            raise ValueError()
        # checking 'v' presence
        if len(self.v) == 0:
            raise Exception()
        # sorting points by time
        ind_sort = np.argsort(self.v)
        times = self.v[ind_sort]
        xy = self.xy[ind_sort]
        x = xy[:, 0]
        y = xy[:, 1]
        # using increment if necessary
        if incr != 1:
            x = x[::incr]
            y = y[::incr]
            times = times[::incr]
        dx = x[1::] - x[:-1]
        dy = y[1::] - y[:-1]
        dt = times[1::] - times[:-1]
        # smoothing if necessary
        if smooth != 0:
            tmp_pts = Points(list(zip(x, y)), times)
            tmp_pts.smooth(tos='lowpass', size=smooth, inplace=True)
            x = tmp_pts.xy[:, 0]
            y = tmp_pts.xy[:, 1]
        # getting velocity between points
        Vx = np.array([(x[i + 1] - x[i])/dt[i]
                       for i in np.arange(len(x) - 1)])
        Vy = np.array([(y[i + 1] - y[i])/dt[i]
                       for i in np.arange(len(y) - 1)])
        # getting xaxis
        if xaxis == 'time':
            x_prof = times[:-1] + dt/2.
        elif xaxis == 'x':
            x_prof = x[:-1] + dx/2.
        elif xaxis == 'y':
            x_prof = y[:-1] + dy/2.
        # returning profiles
        from .profile import Profile
        unit_Vx = self.unit_x/self.unit_v
        Vx *= unit_Vx.asNumber()
        unit_Vx /= unit_Vx.asNumber()
        prof_x = Profile(x_prof, Vx, mask=False, unit_x=self.unit_v,
                         unit_y=unit_Vx)
        unit_Vy = self.unit_y/self.unit_v
        Vy *= unit_Vy.asNumber()
        unit_Vy /= unit_Vy.asNumber()
        prof_y = Profile(x_prof, Vy, mask=False, unit_x=self.unit_v,
                         unit_y=unit_Vy)
        return prof_x, prof_y

    def get_evolution_on_sf(self, SF, axe_x=None):
        """
        Return the evolution of the value represented by a scalar field, on
        the path of the trajectory.

        Parameters
        ----------
        SF : ScalarField object
        axe_x : string, optional
            What put in the x axis (can be 'x', 'y', 'v').
            default is 'v' when available and 'x' else.

        Returns
        -------
        evol : Profile object
        """
        # check parameters
        from .scalarfield import ScalarField
        if not isinstance(SF, ScalarField):
            raise TypeError()
        if len(self.xy) == 0:
            from .profile import Profile
            return Profile()
        if axe_x is None:
            if len(self.v) == len(self.xy):
                axe_x = 'v'
            else:
                axe_x = 'x'
        if not isinstance(axe_x, STRINGTYPES):
            raise TypeError()
        # get x values
        if axe_x == 'v':
            if len(self.v) == 0:
                raise ValueError()
            x_prof = self.v
            unit_x = self.unit_v
        elif axe_x == 'x':
            x_prof = self.xy[:, 0]
            unit_x = self.unit_x
        elif axe_x == 'y':
            x_prof = self.xy[:, 1]
            unit_x = self.unit_y
        else:
            raise ValueError()
        # get the y value
        y_prof = np.empty((len(self.xy)), dtype=float)
        for i, pt in enumerate(self.xy):
            y_prof[i] = SF.get_value(*pt, ind=False, unit=False)
        mask = np.isnan(y_prof)
        unit_y = SF.unit_values
        # returning
        evol = Profile(x_prof, y_prof, mask=mask, unit_x=unit_x,
                       unit_y=unit_y)
        return evol

    def get_evolution_on_tsf(self, TSF, axe_x=None):
        """
        Return the evolution of the value represented by scalar fields, on
        the path of the trajectory.
        Timse of the TSF must be consistent with the times of the Points.

        Parameters
        ----------
        TSF : tsf.TemporalScalarField object
        axe_x : string, optional
            What put in the x axis (can be 'x', 'y', 'v').
            default is 'v' (associated with time)

        Returns
        -------
        evol : Profile object
        """
        # check parameters
        from . import temporalscalarfields as tsf
        from .profile import Profile
        if not isinstance(TSF, tsf.TemporalScalarFields):
            raise TypeError()
        if len(self.xy) == 0:
            return Profile()
        if axe_x is None:
            axe_x = 'v'
        if not isinstance(axe_x, STRINGTYPES):
            raise TypeError()
        # get x values
        if axe_x == 'v':
            if len(self.v) == 0:
                raise ValueError()
            x_prof = self.v
            unit_x = self.unit_v
        elif axe_x == 'x':
            x_prof = self.xy[:, 0]
            unit_x = self.unit_x
        elif axe_x == 'y':
            x_prof = self.xy[:, 1]
            unit_x = self.unit_y
        else:
            raise ValueError()
        # get the y value
        times = self.v
        y_prof = np.empty((len(self.xy)), dtype=float)
        for i, pt in enumerate(self.xy):
            time = times[i]
            SF = TSF.fields[TSF.times == time][0]
            y_prof[i] = SF.get_value(*pt, ind=False, unit=False)
        mask = np.isnan(y_prof)
        unit_y = TSF.unit_values
        # returning
        evol = Profile(x_prof, y_prof, mask=mask, unit_x=unit_x,
                       unit_y=unit_y)
        return evol

    def fit(self, kind='polynomial', order=2, simplify=False):
        """
        Return the parametric coefficients of the fitting curve on the points.

        Parameters
        ----------
        kind : string, optional
            The kind of fitting used. Can be 'polynomial' or 'ellipse'.
        order : integer
            Approximation order for the fitting.
        Simplify : boolean or string, optional
            Can be False (default), 'x' or 'y'. Perform a simplification
            (see Points.Siplify()) before the fitting.

        Returns
        -------
        p : array, only for polynomial fitting
            Polynomial coefficients, highest power first
        radii : array, only for ellipse fitting
            Ellipse demi-axes radii.
        center : array, only for ellipse fitting
           Ellipse center coordinates.
        alpha : number
            Angle between the x axis and the major axis.
        """
        if not isinstance(order, int):
            raise TypeError("'order' must be an integer")
        if not isinstance(kind, STRINGTYPES):
            raise TypeError("'kind' must be a string")
        if not simplify:
            xytmp = self.xy
        elif simplify == 'x':
            xytmp = self.simplify(axe=0).xy
        elif simplify == 'y':
            xytmp = self.simplify(axe=1).xy

        if kind == 'polynomial':
            p = np.polyfit(xytmp[:, 0], xytmp[:, 1], deg=order)
            return p
        elif kind == 'ellipse':
            import fit_ellipse as fte
            res = fte.fit_ellipse(xytmp)
            radii, center, alpha = fte.get_parameters(res)
            return radii, center, alpha

    def add(self, pt, v=None):
        """
        Add a new point.

        Parameters
        ----------
        pt : 2x1 array of numbers
            Point to add.
        v : number, optional
            Value of the point (needed if other points have values).
        """
        # check parameters
        if not isinstance(pt, ARRAYTYPES):
            raise TypeError()
        pt = np.array(pt, subok=True)
        if not pt.shape == (2,):
            raise ValueError()
        if not isinstance(pt[0], NUMBERTYPES):
            raise TypeError()
        if v is not None:
            if not isinstance(v, NUMBERTYPES):
                raise TypeError()
        # store new data
        self.__xy = np.append(self.xy, [pt], axis=0)
        if v is None and self.v.shape[0] != 0:
            raise ValueError('You should specify an associated value : v')
        if v is not None:
            self.__v = np.append(self.v, v)

    def remove(self, ind):
        """
        Remove the point number 'ind' of the points cloud.
        In place.

        Parameters
        ----------
        ind : integer or array of integer
        """
        if isinstance(ind, INTEGERTYPES):
            ind = [ind]
        elif isinstance(ind, ARRAYTYPES):
            if not np.all([isinstance(val, int) for val in ind]):
                raise TypeError("'ind' must be an integer or an array of"
                                " integer")
            ind = np.array(ind)
        else:
            raise TypeError("'ind' must be an integer or an array of integer")
        tmp_v = self.v.copy()
        self.xy = np.delete(self.xy, ind, axis=0)
        if len(self.v) == len(self.xy) + 1:
            self.v = np.delete(tmp_v, ind, axis=0)

    def change_unit(self, axe, new_unit):
        """
        Change the unit of an axe.

        Parameters
        ----------
        axe : string
            'y' for changing the profile y axis unit
            'x' for changing the profile x axis unit
            'v' for changing the profile values unit
        new_unit : Unum.unit object or string
            The new unit.
        """
        if isinstance(new_unit, STRINGTYPES):
            new_unit = make_unit(new_unit)
        if not isinstance(new_unit, unum.Unum):
            raise TypeError()
        if not isinstance(axe, STRINGTYPES):
            raise TypeError()
        if axe == 'x':
            old_unit = self.unit_x
            new_unit = old_unit.asUnit(new_unit)
            fact = new_unit.asNumber()
            self.xy[:, 0] *= fact
            self.unit_x = new_unit/fact
        elif axe == 'y':
            old_unit = self.unit_y
            new_unit = old_unit.asUnit(new_unit)
            fact = new_unit.asNumber()
            self.xy[:, 1] *= fact
            self.unit_y = new_unit/fact
        elif axe == 'v':
            old_unit = self.unit_v
            new_unit = old_unit.asUnit(new_unit)
            fact = new_unit.asNumber()
            self.v *= fact
            self.unit_v = new_unit/fact
        else:
            raise ValueError()

    def set_origin(self, x=None, y=None):
        """
        Set the given point (x, y) as the new referential.

        Parameters
        ----------
        x: number
            .
        y: number
            .
        """
        # Check
        if x is not None:
            if not isinstance(x, NUMBERTYPES):
                raise TypeError()
            self.xy[:, 0] -= x
        if y is not None:
            if not isinstance(y, NUMBERTYPES):
                raise TypeError()
            self.xy[:, 1] -= y

    def crop(self, intervx=None, intervy=None, intervv=None, inplace=True,
             ind=False):
        """
        Crop the points cloud.

        Parameters
        ----------
        intervx : 2x1 tuple
            Interval on x axis
        intervy : 2x1 tuple
            Interval on y axis
        intervv : 2x1 tuple
            Interval on v values

        Returns
        -------
        tmp_pts : Points object
            croped version of the point cloud.
        """
        #
        if inplace:
            tmp_pts = self
        else:
            tmp_pts = self.copy()
        # Getting cropping mask
        if ind:
            if intervx is not None or intervy is not None:
                raise ValueError()
            mask = np.ones(len(self.xy), dtype=bool)
            mask[intervv[0]:intervv[1]] = False
        else:
            mask = np.zeros(len(self.xy), dtype=bool)
            if intervx is not None:
                out_zone = np.logical_or(self.xy[:, 0] < intervx[0],
                                         self.xy[:, 0] > intervx[1])
                mask = np.logical_or(mask, out_zone)
            if intervy is not None:
                out_zone = np.logical_or(self.xy[:, 1] < intervy[0],
                                         self.xy[:, 1] > intervy[1])
                mask = np.logical_or(mask, out_zone)
            if intervv is not None and len(self.v) != 0:
                out_zone = np.logical_or(self.v < intervv[0],
                                         self.v > intervv[1])
                mask = np.logical_or(mask, out_zone)
        # Cropping values
        tmp_pts.__xy = tmp_pts.xy[~mask, :]
        if len(tmp_pts.v) != 0:
            tmp_pts.__v = tmp_pts.v[~mask]
        # returning
        if not inplace:
            return tmp_pts

    def cut(self, intervx=None, intervy=None):
        """
        Return a point cloud where the given area has been removed.

        Parameters
        ----------
        intervx : 2x1 tuple
            Interval on x axis
        intervy : 2x1 tuple
            Interval on y axis

        Returns
        -------
        tmp_pts : Points object
            Cutted version of the point cloud.
        """
        tmp_pts = self.copy()
        mask = np.ones(len(self.xy))
        if intervx is not None:
            out_zone = np.logical_and(self.xy[:, 0] > intervx[0],
                                      self.xy[:, 0] < intervx[1])
            mask = np.logical_and(mask, out_zone)
        if intervy is not None:
            out_zone = np.logical_and(self.xy[:, 1] > intervy[0],
                                      self.xy[:, 1] < intervy[1])
            mask = np.logical_and(mask, out_zone)
        tmp_pts.xy = tmp_pts.xy[~mask, :]
        if len(tmp_pts.v) != 0:
            tmp_pts.v = tmp_pts.v[~mask]
        return tmp_pts

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
            tmp_pt = self
        else:
            tmp_pt = self.copy()
        # adapt unit
        if isinstance(scalex, unum.Unum):
            new_unit = scalex*tmp_pt.unit_x
            fact = new_unit.asNumber()
            new_unit /= fact
            tmp_pt.unit_x = new_unit
            scalex = fact
        if isinstance(scaley, unum.Unum):
            new_unit = scaley*tmp_pt.unit_y
            fact = new_unit.asNumber()
            new_unit /= fact
            tmp_pt.unit_y = new_unit
            scaley = fact
        if isinstance(scalev, unum.Unum):
            new_unit = scalev*tmp_pt.unit_v
            fact = new_unit.asNumber()
            new_unit /= fact
            tmp_pt.unit_v = new_unit
            scalev = fact
        # loop
        if scalex != 1. or scaley != 1.:
            tmp_pt.xy *= np.array([scalex, scaley])
        if scalev != 1.:
            tmp_pt.v *= scalev
        # returning
        if not inplace:
            return tmp_pt

    def rotate(self, angle, inplace=False):
        """
        Rotate the point set.

        Parameters
        ----------
        angle: number
            Rotation angle in radian.
        """
        # Checks
        if inplace:
            tmpp = self
        else:
            tmpp = self.copy()
        #
        x, y = self.xy[:, 0], self.xy[:, 1]
        new_x = x*np.cos(angle) - y*np.sin(angle)
        new_y = y*np.cos(angle) + x*np.sin(angle)
        tmpp.xy[:, 0] = new_x
        tmpp.xy[:, 1] = new_y
        # Return
        return tmpp


    def remove_doublons(self, method='average', inplace=False, eps_rel=1e-6):
        """
        Replace values associated to the same 'v' by their average.

        Parameters
        ----------
        method : string in {'average', 'max', 'min'}
           Method used to remove the doublons.
        """
        if len(self.v) == 0:
            raise Exception()
        if inplace:
            tmp_pt = self
        else:
            tmp_pt = self.copy()
        vs = tmp_pt.v
        ord_magn = (np.sum(vs**2)/len(vs))**.5
        nmb_dec = -int(round(np.log10(ord_magn*eps_rel)))
        tmp_vs = np.round(vs, decimals=nmb_dec)
        new_vs = np.sort(list(set(tmp_vs)))
        if method == 'average':
            new_xy = [np.mean(tmp_pt.xy[tmp_vs == vi], axis=0)
                      for vi in new_vs]
        elif method == 'min':
            new_xy = [np.min(tmp_pt.xy[tmp_vs == vi], axis=0)
                      for vi in new_vs]
        elif method == 'max':
            new_xy = [np.max(tmp_pt.xy[tmp_vs == vi], axis=0)
                      for vi in new_vs]
        else:
            raise ValueError()
        tmp_pt.xy = new_xy
        tmp_pt.v = new_vs
        # returning
        if not inplace:
            return tmp_pt

    def reverse(self):
        """
        Return a Points object where x and y axis are swaped.
        """
        tmp_pt = self.copy()
        xy_tmp = tmp_pt.xy*0
        xy_tmp[:, 0] = tmp_pt.xy[:, 1]
        xy_tmp[:, 1] = tmp_pt.xy[:, 0]
        tmp_pt.xy = xy_tmp
        return tmp_pt

    def decompose(self):
        """
        return a tuple of Points object, with only one point per object.
        """
        if len(self) == 1:
            return [self]
        if len(self) != len(self.v):
            raise Exception()
        pts_tupl = []
        for i in np.arange(len(self)):
            pts_tupl.append(Points([self.xy[i]], [self.v[i]], self.unit_x,
                                   self.unit_y, self.unit_v, self.name))
        return pts_tupl

    def sort(self, ref='x', inplace=False):
        """
        Sort the points according to the reference.

        Parameters
        ----------
        ref : string or array of indice
            can be 'x', 'y' or 'v' to sort according those values or an
            array of indice
        inplace ; boolean
            If 'True', sort in place, else, return an new sorted instance.
        """
        # check parameters
        if isinstance(ref, STRINGTYPES):
            if ref not in ['x', 'y', 'v']:
                raise ValueError
        elif isinstance(ref, ARRAYTYPES):
            ref = np.array(ref, dtype=int)
            if len(ref) != len(self.xy):
                raise ValueError()
        else:
            raise TypeError()
        if not isinstance(inplace, bool):
            raise TypeError()
        # get order
        if ref == 'x':
            order = np.argsort(self.xy[:, 0])
        elif ref == 'y':
            order = np.argsort(self.xy[:, 1])
        elif ref == 'v':
            if len(self.v) == 0:
                raise ValueError()
            order = np.argsort(self.v)
        else:
            order = ref
        # reordering
        if inplace:
            tmp_pt = self
        else:
            tmp_pt = self.copy()
        tmp_pt.xy = tmp_pt.xy[order]
        if len(tmp_pt.v) == len(tmp_pt.xy):
            tmp_pt.v = tmp_pt.v[order]
        # returning
        if not inplace:
            return tmp_pt

    def order_on_line(self, inplace=False):
        """
        Re-order the set of points to try to create a line.
        """
        if inplace:
            tmp_pts = self
        else:
            tmp_pts = self.copy()
        #
        from sklearn.neighbors import NearestNeighbors
        clf = NearestNeighbors(2).fit(self.xy)
        G = clf.kneighbors_graph()
        import networkx as nx
        T = nx.from_scipy_sparse_matrix(G)
        order = list(nx.dfs_preorder_nodes(T, 0))
        paths = [list(nx.dfs_preorder_nodes(T, i)) for i in range(len(self.xy))]
        mindist = np.inf
        minidx = 0
        for i in range(len(self.xy)):
            p = paths[i]           # order of nodes
            ordered = self.xy[p]    # ordered nodes
            # find cost of that order by the sum of euclidean distances between points (i) and (i+1)
            cost = (((ordered[:-1] - ordered[1:])**2).sum(1)).sum()
            if cost < mindist:
                mindist = cost
                minidx = i
        opt_order = paths[minidx]
        new_x = self.xy[:, 0][opt_order]
        new_y = self.xy[:, 1][opt_order]
        # return
        tmp_pts.xy = np.array(list(zip(new_x, new_y)))
        return tmp_pts

    def remove_nans(self, inplace=False):
        """
        Remove the points containing nans values.
        """
        if inplace:
            tmp_pts = self
        else:
            tmp_pts = self.copy()
        # get indices to remove
        inds = np.logical_or(np.isnan(tmp_pts.xy[:, 0]),
                             np.isnan(tmp_pts.xy[:, 1]))
        inds = np.logical_or(inds, np.isnan(tmp_pts.v))
        # remove
        for ind in np.where(inds)[0][::-1]:
            tmp_pts.remove(ind)
        # return
        if not inplace:
            return tmp_pts

    def augment_resolution(self, fact=2, interp='linear', inplace=False):
        """
        Augment the temporal resolution of the points.
        Only have sense if points are sorted to set some kind of trajectory.

        Parameters
        ----------
        fact : integer
            Resolution augmentation needed (default is '2', for a result
            profile with twice more points)
        interp : string in ['linear', 'nearest', slinear', 'quadratic, 'cubic']
            Specifies the kind of interpolation as a string
            (Default is 'linear'). slinear', 'quadratic' and 'cubic' refer
            to a spline interpolation of first, second or third order.
        inplace bool
            .

        Note
        ----
        If masked values are present, they are interpolated as well, using the
        surrounding values.
        """
        if inplace:
            tmp_pts = self
        else:
            tmp_pts = self.copy()
        is_v = len(self.v) == len(self.xy)
        # get and interpolate
        from .profile import Profile
        tmp_x = Profile(list(range(len(self.xy))), self.xy[:, 0])
        tmp_x.augment_resolution(fact=fact, interp=interp, inplace=True)
        tmp_y = Profile(list(range(len(self.xy))), self.xy[:, 1])
        tmp_y.augment_resolution(fact=fact, interp=interp, inplace=True)
        if is_v:
            tmp_v = Profile(list(range(len(self.xy))), self.v)
            tmp_v.augment_resolution(fact=fact, interp=interp, inplace=True)
        # store
        tmp_pts.xy = list(zip(tmp_x.y, tmp_y.y))
        if is_v:
            tmp_pts.v = tmp_v.y
        if not inplace:
            return tmp_pts

    def smooth(self, tos='gaussian', size=None, inplace=False, **kw):
        """
        Return a smoothed points field.

        Parameters
        ----------
        tos : string, optional
            Type of smoothing, can be 'uniform' (default), 'gaussian'
            or 'lowpass'.
        size : number, optional
            radius of the smoothing for 'uniform',
            radius of the smoothing for 'gaussian',
            cut off frequency for 'lowpass'
            Default are 3 for 'uniform',  1 for 'gaussian' and 0.1 for '
            lowpass'.
        inplace : boolean
            If 'False', return a smoothed points field
            else, smooth in place.
        kw : dic
            Additional parameters for ndimage methods
            (See ndimage documentation)
        """
        if not isinstance(tos, STRINGTYPES):
            raise TypeError("'tos' must be a string")
        if size is None and tos == 'uniform':
            size = 3
        elif size is None and tos == 'gaussian':
            size = 1
        elif size is None and tos == 'lowpass':
            size = 0.1
        # default smoothing border mode to 'nearest'
        if tos in ['uniform', 'gaussian'] and 'mode' not in list(kw.keys()):
            kw.update({'mode': 'nearest'})
        # getting data
        if not inplace:
            tmp_pts = self.copy()
        y = self.xy[:, 1]
        x = self.xy[:, 0]
        if len(self.v) != 0:
            v = self.v
        # smoothing
        if tos == "uniform":
            x = ndimage.uniform_filter(x, size, **kw)
            y = ndimage.uniform_filter(y, size, **kw)
            if len(self.v) != 0:
                v = ndimage.uniform_filter(v, size, **kw)
        elif tos == "gaussian":
            x = ndimage.gaussian_filter(x, size, **kw)
            y = ndimage.gaussian_filter(y, size, **kw)
            if len(self.v) != 0:
                v = ndimage.gaussian_filter(v, size, **kw)
        elif tos == 'lowpass':
            from scipy import signal
            x = self.xy[:, 0]
            y = self.xy[:, 1]
            N = 2
            Wn = size
            B, A = signal.butter(N, Wn, output='ba')
            x = signal.filtfilt(B, A, x)
            y = signal.filtfilt(B, A, y)
            if len(self.v) != 0:
                v = signal.filtfilt(B, A, v)

        else:
            raise ValueError("'tos' must be 'uniform', 'gaussian' or "
                             "'lowpass'")
        # storing
        if inplace:
            self.xy[:, 0] = x
            self.xy[:, 1] = y
            if len(self.v) != 0:
                self.v = v
        else:
            tmp_pts.xy[:, 0] = x
            tmp_pts.xy[:, 1] = y
            if len(tmp_pts.v) != 0:
                tmp_pts.v = v
            return tmp_pts

    def _display(self, kind=None, axe_x=None, axe_y=None, axe_color=None,
                 **plotargs):
        if len(self.xy) == 0:
            return None
        # x values
        if axe_x == 'x' or axe_x is None:
            x_values = self.xy[:, 0]
        elif axe_x == 'y':
            x_values = self.xy[:, 1]
        elif axe_x == 'v':
            if len(self.v) != len(self.xy):
                raise Exception()
            x_values = self.v
        else:
            raise ValueError()
        # y values
        if axe_y == 'x':
            y_values = self.xy[:, 0]
        elif axe_y == 'y' or axe_y is None:
            y_values = self.xy[:, 1]
        elif axe_y == 'v':
            if len(self.v) != len(self.xy):
                raise Exception()
            y_values = self.v
        else:
            raise ValueError()
        # color values
        if axe_color == 'x':
            color_values = self.xy[:, 0]
        elif axe_color == 'y':
            color_values = self.xy[:, 1]
        elif axe_color in [None, 'v']:
            if len(self.v) != len(self.xy):
                color_values = None
                if ('c' not in plotargs.keys()
                    and 'color' not in plotargs.keys()):
                    plotargs['color'] = 'k'
            color_values = self.v
        else:
            raise ValueError()
        # check c and color
        if 'c' in plotargs.keys():
            plotargs['color'] = None
        if 'color' in plotargs.keys():
            plotargs['c'] = None
        # display
        dp = pplt.Displayer(x=x_values, y=y_values, values=color_values,
                            kind=kind, **plotargs)
        ax = dp.draw()
        return ax

    def display(self, kind=None, axe_x=None, axe_y=None, axe_color=None,
                **plotargs):
        """
        Display the set of points.

        Parameters
        ----------
        kind : string, optional
            Can be 'plot' (default if points have not values).
            or 'scatter' (default if points have values).
            or 'colored_plot'.
        axe_x, axe_y, axe_color : strings in ['x', 'y', 'v']
            To determine wich value has to be plotted along which axis, and
            whith value is used to color the scattered points.
            Default plot 'y' to 'x' with colors from 'v'.
        **plotargs : dic
            Additionnal arguments sent to 'plot' or 'scatter'
        """
        # default values
        if axe_x is None:
            if axe_y != 'x':
                axe_x = 'x'
            else:
                axe_x = 'y'
        if axe_y is None:
            if axe_x != 'y':
                axe_y = 'y'
            else:
                axe_y = 'x'
        if axe_color is None:
            axes = ['x', 'y', 'v']
            try:
                axes.remove(axe_x)
                axes.remove(axe_y)
            except ValueError:
                axes = ['v']
            axe_color = axes[0]

        # display the values
        plot = self._display(kind, axe_x=axe_x, axe_y=axe_y,
                             axe_color=axe_color, **plotargs)
        if len(self.v) != 0 and kind is not 'plot':
            cb = plt.colorbar(plot)
            cb.set_label(self.unit_v.strUnit())
            # cb label
            if axe_color == 'x':
                cb.set_label('X ' + self.unit_x.strUnit())
            elif axe_color == 'y':
                cb.set_label('Y ' + self.unit_y.strUnit())
            else:
                cb.set_label('V ' + self.unit_v.strUnit())
        # x axis label
        if axe_x == 'x':
            plt.xlabel('X ' + self.unit_x.strUnit())
        elif axe_x == 'y':
            plt.xlabel('Y ' + self.unit_y.strUnit())
        else:
            plt.xlabel('V ' + self.unit_v.strUnit())
        # y axis label
        if axe_y == 'x':
            plt.ylabel('X ' + self.unit_x.strUnit())
        elif axe_y == 'y':
            plt.ylabel('Y ' + self.unit_y.strUnit())
        else:
            plt.ylabel('V ' + self.unit_v.strUnit())
        if self.name is None:
            plt.title('Set of points')
        else:
            plt.title(self.name)
        plt.axis('equal')
        return plot

    def display3D(self, kind='plot', xlabel='', ylabel='', zlabel='',
                  title='', **plotargs):
        """
        Display the points on a 3D graph.

        Parameters
        ----------
        kind : string, optional
            Kind of graph to use, can be 'plot' or 'surf'.
        xlabel, ylabel, zlabel : string, optional
            Label fo each axis (respectively 'x', 'y', and 'v')
        title : strin, optional
            Title
        **plotargs :
            Additional parameters feeded to matplotlib
        """
        # create 3D plot
        ax = plt.gca(projection='3d')
        # display data
        if kind == 'plot':
            ax.plot(self.xy[:, 0], self.xy[:, 1], self.v, **plotargs)
        elif kind == 'surf':
            ax.plot_trisurf(self.xy[:, 0], self.xy[:, 1], self.v)
        else:
            raise ValueError()
        # labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        # title
        ax.set_title(title)
        #
        plt.tight_layout()
        return ax

    def export_to_profile(self, axe_x='x', axe_y='y'):
        """
        Export the unsorted point object to a sorted Profile object.

        Parameters
        ----------
        axe_x, axe_y : strings in ['x', 'y', 'v']
            Which value used to construct the profile
        """
        # check
        if axe_x not in ['x', 'y', 'v']:
            raise ValueError()
        if axe_y not in ['x', 'y', 'v']:
            raise ValueError()
        # get data
        if axe_x == 'x':
            x = self.xy[:, 0]
            unit_x = self.unit_x
        elif axe_x == 'y':
            x = self.xy[:, 1]
            unit_x = self.unit_y
        else:
            x = self.v
            unit_x = self.unit_v
        if axe_y == 'x':
            y = self.xy[:, 0]
            unit_y = self.unit_x
        elif axe_y == 'y':
            y = self.xy[:, 1]
            unit_y = self.unit_y
        else:
            y = self.v
            unit_y = self.unit_v
        # construct profile
        from .profile import Profile
        prof = Profile(x=x, y=y, mask=False, unit_x=unit_x, unit_y=unit_y)
        return prof
