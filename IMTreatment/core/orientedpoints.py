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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import unum
import unum.units as units
try:
    units.counts = unum.Unum.unit('counts')
    units.pixel = unum.Unum.unit('pixel')
except:
    pass
from ..utils.types import ARRAYTYPES, NUMBERTYPES, STRINGTYPES
from . import points as pts


class OrientedPoints(pts.Points):
    """
    Class representing a set of points with associated orientations.
    You can use 'make_unit' to provide unities.

    Parameters
    ----------
    xy : nx2 arrays.
        Representing the coordinates of each point of the set (n points).
    orientations : nxdx2 array
        Representing the orientations of each point in the set
        (d orientations for each n points). Can be 'None' if a point have no
        orientation.
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

    def __init__(self, xy=np.empty((0, 2), dtype=float), orientations=[], v=[],
                 unit_x='', unit_y='', unit_v='', name=''):
        # check parameters
        if not isinstance(orientations, ARRAYTYPES):
            raise TypeError()
        orientations = np.array(orientations)
        if len(xy) != 0 and not orientations.ndim == 3:
            raise ValueError("'orientations' must have 3 dimensions, not {}"
                             .format(orientations.ndim))
        if not orientations.shape[0:3:2] != [len(xy), 2]:
            raise ValueError()
        # initialize data
        pts.Points.__init__(self, xy=xy, v=v, unit_x=unit_x, unit_y=unit_y,
                            unit_v=unit_v, name=name)
        self.orientations = orientations

    def __iter__(self):
        for i in np.arange(len(self.xy)):
            pts_iter = pts.Points.__iter__(self)
            pts_value = next(pts_iter)
            if type(pts_value) is tuple:
                yield pts_value + (self.orientations[i],)
            else:
                yield (pts_value,) + (self.orientations[i],)

    def __add__(self, obj):
        if isinstance(obj, pts.Points):
            tmp_pts = pts.Points.__add__(self, obj)
            if len(self.xy) == 0:
                tmp_ori = obj.orientations
            elif len(obj.xy) == 0:
                tmp_ori = self.orientations
            else:
                tmp_ori = np.append(self.orientations, obj.orientations,
                                    axis=0)
            tmp_opts = OrientedPoints()
            tmp_opts.import_from_Points(tmp_pts, tmp_ori)
            return tmp_opts

    @property
    def orientations(self):
        return self.__orientations

    @orientations.setter
    def orientations(self, new_ori):
        if not isinstance(new_ori, ARRAYTYPES):
            raise TypeError()
        new_ori = np.array(new_ori)
        if new_ori.dtype not in NUMBERTYPES:
            raise TypeError()
        if len(self.xy) != 0 and new_ori.shape[0:3:2] != (len(self.xy), 2):
            raise ValueError("'orientations' shape must be (n, d, 2)  (with n"
                             " the number of points ({}) and d the number of "
                             "directions), not {}"
                             .format(len(self.xy), new_ori.shape))
        self.__orientations = new_ori

    def get_streamlines(self, vf, delta=.25, interp='linear',
                        reverse_direction=False):
        """
        Return the streamlines coming from the points, based on the given
        field.

        Parameters
        ----------
        vf : VectorField or velocityField object
            Field on which compute the streamlines
        delta : number, optional
            Spatial discretization of the stream lines,
            relative to a the spatial discretization of the field.
        interp : string, optional
            Used interpolation for streamline computation.
            Can be 'linear'(default) or 'cubic'
        reverse_direction : boolean, optional
            If True, the streamline goes upstream.

        Returns
        -------
        streams : tuple of Points objects
            Each Points object represent a streamline
        """
        # check parameters
        from .field_treatment import get_streamlines
        from . import vectorfield as vfp
        if not isinstance(vf, vfp.VectorField):
            raise TypeError()
        # getting streamlines
        streams = get_streamlines(vf, self.xy, delta=delta, interp=interp,
                                  reverse_direction=reverse_direction)
        return streams

    def get_streamlines_from_orientations(self, vf, delta=.25, interp='linear',
                                          reverse_direction=False):
        """
        Return the streamlines coming from the points orientations, based on
        the given field.

        Parameters
        ----------
        vf : VectorField or velocityField object
            Field on which compute the streamlines
        delta : number, optional
            Spatial discretization of the stream lines,
            relative to a the spatial discretization of the field.
        interp : string, optional
            Used interpolation for streamline computation.
            Can be 'linear'(default) or 'cubic'
        reverse_direction : boolean or tuple of boolean, optional
            If 'False' (default), the streamline goes downstream.
            If 'True', the streamline goes upstream.
            a tuple of booleans can be specified to apply different behaviors
            to the different orientations

        Returns
        -------
        streams : tuple of Points objects
            Each Points object represent a streamline
        """
        # check parameters
        nmb_dir = self.orientations.shape[1]
        from ..field_treatment import get_streamlines
        from . import vectorfield as vfp
        if not isinstance(vf, vfp.VectorField):
            raise TypeError()
        if isinstance(reverse_direction, bool):
            reverse_direction = np.array([reverse_direction]*nmb_dir)
        elif isinstance(reverse_direction, ARRAYTYPES):
            reverse_direction = np.array(reverse_direction)
        else:
            raise TypeError()
        if reverse_direction.shape != (nmb_dir,):
            raise ValueError()
        # get coef
        coefx = (vf.axe_x[1] - vf.axe_x[0])*.25
        coefy = (vf.axe_x[1] - vf.axe_x[0])*.25
        # get streamlines
        streams = []
        # for each points and each directions
        for i, pt in enumerate(self.xy):
            for n in np.arange(nmb_dir):
                if np.all(self.orientations[i, n] == [0, 0]):
                    continue
                # get streamlines
                ev = self.orientations[i, n]
                pt1 = [pt[0] - ev[0]*coefx, pt[1] - ev[1]*coefy]
                pt2 = [pt[0] + ev[0]*coefx, pt[1] + ev[1]*coefy]
                reverse = reverse_direction[n]
                tmp_stream = get_streamlines(vf, [pt1, pt2],
                                             reverse=reverse)
                # if we are out of field
                if tmp_stream in [[], None]:
                    continue
                if not isinstance(tmp_stream, ARRAYTYPES):
                    tmp_stream = [tmp_stream]
                # add the first point
                for st in tmp_stream:
                    st.xy = np.append([pt], st.xy, axis=0)
                # store
                streams += tmp_stream
        # returning
        return streams

    def import_from_Points(self, pts, orientations):
        """
        Import data from a Points object
        """
        self.xy = pts.xy
        self.v = pts.v
        self.unit_x = pts.unit_x
        self.unit_y = pts.unit_y
        self.unit_v = pts.unit_v
        self.name = pts.name
        self.orientations = orientations

    def add(self, pt, orientations, v=None):
        """
        Add a new point.

        Parameters
        ----------
        pt : 2x1 array of numbers
            Point to add.
        orientations : dx2 array
            orientations associated to the points (d orientations)
        v : number, optional
            Value of the point (needed if other points have values).
        """
        pts.Points.add(self, pt, v)
        if len(self.orientations) == 0:
            self.orientations = np.array([orientations])
        else:
            self.orientations = np.append(self.orientations, [orientations],
                                          axis=0)

    def remove(self, ind):
        """
        Remove the point number 'ind' of the points cloud.
        In place.

        Parameters
        ----------
        ind : integer or array of integer
        """
        pts.Points.remove(self, ind)
        self.orientations = np.delete(self.orientations, ind, axis=0)

    def crop(self, intervx=None, intervy=None, intervv=None, inplace=True):
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
        if inplace:
            tmp_pts = self
        else:
            tmp_pts = self.copy()
        # check if sometyhing to do
        if len(tmp_pts.xy) == 0:
            if not inplace:
                return tmp_pts
            else:
                return None
        # crop orientations
        # TODO : not efficient at all
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
        # use inheritance
        super(OrientedPoints, tmp_pts).crop(intervx=intervx, intervy=intervy,
                                            intervv=intervv, inplace=True)
        tmp_pts.orientations = tmp_pts.orientations[~mask, :]
        # returning
        if not inplace:
            return tmp_pts

    def decompose(self):
        """
        Return a tuple of OrientedPoints object, with only one point per
        object.
        """
        if len(self) == 1:
            return [self]
        if len(self) != len(self.v):
            raise Exception()
        pts_tupl = []
        for i in np.arange(len(self)):
            pts_tupl.append(OrientedPoints([self.xy[i]],
                                           [self.orientations[i]],
                                           [self.v[i]], self.unit_x,
                                           self.unit_y, self.unit_v,
                                           self.name))
        return pts_tupl

    def _display(self, kind=None, **plotargs):
        # display like a Points object
        plot = super(OrientedPoints, self)._display(kind=kind, **plotargs)
        # setting color
        if 'color' in list(plotargs.keys()):
            colors = [plotargs.pop('color')]
        else:
            colors = mpl.rcParams['axes.color_cycle']
        # displaying orientation lines
        x_range = plt.xlim()
        Dx = x_range[1] - x_range[0]
        y_range = plt.ylim()
        Dy = y_range[1] - y_range[0]
        coef = np.min([Dx, Dy])/10.
        for i in np.arange(len(self.xy)):
            loc_oris = self.orientations[i]
            if np.all(loc_oris == [[0, 0], [0, 0]]):
                continue
            color = colors[i % len(colors)]
            pt = self.xy[i]
            for ori in loc_oris:
                line_x = [pt[0] - ori[0]*coef, pt[0] + ori[0]*coef]
                line_y = [pt[1] - ori[1]*coef, pt[1] + ori[1]*coef]
                plt.plot(line_x, line_y, color=color)
        return plot
