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

from .. import plotlib as pplt
import numpy as np
import unum
import copy
from ..utils import make_unit
from ..utils.types import ARRAYTYPES, INTEGERTYPES, NUMBERTYPES, STRINGTYPES
from . import points as pts


class TemporalPoints(object):
    """
    Class representing a set of time evolving points.
    """

    def __init__(self):
        self.point_sets = []
        self.__times = np.array([], dtype=float)
        self.__unit_times = make_unit("")
        self.unit_x = make_unit('')
        self.unit_y = make_unit('')
        self.unit_times = make_unit('')

    def __iter__(self):
        for i in np.arange(len(self.point_sets)):
            yield self.times[i], self.point_sets[i]

    def __len__(self):
        return len(self.point_sets)

    @property
    def unit_x(self):
        return self.__unit_x

    @unit_x.setter
    def unit_x(self, value):
        self.__unit_x = value
        for pt in self.point_sets:
            pt.unit_x = value

    @property
    def unit_y(self):
        return self.__unit_y

    @unit_y.setter
    def unit_y(self, value):
        self.__unit_y = value
        for pt in self.point_sets:
            pt.unit_y = value

    @property
    def times(self):
        return self.__times

    @times.setter
    def times(self, values):
        if not isinstance(values, ARRAYTYPES):
            raise TypeError()
        if len(self.point_sets) != len(values):
            raise ValueError("New number of time ({}) do not corespond to "
                             "the number of fields ({})"
                             .format(len(values), len(self.point_sets)))
        self.__times = values

    @times.deleter
    def times(self):
        raise Exception("Nope, can't do that")

    @property
    def dt(self):
        if len(self.times) > 1:
            return self.times[1] - self.times[0]
        else:
            return None

    @property
    def unit_times(self):
        return self.__unit_times

    @unit_times.setter
    def unit_times(self, new_unit_times):
        if isinstance(new_unit_times, unum.Unum):
            if new_unit_times.asNumber() == 1:
                self.__unit_times = new_unit_times
            else:
                raise ValueError()
        elif isinstance(new_unit_times, STRINGTYPES):
            self.__unit_times == make_unit(new_unit_times)
        else:
            raise TypeError()

    @unit_times.deleter
    def unit_times(self):
        raise Exception("Nope, can't do that")

    def scale(self, scalex=None, scaley=None, scalev=None, scalet=None,
              inplace=False):
        """
        Scale the point sets.

        Parameters
        ----------
        scalex, scaley, scalev : numbers or Unum objects
            Scale for the axis and the values.
        inplace : boolean
            .
        """
        if inplace:
            tmp_ps = self
        else:
            tmp_ps = self.copy()
        for i in range(len(self.point_sets)):
            tmp_ps.point_sets[i].scale(scalex=scalex,
                                       scaley=scaley,
                                       scalev=scalev,
                                       inplace=True)
        # scale the time
        if scalet is None:
            pass
        elif isinstance(scalet, NUMBERTYPES):
            tmp_ps.times *= scalet
        elif isinstance(scalet, unum.Unum):
            new_unit = tmp_ps.unit_times*scalet
            fact = new_unit.asNumber()
            new_unit /= fact
            tmp_ps.unit_times = new_unit
            tmp_ps.times *= fact
        else:
            raise TypeError()
        # returning
        return tmp_ps

    def change_unit(self, axe, new_unit):
        """
        Change the unit of an axe.

        Parameters
        ----------
        axe : string
            'y' for changing the profile y axis unit
            'x' for changing the profile x axis unit
            'values' for changing values unit
            'time' for changing time unit
        new_unit : Unum.unit object or string
            The new unit.
        """
        if isinstance(new_unit, STRINGTYPES):
            new_unit = make_unit(new_unit)
        if not isinstance(new_unit, unum.Unum):
            raise TypeError()
        if not isinstance(axe, STRINGTYPES):
            raise TypeError()
        if axe in ['x', 'y', 'values']:
            for pt in self.point_sets:
                pt.change_unit(axe, new_unit)
        elif axe == 'time':
            old_unit = self.unit_times
            new_unit = old_unit.asUnit(new_unit)
            fact = new_unit.asNumber()
            self.times *= fact
            self.unit_times = new_unit/fact
        else:
            raise ValueError()

    def add_pts(self, pt, time=0., unit_times=""):
        """
        Add a Pointsobject to the existing point set.

        Parameters
        ----------
        pt : Point object
            The points to add.
        time : number
            time associated to the field.
        unit_time : Unum object
            time unit.
        """
        if not isinstance(pt, pts.Points):
            raise TypeError()
        if not isinstance(time, NUMBERTYPES):
            raise TypeError("'time' should be a number, not {}"
                            .format(type(time)))
        if isinstance(unit_times, unum.Unum):
            if unit_times.asNumber() != 1:
                raise ValueError()
        elif isinstance(unit_times, STRINGTYPES):
            unit_times = make_unit(unit_times)
        else:
            raise TypeError()
        # if this is the first field
        if len(self.point_sets) == 0:
            self.unit_x = pt.unit_x
            self.unit_y = pt.unit_y
            self.unit_times = unit_times
            self.__times = np.asarray([time], dtype=float)
        # if not
        else:
            # storing time
            time = (time*self.unit_times/unit_times).asNumber()
            self.__times = np.append(self.__times, time)
        # use default constructor
        self.point_sets.append(pt)
        # sorting the field with time
        self.__sort_field_by_time()

    def remove_pts(self, indice):
        """
        Remove points of the existing point set.

        Parameters
        ----------
        indice : integer or list of integers
            Point indice(s) to remove.
        """
        if isinstance(indice, INTEGERTYPES):
            indice = [indice]
        for i in indice:
            self.__times = np.delete(self.times, i)
            self.point_sets = np.delete(self.point_sets, i)

    def crop(self, intervx=None, intervy=None, intervv=None, intervt=None,
             full_output=False, ind=False, inplace=False):
        """
        Return a croped field in respect with given intervals.

        Parameters
        ----------
        intervx : array, optional
            interval wanted along x
        intervy : array, optional
            interval wanted along y
        intervv : array, optional
            interval wanted along v
        intervt : array, optional
            interval wanted along time
        full_output : boolean, optional
            If 'True', cutting indices are also returned
        inplace : boolean, optional
            If 'True', fields are croped in place.
        """
        # check parameters
        if intervt is not None:
            if not isinstance(intervt, ARRAYTYPES):
                raise TypeError()
            intervt = np.array(intervt, dtype=float)
            if intervt.shape != (2, ):
                raise ValueError()
        # get wanted times
        if intervt is not None:
            if not ind:
                if intervt[0] < self.times[0]:
                    ind1 = 0
                elif intervt[0] > self.times[-1]:
                    raise ValueError()
                else:
                    ind1 = np.where(intervt[0] <= self.times)[0][0]
                if intervt[1] > self.times[-1]:
                    ind2 = len(self.times) - 1
                elif intervt[1] < self.times[0]:
                    raise ValueError()
                else:
                    ind2 = np.where(intervt[1] >= self.times)[0][-1]
                intervt = [ind1, ind2]
        # crop
        if inplace:
            croppts = self
        else:
            croppts = self.copy()
        # temporal
        if intervt is not None:
            croppts.point_sets = croppts.point_sets[intervt[0]:intervt[1] + 1]
            croppts.times = croppts.times[intervt[0]:intervt[1] + 1]
        # spatial
        for pt in self.point_sets:
            pt.crop(intervx=intervx, intervy=intervy, intervv=intervv,
                    inplace=True)
        # returning
        return croppts

    def smooth(self, tos='gaussian', size=None, inplace=False, **kw):
        """
        Return a smoothed points field set.

        Parameters
        ----------
        tos : string, optional
            Type of smoothing, can be 'uniform', 'gaussian' (default)
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
        if inplace:
            tmp_pts = self
        else:
            tmp_pts = self.copy()
        for pt in tmp_pts.point_sets:
            pt.smooth(tos=tos, size=size, inplace=True, **kw)
        return tmp_pts

    def copy(self):
        """
        Return a copy of the velocityfields
        """
        return copy.deepcopy(self)

    def __sort_field_by_time(self):
        if len(self.point_sets) in [0, 1]:
            return None
        ind_sort = np.argsort(self.times)
        self.times = self.times[ind_sort]
        self.point_sets = [self.point_sets[i] for i in ind_sort]

    def display(self, sharecb=True, buffer_size=100, cpkw={}):
        """
        Create a windows to display temporals point sets, controlled by
        buttons.

        Parameters
        ----------
        sharecb: boolean
            Do all the vector field serie has to share the same colorbar or
            not.
        buffer_size: number
            Number of displays to keep in memory (faster, but use memory).
        cpkw: dict
            Drawing argument for points

        Display control
        ---------------
        The display can be controlled using the button, but also the keyboard:
        space, right arrow or + : next field
        backspace, left arrow or - : previous field
        up arrow : last field
        down arrow : first field
        number + enter : goto a specific frame
        p : play the animated fields
        number + i : set the animation increment
        number + t : set the animation time interval (ms)
        q : close
        s : save an image
        """
        # get data
        x = [list(pt.xy[:, 0]) for pt in self.point_sets]
        y = [list(pt.xy[:, 1]) for pt in self.point_sets]
        # check if there is data
        if len(np.concatenate(x)) == 0:
            return None
        # default args
        args = {'kind': 'scatter', 'marker': 'o', 'color': 'k'}
        if 'kind' in list(cpkw.keys()):
            if cpkw['kind'] != 'plot':
                args = {}
        args.update(cpkw)
        db = pplt.Displayer(x, y, buffer_size=buffer_size, **args)
        # plot
        if len(self.times) == 1:
            db.draw(0, rescale=False)
        else:
            pplt.ButtonManager(db)
        return db
