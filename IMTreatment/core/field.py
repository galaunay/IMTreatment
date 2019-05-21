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

import numpy as np
import unum
import copy
from ..utils.types import ARRAYTYPES, NUMBERTYPES, STRINGTYPES
from ..utils import make_unit


class Field(object):

    def __init__(self):
        self.__axe_x = np.array([], dtype=float)
        self.__axe_y = np.array([], dtype=float)
        self.__unit_x = make_unit('')
        self.__unit_y = make_unit('')
        self.xy_scale = make_unit("")

    def __iter__(self):
        for i, x in enumerate(self.axe_x):
            for j, y in enumerate(self.axe_y):
                yield [i, j], [x, y]

    @property
    def axe_x(self):
        return self.__axe_x

    @axe_x.setter
    def axe_x(self, new_axe_x):
        if not isinstance(new_axe_x, ARRAYTYPES):
            raise TypeError()
        new_axe_x = np.array(new_axe_x, dtype=float)
        # if new_axe_x.shape == self.__axe_x.shape or len(self.__axe_x) == 0:
        self.__axe_x = new_axe_x
        # else:
        #     raise ValueError()

    @axe_x.deleter
    def axe_x(self):
        raise Exception("Nope, can't do that")

    @property
    def dx(self):
        return self.axe_x[1] - self.axe_x[0]

    @property
    def axe_y(self):
        return self.__axe_y

    @axe_y.setter
    def axe_y(self, new_axe_y):
        if not isinstance(new_axe_y, ARRAYTYPES):
            raise TypeError()
        new_axe_y = np.array(new_axe_y, dtype=float)
        # if new_axe_y.shape == self.__axe_y.shape or len(self.__axe_y) == 0:
        self.__axe_y = new_axe_y
        # else:
        #     raise ValueError()

    @axe_y.deleter
    def axe_y(self):
        raise Exception("Nope, can't do that")

    @property
    def dy(self):
        return self.axe_y[1] - self.axe_y[0]

    @property
    def unit_x(self):
        return self.__unit_x

    @unit_x.setter
    def unit_x(self, new_unit_x):
        if isinstance(new_unit_x, unum.Unum):
            if np.isclose(new_unit_x.asNumber(), 1):
                self.__unit_x = new_unit_x
            else:
                raise ValueError()
        elif isinstance(new_unit_x, STRINGTYPES):
            self.__unit_x = make_unit(new_unit_x)
        else:
            raise TypeError()

    @unit_x.deleter
    def unit_x(self):
        raise Exception("Nope, can't do that")

    @property
    def unit_y(self):
        return self.__unit_y

    @unit_y.setter
    def unit_y(self, new_unit_y):
        if isinstance(new_unit_y, unum.Unum):
            if np.isclose(new_unit_y.asNumber(), 1):
                self.__unit_y = new_unit_y
            else:
                raise ValueError()
        elif isinstance(new_unit_y, STRINGTYPES):
            self.__unit_y = make_unit(new_unit_y)
        else:
            raise TypeError()

    @unit_y.deleter
    def unit_y(self):
        raise Exception("Nope, can't do that")

    @property
    def shape(self):
        return self.__axe_x.shape[0], self.__axe_y.shape[0]

    def check_consistency(self):
        """
        Raise errors if the field show some
        incoherences.
        """
        if not isinstance(self.axe_x, np.ndarray):
            raise Exception()
        if not isinstance(self.axe_y, np.ndarray):
            raise Exception()
        if not isinstance(self.unit_x, unum.Unum):
            raise Exception()
        if not isinstance(self.unit_y, unum.Unum):
            raise Exception()
        if not isinstance(self.xy_scale, unum.Unum):
            raise Exception()

    def copy(self):
        """
        Return a copy of the Field object.
        """
        return copy.deepcopy(self)

    def get_indice_on_axe(self, direction, value, kind='bounds'):
        """
        Return, on the given axe, the indices representing the positions
        surrounding 'value'.
        if 'value' is exactly an axe position, return just one indice.

        Parameters
        ----------
        direction : int
            1 or 2, for axes choice.
        value : number
        kind : string
            If 'bounds' (default), return the bounding indices.
            if 'nearest', return the nearest indice
            if 'decimal', return a decimal indice (interpolated)

        Returns
        -------
        interval : 2x1 or 1x1 array of integer
        """
        if not isinstance(direction, NUMBERTYPES):
            raise TypeError("'direction' must be a number.")
        if not (direction == 1 or direction == 2):
            raise ValueError("'direction' must be 1 or 2.")
        if not isinstance(value, NUMBERTYPES):
            raise TypeError("'value' must be a number.")
        if direction == 1:
            axe = self.axe_x
            if value < axe[0] or value > axe[-1]:
                raise ValueError("'value' is out of bound.")
        else:
            axe = self.axe_y
            if value < axe[0] or value > axe[-1]:
                raise ValueError("'value' is out of bound.")
        if not isinstance(kind, STRINGTYPES):
            raise TypeError()
        if kind not in ['bounds', 'nearest', 'decimal']:
            raise ValueError()
        # getting the borning indices
        ind = np.searchsorted(axe, value)
        if axe[ind] == value:
            inds = [ind, ind]
        else:
            inds = [int(ind - 1), int(ind)]
        # returning bounds
        if kind == 'bounds':
            return inds
        # returning nearest
        elif kind == 'nearest':
            if inds[0] == inds[1]:
                return inds[0]
            if np.abs(axe[inds[0]] - value) < np.abs(axe[inds[1]] - value):
                ind = inds[0]
            else:
                ind = inds[1]
            return int(ind)
        # returning decimal
        elif kind == 'decimal':
            if inds[0] == inds[1]:
                return inds[0]
            value_1 = axe[inds[0]]
            value_2 = axe[inds[1]]
            delta = np.abs(value_2 - value_1)
            return (inds[0]*np.abs(value - value_2)/delta +
                    inds[1]*np.abs(value - value_1)/delta)

    def get_points_around(self, center, radius, ind=False):
        """
        Return the list of points or the scalar field that are in a circle
        centered on 'center' and of radius 'radius'.

        Parameters
        ----------
        center : array
            Coordonate of the center point (in axes units).
        radius : float
            radius of the cercle (in axes units).
        ind : boolean, optional
            If 'True', radius and center represent indices on the field.
            if 'False', radius and center are expressed in axis unities.

        Returns
        -------
        indices : array
            Array contening the indices of the contened points.
            [(ind1x, ind1y), (ind2x, ind2y), ...].
            You can easily put them in the axes to obtain points coordinates
        """
        # checking parameters
        if not isinstance(center, ARRAYTYPES):
            raise TypeError("'center' must be an array")
        center = np.array(center, dtype=float)
        if not center.shape == (2,):
            raise ValueError("'center' must be a 2x1 array")
        if not isinstance(radius, NUMBERTYPES):
            raise TypeError("'radius' must be a number")
        if not radius > 0:
            raise ValueError("'radius' must be positive")
        # getting indice data when 'ind=False'
        if not ind:
            dx = self.axe_x[1] - self.axe_x[0]
            dy = self.axe_y[1] - self.axe_y[0]
            delta = (dx + dy)/2.
            radius = radius/delta
            center_x = self.get_indice_on_axe(1, center[0], kind='decimal')
            center_y = self.get_indice_on_axe(2, center[1], kind='decimal')
            center = np.array([center_x, center_y])
        # pre-computing somme properties
        radius2 = radius**2
        radius_int = radius/np.sqrt(2)
        # isolating possibles indices
        inds_x = np.arange(np.int(np.ceil(center[0] - radius)),
                           np.int(np.floor(center[0] + radius)) + 1)
        inds_y = np.arange(np.int(np.ceil(center[1] - radius)),
                           np.int(np.floor(center[1] + radius)) + 1)
        inds_x, inds_y = np.meshgrid(inds_x, inds_y)
        inds_x = inds_x.flatten()
        inds_y = inds_y.flatten()
        # loop on possibles points
        inds = []
        for i in np.arange(len(inds_x)):
            x = inds_x[i]
            y = inds_y[i]
            # test if the point is in the square 'compris' in the cercle
            if x <= center[0] + radius_int \
                    and x >= center[0] - radius_int \
                    and y <= center[1] + radius_int \
                    and y >= center[1] - radius_int:
                inds.append([x, y])
            # test if the point is the center
            elif all([x, y] == center):
                pass
            # test if the point is in the circle
            elif ((x - center[0])**2 + (y - center[1])**2 <= radius2):
                inds.append([x, y])
        return np.array(inds, subok=True)

    def scale(self, scalex=None, scaley=None, inplace=False,
              output_reverse=False):
        """
        Scale the Field.

        Parameters
        ----------
        scalex, scaley : numbers or Unum objects
            Scale for the axis
        inplace : boolean
            .
        """
        if inplace:
            tmp_f = self
        else:
            tmp_f = self.copy()
        # set xy_scale
        if scalex is not None and scaley is not None:
            tmp_f.xy_scale *= scalex/scaley
        elif scalex is not None:
            tmp_f.xy_scale *= scalex
        elif scaley is not None:
            tmp_f.xy_scale *= 1/scaley
        # x
        reversex = False
        if scalex is None:
            pass
        elif isinstance(scalex, NUMBERTYPES):
            tmp_f.axe_x *= scalex
            if scalex < 0:
                reversex = True
        elif isinstance(scalex, unum.Unum):
            new_unit = tmp_f.unit_x * scalex
            fact = new_unit.asNumber()
            new_unit /= fact
            tmp_f.unit_x = new_unit
            tmp_f.axe_x *= fact
            if fact < 0:
                reversex = True
        else:
            raise TypeError()
        if reversex:
            tmp_f.axe_x = tmp_f.axe_x[::-1]
        # y
        reversey = False
        if scaley is None:
            pass
        elif isinstance(scaley, NUMBERTYPES):
            tmp_f.axe_y *= scaley
            if scaley < 0:
                reversey = True
        elif isinstance(scaley, unum.Unum):
            new_unit = tmp_f.unit_y*scaley
            fact = new_unit.asNumber()
            new_unit /= fact
            tmp_f.unit_y = new_unit
            tmp_f.axe_y *= fact
            if fact < 0:
                reversey = True
        else:
            raise TypeError()
        if reversey:
            tmp_f.axe_y = tmp_f.axe_y[::-1]
        # returning
        if output_reverse:
            if inplace:
                return reversex, reversey
            else:
                return tmp_f, reversex, reversey
        else:
            if inplace:
                pass
            else:
                return tmp_f

    def rotate(self, angle, inplace=False):
        """
        Rotate the field.

        Parameters
        ----------
        angle : integer
            Angle in degrees (positive for trigonometric direction).
            In order to preserve the orthogonal grid, only multiples of
            90° are accepted (can be negative multiples).
        inplace : boolean, optional
            If 'True', Field is rotated in place, else, the function return a
            rotated field.

        Returns
        -------
        rotated_field : Field object, optional
            Rotated field.
        """
        # check params
        if not isinstance(angle, NUMBERTYPES):
            raise TypeError()
        if angle % 90 != 0:
            raise ValueError()
        if not isinstance(inplace, bool):
            raise TypeError()
        # get dat
        if inplace:
            tmp_field = self
        else:
            tmp_field = self.copy()
        # normalize angle
        angle = angle % 360
        # rotate
        if angle == 0:
            pass
        elif angle == 90:
            tmp_field.__axe_x, tmp_field.__axe_y \
                = tmp_field.axe_y[::-1], tmp_field.axe_x
            tmp_field.__unit_x, tmp_field.__unit_y \
                = tmp_field.unit_y, tmp_field.unit_x
        elif angle == 180:
            tmp_field.__axe_x, tmp_field.__axe_y \
                = tmp_field.axe_x[::-1], tmp_field.axe_y[::-1]
        elif angle == 270:
            tmp_field.__axe_x, tmp_field.__axe_y \
                = tmp_field.axe_y, tmp_field.axe_x[::-1]
            tmp_field.__unit_x, tmp_field.__unit_y \
                = tmp_field.unit_y, tmp_field.unit_x
        else:
            raise Exception()
        # correction non-crescent axis
        if tmp_field.axe_x[-1] < tmp_field.axe_x[0]:
            tmp_field.__axe_x = -tmp_field.axe_x
        if tmp_field.axe_y[-1] < tmp_field.axe_y[0]:
            tmp_field.__axe_y = -tmp_field.axe_y
        # returning
        if not inplace:
            return tmp_field

    def change_unit(self, axe, new_unit):
        """
        Change the unit of an Field.

        Parameters
        ----------
        axe : string
            'y' for changing the profile y axis unit
            'x' for changing the profile x axis unit
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
            self.unit_x = new_unit/fact
            self.axe_x *= fact
        elif axe == 'y':
            old_unit = self.unit_y
            new_unit = old_unit.asUnit(new_unit)
            fact = new_unit.asNumber()
            self.unit_y = new_unit/fact
            self.axe_y *= fact
        else:
            raise ValueError()

    def set_origin(self, x=None, y=None):
        """
        Modify the axis in order to place the origin at the givev point (x, y)

        Parameters
        ----------
        x : number
        y : number
        """
        if x is not None:
            if not isinstance(x, NUMBERTYPES):
                raise TypeError("'x' must be a number")
            self.axe_x -= x
        if y is not None:
            if not isinstance(y, NUMBERTYPES):
                raise TypeError("'y' must be a number")
            self.axe_y -= y

    def crop(self, intervx=None, intervy=None, full_output=False,
             ind=False, inplace=False):
        """
        Crop the field in respect with given intervals.

        Parameters
        ----------
        intervx : array, optional
            interval wanted along x
        intervy : array, optional
            interval wanted along y
        full_output : boolean, optional
            If 'True', cutting indices are alson returned
        ind : boolean, optional
            If 'True', intervals are understood as indices along axis.
            If 'False' (default), intervals are understood in axis units.
        inplace : boolean, optional
            If 'True', the field is croped in place.
        """
        # default values
        axe_x, axe_y = self.axe_x, self.axe_y
        if intervx is None:
            if ind:
                intervx = [0, len(axe_x)]
            else:
                intervx = [axe_x[0], axe_x[-1]]
        if intervy is None:
            if ind:
                intervy = [0, len(axe_y)]
            else:
                intervy = [axe_y[0], axe_y[-1]]
        # checking parameters
        if not isinstance(intervx, ARRAYTYPES):
            raise TypeError("'intervx' must be an array of two numbers")
        intervx = np.array(intervx, dtype=float)
        if intervx.ndim != 1:
            raise ValueError("'intervx' must be an array of two numbers")
        if intervx.shape != (2,):
            raise ValueError("'intervx' must be an array of two numbers")
        if intervx[0] > intervx[1]:
            raise ValueError("'intervx' values must be crescent")
        if not isinstance(intervy, ARRAYTYPES):
            raise TypeError("'intervy' must be an array of two numbers")
        intervy = np.array(intervy, dtype=float)
        if intervy.ndim != 1:
            raise ValueError("'intervy' must be an array of two numbers")
        if intervy.shape != (2,):
            raise ValueError("'intervy' must be an array of two numbers")
        if intervy[0] > intervy[1]:
            raise ValueError("'intervy' values must be crescent")
        # checking crooping windows
        if ind:
            if intervx[0] < 0 or intervx[1] == 0 or \
                    intervy[0] < 0 or intervy[1] == 0:
                raise ValueError("Invalid cropping window")
        else:
            if np.all(intervx < axe_x[0]) or np.all(intervx > axe_x[-1])\
                    or np.all(intervy < axe_y[0]) \
                    or np.all(intervy > axe_y[-1]):
                raise ValueError("Invalid cropping window")
        # finding interval indices
        if ind:
            indmin_x = int(intervx[0])
            indmax_x = int(intervx[1])
            indmin_y = int(intervy[0])
            indmax_y = int(intervy[1])
        else:
            if intervx[0] <= axe_x[0]:
                indmin_x = 0
            else:
                indmin_x = self.get_indice_on_axe(1, intervx[0])[-1]
            if intervx[1] >= axe_x[-1]:
                indmax_x = len(axe_x) - 1
            else:
                indmax_x = self.get_indice_on_axe(1, intervx[1])[0]
            if intervy[0] <= axe_y[0]:
                indmin_y = 0
            else:
                indmin_y = self.get_indice_on_axe(2, intervy[0])[-1]
            if intervy[1] >= axe_y[-1]:
                indmax_y = len(axe_y) - 1
            else:
                indmax_y = self.get_indice_on_axe(2, intervy[1])[0]
        # cropping the field
        if inplace:
            axe_x = self.axe_x[indmin_x:indmax_x + 1]
            axe_y = self.axe_y[indmin_y:indmax_y + 1]
            self.__axe_x = axe_x
            self.__axe_y = axe_y
            if full_output:
                return indmin_x, indmax_x, indmin_y, indmax_y
        else:
            cropfield = self.copy()
            cropfield.__axe_x = self.axe_x[indmin_x:indmax_x + 1]
            cropfield.__axe_y = self.axe_y[indmin_y:indmax_y + 1]
            if full_output:
                return indmin_x, indmax_x, indmin_y, indmax_y, cropfield
            else:
                return cropfield

    def extend(self, nmb_left=0, nmb_right=0, nmb_up=0, nmb_down=0,
               inplace=False):
        """
        Add columns or lines of masked values at the field.

        Parameters
        ----------
        nmb_**** : integers
            Number of lines/columns to add in each direction.
        inplace : bool
            If 'False', return a new extended field, if 'True', modify the
            field inplace.
        Returns
        -------
        Extended_field : Field object, optional
            Extended field.
        """
        new_axe_x = self.axe_x.copy()
        new_axe_y = self.axe_y.copy()
        if nmb_left != 0:
            dx = self.axe_x[1] - self.axe_x[0]
            x0 = self.axe_x[0]
            new_xs = np.arange(x0-dx*nmb_left, x0, dx)
            new_axe_x = np.concatenate((new_xs, new_axe_x))
        if nmb_right != 0:
            dx = self.axe_x[-1] - self.axe_x[-2]
            x0 = self.axe_x[-1]
            new_xs = np.arange(x0 + dx, x0+dx*(nmb_right + 1), dx)
            new_axe_x = np.concatenate((new_axe_x, new_xs))
        if nmb_down != 0:
            dy = self.axe_y[1] - self.axe_y[0]
            y0 = self.axe_y[0]
            new_ys = np.arange(y0-dy*nmb_down, y0, dy)
            new_axe_y = np.concatenate((new_ys, new_axe_y))
        if nmb_up != 0:
            dy = self.axe_y[-1] - self.axe_y[-2]
            y0 = self.axe_y[-1]
            new_ys = np.arange(y0 + dy, y0+dy*(nmb_up + 1), dy)
            new_axe_y = np.concatenate((new_axe_y, new_ys))
        if inplace:
            self.__axe_x = new_axe_x
            self.__axe_y = new_axe_y
        else:
            fi = self.copy()
            fi.__axe_x = new_axe_x
            fi.__axe_y = new_axe_y
            return fi

    def __clean(self):
        self.__init__()
