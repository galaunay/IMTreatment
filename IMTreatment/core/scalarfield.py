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

import copy
import warnings

import scipy.ndimage.measurements as msr
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as spinterp
import unum
from scipy import ndimage

from .. import plotlib as pplt
from . import field as fld, profile as prof, points as pts
from ..utils import make_unit
from ..utils.types import ARRAYTYPES, NUMBERTYPES, STRINGTYPES


class ScalarField(fld.Field):
    """
    Class representing a scalar field (2D field, with one component on each
    point).
    """

    def __init__(self):
        fld.Field.__init__(self)
        self.__values = np.array([])
        self._values_dtype = None
        self.__mask = np.array([], dtype=bool)
        self.__unit_values = make_unit("")

    def __eq__(self, another):
        if not isinstance(another, ScalarField):
            return False
        if not np.allclose(self.axe_x, another.axe_x):
            return False
        if not np.allclose(self.axe_y, another.axe_y):
            return False
        if not np.all(self.mask == another.mask):
            return False
        if not np.allclose(self.values[~self.mask],
                      another.values[~another.mask]):
            return False
        if not np.all(self.unit_x == another.unit_x):
            return False
        if not np.all(self.unit_y == another.unit_y):
            return False
        if not np.all(self.unit_values == another.unit_values):
            return False
        return True

    def __neg__(self):
        if self._values_dtype == np.uint8:
            raise Exception('Cannot invert unsigned integers')
        tmpsf = self.copy()
        tmpsf.values = -tmpsf.values
        return tmpsf

    def __add__(self, otherone):
        # if we add with a ScalarField object
        if isinstance(otherone, ScalarField):
            # test unities system
            try:
                self.unit_values + otherone.unit_values
                self.unit_x + otherone.unit_x
                self.unit_y + otherone.unit_y
            except:
                raise ValueError("I think these units don't match, fox")
            # identical shape and axis
            if np.all(self.axe_x == otherone.axe_x) and \
                    np.all(self.axe_y == otherone.axe_y):
                tmpsf = self.copy()
                fact = otherone.unit_values/self.unit_values
                tmpsf.values += np.array(otherone.values*fact.asNumber(),
                                         dtype=self._values_dtype)
                tmpsf.mask = np.logical_or(self.mask, otherone.mask)
            # different shape, partially same axis
            else:
                # getting shared points
                new_ind_x = np.array([val in otherone.axe_x
                                      for val in self.axe_x])
                new_ind_y = np.array([val in otherone.axe_y
                                      for val in self.axe_y])
                new_ind_xo = np.array([val in self.axe_x
                                       for val in otherone.axe_x])
                new_ind_yo = np.array([val in self.axe_y
                                       for val in otherone.axe_y])
                if not np.any(new_ind_x) or not np.any(new_ind_y):
                    raise ValueError("Incompatible shapes")
                new_ind_Y, new_ind_X = np.meshgrid(new_ind_y, new_ind_x)
                new_ind_value = np.logical_and(new_ind_X, new_ind_Y)
                new_ind_Yo, new_ind_Xo = np.meshgrid(new_ind_yo, new_ind_xo)
                new_ind_valueo = np.logical_and(new_ind_Xo, new_ind_Yo)
                # getting new axis and values
                new_axe_x = self.axe_x[new_ind_x]
                new_axe_y = self.axe_y[new_ind_y]
                fact = otherone.unit_values/self.unit_values
                new_values = (self.values[new_ind_value] +
                              otherone.values[new_ind_valueo] *
                              fact.asNumber())
                new_values = new_values.reshape((len(new_axe_x),
                                                 len(new_axe_y)))
                new_mask = np.logical_or(self.mask[new_ind_value],
                                         otherone.mask[new_ind_valueo])
                new_mask = new_mask.reshape((len(new_axe_x), len(new_axe_y)))
                # creating sf
                tmpsf = ScalarField()
                tmpsf.import_from_arrays(new_axe_x, new_axe_y, new_values,
                                         mask=new_mask, unit_x=self.unit_x,
                                         unit_y=self.unit_y,
                                         unit_values=self.unit_values)
            return tmpsf
        # if we add with a number
        elif isinstance(otherone, NUMBERTYPES):
            tmpsf = self.copy()
            tmpsf.values += otherone
            return tmpsf
        elif isinstance(otherone, unum.Unum):
            try:
                self.unit_values + otherone
            except:
                raise ValueError("Given number have to be consistent with"
                                 "the scalar field (same units)")
            tmpsf = self.copy()
            tmpsf.values += (otherone/self.unit_values).asNumber()
            return tmpsf
        else:
            raise TypeError("You can only add a scalarfield "
                            "with others scalarfields or with numbers")

    def __radd__(self, obj):
        return self.__add__(obj)

    def __sub__(self, obj):
        return self.__add__(-obj)

    def __rsub__(self, obj):
        return self.__neg__() + obj

    def __truediv__(self, obj):
        if isinstance(obj, NUMBERTYPES):
            tmpsf = self.copy()
            tmpsf.values /= obj
            return tmpsf
        elif isinstance(obj, unum.Unum):
            tmpsf = self.copy()
            unit_values = tmpsf.unit_values / obj
            tmpsf.values *= unit_values.asNumber()
            unit_values /= unit_values.asNumber()
            tmpsf.unit_values = unit_values
            return tmpsf
        elif isinstance(obj, ARRAYTYPES):
            obj = np.array(obj, subok=True)
            if not obj.shape == self.shape:
                raise ValueError()
            tmpsf = self.copy()
            mask = np.logical_or(self.mask, obj == 0)
            not_mask = np.logical_not(mask)
            tmpsf.values[not_mask] = tmpsf.values[not_mask] / obj[not_mask]
            tmpsf.mask = mask
            return tmpsf
        elif isinstance(obj, ScalarField):
            if np.any(self.axe_x != obj.axe_x)\
                    or np.any(self.axe_y != obj.axe_y)\
                    or self.unit_x != obj.unit_x\
                    or self.unit_y != obj.unit_y:
                raise ValueError("Fields are not consistent")
            tmpsf = self.copy()
            filt_nan = obj.values != 0
            values = np.zeros(shape=self.values.shape)
            values[filt_nan] = self.values[filt_nan]/obj.values[filt_nan]
            mask = np.logical_or(self.mask, obj.mask)
            mask = np.logical_or(mask, np.logical_not(filt_nan))
            unit = self.unit_values / obj.unit_values
            tmpsf.values = values*unit.asNumber()
            tmpsf.mask = mask
            tmpsf.unit_values = unit/unit.asNumber()
            return tmpsf
        else:
            raise TypeError("Unsupported operation between {} and a "
                            "ScalarField object".format(type(obj)))

    __div__ = __truediv__

    def __rtruediv__(self, obj):
        if isinstance(obj, NUMBERTYPES):
            tmpsf = self.copy()
            tmpsf.values = obj/tmpsf.values
            tmpsf.unit_values = 1/tmpsf.unit_values
            return tmpsf
        elif isinstance(obj, unum.Unum):
            tmpsf = self.copy()
            tmpsf.values = obj.asNumber()/tmpsf.values
            tmpsf.unit_values = obj/obj.asNumber()/tmpsf.unit_values
            return tmpsf
        elif isinstance(obj, ARRAYTYPES):
            obj = np.array(obj, subok=True)
            if not obj.shape == self.shape:
                raise ValueError()
            tmpsf = self.copy()
            mask = np.logical_or(self.mask, obj == 0)
            not_mask = np.logical_not(mask)
            tmpsf.values[not_mask] = obj[not_mask] / tmpsf.values[not_mask]
            tmpsf.mask = mask
            return tmpsf
        elif isinstance(obj, ScalarField):
            if np.any(self.axe_x != obj.axe_x)\
                    or np.any(self.axe_y != obj.axe_y)\
                    or self.unit_x != obj.unit_x\
                    or self.unit_y != obj.unit_y:
                raise ValueError("Fields are not consistent")
            tmpsf = self.copy()
            values = obj.values / self.values
            mask = np.logical_or(self.mask, obj.mask)
            unit = obj.unit_values / self.unit_values
            tmpsf.values = values*unit.asNumber()
            tmpsf.mask = mask
            tmpsf.unit_values = unit/unit.asNumber()
            return tmpsf
        else:
            raise TypeError("Unsupported operation between {} and a "
                            "ScalarField object".format(type(obj)))

    def __mul__(self, obj):
        if isinstance(obj, NUMBERTYPES):
            tmpsf = self.copy()
            tmpsf.values *= obj
            tmpsf.mask = self.mask
            return tmpsf
        elif isinstance(obj, unum.Unum):
            tmpsf = self.copy()
            tmpsf.values *= obj.asNumber()
            tmpsf.unit_values *= obj/obj.asNumber()
            tmpsf.mask = self.mask
            return tmpsf
        elif isinstance(obj, ARRAYTYPES):
            obj = np.array(obj, subok=True)
            if not obj.shape == self.shape:
                raise ValueError()
            tmpsf = self.copy()
            mask = self.mask
            not_mask = np.logical_not(mask)
            tmpsf.values[not_mask] *= obj[not_mask]
            tmpsf.mask = mask
            return tmpsf
        elif isinstance(obj, np.ma.MaskedArray):
            if obj.shape != self.values.shape:
                raise ValueError("Fields are not consistent")
            tmpsf = self.copy()
            tmpsf.values *= obj
            tmpsf.mask = obj.mask
            return tmpsf
        elif isinstance(obj, ScalarField):
            if np.any(self.axe_x != obj.axe_x)\
                    or np.any(self.axe_y != obj.axe_y)\
                    or self.unit_x != obj.unit_x\
                    or self.unit_y != obj.unit_y:
                raise ValueError("Fields are not consistent")
            tmpsf = self.copy()
            values = self.values * obj.values
            mask = np.logical_or(self.mask, obj.mask)
            unit = self.unit_values * obj.unit_values
            tmpsf.values = values*unit.asNumber()
            tmpsf.mask = mask
            tmpsf.unit_values = unit/unit.asNumber()
            return tmpsf
        else:
            raise TypeError("Unsupported operation between {} and a "
                            "ScalarField object".format(type(obj)))
    __rmul__ = __mul__

    def __abs__(self):
        tmpsf = self.copy()
        tmpsf.values = np.abs(tmpsf.values)
        return tmpsf

    def __pow__(self, number):
        if not isinstance(number, NUMBERTYPES):
            raise TypeError("You only can use a number for the power "
                            "on a Scalar field")
        tmpsf = self.copy()
        tmpsf.values[np.logical_not(tmpsf.mask)] \
            = np.power(tmpsf.values[np.logical_not(tmpsf.mask)], number)
        tmpsf.mask = self.mask
        tmpsf.unit_values = np.power(tmpsf.unit_values, number)
        return tmpsf

    def __iter__(self):
        data = self.values
        mask = self.mask
        for ij, xy in fld.Field.__iter__(self):
            i = ij[0]
            j = ij[1]
            if not mask[i, j]:
                yield ij, xy, data[i, j]

    @property
    def values(self):
        val = self.__values
        try:
            val[self.mask] = np.nan
        except ValueError:
            val[self.mask] = 0
        return val

    @values.setter
    def values(self, new_values):
        if not isinstance(new_values, ARRAYTYPES):
            raise TypeError()
        new_values = np.asarray(new_values)
        if self.shape == new_values.shape:
            # adapting mask to 'nan' values
            self.__mask = np.isnan(new_values)
            # storing data
            self.__values = new_values
        else:
            raise ValueError("'values' should have the same shape as the "
                             "original values : {}, not {}."
                             .format(self.shape, new_values.shape))

    @values.deleter
    def values(self):
        raise Exception("Nope, can't do that")

    @property
    def mask(self):
        return self.__mask

    @mask.setter
    def mask(self, new_mask):
        # check 'new_mask' coherence
        if isinstance(new_mask, (bool, np.bool_)):
            fill_value = new_mask
            new_mask = np.empty(self.shape, dtype=bool)
            new_mask.fill(fill_value)
        elif isinstance(new_mask, ARRAYTYPES):
            new_mask = np.asarray(new_mask, dtype=bool)
        else:
            raise TypeError("'mask' should be an array or a boolean,"
                            " not a {}".format(type(new_mask)))
        if self.shape != new_mask.shape:
            raise ValueError()
        # # check if the new mask don'r reveal masked values
        # if np.any(np.logical_not(new_mask[self.mask])):
        #     raise Warning("This mask reveal masked values, maybe you should"
        #                   "use the 'fill' function instead")
        # # nanify masked values
        # self.values[new_mask] = np.NaN
        # store mask
        self.__mask = new_mask

    @mask.deleter
    def mask(self):
        raise Exception("Nope, can't do that")

    @property
    def mask_as_sf(self):
        tmp_sf = ScalarField()
        tmp_sf.import_from_arrays(self.axe_x, self.axe_y, self.mask,
                                  mask=False, unit_x=self.unit_x,
                                  unit_y=self.unit_y,
                                  unit_values=self.unit_values)
        return tmp_sf

    @property
    def unit_values(self):
        return self.__unit_values

    @unit_values.setter
    def unit_values(self, new_unit_values):
        if isinstance(new_unit_values, unum.Unum):
            if (new_unit_values.asNumber() - 1.) < 1e-10:
                self.__unit_values = new_unit_values
            else:
                raise ValueError()
        elif isinstance(new_unit_values, STRINGTYPES):
            self.__unit_values = make_unit(new_unit_values)
        else:
            raise TypeError()

    @unit_values.deleter
    def unit_values(self):
        raise Exception("Nope, can't do that")

    @property
    def min(self):
        return np.min(self.values[np.logical_not(self.mask)])

    @property
    def max(self):
        return np.max(self.values[np.logical_not(self.mask)])

    @property
    def mean(self):
        return np.mean(self.values[np.logical_not(self.mask)])

    @property
    def median(self):
        return np.median(self.values[np.logical_not(self.mask)])

    def import_from_arrays(self, axe_x, axe_y, values, mask=None,
                           unit_x="", unit_y="", unit_values="",
                           dtype=np.float, dontchecknans=False,
                           dontcheckunits=False):
        """
        Set the field from a set of arrays.

        Parameters
        ----------
        axe_x : array
            Discretized axis value along x
        axe_y : array
            Discretized axis value along y
        values : array or masked array
            Values of the field at the discritized points
        unit_x : String unit, optionnal
            Unit for the values of axe_x
        unit_y : String unit, optionnal
            Unit for the values of axe_y
        unit_values : String unit, optionnal
            Unit for the scalar field
        dontchecknans: boolean
            Don't check for nans values (faster)
        dontcheckunits: boolean
            Don't check for unit consistency (faster)
        dtype: Numerical type
            Numerical type to store the data to.
            Should be a type supported by numpy arrays.
            Default to 'float'.

        """
        # # overwrite previous
        # self.__clean()
        # Use numpy arrays
        self._values_dtype = dtype
        axe_x = np.array(axe_x, dtype=float)
        axe_y = np.array(axe_y, dtype=float)
        values = np.array(values, dtype=self._values_dtype)
        if mask is None:
            mask = np.zeros(values.shape, dtype=bool)
        elif isinstance(mask, bool):
            nmask = np.empty(values.shape, dtype=bool)
            nmask.fill(mask)
            mask = nmask
        else:
            mask = np.array(mask, dtype=bool)
        # Be sure nan values are masked
        if not dontchecknans:
            if np.isnan(np.sum(values)):
                mask = np.logical_or(mask, np.isnan(values))
        # Be sure axes are one-dimensional
        if axe_x.ndim >= 2:
            if np.all(axe_x[0, 0] == axe_x[:, 0]):
                axe_x = axe_x[0]
            else:
                axe_x = axe_x[:, 0]
        if axe_y.ndim >= 2:
            if np.all(axe_y[0, 0] == axe_y[:, 0]):
                axe_y = axe_y[0]
            else:
                axe_y = axe_y[:, 0]
        # Check if axis are evenly spaced
        delta_x = axe_x[1:] - axe_x[:-1]
        delta_y = axe_y[1:] - axe_y[:-1]
        epsx_abs = delta_x*1e-6
        epsy_abs = delta_y*1e-6
        if np.any(delta_y - delta_y[0] > epsy_abs) or \
           np.any(delta_x - delta_x[0] > epsx_abs):
            warnings.warn("Axis are not evenly spaced.\n"
                          "Consider using 'make_evenly_spaced' method"
                          " to avoid bad surprises.")
        # Be sure values is of the good shape
        if len(axe_x) == values.shape[1] and \
           len(axe_y) == values.shape[0] and \
           len(axe_x) != len(axe_y):
            values = values.transpose()
            try:
                mask = mask.transpose()
            except:
                pass
        # Be sure axes are crescent
        if axe_x[0] > axe_x[-1]:
            axe_x = axe_x[::-1]
            values = values[::-1]
        if axe_y[0] > axe_y[-1]:
            axe_y = axe_y[::-1]
            values = values[:, ::-1]
        # storing datas
        self.axe_x = axe_x
        self.axe_y = axe_y
        self.__values = values
        self.__mask = mask
        if dontcheckunits:
            self._Field__unit_x = unit_x
            self._Field__unit_y = unit_y
        else:
            self.unit_x = unit_x
            self.unit_y = unit_y
        self.unit_values = unit_values

    def get_props(self):
        """
        Print the ScalarField main properties
        """
        text = "Shape: {}".format(self.shape)
        unit_x = self.unit_x.strUnit()
        text += "Axe x: [{}..{}]{}".format(self.axe_x[0], self.axe_x[-1],
                                           unit_x)
        unit_y = self.unit_y.strUnit()
        text += "Axe y: [{}..{}]{}".format(self.axe_y[0], self.axe_y[-1],
                                           unit_y)
        unit_values = self.unit_values.strUnit()
        text += "Values: [{}..{}]{}".format(self.min, self.max, unit_values)
        nmb_mask = np.sum(self.mask)
        nmb_tot = self.shape[0]*self.shape[1]
        text += "Masked values: {}/{}".format(nmb_mask, nmb_tot)
        return text

    def get_value(self, x, y, ind=False, unit=False):
        """
        Return the scalar field value on the point (x, y).
        If ind is true, x and y are indices,
        else, x and y are value on axes (interpolated if necessary).
        """
        if not isinstance(ind, bool):
            raise TypeError("'ind' must be a boolean")
        if ind:
            if not isinstance(x, int) or not isinstance(y, int):
                raise TypeError("'x' and 'y' must be integers")
            if x > len(self.axe_x) - 1 or y > len(self.axe_y) - 1\
                    or x < 0 or y < 0:
                raise ValueError("'x' and 'y' must be correct indices")
        else:
            if not isinstance(x, NUMBERTYPES)\
                    or not isinstance(y, NUMBERTYPES):
                raise TypeError("'x' and 'y' must be numbers")
            if x > self.axe_x[-1] or y > self.axe_y[-1]\
                    or x < self.axe_x[0] or y < self.axe_y[0]:
                raise ValueError("'x' and 'y' are out of axes")
        if unit:
            unit = self.unit_values
        else:
            unit = 1.
        if ind:
            return self.values[x, y]*unit
        else:
            ind_x = None
            ind_y = None
            # getting indices interval
            inds_x = self.get_indice_on_axe(1, x)
            inds_y = self.get_indice_on_axe(2, y)
            # if something masked
            if np.sum(self.mask[inds_x, inds_y]) != 0:
                res = np.NaN
            # if we are on a grid point
            elif inds_x[0] == inds_x[1] and inds_y[0] == inds_y[1]:
                res = self.values[inds_x[0], inds_y[0]]*unit
            # if we are on a x grid branch
            elif inds_x[0] == inds_x[1]:
                ind_x = inds_x[0]
                pos_y1 = self.axe_y[inds_y[0]]
                pos_y2 = self.axe_y[inds_y[1]]
                value1 = self.values[ind_x, inds_y[0]]
                value2 = self.values[ind_x, inds_y[1]]
                i_value = ((value2*np.abs(pos_y1 - y) +
                           value1*np.abs(pos_y2 - y)) /
                           np.abs(pos_y1 - pos_y2))
                res = i_value*unit
            # if we are on a y grid branch
            elif inds_y[0] == inds_y[1]:
                ind_y = inds_y[0]
                pos_x1 = self.axe_x[inds_x[0]]
                pos_x2 = self.axe_x[inds_x[1]]
                value1 = self.values[inds_x[0], ind_y]
                value2 = self.values[inds_x[1], ind_y]
                i_value = ((value2*np.abs(pos_x1 - x) +
                            value1*np.abs(pos_x2 - x)) /
                           np.abs(pos_x1 - pos_x2))
                return i_value*unit
            # if we are in the middle of nowhere (linear interpolation)
            else:
                ind_x = inds_x[0]
                ind_y = inds_y[0]
                a, b = np.meshgrid(self.axe_x[ind_x:ind_x + 2],
                                   self.axe_y[ind_y:ind_y + 2], indexing='ij')
                values = self.values[ind_x:ind_x + 2, ind_y:ind_y + 2]
                a = a.flatten()
                b = b.flatten()
                pts = list(zip(a, b))
                interp_vx = spinterp.LinearNDInterpolator(pts,
                                                          values.flatten())
                i_value = float(interp_vx(x, y))
                res = i_value*unit
            return res

    def get_zones_centers(self, bornes=[0.75, 1], rel=True,
                          kind='ponderated'):
        """
        Return a pts.Points object contening centers of the zones
        lying in the given bornes.

        Parameters
        ----------
        bornes : 2x1 array, optionnal
            Trigger values determining the zones.
            '[inferior borne, superior borne]'
        rel : Boolean
            If 'rel' is 'True' (default), values of 'bornes' are relative to
            the extremum values of the field.
            If 'rel' is 'False', values of bornes are treated like absolute
            values.
        kind : string, optional
            if 'kind' is 'center', given points are geometrical centers,
            if 'kind' is 'extremum', given points are
            extrema (min or max) on zones
            if 'kind' is 'ponderated'(default, given points are centers of
            mass, ponderated by the scaler field.

        Returns
        -------
        pts : pts.Points object
            Contening the centers coordinates
        """
        # correcting python's problem with egality...
        bornes[0] -= 0.00001*abs(bornes[0])
        bornes[1] += 0.00001*abs(bornes[1])
        # checking parameters coherence
        if not isinstance(bornes, ARRAYTYPES):
            raise TypeError("'bornes' must be an array")
        if not isinstance(bornes, np.ndarray):
            bornes = np.array(bornes, dtype=float)
        if not bornes.shape == (2,):
            raise ValueError("'bornes' must be a 2x1 array")
        if bornes[0] == bornes[1]:
            return None
        if not bornes[0] < bornes[1]:
            raise ValueError("'bornes' must be crescent")
        if not isinstance(rel, bool):
            raise TypeError("'rel' must be a boolean")
        if not isinstance(kind, STRINGTYPES):
            raise TypeError("'kind' must be a string")
        # compute minimum and maximum if 'rel=True'
        if rel:
            if bornes[0]*bornes[1] < 0:
                raise ValueError("In relative 'bornes' must have the same"
                                 " sign")
            mini = self.min
            maxi = self.max
            if np.abs(bornes[0]) > np.abs(bornes[1]):
                bornes[1] = abs(maxi - mini)*bornes[1] + maxi
                bornes[0] = abs(maxi - mini)*bornes[0] + maxi
            else:
                bornes[1] = abs(maxi - mini)*bornes[1] + mini
                bornes[0] = abs(maxi - mini)*bornes[0] + mini
        # check if the zone exist
        else:
            mini = self.min
            maxi = self.max
            if maxi < bornes[0] or mini > bornes[1]:
                return None
        # getting data
        values = self.values
        mask = self.mask
        if np.any(mask):
            warnings.warn("There is masked values, algorithm can give "
                          "strange results")
        # check if there is more than one point superior
        aoi = np.logical_and(values >= bornes[0], values <= bornes[1])
        if np.sum(aoi) == 1:
            inds = np.where(aoi)
            x = self.axe_x[inds[0][0]]
            y = self.axe_y[inds[1][0]]
            return pts.Points([[x, y]], unit_x=self.unit_x,
                              unit_y=self.unit_y)
        zones = np.logical_and(np.logical_and(values >= bornes[0],
                                              values <= bornes[1]),
                               np.logical_not(mask))
        # compute the center with labelzones
        labeledzones, nmbzones = msr.label(zones)
        inds = []
        if kind == 'extremum':
            mins, _, ind_min, ind_max = msr.extrema(values,
                                                    labeledzones,
                                                    np.arange(nmbzones) + 1)
            for i in np.arange(len(mins)):
                if bornes[np.argmax(np.abs(bornes))] < 0:
                    inds.append(ind_min[i])
                else:
                    inds.append(ind_max[i])
        elif kind == 'center':
            inds = msr.center_of_mass(np.ones(self.shape),
                                      labeledzones,
                                      np.arange(nmbzones) + 1)
        elif kind == 'ponderated':
            inds = msr.center_of_mass(np.abs(values), labeledzones,
                                      np.arange(nmbzones) + 1)
        else:
            raise ValueError("Invalid value for 'kind'")
        coords = []
        for ind in inds:
            indx = ind[0]
            indy = ind[1]
            if indx % 1 == 0:
                x = self.axe_x[int(indx)]
            else:
                dx = self.axe_x[1] - self.axe_x[0]
                x = self.axe_x[int(indx)] + dx*(indx % 1)
            if indy % 1 == 0:
                y = self.axe_y[int(indy)]
            else:
                dy = self.axe_y[1] - self.axe_y[0]
                y = self.axe_y[int(indy)] + dy*(indy % 1)
            coords.append([x, y])
        coords = np.array(coords, dtype=float)
        if len(coords) == 0:
            return None
        return pts.Points(coords, unit_x=self.unit_x, unit_y=self.unit_y)

    def get_nearest_extrema(self, pts, extrema='max', ind=False):
        """
        For a given set of points, return the positions of the nearest local
        extrema (minimum or maximum).

        Parameters
        ----------
        pts : Nx2 array
            Set of pts.Points position.

        Returns
        -------
        extremum_pos : Nx2 array
        """
        # get data
        tmp_sf = self.copy()
        tmp_sf.mirroring(direction='x', position=tmp_sf.axe_x[0],
                         inds_to_mirror=1, inplace=True)
        tmp_sf.mirroring(direction='x', position=tmp_sf.axe_x[-1],
                         inds_to_mirror=1, inplace=True)
        tmp_sf.mirroring(direction='y', position=tmp_sf.axe_y[0],
                         inds_to_mirror=1, inplace=True)
        tmp_sf.mirroring(direction='y', position=tmp_sf.axe_y[-1],
                         inds_to_mirror=1, inplace=True)
        dx = tmp_sf.axe_x[1] - tmp_sf.axe_x[0]
        dy = tmp_sf.axe_y[1] - tmp_sf.axe_y[0]
        # get gradient field
        grad_x, grad_y = np.gradient(tmp_sf.values, dx, dy)
        from . import vectorfield as vf
        tmp_vf = vf.VectorField()
        tmp_vf.import_from_arrays(tmp_sf.axe_x, tmp_sf.axe_y, grad_x, grad_y,
                                  unit_x=tmp_sf.unit_x, unit_y=tmp_sf.unit_y,
                                  unit_values=tmp_sf.unit_values)
        # extract the streamline from the gradient field
        from ..field_treatment import get_streamlines
        if extrema == 'min':
            reverse = True
        else:
            reverse = False
        sts = get_streamlines(tmp_vf, pts, reverse=reverse, resolution=0.1)
        # get the final converged points
        extremum_pos = []
        if isinstance(sts, ARRAYTYPES):
            for i, st in enumerate(sts):
                if len(st.xy) == 0:
                    extremum_pos.append(pts[i])
                else:
                    extremum_pos.append(st.xy[-1])
        else:
            extremum_pos.append(sts.xy[-1])
        extremum_pos = np.array(extremum_pos)
        # returning
        return extremum_pos

    def get_profile(self, direction, position, ind=False, interp='linear'):
        """
        Return a profile of the scalar field, at the given position (or at
        least at the nearest possible position).
        If position is an interval, the fonction return an average profile
        in this interval.

        Parameters
        ----------
        direction : string in ['x', 'y']
            Direction along which we choose a position.
        position : float, interval of float or string
            Position, interval in which we want a profile or 'all'
        ind : boolean
            If 'True', position has to be given in indices
            If 'False' (default), position has to be given in axis unit.
        interp : string in ['nearest', 'linear']
            if 'nearest', get the profile at the nearest position on the grid,
            if 'linear', use linear interpolation to get the profile at the
            exact position

        Returns
        -------
        profile : prof.Profile object
            Wanted profile
        """
        # checking parameters
        if direction not in ['x', 'y']:
            raise TypeError("'direction' must be 'x' or 'y'")
        if not isinstance(position, NUMBERTYPES + ARRAYTYPES + STRINGTYPES):
            raise TypeError()
        if isinstance(position, ARRAYTYPES):
            position = np.array(position, dtype=float)
            if not position.shape == (2,):
                raise ValueError("'position' must be a number or an interval")
        elif isinstance(position, STRINGTYPES):
            if position != 'all':
                raise ValueError()
        if not isinstance(ind, bool):
            raise TypeError()
        if not isinstance(interp, STRINGTYPES):
            raise TypeError()
        if interp not in ['nearest', 'linear']:
            raise ValueError()
        # getting data
        if direction == 'x':
            axe = self.axe_x
            unit_x = self.unit_y
            unit_y = self.unit_values
        else:
            axe = self.axe_y
            unit_x = self.unit_x
            unit_y = self.unit_values
        # applying interval type
        if isinstance(position, ARRAYTYPES) and not ind:
            for pos in position:
                if pos > axe.max():
                    pos = axe.max()
                if pos < axe.min():
                    pos = axe.min()
        elif isinstance(position, ARRAYTYPES) and ind:
            if np.min(position) < -len(axe) + 1 or \
               np.max(position) > len(axe) - 1:
                raise ValueError("'position' must be included in"
                                 " the choosen axis values")
        elif isinstance(position, NUMBERTYPES) and not ind:
            if position > axe.max() or position < axe.min():
                raise ValueError("'position' must be included in the choosen"
                                 " axis values (here [{0},{1}])"
                                 .format(axe.min(), axe.max()))
        elif isinstance(position, NUMBERTYPES) and ind:
            if np.min(position) < 0 or np.max(position) > len(axe) - 1:
                raise ValueError("'position' must be included in the choosen"
                                 " axis values (here [{0},{1}])"
                                 .format(0, len(axe) - 1))
        elif position == 'all':
            position = np.array([axe[0], axe[-1]])
        else:
            raise ValueError()
        # if use interpolation
        if isinstance(position, NUMBERTYPES) and interp == 'linear':
            if direction == 'x':
                axe = self.axe_y
                if ind:
                    position = self.axe_x[position]
                vals = [self.get_value(position, axe_i) for axe_i in axe]
                tmp_prof = prof.Profile(x=axe, y=vals, mask=False,
                                        unit_x=self.unit_y,
                                        unit_y=self.unit_values)
            if direction == 'y':
                axe = self.axe_x
                if ind:
                    position = self.axe_y[position]
                vals = [self.get_value(axe_i, position) for axe_i in axe]
                tmp_prof = prof.Profile(x=axe, y=vals, mask=False,
                                        unit_x=self.unit_x,
                                        unit_y=self.unit_values)
            return tmp_prof
        # if not
        if isinstance(position, NUMBERTYPES) and not ind:
            for i in np.arange(1, len(axe)):
                if (axe[i] >= position and axe[i-1] <= position) \
                        or (axe[i] <= position and axe[i-1] >= position):
                    break
            if np.abs(position - axe[i]) > np.abs(position - axe[i-1]):
                finalindice = i-1
            else:
                finalindice = i
            if direction == 'x':
                prof_mask = self.mask[finalindice, :]
                profile = self.values[finalindice, :]
                axe = self.axe_y
            else:
                prof_mask = self.mask[:, finalindice]
                profile = self.values[:, finalindice]
                axe = self.axe_x
        elif isinstance(position, NUMBERTYPES) and ind:
            if direction == 'x':
                prof_mask = self.mask[position, :]
                profile = self.values[position, :]
                axe = self.axe_y
            else:
                prof_mask = self.mask[:, position]
                profile = self.values[:, position]
                axe = self.axe_x
        # Calculation of the profile for an interval of position
        elif isinstance(position, ARRAYTYPES) and not ind:
            axe_mask = np.logical_and(axe >= position[0], axe <= position[1])
            if direction == 'x':
                prof_mask = self.mask[axe_mask, :].mean(0)
                profile = self.values[axe_mask, :].mean(0)
                axe = self.axe_y
            else:
                prof_mask = self.mask[:, axe_mask].mean(1)
                profile = self.values[:, axe_mask].mean(1)
                axe = self.axe_x
        elif isinstance(position, ARRAYTYPES) and ind:
            if direction == 'x':
                prof_mask = self.mask[position[0]:position[1], :].mean(0)
                profile = self.values[position[0]:position[1], :].mean(0)
                axe = self.axe_y
            else:
                prof_mask = self.mask[:, position[0]:position[1]].mean(1)
                profile = self.values[:, position[0]:position[1]].mean(1)
                axe = self.axe_x
        return prof.Profile(axe, profile, prof_mask, unit_x, unit_y)

    def get_histogram(self, cum=False, normalized=False, bins=None,
                      range=None):
        """
        Return the image histogram.

        Parameters
        ==========
        cum: boolean
            If True, get a cumulative histogram.

        Returns
        =======
        hist: array of numbers
            Histogram.
        """
        if bins is None:
            bins = 10
        hist, xs = np.histogram(self.values.flatten(),
                                bins=bins,
                                range=range,
                                density=normalized)
        xs = xs[0:-1] + np.mean(xs[0:2])
        if cum:
            hist = np.cumsum(hist)

        return prof.Profile(xs, hist, mask=False, unit_x=self.unit_values,
                            unit_y="counts")

    def get_spatial_autocorrelation(self, direction, window_len=None):
        """
        Return the spatial auto-correlation along the wanted direction.

        Take the middle point for reference for correlation computation.

        Parameters
        ----------
        direction : string
            'x' or 'y'
        window_len : integer, optional
            Window length for sweep correlation. if 'None' (default), all the
            signal is used, and boundary effect can be seen.

        Returns
        -------
        profile : prof.Profile object
            Spatial correlation
        """
        # Direction X
        if direction == 'x':
            if window_len is None:
                window_len = int(len(self.axe_x) - 2)
            # loop on profiles
            cor = np.zeros(int(np.floor(window_len/2.)*2 + 1))
            nmb = 0
            for i, y in enumerate(self.axe_y):
                tmp_prof = self.get_profile('y', i, ind=True)
                cor += tmp_prof.get_auto_correlation(window_len, raw=True)
                nmb += 1
            cor /= nmb
            # returning
            dx = self.axe_x[1] - self.axe_x[0]
            x = np.arange(0, len(cor)*dx, dx)
            return prof.Profile(x=x, y=cor, unit_x=self.unit_x,
                                unit_y=make_unit(''))
        elif direction == 'y':
            if window_len is None:
                window_len = int(len(self.axe_y) - 2)
            # loop on profiles
            cor = np.zeros(int(np.floor(window_len/2.)*2 + 1))
            nmb = 0
            for i, x in enumerate(self.axe_x):
                tmp_prof = self.get_profile('x', i, ind=True)
                cor += tmp_prof.get_auto_correlation(window_len, raw=True)
                nmb += 1
            cor /= nmb
            # returning
            dy = self.axe_y[1] - self.axe_y[0]
            y = np.arange(0, len(cor)*dy, dy)
            return prof.Profile(x=y, y=cor, unit_x=self.unit_y,
                                unit_y=make_unit(''))
        else:
            raise ValueError()

    def get_spatial_spectrum(self, direction, intervx=None, intervy=None,
                             welch_seglen=None, scaling='base', fill='linear'):
        """
        Return a spatial spectrum.

        Parameters
        ----------
        direction : string
            'x' or 'y'.
        intervx and intervy : 2x1 arrays of number, optional
            To chose the zone where to calculate the spectrum.
            If not specified, the biggest possible interval is choosen.
        welch_seglen : integer, optional
            If specified, welch's method is used (dividing signal into
            overlapping segments, and averaging periodogram) with the given
            segments length (in number of points).
        scaling : string, optional
            If 'base' (default), result are in component unit.
            If 'spectrum', the power spectrum is returned (in unit^2).
            If 'density', the power spectral density is returned
            (in unit^2/(1/unit_axe))
        fill : string or float
            Specifies the way to treat missing values.
            A value for value filling.
            A string ('linear', 'nearest' or 'cubic') for interpolation.

        Returns
        -------
        spec : prof.Profile object
            Magnitude spectrum.

        Notes
        -----
        If there is missing values on the field, 'fill' is used to linearly
        interpolate the missing values (can impact the spectrum).
        """
        # check parameters
        if not isinstance(direction, STRINGTYPES):
            raise TypeError()
        if direction not in ['x', 'y']:
            raise ValueError()
        if intervx is None:
            intervx = [self.axe_x[0], self.axe_x[-1]]
        if not isinstance(intervx, ARRAYTYPES):
            raise TypeError()
        intervx = np.array(intervx)
        if intervx[0] < self.axe_x[0]:
            intervx[0] = self.axe_x[0]
        if intervx[1] > self.axe_x[-1]:
            intervx[1] = self.axe_x[-1]
        if intervx[1] <= intervx[0]:
            raise ValueError()
        if intervy is None:
            intervy = [self.axe_y[0], self.axe_y[-1]]
        if not isinstance(intervy, ARRAYTYPES):
            raise TypeError()
        intervy = np.array(intervy)
        if intervy[0] < self.axe_y[0]:
            intervy[0] = self.axe_y[0]
        if intervy[1] > self.axe_y[-1]:
            intervy[1] = self.axe_y[-1]
        if intervy[1] <= intervy[0]:
            raise ValueError()
        if isinstance(fill, NUMBERTYPES):
            value = fill
            fill = 'value'
        else:
            value = 0.
        # getting data
        tmp_SF = self.crop(intervx=intervx, intervy=intervy, inplace=False)
        tmp_SF.fill(kind=fill, value=value, inplace=True, reduce_tri=True)
        # getting spectrum
        if direction == 'x':
            # first spectrum
            prof = tmp_SF.get_profile('y', tmp_SF.axe_y[0])
            spec = prof.get_spectrum(welch_seglen=welch_seglen,
                                     scaling=scaling, fill=fill)
            # otherones
            for y in tmp_SF.axe_y[1::]:
                prof = tmp_SF.get_profile('y', y)
                spec += prof.get_spectrum(welch_seglen=welch_seglen,
                                          scaling=scaling, fill=fill)
            spec /= len(tmp_SF.axe_y)
        else:
            # first spectrum
            prof = tmp_SF.get_profile('x', tmp_SF.axe_x[0])
            spec = prof.get_spectrum(welch_seglen=welch_seglen,
                                     scaling=scaling, fill=fill)
            # otherones
            for x in tmp_SF.axe_x[1::]:
                prof = tmp_SF.get_profile('x', x)
                spec += prof.get_spectrum(welch_seglen=welch_seglen,
                                          scaling=scaling, fill=fill)
            spec /= len(tmp_SF.axe_x)
        return spec

    def get_norm(self, norm=2, normalized='perpoint'):
        """
        Return the field norm
        """
        values = self.values[~self.mask]
        norm = (np.sum(np.abs(values)**norm))**(1./norm)
        if normalized == 'perpoint':
            norm /= np.sum(~self.mask)
        return norm

    def get_interpolator(self, interp="linear"):
        """
        Return the field interpolator.

        Parameters
        ----------
        kind : {‘linear’, ‘cubic’, ‘quintic’}, optional
            The kind of spline interpolation to use. Default is ‘linear’.
        """
        return spinterp.interp2d(self.axe_x, self.axe_y,
                                 self.values.transpose(),
                                 kind=interp)

    def integrate_over_line(self, direction, interval):
        """
        Return the integral on an interval and along a direction
        ('x' or 'y').
        Discretized integral is computed with a trapezoidal algorithm.

        Parameters
        ----------
        direction : string in ['x', 'y']
            Direction along which we choose a position.
        interval : interval of numbers
            Interval on which we want to calculate the integral.

        Returns
        -------
        integral : float
            Result of the integrale calcul
        unit : Unit object
            Unit of the result

        """
        profile = self.get_profile(direction, interval)
        if np.any(profile.mask):
            raise Exception("Masked values on the line")
        integrale = np.trapz(profile.y, profile.x)
        if direction == 1:
            unit = self.unit_values*self.unit_y
        else:
            unit = self.unit_values*self.unit_x
        return integrale*unit

    def integrate_over_surface(self, intervx=None, intervy=None):
        """
        Return the integral on a surface.
        Discretized integral is computed with a very rustic algorithm
        which just sum the value on the surface.
        if 'intervx' and 'intervy' are given, return the integral over the
        delimited surface.
        WARNING : Only works (and badly) with regular axes.

        Parameters
        ----------
        intervx : interval of numbers, optional
            Interval along x on which we want to compute the integrale.
        intervy : interval of numbers, optional
            Interval along y on which we want to compute the integrale.

        Returns
        -------
        integral : float
            Result of the integrale computation.
        unit : Unit object
            The unit of the integrale result.
        """
        if intervx is None:
            intervx = [-np.inf, np.inf]
        if intervy is None:
            intervy = [-np.inf, np.inf]
        cropfield = self.crop(intervx=intervx, intervy=intervy, inplace=False)
        if np.any(cropfield.mask):
            raise Exception("Masked values on the surface")
        axe2_x, axe2_y = cropfield.axe_x, cropfield.axe_y
        unit_x, unit_y = cropfield.unit_x, cropfield.unit_y
        integral = (cropfield.values.sum() *
                    np.abs(axe2_x[-1] - axe2_x[0]) *
                    np.abs(axe2_y[-1] - axe2_y[0]) /
                    len(axe2_x) /
                    len(axe2_y))
        unit = cropfield.unit_values*unit_x*unit_y
        return integral*unit

    def copy(self):
        """
        Return a copy of the scalarfield.
        """
        return copy.deepcopy(self)

    def export_to_scatter(self, mask=None):
        """
        Return the scalar field under the form of a pts.Points object.

        Parameters
        ----------
        mask : array of boolean, optional
            Mask to choose values to extract
            (values are taken where mask is False).

        Returns
        -------
        Pts : pts.Points object
            Contening the ScalarField points.
        """
        if mask is None:
            mask = np.zeros(self.shape)
        if not isinstance(mask, ARRAYTYPES):
            raise TypeError("'mask' must be an array of boolean")
        mask = np.array(mask)
        if mask.shape != self.shape:
            raise ValueError("'mask' must have the same dimensions as"
                             "the ScalarField :{}".format(self.shape))
        # récupération du masque
        mask = np.logical_or(mask, self.mask)
        tmp_pts = None
        v = np.array([], dtype=float)
        # boucle sur les points
        for inds, pos, value in self:
            if mask[inds[0], inds[1]]:
                continue
            if tmp_pts is None:
                tmp_pts = [pos]
            else:
                tmp_pts = np.append(tmp_pts, [pos], axis=0)
            v = np.append(v, value)
        return pts.Points(tmp_pts, v, self.unit_x, self.unit_y, self.unit_values)

    def scale(self, scalex=None, scaley=None, scalev=None, inplace=False):
        """
        Scale the ScalarField.

        Parameters
        ----------
        scalex, scaley, scalev : numbers or Unum objects
            Scale for the axis and the values
        inplace : boolean
            .
        """
        if inplace:
            tmp_f = self
        else:
            tmp_f = self.copy()
        # xy
        revx, revy = fld.Field.scale(tmp_f, scalex=scalex, scaley=scaley,
                                     inplace=True,
                                     output_reverse=True)
        # v
        if scalev is None:
            pass
        elif isinstance(scalev, NUMBERTYPES):
            tmp_f.values *= scalev
        elif isinstance(scalev, unum.Unum):
            new_unit = tmp_f.unit_values*scalev
            fact = new_unit.asNumber()
            new_unit /= fact
            tmp_f.unit_values = new_unit
            tmp_f.values *= fact
        else:
            raise TypeError()
        if revx and revy:
            tmp_f.values = tmp_f.values[::-1, ::-1]
        elif revx:
            tmp_f.values = tmp_f.values[::-1, :]
        elif revy:
            tmp_f.values = tmp_f.values[:, ::-1]
        # returning
        if not inplace:
            return tmp_f

    def rotate(self, angle, inplace=False):
        """
        Rotate the scalar field.

        Parameters
        ----------
        angle : integer
            Angle in degrees (positive for trigonometric direction).
            In order to preserve the orthogonal grid, only multiples of
            90° are accepted (can be negative multiples).
        inplace : boolean, optional
            If 'True', scalar field is rotated in place, else, the function
            return a rotated field.

        Returns
        -------
        rotated_field : ScalarField object, optional
            Rotated scalar field.
        """
        # check params
        if not isinstance(angle, NUMBERTYPES):
            raise TypeError()
        if angle % 90 != 0:
            raise ValueError()
        if not isinstance(inplace, bool):
            raise TypeError()
        # get data
        if inplace:
            tmp_field = self
        else:
            tmp_field = self.copy()
        # normalize angle
        angle = angle % 360
        # rotate the parent
        fld.Field.rotate(tmp_field, angle, inplace=True)
        # rotate
        nmb_rot90 = int(angle/90)
        tmp_field.__values = np.rot90(tmp_field.values, nmb_rot90)
        tmp_field.__mask = np.rot90(tmp_field.mask, nmb_rot90)
        # returning
        if not inplace:
            return tmp_field

    def change_dtype(self, new_type):
        """
        Change the values dtype.
        """
        if new_type != self._values_dtype:
            self.values = np.array(self.values, dtype=new_type)
            self._values_dtype = new_type

    def change_unit(self, axe, new_unit):
        """
        Change the unit of an axe.

        Parameters
        ----------
        axe : string
            'y' for changing the profile y axis unit
            'x' for changing the profile x axis unit
            'values' or changing values unit
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
            fld.Field.change_unit(self, axe, new_unit)
        elif axe == 'y':
            fld.Field.change_unit(self, axe, new_unit)
        elif axe == 'values':
            old_unit = self.unit_values
            new_unit = old_unit.asUnit(new_unit)
            fact = new_unit.asNumber()
            self.values *= fact
            self.unit_values = new_unit/fact
        else:
            raise ValueError()

    def crop(self, intervx=None, intervy=None, ind=False,
             inplace=False):
        """
        Crop the area in respect with given intervals.

        Parameters
        ----------
        intervx : array, optional
            interval wanted along x
        intervy : array, optional
            interval wanted along y
        ind : boolean, optional
            If 'True', intervals are understood as indices along axis.
            If 'False' (default), intervals are understood in axis units.
        inplace : boolean, optional
            If 'True', the field is croped in place.
        """
        if inplace:
            values = self.values
            mask = self.mask
            indmin_x, indmax_x, indmin_y, indmax_y = \
                fld.Field.crop(self, intervx, intervy, full_output=True,
                               ind=ind, inplace=True)
            self.__values = values[indmin_x:indmax_x + 1,
                                   indmin_y:indmax_y + 1]
            self.__mask = mask[indmin_x:indmax_x + 1,
                               indmin_y:indmax_y + 1]
        else:
            indmin_x, indmax_x, indmin_y, indmax_y, cropfield = \
                fld.Field.crop(self, intervx=intervx, intervy=intervy,
                               full_output=True, ind=ind, inplace=False)
            cropfield.__values = self.values[indmin_x:indmax_x + 1,
                                             indmin_y:indmax_y + 1]
            cropfield.__mask = self.mask[indmin_x:indmax_x + 1,
                                         indmin_y:indmax_y + 1]
            return cropfield

    def extend(self, nmb_left=0, nmb_right=0, nmb_up=0, nmb_down=0, value=None,
               inplace=False, ind=True):
        """
        Add columns or lines of masked values at the scalarfield.

        Parameters
        ----------
        nmb_**** : integers
            Number of lines/columns to add in each direction.
        value : None or number
            Value used to fill the new columns and lines. If 'value' is not
            given, new columns and lines are masked.
        inplace : bool
            If 'False', return a new extended field, if 'True', modify the
            field inplace.
        Returns
        -------
        Extended_field : Field object, optional
            Extended field.
        """
        if not ind:
            dx = self.axe_x[1] - self.axe_x[0]
            dy = self.axe_y[1] - self.axe_y[0]
            nmb_left = np.ceil(nmb_left/dx)
            nmb_right = np.ceil(nmb_right/dx)
            nmb_up = np.ceil(nmb_up/dy)
            nmb_down = np.ceil(nmb_down/dy)
            ind = True
        # check params
        if not (isinstance(nmb_left, int) or nmb_left % 1 == 0):
            raise TypeError()
        if not (isinstance(nmb_right, int) or nmb_right % 1 == 0):
            raise TypeError()
        if not (isinstance(nmb_up, int) or nmb_up % 1 == 0):
            raise TypeError()
        if not (isinstance(nmb_down, int) or nmb_down % 1 == 0):
            raise TypeError()
        nmb_left = int(nmb_left)
        nmb_right = int(nmb_right)
        nmb_up = int(nmb_up)
        nmb_down = int(nmb_down)
        if np.any(np.array([nmb_left, nmb_right, nmb_up, nmb_down]) < 0):
            raise ValueError()
        # used herited method to extend the field
        tmpsf = fld.Field.extend(self, nmb_left=nmb_left,
                                 nmb_right=nmb_right, nmb_up=nmb_up,
                                 nmb_down=nmb_down, inplace=False)
        new_shape = tmpsf.shape
        # extend the value ans mask
        if value is None:
            new_values = np.zeros(new_shape, dtype=self._values_dtype)
            new_mask = np.ones(new_shape, dtype=bool)
        else:
            new_values = np.ones(new_shape, dtype=self._values_dtype)*value
            new_mask = np.zeros(new_shape, dtype=bool)
        if nmb_right == 0:
            slice_x = slice(nmb_left, new_values.shape[0] + 2)
        else:
            slice_x = slice(nmb_left, -nmb_right)
        if nmb_up == 0:
            slice_y = slice(nmb_down, new_values.shape[1] + 2)
        else:
            slice_y = slice(nmb_down, -nmb_up)
        new_values[slice_x, slice_y] = self.values
        new_mask[slice_x, slice_y] = self.mask
        # return
        if inplace:
            self.import_from_arrays(axe_x=tmpsf.axe_x, axe_y=tmpsf.axe_y,
                                    values=new_values, mask=new_mask,
                                    unit_x=tmpsf.unit_x,
                                    unit_y=tmpsf.unit_y,
                                    unit_values=tmpsf.unit_values)
        else:
            new_field.values = new_values
            new_field.mask = new_mask
            return new_field

    def mirroring(self, direction, position, inds_to_mirror='all',
                  mir_coef=1, interp=None, value=0, inplace=False):
        """
        Return a field with additional mirrored values.

        Parameters
        ----------
        direction : string in ['x', 'y']
            Axe on which place the symetry plane
        position : number
            Position of the symetry plane alogn the given axe
        inds_to_mirror : integer
            Number of vector rows to symetrize (default is all)
        mir_coef : number, optional
            Optional coefficient applied only to the mirrored values.
        inplace : boolean, optional
            .
        interp : string, optional
            If specified, method used to fill the gap near the
            symetry plane by interpoaltion.
            'value' : fill with the given value,
            'nearest' : fill with the nearest value,
            'linear' (default): fill using linear interpolation
            (Delaunay triangulation),
            'cubic' : fill using cubic interpolation (Delaunay triangulation)
        value : array, optional
            Value at the symetry plane, in case of interpolation
        """
        # check params
        if direction not in ['x', 'y']:
            raise TypeError()
        if not isinstance(position, NUMBERTYPES):
            raise TypeError("Position should be a number, not {}"
                            .format(type(position)))
        position = float(position)
        # get data
        axe_x = self.axe_x
        axe_y = self.axe_y
        if inplace:
            tmp_vf = self
        else:
            tmp_vf = self.copy()
        tmp_vf.crop_masked_border(inplace=True)
        # get side to mirror
        if direction == 'x':
            axe = axe_x
            x_median = (axe_x[-1] + axe_x[0])/2.
            delta = axe_x[1] - axe_x[0]
            if position < axe_x[0]:
                border = axe_x[0]
                side = 'left'
            elif position > axe_x[-1]:
                border = axe_x[-1]
                side = 'right'
            elif position < x_median:
                tmp_vf.crop(intervx=[position, axe_x[-1]], ind=False,
                            inplace=True)
                side = 'left'
                axe_x = tmp_vf.axe_x
                border = axe_x[0]
            elif position > x_median:
                tmp_vf.crop(intervx=[axe_x[0], position], ind=False,
                            inplace=True)
                side = 'right'
                axe_x = tmp_vf.axe_x
                border = axe_x[-1]
            else:
                raise ValueError()
        elif direction == 'y':
            axe = axe_y
            y_median = (axe_y[-1] + axe_y[0])/2.
            delta = axe_y[1] - axe_y[0]
            if position < axe_y[0]:
                border = axe_y[0]
                side = 'down'
            elif position > axe_y[-1]:
                border = axe_y[-1]
                side = 'up'
            elif position < y_median:
                tmp_vf.crop(intervy=[position, axe_y[-1]], ind=False,
                            inplace=True)
                side = 'down'
                axe_y = tmp_vf.axe_y
                border = axe_y[0]
            elif position > y_median:
                tmp_vf.crop(intervy=[axe_y[0], position], ind=False,
                            inplace=True)
                side = 'up'
                axe_y = tmp_vf.axe_y
                border = axe_y[-1]
            else:
                raise ValueError()
        else:
            raise ValueError()
        # get length to mirror
        if inds_to_mirror == 'all' or inds_to_mirror > len(axe):
            inds_to_mirror = len(axe) - 1
        if side in ['left', 'down']:
            delta_gap = -(position - border)/delta
        else:
            delta_gap = (position - border)/delta
        inds_to_add = np.ceil(inds_to_mirror + 2*delta_gap) - 1
        # extend the field
        tmp_dic = {'nmb_{}'.format(side): inds_to_add}
        tmp_vf.extend(inplace=True, **tmp_dic)
        new_axe_x = tmp_vf.axe_x
        new_axe_y = tmp_vf.axe_y
        # filling mirrored part with interpolated values
        for i, x in enumerate(new_axe_x):
            for j, y in enumerate(new_axe_y):
                # if point is not masked
                if not tmp_vf.mask[i, j]:
                    continue
                # get mirror point position
                if direction == 'x':
                    mir_pos = [position - (x - position), y]
                else:
                    mir_pos = [x, position - (y - position)]
                # if mirror point is outside hte field
                if mir_pos[0] < new_axe_x[0] or mir_pos[0] > new_axe_x[-1] \
                        or mir_pos[1] < new_axe_y[0] \
                        or mir_pos[1] > new_axe_y[-1]:
                    continue
                # get mirror point value
                mir_val = tmp_vf.get_value(*mir_pos, ind=False)
                # if mirror point can't be interpolated (masked)
                if np.isnan(mir_val):
                    continue
                # Sotring the new value in the field
                tmp_vf.values[i, j] = mir_val*mir_coef
                tmp_vf.mask[i, j] = False
        # getting mask
        masked_values = np.any(tmp_vf.mask)
        # interpolating between mirror images
        if interp is None:
            pass
        elif interp == 'value' and masked_values:
            tmp_vf.fill(kind='value', value=value, inplace=True, crop=False,
                        reduce_tri=False)
        elif interp == 'linear' and masked_values:
            # getting data
            new_axe_x = tmp_vf.axe_x
            new_axe_y = tmp_vf.axe_y
            values = tmp_vf.values
            mask = tmp_vf.mask
            # direction x
            if side in ['right', 'left']:
                # get last column
                ind_last = np.where(mask[:, 0])[0][0] - 1
                # get number of missing values
                nmb_masked = np.sum(mask[:, 0])
                # loop on lines
                for i in np.arange(len(new_axe_y)):
                    tmp_val = np.linspace(values[ind_last - 1, i],
                                          values[ind_last + nmb_masked, i],
                                          nmb_masked + 2)[1:-1]
                    values[ind_last:ind_last + nmb_masked, i] = tmp_val
            else:
                # get last column
                ind_last = np.where(mask[0, :])[0][0]
                # get number of missing values
                nmb_masked = np.sum(mask[0, :])
                # loop on lines
                for i in np.arange(len(new_axe_x)):
                    tmp_val = np.linspace(values[i, ind_last - 1],
                                          values[i, ind_last + nmb_masked],
                                          nmb_masked + 2)[1:-1]
                    values[i, ind_last:ind_last + nmb_masked] = tmp_val
            tmp_vf.values = values
            tmp_vf.mask = False
        # slower interplation method
        elif interp is not None and masked_values:
            # getting data
            x, y = tmp_vf.axe_x, tmp_vf.axe_y
            values = tmp_vf.values
            mask = tmp_vf.mask
            # geting filters from mask
            if interp in ['nearest', 'linear', 'cubic']:
                X, Y = np.meshgrid(x, y, indexing='ij')
                xy = [X.flat[:], Y.flat[:]]
                xy = np.transpose(xy)
                filt = np.logical_not(mask)
                xy_masked = xy[mask.flatten()]
            # getting the zone to interpolate
                xy_good = xy[filt.flatten()]
                values_good = values[filt]
            else:
                raise ValueError()
            # adding the value at the symetry plane
            if direction == 'x':
                addit_xy = list(zip([position]*len(tmp_vf.axe_y),
                                    tmp_vf.axe_y))
                addit_values = [value]*len(tmp_vf.axe_y)
            else:
                addit_xy = list(zip(tmp_vf.axe_x,
                                    [position]*len(tmp_vf.axe_x)))
                addit_values = [value]*len(tmp_vf.axe_x)
            xy_good = np.concatenate((xy_good, addit_xy))
            values_good = np.concatenate((values_good, addit_values))
            # if interpolation
            if interp == 'value':
                values[mask] = value
            elif interp == 'nearest':
                nearest = spinterp.NearestNDInterpolator(xy_good, values_good)
                values[mask] = nearest(xy_masked)
            elif interp == 'linear':
                linear = spinterp.LinearNDInterpolator(xy_good, values_good)
                values[mask] = linear(xy_masked)
                new_mask = np.isnan(values)
                if np.any(new_mask):
                    nearest = spinterp.NearestNDInterpolator(xy_good,
                                                             values_good)
                    values[new_mask] = nearest(xy[new_mask.flatten()])
            elif interp == 'cubic':
                cubic = spinterp.CloughTocher2DInterpolator(xy_good,
                                                            values_good)
                values[mask] = cubic(xy_masked)
                new_mask = np.isnan(values)
                if np.any(new_mask):
                    nearest = spinterp.NearestNDInterpolator(xy_good,
                                                             values_good)
                    values[new_mask] = nearest(xy[new_mask.flatten()])
            else:
                raise ValueError("unknown 'tof' value")
            tmp_vf.values = values
            tmp_vf.mask = False
        # returning
        if not inplace:
            return tmp_vf

    def crop_masked_border(self, hard=False, inplace=False):
        """
        Crop the masked border of the field in place or not.

        Parameters
        ----------
        hard : boolean, optional
            If 'True', partially masked border are croped as well.
        """
        #
        if inplace:
            tmp_vf = self
        else:
            tmp_vf = self.copy()
        # checking masked values presence
        mask = tmp_vf.mask
        if not np.any(mask):
            return None
        # hard cropping
        if hard:
            # remove trivial borders
            tmp_vf.crop_masked_border(hard=False, inplace=True)
            # until there is no more masked values
            while np.any(tmp_vf.mask):
                # getting number of masked value on each border
                bd1 = np.sum(tmp_vf.mask[0, :])
                bd2 = np.sum(tmp_vf.mask[-1, :])
                bd3 = np.sum(tmp_vf.mask[:, 0])
                bd4 = np.sum(tmp_vf.mask[:, -1])
                # getting more masked border
                more_masked = np.argmax([bd1, bd2, bd3, bd4])
                # deleting more masked border
                if more_masked == 0:
                    len_x = len(tmp_vf.axe_x)
                    tmp_vf.crop(intervx=[1, len_x], ind=True, inplace=True)
                elif more_masked == 1:
                    len_x = len(tmp_vf.axe_x)
                    tmp_vf.crop(intervx=[0, len_x - 2], ind=True,
                                inplace=True)
                elif more_masked == 2:
                    len_y = len(tmp_vf.axe_y)
                    tmp_vf.crop(intervy=[1, len_y], ind=True,
                                inplace=True)
                elif more_masked == 3:
                    len_y = len(tmp_vf.axe_y)
                    tmp_vf.crop(intervy=[0, len_y - 2], ind=True,
                                inplace=True)
        # soft cropping
        else:
            axe_x_m = np.logical_not(np.all(mask, axis=1))
            axe_y_m = np.logical_not(np.all(mask, axis=0))
            axe_x_min = np.where(axe_x_m)[0][0]
            axe_x_max = np.where(axe_x_m)[0][-1]
            axe_y_min = np.where(axe_y_m)[0][0]
            axe_y_max = np.where(axe_y_m)[0][-1]
            tmp_vf.crop([axe_x_min, axe_x_max],
                        [axe_y_min, axe_y_max],
                        ind=True, inplace=True)
        # returning
        if not inplace:
            return tmp_vf

    def fill(self, kind='linear', value=0., inplace=False, reduce_tri=True,
             crop=False):
        """
        Fill the masked part of the array.

        Parameters
        ----------
        kind : string, optional
            Type of algorithm used to fill.
            'value' : fill with the given value
            'nearest' : fill with the nearest value
            'linear' (default): fill using linear interpolation
            (Delaunay triangulation)
            'cubic' : fill using cubic interpolation (Delaunay triangulation)
        value : number
            Value used to fill (for kind='value').
        inplace : boolean, optional
            If 'True', fill the ScalarField in place.
            If 'False' (default), return a filled version of the field.
        reduce_tri : boolean, optional
            If 'True', treatment is used to reduce the triangulation effort
            (faster when a lot of masked values)
            If 'False', no treatment
            (faster when few masked values)
        crop : boolean, optional
            If 'True', SF borders are croped before filling.
                """
        # check parameters coherence
        if not isinstance(kind, STRINGTYPES):
            raise TypeError("'kind' must be a string")
        if not isinstance(value, NUMBERTYPES):
            raise TypeError("'value' must be a number")
        if crop:
            self.crop_masked_border(hard=False, inplace=True)
        # getting data
        x, y = self.axe_x, self.axe_y
        values = self.values
        mask = self.mask
        if kind in ['nearest', 'linear', 'cubic']:
            X, Y = np.meshgrid(x, y, indexing='ij')
            xy = [X.flat[:], Y.flat[:]]
            xy = np.transpose(xy)
            filt = np.logical_not(mask)
            xy_masked = xy[mask.flatten()]
        # getting the zone to interpolate
        if reduce_tri and kind in ['nearest', 'linear', 'cubic']:
            import scipy.ndimage as spim
            dilated = spim.binary_dilation(self.mask,
                                           np.arange(9).reshape((3, 3)))
            filt_good = np.logical_and(filt, dilated)
            xy_good = xy[filt_good.flatten()]
            values_good = values[filt_good]
        elif not reduce_tri and kind in ['nearest', 'linear', 'cubic']:
            xy_good = xy[filt.flatten()]
            values_good = values[filt]
        else:
            pass
        # if there is nothing to do...
        if not np.any(mask):
            pass
        # if interpolation
        elif kind == 'value':
            values[mask] = value
        elif kind == 'nearest':
            nearest = spinterp.NearestNDInterpolator(xy_good, values_good)
            values[mask] = nearest(xy_masked)
        elif kind == 'linear':
            linear = spinterp.LinearNDInterpolator(xy_good, values_good)
            values[mask] = linear(xy_masked)
            new_mask = np.isnan(values)
            if np.any(new_mask):
                nearest = spinterp.NearestNDInterpolator(xy_good, values_good)
                values[new_mask] = nearest(xy[new_mask.flatten()])
        elif kind == 'cubic':
            cubic = spinterp.CloughTocher2DInterpolator(xy_good, values_good)
            values[mask] = cubic(xy_masked)
            new_mask = np.isnan(values)
            if np.any(new_mask):
                nearest = spinterp.NearestNDInterpolator(xy_good, values_good)
                values[new_mask] = nearest(xy[new_mask.flatten()])
        else:
            raise ValueError("unknown 'tof' value")
        # returning
        if inplace:
            self.values = values
            self.mask = False
        else:
            sf = ScalarField()
            sf.import_from_arrays(x, y, values, mask=False, unit_x=self.unit_x,
                                  unit_y=self.unit_y,
                                  unit_values=self.unit_values)
            return sf

    def smooth(self, tos='uniform', size=None, inplace=False, **kw):
        """
        Smooth the scalarfield in place.
        Warning : fill up the field (should be used carefully with masked field
        borders)

        Parameters
        ----------
        tos : string, optional
            Type of smoothing, can be 'uniform' (default) or 'gaussian'
            (See ndimage module documentation for more details)
        size : number, optional
            Size of the smoothing (is radius for 'uniform' and
            sigma for 'gaussian') in indice number.
            Default is 3 for 'uniform' and 1 for 'gaussian'.
        inplace : boolean, optional
            If True, Field is smoothed in place,
            else, the smoothed field is returned.
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
        # filling up the field before smoothing
        if inplace:
            self.fill(inplace=True)
            values = self.values
        else:
            tmp_sf = self.fill(inplace=False)
            values = tmp_sf.values
        # smoothing
        if tos == "uniform":
            values = ndimage.uniform_filter(values, size, **kw)
        elif tos == "gaussian":
            values = ndimage.gaussian_filter(values, size, **kw)
        else:
            raise ValueError("'tos' must be 'uniform' or 'gaussian'")
        # storing
        if inplace:
            self.values = values
        else:
            tmp_sf.values = values
            return tmp_sf

    def make_evenly_spaced(self, interp="linear", res=1):
        """
        Use interpolation to make the field evenly spaced

        Parameters
        ----------
        interp : {‘linear’, ‘cubic’, ‘quintic’}, optional
            The kind of spline interpolation to use. Default is ‘linear’.
        res : number
            Resolution of the resulting field.
            A value of 1 meaning a spatial resolution equal to the smallest
            space along the two axis for the initial field.
        """
        # get data
        axex = self.axe_x
        axey = self.axe_y
        dx = np.min(axex[1:] - axex[:-1])/res
        dy = np.min(axey[1:] - axey[:-1])/res
        Dx = axex[-1] - axex[0]
        Dy = axey[-1] - axey[0]
        #
        interp = self.get_interpolator(interp=interp)
        # new_x = np.arange(axex[0], axex[-1] + .1*dx, dx)
        new_x = np.linspace(axex[0], axex[-1], int(Dx/dx))
        # new_y = np.arange(axey[0], axey[-1] + .1*dy, dy)
        new_y = np.linspace(axey[0], axey[-1], int(Dy/dy))
        new_values = interp(new_x, new_y)
        # store
        self.import_from_arrays(new_x, new_y, new_values.transpose(),
                                mask=False, unit_x=self.unit_x,
                                unit_y=self.unit_y,
                                unit_values=self.unit_values)

    def reduce_spatial_resolution(self, fact, inplace=False):
        """
        Reduce the spatial resolution of the field by a factor 'fact'

        Parameters
        ----------
        fact : int
            Reducing factor.
        inplace : boolean, optional
            .
        """
        if not isinstance(fact, int):
            raise TypeError()
        if fact < 1:
            raise ValueError()
        if fact == 1:
            if inplace:
                pass
            else:
                return self.copy()
        if fact % 2 == 0:
            pair = True
        else:
            pair = False
        # get new axis
        axe_x = self.axe_x
        axe_y = self.axe_y
        if pair:
            new_axe_x = (axe_x[np.arange(fact/2 - 1, len(axe_x) - fact/2,
                                         fact, dtype=int)] +
                         axe_x[np.arange(fact/2, len(axe_x) - fact/2 + 1,
                                         fact, dtype=int)])/2.
            new_axe_y = (axe_y[np.arange(fact/2 - 1, len(axe_y) - fact/2,
                                         fact, dtype=int)] +
                         axe_y[np.arange(fact/2, len(axe_y) - fact/2 + 1,
                                         fact, dtype=int)])/2.
        else:
            new_axe_x = axe_x[np.arange((fact - 1)/2,
                                        len(axe_x) - (fact - 1)/2,
                                        fact, dtype=int)]
            new_axe_y = axe_y[np.arange((fact - 1)/2,
                                        len(axe_y) - (fact - 1)/2,
                                        fact, dtype=int)]
        # get new values
        values = self.values
        mask = self.mask
        if pair:
            inds_x = np.arange(fact/2, len(axe_x) - fact/2 + 1,
                               fact, dtype=int)
            inds_y = np.arange(fact/2, len(axe_y) - fact/2 + 1,
                               fact, dtype=int)
            new_values = np.zeros((len(inds_x), len(inds_y)))
            new_mask = np.zeros((len(inds_x), len(inds_y)))
            for i in np.arange(len(inds_x)):
                intervx = slice(inds_x[i] - int(fact/2),
                                inds_x[i] + int(fact/2))
                for j in np.arange(len(inds_y)):
                    intervy = slice(inds_y[j] - int(fact/2),
                                    inds_y[j] + int(fact/2))
                    if np.all(mask[intervx, intervy]):
                        new_mask[i, j] = True
                        new_values[i, j] = 0.
                    else:
                        new_values[i, j] = np.mean(values[intervx, intervy]
                                                   [~mask[intervx, intervy]])

        else:
            inds_x = np.arange((fact - 1)/2, len(axe_x) - (fact - 1)/2, fact)
            inds_y = np.arange((fact - 1)/2, len(axe_y) - (fact - 1)/2, fact)
            new_values = np.zeros((len(inds_x), len(inds_y)))
            new_mask = np.zeros((len(inds_x), len(inds_y)))
            for i in np.arange(len(inds_x)):
                intervx = slice(inds_x[i] - (fact - 1)/2,
                                inds_x[i] + (fact - 1)/2 + 1)
                for j in np.arange(len(inds_y)):
                    intervy = slice(inds_y[j] - (fact - 1)/2,
                                    inds_y[j] + (fact - 1)/2 + 1)
                    if np.all(mask[intervx, intervy]):
                        new_mask[i, j] = True
                        new_values[i, j] = 0.
                    else:
                        new_values[i, j] = np.mean(values[intervx, intervy]
                                                   [~mask[intervx, intervy]])
        # ensuring right data type
        new_values = np.array(new_values, dtype=self._values_dtype)
        # returning
        if inplace:
            self.__init__()
            self.import_from_arrays(new_axe_x, new_axe_y, new_values,
                                    mask=new_mask,
                                    unit_x=self.unit_x, unit_y=self.unit_y,
                                    unit_values=self.unit_values,
                                    dtype=self._values_dtype)

        else:
            sf = ScalarField()
            sf.import_from_arrays(new_axe_x, new_axe_y, new_values,
                                  mask=new_mask,
                                  unit_x=self.unit_x, unit_y=self.unit_y,
                                  unit_values=self.unit_values,
                                  dtype=self._values_dtype)
            return sf

    def __clean(self):
        self.__init__()

    def _display(self, component=None, kind=None, **plotargs):
        # getting datas
        axe_x, axe_y = self.axe_x, self.axe_y
        unit_x, unit_y = self.unit_x, self.unit_y
        X, Y = np.meshgrid(self.axe_y, self.axe_x)
        # getting wanted component
        if component is None or component == 'values':
            values = self.values.astype(dtype=self._values_dtype)
            mask = self.mask
            try:
                values[mask] = np.NaN
            except ValueError:
                values[mask] = 0
        elif component == 'mask':
            values = self.mask
        else:
            raise ValueError("unknown value of 'component' parameter : {}"
                             .format(component))
        dp = pplt.Displayer(x=axe_x, y=axe_y, values=values, kind=kind,
                            **plotargs)
        plot = dp.draw()
        pplt.DataCursorTextDisplayer(dp)
        # setting labels
        if unit_x.strUnit() == "[]":
            plt.xlabel("x")
        else:
            plt.xlabel("x " + unit_x.strUnit())
        if unit_y.strUnit() == "[]":
            plt.ylabel("y")
        else:
            plt.ylabel("y " + unit_y.strUnit())
        return plot

    def display(self, component=None, kind=None, **plotargs):
        """
        Display the scalar field.

        Parameters
        ----------
        component : string, optional
            Component to display, can be 'values' or 'mask'
        kind : string, optinnal
            If 'imshow': (default) each datas are plotted (imshow),
            if 'contour': contours are ploted (contour),
            if 'contourf': filled contours are ploted (contourf).
        **plotargs : dict
            Arguments passed to the 'contourf' function used to display the
            scalar field.

        Returns
        -------
        fig : figure reference
            Reference to the displayed figure.
        """
        displ = self._display(component, kind, **plotargs)
        plt.title("")
        cb = plt.colorbar(displ)
        if self.unit_values.strUnit() == "[]":
            cb.set_label("Values")
        else:
            cb.set_label("Values {}".format(self.unit_values.strUnit()))
        # search for limits in case of masked field
        if component != 'mask':
            mask = self.mask
            for i in np.arange(len(self.axe_x)):
                if not np.all(mask[i, :]):
                    break
            xmin = self.axe_x[i]
            for i in np.arange(len(self.axe_x) - 1, -1, -1):
                if not np.all(mask[i, :]):
                    break
            xmax = self.axe_x[i]
            for i in np.arange(len(self.axe_y)):
                if not np.all(mask[:, i]):
                    break
            ymin = self.axe_y[i]
            for i in np.arange(len(self.axe_y) - 1, -1, -1):
                if not np.all(mask[:, i]):
                    break
            ymax = self.axe_y[i]
            plt.xlim([xmin, xmax])
            plt.ylim([ymin, ymax])
        return displ
