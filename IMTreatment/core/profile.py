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
import numpy as np
import unum
import copy
from scipy import stats
import scipy.interpolate as spinterp
from scipy import ndimage
import scipy.optimize as spopt
import scipy.signal as spsign
from ..utils import make_unit
from .. import plotlib as pplt
from ..utils.types import ARRAYTYPES, INTEGERTYPES, NUMBERTYPES, STRINGTYPES
from .scalarfield import ScalarField


class Profile(object):
    """
    Class representing a profile.
    You can use 'make_unit' to provide unities.

    Parameters
    ----------
    x, y : arrays
        Profile values.
    unit_x, unit_y : Unit objects
        Values unities.
    name : string, optionnal
        A name for the profile.
    """

    def __init__(self, x=[], y=[], mask=False, unit_x="",
                 unit_y="", name=""):
        """
        Profile builder.
        """
        if not isinstance(x, ARRAYTYPES):
            raise TypeError("'x' must be an array")
        if not isinstance(x, np.ma.MaskedArray):
            x = np.array(x, dtype=float)
        if not isinstance(y, ARRAYTYPES):
            raise TypeError("'y' must be an array")
        if not isinstance(y, np.ma.MaskedArray):
            y = np.array(y, dtype=float)
        if isinstance(mask, bool):
            mask = np.empty(x.shape, dtype=bool)
            mask.fill(False)
        if not isinstance(mask, ARRAYTYPES):
            raise TypeError("'mask' must be an array")
        if not isinstance(mask, (np.ndarray, np.ma.MaskedArray)):
            mask = np.array(mask, dtype=bool)
        if not isinstance(name, STRINGTYPES):
            raise TypeError("'name' must be a string")
        if isinstance(unit_x, STRINGTYPES):
            unit_x = make_unit(unit_x)
        if not isinstance(unit_x, unum.Unum):
            raise TypeError("'unit_x' must be a 'Unit' object")
        if isinstance(unit_y, STRINGTYPES):
            unit_y = make_unit(unit_y)
        if not isinstance(unit_y, unum.Unum):
            raise TypeError("'unit_y' must be a 'Unit' object")
        if not len(x) == len(y):
            raise ValueError("'x' and 'y' must have the same length")
        order = np.argsort(x)
        self.__x = x[order]
        self.__y = y[order]
        self.mask = mask[order]
        self.mask = np.logical_or(self.mask, np.isnan(self.y))
        self.name = name
        self.unit_x = unit_x.copy()
        self.unit_y = unit_y.copy()

    def __neg__(self):
        return Profile(self.x, -self.y, mask=self.mask, unit_x=self.unit_x,
                       unit_y=self.unit_y, name=self.name)

    def __add__(self, otherone):
        if isinstance(otherone, NUMBERTYPES):
            y = self.y + otherone
            name = self.name
            mask = self.mask
        elif isinstance(otherone, unum.Unum):
            y = self.y + (otherone/self.unit_y).asNumber()
            name = self.name
            mask = self.mask
        elif isinstance(otherone, Profile):
            try:
                self.unit_x + otherone.unit_x
                self.unit_y + otherone.unit_y
            except:
                raise ValueError("Profiles have not the same unit system")
            if not len(self.x) == len(otherone.x):
                raise ValueError("Profiles have not the same length")
            if not all(self.x == otherone.x):
                raise ValueError("Profiles have not the same x axis")
            y = self.y + (self.unit_y/otherone.unit_y*otherone.y).asNumber()
            name = ""
            mask = np.logical_or(self.mask, otherone.mask)
        else:
            raise TypeError("You only can substract Profile with "
                            "Profile or number")
        return Profile(self.x, y, mask=mask, unit_x=self.unit_x,
                       unit_y=self.unit_y, name=name)

    __radd__ = __add__

    def __sub__(self, otherone):
        return self.__add__(-otherone)

    def __rsub__(self, otherone):
        return -self.__add__(-otherone)

    def __mul__(self, otherone):
        if isinstance(otherone, NUMBERTYPES):
            y = self.y*otherone
            new_unit_y = self.unit_y
        elif isinstance(otherone, unum.Unum):
            new_unit_y = self.unit_y*otherone
            y = self.y*new_unit_y.asNumber()
            new_unit_y = new_unit_y/new_unit_y.asNumber()
        elif isinstance(otherone, Profile):
            if np.any(otherone.x - self.x > 1e-6*np.mean(abs(self.x))):
                raise ValueError("Given profiles does'nt share the same"
                                 " x axis")
            if not otherone.unit_x == self.unit_x:
                raise ValueError("Given profiles does'nt share the same"
                                 " x units")
            # get shared x
            # TODO : find another way to deal with quasi equal values
            dx = self.x[1] - self.x[0]
            shared_x = [0, 0]
            if otherone.x[0] in self.x:
                shared_x[0] = otherone.x[0] - dx/2.
            else:
                shared_x[0] = self.x[0] - dx/2.
            if otherone.x[-1] in self.x:
                shared_x[1] = otherone.x[-1] + dx/2.
            else:
                shared_x[1] = self.x[-1] + dx/2.
            # get filters
            filt_self = np.logical_and(self.x <= shared_x[1],
                                       self.x >= shared_x[0])
            filt_other = np.logical_and(otherone.x <= shared_x[1],
                                        otherone.x >= shared_x[0])
            # get values
            x = self.x[filt_self]
            values = self.y[filt_self]*otherone.y[filt_other]
            mask = np.logical_or(self.mask[filt_self],
                                 otherone.mask[filt_other])
            unit_y = self.unit_y*otherone.unit_y
            tmp_prof = Profile(x, values, mask=mask, unit_x=self.unit_x,
                               unit_y=unit_y)
            return tmp_prof
        else:
            raise TypeError("You only can multiply Profile with number and "
                            "other profiles")
        return Profile(x=self.x, y=y, unit_x=self.unit_x, unit_y=new_unit_y,
                       name=self.name)

    __rmul__ = __mul__

    def __truediv__(self, otherone):
        if isinstance(otherone, NUMBERTYPES):
            y = self.y/otherone
            mask = self.mask
            name = self.name
            unit_y = self.unit_y
        elif isinstance(otherone, unum.Unum):
            tmpunit = self.unit_y/otherone
            y = self.y*(tmpunit.asNumber())
            mask = self.mask
            name = self.name
            unit_y = tmpunit/tmpunit.asNumber()
        elif isinstance(otherone, Profile):
            if not np.all(self.x == otherone.x):
                raise ValueError("Profile has to have identical x axis in "
                                 "order to divide them")
            else:
                mask = np.logical_or(self.mask, otherone.mask)
                tmp_unit = self.unit_y/otherone.unit_y
                y_tmp = self.y.copy()
                y_tmp[otherone.y == 0] = np.NaN
                otherone.y[otherone.y == 0] = 1
                y = y_tmp/otherone.y*tmp_unit.asNumber()
                name = ""
                unit_y = tmp_unit/tmp_unit.asNumber()
        else:
            raise TypeError("You only can divide Profile with number")
        return Profile(self.x, y, mask, self.unit_x, unit_y, name=name)

    __div__ = __truediv__

    def __pow__(self, number):
        if not isinstance(number, NUMBERTYPES):
            raise TypeError("You only can use a number for the power "
                            "on a Profile")
        y = np.power(self.y, number)
        unit_y = np.power(self.unit_y, number)
        return Profile(x=self.x, y=y, unit_x=self.unit_x, unit_y=unit_y,
                       name=self.name)

    def __len__(self):
        return len(self.x)

    def __eq__(self, obj):
        if not isinstance(obj, Profile):
            return False
        if not np.all(self.mask == obj.mask):
            return False
        if not np.all(self.x[~self.mask] == obj.x[~obj.mask]):
            return False
        if not np.all(self.y[~self.mask] == obj.y[~obj.mask]):
            return False
        if not self.unit_x == obj.unit_x:
            return False
        if not self.unit_y == obj.unit_y:
            return False
        return True

    # def __getitem__(self, ind):
    #     if isinstance(ind, ARRAYTYPES):
    #         ind = np.array(ind)
    #         if len(ind) == len(self):
    #             tmp_pts = self.copy()
    #             tmp_pts.x = self.x[ind]
    #             tmp_pts.y = self.y[ind]
    #             tmp_pts.mask = self.mask[ind]
    #             return tmp_pts
    #         else:
    #             raise ValueError()
    #     elif isinstance(ind, int):
    #         return self.x[ind], self.y[ind]
    #     else:
    #         raise TypeError()

    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, values):
        if isinstance(values, ARRAYTYPES):
            self.__x = np.array(values, dtype=float)
        else:
            raise Exception("'x' should be an array, not {}"
                            .format(type(values)))
        if len(values) != len(self.__y):
            raise Exception("Please set 'y' before 'x'")
        if len(self.__x) > 1:
            ind_sort = np.argsort(self.__x)
            self.__x = self.__x[ind_sort]
            self.__y = self.__y[ind_sort]
            self.__mask = self.__mask[ind_sort]

    @x.deleter
    def x(self):
        raise Exception("Nope, can't delete 'x'")

    @property
    def y(self):
        self.__y[self.__mask] = np.NaN
        return self.__y

    @y.setter
    def y(self, values):
        if isinstance(values, np.ma.MaskedArray):
            self.__y = values.data
            self.mask = values.mask
        elif isinstance(values, ARRAYTYPES):
            self.__y = np.array(values)*1.
            self.mask = np.isnan(values)
        else:
            raise Exception()

    @y.deleter
    def y(self):
        raise Exception("Nope, can't delete 'y'")

    @property
    def mask(self):
        return self.__mask

    @mask.setter
    def mask(self, mask):
        if isinstance(mask, bool):
            self.__mask = np.empty(self.y.shape, dtype=bool)
            self.__mask.fill(mask)
        elif isinstance(mask, ARRAYTYPES):
            self.__mask = np.array(mask, dtype=bool)
        else:
            raise Exception()
        self.__y[self.__mask] = np.NaN

    @mask.deleter
    def mask(self):
        raise Exception("Nope, can't delete 'mask'")

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
    def max(self):
        """
        Return the maxima along an axe.

        Parameters
        ----------
        axe : integer, optionnal
            Axe along which we want the maxima.

        Returns
        -------
        max : number
            Maxima along 'axe'.
        """
        if np.all(self.mask):
            return None
        return np.max(self.y[np.logical_not(self.mask)])

    @property
    def min(self):
        """
        Return the minima along an axe.

        Parameters
        ----------
        axe : integer, optionnal
            Axe along which we want the minima.

        Returns
        -------
        max : number
            Minima along 'axe'.
        """
        if np.all(self.mask):
            return None
        return np.min(self.y[np.logical_not(self.mask)])

    @property
    def mean(self):
        """
        Return the minima along an axe.

        Parameters
        ----------
        axe : integer, optionnal
            Axe along which we want the minima.

        Returns
        -------
        max : number
            Minima along 'axe'.
        """
        if np.all(self.mask):
            return None
        return np.mean(self.y[np.logical_not(self.mask)])

    def copy(self):
        """
        Return a copy of the Profile object.
        """
        return copy.deepcopy(self)

    def get_props(self):
        """
        Print the Profile main properties
        """
        text = "Length: {}".format(len(self.x))
        unit_x = self.unit_x.strUnit()
        text += "x: [{}..{}]{}".format(self.x[0], self.x[-1], unit_x)
        unit_y = self.unit_y.strUnit()
        text += "y: [{}..{}]{}".format(self.y[0], self.y[-1], unit_y)
        nmb_mask = np.sum(self.mask)
        nmb_tot = len(self.mask)
        text += "Masked values: {}/{}".format(nmb_mask, nmb_tot)
        return text

    def get_interpolator(self, kind='linear', bounds_error=True,
                         fill_value=np.nan):
        """
        Return an interpolator of the profile

        Parameters
        ----------
        kind : str or int, optional
            Specifies the kind of interpolation as a string (‘linear’,
            ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic, ‘cubic’ where ‘slinear’,
            ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first,
            second or third order) or as an integer specifying the order of
            the spline interpolator to use. Default is ‘linear’.
        bounds_error : bool, optional
            If True, a ValueError is raised any time interpolation is
            attempted on a value outside of the range of x (where
            extrapolation is necessary). If False, out of bounds
            values are assigned `fill_value`. By default, an error is
            raised unless `fill_value="extrapolate"`.
        fill_value : array-like or (array-like, array_like) or "extrapolate"
            - if a ndarray (or float), this value will be used to fill in for
            requested points outside of the data range. If not provided, then
            the default is NaN. The array-like must broadcast properly to the
            dimensions of the non-interpolation axes.
            - If a two-element tuple, then the first element is used as a
            fill value for ``x_new < x[0]`` and the second element is used for
            ``x_new > x[-1]``. Anything that is not a 2-element tuple (e.g.,
            list or ndarray, regardless of shape) is taken to be a single
            array-like argument meant to be used for both bounds as
            ``below, above = fill_value, fill_value``.
            - If "extrapolate", then points outside the data range will be
            extrapolated.

        Returns
        -------
        interpolator : function
            Take a single value 'x' and return the interpolated value of 'y'.

        Note
        ----
        Use scipy.interpolate module

        """
        valid_x = self.x[~self.mask]
        valid_y = self.y[~self.mask]
        interpo = spinterp.interp1d(valid_x, valid_y, kind=kind,
                                    bounds_error=bounds_error,
                                    fill_value=fill_value)
        return interpo

    def get_interpolated_values(self, x=None, y=None, ind=False):
        """
        Get the interpolated (or not) value for given 'x' or 'y' values.

        If several possibilities are possible, an array with all the results
        is returned.

        Parameters
        ----------
        x : number or array of number
            Value(s) of x, for which we want the y value.
        y : number or array of number
            Value(s) of y, for which we want the x value.
        ind : boolean
            If 'True', 'x' and 'y' are treated as indices, else, they are
            treated as position alog axis.
        Returns
        -------
        i_values : number or array
            Interpolated value(s).
        """
        if x is None and y is None:
            raise Warning("Ok, but i'll do nothing if i don't have a 'x' "
                          "or a 'y' value")
        if y is not None and x is not None:
            raise ValueError("Maybe you would like to look at the help "
                             "one more time...")
        if x is not None:
            if isinstance(x, NUMBERTYPES):
                return self._get_interpolated_single_value(x=x, ind=ind)
            else:
                res = []
                for xi in x:
                    val = self._get_interpolated_single_value(x=xi, ind=ind)
                    if len(val) == 0:
                        res.append(np.nan)
                    else:
                        res.append(val[0])
                return np.array(res, dtype=float)
        if y is not None:
            if isinstance(y, NUMBERTYPES):
                return self._get_interpolated_single_value(y=y, ind=ind)
            else:
                res = []
                for yi in y:
                    val = self._get_interpolated_single_value(y=yi, ind=ind)
                    if len(val) == 0:
                        res.append(np.nan)
                    else:
                        res.append(val[0])
                return np.array(res, dtype=float)

    def _get_interpolated_single_value(self, x=None, y=None, ind=False):
        """
        Get the interpolated (or not) value for a given 'x' or 'y' value.

        If several possibilities are possible, an array with all the results
        is returned.

        Parameters
        ----------
        x : number, optionnal
            Value of x, for which we want the y value.
        y : number, optionnal
            Value of y, for which we want the x value.
        ind : boolean
            If 'True', 'x' and 'y' are treated as indices, else, they are
            treated as position alog axis.
        Returns
        -------
        i_values : number or array
            Interpolated value(s).
        """
        if x is not None:
            if not isinstance(x, NUMBERTYPES):
                raise TypeError("'x' must be a number")
        if y is not None:
            if not isinstance(y, NUMBERTYPES):
                raise TypeError("'y' must be a number")

        if not isinstance(ind, bool):
            raise TypeError()
        # getting data
        if x is not None:
            value = x
            if ind:
                values = np.arange(len(self.x))
            else:
                values = np.array(self.x)
            values2 = np.ma.masked_array(self.y, self.mask)
        else:
            value = y
            if ind:
                values = np.arange(self.y)
            else:
                values = np.ma.masked_array(self.y, self.mask)
            values2 = np.array(self.x)
        # if the wanted value is already present
        if np.any(value == values):
            i_values = values2[np.where(value == values)[0]]
        # if we have to do an interpolation
        else:
            i_values = []
            for ind in np.arange(0, len(values) - 1):
                val_i = values[ind]
                val_ipp = values[ind + 1]
                val2_i = values2[ind]
                val2_ipp = values2[ind + 1]
                if (val_i >= value and val_ipp < value) \
                        or (val_i <= value and val_ipp > value):
                    i_value = ((val2_i*np.abs(val_ipp - value) +
                               val2_ipp*np.abs(values[ind] - value)) /
                               np.abs(values[ind] - val_ipp))
                    i_values.append(i_value)
        # returning
        return i_values

    def get_value_position(self, value, ind=False):
        """
        Return the interpolated position(s) of the wanted value.

        Parameters
        ----------
        value : number
            .
        ind : boolean
           If 'True', return the value indices, else, return the 'y' position.
           (Default is 'False')
        """
        # check parameters
        if not isinstance(value, NUMBERTYPES):
            raise TypeError()
        # if the asked value is present
        if np.any(self.y == value):
            ind_0 = np.argwhere(self.y == value)[0]
            if ind:
                return ind_0
            else:
                pos_0 = [self.x[i] for i in ind_0]
                return pos_0
        # search for positions
        y = self.y
        y[self.mask] = 0
        y = y - value
        x = self.x
        sign = y > 0
        chang = np.abs(np.logical_xor(sign[1::], sign[0:-1:]))
        mask_chang = np.logical_or(self.mask[1::], self.mask[0:-1:])
        chang[mask_chang] = False
        ind_0 = np.argwhere(chang).flatten()
        # get interpolated 0-value position
        val_1 = np.abs(y[ind_0])
        val_2 = np.abs(y[ind_0 + 1])
        ind_0 = np.asarray(ind_0, dtype=float)
        if ind:
            x_1 = ind_0
            x_2 = ind_0 + 1
        else:
            ind_0 = np.asarray(ind_0, dtype=int)
            x_1 = x[ind_0]
            x_2 = x[ind_0 + 1]
        pos_0 = x_1 + val_1/(val_1 + val_2)*(x_2 - x_1)
        # returning
        return pos_0

    def get_integral(self):
        """
        Return the profile integral, and is unit.
        Use the trapezoidal aproximation.
        """
        filt = np.logical_not(self.mask)
        x = self.x[filt]
        y = self.y[filt]
        unit = self.unit_y*self.unit_x
        return np.trapz(y, x)*unit.asNumber(), unit/unit.asNumber()

    def get_gradient(self, position=None, wanted_dx=None):
        """
        Return the profile gradient.
        If 'position' is renseigned, interpolations or finite differences
        are used to get the gradient at x = position.
        Else, a profile with gradient at profile points is returned.
        Warning : only work with evenly spaced x

        Parameters
        ----------

        position : number, optional
            Wanted point position
        wanted_dx : number, optional
            interval on which compute gradient when position is
            renseigned (default is dx similar to axis).
        """
        # check if x values are evenly spaced
        dxs = self.x[1::] - self.x[:-1:]
        dx = self.x[1] - self.x[0]
        if not np.all(np.abs(dxs - dx) < 1e-6*np.max(dxs)):
            raise Exception("Not evenly spaced x values : \n {}".format(dxs))
        if position is None:
            tmp_prof = self.copy()
            tmp_prof.y = np.gradient(self.y, self.x[1] - self.x[0])
            unit = tmp_prof.unit_y/tmp_prof.unit_x
            tmp_prof.y *= unit.asNumber()
            tmp_prof.unit_y = unit/unit.asNumber()
            return tmp_prof
        elif isinstance(position, NUMBERTYPES):
            if wanted_dx is None:
                dx = self.x[1] - self.x[0]
            else:
                dx = wanted_dx
            interp = spinterp.UnivariateSpline(self.x, self.y, k=1, s=0)
            if np.all(position < self.x):
                x = [position, position + dx]
                y = interp(x)
                grad = np.gradient(y, dx)[0]
            elif np.all(position > self.x):
                x = [position - dx, position]
                y = interp(x)
                grad = np.gradient(y, dx)[1]
            else:
                x = [position - dx, position, position + dx]
                y = interp(x)
                grad = np.gradient(y, dx)[1]
            return grad
        else:
            raise TypeError()

    def get_spectrum(self, wanted_x=None, welch_seglen=None,
                     scaling='base', fill='linear', mask_error=True,
                     detrend='constant'):
        """
        Return a Profile object, with the frequential spectrum of 'component',
        on the point 'pt'.

        Parameters
        ----------
        wanted_x : 2x1 array, optional
            Time interval in which compute spectrum (default is all).
        welch_seglen : integer, optional
            If specified, welch's method is used (dividing signal into
            overlapping segments, and averaging periodogram) with the given
            segments length (in number of points).
        scaling : string, optional
            If 'base' (default), result are in component unit.
            If 'spectrum', the power spectrum is returned (in unit^2).
            If 'density', the power spectral density is returned
            (in unit^2/(1/unit_x))
        fill : string or float
            Specifies the way to treat missing values.
            A value for value filling.
            A string ('linear', 'nearest', 'zero', 'slinear', 'quadratic',
            'cubic' where 'slinear', 'quadratic' and 'cubic' refer to a spline
            interpolation of first, second or third order) for interpolation.
        mask_error : boolean
            If 'False', instead of raising an error when masked value appear on
            time profile, '(None, None)' is returned.
        detrend : string, optional
            Method used to detrend the profile. Can be 'none',
            'constant' (default) or 'linear'.

        Returns
        -------
        magn_prof : Profile object
            Magnitude spectrum.
        """
        from scipy.signal import periodogram, welch
        from scipy.signal import detrend as spdetrend
        tmp_prof = self.copy()
        # fill if asked (and if necessary)
        if isinstance(fill, NUMBERTYPES):
            tmp_prof.fill(kind='value', fill_value=fill, inplace=True)
        elif isinstance(fill, STRINGTYPES):
            tmp_prof.fill(kind=fill, inplace=True)
        else:
            raise Exception()
        values = tmp_prof.y - np.mean(tmp_prof.y)
        time = tmp_prof.x
        # detrend
        if detrend == 'constant':
            values = spdetrend(values, type='constant')
        elif detrend == 'linear':
            values = spdetrend(values, type='linear')
        elif detrend == 'none':
            pass
        else:
            raise ValueError()
        # getting spectrum
        fs = 1/(time[1] - time[0])
        if welch_seglen is None or welch_seglen >= len(time):
            if scaling == 'base':
                frq, magn = periodogram(values, fs, scaling='spectrum',
                                        detrend='linear')
                magn = np.sqrt(magn)
            else:
                frq, magn = periodogram(values, fs, scaling=scaling,
                                        detrend='linear')
        else:
            if scaling == 'base':
                frq, magn = welch(values, fs, scaling='spectrum',
                                  nperseg=welch_seglen)
                magn = np.sqrt(magn)
            else:
                frq, magn = welch(values, fs, scaling=scaling,
                                  nperseg=welch_seglen)
        # sretting unit
        if scaling == 'base':
            unit_y = self.unit_y
        elif scaling == 'spectrum':
            unit_y = self.unit_y**2
        elif scaling == 'density':
            unit_y = self.unit_y**2*self.unit_x
        else:
            raise Exception()
        magn_prof = Profile(frq, magn, unit_x=1./self.unit_x,
                            unit_y=unit_y)
        return magn_prof

    def get_wavelet_transform(self, widths=None, fill='linear', raw=False,
                              verbose=False):
        """
        Return the wavelet transformation of the profile.

        Parameters
        ----------
        widths : array of number, optional
            Widths of the wavelet to use (by default use 100 homogeneously
            distributed wavelets)
        fill : string or float
            Specifies the way to treat missing values.
            A value for value filling.
            A string ('linear', 'nearest', 'zero', 'slinear', 'quadratic,
            'cubic' where 'slinear', 'quadratic' and 'cubic' refer to a spline
            interpolation of first, second or third order) for interpolation.
        raw : bool
            if 'True', return an array, else (default), return a ScalarField
            object.
        verbose : boolean
            If 'True', display message on the computing advancement.

        Warning
        -------
        Only work with uniformely spaced data.

        """
        # check
        dx = self.x[1] - self.x[0]
        if widths is None:
            widths = np.linspace(0, len(self.y) - 1, 101)[1::]
        else:
            if not isinstance(widths, ARRAYTYPES):
                raise TypeError()
            widths = np.array(widths)
            if widths.ndim != 1:
                raise ValueError()
            widths = widths/dx
        if np.any((self.x[1::] - self.x[0:-1]) - dx > 1e-6):
            raise ValueError()
        tmp_prof = self.copy()
        # fill if asked (and if necessary)
        if isinstance(fill, NUMBERTYPES):
            tmp_prof.fill(kind='value', fill_value=fill, inplace=True)
        elif isinstance(fill, STRINGTYPES):
            tmp_prof.fill(kind=fill, inplace=True)
        else:
            raise Exception()
        values = tmp_prof.y - np.mean(tmp_prof.y)
        # compute wavelet
        from scipy.signal import cwt, ricker
        wav = cwt(values, ricker, widths)
        # return
        if raw:
            return wav
        else:
            SF = ScalarField()
            new_y = widths*dx
            mask = np.array([self.mask for i in np.arange(len(widths))])
            mask = np.transpose(mask)
            SF.import_from_arrays(self.x, new_y, np.transpose(wav),
                                  mask=mask,
                                  unit_x=self.unit_x, unit_y=self.unit_x,
                                  unit_values=self.unit_y)
            return SF

    def get_pdf(self, bw_method='scott', resolution=1000, raw=False):
        """
        Return the probability density function.

        Parameters
        ----------
        bw_method : str, scalar or callable, optional
            The method used to calculate the estimator bandwidth. This can be
            'scott', 'silverman', a scalar constant or a callable. If a scalar,
            this will be used directly as kde.factor. If a callable, it should
            take a gaussian_kde instance as only parameter and return a scalar.
            If None (default), 'scott' is used.
            See 'scipy.stats.kde' for more details.
        resolution : integer, optional
            Resolution of the returned pdf.
        raw : boolean, optional
            If 'True', return an array, else, return a Profile object.
        """
        # check params
        if not isinstance(resolution, int):
            raise TypeError()
        if resolution < 1:
            raise ValueError()
        if not isinstance(raw, bool):
            raise TypeError()
        # remove masked values
        filt = np.logical_not(self.mask)
        y = self.y[filt]
        y_min = np.min(y)
        y_max = np.max(y)
        # get kernel
        import scipy.stats.kde as spkde
        kernel = spkde.gaussian_kde(y)
        # get values
        pdf_x = np.linspace(y_min, y_max, resolution)
        pdf_y = kernel(pdf_x)
        # returning
        if raw:
            return pdf_y
        else:
            prof = Profile(pdf_x, pdf_y, mask=False, unit_x=self.unit_y,
                           unit_y='')
            return prof

    def get_auto_correlation(self, window_len, raw=False):
        """
        Return the auto-correlation profile.

        This algorithm make auto-correlation for all the possible values,
        and an average of the resulting profile.
        Profile are normalized, so the central value of the returned profile
        should be 1.

        Parameters
        ----------
        window_len : integer
            Window length for sweep correlation.
        raw : bool, optional
            If 'True', return an array
            If 'False' (default), return a profile
        """
        # checking parameters
        if not isinstance(window_len, int):
            raise TypeError()
        if window_len >= len(self.y) - 1:
            raise ValueError()
        window_len = int(np.floor(window_len/2.))
        # loop on possible central values
        corr = np.zeros(2*window_len + 1)
        corr_ad = 0
        nmb = 0
        for i in np.arange(window_len, len(self.y) - window_len - 1):
            central_val = self.y[i]
            surr_vals = self.y[i - window_len:i + window_len + 1]
            tmp_corr = surr_vals*central_val
            corr_ad += central_val**2
            corr += tmp_corr
            nmb += 1
        corr /= corr_ad
        # returning
        if raw:
            return corr
        else:
            dx = self.x[1] - self.x[0]
            x = np.arange(0, dx*len(corr), dx)
            return Profile(x, corr, unit_x=self.unit_x, unit_y=make_unit(''))

    def get_fitting(self, func, p0=None, output_param=False):
        """
        Use non-linear least squares to fit a function, f, to the profile.

        Parameters
        ----------
        func : callable
            The model function, f(x, ...). It must take the independent
            variable as the first argument and the parameters to fit as
            separate remaining arguments.
        p0 : None, scalar, or M-length sequence
            Initial guess for the parameters. If None, then the initial values
            will all be 1 (if the number of parameters for the function can be
            determined using introspection, otherwise a ValueError is raised).
        output_param : boolean, optional
            If 'False' (default), return only a Profile with fitted values
            If 'True', return also the parameters values.

        Returns
        -------
        fit_prof : Profile obect
            The Fitted profile.
        params : tuple, optional
            Fitting parameters.
        """
        # getting data
        xdata = self.x
        ydata = self.y
        if p0 is None:
            nmb_arg = func.__code__.co_argcount
            p0 = [1]*(nmb_arg - 1)

        # minimize function
        def min_func(args, x, y):
            return y - func(x, *args)
        # fitting params
        param, mess = spopt.leastsq(min_func, p0, (xdata, ydata))
        # creating profile
        fit_prof = Profile(xdata, func(xdata, *param), unit_x=self.unit_x,
                           unit_y=self.unit_y, name=self.name)
        # returning
        if output_param:
            return fit_prof, param
        else:
            return fit_prof

    def get_distribution(self, output_format='normalized', resolution=100,
                         bw_method='scott'):
        """
        Return he distribution of y values by using gaussian kernel estimator.

        Parameters
        ----------
        output_format : string, optional
            'normalized' (default) : give position probability (integral egal 1).
            'ponderated' : give position probability ponderated by the number
                           or points (integral egal number of points).
            'concentration' : give local concentration (in point per length).
        resolution : integer
            Resolution of the resulting profile (number of values in it).
        bw_method : str or scalar, optional
            The method used to calculate the estimator bandwidth.
            Can be 'scott', 'silverman' or a number to set manually the
            gaussians std
            (it should aproximately be the size of the density
            node you want to see).
            (see 'scipy.stats.gaussian_kde' documentation for more details)

        Returns
        -------
        distrib : Profile object
            The y values distribution.
        """
        # checking parameters coherence
        if not isinstance(resolution, int):
            raise TypeError()
        if resolution < 2:
            raise ValueError()

        # getting data
        filt = np.logical_not(self.mask)
        x = self.x[filt]
        y = self.y[filt]

        # AD bandwidth
        width_x = np.max(self.x) - np.min(self.x)
        width_y = np.max(self.y) - np.min(self.y)
        if isinstance(bw_method, NUMBERTYPES):
            if width_x > width_y:
                ad_len = width_y
            else:
                ad_len = width_x
            ad_bw_method = bw_method/float(ad_len)
        else:
            ad_bw_method = bw_method

        # getting kernel estimator
        kernel = stats.gaussian_kde(y, bw_method=ad_bw_method)

        # getting distribution
        distrib_x = np.linspace(np.min(y) - np.abs(np.min(y)*kernel.factor/2.),
                                np.max(y) + np.abs(np.max(y)*kernel.factor/2.),
                                resolution)
        distrib_y = kernel(distrib_x)

        # normalizing
        if output_format == "normalized":
            unit_y = make_unit('')
        elif output_format == 'ponderated':
            distrib_y *= len(x)
            unit_y = make_unit('')
        elif output_format == "percentage":
            distrib_y *= 100.
            unit_y = make_unit('')
        elif output_format == "concentration":
            unit_y = 1./self.unit_y
            distrib_y *= len(x)
        else:
            raise ValueError()

        # returning
        distrib = Profile(distrib_x, distrib_y, mask=False, unit_x=self.unit_y,
                          unit_y=unit_y)
        return distrib

    def get_extrema_position(self, smoothing=None, ind=False, debug=False):
        """
        Return the local extrema of the profile.

        Parameters
        ----------
        smoothing : number, optional
            Size of the gaussian smoothing to apply before extrema detection.
        ind : bool, optional
            If 'True', return indice position, else, return position along
            x axis (default is 'False').

        Returns
        -------
        min_pos, max_pos : arrays of numbers
            .
        """
        # get data
        prof = self.copy()
        # smooth if necessary
        if smoothing is not None:
            prof.smooth(tos='gaussian', size=smoothing, inplace=True)
        # get gradients
        grad = prof.get_gradient()
        grad2 = grad.get_gradient()
        tmp_y = grad2.y.copy()
        tmp_y[np.isnan(tmp_y)] = 0
        # gradient computation reduce the number of available points,
        # be sure not to mask valid points
        tmp_y[0:-1] += tmp_y[1::]
        tmp_y[1::] += tmp_y[0:-1]
        max_filt = tmp_y < 0
        grad_max = grad.copy()
        grad_max.mask = np.logical_or(np.logical_not(max_filt), grad.mask)
        grad_min = grad.copy()
        grad_min.mask = np.logical_or(max_filt, grad.mask)
        # get 0-value positions
        pos_max = grad_max.get_value_position(0, ind=ind)
        pos_min = grad_min.get_value_position(0, ind=ind)
        # debug
        if debug:
            plt.figure()
            prof.display(marker='o')
            plt.figure()
            grad.display(marker='o')
            plt.figure()
            grad2.display(marker='o')
            plt.figure()
            grad_min.display(marker='o')
            plt.figure()
            grad_max.display(marker='o')
            print(pos_min, pos_max)
            plt.show()
        # returning
        return pos_min, pos_max

    def get_convolution(self, other_prof, mode='full'):
        """
        Return the convolution with the give profile.

        Parameters
        ----------
        other_prof : Profile object
            .
        mode : {‘full’, ‘valid’, ‘same’}, optional
            ‘full’:
                By default, mode is ‘full’. This returns the convolution at
                each point of overlap, with an output shape of (N+M-1,). At
                the end-points of the convolution, the signals do not overlap
                completely, and boundary effects may be seen.
            ‘same’:
                Mode same returns output of length max(M, N). Boundary effects
                are still visible.
            ‘valid’:
                Mode valid returns output of length max(M, N) - min(M, N) + 1.
                The convolution product is only given for points where the
                signals overlap completely. Values outside the signal boundary
                have no effect.

        Returns
        -------
        conv_prof : Profile object
            Result of the convolution

        Notes
        -----
        Use the numpy function 'convolve'.
        """
        # check
        if not isinstance(other_prof, Profile):
            raise TypeError()
        if not isinstance(mode, STRINGTYPES):
            raise TypeError()
        if mode not in ['full', 'same', 'valid']:
            raise ValueError()
        if not np.all(self.x[1:] - self.x[0:-1] - (self.x[1] - self.x[0]) <
                      self.x[-1]*1e-10):
            raise Exception("Profiles should have orthogonal x axis")
        if not np.all(other_prof.x[1:] - other_prof.x[0:-1] -
                      other_prof.x[1] - other_prof.x[0] <
                      1e-6*np.mean(other_prof.x)):
            raise Exception("Profiles should have orthogonal x axis")
        if not self.x[1] - self.x[0] == other_prof.x[1] - other_prof.x[0]:
            raise Exception("Profiles should have same x discretization step")
        if not self.unit_x == other_prof.unit_x:
            raise Exception("Profiles should have the same x unit")
        dx = self.x[1] - self.x[0]
        # get convolution
        conv = np.convolve(self.y, other_prof.y, mode=mode)
        # store in a profile
        conv_prof = Profile(np.arange(0, dx*len(conv), dx),
                            conv, unit_x=self.unit_x,
                            unit_y=self.unit_y*other_prof.unit_y)
        # return
        return conv_prof

    def get_dephasage(self, other_profile, conv='difference'):
        """
        Return the dephasage between the two profiles using convolution

        Parameters
        ----------
        conv : string in ['classic', 'difference']
            The convection type to use

        Returns
        -------
        dep : number
            Dephasage, in profiles unit
        """
        # TODO : Repair 'classic'
        if conv == 'classic':
            raise Exception("'classic' is broken for now...")
            tmp_conv = self.get_convolution(other_profile, mode='same')
            _, maxs = tmp_conv.get_extrema_position()
            ind_closer = np.argmin(np.abs(maxs - (len(tmp_conv) + 1)/2.))
            tmp_deph = ((len(tmp_conv) + 1)/2. - maxs[ind_closer])/2
        elif conv == 'difference':
            tmp_conv = self.get_convolution_of_difference(other_profile,
                                                          normalized=True)
            tmp_deph = tmp_conv.get_value_position(tmp_conv.min)[0]
        return tmp_deph

    def get_convolution_of_difference(self, other_profile, normalized=True):
        """
        Return a convolution that use difference instead of multiplication.

        Note
        ----
        Difference is not normaized, but averaged on the available points.
        """
        # TODO : change the returned x axis so that it corepsond to the
        #        depahsage
        # WARNING : 'get_dephasage' strongly depend on this function,
        #           do ot change thing here without making the apropriate
        #           changes in 'get_sephasage'
        # check
        if not self.unit_x == other_profile.unit_x:
            raise ValueError()
        if not self.unit_y == other_profile.unit_y:
            raise ValueError()
        # get data
        y_short = self.y
        y_long_o = other_profile.y
        y_short_mask = self.mask
        y_long_mask_o = other_profile.mask
        x = other_profile.x - other_profile.x[0]
        len_max = len(y_long_o)
        len_min = len(y_short)
        # create elongated long profile
        y_long = np.zeros((len_max + 2*len_min - 2), dtype=float)
        y_long_mask = np.zeros((len_max + 2*len_min - 2), dtype=bool)
        y_long[len_min - 1:len_max + len_min - 1] = y_long_o
        y_long_mask[len_min - 1:len_max + len_min - 1] = y_long_mask_o
        # calculate diff for each shift
        diffs = []
        for i in range(len_max + len_min - 1):
            tmp_y_long_mask = y_long_mask[i:i+len_min]
            tmp_y_short_mask = y_short_mask
            tmp_filter = np.logical_not(np.logical_or(tmp_y_long_mask,
                                                      tmp_y_short_mask))
            if not np.any(tmp_filter):
                diffs.append(np.NaN)
                continue
            tmp_y_short = y_short[tmp_filter]
            tmp_y_long = y_long[i:i+len_min][tmp_filter]
            diff = np.sum(np.abs(tmp_y_long - tmp_y_short))/len(tmp_y_long)
            # normalize
            if normalized:
                norm = np.sum(np.abs(tmp_y_long))/len(tmp_y_long)
                if norm == 0:
                    diff = np.NaN
                else:
                    diff /= norm
            diffs.append(diff)
        # compute x axis
        dx = x[1] - x[0]
        x = np.arange(0, len(diffs)*dx, dx)[0:len(diffs)]
        x -= (x[-1] + x[0])/2.
        delta_x = ((other_profile.x[0] + other_profile.x[-1])/2. -
                   (self.x[0] + self.x[-1])/2.)
        x += delta_x
        # returning
        return Profile(x, diffs, mask=np.isnan(diffs), unit_x=self.unit_x,
                       unit_y=self.unit_y)

    def spectral_filtering(self, fmin=None, fmax=None, order=2):
        """
        Perform a spectral filtering (highpass, lowpass, bandpass).

        Parameters
        ----------
        fmin, fmax : numbers
            Minimal and maximal frequencies
        order : integer, optional
            Butterworth filter order

        Returns
        -------
        filt_prof : Profile object
            Filtered profile
        """
        # Interpolate missing values
        tmp_prof = self.fill("linear", inplace=False)
        # remove trend
        mean_y = np.mean(tmp_prof.y)
        tmp_prof.y -= mean_y
        # define butterworth filter
        fs = len(tmp_prof.x)/(tmp_prof.x[-1] - tmp_prof.x[0])
        nyq = 0.5 * fs
        low = fmin / nyq
        high = fmax / nyq
        b, a = spsign.butter(order, [low, high], btype='band')
        # Apply filter to data
        y = spsign.lfilter(b, a, tmp_prof.y)
        # Readd average
        y += mean_y
        # return
        return Profile(tmp_prof.x, y, mask=tmp_prof.mask,
                       unit_x=tmp_prof.unit_x,
                       unit_y=tmp_prof.unit_y)

    def add_point(self, x, y):
        """
        Add the given point to the profile.
        """
        pos_ind = np.searchsorted(self.x, x)
        self.y = np.concatenate((self.y[0:pos_ind], [y], self.y[pos_ind::]))
        self.x = np.concatenate((self.x[0:pos_ind], [x], self.x[pos_ind::]))

    def add_points(self, prof):
        """
        Add points from another profile.
        """
        # if self is an empty profile
        if len(self) == 0:
            self.unit_x = prof.unit_x
            self.unit_y = prof.unit_y
        # check unities
        if (prof.unit_x != self.unit_x or prof.unit_y != self.unit_y):
                raise ValueError()
        # append daa
        tmp_x = np.append(self.x, prof.x)
        tmp_y = np.append(self.y, prof.y)
        tmp_mask = np.append(self.mask, prof.mask)
        # sort by time
        ind_sort = np.argsort(tmp_x)
        self.__x = tmp_x[ind_sort]
        self.__y = tmp_y[ind_sort]
        self.mask = tmp_mask[ind_sort]

    def remove_point(self, ind):
        """
        Remove a point from the profile

        Parameters
        ----------
        ind : integer
            Idice of the point to remove

        """
        # check
        if not isinstance(ind, int):
            raise TypeError()
        if not ind >= 0:
            raise ValueError()
        # remove the point
        self.y = np.concatenate((self.y[0:ind], self.y[ind + 1::]))
        self.x = np.concatenate((self.x[0:ind], self.x[ind + 1::]))

    def crop_masked_border(self, inplace=False):
        """
        Remove the masked values at the border of the profile in place or not.
        """
        if inplace:
            tmp_prof = self
        else:
            tmp_prof = self.copy()
        mask = tmp_prof.mask
        inds_not_masked = np.where(np.logical_not(mask))[0]
        first = inds_not_masked[0]
        last = inds_not_masked[-1] + 1
        tmp_prof.crop([first, last], ind=True, inplace=True)
        # returning
        if not inplace:
            return tmp_prof

    def crop(self, intervx=None, intervy=None, ind=False, inplace=False):
        """
        Crop the profile along 'x'.

        Parameters
        ----------
        intervx : array of two numbers
            Bound values of x.
        intervy : array of two numbers
            Bound values of y.
        ind : Boolean, optionnal
            If 'False' (Default), 'intervx' and 'intervy' are values along x
            axis, if 'True', 'intervx' and 'intervy' are indices of values
            along x.
        inplace : boolean, optional
            .
        """
        # checking parameters coherence
        if intervx is not None:
            if not isinstance(intervx, ARRAYTYPES):
                raise TypeError("'intervx' must be an array")
            intervx = np.array(intervx)
            if not intervx.shape == (2,):
                raise ValueError("'intervx' must be an array with only two"
                                 "values")
            if intervx[0] >= intervx[1]:
                raise ValueError("'intervx' values must be crescent")
        if intervy is not None:
            if not isinstance(intervy, ARRAYTYPES):
                raise TypeError("'intervy' must be an array")
            intervy = np.array(intervy)
            if not intervy.shape == (2,):
                raise ValueError("'intervy' must be an array with only two"
                                 "values")
            if intervy[0] >= intervy[1]:
                raise ValueError("'intervy' values must be crescent")
        new_x = self.x.copy()
        new_y = self.y.copy()
        new_mask = self.mask.copy()
        # treat intervx with ind=False
        if not ind and intervx is not None:
            if all(intervx < np.min(self.x))\
                    or all(intervx > np.max(self.x)):
                raise ValueError("'intervx' values are out of profile")
            ind1 = 0
            ind2 = -1
            for i in np.arange(len(new_x)-1, 0, -1):
                if new_x[i] == intervx[0]:
                    ind1 = i
                elif new_x[i] == intervx[1]:
                    ind2 = i + 1
                elif (new_x[i] > intervx[0] and new_x[i-1] < intervx[0]) \
                        or (new_x[i] < intervx[0] and
                            new_x[i-1] > intervx[0]):
                    ind1 = i + 1
                elif (new_x[i] > intervx[1] and new_x[i-1] < intervx[1]) \
                        or (new_x[i] < intervx[1] and
                            new_x[i-1] > intervx[1]):
                    ind2 = i
            indices = [ind1, ind2]
            new_x = new_x[indices[0]:indices[1]]
            new_y = new_y[indices[0]:indices[1]]
            new_mask = new_mask[indices[0]:indices[1]]
        # treat intervy with ind=False
        if not ind and intervy is not None:
            filt = np.logical_and(self.y > intervy[0], self.y < intervy[1])
            new_x = new_x[filt]
            new_y = new_y[filt]
            new_mask = new_mask[filt]
        # treat intervx with ind=True
        if ind and intervx is not None:
            intervx = np.array(intervx, dtype=int)
            if any(intervx < 0) or any(intervx > len(self.x)):
                raise ValueError("'intervx' indices are out of profile")
            new_x = self.x[intervx[0]:intervx[1]]
            new_y = self.y[intervx[0]:intervx[1]]
            new_mask = self.mask[intervx[0]:intervx[1]]
        # treat intervy with ind=True
        if ind and intervy is not None:
            raise ValueError("Specifying 'intervy' with indices has no sens")
        # return
        if inplace:
            self.y = new_y
            self.x = new_x
            self.mask = new_mask
            return None
        else:
            tmp_prof = Profile(new_x, new_y, new_mask, self.unit_x,
                               self.unit_y)
            return tmp_prof

    def scale(self, scalex=1., scaley=1., inplace=False):
        """
        Change the scale of the axis.

        Parameters
        ----------
        scalex, scaley : numbers or Unum objects
            scales along x and y
        inplace : boolean, optional
            If 'True', scaling is done in place, else, a new instance is
            returned.
        """
        # check params
        if not isinstance(scalex, NUMBERTYPES + (unum.Unum, )):
            raise TypeError()
        if not isinstance(scaley, NUMBERTYPES + (unum.Unum, )):
            raise TypeError()
        if not isinstance(inplace, bool):
            raise TypeError()
        if inplace:
            tmp_prof = self
        else:
            tmp_prof = self.copy()
        # adapt unit
        if isinstance(scalex, unum.Unum):
            new_unit = scalex*tmp_prof.unit_x
            fact = new_unit.asNumber()
            new_unit /= fact
            tmp_prof.unit_x = new_unit
            scalex = fact
        if isinstance(scaley, unum.Unum):
            new_unit = scaley*tmp_prof.unit_y
            fact = new_unit.asNumber()
            new_unit /= fact
            tmp_prof.unit_y = new_unit
            scaley = fact
        # loop
        if scalex != 1.:
            tmp_prof.x *= scalex
        if scaley != 1.:
            tmp_prof.y *= scaley
        # returning
        if not inplace:
            return tmp_prof

    def rotate(self, angle, inplace=False):
        """
        Rotate the profile.

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
        x, y = tmpp.x, tmpp.y
        angle *= -1
        new_x = x*np.cos(angle) - y*np.sin(angle)
        new_y = y*np.cos(angle) + x*np.sin(angle)
        tmpp.y = new_y
        tmpp.x = new_x
        #
        return tmpp

    def fill(self, kind='slinear', fill_value=0., inplace=False, crop=False):
        """
        Return a filled profile (no more masked values).

        Warning : If 'crop' is False, border masked values can't be
        interpolated and are filled with 'fill_value' or the nearest value.

        Parameters
        ----------
        kind : string or int, optional
            Specifies the kind of interpolation as a string ('value', 'linear',
            'nearest', 'zero', 'slinear', 'quadratic, 'cubic' where 'slinear',
            'quadratic' and 'cubic' refer to a spline interpolation of first,
            second or third order) or as an integer specifying the order of
            the spline interpolator to use. Default is 'linear'.
        fill_value : number, optional
            For kind = 'value', filling value.
        inplace : boolean, optional
            .
        crop : boolean, optional
            .

        Returns
        -------
        prof : Profile object
            Filled profile
        """
        # check if filling really necessary
        if not np.any(self.mask) and inplace:
            return None,
        elif not np.any(self.mask):
            return self.copy()
        # crop if asked
        if crop:
            self.crop_masked_border(hard=False, inplace=True)
        # get mask
        mask = self.mask
        filt = np.logical_not(mask)
        if np.all(mask):
            raise Exception("There is no values on this profile")
        # check fill type
        if kind == 'value':
            new_y = copy.copy(self.y)
            new_y[filt] = fill_value
        else:
            # making interpolation on existent values
            x = self.x[filt]
            y = self.y[filt]
            interp = spinterp.interp1d(x, y, kind=kind,
                                       bounds_error=False,
                                       fill_value=fill_value)
            # replacing missing values
            new_y = copy.copy(self.y)
            missing_x = self.x[mask]
            new_y[mask] = interp(missing_x)
            # replacing border value by nearest value
            inds_masked = np.where(np.logical_not(mask))[0]
            first = inds_masked[0]
            last = inds_masked[-1]
            new_y[0:first] = new_y[first]
            new_y[last + 1::] = new_y[last]
        # returning
        if inplace:
            self.y = new_y
            self.mask = False
        else:
            tmp_prof = Profile(self.x, new_y, mask=False, unit_x=self.unit_x,
                               unit_y=self.unit_y, name=self.name)
            return tmp_prof

    def augment_resolution(self, fact=2, interp='linear', inplace=True):
        """
        Augment the temporal resolution of the profile.

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
        # check parameters
        if not isinstance(fact, int):
            raise TypeError()
        if fact <= 0:
            raise TypeError()
        if not isinstance(interp, STRINGTYPES):
            raise TypeError()
        if interp not in ['linear', 'nearest', 'zero', 'slinear', 'quadratic',
                          'cubic']:
            raise ValueError()
        if not isinstance(inplace, bool):
            raise TypeError()
        # get data
        if inplace:
            tmp_prof = self
        else:
            tmp_prof = self.copy()
        tmp_prof.crop_masked_border(inplace=True)
        filt = np.logical_not(tmp_prof.mask)
        # interpolate using scipy
        old_inds = np.arange(0, len(tmp_prof.x)*fact, fact)
        new_inds = np.arange(old_inds[-1] + 1)
        interp_x = spinterp.interp1d(old_inds[filt], tmp_prof.x[filt],
                                     kind=interp, assume_sorted=True)
        new_x = interp_x(new_inds)
        interp_y = spinterp.interp1d(old_inds[filt], tmp_prof.y[filt],
                                     kind=interp, assume_sorted=True)
        new_y = interp_y(new_inds)
        # return
        tmp_prof.y = new_y
        tmp_prof.x = new_x
        tmp_prof.mask = np.zeros(len(new_x), dtype=bool)
        if not inplace:
            return tmp_prof

    def change_unit(self, axe, new_unit):
        """
        Change the unit of an axe.

        Parameters
        ----------
        axe : string
            'y' for changing the profile values unit
            'x' for changing the profile axe unit
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
            self.x *= fact
            self.unit_x = new_unit/fact
        elif axe == 'y':
            old_unit = self.unit_y
            new_unit = old_unit.asUnit(new_unit)
            fact = new_unit.asNumber()
            self.y *= fact
            self.unit_y = new_unit/fact
        else:
            raise ValueError()

    def evenly_space(self, kind_interpolation='linear', dx=None, inplace=False):
        """
        Return a profile with evenly spaced x values.
        Use interpolation to get missing values.

        Parameters
        ----------
        kind_interpolation : string or int, optional
            Specifies the kind of interpolation as a string ('value', 'linear',
            'nearest', 'zero', 'slinear', 'quadratic, 'cubic' where 'slinear',
            'quadratic' and 'cubic' refer to a spline interpolation of first,
            second or third order) or as an integer specifying the order of
            the spline interpolator to use. Default is 'linear'.
        """
        if inplace:
            tmpp = self
        else:
            tmpp = self.copy()
        # checking if evenly spaced
        dxs = self.x[1::] - self.x[:-1:]
        dxi = self.x[1] - self.x[0]
        if np.all(np.abs(dxs - dxi) < 1e-6*np.max(dxs)):
            return self.copy()
        # getting data
        mask = self.mask
        filt = np.logical_not(mask)
        x = self.x[filt]
        y = self.y[filt]
        if dx is None:
            mean_dx = np.average(dxs)
        else:
            mean_dx = dx
        min_x = np.min(self.x)
        max_x = np.max(self.x)
        nmb_interv = np.int((max_x - min_x)/mean_dx)
        # creating new evenly spaced x axis
        new_x = np.linspace(min_x, max_x, nmb_interv + 1)
        # interpolate to obtain y values
        interp = spinterp.interp1d(x, y, kind_interpolation)
        new_y = interp(new_x[1:-1])
        new_y = np.concatenate(([self.y[0]], new_y, [self.y[-1]]))
        # return profile
        tmpp.__init__(new_x, new_y, mask=False, unit_x=self.unit_x,
                      unit_y=self.unit_y, name=self.name)
        return tmpp

    def smooth(self, tos='uniform', size=None, direction='y',
               inplace=False, **kw):
        """
        Return a smoothed profile.
        Warning : fill up the field

        Parameters
        ----------
        tos : string, optional
            Type of smoothing, can be 'uniform' (default) or 'gaussian'
            (See ndimage module documentation for more details)
        size : number, optional
            Size of the smoothing (is radius for 'uniform' and
            sigma for 'gaussian').
            Default is 3 for 'uniform' and 1 for 'gaussian'.
        dir : string, optional
            In which direction smoothing (can be 'x', 'y' or 'xy').
        inplace : boolean
            If 'False', return a smoothed profile
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
        if direction not in ['x', 'y', 'xy']:
            raise ValueError()
        # default smoothing border mode to 'nearest'
        if 'mode' not in list(kw.keys()):
            kw.update({'mode': 'nearest'})
        # getting data
        if inplace:
            self.fill(inplace=True)
            y = self.y
            x = self.x
        else:
            tmp_prof = self.copy()
            tmp_prof.fill(inplace=True)
            y = tmp_prof.y
            x = tmp_prof.x
        # smoothing
        if tos == "uniform":
            if direction == 'y':
                y = ndimage.uniform_filter(y, size, **kw)
            if direction == 'x':
                x = ndimage.uniform_filter(x, size, **kw)
            if direction == 'xy':
                x = ndimage.uniform_filter(x, size, **kw)
                y = ndimage.uniform_filter(y, size, **kw)
        elif tos == "gaussian":
            if direction == 'y':
                y = ndimage.gaussian_filter(y, size, **kw)
            if direction == 'x':
                x = ndimage.gaussian_filter(x, size, **kw)
            if direction == 'xy':
                x = ndimage.gaussian_filter(x, size, **kw)
                y = ndimage.gaussian_filter(y, size, **kw)
        else:
            raise ValueError("'tos' must be 'uniform' or 'gaussian'")
        # storing
        if inplace:
            self.x = x
            self.y = y
        else:
            tmp_prof.x = x
            tmp_prof.y = y
            return tmp_prof

    def remove_doublons(self, method='average', inplace=False, eps_rel=1e-6):
        """
        Replace values associated to the same 'x' by their average.

        Parameters
        ----------
        method : string in {'average', 'max', 'min'}
           Method used to remove the doublons.
        """
        if inplace:
            tmp_prof = self
        else:
            tmp_prof = self.copy()
        ord_magn = (np.sum(tmp_prof.x**2)/len(tmp_prof.x))**.5
        nmb_dec = -int(round(np.log10(ord_magn*eps_rel)))
        tmp_x = np.round(tmp_prof.x, decimals=nmb_dec)
        new_x = np.sort(list(set(tmp_x)))
        if method == 'average':
            new_y = [np.mean(tmp_prof.y[tmp_x == xi]) for xi in new_x]
        elif method == 'min':
            new_y = [np.min(tmp_prof.y[tmp_x == xi]) for xi in new_x]
        elif method == 'max':
            new_y = [np.max(tmp_prof.y[tmp_x == xi]) for xi in new_x]
        else:
            raise ValueError()
        tmp_prof.y = new_y
        tmp_prof.x = new_x
        # returning
        if not inplace:
            return tmp_prof

    def remove_nans(self, inplace=False):
        """
        Remove the NaNs points from the profile.
        """
        if inplace:
            tmp_p = self
        else:
            tmp_p = self.copy()
        #
        mask = np.isnan(tmp_p.y)
        if np.any(mask):
            tmp_p.y = tmp_p.y[~mask]
            tmp_p.x = tmp_p.x[~mask]
        #
        return tmp_p


    def remove_marginal_values(self, fact=5, inplace=False):
        """
        Remove (mask) the marginal values.

        Parameters
        ----------
        fact : positive number
            Number of standard deviation in the 'acceptable' value range.
            (default to 5)
        """
        if inplace:
            tmp_p = self
        else:
            tmp_p = self.copy()
        # Mask marginal values based on standard deviation
        mean = tmp_p.mean
        std = np.std(tmp_p.y[~tmp_p.mask])
        tmp_p.mask[tmp_p.y < mean - fact*std] = True
        tmp_p.mask[tmp_p.y > mean + fact*std] = True
        # Return
        return tmp_p

    def remove_local_marginal_values(self, fact=5, neighb=20, inplace=False):
        """
        Remove (mask) the local marginal values.

        Parameters
        ----------
        fact : positive number
            Number of standard deviation in the 'acceptable' value range.
            (default to 5)
        neighb: positive number
            Size of the neighbourhood to consider to check if a value
            if marginal or not.
        """
        # check
        if inplace:
            tmp_p = self
        else:
            tmp_p = self.copy()
        # get smoothed version
        prof_smooth = tmp_p.smooth(tos='gaussian', size=neighb, inplace=False)
        # get difference from smoothed version
        prof_diff = tmp_p - prof_smooth
        # get std of diff
        mean = np.mean(prof_diff.y[~prof_diff.mask])
        std = np.std(prof_diff.y[~prof_diff.mask])
        # mask based on the stadard deviation of the diff
        tmp_p.mask[prof_diff.y < mean - fact*std] = True
        tmp_p.mask[prof_diff.y > mean + fact*std] = True
        # return
        return tmp_p

    def _display(self, kind='plot', reverse=False, **plotargs):
        """
        Private Displayer.
        Just display the curve, not axes and title.

        Parameters
        ----------
        kind : string
            Kind of display to plot ('plot', 'semilogx', 'semilogy', 'loglog')
        reverse : Boolean, optionnal
            If 'False', x is put in the abscissa and y in the ordinate. If
            'True', the inverse.
        color : string, number or array of numbers
            Color of the line (can be an array for evolutive color)
        color_label : string
            Label for the colorbar if color is an array
        **plotargs : dict, optionnale
            Additional argument for the 'plot' command.

        Returns
        -------
        fig : Plot reference
            Reference to the displayed plot.
        """
        if not reverse:
            x = self.x.copy()
            y = self.y.copy()
            if any(self.mask):
                y[self.mask] = np.nan
        else:
            x = self.y.copy()
            y = self.x.copy()
            if any(self.mask):
                x[self.mask] = np.nan
        # if not reverse:
        #     x = self.x[~self.mask]
        #     y = self.y[~self.mask]
        # else:
        #     x = self.y[~self.mask]
        #     y = self.x[~self.mask]
        # check if color is an array
        if 'color_label' in list(plotargs.keys()):
            color_label = plotargs.pop('color_label')
        else:
            color_label = ''
        if 'color' in list(plotargs.keys()):
            if isinstance(plotargs['color'], ARRAYTYPES):
                if len(plotargs['color']) == len(x):
                    color = plotargs.pop('color')
                    plot = pplt.colored_plot(x, y, z=color, log=kind,
                                             color_label=color_label,
                                             **plotargs)
                    pplt.DataCursorPoints(plot, x, y)
                    return plot
        # check log error
        ind_to_del = np.zeros(len(x), dtype=bool)
        if kind in ['semilogx', 'loglog'] and np.any(x <= 0):
            ind_to_del = np.logical_or(ind_to_del, x <= 0)
        elif kind in ['semilogy', 'loglog'] and np.any(y <= 0):
            ind_to_del = np.logical_or(ind_to_del, y <= 0)
        ind_to_keep = np.logical_not(ind_to_del)
        x = x[ind_to_keep]
        y = y[ind_to_keep]
        # display normal plot
        dp = pplt.Displayer(x=x, y=y, kind=kind, **plotargs)
        plot = dp.draw()
        return plot

    def display(self, kind='plot', reverse=False, **plotargs):
        """
        Display the profile.

        Parameters
        ----------
        reverse : Boolean, optionnal
            If 'False', x is put in the abscissa and y in the ordinate. If
            'True', the inverse.
        kind : string
            Kind of display to plot ('plot', 'semilogx', 'semilogy', 'loglog')
        **plotargs : dict, optionnale
            Additional argument for the 'plot' command.

        Returns
        -------
        fig : Plot reference
            Reference to the displayed plot.
        """
        plot = self._display(kind, reverse, **plotargs)
        plt.title(self.name)
        if not reverse:
            plt.xlabel("{0}".format(self.unit_x.strUnit()))
            plt.ylabel("{0}".format(self.unit_y.strUnit()))
        else:
            plt.xlabel("{0}".format(self.unit_y.strUnit()))
            plt.ylabel("{0}".format(self.unit_x.strUnit()))
        return plot
