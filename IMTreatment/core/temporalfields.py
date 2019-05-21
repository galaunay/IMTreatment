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
from ..utils import make_unit, ProgressCounter
from ..utils.types import ARRAYTYPES, INTEGERTYPES, NUMBERTYPES, STRINGTYPES
from . import field as fld
from . import fields as flds
from . import scalarfield as sf
from . import points as pts
from . import profile as prof
from . import vectorfield as vf


class TemporalFields(flds.Fields, fld.Field):
    """
    Class representing a set of time evolving fields.
    All fields added to this object has to have the same axis system.
    """

    def __init__(self):
        fld.Field.__init__(self)
        flds.Fields.__init__(self)
        self.fields = []
        self.__times = np.array([], dtype=float)
        self.__unit_times = make_unit("")
        self.field_type = None
        self.axe_x = []
        self.axe_y = []
        self.unit_x = make_unit('')
        self.unit_y = make_unit('')
        self.unit_times = make_unit('')

    def __add__(self, other):
        if isinstance(other, self.fields[0].__class__):
            tmp_TF = self.copy()
            for i in np.arange(len(tmp_TF.fields)):
                tmp_TF.fields[i] += other
            return tmp_TF
        elif isinstance(other, self.__class__):
            tmp_tf = self.copy()
            if np.all(self.times == other.times):
                for i in np.arange(len(self.fields)):
                    tmp_tf.fields[i] += other.fields[i]
            else:
                for i in np.arange(len(other.fields)):
                    tmp_tf.add_field(other.fields[i])
            return tmp_tf
        else:
            tmp_TF = self.copy()
            for i in np.arange(len(tmp_TF.fields)):
                tmp_TF.fields[i] += other
            return tmp_TF
        # else:
        #     raise TypeError("cannot concatenate {} with"
        #                     " {}.".format(self.__class__, type(other)))

    def __sub__(self, other):
        return self.__add__(-other)

    def __neg__(self):
        tmp_tf = self.copy()
        for i in np.arange(len(self.fields)):
            tmp_tf.fields[i] = -tmp_tf.fields[i]
        return tmp_tf

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            if not len(self) == len(other):
                raise Exception()
            if not np.all(self.axe_x == other.axe_x) \
                    and np.all(self.axe_y == other.axe_y):
                raise Exception()
            if not np.all(self.times == other.times):
                raise Exception()
            vfs = self.__class__()
            for i in np.arange(len(self.fields)):
                vfs.add_field(self.fields[i]*other.fields[i])
            return vfs
        elif isinstance(other, (NUMBERTYPES, unum.Unum)):
            final_vfs = self.__class__()
            for field in self.fields:
                final_vfs.add_field(field*other)
            return final_vfs
        else:
            raise TypeError("You can only multiply a temporal velocity field "
                            "by numbers")

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            if not len(self) == len(other):
                raise Exception()
            if not np.all(self.axe_x == other.axe_x) \
                    and np.all(self.axe_y == other.axe_y):
                raise Exception()
            if not np.all(self.times == other.times):
                raise Exception()
            vfs = self.__class__()
            for i in np.arange(len(self.fields)):
                vfs.add_field(self.fields[i]/other.fields[i])
            return vfs
        elif isinstance(other, self.fields[0].__class__):
            if not np.all(self.axe_x == other.axe_x) \
                    and np.all(self.axe_y == other.axe_y):
                raise Exception()
            vfs = self.__class__()
            for i in np.arange(len(self.fields)):
                vfs.add_field(self.fields[i]/other)
            return vfs
        elif isinstance(other, (NUMBERTYPES, unum.Unum)):
            final_vfs = self.__class__()
            for i, field in enumerate(self.fields):
                final_vfs.add_field(other, time=self.times[i],
                                    unit_times=self.unit_times)
            return final_vfs
        else:
            raise TypeError("")

    __div__ = __truediv__

    def __pow__(self, number):
        if not isinstance(number, NUMBERTYPES):
            raise TypeError("You only can use a number for the power "
                            "on a Vectorfield")
        final_vfs = self.__class__()
        for field in self.fields:
            final_vfs.add_field(np.power(field, number))
        return final_vfs

    def __iter__(self):
        for i in np.arange(len(self.fields)):
            yield self.times[i], self.fields[i]

    def __eq__(self, obj):
        if not isinstance(obj, self.__class__):
            return False
        if not fld.Field.__eq__(self, obj):
            return False
        if not flds.Fields.__eq__(self, obj):
            return False
        for attr in ['fields', 'times', 'axe_x', 'axe_y']:
            if not np.all(self.__getattribute__(attr) ==
                          obj.__getattribute__(attr)):
                return False
        for attr in ['field_type', 'unit_x', 'unit_y', 'unit_times']:
            if not (self.__getattribute__(attr) ==
                    obj.__getattribute__(attr)):
                return False
        return True

    @fld.Field.axe_x.setter
    def axe_x(self, value):
        fld.Field.axe_x.fset(self, value)
        for field in self.fields:
            field.axe_x = value
            field.xy_scale = self.xy_scale

    @fld.Field.axe_y.setter
    def axe_y(self, value):
        fld.Field.axe_y.fset(self, value)
        for field in self.fields:
            field.axe_y = value
            field.xy_scale = self.xy_scale

    @fld.Field.unit_x.setter
    def unit_x(self, value):
        fld.Field.unit_x.fset(self, value)
        for field in self.fields:
            field.unit_x = value

    @fld.Field.unit_y.setter
    def unit_y(self, value):
        fld.Field.unit_y.fset(self, value)
        for field in self.fields:
            field.unit_y = value

    @property
    def mask(self):
        dim = (len(self.fields), self.shape[0], self.shape[1])
        mask_f = np.empty(dim, dtype=bool)
        for i, field in enumerate(self.fields):
            mask_f[i, :, :] = field.mask[:, :]
        return mask_f

    @property
    def mask_as_sf(self):
        dim = len(self.fields)
        mask_f = np.empty(dim, dtype=object)
        for i, field in enumerate(self.fields):
            mask_f[i] = field.mask_as_sf
        return mask_f

    @property
    def mask_cum(self):
        cum_mask = np.sum(self.mask, axis=0)
        cum_mask = cum_mask == len(self.mask)
        return cum_mask

    @property
    def mask_cum_as_sf(self):
        cum_mask = self.mask_cum
        tmp_sf = sf.ScalarField()
        tmp_sf.import_from_arrays(self.axe_x, self.axe_y, cum_mask, mask=None,
                                  unit_x=self.unit_x, unit_y=self.unit_y,
                                  unit_values='')
        return tmp_sf

    @property
    def times(self):
        return self.__times

    @times.setter
    def times(self, values):
        if not isinstance(values, ARRAYTYPES):
            raise TypeError()
        if len(self.fields) != len(values):
            raise ValueError("New number of time ({}) do not corespond to "
                             "the number of fields ({})"
                             .format(len(values), len(self.fields)))
        self.__times = values

    @times.deleter
    def times(self):
        raise Exception("Nope, can't do that")

    @property
    def dt(self):
        return self.times[1] - self.times[0]

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

    @property
    def unit_values(self):
        if len(self.fields) != 0:
            return self[0].unit_values

    def get_mean_field(self, nmb_min=1, dtype=None):
        """
        Calculate the mean velocity field, from all the fields.

        Parameters
        ----------
        nmb_min : integer, optional
            Minimum number of values used to make a mean. else, the value is
            masked
        dtype : type
            Specify the output values type (default to the same one as fields).
        """
        # checks
        if len(self.fields) == 0:
            raise ValueError("There is no fields in this object")
        if self.field_type == vf.VectorField:
            value = [0., 0.]
        else:
            value = 0.
        if not dtype:
            dtype = self.fields[0]._values_dtype
        #
        result_f = self.fields[0].copy()
        result_f.change_dtype(float)
        result_f.fill(kind='value', value=value, crop=False, inplace=True)
        mask_cum = np.zeros(self.shape, dtype=int)
        mask_cum[np.logical_not(self.fields[0].mask)] += 1
        i = 0
        for field in self.fields[1::]:
            i += 1
#            print("{},  {}".format(i, field.unit_values))
            added_field = field.copy()
            added_field.fill(kind='value', value=0., inplace=True)
            result_f += added_field
            mask_cum[np.logical_not(field.mask)] += 1
        mask = mask_cum <= nmb_min
        result_f.mask = mask
        result_f.change_dtype(dtype)
        fact = mask_cum
        fact[mask] = 1
        result_f /= fact
        result_f.xy_scale = self.fields[0].xy_scale
        return result_f

    def get_interpolated_field(self, time):
        """
        Return the interpolated field happening at the time 'time'.
        """
        # check
        assert isinstance(time, NUMBERTYPES)
        assert time >= self.times[0]
        assert time <= self.times[-1]
        # if time is in self.times
        if np.any(self.times == time):
            return self.fields[self.times == time][0]
        # else, get the surrounding fields
        ind_time = np.argwhere(self.times > time)[0][0]
        denom = self.times[ind_time] - self.times[ind_time - 1]
        coef1 = (self.times[ind_time] - time)/denom
        coef2 = (time - self.times[ind_time - 1])/denom
        new_field = self.fields[ind_time]*coef2 + \
                    self.fields[ind_time - 1]*coef1
        new_field.time = time
        # returning
        assert isinstance(new_field, self.fields[0].__class__)
        return new_field

    def get_fluctuant_fields(self, nmb_min_mean=1):
        """
        Calculate the fluctuant fields (fields minus mean field).

        Parameters
        ----------
        nmb_min_mean : number, optional
            Parameter for mean computation (see 'get_mean_field' doc).

        Returns
        -------
        fluct_fields : TemporalScalarFields or TemporalVectorFields object
            Contening fluctuant fields.
        """
        fluct_fields = self.__class__()
        mean_field = self.get_mean_field(nmb_min=nmb_min_mean)
        for i, field in enumerate(self.fields):
            fluct_fields.add_field(field - mean_field, time=self.times[i],
                                   unit_times=self.unit_times)
        return fluct_fields

    def get_spatial_spectrum(self, component, direction, intervx=None,
                             intervy=None, intervtime=None, welch_seglen=None,
                             scaling='base', fill='linear'):
        """
        Return a spatial spectrum.
        If more than one time are specified, spectrums are averaged.

        Parameters
        ----------
        component : string
            Should be an attribute name of the stored fields.
        direction : string
            Direction in which perform the spectrum ('x' or 'y').
        intervx and intervy : 2x1 arrays of number, optional
            To chose the zone where to calculate the spectrum.
            If not specified, the biggest possible interval is choosen.
        intervtime : 2x1 array, optional
            Interval of time on which averaged the spectrum.
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

        Notes
        -----
        If there is missing values on the field, 'fill' is used to linearly
        interpolate the missing values (can impact the spectrum).
        """
        # check parameters
        try:
            self[0].__getattribute__('{}_as_sf'.format(component))
        except AttributeError():
            raise ValueError()
        if not isinstance(direction, STRINGTYPES):
            raise TypeError()
        if direction not in ['x', 'y']:
            raise ValueError()
        if intervtime is None:
            intervtime = [self.times[0], self.times[-1]]
        if not isinstance(intervtime, ARRAYTYPES):
            raise TypeError()
        intervtime = np.array(intervtime)
        if not intervtime.shape == (2,):
            raise ValueError()
        if intervtime[0] < self.times[0]:
            intervtime[0] = self.times[0]
        if intervtime[-1] > self.times[-1]:
            intervtime[-1] = self.times[-1]
        if intervtime[0] >= intervtime[1]:
            raise ValueError()
        # loop on times
        spec = 0
        nmb = 0
        for i, time in enumerate(self.times):
            if time < intervtime[0] or time > intervtime[1]:
                continue
            comp = self[i].__getattribute__('{}_as_sf'.format(component))
            if spec == 0:
                spec = comp.get_spatial_spectrum(direction, intervx=intervx,
                                                 intervy=intervy,
                                                 welch_seglen=welch_seglen,
                                                 scaling=scaling, fill=fill)
            else:
                spec += comp.get_spatial_spectrum(direction, intervx=intervx,
                                                  intervy=intervy,
                                                  welch_seglen=welch_seglen,
                                                  scaling=scaling, fill=fill)
            nmb += 1
        # returning
        spec /= nmb
        return spec

    def get_time_profile(self, component, pt, wanted_times=None, ind=False):
        """
        Return a profile contening the time evolution of the given component.

        Parameters
        ----------
        component : string
            Should be an attribute name of the stored fields.
        pt : 2x1 array of numbers, or pts.Points object
            Wanted position for the time profile, in axis units.
        wanted_times : 2x1 array of numbers
            Time interval in which getting profile (default is all).
        ind : boolean, optional
            If 'True', values are undersood as indices.

        Returns
        -------
        profile : prof.Profile object

        """
        # check parameters coherence
        if not isinstance(component, STRINGTYPES):
            raise TypeError("'component' must be a string")
        if isinstance(pt, ARRAYTYPES):
            if ind:
                if pt[0] % 1 != 0 or pt[1] % 1 != 0:
                    raise ValueError()
                ind_x = int(pt[0])
                ind_y = int(pt[1])
            else:
                ind_x = self.get_indice_on_axe(1, pt[0], kind='nearest')
                ind_y = self.get_indice_on_axe(2, pt[1], kind='nearest')
            axe_x, axe_y = self.axe_x, self.axe_y
            if not (0 <= ind_x < len(axe_x) and 0 <= ind_y < len(axe_y)):
                raise ValueError("'x' ans 'y' values out of bounds")
            pt = np.array([[ind_x, ind_y]]*len(self.times), dtype=int)
            mask_times = np.zeros(len(self.times), dtype=bool)
        if isinstance(pt, pts.Points):
            mask_times = [time not in pt.v for time in self.times]
            mask_times = np.array(mask_times, dtype=bool)
            pt = [[self.get_indice_on_axe(1, pt.xy[i, 0], kind='nearest'),
                   self.get_indice_on_axe(2, pt.xy[i, 1], kind='nearest')]
                  for i in range(len(pt.xy))]
            pt = np.array(pt, dtype=int)
        if wanted_times is not None:
            if wanted_times[-1] <= wanted_times[0]:
                raise ValueError()
            mask_times = np.logical_or(self.times < wanted_times[0],
                                       mask_times)
            mask_times = np.logical_or(self.times > wanted_times[1],
                                       mask_times)
        # getting wanted time if necessary
        w_times_ind = np.arange(len(self.times))[~mask_times]
        # getting component values
        dim = len(w_times_ind)
        compo = np.empty(dim, dtype=float)
        masks = np.empty(dim, dtype=float)
        for i, time_ind in enumerate(w_times_ind):
            ind_x, ind_y = pt[i]
            compo[i] = self.fields[time_ind].__getattribute__(component)[ind_x,
                                                                         ind_y]
            masks[i] = self.fields[time_ind].mask[ind_x, ind_y]
        # gettign others datas
        time = self.times[w_times_ind]
        unit_time = self.unit_times
        unit_values = self.unit_values
        # getting position indices
        return prof.Profile(time, compo, masks, unit_x=unit_time,
                            unit_y=unit_values)

    def inject_time_profile(self, comp, pt, prof, ind=False):
        """
        Overwrite the value at the given points with data from the time
        profile.

        Parameters
        ----------
        comp : string
            Should be an attribute name of the stored fields.
        pt : 2x1 array of numbers, or pts.Points object
            Wanted position for the time profile, in axis units.
        prof : Profile object
            Profile used to overwrite data at point
        ind : boolean, optional
            If 'True', values are understood as indices.
        """
        # check
        if not isinstance(comp, STRINGTYPES):
            raise TypeError("'comp' must be a string")
        if isinstance(pt, ARRAYTYPES):
            if ind:
                if pt[0] % 1 != 0 or pt[1] % 1 != 0:
                    raise ValueError()
                ind_x = int(pt[0])
                ind_y = int(pt[1])
            else:
                ind_x = self.get_indice_on_axe(1, pt[0], kind='nearest')
                ind_y = self.get_indice_on_axe(2, pt[1], kind='nearest')
            axe_x, axe_y = self.axe_x, self.axe_y
            if not (0 <= ind_x < len(axe_x) and 0 <= ind_y < len(axe_y)):
                raise ValueError("'x' ans 'y' values out of bounds")
            pt = np.array([[ind_x, ind_y]]*len(self.times), dtype=int)
            mask_times = np.zeros(len(self.times), dtype=bool)
        if isinstance(pt, pts.Points):
            mask_times = [time not in pt.v for time in self.times]
            mask_times = np.array(mask_times, dtype=bool)
            pt = [[self.get_indice_on_axe(1, pt.xy[i, 0], kind='nearest'),
                   self.get_indice_on_axe(2, pt.xy[i, 1], kind='nearest')]
                  for i in range(len(pt.xy))]
            pt = np.array(pt, dtype=int)
        if len(prof) != len(self):
            raise ValueError()
        # do it
        for i in range(len(prof)):
            self.fields[i].__getattribute__(comp)[ind_x, ind_y] = prof.y[i]

    def get_temporal_spectrum(self, component, pt, ind=False,
                              wanted_times=None, welch_seglen=None,
                              scaling='base', fill='linear', mask_error=True,
                              detrend='constant'):
        """
        Return a Profile object, with the temporal spectrum of
        'component' on the point 'pt'.

        Parameters
        ----------
        component : string
            .
        pt : 2x1 array of numbers
            .
        ind : boolean
            If true, 'pt' is read as indices,
            else, 'pt' is read as coordinates.
        wanted_times : 2x1 array, optional
            Time interval in which compute spectrum (default is all).
        welch_seglen : integer, optional
            If specified, welch's method is used (dividing signal into
            overlapping segments, and averaging periodogram) with the given
            segments length (in number of points).
        scaling : string, optional
            If 'base' (default), result are in component unit.
            If 'spectrum', the power spectrum is returned (in unit^2).
            If 'density', the power spectral density is returned (in unit^2/Hz)
        fill : string or float
            Specifies the way to treat missing values.
            A value for value filling.
            A string ('linear', 'nearest', 'zero', 'slinear', 'quadratic,
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
        magn_prof : prof.Profile object
            Magnitude spectrum.
        """
        # checking parameters coherence
        if not isinstance(pt, ARRAYTYPES):
            raise TypeError("'pt' must be a 2x1 array")
        if ind:
            pt = np.array(pt, dtype=int)
        else:
            pt = np.array(pt, dtype=float)
        if not pt.shape == (2,):
            raise ValueError("'pt' must be a 2x1 array")
        if ind and (not isinstance(pt[0], INTEGERTYPES) or
                    not isinstance(pt[1], INTEGERTYPES)):
            raise TypeError("If 'ind' is True, 'pt' must be an array of two"
                            " integers")
        if not isinstance(ind, bool):
            raise TypeError("'ind' must be a boolean")
        x = pt[0]
        y = pt[1]
        # getting time profile
        time_prof = self.get_time_profile(component, pt=[x, y], ind=ind,
                                          wanted_times=wanted_times)
        magn_prof = time_prof.get_spectrum(welch_seglen=welch_seglen,
                                           scaling=scaling, fill=fill,
                                           mask_error=mask_error,
                                           detrend=detrend)
        return magn_prof

    def get_temporal_spectrum_over_area(self, component, intervx, intervy,
                                        ind=False, welch_seglen=None,
                                        scaling='base', fill='linear',
                                        detrend='constant'):
        """
        Return a prof.Profile object, contening a mean spectrum of the given
        component, on all the points included in the given intervals.

        Parameters
        ----------
        component : string
            Scalar component ('Vx', 'Vy', 'magnitude', ...).
        intervx, intervy : 2x1 arrays of numbers
            Defining the square on which averaging the spectrum.
            (in axes values)
        ind : boolean
            If true, 'pt' is read as indices,
            else, 'pt' is read as coordinates.
        welch_seglen : integer, optional
            If specified, welch's method is used (dividing signal into
            overlapping segments, and averaging periodogram) with the given
            segments length (in number of points).
        scaling : string, optional
            If 'base' (default), result are in component unit.
            If 'spectrum', the power spectrum is returned (in unit^2).
            If 'density', the power spectral density is returned (in unit^2/Hz)
        fill : string or float
            Specifies the way to treat missing values.
            A value for value filling.
            A string ('linear', 'nearest', 'zero', 'slinear', 'quadratic,
            'cubic' where 'slinear', 'quadratic' and 'cubic' refer to a spline
            interpolation of first, second or third order) for interpolation.
        detrend : string, optional
            Method used to detrend the profile. Can be 'none',
            'constant' (default) or 'linear'.

        Returns
        -------
        magn_prof : prof.Profile object
            Averaged magnitude spectrum.
        """
        # checking parameters coherence
        if not isinstance(component, STRINGTYPES):
            raise TypeError("'component' must be a string")
        if not isinstance(intervx, ARRAYTYPES):
            raise TypeError("'intervx' must be an array")
        if not isinstance(intervy, ARRAYTYPES):
            raise TypeError("'intervy' must be an array")
        if not isinstance(intervx[0], NUMBERTYPES):
            raise TypeError("'intervx' must be an array of numbers")
        if not isinstance(intervy[0], NUMBERTYPES):
            raise TypeError("'intervy' must be an array of numbers")
        axe_x, axe_y = self.axe_x, self.axe_y
        # checking interval values and getting bound indices
        if ind:
            if not isinstance(intervx[0], int)\
                    or not isinstance(intervx[1], int)\
                    or not isinstance(intervy[0], int)\
                    or not isinstance(intervy[1], int):
                raise TypeError("'intervx' and 'intervy' must be arrays of"
                                " integer if 'ind' is 'True'")
            if intervx[0] < 0 or intervy[0] < 0\
                    or intervx[-1] >= len(axe_x)\
                    or intervy[-1] >= len(axe_y):
                raise ValueError("intervals are out of bounds")
            ind_x_min = intervx[0]
            ind_x_max = intervx[1]
            ind_y_min = intervy[0]
            ind_y_max = intervy[1]
        else:
            axe_x_min = np.min(axe_x)
            axe_x_max = np.max(axe_x)
            axe_y_min = np.min(axe_y)
            axe_y_max = np.max(axe_y)
            if np.min(intervx) < axe_x_min\
                    or np.max(intervx) > axe_x_max\
                    or np.min(intervy) < axe_y_min\
                    or np.max(intervy) > axe_y_max:
                raise ValueError("intervals ({}) are out of bounds ({})"
                                 .format([intervx, intervy],
                                         [[axe_x_min, axe_x_max],
                                          [axe_y_min, axe_y_max]]))
            ind_x_min = self.get_indice_on_axe(1, intervx[0])[-1]
            ind_x_max = self.get_indice_on_axe(1, intervx[1])[0]
            ind_y_min = self.get_indice_on_axe(2, intervy[0])[-1]
            ind_y_max = self.get_indice_on_axe(2, intervy[1])[0]
        # Averaging ponctual spectrums
        magn = 0.
        nmb_fields = (ind_x_max - ind_x_min + 1)*(ind_y_max - ind_y_min + 1)
        real_nmb_fields = nmb_fields
        for i in np.arange(ind_x_min, ind_x_max + 1):
            for j in np.arange(ind_y_min, ind_y_max + 1):
                tmp_m = self.get_temporal_spectrum(component, [i, j], ind=True,
                                                   welch_seglen=welch_seglen,
                                                   scaling=scaling,
                                                   fill=fill, mask_error=True,
                                                   detrend=detrend)
                # check if the position is masked
                if tmp_m is None:
                    real_nmb_fields -= 1
                else:
                    magn = magn + tmp_m
        if real_nmb_fields == 0:
            raise Exception("I can't find a single non-masked time profile"
                            ", maybe you will want to try 'zero_fill' "
                            "option")
        magn = magn/real_nmb_fields
        return magn

    def get_spectrum_map(self, comp, welch_seglen=None, nmb_pic=1,
                         spec_smooth=None,
                         verbose=True):
        """
        Return the temporal spectrum map.

        Parameters
        ----------
        comp : string
            Component to get the spectrum from.
        welch_seglen : integer
            .
        nmb_pic : integer
            Number of succesive spectrum pic to detect
        spec_smooth : number
            .
        verbose : bool
            .

        Returns
        -------
        map_freq_sf :
            .
        map_freq_quality_sf :
            .
        """
        # check
        try:
            self.fields[0].__getattribute__(comp)
        except AttributeError():
            raise ValueError()
        # prepare
        map_freq = []
        map_freq_quality = []
        map_freq_mask = np.zeros(self.shape, dtype=bool)
        for i in range(nmb_pic):
            map_freq.append(np.zeros(self.shape, dtype=float))
            map_freq_quality.append(np.zeros(self.shape, dtype=float))
        if verbose:
            PG = ProgressCounter(init_mess="Begin spectrum map computation on {}"
                                 .format(comp),
                                 nmb_max=self.shape[0]*self.shape[1],
                                 name_things="points")
        # loop on field points
        for i, x in enumerate(self.axe_x):
            for j, y in enumerate(self.axe_y):
                if verbose:
                    PG.print_progress()
                # get local spectrum
                tmp_prof = self.get_time_profile(comp, [x, y])
                # check if should be masked
                if np.sum(tmp_prof.mask)/float(len(tmp_prof)) > .5:
                    map_freq_mask[i, j] = True
                    continue
                spec = tmp_prof.get_spectrum(welch_seglen=welch_seglen)
                # smooth if necessary
                if spec_smooth is not None:
                    spec.smooth(tos='gaussian', size=spec_smooth, inplace=True)
                # get maximale frequences
                for n in range(nmb_pic):
                    spec_max = spec.max
                    max_pos_ind = spec.get_value_position(spec_max,
                                                          ind=True)[0]
                    map_freq[n][i, j] = spec.x[max_pos_ind]
                    # get spectrum 'quality'
                    filt = np.logical_not(spec.mask)
                    spec_var = np.mean((spec.y[filt] -
                                       np.mean(spec.y[filt]))**2)**.5
                    map_freq_quality[n][i, j] = (spec_max - spec.mean)/spec_var
                    # remove this particular pic
                    spec.mask[max_pos_ind] = True
        # store results
        maps_freq = []
        maps_freq_quality = []
        for i in range(nmb_pic):
            map_freq_sf = sf.ScalarField()
            map_freq_sf.import_from_arrays(axe_x=self.axe_x, axe_y=self.axe_y,
                                           values=map_freq[i],
                                           mask=map_freq_mask,
                                           unit_x=self.unit_x,
                                           unit_y=self.unit_y,
                                           unit_values=spec.unit_y)
            map_freq_quality_sf = sf.ScalarField()
            map_freq_quality_sf.import_from_arrays(axe_x=self.axe_x,
                                                   axe_y=self.axe_y,
                                                   values=map_freq_quality[i],
                                                   mask=map_freq_mask,
                                                   unit_x=self.unit_x,
                                                   unit_y=self.unit_y,
                                                   unit_values='')
            maps_freq.append(map_freq_sf)
            maps_freq_quality.append(map_freq_quality_sf)
        # return
        if nmb_pic == 1:
            return maps_freq[0], maps_freq_quality[0]
        else:
            return maps_freq, maps_freq_quality

    def _get_comp_spectral_filtering(self, comp, fmin, fmax, order=2):
        """
        Perform a temporal spectral filtering on the wanted component

        Parameters
        ----------
        fmin, fmax : numbers
            Minimal and maximal frequencies
        order : integer, optional
            Butterworth filter order

        Returns
        -------
        filt_tf : TemporalFields
            Filtered temporal field
        """
        # check
        try:
            tf = self.__getattribute__("{}_as_sf".format(comp))
        except AttributeError():
            raise ValueError()
        # loop on space
        xys = [[x, y]
               for x in tf.axe_x
               for y in tf.axe_y]
        for x, y in xys:
            tmp_prof = tf.get_time_profile('values', [x, y])
            filt_tmp_prof = tmp_prof.spectral_filtering(fmin, fmax,
                                                        order=order)
            tf.inject_time_profile('values', [x, y], filt_tmp_prof)
        return tf

    def get_recurrence_map(self, norm=2, verbose=False, bandwidth=None,
                           normalized=False):
        """
        Return the recurrence map associated with the 2-norm.

        Returns
        -------
        rec_map : sf.ScalarField object
            .
        """
        length = len(self.fields)
        rec_map = np.zeros((length, length))
        if verbose:
            pc = ProgressCounter(init_mess="Begin recurence map computation",
                                 nmb_max=int(length**2/2. + length/2.),
                                 name_things='norms', perc_interv=1)
        if self.field_type == vf.VectorField:
            field_type = 0
            nmb_val = self.shape[0]*self.shape[1]*2
        elif self.field_type == sf.ScalarField:
            field_type = 1
            nmb_val = self.shape[0]*self.shape[1]
        else:
            raise Exception()
        inv_norm = 1./norm
        if norm % 2 == 0:
            need_abs = False
        else:
            need_abs = True
        # norm
        if normalized:
            if field_type == 0:
                magnitude = self.get_mean_field().magnitude
                normalizer = np.sum(magnitude**norm)**inv_norm/nmb_val
            else:
                magnitude = self.get_mean_field().values
                normalizer = np.sum(magnitude**norm)**inv_norm/nmb_val
        # double loop
        for i in range(length):
            for j in range(length):
                if i > j:
                    continue
                if i == j:
                    rec_map[i, j] = 0
                    continue
                if bandwidth is not None:
                    if np.abs(i - j) > bandwidth:
                        rec_map[i, j] = np.nan
                        rec_map[j, i] = np.nan
                        continue
                if verbose:
                    pc.print_progress()
                # get difference fields
                if field_type == 0:
                    valuesx = self.fields[i].comp_x - self.fields[j].comp_x
                    valuesy = self.fields[i].comp_y - self.fields[j].comp_y
                    values = np.concatenate((valuesx, valuesy))
                elif field_type == 1:
                    values = self.fields[i].values - self.fields[j].values
                # get norm
                if need_abs:
                    res = np.sum(np.abs(values)**norm)**inv_norm/nmb_val
                else:
                    res = np.sum(values**norm)**inv_norm/nmb_val
                rec_map[i, j] = res
                rec_map[j, i] = res
        # normalize
        if normalized:
            rec_map /= normalizer
        # return
        rec_map_sf = sf.ScalarField()
        rec_map_sf.import_from_arrays(axe_x=self.times, axe_y=self.times,
                                      values=rec_map,
                                      unit_x=self.unit_times,
                                      unit_y=self.unit_times,
                                      unit_values=self.unit_values)
        return rec_map_sf

    def extend(self, nmb_left=0, nmb_right=0, nmb_up=0, nmb_down=0,
               inplace=False):
        """
        Add columns or lines of masked values at the fields.

        Parameters
        ----------
        nmb_**** : integers
            Number of lines/columns to add in each direction.
        inplace : bool
            If 'False', return a new extended field, if 'True', modify the
            field inplace.
        Returns
        -------
        Extended_field : TemporalFields object, optional
            Extended field.
        """
        if inplace:
            tmp_tf = self
        else:
            tmp_tf = self.copy()
        # scale axis
        fld.Field.extend(tmp_tf, nmb_left=nmb_left, nmb_right=nmb_right,
                         nmb_up=nmb_up, nmb_down=nmb_down, inplace=True)
        # scale fields
        for i, _ in enumerate(tmp_tf.fields):
            tmp_tf.fields[i].extend(nmb_left=nmb_left, nmb_right=nmb_right,
                                    nmb_up=nmb_up, nmb_down=nmb_down,
                                    inplace=True)
        # return
        if not inplace:
            return tmp_tf

    def scale(self, scalex=None, scaley=None, scalev=None, scalet=None,
              inplace=False):
        """
        Scale the Fields.

        Parameters
        ----------
        scalex, scaley, scalev : numbers or Unum objects
            Scale for the axis and the values.
        inplace : boolean
            .
        """
        if inplace:
            tmp_f = self
        else:
            tmp_f = self.copy()
        # scale the field (automaticly scale the fields axis)
        # for compatibility purpose
        if not hasattr(self, "xy_scale"):
            self.xy_scale = make_unit('')
        for field in self.fields:
            if not hasattr(field, "xy_scale"):
                field.xy_scale = self.xy_scale
        fld.Field.scale(tmp_f, scalex=scalex, scaley=scaley,
                        inplace=True)
        # scale the values
        flds.Fields.scale(tmp_f, scalev=scalev, inplace=True)
        # scale the time
        if scalet is None:
            pass
        elif isinstance(scalet, NUMBERTYPES):
            tmp_f.times *= scalet
        elif isinstance(scalet, unum.Unum):
            new_unit = tmp_f.unit_times*scalet
            fact = new_unit.asNumber()
            new_unit /= fact
            tmp_f.unit_times = new_unit
            tmp_f.times *= fact
        else:
            raise TypeError()
        # returning
        if not inplace:
            return tmp_f

    def make_evenly_spaced(self, interp='linear', res=1):
        """
        Use interpolation to make the fields evenly spaced

        Parameters
        ----------
        interp : {‘linear’, ‘cubic’, ‘quintic’}, optional
            The kind of spline interpolation to use. Default is ‘linear’.
        res : number
            Resolution of the resulting field.
            A value of 1 meaning a spatial resolution equal to the smallest
            space along the two axis for the initial field.
        """
        # raise NotImplementedError('Not implemented yet')
        for field in self.fields:
            field.make_evenly_spaced(interp=interp, res=res)
        self._Field__axe_x = self.fields[0].axe_x
        self._Field__axe_y = self.fields[0].axe_y
        self.__mask = False

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
            for field in self.fields:
                field.change_unit(axe, new_unit)
        elif axe == 'time':
            old_unit = self.unit_times
            new_unit = old_unit.asUnit(new_unit)
            fact = new_unit.asNumber()
            self.times *= fact
            self.unit_times = new_unit/fact
        else:
            raise ValueError()
        if axe in ['x', 'y']:
            fld.Field.change_unit(self, axe, new_unit)

    def add_field(self, field, time=0., unit_times="", copy=True):
        """
        Add a field to the existing fields.

        Parameters
        ----------
        field : vf.VectorField or sf.ScalarField object
            The field to add.
        time : number
            time associated to the field.
        unit_time : Unum object
            time unit.
        """
        # TODO : pas de vérification de la cohérence des unités !
        # checking parameters
        if not isinstance(field, (vf.VectorField, sf.ScalarField)):
            raise TypeError()
        if self.field_type is None:
            self.field_type = field.__class__
        if not isinstance(field, self.field_type):
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
        if len(self.fields) == 0:
            self.axe_x = field.axe_x
            self.axe_y = field.axe_y
            self.unit_x = field.unit_x
            self.unit_y = field.unit_y
            self.unit_times = unit_times
            self.__times = np.asarray([time], dtype=float)
            self.field_type = field.__class__
        # if not
        else:
            # checking field type
            if not isinstance(field, self.field_type):
                raise TypeError()
            # checking axis
            axe_x, axe_y = self.axe_x, self.axe_y
            vaxe_x, vaxe_y = field.axe_x, field.axe_y
            if not np.all(axe_x == vaxe_x) and np.all(axe_y == vaxe_y):
                raise ValueError("Axes of the new field must be consistent "
                                 "with current axes")
            # storing time
            time = (time*self.unit_times/unit_times).asNumber()
            self.__times = np.append(self.__times, time)
        # use default constructor
        flds.Fields.add_field(self, field, copy=copy)
        # sorting the field with time
        self.__sort_field_by_time()

    def remove_fields(self, fieldnumbers):
        """
        Remove field(s) of the existing fields.

        Parameters
        ----------
        fieldnumber : integer or list of integers
            Velocity field(s) number(s) to remove.
        """
        if isinstance(fieldnumbers, INTEGERTYPES):
            fieldnumbers = [fieldnumbers]
        for nmb in fieldnumbers:
            self.__times = np.delete(self.times, nmb)
        flds.Fields.remove_field(self, fieldnumbers)

    def reduce_spatial_resolution(self, fact, inplace=False, verbose=False):
        """
        Reduce the spatial resolution of the fields by a factor 'fact'

        Parameters
        ----------
        fact : int
            Reducing factor.
        inplace : boolean, optional
            .
        """
        if inplace:
            tmpfs = self
        else:
            tmpfs = self.copy()
        # verbose
        if verbose:
            pg = ProgressCounter(f"Reducing resolution by {fact}",
                                 len(self), name_things='fields')
        #
        for f in tmpfs.fields:
            f.reduce_spatial_resolution(fact=fact, inplace=True)
            if verbose:
                pg.print_progress()
        #
        self.axe_x = self.fields[0].axe_x
        self.axe_y = self.fields[0].axe_y
        return tmpfs

    def reduce_temporal_resolution(self, nmb_in_interval, mean=True,
                                   inplace=False):
        """
        Return a TemporalVelocityFields, contening one field for each
        'nmb_in_interval' field in the initial TFVS.

        Parameters
        ----------
        nmb_in_interval : integer
            Length of the interval.
            (one field is kept for each 'nmb_in_interval fields)
        mean : boolean, optional
            If 'True', the resulting fields are average over the interval.
            Else, fields are taken directly.
        inplace : boolean, optional

        Returns
        -------
        TVFS : TemporalVelocityFields
        """
        # cehck parameters
        if not isinstance(nmb_in_interval, int):
            raise TypeError("'nmb_in_interval' must be an integer")
        if nmb_in_interval <= 0:
            raise TypeError("'nmb_in_interval' must be a positive integer")

        if nmb_in_interval == 1:
            return self.copy()
        if nmb_in_interval >= len(self):
            raise ValueError("'nmb_in_interval' is too big")
        if not isinstance(mean, bool):
            raise TypeError("'mean' must be a boolean")
        #
        tmp_TFS = self.__class__()
        i = 0
        times = self.times
        while True:
            tmp_f = self[i]
            time = times[i]
            if mean:
                for j in np.arange(i + 1, i + nmb_in_interval):
                    tmp_f += self[j]
                    time += times[j]
                tmp_f /= nmb_in_interval
                time /= nmb_in_interval
            tmp_TFS.add_field(tmp_f, time, self.unit_times)
            i += nmb_in_interval
            if i + nmb_in_interval >= len(self):
                break
        # returning
        if inplace:
            self.fields = tmp_TFS.fields
            self.times = tmp_TFS.times
        else:
            return tmp_TFS

    def augment_temporal_resolution(self, fact=2, inplace=False):
        """
        Augment the temporal resolution using temporal interpoalation.

        Parameters
        ----------
        fact : integer
            Temporal resolution ratio.
        inplace : bool
            .
        """
        # check
        assert type(fact) in [int], "TypeError"
        assert fact > 0, "ValueError"
        assert type(inplace) == bool, "TypeError"
        # fact = 1 (fool...)
        if fact == 1:
            if inplace:
                return None
            else:
                return self.copy()
        # get data
        if inplace:
            tf = self
        else:
            tf = self.copy()
        # get new times
        new_times = []
        for i in range(len(tf.times) - 1):
            tmp_times = np.linspace(tf.times[i], tf.times[i + 1],
                                    fact + 1)[0:-1]
            new_times.append(tmp_times)
        new_times = np.array(new_times).flatten()
        new_times = np.append(new_times, tf.times[-1])
        # loop on new times
        new_fields = []
        for time in new_times:
            new_fields.append(tf.get_interpolated_field(time))
        # store
        tf.fields = new_fields
        tf.times = new_times
        # returning
        if not inplace:
            return tf

    def crop_masked_border(self, hard=False, inplace=False):
        """
        Crop the masked border of the velocity fields in place.

        Parameters
        ----------
        hard : boolean, optional
            If 'True', partially masked border are croped as well.
        inplace : boolean, optional
            If 'True', crop the F in place,
            else, return a croped TF.
        """
        #get cumulated mask
        mask_cum = self.mask_cum
        # checking masked values presence
        if not np.any(mask_cum):
            return None
        # hard cropping
        if hard:
            if inplace:
                tmp_tf = self
            else:
                tmp_tf = self.copy()
            # remove trivial borders
            tmp_tf.crop_masked_border(hard=False, inplace=True)
            # until there is no more masked values
            while True:
                # getting mask
                masks = tmp_tf.mask
                mask = np.sum(masks, axis=0)
                mask = mask == len(tmp_tf.fields)
                # getting number of masked value on each border
                bd1 = np.sum(mask[0, :])
                bd2 = np.sum(mask[-1, :])
                bd3 = np.sum(mask[:, 0])
                bd4 = np.sum(mask[:, -1])
                # getting more masked border
                more_masked = np.argmax([bd1, bd2, bd3, bd4])
                # check remaining masked values
                if [bd1, bd2, bd3, bd4][more_masked] == 0:
                    break
                # deleting more masked border
                if more_masked == 0:
                    len_x = len(tmp_tf.axe_x)
                    tmp_tf.crop(intervx=[1, len_x], ind=True, inplace=True)
                elif more_masked == 1:
                    len_x = len(tmp_tf.axe_x)
                    tmp_tf.crop(intervx=[0, len_x - 2], ind=True,
                                inplace=True)
                elif more_masked == 2:
                    len_y = len(tmp_tf.axe_y)
                    tmp_tf.crop(intervy=[1, len_y], ind=True, inplace=True)
                elif more_masked == 3:
                    len_y = len(tmp_tf.axe_y)
                    tmp_tf.crop(intervy=[0, len_y - 2], ind=True, inplace=True)
            if not inplace:
                return tmp_tf
        # soft cropping
        else:
            # getting positions to remove
            # (column or line with only masked values)
            axe_y_m = ~np.all(mask_cum, axis=0)
            axe_x_m = ~np.all(mask_cum, axis=1)
            # skip if nothing to do
            if not np.any(axe_y_m) or not np.any(axe_x_m):
                return None
            # getting indices where we need to cut
            axe_x_min = np.where(axe_x_m)[0][0]
            axe_x_max = np.where(axe_x_m)[0][-1]
            axe_y_min = np.where(axe_y_m)[0][0]
            axe_y_max = np.where(axe_y_m)[0][-1]
            # crop
            if inplace:
                self.crop(intervx=[axe_x_min, axe_x_max],
                          intervy=[axe_y_min, axe_y_max], ind=True,
                          inplace=True)
            else:
                tmp_tf = self.copy()
                tmp_tf.crop(intervx=[axe_x_min, axe_x_max],
                            intervy=[axe_y_min, axe_y_max], ind=True,
                            inplace=True)
                return tmp_tf

    def mirroring(self, direction, position, inds_to_mirror='all', mir_coef=1.,
                  inplace=False, interp=None, value=[0, 0]):
        """
        Return the fields with additional mirrored values.

        Parameters
        ----------
        direction : integer
            Axe on which place the symetry plane (1 for x and 2 for y)
        position : number
            Position of the symetry plane along the given axe
        inds_to_mirror : integer
            Number of vector rows to symetrize (default is all)
        mir_coef : number or 2x1 array, optional
            Optional coefficient(s) applied only to the mirrored values.
            It can be an array first value is for 'comp_x' and second one to
            'comp_y' (for vector fields)
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
        if inplace:
            tmp_tf = self
        else:
            tmp_tf = self.copy()
        # mirror fields
        for i in range(len(self.fields)):
            tmp_tf.fields[i].mirroring(direction=direction, position=position,
                                       inds_to_mirror=inds_to_mirror,
                                       mir_coef=mir_coef, inplace=True,
                                       interp=interp, value=value)
        # update field
        tmp_tf.__axe_x = self.fields[0].axe_x
        tmp_tf.__axe_y = self.fields[0].axe_y
        # return
        if not inplace:
            return tmp_tf

    def remove_weird_fields(self, std_coef=3.29, treatment='interpolate',
                            inplace=False):
        """
        Look at the time evolution of spatial mean magnitude to identify and
        replace weird fields.

        Parameters
        ----------
        std_coef : number
            Fields associated with mean magnitude outside the interval
            [mean - std_coef*std, mean - std_coef*std] are treated as weird
            fields. Default value of '3.29' corespond for a 99.9% interval.
        treatment : string in ['remove', 'interpolate']
            Type of treatment for the weird fields
            (default is 'interpolate')
        inplace : bool
            .

        Returns
        -------
        tf : TemporalField
            treated temporal field
        """
        # get data
        if inplace:
            tmp_tf = self
        else:
            tmp_tf = self.copy()
        # get weird fields indices
        mean_magn = []
        for field in tmp_tf.fields:
            mean_magn.append(np.sum(field.magnitude[~field.mask]))
        mean = np.mean(mean_magn)
        mean_eps = np.std(mean_magn)*std_coef
        filt = np.logical_or(mean_magn < mean - mean_eps,
                             mean_magn > mean + mean_eps)
        weird_inds = np.arange(len(tmp_tf))[filt]
        # treat weird fields
        if treatment == 'interpolate':
            # replace weird fields with interpolations
            for weird_ind in weird_inds:
                eps = 1
                while True:
                    if (weird_ind + eps in weird_inds or
                            weird_ind - eps in weird_inds):
                        eps += 1
                    else:
                        break
                tmp_tf.fields[weird_ind] = (tmp_tf.fields[weird_ind + eps] +
                                            tmp_tf.fields[weird_ind - eps])/2
        elif treatment == 'remove':
            tmp_tf.remove_fields(weird_inds)
        else:
            raise ValueError()
        # return
        if not inplace:
            return tmp_tf

    def crop(self, intervx=None, intervy=None, intervt=None, full_output=False,
             ind=False, inplace=False):
        """
        Return a croped field in respect with given intervals.

        Parameters
        ----------
        intervx : array, optional
            interval wanted along x
        intervy : array, optional
            interval wanted along y
        intervt : array, optional
            interval wanted along time
        full_output : boolean, optional
            If 'True', cutting indices are alson returned
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
            else:
                intervt = [int(intervt[0]), int(intervt[1])]
        # crop
        if inplace:
            cropfield = self
        else:
            cropfield = self.copy()
        # temporal
        if intervt is not None:
            cropfield.fields = cropfield.fields[intervt[0]:intervt[1] + 1]
            cropfield.times = cropfield.times[intervt[0]:intervt[1] + 1]
        # spatial
        fld.Field.crop(cropfield, intervx=intervx, intervy=intervy, ind=ind,
                       inplace=True)
        for field in cropfield.fields:
            field.crop(intervx=intervx, intervy=intervy, ind=ind,
                       inplace=True)
        # returning
        if not inplace:
            return cropfield

    def set_origin(self, x=None, y=None):
        """
        Modify the axis in order to place the origin at the actual point (x, y)

        Parameters
        ----------
        x : number
        y : number
        """
        fld.Field.set_origin(self, x, y)

    def copy(self):
        """
        Return a copy of the velocityfields
        """
        return copy.deepcopy(self)

    def __sort_field_by_time(self):
        if len(self.fields) in [0, 1]:
            return None
        ind_sort = np.argsort(self.times)
        self.times = self.times[ind_sort]
        self.fields = self.fields[ind_sort]

    def display_multiple(self, component=None, kind=None, inds=None,
                         sharecb=False, sharex=False, sharey=False,
                         ncol=None, nrow=None, **plotargs):
        """
        Display a component of the velocity fields.

        Parameters
        ----------
        component : string, optional
            component to display
        kind : string, optional
            Kind of display wanted.
        fields_ind : array of indices
            Indices of fields to display.
        samecb : boolean, optional
            If 'True', the same color system is used for all the fields.
            You have to pass 'vmin' and 'vmax', to have correct results.
        ncol, nrow : int, optional
            Wanted number of columns and rows. If not specified, these values
            are computed so that ncol ~ nrow.
        plotargs : dict, optional
            Arguments passed to the function used to display the vector field.
        """
        nmb_fields = len(inds)
        # getting values
        if component is None or component == 'V':
            try:
                values = [[self.fields[ind].comp_x, self.fields[ind].comp_y]
                          for ind in inds]
            except AttributeError:
                values = [self.fields[ind].values for ind in inds]
        else:
            values = [self.fields[ind].__getattribute__(component)
                      for ind in inds]
        # display
        if sharex or sharey:
            plotargs['adjustable'] = 'datalim'
        db = pplt.Displayer(x=[self.axe_x]*nmb_fields,
                            y=[self.axe_y]*nmb_fields,
                            values=values, kind=kind, **plotargs)
        plot = db.draw_multiple(inds=list(range(len(inds))), sharecb=sharecb,
                                sharex=sharex,
                                sharey=sharey, ncol=ncol, nrow=nrow)
        return plot

    def display(self, compo=None, kind=None, sharecb=True, buffer_size=100,
                **plotargs):
        """
        Create a windows to display temporals field, controlled by buttons.

        Parameters
        ----------
        compo: string
            Component to plot.
        kind: string
            Kind of plot to use.
        sharecb: boolean
            Do all the vector field serie has to share the same colorbar or
            not.
        buffer_size: number
            Number of displays to keep in memory (faster, but use memory).
        **plotargs : dic
            Arguments passed to the plot command.

        Display control
        ---------------
        The display can be controlled useing the button, but also the keyboard:
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
        nmb_fields = len(self.fields)
        # getting values
        if compo is None or compo == 'V':
            try:
                values = np.asarray([[field.comp_x, field.comp_y]
                                     for field in self.fields])
            except AttributeError:
                values = np.asarray([field.values for field in self.fields])
        else:
            values = np.asarray([field.__getattribute__(compo)
                                 for field in self.fields])
        # check arguments for colorbar drawning
        if 'norm' in list(plotargs.keys()):
            sharecb = True
            normcb = plotargs['norm']
        else:
            normcb = None
        # display
        db = pplt.Displayer(x=[self.axe_x]*nmb_fields,
                            y=[self.axe_y]*nmb_fields,
                            values=values, kind=kind,
                            buffer_size=buffer_size,
                            **plotargs)
        win = pplt.ButtonManager(db,
                                 xlabel="X " + self.unit_x.strUnit(),
                                 ylabel="Y " + self.unit_y.strUnit(),
                                 sharecb=sharecb, normcb=normcb)
        pplt.DataCursorTextDisplayer(db)
        return db

    def display_animate(self, compo=None, interval=500, fields_inds=None,
                        repeat=True,
                        **plotargs):
        """
        Display fields animated in time.

        Parameters
        ----------
        compo : string
            Composante to display
        interval : number, optionnal
            interval between two frames in milliseconds.
        fields_ind : array of indices
            Indices of wanted fields. by default, all the fields are displayed
        repeat : boolean, optional
            if True, the animation is repeated infinitely.
        additional arguments can be passed (scale, vmin, vmax,...)
        """
        from matplotlib import animation
        if fields_inds is None:
            fields_inds = len(self.fields)
        # getting data
        if self.field_type == vf.VectorField:
            if compo == 'V' or compo is None:
                comp = self.fields
            elif compo == 'magnitude':
                comp = self.magnitude_as_sf
            elif compo == 'x':
                comp = self.Vx_as_sf
            elif compo == 'y':
                comp = self.Vy_as_sf
            elif compo == 'mask':
                comp = self.mask_as_sf
            else:
                raise ValueError()
        elif self.field_type == sf.ScalarField:
            if compo == 'values' or compo is None:
                comp = self.values_as_sf
            elif compo == 'mask':
                comp = self.mask_as_sf
            else:
                raise ValueError()
        else:
            raise TypeError()
        if 'kind' in list(plotargs.keys()):
            kind = plotargs['kind']
        else:
            kind = None
        # display a vector field (quiver)
        if isinstance(comp[0], vf.VectorField)\
                and (kind is None or kind == "quiver"):
            fig = plt.figure()
            ax = plt.gca()
            displ = comp[0].display(**plotargs)
            ttl = plt.title('')
            anim = animation.FuncAnimation(fig, self._update_vf,
                                           frames=fields_inds,
                                           interval=interval, blit=False,
                                           repeat=repeat,
                                           fargs=(fig, ax, displ, ttl, comp,
                                                  compo, plotargs))
            return anim,
        # display a scalar field (contour, contourf or imshow) or a streamplot
        elif isinstance(comp[0], sf.ScalarField)\
                or isinstance(comp[0], vf.VectorField):
            fig = plt.figure()
            ax = plt.gca()
            displ = comp[0].display(**plotargs)
            ttl = plt.suptitle('')
            anim = animation.FuncAnimation(fig, self._update_sf,
                                           frames=fields_inds,
                                           interval=interval, blit=False,
                                           repeat=repeat,
                                           fargs=(fig, ax, displ, ttl, comp,
                                                  compo, plotargs))
            return anim,
        else:
            raise ValueError("I don't know any '{}' composant".format(compo))

    def record_animation(self, anim, filepath, kind='gif', fps=30, dpi=100,
                         bitrate=50, imagemagick_path=None):
        """
        Record an animation in a gif file.
        You must create an animation (using 'display_animate' for example)
        before calling this method.
        You may have to specify the path to imagemagick in orfer to use it.


        Parameters
        ----------

        """
        import matplotlib
        if imagemagick_path is None:
            imagemagick_path = r"C:\Program Files\ImageMagick\convert.exe"
        matplotlib.rc('animation', convert_path=imagemagick_path)
        if kind == 'gif':
            writer = matplotlib.animation.ImageMagickWriter(fps=fps,
                                                            bitrate=bitrate)
            anim.save(filepath, writer=writer, fps=fps, dpi=dpi)
        elif kind == 'mp4':
            anim.save(filepath, writer='fmpeg', fps=fps, bitrate=bitrate)

    def _update_sf(self, num, fig, ax, displ, ttl, comp, compo, plotargs):
        plt.sca(ax)
        ax.cla()
        displ = comp[num]._display(**plotargs)
        title = "{}, at t={:.3} {}"\
            .format(compo, float(self.times[num]),
                    self.unit_times.strUnit())
        ttl.set_text(title)
        return displ,

    def _update_vf(self, num, fig, ax, displ, ttl, comp, compo, plotargs):
        plt.sca(ax)
        ax.cla()
        displ = comp[num]._display(**plotargs)
        title = "{}, at t={:.2f} {}"\
            .format(compo, float(self.times[num]),
                    self.unit_times.strUnit())
        ttl.set_text(title)
        return displ,
