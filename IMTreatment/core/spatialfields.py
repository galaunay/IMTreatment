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
from ..utils.types import ARRAYTYPES, INTEGERTYPES, NUMBERTYPES, STRINGTYPES
from ..utils import make_unit
from . import fields as flds
from . import scalarfield as sf
from . import vectorfield as vf
from . import profile as prof


class SpatialFields(flds.Fields):
    """
    """

    def __init__(self):
        self.unit_x = make_unit('')
        self.unit_y = make_unit('')
        self.unit_values = make_unit('')
        self.fields_type = None

    def __neg__(self):
        tmp_tf = self.copy()
        for i in np.arange(len(self.fields)):
            tmp_tf.fields[i] = -tmp_tf.fields[i]
        return tmp_tf

    def __mul__(self, other):
        if isinstance(other, (NUMBERTYPES, unum.Unum)):
            final_vfs = self.__class__()
            for field in self.fields:
                final_vfs.add_field(field*other)
            return final_vfs
        else:
            raise TypeError("You can only multiply a temporal velocity field "
                            "by numbers")

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, (NUMBERTYPES, unum.Unum)):
            final_vfs = self.__class__.__init__()
            for field in self.fields:
                final_vfs.add_field(field/other)
            return final_vfs
        else:
            raise TypeError("You can only divide a temporal velocity field "
                            "by numbers")

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
    def unit_x(self):
        return self.__unit_x

    @unit_x.setter
    def unit_x(self, unit):
        self.__unit_x = unit
        for field in self.fields:
            field.unit_x = unit

    @property
    def unit_y(self):
        return self.__unit_y

    @unit_y.setter
    def unit_y(self, unit):
        self.__unit_y = unit
        for field in self.fields:
            field.unit_y = unit

    @property
    def unit_values(self):
        return self._SpatialFields__unit_values

    @unit_values.setter
    def unit_values(self, unit):
        self.__unit_values = unit
        for field in self.fields:
            field.unit_values = unit

    @property
    def x_min(self):
        return np.min([field.axe_x[0] for field in self.fields])

    @property
    def x_max(self):
        return np.max([field.axe_x[-1] for field in self.fields])

    @property
    def y_min(self):
        return np.min([field.axe_y[0] for field in self.fields])

    @property
    def y_max(self):
        return np.max([field.axe_y[-1] for field in self.fields])

    def add_field(self, field, copy=True):
        """
        """
        # check
        if not isinstance(field, self.fields_type):
            raise TypeError()
        # first field
        if len(self.fields) == 0:
            self.unit_x = field.unit_x
            self.unit_y = field.unit_y
            self.unit_values = field.unit_values
        # other ones
        else:
            try:
                field.change_unit('x', self.unit_x)
                field.change_unit('y', self.unit_y)
                field.change_unit('values', self.unit_values)
            except unum.IncompatibleUnitsError:
                raise ValueError("Inconsistent unit system")
        # crop fields
        field.crop_masked_border(hard=False, inplace=True)
        # add field
        flds.Fields.add_field(self, field, copy=copy)

    def get_value(self, x, y, unit=False, error=True):
        """
        Return the field component(s) on the point (x, y).

        Parameters
        ----------
        x, y : number
            Point coordinates
        unit : boolean, optional
            If 'True', component(s) is(are) returned with its unit.
        error : boolean, optional
            If 'True', raise an error if the asked point is outside the fields.
            If 'False', return 'None'
        """
        # get interesting fields
        inter_ind = []
        for i, field in enumerate(self.fields):
            if (x > field.axe_x[0] and x < field.axe_x[-1] and
                    y > field.axe_y[0] and y < field.axe_y[-1]):
                inter_ind.append(i)
        # get values (mean over fields if necessary)
        if len(inter_ind) == 0:
            if error:
                raise ValueError("coordinates outside the fields")
            else:
                return None
        elif len(inter_ind) == 1:
            values = self.fields[inter_ind[0]].get_value(x, y,
                                                         ind=False, unit=False)
        else:
            values = self.fields[inter_ind[0]].get_value(x, y,
                                                         ind=False, unit=False)
            for field in self.fields[inter_ind][1::]:
                values += field.get_value(x, y, ind=False, unit=False)
            values /= len(inter_ind)
        return values

    def get_values_on_grid(self, axe_x, axe_y):
        """
        Return a all the fields in a single evenly-spaced grid.
        (Use interpolation to get the data on the grid points)

        Parameters
        ----------
        axe_x, axe_y : arrays of ndim 1
            Representing the grid axis.
        """
        # check
        if not isinstance(axe_x, ARRAYTYPES):
            raise TypeError()
        if not isinstance(axe_y, ARRAYTYPES):
            raise TypeError()
        axe_x = np.array(axe_x)
        axe_y = np.array(axe_y)
        if isinstance(self.fields[0], sf.ScalarField):
            values = np.zeros(shape=(len(axe_x), len(axe_y)), dtype=float)
            mask = np.zeros(shape=(len(axe_x), len(axe_y)), dtype=bool)
            for i, x in enumerate(axe_x):
                for j, y in enumerate(axe_y):
                    val = self.get_value(x, y, unit=False, error=False)
                    if val is None:
                        mask[i, j] = True
                    else:
                        values[i, j] = val
            tmp_f = sf.ScalarField
            tmp_f.import_from_arrays(axe_x, axe_y, values, mask=mask,
                                     unit_x=self.unit_x, unit_y=self.unit_y,
                                     unit_values=self.unit_values)
        elif isinstance(self.fields[0], vf.VectorField):
            Vx = np.zeros(shape=(len(axe_x), len(axe_y)), dtype=float)
            Vy = np.zeros(shape=(len(axe_x), len(axe_y)), dtype=float)
            mask = np.zeros(shape=(len(axe_x), len(axe_y)), dtype=bool)
            for i, x in enumerate(axe_x):
                for j, y in enumerate(axe_y):
                    val = self.get_value(x, y, unit=False, error=False)
                    if val is None:
                        mask[i, j] = True
                    else:
                        Vx[i, j] = val[0]
                        Vy[i, j] = val[1]

            tmp_f = vf.VectorField()
            tmp_f.import_from_arrays(axe_x, axe_y, Vx, Vy,
                                     mask=mask,
                                     unit_x=self.unit_x, unit_y=self.unit_y,
                                     unit_values=self.unit_values)
        else:
            raise Exception()
        return tmp_f

    def get_profile(self, direction, position, component=None):
        """
        Return a profile of the current fields.

        Parameters
        ----------
        direction : integer
            Direction along which we choose a position (1 for x and 2 for y)
        position : float, interval of float
            Position, interval in which we want a profile

        component : string
            Component wanted for the profile.

        Returns
        -------
        profile : Profile object
            Wanted profile
        """
        # getting data
        if self.fields_type == vf.VectorField:
            if component is None:
                component = 'magnitude'
            else:
                try:
                    comp = self.__getattribute__("{}_as_sf".format(component))
                except AttributeError:
                    raise ValueError()
        elif self.fields_type == sf.ScalarField:
            if component is None:
                component = "values"
            try:
                comp = self.__getattribute__("{}_as_sf".format(component))
            except AttributeError:
                raise ValueError()
        # get fields of interest
        inter_ind = []
        if direction == 1:
            for i, field in enumerate(self.fields):
                if np.any(position < field.axe_x[-1]) and \
                   np.any(position > field.axe_x[0]):
                    inter_ind.append(i)
        elif direction == 2:
            for i, field in enumerate(self.fields):
                if np.any(position < field.axe_y[-1]) and \
                   np.any(position > field.axe_y[0]):
                    inter_ind.append(i)
        # get profiles
        if len(inter_ind) == 0:
            return None
        elif len(inter_ind) == 1:
            return comp[inter_ind[0]].get_profile(direction=direction,
                                                  position=position,
                                                  ind=False)[0]
        else:
            # get profiles
            x = np.array([])
            y = np.array([])
            for ind in inter_ind:
                tmp_comp = comp[ind]
                tmp_prof = tmp_comp.get_profile(direction=direction,
                                                position=position,
                                                ind=False)[0]
                x = np.concatenate((x, tmp_prof.x))
                y = np.concatenate((y, tmp_prof.y))
            # recreate profile object
            ind_sort = np.argsort(x)
            x = x[ind_sort]
            y = y[ind_sort]
            fin_prof = prof.Profile(x, y, mask=False, unit_x=tmp_prof.unit_x,
                                    unit_y=tmp_prof.unit_y)

            return fin_prof

    def _display(self, compo=None, **plotargs):
        """
        """
        # check params
        if len(self.fields) == 0:
            raise Exception()
        # getting data
        if self.fields_type == vf.VectorField:
            if compo == 'V' or compo is None:
                comp = self.fields
            else:
                try:
                    comp = self.__getattribute__("{}_as_sf".format(compo))
                except AttributeError:
                    raise ValueError()
        elif self.fields_type == sf.ScalarField:
            if compo is None:
                compo = "values"
            try:
                comp = self.__getattribute__("{}_as_sf".format(compo))
            except AttributeError:
                raise ValueError()
        else:
            raise TypeError()
        # getting max and min
        v_min = np.min([field.min for field in comp])
        v_max = np.max([field.max for field in comp])
        if 'vmin' in list(plotargs.keys()):
            v_min = plotargs.pop('vmin')
        if 'vmax' in list(plotargs.keys()):
            v_max = plotargs.pop('vmax')
        norm = plt.Normalize(v_min, v_max)
        if 'norm' not in list(plotargs.keys()):
            plotargs['norm'] = norm
        # setting default kind of display
        if 'kind' not in list(plotargs.keys()):
            plotargs['kind'] = None
        if plotargs['kind'] == 'stream':
            if 'density' not in list(plotargs.keys()):
                plotargs['density'] = 1.
            plotargs['density'] = plotargs['density']/(len(self.fields))**.5
        # display
        for field in comp:
            field._display(**plotargs)
        plt.xlim(self.x_min, self.x_max)
        plt.ylim(self.y_min, self.y_max)

    def display(self, compo=None, **plotargs):
        self._display(compo=compo, **plotargs)
        plt.axis('image')
        cb = plt.colorbar()
        cb.set_label(self.unit_values.strUnit())
        plt.tight_layout()

    def copy(self):
        return copy.deepcopy(self)
