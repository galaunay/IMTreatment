#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
IMTreatment3 module

    Auteur : Gaby Launay
"""


import matplotlib.pyplot as plt
import Plotlib as pplt
import numpy as np
import pdb
import unum
import copy
from scipy import ndimage
from ..utils import make_unit
from ..utils.types import ARRAYTYPES, INTEGERTYPES, NUMBERTYPES, STRINGTYPES
from . import field as field
from . import scalarfield as sf


class VectorField(field.Field):
    """
    Class representing a vector field (2D field, with two components on each
    point).

    Principal methods
    -----------------
    "import_from_*" : allows to easily create or import vector fields.

    "export_to_*" : allows to export.

    "display" : display the vector field, with these unities.

    Examples
    --------
    >>> import IMTreatment3 as imt
    >>> VF = imt.VectorField()
    >>> unit_axe = make_unit('cm')
    >>> unit_K = make_unit('K')
    >>> comp_x = [[4, 8], [4, 8]]
    >>> comp_y = [[1, 2], [3, 4]]
    >>> VF.import_from_arrays([1,2], [1,2], comp_x, comp_y,  unit_axe,
    ...                       unit_axe, unit_K, unit_K)
    >>> VF.display()
    """

    ### Operators ###
    def __init__(self):
        super(VectorField, self).__init__()
        self.__comp_x = np.array([], dtype=float)
        self.__comp_y = np.array([], dtype=float)
        self.__mask = np.array([], dtype=bool)
        self.__unit_values = make_unit('')

    def __neg__(self):
        tmpvf = self.copy()
        tmpvf.comp_x = -tmpvf.comp_x
        tmpvf.comp_y = -tmpvf.comp_y
        return tmpvf

    def __add__(self, other):
        if isinstance(other, VectorField):
            # test unities system
            try:
                self.unit_values + other.unit_values
                self.unit_x + other.unit_x
                self.unit_y + other.unit_y
            except:
                raise ValueError("I think these units don't match, fox")
            # identical shape and axis
            if np.all(self.axe_x == other.axe_x) and \
                    np.all(self.axe_y == other.axe_y):
                tmpvf = self.copy()
                fact = (other.unit_values/self.unit_values).asNumber()
                tmpvf.comp_x = self.comp_x + other.comp_x*fact
                tmpvf.comp_y = self.comp_y + other.comp_y*fact
                tmpvf.mask = np.logical_or(self.mask, other.mask)
                return tmpvf
            # different shape, partially same axis
            else:
                # getting shared points
                new_ind_x = np.array([np.any(np.abs(val - other.axe_x)
                                      < np.abs(val)*1e-4)
                                      for val in self.axe_x])
                new_ind_y = np.array([np.any(np.abs(val - other.axe_y)
                                      < np.abs(val)*1e-4)
                                      for val in self.axe_y])
                new_ind_xo = np.array([np.any(np.abs(val - self.axe_x)
                                       < np.abs(val)*1e-4)
                                       for val in other.axe_x])
                new_ind_yo = np.array([np.any(np.abs(val - self.axe_y)
                                       < np.abs(val)*1e-4)
                                       for val in other.axe_y])
                if not np.any(new_ind_x) or not np.any(new_ind_y):
                    raise ValueError("Incompatible shapes")
                new_ind_Y, new_ind_X = np.meshgrid(new_ind_y, new_ind_x)
                new_ind_value = np.logical_and(new_ind_X, new_ind_Y)
                new_ind_Yo, new_ind_Xo = np.meshgrid(new_ind_yo, new_ind_xo)
                new_ind_valueo = np.logical_and(new_ind_Xo, new_ind_Yo)
                # getting new axis and values
                new_axe_x = self.axe_x[new_ind_x]
                new_axe_y = self.axe_y[new_ind_y]
                fact = other.unit_values/self.unit_values
                new_comp_x = (self.comp_x[new_ind_value]
                              + other.comp_x[new_ind_valueo]
                              * fact.asNumber())
                new_comp_y = (self.comp_y[new_ind_value]
                              + other.comp_y[new_ind_valueo]
                              * fact.asNumber())
                new_comp_x = new_comp_x.reshape((len(new_axe_x),
                                                 len(new_axe_y)))
                new_comp_y = new_comp_y.reshape((len(new_axe_x),
                                                 len(new_axe_y)))
                new_mask = np.logical_or(self.mask[new_ind_value],
                                         other.mask[new_ind_valueo])
                new_mask = new_mask.reshape((len(new_axe_x), len(new_axe_y)))
                # creating vf
                tmpvf = VectorField()
                tmpvf.import_from_arrays(new_axe_x, new_axe_y, new_comp_x,
                                         new_comp_y,
                                         mask=new_mask, unit_x=self.unit_x,
                                         unit_y=self.unit_y,
                                         unit_values=self.unit_values)
                return tmpvf
        elif isinstance(other, ARRAYTYPES):
            other = np.array(other, subok=True)
            if other.shape != self.shape:
                raise ValueError()
            tmpvf = self.copy()
            tmpvf.comp_x = self.comp_x + other
            tmpvf.comp_y = self.comp_y + other
            tmpvf.mask = self.mask
            return tmpvf
        elif isinstance(other, unum.Unum):
            tmpvf = self.copy()
            fact = (other / self.unit_values).asNumber()
            tmpvf.comp_x = self.comp_x + fact
            tmpvf.comp_y = self.comp_y + fact
            tmpvf.mask = self.mask
            return tmpvf
        elif isinstance(other, NUMBERTYPES):
            tmpvf = self.copy()
            tmpvf.comp_x = self.comp_x + other
            tmpvf.comp_y = self.comp_y + other
            tmpvf.mask = self.mask
            return tmpvf
        else:
            raise TypeError("You can only add a velocity field "
                            "with others velocity fields")

    __radd__ = __add__

    def __sub__(self, other):
        other_tmp = other.__neg__()
        tmpvf = self.__add__(other_tmp)
        return tmpvf

    def __truediv__(self, other):
        if isinstance(other, ARRAYTYPES):
            other = np.array(other, subok=True)
            if other.shape != self.shape:
                raise ValueError()
            tmpvf = self.copy()
            tmpvf.comp_x = self.comp_x / other
            tmpvf.comp_y = self.comp_y / other
            tmpvf.mask = np.logical_or(self.mask, other == 0)
            return tmpvf
        elif isinstance(other, unum.Unum):
            tmpvf = self.copy()
            new_unit = tmpvf.unit_values/other
            scale = new_unit.asNumber()
            new_unit /= scale
            tmpvf.unit_values = new_unit
            tmpvf.comp_x *= scale
            tmpvf.comp_y *= scale
            tmpvf.mask = self.mask
            return tmpvf
        elif isinstance(other, NUMBERTYPES):
            tmpvf = self.copy()
            tmpvf.comp_x /= other
            tmpvf.comp_y /= other
            tmpvf.mask = self.mask
            return tmpvf
        else:
            raise TypeError("You can only divide a vector field "
                            "by numbers")

    __div__ = __truediv__

    def __rtruediv__(self, other):
        return other * self**(-1)

    __rdiv__ = __rtruediv__

    def __mul__(self, other):
        if isinstance(other, ARRAYTYPES):
            other = np.array(other, subok=True)
            if other.shape != self.shape:
                raise ValueError()
            tmpvf = self.copy()
            tmpvf.comp_x = self.comp_x * other
            tmpvf.comp_y = self.comp_y * other
            tmpvf.mask = self.mask
            return tmpvf
        elif isinstance(other, unum.Unum):
            tmpvf = self.copy()
            new_unit = tmpvf.unit_values*other
            scale = new_unit.asNumber()
            new_unit /= scale
            tmpvf.unit_values = new_unit
            tmpvf.comp_x *= scale
            tmpvf.comp_y *= scale
            tmpvf.mask = self.mask
            return tmpvf
        elif isinstance(other, NUMBERTYPES):
            tmpvf = self.copy()
            tmpvf.comp_x *= other
            tmpvf.comp_y *= other
            tmpvf.mask = self.mask
            return tmpvf
        elif isinstance(other, sf.ScalarField):
            if other.shape != self.shape:
                raise ValueError()
            tmpvf = self.copy()
            tmpvf.comp_x *= other.values
            tmpvf.comp_y *= other.values
            tmpvf.mask = np.logical_or(other.mask, self.mask)
            return tmpvf
        else:
            raise TypeError("You can only multiply a vector field "
                            "by numbers")

    __rmul__ = __mul__

    def __sqrt__(self):
        tmpvf = self.copy()
        tmpvf.comp_x = np.sqrt(tmpvf.comp_x)
        tmpvf.comp_y = np.sqrt(tmpvf.comp_y)
        return tmpvf

    def __pow__(self, number):
        if not isinstance(number, NUMBERTYPES):
            raise TypeError("You only can use a number for the power "
                            "on a Vectorfield")
        tmpvf = self.copy()
        tmpvf.comp_x = np.power(tmpvf.comp_x, number)
        tmpvf.comp_y = np.power(tmpvf.comp_y, number)
        return tmpvf

    def __abs__(self):
        tmpvf = self.copy()
        tmpvf.comp_x = np.abs(tmpvf.comp_x)
        tmpvf.comp_y = np.abs(tmpvf.comp_y)
        return tmpvf

    def __iter__(self):
        mask = self.mask
        datax = self.comp_x
        datay = self.comp_y
        for ij, xy in field.Field.__iter__(self):
            i = ij[0]
            j = ij[1]
            if not mask[i, j]:
                yield ij, xy, [datax[i, j], datay[i, j]]

    ### Attributes ###
    @property
    def comp_x(self):
        return self.__comp_x

    @comp_x.setter
    def comp_x(self, new_comp_x):
        if not isinstance(new_comp_x, ARRAYTYPES):
            raise TypeError()
        new_comp_x = np.array(new_comp_x)
        if not new_comp_x.shape == self.shape:
            raise ValueError("'comp_x' must be coherent with axis system")
        # storing dat
        self.__comp_x = new_comp_x

    @comp_x.deleter
    def comp_x(self):
        raise Exception("Nope, can't do that")

    @property
    def comp_x_as_sf(self):
        tmp_sf = sf.ScalarField()
        tmp_sf.import_from_arrays(self.axe_x, self.axe_y, self.comp_x,
                                  mask=self.mask, unit_x=self.unit_x,
                                  unit_y=self.unit_y,
                                  unit_values=self.unit_values)
        return tmp_sf

    @property
    def comp_y(self):
        return self.__comp_y

    @comp_y.setter
    def comp_y(self, new_comp_y):
        if not isinstance(new_comp_y, ARRAYTYPES):
            raise TypeError()
        new_comp_y = np.array(new_comp_y)
        if not new_comp_y.shape == self.shape:
            raise ValueError()
        # storing data
        self.__comp_y = new_comp_y

    @comp_y.deleter
    def comp_y(self):
        raise Exception("Nope, can't do that")

    @property
    def comp_y_as_sf(self):
        tmp_sf = sf.ScalarField()
        tmp_sf.import_from_arrays(self.axe_x, self.axe_y, self.comp_y,
                                  mask=self.mask, unit_x=self.unit_x,
                                  unit_y=self.unit_y,
                                  unit_values=self.unit_values)
        return tmp_sf

    @property
    def mask(self):
        return self.__mask

    @mask.setter
    def mask(self, new_mask):
        # check 'new_mask' coherence
        if isinstance(new_mask, bool):
            fill_value = new_mask
            new_mask = np.empty(self.shape, dtype=bool)
            new_mask.fill(fill_value)
        elif isinstance(new_mask, ARRAYTYPES):
            if not isinstance(new_mask.flat[0], np.bool_):
                raise TypeError()
            new_mask = np.array(new_mask, dtype=bool)
        else:
            raise TypeError("'mask' should be an array or a boolean,"
                            " not a {}".format(type(new_mask)))
        if self.shape != new_mask.shape:
            raise ValueError()
        # check if the new mask don'r reveal masked values
        if np.any(np.logical_not(new_mask[self.mask])):
            raise Warning("This mask reveal masked values, maybe you should"
                          "use the 'fill' function instead")
        # nanify masked values
        self.comp_x[new_mask] = np.NaN
        self.comp_y[new_mask] = np.NaN
        # store mask
        self.__mask = new_mask

    @mask.deleter
    def mask(self):
        raise Exception("Nope, can't do that")

    @property
    def mask_as_sf(self):
        tmp_sf = sf.ScalarField()
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
            if new_unit_values.asNumber() == 1:
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

    ### Properties ###

    @property
    def min(self):
        return np.min(self.magnitude)

    @property
    def max(self):
        return np.max(self.magnitude[np.logical_not(self.mask)])

    @property
    def magnitude(self):
        """
        Return a scalar field with the velocity field magnitude.
        """
        comp_x, comp_y = self.comp_x, self.comp_y
        mask = self.mask
        values = (comp_x**2 + comp_y**2)**(.5)
        values[mask] = np.NaN
        return values

    @property
    def magnitude_as_sf(self):
        """
        Return a scalarfield with the velocity field magnitude.
        """
        tmp_sf = sf.ScalarField()
        tmp_sf.import_from_arrays(self.axe_x, self.axe_y, self.magnitude,
                                  mask=self.mask, unit_x=self.unit_x,
                                  unit_y=self.unit_y,
                                  unit_values=self.unit_values)
        return tmp_sf

    @property
    def theta(self):
        """
        Return a scalar field with the vector angle (in reference of the unit_y
        vector [1, 0]).

        Parameters:
        -----------
        low_velocity_filter : number
            If not zero, points where V < Vmax*low_velocity_filter are masked.

        Returns:
        --------
        theta_sf : sf.ScalarField object
            Contening theta field.
        """
        # get data
        comp_x, comp_y = self.comp_x, self.comp_y
        not_mask = np.logical_not(self.mask)
        theta = np.zeros(self.shape)
        # getting angle
        norm = self.magnitude
        not_mask = np.logical_and(not_mask, norm != 0)
        theta[not_mask] = comp_x[not_mask]/norm[not_mask]
        theta[not_mask] = np.arccos(theta[not_mask])
        tmp_comp_y = comp_y.copy()
        tmp_comp_y[~not_mask] = 0
        sup_not_mask = tmp_comp_y < 0
        theta[sup_not_mask] = 2*np.pi - theta[sup_not_mask]
        return theta

    @property
    def theta_as_sf(self):
        """
        Return a scalarfield with the velocity field angles.
        """
        tmp_sf = sf.ScalarField()
        tmp_sf.import_from_arrays(self.axe_x, self.axe_y, self.theta,
                                  mask=False, unit_x=self.unit_x,
                                  unit_y=self.unit_y,
                                  unit_values=self.unit_values)
        return tmp_sf

    ### Field Maker ###
    def import_from_arrays(self, axe_x, axe_y, comp_x, comp_y, mask=False,
                           unit_x="", unit_y="", unit_values=""):
        """
        Set the vector field from a set of arrays.

        Parameters
        ----------
        axe_x : array
            Discretized axis value along x
        axe_y : array
            Discretized axis value along y
        comp_x : array or masked array
            Values of the x component at the discritized points
        comp_y : array or masked array
            Values of the y component at the discritized points
        mask : array of boolean, optional
            Mask on comp_x and comp_y
        unit_x : string, optionnal
            Unit for the values of axe_x
        unit_y : string, optionnal
            Unit for the values of axe_y
        unit_values : string, optionnal
            Unit for the field components.
        """
        self.__clean()
        self.axe_x = axe_x
        self.axe_y = axe_y
        self.comp_x = comp_x
        self.comp_y = comp_y
        self.mask = mask
        self.unit_x = unit_x
        self.unit_y = unit_y
        self.unit_values = unit_values

    ### Watchers ###
    def get_value(self, x, y, ind=False, unit=False):
        """
        Return the vectir field compoenents on the point (x, y).
        If ind is true, x and y are indices,
        else, x and y are value on axes (interpolated if necessary).
        """
        return np.array([self.comp_x_as_sf.get_value(x, y, ind=ind, unit=unit),
                         self.comp_y_as_sf.get_value(x, y, ind=ind, unit=unit)])

    def get_profile(self, component, direction, position, ind=False,
                    interp='linear'):
        """
        Return a profile of the vector field component, at the given position
        (or at least at the nearest possible position).
        If position is an interval, the fonction return an average profile
        in this interval.

        Function
        --------
        profile, cutposition = get_profile(component, direction, position, ind)

        Parameters
        ----------
        component : integer
            component to treat.
        direction : integer
            Direction along which we choose a position (1 for x and 2 for y).
        position : float or interval of float
            Position or interval in which we want a profile.
        ind : boolean, optional
            If 'True', position is taken as an indice
            Else (default), position is in the field units.
        interp : string in ['nearest', 'linear']
            if 'nearest', get the profile at the nearest position on the grid,
            if 'linear', use linear interpolation to get the profile at the
            exact position

        Returns
        -------
        profile : Profile object
            Asked profile.
        cutposition : array or number
            Final position or interval in which the profile has been taken.
        """
        if not isinstance(component, int):
            raise TypeError("'component' must be an integer")
        if component == 1:
            return self.comp_x_as_sf.get_profile(direction, position, ind,
                                                 interp=interp)
        elif component == 2:
            return self.comp_y_as_sf.get_profile(direction, position, ind,
                                                 interp=interp)
        else:
            raise ValueError("'component' must have the value of 1 or 2")

    def copy(self):
        """
        Return a copy of the vectorfield.
        """
        return copy.deepcopy(self)

    def get_norm(self, norm=2, normalized='perpoint'):
        """
        Return the field norm
        """
        values = np.concatenate((self.comp_x[~self.mask],
                                 self.comp_y[~self.mask]))
        res = (np.sum(np.abs(values)**norm))**(1./norm)
        if normalized == "perpoint":
            res /= np.sum(~self.mask)
        return res

    ### Modifiers ###
    def scale(self, scalex=None, scaley=None, scalev=None, inplace=False):
        """
        Scale the VectorField.

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
        # xy
        revx, revy = field.Field.scale(tmp_f, scalex=scalex, scaley=scaley, inplace=True,
                                 output_reverse=True)
        # v
        if scalev is None:
            pass
        elif isinstance(scalev, NUMBERTYPES):
            tmp_f.comp_x *= scalev
            tmp_f.comp_y *= scalev
        elif isinstance(scalev, unum.Unum):
            new_unit = tmp_f.unit_values*scalev
            fact = new_unit.asNumber()
            new_unit /= fact
            tmp_f.unit_values = new_unit
            tmp_f.comp_x *= fact
            tmp_f.comp_y *= fact
        else:
            raise TypeError()
        if revx and revy:
            tmp_f.comp_x = -tmp_f.comp_x[::-1, ::-1]
            tmp_f.comp_y = -tmp_f.comp_y[::-1, ::-1]
        elif revx:
            tmp_f.comp_x = -tmp_f.comp_x[::-1, :]
            tmp_f.comp_y = tmp_f.comp_y[::-1, :]
        elif revy:
            tmp_f.comp_x = tmp_f.comp_x[:, ::-1]
            tmp_f.comp_y = -tmp_f.comp_y[:, ::-1]
        # returning
        if not inplace:
            return tmp_f

    def rotate(self, angle, inplace=False):
        """
        Rotate the vector field.

        Parameters
        ----------
        angle : integer
            Angle in degrees (positive for trigonometric direction).
            In order to preserve the orthogonal grid, only multiples of
            90° are accepted (can be negative multiples).
        inplace : boolean, optional
            If 'True', vector field is rotated in place, else, the function
            return a rotated field.

        Returns
        -------
        rotated_field : VectorField object, optional
            Rotated vector field.
        """
        # check params
        if not isinstance(angle, NUMBERTYPES):
            raise TypeError()
        if angle%90 != 0:
            raise ValueError()
        if not isinstance(inplace, bool):
            raise TypeError()
        # get data
        if inplace:
            tmp_field = self
        else:
            tmp_field = self.copy()
        # normalize angle
        angle = angle%360
        # rotate the parent
        field.Field.rotate(tmp_field, angle, inplace=True)
        # rotate
        nmb_rot90 = int(angle/90)
        comp_x = np.rot90(tmp_field.comp_x, nmb_rot90)
        comp_y = np.rot90(tmp_field.comp_y, nmb_rot90)
        mask = np.rot90(tmp_field.mask, nmb_rot90)
        comp_x2 = np.cos(angle/180.*np.pi)*comp_x - np.sin(angle/180.*np.pi)*comp_y
        comp_y2 = np.cos(angle/180.*np.pi)*comp_y + np.sin(angle/180.*np.pi)*comp_x
        tmp_field.__comp_x, tmp_field.__comp_y = comp_x, comp_y
        tmp_field.__comp_x = comp_x2
        tmp_field.__comp_y = comp_y2
        tmp_field.__mask = mask
        # returning
        if not inplace:
            return tmp_field

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
            field.Field.change_unit(self, axe, new_unit)
        elif axe == 'y':
            field.Field.change_unit(self, axe, new_unit)
        elif axe =='values':
            old_unit = self.unit_values
            new_unit = old_unit.asUnit(new_unit)
            fact = new_unit.asNumber()
            self.comp_x *= fact
            self.comp_y *= fact
            self.unit_values = new_unit/fact
        else:
            raise ValueError()

    def smooth(self, tos='uniform', size=None, inplace=False, **kw):
        """
        Smooth the vectorfield in place.
        Warning : fill up the field (should be used carefully with masked field
        borders)

        Parameters :
        ------------
        tos : string, optional
            Type of smoothing, can be 'uniform' (default) or 'gaussian'
            (See ndimage module documentation for more details)
        size : number, optional
            Size of the smoothing (is radius for 'uniform' and
            sigma for 'gaussian').
            Default is 3 for 'uniform' and 1 for 'gaussian'.
        inplace : boolean, optional
            .
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
        # getting field
        if inplace:
            tmp_vf = self
        else:
            tmp_vf = self.copy()
        # filling up the field before smoothing
        tmp_vf.fill(inplace=True)
        # smoothing
        if tos == "uniform":
            tmp_vf.comp_x = ndimage.uniform_filter(tmp_vf.comp_x, size, **kw)
            tmp_vf.comp_y = ndimage.uniform_filter(tmp_vf.comp_y, size, **kw)
        elif tos == "gaussian":
            tmp_vf.comp_x = ndimage.gaussian_filter(tmp_vf.comp_x, size, **kw)
            tmp_vf.comp_y = ndimage.gaussian_filter(tmp_vf.comp_y, size, **kw)
        else:
            raise ValueError("'tos' must be 'uniform' or 'gaussian'")
        # storing
        if not inplace:
            return tmp_vf

    def make_evenly_spaced(self, interp="linear"):
        """
        Use interpolation to make the field evenly spaced

        Parameters
        ----------
        kind : {‘linear’, ‘cubic’, ‘quintic’}, optional
            The kind of spline interpolation to use. Default is ‘linear’.
        """
        # get data
        vx = self.comp_x_as_sf
        vy = self.comp_y_as_sf
        #
        vx.make_evenly_spaced(interp=interp)
        vy.make_evenly_spaced(interp=interp)
        # store
        self.import_from_arrays(vx.axe_x, vx.axe_y, vx.values,
                                vy.values, mask=False,
                                unit_x=self.unit_x,
                                unit_y=self.unit_y,
                                unit_values=self.unit_values)

    def fill(self, kind='linear', value=[0., 0.], inplace=False,
             reduce_tri=True, crop=False):
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
        value : 2x1 array of numbers
            Values used to fill (for kind='value').
        inplace : boolean, optional
            If 'True', fill the sf.ScalarField in place.
            If 'False' (default), return a filled version of the field.
        reduce_tri : boolean, optional
            If 'True', treatment is used to reduce the triangulation effort
            (faster when a lot of masked values)
            If 'False', no treatment (faster when few masked values)
        crop : boolean, optional
            If 'True', TVF borders are croped before filling.
        """
        # check parameters coherence
        if isinstance(value, NUMBERTYPES):
            value = [value, value]
        if not isinstance(value, ARRAYTYPES):
            raise TypeError()
        value = np.array(value)
        if not value.shape == (2,):
            raise ValueError()
        if crop:
            self.crop_masked_border(hard=False, inplace=True)
        # filling components
        comp_x = self.comp_x_as_sf
        comp_y = self.comp_y_as_sf
        new_comp_x = comp_x.fill(kind=kind, value=value[0], inplace=False,
                                 reduce_tri=reduce_tri)
        new_comp_y = comp_y.fill(kind=kind, value=value[1], inplace=False,
                                 reduce_tri=reduce_tri)
        # returning
        if inplace:
            self.comp_x = new_comp_x.values
            self.comp_y = new_comp_y.values
            mask = np.empty(self.shape, dtype=bool)
            mask.fill(False)
            self.__mask = mask
        else:
            vf = VectorField()
            vf.import_from_arrays(self.axe_x, self.axe_y, new_comp_x.values,
                                  new_comp_y.values, mask=False,
                                  unit_x=self.unit_x, unit_y=self.unit_y,
                                  unit_values=self.unit_values)
            return vf

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
            indmin_x, indmax_x, indmin_y, indmax_y = \
                field.Field.crop(self, intervx, intervy, full_output=True,
                           ind=ind, inplace=True)
            self.__comp_x = self.comp_x[indmin_x:indmax_x + 1,
                                        indmin_y:indmax_y + 1]
            self.__comp_y = self.comp_y[indmin_x:indmax_x + 1,
                                        indmin_y:indmax_y + 1]
            self.__mask = self.mask[indmin_x:indmax_x + 1,
                                    indmin_y:indmax_y + 1]
        else:
            indmin_x, indmax_x, indmin_y, indmax_y, cropfield = \
                field.Field.crop(self, intervx=intervx, intervy=intervy,
                           full_output=True, ind=ind)
            cropfield.__comp_x = self.comp_x[indmin_x:indmax_x + 1,
                                             indmin_y:indmax_y + 1]
            cropfield.__comp_y = self.comp_y[indmin_x:indmax_x + 1,
                                             indmin_y:indmax_y + 1]
            cropfield.__mask = self.mask[indmin_x:indmax_x + 1,
                                         indmin_y:indmax_y + 1]
            return cropfield

    def crop_masked_border(self, hard=False, inplace=False):
        """
        Crop the masked border of the field in place.

        Parameters
        ----------
        hard : boolean, optional
            If 'True', partially masked border are croped as well.
        """
        #
        if inplace:
            tmp_sf = self
        else:
            tmp_sf = self.copy()
        # checking masked values presence
        mask = tmp_sf.mask
        if not np.any(mask):
            return None
        # hard cropping
        if hard:
            # remove trivial borders
            tmp_sf.crop_masked_border(hard=False, inplace=True)
            # until there is no more masked values
            while np.any(tmp_sf.mask):
                # getting number of masked value on each border
                bd1 = np.sum(tmp_sf.mask[0, :])
                bd2 = np.sum(tmp_sf.mask[-1, :])
                bd3 = np.sum(tmp_sf.mask[:, 0])
                bd4 = np.sum(tmp_sf.mask[:, -1])
                # getting more masked border
                more_masked = np.argmax([bd1, bd2, bd3, bd4])
                # deleting more masked border
                if more_masked == 0:
                    len_x = len(tmp_sf.axe_x)
                    tmp_sf.crop(intervx=[1, len_x], ind=True, inplace=True)
                elif more_masked == 1:
                    len_x = len(tmp_sf.axe_x)
                    tmp_sf.crop(intervx=[0, len_x - 2], ind=True, inplace=True)
                elif more_masked == 2:
                    len_y = len(self.axe_y)
                    tmp_sf.crop(intervy=[1, len_y], ind=True,
                                inplace=True)
                elif more_masked == 3:
                    len_y = len(tmp_sf.axe_y)
                    tmp_sf.crop(intervy=[0, len_y - 2], ind=True, inplace=True)
        # soft cropping
        else:
            axe_x_m = np.logical_not(np.all(mask, axis=1))
            axe_y_m = np.logical_not(np.all(mask, axis=0))
            axe_x_min = np.where(axe_x_m)[0][0]
            axe_x_max = np.where(axe_x_m)[0][-1]
            axe_y_min = np.where(axe_y_m)[0][0]
            axe_y_max = np.where(axe_y_m)[0][-1]
            tmp_sf.crop([axe_x_min, axe_x_max], [axe_y_min, axe_y_max],
                        ind=True, inplace=True)
        # returning
        if not inplace:
            return tmp_sf

    def extend(self, nmb_left=0, nmb_right=0, nmb_up=0, nmb_down=0,
               inplace=False):
        """
        Add columns or lines of masked values at the vectorfield.

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
        if inplace:
            field.Field.extend(self, nmb_left=nmb_left, nmb_right=nmb_right,
                         nmb_up=nmb_up, nmb_down=nmb_down, inplace=True)
            new_shape = self.shape
        else:
            new_field = field.Field.extend(self, nmb_left=nmb_left,
                                     nmb_right=nmb_right, nmb_up=nmb_up,
                                     nmb_down=nmb_down, inplace=False)
            new_shape = new_field.shape
        new_Vx = np.zeros(new_shape, dtype=float)
        new_Vy = np.zeros(new_shape, dtype=float)
        if nmb_right == 0:
            slice_x = slice(nmb_left, new_Vx.shape[0] + 2)
        else:
            slice_x = slice(nmb_left, -nmb_right)
        if nmb_up == 0:
            slice_y = slice(nmb_down, new_Vx.shape[1] + 2)
        else:
            slice_y = slice(nmb_down, -nmb_up)
        new_Vx[slice_x, slice_y] = self.comp_x
        new_Vy[slice_x, slice_y] = self.comp_y
        new_mask = np.ones(new_shape, dtype=bool)
        new_mask[slice_x, slice_y] = self.mask
        if inplace:
            self.comp_x = new_Vx
            self.comp_y = new_Vy
            self.__mask = new_mask
        else:
            new_field.comp_x = new_Vx
            new_field.comp_y = new_Vy
            new_field.__mask = new_mask
            return new_field

    def mirroring(self, direction, position, inds_to_mirror='all', mir_coef=1.,
                  inplace=False, interp=None, value=[0, 0]):
        """
        Return a field with additional mirrored values.

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
            If ana array first value is for 'comp_x' and second one to 'comp_y'
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
        # getting components
        vx = self.comp_x_as_sf
        vy = self.comp_y_as_sf
        xy_scale = self.xy_scale
        # treating sign changments
        if isinstance(mir_coef, NUMBERTYPES):
            if direction == 1:
                coefx = -1
                coefy = 1
            else:
                coefx = 1
                coefy = -1
            coefx *= mir_coef
            coefy *= mir_coef
        elif isinstance(mir_coef, ARRAYTYPES):
            coefx = mir_coef[0]
            coefy = mir_coef[1]
        else:
            raise ValueError()
        # mirroring on components
        vx.mirroring(direction, position, inds_to_mirror=inds_to_mirror,
                     interp=interp, value=value[0], inplace=True,
                     mir_coef=coefx)
        vy.mirroring(direction, position, inds_to_mirror=inds_to_mirror,
                     interp=interp, value=value[1], inplace=True,
                     mir_coef=coefy)
        # storing
        if inplace:
            tmp_vf = self
        else:
            tmp_vf = VectorField()
        mask = np.logical_or(vx.mask, vy.mask)
        tmp_vf.import_from_arrays(vx.axe_x, vx.axe_y, vx.values, vy.values,
                                  mask=mask, unit_x=vx.unit_x,
                                  unit_y=vy.unit_y,
                                  unit_values=vx.unit_values)
        tmp_vf.xy_scale = xy_scale
        # returning
        if not inplace:
            return tmp_vf

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
        # reducing
        Vx = self.comp_x_as_sf
        Vy = self.comp_y_as_sf
        Vx.reduce_spatial_resolution(fact, inplace=True)
        Vy.reduce_spatial_resolution(fact, inplace=True)
        # returning
        if inplace:
            self.__init__()
            self.import_from_arrays(Vx.axe_x, Vx.axe_y, Vx.values,
                                    Vy.values,
                                    mask=Vx.mask, unit_x=self.unit_x,
                                    unit_y=self.unit_y,
                                    unit_values=self.unit_values)
        else:
            vf = VectorField()
            vf.import_from_arrays(Vx.axe_x, Vx.axe_y, Vx.values, Vy.values,
                                  mask=Vx.mask, unit_x=self.unit_x,
                                  unit_y=self.unit_y,
                                  unit_values=self.unit_values)
            return vf

    def __clean(self):
        self.__init__()

    ### Displayers ###
    def _display(self, component=None, kind=None, axis='image', **plotargs):
        # TODO : compatibility purpose
        if not hasattr(self, "xy_scale"):
            self.xy_scale = make_unit("")
        # adapt xy scale if streamplot is needed
        if kind == "stream" and self.xy_scale.asNumber() != 1.:
            magn = self.magnitude
            new_comp_x = self.comp_x*self.xy_scale.asNumber()
            new_magn = (new_comp_x**2 + self.comp_y**2)**.5
            values = [new_comp_x/new_magn*magn, self.comp_y/new_magn*magn]
        elif component in [None, 'V']:
            values = [self.comp_x, self.comp_y]
        elif component in ['comp_x', 'x']:
            values = self.comp_x
        elif component in ['comp_y', 'y']:
            values = self.comp_y
        else:
            values = self.__getattribute__(component)
        dp = pplt.Displayer(x=self.axe_x, y=self.axe_y, values=values,
                            kind=kind, **plotargs)
        plot = dp.draw(cb=False)
        pplt.DataCursorTextDisplayer(dp)
        unit_x, unit_y = self.unit_x, self.unit_y
        plt.xlabel("X " + unit_x.strUnit())
        plt.ylabel("Y " + unit_y.strUnit())
        return plot

    def display(self, component=None, kind=None, **plotargs):
        """
        Display something from the vector field.
        If component is not given, a quiver is displayed.
        If component is an integer, the coresponding component of the field is
        displayed.

        Parameters
        ----------
        component : string, optional
            Component to display, can be 'V', 'x', 'y', 'mask'
        kind : string, optinnal
            Scalar plots :
            if 'None': each datas are plotted (imshow),
            if 'contour': contours are ploted  (contour),
            if 'contourf': filled contours are ploted (contourf).
            Vector plots :
            if 'quiver': quiver plot,
            if 'stream': streamlines,
            if '3D': tri-dimensionnal plot.

        plotargs : dict
            Arguments passed to the function used to display the vector field.

        Returns
        -------
        fig : figure reference
            Reference to the displayed figure
        """
        displ = self._display(component, kind, **plotargs)
        unit_values = self.unit_values
        Vx, Vy = self.comp_x, self.comp_y
        if component is None or component == 'V':
            if kind == 'quiver' or kind is None:
                if 'C' not in list(plotargs.keys()):
                    cb = plt.colorbar(displ)
                    cb.set_label("Magnitude " + unit_values.strUnit())
                legendarrow = round(np.max([Vx.max(), Vy.max()]))
                plt.quiverkey(displ, 1.075, 1.075, legendarrow,
                              "$" + str(legendarrow)
                              + unit_values.strUnit() + "$",
                              labelpos='W', fontproperties={'weight': 'bold'})
            elif kind in ['stream', 'track']:
                if not 'color' in list(plotargs.keys()):
                    cb = plt.colorbar(displ.lines)
                    cb.set_label("Magnitude " + unit_values.strUnit())
            plt.title("Values " + unit_values.strUnit())
        elif component in ['x', 'comp_x']:
            cb = plt.colorbar(displ)
            cb.set_label("Vx " + unit_values.strUnit())
            plt.title("Vx " + unit_values.strUnit())
        elif component in ['y', 'comp_y']:
            cb = plt.colorbar(displ)
            cb.set_label("Vy " + unit_values.strUnit())
            plt.title("Vy " + unit_values.strUnit())
        elif component == 'mask':
            cb = plt.colorbar(displ)
            cb.set_label("Mask ")
            plt.title("Mask")
        elif component == 'magnitude':
            cb = plt.colorbar(displ)
            cb.set_label("Magnitude")
            plt.title("Magnitude")
        else:
            raise ValueError("Unknown 'component' value")
        return displ
