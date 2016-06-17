#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
IMTreatment3 module

    Auteur : Gaby Launay
"""


import numpy as np
import pdb
import copy
from ..utils.types import ARRAYTYPES, INTEGERTYPES, NUMBERTYPES, STRINGTYPES
from . import field as field
from . import scalarfield as sf
from . import vectorfield as vf


class Fields(object):
    """
    Class representing a set of fields. These fields can have
    differente positions along axes, or be successive view of the same area.
    It's recommended to use TemporalVelocityFields or SpatialVelocityFields
    instead of this one.
    """

    ### Operators ###
    def __init__(self):
        self.fields = np.array([], dtype=object)

    def __len__(self):
        return len(self.fields)

    def __iter__(self):
        return self.fields.__iter__()

    def __getitem__(self, fieldnumber):
        return self.fields[fieldnumber]

    ### Watchers ###
    def copy(self):
        """
        Return a copy of the velocityfields
        """
        return copy.deepcopy(self)

    ### Modifiers ###
    def scale(self, scalex=None, scaley=None, scalev=None, inplace=False):
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
        # scale the fields
        for i, _ in enumerate(tmp_f.fields):
            tmp_f.fields[i].scale(scalex=scalex, scaley=scaley, scalev=scalev,
                                  inplace=True)
        # returning
        if not inplace:
            return tmp_f

    def rotate(self, angle, inplace=False):
        """
        Rotate the fields.

        Parameters
        ----------
        angle : integer
            Angle in degrees (positive for trigonometric direction).
            In order to preserve the orthogonal grid, only multiples of
            90° are accepted (can be negative multiples).
        inplace : boolean, optional
            If 'True', fields is rotated in place, else, the function
            return rotated fields.

        Returns
        -------
        rotated_field : TemporalFields or child object, optional
            Rotated fields.
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
        # rotate fields
        for i in np.arange(len(tmp_field.fields)):
            tmp_field.fields[i].rotate(angle=angle, inplace=True)
        # returning
        if not inplace:
            return tmp_field

    def add_field(self, field):
        """
        Add a field to the existing fields.

        Parameters
        ----------
        field : sf.VectorField or sf.ScalarField object
            The field to add.
        """
        if not isinstance(field, (vf.VectorField, sf.ScalarField)):
            raise TypeError("'vectorfield' must be a VelocityField object")
        self.fields = np.append(self.fields, field.copy())

    def remove_field(self, fieldnumbers):
        """
        Remove a field of the existing fields.

        Parameters
        ----------
        fieldnumber : integer or list of integers
            Velocity field(s) number(s) to remove.
        """
        if isinstance(fieldnumbers, INTEGERTYPES):
            fieldnumbers = [fieldnumbers]
        for nmb in fieldnumbers:
            self.fields = np.delete(self.fields, nmb)

    def set_origin(self, x=None, y=None):
        """
        Modify the axis in order to place the origin at the actual point (x, y)

        Parameters
        ----------
        x : number
        y : number
        """
        if x is not None:
            if not isinstance(x, NUMBERTYPES):
                raise TypeError("'x' must be a number")
            for field in self.fields:
                field.set_origin(x, None)
        if y is not None:
            if not isinstance(y, NUMBERTYPES):
                raise TypeError("'y' must be a number")
            for field in self.fields:
                field.set_origin(None, y)

    def smooth(self, tos='uniform', size=None, inplace=False, **kw):
        """
        Smooth the fields in place.
        Warning : fill up the field (should be used carefully with masked field
        borders)

        Parameters :
        ------------
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
        if inplace:
            tmp_f = self
        else:
            tmp_f = self.copy()
        # loop on fields
        for i, _ in enumerate(tmp_f.fields):
            tmp_f.fields[i].smooth(tos=tos, size=size, inplace=True, **kw)
        # returning
        if not inplace:
            return tmp_f
