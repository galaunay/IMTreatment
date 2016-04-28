#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
IMTreatment module

    Auteur : Gaby Launay
"""


import pdb
from ..types import ARRAYTYPES, INTEGERTYPES, NUMBERTYPES, STRINGTYPES
from ..core import SpatialFields, VectorField, Fields


class SpatialVectorFields(SpatialFields):
    """
    Class representing a set of spatial-evolving velocity fields.
    """

    def __init__(self):
        Fields.__init__(self)
        self.fields_type = VectorField

    @property
    def Vx_as_sf(self):
        return [field.comp_x_as_sf for field in self.fields]

    @property
    def Vy_as_sf(self):
        return [field.comp_y_as_sf for field in self.fields]

    @property
    def magnitude_as_sf(self):
        return [field.magnitude_as_sf for field in self.fields]

    @property
    def theta_as_sf(self):
        return [field.theta_as_sf for field in self.fields]
