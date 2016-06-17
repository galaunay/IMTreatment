#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
IMTreatment3 module

    Auteur : Gaby Launay
"""


from ..utils.types import ARRAYTYPES, INTEGERTYPES, NUMBERTYPES, STRINGTYPES
from . import scalarfield as sf
from . import spatialfields as sfs
from . import fields as flds


class SpatialScalarFields(sfs.SpatialFields):

    def __init__(self):
        flds.Fields.__init__(self)
        self.fields_type = sf.ScalarField

    @property
    def values_as_sf(self):
        return self.fields
