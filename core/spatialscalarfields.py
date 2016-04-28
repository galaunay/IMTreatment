#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
IMTreatment module

    Auteur : Gaby Launay
"""


from ..types import ARRAYTYPES, INTEGERTYPES, NUMBERTYPES, STRINGTYPES
import scalarfield as sf
import spatialfields as sfs
import fields as flds

class SpatialScalarFields(sfs.SpatialFields):

    def __init__(self):
        flds.Fields.__init__(self)
        self.fields_type = sf.ScalarField

    @property
    def values_as_sf(self):
        return self.fields
