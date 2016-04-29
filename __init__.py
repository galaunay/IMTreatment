#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
IMTreatment - A fields study package
====================================

Provides
  1. Class representing 2D fields of 1 component (ScalarField)
  2. Class representing 2D fields of 2 components (VectorField)
  3. Class representing a velocity field (VelocityField)
  4. Classes representing sets of velocityfields
     (Spatial- and TemporalVelocityFields)
  5. Class representing profiles (Profile)
  6. Class representing scatter points (Points)

How to use the documentation
----------------------------
Documentation is available in docstrings provided
with the code.

Code snippets are indicated by three greater-than signs::

  >>> x = 42
  >>> x = x + 1

Use the built-in ``help`` function to view a function's docstring::

  >>> help(np.sort)
  ... # doctest: +SKIP

Available subpackages
---------------------
boundary_layer
    Functions used in boundary layer computation.
vortex_detection
    Functions used in vortex tracking and detection.

Particular warnings
-------------------
It is strongly recommended not to use "import *" on this package.
"""
# from .core import make_unit,\
#     Profile, Points, OrientedPoints,\
#     ScalarField, VectorField, Fields, TemporalFields,\
#     TemporalScalarFields, TemporalVectorFields, SpatialVectorFields,\
#     SpatialScalarFields
from .utils import make_unit
from .core import Profile, Points, OrientedPoints, ScalarField, VectorField,\
    TemporalFields, TemporalScalarFields, TemporalVectorFields
# __all__ = ["ScalarField", "VectorField",
#            "Fields", "TemporalFields", "TemporalScalarFields",
#            "TemporalVectorFields", "Points", "OrientedPoints", "Profile",
#            "make_unit"]
