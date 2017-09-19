# -*- coding: utf-8 -*-
#!/bin/env python3

# Copyright (C) 2003-2007 Gaby Launay

# Author: Gaby Launay  <gaby.launay@tutanota.com>
# URL: https://framagit.org/gabylaunay/IMTreatment
# Version: 1.0

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
