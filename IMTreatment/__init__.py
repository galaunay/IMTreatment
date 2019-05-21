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

"""
IMTreatment - A fields study package
====================================

This module has been written to carry out analysis and more
specifically structure detection on PIV velocity fields. It is now
more general and can handle different kind of data (point cloud,
scalar and vector field, ...) and perform classical and more advanced
analysis on them (spectra, pod, post-processing, visualization, ...).


General data analysis
---------------------
1. Class representing 2D fields of 1 component (``ScalarField``)
2. Class representing 2D fields of 2 components (``VectorField``)
3. Classes representing sets of scalar fields vector fields
   (``SpatialScalarFields``, ``TemporalScalarField``,
   ``SpatialVectorField`` and ``TemporalVectorFields``)
4. Class representing profiles (``Profile``)
5. Class representing scatter points (``Points``)
6. Module for modal decomposition (POD, DMD) and reconstruction (``pod``)
7. Module to import/export data from/to Davis, matlab, ascii, pivmat and
   images files (``file_operation``)
8. Functionalities to visualize those data (``plotlib``)


Flow analysis
-------------
1. Module to create artificial vortices: Burger, Jill, Rankine, ...
   and to simulate their motion in potential flows (``vortex_creation``)
2. Module providing several vortex criterions computation
   (``vortex_criterions``)
3. Module to automatically detect and track critical points
   (``vortex_detection``)
4. Module to compute the evolution of some vortex properties
   (``vortex_properties``)
5. Module to generate potential flows with arbitrary geometries
   (``potential_flow``)


Dependencies
------------
Mandatory::
    numpy
    matplotlib
    scipy
    unum
    modred

Optional::
    sklearn
    networkx
    colorama
    h5py

Particular warnings
-------------------
It is strongly recommended not to use "import *" on this package.
"""

name = "IMTreatment"

from .utils import make_unit
from .core import Profile, Points, OrientedPoints, ScalarField, VectorField,\
    TemporalFields, TemporalScalarFields, TemporalVectorFields,\
    TemporalPoints
