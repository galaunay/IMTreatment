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

from ..utils.types import ARRAYTYPES, INTEGERTYPES, NUMBERTYPES, STRINGTYPES
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
