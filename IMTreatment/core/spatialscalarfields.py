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
