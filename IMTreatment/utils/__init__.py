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
Utlities used by IMTreatment modules
"""

from .units import make_unit
from .types import  ARRAYTYPES, INTEGERTYPES, STRINGTYPES, NUMBERTYPES
from .files import Files, remove_files_in_dirs
from .progresscounter import ProgressCounter
from .codeinteraction import RemoveFortranOutput
from .multithreading import MultiThreading
from .memoization import memoize_to_file
