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

from .vortex_detection import \
    get_critical_points, get_separation_position,\
    get_critical_line, CritPoints,\
    get_vortex_radius, get_vortex_radius_time_evolution,\
    get_vortex_circulation,\
    get_vortex_property_time_evolution, \
    get_vortex_property, _get_vortex_position_on_VF,\
    get_vortex_position
