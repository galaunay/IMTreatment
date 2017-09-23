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

import os
import unittest

import numpy as np
import pytest

import unum
from IMTreatment import Points, file_operation as imtio, make_unit
import matplotlib.pyplot as plt


class TestPoints(object):

    def setup(self):
        try:
            os.chdir(os.path.dirname(os.path.realpath(__file__)))
        except:
            pass
        # unit_x = make_unit('m')
        # unit_y = make_unit('dm')
        # unit_v = make_unit('m/s**2')
        # x = np.genfromtxt('points_x')
        # y = np.genfromtxt('points_y')
        # v = np.genfromtxt('points_v')
        # x2 = np.genfromtxt('points_x2')
        # y2 = np.genfromtxt('points_y2')
        # v2 = np.genfromtxt('points_v2')
        # self.P1 = Points(list(zip(x, y)), v, unit_x=unit_x,
        #                  unit_y=unit_y, unit_v=unit_v)
        # self.P2 = Points(list(zip(x2, y2)), v2, unit_x=unit_x,
        #                  unit_y=unit_y, unit_v=unit_v)
        # imtio.export_to_file(self.P1, "P1.cimt")
        # imtio.export_to_file(self.P2, "P2.cimt")
        self.P1 = imtio.import_from_file("P1.cimt")
        self.P2 = imtio.import_from_file("P2.cimt")

    def test_copy(self):
        res_a = self.P1.copy()
        imtio.export_to_file(res_a, "P1_copy_a.cimt")
        res_a2 = imtio.import_from_file("P1_copy_a.cimt")
        assert res_a == res_a2




# TEMP
pytest.main(['test_points.py'])
# TEMP - End
