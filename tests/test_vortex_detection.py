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

import unum
from IMTreatment import VectorField, file_operation as imtio, make_unit, \
    TemporalVectorFields
import IMTreatment.vortex_detection as vod
import matplotlib.pyplot as plt


class VODTest(unittest.TestCase):

    def setUp(self):
        try:
            os.chdir(os.path.dirname(os.path.realpath(__file__)))
        except:
            pass
        unit_x = make_unit('m')
        unit_y = make_unit('m')
        unit_values = make_unit('m/s')
        x = np.genfromtxt('axe_x')
        y = np.genfromtxt('axe_y')
        vx = np.genfromtxt('vod_vx')
        vy = np.genfromtxt('vod_vy')
        self.VF1 = VectorField()
        self.VF1.import_from_arrays(axe_x=x, axe_y=y,
                                    comp_x=vx, comp_y=vy,
                                    mask=False,
                                    unit_x=unit_x,
                                    unit_y=unit_y,
                                    unit_values=unit_values)
        self.TVF1 = TemporalVectorFields()
        for i in range(10):
            self.TVF1.add_field(self.VF1*np.cos(i/10*np.pi), i/10, "s")

    def test_get_critical_points(self):
        res_a = vod.get_critical_points(self.VF1)
        # imtio.export_to_file(res_a, "VF1_get_critical_points_a.cimt")
        res_a2 = imtio.import_from_file("VF1_get_critical_points_a.cimt")
        self.assertEqual(res_a, res_a2)
        #
        res_b = vod.get_critical_points(self.TVF1)
        # imtio.export_to_file(res_b, "VF1_get_critical_points_b.cimt")
        res_b2 = imtio.import_from_file("VF1_get_critical_points_b.cimt")
        self.assertEqual(res_b, res_b2)
        #
        res_c = vod.get_critical_points(self.TVF1, kind='gam_vort')
        # imtio.export_to_file(res_c, "VF1_get_critical_points_c.cimt")
        res_c2 = imtio.import_from_file("VF1_get_critical_points_c.cimt")
        self.assertEqual(res_c, res_c2)
        #
        res_d = vod.get_critical_points(self.VF1, kind='gam_vort')
        # imtio.export_to_file(res_d, "VF1_get_dritical_points_d.cimt")
        res_d2 = imtio.import_from_file("VF1_get_dritical_points_d.cimt")
        self.assertEqual(res_d, res_d2)

# # TEMP
# unittest.main()
# # TEMP - End
