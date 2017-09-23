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
from IMTreatment import VectorField, file_operation as imtio, make_unit, \
    TemporalVectorFields
import IMTreatment.vortex_detection as vod


class TestVortexDetection(object):

    def setup(self):
        try:
            os.chdir(os.path.dirname(os.path.realpath(__file__)))
        except:
            pass
        self.VF1_nomask = imtio.import_from_file('VF1_nomask.cimt')
        self.TVF1_nomask = imtio.import_from_file('TVF1_nomask.cimt')

    def test_get_critical_points(self):
        res_a = vod.get_critical_points(self.VF1_nomask,
                                        mirroring=[['x', 0.]])
        # imtio.export_to_file(res_a, "VF1_get_critical_points_a.cimt")
        res_a2 = imtio.import_from_file("VF1_get_critical_points_a.cimt")
        assert res_a == res_a2
        #
        res_b = vod.get_critical_points(self.TVF1_nomask)
        # imtio.export_to_file(res_b, "VF1_get_critical_points_b.cimt")
        res_b2 = imtio.import_from_file("VF1_get_critical_points_b.cimt")
        assert res_b == res_b2
        #
        res_c = vod.get_critical_points(self.TVF1_nomask, kind='gam_vort')
        # imtio.export_to_file(res_c, "VF1_get_critical_points_c.cimt")
        res_c2 = imtio.import_from_file("VF1_get_critical_points_c.cimt")
        assert res_c == res_c2
        #
        res_d = vod.get_critical_points(self.VF1_nomask, kind='gam_vort',
                                        mirroring=[['x', 0.]])
        # imtio.export_to_file(res_d, "VF1_get_dritical_points_d.cimt")
        res_d2 = imtio.import_from_file("VF1_get_dritical_points_d.cimt")
        assert res_d == res_d2
        #
        res_e = vod.get_critical_points(self.TVF1_nomask, kind='pbi_cell')
        # imtio.export_to_file(res_e, "VF1_get_eritical_points_e.cimt")
        res_e2 = imtio.import_from_file("VF1_get_eritical_points_e.cimt")
        assert res_e == res_e2
        #
        res_f = vod.get_critical_points(self.VF1_nomask, kind='pbi_cell',
                                        mirroring=[['x', 0.]])
        # imtio.export_to_file(res_f, "VF1_get_fritical_points_f.cimt")
        res_f2 = imtio.import_from_file("VF1_get_fritical_points_f.cimt")
        assert res_f == res_f2
        #
        res_g = vod.get_critical_points(self.TVF1_nomask, kind='pbi_crit')
        # imtio.export_to_file(res_g, "VF1_get_gritical_points_g.cimt")
        res_g2 = imtio.import_from_file("VF1_get_gritical_points_g.cimt")
        assert res_g == res_g2
        #
        res_h = vod.get_critical_points(self.VF1_nomask, kind='pbi_crit',
                                        smoothing_size=2)
        # imtio.export_to_file(res_h, "VF1_get_hritical_points_h.cimt")
        res_h2 = imtio.import_from_file("VF1_get_hritical_points_h.cimt")
        assert res_h == res_h2

# TEMP
pytest.main(['test_vortex_detection.py'])
# pytest.main(['--pdb', 'test_vortex_detection.py'])
# TEMP - End
