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

import os
import sys
try:
    dirname = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(dirname)
    os.chdir(dirname)
except:
    pass
import matplotlib as mpl
mpl.use('Agg')

import pytest
from helper import sane_parameters, parametric_test

from IMTreatment import field_treatment as imtft, file_operation as imtio, \
    make_unit
import matplotlib.pyplot as plt


class TestFieldTreatment(object):
    """ Done """

    def setup(self):
        sane_parameters()
        self.VF1 = imtio.import_from_file("VF1.cimt")
        self.VF1_nomask = imtio.import_from_file("VF1_nomask.cimt")
        self.SF1 = imtio.import_from_file("SF1.cimt")
        self.SF2 = imtio.import_from_file("SF2.cimt")
        self.TVF1 = imtio.import_from_file("TVF1.cimt")
        self.Prof1 = imtio.import_from_file("Prof1.cimt")

    def test_get_gradients(self):
        res_a = imtft.get_gradients(self.VF1)
        # imtio.export_to_file(res_a, "VF1_get_gradients_a.cimt")
        res_a2 = imtio.import_from_file("VF1_get_gradients_a.cimt")
        assert res_a == res_a2
        #
        res_b = imtft.get_gradients(self.SF1)
        # imtio.export_to_file(res_b, "SF1_get_gradients_b.cimt")
        res_b2 = imtio.import_from_file("SF1_get_gradients_b.cimt")
        assert res_b == res_b2
        #
        res_c = imtft.get_gradients(self.Prof1)
        # imtio.export_to_file(res_c, "Prof1_get_gradients_c.cimt")
        res_c2 = imtio.import_from_file("Prof1_get_gradients_c.cimt")
        assert res_c == res_c2

    def test_get_jacobian_eigenproperties(self):
        res_a = imtft.get_jacobian_eigenproperties(self.VF1)
        # imtio.export_to_file(res_a, "VF1_get_jacobian_eigenproperties_a.cimt")
        res_a2 = imtio.import_from_file("VF1_get_jacobian_eigenproperties_a.cimt")
        assert res_a == res_a2

    def test_get_Kenwright_field(self):
        res_a = imtft.get_Kenwright_field(self.VF1)
        # imtio.export_to_file(res_a, "VF1_get_Kenwright_field_a.cimt")
        res_a2 = imtio.import_from_file("VF1_get_Kenwright_field_a.cimt")
        assert res_a[0] == res_a2[0]
        assert res_a[1] == res_a2[1]

    def test_get_grad_field(self):
        res_a = imtft.get_grad_field(self.VF1)
        # imtio.export_to_file(res_a, "VF1_get_grad_field_a.cimt")
        res_a2 = imtio.import_from_file("VF1_get_grad_field_a.cimt")
        assert res_a == res_a2

    def test_get_track_field(self):
        res_a = imtft.get_track_field(self.VF1)
        # imtio.export_to_file(res_a, "VF1_get_track_field_a.cimt")
        res_a2 = imtio.import_from_file("VF1_get_track_field_a.cimt")
        assert res_a == res_a2

    def test_get_divergence(self):
        res_a = imtft.get_divergence(self.VF1)
        # imtio.export_to_file(res_a, "VF1_get_divergence_a.cimt")
        res_a2 = imtio.import_from_file("VF1_get_divergence_a.cimt")
        assert res_a == res_a2
        #
        res_b = imtft.get_divergence(self.TVF1)
        # imtio.export_to_file(res_b, "VF1_get_divergence_b.cimt")
        res_b2 = imtio.import_from_file("VF1_get_divergence_b.cimt")
        assert res_b == res_b2

    def test_get_streamlines_fast(self):
        res_a = imtft.get_streamlines_fast(self.VF1_nomask, [13, 3.4])
        # imtio.export_to_file(res_a, "VF1_get_streamlines_fast_a.cimt")
        res_a2 = imtio.import_from_file("VF1_get_streamlines_fast_a.cimt")
        assert res_a[0] == res_a2[0]

    def test_get_streamlines(self):
        res_a = imtft.get_streamlines(self.VF1, [13, 3.4])
        # imtio.export_to_file(res_a, "VF1_get_streamlines_a.cimt")
        res_a2 = imtio.import_from_file("VF1_get_streamlines_a.cimt")
        assert res_a == res_a2

    def test_get_fieldlines(self):
        res_a = imtft.get_fieldlines(self.VF1, [[13, 3.4], [3.2, 5.43]])
        # imtio.export_to_file(res_a, "VF1_get_fieldlines_a.cimt")
        res_a2 = imtio.import_from_file("VF1_get_fieldlines_a.cimt")
        assert res_a == res_a2

    def test_get_tracklines(self):
        res_a = imtft.get_tracklines(self.VF1, [13, 3.4])
        # imtio.export_to_file(res_a, "VF1_get_tracklines_a.cimt")
        res_a2 = imtio.import_from_file("VF1_get_tracklines_a.cimt")
        assert res_a == res_a2

    def test_get_shear_stress(self):
        res_a = imtft.get_shear_stress(self.VF1)
        # imtio.export_to_file(res_a, "VF1_get_shear_stress_a.cimt")
        res_a2 = imtio.import_from_file("VF1_get_shear_stress_a.cimt")
        assert res_a == res_a2

    def test_get_swirling_vector(self):
        with pytest.raises(Warning):
            res_a = imtft.get_swirling_vector(self.VF1)
        # imtio.export_to_file(res_a, "VF1_get_swirling_vector_a.cimt")
        # res_a2 = imtio.import_from_file("VF1_get_swirling_vector_a.cimt")
        # assert res_a == res_a2

    def test_extrapolated_until_wall(self):
        res_a = imtft.extrapolate_until_wall(self.SF1, 'x', position=-0.4)
        # imtio.export_to_file(res_a, "SF1_extrapolate_until_wall_a.cimt")
        res_a2 = imtio.import_from_file("SF1_extrapolate_until_wall_a.cimt")
        assert res_a == res_a2
        #
        res_b = imtft.extrapolate_until_wall(self.VF1, 'y', position=-3.4)
        # imtio.export_to_file(res_b, "VF1_extrapolate_until_wall_b.cimt")
        res_b2 = imtio.import_from_file("VF1_extrapolate_until_wall_b.cimt")
        assert res_b == res_b2
        #
        res_c = imtft.extrapolate_until_wall(self.VF1, 'x', position=50.1,
                                             kind_interp='cubic')
        # imtio.export_to_file(res_c, "VF1_extrapolate_until_wall_c.cimt")
        res_c2 = imtio.import_from_file("VF1_extrapolate_until_wall_c.cimt")
        assert res_c == res_c2

# TEMP
pytest.main(['test_field_treatment.py'])
# pytest.main(['-v', 'test_field_treatment.py'])
# pytest.main(['--pdb', 'test_field_treatment.py'])
# TEMP - End
