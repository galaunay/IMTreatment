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

import matplotlib.pyplot as plt
import numpy as np

import pytest
import unum
from IMTreatment import (field_treatment as imtft, file_operation as imtio,
                         make_unit, TemporalVectorFields, TemporalScalarFields)
from IMTreatment.core import TemporalVectorFields


class TestFileOperation(object):
    """ Done """

    def setup(self):
        try:
            os.chdir(os.path.dirname(os.path.realpath(__file__)))
        except:
            pass
        self.SF1_nomask = imtio.import_from_file("SF1_nomask.cimt")
        self.SF1 = imtio.import_from_file("SF1.cimt")
        self.VF1 = imtio.import_from_file("VF1.cimt")
        self.VF2 = imtio.import_from_file("VF2.cimt")
        self.Prof1 = imtio.import_from_file("Prof1.cimt")
        self.P1 = imtio.import_from_file("P1.cimt")

        TSF1 = TemporalScalarFields()
        for i in range(10):
            TSF1.add_field(self.SF1*np.cos(i/10*np.pi), i*0.8)
        imtio.export_to_file(TSF1, "TSF1.cimt")
        TSF1_nomask = TemporalScalarFields()
        for i in range(10):
            TSF1_nomask.add_field(self.SF1*np.cos(i/10*np.pi), i*0.8)
        imtio.export_to_file(TSF1_nomask, "TSF1_nomask.cimt")

        self.TSF1 = imtio.import_from_file("TSF1.cimt")
        self.TSF1_nomask = imtio.import_from_file("TSF1_nomask.cimt")

        # self.VF1_nomask = imtio.import_from_file("VF1_nomask.cimt")
        # self.SF2 = imtio.import_from_file("SF2.cimt")
        # self.TVF1 = imtio.import_from_file("TVF1.cimt")

    def test_imt_to_imts(self):
        imtio.imts_to_imt("VFs_imts_to_imt",
                          "TVF_imts_to_imt_a.cimt",
                          'TVF')
        TVF = TemporalVectorFields()
        TVF.add_field(self.VF1)
        TVF.add_field(self.VF2)
        TVF2 = imtio.import_from_file("TVF_imts_to_imt_a.cimt")
        assert TVF == TVF2

    def test_import_export_to_matlab(self):
        imtio.export_to_matlab(self.VF1, 'VF1.mat')
        VF1b = imtio.import_from_matlab('VF1.mat', 'VectorField',
                                        axe_x='axe_x',
                                        axe_y='axe_y',
                                        unit_x='unit_x',
                                        unit_y='unit_y',
                                        comp_x='comp_x',
                                        comp_y='comp_y',
                                        mask='mask',
                                        unit_values='unit_values')
        assert self.VF1 == VF1b
        #
        imtio.export_to_matlab(self.SF1, 'SF1.mat')
        SF1b = imtio.import_from_matlab('SF1.mat', 'ScalarField',
                                        axe_x='axe_x',
                                        axe_y='axe_y',
                                        unit_x='unit_x',
                                        unit_y='unit_y',
                                        values='values',
                                        mask='mask',
                                        unit_values='unit_values')
        assert self.SF1 == SF1b
        #
        imtio.export_to_matlab(self.Prof1, 'Prof1.mat')
        Prof1b = imtio.import_from_matlab('Prof1.mat', 'Profile',
                                          x='x',
                                          y='y',
                                          mask='mask',
                                          unit_x='unit_x',
                                          unit_y='unit_y')
        assert self.Prof1 == Prof1b
        #
        imtio.export_to_matlab(self.P1, 'P1.mat')
        P1b = imtio.import_from_matlab('P1.mat', 'Points',
                                       xy='xy',
                                       v='v',
                                       unit_x='unit_x',
                                       unit_y='unit_y',
                                       unit_v='unit_v')
        assert self.P1 == P1b

    def test_export_to_vtk(self):
        try:
            import pyvtk
        except ImportError:
            with pytest.raises(Exception):
                imtio.export_to_vtk(self.VF1, "VF1_export_to_vtk_a.cimt")
            with pytest.raises(Exception):
                imtio.export_to_vtk(self.SF1, "SF1_export_to_vtk_a.cimt")
            with pytest.raises(Exception):
                imtio.export_to_vtk(self.P1, "P1_export_to_vtk_a.cimt")
        else:
            imtio.export_to_vtk(self.VF1, "VF1_export_to_vtk_a.cimt")
            res_a2 = imtio.import_from_file("VF1_export_to_vtk_a.cimt")
            assert self.VF1 == res_a2
            #
            imtio.export_to_vtk(self.SF1, "SF1_export_to_vtk_a.cimt")
            res_a2 = imtio.import_from_file("SF1_export_to_vtk_a.cimt")
            assert self.SF1 == res_a2
            #
            imtio.export_to_vtk(self.P1, "P1_export_to_vtk_a.cimt")
            res_a2 = imtio.import_from_file("P1_export_to_vtk_a.cimt")
            assert self.P1 == res_a2

    def test_import_from_picture(self):
        imtio.export_to_picture(self.SF1_nomask, "SF1.png")
        res_a = imtio.import_from_picture("SF1.png")
        # imtio.export_to_file(res_a, "SF1_from_png.cimt")
        res_a2 = imtio.import_from_file("SF1_from_png.cimt")
        assert res_a == res_a2

    def test_import_from_pictures(self):
        imtio.export_to_pictures(self.TSF1_nomask, "TSF1")
        res_a = imtio.import_from_pictures("TSF1[0-9]+.png")
        # imtio.export_to_file(res_a, "TSF1_from_pngs.cimt")
        res_a2 = imtio.import_from_file("TSF1_from_pngs.cimt")
        assert res_a == res_a2

    def test_import_profile_from_ascii(self):
        imtio.export_to_ascii("Prof1.txt", self.Prof1)
        res_a = imtio.import_profile_from_ascii("Prof1.txt", x_col=1,
                                                y_col=2,
                                                unit_x=self.Prof1.unit_x,
                                                unit_y=self.Prof1.unit_y)
        assert res_a == self.Prof1

    def test_import_sf_from_ascii(self):
        imtio.export_to_ascii("SF1.txt", self.SF1)
        res_a = imtio.import_sf_from_ascii("SF1.txt",
                                           x_col=1,
                                           y_col=2,
                                           vx_col=3,
                                           unit_x=self.SF1.unit_x,
                                           unit_y=self.SF1.unit_y,
                                           unit_values=self.SF1.unit_values)
        assert res_a == self.SF1

    def test_import_vf_from_ascii(self):
        imtio.export_to_ascii("VF1.txt", self.VF1)
        res_a = imtio.import_vf_from_ascii("VF1.txt",
                                           x_col=1,
                                           y_col=2,
                                           vx_col=3,
                                           vy_col=4,
                                           unit_x=self.VF1.unit_x,
                                           unit_y=self.VF1.unit_y,
                                           unit_values=self.VF1.unit_values)
        assert res_a == self.VF1

    def test_import_points_from_ascii(self):
        imtio.export_to_ascii("P1.txt", self.P1)
        res_a = imtio.import_pts_from_ascii("P1.txt",
                                            x_col=1,
                                            y_col=2,
                                            v_col=3,
                                            unit_x=self.P1.unit_x,
                                            unit_y=self.P1.unit_y,
                                            unit_v=self.P1.unit_v)
        assert res_a == self.P1
        #
        imtio.export_to_ascii("P1_b.txt", self.P1)
        res_a = imtio.import_pts_from_ascii("P1_b.txt",
                                            x_col=1,
                                            y_col=2,
                                            unit_x=self.P1.unit_x,
                                            unit_y=self.P1.unit_y)
        assert np.all(res_a.xy == self.P1.xy)
        assert res_a.unit_x == self.P1.unit_x
        assert res_a.unit_y == self.P1.unit_y



# TEMP
# pytest.main(['test_file_operation.py'])
# pytest.main(['-v', 'test_file_operation.py'])
# pytest.main(['--pdb', 'test_file_operation.py'])
# TEMP - End
