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

import numpy as np
import pytest
from helper import sane_parameters, parametric_test

import unum
from IMTreatment import Points, file_operation as imtio, make_unit
import matplotlib.pyplot as plt


class TestPoints(object):

    def setup(self):
        sane_parameters()
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
        # imtio.export_to_file(res_a, "P1_copy_a.cimt")
        res_a2 = imtio.import_from_file("P1_copy_a.cimt")
        assert res_a == res_a2

    @pytest.mark.mpl_image_compare
    def test_display_Points_a(self):
        fig = plt.figure()
        self.P1.display()
        return fig

    @pytest.mark.mpl_image_compare
    def test_display_Points_b(self):
        fig = plt.figure()
        self.P1.display(kind='scatter')
        return fig

    @pytest.mark.mpl_image_compare
    def test_display_Points_c(self):
        fig = plt.figure()
        self.P1.display(kind='colored_plot')
        return fig

    @pytest.mark.mpl_image_compare
    def test_display_Points_d(self):
        fig = plt.figure()
        self.P1.display(axe_x="y", axe_y='v')
        return fig

    @pytest.mark.mpl_image_compare
    def test_display_Points_e(self):
        fig = plt.figure()
        self.P1.display(axe_x="v", axe_y='x', axe_color='y')
        return fig

    @pytest.mark.mpl_image_compare
    def test_display_Points_f(self):
        fig = plt.figure()
        self.P1.display(kind='plot', marker=".", color='b', ls=':')
        return fig

if __name__ == "__main__":
    pytest.main(['test_points.py'])
