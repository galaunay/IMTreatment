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

import matplotlib.pyplot as plt
from IMTreatment import VectorField, file_operation as imtio, make_unit


class TestVectorField(object):
    """ Done """

    def setup(self):
        sane_parameters()
        self.VF1 = imtio.import_from_file("VF1.cimt")
        self.VF1_nomask = imtio.import_from_file("VF1_nomask.cimt")
        self.VF2 = imtio.import_from_file("VF2.cimt")
        self.VF_notevenlyspaced = imtio.import_from_file("VF_notevenlyspaced.cimt")

    def test_import_from_arrays(self):
        # creating a VF field using 'import_from_arrays
        axe_x = np.arange(10)
        unit_x = make_unit('m')
        unit_y = make_unit('km')
        unit_values = make_unit('m/s')
        axe_y = np.arange(20)*.01
        vx = np.arange(len(axe_x)*len(axe_y)).reshape(len(axe_y), len(axe_x))
        vx = np.array(vx, dtype=float)
        vy = np.random.rand(len(axe_y), len(axe_x))
        mask = np.random.rand(len(axe_y), len(axe_x)) > 0.75
        vx[mask] = np.nan
        vx = np.pi*vx
        vy[mask] = np.nan
        vy = np.pi*vy
        vf = VectorField()
        vf.import_from_arrays(axe_x, axe_y, vx, vy, mask=mask,
                              unit_x=unit_x, unit_y=unit_y,
                              unit_values=unit_values)
        vx = vx.transpose()
        vy = vy.transpose()
        mask = mask.transpose()
        # tests
        assert np.all(vf.axe_x == axe_x)
        assert np.all(vf.axe_y == axe_y)
        assert np.all(vf.comp_x[~vf.mask] == vx[~mask])
        assert np.all(vf.comp_y[~vf.mask] == vy[~mask])
        assert np.all(vf.mask == mask)
        assert vf.unit_x == unit_x
        assert vf.unit_y == unit_y
        assert vf.unit_values == unit_values

    def test_operations(self):
        # get datas
        axe_x, axe_y = self.VF1.axe_x, self.VF1.axe_y
        vx = self.VF1.comp_x
        vy = self.VF1.comp_y
        mask = self.VF1.mask
        vx2 = self.VF2.comp_x
        vy2 = self.VF2.comp_y
        mask2 = self.VF2.mask
        unit_x, unit_y = self.VF1.unit_x, self.VF1.unit_y
        unit_values = self.VF1.unit_values
        # neg
        vf = -self.VF1.copy()
        assert np.all(vf.axe_x == axe_x)
        assert np.all(vf.axe_y == axe_y)
        assert np.all(vf.comp_x[~vf.mask] == -vx[~mask])
        assert np.all(vf.comp_y[~vf.mask] == -vy[~mask])
        assert np.all(vf.mask == mask)
        assert vf.unit_x == unit_x
        assert vf.unit_y == unit_y
        assert vf.unit_values == unit_values
        # add
        nmb = 5
        unt = 500*make_unit('mm/s')
        vx_f = (nmb + vx + unt.asNumber()/1000. + vx2 +
                unt.asNumber()/1000. + vx + nmb)
        vy_f = (nmb + vy + unt.asNumber()/1000. + vy2 +
                unt.asNumber()/1000. + vy + nmb)
        mask_f = np.logical_or(mask, mask2)
        vf = nmb + self.VF1 + unt + self.VF2 + unt + self.VF1 + nmb
        assert np.all(vf.axe_x == axe_x)
        assert np.all(vf.axe_y == axe_y)
        assert np.all(vf.comp_x[~mask_f] == vx_f[~mask_f])
        assert np.all(vf.comp_y[~mask_f] == vy_f[~mask_f])
        assert np.all(vf.mask == mask_f)
        assert vf.unit_x == unit_x
        assert vf.unit_y == unit_y
        assert vf.unit_values == unit_values
        # sub
        nmb = 5
        unt = 500*make_unit('mm/s')
        vx_f = (nmb - vx - unt.asNumber()/1000. - vx2 -
                unt.asNumber()/1000. - vx - nmb)
        vy_f = (nmb - vy - unt.asNumber()/1000. - vy2 -
                unt.asNumber()/1000. - vy - nmb)
        mask_f = np.logical_or(mask, mask2)
        vf = nmb - self.VF1 - unt - self.VF2 - unt - self.VF1 - nmb
        assert np.all(vf.axe_x == axe_x)
        assert np.all(vf.axe_y == axe_y)
        assert np.all(vf.comp_x[~mask_f] == vx_f[~mask_f])
        assert np.all(vf.comp_y[~mask_f] == vy_f[~mask_f])
        assert np.all(vf.mask == mask_f)
        assert vf.unit_x == unit_x
        assert vf.unit_y == unit_y
        assert vf.unit_values == unit_values
        # mul
        nmb = 5.23
        unt = 500.*make_unit('mm/s')
        unt_n = 500./1000.
        vx_f = (nmb * vx * unt_n + vx2 *
                unt_n * nmb)
        vy_f = (nmb * vy * unt_n + vy2 *
                unt_n * nmb)
        unit_values = make_unit('mm/s')*make_unit('m/s')*1e3
        mask_f = np.logical_or(mask, mask2)
        vf = nmb * self.VF1 * unt + self.VF2 * unt * nmb
        assert np.all(vf.axe_x == axe_x)
        assert np.all(vf.axe_y == axe_y)
        assert np.all(vf.mask == mask_f)
        assert np.all(vf.comp_x[~vf.mask] - vx_f[~mask_f] < 1e-6)
        assert np.all(vf.comp_y[~vf.mask] - vy_f[~mask_f] < 1e-6)
        assert vf.unit_x == unit_x
        assert vf.unit_y == unit_y
        assert vf.unit_values == unit_values
        # div
        nmb = 5.23
        unt = 500.*make_unit('kg*mm/s')
        unt_n = 500./1000.
        vx_f = nmb / (vx / unt_n) + 1./(vx2 / unt_n)
        vy_f = nmb / (vy / unt_n) + 1./(vy2 / unt_n)
        unit_values = 1./(make_unit('m/s')/make_unit('kg*mm/s'))*1e3
        mask_f = np.logical_or(mask, mask2)
        vf = nmb / (self.VF1 / unt) + 1./(self.VF2 / unt)
        assert np.all(vf.axe_x == axe_x)
        assert np.all(vf.axe_y == axe_y)
        assert np.all(vf.mask == mask_f)
        assert np.all(vf.comp_x[~vf.mask] - vx_f[~mask_f] < 1e-6)
        assert np.all(vf.comp_y[~vf.mask] - vy_f[~mask_f] < 1e-6)
        assert vf.unit_x == unit_x
        assert vf.unit_y == unit_y
        assert vf.unit_values == unit_values
        # abs
        unit_values = self.VF1.unit_values
        vf = np.abs(self.VF1)
        assert np.all(vf.axe_x == axe_x)
        assert np.all(vf.axe_y == axe_y)
        assert np.all(vf.comp_x[~vf.mask] == np.abs(vx[~mask]))
        assert np.all(vf.comp_y[~vf.mask] == np.abs(vy[~mask]))
        assert np.all(vf.mask == mask)
        assert vf.unit_x == unit_x
        assert vf.unit_y == unit_y
        assert vf.unit_values == unit_values
        # pow
        unit_values = self.VF1.unit_values**3.544186
        vf = (np.abs(self.VF1) + 1)**3.544186
        assert np.all(vf.axe_x == axe_x)
        assert np.all(vf.axe_y == axe_y)
        assert np.all(vf.comp_x[~vf.mask] -
                      (np.abs(vx[~mask]) + 1)**3.544186 <
                      1e-6)
        assert np.all(vf.comp_y[~vf.mask] -
                      (np.abs(vy[~mask]) + 1)**3.544186 <
                      1e-6)
        assert np.all(vf.mask == mask)
        assert vf.unit_x == unit_x
        assert vf.unit_y == unit_y
        assert vf.unit_values == unit_values

    def test_iter(self):
        res = list(self.VF1.__iter__())
        # imtio.export_to_file(res, "VF1_iter.cimt")
        res2 = imtio.import_from_file("VF1_iter.cimt")
        assert res == res2

    def test_trim_area(self):
        axe_x, axe_y = self.VF1.axe_x, self.VF1.axe_y
        vx = self.VF1.comp_x
        vy = self.VF1.comp_y
        mask = self.VF1.mask
        vf = self.VF1.crop([axe_x[3], axe_x[-4]], [axe_y[2], axe_y[-7]])
        assert np.all(vf.axe_x == axe_x[3:-3])
        assert np.all(vf.axe_y == axe_y[2:-6])
        assert np.all(vf.comp_x[~vf.mask] ==
                      vx[3:-3, 2:-6][~mask[3:-3, 2:-6]])
        assert np.all(vf.comp_y[~vf.mask] ==
                      vy[3:-3, 2:-6][~mask[3:-3, 2:-6]])

    def test_min_max(self):
        mini = self.VF1.min
        maxi = self.VF1.max
        assert mini == 0.54631058701128665
        assert maxi == 97.570083275983819

    def test_get_value(self):
        res = self.VF1.get_value(7.5, 20.12)
        # imtio.export_to_file(res, "VF1_get_value.cimt")
        res2 = imtio.import_from_file("VF1_get_value.cimt")
        assert np.all(res == res2)
        res_b = self.VF1.get_value(5, 10, ind=True)
        # imtio.export_to_file(res_b, "VF1_get_value_b.cimt")
        res_b2 = imtio.import_from_file("VF1_get_value_b.cimt")
        assert np.all(res_b == res_b2)

    def test_magnitude(self):
        res = self.VF1.magnitude
        # imtio.export_to_file(res, "VF1_magnitude.cimt")
        res2 = imtio.import_from_file("VF1_magnitude.cimt")
        assert np.all(res[~np.isnan(res)] == res2[~np.isnan(res2)])

    def test_magnitude_as_sf(self):
        res = self.VF1.magnitude_as_sf
        # imtio.export_to_file(res, "VF1_magnitude_as_sf.cimt")
        res2 = imtio.import_from_file("VF1_magnitude_as_sf.cimt")
        assert res == res2

    def test_theta(self):
        res = self.VF1.theta
        # imtio.export_to_file(res, "VF1_theta.cimt")
        res2 = imtio.import_from_file("VF1_theta.cimt")
        assert np.all(res == res2)

    def test_theta_as_sf(self):
        res = self.VF1.theta_as_sf
        # imtio.export_to_file(res, "VF1_theta_as_sf.cimt")
        res2 = imtio.import_from_file("VF1_theta_as_sf.cimt")
        assert res == res2

    def test_get_profile(self):
        res = self.VF1.get_profile('vx', 'x', 7.2)
        # imtio.export_to_file(res, "VF1_get_profile.cimt")
        res2 = imtio.import_from_file("VF1_get_profile.cimt")
        assert res == res2
        res_b = self.VF1.get_profile('vy', 'x', 3, ind=True)
        # imtio.export_to_file(res_b, "VF1_get_profile_b.cimt")
        res_b2 = imtio.import_from_file("VF1_get_profile_b.cimt")
        assert res_b == res_b2
        res_c = self.VF1.get_profile('vy', 'y', 2.34)
        # imtio.export_to_file(res_c, "VF1_get_profile_c.cimt")
        res_c2 = imtio.import_from_file("VF1_get_profile_c.cimt")
        assert res_c == res_c2
        res_d = self.VF1.get_profile('vx', 'y', 19, ind=True)
        # imtio.export_to_file(res_d, "VF1_get_profile_d.cimt")
        res_d2 = imtio.import_from_file("VF1_get_profile_d.cimt")
        assert res_d == res_d2

    def test_copy(self):
        res_a = self.VF1.copy()
        # imtio.export_to_file(res_a, "VF1_copy_a.cimt")
        res_a2 = imtio.import_from_file("VF1_copy_a.cimt")
        assert res_a == res_a2

    def test_get_norm(self):
        res_a = self.VF1.get_norm()
        # imtio.export_to_file(res_a, "VF1_get_norm_a.cimt")
        res_a2 = imtio.import_from_file("VF1_get_norm_a.cimt")
        assert res_a == res_a2

    def test_scale(self):
        res_a = self.VF1.scale(scalex=1.65, scaley=0.64, scalev=9,
                               inplace=False)
        # imtio.export_to_file(res_a, "VF1_scale_a.cimt")
        res_a2 = imtio.import_from_file("VF1_scale_a.cimt")
        assert res_a == res_a2

    def test_rotate(self):
        res_a = self.VF1.copy()
        res_a.rotate(270)
        # imtio.export_to_file(res_a, "VF1_rotate_a.cimt")
        res_a2 = imtio.import_from_file("VF1_rotate_a.cimt")
        assert res_a == res_a2

    def test_change_unit(self):
        res_a = self.VF1.copy()
        res_a.change_unit("x", "km")
        res_a.change_unit('y', 'um')
        res_a.change_unit('values', 'km/us')
        # imtio.export_to_file(res_a, "VF1_change_unit_a.cimt")
        res_a2 = imtio.import_from_file("VF1_change_unit_a.cimt")
        assert res_a == res_a2

    def test_smooth(self):
        res_a = self.VF1.smooth(tos='uniform', size=7, inplace=False)
        # imtio.export_to_file(res_a, "VF1_smooth_a.cimt")
        res_a2 = imtio.import_from_file("VF1_smooth_a.cimt")
        assert res_a == res_a2

    def test_make_evenly_spaced(self):
        res_a = self.VF_notevenlyspaced.make_evenly_spaced()
        # imtio.export_to_file(res_a, "VF1_make_evenly_spaced_a.cimt")
        res_a2 = imtio.import_from_file("VF1_make_evenly_spaced_a.cimt")
        assert res_a == res_a2

    def test_fill(self):
        res_a = self.VF1.fill(inplace=False)
        # imtio.export_to_file(res_a, "VF1_fill_a.cimt")
        res_a2 = imtio.import_from_file("VF1_fill_a.cimt")
        assert res_a == res_a2

    def test_crop(self):
        res = self.VF1.crop(intervx=[7.4, 19.0],
                            intervy=[2.1, 8], inplace=False)
        # imtio.export_to_file(res, "VF1_crop.cimt")
        res2 = imtio.import_from_file("VF1_crop.cimt")
        assert res == res2

    def test_crop_masked_border(self):
        tmpvf = self.VF1.copy()
        tmpvf.mask[0:3, :] = True
        tmpvf.mask[-1:, :] = True
        tmpvf.mask[:, 0:5] = True
        tmpvf.mask[:, -4:] = True
        res = tmpvf.crop_masked_border(inplace=False)
        # imtio.export_to_file(res, "VF1_crop_masked_border.cimt")
        res2 = imtio.import_from_file("VF1_crop_masked_border.cimt")
        assert res == res2
        #
        tmpvf = self.VF1_nomask.copy()
        tmpvf.mask[3:-2, :] = False
        tmpvf.mask[:, 5:-4] = False
        res = self.VF1.crop_masked_border(hard=True, inplace=False)
        # imtio.export_to_file(res, "VF1_crop_masked_border_2.cimt")
        res2 = imtio.import_from_file("VF1_crop_masked_border_2.cimt")
        assert res == res2

    def test_extend(self):
        res_a = self.VF1.extend(nmb_down=4, nmb_left=2, nmb_right=7, nmb_up=12,
                                inplace=False)
        # imtio.export_to_file(res_a, "VF1_extend_a.cimt")
        res_a2 = imtio.import_from_file("VF1_extend_a.cimt")
        assert res_a == res_a2

    def test_mirroring(self):
        res_a = self.VF1.mirroring('x', 4)
        # imtio.export_to_file(res_a, "VF1_mirroring_a.cimt")
        res_a2 = imtio.import_from_file("VF1_mirroring_a.cimt")
        assert res_a == res_a2
        #
        res_b = self.VF1.copy()
        res_b.mirroring('x', 10, mir_coef=-1.2, inplace=True)
        # imtio.export_to_file(res_b, "VF1_mirroring_b.cimt")
        res_b2 = imtio.import_from_file("VF1_mirroring_b.cimt")
        assert res_b == res_b2

    def test_reduce_spatial_resolution(self):
        res = self.VF1.reduce_spatial_resolution(4, inplace=False)
        # imtio.export_to_file(res, "VF1_reduce_spatial_resolution.cimt")
        res2 = imtio.import_from_file("VF1_reduce_spatial_resolution.cimt")
        assert res == res2

    @pytest.mark.mpl_image_compare
    def test_display_VF_a(self):
        fig = plt.figure()
        self.VF1.display()
        return fig

    @pytest.mark.mpl_image_compare
    def test_display_VF_b(self):
        fig = plt.figure()
        self.VF1.display("magnitude")
        return fig

    @pytest.mark.mpl_image_compare
    def test_display_VF_c(self):
        fig = plt.figure()
        self.VF1.display("mask")
        return fig

    @pytest.mark.mpl_image_compare
    def test_display_VF_d(self):
        fig = plt.figure()
        self.VF1.display("x")
        return fig

    @pytest.mark.mpl_image_compare
    def test_display_VF_e(self):
        fig = plt.figure()
        self.VF1.display("y")
        return fig

    @pytest.mark.mpl_image_compare
    def test_display_VF_f(self):
        fig = plt.figure()
        self.VF1.display(kind='stream')
        return fig

    @pytest.mark.mpl_image_compare
    def test_display_VF_g(self):
        fig = plt.figure()
        self.VF1.display(kind='stream', density=2, color='k')
        return fig

# TEMP
pytest.main(["test_vectorfield.py"])
# TEMP - End
