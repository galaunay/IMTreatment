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
from helper import parametric_test, sane_parameters

import unum
from IMTreatment import Profile, file_operation as imtio, make_unit
import matplotlib.pyplot as plt


class TestProfile(object):
    """ Done """

    def setup(self):
        sane_parameters()
        # unit_x = make_unit('m')
        # unit_y = make_unit('m/s')
        # x = np.genfromtxt('prof_x')
        # y = np.genfromtxt('prof_y')
        # y2 = np.genfromtxt('prof_y2')
        # mask = np.genfromtxt('prof_mask')
        # self.Prof1 = Profile(x, y, mask=mask, unit_x=unit_x,
        #                      unit_y=unit_y)
        # self.Prof2 = Profile(x, y2, mask=mask, unit_x=unit_x,
        #                      unit_y=unit_y)
        # self.Prof_evenlyspaced = Profile(np.linspace(0, 30, 100),
        #                                  y, mask=mask, unit_x=unit_x,
        #                                  unit_y=unit_y)
        # self.Prof2_evenlyspaced = Profile(np.linspace(0, 30, 100),
        #                                   y2, mask=mask, unit_x=unit_x,
        #                                   unit_y=unit_y)
        # self.Prof_nomask = Profile(x, y, mask=False, unit_x=unit_x,
        #                            unit_y=unit_y)
        # self.Prof_evenlyspaced_nomask = Profile(np.linspace(0, 30, 100),
        #                                         y, mask=False, unit_x=unit_x,
        #                                         unit_y=unit_y)
        # imtio.export_to_file(self.Prof1, "Prof1.cimt")
        # imtio.export_to_file(self.Prof2, "Prof2.cimt")
        # imtio.export_to_file(self.Prof_evenlyspaced, "Prof_evenlyspaced.cimt")
        # imtio.export_to_file(self.Prof2_evenlyspaced, "Prof2_evenlyspaced.cimt")
        # imtio.export_to_file(self.Prof_nomask, "Prof_nomask.cimt")
        # imtio.export_to_file(self.Prof_evenlyspaced_nomask,
        #                      "Prof_evenlyspaced_nomask.cimt")
        self.Prof1 = imtio.import_from_file('Prof1.cimt')
        self.Prof2 = imtio.import_from_file('Prof2.cimt')
        self.Prof_evenlyspaced = imtio.import_from_file('Prof_evenlyspaced.cimt')
        self.Prof2_evenlyspaced = imtio.import_from_file('Prof2_evenlyspaced.cimt')
        self.Prof_nomask = imtio.import_from_file('Prof_nomask.cimt')
        self.Prof_evenlyspaced_nomask = imtio.import_from_file('Prof_evenlyspaced_nomask.cimt')

    def test_properties(self):
        prof = self.Prof1
        prof.unit_x = 'kg'
        prof.unit_y = 's'
        assert prof.unit_x.strUnit() == make_unit('kg').strUnit()
        assert prof.unit_y.strUnit() == make_unit('s').strUnit()

    def test_get_props(self):
        fun = self.Prof1.get_props
        kwargs = [{}]
        parametric_test(fun, kwargs)

    def test_add(self):
        res_a = self.Prof2 + self.Prof1
        # imtio.export_to_file(res_a, "Prof1_add_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_add_a.cimt")
        assert res_a == res_a2
        #
        res_b = self.Prof2 + 18
        # imtio.export_to_file(res_b, "Prof1_add_b.cimt")
        res_b2 = imtio.import_from_file("Prof1_add_b.cimt")
        assert res_b == res_b2
        #
        res_c = self.Prof2 + 18*make_unit('km/s')
        # imtio.export_to_file(res_c, "Prof1_add_c.cimt")
        res_c2 = imtio.import_from_file("Prof1_add_c.cimt")
        assert res_c == res_c2

    def test_sub(self):
        res_a = self.Prof1 - self.Prof2
        # imtio.export_to_file(res_a, "Prof1_sub_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_sub_a.cimt")
        assert res_a == res_a2
        #
        res_b = self.Prof1 - 39
        # imtio.export_to_file(res_b, "Prof1_sub_b.cimt")
        res_b2 = imtio.import_from_file("Prof1_sub_b.cimt")
        assert res_b == res_b2
        #
        res_c = self.Prof1 - 23*make_unit('km/us')
        # imtio.export_to_file(res_c, "Prof1_sub_c.cimt")
        res_c2 = imtio.import_from_file("Prof1_sub_c.cimt")
        assert res_c == res_c2

    def test_mul(self):
        res_a = self.Prof1 * self.Prof2
        # imtio.export_to_file(res_a, "Prof1_mul_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_mul_a.cimt")
        assert res_a == res_a2
        #
        res_b = self.Prof1 * 14
        # imtio.export_to_file(res_b, "Prof1_mul_b.cimt")
        res_b2 = imtio.import_from_file("Prof1_mul_b.cimt")
        assert res_b == res_b2
        #
        res_c = self.Prof1 * 32*make_unit('Hz')
        # imtio.export_to_file(res_c, "Prof1_mul_c.cimt")
        res_c2 = imtio.import_from_file("Prof1_mul_c.cimt")
        assert res_c == res_c2

    def test_div(self):
        res_a = self.Prof1 / self.Prof2
        # imtio.export_to_file(res_a, "Prof1_div_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_div_a.cimt")
        assert res_a == res_a2
        #
        res_b = self.Prof1 / 14
        # imtio.export_to_file(res_b, "Prof1_div_b.cimt")
        res_b2 = imtio.import_from_file("Prof1_div_b.cimt")
        assert res_b == res_b2
        #
        res_c = self.Prof1 / (32*make_unit('Hz'))
        # imtio.export_to_file(res_c, "Prof1_div_c.cimt")
        res_c2 = imtio.import_from_file("Prof1_div_c.cimt")
        assert res_c == res_c2

    def test_power(self):
        res_a = self.Prof1**3.1564
        # imtio.export_to_file(res_a, "Prof1_power_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_power_a.cimt")
        assert res_a == res_a2

    def test_min(self):
        res_a = self.Prof1.min
        # imtio.export_to_file(res_a, "Prof1_min_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_min_a.cimt")
        assert res_a == res_a2

    def test_max(self):
        res_a = self.Prof1.max
        # imtio.export_to_file(res_a, "Prof1_max_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_max_a.cimt")
        assert res_a == res_a2

    def test_mean(self):
        res_a = self.Prof1.mean
        # imtio.export_to_file(res_a, "Prof1_mean_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_mean_a.cimt")
        assert res_a == res_a2

    def test_get_interpolator(self):
        res_a = self.Prof1.get_interpolator()(13)
        # imtio.export_to_file(res_a, "Prof1_get_interpolator_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_get_interpolator_a.cimt")
        assert res_a == res_a2

    def test_get_interpolated_values(self):
        res_a = self.Prof1.get_interpolated_values(x=13.2)
        # imtio.export_to_file(res_a, "Prof1_get_interpolated_values_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_get_interpolated_values_a.cimt")
        assert res_a == res_a2
        #
        res_b = self.Prof1.get_interpolated_values(y=12.12)
        # imtio.export_to_file(res_b, "Prof1_get_interpolated_values_b.cimt")
        res_b2 = imtio.import_from_file("Prof1_get_interpolated_values_b.cimt")
        assert res_b == res_b2
        #
        res_c = self.Prof1.get_interpolated_values(x=78, ind=True)
        # imtio.export_to_file(res_c, "Prof1_get_interpolated_values_c.cimt")
        res_c2 = imtio.import_from_file("Prof1_get_interpolated_values_c.cimt")
        assert res_c == res_c2
        #
        res_d = self.Prof1.get_interpolated_values(x=78, ind=True)
        # imtio.export_to_file(res_d, "Prof1_get_interpolated_values_d.cimt")
        res_d2 = imtio.import_from_file("Prof1_get_interpolated_values_d.cimt")
        assert res_d == res_d2
        #
        res_e = self.Prof1.get_interpolated_values(x=[13.2, 14.2, 15.2, 16.2])
        # imtio.export_to_file(res_e, "Prof1_get_interpolated_values_e.cimt")
        res_e2 = imtio.import_from_file("Prof1_get_interpolated_values_e.cimt")
        assert np.all(res_e[~np.isnan(res_e)] == res_e2[~np.isnan(res_e2)])
        #
        res_f = self.Prof1.get_interpolated_values(y=[.2, .3, .5, .7])
        # imtio.export_to_file(res_f, "Prof1_get_interpolated_values_f.cimt")
        res_f2 = imtio.import_from_file("Prof1_get_interpolated_values_f.cimt")
        assert np.all(res_f[~np.isnan(res_f)] == res_f2[~np.isnan(res_f2)])

    def test_get_value_position(self):
        res_a = self.Prof1.get_value_position(self.Prof1.mean)
        # imtio.export_to_file(res_a, "Prof1_get_value_position_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_get_value_position_a.cimt")
        assert np.all(res_a == res_a2)

    def test_get_integral(self):
        res_a = self.Prof1.get_integral()
        # imtio.export_to_file(res_a, "Prof1_get_integral_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_get_integral_a.cimt")
        assert res_a == res_a2

    def test_get_gradient(self):
        res_a = self.Prof_evenlyspaced.get_gradient()
        # imtio.export_to_file(res_a, "Prof1_get_gradient_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_get_gradient_a.cimt")
        assert res_a == res_a2
        #
        res_b = self.Prof_evenlyspaced_nomask.get_gradient(position=14.36)
        # imtio.export_to_file(res_b, "Prof1_get_gradient_b.cimt")
        res_b2 = imtio.import_from_file("Prof1_get_gradient_b.cimt")
        assert res_b == res_b2
        #
        res_c = self.Prof_evenlyspaced_nomask.get_gradient(
            position=self.Prof_evenlyspaced_nomask.x[0] - 1)
        # imtio.export_to_file(res_c, "Prof1_get_gradient_c.cimt")
        res_c2 = imtio.import_from_file("Prof1_get_gradient_c.cimt")
        assert res_c == res_c2
        #
        res_d = self.Prof_evenlyspaced_nomask.get_gradient(
            position=self.Prof_evenlyspaced_nomask.x[-1] + 1)
        # imtio.export_to_file(res_d, "Prof1_get_gradient_d.cimt")
        res_d2 = imtio.import_from_file("Prof1_get_gradient_d.cimt")
        assert res_d == res_d2

    def test_get_spectrum(self):
        res_a = self.Prof1.get_spectrum()
        # imtio.export_to_file(res_a, "Prof1_get_spectrum_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_get_spectrum_a.cimt")
        assert res_a == res_a2
        #
        res_b = self.Prof1.get_spectrum(welch_seglen=10)
        # imtio.export_to_file(res_b, "Prof1_get_spectrum_b.cimt")
        res_b2 = imtio.import_from_file("Prof1_get_spectrum_b.cimt")
        assert res_b == res_b2
        #
        res_c = self.Prof1.get_spectrum(scaling="spectrum", detrend='linear')
        # imtio.export_to_file(res_c, "Prof1_get_spectrum_c.cimt")
        res_c2 = imtio.import_from_file("Prof1_get_spectrum_c.cimt")
        assert res_c == res_c2
        #
        res_d = self.Prof1.get_spectrum(scaling="density", detrend='none')
        # imtio.export_to_file(res_d, "Prof1_get_spectrum_d.cimt")
        res_d2 = imtio.import_from_file("Prof1_get_spectrum_d.cimt")
        assert res_d == res_d2
        #
        res_e = self.Prof1.get_spectrum(scaling="density")
        # imtio.export_to_file(res_e, "Prof1_get_spectrum_e.cimt")
        res_e2 = imtio.import_from_file("Prof1_get_spectrum_e.cimt")
        assert res_e == res_e2

    def test_get_wavelet_transform(self):
        res_a = self.Prof_evenlyspaced.get_wavelet_transform()
        # imtio.export_to_file(res_a, "Prof1_get_wavelet_transform_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_get_wavelet_transform_a.cimt")
        assert res_a == res_a2
        #
        res_b = self.Prof_evenlyspaced.get_wavelet_transform(
            widths=np.linspace(0, len(self.Prof_evenlyspaced_nomask.y) - 1, 10)[1::])
        # imtio.export_to_file(res_b, "Prof1_get_wavelet_transform_b.cimt")
        res_b2 = imtio.import_from_file("Prof1_get_wavelet_transform_b.cimt")
        assert res_b == res_b2
        #
        res_c = self.Prof_evenlyspaced.get_wavelet_transform(fill=2)
        # imtio.export_to_file(res_c, "Prof1_get_wavelet_transform_c.cimt")
        res_c2 = imtio.import_from_file("Prof1_get_wavelet_transform_c.cimt")
        assert res_c == res_c2

    def test_get_pdf(self):
        res_a = self.Prof1.get_pdf()
        # imtio.export_to_file(res_a, "Prof1_get_pdf_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_get_pdf_a.cimt")
        assert res_a == res_a2

    def test_get_auto_correlation(self):
        res_a = self.Prof1.get_auto_correlation(13)
        # imtio.export_to_file(res_a, "Prof1_get_auto_correlation_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_get_auto_correlation_a.cimt")
        assert res_a == res_a2

    def test_get_fitting(self):
        def fit_fun(x, a, b, c, d):
            return a*x + b*x**2 + c*x**3 + d
        res_a = self.Prof1.get_fitting(fit_fun)
        # imtio.export_to_file(res_a, "Prof1_get_fitting_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_get_fitting_a.cimt")
        assert res_a == res_a2
        #
        res_b = self.Prof1.get_fitting(fit_fun, output_param=True)
        # imtio.export_to_file(res_b, "Prof1_get_fitting_b.cimt")
        res_b2 = imtio.import_from_file("Prof1_get_fitting_b.cimt")
        assert res_b[0] == res_b2[0]
        assert np.all(res_b[1] == res_b2[1])

    def test_get_distribution(self):
        fun = self.Prof1.get_distribution
        kwargs = [{},
                  {'bw_method': 2},
                  {'output_format': 'ponderated'},
                  {'output_format': 'concentration'}]
        parametric_test(fun, kwargs)

    def test_get_extrema_position(self):
        fun = self.Prof_evenlyspaced.get_extrema_position
        kwargs = [{},
                  {'smoothing': 4}]
        parametric_test(fun, kwargs)

    def test_get_convolution(self):
        res_a = self.Prof_evenlyspaced.get_convolution(self.Prof2_evenlyspaced)
        # imtio.export_to_file(res_a, "Prof1_get_convolution_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_get_convolution_a.cimt")
        assert res_a == res_a2

    def test_get_dephasage(self):
        res_a = self.Prof1.get_dephasage(self.Prof2)
        # imtio.export_to_file(res_a, "Prof1_get_dephasage_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_get_dephasage_a.cimt")
        assert res_a == res_a2

    def test_get_convolution_of_difference(self):
        res_a = self.Prof1.get_convolution_of_difference(self.Prof2)
        # imtio.export_to_file(res_a, "Prof1_get_convolution_of_difference_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_get_convolution_of_difference_a.cimt")
        assert res_a == res_a2

    def test_spectral_filtering(self):
        res_a = self.Prof1.spectral_filtering(fmin=.2, fmax=.3)
        # imtio.export_to_file(res_a, "Prof1_spectral_filtering_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_spectral_filtering_a.cimt")
        assert res_a == res_a2

    def test_add_point(self):
        res_a = self.Prof1.add_point(3, 8)
        # imtio.export_to_file(res_a, "Prof1_add_point_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_add_point_a.cimt")
        assert res_a == res_a2

    def test_add_points(self):
        res_a = self.Prof1.add_points(self.Prof2)
        # imtio.export_to_file(res_a, "Prof1_add_points_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_add_points_a.cimt")
        assert res_a == res_a2

    def test_remove_point(self):
        res_a = self.Prof1.remove_point(34)
        # imtio.export_to_file(res_a, "Prof1_remove_point_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_remove_point_a.cimt")
        assert res_a == res_a2

    def test_crop_masked_border(self):
        self.Prof1.mask[0:10] = True
        self.Prof1.mask[-7::] = True
        res_a = self.Prof1.crop_masked_border()
        # imtio.export_to_file(res_a, "Prof1_crop_masked_border_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_crop_masked_border_a.cimt")
        assert res_a == res_a2

    def test_crop(self):
        res_a = self.Prof1.crop(intervx=[9, 15])
        # imtio.export_to_file(res_a, "Prof1_crop_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_crop_a.cimt")
        assert res_a == res_a2
        #
        res_b = self.Prof1.crop(intervy=[.5, .7])
        # imtio.export_to_file(res_b, "Prof1_crop_b.cimt")
        res_b2 = imtio.import_from_file("Prof1_crop_b.cimt")
        assert res_b == res_b2

    def test_scale(self):
        res_a = self.Prof1.scale(scalex=1.5, scaley=.89, inplace=False)
        # imtio.export_to_file(res_a, "Prof1_scale_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_scale_a.cimt")
        assert res_a == res_a2

    def test_fill(self):
        res_a = self.Prof1.fill()
        # imtio.export_to_file(res_a, "Prof1_fill_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_fill_a.cimt")
        assert res_a == res_a2

    def test_augment_resolution(self):
        res_a = self.Prof1.augment_resolution(2)
        # imtio.export_to_file(res_a, "Prof1_augment_resolution_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_augment_resolution_a.cimt")
        assert res_a == res_a2

    def test_change_unit(self):
        res_a = self.Prof1.change_unit('x', 'km')
        # imtio.export_to_file(res_a, "Prof1_change_unit_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_change_unit_a.cimt")
        assert res_a == res_a2
        #
        res_b = self.Prof1.change_unit('y', 'um/ks')
        # imtio.export_to_file(res_b, "Prof1_change_unit_b.cimt")
        res_b2 = imtio.import_from_file("Prof1_change_unit_b.cimt")
        assert res_b == res_b2

    def test_evenly_space(self):
        res_a = self.Prof1.evenly_space()
        # imtio.export_to_file(res_a, "Prof1_evenly_space_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_evenly_space_a.cimt")
        assert res_a == res_a2

    def test_smooth(self):
        res_a = self.Prof1.smooth(tos='gaussian', size=4)
        # imtio.export_to_file(res_a, "Prof1_smooth_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_smooth_a.cimt")
        assert res_a == res_a2

    def test_remove_doublons(self):
        self.Prof1.x[0:50] = self.Prof1.x[50:100]
        res_a = self.Prof1.remove_doublons()
        # imtio.export_to_file(res_a, "Prof1_remove_doublons_a.cimt")
        res_a2 = imtio.import_from_file("Prof1_remove_doublons_a.cimt")
        assert res_a == res_a2

    @pytest.mark.mpl_image_compare
    def test_display_Prof_a(self):
        fig = plt.figure()
        self.Prof1.display()
        return fig

    @pytest.mark.mpl_image_compare
    def test_display_Prof_b(self):
        fig = plt.figure()
        self.Prof1.display(kind='semilogx')
        return fig

    @pytest.mark.mpl_image_compare
    def test_display_Prof_c(self):
        fig = plt.figure()
        self.Prof1.display(kind='semilogy')
        return fig

    @pytest.mark.mpl_image_compare
    def test_display_Prof_d(self):
        fig = plt.figure()
        self.Prof1.display(kind='loglog')
        return fig

    @pytest.mark.mpl_image_compare
    def test_display_Prof_e(self):
        fig = plt.figure()
        self.Prof1.display(reverse=True)
        return fig

    @pytest.mark.mpl_image_compare
    def test_display_Prof_f(self):
        fig = plt.figure()
        self.Prof1.display('plot', marker='o', lw=5, ls=':')
        return fig

# TEMP
pytest.main(['test_profile.py'])
# test = TestProfile()
# test.setup()
# test.test_get_interpolated_values()
# TEMP - End
