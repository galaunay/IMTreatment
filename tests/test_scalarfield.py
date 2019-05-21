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
import unum
from IMTreatment import ScalarField, file_operation as imtio, make_unit


class TestScalarField(object):
    """ Done """

    def setup(self):
        sane_parameters()
        self.SF1 = imtio.import_from_file("SF1.cimt")
        self.SF1_nomask = imtio.import_from_file("SF1_nomask.cimt")
        self.SF2 = imtio.import_from_file("SF2.cimt")
        self.SF_notevenlyspaced = imtio.import_from_file("SF_notevenlyspaced.cimt")

    def test_import_from_arrays(self):
        # creating a SF field using 'import_from_arrays
        axe_x = np.arange(10)
        unit_x = make_unit('m')
        unit_y = make_unit('km')
        unit_values = make_unit('m/s')
        axe_y = np.arange(20)*.01
        values = np.arange(len(axe_x)*len(axe_y)).reshape(len(axe_y),
                                                          len(axe_x))
        values = np.array(values, dtype=float)
        mask = np.random.rand(len(axe_y), len(axe_x)) > 0.75
        values[mask] = np.nan
        values = np.pi*values
        sf = ScalarField()
        sf.import_from_arrays(axe_x, axe_y, values, mask=mask,
                              unit_x=unit_x, unit_y=unit_y,
                              unit_values=unit_values)
        values = values.transpose()
        mask = mask.transpose()
        # tests
        assert np.all(sf.axe_x == axe_x)
        assert np.all(sf.axe_y == axe_y)
        assert np.all(sf.values[~sf.mask] == values[~mask])
        assert np.all(sf.mask == mask)
        assert sf.unit_x == unit_x
        assert sf.unit_y == unit_y
        assert sf.unit_values == unit_values

    def test_operations(self):
        # get datas
        axe_x, axe_y = self.SF1.axe_x, self.SF1.axe_y
        values = self.SF1.values
        mask = self.SF1.mask
        values2 = self.SF2.values
        mask2 = self.SF2.mask
        unit_x, unit_y = self.SF1.unit_x, self.SF1.unit_y
        unit_values = self.SF1.unit_values
        # neg
        sf = -self.SF1
        assert np.all(sf.axe_x == axe_x)
        assert np.all(sf.axe_y == axe_y)
        assert np.all(sf.values[~sf.mask] == -values[~mask])
        assert np.all(sf.mask == mask)
        assert sf.unit_x == unit_x
        assert sf.unit_y == unit_y
        assert sf.unit_values == unit_values
        # add
        nmb = 5
        unt = 500*make_unit('mm/s')
        values_f = (nmb + values + unt.asNumber()/1000. + values2 +
                    unt.asNumber()/1000. + values + nmb)
        mask_f = np.logical_or(mask, mask2)
        sf = nmb + self.SF1 + unt + self.SF2 + unt + self.SF1 + nmb
        assert np.all(sf.axe_x == axe_x)
        assert np.all(sf.axe_y == axe_y)
        assert np.all(sf.values[~mask_f] == values_f[~mask_f])
        assert np.all(sf.mask == mask_f)
        assert sf.unit_x == unit_x
        assert sf.unit_y == unit_y
        assert sf.unit_values == unit_values
        # sub
        nmb = 5
        unt = 500*make_unit('mm/s')
        values_f = (nmb - values - unt.asNumber()/1000. - values2 -
                    unt.asNumber()/1000. - values - nmb)
        mask_f = np.logical_or(mask, mask2)
        sf = nmb - self.SF1 - unt - self.SF2 - unt - self.SF1 - nmb
        assert np.all(sf.axe_x == axe_x)
        assert np.all(sf.axe_y == axe_y)
        assert np.all(sf.values[~mask_f] == values_f[~mask_f])
        assert np.all(sf.mask == mask_f)
        assert sf.unit_x == unit_x
        assert sf.unit_y == unit_y
        assert sf.unit_values == unit_values
        # mul
        nmb = 5.23
        unt = 500.*make_unit('mm/s')
        unt_n = 500./1000.
        values_f = (nmb * values * unt_n * values2 *
                    unt_n * values * nmb)
        unit_values = make_unit('mm/s')**2*make_unit('m/s')**3*1e6
        mask_f = np.logical_or(mask, mask2)
        sf = nmb * self.SF1 * unt * self.SF2 * unt * self.SF1 * nmb
        assert np.all(sf.axe_x == axe_x)
        assert np.all(sf.axe_y == axe_y)
        assert np.all(sf.values[~mask_f] - values_f[~mask_f] < 1e-6)
        assert np.all(sf.mask == mask_f)
        assert sf.unit_x == unit_x
        assert sf.unit_y == unit_y
        assert sf.unit_values == unit_values
        # div
        nmb = 5.23
        unt = 500.*make_unit('mm/s')
        unt_n = 500./1000.
        values_f = (nmb / values / unt_n / values2 /
                    unt_n / values / nmb)
        unit_values = 1./(make_unit('mm/s')**2*make_unit('m/s')**3*1e6)
        mask_f = np.logical_or(mask, mask2)
        sf = nmb / self.SF1 / unt / self.SF2 / unt / self.SF1 / nmb
        assert np.all(sf.axe_x == axe_x)
        assert np.all(sf.axe_y == axe_y)
        assert np.all(sf.values[~mask_f] - values_f[~mask_f] < 1e-6)
        assert np.all(sf.mask == mask_f)
        assert sf.unit_x == unit_x
        assert sf.unit_y == unit_y
        assert sf.unit_values == unit_values
        # abs
        unit_values = self.SF1.unit_values
        sf = np.abs(self.SF1)
        assert np.all(sf.axe_x == axe_x)
        assert np.all(sf.axe_y == axe_y)
        assert np.all(sf.values[~sf.mask] == np.abs(values[~mask]))
        assert np.all(sf.mask == mask)
        assert sf.unit_x == unit_x
        assert sf.unit_y == unit_y
        assert sf.unit_values == unit_values
        # pow
        unit_values = self.SF1.unit_values**3.544186
        sf = (np.abs(self.SF1) + 1)**3.544186
        assert np.all(sf.axe_x == axe_x)
        assert np.all(sf.axe_y == axe_y)
        assert np.all(sf.values[~sf.mask] -
                      (np.abs(values[~mask]) + 1)**3.544186 <
                      1e-6)
        assert np.all(sf.mask == mask)
        assert sf.unit_x == unit_x
        assert sf.unit_y == unit_y
        assert sf.unit_values == unit_values

    def test_iter(self):
        axe_x, axe_y = self.SF1.axe_x, self.SF1.axe_y
        ind_x = np.arange(len(axe_x))
        ind_y = np.arange(len(axe_y))
        axe_x, axe_y = np.meshgrid(axe_x, axe_y)
        ind_x, ind_y = np.meshgrid(ind_x, ind_y)
        axe_x = np.transpose(axe_x)
        axe_y = np.transpose(axe_y)
        ind_x = np.transpose(ind_x)
        ind_y = np.transpose(ind_y)
        values = self.SF1.values
        mask = self.SF1.mask
        ind_x = ind_x[~mask]
        ind_y = ind_y[~mask]
        axe_x = axe_x[~mask]
        axe_y = axe_y[~mask]
        values = values[~mask]
        ind2_x = []
        ind2_y = []
        axe2_x = []
        axe2_y = []
        values2 = []
        for ij, xy, val in self.SF1:
            ind2_x.append(ij[0])
            ind2_y.append(ij[1])
            axe2_x.append(xy[0])
            axe2_y.append(xy[1])
            values2.append(val)
        assert np.all(ind_x == ind2_x)

    def test_trim_area(self):
        axe_x, axe_y = self.SF1.axe_x, self.SF1.axe_y
        values = self.SF1.values
        mask = self.SF1.mask
        sf = self.SF1.crop([axe_x[3], axe_x[-4]], [axe_y[2], axe_y[-7]])
        assert np.all(sf.axe_x == axe_x[3:-3])
        assert np.all(sf.axe_y == axe_y[2:-6])
        assert np.all(sf.values[~sf.mask] ==
                      values[3:-3, 2:-6][~mask[3:-3, 2:-6]])

    def test_min_max_mean(self):
        mini = self.SF1.min
        maxi = self.SF1.max
        mean = self.SF1.mean
        assert mini == -80.685138559926955
        assert maxi == -0.11918431073522018
        assert mean == -38.416196172428371

    def test_get_values(self):
        value = self.SF1.get_value(7.5, 20.12)
        value2 = self.SF1.get_value(5, 10, ind=True)
        assert value == -39.27599375418945
        assert value2 == -77.512615587040656

    def test_get_zones_centers(self):
        self.SF1.fill(inplace=True)
        zones_xy = np.genfromtxt("value_zones_center")
        zones = self.SF1.get_zones_centers(bornes=[0.75, 1])
        assert np.all(zones_xy == zones.xy)

    def test_get_nearest_extrema(self):
        self.SF1.fill(inplace=True)
        center = self.SF1.get_nearest_extrema([7.5, 20.12])
        assert (6.3020200717715031 - center[0][0]) < 1e-6
        assert (22.1880168513963919 - center[0][1]) < 1e-6

    def test_get_profile(self):
        profile = self.SF1.get_profile('x', 9.98)
        prof_x, prof_y = np.genfromtxt("profile")
        assert np.all(prof_x[~profile.mask] == profile.x[~profile.mask])
        assert np.all(prof_y[~profile.mask] == profile.y[~profile.mask])

    def test_get_spatial_autocorrelation(self):
        res_x = self.SF1.get_spatial_autocorrelation('x')
        # imtio.export_to_file(res_x, "spatial_autocorrelation_x.cimt")
        res_x2 = imtio.import_from_file("spatial_autocorrelation_x.cimt")
        assert res_x == res_x2
        res_y = self.SF1.get_spatial_autocorrelation('y')
        # imtio.export_to_file(res_y, "spatial_autocorrelation_y.cimt")
        res_y2 = imtio.import_from_file("spatial_autocorrelation_y.cimt")
        assert res_y == res_y2

    def test_get_spatial_spectrum(self):
        res_x = self.SF1.get_spatial_spectrum('x', intervx=[5, 19],
                                              intervy=[10.2, 29])
        # imtio.export_to_file(res_x, "get_spatial_spectrum_x.cimt")
        res_x2 = imtio.import_from_file("get_spatial_spectrum_x.cimt")
        assert res_x == res_x2
        res_y = self.SF1.get_spatial_spectrum('y', intervx=[5, 19],
                                              intervy=[10.2, 29])
        # imtio.export_to_file(res_y, "get_spatial_spectrum_y.cimt")
        res_y2 = imtio.import_from_file("get_spatial_spectrum_y.cimt")
        assert res_y == res_y2

    def test_get_norm(self):
        res = self.SF1.get_norm()
        # imtio.export_to_file(res, "get_norm.cimt")
        res2 = imtio.import_from_file("get_norm.cimt")
        assert res == res2

    def test_get_interpolator(self):
        res = self.SF1_nomask.get_interpolator()
        # imtio.export_to_file(res, "get_interpolator.cimt")
        res2 = imtio.import_from_file("get_interpolator.cimt")
        assert res(5, 9.2), res2(5 == 9.2)

    def test_integrate_over_line(self):
        res_x = self.SF1_nomask.integrate_over_line('x', [3.2, 17.8])
        # imtio.export_to_file(res_x, "integrate_over_line_x.cimt")
        res_x2 = imtio.import_from_file("integrate_over_line_x.cimt")
        assert res_x == res_x2
        res_y = self.SF1_nomask.integrate_over_line('y', [3.2, 17.8])
        # imtio.export_to_file(res_y, "integrate_over_line_y.cimt")
        res_y2 = imtio.import_from_file("integrate_over_line_y.cimt")
        assert res_y == res_y2

    def test_integrate_over_surface(self):
        res = self.SF1_nomask.integrate_over_surface(intervx=[2.4, 19.8],
                                                     intervy=[5.6, 8.3])
        # imtio.export_to_file(res, "integrate_over_surface.cimt")
        res2 = imtio.import_from_file("integrate_over_surface.cimt")
        assert res == res2

    def test_copy(self):
        res = self.SF1.copy()
        # imtio.export_to_file(res, "copy.cimt")
        res2 = imtio.import_from_file("copy.cimt")
        assert res == res2

    def test_export_to_scatter(self):
        res = self.SF1.export_to_scatter()
        # imtio.export_to_file(res, "export_to_scatter.cimt")
        res2 = imtio.import_from_file("export_to_scatter.cimt")
        assert res == res2

    def test_scale(self):
        res = self.SF1.scale(scalex=1.45, scaley=.98, scalev=3.4,
                             inplace=False)
        # imtio.export_to_file(res, "scale.cimt")
        res2 = imtio.import_from_file("scale.cimt")
        assert res == res2

    def test_rotate(self):
        res = self.SF1.rotate(270, inplace=False)
        # imtio.export_to_file(res, "rotate.cimt")
        res2 = imtio.import_from_file("rotate.cimt")
        assert res == res2

    def test_change_unit(self):
        res = self.SF1.copy()
        res.change_unit('x', 'um')
        res.change_unit('y', 'm')
        # imtio.export_to_file(res, "change_unit.cimt")
        res2 = imtio.import_from_file("change_unit.cimt")
        assert res == res2
        with pytest.raises(unum.IncompatibleUnitsError):
            res.change_unit('x', 'm/s')

    def test_crop(self):
        res = self.SF1.crop(intervx=[7.4, 19.0],
                            intervy=[2.1, 8], inplace=False)
        # imtio.export_to_file(res, "crop.cimt")
        res2 = imtio.import_from_file("crop.cimt")
        assert res == res2

    def test_extend(self):
        res = self.SF1.extend(nmb_left=1,
                              nmb_right=5,
                              nmb_down=7,
                              nmb_up=4,
                              inplace=False)
        # imtio.export_to_file(res, "extend.cimt")
        res2 = imtio.import_from_file("extend.cimt")
        assert res == res2

    def test_mirroring(self):
        res = self.SF1.mirroring('x', 35, mir_coef=1.5)
        # imtio.export_to_file(res, "mirroring.cimt")
        res2 = imtio.import_from_file("mirroring.cimt")
        assert res == res2
        #
        tmpsf = self.SF1.copy()
        tmpsf.mirroring('x', 35, mir_coef=1.5, inplace=True)
        res_b = tmpsf
        # imtio.export_to_file(res_b, "mirroring_b.cimt")
        res_b2 = imtio.import_from_file("mirroring_b.cimt")
        assert res_b == res_b2

    def test_crop_masked_border(self):
        tmpsf = self.SF1.copy()
        tmpsf.mask[0:3, :] = True
        tmpsf.mask[-1:, :] = True
        tmpsf.mask[:, 0:5] = True
        tmpsf.mask[:, -4:] = True
        res = tmpsf.crop_masked_border(inplace=False)
        # imtio.export_to_file(res, "crop_masked_border.cimt")
        res2 = imtio.import_from_file("crop_masked_border.cimt")
        assert res == res2
        #
        tmpsf = self.SF1_nomask.copy()
        tmpsf.mask[3:-2, :] = False
        tmpsf.mask[:, 5:-4] = False
        res = self.SF1.crop_masked_border(hard=True, inplace=False)
        # imtio.export_to_file(res, "crop_masked_border_2.cimt")
        res2 = imtio.import_from_file("crop_masked_border_2.cimt")
        assert res == res2

    def test_fill(self):
        res = self.SF1.fill(inplace=False)
        # imtio.export_to_file(res, "fill.cimt")
        res2 = imtio.import_from_file("fill.cimt")
        assert res == res2

    def test_smooth(self):
        res = self.SF1.smooth(tos='uniform', size=4, inplace=False)
        # imtio.export_to_file(res, "smooth.cimt")
        res2 = imtio.import_from_file("smooth.cimt")
        assert res == res2

    def test_make_evenly_spaced(self):
        res = self.SF_notevenlyspaced.make_evenly_spaced()
        # imtio.export_to_file(res, "make_evenly_spaced.cimt")
        res2 = imtio.import_from_file("make_evenly_spaced.cimt")
        assert res == res2

    def test_reduce_spatial_resolution(self):
        res = self.SF1.reduce_spatial_resolution(4, inplace=False)
        # imtio.export_to_file(res, "reduce_spatial_resolution.cimt")
        res2 = imtio.import_from_file("reduce_spatial_resolution.cimt")
        assert res == res2

    @pytest.mark.mpl_image_compare
    def test_display_SF_a(self):
        fig = plt.figure()
        self.SF1.display()
        return fig

    @pytest.mark.mpl_image_compare
    def test_display_SF_b(self):
        fig = plt.figure()
        self.SF1.display(kind='contour')
        return fig

    @pytest.mark.mpl_image_compare
    def test_display_SF_c(self):
        fig = plt.figure()
        self.SF1.display(kind='contourf')
        return fig

    @pytest.mark.mpl_image_compare
    def test_display_SF_d(self):
        fig = plt.figure()
        self.SF1.display(kind='contourf', levels=[-1, 0, 1])
        return fig

    @pytest.mark.mpl_image_compare
    def test_display_SF_e(self):
        fig = plt.figure()
        self.SF1.display('mask')
        return fig
