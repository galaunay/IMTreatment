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

import unittest
from IMTreatment import make_unit, ScalarField
import numpy as np


### FIELD TEST ###
class FieldTest(unittest.TestCase):
    pass


### SCALARFIELD TEST ###
class SFTest(unittest.TestCase):

    def setUp(self):
        unit_x = make_unit('m')
        unit_y = make_unit('km')
        unit_values = make_unit('m/s')
        values = np.genfromtxt("values")
        axe_x = np.genfromtxt("axe_x")
        axe_y = np.genfromtxt("axe_y")
        mask = np.genfromtxt("mask")
        values2 = np.genfromtxt("values2")
        mask2 = np.genfromtxt("mask2")
        self.SF1 = ScalarField()
        self.SF1.import_from_arrays(axe_x, axe_y, values,
                                    mask=mask,
                                    unit_x=unit_x, unit_y=unit_y,
                                    unit_values=unit_values)
        self.SF2 = ScalarField()
        self.SF2.import_from_arrays(axe_x, axe_y, values2,
                                    mask=mask2,
                                    unit_x=unit_x, unit_y=unit_y,
                                    unit_values=unit_values)

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
        self.assertEqual(np.all(sf.axe_x == axe_x), True)
        self.assertEqual(np.all(sf.axe_y == axe_y), True)
        self.assertEqual(np.all(sf.values[~sf.mask] == values[~mask]), True)
        self.assertEqual(np.all(sf.mask == mask), True)
        self.assertEqual(sf.unit_x, unit_x)
        self.assertEqual(sf.unit_y, unit_y)
        self.assertEqual(sf.unit_values, unit_values)

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
        self.assertEqual(np.all(sf.axe_x == axe_x), True)
        self.assertEqual(np.all(sf.axe_y == axe_y), True)
        self.assertEqual(np.all(sf.values[~sf.mask] == -values[~mask]), True)
        self.assertEqual(np.all(sf.mask == mask), True)
        self.assertEqual(sf.unit_x, unit_x)
        self.assertEqual(sf.unit_y, unit_y)
        self.assertEqual(sf.unit_values, unit_values)
        # add
        nmb = 5
        unt = 500*make_unit('mm/s')
        values_f = (nmb + values + unt.asNumber()/1000. + values2 +
                    unt.asNumber()/1000. + values + nmb)
        mask_f = np.logical_or(mask, mask2)
        sf = nmb + self.SF1 + unt + self.SF2 + unt + self.SF1 + nmb
        self.assertEqual(np.all(sf.axe_x == axe_x), True)
        self.assertEqual(np.all(sf.axe_y == axe_y), True)
        self.assertEqual(np.all(sf.values[~mask_f] ==
                                values_f[~mask_f]), True)
        self.assertEqual(np.all(sf.mask == mask_f), True)
        self.assertEqual(sf.unit_x, unit_x)
        self.assertEqual(sf.unit_y, unit_y)
        self.assertEqual(sf.unit_values, unit_values)
        # sub
        nmb = 5
        unt = 500*make_unit('mm/s')
        values_f = (nmb - values - unt.asNumber()/1000. - values2 -
                    unt.asNumber()/1000. - values - nmb)
        mask_f = np.logical_or(mask, mask2)
        sf = nmb - self.SF1 - unt - self.SF2 - unt - self.SF1 - nmb
        self.assertEqual(np.all(sf.axe_x == axe_x), True)
        self.assertEqual(np.all(sf.axe_y == axe_y), True)
        self.assertEqual(np.all(sf.values[~mask_f] ==
                                values_f[~mask_f]), True)
        self.assertEqual(np.all(sf.mask == mask_f), True)
        self.assertEqual(sf.unit_x, unit_x)
        self.assertEqual(sf.unit_y, unit_y)
        self.assertEqual(sf.unit_values, unit_values)
        # mul
        nmb = 5.23
        unt = 500.*make_unit('mm/s')
        unt_n = 500./1000.
        values_f = (nmb * values * unt_n * values2 *
                    unt_n * values * nmb)
        unit_values = make_unit('mm/s')**2*make_unit('m/s')**3*1e6
        mask_f = np.logical_or(mask, mask2)
        sf = nmb * self.SF1 * unt * self.SF2 * unt * self.SF1 * nmb
        self.assertEqual(np.all(sf.axe_x == axe_x), True)
        self.assertEqual(np.all(sf.axe_y == axe_y), True)
        self.assertAlmostEqual(
            np.all(sf.values[~mask_f] - values_f[~mask_f] < 1e-6),
            True)
        self.assertAlmostEqual(np.all(sf.mask == mask_f), True)
        self.assertEqual(sf.unit_x, unit_x)
        self.assertEqual(sf.unit_y, unit_y)
        self.assertEqual(sf.unit_values, unit_values)
        # div
        nmb = 5.23
        unt = 500.*make_unit('mm/s')
        unt_n = 500./1000.
        values_f = (nmb / values / unt_n / values2 /
                    unt_n / values / nmb)
        unit_values = 1./(make_unit('mm/s')**2*make_unit('m/s')**3*1e6)
        mask_f = np.logical_or(mask, mask2)
        sf = nmb / self.SF1 / unt / self.SF2 / unt / self.SF1 / nmb
        self.assertEqual(np.all(sf.axe_x == axe_x), True)
        self.assertEqual(np.all(sf.axe_y == axe_y), True)
        self.assertAlmostEqual(
            np.all(sf.values[~mask_f] - values_f[~mask_f] < 1e-6),
            True)
        self.assertAlmostEqual(np.all(sf.mask == mask_f), True)
        self.assertEqual(sf.unit_x, unit_x)
        self.assertEqual(sf.unit_y, unit_y)
        self.assertEqual(sf.unit_values, unit_values)
        # abs
        unit_values = self.SF1.unit_values
        sf = np.abs(self.SF1)
        self.assertEqual(np.all(sf.axe_x == axe_x), True)
        self.assertEqual(np.all(sf.axe_y == axe_y), True)
        self.assertEqual(np.all(sf.values[~sf.mask] == np.abs(values[~mask])),
                         True)
        self.assertEqual(np.all(sf.mask == mask), True)
        self.assertEqual(sf.unit_x, unit_x)
        self.assertEqual(sf.unit_y, unit_y)
        self.assertEqual(sf.unit_values, unit_values)
        # pow
        unit_values = self.SF1.unit_values**3.544186
        sf = (np.abs(self.SF1) + 1)**3.544186
        self.assertEqual(np.all(sf.axe_x == axe_x), True)
        self.assertEqual(np.all(sf.axe_y == axe_y), True)
        self.assertEqual(np.all(sf.values[~sf.mask] -
                                (np.abs(values[~mask]) + 1)**3.544186 <
                                1e-6),
                         True)
        self.assertEqual(np.all(sf.mask == mask), True)
        self.assertEqual(sf.unit_x, unit_x)
        self.assertEqual(sf.unit_y, unit_y)
        self.assertEqual(sf.unit_values, unit_values)

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
        self.assertEqual(np.all(ind_x == ind2_x), True)

    def test_trim_area(self):
        axe_x, axe_y = self.SF1.axe_x, self.SF1.axe_y
        values = self.SF1.values
        mask = self.SF1.mask
        sf = self.SF1.crop([axe_x[3], axe_x[-4]], [axe_y[2], axe_y[-7]])
        self.assertEqual(np.all(sf.axe_x == axe_x[3:-3]), True)
        self.assertEqual(np.all(sf.axe_y == axe_y[2:-6]), True)
        self.assertEqual(np.all(sf.values[~sf.mask] ==
                                values[3:-3, 2:-6][~mask[3:-3, 2:-6]]), True)

    def test_crop_border(self):
        tmpsf = self.SF1.copy()
        mask = tmpsf.mask
        mask[0:3, :] = True
        mask[-1:, :] = True
        mask[:, 0:5] = True
        mask[:, -4:] = True
        tmpsf.mask = mask
        tmpsf.crop_masked_border(inplace=True)
        values_f = self.SF1.values[3:-1, 5:-4]
        mask_f = self.SF1.mask[3:-1, 5:-4]
        self.assertTrue(np.all(mask_f == tmpsf.mask))
        self.assertTrue(np.all(values_f[~mask_f] ==
                               tmpsf.values[~tmpsf.mask]))

    def test_min_max_mean(self):
        mini = self.SF1.min
        maxi = self.SF1.max
        mean = self.SF1.mean
        self.assertEqual(mini, -80.685138559926955)
        self.assertEqual(maxi, -0.11918431073522018)
        self.assertEqual(mean, -38.416196172428371)

    def test_get_values(self):
        value = self.SF1.get_value(7.5, 20.12)
        value2 = self.SF1.get_value(5, 10, ind=True)
        self.assertEqual(value, -39.27599375418945)
        self.assertEqual(value2, -77.512615587040656)

    def test_fill(self):
        self.SF1.fill(inplace=True)
        value_filled = np.genfromtxt("values_filled")
        self.assertFalse(np.any(self.SF1.mask))
        self.assertTrue(np.all(value_filled == self.SF1.values))

    def test_get_zones_centers(self):
        self.SF1.fill(inplace=True)
        zones_xy = np.genfromtxt("value_zones_center")
        zones = self.SF1.get_zones_centers(bornes=[0.75, 1])
        self.assertTrue(np.all(zones_xy == zones.xy))

    def test_get_nearest_extrema(self):
        self.SF1.fill(inplace=True)
        center = self.SF1.get_nearest_extrema([7.5, 20.12])
        self.assertAlmostEqual(6.3020200717715031, center[0][0])
        self.assertAlmostEqual(22.1880168513963919, center[0][1])

    def test_get_profile(self):
        profile = self.SF1.get_profile(1, 9.98)
        prof_x, prof_y = np.genfromtxt("profile")
        self.assertTrue(np.all(prof_x[~profile.mask] ==
                               profile.x[~profile.mask]))
        self.assertTrue(np.all(prof_y[~profile.mask] ==
                               profile.y[~profile.mask]))

unittest.main()

if __name__ == '__main__':
    unittest.main()
