# -*- coding: utf-8 -*-
"""
Created on Fri May  9 01:22:13 2014

@author: muahah
"""
import unittest
from IMTreatment import *
import numpy as np
import pdb


### FIELD TEST ###
class FieldTest(unittest.TestCase):
    pass


### SCALARFIELD TEST ###
class SFTest(unittest.TestCase):

    def setUp(self):
        axe_x = np.arange(23)
        unit_x = make_unit('m')
        unit_y = make_unit('km')
        unit_values = make_unit('m/s')
        axe_y = np.arange(47)*.01
        values = np.random.rand(len(axe_y), len(axe_x))
        values = values*np.random.randint(-100, 100)
        mask = [np.random.rand(len(axe_y), len(axe_x)) < 0.1]
        values = np.ma.masked_array(values, mask)
        values2 = np.random.rand(len(axe_y), len(axe_x))
        values2 = values2*np.random.randint(-100, 100)
        mask2 = [np.random.rand(len(axe_y), len(axe_x)) < 0.1]
        values2 = np.ma.masked_array(values2, mask2)
        self.SF1 = ScalarField()
        self.SF1.import_from_arrays(axe_x, axe_y, values,
                                    unit_x, unit_y, unit_values)
        self.SF2 = ScalarField()
        self.SF2.import_from_arrays(axe_x, axe_y, values2,
                                    unit_x, unit_y, unit_values)

    def test_import_from_arrays(self):
        # creating a SF field using 'import_from_arrays
        axe_x = np.arange(10)
        unit_x = make_unit('m')
        unit_y = make_unit('km')
        unit_values = make_unit('m/s')
        axe_y = np.arange(20)*.01
        values = np.arange(len(axe_x)*len(axe_y)).reshape(len(axe_y),
                                                          len(axe_x))
        mask = [values % 6 == 0]
        values = np.pi*values
        values = np.ma.masked_array(values, mask)
        sf = ScalarField()
        sf.import_from_arrays(axe_x, axe_y, values, unit_x, unit_y,
                              unit_values)
        # tests
        self.assertEqual(np.all(sf.get_axes()[0] == axe_x), True)
        self.assertEqual(np.all(sf.get_axes()[1] == axe_y), True)
        self.assertEqual(np.all(sf.get_values().data == values), True)
        self.assertEqual(np.all(sf.get_values().mask == mask), True)
        self.assertEqual(sf.get_axe_units()[0], unit_x)
        self.assertEqual(sf.get_axe_units()[1], unit_y)
        self.assertEqual(sf.get_values_unit(), unit_values)

    def test_operations(self):
        # get datas
        axe_x, axe_y = self.SF1.get_axes()
        values = self.SF1.get_values().data
        mask = self.SF1.get_values().mask
        values2 = self.SF2.get_values().data
        mask2 = self.SF2.get_values().mask
        unit_x, unit_y = self.SF1.get_axe_units()
        unit_values = self.SF1.get_values_unit()
        # neg
        sf = -self.SF1
        self.assertEqual(np.all(sf.get_axes()[0] == axe_x), True)
        self.assertEqual(np.all(sf.get_axes()[1] == axe_y), True)
        self.assertEqual(np.all(sf.get_values().data == -values), True)
        self.assertEqual(np.all(sf.get_values().mask == mask), True)
        self.assertEqual(sf.get_axe_units()[0], unit_x)
        self.assertEqual(sf.get_axe_units()[1], unit_y)
        self.assertEqual(sf.get_values_unit(), unit_values)
        # add
        nmb = 5
        unt = 500*make_unit('mm/s')
        values_f = (nmb + values + unt.asNumber()/1000. + values2
            + unt.asNumber()/1000. + values + nmb)
        mask_f = np.logical_or(mask, mask2)
        sf = nmb + self.SF1 + unt + self.SF2 + unt + self.SF1 + nmb
        self.assertEqual(np.all(sf.get_axes()[0] == axe_x), True)
        self.assertEqual(np.all(sf.get_axes()[1] == axe_y), True)
        self.assertEqual(np.all(sf.get_values().data[~mask_f]
                                == values_f[~mask_f]), True)
        self.assertEqual(np.all(sf.get_values().mask == mask_f), True)
        self.assertEqual(sf.get_axe_units()[0], unit_x)
        self.assertEqual(sf.get_axe_units()[1], unit_y)
        self.assertEqual(sf.get_values_unit(), unit_values)
        # sub
        nmb = 5
        unt = 500*make_unit('mm/s')
        values_f = (nmb - values - unt.asNumber()/1000. - values2
            - unt.asNumber()/1000. - values - nmb)
        mask_f = np.logical_or(mask, mask2)
        sf = nmb - self.SF1 - unt - self.SF2 - unt - self.SF1 - nmb
        self.assertEqual(np.all(sf.get_axes()[0] == axe_x), True)
        self.assertEqual(np.all(sf.get_axes()[1] == axe_y), True)
        self.assertEqual(np.all(sf.get_values().data[~mask_f]
                                == values_f[~mask_f]), True)
        self.assertEqual(np.all(sf.get_values().mask == mask_f), True)
        self.assertEqual(sf.get_axe_units()[0], unit_x)
        self.assertEqual(sf.get_axe_units()[1], unit_y)
        self.assertEqual(sf.get_values_unit(), unit_values)
        # mul
        nmb = 5.23
        unt = 500.*make_unit('mm/s')
        unt_n = 500./1000.
        values_f = (nmb * values * unt_n * values2
            * unt_n * values * nmb)
        unit_values = make_unit('mm/s')**2*make_unit('m/s')**3*1e6
        mask_f = np.logical_or(mask, mask2)
        sf = nmb * self.SF1 * unt * self.SF2 * unt * self.SF1 * nmb
        self.assertEqual(np.all(sf.get_axes()[0] == axe_x), True)
        self.assertEqual(np.all(sf.get_axes()[1] == axe_y), True)
        self.assertAlmostEqual(
            np.all(sf.get_values().data[~mask_f] - values_f[~mask_f] < 1e-6),
            True)
        self.assertAlmostEqual(np.all(sf.get_values().mask == mask_f), True)
        self.assertEqual(sf.get_axe_units()[0], unit_x)
        self.assertEqual(sf.get_axe_units()[1], unit_y)
        self.assertEqual(sf.get_values_unit(), unit_values)
        # div
        nmb = 5.23
        unt = 500.*make_unit('mm/s')
        unt_n = 500./1000.
        values_f = (nmb / values / unt_n / values2
            / unt_n / values / nmb)
        unit_values = 1./(make_unit('mm/s')**2*make_unit('m/s')**3*1e6)
        mask_f = np.logical_or(mask, mask2)
        sf = nmb / self.SF1 / unt / self.SF2 / unt / self.SF1 / nmb
        self.assertEqual(np.all(sf.get_axes()[0] == axe_x), True)
        self.assertEqual(np.all(sf.get_axes()[1] == axe_y), True)
        self.assertAlmostEqual(
            np.all(sf.get_values().data[~mask_f] - values_f[~mask_f] < 1e-6),
            True)
        self.assertAlmostEqual(np.all(sf.get_values().mask == mask_f), True)
        self.assertEqual(sf.get_axe_units()[0], unit_x)
        self.assertEqual(sf.get_axe_units()[1], unit_y)
        self.assertEqual(sf.get_values_unit(), unit_values)
        # abs
        unit_values = self.SF1.get_values_unit()
        sf = np.abs(self.SF1)
        self.assertEqual(np.all(sf.get_axes()[0] == axe_x), True)
        self.assertEqual(np.all(sf.get_axes()[1] == axe_y), True)
        self.assertEqual(np.all(sf.get_values().data == np.abs(values)), True)
        self.assertEqual(np.all(sf.get_values().mask == mask), True)
        self.assertEqual(sf.get_axe_units()[0], unit_x)
        self.assertEqual(sf.get_axe_units()[1], unit_y)
        self.assertEqual(sf.get_values_unit(), unit_values)
        # pow
        unit_values = self.SF1.get_values_unit()**3.544186
        sf = self.SF1**3.544186
        self.assertEqual(np.all(sf.get_axes()[0] == axe_x), True)
        self.assertEqual(np.all(sf.get_axes()[1] == axe_y), True)
        self.assertEqual(np.all(sf.get_values().data == np.power(values,
                                                                 3.544186)),
                                True)
        self.assertEqual(np.all(sf.get_values().mask == mask), True)
        self.assertEqual(sf.get_axe_units()[0], unit_x)
        self.assertEqual(sf.get_axe_units()[1], unit_y)
        self.assertEqual(sf.get_values_unit(), unit_values)

    def test_iter(self):
        axe_x, axe_y = self.SF1.get_axes()
        axe_x, axe_y = np.meshgrid(axe_x, axe_y)
        ind_x = np.arange(len(axe_x))
        ind_y = np.arange(len(axe_y))
        ind_x, ind_y = np.meshgrid(ind_x, ind_y)
        axe_x = np.transpose(axe_x)
        axe_y = np.transpose(axe_y)
        ind_x = np.transpose(ind_x)
        ind_y = np.transpose(ind_y)
        values = np.transpose(self.SF1.get_values().data)
        mask = np.transpose(self.SF1.get_values().mask)
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
        axe_x, axe_y = self.SF1.get_axes()
        values = self.SF1.get_values()
        sf = self.SF1.trim_area([axe_x[3], axe_x[-4]], [axe_y[2], axe_y[-7]])
        self.assertEqual(np.all(sf.get_axes()[0] == axe_x[3:-3]), True)
        self.assertEqual(np.all(sf.get_axes()[1] == axe_y[2:-6]), True)
        self.assertEqual(np.all(sf.get_values() == values[2:-6, 3:-3]), True)

    def test_crop_border(self):
        pass


if __name__ == '__main__':
    unittest.main()
