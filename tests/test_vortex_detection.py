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

import matplotlib.pyplot as plt
import unum

from IMTreatment import VectorField, file_operation as imtio, make_unit, \
    TemporalVectorFields
import IMTreatment.vortex_detection as vod
import IMTreatment.vortex_criterions as voc


class TestVortexDetection(object):

    def setup(self):
        sane_parameters()
        # VF1 = VectorField()
        # VF1.import_from_arrays(np.linspace(0, 24, 40),
        #                        np.linspace(-15.5, 15.5, 40),
        #                        (np.random.rand(40, 40) - .5)*12.3,
        #                        (np.random.rand(40, 40) - .5)*4.35,
        #                        mask=False,
        #                        unit_x='m',
        #                        unit_y='m',
        #                        unit_values='m/s')
        # VF1.smooth(tos='gaussian', size=2, inplace=True)
        # TVF1 = TemporalVectorFields()
        # for i in range(10):
        #     TVF1.add_field(VF1*np.cos(i/5*np.pi), i*0.834)
        # # imtio.export_to_file(VF1, 'VF1_nomask.cimt')
        # # imtio.export_to_file(TVF1, 'TVF1_nomask.cimt')
        self.VF1_nomask = imtio.import_from_file('VF1_nomask.cimt')
        self.TVF1_nomask = imtio.import_from_file('TVF1_nomask.cimt')

    def test_get_critical_points_a(self):
        res_a = vod.get_critical_points(self.VF1_nomask,
                                        mirroring=[['x', 0.]],
                                        thread='all')
        # imtio.export_to_file(res_a, "VF1_get_critical_points_a.cimt")
        res_a2 = imtio.import_from_file("VF1_get_critical_points_a.cimt")
        assert len(res_a2.foc) > 0
        assert res_a == res_a2

    def test_get_critical_points_b(self):
        res_b = vod.get_critical_points(self.TVF1_nomask)
        # imtio.export_to_file(res_b, "VF1_get_critical_points_b.cimt")
        res_b2 = imtio.import_from_file("VF1_get_critical_points_b.cimt")
        assert len(res_b2.foc) > 0
        assert res_b == res_b2

    def test_get_critical_points_c(self):
        res_c = vod.get_critical_points(self.TVF1_nomask, kind='gam_vort',
                                        thread='all')
        # imtio.export_to_file(res_c, "VF1_get_critical_points_c.cimt")
        res_c2 = imtio.import_from_file("VF1_get_critical_points_c.cimt")
        assert len(res_c2.foc) > 0
        assert res_c == res_c2

    def test_get_critical_points_d(self):
        res_d = vod.get_critical_points(self.VF1_nomask, kind='gam_vort',
                                        mirroring=[['x', 0.]])
        # imtio.export_to_file(res_d, "VF1_get_dritical_points_d.cimt")
        res_d2 = imtio.import_from_file("VF1_get_dritical_points_d.cimt")
        assert len(res_d2.foc) > 0
        assert res_d == res_d2

    def test_get_critical_points_e(self):
        res_e = vod.get_critical_points(self.TVF1_nomask, kind='pbi_cell')
        # imtio.export_to_file(res_e, "VF1_get_eritical_points_e.cimt")
        res_e2 = imtio.import_from_file("VF1_get_eritical_points_e.cimt")
        assert len(res_e2.foc) > 0
        assert res_e == res_e2

    def test_get_critical_points_f(self):
        res_f = vod.get_critical_points(self.VF1_nomask, kind='pbi_cell',
                                        mirroring=[['x', 0.]])
        # imtio.export_to_file(res_f, "VF1_get_fritical_points_f.cimt")
        res_f2 = imtio.import_from_file("VF1_get_fritical_points_f.cimt")
        assert len(res_f2.foc) > 0
        assert res_f == res_f2

    def test_get_critical_points_g(self):
        res_g = vod.get_critical_points(self.TVF1_nomask, kind='pbi_crit',
                                        thread='all')
        # imtio.export_to_file(res_g, "VF1_get_gritical_points_g.cimt")
        res_g2 = imtio.import_from_file("VF1_get_gritical_points_g.cimt")
        assert len(res_g2.foc) > 0
        assert res_g == res_g2

    def test_get_critical_points_h(self):
        res_h = vod.get_critical_points(self.VF1_nomask, kind='pbi_crit',
                                        smoothing_size=2)
        # imtio.export_to_file(res_h, "VF1_get_hritical_points_h.cimt")
        res_h2 = imtio.import_from_file("VF1_get_hritical_points_h.cimt")
        assert len(res_h2.foc) > 0
        assert res_h == res_h2

    def test_compute_traj(self):
        res_g = vod.get_critical_points(self.TVF1_nomask, kind='pbi_crit',
                                        thread='all')
        #
        res_g.compute_traj()
        # imtio.export_to_file(res_g, "VF1_compute_traj1_g.cimt")
        res_g2 = imtio.import_from_file("VF1_compute_traj1_g.cimt")
        assert res_g == res_g2
        #
        res_g.compute_traj(epsilon=4)
        # imtio.export_to_file(res_g, "VF1_compute_traj2_g.cimt")
        res_g2 = imtio.import_from_file("VF1_compute_traj2_g.cimt")
        assert res_g == res_g2
        #
        res_g.compute_traj(epsilon=4*make_unit('m'))
        # imtio.export_to_file(res_g, "VF1_compute_traj3_g.cimt")
        res_g2 = imtio.import_from_file("VF1_compute_traj3_g.cimt")
        assert res_g == res_g2
        #
        res_g.compute_traj(close_traj=True)
        # imtio.export_to_file(res_g, "VF1_compute_traj4_g.cimt")
        res_g2 = imtio.import_from_file("VF1_compute_traj4_g.cimt")
        assert res_g == res_g2

    def test_get_point_density(self):
        cps = vod.get_critical_points(self.TVF1_nomask, kind='pbi_crit',
                                      thread='all')
        #
        dens = cps.get_points_density('foc')
        # imtio.export_to_file(dens, "VF1_get_points_density_a.cimt")
        dens_2 = imtio.import_from_file("VF1_get_points_density_a.cimt")
        assert dens == dens_2

    def test_break_trajectories(self):
        cps = vod.get_critical_points(self.TVF1_nomask, kind='pbi_crit',
                                      thread='all')
        cps.compute_traj()
        cps.break_trajectories()
        # imtio.export_to_file(cps, "VF1_break_trajectories_a.cimt")
        cps_2 = imtio.import_from_file("VF1_break_trajectories_a.cimt")
        assert cps == cps_2

    def test_get_mean_trajectory(self):
        cps = vod.get_critical_points(self.TVF1_nomask, kind='pbi_crit',
                                      thread='all')
        cps.compute_traj()
        means = cps.get_mean_trajectory("foc", min_len=0, min_nmb_to_avg=0)
        # imtio.export_to_file(means, "VF1_get_mean_trajectory_a.cimt")
        means_2 = imtio.import_from_file("VF1_get_mean_trajectory_a.cimt")
        assert means == means_2

    def test_get_vortex_position(self):
        fun = vod.get_vortex_position
        kwargs = [{'obj': self.VF1_nomask,
                   'criterion': voc.get_residual_vorticity},
                  {'obj': self.VF1_nomask,
                   'criterion': voc.get_residual_vorticity,
                   'threshold': 0.2}]
        parametric_test(fun, kwargs)

    def test_mean_trajectories(self):
        pass

    def test_crit_points(self):
        pass

    def test_topo_points(self):
        pass

# TEMP
pytest.main(['test_vortex_detection.py'])
# pytest.main(['--pdb', 'test_vortex_detection.py'])
# test = TestVortexDetection()
# test.setup()
# test.test_get_vortex_position()
# TEMP - End
