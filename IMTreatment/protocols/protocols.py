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

import numpy as np
import psutil
import warnings
import matplotlib.pyplot as plt
from .. import file_operation as imtio
from .. import pod as imtpod
from .. import vortex_detection as imtvod
from os.path import join
import os


class POD_CP_protocol(object):
    def __init__(self, name, imtpath, respath, crop_x=[-np.inf, np.inf],
                 crop_y=[-np.inf, np.inf], crop_t=[-np.inf, np.inf],
                 hard_crop=True,
                 pod_coh=0.05, mirroring=[[2, 0], [1, 0]], eps_traj=15.,
                 red_fact=1., nmb_min_in_traj=1, det_fact=1,
                 thread='all', remove_weird=False):
        self.name = name
        self.imtpath = imtpath
        self.respath = respath
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.crop_t = crop_t
        self.hard_crop = hard_crop
        self.pod = None
        self.pod_coh = pod_coh
        self.mirroring = mirroring
        self.eps_traj = eps_traj
        if det_fact > 1:
            self.detailled = True
        else:
            self.detailled = False
        self.det_fact = det_fact
        self.red_fact = red_fact
        self.nmb_min_in_traj = nmb_min_in_traj
        self.len_data = None
        if thread == "all":
            self.thread = psutil.cpu_coun()
        else:
            self.thread = thread
        self.remove_weird = remove_weird

    def prepare_data(self):
        print("    Preparing data    ")
        # import data
        tvf = imtio.import_from_file(join(self.imtpath,
                                          "{}.cimt".format(self.name)))
        # crop
        if self.hard_crop:
            tvf.crop_masked_border(hard=True, inplace=True)
        else:
            tvf.crop_masked_border(inplace=True)
        tvf.crop(intervx=self.crop_x, intervy=self.crop_y, intervt=self.crop_t,
                 inplace=True)
        # adjust temporal resolution
        if self.red_fact < 1:
            tvf.reduce_temporal_resolution(int(np.round(1./self.red_fact)),
                                           mean=False, inplace=True)
        # check for weird fields
        if self.remove_weird:
            tvf.remove_weird_fields(std_coef=3., treatment='interpolate',
                                    inplace=True)
        # save a display
        mean_field = tvf.get_mean_field()
        mean_field.crop_masked_border(hard=False, inplace=True)
        plt.figure()
        mean_field.display(kind='stream', density=3)
        plt.savefig(join(self.respath, "{}/mean_field.png".format(self.name)))
        plt.close(plt.gcf())
        # store
        imtio.export_to_file(tvf,
                             join(self.imtpath,
                                  "{}_cln.cimt".format(self.name)))
        imtio.export_to_file(mean_field,
                             join(self.imtpath,
                                  "{}_mean.cimt".format(self.name)))
        del tvf

    def pod_decomp(self):
        print("    Making POD decomposition    ")
        # improt data
        tvf = imtio.import_from_file(join(self.imtpath, "{}_cln.cimt"
                                          .format(self.name)))
        # make the decomposition
        pod = imtpod.modal_decomposition(tvf, kind='pod',
                                         wanted_modes='all',
                                         max_vecs_per_node=len(tvf) + 1,
                                         verbose=False)
        # save a display
        pod.display()
        plt.savefig(join(self.respath, "{}/pod.png".format(self.name)))
        plt.close(plt.gcf())
        # save data
        imtio.export_to_file(pod, join(self.imtpath, "{}_pod.cimt"
                                       .format(self.name)))
        del pod

    def pod_reconstr(self, detailled=False):
        if self.detailled:
            print("    Making detailled POD reconstruction    ")
        else:
            print("    Making POD reconstruction    ")
        # improt data
        pod = imtio.import_from_file(join(self.imtpath, "{}_pod.cimt"
                                          .format(self.name)))
        # make reconstruction
        if detailled:
            pod.augment_temporal_resolution(fact=self.det_fact,
                                            interp='linear',
                                            inplace=True)
        wanted_modes = pod.modes_nmb[pod.get_spatial_coherence(raw=True) >
                                     self.pod_coh]
        wanted_modes = np.array(wanted_modes)
        wanted_modes = wanted_modes[wanted_modes < len(pod.modes)/2.]
        rec = pod.reconstruct(wanted_modes=wanted_modes)
        coh = pod.get_spatial_coherence()
        del pod
        # save a display
        plt.figure()
        coh.display(color='k')
        plt.xlim(0, np.max(wanted_modes)*2)
        plt.axhline(self.pod_coh, linestyle='--', color='r')
        plt.plot(wanted_modes, coh.y[wanted_modes], 'or')
        if detailled:
            plt.savefig(join(self.respath, "{}/rec_det.png".format(self.name)))
        else:
            plt.savefig(join(self.respath, "{}/rec.png".format(self.name)))
        plt.close(plt.gcf())
        # save data
        if detailled:
            imtio.export_to_file(rec, join(self.imtpath, "{}_rec_det.cimt"
                                           .format(self.name)))
        else:
            imtio.export_to_file(rec, join(self.imtpath, "{}_rec.cimt"
                                           .format(self.name)))
        del rec

    def CP_detection(self, detailled=False):
        # improt data
        if self.detailled:
            print("    Making detailled CP detection")
        else:
            print("    Making CP detection")
        mem_available = psutil.virtual_memory().free
        if detailled:
            rec = imtio.import_from_file(join(self.imtpath, "{}_rec_det.cimt"
                                              .format(self.name)))
        else:
            rec = imtio.import_from_file(join(self.imtpath, "{}_rec.cimt"
                                              .format(self.name)))
        # Check if there is enough memory to use
        filesize = mem_available - psutil.virtual_memory().free
        possible_nmb_of_copies = int(mem_available/(filesize*2))
        if possible_nmb_of_copies < 1:
            possible_nmb_of_copies = 1
        if possible_nmb_of_copies < self.thread:
            warnings.warn("Use {} instead of {} threads because of "
                          "memory limitation"
                          .format(possible_nmb_of_copies, self.thread))
            self.thread = possible_nmb_of_copies
        # getting cp positions
        traj = imtvod.get_critical_points(rec, kind='pbi',
                                          mirroring=self.mirroring,
                                          verbose=False, thread=self.thread)
        traj.compute_traj(epsilon=self.eps_traj)
        traj.clean_traj(self.nmb_min_in_traj)
        # save display
        plt.figure()
        traj.display_traj('x', filt=[True, True, False, False, False],
                          marker=None)
        if detailled:
            plt.savefig(join(self.respath, "{}/traj_x_cln_det.png"
                             .format(self.name)))
        else:
            plt.savefig(join(self.respath, "{}/traj_x_cln.png"
                             .format(self.name)))
        plt.close(plt.gcf())
        plt.figure()
        traj.display_traj('y', filt=[True, True, False, False, False],
                          marker=None)
        if detailled:
            plt.savefig(join(self.respath, "{}/traj_y_cln_det.png"
                             .format(self.name)))
        else:
            plt.savefig(join(self.respath, "{}/traj_y_cln.png"
                             .format(self.name)))
        plt.close(plt.gcf())
        # save
        if detailled:
            imtio.export_to_file(traj, join(self.imtpath, "{}_traj_det.cimt"
                                            .format(self.name)))
        else:
            imtio.export_to_file(traj, join(self.imtpath, "{}_traj.cimt"
                                            .format(self.name)))
        del traj

    def compute_everything(self):
        if not os.path.exists(join(self.respath, "{}".format(self.name))):
            os.mkdir(join(self.respath, "{}".format(self.name)))
        if not os.path.exists(join(self.imtpath, "{}_cln.cimt"
                                   .format(self.name))):
            self._display_heading()
            self.prepare_data()
            self.pod_decomp()
            self.pod_reconstr()
            self.CP_detection()
        elif not os.path.exists(join(self.imtpath, "{}_pod.cimt"
                                     .format(self.name))):
            self._display_heading()
            self.pod_decomp()
            self.pod_reconstr()
            self.CP_detection()
        elif not os.path.exists(join(self.imtpath, "{}_rec.cimt"
                                     .format(self.name))):
            self._display_heading()
            self.pod_reconstr()
            self.CP_detection()
        elif not os.path.exists(join(self.imtpath, "{}_traj.cimt"
                                     .format(self.name))):
            self._display_heading()
            self.CP_detection()
        else:
            pass
        # detailled study
        if self.detailled:
            if not os.path.exists(join(self.imtpath, "{}_rec_det.cimt"
                                       .format(self.name))):
                self._display_heading()
                self.pod_reconstr(detailled=True)
                self.CP_detection(detailled=True)
            elif not os.path.exists(join(self.imtpath, "{}_traj_det.cimt"
                                         .format(self.name))):
                self._display_heading()
                self.CP_detection(detailled=True)
        else:
            pass

    def recompute_everything(self):
        if not os.path.exists(join(self.respath, "{}".format(self.name))):
            os.mkdir(join(self.respath, "{}".format(self.name)))
        self.prepare_data()
        self.pod_decomp()
        self.pod_reconstr()
        self.CP_detection()

    def _display_heading(self):
        title = "=   {}   =".format(self.name)
        print(("\n" + "="*len(title)))
        print(title)
        print(("="*len(title)))
