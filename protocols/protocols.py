import numpy as np
import matplotlib.pyplot as plt
from .. import file_operation as imtio
from .. import pod as imtpod
from .. import vortex_detection as imtvod
from os.path import join
import os


class POD_CP_protocol(object):
    def __init__(self, name, imtpath, respath, crop_x=[-np.inf, np.inf],
                 crop_y=[-np.inf, np.inf], hard_crop=True,
                 pod_coh=0.05, mirroring=[[2, 0], [1, 0]], eps_traj=15.,
                 temporal_scale=1., nmb_min_in_traj=1) :
        self.name = name
        self.imtpath = imtpath
        self.respath = respath
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.hard_crop = hard_crop
        self.pod = None
        self.pod_coh = pod_coh     
        self.mirroring = mirroring
        self.eps_traj = eps_traj
        self.temporal_scale = temporal_scale
        self.nmb_min_in_traj = nmb_min_in_traj
        self.len_data = None
        
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
        tvf.crop(intervx=self.crop_x, intervy=self.crop_y, inplace=True)
        # adjust temporal resolution
        if self.temporal_scale < 1:
            tvf.reduce_temporal_resolution(int(np.round(1./self.temporal_scale)),
                                           mean=False, inplace=True)
        # store
        imtio.export_to_file(tvf, join(self.imtpath, "{}_cln.cimt".format(self.name)))
        
    def pod_decomp(self):
        print("    Making POD decomposition    ")
        # improt data
        tvf = imtio.import_from_file(join(self.imtpath, "{}_cln.cimt".format(self.name)))
        # make the decomposition
        pod = imtpod.modal_decomposition(tvf, kind='pod',
                                         wanted_modes='all',
                                         max_vecs_per_node=len(tvf) + 1,
                                         verbose=False)
        # save data
        imtio.export_to_file(pod, join(self.imtpath, "{}_pod.cimt".format(self.name)))
        # save a display        
        plt.figure()
        pod.display()
        plt.savefig(join(self.respath, "{}/pod.png".format(self.name)))
        plt.close(plt.gcf())
    
    def pod_reconstr(self):
        print("    Making POD reconstruction    ")
        # improt data
        pod = imtio.import_from_file(join(self.imtpath, "{}_pod.cimt".format(self.name)))
        # make reconstruction
        if self.temporal_scale > 1:
            pod.augment_temporal_resolution(fact=self.temporal_scale,
                                            interp='linear',
                                            inplace=True)
        wanted_modes = pod.modes_nmb[pod.get_spatial_coherence(raw=True)
                                     > self.pod_coh]
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
        plt.savefig(join(self.respath, "{}/rec.png".format(self.name)))
        plt.close(plt.gcf())
        # save data
        imtio.export_to_file(rec, join(self.imtpath, "{}_rec.cimt".format(self.name)))
        
    def CP_detection(self):
        # improt data
        rec = imtio.import_from_file(join(self.imtpath, "{}_rec.cimt".format(self.name)))
        traj = imtvod.get_critical_points(rec, kind='pbi',
                                          mirroring=self.mirroring,
                                          verbose=False, thread=4)
        traj.compute_traj(epsilon=self.eps_traj)
        traj.clean_traj(self.nmb_min_in_traj)       
        # save display
        plt.figure()
        traj.display_traj('x')
        plt.savefig(join(self.respath, "{}/traj_x.png".format(self.name)))
        plt.close(plt.gcf())
        plt.figure()
        traj.display_traj('y')
        plt.savefig(join(self.respath, "{}/traj_y.png".format(self.name)))
        plt.close(plt.gcf())
        plt.figure()
        traj.display_traj('x', filt=[True, True, False, False, False])
        plt.savefig(join(self.respath, "{}/traj_x_cln.png".format(self.name)))
        plt.close(plt.gcf())
        plt.figure()
        traj.display_traj('y', filt=[True, True, False, False, False])
        plt.savefig(join(self.respath, "{}/traj_y_cln.png".format(self.name)))
        plt.close(plt.gcf())
        imtio.export_to_file(traj, join(self.imtpath, "{}_traj.cimt".format(self.name)))
        
    def compute_everything(self):
        if not os.path.exists(join(self.respath, "{}".format(self.name))):
            os.mkdir(join(self.respath, "{}".format(self.name)))
        if not os.path.exists(join(self.imtpath, "{}_cln.cimt".format(self.name))):
            self.prepare_data()
            self.pod_decomp()
            self.pod_reconstr()
            self.CP_detection()
        elif not os.path.exists(join(self.imtpath, "{}_pod.cimt".format(self.name))):
            self.pod_decomp()
            self.pod_reconstr()
            self.CP_detection()
        elif not os.path.exists(join(self.imtpath, "{}_rec.cimt".format(self.name))):
            self.pod_reconstr()
            self.CP_detection()
        elif not os.path.exists(join(self.imtpath, "{}_traj.cimt".format(self.name))):
            self.CP_detection()
        else:
            print("    Nothing to do")
        
    def recompute_everything(self):
        if not os.path.exists(join(self.respath, "{}".format(self.name))):
            os.mkdir(join(self.respath, "{}".format(self.name)))
        self.prepare_data()
        self.pod_decomp()
        self.pod_reconstr()
        self.CP_detection()