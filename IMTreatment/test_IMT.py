# -*- coding: utf-8 -*-
#! /usr/bin/python3
"""
Created on Tue Feb 25 00:19:30 2014

@author: muahah
"""

import numpy as np
import sys
import os
sys.path.append(r'/media/FREECOM HDD/These/Modules_Python')
sys.path.append(r'/home/glaunay/Freecom/These/Modules_Python')
from IMTreatment import TemporalVectorFields, TemporalScalarFields, \
    VectorField, ScalarField, Profile, Points
import IMTreatment.vortex_detection as vod
import IMTreatment.vortex_criterions as imtcrit
import IMTreatment.file_operation as imtio
import IMTreatment.boundary_layer as imtbl
import IMTreatment.field_treatment as imtft
import IMTreatment.vortex_creation as imtvc
from IMTreatment.utils.types import TypeTest, ReturnTest
from IMTreatment.utils import make_unit
import IMTreatment.pod as pod
import IMTreatment.utils as imtutls
import Plotlib as pplt
# import IMTreatment3.pod as pod
import matplotlib.pyplot as plt
plt.ion()
#import guiqwt.pyplot as plt
import scipy.interpolate as spinterp
import scipy.spatial as spspat
import scipy.signal as spsign
import time
from pprint import pprint
import timeit
import warnings
warnings.filterwarnings('error')



if True:
    ####################
    ### TEST POINTS  ###
    ####################
    nmb_pts = 100
    interv_y = [-50, 50]
    interv_x = [-200, 200]
    interv_v = [0, 100]
    dx = interv_x[1] - interv_x[0]
    dy = interv_y[1] - interv_y[0]
    dv = interv_v[1] - interv_v[0]
    xy = np.empty((nmb_pts, 2), dtype=float)
    v = np.empty((nmb_pts,), dtype=float)
    for i in np.arange(nmb_pts):
        xy[i, :] = [np.random.rand()*dx + interv_x[0],
                    np.random.rand()*dy + interv_y[0]]
        v[i] = np.random.rand()*dv - interv_v[0]
    pts = Points(xy, v, unit_x=make_unit("m"), unit_y=make_unit("m"))


    ########################
    ### TEST CRITPOINTS  ###
    ########################
    cp = vod.CritPoints('s')
    for t in range(100):
        xy = np.empty((nmb_pts, 2), dtype=float)
        v = np.empty((nmb_pts,), dtype=float)
        for i in np.arange(nmb_pts):
            xy[i, :] = [np.random.rand()*dx + interv_x[0],
                        np.random.rand()*dy + interv_y[0]]
            v[i] = np.random.rand()*dv - interv_v[0]
        pts = Points(xy, v, unit_x=make_unit("m"), unit_y=make_unit("m"))
        for i in np.arange(nmb_pts):
            xy[i, :] = [np.random.rand()*dx + interv_x[0],
                        np.random.rand()*dy + interv_y[0]]
            v[i] = np.random.rand()*dv - interv_v[0]
        pts2 = Points(xy, v, unit_x=make_unit("m"), unit_y=make_unit("m"))
        cp.add_point(foc=pts, foc_c=pts2, time=t)


    ####################
    ### TEST PROFILE ###
    ####################
    dim = 100000
    x = np.arange(0, dim, 1)
    f = 0.01
    f2 = 0.001
    y = np.sin(x*np.pi*2*f) + np.sin(x*np.pi*2*f2) + np.random.rand(dim)*.5
    prof = Profile(x, y, unit_x=make_unit('m'), unit_y=make_unit('mm'),
                   name="test")
    mask = np.zeros((len(x)))
    mask[0:20] = 1
    mask[-10:] = 1
    prof.mask = mask


    ########################
    ### TEST SCALARFIELD ###
    ########################
    SF = ScalarField()
    dim1 = 50
    dim2 = 50
    axe_x = np.arange(dim1)*1
    axe_y = np.arange(dim2)*.5
    Y, X = np.meshgrid(axe_x, axe_y, indexing='ij')
    values = 5.*np.cos(4*X/axe_x[-1]*2.*np.pi) + 1*np.cos(4*Y/axe_y[-1]*2*np.pi)
    mask = np.random.rand(dim1, dim2)
    mask = mask < 0.
    SF.import_from_arrays(axe_x, axe_y, values, mask=mask, unit_x=make_unit('mm'),
                          unit_y=make_unit('mm'), unit_values=make_unit('m/s'))


    #########################
    ### TEST VECTORFIELD ###
    #########################
    dim1 = 10
    dim2 = 20
    axe_x = np.arange(dim1)*2. + 2.87236
    axe_y = np.arange(dim2)*.7 + 1.68765
    Vx = (np.random.rand(dim1, dim2) - .5)*23.
    Vy = (np.random.rand(dim1, dim2) - .5)*14.
    mask = np.random.rand(dim1, dim2)
    mask = mask < 0.1
    VF = VectorField()
    VF.import_from_arrays(axe_x, axe_y, Vx, Vy, mask=mask, unit_x=make_unit('m'),
                          unit_y=make_unit('m'), unit_values=make_unit('m/s'))
    VF.smooth(tos='gaussian', size=5, inplace=True)
    VF.comp_x += 2*(np.random.rand(*VF.shape) - 0.5)*0.
    VF.comp_y += 2*(np.random.rand(*VF.shape) - 0.5)*0.
#    VF.mask = mask

    ###################################
    ### TEST TEMPORAL VELOCITYFIELD ###.
    ###################################
    TVF = TemporalVectorFields()
    for i in np.arange(1, 200):
        freq = 0.1
        freq2 = 0.3
        Ts = .5
        f = VF*1.*np.cos((2.*np.pi*i*Ts)*freq)\
            + 1.*VF*np.cos((2.*np.pi*i*Ts)*freq2)\
            + 0.1*(np.random.rand(*VF.shape)-0.5)\
            + .01
        t = i*Ts
#        mask = np.random.rand(*VF.shape)
#        mask = mask < 0.1
#        f.mask = mask
        TVF.add_field(f, time=t, unit_times="s")



# # signal and spectrum
# plt.close('all')
# colors = pplt.get_color_cycles()
# f = 0.01
# f2 = 0.001
# cfreq = [0.009, 0.011]
# order = 2
# fig, axs = plt.subplots(2, 2)
# plt.sca(axs[0, 0])
# prof.display()
# plt.xlim(0, 30/f)
# plt.sca(axs[1, 0])
# spec = prof.get_spectrum()
# spec.display('loglog')
# plt.axvspan(cfreq[0], cfreq[1], color=colors[1], alpha=.2)
# for cf in cfreq:
#     plt.axvline(cf, color=colors[1])
# # make spectral filtering
# plt.sca(axs[0, 1])
# fprof = prof.spectral_filtering(*cfreq, order=order)
# fprof.display()
# plt.xlim(0, 30/f)
# plt.sca(axs[1, 1])
# fspec = fprof.get_spectrum()
# fspec.display('loglog')
# plt.axvspan(cfreq[0], cfreq[1], color=colors[1], alpha=.2)
# for cf in cfreq:
#     plt.axvline(cf, color=colors[1])


# Test on TVF
fmin = 0.25
fmax = 0.35
spec = TVF.get_temporal_spectrum_over_area('comp_x',
                                           [TVF.axe_x[0], TVF.axe_x[-1]],
                                           [TVF.axe_y[0], TVF.axe_y[-1]])

# make filtering
# tsf = TVF._get_comp_spectral_filtering("Vx", fmin, fmax, order=2)
tvf = TVF.get_spectral_filtering(fmin, fmax, order=2)

# Check
fspec = tvf.get_temporal_spectrum_over_area('comp_x',
                                            [TVF.axe_x[0], TVF.axe_x[-1]],
                                            [TVF.axe_y[0], TVF.axe_y[-1]])

plt.figure()
colors = pplt.get_color_cycles()
spec.display()
fspec.display()
plt.axvspan(fmin, fmax, color=colors[1], alpha=.2)
for bwi in [fmin, fmax]:
    plt.axvline(bwi, color=colors[1])
