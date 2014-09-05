# -*- coding: utf-8 -*-
"""
Created on Thu Mar 06 13:29:17 2014

@author: glaunay
"""

from ..core import Points, ScalarField, VectorField, make_unit,\
    ARRAYTYPES, NUMBERTYPES, STRINGTYPES, \
    TemporalVectorFields, SpatialVectorFields, TemporalScalarFields,\
    SpatialScalarFields
from IMTreatment import *
import numpy as np
import pdb
import modred

def pod(TF, wanted_modes='all'):
    """
    Compute POD modes of the given fields using the snapshot method.
    """
    # test parameters
    if not isinstance(TF, TemporalFields):
        raise TypeError()
    if isinstance(wanted_modes, STRINGTYPES):
        if not wanted_modes == 'all':
            raise ValueError()
        wanted_modes = np.arange(len(TF.fields))
    elif isinstance(wanted_modes, NUMBERTYPES):
        wanted_modes = np.array([wanted_modes])
    elif isinstance(wanted_modes, ARRAYTYPES):
        wanted_modes = np.array(wanted_modes)
        if wanted_modes.min() < 0 or wanted_modes.max() > len(TF.fields):
            raise ValueError()
    else:
        raise TypeError()
    # link data
    if isinstance(TF, TemporalScalarFields):
        snaps = [modred.VecHandleInMemory(TF.fields[i].values)
                 for i in np.arange(len(TF.fields))]
    elif isinstance(TF, TemporalVectorFields):
        snaps = [modred.VecHandleInMemory(TF.fields[i].comp_x)
                 for i in np.arange(len(TF.fields))]
    modes = [modred.VecHandleInMemory(np.zeros(TF.fields[0].shape))
             for i in np.arange(len(wanted_modes))]
    my_POD = modred.PODHandles(np.vdot)
    # decomposing and getting modes
    eigvect, eigvals = my_POD.compute_decomp(snaps)
    eigvect = np.array(eigvect)
    eigvals = np.array(eigvals)
    my_POD.compute_modes(wanted_modes, modes)
    # getting temporal evolution (maybe is there a better way to do that)
    temporal_prof = []
    for i in np.arange(len(modes)):
        tmp_prof = [np.vdot(modes[i].get(), TF.fields[j].values)
                   for j in np.arange(len(TF.fields))]
        tmp_prof = Profile(TF.times, tmp_prof, mask=False,
                           unit_x=TF.unit_times,
                           unit_y=TF.unit_values)
        temporal_prof.append(tmp_prof)
    # returning
    modes_f = []
    for i in np.arange(len(modes)):
        if isinstance(TF, TemporalScalarFields):
            tmp_field = ScalarField()
            tmp_field.import_from_arrays(TF.axe_x, TF.axe_y, modes[i].get(),
                                         mask=False, unit_x=TF.unit_x,
                                         unit_y=TF.unit_y,
                                         unit_values=TF.unit_values)
        else:
            tmp_field = VectorField()
            tmp_field.import_from_arrays(TF.axe_x, TF.axe_y, modes[i].get()[0],
                                         modes[i].get([1]),
                                         mask=False, unit_x=TF.unit_x,
                                         unit_y=TF.unit_y,
                                         unit_values=TF.unit_values)
        modes_f.append(tmp_field)
    return temporal_prof, modes_f


# # field creation
#size_x, size_y = 100, 100
#nmb = 100
#Tt = 4.
#Tx = 50.
#Ty = 20.
#coef_rand = 10
#x = np.arange(size_x)
#y = np.arange(size_y)
#X, Y = np.meshgrid(x, y)
#fields = TemporalVectorFields()
#for n in np.arange(nmb):
#    array = np.sin(X/Tx*2*np.pi) + np.sin(Y/Ty*2*np.pi)
#    array += coef_rand*np.random.rand(size_x, size_y)
#    array *= np.sin(n/Tt*2*np.pi)
#    field = VectorField()
#    field.import_from_arrays(x, y, array, array)
#    fields.add_field(field)
#
## POD
#nmb_modes = 10
#snaps = [modred.VecHandleInMemory(fields.fields[i].comp_x) for i in np.arange(len(fields.fields))]
#modes = [modred.VecHandleInMemory() for i in np.arange(nmb_modes)]
#my_POD = modred.PODHandles(np.vdot)
#eigvect, eigval = my_POD.compute_decomp(snaps)
#my_POD.compute_modes(np.arange(nmb_modes), modes)
#
## display
##   initial
#plt.figure()
#plt.imshow(fields.fields[2].comp_x)
## modes
#plt.figure()
#for i in [0, 1, 2, 3]:
#    plt.subplot(2, 2, i+1)
#    plt.imshow(modes[i].get())
#    plt.colorbar()
#    plt.title('eigval = {}'.format(eigval[i]))
## reconstruction 1 mode
#plt.figure()
#plt.imshow(modes[0].get())