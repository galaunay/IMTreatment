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

    Parameters
    ----------
    TF : TemporalFields (TemporalScalarFields or TemporalVectorFields)
        Fields to extract modes from
    wanted_modes : string or number or array of numbers
        If 'all', extract all modes,
        If a number, extract the associated mode,
        If an array, extract the associated modes.

    Retuns
    ------
    mean_field : ScalarField or VectorField object
        Mean field.
    temporal_prof : tuple of Profile object
        Modes temporal evolution.
    modes : tuple of ScalarField or VectorField objects
        Modes.
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
    mean_field = TF.get_mean_field()
    TF = TF - mean_field
    if isinstance(TF, TemporalScalarFields):
        snaps = [modred.VecHandleInMemory(TF.fields[i].values)
                 for i in np.arange(len(TF.fields))]
    elif isinstance(TF, TemporalVectorFields):
        values = [np.array([TF.fields[i].comp_x, TF.fields[i].comp_y]).transpose()
                  for i in np.arange(len(TF.fields))]
        snaps = [modred.VecHandleInMemory(values[i])
                 for i in np.arange(len(TF.fields))]
    my_POD = modred.PODHandles(np.vdot)
    # decomposing and getting modes
    eigvect, eigvals = my_POD.compute_decomp(snaps)
    eigvect = np.array(eigvect)
    eigvals = np.array(eigvals)
        # Correction if missing modes (why ?)
    wanted_modes = wanted_modes[wanted_modes < len(eigvals)]
    modes = [modred.VecHandleInMemory(np.zeros(TF.fields[0].shape))
             for i in np.arange(len(wanted_modes))]
    my_POD.compute_modes(wanted_modes, modes)
    # getting temporal evolution (maybe is there a better way to do that)
    temporal_prof = []
    for i in np.arange(len(modes)):
        if isinstance(TF, TemporalScalarFields):
            tmp_prof = [np.vdot(modes[i].get(), TF.fields[j].values)
                        for j in np.arange(len(TF.fields))]
        elif isinstance(TF, TemporalVectorFields):
            values = [np.array([TF.fields[j].comp_x, TF.fields[j].comp_y]).transpose()
                      for j in np.arange(len(TF.fields))]
            tmp_prof = [np.vdot(modes[i].get(), values[j])
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
            comp_x = modes[i].get().transpose()[0]
            comp_y = modes[i].get().transpose()[1]
            tmp_field.import_from_arrays(TF.axe_x, TF.axe_y, comp_x, comp_y,
                                         mask=False, unit_x=TF.unit_x,
                                         unit_y=TF.unit_y,
                                         unit_values=TF.unit_values)
        modes_f.append(tmp_field)
    return temporal_prof, modes_f, mean_field


def reconstruct(modes, temporal_profiles, mean_field=None, wanted_modes='all'):
    """
    Recontruct fields resolved in time from modes.

    Parameters
    ----------
    modes : tuple of ScalarField or VectorField objects
        Modes
    temporal_profiles : tuple of profiles
        Temporal evolution associated to the modes.
    mean_field : ScalarField or VectorField objects
        Mean field
    wanted_modes : string or number or array of numbers, optional
        wanted modes for reconstruction, can be 'all' for all modes, a mode
        number or an array of modes numbers.
    Returns
    -------
    TF : TemporalFields (TemporalScalarFields or TemporalVectorFields)
        Reconstructed fields.
    """
    # check parameters
    if not isinstance(modes, ARRAYTYPES):
        raise TypeError()
    modes = np.array(modes)
    if not isinstance(modes[0], (ScalarField, VectorField)):
        raise TypeError()
    if not isinstance(temporal_profiles, ARRAYTYPES):
        raise TypeError()
    temporal_profiles = np.array(temporal_profiles)
    if not isinstance(temporal_profiles[0], Profile):
        raise TypeError()
    if not len(modes) == len(temporal_profiles):
        raise ValueError()
    if mean_field is None:
        mean_field = modes[0]*0.
        mean_field.mask = False
    if not isinstance(mean_field, modes[0].__class__):
        raise TypeError()
    if isinstance(wanted_modes, STRINGTYPES):
        if wanted_modes == 'all':
            wanted_modes = np.arange(len(modes))
    elif isinstance(wanted_modes, NUMBERTYPES):
        wanted_modes = np.array([wanted_modes])
    elif isinstance(wanted_modes, ARRAYTYPES):
        wanted_modes = np.array(wanted_modes)
        if not isinstance(wanted_modes[0], NUMBERTYPES):
            raise TypeError()
    else:
        raise TypeError()
    if wanted_modes.max() > len(modes):
        raise ValueError()
    # getting datas
    axe_x, axe_y = modes[0].axe_x, modes[0].axe_y
    unit_x, unit_y = modes[0].unit_x, modes[0].unit_y
    unit_values = modes[0].unit_values
    times = temporal_profiles[0].x
    ind_times = np.arange(len(times))
    unit_times = temporal_profiles[0].unit_x
    # TSF
    if isinstance(modes[0], ScalarField):
        # mean field
        tmp_tf = np.array([mean_field.values]*len(ind_times))
        # loop on the modes
        for n in wanted_modes:
            for t in ind_times:
                tmp_tf[t] += modes[n].values*temporal_profiles[n].y[t]
        # returning
        TF = TemporalScalarFields()
        for t in ind_times:
            tmp_sf = ScalarField()
            tmp_sf.import_from_arrays(axe_x, axe_y, tmp_tf[t], unit_x=unit_x,
                                      unit_y=unit_y, unit_values=unit_values)
            TF.add_field(tmp_sf, time=times[t], unit_times=unit_times)
    # TVF
    elif isinstance(modes[0], VectorField):
        # first mode
        tmp_tf_x = np.array([mean_field.comp_x]*len(ind_times))
        tmp_tf_y = np.array([mean_field.comp_y]*len(ind_times))
        # loop on the other modes
        for n in wanted_modes:
            for t in ind_times:
                tmp_tf_x[t] += modes[n].comp_x*temporal_profiles[n].y[t]
                tmp_tf_y[t] += modes[n].comp_y*temporal_profiles[n].y[t]
        # returning
        TF = TemporalVectorFields()
        for t in ind_times:
            tmp_vf = VectorField()
            tmp_vf.import_from_arrays(axe_x, axe_y, tmp_tf_x[t], tmp_tf_y[t],
                                      unit_x=unit_x, unit_y=unit_y,
                                      unit_values=unit_values)
            TF.add_field(tmp_vf, time=times[t], unit_times=unit_times)
    return TF
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