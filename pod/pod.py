# -*- coding: utf-8 -*-
"""
Created on Thu Mar 06 13:29:17 2014

@author: glaunay
"""

from ..core import Points, ScalarField, VectorField, make_unit,\
    ARRAYTYPES, NUMBERTYPES, STRINGTYPES, \
    TemporalVectorFields, SpatialVectorFields, TemporalScalarFields,\
    SpatialScalarFields, Field
from IMTreatment import *
import numpy as np
import pdb
import modred

class ModalFields(Field):
    """
    Class representing the result of a modal decomposition.
    """

    def __init__(self, mean_field, modes, modes_numbers, temporal_evolutions,
                 eigvals, eigvects):
        """
        Constructor
        """
        Field.__init__(self)
        # check parameters
        self.field_class = mean_field.__class__
        if not isinstance(modes, ARRAYTYPES):
            raise TypeError()
        modes = np.array(modes)
        if not isinstance(modes[0], self.field_class):
            raise TypeError()
        if not isinstance(modes_numbers, ARRAYTYPES):
            raise TypeError()
        modes_numbers = np.array(modes_numbers)
        if not modes_numbers.dtype == int:
            raise TypeError()
        if not len(modes_numbers) == len(modes):
            raise ValueError()
        if not isinstance(modes_numbers, ARRAYTYPES):
            raise TypeError()
        modes_numbers = np.array(modes_numbers)
        if not isinstance(temporal_evolutions[0], Profile):
            raise TypeError()
        if not len(temporal_evolutions) == len(modes):
            raise ValueError()
        if not isinstance(eigvals, Profile):
            raise TypeError()
        if not len(eigvals) == len(modes):
            raise ValueError()
        if not isinstance(eigvects, ARRAYTYPES):
            raise TypeError()
        eigvects = np.array(eigvects)
        if not eigvects.shape == (len(temporal_evolutions[0].x), len(modes)):
            raise ValueError()
        # storing
        self.mean_field = mean_field
        self.axe_x = mean_field.axe_x
        self.axe_y = mean_field.axe_y
        self.unit_x = mean_field.unit_x
        self.unit_y = mean_field.unit_y
        self.unit_values = mean_field.unit_values
        self.modes = modes
        self.modes_nmb = modes_numbers
        self.temp_evo = temporal_evolutions
        self.times = temporal_evolutions[0].x
        self.unit_times = temporal_evolutions[0].unit_x
        self.eigvals = eigvals
        self.eigvects = eigvects

    @property
    def modes_as_tf(self):
        if self.field_class == VectorField:
            tmp_tf = TemporalVectorFields()
            for i in np.arange(len(self.modes)):
                tmp_tf.add_field(self.modes[i], time=i,
                                 unit_times="")
            return tmp_tf.display(**kw)
        elif self.field_class == ScalarField:
            tmp_tf = TemporalScalarFields()
            for i in np.arange(len(self.modes)):
                tmp_tf.add_field(self.modes[i], time=self.times[i],
                                 unit_times=self.unit_times)
            return tmp_tf.display(**kw)

    def reconstruct(self, wanted_modes='all'):
        """
        Recontruct fields resolved in time from modes.

        Parameters
        ----------
        wanted_modes : string or number or array of numbers, optional
            wanted modes for reconstruction, can be 'all' for all modes, a mode
            number (begin at 0) or an array of modes numbers.
        Returns
        -------
        TF : TemporalFields (TemporalScalarFields or TemporalVectorFields)
            Reconstructed fields.
        """
        # check parameters

        if isinstance(wanted_modes, STRINGTYPES):
            if wanted_modes == 'all':
                wanted_modes = np.arange(len(self.modes))
        elif isinstance(wanted_modes, NUMBERTYPES):
            wanted_modes = np.array([wanted_modes])
        elif isinstance(wanted_modes, ARRAYTYPES):
            wanted_modes = np.array(wanted_modes)
            if not isinstance(wanted_modes[0], NUMBERTYPES):
                raise TypeError()
        else:
            raise TypeError()
        if wanted_modes.max() > len(self.modes):
            raise ValueError()
        # getting datas
        ind_times = np.arange(len(self.times))
        # TSF
        if self.field_class == ScalarField:
            # mean field
            tmp_tf = np.array([self.mean_field.values]*len(self.times))
            # loop on the modes
            for n in wanted_modes:
                for t in ind_times:
                    tmp_tf[t] += self.modes[n].values*self.temp_evo[n].y[t]
            # returning
            TF = TemporalScalarFields()
            for t in ind_times:
                tmp_sf = ScalarField()
                tmp_sf.import_from_arrays(self.axe_x, self.axe_y, tmp_tf[t],
                                          unit_x=self.unit_x,
                                          unit_y=self.unit_y,
                                          unit_values=self.unit_values)
                TF.add_field(tmp_sf, time=self.times[t],
                             unit_times=self.unit_times)
        # TVF
        elif self.field_class == VectorField:
            # first mode
            tmp_tf_x = np.array([self.mean_field.comp_x]*len(self.times))
            tmp_tf_y = np.array([self.mean_field.comp_y]*len(self.times))
            # loop on the other modes
            for n in wanted_modes:
                for t in ind_times:
                    tmp_tf_x[t] += self.modes[n].comp_x*self.temp_evo[n].y[t]
                    tmp_tf_y[t] += self.modes[n].comp_y*self.temp_evo[n].y[t]
            # returning
            TF = TemporalVectorFields()
            for t in ind_times:
                tmp_vf = VectorField()
                tmp_vf.import_from_arrays(self.axe_x, self.axe_y,
                                          tmp_tf_x[t], tmp_tf_y[t],
                                          unit_x=self.unit_x,
                                          unit_y=self.unit_y,
                                          unit_values=self.unit_values)
                TF.add_field(tmp_vf, time=self.times[t],
                             unit_times=self.unit_times)
        return TF


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
    modal_field : ModalField object
        .
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
    ind_fields = np.arange(len(TF.fields))
    mean_field = TF.get_mean_field()
    TF = TF - mean_field
    if isinstance(TF, TemporalScalarFields):
        snaps = [modred.VecHandleInMemory(TF.fields[i].values)
                 for i in np.arange(len(TF.fields))]
    elif isinstance(TF, TemporalVectorFields):
        values = [[TF.fields[t].comp_x, TF.fields[t].comp_y] for t in ind_fields]
        values = np.transpose(values, (0, 2, 3, 1))
        snaps = [modred.VecHandleInMemory(values[i]) for i in ind_fields]
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
            comp_x = modes[i].get()[:, :, 0]
            comp_y = modes[i].get()[:, :, 1]
            tmp_field.import_from_arrays(TF.axe_x, TF.axe_y, comp_x, comp_y,
                                         mask=False, unit_x=TF.unit_x,
                                         unit_y=TF.unit_y,
                                         unit_values=TF.unit_values)
        modes_f.append(tmp_field)
    eigvals = Profile(wanted_modes, eigvals[wanted_modes], mask=False,
                      unit_x=TF.unit_times, unit_y='')
    modal_field = ModalFields(mean_field, modes_f, wanted_modes, temporal_prof,
                              eigvals, eigvect[:, wanted_modes])
    return modal_field



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