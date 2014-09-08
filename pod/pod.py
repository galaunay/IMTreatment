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
import matplotlib.pyplot as plt

class ModalFields(Field):
    """
    Class representing the result of a modal decomposition.
    """

    def __init__(self, decomp_type, mean_field, modes, modes_numbers,
                 temporal_evolutions,
                 eigvals=None, eigvects=None, ritz_vals=None, mode_norms=None,
                 growth_rate=None, pulsation=None):
        """
        Constructor
        """
        Field.__init__(self)
        # check parameters
        if not decomp_type in ['pod', 'dmd']:
            raise ValueError()
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
        if eigvals is not None:
            if not isinstance(eigvals, Profile):
                raise TypeError()
            if not len(eigvals) == len(modes):
                raise ValueError()
        if eigvects is not None:
            if not isinstance(eigvects, ARRAYTYPES):
                raise TypeError()
            eigvects = np.array(eigvects)
            if not eigvects.shape == (len(temporal_evolutions[0].x), len(modes)):
                raise ValueError()
        if ritz_vals is not None:
            if not isinstance(ritz_vals, Profile):
                raise TypeError()
            if not len(ritz_vals) == len(modes):
                raise ValueError()
        if mode_norms is not None:
            if not isinstance(mode_norms, Profile):
                raise TypeError()
            if not len(mode_norms) == len(modes):
                raise ValueError()
        if growth_rate is not None:
            if not isinstance(growth_rate, Profile):
                raise TypeError()
            if not len(growth_rate) == len(modes):
                raise ValueError()
        if pulsation is not None:
            if not isinstance(pulsation, Profile):
                raise TypeError()
            if not len(pulsation) == len(modes):
                raise ValueError()
        # storing
        self.decomp_type = decomp_type
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
        if eigvals is not None:
            self.eigvals = eigvals
        if eigvects is not None:
            self.eigvects = eigvects
        if ritz_vals is not None:
            self.ritz_vals = ritz_vals
        if mode_norms is not None:
            self.mode_norms = mode_norms
        if growth_rate is not None:
            self.growth_rate = growth_rate
        if pulsation is not None:
            self.pulsation = pulsation

    @property
    def modes_as_tf(self):
        if self.field_class == VectorField:
            tmp_tf = TemporalVectorFields()
            for i in np.arange(len(self.modes)):
                tmp_tf.add_field(self.modes[i], time=i,
                                 unit_times="")
            return tmp_tf
        elif self.field_class == ScalarField:
            tmp_tf = TemporalScalarFields()
            for i in np.arange(len(self.modes)):
                tmp_tf.add_field(self.modes[i], time=self.times[i],
                                 unit_times=self.unit_times)
            return tmp_tf

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
#            if self.decomp_type == 'pod':
            for n in wanted_modes:
                for t in ind_times:
                    tmp_tf[t] += self.modes[n].values*self.temp_evo[n].y[t]
#            elif self.decomp_type == 'dmd':
#                for t in ind_times:
#                    for n in wanted_modes:
#                        sigma = self.growth_rate.y[n]
#                        omega = self.pulsation.y[n]
#                        comp = np.complex(0, 1)
#                        tmp_tf[t] += self.modes[n].values\
#                                   * np.exp((sigma + comp*omega)\
#                                            * self.times[t])
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

    def display(self):
        """
        Display some important diagram for the decomposition.
        """
        if self.decomp_type == 'pod':
            plt.figure()
            plt.subplot(2, 3, 1)

            plt.subplot(2, 3, 2)
            self.modes[0].display()
            plt.title("Mode 1")
            plt.subplot(2, 3, 4)
            self.eigvals.display()
            plt.title('Eigenvalues evolution')
            plt.subplot(2, 3, 5)
            self.temp_evo[0].display()
            plt.title("Temporal evolution of mode 1")
            plt.subplot(2, 3, 3)
            self.modes[1].display()
            plt.title("Mode 2")
            plt.subplot(2, 3, 6)
            self.temp_evo[1].display()
            plt.title("Temporal evolution of mode 2")
        elif self.decomp_type == 'dmd':
            self.pulsation.change_unit('y', 'rad/s')
            self.growth_rate.change_unit('y', '1/s')
            plt.figure()
            plt.subplot(2, 3, 1)
            plt.plot(np.real(self.ritz_vals.y), np.imag(self.ritz_vals.y), 'o')
            plt.title("Ritz eigenvalues in the complexe plane")
            plt.xlabel("Real part of Ritz eigenvalue")
            plt.ylabel("Imaginary part of Ritz eigenvalue")
            plt.subplot(2, 3, 2)
            plt.plot(self.pulsation.y, self.growth_rate.y, 'o')
            plt.title("Growth rate spectrum")
            plt.xlabel("Pulsation [rad/s]")
            plt.ylabel("Growth rate [1/s]")
            plt.subplot(2, 3, 3)
            sorted_omega = np.sort(self.pulsation.y)
            delta_omega = np.abs(sorted_omega[1] - sorted_omega[0])
            width = delta_omega/2.
            plt.bar(self.pulsation.y - width/2., self.mode_norms.y,
                    width=width)
            plt.title("Mode amplitude spectrum")
            plt.xlabel("Pulsation [rad/s]")
            plt.ylabel("Mode amplitude []")
            plt.subplot(2, 3, 4)
            stab_sort = np.argsort(self.growth_rate.y)
            self.modes[stab_sort[-1]].display()
            plt.title("More instable mode (pulsation={:.2f})"
                      .format(self.pulsation.y[stab_sort[-1]]))
            plt.subplot(2, 3, 5)
            self.modes[stab_sort[-2]].display()
            plt.title("Second more instable mode (pulsation={:.2f})"
                      .format(self.pulsation.y[stab_sort[-2]]))
            plt.subplot(2, 3, 6)
            norm_sort = np.argsort(self.mode_norms.y)
            self.modes[norm_sort[-1]].display()
            plt.title("Mode with the bigger norm (pulsation={:.2f})"
                      .format(self.pulsation.y[norm_sort[-1]]))


def modal_decomposition(TF, kind='pod', wanted_modes='all'):
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
    if not isinstance(kind, STRINGTYPES):
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
    # remove mean field or not
    ind_fields = np.arange(len(TF.fields))
    mean_field = TF.get_mean_field()
    TF = TF - mean_field
    # link data
    if isinstance(TF, TemporalScalarFields):
        snaps = [modred.VecHandleInMemory(TF.fields[i].values)
                 for i in np.arange(len(TF.fields))]
        values = [TF.fields[t].values for t in ind_fields]
    elif isinstance(TF, TemporalVectorFields):
        values = [[TF.fields[t].comp_x, TF.fields[t].comp_y] for t in ind_fields]
        values = np.transpose(values, (0, 2, 3, 1))
        snaps = [modred.VecHandleInMemory(values[i]) for i in ind_fields]
    # setting the decomposition mode
    eigvals = None
    eigvect = None
    ritz_vals = None
    mode_norms = None
    growth_rate = None
    pulsation = None
    if kind == 'pod':
        my_decomp = modred.PODHandles(np.vdot)
        eigvect, eigvals = my_decomp.compute_decomp(snaps)
        wanted_modes = wanted_modes[wanted_modes < len(eigvals)]
        eigvect = np.array(eigvect)
        eigvect = eigvect[:, wanted_modes]
        eigvals = Profile(wanted_modes, eigvals[wanted_modes], mask=False,
                          unit_x=TF.unit_times, unit_y='')
    elif kind == 'dmd':
        my_decomp = modred.DMDHandles(np.vdot, verbosity=1)
        ritz_vals, mode_norms, build_coeffs = my_decomp.compute_decomp(snaps)
        wanted_modes = wanted_modes[wanted_modes < len(ritz_vals)]
        # supplementary charac
        delta_t = TF.times[1] - TF.times[0]
        lambd_i = np.imag(ritz_vals)
        lambd_r = np.real(ritz_vals)
        lambd_mod = np.sqrt(lambd_i**2 + lambd_r**2)
        lambd_arg = np.zeros((len(ritz_vals)))
        mask = np.logical_and(lambd_i == 0, lambd_r <= 0)
        filt = np.logical_not(mask)
        lambd_arg[mask] = np.pi
        lambd_arg[filt] = 2*np.arctan(lambd_i[filt]/(lambd_r[filt]
                                                     + lambd_mod[filt]))
        sigma = np.log(lambd_mod)/delta_t
        omega = lambd_arg/delta_t
        # creating profiles
        ritz_vals = Profile(wanted_modes, ritz_vals[wanted_modes], mask=False,
                            unit_x=TF.unit_times, unit_y='')
        mode_norms = Profile(wanted_modes, mode_norms[wanted_modes],
                             mask=False, unit_x=TF.unit_times, unit_y='')
        growth_rate = Profile(wanted_modes, sigma[wanted_modes], mask=False,
                              unit_x=TF.unit_times, unit_y=1/TF.unit_times)
        pulsation = Profile(wanted_modes, omega[wanted_modes], mask=False,
                            unit_x=TF.unit_times,
                            unit_y=make_unit('rad')/TF.unit_times)
    else:
        raise ValueError()
    # decomposing and getting modes
    modes = [modred.VecHandleInMemory(np.zeros(TF.fields[0].shape))
             for i in np.arange(len(wanted_modes))]
    my_decomp.compute_modes(wanted_modes, modes)
    # getting temporal evolution (maybe is there a better way to do that)
    # TODO : améliorer pob reconstruction
    temporal_prof = []
    if kind == 'pod':
        for i in np.arange(len(modes)):
            tmp_prof = [np.vdot(modes[i].get(), values[j])
                        for j in np.arange(len(TF.fields))]
            tmp_prof = Profile(TF.times, tmp_prof, mask=False,
                               unit_x=TF.unit_times,
                               unit_y=TF.unit_values)
            temporal_prof.append(tmp_prof)
    elif kind == 'dmd':
        temporal_prof = []
        for n in np.arange(len(modes)):
            tmp_prof = np.exp((growth_rate.y[n]
                               + np.complex(0, 1)*pulsation.y[n])
                              * TF.times)
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
    modal_field = ModalFields(kind, mean_field, modes_f, wanted_modes,
                              temporal_prof, eigvals=eigvals, eigvects=eigvect,
                              ritz_vals=ritz_vals, mode_norms=mode_norms,
                              growth_rate=growth_rate, pulsation=pulsation)
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