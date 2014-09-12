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
                 temporal_evolutions, modes_nrj,
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
        if not isinstance(modes_nrj, Profile):
            raise TypeError()
        if not len(modes_nrj.x) == len(modes):
            raise ShapeError()
        if eigvals is not None:
            if not isinstance(eigvals, Profile):
                raise TypeError()
            if not len(eigvals) == len(modes):
                raise ValueError()
        if eigvects is not None:
            if not isinstance(eigvects, ARRAYTYPES):
                raise TypeError()
            eigvects = np.array(eigvects)
            if not eigvects.shape == (len(temporal_evolutions[0].x),
                                      len(modes)):
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
        self.modes_nrj = modes_nrj
        self.modes_nrj_cum = modes_nrj.copy()
        self.modes_nrj_cum.y = np.cumsum(self.modes_nrj.y)
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

    def reconstruct(self, wanted_modes='all', times=None):
        """
        Recontruct fields resolved in time from modes.

        Parameters
        ----------
        wanted_modes : string or number or array of numbers, optional
            wanted modes for reconstruction :
            If 'all' (default), all modes are used
            If an array of integers, the wanted modes are used
            If an integer, the wanted first modes are used.
        times : tuple of numbers
            If specified, reconstruction is computed on the wanted times,
            else, times used for decomposition are used.
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
            if self.decomp_type == 'pod':
                wanted_modes = np.arange(wanted_modes)
            elif self.decomp_type == 'dmd':
                wanted_modes = (np.argsort(np.abs(self.growth_rate.y))
                                [0:wanted_modes])
            else:
                raise ValueError()
        elif isinstance(wanted_modes, ARRAYTYPES):
            wanted_modes = np.array(wanted_modes)
            if not isinstance(wanted_modes[0], NUMBERTYPES):
                raise TypeError()
            if wanted_modes.max() > len(self.modes):
                raise ValueError()
        else:
            raise TypeError()
        if times is None:
            times = self.times
        if not isinstance(times, ARRAYTYPES):
            raise TypeError()
        times = np.array(times)
        if times.ndim != 1:
            raise ValueError()
        if self.decomp_type == 'pod':
            if (times.max() > self.times.max()
                    or times.min() < self.times.min()):
                raise ValueError()
        # getting datas
        ind_times = np.arange(len(times))
        # reconstruction temporal evolution if needed
        if self.decomp_type == 'pod':
            temp_evo = self.temp_evo
        elif self.decomp_type == 'dmd' and np.all(self.times == times):
            temp_evo = self.temp_evo
        elif self.decomp_type == 'dmd' and not np.all(self.times == times):
            temp_evo = []
            delta_t1 = self.times[1] - self.times[0]
            ks = times/delta_t1
            for n in np.arange(len(self.modes)):
                tmp_prof = [self.ritz_vals.y[n]**(k - 1) for k in ks]
                tmp_prof = Profile(times, tmp_prof, mask=False,
                                   unit_x=self.unit_times,
                                   unit_y=self.unit_values)
                temp_evo.append(tmp_prof)
        # TSF
        if self.field_class == ScalarField:
            # mean field
            tmp_tf = np.array([self.mean_field.values]*len(times))
            # loop on the modes
            for n in wanted_modes:
                tmp_mode = self.modes[n].values
                tmp_prof = temp_evo[n]
                for t in ind_times:
                    coef = tmp_prof.get_interpolated_value(x=times[t])[0]
                    tmp_tf[t] += np.real(tmp_mode*coef)
            # returning
            TF = TemporalScalarFields()
            for t in ind_times:
                tmp_sf = ScalarField()
                tmp_sf.import_from_arrays(self.axe_x, self.axe_y, tmp_tf[t],
                                          unit_x=self.unit_x,
                                          unit_y=self.unit_y,
                                          unit_values=self.unit_values)
                TF.add_field(tmp_sf, time=times[t], unit_times=self.unit_times)
        # TVF
        elif self.field_class == VectorField:
            # mean field
            tmp_tf_x = np.array([self.mean_field.comp_x]*len(times))
            tmp_tf_y = np.array([self.mean_field.comp_y]*len(times))
            # loop on the modes
            for n in wanted_modes:
                tmp_mode_x = self.modes[n].comp_x
                tmp_mode_y = self.modes[n].comp_y
                tmp_prof = temp_evo[n]
                for t in ind_times:
                    coef = tmp_prof.get_interpolated_value(x=times[t])[0]
                    tmp_tf_x[t] += np.real(tmp_mode_x*coef)
                    tmp_tf_y[t] += np.real(tmp_mode_y*coef)
            # returning
            TF = TemporalVectorFields()
            for t in ind_times:
                tmp_vf = VectorField()
                tmp_vf.import_from_arrays(self.axe_x, self.axe_y,
                                          tmp_tf_x[t], tmp_tf_y[t],
                                          unit_x=self.unit_x,
                                          unit_y=self.unit_y,
                                          unit_values=self.unit_values)
                TF.add_field(tmp_vf, time=times[t], unit_times=self.unit_times)
        return TF

    def get_temporal_coherence(self, raw=False):
        """
        Return a profile where each value represent the probability for a mode
        to be coherent (non-random).
        Can be used to determine the modes to take to filter the turbulence
        (and so perform a tri-decomposition (mean + coherent + turbulent))

        Parameters
        ----------
        raw : bool, optional
            If 'False' (default), a Profile object is returned
            If 'True', an array is returned

        Returns
        -------
        var_spec : array or Profile object
            Probability estimation for each mode of being coherent in time.

        Notes
        -----
        Returned values are, for each modes, the variance of the normalized
        spectrum of the temporal evolution.
        Variance is high when spectrum show predominant frequencies
        (coherent behavior), inversely, variance is low when spectrum is
        nearly uniform (random behavior).
        """
        # computing maximal variance
        max_std_spec = np.zeros(len(self.times))
        max_std_spec[0] = 1
        max_std_spec /= np.trapz(max_std_spec)
        max_std = np.std(max_std_spec)
        var_spec = np.empty((len(self.modes)))
        for n in np.arange(len(self.modes)):
            prof = self.temp_evo[n]
            spec = prof.get_spectrum(scaling='density')
            spec /= np.trapz(spec.y)
            var_spec[n] = np.std(spec.y)/max_std
        if raw:
            return var_spec
        else:
            prof = Profile(np.arange(len(self.modes)), var_spec, unit_x='',
                           unit_y='', name="")
            return prof
#        from scipy.stats import normaltest
#        p_vals = np.empty((len(self.modes)))
#        for n in np.arange(len(self.modes)):
#            values = self.temp_evo[n].y
#            p_vals[n] = normaltest(values)[1]

    def display(self, figsize=(15, 10)):
        """
        Display some important diagram for the decomposition.
        """
        if self.decomp_type == 'pod':
            plt.figure(figsize=figsize)
            plt.subplot(2, 3, 1)
            p_vals = self.get_temporal_coherence()
            p_vals.display()
            plt.title('Coherence indicator')
            plt.xlabel('Modes')
            plt.ylabel('Coherence')
            plt.subplot(2, 3, 2)
            self.modes[0].display()
            plt.title("Mode 1")
            plt.subplot(2, 3, 4)
            tmp_prof = self.modes_nrj_cum.copy()
            tmp_prof.y /= np.sum(self.modes_nrj.y)
            tmp_prof.display()
            plt.title('Cumulative modes energy')
            plt.ylim(ymin=0, ymax=1)
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
            plt.figure(figsize=figsize)
            plt.subplot(2, 3, 1)
            plt.plot(np.real(self.ritz_vals.y), np.imag(self.ritz_vals.y), 'o')
            plt.title("Ritz eigenvalues in the complexe plane")
            plt.xlabel("Real part of Ritz eigenvalue")
            plt.ylabel("Imaginary part of Ritz eigenvalue")
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.subplot(2, 3, 2)
            plt.plot(self.pulsation.y, self.growth_rate.y, 'o')
            plt.title("Growth rate spectrum")
            plt.xlabel("Pulsation [rad/s]")
            plt.ylabel("Growth rate [1/s]")
            x_max = np.max([np.abs(plt.xlim()[0]), np.abs(plt.xlim()[1])])
            plt.xlim(-x_max, x_max)
            plt.subplot(2, 3, 3)
            sorted_omega = np.sort(self.pulsation.y)
            delta_omega = np.mean(np.abs(sorted_omega[1::]
                                  - sorted_omega[0:-1:]))
            width = delta_omega/2.
            plt.bar(self.pulsation.y - width/2., self.mode_norms.y,
                    width=width)
            plt.title("Mode amplitude spectrum")
            plt.xlabel("Pulsation [rad/s]")
            plt.ylabel("Mode amplitude []")
            plt.subplot(2, 3, 5)
            stab_sort = np.argsort(np.abs(self.growth_rate.y))
            tmp_sf = self.modes[stab_sort[0]].copy()
            if isinstance(tmp_sf, ScalarField):
                tmp_sf.values = np.real(tmp_sf.values)
            elif isinstance(tmp_sf, VectorField):
                tmp_sf.comp_x = np.real(tmp_sf.comp_x)
                tmp_sf.comp_y = np.real(tmp_sf.comp_y)
            tmp_sf.display()
            plt.title("More stable mode (pulsation={:.2f})\n"
                      "(Real representation)"
                      .format(self.pulsation.y[stab_sort[-0]]))
            plt.subplot(2, 3, 4)
            tmp_prof = self.modes_nrj.copy()
            tmp_prof.y /= np.sum(tmp_prof.y)
            tmp_prof.display()
            plt.title('Modes energy')
            plt.ylim(ymin=0, ymax=1)
            plt.subplot(2, 3, 6)
            norm_sort = np.argsort(self.mode_norms.y)
            tmp_sf = self.modes[norm_sort[-1]].copy()
            if isinstance(tmp_sf, ScalarField):
                tmp_sf.values = np.real(tmp_sf.values)
            elif isinstance(tmp_sf, VectorField):
                tmp_sf.comp_x = np.real(tmp_sf.comp_x)
                tmp_sf.comp_y = np.real(tmp_sf.comp_y)
            tmp_sf.display()
            plt.title("Mode with the bigger norm (pulsation={:.2f})\n"
                      "(Real representation)"
                      .format(self.pulsation.y[norm_sort[-1]]))
        plt.tight_layout()



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
    ### Test parameters
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
    ### Link data
    if isinstance(TF, TemporalScalarFields):
        snaps = [modred.VecHandleInMemory(TF.fields[i].values)
                 for i in np.arange(len(TF.fields))]
        values = [TF.fields[t].values for t in ind_fields]
    elif isinstance(TF, TemporalVectorFields):
        values = [[TF.fields[t].comp_x, TF.fields[t].comp_y]
                  for t in ind_fields]
        values = np.transpose(values, (0, 2, 3, 1))
        snaps = [modred.VecHandleInMemory(values[i]) for i in ind_fields]
    ### Setting the decomposition mode
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
        my_decomp = modred.DMDHandles(np.vdot)
        ritz_vals, mode_norms, build_coeffs = my_decomp.compute_decomp(snaps)
        wanted_modes = wanted_modes[wanted_modes < len(ritz_vals)]
        # supplementary charac
        delta_t = TF.times[1] - TF.times[0]
        lambd_i = np.imag(ritz_vals)
        lambd_r = np.real(ritz_vals)
        lambd_mod = np.abs(ritz_vals)
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
                              unit_x=TF.unit_times, unit_y=1./TF.unit_times)
        pulsation = Profile(wanted_modes, omega[wanted_modes], mask=False,
                            unit_x=TF.unit_times,
                            unit_y=make_unit('rad')/TF.unit_times)
    else:
        raise ValueError("Unknown kind of decomposition : {}".format(kind))
    ### Decomposing and getting modes
    modes = [modred.VecHandleInMemory(np.zeros(TF.fields[0].shape))
             for i in np.arange(len(wanted_modes))]
    my_decomp.compute_modes(wanted_modes, modes)
    ### Getting temporal evolution (maybe is there a better way to do that)
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
            tmp_prof = [ritz_vals.y[n]**(k)
                        for k in np.arange(len(TF.fields))]
            tmp_prof = Profile(TF.times, tmp_prof, mask=False,
                               unit_x=TF.unit_times,
                               unit_y=TF.unit_values)
            temporal_prof.append(tmp_prof)
    ### Getting NRJ repartition on modes
    modes_nrj = np.zeros((len(modes),))
    for n in np.arange(len(modes)):
        if isinstance(TF, TemporalScalarFields):
            magnitude = 1./2.*np.real(modes[n].get())**2
        elif isinstance(TF, TemporalVectorFields):
            magnitude = 1./2.*(np.real(modes[n].get()[:, :, 0])**2
                               + np.real(modes[n].get()[:, :, 1])**2)
        coef_temp = np.mean(np.real(temporal_prof[n].y)**2)
        modes_nrj[n] = np.sum(magnitude)*coef_temp
    modes_nrj = Profile(np.arange(len(modes)), modes_nrj, mask=False,
                        unit_x="", unit_y=TF.unit_values**2, name="")
    ### Returning
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
                              temporal_prof, modes_nrj,
                              eigvals=eigvals, eigvects=eigvect,
                              ritz_vals=ritz_vals, mode_norms=mode_norms,
                              growth_rate=growth_rate, pulsation=pulsation)
    return modal_field
