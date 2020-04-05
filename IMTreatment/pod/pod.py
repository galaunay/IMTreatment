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

from ..core import ScalarField, VectorField, \
    TemporalVectorFields, TemporalScalarFields,\
    Field, Profile, TemporalFields
from ..utils import make_unit
from ..utils.types import ARRAYTYPES, NUMBERTYPES, STRINGTYPES
import numpy as np
try:
    import modred
    MODRED = True
except ImportError:
    MODRED = False
import matplotlib.pyplot as plt
from .. import plotlib as pplt
import scipy.integrate as spint


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
        # Check if modred available
        if not MODRED:
            raise Exception("This feature need 'modred' to be installed")
        Field.__init__(self)
        # check parameters
        if decomp_type not in ['pod', 'dmd', 'bpod']:
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
        self.__temp_coh = None
        self.__spat_coh = None

    @property
    def modes_as_tf(self):
        if self.field_class == VectorField:
            tmp_tf = TemporalVectorFields()
            for i in np.arange(len(self.modes)):
                tmp_tf.add_field(self.modes[i], time=i + 1,
                                 unit_times="")
            return tmp_tf
        elif self.field_class == ScalarField:
            tmp_tf = TemporalScalarFields()
            for i in np.arange(len(self.modes)):
                tmp_tf.add_field(self.modes[i], time=i + 1,
                                 unit_times="")
            return tmp_tf

    def crop_modal_base(self, modes_to_keep=None, modes_to_remove=None,
                        inplace=True):
        """
        Remove some modes from the modal base.

        Parameters
        ----------
        modes_to_keep : array of integers or integer
            If an integer, the first N modes are kept, if an array of indices,
            all the associated modes are conserved.
        modes_to_remove :  array of integers or integer
            If an integer, the last N modes are removed,
            if an array of indices, all the associated modes are removed.
        """
        # check
        if modes_to_keep is None and modes_to_remove is None:
            raise ValueError()
        if modes_to_keep is not None:
            if isinstance(modes_to_keep, int):
                modes_to_keep = np.arange(modes_to_keep)
            if not isinstance(modes_to_keep, ARRAYTYPES):
                raise TypeError()
            modes_to_keep = np.array(modes_to_keep, dtype=int)
        else:
            if isinstance(modes_to_remove, int):
                modes_to_remove = np.arrange(len(self.modes) - modes_to_remove,
                                             len(self.modes))
            if not isinstance(modes_to_remove, ARRAYTYPES):
                raise TypeError()
            modes_to_remove = np.array(modes_to_remove, dtype=int)
            modes_to_keep = np.arrange(len(self.modes))
            modes_to_keep = np.delete(modes_to_keep, modes_to_remove)
        # get data
        if inplace:
            tmp_pod = self
        else:
            tmp_pod = self.copy()
        # prepare new containers
        new_modes = []
        new_temp_evo = []
        if self.decomp_type == 'pod':
            new_eigvals = Profile(unit_x=self.eigvals.unit_x,
                                  unit_y=self.eigvals.unit_y)
            new_eigvects = []
        elif self.decomp_type == "dmd":
            new_growth_rate = Profile(unit_x=self.eigvals.unit_x,
                                      unit_y=self.eigvals.unit_y)
            new_mode_norms = Profile(unit_x=self.eigvals.unit_x,
                                     unit_y=self.eigvals.unit_y)
            new_pulsation = Profile(unit_x=self.eigvals.unit_x,
                                    unit_y=self.eigvals.unit_y)
            new_ritz_vals = Profile(unit_x=self.eigvals.unit_x,
                                    unit_y=self.eigvals.unit_y)
        else:
            raise Exception('Not implemented for {}'.format(self.decomp_type))
        # loop on wanted modes
        for ind in modes_to_keep:
            new_modes.append(tmp_pod.modes[ind])
            new_temp_evo.append(tmp_pod.temp_evo[ind])
            if self.decomp_type == 'pod':
                new_eigvals.add_point(tmp_pod.eigvals.x[ind],
                                      tmp_pod.eigvals.y[ind])
                tmp_vect = []
                for ind2 in modes_to_keep:
                    tmp_vect.append(tmp_pod.eigvects[ind, ind2])
                new_eigvects.append(tmp_vect)
            if self.decomp_type == "dmd":
                new_growth_rate.add_point(tmp_pod.growth_rate.x[ind],
                                          tmp_pod.growth_rate.y[ind])
                new_mode_norms.add_point(tmp_pod.mode_norms.x[ind],
                                         tmp_pod.mode_norms.y[ind])
                new_pulsation.add_point(tmp_pod.pulsation.x[ind],
                                        tmp_pod.pulsation.y[ind])
                new_ritz_vals.add_point(tmp_pod.ritz_vals.x[ind],
                                        tmp_pod.ritz_vals.y[ind])
        # store
        tmp_pod.modes_nmb = tmp_pod.modes_nmb[modes_to_keep]
        tmp_pod.modes = new_modes
        tmp_pod.temp_evo = new_temp_evo
        if self.decomp_type == 'pod':
            tmp_pod.eigvals = new_eigvals
            tmp_pod.eigvects = np.array(new_eigvects, dtype=float)
        if self.decomp_type == "dmd":
            tmp_pod.growth_rate = new_growth_rate
            tmp_pod.mode_norms = new_mode_norms
            tmp_pod.pulsation = new_pulsation
            tmp_pod.ritz_vals = new_ritz_vals
        tmp_pod._ModalFields__spat_coh = None
        tmp_pod._ModalFields__temp_coh = None
        # return
        if not inplace:
            return tmp_pod

    def crop(self, intervx=None, intervy=None, intervt=None, ind=False,
             inplace=False):
        """
        Crop the POD modes and/or temporal evolutions, according to the given
        intervals.

        Parameters
        ----------
        intervx : 2x1 array of numbers
            Interval of x to keep.
        intervy : 2x1 array of numbers
            Interval of y to keep.
        intervt : 2x1 array of numbers
            Interval of time to keep.
        ind : boolean, optional
            If 'True', intervals are understood as indices along axis.
            If 'False' (default), intervals are understood in axis units.
        inplace : boolean, optional
            If 'True', the field is croped in place.
        """
        # check not necessary because done in 'VectorField.crop()'
        # get data
        if inplace:
            tmp_mf = self
        else:
            tmp_mf = self.copy()
        # temporal crop (first for efficiency)
        if intervt is not None:
            times = self.times
            # get time indices
            if not ind:
                intervt = np.array(intervt, dtype=float)
                if intervt[0] <= times[0]:
                    ind_1 = 0
                elif intervt[0] >= times[-1]:
                    raise ValueError()
                else:
                    ind_1 = np.where(times > intervt[0])[0][0]
                if intervt[1] >= times[-1]:
                    ind_2 = len(times)
                elif intervt[1] <= times[0]:
                    raise ValueError()
                else:
                    ind_2 = np.where(times > intervt[1])[0][0]
                intervt = [ind_1, ind_2]
            intervt = np.array(intervt, dtype=int)
            # crop
            tmp_mf.times = tmp_mf.times[intervt[0]:intervt[1]]
            tmp_mf._ModalFields__temp_coh = None
            for prof in tmp_mf.temp_evo:
                prof.crop(intervx=intervt, ind=True, inplace=True)
        # spatial crop
        if intervx is not None or intervy is not None:
            # crop Field and mean field
            super(ModalFields, tmp_mf).crop(intervx=intervx, intervy=intervy,
                                            full_output=False, ind=ind,
                                            inplace=True)
            tmp_mf.mean_field.crop(intervx=intervx, intervy=intervy, ind=ind,
                                   inplace=True)
            # loop on modes
            for mode in tmp_mf.modes:
                mode.crop(intervx=intervx, intervy=intervy,
                          ind=ind, inplace=True)
        # return
        if not inplace:
            return tmp_mf

    def scale(self, scalex=1., scaley=1., scalet=1., scalev=1.,
              inplace=False):
        """
        """
        #
        if inplace:
            tmp_mf = self
        else:
            tmp_mf = self.copy()
        # scale mean field
        tmp_mf.mean_field.scale(scalex=scalex, scaley=scaley, scalev=scalev,
                                inplace=True)
        # scale axe and unities
        tmp_mf.axe_x = tmp_mf.mean_field.axe_x
        tmp_mf.axe_y = tmp_mf.mean_field.axe_y
        tmp_mf.unit_x = tmp_mf.mean_field.unit_x
        tmp_mf.unit_y = tmp_mf.mean_field.unit_y
        tmp_mf.unit_values = tmp_mf.mean_field.unit_values
        # scale temporal evolutions
        for i in range(len(tmp_mf.temp_evo)):
            tmp_mf.temp_evo[i].scale(scalex=scalet, scaley=scalev,
                                     inplace=True)
        # scale times and unit
        tmp_mf.times = tmp_mf.temp_evo[0].x
        tmp_mf.unit_times = tmp_mf.temp_evo[0].unit_x
        # scale modes
        for i in range(len(tmp_mf.modes)):
            tmp_mf.modes[i].scale(scalex=scalex, scaley=scaley,
                                  inplace=True)
        # return
        if not inplace:
            return tmp_mf

    def smooth_temporal_evolutions(self, tos='uniform', size=None,
                                   inplace=True):
        """
        Smooth the temporal evolutions (to do before reconstructing)

        Parameters
        ----------
        tos : string, optional
            Type of smoothing, can be ‘uniform’ (default) or ‘gaussian’
            (See ndimage module documentation for more details)
        size : number, optional
            Size of the smoothing
            (is radius for ‘uniform’ and sigma for ‘gaussian’).
            Default is 3 for ‘uniform’ and 1 for ‘gaussian’.
        """
        if inplace:
            for n in np.arange(len(self.modes)):
                self.temp_evo[n].smooth(tos=tos, size=size, inplace=True)
        else:
            tmp_MF = self.copy()
            for n in np.arange(len(tmp_MF.modes)):
                tmp_MF.temp_evo[n].smooth(tos=tos, size=size, inplace=True)
            return tmp_MF

    def smooth_spatial_evolutions(self, tos='uniform', size=None, inplace=True):
        """
        Smooth the spatial evolutions (to do before reconstructing)

        Parameters
        ----------
        tos : string, optional
            Type of smoothing, can be ‘uniform’ (default) or ‘gaussian’
            (See ndimage module documentation for more details)
        size : number, optional
            Size of the smoothing
            (is radius for ‘uniform’ and sigma for ‘gaussian’).
            Default is 3 for ‘uniform’ and 1 for ‘gaussian’.
        """
        if inplace:
            for n in np.arange(len(self.modes)):
                self.modes[n].smooth(tos=tos, size=size, inplace=True)
        else:
            tmp_MF = self.copy()
            for n in np.arange(len(tmp_MF.modes)):
                tmp_MF.mdes[n].smooth(tos=tos, size=size, inplace=True)
            return tmp_MF

    def augment_temporal_resolution(self, fact=2, interp='linear',
                                    inplace=True, verbose=False):
        """
        Augment the temporal resolution (to augmente the temporal resolution
        of the reconstructed fields).

        Parameters
        ----------
        fact : integer
            Resolution augmentation needed (default is '2', for a result
            profile with twice more points)
        interp : string in ['linear', 'nearest', 'slinear', 'quadratic',
                            'cubic']
            Specifies the kind of interpolation as a string
            (Default is 'linear'). slinear', 'quadratic' and 'cubic' refer
            to a spline interpolation of first, second or third order.
        inplace : bool
            .
        verbose : bool
            .
        """
        # check parameters
        if not isinstance(fact, int):
            if int(fact) - fact == 0:
                fact = int(fact)
            else:
                raise TypeError()
        if fact <= 0:
            raise TypeError()
        if not isinstance(interp, STRINGTYPES):
            raise TypeError()
        if interp not in ['linear', 'nearest', 'slinear', 'quadratic',
                          'cubic']:
            raise ValueError()
        if not isinstance(inplace, bool):
            raise TypeError()
        # get data
        if inplace:
            tmp_pod = self
        else:
            tmp_pod = self.copy()
        # loop on the temporal evolutions
        for prof in tmp_pod.temp_evo:
            prof.augment_resolution(fact=fact, interp=interp, inplace=True)
        # change other little things
        tmp_pod.times = tmp_pod.temp_evo[0].x
        tmp_pod._ModalFields__spat_coh = None
        tmp_pod._ModalFields__temp_coh = None
        # return
        if not inplace:
            return tmp_pod

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
            if self.decomp_type in ['pod', 'bpod']:
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
        if self.decomp_type in ['pod', 'bpod']:
            if (times.max() > self.times.max() or
                    times.min() < self.times.min()):
                raise ValueError()
        # getting datas
        ind_times = np.arange(len(times))
        super_mask = self.modes[0].mask
        # reconstruction temporal evolution if needed
        if self.decomp_type in ['pod', 'bpod']:
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
                    coef = tmp_prof.get_interpolated_values(x=times[t])[0]
                    tmp_tf[t] += np.real(tmp_mode*coef)
            # returning
            TF = TemporalScalarFields()
            for t in ind_times:
                tmp_sf = ScalarField()
                tmp_sf.import_from_arrays(self.axe_x, self.axe_y, tmp_tf[t],
                                          mask=super_mask,
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
                    coef = tmp_prof.get_interpolated_values(x=times[t])[0]
                    tmp_tf_x[t] += np.real(tmp_mode_x*coef)
                    tmp_tf_y[t] += np.real(tmp_mode_y*coef)
            # returning
            TF = TemporalVectorFields()
            for t in ind_times:
                tmp_vf = VectorField()
                tmp_vf.import_from_arrays(self.axe_x, self.axe_y,
                                          tmp_tf_x[t], tmp_tf_y[t],
                                          mask=super_mask,
                                          unit_x=self.unit_x,
                                          unit_y=self.unit_y,
                                          unit_values=self.unit_values)
                TF.add_field(tmp_vf, time=times[t], unit_times=self.unit_times)
        return TF

    def get_modes_energy(self, cum=False, raw=False):
        """
        Return a profile whith the modes mean energy.

        Parameters
        ----------
        cum : boolean, optional
            If 'False' (default), return the modes energy
            If 'True', return the cumulative modes energy
        raw : bool, optional
            If 'False' (default), a Profile object is returned
            If 'True', an array is returned

        Returns
        -------
        modes_nrj : array or Profile object
            Energy for each modes.

        """
        modes_nrj = np.zeros((len(self.modes),))
        for n in np.arange(len(self.modes)):
            if isinstance(self.modes[0], ScalarField):
                magnitude = 1./2.*np.real(self.modes[n].values)**2
            elif isinstance(self.modes[0], VectorField):
                magnitude = 1./2.*(np.real(self.modes[n].comp_x)**2 +
                                   np.real(self.modes[n].comp_y)**2)
            coef_temp = np.mean(np.real(self.temp_evo[n].y)**2)
            modes_nrj[n] = np.mean(magnitude)*coef_temp
        # cum or not
        if cum:
            modes_nrj = np.cumsum(modes_nrj)
        # returning
        if raw:
            return modes_nrj
        else:
            modes_nrj = Profile(np.arange(len(self.modes)), modes_nrj,
                                mask=False,
                                unit_x="", unit_y=self.modes[0].unit_values**2,
                                name="")
        return modes_nrj

    def get_critical_kappa(self, Nx, Ny=None):
        """
        Return the critical value of kappa.
        A mode with a kappa value superior to the critical value have only
        .3% chance to be a random mode.
        """
        if Ny is None:
            a = 9.57253208
            b = 0.92940704
            c = 0.98559246
            kappa_crit = np.exp(a/Nx**b) - c
            return kappa_crit
        else:
            a, b, c = (22.885989, 1.552608, 1.996703)
            return np.exp(a/Nx**b) + np.exp(a/Ny**b) - c

    def get_temporal_coherence(self, raw=False):
        """
        Return a profile where each value represent the probability for a mode
        to be temporaly non-random (values above 0 have only 5% chance to be
        associated with random modes).
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
        if self._ModalFields__temp_coh is not None:
            var_spec = self._ModalFields__temp_coh
        else:
            # computing maximal variance
            max_std_spec = np.zeros(len(self.times))
            max_std_spec[0] = 1
            max_std_spec /= np.trapz(max_std_spec)
            max_std = np.std(max_std_spec)
            # computing critical kappa
            k_crit = self.get_critical_kappa(len(self.temp_evo[0].x))
            # getting spectrum variance on modes
            var_spec = np.empty((len(self.modes)))
            for n in np.arange(len(self.modes)):
                prof = self.temp_evo[n]
                spec = prof.get_spectrum(scaling='spectrum')**.5
                spec /= np.trapz(spec.y)
                var_spec[n] = np.std(spec.y)/max_std
                if var_spec[n] < 0:
                    var_spec[n] = 0
            # normalize with critical kappa
            var_spec = (var_spec - k_crit)/(1 - k_crit)
            # storing on object
            self._ModalFields__temp_coh = var_spec
        # returning
        if raw:
            return var_spec
        else:
            prof = Profile(np.arange(len(self.modes)), var_spec, unit_x='',
                           unit_y='', name="")
            return prof

    def get_spatial_coherence(self, raw=False):
        """
        Return a profile where each value represent the probability for a mode
        to be spatialy non-random (values from 0 to 1).
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
            Probability estimation for each mode of being coherent in space.

        Notes
        -----
        Returned values are, for each modes, the variance of the
        normalized two-dimensional spectrum of Vx and Vy.
        Variance is high when spectrum show predominant frequencies
        (coherent behavior), inversely, variance is low when spectrum is
        nearly uniform (random behavior).
        """
        if self._ModalFields__spat_coh is not None:
            var_spec = self._ModalFields__spat_coh
        else:
            # data
            shape = self.modes[0].shape
            axe_x = self.modes[0].axe_x
            axe_y = self.modes[0].axe_y
            center_x = np.int(len(axe_x)/2.)
            center_y = np.int(len(axe_y)/2.)

            def get_field_spec_std(data):
                data -= np.mean(data)
                spec = np.abs(np.real(np.fft.rfft2(data)))
                spec = np.fft.fftshift(spec)
                spec /= spint.simps(spint.simps(spec))
                return np.std(spec)
            # computing minimal variance
            min_std = 0.
            for i in np.arange(20):
                min_std_field = np.random.rand(*shape)*2. - 1.
                min_std += get_field_spec_std(min_std_field)
            min_std /= 20.
            # computing maximal variance
            max_std_spec = np.zeros(shape)
            max_std_spec[center_x:center_x + 2, center_y:center_y + 2] = 1.
            max_std_spec /= spint.simps(spint.simps(max_std_spec))
            max_std = np.std(max_std_spec)
#            # computing critical kappa
#            k_crit = self.get_critical_kappa(len(self.axe_x)*len(self.axe_y))
            # computing spectrum variation on each mode
            var_spec = np.empty((len(self.modes)))
            for n in np.arange(len(self.modes)):
                if isinstance(self.modes[n], ScalarField):
                    data = self.modes[n].values
                    var_spec[n] = get_field_spec_std(data)
                elif isinstance(self.modes[n], VectorField):
                    datax = self.modes[n].comp_x
                    datay = self.modes[n].comp_y
                    stdx = get_field_spec_std(datax)
                    stdy = get_field_spec_std(datay)
                    var_spec[n] = (stdx + stdy)/2.
                else:
                    raise Exception()
            # normalize with min and max kappa
            var_spec = (var_spec - min_std)/(max_std - min_std)
            # storing in object
            self._ModalFields__spat_coh = var_spec
        if raw:
            return var_spec
        else:
            prof = Profile(np.arange(len(self.modes)), var_spec, unit_x='',
                           unit_y='', name="")
            return prof

    def display(self):
        # modes
        fig1 = plt.figure()
        modes = self.modes_as_tf
        if hasattr(modes, "magnitude"):
            plot1 = modes.display('magnitude')
        else:
            plot1 = modes.display()
        plt.title('POD modes')
        # temporal evolutions
        fig2 = plt.figure()
        tmp_x = [prof.x for prof in self.temp_evo]
        tmp_y = [prof.y for prof in self.temp_evo]
        plot2 = pplt.Displayer(tmp_x, tmp_y, data_type='profile', color='k')
        bm = pplt.ButtonManager(plot2)
        plot2.button_manager.link_to_other_graph(plot1)
        plt.title('Modes temporal evolution')
        # temporal evolution spectrum
        fig3 = plt.figure()
        specs = [prof.get_spectrum() for prof in self.temp_evo]
        tmp_x = [prof.x[1::] for prof in specs]
        tmp_y = [prof.y[1::] for prof in specs]
        plot3 = pplt.Displayer(tmp_x, tmp_y, data_type='profile',
                               kind='loglog', color='k')
        bm = pplt.ButtonManager(plot3)
        plot3.button_manager.link_to_other_graph(plot2)
        plt.title('Modes temporal evolution spectra')
        # cum nrj
        fig4 = plt.figure()
        cum_nrj = self.get_modes_energy(cum=True)
        tmp_x = [[x] for x in cum_nrj.x]
        tmp_y = [[y] for y in cum_nrj.y]
        plot4 = pplt.Displayer(tmp_x, tmp_y, data_type='profile', kind='plot',
                               marker='o', mfc='k', ls='none')
        bm = pplt.ButtonManager(plot4)
        cum_nrj.display(color='k', ls='-', marker=None)
        plt.xlim(xmin=0)
        plt.ylim(0, 1)
        plot4.button_manager.link_to_other_graph(plot2)
        plt.title('Modes cumulative energy')
        return plot1, plot2, plot3, plot4

    def display_recap(self, figsize=(15, 10)):
        """
        Display some important diagram for the decomposition.
        """
        print("Obsolete !")
        if self.decomp_type in ['pod', 'bpod']:
            plt.figure(figsize=figsize)
            plt.subplot(2, 3, 1)
            p_vals = self.get_spatial_coherence()
            p_vals.display(color='k')
            plt.axhline(0, color='k', linestyle=":")
            plt.title('Coherence indicator')
            plt.xlabel('Modes')
            plt.ylabel('Coherence')
            plt.subplot(2, 3, 2)
            self.modes[0].display()
            plt.title("Mode 1")
            plt.subplot(2, 3, 4)
            tmp_prof = self.get_modes_energy(cum=True)
            modes_nrj = self.get_modes_energy(raw=True)
            tmp_prof.y /= np.sum(modes_nrj)
            tmp_prof.display(color='k')
            plt.title('Cumulative modes energy')
            plt.ylim(ymin=0, ymax=1)
            plt.subplot(2, 3, 5)
            self.temp_evo[0].display(color='k')
            plt.title("Temporal evolution of mode 1")
            plt.subplot(2, 3, 3)
            self.modes[1].display()
            plt.title("Mode 2")
            plt.subplot(2, 3, 6)
            self.temp_evo[1].display(color='k')
            plt.title("Temporal evolution of mode 2")
        elif self.decomp_type == 'dmd':
            self.pulsation.change_unit('y', 'rad/s')
            self.growth_rate.change_unit('y', '1/s')
            plt.figure(figsize=figsize)
            plt.subplot(2, 3, 1)
            plt.plot(np.real(self.ritz_vals.y), np.imag(self.ritz_vals.y),
                     'ko')
            plt.title("Ritz eigenvalues in the complexe plane")
            plt.xlabel("Real part of Ritz eigenvalue")
            plt.ylabel("Imaginary part of Ritz eigenvalue")
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.subplot(2, 3, 2)
            plt.plot(self.pulsation.y, self.growth_rate.y, 'ko')
            plt.title("Growth rate spectrum")
            plt.xlabel("Pulsation [rad/s]")
            plt.ylabel("Growth rate [1/s]")
            x_max = np.max([np.abs(plt.xlim()[0]), np.abs(plt.xlim()[1])])
            plt.xlim(-x_max, x_max)
            plt.subplot(2, 3, 3)
            sorted_omega = np.sort(self.pulsation.y)
            delta_omega = np.mean(np.abs(sorted_omega[1::] -
                                  sorted_omega[0:-1:]))
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
            tmp_prof = self.get_modes_energy()
            tmp_prof.y /= np.sum(tmp_prof.y)
            plt.plot(self.pulsation.y, tmp_prof.y, 'ko')
            plt.title('Modes energy')
            plt.ylabel("Normalized kinetic energy []")
            plt.xlabel("Pulsation [rad/s]")
            plt.ylim(ymin=0)
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


def _tsf_to_POD(tsf):
    # getting datas
    props = {}
    tsf = tsf.copy()
    tsf.crop_masked_border(inplace=True)
    tsf.fill(tof='spatial', kind='linear', inplace=True)
    ind_fields = np.arange(len(tsf.fields))
    props['ind_fields'] = ind_fields
    props['f_shape'] = tsf.fields[0].shape
    props['mean_field'] = tsf.get_mean_field()
    tsf = tsf.get_fluctuant_fields()
    props['super_mask'] = np.sum(tsf.mask, axis=0) == tsf.mask.shape[0]
    props['axe_x'], props['axe_y'] = tsf.axe_x, tsf.axe_y
    props['times'] = tsf.times
    props['unit_x'], props['unit_y'] = tsf.unit_x, tsf.unit_y
    props['unit_values'] = tsf.unit_values
    props['unit_times'] = tsf.unit_times
    # Link data
    values = [tsf.fields[t].values for t in ind_fields]
    del tsf
    snaps = [modred.VecHandleInMemory(values[t])
             for t in ind_fields]
    del values
    # return
    return snaps, props


def _tvf_to_POD(tvf):
    # getting datas
    props = {}
    tvf = tvf.copy()
    tvf.crop_masked_border(inplace=True)
    tvf.fill(tof='spatial', kind='linear', inplace=True)
    ind_fields = np.arange(len(tvf.fields))
    props['ind_fields'] = ind_fields
    props['f_shape'] = tvf.fields[0].shape
    props['mean_field'] = tvf.get_mean_field()
    tvf = tvf.get_fluctuant_fields()
    props['super_mask'] = np.sum(tvf.mask, axis=0) == tvf.mask.shape[0]
    props['axe_x'], props['axe_y'] = tvf.axe_x, tvf.axe_y
    props['times'] = tvf.times
    props['unit_x'], props['unit_y'] = tvf.unit_x, tvf.unit_y
    props['unit_values'] = tvf.unit_values
    props['unit_times'] = tvf.unit_times
    # Linking to snaps
    values = [[tvf.fields[t].comp_x, tvf.fields[t].comp_y]
              for t in ind_fields]
    del tvf
    snaps = [modred.VecHandleInMemory(np.transpose(values[i], (1, 2, 0)))
             for i in ind_fields]
    del values
    # return
    return snaps, props


def _POD_to_tsf(modes, props):
    modes_f = []
    for i in np.arange(len(modes)):
        tmp_field = ScalarField()
        tmp_field.import_from_arrays(props['axe_x'], props['axe_y'],
                                     modes[i].get(),
                                     mask=props['super_mask'],
                                     unit_x=props['unit_x'],
                                     unit_y=props['unit_y'],
                                     unit_values=props['unit_values'])
        modes[i] = 0
        modes_f.append(tmp_field)
    return modes_f


def _POD_to_tvf(modes, props):
    locals().update(props)
    modes_f = []
    for i in np.arange(len(modes)):
        tmp_field = VectorField()
        comp_x = modes[i].get()[:, :, 0]
        comp_y = modes[i].get()[:, :, 1]
        modes[i] = 0
        tmp_field.import_from_arrays(props['axe_x'], props['axe_y'],
                                     comp_x, comp_y,
                                     mask=props['super_mask'],
                                     unit_x=props['unit_x'],
                                     unit_y=props['unit_y'],
                                     unit_values=props['unit_values'])
        modes[i] = 0
        modes_f.append(tmp_field)
    return modes_f


def modal_decomposition(obj, kind='pod', obj2=None, wanted_modes='all',
                        max_vecs_per_node=1000, verbose=True):
    """
    Compute POD modes of the given fields using the snapshot method.

    Parameters
    ----------
    obj : TemporalFields, Profile or Points object
        Fields to extract modes from
    obj2 : same as obj
        Only used as second dataset for BPOD
    kind : string, optional
        Kind of decomposition, can be 'pod' (default), 'bpod' or 'dmd'.
    wanted_modes : string or number or array of numbers
        If 'all', extract all modes,
        If a number, extract first modes,
        If an array, extract the associated modes.
    max_vecs_per_node : integer, optional
        Number of fields that can be charged in memory.
        (More is faster but can lead to MemoryError)
    verbose : boolean, optional
        If 'True', display information.

    Returns
    -------
    modal_field : ModalField object
        .

    Notes
    -----
    You can use partially masked fields as input.
    If so, the asked values are lineary interpolated before doing the
    decomposition.
    """
    # check if modred is available
    if not MODRED:
        raise Exception("This feature need 'modred' to be installed")
    # Test parameters
    if not isinstance(obj, (TemporalFields)):
        raise TypeError()
    if kind == "bpod":
        if obj2 is None:
            raise ValueError()
        if not isinstance(obj2, obj.__class__):
            raise TypeError()
        if not obj2.shape == obj.shape:
            raise ValueError()
    if not isinstance(kind, STRINGTYPES):
        raise TypeError()
    if kind not in ['pod', 'bpod', 'dmd']:
        raise ValueError()
    if isinstance(wanted_modes, STRINGTYPES):
        if not wanted_modes == 'all':
            raise ValueError()
        wanted_modes = np.arange(len(obj.fields))
    elif isinstance(wanted_modes, NUMBERTYPES):
        wanted_modes = np.arange(wanted_modes)
    elif isinstance(wanted_modes, ARRAYTYPES):
        wanted_modes = np.array(wanted_modes)
        if wanted_modes.min() < 0 or wanted_modes.max() > len(obj.fields):
            raise ValueError()
    else:
        raise TypeError()
    try:
        max_vecs_per_node = int(max_vecs_per_node)
    except:
        raise TypeError()
    # getting datas
    if isinstance(obj, TemporalScalarFields):
        obj_type = 'TSF'
        snaps, props = _tsf_to_POD(obj)
        if kind == "bpod":
            snaps2, _ = _tsf_to_POD(obj2)
    elif isinstance(obj, TemporalVectorFields):
        obj_type = 'TVF'
        snaps, props = _tvf_to_POD(obj)
        if kind == "bpod":
            snaps2, _ = _tvf_to_POD(obj2)
    else:
        raise TypeError()
    globals().update(props)
    # Setting the decomposition mode
    eigvals = None
    eigvect = None
    ritz_vals = None
    mode_norms = None
    growth_rate = None
    pulsation = None
    if kind == 'pod':
        my_decomp = modred.PODHandles(np.vdot,
                                      max_vecs_per_node=max_vecs_per_node,
                                      verbosity=verbose)
        eigvals, eigvect = my_decomp.compute_decomp(snaps)
        del snaps
        wanted_modes = wanted_modes[wanted_modes < len(eigvals)]
        eigvect = np.array(eigvect)
        eigvect = eigvect[:, wanted_modes]
        eigvals = Profile(wanted_modes, eigvals[wanted_modes], mask=False,
                          unit_x=props['unit_times'], unit_y='')
    elif kind == 'bpod':
        my_decomp = modred.BPODHandles(np.vdot,
                                       max_vecs_per_node=max_vecs_per_node,
                                       verbosity=verbose)
        eigvect, eigvals, eigvect_l = my_decomp.compute_decomp(snaps, snaps2)
        del snaps
        del snaps2
        wanted_modes = wanted_modes[wanted_modes < len(eigvals)]
        eigvect = np.array(eigvect)
        eigvect = eigvect[:, wanted_modes]
        eigvals = Profile(wanted_modes, eigvals[wanted_modes], mask=False,
                          unit_x=props['unit_times'], unit_y='')
    elif kind == 'dmd':
        my_decomp = modred.DMDHandles(np.vdot,
                                      max_vecs_per_node=max_vecs_per_node,
                                      verbosity=verbose)
        ritz_vals, mode_norms, build_coeffs = my_decomp.compute_decomp(snaps)
        del snaps
        wanted_modes = wanted_modes[wanted_modes < len(ritz_vals)]
        # supplementary charac
        delta_t = props['times'][1] - props['times'][0]
        lambd_i = np.imag(ritz_vals)
        lambd_r = np.real(ritz_vals)
        lambd_mod = np.abs(ritz_vals)
        lambd_arg = np.zeros((len(ritz_vals)))
        mask = np.logical_and(lambd_i == 0, lambd_r <= 0)
        filt = np.logical_not(mask)
        lambd_arg[mask] = np.pi
        lambd_arg[filt] = 2*np.arctan(lambd_i[filt]/(lambd_r[filt] +
                                                     lambd_mod[filt]))
        sigma = np.log(lambd_mod)/delta_t
        omega = lambd_arg/delta_t
        # creating profiles
        unit_times = props['unit_times']
        ritz_vals = Profile(wanted_modes, ritz_vals[wanted_modes], mask=False,
                            unit_x=unit_times, unit_y='')
        mode_norms = Profile(wanted_modes, mode_norms[wanted_modes],
                             mask=False, unit_x=unit_times, unit_y='')
        growth_rate = Profile(wanted_modes, sigma[wanted_modes], mask=False,
                              unit_x=unit_times, unit_y=1./unit_times)
        pulsation = Profile(wanted_modes, omega[wanted_modes], mask=False,
                            unit_x=unit_times,
                            unit_y=make_unit('rad')/unit_times)
    else:
        raise ValueError("Unknown kind of decomposition : {}".format(kind))
    f_shape = props['f_shape']
    unit_values = props['unit_values']
    unit_times = props['unit_times']
    times = props['times']
    ind_fields = props['ind_fields']
    # Decomposing and getting modes
    if kind in ['pod', 'dmd']:
        modes = [modred.VecHandleInMemory(np.zeros(f_shape))
                 for i in np.arange(len(wanted_modes))]
        my_decomp.compute_modes(wanted_modes, modes)
    elif kind in ['bpod']:
        modes = [modred.VecHandleInMemory(np.zeros(f_shape))
                 for i in np.arange(len(wanted_modes))]
        adj_modes = [modred.VecHandleInMemory(np.zeros(f_shape))
                     for i in np.arange(len(wanted_modes))]
        my_decomp.compute_direct_modes(wanted_modes, modes)
        my_decomp.compute_adjoint_modes(wanted_modes, adj_modes)
    # Getting temporal evolution
    temporal_prof = []
    if kind in ['pod']:
        for n in np.arange(len(modes)):
            tmp_prof = eigvect[:, n]*eigvals.y[n]**.5
            tmp_prof = Profile(times, tmp_prof, mask=False,
                               unit_x=unit_times,
                               unit_y=unit_values)
            temporal_prof.append(tmp_prof)
    if kind in ['bpod']:
        for n in np.arange(len(modes)):
            tmp_prof = eigvect[:, n]*eigvals.y[n]**.5
            tmp_times = times[0:len(tmp_prof)]
            tmp_prof = Profile(tmp_times, tmp_prof, mask=False,
                               unit_x=unit_times,
                               unit_y=unit_values)
            temporal_prof.append(tmp_prof)
    elif kind == 'dmd':
        for n in np.arange(len(modes)):
            tmp_prof = np.real([ritz_vals.y[n]**(k) for k in ind_fields])
            tmp_prof = Profile(times, tmp_prof, mask=False, unit_x=unit_times,
                               unit_y=obj.unit_values)
            temporal_prof.append(tmp_prof)
    # Returning
    if obj_type == "TSF":
        modes_f = _POD_to_tsf(modes, props)
    elif obj_type == "TVF":
        modes_f = _POD_to_tvf(modes, props)
    del modes
    modal_field = ModalFields(kind, props['mean_field'], modes_f, wanted_modes,
                              temporal_prof,
                              eigvals=eigvals, eigvects=eigvect,
                              ritz_vals=ritz_vals, mode_norms=mode_norms,
                              growth_rate=growth_rate, pulsation=pulsation)
    return modal_field
