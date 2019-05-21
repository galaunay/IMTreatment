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
import scipy.interpolate as spinterp
from matplotlib import pyplot as plt

from . import scalarfield as sf, temporalfields as tf
from ..utils.types import ARRAYTYPES, INTEGERTYPES, NUMBERTYPES, STRINGTYPES
from IMTreatment.utils import ProgressCounter


class TemporalScalarFields(tf.TemporalFields):
    """
    Class representing a set of time-evolving scalar fields.
    """

    @property
    def values_as_sf(self):
        return self

    @property
    def values(self):
        dim = (len(self), self.shape[0], self.shape[1])
        values = np.empty(dim, dtype=float)
        for i, field in enumerate(self.fields):
            values[i, :, :] = field.values[:, :]
        return values

    def get_min_field(self, nmb_min=1):
        """
        Calculate the minimum scalar field, from all the fields.

        Parameters
        ----------
        nmb_min : integer, optional
            Minimum number of values used to take a minimum value.
            Else, the value is masked.
        """
        if len(self.fields) == 0:
            raise ValueError("There is no fields in this object")
        result_f = self.fields[0].copy()
        mask_cum = np.zeros(self.shape, dtype=int)
        mask_cum[np.logical_not(self.fields[0].mask)] += 1
        for field in self.fields[1::]:
            new_min_mask = np.logical_and(field.values < result_f.values,
                                          np.logical_not(field.mask))
            result_f.values[new_min_mask] = field.values[new_min_mask]
            mask_cum[np.logical_not(field.mask)] += 1
        mask = mask_cum <= nmb_min
        result_f.mask = mask
        return result_f

    def get_max_field(self, nmb_min=1):
        """
        Calculate the maximum scalar field, from all the fields.

        Parameters
        ----------
        nmb_min : integer, optional
            Minimum number of values used to take a maximum value.
            Else, the value is masked.
        """
        if len(self.fields) == 0:
            raise ValueError("There is no fields in this object")
        result_f = self.fields[0].copy()
        mask_cum = np.zeros(self.shape, dtype=int)
        mask_cum[np.logical_not(self.fields[0].mask)] += 1
        for field in self.fields[1::]:
            new_max_mask = np.logical_and(field.values > result_f.values,
                                          np.logical_not(field.mask))
            result_f.values[new_max_mask] = field.values[new_max_mask]
            mask_cum[np.logical_not(field.mask)] += 1
        mask = mask_cum <= nmb_min
        result_f.mask = mask
        return result_f

    def get_background(self):
        """
        Return the background image based on the histogram of values
        at each point
        """
        raise Exception("Not really working")
        if self.fields[0]._values_dtype != np.uint8:
            raise NotImplementedError()
        values = np.zeros(self.shape, dtype=np.uint8)
        for i, j in np.ndindex(self.shape):
            print(i, j)
            vals = []
            for n in range(len(self.fields)):
                vals.append(self.fields[n].values[i, j])
            hist, _ = np.histogram(vals, bins=254, range=[0, 254])
            val = np.argmax(hist)
            values[i, j] = val
        return values

    def get_phase_map(self, freq, tf=None, check_spec=None, verbose=True):
        """
        Return the phase map of the temporal scalar field for
        the given frequency.

        Parameters
        ----------
        freq: number
            Wanted frequency
        tf: Integer
            Last time indice to use
        check_spec: Integer
            If not None, specify the number of spectrum to display
            (useful to check if choosen frequencies are relevant).
        verbose: Boolean
            .

        Returns
        -------
        phase_map: ScalarField object
            .
        """
        #
        phases = np.zeros(self.shape, dtype=float)
        norms = np.zeros(self.shape, dtype=float)
        compo = "values"
        if tf is None:
            tf = len(self.fields)
        # select spectrum to display
        if check_spec is not None:
            check_spec_inds = np.random.choice(range(self.shape[0] *
                                                     self.shape[1]),
                                               check_spec)
        # get phases
        if verbose:
            pg = ProgressCounter(init_mess="Computing phase maps",
                                 nmb_max=self.shape[0]*self.shape[1],
                                 name_things="profiles")
        for i, x in enumerate(self.axe_x):
            for j, y in enumerate(self.axe_y):
                if verbose:
                    pg.print_progress()
                # get profile
                profile = self.get_time_profile(compo, (x, y),
                                                wanted_times=[0, tf])
                prof = profile.y
                dx = profile.x[1] - profile.x[0]
                # get fft
                fft = np.fft.fft(prof)
                fft = fft[0:int(len(fft)/2)]
                fft_norm = np.abs(fft)
                fft_phase = np.angle(fft)
                fft_f = np.fft.fftfreq(len(prof), dx)[0:len(fft)]
                # get phase at the wanted frequency
                ind = np.argmin(abs(fft_f - freq))
                # store phases and norms
                phases[i, j] = fft_phase[ind]
                norms[i, j] = fft_norm[ind]
                # display spectrum if asked
                if check_spec:
                    if i*len(self.axe_x)+j in check_spec_inds:
                        fig = plt.figure()
                        ax = fig.add_subplot(1, 1, 1)
                        ax.loglog(fft_f, fft_norm)
                        ax2 = ax.twinx()
                        ax2.plot([], [])
                        ax2.semilogx(fft_f, fft_phase)
                        plt.axvline(fft_f[ind], color="k", ls="--")
                        plt.show(block=True)
        # return
        norm = sf.ScalarField()
        norm.import_from_arrays(axe_x=self.axe_x,
                                axe_y=self.axe_y,
                                values=norms.transpose(),
                                unit_x=self.unit_x,
                                unit_y=self.unit_y,
                                unit_values="")
        phase = sf.ScalarField()
        phase.import_from_arrays(axe_x=self.axe_x,
                                 axe_y=self.axe_y,
                                 values=phases.transpose(),
                                 unit_x=self.unit_x,
                                 unit_y=self.unit_y,
                                 unit_values="rad")
        return norm, phase

    def get_spectral_filtering(self, fmin, fmax, order=2, inplace=False):
        """
        Perform a temporal spectral filtering

        Parameters
        ----------
        fmin, fmax : numbers
            Minimal and maximal frequencies
        order : integer, optional
            Butterworth filter order

        Returns
        -------
        filt_tsf : TemporalScalarFields
            Filtered temporal field
        """
        # prepare
        if inplace:
            tsf = self
        else:
            tsf = self.copy()
        # make spectral filtering on values
        ftsf = self._get_comp_spectral_filtering('values', fmin=fmin,
                                                 fmax=fmax, order=order)
        tsf.fields = ftsf.fields
        # return
        if not inplace:
            return tsf

    def fill(self, tof='spatial', kind='linear', value=0.,
             inplace=False, crop=False):
        """
        Fill the masked part of the array in place.

        Parameters
        ----------
        tof : string
            Can be 'temporal' for temporal interpolation, or 'spatial' for
            spatial interpolation.
        kind : string, optional
            Type of algorithm used to fill.
            'value' : fill with a given value
            'nearest' : fill with nearest available data
            'linear' : fill using linear interpolation
            'cubic' : fill using cubic interpolation
        value : 2x1 array
            Value for filling, '[Vx, Vy]' (only usefull with tof='value')
        inplace : boolean, optional
            .
        crop : boolean, optional
            If 'True', TVF borders are croped before filling.
        """
        # TODO : utiliser Profile.fill au lieu d'une nouvelle méthode de
        #        filling
        # checking parameters coherence
        if len(self.fields) < 3 and tof == 'temporal':
            raise ValueError("Not enough fields to fill with temporal"
                             " interpolation")
        if not isinstance(tof, STRINGTYPES):
            raise TypeError()
        if tof not in ['temporal', 'spatial']:
            raise ValueError()
        if not isinstance(kind, STRINGTYPES):
            raise TypeError()
        if kind not in ['value', 'nearest', 'linear', 'cubic']:
            raise ValueError()
        if crop:
            self.crop_masked_border(hard=False, inplace=True)
        # temporal interpolation
        if tof == 'temporal':
            # getting datas
            # getting super mask (0 where no value are masked and where all
            # values are masked)
            masks = self.mask
            sum_masks = np.sum(masks, axis=0)
            super_mask = np.logical_and(0 < sum_masks,
                                        sum_masks < len(self.fields) - 2)
            # loop on each field position
            for i, j in np.argwhere(super_mask):
                prof = self.get_time_profile('values', i, j, ind=True)
                # creating interpolation function
                if kind == 'value':
                    def interp(x):
                        return value
                elif kind == 'nearest':
                    raise Exception("Not implemented yet")
                elif kind == 'linear':
                    prof_filt = np.logical_not(prof.mask)
                    interp = spinterp.interp1d(prof.x[prof_filt],
                                               prof.y[prof_filt],
                                               kind='linear')

                elif kind == 'cubic':
                    prof_filt = np.logical_not(prof.mask)
                    interp = spinterp.interp1d(prof.x[prof_filt],
                                               prof.y[prof_filt],
                                               kind='cubic')
                else:
                    raise ValueError("Invalid value for 'kind'")
                # inplace or not
                fields = self.fields.copy()
                # loop on all profile masked points
                for ind_masked in prof.mask:
                    try:
                        interp_val = interp(prof.x[prof.mask])
                    except ValueError:
                        continue
                    # putting interpolated value in the field
                    fields[prof.mask].values[i, j] = interp_val
                    fields[prof.mask].mask[i, j] = False
        # spatial interpolation
        elif tof == 'spatial':
            if inplace:
                fields = self.fields
            else:
                tmp_tsf = self.copy()
                fields = tmp_tsf.fields
            for i, field in enumerate(fields):
                fields[i].fill(kind=kind, value=value, inplace=True)
        else:
            raise ValueError("Unknown parameter for 'tof' : {}".format(tof))
        # returning
        if inplace:
            self.fields = fields
        else:
            tmp_tsf = self.copy()
            tmp_tsf.fields = fields
            return tmp_tsf
