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
from ..utils.types import ARRAYTYPES, INTEGERTYPES, NUMBERTYPES, STRINGTYPES
from ..utils import make_unit
from . import scalarfield as sf
from . import profile as prof
from . import temporalscalarfields as tsf
from . import temporalfields as tf


class TemporalVectorFields(tf.TemporalFields):
    """
    Class representing a set of time-evolving velocity fields.
    """

    @property
    def Vx_as_sf(self):
        values = tsf.TemporalScalarFields()
        for i, field in enumerate(self.fields):
            values.add_field(field.comp_x_as_sf, time=self.times[i],
                             unit_times=self.unit_times)
        return values

    @property
    def Vx(self):
        dim = (len(self), self.shape[0], self.shape[1])
        values = np.empty(dim, dtype=float)
        for i, field in enumerate(self.fields):
            values[i, :, :] = field.comp_x[:, :]
        return values

    @property
    def Vy_as_sf(self):
        values = tsf.TemporalScalarFields()
        for i, field in enumerate(self.fields):
            values.add_field(field.comp_y_as_sf, time=self.times[i],
                             unit_times=self.unit_times)
        return values

    @property
    def Vy(self):
        dim = (len(self), self.shape[0], self.shape[1])
        values = np.empty(dim, dtype=float)
        for i, field in enumerate(self.fields):
            values[i, :, :] = field.comp_y[:, :]
        return values

    @property
    def magnitude_as_sf(self):
        values = tsf.TemporalScalarFields()
        for i, field in enumerate(self.fields):
            values.add_field(field.magnitude_as_sf, time=self.times[i],
                             unit_times=self.unit_times)
        return values

    @property
    def magnitude(self):
        dim = (len(self), self.shape[0], self.shape[1])
        values = np.empty(dim, dtype=float)
        for i, field in enumerate(self.fields):
            values[i, :, :] = field.magnitude[:, :]
        return values

    @property
    def theta_as_sf(self):
        values = tsf.TemporalScalarFields()
        for i, field in enumerate(self.fields):
            values.add_field(field.theta_as_sf, time=self.times[i],
                             unit_times=self.unit_times)
        return values

    @property
    def theta(self):
        dim = (len(self), self.shape[0], self.shape[1])
        values = np.empty(dim, dtype=float)
        for i, field in enumerate(self.fields):
            values[i, :, :] = field.theta[:, :]
        return values

    def get_time_auto_correlation(self):
        """
        Return auto correlation based on Vx and Vy.
        """
        Vx0 = self.fields[0].comp_x
        Vy0 = self.fields[0].comp_y
        norm = np.mean(Vx0*Vx0 + Vy0*Vy0)
        corr = np.zeros((len(self.times),))
        for i, time in enumerate(self.times):
            Vxi = self.fields[i].comp_x
            Vyi = self.fields[i].comp_y
            corr[i] = np.mean(Vx0*Vxi + Vy0*Vyi)/norm
        return prof.Profile(self.times, corr, mask=False,
                            unit_x=self.unit_times,
                            unit_y=make_unit(''))

    def get_mean_kinetic_energy(self):
        """
        Calculate the mean kinetic energy.
        """
        final_sf = sf.ScalarField()
        mean_vf = self.get_mean_field()
        values_x = mean_vf.comp_x_as_sf
        values_y = mean_vf.comp_y_as_sf
        final_sf = 1./2*(values_x**2 + values_y**2)
        return final_sf

    def get_tke(self):
        """
        Calculate the turbulent kinetic energy.
        """
        mean_field = self.get_mean_field()
        mean_x = mean_field.comp_x_as_sf
        mean_y = mean_field.comp_y_as_sf
        del mean_field
        tke = tsf.TemporalScalarFields()
        for i in np.arange(len(self.fields)):
            comp_x = self.fields[i].comp_x_as_sf - mean_x
            comp_y = self.fields[i].comp_y_as_sf - mean_y
            tke.add_field(1./2*(comp_x**2 + comp_y**2),
                          time=self.times[i],
                          unit_times=self.unit_times)
        return tke

    def get_turbulent_intensity(self):
        """
        Calculate the turbulent intensity.

        TI = sqrt(2/3*k)/sqrt(Vx**2 + Vy**2)
        """
        turb_int = (2./3.*self.get_tke())**(.5)/self.magnitude_as_sf
        return turb_int

    def get_mean_tke(self):
        tke = self.get_tke()
        mean_tke = tke.get_mean_field()
        return mean_tke

    def get_reynolds_stress(self, nmb_val_min=1):
        """
        Calculate the reynolds stress.

        Returns
        -------
        Re_xx, Re_yy, Re_xy : ScalarField objects
            Reynolds shear stress
        """
        # getting fluctuating velocities
        turb_vf = self.get_fluctuant_fields()
        u_p = turb_vf.Vx
        v_p = turb_vf.Vy
        mask = turb_vf.mask
        # rs_xx
        rs_xx = np.zeros(self.shape, dtype=float)
        rs_yy = np.zeros(self.shape, dtype=float)
        rs_xy = np.zeros(self.shape, dtype=float)
        mask_rs = np.zeros(self.shape, dtype=bool)
        # boucle sur les points du champ
        for i in np.arange(self.shape[0]):
            for j in np.arange(self.shape[1]):
                # boucle sur le nombre de champs
                nmb_val = 0
                for n in np.arange(len(turb_vf.fields)):
                    # check if masked
                    if not mask[n][i, j]:
                        rs_yy[i, j] += v_p[n][i, j]**2
                        rs_xx[i, j] += u_p[n][i, j]**2
                        rs_xy[i, j] += u_p[n][i, j]*v_p[n][i, j]
                        nmb_val += 1
                if nmb_val > nmb_val_min:
                    rs_xx[i, j] /= nmb_val
                    rs_yy[i, j] /= nmb_val
                    rs_xy[i, j] /= nmb_val
                else:
                    rs_xx[i, j] = 0
                    rs_yy[i, j] = 0
                    rs_xy[i, j] = 0
                    mask_rs[i, j] = True
        # masking and storing
        axe_x, axe_y = self.axe_x, self.axe_y
        unit_x, unit_y = self.unit_x, self.unit_y
        unit_values = self.unit_values
        rs_xx_sf = sf.ScalarField()
        rs_xx_sf.import_from_arrays(axe_x, axe_y, rs_xx, mask_rs,
                                    unit_x, unit_y, unit_values**2)
        rs_yy_sf = sf.ScalarField()
        rs_yy_sf.import_from_arrays(axe_x, axe_y, rs_yy, mask_rs,
                                    unit_x, unit_y, unit_values**2)
        rs_xy_sf = sf.ScalarField()
        rs_xy_sf.import_from_arrays(axe_x, axe_y, rs_xy, mask_rs,
                                    unit_x, unit_y, unit_values**2)
        return (rs_xx_sf, rs_yy_sf, rs_xy_sf)

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
        filt_tvf : TemporalVectorFields
            Filtered temporal field
        """
        # prepare
        if inplace:
            tvf = self
        else:
            tvf = self.copy()
        # make spectral filtering on Vx and Vy
        ftsfx = self._get_comp_spectral_filtering('Vx', fmin=fmin,
                                                  fmax=fmax, order=order)
        ftsfy = self._get_comp_spectral_filtering('Vy', fmin=fmin,
                                                  fmax=fmax, order=order)
        for i in range(len(self)):
            tvf.fields[i].comp_x = ftsfx.fields[i].values
            tvf.fields[i].comp_y = ftsfy.fields[i].values
        # return
        if not inplace:
            return tvf

    def fill(self, tof='spatial', kind='linear', value=[0., 0.],
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
        # TODO : utiliser prof.Profile.fill au lieu d'une nouvelle méthode de
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
        if isinstance(value, NUMBERTYPES):
            value = [value, value]
        elif not isinstance(value, ARRAYTYPES):
            raise TypeError()
        value = np.array(value)
        if crop:
            self.crop_masked_border(hard=False, inplace=True)
        # temporal interpolation
        if tof == 'temporal':
            # getting super mask (0 where no value are masked and where all
            # values are masked)
            masks = self.mask
            sum_masks = np.sum(masks, axis=0)
            super_mask = np.logical_and(0 < sum_masks,
                                        sum_masks < len(self.fields) - 2)
            # loop on each field position
            for i, j in np.argwhere(super_mask):
                # get time profiles
                prof_x = self.get_time_profile(component='comp_x', pt=[i, j],
                                               ind=True)
                prof_y = self.get_time_profile(component='comp_y', pt=[i, j],
                                               ind=True)
                # getting masked position on profile
                inds_masked_x = np.where(prof_x.mask)[0]
                inds_masked_y = np.where(prof_y.mask)[0]
                # creating interpolation function
                if kind == 'value':
                    def interp_x(x):
                        return value[0]
                    def interp_y(x):
                        return value[1]
                elif kind == 'nearest':
                    raise Exception("Not implemented yet")
                elif kind == 'linear':
                    prof_filt = np.logical_not(prof_x.mask)
                    interp_x = spinterp.interp1d(prof_x.x[prof_filt],
                                                 prof_x.y[prof_filt],
                                                 kind='linear')
                    prof_filt = np.logical_not(prof_y.mask)
                    interp_y = spinterp.interp1d(prof_y.x[prof_filt],
                                                 prof_y.y[prof_filt],
                                                 kind='linear')
                elif kind == 'cubic':
                    prof_filt = np.logical_not(prof_x.mask)
                    interp_x = spinterp.interp1d(prof_x.x[prof_filt],
                                                 prof_x.y[prof_filt],
                                                 kind='cubic')
                    prof_filt = np.logical_not(prof_y.mask)
                    interp_y = spinterp.interp1d(prof_y.x[prof_filt],
                                                 prof_y.y[prof_filt],
                                                 kind='cubic')
                else:
                    raise ValueError("Invalid value for 'kind'")
                # inplace or not
                fields = self.fields.copy()
                # loop on all x profile masked points
                for ind_masked in inds_masked_x:
                    try:
                        interp_val = interp_x(prof_x.x[ind_masked])
                    except ValueError:
                        continue
                    # putting interpolated value in the field
                    fields[ind_masked].comp_x[i, j] = interp_val
                    fields[ind_masked].mask[i, j] = False
                # loop on all y profile masked points
                for ind_masked in inds_masked_y:
                    try:
                        interp_val = interp_y(prof_y.x[ind_masked])
                    except ValueError:
                        continue
                    # putting interpolated value in the field
                    fields[ind_masked].comp_y[i, j] = interp_val
                    fields[ind_masked].mask[i, j] = False
        # spatial interpolation
        elif tof == 'spatial':
            if inplace:
                fields = self.fields
            else:
                tmp_tvf = self.copy()
                fields = tmp_tvf.fields
            for i, field in enumerate(fields):
                fields[i].fill(kind=kind, value=value, inplace=True)
        else:
            raise ValueError("Unknown parameter for 'tof' : {}".format(tof))
        # returning
        if inplace:
            self.fields = fields
        else:
            tmp_tvf = self.copy()
            tmp_tvf.fields = fields
            return tmp_tvf
