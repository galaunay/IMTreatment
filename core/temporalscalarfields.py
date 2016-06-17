#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
IMTreatment3 module

    Auteur : Gaby Launay
"""


# import warnings
# arnings.filterwarnings('error')
import numpy as np
import scipy.interpolate as spinterp
from ..utils.types import ARRAYTYPES, INTEGERTYPES, NUMBERTYPES, STRINGTYPES
from . import temporalfields as tf

class TemporalScalarFields(tf.TemporalFields):
    """
    Class representing a set of time-evolving scalar fields.

    Principal methods
    -----------------
    "add_field" : add a scalar field.

    "remove_field" : remove a field.

    "display" : display the scalar field, with these unities.

    "display_animate" : display an animation of a component of the velocity
    fields set.

    "calc_*" : give access to a bunch of derived statistical fields.
    """

    ### Attributes ###
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

    ### Watchers ###
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

    ### Modifiers ###
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
        # TODO : utiliser Profile.fill au lieu d'une nouvelle méthode de filling
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
