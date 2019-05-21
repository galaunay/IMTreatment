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

from ..core import ScalarField, VectorField, TemporalScalarFields,\
    TemporalVectorFields
from ..utils.types import ARRAYTYPES, NUMBERTYPES, STRINGTYPES
from ..utils import make_unit
from ..field_treatment import get_gradients
import numpy as np
from scipy import linalg


def get_angle_deviation(vectorfield, radius=None, ind=False, mask=None,
                        raw=False, local_treatment='none', order=1):
    """
    Return the angle deviation field.

    Parameters
    ----------
    vectorfield : VectorField object
        .
    radius : number, optionnal
        The radius used to choose the zone where to compute
        for each field oint. If not mentionned, a value is choosen in
        ordre to have about 8 points in the circle. It allow to get good
        result, without big computation cost.
    ind : boolean
        If 'True', radius is expressed on number of vectors.
        If 'False' (default), radius is expressed on axis unit.
    mask : array of boolean, optionnal
        Has to be an array of the same size of the vector field object,
        gamma will be compute only where mask is 'False'.
    raw : boolean, optional
        If 'False' (default), a ScalarField is returned,
        if 'True', an array is returned.
    local_treatment : string, optional
        If 'None' (default), angles are taken directly from the velocity field
        If 'galilean_inv', angles are taken from localy averaged velocity field
        if 'local', angles are taken from velocity fields where the velocity of
        the central point is localy substracted.
    order : number, optional
        Order used to compute the deviation
        (default 1 for sum of differences, 2 for standart deviation (std)
        or more)
    """
    # Checking parameters coherence
    if not isinstance(vectorfield, VectorField):
        raise TypeError("'vectorfield' must be a VectorField object")
    if radius is None:
        radius = 1.9
        ind = True
    if not isinstance(radius, NUMBERTYPES):
        raise TypeError("'radius' must be a number")
    if not isinstance(ind, bool):
        raise TypeError("'ind' must be a boolean")
    axe_x, axe_y = vectorfield.axe_x, vectorfield.axe_y
    if mask is None:
        mask = np.zeros(vectorfield.shape)
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'zone' must be an array of boolean")
    else:
        mask = np.array(mask)
    if not isinstance(raw, bool):
        raise TypeError()
    if not isinstance(local_treatment, STRINGTYPES):
        raise TypeError()
    if local_treatment not in ['none', 'galilean_inv', 'local']:
        raise ValueError()
    if not isinstance(order, NUMBERTYPES):
        raise TypeError()
    if order < 1:
        raise ValueError()
    # Getting data
    theta = vectorfield.theta
    mask, nmbpts, mask_dev, mask_border, mask_surr, motif =\
        _non_local_criterion_precomputation(vectorfield, mask, radius, ind,
                                            dev_pass=False)
    # Computing criterion
    # creating reference dispersion functions
    best_fun = np.array([2.*np.pi/nmbpts]*nmbpts)
    worse_fun = np.array([0]*(nmbpts - 1) + [2.*np.pi])
    worse_value = (np.sum(np.abs(worse_fun - best_fun)**order))**(1./order)
    # Loop on points
    deviation = np.zeros(vectorfield.shape)
    for inds, pos, _ in vectorfield:
        ind_x = inds[0]
        ind_y = inds[1]
        # stop if masked or on border or with a masked surrouinding point
        if mask[ind_x, ind_y] or mask_surr[ind_x, ind_y]\
                or mask_border[ind_x, ind_y]:
            continue
        # getting neighbour points
        indsaround = motif + inds
        # getting neighbour angles  (galilean inv or not)
        if local_treatment == 'none':
            angles = theta[indsaround[:, 0], indsaround[:, 1]]
        elif local_treatment == 'galilean_inv':
            tmp_Vx = vectorfield.comp_x[indsaround[:, 0], indsaround[:, 1]]
            tmp_Vy = vectorfield.comp_y[indsaround[:, 0], indsaround[:, 1]]
            tmp_Vx -= np.mean(tmp_Vx)
            tmp_Vy -= np.mean(tmp_Vy)
            angles = _get_angles(tmp_Vx, tmp_Vy)
        elif local_treatment == 'local':
            tmp_Vx = vectorfield.comp_x[indsaround[:, 0], indsaround[:, 1]]
            tmp_Vy = vectorfield.comp_y[indsaround[:, 0], indsaround[:, 1]]
            tmp_Vx -= vectorfield.comp_x[ind_x, ind_y]
            tmp_Vy -= vectorfield.comp_x[ind_x, ind_y]
            angles = _get_angles(tmp_Vx, tmp_Vy)
        # getting neightbour angles repartition
        angles = np.sort(angles)
        d_angles = np.empty(angles.shape)
        d_angles[0:-1] = angles[1::] - angles[:-1:]
        d_angles[-1] = angles[0] + 2*np.pi - angles[-1]
        # getting neighbour angles deviation
        deviation[ind_x, ind_y] = (1 - (np.sum(np.abs(d_angles - best_fun) **
                                               order)) **
                                   (1./order)/worse_value)
    # Applying masks
    mask = np.logical_or(mask, mask_border)
    mask = np.logical_or(mask, mask_surr)
    # Creating gamma ScalarField
    if raw:
        return np.ma.masked_array(deviation, mask)
    else:
        deviation_sf = ScalarField()
        unit_x, unit_y = vectorfield.unit_x, vectorfield.unit_y
        deviation_sf.import_from_arrays(axe_x, axe_y, deviation, mask,
                                        unit_x=unit_x, unit_y=unit_y,
                                        unit_values=make_unit(''))
        return deviation_sf


def get_gamma(vectorfield, radius=None, ind=False, kind='gamma1', mask=None,
              raw=False, dev_pass=False):
    """
    Return the gamma scalar field. Gamma criterion is used in
    vortex analysis.
    The fonction recognize if the field is ortogonal, and use an
    apropriate algorithm.

    Parameters
    ----------
    vectorfield : VectorField or TemporalVectorFields object
        .
    radius : number, optionnal
        The radius used to choose the zone where to compute
        gamma for each point. If not mentionned, a value is choosen in
        ordre to have about 8 points in the circle. It allow to get good
        result, without big computation cost.
    ind : boolean
        If 'True', radius is expressed on number of vectors.
        If 'False' (default), radius is expressed on axis unit.
    kind : string
        If 'gamma1' (default), compute gamma1 criterion.
        If 'gamma1b', compute gamma1 criterion with velocity corrector.
        (multiply with the mean velocity)
        If 'gamma2', compute gamma2 criterion (with relative velocities)
        If 'gamma2b', compute gamma2 criterion with a velocity corrector.
        (hide uniform velocity zone)
    mask : array of boolean, optionnal
        Has to be an array of the same size of the vector field object,
        gamma will be compute only where mask is 'False'.
    raw : boolean, optional
        If 'False' (default), a ScalarField is returned,
        if 'True', an array is returned.
    dev_pass : boolean, optional
        If 'True', the algorithm compute gamma criterion only where the
        velocity angles deviation is strong (faster if there is few points).
        Work only with 'gamma1'
    """
    # Checking parameters coherence
    if not isinstance(vectorfield, (VectorField, TemporalVectorFields)):
        raise TypeError()
    if radius is None:
        radius = 1.9
        ind = True
    if not isinstance(radius, NUMBERTYPES):
        raise TypeError("'radius' must be a number")
    if not isinstance(ind, bool):
        raise TypeError("'ind' must be a boolean")
    axe_x, axe_y = vectorfield.axe_x, vectorfield.axe_y
    if not isinstance(kind, STRINGTYPES):
        raise TypeError("'kind' must be a string")
    if kind not in ['gamma1', 'gamma2', 'gamma1b', 'gamma2b']:
        raise ValueError("Unkown value for kind")
    if mask is None:
        mask = np.zeros(vectorfield.shape)
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'zone' must be an array of boolean")
    else:
        mask = np.array(mask)
    if kind in ['gamma2', 'gamma2b']:
        dev_pass = False
    # if TemporalVectorFields
    if isinstance(vectorfield, TemporalVectorFields):
        gammas = TemporalScalarFields()
        for time, field in zip(vectorfield.times, vectorfield.fields):
            gamma = get_gamma(field, radius=radius, ind=ind, kind=kind,
                              mask=mask, raw=raw, dev_pass=dev_pass)
            gammas.add_field(gamma, time=time,
                             unit_times=vectorfield.unit_times)
        return gammas
    # getting data and masks
    Vx = vectorfield.comp_x
    Vy = vectorfield.comp_y
    norm_v = vectorfield.magnitude
    mask, nmbpts, mask_dev, mask_border, mask_surr, motif =\
        _non_local_criterion_precomputation(vectorfield, mask, radius, ind,
                                            dev_pass)
    # getting the vectors between center and neighbouring
    deltax = axe_x[1] - axe_x[0]
    deltay = axe_y[1] - axe_y[0]
    vector_a_x = np.zeros(motif.shape[0])
    vector_a_y = np.zeros(motif.shape[0])
    for i, indaround in enumerate(motif):
        vector_a_x[i] = indaround[0]*deltax
        vector_a_y[i] = indaround[1]*deltay
    norm_vect_a = (vector_a_x**2 + vector_a_y**2)**(.5)
    # Loop on points
    gammas = np.zeros(vectorfield.shape)
    for inds, pos, _ in vectorfield:
        ind_x = inds[0]
        ind_y = inds[1]
        # stop if masked or on border or with a masked surrouinding point
        if mask[ind_x, ind_y] or mask_surr[ind_x, ind_y]\
                or mask_border[ind_x, ind_y] or mask_dev[ind_x, ind_y]:
            continue
        # getting neighbour points
        indsaround = motif + inds
        # If necessary, compute mean velocity on points (gamma2)
        v_mean = [0., 0.]
        v_mean2 = [0., 0.]
        fact = 1
        if kind in ['gamma1b', 'gamma2', 'gamma2b']:
            v_mean[0] = np.mean(Vx[indsaround[:, 0], indsaround[:, 1]])
            v_mean[1] = np.mean(Vy[indsaround[:, 0], indsaround[:, 1]])
        if kind in ['gamma2b']:
            v_mean2[0] = np.mean((Vx[indsaround[:, 0], indsaround[:, 1]] -
                                 v_mean[0])**2)
            v_mean2[1] = np.mean((Vy[indsaround[:, 0], indsaround[:, 1]] -
                                 v_mean[1])**2)
            fact = np.sqrt(v_mean2[0] + v_mean2[1]) / \
                np.sqrt(v_mean[0]**2 + v_mean[1]**2)
            if np.abs(fact) > 1:
                fact = 1.
        # Loop on neighbouring points
        gamma = 0.
        for i, indaround in enumerate(indsaround):
            inda_x = indaround[0]
            inda_y = indaround[1]
            # getting vectors for scalar product
            if kind in ['gamma1', 'gamma1b']:
                vector_b_x = Vx[inda_x, inda_y]
                vector_b_y = Vy[inda_x, inda_y]
                denom = norm_v[inda_x, inda_y]*norm_vect_a[i]
                if denom != 0:
                    gamma += (vector_a_x[i]*vector_b_y -
                              vector_a_y[i]*vector_b_x)/denom
            elif kind in ['gamma2', 'gamma2b']:
                vector_b_x = Vx[inda_x, inda_y] - v_mean[0]
                vector_b_y = Vy[inda_x, inda_y] - v_mean[1]
                denom = (vector_b_x**2 + vector_b_y**2)**.5*norm_vect_a[i]
                if denom != 0:
                    gamma += (vector_a_x[i]*vector_b_y -
                              vector_a_y[i]*vector_b_x)/denom
        # adapting with factors
        if kind in ['gamma1', 'gamma2']:
            gamma = gamma/nmbpts
        elif kind == 'gamma1b':
            gamma = gamma/nmbpts*np.sqrt(v_mean[0]**2 + v_mean[1]**2)
        elif kind == 'gamma2b':
            gamma = gamma/nmbpts*fact
        # storing computed gamma value
        gammas[ind_x, ind_y] = gamma
    # Applying masks
    mask = np.logical_or(mask, mask_border)
    mask = np.logical_or(mask, mask_surr)
    # Creating gamma ScalarField
    if raw:
        return np.ma.masked_array(gammas, mask)
    else:
        gamma_sf = ScalarField()
        unit_x, unit_y = vectorfield.unit_x, vectorfield.unit_y
        gamma_sf.import_from_arrays(axe_x, axe_y, gammas, mask,
                                    unit_x=unit_x, unit_y=unit_y,
                                    unit_values=make_unit(''))
        return gamma_sf


def get_NL_residual_vorticity(vectorfield, radius=None, ind=False, mask=None,
                              raw=False):
    """
    Return the residual vorticity computed with non-local gradients.

    Parameters
    ----------
    vectorfield : VectorField object
        .
    radius : number, optionnal
        The radius used to choose the zone where to compute
        gamma for each point. If not mentionned, a value is choosen in
        ordre to have about 8 points in the circle. It allow to get good
        result, without big computation cost.
    ind : boolean
        If 'True', radius is expressed on number of vectors.
        If 'False' (default), radius is expressed on axis unit.
    mask : array of boolean, optionnal
        Has to be an array of the same size of the vector field object,
        gamma will be compute only where mask is 'False'.
    raw : boolean, optional
        If 'False' (default), a ScalarField is returned,
        if 'True', an array is returned.
    """
    # Checking parameters coherence
    if not isinstance(vectorfield, VectorField):
        raise TypeError("'vectorfield' must be a VectorField object")
    if radius is None:
        radius = 1.9
        ind = True
    if not isinstance(radius, NUMBERTYPES):
        raise TypeError("'radius' must be a number")
    if not isinstance(ind, bool):
        raise TypeError("'ind' must be a boolean")
    axe_x, axe_y = vectorfield.axe_x, vectorfield.axe_y
    if mask is None:
        mask = np.zeros(vectorfield.shape)
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'zone' must be an array of boolean")
    else:
        mask = np.array(mask)
    # getting data and masks
    Vx = vectorfield.comp_x
    Vy = vectorfield.comp_y
    mask, nmbpts, mask_dev, mask_border, mask_surr, motif =\
        _non_local_criterion_precomputation(vectorfield, mask, radius, ind,
                                            dev_pass=False)
    # Loop on points to get non-local gradients
    Exx = np.zeros(vectorfield.shape, dtype=float)
    Exy = np.zeros(vectorfield.shape, dtype=float)
    Eyx = np.zeros(vectorfield.shape, dtype=float)
    Eyy = np.zeros(vectorfield.shape, dtype=float)
    for inds, pos, _ in vectorfield:
        ind_x = inds[0]
        ind_y = inds[1]
        # stop if masked or on border or with a masked surrouinding point
        if mask[ind_x, ind_y] or mask_surr[ind_x, ind_y]\
                or mask_border[ind_x, ind_y] or mask_dev[ind_x, ind_y]:
            continue
        # getting neighbour points
        indsaround = motif + inds
        # non-local gradients computation by linear fitting
        ind_xs = indsaround[:, 0]
        ind_ys = indsaround[:, 1]
        Vxs = Vx[ind_xs, ind_ys]
        Vys = Vy[ind_xs, ind_ys]
        Exx[ind_x, ind_y], a_xx = np.polyfit(axe_x[ind_xs], Vxs, 1)
        Exy[ind_x, ind_y], a_xy = np.polyfit(axe_y[ind_ys], Vxs, 1)
        Eyx[ind_x, ind_y], _ = np.polyfit(axe_x[ind_xs], Vys, 1)
        Eyy[ind_x, ind_y], _ = np.polyfit(axe_y[ind_ys], Vys, 1)
    # getting principal rate of strain (s)
    s = np.sqrt(4*Exx**2 + (Exy + Eyx)**2)/2.
    # getting the vorticity-tensor component
    omega = (Eyx - Exy)/2.
    omega_abs = np.abs(omega)
    sign_omega = np.sign(omega)
    sign_omega[sign_omega == 0] = 1
    filt = s < omega_abs
    # getting the residual vorticity
    res_vort = np.zeros(vectorfield.shape)
    res_vort[filt] = sign_omega[filt]*(omega_abs[filt] - s[filt])
    # Applying masks
    mask = np.logical_or(mask, mask_border)
    mask = np.logical_or(mask, mask_surr)
    # Creating gamma ScalarField
    if raw:
        return np.ma.masked_array(res_vort, mask)
    else:
        gamma_sf = ScalarField()
        unit_values = vectorfield.unit_values/vectorfield.unit_x
        res_vort *= unit_values.asNumber()
        unit_values /= unit_values.asNumber()
        unit_x, unit_y = vectorfield.unit_x, vectorfield.unit_y
        gamma_sf.import_from_arrays(axe_x, axe_y, res_vort, mask,
                                    unit_x=unit_x, unit_y=unit_y,
                                    unit_values=unit_values)
        return gamma_sf


def get_kappa(vectorfield, radius=None, ind=False, kind='kappa1', mask=None,
              raw=False, dev_pass=False):
    """
    Return the kappa scalar field. Kappa criterion is used in
    vortex analysis.
    The fonction recognize if the field is ortogonal, and use an
    apropriate algorithm.

    Parameters
    ----------
    vectorfield : VectorField object
    radius : number, optionnal
        The radius used to choose the zone where to compute
        kappa for each point. If not mentionned, a value is choosen in
        ordre to have about 8 points in the circle. It allow to get good
        result, without big computation cost.
    ind : boolean
        If 'True', radius is expressed on number of vectors.
        If 'False' (default), radius is expressed on axis unit.
    kind : string
        If 'kappa1' (default), compute kappa1 criterion.
        If 'kappa2', compute kappa2 criterion (with relative velocities).
    mask : array of boolean, optionnal
        Has to be an array of the same size of the vector field object,
        kappa will be compute only where mask is 'False'.
    raw : boolean, optional
        If 'False' (default), a ScalarField is returned,
        if 'True', an array is returned.
    dev_pass : boolean, optional
        If 'True', the algorithm compute gamma criterion only where the
        velocity angles deviation is strong (faster if there is few points)
    """
    # Checking parameters coherence
    if not isinstance(vectorfield, VectorField):
        raise TypeError("'vectorfield' must be a VectorField object")
    if radius is None:
        radius = 1.9
        ind = True
    if not isinstance(radius, NUMBERTYPES):
        raise TypeError("'radius' must be a number")
    if not isinstance(ind, bool):
        raise TypeError("'ind' must be a boolean")
    axe_x, axe_y = vectorfield.axe_x, vectorfield.axe_y
    if not isinstance(kind, STRINGTYPES):
        raise TypeError("'kind' must be a string")
    if kind not in ['kappa1', 'kappa2']:
        raise ValueError("Unkown value for kind")
    if mask is None:
        mask = np.zeros(vectorfield.shape)
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'zone' must be an array of boolean")
    else:
        mask = np.array(mask)
    # getting data and masks
    Vx = vectorfield.comp_x
    Vy = vectorfield.comp_y
    norm_v = vectorfield.magnitude
    mask, nmbpts, mask_dev, mask_border, mask_surr, motif =\
        _non_local_criterion_precomputation(vectorfield, mask, radius, ind,
                                            dev_pass)
    # getting the vectors between center and neighbouring
    deltax = axe_x[1] - axe_x[0]
    deltay = axe_y[1] - axe_y[0]
    vector_a_x = np.zeros(motif.shape[0])
    vector_a_y = np.zeros(motif.shape[0])
    for i, indaround in enumerate(motif):
        vector_a_x[i] = indaround[0]*deltax
        vector_a_y[i] = indaround[1]*deltay
    norm_vect_a = (vector_a_x**2 + vector_a_y**2)**(.5)
    # Loop on points
    kappas = np.zeros(vectorfield.shape)
    for inds, pos, _ in vectorfield:
        ind_x = inds[0]
        ind_y = inds[1]
        # stop if masked or on border or with a masked surrouinding point
        if mask[ind_x, ind_y] or mask_surr[ind_x, ind_y]\
                or mask_border[ind_x, ind_y] or mask_dev[ind_x, ind_y]:
            continue
        # getting neighbour points
        indsaround = motif + np.array(inds)
        # If necessary, compute mean velocity on points (kappa2)
        v_mean = [0., 0.]
        if kind == 'kappa2':
            nmbpts = len(indsaround)
            for indaround in indsaround:
                v_mean[0] += Vx[ind_x, ind_y]
                v_mean[1] += Vy[ind_x, ind_y]
            v_mean[0] /= nmbpts
            v_mean[1] /= nmbpts
        # Loop on neighbouring points
        kappa = 0.
        nmbpts = len(indsaround)
        for i, indaround in enumerate(indsaround):
            inda_x = indaround[0]
            inda_y = indaround[1]
            # getting vectors for scalar product
            vector_b_x = Vx[inda_x, inda_y] - v_mean[0]
            vector_b_y = Vy[inda_x, inda_y] - v_mean[1]
            if kind == 'kappa1':
                denom = norm_v[inda_x, inda_y]*norm_vect_a[i]
            else:
                denom = (vector_b_x**2 + vector_b_y**2)**.5*norm_vect_a[i]
            # getting scalar product
            if denom != 0:
                kappa += (vector_a_x[i]*vector_b_x +
                          vector_a_y[i]*vector_b_y)/denom
        # storing computed kappa value
        kappas[ind_x, ind_y] = kappa/nmbpts
    # Applying masks
    mask = np.logical_or(mask, mask_border)
    mask = np.logical_or(mask, mask_surr)
    # Creating kappa ScalarField
    if raw:
        return np.ma.masked_array(kappas, mask)
    else:
        kappa_sf = ScalarField()
        unit_x, unit_y = vectorfield.unit_x, vectorfield.unit_y
        kappa_sf.import_from_arrays(axe_x, axe_y, kappas, mask,
                                    unit_x=unit_x, unit_y=unit_y,
                                    unit_values=make_unit(''))
        return kappa_sf


def get_iota(vectorfield, mask=None, radius=None, ind=False, raw=False):
    """
    Return the iota scalar field. iota criterion is used in
    vortex analysis.
    The fonction is only usable on orthogonal fields.
    Warning : This function is minimum at the saddle point center, and
    maximum around this point.

    Parameters
    ----------
    vectorfield : VectorField object
    mask : array of boolean, optionnal
        Has to be an array of the same size of the vector field object,
        iota2 will be compute only where zone is 'False'.
    radius : number, optionam
        If specified, the velocity field is smoothed with gaussian filter
        of the given radius before computing the vectors angles.
    ind : boolean, optional
        If 'True', radius is an indice number, if 'False', radius if in the
        field units (default).
    raw : boolean, optional
        If 'False' (default), a ScalarField is returned,
        if 'True', an array is returned.
    """
    if not isinstance(vectorfield, VectorField):
        raise TypeError("'vectorfield' must be a VectorField object")
    if mask is None:
        mask = np.zeros(vectorfield.shape)
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'mask' must be an array of boolean")
    else:
        mask = np.array(mask)
    axe_x, axe_y = vectorfield.axe_x, vectorfield.axe_y
    # smoothing if necessary and getting theta
    if radius is not None:
        if not ind:
            dx = axe_x[1] - axe_x[0]
            dy = axe_y[1] - axe_y[0]
            radius = radius/((dx + dy)/2.)
            ind = True
        tmp_vf = vectorfield.copy()
        tmp_vf.smooth(tos='gaussian', size=radius, inplace=True)
        theta = tmp_vf.theta
        mask = np.logical_or(mask, tmp_vf.mask)
    else:
        theta = vectorfield.theta
        mask = np.logical_or(mask, vectorfield.mask)
    # calcul du gradients de theta
    # necesary steps to avoid big gradients by passing from 0 to 2*pi
    theta1 = theta.copy()
    theta2 = theta.copy()
    theta2[theta2 > np.pi] -= 2*np.pi
    theta1_x, theta1_y = np.gradient(theta1)
    theta2_x, theta2_y = np.gradient(theta2)
    filtx = np.abs(theta1_x) > np.abs(theta2_x)
    filty = np.abs(theta1_y) > np.abs(theta2_y)
    theta_x = theta1_x.copy()
    theta_x[filtx] = theta2_x[filtx]
    theta_y = theta1_y.copy()
    theta_y[filty] = theta2_y[filty]
    iota = 1/2.*np.sqrt(theta_x**2 + theta_y**2)
    # getting mask
    maskf = np.logical_or(vectorfield.mask, np.isnan(iota))
    # returning
    if raw:
        return np.ma.masked_array(iota, maskf)
    else:
        iota_sf = ScalarField()
        unit_x, unit_y = vectorfield.unit_x, vectorfield.unit_y
        iota_sf.import_from_arrays(axe_x, axe_y, iota, maskf,
                                   unit_x=unit_x, unit_y=unit_y,
                                   unit_values=make_unit(''))
        return iota_sf


def get_enstrophy(vectorfield, radius=None, ind=False, mask=None,
                  raw=False):
    """
    Return the enstriphy field.

    Parameters
    ----------
    vectorfield : VectorField object
        .
    radius : number, optionnal
        The radius used to choose the zone where to integrate
        enstrophy for each point. If not mentionned, a value is choosen in
        ordre to have about 8 points in the circle.
    ind : boolean
        If 'True', radius is expressed on number of vectors.
        If 'False' (default), radius is expressed on axis unit.
    mask : array of boolean, optionnal
        Has to be an array of the same size of the vector field object,
        gamma will be compute only where mask is 'False'.
    raw : boolean, optional
        If 'False' (default), a ScalarField is returned,
        if 'True', an array is returned.
    """
    # Checking parameters coherence
    if not isinstance(vectorfield, VectorField):
        raise TypeError("'vectorfield' must be a VectorField object")
    if radius is None:
        radius = 1.9
        ind = True
    if not isinstance(radius, NUMBERTYPES):
        raise TypeError("'radius' must be a number")
    if not isinstance(ind, bool):
        raise TypeError("'ind' must be a boolean")
    axe_x, axe_y = vectorfield.axe_x, vectorfield.axe_y
    if mask is None:
        mask = np.zeros(vectorfield.shape)
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'mask' must be an array of boolean")
    else:
        mask = np.array(mask)
    # getting data and masks
    vort2 = get_vorticity(vectorfield, raw=False)**2
    unit_values = vort2.unit_values*vectorfield.unit_x*vectorfield.unit_y
    vort2 = vort2.values
    mask, nmbpts, mask_dev, mask_border, mask_surr, motif =\
        _non_local_criterion_precomputation(vectorfield, mask, radius, ind,
                                            dev_pass=False)
    dv = ((vectorfield.axe_x[1] - vectorfield.axe_x[0]) *
          (vectorfield.axe_y[1] - vectorfield.axe_y[0]))
    # Loop on points
    enstrophy = np.zeros(vectorfield.shape)
    for inds, pos, _ in vectorfield:
        ind_x = inds[0]
        ind_y = inds[1]
        # stop if masked or on border or with a masked surrouinding point
        if mask[ind_x, ind_y] or mask_surr[ind_x, ind_y]\
                or mask_border[ind_x, ind_y]:
            continue
        # getting neighbour points
        indsaround = motif + inds
        # Loop on neighbouring points
        loc_enstrophy = 0.
        for i, indaround in enumerate(indsaround):
            loc_enstrophy += vort2[indaround[0], indaround[1]]
        # storing computed gamma value
        enstrophy[ind_x, ind_y] = loc_enstrophy*dv
    # Applying masks
    mask = np.logical_or(mask, mask_border)
    mask = np.logical_or(mask, mask_surr)
    # Creating gamma ScalarField
    if raw:
        return np.ma.masked_array(enstrophy, mask)
    else:
        enstrophy_sf = ScalarField()
        unit_x, unit_y = vectorfield.unit_x, vectorfield.unit_y
        scale = unit_values.asNumber()
        enstrophy *= scale
        unit_values = unit_values/scale
        enstrophy_sf.import_from_arrays(axe_x, axe_y, enstrophy, mask,
                                        unit_x=unit_x, unit_y=unit_y,
                                        unit_values=unit_values)
        return enstrophy_sf


def _non_local_criterion_precomputation(vectorfield, mask, radius, ind,
                                        dev_pass):
    """
    """
    # Importing data from vectorfield (velocity, axis and mask)
    mask = np.logical_or(mask, vectorfield.mask)
    # Compute motif and motif angles on an arbitrary point
    axe_x, axe_y = vectorfield.axe_x, vectorfield.axe_y
    indcentral = [int(len(axe_x)/2.), int(len(axe_y)/2.)]
    if ind:
        motif = vectorfield.get_points_around(indcentral, radius, ind)
        motif = motif - indcentral
        motif = np.delete(motif, int(len(motif)/2), axis=0)
    else:
        ptcentral = [axe_x[indcentral[0]], axe_y[indcentral[1]]]
        motif = vectorfield.get_points_around(ptcentral, radius, ind)
        motif = motif - indcentral
    nmbpts = len(motif)
    # Generating masks
    # creating surrounding masked point zone mask
    mask_surr = np.zeros(mask.shape)
    inds_masked = np.transpose(np.where(mask))
    for ind_masked in inds_masked:
        for i, j in motif + ind_masked:
            # continue if outside the field
            if i < 0 or j < 0 or i >= mask_surr.shape[0]\
                    or j >= mask_surr.shape[1]:
                continue
            mask_surr[i, j] = True
    # creating near-border zone mask
    if ind:
        indx = np.arange(len(axe_x))
        indy = np.arange(len(axe_y))
        border_x = np.logical_or(indx <= indx[0] + (int(radius) - 1),
                                 indx >= indx[-1] - (int(radius) - 1))
        border_y = np.logical_or(indy <= indy[0] + (int(radius) - 1),
                                 indy >= indy[-1] - (int(radius) - 1))
        border_x, border_y = np.meshgrid(border_x, border_y)
        mask_border = np.transpose(np.logical_or(border_x, border_y))
    else:
        delta = (axe_x[1] - axe_x[0] + axe_y[1] - axe_y[0])/2
        border_x = np.logical_or(axe_x <= axe_x[0] + (radius - delta),
                                 axe_x >= axe_x[-1] - (radius - delta))
        border_y = np.logical_or(axe_y <= axe_y[0] + (radius - delta),
                                 axe_y >= axe_y[-1] - (radius - delta))
        border_x, border_y = np.meshgrid(border_x, border_y)
        mask_border = np.transpose(np.logical_or(border_x, border_y))
    # creating dev mask
    mask_dev = np.zeros(vectorfield.shape)
    if dev_pass:
        dev = get_angle_deviation(vectorfield, radius=radius, ind=ind,
                                  raw=True)
        mask_dev = dev < 0.1
    # returning
    return (mask, nmbpts, mask_dev, mask_border, mask_surr,
            motif)


def _get_angles(Vx, Vy, check=False):
    """
    Return the angles from velocities vectors.
    """
    if check:
        # check parameters
        if not isinstance(Vx, ARRAYTYPES):
            raise TypeError()
        Vx = np.array(Vx)
        if not Vx.ndim == 1:
            raise ValueError()
        if not isinstance(Vy, ARRAYTYPES):
            raise TypeError()
        Vy = np.array(Vy)
        if not Vy.ndim == 1:
            raise ValueError()
        if not Vx.shape == Vy.shape:
            raise ValueError()
    # get data
    norm = (Vx**2 + Vy**2)**(.5)
    # getting angle
    theta = np.arccos(Vx/norm)
    theta[Vy < 0] = 2*np.pi - theta[Vy < 0]
    return theta


def get_divergence(vf, raw=False):
    """
    Return a scalar field with the 2D divergence.

    Parameters
    ----------
    vf : VectorField or TemporalVectorfields
        Field(s) on which compute shear stress
    raw : boolean, optional
        If 'True', return an arrays,
        if 'False' (default), return a ScalarField object.

    Returns
    -------
    div : ScalarField or TemporalScalarFields
        Divergence field(s)
    """
    if isinstance(vf, VectorField):
        tmp_vf = vf.fill(inplace=False)
        axe_x, axe_y = tmp_vf.axe_x, tmp_vf.axe_y
        comp_x, comp_y = tmp_vf.comp_x, tmp_vf.comp_y
        mask = tmp_vf.mask
        dx = axe_x[1] - axe_x[0]
        dy = axe_y[1] - axe_y[0]
        Exx, _ = np.gradient(comp_x, dx, dy)
        _, Eyy = np.gradient(comp_y, dx, dy)
        div = Exx + Eyy
        if raw:
            return div
        else:
            unit_x, unit_y = tmp_vf.unit_x, tmp_vf.unit_y
            unit_values = vf.unit_values/vf.unit_x
            div *= unit_values.asNumber()
            unit_values /= unit_values.asNumber()
            div_sf = ScalarField()
            div_sf.import_from_arrays(axe_x, axe_y, div, mask=mask,
                                      unit_x=unit_x, unit_y=unit_y,
                                      unit_values=unit_values)
            return div_sf
    elif isinstance(vf, TemporalVectorFields):
        if raw:
            div_tsf = np.empty((len(vf.fields), vf.shape[0], vf.shape[1]),
                               dtype=float)
            for i, field in enumerate(vf.fields):
                div_tsf[i] = get_divergence(field, raw=True)
        else:
            div_tsf = TemporalScalarFields()
            for i, field in enumerate(vf.fields):
                tmp_div = get_divergence(field, raw=False)
                div_tsf.add_field(tmp_div, time=vf.times[i],
                                  unit_times=vf.unit_times)
        return div_tsf
    else:
        raise TypeError()


def get_vorticity(vf, raw=False):
    """
    Return a scalar field with the z component of the vorticity.

    Parameters
    ----------
    vf : VectorField or TemporalVectorfields
        Field(s) on which compute shear stress
    raw : boolean, optional
        If 'True', return an arrays,
        if 'False' (default), return a ScalarField object.

    Returns
    -------
    vort : ScalarField or TemporalScalarFields
        Vorticity field(s)
    """
    if isinstance(vf, VectorField):
        tmp_vf = vf.fill(inplace=False)
        axe_x, axe_y = tmp_vf.axe_x, tmp_vf.axe_y
        comp_x, comp_y = tmp_vf.comp_x, tmp_vf.comp_y
        mask = tmp_vf.mask
        dx = axe_x[1] - axe_x[0]
        dy = axe_y[1] - axe_y[0]
        _, Exy = np.gradient(comp_x, dx, dy)
        Eyx, _ = np.gradient(comp_y, dx, dy)
        vort = (Eyx - Exy)/2.
        if raw:
            return vort
        else:
            unit_x, unit_y = tmp_vf.unit_x, tmp_vf.unit_y
            unit_values = vf.unit_values/vf.unit_x
            vort *= unit_values.asNumber()
            unit_values /= unit_values.asNumber()
            vort_sf = ScalarField()
            vort_sf.import_from_arrays(axe_x, axe_y, vort, mask=mask,
                                       unit_x=unit_x, unit_y=unit_y,
                                       unit_values=unit_values)
            return vort_sf
    elif isinstance(vf, TemporalVectorFields):
        if raw:
            vort_tsf = np.empty((len(vf.fields), vf.shape[0], vf.shape[1]),
                                dtype=float)
            for i, field in enumerate(vf.fields):
                vort_tsf[i] = get_vorticity(field, raw=True)
        else:
            vort_tsf = TemporalScalarFields()
            for i, field in enumerate(vf.fields):
                tmp_vort = get_vorticity(field, raw=False)
                vort_tsf.add_field(tmp_vort, time=vf.times[i],
                                   unit_times=vf.unit_times)
        return vort_tsf
    else:
        raise TypeError()


def get_stokes_vorticity(vf, window_size=2, raw=False):
    """
    Return a scalar field with the z component of the vorticity using
    Stokes' theorem.

    Parameters
    ----------
    vf : VectorField or Velocityfield
        Field on which compute shear stress
    window_size : integer, optional
        Window size for stokes approximation of the vorticity.
    raw : boolean, optional
        If 'True', return an arrays,
        if 'False' (default), return a ScalarField object.

    Notes
    -----
    Seal et al., “Quantitative characteristics of a laminar,
    unsteady necklace vortex system at a rectangular block-flat plate
    juncture,” Journal of Fluid Mechanics, vol. 286, pp. 117–135, 1995.

    """
    # getting data
    axe_x, axe_y = vf.axe_x, vf.axe_y
    dx = axe_x[1] - axe_x[0]
    dy = axe_y[1] - axe_y[0]
    Vx = vf.comp_x
    Vy = vf.comp_y
    mask = vf.mask
    # creating new axis
    new_axe_x = np.arange(np.mean(axe_x[0:window_size]),
                          np.mean(axe_x[-window_size::] + dx*.9),
                          dx)
    new_axe_y = np.arange(np.mean(axe_y[0:window_size]),
                          np.mean(axe_y[-window_size::] + dy*.9),
                          dy)
    # Loop on field
    vort = np.zeros((len(new_axe_x), len(new_axe_y)))
    new_mask = np.zeros((len(new_axe_x), len(new_axe_y)), dtype=bool)
    for i in np.arange(len(axe_x) - window_size + 1):
        for j in np.arange(len(axe_y) - window_size + 1):
            # reinitialazing
            tmp_vort = 0.
            # checking masked values
            if np.any(mask[i:i + window_size, j:j + window_size]):
                new_mask[i, j] = True
                continue
            # summing over first border
            bord_vec = -Vy[i, j:j + window_size].copy()
            tmp_vort += np.trapz(bord_vec, dx=dy)
            # summing over second border
            bord_vec = Vy[i + window_size - 1, j:j + window_size].copy()
            tmp_vort += np.trapz(bord_vec, dx=dy)
            # summing over third border
            bord_vec = Vx[i:i + window_size, j].copy()
            tmp_vort += np.trapz(bord_vec, dx=dx)
            # summing over fourth border
            bord_vec = -Vx[i:i + window_size, j + window_size - 1].copy()
            tmp_vort += np.trapz(bord_vec, dx=dx)
            # adding coefficients
            tmp_vort *= 1./(dx*dy*window_size**2)
            # storing
            vort[i, j] = tmp_vort
    # returning
    if raw:
        return vort
    else:
        unit_values = vf.unit_values/vf.unit_x
        vort *= unit_values.asNumber()
        unit_values /= unit_values.asNumber()
        vort_sf = ScalarField()
        vort_sf.import_from_arrays(new_axe_x, new_axe_y, vort, mask=new_mask,
                                   unit_x=vf.unit_x, unit_y=vf.unit_y,
                                   unit_values=unit_values)
        return vort_sf


def get_swirling_strength(vf, raw=False):
    """
    Return a scalar field with the swirling strength
    (imaginary part of the eigenvalue of the velocity Jacobian)

    Parameters
    ----------
    vf : VectorField or Velocityfield
        Field on which compute shear stress
    raw : boolean, optional
        If 'True', return an arrays,
        if 'False' (default), return a ScalarField object.

    Notes
    -----
    Zhou, J., R. J. Adrian, S. Balachandar, et T. M. Kendall.
    « Mechanisms for generating coherent packets of hairpin vortices in
    channel flow ». Journal of Fluid Mechanics 387 (mai 1999): 353‑96.

    """
    if not isinstance(vf, VectorField):
        raise TypeError()
    tmp_vf = vf.copy()
    tmp_vf.fill()
    # Getting gradients and axes
    axe_x, axe_y = tmp_vf.axe_x, tmp_vf.axe_y
    mask = tmp_vf.mask
    du_dx, du_dy, dv_dx, dv_dy = get_gradients(vf, raw=True)
    # swirling stregnth matrix
    swst = np.zeros(tmp_vf.shape)
    # loop on  points
    for i in np.arange(len(axe_x)):
        for j in np.arange(len(axe_y)):
            if not mask[i, j]:
                lapl = [[du_dx[i, j], du_dy[i, j]],
                        [dv_dx[i, j], dv_dy[i, j]]]
                eigvals = np.linalg.eigvals(lapl)
                swst[i, j] = np.max(np.imag(eigvals))
    mask = np.logical_or(mask, np.isnan(swst))
    # creating ScalarField object
    if raw:
        return swst
    else:
        unit_x, unit_y = tmp_vf.unit_x, tmp_vf.unit_y
        # TODO: Implement unities
        unit_values = ""
        tmp_sf = ScalarField()
        tmp_sf.import_from_arrays(axe_x, axe_y, swst, mask=mask,
                                  unit_x=unit_x, unit_y=unit_y,
                                  unit_values=unit_values)
        return tmp_sf


def get_improved_swirling_strength(vf, raw=False):
    """
    Return a scalar field with the improved swirling strength

    Parameters
    ----------
    vf : VectorField or Velocityfield
        Field on which compute shear stress
    raw : boolean, optional
        If 'True', return an arrays,
        if 'False' (default), return a ScalarField object.

    Notes
    -----
    Chakraborty, Pinaki, S. Balachandar, et Ronald J. Adrian.
    « On the Relationships between Local Vortex Identification Schemes ».
    Journal of Fluid Mechanics 535 (5 juillet 2005): 189‑214.

    """
    if not isinstance(vf, VectorField):
        raise TypeError()
    tmp_vf = vf.copy()
    tmp_vf.fill()
    # Getting gradients and axes
    axe_x, axe_y = tmp_vf.axe_x, tmp_vf.axe_y
    comp_x, comp_y = tmp_vf.comp_x, tmp_vf.comp_y
    mask = tmp_vf.mask
    dx = axe_x[1] - axe_x[0]
    dy = axe_y[1] - axe_y[0]
    du_dx, du_dy = np.gradient(comp_x, dx, dy)
    dv_dx, dv_dy = np.gradient(comp_y, dx, dy)
    # swirling stregnth matrix
    swst = np.zeros(tmp_vf.shape)
    # loop on  points
    for i in np.arange(len(axe_x)):
        for j in np.arange(len(axe_y)):
            if not mask[i, j]:
                lapl = [[du_dx[i, j], du_dy[i, j]],
                        [dv_dx[i, j], dv_dy[i, j]]]
                eigvals = np.linalg.eigvals(lapl)
                lambcr = np.real(eigvals[0])
                lambci = np.abs(np.imag(eigvals[0]))
                if lambci == 0:
                    mask[i, j] = True
                swst[i, j] = lambcr/lambci
    mask = np.logical_or(mask, np.isnan(swst))
    # creating ScalarField object
    if raw:
        return swst
    else:
        unit_x, unit_y = tmp_vf.unit_x, tmp_vf.unit_y
        # TODO: Implement unities
        unit_values = ""
        tmp_sf = ScalarField()
        tmp_sf.import_from_arrays(axe_x, axe_y, swst, mask=mask,
                                  unit_x=unit_x, unit_y=unit_y,
                                  unit_values=unit_values)
        return tmp_sf


def get_q_criterion(vectorfield, mask=None, raw=False):
    """
    Return the scalar field of the 2D Q criterion .
    Define as "1/2*(R**2 - S**2)" , with "R" the deformation tensor,
    norm and "S" the rate of rotation tensor norm.

    Parameters
    ----------
    vectorfield : VectorField object
    mask : array of boolean, optional
        Has to be an array of the same size of the vector field object,
        Q criterion will be compute only where zone is 'False'.
    raw : boolean, optional
        If 'False' (default), a ScalarField is returned,
        if 'True', an array is returned.
    """
    if not isinstance(vectorfield, VectorField):
        raise TypeError("'vectorfield' must be a VectorField object")
    if mask is None:
        mask = np.zeros(vectorfield.shape)
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'mask' must be an array of boolean")
    else:
        mask = np.array(mask)
    axe_x, axe_y = vectorfield.axe_x, vectorfield.axe_y
    # calcul des gradients
    Exx, Exy, Eyx, Eyy = get_gradients(vectorfield, raw=True)
    # calcul de Qcrit
    norm_rot2 = 1/2.*(Exy - Eyx)**2
    norm_shear2 = (Exx**2 + 1./2.*(Exy + Eyx)**2 + Eyy**2)
    qcrit = .5*(norm_rot2) - norm_shear2
    unit_values = (vectorfield.unit_values/vectorfield.unit_x)**2
    scale = unit_values.asNumber()
    qcrit *= scale
    unit_values = unit_values/scale
    if raw:
        return np.ma.masked_array(qcrit, mask)
    else:
        q_sf = ScalarField()
        q_sf.import_from_arrays(axe_x, axe_y, qcrit, mask,
                                unit_x=vectorfield.unit_x,
                                unit_y=vectorfield.unit_y,
                                unit_values=unit_values)
        return q_sf


def get_Nk_criterion(vectorfield, mask=None, raw=False):
    """
    Return the scalar field of the 2D Nk criterion .
    Define as "||Omega||/||S||" , with "||Omega||" the rotation rate tensor
    norm and ||S|| the shear rate tensor norm.

    Parameters
    ----------
    vectorfield : VectorField object
    mask : array of boolean, optional
        Has to be an array of the same size of the vector field object,
        Nk criterion will be compute only where zone is 'False'.
    raw : boolean, optional
        If 'False' (default), a ScalarField is returned,
        if 'True', an array is returned.

    Notes
    -----
    See J. Jeong and F. Hussain, “On the identification of a vortex,” Journal
    of Fluid Mechanics, vol. 285, pp. 69–94, 1995.

    """
    if not isinstance(vectorfield, VectorField):
        raise TypeError("'vectorfield' must be a VectorField object")
    if mask is None:
        mask = np.zeros(vectorfield.shape)
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'mask' must be an array of boolean")
    else:
        mask = np.array(mask)
    axe_x, axe_y = vectorfield.axe_x, vectorfield.axe_y
    # calcul des gradients
    Exx, Exy, Eyx, Eyy = get_gradients(vectorfield, raw=True)
    # calcul de Nk
    norm_rot = 1./2.**.5*np.abs(Exy - Eyx)
    norm_shear = (Exx**2 + 1./2.*(Exy + Eyx)**2 + Eyy**2)**.5
    Nkcrit = norm_rot/norm_shear
    unit_values = make_unit('')
    if raw:
        return np.ma.masked_array(Nkcrit, mask)
    else:
        q_sf = ScalarField()
        q_sf.import_from_arrays(axe_x, axe_y, Nkcrit, mask,
                                unit_x=vectorfield.unit_x,
                                unit_y=vectorfield.unit_y,
                                unit_values=unit_values)
        return q_sf


def get_delta_criterion(vectorfield, mask=None, raw=False):
    """
    Return the scalar field of the 2D Delta criterion .
    Define as "(Q/3)**3 + (R/2)**2" , with "Q" the Q criterion,
    and "R" the determinant of the jacobian matrice of the velocity.

    Parameters
    ----------
    vectorfield : VectorField object
    mask : array of boolean, optional
        Has to be an array of the same size of the vector field object,
        iota2 will be compute only where zone is 'False'.
    raw : boolean, optional
        If 'False' (default), a ScalarField is returned,
        if 'True', an array is returned.

    Note
    ----
    Negative values of Delta mean that the local streamline pattern is closed
    or spiraled.
    """
    if not isinstance(vectorfield, VectorField):
        raise TypeError("'vectorfield' must be a VectorField object")
    if mask is None:
        mask = np.zeros(vectorfield.shape, dtype=bool)
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'mask' must be an array of boolean")
    else:
        mask = np.array(mask, dtype=bool)
    axe_x, axe_y = vectorfield.axe_x, vectorfield.axe_y
    # calcul des gradients
    Exx, Exy, Eyx, Eyy = get_gradients(vectorfield, raw=True)
    # calcul de Q
    norm_rot2 = 1/2.*(Exy - Eyx)**2
    norm_shear2 = (Exx**2 + 1./2.*(Exy + Eyx)**2 + Eyy**2)
    Q = .5*(norm_rot2) - norm_shear2
    # calcul de R
    R = np.zeros(Exx.shape)
    for i in np.arange(Exx.shape[0]):
        for j in np.arange(Exx.shape[1]):
            Jac = [[Exx[i, j], Exy[i, j]], [Eyx[i, j], Eyy[i, j]]]
            eigval, eigvect = np.linalg.eig(Jac)
            if np.all(np.imag(eigval) == 0):
                mask[i, j] = True
                continue
            R[i, j] = -np.linalg.det(Jac)
    # calcul de Delta
    delta = (Q/3.)**3 + (R/2.)**2
    unit_values = ((vectorfield.unit_values/vectorfield.unit_x)**2)**3
    scale = unit_values.asNumber()
    delta *= scale
    unit_values = unit_values/scale
    if raw:
        return np.ma.masked_array(delta, mask)
    else:
        delta_sf = ScalarField()
        delta_sf.import_from_arrays(axe_x, axe_y, delta, mask,
                                    unit_x=vectorfield.unit_x,
                                    unit_y=vectorfield.unit_y,
                                    unit_values=unit_values)
        return delta_sf


def get_lambda2(vectorfield, mask=None, raw=False):
    """
    Return the lambda2 scalar field. According to ... vortex are defined by
    zone of negative values of lambda2.
    The fonction is only usable on orthogonal fields.

    Parameters
    ----------
    vectorfield : VectorField object
    mask : array of boolean, optionnal
        Has to be an array of the same size of the vector field object,
        iota2 will be compute only where zone is 'False'.
    raw : boolean, optional
        If 'False' (default), a ScalarField is returned,
        if 'True', an array is returned.
    """
    # check parameter
    if not isinstance(vectorfield, VectorField):
        raise TypeError("'vectorfield' must be a VectorField object")
    if mask is None:
        mask = np.zeros(vectorfield.shape)
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'mask' must be an array of boolean")
    else:
        mask = np.array(mask)
    mask = np.logical_or(mask, vectorfield.mask)
    # getting velocity gradients
    Udx, Udy, Vdx, Vdy = get_gradients(vectorfield, raw=True)
    mask = np.logical_or(mask, Udx.mask)
    # creating returning matrix
    lambda2 = np.zeros(vectorfield.shape)
    # loop on points
    for i in np.arange(lambda2.shape[0]):
        for j in np.arange(lambda2.shape[1]):
            # check if masked
            if mask[i, j]:
                continue
            # getting symmetric and antisymetric parts
            S = 1./2.*np.array([[2*Udx[i, j], Udy[i, j] + Vdx[i, j]],
                                [Vdx[i, j] + Udy[i, j], 2*Vdy[i, j]]])
            Omega = 1./2.*np.array([[0, Udy[i, j] - Vdx[i, j]],
                                    [Vdx[i, j] - Udy[i, j], 0]])
            # getting S^2 + Omega^2
            M = np.dot(S, S) + np.dot(Omega, Omega)
            # getting second eigenvalue
            lambds = linalg.eig(M, left=False, right=False)
            l2 = np.min(np.real(lambds))
            # storing lambda2
            lambda2[i, j] = l2
    # returning
    if raw:
        return np.ma.masked_array(l2, mask)
    else:
        lambd_sf = ScalarField()
        axe_x, axe_y = vectorfield.axe_x, vectorfield.axe_y
        unit_x, unit_y = vectorfield.unit_x, vectorfield.unit_y
        lambd_sf.import_from_arrays(axe_x, axe_y, lambda2, mask,
                                    unit_x=unit_x, unit_y=unit_y,
                                    unit_values=make_unit(''))
        return lambd_sf


def get_residual_vorticity(vf, raw=False):
    """
    Return a scalar field with the residual of the vorticity.
    (see Kolar (2007)).

    Parameters
    ----------
    vf : VectorField or Velocityfield
        Field on which compute shear stress
    raw : boolean, optional
        If 'True', return an arrays,
        if 'False' (default), return a ScalarField object.
    """
    if isinstance(vf, VectorField):
        # getting data
        tmp_vf = vf.copy()
        tmp_vf.fill(inplace=True, reduce_tri=True)
        axe_x, axe_y = tmp_vf.axe_x, tmp_vf.axe_y
        comp_x, comp_y = tmp_vf.comp_x, tmp_vf.comp_y
        mask = tmp_vf.mask
        dx = axe_x[1] - axe_x[0]
        dy = axe_y[1] - axe_y[0]
        # getting gradients
        Exx, Exy = np.gradient(comp_x, dx, dy)
        Eyx, Eyy = np.gradient(comp_y, dx, dy)
        # getting principal rate of strain (s)
        s = np.sqrt(4*Exx**2 + (Exy + Eyx)**2)/2.
        # getting the vorticity-tensor component
        omega = (Eyx - Exy)/2.
        omega_abs = np.abs(omega)
        sign_omega = np.zeros(omega.shape, dtype=int)
        sign_omega[omega_abs == 0] = 1.
        sign_omega[omega_abs != 0] = (omega[omega_abs != 0] /
                                      omega_abs[omega_abs != 0])
        filt = s < omega_abs
        # getting the residual vorticity
        res_vort = np.zeros(tmp_vf.shape)
        res_vort[filt] = sign_omega[filt]*(omega_abs[filt] - s[filt])
        if raw:
            return res_vort
        else:
            unit_x, unit_y = tmp_vf.unit_x, tmp_vf.unit_y
            unit_values = vf.unit_values/vf.unit_x
            res_vort *= unit_values.asNumber()
            unit_values /= unit_values.asNumber()
            vort_sf = ScalarField()
            vort_sf.import_from_arrays(axe_x, axe_y, res_vort, mask=mask,
                                       unit_x=unit_x, unit_y=unit_y,
                                       unit_values=unit_values)
            return vort_sf
    elif isinstance(vf, TemporalVectorFields):
        if raw:
            tvf = []
            for field in vf.fields:
                tvf.append(get_residual_vorticity(field, raw=True))
        else:
            tvf = TemporalScalarFields()
            for i, field in enumerate(vf.fields):
                tvf.add_field(get_residual_vorticity(field, raw=False),
                              time=vf.times[i], unit_times=vf.unit_times)
        # return
        return tvf
    else:
        raise TypeError()


def get_shear_vorticity(vf, raw=False):
    """
    Return a scalar field with the shear vorticity.
    (see Kolar (2007)).

    Parameters
    ----------
    vf : VectorField or VectorFields
        Field on which compute shear stress
    raw : boolean, optional
        If 'True', return an arrays,
        if 'False' (default), return a ScalarField object.
    """
    if isinstance(vf, VectorField):
        # getting data
        tmp_vf = vf.copy()
        tmp_vf.crop_masked_border()
        tmp_vf.fill(inplace=True, reduce_tri=True)
        axe_x, axe_y = tmp_vf.axe_x, tmp_vf.axe_y
        comp_x, comp_y = tmp_vf.comp_x, tmp_vf.comp_y
        mask = tmp_vf.mask
        dx = axe_x[1] - axe_x[0]
        dy = axe_y[1] - axe_y[0]
        # getting gradients
        Exx, Exy = np.gradient(comp_x, dx, dy)
        Eyx, Eyy = np.gradient(comp_y, dx, dy)
        # getting principal rate of strain (s)
        s = np.sqrt(4*Exx**2 + (Exy + Eyx)**2)/2.
        # getting the vorticity-tensor component
        omega = (Eyx - Exy)/2.
        omega_abs = np.abs(omega)
        sign_omega = omega/omega_abs
        filt1 = s < omega_abs
        filt2 = np.logical_not(filt1)
        # getting the residual vorticity
        sh_vort = np.zeros(vf.shape)
        sh_vort[filt2] = omega[filt2]
        sh_vort[filt1] = sign_omega[filt1]*(s[filt1])
        if raw:
            return sh_vort
        else:
            unit_x, unit_y = tmp_vf.unit_x, tmp_vf.unit_y
            unit_values = vf.unit_values/vf.unit_x
            sh_vort *= unit_values.asNumber()
            unit_values /= unit_values.asNumber()
            vort_sf = ScalarField()
            vort_sf.import_from_arrays(axe_x, axe_y, sh_vort, mask=mask,
                                       unit_x=unit_x, unit_y=unit_y,
                                       unit_values=unit_values)
            return vort_sf
    elif isinstance(vf, TemporalVectorFields):
        ret = TemporalScalarFields()
        for i, field in enumerate(vf.fields):
            tmp_ret = get_shear_vorticity(field)
            ret.add_field(tmp_ret, time=vf.times[i],
                          unit_times=vf.unit_times)
        return ret
    else:
        raise TypeError()
