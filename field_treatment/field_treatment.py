# -*- coding: utf-8 -*-
"""
Created on Fri May 16 22:37:21 2014

@author: muahah
"""

import pdb
import numpy as np
import scipy.interpolate as spinterp
from scipy.optimize import leastsq
from ..core import Points, ScalarField, VectorField, Profile, \
    ARRAYTYPES, NUMBERTYPES, STRINGTYPES


def get_gradients(field, raw=False):
    """
    Return gradients along x and y.

    (Obtained arrays corespond to components of the Jacobian matrix)

    Parameters
    ----------
    vf : VelocityField, ScalarField or Profile object
        Field/profile to compute gradient from.
    raw : boolean
        If 'False' (default), ScalarFields objects are returned.
        If 'True', arrays are returned.

    Returns
    -------
    grad : tuple of ScalarField or arrays or Profile
        For VectorField input : (dVx/dx, dVx/dy, dVy/dx, dVy/dy),
        for ScalarField input : (dV/dx, dV/dy).
        for Profile input : dy/dx.
    """
    if isinstance(field, ScalarField):
        dx = field.axe_x[1] - field.axe_x[0]
        dy = field.axe_y[1] - field.axe_y[0]
        grad_x, grad_y = np.gradient(np.ma.masked_array(field.values,
                                                        field.mask),
                                     dx, dy)
        # applying masks
        mask = np.logical_or(grad_x.mask, grad_y.mask)
        if raw:
            # returning
            grad_x.mask = mask
            grad_y.mask = mask
            return grad_x, grad_y
        else:
            # arranging units
            gradx = ScalarField()
            unit_values_x = field.unit_values/field.unit_x
            factx = unit_values_x.asNumber()
            unit_values_x = unit_values_x/factx
            grad_x *= factx
            unit_values_y = field.unit_values/field.unit_y
            facty = unit_values_y.asNumber()
            unit_values_y = unit_values_y/facty
            grad_y *= facty
            #pdb.set_trace()

            # returning
            gradx.import_from_arrays(field.axe_x, field.axe_y, grad_x.data,
                                     mask=mask, unit_x=field.unit_x,
                                     unit_y=field.unit_y,
                                     unit_values=unit_values_x)
            grady = ScalarField()
            grady.import_from_arrays(field.axe_x, field.axe_y, grad_y.data,
                                     mask=mask, unit_x=field.unit_x,
                                     unit_y=field.unit_y,
                                     unit_values=unit_values_y)
            return gradx, grady
    elif isinstance(field, VectorField):
        dx = field.axe_x[1] - field.axe_x[0]
        dy = field.axe_y[1] - field.axe_y[0]
        Vx_dx, Vx_dy = np.gradient(np.ma.masked_array(field.comp_x,
                                                      field.mask),
                                   dx, dy)
        Vy_dx, Vy_dy = np.gradient(np.ma.masked_array(field.comp_y,
                                                      field.mask),
                                   dx, dy)
        # applying masks
        mask = np.logical_or(Vx_dx.mask, Vx_dy.mask)
        mask = np.logical_or(mask, Vy_dx.mask)
        mask = np.logical_or(mask, Vy_dy.mask)
        if raw:
            # returning
            Vx_dx.mask = mask
            Vx_dy.mask = mask
            Vy_dx.mask = mask
            Vy_dy.mask = mask
            return (Vx_dx, Vx_dy, Vy_dx, Vy_dy)
        else:
            #arragning unit
            unit_values_x = field.unit_values/field.unit_x
            factx = unit_values_x.asNumber()
            unit_values_x /= factx
            Vx_dx *= factx
            Vy_dx *= factx
            unit_values_y = field.unit_values/field.unit_y
            facty = unit_values_y.asNumber()
            unit_values_y /= facty
            Vx_dy *= facty
            Vy_dy *= facty
            #returning
            grad1 = ScalarField()
            grad1.import_from_arrays(field.axe_x, field.axe_y, Vx_dx.data,
                                     mask=mask, unit_x=field.unit_x,
                                     unit_y=field.unit_y,
                                     unit_values=unit_values_x)
            grad2 = ScalarField()
            grad2.import_from_arrays(field.axe_x, field.axe_y, Vx_dy.data,
                                     mask=mask, unit_x=field.unit_x,
                                     unit_y=field.unit_y,
                                     unit_values=unit_values_y)
            grad3 = ScalarField()
            grad3.import_from_arrays(field.axe_x, field.axe_y, Vy_dx.data,
                                     mask=mask, unit_x=field.unit_x,
                                     unit_y=field.unit_y,
                                     unit_values=unit_values_x)
            grad4 = ScalarField()
            grad4.import_from_arrays(field.axe_x, field.axe_y, Vy_dy.data,
                                     mask=mask, unit_x=field.unit_x,
                                     unit_y=field.unit_y,
                                     unit_values=unit_values_y)
            return grad1, grad2, grad3, grad4
    elif isinstance(field, Profile):
        tmp_prof = field.evenly_space('quadratic')
        axe_x = tmp_prof.x
        dx = axe_x[1] - axe_x[0]
        mask = tmp_prof.mask
        grad = np.gradient(tmp_prof.y)
        if raw :
            return grad
        else:
            unit_y = tmp_prof.unit_y/tmp_prof.unit_x
            facty = unit_y.asNumber()
            unit_y /= facty
            grad *= facty
            prof_grad = Profile(axe_x, grad, mask=mask,
                                unit_x=tmp_prof.unit_x, unit_y=unit_y)
            return prof_grad
    else:
        raise TypeError()


def reconstruct_from_gradients(field_dx, field_dy, field2_dx=None,
                               field2_dy=None, ols=False, maxiter=10):
    """
    Reconstruct a field with the gradients of this field.

    Parameters
    ----------
    field_dx, field_dy : ScalarField objects
        Gradients along x and y.
    field2_dx, field_dy : ScalarField objects, optional
        Gradients for the second component, if the reconstructed field is a
        vector field.
    ols : boolean, optional
        If 'True', ordinary least square is used to get a more precise result
        (can be quite long).
        If 'False' (default), a simple reconstruction based on taylor
        developement is used.
    maxiter : integer, optional
        Maximum number of iteration for the ols solver (default: 10)
        (more mean accurate results but slower computation).


    Returns
    -------
    rec_field : ScalarField or VectorField object
        Reconstructed field.

    Notes
    -----
    Given result can only be relative values
    (because of the information lost while derivating).
    The returned result are normalized so that the mean of the field is egal
    to zero.
    """
    ### checking parameters
    if not isinstance(field_dx, ScalarField):
        raise TypeError()
    if not isinstance(field_dy, ScalarField):
        raise TypeError()
    if field2_dx is None and field2_dy is None:
        kind = 'SF'
        if field2_dx is not None or field2_dy is not None:
            raise ValueError()
    else:
        kind = 'VF'
        if not isinstance(field2_dx, ScalarField):
            raise TypeError()
        if not isinstance(field2_dy, ScalarField):
            raise TypeError()
    ### recursive loop for VF
    if kind == 'VF':
        Vx = reconstruct_from_gradients(field_dx, field_dy, ols=ols)
        Vy = reconstruct_from_gradients(field2_dx, field2_dy, ols=ols)
        mask = np.logical_or(Vx.mask, Vy.mask)
        V = VectorField()
        V.import_from_arrays(Vx.axe_x, Vx.axe_y, Vx.values, Vy.values,
                             mask=mask, unit_x=Vx.unit_x, unit_y=Vx.unit_y,
                             unit_values=Vx.unit_values)
        return V
    ### reconstruction for SF
    else:
        ### getting data
        axe_x = field_dx.axe_x
        axe_y = field_dy.axe_y
        dx = axe_x[1] - axe_x[0]
        dy = axe_y[1] - axe_y[0]
        du_dx = field_dx.values
        du_dy = field_dy.values
        unit_values = field_dx.unit_values*field_dx.unit_x
        fact_unit_values = unit_values.asNumber()
        unit_values /= unit_values.asNumber()

        ###  Getting borders
        # 1 (0, 0)
        u1 = np.zeros(du_dx.shape)
        for i in np.arange(1, u1.shape[0]):
            u1[i, 0] = u1[i - 1, 0] + dx*(du_dx[i - 1, 0]
                                          + du_dx[i, 0])/2.
        for i in np.arange(1, u1.shape[1]):
            u1[0, i] = u1[0, i - 1] + dy*(du_dy[0, i - 1]
                                          + du_dy[0, i])/2.
        # 2(0, 1)
        u2 = np.zeros(du_dx.shape)
        for i in np.arange(1, u2.shape[0]):
            u2[i, -1] = u2[i - 1, -1] + dx*(du_dx[i - 1, -1]
                                            + du_dx[i, -1])/2.
        for i in np.arange(u2.shape[1] - 2, -1, -1):
            u2[0, i] = u2[0, i + 1] - dy*(du_dy[0, i + 1]
                                          + du_dy[0, i])/2.
        # 3 (1, 0)
        u3 = np.zeros(du_dx.shape)
        for i in np.arange(u3.shape[0] - 2, -1, -1):
            u3[i, 0] = u3[i + 1, 0] - dx*(du_dx[i + 1, 0]
                                          + du_dx[i, 0])/2.
        for i in np.arange(1, u3.shape[1]):
            u3[-1, i] = u3[-1, i - 1] + dy*(du_dy[-1, i - 1]
                                            + du_dy[-1, i])/2.
        # 4 (1, 1)
        u4 = np.zeros(du_dx.shape)
        for i in np.arange(u4.shape[0] - 2, -1, -1):
            u4[i, -1] = u4[i + 1, -1] - dx*(du_dx[i + 1, -1]
                                            + du_dx[i, -1])/2.
        for i in np.arange(u4.shape[1] - 2, -1, -1):
            u4[-1, i] = u4[-1, i + 1] - dy*(du_dy[-1, i + 1]
                                            + du_dy[-1, i])/2.
        # concatenate borders
        filt_border = np.zeros(u1.shape, dtype=bool)
        filt_border[:, 0] = True
        filt_border[:, -1] = True
        filt_border[0, :] = True
        filt_border[-1, :] = True
        u4 = (u4 - (u4[-1, 0] - u1[-1, 0] + u4[0, -1] - u1[0, -1])/2.)
        u4[0:-1, 0:-1] = 0.
        ua = u1 + u4
        ua[-1, 0] /= 2.
        ua[0, -1] /= 2.
        ua[filt_border] -= np.mean(ua[filt_border])
        u3 = (u3 - (u3[0, 0] - u2[0, 0] + u3[-1, -1] - u2[-1, -1])/2.)
        u3[0:-1, 1::] = 0.
        ub = u2 + u3
        ub[0, 0] /= 2.
        ub[-1, -1] /= 2.
        ub[filt_border] -= np.mean(ub[filt_border])
        u = (ua + ub)/2.
        ### Getting center
        # 1 from (0, 0)
        u1 = u.copy()
        for i in np.arange(1, u1.shape[0]):
            for j in np.arange(1, u1.shape[1]):
                est_a = u1[i - 1, j] + dx*(du_dx[i - 1, j]
                                           + du_dx[i, j])/2.
                est_b = u1[i, j - 1] + dy*(du_dy[i, j - 1]
                                           + du_dy[i, j])/2.
                u1[i, j] = (est_a + est_b)/2.
        # 2 from (0, 1)
        u2 = u.copy()
        for i in np.arange(1, u2.shape[0]):
            for j in np.arange(u2.shape[1] - 2, -1, -1):
                est_a = u2[i - 1, j] + dx*(du_dx[i - 1, j]
                                           + du_dx[i, j])/2.
                est_b = u2[i, j + 1] - dy*(du_dy[i, j + 1]
                                           + du_dy[i, j])/2.
                u2[i, j] = (est_a + est_b)/2.
        # 3 from (1, 0)
        u3 = u.copy()
        for i in np.arange(u3.shape[0] - 2, -1, -1):
            for j in np.arange(1, u3.shape[1]):
                est_a = u3[i + 1, j] - dx*(du_dx[i + 1, j]
                                           + du_dx[i, j])/2.
                est_b = u3[i, j - 1] + dy*(du_dy[i, j - 1]
                                           + du_dy[i, j])/2.
                u3[i, j] = (est_a + est_b)/2.
        # 4 from (1, 1)
        u4 = u.copy()
        for i in np.arange(u4.shape[0] - 2, -1, -1):
            for j in np.arange(u2.shape[1] - 2, -1, -1):
                est_a = u4[i + 1, j] - dx*(du_dx[i + 1, j]
                                           + du_dx[i, j])/2.
                est_b = u4[i, j + 1] - dy*(du_dy[i, j + 1]
                                           + du_dy[i, j])/2.
                u4[i, j] = (est_a + est_b)/2.
        # normalize
        filt_corn = np.zeros(u.shape, dtype=int)
        filt_corn[0, 0] = 1
        filt_corn[0, -1] = 1
        filt_corn[-1, 0] = 1
        filt_corn[-1, -1] = 1
        u1 -= np.mean(u1[filt_corn])
        u2 -= np.mean(u2[filt_corn])
        u3 -= np.mean(u3[filt_corn])
        u4 -= np.mean(u4[filt_corn])
        # concatenate (with linear weight fonction)
        ### TODO : let or remove ?
        X, Y = np.meshgrid(np.arange(u.shape[1]), np.arange(u.shape[0]),
                           dtype=float)
        weight1 = X[::-1, ::-1] + Y[::-1, ::-1]
        weight2 = X[::-1, :] + Y[::-1, :]
        weight3 = X[:, ::-1] + Y[:, ::-1]
        weight4 = X + Y
        weight_tot = weight1 + weight2 + weight3 + weight4
        u = (u1*weight1 + u2*weight2 + u3*weight3 + u4*weight4)/weight_tot

        ### Second OLS reconstruction
        if ols:
            def min_funct(u, du_dx, du_dy, dx, dy):
                u = u.reshape(du_dx.shape)
                du2_dx, du2_dy = np.gradient(u, dx, dy)
                res = np.abs(du_dx - du2_dx) + np.abs(du_dy - du2_dy)
                print(np.sum(res))
                return res.flatten()
            u, mess = leastsq(min_funct, u.flatten(),
                              args=(du_dx, du_dy, dx, dy),
                              maxfev=maxiter*len(u.flatten()))
            u = u.reshape(du_dx.shape)
        ### retuning
        u *= fact_unit_values
        u = u - np.mean(u)
        U = ScalarField()
        U.import_from_arrays(axe_x, axe_y, u,
                             unit_x=field_dx.unit_x, unit_y=field_dx.unit_y,
                             unit_values=unit_values)
#        if ols:
#            U.smooth(tos='gaussian', size=1, inplace=True)
        return U


def get_jacobian_eigenproperties(field, raw=False, eig_val=True,
                                 eig_vect=True):
    """
    Return eigenvalues and eigenvectors of the jacobian matrix on all the
    field.

    Parameters
    ----------
    field : VectorField object
        .
    raw : boolean
        If 'False' (default), ScalarFields objects are returned.
        If 'True', arrays are returned.
    eig_val : boolean, optional
        If 'True', eigenvalues are returned.
    eig_vect : boolean, optional
        If 'True', eigenvectors are returned.

    Returns
    -------
    eig_val1_re : ScalarField object, or array
        Real part of the first eigenvalue.
    eig_val1_im : ScalarField object, or array
        Imaginary part of the first eigenvalue.
    eig_val2_re : ScalarField object, or array
        Real part of the second eigenvalue.
    eig_val2_im : ScalarField object, or array
        Imaginary part of the second eigenvalue.
    eig1_vf : VectorField object, or tuple of arrays
        Eigenvector associated with first eigenvalue.
    eig2_vf : VectorField object, or tuple of arrays
        Eigenvector associated with second eigenvalue.
    """
    # getting datas
    Vx_dx, Vx_dy, Vy_dx, Vy_dy = get_gradients(field, raw=True)
    shape = Vx_dx.shape
    mask = Vx_dx.mask
    mask = np.logical_or(mask, Vx_dy.mask)
    mask = np.logical_or(mask, Vy_dx.mask)
    mask = np.logical_or(mask, Vy_dy.mask)
    Vx_dx = Vx_dx.data
    Vx_dy = Vx_dy.data
    Vy_dx = Vy_dx.data
    Vy_dy = Vy_dy.data
    # loop on flatten arrays
    eig1 = np.zeros(shape, dtype=complex)
    eig2 = np.zeros(shape, dtype=complex)
    eig1v_x = np.zeros(shape)
    eig1v_y = np.zeros(shape)
    eig2v_x = np.zeros(shape)
    eig2v_y = np.zeros(shape)
    for i in np.arange(shape[0]*shape[1]):
        # breaking when masked
        if mask.flat[i]:
            continue
        # getting the local jacobian matrix
        loc_jac = [[Vx_dx.flat[i], Vx_dy.flat[i]],
                   [Vy_dx.flat[i], Vy_dy.flat[i]]]
        # getting local max eigenvalue and associated eigenvectors
        loc_eigv, loc_eigvect = np.linalg.eig(loc_jac)
        max_eig = np.argmax(loc_eigv)
        if max_eig == 0:
            min_eig = 1
        else:
            min_eig = 0
        # storing in arrays
        eig1.flat[i] = loc_eigv[max_eig]
        eig2.flat[i] = loc_eigv[min_eig]
        eig1v_x.flat[i] = loc_eigvect[0, max_eig]
        eig1v_y.flat[i] = loc_eigvect[1, max_eig]
        eig2v_x.flat[i] = loc_eigvect[0, min_eig]
        eig2v_y.flat[i] = loc_eigvect[1, min_eig]
    # extracting real and imaginary
    eig1_r = np.real(eig1)
    eig1_i = np.imag(eig1)
    eig2_r = np.real(eig2)
    eig2_i = np.imag(eig2)
    # returning
    if raw:
        ret = ()
        if eig_val:
            ret += (eig1_r, eig1_i, eig2_r, eig2_i)
        if eig_vect:
            ret += ((eig1v_x, eig1v_y), (eig2v_x, eig2v_y))
        return ret
    else:
        ret = ()
        if eig_val:
            eig1_re_sf = ScalarField()
            eig1_re_sf.import_from_arrays(field.axe_x, field.axe_y, eig1_r,
                                          mask=mask, unit_x=field.unit_x,
                                          unit_y=field.unit_y,
                                          unit_values="")
            eig1_im_sf = ScalarField()
            eig1_im_sf.import_from_arrays(field.axe_x, field.axe_y, eig1_i,
                                          mask=mask, unit_x=field.unit_x,
                                          unit_y=field.unit_y,
                                          unit_values="")
            eig2_re_sf = ScalarField()
            eig2_re_sf.import_from_arrays(field.axe_x, field.axe_y, eig2_r,
                                          mask=mask, unit_x=field.unit_x,
                                          unit_y=field.unit_y,
                                          unit_values="")
            eig2_im_sf = ScalarField()
            eig2_im_sf.import_from_arrays(field.axe_x, field.axe_y, eig2_i,
                                          mask=mask, unit_x=field.unit_x,
                                          unit_y=field.unit_y,
                                          unit_values="")
            ret += (eig1_re_sf, eig1_im_sf, eig2_re_sf, eig2_im_sf)
        if eig_vect:
            eig1_vf = VectorField()
            eig1_vf.import_from_arrays(field.axe_x, field.axe_y, eig1v_x, eig1v_y,
                                       mask=mask, unit_x=field.unit_x,
                                       unit_y=field.unit_y,
                                       unit_values="")
            eig2_vf = VectorField()
            eig2_vf.import_from_arrays(field.axe_x, field.axe_y, eig2v_x, eig2v_y,
                                       mask=mask, unit_x=field.unit_x,
                                       unit_y=field.unit_y,
                                       unit_values="")
            ret += (eig1_vf, eig2_vf)
        return ret


def get_Kenwright_field(field, raw=False):
    """
    Return a field with the vector product between the velocity field and the
    eigen vectors of the velocity jacobian.
    Values of 0 on these field should represent separation lines
    or re-attachment lines
    (See Kenwright et al (1998)).

    Parameters
    ----------
    field : VectorField
        .
    raw : bool, optional
        If 'False' (default), ScalarFields are returned
        If 'True', arrays are returned

    Returns
    -------
    K1_field : ScalarField
        Vector product with the principal eigen vector
    K2_field : ScalarField
        Vector product with the oher eigen vector
    """
    # get jacobian  eigen properties
    eigval1, eigval2, eigvect1, eigvect2 = get_jacobian_eigenproperties(field)
    K1 = np.zeros(field.shape)
    K2 = np.zeros(field.shape)
    # making vector product
    for xi in np.arange(len(field.axe_x)):
        for yi in np.arange(len(field.axe_y)):
            K1[xi, yi] = (eigvect1.comp_x[xi, yi]*field.comp_y[xi, yi]
                          - eigvect1.comp_y[xi, yi]*field.comp_x[xi, yi])
            K2[xi, yi] = (eigvect2.comp_x[xi, yi]*field.comp_y[xi, yi]
                          - eigvect2.comp_y[xi, yi]*field.comp_x[xi, yi])
    # returning
    if raw:
        return K1, K2
    else:
        K1_sf = ScalarField()
        K1_sf.import_from_arrays(field.axe_x, field.axe_y, K1,
                                 unit_x=field.unit_x, unit_y=field.unit_y,
                                 unit_values=field.unit_values)
        K2_sf = ScalarField()
        K2_sf.import_from_arrays(field.axe_x, field.axe_y, K2,
                                 unit_x=field.unit_x, unit_y=field.unit_y,
                                 unit_values=field.unit_values)
        return K1_sf, K2_sf


def get_grad_field(field, direction=1):
    """
    Return a field based on original field gradients.
    (Vx = dV/dx, Vy = DV/Vy)

    Parameters
    ----------
    field : VectorField object
        .
    direction : integer (1 or 2)
        if '1', return Vx gradients
        if '2', return Vy gradients.

    Returns
    -------
    gfield : VectorField object
        .
    """
    # checking parameters:
    if not direction in [1, 2]:
        raise ValueError()
    # getting gradients
    field.comp_x = field.comp_x
    field.comp_y = field.comp_y
    grads = get_gradients(field)
    # returning
    if direction == 1:
        gvx = grads[0]
        gvy = grads[1]
    else:
        gvx = grads[2]
        gvy = grads[3]
    gfield = VectorField()
    gfield.import_from_arrays(field.axe_x, field.axe_y, comp_x=gvx.values,
                              comp_y=gvy.values, unit_x=field.unit_x,
                              unit_y=field.unit_y,
                              unit_values=field.unit_values)
    return gfield


def get_track_field(vf):
    """
    Return the track field.
    (Vx, Vy) => (-Vy, Vx)
    """
    Vx = -vf.comp_y
    Vy = vf.comp_x
    track_field = VectorField()
    track_field.import_from_arrays(vf.axe_x, vf.axe_y, Vx, Vy, mask=vf.mask,
                                   unit_x=vf.unit_x, unit_y=vf.unit_y,
                                   unit_values=vf.unit_values)
    return track_field


def get_streamlines(vf, xy, delta=.25, interp='linear',
                    reverse_direction=False):
    """
    Return a tuples of Points object representing the streamline begining
    at the points specified in xy.
    Warning : fill the field before computing streamlines, can give bad
    results if the field have a lot of masked values.

    Parameters
    ----------
    vf : VectorField or velocityField object
        Field on which compute the streamlines
    xy : tuple
        Tuple containing each starting point for streamline.
    delta : number, optional
        Spatial discretization of the stream lines,
        relative to a the spatial discretization of the field.
    interp : string, optional
        Used interpolation for streamline computation.
        Can be 'linear'(default) or 'cubic'
    reverse_direction : boolean, optional
        If True, the streamline goes upstream.

    Returns
    -------
    streams : tuple of Points object
        Each Points object represent a streamline

    """
    # checking parameters coherence
    if not isinstance(vf, VectorField):
        raise TypeError()
    axe_x, axe_y = vf.axe_x, vf.axe_y
    if not isinstance(xy, ARRAYTYPES):
        raise TypeError("'xy' must be a tuple of arrays")
    xy = np.array(xy, dtype=float)
    if xy.shape == (2,):
        xy = [xy]
    elif len(xy.shape) == 2 and xy.shape[1] == 2:
        pass
    else:
        raise ValueError("'xy' must be a tuple of arrays")
    if not isinstance(delta, NUMBERTYPES):
        raise TypeError("'delta' must be a number")
    if delta <= 0:
        raise ValueError("'delta' must be positive")
    if delta > len(axe_x) or delta > len(axe_y):
        raise ValueError("'delta' is too big !")
    if not isinstance(interp, STRINGTYPES):
        raise TypeError("'interp' must be a string")
    if not (interp == 'linear' or interp == 'cubic'):
        raise ValueError("Unknown interpolation method")
    # getting datas
    tmpvf = vf.copy()
    tmpvf.fill()
    unit_x, unit_y = vf.unit_x, vf.unit_y
    Vx, Vy = vf.comp_x, vf.comp_y
    deltaabs = delta * ((axe_x[-1]-axe_x[0])/len(axe_x)
                        + (axe_y[-1]-axe_y[0])/len(axe_y))/2.
    deltaabs2 = deltaabs**2
    # interpolation du champ de vitesse
    if interp == 'linear':
        interp_vx = spinterp.RectBivariateSpline(axe_x, axe_y, Vx,
                                                 kx=1, ky=1)
        interp_vy = spinterp.RectBivariateSpline(axe_x, axe_y, Vy,
                                                 kx=1, ky=1)

    elif interp == 'cubic':
        interp_vx = spinterp.RectBivariateSpline(axe_x, axe_y, Vx,
                                                 kx=3, ky=3)
        interp_vy = spinterp.RectBivariateSpline(axe_x, axe_y, Vy,
                                                 kx=3, ky=3)
    # Calcul des streamlines
    streams = []
    longmax = (len(axe_x)+len(axe_y))/delta
    for coord in xy:
        # getting coordinates
        x = coord[0]
        y = coord[1]
        x = float(x)
        y = float(y)
        # si le points est en dehors, on ne fait rien
        if x < axe_x[0] or x > axe_x[-1] or y < axe_y[0] or y > axe_y[-1]:
            continue
        # initialisation du calcul d'une streamline
        stream = np.zeros((longmax, 2))
        stream[0, :] = [x, y]
        i = 1
        # calcul d'une streamline
        while True:
            tmp_vx = interp_vx(stream[i-1, 0], stream[i-1, 1])[0, 0]
            tmp_vy = interp_vy(stream[i-1, 0], stream[i-1, 1])[0, 0]
            norm = np.linalg.norm([tmp_vx, tmp_vy])
            # tests d'arret
            if tmp_vx == 0 and tmp_vy == 0:
                # if masked value
                break
            if i >= longmax-1:
                break
            if i > 15:
                no = [stream[0:i-10, 0] - stream[i-1, 0],
                      stream[0:i-10, 1] - stream[i-1, 1]]
                no = no[0]**2 + no[1]**2
                if any(no < deltaabs2/2):
                    break
            # calcul des dx et dy pour une streamline
            if reverse_direction:
                norm = -norm
            dx = tmp_vx/norm*deltaabs
            dy = tmp_vy/norm*deltaabs
            stream[i, :] = [stream[i-1, 0] + dx, stream[i-1, 1] + dy]
            i += 1
            # tests d'arret
            x = stream[i-1, 0]
            y = stream[i-1, 1]
            if (x < axe_x[0] or x > axe_x[-1] or y < axe_y[0]
                    or y > axe_y[-1]):
                break
        # creating Points objects for returning
        stream = stream[:i]
        pts = Points(stream, unit_x=unit_x, unit_y=unit_y,
                     name='streamline at x={:.3f}, y={:.3f}'.format(x, y))
        streams.append(pts)
    # returning
    if len(streams) == 0:
        return None
    elif len(streams) == 1:
        return streams
    else:
        return streams


def get_tracklines(vf, xy, delta=.25, interp='linear',
                   reverse_direction=False):
    """
    Return a tuples of Points object representing the tracklines begining
    at the points specified in xy.
    Warning : fill the field before computing streamlines, can give bad
    results if the field have a lot of masked values.

    Parameters
    ----------
    vf : VectorField or velocityField object
        Field on which compute the tracklines
    xy : tuple
        Tuple containing each starting point for tracklines.
    delta : number, optional
        Spatial discretization of the tracklines,
        relative to a the spatial discretization of the field.
    interp : string, optional
        Used interpolation for trackline computation.
        Can be 'linear'(default) or 'cubic'
    reverse_direction : boolean, optional
        If True, the trackline goes upstream.

    Returns
    -------
    streams : tuple of Points object
        Each Points object represent a trackline

    """
    # get the track field
    track_field = get_track_field(vf)
    return get_streamlines(track_field, xy, delta=delta, interp=interp,
                           reverse_direction=reverse_direction)


def get_shear_stress(vf, raw=False):
    """
    Return a vector field with the shear stress.

    Parameters
    ----------
    vf : VectorField or Velocityfield
        Field on which compute shear stress
    raw : boolean, optional
        If 'True', return two arrays,
        if 'False' (default), return a VectorField object.
    """
    if not isinstance(vf, VectorField):
        raise TypeError()
    tmp_vf = vf.copy()
    tmp_vf.fill(crop_border=True)
    # Getting gradients and axes
    axe_x, axe_y = tmp_vf.axe_x, tmp_vf.axe_y
    comp_x, comp_y = tmp_vf.comp_x, tmp_vf.comp_y
    mask = tmp_vf.mask
    dx = axe_x[1] - axe_x[0]
    dy = axe_y[1] - axe_y[0]
    du_dx, du_dy = np.gradient(comp_x, dx, dy)
    dv_dx, dv_dy = np.gradient(comp_y, dx, dy)
    # swirling vectors matrix
    comp_x = dv_dx
    comp_y = du_dy
    # creating vectorfield object
    if raw:
        return (comp_x, comp_y)
    else:
        tmpvf = VectorField()
        unit_x, unit_y = tmp_vf.unit_x, tmp_vf.unit_y
        unit_values = tmp_vf.unit_values
        tmpvf.import_from_arrays(axe_x, axe_y, comp_x, comp_y, mask=mask,
                                 unit_x=unit_x, unit_y=unit_y,
                                 unit_values=unit_values)
        return tmpvf


def get_swirling_vector(vf, raw=False):
    """
    Return a vector field with the swirling vectors
    (eigenvectors of the velocity laplacian matrix
    ponderated by eigenvalues)
    (Have to be adjusted : which part of eigenvalues
    and eigen vectors take ?)

    Parameters
    ----------
    vf : VectorField or Velocityfield
        Field on which compute shear stress
    raw : boolean, optional
        If 'True', return an arrays,
        if 'False' (default), return a ScalarField object.
    """
    raise Warning("Useless (not finished)")
    if not isinstance(vf, VectorField):
        raise TypeError()
    tmp_vf = vf.copy()
    tmp_vf.fill(crop_border=True)
    # Getting gradients and axes
    axe_x, axe_y = tmp_vf.axe_x, tmp_vf.axe_y
    comp_x, comp_y = tmp_vf.comp_x, tmp_vf.comp_y
    dx = axe_x[1] - axe_x[0]
    dy = axe_y[1] - axe_y[0]
    du_dx, du_dy = np.gradient(comp_x, dx, dy)
    dv_dx, dv_dy = np.gradient(comp_y, dx, dy)
    # swirling vectors matrix
    comp_x = np.zeros(tmp_vf.shape)
    comp_y = np.zeros(tmp_vf.shape)
    # loop on  points
    for i in np.arange(0, len(axe_x)):
        for j in np.arange(0, len(axe_y)):
            lapl = [[du_dx[i, j], du_dy[i, j]],
                    [dv_dx[i, j], dv_dy[i, j]]]
            eigvals, eigvect = np.linalg.eig(lapl)
            eigvals = np.imag(eigvals)
            eigvect = np.real(eigvect)
            if eigvals[0] > eigvals[1]:
                comp_x[i, j] = eigvect[0][0]
                comp_y[i, j] = eigvect[0][1]
            else:
                comp_x[i, j] = eigvect[1][0]
                comp_y[i, j] = eigvect[1][1]
    # creating vectorfield object
    if raw:
        return (comp_x, comp_y)
    else:
        unit_x, unit_y = tmp_vf.unit_x, tmp_vf.unit_y
        # TODO: implémenter unité
        unit_values = ""
        tmp_vf = VectorField()
        tmp_vf.import_from_arrays(axe_x, axe_y, comp_x, comp_y, mask=False,
                                  unit_x=unit_x, unit_y=unit_y,
                                  unit_values=unit_values)
        return tmp_vf
