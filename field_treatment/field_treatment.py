# -*- coding: utf-8 -*-
"""
Created on Fri May 16 22:37:21 2014

@author: muahah
"""

import pdb
import numpy as np
import scipy.interpolate as spinterp
from ..core import Points, ScalarField, VectorField,\
    ARRAYTYPES, NUMBERTYPES, STRINGTYPES
import matplotlib.pyplot as plt


def get_gradients(field, raw):
    """
    Return gradients along x and y.

    (Obtained arrays corespond to components of the Jacobian matrix)

    Parameters
    ----------
    vf : VelocityField or ScalarField object
        Field to comput gradient from.
    raw : boolean
        If 'False' (default), ScalarFields objects are returned.
        If 'True', arrays are returned.

    Returns
    -------
    grad : tuple of ScalarField or arrays
        For VectorField input : (dVx/dx, dVx/dy, dVy/dx, dVy/dy),
        for ScalarField input : (dV/dx, dV/dy).
    """
    dx = field.axe_x[1] - field.axe_x[0]
    dy = field.axe_y[1] - field.axe_y[0]
    if isinstance(field, ScalarField):
        grad_x, grad_y = np.gradient(np.ma.masked_array(field.values,
                                                        field.mask),
                                     dx, dy)
        gradx = ScalarField()
        unit_values_x = field.unit_values/field.unit_x
        factx = unit_values_x.asNumber()
        unit_values_x /= factx
        grad_x *= factx
        unit_values_y = field.unit_values/field.unit_y
        facty = unit_values_x.asNumber()
        unit_values_y /= facty
        grad_y *= facty
        # returning
        if raw:
            return grad_x, grad_y
        else:
            gradx.import_from_arrays(field.axe_x, field.axe_y, grad_x.data,
                                     mask=grad_x.mask, unit_x=field.unit_x,
                                     unit_y=field.unit_y,
                                     unit_values=unit_values_x)
            grady = ScalarField()
            grady.import_from_arrays(field.axe_x, field.axe_y, grad_y.data,
                                     mask=grad_x.mask, unit_x=field.unit_x,
                                     unit_y=field.unit_y,
                                     unit_values=unit_values_y)
            return gradx, grady
    elif isinstance(field, VectorField):
        Vx_dx, Vx_dy = np.gradient(np.ma.masked_array(field.comp_x, field.mask)
                                   , dx, dy)
        Vy_dx, Vy_dy = np.gradient(np.ma.masked_array(field.comp_y, field.mask)
                                   , dx, dy)
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
        # returning
        if raw:
            return (Vx_dx, Vx_dy, Vy_dx, Vy_dy)
        else:
            grad1 = ScalarField()
            grad1.import_from_arrays(field.axe_x, field.axe_y, Vx_dx.data,
                                     mask=Vx_dx.mask, unit_x=field.unit_x,
                                     unit_y=field.unit_y,
                                     unit_values=unit_values_x)
            grad2 = ScalarField()
            grad2.import_from_arrays(field.axe_x, field.axe_y, Vx_dy.data,
                                     mask=Vx_dy.mask, unit_x=field.unit_x,
                                     unit_y=field.unit_y,
                                     unit_values=unit_values_y)
            grad3 = ScalarField()
            grad3.import_from_arrays(field.axe_x, field.axe_y, Vy_dx.data,
                                     mask=Vy_dx.mask, unit_x=field.unit_x,
                                     unit_y=field.unit_y,
                                     unit_values=unit_values_x)
            grad4 = ScalarField()
            grad4.import_from_arrays(field.axe_x, field.axe_y, Vy_dy.data,
                                     mask=Vy_dy.mask, unit_x=field.unit_x,
                                     unit_y=field.unit_y,
                                     unit_values=unit_values_y)
            return grad1, grad2, grad3, grad4
    else:
        raise TypeError()


def get_jacobian_eigenproperties(field, raw=False):
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

    Returns
    -------
    eig1_sf : ScalarField object, or array
        First eigenvalue.
    eig2_sf : ScalarField object, or array
        Second eigenvalue.
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
    eig1 = np.zeros(shape)
    eig2 = np.zeros(shape)
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
        eig1v_x.flat[i] = loc_eigvect[0, max_eig]*loc_eigv[max_eig]
        eig1v_y.flat[i] = loc_eigvect[1, max_eig]*loc_eigv[max_eig]
        eig2v_x.flat[i] = loc_eigvect[0, min_eig]*loc_eigv[min_eig]
        eig2v_y.flat[i] = loc_eigvect[1, min_eig]*loc_eigv[min_eig]
    #storing
    if raw :
        return eig1, eig2, (eig1v_x, eig1v_y), (eig2v_x, eig2v_y)
    else:
        eig1_sf = ScalarField()
        eig1_sf.import_from_arrays(field.axe_x, field.axe_y, eig1,
                                  mask=mask, unit_x=field.unit_x,
                                  unit_y=field.unit_y,
                                  unit_values="")
        eig2_sf = ScalarField()
        eig2_sf.import_from_arrays(field.axe_x, field.axe_y, eig2,
                                  mask=mask, unit_x=field.unit_x,
                                  unit_y=field.unit_y,
                                  unit_values="")
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
        return eig1_sf, eig2_sf, eig1_vf, eig2_vf

def get_grad_field(field, direction=1):
    """
    Return a field based on original field gradients.
    (V = dV/dx, Vy = DV/Vy)

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
    grads = get_gradient(field)
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
    mask = vf.mask
    deltaabs = delta * ((axe_x[-1]-axe_x[0])/len(axe_x)
                        + (axe_y[-1]-axe_y[0])/len(axe_y))/2.
    deltaabs2 = deltaabs**2
    # check if there are masked values
    if np.any(mask):
        raise Exception()
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
        alpha = 1
        # calcul d'une streamline
        while True:
            tmp_vx = interp_vx(stream[i-1, 0], stream[i-1, 1])[0, 0]
            tmp_vy = interp_vy(stream[i-1, 0], stream[i-1, 1])[0, 0]
            norm = np.linalg.norm([tmp_vx, tmp_vy])
            # tests d'arret
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
        return streams[0]
    else:
        return streams


#def get_tracklines(vf, xy, delta=.25, interp='linear',
#                   reverse_direction=False, direction=1, smooth=0):
#    """
#    Return a tuples of Points object representing the trackline begining
#    at the points specified in xy.
#
#    Parameters
#    ----------
#    vf : VectorField or VelocityField object
#        Field on which compute the tracklines.
#    xy : tuple
#        Tuple containing each starting point for streamline.
#    delta : number, optional
#        Spatial discretization of the tracklines,
#        relative to a the spatial discretization of the field.
#    interp : string, optional
#        Used interpolation for trackline computation.
#        Can be 'linear'(default) or 'cubic'
#    direction : '1' or '2'
#        Direction of gradients to consider (see 'get_grad_field' doc).
#    smooth : number, optional
#        Optional smooth for noisy fields.
#    """
#    # checking parameters
#    if not isinstance(smooth, NUMBERTYPES):
#        raise TypeError()
#    if not smooth >= 0:
#        raise ValueError()
#    # getting grad field
#    gfield = get_grad_field(vf, direction)
#    if smooth != 0:
#        gfield.smooth('gaussian', size=smooth)
#    lines = get_streamlines(gfield, xy, delta=delta, interp=interp,
#                            reverse_direction=reverse_direction)
#    return lines


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


def get_vorticity(vf, raw=False):
    """
    Return a scalar field with the z component of the vorticity.

    Parameters
    ----------
    vf : VectorField or Velocityfield
        Field on which compute shear stress
    raw : boolean, optional
        If 'True', return an arrays,
        if 'False' (default), return a ScalarField object.
    """
    if not isinstance(vf, VectorField):
        raise TypeError()
    tmp_vf = vf.copy()
    tmp_vf.fill(crop_border=True)
    axe_x, axe_y = tmp_vf.axe_x, tmp_vf.axe_y
    comp_x, comp_y = tmp_vf.comp_x, tmp_vf.comp_y
    mask = tmp_vf.mask
    dx = axe_x[1] - axe_x[0]
    dy = axe_y[1] - axe_y[0]
    _, Exy = np.gradient(comp_x, dx, dy)
    Eyx, _ = np.gradient(comp_y, dx, dy)
    vort = Eyx - Exy
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


def get_swirling_strength(vf, raw=False):
    """
    Return a scalar field with the swirling strength
    (imaginary part of the eigenvalue of the velocity laplacian matrix)

    Parameters
    ----------
    vf : VectorField or Velocityfield
        Field on which compute shear stress
    raw : boolean, optional
        If 'True', return an arrays,
        if 'False' (default), return a ScalarField object.
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
        # TODO: implémenter unité
        unit_values = ""
        tmp_sf = ScalarField()
        tmp_sf.import_from_arrays(axe_x, axe_y, swst, mask=mask,
                                  unit_x=unit_x, unit_y=unit_y,
                                  unit_values=unit_values)
        return tmp_sf


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
