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


def get_gradient(field):
    """
    Return ScalarFields object with gradients along x and y.

    Parameters
    ----------
    vf : VelocityField or ScalarField object
        Field to comput gradient from.

    Returns
    -------
    grad : tuple of ScalarField
        For VectorField : (dVx/dx, dVx/dy, dVy/dx, dVy/dy),
        for ScalarField : (dV/dx, dV/dy).
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
                                   ,dx, dy)
        Vy_dx, Vy_dy = np.gradient(np.ma.masked_array(field.comp_y, field.mask)
                                   ,dx, dy)
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
    if not isinstance(vf, VectorField):
        raise TypeError()
    axe_x, axe_y = vf.axe_x, vf.axe_y
    # checking parameters coherence
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
    # interpolation lineaire du champ de vitesse
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
        x = coord[0]
        y = coord[1]
        x = float(x)
        y = float(y)
        # si le points est en dehors, on ne fait rien
        if x < axe_x[0] or x > axe_x[-1] or y < axe_y[0] or y > axe_y[-1]:
            continue
        # calcul d'une streamline
        stream = np.zeros((longmax, 2))
        stream[0, :] = [x, y]
        i = 1
        while True:
            tmp_vx = interp_vx(stream[i-1, 0], stream[i-1, 1])[0, 0]
            tmp_vy = interp_vy(stream[i-1, 0], stream[i-1, 1])[0, 0]
            # tests d'arret
            if i >= longmax-1:
                break
            if i > 15:
                norm = [stream[0:i-10, 0] - stream[i-1, 0],
                        stream[0:i-10, 1] - stream[i-1, 1]]
                norm = norm[0]**2 + norm[1]**2
                if any(norm < deltaabs2/2):
                    break
            # calcul des dx et dy
            norm = (tmp_vx**2 + tmp_vy**2)**(.5)
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
        stream = stream[:i]
        pts = Points(stream, unit_x=unit_x, unit_y=unit_y,
                     name='streamline at x={:.3f}, y={:.3f}'.format(x, y))
        streams.append(pts)
    if len(streams) == 0:
        return None
    elif len(streams) == 1:
        return streams[0]
    else:
        return streams


def get_tracklines(vf, xy, delta=.25, interp='linear',
                   reverse_direction=False):
    """
    Return a tuples of Points object representing the trackline begining
    at the points specified in xy.
    A trackline follow the general direction of the vectorfield
    (as a streamline), but favor the small velocity. This behavior allow
    the track following.

    Parameters
    ----------
    vf : VectorField or VelocityField object
        Field on which compute the tracklines.
    xy : tuple
        Tuple containing each starting point for streamline.
    delta : number, optional
        Spatial discretization of the tracklines,
        relative to a the spatial discretization of the field.
    interp : string, optional
        Used interpolation for trackline computation.
        Can be 'linear'(default) or 'cubic'
    """
    if not isinstance(vf, VectorField):
        raise TypeError()
    if not isinstance(xy, ARRAYTYPES):
        raise TypeError("'xy' must be a tuple of arrays")
    xy = np.array(xy, dtype=float)
    if xy.shape == (2,):
        xy = [xy]
    elif len(xy.shape) == 2 and xy.shape[1] == 2:
        pass
    else:
        raise ValueError("'xy' must be a tuple of arrays")
    axe_x, axe_y = vf.axe_x, vf.axe_y
    Vx, Vy = vf.comp_x, vf.comp_y
    Vx = Vx.flatten()
    Vy = Vy.flatten()
    Magn = vf.magnitude.flatten()
    if not isinstance(delta, NUMBERTYPES):
        raise TypeError("'delta' must be a number")
    if delta <= 0:
        raise ValueError("'delta' must be positive")
    if delta > len(axe_x) or delta > len(axe_y):
        raise ValueError("'delta' is too big !")
    deltaabs = delta * ((axe_x[-1]-axe_x[0])/len(axe_x)
                        + (axe_y[-1]-axe_y[0])/len(axe_y))/2
    deltaabs2 = deltaabs**2
    if not isinstance(interp, STRINGTYPES):
        raise TypeError("'interp' must be a string")
    if not (interp == 'linear' or interp == 'cubic'):
        raise ValueError("Unknown interpolation method")
    # interpolation lineaire du champ de vitesse
    a, b = np.meshgrid(axe_x, axe_y)
    a = a.flatten()
    b = b.flatten()
    pts = zip(a, b)
    if interp == 'linear':
        interp_vx = spinterp.LinearNDInterpolator(pts, Vx.flatten())
        interp_vy = spinterp.LinearNDInterpolator(pts, Vy.flatten())
        interp_magn = spinterp.LinearNDInterpolator(pts, Magn.flatten())
    elif interp == 'cubic':
        interp_vx = spinterp.CloughTocher2DInterpolator(pts, Vx.flatten())
        interp_vy = spinterp.CloughTocher2DInterpolator(pts, Vy.flatten())
        interp_magn = spinterp.CloughTocher2DInterpolator(pts,
                                                          Magn.flatten())
    # Calcul des streamlines
    streams = []
    longmax = (len(axe_x)+len(axe_y))/delta
    for coord in xy:
        x = coord[0]
        y = coord[1]
        x = float(x)
        y = float(y)
        # si le points est en dehors, on ne fait rien
        if x < axe_x[0] or x > axe_x[-1] or y < axe_y[0] or y > axe_y[-1]:
            continue
        # calcul d'une streamline
        stream = np.zeros((longmax, 2))
        stream[0, :] = [x, y]
        i = 1
        while True:
            tmp_vx = interp_vx(stream[i-1, 0], stream[i-1, 1])
            tmp_vy = interp_vy(stream[i-1, 0], stream[i-1, 1])
            # tests d'arret
            if np.isnan(tmp_vx) or np.isnan(tmp_vy):
                break
            if i >= longmax-1:
                break
            if i > 15:
                norm = [stream[0:i-10, 0] - stream[i-1, 0],
                        stream[0:i-10, 1] - stream[i-1, 1]]
                norm = norm[0]**2 + norm[1]**2
                if any(norm < deltaabs2/2):
                    break
            # searching the lower velocity in an angle of pi/2 in the
            # flow direction
            x_tmp = stream[i-1, 0]
            y_tmp = stream[i-1, 1]
            norm = (tmp_vx**2 + tmp_vy**2)**(.5)
            #    finding 10 possibles positions in the flow direction
            angle_op = np.pi/4
            theta = np.linspace(-angle_op/2, angle_op/2, 10)
            e_x = tmp_vx/norm
            e_y = tmp_vy/norm
            dx = deltaabs*(e_x*np.cos(theta) - e_y*np.sin(theta))
            dy = deltaabs*(e_x*np.sin(theta) + e_y*np.cos(theta))
            #    finding the step giving the lower velocity
            magn = interp_magn(x_tmp + dx, y_tmp + dy)
            ind_low_magn = np.argmin(magn)
            #    getting the final steps (ponderated)
            theta = theta[ind_low_magn]/2.
            dx = deltaabs*(e_x*np.cos(theta) - e_y*np.sin(theta))
            dy = deltaabs*(e_x*np.sin(theta) + e_y*np.cos(theta))
            # calcul des dx et dy
            stream[i, :] = [x_tmp + dx, y_tmp + dy]
            i += 1
        stream = stream[:i-1]
        pts = Points(stream, unit_x=vf.unit_x, unit_y=vf.unit_y,
                     name='streamline at x={:.3f}, y={:.3f}'.format(x, y))
        streams.append(pts)
    if len(streams) == 0:
        return None
    elif len(streams) == 1:
        return streams[0]
    else:
        return streams


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
