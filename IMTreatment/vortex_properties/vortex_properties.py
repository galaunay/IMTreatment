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

import scipy.ndimage.measurements as msr
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

from ..core import Points, Profile, ScalarField
from ..utils import ProgressCounter, make_unit
from ..vortex_criterions import (get_NL_residual_vorticity,
                                 get_residual_vorticity, get_vorticity)


def get_vortex_radius(VF, vort_center, NL_radius=None, eps_detection=0.1,
                      output_center=False, output_unit=False):
    """
    Return the radius of the given vortex, use the residual vorticity.

    Parameters
    ----------
    VF : vectorfield object
        Velocity field on which compute gamma2.
    vort_center : 2x1 array
        Approximate position of the vortex center.
    NL_radius : number, optional
        IF specified, radius used for the non-local computation of the
        gradients.
    eps_detection : number
        epsilon used to determine the edge of the vortex (default is 0.1 for
        10% of the maximum vorticity value).
    output_center : boolean, optional
        If 'True', return the associated vortex center, computed using center
        of mass algorythm.
    output_unit ; boolean, optional
        If 'True', return the associated unit.

    Returns
    -------
    radius : number
        Average radius of the vortex. If no vortex is found, 0 is returned.
    center : 2x1 array of numbers
        If 'output_center' is 'True', contain the newly computed vortex center.
    unit_radius : Unit object
        Radius unity
    """

    # getting data
    if NL_radius is None:
        vort = get_residual_vorticity(VF)
    else:
        vort = get_NL_residual_vorticity(VF, radius=NL_radius, ind=False,
                                         mask=None, raw=False)
    dx = VF.axe_x[1] - VF.axe_x[0]
    dy = VF.axe_y[1] - VF.axe_y[0]
    # getting the ~initial maximum vorticity of the zone
    ind_x = VF.get_indice_on_axe(1, vort_center[0], kind='nearest')
    ind_y = VF.get_indice_on_axe(2, vort_center[1], kind='nearest')
    max_vort = vort.values[ind_x, ind_y]
    # getting zone around point
    zones, nmb_zones = msr.label(vort.values > eps_detection*max_vort)
    zone_number = zones[ind_x, ind_y]
    # getting the real maximum vorticity
    max_vort = np.max(vort.values[zones == zone_number])
    # getting the real zones
    zones, nmb_zones = msr.label(vort.values > eps_detection*max_vort)
    zone_number = zones[ind_x, ind_y]
    # getting the zone area and radius
    area = np.sum(zones == zone_number)*dx*dy
    radius = (area/np.pi)**.5
    # getting the new center
    if output_center:
        center = np.array(msr.center_of_mass(np.abs(vort.values),
                                             zones == zone_number))
        center[0] = VF.axe_x[0] + center[0]*dx
        center[1] = VF.axe_y[0] + center[1]*dy
    # optional computed unit
    if output_unit:
        unit_radius = (VF.unit_x**2 + VF.unit_y**2)**.5
    # return
    if not output_unit and not output_center:
        return radius
    elif output_unit and not output_center:
        return radius, unit_radius
    elif not output_unit and output_center:
        return radius, center
    else:
        return radius, center, unit_radius


def get_vortex_radius_time_evolution(TVFS, traj, NL_radius=None,
                                     eps_detection=0.1,
                                     output_center=False, verbose=False):
    """
    Return the radius evolution in time for the given vortex center trajectory.

    Use the criterion $|gamma2| > 2/pi$. The returned radius is an average value
    if the vortex zone is not circular.

    Parameters
    ----------
    TVFS : TemporalField object
        Velocity field on which compute gamma2.
    traj : Points object
        Trajectory of the vortex.
    NL_radius : number, optional
        IF specified, radius used for the non-local computation of the
        gradients.
    eps_detection : number
        epsilon used to determine the edge of the vortex (default is 0.1 for
        10% of the maximum vorticity value).
    output_center : boolean, optional
        If 'True', return a Points object with associated vortex centers,
        computed using center of mass algorythm.
    verbose : boolean
        .

    Returns
    -------
    radius : Profile object
        Average radius of the vortex. If no vortex is found, 0 is returned.
    center : Points object
        If 'output_center' is 'True', contain the newly computed vortex center.
    """
    radii = np.empty((len(traj.xy),))
    if verbose:
        pg = ProgressCounter(init_mess="Begin vortex radii detection",
                             nmb_max=len(traj.xy),
                             name_things='fields',
                             perc_interv=1)
    # computing with vortex center
    if output_center:
        centers = Points(unit_x=TVFS.unit_x, unit_y=TVFS.unit_y,
                         unit_v=TVFS.unit_times)

        for i, pt in enumerate(traj):
            if verbose:
                pg.print_progress()
            # getting time and associated velocity field
            time = traj.v[i]
            field = TVFS.fields[TVFS.times == time][0]
            # getting radius and center
            rad, cent = get_vortex_radius(field, traj.xy[i],
                                          NL_radius=NL_radius,
                                          eps_detection=eps_detection,
                                          output_center=True)
            radii[i] = rad
            centers.add(cent, time)
    # computing without vortex centers
    else:
        for i, _ in enumerate(traj):
            if verbose:
                pg.print_progress()
            # getting time and associated velocity field
            time = traj.v[i]
            field = TVFS.fields[TVFS.times == time][0]
            # getting radius
            radii[i] = get_vortex_radius(field, traj.xy[i],
                                         NL_radius=NL_radius,
                                         eps_detection=eps_detection,
                                         output_center=False)
    # returning
    mask = radii == 0.
    radii_prof = Profile(traj.v, radii, mask=mask, unit_x=TVFS.unit_times,
                         unit_y=TVFS.unit_x)
    if output_center:
        return radii_prof, centers
    else:
        return radii_prof


def get_vortex_property(VF, vort_center, size_crit=None, size_crit_lim=0.1,
                        prop_crit=None, output_unit=False, verbose=False):
    """
    Return a property of a particular vortex.

    Parameters
    ----------
    VF : vectorfield object
        Base velocity field.
    vort_center : 2x1 array
        Approximate position of the vortex center.
    size_crit : function or 'value'
        Function applied to 'VF' and returning a ScalarField used to get the
        vortex area.
        (Default is residual vorticity)
        If 'value', only the value at the given point is returned.
    size_crit_lim : number
        Used to determine the size criterion interval defining the vortex area
        (i.e. the vortex area is the area around the vortex center where
        the size criterion is superior to 'size_crit_lim' times the value at
        the center)
        (Default is 0.1 (10%))
        Useless if 'size_crit='value'
    prop_crit : function
        Function applied to 'VF' and returning a ScalarField used to get the
        property value (Default is residual vorticity)
    output_unit : boolean, optional
        If 'True', return the associated unit.
    verbose : bool
        If 'True', display information and graph along computation.

    Returns
    -------
    prop : number
        Property associated to the vortex.
        (Is the integral of 'prop_crit' result on the area defined by
        'size_crit')
    """
    # default behavior
    if size_crit is None:
        size_crit = get_residual_vorticity
    if prop_crit is None:
        prop_crit = get_residual_vorticity
    # getting data
    if size_crit == "value":
        prop_crit = prop_crit(VF)
        val = prop_crit.get_value(*vort_center)
        if output_unit:
            return val, prop_crit.unit_values
        else:
            return val
    if size_crit == prop_crit:
        prop_crit = size_crit(VF)
        size_crit = np.abs(prop_crit)
    else:
        size_crit = np.abs(size_crit(VF))
        prop_crit = prop_crit(VF)
    ind_x = VF.get_indice_on_axe(1, vort_center[0], kind='nearest')
    ind_y = VF.get_indice_on_axe(2, vort_center[1], kind='nearest')
    unit_int = prop_crit.unit_values*prop_crit.unit_x*prop_crit.unit_y
    # check if value is positive at the vortex center
    if VF.magnitude[ind_x, ind_y] <= 0:
        raise ValueError()
    # get first guess vortex zone (use value at center as a predica of
    #    the maxima)
    tmp_maxi = size_crit.values[ind_x, ind_y]
    vort_zones = size_crit.values > tmp_maxi*size_crit_lim
    vort_zones_labs, _ = msr.label(vort_zones)
    lab = vort_zones_labs[ind_x, ind_y]
    # if we're outside, just give up
    if lab == 0:
        if output_unit:
            return 0, unit_int
        else:
            return 0
    # get final vortex zone with real maxima
    size_crit_maxi = np.max(size_crit.values[lab == vort_zones_labs])
    vort_zones = size_crit.values > size_crit_maxi*size_crit_lim
    vort_zones_labs, _ = msr.label(vort_zones)
    lab = vort_zones_labs[ind_x, ind_y]
    # get prop_crit integral along zone
    tmp_prop_crit = prop_crit.values.copy()
    tmp_prop_crit[lab != vort_zones_labs] = 0.
    dx = VF.axe_x[1] - VF.axe_x[0]
    dy = VF.axe_y[1] - VF.axe_y[0]
    prop = simps(simps(tmp_prop_crit, dx=dy), dx=dx)
    # verbose
    if verbose:
        tmp_vort_zone = ScalarField()
        tmp_vort_zone.import_from_arrays(VF.axe_x, VF.axe_y,
                                         values=lab == vort_zones_labs)
        fig, axs = plt.subplots(2, 1)
        plt.sca(axs[0])
        size_crit.display()
        VF.display(kind='stream', color='w')
        tmp_vort_zone._display(kind='contour', levels=[-1e30, 0.5, 1e30],
                               colors='r')
        plt.plot([], color='r', label='Detected vortex zone')
        plt.plot(*vort_center, marker='o', linestyle='none', mec='k', mfc='w',
                 label="Vortex center")
        plt.title("Vortex size detection")
        plt.legend()
        plt.sca(axs[1])
        prop_crit.display()
        VF.display(kind='stream', color='w')
        tmp_vort_zone._display(kind='contour', levels=[-1e30, 0.5, 1e30],
                               colors='r')
        plt.plot([], color='r', label='Detected vortex zone')
        plt.plot(*vort_center, marker='o', linestyle='none', mec='k', mfc='w',
                 label="Vortex center")
        plt.title("Vortex property integration on vortex zone\nProp={:.2f} {}"
                  .format(prop, unit_int.strUnit()))
        plt.legend()
    # returning
    if output_unit:
        return prop, unit_int
    else:
        return prop


def get_vortex_property_time_evolution(TVFs, vort_center_traj, size_crit=None,
                                       size_crit_lim=0.1,
                                       prop_crit=None, output_unit=False,
                                       verbose=0):
    """
    Return a property of a particular vortex.

    Parameters
    ----------
    TVFs : TemporalVectorFields object
        Base velocity fields.
    vort_center : Points object
        Approximate position of the vortex centers along times.
    size_crit : function or 'value'
        Function applied to 'VF' and returning a ScalarField used to get the
        vortex area.
        (Default is residual vorticity)
        if 'value', only return the value at point.
    size_crit_lim : number
        Used to determine the size criterion interval defining the vortex area
        (i.e. the vortex area is the area around the vortex center where
        the size criterion is superior to 'size_crit_lim' times the value at
        the center)
        (Default is 0.1 (10%))
        Useless if "size_crit='value".
    prop_crit : function
        Function applied to 'VF' and returning a ScalarField used to get the
        property value (Default is residual vorticity)
    verbose : integer
        specified the number of fields to verbosify.
        Default is 0.

    Returns
    -------
    prop : Profile object
        Evolution of the property associated with the vortex long time.
    """
    # prepare storage
    times = []
    props = []
    if verbose == 1:
        field_to_verbosify = [len(vort_center_traj)/2]
    elif verbose == 2:
        field_to_verbosify = [len(vort_center_traj)/3,
                              len(vort_center_traj)*2/3]
    else:
        field_to_verbosify = np.round(np.linspace(0, len(vort_center_traj),
                                                  verbose))
    # loop on fields
    for i in range(len(TVFs)):
        # pass if vortex center is not defined for this time
        if TVFs.times[i] not in vort_center_traj.v:
            continue
        ind_traj = np.where(vort_center_traj.v == TVFs.times[i])[0][0]
        # verbosify (or not...)
        if len(times) in field_to_verbosify:
            verbose = True
        else:
            verbose = False
        # get the wanted property
        field = TVFs.fields[i]
        vc = vort_center_traj.xy[ind_traj]
        prop, unit = get_vortex_property(VF=field, vort_center=vc,
                                         size_crit=size_crit,
                                         size_crit_lim=size_crit_lim,
                                         prop_crit=prop_crit, verbose=verbose,
                                         output_unit=True)
        times.append(TVFs.times[i])
        props.append(prop)
    # store on a Profile object
    prof_prop = Profile(x=times, y=props, unit_x=TVFs.unit_times,
                        unit_y=unit)
    # returning
    return prof_prop


def get_vortex_circulation(VF, vort_center, epsilon=0.1, output_unit=False,
                           verbose=False):
    """
    Return the circulation of the given vortex.

    $\Gamma = \int_S \omega dS$
    avec : $S$ : surface su vortex ($| \omega | > \epsilon$)

    Recirculation is representative of the swirling strength.

    Warning : integral on complex domain is complex (you don't say?),
    here is just implemented a sum of accessible values on the domain.

    Parameters
    ----------
    VF : vectorfield object
        Velocity field on which compute gamma2.
    vort_center : 2x1 array
        Approximate position of the vortex center.
    epsilon : float, optional
        Relative seuil for the vorticity integral (default is 0.1).
    output_unit : boolean, optional
        If 'True', circulation unit is returned.

    Returns
    -------
    circ : float
        Vortex virculation.
    """
    # getting data
    ind_x = VF.get_indice_on_axe(1, vort_center[0], kind='nearest')
    ind_y = VF.get_indice_on_axe(2, vort_center[1], kind='nearest')
    dx = VF.axe_x[1] - VF.axe_x[0]
    dy = VF.axe_y[1] - VF.axe_y[0]
    vort = get_vorticity(VF)
    # find omega > 0.1 zones and label them
    max_vort = np.max(np.abs(vort.values[~vort.mask]))
    vort_zone = np.abs(vort.values)/max_vort > epsilon
    vort_zone, nmb_zone = msr.label(vort_zone)
    # get wanted zone label
    lab = vort_zone[ind_x, ind_y]
    # if we are outside a zone
    if lab == 0:
        if output_unit:
            return 0., make_unit("")
        else:
            return 0.
    if verbose:
        plt.figure()
        vort.display()
        plt.plot(vort_center[0], vort_center[1], 'ok')
        plt.figure()
        plt.imshow(vort_zone == lab)
    # else, we compute the circulation
    circ = np.sum(vort.values[vort_zone == lab])*dx*dy
    # if necessary, we compute the unit
    unit_circ = vort.unit_values*VF.unit_x*VF.unit_y
    circ *= unit_circ.asNumber()
    unit_circ /= unit_circ.asNumber()
    # returning
    if output_unit:
        return circ, unit_circ
    else:
        return circ
