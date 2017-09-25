# -*- coding: utf-8 -*-
#!/bin/env python3

# Copyright (C) 2003-2007 Gaby Launay

# Author: Gaby Launay  <gaby.launay@tutanota.com>
# URL: https://framagit.org/gabylaunay/IMTreatment
# Version: 1.0

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
import matplotlib.pyplot as plt

import IMTreatment.vortex_detection as vod
import IMTreatment.pod as pod
from IMTreatment import ScalarField, TemporalVectorFields, VectorField


def message(mess, level=0):
    text = '=== ' + '  '*level + mess
    print(text)


def scalarfield():
    message('Creating a scalar field')
    shape = (40, 30)
    axe_x = np.linspace(-12, 38, shape[0])
    axe_y = np.linspace(0, 35, shape[1])
    values = (np.random.rand(*shape) - .5)*14
    sf = ScalarField()
    sf.import_from_arrays(axe_x=axe_x, axe_y=axe_y,
                          values=values, mask=False,
                          unit_x='m', unit_y='m',
                          unit_values='m/s')
    sf.smooth(tos='gaussian', size=2, inplace=True)
    raw_sf = sf.copy()

    message('Displaying the scalar field')
    fig, axs = plt.subplots(2, 2)
    plt.suptitle('Scalar field')
    plt.sca(axs[0, 0])
    sf.display()
    plt.title('As image (default)')
    plt.sca(axs[0, 1])
    sf.display(interpolation='bicubic')
    plt.title('As image (interpolated)')
    plt.sca(axs[1, 0])
    sf.display(kind='contourf')
    plt.title('Filled contour')
    plt.sca(axs[1, 1])
    sf.display(kind='contour')
    plt.title('Contour')
    plt.show()

    message('Masking some values')
    sf.mask = np.random.rand(*sf.shape) > .9
    sf.mask = np.logical_or(sf.mask, sf.values < .5*sf.min)
    plt.figure()
    sf.display()
    plt.title('Scalar field with masked values')
    plt.show()

    message('Interpolating masked values')
    sf.fill(kind='linear', inplace=True)
    plt.figure()
    sf.display()
    plt.title('Scalar field with interpolated masked values')
    plt.show()

    message('Removing a portion of the field')
    sf.crop(intervx=[-20, 20], intervy=[10, 30], inplace=True)
    plt.figure()
    sf.display()
    plt.title('Cropped')
    plt.show()

    message('Changing the units')
    sf.change_unit('x', 'km')
    sf.change_unit('y', 'km')
    sf.change_unit('values', 'km/h')
    plt.figure()
    sf.display()
    plt.title('With other units')
    plt.show()

    message('Modifying the origin')
    sf.set_origin(10, 20)
    plt.figure()
    sf.display()
    plt.plot(0, 0, 'ok')
    plt.title('With another origin')
    plt.show()

    sf = raw_sf.copy()

    message('Extracting a profile')
    profile = sf.get_profile('y', 12)
    fig, axs = plt.subplots(2, 1)
    plt.sca(axs[0])
    raw_sf.display()
    plt.axhline(12, ls=':', color='k')
    plt.sca(axs[1])
    profile.display()
    plt.title('Profile at y=12')
    plt.show()

    message('Extracting spatial spectrum')
    spec = sf.get_spatial_spectrum('x')
    fig, axs = plt.subplots(2, 1)
    plt.sca(axs[0])
    raw_sf.display()
    plt.sca(axs[1])
    spec.display()
    plt.title('Spatial spectrum along x')
    plt.show()

    message('Integrate')
    res = sf.integrate_over_surface(intervx=[0, 20], intervy=[10, 30])
    plt.figure()
    sf.display()
    plt.plot([0, 20, 20, 0, 0], [10, 10, 30, 30, 10], ls=':', color='k')
    plt.title(f'Integrate over the area: {res:.2f}')
    plt.show()

def vectorfield():
    message('Creating a vector field')
    shape = (40, 30)
    axe_x = np.linspace(-12, 38, shape[0])
    axe_y = np.linspace(0, 35, shape[1])
    vx = (np.random.rand(*shape) - .5)*14
    sf = ScalarField()
    vy = (np.random.rand(*shape) - .5)*14
    vf = VectorField()
    vf.import_from_arrays(axe_x=axe_x, axe_y=axe_y,
                          comp_x=vx, comp_y=vy, mask=False,
                          unit_x='m', unit_y='m',
                          unit_values='m/s')
    vf.smooth(tos='gaussian', size=2, inplace=True)
    raw_vf = vf.copy()

    message('Displaying the vector field')
    fig, axs = plt.subplots(2, 2)
    plt.sca(axs[0, 0])
    vf.display()
    plt.title('quiver plot (default)')
    plt.sca(axs[1, 0])
    vf.display(kind='stream')
    plt.title('Streamlines')
    plt.sca(axs[0, 1])
    vf.display('magnitude')
    plt.title('Magnitude')
    plt.sca(axs[1, 1])
    vf.display('magnitude')
    vf.display(kind='stream', color='k')
    plt.title('Magnitude + streamlines')
    plt.show()

    message('Export to scalarfield')
    vx = vf.comp_x_as_sf
    vy = vf.comp_y_as_sf
    vm = vf.magnitude_as_sf
    fig, axs = plt.subplots(2, 2)
    plt.sca(axs[0, 0])
    vf.display(kind='stream')
    plt.title('Velocity field')
    plt.sca(axs[1, 0])
    vx.display()
    plt.title('Vx')
    plt.sca(axs[0, 1])
    vy.display()
    plt.title('Vy')
    plt.sca(axs[1, 1])
    vm.display()
    plt.title('Magnitude')
    plt.show()

    message('Other operations')
    print('    All operation that can be realized on a scalar field can'
          ' also be done on vector fields')
    fig, axs = plt.subplots(2, 2)
    plt.sca(axs[0, 0])
    tmpvf = vf.crop(intervx=[0, 20], intervy=[10, 20])
    tmpvf.display('magnitude')
    tmpvf.display(kind='stream', color='k')
    plt.title('Cropped')
    plt.sca(axs[0, 1])
    tmpvf = vf.copy()
    tmpvf.set_origin(20, 10)
    tmpvf.display('magnitude')
    tmpvf.display(kind='stream', color='k')
    plt.plot([0], [0], 'ok')
    plt.title('With modified origin')
    plt.sca(axs[1, 0])
    tmpvf = vf.copy()
    tmpvf.mask = tmpvf.magnitude < tmpvf.max*.2
    tmpvf.display('magnitude')
    tmpvf.display(kind='stream', color='k')
    plt.axhline(20, ls=':', color='r')
    plt.title('With masked values')
    plt.sca(axs[1, 1])
    prof = tmpvf.get_profile('vx', 'y', 20)
    prof.display()
    plt.title('Profile of Vx at y=20')
    plt.show()


def temporalvectorfield():
    message('Creating a series of vector field')
    shape = (40, 30)
    axe_x = np.linspace(-12, 38, shape[0])
    axe_y = np.linspace(0, 35, shape[1])
    vx = (np.random.rand(*shape) - .5)*14
    sf = ScalarField()
    vy = (np.random.rand(*shape) - .5)*14
    vf = VectorField()
    vf.import_from_arrays(axe_x=axe_x, axe_y=axe_y,
                          comp_x=vx, comp_y=vy, mask=False,
                          unit_x='m', unit_y='m',
                          unit_values='m/s')
    vf.smooth(tos='gaussian', size=2, inplace=True)
    tvf = TemporalVectorFields()
    dt = 0.1
    for i in range(20):
        tmpvf = vf.copy()
        tmpvf *= (1 + i/20)
        tmpvf += np.cos(i/10*np.pi)
        tmpvf.comp_x += i/20*tmpvf.max/3
        tvf.add_field(tmpvf, i*dt, 's')

    message('Displaying the vector field series')
    plt.figure()
    tvf.display(scale=50)
    plt.title('Vector field series')
    plt.show()

    message('Computing mean field')
    mean = tvf.get_mean_field()
    plt.figure()
    mean.display('magnitude')
    mean.display(kind='stream', color='k')
    plt.title('Mean field')
    plt.show()

    message('Getting fluctuant fields')
    fluc = tvf.get_fluctuant_fields()
    plt.figure()
    fluc.display()
    plt.title('Fluctuation fields')
    plt.show()

    message('Computing mean turbulent kinetic energy')
    mean = tvf.get_mean_tke()
    plt.figure()
    mean.display()
    plt.title('Mean TKE')
    plt.show()

    message('Computing Reynolds stress')
    mean = tvf.get_mean_field()
    rsxx, rsyy, rsxy = tvf.get_reynolds_stress()
    fig, axs = plt.subplots(2, 2)
    plt.sca(axs[0, 0])
    mean.display()
    plt.title('Mean field')
    plt.sca(axs[0, 1])
    rsxy.display()
    plt.title('Reynolds stress xy')
    plt.sca(axs[1, 0])
    rsxx.display()
    plt.title('Reynolds stress xx')
    plt.sca(axs[1, 1])
    rsyy.display()
    plt.title('Reynolds stress yy')
    plt.show()

    message('Extracting time profile')
    mean = tvf.get_mean_field()
    prof = tvf.get_time_profile('comp_x', [5, 10])
    fig, axs = plt.subplots(2, 1)
    plt.sca(axs[0])
    mean.display()
    plt.plot([5], [10], 'ok')
    plt.title('Mean field')
    plt.sca(axs[1])
    prof.display()
    plt.title('Time evolution of Vx at the point (5, 10)')
    plt.show()

    message('Extracting components')
    tsf = tvf.Vx_as_sf
    plt.figure()
    tsf.display()
    plt.title('Vx fields')
    plt.show()

    message('Getting temporal spectrum')
    prof = tvf.get_time_profile('comp_x', [5, 10])
    spec = prof.get_spectrum()
    plt.figure()
    spec.display()
    plt.title('Velocity spectrum at (5, 10)')
    plt.show()

def pod():
    print('Not implemented yet')

def vortex_detection():
    message('Creating a series of vector field')
    shape = (40, 30)
    axe_x = np.linspace(-12, 38, shape[0])
    axe_y = np.linspace(0, 35, shape[1])
    vx = (np.random.rand(*shape) - .5)*14
    sf = ScalarField()
    vy = (np.random.rand(*shape) - .5)*14
    vf = VectorField()
    vf.import_from_arrays(axe_x=axe_x, axe_y=axe_y,
                          comp_x=vx, comp_y=vy, mask=False,
                          unit_x='m', unit_y='m',
                          unit_values='m/s')
    vf.smooth(tos='gaussian', size=2, inplace=True)
    tvf = TemporalVectorFields()
    dt = 0.1
    for i in range(20):
        tmpvf = vf.copy()
        tmpvf *= (1 + i/40)
        tmpvf += np.cos(i/20*np.pi)
        tmpvf.comp_x += i/40*tmpvf.max/3
        tvf.add_field(tmpvf, i*dt, 's')

    message('Detecting vortex (can be quite long due to streamlines display)')
    cp = vod.get_critical_points(tvf, kind='pbi')
    plt.figure()
    cp.display(fields=tvf)
    tvf.display('magnitude')
    tvf.display(kind='stream', color='k')
    plt.show()
