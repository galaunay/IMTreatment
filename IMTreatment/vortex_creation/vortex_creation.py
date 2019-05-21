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
from ..core import VectorField, TemporalVectorFields, Points
from ..utils import make_unit
from ..utils.types import NUMBERTYPES, ARRAYTYPES
import copy
import matplotlib.pyplot as plt
from .. import boundary_layer as imtbl
from .. import potential_flow as pf
import scipy.interpolate as spinterp


class Vortex(object):
    """
    """

    def __init__(self, x0, y0, movable, idi=0):
        """

        """
        self.__x0 = x0
        self.__y0 = y0
        self.movable = movable
        self.rot_dir = 1
        self.id = idi

    @property
    def x0(self):
        return self.__x0

    @x0.setter
    def x0(self, new_x0):
        if not self.movable:
            raise Exception()
        self.__x0 = new_x0

    @property
    def y0(self):
        return self.__y0

    @y0.setter
    def y0(self, new_y0):
        if not self.movable:
            raise Exception()
        self.__y0 = new_y0

    def get_vector_field(self, axe_x, axe_y, unit_x='', unit_y=''):
        # preparing
        Vx = np.empty((len(axe_x), len(axe_y)), dtype=float)
        Vy = np.empty(Vx.shape, dtype=float)
        # getting velocities
        for i, x in enumerate(axe_x):
            for j, y in enumerate(axe_y):
                Vx[i, j], Vy[i, j] = self.get_vector(x, y)
        # returning
        vf = VectorField()
        vf.import_from_arrays(axe_x, axe_y, Vx, Vy, mask=False, unit_x=unit_x,
                              unit_y=unit_y, unit_values=make_unit('m/s'))
        return vf

    @staticmethod
    def _get_theta(x0, x, y0, y):
        """
        """
        dx = x - x0
        dy = y - y0
        if dx == 0 and dy == 0:
            theta = 0
        elif dx == 0:
            if dy > 0:
                theta = np.pi/2.
            else:
                theta = -np.pi/2.
        elif dx > 0.:
            theta = np.arctan((dy)/(dx))
        else:
            theta = np.arctan((dy)/(dx)) - np.pi
        return theta

    @staticmethod
    def _get_r(x0, x, y0, y):
        return ((x - x0)**2 + (y - y0)**2)**.5

    @staticmethod
    def _cyl_to_cart(theta, comp_r, comp_phi):
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        comp_x = comp_r*c_theta - comp_phi*s_theta
        comp_y = comp_r*s_theta + comp_phi*c_theta
        return comp_x, comp_y

    def copy(self):
        return copy.deepcopy(self)

    def symetrize(self):
        """
        Return a symetrized copy of the vortex
        """
        tmp_vort = self.copy()
        tmp_vort.rot_dir *= -1
        return tmp_vort


class SolidVortex(Vortex):
    """
    Representing a solid rotation.
    """
    def __init__(self, x0=0., y0=0., omega=1., movable=True, idi=0):
        """
        Parameters
        ----------
        x0, y0 : numbers, optional
            Position of the vortex center (default : [0, 0]).
        omega : number, optional
            rotation velocity (rad/s)
        """
        super(SolidVortex, self).__init__(x0, y0, movable=movable, idi=idi)
        if omega < 0:
            self.rot_dir = -1
        else:
            self.rot_dir = 1
        self.omega = np.abs(omega)

    def get_vector(self, x, y):
        """
        Return the velocity vector at the given point.
        """
        # compute r
        r = self._get_r(self.x0, x, self.y0, y)
        # compute theta
        theta = self._get_theta(self.x0, x, self.y0, y)
        # compute velocity in cylindrical referentiel
        Vr = 0.
        Vphi = r*self.omega*self.rot_dir
        # get velocity in the cartesian refenrentiel
        Vx, Vy = self._cyl_to_cart(theta, Vr, Vphi)
        # returning
        return Vx, Vy


class FreeVortex(Vortex):
    """
    Representing a Free (irrotational) Vortex.
    Due to its definition, the center of the vortex is a singular point
    (V = inf) set to 0 in this implementation.
    """
    def __init__(self, x0=0., y0=0., gamma=1., movable=True, idi=0):
        """
        Parameters
        ----------
        x0, y0 : numbers, optional
            Position of the vortex center (default : [0, 0]).
        gamma : number, optional
            Cirdculation of the free-vortex (m^2/s).
        """
        super(FreeVortex, self).__init__(x0, y0, movable=movable, idi=idi)
        if gamma < 0:
            self.rot_dir = -1
        else:
            self.rot_dir = 1
        self.gamma = np.abs(gamma)

    def get_vector(self, x, y):
        """
        Return the velocity vector at the given point.
        """
        # compute r
        r = self._get_r(self.x0, x, self.y0, y)
        # compute theta
        theta = self._get_theta(self.x0, x, self.y0, y)
        # compute velocity in cylindrical referentiel
        Vr = 0.
        if r == 0:
            Vphi = 0
        else:
            Vphi = self.gamma/(2*np.pi*r)*self.rot_dir
        # get velocity in the cartesian refenrentiel
        Vx, Vy = self._cyl_to_cart(theta, Vr, Vphi)
        # returning
        return Vx, Vy


class BurgerVortex(Vortex):
    """
    Representing a Burger Vortex, a stationnary self-similar flow, caused by
    the balance between vorticity creation at the center and vorticity
    diffusion.

    Notes
    -----
    Analytical Vortex Solutions to the Navier-Stokes Equation.
    Thesis for the degree of Doctor of Philosophy, Växjö University,
    Sweden 2007.
    """
    def __init__(self, x0=0., y0=0., alpha=1e-6, ksi=1., viscosity=1e-6,
                 movable=True, idi=0):
        """
        Parameters
        ----------
        x0, y0 : numbers, optional
            Position of the vortex center (default : [0, 0]).
        alpha : number, optional
            Positive constant (default : 1e-6), determine te size of the
            vortex. To have a vortex of radius 'R', 'alpha' should be equal to
            '9.2*viscosity/R**2'.
        ksi : number, optional
            Constant (default : 1.), make the overall velocity augment.
            Can also be used to switch the rotation direction.
        viscosity : number, optional
            Viscosity (default : 1e-6 (water))
        """
        super(BurgerVortex, self).__init__(x0, y0, movable=movable, idi=idi)
        if ksi < 0:
            self.rot_dir = -1
        else:
            self.rot_dir = 1
        self.ksi = np.abs(ksi)
        self.alpha = alpha
        self.viscosity = viscosity
        self.C1 = -self.alpha/(4.*self.viscosity)
        self.C2 = self.ksi/(2*np.pi)
        self.C3 = -1/2.*self.alpha

    def get_vector(self, x, y):
        """
        Return the velocity vector at the given point.
        """
        # compute r
        r = self._get_r(self.x0, x, self.y0, y)
        # compute theta
        theta = self._get_theta(self.x0, x, self.y0, y)
        # compute velocity in clyndrical referentiel
        Vr = self.C3*r
        if r == 0:
            Vphi = 0
        else:
            Vphi = (self.C2/r) \
                * (1 - np.exp(self.C1*r**2))*self.rot_dir
        # get velocity in the cartesian refenrentiel
        Vx, Vy = self._cyl_to_cart(theta, Vr, Vphi)
        # returning
        return Vx, Vy


class HillVortex(Vortex):
    """
    Representing a Hill Vortex, a convected vortex sphere in a inviscid flow.

    Notes
    -----
    Analytical Vortex Solutions to the Navier-Stokes Equation.
    Thesis for the degree of Doctor of Philosophy, Växjö University,
    Sweden 2007.
    """
    def __init__(self, x0=0, y0=0, U=1., rv=1., rot_dir=1, unit_values='',
                 movable=True, idi=0):
        """
        Parameters
        ----------
        x0, y0 : numbers, optional
            Position of the vortex center (default : [0, 0]).
        U : number
            Convection velocity (m/s)
        rv : number
            Vortex radius
        """
        super(HillVortex, self).__init__(x0, y0, movable=movable, idi=idi)
        self.U = U
        self.rv = rv
        self.rot_dir = rot_dir

    def get_vector(self, x, y):
        """
        Return the velocity vector at the given point.
        """
        # compute r
        r = self._get_r(self.x0, x, self.y0, y)
        # compute theta
        theta = self._get_theta(self.x0, x, self.y0, y)
        # compute velocity in clyndrical referentiel
        Vr = -3./4.*self.U*r*(1 - r**2/self.rv**2) \
            * 2*np.sin(theta)*np.cos(theta)
        Vphi = 3./2.*self.U*np.sin(theta)**2 \
            * r*(1 - 2*r**2/self.rv**2)*self.rot_dir
        # get velocity in the cartesian refenrentiel
        Vx, Vy = self._cyl_to_cart(theta, Vr, Vphi)
        # returning
        return Vx, Vy


class LambOseenVortex(Vortex):
    """
    Representing a Lamb-Oseen Vortex, a vortex with decay due to viscosity.
    (satisfy NS)

    Notes
    -----
    Analytical Vortex Solutions to the Navier-Stokes Equation.
    Thesis for the degree of Doctor of Philosophy, Växjö University,
    Sweden 2007.
    """
    def __init__(self, x0=0, y0=0, ksi=1., t=1., viscosity=1e-6, movable=True,
                 idi=0):
        """
        Parameters
        ----------
        x0, y0 : numbers, optional
            Position of the vortex center (default : [0, 0]).
        ksi : number
            Overall velocity factor.
        t : number
            Time.
        viscosity : number
            Viscosity (default : 1e-6 (water))
        """
        super(LambOseenVortex, self).__init__(x0, y0, movable=movable, idi=idi)
        if ksi < 0:
            self.rot_dir = -1
        else:
            self.rot_dir = 1
        self.ksi = np.abs(ksi)
        self.t = t
        self.viscosity = viscosity

    def get_vector(self, x, y):
        """
        Return the velocity vector at the given point.
        """
        # compute r
        r = self._get_r(self.x0, x, self.y0, y)
        # compute theta
        theta = self._get_theta(self.x0, x, self.y0, y)
        # compute velocity in clyndrical referentiel
        Vr = 0.
        if r == 0:
            Vphi = 0
        else:
            Vphi = self.ksi/(2*np.pi*r)\
                * (1 - np.exp(-r**2/(4.*self.viscosity*self.t)))*self.rot_dir
        # get velocity in the cartesian refenrentiel
        Vx, Vy = self._cyl_to_cart(theta, Vr, Vphi)
        # returning
        return Vx, Vy


class RankineVortex(Vortex):
    """
    Representing a Rankine Vortex, with an inner zone or forced vortex, and
    an outer zone of free vortex.

    Notes
    -----
    Giaiotti, DARIO B., et FULVIO Stel. « The Rankine vortex model ».
    PhD course on Environmental Fluid Mechanics-ICTP/University of Trieste,
    2006.

    """
    def __init__(self, x0=0., y0=0., circ=1., rv=1., movable=True, idi=0):
        """
        Parameters
        ----------
        x0, y0 : numbers, optional
            Position of the vortex center (default : [0, 0]).
        rv : number
            Vortex inner zone radius
        circ : number
            Vortex circulation (m^2/s)
        unit_values : string
            Velocity unity
        """
        super(RankineVortex, self).__init__(x0, y0, movable=movable, idi=idi)
        self.rv = rv
        if circ < 0:
            self.rot_dir = -1
        else:
            self.rot_dir = 1
        self.circ = np.abs(circ)

    def get_vector(self, x, y):
        """
        Return the velocity vector at the given point.
        """
        # compute r
        r = self._get_r(self.x0, x, self.y0, y)
        # compute theta
        theta = self._get_theta(self.x0, x, self.y0, y)
        # compute velocity in clyndrical referentiel
        Vr = 0.
        if r <= self.rv:
            Vphi = self.circ*r/(2*np.pi*self.rv**2)*self.rot_dir
        else:
            Vphi = self.circ/(2*np.pi*r)*self.rot_dir
        # get velocity in the cartesian refenrentiel
        Vx, Vy = self._cyl_to_cart(theta, Vr, Vphi)
        # returning
        return Vx, Vy


class LambChaplyginVortex(Vortex):
    """
    Representing a Lamb-Chaplygin dipole vortex, with potential flow in the
    exterior region, and a linear relation between stream function and
    vorticity in the inner region.

    Notes
    -----
    Analytical Vortex Solutions to the Navier-Stokes Equation.
    Thesis for the degree of Doctor of Philosophy, Växjö University,
    Sweden 2007.
    """
    def __init__(self, x0=0, y0=0, U=1., rv=1., Bessel_root_nmb=1,
                 movable=True, idi=0):
        """
        Parameters
        ----------
        x0, y0 : numbers, optional
            Position of the vortex center (default : [0, 0]).
        U : number
            Convection velocity (m/s)
        rv : number
            Delimitation radius between interior and exterior (m).
        Bessel_root_nmb : integer
            Bessel root number evaluated to choose the constant k

        """
        super(LambChaplyginVortex, self).__init__(x0, y0, movable=movable,
                                                  idi=idi)
        self.U = U
        self.rv = rv
        self.Bessel_root_nmb = Bessel_root_nmb

    def get_vector(self, x, y):
        """
        Return the velocity vector at the given point.
        """
        from scipy.special import jn, jn_zeros
        # compute r
        r = self._get_r(self.x0, x, self.y0, y)
        # compute theta
        theta = self._get_theta(self.x0, x, self.y0, y)
        # compute Bessel root number
        k = jn_zeros(1, self.Bessel_root_nmb)[-1]/self.rv
        # compute stream function
        if r < self.rv:
            J1_p = (jn(2, k*r) - jn(0, k*r))/(-2)
            Vr = -2*self.U/(r*k*jn(0, k*self.rv))*jn(1, k*r)*np.cos(theta)
            Vphi = 2*self.U/(jn(0, k*self.rv))*np.sin(theta)*J1_p
        else:
            Vr = -self.U*(r - self.rv**2/r)*np.cos(theta)
            Vphi = +self.U*(1 + self.rv**2/r**2)*np.sin(theta)
        # get velocity in the cartesian refenrentiel
        Vx, Vy = self._cyl_to_cart(theta, Vr, Vphi)
        # returning
        return Vx, Vy

    def J(self, order, x):
        """
        Return the value of the Bessel function with the given order ar the
        given point.

        Parameters
        ----------
        order : number
            Order of the Bessel function
        x : number
            Value where we want the Bessel function evaluation.

        Returns
        -------
        y : number
            Bessel function value at 'x'
        """
        from scipy.special import jn
        return jn(order, x)


class PersoVortex(Vortex):
    """
    Representing a slice of a 3D vortex.
    """
    def __init__(self, x0=0, y0=0, radius=1., vort_max=None, circ=None,
                 node_ratio=0., nu=1e-6, movable=True, idi=0):
        """
        Representing a slice of a 3D vortex.

        Notes
        -----
        Modified Burger vortex :

        .. math::
            u_{\\theta} = \\frac{\\Gamma}{2\\pi r} \\left[ 1 - \\exp
            \\left( -\\frac{\\alpha r^2}{4 \\nu} \\right) \\right]

            u_r = -\\frac{\\alpha r}{2}

            u_z = \\alpha z

        respectant les propriétés suivantes (vorticité et rayon à i%) :

        .. math::
            \\omega = \\frac{\\alpha \\Gamma}{4 \\pi \\nu} \\exp \\left(
            -\\frac{\\alpha r^2}{4 \\nu} \\right)

            R_{i} = \\left( -\\frac{16 \\ln(i)\\nu}{\\alpha} \\right)^{0.5}

        Parameters
        ----------
        x0, y0 : numbers, optional
            Position of the vortex center (default : [0, 0]).
        """
        # check
        alpha = nu/(radius/2)**2
        alpha = 16*np.log(2)*nu/radius**2
        if circ is None:
            if vort_max is None:
                raise ValueError()
            circ = 2*vort_max*4*np.pi*nu/(alpha)
        # store
        super(PersoVortex, self).__init__(x0, y0, movable=movable, idi=idi)
        self.radius = radius
        self.circ = circ
        self.node_ratio = node_ratio
        self.nu = nu
        self.alpha = alpha
        # compute vortex constant (for fast velocity computation)
        self.C1 = -self.alpha/(4.*self.nu)
        self.C2 = self.circ/(2*np.pi)
        self.C3 = -1/2.*self.alpha

    def get_vector(self, x, y):
        """
        Return the velocity vector at the given point.
        """
        # compute r
        r = self._get_r(self.x0, x, self.y0, y)
        # compute theta
        theta = self._get_theta(self.x0, x, self.y0, y)
        # compute velocity in clyndrical referentiel
        if r == 0:
            Vphi = 0
        else:
            Vphi = (self.C2/r) \
                * (1 - np.exp(self.C1*r**2))*self.rot_dir
        Vr = -np.abs(Vphi)*self.node_ratio
        # get velocity in the cartesian refenrentiel
        Vx, Vy = self._cyl_to_cart(theta, Vr, Vphi)
        # returning
        return Vx, Vy

class CustomField(object):
    """
    Representing a custom field.

    Parameters
    ----------
    funct : function
        Representing the field.
        Has to take (x, y) as input and return (Vx, Vy).
    """
    def __init__(self, funct, unit_values='', idi=0):
        # check params
        try:
            Vx, Vy = funct(1., 1.)
        except TypeError:
            raise TypeError()
        else:
            if (not isinstance(Vx, NUMBERTYPES)
                    or not isinstance(Vy, NUMBERTYPES)):
                raise TypeError()
        # store
        self.funct = funct
        self.unit_values = unit_values
        self.id = idi

    def copy(self):
        """
        """
        return copy.deepcopy(self)

    def get_vector(self, x, y):
        """
        """
        return self.funct(x, y)

    def get_vector_field(self, axe_x, axe_y, unit_x='', unit_y=''):
        # preparing
        Vx = np.empty((len(axe_x), len(axe_y)), dtype=float)
        Vy = np.empty(Vx.shape, dtype=float)
        mask = np.empty(Vx.shape, dtype=bool)
        # getting velocities
        for i, x in enumerate(axe_x):
            for j, y in enumerate(axe_y):
                Vx[i, j], Vy[i, j] = self.funct(x, y)
        mask = np.logical_or(np.isnan(Vx), np.isnan(Vy))
        # returning
        vf = VectorField()
        vf.import_from_arrays(axe_x, axe_y, Vx, Vy, mask=mask, unit_x=unit_x,
                              unit_y=unit_y, unit_values=self.unit_values)
        return vf


class Wall(object):
    """
    Representing a wall
    """
    def __init__(self, x=None, y=None):
        """
        Representing a wall

        Parameters
        ----------
        x, y : numbers
            Position(s) of the wall along 'x' and 'y'.
        """
        # check
        if x is not None:
            self.direction = 'x'
            self.position = x
            if y is not None:
                raise ValueError("'x' and 'y' cannot be both defined")
        elif y is not None:
            self.direction = 'y'
            self.position = y
        else:
            raise ValueError("'x' or 'y' should be defined")

    def get_symmetry(self, pt):
        """
        Give the symmetry of 'pt' according to the wall.
        """
        # check
        if not isinstance(pt, ARRAYTYPES):
            raise TypeError()
        pt = np.array(pt)
        if not pt.shape == (2,):
            raise ValueError()
        # get symmetry
        if self.direction == 'x':
            new_x = self.position + (self.position - pt[0])
            return [new_x, pt[1]]
        else:
            new_y = self.position + (self.position - pt[1])
            return [pt[0], new_y]


class VortexSystem(object):
    """
    Representing a set of vortex.
    """

    def __init__(self):
        """
        Representing a set of vortex.
        """
        self.vortex = []
        self.im_vortex = []
        self.nmb_vortex = 0
        self.walls = []
        self.custfields = []
        self.to_refresh = False
        self.curr_id = 0
        self.curr_cf_id = 0

    def copy(self):
        """
        Return a copy.
        """
        return copy.deepcopy(self)

    def add_vortex(self, vortex):
        """
        Add a vortex, or a custom field to the set.

        Parameters
        ----------
        vortex : Vortex or CustomField object
            vortex or field to add to the set
        """
        if not isinstance(vortex, Vortex):
            raise TypeError()
        vort = vortex.copy()
        vort.id = self.curr_id
        self.curr_id += 1
        self.vortex.append(vort)
        self.nmb_vortex += 1
        self.to_refresh = True

    def add_wall(self, wall):
        """
        Add a wall to the vortex system

        Parameters
        ----------
        wall : Wall object
            Wall to add.
        """
        if not isinstance(wall, Wall):
            raise TypeError()
        self.walls.append(wall)
        self.to_refresh = True

    def add_custom_field(self, custfield):
        """
        Add a custom field to the vortex system

        Parameters
        ----------
        custfield : CustomField object
            Custom field to add.
        """
        if not isinstance(custfield, CustomField):
            raise TypeError()
        custfield.id = self.curr_cf_id
        self.curr_cf_id += 1
        self.custfields.append(custfield)

    def remove_vortex(self, ind):
        """
        Remove a vortex or a custom field from the set.

        Parameters
        ----------
        ind : integer
            Vortex indice to remove.
        """
        if ind < 0:
            ind = len(self.vortex) + ind
        id_to_remove = self.vortex[ind].id
        # remove vortex
        for i in np.arange(len(self.vortex) - 1, -1, -1):
            vort = self.vortex[i]
            if vort.id == id_to_remove:
                self.vortex.pop(i)
            elif vort.id > id_to_remove:
                self.vortex[i].id -= 1
        # remove imaginary vortex
        for i in np.arange(len(self.im_vortex) - 1, -1, -1):
            vort = self.im_vortex[i]
            if vort.id == id_to_remove:
                self.im_vortex.pop(i)
            elif vort.id > id_to_remove:
                self.im_vortex[i].id -= 1
        # adapt current id
        self.curr_id -= 1

    def display(self):
        """
        Display a representation of the vortex system
        """
        # define vortx centers colors
        colors = [plt.cm.jet(i) for i in np.linspace(0, 1, len(self.vortex))]

        # refresh if necessary
        if self.to_refresh:
            self.refresh_imaginary_vortex()
        # display vortex
        for vort in self.vortex:
            color = colors[vort.id]
            plt.plot(vort.x0, vort.y0, marker='o', mec='w', mfc=color)
        # display walls
        for wall in self.walls:
            if wall.direction == 'x':
                plt.axvline(wall.position, color='k')
            else:
                plt.axhline(wall.position, color='k')
        # display imaginary vortex
        for ivort in self.im_vortex:
            color = colors[ivort.id]
            plt.plot(ivort.x0, ivort.y0, marker='o', mec='w', mfc=color)

    def display_compounded_vector(self, x, y, scale=1., detailed=False):
        """
        Display the compounded vector in 'x', 'y' position.
        """
        # get data
        colors = [plt.cm.jet(i) for i in np.linspace(0, 1, len(self.vortex))]
        vort_vects, vort_ids, cf_vects, cf_ids \
            = self.get_compounded_vector(x, y, detailed=detailed)
        norm = plt.Normalize(0, len(self.vortex))
        # loop on vortex vectors
        for i, vect in enumerate(vort_vects):
            idi = vort_ids[i]
            if i >= len(colors):
                color = 'k'
            else:
                color = colors[i]
            plt.quiver(x, y, *vect, color=color, norm=norm, scale=scale)
        # loop on custom fields
        for i, vect in enumerate(cf_vects):
            color = 'k'
            plt.quiver(x, y, *vect, color=color, norm=norm, scale=scale)

    def get_vector(self, x, y, vortex_ids='all', cf_ids='all'):
        """
        Return the resulting velocity vector, at the given point.

        Parameters
        ----------
        x, y : numbers
            Position of the wanted vector.
        vortex_ids : list of integers, 'all' or 'none'
            Can be used to filter the wanted vortex influence by ids.
        cf_ids : list of integers, 'all' or 'none'
            Can be used to filter the wanted custom fields influence by ids.
        Returns
        -------
        Vx, Vy : numbers
            Velocity components.
        """
        # check
        if vortex_ids == 'all':
            vortex_ids = np.arange(self.curr_id)
        elif vortex_ids == 'none':
            vortex_ids = []
        if cf_ids == 'all':
            cf_ids = np.arange(self.curr_cf_id)
        elif cf_ids == 'none':
            cf_ids = []
        # refesh if necessary
        if self.to_refresh:
            self.refresh_imaginary_vortex()
        # create vector
        Vx = 0.
        Vy = 0.
        # add vortex participation
        for vort in self.vortex:
            if vort.id not in vortex_ids:
                continue
            tmp_Vx, tmp_Vy = vort.get_vector(x, y)
            Vx += tmp_Vx
            Vy += tmp_Vy
        # add imaginary vortex participation
        for vort in self.im_vortex:
            if vort.id not in vortex_ids:
                continue
            tmp_Vx, tmp_Vy = vort.get_vector(x, y)
            Vx += tmp_Vx
            Vy += tmp_Vy
        # add custom fields
        for cst_field in self.custfields:
            if cst_field.id not in cf_ids:
                continue
            tmp_Vx, tmp_Vy = cst_field.get_vector(x, y)
            Vx += tmp_Vx
            Vy += tmp_Vy
        # returning
        return Vx, Vy

    def get_compounded_vector(self, x, y, detailed=False):
        """
        Return a list f vector, for each structure of the system

        Parameters
        ----------
        x, y : numbers
            Wanted vector coordinates
        detailed : boolean, optional
            If 'False' (default), vortex contribution and its imaginary vortex
            contribution are summed up. Else, return one vector for each
            contribution.

        Returns
        -------
        vects : Nx2 array of numbers
            Vectors associated to vortex contribution
        vortex_ids : Nx1 array of integer
            Associated id
        verts_cf : Mx2 array of numbers
            Vectors associated to custom fields contribution
        cf_ids : Mx1 array of integer
            Associated id
        """
        vects = []
        vects_cf = []
        if detailed:
            vortex_ids = []
            cf_ids = []
            # loop on vortex
            for vort in self.vortex:
                vects.append(vort.get_vector(x, y))
                vortex_ids.append(vort.id)
            # loop on imaginary vortex
            for vort in self.im_vortex:
                vects.append(vort.get_vector(x, y))
                vortex_ids.append(vort.id)
            # loop on custom field
            for cf in self.custfields:
                vects_cf.append(cf.get_vector(x, y))
                cf_ids.append(cf.id)
        else:
            vortex_ids = np.arange(self.curr_id)
            cf_ids = np.arange(self.curr_cf_id)
            # loop on vortex ids
            for idi in vortex_ids:
                vects.append(self.get_vector(x, y, vortex_ids=[idi],
                                             cf_ids='none'))
            # loop on cf ids
            for idi in cf_ids:
                vects_cf.append(self.get_vector(x, y, vortex_ids='none',
                                                cf_ids=[idi]))
        # return
        return vects, vortex_ids, vects_cf, cf_ids

    def refresh_imaginary_vortex(self):
        """
        """
        self.im_vortex = []
        for wall in self.walls:
            for i, ivort in enumerate(self.im_vortex[:]):
                im_vort = ivort.symetrize()
                im_vort.movable = True
                im_vort.x0, im_vort.y0 = wall.get_symmetry((ivort.x0,
                                                            ivort.y0))
                self.im_vortex.append(im_vort)
            for i, vort in enumerate(self.vortex):
                im_vort = vort.symetrize()
                im_vort.movable = True
                im_vort.x0, im_vort.y0 = wall.get_symmetry((vort.x0, vort.y0))
                self.im_vortex.append(im_vort)
        self.to_refresh = False

    def get_vector_field(self, axe_x, axe_y, unit_x='', unit_y=''):
        """
        Return a vector field on the given grid

        Parameters
        ----------
        axe_x, axe_y : arrays of crescent numbers
            x and y axis
        unit_x, unit_y : string or Unum objects
            Axis unities

        Returns
        -------
        vf :  VectorField object
            .
        """
        # check
        if not isinstance(axe_x, ARRAYTYPES):
            raise TypeError()
        if not isinstance(axe_y, ARRAYTYPES):
            raise TypeError()
        # refresh if necessary
        if self.to_refresh:
            self.refresh_imaginary_vortex()
        # get data
        axe_x = np.array(axe_x)
        axe_y = np.array(axe_y)
        vx = np.zeros((len(axe_x), len(axe_y)))
        vy = vx.copy()
        VF = VectorField()
        VF.import_from_arrays(axe_x, axe_y, vx, vy, mask=False, unit_x=unit_x,
                              unit_y=unit_y, unit_values='m/s')
        # add vortex participation
        for vort in self.vortex:
            VF += vort.get_vector_field(axe_x, axe_y,
                                        unit_x=unit_x, unit_y=unit_y)
        # add imaginary vortex participation
        for ivort in self.im_vortex:
            VF += ivort.get_vector_field(axe_x, axe_y,
                                         unit_x=unit_x, unit_y=unit_y)
        # add custom fields
        for cst_field in self.custfields:
            VF += cst_field.get_vector_field(axe_x, axe_y,
                                             unit_x=unit_x, unit_y=unit_y)
        # returning
        return VF

    def get_evolution(self, dt=1.):
        """
        Change the position of the vortex, according to the resulting velocity
        field and the time step.

        Parameters
        ----------
        dt : number
            time step.

        Returns
        -------
        vs : VortexSystem object
            New vortex system at t+dt

        """
        # check
        if not isinstance(dt, NUMBERTYPES):
            raise TypeError()
        if dt <= 0:
            raise ValueError()
        new_vs = self.copy()
        # loop on vortex
        for i, vort in enumerate(self.vortex):
            if not vort.movable:
                continue
            # get velocity on the vortex core
            Vx, Vy = self.get_vector(vort.x0, vort.y0)
            # get the vortex core dispacement
            dx, dy = dt*Vx, dt*Vy
            # change the vortex position in the new vortex system
            new_vs.vortex[i].x0 += dx
            new_vs.vortex[i].y0 += dy
        new_vs.refresh_imaginary_vortex()
        # returning
        return new_vs

    def get_temporal_vector_field(self, dt, axe_x, axe_y, nmb_it, unit_x='',
                                  unit_y='', unit_time=''):
        """
        """
        # create tvf
        time = 0.
        tmp_tvf = TemporalVectorFields()
        tmp_vf = self.get_vector_field(axe_x, axe_y, unit_x=unit_x,
                                       unit_y=unit_y)
        tmp_tvf.add_field(tmp_vf, time=time, unit_times=unit_time)
        time += dt
        # make time iterations
        tmp_vs = self.copy()
        for i in np.arange(nmb_it):
            tmp_vs = tmp_vs.get_evolution(dt=dt)
            tmp_vf = tmp_vs.get_vector_field(axe_x, axe_y, unit_x=unit_x,
                                             unit_y=unit_y)
            tmp_tvf.add_field(tmp_vf, time=time, unit_times=unit_time)
            time += dt
        # returning
        return tmp_tvf

    def get_vortex_evolution(self, dt, nmb_it, unit_time=''):
        """
        """
        time = 0.
        # create storage
        tmp_pts = []
        for vort in self.vortex:
            tmp_pts.append(Points(xy=[[vort.x0, vort.y0]], v=[time],
                                  unit_v=unit_time))
        # make time iterations
        tmp_vs = self.copy()
        for i in np.arange(nmb_it):
            tmp_vs = tmp_vs.get_evolution(dt=dt)
            # add vortex positions
            for i, vort in enumerate(tmp_vs.vortex):
                tmp_pts[i].add([vort.x0, vort.y0], v=time + dt)
            time += dt
        # returning
        return tmp_pts

    def get_velocity_map(self, axe_x, axe_y, vort, unit_x='m', unit_y='m'):
        """
        Return the velocity map of vortex 'vort' on the vortex system
        (velocity that the vortex 'vort' will have if placed at different
        space position on the vortex system)
        """
        # prepare
        comp_x = np.zeros((len(axe_x), len(axe_y)))
        comp_y = np.zeros(comp_x.shape)
        tmp_vs = self.copy()
        tmp_vort = vort.copy()
        # loop on frid points
        for i, x in enumerate(axe_x):
            for j, y in enumerate(axe_y):
                tmp_vort.x0 = x
                tmp_vort.y0 = y
                tmp_vs.add_vortex(tmp_vort)
                V = tmp_vs.get_vector(x, y)
                comp_x[i, j] = V[0]
                comp_y[i, j] = V[1]
                tmp_vs.remove_vortex(-1)
        # return
        tmp_vf = VectorField()
        tmp_vf.import_from_arrays(axe_x=axe_x, axe_y=axe_y, comp_x=comp_x,
                                  comp_y=comp_y, mask=False,
                                  unit_x=unit_x, unit_y=unit_y,
                                  unit_values='m/s')
        return tmp_vf


class HsvSystem(VortexSystem):

    def __init__(self, u_D, D, h, delta, vertical_wall=True):
        super(HsvSystem, self).__init__()
        # store data
        self.u_D = u_D
        self.D = D
        self.h = h
        self.delta = delta
        # add walls
        if vertical_wall:
            self.add_wall(Wall(x=0))
        self.add_wall(Wall(y=0))
        # add bg_field
        bl_funct = self._get_bl_profile(u_D, delta, h)
        sd_funct = self._get_velocity_slowdown(D)

        def bg_field(x, y):
            Vx = bl_funct(y)*sd_funct(x)
            return Vx, 0
        self.add_custom_field(CustomField(bg_field, unit_values='m/s'))
        #

    @staticmethod
    def _get_bl_profile(u_D, delta, h):
        blas_bl = imtbl.BlasiusBL(u_D, 1e-6, 1000.)
        x_prof = blas_bl.get_x_from_delta(delta)
        prof = blas_bl.get_profile(x=x_prof, y=np.linspace(0, h, 100))
        bl_funct = spinterp.interp1d(prof.x, prof.y, kind='linear', axis=0,
                                     copy=True, bounds_error=False,
                                     fill_value=0)
        return bl_funct

    @staticmethod
    def _get_velocity_slowdown(D):
        # create potential flow
        PS = pf.System(u_inf=1., alpha=0.)
        PS.add_object(2, [[0, -D/2], [0, D/2], [D, D/2], [D, -D/2]], res=20)
        PS.objects_2D[0].inverse_normals()
        Vx, _ = PS.compute_velocity_on_line([-5*D, 0], [0, 0], res=200,
                                            raw=False, remove_solid=False)
        Vx.x -= 5*D
        sd_funct = spinterp.interp1d(Vx.x, Vx.y, kind='linear', axis=0,
                                     copy=True, bounds_error=False,
                                     fill_value=0)
        return sd_funct


class StepSystem(VortexSystem):

    def __init__(self, u_D, H, h, delta):
        super(StepSystem, self).__init__()
        # store data
        self.u_D = u_D
        self.h = h
        self.H = H
        self.delta = delta
        # add walls
        self.add_wall(Wall(x=0))
        self.add_wall(Wall(y=0))
        # add bg_field
        bl_funct = self._get_bl_profile(delta, h, u_D)
        dev_funct = self._get_velocity_deviation(H, h, u_D)

        def bg_field(x, y):
            V = dev_funct(x, y)
            V[0] *= bl_funct(y)
            bl2 = bl_funct(-x)  # TEMPORARY
            V[1] *= bl2         # TEMPORARY
            V[0] *= bl2         # TEMPORARY
            return V
        self.add_custom_field(CustomField(bg_field, unit_values='m/s'))

    @staticmethod
    def _get_bl_profile(delta, h, u_D):
        blas_bl = imtbl.BlasiusBL(u_D, 1e-6, 1000.)
        x_prof = blas_bl.get_x_from_delta(delta)
        prof = blas_bl.get_profile(x=x_prof, y=np.linspace(0, h, 100))
        bl_funct = spinterp.interp1d(prof.x, prof.y/u_D, kind='linear', axis=0,
                                     copy=True, bounds_error=False,
                                     fill_value=0)
        return bl_funct

    @staticmethod
    def _get_velocity_deviation(H, h, u_D):
        # create potential flow
        PS = pf.System(u_inf=u_D, alpha=0.)
        PS.add_object(2, [[1.*H, -H], [0, -H], [0, H], [1.*H, H]], res=20)
        PS.objects_2D[0].inverse_normals()
        axe_x = np.linspace(-3*H, 0, 31)
        axe_y = np.arange(0, h + H/10., H/10.)
        VF = PS.compute_velocity_on_grid(axe_x, axe_y, remove_solid=True)
        interp_x = spinterp.interp2d(axe_x, axe_y, np.transpose(VF.comp_x),
                                     fill_value=1.)
        interp_y = spinterp.interp2d(axe_x, axe_y, np.transpose(VF.comp_y),
                                     fill_value=0)

        def dev_funct(x, y):
            return [interp_x(x, y)[0], interp_y(x, y)[0]]
        return dev_funct
