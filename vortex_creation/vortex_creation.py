# -*- coding: utf-8 -*-

import numpy as np
from ..core import VectorField, make_unit, NUMBERTYPES
import pdb
import copy


class Vortex(object):
    """
    """
    def get_vector_field(self, axe_x, axe_y, unit_x, unit_y):
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
                              unit_y=unit_y, unit_values=make_unit(''))
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
        return np.sqrt((x - x0)**2 + (y - y0)**2)

    @staticmethod
    def _cyl_to_cart(theta, comp_r, comp_phi):
        comp_x = comp_r*np.cos(theta) - comp_phi*np.sin(theta)
        comp_y = comp_r*np.sin(theta) + comp_phi*np.cos(theta)
        return comp_x, comp_y

    def copy(self):
        return copy.deepcopy(self)


class FreeVortex(Vortex):
    """
    Representing a Free (irrotational) Vortex.
    """
    def __init__(self, x0=0., y0=0., gamma=1.):
        """
        Parameters
        ----------
        x0, y0 : numbers, optional
            Position of the vortex center (default : [0, 0]).
        gamma : number, optional
            Cirdculation of the free-vortex.
        """
        self.x0 = x0
        self.y0 = y0
        self.gamma = gamma

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
            Vphi = self.gamma/(2*np.pi*r)
        # get velocity in the cartesian refenrentiel
        Vx, Vy = self._cyl_to_cart(theta, Vr, Vphi)
        # returning
        return Vx, Vy


class BurgerVortex(Vortex):
    """
    Representing a Burger Vortex, a stationnary self-similar flow, caused by
    the balance between vorticity creation at the center and vorticity
    diffusion.
    """
    def __init__(self, x0=0., y0=0., alpha=1e-6, ksi=1., viscosity=1e-6):
        """
        Parameters
        ----------
        x0, y0 : numbers, optional
            Position of the vortex center (default : [0, 0]).
        alpha : number, optional
            Positive constant (default : 1e-6), low value for big vortex.
        ksi : number, optional
            Constant (default : 1.), make the overall velocity augment.
            Can also be used to switch the rotation direction.
        viscosity : number, optional
            Viscosity (default : 1e-6 (water))
        """
        self.x0 = x0
        self.y0 = y0
        self.alpha = alpha
        self.ksi = ksi
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
        Vr = -1./.2*self.alpha*r
        if r == 0:
            Vphi = 0
        else:
            Vphi = self.ksi/(2*np.pi*r)\
                * (1 - np.exp(-self.alpha*r**2/(4.*self.viscosity)))
        # get velocity in the cartesian refenrentiel
        Vx, Vy = self._cyl_to_cart(theta, Vr, Vphi)
        # returning
        return Vx, Vy


class HillVortex(Vortex):
    """
    Representing a Hill Vortex, a convected vortex sphere.
    """
    def __init__(self, x0=0, y0=0, U=1., rv=1.):
        """
        Parameters
        ----------
        x0, y0 : numbers, optional
            Position of the vortex center (default : [0, 0]).
        U : number
            Convection velocity
        rv : number
            Vortex radius
        """
        self.x0 = x0
        self.y0 = y0
        self.U = U
        self.rv = rv

    def get_vector(self, x, y):
        """
        Return the velocity vector at the given point.
        """
        # compute r
        r = self._get_r(self.x0, x, self.y0, y)
        # compute theta
        theta = self._get_theta(self.x0, x, self.y0, y)
        # compute velocity in clyndrical referentiel
        Vr = 0
        Vphi = -3./4.*self.U*r**2*(1 - r**2/self.rv**2)*np.sin(theta)**2
        # get velocity in the cartesian refenrentiel
        Vx, Vy = self._cyl_to_cart(theta, Vr, Vphi)
        # returning
        return Vx, Vy


class LambOseenVortex(Vortex):
    """
    Representing a Lamb-Oseen Vortex, a solution to the laminar NS
    equation.
    """
    def __init__(self, x0=0, y0=0, ksi=1., t=1., viscosity=1e-6):
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
        self.x0 = x0
        self.y0 = y0
        self.ksi = ksi
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
                * (1 - np.exp(-r**2/(4.*self.viscosity*self.t)))
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
    def __init__(self, funct):
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

    def copy(self):
        """
        """
        return copy.deepcopy(self)

    def get_vector(self, x, y):
        """
        """
        print(self.funct(x, y))
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
                              unit_y=unit_y, unit_values=make_unit(''))
        return vf


class VortexSystem(object):
    """
    Representing a set of vortex.
    """

    def __init__(self):
        """
        """
        self.vortex = []
        self.nmb_vortex = 0

    def add(self, vortex):
        if not isinstance(vortex, (Vortex, CustomField)):
            raise TypeError()
        self.vortex.append(vortex.copy())
        self.nmb_vortex += 1

    def remove(self, ind):
        """
        """
        self.vortex = self.vortex[0:ind] + self.vortex[ind + 1::]

    def get_vector(self, x, y):
        """
        Return the velocity at the given point.
        """
        Vx = 0.
        Vy = 0.
        # add vortex participation
        if not len(self.vortex) == 0:
            Vx, Vy = self.vortex.get_vector(x, y)
            for vort in self.vortex[1::]:
                tmp_Vx, mp_Vy = vort.get_vector_field(x, y)
                Vx += Vx
                Vy += Vy
        # returning
        return Vx, Vy

    def get_vector_field(self, axe_x, axe_y, unit_x='', unit_y=''):
        # add vortex participation
        if not len(self.vortex) == 0:
            vf = self.vortex[0].get_vector_field(axe_x, axe_y,
                                                 unit_x=unit_x, unit_y=unit_y)
            for vort in self.vortex[1::]:
                vf += vort.get_vector_field(axe_x, axe_y,
                                            unit_x=unit_x, unit_y=unit_y)
        else:
            raise Exception('No field defined')
        # returning
        return vf
