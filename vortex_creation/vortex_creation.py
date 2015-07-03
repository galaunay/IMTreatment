# -*- coding: utf-8 -*-

import numpy as np
from ..core import VectorField, make_unit, NUMBERTYPES, ARRAYTYPES,\
    TemporalVectorFields
import pdb
import copy
import matplotlib.pyplot as plt


class Vortex(object):
    """
    """

    def __init__(self, x0, y0):
        """

        """
        self.__x0 = x0
        self.__y0 = y0
        self.rot_dir = 1

    @property
    def x0(self):
        return self.__x0

    @x0.setter
    def x0(self, new_x0):
        self.__x0 = new_x0

    @property
    def y0(self):
        return self.__y0

    @y0.setter
    def y0(self, new_y0):
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
        return np.sqrt((x - x0)**2 + (y - y0)**2)

    @staticmethod
    def _cyl_to_cart(theta, comp_r, comp_phi):
        comp_x = comp_r*np.cos(theta) - comp_phi*np.sin(theta)
        comp_y = comp_r*np.sin(theta) + comp_phi*np.cos(theta)
        return comp_x, comp_y

    def copy(self):
        return copy.deepcopy(self)


class SolidVortex(Vortex):
    """
    Representing a solid rotation.
    """
    def __init__(self, x0=0., y0=0., omega=1.):
        """
        Parameters
        ----------
        x0, y0 : numbers, optional
            Position of the vortex center (default : [0, 0]).
        omega : number, optional
            rotation velocity (rad/s)
        """
        super(SolidVortex, self).__init__(x0, y0)
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
    def __init__(self, x0=0., y0=0., gamma=1.):
        """
        Parameters
        ----------
        x0, y0 : numbers, optional
            Position of the vortex center (default : [0, 0]).
        gamma : number, optional
            Cirdculation of the free-vortex (m^2/s).
        """
        super(FreeVortex, self).__init__(x0, y0)
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
        super(BurgerVortex, self).__init__(x0, y0)
        if ksi < 0:
            self.rot_dir = -1
        else:
            self.rot_dir = 1
        self.ksi = np.abs(ksi)
        self.alpha = alpha
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
                * (1 - np.exp(-self.alpha*r**2/(4.*self.viscosity)))*self.rot_dir
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
    def __init__(self, x0=0, y0=0, U=1., rv=1., rot_dir=1, unit_values=''):
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
        super(HillVortex, self).__init__(x0, y0)
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
        Vr = -3./4.*self.U*r*(1 - r**2/self.rv**2)*2*np.sin(theta)*np.cos(theta)
        Vphi = 3./2.*self.U*np.sin(theta)**2*r*(1 - 2*r**2/self.rv**2)*self.rot_dir
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
        super(LambOseenVortex, self).__init__(x0, y0)
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
    def __init__(self, x0=0., y0=0., circ=1., rv=1.):
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
        super(RankineVortex, self).__init__(x0, y0)
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
            Vphi = self.circ*r/(2*np.pi*self.rv**2)
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
    def __init__(self, x0=0, y0=0, U=1., rv=1., Bessel_root_nmb=1):
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
        super(LambChaplyginVortex, self).__init__(x0, y0)
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

        Return
        ------
        y : number
            Bessel function value at 'x'
        """
        from scipy.special import jn
        return jn(order, x)


class CustomField(object):
    """
    Representing a custom field.

    Parameters
    ----------
    funct : function
        Representing the field.
        Has to take (x, y) as input and return (Vx, Vy).
    """
    def __init__(self, funct, unit_values=''):
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
        if not isinstance(vortex, (Vortex, CustomField)):
            raise TypeError()
        self.vortex.append(vortex.copy())
        self.nmb_vortex += 1
        ### TODO : pas optimal
        self.refresh_imaginary_vortex()

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
        ### TODO : pas optimal
        self.refresh_imaginary_vortex()

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
        self.custfields.append(custfield)

    def remove_vortex(self, ind):
        """
        Remove a vortex or a custom field from the set.

        Parameters
        ----------
        ind : integer
            Vortex indice to remove.
        """
        self.vortex = self.vortex[0:ind] + self.vortex[ind + 1::]

    def display(self):
        """
        Display a representation of the vortex system
        """
        for vort in self.vortex:
            plt.plot(vort.x0, vort.y0, marker='o', mec='k', mfc='w')
        for wall in self.walls:
            if wall.direction == 'x':
                plt.axvline(wall.position, color='k')
            else:
                plt.axhline(wall.position, color='k')
        for ivort in self.im_vortex:
            plt.plot(ivort.x0, ivort.y0, marker='o', mec='w', mfc='k')

    def get_vector(self, x, y):
        """
        Return the resulting velocity vector, at the given point.

        Parameters
        ----------
        x, y : numbers
            Position of the wanted vector.

        Returns
        -------
        Vx, Vy : numbers
            Velocity components.
        """
        Vx = 0.
        Vy = 0.
        # add vortex participation
        for vort in self.vortex:
            tmp_Vx, tmp_Vy = vort.get_vector(x, y)
            Vx += tmp_Vx
            Vy += tmp_Vy
        # add imaginary vortex participation
        for vort in self.im_vortex:
            tmp_Vx, tmp_Vy = vort.get_vector(x, y)
            Vx += tmp_Vx
            Vy += tmp_Vy
        # add custom fields
        for cst_field in self.custfields:
            tmp_Vx, tmp_Vy = cst_field.get_vector(x, y)
            Vx += tmp_Vx
            Vy += tmp_Vy
        # returning
        return Vx, Vy

    def refresh_imaginary_vortex(self):
        """
        """
        self.im_vortex = []
        for wall in self.walls:
            for vort in self.vortex:
                im_vort = vort.copy()
                im_vort.x0, im_vort.y0 = wall.get_symmetry((vort.x0, vort.y0))
                im_vort.rot_dir *= -1
                self.im_vortex.append(im_vort)


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

