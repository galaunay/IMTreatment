# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:21:52 2014

@author: muahah
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from .. import Profile, make_unit, ScalarField
import pdb

NUMBERTYPES = (int, float, long)
ARRAYTYPES = (list, np.ndarray)


class BlasiusBL(object):
    """
    Class representing a Blasius-like boundary layer.

    Constructor parameters
    ----------------------
    Uinf : number
        Flown velocity away from the wall (m/s).
    nu : number
        Kinematic viscosity (m²/s).
    """

    def __init__(self, Uinf, nu):
        """
        Class constructor.
        """
        if not isinstance(Uinf, NUMBERTYPES):
            raise TypeError("Uinf is not a number")
        if not isinstance(nu, NUMBERTYPES):
            raise TypeError("nu is not a number")
        self.Uinf = Uinf
        self.nu = nu

    def display(self, intervx, allturbulent=False, **plotargs):
        """
        Display the Blasius boundray layer.

        Parameters
        ----------
        intervx : array of numbers
            x values where we want the BL.
        allturbulent : boolean, optional
            if True, the all boundary layer is considered turbulent.
        """
        x = np.linspace(intervx[0], intervx[1], 1000)
        delta, _, _ = self.get_thickness(x, allturbulent)
        if not "label" in plotargs:
            plotargs["label"] = "Blasius theorical BL"
        fig = delta.display(**plotargs)
        return fig

    def get_thickness(self, x, allTurbulent=False):
        """
        Return the boundary layer thickness and the friction coefficient
        according to blasius theory.

        Fonction
        --------
        delta, Cf = BlasiusBL(allTurbulent=False)

        Parameters
        ---------
        x : number or array of number
            Position where the boundary layer thickness is computed (m)
        (can be a list).
        allTurbulent : bool, optional
            if True, the all boundary layer is considered turbulent.

        Returns
        -------
        delta : Profile object
            Boundary layer thickness profile (m)
        Cf : Profile object
            Friction coefficient profile (s.u)
        Rex : Profile object
            Reynolds number on the distance from the border.
        """
        if not isinstance(x, (NUMBERTYPES, ARRAYTYPES)):
            raise TypeError("x is not a number or a list")
        if isinstance(x, NUMBERTYPES):
            x = np.array([x])
        if not isinstance(allTurbulent, bool):
            raise TypeError("'allTurbulent' has to be a boolean")
        delta = []
        Cf = []
        Rex = []
        for xpos in x:
            if xpos == 0:
                delta.append(0)
                Cf.append(0)
                Rex.append(0)
            else:
                Rex.append(self.Uinf*xpos/self.nu)
                if Rex[-1] < 5e5 and not allTurbulent:
                    delta.append(xpos*4.92/np.power(Rex[-1], 0.5))
                    Cf.append(0.664/np.power(Rex[-1], 0.5))
                else:
                    delta.append(xpos*0.3806/np.power(Rex[-1], 0.2))
                    Cf.append(0.0592/np.power(Rex[-1], 0.2))
        delta = Profile(x, delta, unit_x=make_unit('m'),
                        unit_y=make_unit('m'))
        Cf = Profile(x, Cf, unit_x=make_unit('m'),
                     unit_y=make_unit(''))
        Rex = Profile(x, Rex, unit_x=make_unit('m'),
                      unit_y=make_unit(''))
        return delta, Cf, Rex

    def get_profile(self, x, turbulent=False):
        """
        Return a Blasius-like (laminar) profile at the given position.

        Parameters
        ----------
        x : number
            Position of the profile along x axis
        turbulent : bool, optional
            if True, the boundary layer is considered turbulent.

        Returns
        -------
        prof : Profile Object
            Wanted Blasius-like profile.
        """
        # derivate function
        if not turbulent:
            def f_deriv(F, theta):
                """
                y' = dy/dx
                y'' = dy'/dx
                dy''/dx = -1/2*y*dy'/dx
                """
                return [F[1], F[2], -1./2.*F[0]*F[2]]
            # profile initial values
            f0 = [0, 0, 0.332]
            # x values
            theta = np.linspace(0, 10, 1000)
            # solving with scipy ode solver
            sol = odeint(f_deriv, f0, theta)
            #getting adimensionnale velocity
            u_over_U = sol[:, 1]
            # getting dimensionnal values
            u = u_over_U*self.Uinf
            y = theta*np.sqrt(x)*np.sqrt(self.nu/self.Uinf)
        else:
            delta, _, _ = self.get_thickness(x, allTurbulent=True)
            theta = np.linspace(0, 2, 200)
            u_over_U = np.power(theta, 1./7.)
            u_over_U[theta > 1] = 1.
            y = theta*delta.y[0]
            u = u_over_U*self.Uinf
        return Profile(y, u, unit_x=make_unit('m'), unit_y=make_unit('m/s'))


class WallLaw(object):
    """
    Class representing a law of the wall profile.
    By default, the used liquid is water.

    Parameters
    ----------
    h : number
        Water depth (m)
    tau : number
        The wall shear stress (Pa)
    visc_c : number, optional
        Kinematic viscosity (m²/s)
    rho : number, optional
        liquid density (kg/m^3)
    """

    def __init__(self, h, tau, delta, visc_c=1e-6, rho=1000):
        """
        Class constructor.
        """
        if not isinstance(h, NUMBERTYPES):
            raise TypeError("'h' has to be a number")
        if h <= 0:
            raise ValueError("'h' has to be positive")
        if not isinstance(tau, NUMBERTYPES):
            raise TypeError("'tau' has to be a number")
        if not isinstance(delta, NUMBERTYPES):
            raise TypeError("'delta' has to be a number")
        if tau < 0:
            raise ValueError("'tau' has to be a positive number")
        self.k = 0.4
        self.A = 5.5
        self.rho = rho
        self.tau = tau
        self.delta = delta
        self.Utau = np.sqrt(self.tau/self.rho)
        self.visc_c = visc_c
        self.visc_d = self.visc_c*self.rho
        self.h = h

    def display(self, dy, **plotArgs):
        """
        Display the velocity profile.

        Parameters
        ----------
        dy : number
            Resolution along the water depth (m).

        Returns
        -------
        fig : figure reference
            Reference to the displayed figure.
        """

        if not isinstance(dy, NUMBERTYPES):
            raise TypeError("'dy' has to be a number")
        if dy < 0:
            raise ValueError("'dy' has to be a positive number")
        if dy > self.h:
            raise ValueError("'dy' has to be smaller than the water depth")
        y = np.arange(0, self.h, dy)
        prof = self.get_profile(y)
        Umoy, _ = prof.get_integral()
        fig = prof.display(reverse=True, label=("tau={0:.4f} : "
                           "Umoyen = {1:.4f}").format(self.tau, Umoy))
        y5 = 5.*self.visc_c*np.sqrt(self.rho/self.tau)
        y30 = 30.*self.visc_c*np.sqrt(self.rho/self.tau)
        mini = prof.get_min()
        maxi = prof.get_max()
        plt.plot([mini, maxi], [y5, y5], 'r--')
        plt.plot([mini, maxi], [y30, y30], 'r--')
        plt.xlabel("Velocity (m/s)")
        plt.ylabel("y (m)")
        plt.title("velocity profile according to the log-defect law")
        return fig

    def get_profile(self, y):
        """
        Return a log-defect profile, according to the given parameters.

        Parameters
        ----------
        y : array
            Value of y in which calculate the profile (m).

        Returns
        -------
        prof : Profile object
            the profile for values of 'y'
        """
        if not isinstance(y, ARRAYTYPES):
            raise TypeError("'y' has to be an array")
        if any(y < 0):
            raise ValueError("'y' has to be an array of positive number")
        if any(y > self.h):
            raise ValueError("'y' has to be smaller than the water depth")
        y = np.array(y)
        yplus = y*self.Utau/self.visc_c
        ylimscv = 11.63
        ylimlog = 450.
        Ufin = []
        for i, yp in enumerate(yplus):
            if yp < 0:
                Utmp = 0
            elif yp <= ylimscv:
                Utmp = self._scv(yp)*self.Utau
            elif yp <= ylimlog:
                Utmp = self._inner_turb(yp)*self.Utau
            elif y[i] <= self.delta:
                Utmp = self._outer_turb(yp)*self.Utau
            else:
                Utmp = self._undisturbed(yp)*self.Utau
            Ufin.append(Utmp)
        Ufin = Profile(y, Ufin, make_unit("m"), make_unit("m/s"))
        return Ufin

    def integral(self, x, y):
        return np.trapz(y, x)

    def _scv(self, yp):
        """
        Calculate u/Utau in the scv.
        """
        return yp

    def _inner_turb(self, yp):
        """
        Calculate u/Utau in the inner part of the bl.
        """
        return 1./self.k*np.log(yp) + 5.1

    def _outer_turb(self, yp):
        """
        Calculate u/Utau in the outer part of the bl.
        """
        return self._inner_turb(yp)

    def _undisturbed(self, yp):
        """
        Calculate u/Utau outside of the bl
        """
        return self._outer_turb(self.delta*self.Utau/self.visc_c)


class WakeLaw(WallLaw):
    """
    Class representing a law of the wake profile using Coles theory.
    By default, the used liquid is water.

    Parameters
    ----------
    h : number
        Water depth (m)
    tau : number
        The wall shear stress (Pa)
    delta : number
        The boundary layer thickness (m)
    Cc : number, optional
        The Coles parameters (n.u) (0.45 by default)
    visc_c : number, optional
        Kinematic viscosity (m²/s)
    rho : number, optional
        liquid density (kg/m^3)
    """

    def _inner_turb(self, yp):
        """
        Calculate u/Utau in the turbulent part of the bl.
        """
        return self._wake_law(yp)

    def _outer_turb(self, yp):
        """
        Calculate u/Utau outside of the bl.
        """
        return self._wake_law(yp)

    def _wake_law(self, yp):
        """
        Calculate the defect composante of the law.
        Take yp.
        Return u/Utau.
        """
        y = yp/self.Utau*self.visc_c
        Cc = 0.55
        return (1./self.k*np.log(self.Utau*y/self.visc_c) + 5.1
                + 2.*Cc/self.k*np.sin(np.pi*y/(2.*self.delta))**2)


def get_bl_thickness(obj, direction=1,  perc=0.95):
    """
    Return a boundary layer thickness if 'obj' is a Profile.
    Return a profile of boundary layer thicknesses if 'obj' is a ScalarField.
    WARNING : the wall must be at x=0.

    Parameters
    ----------
    obj : Profile or ScalarField object
        Vx field.
    direction : integer, optional
        If 'obj' is a ScalarField, determine the swept axis
        (1 for x and 2 for y).
    perc : float, optionnal
        Percentage used in the bl calculation (95% per default).

    Returns
    -------
    BLT : float or profile
        Boundary layer thickness, in axe x unit.
    """
    if isinstance(obj, Profile):
        maxi = obj.max
        if maxi is None:
            return 0
        value = obj.get_interpolated_value(y=maxi*perc)
        return value[0]
    elif isinstance(obj, ScalarField):
        if direction == 1:
            axe = obj.axe_x
        else:
            axe = obj.axe_y
        profiles = [obj.get_profile(direction, x) for x in axe]
        values = [get_bl_thickness(prof, perc=perc) for prof, _ in profiles]
        return Profile(axe, values, unit_x=obj.unit_x, unit_y=obj.unit_y)
    else:
        raise TypeError("Can't compute (yet ?) BL thickness on this kind of"
                        " data : {}".format(type(obj)))


def get_displ_thickness(obj, direction=1):
    """
    Return a displacement thickness if 'obj' is a Profile.
    Return a profile of displacement thicknesses if 'obj' is a Scalarfield.
    WARNING : the wall must be at x=0.

    Parameters
    ----------
    obj : Profile or ScalarField object
    direction : integer, optional
        If 'obj' is a ScalarField, determine the swept axis
        (1 for x and 2 for y).

    Returns
    -------
    delta : float or Profile
        Boundary layer displacement thickness, in axe x unit.
    """
    if isinstance(obj, Profile):
        bl_perc = 0.95
        # cut the profile in order to only keep the BL
        bl_thick = get_bl_thickness(obj, perc=bl_perc)
        if bl_thick == 0:
            return 0
        obj = obj.trim([0, bl_thick])
        # removing negative and masked points
        if isinstance(obj.y, np.ma.MaskedArray):
            mask = np.logical_and(obj.y.mask, obj.x < 0)
            obj.x = obj.x[~mask]
            obj.y = obj.y._data[~mask]
        # if there is no more value in the profile (all masked)
        if len(obj.x) == 0:
            return 0
        # adding a x(0) value if necessary
        if obj.x[0] != 0:
            pos_x = np.append([0], obj.x)
            pos_y = np.append([0], obj.y)
        else:
            pos_x = obj.x
            pos_y = obj.y
        # computing bl displacement thickness
        fonct = 1 - pos_y/np.max(pos_y)
        delta = np.trapz(fonct, pos_x)
        return delta
    elif isinstance(obj, ScalarField):
        if direction == 1:
            axe = obj.axe_x
        else:
            axe = obj.axe_y
        profiles = [obj.get_profile(direction, x) for x in axe]
        values = [get_displ_thickness(prof) for prof, _ in profiles]
        return Profile(axe, values, unit_x=obj.unit_x, unit_y=obj.unit_y)
    else:
        raise TypeError("Can't compute (yet ?) BL displacement thickness on"
                        "this kind of data : {}".format(type(obj)))


def get_momentum_thickness(obj, direction=1):
    """
    Return a momentum thickness if 'obj' is a Profile.
    Return a profile of momentum thicknesses if 'obj' is a Scalarfield.
    WARNING : the wall must be at x=0.

    Parameters
    ----------
    obj : Profile or ScalarField object
    direction : integer, optional
        If 'obj' is a ScalarField, determine the swept axis
        (1 for x and 2 for y).

    Returns
    -------
    delta : float or Profile
        Boundary layer momentum thickness, in axe x unit.
    """
    if isinstance(obj, Profile):
        bl_perc = 0.95
        # cut the profile in order to only keep the BL
        bl_thick = get_bl_thickness(obj, perc=bl_perc)
        if bl_thick == 0:
            return 0
        obj = obj.trim([0, bl_thick])
        # removing negative and masked points
        if isinstance(obj.y, np.ma.MaskedArray):
            mask = np.logical_and(obj.y.mask, obj.x < 0)
            obj.x = obj.x[~mask]
            obj.y = obj.y._data[~mask]
        # if there is no more profile (all masked)
        if len(obj.x) == 0:
            return 0
        # adding a x(0) value
        if obj.x[0] != 0:
            pos_x = np.append([0], obj.x)
            pos_y = np.append([0], obj.y)
        else:
            pos_x = obj.x
            pos_y = obj.y
        # computing bl momentum thickness
        fonct = pos_y/np.max(pos_y)*(1 - pos_y/np.max(pos_y))
        delta = np.trapz(fonct, pos_x)
        return delta
    elif isinstance(obj, ScalarField):
        if direction == 1:
            axe = obj.axe_x
        else:
            axe = obj.axe_y
        profiles = [obj.get_profile(direction, x) for x in axe]
        values = [get_momentum_thickness(prof) for prof, _ in profiles]
        return Profile(axe, values, unit_x=obj.unit_x, unit_y=obj.unit_y)
    else:
        raise TypeError("Can't compute (yet ?) BL momentum thickness on"
                        "this kind of data")


def get_shape_factor(obj, direction=1):
    """
    Return a shape factor if 'obj' is a Profile.
    Return a profile of shape factors if 'obj' is a Scalarfield.
    WARNING : the wall must be at x=0.

    Parameters
    ----------
    obj : Profile or ScalarField object
    direction : integer, optional
        If 'obj' is a ScalarField, determine the swept axis
        (1 for x and 2 for y).

    Returns
    -------
    delta : float or Profile
        Boundary layer shape factor, in axe x unit.
    """
    displ = get_displ_thickness(obj, direction)
    mom = get_momentum_thickness(obj, direction)
    shape_factor = displ/mom
    if isinstance(shape_factor, Profile):
        shape_factor.mask = np.logical_or(shape_factor.mask, mom.y <= 0)
    return shape_factor


def get_shear_stress(obj, viscosity=1e-3, direction=1, method='simple',
                     respace=False):
    """
    Return the wall shear stress.
    If velocities values are missing near the wall, an extrapolation
    (bad accuracy) is used.
    Warning : the wall must be at x=0

    Parameters
    ----------
    obj : Profile or ScalarField object
        .
    viscosity : number, optional
        Dynamic viscosity (default to water : 1e-3)
    direction : integer, optional
        If 'obj' is a ScalarField, determine the swept axis
        (1 for x and 2 for y).
    method : string, optional
        'simple' (default) : use simple gradient computation
        'wall_law' : use the linear part of the 'law of the wall' model
        (more accurate but need some points in the viscous sublayer)
    respace : bool, optional
        Use linear interpolation to create an evenly spaced profile.
    """
    unit_visc = make_unit('kg/(m*s)')
    # check parameters
    if not isinstance(viscosity, NUMBERTYPES):
        raise TypeError()
    if viscosity <= 0:
        raise ValueError()
    if not direction in [1, 2]:
        raise ValueError()
    # if obj is a profile
    if isinstance(obj, Profile):
        if method == 'simple':
            # add zero if necessary
            if obj.x[0] > 0:
                obj.x = np.concatenate(([0], obj.x))
                obj.y = np.concatenate(([0], obj.y))
            # respace if asked
            if respace:
                obj = obj.evenly_space('linear')
            # compute gradients and return shear stress
            tmp_prof = obj.get_gradient()*viscosity*unit_visc
            return tmp_prof
        elif method == 'wall_law':
            new_x = obj.x[obj.x > 0]
            new_y = obj.y[obj.x > 0]/new_x*viscosity
            new_unit_y = obj.unit_y/obj.unit_x*unit_visc
            mask = obj.mask[obj.x > 0]
            return Profile(x=new_x, y=new_y, mask=mask, unit_x=obj.unit_x,
                           unit_y=new_unit_y, name=obj.name)
        else:
            raise ValueError()
    elif isinstance(obj, ScalarField):
        raise Exception("not implemented yet")
    else:
        raise TypeError()
