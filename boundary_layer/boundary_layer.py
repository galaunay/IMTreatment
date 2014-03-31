# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:21:52 2014

@author: muahah
"""

import numpy as np
import matplotlib.pyplot as plt
from .. import Profile, make_unit, ScalarField
import pdb

NumberTypes = (int, float, long)
ArrayTypes = (list, np.ndarray)


class BlasiusBL:
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
        if not isinstance(Uinf, NumberTypes):
            raise TypeError("Uinf is not a number")
        if not isinstance(nu, NumberTypes):
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
        if not isinstance(x, (NumberTypes, ArrayTypes)):
            raise TypeError("x is not a number or a list")
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


class DefectLaw:
    """
    Class representing the log-defect law profile using Coles theory.
    By default, the used liquid is water.

    Parameters
    ----------
    h : number
        Water depth (m)
    tau : number
        The wall shear stress (Pa)
    Cc : number, optional
        The Coles parameters (n.u) (0.45 by default)
    visc_c : number, optional
        Kinematic viscosity (m²/s)
    rho : number, optional
        liquid density (kg/m^3)
    """

    def __init__(self, h, tau, Cc=0.45, visc_c=1e-6, rho=1000):
        """
        Class constructor.
        """
        if not isinstance(h, NumberTypes):
            raise TypeError("'h' has to be a number")
        if h <= 0:
            raise ValueError("'h' has to be positive")
        if not isinstance(tau, NumberTypes):
            raise TypeError("'tau' has to be a number")
        if tau < 0:
            raise ValueError("'tau' has to be a positive number")
        if not isinstance(Cc, NumberTypes):
            raise TypeError("'Cc' has to be a number")
        self.k = 0.4
        self.A = 5.5
        self.rho = rho
        self.tau = tau
        self.Utau = np.sqrt(self.tau/self.rho)
        self.Cc = Cc
        self.visc_c = visc_c
        self.visc_d = self.visc_c*self.rho
        self.h = h

    def display(self, dy, **plotArgs):
        """
        Display a velocity profile according to the log-defect law.

        Parameters
        ----------
        dy : number
            Resolution along the water depth (m).

        Returns
        -------
        fig : figure reference
            Reference to the displayed figure.
        """

        if not isinstance(dy, NumberTypes):
            raise TypeError("'dy' has to be a number")
        if dy < 0:
            raise ValueError("'dy' has to be a positive number")
        if dy > self.h:
            raise ValueError("'dy' has to be smaller than the water depth")
        y = np.arange(0, self.h, dy)
        prof = self.GetProfile(y)
        Umoy = prof.get_integral()
        fig = prof.display(reverse=True, label="tau={0:.4f} et Cc={2:.2f}: "
                           "Umoyen = {1:.4f}".format(self.tau, Umoy, self.Cc))
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

    def _defect_law(self, y):
        """
        Calculate the defect composante of the law.
        """
        return 2.*self.Cc/self.k*np.sin(np.pi*y/(2.*self.h))**2.

    def _log_law(self, y):
        """
        Calculate the log composante of the law.
        """
        return 1./self.k*np.log(y*self.Utau/self.visc_c)

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
        if not isinstance(y, ArrayTypes):
            raise TypeError("'y' has to be an array")
        if any(y < 0):
            raise ValueError("'y' has to be an array of positive number")
        if any(y > self.h):
            raise ValueError("'y' has to be smaller than the water depth")
        y = np.array(y)
        Ufin = []
        for y1 in y:
            y1plus = y1*self.Utau/self.visc_c
            ylimscv = 11.63
            ylimlog = 11.63
            if y1plus == 0:
                Utmp = 0
            elif y1plus <= ylimscv:
                Utmp = y1plus*self.Utau
            elif y1plus <= ylimlog:
                Utmp1 = y1plus*self.Utau
                ll = self._LogLaw(y1, self.Utau)
                dl = self._DefectLaw(y1, self.Utau, self.Cc)
                Utmp2 = self.Utau*(ll - dl + self.A)
                Utmp = ((abs(ylimlog - y1plus)*Utmp1
                         + abs(ylimscv - y1plus)*Utmp2)
                        / abs(ylimlog - ylimscv))
            else:
                ll = self._log_law(y1)
                dl = self._defect_law(y1)
                Utmp = self.Utau*(ll - dl + self.A)
            Ufin.append(Utmp)
        Ufin = Profile(y, Ufin, make_unit("m"), make_unit("m/s"))
        return Ufin

    def integral(self, x, y):
        return np.trapz(y, x)


def get_bl_thickness(obj, direction=1, perc=0.95):
    """
    Return a boundary layer thickness if 'obj' is a Profile.
    Return a profile of boundary layer thicknesses if 'obj' is a ScalarField.
    WARNING : the wall must be at x=0.

    Parameters
    ----------
    obj : Profile or ScalarField object
    direction : integer, optional
        If 'obj' is a ScalarField, determine the swept axis
        (1 for x and 2 for y).
    perc : float, optionnal
        Percentage used in the bl calculation (90% per default).

    Returns
    -------
    BLT : float or profile
        Boundary layer thickness, in axe x unit.
    """
    if isinstance(obj, Profile):
        maxi = obj.get_max()
        if maxi is None:
            return 0
        value = obj.get_interpolated_value(y=maxi*perc)
        return value[0]
    elif isinstance(obj, ScalarField):
        axe = obj.get_axes()[direction - 1]
        profiles = [obj.get_profile(direction, x) for x in axe]
        values = [get_bl_thickness(prof) for prof, _ in profiles]
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
        axe = obj.get_axes()[direction - 1]
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
        axe = obj.get_axes()[direction - 1]
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
    shape_factor = get_displ_thickness(obj, direction)\
        / get_momentum_thickness(obj, direction)
    return shape_factor
