# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:21:52 2014

@author: muahah
"""

import numpy as np
import matplotlib.pyplot as plt
from .. import Profile, make_unit

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
        """
        if not isinstance(x, (NumberTypes, ArrayTypes)):
            raise TypeError("x is not a number or a list")
        if not isinstance(allTurbulent, bool):
            raise TypeError("'allTurbulent' has to be a boolean")
        delta = []
        Cf = []
        for xpos in x:
            if xpos == 0:
                delta.append(0)
                Cf.append(0)
            else:
                Rex = self.Uinf*xpos/self.nu
                if Rex < 5e5 and not allTurbulent:
                    delta.append(xpos*4.92/np.power(Rex, 0.5))
                    Cf.append(0.664/np.power(Rex, 0.5))
                else:
                    delta.append(xpos*0.3806/np.power(Rex, 0.2))
                    Cf.append(0.0592/np.power(Rex, 0.2))
        delta = Profile(x, delta, unit_x=make_unit('m'),
                        unit_y=make_unit('m'))
        Cf = Profile(x, Cf, unit_x=make_unit('m'),
                     unit_y=make_unit(''))
        return delta, Cf


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


def get_bl_thickness(profile, perc=0.95):
    """
    Return the profile boundary layer thickness.
    WARNING : the wall must be at x=0.

    Parameters
    ----------
    profile : Profile object
    perc : float, optionnal
        Percentage used in the bl calculation (90% per default).

    Returns
    -------
    BLT : float
        Boundary layer thickness, in axe x unit.
    """
    value = profile.get_interpolated_value(y=profile.get_max()*perc)
    return value[0]


def get_displ_thickness(profile):
    """
    Return the profile displacement thickness.
    WARNING : the wall must be at x=0.

    Parameters
    ----------
    profile : Profile object

    Returns
    -------
    delta : float
        Boundary layer displacement thickness, in axe x unit.
    """
    bl_perc = 0.95
    # cut the profile in order to only keep the BL
    bl_thick = get_bl_thickness(profile, perc=bl_perc)
    profile = profile.trim([0, bl_thick])
    # adding a x(0) value
    if profile.x[0] != 0:
        pos_x = np.append([0], profile.x)
        pos_y = np.append([0], profile.y)
    else:
        pos_x = profile.x
        pos_y = profile.y
    fonct = 1 - np.abs(pos_y)/(np.max(np.abs(pos_y)))
    delta = np.trapz(fonct, pos_x)
    return delta


def get_momentum_thickness(profile):
    """
    Return the profile momentum thickness.
    WARNING : the wall must be at x=0.

    Parameters
    ----------
    profile : Profile object

    Returns
    -------
   delta : float
        Boundary layer momentum thickness, in axe x unit.
    """
    bl_perc = 0.95
    # cut the profile in order to only keep the BL
    bl_thick = get_bl_thickness(profile, perc=bl_perc)
    profile = profile.trim([0, bl_thick])
    # adding a x(0) value
    if profile.x[0] != 0:
        pos_x = np.append([0], profile.x)
        pos_y = np.append([0], profile.y)
    else:
        pos_x = profile.x
        pos_y = profile.y
    fonct = np.abs(pos_y)/np.max(np.abs(pos_y))*(1 - np.abs(pos_y)
                                                 / np.max(np.abs(pos_y)))
    delta = np.trapz(fonct, pos_x)
    return delta


def get_shape_factor(profile):
    """
    Return the profile shape factor.
    WARNING : the wall must be at x=0.

    Parameters
    ----------
    profile : Profile object

    Returns
    -------
    shape_factor : float
        Boundary layer shape factor, in axe x unit.
    """
    shape_factor = get_displ_thickness(profile)\
        / get_momentum_thickness(profile)
    profile.display(reverse=True)
    return shape_factor
