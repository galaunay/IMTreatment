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
import matplotlib.pyplot as plt
from scipy.integrate import odeint, ode
from scipy.interpolate import UnivariateSpline
import warnings
from .. import Profile, make_unit, ScalarField, VectorField, \
    TemporalVectorFields
import scipy.integrate as spinteg

NUMBERTYPES = (int, float, int, np.float, np.float16, np.float32)
ARRAYTYPES = (list, np.ndarray)


class BlasiusBL(object):
    """
    Class representing a Blasius-like boundary layer.

    Parameters
    ----------
    Uinf : number
        Flown velocity away from the wall (m/s).
    nu : number
        Kinematic viscosity (m²/s).
    rho : number
        Density (kg/m^3)
    """

    def __init__(self, Uinf, nu, rho):
        """
        Class constructor.
        """
        if not isinstance(Uinf, NUMBERTYPES):
            raise TypeError("Uinf is not a number")
        if not isinstance(nu, NUMBERTYPES):
            raise TypeError("nu is not a number")
        if not isinstance(rho, NUMBERTYPES):
            raise TypeError("rho is not a number")
        self.Uinf = Uinf
        self.nu = nu
        self.rho = rho

    def get_Rex(self, x):
        """
        Return the Reynolds number based on the distance from the beginning of
        the plate.
        """
        Re_x = self.Uinf*x/self.nu
        if Re_x > 2e5:
            print('Warning : Turbulent BL')
        return Re_x

    def get_BL_properties(self, x, allTurbulent=False, bl_perc="99"):
        """
        Return the boundary layer properties according to blasius theory.

        Parameters
        ----------
        x : number or array of number
            Position where the boundary layer thickness is computed (m)
        (can be a list).
        allTurbulent : bool, optional
            if True, the all boundary layer is considered turbulent.
        bl_perc : string in ['95', '99']
            Percentage of maximum velocity defining the BL thickness

        Returns
        -------
        delta : array of numbers
            Boundary layer thickness profile (m)
        delta2 : array of numbers
            Boundary layer momentum thickness (m)
        delta_star : array of numbers
            Boundary layer displacement thickness (m)
        H : array of numbers
            Shape factor
        Cf : array of numbers
            Friction coefficient profile
        Rex : array of numbers
            Reynolds number on the distance from the border.
        tau_w : array of numbers
            Wall shear stress (Pa)
        """
        # check
        if not isinstance(x, (NUMBERTYPES, ARRAYTYPES)):
            raise TypeError("x is not a number or a list")
        if isinstance(x, NUMBERTYPES):
            x = np.array([x])
        if not isinstance(allTurbulent, bool):
            raise TypeError("'allTurbulent' has to be a boolean")
        if bl_perc == "95":
            perc_coef_lam = 3.92
        elif bl_perc == "99":
            perc_coef_lam = 4.91
        else:
            raise ValueError()
        # Loop on c position
        delta = []
        delta2 = []
        delta_star = []
        H = np.ones(len(x))*2.59
        Cf = []
        Rex = []
        tau_w = []
        for xpos in x:
            if xpos == 0:
                delta.append(0)
                delta2.append(0)
                delta_star.append(0)
                Cf.append(0)
                Rex.append(0)
                tau_w.append(0)
            else:
                Rex.append(self.Uinf*xpos/self.nu)
                if Rex[-1] < 5e5 and not allTurbulent:
                    delta.append(xpos*perc_coef_lam/np.power(Rex[-1], 0.5))
                    delta2.append(0.664*(self.nu*xpos/self.Uinf)**.5)
                    delta_star.append(delta2[-1]*2.59)
                    Cf.append(0.664/np.power(Rex[-1], 0.5))
                    tau_w.append(Cf[-1]*1./2.*self.rho*self.Uinf**2)
                else:
                    if bl_perc == '95':
                        raise NotImplementedError()
                    delta.append(xpos*0.3806/np.power(Rex[-1], 0.2))
                    delta2.append(0)
                    delta_star.append(delta2[-1]*2.59)
                    Cf.append(0.0592/np.power(Rex[-1], 0.2))
                    tau_w.append(Cf[-1]*1./2.*self.rho*self.Uinf**2)
        delta = np.array(delta)
        delta2 = np.array(delta2)
        delta_star = np.array(delta_star)
        H = np.array(H)
        Cf = np.array(Cf)
        Rex = np.array(Rex)
        tau_w = np.array(tau_w)
        return delta, delta2, delta_star, H, Cf, Rex, tau_w

    def get_thickness_with_confinement(self, x, h, allTurbulent=False):
        """
        Return the boundary layer thickness and the friction coefficient
        according to blasius theory and adapted for use with low water levels.
        (Only valid in laminar BL)
        Fonction
        --------
        delta, Cf = BlasiusBL(allTurbulent=False)

        Parameters
        ----------
        x : number or array of number
            Position where the boundary layer thickness is computed (m)
            (can be a list).
        h : number
            Water depth (m).
        allTurbulent : bool, optional
            if True, the all boundary layer is considered turbulent.

        Returns
        -------
        delta : Profile object
            Boundary layer thickness profile (m)
        """
        if not isinstance(x, (NUMBERTYPES, ARRAYTYPES)):
            raise TypeError("x is not a number or a list")
        if isinstance(x, NUMBERTYPES):
            x = np.array([x])
        if isinstance(h, NUMBERTYPES):
            h = np.array([h])
        delta_blas = self.get_BL_properties(x, allTurbulent=allTurbulent)[0]
        delta_perso = delta_blas*(1 - 0.26547*delta_blas/h)
        # returning
        delta = Profile(x, delta_perso, unit_x=make_unit('m'),
                        unit_y=make_unit('m'))
        return delta

    def get_profile(self, x, y=None, allTurbulent=False):
        """
        Return a Blasius-like (laminar) profile at the given position.

        Parameters
        ----------
        x : number
            Position of the profile along x axis
        y : array of numbers
            Point along y where to compute the profile (if not specified,
            200 homogeneously placed points are used)
        allTurbulent : bool, optional
            if True, the boundary layer is considered turbulent.

        Returns
        -------
        prof : Profile Object
            Wanted Blasius-like profile.
        """
        # check
        if not isinstance(x, NUMBERTYPES):
            raise TypeError()
        # Not turbulent case
        if not allTurbulent:
            def f_deriv(F, theta):
                """
                y' = dy/dx
                y'' = dy'/dx
                dy''/dx = -1/2*y*dy'/dx
                """
                return [F[1], F[2], -1./2.*F[0]*F[2]]
            # profile initial values
            f0 = [0, 0, 0.332]
            # y values
            if y is None:
                theta = np.linspace(0, 10, 200)
                y = theta*np.sqrt(x)*np.sqrt(self.nu/self.Uinf)
            else:
                theta = y/(np.sqrt(x)*np.sqrt(self.nu/self.Uinf))
            # solving with scipy ode solver
            sol = odeint(f_deriv, f0, theta)
            # getting adimensionnale velocity
            u_over_U = sol[:, 1]
            # getting dimensionnal values
            u = u_over_U*self.Uinf
        # Turbulent case
        else:
            delta, _, _ = self.get_thickness(x, allTurbulent=True)
            if y is None:
                theta = np.linspace(0, 10, 200)
                y = theta*delta.y[0]
            else:
                theta = y/delta.y[0]
            u_over_U = np.power(theta, 1./7.)
            u_over_U[theta > 1] = 1.
            u = u_over_U*self.Uinf
        return Profile(y, u, unit_x=make_unit('m'), unit_y=make_unit('m/s'))

    def get_profile_with_confinement(self, x, h, y=None, allTurbulent=False):
        """
        Return a Blasius-like (laminar) profile at the given position, ajusted
        for confined BL.

        Parameters
        ----------
        x : number
            Position of the profile along x axis
        h : number
            Pater level.
        y : array of numbers
            Point along y where to compute the profile (if not specified,
            200 homogeneously placed points are used)
        allTurbulent : bool, optional
            if True, the boundary layer is considered turbulent.

        Returns
        -------
        prof : Profile Object
            Wanted Blasius-like profile.
        """
        # check
        if not isinstance(x, NUMBERTYPES):
            raise TypeError()
        if not isinstance(h, NUMBERTYPES):
            raise TypeError()
        # get
        delta = self.get_thickness_with_confinement(x, h,
                                                    allTurbulent=allTurbulent)
        eq_x = self.get_x_from_delta(delta.y[0])
        # return
        return self.get_profile(eq_x, y=y, allTurbulent=allTurbulent)

    def get_x_from_delta(self, delta, allTurbulent=False):
        """
        Return a the x value that give the wanted delta.
        """
        # getting the laminar value of x
        xpos = (delta**2*self.Uinf)/(4.92**2*self.nu)
        # checking if turbulent
        Re_x = self.get_Rex(xpos)
        if Re_x > 5e5 or allTurbulent:
            xpos = ((delta*self.Uinf**(.2))/(0.3806*self.nu**(.2)))**(1./.8)
        return xpos


class FalknerSkanBL(object):

    def __init__(self, nu, m, c0, L):
        """
        Represent a Boundary layer with a pressure gradient, according to
        Falkner-Skan.

        Parameters
        ----------
        nu : number
            Viscosity
        m, c0 , L: integer and numbers
            Pressure gradient parameters. Such as U_e = c0*(x/L)^m.
        """
        self.nu = nu
        self.m = m
        self.c0 = c0
        self.L = L
        self.beta = (2*self.m)/(self.m + 1)

    def get_u_e(self, x):
        """
        Return the external velocity at position 'x'.
        """
        return self.c0*(x/self.L)**self.m

    def get_mu(self, x, y):
        """
        return the \mu parameter at the position '(x, y)'.
        """
        return y*(self.c0*(self.m + 1)/(2*self.nu*self.L))**.5 * \
            (x/self.L)**((self.m - 1)/2.)

    def get_y(self, x, mu):
        """
        Return the 'y' vaue associated with \mu parameter.
        """
        return mu*(self.nu*x/self.get_u_e(x))**.5
        return mu/((self.c0*(self.m + 1)/(2*self.nu*self.L))**.5 *
                   (x/self.L)**((self.m - 1)/2.))

    def get_f_function(self, relerr=1e-5, max_it=1000, verbose=False):
        """
        Return the 'f' function appearing in the Falkner-skan equation, and its
        derivatives.

        Notes
        -----
        * Falkner-Skan equation :

        .. math::
            f''' + ff'' + \\beta \\left( 1 - f'^2 \\right) = 0

            f(0) = f'(0) = 0 \\: \\text{ and } \\: f'(\\infty) = 1

        * Associated constants :

        .. math::
            \\beta = \\frac{2m}{m + 1}

            -0.0905 \le m \le 2

        * Remark on numerical resolution :

            The Falkner-Skan equation is a ODE system with BVP
            (boundary value problem). Classical ODE algorithm such as
            Runge-Kutta or Vode cannot take care of the :math:`f'(\\infty)=0`.
            This ODE system is so solved with a shooting method.
        """
        import warnings
        warnings.filterwarnings('error')
        # Set mu values
        mu = np.linspace(0, 10, 100)

        # Implement Falkner-Skan equation
        def deriv(Y, t):
            Y_prime = [0, 0, 0]
            Y_prime[0] = Y[1]
            Y_prime[1] = Y[2]
            Y_prime[2] = - (self.m + 1)/2.*Y[0]*Y[2] - self.m*(1. - Y[1]**2)
            return Y_prime

        def deriv2(t, Y):
            return deriv(Y, t)

        # Create ODE system for resolution
        integrator = "lsoda"
        ode_maxsteps = 10000
        ODE = ode(deriv2)
        ODE.set_integrator(integrator, nsteps=ode_maxsteps)

        # Prepare shooting method to solve BVP (boundary value problem)
        aimed_Y1p = 1.
        first_Y3_guess = 0.33 + 0.77*np.log(2.207*(self.m + 0.45696))
        interval_Y3 = [first_Y3_guess - .33, first_Y3_guess + .33]
        # first guesses for f'''(0)
        tmp_Y3s = np.linspace(*interval_Y3, num=100)
        tmp_Y1ps = []
        tmp_Y2ps = []
        for tmpY3 in tmp_Y3s:
            ODE.set_initial_value([0, 0, tmpY3])
            try:
                prof = ODE.integrate(mu[-1])
            except RuntimeWarning:
                tmp_Y1ps.append(np.nan)
                tmp_Y2ps.append(np.nan)
            else:
                tmp_Y1ps.append(prof[1])
                tmp_Y2ps.append(prof[2])
        tmp_Y1ps = np.array(tmp_Y1ps)
        tmp_Y2ps = np.array(tmp_Y2ps)[~np.isnan(tmp_Y1ps)]
        tmp_Y3s = tmp_Y3s[~np.isnan(tmp_Y1ps)]
        tmp_Y1ps = tmp_Y1ps[~np.isnan(tmp_Y1ps)]
        # check if we get a cool interval with apropriate derivates
        resid = tmp_Y1ps - aimed_Y1p
        ind_min = np.argsort(np.abs(tmp_Y1ps - aimed_Y1p))
        if np.min(resid)*np.max(resid) < 0:
            tmp_neg = resid[:-1]*resid[1::]
            ind1 = np.where(tmp_neg < 0)[0]
            ind2 = np.where(tmp_Y2ps[:-1]*tmp_Y2ps[1:] < 0)[0]
            ind = [ind1[i]
                   for i in np.arange(len(ind1))
                   if ind1[i] == ind2[i]][0]
            interval_Y1p = [tmp_Y1ps[ind], tmp_Y1ps[ind + 1]]
            interval_Y3 = [tmp_Y3s[ind], tmp_Y3s[ind + 1]]
        # check if we get a lesser cool interval
        elif ind_min[0] in [0, len(tmp_Y1ps) - 1]:
            raise Exception("Guessing interval for f''(0) need to be enlarged"
                            "\n tmp_Y1ps={}".format(tmp_Y1ps))
        else:
            interval_Y1p = [tmp_Y1ps[ind_min[0]], tmp_Y1ps[ind_min[1]]]
            interval_Y3 = [tmp_Y3s[ind_min[0]], tmp_Y3s[ind_min[1]]]

        # Shooting method to solve iteratively boundary conditions
        nmb_it = 0
        if verbose:
            print("+++ Shooting method iteration +++")
        mid_Y3 = 1e30
        while True:
            new_mid_Y3 = ((interval_Y3[0]*np.abs(1 - interval_Y1p[1]) +
                           interval_Y3[1]*np.abs(1 - interval_Y1p[0])) /
                          (np.abs(1 - interval_Y1p[1]) +
                           np.abs(1 - interval_Y1p[0])))
            if np.abs(new_mid_Y3 - mid_Y3) < 1e-10:
                if verbose:
                    print("+++    Can't do better than that "
                          "(stuck in local minimum ?)")
                break
            mid_Y3 = new_mid_Y3
            if verbose:
                print(("+++    ({})     new tested Y3 = {:.4f}"
                       .format(nmb_it, mid_Y3)))
            # compute new Y1p at middle
            ODE.set_initial_value([0, 0, mid_Y3])
            prof = ODE.integrate(mu[-1])
            new_Y1p = prof[1]
            # replace interval with new values
            if (interval_Y1p[0] - aimed_Y1p)*(new_Y1p - aimed_Y1p) < 0:
                interval_Y1p[1] = new_Y1p
                interval_Y3[1] = mid_Y3
            else:
                interval_Y1p[0] = new_Y1p
                interval_Y3[0] = mid_Y3
            err = np.abs((new_Y1p - aimed_Y1p)/np.mean([new_Y1p, aimed_Y1p]))
            if verbose:
                print(("+++             err = {:.4f}".format(err)))
            # stopping test
            if nmb_it > max_it:
                if verbose:
                    print("+++    Maximum number of iteration reached")
                break
            if err < relerr:
                if verbose:
                    print(("+++    Converged in {} iterations !"
                           .format(nmb_it)))
                    break
            # incr
            nmb_it += 1
        Y3 = mid_Y3

        # Solve ODE with the new set of boundary conditions : f(0)=f'(0)= 0 and
        # f''(0) = Y3
        sol = odeint(deriv, y0=[0, 0, Y3], t=mu)

        # Return
        return mu, sol

    def get_profile(self, x, relerr=1e-5, max_it=1000, verbose=False):
        """
        return the velocity profile at 'x' position.
        """
        # solve ODE system using shooting method and Vode
        mu, sol = self.get_f_function(relerr=relerr, max_it=max_it,
                                      verbose=verbose)
        # get y values
        y = self.get_y(x, mu)

        # check solution
        dmu = mu[1] - mu[0]
        F0 = sol[:, 0]
        F1 = sol[:, 1]
        F2 = sol[:, 2]
        F3 = np.gradient(F2, dmu)
        res = np.mean(F3 + (self.m + 1)/2*F0*F2 + self.m*(1 - F1**2))
        print("+++ Equation +++")
        print(("+++     should be zero : {:.4f}".format(res)))
        print("+++ Boundary conditions +++")
        print(("+++     f(0)={:.3f} (should be 0)".format(sol[0, 0])))
        print(("+++     f'(0)={:.3f} (should be 0)".format(sol[0, 1])))
        print(("+++     f'(inf)={:.3f} (should be 1)".format(sol[-1, 1])))
        vx = sol[:, 1]*self.get_u_e(x)
        return y, vx


class ThwaitesBL(object):

    def __init__(self, u_e, nu=1.e-6):
        """
        Represent a The solution of the boundary layer momentum equation
        using Thwaites solution.

        Parameters
        ----------
        u_e : function
            External velocity, should take a distance in 'm' as argument
            and returning a velocity in 'm/s'.
        nu : number
            Viscosity (default to water, at 1e-6)
        """
        self.nu = nu
        self.u_e = u_e
        # classical empirical coefficients
        self.emp_coef_1 = 0.441
        self.emp_coef_2 = 5.68
#        # adapted empirical coefficients
#        self.emp_coef_1 = 0.19199
#        self.emp_coef_2 = 10.6612

    def u_e_pow(self, x):
        """
        Function u_e(x)**4.68
        (used int numerical resolution)
        """
        return self.u_e(x)**(self.emp_coef_2 - 1)

    def get_momentum_thikness(self, x):
        """
        Return the evolution of the boundary layer momentum thikness.

        Parameters
        ----------
        x : number or array of numbers
            Position along the flat plan until where we want to solve the
            momentum equation.
            (has no influence on the resoluion accuracy)

        Returns
        -------
        delta2 : position and value of momentum thicknesses
        """
        if isinstance(x, NUMBERTYPES):
            res, eps = spinteg.quad(func=self.u_e_pow, a=0, b=x)
            denom = self.u_e(x)**self.emp_coef_2
            if denom == 0:
                delta2 = None
            else:
                delta2 = (self.emp_coef_1*self.nu/denom*res)**.5
        elif isinstance(x, ARRAYTYPES):
            x = np.array(x, dtype=float)
            delta2 = []
            for xi in x:
                delta2.append(self.get_momentum_thikness(xi))
            delta2 = np.array(delta2, dtype=float)
        else:
            raise TypeError()
        return np.array(delta2)

    def get_BL_properties(self, x):
        """
        Return the boundary layer properties for given values of x

        Parameters
        ----------
        x : number or array of numbers
            Position along the flat plan until where we want to solve the
            momentum equation.
            (has no influence on the resoluion accuracy)

        Returns
        -------
        lambda : array of numbers
            Dimensionless pressure gradient.
            (According to Kays and Crawford, a value of -0.082 is caracteristic
            of the separation phenomenon.)
        delta2 : array of numbers
            Boundary layer momentum thickness
        delta_star : array of numbers
            Boundary layer displacement thickness
        H : array of numbers
            H factor
        """
        # get u_e and grad u_e
        u_e = np.array([self.u_e(xi) for xi in x])
        d_u_e = np.gradient(u_e)
        # get delta2
        delta2 = self.get_momentum_thikness(x)
        # get lambda
        lambd = delta2**2/self.nu*d_u_e
        # get H factor
        H = []
        for lambdi in lambd:
            if lambdi >= 0 and lambdi < 0.1:
                H.append(2.61 - 3.75*lambdi + 5.24*lambdi**2)
            elif lambdi < 0 and lambdi > -0.1:
                H.append(2.088 + 0.0731/(lambdi + 0.14))
            else:
                H.append(np.nan)
        H = np.array(H, dtype=float)
        # get delta_star
        delta_star = H*delta2
        return lambd, delta2, delta_star, H


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
        Ufin = Profile(y, Ufin, mask=False, unit_x=make_unit("m"),
                       unit_y=make_unit("m/s"))
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
        Kinematic viscosity (m²/s) (defaul = 1e-6)
    rho : number, optional
        liquid density (kg/m^3) (default = 1000)
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
        return (1./self.k*np.log(self.Utau*y/self.visc_c) + 5.1 +
                2.*Cc/self.k*np.sin(np.pi*y/(2.*self.delta))**2)


def get_bl_thickness(obj, direction=1, perc=0.95):
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
        value = obj.get_interpolated_values(y=maxi*perc)
        return value[0]
    elif isinstance(obj, ScalarField):
        if direction == 1:
            axe = obj.axe_x
        else:
            axe = obj.axe_y
        profiles = [obj.get_profile(direction, x) for x in axe]
        values = [get_bl_thickness(prof, perc=perc) for prof in profiles]
        return Profile(axe, values, unit_x=obj.unit_x, unit_y=obj.unit_y)
    else:
        raise TypeError("Can't compute (yet ?) BL thickness on this kind of"
                        " data : {}".format(type(obj)))


def get_clauser_thickness(obj, direction=1, rho=1000, nu=1e-6, tau=None):
    """
    Return the profile Clauser's thickness defined in 'Clauser (1956)'.
    (Delta_star = integrale_0_h (u_top - u)/u_star dy)

    Parameters
    ----------
    obj : Profile or ScalarField object
    direction : integer, optional
        If 'obj' is a ScalarField, determine the swept axis
        (1 for x and 2 for y).
    rho : number, optional
        Density of the fluid (default fo water : 1000 kg/m^3)
    nu : number, optional
        Kinematic viscosity for the fluid (default for water : 1e-6 m^2/s)
    tau : number, optional
        Wall shear stress, if not specified, 'get_shear_stress' is used to
        compute it.

    Returns
    -------
    Delta_star : float or Profile
        Boundary layer Clauser thickness, in axe x unit.
    """
    # if obj is a profile, getting Delta_star
    if isinstance(obj, Profile):
        # getting u_star
        if tau is None:
            tau = get_shear_stress(obj, direction=direction, nu=nu, rho=rho)
            tau.change_unit('y', 'kg/m/s**2')
            tau = tau.y[0]
        u_star = np.sqrt(tau/rho)
        # getting v_top
        v_top = obj.y[-1]
        Delta_star = get_displ_thickness(obj)*v_top/u_star
        return Delta_star
    # if obj is a scalarField
    elif isinstance(obj, ScalarField):
        if direction == 1:
            axe = obj.axe_x
        else:
            axe = obj.axe_y
        profiles = [obj.get_profile(direction, x) for x in axe]
        values = [get_clauser_thickness(prof, direction=direction, rho=rho,
                                        nu=nu)
                  for prof, _ in profiles]
        return Profile(axe, values, unit_x=obj.unit_x, unit_y=obj.unit_y)


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
        obj = obj.crop([0, bl_thick])
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
        values = [get_displ_thickness(prof) for prof in profiles]
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
        obj = obj.crop([0, bl_thick])
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
        values = [get_momentum_thickness(prof) for prof in profiles]
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


def get_shear_stress(obj, direction=1, method='simple',
                     respace=False, tau_w_guess=1e-6, rho=1000., nu=1.e-6):
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
        'wall_law_lin' : use the linear part of the 'law of the wall' model
        (need some points in the viscous sublayer)
        'wall_law_log' : use the log part of the 'law of the wall' model
        (only valid in the log layer)
    respace : bool, optional
        Use linear interpolation to create an evenly spaced profile.
    tau_w_guess : number, optional
        For 'Wall_law_log' method, initial guess for tau_w resolution.
    rho : number, optional
        Density of the fluid (default fo water : 1000 kg/m^3)
    nu : number, optional
        Kinematic viscosity for the fluid (default for water : 1e-6 m^2/s)
    """
    unit_visc = make_unit('m^2/s')
    unit_rho = make_unit('kg/m^3')
    # check parameters
    if not isinstance(nu, NUMBERTYPES):
        raise TypeError()
    if nu <= 0:
        raise ValueError()
    if direction not in [1, 2]:
        raise ValueError()
    # if obj is a profile
    if isinstance(obj, Profile):
        if method == 'simple':
            # respace if asked
            if respace:
                obj = obj.evenly_space('linear')
            # compute gradients and return shear stress
            tmp_prof = obj.get_gradient()*nu*rho*unit_visc*unit_rho
            return tmp_prof
        elif method == 'wall_law_lin':
            new_x = obj.x[obj.x > 0]
            new_y = obj.y[obj.x > 0]/new_x*nu*rho
            new_unit_y = obj.unit_y/obj.unit_x*unit_visc*unit_rho
            mask = obj.mask[obj.x > 0]
            return Profile(x=new_x, y=new_y, mask=mask, unit_x=obj.unit_x,
                           unit_y=new_unit_y, name=obj.name)
        elif method == 'wall_law_log':
            # getting data
            import scipy.optimize as spopt
            x = obj.x[obj.x > 0]
            y = obj.y[obj.x > 0]
            mask = obj.mask[obj.x > 0]

            # log law of the wall
            def func(u_star, U, rho, y, nu):
                u_star = u_star[0]
                k = 0.41
                C = 5.1
                # compute residual
                if u_star < 0:
                    res = -(U/u_star - 1./k*np.log(np.abs(y*u_star/nu)) -
                            C)
                elif u_star == 0:
                    res = U/u_star - 1./k*np.log(-1e5) - C
                else:
                    res = U/u_star - 1./k*np.log(y*u_star/nu) - C
                return res
            # solving
            u_stars = np.zeros(len(x))
            for i in np.arange(len(u_stars)):
                u_stars[i] = spopt.fsolve(func, (1e-6,),
                                          (y[i], rho, x[i], nu))
            tau_w = rho*np.array(u_stars)**2
            unit_tau = obj.unit_y/obj.unit_x*unit_visc*unit_rho
            # returning
            return Profile(x=x, y=tau_w, mask=mask, unit_x=obj.unit_x,
                           unit_y=unit_tau, name=obj.name)
        else:
            raise ValueError()
    elif isinstance(obj, ScalarField):
        # get axis
        if direction == 1:
            axe = obj.axe_x
        else:
            axe = obj.axe_y
        # get profiles
        profiles = [obj.get_profile(direction, pos_ind, ind=True)
                    for pos_ind in range(len(axe))]
        # get shear stresses
        values = np.zeros(obj.shape)
        for i in range(len(profiles)):
            if direction == 1:
                values[i, :] = get_shear_stress(profiles[i], direction=1,
                                                method=method, respace=False,
                                                tau_w_guess=tau_w_guess,
                                                rho=rho, nu=nu).y
            else:
                values[:, i] = get_shear_stress(profiles[i], direction=1,
                                                method=method, respace=False,
                                                tau_w_guess=tau_w_guess,
                                                rho=rho, nu=nu).y
        # returning
        mask = np.isnan(values)
        unit_values = profiles[0].unit_y
        tau_w = ScalarField()
        tau_w.import_from_arrays(obj.axe_x, obj.axe_y, values, mask=mask,
                                 unit_x=obj.unit_x, unit_y=obj.unit_y,
                                 unit_values=unit_values)
        return tau_w
    else:
        raise TypeError()


def get_separation_position(obj, wall_direction, wall_position,
                            interval=None, nmb_lines=4):
    """
    Compute and return the separation points position.
    Separation points position is computed by searching zero streamwise
    velocities on surrounding field lines and by extrapolating at
    the wanted 'wall_position'.
    If specified, 'interval' must include separation points on the 4 nearest
    field line.
    If multiples changments of streamwise velocity are found, return the mean
    positions of those points.

    Parameters
    ----------
    obj : ScalarField, VectorField, VectorField or TemporalVelocityField
        If 'VectorField' or 'VectorField', wall_direction is used to
        determine the interesting velocity component.
    wall_direction : integer
        1 for a wall at a given value of x,
        2 for a wall at a given value of y.
    wall_position : number
        Position of the wall.
    interval : 2x1 array of numbers, optional
        Optional interval in which search for the detachment points.
    nmb_lines : int
        Number of lines to take into account to make the extrapolation.
        (default is 4)
    """
    # checking parameters coherence
    if not isinstance(obj, (ScalarField, VectorField,
                            TemporalVectorFields)):
        raise TypeError("Unknown type for 'obj' : {}".format(type(obj)))
    if not isinstance(wall_direction, NUMBERTYPES):
        raise TypeError("'wall_direction' must be a number")
    if wall_direction != 1 and wall_direction != 2:
        raise ValueError("'wall_direction' must be 1 or 2")
    if not isinstance(wall_position, NUMBERTYPES):
        raise ValueError("'wall_position' must be a number")
    axe_x, axe_y = obj.axe_x, obj.axe_y
    if interval is None:
        if wall_direction == 2:
            interval = [np.min(axe_x), np.max(axe_x)]
        else:
            interval = [np.min(axe_y), np.max(axe_y)]
    if not isinstance(interval, ARRAYTYPES):
        raise TypeError("'interval' must be a array")

    # Get data according to 'obj' type
    if isinstance(obj, ScalarField):
        # checking masked values
        if np.any(obj.mask):
            warnings.warn("I can give weird results if masked values remains")
        V = obj.values_as_sf
        if wall_direction == 1:
            axe = axe_x
        else:
            axe = axe_y
    elif isinstance(obj, VectorField):
        if np.any(obj.mask):
            warnings.warn("I can give weird results if masked values remains")
        if wall_direction == 1:
            V = obj.comp_y_as_sf
            axe = axe_x
        else:
            V = obj.comp_x_as_sf
            axe = axe_y
    elif isinstance(obj, TemporalVectorFields):
        if np.any(obj.fields[0].mask):
            warnings.warn("I can give weird results if masked values remains")
        pts = []
        times = obj.times
        if wall_direction == 1:
            unit_axe = obj.unit_y
        else:
            unit_axe = obj.unit_x
        for field in obj.fields:
            pts.append(get_separation_position(field,
                                               wall_direction=wall_direction,
                                               wall_position=wall_position,
                                               interval=interval))
        return Profile(times, pts, unit_x=obj.unit_times,
                       unit_y=unit_axe)
    else:
        raise ValueError("Unknown type for 'obj'")

    # Getting separation position
    # Getting lines around wall
    if wall_position < axe[0]:
        lines_pos = axe[0:nmb_lines]
    elif wall_position > axe[-1]:
        lines_pos = axe[-nmb_lines-1:-1]
    else:
        inds = V.get_indice_on_axe(wall_direction, wall_position)
        if len(inds) == 1:
            inds = [inds[0], inds[0] + 1]
        if inds[0] - nmb_lines/2 < 0:
            lines_pos = axe[0:nmb_lines]
        elif inds[-1] + nmb_lines/2 > len(axe):
            lines_pos = axe[-nmb_lines-1:-1]
        else:
            lines_pos = axe[inds[0] - nmb_lines/2:inds[1] + nmb_lines/2]
    # Getting separation points on surrounding lines
    seps = np.array([])
    new_lines_pos = np.array([])
    for lp in lines_pos:
        # extraction one line
        tmp_profile = V.get_profile(wall_direction, lp)
        # getting the velocity sign changment on the line
        values = tmp_profile.get_interpolated_values(y=0)
        values = np.array(values)
        # masking with 'interval'
        values = values[np.logical_and(values > interval[0],
                                       values < interval[1])]
        if len(values) == 0:
            continue
        seps = np.append(seps, np.mean(values))
        new_lines_pos = np.append(new_lines_pos, lp)
    if len(seps) == 0:
        raise Exception("Can't find sign chagment on the given interval")
    elif len(seps) != nmb_lines:
        warnings.warn("extrapolation done on only {} points"
                      " instead of {}".format(len(seps), nmb_lines))
    lines_pos = new_lines_pos
    # Deleting lines where no separation points were found
    if np.any(np.isnan(seps)):
        warnings.warn("I can't find a separation points on one (or more)"
                      " line(s). You may want to change 'interval' values")
        seps = seps[~np.isnan(seps)]
        lines_pos = lines_pos[~np.isnan(seps)]
    interp = UnivariateSpline(lines_pos, seps, k=1)
    return float(interp(wall_position))
