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

from IMTreatment import VectorField, ScalarField, Profile
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.path as mplpath

ARRAYTYPES = (np.ndarray, list, tuple)
NUMBERTYPES = (int, int, float, complex)
STRINGTYPES = (str, str)


# object class
class System(object):
    """
    Representing a potential system (boundaries, sources, freestream, ...)
    """
    def __init__(self, u_inf=0., alpha=0.):
        """
        Parameters
        ----------
        u_inf : number, optional
            Free Stream velocity (m/s).

        """
        self.u_inf = u_inf
        self.alpha = alpha
        self.objects_0D = []
        self.objects_1D = []
        self.objects_2D = []
        self.sigma_to_refresh = False

    def add_object(self, dim, coords, kind='wall', res=1):
        """
        Add an object to the system

        Parameters
        ----------
        dim : integer
            Can be '0' for a point, '1' for a boundary or '2' for a solid.
        coords : array of number
            Coordinates of the object (1x2 array if 'dim=1', nx2 arrays else).
        kind : string, optional
            Kind of object (default is 'wall').
        res : integer
            Resolution, length of the wanted segments.
        """
        # check
        if not isinstance(dim, NUMBERTYPES):
            raise TypeError()
        if not isinstance(coords, ARRAYTYPES):
            raise TypeError()
        coords = np.array(coords)
        # object_0D
        if dim == 0:
            self.objects_0D.append(object_0D(coords, kind, res=res))
        # object_1D
        elif dim == 1:
            self.objects_1D.append(object_1D(coords, kind, res=res))
        # object_2D
        elif dim == 2:
            self.objects_2D.append(object_2D(coords, kind, res=res))
        else:
            raise ValueError()
        #
        self.sigma_to_refresh = True

    def get_solid_panels(self):
        """
        Return all the solid panels.
        """
        panels = np.array([], dtype=object)
        for obj in self.objects_1D:
            if obj.kind == 'wall':
                panels = np.append(panels, obj.panels)
        for obj in self.objects_2D:
            if obj.kind == 'wall':
                panels = np.append(panels, obj.panels)
        panels = np.array(panels, subok=True)
        return panels.flatten()

    def get_solid_paths(self):
        """
        Return the solid paths.
        """
        paths = []
        for obj in self.objects_2D:
            if obj.kind == 'wall':
                paths.append(obj.path)
        return paths

    def solving_sigma(self):
        """
        Solve sigma values for the current boundaries.
        """
        # getting solid panels
        solid_panels = self.get_solid_panels()
        nmb_solid_panels = len(solid_panels)
        # computes the source influence matrix
        A = np.empty((nmb_solid_panels, nmb_solid_panels), dtype=float)
        np.fill_diagonal(A, 0.5)
        for i, p_i in enumerate(solid_panels):
            for j, p_j in enumerate(solid_panels):
                if i == j:
                    continue
                A[i, j] = 0.5/np.pi*integral(p_i.xc, p_i.yc, p_j,
                                             p_i.cosbeta, p_i.sinbeta)
        # computes the RHS of the linear system
        b = - self.u_inf * np.cos([self.alpha - p.beta for p in solid_panels])
        # solves the linear system
        sigma = np.linalg.solve(A, b)
        for i, panel in enumerate(solid_panels):
            panel.sigma = sigma[i]
        self.sigma_to_refresh = False

        # computes the matrix of the linear system
        A = np.empty((len(solid_panels), len(solid_panels)), dtype=float)
        np.fill_diagonal(A, 0.0)
        for i, p_i in enumerate(solid_panels):
            for j, p_j in enumerate(solid_panels):
                if i != j:
                    A[i, j] = 0.5/np.pi*integral(p_i.xc, p_i.yc, p_j,
                                                 -np.sin(p_i.beta),
                                                 np.cos(p_i.beta))
        # computes the RHS of the linear system
        b = self.u_inf * np.sin([self.alpha - panel.beta
                                 for panel in solid_panels])
        # computes the tangential velocity at each panel center-point
        vt = np.dot(A, sigma) + b
        for i, panel in enumerate(solid_panels):
            panel.vt = vt[i]
#
#        ### cp ###
#        for panel in self.solid_panels:
#            panel.cp = 1.0 - (panel.vt/u_inf)**2

    def compute_velocity(self, x, y):
        """
        Return the velocity at the given point.
        """
        # update sigma if necessary
        if self.sigma_to_refresh:
            self.solving_sigma()
        solid_panels = self.get_solid_panels()
        # check if on panel
        for pan in solid_panels:
            if pan.on_panel(x, y):
                return pan.vector
        # compute u
        u = self.u_inf*np.cos(self.alpha)\
            + 0.5/np.pi*sum([p.sigma*integral(x, y, p, 1, 0)
                             for p in solid_panels])
        # compute v
        v = self.u_inf*np.sin(self.alpha)\
            + 0.5/np.pi*sum([p.sigma*integral(x, y, p, 0, 1)
                             for p in solid_panels])
        # check if velocity isinstance not too high
        if np.abs(u) > 1e30 or np.abs(v) > 1e30:
            u = 0.
            v = 0.
        return u, v

    def compute_velocity_on_grid(self, grid_x, grid_y, raw=False,
                                 remove_solid=False):
        """
        Returns the velocity field on the given grid.
        """
        # update sigma if necessary
        if self.sigma_to_refresh:
            self.solving_sigma()
        # creating the grid
        X, Y = np.meshgrid(grid_x, grid_y)
        Nx, Ny = X.shape
        u, v = np.empty((Nx, Ny), dtype=float), np.empty((Nx, Ny), dtype=float)
        # mask if we don't want velocities in the solids
        paths = self.get_solid_paths()
        mask = np.zeros((Nx, Ny), dtype=bool)
        #compute velocities
        for i in range(Nx):
            for j in range(Ny):
                x = X[i, j]
                y = Y[i, j]
                if remove_solid:
                    in_solid = np.any([path.contains_point((x, y),
                                                           radius=-1e-10)
                                       for path in paths])
                    if in_solid:
                        mask[i, j] = True
                        continue
                u_i, v_i = self.compute_velocity(x, y)
                u[i, j] = u_i
                v[i, j] = v_i
        # returning
        if raw:
            return u, v
        else:
            vf = VectorField()
            u = np.transpose(u)
            v = np.transpose(v)
            mask = np.transpose(mask)
            vf.import_from_arrays(grid_x, grid_y, u, v, mask=mask,
                                  unit_x='m', unit_y='m', unit_values='m/s')
            return vf

    def compute_velocity_on_line(self, xya, xyb, res=100, raw=False,
                                 remove_solid=False):
        """
        Return the velocity on the given line.
        """
        # update sigma if necessary
        if self.sigma_to_refresh:
            self.solving_sigma()
        # creating the line
        xa, ya = xya
        xb, yb = xyb
        x = np.linspace(xa, xb, res)
        y = np.linspace(ya, yb, res)
        vx = np.empty(x.shape)
        vy = np.empty(x.shape)
        # mask if we don't want velocities on solid
        mask = np.zeros(x.shape)
        paths = self.get_solid_paths()
        # loop on line
        for i, _ in enumerate(x):
            if remove_solid:
                in_solid = np.any([path.contains_point((x[i], y[i]))
                                   for path in paths])
                if in_solid:
                    mask[i] = True
                    continue
            v = self.compute_velocity(x[i], y[i])
            vx[i] = v[0]
            vy[i] = v[1]
        # returning
        if raw:
            return vx, vy
        else:
            x = x - np.min(x)
            y = y - np.min(y)
            new_x = np.sqrt(x**2 + y**2) - np.sqrt(x[0]**2 + y[0]**2)
            prof_vx = Profile(new_x, vx, mask=mask, unit_x='m',
                              unit_y='m/s')
            prof_vy = Profile(new_x, vy, mask=mask, unit_x='m',
                              unit_y='m/s')
            return prof_vx, prof_vy

    def compute_pressure_from_velocity(self, obj, raw=False):
        """
        Return the pressure coefficient computed from velocity data.
        """
        if isinstance(obj, VectorField):
            vx = obj.comp_x
            vy = obj.comp_y
            grid_x = obj.axe_x
            grid_y = obj.axe_y
            mask = obj.mask
            cp = 1. - (vx**2 + vy**2)/self.u_inf**2
            mask = np.logical_or(np.isnan(cp), mask)
            if raw:
                return cp
            else:
                sf = ScalarField()
                sf.import_from_arrays(grid_x, grid_y, cp, mask=mask,
                                      unit_x='m', unit_y='m', unit_values='')
                return sf
        elif isinstance(obj, (tuple, list)):
            if isinstance(obj[0], Profile):
                vx = obj[0].y
                vy = obj[1].y
                mask = np.logical_or(obj[0].mask, obj[1].mask)
                cp = 1. - (vx**2 + vy**2)/self.u_inf**2
                if raw:
                    return cp
                else:
                    prof = Profile(obj[0].x, cp, mask=mask,
                                   unit_x=obj[0].unit_x,
                                   unit_y='')
                    return prof
            else:
                raise TypeError()
        else:
            raise TypeError()

    def display(self, solid=True, panels=True):
        """
        Display the current geometry.
        """
        plt.grid(True)
        for obj in self.objects_0D:
            obj.display(panels=panels)
        for obj in self.objects_1D:
            obj.display(panels=panels)
        for obj in self.objects_2D:
            obj.display(panels=panels, solid=solid)


class object_0D(object):
    """
    Representing a Source or a sink.
    """
    def __init__(self, x, y):
        """
        Representing a Source or a sink.

        Parameters
        ----------
        x, y : number
            Coordinate of the point.
        """
        # check
        if not isinstance(x, NUMBERTYPES):
            raise TypeError()
        if not isinstance(y, NUMBERTYPES):
            raise TypeError()
        # store
        self.x = x
        self.y = y


class object_1D(object):
    """
    Representing a boundary (wall, source or sink).
    """
    def __init__(self, coords, kind, res):
        """
        Representing a solid (wall, source or sink).

        Parameters
        ----------
        coords : nx2 array of numbers
            Coordinate of the boundary path.
        kind : string
            Kind of object (default is 'wall').
        res : integer
            Resolution, number of segments on the shortest line (default=1)
        """
        # check
        if not isinstance(coords, ARRAYTYPES):
            raise TypeError()
        coords = np.array(coords, dtype=float)
        if not coords.ndim == 2:
            raise ValueError()
        if not isinstance(kind, STRINGTYPES):
            raise TypeError()
        if not isinstance(res, int):
            raise TypeError()
        if res < 1:
            raise ValueError()
        # store
        self.x = coords[:, 0]
        self.y = coords[:, 1]
        self.nmb_panels = 0
        self.panels = np.empty((1,), dtype=object)
        self.path = None
        self.kind = kind
        # refine if necessary
        self.refine(res)
        # add the panels
        self._actualize_panels()
        # actualize the path
        self._actualize_path()
        # TODO : check normals and coherence

    def _actualize_panels(self):
        # Add the coresponding panels
        self.nmb_panels = len(self.x) - 1
        self.panels = np.empty(shape=(self.nmb_panels,),
                               dtype=object)
        # creating the panels objects
        for i in np.arange(self.nmb_panels):
                self.panels[i] = Panel([self.x[i], self.y[i]],
                                       [self.x[i + 1], self.y[i + 1]])

    def _actualize_path(self):
        for i in np.arange(self.nmb_panels):
            coords = list(zip(self.x, self.y))
            self.path = mplpath.Path(coords)

    def inverse_normals(self):
        """
        Inverse normals.
        """
        self.x = self.x[::-1]
        self.y = self.y[::-1]
        self._actualize_panels()
        self._actualize_path()

    def refine(self, res):
        """
        Refine each segments of the solid.
        """
        x = self.x
        y = self.y
        new_x = []
        new_y = []
        # getting delta (length of the panels)
        lens_x = np.array(x[1::] - x[0:-1])
        lens_y = np.array(y[1::] - y[0:-1])
        lens = (lens_x**2 + lens_y**2)**.5
        min_len = np.min(lens)
        delta = min_len/float(res)
        # refining
        for i in np.arange(len(self.x) - 1):
            tmp_nmb = int(np.round(lens[i]/delta))
            if tmp_nmb <= 1:
                new_x += [x[i]]
                new_y += [y[i]]
                continue
            new_x += list(np.linspace(x[i], x[i + 1], tmp_nmb + 1)[:-1])
            new_y += list(np.linspace(y[i], y[i + 1], tmp_nmb + 1)[:-1])
        # closing
        new_x += [x[0]]
        new_y += [y[0]]
        new_x = np.array(new_x, dtype=float)
        new_y = np.array(new_y, dtype=float)
        # storing new values
        self.x = new_x
        self.y = new_y
        # updating panels ans path
        self._actualize_panels()
        self._actualize_path()

    def display(self, panels=True):
        """
        Display the 1D object
        """
        plt.plot(self.x, self.y, 'k-')
        if panels:
            for pan in self.panels:
                pan.display(wall_vel=False, nodes=True)


class object_2D(object_1D):
    """
    Representing a solid (wall, source or sink).
    """

    def __init__(self, coords, kind, res):
        """
        Representing a solid (wall, source or sink).

        Parameters
        ----------
        coords : nx2 array of numbers
            Coordinate of the boundary path.
        kind : string
            Kind of object (default is 'wall').
        res : integer
            Resolution, lenght of the wanted segments.
        """
        # check
        if not isinstance(coords, ARRAYTYPES):
            raise TypeError()
        coords = np.array(coords, dtype=float)
        if not coords.ndim == 2:
            raise ValueError()
        if not isinstance(kind, STRINGTYPES):
            raise TypeError()
        # store
        self.x = coords[:, 0]
        self.y = coords[:, 1]
        self.nmb_panels = 0
        self.panels = np.empty((1,), dtype=object)
        self.path = None
        self.kind = kind
        # close the path if necessary
        if self.x[0] != self.x[-1] or self.y[0] != self.y[-1]:
            self.x = np.append(self.x, self.x[0])
            self.y = np.append(self.y, self.y[0])
        # refine if necessary
        self.refine(res)
        # add the panels
        self._actualize_panels()
        # actualize the path
        self._actualize_path()
        # TODO : check normals and coherence

    def display(self, solid=True, panels=True):
        """
        Display the 3D object
        """
        object_1D.display(self, panels=panels)
        if solid:
            plt.fill(self.x, self.y, 'k')


# Panel class (discretization element)
class Panel(object):
    """
    Contains information related to a panel.
    """
    def __init__(self, xya, xyb, sigma=0.):
        """Initializes the panel.

        Arguments
        ---------
        xa, ya -- coordinates of the first end-point of the panel.
        xb, yb -- coordinates of the second end-point of the panel.
        """
        # checl
        self.xa = xya[0]
        self.ya = xya[1]
        self.xb = xyb[0]
        self.yb = xyb[1]
        if self.xa == self.xb and self.ya == self.yb:
            raise ValueError()
        self.xc, self.yc = (self.xa + self.xb)/2., (self.ya + self.yb)/2.
        self.length = np.sqrt((self.xb - self.xa)**2 + (self.yb - self.ya)**2)
        self.length2 = self.length**2
        # orientation of the panel (angle between x-axis and panel's normal)
        if self.xb - self.xa <= 0.:
            self.beta = np.arccos((self.yb - self.ya)/self.length)
        elif self.xb - self.xa > 0.:
            self.beta = np.pi + np.arccos(-(self.yb - self.ya)/self.length)
        self.sinbeta = np.sin(self.beta)
        self.cosbeta = np.cos(self.beta)
        self.sigma = sigma                          # source strength
        self.vt = 0.                                # tangential velocity
        self.cp = 0.                                # pressure coefficient
        # precomputation to 'on_panel' test
        if self.xa == self.xb:
            self.vertical = True
        else:
            self.vertical = False
        if self.ya == self.yb:
            self.horizontal = True
        else:
            self.horizontal = False
        if not self.vertical and not self.horizontal:
            self.coef_a = (self.yb - self.ya)/(self.xb - self.xa)
            self.coef_b = self.yb - self.coef_a*self.xb
            self.coef_C1 = (1 + self.coef_a**2)**.5

    @property
    def vector(self):
        return (-self.vt*np.sin(self.beta), self.vt*np.cos(self.beta))

    def display(self, wall_vel=True, nodes=True):
        """
        Display the panel
        """
        # plotting the panels
        plt.plot([self.xa, self.xb], [self.ya, self.yb], 'k-')
        if nodes:
            plt.plot([self.xa, self.xb], [self.ya, self.yb], 'ro')
            plt.plot([self.xc], [self.yc], 'ok')
        if wall_vel:
            vx, vy = self.vector
            plt.quiver(self.xc, self.yc, vx, vy)
        return plt.gcf()

    def on_panel(self, x, y, eps_abs=1e-4):
        """
        Check if the point is on the panel.
        """
        # check if around the point c
        dist_c = (self.xc - x)**2 + (self.yc - y)**2
        if dist_c > self.length2/2.:
            return False
        # check if on the panle line
        if self.vertical:
            dist_line = np.abs(x - self.xc)
        elif self.horizontal:
            dist_line = np.abs(y - self.yc)
        else:
            dist_line = np.abs(self.coef_a*x - y + self.coef_b)/self.coef_C1
        if dist_line < eps_abs:
            return True
        else:
            return False


#def Point(object):
#    """
#    Contains information related to a point.
#    """
#
#    def __init__(self, x, y, sigma=0.):
#        self.x = x
#        self.y = y
#        self.sigma = sigma
#
#    def display(self, **kwargs):
#        plt.plot(self.x, self.y, 'o', **kwargs)

def integral(x, y, panel, dxdz, dydz):
    """Evaluates the contribution of a panel at one point.

    Arguments
    ---------
    x, y -- Cartesian coordinates of the point.
    panel -- panel which contribution is evaluated.
    dxdz -- derivative of x in the z-direction.
    dydz -- derivative of y in the z-direction.

    Returns
    -------
    Integral over the panel of the influence at one point.
    """
    sin_beta = panel.sinbeta
    cos_beta = panel.cosbeta
    # coef de calcul (for optimization purpose)
    pre_xs = x - panel.xa
    pre_ys = y - panel.ya
    a = pre_xs**2 + pre_ys**2
    b = 2*pre_xs*sin_beta - 2*pre_ys*cos_beta
    c = sin_beta**2 + cos_beta**2

    def func(s):
        xs = pre_xs + sin_beta*s
        ys = pre_ys - cos_beta*s
        denom = a + b*s + c*s**2
        return (xs*dxdz + ys*dydz)/denom
    res = integrate.quad(lambda s: func(s), 0., panel.length, limit=50)
    if res[1] > 1e-5:
        print("integration pb, check your geometry"
              "(very small panel somewhere ?) !")
        print((x, y))
        return res[0]
    else:
        return res[0]


def get_separation_position(D=1., Vd=1., x_obst=10., theta_max=np.inf,
                            nu=1e-6, res_pot=20, res_int=500):
    """
    Use potential flow to get velocity distribution and Thwaites BL
    equation to get the separation position.

    Parameters
    ----------
    D : number
        Obstacle diameter [m].
    Vd : number
        Bulk velocity [m/s].
    x_obst : number
        Distance from CL birth to obstacle [m].
    theta_max : number
        Maximum value of BL momentum thickness (in case of confinement) [m].
    nu : number
        Kinematic viscosity [].
    res_pot : integer
        Resolution for potential flow model (default=20).
    res_int : integer
        Resolution for Thwaites integral resolution (default=500).

    Returns
    -------
    x_sep : number
        Separation position
    """
    # creating system
    syst = System(u_inf=Vd, alpha=0.)
    syst.add_object(2, [[-D/2., -D/2.], [-D/2., D/2.],
                        [D/2., D/2.], [D/2., -D/2.]],
                    kind='wall', res=res_pot)
    syst.objects_2D[0].inverse_normals()
    syst.solving_sigma()

    # getting profiles along symmetry plane
    dx = x_obst
    x_f = -D/2.
    x_0 = x_f - dx
    prof_vit = syst.compute_velocity_on_line([x_f, 0], [x_0, 0], res=res_int,
                                             remove_solid=True)[0]
    # compute theta**2
    U = prof_vit.y
    theta_0 = 0.
    x_0 = -dx
    x_f = 0.
    x = np.linspace(x_0, x_f, res_int)
    U5 = prof_vit.y**5
    integral_U = [np.trapz(U5[0:i], prof_vit.x[0:i])
                  for i in np.arange(len(prof_vit.x))]
    theta2 = theta_0**2*(U[0]/U)**6 + 0.45*nu/U**6*integral_U
    theta2[theta2 > theta_max**2] = theta_max**2
    # compute m
    deriv_U = np.gradient(U, x[1]-x[0])
    m = -theta2/nu*deriv_U
    # find separation
    m = Profile(x, m, unit_x='m', unit_y='')
    x_sep = m.get_value_position(0.09)[0]
    return -x_sep


def get_gradP_length(D=1., Vd=1., perc=0.1, nu=1e-6, res_pot=20, res_int=500,
                     eps=1e-6):
    """
    Return the position before the obstacle where the pressure is equal
    at the wanted percentage of the pressure at the obstacle,
    using potential flow theory.

    Parameters
    ----------
    D : number
        Obstacle diameter [m].
    Vd : number
        Bulk velocity [m/s].
    perc : number
        wanted percentage (default to 0.1).
    nu : number
        Kinematic viscosity [].
    res_pot : integer
        Resolution for potential flow model (default=20).
    res_int : integer
        Resolution for Thwaites integral resolution (default=500).
    eps : number
        Wanted precision on the position (for the iterative solver)
    Returns
    -------
    dP_len : number
        Position of 10% pressure.
    """
    # creating system
    syst = System(u_inf=Vd, alpha=0.)
    syst.add_object(2, [[-D/2., -D/2.], [-D/2., D/2.],
                        [D/2., D/2.], [D/2., -D/2.]],
                    kind='wall', res=res_pot)
    syst.objects_2D[0].inverse_normals()
    syst.solving_sigma()

    # iterative finder
    dx = 5*D
    x_f = -D/2.
    x_0 = x_f - dx
    res_int = 3
    P_max = 1.
    i = 0
    while True:
        i += 1
        # getting profiles along symmetry plane
        prof_vit = syst.compute_velocity_on_line([x_f, 0], [x_0, 0],
                                                 res=res_int,
                                                 remove_solid=True)
        prof_P = syst.compute_pressure_from_velocity(prof_vit)
        # get 10% pressure
        ind = np.where(prof_P.y < perc*P_max)[0]
        # check if on the good interval
        if len(ind) == 0:
            x_0 -= dx
            continue
        else:
            ind = ind[-1]
        # return if precision is achieved
        if np.abs(perc*P_max - prof_P.y[ind]) < eps:
            x_pos = -(prof_P.x[ind] + x_f)
            return x_pos
        # raise error if too man iterations
        if i > 100:
            raise Exception()
        # another run !
        x_0 = prof_P.x[ind] + x_f
        x_f = prof_P.x[ind + 1] + x_f
