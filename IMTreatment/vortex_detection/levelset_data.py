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


def get_sol():
    # Computed with 'compute_coefs()'
    s00 = '-dx*(-Vx_1*Vy_2*dy - Vx_1*Vy_3*dy + Vx_1*Vy_4*dy + Vx_2*Vy_1*dy + Vx_3*Vy_1*dy - Vx_4*Vy_1*dy + (dy*(2*Vx_1*Vy_2 - Vx_1*Vy_4 - 2*Vx_2*Vy_1 + Vx_2*Vy_3 - Vx_3*Vy_2 + Vx_4*Vy_1)/(2*(Vx_1*Vy_2 - Vx_1*Vy_4 - Vx_2*Vy_1 + Vx_2*Vy_3 - Vx_3*Vy_2 + Vx_3*Vy_4 + Vx_4*Vy_1 - Vx_4*Vy_3)) - sqrt(dy**2*(Vx_1**2*Vy_4**2 - 2*Vx_1*Vx_2*Vy_3*Vy_4 - 2*Vx_1*Vx_3*Vy_2*Vy_4 - 2*Vx_1*Vx_4*Vy_1*Vy_4 + 4*Vx_1*Vx_4*Vy_2*Vy_3 + Vx_2**2*Vy_3**2 + 4*Vx_2*Vx_3*Vy_1*Vy_4 - 2*Vx_2*Vx_3*Vy_2*Vy_3 - 2*Vx_2*Vx_4*Vy_1*Vy_3 + Vx_3**2*Vy_2**2 - 2*Vx_3*Vx_4*Vy_1*Vy_2 + Vx_4**2*Vy_1**2))/(2*Vx_1*Vy_2 - 2*Vx_1*Vy_4 - 2*Vx_2*Vy_1 + 2*Vx_2*Vy_3 - 2*Vx_3*Vy_2 + 2*Vx_3*Vy_4 + 2*Vx_4*Vy_1 - 2*Vx_4*Vy_3))*(Vx_1*Vy_2 - Vx_1*Vy_4 - Vx_2*Vy_1 + Vx_2*Vy_3 - Vx_3*Vy_2 + Vx_3*Vy_4 + Vx_4*Vy_1 - Vx_4*Vy_3))/(dy*(Vx_1*Vy_3 - Vx_1*Vy_4 - Vx_2*Vy_3 + Vx_2*Vy_4 - Vx_3*Vy_1 + Vx_3*Vy_2 + Vx_4*Vy_1 - Vx_4*Vy_2))'
    s01 = 'dy*(2*Vx_1*Vy_2 - Vx_1*Vy_4 - 2*Vx_2*Vy_1 + Vx_2*Vy_3 - Vx_3*Vy_2 + Vx_4*Vy_1)/(2*(Vx_1*Vy_2 - Vx_1*Vy_4 - Vx_2*Vy_1 + Vx_2*Vy_3 - Vx_3*Vy_2 + Vx_3*Vy_4 + Vx_4*Vy_1 - Vx_4*Vy_3)) - sqrt(dy**2*(Vx_1**2*Vy_4**2 - 2*Vx_1*Vx_2*Vy_3*Vy_4 - 2*Vx_1*Vx_3*Vy_2*Vy_4 - 2*Vx_1*Vx_4*Vy_1*Vy_4 + 4*Vx_1*Vx_4*Vy_2*Vy_3 + Vx_2**2*Vy_3**2 + 4*Vx_2*Vx_3*Vy_1*Vy_4 - 2*Vx_2*Vx_3*Vy_2*Vy_3 - 2*Vx_2*Vx_4*Vy_1*Vy_3 + Vx_3**2*Vy_2**2 - 2*Vx_3*Vx_4*Vy_1*Vy_2 + Vx_4**2*Vy_1**2))/(2*Vx_1*Vy_2 - 2*Vx_1*Vy_4 - 2*Vx_2*Vy_1 + 2*Vx_2*Vy_3 - 2*Vx_3*Vy_2 + 2*Vx_3*Vy_4 + 2*Vx_4*Vy_1 - 2*Vx_4*Vy_3)'
    s10 = '-dx*(-Vx_1*Vy_2*dy - Vx_1*Vy_3*dy + Vx_1*Vy_4*dy + Vx_2*Vy_1*dy + Vx_3*Vy_1*dy - Vx_4*Vy_1*dy + (dy*(2*Vx_1*Vy_2 - Vx_1*Vy_4 - 2*Vx_2*Vy_1 + Vx_2*Vy_3 - Vx_3*Vy_2 + Vx_4*Vy_1)/(2*(Vx_1*Vy_2 - Vx_1*Vy_4 - Vx_2*Vy_1 + Vx_2*Vy_3 - Vx_3*Vy_2 + Vx_3*Vy_4 + Vx_4*Vy_1 - Vx_4*Vy_3)) + sqrt(dy**2*(Vx_1**2*Vy_4**2 - 2*Vx_1*Vx_2*Vy_3*Vy_4 - 2*Vx_1*Vx_3*Vy_2*Vy_4 - 2*Vx_1*Vx_4*Vy_1*Vy_4 + 4*Vx_1*Vx_4*Vy_2*Vy_3 + Vx_2**2*Vy_3**2 + 4*Vx_2*Vx_3*Vy_1*Vy_4 - 2*Vx_2*Vx_3*Vy_2*Vy_3 - 2*Vx_2*Vx_4*Vy_1*Vy_3 + Vx_3**2*Vy_2**2 - 2*Vx_3*Vx_4*Vy_1*Vy_2 + Vx_4**2*Vy_1**2))/(2*Vx_1*Vy_2 - 2*Vx_1*Vy_4 - 2*Vx_2*Vy_1 + 2*Vx_2*Vy_3 - 2*Vx_3*Vy_2 + 2*Vx_3*Vy_4 + 2*Vx_4*Vy_1 - 2*Vx_4*Vy_3))*(Vx_1*Vy_2 - Vx_1*Vy_4 - Vx_2*Vy_1 + Vx_2*Vy_3 - Vx_3*Vy_2 + Vx_3*Vy_4 + Vx_4*Vy_1 - Vx_4*Vy_3))/(dy*(Vx_1*Vy_3 - Vx_1*Vy_4 - Vx_2*Vy_3 + Vx_2*Vy_4 - Vx_3*Vy_1 + Vx_3*Vy_2 + Vx_4*Vy_1 - Vx_4*Vy_2))'
    s11 = 'dy*(2*Vx_1*Vy_2 - Vx_1*Vy_4 - 2*Vx_2*Vy_1 + Vx_2*Vy_3 - Vx_3*Vy_2 + Vx_4*Vy_1)/(2*(Vx_1*Vy_2 - Vx_1*Vy_4 - Vx_2*Vy_1 + Vx_2*Vy_3 - Vx_3*Vy_2 + Vx_3*Vy_4 + Vx_4*Vy_1 - Vx_4*Vy_3)) + sqrt(dy**2*(Vx_1**2*Vy_4**2 - 2*Vx_1*Vx_2*Vy_3*Vy_4 - 2*Vx_1*Vx_3*Vy_2*Vy_4 - 2*Vx_1*Vx_4*Vy_1*Vy_4 + 4*Vx_1*Vx_4*Vy_2*Vy_3 + Vx_2**2*Vy_3**2 + 4*Vx_2*Vx_3*Vy_1*Vy_4 - 2*Vx_2*Vx_3*Vy_2*Vy_3 - 2*Vx_2*Vx_4*Vy_1*Vy_3 + Vx_3**2*Vy_2**2 - 2*Vx_3*Vx_4*Vy_1*Vy_2 + Vx_4**2*Vy_1**2))/(2*Vx_1*Vy_2 - 2*Vx_1*Vy_4 - 2*Vx_2*Vy_1 + 2*Vx_2*Vy_3 - 2*Vx_3*Vy_2 + 2*Vx_3*Vy_4 + 2*Vx_4*Vy_1 - 2*Vx_4*Vy_3)'
    sol = np.array([[s00, s01], [s10, s11]])
    return sol


def compute_coefs():
    try:
        import sympy
    except ImportError:
        raise Exception('You will need \'sympy\' to use this function')
        # prepare the analytical solution
    a1, b1, c1, d1, x, y = sympy.symbols('a1, b1, c1, d1, x, y')
    def funct(params, x, y, Vx):
        a1, b1, c1, d1 = tuple(params)
        return sympy.Eq(a1*x + b1*y + c1*x*y + d1, Vx)
    def funct1(dx, dy, Vx):
        return sympy.Eq(d1, Vx)
    def funct2(dx, dy, Vx):
        return sympy.Eq(a1*dx + d1, Vx)
    def funct3(dx, dy, Vx):
        return sympy.Eq(b1*dy + d1, Vx)
    def funct4(dx, dy, Vx):
        return sympy.Eq(a1*dx + b1*dy + c1*dx*dy + d1, Vx)
    # get the analytical solution for arbitrary Vx and Vy
    Vx_1, Vx_2, Vx_3, Vx_4 = sympy.symbols('Vx_1, Vx_2, Vx_3, Vx_4')
    Vy_1, Vy_2, Vy_3, Vy_4 = sympy.symbols('Vy_1, Vy_2, Vy_3, Vy_4')
    dx, dy = sympy.symbols('dx, dy')
    bl11 = funct1(dx, dy, Vx_1)
    bl12 = funct2(dx, dy, Vx_2)
    bl13 = funct3(dx, dy, Vx_3)
    bl14 = funct4(dx, dy, Vx_4)
    sol1 = sympy.solve([bl11, bl12, bl13, bl14], [a1, b1, c1, d1])
    params = [sol1[a1], sol1[b1], sol1[c1], sol1[d1]]
    eq1 = funct(params, x, y, 0.)
    eq2 = eq1.subs(Vx_1, Vy_1)
    eq2 = eq2.subs(Vx_2, Vy_2)
    eq2 = eq2.subs(Vx_3, Vy_3)
    eq2 = eq2.subs(Vx_4, Vy_4)
    sol = sympy.solve([eq1, eq2], [x, y])
    sol = np.array(sol)
    if sol.ndim == 1:
        sol = np.array([sol])
    for i in np.arange(sol.shape[0]):
        for j in np.arange(sol.shape[1]):
            sol[i, j] = sol[i, j].__repr__()
    return sol
