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
import re
import unum
import unum.units as units
try:
    units.counts = unum.Unum.unit('counts')
    units.px = unum.Unum.unit('px')
except:
    pass
from ..utils.types import ARRAYTYPES, INTEGERTYPES, NUMBERTYPES, STRINGTYPES


def make_unit(string):
    """
    Function helping for the creation of units. For more details, see the
    Unum module documentation.

    Parameters
    ----------
    string : string
        String representing some units.

    Returns
    -------
    unit : unum.Unum object
        unum object representing the given units

    Examples
    --------
    >>> make_unit("m/s")
    1 [m/s]
    >>> make_unit("N/m/s**3")
    1 [kg/s4]
    """
    if string == "":
        return unum.Unum({})
    # Safe check
    forbidden = ['import', '=', '\n', ';', ':', '"', "'", "open"]
    forbidden += [key for key in globals().keys()]
    for f in forbidden:
        if f in string:
            raise Exception("Unauthorized string: {}".format(f))
    if re.match("[a-zA-Z]+\(.*\)", string):
        raise Exception("No call allowed here")
    # Replace '^' by '**'
    string = string.replace('^', '**')
    # exec
    env = {}
    exec("from unum.units import *;res = {}".format(string), env)
    return env['res']


def make_unit_old(string):
    """
    Function helping for the creation of units. For more details, see the
    Unum module documentation.

    Parameters
    ----------
    string : string
        String representing some units.

    Returns
    -------
    unit : unum.Unum object
        unum object representing the given units

    Examples
    --------
    >>> make_unit("m/s")
    1 [m/s]
    >>> make_unit("N/m/s**3")
    1 [kg/s4]
    """
    if len(string) == 0:
        return unum.Unum({})
    brackets = ['(', ')', '[', ']']
    symbambig = {"**": "^"}
    operators = ['*', '^', '/']
    toremove = [' ']

    def spliting(string):
        """
        Split the given string to elemental brick.
        """
        # suppression symboles inutiles
        for symb in toremove:
            string = string.replace(symb, '')
        # remplacement symboles ambigues
        for key in symbambig:
            string = string.replace(key, symbambig[key])
        # découpage de la chaine de caractère
        pieces = [string]
        for symb in operators + brackets:
            j = len(pieces)-1
            while True:
                if (pieces[j].find(symb) != -1 and
                        len(pieces[j]) != len(symb)):
                    splitpiece = list(pieces[j].partition(symb))
                    pieces[j:j+1] = splitpiece
                    j += 2
                else:
                    j -= 1
                if j < 0:
                    break
        # Suppression des caractères nuls
        for j in np.arange(len(pieces)-1, -1, -1):
            if pieces[j] == '':
                pieces[j:j+1] = []
        # Add power where needed
        for i, piece in enumerate(pieces):
            match = re.match('([a-zA-Z]+)([0-9]+)', piece)
            if match is not None:
                pieces[i:i+1] = [match.groups()[0], "^", match.groups()[1]]
        return pieces

    def app_brackets(strlist):
        """
        Apply all the brackets.
        """
        level = 0
        levels = []
        for i in np.arange(len(strlist)):
            if strlist[i] in ['(', '[']:
                level += 1
            elif strlist[i-1] in [')', ']'] and i-1 > 0:
                level -= 1
            if level < 0:
                raise ValueError("I think you have a problem"
                                 " with your brackets, dude")
            levels.append(level)
        # boucle sur les parenthèses
        while True:
            if len(strlist) == 1:
                return app_one_bracket(strlist)
            first_ind = levels.index(np.max(levels))
            last_ind = None
            for i in np.arange(first_ind, len(levels)):
                if levels[i] < levels[first_ind]:
                    last_ind = i-1
                    break
            if last_ind is None:
                last_ind = len(levels) - 1
            if np.all(np.array(levels) == 0):
                result = app_one_bracket(strlist)
                return result
            else:
                strlist[first_ind:last_ind+1] \
                    = [app_one_bracket(strlist[first_ind+1:last_ind])]
            levels[first_ind:last_ind+1] = [np.max(levels)-1]

    def app_one_bracket(stringlist):
        """
        Apply one bracket.
        """
        stringlist = list(stringlist)
        # traitement des nombres
        for j in np.arange(len(stringlist)-1, -1, -1):
            try:
                stringlist[j] = float(stringlist[j])
                if stringlist[j] % 1 == 0:
                    stringlist[j] = int(stringlist[j])
            except:
                pass
        # traitement des unités
        for j in np.arange(len(stringlist)-1, -1, -1):
            if not isinstance(stringlist[j], (unum.Unum, NUMBERTYPES)):
                if not stringlist[j] in operators + brackets:
                    if not stringlist[j] in unum.Unum.getUnitTable():
                        raise ValueError("I don't know this unit : '{}',"
                                         "little bird. (or maybe it's an "
                                         "operator i'm missing...)."
                                         .format(stringlist[j]))
                    stringlist[j] = unum.Unum({stringlist[j]: 1})
        # traitement des opérateurs
        liste = stringlist
        # ^
        for ind in np.arange(len(liste)-1, 0, -1):
            if isinstance(liste[ind], unum.Unum):
                continue
            if liste[ind] == '^':
                liste[ind - 1:ind + 2] = [liste[ind - 1]**liste[ind + 1]]
                ind -= 2
                if ind < 0:
                    break
        # /
        for ind in np.arange(len(liste)-1, 0, -1):
            if isinstance(liste[ind], unum.Unum):
                continue
            if liste[ind] == '/':
                liste[0:1] = [liste[0]/liste[ind + 1]]
                liste[ind:ind+2] = []
                #liste[ind - 1:ind + 2] = [liste[ind - 1]/liste[ind + 1]]
                ind -= 2
                if ind < 0:
                    break
        # *
        for ind in np.arange(len(liste)-1, 0, -1):
            if isinstance(liste[ind], unum.Unum):
                continue
            if liste[ind] == '*':
                liste[ind - 1:ind + 2] = [liste[ind - 1]*liste[ind + 1]]
                ind -= 2
                if ind < 0:
                    break
        return liste[0]
    strlist = spliting(string)
    unity = app_brackets(strlist)
    return unity
