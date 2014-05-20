# -*- coding: utf-8 -*-
"""
Module IMTreatment.

    Auteur : Gaby Launay
"""

import scipy.interpolate as spinterp
import scipy.ndimage.measurements as msr
import scipy.io as spio
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import pdb
import timeit
import sets
import glob
import unum
import unum.units as units
import copy
import os
from scipy import ndimage
try:
    units.counts = unum.Unum.unit('counts')
    units.pixel = unum.Unum.unit('pixel')
except:
    pass


ARRAYTYPES = (np.ndarray, list, tuple)
NUMBERTYPES = (int, long, float, complex)
STRINGTYPES = (str, unicode)
MYTYPES = ('Profile', 'ScalarField', 'VectorField', 'VelocityField',
           'VelocityFields', 'TemporalVelocityFields', 'patialVelocityFields')


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
    if not isinstance(string, STRINGTYPES):
        raise TypeError("Units should be define by a string, big boy")
    if len(string) == 0:
        return unum.Unum({})
    brackets = ['(', ')']
    symbambig = {"**": "^"}
    operators = ['*', '^', '/']

    def spliting(string):
        """
        Split the given string to elemental brick.
        """
        #remplacement symboles ambigues
        for key in symbambig:
            string = string.replace(key, symbambig[key])
        #découpage de la chaine de caractère
        pieces = [string]
        for symb in operators + brackets:
            j = len(pieces)-1
            while True:
                if (pieces[j].find(symb) != -1
                        and len(pieces[j]) != len(symb)):
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
        return pieces

    def app_brackets(strlist):
        """
        Apply all the brackets.
        """
        level = 0
        levels = []
        for i in np.arange(len(strlist)):
            if strlist[i] == '(':
                level += 1
            elif (strlist[i-1] == ')') and (i-1 > 0):
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
                liste[ind - 1:ind + 2] = [liste[ind - 1]/liste[ind + 1]]
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


class Points(object):
    """
    Class representing a set of points.
    You can use 'make_unit' to provide unities.

    Parameters
    ----------
    xy : tuple of 2x1 arrays or tuple of 3x1 arrays.
        Representing the coordinates of each point of the set.
    v : array, optional
        Representing values attached at each points.
    unit_x : Unit object, optional
        X unit_y.
    unit_y : Unit object, optional
        Y unit_y.
    unit_v : Unit object, optional
        values unit_y.
    name : string, optional
        Name of the points set
    """
    def __init__(self, xy=[], v=None, unit_x=make_unit(''),
                 unit_y=make_unit(''),
                 unit_v=make_unit(''), name=None):
        """
        Points builder.
        """
        if not isinstance(xy, ARRAYTYPES):
            raise TypeError("'xy' must be a tuple of 2x1 arrays")
        xy = np.array(xy, subok=True, dtype=float)
        if xy.shape != (0,) and (len(xy.shape) != 2 or xy.shape[1] != 2):
            raise ValueError("'xy' must be a tuple of 2x1 arrays")
        if v is not None:
            if not isinstance(v, ARRAYTYPES):
                raise TypeError("'v' must be an array")
            v = np.array(v, dtype=float)
            if not xy.shape[0] == v.shape[0]:
                raise ValueError("'v' ans 'xy' must have the same dimensions")
        if not isinstance(unit_x, unum.Unum)   \
                or not isinstance(unit_y, unum.Unum):
            raise TypeError("'unit_x' and 'unit_y' must be Unit objects")
        if name is not None:
            if not isinstance(name, STRINGTYPES):
                raise TypeError("'name' must be a string")
        self.__classname__ = 'Points'
        self.xy = xy
        self.v = v
        self.unit_v = unit_v
        self.unit_x = unit_x
        self.unit_y = unit_y

        self.name = name

    def __iter__(self):
        if self.v is None:
            for i in np.arange(len(self.xy)):
                yield self.xy[i]
        else:
            for i in np.arange(len(self.xy)):
                yield self.xy[i], self.v[i]

    def __len__(self):
        return self.xy.shape[0]

    def import_from_ascii(self, filename, x_col=1, y_col=2, v_col=None,
                          unit_x=make_unit(""), unit_y=make_unit(""),
                          unit_v=make_unit(""), **kwargs):
        """
        Import a Points object from an ascii file.

        Parameters
        ----------
        x_col, y_col, v_col : integer, optional
            Colonne numbers for the given variables
            (begining at 1).
        unit_x, unit_y, unit_v : Unit objects, optional
            Unities for the given variables.
        **kwargs :
            Possibles additional parameters are the same as those used in the
            numpy function 'genfromtext()' :
            'delimiter' to specify the delimiter between colonnes.
            'skip_header' to specify the number of colonne to skip at file
                begining
            ...
        """
        # validating parameters
        if v_col is None:
            v_col = 0
        if not isinstance(x_col, int) or not isinstance(y_col, int)\
                or not isinstance(v_col, int):
            raise TypeError("'x_col', 'y_col' and 'v_col' must be integers")
        if x_col < 1 or y_col < 1:
            raise ValueError("Colonne number out of range")
        # 'names' deletion, if specified (dangereux pour la suite)
        if 'names' in kwargs:
            kwargs.pop('names')
        # extract data from file
        data = np.genfromtxt(filename, **kwargs)
        # get axes
        x = data[:, x_col-1]
        y = data[:, y_col-1]
        self.xy = zip(x, y)
        if v_col != 0:
            v = data[:, v_col-1]
        else:
            v = None
        self.__init__(zip(x, y), v, unit_x, unit_y, unit_v)

    def export_to_vtk(self, filepath, axis=None, line=False):
        """
        Export the scalar field to a .vtk file, for Mayavi use.

        Parameters
        ----------
        filepath : string
            Path where to write the vtk file.
        axis : tuple of strings, optional
            By default, points field axe are set to (x,y), if you want
            different axis, you have to specified them here.
            For example, "('z', 'y')", put the x points field axis values
            in vtk z axis, and y points field axis in y vtk axis.
        line : boolean, optional
            If 'True', lines between points are writen instead of points.
        """
        import pyvtk
        if not os.path.exists(os.path.dirname(filepath)):
            raise ValueError("'filepath' is not a valid path")
        if axis is None:
            axis = ('x', 'y')
        if not isinstance(axis, ARRAYTYPES):
            raise TypeError("'axis' must be a 2x1 tuple")
        if not isinstance(axis[0], STRINGTYPES) \
                or not isinstance(axis[1], STRINGTYPES):
            raise TypeError("'axis' must be a 2x1 tuple of strings")
        if not axis[0] in ['x', 'y', 'z'] or not axis[1] in ['x', 'y', 'z']:
            raise ValueError("'axis' strings must be 'x', 'y' or 'z'")
        if axis[0] == axis[1]:
            raise ValueError("'axis' strings must be different")
        if not isinstance(line, bool):
            raise TypeError("'line' must be a boolean")
        v = self.v
        x = self.xy[:, 0]
        y = self.xy[:, 1]
        if v is None:
            v = np.zeros(self.xy.shape[0])
        point_data = pyvtk.PointData(pyvtk.Scalars(v, 'Points values'))
        x_vtk = np.zeros(self.xy.shape[0])
        y_vtk = np.zeros(self.xy.shape[0])
        z_vtk = np.zeros(self.xy.shape[0])
        if axis[0] == 'x':
            x_vtk = x
        elif axis[0] == 'y':
            y_vtk = x
        else:
            z_vtk = x
        if axis[1] == 'x':
            x_vtk = y
        elif axis[1] == 'y':
            y_vtk = y
        else:
            z_vtk = y
        pts = zip(x_vtk, y_vtk, z_vtk)
        vertex = np.arange(x_vtk.shape[0])
        if line:
            grid = pyvtk.UnstructuredGrid(pts, line=vertex)
        else:
            grid = pyvtk.UnstructuredGrid(pts, vertex=vertex)
        data = pyvtk.VtkData(grid, 'Scalar Field from python', point_data)
        data.tofile(filepath)

    def export_to_matlab(self, filepath, name, **kwargs):
        """
        Write the point object in a amatlab file.

        Parameters
        ----------
        filepath : string
        global_name : string, optional
            If specified, 'x', 'y' and 'v' values are stored in a matlab
            structure object name 'global_name'.
        """
        from .file_operation import export_to_matlab
        dic = export_to_matlab(self, name)
        spio.savemat(self, filepath, dic, **kwargs)

    def copy(self):
        """
        Return a copy of the Points object.
        """
        tmp_pts = Points(copy.copy(self.xy), copy.copy(self.v),
                         unit_x=self.unit_x.copy(), unit_y=self.unit_y.copy(),
                         unit_v=self.unit_v.copy(), name=self.name)
        return tmp_pts

    def __add__(self, another):
        if isinstance(another, Points):
            try:
                self.unit_x + another.unit_x
                self.unit_y + another.unit_y
                if self.v is not None and another.v is not None:
                    self.unit_v + another.unit_v
            except unum.IncompatibleUnitsError:
                raise ValueError("Units system are not the same")
            xy = another.xy
            xy[:, 0] = xy[:, 0]*(self.unit_x/another.unit_x).asNumber()
            xy[:, 1] = xy[:, 1]*(self.unit_y/another.unit_y).asNumber()
            if self.v is None or another.v is None:
                return Points(np.append(self.xy, xy, axis=0),
                              unit_x=self.unit_x,
                              unit_y=self.unit_y)
            else:
                v_tmp = another.v*(self.unit_v/another.unit_v).asNumber()
                v = np.append(self.v, v_tmp)
                return Points(np.append(self.xy, xy, axis=0), v,
                              unit_x=self.unit_x,
                              unit_y=self.unit_y,
                              unit_v=self.unit_v)

        else:
            raise StandardError("You can't add {} to Points objects"
                                .format(type(another)))

    def _display(self, kind=None, **plotargs):
        if kind is None:
            if self.v is None:
                kind = 'plot'
            else:
                kind = 'scatter'
        if kind == 'scatter':
            if self.v is None:
                plot = plt.scatter(self.xy[:, 0], self.xy[:, 1], **plotargs)
            else:
                if not 'cmap' in plotargs:
                    plotargs['cmap'] = plt.cm.jet
                if not 'c' in plotargs:
                    plotargs['c'] = self.v
                plot = plt.scatter(self.xy[:, 0], self.xy[:, 1], **plotargs)
        elif kind == 'plot':
            plot = plt.plot(self.xy[:, 0], self.xy[:, 1], **plotargs)
        return plot

    def remove(self, ind):
        """
        Remove the point number 'ind' of the points cloud.
        In place.

        Parameters
        ----------
        ind : integer or array of integer
        """
        if isinstance(ind, int):
            ind = [ind]
        elif isinstance(ind, ARRAYTYPES):
            if not np.all([isinstance(val, int) for val in ind]):
                raise TypeError("'ind' must be an integer or an array of"
                                " integer")
            ind = np.array(ind)
        else:
            raise TypeError("'ind' must be an integer or an array of integer")
        self.xy = np.delete(self.xy, ind, axis=0)
        self.v = np.delete(self.v, ind, axis=0)

    def display(self, kind=None, **plotargs):
        """
        Display the set of points.

        Parameters
        ----------
        kind : string, optional
            Can be 'plot' (default if points have not values).
            or 'scatter' (default if points have values).
        """
        plot = self._display(kind, **plotargs)
        if self.v is not None and kind == 'scatter':
            cb = plt.colorbar(plot)
            cb.set_label(self.unit_v.strUnit())
        plt.xlabel('X ' + self.unit_x.strUnit())
        plt.ylabel('Y ' + self.unit_y.strUnit())
        if self.name is None:
            plt.title('Set of points')
        else:
            plt.title(self.name)
        return plot

    def trim(self, interv_x=None, interv_y=None):
        """
        Return a trimmed point cloud.

        Parameters
        ----------
        interv_x : 2x1 tuple
            Interval on x axis
        interv_y : 2x1 tuple
            Interval on y axis

        Returns
        -------
        tmp_pts : Points object
            Trimmed version of the point cloud.
        """
        tmp_pts = self.copy()
        mask = np.zeros(len(self.xy))
        if interv_x is not None:
            out_zone = np.logical_or(self.xy[:, 0] < interv_x[0],
                                     self.xy[:, 0] > interv_x[1])
            mask = np.logical_or(mask, out_zone)
        if interv_y is not None:
            out_zone = np.logical_or(self.xy[:, 1] < interv_y[0],
                                     self.xy[:, 1] > interv_y[1])
            mask = np.logical_or(mask, out_zone)
        tmp_pts.xy = tmp_pts.xy[~mask, :]
        if tmp_pts.v is not None:
            tmp_pts.v = tmp_pts.v[~mask]
        return tmp_pts

    def cut(self, interv_x=None, interv_y=None):
        """
        Return a point cloud where the given area has been removed.

        Parameters
        ----------
        interv_x : 2x1 tuple
            Interval on x axis
        interv_y : 2x1 tuple
            Interval on y axis

        Returns
        -------
        tmp_pts : Points object
            Cutted version of the point cloud.
        """
        tmp_pts = self.copy()
        mask = np.ones(len(self.xy))
        if interv_x is not None:
            out_zone = np.logical_and(self.xy[:, 0] > interv_x[0],
                                      self.xy[:, 0] < interv_x[1])
            mask = np.logical_and(mask, out_zone)
        if interv_y is not None:
            out_zone = np.logical_and(self.xy[:, 1] > interv_y[0],
                                      self.xy[:, 1] < interv_y[1])
            mask = np.logical_and(mask, out_zone)
        tmp_pts.xy = tmp_pts.xy[~mask, :]
        if tmp_pts.v is not None:
            tmp_pts.v = tmp_pts.v[~mask]
        return tmp_pts

    def reverse(self):
        """
        Return a Points object where x and y axis are swaped.
        """
        tmp_pt = self.copy()
        xy_tmp = tmp_pt.xy*0
        xy_tmp[:, 0] = tmp_pt.xy[:, 1]
        xy_tmp[:, 1] = tmp_pt.xy[:, 0]
        tmp_pt.xy = xy_tmp
        return tmp_pt

    def decompose(self):
        """
        return a tuple of Points object, with only one point per object.
        """
        if len(self) == 1:
            raise StandardError()
        if len(self) != len(self.v):
            raise StandardError()
        pts_tupl = []
        for i in np.arange(len(self)):
            pts_tupl.append(Points([self.xy[i]], [self.v[i]], self.unit_x,
                                   self.unit_y, self.unit_v, self.name))
        return pts_tupl

    def fit(self, kind='polynomial', order=2, simplify=False):
        """
        Return the parametric coefficients of the fitting curve on the points.

        Parameters
        ----------
        kind : string, optional
            The kind of fitting used. Can be 'polynomial' or 'ellipse'.
        order : integer
            Approximation order for the fitting.
        Simplify : boolean or string, optional
            Can be False (default), 'x' or 'y'. Perform a simplification
            (see Points.Siplify()) before the fitting.

        Returns
        -------
        p : array, only for polynomial fitting
            Polynomial coefficients, highest power first
        radii : array, only for ellipse fitting
            Ellipse demi-axes radii.
        center : array, only for ellipse fitting
           Ellipse center coordinates.
        alpha : number
            Angle between the x axis and the major axis.
        """
        if not isinstance(order, int):
            raise TypeError("'order' must be an integer")
        if not isinstance(kind, STRINGTYPES):
            raise TypeError("'kind' must be a string")
        if not simplify:
            xytmp = self.xy
        elif simplify == 'x':
            xytmp = self.simplify(axe=0).xy
        elif simplify == 'y':
            xytmp = self.simplify(axe=1).xy

        if kind == 'polynomial':
            p = np.polyfit(xytmp[:, 0], xytmp[:, 1], order)
            return p
        elif kind == 'ellipse':
            import fit_ellipse as fte
            res = fte.fit_ellipse(xytmp)
            radii, center, alpha = fte.get_parameters(res)
            return radii, center, alpha


class Profile(object):
    """
    Class representing a profile.
    You can use 'make_unit' to provide unities.

    Parameters
    ----------
    x, y : arrays
        Profile values.
    unit_x, unit_y : Unit objects
        Values unities.
    name : string, optionnal
        A name for the profile.
    """

    def __init__(self, x=[], y=[], unit_x=make_unit(""), unit_y=make_unit(""),
                 name=""):
        """
        Profile builder.
        """
        if not isinstance(x, ARRAYTYPES):
            raise TypeError("'x' must be an array")
        if not isinstance(x, (np.ndarray, np.ma.MaskedArray)):
            x = np.array(x, dtype=float)
        if not isinstance(y, ARRAYTYPES):
            raise TypeError("'y' must be an array")
        if not isinstance(y, (np.ndarray, np.ma.MaskedArray)):
            y = np.array(y, dtype=float)
        if not isinstance(name, STRINGTYPES):
            raise TypeError("'name' must be a string")
        if not isinstance(unit_x, unum.Unum):
            raise TypeError("'unit_x' must be a 'Unit' object")
        if not isinstance(unit_y, unum.Unum):
            raise TypeError("'unit_y' must be a 'Unit' object")
        if not len(x) == len(y):
            raise ValueError("'x' and 'y' must have the same length")
        self.__classname__ = "Profile"
        order = np.argsort(x)
        self.x = x[order]
        self.y = y[order]
        self.name = name
        self.unit_x = unit_x.copy()
        self.unit_y = unit_y.copy()

    def __neg__(self, otherone):
        if isinstance(otherone, NUMBERTYPES):
            y = self.y - otherone
            name = self.name
        elif isinstance(otherone, unum.Unum):
            y = self.y - otherone/self.unit_y
            name = self.name
        elif isinstance(otherone, Profile):
            try:
                self.unit_x + otherone.unit_x
                self.unit_y + otherone.unit_y
            except:
                raise ValueError("Profiles have not the same unit system")
            if not len(self.x) == len(otherone.x):
                raise ValueError("Profiles have not the same length")
            if not all(self.x == otherone.x):
                raise ValueError("Profiles have not the same x axis")
            y = self.y - self.unit_y/otherone.unit_y*otherone.y
            name = "{0} - {1}".format(self.name, otherone.name)
        else:
            raise TypeError("You only can substract Profile with "
                            "Profile or number")
        return Profile(self.x, y, self.unit_x, self.unit_y, name=name)

    def __add__(self, otherone):
        if isinstance(otherone, NUMBERTYPES):
            y = self.y + otherone
            name = self.name
        elif isinstance(otherone, unum.Unum):
            y = self.y + otherone/self.unit_y
            name = self.name
        elif isinstance(otherone, Profile):
            try:
                self.unit_x + otherone.unit_x
                self.unit_y + otherone.unit_y
            except:
                raise ValueError("Profiles have not the same unit system")
            if not len(self.x) == len(otherone.x):
                raise ValueError("Profiles have not the same length")
            if not all(self.x == otherone.x):
                raise ValueError("Profiles have not the same x axis")
            y = self.y + (self.unit_y/otherone.unit_y*otherone.y).asNumber()
            name = ""
        else:
            raise TypeError("You only can substract Profile with "
                            "Profile or number")
        return Profile(self.x, y, self.unit_x, self.unit_y, name=name)

    __radd__ = __add__

    def __mul__(self, otherone):
        if isinstance(otherone, NUMBERTYPES):
            y = self.y*otherone
            name = self.name
            unit_y = self.unit_y
        elif isinstance(otherone, unum.Unum):
            tmpunit = otherone/self.unit_y
            y = self.y*tmpunit.asunit()
            name = self.name
            unit_y = self.unit_y*(tmpunit/tmpunit.asNumber())
        else:
            raise TypeError("You only can multiply Profile with number")
        return Profile(self.x, y, self.unit_x, unit_y, name=name)

    __rmul__ = __mul__

    def __truediv__(self, otherone):
        if isinstance(otherone, NUMBERTYPES):
            y = self.y/otherone
            name = self.name
            unit_y = self.unit_y
        elif isinstance(otherone, unum.Unum):
            tmpunit = self.unit_y/otherone
            y = self.y*(tmpunit.asNumber())
            name = self.name
            unit_y = tmpunit/tmpunit.asNumber()
        elif isinstance(otherone, Profile):
            if not np.all(self.x == otherone.x):
                raise ValueError("Profile has to have identical x axis in "
                                 "order to divide them")
            else:
                tmp_unit = self.unit_y/otherone.unit_y
                y_tmp = self.y.copy()
                y_tmp[otherone.y == 0] = 0
                otherone.y[otherone.y == 0] = 1
                y = y_tmp/otherone.y*tmp_unit.asNumber()
                name = ""
                unit_y = tmp_unit/tmp_unit.asNumber()
        else:
            raise TypeError("You only can divide Profile with number")
        return Profile(self.x, y, self.unit_x, unit_y, name=name)

    __div__ = __truediv__

    def __sqrt__(self):
        y = np.sqrt(self.y)
        unit_y = np.sqrt(self.unit_y)
        return Profile(self.x, y, self.unit_x, unit_y, name=self.name)

    def __pow__(self, number):
        if not isinstance(number, NUMBERTYPES):
            raise TypeError("You only can use a number for the power "
                            "on a Profile")
        y = np.power(self.y, number)
        return Profile(self.x, y, self.unit_x, self.unit_y, name=self.name)

    def __len__(self):
        return len(self.x)

    def export_to_file(self, filepath, compressed=True, **kw):
        """
        Write the Profile object in the specified file usint the JSON format.
        Additionnals arguments for the JSON encoder may be set with the **kw
        argument. Such arguments may be 'indent' (for visual indentation in
        file, default=0) or 'encoding' (to change the file encoding,
        default='utf-8').
        If existing, specified file will be truncated. If not, it will
        be created.

        Parameters
        ----------
        filepath : string
            Path specifiing where to save the ScalarField.
        compressed : boolean, optional
            If 'True' (default), the json file is compressed using gzip.
        """
        import IMTreatment.io.io as imtio
        imtio.export_to_file(self, filepath, compressed, **kw)

    def import_from_file(self, filepath, **kw):
        """
        Load a Profile object from the specified file using the JSON
        format.
        Additionnals arguments for the JSON decoder may be set with the **kw
        argument. Such as'encoding' (to change the file
        encoding, default='utf-8').

        Parameters
        ----------
        filepath : string
            Path specifiing the Profile to load.
        """
        import IMTreatment.io.io as imtio
        tmp_p = imtio.import_from_file(filepath, **kw)
        if tmp_p.__classname__ != self.__classname__:
            raise IOError("This file do not contain a Profile, cabron")
        self.__init__(tmp_p.x, tmp_p.y, tmp_p.unit_x, tmp_p.unit_y, tmp_p.name)

#    def find_indices_in_axe(self, x=None, y=None):
#        """
#        Find the position of a given value along an axe, return indice(s) of
#        this position. Give only the first value find
#        (Ideal if the axe is crescent).
#
#        Parameters
#        ----------
#        xw : number, optionnal
#            Wanted value of X
#        yw : number, optionnal
#            Wanted valur of Y
#
#        Returns
#        -------
#        xi or yi : array
#            Contening the indice or the two indices limiting the wanted value
#        """
#        if x is not None:
#            if not isinstance(x, NUMBERTYPES):
#                raise TypeError("'x' has to be a number")
#            if all(x < self.x) or all(x > self.x):
#                raise ValueError("Specified 'x' is out of bound")
#        if y is not None:
#            if not isinstance(y, NUMBERTYPES):
#                raise TypeError("'y' has to be a number")
#            if all(y < self.y) or all(y > self.y):
#                raise ValueError("Specified 'y' is out of bound")
#        if x is None and y is None:
#            raise Warning("Ok, but i'll do nothing without a number")
#        if x is not None and y is not None:
#            raise ValueError("You cannot specify two values at the same time")
#        if x is not None:
#            value = x
#            axe = self.x
#        else:
#            value = y
#            axe = self.y
#        for i in np.arange(0, len(axe)-1):
#            if axe[i] == value:
#                return [i]
#            if ((axe[i] > value and axe[i+1] < value)
#                    or (axe[i] < value and axe[i+1] > value)):
#                return [i, i+1]
#

    def get_comp(self, comp):
        """
        Give access to the selected Profile component.
        """
        if not isinstance(comp, STRINGTYPES):
            raise TypeError("'comp' must be a string")
        if comp == "x":
            return self.x
        elif comp == 'y':
            return self.y
        elif comp == 'unit_y':
            return self.unit_y
        elif comp == 'unit_x':
            return self.unit_x
        else:
            raise ValueError("Unknown component : {}".format(comp))

    def get_interpolated_value(self, x=None, y=None):
        """
        Get the interpolated (or not) value for a given 'x' or 'y' value.
        It is obvious that you can't specify 'x' and 'y' at the same time.
        If several possibilities are possible, an array with all the results
        is returned.

        Parameters
        ----------
        x : number, optionnal
            Value of x, for which we want the y value.
        y : number, optionnal
            Value of y, for which we want the x value.

        Returns
        -------
        i_values : number or array
            Interpolated value(s).
        """
        if x is not None:
            if not isinstance(x, NUMBERTYPES):
                raise TypeError("'x' must be a number")
        if y is not None:
            if not isinstance(y, NUMBERTYPES):
                raise TypeError("'y' must be a number")
        if x is None and y is None:
            raise Warning("Ok, but i'll do nothing if i don't have a 'x' "
                          "or a 'y' value")
        if y is not None and x is not None:
            raise ValueError("Maybe you would like to look at the help "
                             "one more time...")
        if x is not None:
            value = x
            values = np.array(self.x)
            values2 = np.array(self.y)
        else:
            value = y
            values = np.array(self.y)
            values2 = np.array(self.x)
        i_values = []
        for ind in np.arange(0, len(values) - 1):
            val_i = values[ind]
            val_ipp = values[ind + 1]
            val2_i = values2[ind]
            val2_ipp = values2[ind + 1]
            if (val_i >= value and val_ipp < value) \
                    or (val_i <= value and val_ipp > value):
                i_value = ((val2_i*np.abs(val_ipp - value)
                           + val2_ipp*np.abs(values[ind] - value))
                           / np.abs(values[ind] - val_ipp))
                i_values.append(i_value)
        return i_values

    def copy(self):
        """
        Return a copy of the Profile object.
        """
        return Profile(self.x, self.y, self.unit_x, self.unit_y, self.name)

    def get_max(self, axe=2):
        """
        Return the maxima along an axe.

        Parameters
        ----------
        axe : integer, optionnal
            Axe along which we want the maxima.

        Returns
        -------
        max : number
            Maxima along 'axe'.
        """
        if not isinstance(axe, int):
            raise TypeError("'axe' must be an integer")
        if not (axe == 1 or axe == 2):
            raise ValueError("'axe' must be 1 or 2")
        if axe == 1:
            try:
                if np.all(self.x.mask):
                    return None
            except AttributeError:
                pass
            return np.max(self.x)
        if axe == 2:
            try:
                if np.all(self.y.mask):
                    return None
            except AttributeError:
                pass
            return np.max(self.y)

    def get_min(self, axe=2):
        """
        Return the minima along an axe.

        Parameters
        ----------
        axe : integer, optionnal
            Axe along which we want the minima.

        Returns
        -------
        max : number
            Minima along 'axe'.
        """
        if not isinstance(axe, int):
            raise TypeError("'axe' must be an integer")
        if not (axe == 1 or axe == 2):
            raise ValueError("'axe' must be 1 or 2")
        if axe == 1:
            return np.min(self.x)
        if axe == 2:
            return np.min(self.y)

    def get_integral(self):
        """
        Return the profile integral, and is unit.
        Use the trapezoidal aproximation.
        """
        return np.trapz(self.y, self.x), self.unit_y*self.unit_x

    def set_unit(self, comp, unity):
        """
        Write the selected component in the given unity (if possible).

        Parameters
        ----------
        comp : string
            Profile component to change.
        unity : Unum.unit object
            Unity (you can use make_unit to make one).
        """
        if not isinstance(comp, STRINGTYPES):
            raise TypeError("'comp' must be a string")
        if not isinstance(unity, unum.Unum):
            raise TypeError("'unity' must be a Unum object")
        if comp == "x":
            unit_tmp = self.unit_x.asUnit(unity)
            self.x = self.x*unit_tmp.asNumber()
            self.unit_x = unit_tmp/unit_tmp.asNumber()
        elif comp == 'y':
            unit_tmp = self.unit_y.asUnit(unity)
            self.y = self.y*unit_tmp.asNumber()
            self.unit_y = unit_tmp/unit_tmp.asNumber()
        else:
            raise ValueError("Unknown component : {}".format(comp))

    def trim(self, interval, ind=False):
        """
        Return a trimed copy of the profile along x with respect to 'interval'.

        Parameters
        ----------
        interval : array of two numbers
            Bound values of x.
        ind : Boolean, optionnal
            If 'False' (Default), 'interval' are values along x axis,
            if 'True', 'interval' are indices of values along x.
        """
        # checking parameters coherence
        if not isinstance(interval, ARRAYTYPES):
            raise TypeError("'interval' must be an array")
        interval = np.array(interval, dtype=float)
        if not interval.shape == (2,):
            raise ValueError("'interval' must be an array with only two"
                             "values")
        if interval[0] >= interval[1]:
            raise ValueError("'interval' values must be crescent")
        # given position is not an indice
        if not ind:
            if all(interval < np.min(self.x))\
                    or all(interval > np.max(self.x)):
                raise ValueError("'interval' values are out of profile")
            ind1 = 0
            ind2 = -1
            for i in np.arange(len(self.x)-1, 0, -1):
                if self.x[i] == interval[0]:
                    ind1 = i
                elif self.x[i] == interval[1]:
                    ind2 = i + 1
                elif (self.x[i] > interval[0] and self.x[i-1] < interval[0]) \
                        or (self.x[i] < interval[0]
                            and self.x[i-1] > interval[0]):
                    ind1 = i + 1
                elif (self.x[i] > interval[1] and self.x[i-1] < interval[1]) \
                        or (self.x[i] < interval[1]
                            and self.x[i-1] > interval[1]):
                    ind2 = i
            indices = [ind1, ind2]
            #indices.sort()
            x_new = self.x[indices[0]:indices[1]]
            y_new = self.y[indices[0]:indices[1]]
        # given position is an indice
        else:
            if any(interval < 0) or any(interval > len(self.x) - 1):
                raise ValueError("'interval' indices are out of profile")
            x_new = self.x[interval[0]:interval[1]]
            y_new = self.y[interval[0]:interval[1]]
        tmp_prof = Profile(x_new, y_new, self.unit_x, self.unit_y)
        return tmp_prof

    def smooth(self, meandist=2):
        """
        Return a smoothed profile using a mean on a given number of points

        Parameters
        ----------
        meandist : int, optionnal
            Number of points on which making a mean

        Returns
        -------
        SProfile : Profile object
            The smoothed profile
        """
        if not isinstance(meandist, int):
            raise TypeError("'meandist' must be an integer")
        if meandist > len(self.x):
            raise ValueError("'meandist' must be smaller than the length of "
                             "the profile")
        smoothx = np.array([], dtype=float)
        smoothy = np.array([], dtype=float)
        for i in np.arange(0, len(self.x)-meandist+1):
            locsmoothx = 0
            locsmoothy = 0
            for j in np.arange(0, meandist):
                locsmoothx += self.x[i+j]
                locsmoothy += self.y[i+j]
            smoothx = np.append(smoothx, locsmoothx/meandist)
            smoothy = np.append(smoothy, locsmoothy/meandist)
        return Profile(smoothx, smoothy, self.unit_x, self.unit_y,
                       self.name + " (Smoothed)")

    def _display(self, kind='plot', reverse=False, **plotargs):
        """
        Private Displayer.
        Just display the curve, not axes and title.

        Parameters
        ----------
        kind : string
            Kind of display to plot ('plot', 'semilogx', 'semilogy')
        reverse : Boolean, optionnal
            If 'False', x is put in the abscissa and y in the ordinate. If
            'True', the inverse.
        **plotargs : dict, optionnale
            Additional argument for the 'plot' command.

        Returns
        -------
        fig : Plot reference
            Reference to the displayed plot.
        """
        try:
            plotargs["label"]
        except KeyError:
            plotargs["label"] = self.name
        if not reverse:
            x = self.x
            y = self.y
        else:
            x = self.y
            y = self.x
        if kind == 'plot':
            plot = plt.plot(x, y, **plotargs)
        elif kind == 'semilogx':
            plot = plt.semilogx(x, y, **plotargs)
        elif kind == 'semilogy':
            plot = plt.semilogy(x, y, **plotargs)
        elif kind == 'loglog':
            plot = plt.loglog(x, y, **plotargs)
        else:
            raise ValueError("Unknown plot type : {}.".format(kind))
        return plot

    def display(self, kind='plot', reverse=False,  **plotargs):
        """
        Display the profile.

        Parameters
        ----------
        reverse : Boolean, optionnal
            If 'False', x is put in the abscissa and y in the ordinate. If
            'True', the inverse.
        kind : string
            Kind of display to plot ('plot', 'semilogx', 'semilogy')
        **plotargs : dict, optionnale
            Additional argument for the 'plot' command.

        Returns
        -------
        fig : Plot reference
            Reference to the displayed plot.
        """
        plot = self._display(kind, reverse, **plotargs)
        plt.title(self.name)
        if not reverse:
            plt.xlabel("{0}".format(self.unit_x.strUnit()))
            plt.ylabel("{0}".format(self.unit_y.strUnit()))
        else:
            plt.xlabel("{0}".format(self.unit_y.strUnit()))
            plt.ylabel("{0}".format(self.unit_x.strUnit()))
        return plot


class Field(object):

    def __init__(self):
        self.__axe_x = np.array([], dtype=float)
        self.__axe_y = np.array([], dtype=float)
        self.__unit_x = make_unit('')
        self.__unit_y = make_unit('')

    def __iter__(self):
        for i, x in enumerate(self.axe_x):
            for j, y in enumerate(self.axe_y):
                yield [i, j], [x, y]

    @property
    def axe_x(self):
        return self.__axe_x

    @axe_x.setter
    def axe_x(self, new_axe_x):
        if not isinstance(new_axe_x, ARRAYTYPES):
            raise TypeError()
        new_axe_x = np.array(new_axe_x, dtype=float)
        if new_axe_x.shape == self.__axe_x.shape or len(self.__axe_x) == 0:
            self.__axe_x = new_axe_x
        else:
            raise ValueError()

    @axe_x.deleter
    def axe_x(self):
        raise Exception("Nope, can't do that")

    @property
    def axe_y(self):
        return self.__axe_y

    @axe_y.setter
    def axe_y(self, new_axe_y):
        if not isinstance(new_axe_y, ARRAYTYPES):
            raise TypeError()
        new_axe_y = np.array(new_axe_y, dtype=float)
        if new_axe_y.shape == self.__axe_y.shape or len(self.__axe_y) == 0:
            self.__axe_y = new_axe_y
        else:
            raise ValueError()

    @axe_y.deleter
    def axe_y(self):
        raise Exception("Nope, can't do that")

    @property
    def unit_x(self):
        return self.__unit_x

    @unit_x.setter
    def unit_x(self, new_unit_x):
        if isinstance(new_unit_x, unum.Unum):
            if new_unit_x.asNumber() == 1:
                self.__unit_x = new_unit_x
            else:
                raise ValueError()
        elif isinstance(new_unit_x, STRINGTYPES):
            self.__unit_x == make_unit(new_unit_x)
        else:
            raise TypeError()

    @unit_x.deleter
    def unit_x(self):
        raise Exception("Nope, can't do that")

    @property
    def unit_y(self):
        return self.__unit_y

    @unit_y.setter
    def unit_y(self, new_unit_y):
        if isinstance(new_unit_y, unum.Unum):
            if new_unit_y.asNumber() == 1:
                self.__unit_y = new_unit_y
            else:
                raise ValueError()
        elif isinstance(new_unit_y, STRINGTYPES):
            self.__unit_y == make_unit(new_unit_y)
        else:
            raise TypeError()

    @unit_y.deleter
    def unit_y(self):
        raise Exception("Nope, can't do that")

    @property
    def shape(self):
        return self.__axe_x.shape[0], self.__axe_y.shape[0]

#    def get_axes(self):
#        """
#        Return the field axes.
#
#        Returns
#        -------
#        axe_x : array
#            Axe along X.
#        axe_y : array
#            Axe along Y.
#        """
#        return self.axe_x.copy(), self.axe_y.copy()

#    def get_axe_units(self):
#        """
#        Return the axis unities.
#
#        Returns:
#        --------
#        unit_x : unit object
#            Axe x unit
#        unit_y : unit object
#            Axe y unit
#        """
#        return self.unit_x.copy(), self.unit_y.copy()

    def copy(self):
        """
        Return a copy of the Field object.
        """
        return copy.deepcopy(self)

#    def set_axes(self, axe_x=None, axe_y=None):
#        """
#        Load new axes in the field.
#
#        Parameters
#        ----------
#        axe_x : array
#            One-dimensionale array representing the position of the scalar
#            values along the X axe.
#        axe_y : array
#            idem for the Y axe.
#        """
#        if axe_x is not None:
#            if not isinstance(axe_x, ARRAYTYPES):
#                raise TypeError("'axe_x' must be an array")
#            axe_x = np.array(axe_x, dtype=float)
#            if self.axe_x.shape == axe_x.shape:
#                self.axe_x = axe_x
#            else:
#                raise ValueError("Inconsistent size of 'axe_x'")
#        if axe_y is not None:
#            if not isinstance(axe_y, ARRAYTYPES):
#                raise TypeError("'axe_y' must be an array")
#            axe_y = np.array(axe_y, dtype=float)
#            if axe_y.shape == self.axe_y.shape:
#                self.axe_y = axe_y
#            else:
#                raise ValueError("Inconsistent size of 'axe_y'")

#    def set_axe_units(self, unit_x=None, unit_y=None):
#        """
#        Load unities into the field axis.
#
#        Parameters
#        ----------
#        unit_x : Unit object
#            Axis X unit.
#        unit_y : Unit object
#            Axis Y unit.
#        """
#        if unit_x is not None:
#            if not isinstance(unit_x, unum.Unum):
#                raise TypeError("'unit_x' must be an unit object")
#            self.unit_x = unit_x
#        if unit_y is not None:
#            if not isinstance(unit_y, unum.Unum):
#                raise TypeError("'unit_y' must be an unit object")
#            self.unit_y = unit_y

    def set_origin(self, x=None, y=None):
        """
        Modify the axis in order to place the origin at the givev point (x, y)

        Parameters
        ----------
        x : number
        y : number
        """
        if x is not None:
            if not isinstance(x, NUMBERTYPES):
                raise TypeError("'x' must be a number")
            self.axe_x -= x
        if y is not None:
            if not isinstance(y, NUMBERTYPES):
                raise TypeError("'y' must be a number")
            self.axe_y -= y

    def trim_area(self, intervalx=None, intervaly=None, full_output=False):
        """
        Return a trimed field in respect with given intervals.

        Parameters
        ----------
        intervalx : array, optional
            interval wanted along x
        intervaly : array, optional
            interval wanted along y
        full_output : boolean, optional
            If 'True', cutting indices are alson returned
        """
        axe_x, axe_y = self.axe_x, self.axe_y
        if intervalx is None:
            intervalx = [axe_x[0], axe_x[-1]]
        if intervaly is None:
            intervaly = [axe_y[0], axe_y[-1]]
        if not isinstance(intervalx, ARRAYTYPES):
            raise TypeError("'intervalx' must be an array of two numbers")
        intervalx = np.array(intervalx, dtype=float)
        if intervalx.ndim != 1:
            raise ValueError("'intervalx' must be an array of two numbers")
        if intervalx.shape != (2,):
            raise ValueError("'intervalx' must be an array of two numbers")
        if intervalx[0] > intervalx[1]:
            raise ValueError("'intervalx' values must be crescent")
        if not isinstance(intervaly, ARRAYTYPES):
            raise TypeError("'intervaly' must be an array of two numbers")
        intervaly = np.array(intervaly, dtype=float)
        if intervaly.ndim != 1:
            raise ValueError("'intervaly' must be an array of two numbers")
        if intervaly.shape != (2,):
            raise ValueError("'intervaly' must be an array of two numbers")
        if intervaly[0] > intervaly[1]:
            raise ValueError("'intervaly' values must be crescent")
        # finding interval indices
        if intervalx[0] <= axe_x[0]:
            indmin_x = 0
        else:
            indmin_x = self.get_indice_on_axe(1, intervalx[0])[-1]
        if intervalx[1] >= axe_x[-1]:
            indmax_x = len(axe_x) - 1
        else:
            indmax_x = self.get_indice_on_axe(1, intervalx[1])[0]
        if intervaly[0] <= axe_y[0]:
            indmin_y = 0
        else:
            indmin_y = self.get_indice_on_axe(2, intervaly[0])[-1]
        if intervaly[1] >= axe_y[-1]:
            indmax_y = len(axe_y) - 1
        else:
            indmax_y = self.get_indice_on_axe(2, intervaly[1])[0]
        trimfield = self.__class__()
        trimfield.axe_x = self.axe_x[indmin_x:indmax_x + 1]
        trimfield.axe_y = self.axe_y[indmin_y:indmax_y + 1]
        if full_output:
            return indmin_x, indmax_x, indmin_y, indmax_y, trimfield
        else:
            return trimfield

    def get_indice_on_axe(self, direction, value, nearest=False):
        """
        Return, on the given axe, the indices representing the positions
        surrounding 'value'.
        if 'value' is exactly an axe position, return just one indice.

        Parameters
        ----------
        direction : int
            1 or 2, for axes choice.
        value : number
        nearest : boolean
            If 'True', only the nearest indice is returned.

        Returns
        -------
        interval : 2x1 or 1x1 array of integer
        """
        if not isinstance(direction, NUMBERTYPES):
            raise TypeError("'direction' must be a number.")
        if not (direction == 1 or direction == 2):
            raise ValueError("'direction' must be 1 or 2.")
        if not isinstance(value, NUMBERTYPES):
            raise TypeError("'value' must be a number.")
        if direction == 1:
            axe = self.axe_x
            if value < axe[0] or value > axe[-1]:
                raise ValueError("'value' is out of bound.")
        else:
            axe = self.axe_y
            if value < axe[0] or value > axe[-1]:
                raise ValueError("'value' is out of bound.")
        # getting the borning indices
        ind = np.searchsorted(axe, value)
        if axe[ind] == value:
            inds = [ind]
        else:
            inds = [int(ind - 1), int(ind)]
        if not nearest:
            return inds
        # getting the nearest indice
        else:
            if len(inds) != 1:
                if np.abs(axe[inds[0]] - value) < np.abs(axe[inds[1]] - value):
                    ind = inds[0]
                else:
                    ind = inds[1]
            else:
                ind = inds[0]
            return int(ind)

    def get_points_around(self, center, radius, ind=False):
        """
        Return the list of points or the scalar field that are in a circle
        centered on 'center' and of radius 'radius'.

        Parameters
        ----------
        center : array
            Coordonate of the center point (in axes units).
        radius : float
            radius of the cercle (in axes units).
        ind : boolean, optional
            If 'True', radius and center represent indices on the field.
            if 'False', radius and center are expressed in axis unities.

        Returns
        -------
        indices : array
            Array contening the indices of the contened points.
            [(ind1x, ind1y), (ind2x, ind2y), ...].
            You can easily put them in the axes to obtain points coordinates
        """
        #checkin parameters coherence
        if not isinstance(center, ARRAYTYPES):
            raise TypeError("'center' must be an array")
        center = np.array(center, dtype=float)
        if not center.shape == (2,):
            raise ValueError("'center' must be a 2x1 array")
        if not isinstance(radius, NUMBERTYPES):
            raise TypeError("'radius' must be a number")
        if not radius > 0:
            raise ValueError("'radius' must be positive")
        # getting somme properties
        radius2 = radius**2
        radius_int = radius/np.sqrt(2)
        inds = []
        for indices, coord, _ in self:
            if ind:
                x = indices[0]
                y = indices[1]
            else:
                x = coord[0]
                y = coord[1]
            # test if the point is not in the square surrounding the cercle
            if x >= center[0] + radius \
                    and x <= center[0] - radius \
                    and y >= center[1] + radius \
                    and y <= center[1] - radius:
                pass
            # test if the point is in the square 'compris' in the cercle
            elif x <= center[0] + radius_int \
                    and x >= center[0] - radius_int \
                    and y <= center[1] + radius_int \
                    and y >= center[1] - radius_int:
                inds.append(indices)
            # test if the point is the center
            elif all([x, y] == center):
                pass
            # test if the point is in the circle
            elif ((x - center[0])**2 + (y - center[1])**2 <= radius2):
                inds.append(indices)
        return np.array(inds, subok=True)


class ScalarField(Field):
    """
    Class representing a scalar field (2D field, with one component on each
    point).

    Principal methods
    -----------------
    "import_from_*" : allows to easily create or import scalar fields.

    "export_to_*" : allows to export.

    "display" : display the scalar field, with these unities.

    Examples
    --------
    >>> import IMTreatment as imt
    >>> SF = imt.ScalarField()
    >>> unit_axe = imt.make_unit('cm')
    >>> unit_K = imt.make_unit('K')
    >>> SF.import_from_arrays([1,2], [1,2], [[4,8], [4,8]], unit_axe, unit_axe,
    ...                       unit_K)
    >>> SF.display()
    """

    def __init__(self):
        Field.__init__(self)
        self.__values = np.array([])
        self.__mask = np.array([], dtype=bool)
        self.__unit_values = make_unit("")

    def __eq__(self, another):
        if not isinstance(another, ScalarField):
            return False
        if not np.all(self.axe_x == another.axe_x):
            return False
        if not np.all(self.axe_y == another.axe_y):
            return False
        if not np.all(self.values == another.values):
            return False
        if not np.all(self.mask == another.mask):
            return False
        if not np.all(self.unit_x == another.unit_x):
            return False
        if not np.all(self.unit_y == another.unit_y):
            return False
        if not np.all(self.unit_values == another.unit_values):
            return False
        return True

    def __neg__(self):
        tmpsf = self.copy()
        tmpsf.values = -tmpsf.values
        return tmpsf

    def __add__(self, otherone):
        # if we add with a ScalarField object
        if isinstance(otherone, ScalarField):
            if all(self.axe_x != otherone.axe_x) or \
                    all(self.axe_y != otherone.axe_y):
                raise ValueError("Scalar fields have to be consistent "
                                 "(same dimensions)")
            try:
                self.unit_values + otherone.unit_values
                self.unit_x + otherone.unit_x
                self.unit_y + otherone.unit_y
            except:
                raise ValueError("I think these units don't match, fox")
            tmpsf = self.copy()
            fact = otherone.unit_values/self.unit_values
            tmpsf.values += otherone.values*fact.asNumber()
            tmpsf.mask = np.logical_or(self.mask, otherone.mask)
            return tmpsf
        # if we add with a number
        elif isinstance(otherone, NUMBERTYPES):
            tmpsf = self.copy()
            tmpsf.values += otherone
            return tmpsf
        elif isinstance(otherone, unum.Unum):
            try:
                self.unit_values + otherone
            except:
                pdb.set_trace()
                raise ValueError("Given number have to be consistent with"
                                 "the scalar field (same units)")
            tmpsf = self.copy()
            tmpsf.values += (otherone/self.unit_values).asNumber()
            return tmpsf
        else:
            raise TypeError("You can only add a scalarfield "
                            "with others scalarfields or with numbers")

    def __radd__(self, obj):
        return self.__add__(obj)

    def __sub__(self, obj):
        return self.__add__(-obj)

    def __rsub__(self, obj):
        return self.__neg__() + obj

    def __truediv__(self, obj):
        if isinstance(obj, NUMBERTYPES):
            tmpsf = self.copy()
            tmpsf.values /= obj
            return tmpsf
        elif isinstance(obj, unum.Unum):
            tmpsf = self.copy()
            tmpsf.values /= obj.asNumber()
            tmpsf.unit_values /= obj/obj.asNumber()
            return tmpsf
        elif isinstance(obj, ScalarField):
            if np.any(self.axe_x != obj.axe_x)\
                    or np.any(self.axe_y != obj.axe_y)\
                    or self.unit_x != obj.unit_x\
                    or self.unit_y != obj.unit_y:
                raise ValueError("Fields are not consistent")
            tmpsf = self.copy()
            values = self.values / obj.values
            mask = np.logical_or(self.mask, obj.mask)
            unit = self.unit_values / obj.unit_values
            tmpsf.values = values*unit.asNumber()
            tmpsf.mask = mask
            tmpsf.unit_values = unit/unit.asNumber()
            return tmpsf
        else:
            raise TypeError("Unsupported operation between {} and a "
                            "ScalarField object".format(type(obj)))

    __div__ = __truediv__

    def __rdiv__(self, obj):
        if isinstance(obj, NUMBERTYPES):
            tmpsf = self.copy()
            tmpsf.values = obj/tmpsf.values
            tmpsf.unit_values = 1/tmpsf.unit_values
            return tmpsf
        elif isinstance(obj, unum.Unum):
            tmpsf = self.copy()
            tmpsf.values = obj.asNumber()/tmpsf.values
            tmpsf.unit_values = obj/obj.asNumber()/tmpsf.unit_values
            return tmpsf
        elif isinstance(obj, ScalarField):
            if np.any(self.axe_x != obj.axe_x)\
                    or np.any(self.axe_y != obj.axe_y)\
                    or self.unit_x != obj.unit_x\
                    or self.unit_y != obj.unit_y:
                raise ValueError("Fields are not consistent")
            tmpsf = self.copy()
            values = obj.values / self.values
            mask = np.logical_or(self.mask, obj.mask)
            unit = obj.unit_values / self.unit_values
            tmpsf.values = values*unit.asNumber()
            tmpsf.mask = mask
            tmpsf.unit_values = unit/unit.asNumber()
            return tmpsf
        else:
            raise TypeError("Unsupported operation between {} and a "
                            "ScalarField object".format(type(obj)))

    def __mul__(self, obj):
        if isinstance(obj, NUMBERTYPES):
            tmpsf = self.copy()
            tmpsf.values *= obj
            return tmpsf
        elif isinstance(obj, unum.Unum):
            tmpsf = self.copy()
            tmpsf.values *= obj.asNumber()
            tmpsf.unit_values *= obj/obj.asNumber()
            return tmpsf
        elif isinstance(obj, np.ma.core.MaskedArray):
            if obj.shape != self.values.shape:
                raise ValueError("Fields are not consistent")
            tmpsf = self.copy()
            tmpsf.values *= obj
            return tmpsf
        elif isinstance(obj, ScalarField):
            if np.any(self.axe_x != obj.axe_x)\
                    or np.any(self.axe_y != obj.axe_y)\
                    or self.unit_x != obj.unit_x\
                    or self.unit_y != obj.unit_y:
                raise ValueError("Fields are not consistent")
            tmpsf = self.copy()
            values = self.values * obj.values
            mask = np.logical_or(self.mask, obj.mask)
            unit = self.unit_values * obj.unit_values
            tmpsf.values = values*unit.asNumber()
            tmpsf.mask = mask
            tmpsf.unit_values = unit/unit.asNumber()
            return tmpsf
        else:
            raise TypeError("Unsupported operation between {} and a "
                            "ScalarField object".format(type(obj)))
    __rmul__ = __mul__

    def __abs__(self):
        tmpsf = self.copy()
        tmpsf.values = np.abs(tmpsf.values)
        return tmpsf

    def __pow__(self, number):
        if not isinstance(number, NUMBERTYPES):
            raise TypeError("You only can use a number for the power "
                            "on a Scalar field")
        tmpsf = self.copy()
        tmpsf.values = np.power(tmpsf.values, number)
        tmpsf.unit_values = np.power(tmpsf.unit_values, number)
        return tmpsf

    def __iter__(self):
        data = self.values
        mask = self.mask
        for ij, xy in Field.__iter__(self):
            i = ij[0]
            j = ij[1]
            if not mask[i, j]:
                yield ij, xy, data[i, j]

    @property
    def values(self):
        return self.__values

    @values.setter
    def values(self, new_values):
        if not isinstance(new_values, ARRAYTYPES):
            raise TypeError()
        new_values = np.array(new_values)
        if self.shape == new_values.shape:
            # adapting mask to 'nan' values
            add_mask = np.isnan(new_values)
            if self.mask.shape == (0,):
                self.mask = add_mask
            else:
                self.mask = np.logical_or(self.mask, add_mask)
            # storing data
            self.__values = new_values
        else:
            raise ValueError("'values' should have the same shape as the "
                             "original values : {}, not {}."
                             .format(self.shape, new_values.shape))

    @values.deleter
    def values(self):
        raise Exception("Nope, can't do that")

    @property
    def mask(self):
        return self.__mask

    @mask.setter
    def mask(self, new_mask):
        # check 'new_mask' coherence
        if isinstance(new_mask, bool):
            fill_value = new_mask
            new_mask = np.empty(self.shape, dtype=bool)
            new_mask.fill(fill_value)
        elif isinstance(new_mask, ARRAYTYPES):
            new_mask = np.array(new_mask)
        else:
            raise TypeError("'mask' should be an array or a boolean,"
                            " not a {}".format(type(new_mask)))
        if self.shape != new_mask.shape:
            raise ValueError()
        # check if the new mask don'r reveal masked values
        if np.any(np.logical_not(new_mask[self.mask])):
            pdb.set_trace()
            raise Warning("This mask reveal masked values, maybe you should"
                          "use the 'fill' function instead")
        # store mask
        self.__mask = new_mask

    @mask.deleter
    def mask(self):
        raise Exception("Nope, can't do that")

    @property
    def unit_values(self):
        return self.__unit_values

    @unit_values.setter
    def unit_values(self, new_unit_values):
        if isinstance(new_unit_values, unum.Unum):
            if new_unit_values.asNumber() == 1:
                self.__unit_values = new_unit_values
            else:
                raise ValueError()
        elif isinstance(new_unit_values, STRINGTYPES):
            self.__unit_values == make_unit(new_unit_values)
        else:
            raise TypeError()

    @unit_values.deleter
    def unit_values(self):
        raise Exception("Nope, can't do that")

    @property
    def min(self):
        return np.min(self.values[np.logical_not(self.mask)])

    @property
    def max(self):
        return np.max(self.values[np.logical_not(self.mask)])

    @property
    def mean(self):
        return np.mean(self.values[np.logical_not(self.mask)])

    def import_from_arrays(self, axe_x, axe_y, values, mask=False,
                           unit_x="", unit_y="", unit_values=""):
        """
        Set the field from a set of arrays.

        Parameters
        ----------
        axe_x : array
            Discretized axis value along x
        axe_y : array
            Discretized axis value along y
        values : array or masked array
            Values of the field at the discritized points
        unit_x : String unit, optionnal
            Unit for the values of axe_x
        unit_y : String unit, optionnal
            Unit for the values of axe_y
        unit_values : String unit, optionnal
            Unit for the scalar field
        """
        # checking parameters coherence
        if (values.shape[0] != axe_x.shape[0] or
                values.shape[1] != axe_y.shape[0]):
            raise ValueError("Dimensions of 'axe_x', 'axe_y' and 'values' must"
                             " be consistents")
        # storing datas
        self.axe_x = axe_x
        self.axe_y = axe_y
        self.values = values
        self.mask = mask
        self.unit_x = unit_x
        self.unit_y = unit_y
        self.unit_values = unit_values
        self.crop_masked_border()

#    def import_from_file(self, filepath, **kw):
#        """
#        Load a ScalarField object from the specified file using the JSON
#        format.
#        Additionnals arguments for the JSON decoder may be set with the **kw
#        argument. Such as'encoding' (to change the file
#        encoding, default='utf-8').
#
#        Parameters
#        ----------
#        filepath : string
#            Path specifiing the ScalarField to load.
#        """
#        import IMTreatment.io.io as imtio
#        tmpsf = imtio.import_from_file(filepath, **kw)
#        if tmpsf.__classname__ != self.__classname__:
#            raise IOError("This file do not contain a ScalarField, cabron.")
#        self = tmpsf

#    def export_to_file(self, filepath, compressed=True, **kw):
#        """
#        Write the Profile object in the specified file usint the JSON format.
#        Additionnals arguments for the JSON encoder may be set with the **kw
#        argument. Such arguments may be 'indent' (for visual indentation in
#        file, default=0) or 'encoding' (to change the file encoding,
#        default='utf-8').
#        If existing, specified file will be truncated. If not, it will
#        be created.
#
#        Parameters
#        ----------
#        filepath : string
#            Path specifiing where to save the ScalarField.
#        compressed : boolean, optional
#            If 'True' (default), the json file is compressed using gzip.
#        """
#        import IMTreatment.io.io as imtio
#        imtio.export_to_file(self, filepath, compressed, **kw)

#    def export_to_vtk(self, filepath, axis=None):
#        """
#        Export the scalar field to a .vtk file, for Mayavi use.
#
#        Parameters
#        ----------
#        filepath : string
#            Path where to write the vtk file.
#        axis : tuple of strings
#            By default, scalar field axe are set to (x,y), if you want
#            different axis, you have to specified them here.
#            For example, "('z', 'y')", put the x scalar field axis values
#            in vtk z axis, and y scalar field axis in y vtk axis.
#        """
#        import pyvtk
#        if not os.path.exists(os.path.dirname(filepath)):
#            raise ValueError("'filepath' is not a valid path")
#        if axis is None:
#            axis = ('x', 'y')
#        if not isinstance(axis, ARRAYTYPES):
#            raise TypeError("'axis' must be a 2x1 tuple")
#        if not isinstance(axis[0], STRINGTYPES) \
#                or not isinstance(axis[1], STRINGTYPES):
#            raise TypeError("'axis' must be a 2x1 tuple of strings")
#        if not axis[0] in ['x', 'y', 'z'] or not axis[1] in ['x', 'y', 'z']:
#            raise ValueError("'axis' strings must be 'x', 'y' or 'z'")
#        if axis[0] == axis[1]:
#            raise ValueError("'axis' strings must be different")
#        V = self.values.flatten()
#        x = self.axe_x
#        y = self.axe_y
#        point_data = pyvtk.PointData(pyvtk.Scalars(V, 'Scalar Field'))
#        x_vtk = 0.
#        y_vtk = 0.
#        z_vtk = 0.
#        if axis[0] == 'x':
#            x_vtk = x
#        elif axis[0] == 'y':
#            y_vtk = x
#        else:
#            z_vtk = x
#        if axis[1] == 'x':
#            x_vtk = y
#        elif axis[1] == 'y':
#            y_vtk = y
#        else:
#            z_vtk = y
#        grid = pyvtk.RectilinearGrid(x_vtk, y_vtk, z_vtk)
#        data = pyvtk.VtkData(grid, 'Scalar Field from python', point_data)
#        data.tofile(filepath)

    def export_to_scatter(self, mask=None):
        """
        Return the scalar field under the form of a Points object.

        Parameters
        ----------
        mask : array of boolean, optional
            Mask to choose values to extract
            (values are taken where mask is False).

        Returns
        -------
        Pts : Points object
            Contening the ScalarField points.
        """
        if mask is None:
            mask = np.zeros(self.shape)
        if not isinstance(mask, ARRAYTYPES):
            raise TypeError("'mask' must be an array of boolean")
        mask = np.array(mask)
        if mask.shape != self.shape:
            raise ValueError("'mask' must have the same dimensions as"
                             "the ScalarField :{}".format(self.shape))
        # récupération du masque
        mask = np.logical_or(mask, self.mask)
        pts = None
        v = np.array([], dtype=float)
        # boucle sur les points
        for inds, pos, value in self:
            if mask[inds[0], inds[1]]:
                continue
            if pts is None:
                pts = [pos]
            else:
                pts = np.append(pts, [pos], axis=0)
            v = np.append(v, value)
        return Points(pts, v, self.unit_x, self.unit_y, self.unit_values)

#    def get_comp(self, componentname, raw=False, masked=True):
#        """
#        Return a ScalarField object representing a component of the
#        Vectorfield object.
#
#        Parameters
#        ----------
#        componentname : string
#            Can be 'values' or 'mask'.
#        raw : boolean, optional
#            If 'False' (default), return a ScalarField object,
#            if 'True', return a masked array.
#        masked : boolean, optional
#            If 'True' (default), returned np.array can be masked array,
#            If 'False', returned array are always brut np.array
#
#        Returns
#        -------
#        component : ScalarField object or numpy masked array
#        """
#        if not isinstance(componentname, STRINGTYPES):
#            raise TypeError("'componentname' must be a string")
#        if componentname == 'values':
#            if masked:
#                values = self.values
#            else:
#                values = self.values.data
#        elif componentname == 'mask':
#            values = self.values.mask
#            if isinstance(values, np.bool_):
#                if values:
#                    values = np.ones(self.get_dim(), dtype=bool)
#                else:
#                    values = np.zeros(self.get_dim(), dtype=bool)
#        elif componentname == "unit_values":
#            values = self.unit_values
#        else:
#            raise ValueError("unknown value of 'componentname'")
#        if raw:
#            return values.copy()
#        if isinstance(values, unum.Unum):
#            return values.copy()
#        else:
#            axe_x, axe_y = self.get_axes()
#            unit_x, unit_y = self.get_axe_units()
#            unit_values = self.get_comp('unit_values')
#            tmpsf = ScalarField()
#            tmpsf.import_from_arrays(axe_x, axe_y, values, unit_x, unit_y,
#                                     unit_values)
#            return tmpsf

#    def get_values(self):
#        """
#        Return the field values.
#        """
#        return self.values
#
#    def get_mask(self):
#        """
#        Return the scalarfield mask.
#        """
#        return self.values.mask

#    def set_values(self, values):
#        """
#        fill the scalar field with the given values
#        """
#        if not isinstance(values, ARRAYTYPES):
#            raise TypeError("'values' must be an array")
#        values = np.array(values)
#        if not np.all(values.shape == self.get_dim()):
#            raise ValueError("'values' shape must agree with the "
#                             "original field shape")
#        self.values = values
#
#    def set_mask(self, mask):
#        """
#        fill the scalar field mask with the given values.
#        """
#        if not isinstance(mask, ARRAYTYPES):
#            raise TypeError("'values' must be an array")
#        mask = np.array(mask)
#        if not np.all(mask.shape == self.get_dim()):
#            raise ValueError("'values' shape must agree with the "
#                             "original field shape")
#        self.values.mask = mask

#    def set_comp(self, componentname, value):
#        """
#        Fill the component 'componentname' with 'value'.
#
#        Parameters
#        ----------
#        componentname : string
#            Can be 'values' or 'mask'.
#        value : array
#            Array with the same shape as the initial component
#        """
#        if not isinstance(componentname, STRINGTYPES):
#            raise TypeError("'componentname' must be a string")
#        if componentname == 'values':
#            if not isinstance(value, ARRAYTYPES):
#                raise TypeError("'value' must be an array")
#            if not isinstance(value, np.ma.MaskedArray):
#                value = np.ma.masked_array(value)
#            if self.get_dim() != value.shape:
#                raise ValueError("'value' dimensions are inconsistent with "
#                                 "the ScalarField shape")
#            self.values = value
#        elif componentname == 'mask':
#            if not isinstance(value, ARRAYTYPES):
#                raise TypeError("'value' must be an array")
#            if not isinstance(value, np.ma.MaskedArray):
#                value = np.ma.masked_array(value)
#            if self.get_dim() != value.shape:
#                raise ValueError("'value' dimensions are inconsistent with"
#                                 "the ScalarField shape")
#            self.values.mask = value
#        elif componentname == "unit_values":
#            if not isinstance(value, unum.Unum):
#                raise TypeError("'value' must here be a unit obejct")
#            self.unit_values = value
#        else:
#            raise ValueError("Unknown 'componentname' value")

#    def get_dim(self):
#        """
#        Return the scalar field dimension.
#
#        Returns
#        -------
#        shape : tuple
#            Tuple of the dimensions (along X and Y) of the scalar field.
#        """
#        return self.values.shape

#    def get_min(self, unit=False):
#        """
#        Return the minima of the field.
#
#        Parameters
#        ----------
#        unit : boolean, optional
#            If True, a unit object is returned,
#            else (default), a float is returned.
#
#        Returns
#        -------
#        mini : float or unit object
#            Minima on the field
#        """
#        if unit:
#            return np.min(self.values)*self.unit_values
#        else:
#            return np.min(self.values)

#    def get_max(self, unit=False):
#        """
#        Return the maxima of the field.
#
#        Parameters
#        ----------
#        unit : boolean, optional
#            If True, a unit object is returned,
#            else (default), a float is returned.
#
#        Returns
#        -------
#        maxi : float or unit object
#            Maxima on the field
#        """
#        if unit:
#            return np.max(self.values)*self.unit_values
#        else:
#            return np.max(self.values)

#    def get_mean(self, unit=False):
#        """
#        Return the mean value of the field.
#
#        Parameters
#        ----------
#        unit : boolean, optional
#            If True, a unit object is returned,
#            else (default), a float is returned.
#
#        Returns
#        -------
#        mean : float or unit object
#            Mean value of the field.
#        """
#        if unit:
#            return np.mean(self.values)*self.unit_values
#        else:
#            return np.mean(self.values)

    def get_value(self, x, y, ind=False, unit=False):
        """
        Return the scalar field value on the point (x, y).
        If ind is true, x and y are indices,
        else, x and y are value on axes (interpolated if necessary).
        """
        if not isinstance(ind, bool):
            raise TypeError("'ind' must be a boolean")
        if ind:
            if not isinstance(x, int) or not isinstance(y, int):
                raise TypeError("'x' and 'y' must be integers")
            if x > len(self.axe_x) - 1 or y > len(self.axe_y) - 1\
                    or x < 0 or y < 0:
                raise ValueError("'x' and 'y' must be correct indices")
        else:
            if not isinstance(x, NUMBERTYPES)\
                    or not isinstance(y, NUMBERTYPES):
                raise TypeError("'x' and 'y' must be numbers")
            if x > np.max(self.axe_x) or y > np.max(self.axe_y)\
                    or x < np.min(self.axe_x) or y < np.min(self.axe_y):
                raise ValueError("'x' and 'y' are out of axes")
        if unit:
            unit = self.unit_values
        else:
            unit = 1.
        if ind:
            return self.values[y, x]*unit
        else:
            ind_x = None
            ind_y = None
            # getting indices interval
            inds_x = self.get_indice_on_axe(1, x)
            inds_y = self.get_indice_on_axe(2, y)
            # if we are on a grid point
            if len(inds_x) == 1 and len(inds_y) == 1:
                return self.values[inds_y[0], inds_x[0]]*unit
            # if we are on a x grid branch
            elif len(inds_x) == 1:
                ind_x = inds_x[0]
                pos_y1 = self.axe_y[inds_y[0]]
                pos_y2 = self.axe_y[inds_y[1]]
                value1 = self.values[inds_y[0], ind_x]
                value2 = self.values[inds_y[1], ind_x]
                i_value = ((value2*np.abs(pos_y1 - y)
                           + value1*np.abs(pos_y2 - y))
                           / np.abs(pos_y1 - pos_y2))
                return i_value*unit
            # if we are on a x grid branch
            elif len(inds_y) == 1:
                ind_y = inds_y[0]
                pos_x1 = self.axe_x[inds_x[0]]
                pos_x2 = self.axe_x[inds_x[1]]
                value1 = self.values[ind_y, inds_x[0]]
                value2 = self.values[ind_y, inds_x[1]]
                i_value = ((value2*np.abs(pos_x1 - x)
                            + value1*np.abs(pos_x2 - x))
                           / np.abs(pos_x1 - pos_x2))
                return i_value*unit
            # if we are in the middle of nowhere (linear interpolation)
            ind_x = inds_x[0]
            ind_y = inds_y[0]
            a, b = np.meshgrid(self.axe_x[ind_x:ind_x + 2],
                               self.axe_y[ind_y:ind_y + 2])
            values = self.values[ind_y:ind_y + 2, ind_x:ind_x + 2]
            a = a.flatten()
            b = b.flatten()
            pts = zip(a, b)
            interp_vx = spinterp.LinearNDInterpolator(pts, values.flatten())
            i_value = float(interp_vx(x, y))
            return i_value*unit

    def get_zones_centers(self, bornes=[0.75, 1], rel=True,
                          kind='ponderated'):
        """
        Return a Points object contening centers of the zones
        lying in the given bornes.

        Parameters
        ----------
        bornes : 2x1 array, optionnal
            Trigger values determining the zones.
            '[inferior borne, superior borne]'
        rel : Boolean
            If 'rel' is 'True' (default), values of 'bornes' are relative to
            the extremum values of the field.
            If 'rel' is 'False', values of bornes are treated like absolute
            values.
        kind : string, optional
            if 'kind' is 'center', given points are geometrical centers,
            if 'kind' is 'extremum', given points are
            extrema (min or max) on zones
            if 'kind' is 'ponderated'(default, given points are centers of
            mass, ponderated by the scaler field.

        Returns
        -------
        pts : Points object
            Contening the centers coordinates
        """
        # correcting python's problem with egality...
        bornes[0] -= 0.00001*abs(bornes[0])
        bornes[1] += 0.00001*abs(bornes[1])
        # checking parameters coherence
        if not isinstance(bornes, ARRAYTYPES):
            raise TypeError("'bornes' must be an array")
        if not isinstance(bornes, np.ndarray):
            bornes = np.array(bornes, dtype=float)
        if not bornes.shape == (2,):
            raise ValueError("'bornes' must be a 2x1 array")
        if not bornes[0] < bornes[1]:
            raise ValueError("'bornes' must be crescent")
        if not isinstance(rel, bool):
            raise TypeError("'rel' must be a boolean")
        if not isinstance(kind, STRINGTYPES):
            raise TypeError("'kind' must be a string")
        # compute minimum and maximum if 'rel=True'
        if rel:
            if bornes[0]*bornes[1] < 0:
                raise ValueError("In relative 'bornes' must have the same"
                                 " sign")
            mini = self.min
            maxi = self.max
            if np.abs(bornes[0]) > np.abs(bornes[1]):
                bornes[1] = abs(maxi - mini)*bornes[1] + maxi
                bornes[0] = abs(maxi - mini)*bornes[0] + maxi
            else:
                bornes[1] = abs(maxi - mini)*bornes[1] + mini
                bornes[0] = abs(maxi - mini)*bornes[0] + mini
        # check if the zone exist
        else:
            mini = self.min
            maxi = self.max
            if maxi < bornes[0] or mini > bornes[1]:
                return None
        # getting data
        values = self.values
        mask = self.mask
        if np.any(mask):
            raise UserWarning("There is masked values, algorithm can give "
                              "strange results")
        # check if there is more than one point superior
        aoi = np.logical_and(values >= bornes[0], values <= bornes[1])
        if np.sum(aoi) == 1:
            inds = np.where(aoi)
            x = self.axe_x[inds[1][0]]
            y = self.axe_y[inds[0][0]]
            return Points([[x, y]], unit_x=self.unit_x,
                          unit_y=self.unit_y)
        zones = np.logical_and(np.logical_and(values >= bornes[0],
                                              values <= bornes[1]),
                               np.logical_not(mask))
        # compute the center with labelzones
        labeledzones, nmbzones = msr.label(zones)
        inds = []
        if kind == 'extremum':
            mins, _, ind_min, ind_max = msr.extrema(values,
                                                    labeledzones,
                                                    np.arange(nmbzones) + 1)
            for i in np.arange(len(mins)):
                if bornes[np.argmax(np.abs(bornes))] < 0:
                    inds.append(ind_min[i])
                else:
                    inds.append(ind_max[i])
        elif kind == 'center':
            inds = msr.center_of_mass(np.ones(self.shape),
                                      labeledzones,
                                      np.arange(nmbzones) + 1)
        elif kind == 'ponderated':
            inds = msr.center_of_mass(np.abs(values), labeledzones,
                                      np.arange(nmbzones) + 1)
        else:
            raise ValueError("Invalid value for 'kind'")
        coords = []
        for ind in inds:
            indx = ind[1]
            indy = ind[0]
            if indx % 1 == 0:
                x = self.axe_x[indx]
            else:
                dx = self.axe_x[1] - self.axe_x[0]
                x = self.axe_x[int(indx)] + dx*(indx % 1)
            if indy % 1 == 0:
                y = self.axe_y[indy]
            else:
                dy = self.axe_y[1] - self.axe_y[0]
                y = self.axe_y[int(indy)] + dy*(indy % 1)
            coords.append([x, y])
        coords = np.array(coords, dtype=float)
        if len(coords) == 0:
            return None
        return Points(coords, unit_x=self.unit_x, unit_y=self.unit_y)

    def get_profile(self, direction, position):
        """
        Return a profile of the scalar field, at the given position (or at
        least at the nearest possible position).
        If position is an interval, the fonction return an average profile
        in this interval. (Tested)

        Function
        --------
        axe, profile, cutposition = get_profile(direction, position)

        Parameters
        ----------
        direction : integer
            Direction along which we choose a position (1 for x and 2 for y)
        position : float or interval of float
            Position or interval in which we want a profile

        Returns
        -------
        profile : Profile object
            Wanted profile
        cutposition : array or number
            Final position or interval in which the profile has been taken
        """
        if not isinstance(direction, int):
            raise TypeError("'direction' must be an integer between 1 and 2")
        if not (direction == 1 or direction == 2):
            raise ValueError("'direction' must be an integer between 1 and 2")
        if not isinstance(position, NUMBERTYPES + ARRAYTYPES):
            raise TypeError("'position' must be a number or an interval")
        if isinstance(position, ARRAYTYPES):
            position = np.array(position, dtype=float)
            if not position.shape == (2,):
                raise ValueError("'position' must be a number or an interval")
        if direction == 1:
            axe = self.axe_x
            unit_x = self.unit_y
            unit_y = self.unit_values
        else:
            axe = self.axe_y
            unit_x = self.unit_x
            unit_y = self.unit_values
        if isinstance(position, ARRAYTYPES):
            for pos in position:
                if pos > axe.max() or pos < axe.min():
                    raise ValueError("'position' must be included in"
                                     " the choosen axis values")
        else:
            if position > axe.max() or position < axe.min():
                raise ValueError("'position' must be included in the choosen"
                                 " axis values (here [{0},{1}])"
                                 .format(axe.min(), axe.max()))
        # computation of the profile for a single position
        if isinstance(position, NUMBERTYPES):
            for i in np.arange(1, len(axe)):
                if (axe[i] >= position and axe[i-1] <= position) \
                        or (axe[i] <= position and axe[i-1] >= position):
                    break
            if np.abs(position - axe[i]) > np.abs(position - axe[i-1]):
                finalindice = i-1
            else:
                finalindice = i
            if direction == 1:
                profile = self.values[finalindice, :]
                axe = self.axe_y
                cutposition = self.axe_x[finalindice]
            else:
                profile = self.values[:, finalindice]
                axe = self.axe_x
                cutposition = self.axe_y[finalindice]
        # Calculation of the profile for an interval of position
        else:
            axe_mask = np.logical_and(axe >= position[0], axe <= position[1])
            if direction == 1:
                profile = self.values[axe_mask, :].mean(1)
                axe = self.axe_y
                cutposition = self.axe_x[axe_mask]
            else:
                profile = self.values[:, axe_mask].mean(0)
                axe = self.axe_x
                cutposition = self.axe_y[axe_mask]
        return Profile(axe, profile, unit_x, unit_y, "Profile"), cutposition

#    def get_curve(self, bornes=[.75, 1], rel=True, order=5):
#        """
#        Return a Points object, representing the choosen zone, and polynomial
#        interpolation coefficient of these points.
#
#        Parameters
#        ----------
#        bornes : 2x1 array, optionnal
#            Trigger values determining the zones.
#            '[inferior borne, superior borne]'
#        rel : Boolean
#            If 'rel' is 'True' (default), values of 'bornes' are relative to
#            the extremum values of the field.
#            If 'rel' is 'False', values of bornes are treated like absolute
#            values.
#        order : number
#            Order of the polynomial interpolation (default=5).
#
#        Returns
#        -------
#        pts : Points object
#        coefp : array of number
#            interpolation coefficients (higher order first).
#        """
#        if not isinstance(bornes, ARRAYTYPES):
#            raise TypeError("'bornes' must be an array")
#        if not isinstance(bornes, np.ndarray):
#            bornes = np.array(bornes)
#        if not bornes.shape == (2,):
#            raise ValueError("'bornes' must be a 2x1 array")
#        if not bornes[0] < bornes[1]:
#            raise ValueError("'bornes' must be crescent")
#        if not isinstance(rel, bool):
#            raise TypeError("'rel' must be a boolean")
#        if rel:
#            if np.abs(bornes[0]) > np.abs(bornes[1]):
#                bornes *= np.abs(self.get_min())
#                coef = -1
#            else:
#                bornes *= np.abs(self.get_max())
#                coef = 1
#        # récupération des zones
#        zone = np.logical_and(self.values > bornes[0], self.values < bornes[1])
#        labeledzones, nmbzones = msr.label(zone)
#        # vérification du nombre de zones et récupération de la plus grande
#        areas = []
#        if nmbzones > 1:
#            zones = msr.find_objects(labeledzones, nmbzones)
#            area = []
#            for i in np.arange(nmbzones):
#                slices = zones[i]
#                area = (slices[0].stop - slices[0].start) *  \
#                       (slices[1].stop - slices[1].start)
#                areas.append(area)
#            areas = np.array(areas)
#            ind = areas.argmax()
#            labeledzones = labeledzones == ind + 1
#        # Récupération des points
#        mask = labeledzones == 0
#        pts = self.export_to_scatter(mask=mask)
#        pts.v = pts.v*coef
#        # interpolation
#        coefp = pts.fit(order=order)
#        return pts, coefp

#    def get_bl(self, direction, kind="default", perc=.99):
#        """
#        Return one of the boundary layer characteristic.
#        WARNING : the wall must be at x=0.
#
#        Parameters
#        ----------
#        direction : integer
#            Direction of the wall supporting the BL.
#        kind : string
#            Type of boundary layer thickness you want.
#            default : For a bl thickness at a given value (typically 90%).
#            displacement : For the bl displacement thickness.
#            momentum : For the bl momentum thickness.
#            H factor : For the shape factor.
#        perc : number
#            Relative limit velocity defining the default boundary layer.
#            (Only usefull with kind='default')
#
#        Returns
#        -------
#        profile : Profile object
#            Wanted isocurve.
#        """
#        if not isinstance(perc, NUMBERTYPES):
#            raise TypeError("'value' must be a number")
#        if not isinstance(direction, int):
#            raise TypeError("'direction' must be an integer")
#        if not isinstance(kind, str):
#            raise TypeError("'kind' must be a string")
#        if not (direction == 1 or direction == 2):
#            raise ValueError("'direction' must be '1' or '2'")
#        if not (perc <= 1 and perc > 0):
#            raise ValueError("'value' must be between 0 and 1")
#        isoc = []
#        axec = []
#        axe_x, axe_y = self.get_axes()
#        if direction == 1:
#            axe = axe_x
#            unit_x = self.unit_x.copy()
#            unit_y = self.unit_y.copy()
#        else:
#            axe = axe_y
#            unit_x = self.unit_y.copy()
#            unit_y = self.unit_x.copy()
#        for axevalue in axe:
#            profile, _ = self.get_profile(direction, axevalue)
#            # vérification du nombre de valeurs non-masquée
#            if profile is None:
#                continue
#            if len(profile.y[profile.y.mask]) > 0.5*len(profile.y):
#                continue
#            from .boundary_layer import get_bl_thickness, get_displ_thickness,\
#                get_momentum_thickness, get_shape_factor
#            if kind == "default":
#                val = get_bl_thickness(profile, perc=perc)
#            elif kind == "displacement":
#                val = get_displ_thickness(profile)
#            elif kind == "momentum":
#                val = get_momentum_thickness(profile)
#            elif kind == "H factor":
#                val = get_shape_factor(profile)
#            else:
#                raise ValueError("Unknown value for 'kind'")
#            isoc.append(val)
#            axec.append(axevalue)
#        return Profile(axec, isoc, unit_x, unit_y, "Boundary Layer")

    def copy(self):
        """
        Return a copy of the scalarfield.
        """
        return copy.deepcopy(self)

    def trim_area(self, intervalx=None, intervaly=None):
        """
        Return a trimed  area in respect with given intervals.

        Parameters
        ----------
        intervalx : array, optional
            interval wanted along x
        intervaly : array, optional
            interval wanted along y
        """
        indmin_x, indmax_x, indmin_y, indmax_y, trimfield = \
            Field.trim_area(self, intervalx, intervaly, full_output=True)
        trimfield.values = self.values[indmin_x:indmax_x + 1,
                                       indmin_y:indmax_y + 1]
        return trimfield

    def crop_masked_border(self):
        """
        Crop the masked border of the field in place.
        """
        axe_x, axe_y = self.axe_x, self.axe_y
        # checking masked values presence
        mask = self.mask
        if np.any(mask):
            return None
        # getting values
        values = self.values
        # crop border along y
        axe_y_m = np.logical_not(np.all(mask, axis=0))
        if np.any(axe_y_m):
            values = values[:, axe_y_m]
            mask = mask[:, axe_y_m]
        # crop values along x
        axe_x_m = np.logical_not(np.all(mask, axis=1))
        if np.any(axe_x_m):
            values = values[axe_x_m, :]
            mask = mask[axe_x_m, :]
        # storing cropped values
        cropped_field = self.__class__()
        Field.__init__(cropped_field)
        cropped_field.axe_x = axe_x[axe_x_m]
        cropped_field.axe_y = axe_y[axe_y_m]
        cropped_field.values = values
        self = cropped_field

    def fill(self, tof='interplin', value=0., crop_border=True):
        # TODO : Make this fucking functionnality work !
        """
        Fill the masked part of the array in place.

        Parameters
        ----------
        tof : string, optional
            Type of algorithm used to fill.
            'value' : fill with a given value
            'interplin' : fill using linear interpolation
            'interpcub' : fill using cubic interpolation
        value : number
            Value for filling (only usefull with tof='value')
        crop_border : boolean
            If 'True' (default), masked borders of the field are cropped
            before filling. Else, values on border are extrapolated (poorly).
        """
        # check parameters coherence
        if not isinstance(tof, STRINGTYPES):
            raise TypeError("'tof' must be a string")
        if not isinstance(value, NUMBERTYPES):
            raise TypeError("'value' must be a number")
        # deleting the masked border (useless field part)
        if crop_border:
            self.crop_masked_border()
        mask = self.mask
        not_mask = np.logical_not(mask)
        values = self.values
        # if there is nothing to do...
        if not np.any(mask):
            pass
        elif tof == 'interplin':
            inds_x = np.arange(values.shape[1])
            inds_y = np.arange(values.shape[0])
            grid_x, grid_y = np.meshgrid(inds_x, inds_y)
            f = spinterp.interp2d(grid_y[not_mask],
                                  grid_x[not_mask],
                                  values[not_mask])
            for inds, masked in np.ndenumerate(mask):
                if masked:
                    values[inds[0], inds[1]] = f(inds[0], inds[1])
            self.values = values
            mask = np.empty(self.shape, dtype=bool)
            mask.fill(False)
            self.__mask = mask
        elif tof == 'interpcub':
            inds_x = np.arange(values.shape[1])
            inds_y = np.arange(values.shape[0])
            grid_x, grid_y = np.meshgrid(inds_x, inds_y)
            f = spinterp.interp2d(grid_y[not_mask],
                                  grid_x[not_mask],
                                  values[not_mask],
                                  kind='cubic')
            for inds, masked in np.ndenumerate(mask):
                if masked:
                    values[inds[1], inds[0]] = f(inds[1], inds[0])
            self.values = values
            self.__ScalarField_mask = np.empty(self.shape).fill(False)
        elif tof == 'value':
            values[mask] = value
            self.values = values
            mask = np.empty(self.shape, dtype=bool)
            mask.fill(False)
            self.__mask = mask
        else:
            raise ValueError("unknown 'tof' value")

    def smooth(self, tos='uniform', size=None, **kw):
        """
        Smooth the scalarfield in place.
        Warning : fill up the field (should be used carefully with masked field
        borders)

        Parameters :
        ------------
        tos : string, optional
            Type of smoothing, can be 'uniform' (default) or 'gaussian'
            (See ndimage module documentation for more details)
        size : number, optional
            Size of the smoothing (is radius for 'uniform' and
            sigma for 'gaussian').
            Default is 3 for 'uniform' and 1 for 'gaussian'.
        kw : dic
            Additional parameters for ndimage methods
            (See ndimage documentation)
        """
        if not isinstance(tos, STRINGTYPES):
            raise TypeError("'tos' must be a string")
        if size is None and tos == 'uniform':
            size = 3
        elif size is None and tos == 'gaussian':
            size = 1
        # filling up the field before smoothing
        self.fill()
        # mask treatment
        values = self.values
        # smoothing
        if tos == "uniform":
            values = ndimage.uniform_filter(values, size, **kw)
        elif tos == "gaussian":
            values = ndimage.gaussian_filter(values, size, **kw)
        else:
            raise ValueError("'tos' must be 'uniform' or 'gaussian'")
        # storing
        self.values = values

    def integrate_over_line(self, direction, interval):
        """
        Return the integral on an interval and along a direction
        (1 for x and 2 for y).
        Discretized integral is computed with a trapezoidal algorithm.

        Function
        --------
        integrale, unit = integrate_over_line(direction, interval)

        Parameters
        ----------
        direction : integer
            Direction along which we choose a position (1 for x and 2 for y)
        interval : interval of numbers
            Interval on which we want to calculate the integrale

        Returns
        -------
        integral : float
            Result of the integrale calcul
        unit : Unit object
            Unit of the result

        """
        profile, _ = self.get_profile(direction, interval)
        integrale = np.trapz(profile.y, profile.x)
        if direction == 1:
            unit = self.unit_values*self.unit_y
        else:
            unit = self.unit_values*self.unit_x
        return integrale*unit

    def integrate_over_surface(self, intervalx=None, intervaly=None):
        """
        Return the integral on a surface.
        Discretized integral is computed with a very rustic algorithm
        which just sum the value on the surface.
        if 'intervalx' and 'intervaly' are given, return the integral over the
        delimited surface.
        WARNING : Only works (and badly) with regular axes.

        Function
        --------
        integrale, unit = integrate_over_surface(intervalx, intervaly)

        Parameters
        ----------
        intervalx : interval of numbers, optional
            Interval along x on which we want to compute the integrale.
        intervaly : interval of numbers, optional
            Interval along y on which we want to compute the integrale.

        Returns
        -------
        integral : float
            Result of the integrale computation.
        unit : Unit object
            The unit of the integrale result.
        """
        axe_x, axe_y = self.axe_x, self.axe_y
        if intervalx is None:
            intervalx = [axe_x[0], axe_x[-1]]
        if intervaly is None:
            intervaly = [axe_y[0], axe_y[-1]]
        trimfield = self.trim_area(intervalx, intervaly)
        axe2_x, axe2_y = trimfield.axe_x, trimfield.axe_y
        unit_x, unit_y = trimfield.unit_x, trimfield.unit_y
        integral = (trimfield.values.sum()
                    * np.abs(axe2_x[-1] - axe2_x[0])
                    * np.abs(axe2_y[-1] - axe2_y[0])
                    / len(axe2_x)
                    / len(axe2_y))
        unit = trimfield.unit_values*unit_x*unit_y
        return integral*unit

    def _display(self, kind=None, **plotargs):
        # getting datas
        axe_x, axe_y = self.axe_x, self.axe_y
        values = np.transpose(self.values)
        mask = np.transpose(self.mask)
        print(mask)
        unit_x, unit_y = self.unit_x, self.unit_y
        X, Y = np.meshgrid(self.axe_y, self.axe_x)
        values[mask] = np.nan
        # displaying according to 'kind'
        if kind == 'contour':
            if (not 'cmap' in plotargs.keys()
                    and not 'colors' in plotargs.keys()):
                plotargs['cmap'] = cm.jet
            displ = plt.contour(axe_x, axe_y, values, linewidth=1, **plotargs)
        elif kind == 'contourf':
            if 'cmap' in plotargs.keys() or 'colors' in plotargs.keys():
                displ = plt.contourf(axe_x, axe_y, values, linewidth=1,
                                     **plotargs)
            else:
                displ = plt.contourf(axe_x, axe_y, values, cmap=cm.jet,
                                     linewidth=1, **plotargs)
        elif kind == "imshow" or kind is None:
            if not 'cmap' in plotargs.keys():
                plotargs['cmap'] = cm.jet
            if not 'interpolation' in plotargs.keys():
                plotargs['interpolation'] = 'nearest'
            if not 'aspect' in plotargs.keys():
                plotargs['aspect'] = 'equal'
            delta_x = axe_x[1] - axe_x[0]
            delta_y = axe_y[1] - axe_y[0]
            displ = plt.imshow(values,
                               extent=(axe_x[0] - delta_x/2.,
                                       axe_x[-1] + delta_x/2.,
                                       axe_y[0] - delta_y/2.,
                                       axe_y[-1] + delta_y/2.),
                               origin='lower', **plotargs)
        else:
            raise ValueError("Unknown 'kind' of plot for ScalarField object")
        plt.axis('equal')
        plt.xlabel("X " + unit_x.strUnit())
        plt.ylabel("Y " + unit_y.strUnit())
        return displ

    def display(self, kind=None, **plotargs):
        """
        Display the scalar field.

        Parameters
        ----------
        kind : string, optinnal
            If 'imshow': (default) each datas are plotted (imshow),
            if 'contour': contours are ploted (contour),
            if 'contourf': filled contours are ploted (contourf),
            if '3D': a tri-dimensionnal plot is created.
        **plotargs : dict
            Arguments passed to the 'contourf' function used to display the
            scalar field.

        Returns
        -------
        fig : figure reference
            Reference to the displayed figure.
        """
        displ = self._display(kind, **plotargs)
        plt.title("Scalar field Values " + self.unit_values.strUnit())
        cb = plt.colorbar(displ, shrink=1, aspect=5)
        cb.set_label(self.unit_values.strUnit())
        # search for limits in case of masked field
        mask = self.mask
        for i in np.arange(len(self.axe_x)):
            if not np.all(mask[i, :]):
                break
        xmin = self.axe_x[i]
        for i in np.arange(len(self.axe_x) - 1, -1, -1):
            if not np.all(mask[i, :]):
                break
        xmax = self.axe_x[i]
        for i in np.arange(len(self.axe_y)):
            if not np.all(mask[:, i]):
                break
        ymin = self.axe_y[i]
        for i in np.arange(len(self.axe_y) - 1, -1, -1):
            if not np.all(mask[:, i]):
                break
        ymax = self.axe_y[i]
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        return displ


class VectorField(Field):
    """
    Class representing a vector field (2D field, with two components on each
    point).

    Principal methods
    -----------------
    "import_from_*" : allows to easily create or import vector fields.

    "export_to_*" : allows to export.

    "display" : display the vector field, with these unities.

    Examples
    --------
    >>> import IMTreatment as imt
    >>> VF = imt.VectorField()
    >>> unit_axe = make_unit('cm')
    >>> unit_K = make_unit('K')
    >>> comp_x = [[4, 8], [4, 8]]
    >>> comp_y = [[1, 2], [3, 4]]
    >>> VF.import_from_arrays([1,2], [1,2], comp_x, comp_y,  unit_axe,
    ...                       unit_axe, unit_K, unit_K)
    >>> VF.display()
    """

    def __init__(self):
        Field.__init__(self)
        self.__comp_x = np.array([], dtype=float)
        self.__comp_y = np.array([], dtype=float)
        self.__mask = np.array([], dtype=bool)
        self.__unit_values = make_unit('')

    def __neg__(self):
        tmpvf = self.copy()
        tmpvf.comp_x = -tmpvf.comp_x
        tmpvf.comp_y = -tmpvf.comp_y
        return tmpvf

    def __add__(self, other):
        if isinstance(other, VectorField):
            axe_x, axe_y = self.axe_x, self.axe_y
            oaxe_x, oaxe_y = other.axe_x, other.axe_y
            if (all(axe_x != oaxe_x) or all(axe_y != oaxe_y)):
                raise ValueError("Vector fields have to be consistent "
                                 "(same dimensions)")
            try:
                self.unit_values + other.unit_values
                self.unit_x + other.unit_x
                self.unit_y + other.unit_y
            except:
                raise ValueError("I think these units don't match, fox")
            tmpvf = self.copy()
            fact = (other.unit_values / self.unit_values).asNumber()
            tmpvf.comp_x = self.comp_x + other.comp_x*fact
            tmpvf.comp_y = self.comp_y + other.comp_y*fact
            return tmpvf
        elif isinstance(other, unum.Unum):
            tmpvf = self.copy()
            fact = (other / self.unit_values).asNumber()
            tmpvf.comp_x = self.comp_x + fact
            tmpvf.comp_y = self.comp_y + fact
            return tmpvf
        else:
            raise TypeError("You can only add a velocity field "
                            "with others velocity fields")

    def __sub__(self, other):
        other_tmp = other.__neg__()
        tmpvf = self.__add__(other_tmp)
        return tmpvf

    def __truediv__(self, number):
        if isinstance(number, unum.Unum):
            tmpvf = self.copy()
            tmpvf.comp_x /= (number/self.unit_values).asNumber()
            tmpvf.comp_y /= (number/self.unit_values).asNumber()
            return tmpvf
        elif isinstance(number, NUMBERTYPES):
            tmpvf = self.copy()
            tmpvf.comp_x /= number
            tmpvf.comp_y /= number
            return tmpvf
        else:
            raise TypeError("You can only divide a vector field "
                            "by numbers")

    __div__ = __truediv__

    def __mul__(self, number):
        if isinstance(number, unum.Unum):
            tmpvf = self.copy()
            tmpvf.comp_x *= (number/self.unit_values).asNumber()
            tmpvf.comp_y *= (number/self.unit_values).asNumber()
            return tmpvf
        elif isinstance(number, NUMBERTYPES):
            tmpvf = self.copy()
            tmpvf.comp_x *= number
            tmpvf.comp_y *= number
            return tmpvf
        else:
            raise TypeError("You can only multiply a vector field "
                            "by numbers")

    __rmul__ = __mul__

    def __sqrt__(self):
        tmpvf = self.copy()
        tmpvf.comp_x = np.sqrt(tmpvf.comp_x)
        tmpvf.comp_y = np.sqrt(tmpvf.comp_y)
        return tmpvf

    def __pow__(self, number):
        if not isinstance(number, NUMBERTYPES):
            raise TypeError("You only can use a number for the power "
                            "on a Vectorfield")
        tmpvf = self.copy()
        tmpvf.comp_x = np.power(tmpvf.comp_x, number)
        tmpvf.comp_y = np.power(tmpvf.comp_y, number)
        return tmpvf

    def __iter__(self):
        mask = self.mask
        datax = self.comp_x
        datay = self.comp_y
        for ij, xy in Field.__iter__(self):
            i = ij[0]
            j = ij[1]
            if not mask[j, i]:
                yield ij, xy, [datax[j, i], datay[j, i]]

    @property
    def comp_x(self):
        return self.__comp_x

    @comp_x.setter
    def comp_x(self, new_comp_x):
        if not isinstance(new_comp_x, ARRAYTYPES):
            raise TypeError()
        new_comp_x = np.array(new_comp_x)
        if not new_comp_x.shape == self.shape:
            raise ValueError("'comp_x' must be coherent with axis system")
        # adapting mask to 'nan' values
        add_mask = np.isnan(new_comp_x)
        if self.mask.shape == (0,):
            self.mask = add_mask
        else:
            self.mask = np.logical_or(self.mask, add_mask)
        # storing data
        self.__comp_x = new_comp_x

    @comp_x.deleter
    def comp_x(self):
        raise Exception("Nope, can't do that")

    @property
    def comp_x_as_sf(self):
        tmp_sf = ScalarField()
        tmp_sf.import_from_arrays(self.axe_x, self.axe_y, self.comp_x,
                                  mask=self.mask, unit_x=self.unit_x,
                                  unit_y=self.unit_y,
                                  unit_values=self.unit_values)
        return tmp_sf

    @property
    def comp_y(self):
        return self.__comp_y

    @comp_y.setter
    def comp_y(self, new_comp_y):
        if not isinstance(new_comp_y, ARRAYTYPES):
            raise TypeError()
        new_comp_y = np.array(new_comp_y)
        if not new_comp_y.shape == self.shape:
            raise ValueError()
        # adapting mask to 'nan' values
        add_mask = np.isnan(new_comp_y)
        if self.mask.shape == (0,):
            self.mask = add_mask
        else:
            self.mask = np.logical_or(self.mask, add_mask)
        # storing data
        self.__comp_y = new_comp_y

    @comp_y.deleter
    def comp_y(self):
        raise Exception("Nope, can't do that")

    @property
    def comp_y_as_sf(self):
        tmp_sf = ScalarField()
        tmp_sf.import_from_arrays(self.axe_x, self.axe_y, self.comp_y,
                                  mask=self.mask, unit_x=self.unit_x,
                                  unit_y=self.unit_y,
                                  unit_values=self.unit_values)
        return tmp_sf

    @property
    def mask(self):
        return self.__mask

    @mask.setter
    def mask(self, new_mask):
        # check 'new_mask' coherence
        if isinstance(new_mask, bool):
            fill_value = new_mask
            new_mask = np.empty(self.shape, dtype=bool)
            new_mask.fill(fill_value)
        elif isinstance(new_mask, ARRAYTYPES):
            new_mask = np.array(new_mask)
        else:
            raise TypeError("'mask' should be an array or a boolean,"
                            " not a {}".format(type(new_mask)))
        if self.shape != new_mask.shape:
            raise ValueError()
        # check if the new mask don'r reveal masked values
        if np.any(np.logical_not(new_mask[self.mask])):
            raise Warning("This mask reveal masked values, maybe you should"
                          "use the 'fill' function instead")
        # store mask
        self.__mask = new_mask

    @mask.deleter
    def mask(self):
        raise Exception("Nope, can't do that")

    @property
    def mask_as_sf(self):
        tmp_sf = ScalarField()
        tmp_sf.import_from_arrays(self.axe_x, self.axe_y, self.mask,
                                  mask=False, unit_x=self.unit_x,
                                  unit_y=self.unit_y,
                                  unit_values=self.unit_values)
        return tmp_sf

    @property
    def unit_values(self):
        return self.__unit_values

    @unit_values.setter
    def unit_values(self, new_unit_values):
        if isinstance(new_unit_values, unum.Unum):
            if new_unit_values.asNumber() == 1:
                self.__unit_values = new_unit_values
            else:
                raise ValueError()
        elif isinstance(new_unit_values, STRINGTYPES):
            self.__unit_values == make_unit(new_unit_values)
        else:
            raise TypeError()

    @unit_values.deleter
    def unit_values(self):
        raise Exception("Nope, can't do that")

#    def import_from_davis(self, filename):
#        """
#        Import a vector field from a .VC7 file
#
#        Parameters
#        ----------
#        filename : string
#            Path to the file to import.
#        """
#        if not isinstance(filename, STRINGTYPES):
#            raise TypeError("'filename' must be a string")
#        if not os.path.exists(filename):
#            raise ValueError("'filename' must ne an existing file")
#        if os.path.isdir(filename):
#            filename = glob.glob(os.path.join(filename, '*.vc7'))[0]
#        _, ext = os.path.splitext(filename)
#        if not (ext == ".vc7" or ext == ".VC7"):
#            raise ValueError("'filename' must be a vc7 file")
#        v = IM.VC7(filename)
#        self.comp_x = ScalarField()
#        self.comp_y = ScalarField()
#        # traitement des unités
#        unit_x = v.buffer['scaleX']['unit'].split("\x00")[0]
#        unit_x = unit_x.replace('[', '')
#        unit_x = unit_x.replace(']', '')
#        unit_y = v.buffer['scaleY']['unit'].split("\x00")[0]
#        unit_y = unit_y.replace('[', '')
#        unit_y = unit_y.replace(']', '')
#        unit_values = v.buffer['scaleI']['unit'].split("\x00")[0]
#        unit_values = unit_values.replace('[', '')
#        unit_values = unit_values.replace(']', '')
#        # vérification de l'ordre des axes (et correction)
#        x = v.Px[0, :]
#        y = v.Py[:, 0]
#        Vx = v.Vx[0]
#        Vy = v.Vy[0]
#        if x[-1] < x[0]:
#            x = x[::-1]
#            Vx = Vx[:, ::-1]
#            Vy = Vy[:, ::-1]
#        if y[-1] < y[0]:
#            y = y[::-1]
#            Vx = Vx[::-1, :]
#            Vy = Vy[::-1, :]
#        self.import_from_arrays(x, y, Vx, Vy, make_unit(unit_x),
#                                make_unit(unit_y), make_unit(unit_values))

#    def import_from_ascii(self, filename, x_col=1, y_col=2, vx_col=3,
#                          vy_col=4, unit_x=make_unit(""), unit_y=make_unit(""),
#                          unit_values=make_unit(""), **kwargs):
#        """
#        Import a vectorfield from an ascii file.
#
#        Parameters
#        ----------
#        x_col, y_col, vx_col, vy_col : integer, optional
#            Colonne numbers for the given variables
#            (begining at 1).
#        unit_x, unit_y, unit_v : Unit objects, optional
#            Unities for the given variables.
#        **kwargs :
#            Possibles additional parameters are the same as those used in the
#            numpy function 'genfromtext()' :
#            'delimiter' to specify the delimiter between colonnes.
#            'skip_header' to specify the number of colonne to skip at file
#                begining
#            ...
#        """
#        # validating parameters
#        if not isinstance(x_col, int) or not isinstance(y_col, int)\
#                or not isinstance(vx_col, int) or not isinstance(vy_col, int):
#            raise TypeError("'x_col', 'y_col', 'vx_col' and 'vy_col' must "
#                            "be integers")
#        if x_col < 1 or y_col < 1 or vx_col < 1 or vy_col < 1:
#            raise ValueError("Colonne number out of range")
#        # 'names' deletion, if specified (dangereux pour la suite)
#        if 'names' in kwargs:
#            kwargs.pop('names')
#        # extract data from file
#        data = np.genfromtxt(filename, **kwargs)
#        # get axes
#        x = data[:, x_col-1]
#        x_org = np.unique(x)
#        y = data[:, y_col-1]
#        y_org = np.unique(y)
#        vx = data[:, vx_col-1]
#        vy = data[:, vy_col-1]
#        # Masking all the initial fields (to handle missing values)
#        vx_org = np.zeros((y_org.shape[0], x_org.shape[0]))
#        vx_org_mask = np.ones(vx_org.shape)
#        vx_org = np.ma.masked_array(vx_org, vx_org_mask)
#        vy_org = np.zeros((y_org.shape[0], x_org.shape[0]))
#        vy_org_mask = np.ones(vy_org.shape)
#        vy_org = np.ma.masked_array(vy_org, vy_org_mask)
#        #loop on all 'v' values
#        for i in np.arange(vx.shape[0]):
#            x_tmp = x[i]
#            y_tmp = y[i]
#            vx_tmp = vx[i]
#            vy_tmp = vy[i]
#            #find x index
#            for j in np.arange(x_org.shape[0]):
#                if x_org[j] == x_tmp:
#                    x_ind = j
#            #find y index
#            for j in np.arange(y_org.shape[0]):
#                if y_org[j] == y_tmp:
#                    y_ind = j
#            #put the value at its place
#            vx_org[y_ind, x_ind] = vx_tmp
#            vy_org[y_ind, x_ind] = vy_tmp
#        # Treating 'nan' values
#        vx_org.mask = np.logical_or(vx_org.mask, np.isnan(vx_org.data))
#        vy_org.mask = np.logical_or(vy_org.mask, np.isnan(vy_org.data))
#        #store field in attributes
#        self.import_from_arrays(x_org, y_org, vx_org, vy_org, unit_x, unit_y,
#                                unit_values)

    def import_from_arrays(self, axe_x, axe_y, comp_x, comp_y, mask=False,
                           unit_x="", unit_y="", unit_values=""):
        """
        Set the vector field from a set of arrays.

        Parameters
        ----------
        axe_x : array
            Discretized axis value along x
        axe_y : array
            Discretized axis value along y
        comp_x : array or masked array
            Values of the x component at the discritized points
        comp_y : array or masked array
            Values of the y component at the discritized points
        mask : array of boolean, optional
            Mask on comp_x and comp_y
        unit_x : string, optionnal
            Unit for the values of axe_x
        unit_y : string, optionnal
            Unit for the values of axe_y
        unit_values : string, optionnal
            Unit for the field components.
        """
        self.axe_x = axe_x
        self.axe_y = axe_y
        self.comp_x = comp_x
        self.comp_y = comp_y
        self.mask = mask
        self.unit_x = unit_x
        self.unit_y = unit_y
        self.unit_values = unit_values

    def export_to_velocityfield(self, time=None, unit_time=''):
        if time is None:
            time = 0
        axe_x, axe_y = self.axe_x, self.axe_y
        unit_x, unit_y = self.unit_x, self.unit_y
        value_x = self.comp_x
        value_y = self.comp_y
        mask = self.mask
        unit_values = self.unit_values
        tmpvf = VelocityField()
        tmpvf.import_from_arrays(axe_x, axe_y, value_x, value_y, mask=mask,
                                 time=time, unit_x=unit_x, unit_y=unit_y,
                                 unit_values=unit_values, unit_time=unit_time)
        return tmpvf

#    def import_from_scalarfields(self, comp_x, comp_y):
#        """
#        Import a vector field from two scalarfields.
#
#        Parameters
#        ----------
#        CompX : ScalarField
#            x component of th vector field.
#        CompY : ScalarField
#            y component of th vector field.
#        """
#        if not isinstance(comp_x, ScalarField):
#            raise TypeError("'comp_x' must be a ScalarField object")
#        if not isinstance(comp_y, ScalarField):
#            raise TypeError("'comp_y' must be a ScalarField object")
#        axe_x, axe_y = comp_x.get_axes()
#        xunit_x, xunit_y = comp_x.get_axe_units()
#        xunit_values = comp_x.get_unit_values()
#        yunit_x, yunit_y = comp_y.get_axe_units()
#        yunit_values = comp_y.get_unit_values()
#        comp_x = comp_x.get_comp('values', raw=True)
#        comp_y = comp_y.get_comp('values', raw=True)
#        if not comp_x.shape == comp_y.shape:
#            raise ValueError("'comp_x' and 'comp_y' must have the same "
#                             "dimensions")
#        if not (xunit_x == yunit_x or xunit_y == yunit_y
#                or xunit_values == yunit_values):
#            raise ValueError("Unities of the two components and their axis "
#                             "must be the same")
#        self.import_from_arrays(axe_x, axe_y, comp_x, comp_y, xunit_x, xunit_y,
#                                xunit_values)

#    def import_from_file(self, filepath, **kw):
#        """
#        Load a VectorField object from the specified file using the JSON
#        format.
#        Additionnals arguments for the JSON decoder may be set with the **kw
#        argument. Such as'encoding' (to change the file
#        encoding, default='utf-8').
#
#        Parameters
#        ----------
#        filepath : string
#            Path specifiing the VectorField to load.
#        """
#        import IMTreatment.io.io as imtio
#        tmpvf = imtio.import_from_file(filepath, **kw)
#        if tmpvf.__classname__ != self.__classname__:
#            raise IOError("This file do not contain a VectorField, cabron.")
#        self = tmpvf

#    def export_to_file(self, filepath, compressed=True, **kw):
#        """
#        Write the Profile object in the specified file usint the JSON format.
#        Additionnals arguments for the JSON encoder may be set with the **kw
#        argument. Such arguments may be 'indent' (for visual indentation in
#        file, default=0) or 'encoding' (to change the file encoding,
#        default='utf-8').
#        If existing, specified file will be truncated. If not, it will
#        be created.
#
#        Parameters
#        ----------
#        filepath : string
#            Path specifiing where to save the ScalarField.
#        compressed : boolean, optional
#            If 'True' (default), the json file is compressed using gzip.
#        """
#        import IMTreatment.io.io as imtio
#        imtio.export_to_file(self, filepath, compressed, **kw)

#    def export_to_vtk(self, filepath, axis=None):
#        """
#        Export the vector field to a .vtk file, for Mayavi use.
#
#        Parameters
#        ----------
#        filepath : string
#            Path where to write the vtk file.
#        axis : tuple of strings
#            By default, scalar field axe are set to (x,y), if you want
#            different axis, you have to specified them here.
#            For example, "('z', 'y')", put the x scalar field axis values
#            in vtk z axis, and y scalar field axis in y vtk axis.
#        """
#        import pyvtk
#        if not os.path.exists(os.path.dirname(filepath)):
#            raise ValueError("'filepath' is not a valid path")
#        if axis is None:
#            axis = ('x', 'y')
#        if not isinstance(axis, ARRAYTYPES):
#            raise TypeError("'axis' must be a 2x1 tuple")
#        if not isinstance(axis[0], STRINGTYPES) \
#                or not isinstance(axis[1], STRINGTYPES):
#            raise TypeError("'axis' must be a 2x1 tuple of strings")
#        if not axis[0] in ['x', 'y', 'z'] or not axis[1] in ['x', 'y', 'z']:
#            raise ValueError("'axis' strings must be 'x', 'y' or 'z'")
#        if axis[0] == axis[1]:
#            raise ValueError("'axis' strings must be different")
#        Vx, Vy = self.get_comp('x', raw=True), self.get_comp('y', raw=True)
#        Vx = Vx.flatten()
#        Vy = Vy.flatten()
#        x, y = self.get_axes()
#        x_vtk = 0.
#        y_vtk = 0.
#        z_vtk = 0.
#        vx_vtk = np.zeros(Vx.shape)
#        vy_vtk = np.zeros(Vx.shape)
#        vz_vtk = np.zeros(Vx.shape)
#        if axis[0] == 'x':
#            x_vtk = x
#            vx_vtk = Vx
#        elif axis[0] == 'y':
#            y_vtk = x
#            vy_vtk = Vx
#        else:
#            z_vtk = x
#            vz_vtk = Vx
#        if axis[1] == 'x':
#            x_vtk = y
#            vx_vtk = Vy
#        elif axis[1] == 'y':
#            y_vtk = y
#            vy_vtk = Vy
#        else:
#            z_vtk = y
#            vz_vtk = Vy
#        vect = zip(vx_vtk, vy_vtk, vz_vtk)
#        point_data = pyvtk.PointData(pyvtk.Vectors(vect, "Vector field"))
#        grid = pyvtk.RectilinearGrid(x_vtk, y_vtk, z_vtk)
#        data = pyvtk.VtkData(grid, 'Vector Field from python', point_data)
#        data.tofile(filepath)
#
#    def get_values(self):
#        """
#        Return the values of the 2 components.
#        """
#        return self.comp_x, self.comp_y

#    def get_mask(self):
#        """
#        Return the scalarfield mask.
#        """
#        Vx, Vy = self.get_values()
#        return np.logical_or(Vx.mask, Vy.mask)

#    def set_values(self, comp_x=None, comp_y=None):
#        """
#        fill the scalar field with the given values
#        """
#        if comp_x is not None:
#            if not isinstance(comp_x, ARRAYTYPES):
#                raise TypeError("'comp_x' must be an array")
#            comp_x = np.array(comp_x)
#            if not np.all(comp_x.shape == self.get_dim()):
#                raise ValueError("'comp_x' shape must agree with the "
#                                 "original field shape")
#            self.comp_x = comp_x
#        if comp_y is not None:
#            if not isinstance(comp_y, ARRAYTYPES):
#                raise TypeError("'comp_y' must be an array")
#            comp_y = np.array(comp_y)
#            if not np.all(comp_y.shape == self.get_dim()):
#                raise ValueError("'comp_y' shape must agree with the "
#                                 "original field shape")
#            self.comp_y = comp_y

#    def set_mask(self, mask):
#        """
#        fill the scalar field mask with the given values.
#        """
#        if not isinstance(mask, ARRAYTYPES):
#            raise TypeError("'values' must be an array")
#        mask = np.array(mask)
#        if not np.all(mask.shape == self.get_dim()):
#            raise ValueError("'values' shape must agree with the "
#                             "original field shape")
#        Vx, Vy = self.get_values()
#        Vx.mask = mask
#        Vy.mask = mask

#    def get_comp(self, componentname, raw=False, masked=True):
#        """
#        Return a ScalarField object representing a component of the
#        Vectorfield object.
#
#        Parameters
#        ----------
#        componentname : string
#            Can be 'Vx', 'Vy', 'mask' or 'unit_values'.
#        raw : boolean, optional
#            If 'False' (default), return a ScalarField object,
#            if 'True', return a masked array.
#        masked : boolean, optional
#            If 'True' (default), returned np.array can be masked array,
#            If 'False', returned array are always brut np.array
#
#        Returns
#        -------
#        component : ScalarField object or numpy masked array
#        """
#        if not isinstance(componentname, STRINGTYPES):
#            raise TypeError("'componentname' must be a string")
#        if componentname == 'V':
#            if masked:
#                values = (self.comp_x.copy(), self.comp_y.copy())
#            else:
#                values = (self.comp_x.data.copy(), self.comp_y.data.copy())
#        if componentname == 'Vx':
#            if masked:
#                values = self.comp_x.copy()
#            else:
#                values = self.comp_x.data.copy()
#        elif componentname == 'Vy':
#            if masked:
#                values = self.comp_y.copy()
#            else:
#                values = self.comp_y.data.copy()
#        elif componentname == 'mask':
#            values = np.logical_or(self.comp_x.mask, self.comp_y.mask)
#            if isinstance(values, np.bool_):
#                if values:
#                    values = np.ones(self.get_dim(), dtype=bool)
#                else:
#                    values = np.zeros(self.get_dim(), dtype=bool)
#        elif componentname == "unit_values":
#            return self.unit_values.copy()
#        else:
#            raise ValueError("unknown value of 'componentname'")
#        if raw:
#            return values
#        elif isinstance(values, tuple):
#            axe_x, axe_y = self.get_axes()
#            unit_x, unit_y = self.get_axe_units()
#            unit_values = self.get_comp('unit_values')
#            tmpsf = VectorField()
#            tmpsf.import_from_arrays(axe_x, axe_y, values[0], values[1],
#                                     unit_x, unit_y, unit_values)
#        else:
#            axe_x, axe_y = self.get_axes()
#            unit_x, unit_y = self.get_axe_units()
#            unit_values = self.get_comp('unit_values')
#            tmpsf = ScalarField()
#            tmpsf.import_from_arrays(axe_x, axe_y, values, unit_x, unit_y,
#                                     unit_values)
#            return tmpsf

#    def set_comp(self, componentname, value):
#        """
#        Fill the component 'componentname' with 'value'.
#
#        Parameters
#        ----------
#        componentname : string
#            Can be 'Vx', 'Vy', 'mask'.
#        value : array
#            Array with the same shape as the initial component
#        """
#        if not isinstance(componentname, STRINGTYPES):
#            raise TypeError("'componentname' must be a string")
#        if componentname == 'Vx':
#            if not isinstance(value, ARRAYTYPES):
#                raise TypeError("'value' must be an array")
#            if not isinstance(value, np.ma.MaskedArray):
#                value = np.ma.masked_array(value)
#            if self.get_dim() != value.shape:
#                raise ValueError("'value' dimensions are inconsistent with the"
#                                 " ScalarField shape")
#            self.comp_x = value
#        elif componentname == 'Vy':
#            if not isinstance(value, ARRAYTYPES):
#                raise TypeError("'value' must be an array")
#            if not isinstance(value, np.ma.MaskedArray):
#                value = np.ma.masked_array(value)
#            if self.get_dim() != value.shape:
#                raise ValueError("'value' dimensions are inconsistent with the"
#                                 " ScalarField shape")
#            self.comp_y = value
#        elif componentname == 'mask':
#            if not isinstance(value, ARRAYTYPES):
#                raise TypeError("'value' must be an array")
#            if not isinstance(value, np.ma.MaskedArray):
#                value = np.ma.masked_array(value)
#            if self.get_dim() != value.shape:
#                raise ValueError("'value' dimensions are inconsistent with the"
#                                 " ScalarField shape")
#            self.comp_x.mask = value
#            self.comp_y.mask = value
#        elif componentname == "unit_values":
#            if not isinstance(value, unum.Unum):
#                raise TypeError("'unit_value' should be a unit object")
#            self.unit_values = value
#        else:
#            raise ValueError("Unknown 'componentname' value")

    def copy(self):
        """
        Return a copy of the vectorfield.
        """
        return copy.deepcopy(self)

#    def get_dim(self):
#        """
#        Return the vector field dimensions.
#
#        Returns
#        -------
#        shape : tuple
#            Tuple of the dimensions (along X and Y) of the scalar field.
#        """
#        return self.comp_x.shape

#    def get_min(self, unit=False):
#        """
#        Return the minima of the magnitude of the field.
#
#        Parameters
#        ----------
#        unit : boolean, optinal
#            If True, a unit object is returned,
#            else (default), a float is returned.
#
#        Returns
#        -------
#        mini : float
#            Minima on the field
#        """
#        return self.get_magnitude().get_min(unit)

#    def get_max(self, unit=False):
#        """
#        Return the maxima of the magnitude of the field.
#
#        Parameters
#        ----------
#        unit : boolean, optinal
#            If True, a unit object is returned,
#            else (default), a float is returned.
#
#        Returns
#        -------
#        maxi: float
#            Maxima on the field
#        """
#        return self.get_magnitude().get_max(unit)

    def get_profile(self, component, direction, position):
        """
        Return a profile of the vector field component, at the given position
        (or at least at the nearest possible position).
        If position is an interval, the fonction return an average profile
        in this interval.

        Function
        --------
        axe, profile, cutposition = get_profile(component, direction, position)

        Parameters
        ----------
        component : integer
            component to treat.
        direction : integer
            Direction along which we choose a position (1 for x and 2 for y).
        position : float or interval of float
            Position or interval in which we want a profile.

        Returns
        -------
        profile : Profile object
            Asked profile.
        cutposition : array or number
            Final position or interval in which the profile has been taken.
        """
        if not isinstance(component, int):
            raise TypeError("'component' must be an integer")
        if component == 1:
            return self.comp_x_as_sf.get_profile(direction, position)
        elif component == 2:
            return self.comp_y_as_sf.get_profile(direction, position)
        else:
            raise ValueError("'component' must have the value of 1 or 2")

#    def get_bl(self, component, value, direction, rel=False, kind="default"):
#        """
#        Return the thickness of the boundary layer on the scalar field,
#        accoirdinf to the given component.
#        Warning : Just return the interpolated position of the first value
#        encontered.
#
#        Parameters
#        ----------
#        component : integer
#            Component to work on.
#        value : number
#            The wanted isovalue.
#        direction : integer
#            Direction along which the isocurve is drawn.
#        rel : Bool, optionnal
#            Determine if 'value' is absolute or relative to the maximum in
#            each line/column.
#        kind : string
#            Type of boundary layer thickness you want.
#            default : For a bl thickness at a given value (typically 90%).
#            displacement : For the bl displacement thickness.
#            momentum : For the bl momentum thickness.
#
#        Returns
#        -------
#        isoc : Profile object
#            Asked isocurve
#        """
#        if not isinstance(component, int):
#            raise TypeError("'component' must be an integer")
#        if not (component == 1 or component == 2):
#            raise ValueError("'component' must be 1 or 2")
#        if component == 1:
#            return self.get_comp('x').get_bl(direction, value, rel, kind)
#        else:
#            return self.get_comp('Vy').get_bl(value, direction, rel, kind)


    @property
    def magnitude(self):
        """
        Return a scalar field with the velocity field magnitude.
        """
        comp_x, comp_y = self.comp_x, self.comp_y
        mask = self.mask
        values = (comp_x**2 + comp_y**2)**(.5)
        values[mask] == np.nan
        return values

    @property
    def magnitude_as_sf(self):
        """
        Return a scalarfield with the velocity field magnitude.
        """
        tmp_sf = ScalarField()
        tmp_sf.import_from_arrays(self.axe_x, self.axe_y, self.magnitude,
                                  mask=False, unit_x=self.unit_x,
                                  unit_y=self.unit_y,
                                  unit_values=self.unit_values)
        return tmp_sf

    @property
    def theta(self):
        """
        Return a scalar field with the vector angle (in reference of the unit_y
        vector [1, 0]).

        Parameters:
        -----------
        low_velocity_filter : number
            If not zero, points where V < Vmax*low_velocity_filter are masked.

        Returns:
        --------
        theta_sf : ScalarField object
            Contening theta field.
        """
        # get data
        comp_x, comp_y = self.comp_x, self.comp_y
        not_mask = np.logical_not(self.mask)
        theta = np.zeros(self.shape)
        # getting angle
        norm = self.magnitude
        theta[not_mask] = comp_x[not_mask]/norm[not_mask]
        theta[not_mask] = np.arccos(theta[not_mask])
        theta[comp_y < 0] = 2*np.pi - theta[comp_y < 0]
        return theta

    @property
    def theta_as_sf(self):
        """
        Return a scalarfield with the velocity field angles.
        """
        tmp_sf = ScalarField()
        tmp_sf.import_from_arrays(self.axe_x, self.axe_y, self.theta,
                                  mask=False, unit_x=self.unit_x,
                                  unit_y=self.unit_y,
                                  unit_values=self.unit_values)
        return tmp_sf

    def smooth(self, tos='uniform', size=None, **kw):
        """
        Smooth the vectorfield in place.
        Warning : fill up the field (should be used carefully with masked field
        borders)

        Parameters :
        ------------
        tos : string, optional
            Type of smoothing, can be 'uniform' (default) or 'gaussian'
            (See ndimage module documentation for more details)
        size : number, optional
            Size of the smoothing (is radius for 'uniform' and
            sigma for 'gaussian').
            Default is 3 for 'uniform' and 1 for 'gaussian'.
        kw : dic
            Additional parameters for ndimage methods
            (See ndimage documentation)
        """
        if not isinstance(tos, STRINGTYPES):
            raise TypeError("'tos' must be a string")
        if size is None and tos == 'uniform':
            size = 3
        elif size is None and tos == 'gaussian':
            size = 1
        # filling up the field before smoothing
        self.fill()
        # getting data
        Vx, Vy = self.comp_x, self.comp_y
        # smoothing
        if tos == "uniform":
            Vx = ndimage.uniform_filter(Vx, size, **kw)
            Vy = ndimage.uniform_filter(Vy, size, **kw)
        elif tos == "gaussian":
            Vx = ndimage.gaussian_filter(Vx, size, **kw)
            Vy = ndimage.gaussian_filter(Vy, size, **kw)
        else:
            raise ValueError("'tos' must be 'uniform' or 'gaussian'")
        # storing
        self.comp_x = Vx
        self.comp_y = Vy

    def fill(self, tof='interplin', value=0., crop_border=True):
        # TODO : Make this fucking functionnality work !
        """
        Fill the masked part of the field in place.

        Parameters
        ----------
        tof : string, optional
            Type of algorithm used to fill.
            'value' : fill with a given value
            'interplin' : fill using linear interpolation
            'interpcub' : fill using cubic interpolation
        value : number
            Value for filling (only usefull with tof='value')
        crop_border : boolean
            If 'True' (default), masked borders of the field are cropped
            before filling. Else, values on border are extrapolated (poorly).
        """
        # check parameters coherence
        if not isinstance(tof, STRINGTYPES):
            raise TypeError("'tof' must be a string")
        if not isinstance(value, NUMBERTYPES):
            raise TypeError("'value' must be a number")
        # deleting the masked border (useless field part)
        if crop_border:
            self.crop_masked_border()
        mask = self.mask
        not_mask = np.logical_not(mask)
        # if there is nothing to do...
        if not np.any(mask):
            pass
        elif tof == 'interplin':
            inds_x = np.arange(self.shape[0])
            inds_y = np.arange(self.shape[1])
            grid_y, grid_x = np.meshgrid(inds_y, inds_x)
            Vx = self.comp_x
            Vy = self.comp_y
            fx = spinterp.interp2d(grid_x[not_mask],
                                   grid_y[not_mask],
                                   Vx[not_mask])
            fy = spinterp.interp2d(grid_x[not_mask],
                                   grid_y[not_mask],
                                   Vy[not_mask])
            for inds, masked in np.ndenumerate(mask):
                if masked:
                    Vx[inds[0], inds[1]] = fx(inds[0], inds[1])
                    Vy[inds[0], inds[1]] = fy(inds[0], inds[1])
            mask = np.zeros(Vx.shape)
            self.comp_x = Vx
            self.comp_y = Vy
            self.__mask = np.empty(self.shape, dtype=bool)
            self.__mask.fill(False)
        elif tof == 'interpcub':
            inds_x = np.arange(self.shape[0])
            inds_y = np.arange(self.shape[1])
            grid_y, grid_x = np.meshgrid(inds_y, inds_x)
            Vx = self.comp_x
            Vy = self.comp_y
            fx = spinterp.interp2d(grid_x[not_mask],
                                   grid_y[not_mask],
                                   Vx[not_mask],
                                   kind='cubic')
            fy = spinterp.interp2d(grid_x[not_mask],
                                   grid_y[not_mask],
                                   Vy[not_mask],
                                   kind='cubic')
            for inds, masked in np.ndenumerate(mask):
                if masked:
                    Vx[inds[0], inds[1]] = fx(inds[0], inds[1])
                    Vy[inds[0], inds[1]] = fy(inds[0], inds[1])
            mask = np.zeros(Vx.shape)
            self.comp_x = Vx
            self.comp_y = Vy
            self.__mask = np.empty(self.shape, dtype=bool)
            self.__mask.fill(False)
        elif tof == 'value':
            Vx = self.comp_x
            Vy = self.comp_y
            Vx[mask] = value
            Vy[mask] = value
            self.comp_x = Vx
            self.comp_y = Vy
            self.__mask = np.empty(self.shape, dtype=bool)
            self.__mask.fill(False)
        else:
            raise ValueError("unknown 'tof' value")

    def trim_area(self, intervalx=None, intervaly=None):
        """
        Return a trimed  area in respect with given intervals.

        Parameters
        ----------
        intervalx : array, optional
            interval wanted along x
        intervaly : array, optional
            interval wanted along y
        """
        indmin_x, indmax_x, indmin_y, indmax_y, trimfield = \
            Field.trim_area(self, intervalx, intervaly, full_output=True)
        trimfield.comp_x = self.comp_x[indmin_x:indmax_x + 1,
                                       indmin_y:indmax_y + 1]
        trimfield.comp_y = self.comp_y[indmin_x:indmax_x + 1,
                                       indmin_y:indmax_y + 1]
        return trimfield

    def crop_masked_border(self):
        """
        Crop the masked border of the field in place.
        """
        axe_x, axe_y = self.axe_x, self.axe_y
        # checking masked values presence
        Vx, Vy = self.comp_x, self.comp_y
        mask = self.mask
        if not np.any(mask):
            return None
        # crop border along y
        axe_y_m = np.logical_not(np.all(mask, axis=0))
        if not np.any(axe_y_m):
            Vx = Vx[:, axe_y_m]
            Vy = Vy[:, axe_y_m]
            mask = mask[:, axe_y_m]
        # crop values along x
        axe_x_m = np.logical_not(np.all(mask, axis=1))
        if np.any(axe_x_m):
            Vx = Vx[axe_x_m, :]
            Vy = Vy[axe_x_m, :]
            mask = mask[axe_x_m, :]
        # storing cropped values
        unit_x, unit_y = self.unit_x, self.unit_y
        unit_values = self.unit_values
        tmp_vf = self.__class__()
        VectorField.import_from_arrays(tmp_vf, axe_x, axe_y, Vx, Vy, mask=mask,
                                       unit_x=unit_x, unit_y=unit_y,
                                       unit_values=unit_values)
        self = tmp_vf

    def _display(self, component=None, kind=None, **plotargs):
        if kind is not None:
            if not isinstance(kind, STRINGTYPES):
                raise TypeError("'kind' must be a string")
        axe_x, axe_y = self.axe_x, self.axe_y
        if component is None:
            Vx = np.transpose(self.comp_x)
            Vy = np.transpose(self.comp_y)
            magn = np.transpose(self.magnitude)
            unit_x, unit_y = self.unit_x, self.unit_y
            if kind == 'stream':
                if not 'color' in plotargs.keys():
                    plotargs['color'] = magn
                displ = plt.streamplot(axe_x, axe_y, Vx, Vy, **plotargs)
            elif kind == 'quiver' or kind is None:
                if 'C' in plotargs.keys():
                    C = plotargs.pop('C')
                    if not (C == 0 or C is None):
                        displ = plt.quiver(axe_x, axe_y, Vx, Vy, C, **plotargs)
                    else:
                        displ = plt.quiver(axe_x, axe_y, Vx, Vy, **plotargs)
                else:
                    displ = plt.quiver(axe_x, axe_y, Vx, Vy, magn, **plotargs)
                ax = plt.gca()
                ax.set_aspect('equal')
            else:
                raise ValueError("Unknown value of 'kind'")
        elif component == "x":
            if kind == '3D':
                displ = self.comp_x_as_sf.Display3D()
            else:
                displ = self.comp_x_as_sf._display(kind)
        elif component == "y":
            if kind == '3D':
                displ = self.comp_y_as_sf.Display3D()
            else:
                displ = self.comp_y_as_sf._display(kind)
        elif component == "mask":
            if kind == '3D':
                displ = self.mask_as_sf.Display3D()
            else:
                displ = self.mask_as_sf._display(kind)
        else:
            raise TypeError("Unknown value of 'component'")
        return displ

    def display(self, component=None, kind=None, **plotargs):
        """
        Display something from the vector field.
        If component is not given, a quiver is displayed.
        If component is an integer, the coresponding component of the field is
        displayed.

        Parameters
        ----------
        component : string, optional
            Component to display, can be 'Vx', 'Vy', 'mask'
        kind : string, optinnal
            Scalar plots :
            if 'None': each datas are plotted (imshow),
            if 'contour': contours are ploted  (contour),
            if 'contourf': filled contours are ploted (contourf).
            Vector plots :
            if 'quiver': quiver plot,
            if 'stream': streamlines,
            if '3D': tri-dimensionnal plot.

        plotargs : dict
            Arguments passed to the function used to display the vector field.

        Returns
        -------
        fig : figure reference
            Reference to the displayed figure
        """
        displ = self._display(component, kind, **plotargs)
        unit_x, unit_y = self.axe_x, self.axe_y
        unit_values = self.unit_values
        Vx, Vy = self.comp_x, self.comp_y
        if component is None:
            if kind == 'quiver' or kind is None:
                cb = plt.colorbar()
                cb.set_label("Magnitude " + unit_values.strUnit())
                legendarrow = round(np.max([Vx.max(), Vy.max()]))
                plt.quiverkey(displ, 1.075, 1.075, legendarrow,
                              "$" + str(legendarrow)
                              + unit_values.strUnit() + "$",
                              labelpos='W', fontproperties={'weight': 'bold'})
            plt.title("Values " + unit_values.strUnit())
        elif component == 'x':
            plt.title("Vx " + unit_values.strUnit())
        elif component == 'y':
            plt.title("Vy " + unit_values.strUnit())
        elif component == 'mask':
            plt.title("Mask")
        else:
            raise ValueError("Unknown 'component' value")
        return displ

class VelocityFields(object):
    """
    Class representing a set of velocity fields. These fields can have
    differente positions along axes, or be successive view of the same area.
    It's recommended to use TemporalVelocityFields or SpatialVelocityFields
    insteas of this one.
    """

    def __init__(self):
        self.fields = []
        pass

    def __len__(self):
        return len(self.fields)

    def __iter__(self):
        return self.fields.__iter__()

    def __getitem__(self, fieldnumber):
        return self.fields[fieldnumber]

    def remove_field(self, fieldnumber):
        """
        Remove a field of the existing fields.

        Parameters
        ----------
        fieldnumber : integer
            The number of the velocity field to remove.
        """
        self.fields[fieldnumber:fieldnumber + 1] = []

    def add_field(self, velocityfield):
        """
        Add a field to the existing fields.

        Parameters
        ----------
        velocityfield : VelocityField object
            The velocity field to add.
        """
        if not isinstance(velocityfield, VelocityField):
            raise TypeError("'velocityfield' must be a VelocityField object")
        self.fields.append(velocityfield.copy())

    def get_field(self, fieldnumber):
        """
        Return the velocity field referenced by the given fieldnumber.

        Parameters
        ----------
        fieldnumber : integer
            Reference to the wanted field number.

        Returns
        -------
        field : VelocityField object
            The wanted velocity field.
        """
        field = self.fields[fieldnumber]
        return field

    def copy(self):
        """
        Return a copy of the velocityfields
        """
        return copy.deepcopy(self)

    def set_origin(self, x=None, y=None):
        """
        Modify the axis in order to place the origin at the actual point (x, y)

        Parameters
        ----------
        x : number
        y : number
        """
        if x is not None:
            if not isinstance(x, NUMBERTYPES):
                raise TypeError("'x' must be a number")
            for field in self.fields:
                field.set_origin(x, None)
        if y is not None:
            if not isinstance(y, NUMBERTYPES):
                raise TypeError("'y' must be a number")
            for field in self.fields:
                field.set_origin(None, y)


class TemporalVelocityFields(VelocityFields, Field):
    """
    Class representing a set of time-evolving velocity fields.

    Principal methods
    -----------------
    "add_field" : add a velocity field.

    "remove_field" : remove a field.

    "display" : display the vector field, with these unities.

    "display_animate" : display an animation of a component of the velocity
    fields set.

    "calc_*" : give access to a bunch of derived statistical fields.
    """

    def __add__(self, other):
        if isinstance(other, TemporalVelocityFields):
            tmp_tvfs = self.copy()
            tmp_tvfs._clear_derived()
            tmp_tvfs.fields += other.fields
            return tmp_tvfs
        else:
            raise TypeError("cannot concatenate temporal velocity fields with"
                            " {}.".format(type(other)))

    def __mul__(self, other):
        if isinstance(other, (NUMBERTYPES, unum.Unum)):
            final_vfs = TemporalVelocityFields()
            for field in self.fields:
                final_vfs.add_field(field*other)
            return final_vfs
        else:
            raise TypeError("You can only multiply a temporal velocity field "
                            "by numbers")

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, (NUMBERTYPES, unum.Unum)):
            final_vfs = TemporalVelocityFields()
            for field in self.fields:
                final_vfs.add_field(field/other)
            return final_vfs
        else:
            raise TypeError("You can only divide a temporal velocity field "
                            "by numbers")

    __div__ = __truediv__

    def __pow__(self, number):
        if not isinstance(number, NUMBERTYPES):
            raise TypeError("You only can use a number for the power "
                            "on a Vectorfield")
        final_vfs = TemporalVelocityFields()
        for field in self.fields:
            final_vfs.add_field(np.power(field, number))
        return final_vfs

    def _clear_derived(self):
        """
        Delete all the derived fields, in case of changement in the base
        fields.
        """
        derived = ["mean_vf", "turbulent_vf", "mean_kinetic_energy",
                   "tke", "mean_tke", "rs_xx", "rs_xy", "rs_yy"]
        for attr in derived:
            try:
                del self.__dict__[attr]
            except KeyError:
                pass

    @Field.axe_x.setter
    def axe_x(self, value):
        Field.axe_x = value
        for field in self.fields:
            field.axe_x = value

    @Field.axe_y.setter
    def axe_y(self, value):
        Field.axe_y = value
        for field in self.fields:
            field.axe_y = value

    @Field.unit_x.setter
    def unit_x(self, value):
        Field.unit_x = value
        for field in self.fields:
            field.unit_x = value

    @Field.unit_y.setter
    def unit_y(self, value):
        Field.unit_y = value
        for field in self.fields:
            field.unit_y = value

    def set_origin(self, ):
        # continuer à surcharger les méthode de Field nécessaires

    def add_field(self, velocityfield):
        """
        Add a field to the existing fields.

        Parameters
        ----------
        velocityfield : VelocityField object
            The velocity field to add.
        """
        # delete derived fields because base fields are modified
        if not len(self.fields) == 0:
            axes = self.fields[0].get_axes()
            vaxes = velocityfield.get_axes()
            if not all(axes[0] == vaxes[0]) and all(axes[1] == vaxes[1]):
                raise ValueError("Axes of the new field must be consistent "
                                 "with current axes")
        VelocityFields.add_field(self, velocityfield)
        self._clear_derived()

#    def get_comp(self, componentname, raw=False, masked=True):
#        """
#        Return a reference to the field designed by 'fieldname'.
#
#        Parameters
#        ----------
#        componentname : string
#            Name of the component.
#        raw : boolean, optional
#            .
#        masked : boolean, optional
#            .
#
#        Returns
#        -------
#        component : VelocityField, VectorField or ScalarField object or array
#        of these objects
#            Reference to the field.
#        """
#        values = None
#        if not isinstance(componentname, str):
#            raise TypeError("'componentname' must be a string")
#        # Temporal Velocity Field attributes
#        elif componentname == "fields":
#            values = copy.deepcopy(self.fields)
#        elif componentname == "mean_vf":
#            try:
#                values = self.mean_vf.copy()
#            except AttributeError:
#                self.calc_mean_vf()
#                values = self.mean_vf.copy()
#        elif componentname == "turbulent_vf":
#            try:
#                values = self.turbulent_vf.copy()
#            except AttributeError:
#                self.calc_turbulent_vf()
#                values = self.turbulent_vf.copy()
#        elif componentname == "mean_kinetic_energy":
#            try:
#                values = self.mean_kinetic_energy.copy()
#            except AttributeError:
#                self.calc_mean_kinetic_energy()
#                values = self.mean_kinetic_energy.copy()
#        elif componentname == "turbulent_kinetic_energy":
#            try:
#                values = self.turbulent_kinetic_energy.copy()
#            except AttributeError:
#                self.calc_turbulent_kinetic_energy()
#                values = self.turbulent_kinetic_energy.copy()
#        elif componentname == "rs_xx":
#            try:
#                values = self.rs_xx.copy()
#            except AttributeError:
#                self.calc_reynolds_stress()
#                values = self.rs_xx.copy()
#        elif componentname == "rs_yy":
#            try:
#                values = self.rs_yy.copy()
#            except AttributeError:
#                self.calc_reynolds_stress()
#                values = self.rs_yy.copy()
#        elif componentname == "rs_xy":
#            try:
#                values = self.rs_xy.copy()
#            except AttributeError:
#                self.calc_reynolds_stress()
#                values = self.rs_xy.copy()
#        elif componentname == "tke":
#            try:
#                values = self.tke.copy()
#            except AttributeError:
#                self.calc_tke()
#                values = copy.deepcopy(self.tke)
#        elif componentname == "mean_tke":
#            try:
#                values = self.mean_tke.copy()
#            except AttributeError:
#                self.calc_mean_tke()
#                values = self.mean_tke.copy()
#        # values treatment
#        if values is not None:
#            if not raw:
#                return values
#            elif isinstance(values, ScalarField):
#                return values.get_comp('values', raw=raw, masked=masked)
#            elif isinstance(values, VectorField):
#                return values.get_comp('V', raw=raw, masked=masked)
#            elif isinstance(values[0], ScalarField):
#                return [sf.get_comp('values', raw=raw, masked=masked)
#                        for sf in values]
#            elif isinstance(values[0], VectorField):
#                return [vf.get_comp('V', raw=raw, masked=masked)
#                        for vf in values]
#            else:
#                raise StandardError()
#        # Velocity Field attributes
#        elif len(self.fields) != 0:
#            try:
#                self.fields[0].get_comp(componentname, raw=raw)
#            except ValueError:
#                pass
#            else:
#                tmp_fields = np.zeros((len(self.fields),), dtype=np.ndarray)
#                for i, field in enumerate(self.fields):
#                    tmp_fields[i] = field.get_comp(componentname, raw=raw,
#                                                   masked=masked)
#                return np.array(tmp_fields)
#        raise ValueError("Unknown component : {}".format(componentname))


    def get_axes(self):
        """)
        Return fields axis
        """
        return self[0].get_axes()

    def get_axe_units(self):
        """
        Return fields axis unities
        """
        return self[0].get_axe_units()

    def get_indice_on_axe(self, direction, value, nearest=False):
        """
        Return, on the given axe, the indices representing the positions
        surrounding 'value'.
        if 'value' is exactly an axe position, return just one indice.

        Parameters
        ----------
        direction : int
            1 or 2, for axes choice.
        value : number
        nearest : boolean
            If 'True', only the nearest indice is returned.

        Returns
        -------
        interval : 2x1 or 1x1 array of integer
        """
        return self.fields[0].get_indice_on_axe(direction, value, nearest)

    def get_dim(self):
        """
        Return the fields dimension.
        """
        if len(self.fields) == 0:
            return (0,)
        return self.fields[0].get_dim()

    def copy(self):
        """
        Return a copy of the velocityfields
        """
        return copy.deepcopy(self)

    def get_time_profile(self, component, x, y):
        """
        Return a profile contening the time evolution of the given component.

        Parameters
        ----------
        component : string
        x, y : numbers
            Wanted position for the time profile, in axis units.

        Returns
        -------
        profile : Profile object

        """
        # check parameters coherence
        if not isinstance(component, STRINGTYPES):
            raise TypeError("'component' must be a string")
        if not isinstance(x, NUMBERTYPES) or not isinstance(y, NUMBERTYPES):
            raise TypeError("'x' and 'y' must be numbers")
        axe_x, axe_y = self.get_axes()
        if x < np.min(axe_x) or x > np.max(axe_x)\
                or y < np.min(axe_y) or y > np.max(axe_y):
            raise ValueError("'x' ans 'y' values out of bounds")
        compo_init = self[0].get_comp(component)
        compos = self.get_comp(component, raw=True)
        # if the given object is a ScalarField
        if isinstance(compo_init, ScalarField):
            time = self.get_comp('time')
            unit_time = self[0].unit_time
            values = np.zeros(len(compos))
            mask = np.zeros(len(compos))
            unit_values = self[0].unit_values
            # getting position indices
            ind_x = compo_init.get_indice_on_axe(1, x, nearest=True)
            ind_y = compo_init.get_indice_on_axe(2, y, nearest=True)
            for i in np.arange(len(compos)):
                values[i] = compos[i].data[ind_y, ind_x]
                mask[i] = compos[i].mask[ind_y, ind_x]
            values = np.ma.masked_array(values, mask)
            return Profile(time, values, unit_x=unit_time, unit_y=unit_values)
        else:
            raise ValueError("Unvalid component for a time profile")

    def get_spectrum(self, component, pt, ind=False, zero_fill=False,
                     interp=False, raw_spec=False, mask_error=True):
        """
        Return a Profile object, with the frequential spectrum of 'component',
        on the point 'pt'.

        Parameters
        ----------
        component : string
        pt : 2x1 array of numbers
        ind : boolean
            If true, 'pt' is read as indices,
            else, 'pt' is read as coordinates.
        zero_fill : boolean
            If True, field masked values are filled by zeros.
        interp : boolean
            If 'True', linear interpolation is used to get component between
            grid points, else, the nearest point is choosen.
        raw_spec: boolean
            If 'False' (default), returned spectrum is
            'abs(raw_spec)/length_signal' in order to have coherent ordinate
            axis.
            Else, raw spectrum is returned.
        mask_error : boolean
            If 'False', instead of raising an error when masked value appear on
            time profile, '(None, None)' is returned.

        Returns
        -------
        magn_prof : Profile object
            Magnitude spectrum.
        phase_prof : Profile object
            Phase spectrum
        """
        # checking parameters coherence
        if not isinstance(pt, ARRAYTYPES):
            raise TypeError("'pt' must be a 2x1 array")
        if ind:
            pt = np.array(pt, dtype=int)
        else:
            pt = np.array(pt, dtype=float)
        if not pt.shape == (2,):
            raise ValueError("'pt' must be a 2x1 array")
        if ind and (not isinstance(pt[0], int) or not isinstance(pt[1], int)):
            raise TypeError("If 'ind' is True, 'pt' must be an array of two"
                            " integers")
        if not isinstance(ind, bool):
            raise TypeError("'ind' must be a boolean")
        if not isinstance(interp, bool):
            raise TypeError("'interp' must be a boolean")
        if not isinstance(raw_spec, bool):
            raise TypeError("'raw_spec' must be a boolean")
        x = pt[0]
        y = pt[1]
        axe_x, axe_y = self.get_axes()
        comp = self.get_comp(component)
        if isinstance(comp[0], ScalarField):
            # getting indices points
            ind_x = None
            ind_y = None
            if ind:
                ind_x = int(x)
                ind_y = int(y)
            elif not interp:
                inds_x = self.fields[0].get_indice_on_axe(1, pt[0])
                if len(inds_x) == 1:
                    ind_x = inds_x[0]
                else:
                    vals = [axe_x[inds_x[0]], axe_x[inds_x[1]]] - x
                    ind_x = inds_x[np.argmin(vals)]
                inds_y = self.fields[0].get_indice_on_axe(2, pt[1])
                if len(inds_y) == 1:
                    ind_y = inds_y[0]
                else:
                    vals = [axe_y[inds_y[0]], axe_y[inds_y[1]]] - y
                    ind_y = inds_y[np.argmin(vals)]
            # getting temporal evolution
            time = []
            values = []
            for i in np.arange(len(comp)):
                sf = comp[i]
                time.append(self[i].time)
                if interp:
                    values.append(sf.get_value(x, y, ind=ind))
                else:
                    try:
                        values.append(sf.get_value(ind_y, ind_x, ind=True))
                    except:
                        pdb.set_trace()
            # checking if position is masked
            for i, val in enumerate(values):
                if np.ma.is_masked(val):
                    if zero_fill:
                        values[i] = 0
                    else:
                        if mask_error:
                            raise StandardError("Masked values on time "
                                                "profile")
                        else:
                            return None, None

            # getting spectrum
            n = len(values)
            Y = np.fft.rfft(values)
            if raw_spec:
                magn = np.abs(Y)
            else:
                magn = np.abs(Y)/(n/2)
            phase = np.angle(Y)
            # getting frequencies
            Ts = time[1] - time[0]
            frq = np.fft.rfftfreq(n, Ts)
        else:
            raise ValueError("Not implemented yet on {}".format(type(comp[0])))
        magn_prof = Profile(frq, magn, unit_x=make_unit('Hz'),
                            unit_y=comp[0].get_comp('unit_values'))
        phase_prof = Profile(frq, phase, unit_x=make_unit('Hz'),
                             unit_y=make_unit('rad'))
        return magn_prof, phase_prof

    def get_spectrum_over_area(self, component, intervalx, intervaly,
                               ind=False, zero_fill=False, interp=False,
                               raw_spec=False):
        """
        Return a Profile object, contening a mean spectrum of the given
        component, on all the points included in the given intervals.

        Parameters
        ----------
        component : string
            Scalar component ('Vx', 'Vy', 'magnitude', ...).
        intervalx, intervaly : 2x1 arrays of numbers
            Defining the square on which averaging the spectrum.
            (in axes values)
        ind : boolean
            If true, 'pt' is read as indices,
            else, 'pt' is read as coordinates.
        zero_fill : boolean
            If True, field masked values are filled by zeros.
        interp : boolean
            If 'True', linear interpolation is used to get component between
            grid points, else, the nearest point is choosen.
        raw_spec: boolean
            If 'False' (default), returned spectrum is
            'abs(raw_spec)/length_signal' in order to have coherent ordinate
            axis.
            Else, raw spectrum is returned.

        Returns
        -------
        magn_prof : Profile object
            Averaged magnitude spectrum.
        phase_prof : Profile object
            Averaged phase spectrum
        """
        # checking parameters coherence
        if not isinstance(component, STRINGTYPES):
            raise TypeError("'component' must be a string")
        if not isinstance(intervalx, ARRAYTYPES):
            raise TypeError("'intervalx' must be an array")
        if not isinstance(intervaly, ARRAYTYPES):
            raise TypeError("'intervaly' must be an array")
        if not isinstance(intervalx[0], NUMBERTYPES):
            raise TypeError("'intervalx' must be an array of numbers")
        if not isinstance(intervaly[0], NUMBERTYPES):
            raise TypeError("'intervaly' must be an array of numbers")
        axe_x, axe_y = self.get_axes()
        # checking interval values
        if ind:
            if not isinstance(intervalx[0], int)\
                    or not isinstance(intervalx[1], int)\
                    or not isinstance(intervaly[0], int)\
                    or not isinstance(intervaly[1], int):
                raise TypeError("'intervalx' and 'intervaly' must be arrays of"
                                " integer if 'ind' is 'True'")
            if intervalx[0] < 0 or intervaly[0] < 0\
                    or intervalx[-1] >= len(axe_x)\
                    or intervaly[-1] >= len(axe_y):
                raise ValueError("intervals are out of bounds")
        else:
            axe_x_min = np.min(axe_x)
            axe_x_max = np.max(axe_x)
            axe_y_min = np.min(axe_y)
            axe_y_max = np.max(axe_y)
            if np.min(intervalx) < axe_x_min\
                    or np.max(intervalx) > axe_x_max\
                    or np.min(intervaly) < axe_y_min\
                    or np.max(intervaly) > axe_y_max:
                raise ValueError("intervals are out of bounds")
        # Getting indices bounds
        if not ind:
            ind_x_min = self.get_indice_on_axe(1, intervalx[0])[0]
            ind_x_max = self.get_indice_on_axe(1, intervalx[1])[-1]
            ind_y_min = self.get_indice_on_axe(2, intervaly[0])[0]
            ind_y_max = self.get_indice_on_axe(2, intervaly[1])[-1]
        else:
            ind_x_min = intervalx[0]
            ind_x_max = intervalx[1]
            ind_y_min = intervaly[0]
            ind_y_max = intervaly[1]
        # Averaging ponctual spectrums
        magn = 0.
        phase = 0.
        nmb_fields = (ind_x_max - ind_x_min + 1)*(ind_y_max - ind_y_min + 1)
        real_nmb_fields = nmb_fields
        for i in np.arange(ind_x_min, ind_x_max + 1):
            for j in np.arange(ind_y_min, ind_y_max + 1):
                tmp_m, tmp_p = self.get_spectrum(component, [i, j], ind=True,
                                                 zero_fill=zero_fill,
                                                 interp=interp,
                                                 raw_spec=raw_spec,
                                                 mask_error=False)
                # check if the position is masked
                if tmp_m is None:
                    real_nmb_fields -= 1
                else:
                    magn = magn + tmp_m
                    phase = phase + tmp_p
        if real_nmb_fields == 0:
            raise StandardError("I can't find a single non-masked time profile"
                                ", maybe you will want to try 'zero_fill' "
                                "option")
        magn = magn/real_nmb_fields
        phase = phase/real_nmb_fields
        return magn, phase

    def calc_mean_vf(self, nmb_min=1):
        """
        Calculate the mean velocity field, from all the fields.

        Parameters
        ----------
        nmb_min : integer, optional
            Minimum number of values used to make a mean. else, the value is
            masked
        """
        if len(self.fields) == 0:
            raise ValueError("There is no fields in this object")
        nmbvalues = np.zeros(self.get_dim())
        mean_vx = np.zeros(self.get_dim())
        mean_vy = np.zeros(self.get_dim())
        mask_tot = np.zeros(self.get_dim())
        time = 0
        for field in self.fields:
            mask = field.get_comp('mask', raw=True)
            values_x = field.get_comp('Vx', raw=True, masked=False)
            values_y = field.get_comp('Vy', raw=True, masked=False)
            values_x[mask] = 0
            values_y[mask] = 0
            nmbvalues[np.logical_not(mask)] += 1
            mean_vx += values_x
            mean_vy += values_y
            time += field.time
        mask_tot[nmbvalues < nmb_min] = True
        nmbvalues[nmbvalues == 0] = 1
        mean_vx = mean_vx / nmbvalues
        mean_vy = mean_vy / nmbvalues
        time = time/len(self.fields)
        #stockage
        values_x = np.ma.masked_array(mean_vx, mask_tot)
        values_y = np.ma.masked_array(mean_vy, mask_tot)
        axe_x, axe_y = self.get_axes()
        unit_x, unit_y = self.get_axe_units()
        unit_time = self.fields[0].get_comp('unit_time')
        unit_values = self.fields[0].get_comp('unit_values')
        time = time
        mean_vf = VelocityField()
        mean_vf.import_from_arrays(axe_x, axe_y, values_x, values_y, time,
                                   unit_x, unit_y, unit_values, unit_time)
        if np.all(mask_tot):
            raise Warning("All datas masked in this mean field."
                          "Try a smaller 'nmb_min'")
        self.mean_vf = mean_vf

    def calc_turbulent_vf(self):
        """
        Calculate the turbulent fields (instantaneous fields minus mean field)
        """
        self.turbulent_vf = TemporalVelocityFields()
        mean_vf = self.get_comp('mean_vf')
        for field in self.fields:
            tmp_field = field - mean_vf
            tmp_field.time = field.time
            self.turbulent_vf.add_field(tmp_field)

    def calc_mean_kinetic_energy(self):
        """
        Calculate the mean kinetic energy.
        """
        final_sf = ScalarField()
        mean_vf = self.get_comp('mean_vf')
        values_x = mean_vf.get_comp('Vx')
        values_y = mean_vf.get_comp('Vy')
        final_sf = 1./2*(values_x**2 + values_y**2)
        self.mean_kinetic_energy = final_sf

    def calc_tke(self):
        """
        Calculate the turbulent kinetic energy.
        """
        turb_vfs = self.get_comp('turbulent_vf')
        vx_p = turb_vfs.get_comp('Vx')
        vy_p = turb_vfs.get_comp('Vy')
        self.tke = []
        for i in np.arange(len(vx_p)):
            self.tke.append(1./2*(vx_p[i]**2 + vy_p[i]**2))

    def calc_mean_tke(self):
        self.mean_tke = self.get_comp('tke')[0]
        values = self.tke[0].values
        for field in self.tke:
            values += field.values
        values /= len(self.tke)
        self.mean_tke.values = values

    def calc_reynolds_stress(self, nmb_val_min=1):
        """
        Calculate the reynolds stress.
        """
        # getting fluctuating velocities
        turb_vf = self.get_comp('turbulent_vf')
        u_p = turb_vf.get_comp('Vx')
        v_p = turb_vf.get_comp('Vy')
        # rs_xx
        rs_xx = np.zeros(u_p[0].get_dim())
        mask_rs_xx = np.zeros(u_p[0].get_dim())
        # boucle sur les points du champ
        for i in np.arange(rs_xx.shape[0]):
            for j in np.arange(rs_xx.shape[1]):
                # boucle sur le nombre de champs
                nmb_val = 0
                for n in np.arange(len(turb_vf.fields)):
                    # check if masked
                    if not u_p[n].values.mask[i, j]:
                        rs_xx[i, j] += u_p[n].get_value(j, i, ind=True)**2
                        nmb_val += 1
                if nmb_val > nmb_val_min:
                    rs_xx[i, j] /= nmb_val
                else:
                    rs_xx[i, j] = 0
                    mask_rs_xx[i, j] = True
        # rs_yy
        rs_yy = np.zeros(v_p[0].values.shape)
        mask_rs_yy = np.zeros(v_p[0].values.shape)
        # boucle sur les points du champ
        for i in np.arange(rs_yy.shape[0]):
            for j in np.arange(rs_yy.shape[1]):
                # boucle sur le nombre de champs
                nmb_val = 0
                for n in np.arange(len(turb_vf.fields)):
                    # check if masked
                    if not v_p[n].values.mask[i, j]:
                        rs_yy[i, j] += v_p[n].get_value(j, i, ind=True)**2
                        nmb_val += 1
                if nmb_val > nmb_val_min:
                    rs_yy[i, j] /= nmb_val
                else:
                    rs_yy[i, j] = 0
                    mask_rs_yy[i, j] = True
        # rs_xy
        rs_xy = np.zeros(u_p[0].get_dim())
        mask_rs_xy = np.zeros(u_p[0].get_dim())
        # boucle sur les points du champ
        for i in np.arange(rs_xy.shape[0]):
            for j in np.arange(rs_xy.shape[1]):
                # boucle sur le nombre de champs
                nmb_val = 0
                for n in np.arange(len(turb_vf.fields)):
                    # check if masked
                    if not (u_p[n].get_comp('mask', raw=True)[i, j]
                            or v_p[n].get_comp('mask', raw=True)[i, j]):
                        rs_xy[i, j] += (u_p[n].get_value(j, i, ind=True)
                                        * v_p[n].get_value(j, i, ind=True))
                        nmb_val += 1
                if nmb_val > nmb_val_min:
                    rs_xy[i, j] /= nmb_val
                else:
                    rs_xy[i, j] = 0
                    mask_rs_xy[i, j] = True
        # masking and storing
        axe_x, axe_y = self.fields[0].get_axes()
        unit_x, unit_y = self.fields[0].get_axe_units()
        unit_values = self.fields[0].get_comp('unit_values')
        self.rs_xx = ScalarField()
        rs_xx = np.ma.masked_array(rs_xx, mask_rs_xx)
        self.rs_xx.import_from_arrays(axe_x, axe_y, rs_xx,
                                      unit_x, unit_y, unit_values)
        self.rs_yy = ScalarField()
        rs_yy = np.ma.masked_array(rs_yy, mask_rs_yy)
        self.rs_yy.import_from_arrays(axe_x, axe_y, rs_yy,
                                      unit_x, unit_y, unit_values)
        self.rs_xy = ScalarField()
        rs_xy = np.ma.masked_array(rs_xy, mask_rs_xy)
        self.rs_xy.import_from_arrays(axe_x, axe_y, rs_xy,
                                      unit_x, unit_y, unit_values)

#    def calc_detachment_positions(self, wall_direction=2, wall_position=None,
#                                  interval=None):
#        """
#        Return a Profile object of the temporal evolution of the detachment
#        point, using the 'calc_detachment_position' fonction of
#        VelocityField objects.
#
#        Parameters
#        ----------
#        wall_direction : integer, optional
#            1 for a wall at a given value of x,
#            2 for a wall at a given value of y (default).
#        wall_position : number, optional
#            Position of the wall. The default position is the minimum value
#            on the axe.
#        interval : 2x1 array of numbers, optional
#            Optional interval in which search for the detachment points.
#
#        """
#        x = np.zeros(len(self.fields))
#        time = np.zeros(len(self.fields))
#        i = 0
#        for field in self.fields:
#            x[i] = field.calc_detachment_position(wall_direction=
#                                                  wall_direction,
#                                                  wall_position=wall_position,
#                                                  interval=interval)
#            time[i] = field.time
#            i += 1
#        if wall_direction == 1:
#            unit_x = self.fields[0].V.comp_x.unit_y
#        else:
#            unit_x = self.fields[0].V.comp_x.unit_x
#        return Profile(time, x, self.fields[0].unit_time, unit_x)

    def reduce_temporal_resolution(self, nmb_in_interval, mean=True):
        """
        Return a TemporalVelocityFields, contening one field for each
        'nmb_in_interval' field in the initial TFVS.

        Parameters
        ----------
        nmb_in_interval : integer
            Length of the interval.
            (one field is kept for each 'nmb_in_interval fields)
        mean : boolean, optional
            If 'True', the resulting fields are average over the interval.
            Else, fields are taken directly.

        Returns
        -------
        TVFS : TemporalVelocityFields
        """
        if not isinstance(nmb_in_interval, int):
            raise TypeError("'nmb_in_interval' must be an integer")
        if nmb_in_interval == 1:
            return self.copy()
        if nmb_in_interval >= len(self):
            raise ValueError("'nmb_in_interval' is too big")
        if not isinstance(mean, bool):
            raise TypeError("'mean' must be a boolean")
        tmp_TVFS = TemporalVelocityFields()
        i = 0
        while True:
            tmp_vf = self[i]
            time = self[i].time
            if mean:
                for j in np.arange(i + 1, i + nmb_in_interval):
                    tmp_vf += self[j]
                    time += self[j].time
                tmp_vf /= nmb_in_interval
                time /= nmb_in_interval
                tmp_vf.time = time
            tmp_TVFS.add_field(tmp_vf)
            i += nmb_in_interval
            if i + nmb_in_interval > len(self):
                break
        return tmp_TVFS

    def fill(self, kind='temporal', tof='interplin', value=[0., 0.],
             crop_border=True):
        """
        Fill the masked part of the array in place.

        Parameters
        ----------
        kind : string
            Can be 'temporal' for temporal interpolation, or 'spatial' for
            spatial interpolation.
        tof : string, optional
            Type of algorithm used to fill.
            'value' : fill with a given value
            'interplin' : fill using linear interpolation
            'interpcub' : fill using cubic interpolation
        value : 2x1 array
            Value for filling, '[Vx, Vy]' (only usefull with tof='value')
        crop_border : boolean
            If 'True' (default), masked borders of the field are cropped
            before filling. Else, values on border are extrapolated (poorly).
        """
        # checking parameters coherence
        if len(self.fields) < 3 and kind == 'temporal':
            raise ValueError("Not enough fields to fill with temporal"
                             " interpolation")
        # cropping masked borders if necessary
        if crop_border:
            self.crop_masked_border()
        # temporal interpolation
        if kind == 'temporal':
            # getting datas
            axe_x, axe_y = self.get_axes()
            # getting super mask (0 where no value are masked and where all
            # values are masked)
            super_mask = self.get_comp('mask', raw=True)
            super_mask = np.sum(super_mask, axis=0)
            super_mask[super_mask == len(self.fields)] = 0
            # loop on each field position
            for j, i in np.argwhere(super_mask):
                prof_x = self.get_time_profile('Vx', axe_x[i], axe_y[j])
                prof_y = self.get_time_profile('Vy', axe_x[i], axe_y[j])
                # checking if all time profile value are masked
                if np.all(np.logical_or(prof_x.y.mask, prof_x.y.mask)):
                    continue
                # getting masked position on profile
                inds_masked_x = np.where(prof_x.y.mask)
                inds_masked_y = np.where(prof_y.y.mask)
                # creating interpolation function
                if tof == 'value':
                    def interp_x(x):
                        return value[0]
                    def interp_y(x):
                        return value[1]
                elif tof == 'interplin':
                    interp_x = spinterp.interp1d(prof_x.x[~prof_x.y.mask],
                                                 prof_x.y[~prof_x.y.mask],
                                                 kind='linear')
                    interp_y = spinterp.interp1d(prof_y.x[~prof_y.y.mask],
                                                 prof_y.y[~prof_y.y.mask],
                                                 kind='linear')
                elif tof == 'interpcub':
                    interp_x = spinterp.interp1d(prof_x.x[~prof_x.y.mask],
                                                 prof_x.y[~prof_x.y.mask],
                                                 kind='cubic')
                    interp_y = spinterp.interp1d(prof_y.x[~prof_y.y.mask],
                                                 prof_y.y[~prof_y.y.mask],
                                                 kind='cubic')
                # loop on all x profile masked points
                for ind_masked in inds_masked_x[0]:
                    try:
                        interp_val = interp_x(prof_x.x[ind_masked])
                    except ValueError:
                        continue
                    # putting interpolated value in the field
                    self[ind_masked].comp_x[j, i] = interp_val
                # loop on all y profile masked points
                for ind_masked in inds_masked_y[0]:
                    try:
                        interp_val = interp_y(prof_y.x[ind_masked])
                    except ValueError:
                        continue
                    # putting interpolated value in the field
                    self[ind_masked].comp_y[j, i] = interp_val
        # spatial interpolation
        elif kind == 'spatial':
            for field in self.fields:
                field.fill(tof=tof, value=value, crop_border=True)

        else:
            raise ValueError("Unknown parameter for 'kind' : {}".format(kind))

    def trim_area(self, intervalx=None, intervaly=None):
        """
        Trim the area and the axes in respect with given intervals.

        Parameters
        ----------
        intervalx : array, optional
            interval wanted along x axe.
        intervaly : array, optional
            interval wanted along y axe.
        """
        tmp_vfs = TemporalVelocityFields()
        for field in self.fields:
            tmp_field = field.trim_area(intervalx, intervaly)
            tmp_vfs.add_field(tmp_field)
        return tmp_vfs

    def crop_masked_border(self):
        """
        Crop the masked border of the velocity fields in place.
        """
        # getting big mask (where all the value are masked)
        masks_temp = self.get_comp('mask', raw=True)
        mask_temp = np.sum(masks_temp)
        mask_temp = mask_temp == len(masks_temp)
        # checking masked values presence
        if not np.any(mask_temp):
            return None
        # getting positions to remove (column or line with only masked values)
        axe_y_m = ~np.all(mask_temp, axis=1)
        axe_x_m = ~np.all(mask_temp, axis=0)
        # skip if nothing to do
        if not np.any(axe_y_m) or not np.any(axe_x_m):
            return None
        # else loop on fields
        for i in np.arange(len(self.fields)):
            #getting datas for one field
            values_x = self.fields[i].get_comp('Vx', raw=True, masked=False)
            values_y = self.fields[i].get_comp('Vy', raw=True, masked=False)
            mask = self.fields[i].get_comp('mask', raw=True)
            # crop values along y
            if np.any(axe_y_m):
                # deleting useless part along y
                values_x = values_x[axe_y_m, :]
                values_y = values_y[axe_y_m, :]
                mask = mask[axe_y_m, :]
            # crop values along x
            if np.any(axe_y_m):
                # deleting useless part along y
                values_x = values_x[:, axe_x_m]
                values_y = values_y[:, axe_x_m]
                mask = mask[:, axe_x_m]
            # storing croped values
            self.fields[i].set_comp('Vx', np.ma.masked_array(values_x, mask))
            self.fields[i].set_comp('Vy', np.ma.masked_array(values_y, mask))
            self.fields[i]._clear_derived()
            #crop axis
            axe_x, axe_y = self.get_axes()
            new_axe_x = axe_x[axe_x_m]
            new_axe_y = axe_y[axe_y_m]
            self.fields[i].set_axes(new_axe_x, new_axe_y)

    def display(self, fieldname="fields", **plotargs):
        """
        Display a component of the velocity fields.
        If 'component" is a component name, the coresponding component of the
        field is displayed. Else, a quiver is displayed.

        Parameters
        ----------
        fieldname : string, optional
            Fields to display ('fields', 'mean_vf', 'turbulent_vf')
        plotargs : dict, optional
            Arguments passed to the function used to display the vector field.
        """
        fields = self.get_comp(fieldname)
        if isinstance(fields, ARRAYTYPES):
            nmbfields = len(fields)
            colnmb = round(np.sqrt(nmbfields))
            if len(fields) % colnmb == 0:
                linenmb = nmbfields/colnmb
            else:
                linenmb = int(len(fields)/colnmb + 1)
            i = 1
            for field in fields:
                displ = plt.subplot(linenmb, colnmb, i)
                field.display(**plotargs)
                plt.title(fieldname + " (field number " + str(i-1) +
                          "), at t=" + str(self[i-1].time*self[i-1].unit_time))
                i += 1
            plt.suptitle(fieldname, fontsize=18)
        elif isinstance(fields, (VelocityField, VectorField)):
            displ = fields.display(**plotargs)
            plt.title(fieldname)
        else:
            displ = fields.display(**plotargs)
            plt.title(fieldname)
        return displ

    def display_animate(self, compo='V', interval=500, repeat=True,
                        **plotargs):
        """
        Display fields animated in time.

        Parameters
        ----------
        compo : string
            Composante to display
        interval : number, optionnal
            interval between two frames in milliseconds.
        repeat : boolean, optional
            if True, the animation is repeated infinitely.
        additional arguments can be passed (scale, vmin, vmax,...)
        """
        from matplotlib import animation
        comp = self.get_comp(compo)
        if 'kind' in plotargs.keys():
            kind = plotargs['kind']
        else:
            kind = None
        try:
            comp[0]
        except TypeError:
            raise ValueError("Cannot display an animation on one field")
        # display a vector field (quiver)
        if isinstance(comp[0], VectorField)\
                and (kind is None or kind == "quiver"):
            fig = plt.figure()
            ax = plt.gca()
            displ = comp[0].display(**plotargs)
            ttl = plt.title('')

            def update(num, fig, ax, displ, ttl, comp):
                vx = comp[num].get_comp('Vx', raw=True)
                vy = comp[num].get_comp('Vy', raw=True)
                magn = comp[num].get_magnitude().get_comp('values', raw=True)
                displ.set_UVC(vx, vy, magn)
                title = "{}, at t={:.3} {}"\
                    .format(compo, float(self[num].time),
                            self[num].unit_time.strUnit())
                ttl.set_text(title)
                return ax
            anim = animation.FuncAnimation(fig, update,
                                           frames=len(comp),
                                           interval=interval, blit=False,
                                           repeat=repeat,
                                           fargs=(fig, ax, displ, ttl, comp))
            return anim,
        # display a scalar field (contour, contourf or imshow) or a streamplot
        elif isinstance(comp[0], ScalarField)\
                or isinstance(comp[0], VectorField):
            fig = plt.figure()
            ax = plt.gca()
            displ = comp[0].display(**plotargs)
            ttl = plt.suptitle('')

            def update(num, fig, ax, displ, ttl, comp):
                ax.cla()
                displ = comp[num]._display(**plotargs)
                title = "{}, at t={:.3} {}"\
                    .format(compo, float(self[num].time),
                            self[num].unit_time.strUnit())
                ttl.set_text(title)
                return displ,
            anim = animation.FuncAnimation(fig, update,
                                           frames=len(comp),
                                           interval=interval, blit=False,
                                           repeat=repeat,
                                           fargs=(fig, ax, displ, ttl, comp))
            return anim,
        else:
            raise ValueError("I don't know any '{}' composant".format(compo))

    def record_animation(self, anim, filepath, kind='gif', fps=30, dpi=100,
                         bitrate=50, imagemagick_path=None):
        """
        Record an animation in a gif file.
        You must create an animation (using 'display_animate' for example)
        before calling this method.
        You may have to specify the path to imagemagick in orfer to use it.


        Parameters
        ----------

        """
        import matplotlib
        if imagemagick_path is None:
            imagemagick_path = r"C:\Program Files\ImageMagick\convert.exe"
        matplotlib.rc('animation', convert_path=imagemagick_path)
        if kind == 'gif':
            writer = matplotlib.animation.ImageMagickWriter(fps=fps,
                                                            bitrate=bitrate)
            anim.save(filepath, writer=writer, fps=fps, dpi=dpi)
        elif kind == 'mp4':
            anim.save(filepath, writer='fmpeg', fps=fps, bitrate=bitrate)

    def display_time_profiles(self, componentname, direction, positions,
                              **plotargs):
        """
        Display multiples profiles of the same cut, but for differents times.

        Parameters
        ----------
        componentname : string
            Component wanted for the profile.
        direction : integer
            Direction along which we choose a position (1 for x and 2 for y).
        positions : tuple of numbers
            Positions in which we want a profile (possibly a slice))
        **plotargs :
            Supplementary arguments for the plot() function.
        """
        for field in self.fields:
            profile, positions = field.get_profile(componentname, direction,
                                                   positions)
            if direction == 1:
                displ = profile._display(label="t={0}".format(field.time *
                                                              field.unit_time),
                                         **plotargs)
                plt.xlabel("{0} {1}".format(componentname, profile.unit_x))
                plt.ylabel("Y {0}".format(profile.unit_y))
            else:
                displ = profile.display(label="t={0}".format(field.time *
                                                             field.unit_time),
                                        **plotargs)
                plt.ylabel("{0} {1}".format(componentname, profile.unit_y))
                plt.xlabel("X {0}".format(profile.unit_x))
        plt.title(componentname)
        plt.legend()
        return displ


class SpatialVelocityFields(VelocityFields):
    """
    Class representing a set of spatial-evolving velocity fields.

    Principal methods
    -----------------
    "add_field" : add a velocity field.

    "remove_field" : remove a field.

    "display" : display the vector field, with these unities.

    "get_bl*" : compute different kind of boundary layer thickness.
    """

    def import_from_svfs(self, svfs):
        """
        Import velocity fields from another spatialvelocityfields.

        Parameters
        ----------
        tvfs : SpatialVelocityFields object
            Velocity fields to copy.
        """
        for componentkey in svfs.__dict__.keys():
            component = svfs.__dict__[componentkey]
            if isinstance(component, (VelocityField, VectorField,
                                      ScalarField)):
                self.__dict__[componentkey] \
                    = component.copy()
            elif isinstance(component, NUMBERTYPES):
                self.__dict__[componentkey] \
                    = component*1
            elif isinstance(component, STRINGTYPES):
                self.__dict__[componentkey] \
                    = component
            elif isinstance(component, unum.Unum):
                self.__dict__[componentkey] \
                    = component.copy()
            else:
                raise TypeError("Unknown attribute type in VelocityFields "
                                "object (" + str(componentkey) + " : "
                                + str(type(component)) + ")."
                                " You must implemente it.")

    def copy(self):
        """
        Return a copy of the velocityfields
        """
        tmp_svfs = SpatialVelocityFields()
        tmp_svfs.import_from_svfs(self)
        return tmp_svfs

    def get_bl(self, componentname, direction, value, rel=False,
               kind="default"):
        """
        Return a profile on all the velocity fields.
        """
        if not isinstance(self.fields[0].get_comp(componentname),
                          ScalarField):
            raise ValueError("'componentname' must make reference to a "
                             "ScalarField")
        axe = np.array([], dtype=float)
        bl = np.array([], dtype=float)
        for field in self.fields:
            compo = field.get_comp(componentname)
            tmp_bl = compo.get_bl(direction, value, rel=rel, kind=kind)
            axe = np.append(axe, tmp_bl.x)
            bl = np.append(bl, tmp_bl.y)
        conc = zip(axe, bl)
        conc = np.array(conc, dtype=[('x', float), ('y', float)])
        conc = np.sort(conc, axis=0, order='x')
        axe, blayer = zip(*conc)
        if direction == 1:
            unit_x, unit_y = self.get_axe_units()
        else:
            unit_y, unit_x = self.get_axe_units()
        bl_profile = Profile(axe, blayer, unit_x, unit_y, "Boundary Layer")
        return bl_profile

    def _display(self, componentname="V", scale=1, fieldnumber=None,
                 **plotargs):
        """
        Display all the velocity fields on a single figure.
        If fieldnumber is precised, only the wanted field is displayed.
        ----------
        componentname : string, optional
            Component to display ('V', 'Vx', Vy' or 'magnitude').
        scale : number, optional
            If the display is a quiver, arrows are scaled by this factor.
        fieldnumber : interger, optional
            The number of the field to display
        plotargs : dicte, optional
            Arguments passed to the function used to display the vector field.
        """
        if fieldnumber is None:
            fieldnumber = np.arange(0, len(self.fields))
        if isinstance(fieldnumber, int):
            if fieldnumber > len(self.fields)-1:
                raise ValueError("Field number {0} do not exist"
                                 .format(fieldnumber))
            self.fields[fieldnumber]._display(componentname, scale,
                                              **plotargs)
        elif isinstance(fieldnumber, ARRAYTYPES):
            comp = self.fields[0].get_comp(componentname)
            if isinstance(comp, ScalarField):
                try:
                    plotargs["levels"]
                except KeyError:
                    mins = []
                    maxs = []
                    for nmb in fieldnumber:
                        field = self.fields[nmb]
                        mins.append(field.get_min(componentname))
                        maxs.append(field.get_max(componentname))
                    mins.sort()
                    cmin = mins[0]
                    maxs.sort()
                    cmax = maxs[-1]
                    levels = np.arange(cmin, cmax, (cmax-cmin)/19)
                    plotargs["levels"] = levels
            elif isinstance(comp, VectorField):
                pass
            self.fields[fieldnumber[0]]._display(componentname, scale,
                                                 **plotargs)
            for nmb in fieldnumber[1:]:
                field = self.fields[nmb]
                field._display(componentname, scale, **plotargs)
        else:
            raise TypeError("'fieldnumber' must be an interger, an array of "
                            "integers or None")

    def display(self, componentname="V", scale=1, fieldnumber=None,
                **plotargs):
        """
        Display the spatial velocity fields.
        """
        self._display(componentname, scale, fieldnumber, **plotargs)
        cbar = plt.colorbar()
        compo = self.fields[0].get_comp(componentname)
        if isinstance(compo, ScalarField):
            unit = compo.unit_values.Getunit()
        elif isinstance(compo, VectorField):
            unit = compo.get_unit_values
        else:
            unit = compo.get_unit_values()
        cbar.set_label("{0} {1}".format(componentname, unit))
