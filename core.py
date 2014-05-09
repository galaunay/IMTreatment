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
        xy = np.array(xy, subok=True)
        if xy.shape != (0,) and (len(xy.shape) != 2 or xy.shape[1] != 2):
            raise ValueError("'xy' must be a tuple of 2x1 arrays")
        if v is not None:
            if not isinstance(v, ARRAYTYPES):
                raise TypeError("'v' must be an array")
            v = np.array(v)
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
                fig = plt.scatter(self.xy[:, 0], self.xy[:, 1], **plotargs)
            else:
                if not 'cmap' in plotargs:
                    plotargs['cmap'] = plt.cm.jet
                if not 'c' in plotargs:
                    plotargs['c'] = self.v
                fig = plt.scatter(self.xy[:, 0], self.xy[:, 1], **plotargs)
        elif kind == 'plot':
            fig = plt.plot(self.xy[:, 0], self.xy[:, 1], **plotargs)
        return fig

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
        fig = self._display(kind, **plotargs)
        if self.v is not None and kind == 'scatter':
            cb = plt.colorbar(fig)
            cb.set_label(self.unit_v.strUnit())
        plt.xlabel('X ' + self.unit_x.strUnit())
        plt.ylabel('Y ' + self.unit_y.strUnit())
        if self.name is None:
            plt.title('Set of points')
        else:
            plt.title(self.name)

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
            x = np.array(x)
        if not isinstance(y, ARRAYTYPES):
            raise TypeError("'y' must be an array")
        if not isinstance(y, (np.ndarray, np.ma.MaskedArray)):
            y = np.array(y)
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

    def get_copy(self):
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
        interval = np.array(interval)
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
        smoothx = np.array([])
        smoothy = np.array([])
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
            fig = plt.plot(x, y, **plotargs)
        elif kind == 'semilogx':
            fig = plt.semilogx(x, y, **plotargs)
        elif kind == 'semilogy':
            fig = plt.semilogy(x, y, **plotargs)
        elif kind == 'loglog':
            fig = plt.loglog(x, y, **plotargs)
        else:
            raise ValueError("Unknown plot type : {}.".format(kind))
        return fig

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
        fig = self._display(kind, reverse, **plotargs)
        plt.title(self.name)
        if not reverse:
            plt.xlabel("{0}".format(self.unit_x.strUnit()))
            plt.ylabel("{0}".format(self.unit_y.strUnit()))
        else:
            plt.xlabel("{0}".format(self.unit_y.strUnit()))
            plt.ylabel("{0}".format(self.unit_x.strUnit()))
        return fig


class Field(object):

    def __init__(self):
        self.axe_x = np.array([])
        self.axe_y = np.array([])
        self.unit_x = make_unit('')
        self.unit_y = make_unit('')

    def __iter__(self):
        for i, x in enumerate(self.axe_x):
            for j, y in enumerate(self.axe_y):
                yield [i, j], [x, y]

    def get_axes(self):
        """
        Return the field axes.

        Returns
        -------
        axe_x : array
            Axe along X.
        axe_y : array
            Axe along Y.
        """
        return self.axe_x, self.axe_y

    def get_axe_units(self):
        """
        Return the axis unities.

        Returns:
        --------
        unit_x : unit object
            Axe x unit
        unit_y : unit object
            Axe y unit
        """
        return self.unit_x, self.unit_y

    def get_copy(self):
        """
        Return a copy of the Field object.
        """
        return copy.deepcopy(self)

    def set_axes(self, axe_x=None, axe_y=None):
        """
        Load new axes in the field.

        Parameters
        ----------
        axe_x : array
            One-dimensionale array representing the position of the scalar
            values along the X axe.
        axe_y : array
            idem for the Y axe.
        """
        if axe_x is not None:
            if not isinstance(axe_x, ARRAYTYPES):
                raise TypeError("'axe_x' must be an array")
            axe_x = np.array(axe_x)
            if self.axe_x.shape == axe_x.shape:
                self.axe_x = axe_x
            else:
                raise ValueError("Inconsistent size of 'axe_x'")
        if axe_y is not None:
            if not isinstance(axe_y, ARRAYTYPES):
                raise TypeError("'axe_y' must be an array")
            axe_y = np.array(axe_y)
            if axe_y.shape == self.axe_y.shape:
                self.axe_y = axe_y
            else:
                raise ValueError("Inconsistent size of 'axe_y'")

    def set_axe_units(self, unit_x=None, unit_y=None):
        """
        Load unities into the field axis.

        Parameters
        ----------
        unit_x : Unit object
            Axis X unit.
        unit_y : Unit object
            Axis Y unit.
        """
        if unit_x is not None:
            if not isinstance(unit_x, unum.Unum):
                raise TypeError("'unit_x' must be an unit object")
            self.unit_x = unit_x
        if unit_y is not None:
            if not isinstance(unit_y, unum.Unum):
                raise TypeError("'unit_y' must be an unit object")
            self.unit_y = unit_y

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
        if intervalx is None:
            intervalx = [self.axe_x[0], self.axe_x[-1]]
        if intervaly is None:
            intervaly = [self.axe_y[0], self.axe_y[-1]]
        if not isinstance(intervalx, ARRAYTYPES):
            raise TypeError("'intervalx' must be an array of two numbers")
        intervalx = np.array(intervalx)
        if intervalx.ndim != 1:
            raise ValueError("'intervalx' must be an array of two numbers")
        if intervalx.shape != (2,):
            raise ValueError("'intervalx' must be an array of two numbers")
        if intervalx[0] > intervalx[1]:
            raise ValueError("'intervalx' values must be crescent")
        if not isinstance(intervaly, ARRAYTYPES):
            raise TypeError("'intervaly' must be an array of two numbers")
        intervaly = np.array(intervaly)
        if intervaly.ndim != 1:
            raise ValueError("'intervaly' must be an array of two numbers")
        if intervaly.shape != (2,):
            raise ValueError("'intervaly' must be an array of two numbers")
        if intervaly[0] > intervaly[1]:
            raise ValueError("'intervaly' values must be crescent")
        # finding interval indices
        if intervalx[0] <= self.axe_x[0]:
            indmin_x = 0
        else:
            indmin_x = self.get_indice_on_axe(1, intervalx[0])[-1]
        if intervalx[1] >= self.axe_x[-1]:
            indmax_x = len(self.axe_x) - 1
        else:
            indmax_x = self.get_indice_on_axe(1, intervalx[1])[0]
        if intervaly[0] <= self.axe_y[0]:
            indmin_y = 0
        else:
            indmin_y = self.get_indice_on_axe(2, intervaly[0])[-1]
        if intervaly[1] >= self.axe_y[-1]:
            indmax_y = len(self.axe_y) - 1
        else:
            indmax_y = self.get_indice_on_axe(2, intervaly[1])[0]
        trimfield = self.get_copy()
        trimfield.axe_x = trimfield.axe_x[indmin_x:indmax_x + 1]
        trimfield.axe_y = trimfield.axe_y[indmin_y:indmax_y + 1]
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
            inds = [ind - 1, ind]
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
            return ind

    def get_points_around(self, center, radius):
        """
        Return the list of points or the scalar field that are in a circle
        centered on 'center' and of radius 'radius'.

        Parameters
        ----------
        center : array
            Coordonate of the center point (in axes units).
        radius : float
            radius of the cercle (in axes units).

        Returns
        -------
        indices : array
            Array contening the indices of the contened points.
            [(ind1x, ind1y), (ind2x, ind2y), ...].
            You can easily put them in the axes to obtain points coordinates
        """
        if not isinstance(center, ARRAYTYPES):
            raise TypeError("'center' must be an array")
        center = np.array(center)
        if not center.shape == (2,):
            raise ValueError("'center' must be a 2x1 array")
        if not isinstance(radius, NUMBERTYPES):
            raise TypeError("'radius' must be a number")
        if not radius > 0:
            raise ValueError("'radius' must be positive")
        radius2 = radius**2
        radius_int = radius/np.sqrt(2)
        inds = []
        for indices, coord, _ in self:
            # test if the point is not in the square surrounding the cercle
            if coord[0] >= center[0] + radius \
                    and coord[0] <= center[0] - radius \
                    and coord[1] >= center[1] + radius \
                    and coord[1] <= center[1] - radius:
                pass
            # test if the point is in the square 'compris' in the cercle
            elif coord[0] <= center[0] + radius_int \
                    and coord[0] >= center[0] - radius_int \
                    and coord[1] <= center[1] + radius_int \
                    and coord[1] >= center[1] - radius_int:
                inds.append(indices)
            # test if the point is the center
            elif all(coord == center):
                pass
            # test if the point is in the circle
            elif ((coord[0] - center[0])**2 + (coord[1] - center[1])**2
                    <= radius2):
                inds.append(indices)
        return inds


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
        self.values = np.ma.masked_array([])

    def __eq__(self, another):
        if not isinstance(another, ScalarField):
            return False
        if not np.all(self.get_axes()[0] == another.get_axes()[0]):
            return False
        if not np.all(self.get_axes()[1] == another.get_axes()[1]):
            return False
        if not np.all(self.get_comp('values').data
                == another.get_comp('values').data):
            return False
        if not np.all(self.get_comp('mask', raw=True)
                == another.get_comp('mask', raw=True)):
            return False
        if not np.all(self.get_axe_units()[0] == another.get_axe_units()[0]):
            return False
        if not np.all(self.get_axe_units()[1] == another.get_axe_units()[1]):
            return False
        if not np.all(self.get_unit_values() == another.get_unit_values()):
            return False
        return True


    def __neg__(self):
        tmpsf = self.get_copy()
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
            tmpsf = self.get_copy()
            fact = otherone.unit_values/self.unit_values
            tmpsf.values += otherone.values*fact.asNumber()
            return tmpsf
        # if we add with a number
        elif isinstance(otherone, NUMBERTYPES):
            tmpsf = self.get_copy()
            tmpsf.values += otherone
            return tmpsf
        elif isinstance(otherone, unum.Unum):
            try:
                self.unit_values + otherone
            except:
                pdb.set_trace()
                raise ValueError("Given number have to be consistent with"
                                 "the scalar field (same units)")
            tmpsf = self.get_copy()
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
            tmpsf = self.get_copy()
            tmpsf.values /= obj
            return tmpsf
        elif isinstance(obj, unum.Unum):
            tmpsf = self.get_copy()
            tmpsf.values /= obj.asNumber()
            tmpsf.unit_values /= obj/obj.asNumber()
            return tmpsf
        elif isinstance(obj, ScalarField):
            if np.any(self.axe_x != obj.axe_x)\
                    or np.any(self.axe_y != obj.axe_y)\
                    or self.unit_x != obj.unit_x\
                    or self.unit_y != obj.unit_y:
                raise ValueError("Fields are not consistent")
            tmpsf = self.get_copy()
            values = self.values / obj.values
            unit = self.unit_values / obj.unit_values
            tmpsf.values = values*unit.asNumber()
            tmpsf.unit_values = unit/unit.asNumber()
            return tmpsf
        else:
            raise TypeError("Unsupported operation between {} and a "
                            "ScalarField object".format(type(obj)))

    __div__ = __truediv__

    def __rdiv__(self, obj):
        if isinstance(obj, NUMBERTYPES):
            tmpsf = self.get_copy()
            tmpsf.values = obj/tmpsf.values
            tmpsf.unit_values = 1/tmpsf.unit_values
            return tmpsf
        elif isinstance(obj, unum.Unum):
            tmpsf = self.get_copy()
            tmpsf.values = obj.asNumber()/tmpsf.values
            tmpsf.unit_values = obj/obj.asNumber()/tmpsf.unit_values
            return tmpsf
        elif isinstance(obj, ScalarField):
            if np.any(self.axe_x != obj.axe_x)\
                    or np.any(self.axe_y != obj.axe_y)\
                    or self.unit_x != obj.unit_x\
                    or self.unit_y != obj.unit_y:
                raise ValueError("Fields are not consistent")
            tmpsf = self.get_copy()
            values = obj.values / self.values
            unit =  obj.unit_values / self.unit_values
            tmpsf.values = values*unit.asNumber()
            tmpsf.unit_values = unit/unit.asNumber()
            return tmpsf
        else:
            raise TypeError("Unsupported operation between {} and a "
                            "ScalarField object".format(type(obj)))

    def __mul__(self, obj):
        if isinstance(obj, NUMBERTYPES):
            tmpsf = self.get_copy()
            tmpsf.values *= obj
            return tmpsf
        elif isinstance(obj, unum.Unum):
            tmpsf = self.get_copy()
            tmpsf.values *= obj.asNumber()
            tmpsf.unit_values *= obj/obj.asNumber()
            return tmpsf
        elif isinstance(obj, np.ma.core.MaskedArray):
            if obj.shape != self.values.shape:
                raise ValueError("Fields are not consistent")
            tmpsf = self.get_copy()
            tmpsf.values *= obj
            return tmpsf
        elif isinstance(obj, ScalarField):
            if np.any(self.axe_x != obj.axe_x)\
                    or np.any(self.axe_y != obj.axe_y)\
                    or self.unit_x != obj.unit_x\
                    or self.unit_y != obj.unit_y:
                raise ValueError("Fields are not consistent")
            tmpsf = self.get_copy()
            values = self.values * obj.values
            unit = self.unit_values * obj.unit_values
            tmpsf.values = values*unit.asNumber()
            tmpsf.unit_values = unit/unit.asNumber()
            return tmpsf
        else:
            raise TypeError("Unsupported operation between {} and a "
                            "ScalarField object".format(type(obj)))
    __rmul__ = __mul__

    def __abs__(self):
        tmpsf = self.get_copy()
        tmpsf.values = np.abs(tmpsf.values)
        return tmpsf

    def __pow__(self, number):
        if not isinstance(number, NUMBERTYPES):
            raise TypeError("You only can use a number for the power "
                            "on a Scalar field")
        tmpsf = self.get_copy()
        tmpsf.values = np.power(tmpsf.values, number)
        tmpsf.unit_values = np.power(tmpsf.unit_values, number)
        return tmpsf

    def __iter__(self):
        try:
            mask = self.values.mask
            data = self.values.data
        except AttributeError:
            data = self.values
            mask = np.zeros(data.shape)
        for ij, xy in Field.__iter__(self):
            i = ij[0]
            j = ij[1]
            if not mask[j, i]:
                yield ij, xy, data[j, i]

#    def import_from_davis(self, filename):
#        """
#        Import a scalar field from a .IM7 file.
#
#        Parameters
#        ----------
#        filename : string
#            Path to the IM7 file.
#        """
#        if not isinstance(filename, STRINGTYPES):
#            raise TypeError("'filename' must be a string")
#        if not os.path.exists(filename):
#            raise ValueError("I did not find your file, boy")
#        _, ext = os.path.splitext(filename)
#        if not (ext == ".im7" or ext == ".IM7"):
#            raise ValueError("I need the file to be an IM7 file")
#        v = IM.IM7(filename)
#        axe_x = v.Px[0, :]
#        axe_y = v.Py[:, 0]
#        values = v.getmaI()[0]*v.buffer['scaleI']['factor']
#        unit_x = v.buffer['scaleX']['unit'].split("\x00")[0]
#        unit_x = unit_x.replace('[', '')
#        unit_x = unit_x.replace(']', '')
#        unit_x = make_unit(unit_x)
#        unit_y = v.buffer['scaleY']['unit'].split("\x00")[0]
#        unit_y = unit_y.replace('[', '')
#        unit_y = unit_y.replace(']', '')
#        unit_y = make_unit(unit_y)
#        unit_values = v.buffer['scaleI']['unit'].split("\x00")[0]
#        unit_values = unit_values.replace('[', '')
#        unit_values = unit_values.replace(']', '')
#        unit_values = make_unit(unit_values)
#        # check if axe are crescent
#        if axe_y[-1] < axe_y[0]:
#            axe_y = axe_y[::-1]
#            values = values[::-1, :]
#        if axe_x[-1] < axe_x[0]:
#            axe_x = axe_x[::-1]
#            values = values[:, ::-1]
#        self.import_from_arrays(axe_x=axe_x, axe_y=axe_y, values=values,
#                                unit_x=unit_x, unit_y=unit_y,
#                                unit_values=unit_values)

#    def import_from_ascii(self, filename, x_col=1, y_col=2, v_col=3,
#                          unit_x=make_unit(""), unit_y=make_unit(""),
#                          unit_values=make_unit(""), **kwargs):
#        """
#        Import a scalarfield from an ascii file.
#
#        Parameters
#        ----------
#        x_col, y_col, v_col : integer, optional
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
#                or not isinstance(v_col, int):
#            raise TypeError("'x_col', 'y_col' and 'v_col' must be integers")
#        if x_col < 1 or y_col < 1 or v_col < 1:
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
#        v = data[:, v_col-1]
#        # Masking all the initial field (to handle missing values)
#        v_org = np.zeros((x_org.shape[0], y_org.shape[0]))
#        v_org_mask = np.ones(v_org.shape)
#        v_org = np.ma.masked_array(v_org, v_org_mask)
#        #loop on all 'v' values
#        for i in np.arange(v.shape[0]):
#            x_tmp = x[i]
#            y_tmp = y[i]
#            v_tmp = v[i]
#            #find x index
#            for j in np.arange(x_org.shape[0]):
#                if x_org[j] == x_tmp:
#                    x_ind = j
#            #find y index
#            for j in np.arange(y_org.shape[0]):
#                if y_org[j] == y_tmp:
#                    y_ind = j
#            #put the value at its place
#            v_org[x_ind, y_ind] = v_tmp
#        # treating 'nan' values
#        v_org.mask = np.logical_and(v_org.mask, np.isnan(v_org.data))
#        #store field in attributes
#        self.import_from_arrays(axe_x=x_org, axe_y=y_org, values=v_org,
#                                unit_x=unit_x, unit_y=unit_y,
#                                unit_values=make_unit(''))

    def import_from_arrays(self, axe_x, axe_y, values, unit_x=make_unit(""),
                           unit_y=make_unit(""), unit_values=make_unit("")):
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
        unit_x : Unit object, optionnal
            Unit for the values of axe_x
        unit_y : Unit object, optionnal
            Unit for the values of axe_y
        unit_values : Unit object, optionnal
            Unit for the scalar field
        """
        # checking parameters coherence
        if not isinstance(axe_x, ARRAYTYPES):
            raise TypeError("'axe_x' must be an array")
        else:
            axe_x = np.array(axe_x)
        if not axe_x.ndim == 1:
            raise ValueError("'axe_x' must be a one dimension array")
        if not isinstance(axe_y, ARRAYTYPES):
            raise TypeError("'axe_y' must be an array")
        else:
            axe_y = np.array(axe_y)
        if not axe_y.ndim == 1:
            raise ValueError("'axe_y' must be a one dimension array")
        if not isinstance(values, ARRAYTYPES):
            raise TypeError("'values' must be an array")
        elif isinstance(values, (list, tuple)):
            values = np.array(values)
        if not values.ndim == 2:
            raise ValueError("'values' must be a two dimension array")
        if unit_x is not None:
            if not isinstance(unit_x, unum.Unum):
                raise TypeError("'unit_x' must be an Unit object")
        if unit_y is not None:
            if not isinstance(unit_y, unum.Unum):
                raise TypeError("'unit_y' must be an Unit object")
        if unit_values is not None:
            if not isinstance(unit_values, unum.Unum):
                raise TypeError("'unit_values' must be an Unit object")
        if (values.shape[0] != axe_y.shape[0] or
                values.shape[1] != axe_x.shape[0]):
            raise ValueError("Dimensions of 'axe_x', 'axe_y' and 'values' must"
                             " be consistents")
        # storing datas
        unit_x_value = unit_x._value
        unit_y_value = unit_y._value
        unit_values_value = unit_values._value
        self.axe_x = axe_x.copy()*unit_x_value
        self.axe_y = axe_y.copy()*unit_y_value
        if isinstance(values, np.ma.MaskedArray):
            mask = values.mask
            values = values.data
        else:
            values = np.array(values)
            mask = np.zeros(values.shape)
        values = values*unit_values_value
        self.values = np.ma.masked_array(values, mask, copy=True)
        self.unit_x = unit_x.copy()/unit_x_value
        self.unit_y = unit_y.copy()/unit_y_value
        self.unit_values = unit_values.copy()/unit_values_value
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
            mask = np.zeros(self.get_dim())
        if not isinstance(mask, ARRAYTYPES):
            raise TypeError("'mask' must be an array of boolean")
        mask = np.array(mask)
        if mask.shape != self.get_dim():
            raise ValueError("'mask' must have the same dimensions as"
                             "the ScalarField :{}".format(self.get_dim()))
        # récupération du masque
        mask = np.logical_or(mask, self.values.mask)
        pts = None
        v = np.array([])

        for inds, pos, value in self:
            if mask[inds[1], inds[0]]:
                continue
            if pts is None:
                pts = [pos]
            else:
                pts = np.append(pts, [pos], axis=0)
            v = np.append(v, value)
        return Points(pts, v, self.unit_x, self.unit_y, self.unit_values)

    def get_comp(self, componentname, raw=False):
        """
        Return a ScalarField object representing a component of the
        Vectorfield object.

        Parameters
        ----------
        componentname : string
            Can be 'values' or 'mask'.
        raw : boolean, optional
            If 'False' (default), return a ScalarField object,
            if 'True', return a masked array.

        Returns
        -------
        component : ScalarField object or numpy masked array
        """
        if not isinstance(componentname, STRINGTYPES):
            raise TypeError("'componentname' must be a string")
        if componentname == 'values':
            values = self.values
        elif componentname == 'mask':
            values = self.values.mask
        else:
            raise ValueError("unknown value of 'componentname'")
        if raw:
            return values
        else:
            axe_x, axe_y = self.get_axes()
            unit_x, unit_y = self.get_axe_units()
            unit_values = self.get_unit_values()
            tmpsf = ScalarField()
            tmpsf.import_from_arrays(axe_x, axe_y, values, unit_x, unit_y,
                                     unit_values)
            return tmpsf

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

    def get_unit_values(self):
        """
        Return the values unit
        """
        return self.unit_values

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

    def set_comp(self, componentname, value):
        """
        Fill the component 'componentname' with 'value'.

        Parameters
        ----------
        componentname : string
            Can be 'values' or 'mask'.
        value : array
            Array with the same shape as the initial component
        """
        if not isinstance(componentname, STRINGTYPES):
            raise TypeError("'componentname' must be a string")
        if not isinstance(value, ARRAYTYPES):
            raise TypeError("'value' must be an array")
        if not isinstance(value, np.ma.MaskedArray):
             value = np.ma.masked_array(value)
        if self.get_dim() != value.shape:
            raise ValueError("'value' dimensions are inconsistent with the "
                             "ScalarField shape")
        if componentname == 'values':
            self.values = value
        elif componentname == 'mask':
            self.values.mask = value
        else:
            raise ValueError("Unknown 'componentname' value")

    def set_unit_values(self, unit_values=None):
        """
        Load unities into the scalar field values.

        Parameters
        ----------
        unit_values : Unit object
            Values unit.
        """
        if unit_values is not None:
            if not isinstance(unit_values, unum.Unum):
                raise TypeError("'unit_values' must be an Unit object")
            self.unit_values = unit_values

    def get_dim(self):
        """
        Return the scalar field dimension.

        Returns
        -------
        shape : tuple
            Tuple of the dimensions (along X and Y) of the scalar field.
        """
        return self.values.shape

    def get_min(self, unit=False):
        """
        Return the minima of the field.

        Parameters
        ----------
        unit : boolean, optional
            If True, a unit object is returned,
            else (default), a float is returned.

        Returns
        -------
        mini : float or unit object
            Minima on the field
        """
        if unit:
            return np.min(self.values)*self.unit_values
        else:
            return np.min(self.values)

    def get_max(self, unit=False):
        """
        Return the maxima of the field.

        Parameters
        ----------
        unit : boolean, optional
            If True, a unit object is returned,
            else (default), a float is returned.

        Returns
        -------
        maxi : float or unit object
            Maxima on the field
        """
        if unit:
            return np.max(self.values)*self.unit_values
        else:
            return np.max(self.values)

    def get_mean(self, unit=False):
        """
        Return the mean value of the field.

        Parameters
        ----------
        unit : boolean, optional
            If True, a unit object is returned,
            else (default), a float is returned.

        Returns
        -------
        mean : float or unit object
            Mean value of the field.
        """
        if unit:
            return np.mean(self.values)*self.unit_values
        else:
            return np.mean(self.values)

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
            axe_x, axe_y = self.get_axes()
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
            a, b = np.meshgrid(axe_x[ind_x:ind_x + 2], axe_y[ind_y:ind_y + 2])
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
            bornes = np.array(bornes)
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
            mini = self.get_min()
            maxi = self.get_max()
            if np.abs(bornes[0]) > np.abs(bornes[1]):
                bornes[1] = abs(maxi - mini)*bornes[1] + maxi
                bornes[0] = abs(maxi - mini)*bornes[0] + maxi
            else:
                bornes[1] = abs(maxi - mini)*bornes[1] + mini
                bornes[0] = abs(maxi - mini)*bornes[0] + mini
        # check if the zone exist
        else:
            mini = self.get_min()
            maxi = self.get_max()
            if maxi < bornes[0] or mini > bornes[1]:
                return None
        # check if there is more than one point superior
        aoi = np.logical_and(self.values >= bornes[0],
                             self.values <= bornes[1])
        if np.sum(aoi) == 1:
            inds = np.where(aoi)
            x = self.axe_x[inds[1][0]]
            y = self.axe_y[inds[0][0]]
            return Points([[x, y]], unit_x=self.unit_x,
                          unit_y=self.unit_y)
        zones = np.logical_and(np.logical_and(self.values >= bornes[0],
                                              self.values <= bornes[1]),
                               np.logical_not(np.ma.getmaskarray(self.values)))
        # compute the center with labelzones
        labeledzones, nmbzones = msr.label(zones)
        inds = []
        if kind == 'extremum':
            mins, _, ind_min, ind_max = msr.extrema(self.values,
                                                    labeledzones,
                                                    np.arange(nmbzones) + 1)
            for i in np.arange(len(mins)):
                if bornes[np.argmax(np.abs(bornes))] < 0:
                    inds.append(ind_min[i])
                else:
                    inds.append(ind_max[i])
        elif kind == 'center':
            inds = msr.center_of_mass(np.ones(self.get_dim()),
                                      labeledzones,
                                      np.arange(nmbzones) + 1)
        elif kind == 'ponderated':
            inds = msr.center_of_mass(np.abs(self.values), labeledzones,
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
        coords = np.array(coords)
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
            position = np.array(position)
            if not position.shape == (2,):
                raise ValueError("'position' must be a number or an interval")
        if direction == 1:
            axe = self.axe_x
            unit_x = self.unit_y.copy()
            unit_y = self.unit_values.copy()
        else:
            axe = self.axe_y
            unit_x = self.unit_x.copy()
            unit_y = self.unit_values.copy()
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
                profile = self.values[:, finalindice]
                axe = self.axe_y
                cutposition = self.axe_x[finalindice]
            else:
                profile = self.values[finalindice, :]
                axe = self.axe_x
                cutposition = self.axe_y[finalindice]
        # Calculation of the profile for an interval of position
        else:
            axe_mask = np.logical_and(axe >= position[0], axe <= position[1])
            if direction == 1:
                profile = self.values[:, axe_mask].mean(1)
                axe = self.axe_y
                cutposition = self.axe_x[axe_mask]
            else:
                profile = self.values[axe_mask, :].mean(0)
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

    def get_copy(self):
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
        tmpsf = self.get_copy()
        tmpsf.axe_x = tmpsf.axe_x[indmin_x:indmax_x + 1]
        tmpsf.axe_y = tmpsf.axe_y[indmin_y:indmax_y + 1]
        tmpsf.values = tmpsf.values[indmin_y:indmax_y + 1,
                                    indmin_x:indmax_x + 1]
        return tmpsf

    def crop_masked_border(self):
        """
        Crop the masked border of the field in place.
        """
        axe_x, axe_y = self.get_axes()
        # checking masked values presence
        values = self.get_comp('values', raw=True)
        if not np.ma.is_masked(values):
            return None
        # getting datas
        values = values.data
        mask = self.get_comp('mask', raw=True)
        # crop border along y
        axe_y_m = ~np.all(mask, axis=1)
        if np.any(axe_y_m):
            values = values[axe_y_m, :]
            mask = mask[axe_y_m, :]
        # crop values along x
        axe_x_m = ~np.all(mask, axis=0)
        if np.any(axe_x_m):
            values = values[:, axe_x_m]
            mask = mask[:, axe_x_m]
        # storing cropped values
        self.values = np.ma.masked_array(values, mask)
        # crop axis
        self.axe_x = axe_x[axe_x_m]
        self.axe_y = axe_y[axe_y_m]

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
        mask = self.get_comp('mask', raw=True)
        values = self.get_comp('values', raw=True).data
        # if there is nothing to do...
        if not np.any(mask):
            pass
        elif tof == 'interplin':
            inds_x = np.arange(values.shape[1])
            inds_y = np.arange(values.shape[0])
            grid_x, grid_y = np.meshgrid(inds_x, inds_y)
            f = spinterp.interp2d(grid_y[~mask],
                                  grid_x[~mask],
                                  values[~mask])
            for inds, masked in np.ndenumerate(mask):
                if masked:
                    values[inds[0], inds[1]] = f(inds[0], inds[1])
            mask = np.zeros(values.shape)
            self.set_comp('values', np.ma.masked_array(values, mask))
        elif tof == 'interpcub':
            inds_x = np.arange(values.shape[1])
            inds_y = np.arange(values.shape[0])
            grid_x, grid_y = np.meshgrid(inds_x, inds_y)
            f = spinterp.interp2d(grid_y[~mask],
                                  grid_x[~mask],
                                  values[~mask],
                                  kind='cubic')
            for inds, masked in np.ndenumerate(mask):
                if masked:
                    values[inds[1], inds[0]] = f(inds[1], inds[0])
            mask = np.zeros(values.shape)
            self.set_comp('values', np.ma.masked_array(values, mask))
        elif tof == 'value':
            tmp_val = self.get_comp('values', raw=True)
            tmp_val[self.values.mask] = value
            self.set_comp('values', tmp_val)
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
        values = self.values.data
        mask = self.get_comp('mask', raw=True)
        # smoothing
        if tos == "uniform":
            values = ndimage.uniform_filter(values, size, **kw)
        elif tos == "gaussian":
            values = ndimage.gaussian_filter(values, size, **kw)
        else:
            raise ValueError("'tos' must be 'uniform' or 'gaussian'")
        # storing
        self.set_comp('values', np.ma.masked_array(values, mask))

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
        axe_x, axe_y = self.get_axes()
        if intervalx is None:
            intervalx = [axe_x[0], axe_x[-1]]
        if intervaly is None:
            intervaly = [axe_y[0], axe_y[-1]]
        trimfield = self.trim_area(intervalx, intervaly)
        axe2_x, axe2_y = trimfield.get_axes()
        unit_x, unit_y = trimfield.get_axe_units()
        integral = (trimfield.get_comp('values', raw=True).sum()
                    * np.abs(axe2_x[-1] - axe2_x[0])
                    * np.abs(axe2_y[-1] - axe2_y[0])
                    / len(axe2_x)
                    / len(axe2_y))
        unit = trimfield.unit_values*unit_x*unit_y
        return integral*unit

    def _display(self, kind=None, **plotargs):
        X, Y = np.meshgrid(*self.get_axes())
        values = self.get_comp('values', raw=True)
        axe_x, axe_y = self.get_axes()
        unit_x, unit_y = self.get_axe_units()
        if kind == 'contour':
            if (not 'cmap' in plotargs.keys()
                    and not 'colors' in plotargs.keys()):
                plotargs['cmap'] = cm.jet
            fig = plt.contour(X, Y, values, linewidth=1, **plotargs)
        elif kind == 'contourf':
            if 'cmap' in plotargs.keys() or 'colors' in plotargs.keys():
                fig = plt.contourf(X, Y, values, linewidth=1, **plotargs)
            else:
                fig = plt.contourf(X, Y, values, cmap=cm.jet, linewidth=1,
                                   **plotargs)
        elif kind is None:
            if not 'cmap' in plotargs.keys():
                plotargs['cmap'] = cm.jet
            if not 'interpolation' in plotargs.keys():
                plotargs['interpolation'] = 'nearest'
            if not 'aspect' in plotargs.keys():
                plotargs['aspect'] = 'equal'
            delta_x = axe_x[1] - axe_x[0]
            delta_y = axe_y[1] - axe_y[0]
            fig = plt.imshow(values,
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
        return fig

    def display(self, kind=None, **plotargs):
        """
        Display the scalar field.

        Parameters
        ----------
        kind : string, optinnal
            If 'None': each datas are plotted (imshow),
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
        fig = self._display(kind, **plotargs)
        plt.title("Scalar field Values " + self.unit_values.strUnit())
        cb = plt.colorbar(fig, shrink=1, aspect=5)
        cb.set_label(self.get_unit_values().strUnit())
        # search for limits in case of masked field
        mask = self.get_comp('mask', raw=True)
        for i in np.arange(len(self.axe_x)):
            if not np.all(mask[:, i]):
                break
        xmin = self.axe_x[i]
        for i in np.arange(len(self.axe_x) - 1, -1, -1):
            if not np.all(mask[:, i]):
                break
        xmax = self.axe_x[i]
        for i in np.arange(len(self.axe_y)):
            if not np.all(mask[i, :]):
                break
        ymin = self.axe_y[i]
        for i in np.arange(len(self.axe_y) - 1, -1, -1):
            if not np.all(mask[i, :]):
                break
        ymax = self.axe_y[i]
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        return fig

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
        self.comp_x = np.array([])
        self.comp_y = np.array([])
        self.unit_values = make_unit('')

    def __neg__(self):
        tmpvf = self.get_copy()
        tmpvf.comp_x = -tmpvf.comp_x
        tmpvf.comp_y = -tmpvf.comp_y
        return tmpvf

    def __add__(self, other):
        if isinstance(other, VectorField):
            if (all(self.comp_x.axe_x != other.comp_x.axe_x) or
                    all(self.comp_x.axe_y != other.comp_x.axe_y)):
                raise ValueError("Vector fields have to be consistent "
                                 "(same dimensions)")
            try:
                self.unit_values + other.unit_values
                self.unit_x + other.unit_x
                self.unit_y + other.unit_y
            except:
                raise ValueError("I think these units don't match, fox")
            tmpvf = self.get_copy()
            tmpvf.comp_x = (self.comp_x
                            + other.comp_x*other.unit_values/self.unit_values)
            tmpvf.comp_y = (self.comp_y
                            + other.comp_y*other.unit_values/self.unit_values)
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
            tmpvf = self.get_copy()
            tmpvf.comp_x /= (number/self.unit_values).asNumber()
            tmpvf.comp_y /= (number/self.unit_values).asNumber()
            return tmpvf
        elif isinstance(number, NUMBERTYPES):
            tmpvf = self.get_copy()
            tmpvf.comp_x /= number
            tmpvf.comp_y /= number
            return tmpvf
        else:
            raise TypeError("You can only divide a vector field "
                            "by numbers")

    __div__ = __truediv__

    def __mul__(self, number):
        if isinstance(number, unum.Unum):
            tmpvf = self.get_copy()
            tmpvf.comp_x *= (number/self.unit_values).asNumber()
            tmpvf.comp_y *= (number/self.unit_values).asNumber()
            return tmpvf
        elif isinstance(number, NUMBERTYPES):
            tmpvf = self.get_copy()
            tmpvf.comp_x *= number
            tmpvf.comp_y *= number
            return tmpvf
        else:
            raise TypeError("You can only multiply a vector field "
                            "by numbers")

    __rmul__ = __mul__

    def __sqrt__(self):
        tmpvf = self.get_copy()
        tmpvf.comp_x = np.sqrt(tmpvf.comp_x)
        tmpvf.comp_y = np.sqrt(tmpvf.comp_y)
        return tmpvf

    def __pow__(self, number):
        if not isinstance(number, NUMBERTYPES):
            raise TypeError("You only can use a number for the power "
                            "on a Vectorfield")
        tmpvf = self.get_copy()
        tmpvf.comp_x = np.power(tmpvf.comp_x, number)
        tmpvf.comp_y = np.power(tmpvf.comp_y, number)
        return tmpvf

    def __iter__(self):
        try:
            maskx = self.comp_x.mask
            masky = self.comp_y.mask
            datax = self.comp_x.data
            datay = self.comp_y.data
        except AttributeError:
            datax = self.comp_x
            datay = self.comp_y
            maskx = np.zeros(datax.shape)
            masky = np.zeros(datay.shape)
        mask = np.logical_or(maskx, masky)
        for ij, xy in Field.__iter__(self):
            i = ij[0]
            j = ij[1]
            if not mask[j, i]:
                yield ij, xy, [datax[j, i], datay[j, i]]

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

    def import_from_arrays(self, axe_x, axe_y, comp_x, comp_y,
                           unit_x=make_unit(""), unit_y=make_unit(""),
                           unit_values=make_unit("")):
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
        unit_x : Unit object, optionnal
            Unit for the values of axe_x
        unit_y : Unit object, optionnal
            Unit for the values of axe_y
        unit_values : Unit object, optionnal
            Unit for the field components.
        """
        if not isinstance(axe_x, ARRAYTYPES):
            raise TypeError("'axe_x' must be an array")
        else:
            axe_x = np.array(axe_x)
        if not axe_x.ndim == 1:
            raise ValueError("'axe_x' must be a one dimension array")
        if not isinstance(axe_y, ARRAYTYPES):
            raise TypeError("'axe_y' must be an array")
        else:
            axe_y = np.array(axe_y)
        if not axe_y.ndim == 1:
            raise ValueError("'axe_y' must be a one dimension array")
        if not isinstance(comp_x, ARRAYTYPES):
            raise TypeError("'comp_x' must be an array")
        elif isinstance(comp_x, (list, tuple)):
            comp_x = np.array(comp_x)
        if not comp_x.ndim == 2:
            raise ValueError("'comp_x' must be a two dimension array")
        if not isinstance(comp_y, ARRAYTYPES):
            raise TypeError("'comp_y' must be an array")
        elif isinstance(comp_y, (list, tuple)):
            comp_y = np.array(comp_y)
        if not comp_y.ndim == 2:
            raise ValueError("'comp_y' must be a two dimension array")
        if unit_x is not None:
            if not isinstance(unit_x, unum.Unum):
                raise TypeError("'unit_x' must be an Unit object")
        if unit_y is not None:
            if not isinstance(unit_y, unum.Unum):
                raise TypeError("'unit_y' must be an Unit object")
        if unit_values is not None:
            if not isinstance(unit_values, unum.Unum):
                raise TypeError("'unit_values' must be an Unit object")
        if (comp_x.shape[0] != axe_y.shape[0] or
                comp_x.shape[1] != axe_x.shape[0]):
            raise ValueError("Dimensions of 'axe_x', 'axe_y' and 'comp_x' must"
                             " be consistents")
        if (comp_y.shape[0] != axe_y.shape[0] or
                comp_y.shape[1] != axe_x.shape[0]):
            raise ValueError("Dimensions of 'axe_x', 'axe_y' and 'comp_y' must"
                             " be consistents")
        self.axe_x = axe_x
        self.axe_y = axe_y
        self.comp_x = comp_x
        self.comp_y = comp_y
        self.unit_x = unit_x
        self.unit_y = unit_y
        self.unit_values = unit_values

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

    def get_unit_values(self):
        """
        Return the values unit
        """
        return self.unit_values

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

    def set_unit_values(self, unit_values=None):
        """
        Load unities into the scalar field values.

        Parameters
        ----------
        unit_values : Unit object
            Values unit.
        """
        if unit_values is not None:
            if not isinstance(unit_values, unum.Unum):
                raise TypeError("'unit_values' must be an Unit object")
            self.unit_values = unit_values

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

    def get_comp(self, componentname, raw=False):
        """
        Return a ScalarField object representing a component of the
        Vectorfield object.

        Parameters
        ----------
        componentname : string
            Can be 'x', 'y' or 'mask'.
        raw : boolean, optional
            If 'False' (default), return a ScalarField object,
            if 'True', return a masked array.

        Returns
        -------
        component : ScalarField object or numpy masked array
        """
        if not isinstance(componentname, STRINGTYPES):
            raise TypeError("'componentname' must be a string")
        if componentname == 'x':
            values = self.comp_x
        elif componentname == 'y':
            values = self.comp_y
        elif componentname == 'mask':
            values = np.logical_or(self.comp_x.mask, self.comp_y.mask)
        else:
            raise ValueError("unknown value of 'componentname'")
        if raw:
            return values
        else:
            axe_x, axe_y = self.get_axes()
            unit_x, unit_y = self.get_axe_units()
            unit_values = self.get_unit_values()
            tmpsf = ScalarField()
            tmpsf.import_from_arrays(axe_x, axe_y, values, unit_x, unit_y,
                                     unit_values)
            return tmpsf

    def set_comp(self, componentname, value):
        """
        Fill the component 'componentname' with 'value'.

        Parameters
        ----------
        componentname : string
            Can be 'x', 'y', 'mask'.
        value : array
            Array with the same shape as the initial component
        """
        if not isinstance(componentname, STRINGTYPES):
            raise TypeError("'componentname' must be a string")
        if not isinstance(value, ARRAYTYPES):
            raise TypeError("'value' must be an array")
        if not isinstance(value, np.ma.MaskedArray):
             value = np.ma.masked_array(value)
        if self.get_dim() != value.shape:
            raise ValueError("'value' dimensions are inconsistent with the "
                             "ScalarField shape")
        if componentname == 'x':
            self.comp_x = value
        elif componentname == 'y':
            self.comp_y = value
        elif componentname == 'mask':
            self.comp_x.mask = value
            self.comp_y.mask = value
        else:
            raise ValueError("Unknown 'componentname' value")

    def get_copy(self):
        """
        Return a copy of the vectorfield.
        """
        return copy.deepcopy(self)

    def get_dim(self):
        """
        Return the vector field dimensions.

        Returns
        -------
        shape : tuple
            Tuple of the dimensions (along X and Y) of the scalar field.
        """
        return self.comp_x.shape

    def get_min(self, unit=False):
        """
        Return the minima of the magnitude of the field.

        Parameters
        ----------
        unit : boolean, optinal
            If True, a unit object is returned,
            else (default), a float is returned.

        Returns
        -------
        mini : float
            Minima on the field
        """
        return self.get_magnitude().get_min(unit)

    def get_max(self, unit=False):
        """
        Return the maxima of the magnitude of the field.

        Parameters
        ----------
        unit : boolean, optinal
            If True, a unit object is returned,
            else (default), a float is returned.

        Returns
        -------
        maxi: float
            Maxima on the field
        """
        return self.get_magnitude().get_max(unit)

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
            return self.get_comp('x').get_profile(direction, position)
        elif component == 2:
            return self.get_comp('y').get_profile(direction, position)
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
#            return self.get_comp('y').get_bl(value, direction, rel, kind)

    def get_streamlines(self, xy, delta=.25, interp='linear',
                        reverse_direction=False):
        """
        Return a tuples of Points object representing the streamline begining
        at the points specified in xy.

        Parameters
        ----------
        xy : tuple
            Tuple containing each starting point for streamline.
        delta : number, optional
            Spatial discretization of the stream lines,
            relative to a the spatial discretization of the field.
        interp : string, optional
            Used interpolation for streamline computation.
            Can be 'linear'(default) or 'cubic'
        reverse_direction : boolean, optional
            If True, the streamline goes upstream.
        """
        if not isinstance(xy, ARRAYTYPES):
            raise TypeError("'xy' must be a tuple of arrays")
        xy = np.array(xy)
        if xy.shape == (2,):
            xy = [xy]
        elif len(xy.shape) == 2 and xy.shape[1] == 2:
            pass
        else:
            raise ValueError("'xy' must be a tuple of arrays")
        axe_x, axe_y = self.get_axes()
        Vx, Vy = self.get_comp('x', raw=True), self.get_comp('y', raw=True)
        Vx = Vx.flatten()
        Vy = Vy.flatten()
        if not isinstance(delta, NUMBERTYPES):
            raise TypeError("'delta' must be a number")
        if delta <= 0:
            raise ValueError("'delta' must be positive")
        if delta > len(axe_x) or delta > len(axe_y):
            raise ValueError("'delta' is too big !")
        deltaabs = delta * ((axe_x[-1]-axe_x[0])/len(axe_x)
                            + (axe_y[-1]-axe_y[0])/len(axe_y))/2
        deltaabs2 = deltaabs**2
        if not isinstance(interp, STRINGTYPES):
            raise TypeError("'interp' must be a string")
        if not (interp == 'linear' or interp == 'cubic'):
            raise ValueError("Unknown interpolation method")
        # interpolation lineaire du champ de vitesse
        a, b = np.meshgrid(axe_x, axe_y)
        a = a.flatten()
        b = b.flatten()
        pts = zip(a, b)
        if interp == 'linear':
            interp_vx = spinterp.LinearNDInterpolator(pts, Vx.flatten())
            interp_vy = spinterp.LinearNDInterpolator(pts, Vy.flatten())
        elif interp == 'cubic':
            interp_vx = spinterp.CloughTocher2DInterpolator(pts, Vx.flatten())
            interp_vy = spinterp.CloughTocher2DInterpolator(pts, Vy.flatten())

        # Calcul des streamlines
        streams = []
        longmax = (len(axe_x)+len(axe_y))/delta
        for coord in xy:
            x = coord[0]
            y = coord[1]
            x = float(x)
            y = float(y)
            # si le points est en dehors, on ne fait rien
            if x < axe_x[0] or x > axe_x[-1] or y < axe_y[0] or y > axe_y[-1]:
                continue
            # calcul d'une streamline
            stream = np.zeros((longmax, 2))
            stream[0, :] = [x, y]
            i = 1
            while True:
                tmp_vx = interp_vx(stream[i-1, 0], stream[i-1, 1])
                tmp_vy = interp_vy(stream[i-1, 0], stream[i-1, 1])
                # tests d'arret
                if np.isnan(tmp_vx) or np.isnan(tmp_vy):
                    break
                if i >= longmax-1:
                    break
                if i > 15:
                    norm = [stream[0:i-10, 0] - stream[i-1, 0],
                            stream[0:i-10, 1] - stream[i-1, 1]]
                    norm = norm[0]**2 + norm[1]**2
                    if any(norm < deltaabs2/2):
                        break
                # calcul des dx et dy
                norm = (tmp_vx**2 + tmp_vy**2)**(.5)
                if reverse_direction:
                    norm = -norm
                dx = tmp_vx/norm*deltaabs
                dy = tmp_vy/norm*deltaabs
                stream[i, :] = [stream[i-1, 0] + dx, stream[i-1, 1] + dy]
                i += 1
            stream = stream[:i-1]
            pts = Points(stream, unit_x=self.comp_x.unit_x,
                         unit_y=self.comp_y.unit_y,
                         name='streamline at x={:.3f}, y={:.3f}'.format(x, y))
            streams.append(pts)
        if len(streams) == 0:
            return None
        elif len(streams) == 1:
            return streams[0]
        else:
            return streams

    def get_tracklines(self, xy, delta=.25, interp='linear',
                       reverse_direction=False):
        """
        Return a tuples of Points object representing the trackline begining
        at the points specified in xy.
        A trackline follow the general direction of the vectorfield
        (as a streamline), but favor the small velocity. This behavior allow
        the track following.

        Parameters
        ----------
        xy : tuple
            Tuple containing each starting point for streamline.
        delta : number, optional
            Spatial discretization of the tracklines,
            relative to a the spatial discretization of the field.
        interp : string, optional
            Used interpolation for trackline computation.
            Can be 'linear'(default) or 'cubic'
        """
        if not isinstance(xy, ARRAYTYPES):
            raise TypeError("'xy' must be a tuple of arrays")
        xy = np.array(xy)
        if xy.shape == (2,):
            xy = [xy]
        elif len(xy.shape) == 2 and xy.shape[1] == 2:
            pass
        else:
            raise ValueError("'xy' must be a tuple of arrays")
        axe_x, axe_y = self.get_axes()
        Vx, Vy = self.get_comp('x', raw=True), self.get_comp('y', raw=True)
        Vx = Vx.flatten()
        Vy = Vy.flatten()
        Magn = self.get_magnitude().get_comp('values').flatten()
        if not isinstance(delta, NUMBERTYPES):
            raise TypeError("'delta' must be a number")
        if delta <= 0:
            raise ValueError("'delta' must be positive")
        if delta > len(axe_x) or delta > len(axe_y):
            raise ValueError("'delta' is too big !")
        deltaabs = delta * ((axe_x[-1]-axe_x[0])/len(axe_x)
                            + (axe_y[-1]-axe_y[0])/len(axe_y))/2
        deltaabs2 = deltaabs**2
        if not isinstance(interp, STRINGTYPES):
            raise TypeError("'interp' must be a string")
        if not (interp == 'linear' or interp == 'cubic'):
            raise ValueError("Unknown interpolation method")
        # interpolation lineaire du champ de vitesse
        a, b = np.meshgrid(axe_x, axe_y)
        a = a.flatten()
        b = b.flatten()
        pts = zip(a, b)
        if interp == 'linear':
            interp_vx = spinterp.LinearNDInterpolator(pts, Vx.flatten())
            interp_vy = spinterp.LinearNDInterpolator(pts, Vy.flatten())
            interp_magn = spinterp.LinearNDInterpolator(pts, Magn.flatten())
        elif interp == 'cubic':
            interp_vx = spinterp.CloughTocher2DInterpolator(pts, Vx.flatten())
            interp_vy = spinterp.CloughTocher2DInterpolator(pts, Vy.flatten())
            interp_magn = spinterp.CloughTocher2DInterpolator(pts,
                                                              Magn.flatten())
        # Calcul des streamlines
        streams = []
        longmax = (len(axe_x)+len(axe_y))/delta
        for coord in xy:
            x = coord[0]
            y = coord[1]
            x = float(x)
            y = float(y)
            # si le points est en dehors, on ne fait rien
            if x < axe_x[0] or x > axe_x[-1] or y < axe_y[0] or y > axe_y[-1]:
                continue
            # calcul d'une streamline
            stream = np.zeros((longmax, 2))
            stream[0, :] = [x, y]
            i = 1
            while True:
                tmp_vx = interp_vx(stream[i-1, 0], stream[i-1, 1])
                tmp_vy = interp_vy(stream[i-1, 0], stream[i-1, 1])
                # tests d'arret
                if np.isnan(tmp_vx) or np.isnan(tmp_vy):
                    break
                if i >= longmax-1:
                    break
                if i > 15:
                    norm = [stream[0:i-10, 0] - stream[i-1, 0],
                            stream[0:i-10, 1] - stream[i-1, 1]]
                    norm = norm[0]**2 + norm[1]**2
                    if any(norm < deltaabs2/2):
                        break
                # searching the lower velocity in an angle of pi/2 in the
                # flow direction
                x_tmp = stream[i-1, 0]
                y_tmp = stream[i-1, 1]
                norm = (tmp_vx**2 + tmp_vy**2)**(.5)
                #    finding 10 possibles positions in the flow direction
                angle_op = np.pi/4
                theta = np.linspace(-angle_op/2, angle_op/2, 10)
                e_x = tmp_vx/norm
                e_y = tmp_vy/norm
                dx = deltaabs*(e_x*np.cos(theta) - e_y*np.sin(theta))
                dy = deltaabs*(e_x*np.sin(theta) + e_y*np.cos(theta))
                #    finding the step giving the lower velocity
                magn = interp_magn(x_tmp + dx, y_tmp + dy)
                ind_low_magn = np.argmin(magn)
                #    getting the final steps (ponderated)
                theta = theta[ind_low_magn]/2.
                dx = deltaabs*(e_x*np.cos(theta) - e_y*np.sin(theta))
                dy = deltaabs*(e_x*np.sin(theta) + e_y*np.cos(theta))
                # calcul des dx et dy
                stream[i, :] = [x_tmp + dx, y_tmp + dy]
                i += 1
            stream = stream[:i-1]
            pts = Points(stream, unit_x=self.comp_x.unit_x,
                         unit_y=self.comp_y.unit_y,
                         name='streamline at x={:.3f}, y={:.3f}'.format(x, y))
            streams.append(pts)
        if len(streams) == 0:
            return None
        elif len(streams) == 1:
            return streams[0]
        else:
            return streams

    def get_magnitude(self):
        """
        Return a scalar field with the velocity field magnitude.
        """
        comp_x, comp_y = (self.get_comp('x', raw=True),
                          self.get_comp('y', raw=True))
        unit_x, unit_y = self.get_axe_units()
        values = np.sqrt(comp_x**2 + comp_y**2)
        unit_values = (unit_x**2 + unit_y**2)**(.5)
        magn = self.get_comp('x')
        magn.set_comp('values', values)
        magn.set_unit_values(unit_values)
        return magn

    def get_shear_stress(self):
        """
        Return a vector field with the shear stress
        """
        # Getting gradients and axes
        axe_x, axe_y = self.get_axes()
        comp_x, comp_y = (self.get_comp('x', raw=True),
                          self.get_comp('y', raw=True))
        dx = axe_x[1] - axe_x[0]
        dy = axe_y[1] - axe_y[0]
        du_dy, _ = np.gradient(comp_x, dy, dx)
        _, dv_dx = np.gradient(comp_y, dy, dx)
        # swirling vectors matrix
        comp_x = dv_dx
        comp_y = du_dy
        # creating vectorfield object
        tmpvf = self.get_copy()
        tmpvf.set_comp('x', comp_x)
        tmpvf.set_comp('y', comp_y)
        return tmpvf

    def get_vorticity(self):
        """
        Return a scalar field with the z component of the vorticity.
        """
        axe_x, axe_y = self.get_axes()
        comp_x, comp_y = (self.get_comp('x', raw=True),
                          self.get_comp('y', raw=True))
        dx = axe_x[1] - axe_x[0]
        dy = axe_y[1] - axe_y[0]
        _, Exy = np.gradient(comp_x, dy, dx)
        Eyx, _ = np.gradient(comp_y, dy, dx)
        vort = Eyx - Exy
        vort_sf = self.get_comp('x').get_copy()
        vort_sf.set_comp('values', vort)
        return vort_sf

    def get_swirling_strength(self):
        """
        Return a scalar field with the swirling strength
        (imaginary part of the eigenvalue of the velocity laplacian matrix)
        """
        # Getting gradients and axes
        axe_x, axe_y = self.get_axes()
        comp_x, comp_y = (self.get_comp('x', raw=True),
                          self.get_comp('y', raw=True))
        dx = axe_x[1] - axe_x[0]
        dy = axe_y[1] - axe_y[0]
        du_dy, du_dx = np.gradient(comp_x, dy, dx)
        dv_dy, dv_dx = np.gradient(comp_y, dy, dx)
        # swirling stregnth matrix
        swst = np.zeros(self.comp_x.values.shape)
        mask = np.logical_or(np.logical_or(du_dx.mask, du_dy.mask),
                             np.logical_or(dv_dx.mask, dv_dy.mask))
        # loop on  points
        for i in np.arange(0, len(axe_y)):
            for j in np.arange(0, len(axe_x)):
                if not mask[i, j]:
                    lapl = [[du_dx[i, j], du_dy[i, j]],
                            [dv_dx[i, j], dv_dy[i, j]]]
                    eigvals = np.linalg.eigvals(lapl)
                    swst[i, j] = np.max(np.imag(eigvals))
        # creating ScalarField object
        swst = np.ma.masked_array(swst, mask)
        tmpsf = self.get_comp('x').get_copy()
        ### TODO : implementer unité swst
        tmpsf.set_comp('values', swst)
        return tmpsf

    def get_swirling_vector(self):
        """
        Return a scalar field with the swirling vectors
        (eigenvectors of the velocity laplacian matrix
        ponderated by eigenvalues)
        (Have to be adjusted : which part of eigenvalues
        and eigen vectors take ?)
        """
        # Getting gradients and axes
        axe_x, axe_y = self.get_axes()
        comp_x, comp_y = (self.get_comp('x', raw=True),
                          self.get_comp('y', raw=True))
        dx = axe_x[1] - axe_x[0]
        dy = axe_y[1] - axe_y[0]
        du_dy, du_dx = np.gradient(comp_x, dy, dx)
        dv_dy, dv_dx = np.gradient(comp_y, dy, dx)
        # swirling vectors matrix
        comp_x = np.zeros(self.comp_x.values.shape)
        comp_y = np.zeros(self.comp_x.values.shape)
        mask = np.logical_or(np.logical_or(du_dx.mask, du_dy.mask),
                             np.logical_or(dv_dx.mask, dv_dy.mask))
        # loop on  points
        for i in np.arange(0, len(axe_y)):
            for j in np.arange(0, len(axe_x)):
                if not mask[i, j]:
                    lapl = [[du_dx[i, j], du_dy[i, j]],
                            [dv_dx[i, j], dv_dy[i, j]]]
                    eigvals, eigvect = np.linalg.eig(lapl)
                    eigvals = np.imag(eigvals)
                    eigvect = np.imag(eigvect)
                    if eigvals[0] > eigvals[1]:
                        comp_x[i, j] = eigvect[0][0]
                        comp_y[i, j] = eigvect[0][1]
                    else:
                        comp_x[i, j] = eigvect[1][0]
                        comp_y[i, j] = eigvect[1][1]
        # creating vectorfield object
        comp_x = np.ma.masked_array(comp_x, mask)
        comp_y = np.ma.masked_array(comp_y, mask)
        tmpvf = self.get_copy()
        tmpvf.set_comp('x', comp_x)
        tmpvf.set_comp('y', comp_y)
        return tmpvf

    def get_theta(self, low_velocity_filter=0.):
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
        Vx, Vy = self.get_comp('x', raw=True), self.get_comp('y', raw=True)
        Vx = Vx.data
        Vy = Vy.data
        mask = self.get_comp('mask', raw=True)
        theta = np.zeros(self.get_dim())
        # getting angle
        norm = np.sqrt(Vx**2 + Vy**2)
        if low_velocity_filter != 0:
            mask_lvf = norm < np.max(norm)*low_velocity_filter
            mask = np.logical_or(mask, mask_lvf)
        tmp_mask = np.logical_and(norm != 0, ~mask)
        theta[tmp_mask] = Vx[tmp_mask]/norm[tmp_mask]
        theta[tmp_mask] = np.arccos(theta[tmp_mask])
        theta[Vy < 0] = 2*np.pi - theta[Vy < 0]
        theta = np.ma.masked_array(theta, mask)
        tmpsf = self.get_comp('x').get_copy()
        tmpsf.set_comp('values', theta)
        return tmpsf

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
        Vx, Vy = self.get_comp('x', raw=True), self.get_comp('y', raw=True)
        mask = self.get_comp('mask', raw=True)
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
        Vx = np.ma.masked_array(Vx, mask)
        Vy = np.ma.masked_array(Vy, mask)
        self.set_comp('x', Vx)
        self.set_comp('y', Vy)

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
        mask = self.get_comp('mask', raw=True)
        # if there is nothing to do...
        if not np.any(mask):
            pass
        elif tof == 'interplin':
            inds_x = np.arange(self.get_dim()[1])
            inds_y = np.arange(self.get_dim()[0])
            grid_x, grid_y = np.meshgrid(inds_x, inds_y)
            Vx, Vy = self.get_comp('x', raw=True), self.get_comp('y', raw=True)
            Vx = Vx.data
            Vy = Vy.data
            fx = spinterp.interp2d(grid_y[~mask],
                                   grid_x[~mask],
                                   Vx[~mask])
            fy = spinterp.interp2d(grid_y[~mask],
                                   grid_x[~mask],
                                   Vy[~mask])
            for inds, masked in np.ndenumerate(mask):
                if masked:
                    Vx[inds[0], inds[1]] = fx(inds[0], inds[1])
                    Vy[inds[0], inds[1]] = fy(inds[0], inds[1])
            mask = np.zeros(Vx.shape)
            self.set_comp('x', Vx)
            self.set_comp('y', Vy)
        elif tof == 'interpcub':
            inds_x = np.arange(self.values.shape[1])
            inds_y = np.arange(self.values.shape[0])
            grid_x, grid_y = np.meshgrid(inds_x, inds_y)
            Vx, Vy = self.get_comp('x', raw=True), self.get_comp('y', raw=True)
            Vx = Vx.data
            Vy = Vy.data
            fx = spinterp.interp2d(grid_y[~mask],
                                   grid_x[~mask],
                                   Vx[~mask], kind='cubic')
            fy = spinterp.interp2d(grid_y[~mask],
                                   grid_x[~mask],
                                   Vy[~mask], kind='cubic')
            for inds, masked in np.ndenumerate(mask):
                if masked:
                    Vx[inds[0], inds[1]] = fx(inds[0], inds[1])
                    Vy[inds[0], inds[1]] = fy(inds[0], inds[1])
            mask = np.zeros(Vx.shape)
            self.set_comp('x', Vx)
            self.set_comp('y', Vy)
        elif tof == 'value':
            Vx, Vy = self.get_comp('x', raw=True), self.get_comp('y', raw=True)
            Vx[mask] = value
            Vy[mask] = value
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
        tmpsf = self.get_copy()
        tmpsf.axe_x = tmpsf.axe_x[indmin_x:indmax_x + 1]
        tmpsf.axe_y = tmpsf.axe_y[indmin_y:indmax_y + 1]
        tmpsf.comp_x = tmpsf.comp_x[indmin_y:indmax_y + 1,
                                    indmin_x:indmax_x + 1]
        tmpsf.comp_y = tmpsf.comp_y[indmin_y:indmax_y + 1,
                                    indmin_x:indmax_x + 1]
        return tmpsf

    def crop_masked_border(self):
        """
        Crop the masked border of the field in place.
        """
        axe_x, axe_y = self.get_axes()
        # checking masked values presence
        Vx, Vy = self.get_comp('x', raw=True), self.get_comp('y', raw=True)
        if not (np.ma.is_masked(Vx), np.ma.is_masked(Vy)):
            return None
        # getting datas
        Vx = Vx.data
        Vy = Vy.data
        mask = self.get_comp('mask', raw=True)
        # crop border along y
        axe_y_m = ~np.all(mask, axis=1)
        if np.any(axe_y_m):
            Vx = Vx[axe_y_m, :]
            Vy = Vy[axe_y_m, :]
            mask = mask[axe_y_m, :]
        # crop values along x
        axe_x_m = ~np.all(mask, axis=0)
        if np.any(axe_x_m):
            Vx = Vx[:, axe_x_m]
            Vy = Vy[:, axe_x_m]
            mask = mask[:, axe_x_m]
        # storing cropped values
        self.comp_x = np.ma.masked_array(Vx, mask)
        self.comp_y = np.ma.masked_array(Vy, mask)
        # crop axis
        self.axe_x = axe_x[axe_x_m]
        self.axe_y = axe_y[axe_y_m]

    def _display(self, component=None, kind=None, **plotargs):
        if kind is not None:
            if not isinstance(kind, STRINGTYPES):
                raise TypeError("'kind' must be a string")
        axe_x, axe_y = self.get_axes()
        if component is None:
            Vx, Vy = self.get_comp('x', raw=True), self.get_comp('y', raw=True)
            magn = self.get_magnitude().get_comp('values', raw=True)
            unit_x, unit_y = self.get_axe_units()
            if kind == 'stream':
                if not 'color' in plotargs.keys():
                    plotargs['color'] = magn.data
                fig = plt.streamplot(axe_x, axe_y, Vx, Vy, **plotargs)
            elif kind == 'quiver' or kind is None:
                if 'C' in plotargs.keys():
                    C = plotargs.pop('C')
                    if not (C == 0 or C is None):
                        fig = plt.quiver(axe_x, axe_y, Vx, Vy, C, **plotargs)
                    else:
                        fig = plt.quiver(axe_x, axe_y, Vx, Vy, **plotargs)
                else:
                    fig = plt.quiver(axe_x, axe_y, Vx, Vy, magn, **plotargs)
            else:
                raise ValueError("Unknown value of 'kind'")
        elif component == "x":
            if kind == '3D':
                fig = self.get_comp('x').Display3D()
            else:
                fig = self.get_comp('x')._display(kind)
        elif component == "y":
            if kind == '3D':
                fig = self.get_comp('y').Display3D()
            else:
                fig = self.get_comp('y')._display(kind)
        else:
            raise TypeError("Unknown value of 'component'")
        return fig

    def display(self, component=None, kind=None, **plotargs):
        """
        Display something from the vector field.
        If component is not given, a quiver is displayed.
        If component is an integer, the coresponding component of the field is
        displayed.

        Parameters
        ----------
        component : string, optional
            Component to display, can be 'x', 'y', 'mask'
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
        fig = self._display(component, kind, **plotargs)
        unit_x, unit_y = self.get_axe_units()
        unit_values = self.get_unit_values()
        Vx, Vy = self.get_comp('x', raw=True), self.get_comp('y', raw=True)
        if component is None:
            if kind == 'quiver' or kind is None:
                cb = plt.colorbar()
                cb.set_label("Magnitude " + unit_values.strUnit())
                legendarrow = round(np.max([Vx.max(), Vy.max()]))
                plt.quiverkey(fig, 1.075, 1.075, legendarrow,
                              "$" + str(legendarrow)
                              + unit_values.strUnit() + "$",
                              labelpos='W', fontproperties={'weight': 'bold'})
            plt.title("Values " + unit_values.strUnit())
        elif component == 'x':
            plt.title("Vx " + unit_values.strUnit())
        elif component == 'y':
            plt.title("Vy " + unit_values.strUnit())
        else:
            raise ValueError("Unknown 'component' value")
        return fig


class VelocityField(object):
    """
    Class representing a velocity field and all its derived fields.
    Contrary to the 'VectorField' class, here the derived fields are stocked
    in the object.

    Principal methods
    -----------------
    "import_from_*" : allows to easily create or import velocity fields.

    "export_to_*" : allows to export.

    "display" : display the vector field, with these unities.
    """
    def __init__(self):
        pass

    def __neg__(self):
        final_vf = VelocityField()
        final_vf.import_from_vectorfield(-self.V, self.time, self.unit_time)
        return final_vf

    def __add__(self, other):
        if isinstance(other, VelocityField):
            if (all(self.V.comp_x.axe_x != other.V.comp_x.axe_x) or
                    all(self.V.comp_x.axe_y != other.V.comp_x.axe_y)):
                raise ValueError("Vector fields have to be consistent "
                                 "(same dimensions)")
            try:
                self.V.comp_x.unit_values + other.V.comp_x.unit_values
                self.V.comp_x.unit_x + other.V.comp_x.unit_x
                self.V.comp_x.unit_y + other.V.comp_x.unit_y
                self.unit_time + other.unit_time
            except:
                raise ValueError("I think these units don't match, fox")
            time = (self.time + other.time)/2
            final_vf = VelocityField()
            final_vf.import_from_vectorfield(self.V + other.V, time,
                                             self.unit_time)
            return final_vf
        else:
            raise TypeError("You can only add a velocity field "
                            "with others velocity fields")

    def __sub__(self, other):
        tmp_other = other.__neg__()
        tmp_sf = self.__add__(tmp_other)
        return tmp_sf

    def __truediv__(self, number):
        if isinstance(number, NUMBERTYPES):
            final_vf = VelocityField()
            final_vf.import_from_vectorfield(self.V/number, self.time,
                                             self.unit_time)
            return final_vf
        if isinstance(number, unum.Unum):
            final_vf = VelocityField()
            final_vf.import_from_vectorfield(self.V/number, self.time,
                                             self.unit_time)
            return final_vf
        else:
            raise TypeError("You can only divide a velocity field "
                            "by numbers")

    __div__ = __truediv__

    def __mul__(self, number):
        if isinstance(number, NUMBERTYPES):
            final_vf = VelocityField()
            final_vf.import_from_vectorfield(self.V*number, self.time,
                                             self.unit_time)
            return final_vf
        if isinstance(number, unum.Unum):
            final_vf = VelocityField()
            final_vf.import_from_vectorfield(self.V*number, self.time,
                                             self.unit_time)
            return final_vf
        else:
            raise TypeError("You can only multiply a velocity field "
                            "by numbers")

    __rmul__ = __mul__

    def __sqrt__(self):
        final_vf = VelocityField()
        final_vf.import_from_vectorfield(np.sqrt(self.V), time=self.time,
                                         unit_time=self.unit_time)
        return final_vf

    def __pow__(self, number):
        if not isinstance(number, NUMBERTYPES):
            raise TypeError("You only can use a number for the power "
                            "on a Vectorfield")
        final_vf = VelocityField()
        final_vf.import_from_vectorfield(np.power(self.V, number),
                                         time=self.time,
                                         unit_time=self.unit_time)
        return final_vf

    def import_from_davis(self, filename, time=0, unit_time=make_unit("s")):
        """
        Import a velocity field from a .VC7 file.

        Parameters
        ----------
        filename : string
            Path to the file to import.
        """
        # entry tests
        if not (isinstance(time, NUMBERTYPES)):
            raise TypeError("'time' must be a number")
        if not isinstance(unit_time, unum.Unum):
            raise TypeError("'unit_time' must be a Unit object")
        if not isinstance(filename, STRINGTYPES):
            raise TypeError("'filename' must be a string")
        if not os.path.exists(filename):
            raise ValueError("'filename' must ne an existing file")
        if os.path.isdir(filename):
            filename = glob.glob(os.path.join(filename, '*.vc7'))[0]
        _, ext = os.path.splitext(filename)
        if not (ext == ".vc7" or ext == ".VC7"):
            raise ValueError("'filename' must be a vc7 file")
        # cleaning in case values are already been set
        self._clear_derived()
        # vc7 file importation
        v = IM.VC7(filename)
        # units treatment
        unit_x = v.buffer['scaleX']['unit'].split("\x00")[0]
        unit_x = unit_x.replace('[', '')
        unit_x = unit_x.replace(']', '')
        unit_y = v.buffer['scaleY']['unit'].split("\x00")[0]
        unit_y = unit_y.replace('[', '')
        unit_y = unit_y.replace(']', '')
        unit_values = v.buffer['scaleI']['unit'].split("\x00")[0]
        unit_values = unit_values.replace('[', '')
        unit_values = unit_values.replace(']', '')
        # axis correction
        x = v.Px[0, :]
        y = v.Py[:, 0]
        Vx = v.Vx[0]
        Vy = v.Vy[0]
        if x[-1] < x[0]:
            x = x[::-1]
            Vx = Vx[:, ::-1]
            Vy = Vy[:, ::-1]
        if y[-1] < y[0]:
            y = y[::-1]
            Vx = Vx[::-1, :]
            Vy = Vy[::-1, :]
        comp_x = ScalarField()
        comp_y = ScalarField()
        comp_x.import_from_arrays(x, y, Vx,
                                  make_unit(unit_x),
                                  make_unit(unit_y),
                                  make_unit(unit_values))
        comp_y.import_from_arrays(x, y, Vy,
                                  make_unit(unit_x),
                                  make_unit(unit_y),
                                  make_unit(unit_values))
        self.V = VectorField()
        self.V.import_from_scalarfields(comp_x, comp_y)
        timefile = v.attributes['_TIME'].split(':')
        timefile = float(timefile[0])*3600 + float(timefile[1])*60\
            + float(timefile[2])
        self.time = time
        self.unit_time = unit_time

    def import_from_ascii(self, filename, x_col=1, y_col=2, vx_col=3,
                          vy_col=4, unit_x=make_unit(""), unit_y=make_unit(""),
                          unit_values=make_unit(""), time=0,
                          unit_time=make_unit('s'), **kwargs):
        """
        Import a velocityfield from an ascii file.

        Parameters
        ----------
        x_col, y_col, vx_col, vy_col : integer, optional
            Colonne numbers for the given variables
            (begining at 1).
        unit_x, unit_y, unit_v : Unit objects, optional
            Unities for the given variables.
        time : number, optional
            Time of the instantaneous field.
        unit_time : Unit object, optional
            Time unit, 'second' by default.
        **kwargs :
            Possibles additional parameters are the same as those used in the
            numpy function 'genfromtext()' :
            'delimiter' to specify the delimiter between colonnes.
            'skip_header' to specify the number of colonne to skip at file
                begining
            ...
        """
        VF = VectorField()
        VF.import_from_ascii(filename, x_col, y_col, vx_col, vy_col, unit_x,
                             unit_y, unit_values, **kwargs)
        self.import_from_vectorfield(VF, time, unit_time)

    def import_from_matlab(self, filename, time, unit_time=make_unit("s")):
        """
        Import a velocityfield from a maltab file.
        """
        pass

    def import_from_scalarfields(self, comp_x, comp_y, time,
                                 unit_time=make_unit("s")):
        """
        Import a velocity field from two scalarfields.

        Parameters
        ----------
        CompX : ScalarField
            X component of the vector field.
        CompY : ScalarField
            Y component of the vector field.
        time : float
            Time when the velocity field has been taken
        unit_time : Unit object, optional
            Unit for the specified time (default is 'second')
        """
        if not isinstance(time, NUMBERTYPES):
            raise TypeError("'time' must be a number")
        if not isinstance(unit_time, unum.Unum):
            raise TypeError("'unit_time' must be a Unit object")
        # cleaning in case values are already been set
        self._clear_derived()
        self.V = VectorField()
        self.V.import_from_scalarfields(comp_x, comp_y)
        self.time = time
        self.unit_time = unit_time.copy()

    def import_from_vectorfield(self, vectorfield, time=0,
                                unit_time=make_unit('s')):
        """
        Import a velocity field from a vectorfield.

        Parameters
        ----------
        vectorfield : VectorField object
            The vector field to import.
        time : float, optional
            Time when the velocity field has been taken
        unit_time : Unit object, optional
            Unit for the specified time (default is 'second')
        """
        if not isinstance(vectorfield, VectorField):
            raise TypeError("'vectorfield' must be a VectorField object")
        if not isinstance(time, NUMBERTYPES):
            raise TypeError("'time' must be a number")
        if not isinstance(unit_time, unum.Unum):
            raise TypeError("'unit_time' must be a Unit object")
        # cleaning in case values are already been set
        self._clear_derived()
        self.V = vectorfield.get_copy()
        self.time = time
        self.unit_time = unit_time.copy()

    def import_from_velocityfield(self, velocityfield):
        """
        Import a velocity field from another velocityfield.

        Parameters
        ----------
        velocityfield : velocityField object
            The velocity field to copy.
        """
        if not isinstance(velocityfield, VelocityField):
            raise TypeError("'velocityfield' must be a VelocityField object")
        # cleaning in case values are already been set
        self._clear_derived()
        self.V = velocityfield.V.get_copy()
        self.time = velocityfield.time*1
        self.unit_time = velocityfield.unit_time.copy()

    def import_from_file(self, filepath, **kw):
        """
        Load a VelocityField object from the specified file using the JSON
        format.
        Additionnals arguments for the JSON decoder may be set with the **kw
        argument. Such as'encoding' (to change the file
        encoding, default='utf-8').

        Parameters
        ----------
        filepath : string
            Path specifiing the VelocityField to load.
        """
        # cleaning in case values are already been set
        import IMTreatment.io.io as imtio
        self._clear_derived()
        tmp_vf = imtio.import_from_file(filepath, **kw)
        if tmp_vf.__classname__ != self.__classname__:
            raise IOError("This file do not contain a VelocityField, cabron.")
        self.import_from_velocityfield(tmp_vf)

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

    def _clear_derived(self):
        """
        Delete all the derived fields, in case of changement in the base
        fields.
        """
        attributes = dir(self)
        for attr in attributes:
            # check if the attribute is a component
            try:
                self.get_comp(attr)
            except ValueError:
                pass
            else:
                # check if the attribute is the base field
                if attr not in ['V', 'time', 'unit_time']:
                    del self.__dict__[attr]

    def get_comp(self, componentname):
        """
        Return a reference to the component designed by 'componentname'.

        Parameters
        ----------
        componentname : string
            Name of the component.

        Returns
        -------
        component : ScalarField
            Reference to the velocity field component.
        """
        if not isinstance(componentname, STRINGTYPES):
            raise TypeError("'componentname' must be a string")
        if componentname == "V":
            return self.V
        if componentname == "mask":
            tmp_mask = self.V.comp_x.get_copy()
            if isinstance(self.V.comp_x.values, np.ma.MaskedArray):
                tmp_mask.values = self.V.comp_x.values.mask
            else:
                tmp_mask.values = np.zeros(tmp_mask.values.shape)
            return tmp_mask
        elif componentname == "Vx":
            return self.V.comp_x
        elif componentname == "Vy":
            return self.V.comp_y
        elif componentname == "magnitude":
            try:
                return self.magnitude
            except AttributeError:
                self.calc_magnitude()
                return self.magnitude
        elif componentname == "theta":
            try:
                return self.theta
            except AttributeError:
                self.theta = self.V.get_theta()
                return self.theta
        elif componentname == "gamma1":
            try:
                return self.gamma1
            except AttributeError:
                self.calc_gamma1()
                return self.gamma1
        elif componentname == "gamma2":
            try:
                return self.gamma2
            except AttributeError:
                self.calc_gamma2()
                return self.gamma2
        elif componentname == "kappa1":
            try:
                return self.kappa1
            except AttributeError:
                self.calc_kappa1()
                return self.kappa1
        elif componentname == "kappa2":
            try:
                return self.kappa2
            except AttributeError:
                self.calc_kappa2()
                return self.kappa2
        elif componentname == "iota":
            try:
                return self.iota
            except AttributeError:
                self.calc_iota()
                return self.iota
        elif componentname == "qcrit":
            try:
                return self.qcrit
            except AttributeError:
                self.calc_q_criterion()
                return self.qcrit
        elif componentname == "swirling_strength":
            try:
                return self.swirling_strength
            except AttributeError:
                self.calc_swirling_strength()
                return self.swirling_strength
        elif componentname == "sigma":
            try:
                return self.sigma
            except AttributeError:
                self.calc_sigma()
                return self.sigma
        elif componentname == "time":
            return self.time
        elif componentname == "unit_time":
            return self.unit_time
        else:
            raise ValueError("'componentname' must be a known component ({0} "
                             "is actually unknown)".format(componentname))

#    def set_comp(self, componentname, scalarfield):
#        """
#        Set the velocity field component to the given scalarfield.
#
#        Parameters
#        ----------
#        componentname : string
#            Name of the component to replace.
#        scalarfield : ScalarField object
#            Scalarfield to set in.
#
#        """
#        if not isinstance(scalarfield, ScalarField):
#            raise TypeError("'scalarfield' must be a ScalarField object")
#        if not (scalarfield.get_dim() == self.V.comp_x.get_dim()):
#            raise ValueError("'scalarfield' must have the same dimensions "
#                             "than the velocity field")
#        compo = self.get_comp(componentname)
#        compo.import_from_scalarfields(scalarfield)

    def set_axes(self, axe_x=None, axe_y=None):
        """
        Load new axes in the velocity field.

        Parameters
        ----------
        axe_x : array, optional
            One-dimensionale array representing the position of the
            values along the X axe.
        axe_y : array, optional
            idem for the Y axe.
        """
        for componentname in self.__dict__.keys():
            compo = self.get_comp(componentname)
            if isinstance(compo, (ScalarField, VectorField)):
                compo.set_axes(axe_x, axe_y)

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
            self.V.set_origin(x, None)
        if y is not None:
            if not isinstance(y, NUMBERTYPES):
                raise TypeError("'y' must be a number")
            self.V.set_origin(None, y)

    def get_dim(self):
        """
        Return the velocity field dimension.

        Returns
        -------
        shape : tuple
            Tuple of the dimensions (along X and Y) of the scalar field.
        """
        return self.V.get_dim()

    def get_min(self, componentname='V', unit=False):
        """
        Return the minima of the field component.

        Parameters
        ----------
        componentname : string
            Wanted component
        unit : boolean, optional
            If True, a unit object is returned,
            else (default), a float is returned.

        Returns
        -------
        min : float
            Minima on the component of the field
        """
        comp = self.get_comp(componentname)
        if isinstance(comp, (ScalarField, VectorField)):
            return comp.get_min(unit)
        else:
            raise ValueError("I can't compute a minima on this thing")

    def get_max(self, componentname='V', unit=False):
        """
        Return the maxima of the field component.

        Parameters
        ----------
        componentname : string, optiona
            Wanted component
        unit : boolean, optional
            If True, a unit object is returned,
            else (default), a float is returned.

        Returns
        -------
        max : float
            Maxima on the component of the field
        """
        comp = self.get_comp(componentname)
        if isinstance(comp, (ScalarField, VectorField)):
            return comp.get_max(unit)
        else:
            raise ValueError("I can't compute the maxima on that sort of "
                             "thing")

    def get_axes(self):
        """
        Return the velocity field axes.

        Returns
        -------
        axe_x : array
            Axe along X.
        axe_y : array
            Axe along Y.
        """
        return self.V.get_axes()

    def get_profile(self, componentname, direction, position):
        """
        Return a profile of the velocity field component, at the given position
        (or at least at the nearest possible position).
        If 'position' is an interval, the fonction return an average profile
        in this interval.

        Function
        --------
        axe, profile, cutposition = get_profile(component, direction, position)

        Parameters
        ----------
        component : string
            component to treat.
        direction : integer
            Direction along which we choose a position (1 for x and 2 for y).
        position : float or interval of float
            Position or interval in which we want a profile.

        Returns
        -------
        profile : Profile Object
            Asked profile.
        cutposition : array or number
            Final position or interval in which the profile has been taken.
        """
        compo = self.get_comp(componentname)
        return compo.get_profile(direction, position)

    def get_area(self, intervalx, intervaly):
        """
        Return the velocity field delimited by two intervals.

        Fonction
        --------
        trimfield = get_area(intervalx, intervaly)

        Parameters
        ----------
        intervalx : array
            interval wanted along x.
        intervaly : array
            interval wanted along y.

        Returns
        -------
        trimfield : VectorField object
            The wanted trimmed field from the main vector field.
        """
        if not isinstance(intervalx, ARRAYTYPES):
            raise TypeError("'intervalx' must be an array of two numbers")
        intervalx = np.array(intervalx)
        if intervalx.shape != (2,):
            raise ValueError("'intervalx' must be an array of two numbers")
        if intervalx[0] > intervalx[1]:
            raise ValueError("'intervalx' values must be crescent")
        if not isinstance(intervaly, ARRAYTYPES):
            raise TypeError("'intervaly' must be an array of two numbers")
        intervaly = np.array(intervaly)
        if intervaly.shape != (2,):
            raise ValueError("'intervaly' must be an array of two numbers")
        if intervaly[0] > intervaly[1]:
            raise ValueError("'intervaly' values must be crescent")
        trimfield = VelocityField()
        trimfield.import_from_vectorfield(self.V.get_area(intervalx,
                                                          intervaly),
                                          self.time)
        for componentname in self.__dict__.keys():
            if not componentname == "V":
                compo = self.get_comp(componentname)
                trimfield.set_comp(componentname,
                                   compo.get_area(intervalx, intervaly))
        return trimfield

    def get_copy(self):
        """
        Return a copy of the velocityfield.
        """
        copy = VelocityField()
        copy.import_from_velocityfield(self)
        return copy

    def calc_magnitude(self):
        """
        Compute and store the velocity field magnitude.
        """
        self.magnitude = self.V.get_magnitude()

    def calc_theta(self, *args):
        """
        Compute and store velocity field vector angles.
        """
        self.theta = self.V.get_theta(*args)

    def calc_vorticity(self):
        """
        Compute and store the velocity field vorticity.
        """
        self.vorticity = self.V.get_vorticity()

    def calc_sigma(self, radius=None, mask=None):
        """
        Compute and store the sigma criterion for vortex analysis
        """
        from IMTreatment.vortex_detection.vortex_detection import get_sigma
        self.sigma = get_sigma(self.V, radius)

    def calc_gamma1(self, radius=None, ind=False, mask=None):
        """
        Compute and store the gamma1 criterion for vortex analysis
        """
        from IMTreatment.vortex_detection.vortex_detection import get_gamma
        self.gamma1 = get_gamma(self.V, radius=radius, ind=ind,
                                kind='gamma1', mask=mask)

    def calc_gamma2(self, radius=None, ind=False, mask=None):
        """
        Compute and store the gamma2 criterion for vortex analysis
        """
        from IMTreatment.vortex_detection.vortex_detection import get_gamma
        self.gamma2 = get_gamma(self.V, radius=radius, ind=ind,
                                kind='gamma2', mask=mask)

    def calc_kappa1(self, radius=None, ind=False, mask=None):
        """
        Compute and store the kappa1 criterion for vortex analysis
        """
        from IMTreatment.vortex_detection.vortex_detection import get_kappa
        self.kappa1 = get_kappa(self.V, radius=radius, ind=ind,
                                kind='kappa1', mask=mask)

    def calc_kappa2(self, radius=None, ind=False, mask=None):
        """
        Compute and store the kappa2 criterion for vortex analysis
        """
        from IMTreatment.vortex_detection.vortex_detection import get_kappa
        self.kappa2 = get_kappa(self.V, radius=radius, ind=ind,
                                kind='kappa2', mask=mask)

    def calc_iota(self, mask=None, sigmafilter=False):
        """
        Compute and store the kappa2 criterion for vortex analysis
        """
        from IMTreatment.vortex_detection.vortex_detection import get_iota
        self.iota = get_iota(self.V, mask)

    def calc_q_criterion(self, mask=None):
        """
        Compute and store the Q criterion for vortex analysis
        """
        from IMTreatment.vortex_detection.vortex_detection\
            import get_q_criterion
        self.qcrit = get_q_criterion(self.V, mask)

    def calc_swirling_strength(self):
        """
        Compute and store the swirling strength for vortex analysis.
        """
        self.swirling_strength = self.V.get_swirling_strength()

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
        if intervalx is not None:
            if not isinstance(intervalx, ARRAYTYPES):
                raise TypeError("'intervalx' must be an array of two numbers")
            intervalx = np.array(intervalx)
            if intervalx.ndim != 1:
                raise ValueError("'intervalx' must be an array of two numbers")
            if intervalx.shape != (2,):
                raise ValueError("'intervalx' must be an array of two numbers")
            if intervalx[0] > intervalx[1]:
                raise ValueError("'intervalx' values must be crescent")
        if intervaly is not None:
            if not isinstance(intervaly, ARRAYTYPES):
                raise TypeError("'intervaly' must be an array of two numbers")
            intervaly = np.array(intervaly)
            if intervaly.ndim != 1:
                raise ValueError("'intervaly' must be an array of two numbers")
            if intervaly.shape != (2,):
                raise ValueError("'intervaly' must be an array of two numbers")
            if intervaly[0] > intervaly[1]:
                raise ValueError("'intervaly' values must be crescent")
        tmp_vf = VelocityField()
        V = self.V.trim_area(intervalx, intervaly)
        tmp_vf.import_from_vectorfield(V, self.time, self.unit_time)
        return tmp_vf

    def _display(self, componentname="V", **plotargs):
        if not isinstance(componentname, str):
            raise TypeError("'componentname' must be a string")
        compo = self.get_comp(componentname)
        if isinstance(compo, ScalarField):
            fig = compo._display(**plotargs)
        if isinstance(compo, VectorField):
            fig = compo._display(**plotargs)
        return fig

    def crop_masked_border(self):
        """
        Crop the masked border of the vector field in place.
        """
        self.V.crop_masked_border()
        self._clear_derived()

    def fill(self, tof='interplin', value=[0., 0.], crop_border=True):
        """
        Fill the masked part of the velocity field in place.

        Parameters
        ----------
        tof : string, optional
            Type of algorithm used to fill.
            'value' : fill with a given value
            'interplin' : fill using linear interpolation
            'interpcub' : fill using cubic interpolation
        value : 2x1 array
            Value for filling '[Vx, Vy]' (only usefull with tof='value')
        crop_border : boolean
            If 'True' (default), masked borders of the field are cropped
            before filling. Else, values on border are extrapolated (poorly).
        """
        self.V.fill(tof=tof, value=value, crop_border=crop_border)
        self._clear_derived()

    def display(self, componentname="V", **plotargs):
        """
        Display something from the velocity field.
        If component is not given, a quiver is displayed.
        If component is a component name, the coresponding component of the
        field is displayed.

        Parameters
        ----------
        component : string, optional
            Component to display ('V', 'Vx', Vy' or 'magnitude').
        plotargs : dict
            Arguments passed to the function used to display the vector field.
        """

        if not isinstance(componentname, str):
            raise TypeError("'componentname' must be a string")
        compo = self.get_comp(componentname)
        if isinstance(compo, ScalarField):
            fig = compo.display(**plotargs)
            plt.title(componentname + " " + compo.unit_values.strUnit()
                      + ", at t=" + str(self.time*self.unit_time))
        elif isinstance(compo, VectorField):
            fig = compo.display(**plotargs)
            plt.title(componentname + " " + compo.comp_x.unit_values.strUnit()
                      + ", at t=" + str(self.time*self.unit_time))
        else:
            raise StandardError("I don't know how to plot a {}"
                                .format(type(compo)))
        return fig

    def __display_profile__(self, componentname, direction, position,
                            **plotargs):
        if not isinstance(componentname, str):
            raise TypeError("'componentname' must be a string")
        compo = self.get_comp(componentname)
        if not isinstance(compo, ScalarField):
            raise TypeError("'componentname' must be a refenrence to a"
                            " scalarfield object")
        fig, cutposition = compo.__display_profile__(direction, position,
                                                     **plotargs)
        axelabel = componentname + " " + compo.unit_values.strUnit()
        if direction == 1:
            plt.xlabel(axelabel)
        else:
            plt.ylabel(axelabel)
        return fig, cutposition

    def display_profile(self, componentname, direction, position, **plotargs):
        """
        Display the profile of the given component at a fixed position on the
        given direction.

        Parameters
        ----------
        componentname : string
            Component wanted for the profile.
        direction : integer
            Direction along which we choose a position (1 for x and 2 for y).
        position : float or interval of float
            Position or interval in which we want a profile.
        **plotargs : dict, optional
            Supplementary arguments for the plot() function.
        """
        fig, cutposition = self.__display_profile__(componentname, direction,
                                                    position, **plotargs)
        compo = self.get_comp(componentname)
        if direction == 1:
            plt.title = "{0} {1}, at {2}" \
                        .format(componentname,
                                compo.unit_values.strUnit(),
                                cutposition*self.V.CompX.unit_x)
        else:
            plt.title = "{0} {1}, at {2}" \
                        .format(componentname,
                                compo.unit_values.strUnit(),
                                cutposition*self.V.CompX.unit_y)
        return fig

    def display_multiple_profiles(self, componentname, direction, positions,
                                  meandist=0, **plotargs):
        """
        Display profiles of a velocity field component, at given positions
        (or at least at the nearest possible positions).
        If 'meandist' is non-zero, profiles will be averaged on the interval
        [position - meandist, position + meandist].

        Parameters
        ----------
        componentname : string
            Component wanted for the profile.
        direction : integer
            Direction along which we choose a position (1 for x and 2 for y).
        positions : tuple of numbers
            Positions in which we want a profile.
        meandist : number
            Distance for the profil average.
        **plotargs :
            Supplementary arguments for the plot() function.
        """
        if not isinstance(componentname, str):
            raise TypeError("'component' must be a string")
        compo = self.get_comp(componentname)
        if not isinstance(compo, ScalarField):
            raise TypeError("'componentname' must be a refenrence to a"
                            " scalarfield object")
        compo.display_multiple_profiles(direction, positions, meandist,
                                        **plotargs)
        if direction == 1:
            plt.title(componentname + " " + compo.unit_values.strUnit() + ", "
                      + "for given values of X")
            plt.xlabel(componentname + " " + compo.unit_values.strUnit())
        else:
            plt.title(componentname + " " + compo.unit_values.strUnit() + ", "
                      + "for given values of Y")
            plt.ylabel(componentname + " " + compo.unit_values.strUnit())


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

    def import_from_davis(self, fieldspath, dt=1, fieldnumbers=None, incr=1):
        """
        Import velocity fields from  .VC7 files.
        'fieldspath' can be a tuple of path to vc7 files or a path to a
        folder. In this last case, all vc7 file present in the folder are
        imported.

        Parameters
        ----------
        fieldspath : string or tuple of string
        fieldnumbers : 2x1 tuple of int
            Interval of fields to import, default is all.
        incr : integer
            Incrementation between fields to take. Default is 1, meaning all
            fields are taken.
        dt : number
            interval of time between fields.
        """
        self.fields = []
        if fieldnumbers is not None:
            if not isinstance(fieldnumbers, ARRAYTYPES):
                raise TypeError("'fieldnumbers' must be a 2x1 array")
            if not len(fieldnumbers) == 2:
                raise TypeError("'fieldnumbers' must be a 2x1 array")
            if not isinstance(fieldnumbers[0], int) \
                    or not isinstance(fieldnumbers[1], int):
                raise TypeError("'fieldnumbers' must be an array of integers")
        if not isinstance(incr, int):
            raise TypeError("'incr' must be an integer")
        if incr <= 0:
            raise ValueError("'incr' must be positive")
        if not isinstance(dt, NUMBERTYPES):
            raise TypeError("'dt' must be a number")
        # fields are defining by a single path
        if isinstance(fieldspath, STRINGTYPES):
            if not os.path.exists(fieldspath):
                raise ValueError("'fieldspath' must be a valid path")
            # fields are defining by a directory
            if os.path.isdir(fieldspath):
                import glob
                fieldspath = glob.glob(os.path.join(fieldspath, '*.vc7'))
                if fieldnumbers is None:
                    fieldnumbers = [0, len(fieldspath) - 1]
                i = -1
                # importing all the files in 'fieldnumbers', by 'incr' steps
                for path in fieldspath[fieldnumbers[0]:fieldnumbers[1] + 1]:
                    i += 1
                    if i % incr != 0:
                        continue
                    tmp_vf = VelocityField()
                    tmp_vf.import_from_davis(path)
                    self.add_field(tmp_vf)
            # fields are defining by a single file
            else:
                tmp_vf = VelocityField()
                tmp_vf.import_from_davis(fieldspath)
                self.add_field(tmp_vf)
        # fields are defining by a set of file path
        elif isinstance(fieldspath, ARRAYTYPES):
            if not isinstance(fieldspath[0], STRINGTYPES):
                raise TypeError("'fieldspath' must be a string or a tuple of"
                                " string")
            for path in fieldspath:
                tmp_vf = VelocityField()
                tmp_vf.import_from_davis(path)
                self.add_field(tmp_vf)
        else:
            raise TypeError("'fieldspath' must be a string or a tuple of"
                            " string")
        # time implementation
        t = 0
        for field in self.fields:
            field.time = t
            t += dt*incr

    def import_from_vf(self, velocityfields):
        """
        Import velocity fields from another velocityfields.

        Parameters
        ----------
        velocityfields : velocityFields object
            Velocity fields to copy.
        """
        for componentkey in velocityfields.__dict__.keys():
            component = velocityfields.__dict__[componentkey]
            if isinstance(component, (VelocityField, VectorField,
                                      ScalarField)):
                self.__dict__[componentkey] \
                    = component.get_copy()
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

    def import_from_ascii(self, filepath, incr=1, interval=None,
                          x_col=1, y_col=2, vx_col=3,
                          vy_col=4, unit_x=make_unit(""), unit_y=make_unit(""),
                          unit_values=make_unit(""), times=[],
                          unit_time=make_unit(''), **kwargs):
        """
        Import velocityfields from an ascii files.
        Warning : txt files are taken in alpha-numerical order
        ('file2.txt' is taken before 'file20.txt').
        So you should name your files properly.

        Parameters
        ----------
        filepath : string
            Pathname pattern to the ascii files.
            (Example: >>> r"F:\datas\velocities_*.txt")
        incr : integer, optional
            Increment value between two fields taken.
        interval : 2x1 array, optional
            Interval in which take fields.
        x_col, y_col, vx_col, vy_col : integer, optional
            Colonne numbers for the given variables
            (begining at 1).
        unit_x, unit_y, unit_v : Unit objects, optional
            Unities for the given variables.
        times : array of number, optional
            Times of the instantaneous fields.
        unit_time : Unit object, optional
            Time unit, 'second' by default.
        **kwargs :
            Possibles additional parameters are the same as those used in the
            numpy function 'genfromtext()' :
            'delimiter' to specify the delimiter between colonnes.
            'skip_header' to specify the number of colonne to skip at file
                begining
            ...
        """
        if not isinstance(incr, int):
            raise TypeError("'incr' must be an integer")
        if incr < 1:
            raise ValueError("'incr' must be superior to 1")
        if interval is not None:
            if not isinstance(interval, ARRAYTYPES):
                raise TypeError("'interval' must be an array")
            if not len(interval) == 2:
                raise ValueError("'interval' must be a 2x1 array")
            if interval[0] > interval[1]:
                interval = [interval[1], interval[0]]
        paths = glob.glob(filepath)
        if interval is None:
            interval = [0, len(paths)-1]
        if interval[0] < 0 or interval[1] > len(paths):
            raise ValueError("'interval' is out of bounds")
        if times == []:
            times = np.arange(len(paths))
        if len(paths) != len(times):
            raise ValueError("Not enough values in 'times'")
        ref_path_len = len(paths[0])
        for i in np.arange(interval[0], interval[1] + 1, incr):
            path = paths[i]
            if len(path) != ref_path_len:
                raise Warning("You should check your files names,"
                              "i may have taken them in the wrong order.")
            tmp_vf = VelocityField()
            tmp_vf.import_from_ascii(path, x_col, y_col, vx_col, vy_col,
                                     unit_x, unit_y, unit_values, times[i],
                                     unit_time, **kwargs)
            self.add_field(tmp_vf)

    def import_from_file(self, filepath, **kw):
        """
        Load a VelocityFields object from the specified file using the JSON
        format.
        Additionnals arguments for the JSON decoder may be set with the **kw
        argument. Such as'encoding' (to change the file
        encoding, default='utf-8').

        Parameters
        ----------
        filepath : string
            Path specifiing the VelocityFields to load.
        """
        import IMTreatment.io.io as imtio
        tmp_vf = imtio.import_from_file(filepath, **kw)
        if tmp_vf.__classname__ != self.__classname__:
            raise IOError("This file do not contain a {}"
                          ", cabron.".format(self.__classname__))
        self.import_from_vf(tmp_vf)

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
        self.fields.append(velocityfield.get_copy())

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

    def get_copy(self):
        """
        Return a copy of the velocityfields
        """
        tmp_vfs = VelocityFields()
        tmp_vfs.import_from_vfs(self)
        return tmp_vfs

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


class TemporalVelocityFields(VelocityFields):
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
            tmp_tvfs = self.get_copy()
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

    def __sqrt__(self):
        final_vfs = TemporalVelocityFields()
        for field in self.fields:
            final_vfs.add_field(np.sqrt(field))
        return final_vfs

    def __pow__(self, number):
        if not isinstance(number, NUMBERTYPES):
            raise TypeError("You only can use a number for the power "
                            "on a Vectorfield")
        final_vfs = TemporalVelocityFields()
        for field in self.fields:
            final_vfs.add_field(np.power(field, number))
        return final_vfs

    def import_from_tvfs(self, tvfs):
        """
        Import velocity fields from another temporalvelocityfields.

        Parameters
        ----------
        tvfs : TemporalVelocityFields object
            Velocity fields to copy.
        """
        # delete derived fields because base fields are modified
        self._clear_derived()
        for componentkey in tvfs.__dict__.keys():
            component = tvfs.__dict__[componentkey]
            if isinstance(component, (VelocityField, VectorField,
                                      ScalarField)):
                self.__dict__[componentkey] \
                    = component.get_copy()
            elif isinstance(component, NUMBERTYPES):
                self.__dict__[componentkey] \
                    = copy.deepcopy(component)
            elif isinstance(component, STRINGTYPES):
                self.__dict__[componentkey] \
                    = copy.deepcopy(component)
            elif isinstance(component, unum.Unum):
                self.__dict__[componentkey] \
                    = component.copy()
            elif isinstance(component, ARRAYTYPES):
                self.__dict__[componentkey] \
                    = copy.deepcopy(component)
            else:
                raise TypeError("Unknown attribute type in VelocityFields "
                                "object (" + str(componentkey) + " : "
                                + str(type(component)) + ")."
                                " You must implemente it.")

    def _clear_derived(self):
        """
        Delete all the derived fields, in case of changement in the base
        fields.
        """
        attributes = dir(self)
        for attr in attributes:
            # check if the attribute is a component
            try:
                self.get_comp(attr)
            except ValueError:
                pass
            else:
                # check if the attribute is the base field
                if attr not in ['fields']:
                    del self.__dict__[attr]

    def add_field(self, velocityfield):
        """
        Add a field to the existing fields.

        Parameters
        ----------
        velocityfield : VelocityField object
            The velocity field to add.
        """
        # delete derived fields because base fields are modified
        self._clear_derived()
        if not len(self.fields) == 0:
            axes = self.fields[0].get_axes()
            vaxes = velocityfield.get_axes()
            if not all(axes[0] == vaxes[0]) and all(axes[1] == vaxes[1]):
                raise ValueError("Axes of the new field must be consistent "
                                 "with current axes")
        VelocityFields.add_field(self, velocityfield)

    def get_comp(self, componentname):
        """
        Return a reference to the field designed by 'fieldname'.

        Parameters
        ----------
        componentname : string
            Name of the component.

        Returns
        -------
        component : VelocityField, VectorField or ScalarField object or array
        of these objects
            Reference to the field.
        """
        if not isinstance(componentname, str):
            raise TypeError("'componentname' must be a string")
        # Temporal Velocity Field attributes
        elif componentname == "fields":
            return self.fields
        elif componentname == "mean_vf":
            try:
                return self.mean_vf
            except AttributeError:
                self.calc_mean_vf()
                return self.mean_vf
        elif componentname == "turbulent_vf":
            try:
                return self.turbulent_vf
            except AttributeError:
                self.calc_turbulent_vf()
                return self.turbulent_vf
        elif componentname == "mean_kinetic_energy":
            try:
                return self.mean_kinetic_energy
            except AttributeError:
                self.calc_mean_kinetic_energy()
                return self.mean_kinetic_energy
        elif componentname == "turbulent_kinetic_energy":
            try:
                return self.turbulent_kinetic_energy
            except AttributeError:
                self.calc_turbulent_kinetic_energy()
                return self.turbulent_kinetic_energy
        elif componentname == "rs_xx":
            try:
                return self.rs_xx
            except AttributeError:
                self.calc_reynolds_stress()
                return self.rs_xx
        elif componentname == "rs_yy":
            try:
                return self.rs_yy
            except AttributeError:
                self.calc_reynolds_stress()
                return self.rs_yy
        elif componentname == "rs_xy":
            try:
                return self.rs_xy
            except AttributeError:
                self.calc_reynolds_stress()
                return self.rs_xy
        elif componentname == "tke":
            try:
                return self.tke
            except AttributeError:
                self.calc_tke()
                return self.tke
        elif componentname == "mean_tke":
            try:
                return self.mean_tke
            except AttributeError:
                self.calc_mean_tke()
                return self.mean_tke
        # Velocity Field attributes
        elif len(self.fields) != 0:
            try:
                self.fields[0].get_comp(componentname)
            except ValueError:
                pass
            else:
                tmp_fields = []
                for field in self.fields:
                    tmp_fields.append(field.get_comp(componentname))
                return tmp_fields
        raise ValueError("Unknown component : {}".format(componentname))

    def get_axes(self):
        """
        Return fields axis
        """
        return self[0].V.get_axes()

    def get_dim(self):
        """
        Return the fields dimension.
        """
        if len(self.fields) == 0:
            return (0,)
        return self.fields[0].get_dim()

    def get_copy(self):
        """
        Return a copy of the velocityfields
        """
        tmp_tvfs = TemporalVelocityFields()
        tmp_tvfs.import_from_tvfs(self)
        return tmp_tvfs

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
        axe_x, axe_y = self[0].get_axes()
        if x < np.min(axe_x) or x > np.max(axe_x)\
                or y < np.min(axe_y) or y > np.max(axe_y):
            raise ValueError("'x' ans 'y' values out of bounds")
        compo = self.get_comp(component)
        if not isinstance(compo, ARRAYTYPES):
            raise ValueError("Unvalid component for a time profile")
        # if the given object is a ScalarField
        if isinstance(compo[0], ScalarField):
            time = []
            unit_time = self[0].unit_time
            values = np.zeros(len(compo))
            mask = np.zeros(len(compo))
            unit_values = compo[0].unit_values
            # getting position indices
            ind_x = compo[0].get_indice_on_axe(1, x, nearest=True)
            ind_y = compo[0].get_indice_on_axe(2, y, nearest=True)
            for i in np.arange(len(compo)):
                time.append(self[i].time)
                values[i] = compo[i].values.data[ind_y, ind_x]
                mask[i] = compo[i].values.mask[ind_y, ind_x]
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
        pt = np.array(pt)
        if not pt.shape == (2,):
            raise ValueError("'pt' must be a 2x1 array")
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
                ind_x = x
                ind_y = y
            elif not interp:
                inds_x = self.fields[0].V.comp_x.get_indice_on_axe(1, pt[0])
                if len(inds_x) == 1:
                    ind_x = inds_x[0]
                else:
                    vals = [axe_x[inds_x[0]], axe_x[inds_x[1]]] - x
                    ind_x = inds_x[np.argmin(vals)]
                inds_y = self.fields[0].V.comp_x.get_indice_on_axe(2, pt[1])
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
                    values.append(sf.values[ind_y, ind_x])
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
                            unit_y=comp[0].unit_values)
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
        axe_x, axe_y = self[0].get_axes()
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
            tmp_sf = self.fields[0].V.comp_x
            ind_x_min = tmp_sf.get_indice_on_axe(1, intervalx[0])[0]
            ind_x_max = tmp_sf.get_indice_on_axe(1, intervalx[1])[-1]
            ind_y_min = tmp_sf.get_indice_on_axe(2, intervaly[0])[0]
            ind_y_max = tmp_sf.get_indice_on_axe(2, intervaly[1])[-1]
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
            maskx = np.ma.getmaskarray(field.V.comp_x.values)
            masky = np.ma.getmaskarray(field.V.comp_y.values)
            mask = np.logical_or(maskx,
                                 masky)
            values_x = field.V.comp_x.values.data
            values_y = field.V.comp_y.values.data
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
        comp_x = self.fields[0].V.comp_x.get_copy()
        comp_x.values = np.ma.masked_array(mean_vx, mask_tot)
        comp_y = self.fields[0].V.comp_y.get_copy()
        comp_y.values = np.ma.masked_array(mean_vy, mask_tot)
        mean_vf = VelocityField()
        mean_vf.import_from_scalarfields(comp_x, comp_y, time=time,
                                         unit_time=self.fields[0].unit_time)
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
        final_sf = 1./2*(mean_vf.V.comp_x**2
                         + mean_vf.V.comp_y**2)
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
        rs_xx = np.zeros(u_p[0].values.shape)
        mask_rs_xx = np.zeros(u_p[0].values.shape)
        # boucle sur les points du champ
        for i in np.arange(rs_xx.shape[0]):
            for j in np.arange(rs_xx.shape[1]):
                # boucle sur le nombre de champs
                nmb_val = 0
                for n in np.arange(len(turb_vf.fields)):
                    # check if masked
                    if not u_p[n].values.mask[i, j]:
                        rs_xx[i, j] += u_p[n].values[i, j]**2
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
                        rs_yy[i, j] += v_p[n].values[i, j]**2
                        nmb_val += 1
                if nmb_val > nmb_val_min:
                    rs_yy[i, j] /= nmb_val
                else:
                    rs_yy[i, j] = 0
                    mask_rs_yy[i, j] = True
        # rs_xy
        rs_xy = np.zeros(u_p[0].values.shape)
        mask_rs_xy = np.zeros(u_p[0].values.shape)
        # boucle sur les points du champ
        for i in np.arange(rs_xy.shape[0]):
            for j in np.arange(rs_xy.shape[1]):
                # boucle sur le nombre de champs
                nmb_val = 0
                for n in np.arange(len(turb_vf.fields)):
                    # check if masked
                    if not (u_p[n].values.mask[i, j]
                            or v_p[n].values.mask[i, j]):
                        rs_xy[i, j] += (u_p[n].values[i, j]
                                        * v_p[n].values[i, j])
                        nmb_val += 1
                if nmb_val > nmb_val_min:
                    rs_xy[i, j] /= nmb_val
                else:
                    rs_xy[i, j] = 0
                    mask_rs_xy[i, j] = True
        # masking and storing
        axe_x = self.fields[0].V.comp_x.axe_x
        axe_y = self.fields[0].V.comp_x.axe_y
        unit_x = self.fields[0].V.comp_x.unit_x
        unit_y = self.fields[0].V.comp_x.unit_y
        unit_values = self.fields[0].V.comp_x.unit_values
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
            return self.get_copy()
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
            super_mask = np.sum(super_mask).values.data
            super_mask[super_mask == len(self.fields)] = 0
            # loop on each field position
            for j, i in np.argwhere(super_mask):
                print(i, j)
                prof_x = self.get_time_profile('Vx', axe_x[i], axe_y[j])
                prof_y = self.get_time_profile('Vy', axe_x[i], axe_y[j])
                # checking if all time profile value are masked
                if np.all(np.logical_or(prof_x.y.mask, prof_x.y.mask)):
                    continue
                # getting masked position on profile
                inds_masked_x = np.where(prof_x.y.mask)
                inds_masked_y = np.where(prof_y.y.mask)
                # creating interpolation function
                if tof == 'interplin':
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
                    if tof == 'value':
                        interp_val = value[0]
                    elif tof in ['interplin', 'interpcubic']:
                        try:
                            interp_val = interp_x(prof_x.x[ind_masked])
                        except ValueError:
                            continue
                    # putting interpolated value in the field
                    self[ind_masked].V.comp_x.values[j, i] = interp_val
                # loop on all y profile masked points
                for ind_masked in inds_masked_y[0]:
                    if tof == 'value':
                        interp_val = value[1]
                    elif tof in ['interplin', 'interpcubic']:
                        try:
                            interp_val = interp_y(prof_y.x[ind_masked])
                        except ValueError:
                            continue
                    # putting interpolated value in the field
                    self[ind_masked].V.comp_y.values[j, i] = interp_val
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
        mask_temp = np.sum(masks_temp).values.data
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
            values_x = self.fields[i].V.comp_x.values.data
            values_y = self.fields[i].V.comp_y.values.data
            mask = self.fields[i].V.comp_x.values.mask
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
            self.fields[i].V.comp_x.values = np.ma.masked_array(values_x, mask)
            self.fields[i].V.comp_y.values = np.ma.masked_array(values_y, mask)
            self.fields[i]._clear_derived()
            #crop axis
            axe_x, axe_y = self.fields[i].get_axes()
            new_axe_x = axe_x[axe_x_m]
            new_axe_y = axe_y[axe_y_m]
            self.fields[i].V.comp_x.axe_x = new_axe_x
            self.fields[i].V.comp_y.axe_x = new_axe_x
            self.fields[i].V.comp_x.axe_y = new_axe_y
            self.fields[i].V.comp_y.axe_y = new_axe_y

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
                plt.subplot(linenmb, colnmb, i)
                field.display(**plotargs)
                plt.title(fieldname + " (field number " + str(i-1) +
                          "), at t=" + str(self[i-1].time*self[i-1].unit_time))
                i += 1
            plt.suptitle(fieldname, fontsize=18)
        elif isinstance(fields, (VelocityField, VectorField)):
            fields.display(**plotargs)
            plt.title(fieldname)
        else:
            fields.display(**plotargs)
            plt.title(fieldname)

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
        # display a vector field (quiver)
        if isinstance(comp[0], VectorField):
            fig = plt.figure()
            ax = comp[0].display(**plotargs)
            ttl = plt.title('')

            def update(num):
                vx = comp[num].comp_x.values
                vy = comp[num].comp_y.values
                magn = comp[num].get_magnitude().values
                title = "{}, at t={:.3} {}"\
                    .format(compo, float(self[num].time),
                            self[num].unit_time.strUnit())
                ttl.set_text(title)
                ax.set_UVC(vx, vy, magn)
                return ax
            anim = animation.FuncAnimation(fig, update,
                                           frames=len(comp),
                                           interval=interval, blit=False,
                                           repeat=repeat)
            return anim
            plt.show()
        # display a scalar field (contour, contourf or imshow)
        elif isinstance(comp[0], ScalarField):
            if not 'kind' in plotargs:
                plotargs['kind'] = None
            kind = plotargs['kind']
            fig = plt.figure()
            ax = comp[0].display(**plotargs)
            ttl = plt.title('')

            def update(num, ax, ttl):
                if kind is None:
                    val = comp[num].values
                    ax.set_data(val)
                else:
                    ### TODO: suffit pas !
                    ax.ax.cla()
                    ax = comp[num]._display(**plotargs)
                    ttl = plt.title('')
                title = "{}, at t={:.3} {}"\
                    .format(compo, float(self[num].time),
                            self[num].unit_time.strUnit())
                ttl.set_text(title)
                return ax
            anim = animation.FuncAnimation(fig, update,
                                           frames=len(comp),
                                           interval=interval, blit=False,
                                           repeat=repeat,
                                           fargs=(ax, ttl))
            return anim
            plt.show()

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
                profile._display(label="t={0}".format(field.time *
                                                      field.unit_time),
                                 **plotargs)
                plt.xlabel("{0} {1}".format(componentname, profile.unit_x))
                plt.ylabel("Y {0}".format(profile.unit_y))
            else:
                profile.display(label="t={0}".format(field.time *
                                                     field.unit_time),
                                **plotargs)
                plt.ylabel("{0} {1}".format(componentname, profile.unit_y))
                plt.xlabel("X {0}".format(profile.unit_x))
        plt.title(componentname)
        plt.legend()


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
                    = component.get_copy()
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

    def get_copy(self):
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
        axe = np.array([])
        bl = np.array([])
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
            unit_x = self.fields[0].V.comp_x.unit_x
            unit_y = self.fields[0].V.comp_x.unit_y
        else:
            unit_x = self.fields[0].V.comp_x.unit_y
            unit_y = self.fields[0].V.comp_x.unit_x
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
            unit = compo.comp_x.unit_value.Getunit()
        else:
            unit = compo.V.comp_x.unit_values.Getunit()
        cbar.set_label("{0} {1}".format(componentname, unit))

    def display_profile(self, componentname, direction, position, **plotargs):
        """
        Display the profile of the given component at a fixed position on the
        given direction.

        Parameters
        ----------
        componentname : string
            Component wanted for the profile.
        direction : integer
            Direction along which we choose a position (1 for x and 2 for y).
        position : float or interval of float
            Position or interval in which we want a profile.
        **plotargs : dict, optional
            Supplementary arguments for the plot() function.
        """
        display = False
        for field in self.fields:
            try:
                field.display_profile(componentname, direction, position,
                                      **plotargs)
            except ValueError:
                pass
            else:
                display = True
        if not display:
            raise ValueError("'position' must be included in"
                             " the choosen axis values")

    def display_multiple_profiles(self, component, direction, positions,
                                  meandist=0):
        """
        Display some profiles on the velocity field.
        """
        pass


###############################################
### EXAMPLES TEST      pygraphtest          ###
###############################################
#
#def pygraphtest():
#    from pycallgraph import PyCallGraph
#    from pycallgraph import Config
#    from pycallgraph.output import GraphvizOutput
#    from pycallgraph import GlobbingFilter
#    os.environ['PATH'] += ';E:\\Prog\\Portable GraphViz\\App\\bin\\'
#    config = Config()#max_depth=2)
#    config.trace_filter = GlobbingFilter(exclude=['Unit.*'])
#    graphviz = GraphvizOutput(output_file='PyCallGraph.png')
#    with PyCallGraph(output=graphviz, config=config):
#        main()
#
#if __name__ == "__main__":
#    #pygraphtest()
#    main()
