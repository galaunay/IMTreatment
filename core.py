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


class PTest(object):
    """
    Decorator used to test input parameters types.
    """
    #FIXME:  +++ need to use OrderedDict instead of classical dicts +++

    def __init__(self, *types, **kwtypes):
        print("Making a test !")
        self.types = list(types)
        self.ktypes = list([None]*len(types))
        self.ktypes += kwtypes.keys()
        self.types += kwtypes.values()

    def __call__(self, funct):
        return self.decorator(funct)

    def decorator(self, funct):
        def new_funct(*args, **kwargs):
            len_args = len(args)
            types = self.types[0:len_args]
            kwtypes = {}
            # defining parameters name for error message
            order_str = ['First', 'Second', 'Third', 'Fourth', 'Fifth',
                         'Sixth', 'Seventh', 'Eighth', 'Ninth', 'Tenth']
            # test if there is not too much arguments
            if len_args + len(kwargs) > len(self.types):
                return funct(*args, **kwargs)
            # storing given keywords parameters
            for i, key in enumerate(kwargs.keys()):
                kwtypes.update({self.ktypes[i + len_args]:
                                self.types[len_args + i]})
            # treat non-keyword parameters
            for i, arg in enumerate(args):
                # raise error if argument type is not adequate
                if not isinstance(args[i], types[i]):
                    actual_types = str(types[i]).replace('type ', '')\
                                                .replace('>', '')\
                                                .replace('<', '')
                    wanted_types = str(type(args[i])).replace('type ', '')\
                                                     .replace('>', '')\
                                                     .replace('<', '')
                    raise TypeError("{} parameter should be {}, not {}."
                                    .format(order_str[i], actual_types,
                                            wanted_types))
            # treat keyword parameters
            for j, key in enumerate(kwargs.keys()):
                # test if keyword param exist
                if not key in kwtypes.keys():
                    return funct(*args, **kwargs)
                # return error if keyword argument type is not adequat
                if not isinstance(kwargs[key], kwtypes[key]):
                    actual_types = str(kwtypes[key]).replace('<type ', '')\
                                                    .replace('>', '')
                    wanted_types = str(type(kwargs[key])).replace('<type ', '')\
                                                         .replace('>', '')
                    raise TypeError("'{}' should be {}, not {}."
                                    .format(key, actual_types, wanted_types))
            return funct(*args, **kwargs)
        new_funct.__doc__ = funct.__doc__
        new_funct.__name__ = funct.__name__
        return new_funct


#@PTest(STRINGTYPES)
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
    #@PTest(object, xy=ARRAYTYPES, v=(None, ARRAYTYPES), unit_x=unum.Unum,
    #       unit_y=unum.Unum, unit_v=unum.Unum, name=(None, STRINGTYPES))
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

    ### Operators ###
    def __init__(self):
        self.__axe_x = np.array([], dtype=float)
        self.__axe_y = np.array([], dtype=float)
        self.__unit_x = make_unit('')
        self.__unit_y = make_unit('')

    def __iter__(self):
        for i, x in enumerate(self.axe_x):
            for j, y in enumerate(self.axe_y):
                yield [i, j], [x, y]

    ### Attributes ###
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
            self.__unit_x = make_unit(new_unit_x)
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
            self.__unit_y = make_unit(new_unit_y)
        else:
            raise TypeError()

    @unit_y.deleter
    def unit_y(self):
        raise Exception("Nope, can't do that")

    ### Properties ###
    @property
    def shape(self):
        return self.__axe_x.shape[0], self.__axe_y.shape[0]

    ### Watchers ###
    def copy(self):
        """
        Return a copy of the Field object.
        """
        return copy.deepcopy(self)

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

    ### Modifiers ###
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

    def trim_area(self, intervalx=None, intervaly=None, full_output=False,
                  ind=False, inplace=False):
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
        ind : boolean, optional
            If 'True', intervals are understood as indices along axis.
            If 'False' (default), intervals are understood in axis units.
        inplace : boolean, optional
            If 'True', the field is trimed in place.
        """
        # default values
        axe_x, axe_y = self.axe_x, self.axe_y
        if intervalx is None:
            if ind:
                intervalx = [0, len(axe_x)]
            else:
                intervalx = [axe_x[0], axe_x[-1]]
        if intervaly is None:
            if ind:
                intervaly = [0, len(axe_y)]
            else:
                intervaly = [axe_y[0], axe_y[-1]]
        # checking parameters
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
        # checking triming windows
        if ind:
            if intervalx[0] < 0 or intervalx[1] == 0 or \
                    intervaly[0] < 0 or intervaly[1] == 0:
                raise ValueError("Invalid trimming window")
        else:
            if np.all(intervalx < axe_x[0]) or np.all(intervalx > axe_x[-1])\
                    or np.all(intervaly < axe_y[0]) \
                    or np.all(intervaly > axe_y[-1]):
                raise ValueError("Invalid trimming window")
        # finding interval indices
        if ind:
            indmin_x = intervalx[0]
            indmax_x = intervalx[1]
            indmin_y = intervaly[0]
            indmax_y = intervaly[1]
        else:
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
        # trimming the field
        if inplace:
            axe_x = self.axe_x[indmin_x:indmax_x + 1]
            axe_y = self.axe_y[indmin_y:indmax_y + 1]
            self.__axe_x = axe_x
            self.__axe_y = axe_y
            if full_output:
                return indmin_x, indmax_x, indmin_y, indmax_y
        else:
            trimfield = self.copy()
            trimfield.__axe_x = self.axe_x[indmin_x:indmax_x + 1]
            trimfield.__axe_y = self.axe_y[indmin_y:indmax_y + 1]
            if full_output:
                return indmin_x, indmax_x, indmin_y, indmax_y, trimfield
            else:
                return trimfield


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

    ### Operators ###
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

    ### Attributes ###
    @property
    def values(self):
        return self.__values

    @values.setter
    def values(self, new_values):
        if not isinstance(new_values, ARRAYTYPES):
            raise TypeError()
        new_values = np.array(new_values, dtype=float)
        if self.shape == new_values.shape:
            # adapting mask to 'nan' values
            self.__mask = np.isnan(new_values)
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
            new_mask = np.array(new_mask, dtype=bool)
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

    ### Properties ###
    @property
    def min(self):
        return np.min(self.values[np.logical_not(self.mask)])

    @property
    def max(self):
        return np.max(self.values[np.logical_not(self.mask)])

    @property
    def mean(self):
        return np.mean(self.values[np.logical_not(self.mask)])

    ### Field maker ###
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

    ### Watchers ###
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
            x = self.axe_x[inds[0][0]]
            y = self.axe_y[inds[1][0]]
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
            indx = ind[0]
            indy = ind[1]
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

    def copy(self):
        """
        Return a copy of the scalarfield.
        """
        return copy.deepcopy(self)

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

    ### Modifiers ###
    def trim_area(self, intervalx=None, intervaly=None, ind=False,
                  inplace=False):
        """
        Return a trimed  area in respect with given intervals.

        Parameters
        ----------
        intervalx : array, optional
            interval wanted along x
        intervaly : array, optional
            interval wanted along y
        ind : boolean, optional
            If 'True', intervals are understood as indices along axis.
            If 'False' (default), intervals are understood in axis units.
        inplace : boolean, optional
            If 'True', the field is trimed in place.
        """
        if inplace:
            values = self.values
            mask = self.mask
            indmin_x, indmax_x, indmin_y, indmax_y = \
                Field.trim_area(self, intervalx, intervaly, full_output=True,
                                ind=ind, inplace=True)
            self.__values = values[indmin_x:indmax_x + 1,
                                   indmin_y:indmax_y + 1]
            self.__mask = mask[indmin_x:indmax_x + 1,
                               indmin_y:indmax_y + 1]
        else:
            indmin_x, indmax_x, indmin_y, indmax_y, trimfield = \
                Field.trim_area(self, intervalx, intervaly, full_output=True,
                                ind=ind)
            trimfield.__values = self.values[indmin_x:indmax_x + 1,
                                             indmin_y:indmax_y + 1]
            trimfield.__mask = self.mask[indmin_x:indmax_x + 1,
                                         indmin_y:indmax_y + 1]
            return trimfield

    def crop_masked_border(self):
        """
        Crop the masked border of the field in place.
        """
        # checking masked values presence
        mask = self.mask
        if not np.any(mask):
            return None
        # getting indices where we need to cut
        axe_x_m = np.logical_not(np.all(mask, axis=1))
        axe_y_m = np.logical_not(np.all(mask, axis=0))
        axe_x_min = np.where(axe_x_m)[0][0]
        axe_x_max = np.where(axe_x_m)[0][-1]
        axe_y_min = np.where(axe_y_m)[0][0]
        axe_y_max = np.where(axe_y_m)[0][-1]
        self.trim_area([axe_x_min, axe_x_max],
                       [axe_y_min, axe_y_max],
                       ind=True, inplace=True)

    def fill(self, tof='linear', value=0., remaining_values=np.nan,
             crop_border=True, ):
        """
        Fill the masked part of the array in place.

        Parameters
        ----------
        tof : string, optional
            Type of algorithm used to fill.
            'fill' : fill with a given value
            'nearest' : fill with nearest point value
            'linear' : fill using linear interpolation
            'cubic' : fill using cubic interpolation
        value : number
            Value used to fill (for 'tof=fill').
        remaining_values:
            Values used to fill the field, where mask remain after
            interpolation (typicaly borders or corner).
            can be 'nearest' for second pass with nearest filling.
        crop_border : boolean
            If 'True' (default), masked borders of the field are cropped
            before filling. Else, values on border are extrapolated (poorly).
        """
        # check parameters coherence
        if not isinstance(tof, STRINGTYPES):
            raise TypeError("'tof' must be a string")
        if not isinstance(value, NUMBERTYPES):
            raise TypeError("'value' must be a number")
        if not (isinstance(remaining_values, NUMBERTYPES)
                or remaining_values == 'nearest'):
            raise TypeError()
        # deleting the masked border (useless field part)
        if crop_border:
            self.crop_masked_border()
        axe_x, axe_y = self.axe_x, self.axe_y
        mask = self.mask
        not_mask = np.logical_not(mask)
        values = self.values
        values[mask] = np.nan
        # if there is nothing to do...
        if not np.any(mask):
            pass
        elif tof in ['linear', 'cubic', 'nearest']:
            X, Y = np.meshgrid(axe_x, axe_y)
            xy = np.array(zip(X.flatten('F'), Y.flatten('F')), subok=True)
            xy_nm = xy[not_mask.flatten()]
            xy_m = xy[mask.flatten()]
            values_nm = values[not_mask].flatten()
            xy_interp = spinterp.griddata(xy_nm, values_nm, xy_m, method=tof)
            values[mask] = xy_interp
            if remaining_values == 'nearest':
                self.values = values
                self.mask = np.isnan(values)
                self.fill(tof='nearest', crop_border=False)
            elif not np.isnan(remaining_values):
                values[np.isnan(values)] = remaining_values
                self.values = values
                self.mask = np.isnan(values)
            else:
                self.values = values
                self.mask = np.isnan(values)
        elif tof == 'fill':
            values[mask] = value
            self.values = values
            self.mask = np.isnan(values)
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

    ### Displayers ###
    def _display(self, component=None, kind=None, **plotargs):
        # getting datas
        axe_x, axe_y = self.axe_x, self.axe_y
        unit_x, unit_y = self.unit_x, self.unit_y
        X, Y = np.meshgrid(self.axe_y, self.axe_x)
        # getting wanted component
        if component is None or component == 'values':
            values = np.transpose(self.values)
            mask = np.transpose(self.mask)
            values[mask] = np.nan
        elif component == 'mask':
            values = np.transpose(self.mask)
        else:
            raise ValueError("unknown value of 'component' parameter : {}"
                             .format(component))
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
        # setting labels
        plt.axis('equal')
        plt.xlabel("X " + unit_x.strUnit())
        plt.ylabel("Y " + unit_y.strUnit())
        return displ

    def display(self, component=None, kind=None, **plotargs):
        """
        Display the scalar field.

        Parameters
        ----------
        component : string, optional
            Component to display, can be 'values' or 'mask'
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
        displ = self._display(component, kind,  **plotargs)
        plt.title("Scalar field Values " + self.unit_values.strUnit())
        cb = plt.colorbar(displ, shrink=1, aspect=5)
        cb.set_label(self.unit_values.strUnit())
        # search for limits in case of masked field
        if component != 'mask':
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

    ### Operators ###
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
            if not mask[i, j]:
                yield ij, xy, [datax[i, j], datay[i, j]]

    ### Attributes ###
    @property
    def comp_x(self):
        return self.__comp_x

    @comp_x.setter
    def comp_x(self, new_comp_x):
        if not isinstance(new_comp_x, ARRAYTYPES):
            raise TypeError()
        new_comp_x = np.array(new_comp_x, dtype=float)
        if not new_comp_x.shape == self.shape:
            raise ValueError("'comp_x' must be coherent with axis system")
        # adapting mask to 'nan' values
        if self.__comp_y.shape != (0,):
            self.__mask = np.logical_or(np.isnan(new_comp_x),
                                        np.isnan(self.__comp_y))
        else:
            self.__mask = np.isnan(new_comp_x)
        # storing dat
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
        new_comp_y = np.array(new_comp_y, dtype=float)
        if not new_comp_y.shape == self.shape:
            raise ValueError()
        # adapting mask to 'nan' values
        if self.__comp_x.shape != (0,):
            self.__mask = np.logical_or(np.isnan(new_comp_y),
                                        np.isnan(self.__comp_x))
        else:
            self.__mask = np.isnan(new_comp_y)
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
            if not isinstance(new_mask.flat[0], np.bool_):
                raise TypeError()
            new_mask = np.array(new_mask, dtype=bool)
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
            self.__unit_values = make_unit(new_unit_values)
        else:
            raise TypeError()

    @unit_values.deleter
    def unit_values(self):
        raise Exception("Nope, can't do that")

    ### Properties ###
    @property
    def magnitude(self):
        """
        Return a scalar field with the velocity field magnitude.
        """
        comp_x, comp_y = self.comp_x, self.comp_y
        mask = self.mask
        values = (comp_x**2 + comp_y**2)**(.5)
        values[mask] = np.nan
        return values

    @property
    def magnitude_as_sf(self):
        """
        Return a scalarfield with the velocity field magnitude.
        """
        tmp_sf = ScalarField()
        tmp_sf.import_from_arrays(self.axe_x, self.axe_y, self.magnitude,
                                  mask=self.mask, unit_x=self.unit_x,
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


    ### Field Maker ###
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

    ### Watchers ###
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

    def copy(self):
        """
        Return a copy of the vectorfield.
        """
        return copy.deepcopy(self)

    ### Modifiers ###
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

    def fill(self, tof='linear', value=0., remaining_values=np.nan,
             crop_border=True):
        """
        Fill the masked part of the field in place.

        Parameters
        ----------
        tof : string, optional
            Type of algorithm used to fill.
            'fill' : fill with a given value
            'nearest' : fill with nearest point value
            'linear' : fill using linear interpolation
            'cubic' : fill using cubic interpolation
        value : number
            Value for filling (only usefull with tof='fill')
        remaining_values: 2x1 array or string
            Values used to fill the field, where mask remain after
            interpolation (typicaly borders or corner).
            can be 'nearest' for second pass with nearest filling.
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
        axe_x, axe_y = self.axe_x, self.axe_y
        mask = self.mask
        not_mask = np.logical_not(mask)
        comp_x = self.comp_x
        comp_y = self.comp_y
        comp_x[mask] = np.nan
        comp_y[mask] = np.nan
        # if there is nothing to do...
        if not np.any(mask):
            pass
        elif tof in ['linear', 'cubic', 'nearest']:
            X, Y = np.meshgrid(axe_x, axe_y)
            xy = np.array(zip(X.flatten('F'), Y.flatten('F')), subok=True)
            xy_nm = xy[not_mask.flatten()]
            xy_m = xy[mask.flatten()]
            comp_x_nm = comp_x[not_mask].flatten()
            comp_y_nm = comp_y[not_mask].flatten()
            comp_x_interp = spinterp.griddata(xy_nm, comp_x_nm, xy_m,
                                              method=tof)
            comp_y_interp = spinterp.griddata(xy_nm, comp_y_nm, xy_m,
                                              method=tof)
            comp_x[mask] = comp_x_interp
            comp_y[mask] = comp_y_interp
            if remaining_values == 'nearest':
                self.comp_x = comp_x
                self.comp_y = comp_y
                self.mask = np.logical_or(np.isnan(comp_x),
                                          np.isnan(comp_y))
                self.fill(tof='nearest', crop_border=False)
            elif not np.isnan(remaining_values):
                comp_x[np.isnan(comp_x)] = remaining_values[0]
                comp_y[np.isnan(comp_x)] = remaining_values[1]
                self.comp_x = comp_x
                self.comp_y = comp_y
                self.mask = np.logical_or(np.isnan(comp_x),
                                          np.isnan(comp_y))
            else:
                self.comp_x = comp_x
                self.comp_y = comp_y
                self.mask = np.logical_or(np.isnan(comp_x),
                                          np.isnan(comp_y))
        elif tof == 'fill':
            Vx = self.comp_x
            Vy = self.comp_y
            Vx[mask] = value
            Vy[mask] = value
            self.comp_x = Vx
            self.comp_y = Vy
            self.mask = np.logical_or(np.isnan(comp_x), np.isnan(comp_y))
        else:
            raise ValueError("unknown 'tof' value")

    def trim_area(self, intervalx=None, intervaly=None, ind=False,
                  inplace=False):
        """
        Return a trimed  area in respect with given intervals.

        Parameters
        ----------
        intervalx : array, optional
            interval wanted along x
        intervaly : array, optional
            interval wanted along y
        ind : boolean, optional
            If 'True', intervals are understood as indices along axis.
            If 'False' (default), intervals are understood in axis units.
        inplace : boolean, optional
            If 'True', the field is trimed in place.
        """
        if inplace:
            indmin_x, indmax_x, indmin_y, indmax_y = \
                Field.trim_area(self, intervalx, intervaly, full_output=True,
                                ind=ind, inplace=True)
            self.__comp_x = self.comp_x[indmin_x:indmax_x + 1,
                                        indmin_y:indmax_y + 1]
            self.__comp_y = self.comp_y[indmin_x:indmax_x + 1,
                                        indmin_y:indmax_y + 1]
            self.__mask = self.mask[indmin_x:indmax_x + 1,
                                    indmin_y:indmax_y + 1]
        else:
            indmin_x, indmax_x, indmin_y, indmax_y, trimfield = \
                Field.trim_area(self, intervalx, intervaly, full_output=True,
                                ind=ind)
            trimfield.__comp_x = self.comp_x[indmin_x:indmax_x + 1,
                                             indmin_y:indmax_y + 1]
            trimfield.__comp_y = self.comp_y[indmin_x:indmax_x + 1,
                                             indmin_y:indmax_y + 1]
            trimfield.__mask = self.mask[indmin_x:indmax_x + 1,
                                         indmin_y:indmax_y + 1]
            return trimfield

    def crop_masked_border(self):
        """
        Crop the masked border of the field in place.
        """
        # checking masked values presence
        mask = self.mask
        if np.any(mask):
            return None
        # getting indices where we need to cut
        axe_x_m = np.logical_not(np.all(mask, axis=1))
        axe_y_m = np.logical_not(np.all(mask, axis=0))
        axe_x_min = np.where(axe_x_m)[0][0]
        axe_x_max = np.where(axe_x_m)[0][-1]
        axe_y_min = np.where(axe_y_m)[0][0]
        axe_y_max = np.where(axe_y_m)[0][-1]
        self.trim_area([axe_x_min, axe_x_max], [axe_y_min, axe_y_max],
                       ind=True, inplace=True)

    ### Displayers ###
    def _display(self, component=None, kind=None, **plotargs):
        if kind is not None:
            if not isinstance(kind, STRINGTYPES):
                raise TypeError("'kind' must be a string")
        axe_x, axe_y = self.axe_x, self.axe_y
        if component is None or component == 'V':
            Vx = np.transpose(self.comp_x)
            Vy = np.transpose(self.comp_y)
            mask = np.transpose(self.mask)
            Vx = np.ma.masked_array(Vx, mask)
            Vy = np.ma.masked_array(Vy, mask)
            magn = np.transpose(self.magnitude)
            magn = np.ma.masked_array(magn, mask)
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
            else:
                raise ValueError("Unknown value of 'kind'")
            plt.axis('equal')
            plt.xlabel("X " + unit_x.strUnit())
            plt.ylabel("Y " + unit_y.strUnit())
        elif component == "x":
            if kind == '3D':
                displ = self.comp_x_as_sf.Display3D()
            else:
                displ = self.comp_x_as_sf._display(kind=kind, **plotargs)
        elif component == "y":
            if kind == '3D':
                displ = self.comp_y_as_sf.Display3D()
            else:
                displ = self.comp_y_as_sf._display(kind=kind, **plotargs)
        elif component == "mask":
            if kind == '3D':
                displ = self.mask_as_sf.Display3D()
            else:
                displ = self.mask_as_sf._display(kind=kind, **plotargs)
        elif component == "magnitude":
            if kind == '3D':
                displ = self.magnitude_as_sf.Display3D()
            else:
                displ = self.magnitude_as_sf._display(kind=kind, **plotargs)
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
            Component to display, can be 'V', 'x', 'y', 'mask'
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
        unit_values = self.unit_values
        Vx, Vy = self.comp_x, self.comp_y
        if component is None or component == 'V':
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
            cb = plt.colorbar()
            cb.set_label("Vx " + unit_values.strUnit())
            plt.title("Vx " + unit_values.strUnit())
        elif component == 'y':
            cb = plt.colorbar()
            cb.set_label("Vy " + unit_values.strUnit())
            plt.title("Vy " + unit_values.strUnit())
        elif component == 'mask':
            cb = plt.colorbar()
            cb.set_label("Mask ")
            plt.title("Mask")
        elif component == 'magnitude':
            cb = plt.colorbar()
            cb.set_label("Magnitude")
            plt.title("Magnitude")
        else:
            raise ValueError("Unknown 'component' value")
        return displ


class Fields(object):
    """
    Class representing a set of fields. These fields can have
    differente positions along axes, or be successive view of the same area.
    It's recommended to use TemporalVelocityFields or SpatialVelocityFields
    instead of this one.
    """

    ### Operators ###
    def __init__(self):
        self.fields = np.array([], dtype=object)

    def __len__(self):
        return len(self.fields)

    def __iter__(self):
        return self.fields.__iter__()

    def __getitem__(self, fieldnumber):
        return self.fields[fieldnumber]

    ### Watchers ###
    def copy(self):
        """
        Return a copy of the velocityfields
        """
        return copy.deepcopy(self)

    ### Modifiers ###
    def add_field(self, field):
        """
        Add a field to the existing fields.

        Parameters
        ----------
        field : VectorField or ScalarField object
            The field to add.
        """
        if not isinstance(field, (VectorField, ScalarField)):
            raise TypeError("'vectorfield' must be a VelocityField object")
        self.fields = np.append(self.fields, field.copy())

    def remove_field(self, fieldnumber):
        """
        Remove a field of the existing fields.

        Parameters
        ----------
        fieldnumber : integer
            The number of the velocity field to remove.
        """
        self.fields = np.delete(self.fields, fieldnumber)

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


class TemporalFields(Fields, Field):
    """
    Class representing a set of time evolving fields.
    All fields added to this object has to have the same axis system.
    """

    ### Operators ###
    def __init__(self):
        Field.__init__(self)
        Fields.__init__(self)
        self.__times = np.array([], dtype=float)
        self.__unit_times = make_unit("")
        self.field_type = None

    def __add__(self, other):
        if isinstance(other, self.__class__):
            tmp_tfs = self.copy()
            otimes = other.times
            ounit_times = other.unit_times
            for i, ofield in enumerate(other.fields):
                tmp_tfs.add_field(ofield, otimes[i], ounit_times)
            return tmp_tfs
        else:
            raise TypeError("cannot concatenate {} with"
                            " {}.".format(self.__class__, type(other)))

    def __mul__(self, other):
        if isinstance(other, (NUMBERTYPES, unum.Unum)):
            final_vfs = self.__class__.__init__()
            for field in self.fields:
                final_vfs.add_field(field*other)
            return final_vfs
        else:
            raise TypeError("You can only multiply a temporal velocity field "
                            "by numbers")

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, (NUMBERTYPES, unum.Unum)):
            final_vfs = self.__class__.__init__()
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
        final_vfs = self.__class__.__init__()
        for field in self.fields:
            final_vfs.add_field(np.power(field, number))
        return final_vfs

    def __iter__(self):
        for i in np.arange(len(self.fields)):
            yield self.times[i], self.fields[i]

    ### Attributes ###
    @Field.axe_x.setter
    def axe_x(self, value):
        Field.axe_x.fset(self, value)
        for field in self.fields:
            field.axe_x = value

    @Field.axe_y.setter
    def axe_y(self, value):
        Field.axe_y.fset(self, value)
        for field in self.fields:
            field.axe_y = value

    @Field.unit_x.setter
    def unit_x(self, value):
        Field.unit_x.fset(self, value)
        for field in self.fields:
            field.unit_x = value

    @Field.unit_y.setter
    def unit_y(self, value):
        Field.unit_y.fset(self, value)
        for field in self.fields:
            field.unit_y = value

    @property
    def mask(self):
        dim = (len(self.fields), self.shape[0], self.shape[1])
        mask_f = np.empty(dim, dtype=bool)
        for i, field in enumerate(self.fields):
            mask_f[i, :, :] = field.mask[:, :]
        return mask_f

    @property
    def mask_as_sf(self):
        dim = len(self.fields)
        mask_f = np.empty(dim, dtype=object)
        for i, field in enumerate(self.fields):
            mask_f[i] = field.mask_as_sf
        return mask_f

    @property
    def times(self):
        return self.__times

    @times.setter
    def times(self, values):
        if not isinstance(values, ARRAYTYPES):
            raise TypeError()
        if len(self.__times) != len(values):
            raise ValueError()
        self.__times = values

    @times.deleter
    def times(self):
        raise Exception("Nope, can't do that")
    # TODO : HERE !!!!!

    @property
    def unit_times(self):
        return self.__unit_times

    @unit_times.setter
    def unit_times(self, new_unit_times):
        if isinstance(new_unit_times, unum.Unum):
            if new_unit_times.asNumber() == 1:
                self.__unit_times = new_unit_times
            else:
                raise ValueError()
        elif isinstance(new_unit_times, STRINGTYPES):
            self.__unit_times == make_unit(new_unit_times)
        else:
            raise TypeError()

    @unit_times.deleter
    def unit_times(self):
        raise Exception("Nope, can't do that")

    @property
    def unit_values(self):
        if len(self.fields) != 0:
            return self[0].unit_values

    ### Watchers ###
    def get_mean_field(self, nmb_min=1):
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
        result_f = self.fields[0].copy()
        mask_cum = np.zeros(self.shape)
        for field in self.fields[1::]:
            result_f += field
            mask_cum += np.logical_not(field.mask)
        result_f /= len(self.fields)
        result_f.mask = mask_cum <= nmb_min
        return result_f

    def get_fluctuant_fields(self, nmb_min_mean=1):
        """
        Calculate the fluctuant fields (fields minus mean field).

        Parameters
        ----------
        nmb_min_mean : number, optional
            Parameter for mean computation (see 'get_mean_field' doc).

        Returns
        -------
        fluct_fields : TemporalScalarFields or TemporalVectorFields object
            Contening fluctuant fields.
        """
        fluct_fields = self.__class__()
        mean_field = self.get_mean_field(nmb_min=nmb_min_mean)
        for field in self.fields:
            fluct_fields.add_field(field - mean_field)
        return fluct_fields

    def get_time_profile(self, component, x, y, ind=False):
        """
        Return a profile contening the time evolution of the given component.

        Parameters
        ----------
        component : string
            Should be an attribute name of the stored fields.
        x, y : numbers
            Wanted position for the time profile, in axis units.
        ind : boolean, optional
            If 'True', values are undersood as indices.

        Returns
        -------
        profile : Profile object

        """
        # check parameters coherence
        if not isinstance(component, STRINGTYPES):
            raise TypeError("'component' must be a string")
        if not isinstance(x, NUMBERTYPES) or not isinstance(y, NUMBERTYPES):
            raise TypeError("'x' and 'y' must be numbers")
        if ind:
            if not (isinstance(x, int) and isinstance(y, int)):
                raise TypeError()
            ind_x = x
            ind_y = y
        else:
            ind_x = self.get_indice_on_axe(1, x, nearest=True)
            ind_y = self.get_indice_on_axe(2, y, nearest=True)
        axe_x, axe_y = self.axe_x, self.axe_y
        if not (0 <= ind_x < len(axe_x) and 0 <= ind_y < len(axe_y)):
            raise ValueError("'x' ans 'y' values out of bounds")
        # getting component values
        dim = (len(self.fields), self.shape[0], self.shape[1])
        compo = np.empty(dim)
        masks = np.empty(dim)
        for i, field in enumerate(self.fields):
            compo[i] = field.__getattribute__(component)
            masks[i] = field.mask
        # gettign others datas
        time = self.times
        unit_time = self.unit_times
        prof_values = np.zeros(len(compo))
        prof_mask = np.zeros(len(compo))
        unit_values = self.unit_values
        # getting position indices
        for i in np.arange(len(compo)):
            prof_values[i] = compo[i, ind_x, ind_y]
            prof_mask[i] = masks[i, ind_x, ind_y]
        prof_values = np.ma.masked_array(prof_values, prof_mask)
        return Profile(time, prof_values, unit_x=unit_time, unit_y=unit_values)

    def get_spectrum(self, component, pt, ind=False, welch_seglen=None,
                     scaling='base', zero_fill=False, mask_error=True):
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
        welch_seglen : integer, optional
            If specified, welch's method is used (dividing signal into
            overlapping segments, and averaging periodogram) with the given
            segments length (in number of points).
        scaling : string, optional
            If 'base' (default), result are in component unit.
            If 'spectrum', the power spectrum is returned (in unit^2).
            If 'density', the power spectral density is returned (in unit^2/Hz)
        zero_fill : boolean
            If True, field masked values are filled by zeros.
        mask_error : boolean
            If 'False', instead of raising an error when masked value appear on
            time profile, '(None, None)' is returned.

        Returns
        -------
        magn_prof : Profile object
            Magnitude spectrum.
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
        x = pt[0]
        y = pt[1]
        # getting ime profile
        time_prof = self.get_time_profile(component, x, y, ind=ind)
        values = time_prof.y
        time = time_prof.x
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
        from scipy.signal import periodogram, welch
        fs = 1/(time[1] - time[0])
        if welch_seglen is None or welch_seglen >= len(time):
            if scaling == 'base':
                frq, magn = periodogram(values, fs, scaling='spectrum')
                magn = np.sqrt(magn)
            else:
                frq, magn = periodogram(values, fs, scaling=scaling)
        else:
            if scaling == 'base':
                frq, magn = welch(values, fs, scaling='spectrum',
                                  nperseg=welch_seglen)
                magn = np.sqrt(magn)
            else:
                frq, magn = welch(values, fs, scaling=scaling,
                                  nperseg=welch_seglen)
        magn_prof = Profile(frq, magn, unit_x=make_unit('Hz'),
                            unit_y=self.unit_values)
        return magn_prof

    def get_spectrum_over_area(self, component, intervalx, intervaly,
                               ind=False, zero_fill=False,
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

        Returns
        -------
        magn_prof : Profile object
            Averaged magnitude spectrum.
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
        axe_x, axe_y = self.axe_x, self.axe_y
        # checking interval values and getting bound indices
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
            ind_x_min = intervalx[0]
            ind_x_max = intervalx[1]
            ind_y_min = intervaly[0]
            ind_y_max = intervaly[1]
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
            ind_x_min = self.get_indice_on_axe(1, intervalx[0])[0]
            ind_x_max = self.get_indice_on_axe(1, intervalx[1])[-1]
            ind_y_min = self.get_indice_on_axe(2, intervaly[0])[0]
            ind_y_max = self.get_indice_on_axe(2, intervaly[1])[-1]
        # Averaging ponctual spectrums
        magn = 0.
        nmb_fields = (ind_x_max - ind_x_min + 1)*(ind_y_max - ind_y_min + 1)
        real_nmb_fields = nmb_fields
        for i in np.arange(ind_x_min, ind_x_max + 1):
            for j in np.arange(ind_y_min, ind_y_max + 1):
                tmp_m = self.get_spectrum(component, [i, j], ind=True,
                                          zero_fill=zero_fill,
                                          mask_error=False)
                # check if the position is masked
                if tmp_m is None:
                    real_nmb_fields -= 1
                else:
                    magn = magn + tmp_m
        if real_nmb_fields == 0:
            raise StandardError("I can't find a single non-masked time profile"
                                ", maybe you will want to try 'zero_fill' "
                                "option")
        magn = magn/real_nmb_fields
        return magn

    ### Modifiers ###
    def add_field(self, field, time=0., unit_times=""):
        """
        Add a field to the existing fields.

        Parameters
        ----------
        field : VectorField or ScalarField object
            The field to add.
        time : number
            time associated to the field.
        unit_time : Unum object
            time unit.
        """
        # TODO : pas de vérification de la cohérence des unitées !
        # checking parameters
        if not isinstance(field, (VectorField, ScalarField)):
            raise TypeError()
        if not isinstance(time, NUMBERTYPES):
            raise TypeError()
        if isinstance(unit_times, unum.Unum):
            if unit_times.asNumber() != 1:
                raise ValueError()
        elif isinstance(unit_times, STRINGTYPES):
            unit_times = make_unit(unit_times)
        else:
            raise TypeError()
        # if this is the first field
        if len(self.fields) == 0:
            self.axe_x = field.axe_x
            self.axe_y = field.axe_y
            self.unit_x = field.unit_x
            self.unit_y = field.unit_y
            self.unit_times = unit_times
            self.__times = np.array([time])
            self.field_type = field.__class__
        # if not
        else:
            # checking field type
            if not isinstance(field, self.field_type):
                raise TypeError()
            # checking axis
            axe_x, axe_y = self.axe_x, self.axe_y
            vaxe_x, vaxe_y = field.axe_x, field.axe_y
            if not np.all(axe_x == vaxe_x) and np.all(axe_y == vaxe_y):
                raise ValueError("Axes of the new field must be consistent "
                                 "with current axes")
            # storing time
            time = (time*self.unit_times/unit_times).asNumber()
            self.__times = np.append(self.__times, time)
        # use default constructor
        Fields.add_field(self, field)
        # sorting the field with time
        self.__sort_field_by_time()

    def remove_field(self, fieldnumber):
        """
        Remove the wanted field.
        """
        self.__times = np.delete(self.times, fieldnumber)
        Fields.remove_field(self, fieldnumber)

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
        tmp_TFS = self.__class__()
        i = 0
        times = self.times
        while True:
            tmp_f = self[i]
            time = times[i]
            if mean:
                for j in np.arange(i + 1, i + nmb_in_interval):
                    tmp_f += self[j]
                    time += times[j]
                tmp_f /= nmb_in_interval
                time /= nmb_in_interval
            tmp_TFS.add_field(tmp_f, time, self.unit_times)
            i += nmb_in_interval
            if i + nmb_in_interval >= len(self):
                break
        return tmp_TFS

    def crop_masked_border(self):
        """
        Crop the masked border of the velocity fields in place.
        """
        # getting big mask (where all the value are masked)
        masks_temp = self.mask
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
        # getting indices where we need to cut
        axe_x_min = np.where(axe_x_m)[0][0]
        axe_x_max = np.where(axe_x_m)[0][-1]
        axe_y_min = np.where(axe_y_m)[0][0]
        axe_y_max = np.where(axe_y_m)[0][-1]
        # trim
        self.trim_area([axe_x_min, axe_x_max], [axe_y_min, axe_y_max],
                       ind=True, inplace=True)

    def trim_area(self, intervalx=None, intervaly=None, full_output=False,
                  ind=False, inplace=False):
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
        inplace : boolean, optional
            If 'True', fields are trimed in place.
        """
        if inplace:
            Field.trim_area(self, intervalx, intervaly, ind=ind,
                            inplace=inplace)
            for field in self.fields:
                field.trim_area(intervalx, intervaly, ind=ind,
                                inplace=inplace)
        else:
            trimfield = self.__class__()
            for i, field in enumerate(self.fields):
                trimfield.add_field(field.trim_area(intervalx, intervaly,
                                                    ind=ind),
                                    self.times[i], self.unit_times)
            return trimfield

    def set_origin(self, x=None, y=None):
        """
        Modify the axis in order to place the origin at the actual point (x, y)

        Parameters
        ----------
        x : number
        y : number
        """
        Field.set_origin(self, x, y)
        Fields.set_origin(self, x, y)

    def copy(self):
        """
        Return a copy of the velocityfields
        """
        return copy.deepcopy(self)

    def __sort_field_by_time(self):
        if len(self.fields) in [0, 1]:
            return None
        ind_sort = np.argsort(self.times)
        self.times = self.times[ind_sort]
        self.fields = self.fields[ind_sort]

    ### Displayers ###
    def display(self, component, kind=None,  fields_ind=None, samecb=False,
                same_axes=False, **plotargs):
        """
        Display a component of the velocity fields.

        Parameters
        ----------
        fieldname : string, optional
            Fields to display ('fields', 'mean_vf', 'turbulent_vf')
        kind : string, optional
            Kind of display wanted.
        fields_ind : array of indices
            Indices of fields to display.
        samecb : boolean, optional
            If 'True', the same color system is used for all the fields.
            You have to pass 'vmin' and 'vmax', to have correct results.
        plotargs : dict, optional
            Arguments passed to the function used to display the vector field.
        """
        # sharing the space between fields
        nmb_fields = len(fields_ind)
        nmb_col = int(np.sqrt(nmb_fields))
        nmb_lines = int(np.ceil(float(nmb_fields)/nmb_col))
        times = self.times
        # If we want only one colorbar
        if samecb:
            if not 'vmin' in plotargs.keys() or not 'vmax' in plotargs.keys():
                raise ValueError()
            fig, axes = plt.subplots(nrows=nmb_lines, ncols=nmb_col,
                                     sharex=same_axes, sharey=same_axes)
            # displaying the wanted fields
            for i, field_ind in enumerate(fields_ind):
                plt.sca(axes.flat[i])
                im = self.fields[field_ind]._display(component=component,
                                                     kind=kind, **plotargs)
                plt.title("t = {:.2f}{}".format(times[i],
                                                self.unit_times.strUnit()))
            # deleting the non-wanted axes
            for ax in axes.flat[nmb_fields::]:
                plt.sca(ax)
                im = self.fields[field_ind]._display(component=component,
                                                     kind=kind, **plotargs)
                fig.delaxes(ax)
            # adding the colorbar
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.90, 0.1, 0.05, .8])
            fig.colorbar(im, cax=cbar_ax)
            plt.tight_layout(rect=[0., 0., 0.85, 1.])
        else:
            fig, axes = plt.subplots(nrows=nmb_lines, ncols=nmb_col,
                                     sharex=same_axes, sharey=same_axes)
            # displaying the wanted fields
            for i, field_ind in enumerate(fields_ind):
                plt.sca(axes.flat[i])
                im = self.fields[field_ind].display(component=component,
                                                    kind=kind, **plotargs)
                plt.title("t = {:.2f}{}".format(times[i],
                                                self.unit_times.strUnit()))
            # deleting the non-wanted axes
            for ax in axes.flat[nmb_fields::]:
                plt.sca(ax)
                im = self.fields[field_ind]._display(component=component,
                                                     kind=kind, **plotargs)
                fig.delaxes(ax)
            plt.tight_layout()

    def display_controlled(self, compo='V', **plotargs):
        """
        Create a windows to display temporals field, controlled by buttons.
        http://matplotlib.org/1.3.1/examples/widgets/buttons.html
        """
        from matplotlib.widgets import Button
        # getting data
        if isinstance(self, TemporalVectorFields):
            if compo == 'V' or compo is None:
                comp = self.fields
            elif compo == 'x':
                comp = self.Vx_as_sf
            elif compo == 'y':
                comp = self.vy_as_sf
            elif compo == 'mask':
                comp = self.mask_as_sf
            else:
                raise ValueError()
        elif isinstance(self, TemporalScalarFields):
            if compo == 'values' or compo is None:
                comp = self.values_as_sf
            elif compo == 'mask':
                comp = self.mask_as_sf
            else:
                raise ValueError()
        else:
            raise TypeError()
        if 'kind' in plotargs.keys():
            kind = plotargs['kind']
        else:
            kind = None

        # button gestion class
        class Index(object):

            def __init__(self, compo, comp, kind):
                self.fig = plt.figure()
                self.incr = 1
                self.ind = 0
                self.compo = compo
                self.comp = comp
                self.kind = kind
                # display initial

            def next(self, event, incr):
                self.ind += incr

            def prev(self, event, incr):
                self.ind += incr

            def update(self):
                # use __uypdate_sf et __update_vf
                pass

        #window creation
        callback = Index(compo, comp, kind)
        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(callback.next)
        bprev = Button(axprev, 'Previous')
        bprev.on_clicked(callback.prev)

    def display_animate(self, compo='V', interval=500, fields_inds=None,
                        repeat=True,
                        **plotargs):
        """
        Display fields animated in time.

        Parameters
        ----------
        compo : string
            Composante to display
        interval : number, optionnal
            interval between two frames in milliseconds.
        fields_ind : array of indices
            Indices of wanted fields. by default, all the fields are displayed
        repeat : boolean, optional
            if True, the animation is repeated infinitely.
        additional arguments can be passed (scale, vmin, vmax,...)
        """
        from matplotlib import animation
        if fields_inds is None:
            fields_inds = len(self.fields)
        # getting data
        if isinstance(self, TemporalVectorFields):
            if compo == 'V' or compo is None:
                comp = self.fields
            elif compo == 'x':
                comp = self.Vx_as_sf
            elif compo == 'y':
                comp = self.vy_as_sf
            elif compo == 'mask':
                comp = self.mask_as_sf
            else:
                raise ValueError()
        elif isinstance(self, TemporalScalarFields):
            if compo == 'values' or compo is None:
                comp = self.values_as_sf
            elif compo == 'mask':
                comp = self.mask_as_sf
            else:
                raise ValueError()
        else:
            raise TypeError()
        if 'kind' in plotargs.keys():
            kind = plotargs['kind']
        else:
            kind = None
        # display a vector field (quiver)
        if isinstance(comp[0], VectorField)\
                and (kind is None or kind == "quiver"):
            fig = plt.figure()
            ax = plt.gca()
            displ = comp[0].display(**plotargs)
            ttl = plt.title('')
            anim = animation.FuncAnimation(fig, self.__update_vf,
                                           frames=fields_inds,
                                           interval=interval, blit=False,
                                           repeat=repeat,
                                           fargs=(fig, ax, displ, ttl, comp,
                                                  compo, plotargs))
            return anim,
        # display a scalar field (contour, contourf or imshow) or a streamplot
        elif isinstance(comp[0], ScalarField)\
                or isinstance(comp[0], VectorField):
            fig = plt.figure()
            ax = plt.gca()
            displ = comp[0].display(**plotargs)
            ttl = plt.suptitle('')
            anim = animation.FuncAnimation(fig, self.__update_sf,
                                           frames=fields_inds,
                                           interval=interval, blit=False,
                                           repeat=repeat,
                                           fargs=(fig, ax, displ, ttl, comp,
                                                  compo, plotargs))
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

    def __update_sf(self, num, fig, ax, displ, ttl, comp, compo, plotargs):
        ax.cla()
        displ = comp[num]._display(**plotargs)
        title = "{}, at t={:.3} {}"\
            .format(compo, float(self.times[num]),
                    self.unit_times.strUnit())
        ttl.set_text(title)
        return displ,

    def __update_vf(self, num, fig, ax, displ, ttl, comp, compo, plotargs):
        vx = np.transpose(comp[num].comp_x)
        vy = np.transpose(comp[num].comp_y)
        mask = np.transpose(comp[num].mask)
        vx = np.ma.masked_array(vx, mask)
        vy = np.ma.masked_array(vy, mask)
        magn = np.transpose(comp[num].magnitude)
        displ.set_UVC(vx, vy, magn)
        title = "{}, at t={:.2f} {}"\
            .format(compo, float(self.times[num]),
                    self.unit_times.strUnit())
        ttl.set_text(title)
        return ax

class TemporalScalarFields(TemporalFields):
    """
    Class representing a set of time-evolving scalar fields.

    Principal methods
    -----------------
    "add_field" : add a scalar field.

    "remove_field" : remove a field.

    "display" : display the scalar field, with these unities.

    "display_animate" : display an animation of a component of the velocity
    fields set.

    "calc_*" : give access to a bunch of derived statistical fields.
    """

    ### Attributes ###
    @property
    def values(self):
        dim = len(self)
        values = np.empty(dim, dtype=object)
        for i, field in enumerate(self.fields):
            values[i] = field
        return values

    @property
    def values_as_sf(self):
        dim = (len(self), self.shape[0], self.shape[1])
        values = np.empty(dim, dtype=float)
        for i, field in enumerate(self.fields):
            values[i, :, :] = field.values[:, :]
        return values

    ### Modifiers ###
    def fill(self, kind='temporal', tof='interplin', value=0.,
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
        value : number
            Value for filling (only usefull with tof='value')
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
            axe_x, axe_y = self.axe_x, self.axe_y
            # getting super mask (0 where no value are masked and where all
            # values are masked)
            super_mask = self.mask
            super_mask = np.sum(super_mask, axis=0)
            super_mask[super_mask == len(self.fields)] = 0
            # loop on each field position
            for i, j in np.argwhere(super_mask):
                prof = self.get_time_profile('values', i, j, ind=True)
                # checking if all time profile value are masked
                if np.all(prof.y.mask):
                    continue
                # getting masked position on profile
                inds_masked = np.where(prof.y.mask)[0]
                # creating interpolation function
                if tof == 'value':
                    def interp_x(x):
                        return value
                elif tof == 'interplin':
                    interp = spinterp.interp1d(prof.x[~prof.y.mask],
                                               prof.y[~prof.y.mask],
                                               kind='linear')
                elif tof == 'interpcub':
                    interp = spinterp.interp1d(prof.x[~prof.y.mask],
                                               prof.y[~prof.y.mask],
                                               kind='cubic')
                # loop on all profile masked points
                for ind_masked in inds_masked:
                    try:
                        interp_val = interp(prof.x[ind_masked])
                    except ValueError:
                        continue
                    # putting interpolated value in the field
                    self[ind_masked].values[i, j] = interp_val
        # spatial interpolation
        elif kind == 'spatial':
            for field in self.fields:
                field.fill(tof=tof, value=value, crop_border=True)

        else:
            raise ValueError("Unknown parameter for 'kind' : {}".format(kind))


class TemporalVectorFields(TemporalFields):
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

    ### Attributes ###
    @property
    def Vx_as_sf(self):
        dim = len(self)
        values = np.empty(dim, dtype=object)
        for i, field in enumerate(self.fields):
            values[i] = field.comp_x_as_sf
        return values

    @property
    def Vx(self):
        dim = (len(self), self.shape[0], self.shape[1])
        values = np.empty(dim, dtype=float)
        for i, field in enumerate(self.fields):
            values[i, :, :] = field.comp_x[:, :]
        return values

    @property
    def Vy_as_sf(self):
        dim = len(self)
        values = np.empty(dim, dtype=object)
        for i, field in enumerate(self.fields):
            values[i] = field.comp_y_as_sf
        return values

    @property
    def Vy(self):
        dim = (len(self), self.shape[0], self.shape[1])
        values = np.empty(dim, dtype=float)
        for i, field in enumerate(self.fields):
            values[i, :, :] = field.comp_y[:, :]
        return values

    ### Watchers ###
    def get_mean_kinetic_energy(self):
        """
        Calculate the mean kinetic energy.
        """
        final_sf = ScalarField()
        mean_vf = self.get_mean_field()
        values_x = mean_vf.comp_x_as_sf
        values_y = mean_vf.comp_y_as_sf
        final_sf = 1./2*(values_x**2 + values_y**2)
        self.mean_kinetic_energy = final_sf

    def get_tke(self):
        """
        Calculate the turbulent kinetic energy.
        """
        turb_vfs = self.get_fluctuant_fields()
        vx_p = turb_vfs.Vx_as_sf
        vy_p = turb_vfs.Vy_as_sf
        tke = []
        for i in np.arange(len(vx_p)):
            tke.append(1./2*(vx_p[i]**2 + vy_p[i]**2))
        return tke

    def get_mean_tke(self):
        tke = self.get_tke()
        mean_tke = tke[0]
        for field in tke[1::]:
            mean_tke += field
        mean_tke /= len(tke)
        return mean_tke

    def get_reynolds_stress(self, nmb_val_min=1):
        """
        Calculate the reynolds stress.
        """
        # getting fluctuating velocities
        turb_vf = self.get_fluctuant_fields()
        u_p = turb_vf.Vx
        v_p = turb_vf.Vy
        mask = turb_vf.mask
        # rs_xx
        rs_xx = np.zeros(self.shape, dtype=float)
        rs_yy = np.zeros(self.shape, dtype=float)
        rs_xy = np.zeros(self.shape, dtype=float)
        mask_rs = np.zeros(self.shape, dtype=bool)
        # boucle sur les points du champ
        for i in np.arange(self.shape[0]):
            for j in np.arange(self.shape[1]):
                # boucle sur le nombre de champs
                nmb_val = 0
                for n in np.arange(len(turb_vf.fields)):
                    # check if masked
                    if not mask[n][i, j]:
                        rs_yy[i, j] += v_p[n][i, j]**2
                        rs_xx[i, j] += u_p[n][i, j]**2
                        rs_xy[i, j] += u_p[n][i, j]*v_p[n][i, j]
                        nmb_val += 1
                if nmb_val > nmb_val_min:
                    rs_xx[i, j] /= nmb_val
                    rs_yy[i, j] /= nmb_val
                    rs_xy[i, j] /= nmb_val
                else:
                    rs_xx[i, j] = 0
                    rs_yy[i, j] = 0
                    rs_xy[i, j] = 0
                    mask_rs[i, j] = True
        # masking and storing
        axe_x, axe_y = self.axe_x, self.axe_y
        unit_x, unit_y = self.unit_x, self.unit_y
        unit_values = self.unit_values
        rs_xx_sf = ScalarField()
        rs_xx_sf.import_from_arrays(axe_x, axe_y, rs_xx, mask_rs,
                                    unit_x, unit_y, unit_values)
        rs_yy_sf = ScalarField()
        rs_yy_sf.import_from_arrays(axe_x, axe_y, rs_yy, mask_rs,
                                    unit_x, unit_y, unit_values)
        rs_xy_sf = ScalarField()
        rs_xy_sf.import_from_arrays(axe_x, axe_y, rs_xy, mask_rs,
                                    unit_x, unit_y, unit_values)
        return (rs_xx_sf, rs_yy_sf, rs_xy_sf)

    ### Modifiers ###
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
            axe_x, axe_y = self.axe_x, self.axe_y
            # getting super mask (0 where no value are masked and where all
            # values are masked)
            masks = self.mask
            sum_masks = np.sum(masks, axis=0)
            super_mask = np.logical_and(0 < sum_masks,
                                        sum_masks < len(self.fields) - 2)
            # loop on each field position
            for i, j in np.argwhere(super_mask):
                prof_x = self.get_time_profile('comp_x', i, j, ind=True)
                prof_y = self.get_time_profile('comp_y', i, j, ind=True)
                # getting masked position on profile
                inds_masked_x = np.where(prof_x.y.mask)[0]
                inds_masked_y = np.where(prof_y.y.mask)[0]
                # creating interpolation function
                if tof == 'value':
                    def interp_x(x):
                        return value[0]
                    def interp_y(x):
                        return value[1]
                elif tof == 'interplin':
                    prof_filt = np.logical_not(prof_x.y.mask)
                    interp_x = spinterp.interp1d(prof_x.x[prof_filt],
                                                 prof_x.y[prof_filt],
                                                 kind='linear')
                    prof_filt = np.logical_not(prof_y.y.mask)
                    interp_y = spinterp.interp1d(prof_y.x[prof_filt],
                                                 prof_y.y[prof_filt],
                                                 kind='linear')
                elif tof == 'interpcub':
                    prof_filt = np.logical_not(prof_x.y.mask)
                    interp_x = spinterp.interp1d(prof_x.x[prof_filt],
                                                 prof_x.y[prof_filt],
                                                 kind='cubic')
                    prof_filt = np.logical_not(prof_y.y.mask)
                    interp_y = spinterp.interp1d(prof_y.x[prof_filt],
                                                 prof_y.y[prof_filt],
                                                 kind='cubic')
                # loop on all x profile masked points
                for ind_masked in inds_masked_x:
                    try:
                        interp_val = interp_x(prof_x.x[ind_masked])
                    except ValueError:
                        continue
                    # putting interpolated value in the field
                    self[ind_masked].comp_x[i, j] = interp_val
                    self[ind_masked].mask[i, j] = False
                # loop on all y profile masked points
                for ind_masked in inds_masked_y:
                    try:
                        interp_val = interp_y(prof_y.x[ind_masked])
                    except ValueError:
                        continue
                    # putting interpolated value in the field
                    self[ind_masked].comp_y[i, j] = interp_val
                    self[ind_masked].mask[i, j] = False
        # spatial interpolation
        elif kind == 'spatial':
            for field in self.fields:
                field.fill(tof=tof, value=value, crop_border=True)

        else:
            raise ValueError("Unknown parameter for 'kind' : {}".format(kind))


class SpatialScalarFields(Fields):
    pass


class SpatialVectorFields(Fields):
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
            if isinstance(component, (VectorField, ScalarField)):
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
