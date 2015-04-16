# -*- coding: utf-8 -*-
"""
IMTreatment module

    Auteur : Gaby Launay
"""

import scipy.interpolate as spinterp
import scipy.ndimage.measurements as msr
import scipy.optimize as spopt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import Plotlib as pplt
# import guiqwt.pyplot as plt
import numpy as np
import pdb
import unum
import unum.units as units
import copy
from scipy import ndimage
from scipy import stats
try:
    units.counts = unum.Unum.unit('counts')
    units.pixel = unum.Unum.unit('pixel')
except:
    pass

ARRAYTYPES = (np.ndarray, list, tuple)
NUMBERTYPES = (int, long, float, complex, np.float, np.float16, np.float32,
               np.float64, np.int, np.int16, np.int32, np.int64, np.int8)
STRINGTYPES = (str, unicode)
MYTYPES = ('Profile', 'ScalarField', 'VectorField', 'VelocityField',
           'VelocityFields', 'TemporalVelocityFields', 'patialVelocityFields')


class ShapeError(StandardError):
    pass


class PTest(object):
    """
    Decorator used to test input parameters types.
    """
    #FIXME:  +++ need to use OrderedDict instead of classical dicts +++

    def __init__(self, *types, **kwtypes):
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
                    wanted_types = str(type(kwargs[key]))\
                        .replace('<type ', '')\
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
    brackets = ['(', ')', '[', ']']
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


class Points(object):
    """
    Class representing a set of points.
    You can use 'make_unit' to provide unities.

    Parameters
    ----------
    xy : nx2 array.
        Representing the coordinates of each point of the set (n points).
    v : n array, optional
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

    ### Operators ###
    def __init__(self, xy=np.empty((0, 2), dtype=float), v=[],
                 unit_x='', unit_y='', unit_v='', name=''):
        """
        Points builder.
        """
        self.__v = []
        self.xy = xy
        self.v = v
        self.unit_v = unit_v
        self.unit_x = unit_x
        self.unit_y = unit_y
        self.name = name

    def __iter__(self):
        if self.v is None or len(self.v) == 0:
            for i in np.arange(len(self.xy)):
                yield self.xy[i]
        else:
            for i in np.arange(len(self.xy)):
                yield self.xy[i], self.v[i]

    def __len__(self):
        return self.xy.shape[0]

    def __add__(self, another):
        if isinstance(another, Points):
            # trivial additions
            if len(self.xy) == 0:
                return another.copy()
            elif len(another.xy) == 0:
                return self.copy()
            # checking unit systems
            if len(self.xy) != 0:
                try:
                    self.unit_x + another.unit_x
                    self.unit_y + another.unit_y
                    if self.v is not None and another.v is not None:
                        self.unit_v + another.unit_v
                except unum.IncompatibleUnitsError:
                    raise ValueError("Units system are not the same")
            else:
                self.unit_x = another.unit_x
                self.unit_y = another.unit_y
                if another.v is not None:
                    self.unit_v = another.unit_v
            # compacting coordinates
            if another.xy.shape == (0,):
                new_xy = self.xy
            elif self.xy.shape == (0,):
                xy = another.xy
                xy[:, 0] = xy[:, 0]*(self.unit_x/another.unit_x).asNumber()
                xy[:, 1] = xy[:, 1]*(self.unit_y/another.unit_y).asNumber()
                new_xy = xy
            elif another.xy.shape == (0,) and self.xy.shape == (0,):
                new_xy = np.array([[]])
            else:
                xy = another.xy
                xy[:, 0] = xy[:, 0]*(self.unit_x/another.unit_x).asNumber()
                xy[:, 1] = xy[:, 1]*(self.unit_y/another.unit_y).asNumber()
                new_xy = np.append(self.xy, xy, axis=0)
            # testing v presence
            v_presence = True
            if self.v is None and another.v is None:
                if len(self.xy) != 0:
                    v_presence = False
            elif self.v is not None and another.v is not None:
                    v_presence = True
            else:
                raise Exception()
            # compacting points and returning
            if v_presence:
                if self.v is None and another.v is None:
                    v = np.array([])
                elif self.v is None:
                    v = another.v*(self.unit_v/another.unit_v).asNumber()
                elif another.v is None:
                    v = self.v
                else:
                    v_tmp = another.v*(self.unit_v/another.unit_v).asNumber()
                    v = np.append(self.v, v_tmp)
                return Points(new_xy, v,
                              unit_x=self.unit_x,
                              unit_y=self.unit_y,
                              unit_v=self.unit_v)
            else:
                return Points(new_xy,
                              unit_x=self.unit_x,
                              unit_y=self.unit_y)
        else:
            raise StandardError("You can't add {} to Points objects"
                                .format(type(another)))

    ### Attributes ###
    @property
    def xy(self):
        # XXX: for compatibility purpose, to remove
        try:
            return self.__xy
        except AttributeError:
            return self.__dict__['xy']

    @xy.setter
    def xy(self, values):
        if not isinstance(values, ARRAYTYPES):
            raise TypeError("'xy' should be an array, not {}"
                            .format(type(values)))
        values = np.array(values, subok=True)
        if not values.ndim == 2:
            raise ValueError("ndim of xy is {} and should be 2"
                             .format(values.ndim))
        if not values.shape[1] == 2:
            raise ValueError()
        self.__xy = values
        # XXX: for compatibility purpose, to remove
        try:
            if len(values) != len(self.__v):
                self.__v == np.array([])
        except AttributeError:
            if len(values) != len(self.__dict__['v']):
                self.__dict__['v'] == np.array([])

    @xy.deleter
    def xy(self):
        raise Exception("Nope, can't do that")

    @property
    def v(self):
        # XXX: for compatibility purpose, to remove
        try:
            return self.__v
        except AttributeError:
            return self.__dict__['v']

    @v.setter
    def v(self, values):
        if not isinstance(values, ARRAYTYPES):
            raise TypeError
        values = np.array(values, subok=True)
        if not values.ndim == 1:
            raise ValueError()
        if not len(values) in [0, len(self.__xy)]:
            raise ValueError()
        self.__v = values

    @v.deleter
    def v(self):
        raise Exception("Nope, can't do that")

    @property
    def unit_x(self):
        # XXX: for compatibility purpose, to remove
        try:
            return self.__unit_x
        except AttributeError:
            return self.__dict__['unit_x']

    @unit_x.setter
    def unit_x(self, unit):
        if isinstance(unit, unum.Unum):
            self.__unit_x = unit
        elif isinstance(unit, STRINGTYPES):
            try:
                self.__unit_x = make_unit(unit)
            except (ValueError, TypeError):
                raise Exception()
        else:
            raise Exception()

    @unit_x.deleter
    def unit_x(self):
        raise Exception("Nope, can't delete 'unit_x'")

    @property
    def unit_y(self):
        # XXX: for compatibility purpose, to remove
        try:
            return self.__unit_y
        except AttributeError:
            return self.__dict__['unit_y']

    @unit_y.setter
    def unit_y(self, unit):
        if isinstance(unit, unum.Unum):
            self.__unit_y = unit
        elif isinstance(unit, STRINGTYPES):
            try:
                self.__unit_y = make_unit(unit)
            except (ValueError, TypeError):
                raise Exception()
        else:
            raise Exception()

    @unit_y.deleter
    def unit_y(self):
        raise Exception("Nope, can't delete 'unit_y'")

    @property
    def unit_v(self):
        # XXX: for compatibility purpose, to remove
        try:
            return self.__unit_v
        except AttributeError:
            return self.__dict__['unit_v']

    @unit_v.setter
    def unit_v(self, unit):
        if isinstance(unit, unum.Unum):
            self.__unit_v = unit
        elif isinstance(unit, STRINGTYPES):
            try:
                self.__unit_v = make_unit(unit)
            except (ValueError, TypeError):
                raise Exception()
        else:
            raise Exception()

    @unit_v.deleter
    def unit_v(self):
        raise Exception("Nope, can't delete 'unit_v'")

    @property
    def name(self):
        # XXX: for compatibility purpose, to remove
        try:
            return self.__name
        except AttributeError:
            return self.__dict__['name']

    @name.setter
    def name(self, name):
        if isinstance(name, STRINGTYPES):
            self.__name = name
        else:
            raise Exception()

    @name.deleter
    def name(self):
        raise Exception("Nope, can't delete 'name'")

    ### Properties ###

    ### Watchers ###
    def copy(self):
        """
        Return a copy of the Points object.
        """
        return copy.deepcopy(self)

    def get_points_density(self, bw_method=None, resolution=100,
                           output_format=None, raw=False):
        """
        Return a ScalarField with points density.

        Parameters:
        -----------
        bw_method : str, scalar or callable, optional
            The method used to calculate the estimator bandwidth.
            This can be ‘scott’, ‘silverman’, a scalar constant or
            a callable. If a scalar, this will be used as std
            (it should aprocimately be the size of the density
            node you want to see).
            If a callable, it should take a gaussian_kde instance as only
            parameter and return a scalar. If None (default), ‘scott’ is used.
            See Notes for more details.
        resolution : integer or 2x1 tuple of integers, optional
            Resolution for the resulting field.
            Can be a tuple in order to specify resolution along x and y.
        output_format : string, optional
            'normalized' (default) : give position probability
                                     (integral egal 1).
            'ponderated' : give position probability ponderated by the number
                           or points (integral egal number of points).
            'concentration' : give local concentration (in point per surface).

        raw : boolean, optional
            If 'False' (default), return a ScalarField object,
            if 'True', return numpy array.

        Returns
        -------
        density : array, ScalarField object or None
            Return 'None' if there is not enough points in the cloud.
        """
        # checking points length
        if len(self.xy) < 2:
            return None
        resolution = int(resolution)
        # getting data
        min_x = np.min(self.xy[:, 0])
        max_x = np.max(self.xy[:, 0])
        min_y = np.min(self.xy[:, 1])
        max_y = np.max(self.xy[:, 1])
        # checking parameters
        if isinstance(resolution, int):
            width_x = max_x - min_x
            width_y = max_y - min_y
            if width_x == 0 and width_y == 0:
                raise Exception()
            elif width_x == 0:
                min_x = min_x - width_y/2.
                max_x = max_x + width_y/2.
                width_x = width_y
                res_x = resolution
                res_y = resolution
            elif width_y == 0:
                min_y = min_y - width_x/2.
                max_y = max_y + width_x/2.
                width_y = width_x
                res_x = resolution
                res_y = resolution
            elif width_x > width_y:
                res_x = resolution
                res_y = int(np.round(resolution*width_y/width_x))
            else:
                res_y = resolution
                res_x = int(np.round(resolution*width_x/width_y))
        elif isinstance(resolution, ARRAYTYPES):
            if len(resolution) != 2:
                raise ValueError()
            res_x = resolution[0]
            res_y = resolution[1]
        else:
            raise TypeError()
        if res_x < 2 or res_y < 2:
            raise ValueError()
        # check potential singular covariance matrix situations
        if (np.all(self.xy[:, 0] == self.xy[0, 0])
                or np.all(self.xy[:, 1] == self.xy[0, 1])):
            return None
        # get kernel using scipy
        if isinstance(bw_method, NUMBERTYPES):
            if width_x > width_y:
                bw_method /= width_y
            else:
                bw_method /= width_x
        kernel = stats.gaussian_kde(self.xy.transpose(),
                                    bw_method=bw_method)
        # little adaptation to avoi streched density map

        if width_x > width_y:
            kernel.inv_cov[0, 0] *= (width_x/width_y)**2
        else:
            kernel.inv_cov[1, 1] *= (width_y/width_x)**2
        kernel.inv_cov[0, 1] *= 0
        kernel.inv_cov[1, 0] *= 0
        # creating grid
        dx = (max_x - min_x)/10.
        dy = (max_y - min_y)/10.
        axe_x = np.linspace(min_x - dx, max_x + dx, res_x)
        axe_y = np.linspace(min_y - dy, max_y + dy, res_y)
        X, Y = np.meshgrid(axe_x, axe_y)
        X = X.flatten()
        Y = Y.flatten()
        positions = np.array([[X[i], Y[i]]
                              for i in np.arange(len(X))]).transpose()
        # estimate density
        values = kernel(positions)
        values = values.reshape((res_y, res_x)).transpose()
        if output_format is None or output_format == "normalized":
            unit_values = make_unit('')
        elif output_format == 'ponderated':
            values = values*len(self.xy)
            unit_values = make_unit('')
        elif output_format == "percentage":
            values = values*100
            unit_values = make_unit('')
        elif output_format == "concentration":
            unit_values = 1/self.unit_x/self.unit_y
            values = values*len(self.xy)
        else:
            raise ValueError()
        # return
        if np.all(np.isnan(values)) or np.all(values == np.inf) :
            return None
        if raw:
            return values
        else:
            sf = ScalarField()
            sf.import_from_arrays(axe_x, axe_y, values, mask=False,
                                  unit_x=self.unit_x, unit_y=self.unit_y,
                                  unit_values=unit_values)
            return sf

    def get_points_density2(self, res, subres=None, raw=False,
                            ponderated=False):
        """
        Return a ScalarField with points density.

        Parameters:
        -----------
        res : number or 2x1 array of numbers
            fdensity field number of subdivision.
            Can be the same number for both axis,  or one number per axis
            (need to give a tuple).
        raw : boolean, optional
            If 'False' (default), return a ScalarField object,
            if 'True', return numpy array.
        ponderated : boolean, optiona
            If 'True', values associated to points are used to ponderate the
            density field. Default is 'False'.
        subres : odd integer, optional
            If specified, a subgrid of resolution res*subres is used to
            make result more accurate.
        """
        # checking parameters
        if isinstance(res, int):
            res_x = res
            res_y = res
        elif isinstance(res, ARRAYTYPES):
            if len(res) != 2:
                raise ValueError()
            res_x = res[0]
            res_y = res[1]
        else:
            raise TypeError()
        if not isinstance(raw, bool):
            raise TypeError()
        if not isinstance(ponderated, bool):
            raise TypeError()
        if isinstance(subres, int) and subres > 0:
            subres = np.floor(subres/2)*2
            subres2 = (subres)/2
        elif subres is None:
            pass
        else:
            raise TypeError()
        # If we use a subgrid
        if subres is not None:
            # creating grid
            min_x = np.min(self.xy[:, 0])
            max_x = np.max(self.xy[:, 0])
            min_y = np.min(self.xy[:, 1])
            max_y = np.max(self.xy[:, 1])
            dx = (max_x - min_x)/(res_x)
            dy = (max_y - min_y)/(res_y)
            sub_dx = dx/subres
            sub_dy = dy/subres
            axe_x = np.arange(min_x - dx/2, max_x + dx/2 + sub_dx, sub_dx)
            axe_y = np.arange(min_y - dy/2, max_y + dy/2 + sub_dy, sub_dy)
            values = np.zeros((len(axe_x), len(axe_y)))
            # filling grid with density
            for i, pt in enumerate(self.xy):
                x = pt[0]
                y = pt[1]
                ind_x = np.argmin(np.abs(axe_x - x))
                ind_y = np.argmin(np.abs(axe_y - y))
                slic_x = slice(ind_x - subres2 + 1, ind_x + subres2)
                slic_y = slice(ind_y - subres2 + 1, ind_y + subres2)
                if ponderated:
                    values[slic_x, slic_y] += self.v[i]
                else:
                    values[slic_x, slic_y] += 1
            values /= (dx*dy)
            values = values[subres2:-subres2, subres2:-subres2]
            axe_x = axe_x[subres2:-subres2]
            axe_y = axe_y[subres2:-subres2]
        # if we do not use a subgrid
        else:
            # creating grid
            min_x = np.min(self.xy[:, 0])
            max_x = np.max(self.xy[:, 0])
            min_y = np.min(self.xy[:, 1])
            max_y = np.max(self.xy[:, 1])
            axe_x, dx = np.linspace(min_x, max_x, res_x, retstep=True)
            axe_y, dy = np.linspace(min_y, max_y, res_y, retstep=True)
            values = np.zeros((len(axe_x), len(axe_y)))
            # filling grid with density
            for i, pt in enumerate(self.xy):
                x = pt[0]
                y = pt[1]
                ind_x = np.argmin(np.abs(axe_x - x))
                ind_y = np.argmin(np.abs(axe_y - y))
                if ponderated:
                    values[ind_x, ind_y] += self.v[i]
                else:
                    values[ind_x, ind_y] += 1
            values /= (dx*dy)
        # return the field
        if raw:
            return values
        else:
            sf = ScalarField()
            if ponderated:
                unit_values = self.unit_v/self.unit_x/self.unit_y
            else:
                unit_values = 1/self.unit_x/self.unit_y
            sf.import_from_arrays(axe_x, axe_y, values, mask=False,
                                  unit_x=self.unit_x, unit_y=self.unit_y,
                                  unit_values=unit_values)
            return sf

    def get_velocity(self, incr=1, smooth=None):
        """
        Assuming that associated 'v' values are times for each points,
        compute the velocity of the trajectory.

        Parameters
        ----------
        incr : integer, optional
            Increment use to get used points (default is 1).
        smooth : number, optional
            Size of smoothing (gaussian smoothing) applied on x and y values,
            before computing velocities.

        Return
        ------
        Vx : Profile object
            Profile of x velocity versus time.
        Vy : Profile object
            Profile of y velocity versus time.
        """
        if smooth is not None:
            if not isinstance(smooth, NUMBERTYPES):
                raise TypeError()
            if smooth < 0:
                raise ValueError()
        # checking 'v' presence
        if len(self.v) == 0:
            raise Exception()
        # sorting points by time
        ind_sort = np.argsort(self.v)
        times = self.v[ind_sort]
        dt = times[1::] - times[:-1]
        xy = self.xy[ind_sort]
        x = xy[:, 0]
        y = xy[:, 1]
        # using increment if necessary
        if incr != 1:
            x = x[::incr]
            y = y[::incr]
            times = times[::incr]
            dt = times[1::] - times[:-1]
        # smoothing if necessary
        if smooth is not None:
            x = ndimage.gaussian_filter(x, smooth, mode='nearest')
            y = ndimage.gaussian_filter(y, smooth, mode='nearest')
        # getting velocity between points
        Vx = np.array([(x[i + 1] - x[i])/dt[i]
                       for i in np.arange(len(x) - 1)])
        Vy = np.array([(y[i + 1] - y[i])/dt[i]
                       for i in np.arange(len(y) - 1)])
        # returning profiles
        unit_Vx = self.unit_x/self.unit_v
        Vx *= unit_Vx.asNumber()
        unit_Vx /= unit_Vx.asNumber()
        time_x = times[:-1] + dt/2.
        prof_x = Profile(time_x, Vx, mask=False, unit_x=self.unit_v,
                         unit_y=unit_Vx)
        unit_Vy = self.unit_y/self.unit_v
        Vy *= unit_Vy.asNumber()
        unit_Vy /= unit_Vy.asNumber()
        time_y = times[:-1] + dt/2.
        prof_y = Profile(time_y, Vy, mask=False, unit_x=self.unit_v,
                         unit_y=unit_Vy)
        return prof_x, prof_y

    def get_evolution_on_sf(self, SF, axe_x=None):
        """
        Return the evolution of the value represented by a scalar field, on
        the path of the trajectory.

        Parameters
        ----------
        SF : ScalarField object
        axe_x : string, optional
            What put in the x axis (can be 'x', 'y', 'v').
            default is 'v' when available and 'x' else.

        Returns
        -------
        evol : Profile object
        """
        # check parameters
        if not isinstance(SF, ScalarField):
            raise TypeError()
        if len(self.xy) == 0:
            return Profile()
        if axe_x is None:
            if len(self.v) == len(self.xy):
                axe_x = 'v'
            else:
                axe_x = 'x'
        if not isinstance(axe_x, STRINGTYPES):
            raise TypeError()
        # get x values
        if axe_x == 'v':
            if len(self.v) == 0:
                raise ValueError()
            x_prof = self.v
            unit_x = self.unit_v
        elif axe_x == 'x':
            x_prof = self.xy[:, 0]
            unit_x = self.unit_x
        elif axe_x == 'y':
            x_prof = self.xy[:, 1]
            unit_x = self.unit_y
        else:
            raise ValueError()
        # get the y value
        y_prof = np.empty((len(self.xy)), dtype=float)
        for i, pt in enumerate(self.xy):
            y_prof[i] = SF.get_value(*pt, ind=False, unit=False)
        mask = np.isnan(y_prof)
        unit_y = SF.unit_values
        # returning
        evol = Profile(x_prof, y_prof, mask=mask, unit_x=unit_x,
                       unit_y=unit_y)
        return evol

    def get_evolution_on_tsf(self, TSF, axe_x=None):
        """
        Return the evolution of the value represented by scalar fields, on
        the path of the trajectory.
        Timse of the TSF must be consistent with the times of the Points.

        Parameters
        ----------
        TSF : TemporalScalarField object
        axe_x : string, optional
            What put in the x axis (can be 'x', 'y', 'v').
            default is 'v' (associated with time)

        Returns
        -------
        evol : Profile object
        """
        # check parameters
        if not isinstance(TSF, TemporalScalarFields):
            raise TypeError()
        if len(self.xy) == 0:
            return Profile()
        if axe_x is None:
            axe_x = 'v'
        if not isinstance(axe_x, STRINGTYPES):
            raise TypeError()
        # get x values
        if axe_x == 'v':
            if len(self.v) == 0:
                raise ValueError()
            x_prof = self.v
            unit_x = self.unit_v
        elif axe_x == 'x':
            x_prof = self.xy[:, 0]
            unit_x = self.unit_x
        elif axe_x == 'y':
            x_prof = self.xy[:, 1]
            unit_x = self.unit_y
        else:
            raise ValueError()
        # get the y value
        times = self.v
        y_prof = np.empty((len(self.xy)), dtype=float)
        for i, pt in enumerate(self.xy):
            time = times[i]
            SF = TSF.fields[TSF.times == time][0]
            y_prof[i] = SF.get_value(*pt, ind=False, unit=False)
        mask = np.isnan(y_prof)
        unit_y = TSF.unit_values
        # returning
        evol = Profile(x_prof, y_prof, mask=mask, unit_x=unit_x,
                       unit_y=unit_y)
        return evol

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

    ### Modifiers ###
    def add(self, pt, v=None):
        """
        Add a new point.

        Parameters
        ----------
        pt : 2x1 array of numbers
            Point to add.
        v : number, optional
            Value of the point (needed if other points have values).
        """
        # check parameters
        if not isinstance(pt, ARRAYTYPES):
            raise TypeError()
        pt = np.array(pt, subok=True)
        if not pt.shape == (2,):
            raise ValueError()
        if not isinstance(pt[0], NUMBERTYPES):
            raise TypeError()
        if v is not None:
            if not isinstance(v, NUMBERTYPES):
                raise TypeError()
        # store new data
        self.xy = np.append(self.xy, [pt], axis=0)
        if v is None and self.v.shape[0] != 0:
            raise ValueError('You should specify an associated value : v')
        if v is not None:
            self.v = np.append(self.v, v)

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
        tmp_v = self.v.copy()
        self.xy = np.delete(self.xy, ind, axis=0)
        if len(self.v) == len(self.xy) + 1:
            self.v = np.delete(tmp_v, ind, axis=0)

    def change_unit(self, axe, new_unit):
        """
        Change the unit of an axe.

        Parameters
        ----------
        axe : string
            'y' for changing the profile y axis unit
            'x' for changing the profile x axis unit
            'v' for changing the profile values unit
        new_unit : Unum.unit object or string
            The new unit.
        """
        if isinstance(new_unit, STRINGTYPES):
            new_unit = make_unit(new_unit)
        if not isinstance(new_unit, unum.Unum):
            raise TypeError()
        if not isinstance(axe, STRINGTYPES):
            raise TypeError()
        if axe == 'x':
            old_unit = self.unit_x
            new_unit = old_unit.asUnit(new_unit)
            fact = new_unit.asNumber()
            self.xy[:, 0] *= fact
            self.unit_x = new_unit/fact
        elif axe == 'y':
            old_unit = self.unit_y
            new_unit = old_unit.asUnit(new_unit)
            fact = new_unit.asNumber()
            self.xy[:, 1] *= fact
            self.unit_y = new_unit/fact
        elif axe == 'v':
            old_unit = self.unit_v
            new_unit = old_unit.asUnit(new_unit)
            fact = new_unit.asNumber()
            self.v *= fact
            self.unit_v = new_unit/fact
        else:
            raise ValueError()

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

    def scale(self, scalex=1., scaley=1., scalev=1., inplace=False):
        """
        Change the scale of the axis.

        Parameters
        ----------
        scalex, scaley, scalev : numbers
            scales along x, y and v
        inplace : boolean, optional
            If 'True', scaling is done in place, else, a new instance is
            returned.
        """
        # check params
        if not isinstance(scalex, NUMBERTYPES):
            raise TypeError()
        if not isinstance(scaley, NUMBERTYPES):
            raise TypeError()
        if not isinstance(scalev, NUMBERTYPES):
            raise TypeError()
        if not isinstance(inplace, bool):
            raise TypeError()
        if inplace:
            tmp_pt = self
        else:
            tmp_pt = self.copy()
        # loop
        if scalex != 1. or scaley != 1.:
            tmp_pt.xy *= np.array([scalex, scaley])
        if scalev != 1.:
            tmp_pt.v *= scalev
        # returning
        if not inplace:
            return tmp_pt

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
        if len(tmp_pts.v) != 0:
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
            return [self]
        if len(self) != len(self.v):
            raise StandardError()
        pts_tupl = []
        for i in np.arange(len(self)):
            pts_tupl.append(Points([self.xy[i]], [self.v[i]], self.unit_x,
                                   self.unit_y, self.unit_v, self.name))
        return pts_tupl

    def sort(self, ref='x', inplace=False):
        """
        Sort the points according to the reference.

        Parameters
        ----------
        ref : string or array of indice
            can be 'x', 'y' or 'v' to sort according those values or an
            array of indice
        inplace ; boolean
            If 'True', sort in place, else, return an new sorted instance.
        """
        # check parameters
        if isinstance(ref, STRINGTYPES):
            if ref not in ['x', 'y', 'v']:
                raise ValueError
        elif isinstance(ref, ARRAYTYPES):
            ref = np.array(ref, dtype=int)
            if len(ref) != len(self.xy):
                raise ValueError()
        else:
            raise TypeError()
        if not isinstance(inplace, bool):
            raise TypeError()
        # get order
        if ref == 'x':
            order = np.argsort(self.xy[:, 0])
        elif ref == 'y':
            order = np.argsort(self.xy[:, 1])
        elif ref == 'v':
            if len(self.v) == 0:
                raise ValueError()
            order = np.argsort(self.v)
        else:
            order = ref
        # reordering
        if inplace:
            tmp_pt = self
        else:
            tmp_pt = self.copy()
        tmp_pt.xy = tmp_pt.xy[order]
        tmp_pt.v = tmp_pt.v[order]
        # returning
        if not inplace:
            return tmp_pt

    ### Displayers ###
    def _display(self, kind=None, reverse=False, **plotargs):
        if kind is None:
            if self.v is None:
                kind = 'plot'
            else:
                kind = 'scatter'
        if reverse:
            x_values = self.xy[:, 1]
            y_values = self.xy[:, 0]
        else:
            x_values = self.xy[:, 0]
            y_values = self.xy[:, 1]
        if kind == 'scatter':
            if self.v is None:
                plot = plt.scatter(x_values, y_values, **plotargs)
            else:
                if not 'cmap' in plotargs:
                    plotargs['cmap'] = plt.cm.jet
                if not 'c' in plotargs:
                    plotargs['c'] = self.v
                plot = plt.scatter(x_values, y_values, **plotargs)
        elif kind == 'plot':
            plot = plt.plot(x_values, y_values, **plotargs)
        return plot

    def display(self, kind=None, reverse=False, **plotargs):
        """
        Display the set of points.

        Parameters
        ----------
        kind : string, optional
            Can be 'plot' (default if points have not values).
            or 'scatter' (default if points have values).
        reverse : boolean, optional
            If 'True', axis are reversed.
        """
        plot = self._display(kind, **plotargs)
        if len(self.v) != 0 and kind is not 'plot':
            cb = plt.colorbar(plot)
            cb.set_label(self.unit_v.strUnit())
        if reverse:
            plt.ylabel('X ' + self.unit_x.strUnit())
            plt.xlabel('Y ' + self.unit_y.strUnit())
        else:
            plt.xlabel('X ' + self.unit_x.strUnit())
            plt.ylabel('Y ' + self.unit_y.strUnit())
        if self.name is None:
            plt.title('Set of points')
        else:
            plt.title(self.name)
        return plot


class OrientedPoints(Points):
    """
    Class representing a set of points with associated orientations.
    You can use 'make_unit' to provide unities.

    Parameters
    ----------
    xy : nx2 arrays.
        Representing the coordinates of each point of the set (n points).
    orientations : nxdx2 array
        Representing the orientations of each point in the set
        (d orientations for each n points). Can be 'None' if a point have no
        orientation.
    v : n array, optional
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

    ### Operators ###
    def __init__(self, xy=np.empty((0, 2), dtype=float), orientations=[], v=[],
                 unit_x='', unit_y='', unit_v='', name=''):
        # check parameters
        if not isinstance(orientations, ARRAYTYPES):
            raise TypeError()
        orientations = np.array(orientations)
        if len(xy) != 0 and not orientations.ndim == 3:
            raise ShapeError("'orientations' must have 3 dimensions, not {}"
                             .format(orientations.ndim))
        if not orientations.shape[0:3:2] != [len(xy), 2]:
            raise ShapeError()
        # initialize data
        Points.__init__(self, xy=xy, v=v, unit_x=unit_x, unit_y=unit_y,
                        unit_v=unit_v, name=name)
        self.orientations = orientations

    def __iter__(self):
        for i in np.arange(len(self.xy)):
            yield Points.__iter__(self)[i], self.orientations[i]

    def __add__(self, obj):
        if isinstance(obj, Points):
            tmp_pts = Points.__add__(self, obj)
            if len(self.xy) == 0:
                tmp_ori = obj.orientations
            elif len(obj.xy) == 0:
                tmp_ori = self.orientations
            else:
                tmp_ori = np.append(self.orientations, obj.orientations, axis=0)
            tmp_opts = OrientedPoints()
            tmp_opts.import_from_Points(tmp_pts, tmp_ori)
            return tmp_opts

    ### Attributes ###
    @property
    def orientations(self):
        return self.__orientations

    @orientations.setter
    def orientations(self, new_ori):
        if not isinstance(new_ori, ARRAYTYPES):
            raise TypeError()
        new_ori = np.array(new_ori)
        if new_ori.dtype not in NUMBERTYPES:
            raise TypeError()
        if len(self.xy) != 0 and new_ori.shape[0:3:2] != (len(self.xy), 2):
            raise ShapeError("'orientations' shape must be (n, d, 2)  (with n"
                             " the number of points ({}) and d the number of "
                             "directions), not {}"
                             .format(len(self.xy), new_ori.shape))
        self.__orientations = new_ori

    ### Watchers ###
    def get_streamlines(self, vf, delta=.25, interp='linear',
                        reverse_direction=False):
        """
        Return the streamlines coming from the points, based on the given
        field.

        Parameters
        ----------
        vf : VectorField or velocityField object
            Field on which compute the streamlines
        delta : number, optional
            Spatial discretization of the stream lines,
            relative to a the spatial discretization of the field.
        interp : string, optional
            Used interpolation for streamline computation.
            Can be 'linear'(default) or 'cubic'
        reverse_direction : boolean, optional
            If True, the streamline goes upstream.

        Returns
        -------
        streams : tuple of Points objects
            Each Points object represent a streamline
        """
        # check parameters
        from .field_treatment import get_streamlines
        if not isinstance(vf, VectorField):
            raise TypeError()
        # getting streamlines
        streams = get_streamlines(vf, self.xy, delta=delta, interp=interp,
                                  reverse_direction=reverse_direction)
        return streams

    def get_streamlines_from_orientations(self, vf, delta=.25, interp='linear',
                                          reverse_direction=False):
        """
        Return the streamlines coming from the points orientations, based on
        the given field.

        Parameters
        ----------
        vf : VectorField or velocityField object
            Field on which compute the streamlines
        delta : number, optional
            Spatial discretization of the stream lines,
            relative to a the spatial discretization of the field.
        interp : string, optional
            Used interpolation for streamline computation.
            Can be 'linear'(default) or 'cubic'
        reverse_direction : boolean or tuple of boolean, optional
            If 'False' (default), the streamline goes downstream.
            If 'True', the streamline goes upstream.
            a tuple of booleans can be specified to apply different behaviors
            to the different orientations

        Returns
        -------
        streams : tuple of Points objects
            Each Points object represent a streamline
        """
        # check parameters
        nmb_dir = self.orientations.shape[1]
        from .field_treatment import get_streamlines
        if not isinstance(vf, VectorField):
            raise TypeError()
        if isinstance(reverse_direction, bool):
            reverse_direction = np.array([reverse_direction]*nmb_dir)
        elif isinstance(reverse_direction, ARRAYTYPES):
            reverse_direction = np.array(reverse_direction)
        else:
            raise TypeError()
        if reverse_direction.shape != (nmb_dir,):
            raise ShapeError()
        # get coef
        coef = np.max([vf.axe_x[1] - vf.axe_x[0],
                       vf.axe_y[1] - vf.axe_y[0]])*2.
        # get streamlines
        streams = []
        # for each points and each directions
        for i, pt in enumerate(self.xy):
            for n in np.arange(nmb_dir):
                if np.all(self.orientations[i, n] == [0, 0]):
                    continue
                # get streamlines
                pt1 = pt - self.orientations[i, n]*coef
                pt2 = pt + self.orientations[i, n]*coef
                reverse = reverse_direction[n]
                tmp_stream = get_streamlines(vf, [pt1, pt2], delta=delta,
                                             interp=interp,
                                             reverse_direction=reverse)
                # if we are out of field
                if tmp_stream is None:
                    continue
                # add the first point
                for st in tmp_stream:
                    st.xy = np.append([pt], st.xy, axis=0)
                streams += tmp_stream
        # returning
        return streams

    ### Modifiers ###
    def import_from_Points(self, pts, orientations):
        """
        Import data from a Points object
        """
        self.xy = pts.xy
        self.v = pts.v
        self.unit_x = pts.unit_x
        self.unit_y = pts.unit_y
        self.unit_v = pts.unit_v
        self.name = pts.name
        self.orientations = orientations

    def add(self, pt, orientations, v=None):
        """
        Add a new point.

        Parameters
        ----------
        pt : 2x1 array of numbers
            Point to add.
        orientations : dx2 array
            orientations associated to the points (d orientations)
        v : number, optional
            Value of the point (needed if other points have values).
        """
        Points.add(self, pt, v)
        if len(self.orientations) == 0:
            self.orientations = np.array([orientations])
        else:
            self.orientations = np.append(self.orientations, [orientations],
                                          axis=0)

    def remove(self, ind):
        """
        Remove the point number 'ind' of the points cloud.
        In place.

        Parameters
        ----------
        ind : integer or array of integer
        """
        Points.remove(self, ind)
        self.orientations = np.delete(self.orientations, ind, axis=0)

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
        tmp_pts : OrientedPoints object
            Trimmed version of the point cloud.
        """
        Points.trim(self, interv_x, interv_y)
        mask = np.zeros(len(self.xy))
        if interv_x is not None:
            out_zone = np.logical_or(self.xy[:, 0] < interv_x[0],
                                     self.xy[:, 0] > interv_x[1])
            mask = np.logical_or(mask, out_zone)
        if interv_y is not None:
            out_zone = np.logical_or(self.xy[:, 1] < interv_y[0],
                                     self.xy[:, 1] > interv_y[1])
            mask = np.logical_or(mask, out_zone)
        self.orientations = self.orientations[~mask]

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
        tmp_pts : OrientedPoints object
            Cutted version of the point cloud.
        """
        Points.cut(self, interv_x, interv_y)
        mask = np.ones(len(self.xy))
        if interv_x is not None:
            out_zone = np.logical_and(self.xy[:, 0] > interv_x[0],
                                      self.xy[:, 0] < interv_x[1])
            mask = np.logical_and(mask, out_zone)
        if interv_y is not None:
            out_zone = np.logical_and(self.xy[:, 1] > interv_y[0],
                                      self.xy[:, 1] < interv_y[1])
            mask = np.logical_and(mask, out_zone)
        self.orientations = self.orientations[~mask]

    def decompose(self):
        """
        Return a tuple of OrientedPoints object, with only one point per
        object.
        """
        if len(self) == 1:
            return [self]
        if len(self) != len(self.v):
            raise StandardError()
        pts_tupl = []
        for i in np.arange(len(self)):
            pts_tupl.append(OrientedPoints([self.xy[i]],
                                           [self.orientations[i]],
                                           [self.v[i]], self.unit_x,
                                           self.unit_y, self.unit_v,
                                           self.name))
        return pts_tupl

    ### Displayers ###
    def _display(self, kind=None, **plotargs):
        # display like a Points object
        plot = Points._display(self, kind=kind, **plotargs)
        if kind is None:
            if self.v is None:
                kind = 'plot'
            else:
                kind = 'scatter'
        # setting color
        if 'color' in plotargs.keys():
            colors = [plotargs.pop('color')]
        else:
            colors = mpl.rcParams['axes.color_cycle']
        # displaying orientation lines
        x_range = plt.xlim()
        Dx = x_range[1] - x_range[0]
        y_range = plt.ylim()
        Dy = y_range[1] - y_range[0]
        coef = np.min([Dx, Dy])/40.
        for i in np.arange(len(self.xy)):
            loc_oris = self.orientations[i]
            if np.all(loc_oris == [[0, 0], [0, 0]]):
                continue
            color = colors[i % len(colors)]
            pt = self.xy[i]
            for ori in loc_oris:
                line_x = [pt[0] - ori[0]*coef, pt[0] + ori[0]*coef]
                line_y = [pt[1] - ori[1]*coef, pt[1] + ori[1]*coef]
                plt.plot(line_x, line_y, color=color)
        return plot


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

    ### Operators ###
    def __init__(self, x=[], y=[], mask=False, unit_x="",
                 unit_y="", name=""):
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
            y = np.array(y)
        if isinstance(mask, bool):
            mask = np.empty(x.shape, dtype=bool)
            mask.fill(False)
        if not isinstance(mask, ARRAYTYPES):
            raise TypeError("'mask' must be an array")
        if not isinstance(mask, (np.ndarray, np.ma.MaskedArray)):
            mask = np.array(mask, dtype=bool)
        if not isinstance(name, STRINGTYPES):
            raise TypeError("'name' must be a string")
        if isinstance(unit_x, STRINGTYPES):
            unit_x = make_unit(unit_x)
        if not isinstance(unit_x, unum.Unum):
            raise TypeError("'unit_x' must be a 'Unit' object")
        if isinstance(unit_y, STRINGTYPES):
            unit_y = make_unit(unit_y)
        if not isinstance(unit_y, unum.Unum):
            raise TypeError("'unit_y' must be a 'Unit' object")
        if not len(x) == len(y):
            raise ValueError("'x' and 'y' must have the same length")
        order = np.argsort(x)
        self.x = x[order]
        self.y = y[order]
        self.mask = mask[order]
        self.name = name
        self.unit_x = unit_x.copy()
        self.unit_y = unit_y.copy()

    def __neg__(self):
        return Profile(self.x, -self.y, mask=self.mask, unit_x=self.unit_x,
                       unit_y=self.unit_y, name=self.name)

    def __add__(self, otherone):
        if isinstance(otherone, NUMBERTYPES):
            y = self.y + otherone
            name = self.name
            mask = self.mask
        elif isinstance(otherone, unum.Unum):
            y = self.y + otherone/self.unit_y
            name = self.name
            mask = self.mask
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
            mask = np.logical_or(self.mask, otherone.mask)
        else:
            raise TypeError("You only can substract Profile with "
                            "Profile or number")
        return Profile(self.x, y, mask=mask, unit_x=self.unit_x,
                       unit_y=self.unit_y, name=name)

    __radd__ = __add__

    def __sub__(self, otherone):
        return self.__add__(-otherone)

    def __rsub__(self, otherone):
        return -self.__add__(-otherone)

    def __mul__(self, otherone):
        if isinstance(otherone, NUMBERTYPES):
            y = self.y*otherone
            new_unit_y = self.unit_y
        elif isinstance(otherone, unum.Unum):
            new_unit_y = self.unit_y*otherone
            y = self.y*new_unit_y.asNumber()
            new_unit_y = new_unit_y/new_unit_y.asNumber()
        else:
            raise TypeError("You only can multiply Profile with number")
        return Profile(x=self.x, y=y, unit_x=self.unit_x, unit_y=new_unit_y,
                       name=self.name)

    __rmul__ = __mul__

    def __truediv__(self, otherone):
        if isinstance(otherone, NUMBERTYPES):
            y = self.y/otherone
            mask = self.mask
            name = self.name
            unit_y = self.unit_y
        elif isinstance(otherone, unum.Unum):
            tmpunit = self.unit_y/otherone
            y = self.y*(tmpunit.asNumber())
            mask = self.mask
            name = self.name
            unit_y = tmpunit/tmpunit.asNumber()
        elif isinstance(otherone, Profile):
            if not np.all(self.x == otherone.x):
                raise ValueError("Profile has to have identical x axis in "
                                 "order to divide them")
            else:
                mask = np.logical_or(self.mask, otherone.mask)
                tmp_unit = self.unit_y/otherone.unit_y
                y_tmp = self.y.copy()
                y_tmp[otherone.y == 0] = np.nan
                otherone.y[otherone.y == 0] = 1
                y = y_tmp/otherone.y*tmp_unit.asNumber()
                name = ""
                unit_y = tmp_unit/tmp_unit.asNumber()
        else:
            raise TypeError("You only can divide Profile with number")
        return Profile(self.x, y, mask, self.unit_x, unit_y, name=name)

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
        return Profile(x=self.x, y=y, unit_x=self.unit_x, unit_y=self.unit_y,
                       name=self.name)

    def __len__(self):
        return len(self.x)

    ### Attributes ###
    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, values):
        if isinstance(values, ARRAYTYPES):
            self.__x = np.array(values)*1.
        else:
            raise Exception("'x' should be an array, not {}"
                            .format(type(values)))

    @x.deleter
    def x(self):
        raise Exception("Nope, can't delete 'x'")

    @property
    def y(self):
        self.__y[self.__mask] = np.nan
        return self.__y

    @y.setter
    def y(self, values):
        if isinstance(values, np.ma.MaskedArray):
            self.__y = values.data
            self.__mask = values.mask
        elif isinstance(values, ARRAYTYPES):
            self.__y = np.array(values)*1.
            self.__mask = np.isnan(values)
        else:
            raise Exception()

    @y.deleter
    def y(self):
        raise Exception("Nope, can't delete 'y'")

    @property
    def mask(self):
        return self.__mask

    @mask.setter
    def mask(self, mask):
        if isinstance(mask, bool):
            self.__mask = np.empty(self.x.shape, dtype=bool)
            self.__mask.fill(mask)
        elif isinstance(mask, ARRAYTYPES):
            self.__mask = np.array(mask, dtype=bool)
        else:
            raise Exception()
        self.__y[self.__mask] = np.nan

    @mask.deleter
    def mask(self):
        raise Exception("Nope, can't delete 'mask'")

    @property
    def unit_x(self):
        return self.__unit_x

    @unit_x.setter
    def unit_x(self, unit):
        if isinstance(unit, unum.Unum):
            self.__unit_x = unit
        elif isinstance(unit, STRINGTYPES):
            try:
                self.__unit_x = make_unit(unit)
            except (ValueError, TypeError):
                raise Exception()
        else:
            raise Exception()

    @unit_x.deleter
    def unit_x(self):
        raise Exception("Nope, can't delete 'unit_x'")

    @property
    def unit_y(self):
        return self.__unit_y

    @unit_y.setter
    def unit_y(self, unit):
        if isinstance(unit, unum.Unum):
            self.__unit_y = unit
        elif isinstance(unit, STRINGTYPES):
            try:
                self.__unit_y = make_unit(unit)
            except (ValueError, TypeError):
                raise Exception()
        else:
            raise Exception()

    @unit_y.deleter
    def unit_y(self):
        raise Exception("Nope, can't delete 'unit_y'")

    ### Properties ###
    @property
    def max(self):
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
        if np.all(self.mask):
            return None
        return np.max(self.y[np.logical_not(self.mask)])

    @property
    def min(self):
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
        if np.all(self.mask):
            return None
        return np.min(self.y[np.logical_not(self.mask)])

    @property
    def mean(self):
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
        if np.all(self.mask):
            return None
        return np.mean(self.y[np.logical_not(self.mask)])

    ### Watchers ###
    def copy(self):
        """
        Return a copy of the Profile object.
        """
        return copy.deepcopy(self)

    def get_interpolated_value(self, x=None, y=None):
        """
        Get the interpolated (or not) value for a given 'x' or 'y' value.

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
        # getting data
        if x is not None:
            value = x
            values = np.array(self.x)
            values2 = np.ma.masked_array(self.y, self.mask)
        else:
            value = y
            values = np.ma.masked_array(self.y, self.mask)
            values2 = np.array(self.x)
        # if the wanted value is already present
        if np.any(value == values):
            i_values = values2[np.where(value == values)[0]]
        # if we have to do an interpolation
        else:
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
        # returning
        return i_values

    def get_value_position(self, value):
        """
        Return the interpolated position(s) of the wanted value.

        Parameters
        ----------
        value : number
            .
        """
        # check parameters
        if not isinstance(value, NUMBERTYPES):
            raise TypeError()
        # search for positions
        filt = np.logical_not(self.mask)
        y = self.y[filt] - value
        x = self.x[filt]
        sign = y > 0
        chang = np.abs(sign[1::] - sign[0:-1:])
        ind_0 = np.argwhere(chang)
        pos_0 = np.empty((len(ind_0),), dtype=float)
        for i, ind in enumerate(ind_0):
            v1 = np.abs(y[ind])
            v2 = np.abs(y[ind + 1])
            x1 = x[ind]
            x2 = x[ind + 1]
            pos_0[i] = x1 + v1/(v1 + v2)*(x2 - x1)
        # returning
        return pos_0

    def get_integral(self):
        """
        Return the profile integral, and is unit.
        Use the trapezoidal aproximation.
        """
        filt = np.logical_not(self.mask)
        x = self.x[filt]
        y = self.y[filt]
        unit = self.unit_y*self.unit_x
        return np.trapz(y, x)*unit.asNumber(), unit/unit.asNumber()

    def get_gradient(self, position=None, wanted_dx=None):
        """
        Return the profile gradient.
        If 'position' is renseigned, interpolations or finite differences
        are used to get the gradient at x = position.
        Else, a profile with gradient at profile points is returned.
        Warning : only work with evenly spaced x

        Parameters
        ----------

        position : number, optional
            Wanted point position
        wanted_dx : number, optional
            interval on which compute gradient when position is
            renseigned (default is dx similar to axis).
        """
        # check if x values are evenly spaced
        dxs = self.x[1::] - self.x[:-1:]
        dx = self.x[1] - self.x[0]
        if not np.all(np.abs(dxs - dx) < 1e-6*np.max(dxs)):
            raise Exception("Not evenly spaced x values : \n {}".format(dxs))
        if position is None:
            tmp_prof = self.copy()
            tmp_prof.y = np.gradient(self.y, self.x[1] - self.x[0])
            unit = tmp_prof.unit_y/tmp_prof.unit_x
            tmp_prof.y *= unit.asNumber()
            tmp_prof.unit_y = unit/unit.asNumber()
            return tmp_prof
        elif isinstance(position, NUMBERTYPES):
            if wanted_dx is None:
                dx = self.x[1] - self.x[0]
            else:
                dx = wanted_dx
            interp = spinterp.UnivariateSpline(self.x, self.y, k=3, s=0)
            if np.all(position < self.x):
                x = [position, position + dx]
                y = interp(x)
                grad = np.gradient(y, dx)[0]
            elif np.all(position > self.x):
                x = [position - dx, position]
                y = interp(x)
                grad = np.gradient(y, dx)[1]
            else:
                x = [position - dx, position, position + dx]
                y = interp(x)
                grad = np.gradient(y, dx)[1]
            return grad
        else:
            raise TypeError()

    def get_spectrum(self, wanted_x=None, welch_seglen=None,
                     scaling='base', fill='linear', mask_error=True):
        """
        Return a Profile object, with the frequential spectrum of 'component',
        on the point 'pt'.

        Parameters
        ----------
        wanted_x : 2x1 array, optional
            Time interval in which compute spectrum (default is all).
        welch_seglen : integer, optional
            If specified, welch's method is used (dividing signal into
            overlapping segments, and averaging periodogram) with the given
            segments length (in number of points).
        scaling : string, optional
            If 'base' (default), result are in component unit.
            If 'spectrum', the power spectrum is returned (in unit^2).
            If 'density', the power spectral density is returned
            (in unit^2/(1/unit_x))
        fill : string or float
            Specifies the way to treat missing values.
            A value for value filling.
            A string (‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic,
            ‘cubic’ where ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline
            interpolation of first, second or third order) for interpolation.
        mask_error : boolean
            If 'False', instead of raising an error when masked value appear on
            time profile, '(None, None)' is returned.

        Returns
        -------
        magn_prof : Profile object
            Magnitude spectrum.
        """
        tmp_prof = self.copy()
        # fill if asked (and if necessary)
        if isinstance(fill, NUMBERTYPES):
            tmp_prof.fill(kind='value', fill_value=fill, inplace=True)
        elif isinstance(fill, STRINGTYPES):
            tmp_prof.fill(kind=fill, inplace=True)
        else:
            raise Exception()
        values = tmp_prof.y - np.mean(tmp_prof.y)
        time = tmp_prof.x
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
        # sretting unit
        if scaling == 'base':
            unit_y = self.unit_y
        elif scaling == 'spectrum':
            unit_y = self.unit_y**2
        elif scaling == 'density':
            unit_y = self.unit_y**2*self.unit_x
        else:
            raise Exception()
        magn_prof = Profile(frq, magn, unit_x=1./self.unit_x,
                            unit_y=unit_y)
        return magn_prof

    def get_pdf(self, bw_method='scott', resolution=1000, raw=False):
        """
        Return the probability density function.

        Parameters
        ----------
        bw_method : str, scalar or callable, optional
            The method used to calculate the estimator bandwidth. This can be
            ‘scott’, ‘silverman’, a scalar constant or a callable. If a scalar,
            this will be used directly as kde.factor. If a callable, it should
            take a gaussian_kde instance as only parameter and return a scalar.
            If None (default), ‘scott’ is used.
            See 'scipy.stats.kde' for more details.
        resolution : integer, optional
            Resolution of the returned pdf.
        raw : boolean, optional
            If 'True', return an array, else, return a Profile object.
        """
        # check params
        if not isinstance(resolution, int):
            raise TypeError()
        if resolution < 1:
            raise ValueError()
        if not isinstance(raw, bool):
            raise TypeError()
        # remove masked values
        filt = np.logical_not(self.mask)
        y = self.y[filt]
        y_min = np.min(y)
        y_max = np.max(y)
        # get kernel
        import scipy.stats.kde as spkde
        kernel = spkde.gaussian_kde(y)
        # get values
        pdf_x = np.linspace(y_min, y_max, resolution)
        pdf_y = kernel(pdf_x)
        # returning
        if raw:
            return pdf_y
        else:
            prof = Profile(pdf_x, pdf_y, mask=False, unit_x=self.unit_y,
                           unit_y='')
            return prof

    def get_auto_correlation(self, window_len, raw=False):
        """
        Return the auto-correlation profile.

        This algorithm make auto-correlation for all the possible values,
        and an average of the resulting profile.
        Profile are normalized, so the central value of the returned profile
        should be 1.

        Parameters
        ----------
        window_len : integer
            Window length for sweep correlation.
        raw : bool, optional
            If 'True', return an array
            If 'False' (default), return a profile
        """
        # checking parameters
        if not isinstance(window_len, int):
            raise TypeError()
        if window_len >= len(self.y) - 1:
            raise ValueError()
        window_len = np.floor(window_len/2.)
        # loop on possible central values
        corr = np.zeros(2*window_len + 1)
        corr_ad = 0
        nmb = 0
        for i in np.arange(window_len, len(self.y) - window_len - 1):
            central_val = self.y[i]
            surr_vals = self.y[i - window_len:i + window_len + 1]
            tmp_corr = surr_vals*central_val
            corr_ad += central_val**2
            corr += tmp_corr
            nmb += 1
        corr /= corr_ad
        # returning
        if raw:
            return corr
        else:
            dx = self.x[1] - self.x[0]
            x = np.arange(0, dx*len(corr), dx)
            return Profile(x, corr, unit_x=self.unit_x, unit_y=make_unit(''))

    def get_fitting(self, func, p0=None, output_param=False):
        """
        Use non-linear least squares to fit a function, f, to the profile.

        Parameters
        ----------
        func : callable
            The model function, f(x, ...). It must take the independent
            variable as the first argument and the parameters to fit as
            separate remaining arguments.
        p0 : None, scalar, or M-length sequence
            Initial guess for the parameters. If None, then the initial values
            will all be 1 (if the number of parameters for the function can be
            determined using introspection, otherwise a ValueError is raised).
        output_param : boolean, optional
            If 'False' (default), return only a Profile with fitted values
            If 'True', return also the parameters values.

        Returns
        -------
        fit_prof : Profile obect
            The Fitted profile.
        params : tuple, optional
            Fitting parameters.
        """
        # getting data
        xdata = self.x
        ydata = self.y
        # fitting params
        popt, pcov = spopt.curve_fit(func, xdata, ydata, p0)
        # creating profile
        fit_prof = Profile(xdata, func(xdata, *popt), unit_x=self.unit_x,
                           unit_y=self.unit_y, name=self.name)
        # returning
        if output_param:
            return fit_prof, popt
        else:
            return fit_prof

    def get_distribution(self, output_format='normalized', resolution=100,
                         bw_method='scott'):
        """
        Return he distribution of y values by using gaussian kernel estimator.

        Parameters
        ----------
        output_format : string, optional
            'normalized' (default) : give position probability
                                     (integral egal 1).
            'ponderated' : give position probability ponderated by the number
                           or points (integral egal number of points).
            'concentration' : give local concentration (in point per length).
        resolution : integer
            Resolution of the resulting profile (number of values in it).
        bw_method : str or scalar, optional
            The method used to calculate the estimator bandwidth.
            Can be 'scott', 'silverman' or a number to set manually the
            gaussians width.
            (see 'scipy.stats.gaussian_kde' documentation for more details)
        Returns
        -------
        distrib : Profile object
            The y values distribution.
        """
        # checking parameters coherence
        if not isinstance(resolution, int):
            raise TypeError()
        if resolution < 2:
            raise ValueError()

        # getting data
        filt = np.logical_not(self.mask)
        x = self.x[filt]
        y = self.y[filt]

        # getting kernel estimator
        kernel = stats.gaussian_kde(y, bw_method=bw_method)

        # getting distribution
        distrib_x = np.linspace(np.min(y), np.max(y), resolution)
        distrib_y = kernel(distrib_x)

        # normalizing
        if output_format == "normalized":
            unit_y = make_unit('')
        elif output_format == 'ponderated':
            distrib_y *= len(x)
            unit_y = make_unit('')
        elif output_format == "percentage":
            distrib_y *= 100.
            unit_y = make_unit('')
        elif output_format == "concentration":
            unit_y = 1./self.unit_y
            distrib_y *= len(x)
        else:
            raise ValueError()

        # returning
        distrib = Profile(distrib_x, distrib_y, mask=False, unit_x=self.unit_y,
                          unit_y=unit_y)
        return distrib



    ### Modifiers ###
    def add_point(self, x, y):
        """
        Add the given point to the profile.
        """
        pos_ind = np.searchsorted(self.x, x)
        self.x = np.concatenate((self.x[0:pos_ind], [x], self.x[pos_ind::]))
        self.y = np.concatenate((self.y[0:pos_ind], [y], self.y[pos_ind::]))

    def crop_masked_border(self):
        """
        Remove the masked values at the border of the profile in place.
        """
        mask = self.mask
        inds_not_masked = np.where(mask is False)[0]
        first = inds_not_masked[0]
        last = inds_not_masked[-1] + 1
        self.trim([first, last], ind=True, inplace=True)
        return None

    def trim(self, interval, ind=False, inplace=False):
        """
        Return a trimed copy of the profile along x with respect to 'interval'.

        Parameters
        ----------
        interval : array of two numbers
            Bound values of x.
        ind : Boolean, optionnal
            If 'False' (Default), 'interval' are values along x axis,
            if 'True', 'interval' are indices of values along x.
        inplace : boolean, optional
            .
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
            mask_new = self.mask[indices[0]:indices[1]]
        # given position is an indice
        else:
            if any(interval < 0) or any(interval > len(self.x)):
                raise ValueError("'interval' indices are out of profile")
            x_new = self.x[interval[0]:interval[1]]
            y_new = self.y[interval[0]:interval[1]]
            mask_new = self.mask[interval[0]:interval[1]]
        if inplace:
            self.x = x_new
            self.y = y_new
            self.mask = mask_new
            return None
        else:
            tmp_prof = Profile(x_new, y_new, mask_new, self.unit_x,
                               self.unit_y)
            return tmp_prof

    def fill(self, kind='slinear', fill_value=0., inplace=False, crop=False):
        """
        Return a filled profile (no more masked values).

        Warning : If 'crop' is False, border masked values can't be
        interpolated and are filled with 'fill_value' or the nearest value.

        Parameters
        ----------
        kind : string or int, optional
            Specifies the kind of interpolation as a string ('value', ‘linear’,
            ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic, ‘cubic’ where ‘slinear’,
            ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first,
            second or third order) or as an integer specifying the order of
            the spline interpolator to use. Default is ‘linear’.
        fill_value : number, optional
            For kind = 'value', filling value.
        inplace : boolean, optional
            .
        crop : boolean, optional
            .

        Returns
        -------
        prof : Profile object
            Filled profile
        """
        # check if filling really necessary
        if not np.any(self.mask) and inplace:
            return None,
        elif not np.any(self.mask):
            return self.copy()
        # crop if asked
        if crop:
            self.crop_masked_border()
        # get mask
        mask = self.mask
        filt = np.logical_not(mask)
        if np.all(mask):
            raise Exception("There is no values on this profile")
        # check fill type
        if kind == 'value':
            new_y = copy.copy(self.y)
            new_y[filt] = fill_value
        else:
            # making interpolation on existent values
            x = self.x[filt]
            y = self.y[filt]
            interp = spinterp.interp1d(x, y, kind=kind,
                                       bounds_error=False,
                                       fill_value=fill_value)
            # replacing missing values
            new_y = copy.copy(self.y)
            missing_x = self.x[mask]
            new_y[mask] = interp(missing_x)
            # replacing border value by nearest value
            inds_masked = np.where(np.logical_not(mask))[0]
            first = inds_masked[0]
            last = inds_masked[-1]
            new_y[0:first] = new_y[first]
            new_y[last + 1::] = new_y[last]
        # returning
        if inplace:
            self.y = new_y
            self.mask = False
        else:
            tmp_prof = Profile(self.x, new_y, mask=False, unit_x=self.unit_x,
                               unit_y=self.unit_y, name=self.name)
            return tmp_prof

    def change_unit(self, axe, new_unit):
        """
        Change the unit of an axe.

        Parameters
        ----------
        axe : string
            'y' for changing the profile values unit
            'x' for changing the profile axe unit
        new_unit : Unum.unit object or string
            The new unit.
        """
        if isinstance(new_unit, STRINGTYPES):
            new_unit = make_unit(new_unit)
        if not isinstance(new_unit, unum.Unum):
            raise TypeError()
        if not isinstance(axe, STRINGTYPES):
            raise TypeError()
        if axe == 'x':
            old_unit = self.unit_x
            new_unit = old_unit.asUnit(new_unit)
            fact = new_unit.asNumber()
            self.x *= fact
            self.unit_x = new_unit/fact
        elif axe == 'y':
            old_unit = self.unit_y
            new_unit = old_unit.asUnit(new_unit)
            fact = new_unit.asNumber()
            self.y *= fact
            self.unit_y = new_unit/fact
        else:
            raise ValueError()

    def evenly_space(self, kind_interpolation='linear'):
        """
        Return a profile with evenly spaced x values.
        Use interpolation to get missing values.

        Parameters
        ----------
        kind_interpolation : string or int, optional
            Specifies the kind of interpolation as a string ('value', ‘linear’,
            ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic, ‘cubic’ where ‘slinear’,
            ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first,
            second or third order) or as an integer specifying the order of
            the spline interpolator to use. Default is ‘linear’.
        """
        # checking if evenly spaced
        dxs = self.x[1::] - self.x[:-1:]
        dx = self.x[1] - self.x[0]
        if np.all(np.abs(dxs - dx) < 1e-6*np.max(dxs)):
            return self.copy()
        # getting data
        mask = self.mask
        filt = np.logical_not(mask)
        x = self.x[filt]
        y = self.y[filt]
        mean_dx = np.average(dxs)
        min_x = np.min(self.x)
        max_x = np.max(self.x)
        nmb_interv = np.int((max_x - min_x)/mean_dx)
        # creating new evenly spaced x axis
        new_x = np.linspace(min_x, max_x, nmb_interv + 1)
        # interpolate to obtain y values
        interp = spinterp.interp1d(x, y, kind_interpolation)
        new_y = interp(new_x[1:-1])
        new_y = np.concatenate(([self.y[0]], new_y, [self.y[-1]]))
        # return profile
        return Profile(new_x, new_y, mask=False, unit_x=self.unit_x,
                       unit_y=self.unit_y, name=self.name)

    def smooth(self, tos='uniform', size=None, direction='y',
               inplace=False, **kw):
        """
        Return a smoothed profile.
        Warning : fill up the field

        Parameters :
        ------------
        tos : string, optional
            Type of smoothing, can be 'uniform' (default) or 'gaussian'
            (See ndimage module documentation for more details)
        size : number, optional
            Size of the smoothing (is radius for 'uniform' and
            sigma for 'gaussian').
            Default is 3 for 'uniform' and 1 for 'gaussian'.
        dir : string, optional
            In which direction smoothing (can be 'x', 'y' or 'xy').
        inplace : boolean
            If 'False', return a smoothed profile
            else, smooth in place.
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
        if not direction in ['x', 'y', 'xy']:
            raise ValueError()
        # default smoothing border mode to 'nearest'
        if not 'mode' in kw.keys():
            kw.update({'mode': 'nearest'})
        # getting data
        if inplace:
            self.fill(inplace=True)
            y = self.y
            x = self.x
        else:
            tmp_prof = self.copy()
            tmp_prof.fill(inplace=True)
            y = tmp_prof.y
            x = tmp_prof.x
        # smoothing
        if tos == "uniform":
            if direction == 'y':
                y = ndimage.uniform_filter(y, size, **kw)
            if direction == 'x':
                x = ndimage.uniform_filter(x, size, **kw)
            if direction == 'xy':
                x = ndimage.uniform_filter(x, size, **kw)
                y = ndimage.uniform_filter(y, size, **kw)
        elif tos == "gaussian":
            if direction == 'y':
                y = ndimage.gaussian_filter(y, size, **kw)
            if direction == 'x':
                x = ndimage.gaussian_filter(x, size, **kw)
            if direction == 'xy':
                x = ndimage.gaussian_filter(x, size, **kw)
                y = ndimage.gaussian_filter(y, size, **kw)
        else:
            raise ValueError("'tos' must be 'uniform' or 'gaussian'")
        # storing
        if inplace:
            self.x = x
            self.y = y
        else:
            tmp_prof.x = x
            tmp_prof.y = y
            return tmp_prof

    ### Displayers ###
    def _display(self, kind='plot', reverse=False, **plotargs):
        """
        Private Displayer.
        Just display the curve, not axes and title.

        Parameters
        ----------
        kind : string
            Kind of display to plot ('plot', 'semilogx', 'semilogy', 'loglog')
        reverse : Boolean, optionnal
            If 'False', x is put in the abscissa and y in the ordinate. If
            'True', the inverse.
        color : string, number or array of numbers
            Color of the line (can be an array for evolutive color)
        color_label : string
            Label for the colorbar if color is an array
        **plotargs : dict, optionnale
            Additional argument for the 'plot' command.

        Returns
        -------
        fig : Plot reference
            Reference to the displayed plot.
        """
        if not reverse:
            x = self.x
            y = self.y
        else:
            x = self.y
            y = self.x
        # check if color is an array
        if 'color_label' in plotargs.keys():
            color_label = plotargs.pop('color_label')
        else:
            color_label = ''
        if 'color' in plotargs.keys():
            if isinstance(plotargs['color'], ARRAYTYPES):
                if len(plotargs['color']) == len(x):
                    color = plotargs.pop('color')
                    plot = pplt.colored_plot(x, y, z=color, log=kind,
                                             color_label=color_label,
                                             **plotargs)
                    return plot
        # homogeneous color
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
            Kind of display to plot ('plot', 'semilogx', 'semilogy', 'loglog')
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

    def get_indice_on_axe(self, direction, value, kind='bounds'):
        """
        Return, on the given axe, the indices representing the positions
        surrounding 'value'.
        if 'value' is exactly an axe position, return just one indice.

        Parameters
        ----------
        direction : int
            1 or 2, for axes choice.
        value : number
        kind : string
            If 'bounds' (default), return the bounding indices.
            if 'nearest', return the nearest indice
            if 'decimal', return a decimal indice (interpolated)

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
        if not isinstance(kind, STRINGTYPES):
            raise TypeError()
        if not kind in ['bounds', 'nearest', 'decimal']:
            raise ValueError()
        # getting the borning indices
        ind = np.searchsorted(axe, value)
        if axe[ind] == value:
            inds = [ind, ind]
        else:
            inds = [int(ind - 1), int(ind)]
        # returning bounds
        if kind == 'bounds':
            return inds
        # returning nearest
        elif kind == 'nearest':
            if inds[0] == inds[1]:
                return inds[0]
            if np.abs(axe[inds[0]] - value) < np.abs(axe[inds[1]] - value):
                ind = inds[0]
            else:
                ind = inds[1]
            return int(ind)
        # returning decimal
        elif kind == 'decimal':
            if inds[0] == inds[1]:
                return inds[0]
            value_1 = axe[inds[0]]
            value_2 = axe[inds[1]]
            delta = np.abs(value_2 - value_1)
            return (inds[0]*np.abs(value - value_2)/delta
                    + inds[1]*np.abs(value - value_1)/delta)

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
        #checking parameters
        if not isinstance(center, ARRAYTYPES):
            raise TypeError("'center' must be an array")
        center = np.array(center, dtype=float)
        if not center.shape == (2,):
            raise ValueError("'center' must be a 2x1 array")
        if not isinstance(radius, NUMBERTYPES):
            raise TypeError("'radius' must be a number")
        if not radius > 0:
            raise ValueError("'radius' must be positive")
        # getting indice data when 'ind=False'
        if not ind:
            dx = self.axe_x[1] - self.axe_x[0]
            dy = self.axe_y[1] - self.axe_y[0]
            delta = (dx + dy)/2.
            radius = radius/delta
            center_x = self.get_indice_on_axe(1, center[0], kind='decimal')
            center_y = self.get_indice_on_axe(2, center[1], kind='decimal')
            center = np.array([center_x, center_y])
        # pre-computing somme properties
        radius2 = radius**2
        radius_int = radius/np.sqrt(2)
        # isolating possibles indices
        inds_x = np.arange(np.int(np.ceil(center[0] - radius)),
                           np.int(np.floor(center[0] + radius)) + 1)
        inds_y = np.arange(np.int(np.ceil(center[1] - radius)),
                           np.int(np.floor(center[1] + radius)) + 1)
        inds_x, inds_y = np.meshgrid(inds_x, inds_y)
        inds_x = inds_x.flatten()
        inds_y = inds_y.flatten()
        # loop on possibles points
        inds = []
        for i in np.arange(len(inds_x)):
            x = inds_x[i]
            y = inds_y[i]
            # test if the point is in the square 'compris' in the cercle
            if x <= center[0] + radius_int \
                    and x >= center[0] - radius_int \
                    and y <= center[1] + radius_int \
                    and y >= center[1] - radius_int:
                inds.append([x, y])
            # test if the point is the center
            elif all([x, y] == center):
                pass
            # test if the point is in the circle
            elif ((x - center[0])**2 + (y - center[1])**2 <= radius2):
                inds.append([x, y])
        return np.array(inds, subok=True)

    ### Modifiers ###
    def rotate(self, angle, inplace=False):
        """
        Rotate the field.

        Parameters
        ----------
        angle : integer
            Angle in degrees (positive for trigonometric direction).
            In order to preserve the orthogonal grid, only multiples of
            90° are accepted (can be negative multiples).
        inplace : boolean, optional
            If 'True', Field is rotated in place, else, the function return a
            rotated field.

        Returns
        -------
        rotated_field : Field object, optional
            Rotated field.
        """
        # check params
        if not isinstance(angle, NUMBERTYPES):
            raise TypeError()
        if angle%90 != 0:
            raise ValueError()
        if not isinstance(inplace, bool):
            raise TypeError()
        # get dat
        if inplace:
            tmp_field = self
        else:
            tmp_field = self.copy()
        # normalize angle
        angle = angle%360
        # rotate
        if angle == 0:
            pass
        elif angle == 90:
            tmp_field.__axe_x, tmp_field.__axe_y \
                = tmp_field.axe_y[::-1], tmp_field.axe_x
            tmp_field.__unit_x, tmp_field.__unit_y \
                = tmp_field.unit_y, tmp_field.unit_x
        elif angle == 180:
            tmp_field.__axe_x, tmp_field.__axe_y \
                = tmp_field.axe_x[::-1], tmp_field.axe_y[::-1]
        elif angle == 270:
            tmp_field.__axe_x, tmp_field.__axe_y \
                = tmp_field.axe_y, tmp_field.axe_x[::-1]
            tmp_field.__unit_x, tmp_field.__unit_y \
                = tmp_field.unit_y, tmp_field.unit_x
        else:
            raise Exception()
        # correction non-crescent axis
        if tmp_field.axe_x[-1] < tmp_field.axe_x[0]:
            tmp_field.__axe_x = -tmp_field.axe_x
        if tmp_field.axe_y[-1] < tmp_field.axe_y[0]:
            tmp_field.__axe_y = -tmp_field.axe_y
        # returning
        if not inplace:
            return tmp_field

    def change_unit(self, axe, new_unit):
        """
        Change the unit of an Field.

        Parameters
        ----------
        axe : string
            'y' for changing the profile y axis unit
            'x' for changing the profile x axis unit
        new_unit : Unum.unit object or string
            The new unit.
        """
        if isinstance(new_unit, STRINGTYPES):
            new_unit = make_unit(new_unit)
        if not isinstance(new_unit, unum.Unum):
            raise TypeError()
        if not isinstance(axe, STRINGTYPES):
            raise TypeError()
        if axe == 'x':
            old_unit = self.unit_x
            new_unit = old_unit.asUnit(new_unit)
            fact = new_unit.asNumber()
            self.unit_x = new_unit/fact
        elif axe == 'y':
            old_unit = self.unit_y
            new_unit = old_unit.asUnit(new_unit)
            fact = new_unit.asNumber()
            self.unit_y = new_unit/fact
        else:
            raise ValueError()

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
            indmin_x = int(intervalx[0])
            indmax_x = int(intervalx[1])
            indmin_y = int(intervaly[0])
            indmax_y = int(intervaly[1])
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


    def extend(self, nmb_left=0, nmb_right=0, nmb_up=0, nmb_down=0,
               inplace=False):
        """
        Add columns or lines of masked values at the field.

        Parameters
        ----------
        nmb_**** : integers
            Number of lines/columns to add in each direction.
        inplace : bool
            If 'False', return a new extended field, if 'True', modify the
            field inplace.
        Returns
        -------
        Extended_field : Field object, optional
            Extended field.
        """
        dx = self.axe_x[1] - self.axe_x[0]
        dy = self.axe_y[1] - self.axe_y[0]
        new_axe_x = np.arange(self.axe_x[0] - nmb_left*dx,
                              self.axe_x[-1] + nmb_right*dx + 0.1*dx, dx)
        new_axe_y = np.arange(self.axe_y[0] - nmb_down*dy,
                              self.axe_y[-1] + nmb_up*dy + 0.1*dy, dy)
        if inplace:
            self.__axe_x = new_axe_x
            self.__axe_y = new_axe_y
        else:
            fi = self.copy()
            fi.__axe_x = new_axe_x
            fi.__axe_y = new_axe_y
            return fi

    def __clean(self):
        self.__init__()


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
            # test unities system
            try:
                self.unit_values + otherone.unit_values
                self.unit_x + otherone.unit_x
                self.unit_y + otherone.unit_y
            except:
                raise ValueError("I think these units don't match, fox")
            # identical shape and axis
            if np.all(self.axe_x == otherone.axe_x) and \
                    np.all(self.axe_y == otherone.axe_y):
                tmpsf = self.copy()
                fact = otherone.unit_values/self.unit_values
                tmpsf.values += otherone.values*fact.asNumber()
                tmpsf.mask = np.logical_or(self.mask, otherone.mask)
            # different shape, partially same axis
            else:
                # getting shared points
                new_ind_x = np.array([val in otherone.axe_x
                                      for val in self.axe_x])
                new_ind_y = np.array([val in otherone.axe_y
                                      for val in self.axe_y])
                new_ind_xo = np.array([val in self.axe_x
                                       for val in otherone.axe_x])
                new_ind_yo = np.array([val in self.axe_y
                                       for val in otherone.axe_y])
                if not np.any(new_ind_x) or not np.any(new_ind_y):
                    raise ValueError("Incompatible shapes")
                new_ind_Y, new_ind_X = np.meshgrid(new_ind_y, new_ind_x)
                new_ind_value = np.logical_and(new_ind_X, new_ind_Y)
                new_ind_Yo, new_ind_Xo = np.meshgrid(new_ind_yo, new_ind_xo)
                new_ind_valueo = np.logical_and(new_ind_Xo, new_ind_Yo)
                # getting new axis and values
                new_axe_x = self.axe_x[new_ind_x]
                new_axe_y = self.axe_y[new_ind_y]
                fact = otherone.unit_values/self.unit_values
                new_values = (self.values[new_ind_value]
                              + otherone.values[new_ind_valueo]
                              * fact.asNumber())
                new_values = new_values.reshape((len(new_axe_x),
                                                 len(new_axe_y)))
                new_mask = np.logical_or(self.mask[new_ind_value],
                                         otherone.mask[new_ind_valueo])
                new_mask = new_mask.reshape((len(new_axe_x), len(new_axe_y)))
                # creating sf
                tmpsf = ScalarField()
                tmpsf.import_from_arrays(new_axe_x, new_axe_y, new_values,
                                         mask=new_mask, unit_x=self.unit_x,
                                         unit_y=self.unit_y,
                                         unit_values=self.unit_values)
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
        elif isinstance(obj, ARRAYTYPES):
            obj = np.array(obj, subok=True)
            if not obj.shape == self.shape:
                raise ValueError()
            tmpsf = self.copy()
            mask = np.logical_or(self.mask, obj == 0)
            not_mask = np.logical_not(mask)
            tmpsf.values[not_mask] /= obj[not_mask]
            tmpsf.mask = mask
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
            mask = np.logical_or(mask, obj.values == 0.)
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
        elif isinstance(obj, ARRAYTYPES):
            obj = np.array(obj, subok=True)
            if not obj.shape == self.shape:
                raise ValueError()
            tmpsf = self.copy()
            mask = np.logical_or(self.mask, obj == 0)
            not_mask = np.logical_not(mask)
            tmpsf.values[not_mask] = obj[not_mask] / tmpsf.values[not_mask]
            tmpsf.mask = mask
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
            tmpsf.mask = self.mask
            return tmpsf
        elif isinstance(obj, unum.Unum):
            tmpsf = self.copy()
            tmpsf.values *= obj.asNumber()
            tmpsf.unit_values *= obj/obj.asNumber()
            tmpsf.mask = self.mask
            return tmpsf
        elif isinstance(obj, ARRAYTYPES):
            obj = np.array(obj, subok=True)
            if not obj.shape == self.shape:
                raise ValueError()
            tmpsf = self.copy()
            mask = self.mask
            not_mask = np.logical_not(mask)
            tmpsf.values[not_mask] *= obj[not_mask]
            tmpsf.mask = mask
            return tmpsf
        elif isinstance(obj, np.ma.MaskedArray):
            if obj.shape != self.values.shape:
                raise ValueError("Fields are not consistent")
            tmpsf = self.copy()
            tmpsf.values *= obj
            tmpsf.mask = obj.mask
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
        tmpsf.values[np.logical_not(tmpsf.mask)] \
            = np.power(tmpsf.values[np.logical_not(tmpsf.mask)], number)
        tmpsf.mask = self.mask
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
        new_values = np.array(new_values)
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
                pdb.set_trace()
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
    def min(self):
        return np.min(self.values[np.logical_not(self.mask)])

    @property
    def max(self):
        return np.max(self.values[np.logical_not(self.mask)])

    @property
    def mean(self):
        return np.mean(self.values[np.logical_not(self.mask)])

    ### Field maker ###
    def import_from_arrays(self, axe_x, axe_y, values, mask=None,
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
        if mask is not None:
            mask = np.logical_or(mask, np.isnan(self.values))
            self.mask = mask
        self.unit_x = unit_x
        self.unit_y = unit_y
        self.unit_values = unit_values

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
            return self.values[x, y]*unit
        else:
            ind_x = None
            ind_y = None
            # getting indices interval
            inds_x = self.get_indice_on_axe(1, x)
            inds_y = self.get_indice_on_axe(2, y)
            # if something masked
            if np.any(self.mask[inds_x, inds_y]):
                res = np.nan
            # if we are on a grid point
            elif inds_x[0] == inds_x[1] and inds_y[0] == inds_y[1]:
                res = self.values[inds_x[0], inds_y[0]]*unit
            # if we are on a x grid branch
            elif inds_x[0] == inds_x[1]:
                ind_x = inds_x[0]
                pos_y1 = self.axe_y[inds_y[0]]
                pos_y2 = self.axe_y[inds_y[1]]
                value1 = self.values[ind_x, inds_y[0]]
                value2 = self.values[ind_x, inds_y[1]]
                i_value = ((value2*np.abs(pos_y1 - y)
                           + value1*np.abs(pos_y2 - y))
                           / np.abs(pos_y1 - pos_y2))
                res = i_value*unit
            # if we are on a y grid branch
            elif inds_y[0] == inds_y[1]:
                ind_y = inds_y[0]
                pos_x1 = self.axe_x[inds_x[0]]
                pos_x2 = self.axe_x[inds_x[1]]
                value1 = self.values[inds_x[0], ind_y]
                value2 = self.values[inds_x[1], ind_y]
                i_value = ((value2*np.abs(pos_x1 - x)
                            + value1*np.abs(pos_x2 - x))
                           / np.abs(pos_x1 - pos_x2))
                return i_value*unit
            # if we are in the middle of nowhere (linear interpolation)
            else:
                ind_x = inds_x[0]
                ind_y = inds_y[0]
                a, b = np.meshgrid(self.axe_x[ind_x:ind_x + 2],
                                   self.axe_y[ind_y:ind_y + 2], indexing='ij')
                values = self.values[ind_x:ind_x + 2, ind_y:ind_y + 2]
                a = a.flatten()
                b = b.flatten()
                pts = zip(a, b)
                interp_vx = spinterp.LinearNDInterpolator(pts, values.flatten())
                i_value = float(interp_vx(x, y))
                res = i_value*unit
            return res

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

    def get_profile(self, direction, position, ind=False):
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
        position : float, interval of float or string
            Position, interval in which we want a profile or 'all'
        ind : boolean
            If 'True', position has to be given in indices
            If 'False' (default), position has to be given in axis unit.

        Returns
        -------
        profile : Profile object
            Wanted profile
        cutposition : array or number
            Final position or interval in which the profile has been taken
        """
        # checking parameters
        if not isinstance(direction, int):
            raise TypeError("'direction' must be an integer between 1 and 2")
        if not (direction == 1 or direction == 2):
            raise ValueError("'direction' must be an integer between 1 and 2")
        if not isinstance(position, NUMBERTYPES + ARRAYTYPES + STRINGTYPES):
            raise TypeError()
        if isinstance(position, ARRAYTYPES):
            position = np.array(position, dtype=float)
            if not position.shape == (2,):
                raise ValueError("'position' must be a number or an interval")
        elif isinstance(position, STRINGTYPES):
            if position != 'all':
                raise ValueError()
        if not isinstance(ind, bool):
            raise TypeError()
        # geting data
        if direction == 1:
            axe = self.axe_x
            unit_x = self.unit_y
            unit_y = self.unit_values
        else:
            axe = self.axe_y
            unit_x = self.unit_x
            unit_y = self.unit_values
        # applying interval type
        if isinstance(position, ARRAYTYPES) and not ind:
            for pos in position:
                if pos > axe.max() or pos < axe.min():
                    raise ValueError("'position' must be included in"
                                     " the choosen axis values")
        elif isinstance(position, ARRAYTYPES) and ind:
            if np.min(position) < 0 or np.max(position) > len(axe) - 1:
                raise ValueError("'position' must be included in"
                                 " the choosen axis values")
        elif isinstance(position, NUMBERTYPES) and not ind:
            if position > axe.max() or position < axe.min():
                raise ValueError("'position' must be included in the choosen"
                                 " axis values (here [{0},{1}])"
                                 .format(axe.min(), axe.max()))
        elif isinstance(position, NUMBERTYPES) and ind:
            if np.min(position) < 0 or np.max(position) > len(axe) - 1:
                raise ValueError("'position' must be included in the choosen"
                                 " axis values (here [{0},{1}])"
                                 .format(0, len(axe) - 1))
        elif position == 'all':
            position = np.array([axe[0], axe[-1]])
        else:
            raise ValueError()
        # passage from values to indices
        if isinstance(position, NUMBERTYPES) and not ind:
            for i in np.arange(1, len(axe)):
                if (axe[i] >= position and axe[i-1] <= position) \
                        or (axe[i] <= position and axe[i-1] >= position):
                    break
            if np.abs(position - axe[i]) > np.abs(position - axe[i-1]):
                finalindice = i-1
            else:
                finalindice = i
            if direction == 1:
                prof_mask = self.mask[finalindice, :]
                profile = self.values[finalindice, :]
                axe = self.axe_y
                cutposition = self.axe_x[finalindice]
            else:
                prof_mask = self.mask[:, finalindice]
                profile = self.values[:, finalindice]
                axe = self.axe_x
                cutposition = self.axe_y[finalindice]
        elif isinstance(position, NUMBERTYPES) and ind:
            if direction == 1:
                prof_mask = self.mask[position, :]
                profile = self.values[position, :]
                axe = self.axe_y
                cutposition = self.axe_x[position]
            else:
                prof_mask = self.mask[:, position]
                profile = self.values[:, position]
                axe = self.axe_x
                cutposition = self.axe_y[position]
        # Calculation of the profile for an interval of position
        elif isinstance(position, ARRAYTYPES) and not ind:
            axe_mask = np.logical_and(axe >= position[0], axe <= position[1])
            if direction == 1:
                prof_mask = self.mask[axe_mask, :].mean(0)
                profile = self.values[axe_mask, :].mean(0)
                axe = self.axe_y
                cutposition = self.axe_x[axe_mask]
            else:
                prof_mask = self.mask[:, axe_mask].mean(1)
                profile = self.values[:, axe_mask].mean(1)
                axe = self.axe_x
                cutposition = self.axe_y[axe_mask]
        elif isinstance(position, ARRAYTYPES) and ind:
            if direction == 1:
                prof_mask = self.mask[position[0]:position[1] + 1, :].mean(0)
                profile = self.values[position[0]:position[1] + 1, :].mean(0)
                axe = self.axe_y
                cutposition = self.axe_x[position[0]:position[1] + 1].mean()
            else:
                prof_mask = self.mask[:, position[0]:position[1] + 1].mean(1)
                profile = self.values[:, position[0]:position[1] + 1].mean(1)
                axe = self.axe_x
                cutposition = self.axe_y[position[0]:position[1] + 1].mean()
        return (Profile(axe, profile, prof_mask, unit_x, unit_y, "Profile"),
                cutposition)

    def get_spatial_autocorrelation(self, direction, window_len=None):
        """
        Return the spatial auto-correlation along the wanted direction.

        Take the middle point for reference for correlation computation.

        Parameters
        ----------
        direction : string
            'x' or 'y'
        window_len : integer, optional
            Window length for sweep correlation. if 'None' (default), all the
            signal is used, and boundary effect can be seen.

        Returns
        -------
        profile : Profile object
            Spatial correlation
        """
        # Direction X
        if direction == 'x':
            # loop on profiles
            cor = np.zeros(np.floor(window_len/2.)*2 + 1)
            nmb = 0
            for i, y in enumerate(self.axe_y):
                tmp_prof, _ = self.get_profile(2, i, ind=True)
                cor += tmp_prof.get_auto_correlation(window_len, raw=True)
                nmb += 1
            cor /= nmb
            # returning
            dx = self.axe_x[1] - self.axe_x[0]
            x = np.arange(0, len(cor)*dx, dx)
            return Profile(x=x, y=cor, unit_x=self.unit_x,
                           unit_y=make_unit(''))
        elif direction == 'y':
            # loop on profiles
            cor = np.zeros(np.floor(window_len/2.)*2 + 1)
            nmb = 0
            for i, x in enumerate(self.axe_x):
                tmp_prof, _ = self.get_profile(1, i, ind=True)
                cor += tmp_prof.get_auto_correlation(window_len, raw=True)
                nmb += 1
            cor /= nmb
            # returning
            dy = self.axe_y[1] - self.axe_y[0]
            y = np.arange(0, len(cor)*dy, dy)
            return Profile(x=y, y=cor, unit_x=self.unit_y,
                           unit_y=make_unit(''))
        else:
            raise ValueError()

    def get_spatial_spectrum(self, direction, intervx=None, intervy=None,
                             welch_seglen=None, scaling='base', fill='linear'):
        """
        Return a spatial spectrum.

        Parameters
        ----------
        direction : string
            'x' or 'y'.
        intervx and intervy : 2x1 arrays of number, optional
            To chose the zone where to calculate the spectrum.
            If not specified, the biggest possible interval is choosen.
        welch_seglen : integer, optional
            If specified, welch's method is used (dividing signal into
            overlapping segments, and averaging periodogram) with the given
            segments length (in number of points).
        scaling : string, optional
            If 'base' (default), result are in component unit.
            If 'spectrum', the power spectrum is returned (in unit^2).
            If 'density', the power spectral density is returned
            (in unit^2/(1/unit_axe))
        fill : string or float
            Specifies the way to treat missing values.
            A value for value filling.
            A string (‘linear’, ‘nearest’ or 'cubic') for interpolation.

        Returns
        -------
        spec : Profile object
            Magnitude spectrum.

        Notes
        -----
        If there is missing values on the field, 'fill' is used to linearly
        interpolate the missing values (can impact the spectrum).
        """
         # check parameters
        if not isinstance(direction, STRINGTYPES):
            raise TypeError()
        if not direction in ['x', 'y']:
            raise ValueError()
        if intervx is None:
            intervx = [self.axe_x[0], self.axe_x[-1]]
        if not isinstance(intervx, ARRAYTYPES):
            raise TypeError()
        intervx = np.array(intervx)
        if intervx[0] < self.axe_x[0]:
            intervx[0] = self.axe_x[0]
        if intervx[1] > self.axe_x[-1]:
            intervx[1] = self.axe_x[-1]
        if intervx[1] <= intervx[0]:
            raise ValueError()
        if intervy is None:
            intervy = [self.axe_y[0], self.axe_y[-1]]
        if not isinstance(intervy, ARRAYTYPES):
            raise TypeError()
        intervy = np.array(intervy)
        if intervy[0] < self.axe_y[0]:
            intervy[0] = self.axe_y[0]
        if intervy[1] > self.axe_y[-1]:
            intervy[1] = self.axe_y[-1]
        if intervy[1] <= intervy[0]:
            raise ValueError()
        if isinstance(fill, NUMBERTYPES):
            value = fill
            fill = 'value'
        else:
            value = 0.
        # getting data
        tmp_SF = self.trim_area(intervx, intervy)
        tmp_SF.fill(kind=fill, value=value, inplace=True, reduce_tri=True)
        # getting spectrum
        if direction == 'x':
            # first spectrum
            prof, _ = tmp_SF.get_profile(2, tmp_SF.axe_y[0])
            spec = prof.get_spectrum(welch_seglen=welch_seglen,
                                     scaling=scaling, fill=fill)
            # otherones
            for y in tmp_SF.axe_y[1::]:
                prof, _ = tmp_SF.get_profile(2, y)
                spec += prof.get_spectrum(welch_seglen=welch_seglen,
                                          scaling=scaling, fill=fill)
            spec /= len(tmp_SF.axe_y)
        else:
            # first spectrum
            prof, _ = tmp_SF.get_profile(1, tmp_SF.axe_x[0])
            spec = prof.get_spectrum(welch_seglen=welch_seglen,
                                     scaling=scaling, fill=fill)
            # otherones
            for x in tmp_SF.axe_x[1::]:
                prof, _ = tmp_SF.get_profile(1, x)
                spec += prof.get_spectrum(welch_seglen=welch_seglen,
                                          scaling=scaling, fill=fill)
            spec /= len(tmp_SF.axe_x)
        return spec

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
            intervalx = [-np.inf, np.inf]
        if intervaly is None:
            intervaly = [-np.inf, np.inf]
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
    def rotate(self, angle, inplace=False):
        """
        Rotate the scalar field.

        Parameters
        ----------
        angle : integer
            Angle in degrees (positive for trigonometric direction).
            In order to preserve the orthogonal grid, only multiples of
            90° are accepted (can be negative multiples).
        inplace : boolean, optional
            If 'True', scalar field is rotated in place, else, the function
            return a rotated field.

        Returns
        -------
        rotated_field : ScalarField object, optional
            Rotated scalar field.
        """
        # check params
        if not isinstance(angle, NUMBERTYPES):
            raise TypeError()
        if angle%90 != 0:
            raise ValueError()
        if not isinstance(inplace, bool):
            raise TypeError()
        # get data
        if inplace:
            tmp_field = self
        else:
            tmp_field = self.copy()
        # normalize angle
        angle = angle%360
        # rotate the parent
        Field.rotate(tmp_field, angle, inplace=True)
        # rotate
        nmb_rot90 = int(angle/90)
        tmp_field.__values = np.rot90(tmp_field.values, nmb_rot90)
        tmp_field.__mask = np.rot90(tmp_field.mask, nmb_rot90)
        # returning
        if not inplace:
            return tmp_field

    def change_unit(self, axe, new_unit):
        """
        Change the unit of an axe.

        Parameters
        ----------
        axe : string
            'y' for changing the profile y axis unit
            'x' for changing the profile x axis unit
            'values' or changing values unit
        new_unit : Unum.unit object or string
            The new unit.
        """
        if isinstance(new_unit, STRINGTYPES):
            new_unit = make_unit(new_unit)
        if not isinstance(new_unit, unum.Unum):
            raise TypeError()
        if not isinstance(axe, STRINGTYPES):
            raise TypeError()
        if axe == 'x':
            old_unit = self.unit_x
            Field.change_unit(self, axe, new_unit)
            self.axe_x /= self.unit_x/old_unit
        elif axe == 'y':
            old_unit = self.unit_y
            Field.change_unit(self, axe, new_unit)
            self.axe_y /= self.unit_y/old_unit
        elif axe =='values':
            old_unit = self.unit_values
            new_unit = old_unit.asUnit(new_unit)
            fact = new_unit.asNumber()
            self.values *= fact
            self.unit_values = new_unit/fact
        else:
            raise ValueError()

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

    def extend(self, nmb_left=0, nmb_right=0, nmb_up=0, nmb_down=0,
               inplace=False, ind=True):
        """
        Add columns or lines of masked values at the scalarfield.

        Parameters
        ----------
        nmb_**** : integers
            Number of lines/columns to add in each direction.
        inplace : bool
            If 'False', return a new extended field, if 'True', modify the
            field inplace.
        Returns
        -------
        Extended_field : Field object, optional
            Extended field.
        """
        if not ind:
            dx = self.axe_x[1] - self.axe_x[0]
            dy = self.axe_y[1] - self.axe_y[0]
            nmb_left = np.ceil(nmb_left/dx)
            nmb_right = np.ceil(nmb_right/dx)
            nmb_up = np.ceil(nmb_up/dy)
            nmb_down = np.ceil(nmb_down/dy)
            ind = True
        # check params
        if not (isinstance(nmb_left, int) or nmb_left%1 == 0):
            raise TypeError()
        if not (isinstance(nmb_right, int) or nmb_right%1 == 0):
            raise TypeError()
        if not (isinstance(nmb_up, int) or nmb_up%1 == 0):
            raise TypeError()
        if not (isinstance(nmb_down, int) or nmb_down%1 == 0):
            raise TypeError()
        nmb_left = int(nmb_left)
        nmb_right = int(nmb_right)
        nmb_up = int(nmb_up)
        nmb_down = int(nmb_down)
        if np.any(np.array([nmb_left, nmb_right, nmb_up, nmb_down]) < 0):
            raise ValueError()
        # used herited method to extend the field
        if inplace:
            Field.extend(self, nmb_left=nmb_left, nmb_right=nmb_right,
                         nmb_up=nmb_up, nmb_down=nmb_down, inplace=True)
            new_shape = self.shape
        else:
            new_field = Field.extend(self, nmb_left=nmb_left,
                                     nmb_right=nmb_right, nmb_up=nmb_up,
                                     nmb_down=nmb_down, inplace=False)
            new_shape = new_field.shape
        # extend the value ans mask
        new_values = np.zeros(new_shape, dtype=float)
        if nmb_right == 0:
            slice_x = slice(nmb_left, new_values.shape[0] + 2)
        else:
            slice_x = slice(nmb_left, -nmb_right)
        if nmb_up == 0:
            slice_y = slice(nmb_down, new_values.shape[1] + 2)
        else:
            slice_y = slice(nmb_down, -nmb_up)
        new_values[slice_x, slice_y] = self.values
        new_mask = np.ones(new_shape, dtype=bool)
        new_mask[slice_x, slice_y] = self.mask
        # return
        if inplace:
            self.values = new_values
            self.mask = new_mask
        else:
            new_field.values = new_values
            new_field.mask = new_mask
            return new_field

    def mirroring(self, direction, position, inds_to_mirror='all',
                  mir_coef=1, interp=None, value=0, inplace=False):
        """
        Return a field with additional mirrored values.

        Parameters
        ----------
        direction : integer
            Axe on which place the symetry plane (1 for x and 2 for y)
        position : number
            Position of the symetry plane alogn the given axe
        inds_to_mirror : integer
            Number of vector rows to symetrize (default is all)
        mir_coef : number, optional
            Optional coefficient applied only to the mirrored values.
        inplace : boolean, optional
            .
        interp : string, optional
            If specified, method used to fill the gap near the
            symetry plane by interpoaltion.
            ‘value’ : fill with the given value,
            ‘nearest’ : fill with the nearest value,
            ‘linear’ (default): fill using linear interpolation
            (Delaunay triangulation),
            ‘cubic’ : fill using cubic interpolation (Delaunay triangulation)
        value : array, optional
            Value at the symetry plane, in case of interpolation
        """
        # check params
        if not isinstance(direction, int):
            raise TypeError()
        if not isinstance(position, NUMBERTYPES):
            raise TypeError()
        position = float(position)
        # get data
        axe_x = self.axe_x
        axe_y = self.axe_y
        if inplace:
            tmp_vf = self
        else:
            tmp_vf = self.copy()
        # get side to mirror
        if direction == 1:
            axe = axe_x
            x_median = (axe_x[-1] + axe_x[0])/2.
            delta = axe_x[1] - axe_x[0]
            if position < axe_x[0]:
                border = axe_x[0]
                side = 'left'
            elif position > axe_x[-1]:
                border = axe_x[-1]
                side = 'right'
            elif position < x_median:
                tmp_vf.trim_area(intervalx=[position, axe_x[-1]],
                                 inplace=True)
                side = 'left'
                axe_x = tmp_vf.axe_x
                border = axe_x[0]
            elif position > x_median:
                tmp_vf.trim_area(intervalx=[axe_x[0], position],
                                 inplace=True)
                side = 'right'
                axe_x = tmp_vf.axe_x
                border = axe_x[-1]
            else:
                raise ValueError()
        elif direction == 2:
            axe = axe_y
            y_median = (axe_y[-1] + axe_y[0])/2.
            delta = axe_y[1] - axe_y[0]
            if position < axe_y[0]:
                border = axe_y[0]
                side = 'down'
            elif position > axe_y[-1]:
                border = axe_y[-1]
                side = 'up'
            elif position < y_median:
                tmp_vf.trim_area(intervaly=[position, axe_y[-1]],
                                 inplace=True)
                side = 'down'
                axe_y = tmp_vf.axe_y
                border = axe_y[0]
            elif position > y_median:
                tmp_vf.trim_area(intervaly=[axe_x[0], position],
                                 inplace=True)
                side = 'up'
                axe_y = tmp_vf.axe_y
                border = axe_y[-1]
            else:
                raise ValueError()
        else:
            raise ValueError()
        # get length to mirror
        if inds_to_mirror == 'all' or inds_to_mirror > len(axe):
            inds_to_mirror = len(axe) - 1
        if side in ['left', 'down']:
            delta_gap = -(position - border)/delta
        else:
            delta_gap = (position - border)/delta
        inds_to_add = np.floor(inds_to_mirror + 2*delta_gap)
        # extend the field
        tmp_dic = {'nmb_{}'.format(side): inds_to_add}
        tmp_vf.extend(inplace=True, **tmp_dic)
        new_axe_x = tmp_vf.axe_x
        new_axe_y = tmp_vf.axe_y
        # filling mirrored with interpolated values
        for i, x in enumerate(new_axe_x):
            for j, y in enumerate(new_axe_y):
                if direction == 1:
                    mir_pos = [position - (x - position), y]
                else:
                    mir_pos = [x, position - (y - position)]
                if mir_pos[0] < new_axe_x[0] or mir_pos[0] > new_axe_x[-1] \
                        or mir_pos[1] < new_axe_y[0] \
                        or mir_pos[1] > new_axe_y[-1]:
                    continue
                mir_val = tmp_vf.get_value(*mir_pos, ind=False)
                if not np.isnan(mir_val):
                    tmp_vf.values[i, j] = mir_val*mir_coef
                    tmp_vf.mask[i, j] = False
        # interpolating between mirror images id necessary
        if interp is not None:
            # getting data
            x, y = tmp_vf.axe_x, tmp_vf.axe_y
            values = tmp_vf.values
            mask = tmp_vf.mask
            # geting filters from mask
            if interp in ['nearest', 'linear', 'cubic']:
                X, Y = np.meshgrid(x, y, indexing='ij')
                xy = [X.flat[:], Y.flat[:]]
                xy = np.transpose(xy)
                filt = np.logical_not(mask)
                xy_masked = xy[mask.flatten()]
            # getting the zone to interpolate
                xy_good = xy[filt.flatten()]
                values_good = values[filt]
            # adding the value at the symetry plane
            if direction == 1:
                addit_xy = zip([position]*len(tmp_vf.axe_y), tmp_vf.axe_y)
                addit_values = [value]*len(tmp_vf.axe_y)
            else:
                addit_xy = zip(tmp_vf.axe_x, [position]*len(tmp_vf.axe_x))
                addit_values = [value]*len(tmp_vf.axe_x)
            xy_good = np.concatenate((xy_good, addit_xy))
            values_good = np.concatenate((values_good, addit_values))
            # if interpolation
            if interp == 'value':
                values[mask] = value
            elif interp == 'nearest':
                nearest = spinterp.NearestNDInterpolator(xy_good, values_good)
                values[mask] = nearest(xy_masked)
            elif interp == 'linear':
                linear = spinterp.LinearNDInterpolator(xy_good, values_good)
                values[mask] = linear(xy_masked)
                new_mask = np.isnan(values)
                if np.any(new_mask):
                    nearest = spinterp.NearestNDInterpolator(xy_good, values_good)
                    values[new_mask] = nearest(xy[new_mask.flatten()])
            elif interp == 'cubic':
                cubic = spinterp.CloughTocher2DInterpolator(xy_good, values_good)
                values[mask] = cubic(xy_masked)
                new_mask = np.isnan(values)
                if np.any(new_mask):
                    nearest = spinterp.NearestNDInterpolator(xy_good, values_good)
                    values[new_mask] = nearest(xy[new_mask.flatten()])
            else:
                raise ValueError("unknown 'tof' value")
            tmp_vf.values = values
            tmp_vf.mask = False
        # returning
        if not inplace:
            return tmp_vf

    def crop_masked_border(self, hard=False):
        """
        Crop the masked border of the field in place.

        Parameters
        ----------
        hard : boolean, optional
            If 'True', partially masked border are cropped as well.
        """
        # checking masked values presence
        mask = self.mask
        if not np.any(mask):
            return None
        # hard cropping
        if hard:
            # remove trivial borders
            self.crop_masked_border()
            # until there is no more masked values
            while np.any(self.mask):
                # getting number of masked value on each border
                bd1 = np.sum(self.mask[0, :])
                bd2 = np.sum(self.mask[-1, :])
                bd3 = np.sum(self.mask[:, 0])
                bd4 = np.sum(self.mask[:, -1])
                # getting more masked border
                more_masked = np.argmax([bd1, bd2, bd3, bd4])
                # deleting more masked border
                if more_masked == 0:
                    len_x = len(self.axe_x)
                    self.trim_area(intervalx=[1, len_x], ind=True,
                                   inplace=True)
                elif more_masked == 1:
                    len_x = len(self.axe_x)
                    self.trim_area(intervalx=[0, len_x - 2], ind=True,
                                   inplace=True)
                elif more_masked == 2:
                    len_y = len(self.axe_y)
                    self.trim_area(intervaly=[1, len_y], ind=True,
                                   inplace=True)
                elif more_masked == 3:
                    len_y = len(self.axe_y)
                    self.trim_area(intervaly=[0, len_y - 2], ind=True,
                                   inplace=True)
        # soft cropping
        else:
            axe_x_m = np.logical_not(np.all(mask, axis=1))
            axe_y_m = np.logical_not(np.all(mask, axis=0))
            axe_x_min = np.where(axe_x_m)[0][0]
            axe_x_max = np.where(axe_x_m)[0][-1]
            axe_y_min = np.where(axe_y_m)[0][0]
            axe_y_max = np.where(axe_y_m)[0][-1]
            self.trim_area([axe_x_min, axe_x_max],
                           [axe_y_min, axe_y_max],
                           ind=True, inplace=True)

    def fill(self, kind='linear', value=0., inplace=False, reduce_tri=True,
             crop=False):
        """
        Fill the masked part of the array.

        Parameters
        ----------
        kind : string, optional
            Type of algorithm used to fill.
            'value' : fill with the given value
            'nearest' : fill with the nearest value
            'linear' (default): fill using linear interpolation
            (Delaunay triangulation)
            'cubic' : fill using cubic interpolation (Delaunay triangulation)
        value : number
            Value used to fill (for kind='value').
        inplace : boolean, optional
            If 'True', fill the ScalarField in place.
            If 'False' (default), return a filled version of the field.
        reduce_tri : boolean, optional
            If 'True', treatment is used to reduce the triangulation effort
            (faster when a lot of masked values)
            If 'False', no treatment
            (faster when few masked values)
        crop : boolean, optional
            If 'True', SF borders are cropped before filling.
                """
        # check parameters coherence
        if not isinstance(kind, STRINGTYPES):
            raise TypeError("'kind' must be a string")
        if not isinstance(value, NUMBERTYPES):
            raise TypeError("'value' must be a number")
        if crop:
            self.crop_masked_border()
        # getting data
        x, y = self.axe_x, self.axe_y
        values = self.values
        mask = self.mask
        if kind in ['nearest', 'linear', 'cubic']:
            X, Y = np.meshgrid(x, y, indexing='ij')
            xy = [X.flat[:], Y.flat[:]]
            xy = np.transpose(xy)
            filt = np.logical_not(mask)
            xy_masked = xy[mask.flatten()]
        # getting the zone to interpolate
        if reduce_tri and kind in ['nearest', 'linear', 'cubic']:
            import scipy.ndimage as spim
            dilated = spim.binary_dilation(self.mask,
                                           np.arange(9).reshape((3, 3)))
            filt_good = np.logical_and(filt, dilated)
            xy_good = xy[filt_good.flatten()]
            values_good = values[filt_good]
        elif not reduce_tri and kind in ['nearest', 'linear', 'cubic']:
            xy_good = xy[filt.flatten()]
            values_good = values[filt]
        else:
            pass
        # if there is nothing to do...
        if not np.any(mask):
            pass
        # if interpolation
        elif kind == 'value':
            values[mask] = value
        elif kind == 'nearest':
            nearest = spinterp.NearestNDInterpolator(xy_good, values_good)
            values[mask] = nearest(xy_masked)
        elif kind == 'linear':
            linear = spinterp.LinearNDInterpolator(xy_good, values_good)
            values[mask] = linear(xy_masked)
            new_mask = np.isnan(values)
            if np.any(new_mask):
                nearest = spinterp.NearestNDInterpolator(xy_good, values_good)
                values[new_mask] = nearest(xy[new_mask.flatten()])
        elif kind == 'cubic':
            cubic = spinterp.CloughTocher2DInterpolator(xy_good, values_good)
            values[mask] = cubic(xy_masked)
            new_mask = np.isnan(values)
            if np.any(new_mask):
                nearest = spinterp.NearestNDInterpolator(xy_good, values_good)
                values[new_mask] = nearest(xy[new_mask.flatten()])
        else:
            raise ValueError("unknown 'tof' value")
        # returning
        if inplace:
            self.values = values
            self.mask = False
        else:
            sf = ScalarField()
            sf.import_from_arrays(x, y, values, mask=None, unit_x=self.unit_x,
                                  unit_y=self.unit_y,
                                  unit_values=self.unit_values)
            return sf

    def smooth(self, tos='uniform', size=None, inplace=False, **kw):
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
            sigma for 'gaussian') in indice number.
            Default is 3 for 'uniform' and 1 for 'gaussian'.
        inplace : boolean, optional
            If True, Field is smoothed in place,
            else, the smoothed field is returned.
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
        if inplace:
            self.fill(inplace=True)
            values = self.values
        else:
            tmp_sf = self.fill(inplace=False)
            values = tmp_sf.values
        # smoothing
        if tos == "uniform":
            values = ndimage.uniform_filter(values, size, **kw)
        elif tos == "gaussian":
            values = ndimage.gaussian_filter(values, size, **kw)
        else:
            raise ValueError("'tos' must be 'uniform' or 'gaussian'")
        # storing
        if inplace:
            self.values = values
        else:
            tmp_sf.values = values
            return tmp_sf

    def reduce_spatial_resolution(self, fact, inplace=False):
        """
        Reduce the spatial resolution of the field by a factor 'fact'

        Parameters
        ----------
        fact : int
            Reducing factor.
        inplace : boolean, optional
            .
        """
        if not isinstance(fact, int):
            raise TypeError()
        if fact < 1:
            raise ValueError()
        if fact == 1:
            if inplace:
                pass
            else:
                return self.copy()
        if fact % 2 == 0:
            pair = True
        else:
            pair = False
        # get new axis
        axe_x = self.axe_x
        axe_y = self.axe_y
        if pair:
            new_axe_x = (axe_x[np.arange(fact/2 - 1, len(axe_x) - fact/2,
                                         fact)]
                         + axe_x[np.arange(fact/2, len(axe_x) - fact/2 + 1,
                                           fact)])/2.
            new_axe_y = (axe_y[np.arange(fact/2 - 1, len(axe_y) - fact/2,
                                         fact)]
                         + axe_y[np.arange(fact/2, len(axe_y) - fact/2 + 1,
                                           fact)])/2.
        else:
            new_axe_x = axe_x[np.arange((fact - 1)/2,
                                        len(axe_x) - (fact - 1)/2,
                                        fact)]
            new_axe_y = axe_y[np.arange((fact - 1)/2,
                                        len(axe_y) - (fact - 1)/2,
                                        fact)]
        # get new values
        values = self.values
        mask = self.mask
        if pair:
            inds_x = np.arange(fact/2, len(axe_x) - fact/2 + 1, fact)
            inds_y = np.arange(fact/2, len(axe_y) - fact/2 + 1, fact)
            new_values = np.zeros((len(inds_x), len(inds_y)))
            new_mask = np.zeros((len(inds_x), len(inds_y)))
            for i in np.arange(len(inds_x)):
                interv_x = slice(inds_x[i] - fact/2, inds_x[i] + fact/2)
                for j in np.arange(len(inds_y)):
                    interv_y = slice(inds_y[j] - fact/2, inds_y[j] + fact/2)
                    if np.all(mask[interv_x, interv_y]):
                        new_mask[i, j] = True
                        new_values[i, j] = 0.
                    else:
                        new_values[i, j] = np.mean(values[interv_x, interv_y])

        else:
            inds_x = np.arange((fact - 1)/2, len(axe_x) - (fact - 1)/2, fact)
            inds_y = np.arange((fact - 1)/2, len(axe_y) - (fact - 1)/2, fact)
            new_values = np.zeros((len(inds_x), len(inds_y)))
            new_mask = np.zeros((len(inds_x), len(inds_y)))
            for i in np.arange(len(inds_x)):
                interv_x = slice(inds_x[i] - (fact - 1)/2,
                                 inds_x[i] + (fact - 1)/2 + 1)
                for j in np.arange(len(inds_y)):
                    interv_y = slice(inds_y[j] - (fact - 1)/2,
                                     inds_y[j] + (fact - 1)/2 + 1)
                    if np.all(mask[interv_x, interv_y]):
                        new_mask[i, j] = True
                        new_values[i, j] = 0.
                    else:
                        new_values[i, j] = np.mean(values[interv_x, interv_y])
        # returning
        if inplace:
            self.__init__()
            self.import_from_arrays(new_axe_x, new_axe_y, new_values,
                                    mask=new_mask,
                                    unit_x=self.unit_x, unit_y=self.unit_y,
                                    unit_values=self.unit_values)
        else:
            sf = ScalarField()
            sf.import_from_arrays(new_axe_x, new_axe_y, new_values,
                                  mask=new_mask,
                                  unit_x=self.unit_x, unit_y=self.unit_y,
                                  unit_values=self.unit_values)
            return sf

    def __clean(self):
        self.__init__()

    ### Displayers ###
    def _display(self, component=None, kind=None, **plotargs):
        # getting datas
        axe_x, axe_y = self.axe_x, self.axe_y
        unit_x, unit_y = self.unit_x, self.unit_y
        X, Y = np.meshgrid(self.axe_y, self.axe_x)
        # getting wanted component
        if component is None or component == 'values':
            values = np.transpose(self.values).astype(dtype=float)
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
        #plt.axis('image')
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
        cb = plt.colorbar(displ) #, shrink=1, aspect=5)
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
            # test unities system
            try:
                self.unit_values + other.unit_values
                self.unit_x + other.unit_x
                self.unit_y + other.unit_y
            except:
                raise ValueError("I think these units don't match, fox")
            # identical shape and axis
            if np.all(self.axe_x == other.axe_x) and \
                    np.all(self.axe_y == other.axe_y):
                tmpvf = self.copy()
                fact = (other.unit_values/self.unit_values).asNumber()
                tmpvf.comp_x = self.comp_x + other.comp_x*fact
                tmpvf.comp_y = self.comp_y + other.comp_y*fact
                tmpvf.mask = np.logical_or(self.mask, other.mask)
                return tmpvf
            # different shape, partially same axis
            else:
                # getting shared points
                new_ind_x = np.array([val in other.axe_x
                                      for val in self.axe_x])
                new_ind_y = np.array([val in other.axe_y
                                      for val in self.axe_y])
                new_ind_xo = np.array([val in self.axe_x
                                       for val in other.axe_x])
                new_ind_yo = np.array([val in self.axe_y
                                       for val in other.axe_y])
                if not np.any(new_ind_x) or not np.any(new_ind_y):
                    raise ValueError("Incompatible shapes")
                new_ind_Y, new_ind_X = np.meshgrid(new_ind_y, new_ind_x)
                new_ind_value = np.logical_and(new_ind_X, new_ind_Y)
                new_ind_Yo, new_ind_Xo = np.meshgrid(new_ind_yo, new_ind_xo)
                new_ind_valueo = np.logical_and(new_ind_Xo, new_ind_Yo)
                # getting new axis and values
                new_axe_x = self.axe_x[new_ind_x]
                new_axe_y = self.axe_y[new_ind_y]
                fact = other.unit_values/self.unit_values
                new_comp_x = (self.comp_x[new_ind_value]
                              + other.comp_x[new_ind_valueo]
                              * fact.asNumber())
                new_comp_y = (self.comp_y[new_ind_value]
                              + other.comp_y[new_ind_valueo]
                              * fact.asNumber())
                new_comp_x = new_comp_x.reshape((len(new_axe_x),
                                                 len(new_axe_y)))
                new_comp_y = new_comp_y.reshape((len(new_axe_x),
                                                 len(new_axe_y)))
                new_mask = np.logical_or(self.mask[new_ind_value],
                                         other.mask[new_ind_valueo])
                new_mask = new_mask.reshape((len(new_axe_x), len(new_axe_y)))
                # creating vf
                tmpvf = VectorField()
                tmpvf.import_from_arrays(new_axe_x, new_axe_y, new_comp_x,
                                         new_comp_y,
                                         mask=new_mask, unit_x=self.unit_x,
                                         unit_y=self.unit_y,
                                         unit_values=self.unit_values)
                return tmpvf
        elif isinstance(other, ARRAYTYPES):
            other = np.array(other, subok=True)
            if other.shape != self.shape:
                raise ValueError()
            tmpvf = self.copy()
            tmpvf.comp_x = self.comp_x + other
            tmpvf.comp_y = self.comp_y + other
            tmpvf.mask = self.mask
            return tmpvf
        elif isinstance(other, unum.Unum):
            tmpvf = self.copy()
            fact = (other / self.unit_values).asNumber()
            tmpvf.comp_x = self.comp_x + fact
            tmpvf.comp_y = self.comp_y + fact
            tmpvf.mask = self.mask
            return tmpvf
        elif isinstance(other, NUMBERTYPES):
            tmpvf = self.copy()
            tmpvf.comp_x = self.comp_x + other
            tmpvf.comp_y = self.comp_y + other
            tmpvf.mask = self.mask
            return tmpvf
        else:
            raise TypeError("You can only add a velocity field "
                            "with others velocity fields")

    __radd__ = __add__

    def __sub__(self, other):
        other_tmp = other.__neg__()
        tmpvf = self.__add__(other_tmp)
        return tmpvf

    def __truediv__(self, other):
        if isinstance(other, ARRAYTYPES):
            other = np.array(other, subok=True)
            if other.shape != self.shape:
                raise ValueError()
            tmpvf = self.copy()
            tmpvf.comp_x = self.comp_x / other
            tmpvf.comp_y = self.comp_y / other
            tmpvf.mask = np.logical_or(self.mask, other == 0)
            return tmpvf
        elif isinstance(other, unum.Unum):
            tmpvf = self.copy()
            tmpvf.comp_x /= (other/self.unit_values).asNumber()
            tmpvf.comp_y /= (other/self.unit_values).asNumber()
            tmpvf.mask = self.mask
            return tmpvf
        elif isinstance(other, NUMBERTYPES):
            tmpvf = self.copy()
            tmpvf.comp_x /= other
            tmpvf.comp_y /= other
            tmpvf.mask = self.mask
            return tmpvf
        else:
            raise TypeError("You can only divide a vector field "
                            "by numbers")

    __div__ = __truediv__

    def __rtruediv__(self, other):
        return other * self**(-1)

    __rdiv__ = __rtruediv__

    def __mul__(self, other):
        if isinstance(other, ARRAYTYPES):
            other = np.array(other, subok=True)
            if other.shape != self.shape:
                raise ValueError()
            tmpvf = self.copy()
            tmpvf.comp_x = self.comp_x * other
            tmpvf.comp_y = self.comp_y * other
            tmpvf.mask = self.mask
            return tmpvf
        elif isinstance(other, unum.Unum):
            tmpvf = self.copy()
            tmpvf.comp_x *= (other/self.unit_values).asNumber()
            tmpvf.comp_y *= (other/self.unit_values).asNumber()
            tmpvf.mask = self.mask
            return tmpvf
        elif isinstance(other, NUMBERTYPES):
            tmpvf = self.copy()
            tmpvf.comp_x *= other
            tmpvf.comp_y *= other
            tmpvf.mask = self.mask
            return tmpvf
        elif isinstance(other, ScalarField):
            if other.shape != self.shape:
                raise ValueError()
            tmpvf = self.copy()
            tmpvf.comp_x *= other.values
            tmpvf.comp_y *= other.values
            tmpvf.mask = np.logical_or(other.mask, self.mask)
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

    def __abs__(self):
        tmpvf = self.copy()
        tmpvf.comp_x = np.abs(tmpvf.comp_x)
        tmpvf.comp_y = np.abs(tmpvf.comp_y)
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
        new_comp_x = np.array(new_comp_x)
        if not new_comp_x.shape == self.shape:
            raise ValueError("'comp_x' must be coherent with axis system")
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
        new_comp_y = np.array(new_comp_y)
        if not new_comp_y.shape == self.shape:
            raise ValueError()
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
    def min(self):
        return np.min(self.magnitude)

    @property
    def max(self):
        return np.max(self.magnitude)

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
        not_mask = np.logical_and(not_mask, norm != 0)
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
        self.__clean()
        self.axe_x = axe_x
        self.axe_y = axe_y
        self.comp_x = comp_x
        self.comp_y = comp_y
        self.mask = mask
        self.unit_x = unit_x
        self.unit_y = unit_y
        self.unit_values = unit_values

    ### Watchers ###
    def get_value(self, x, y, ind=False, unit=False):
        """
        Return the vectir field compoenents on the point (x, y).
        If ind is true, x and y are indices,
        else, x and y are value on axes (interpolated if necessary).
        """
        return np.array([self.comp_x_as_sf.get_value(x, y, ind=ind, unit=unit),
                         self.comp_y_as_sf.get_value(x, y, ind=ind, unit=unit)])

    def get_profile(self, component, direction, position, ind=False):
        """
        Return a profile of the vector field component, at the given position
        (or at least at the nearest possible position).
        If position is an interval, the fonction return an average profile
        in this interval.

        Function
        --------
        profile, cutposition = get_profile(component, direction, position, ind)

        Parameters
        ----------
        component : integer
            component to treat.
        direction : integer
            Direction along which we choose a position (1 for x and 2 for y).
        position : float or interval of float
            Position or interval in which we want a profile.
        ind : boolean, optional
            If 'True', position is taken as an indice
            Else (default), position is in the field units.

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
            return self.comp_x_as_sf.get_profile(direction, position, ind)
        elif component == 2:
            return self.comp_y_as_sf.get_profile(direction, position, ind)
        else:
            raise ValueError("'component' must have the value of 1 or 2")

    def copy(self):
        """
        Return a copy of the vectorfield.
        """
        return copy.deepcopy(self)

    ### Modifiers ###
    def rotate(self, angle, inplace=False):
        """
        Rotate the vector field.

        Parameters
        ----------
        angle : integer
            Angle in degrees (positive for trigonometric direction).
            In order to preserve the orthogonal grid, only multiples of
            90° are accepted (can be negative multiples).
        inplace : boolean, optional
            If 'True', vector field is rotated in place, else, the function
            return a rotated field.

        Returns
        -------
        rotated_field : VectorField object, optional
            Rotated vector field.
        """
        # check params
        if not isinstance(angle, NUMBERTYPES):
            raise TypeError()
        if angle%90 != 0:
            raise ValueError()
        if not isinstance(inplace, bool):
            raise TypeError()
        # get data
        if inplace:
            tmp_field = self
        else:
            tmp_field = self.copy()
        # normalize angle
        angle = angle%360
        # rotate the parent
        Field.rotate(tmp_field, angle, inplace=True)
        # rotate
        nmb_rot90 = int(angle/90)
        comp_x = np.rot90(tmp_field.comp_x, nmb_rot90)
        comp_y = np.rot90(tmp_field.comp_y, nmb_rot90)
        mask = np.rot90(tmp_field.mask, nmb_rot90)
        comp_x2 = np.cos(angle/180.*np.pi)*comp_x - np.sin(angle/180.*np.pi)*comp_y
        comp_y2 = np.cos(angle/180.*np.pi)*comp_y + np.sin(angle/180.*np.pi)*comp_x
        tmp_field.__comp_x, tmp_field.__comp_y = comp_x, comp_y
        tmp_field.__comp_x = comp_x2
        tmp_field.__comp_y = comp_y2
        tmp_field.__mask = mask
        # returning
        if not inplace:
            return tmp_field

    def change_unit(self, axe, new_unit):
        """
        Change the unit of an axe.

        Parameters
        ----------
        axe : string
            'y' for changing the profile y axis unit
            'x' for changing the profile x axis unit
            'values' or changing values unit
        new_unit : Unum.unit object or string
            The new unit.
        """
        if isinstance(new_unit, STRINGTYPES):
            new_unit = make_unit(new_unit)
        if not isinstance(new_unit, unum.Unum):
            raise TypeError()
        if not isinstance(axe, STRINGTYPES):
            raise TypeError()
        if axe == 'x':
            old_unit = self.unit_x
            Field.change_unit(self, axe, new_unit)
            self.axe_x /= self.unit_x/old_unit
        elif axe == 'y':
            old_unit = self.unit_y
            Field.change_unit(self, axe, new_unit)
            self.axe_y /= self.unit_y/old_unit
        elif axe =='values':
            old_unit = self.unit_values
            new_unit = old_unit.asUnit(new_unit)
            fact = new_unit.asNumber()
            self.comp_x *= fact
            self.comp_y *= fact
            self.unit_values = new_unit/fact
        else:
            raise ValueError()

    def smooth(self, tos='uniform', size=None, inplace=False, **kw):
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
        inplace : boolean, optional
            .
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
        if inplace:
            self.comp_x = Vx
            self.comp_y = Vy
        else:
            vf = VectorField()
            vf.import_from_arrays(self.axe_x, self.axe_y, Vx, Vy,
                                  unit_x=self.unit_x, unit_y=self.unit_y,
                                  unit_values=self.unit_values)
            return vf

    def fill(self, kind='linear', value=[0., 0.], inplace=False,
             reduce_tri=True, crop=False):
        """
        Fill the masked part of the array.

        Parameters
        ----------
        kind : string, optional
            Type of algorithm used to fill.
            'value' : fill with the given value
            'nearest' : fill with the nearest value
            'linear' (default): fill using linear interpolation
            (Delaunay triangulation)
            'cubic' : fill using cubic interpolation (Delaunay triangulation)
        value : 2x1 array of numbers
            Values used to fill (for kind='value').
        inplace : boolean, optional
            If 'True', fill the ScalarField in place.
            If 'False' (default), return a filled version of the field.
        reduce_tri : boolean, optional
            If 'True', treatment is used to reduce the triangulation effort
            (faster when a lot of masked values)
            If 'False', no treatment (faster when few masked values)
        crop : boolean, optional
            If 'True', TVF borders are cropped before filling.
        """
        # check parameters coherence
        if isinstance(value, NUMBERTYPES):
            value = [value, value]
        if not isinstance(value, ARRAYTYPES):
            raise TypeError()
        value = np.array(value)
        if not value.shape == (2,):
            raise ShapeError()
        if crop:
            self.crop_masked_border()
        # filling components
        comp_x = self.comp_x_as_sf
        comp_y = self.comp_y_as_sf
        new_comp_x = comp_x.fill(kind=kind, value=value[0], inplace=False,
                                 reduce_tri=reduce_tri)
        new_comp_y = comp_y.fill(kind=kind, value=value[1], inplace=False,
                                 reduce_tri=reduce_tri)
        # returning
        if inplace:
            self.comp_x = new_comp_x.values
            self.comp_y = new_comp_y.values
            mask = np.empty(self.shape, dtype=bool)
            mask.fill(False)
            self.__mask = mask
        else:
            vf = VectorField()
            vf.import_from_arrays(self.axe_x, self.axe_y, new_comp_x.values,
                                  new_comp_y.values, mask=False,
                                  unit_x=self.unit_x, unit_y=self.unit_y,
                                  unit_values=self.unit_values)
            return vf

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

    def crop_masked_border(self, hard=False):
        """
        Crop the masked border of the field in place.

        Parameters
        ----------
        hard : boolean, optional
            If 'True', partially masked border are cropped as well.
        """
        # checking masked values presence
        mask = self.mask
        if not np.any(mask):
            return None
        # hard cropping
        if hard:
            # remove trivial borders
            self.crop_masked_border()
            # until there is no more masked values
            while np.any(self.mask):
                # getting number of masked value on each border
                bd1 = np.sum(self.mask[0, :])
                bd2 = np.sum(self.mask[-1, :])
                bd3 = np.sum(self.mask[:, 0])
                bd4 = np.sum(self.mask[:, -1])
                # getting more masked border
                more_masked = np.argmax([bd1, bd2, bd3, bd4])
                # deleting more masked border
                if more_masked == 0:
                    len_x = len(self.axe_x)
                    self.trim_area(intervalx=[1, len_x], ind=True,
                                   inplace=True)
                elif more_masked == 1:
                    len_x = len(self.axe_x)
                    self.trim_area(intervalx=[0, len_x - 2], ind=True,
                                   inplace=True)
                elif more_masked == 2:
                    len_y = len(self.axe_y)
                    self.trim_area(intervaly=[1, len_y], ind=True,
                                   inplace=True)
                elif more_masked == 3:
                    len_y = len(self.axe_y)
                    self.trim_area(intervaly=[0, len_y - 2], ind=True, inplace=True)
        # soft cropping
        else:
            axe_x_m = np.logical_not(np.all(mask, axis=1))
            axe_y_m = np.logical_not(np.all(mask, axis=0))
            axe_x_min = np.where(axe_x_m)[0][0]
            axe_x_max = np.where(axe_x_m)[0][-1]
            axe_y_min = np.where(axe_y_m)[0][0]
            axe_y_max = np.where(axe_y_m)[0][-1]
            self.trim_area([axe_x_min, axe_x_max],
                           [axe_y_min, axe_y_max],
                           ind=True, inplace=True)

    def extend(self, nmb_left=0, nmb_right=0, nmb_up=0, nmb_down=0,
               inplace=False):
        """
        Add columns or lines of masked values at the vectorfield.

        Parameters
        ----------
        nmb_**** : integers
            Number of lines/columns to add in each direction.
        inplace : bool
            If 'False', return a new extended field, if 'True', modify the
            field inplace.
        Returns
        -------
        Extended_field : Field object, optional
            Extended field.
        """
        if inplace:
            Field.extend(self, nmb_left=nmb_left, nmb_right=nmb_right,
                         nmb_up=nmb_up, nmb_down=nmb_down, inplace=True)
            new_shape = self.shape
        else:
            new_field = Field.extend(self, nmb_left=nmb_left,
                                     nmb_right=nmb_right, nmb_up=nmb_up,
                                     nmb_down=nmb_down, inplace=False)
            new_shape = new_field.shape
        new_Vx = np.zeros(new_shape, dtype=float)
        new_Vy = np.zeros(new_shape, dtype=float)
        if nmb_right == 0:
            slice_x = slice(nmb_left, new_Vx.shape[0] + 2)
        else:
            slice_x = slice(nmb_left, -nmb_right)
        if nmb_up == 0:
            slice_y = slice(nmb_down, new_Vx.shape[1] + 2)
        else:
            slice_y = slice(nmb_down, -nmb_up)
        new_Vx[slice_x, slice_y] = self.comp_x
        new_Vy[slice_x, slice_y] = self.comp_y
        new_mask = np.ones(new_shape, dtype=bool)
        new_mask[slice_x, slice_y] = self.mask
        if inplace:
            self.comp_x = new_Vx
            self.comp_y = new_Vy
            self.mask = new_mask
        else:
            new_field.comp_x = new_Vx
            new_field.comp_y = new_Vy
            new_field.mask = new_mask
            return new_field

    def mirroring(self, direction, position, inds_to_mirror='all',
                  inplace=False, interp=None, value=[0, 0]):
        """
        Return a field with additional mirrored values.

        Parameters
        ----------
        direction : integer
            Axe on which place the symetry plane (1 for x and 2 for y)
        position : number
            Position of the symetry plane along the given axe
        inds_to_mirror : integer
            Number of vector rows to symetrize (default is all)
        inplace : boolean, optional
            .
        interp : string, optional
            If specified, method used to fill the gap near the
            symetry plane by interpoaltion.
            'value' : fill with the given value,
            'nearest' : fill with the nearest value,
            'linear' (default): fill using linear interpolation
            (Delaunay triangulation),
            'cubic' : fill using cubic interpolation (Delaunay triangulation)
        value : array, optional
            Value at the symetry plane, in case of interpolation
        """
        # getting components
        vx = self.comp_x_as_sf
        vy = self.comp_y_as_sf
        # treating sign changments
        if direction == 1:
            coefx = -1
            coefy = 1
        else:
            coefx = 1
            coefy = -1
        # mirroring on components
        vx.mirroring(direction, position, inds_to_mirror=inds_to_mirror,
                     interp=interp, value=value[0], inplace=True,
                     mir_coef=coefx)
        vy.mirroring(direction, position, inds_to_mirror=inds_to_mirror,
                     interp=interp, value=value[1], inplace=True,
                     mir_coef=coefy)
        # storing
        if inplace:
            tmp_vf = self
        else:
            tmp_vf = VectorField()
        mask = np.logical_or(vx.mask, vy.mask)
        tmp_vf.import_from_arrays(vx.axe_x, vx.axe_y, vx.values, vy.values,
                                  mask=mask, unit_x=vx.unit_x,
                                  unit_y=vy.unit_y,
                                  unit_values=vx.unit_values)
        # returning
        if not inplace:
            return tmp_vf

    def reduce_spatial_resolution(self, fact, inplace=False):
        """
        Reduce the spatial resolution of the field by a factor 'fact'

        Parameters
        ----------
        fact : int
            Reducing factor.
        inplace : boolean, optional
            .
        """
        # reducing
        Vx = self.comp_x_as_sf
        Vy = self.comp_y_as_sf
        Vx.reduce_spatial_resolution(fact, inplace=True)
        Vy.reduce_spatial_resolution(fact, inplace=True)
        # returning
        if inplace:
            self.__init__()
            self.import_from_arrays(Vx.axe_x, Vx.axe_y, Vx.values,
                                    Vy.values,
                                    mask=Vx.mask, unit_x=self.unit_x,
                                    unit_y=self.unit_y,
                                    unit_values=self.unit_values)
        else:
            vf = VectorField()
            vf.import_from_arrays(Vx.axe_x, Vx.axe_y, Vx.values, Vy.values,
                                  mask=Vx.mask, unit_x=self.unit_x,
                                  unit_y=self.unit_y,
                                  unit_values=self.unit_values)
            return vf

    def __clean(self):
        self.__init__()

    ### Displayers ###
    def _display(self, component=None, kind=None, axis='image', **plotargs):
        if kind is not None:
            if not isinstance(kind, STRINGTYPES):
                raise TypeError("'kind' must be a string")
        axe_x, axe_y = self.axe_x, self.axe_y
        if component is None or component == 'V':
            Vx = np.transpose(self.comp_x).copy()
            Vy = np.transpose(self.comp_y).copy()
            mask = np.transpose(self.mask)

            Vx[mask] = np.inf
            Vy[mask] = np.inf
            magn = np.transpose(self.magnitude)
            magn[mask] = 0
            unit_x, unit_y = self.unit_x, self.unit_y
            if kind == 'stream':
                if not 'color' in plotargs.keys():
                    plotargs['color'] = magn
                displ = plt.streamplot(axe_x, axe_y, Vx, Vy, **plotargs)
            elif kind == 'track':
                from IMTreatment.field_treatment import get_track_field
                track_field = get_track_field(self)
                Vx = np.transpose(track_field.comp_x)
                Vy = np.transpose(track_field.comp_y)
                mask = track_field.mask
                Vx = np.ma.masked_array(Vx, mask)
                Vy = np.ma.masked_array(Vy, mask)
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
            plt.axis(axis)
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
                if 'C' in plotargs.keys():
                    C = plotargs.pop('C')
                    if not (C == 0 or C is None):
                        cb = plt.colorbar()
                        cb.set_label("Magnitude " + unit_values.strUnit())
                legendarrow = round(np.max([Vx.max(), Vy.max()]))
                plt.quiverkey(displ, 1.075, 1.075, legendarrow,
                              "$" + str(legendarrow)
                              + unit_values.strUnit() + "$",
                              labelpos='W', fontproperties={'weight': 'bold'})
            elif kind in ['stream', 'track']:
                if not 'color' in plotargs.keys():
                    cb = plt.colorbar()
                    cb.set_label("Magnitude " + unit_values.strUnit())
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
    def rotate(self, angle, inplace=False):
        """
        Rotate the fields.

        Parameters
        ----------
        angle : integer
            Angle in degrees (positive for trigonometric direction).
            In order to preserve the orthogonal grid, only multiples of
            90° are accepted (can be negative multiples).
        inplace : boolean, optional
            If 'True', fields is rotated in place, else, the function
            return rotated fields.

        Returns
        -------
        rotated_field : TemporalFields or child object, optional
            Rotated fields.
        """
        # check params
        if not isinstance(angle, NUMBERTYPES):
            raise TypeError()
        if angle%90 != 0:
            raise ValueError()
        if not isinstance(inplace, bool):
            raise TypeError()
        # get data
        if inplace:
            tmp_field = self
        else:
            tmp_field = self.copy()
        # normalize angle
        angle = angle%360
        # rotate the parent
        Field.rotate(tmp_field, angle, inplace=True)
        # rotate fields
        for i in np.arange(len(tmp_field.fields)):
            tmp_field.fields[i].rotate(angle=angle, inplace=True)
        # returning
        if not inplace:
            return tmp_field

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
        if isinstance(other, self.fields[0].__class__):
            tmp_TF = self.copy()
            for i in np.arange(len(tmp_TF.fields)):
                tmp_TF.fields[i] += other
            return tmp_TF
        elif isinstance(other, self.__class__):
            tmp_tf = self.copy()
            if np.all(self.times == other.times):
                for i in np.arange(len(self.fields)):
                    tmp_tf.fields[i] += other.fields[i]
            else:
                for i in np.arange(len(other.fields)):
                    tmp_tf.add_field(other.fields[i])
            return tmp_tf

        else:
            raise TypeError("cannot concatenate {} with"
                            " {}.".format(self.__class__, type(other)))

    def __sub__(self, other):
        return self.__add__(-other)

    def __neg__(self):
        tmp_tf = self.copy()
        for i in np.arange(len(self.fields)):
            tmp_tf.fields[i] = -tmp_tf.fields[i]
        return tmp_tf

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            if not len(self) == len(other):
                raise Exception()
            if not np.all(self.axe_x == other.axe_x) \
                    and np.all(self.axe_y == other.axe_y):
                raise Exception()
            if not np.all(self.times == other.times):
                raise Exception()
            vfs = self.__class__()
            for i in np.arange(len(self.fields)):
                vfs.add_field(self.fields[i]*other.fields[i])
            return vfs
        elif isinstance(other, (NUMBERTYPES, unum.Unum)):
            final_vfs = self.__class__()
            for field in self.fields:
                final_vfs.add_field(field*other)
            return final_vfs
        else:
            raise TypeError("You can only multiply a temporal velocity field "
                            "by numbers")

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            if not len(self) == len(other):
                raise Exception()
            if not np.all(self.axe_x == other.axe_x) \
                    and np.all(self.axe_y == other.axe_y):
                raise Exception()
            if not np.all(self.times == other.times):
                raise Exception()
            vfs = self.__class__()
            for i in np.arange(len(self.fields)):
                vfs.add_field(self.fields[i]/other.fields[i])
            return vfs
        elif isinstance(other, (NUMBERTYPES, unum.Unum)):
            final_vfs = self.__class__()
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
        final_vfs = self.__class__()
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
        if len(self.fields) != len(values):
            raise ValueError()
        self.__times = values

    @times.deleter
    def times(self):
        raise Exception("Nope, can't do that")

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
        if isinstance(self, TemporalScalarFields):
            value = 0.
        else:
            value = [0., 0.]
        result_f.fill(kind='value', value=value, crop=False, inplace=True)
        mask_cum = np.zeros(self.shape, dtype=int)
        mask_cum[np.logical_not(self.fields[0].mask)] += 1
        for field in self.fields[1::]:
            added_field = field.copy()
            added_field.fill(kind='value', value=0., inplace=True)
            result_f += added_field
            mask_cum[np.logical_not(field.mask)] += 1
        mask = mask_cum <= nmb_min
        result_f.mask = mask
        fact = mask_cum
        fact[mask] = 1
        result_f /= fact
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
        for i, field in enumerate(self.fields):
            fluct_fields.add_field(field - mean_field, time=self.times[i],
                                   unit_times=self.unit_times)
        return fluct_fields

    def get_spatial_spectrum(self, component, direction, intervx=None,
                             intervy=None, intervtime=None, welch_seglen=None,
                             scaling='base', fill='linear'):
        """
        Return a spatial spectrum.
        If more than one time are specified, spectrums are averaged.

        Parameters
        ----------
        component : string
            Should be an attribute name of the stored fields.
        direction : string
            Direction in which perform the spectrum ('x' or 'y').
        intervx and intervy : 2x1 arrays of number, optional
            To chose the zone where to calculate the spectrum.
            If not specified, the biggest possible interval is choosen.
        intervtime : 2x1 array, optional
            Interval of time on which averaged the spectrum.
        welch_seglen : integer, optional
            If specified, welch's method is used (dividing signal into
            overlapping segments, and averaging periodogram) with the given
            segments length (in number of points).
        scaling : string, optional
            If 'base' (default), result are in component unit.
            If 'spectrum', the power spectrum is returned (in unit^2).
            If 'density', the power spectral density is returned
            (in unit^2/(1/unit_axe))
        fill : string or float
            Specifies the way to treat missing values.
            A value for value filling.
            A string (‘linear’, ‘nearest’ or 'cubic') for interpolation.

        Notes
        -----
        If there is missing values on the field, 'fill' is used to linearly
        interpolate the missing values (can impact the spectrum).
        """
        # check parameters
        try:
            self[0].__getattribute__('{}_as_sf'.format(component))
        except AttributeError():
            raise ValueError()
        if not isinstance(direction, STRINGTYPES):
            raise TypeError()
        if not direction in ['x', 'y']:
            raise ValueError()
        if intervtime is None:
            intervtime = [self.times[0], self.times[-1]]
        if not isinstance(intervtime, ARRAYTYPES):
            raise TypeError()
        intervtime = np.array(intervtime)
        if not intervtime.shape == (2,):
            raise ValueError()
        if intervtime[0] < self.times[0]:
            intervtime[0] = self.times[0]
        if intervtime[-1] > self.times[-1]:
            intervtime[-1] = self.times[-1]
        if intervtime[0] >= intervtime[1]:
            raise ValueError()
        # loop on times
        spec = 0
        nmb = 0
        for i, time in enumerate(self.times):
            if time < intervtime[0] or time > intervtime[1]:
                continue
            comp = self[i].__getattribute__('{}_as_sf'.format(component))
            if spec == 0:
                spec = comp.get_spatial_spectrum(direction, intervx=intervx,
                                                 intervy=intervy,
                                                 welch_seglen=welch_seglen,
                                                 scaling=scaling, fill=fill)
            else:
                spec += comp.get_spatial_spectrum(direction, intervx=intervx,
                                                  intervy=intervy,
                                                  welch_seglen=welch_seglen,
                                                  scaling=scaling, fill=fill)
            nmb += 1
        # returning
        spec /= nmb
        return spec

    def get_time_profile(self, component, x, y, wanted_times=None, ind=False):
        """
        Return a profile contening the time evolution of the given component.

        Parameters
        ----------
        component : string
            Should be an attribute name of the stored fields.
        x, y : numbers
            Wanted position for the time profile, in axis units.
        wanted_times : 2x1 array of numbers
            Time interval in which getting profile (default is all).
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
        if wanted_times is None:
            wanted_times = [self.times[0], self.times[-1]]
        if (np.any(wanted_times < self.times[0])
                or np.any(wanted_times > self.times[-1])):
            raise ValueError()
        if wanted_times[-1] <= wanted_times[0]:
            raise ValueError()
        if ind:
            if not (isinstance(x, int) and isinstance(y, int)):
                raise TypeError()
            ind_x = x
            ind_y = y
        else:
            ind_x = self.get_indice_on_axe(1, x, kind='nearest')
            ind_y = self.get_indice_on_axe(2, y, kind='nearest')
        axe_x, axe_y = self.axe_x, self.axe_y
        if not (0 <= ind_x < len(axe_x) and 0 <= ind_y < len(axe_y)):
            raise ValueError("'x' ans 'y' values out of bounds")
        # getting wanted time if necessary
        if wanted_times is not None:
            times = self.times
            w_times_ind = np.argwhere(np.logical_and(times <= wanted_times[1],
                                                     times >= wanted_times[0]))
            w_times_ind = w_times_ind.squeeze()
        else:
            w_times_ind = np.arange(len(self.times))
        # getting component values
        dim = len(w_times_ind)
        compo = np.empty(dim, dtype=float)
        masks = np.empty(dim, dtype=float)
        for i, time_ind in enumerate(w_times_ind):
            compo[i] = self.fields[time_ind].__getattribute__(component)[ind_x,
                                                                         ind_y]
            masks[i] = self.fields[time_ind].mask[ind_x, ind_y]
        # gettign others datas
        time = self.times[w_times_ind]
        unit_time = self.unit_times
        unit_values = self.unit_values
        # getting position indices
        return Profile(time, compo, masks, unit_x=unit_time,
                       unit_y=unit_values)

    def get_temporal_spectrum(self, component, pt, ind=False,
                              wanted_times=None, welch_seglen=None,
                              scaling='base', fill='linear', mask_error=True):
        """
        Return a Profile object, with the temporal spectrum of 'component',
        on the point 'pt'.

        Parameters
        ----------
        component : string
            .
        pt : 2x1 array of numbers
            .
        ind : boolean
            If true, 'pt' is read as indices,
            else, 'pt' is read as coordinates.
        wanted_times : 2x1 array, optional
            Time interval in which compute spectrum (default is all).
        welch_seglen : integer, optional
            If specified, welch's method is used (dividing signal into
            overlapping segments, and averaging periodogram) with the given
            segments length (in number of points).
        scaling : string, optional
            If 'base' (default), result are in component unit.
            If 'spectrum', the power spectrum is returned (in unit^2).
            If 'density', the power spectral density is returned (in unit^2/Hz)
        fill : string or float
            Specifies the way to treat missing values.
            A value for value filling.
            A string (‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic,
            ‘cubic’ where ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline
            interpolation of first, second or third order) for interpolation.
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
        # getting time profile
        time_prof = self.get_time_profile(component, x, y, ind=ind,
                                          wanted_times=wanted_times)
        magn_prof = time_prof.get_spectrum(welch_seglen=welch_seglen,
                                           scaling=scaling, fill=fill,
                                           mask_error=mask_error)
        return magn_prof

    def get_temporal_spectrum_over_area(self, component, intervalx, intervaly,
                                        ind=False, welch_seglen=None,
                                        scaling='base', fill='linear'):
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
        welch_seglen : integer, optional
            If specified, welch's method is used (dividing signal into
            overlapping segments, and averaging periodogram) with the given
            segments length (in number of points).
        scaling : string, optional
            If 'base' (default), result are in component unit.
            If 'spectrum', the power spectrum is returned (in unit^2).
            If 'density', the power spectral density is returned (in unit^2/Hz)
        fill : string or float
            Specifies the way to treat missing values.
            A value for value filling.
            A string (‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic,
            ‘cubic’ where ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline
            interpolation of first, second or third order) for interpolation.

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
                raise ValueError("intervals ({}) are out of bounds ({})"
                                 .format([intervalx, intervaly],
                                         [[axe_x_min, axe_x_max],
                                          [axe_y_min, axe_y_max]]))
            ind_x_min = self.get_indice_on_axe(1, intervalx[0])[-1]
            ind_x_max = self.get_indice_on_axe(1, intervalx[1])[0]
            ind_y_min = self.get_indice_on_axe(2, intervaly[0])[-1]
            ind_y_max = self.get_indice_on_axe(2, intervaly[1])[0]
        # Averaging ponctual spectrums
        magn = 0.
        nmb_fields = (ind_x_max - ind_x_min + 1)*(ind_y_max - ind_y_min + 1)
        real_nmb_fields = nmb_fields
        for i in np.arange(ind_x_min, ind_x_max + 1):
            for j in np.arange(ind_y_min, ind_y_max + 1):
                tmp_m = self.get_temporal_spectrum(component, [i, j], ind=True,
                                                   welch_seglen=welch_seglen,
                                                   scaling=scaling,
                                                   fill=fill, mask_error=True)
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
    def change_unit(self, axe, new_unit):
        """
        Change the unit of an axe.

        Parameters
        ----------
        axe : string
            'y' for changing the profile y axis unit
            'x' for changing the profile x axis unit
            'values' for changing values unit
            'time' for changing time unit
        new_unit : Unum.unit object or string
            The new unit.
        """
        if isinstance(new_unit, STRINGTYPES):
            new_unit = make_unit(new_unit)
        if not isinstance(new_unit, unum.Unum):
            raise TypeError()
        if not isinstance(axe, STRINGTYPES):
            raise TypeError()
        if axe in ['x', 'y', 'values']:
            for field in self.fields:
                field.change_unit(axe, new_unit)
        elif axe == 'time':
            old_unit = self.unit_times
            new_unit = old_unit.asUnit(new_unit)
            fact = new_unit.asNumber()
            self.times *= fact
            self.unit_times = new_unit/fact
        else:
            raise ValueError()
        if axe in ['x', 'y']:
            Field.change_unit(self, axe, new_unit)

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
        if isinstance(self, TemporalScalarFields) \
                and not isinstance(field, ScalarField):
            raise TypeError()
        if isinstance(self, TemporalVectorFields) \
                and not isinstance(field, VectorField):
            raise TypeError()
        if not isinstance(time, NUMBERTYPES):
            raise TypeError("'time' should be a number, not {}"
                            .format(type(time)))
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

    def reduce_temporal_resolution(self, nmb_in_interval, mean=True,
                                   inplace=False):
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
        inplace : boolean, optional

        Returns
        -------
        TVFS : TemporalVelocityFields
        """
        # cehck parameters
        if not isinstance(nmb_in_interval, int):
            raise TypeError("'nmb_in_interval' must be an integer")
        if nmb_in_interval == 1:
            return self.copy()
        if nmb_in_interval >= len(self):
            raise ValueError("'nmb_in_interval' is too big")
        if not isinstance(mean, bool):
            raise TypeError("'mean' must be a boolean")
        #
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
        # returning
        if inplace:
            self.fields = tmp_TFS.fields
            self.times = tmp_TFS.times
        else:
            return tmp_TFS

    def crop_masked_border(self, hard=False, inplace=False):
        """
        Crop the masked border of the velocity fields in place.

        Parameters
        ----------
        hard : boolean, optional
            If 'True', partially masked border are cropped as well.
        inplace : boolean, optional
            If 'True', crop the F in place,
            else, return a cropped TF.
        """
        # getting big mask (where all the value are masked)
        masks_temp = self.mask
        mask_temp = np.sum(masks_temp, axis=0)
        mask_temp = mask_temp == len(masks_temp)
        # checking masked values presence
        if not np.any(mask_temp):
            return None
        # hard cropping
        if hard:
            if inplace:
                tmp_tf = self
            else:
                tmp_tf = self.copy()
            # remove trivial borders
            tmp_tf.crop_masked_border(hard=False)
            # until there is no more masked values
            while True:
                # getting mask
                masks = tmp_tf.mask
                mask = np.sum(masks, axis=0)
                mask = mask == len(tmp_tf.fields)
                # leaving if no masked values
                if not np.any(mask):
                    break
                # getting number of masked value on each border
                bd1 = np.sum(mask[0, :])
                bd2 = np.sum(mask[-1, :])
                bd3 = np.sum(mask[:, 0])
                bd4 = np.sum(mask[:, -1])
                # getting more masked border
                more_masked = np.argmax([bd1, bd2, bd3, bd4])
                # deleting more masked border
                if more_masked == 0:
                    len_x = len(tmp_tf.axe_x)
                    tmp_tf.trim_area(intervalx=[1, len_x], ind=True,
                                     inplace=True)
                elif more_masked == 1:
                    len_x = len(tmp_tf.axe_x)
                    tmp_tf.trim_area(intervalx=[0, len_x - 2], ind=True,
                                     inplace=True)
                elif more_masked == 2:
                    len_y = len(tmp_tf.axe_y)
                    tmp_tf.trim_area(intervaly=[1, len_y], ind=True,
                                     inplace=True)
                elif more_masked == 3:
                    len_y = len(tmp_tf.axe_y)
                    tmp_tf.trim_area(intervaly=[0, len_y - 2], ind=True,
                                     inplace=True)
            if not inplace:
                return tmp_tf
        # soft cropping
        else:
            # getting positions to remove
            # (column or line with only masked values)
            axe_y_m = ~np.all(mask_temp, axis=0)
            axe_x_m = ~np.all(mask_temp, axis=1)
            # skip if nothing to do
            if not np.any(axe_y_m) or not np.any(axe_x_m):
                return None
            # getting indices where we need to cut
            axe_x_min = np.where(axe_x_m)[0][0]
            axe_x_max = np.where(axe_x_m)[0][-1]
            axe_y_min = np.where(axe_y_m)[0][0]
            axe_y_max = np.where(axe_y_m)[0][-1]
            # trim
            if inplace:
                self.trim_area([axe_x_min, axe_x_max],
                               [axe_y_min, axe_y_max],
                               ind=True, inplace=True)
            else:
                tmp_tf = self.copy()
                tmp_tf.trim_area([axe_x_min, axe_x_max],
                                 [axe_y_min, axe_y_max],
                                 ind=True, inplace=True)
                return tmp_tf

    def trim_area(self, intervalx=None, intervaly=None, intervalt=None,
                  full_output=False,
                  ind=False, inplace=False):
        """
        Return a trimed field in respect with given intervals.

        Parameters
        ----------
        intervalx : array, optional
            interval wanted along x
        intervaly : array, optional
            interval wanted along y
        intervalt : array, optional
            interval wanted along time
        full_output : boolean, optional
            If 'True', cutting indices are alson returned
        inplace : boolean, optional
            If 'True', fields are trimed in place.
        """
        # check parameters
        if intervalt is not None:
            if not isinstance(intervalt, ARRAYTYPES):
                raise TypeError()
            intervalt = np.array(intervalt, dtype=float)
            if intervalt.shape != (2, ):
                raise ValueError()
        # get wanted times
        if intervalt is not None:
            if ind:
                intervalt = np.arange(intervalt[0], intervalt[1] + 1)
            else:
                if intervalt[0] < self.times[0]:
                    ind1 = 0
                elif intervalt[0] > self.times[-1]:
                    raise ValueError()
                else:
                    ind1 = np.where(intervalt[0] <= self.times)[0][0]
                if intervalt[1] > self.times[-1]:
                    ind2 = len(self.times) - 1
                elif intervalt[1] < self.times[0]:
                    raise ValueError()
                else:
                    ind2 = np.where(intervalt[1] >= self.times)[0][-1]
                intervalt = [ind1, ind2]
        ### trim
        if inplace:
            trimfield = self
        else:
            trimfield = self.copy()
        # temporal
        if intervalt is not None:
            trimfield.fields = trimfield.fields[intervalt[0]:intervalt[1] + 1]
            trimfield.times = trimfield.times[intervalt[0]:intervalt[1] + 1]
        # spatial
        Field.trim_area(trimfield, intervalx, intervaly, ind=ind,
                        inplace=True)
        for field in trimfield.fields:
            field.trim_area(intervalx, intervaly, ind=ind,
                            inplace=True)
        # returning
        if not inplace:
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
    def display_multiple(self, component, kind=None,  fields_ind=None,
                         samecb=False, same_axes=False, **plotargs):
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

    def display(self, compo=None, suppl_display=None, **plotargs):
        """
        Create a windows to display temporals field, controlled by buttons.
        http://matplotlib.org/1.3.1/examples/widgets/buttons.html
        """
        from matplotlib.widgets import Button, Slider
        # getting data
        if isinstance(self, TemporalVectorFields):
            if compo == 'V' or compo is None:
                comp = self.fields
            else:
                try:
                    comp = self.__getattribute__("{}_as_sf".format(compo))
                except AttributeError:
                    raise ValueError()
        elif isinstance(self, TemporalScalarFields):
            if compo is None:
                compo = "values"
            try:
                comp = self.__getattribute__("{}_as_sf".format(compo))
            except AttributeError:
                raise ValueError()
        else:
            raise TypeError()
        # setting default kind of display
        if 'kind' in plotargs.keys():
            kind = plotargs['kind']
        else:
            kind = None
#        # getting min and max data
#        if isinstance(comp[0], ScalarField):
#            if 'vmin' not in plotargs.keys():
#                mins = [field.min for field in comp.fields]
#                plotargs['vmin'] = np.min(mins)
#            if 'vmax' not in plotargs.keys():
#                maxs = [field.max for field in comp.fields]
#                plotargs['vmax'] = np.max(maxs)
#        elif isinstance(comp[0], VectorField):
#            if 'clim' not in plotargs.keys() and kind is not 'stream':
#                mins = [np.min(field.magnitude[np.logical_not(field.mask)])
#                        for field in comp]
#                maxs = [np.max(field.magnitude[np.logical_not(field.mask)])
#                        for field in comp]
#                mini = np.min(mins)
#                maxi = np.max(maxs)
#                plotargs['clim'] = [mini, maxi]
#        else:
#            raise Exception()

        # button gestion class
        class Index(object):

            def __init__(self, obj, compo, comp, kind, suppl_display,
                         plotargs):
                self.fig = plt.figure()
                self.ax = self.fig.add_axes(plt.axes([0.1, 0.2, .9, 0.7]))
                self.incr = 1
                self.ind = 0
                self.ind_max = len(comp) - 1
                self.obj = obj
                self.compo = compo
                self.comp = comp
                self.kind = kind
                self.suppl_display = suppl_display
                self.plotargs = plotargs
                # display initial
                self.displ = comp[0].display(**self.plotargs)
                self.ttl = plt.title('')
                self.update()

            def next(self, event):
                new_ind = self.ind + self.incr
                if new_ind <= self.ind_max:
                    self.ind = new_ind
                else:
                    self.ind = self.ind_max
                self.bslid.set_val(self.ind + 1)

            def prev(self, event):
                new_ind = self.ind - self.incr
                if new_ind > 0:
                    self.ind = new_ind
                else:
                    self.ind = 0
                self.bslid.set_val(self.ind + 1)

            def slid(self, event):
                self.ind = int(event) - 1
                self.update()

            def update(self):
                self.obj._update_sf(self.ind, self.fig, self.ax,
                                    self.displ, self.ttl, self.comp,
                                    self.compo, self.plotargs)
                if self.suppl_display is not None:
                    self.suppl_display(self.ind)
                plt.draw()

        #window creation
        callback = Index(self, compo, comp, kind, suppl_display, plotargs)
        axprev = callback.fig.add_axes(plt.axes([0.02, 0.02, 0.1, 0.05]))
        axnext = callback.fig.add_axes(plt.axes([0.88, 0.02, 0.1, 0.05]))
        axslid = callback.fig.add_axes(plt.axes([0.15, 0.02, 0.6, 0.05]))
        callback.bnext = Button(axnext, 'Next')
        callback.bnext.on_clicked(callback.next)
        callback.bprev = Button(axprev, 'Previous')
        callback.bprev.on_clicked(callback.prev)
        callback.bslid = Slider(axslid, "", valmin=1, valfmt='%d',
                                valmax=callback.ind_max, valinit=1)
        callback.bslid.on_changed(callback.slid)
        fig = callback.fig
        del callback
        return fig

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
            anim = animation.FuncAnimation(fig, self._update_vf,
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
            anim = animation.FuncAnimation(fig, self._update_sf,
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

    def _update_sf(self, num, fig, ax, displ, ttl, comp, compo, plotargs):
        plt.sca(ax)
        ax.cla()
        displ = comp[num]._display(**plotargs)
        title = "{}, at t={:.3} {}"\
            .format(compo, float(self.times[num]),
                    self.unit_times.strUnit())
        ttl.set_text(title)
        return displ,

    def _update_vf(self, num, fig, ax, displ, ttl, comp, compo, plotargs):
        plt.sca(ax)
        ax.cla()
        displ = comp[num]._display(**plotargs)
        title = "{}, at t={:.2f} {}"\
            .format(compo, float(self.times[num]),
                    self.unit_times.strUnit())
        ttl.set_text(title)
        return displ,


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
    def values_as_sf(self):
        return self

    @property
    def values(self):
        dim = (len(self), self.shape[0], self.shape[1])
        values = np.empty(dim, dtype=float)
        for i, field in enumerate(self.fields):
            values[i, :, :] = field.values[:, :]
        return values

    ### Modifiers ###
    def fill(self, tof='spatial', kind='linear', value=0.,
             inplace=False, crop=False):
        """
        Fill the masked part of the array in place.

        Parameters
        ----------
        tof : string
            Can be 'temporal' for temporal interpolation, or 'spatial' for
            spatial interpolation.
        kind : string, optional
            Type of algorithm used to fill.
            'value' : fill with a given value
            'nearest' : fill with nearest available data
            'linear' : fill using linear interpolation
            'cubic' : fill using cubic interpolation
        value : 2x1 array
            Value for filling, '[Vx, Vy]' (only usefull with tof='value')
        inplace : boolean, optional
            .
        crop : boolean, optional
            If 'True', TVF borders are cropped before filling.
        """
        # TODO : utiliser Profile.fill au lieu d'une nouvelle méthode de filling
        # checking parameters coherence
        if len(self.fields) < 3 and tof == 'temporal':
            raise ValueError("Not enough fields to fill with temporal"
                             " interpolation")
        if not isinstance(tof, STRINGTYPES):
            raise TypeError()
        if tof not in ['temporal', 'spatial']:
            raise ValueError()
        if not isinstance(kind, STRINGTYPES):
            raise TypeError()
        if kind not in ['value', 'nearest', 'linear', 'cubic']:
            raise ValueError()
        if crop:
            self.crop_masked_border()
        # temporal interpolation
        if tof == 'temporal':
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
                prof = self.get_time_profile('values', i, j, ind=True)
                # creating interpolation function
                if kind == 'value':
                    def interp(x):
                        return value
                elif kind == 'nearest':
                    raise Exception("Not implemented yet")
                elif kind == 'linear':
                    prof_filt = np.logical_not(prof.mask)
                    interp = spinterp.interp1d(prof.x[prof_filt],
                                               prof.y[prof_filt],
                                               kind='linear')

                elif kind == 'cubic':
                    prof_filt = np.logical_not(prof.mask)
                    interp = spinterp.interp1d(prof.x[prof_filt],
                                               prof.y[prof_filt],
                                               kind='cubic')
                else:
                    raise ValueError("Invalid value for 'kind'")
                # inplace or not
                fields = self.fields.copy()
                # loop on all profile masked points
                for ind_masked in prof.mask:
                    try:
                        interp_val = interp(prof.x[prof.mask])
                    except ValueError:
                        continue
                    # putting interpolated value in the field
                    fields[prof.mask].values[i, j] = interp_val
                    fields[prof.mask].mask[i, j] = False
        # spatial interpolation
        elif tof == 'spatial':
            if inplace:
                fields = self.fields
            else:
                tmp_tsf = self.copy()
                fields = tmp_tsf.fields
            for i, field in enumerate(fields):
                fields[i].fill(kind=kind, value=value, inplace=True)
        else:
            raise ValueError("Unknown parameter for 'tof' : {}".format(tof))
        # returning
        if inplace:
            self.fields = fields
        else:
            tmp_tsf = self.copy()
            tmp_tsf.fields = fields
            return tmp_tsf


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
        values = TemporalScalarFields()
        for i, field in enumerate(self.fields):
            values.add_field(field.comp_x_as_sf, time=self.times[i],
                             unit_times=self.unit_times)
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
        values = TemporalScalarFields()
        for i, field in enumerate(self.fields):
            values.add_field(field.comp_y_as_sf, time=self.times[i],
                             unit_times=self.unit_times)
        return values

    @property
    def Vy(self):
        dim = (len(self), self.shape[0], self.shape[1])
        values = np.empty(dim, dtype=float)
        for i, field in enumerate(self.fields):
            values[i, :, :] = field.comp_y[:, :]
        return values

    @property
    def magnitude_as_sf(self):
        values = TemporalScalarFields()
        for i, field in enumerate(self.fields):
            values.add_field(field.magnitude_as_sf)
        return values

    @property
    def magnitude(self):
        dim = (len(self), self.shape[0], self.shape[1])
        values = np.empty(dim, dtype=float)
        for i, field in enumerate(self.fields):
            values[i, :, :] = field.magnitude[:, :]
        return values

    @property
    def theta_as_sf(self):
        values = TemporalScalarFields()
        for i, field in enumerate(self.fields):
            values.add_field(field.theta_as_sf)
        return values

    @property
    def theta(self):
        dim = (len(self), self.shape[0], self.shape[1])
        values = np.empty(dim, dtype=float)
        for i, field in enumerate(self.fields):
            values[i, :, :] = field.theta[:, :]
        return values

    ### Watchers ###
    def get_time_auto_correlation(self):
        """
        Return auto correlation based on Vx and Vy.
        """
        Vx0 = self.fields[0].comp_x
        Vy0 = self.fields[0].comp_y
        norm = np.mean(Vx0*Vx0 + Vy0*Vy0)
        corr = np.zeros((len(self.times),))
        for i, time in enumerate(self.times):
            Vxi = self.fields[i].comp_x
            Vyi = self.fields[i].comp_y
            corr[i] = np.mean(Vx0*Vxi + Vy0*Vyi)/norm
        return Profile(self.times, corr, mask=False, unit_x=self.unit_times,
                       unit_y=make_unit(''))

    def get_mean_kinetic_energy(self):
        """
        Calculate the mean kinetic energy.
        """
        final_sf = ScalarField()
        mean_vf = self.get_mean_field()
        values_x = mean_vf.comp_x_as_sf
        values_y = mean_vf.comp_y_as_sf
        final_sf = 1./2*(values_x**2 + values_y**2)
        return final_sf

    def get_tke(self):
        """
        Calculate the turbulent kinetic energy.
        """
        mean_field = self.get_mean_field()
        mean_x = mean_field.comp_x_as_sf
        mean_y = mean_field.comp_y_as_sf
        del mean_field
        tke = TemporalScalarFields()
        for i in np.arange(len(self.fields)):
            comp_x = self.fields[i].comp_x_as_sf - mean_x
            comp_y = self.fields[i].comp_y_as_sf - mean_y
            tke.add_field(1./2*(comp_x**2 + comp_y**2),
                          time=self.times[i],
                          unit_times=self.unit_times)
        return tke

    def get_turbulent_intensity(self):
        """
        Calculate the turbulent intensity.

        TI = sqrt(2/3*k)/sqrt(Vx**2 + Vy**2)
        """
        turb_int = (2./3.*self.get_tke())**(.5)/self.magnitude_as_sf
        return turb_int

    def get_mean_tke(self):
        tke = self.get_tke()
        mean_tke = tke.get_mean_field()
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
                                    unit_x, unit_y, unit_values**2)
        rs_yy_sf = ScalarField()
        rs_yy_sf.import_from_arrays(axe_x, axe_y, rs_yy, mask_rs,
                                    unit_x, unit_y, unit_values**2)
        rs_xy_sf = ScalarField()
        rs_xy_sf.import_from_arrays(axe_x, axe_y, rs_xy, mask_rs,
                                    unit_x, unit_y, unit_values**2)
        return (rs_xx_sf, rs_yy_sf, rs_xy_sf)

    ### Modifiers ###
    def fill(self, tof='spatial', kind='linear', value=[0., 0.],
             inplace=False, crop=False):
        """
        Fill the masked part of the array in place.

        Parameters
        ----------
        tof : string
            Can be 'temporal' for temporal interpolation, or 'spatial' for
            spatial interpolation.
        kind : string, optional
            Type of algorithm used to fill.
            'value' : fill with a given value
            'nearest' : fill with nearest available data
            'linear' : fill using linear interpolation
            'cubic' : fill using cubic interpolation
        value : 2x1 array
            Value for filling, '[Vx, Vy]' (only usefull with tof='value')
        inplace : boolean, optional
            .
        crop : boolean, optional
            If 'True', TVF borders are cropped before filling.
        """
        # TODO : utiliser Profile.fill au lieu d'une nouvelle méthode de filling
        # checking parameters coherence
        if len(self.fields) < 3 and tof == 'temporal':
            raise ValueError("Not enough fields to fill with temporal"
                             " interpolation")
        if not isinstance(tof, STRINGTYPES):
            raise TypeError()
        if tof not in ['temporal', 'spatial']:
            raise ValueError()
        if not isinstance(kind, STRINGTYPES):
            raise TypeError()
        if kind not in ['value', 'nearest', 'linear', 'cubic']:
            raise ValueError()
        if isinstance(value, NUMBERTYPES):
            value = [value, value]
        elif not isinstance(value, ARRAYTYPES):
            raise TypeError()
        value = np.array(value)
        if crop:
            self.crop_masked_border()
        # temporal interpolation
        if tof == 'temporal':
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
                # get time profiles
                prof_x = self.get_time_profile('comp_x', i, j, ind=True)
                prof_y = self.get_time_profile('comp_y', i, j, ind=True)
                # getting masked position on profile
                inds_masked_x = np.where(prof_x.mask)[0]
                inds_masked_y = np.where(prof_y.mask)[0]
                # creating interpolation function
                if kind == 'value':
                    def interp_x(x):
                        return value[0]
                    def interp_y(x):
                        return value[1]
                elif kind == 'nearest':
                    raise Exception("Not implemented yet")
                elif kind == 'linear':
                    prof_filt = np.logical_not(prof_x.mask)
                    interp_x = spinterp.interp1d(prof_x.x[prof_filt],
                                                 prof_x.y[prof_filt],
                                                 kind='linear')
                    prof_filt = np.logical_not(prof_y.mask)
                    interp_y = spinterp.interp1d(prof_y.x[prof_filt],
                                                 prof_y.y[prof_filt],
                                                 kind='linear')
                elif kind == 'cubic':
                    prof_filt = np.logical_not(prof_x.mask)
                    interp_x = spinterp.interp1d(prof_x.x[prof_filt],
                                                 prof_x.y[prof_filt],
                                                 kind='cubic')
                    prof_filt = np.logical_not(prof_y.mask)
                    interp_y = spinterp.interp1d(prof_y.x[prof_filt],
                                                 prof_y.y[prof_filt],
                                                 kind='cubic')
                else:
                    raise ValueError("Invalid value for 'kind'")
                # inplace or not
                fields = self.fields.copy()
                # loop on all x profile masked points
                for ind_masked in inds_masked_x:
                    try:
                        interp_val = interp_x(prof_x.x[ind_masked])
                    except ValueError:
                        continue
                    # putting interpolated value in the field
                    fields[ind_masked].comp_x[i, j] = interp_val
                    fields[ind_masked].mask[i, j] = False
                # loop on all y profile masked points
                for ind_masked in inds_masked_y:
                    try:
                        interp_val = interp_y(prof_y.x[ind_masked])
                    except ValueError:
                        continue
                    # putting interpolated value in the field
                    fields[ind_masked].comp_y[i, j] = interp_val
                    fields[ind_masked].mask[i, j] = False
        # spatial interpolation
        elif tof == 'spatial':
            if inplace:
                fields = self.fields
            else:
                tmp_tvf = self.copy()
                fields = tmp_tvf.fields
            for i, field in enumerate(fields):
                fields[i].fill(kind=kind, value=value, inplace=True)
        else:
            raise ValueError("Unknown parameter for 'tof' : {}".format(tof))
        # returning
        if inplace:
            self.fields = fields
        else:
            tmp_tvf = self.copy()
            tmp_tvf.fields = fields
            return tmp_tvf

class SpatialFields(Fields):
    """
    """

    ### Operators ###
    def __init__(self):
        self.unit_x = make_unit('')
        self.unit_y = make_unit('')
        self.unit_values = make_unit('')
        self.fields_type = None

    def __neg__(self):
        tmp_tf = self.copy()
        for i in np.arange(len(self.fields)):
            tmp_tf.fields[i] = -tmp_tf.fields[i]
        return tmp_tf

    def __mul__(self, other):
        if isinstance(other, (NUMBERTYPES, unum.Unum)):
            final_vfs = self.__class__()
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
        final_vfs = self.__class__()
        for field in self.fields:
            final_vfs.add_field(np.power(field, number))
        return final_vfs

    def __iter__(self):
        for i in np.arange(len(self.fields)):
            yield self.times[i], self.fields[i]

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
    def unit_x(self):
        return self.__unit_x

    @unit_x.setter
    def unit_x(self, unit):
        self.__unit_x = unit
        for field in self.fields:
            field.unit_x = unit

    @property
    def unit_y(self):
        return self.__unit_y

    @unit_y.setter
    def unit_y(self, unit):
        self.__unit_y = unit
        for field in self.fields:
            field.unit_y = unit

    @property
    def unit_values(self):
        return self._SpatialFields__unit_values

    @unit_values.setter
    def unit_values(self, unit):
        self.__unit_values = unit
        for field in self.fields:
            field.unit_values = unit

    @property
    def x_min(self):
        return np.min([field.axe_x[0] for field in self.fields])

    @property
    def x_max(self):
        return np.max([field.axe_x[-1] for field in self.fields])

    @property
    def y_min(self):
        return np.min([field.axe_y[0] for field in self.fields])

    @property
    def y_max(self):
        return np.max([field.axe_y[-1] for field in self.fields])

    def add_field(self, field):
        """
        """
        # check
        if isinstance(self, SpatialScalarFields):
            if not isinstance(field, ScalarField):
                raise TypeError()
        elif isinstance(self, SpatialVectorFields):
            if not isinstance(field, VectorField):
                raise TypeError()
        # first field
        if len(self.fields) == 0:
            self.unit_x = field.unit_x
            self.unit_y = field.unit_y
            self.unit_values = field.unit_values
        # other ones
        else:
            try:
                field.change_unit('x', self.unit_x)
                field.change_unit('y', self.unit_y)
                field.change_unit('values', self.unit_values)
            except unum.IncompatibleUnitsError:
                raise ValueError("Inconsistent unit system")
        # crop fields
        field.crop_masked_border()
        # add field
        Fields.add_field(self, field)

    def get_value(self, x, y, unit=False, error=True):
        """
        Return the field component(s) on the point (x, y).

        Parameters
        ----------
        x, y : number
            Point coordinates
        unit : boolean, optional
            If 'True', component(s) is(are) returned with its unit.
        error : boolean, optional
            If 'True', raise an error if the asked point is outside the fields.
            If 'False', return 'None'
        """
        # get interesting fields
        inter_ind = []
        for i, field in enumerate(self.fields):
            if (x > field.axe_x[0] and x < field.axe_x[-1]
                    and y > field.axe_y[0] and y < field.axe_y[-1]):
                inter_ind.append(i)
        # get values (mean over fields if necessary)
        if len(inter_ind) == 0:
            if error:
                raise ValueError("coordinates outside the fields")
            else:
                return None
        elif len(inter_ind) == 1:
            values = self.fields[inter_ind[0]].get_value(x, y,
                                                         ind=False, unit=False)
        else:
            values = self.fields[inter_ind[0]].get_value(x, y,
                                                         ind=False, unit=False)
            for field in self.fields[inter_ind][1::]:
                values += field.get_value(x, y, ind=False, unit=False)
            values /= len(inter_ind)
        return values

    def get_values_on_grid(self, axe_x, axe_y):
        """
        Return a all the fields in a single evenly-spaced grid.
        (Use interpolation to get the data on the grid points)

        Parameters
        ----------
        axe_x, axe_y : arrays of ndim 1
            Representing the grid axis.
        """
        # check
        if not isinstance(axe_x, ARRAYTYPES):
            raise TypeError()
        if not isinstance(axe_y, ARRAYTYPES):
            raise TypeError()
        axe_x = np.array(axe_x)
        axe_y = np.array(axe_y)
        if isinstance(self.fields[0], ScalarField):
            values = np.zeros(shape=(len(axe_x), len(axe_y)), dtype=float)
            mask = np.zeros(shape=(len(axe_x), len(axe_y)), dtype=bool)
            for i, x in enumerate(axe_x):
                for j, y in enumerate(axe_y):
                    val = self.get_value(x, y, unit=False, error=False)
                    if val is None:
                        mask[i, j] = True
                    else:
                        values[i, j] = val
            tmp_f = ScalarField
            tmp_f.import_from_arrays(axe_x, axe_y, values, mask=mask,
                                     unit_x=self.unit_x, unit_y=self.unit_y,
                                     unit_values=self.unit_values)
        elif isinstance(self.fields[0], VectorField):
            Vx = np.zeros(shape=(len(axe_x), len(axe_y)), dtype=float)
            Vy = np.zeros(shape=(len(axe_x), len(axe_y)), dtype=float)
            mask = np.zeros(shape=(len(axe_x), len(axe_y)), dtype=bool)
            for i, x in enumerate(axe_x):
                for j, y in enumerate(axe_y):
                    val = self.get_value(x, y, unit=False, error=False)
                    if val is None:
                        mask[i, j] = True
                    else:
                        Vx[i, j] = val[0]
                        Vy[i, j] = val[1]

            tmp_f = VectorField()
            tmp_f.import_from_arrays(axe_x, axe_y, Vx, Vy,
                                     mask=mask,
                                     unit_x=self.unit_x, unit_y=self.unit_y,
                                     unit_values=self.unit_values)
        else:
            raise Exception()
        return tmp_f

    def get_profile(self, direction, position, component=None):
        """
        Return a profile of the current fields.

        Parameters
        ----------
        direction : integer
            Direction along which we choose a position (1 for x and 2 for y)
        position : float, interval of float
            Position, interval in which we want a profile

        component : string
            Component wanted for the profile.

        Returns
        -------
        profile : Profile object
            Wanted profile
        """
        # getting data
        if isinstance(self, SpatialVectorFields):
            if component is None:
                component = 'magnitude'
            else:
                try:
                    comp = self.__getattribute__("{}_as_sf".format(component))
                except AttributeError:
                    raise ValueError()
        elif isinstance(self, SpatialScalarFields):
            if component is None:
                component = "values"
            try:
                comp = self.__getattribute__("{}_as_sf".format(component))
            except AttributeError:
                raise ValueError()
        # get fields of interest
        inter_ind = []
        if direction == 1:
            for i, field in enumerate(self.fields):
                if np.any(position < field.axe_x[-1]) and np.any(position > field.axe_x[0]):
                    inter_ind.append(i)
        elif direction == 2:
            for i, field in enumerate(self.fields):
                if np.any(position < field.axe_y[-1]) and np.any(position > field.axe_y[0]):
                    inter_ind.append(i)
        # get profiles
        if len(inter_ind) == 0:
            return None
        elif len(inter_ind) == 1:
            return comp[inter_ind[0]].get_profile(direction=direction,
                                                  position=position, ind=False)[0]
        else:
            # get profiles
            x = np.array([])
            y = np.array([])
            for ind in inter_ind:
                tmp_comp = comp[ind]
                tmp_prof = tmp_comp.get_profile(direction=direction,
                                                position=position, ind=False)[0]
                x = np.concatenate((x, tmp_prof.x))
                y = np.concatenate((y, tmp_prof.y))
            # recreate profile object
            ind_sort = np.argsort(x)
            x = x[ind_sort]
            y = y[ind_sort]
            fin_prof = Profile(x, y, mask=False, unit_x=tmp_prof.unit_x,
                               unit_y = tmp_prof.unit_y)

            return fin_prof

    def _display(self, compo=None, **plotargs):
        """
        """
        # check params
        if len(self.fields) == 0:
            raise Exception()
        # getting data
        if isinstance(self, SpatialVectorFields):
            if compo == 'V' or compo is None:
                comp = self.fields
            else:
                try:
                    comp = self.__getattribute__("{}_as_sf".format(compo))
                except AttributeError:
                    raise ValueError()
        elif isinstance(self, SpatialScalarFields):
            if compo is None:
                compo = "values"
            try:
                comp = self.__getattribute__("{}_as_sf".format(compo))
            except AttributeError:
                raise ValueError()
        else:
            raise TypeError()
        # getting max and min
        v_min = np.min([field.min for field in comp])
        v_max = np.max([field.max for field in comp])
        if 'vmin' in plotargs.keys():
            v_min = plotargs.pop('vmin')
        if 'vmax' in plotargs.keys():
            v_max = plotargs.pop('vmax')
        norm = plt.Normalize(v_min, v_max)
        if 'norm' not in plotargs.keys():
            plotargs['norm'] = norm
        # setting default kind of display
        if 'kind' not in plotargs.keys():
            plotargs['kind'] = None
        if plotargs['kind'] == 'stream':
            if 'density' not in plotargs.keys():
                plotargs['density'] = 1.
            plotargs['density'] = plotargs['density']/(len(self.fields))**.5
        # display
        for field in comp:
            field._display(**plotargs)
        plt.xlim(self.x_min, self.x_max)
        plt.ylim(self.y_min, self.y_max)

    def display(self, compo=None, **plotargs):
        self._display(compo=compo, **plotargs)
        plt.axis('image')
        cb = plt.colorbar()
        cb.set_label(self.unit_values.strUnit())
        plt.tight_layout()

    def copy(self):
        return copy.deepcopy(self)


class SpatialScalarFields(SpatialFields):

    def __init__(self):
        Fields.__init__(self)
        self.fields_type = ScalarField

    @property
    def values_as_sf(self):
        return self.fields


class SpatialVectorFields(SpatialFields):
    """
    Class representing a set of spatial-evolving velocity fields.
    """

    def __init__(self):
        Fields.__init__(self)
        self.fields_type = VectorField

    @property
    def Vx_as_sf(self):
        return [field.comp_x_as_sf for field in self.fields]

    @property
    def Vy_as_sf(self):
        return [field.comp_y_as_sf for field in self.fields]

    @property
    def magnitude_as_sf(self):
        return [field.magnitude_as_sf for field in self.fields]

    @property
    def theta_as_sf(self):
        return [field.theta_as_sf for field in self.fields]
