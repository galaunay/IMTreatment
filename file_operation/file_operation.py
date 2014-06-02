# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 19:16:04 2014

@author: glaunay
"""

import os
import pdb
import json
import unum
import gzip
from glob import glob
try:
    import IM
except:
    pass
from ..core import Points, Profile, ScalarField, VectorField, make_unit,\
    ARRAYTYPES, NUMBERTYPES, STRINGTYPES, \
    Fields,\
    TemporalVectorFields, SpatialVectorFields, TemporalScalarFields,\
    SpatialScalarFields
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle

import scipy.io as spio



#class MyEncoder(json.JSONEncoder):
#    """
#    Personnal encoder to write module class ass json.
#    """
#    def default(self, obj):
#        """
#        Overwritting the default encoder.
#        """
#        try:
#            obj.__classname__
#        except AttributeError:
#            pass
#        else:
#            dic = obj.__dict__
#            return dic
#        if isinstance(obj, (np.bool, np.bool_)):
#            dic = {'__classname__': 'np.bool', 'value': int(obj)}
#            return dic
#        elif isinstance(obj, unum.Unum):
#            dic = {'value': obj._value, 'unit': obj._unit,
#                   '__classname__': 'Unum'}
#            return dic
#        elif isinstance(obj, np.ma.MaskedArray):
#            dic = {'__classname__': 'MaskedArray', 'values': obj.data,
#                   'dtype': obj.dtype.name, 'mask': obj.mask}
#            return dic
#        elif isinstance(obj, np.ndarray):
#            dic = {'__classname__': 'ndarray', 'values': tuple(obj),
#                   'dtype': obj.dtype.name}
#            return dic
#        return json.JSONEncoder.default(self, obj)
#
#
#class MyDecoder(json.JSONDecoder):
#    """
#    Personnal decoder for module class.
#    """
#    def __init__(self, **kw):
#        """
#        Overwritting constructor.
#        """
#        json.JSONDecoder.__init__(self, object_hook=self.object_hook, **kw)
#
#    def object_hook(self, dic):
#        """
#        Defining object_hook.
#        """
#        if '__classname__' in dic:
#            if dic['__classname__'] == 'Points':
#                obj = Points(dic['xy'], dic['v'], dic['unit_x'],
#                             dic['unit_y'], dic['unit_v'])
#            elif dic['__classname__'] == 'Profile':
#                obj = Profile(dic['x'], dic['y'], dic['unit_x'],
#                              dic['unit_y'], str(dic['name']))
#            elif dic['__classname__'] == 'ScalarField':
#                obj = ScalarField()
#                obj.__dict__ = dic
#            elif dic['__classname__'] == 'VectorField':
#                obj = VectorField()
#                obj.__dict__ = dic
#            elif dic['__classname__'] == 'SpatialVectorFields':
#                obj = SpatialVectorFields()
#                obj.__dict__ = dic
#            elif dic['__classname__'] == 'TemporalVectorFields':
#                obj = TemporalVectorFields()
#                obj.__dict__ = dic
#            elif dic['__classname__'] == 'np.bool':
#                obj = np.bool(dic['value'])
#            elif dic['__classname__'] == 'Unum':
#                obj = unum.units.s*1
#                obj._value = dic['value']
#                obj._unit = dic['unit']
#            elif dic['__classname__'] == 'ndarray':
#                obj = np.array(dic['values'], dtype=dic['dtype'])
#            elif dic['__classname__'] == 'MaskedArray':
#                obj = np.ma.masked_array(dic['values'], dic['mask'],
#                                         dic['dtype'])
#            else:
#                raise IOError("I think i don't know this kind of "
#                              "variable yet... "
#                              "But i'm ready to learn what is a "
#                              "'{0}', buddy."
#                              .format(dic["__classname__"]))
#            return obj
#        else:
#            return dic


def matlab_parser(obj, name):
    classic_types = (int, float, str, unicode)
    array_types = (list, float)
    if isinstance(obj, classic_types):
        return {name: obj}
    elif isinstance(obj, array_types):
        simple = True
        for val in obj:
            if not isinstance(val, classic_types):
                simple = False
        if simple:
            return {name: obj}
        else:
            raise IOError("Matlab can't handle this kind of variable")
    elif isinstance(obj, Points):
        x = np.zeros((obj.xy.shape[0],))
        y = np.zeros((obj.xy.shape[0],))
        for i in np.arange(obj.xy.shape[0]):
            x[i] = obj.xy[i, 0]
            y[i] = obj.xy[i, 1]
        dic = matlab_parser(list(x), 'x')
        dic.update(matlab_parser(list(y), 'y'))
        dic.update(matlab_parser(list(obj.v), 'v'))
        return {name: dic}
    else:
        raise IOError("Can't parser that : \n {}".format(obj))


def export_to_file(obj, filepath, tof='pickle', compressed=True, **kw):
    """
    Write the object in the specified file.
    Additionnals arguments for the JSON encoder may be set with the **kw
    argument.
    If existing, specified file will be truncated. If not, it will
    be created.

    Parameters
    ----------
    obj :
        Object to store (common and IMT objects are supported).
    filepath : string
        Path specifiing where to save the object.
    tof : string
        Type of resulting file, can be :
        'pickle' (default) : create a binary file
        'json' : create a readable xml-like file (but less efficient)
    compressed : boolean, optional
        If 'True' (default), the file is compressed using gzip.
    """
    # checking parameters coherence
    if not isinstance(filepath, STRINGTYPES):
        raise TypeError("I need a string here, son")
    if not os.path.exists(os.path.dirname(filepath)):
        raise IOError("I think this kind of path is invalid, buddy")
    if not isinstance(tof, STRINGTYPES):
        raise TypeError("'tof' must be 'pickle' or 'json'")
    if not isinstance(compressed, bool):
        raise TypeError("'compressed' must be a boolean")
    # creating/filling up the file
    if tof == 'pickle' and compressed:
        if os.path.splitext(filepath)[1] != ".cimt":
            filepath = filepath + ".cimt"
        f = gzip.open(filepath, 'wb')
        pickle.dump(obj, f, protocol=-1)
        f.close()
    elif tof == 'pickle' and not compressed:
        if os.path.splitext(filepath)[1] != ".imt":
            filepath = filepath + ".imt"
        f = open(filepath, 'wb')
        pickle.dump(obj, f, protocol=-1)
        f.close()
#    elif tof == 'json' and compressed:
#        if os.path.splitext(filepath)[1] != ".cjimt":
#            filepath = filepath + ".cjimt"
#        f = gzip.open(filepath, 'w')
#        json.dump(obj, f, cls=MyEncoder, **kw)
#        f.close()
#    elif tof == 'json' and not compressed:
#        if os.path.splitext(filepath)[1] != ".jimt":
#            filepath = filepath + ".jimt"
#        f = open(filepath, 'w')
#        json.dump(obj, f, cls=MyEncoder, **kw)
#        f.close()
    else:
        raise ValueError("I don't even know how you get here...")


def export_to_matlab(obj, name, filepath, **kw):
    if not isinstance(filepath, STRINGTYPES):
        raise TypeError("I need a string here, son")
    if not os.path.exists(os.path.dirname(filepath)):
        raise IOError("I think this kind of path is invalid, buddy")
    dic = matlab_parser(obj, name)
    spio.savemat(filepath, dic, **kw)


def export_to_vtk(obj, filepath, axis=None):
    """
    Export the field to a .vtk file, for Mayavi use.

    Parameters
    ----------
    filepath : string
        Path where to write the vtk file.
    axis : tuple of strings
        By default, field axe are set to (x,y), if you want
        different axis, you have to specified them here.
        For example, "('z', 'y')", put the x field axis values
        in vtk z axis, and y field axis in y vtk axis.
    """
    if isinstance(obj, ScalarField):
        __export_sf_to_vtk(obj, filepath, axis)
    elif isinstance(obj, VectorField):
        __export_vf_to_vtk(obj, filepath, axis)
    else:
        raise TypeError("Cannot (yet) export this kind of object to vtk")


def __export_sf_to_vtk(obj, filepath, axis=None):
    """
    Export the scalar field to a .vtk file, for Mayavi use.

    Parameters
    ----------
    filepath : string
        Path where to write the vtk file.
    axis : tuple of strings
        By default, scalar field axe are set to (x,y), if you want
        different axis, you have to specified them here.
        For example, "('z', 'y')", put the x scalar field axis values
        in vtk z axis, and y scalar field axis in y vtk axis.
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
    V = obj.values.flatten()
    x = obj.axe_x
    y = obj.axe_y
    point_data = pyvtk.PointData(pyvtk.Scalars(V, 'Scalar Field'))
    x_vtk = 0.
    y_vtk = 0.
    z_vtk = 0.
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
    grid = pyvtk.RectilinearGrid(x_vtk, y_vtk, z_vtk)
    data = pyvtk.VtkData(grid, 'Scalar Field from python', point_data)
    data.tofile(filepath)

def __export_vf_to_vtk(obj, filepath, axis=None):
    """
    Export the vector field to a .vtk file, for Mayavi use.

    Parameters
    ----------
    filepath : string
        Path where to write the vtk file.
    axis : tuple of strings
        By default, scalar field axe are set to (x,y), if you want
        different axis, you have to specified them here.
        For example, "('z', 'y')", put the x scalar field axis values
        in vtk z axis, and y scalar field axis in y vtk axis.
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
    Vx, Vy = obj.comp_x, obj.comp_y
    Vx = Vx.flatten()
    Vy = Vy.flatten()
    x, y = obj.axe_x, obj.axe_y
    x_vtk = 0.
    y_vtk = 0.
    z_vtk = 0.
    vx_vtk = np.zeros(Vx.shape)
    vy_vtk = np.zeros(Vx.shape)
    vz_vtk = np.zeros(Vx.shape)
    if axis[0] == 'x':
        x_vtk = x
        vx_vtk = Vx
    elif axis[0] == 'y':
        y_vtk = x
        vy_vtk = Vx
    else:
        z_vtk = x
        vz_vtk = Vx
    if axis[1] == 'x':
        x_vtk = y
        vx_vtk = Vy
    elif axis[1] == 'y':
        y_vtk = y
        vy_vtk = Vy
    else:
        z_vtk = y
        vz_vtk = Vy
    vect = zip(vx_vtk, vy_vtk, vz_vtk)
    point_data = pyvtk.PointData(pyvtk.Vectors(vect, "Vector field"))
    grid = pyvtk.RectilinearGrid(x_vtk, y_vtk, z_vtk)
    data = pyvtk.VtkData(grid, 'Vector Field from python', point_data)
    data.tofile(filepath)

def import_from_file(filepath, **kw):
    """
    Load and return an object from the specified file using the JSON
    format.
    Additionnals arguments for the JSON decoder may be set with the **kw
    argument. Such as'encoding' (to change the file
    encoding, default='utf-8').

    Parameters
    ----------
    filepath : string
        Path specifiing the file to load.
    """
    # getting/guessing wanted files
    if not isinstance(filepath, STRINGTYPES):
        raise TypeError("I need a string here, son")
    if not os.path.exists(filepath):
        if os.path.exists(filepath + ".imt"):
            filepath += ".imt"
        elif os.path.exists(filepath + ".cimt"):
            filepath += ".cimt"
#        elif os.path.exists(filepath + ".jimt"):
#            filepath += ".jimt"
#        elif os.path.exists(filepath + ".cjimt"):
#            filepath += ".cjimt"
        else:
            raise IOError("I think this file doesn't exist, buddy")
    extension = os.path.splitext(filepath)[1]
    # importing file
    if extension == ".imt":
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
    elif extension == ".cimt":
        with gzip.open(filepath, 'rb') as f:
            obj = pickle.load(f)
#    elif extension == ".jimt":
#        f = open(filepath, 'r')
#        obj = json.load(f, cls=MyDecoder, **kw)
#        f.close()
#    elif extension == ".cjimt":
#        with gzip.GzipFile(filepath, 'r') as f:
#            obj = json.load(f, cls=MyDecoder, **kw)
    else:
        raise IOError("File is not readable "
                      "(unknown extension : {})".format(extension))
    return obj


def import_from_IM7(filename):
    """
    Import a scalar field from a .IM7 file.

    Parameters
    ----------
    filename : string
        Path to the IM7 file.
    """
    if not isinstance(filename, STRINGTYPES):
        raise TypeError("'filename' must be a string")
    if not os.path.exists(filename):
        raise ValueError("I did not find your file, boy")
    _, ext = os.path.splitext(filename)
    if not (ext == ".im7" or ext == ".IM7"):
        raise ValueError("I need the file to be an IM7 file")
    v = IM.IM7(filename)
    axe_x = v.Px[0, :]
    axe_y = v.Py[:, 0]
    values = np.transpose(v.I[0]*v.buffer['scaleI']['factor'])
    unit_x = v.buffer['scaleX']['unit'].split("\x00")[0]
    unit_x = unit_x.replace('[', '')
    unit_x = unit_x.replace(']', '')
    unit_x = make_unit(unit_x)
    unit_y = v.buffer['scaleY']['unit'].split("\x00")[0]
    unit_y = unit_y.replace('[', '')
    unit_y = unit_y.replace(']', '')
    unit_y = make_unit(unit_y)
    unit_values = v.buffer['scaleI']['unit'].split("\x00")[0]
    unit_values = unit_values.replace('[', '')
    unit_values = unit_values.replace(']', '')
    unit_values = make_unit(unit_values)
    # check if axe are crescent
    if axe_y[-1] < axe_y[0]:
        axe_y = axe_y[::-1]
        values = values[:, ::-1]
    if axe_x[-1] < axe_x[0]:
        axe_x = axe_x[::-1]
        values = values[::-1, :]
    tmpsf = ScalarField()
    mask = values.mask
    values = values.data
    tmpsf.import_from_arrays(axe_x=axe_x, axe_y=axe_y, values=values,
                             mask=mask,
                             unit_x=unit_x, unit_y=unit_y,
                             unit_values=unit_values)
    return tmpsf


def import_from_IM7s(fieldspath, kind='TSF', dt=1, t0=0, unit_time='s',
                     fieldnumbers=None, incr=1):
    """
    Import scalar fields from .IM7 files.
    'fieldspath' should be a tuple of path to im7 files.
    All im7 file present in the folder are imported.

    Parameters
    ----------
    fieldspath : string or tuple of string
    kind : string, optional
        Kind of object to create with IM7 files.
        (can be 'TSF' or 'SSF').
    fieldnumbers : 2x1 tuple of int
        Interval of fields to import, default is all.
    incr : integer
        Incrementation between fields to take. Default is 1, meaning all
        fields are taken.
    dt : number
        interval of time between fields.
    t0: number, optional
        Time for the first field.
    """
    if isinstance(fieldspath, ARRAYTYPES):
        if not isinstance(fieldspath[0], STRINGTYPES):
            raise TypeError("'fieldspath' must be a string or a tuple of"
                            " string")
    elif isinstance(fieldspath, STRINGTYPES):
        pattern = os.path.join(fieldspath, '*.IM7')
        fieldspath = glob.glob(pattern)
        if len(fieldspath) == 0:
            raise ValueError()
    else:
        raise TypeError()
    if fieldnumbers is not None:
        if not isinstance(fieldnumbers, ARRAYTYPES):
            raise TypeError("'fieldnumbers' must be a 2x1 array")
        if not len(fieldnumbers) == 2:
            raise TypeError("'fieldnumbers' must be a 2x1 array")
        if not isinstance(fieldnumbers[0], int) \
                or not isinstance(fieldnumbers[1], int):
            raise TypeError("'fieldnumbers' must be an array of integers")
    else:
        fieldnumbers = [0, len(fieldspath)]
    if not isinstance(incr, int):
        raise TypeError("'incr' must be an integer")
    if incr <= 0:
        raise ValueError("'incr' must be positive")
    if not isinstance(dt, NUMBERTYPES):
        raise TypeError("'dt' must be a number")
    # Import
    if kind == 'TSF':
        fields = TemporalScalarFields()
    elif kind == 'SSF':
        fields = SpatialScalarFields()
    else:
        raise ValueError()
    # loop on files
    start = fieldnumbers[0]
    end = fieldnumbers[1]
    t = t0
    for path in fieldspath[start:end:incr]:
        tmp_sf = import_from_IM7(path)
        time = t
        fields.add_field(tmp_sf, time, unit_time)
        t += dt*incr
    return fields
    

def import_from_VC7(filename):
    """
    Import a vector field or a velocity field from a .VC7 file

    Parameters
    ----------
    filename : string
        Path to the file to import.
    velocity : boolean, optional
        If 'False' (default), a VectorField object is returned,
        If 'True', a VelocityField object is returned.
    time : number, optional
        time parameter for VelocityField objects.
    """
    if not isinstance(filename, STRINGTYPES):
        raise TypeError("'filename' must be a string")
    if not os.path.exists(filename):
        raise ValueError("'filename' must ne an existing file")
    _, ext = os.path.splitext(filename)
    if not (ext == ".vc7" or ext == ".VC7"):
        raise ValueError("'filename' must be a vc7 file")
    v = IM.VC7(filename)
    # traitement des unités
    unit_x = v.buffer['scaleX']['unit'].split("\x00")[0]
    unit_x = unit_x.replace('[', '')
    unit_x = unit_x.replace(']', '')
    unit_y = v.buffer['scaleY']['unit'].split("\x00")[0]
    unit_y = unit_y.replace('[', '')
    unit_y = unit_y.replace(']', '')
    unit_values = v.buffer['scaleI']['unit'].split("\x00")[0]
    unit_values = unit_values.replace('[', '')
    unit_values = unit_values.replace(']', '')
    # vérification de l'ordre des axes (et correction)
    x = v.Px[0, :]
    y = v.Py[:, 0]
    Vx = np.transpose(v.Vx[0])
    Vy = np.transpose(v.Vy[0])
    if x[-1] < x[0]:
        x = x[::-1]
        Vx = Vx[::-1, :]
        Vy = Vy[::-1, :]
    if y[-1] < y[0]:
        y = y[::-1]
        Vx = Vx[:, ::-1]
        Vy = Vy[:, ::-1]
    tmpvf = VectorField()
    mask = np.logical_or(Vx.mask, Vy.mask)
    Vx = Vx.data
    Vy = Vy.data
    tmpvf.import_from_arrays(x, y, Vx, Vy, mask=mask, unit_x=unit_x,
                             unit_y=unit_y, unit_values=unit_values)
    return tmpvf


def import_from_VC7s(fieldspath, kind='TVF',dt=1, t0=0, unit_time='s',
                     fieldnumbers=None, incr=1):
    """
    Import velocity fields from .VC7 files.
    'fieldspath' should be a tuple of path to vc7 files.
    All vc7 file present in the folder are imported.

    Parameters
    ----------
    fieldspath : string or tuple of string
    kind : string, optional
        Kind of object to create with VC7 files.
        (can be 'TVF' or 'SVF').
    fieldnumbers : 2x1 tuple of int
        Interval of fields to import, default is all.
    incr : integer
        Incrementation between fields to take. Default is 1, meaning all
        fields are taken.
    dt : number
        interval of time between fields.
    t0 : number, optional
        Time for the first field.
    """
    if isinstance(fieldspath, ARRAYTYPES):
        if not isinstance(fieldspath[0], STRINGTYPES):
            raise TypeError("'fieldspath' must be a string or a tuple of"
                            " string")
    elif isinstance(fieldspath, STRINGTYPES):
        pattern = os.path.join(fieldspath, '*.VC7')
        fieldspath = glob(pattern)
        if len(fieldspath) == 0:
            raise ValueError()
    else:
        raise TypeError()
    if fieldnumbers is not None:
        if not isinstance(fieldnumbers, ARRAYTYPES):
            raise TypeError("'fieldnumbers' must be a 2x1 array")
        if not len(fieldnumbers) == 2:
            raise TypeError("'fieldnumbers' must be a 2x1 array")
        if not isinstance(fieldnumbers[0], int) \
                or not isinstance(fieldnumbers[1], int):
            raise TypeError("'fieldnumbers' must be an array of integers")
    else:
        fieldnumbers = [0, len(fieldspath)]
    if not isinstance(incr, int):
        raise TypeError("'incr' must be an integer")
    if incr <= 0:
        raise ValueError("'incr' must be positive")
    if not isinstance(dt, NUMBERTYPES):
        raise TypeError("'dt' must be a number")
    # Import
    if kind == 'TVF':
        fields = TemporalVectorFields()
    elif kind == 'SVF':
        fields = SpatialVectorFields()
    else:
        raise ValueError()
    # loop on files
    start = fieldnumbers[0]
    end = fieldnumbers[1]
    t = t0
    for path in fieldspath[start:end:incr]:
        tmp_vf = import_from_VC7(path)
        time = t
        fields.add_field(tmp_vf, time, unit_time)
        t += dt*incr
    return fields


def import_sf_from_ascii(filename, x_col=1, y_col=2, vx_col=3,
                         unit_x=make_unit(""),
                         unit_y=make_unit(""),
                         unit_values=make_unit(""), **kwargs):
    """
    Import a scalarfield from an ascii file.

    Parameters
    ----------
    x_col, y_col, vx_col: integer, optional
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
    if not isinstance(x_col, int) or not isinstance(y_col, int)\
            or not isinstance(vx_col, int):
        raise TypeError("'x_col', 'y_col', 'vx_col' and 'vy_col' must "
                        "be integers")
    if x_col < 1 or y_col < 1 or vx_col < 1:
        raise ValueError("Colonne number out of range")
    # 'names' deletion, if specified (dangereux pour la suite)
    if 'names' in kwargs:
        kwargs.pop('names')
    # extract data from file
    data = np.genfromtxt(filename, **kwargs)
    # get axes
    x = data[:, x_col-1]
    x_org = np.unique(x)
    y = data[:, y_col-1]
    y_org = np.unique(y)
    vx = data[:, vx_col-1]
    # Masking all the initial fields (to handle missing values)
    vx_org = np.zeros((y_org.shape[0], x_org.shape[0]))
    vx_org_mask = np.ones(vx_org.shape)
    vx_org = np.ma.masked_array(vx_org, vx_org_mask)
    #loop on all 'v' values
    for i in np.arange(vx.shape[0]):
        x_tmp = x[i]
        y_tmp = y[i]
        vx_tmp = vx[i]
        #find x index
        for j in np.arange(x_org.shape[0]):
            if x_org[j] == x_tmp:
                x_ind = j
        #find y index
        for j in np.arange(y_org.shape[0]):
            if y_org[j] == y_tmp:
                y_ind = j
        #put the value at its place
        vx_org[y_ind, x_ind] = vx_tmp
    # Treating 'nan' values
    vx_org.mask = np.logical_or(vx_org.mask, np.isnan(vx_org.data))

    #store field in attributes
    tmpsf = ScalarField()
    tmpsf.import_from_arrays(x_org, y_org, vx_org, unit_x, unit_y,
                        unit_values)
    return tmpsf


def import_vf_from_ascii(filename, x_col=1, y_col=2, vx_col=3,
                         vy_col=4, unit_x=make_unit(""),
                         unit_y=make_unit(""),
                         unit_values=make_unit(""), **kwargs):
    """
    Import a vectorfield from an ascii file.

    Parameters
    ----------
    x_col, y_col, vx_col, vy_col : integer, optional
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
    if not isinstance(x_col, int) or not isinstance(y_col, int)\
            or not isinstance(vx_col, int):
        raise TypeError("'x_col', 'y_col', 'vx_col' and 'vy_col' must "
                        "be integers")
    if x_col < 1 or y_col < 1 or vx_col < 1:
        raise ValueError("Colonne number out of range")
    if vy_col is not None:
        if not isinstance(vy_col, int):
            raise TypeError("'x_col', 'y_col', 'vx_col' and 'vy_col' must "
                            "be integers")
        if vy_col < 1:
            raise ValueError("Colonne number out of range")
    # 'names' deletion, if specified (dangereux pour la suite)
    if 'names' in kwargs:
        kwargs.pop('names')
    # extract data from file
    data = np.genfromtxt(filename, **kwargs)
    # get axes
    x = data[:, x_col-1]
    x_org = np.unique(x)
    y = data[:, y_col-1]
    y_org = np.unique(y)
    vx = data[:, vx_col-1]
    vy = data[:, vy_col-1]
    # Masking all the initial fields (to handle missing values)
    vx_org = np.zeros((y_org.shape[0], x_org.shape[0]))
    vx_org_mask = np.ones(vx_org.shape)
    vx_org = np.ma.masked_array(vx_org, vx_org_mask)
    vy_org = np.zeros((y_org.shape[0], x_org.shape[0]))
    vy_org_mask = np.ones(vy_org.shape)
    vy_org = np.ma.masked_array(vy_org, vy_org_mask)
    #loop on all 'v' values
    for i in np.arange(vx.shape[0]):
        x_tmp = x[i]
        y_tmp = y[i]
        vx_tmp = vx[i]
        vy_tmp = vy[i]
        #find x index
        for j in np.arange(x_org.shape[0]):
            if x_org[j] == x_tmp:
                x_ind = j
        #find y index
        for j in np.arange(y_org.shape[0]):
            if y_org[j] == y_tmp:
                y_ind = j
        #put the value at its place
        vx_org[y_ind, x_ind] = vx_tmp
        vy_org[y_ind, x_ind] = vy_tmp
    # Treating 'nan' values
    vx_org.mask = np.logical_or(vx_org.mask, np.isnan(vx_org.data))
    vy_org.mask = np.logical_or(vy_org.mask, np.isnan(vy_org.data))
    #store field in attributes
    tmpvf = VectorField()
    tmpvf.import_from_arrays(x_org, y_org, vx_org, vy_org, unit_x, unit_y,
                             unit_values)
    return tmpvf


def import_vfs_from_ascii(filepath, kind='TVF', incr=1, interval=None,
                          x_col=1, y_col=2, vx_col=3,
                          vy_col=4, unit_x=make_unit(""),
                          unit_y=make_unit(""),
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
    if kind == 'TVF':
        fields = TemporalVectorFields()
    elif kind == 'SVF':
        fields = SpatialVectorFields()
    else:
        raise ValueError()
    for i in np.arange(interval[0], interval[1] + 1, incr):
        path = paths[i]
        if len(path) != ref_path_len:
            raise Warning("You should check your files names,"
                          "i may have taken them in the wrong order.")
        tmp_vf = VelocityField()
        tmp_vf.import_from_ascii(path, x_col, y_col, vx_col, vy_col,
                                 unit_x, unit_y, unit_values, times[i],
                                 unit_time, **kwargs)
        fields.add_field(tmp_vf)
    return fields


def IM7_to_ScalarField(im7_path, imt_path, **kwargs):
    """
    Transfome an IM7 (davis) file into a, imt exploitable file.
    
    Parameters
    ----------
    im7_path : path to file or directory
        Path to the IM7 file(s) , can be path to a single file or path to
        a directory contening multiples files.
    imt_path : path to file or directory
        Path where to save imt files, has to be the same type of path
        than 'im7_path' (path to file or path to directory)
    kwargs : dict, optional
        Additional arguments for 'import_from_***()'.
    """
    # checking parameters
    if not isinstance(im7_path, STRINGTYPES):
        raise TypeError()
    if not isinstance(imt_path, STRINGTYPES):
        raise TypeError()
    # checking if file or directory
    if os.path.isdir(im7_path):
        TSF = import_from_IM7s(im7_path, **kwargs)
        export_to_file(TSF, imt_path)
    elif os.path.isfile(im7_path):
        SF = import_from_IM7(im7_path, **kwargs)
        export_to_file(SF, imt_path)
    else:
        raise ValueError()
        
def VC7_to_VectorField(vc7_path, imt_path, **kwargs):
    """
    Transfome an VC7 (davis) file into a, imt exploitable file.
    
    Parameters
    ----------
    vc7_path : path to file or directory
        Path to the VC7 file(s) , can be path to a single file or path to
        a directory contening multiples files.
    imt_path : path to file
        Path where to save imt file.
    kwargs : dict, optional
        Additional arguments for 'import_from_***()'.
    """
    # checking parameters
    if not isinstance(vc7_path, STRINGTYPES):
        raise TypeError()
    if not isinstance(imt_path, STRINGTYPES):
        raise TypeError()
    # checking if file or directory
    if os.path.isdir(vc7_path):
        TVF = import_from_VC7s(vc7_path, **kwargs)
        export_to_file(TVF, imt_path)
    elif os.path.isfile(vc7_path):
        VF = import_from_IM7(vc7_path, **kwargs)
        export_to_file(VF, imt_path)
    else:
        raise ValueError()


def davis_to_imt_gui():
    from Tkinter import Tk
    from tkFileDialog import askopenfilename, asksaveasfilename
    # getting importing directory
    win = Tk()
    filetypes = [('Davis files', '.vc7 .im7'), ('other files', '.*')]
    title = "Choose a file or a directory to import"
    davis_path = askopenfilename(filetypes=filetypes, title=title, multiple=True)
    # exit if no file selected
    if len(davis_path) == 0:
        return None
    davis_path = win.tk.splitlist(davis_path)
    win.destroy()
    pdb.set_trace()
    # importing files
    if len(davis_path) == 1:
        davis_path = davis_path[0]
        ext = os.path.splitext(davis_path)[-1]
        if ext in ['.im7', '.IM7']:
            obj = import_from_IM7(davis_path)
        elif ext in ['.vc7', '.VC7']:
            obj = import_from_VC7(davis_path)
        else:
            raise ValueError()
    elif len(davis_path) > 1:
        # getting extension
        ext = os.path.splitext(davis_path[0])[-1]
        # checking if all files have the same extension
        for path in davis_path:
            if not ext == os.path.splitext(path)[-1]:
                raise ValueError()
        if ext in [".im7", ".IM7"]:
            obj = import_from_IM7s(davis_path)
        if ext in [".vc7", ".VC7"]:
            obj = import_from_VC7s(davis_path)
        else:
            raise ValueError()
    else:
        raise ValueError()
    # getting saving directory
    win = Tk()
    filetypes = [('other files', '.*'), ('IMT files', '.cimt')]
    title = "Choose a file or a directory to import"
    imt_path = asksaveasfilename(filetypes=filetypes, title=title)
    win.destroy()
    # saving datas
    export_to_file(obj, imt_path)