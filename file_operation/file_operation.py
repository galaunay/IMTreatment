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
from ..core import Points, Profile, ScalarField, VectorField, make_unit,\
    ARRAYTYPES, NUMBERTYPES, STRINGTYPES, \
    VelocityField, VelocityFields,\
    TemporalVelocityFields, SpatialVelocityFields
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle

import scipy.io as spio



class MyEncoder(json.JSONEncoder):
    """
    Personnal encoder to write module class ass json.
    """
    def default(self, obj):
        """
        Overwritting the default encoder.
        """
        try:
            obj.__classname__
        except AttributeError:
            pass
        else:
            dic = obj.__dict__
            return dic
        if isinstance(obj, (np.bool, np.bool_)):
            dic = {'__classname__': 'np.bool', 'value': int(obj)}
            return dic
        elif isinstance(obj, unum.Unum):
            dic = {'value': obj._value, 'unit': obj._unit,
                   '__classname__': 'Unum'}
            return dic
        elif isinstance(obj, np.ma.MaskedArray):
            dic = {'__classname__': 'MaskedArray', 'values': obj.data,
                   'dtype': obj.dtype.name, 'mask': obj.mask}
            return dic
        elif isinstance(obj, np.ndarray):
            dic = {'__classname__': 'ndarray', 'values': tuple(obj),
                   'dtype': obj.dtype.name}
            return dic
        return json.JSONEncoder.default(self, obj)


class MyDecoder(json.JSONDecoder):
    """
    Personnal decoder for module class.
    """
    def __init__(self, **kw):
        """
        Overwritting constructor.
        """
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, **kw)

    def object_hook(self, dic):
        """
        Defining object_hook.
        """
        if '__classname__' in dic:
            if dic['__classname__'] == 'Points':
                obj = Points(dic['xy'], dic['v'], dic['unit_x'],
                             dic['unit_y'], dic['unit_v'])
            elif dic['__classname__'] == 'Profile':
                obj = Profile(dic['x'], dic['y'], dic['unit_x'],
                              dic['unit_y'], str(dic['name']))
            elif dic['__classname__'] == 'ScalarField':
                obj = ScalarField()
                obj.__dict__ = dic
            elif dic['__classname__'] == 'VectorField':
                obj = VectorField()
                obj.__dict__ = dic
            elif dic['__classname__'] == 'VelocityField':
                obj = VelocityField()
                obj.__dict__ = dic
            elif dic['__classname__'] == 'VelocityFields':
                obj = VelocityFields()
                obj.__dict__ = dic
            elif dic['__classname__'] == 'SpatialVelocityFields':
                obj = SpatialVelocityFields()
                obj.__dict__ = dic
            elif dic['__classname__'] == 'TemporalVelocityFields':
                obj = TemporalVelocityFields()
                obj.__dict__ = dic
            elif dic['__classname__'] == 'np.bool':
                obj = np.bool(dic['value'])
            elif dic['__classname__'] == 'Unum':
                obj = unum.units.s*1
                obj._value = dic['value']
                obj._unit = dic['unit']
            elif dic['__classname__'] == 'ndarray':
                obj = np.array(dic['values'], dtype=dic['dtype'])
            elif dic['__classname__'] == 'MaskedArray':
                obj = np.ma.masked_array(dic['values'], dic['mask'],
                                         dic['dtype'])
            else:
                raise IOError("I think i don't know this kind of "
                              "variable yet... "
                              "But i'm ready to learn what is a "
                              "'{0}', buddy."
                              .format(dic["__classname__"]))
            return obj
        else:
            return dic


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
    elif tof == 'json' and compressed:
        if os.path.splitext(filepath)[1] != ".cjimt":
            filepath = filepath + ".cjimt"
        f = gzip.open(filepath, 'w')
        json.dump(obj, f, cls=MyEncoder, **kw)
        f.close()
    elif tof == 'json' and not compressed:
        if os.path.splitext(filepath)[1] != ".jimt":
            filepath = filepath + ".jimt"
        f = open(filepath, 'w')
        json.dump(obj, f, cls=MyEncoder, **kw)
        f.close()
    else:
        raise ValueError("I don't even know how you get here...")


def export_to_matlab(obj, name, filepath, **kw):
    if not isinstance(filepath, STRINGTYPES):
        raise TypeError("I need a string here, son")
    if not os.path.exists(os.path.dirname(filepath)):
        raise IOError("I think this kind of path is invalid, buddy")
    dic = matlab_parser(obj, name)
    spio.savemat(filepath, dic, **kw)


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
        elif os.path.exists(filepath + ".jimt"):
            filepath += ".jimt"
        elif os.path.exists(filepath + ".cjimt"):
            filepath += ".cjimt"
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
    elif extension == ".jimt":
        f = open(filepath, 'r')
        obj = json.load(f, cls=MyDecoder, **kw)
        f.close()
    elif extension == ".cjimt":
        with gzip.GzipFile(filepath, 'r') as f:
            obj = json.load(f, cls=MyDecoder, **kw)
    else:
        raise IOError("File is not readable "
                      "(unknown extension : {})".format(extension))
    return obj
