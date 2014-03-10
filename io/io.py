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


def export_to_file(obj, filepath, compressed=True, **kw):
    """
    Write the object in the specified file usint the JSON format.
    Additionnals arguments for the JSON encoder may be set with the **kw
    argument. Such arguments may be 'indent' (for visual indentation in
    file, default=0) or 'encoding' (to change the file encoding,
    default='utf-8').
    If existing, specified file will be truncated. If not, it will
    be created.

    Parameters
    ----------
    obj :
        Object to store (common objects and IMT objects are supported).
    filepath : string
        Path specifiing where to save the object.
    compressed : boolean, optional
        If 'True' (default), the json file is compressed using gzip.
    """
    # writing the datas
    if not isinstance(filepath, STRINGTYPES):
        raise TypeError("I need a string here, son")
    if not os.path.exists(os.path.dirname(filepath)):
        raise IOError("I think this kind of path is invalid, buddy")
    if not isinstance(compressed, bool):
        raise TypeError("'compressed' must be a boolean")
    if compressed:
        if os.path.splitext(filepath)[1] != ".cimt":
            filepath = filepath + ".cimt"
        with gzip.GzipFile(filepath, 'w') as outfile:
            outfile.write(json.dumps(obj, cls=MyEncoder, **kw))
    else:
        if os.path.splitext(filepath)[1] != ".imt":
            filepath = filepath + ".imt"
        file_h = open(filepath, 'w')
        json.dump(obj, file_h, cls=MyEncoder, **kw)
        file_h.close()



def export_to_matlab(obj, filepath, **kw):
    if not isinstance(filepath, STRINGTYPES):
        raise TypeError("I need a string here, son")
    if not os.path.exists(os.path.dirname(filepath)):
        raise IOError("I think this kind of path is invalid, buddy")
    # TODO : Pas réussit, peut être besoins d'un parser à lui ...;



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
    # writing datas
    if not isinstance(filepath, STRINGTYPES):
        raise TypeError("I need a string here, son")
    if not os.path.exists(filepath):
        if os.path.exists(filepath + ".imt"):
            filepath += ".imt"
        elif os.path.exists(filepath + ".cimt"):
            filepath += ".cimt"
        else:
            raise IOError("I think this file doesn't exist, buddy")
    extension = os.path.splitext(filepath)[1]
    if extension == ".cimt":
        with gzip.GzipFile(filepath, 'r') as isfile:
            json_text = isfile.read()
            obj = json.loads(json_text, cls=MyDecoder, **kw)
    elif extension == ".imt":
        file_h = open(filepath, 'r')
        obj = json.load(file_h, cls=MyDecoder, **kw)
        file_h.close()
    else:
        raise IOError("File is not readable "
                      "(unknown extension : {})".format(extension))
    return obj
