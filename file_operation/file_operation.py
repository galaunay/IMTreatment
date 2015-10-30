# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 19:16:04 2014

@author: glaunay
"""

import os
import pdb
import gzip
from glob import glob
from ..core import Points, ScalarField, VectorField, make_unit,\
    ARRAYTYPES, NUMBERTYPES, STRINGTYPES, \
    TemporalVectorFields, SpatialVectorFields, TemporalScalarFields,\
    SpatialScalarFields, Profile
from ..tools import ProgressCounter
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
import scipy.io as spio
import scipy.misc as spmisc
from os import path
import matplotlib.pyplot as plt
import re

### Path adaptater ###
def check_path(filepath, newfile=False):
    """
    Normalize and check the validity of the given path to feed importation functions.
    """
    # check
    if not isinstance(filepath, STRINGTYPES):
        raise TypeError()
    if not isinstance(newfile, bool):
        raise ValueError()
    # normalize
    filepath = path.normpath(filepath)
    # check validity
    if newfile:
        filepath, filename = path.split(filepath)
    if not path.exists(filepath):
        # split the path (to check existing part)
        path_compos = []
        p = filepath
        while True:
            p, f = path.split(p)
            if f != "":
                path_compos.append(f)
            else:
                if path != "":
                    path_compos.append(p)
                break
        # check validity recursively
        valid_path = ""
        while True:
            if len(path_compos) == 0:
                break
            new_dir = path_compos.pop()
            new_tested_path = path.join(valid_path, new_dir)
            if not path.exists(new_tested_path):
                err_mess = ur"No '{}' directory/file in '{}' path.".format(new_dir, valid_path)
                err_mess = unicode(err_mess).encode("utf-8")
                raise ValueError(err_mess)
            valid_path = new_tested_path
    # returning
    if newfile:
        filepath = path.join(filepath, filename)
    return filepath
    
def find_file_in_path(regs, dirpath, ask=False):
    """
    Search recursively for a folder containing files matching a regular
    expression, in the given root folder.
    
    Parameters
    ----------
    exts : list of string
        List of regular expressions
    dirpath : string
        Root path to search from
    ask : bool
        If 'True', ask for the wanted folder and return only this one.
        
    Returns
    -------
    folders : list of string
        List of folder containings wanted files.
    """
    # check
    if not os.path.isdir(dirpath):
        raise ValueError()
    if not isinstance(regs, ARRAYTYPES):
        raise TypeError()
    regs = np.array(regs, dtype=unicode)
    # 
    dir_paths = []
    # recursive loop on folders
    for root, dirs, files in os.walk(dirpath):
        for f in files:
            match = np.any([re.match(reg, f) for reg in regs])
            if match:
                break
        if match:
            dir_paths.append(root)
    # choose
    if ask and len(dir_paths) >  1:
        print("{} folders found :".format(len(dir_paths)))
        for i, p in enumerate(dir_paths):
            print("  {} :  {}".format(i + 1, p))
        rep = 0
        while rep not in np.arange(1, len(dir_paths)+1):
            rep = input("Want to go with wich one ?\n")
        dir_paths = [dir_paths[rep - 1]]
    # return
    return dir_paths


    
### Parsers ###
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


### IMT ###
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
    filepath = check_path(filepath)
    extension = os.path.splitext(filepath)[1]
    # importing file
    if extension == ".imt":
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
    elif extension == ".cimt":
        with gzip.open(filepath, 'rb') as f:
            obj = pickle.load(f)
    else:
        raise IOError("File is not readable "
                      "(unknown extension : {})".format(extension))
    return obj


def export_to_file(obj, filepath, compressed=True, **kw):
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
    compressed : boolean, optional
        If 'True' (default), the file is compressed using gzip.
    """
    # checking parameters coherence
    filepath = check_path(filepath, newfile=True)
    if not isinstance(compressed, bool):
        raise TypeError("'compressed' must be a boolean")
    # creating/filling up the file
    if compressed:
        if os.path.splitext(filepath)[1] != ".cimt":
            filepath = filepath + ".cimt"
        f = gzip.open(filepath, 'wb')
        pickle.dump(obj, f, protocol=-1)
        f.close()
    else:
        if os.path.splitext(filepath)[1] != ".imt":
            filepath = filepath + ".imt"
        f = open(filepath, 'wb')
        pickle.dump(obj, f, protocol=-1)
        f.close()


def imts_to_imt(imts_path, imt_path, kind):
    """
    Concatenate some .imt files to one .imt file.

    Parameters
    ----------
    imts_path : string
        Path to the .imt files
    imt_path : string
        Path to store the new imt file.
    kind : string
        Kind of object for the new imt file
        (can be 'TSF' for TemporalScalarFields, 'SSF' for SpatialScalarFields,
        'TVF' for TemporalVectorFields, 'SVF' for SpatialVectorFields)
    """
    # check parameters
    imts_path = check_path(imts_path)
    imt_path = check_path(imt_path, newfile=True)
    if not isinstance(kind, STRINGTYPES):
        raise TypeError()
    # getting paths
    paths = glob(imts_path + "/*")
    # getting data type
    if kind == 'TSF':
        imts_type = 'SF'
        fields = TemporalScalarFields()
    elif kind == 'SSF':
        imts_type = 'SF'
        fields = SpatialScalarFields()
    elif kind == 'SVF':
        imts_type = 'VF'
        fields = TemporalVectorFields()
    elif kind == 'SVF':
        imts_type = 'VF'
        fields = SpatialVectorFields()
    # importing data
    for path in paths:
        basename = os.path.basename(path)
        name, ext = os.path.splitext(basename)
        if ext in ['.imt', '.cimt']:
            field = import_from_file(path)
            if imts_type == 'SF' and not isinstance(field, ScalarField):
                continue
            elif imts_type == 'VF' and not isinstance(field, VectorField):
                continue
            fields.add_field(field)
    # saving data
    export_to_file(fields, imt_path)


### MATLAB ###
def export_to_matlab(obj, name, filepath, **kw):
    filepath = check_path(filepath)
    dic = matlab_parser(obj, name)
    spio.savemat(filepath, dic, **kw)


### VTK ###
def export_to_vtk(obj, filepath, axis=None, **kw):
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
    line : boolean (only for Points object)
        If 'True', lines between points are writen instead of points.
    """
    if isinstance(obj, ScalarField):
        __export_sf_to_vtk(obj, filepath, axis)
    elif isinstance(obj, VectorField):
        __export_vf_to_vtk(obj, filepath, axis)
    elif isinstance(obj, Points):
        __export_pts_to_vtk(obj, filepath, **kw)
    else:
        raise TypeError("Cannot (yet) export this kind of object to vtk")


def __export_pts_to_vtk(pts, filepath, axis=None, line=False):
    """
    Export the Points object to a .vtk file, for Mayavi use.

    Parameters
    ----------
    pts : Point object
        .
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
    v = pts.v
    x = pts.xy[:, 0]
    y = pts.xy[:, 1]
    if v is None:
        v = np.zeros(pts.xy.shape[0])
    point_data = pyvtk.PointData(pyvtk.Scalars(v, 'Points values'))
    x_vtk = np.zeros(pts.xy.shape[0])
    y_vtk = np.zeros(pts.xy.shape[0])
    z_vtk = np.zeros(pts.xy.shape[0])
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


### DAVIS ###
def _get_imx_buffers(filename):
    """
    Return the buffers stored in the given file.
    """
    import platform
    syst = platform.system()
    if syst == 'Linux':
        import libim7
        vbuff, vatts = libim7.readim7(filename)
        atts = vatts.as_dict()
        vectorGrid = vbuff.vectorGrid
        arrays = np.array(vbuff.blocks.transpose((0, 2, 1)))
        fmt = vbuff.header.buffer_format
        libim7.del_buffer(vbuff)
        libim7.del_attributelist(vatts)
        return fmt, vectorGrid, arrays, atts
    elif syst == 'Windows':
        import ReadIM
        vbuff, vatts = ReadIM.extra.get_Buffer_andAttributeList(filename)
        arrays, vbuff2 = ReadIM.extra.buffer_as_array(vbuff)
        arrays = np.array(arrays.transpose((0, 2, 1)))
        atts = ReadIM.extra.att2dict(vatts)
        fmt = vbuff.image_sub_type
        vectorGrid = vbuff.vectorGrid
        ReadIM.DestroyBuffer(vbuff)
        ReadIM.DestroyBuffer(vbuff2)
        return fmt, vectorGrid, arrays, atts
    else:
        raise Exception()

def import_from_IM7(filename, infos=False):
    """
    Import a scalar field from a .IM7 file.

    Parameters
    ----------
    filename : string
        Path to the IM7 file.
    infos : boolean, optional
        If 'True', also return a dictionary with informations on the im7
    """
    if not isinstance(filename, STRINGTYPES):
        raise TypeError("'filename' must be a string")
    if isinstance(filename, unicode):
        raise TypeError("Unfortunately, ReadIM don't support unicode paths...")
    if not os.path.exists(filename):
        raise ValueError("I did not find your file, boy")
    _, ext = os.path.splitext(filename)
    if not (ext == ".im7" or ext == ".IM7"):
        raise ValueError("I need the file to be an IM7 file (not a {} file)"
                         .format(ext))
    # Importing from buffer
    fmt, vectorGrid, v_array, atts =  _get_imx_buffers(filename)
    if v_array.shape[0] == 2:
        mask = v_array[0]
        values = v_array[1]
    elif v_array.shape[0] == 1:
        values = v_array[0]
        mask = np.zeros(values.shape, dtype=bool)
    # Values and Mask
    scale_i = atts['_SCALE_I']
    scale_i = scale_i.split("\n")
    scale_val = scale_i[0].split(' ')
    unit_values = scale_i[1]
    values *= float(scale_val[0])
    values += float(scale_val[1])
    # X
    scale_x = atts['_SCALE_X']
    scale_x = scale_x.split("\n")
    unit_x = scale_x[1]
    scale_val = scale_x[0].split(' ')
    x_init = float(scale_val[1])
    dx = float(scale_val[0])
    len_axe_x = values.shape[0]
    if dx < 0:
        axe_x = x_init + np.arange(len_axe_x - 1, -1, -1)*dx
        values = values[::-1, :]
        mask = mask[::-1, :]
    else:
        axe_x = x_init + np.arange(len_axe_x)*dx
    # Y
    scale_y = atts['_SCALE_Y']
    scale_y = scale_y.split("\n")
    unit_y = scale_y[1]
    scale_val = scale_y[0].split(' ')
    y_init = float(scale_val[1])
    dy = float(scale_val[0])
    len_axe_y = values.shape[1]
    if dy < 0:
        axe_y = y_init + np.arange(len_axe_y - 1, -1, -1)*dy
        values = values[:, ::-1]
        mask = mask[:, ::-1]
    else:
        axe_y = y_init + np.arange(len_axe_y)*dy
    # returning
    tmpsf = ScalarField()
    tmpsf.import_from_arrays(axe_x=axe_x, axe_y=axe_y, values=values,
                             mask=mask,
                             unit_x=unit_x, unit_y=unit_y,
                             unit_values=unit_values)
    if infos:
        return tmpsf, atts
    else:
        return tmpsf


def import_from_IM7s(fieldspath, kind='TSF', fieldnumbers=None, incr=1):
    """
    Import scalar fields from .IM7 files.
    'fieldspath' should be a tuple of path to im7 files.
    All im7 file present in the folder are imported.

    Parameters
    ----------
    fieldspath : string or tuple of string
    kind : string, optional
        Kind of object to create with IM7 files.
        (can be 'TSF' for TemporalScalarFields
         or 'SSF' for SpatialScalarFields).
    fieldnumbers : 2x1 tuple of int
        Interval of fields to import, default is all.
    incr : integer
        Incrementation between fields to take. Default is 1, meaning all
        fields are taken.
    """
    # check parameters
    if isinstance(fieldspath, ARRAYTYPES):
        if not isinstance(fieldspath[0], STRINGTYPES):
            raise TypeError("'fieldspath' must be a string or a tuple of"
                            " string")
        fieldspaths = np.array(fieldspath)
    elif isinstance(fieldspath, STRINGTYPES):
        fieldspath = check_path(fieldspath)
        paths = np.array([f for f in glob(os.path.join(fieldspath, '*'))
                          if os.path.splitext(f)[-1] in ['.im7', '.IM7']])
        # if no file found, search recursively
        if len(paths) == 0:
            poss_paths = find_file_in_path(['.*.im7', '.*.IM7'], fieldspath,
                                           ask=True)
            if len(poss_paths) == 0:
                raise ValueError()
            paths = np.array([f for f in glob(os.path.join(poss_paths[0], '*'))
                              if os.path.splitext(f)[-1] in ['.im7', '.IM7']])
        # Sort path by numbers
        filenames = [os.path.basename(p) for p in paths]
        ind_sort = np.argsort(filenames)
        paths = paths[ind_sort]
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
        fieldnumbers = [0, len(fieldspaths)]
    if not isinstance(incr, int):
        raise TypeError("'incr' must be an integer")
    if incr <= 0:
        raise ValueError("'incr' must be positive")
    # Import
    if kind == 'TSF':
        fields = TemporalScalarFields()
    elif kind == 'SSF':
        fields = SpatialScalarFields()
    else:
        raise ValueError()
    start = fieldnumbers[0]
    end = fieldnumbers[1]
    t = 0.
    # loop on files
    for p in fieldspaths[start:end:incr]:
        tmp_sf, infos = import_from_IM7(p, infos=True)
        try:
            dt = infos['FrameDt0'].split()
        except KeyError:
            dt = [1., ""]
        unit_time = make_unit(dt[1])
        dt = float(dt[0])
        t += dt*incr
        if kind == 'TSF':
            fields.add_field(tmp_sf, t, unit_time)
        else:
            fields.add_field(tmp_sf)
    return fields


def import_from_VC7(filename, infos=False, add_fields=False):
    """
    Import a vector field or a velocity field from a .VC7 file

    Parameters
    ----------
    filename : string
        Path to the file to import.
    infos : boolean, optional
        If 'True', also return a dictionary with informations on the im7
    add_fields : boolean, optional
        If 'True', also return a tuple containing additional fields
        contained in the vc7 field (peak ratio, correlation value, ...)
    """
    # check parameters
    filename = check_path(filename)
    _, ext = os.path.splitext(filename)
    if not (ext == ".vc7" or ext == ".VC7"):
        raise ValueError("'filename' must be a vc7 file")
    # Importing from buffer
    fmt, vectorGrid, v_array, atts = _get_imx_buffers(filename)
    # Values and Mask
    if fmt == 2:
        Vx = v_array[0]
        Vy = v_array[1]
        mask = np.zeros(Vx.shape, dtype=bool)
    elif fmt == 3 or fmt == 1:
        mask = np.logical_not(v_array[0])
        mask2 = np.logical_not(v_array[9])
        mask = np.logical_or(mask, mask2)
        Vx = v_array[1]
        Vy = v_array[2]
    mask = np.logical_or(mask, np.logical_and(Vx == 0., Vy == 0.))
    # additional fields if necessary
    if add_fields and fmt in [1, 3]:
        suppl_fields = []
        for i in np.arange(4, v_array.shape[0]):
            suppl_fields.append(np.transpose(np.array(v_array[i])))
    # Get and apply scale on values
    scale_i = atts['_SCALE_I']
    scale_i = scale_i.split("\n")
    unit_values = scale_i[1]
    scale_val = scale_i[0].split(' ')
    Vx *= float(scale_val[0])
    Vx += float(scale_val[1])
    Vy *= float(scale_val[0])
    Vy += float(scale_val[1])
    # Get and apply scale on X
    scale_x = atts['_SCALE_X']
    scale_x = scale_x.split("\n")
    unit_x = scale_x[1]
    scale_val = scale_x[0].split(' ')
    x_init = float(scale_val[1])
    dx = float(scale_val[0])*vectorGrid
    len_axe_x = Vx.shape[0]
    if dx < 0:
        axe_x = x_init + np.arange(len_axe_x - 1, -1, -1)*dx
        Vx = -Vx[::-1, :]
        Vy = Vy[::-1, :]
        mask = mask[::-1, :]
        if add_fields:
            for i in np.arange(len(suppl_fields)):
                suppl_fields[i] = suppl_fields[i][::-1, :]
    else:
        axe_x = x_init + np.arange(len_axe_x)*dx
    # Get and apply scale on Y
    scale_y = atts['_SCALE_Y']
    scale_y = scale_y.split("\n")
    unit_y = scale_y[1]
    scale_val = scale_y[0].split(' ')
    y_init = float(scale_val[1])
    dy = float(scale_val[0])*vectorGrid
    len_axe_y = Vx.shape[1]
    if dy < 0 or scale_y[1] == 'pixel':
        axe_y = y_init + np.arange(len_axe_y - 1, -1, -1)*dy
        Vx = Vx[:, ::-1]
        Vy = -Vy[:, ::-1]
        mask = mask[:, ::-1]
        if add_fields:
            for i in np.arange(len(suppl_fields)):
                suppl_fields[i] = suppl_fields[i][:, ::-1]
    else:
        axe_y = y_init + np.arange(len_axe_y)*dy
    # returning
    tmpvf = VectorField()
    tmpvf.import_from_arrays(axe_x, axe_y, Vx, Vy, mask=mask, unit_x=unit_x,
                             unit_y=unit_y, unit_values=unit_values)
    if not infos and not add_fields:
        return tmpvf
    res = ()
    res += (tmpvf,)
    if infos:
        res += (atts,)
    if add_fields:
        add_fields = []
        for i in np.arange(len(suppl_fields)):
            tmp_field = ScalarField()
            tmp_field.import_from_arrays(axe_x, axe_y, suppl_fields[i],
                                         unit_x=unit_x, unit_y=unit_y,
                                         unit_values='')
            add_fields.append(tmp_field)
        res += (add_fields,)
    return res

def import_from_VC7s(fieldspath, kind='TVF', fieldnumbers=None, incr=1,
                     add_fields=False, verbose=False):
    """
    Import velocity fields from .VC7 files.
    'fieldspath' should be a tuple of path to vc7 files.
    All vc7 file present in the folder are imported.

    Parameters
    ----------
    fieldspath : string or tuple of string
        If no '.vc7' are found directly under 'fieldspath', present folders are
        recursively serached for '.vc7' files.
    kind : string, optional
        Kind of object to create with VC7 files.
        (can be 'TVF' or 'SVF').
    fieldnumbers : 2x1 tuple of int
        Interval of fields to import, default is all.
    incr : integer
        Incrementation between fields to take. Default is 1, meaning all
        fields are taken.
    add_fields : boolean, optional
        If 'True', also return a tuple containing additional fields
        contained in the vc7 field (peak ratio, correlation value, ...).
    Verbose : bool, optional
        .
    """
    # check and adpat 'fieldspath'
    if isinstance(fieldspath, ARRAYTYPES):
        if not isinstance(fieldspath[0], STRINGTYPES):
            raise TypeError("'fieldspath' must be a string or a tuple of"
                            " string")
        paths = fieldspath
    elif isinstance(fieldspath, STRINGTYPES):
        fieldspath = check_path(fieldspath)
        paths = np.array([f for f in glob(os.path.join(fieldspath, '*'))
                          if os.path.splitext(f)[-1] in ['.vc7', '.VC7']])
        # if no file found, search recursively
        if len(paths) == 0:
            poss_paths = find_file_in_path(['.*.vc7', '.*.VC7'], fieldspath,
                                           ask=True)
            if len(poss_paths) == 0:
                raise ValueError()
            paths = np.array([f for f in glob(os.path.join(poss_paths[0], '*'))
                              if os.path.splitext(f)[-1] in ['.vc7', '.VC7']])
        # Sort path by numbers
        filenames = [os.path.basename(p) for p in paths]
        ind_sort = np.argsort(filenames)
        paths = paths[ind_sort]
    else:
        raise TypeError()
    # check and adapt 'fieldnumbers'
    if fieldnumbers is not None:
        if not isinstance(fieldnumbers, ARRAYTYPES):
            raise TypeError("'fieldnumbers' must be a 2x1 array")
        if not len(fieldnumbers) == 2:
            raise TypeError("'fieldnumbers' must be a 2x1 array")
        if not isinstance(fieldnumbers[0], int) \
                or not isinstance(fieldnumbers[1], int):
            raise TypeError("'fieldnumbers' must be an array of integers")
    else:
        fieldnumbers = [0, len(paths)]
    # check and adpat 'incr'
    if not isinstance(incr, int):
        raise TypeError("'incr' must be an integer")
    if incr <= 0:
        raise ValueError("'incr' must be positive")
    # Prepare containers
    if kind == 'TVF':
        fields = TemporalVectorFields()
    elif kind == 'SVF':
        fields = SpatialVectorFields()
    else:
        raise ValueError()
    # initialize counter
    start = fieldnumbers[0]
    end = fieldnumbers[1]
    nmb_files = int((end - start)/incr)
    pc = ProgressCounter("Begin importation of {} VC7 files"
                         .format(nmb_files),
                         "Done", nmb_files, name_things="VC7 files",
                         perc_interv=10)
    # loop on files
    t = 0.
    if add_fields:
        tmp_vf, add_fields = import_from_VC7(paths[0], add_fields=True)
        suppl_fields = [TemporalScalarFields() for field in add_fields]
    for i, p in enumerate(paths[start:end:incr]):
        if verbose:
            pc.print_progress()
        if add_fields:
            tmp_vf, infos, add_fields = import_from_VC7(p, infos=True,
                                                        add_fields=True)
            dt = infos['FrameDt0'].split()
            unit_time = make_unit(dt[1])
            dt = float(dt[0])
            t += dt*incr
            fields.add_field(tmp_vf, t, unit_time)
            for i, f in enumerate(add_fields):
                suppl_fields[i].add_field(f, t, unit_time)
        else:
            tmp_vf, infos = import_from_VC7(p, infos=True)
            dt = infos['FrameDt0'].split()
            unit_time = make_unit(dt[1])
            dt = float(dt[0])
            t += dt*incr
            fields.add_field(tmp_vf, t, unit_time)
    # return
    if add_fields:
        return fields, suppl_fields
    else:
        return fields


def IM7_to_imt(im7_path, imt_path, kind='SF', compressed=True, **kwargs):
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
    kind : string
        Kind of object to store (can be 'TSF' for TemporalScalarFields,
        'SSF' for SpatialScalarFields or 'SF' for multiple ScalarField)
    compressed : boolean, optional
        If 'True' (default), the file is compressed using gzip.
    kwargs : dict, optional
        Additional arguments for 'import_from_***()'.
    """
    # checking parameters
    im7_path = check_path(im7_path)
    imt_path = check_path(imt_path, newfile=True)
    # checking if file or directory
    if os.path.isdir(im7_path):
        if kind in ['SSF', 'TSF']:
            ST_SF = import_from_IM7s(im7_path, kind=kind, **kwargs)
            export_to_file(ST_SF, imt_path)
        elif kind in ['SF']:
            paths = glob(im7_path + "/*")
            for path in paths:
                name_ext = os.path.basename(path)
                name, ext = os.path.splitext(name_ext)
                if ext not in ['.im7', '.IM7']:
                    continue
                SF = import_from_IM7(path)
                export_to_file(SF, imt_path + "/{}".format(name),
                               compressed=compressed)
    elif os.path.isfile(im7_path):
        SF = import_from_IM7(im7_path, **kwargs)
        export_to_file(SF, imt_path)
    else:
        raise ValueError()


def VC7_to_imt(vc7_path, imt_path, kind='VF', compressed=True, **kwargs):
    """
    Transfome an VC7 (davis) file into a, imt exploitable file.

    Parameters
    ----------
    vc7_path : path to file or directory
        Path to the VC7 file(s) , can be path to a single file or path to
        a directory contening multiples files.
    imt_path : path to file
        Path where to save imt file.
    kind : string
        Kind of object to store (can be 'TVF' for TemporalVectorFields,
        'SVF' for SpatialVectorFields or 'VF' for multiple VectorField)
    compressed : boolean, optional
        If 'True' (default), the file is compressed using gzip.
    kwargs : dict, optional
        Additional arguments for 'import_from_***()'.
    """
    # checking parameters
    vc7_path = check_path(vc7_path)
    imt_path = check_path(imt_path)
    # checking if file or directory
    if os.path.isdir(vc7_path):
        if kind in ['SVF', 'TVF']:
            ST_VF = import_from_VC7s(vc7_path, kind=kind, **kwargs)
            export_to_file(ST_VF, imt_path)
        elif kind in ['VF']:
            paths = glob(vc7_path + "/*")
            for path in paths:
                name_ext = os.path.basename(path)
                name, ext = os.path.splitext(name_ext)
                if ext not in ['.vc7', '.VC7']:
                    continue
                VF = import_from_VC7(path)
                export_to_file(VF, imt_path + "/{}".format(name),
                               compressed=compressed)
    elif os.path.isfile(vc7_path):
        SF = import_from_IM7(vc7_path, **kwargs)
        export_to_file(SF, imt_path)
    else:
        raise ValueError()


def davis_to_imt_gui():
    from Tkinter import Tk
    from tkFileDialog import askopenfilename, asksaveasfilename
    # getting importing directory
    win = Tk()
    filetypes = [('Davis files', '.vc7 .im7'), ('other files', '.*')]
    title = "Choose a file or a directory to import"
    davis_path = askopenfilename(filetypes=filetypes, title=title,
                                 multiple=True)
    # exit if no file selected
    if len(davis_path) == 0:
        return None
    davis_path = win.tk.splitlist(davis_path)
    win.destroy()
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

### PICTURES ###
def import_from_picture(filepath, axe_x=None, axe_y=None, unit_x='', unit_y='',
                        unit_values=''):
    """
    Import a scalar field from a picture file.

    Parameters
    ----------
    filepath : string
        Path to the picture file.
    axe_x :
        .
    axe_y :
        .
    unit_x :
        .
    unit_y :
        .
    unit_values :
        .

    Returns
    -------
    tmp_sf :
        .
    """
    usable_ext = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp',
                  '.BMP']
    filepath = check_path(filepath)
    _, ext = os.path.splitext(filepath)
    if not ext in usable_ext:
        raise ValueError("I need the file to be an supported picture file"
                         "(not a {} file)".format(ext))
    # importing from file
    values = spmisc.imread(filepath, flatten=True).transpose()[:, ::-1]
    # set axis
    if axe_x is None:
        axe_x = np.arange(values.shape[0])
    else:
        if len(axe_x) != values.shape[0]:
            raise ValueError()
    if axe_y is None:
        axe_y = np.arange(values.shape[1])
    else:
        if len(axe_y) != values.shape[1]:
            raise ValueError()
    # create SF
    tmp_sf = ScalarField()
    tmp_sf.import_from_arrays(axe_x, axe_y, values, unit_x=unit_x,
                              unit_y=unit_y, unit_values=unit_values)
    # return
    return tmp_sf


def import_from_pictures(filepath, axe_x=None, axe_y=None, unit_x='', unit_y='',
                         unit_values='', times=None, unit_times=''):
    """
    Import scalar fields from a bunch of picture files.

    Parameters
    ----------
    filepath : string
        Path to the files.
    axe_x :
        .
    axe_y :
        .
    unit_x :
        .
    unit_y :
        .
    unit_values :
        .

    Returns
    -------
    tmp_sf :
        .
    """
    # get paths
    filepath = check_path(filepath)
    paths = glob(filepath)
    tmp_tsf = TemporalScalarFields()
    # check times
    if times is None:
        times = np.arange(len(paths))
    elif len(times) != len(paths):
        raise ValueError()
    # loop on paths
    for i, p in enumerate(paths):
        tmp_sf = import_from_picture(p, axe_x=axe_x, axe_y=axe_y,
                                     unit_x=unit_x, unit_y=unit_y,
                                     unit_values=unit_values)
        tmp_tsf.add_field(tmp_sf, times[i], unit_times=unit_times)
    # returning
    return tmp_tsf

def export_to_picture(SF, filepath):
    """
    Export a scalar field to a picture file.

    Parameters
    ----------
    SF :
        .
    filepath : string
        Path to the picture file.
    """
    filepath = check_path(filepath)
    values = SF.values[:, ::-1].transpose()
    spmisc.imsave(filepath, values)

def export_to_pictures(SFs, filepath):
    """
    Export a scalar fields to a picture file.

    Parameters
    ----------
    SF :
        .
    filename : string
        Path to the picture file.
    """
    #check
    filepath = check_path(filepath, newfile=True)
    # get
    values = []
    if isinstance(SFs, ARRAYTYPES):
        for i in np.arange(len(SFs)):
            values.append(SFs[i].values)
    elif isinstance(SFs, (SpatialScalarFields, TemporalScalarFields)):
        for i in np.arange(len(SFs.fields)):
            values.append(SFs.fields[i].values[:, ::-1].transpose())
    # save
    for i, val in enumerate(values):
        spmisc.imsave(path.join(filepath, "{:0>5}.png".format(i)), val)


### ASCII ###
def import_profile_from_ascii(filepath, x_col=1, y_col=2,
                              unit_x=make_unit(""), unit_y=make_unit(""),
                              **kwargs):
        """
        Import a Profile object from an ascii file.

        Parameters
        ----------
        x_col, y_col : integer, optional
            Colonne numbers for the given variables
            (begining at 1).
        unit_x, unit_y : Unit objects, optional
            Unities for the given variables.
        **kwargs :
            Possibles additional parameters are the same as those used in the
            numpy function 'genfromtext()' :
            'delimiter' to specify the delimiter between colonnes.
            'skip_header' to specify the number of colonne to skip at file
                begining
            ...
        """
        # check
        filepath = check_path(filepath)
        # validating parameters
        if not isinstance(x_col, int) or not isinstance(y_col, int):
            raise TypeError("'x_col', 'y_col' must be integers")
        if x_col < 1 or y_col < 1:
            raise ValueError("Colonne number out of range")
        # 'names' deletion, if specified (dangereux pour la suite)
        if 'names' in kwargs:
            kwargs.pop('names')
        # extract data from file
        data = np.genfromtxt(filepath, **kwargs)
        # get axes
        x = data[:, x_col-1]
        y = data[:, y_col-1]
        prof = Profile(x, y, mask=False, unit_x=unit_x, unit_y=unit_y)
        return prof


def import_pts_from_ascii(pts, filepath, x_col=1, y_col=2, v_col=None,
                          unit_x=make_unit(""), unit_y=make_unit(""),
                          unit_v=make_unit(""), **kwargs):
        """
        Import a Points object from an ascii file.

        Parameters
        ----------
        pts : Points object
            .
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
        # check
        filepath = check_path(filepath)
        # validating parameters
        if not isinstance(pts, Points):
            raise TypeError()
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
        data = np.genfromtxt(filepath, **kwargs)
        # get axes
        x = data[:, x_col-1]
        y = data[:, y_col-1]
        pts.xy = zip(x, y)
        if v_col != 0:
            v = data[:, v_col-1]
        else:
            v = None
        pts.__init__(zip(x, y), v, unit_x, unit_y, unit_v)


def import_sf_from_ascii(filepath, x_col=1, y_col=2, vx_col=3,
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
    # check
    filepath = check_path(filepath)
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
    data = np.genfromtxt(filepath, **kwargs)
    # get axes
    x = data[:, x_col-1]
    x_org = np.sort(np.unique(x))
    y = data[:, y_col-1]
    y_org = np.sort(np.unique(y))
    vx = data[:, vx_col-1]

    # check if structured or not
    X1 = x.reshape(len(x_org), len(y_org))
    X2 = x.reshape(len(y_org), len(x_org))
    if np.allclose(np.mean(X1, axis=1), x_org):
        vx_org = vx.reshape(len(x_org), len(y_org))
        mask = np.isnan(vx_org)
    elif np.allclose(np.mean(X2, axis=0), x_org):
        vx_org = vx.reshape(len(y_org), len(x_org)).transpose()
        vx_org = np.fliplr(vx_org)
        mask = np.isnan(vx_org)
    else:
        # Masking all the initial fields (to handle missing values)
        vx_org = np.zeros((y_org.shape[0], x_org.shape[0]))
        vx_org_mask = np.ones(vx_org.shape)
        vx_org = np.ma.masked_array(vx_org, vx_org_mask)
        #loop on all 'v' values
        x_ind = 0
        y_ind = 0
        for i in np.arange(vx.shape[0]):
            x_tmp = x[i]
            y_tmp = y[i]
            vx_tmp = vx[i]
            #find x index
            if x_org[x_ind] != x_tmp:
                x_ind = np.where(x_tmp == x_org)[0][0]
            #find y index
            if y_org[y_ind] != y_tmp:
                y_ind = np.where(y_tmp == y_org)[0][0]
            #put the value at its place
            vx_org[y_ind, x_ind] = vx_tmp
        # Treating 'nan' values
        mask = np.logical_or(vx_org.mask, np.isnan(vx_org.data))

    #store field in attributes
    tmpsf = ScalarField()
    tmpsf.import_from_arrays(x_org, y_org, vx_org, mask=mask, unit_x=unit_x,
                             unit_y=unit_y, unit_values=unit_values)
    return tmpsf


def import_vf_from_ascii(filepath, x_col=1, y_col=2, vx_col=3,
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
    # check
    filepath = check_path(filepath)
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
    data = np.genfromtxt(filepath, **kwargs)
    # get axes
    x = data[:, x_col-1]
    x_org = np.unique(x)
    y = data[:, y_col-1]
    y_org = np.unique(y)
    vx = data[:, vx_col-1]
    vy = data[:, vy_col-1]
    # Masking all the initial fields (to handle missing values)
    vx_org = np.zeros((x_org.shape[0], y_org.shape[0]))
    vx_org_mask = np.ones(vx_org.shape)
    vx_org = np.ma.masked_array(vx_org, vx_org_mask)
    vy_org = np.zeros((x_org.shape[0], y_org.shape[0]))
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
        vx_org[x_ind, y_ind] = vx_tmp
        vy_org[x_ind, y_ind] = vy_tmp
    # Treating 'nan' values
    vx_org.mask = np.logical_or(vx_org.mask, np.isnan(vx_org.data))
    vy_org.mask = np.logical_or(vy_org.mask, np.isnan(vy_org.data))
    #store field in attributes
    tmpvf = VectorField()
    tmpvf.import_from_arrays(x_org, y_org, vx_org, vy_org,  mask=vx_org.mask,
                             unit_x=unit_x, unit_y=unit_y,
                             unit_values=unit_values)
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
    filepath = check_path(filepath)
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
        tmp_vf = VectorField()
        tmp_vf.import_from_ascii(path, x_col, y_col, vx_col, vy_col,
                                 unit_x, unit_y, unit_values, times[i],
                                 unit_time, **kwargs)
        fields.add_field(tmp_vf)
    return fields


def export_to_ascii(filepath, VF):
    """
    """
    # check
    filepath = check_path(filepath)
    if not isinstance(VF, VectorField):
        raise TypeError()
    f = open(filepath, 'w')
    for i, x in enumerate(VF.axe_x):
        for j, y in enumerate(VF.axe_y):
            f.write("{}\t{}\t{}\t{}\n".format(x, y, VF.comp_x[i, j],
                                            VF.comp_y[i, j]))
    f.close()

### VECTRINO ###
def import_from_VNO(filepath, add_info=True):
    """
    Import data from a VNO file. VNO files contains data from an ADV
    measurement.

    Parameters
    ----------
    filepath : string
        Path to the VNO file

    Returns
    -------
    Vx, Vy, Vz, Vz2 : Profile objects
        Velocity time profile along x, y, z,and z2 axis
    """
    # check filepath
    filepath = check_path(filepath)
    if add_info not in [True, False]:
        raise TypeError()
    # extract data from file
    data = np.genfromtxt(filepath)
    # check data shape
    if data.shape[1] != 20:
        print(data.shape)
        raise ValueError()
    # get profiles
    time = data[:, 1]
    Vx = data[:, 4]
    Vy = data[:, 5]
    Vz = data[:, 6]
    Vz2 = data[:, 7]
    Sx = data[:, 8]
    Sy = data[:, 9]
    Sz = data[:, 10]
    Sz2 = data[:, 11]
    SNRx = data[:, 12]
    SNRy = data[:, 13]
    SNRz = data[:, 14]
    SNRz2 = data[:, 15]
    Corrx = data[:, 16]
    Corry = data[:, 17]
    Corrz = data[:, 18]
    Corrz2 = data[:, 19]
    # print additional informations
    if add_info:
        print()
        print("+++ ADV Measurement informations +++")
        print("+++ Number of measure points : {}".format(len(time)))
        print("+++ Mean velocities :")
        print("+++    Vx  = {}".format(np.mean(Vx)))
        print("+++    Vy  = {}".format(np.mean(Vy)))
        print("+++    Vz  = {}".format(np.mean(Vz)))
        print("+++    Vz2 = {}".format(np.mean(Vz2)))
        print("+++ Mean velocities std :")
        print("+++    Vx_var  = {}".format(np.mean(np.abs(Vx - np.mean(Vx)))))
        print("+++    Vy_var  = {}".format(np.mean(np.abs(Vy - np.mean(Vy)))))
        print("+++    Vz_var  = {}".format(np.mean(np.abs(Vz - np.mean(Vz)))))
        print("+++    Vz2_var = {}".format(np.mean(np.abs(Vz2 - np.mean(Vz2)))))
        print("+++ Mean strength :")
        print("+++    S_x  = {:.0f}".format(np.mean(Sx)))
        print("+++    S_y  = {:.0f}".format(np.mean(Sy)))
        print("+++    S_z  = {:.0f}".format(np.mean(Sz)))
        print("+++    S_z2 = {:.0f}".format(np.mean(Sz2)))
        print("+++ Mean SNR :")
        print("+++    SNR_x  = {:.1f}".format(np.mean(SNRx)))
        print("+++    SNR_y  = {:.1f}".format(np.mean(SNRy)))
        print("+++    SNR_z  = {:.1f}".format(np.mean(SNRz)))
        print("+++    SNR_z2 = {:.1f}".format(np.mean(SNRz2)))
        print("+++ Mean Correlation :")
        print("+++    Corr_x  = {:.1f}".format(np.mean(Corrx)))
        print("+++    Corr_y  = {:.1f}".format(np.mean(Corry)))
        print("+++    Corr_z  = {:.1f}".format(np.mean(Corrz)))
        print("+++    Corr_z2 = {:.1f}".format(np.mean(Corrz2)))
    # print warnings if measure quality is not good enough
    if np.any(np.array([np.mean(Sx), np.mean(Sy), np.mean(Sz), np.mean(Sz2)])
              < 100):
        print("+++ Warning : low signal strength")
    if np.any(np.array([np.mean(SNRx), np.mean(SNRy), np.mean(SNRz),
                        np.mean(SNRz2)]) < 13):
        print("+++ Warning : low SNR ratio")
    if np.any(np.array([np.mean(Corrx), np.mean(Corry), np.mean(Corrz),
                        np.mean(Corrz2)]) < 80):
        print("+++ Warning : low signal correlation")
    # create profiles
    Vx_prof = Profile(time, Vx, mask=False, unit_x="s", unit_y="m/s")
    Vy_prof = Profile(time, Vy, mask=False, unit_x="s", unit_y="m/s")
    Vz_prof = Profile(time, Vz, mask=False, unit_x="s", unit_y="m/s")
    Vz2_prof = Profile(time, Vz2, mask=False, unit_x="s", unit_y="m/s")
    # returning
    return Vx_prof, Vy_prof, Vz_prof, Vz2_prof

