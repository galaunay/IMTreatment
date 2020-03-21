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

import gc
import gzip
import os
from glob import glob
import warnings
try:
    import cv2
    IS_CV2 = True
except ImportError:
    IS_CV2 = False
try:
    import h5py
    IS_H5PY = True
except ImportError:
    IS_H5PY = False


import numpy as np

from ..core import (Points, Profile, ScalarField, SpatialScalarFields,
                    SpatialVectorFields, TemporalScalarFields,
                    TemporalVectorFields,
                    VectorField)
from ..vortex_detection import CritPoints
from ..utils import ProgressCounter, make_unit
from ..utils.types import ARRAYTYPES, STRINGTYPES


try:
    import pickle as pickle
except:
    import pickle
import scipy.io as spio
import imageio
from os import path
import re


def check_path(filepath, newfile=False):
    """
    Normalize and check the validity of the given path to feed importation
    functions.
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
                if p != "":
                    path_compos.append(p)
                break
        # check validity recursively
        valid_path = ""
        while True:
            if len(path_compos) == 0:
                break
            new_dir = path_compos.pop()
            if new_dir == "":
                break
            new_tested_path = path.join(valid_path, new_dir)
            if not path.exists(new_tested_path):
                if valid_path == "":
                    valid_path = os.getcwd()
                err_mess = r"'{}' directory/file not found in '{}' directory." \
                           .format(new_dir, valid_path)
                raise FileNotFoundError(err_mess)
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
    if not path.isdir(dirpath):
        raise ValueError()
    if not isinstance(regs, ARRAYTYPES):
        raise TypeError()
    regs = np.array(regs, dtype=str)
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
    if ask and len(dir_paths) > 1:
        print(("{} folders found :".format(len(dir_paths))))
        for i, p in enumerate(dir_paths):
            print(("  {} :  {}".format(i + 1, p)))
        rep = 0
        while rep not in np.arange(1, len(dir_paths)+1):
            rep = input("Want to go with wich one ?\n")
            try:
                rep = int(rep)
            except:
                pass
        dir_paths = [dir_paths[rep - 1]]
    # return
    return dir_paths


def matlab_parser(obj, name):
    classic_types = (int, float, str, bool)
    array_types = (list, float)
    if obj is None:
        return {name: "None"}
    elif isinstance(obj, classic_types):
        return {name: obj}
    elif isinstance(obj, array_types):
        simple = True
        for val in obj:
            if not isinstance(val, (classic_types, array_types)):
                simple = False
        if simple:
            return {name: obj}
        else:
            return {name: [matlab_parser(obj[i], f"obj{i}")
                           for i in range(len(obj))]}
            raise IOError("Matlab can't handle this kind of variable")
    elif isinstance(obj, Points):
        xy = obj.xy
        xy = [list(comp) for comp in xy]
        dic = matlab_parser(list(obj.v), 'v')
        dic.update(matlab_parser(xy, 'xy'))
        dic.update(matlab_parser(obj.unit_x.strUnit()[1:-1], 'unit_x'))
        dic.update(matlab_parser(obj.unit_y.strUnit()[1:-1], 'unit_y'))
        dic.update(matlab_parser(obj.unit_v.strUnit()[1:-1], 'unit_v'))
        return {name: dic}
    elif isinstance(obj, Profile):
        mask = [int(val) for val in obj.mask]
        dic = matlab_parser(list(obj.x), 'x')
        dic.update(matlab_parser(list(obj.y), 'y'))
        dic.update(matlab_parser(mask, 'mask'))
        dic.update(matlab_parser(obj.unit_x.strUnit()[1:-1], 'unit_x'))
        dic.update(matlab_parser(obj.unit_y.strUnit()[1:-1], 'unit_y'))
        return {name: dic}
    elif isinstance(obj, VectorField):
        x = obj.axe_x
        y = obj.axe_y
        X, Y = np.meshgrid(x, y, indexing='ij')
        X = [list(comp) for comp in X]
        Y = [list(comp) for comp in Y]
        comp_x = obj.comp_x
        comp_x = [list(comp) for comp in comp_x]
        comp_y = obj.comp_y
        comp_y = [list(comp) for comp in comp_y]
        mask = obj.mask
        mask = [list(comp) for comp in mask]
        dic = matlab_parser(list(obj.axe_x), 'axe_x')
        dic.update(matlab_parser(list(obj.axe_y), 'axe_y'))
        dic.update(matlab_parser(list(X), 'X'))
        dic.update(matlab_parser(list(Y), 'Y'))
        dic.update(matlab_parser(comp_x, 'comp_x'))
        dic.update(matlab_parser(comp_y, 'comp_y'))
        dic.update(matlab_parser(mask, 'mask'))
        dic.update(matlab_parser(obj.unit_x.strUnit()[1:-1], 'unit_x'))
        dic.update(matlab_parser(obj.unit_y.strUnit()[1:-1], 'unit_y'))
        dic.update(matlab_parser(obj.unit_values.strUnit()[1:-1],
                                 'unit_values'))
        return {name: dic}
    elif isinstance(obj, ScalarField):
        x = obj.axe_x
        y = obj.axe_y
        X, Y = np.meshgrid(x, y, indexing='ij')
        X = [list(comp) for comp in X]
        Y = [list(comp) for comp in Y]
        values = obj.values
        values = [list(comp) for comp in values]
        mask = obj.mask
        mask = [list(comp) for comp in mask]
        dic = matlab_parser(list(obj.axe_x), 'axe_x')
        dic.update(matlab_parser(list(obj.axe_y), 'axe_y'))
        dic.update(matlab_parser(list(X), 'X'))
        dic.update(matlab_parser(list(Y), 'Y'))
        dic.update(matlab_parser(values, 'values'))
        dic.update(matlab_parser(mask, 'mask'))
        dic.update(matlab_parser(obj.unit_x.strUnit()[1:-1], 'unit_x'))
        dic.update(matlab_parser(obj.unit_y.strUnit()[1:-1], 'unit_y'))
        dic.update(matlab_parser(obj.unit_values.strUnit()[1:-1],
                                 'unit_values'))
        return {name: dic}
    elif isinstance(obj, CritPoints):
        eps = obj.current_epsilon
        if eps is not None:
            foc_traj = list(obj.foc_traj)
            foc_c_traj = list(obj.foc_c_traj)
            node_i_traj = list(obj.node_i_traj)
            node_o_traj = list(obj.node_o_traj)
            sadd_traj = list(obj.sadd_traj)
        else:
            foc_traj = None
            foc_c_traj = None
            node_i_traj = None
            node_o_traj = None
            sadd_traj = None
        foc = list(obj.foc)
        foc_c = list(obj.foc_c)
        node_i = list(obj.node_i)
        node_o = list(obj.node_o)
        sadd = list(obj.sadd)
        times = list(obj.times)
        unit_time = obj.unit_time.strUnit()
        unit_x = obj.unit_x.strUnit()
        unit_y = obj.unit_y.strUnit()
        dic = matlab_parser(eps, "epsilon")
        dic.update(matlab_parser(foc, "foc"))
        dic.update(matlab_parser(foc_traj, "foc_traj"))
        dic.update(matlab_parser(foc_c, "foc_c"))
        dic.update(matlab_parser(foc_c_traj, "foc_c_traj"))
        dic.update(matlab_parser(node_i, "node_i"))
        dic.update(matlab_parser(node_o, "node_o"))
        dic.update(matlab_parser(node_i_traj, "node_i_traj"))
        dic.update(matlab_parser(node_o_traj, "node_o_traj"))
        dic.update(matlab_parser(sadd, "sadd"))
        dic.update(matlab_parser(sadd_traj, "sadd_traj"))
        dic.update(matlab_parser(times, "times"))
        dic.update(matlab_parser(unit_time, "unit_time"))
        dic.update(matlab_parser(unit_x, "unit_x"))
        dic.update(matlab_parser(unit_y, "unit_y"))
        return {name: dic}
    else:
        raise IOError("Can't parser that : \n {}".format(obj))


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
    full_import : boolean
        If 'True', everything is charged in memory, else, data are loaded in
        memory when they are needed.
    """
    # getting/guessing wanted files
    filepath = check_path(filepath)
    extension = path.splitext(filepath)[1]
    # importing file
    if extension == ".imt":
        gc.disable()
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        gc.enable()
    elif extension == ".cimt":
        gc.disable()
        with gzip.open(filepath, 'rb') as f:
            obj = pickle.load(f)
        gc.enable()
    else:
        raise IOError("File is not readable "
                      "(unknown extension : {})".format(extension))
    # For backward support
    attr_name = f"_values_dtype"
    if isinstance(obj, (VectorField, ScalarField)):
        if not hasattr(obj, attr_name):
            setattr(obj, attr_name, float)
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
        # creating/filling up the file
        raise TypeError("'compressed' must be a boolean")
    if compressed:
        gc.disable()
        if path.splitext(filepath)[1] != ".cimt":
            filepath = filepath + ".cimt"
        f = gzip.open(filepath, 'wb')
        pickle.dump(obj, f, protocol=-1)
        f.close()
        gc.enable()
    else:
        gc.disable()
        if path.splitext(filepath)[1] != ".imt":
            filepath = filepath + ".imt"
        f = open(filepath, 'wb')
        pickle.dump(obj, f, protocol=-1)
        f.close()
        gc.enable()


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
    elif kind == 'TVF':
        imts_type = 'VF'
        fields = TemporalVectorFields()
    elif kind == 'SVF':
        imts_type = 'VF'
        fields = SpatialVectorFields()
    else:
        raise ValueError()
    # importing data
    for tmp_path in paths:
        basename = path.basename(tmp_path)
        name, ext = path.splitext(basename)
        if ext in ['.imt', '.cimt']:
            field = import_from_file(tmp_path)
            if imts_type == 'SF' and not isinstance(field, ScalarField):
                continue
            elif imts_type == 'VF' and not isinstance(field, VectorField):
                continue
            fields.add_field(field)
    # saving data
    export_to_file(fields, imt_path)


def import_from_matlab(filepath, obj, show_struct=False, **kwargs):
    """
    Import data from a matlab (.m) file.

    Data should be a dictionnary.

    Parameters
    ----------
    filepath : string
        Path of the matlab file
    obj : string in ['ScalarField', 'VectorField', 'Profile',
                     'Points']
        Kind of object to import to.
    show_struct: boolean
       If True, just show the structure of the file
       (and return it, so you can play with it)
    kwargs :
       Rest of the keyword arguments should indicate where to find
       the necessary information in the matlab dictionnary.

    Example
    -------
    With a matlab file containing a dictionnary with the
    following entries: 'x', 'x_unit', 'y', 'y_unit', 'u', 'v'
    >>> vf = import_from_matlab('data.m', 'VectorField',
    ...                         axe_x='x', axe_y='y',
    ...                         unit_x='x_unit', unit_y='y_unit',
    ...                         comp_x='u', comp_y='v')
    """
    if not IS_H5PY:
        raise Exception('h5py needs to be installed to import from matlab.')
    filepath = check_path(filepath)
    # Get matlab dictionnary
    try:
        data = spio.loadmat(filepath)
        is_h5py = False
    except NotImplementedError:
        data = h5py.File(filepath, 'r')
        is_h5py = True
    if isinstance(data, np.ndarray):
        keys = list(data.dtype.fields.keys())
    else:
        keys = list(data.keys())
    # pivmat !
    dataf = None
    if 'Data' in keys and 'P' in keys:
        dataf = data
        data = dataf['Data']
        if isinstance(data, np.ndarray):
            keys = list(data.dtype.fields.keys())
        else:
            keys = list(data.keys())
    # Create object
    if obj == 'ScalarField':
        res = ScalarField
        default_dic = {'axe_x': 'axe_x', 'axe_y': 'axe_y', 'unit_x': 'unit_x',
                       'unit_y': 'unit_y', 'values': 'values', 'mask': 'mask',
                       'unit_values': 'unit_values'}
    elif obj == 'VectorField':
        res = VectorField
        default_dic = {'axe_x': 'axe_x', 'axe_y': 'axe_y', 'comp_x': 'comp_x',
                       'comp_y': 'comp_y', 'mask': 'mask', 'unit_x': 'unit_x',
                       'unit_y': 'unit_y', 'unit_values': 'unit_values'}
    elif obj == 'Profile':
        res = Profile
        default_dic = {'x': 'x', 'y': 'y', 'mask': 'mask', 'unit_x': 'unit_x',
                       'unit_y': 'unit_y'}
    elif obj == 'Points':
        res = Points
        default_dic = {'xy': 'xy', 'v': 'v', 'unit_x': 'unit_x',
                       'unit_y': 'unit_y', 'unit_v': 'unit_v'}
    else:
        raise Exception()
    default_dic.update(kwargs)
    kwargs = default_dic
    # Just show the structure
    if show_struct:
        if is_h5py:
            print(f"Available keys in {filepath}:\n{keys})")
        else:
            print(f"Available keys in {filepath}:"
                  f"\n{keys}")
        return data
    # Create building dict
    build_dic = {}
    for key, entry in kwargs.items():
        if entry in keys:
            tdata = data[entry]
            while not isinstance(tdata, str) and len(tdata) == 1:
                tdata = tdata[0]
            print(tdata)
            if key in ['unit_x', 'unit_y', 'unit_values']:
                if isinstance(tdata, str):
                    build_dic[key] = tdata
                else:
                    build_dic[key] = ''.join(chr(n) for n in tdata)
            else:
                build_dic[key] = np.array(tdata)
    if is_h5py and dataf is not None:
        dataf.close()
    # Fix unities
    for key, item in build_dic.items():
        try:
            if len(item) == 1:
                build_dic[key] = build_dic[key][0]
        except TypeError:
            pass
    # Fill the object
    if obj in ['ScalarField', 'VectorField']:
        ress = res()
        ress.import_from_arrays(**build_dic)
    else:
        ress = res(**build_dic)
    # Return
    return ress


def export_to_matlab(obj, filepath, **kw):
    filepath = check_path(filepath, newfile=True)
    dic = matlab_parser(obj, "tmp")
    spio.savemat(filepath, dic["tmp"], **kw)


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
    try:
        import pyvtk
    except ImportError:
        raise Exception("You need to install pyvtk to use this "
                        "functionnality")
    if not path.exists(path.dirname(filepath)):
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
    pts = list(zip(x_vtk, y_vtk, z_vtk))
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
    if not path.exists(path.dirname(filepath)):
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
    if not path.exists(path.dirname(filepath)):
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
    vect = list(zip(vx_vtk, vy_vtk, vz_vtk))
    point_data = pyvtk.PointData(pyvtk.Vectors(vect, "Vector field"))
    grid = pyvtk.RectilinearGrid(x_vtk, y_vtk, z_vtk)
    data = pyvtk.VtkData(grid, 'Vector Field from python', point_data)
    data.tofile(filepath)


def _get_imx_buffers(filename):
    """
    Return the buffers stored in the given file.
    """
    try:
        import ReadIM
    except ModuleNotFoundError:
        raise Exception("You need the 'ReadIM' module to import from davis"
                        " files")
    vbuff, vatts = ReadIM.extra.get_Buffer_andAttributeList(filename)
    arrays, vbuff2 = ReadIM.extra.buffer_as_array(vbuff)
    arrays = np.array(arrays.transpose((0, 2, 1)))
    atts = ReadIM.extra.att2dict(vatts)
    fmt = vbuff.image_sub_type
    vectorGrid = vbuff.vectorGrid
    ReadIM.DestroyBuffer(vbuff)
    ReadIM.DestroyBuffer(vbuff2)
    return fmt, vectorGrid, arrays, atts


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
    if not path.exists(filename):
        raise ValueError("I did not find your file, boy")
    _, ext = path.splitext(filename)
    if not (ext == ".im7" or ext == ".IM7"):
        raise ValueError("I need the file to be an IM7 file (not a {} file)"
                         .format(ext))
    # Importing from buffer
    fmt, vectorGrid, v_array, atts = _get_imx_buffers(filename)
    if v_array.shape[0] == 2:
        mask = v_array[0][:, ::1]
        values = v_array[1][:, ::1]
    elif v_array.shape[0] == 1:
        values = v_array[0][:, ::1]
        mask = np.zeros(values.shape, dtype=bool)
    # Values and Mask
    scale_i = atts['_SCALE_I']
    scale_i = scale_i.split("\n")
    scale_val = scale_i[0].split(' ')
    unit_values = scale_i[1]
    try:
        values *= int(scale_val[0])
        values += int(scale_val[1])
    except ValueError:
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
        fieldspaths = np.asarray(fieldspath, dtype=str)
    elif isinstance(fieldspath, STRINGTYPES):
        fieldspath = check_path(fieldspath)
        paths = np.asarray([f for f in glob(path.join(fieldspath, '*'))
                            if path.splitext(f)[-1] in ['.im7', '.IM7']],
                           dtype=str)
        # if no file found, search recursively
        if len(paths) == 0:
            poss_paths = find_file_in_path(['.*.im7', '.*.IM7'], fieldspath,
                                           ask=True)
            if len(poss_paths) == 0:
                raise ValueError()
            paths = np.asarray([f
                                for f in glob(path.join(poss_paths[0], '*'))
                                if path.splitext(f)[-1] in ['.im7', '.IM7']],
                               dtype=str)
        # Sort path by numbers
        filenames = [path.basename(p) for p in paths]
        ind_sort = np.argsort(filenames)
        fieldspaths = paths[ind_sort]
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
    _, ext = path.splitext(filename)
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
    try:
        Vx *= int(scale_val[0])
        Vx += int(scale_val[1])
        Vy *= int(scale_val[0])
        Vy += int(scale_val[1])
    except ValueError:
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
        paths = np.array([f for f in glob(path.join(fieldspath, '*'))
                          if path.splitext(f)[-1] in ['.vc7', '.VC7']])
        # if no file found, search recursively
        if len(paths) == 0:
            poss_paths = find_file_in_path(['.*.vc7', '.*.VC7'], fieldspath,
                                           ask=True)
            if len(poss_paths) == 0:
                raise ValueError()
            paths = np.array([f for f in glob(path.join(poss_paths[0], '*'))
                              if path.splitext(f)[-1] in ['.vc7', '.VC7']])
        # Sort path by numbers
        filenames = [path.basename(p) for p in paths]
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
    pc = ProgressCounter(init_mess="Begin importation of {} VC7 files"
                         .format(nmb_files),
                         nmb_max=nmb_files,
                         name_things="VC7 files",
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
            try:
                dt = infos['FrameDt0'].split()
                unit_time = make_unit(dt[1])
                dt = float(dt[0])
            except KeyError:
                dt = 1.
                unit_time = make_unit("")
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
    if path.isdir(im7_path):
        if kind in ['SSF', 'TSF']:
            ST_SF = import_from_IM7s(im7_path, kind=kind, **kwargs)
            export_to_file(ST_SF, imt_path)
        elif kind in ['SF']:
            paths = glob(im7_path + "/*")
            for tmp_path in paths:
                name_ext = path.basename(tmp_path)
                name, ext = path.splitext(name_ext)
                if ext not in ['.im7', '.IM7']:
                    continue
                SF = import_from_IM7(tmp_path)
                export_to_file(SF, imt_path + "/{}".format(name),
                               compressed=compressed)
    elif path.isfile(im7_path):
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
    if path.isdir(vc7_path):
        if kind in ['SVF', 'TVF']:
            ST_VF = import_from_VC7s(vc7_path, kind=kind, **kwargs)
            export_to_file(ST_VF, imt_path)
        elif kind in ['VF']:
            paths = glob(vc7_path + "/*")
            for tmp_path in paths:
                name_ext = path.basename(tmp_path)
                name, ext = path.splitext(name_ext)
                if ext not in ['.vc7', '.VC7']:
                    continue
                VF = import_from_VC7(tmp_path)
                export_to_file(VF, imt_path + "/{}".format(name),
                               compressed=compressed)
    elif path.isfile(vc7_path):
        SF = import_from_IM7(vc7_path, **kwargs)
        export_to_file(SF, imt_path)
    else:
        raise ValueError()

def import_from_picture(filepath, axe_x=None, axe_y=None, unit_x='', unit_y='',
                        unit_values='', dtype=float):
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
    dtype :
        Type of the stored values (default to float)

    Returns
    -------
    tmp_sf :
        .
    """
    usable_ext = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp',
                  '.BMP']
    filepath = check_path(filepath)
    _, ext = path.splitext(filepath)
    if ext not in usable_ext:
        raise ValueError("I need the file to be an supported picture file"
                         "(not a {} file)".format(ext))
    # importing from file
    values = imageio.imread(filepath, as_gray=True).transpose()[:, ::-1]
    values = np.asarray(values, dtype=dtype)
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
                              unit_y=unit_y, unit_values=unit_values,
                              dtype=dtype)
    # return
    return tmp_sf

def import_from_pictures(filepath, axe_x=None, axe_y=None, unit_x='',
                         unit_y='', unit_values='', times=None,
                         unit_times='', dtype=float, fieldnumbers=None, incr=1,
                         verbose=False):
    """
    Import scalar fields from a bunch of picture files.

    Parameters
    ----------
    filepath : string
        regex matching the files.
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
    dtype :
        Type of the stored values (default to float)
    fieldnumbers: 2x1 array
        Interval of fields to import, default is all.
    incr : integer
        Increment (incr=2 will load only 1 picture over 2).

    Returns
    -------
    tmp_sf :
        .
    """
    # get paths
    # filepath = check_path(filepath)
    paths = glob(filepath)
    paths = sorted(paths)
    tmp_tsf = TemporalScalarFields()
    # check times
    if times is None:
        times = np.arange(len(paths))
    elif len(times) != len(paths):
        raise ValueError()
    # filter by field number
    if fieldnumbers is None:
        start = 0
        end = len(paths)
    else:
        if fieldnumbers[0] < 0:
            raise ValueError()
        if fieldnumbers[1] > len(paths):
            fieldnumbers[1] = len(paths)
        start = fieldnumbers[0]
        end = fieldnumbers[1]
    if verbose:
        pg = ProgressCounter(init_mess="Importing pictures",
                             nmb_max=int((end - start + 1)/incr),
                             name_things="pictures")
    # loop on paths
    for i in np.arange(start, end, incr):
        tmp_sf = import_from_picture(paths[i], axe_x=axe_x, axe_y=axe_y,
                                     unit_x=unit_x, unit_y=unit_y,
                                     unit_values=unit_values,
                                     dtype=dtype)
        tmp_tsf.add_field(tmp_sf, times[i], unit_times=unit_times)
        if verbose:
            pg.print_progress()
    # returning
    return tmp_tsf


def import_from_video(filepath, dx=1, dy=1, unit_x='',
                      unit_y='', unit_values='', dt=None,
                      unit_times='', dtype=float, frame_inds=None, incr=1,
                      verbose=False):
    """
    Import scalar fields from a video.

    Parameters
    ----------
    filepath : string
        regex matching the files.
    dx :
        .
    dy :
        .
    unit_x :
        .
    unit_y :
        .
    unit_values :
        .
    dt: number
        Time interval between two frames. If not specified, is estimated
        from the video.
    dtype :
        Type of the stored values (default to float)
    frame_inds: 2x1 array
        Interval of fields to import, default is all.
    incr : integer
        Increment (incr=2 will load only 1 picture over 2).

    Returns
    -------
    tmp_sf :
        .
    """
    # Chheck if cv2 is available
    if not IS_CV2:
        raise Exception('This feature needs opencv to be installed')
    # Check file
    filepath = check_path(filepath)
    paths = glob(filepath)
    paths = sorted(paths)
    # Open video stream
    vid = cv2.VideoCapture()
    vid.open(filepath)
    max_frame = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get framerate
    if dt is None:
        fps = float(vid.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 1
        dt = 1/fps
    # filter by field number
    if frame_inds is None:
        frame_inds = [0, max_frame]
    if frame_inds[1] > max_frame:
        frame_inds[1] = max_frame
    if verbose:
        start = frame_inds[0]
        end = frame_inds[1]
        pg = ProgressCounter(init_mess="Importing video",
                             nmb_max=int((end - start + 1)/incr),
                             name_things="frames")
    # loop on paths
    tsf = TemporalScalarFields()
    t = 0
    for i in np.arange(0, frame_inds[1], 1):
        if i < frame_inds[0] or i % incr != 0:
            t += dt
            vid.grab()
            continue
        success, im = vid.read()
        if not success:
            if frame_inds[1] != np.inf:
                warnings.warn(f"Can't decode frame number {i}")
            break
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        im = im.transpose()[:, ::-1]
        axe_x = np.arange(0, im.shape[0]*dx - 0.1*dx, dx)
        axe_y = np.arange(0, im.shape[1]*dy - 0.1*dy, dy)
        sf = ScalarField()
        sf.import_from_arrays(axe_x, axe_y, im, mask=False,
                              unit_x=unit_x, unit_y=unit_y,
                              dtype=dtype)
        tsf.add_field(sf, time=t, unit_times=unit_times, copy=False)
        t += dt
        if verbose:
            pg.print_progress()
    if verbose:
        pg.finish()
    # returning
    return tsf


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
    imageio.imwrite(filepath, values)


def export_to_pictures(SFs, filepath):
    """
    Export scalar fields to a picture file.

    Parameters
    ----------
    SF :
        .
    filename : string
        Path to the picture file. Should include a name for the image
        (without the extension).
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
        imageio.imwrite("{}_{:0>5}.png".format(filepath, i), val)


def export_to_video(SFs, filepath, fps=24, colormap=None):
    """
    Export scalar fields to a video file.

    Parameters
    ----------
    SF :
        .
    filename : string
        Path to the video file.
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
    values = np.array(values)
    # normalize between 0 and 255
    maxi = np.max(values)
    mini = np.min(values)
    values = (values - mini)/(maxi - mini)
    # save as a video using opencv
    vid = cv2.VideoWriter(filename=filepath,
                          fourcc=cv2.VideoWriter_fourcc(*"MJPG"),
                          fps=fps,
                          frameSize=(values[0].shape[1],
                                     values[0].shape[0]))
    for val in values:
        if colormap is None:
            tmpval = cv2.cvtColor((val*255).astype('uint8'),
                                  cv2.COLOR_GRAY2RGB)
        else:
            tmpval = np.array(colormap(val)[:, :, 0:3]*255,
                              dtype=np.uint8)
            tmpval = tmpval[:, :, ::-1]
        # # TEMP
        # tmpval2 = cv2.cvtColor((val*255).astype('uint8'),
        #                        cv2.COLOR_GRAY2RGB)
        # tmpval[:, 0:int(tmpval.shape[1]/2), :] \
        #     = tmpval2[:, 0:int(tmpval.shape[1]/2), :]
        # # TEMP - End
        vid.write(tmpval)
    vid.release()

def import_profile_from_ascii(filepath, x_col=1, y_col=2,
                              unit_x=make_unit(""), unit_y=make_unit(""),
                              **kwargs):
        """
        Import a Profile object from an ascii file.

        Parameters
        ----------
        x_col, y_col : integer, optional
            Colonne numbers for the given variables (begining at 1).
        unit_x, unit_y : Unit objects, optional
            Unities for the given variables.
        **kwargs :
            Possibles additional parameters are the same as those used in the
            numpy function 'genfromtext()' :
            'delimiter' to specify the delimiter between colonnes.
            'skip_header' to specify the number of colonne to skip at file
                begining
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


def import_pts_from_ascii(filepath, x_col=1, y_col=2, v_col=None,
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
        # check
        filepath = check_path(filepath)
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
        data = np.genfromtxt(filepath, **kwargs)
        # get axes
        x = data[:, x_col-1]
        y = data[:, y_col-1]
        if v_col != 0:
            v = data[:, v_col-1]
        else:
            v = None
        return Points(list(zip(x, y)), v, unit_x, unit_y, unit_v)


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
    tmpvf.import_from_arrays(x_org, y_org, vx_org, vy_org,
                             mask=vx_org.mask,
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
    Import velocityfields from an ascii file.

    Parameters
    ----------
    filepath : string
        Pathname pattern to the ascii files.
    incr : integer, optional
        Increment value between two fields taken.
    interval : 2x1 array, optional
        Interval in which take fields.
    x_col, y_col, vx_col, vy_col : integer, optional
        Colonne numbers for the given variables (begining at 1).
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

    Note
    ----
    txt files are taken in alpha-numerical order
    ('file2.txt' is taken before 'file20.txt').
    So you should name your files properly.

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


def export_to_ascii(obj, filepath):
    """
    """
    # check
    filepath = check_path(filepath, newfile=True)
    # obj type
    if isinstance(obj, VectorField):
        # open file
        f = open(filepath, 'w')
        # write header
        header = "# X {}\tY {}\tVx {}\tVy {}\n"\
                 .format(obj.unit_x.strUnit(),
                         obj.unit_y.strUnit(),
                         obj.unit_values.strUnit(),
                         obj.unit_values.strUnit())
        f.write(header)
        # write data
        for i, x in enumerate(obj.axe_x):
            for j, y in enumerate(obj.axe_y):
                f.write("{}\t{}\t{}\t{}\n".format(x, y, obj.comp_x[i, j],
                                                  obj.comp_y[i, j]))
        f.close()
    elif isinstance(obj, ScalarField):
        # open file
        f = open(filepath, 'w')
        # write header
        header = "# X {}\tY {}\tValue {}\n"\
                 .format(obj.unit_x.strUnit(),
                         obj.unit_y.strUnit(),
                         obj.unit_values.strUnit())
        f.write(header)
        # write data
        for i, x in enumerate(obj.axe_x):
            for j, y in enumerate(obj.axe_y):
                f.write("{}\t{}\t{}\n".format(x, y, obj.values[i, j]))
        f.close()
    elif isinstance(obj, Profile):
        # open file
        f = open(filepath, 'w')
        # write header
        header = "# X {}\tY {}\n"\
                 .format(obj.unit_x.strUnit(),
                         obj.unit_y.strUnit())
        f.write(header)
        # write data
        for x, y in zip(obj.x, obj.y):
            f.write("{}\t{}\n".format(x, y))
        f.close()
    elif isinstance(obj, Points):
        # open file
        f = open(filepath, 'w')
        if len(obj.v) == 0:
            # write header
            header = "# X {}\tY {}\n"\
                     .format(obj.unit_x.strUnit(),
                             obj.unit_y.strUnit())
            f.write(header)
            # write data
            for x, y in zip(obj.xy[:, 0], obj.xy[:, 1]):
                f.write("{}\t{}\n".format(x, y))
        else:
            # write header
            header = "# X {}\tY {}\tV {}\n"\
                     .format(obj.unit_x.strUnit(),
                             obj.unit_y.strUnit(),
                             obj.unit_v.strUnit())
            f.write(header)
            # write data
            for x, y, v in zip(obj.xy[:, 0], obj.xy[:, 1], obj.v):
                f.write("{}\t{}\t{}\n".format(x, y, v))
        f.close()
    elif isinstance(obj, CritPoints):
        # get info from path
        basename = os.path.splitext(filepath)[0]
        try:
            os.mkdir(basename)
        except FileExistsError:
            pass
        t_unit = obj.unit_time.strUnit()
        if t_unit == "[]":
            t_unit = ""
        for typ in obj.cp_types:
            pts = obj.__getattribute__(typ)
            for t, pt in zip(obj.times, pts):
                export_to_ascii(pt, "{}/{}_t={}{}.txt".format(basename,
                                                              typ, t, t_unit))
    else:
        raise TypeError()
