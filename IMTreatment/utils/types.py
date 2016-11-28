# -*- coding: utf-8 -*-
import numpy as np
from functools import wraps
import re

ARRAYTYPES = (np.ndarray, list, tuple)
INTEGERTYPES = (int, np.int, np.int16, np.int32, np.int64, np.int8)
NUMBERTYPES = (int, float, complex, np.float, np.float16, np.float32,
               np.float64) + INTEGERTYPES
STRINGTYPES = (str, str)


class TypeTest(object):
    enabled = True

    def __init__(self, *arg_types, **kwargs_types):
        self.arg_types = arg_types
        self.kwargs_types = kwargs_types
        self.w_vartypes = {}
        self.is_method = False

    def __call__(self, function):
        if not self.enabled:
            return function
        self.extract_var_info(function)
        return self.decorator(function)

    def extract_var_info(self, function):
        nmb_var = function.__code__.co_argcount
        varnames = function.__code__.co_varnames
        if varnames[0] == 'self':
            self.is_method = True
        if self.is_method:
            varnames = varnames[1::]
        varnames = varnames[0:nmb_var]
        w_vartypes = {}
        for i in range(len(varnames)):
            var = varnames[i]
            if var in list(self.kwargs_types.keys()):
                w_vartypes[var] = self.kwargs_types[var]
            elif i < len(self.arg_types):
                w_vartypes[var] = self.arg_types[i]
            else:
                pass
        self.varnames = varnames
        self.w_vartypes = w_vartypes

    def decorator(self, function):
        @wraps(function)
        def new_function(*args, **kwargs):
            # get given argument types
            given_vartypes = {}
            if self.is_method:
                tmp_args = args[1::]
            else:
                tmp_args = args
            for i in range(len(tmp_args)):
                varname = self.varnames[i]
                given_vartypes[varname] = type(tmp_args[i])
            for varname in list(kwargs.keys()):
                given_vartypes[varname] = type(kwargs[varname])
            # check args types
            for varname in list(given_vartypes.keys()):
                if varname not in list(self.w_vartypes.keys()):
                    continue
                w_type = self.w_vartypes[varname]
                g_type = given_vartypes[varname]
                try:
                    ok = g_type in w_type
                except TypeError:
                    ok = g_type == w_type
                if not ok:
                    w_type_txt = re.findall("'.+'", str(w_type))[0]
                    g_type_txt = re.findall("'.+'", str(g_type))[0]
                    text = ("'{}' should be {}, not {}."
                            .format(varname, w_type_txt, g_type_txt))
                    raise TypeError(text)
            return function(*args, **kwargs)
        return new_function


class ReturnTest(object):
    enabled = True

    def __init__(self, typ):
        self.typ = typ

    def __call__(self, function):
        if not self.enabled:
            return function
        return self.decorator(function)

    def decorator(self, function):
        @wraps(function)
        def new_function(*args, **kwargs):
            res = function(*args, **kwargs)
            if not type(res) == self.typ:
                w_type_txt = re.findall("'.+'", str(self.typ))[0]
                g_type_txt = re.findall("'.+'", str(type(res)))[0]
                text = ("Returned value should be {}, not {}."
                        .format(w_type_txt, g_type_txt))
                raise TypeError(text)
            return res
        return new_function