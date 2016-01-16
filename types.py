# -*- coding: utf-8 -*-
import numpy as np

ARRAYTYPES = (np.ndarray, list, tuple)
INTEGERTYPES = (int, np.int, np.int16, np.int32, np.int64, np.int8)
NUMBERTYPES = (long, float, complex, np.float, np.float16, np.float32,
               np.float64) + INTEGERTYPES
STRINGTYPES = (str, unicode)


class TypeTest(object):

    def __init__(self, *arg_types, **kwargs_types):
        self.arg_types = arg_types
        self.kwargs_types = kwargs_types
        self.w_vartypes = {}
        self.is_method = False

    def __call__(self, function):
        self.extract_var_info(function)
        return self.decorator(function)

    def extract_var_info(self, function):
        nmb_var = function.func_code.co_argcount
        varnames = function.func_code.co_varnames
        if varnames[0] == 'self':
            self.is_method = True
        if self.is_method:
            varnames = varnames[1::]
        varnames = varnames[0:nmb_var]
        w_vartypes = {}
        for i in range(len(varnames)):
            var = varnames[i]
            if var in self.kwargs_types.keys():
                w_vartypes[var] = self.kwargs_types[var]
            else:
                w_vartypes[var] = self.arg_types[i]
        self.varnames = varnames
        self.w_vartypes = w_vartypes

    def decorator(self, function):
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
            for varname in kwargs.keys():
                given_vartypes[varname] = type(kwargs[varname])
            # check args types
            for varname in given_vartypes.keys():
                w_type = self.w_vartypes[varname]
                g_type = given_vartypes[varname]
                try:
                    ok = g_type in w_type
                except TypeError:
                    ok = g_type == w_type
                if not ok:
                    text = ("'{}' should be {}, not {}"
                            .format(varname, w_type, g_type))
                    raise TypeError(text)
            return function(*args, **kwargs)
        new_function.__doc__ = function.__doc__
        new_function.__name__ = function.__name__
        return new_function