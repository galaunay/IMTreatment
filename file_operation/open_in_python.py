# -*- coding: utf-8 -*-
"""
Created on Mon Jun 08 10:00:43 2015

@author: glaunay
"""

import matplotlib.pyplot as plt
import numpy as np
import IMTreatment3.file_operation as imtio
import sys
from os import path
# check
args = sys.argv
if len(args) != 2:
    raise TypeError("Invalid number of argument.")
fp = args[1]
if not path.isfile(fp):
    raise ValueError("Can't find this file.")

# improt according to file type
_, ext = path.splitext(fp)
if ext == ".cimt":
    data = imtio.import_from_file(fp)
elif ext in [".im7", ".IM7"]:
    data = imtio.import_from_IM7(fp)
elif ext in [".vc7", ".VC7"]:
    data = imtio.import_from_VC7(fp)
else:
    raise ValueError("Invalid file type : {}".format(ext))
varname = path.splitext(path.basename(fp))[0]
locals()[varname] = data
data = None
del data

# print information
print("")
print(("Imported a {} object from '{}'".format(type(locals()[varname]),
                                             path.basename(fp))))
print(("Data are accessible through the variable '{}'".format(varname)))
print("")
