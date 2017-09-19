# -*- coding: utf-8 -*-
#!/bin/env python3

# Copyright (C) 2003-2007 Gaby Launay

# Author: Gaby Launay  <gaby.launay@tutanota.com>
# URL: https://framagit.org/gabylaunay/IMTreatment
# Version: 1.0

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
