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
import pickle


def memoize_to_file(file_name):
    """ Decorator that memoize the function result in a file. """

    def decorator(original_func):
        gc.disable()
        try:
            with gzip.open(file_name, 'rb') as f:
                cache = pickle.load(f)
        except (IOError, ValueError):
            cache = {}
        gc.enable()

        def new_func(*args, **kwargs):
            key = pickle.dumps((args, kwargs))
            if key not in cache:
                cache[key] = original_func(*args, **kwargs)
                gc.disable()
                with gzip.open(file_name, 'wb') as f:
                    pickle.dump(cache, f, protocol=-1)
                gc.enable()
            return cache[key]

        return new_func

    return decorator
