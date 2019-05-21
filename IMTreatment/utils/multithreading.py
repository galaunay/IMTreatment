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

from multiprocess import Pool, cpu_count, Manager
from . import ProgressCounter


class MultiThreading(object):
    def __init__(self, funct, data, threads='all'):
        raise Exception("Not functionnal yet !")
        self.funct = funct
        if threads == 'all':
            threads = cpu_count()
        self.pool = Pool(processes=threads)
        self.data = data
        self.PG = None
        self.initializer = None
        self.finalizer = None

    def add_progress_counter(self, init_mess="Beginning", end_mess="Done",
                             name_things='things', perc_interv=5):
        self.PG = ProgressCounter(init_mess=init_mess, end_mess=end_mess,
                                  nmb_max=len(self.data),
                                  name_things=name_things,
                                  perc_interv=perc_interv)
        self.manager = Manager()
        self.manager.register("PG", self.PG)

    def run(self):
        res = self.pool.map_async(self.PG_func_wrapper, self.data)
        self.pool.close()
        self.pool.join()
        return res
