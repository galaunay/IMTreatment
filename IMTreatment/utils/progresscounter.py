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

import time as modtime
import numpy as np


class ProgressCounter(object):
    """
    Declare wherever you want and execute 'print_progress' at the begining of
    each loop.
    """
    def __init__(self, init_mess, nmb_max, end_mess=None, name_things='things',
                 perc_interv=5, pgbar_len=15):
        """
        Progress counter.

        Parameters
        ----------
        init_mess, end_mess : strings
            Initial and closure messages.
        nmb_max : integer
            Maximum number of things to count
        name_things : string, optional
            Name of the things to count (default to 'things')
        perc_interv : number, optional
            Percentage interval between two displays (default to '5')
        pgbar_len : integer, optional
            Length (in caracter) of the progress bar
        """
        self.init_mess = init_mess
        self.end_mess = end_mess
        self.nmb_fin = None
        self.curr_nmb = 1
        self.nmb_max = nmb_max
        self.nmb_max_pad = len(str(nmb_max))
        self.name_things = name_things
        self.perc_interv = perc_interv
        if self.nmb_max == np.inf:
            self.interv = perc_interv
        else:
            self.interv = int(np.round(nmb_max)*perc_interv/100.)
        # check if there is more wanted interval than actual loop
        if self.interv == 0:
            self.interv = 1
        self.t0 = None
        self.pgbar_len = 15

    def _print_init(self):
        print("=== {} ===".format(self.init_mess))

    def _print_end(self):
        print("")
        if self.end_mess is not None:
            print("=== {} ===".format(self.end_mess))

    def start_chrono(self):
        self.t0 = modtime.time()
        self._print_init()

    def print_progress(self):
        if self.nmb_max == np.inf:
            self._print_progress_unknown_nmbmax()
        else:
            self._print_progress_full()

    def finish(self):
        self.curr_nmb = self.nmb_max
        self.print_progress()

    def _print_progress_full(self):
        # start chrono if not
        if self.t0 is None:
            self.start_chrono()
        # get current
        i = self.curr_nmb
        # check if i sup nmb_max
        if i == self.nmb_max + 1:
            print("=== Problem with nmb_max value...", end="")
        # check if we have to display something
        if i % self.interv == 0 or i == self.nmb_max:
            ti = modtime.time()
            if i == 0:
                tf = '---'
            else:
                dt = (ti - self.t0)/i
                tf = self.t0 + dt*self.nmb_max
                tf = self._format_time(tf - self.t0)
            ti = self._format_time(ti - self.t0)
            perc = np.round(i*1./self.nmb_max*100)
            pgbar_nmb = int(perc/100*self.pgbar_len)
            pgbar = ("[" + "#"*pgbar_nmb +
                     "."*(self.pgbar_len - pgbar_nmb) + "]")
            text = ("===    {} {:>.0f}%    {:{max_pad}d}/{} {name}    {}/{}"
                    .format(pgbar, perc,
                            i, self.nmb_max, ti, tf,
                            max_pad=self.nmb_max_pad,
                            name=self.name_things))
            print('\r' + text, end="")
        # increment
        self.curr_nmb += 1
        # check if finished
        if i == self.nmb_max:
            self._print_end()
            return 0

    def _print_progress_unknown_nmbmax(self):
        # start chrono if not
        if self.t0 is None:
            self.start_chrono()
        # get current
        i = self.curr_nmb
        # Dsiaplay each time because we cannot do anything else
        if i % self.interv == 0:
            ti = modtime.time()
            if i == 0:
                tf = '---'
            else:
                dt = (ti - self.t0)/i
            ti = self._format_time(ti - self.t0)
            dt = self._format_time(dt)
            text = ("===    {:{max_pad}d} {name}    {}    ({} / {})"
                    .format(i, ti, dt, self.name_things,
                            max_pad=self.nmb_max_pad,
                            name=self.name_things))
            print('\r' + text, end="")
        # increment
        self.curr_nmb += 1
        # check if finished
        if i == self.nmb_max:
            self._print_end()
            return 0

    def _format_time(self, second):
        ms = int((second % 1)*1000)
        second = int(second)
        m, s = divmod(second, 60)
        h, m = divmod(m, 60)
        j, h = divmod(h, 24)
        # only ms
        if s == 0 and m == 0 and h == 0 and j == 0:
            repr_time = '{:d}ms'.format(ms)
            return repr_time
        repr_time = '{:d}s'.format(s)
        if m != 0:
            repr_time = '{:d}mn'.format(m) + repr_time
        if h != 0:
            repr_time = '{:d}h'.format(h) + repr_time
        if j != 0:
            repr_time = '{:d}j'.format(m) + repr_time
        return repr_time
