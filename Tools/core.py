# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 18:50:02 2015

@author: glaunay
"""
import time as modtime
import numpy as np


class ProgressCounter(object):
    """
    Declare wherever you want, start chrono at the begining of the loop,
    execute 'print_progress' at the begining of each loop.
    """

    def __init__(self, init_mess, end_mess, nmb_max, name_things='things',
                 perc_interv=5):
        self.init_mess = init_mess
        self.end_mess = end_mess
        self.nmb_fin = None
        self.curr_nmb = 0
        self.nmb_max = nmb_max
        self.nmb_max_pad = len(str(nmb_max))
        self.name_things = name_things
        self.perc_interv = perc_interv
        self.interv = int(np.round(nmb_max)*perc_interv/100.)
        self.t0 = None

    def _print_init(self):
        print("+++ {} +++".format(self.init_mess))

    def _print_end(self):
        print("+++ {} +++".format(self.init_mess))

    def start_chrono(self):
        self.t0 = modtime.time()
        self._print_init()

    def print_progress(self):
        # start chrono if not
        if self.t0 is None:
            self.start_chrono()
        # get current
        i = self.curr_nmb
        # check if finished
        if i == self.nmb_max:
            self._print_end()
            return 0
        # check if i sup nmb_max
        if i > self.nmb_max:
            print("    Problem with nmb_max value...")
        if i % self.interv == 0 or i == self.nmb_max - 1:
            ti = modtime.time()
            if i == 0:
                tf = '---'
            else:
                dt = (ti - self.t0)/i
                tf = self.t0 + dt*self.nmb_max
                tf = self._format_time(tf - self.t0)
            ti = self._format_time(ti - self.t0)
            print("+++    {:>3.0f} %    {:{max_pad}d}/{} {name}    {}/{}"
                  .format(np.round(i*1./self.nmb_max*100),
                          i, self.nmb_max, ti, tf, max_pad=self.nmb_max_pad,
                          name=self.name_things))
        # increment
        self.curr_nmb += 1

    def _format_time(self, second):
        second = int(second)
        m, s = divmod(second, 60)
        h, m = divmod(m, 60)
        j, h = divmod(h, 24)
        repr_time = '{:d}s'.format(s)
        if m != 0:
            repr_time = '{:d}mn'.format(m) + repr_time
        if h != 0:
            repr_time = '{:d}h'.format(h) + repr_time
        if j != 0:
            repr_time = '{:d}j'.format(m) + repr_time
        return repr_time