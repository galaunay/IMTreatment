#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
IMTreatment module

    Auteur : Gaby Launay
"""


from __future__ import print_function
import time as modtime
import numpy as np


class ProgressCounter(object):
    """
    Declare wherever you want and execute 'print_progress' at the begining of each loop.
    """
    def __init__(self, init_mess, end_mess, nmb_max, name_things='things',
                 perc_interv=5):
        """
        Progress counter.

        Parameters
        ----------
        init_mess, end_mess : strings
            Initial and closure messages
        nmb_max : integer
            Maximum number of things to count
        name_things : string, optional
            Name of the things to count (default to 'things')
        perc_inerv : number, optional
            Percentage interval between two displays (default to '5')
        """
        self.init_mess = init_mess
        self.end_mess = end_mess
        self.nmb_fin = None
        self.curr_nmb = 1
        self.nmb_max = nmb_max
        self.nmb_max_pad = len(str(nmb_max))
        self.name_things = name_things
        self.perc_interv = perc_interv
        self.interv = int(np.round(nmb_max)*perc_interv/100.)
        # check if there is more wanted interval than actual loop
        if self.interv == 0:
            self.interv = 1
        self.t0 = None

    def _print_init(self):
        print("+++ {} +++".format(self.init_mess))

    def _print_end(self):
        print("")
        print("+++ {} +++".format(self.end_mess))

    def start_chrono(self):
        self.t0 = modtime.time()
        self._print_init()

    def print_progress(self):
        # start chrono if not
        if self.t0 is None:
            self.start_chrono()
        # get current
        i = self.curr_nmb
        # check if i sup nmb_max
        if i == self.nmb_max + 1:
            print("+++ Problem with nmb_max value...", end="")
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
            text = ("+++    {:>3.0f} %    {:{max_pad}d}/{} {name}    {}/{}"
                    .format(np.round(i*1./self.nmb_max*100),
                            i, self.nmb_max, ti, tf, max_pad=self.nmb_max_pad,
                            name=self.name_things))
            print('\r' + text, end="")
        # increment
        self.curr_nmb += 1
        # check if finished
        if i == self.nmb_max:
            self._print_end()
            return 0

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
