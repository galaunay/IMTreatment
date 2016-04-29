#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
IMTreatment module

    Auteur : Gaby Launay
"""


from __future__ import print_function
try:
    from multiprocess import Pool, cpu_count, Manager
    MULTIPROCESSING = True
except ImportError:
    MULTIPROCESSING = False
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
