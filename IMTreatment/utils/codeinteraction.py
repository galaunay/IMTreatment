#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
IMTreatment3 module

    Auteur : Gaby Launay
"""



import os


class RemoveFortranOutput(object):
    """
    Context object to remove Fortran output.

    to be used with 'with' statement.

    Examples
    --------
    >>> with RemoveFortranOutput():
    >>>     # put some fortran functions here
    """

    def __enter__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        self.save = os.dup(1), os.dup(2)
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, type, value, traceback):
        os.dup2(self.save[0], 1)
        os.dup2(self.save[1], 2)
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])
