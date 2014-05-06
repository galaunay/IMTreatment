# -*- coding: utf-8 -*-
"""
Created on Thu Mar 06 13:29:17 2014

@author: glaunay
"""


import numpy as np
import pdb
import modred

def pod(field, modes='all'):
    """
    Compute POD modes of the given fields using the snapshot method.
    """
    ### TEST ###
    # parameters
    taille = 10.
    x = np.arange(taille)
    y = np.arange(taille)
    x, y = np.meshgrid(x, y)
    V0 = 10.*np.cos(x/taille*np.pi)*np.sin(y/taille*np.pi)
    freq = .02
    freq2 = .01
    num_vecs = 20
    num_modes = [0, 1]
    # data construction
    V = [V0.flatten()]
    for t in np.arange(1, num_vecs):
        V = np.concatenate((V, [V0.flatten()*np.cos(2.*np.pi*freq*t)*np.cos(2.*np.pi*freq2*t)]), axis=0)
    V = np.transpose(V)
    # pod extraction
    modes, eig_vals, eig_vecs, corr_mat =\
         modred.compute_POD_matrices_snaps_method(V, num_modes, return_all=True)




