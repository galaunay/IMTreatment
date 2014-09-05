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



 # field creation
size_x, size_y = 100, 100
nmb = 20
Tt = 4.
Tx = 50.
Ty = 20.
coef_rand = 1
x = np.arange(size_x)
y = np.arange(size_y)
X, Y = np.meshgrid(x, y)
fields = []
for n in np.arange(nmb):
    field = np.sin(X/Tx*2*np.pi) + np.sin(Y/Ty*2*np.pi)
    field += coef_rand*np.random.rand(size_x, size_y)
    field *= np.sin(n/Tt*2*np.pi)
    fields.append(field)

# POD
field = fields[3]
modes, eigvals, eigvects, corr= modred.compute_POD_matrices_snaps_method(field, np.arange(3), return_all=True)
plt.figure()
plt.imshow(modes[0])
plt.colorbar()
plt.title('eigval = {}'.format(eigvals[0]))