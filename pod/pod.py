# -*- coding: utf-8 -*-
"""
Created on Thu Mar 06 13:29:17 2014

@author: glaunay
"""


import numpy as np



def scalar_product(field1_x, field1_y, field2_x, field2_y):
    """
    arguments must be numpy arrays
    """
    if not np.all(field1_x.shape == field2_x.shape)\
            or np.all(field1_x.shape == field1_y.shape)\
            or np.all(field2_x.shape == field2_y.shape):
        raise ValueError("'field1' and 'field2' must have the same dimension")
    prod = field1_x*field2_x + field1_y*field2_y
    return prod


def discr_integral(field):
    """
    field must be numpy array.
    Do not take account of axis values.
    """
    integ = np.sum(field)
    return integ


def calc_correlation_matrix(fields_x, fields_y):
    """
    fields must be tuples of numpy arrays.
    """
    nmb_fields = len(fields)
    corr_matrix = np.zeros((nmb_fields, nmb_fields))
    for i in np.arange(nmb_fields):
        for j in np.arange(nmb_fields):
            # TODO : matrice diagonlae, ne pas recalculer Kij partout
            scalar_prod = scalar_product(fields_x[i], fields_y[i],
                                         fields_x[j], fields_y[j])
            corr_matrix[i, j] = discr_integral(scalar_prod)
    return corr_matrix

def calc_eigenvalues(matrix):
    """
    """
    # TODO : peute etre solver plus efficace (a voir)
    lambd, phi = np.linalg.eig(matrix)
    return lambd, phi
    
def pod_decomposition(vfs):
    """
    take  temporal velocity fields
    """
    pass