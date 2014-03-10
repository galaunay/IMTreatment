# -*- coding: utf-8 -*-
"""
Created on Thu Mar 06 13:29:17 2014

@author: glaunay
"""


import numpy as np
import pdb



def scalar_product(field1_x, field1_y, field2_x, field2_y):
    """
    arguments must be numpy arrays
    """
    for field in [field1_x, field1_y, field2_x, field2_y]:
        if not isinstance(field, np.ndarray):
            field = np.array(field)
    if not np.all(field1_x.shape == field2_x.shape)\
            or not np.all(field1_x.shape == field1_y.shape)\
            or not np.all(field2_x.shape == field2_y.shape):
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
    nmb_fields = len(fields_x)
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
    # extraction des champs
    """
    fields_x = []
    fields_y = []
    for field in vfs.fields:
        fields_x.append(field.V.comp_x.values)
        fields_y.append(field.V.comp_y.values)
    # calcul de la matrice de corrélation
    corr = calc_correlation_matrix(fields_x, fields_y)
    # calcul des valeur propres de la matrice de correlatio
    lambd, vectp = calc_eigenvalues(corr)
#    vectp = np.sort(vectp)
#    lambd = np.sort(lambd)
    # calcul des modes propres
    pass
    # retourne val propres et modes propres
    return corr, lambd
    