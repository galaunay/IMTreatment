#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
IMTreatment3 module

    Auteur : Gaby Launay
"""



import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from ..utils.types import ARRAYTYPES, STRINGTYPES, NUMBERTYPES
from matplotlib.collections import LineCollection
import scipy.interpolate as spinterp


def make_cmap(colors, position=None, name='my_cmap'):
    '''
    Return a color map cnstructed with the geiven colors and positions.

    Parameters
    ----------
    colors : Nx1 list of 3x1 tuple
        Each color wanted on the colormap. each value must be between 0 and 1.
    positions : Nx1 list of number, optional
        Relative position of each color on the colorbar. default is an
        uniform repartition of the given colors.
    name : string, optional
        Name for the color map
    '''
    # check
    if not isinstance(colors, ARRAYTYPES):
        raise TypeError()
    colors = np.array(colors, dtype=float)
    if colors.ndim != 2:
        raise ValueError()
    if colors.shape[1] != 3:
        raise ValueError()
    if position is None:
        position = np.linspace(0, 1, len(colors))
    else:
        position = np.array(position)
    if position.shape[0] != colors.shape[0]:
        raise ValueError()
    if not isinstance(name, STRINGTYPES):
        raise TypeError()
    # create colormap
    cdict = {'red': [], 'green': [], 'blue': []}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))
    cmap = mpl.colors.LinearSegmentedColormap(name, cdict, 256)
    # returning
    return cmap


def colored_plot(x, y, z=None, log='plot', min_colors=1000, color_label=None,
                 **kwargs):
    '''
    Plot a colored line with coordinates x and y

    Parameters
    ----------
    x, y : nx1 arrays of numbers
        coordinates of each points
    z : nx1 array of number, optional
        values for the color
    log : string, optional
        Type of axis, can be 'plot' (default), 'semilogx', 'semilogy',
        'loglog'
    min_colors : integer, optional
        Minimal number of different colors in the plot (default to 1000).
    color_label : string, optional
        Colorbar label if color is an array.
    kwargs : dict, optional
        list of arguments to pass to the common plot
        (see matplotlib documentation).
    '''
    # check parameters
    if not isinstance(x, ARRAYTYPES):
        raise TypeError()
    x = np.array(x)
    if not isinstance(y, ARRAYTYPES):
        raise TypeError()
    y = np.array(y)
    if len(x) != len(y):
        raise ValueError()
    length = len(x)
    if z is None:
        pass
    elif isinstance(z, ARRAYTYPES):
        if len(z) != length:
            raise ValueError()
        z = np.array(z)
    elif isinstance(z, NUMBERTYPES):
        z = np.array([z]*length)
    else:
        raise TypeError()
    if log not in ['plot', 'semilogx', 'semilogy', 'loglog']:
        raise ValueError()
    # classical plot if z is None
    if z is None:
        return plt.plot(x, y, **kwargs)
    # filtering nan values
    mask = np.logical_or(np.isnan(x), np.isnan(y))
    mask = np.logical_or(np.isnan(z), mask)
    filt = np.logical_not(mask)
    x = x[filt]
    y = y[filt]
    z = z[filt]
    length = len(x)
    # if length is too small, create artificial additional lines
    if length < min_colors:
        interp_x = spinterp.interp1d(np.linspace(0, 1, length), x)
        interp_y = spinterp.interp1d(np.linspace(0, 1, length), y)
        interp_z = spinterp.interp1d(np.linspace(0, 1, length), z)
        fact = np.ceil(min_colors/(length*1.))
        nmb_colors = length*fact
        x = interp_x(np.linspace(0., 1., nmb_colors))
        y = interp_y(np.linspace(0., 1., nmb_colors))
        z = interp_z(np.linspace(0., 1., nmb_colors))
    # make segments
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # make norm
    if 'norm' in list(kwargs.keys()):
        norm = kwargs.pop('norm')
    else:
        if "vmin" in list(kwargs.keys()):
            mini = kwargs.pop('vmin')
        else:
            mini = np.min(z)
        if "vmax" in list(kwargs.keys()):
            maxi = kwargs.pop('vmax')
        else:
            maxi = np.max(z)
        norm = plt.Normalize(mini, maxi)
    # make cmap
    if 'cmap' in list(kwargs.keys()):
        cmap = kwargs.pop('cmap')
    else:
        cmap = plt.cm.__dict__[mpl.rc_params()['image.cmap']]
    # create line collection
    lc = LineCollection(segments, array=z, norm=norm, cmap=cmap, **kwargs)
    ax = plt.gca()
    ax.add_collection(lc)
    # adjuste og axis idf necessary
    if log in ['semilogx', 'loglog']:
        ax.set_xscale('log')
    if log in ['semilogy', 'loglog']:
        ax.set_yscale('log')
    plt.axis('auto')
    # colorbar
    if color_label is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # fake up the array of the scalar mappable. Urgh...
        sm._A = []
        cb = plt.colorbar(sm)
        cb.set_label(color_label)
    return lc
