# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 18:50:02 2015

@author: glaunay
"""
import time as modtime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from ..core import ARRAYTYPES, STRINGTYPES, NUMBERTYPES
from matplotlib.collections import LineCollection
import scipy.interpolate as spinterp


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


def colored_plot(x, y, z=None, log='plot', min_colors=1000, color_label='',
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
    if 'norm' in kwargs.keys():
        norm = kwargs.pop('norm')
    else:
        if "vmin" in kwargs.keys():
            mini = kwargs.pop('vmin')
        else:
            mini = np.min(z)
        if "vmax" in kwargs.keys():
            maxi = kwargs.pop('vmax')
        else:
            maxi = np.max(z)
        norm = plt.Normalize(mini, maxi)
    # make cmap
    if 'cmap' in kwargs.keys():
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
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # fake up the array of the scalar mappable. Urgh...
    sm._A = []
    cb = plt.colorbar(sm)
    cb.set_label(color_label)
    return lc