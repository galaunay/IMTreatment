# -*- coding: utf-8 -*-
#!/bin/env python3

# Copyright (C) 2003-2007 Gaby Launay

# Author: Gaby Launay  <gaby.launay@tutanota.com>
# URL: https://framagit.org/gabylaunay/IMTreatment

# This file is part of IMTreatment.

# IMTreatment is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.

# IMTreatment is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from collections import OrderedDict
from matplotlib.collections import LineCollection
from os import path

import warnings
import matplotlib as mpl
import matplotlib.animation as mplani
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as spinterp
from matplotlib.widgets import Button, Slider


ARRAYTYPES = (np.ndarray, list, tuple)
NUMBERTYPES = (int, int, float, complex, np.float, np.float16, np.float32,
               np.float64, np.int, np.int16, np.int32, np.int64, np.int8)
STRINGTYPES = (str, str)


def is_sorted(data):
    return all(a <= b for a, b in zip(data[:-1], data[1:]))


def get_color_cycles():
    return mpl.rcParams['axes.prop_cycle'].by_key()['color']


def get_color_gradient(cmap='jet', number=10):
    """
    Return a gradient of color for plot uses.
    """
    cmap = plt.cm.get_cmap(name=cmap)
    colors = [cmap(i/(number - 1)) for i in range(number)]
    return colors


def make_discrete_cmap(interv_centers, cmap=None):
    """
    Create a discrete map, with intervals centered on the given values.
    """
    if cmap is None:
        cmap = plt.rcParams['image.cmap']
    cmap = plt.cm.get_cmap(cmap)
    # Get boundaries
    interv_bounds = [(interv_centers[i+1] + interv_centers[i])/2.
                     for i in range(len(interv_centers) - 1)]
    mini = interv_bounds[0] - abs(interv_bounds[1] - interv_bounds[0])
    maxi = interv_bounds[-1] + abs(interv_bounds[-1] - interv_bounds[-2])
    # Get color list
    color_list = []
    for i in range(len(interv_centers)):
        color_list += [cmap(i/float(len(interv_centers) - 1))]
    # Create new cmap
    new_cmap = mpl.colors.ListedColormap(color_list,
                                         name="Custom_discrete_map")
    norm = plt.Normalize(mini, maxi)
    # return
    return new_cmap, norm


def annotate_multiple(s, xy, xytext, *args, **kwargs):
    ans = []
    ax = plt.gca()
    an = ax.annotate(s, xy[0], xytext=xytext, *args, **kwargs)
    ans.append(an)
    d = {}
    try:
        d['xycoords'] = kwargs['xycoords']
    except KeyError:
        pass
    try:
        d['arrowprops'] = kwargs['arrowprops']
    except KeyError:
        pass
    for tmp_xy in xy[1:]:
        an = ax.annotate(s, tmp_xy, alpha=0.0, xytext=xytext, *args, **kwargs)
        ans.append(an)
    return ans


def mark_axe(txt, ax=None, loc=2, pad=0.3, borderpad=0., font_props=None,
             frameon=True):
    # get axe
    if ax is None:
        ax = plt.gca()
    # mark style
    if font_props is None:
        font_props = dict(fontweight='bold')
    # draw
    txt = mpl.offsetbox.AnchoredText(txt, loc=loc, prop=font_props, pad=pad,
                                     borderpad=borderpad,
                                     frameon=frameon)
    ax.add_artist(txt)
    return txt


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


class Formatter(mpl.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathtext=True):
        self.oom = order
        self.fformat = fformat
        mpl.ticker.ScalarFormatter.__init__(self,
                                            useOffset=offset,
                                            useMathText=mathtext)

    def _set_orderOfMagnitude(self, nothing):
        self.orderOfMagnitude = self.oom

    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '${}$'.format(mpl.ticker._mathdefault(self.format))


def save_animation(animpath, fig=None, fields='all', writer='ffmpeg', fps=24,
                   title="", artist="IMTreatment", comment="",
                   bitrate=-1, codec='ffv1', dpi=150):
        """
        Save the current button manager displays as an animation.

        Parameters
        ----------
        animpath : string
            Path where to save animation
        fig : Figure instance
            Figure to save the animation from (if None, get the current one)
        fields : string or 2x1 list of numbers
            Fields interval to save. Default is 'all' for all the fields.
        writer : string
            Name of the writer to use
            (available writers are listed in
            'matplotlib.animation.writers.list()'
        codec : string
            One of the codec of the choosen writer (default to 'ffv1')
        fps : integer
            Number of frame per second (default to 24)
        bitrate : integer
            Video bitrate in kb/s (default to -1)
            Set this to -1 for letting the writter choose.
        dpi : integer
            dpi of the video images before compression (default to 150)
        title, artist, comment : strings
            Information added to the file metadata
        """
        fig = plt.gcf()
        try:
            bm = fig.button_manager
        except AttributeError:
            raise Exception("The current figure is not associated with a "
                            "button manager")
        bm.save_animation(animpath=animpath, fields=fields, writer=writer,
                          fps=fps, title=title, artist=artist, comment=comment,
                          bitrate=bitrate, codec=codec, dpi=dpi)


def use_perso_style():
    """
    Change matplotlib default style to something nicer
    """
    fp = path.dirname(__file__)
    plt.style.use(path.join(fp, r'perso.mplstyle'))
    # plt.rcParams["backend"] = "gtkcairo"


# Data manipulation:
def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct
    format for LineCollection:
    an array of the form  numlines x (points per line) x 2 (x and y) array
    '''
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


# Interface to LineCollection:
def colored_plot(x, y, z=None, log='plot', min_colors=1000, colorbar=False,
                 color_label='', **kwargs):
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
    colorbar : bool
        .
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
    if len(x) < 2:
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
    segments = make_segments(x, y)
    # make norm
    if 'norm' in list(kwargs.keys()):
        norm = kwargs.pop('norm')
    else:
        norm = plt.Normalize(np.min(z), np.max(z))
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
    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # fake up the array of the scalar mappable. Urgh...
        sm._A = []
        cb = plt.colorbar(sm)
        cb.set_label(color_label)
    return lc


# button gestion class
class ButtonManager(object):
    # TODO :
    #    -Axis limits have a werid behavior when using zoom

    def __init__(self, displayers, xlabel="", ylabel="", sharecb=True,
                 normcb=None, play_interval=2):
        # check if multiple displayers
        try:
            len(displayers)
        except TypeError:
            displayers = [displayers]
        # store Some more informations from displayers
        self.incr = 1
        self.ind = 0
        if len(displayers) == 0:
            raise ValueError()
        self.displayers = displayers
        self.ind_max = self._get_indmax_form_displs()
        for displ in self.displayers:
            displ.button_manager = self
        self.tmp_key_number = None
        self.sharecb = sharecb
        self.linked_graphs = []
        # check if a colorbar should be present
        is_cbs = [displ.mapped_colors for displ in displayers]
        self.is_cb = np.any(is_cbs)
        if self.is_cb:
            self.displ_cb = displayers[np.where(is_cbs)[0][0]]
        else:
            self.displ_cb = None
        # create figure or get the current one if compatible
        tmp_fig = plt.gcf()
        if len(tmp_fig.axes) == 0:
            self.fig = tmp_fig
        elif hasattr(tmp_fig, 'button_manager'):
            tmp_fig.button_manager.add_displayers(displayers)
            return None
        else:
            self.fig = plt.figure()
        # remove default keybinding
        self.fig.canvas.mpl_disconnect(
            self.fig.canvas.manager.key_press_handler_id)
        # associate the figure with the button manager
        self.fig.button_manager = self
        # Create the buttons and slider axes
        if self.is_cb:
            self.ax = mpl.axes.Axes(self.fig, [0.1, 0.2, .725, 0.75])
            self.fig.add_axes(self.ax)
            self.cbax = mpl.axes.Axes(self.fig, [0.85, 0.2, .025, 0.7])
            self.fig.add_axes(self.cbax)
        else:
            self.ax = mpl.axes.Axes(self.fig, [0.1, 0.2, .875, 0.75])
            self.fig.add_axes(self.ax)
            self.cbax = None
        self.axprev = mpl.axes.Axes(self.fig, [0.02, 0.02, 0.1, 0.05])
        self.fig.add_axes(self.axprev)
        self.axnext = mpl.axes.Axes(self.fig, [0.88, 0.02, 0.1, 0.05])
        self.fig.add_axes(self.axnext)
        self.axplay = mpl.axes.Axes(self.fig, [0.73, 0.02, 0.1, 0.05])
        self.fig.add_axes(self.axplay)
        self.axslid = mpl.axes.Axes(self.fig, [0.15, 0.02, 0.5, 0.05])
        self.fig.add_axes(self.axslid)
        # Add the button and slider and connect them
        self.button_kwargs = {"color": "w", "hovercolor": [.5]*3}
        self.slider_kwargs = {"facecolor": [.75]*3, "edgecolor": 'k', "lw": 1}
        self.slider_buff_kwargs = {'color': 'g', 'alpha': 0.5, 'lw': 0}
        self.slider_lims_kwargs = {'color': 'k', 'alpha': .5, 'lw': 0}
        self.bnext = Button(self.axnext, 'Next', **self.button_kwargs)
        self.bprev = Button(self.axprev, 'Previous', **self.button_kwargs)
        self.bplay = Button(self.axplay, 'Play', **self.button_kwargs)
        self.bnext.on_clicked(self.nextf)        # set up the apsect
        self.bprev.on_clicked(self.prevf)
        self.bplay.on_clicked(self.playf)
        self.bslid = Slider(self.axslid, "", valmin=1, valfmt='%d',
                            valmax=self.ind_max+1, valinit=1,
                            **self.slider_kwargs)
        self.bslid.on_changed(self.slid)
        self.slider_faces = {}
        # Add a timer for play
        self.on_play = False
        self.play_timer = self.fig.canvas.new_timer(interval=play_interval)
        self.play_timer.add_callback(self.timer_play)
        self.play_interval = play_interval
        self.could_interact = True
        self.ind_lims = [None, None]
        self.ind_lims_faces = [None, None]
        self.ind_lims_texts = [None, None]
        # add some keyboard shortcut
        self.fig.canvas.mpl_connect('key_press_event', self.keyf)
        # get aspect
        aspect, adjustable = self._get_aspect_from_displs()
        # Compute norm for colorbar, according to 'sharecb' and 'normcb'
        self.cb_norm = None
        self.cb = None
        if self.is_cb:
            cb_norm, cmap = self._get_cb_norm_from_displs(normcb=normcb)
            self.cb_norm = cb_norm
            self.cb = mpl.colorbar.ColorbarBase(self.cbax, norm=self.cb_norm,
                                                cmap=cmap)
            self.cb.set_norm(self.cb_norm)
            self.cb.draw_all()
        # update with initial data
        self.update()
        # set texts and rescale
        self.xlim = [np.min([displ.xlim[0] for displ in self.displayers]),
                     np.max([displ.xlim[1] for displ in self.displayers])]
        self.ylim = [np.min([displ.ylim[0] for displ in self.displayers]),
                     np.max([displ.ylim[1] for displ in self.displayers])]
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        if aspect is not None:
            self.ax.set_aspect(aspect, adjustable=adjustable)
        self.ax.autoscale(False)
        if np.any([displ.data_type in ['field_1D', 'field_2D']
                   for displ in self.displayers]):
            self.ax.relim(visible_only=True)
            self.ax.autoscale_view(tight=True)        # set up the apsect
        else:
            self.ax.set_xlim(*self.xlim)
            self.ax.set_ylim(*self.ylim)
        # ensure that current ax is the good one
        plt.sca(self.ax)

    def _get_cb_norm_from_displs(self, normcb):
        # get cmap for colorbar
        try:
            cmap = self.displ_cb.dargs['cmap']
        except KeyError:
            cmap = mpl.rcParams['image.cmap']
        # secified norm
        if normcb is not None:
            cb_norm = normcb
        # no specified norm, but shared one
        elif self.sharecb:
            for displ in self.displayers:
                norm = displ.get_global_norm()
                displ.dargs['norm'] = norm
            cb_norm = self.displ_cb.dargs['norm']
        # different norm for each plot
        else:
            cb_norm = None
        # return
        return cb_norm, cmap

    def _get_aspect_from_displs(self):
        # get aspect
        aspects = []
        for displ in self.displayers:
            try:
                aspects.append(displ.dargs.pop('aspect'))
            except KeyError:
                pass
        if len(aspects) == 0:
            aspect = 'auto'
        elif np.any([asp == 'equal' for asp in aspects]):
            aspect = 'equal'
        elif np.any([asp == 'auto' for asp in aspects]):
            aspect = 'auto'
        else:
            raise ValueError()
        # get adjustable
        adjustables = []
        for displ in self.displayers:
            try:
                adjustables.append(displ.dargs.pop('adjustable'))
            except KeyError:
                pass
        if len(adjustables) == 0:
            adjustable = 'box'
        else:
            adjustable = adjustables[0]
        return aspect, adjustable

    def _get_indmax_form_displs(self):
        length = np.array([displ.length for displ in self.displayers])
        if np.any(length != self.displayers[0].length):
            raise ValueError()
        ind_max = self.displayers[0].length - 1
        return ind_max

    def prevf(self, event):
        # get increment
        key_number = self._get_key_number()
        if key_number is None:
            incr = self.incr
        else:
            incr = key_number
        #
        if self.ind == 0:
            return None
        # set new ind
        new_ind = self.ind - incr
        if new_ind > 0:
            self.ind = new_ind
        else:
            self.ind = 0
        # update
        self.update()

    def nextf(self, event):
        # get increment
        key_number = self._get_key_number()
        if key_number is None:
            incr = self.incr
        else:
            incr = key_number
        #
        if self.ind == self.ind_max:
            return None
        # set new ind
        new_ind = self.ind + incr
        if new_ind <= self.ind_max:
            self.ind = new_ind
        else:
            self.ind = self.ind_max
        # update
        self.update()

    def playf(self, event):
        # get and update time interval
        self._update_interv()
        self.play_timer.interval = self.play_interval
        #
        if self.on_play:
            self.play_timer.stop()
            self.on_play = False
            self.bplay.label.set_text('Play')
        else:
            self.play_timer.start()
            self.on_play = True
            self.bplay.label.set_text('Stop')

    def timer_play(self):
        # check if lims are set
        if self.ind_lims[1] is not None:
            ind1, ind2 = self.ind_lims
        else:
            ind1 = 0
            ind2 = self.ind_max + 1
        # update the display
        self.ind += self.incr
        if self.ind >= ind2:
            self.ind = ind1
        elif self.ind >= self.ind_max:
            self.ind = 0
        self.update()

    def goto(self):
        key_number = self._get_key_number()
        if key_number is None:
            return None
        if key_number > self.ind_max or key_number < 1:
            return None
        self.ind = key_number - 1
        self.update()

    def goto_end(self, event):
        key_number = self._get_key_number()
        if key_number is None:
            key_number = self.ind_max + 1
        if key_number > self.ind_max + 1:
            return None
        self.ind = key_number - 1
        self.update()

    def goto_beg(self, event):
        key_number = self._get_key_number()
        if key_number is None:
            key_number = 1
        if key_number > self.ind_max + 1:
            return None
        self.ind = key_number - 1
        self.update()

    def set_lims(self, lim1=None, lim2=None):
        # if lims are specified, change it
        if lim1 is not None or lim2 is not None:
            if lim1 is not None:
                if lim1 == 'del':
                    self.ind_lims[0] = None
                    self.ind_lims_faces[0].remove()
                    self.ind_lims_texts[0].remove()
                    self.ind_lims_faces[0] = None
                else:
                    self.ind_lims[0] = lim1
                    fac = self.axslid.axvspan(lim1, lim1+1,
                                              **self.slider_lims_kwargs)
                    self.ind_lims_faces[0] = fac
                    text = self.axslid.text(x=lim1+0.5, y=1.1, ha='center',
                                            s="{}".format(lim1 + 1))
                    self.ind_lims_texts[0] = text
            if lim2 is not None:
                if lim2 == 'del':
                    self.ind_lims[1] = None
                    self.ind_lims_faces[1].remove()
                    self.ind_lims_texts[1].remove()
                    self.ind_lims_faces[1] = None
                else:
                    self.ind_lims[1] = lim2
                    fac = self.axslid.axvspan(lim2, lim2+1,
                                              **self.slider_lims_kwargs)
                    self.ind_lims_faces[1] = fac
                    text = self.axslid.text(x=lim2+0.5, y=1.1, ha='center',
                                            s="{}".format(lim2 + 1))
                    self.ind_lims_texts[1] = text
                    # sort if necessary
                    if self.ind_lims[1] < self.ind_lims[0]:
                        self.ind_lims = self.ind_lims[::-1]
                        self.ind_lims_faces = self.ind_lims_faces[::-1]
                        self.ind_lims_texts = self.ind_lims_texts[::-1]
            return None
        # get ind
        key_nmb = self._get_key_number()
        if key_nmb is None:
            key_nmb = self.ind
        else:
            key_nmb -= 1
        # If ind is already a limite, remove it
        if key_nmb in self.ind_lims:
            if key_nmb == self.ind_lims[0]:
                self.set_lims(lim1='del')
            else:
                self.set_lims(lim2='del')
            return None
        # If lims are already set, remove them
        if self.ind_lims[1] is not None and self.ind_lims[0] is not None:
            self.set_lims(lim1='del', lim2='del')
        # If lims are not set, set the first one
        elif self.ind_lims[0] is None:
            self.set_lims(lim1=key_nmb)
        # Set the second limit
        else:
            self.set_lims(lim2=key_nmb)

    def goto_lims(self):
        self._get_key_number()
        if self.ind_lims[0] is None and self.ind_lims[1] is None:
            return None
        if self.ind_lims[0] is not None:
            if self.ind != self.ind_lims[0]:
                self.ind = self.ind_lims[0]
                self.update()
                return None
        if self.ind_lims[1] is not None:
            if self.ind != self.ind_lims[1]:
                self.ind = self.ind_lims[1]
                self.update()
                return None

    def keyf(self, event):
        # get directions
        if event.key in [' ', 'right', '+', 'l']:
            self.nextf(None)
        elif event.key in ['left', 'backspace', '-', 'h']:
            self.prevf(None)
        elif event.key in ['up', 'k']:
            self.goto_end(None)
        elif event.key in ['down', 'j']:
            self.goto_beg(None)
        elif event.key in ['enter', 'g']:
            self.goto()
        elif event.key in ['p', '.']:
            self.playf(None)
        elif event.key in ['i', '*']:
            self._update_incr()
        elif event.key in ['t', '/']:
            self._update_interv()
        elif event.key in ['q']:
            self.close()
        elif event.key in ['s']:
            self.save()
        elif event.key in ['pagedown', 'a']:
            self.set_lims()
        elif event.key in ['pageup', 'b']:
            self.goto_lims()
        else:
            pass
        # get numbers
        if event.key in ['{}'.format(i) for i in range(10)]:
            if self.tmp_key_number is None:
                self.tmp_key_number = int(event.key)
            else:
                self.tmp_key_number *= 10
                self.tmp_key_number += int(event.key)
        else:
            self.tmp_key_number = None

    def slid(self, event):
        # avoid recursion on update()
        if event - 1 == self.ind:
            return None
        self.ind = int(event) - 1
        self.update()

    def update(self):
        # deactivate buttons
        self.deactivate_buttons()
        # actualize slider
        self.bslid.set_val(self.ind + 1)
        # actualize all displayers
        for displ in self.displayers:
            displ.draw(self.ind, ax=self.ax, cb=False, remove_current=True,
                       rescale=False)
        if self.is_cb and not self.sharecb:
            norm = self.displ_cb.get_norm(self.ind)
            self.cb.set_norm(norm)
            self.cb.draw_all()
        # update linked graphs
        self._update_linked_graphs()
        # reactivate buttons
        self.activate_buttons()
        # update slide apparence
        self._update_slider_faces()

    def _update_slider_faces(self):
        displ = self.displayers[0]
        if not displ.use_buffer:
            return None
        # get buffered indices
        buff_inds = np.where(displ.displ_saved_inds != 0)[0]
        old_span = []
        # remove old faces
        for ind in list(self.slider_faces.keys()):
            if ind not in buff_inds:
                face = self.slider_faces.pop(ind)
                old_span.append(face)
        # add new ones
        ind = self.ind
        if ind not in list(self.slider_faces.keys()):
            # reuse old spans if possible
            if len(old_span) != 0:
                fac = old_span.pop(0)
                xy = fac.xy
                xy[:, 0] += ind - xy[0, 0]
                fac.set_xy(xy)
            # create a new face
            else:
                fac = self.axslid.axvspan(ind, ind+1,
                                          **self.slider_buff_kwargs)
            self.slider_faces[ind] = fac

    def _update_linked_graphs(self):
        # update linked graphs
        if len(self.linked_graphs) != 0:
            for graph in self.linked_graphs:
                if graph.ind != self.ind:
                    graph.ind = self.ind
                    graph.incr = self.incr
                    graph.update()

    def _get_key_number(self):
        key_number = self.tmp_key_number
        self.tmp_key_number = None
        return key_number

    def _update_incr(self):
        key_number = self._get_key_number()
        if key_number is not None:
            self.incr = key_number

    def _update_interv(self):
        key_number = self._get_key_number()
        if key_number is not None:
            self.play_interval = key_number

    def deactivate_buttons(self):
        self.could_interact = False
        self.bnext.set_active(False)
        self.bprev.set_active(False)
        self.bslid.set_active(False)

    def activate_buttons(self):
        self.could_interact = True
        self.bnext.set_active(True)
        self.bprev.set_active(True)
        self.bslid.set_active(True)

    def add_displayers(self, displayers):
        for displ in displayers:
            self.displayers.append(displ)
            displ.button_manager = self
        self.update()

    def link_to_other_graph(self, graph):
        if isinstance(graph, ButtonManager):
            if graph not in self.linked_graphs:
                self.linked_graphs.append(graph)
                graph.link_to_other_graph(self)
        elif hasattr(graph, 'button_manager'):
            self.link_to_other_graph(graph.button_manager)
        else:
            raise TypeError()

    def save_animation(self, animpath, fields='all', writer='ffmpeg', fps=24,
                       title="", artist="IMTreatment", comment="",
                       bitrate=-1, codec='ffv1', dpi=150):
        """
        Save the button manager displays as an animation.

        Parameters
        ----------
        animpath : string
            Path where to save animation
        fields : string or 2x1 list of numbers
            Fields interval to save. Default is 'all' for all the fields.
        writer : string
            Name of the writer to use
            (available writers are listed in
            'matplotlib.animation.writers.list()'
        codec : string
            One of the codec of the choosen writer (default to 'ffv1')
        fps : integer
            Number of frame per second (default to 24)
        bitrate : integer
            Video bitrate in kb/s (default to -1)
            Set this to -1 for letting the writter choose.
        dpi : integer
            dpi of the video images before compression (default to 150)
        title, artist, comment : strings
            Information added to the file metadata
        """
        # Check
        if bitrate == "default":
            bitrate = -1
        # Get first and last field
        if fields == "all":
            fields = [0, self.ind_max]
        # Get writer
        try:
            Writer = mplani.writers[writer]
        except KeyError:
            raise ValueError("{} not available as writer, try one of these "
                             " {}".format(writer, mplani.writers.list()))

        metadata = dict(title=title, artist=artist,
                        comment=comment)
        writer = Writer(fps=fps, metadata=metadata, bitrate=bitrate,
                        codec=codec)
        # write
        backup_ind = self.ind
        self.ind = fields[0]
        self.update()
        with writer.saving(self.fig, animpath, dpi):
            for i in range(fields[1] - fields[0]):
                self.nextf(None)
                writer.grab_frame()
        # restore
        self.ind = backup_ind
        self.update()

    def close(self):
        if self.on_play:
            self.playf(None)
#        self.__del__()

    def __del__(self):
        pass
#        plt.close(self.fig)
#        for displ in self.displayers:
#            del displ
#        del self.displ_cb

    def save(self):
        pass
#        self.fig.canvas.toolbar.save_figure()


class Displayer(object):

    points_default_args = {"kind": "scatter"}
    profile_default_args = {"kind": "plot"}
    field_1D_default_args = {"kind": "matrix",
                             "interpolation": "nearest",
                             "aspect": "equal"}
    field_2D_default_args = {"kind": "quiver",
                             "aspect": "equal"}

    def __init__(self, x, y, values=None, data_type=None, sharebds=True,
                 buffer_size=100, **kwargs):
        # get figure
        if "ax" not in list(kwargs.keys()):
            self.ax = None
            self.fig = None
        else:
            self.ax = kwargs.pop("ax")
            self.fig = self.ax.figure
        # get data
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        if len(self.x) != 0:
            try:
                tmp_x = np.concatenate(self.x)
            except ValueError:
                tmp_x = self.x
            self.xlim = [np.min(tmp_x[~np.isnan(tmp_x)]),
                         np.max(tmp_x[~np.isnan(tmp_x)])]
            try:
                tmp_y = np.concatenate(self.y)
            except:
                tmp_y = self.y
            self.ylim = [np.min(tmp_y[~np.isnan(tmp_y)]),
                         np.max(tmp_y[~np.isnan(tmp_y)])]
        if values is None:
            self.values = None
        else:
            self.values = np.asarray(values)
        self.vmin = None
        self.vmax = None
        self.colors = None
        self.magnitude = None
        self.sharebds = sharebds
        # check if data is multidimensionnal
        self.multidim = False
        try:
            x[0][0]
            self.multidim = True
        except IndexError:
            pass
        if isinstance(x[0], list) and isinstance(y[0], list):
            self.multidim = True
        if self.multidim:
            self.length = len(self.x)
        else:
            self.length = 1
        self.curr_ind = 0
        # place to store the drawings
        self.draws = [None]*self.length
        self.curr_draw = None
        self.displ_saved_inds = np.zeros(self.length, dtype=int)
        self.displ_saved_curr_ind = 1
        if buffer_size is None:
            self.max_saved_displ = 0
            self.use_buffer = False
        else:
            self.max_saved_displ = buffer_size
            self.use_buffer = True
        # try to guess the data type
        tmp_x, tmp_y, tmp_values, tmp_colors, tmp_magn = self.get_data(i=0)
        if data_type is None:
            if values is None:
                if is_sorted(tmp_x) or is_sorted(tmp_y):
                    self.data_type = "profile"
                else:
                    self.data_type = "points"
            else:
                if tmp_values.ndim == 1:
                    self.data_type = "points"
                elif tmp_values.ndim == 2:
                    self.data_type = "field_1D"
                elif tmp_values.ndim == 3:
                    self.data_type = "field_2D"
                else:
                    raise ValueError("Unable to detect the data type")
        elif data_type in ["points", "profile", "field_1D", "field_2D"]:
            self.data_type = data_type
        else:
            raise ValueError("Unknown 'data_type' argument")
        # set default values according to data type
        if self.data_type == "points":
            self.dargs = self.points_default_args.copy()
        elif self.data_type == "profile":
            self.dargs = self.profile_default_args.copy()
        elif self.data_type == "field_1D":
            self.dargs = self.field_1D_default_args.copy()
            if "color" in list(kwargs.keys()):
                if self.multidim:
                    self.colors = [kwargs.pop('color')]*self.length
                else:
                    self.colors = kwargs.pop('color')
            else:
                self.colors = self.values
        elif self.data_type == "field_2D":
            self.dargs = self.field_2D_default_args.copy()
            if self.multidim:
                self.magnitude = [(self.values[i][0]**2 +
                                   self.values[i][1]**2)**.5
                                  for i in range(self.length)]
            else:
                self.magnitude = (self.values[0]**2 +
                                  self.values[1]**2)**.5
            if "color" in list(kwargs.keys()):
                if self.multidim:
                    self.colors = [kwargs.pop('color')]*self.length
                else:
                    self.colors = kwargs.pop('color')
            else:
                self.colors = np.asarray(self.magnitude)
        # check if colors are mapped into data or not
        self.mapped_colors = False
        try:
            ndim = self.colors[0].ndim
            if self.colors[0].shape == self.values.shape[-ndim::]:
                self.mapped_colors = True
        except:
            pass
        # set user defined display arguments
        if 'kind' in list(kwargs.keys()):
            if kwargs['kind'] is None:
                kwargs.pop('kind')
        self.dargs.update(kwargs)

    def get_data(self, i=None):
        if not self.multidim:
            return self.x, self.y, self.values, self.colors, self.magnitude
        elif self.multidim and i is not None:
            if self.values is None:
                tmp_values = None
            else:
                tmp_values = self.values[i]
            if self.colors is None:
                tmp_colors = None
            else:
                tmp_colors = self.colors[i]
            if self.magnitude is None:
                tmp_magn = None
            else:
                tmp_magn = self.magnitude[i]
            return self.x[i], self.y[i], tmp_values, tmp_colors, tmp_magn
        else:
            raise ValueError()

    def get_data_at_point(self, x, y, i=None):
        if i is None:
            i = self.curr_ind
        # Get data
        tmp_x, tmp_y, tmp_values, tmp_colors, tmp_magn = self.get_data(i=i)
        dic = OrderedDict()
        # check if mouse is too far from points
        if (x > self.xlim[1] or x < self.xlim[0] or
                y > self.ylim[1] or y < self.ylim[0]):
            return None
        #
        ind_x = np.argmin(np.abs(x - tmp_x))
        # if a profile
        if self.data_type in ['profile', 'points']:
            dic['x'] = tmp_x[ind_x]
            dic['y'] = tmp_y[ind_x]
        elif self.data_type in ['field_1D']:
            ind_y = np.argmin(np.abs(y - tmp_y))
            dic['x'] = tmp_x[ind_x]
            dic['y'] = tmp_y[ind_y]
            dic['value'] = tmp_values[ind_x, ind_y]
        elif self.data_type in ['field_2D']:
            ind_y = np.argmin(np.abs(y - tmp_y))
            dic['x'] = tmp_x[ind_x]
            dic['y'] = tmp_y[ind_y]
            dic['Vx'] = tmp_values[0, ind_x, ind_y]
            dic['Vy'] = tmp_values[1, ind_x, ind_y]
        else:
            raise Exception()
        return dic

    def get_norm(self, i):
        vmin = np.min(self.colors[i])
        vmax = np.min(self.colors[i])
        return mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    def get_global_norm(self):
        if self.vmin is None:
            tmp_mins = []
            for col in self.colors:
                filt = ~np.isnan(col)
                if np.any(filt):
                    tmp_mins.append(np.min(col[filt]))
            self.vmin = np.min(tmp_mins)
        if self.vmax is None:
            tmp_maxs = []
            for col in self.colors:
                filt = ~np.isnan(col)
                if np.any(filt):
                    tmp_maxs.append(np.max(col[filt]))
            self.vmax = np.max(tmp_maxs)
        return mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax)

    def _toggle_visibility(self, obj):
        # if obj is an array of objects
        try:
            obj[0]
            for single_obj in obj:
                self._toggle_visibility(single_obj)
            return None
        except TypeError:
            pass
        # Remove or add obj according to its type
        # Classic type
        if isinstance(obj, (mpl.lines.Line2D, mpl.image.AxesImage,
                            mpl.collections.Collection)):
            try:
                obj.remove()
            except ValueError:
                self.ax.add_artist(obj)
        # Contour type
        elif isinstance(obj, mpl.contour.QuadContourSet):
            try:
                for tmp_obj in obj.collections:
                    tmp_obj.remove()
            except ValueError:
                for tmp_obj in obj.collections:
                    self.ax.add_artist(tmp_obj)
        # Streamplot type
        elif isinstance(obj, mpl.streamplot.StreamplotSet):
            try:
                obj.lines.remove()
                if not hasattr(obj.arrows, 'patches'):
                    obj.arrows.patches = self.ax.patches
                # TODO : memory leak
                #       (because arrows removing is not implemented)
                self.ax.patches = []
            except ValueError:
                self.ax.add_artist(obj.lines)
                self.ax.patches = obj.arrows.patches
        else:
            raise ValueError("{}".format(obj))

    def draw(self, i=None, ax=None, cb=False, remove_current=False,
             rescale=True):
        # Do nothing if I is too big...
        if i is not None:
            if i >= len(self.draws):
                warnings.warn("Indice too big")
                return self.draws[self.curr_ind]
        self.curr_ind = i
        # check data
        if self.multidim and i is None:
            raise ValueError()
        if i is None:
            i = 0
        # remove current draw
        if (self.curr_draw is not None and
                self.draws[self.curr_draw] is not None and
                remove_current):
            self._toggle_visibility(self.draws[self.curr_draw])
        if self.use_buffer:
            # if draw already computed
            if self.draws[i] is not None:
                self._toggle_visibility(self.draws[i])
                self.curr_draw = i
            # else make a new one
            else:
                self.curr_draw = i
                self.draws[i] = self.draw_new(i=i, ax=ax, cb=cb,
                                              rescale=rescale)
            # keep trace of saved displ
            # (deleting first ones if too much of them)
            self.displ_saved_inds[i] = self.displ_saved_curr_ind
            self.displ_saved_curr_ind += 1
            if np.sum(self.displ_saved_inds != 0) > self.max_saved_displ:
                # delete first displ
                ind = np.min(self.displ_saved_inds[self.displ_saved_inds != 0])
                ind = np.where(ind == self.displ_saved_inds)[0]
                if len(ind) != 0:
                    ind = ind[0]
                    self.draws[ind] = None
                    self.displ_saved_inds[ind] = 0.
            # returning
            return self.draws[i]
        else:
            # returning
            return self.draw_new(i=i, ax=ax, cb=cb, rescale=rescale)

    def draw_new(self, i=None, ax=None, cb=False, rescale=True):
        # check data
        if self.multidim and i is None:
            raise ValueError()
        if i is None:
            i = 0
        # draw
        if ax is None and self.ax is None:
            if self.fig is None:
                self.fig = plt.gcf()
            self.ax = self.fig.gca()
            ax = self.ax
        elif ax is None:
            ax = self.ax
        else:
            self.ax = ax
        tmp_x, tmp_y, tmp_values, tmp_colors, tmp_magn = self.get_data(i=i)
        # continue if there is nothing to draw
        if len(tmp_x) == 0:
            return None
        dargs = self.dargs.copy()
        kind = dargs.pop('kind')
        try:
            aspect = dargs.pop('aspect')
        except KeyError:
            aspect = None
        try:
            adjustable = dargs.pop('adjustable')
        except KeyError:
            adjustable = 'box'
        #
        if kind == 'scatter':
            if tmp_values is None:
                plot = ax.scatter(tmp_x, tmp_y, **dargs)
            else:
                if 'c' not in dargs.keys():
                    dargs['c'] = tmp_values
                plot = ax.scatter(tmp_x, tmp_y, **dargs)
        elif kind == 'plot':
            plot = ax.plot(tmp_x, tmp_y, **dargs)
        elif kind == 'colored_plot':
            plot = ax.plot = colored_plot(tmp_x, tmp_y, z=tmp_values, **dargs)
        elif kind == 'semilogx':
            tmp_filt = ~np.isnan(tmp_x)
            tmp_filt = np.logical_or(tmp_filt, tmp_x < 0)
            tmp_x[~tmp_filt] = np.nan
            plot = ax.semilogx(tmp_x, tmp_y, **dargs)
        elif kind == 'semilogy':
            tmp_filt = ~np.isnan(tmp_y)
            tmp_filt = np.logical_or(tmp_filt, tmp_y < 0)
            tmp_y[~tmp_filt] = np.nan
            plot = ax.semilogy(tmp_x, tmp_y, **dargs)
        elif kind == 'loglog':
            tmp_filt = ~np.isnan(tmp_x)
            tmp_filt = np.logical_or(tmp_filt, tmp_x < 0)
            tmp_x[~tmp_filt] = np.nan
            tmp_filt = ~np.isnan(tmp_y)
            tmp_filt = np.logical_or(tmp_filt, tmp_y < 0)
            tmp_y[~tmp_filt] = np.nan
            plot = ax.loglog(tmp_x, tmp_y, **dargs)
        elif kind == 'matrix':
            delta_x = tmp_x[1] - tmp_x[0]
            delta_y = tmp_y[1] - tmp_y[0]
            plot = ax.imshow(tmp_values.transpose(),
                             extent=(tmp_x[0] - delta_x/2.,
                                     tmp_x[-1] + delta_x/2.,
                                     tmp_y[0] - delta_y/2.,
                                     tmp_y[-1] + delta_y/2.),
                             origin='lower', **dargs)
        elif kind == "contour":
            plot = ax.contour(tmp_x, tmp_y, tmp_values.transpose(), **dargs)
        elif kind == "contourf":
            plot = ax.contourf(tmp_x, tmp_y, tmp_values.transpose(), **dargs)
        elif kind == "quiver":
            if 'color' in list(dargs.keys()):
                C = dargs.pop('color')
                if 'c' in dargs.keys():
                    dargs.pop('c')
            else:
                C = tmp_magn
                if 'c' in dargs.keys():
                    dargs.pop('c')
            plot = ax.quiver(tmp_x, tmp_y, tmp_values[0].transpose(),
                             tmp_values[1].transpose(), C.transpose(),
                             **dargs)
        elif kind == "stream":
            # set adptative linewidth
            if 'lw' in list(dargs.keys()):
                tmp_lw = dargs.pop('lw')
            elif 'linewidth' in list(dargs.keys()):
                tmp_lw = dargs.pop('linewidth')
            else:
                tmp_lw = 1
            if np.array(tmp_lw).shape != ():
                pass
            else:
                tmp_magn[np.isnan(tmp_magn)] = 0
                tmp_lw *= 0.1 + 0.9*tmp_magn/np.max(tmp_magn)
                tmp_lw = tmp_lw.transpose()
            # set color
            if np.array(tmp_colors).shape != ():
                tmp_colors = tmp_colors.transpose()
            # plot
            Vx = tmp_values[0].transpose()
            Vy = tmp_values[1].transpose()
            plot = ax.streamplot(tmp_x, tmp_y, Vx, Vy, color=tmp_colors,
                                 linewidth=tmp_lw, **dargs)
        else:
            raise Exception("Unknown kind of plot : {}".format(kind))
        if aspect is not None:
            try:
                self.ax.set_aspect(aspect, adjustable=adjustable)
            except:
                pass
        if rescale:
            if self.data_type in ['field_1D', 'field_2D']:
                self.ax.relim(visible_only=True)
                self.ax.autoscale_view(tight=True)        # set up the apsect
                self.ax.set_xlim(*self.xlim)
                self.ax.set_ylim(*self.ylim)
            else:
                # self.ax.relim(visible_only=True)
                self.ax.autoscale_view(tight=True)        # set up the apsect
                # self.ax.set_xlim(*self.xlim)
                # self.ax.set_ylim(*self.ylim)
        return plot

    def draw_multiple(self, inds, sharecb=False, sharex=False, sharey=False,
                      ncol=None, nrow=None):
        nmb_fields = len(inds)
        # get figure
        fig = plt.gcf()
        if len(fig.axes) == 0 or len(fig.axes) != len(inds):
            # creating new figure with subplots
            if ncol is None:
                ncol = int(np.sqrt(nmb_fields))
            if nrow is None:
                nrow = int(np.ceil(float(nmb_fields)/ncol))
            if ncol*nrow < len(inds):
                raise ValueError()
            # creating axes
            fig, axs = plt.subplots(nrows=nrow, ncols=ncol, sharex=sharex,
                                    sharey=sharey)
        elif len(fig.axes) == len(inds):
            # reuse the current axes
            axs = np.array(fig.axes)
        # getting min and max
        if sharecb:
            if 'norm' not in list(self.dargs.keys()):
                if "vmin" not in list(self.dargs.keys()):
                    vmin = np.min([np.min(self.colors[ind]) for ind in inds])
                else:
                    vmin = self.dargs.pop('vmin')
                if "vmax" not in list(self.dargs.keys()):
                    vmax = np.max([np.max(self.colors[ind]) for ind in inds])
                else:
                    vmax = self.dargs.pop('vmax')
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
                self.dargs['norm'] = norm
        # displaying the wanted fields
        for i, ind in enumerate(inds):
            ax = axs.flat[i]
            plt.sca(ax)
            self.draw(ind, ax=ax, rescale=True, remove_current=False)
            DataCursorTextDisplayer(self, i=ind)
        # deleting the non-wanted axes
        for ax in axs.flat[nmb_fields::]:
            fig.delaxes(ax)
        plt.tight_layout()
        # same colorbar
        if sharecb:
            vmin = self.dargs['norm'].vmin
            vmax = self.dargs['norm'].vmax
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.05, .025, 0.925])
            mpl.colorbar.ColorbarBase(cbar_ax, norm=norm,
                                      orientation='vertical')
            plt.tight_layout(rect=[0., 0., 0.85, 1.])
        # ensure correct ax for pyplot
        plt.sca(axs.flatten()[nmb_fields - 1])
        # returning
        return fig, axs


class DataCursorTextDisplayer(object):

    def __init__(self, displayer, i=None, precision=3):
        if i is None:
            self.displ_ind = None
        else:
            self.displ_ind = i
        self.displayer = displayer
        self.displayer.ax.format_coord = self.__default_formatter
        self.precision = precision

    def __default_formatter(self, x, y):
        data = self.displayer.get_data_at_point(x, y, i=self.displ_ind)
        if data is None:
            return ''
        text = ''
        for key in list(data.keys()):
            text += "{}=".format(key)
            text += self.__number_formatter(data[key])
            text += "     "
        return text

    def __number_formatter(self, number):
        # if string
        try:
            number.format()
            return number
        except AttributeError:
            pass
        if int(number) == number:
            return "{}".format(int(number))
        else:
            return "{:.4g}".format(number)

    def __get_data_from_ax(self, x, y):
        return self.displayer.get_data_at_point(x, y)


class DataCursorPoints(object):

    def __init__(self, ax, tolerance=5, offsets=(-20, 20),
                 formatter=None, display_all=False,
                 color=(0.76, 0.86, 0.92)):
        """
        A simple data cursor widget that displays the x,y location of a
        matplotlib artist when it is selected.

        Parameters
        ----------
        artists : sequence of matplotlib Artists
            is the matplotlib artist or sequence of artists that will be
            selected.
        tolerance : integer
            is the radius (in points) that the mouse click must be
            within to select the artist.
        offsets : 2x1 tuple of integer
            is a tuple  of (x,y) offsets in points from the selected
            point to the displayed annotation box
        formatter : function
            is a callback function which takes 2 numeric arguments and
            returns a string
        display_all : boolean
            controls whether more than one annotation box will
            be shown if there are multiple axes.  Only one will be shown
            per-axis, regardless.
        color : matplotlib color
            color of the information box

        Notes:
        ------
        Credit to
        http://stackoverflow.com/a/4674445/190597 (Joe Kington)
        http://stackoverflow.com/a/20637433/190597 (unutbu)
        """
        raise Exception("Not (properly) implemented yet")
        # self._points = np.column_stack((x, y))
        if formatter is None:
            self.formatter = self._default_fmt
        else:
            self.formatter = formatter
        self.offsets = offsets
        self.display_all = display_all
        # try:
        #     artists[0]
        # except TypeError:
        #     artists = [artists]
        # self.artists = artists
        self.axes = tuple(set(art.axes for art in self.artists))
        self.figures = tuple(set(ax.figure for ax in self.axes))
        self.current_displayed_xy = []
        self.annotations = {}
        self.color = color
        for ax in self.axes:
            self.annotations[ax] = self.annotate(ax)
        for artist in self.artists:
            artist.set_picker(tolerance)
        for fig in self.figures:
            fig.canvas.mpl_connect('pick_event', self)

    def annotate(self, ax):
        """
        Draws and hides the annotation box for the given axis "ax".
        """
        annotation = ax.annotate(self.formatter, xy=(0, 0), ha='right',
                                 xytext=self.offsets,
                                 textcoords='offset points', va='bottom',
                                 bbox=dict(boxstyle='round,pad=0.5',
                                           fc=self.color, alpha=1.),
                                 arrowprops=dict(arrowstyle='->',
                                                 connectionstyle='arc3,rad=0'))
        annotation.set_visible(False)
        return annotation

    def snap(self, x, y):
        """
        Return the value in self._points closest to (x, y).
        """
        idx = np.nanargmin(((self._points - (x, y))**2).sum(axis=-1))
        return self._points[idx]

    def _default_fmt(self, x, y):
        return 'x: {x:0.2f}\ny: {y:0.2f}'.format(x=x, y=y)

    def get_color_from_event(self, event):
        # get color from scatterplots
        color = None
        try:
            x, y = event.mouseevent.xdata, event.mouseevent.ydata
            x, y = self.snap(x, y)
            colors = event.artist.get_facecolor()
            color = colors[np.logical_and(self._points[:, 0] == x,
                                          self._points[:, 1] == y)][0]
        except:
            pass
        # get color from classic plots
        try:
            color = event.artist.get_color()
            color = mpl.colors.colorConverter.to_rgba_array(color)[0]
        except:
            pass
        # return
        return color

    def __call__(self, event):
        """
        Intended to be called through "mpl_connect".
        """
        x, y = event.mouseevent.xdata, event.mouseevent.ydata
        x, y = self.snap(x, y)
        # get point color to color annotation
        annotation = self.annotations[event.artist.axes]
        color = self.get_color_from_event(event)
        color = np.array(color)
        color = color + 0.75*(1 - color)
        annotation.set_backgroundcolor(color)
        if x is not None:
            if not self.display_all:
                # Hide any other annotation boxes...
                for ann in list(self.annotations.values()):
                    ann.set_visible(False)
            # Update the annotation in the current axis..
            # if already annotated, remove the annotation
            if [x, y] in self.current_displayed_xy:
                annotation.set_visible(False)
                self.current_displayed_xy.remove([x, y])
            else:
                annotation.xy = x, y
                annotation.set_text(self.formatter(x, y))
                annotation.set_visible(True)
                self.current_displayed_xy.append([x, y])
            event.canvas.draw()
