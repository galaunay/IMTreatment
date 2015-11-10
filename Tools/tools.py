# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 18:50:02 2015

@author: glaunay
"""
from __future__ import print_function
import time as modtime
import numpy as np
import shutil
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pdb
from os.path import join
import json
import copy
from ..core import ARRAYTYPES, STRINGTYPES, NUMBERTYPES
from matplotlib.collections import LineCollection
import re
import scipy.interpolate as spinterp
try:
    from multiprocess import Pool, cpu_count, Value, Process, Manager
    MULTIPROCESSING = True
except ImportError:
    MULTIPROCESSING = False


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
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in xrange(2)]
        self.save = os.dup(1), os.dup(2)
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, type, value, traceback):
        os.dup2(self.save[0], 1)
        os.dup2(self.save[1], 2)
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


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
    if color_label is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # fake up the array of the scalar mappable. Urgh...
        sm._A = []
        cb = plt.colorbar(sm)
        cb.set_label(color_label)
    return lc

class Files(object):
    
    def __init__(self):
        """
        Class representing a bunch of files (and/or folders)
        """
        self.paths = []
        self.exist = []
        self.isdir = []
 
    def __add__(self, obj):
        if isinstance(obj, Files):
            tmp_files = self.copy()
            tmp_files.paths += obj.paths
            tmp_files.exist += obj.exist
            tmp_files.isdir += obj.isdir
            return tmp_files
    
    def __repr__(self):
        # build tree
        self.build_tree()
        # use json to parse (not optimal...)
        text = json.dumps(self.tree, indent=2)
        text = re.sub('{\n', "\n", text)
        text = re.sub('":\s', '"', text)
        text = re.sub("\[", "", text)
        text = re.sub("\]\n", "", text)
        text = re.sub("\],\s\n", "\n", text)
        text = re.sub("\},\s\n", "\n", text)
        text = re.sub("}\n", "", text)
        text = re.sub("}$", "", text)
        text = re.sub('"', "", text)
        text = re.sub(',\s\n', "\n", text)
        text = re.sub('{\n', "\n", text)
        text = re.sub('{[\s]*$', "", text)
        text = re.sub('{', "", text)
        return text        

    
    def copy(self):
        return copy.deepcopy(self)
   
    def add_file(self, path):
        # check argument type
        try:
            path = unicode(path)
        except:
            raise TypeError()
        # check if valid path or not
        path = os.path.normpath(path)
        if os.path.exists(path):
            self.paths.append(path)
            self.exist.append(True)
            if os.path.isdir(path):
                self.isdir.append(True)
            else:
                self.isdir.append(False)
        elif os.access(os.path.dirname(path), os.W_OK):
            self.paths.append(path)
            self.exist.append(False)
            self.isdir.append(None)
        else:
            raise Exception()
            
    def remove_file(self, ind):
        del self.paths[ind]
        del self.exist[ind]
        del self.isdir[ind]
        
    def load_files_from_regex(self, rootpath, regex, load_files=True, 
                              load_dirs=True, depth='all'):
        """
        Load existing files from a regular expression.
        """
        # get folder names
        paths = []
        isdir = []
        exist = []
        rootpath = os.path.normpath(rootpath)
        for root, dirs, files in os.walk(rootpath, topdown=True):
            # check depth 
            if depth != "all":
                tmp_depth = (root[len(rootpath) + len(os.path.sep):]
                             .count(os.path.sep))
                if depth < tmp_depth:
                    break
            # load direcories
            if load_dirs:
                for d in dirs:
                    fullpath = join(root, d)
                    if re.match(regex, fullpath):
                        paths.append(fullpath)
                        isdir.append(True)
                        exist.append(True)
            # load files
            if load_files:
                for f in files:
                    fullpath = join(root, f)
                    if re.match(regex, fullpath):
                        paths.append(fullpath)
                        isdir.append(False)
                        exist.append(True)
        # store
        self.paths += paths
        self.isdir += isdir
        self.exist += exist
               
    def build_tree(self):
        """
        Build a file tree
        """
        dic = {}
        for isdir, path in zip(self.isdir, self.paths):
            sep_path  = path.split(os.path.sep)
            tmp_dic = dic
            sep_path_len = len(sep_path)
            for i, fold in enumerate(sep_path):
                # folders
                if isdir or i != sep_path_len - 1:
                    if fold in tmp_dic.keys():
                        tmp_dic = tmp_dic[fold]
                    else:
                        new_dic = {}
                        tmp_dic[fold] = new_dic
                        tmp_dic = new_dic
                # files ()
                else:
                    if 'files' in tmp_dic.keys():
                        tmp_dic['files'].append(fold)
                    else:
                        tmp_dic['files'] = [fold]
        # store
        self.tree = dic
        
    def delete_existing_files(self, recursive=False):
        tmp_isdir = list(np.array(self.isdir)[np.array(self.exist)])
        tmp_paths = list(np.array(self.paths)[np.array(self.exist)])
        nmb_dir = np.sum(tmp_isdir)
        nmb_files = len(tmp_paths) - nmb_dir
        print("+++ Ready to remove {} files and {} directories "
              .format(nmb_files, nmb_dir))
        while True:
            rep = raw_input("+++ Okay with that ? ('o', 'n') \n+++ ")
            if rep in ['o', 'O', 'y', 'Y', 'oui', 'Oui', 'Yes', 'yes', 'YES', 'OUI']:
                rep = True
                break
            elif rep in ['n', 'N', 'No', 'no', 'non', 'Non', 'NON', 'NO']:
                rep = False
                break
        # remove if necessary
        if rep:
            print("")
            PG = ProgressCounter("Begin cleaning", "Done", nmb_files,
                                 name_things='files', perc_interv=10)
            # remove files                    
            for i, p in enumerate(np.array(tmp_paths)):
                if tmp_isdir[i]:
                    continue
                PG.print_progress()
                os.remove(p)
                
            # remove dirs
            for i, p in enumerate(np.array(tmp_paths)):
                if tmp_isdir[i]:
                    # force recursive removing
                    if recursive:
                        shutil.rmtree(p)
                    else:
                        # else, check if each folder is empty
                        try:
                            os.rmdir(p)
                        except WindowsError:
                            print("+++ Following folder is not empty\n"
                                  "{}".format(p))
                            while True:
                                rep = raw_input("+++ Delete anyway ? ('o', 'n') \n+++ ")
                                if rep in ['o', 'O', 'y', 'Y', 'oui', 'Oui',
                                           'Yes', 'yes', 'YES', 'OUI']:
                                    rep = True
                                    break
                                elif rep in ['n', 'N', 'No', 'no', 'non',
                                             'Non', 'NON', 'NO']:
                                    rep = False
                                    break
                            if rep is True:
                                shutil.rmtree(p)
            # 
            for i in range(len(self.exist)):
                self.exist[i] = False
            
        
            
        
    
def remove_files_in_dirs(rootpath, dir_regex, file_regex,
                         depth='all', remove_dir=False, remove_files=True):
    """
    make a recursive search for directories satisfying "dir_regex'
    from "rootpath", and remove all the files satisfying 'file_regex' in it.
    
    Parameters
    ----------
    rootpath : string
        Path where to begin searching.
    dir_regex : string
        Regular expression matching the directory where we want to remove
        stuff.
    file_regex : string
        Regular expression matching the files we want to delete.
    depth : integer or 'all'
        Number of directory layer to go through.
    remove_dir : 
    

    
    """
    # ### TODO : add the possibility to remove empty directories
    # get dirs
    dir_paths = []
    nmb_files = []
    file_paths = []
    nmb_tot_files = 0
    for root, dirs, files in os.walk(rootpath):
        nmb_tot_files += len(files)
        if re.match(dir_regex, root):
            tmp_nmb_files = 0
            for f in files:
                if re.match(file_regex, f):
                    tmp_nmb_files += 1
                    file_paths.append(os.path.join(root, f))
            # check if there is actuelly files in there
            if not tmp_nmb_files == 0:
                nmb_files.append(tmp_nmb_files)
                dir_paths.append(root)
    # ask before deletion
    print("")
    print("+++ Checked {} files".format(nmb_tot_files))
    if np.sum(nmb_files) == 0:
        print("+++ Nothing to delete")
        return None
    print("+++ Ready to remove {} files in directories :".format(np.sum(nmb_files)))
    for i in range(len(dir_paths)):
        print("+++    [{} files] {}".format(nmb_files[i], dir_paths[i]))
    while True:
        rep = raw_input("+++ Okay with that ? ('o', 'n') \n+++ ")
        if rep in ['o', 'O', 'y', 'Y', 'oui', 'Oui', 'Yes', 'yes', 'YES', 'OUI']:
            rep = True
            break
        elif rep in ['n', 'N', 'No', 'no', 'non', 'Non', 'NON', 'NO']:
            rep = False
            break
    # remove if necessary
    if rep:
        print("")
        PG = ProgressCounter("Begin cleaning", "Done", len(file_paths),
                             name_things='files', perc_interv=10)
        for p in file_paths:
            PG.print_progress()
            os.remove(p)





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    