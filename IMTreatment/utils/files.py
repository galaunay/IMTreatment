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

import warnings
import numpy as np
import shutil
import os
from os.path import join
import copy
import re
from ..utils.types import ARRAYTYPES
from ..utils.progresscounter import ProgressCounter

try:
    import colorama
except ImportError:
    colorama = None

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
        text = self.get_tree_representation(max_file_list=10, hide_top=True)
        return text

    def get_tree_representation(self, max_file_list=10, hide_top=False):
        self.build_tree()
        # properties
        tab_color = ""
        folder_color = ""
        file_color = ""
        file_number_color = ""
        tab0 = "|"
        tab1 = tab0 + ">" + tab0
        tab2 = tab0 + " "
        separator = "/"
        heading_separator = "="
        if colorama is not None:
            tab_color = colorama.Fore.BLACK + colorama.Style.BRIGHT
            tab0 = tab_color + tab0 + colorama.Style.NORMAL
            tab1 = tab_color + tab1 + colorama.Style.NORMAL
            tab2 = tab_color + tab2 + colorama.Style.NORMAL
            folder_color = colorama.Fore.BLACK
            file_color = colorama.Fore.GREEN
            file_number_color = colorama.Fore.CYAN
        folders_end_of_line = "\n"
        files_end_of_line = "\n"
        max_file_list = max_file_list

        # recursion function
        def get_info_for_folder(fold, tab1, tab2):
            if not isinstance(fold, dict):
                raise TypeError()
            # if too much files to display, display number of files
            if 'files' in list(fold.keys()):
                if len(fold['files']) > max_file_list:
                    yield (tab1 + file_number_color +
                           '[{} Files]'.format(len(fold['files'])) +
                           files_end_of_line)
            # loop recursively on folder's folders
            for key in list(fold.keys()):
                if key == 'files':
                    pass
                else:
                    yield tab1 + folder_color + key + folders_end_of_line
                    for thing in get_info_for_folder(fold[key], tab1, tab2):
                        yield tab2 + thing
            # loop on folder's files
            if 'files' in list(fold.keys()):
                if len(fold['files']) <= max_file_list:
                    for f in fold['files']:
                        yield tab1 + file_color + f + files_end_of_line
        # get
        text = ""
        tree = self.tree
        # add heading wih common  path
        if hide_top:
            curr_fold = self.tree
            while True:
                if isinstance(curr_fold, dict):
                    if len(list(curr_fold.keys())) == 1 and \
                            list(curr_fold.keys())[0] != "files":
                        text += list(curr_fold.keys())[0] + separator
                        curr_fold = curr_fold[list(curr_fold.keys())[0]]
                    else:
                        break
                else:
                    break
            # put a nice heading
            text = text[:-1]
            heading_sep = heading_separator*(len(text)+2)
            text = (tab0 + tab_color + heading_sep + tab0 +
                    folders_end_of_line +
                    tab0 + " " + text + " " + tab0 + folders_end_of_line +
                    tab_color + tab0 + tab_color + heading_sep + tab0 +
                    folders_end_of_line)
            # update folder
            tree = curr_fold
        # display the tree
        for thing in get_info_for_folder(tree, tab1=tab1,
                                         tab2=tab2):
            text += thing
        return text

    def copy(self):
        return copy.deepcopy(self)

    def add_file(self, path):
        # check argument type
        try:
            path = str(path)
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

    def remove_files(self, arg):
        """
        Remove some files from the files set.

        Parameters
        ----------
        arg : integer, regex or array of integer or regex
            If integer, remove the associated path,
            if a regex, remove the paths that match,
            if an array, delete paths for each element.
        """
        if isinstance(arg, int):
            ind = arg
            del self.paths[ind]
            del self.exist[ind]
            del self.isdir[ind]
        elif isinstance(arg, ARRAYTYPES):
            for thing in arg:
                self.remove_files(thing)
        elif isinstance(arg, str):
            for i in np.arange(len(self.paths) - 1, -1, -1):
                res = re.match(arg, self.paths[i])
                if res is not None:
                    print("remove {}".format(self.paths[i]))
                    self.remove_files(i)
        else:
            raise TypeError()

    def remove_empty_directories():
        """
        """
        pass

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
            sep_path = path.split(os.path.sep)
            tmp_dic = dic
            sep_path_len = len(sep_path)
            for i, fold in enumerate(sep_path):
                # folders
                if isdir or i != sep_path_len - 1:
                    if fold in list(tmp_dic.keys()):
                        tmp_dic = tmp_dic[fold]
                    else:
                        new_dic = {}
                        tmp_dic[fold] = new_dic
                        tmp_dic = new_dic
                # files ()
                else:
                    if 'files' in list(tmp_dic.keys()):
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
            rep = input("+++ Okay with that ? ('o', 'n') \n+++ ")
            if rep in ['o', 'O', 'y', 'Y', 'oui', 'Oui',
                       'Yes', 'yes', 'YES', 'OUI']:
                rep = True
                break
            elif rep in ['n', 'N', 'No', 'no', 'non',
                         'Non', 'NON', 'NO']:
                rep = False
                break
        # remove if necessary
        if rep:
            print("")
            PG = ProgressCounter(init_mess="Begin cleaning",
                                 nmb_max=nmb_files,
                                 name_things='files',
                                 perc_interv=10)
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
                                rep = input("+++ Delete anyway ? ('o', 'n') "
                                            "\n+++ ")
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
    warnings.warn("Deprecated, use 'Files' class instead")
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
    print("+++ Ready to remove {} files in directories :"
          .format(np.sum(nmb_files)))
    for i in range(len(dir_paths)):
        print("+++    [{} files] {}".format(nmb_files[i], dir_paths[i]))
    while True:
        rep = input("+++ Okay with that ? ('o', 'n') \n+++ ")
        if rep in ['o', 'O', 'y', 'Y', 'oui', 'Oui',
                   'Yes', 'yes', 'YES', 'OUI']:
            rep = True
            break
        elif rep in ['n', 'N', 'No', 'no', 'non', 'Non', 'NON', 'NO']:
            rep = False
            break
    # remove if necessary
    if rep:
        print("")
        PG = ProgressCounter(init_mess="Begin cleaning",
                             nmb_max=len(file_paths),
                             name_things='files',
                             perc_interv=10)
        for p in file_paths:
            PG.print_progress()
            os.remove(p)
