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

from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.rst"), "r") as fh:
    long_description = fh.read()

setup(
    name='IMTreatment',
    version='1.1.0',
    description='Tools to analyze scalar and vector fields',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://framagit.org/gabylaunay/IMTreatment',
    author='Gaby Launay',
    author_email='gaby.launay@protonmail.com',
    license='GPLv3',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3 :: Only',
        'Natural Language :: English',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    keywords='scalarfield vectorfield fluid dynamics PIV vortex',
    packages=find_packages(exclude=['contrib', 'docs', 'tests', 'samples']),
    install_requires=['numpy', 'matplotlib', 'scipy', 'unum',
                      'multiprocess', 'psutil'],
    extras_require={
        'Modal decomposition': 'modred',
        'Davis format support': 'ReadIM',
        'Point clustering': 'sklearn',
        'Mean trajectories clustering': 'networkx',
        'Color in console': 'colorama',
        'Pivmat format support': 'h5py',
        'Export to VTK format': 'pyvtk'},
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
