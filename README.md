IMTreatment - A fields study package
====================================
[![Build status](https://gitlab.com/gitlab-org/gitlab-ce/badges/master/build.svg)](https://gitlab.com/gabylaunay/IMTreatment/commits/master)
[![Overall test coverage](https://gitlab.com/gitlab-org/gitlab-ce/badges/master/coverage.svg)](https://framagit.org/gabylaunay/IMTreatment/pipelines)


This module has been written to carry out analysis and more
specifically structure detection on PIV velocity fields. It is now
more general and can handle different kind of data (point cloud,
scalar and vector field, ...) and perform classical and more advanced
analysis on them (spectra, pod, post-processing, visualization, ...).

General data analysis
---------------------
  1. Class representing 2D fields of 1 component (`ScalarField`)
  2. Class representing 2D fields of 2 components (`VectorField`)
  3. Classes representing sets of scalar fields vector fields
     (`SpatialScalarFields`, `TemporalScalarFields`,
      `SpatialVectorFields` and `TemporalVectorFields`)
  4. Class representing profiles (`Profile`)
  5. Class representing scatter points (`Points`)
  6. Module for modal decomposition (POD, DMD) and reconstruction (`pod`)
  7. Module to import/export data from/to Davis, matlab, ascii, pivmat and
     images files (`file_operation`)
  8. Functionalities to visualize those data (`plotlib`)

Flow analysis
-------------
  1. Module to create artificial vortices: Burger, Jill, Rankine, ...
     and to simulate their motion in potential flows (`vortex_creation`)
  2. Module providing several vortex criterions computation
     (`vortex_criterions`)
  3. Module to automatically detect and track critical points
     (`vortex_detection`)
  4. Module to compute the evolution of some vortex properties
     (`vortex_properties`)
  5. Module to generate potential flows with arbitrary geometries
     (`potential_flow`)

Dependencies
------------
Mandatory:
- numpy
- matplotlib
- scipy
- unum
- modred

Optional:
- sklearn (to work with point clustering)
- networkx (to use force-directed algorithms to compare trajectories)
- colorama (to have a nice interface when manipulating files)
- h5py (allow to import data from [pivmat](http://www.fast.u-psud.fr/pivmat/) files)

Installation
------------
A good old `python setup.py install`should do the trick and install
the necessary dependencies.

You can try to run the tests with `run_tests.sh`,
but the test suite is not plateform-independent yet,
and should fail miserably.

If you intend to modify this package, store it some place safe and 
install it as a developement package with `python setup.py develop`.
