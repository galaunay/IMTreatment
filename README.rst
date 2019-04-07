====================================
IMTreatment - A fields study package
====================================
.. image:: https://gitlab.com/gitlab-org/gitlab-ce/badges/master/build.svg
   :target: https://gitlab.com/gabylaunay/IMTreatment/commits/master
   :alt: Build status
.. image:: https://gitlab.com/gitlab-org/gitlab-ce/badges/master/coverage.svg
   :target: https://framagit.org/gabylaunay/IMTreatment/pipelines
   :alt: Overall test coverage
.. image:: https://readthedocs.org/projects/imtreatment/badge/?version=latest
   :target: http://imtreatment.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status


This module has been written to carry out analysis and more specifically structure detection on PIV velocity fields.
It is now more general and can handle different kind of data (point cloud, scalar and vector field, ...) and perform classical and more advanced analysis on them (spectra, pod, post-processing, visualization, ...).

Hosted on FramaGit_.

Full documentation available on ReadTheDocs_.

.. _FramaGit: https://framagit.org/gabylaunay/IMTreatment/

General data analysis
---------------------

1. Class representing 2D fields of 1 component (ScalarField_)
2. Class representing 2D fields of 2 components (VectorField_)
3. Classes representing sets of scalar fields vector fields (SpatialScalarFields_, TemporalScalarFields_, SpatialVectorFields_ and TemporalVectorFields_)
4. Class representing profiles (Profile_)
5. Class representing scatter points (Points_)
6. Module for modal decomposition (POD, DMD) and reconstruction (pod_)
7. Module to import/export data from/to Davis, matlab, ascii, pivmat and images files (file_operation_)
8. Functionalities to visualize those data (plotlib_)

.. _ScalarField: http://imtreatment.readthedocs.io/en/latest/IMTreatment.core.scalarfield.html
.. _TemporalScalarFields: http://imtreatment.readthedocs.io/en/latest/IMTreatment.core.temporalscalarfields.html
.. _SpatialScalarFields: http://imtreatment.readthedocs.io/en/latest/IMTreatment.core.spatialscalarfields.html
.. _VectorField: http://imtreatment.readthedocs.io/en/latest/IMTreatment.core.vectorfield.html
.. _TemporalVectorFields: http://imtreatment.readthedocs.io/en/latest/IMTreatment.core.temporalvectorfields.html
.. _SpatialVectorFields: http://imtreatment.readthedocs.io/en/latest/IMTreatment.core.spatialvectorfields.html
.. _Points: http://imtreatment.readthedocs.io/en/latest/IMTreatment.core.points.html
.. _Profile: http://imtreatment.readthedocs.io/en/latest/IMTreatment.core.profile.html
.. _pod: http://imtreatment.readthedocs.io/en/latest/IMTreatment.pod.pod.html
.. _plotlib: http://imtreatment.readthedocs.io/en/latest/IMTreatment.plotlib.plotlib.html
.. _file_operation: http://imtreatment.readthedocs.io/en/latest/IMTreatment.file_operation.file_operation.html

Flow analysis
-------------

1. Module to create artificial vortices: Burger, Jill, Rankine, ... and to simulate their motion in potential flows (vortex_creation_)
2. Module providing several vortex criterions computation (vortex_criterions_)
3. Module to automatically detect and track critical points (vortex_detection_)
4. Module to compute the evolution of some vortex properties (vortex_properties_)
5. Module to generate potential flows with arbitrary geometries (potential_flow_)

.. _vortex_creation: http://imtreatment.readthedocs.io/en/latest/IMTreatment.vortex_creation.vortex_creation.html
.. _vortex_detection: http://imtreatment.readthedocs.io/en/latest/IMTreatment.vortex_detection.vortex_detection.html
.. _vortex_criterions: http://imtreatment.readthedocs.io/en/latest/IMTreatment.vortex_criterions.vortex_criterions.html
.. _vortex_properties: http://imtreatment.readthedocs.io/en/latest/IMTreatment.vortex_properties.vortex_properties.html
.. _potential_flow: http://imtreatment.readthedocs.io/en/latest/IMTreatment.potential_flow.potential_flow.html

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
- h5py (allow to import data from pivmat_ files)

.. _pivmat: http://www.fast.u-psud.fr/pivmat/

Installation
------------
IMTreatment is now available on Pypi:
``pip install IMTreatment``

Or if you prefer to install it manually from source:
``pip install 'git+https://framagit.org/gabylaunay/IMTreatment.git#egg=IMTreatment'``

Documentation
-------------
IMTreatment is documented inline and in ReadTheDocs_.
you can also use ``build_doc.sh`` to locally build the html doc.

.. _ReadTheDocs: http://imtreatment.readthedocs.io
