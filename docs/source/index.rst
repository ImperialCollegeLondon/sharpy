.. SHARPy documentation master file, created by
   sphinx-quickstart on Wed Oct 19 16:40:48 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SHARPy Documentation
====================


.. image:: https://img.shields.io/endpoint.svg?url=https%3A%2F%2Fraw.githubusercontent.com%2FImperialCollegeLondon%2Fsharpy%2Fmaster%2F.version.json

.. image:: https://codecov.io/gh/ImperialCollegeLondon/sharpy/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/ImperialCollegeLondon/sharpy

.. image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :target: https://opensource.org/licenses/BSD-3-Clause

.. image:: https://readthedocs.org/projects/ic-sharpy/badge/?version=master

.. image:: https://joss.theoj.org/papers/f7ccd562160f1a54f64a81e90f5d9af9/status.svg
   :target: https://joss.theoj.org/papers/f7ccd562160f1a54f64a81e90f5d9af9

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3531966.svg
   :target: https://doi.org/10.5281/zenodo.3531966

Welcome to SHARPy (Simulation of High Aspect Ratio aeroplanes in Python)!

SHARPy is an aeroelastic analysis package currently under development at the Department of Aeronautics,
Imperial College London. It can be used for the structural, aerodynamic, aeroelastic and flight dynamics
analysis of flexible aircraft, flying wings and wind turbines.

This site contains the available documentation of the software. The documentation project is still work in process and,
at the time of writing, only includes documentation for the Python modules of SHARPy, which encompass the aeroelastic
solvers and interfaces to the structural and aerodynamic modules.
The UVLM and structural modules are written in C++ and Fortran, respectively, and do not have
detailed documentation yet.

The objective of this documentation package is to show the user how to run cases by defining the geometry, flow
conditions and desired solution process.

Contents
--------

.. toctree::
   :maxdepth: 1

   content/installation
   content/sharpy_intro
   content/contributing
   content/casefiles
   content/examples
   content/solvers
   content/postproc
   includes/index
   content/test_cases

Citing SHARPy
-------------

SHARPy is archived in Zenodo and has two unique DOIs that can be used depending on whether you'd like to
cite a specific SHARPy version or all versions of SHARPy (i.e. this is the concept DOI).

**Version DOI:**

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3531966.svg
   :target: https://doi.org/10.5281/zenodo.3531966

**Concept DOI:**

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3531965.svg
   :target: https://doi.org/10.5281/zenodo.3531965

For more information on citing and Zenodo, read more_.

.. _more: https://help.zenodo.org/#versioning

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Contact
-------

SHARPy is developed at the Department of Aeronautics, Imperial College London. To get in touch, visit the `Loads Control
and Aeroelastics Lab <http://imperial.ac.uk/aeroelastics>`_ website.
