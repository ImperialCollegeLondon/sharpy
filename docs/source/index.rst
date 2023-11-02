.. SHARPy documentation master file, created by
   sphinx-quickstart on Wed Oct 19 16:40:48 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Simulation of High Aspect Ratio planes in Python [SHARPy]
=========================================================


.. image:: https://img.shields.io/endpoint.svg?url=https%3A%2F%2Fraw.githubusercontent.com%2FImperialCollegeLondon%2Fsharpy%2Fmaster%2F.version.json

.. image:: https://codecov.io/gh/ImperialCollegeLondon/sharpy/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/ImperialCollegeLondon/sharpy

.. image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :target: https://opensource.org/licenses/BSD-3-Clause

.. image:: https://readthedocs.org/projects/ic-sharpy/badge/?version=main

.. image:: https://joss.theoj.org/papers/f7ccd562160f1a54f64a81e90f5d9af9/status.svg
   :target: https://joss.theoj.org/papers/f7ccd562160f1a54f64a81e90f5d9af9

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3531965.svg
   :target: https://doi.org/10.5281/zenodo.3531965

Welcome to SHARPy (Simulation of High Aspect Ratio aeroplanes in Python)!

SHARPy is an aeroelastic analysis package currently under development at the Department of Aeronautics,
Imperial College London. It can be used for the structural, aerodynamic, aeroelastic and flight dynamics
analysis of flexible aircraft, flying wings and wind turbines. Amongst other capabilities_, it offers the following solutions to the user:

* Static aerodynamic, structural and aeroelastic solutions including fuselage effects
* Finding trim conditions for aeroelastic configurations
* Nonlinear, dynamic time domain simulations under a large number of conditions such as:

    + Prescribed trajectories.
    + Free flight.
    + Dynamic follower forces.
    + Control inputs in thrust, control surface deflection...
    + Arbitrary time-domain gusts, including non span-constant ones.
    + Full 3D turbulent fields.

* Multibody dynamics with hinges, articulations and prescribed nodal motions.

    + Applicable to wind turbines.
    + Hinged aircraft.
    + Catapult assisted takeoffs.

* Linear analysis

    + Linearisation around a nonlinear equilibrium.
    + Frequency response analysis.
    + Asymptotic stability analysis.

* Model order reduction

    + Krylov-subspace reduction methods.
    + Balancing reduction methods.

The modular design of SHARPy allows to simulate complex aeroelastic cases involving very flexible aircraft.
The structural solver supports very complex beam arrangements, while retaining geometrical nonlinearity.
The UVLM solver features different wake modelling fidelities while supporting large lifting surface deformations in a
native way. Detailed information on each of the solvers is presented in their respective documentation packages.

Contents
--------

.. toctree::
   :maxdepth: 1

   content/installation
   content/capabilities
   content/publications
   content/examples
   content/contributing
   content/casefiles
   content/solvers
   content/postproc
   includes/index
   content/test_cases
   content/debug
   content/faqs

Citing SHARPy
-------------
SHARPy has been published in the Journal of Open Source Software (JOSS) and the relevant paper can be found
here_.

If you are using SHARPy for your work, please remember to cite it using the paper in JOSS as:

    del Carre et al., (2019). SHARPy: A dynamic aeroelastic simulation toolbox for very flexible aircraft and wind
    turbines. Journal of Open Source Software, 4(44), 1885, https://doi.org/10.21105/joss.01885

The bibtex entry for this citation is:

.. code-block:: none

    @Article{delCarre2019,
    doi = {10.21105/joss.01885},
    url = {https://doi.org/10.21105/joss.01885},
    year = {2019},
    month = dec,
    publisher = {The Open Journal},
    volume = {4},
    number = {44},
    pages = {1885},
    author = {Alfonso del Carre and Arturo Mu{\~{n}}oz-Sim\'on and Norberto Goizueta and Rafael Palacios},
    title = {{SHARPy}: A dynamic aeroelastic simulation toolbox for very flexible aircraft and wind turbines},
    journal = {Journal of Open Source Software}
    }

.. _here: https://joss.theoj.org/papers/10.21105/joss.01885


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Contact
-------

SHARPy is developed at the Department of Aeronautics, Imperial College London. To get in touch, visit the `Loads Control
and Aeroelastics Lab <http://imperial.ac.uk/aeroelastics>`_ website.

.. _capabilities: ./content/capabilities.html
