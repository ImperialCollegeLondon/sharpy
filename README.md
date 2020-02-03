# Simulation of High Aspect Ratio aeroplanes in Python [SHARPy]

![Version badge](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Fraw.githubusercontent.com%2FImperialCollegeLondon%2Fsharpy%2Fmaster%2F.version.json)
[![Build Status](https://travis-ci.org/ImperialCollegeLondon/sharpy.svg?branch=master)](https://travis-ci.org/ImperialCollegeLondon/sharpy)
[![Documentation Status](https://readthedocs.org/projects/ic-sharpy/badge/?version=master)](https://ic-sharpy.readthedocs.io/en/master/?badge=master)
[![codecov](https://codecov.io/gh/ImperialCollegeLondon/sharpy/branch/master/graph/badge.svg)](https://codecov.io/gh/ImperialCollegeLondon/sharpy)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![status](https://joss.theoj.org/papers/f7ccd562160f1a54f64a81e90f5d9af9/status.svg)](https://joss.theoj.org/papers/f7ccd562160f1a54f64a81e90f5d9af9)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3531965.svg)](https://doi.org/10.5281/zenodo.3531965)

SHARPy is a nonlinear aeroelastic analysis package developed at the Department of Aeronautics, Imperial
College London. It can be used for the structural, aerodynamic and aeroelastic analysis of flexible aircraft, flying
wings and wind turbines.

![XHALE](./docs/source/_static/XHALE-render.jpg)

### Contact

For more information on the [research team](http://www.imperial.ac.uk/aeroelastics/people/) developing SHARPy or to get 
in touch, [visit our homepage](http://www.imperial.ac.uk/aeroelastics).

## Physical Models

SHARPy is a modular aeroelastic solver that currently uses two specific models for the structural and aerodynamic response of the system.

For the structural model, SHARPy employs a nonlinear, geometrically-exact displacement and rotation-based composite beam formulation,
augmented with Lagrange multipliers for additional kinematic constraints.
This model has the advantage of providing the solution directly in the physical problem's degrees of freedom, making the 
coupling with the aerodynamic solver simple and not requiring any post-processing. The 1D beam formulation used limits 
the analyses that can be done by SHARPy to slender structures, such as high aspect ratio wings.

The aerodynamic model utilises the Unsteady Vortex Lattice Method (UVLM). The aerodynamic surfaces are modelled as a thin
vortex ring lattice with the boundary conditions enforced at the collocation points in the middle of the vortex rings.
The Kutta condition is also enforced at the trailing edge. The wake can be simulated by either additional vortex rings
or by infinitely long horseshoe vortices, which are ideally suited for steady simulations only.

The input problems can be structural, aerodynamic or coupled, yielding an aeroelastic system.

## [Capabilities](http://ic-sharpy.readthedocs.io/en/latest/content/capabilities.html)

At its core, SHARPy is a nonlinear aeroelastic analysis package that can be used on both free-flying aircraft,
clamped structures and wind turbines. In addition, it supports linearisation of these nonlinear systems about
arbitrary conditions and includes various tools such as: model reduction or frequency analysis.

In short, SHARPy offers (amongst others) the following solutions to the user:
* Static aerodynamic, structural and aeroelastic solutions
* Finding trim conditions for aeroelastic configurations
* Nonlinear, dynamic time domain simulations under a large number of conditions such as:
    + Prescribed trajectories.
    + Free flight.
    + Dynamic follower forces.
    + Control inputs in thrust, control surface deflection...
    + Arbitrary time-domain gusts, including non span-constant ones.
    + Full 3D turbulent fields.
* Multibody dynamics with hinges, articulations and prescribed nodal motions:
    + Applicable to wind turbines.
    + Hinged aircraft.
    + Catapult assisted takeoffs.
* Linear analysis:
    + Linearisation around a nonlinear equilibrium.
    + Frequency response analysis.
    + Asymptotic stability analysis.
* Model order reduction:
    + Krylov-subspace reduction methods.
    + Balancing reduction methods.

## Documentation

The documentation for SHARPy can be found [here](http://ic-sharpy.readthedocs.io).

## Installing SHARPy

For the latest documentation, see the 
[installation docs](https://ic-sharpy.readthedocs.io/en/latest/content/installation.html).

As of version v1.0.1, SHARPy can also be obtained from Docker Hub to avoid compilation
and platform-dependant issues. If you are interested, make sure you check
the [SHARPy Docker distribution docs](https://ic-sharpy.readthedocs.io/en/latest/content/installation.html#using-sharpy-from-a-docker-container).

## Contributing and Bug reports

If you think you can add a useful feature to SHARPy, want to write documentation or you encounter a bug, by all means, 
check out the [collaboration guide](https://ic-sharpy.readthedocs.io/en/latest/content/contributing.html).

## Citing SHARPy

SHARPy has been published in the Journal of Open Source Software (JOSS) and the relevant paper can be found
[here](https://joss.theoj.org/papers/10.21105/joss.01885).

If you are using SHARPy for your work, please remember to cite it using the paper in JOSS as:

`del Carre et al., (2019). SHARPy: A dynamic aeroelastic simulation toolbox for very flexible aircraft and wind
turbines. Journal of Open Source Software, 4(44), 1885, https://doi.org/10.21105/joss.01885`

The bibtex entry for this citation is:

```
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
```


## Continuous Integration Status

SHARPy uses Continuous Integration to control the integrity of its code. The status in the release and develop branches
is:

Master
[![Build Status](https://travis-ci.org/ImperialCollegeLondon/sharpy.svg?branch=master)](https://travis-ci.org/ImperialCollegeLondon/sharpy)

Develop
[![Build Status](https://travis-ci.org/ImperialCollegeLondon/sharpy.svg?branch=develop)](https://travis-ci.org/ImperialCollegeLondon/sharpy)
