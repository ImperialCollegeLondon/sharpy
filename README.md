# Simulation of High Aspect Ratio Planes in Python [SHARPy]
![Version badge](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Fraw.githubusercontent.com%2FImperialCollegeLondon%2Fsharpy%2Fmaster%2F.version.json) [![codecov](https://codecov.io/gh/ImperialCollegeLondon/sharpy/branch/master/graph/badge.svg)](https://codecov.io/gh/ImperialCollegeLondon/sharpy)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)[![Documentation Status](https://readthedocs.org/projects/ic-sharpy/badge/?version=master)](https://ic-sharpy.readthedocs.io/en/master/?badge=master)


__Status__

| Master  | Develop        |
|---------|----------------|
| [![Build Status](https://travis-ci.org/ImperialCollegeLondon/sharpy.svg?branch=master)](https://travis-ci.org/ImperialCollegeLondon/sharpy) | [![Build Status](https://travis-ci.org/ImperialCollegeLondon/sharpy.svg?branch=develop)](https://travis-ci.org/ImperialCollegeLondon/sharpy)|


SHARPy is an aeroelastic analysis package currently under development at the Department of Aeronautics, Imperial
College London. It can be used for the structural, aerodynamic and aeroelastic analysis of flexible aircraft, flying
wings and wind turbines.

<img src="https://github.com/ImperialCollegeLondon/sharpy/raw/master/docs/source/media/XHALE-render.jpg" width=640>

### Contact 

For more information on the [research team](http://www.imperial.ac.uk/aeroelastics/people/) developing SHARPy or to get in touch, [visit our homepage](http://www.imperial.ac.uk/aeroelastics).

## Physical Models

SHARPy is an aeroelastic solver that uses two specific models for the structural and aerodynamic response of the system.

For the structural model, SHARPy employs a nonlinear, geometrically-exact displacement and rotation-based beam formulation.
This model has the advantage of providing the solution directly in the physical problem's degrees of freedom, making the 
coupling with the aerodynamic solver simple and not requiring any post-processing. The 1D beam formulation used limits 
the analyses that can be done by SHARPy to slender structures, such as high aspect ratio wings.

The aerodynamic model utilises the Unsteady Vortex Lattice Method (UVLM). The aerodynamic surfaces are modelled as a thin
vortex ring lattice with the boundary conditions enforced at the collocation points in the middle of the vortex rings.
The Kutta condition is also enforced at the trailing edge. The wake can be simulated by either additional vortex rings 
or by infinitely long horseshoe vortices, which ideally suited for steady simulations only.

The input problems can be structural, aerodynamic or coupled, yielding an aeroelastic system.

## Capabilities

SHARPy offers (amongst others) the following solutions to the user:
* Static aerodynamic, structural and aeroelastic solutions
* Finding trim conditions for aeroelastic configurations
* Nonlinear, dynamic time domain simulations under a large number of conditions such as:
    + Prescribed trajectories
    + Free flight
    + Dynamic follower forces
    + Control inputs in thrust, control surface deflection...
    + Arbitrary time-domain gusts, including non span-constant ones
    + Full 3D turbulent fields
    + Multibody dynamics with hinges, articulations and prescribed nodal motions
    
## Documentation

The documentation for SHARPy can be found [here](http://ic-sharpy.readthedocs.io).

## Installing SHARPy

For the latest documentation, see the [installation docs](https://ic-sharpy.readthedocs.io/en/latest/content/installation.html)

## Contributing

If you think you can add some useful feature to SHARPy, by all means, check out
the [collaboration guide](https://ic-sharpy.readthedocs.io/en/latest/content/contributing.html).

