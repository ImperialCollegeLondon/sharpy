---
title: 'SHARPy: A nonlinear aeroelasticity toolbox'
tags:
    - Python
    - aeroelasticity
    - structural dynamics
    - aerodynamics
    - solar flight
    - wind energy
authors:
    - name: Alfonso del Carre
      orcid: 0000-0002-8133-9481
      affiliation: 1
    - name: Arturo Muñoz-Simón
      orcid: 0000-0003-4840-5392
      affiliation: 1
    - name: Norberto Goizueta
      orcid: 0000-0001-7445-5970
      affiliation: 1
    - name: Rafael Palacios
      orcid: 0000-0002-6706-3220
      affiliation: 1
affiliations:
    - name: Department of Aeronautics, Imperial College London. London, UK.
      index: 1
date: 13 August 2019
bibliography: paper.bib
---

# Summary

Aeroelasticity is the study of the dynamic interaction between unsteady aerodynamics
and structural dynamics on flexible streamlined bodies, which may include
rigid-body dynamics.  Industry standard solutions in aeronautics and wind energy
are built on the assumption of small structural displacements, which lead to linear
or quasi-linear theories. However, advances in areas such as energy storage and generation,
and composite material manufacturing have fostered a new kind of aeroelastic
structures that may undergo large displacements under aerodynamic forces.

In particular, solar-powered High-Altitude Long-Endurance (HALE) aircraft
have recently seen very significant progress. New configurations
are now able
to stay airborne for longer than three weeks at a time.
Extreme efficiency is achieved by reducing the total weight of the aircraft while
increasing the lifting surfaces' aspect ratio.
In a similar quest for extreme efficiency, the wind energy industry
is also trending towards longer and more slender blades, specially for off-shore
applications, where the largest blades are now close to 100-m long.


These longer and much slender structures can present large deflections and have relatively low frequency structural
modes which, in the case of aircraft, can interact with the flight dynamics modes with potentially unstable couplings.
In the case of offshore wind turbines, platform movement may generate important rotor excursions that cause complex
aeroelastic phenomena which conventional quasi-linear methods may not accurately capture.


``SHARPy`` (Simulation of High-Aspect Ratio Planes in Python) is a dynamic aeroelasticity simulation toolbox for
aircraft and wind turbines. It features a versatile interface and core code written in Python3, while computationally
expensive routines are included in libraries coded in C++ and Modern Fortran. SHARPy is easily extended through a
modular object-oriented design, and includes tools for linear and nonlinear analysis of the time-domain aeroelastic
response of flexible bodies in a large number of cases, such as 3-D discrete gust [@del2019ifasd] and turbulent field
input [@Hesse2016; @arxiv2019Deskos], control surface deflection and prescribed motion [@del2019efficient]. In addition, linearised
state-space models can be obtained for frequency domain analysis, controller design and model reduction.


``SHARPy`` relies only on freely-available open-source dependencies
such as [Paraview](https://paraview.org) for post-processing
The computationally
expensive routines written in C++ and Fortran have been designed with Fluid-Structure
Interaction (FSI) problems in mind, this resulting in minimal overhead between
function calls.

## Features
The [structural model](https://github.com/imperialcollegelondon/xbeam)
included in ``SHARPy`` is a Geometrically-Exact Composite Beam (GECB) [@geradin2001; @Hesse2014a]
supports multibody features
such as hinges, joints and absolute and relative nodal velocity constraints.
Rigid body motion can be prescribed or simulated. The structural solver supports
distributed and lumped mass formulation (or a combination of both). Time-integration
is carried out using a Newmark-$\beta$ scheme.

The [aerodynamic solver](https://github.com/imperialcollegelondon/uvlm) is an Unsteady
Vortex Lattice Method (UVLM) [@Katz2001; @Simpson2013-2].
It can simulate an arbitrary number of surfaces together
with their interactions. A non conventional force evaluation scheme is used [@Simpson2013-2] in
order to support large sideslip angles and obtain an induced drag estimation.
In addition to this, added mass effects can be obtained and introduced in the
FSI problem. This can be especially important in the case of very light flexible
aircraft flying at low altitude.

The coupling algorithm included in the code is designed to allow fully coupled
nonlinear simulations, although weakly coupled solutions can be obtained. Independent
structural or aerodynamic simulation are supported natively.
The nonlinear system can also be linearised taking an arbitrary reference condition. The linearised system can be used
for frequency domain analysis, linear model order reduction methods and controller design.

![Static simulation of the XHALE [@del2019ifasd] nonlinear aeroelasticity
testbed.](https://github.com/ImperialCollegeLondon/sharpy/raw/master/docs/source/media/XHALE-render.jpg)


# Acknowledgements
A. Carre gratefully acknowledges the support provided by Airbus Defence and Space. Norberto Goizueta's acknowledges and
thanks the Department of Aeronautics at Imperial College for sponsoring his research.
Arturo Muñoz-Simón's research has received funding from the EU's H2020 research and innovation programme
under the Marie Sklodowska-Curie grant agreement 765579.

# References
