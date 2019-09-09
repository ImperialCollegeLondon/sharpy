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

Aeroelasticity is the study the dynamic interaction between unsteady aerodynamics
and structural dynamics on flexible streamlined bodies, potentially also including
rigid-body dynamics.  Industry standard solutions in aeronautics and wind energy
are built on the assumption of small structural displacements, which lead to linear
or quasi-linear theories. However, advances in areas such as energy storage and generation,
and composite material manufacturing have fostered a new kind of aeroelastic
structures that may undergo large displacements under aerodynamic forces.

In particular, solar-powered High-Altitude Long-Endurance (HALE) aircraft
have recently seen very significant progress. New configurations
such as the latest generation Airbus D&S Zephyr are now able
to stay airborne for longer than three weeks at a time. As HALE platforms become
more reliable, the payload mass and energy requirements increase. The mission
of these aircraft imposes very strict efficiency constraints on the design.
Extreme efficiency is achieved by reducing the total weight of the aircraft while
increasing the lifting surfaces aspect ratio in order to minimise the induced drag
contribution. In a similar quest for extreme efficiency, the wind energy industry
is also trending towards longer and more slender blades, specially for off-shore
applications, where the largest blades are now close to 100-m long.


These longer and much slender structures can potentially present large ($+10%$) deflections
and have relatively low frequency structural modes which, in the case of aircraft, can interact with the flight
dynamics modes with potentially unstable couplings. In the case of offshore wind energy,
the platform movement is generating important rotor excursions. Due to this, the
conventional quasi-linear methods may not accurately capture the relevant
phenomena present in the aeroelastic response of these new configurations.


``SHARPy`` (Simulation of High-Aspect Ratio Planes in Python) is a dynamic aeroelasticity
simulation toolbox for aircraft and wind turbines. It features a versatile interface and core
code written in Python3, while computationally expensive routines are coded in
C++ and Modern Fortran and distributed as shared libraries. SHARPy is easily extended
through a modular object-oriented design, and already includes tools for
linear and nonlinear analysis of the time-domain aeroelastic response of flexible bodies
in a large number of cases, such as 3-D discrete gust [@IFASD] and turbulent field input [@YA1],
control surface deflection and prescribed motion [@Scitech]. In addition, linearised
state-space models can be obtained for frequency domain analysis, controller design and model reduction.


This code is actively developed at the Loads Control and Aeroelasticity Lab at
Imperial College London, and it is used by academic researchers and students, as well as
industrial partners. ``SHARPy`` relies only on Open-Source dependencies
such as Paraview for post-processing, while supporting custom extension modules
without having to modify a line of the distributed code. The computationally
expensive routines written in C++ and Fortran have been designed with Fluid-Structure
Interaction (FSI) problems in mind, this resulting in minimal overhead between
function calls.

The code is automatically documented ([https://ic-sharpy.readthedocs.io/](https://ic-sharpy.readthedocs.io/))
and has Continuous Integration and Code Coverage enabled for all user contributions.

## Features

While ``SHARPy`` is the latest iteration of the in-house aeroelastic solver
at the Loads Control and Aeroelasticity Lab at Imperial College, its theory
is based on the lab's previous ``SHARP`` framework, a prototype tool coded in Matlab. Previously published works
on numerical methods for nonlinear aeroelasticity such as [Palacios2010, Hesse2014a]
and its applications [@Hesse2016; @del2019efficient; @del2019ifasd; @arxiv2019Deskos] give a good
idea of the possibilities.

The structural model included in ``SHARPy`` supports multibody features
such as hinges, joints and absolute and relative nodal velocity constraints.
Rigid body motion can be prescribed or simulated. The structural solver supports
distributed and lumped mass formulation (or a combination of both). Time-integration
is carried out using a Newmark-$\beta$ scheme.

The aerodynamic solver can simulate an arbitrary number of surfaces together
with their interactions. A different force evaluation scheme is used [@Simpson2013-2] in
order to support large sideslip angles and obtain an induced drag estimation.
In addition to this, added mass effects can be obtained and introduced in the
FSI problem. This can be especially important in the case of very light flexible
aircraft flying at low altitude.

The coupling algorithm included in the code is designed to allow fully coupled
nonlinear simulations, although weakly coupled solutions can be obtained. Independent
structural or aerodynamic simulation are supported natively.

The nonlinear system can also be linearised taking an arbitrary reference condition. The linearised system can be used
for frequency domain analysis and linear model order reduction methods and controller design.

The code distributed in the repository includes modules to directly simulate:

* Static linear and nonlinear structural solutions
* Dynamic linear and nonlinear structural solutions
* Dynamic nonlinear multibody solutions
* Static aerodynamic solutions with horseshoe, prescribed or free wake
* Dynamic aerodynamic solutions with fixed, background-flow convected or free wake
* Static aeroelastic solutions and flexible aircraft longitudinal trimming
* Control surface controller in the loop
* Unsteady fully-3D turbulent field input, working in-memory or in-disk, including time-domain interpolation
between snapshots
* Linearisation of the nonlinear system at an arbitrary reference
* Stability analysis and frequency response
* Model order reduction using moment matching methods, modal reduction or balanced truncation


# The ``SHARPy`` aeroelastic framework

![Static simulation of the XHALE nonlinear aeroelasticity
testbed.](https://github.com/ImperialCollegeLondon/sharpy/raw/master/docs/source/media/XHALE-render.jpg)

The main solver in ``SHARPy`` is a time-domain geometrically nonlinear coupled solver [@Palacios2010]
with a Block Gauss-Seidel iteration scheme between the structural and aerodynamic
solutions.

The structural and free-body dynamics are based on the Geometrically Exact Composite
Beam (GECB) [@Geradin2001; @Simpson2013]. The resulting model is a finite element beam
formulation based on nodal displacements and rotations with quadratic 1-D elements.
The GECB formulation based on displacements and rotations presents several
advantages for FSI problems. First, this model allows for geometrically
nonlinear deformations with linear constitutive equations. Second, the
formulation supports cross-terms in the stiffness matrix for the 6 degrees of
freedom. Last, the GECB model formulated in displacements and rotations
is easily coupled with the aerodynamic solver and boundary conditions can
simply be imposed.

The aerodynamic solver is a in-house implementation of the classic Unsteady
Vortex Lattice Method (UVLM) [@Katz2001; @Murua2012a] with added mass effects.
The code is distributed as a C++ library with
seamless interface with Python. The solver is designed so that no data
needs to be copied or transformed when passing from Python to C+ and vice-versa.
Furthermore, the C++ routines use the same memory allocated by the Numpy arrays
in the ``SHARPy`` Python core. This is achieved using C++ custom data types
based on Eigen3 [CITE] ``map`` feature and the C++ standard library.
The code supports several wake convection schemes, including free wake.
Any velocity field is supported using a modular architecture, where
a ``Generator`` prototype is provided. Included custom ``Generators`` cover from
steady velocity fields to high resolution LES turbulent boundary layer simulation
results [@arxiv2019Deskos]. Force and displacements interpolation is not necessary as
the beam and aerodynamic discretisation are set to be matching, so that a
chordwise grid vertex is always uniquely attached to a beam node.

The FSI coupling and time stepping procedure is controlled by the Python core
of ``SHARPy``. The time-domain solution process is coded in the ``DynamicCoupled``
solver, which instantiates two other solvers, a structural and an aerodynamic one.
It is important to notice that all the solvers are derived from the same abstract
class, and can be run independently or as part of a parent solver.
``DynamicCoupled`` supports an arbitrary number of controllers. These can be related
to any data available during execution, including structural, aerodynamic or
flight dynamics quantities. In a similar way, custom postprocessors with functionality
ranging from incidence angle or beam loads calculation to output in Paraview format
can be created and run with no need to modify the original source code.

Nonlinear systems can be linearised applying small perturbations about an
arbitrary reference condition [@Maraniello2019]. Since the UVLM boundary conditions are more
naturally expressed in discrete-time and the linearised structural system needs to be
in the same form as the aerodynamic one for coupling purposes, a Newmark-$\beta$ integration scheme is used to
assemble the linearised GECB system. The resulting discrete-time aeroelastic state-space can be reduced using
moment matching methods based on Krylov subspaces or frequency-balanced reduction methods, which are particularly well
suited for large-scale dynamical systems. The linearised formulation permits the
use of eigenvalue stability analyses that can be used, for instance, for flutter onset predictions or
the design of linear controllers.


## Input and user interface

The main simulation parameters are input through a ``.sharpy`` file, which
is formatted following the ``ConfigObj`` (CITE) convention. This file is
organised in sections, one of which is mandatory and contains basic parameters
such as the name and location of the case, the log file route, and most importantly,
the ordered list of solvers to run during the simulation. A typical case of
nonlinear aeroelastic response would include two solvers for generating the structural and
aerodynamic models, a static coupled solver to initialise the solution, and a dynamic coupled solver.
``Postprocessors`` are treated as a special type of solvers that can be run during or after the simulation
to extract meaningful information from the simulation data. Every one of these solvers requires its
own section in the ``.sharpy`` file containing its settings.

The structural and aerodynamic models are given in HDF5 (CITE) format. To
avoid redundancy and precision errors, the aerodynamic grid for the UVLM is built upon
the structural model. Aerodynamic-related information such as twist angle, airfoil shape
and elastic axis location is given with respect to the structural elements and nodes.

Time-dependent input can be included through ``Generators`` and ``Controllers``.
``Generators`` are intended to be used mainly to create velocity fields that are
then input into the UVLM code. On the other hand, ``Controllers`` can have a broader
scope. They can be used to control lifting surfaces in open and closed loop, or
modify numerical parameters on the loop.

Linearised problems require first the solution of a nonlinear case that is used as the
linearisation reference. Thence, the system is linearised and the state-space assembled attending to
user-specific settings, including model reduction methods. The resulting system is given in
a discrete-time formulation with several post-processors available or it can be saved for the user
to manipulate externally.

# Acknowledgements

A. Carre gratefully acknowledges the support provided by Airbus Defence and Space. Norberto Goizueta's research is
sponsored by the Department of Aeronautics at Imperial College for which the author is truly grateful.
Arturo Muñoz-Simón's research has received funding from the EU's H2020 research and innovation programme
under the Marie Sklodowska-Curie grant agreement 765579.
# References
