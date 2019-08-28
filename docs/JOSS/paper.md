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
        affiliation: "1" # (Multiple affiliations must be quoted)
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

Aeroelasticity is the study the interaction of the aerodynamics 
and the structural dynamics of a body, potentially including flight dynamics. 
In aeronautics and wind energy, the industry standard methods rely on tried and
tested assumptions and simplifications, leading to linear or quasi-linear expressions.
The usual configurations studied in aeroelasticity are relatively rigid, presenting
small structural deflections for the typical load cases, as well as showing a
substantial degree of decoupling between the structural and the rigid body modes.
In these cases, the conventional aeroelastic tools efficiently provide accurate
solutions. However, advances in areas such as energy storage and generation,
and composite material manufacturing have fostered a new kind of aeroelastic
structures.


In the last few years, solar-powered High-Altitude Long-Endurance (HALE) aircraft
have shown very significant progress in their development. New configurations are able
to stay airborne for longer than three weeks at a time. As HALE platforms become
more reliable, the payload mass and energy requirements increase. The mission
of these aircraft imposes very strict efficiency constraints on the design.
Extreme efficiency is achieved by reducing the total weight of the aircraft while
increasing the lifting surfaces aspect ratio in order to minimise the induced drag
contribution. The wind energy industry is also trending towards longer and more
slender blades, specially for the incresingly common off-shore wind farms, in
search of more efficient turbines.


These light, slender structures naturally present lower stiffness than the
conventional aircraft and wind turbines. The need for nonlinear aeroelastic tools
is mainly driven by this fact. The long, slender structures can potentially
present large ($+10%$) deflections and have relatively low frequency structural modes which
can interact with the flight dynamics modes of the aircraft. Due to this, the
conventional quasi-linear methods may not accurately capture the relevant
phenomena present in the aeroelastic response of these new configurations.


``SHARPy`` (Simulation of High-Aspect Ratio Planes in Python) is a linear and
nonlinear aeroelasticity toolbox. It features a versatile interface and core
code written in Python3, while computationally expensive routines are coded in
C++ and Modern Fortran and distributed as shared libraries. SHARPy is easily extended
through a modular object-oriented design, and already includes tools for
linear and nonlinear analysis of the time-domain aeroelastic response of flexible bodies
in a large number of cases, such as 3-D discrete gust [@IFASD] and turbulent field input [@YA1], 
control surface deflection and prescribed motion [@Scitech]. In addition, linearised
state-space models can be obtained for control design and model reduction.


This code is actively developed at the Loads Control and Aeroelasticity Lab at
Imperial College London, and it is used by academic researchers and students, as well as industry members of
the community. ``SHARPy`` relies only on Open-Source dependencies
such as Paraview for post-processing, while supporting custom extension modules
without having to modify a line of the distributed code. The computationally
expensive routines written in C++ and Fortran have been designed with Fluid-Structure
Interaction (FSI) problems in mind, this resulting in minimal overhead between
function calls.

The code is automatically documented ([https://ic-sharpy.readthedocs.io/](https://ic-sharpy.readthedocs.io/))
and has Continuous Integration enabled for all user contributions. 



# The ``SHARPy`` aeroelastic framework

``SHARPY``'s main solver is a time-domain geometrically nonlinear coupled solver
with a Block Gauss-Seidel iteration scheme between the structural and aerodynamic
solutions.

The structural and free-body dynamics are based on the Geometrically Exact Composite
Beam (GECB) [@Geradin2001]. The resulting model is a finite element beam
fomulation based on nodal displacements and rotations with quadratic 1-D elements.
The GECB formulation based on displacements and rotations presents several
advantages for FSI problems. First, this model allows for geometrically
nonlinear deformations with linear constitutive equations. Second, the
formulation supports cross-terms in the stiffness matrix for the 6 degrees of
freedom. Last, the GECB model formulated in displacements and rotations
is easily coupled with the aerodynamic solver and boundary conditions can 
simply be imposed.

The aerodynamic solver is a in-house implementation of the classic Unsteady
Vortex Lattice Method (UVLM) with added mass effects. The code is distributed as a C++ library with
seamless interface with Python. The solver is designed so that no data
needs to be copied or transformed when passing from Python to C+ and vice-versa.
Furthermore, the C++ routines use the same memory allocated by the Numpy arrays
in the ``SHARPy`` Python core. This is achieved using C++ custom data types
based on Eigen3 [CITE] ``map`` feature and the C++ standard library.
The code supports several wake convection schemes, including free wake.
Any velocity field is supported using a modular architecture, where
a ``Generator`` prototype is provided. Included custom ``Generators`` cover from
steady velocity fields to high resolution LES turbulent boundary layer simulation
results [CITE YA1]. Force and displacements interpolation is not necessary as
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

## Input and user interface
The main simulation parameters are input through a ``.sharpy`` file, which
is formatted following the ``ConfigObj`` (CITE) convention. This file is
organised in sections, one of which is mandatory and contains basic parameters
such as the name and location of the case, the log file route, and most importantly,
the ordered list of solvers to run during the simulation. A typical case of
nonlinear aeroelastic response would include two solvers for generating the structural and
aerodynamic models, a static coupled solver to initialise the solution, and a dynamic coupled solver.
Every one of these solvers requires its own section in the ``.sharpy`` file containing
its settings.

The structural and aerodynamic models are given in HDF5 (CITE) format. To
avoid redundancy and precision errors, the aerodynamic grid for the UVLM is built upon
the structural model. Aerodynamic-related information such as twist angle, airfoil shape
and elastic axis location is given with respect to the structural elements and nodes.

Time-dependant input can be included through ``Generators`` and ``Controllers``.
``Generators`` are intended to be used mainly to create velocity fields that are
then in put into the UVLM code. On the other hand, ``Controllers`` can have a broader
scope. They can be used to control lifting surfaces in open and closed loop, or
modify numerical parameters on the loop.

# Capabilities and test cases

``SHARPy`` excels in the nonlinear aeroelasticity domain due to its
relatively long history (CITE SOME OLD PAPER FROM RAFA AND JOSEBA) and
flexibility. The same code is being used on Very Flexible Aircraft
catapult take off studies and High Aspect Ratio Wind Turbines turbulence response.

## Very Flexible Aircraft aeroelastic response in low-altitude turbulence
``SHARPy`` has been used to simulate the response of a very flexible aircraft
subject to low altitude atmospheric turbulence. Synthetic turbulence generation
methods such as the 1-D von K\'{a}rm\'{a}n and the 2-D Kaimal spectra have been
applied in order to obtain velocity fields with similar PSDs to high-fidelity
LES computations. Aeroelastic simulations of aircraft inmersed in the different
turbulent fields showed how synthetic turbulence methods do not capture the larger flow
structures present in the LES simulations. This results in strong discrepancies
in wing root bending loads of up to 58% between different turbulent field generation methods.
Results showed that one-dimensional models, which are the de facto standard for
conventional aircraft provide fairly non-conservative load estimations for typical
very flexible configurations. CITE

## Catapult Take Off analysis and optimisation


## Wind Turbine aeroelastic response to turbulence






# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this: ![Example figure.](figure.png)

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References

























