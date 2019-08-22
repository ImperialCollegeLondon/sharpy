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
increasing the lifting surfaces aspect ration in order to reduced the induced drag
contribution. The wind energy industry is also trending towards longer and more
slender blades, specially for the rising off-shore wind farms, in search of more
efficient turbines.


These light, slender structures naturally present lower stiffness than the
conventional aircraft and wind turbines. The need for nonlinear aeroelastic tools
is mainly driven by this. The long, slender structures can potentially
present large ($+10%$) deflections and have relatively low structural modes which
can interact with the flight dynamics modes of the aircraft. Due to this, the
conventional quasi-linear methods may not accurately capture the relevant
phenomena present in the aeroelastic response of these new configurations.


``SHARPy`` (Simulation of High-Aspect Ratio Planes in Python) is a linear and
nonlinear aeroelasticity toolbox. It features a versatile interface and central
code written in Python3, while computationally expensive routines are coded in
C++ and Modern Fortran and distributed as libraries. SHARPy is easily extended
through a modular object-oriented design, and already includes tools for
linear and nonlinear analysis of the time-domain aeroelastic response of flexible bodies
in a large number of cases, such as 3-D discrete gust [@IFASD] and turbulent field input [@YA1], 
control surface deflection and prescribed motion [@Scitech]. In addition, linearised
state-space models can be obtained for control design and model reduction.


This code is actively developed at the Loads Control and Aeroelasticity Lab at
Imperial College London, and it is used by academic researchers and students, as well as industry members of
the community.  ``SHARPy`` relies only on Open-Source dependencies and software,
such as Paraview for post-processing, while supporting custom extension modules
without having to modify a line of the distributed code. The computationally
expensive routines written in C++ and Fortran have been designed with Fluid-Structure
Interaction (FSI) problems in mind, this resulting in minimal overhead between
function calls.

The code is automatically documented ([https://ic-sharpy.readthedocs.io/](https://ic-sharpy.readthedocs.io/))
and has Continuous Integration enabled for all users contributions. 



# Methods and Numerics
``SHARPY``'s main solver is a time-domain geometrically nonlinear coupled solver
with a Block Gauss-Seidel iteration scheme between the structural and aerodynamic
solutions.





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
