"""Linear SHARPy

The code included herein enables the assembly of linearised state-space systems based on the previous solution
of a nonlinear problem that will be used as linearisation reference.

The code is structured in the following way:

    * Assembler: different state-spaces to assemble, from only structural/aerodynamic to fully coupled aeroelastic

    * Src: source code required for the linearisation and utilities for the manipulation of state-space elements


References:

    Maraniello, S. , Palacios, R.. State-Space Realizations and Internal Balancing in Potential-Flow Aerodynamics
    with Arbitrary Kinematics. AIAA Journal, Vol. 57, No.6, June 2019
"""