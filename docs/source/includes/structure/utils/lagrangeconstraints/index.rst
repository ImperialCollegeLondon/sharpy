LagrangeConstraints library
+++++++++++++++++++++++++++

LagrangeConstraints library

Library used to create the matrices associated to boundary conditions through
the method of Lagrange Multipliers. The source code includes four different sections.

* Basic structures: basic functions and variables needed to organise the library with different Lagrange Constraints to enhance the interaction with this library.

* Auxiliar functions: basic queries that are performed repeatedly.

* Equations: functions that generate the equations associated to the constraint of basic degrees of freedom.

* Lagrange Constraints: different available Lagrange Constraints. They tipically use the basic functions in "Equations" to assembly the required set of equations.

Attributes:
    dict_of_lc (dict): Dictionary including the available Lagrange Contraint identifier
    (``_lc_id``) and the associated ``BaseLagrangeConstraint`` class

Notes:
    To use this library: import sharpy.structure.utils.lagrangeconstraints as lagrangeconstraints

Args:
    lc_list (list): list of all the defined contraints
    MBdict (dict): dictionary with the MultiBody and LagrangeMultipliers information
    MB_beam (list): list of ``Beams`` of each of the bodies that form the system
    MB_tstep (list): list of ``StructTimeStepInfo`` of each of the bodies that form the system
    num_LM_eq (int): number of new equations needed to define the boundary boundary conditions
    sys_size (int): total number of degrees of freedom of the multibody system
    dt (float): time step
    Lambda (np.ndarray): list of Lagrange multipliers values
    Lambda_dot (np.ndarray): list of the first derivative of the Lagrange multipliers values
    dynamic_or_static (str): string defining if the computation is dynamic or static
    LM_C (np.ndarray): Damping matrix associated to the Lagrange Multipliers equations
    LM_K (np.ndarray): Stiffness matrix associated to the Lagrange Multipliers equations
    LM_Q (np.ndarray): Vector of independent terms associated to the Lagrange Multipliers equations


.. toctree::
	:glob:

	./BaseLagrangeConstraint
	./constant_rot_vel_FoR
	./constant_vel_FoR
	./def_rot_axis_FoR_wrt_node
	./def_rot_vel_FoR_wrt_node
	./define_FoR_dof
	./define_node_dof
	./define_num_LM_eq
	./equal_lin_vel_node_FoR
	./free
	./fully_constrained_node_FoR
	./generate_lagrange_matrix
	./hinge_FoR
	./hinge_FoR_wrtG
	./hinge_node_FoR
	./hinge_node_FoR_constant_vel
	./initialise_lc
	./lagrangeconstraint
	./lc_from_string
	./lin_vel_node_wrtA
	./lin_vel_node_wrtG
	./postprocess
	./print_available_lc
	./remove_constraint
	./spherical_FoR
	./spherical_node_FoR
