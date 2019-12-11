Generate cases
++++++++++++++

Generate cases

This library provides functions and classes to help in the definition of SHARPy cases

Examples:

    tests in: tests/utils/generate_cases
    examples: test/coupled/multibody/fix_node_velocity_wrtG/test_fix_node_velocity_wrtG
              test/coupled/multibody/fix_node_velocity_wrtA/test_fix_node_velocity_wrtA
              test/coupled/multibody/double_pendulum/test_double_pendulum_geradin
              test/coupled/prescribed/WindTurbine/test_rotor

Notes:

    To use this library: import sharpy.utils.generate_cases as generate_cases



.. toctree::
	:glob:

	./AerodynamicInformation
	./AeroelasticInformation
	./SimulationInformation
	./StructuralInformation
	./clean_test_files
	./from_node_array_to_elem_matrix
	./from_node_list_to_elem_matrix
	./get_airfoil_camber
	./get_aoacl0_from_camber
	./get_factor_geometric_progression
	./get_mu0_from_camber
	./read_column_sheet_type01
