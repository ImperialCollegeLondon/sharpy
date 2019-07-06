import numpy as np
import unittest
import os


class TestDoublePendulum(unittest.TestCase):
    """
    Validation of a double pendulum with a mass at each tip position

    Reference case: M. Geradin and A. Cardona, "Flexible multibody dynamics : a finite element approach"
    """

    def setUp(self):
        import sharpy.utils.generate_cases as gc

        deg2rad = np.pi/180.

        # Structural properties
        mass_per_unit_length = 1.
        mass_iner = 1e-4
        EA = 1e9
        GA = 1e9
        GJ = 1e9
        EI = 1e9

        # Beam1
        nnodes1 = 11
        l1 = 1.0
        m1 = 1.0
        theta_ini1 = 90.*deg2rad

        # Beam2
        nnodes2 = nnodes1
        l2 = l1
        m2 = m1
        theta_ini2 = 00.*deg2rad

        # airfoils
        airfoil = np.zeros((1,20,2),)
        airfoil[0,:,0] = np.linspace(0.,1.,20)

        # Simulation
        numtimesteps = 100
        dt = 0.01

        # Create the structure
        beam1 = gc.AeroelasticInformation()
        r1 = np.linspace(0.0, l1, nnodes1)
        node_pos1 = np.zeros((nnodes1,3),)
        node_pos1[:, 0] = r1*np.sin(theta_ini1)
        node_pos1[:, 2] = -r1*np.cos(theta_ini1)
        beam1.StructuralInformation.generate_uniform_sym_beam(node_pos1, mass_per_unit_length, mass_iner, EA, GA, GJ, EI, num_node_elem = 3, y_BFoR = 'y_AFoR', num_lumped_mass=1)
        beam1.StructuralInformation.body_number = np.zeros((beam1.StructuralInformation.num_elem,), dtype = int)
        beam1.StructuralInformation.boundary_conditions[0] = 1
        beam1.StructuralInformation.boundary_conditions[-1] = -1
        beam1.StructuralInformation.lumped_mass_nodes = np.array([nnodes1-1], dtype = int)
        beam1.StructuralInformation.lumped_mass = np.ones((1,))*m1
        beam1.StructuralInformation.lumped_mass_inertia = np.zeros((1,3,3))
        beam1.StructuralInformation.lumped_mass_position = np.zeros((1,3))
        beam1.AerodynamicInformation.create_one_uniform_aerodynamics(
                                            beam1.StructuralInformation,
                                            chord = 1.,
                                            twist = 0.,
                                            sweep = 0.,
                                            num_chord_panels = 4,
                                            m_distribution = 'uniform',
                                            elastic_axis = 0.25,
                                            num_points_camber = 20,
                                            airfoil = airfoil)

        beam2 = gc.AeroelasticInformation()
        r2 = np.linspace(0.0, l2, nnodes2)
        node_pos2 = np.zeros((nnodes2,3),)
        node_pos2[:, 0] = r2*np.sin(theta_ini2) + node_pos1[-1, 0]
        node_pos2[:, 2] = -r2*np.cos(theta_ini2) + node_pos1[-1, 2]
        beam2.StructuralInformation.generate_uniform_sym_beam(node_pos2, mass_per_unit_length, mass_iner, EA, GA, GJ, EI, num_node_elem = 3, y_BFoR = 'y_AFoR', num_lumped_mass=1)
        beam2.StructuralInformation.body_number = np.zeros((beam1.StructuralInformation.num_elem,), dtype = int)
        beam2.StructuralInformation.boundary_conditions[0] = 1
        beam2.StructuralInformation.boundary_conditions[-1] = -1
        beam2.StructuralInformation.lumped_mass_nodes = np.array([nnodes2-1], dtype = int)
        beam2.StructuralInformation.lumped_mass = np.ones((1,))*m2
        beam2.StructuralInformation.lumped_mass_inertia = np.zeros((1,3,3))
        beam2.StructuralInformation.lumped_mass_position = np.zeros((1,3))
        beam2.AerodynamicInformation.create_one_uniform_aerodynamics(
                                            beam2.StructuralInformation,
                                            chord = 1.,
                                            twist = 0.,
                                            sweep = 0.,
                                            num_chord_panels = 4,
                                            m_distribution = 'uniform',
                                            elastic_axis = 0.25,
                                            num_points_camber = 20,
                                            airfoil = airfoil)

        beam1.assembly(beam2)

        # Simulation details
        SimInfo = gc.SimulationInformation()
        SimInfo.set_default_values()

        SimInfo.define_uinf(np.array([0.0,1.0,0.0]), 1.)

        SimInfo.solvers['SHARPy']['flow'] = ['BeamLoader',
                                'AerogridLoader',
                                'InitializeMultibody',
                                'DynamicCoupled']
        SimInfo.solvers['SHARPy']['case'] = 'double_pendulum_geradin'
        SimInfo.solvers['SHARPy']['write_screen'] = 'off'
        SimInfo.solvers['SHARPy']['route'] = os.path.dirname(os.path.realpath(__file__)) + '/'
        SimInfo.set_variable_all_dicts('dt', dt)
        SimInfo.define_num_steps(numtimesteps)
        SimInfo.set_variable_all_dicts('rho', 0.0)
        SimInfo.set_variable_all_dicts('velocity_field_input', SimInfo.solvers['SteadyVelocityField'])

        SimInfo.solvers['BeamLoader']['unsteady'] = 'on'

        SimInfo.solvers['AerogridLoader']['unsteady'] = 'on'
        SimInfo.solvers['AerogridLoader']['mstar'] = 2

        SimInfo.solvers['WriteVariablesTime']['FoR_number'] = np.array([0, 1], dtype = int)
        SimInfo.solvers['WriteVariablesTime']['FoR_variables'] = ['mb_quat']
        SimInfo.solvers['WriteVariablesTime']['structure_nodes'] = np.array([nnodes1-1, nnodes1+nnodes2-1], dtype = int)
        SimInfo.solvers['WriteVariablesTime']['structure_variables'] = ['pos']

        SimInfo.solvers['NonLinearDynamicMultibody']['gravity_on'] = True
        SimInfo.solvers['NonLinearDynamicMultibody']['newmark_damp'] = 0.15

        SimInfo.solvers['DynamicCoupled']['structural_solver'] = 'NonLinearDynamicMultibody'
        SimInfo.solvers['DynamicCoupled']['structural_solver_settings'] = SimInfo.solvers['NonLinearDynamicMultibody']
        SimInfo.solvers['DynamicCoupled']['aero_solver'] = 'StepUvlm'
        SimInfo.solvers['DynamicCoupled']['aero_solver_settings'] = SimInfo.solvers['StepUvlm']
        SimInfo.solvers['DynamicCoupled']['postprocessors'] = ['WriteVariablesTime', 'BeamPlot', 'AerogridPlot']
        SimInfo.solvers['DynamicCoupled']['postprocessors_settings'] = {'WriteVariablesTime': SimInfo.solvers['WriteVariablesTime'],
                                                                        'BeamPlot': SimInfo.solvers['BeamPlot'],
                                                                        'AerogridPlot': SimInfo.solvers['AerogridPlot']}

        SimInfo.with_forced_vel = False
        SimInfo.with_dynamic_forces = False

        # Create the MB and BC files
        LC1 = gc.LagrangeConstraint()
        LC1.behaviour = 'hinge_FoR'
        LC1.body_FoR = 0
        LC1.rot_axis_AFoR = np.array([0.0,1.0,0.0])

        LC2 = gc.LagrangeConstraint()
        LC2.behaviour = 'hinge_node_FoR'
        LC2.node_in_body = nnodes1-1
        LC2.body = 0
        LC2.body_FoR = 1
        LC2.rot_axisB = np.array([0.0,1.0,0.0])

        LC = []
        LC.append(LC1)
        LC.append(LC2)

        MB1 = gc.BodyInformation()
        MB1.body_number = 0
        MB1.FoR_position = np.zeros((6,),)
        MB1.FoR_velocity = np.zeros((6,),)
        MB1.FoR_acceleration = np.zeros((6,),)
        MB1.FoR_movement = 'free'
        MB1.quat = np.array([1.0,0.0,0.0,0.0])

        MB2 = gc.BodyInformation()
        MB2.body_number = 1
        MB2.FoR_position = np.array([node_pos2[0, 0], node_pos2[0, 1], node_pos2[0, 2], 0.0, 0.0, 0.0])
        MB2.FoR_velocity = np.zeros((6,),)
        MB2.FoR_acceleration = np.zeros((6,),)
        MB2.FoR_movement = 'free'
        MB2.quat = np.array([1.0,0.0,0.0,0.0])

        MB = []
        MB.append(MB1)
        MB.append(MB2)

        # Write files
        gc.clean_test_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
        SimInfo.generate_solver_file()
        SimInfo.generate_dyn_file(numtimesteps)
        beam1.generate_h5_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
        gc.generate_multibody_file(LC, MB,SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])

    def test_doublependulum(self):
        import sharpy.sharpy_main

        solver_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/double_pendulum_geradin.solver.txt')
        sharpy.sharpy_main.main(['', solver_path])

        # read output and compare
        output_path = os.path.dirname(solver_path) + '/output/double_pendulum_geradin/WriteVariablesTime/'
        pos_tip_data = np.atleast_2d(np.genfromtxt(output_path + "struct_pos_node" + str(11+11-1) + ".dat", delimiter=' '))
        self.assertAlmostEqual(pos_tip_data[-1, 1], 1.481168, 4)
        self.assertAlmostEqual(pos_tip_data[-1, 2], 0.000000, 4)
        self.assertAlmostEqual(pos_tip_data[-1, 3], 0.8766285, 4)

