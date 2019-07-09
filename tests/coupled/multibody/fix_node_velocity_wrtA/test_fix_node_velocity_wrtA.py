import numpy as np
import unittest
import os
import shutil


class TestFixNodeVelocitywrtA(unittest.TestCase):

    def setUp(self):
        import sharpy.utils.generate_cases as gc

        deg2rad = np.pi/180.
        nodes_per_elem = 3

        # beam1: uniform and symmetric with aerodynamic properties equal to zero
        nnodes1 = 11
        length1  = 10.
        mass_per_unit_length = 1.
        mass_iner = 1e-4
        EA = 1e9
        GA = 1e9
        GJ = 1e3
        EI = 1e4

        # Create beam1
        beam1 = gc.AeroelasticInformation()
        # Structural information
        beam1.StructuralInformation.num_node = nnodes1
        beam1.StructuralInformation.num_node_elem = nodes_per_elem
        beam1.StructuralInformation.compute_basic_num_elem()
        beam1.StructuralInformation.set_to_zero(beam1.StructuralInformation.num_node_elem, beam1.StructuralInformation.num_node, beam1.StructuralInformation.num_elem)
        node_pos = np.zeros((nnodes1, 3), )
        node_pos[:, 0] = np.linspace(0.0, length1, nnodes1)
        beam1.StructuralInformation.generate_uniform_sym_beam(node_pos, mass_per_unit_length, mass_iner, EA, GA, GJ, EI, num_node_elem = 3, y_BFoR = 'y_AFoR', num_lumped_mass=1)
        beam1.StructuralInformation.boundary_conditions[0] = 1
        beam1.StructuralInformation.boundary_conditions[-1] = -1
        beam1.StructuralInformation.lumped_mass_nodes = np.array([nnodes1-1], dtype=int)
        beam1.StructuralInformation.lumped_mass = np.array([1.])
        beam1.StructuralInformation.lumped_mass_inertia = np.zeros((1, 3, 3),)
        beam1.StructuralInformation.lumped_mass_position = np.zeros((1, 3),)


        # Aerodynamic information
        airfoil = np.zeros((1,20,2),)
        airfoil[0,:,0] = np.linspace(0.,1.,20)
        beam1.AerodynamicInformation.create_one_uniform_aerodynamics(
                                            beam1.StructuralInformation,
                                            chord = 1.,
                                            twist = 0.,
                                            sweep = 0.,
                                            num_chord_panels = 4,
                                            m_distribution = 'uniform',
                                            elastic_axis = 0.5,
                                            num_points_camber = 20,
                                            airfoil = airfoil)

        # SOLVER CONFIGURATION
        SimInfo = gc.SimulationInformation()
        SimInfo.set_default_values()

        SimInfo.define_uinf(np.array([0.0,1.0,0.0]), 10.)

        SimInfo.solvers['SHARPy']['flow'] = ['BeamLoader',
                                'AerogridLoader',
                                'StaticCoupled',
                                'DynamicCoupled']
        global name
        name = 'fix_node_velocity_wrtA'
        SimInfo.solvers['SHARPy']['case'] = name
        SimInfo.solvers['SHARPy']['write_screen'] = 'off'
        SimInfo.solvers['SHARPy']['route'] = os.path.dirname(os.path.realpath(__file__)) + '/'
        SimInfo.set_variable_all_dicts('dt', 0.1)
        SimInfo.set_variable_all_dicts('rho', 0.0)
        SimInfo.set_variable_all_dicts('velocity_field_input', SimInfo.solvers['SteadyVelocityField'])

        SimInfo.solvers['BeamLoader']['unsteady'] = 'on'

        SimInfo.solvers['AerogridLoader']['unsteady'] = 'on'
        SimInfo.solvers['AerogridLoader']['mstar'] = 2

        SimInfo.solvers['NonLinearStatic']['print_info'] = False

        SimInfo.solvers['StaticCoupled']['structural_solver'] = 'NonLinearStatic'
        SimInfo.solvers['StaticCoupled']['structural_solver_settings'] = SimInfo.solvers['NonLinearStatic']
        SimInfo.solvers['StaticCoupled']['aero_solver'] = 'StaticUvlm'
        SimInfo.solvers['StaticCoupled']['aero_solver_settings'] = SimInfo.solvers['StaticUvlm']

        SimInfo.solvers['NonLinearDynamicMultibody']['gravity_on'] = True

        SimInfo.solvers['WriteVariablesTime']['structure_nodes'] = np.array([0,  int((nnodes1-1)/2), -1], dtype = int)
        SimInfo.solvers['WriteVariablesTime']['structure_variables'] = ['pos']

        SimInfo.solvers['DynamicCoupled']['structural_solver'] = 'NonLinearDynamicMultibody'
        SimInfo.solvers['DynamicCoupled']['structural_solver_settings'] = SimInfo.solvers['NonLinearDynamicMultibody']
        SimInfo.solvers['DynamicCoupled']['aero_solver'] = 'StepUvlm'
        SimInfo.solvers['DynamicCoupled']['aero_solver_settings'] = SimInfo.solvers['StepUvlm']
        SimInfo.solvers['DynamicCoupled']['postprocessors'] = ['WriteVariablesTime', 'BeamPlot', 'AerogridPlot']
        SimInfo.solvers['DynamicCoupled']['postprocessors_settings'] = {'WriteVariablesTime': SimInfo.solvers['WriteVariablesTime'],
                                                                        'BeamPlot': SimInfo.solvers['BeamPlot'],
                                                                        'AerogridPlot': SimInfo.solvers['AerogridPlot']}

        ntimesteps = 10

        SimInfo.define_num_steps(ntimesteps)

        # Define dynamic simulation
        SimInfo.with_forced_vel = False
        SimInfo.with_dynamic_forces = False

        LC2 = gc.LagrangeConstraint()
        LC2.behaviour = 'lin_vel_node_wrtA'
        LC2.velocity = np.array([0.,0.,0.])
        LC2.body_number = 0
        LC2.node_number = int((nnodes1-1)/2)

        LC = []
        # LC.append(LC1)
        LC.append(LC2)

        # Define the multibody infromation for the tower and the rotor
        MB1 = gc.BodyInformation()
        MB1.body_number = 0
        MB1.FoR_position = np.zeros((6,),)
        MB1.FoR_velocity = np.zeros((6,),)
        MB1.FoR_acceleration = np.zeros((6,),)
        MB1.FoR_movement = 'prescribed'
        MB1.quat = np.array([1.0,0.0,0.0,0.0])

        MB = []
        MB.append(MB1)


        gc.clean_test_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
        SimInfo.generate_solver_file()
        SimInfo.generate_dyn_file(ntimesteps)
        beam1.generate_h5_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
        gc.generate_multibody_file(LC, MB,SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])


    # def tearDown():
        # pass

    def test_testfixnodevelocitywrta(self):
        import sharpy.sharpy_main

        solver_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/fix_node_velocity_wrtA.solver.txt')
        print(solver_path)
        sharpy.sharpy_main.main(['', solver_path])

        # read output and compare
        output_path = os.path.dirname(solver_path) + '/output/fix_node_velocity_wrtA/WriteVariablesTime/'
        # quat_data = np.matrix(np.genfromtxt(output_path + 'FoR_00_mb_quat.dat', delimiter=' '))
        pos_tip_data = np.matrix(np.genfromtxt(output_path + "struct_pos_node" + str(-1) + ".dat", delimiter=' '))
        self.assertAlmostEqual(pos_tip_data[-1, 1], 9.996557, 4)
        self.assertAlmostEqual(pos_tip_data[-1, 2], 0.000000, 4)
        self.assertAlmostEqual(pos_tip_data[-1, 3], -0.1795935, 4)

    def tearDowns(self):
        solver_path = os.path.dirname(os.path.realpath(__file__))
        solver_path += '/'
        files_to_delete = [name + '.aero.h5',
                           name + '.dyn.h5',
                           name + '.fem.h5',
                           name + '.mb.h5',
                           name + '.solver.txt']
        for f in files_to_delete:
            os.remove(solver_path + f)

        shutil.rmtree(solver_path + 'output/')

if __name__=='__main__':

    T = TestFixNodeVelocitywrtA()
    T.setUp()
    T.test_testfixnodevelocitywrta()
