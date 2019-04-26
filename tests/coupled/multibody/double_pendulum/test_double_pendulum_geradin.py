import numpy as np
# import importlib
import unittest
import os

# Data from Geradin
# time[s] theta[rad]
geradin_FoR0 = np.array([[-0.0117973, 1.56808],
                        [0.0816564, 1.5394],
                        [0.171988, 1.41698],
                        [0.235203, 1.31521],
                        [0.307327, 1.09265],
                        [0.427399, 0.601124],
                        [0.526338, 0.0899229],
                        [0.646417, -0.394903],
                        [0.748531, -0.765465],
                        [0.868959, -0.94209],
                        [0.905131, -0.956217],
                        [0.965504, -0.903829],
                        [1.06828, -0.69149],
                        [1.16508, -0.425429],
                        [1.29798, -0.240495],
                        [1.42483, -0.0688408],
                        [1.56382, 0.169571],
                        [1.78751, 0.634087],
                        [1.89627, 0.806105],
                        [1.98075, 0.844608],
                        [2.10125, 0.734983],
                        [2.19143, 0.478564],
                        [2.26934, 0.0347871],
                        [2.37751, -0.315796],
                        [2.52803, -0.546627],
                        [2.60034, -0.601682],
                        [2.76314, -0.645155],
                        [2.88987, -0.580701],
                        [3.05893, -0.423295],
                        [3.24611, -0.239453],
                        [3.43335, -0.00201041],
                        [3.51194, 0.157214],
                        [3.59668, 0.423517],
                        [3.6815, 0.756821],
                        [3.73591, 0.87633],
                        [3.81435, 0.901554],
                        [3.93481, 0.751729],
                        [4.04305, 0.468146],
                        [4.17525, 0.0366783],
                        [4.34949, -0.556439],
                        [4.44551, -0.987179],
                        [4.57784, -1.29805],
                        [4.65016, -1.3531],
                        [4.70444, -1.35419],
                        [4.78294, -1.27537],
                        [4.86762, -1.06267],
                        [4.99464, -0.743611]])

geradin_FoR1 = np.array([[0.00756934, 0.0266485],
                        [0.134225, 0.0241027],
                        [0.309222, 0.100987],
                        [0.418117, 0.393606],
                        [0.490855, 0.713752],
                        [0.533195, 0.820103],
                        [0.635787, 0.871642],
                        [0.762124, 0.587696],
                        [0.85826, 0.264156],
                        [0.918194, -0.0720575],
                        [0.996205, -0.422034],
                        [1.05008, -0.784926],
                        [1.09792, -1.1477],
                        [1.1639, -1.47063],
                        [1.27207, -1.82121],
                        [1.38636, -2.09152],
                        [1.47067, -2.20042],
                        [1.53694, -2.26875],
                        [1.67582, -2.12414],
                        [1.8028, -1.84528],
                        [1.89365, -1.5121],
                        [1.97843, -1.2056],
                        [2.00271, -1.07208],
                        [2.08146, -0.765457],
                        [2.14818, -0.431789],
                        [2.19686, -0.0575583],
                        [2.24552, 0.303273],
                        [2.29421, 0.690904],
                        [2.37299, 1.02433],
                        [2.44573, 1.34447],
                        [2.55464, 1.65049],
                        [2.65749, 1.92983],
                        [2.7904, 2.12817],
                        [2.94135, 2.27254],
                        [3.03182, 2.27072],
                        [3.18853, 2.17376],
                        [3.30891, 1.95694],
                        [3.42312, 1.61964],
                        [3.50121, 1.33666],
                        [3.56714, 0.973525],
                        [3.61495, 0.583954],
                        [3.66883, 0.221062],
                        [3.71673, -0.0881084],
                        [3.80076, -0.451606],
                        [3.87271, -0.828262],
                        [3.95678, -1.15156],
                        [3.98681, -1.25937],
                        [4.08307, -1.47571],
                        [4.13729, -1.5304],
                        [4.27618, -1.38579],
                        [4.36701, -1.066],
                        [4.4217, -0.705294],
                        [4.50652, -0.37199],
                        [4.59132, -0.0520868],
                        [4.68815, 0.240774],
                        [4.79703, 0.519993],
                        [4.91188, 0.74549],
                        [4.98432, 0.797635]])

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
        numtimesteps = 500
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
                                # 'InitializeMultibody',
                                'DynamicCoupled']
        SimInfo.solvers['SHARPy']['case'] = 'double_pendulum_geradin'
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

    # def tearDown():
        # pass

    def test_doublependulum(self):
        import sharpy.sharpy_main

        solver_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/double_pendulum_geradin.solver.txt')
        print(solver_path)
        sharpy.sharpy_main.main(['', solver_path])

        # read output and compare
        output_path = os.path.dirname(solver_path) + '/output/double_pendulum_geradin/WriteVariablesTime/'
        # quat_data = np.matrix(np.genfromtxt(output_path + 'FoR_00_mb_quat.dat', delimiter=' '))
        pos_tip_data = np.matrix(np.genfromtxt(output_path + "struct_pos_node" + str(11+11-1) + ".dat", delimiter=' '))
        self.assertAlmostEqual(pos_tip_data[-1, 1], 1.481168, 4)
        self.assertAlmostEqual(pos_tip_data[-1, 2], 0.000000, 4)
        self.assertAlmostEqual(pos_tip_data[-1, 3], 0.8766285, 4)

if __name__=='__main__':

    T = TestDoublePendulum()
    T.setUp()
    T.test_doublependulum()
