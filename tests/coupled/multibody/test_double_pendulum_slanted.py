import numpy as np
import unittest
import os
import shutil
from sharpy.utils.constants import deg2rad


class TestDoublePendulumSlanted(unittest.TestCase):
    """
    Validation of a double pendulum with distributed mass and flared hinge axis at the connection

    As given in https://dx.doi.org/10.2514/6.2024-1441
    """

    def _setUp(self, lateral):
        import sharpy.utils.generate_cases as gc
        import sharpy.utils.algebra as ag

        # Structural properties

        length_beam = 0.5  #meters
        cx_length = 0.02  #meters

        A = cx_length * cx_length  #assume rectangular cross section a= d^2
        material_density = 2700.0  #kg/m^3

        mass_per_unit_length = material_density * A  #kg/m
        mass_iner = (mass_per_unit_length) * (cx_length * cx_length + cx_length * cx_length) / (12.0)

        EA = 2.800e7
        GA = 1.037e7
        GJ = 6.914e2
        EI = 9.333e2

        lateral_ini = lateral

        # Beam1
        global nnodes1
        nnodes1 = 11
        l1 = length_beam
        m1 = 0.0
        theta_ini1 = 90. * deg2rad

        # Beam2
        nnodes2 = nnodes1
        l2 = l1
        m2 = m1
        theta_ini2 = 90. * deg2rad

        # airfoils
        airfoil = np.zeros((1, 20, 2), )
        airfoil[0, :, 0] = np.linspace(0., 1., 20)

        # Simulation
        numtimesteps = 30
        dt = 0.01

        # Create the structure
        beam1 = gc.AeroelasticInformation()
        r1 = np.linspace(0.0, l1, nnodes1)
        node_pos1 = np.zeros((nnodes1, 3), )
        node_pos1[:, 0] = r1 * np.sin(theta_ini1) * np.cos(lateral_ini)
        node_pos1[:, 1] = r1 * np.sin(theta_ini1) * np.sin(lateral_ini)
        node_pos1[:, 2] = -r1 * np.cos(theta_ini1)
        beam1.StructuralInformation.generate_uniform_beam(node_pos1, mass_per_unit_length, mass_iner, mass_iner / 2.0,
                                                          mass_iner / 2.0, np.zeros((3,), ), EA, GA, GA, GJ, EI, EI,
                                                          num_node_elem=3, y_BFoR='y_AFoR', num_lumped_mass=1)
        beam1.StructuralInformation.body_number = np.zeros((beam1.StructuralInformation.num_elem,), dtype=int)
        beam1.StructuralInformation.boundary_conditions[0] = 1
        beam1.StructuralInformation.boundary_conditions[-1] = -1
        beam1.StructuralInformation.lumped_mass_nodes = np.array([nnodes1 - 1], dtype=int)
        beam1.StructuralInformation.lumped_mass = np.ones((1,)) * m1
        beam1.StructuralInformation.lumped_mass_inertia = np.zeros((1, 3, 3))
        beam1.StructuralInformation.lumped_mass_position = np.zeros((1, 3))
        beam1.AerodynamicInformation.create_one_uniform_aerodynamics(
            beam1.StructuralInformation,
            chord=1.,
            twist=0.,
            sweep=0.,
            num_chord_panels=4,
            m_distribution='uniform',
            elastic_axis=0.25,
            num_points_camber=20,
            airfoil=airfoil)

        beam2 = gc.AeroelasticInformation()
        r2 = np.linspace(0.0, l2, nnodes2)
        node_pos2 = np.zeros((nnodes2, 3), )
        node_pos2[:, 0] = r2 * np.sin(theta_ini2) * np.cos(lateral_ini) + node_pos1[-1, 0]
        node_pos2[:, 1] = r2 * np.sin(theta_ini2) * np.sin(lateral_ini) + node_pos1[-1, 1]
        node_pos2[:, 2] = -r2 * np.cos(theta_ini2) + node_pos1[-1, 2] + 0.00000001
        beam2.StructuralInformation.generate_uniform_beam(node_pos2, mass_per_unit_length, mass_iner, mass_iner / 2.0,
                                                          mass_iner / 2.0, np.zeros((3,), ), EA, GA, GA, GJ, EI, EI,
                                                          num_node_elem=3, y_BFoR='y_AFoR', num_lumped_mass=1)
        beam2.StructuralInformation.body_number = np.zeros((beam1.StructuralInformation.num_elem,), dtype=int)
        beam2.StructuralInformation.boundary_conditions[0] = 1
        beam2.StructuralInformation.boundary_conditions[-1] = -1
        beam2.StructuralInformation.lumped_mass_nodes = np.array([nnodes2 - 1], dtype=int)
        beam2.StructuralInformation.lumped_mass = np.ones((1,)) * m2
        beam2.StructuralInformation.lumped_mass_inertia = np.zeros((1, 3, 3))
        beam2.StructuralInformation.lumped_mass_position = np.zeros((1, 3))
        beam2.AerodynamicInformation.create_one_uniform_aerodynamics(
            beam2.StructuralInformation,
            chord=1.,
            twist=0.,
            sweep=0.,
            num_chord_panels=4,
            m_distribution='uniform',
            elastic_axis=0.25,
            num_points_camber=20,
            airfoil=airfoil)

        beam1.assembly(beam2)

        # Simulation details
        SimInfo = gc.SimulationInformation()
        SimInfo.set_default_values()

        SimInfo.define_uinf(np.array([0.0, 1.0, 0.0]), 1.)

        SimInfo.solvers['SHARPy']['flow'] = ['BeamLoader',
                                             'AerogridLoader',
                                             'DynamicCoupled']

        global name_hinge_slanted
        name_hinge_slanted = 'name_hinge_slanted'
        SimInfo.solvers['SHARPy']['case'] = name_hinge_slanted
        SimInfo.solvers['SHARPy']['write_screen'] = 'off'
        SimInfo.solvers['SHARPy']['route'] = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/'
        SimInfo.solvers['SHARPy']['log_folder'] = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__))) + '/output/'
        SimInfo.set_variable_all_dicts('dt', dt)
        SimInfo.define_num_steps(numtimesteps)
        SimInfo.set_variable_all_dicts('rho', 0.0)
        SimInfo.set_variable_all_dicts('velocity_field_input', SimInfo.solvers['SteadyVelocityField'])
        SimInfo.set_variable_all_dicts('output',
                                       os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/output/')

        SimInfo.solvers['BeamLoader']['unsteady'] = 'on'

        SimInfo.solvers['AerogridLoader']['unsteady'] = 'on'
        SimInfo.solvers['AerogridLoader']['initial_align'] = 'off'
        SimInfo.solvers['AerogridLoader']['aligned_grid'] = 'off'
        SimInfo.solvers['AerogridLoader']['mstar'] = 2
        SimInfo.solvers['AerogridLoader']['wake_shape_generator'] = 'StraightWake'
        SimInfo.solvers['AerogridLoader']['wake_shape_generator_input'] = {'u_inf': 1.,
                                                                           'u_inf_direction': np.array([0., 1., 0.]),
                                                                           'dt': dt}

        SimInfo.solvers['WriteVariablesTime']['FoR_number'] = np.array([0, 1], dtype=int)
        SimInfo.solvers['WriteVariablesTime']['FoR_variables'] = ['for_pos', 'mb_quat']
        SimInfo.solvers['WriteVariablesTime']['structure_nodes'] = np.array([nnodes1 - 1, nnodes1 + nnodes2 - 1],
                                                                            dtype=int)
        SimInfo.solvers['WriteVariablesTime']['structure_variables'] = ['pos']

        SimInfo.solvers['NonLinearDynamicMultibody']['gravity_on'] = True
        SimInfo.solvers['NonLinearDynamicMultibody']['gravity'] = 9.81
        SimInfo.solvers['NonLinearDynamicMultibody']['time_integrator'] = 'NewmarkBeta'
        SimInfo.solvers['NonLinearDynamicMultibody']['time_integrator_settings'] = {'newmark_damp': 0.0,
                                                                                    'dt': dt}
        SimInfo.solvers['NonLinearDynamicMultibody']['write_lm'] = True

        SimInfo.solvers['BeamPlot']['include_FoR'] = True

        SimInfo.solvers['DynamicCoupled']['structural_solver'] = 'NonLinearDynamicMultibody'
        SimInfo.solvers['DynamicCoupled']['structural_solver_settings'] = SimInfo.solvers['NonLinearDynamicMultibody']
        SimInfo.solvers['DynamicCoupled']['aero_solver'] = 'NoAero'
        SimInfo.solvers['DynamicCoupled']['aero_solver_settings'] = SimInfo.solvers['NoAero']
        SimInfo.solvers['DynamicCoupled']['postprocessors'] = ['WriteVariablesTime', 'BeamPlot', 'AerogridPlot']
        SimInfo.solvers['DynamicCoupled']['postprocessors_settings'] = {
            'WriteVariablesTime': SimInfo.solvers['WriteVariablesTime'],
            'BeamPlot': SimInfo.solvers['BeamPlot'],
            'AerogridPlot': SimInfo.solvers['AerogridPlot']
            }

        SimInfo.with_forced_vel = False
        SimInfo.with_dynamic_forces = False

        # Create the MB and BC files
        LC1 = gc.LagrangeConstraint()
        LC1.behaviour = 'hinge_FoR'
        LC1.body_FoR = 0
        LC1.rot_axis_AFoR = np.array([0.0, 1.0, 0.0])
        LC1.scalingFactor = 1e6
        LC1.penaltyFactor = 0.

        LC2 = gc.LagrangeConstraint()
        LC2.behaviour = 'hinge_node_FoR'
        LC2.node_in_body = nnodes1 - 1
        LC2.body = 0
        LC2.body_FoR = 1
        LC2.rot_axisB = np.array([np.sin(45.0 * deg2rad), np.cos(45.0 * deg2rad), 0.0])
        LC2.rot_axisA2 = np.array([np.sin(45.0 * deg2rad), np.cos(45.0 * deg2rad), 0.0])
        LC2.scalingFactor = 1e6
        LC2.penaltyFactor = 0.

        LC = []
        LC.append(LC1)
        LC.append(LC2)

        MB1 = gc.BodyInformation()
        MB1.body_number = 0
        MB1.FoR_position = np.zeros(6)
        MB1.FoR_velocity = np.zeros(6)
        MB1.FoR_acceleration = np.zeros(6)
        MB1.FoR_movement = 'free'
        MB1.quat = ag.rotation2quat(ag.rotation3d_z(lateral_ini) @ ag.quat2rotation(np.array([1., 0., 0., 0.])))

        MB2 = gc.BodyInformation()
        MB2.body_number = 1
        MB2.FoR_position = np.array([node_pos2[0, 0], node_pos2[0, 1], node_pos2[0, 2], 0., 0., 0.])
        MB2.FoR_velocity = np.zeros(6)
        MB2.FoR_acceleration = np.zeros(6)
        MB2.FoR_movement = 'free'
        MB2.quat = ag.rotation2quat(ag.rotation3d_z(lateral_ini) @ ag.quat2rotation(np.array([1., 0., 0., 0.])))

        MB = []
        MB.append(MB1)
        MB.append(MB2)

        # Write files
        gc.clean_test_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
        SimInfo.generate_solver_file()
        SimInfo.generate_dyn_file(numtimesteps)
        beam1.generate_h5_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
        gc.generate_multibody_file(LC, MB, SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])

        # Same case with penalty weights
        global name_hinge_slanted_pen
        name_hinge_slanted_pen = 'name_hinge_slanted_pen'
        SimInfo.solvers['SHARPy']['case'] = name_hinge_slanted_pen

        LC1.scalingFactor = 1e-24
        LC1.penaltyFactor = 1e0
        LC2.scalingFactor = 1e-24
        LC2.penaltyFactor = 1e0

        gc.clean_test_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
        SimInfo.generate_solver_file()
        SimInfo.generate_dyn_file(numtimesteps)
        beam1.generate_h5_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
        gc.generate_multibody_file(LC, MB, SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])

        # Same case with rotated global reference
        global name_hinge_slanted_lateralrot
        name_hinge_slanted_lateralrot = 'name_hinge_slanted_lateralrot'
        SimInfo.solvers['SHARPy']['case'] = name_hinge_slanted_lateralrot

        LC2.rot_axisB = np.array([np.sin(45.0 * deg2rad), np.cos(45.0 * deg2rad), 0.0])
        LC2.rot_axisA2 = np.array([np.sin(45.0 * deg2rad), np.cos(45.0 * deg2rad), 0.0])

        gc.clean_test_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
        SimInfo.generate_solver_file()
        SimInfo.generate_dyn_file(numtimesteps)
        beam1.generate_h5_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
        gc.generate_multibody_file(LC, MB, SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])

        #     # Same case with spherical joints
        #     global name_spherical
        #     name_spherical = 'dpg_spherical'
        #     SimInfo.solvers['SHARPy']['case'] = name_spherical

        #     SimInfo.solvers['NonLinearDynamicMultibody']['time_integrator'] = 'NewmarkBeta'
        #     SimInfo.solvers['NonLinearDynamicMultibody']['time_integrator_settings'] = {'newmark_damp': 0.15,
        #                                                                                     'dt': dt}

        #     LC1 = gc.LagrangeConstraint()
        #     LC1.behaviour = 'spherical_FoR'
        #     LC1.body_FoR = 0
        #     LC1.scalingFactor = 1e6

        #     LC2 = gc.LagrangeConstraint()
        #     LC2.behaviour = 'spherical_node_FoR'
        #     LC2.node_in_body = nnodes1-1
        #     LC2.body = 0
        #     LC2.body_FoR = 1
        #     LC2.scalingFactor = 1e6

        gc.clean_test_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
        SimInfo.generate_solver_file()
        SimInfo.generate_dyn_file(numtimesteps)
        beam1.generate_h5_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
        gc.generate_multibody_file(LC, MB, SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])

    def run_and_assert(self, name, lateral):
        import sharpy.sharpy_main
        import sharpy.utils.algebra as ag

        solver_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/' + name + '.sharpy')
        sharpy.sharpy_main.main(['', solver_path])

        # read output and compare
        output_path = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__))) + '/output/' + name + '/WriteVariablesTime/'
        pos_tip_data = np.loadtxt(("%sstruct_pos_node%d.dat" % (output_path, nnodes1 * 2 - 1)), )
        for_pos_tip_data = np.loadtxt(("%sFoR_%02d_for_pos.dat" % (output_path, 0)), )
        quat_tip_data = np.loadtxt(("%sFoR_%02d_mb_quat.dat" % (output_path, 0)), )

        calc_pos_tip_data = ag.rotation3d_z(-lateral) @ (for_pos_tip_data[-1, 1:4]
                                                         + ag.quat2rotation(quat_tip_data[-1, 1:])
                                                         @ pos_tip_data[-1, 1:])

        self.assertAlmostEqual(calc_pos_tip_data[0], 0.80954978, 4)
        self.assertAlmostEqual(calc_pos_tip_data[1], 0.1024842, 4)
        self.assertAlmostEqual(calc_pos_tip_data[2], -0.48183994, 4)

    def test_doublependulum_hinge_slanted(self):
        lateral_hinge = 0. * deg2rad
        self._setUp(lateral_hinge)
        self.run_and_assert(name_hinge_slanted, lateral_hinge)

    def test_doublependulum_hinge_slanted_pen(self):
        lateral_hinge = 0. * deg2rad
        self._setUp(lateral_hinge)
        self.run_and_assert(name_hinge_slanted_pen, lateral_hinge)

    def test_doublependulum_hinge_slanted_lateralrot(self):
        lateral_hinge = 30. * deg2rad
        self._setUp(lateral_hinge)
        self.run_and_assert(name_hinge_slanted_lateralrot, lateral_hinge)

    def tearDown(self):
        solver_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
        solver_path += '/'
        for name in [name_hinge_slanted, name_hinge_slanted_pen, name_hinge_slanted_lateralrot]:
            files_to_delete = [name + '.aero.h5',
                               name + '.dyn.h5',
                               name + '.fem.h5',
                               name + '.mb.h5',
                               name + '.sharpy']
            for f in files_to_delete:
                os.remove(solver_path + f)

        shutil.rmtree(solver_path + 'output/')


if __name__ == '__main__':
    unittest.main()
