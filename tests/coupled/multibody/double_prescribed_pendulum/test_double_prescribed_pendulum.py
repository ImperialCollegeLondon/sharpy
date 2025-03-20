import numpy as np
import os
import shutil
from matplotlib import pyplot as plt
import unittest

import sharpy.sharpy_main
import sharpy.utils.algebra as ag
import sharpy.utils.generate_cases as gc


class TestDoublePrescribedPendulum(unittest.TestCase):
    # this case is not convenient for using a setup method, as the second case definition depends upon the first
    # we'll just shove it all in the run method
    @staticmethod
    def run_and_assert():
        # Simulation inputs
        case_name_free = 'free_double_pendulum_jax'
        case_name_prescribed = 'prescribed_double_pendulum_jax'

        try:
            shutil.rmtree('' + str(os.path.dirname(os.path.realpath(__file__))) + '/cases/')
        except:
            pass

        try:
            shutil.rmtree('' + str(os.path.dirname(os.path.realpath(__file__))) + '/output/')
        except:
            pass

        os.makedirs('' + str(os.path.dirname(os.path.realpath(__file__))) + '/cases/')
        os.makedirs('' + str(os.path.dirname(os.path.realpath(__file__))) + '/output/')

        case_route = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/cases/'
        case_out_folder = case_route + '/output/'

        total_time = 0.05  # physical time (s)
        dt = 0.002  # time step length (s)
        n_tstep = int(total_time / dt)  # number of time steps
        hinge_ang = np.deg2rad(30.)

        # Beam structural properties
        beam_l = 0.5  # beam length (m)
        beam_yz = 0.02  # cross-section width/height (m)
        beam_A = beam_yz ** 2  # square cross-section area`
        beam_rho = 2700.0  # beam material density (kg/m^3)
        beam_m = beam_rho * beam_A  # beam mass per unit length (kg/m)
        beam_iner = beam_m * beam_yz ** 2 / 6.  # beam inertia per unit length (kg m)

        EA = 2.800e7  # axial stiffness
        GA = 1.037e7  # shear stiffness
        GJ = 6.914e2  # torsional stiffness
        EI = 9.333e2  # bending stiffness

        # Toggle on to make spaghetti
        # GJ *= 5e-2
        # EI *= 5e-2

        # Airfoils (required as the NoAero solver still requires an aero grid)
        airfoil = np.zeros((1, 20, 2))
        airfoil[0, :, 0] = np.linspace(0., 1., 20)

        # Beam 0
        beam0_n_nodes = 11  # number of nodes
        beam0_l = beam_l  # beam length
        beam0_ml = 0.0  # lumped mass
        beam0_theta_ini = np.deg2rad(90.)  # initial beam angle

        # Beam 1
        beam1_n_nodes = 11  # number of nodes
        beam1_l = beam_l  # beam length
        beam1_ml = 0.0  # lumped mass
        beam1_theta_ini = np.deg2rad(90.)  # initial beam angle

        for hinge_dir in ('y',):
            # Create beam 0 structure
            beam0 = gc.AeroelasticInformation()  # use aeroelastic as controllers implemented here
            beam0_r = np.linspace(0., beam0_l, beam0_n_nodes)  # local node placement
            beam0_pos_ini = np.zeros((beam0_n_nodes, 3))  # initial global node placement
            match hinge_dir:
                case 'x':
                    beam0_pos_ini[:, 1] = beam0_r * np.sin(beam0_theta_ini)
                    beam0_pos_ini[:, 2] = -beam0_r * np.cos(beam0_theta_ini)
                    y_BFoR = 'x_AFoR'
                case 'y':
                    beam0_pos_ini[:, 0] = beam0_r * np.sin(beam0_theta_ini)
                    beam0_pos_ini[:, 2] = -beam0_r * np.cos(beam0_theta_ini)
                    y_BFoR = 'y_AFoR'
                case _:
                    raise KeyError

            beam0.StructuralInformation.generate_uniform_beam(beam0_pos_ini, beam_m, beam_iner, beam_iner / 2.0,
                                                              beam_iner / 2.0,
                                                              np.zeros(3), EA, GA, GA, GJ, EI, EI,
                                                              num_node_elem=3, y_BFoR=y_BFoR, num_lumped_mass=1)

            beam0.StructuralInformation.body_number = np.zeros(beam0.StructuralInformation.num_elem, dtype=int)
            beam0.StructuralInformation.boundary_conditions[0] = 1
            beam0.StructuralInformation.boundary_conditions[-1] = -1
            beam0.StructuralInformation.lumped_mass_nodes = np.array([beam0_n_nodes - 1], dtype=int)
            beam0.StructuralInformation.lumped_mass = np.ones(1) * beam0_ml
            beam0.StructuralInformation.lumped_mass_inertia = np.zeros((1, 3, 3))
            beam0.StructuralInformation.lumped_mass_position = np.zeros((1, 3))
            beam0.AerodynamicInformation.create_one_uniform_aerodynamics(
                beam0.StructuralInformation, chord=1., twist=0., sweep=0.,
                num_chord_panels=4, m_distribution='uniform', elastic_axis=0.25,
                num_points_camber=20, airfoil=airfoil)

            # Create beam 1 structure
            beam1 = gc.AeroelasticInformation()
            beam1_r = np.linspace(0., beam1_l, beam0_n_nodes)  # local node placement

            beam1_pos_ini = np.zeros((beam0_n_nodes, 3))  # initial global node placement
            match hinge_dir:
                case 'x':
                    beam1_pos_ini[:, 1] = beam1_r * np.sin(beam1_theta_ini) + beam0_pos_ini[-1, 1]
                    beam1_pos_ini[:, 2] = -beam0_r * np.cos(beam1_theta_ini) + beam0_pos_ini[-1, 2]
                case 'y':
                    beam1_pos_ini[:, 0] = beam1_r * np.sin(beam1_theta_ini) + beam0_pos_ini[-1, 0]
                    beam1_pos_ini[:, 2] = -beam1_r * np.cos(beam1_theta_ini) + beam0_pos_ini[-1, 2]
                case _:
                    raise KeyError

            beam1.StructuralInformation.generate_uniform_beam(beam1_pos_ini, beam_m, beam_iner, beam_iner / 2.0,
                                                              beam_iner / 2.0,
                                                              np.zeros(3), EA, GA, GA, GJ, EI, EI,
                                                              num_node_elem=3, y_BFoR=y_BFoR, num_lumped_mass=1)

            beam1.StructuralInformation.body_number = np.zeros(beam1.StructuralInformation.num_elem, dtype=int)
            beam1.StructuralInformation.boundary_conditions[0] = 1
            beam1.StructuralInformation.boundary_conditions[-1] = -1
            beam1.StructuralInformation.lumped_mass_nodes = np.array([beam1_n_nodes - 1], dtype=int)
            beam1.StructuralInformation.lumped_mass = np.ones(1) * beam1_ml
            beam1.StructuralInformation.lumped_mass_inertia = np.zeros((1, 3, 3))
            beam1.StructuralInformation.lumped_mass_position = np.zeros((1, 3))
            beam1.AerodynamicInformation.create_one_uniform_aerodynamics(
                beam1.StructuralInformation, chord=1., twist=0., sweep=0.,
                num_chord_panels=4, m_distribution='uniform', elastic_axis=0.25,
                num_points_camber=20, airfoil=airfoil)

            # Combine beam1 into beam0
            beam0.assembly(beam1)

            # Create the MB and BC parameters
            # Free hinge
            LC0_free = gc.LagrangeConstraint()
            LC0_free.behaviour = 'hinge_FoR'
            LC0_free.body_FoR = 0
            match hinge_dir:
                case 'x':
                    LC0_free.rot_axis_AFoR = np.array((1., 0., 0.))
                case 'y':
                    LC0_free.rot_axis_AFoR = np.array((0., 1., 0.))
            LC0_free.scalingFactor = dt ** -2

            # Free hinge between beams
            LC1_free = gc.LagrangeConstraint()
            LC1_free.behaviour = 'hinge_node_FoR'
            LC1_free.node_in_body = beam0_n_nodes - 1
            LC1_free.body = 0
            LC1_free.body_FoR = 1
            match hinge_dir:
                case 'x':
                    LC1_free.rot_axisB = np.array((np.sin(hinge_ang), np.cos(hinge_ang), 0.))
                    LC1_free.rot_axisA2 = np.array((np.cos(hinge_ang), np.sin(hinge_ang), 0.))
                case 'y':
                    LC1_free.rot_axisB = np.array((np.sin(hinge_ang), np.cos(hinge_ang), 0.))
                    LC1_free.rot_axisA2 = np.array((np.sin(hinge_ang), np.cos(hinge_ang), 0.))
            LC1_free.scalingFactor = dt ** -2

            LC_free = [LC0_free, LC1_free]  # List of LCs

            MB0 = gc.BodyInformation()
            MB0.body_number = 0
            MB0.FoR_position = np.zeros(6)
            MB0.FoR_velocity = np.zeros(6)
            MB0.FoR_acceleration = np.zeros(6)
            MB0.FoR_movement = 'free'
            MB0.quat = np.array([1., 0., 0., 0.])

            MB1 = gc.BodyInformation()
            MB1.body_number = 1
            MB1.FoR_position = np.array([*beam1_pos_ini[0, :], 0., 0., 0.])
            MB1.FoR_velocity = np.zeros(6)
            MB1.FoR_acceleration = np.zeros(6)
            MB1.FoR_movement = 'free'
            MB1.quat = np.array((1., 0., 0., 0.))

            MB = [MB0, MB1]  # List of MBs

            # Simulation details
            SimInfo = gc.SimulationInformation()
            SimInfo.set_default_values()
            SimInfo.with_forced_vel = False
            SimInfo.with_dynamic_forces = False

            SimInfo.solvers['SHARPy']['flow'] = [
                'BeamLoader',
                'AerogridLoader',
                'DynamicCoupled'
            ]

            SimInfo.solvers['SHARPy']['case'] = case_name_free
            SimInfo.solvers['SHARPy']['write_screen'] = False
            SimInfo.solvers['SHARPy']['route'] = case_route
            SimInfo.solvers['SHARPy']['log_folder'] = case_out_folder
            SimInfo.set_variable_all_dicts('dt', dt)
            SimInfo.define_num_steps(n_tstep)
            SimInfo.set_variable_all_dicts('output', case_out_folder)

            SimInfo.solvers['BeamLoader']['unsteady'] = 'on'

            SimInfo.solvers['AerogridLoader']['unsteady'] = 'on'
            SimInfo.solvers['AerogridLoader']['mstar'] = 2
            SimInfo.solvers['AerogridLoader']['wake_shape_generator'] = 'StraightWake'
            SimInfo.solvers['AerogridLoader']['wake_shape_generator_input'] = {'u_inf': 1.,
                                                                               'u_inf_direction': np.array(
                                                                                   (0., 1., 0.)),
                                                                               'dt': dt}

            SimInfo.solvers['NonLinearDynamicMultibodyJAX']['write_lm'] = False
            SimInfo.solvers['NonLinearDynamicMultibodyJAX']['gravity_on'] = True
            SimInfo.solvers['NonLinearDynamicMultibodyJAX']['gravity'] = 9.81
            SimInfo.solvers['NonLinearDynamicMultibodyJAX']['time_integrator'] = 'NewmarkBetaJAX'
            SimInfo.solvers['NonLinearDynamicMultibodyJAX']['time_integrator_settings'] = {'newmark_damp': 0.0,
                                                                                           'dt': dt}

            SimInfo.solvers['BeamPlot']['include_FoR'] = False

            SimInfo.solvers['DynamicCoupled']['structural_solver'] = 'NonLinearDynamicMultibodyJAX'
            SimInfo.solvers['DynamicCoupled']['structural_solver_settings'] = SimInfo.solvers[
                'NonLinearDynamicMultibodyJAX']
            SimInfo.solvers['DynamicCoupled']['aero_solver'] = 'NoAero'
            SimInfo.solvers['DynamicCoupled']['aero_solver_settings'] = SimInfo.solvers['NoAero']
            SimInfo.solvers['DynamicCoupled']['postprocessors'] = ['BeamPlot']
            SimInfo.solvers['DynamicCoupled']['postprocessors_settings'] = {'BeamPlot': {}}

            # Write files
            gc.clean_test_files(case_route, case_name_free)
            SimInfo.generate_solver_file()
            SimInfo.generate_dyn_file(n_tstep)
            beam0.generate_h5_files(case_route, case_name_free)
            gc.generate_multibody_file(LC_free, MB, case_route, case_name_free, use_jax=True)

            # Run SHARPy for free case
            case_data_free = sharpy.sharpy_main.main(['', case_route + case_name_free + '.sharpy'])

            # Calculate constraint angles
            quat0 = [case_data_free.structure.timestep_info[i].mb_quat[0, :] for i in range(n_tstep)]
            quat1 = [case_data_free.structure.timestep_info[i].mb_quat[1, :] for i in range(n_tstep)]
            psi_ga = [case_data_free.structure.timestep_info[i].psi[int((beam0_n_nodes - 3) / 2), 1, :] for i in
                      range(n_tstep)]

            psi_a = [ag.quat2crv(quat0[i]) for i in range(n_tstep)]

            psi_hg = [ag.rotation2crv(ag.quat2rotation(quat0[i])
                                      @ ag.crv2rotation(psi_ga[i])
                                      @ ag.quat2rotation(quat1[i]).T) for i in range(n_tstep)]

            # Calculate constraint velocities
            omega_a = [case_data_free.structure.timestep_info[i].mb_FoR_vel[0, 3:] for i in range(n_tstep)]
            omega_h = [case_data_free.structure.timestep_info[i].mb_FoR_vel[1, 3:] for i in range(n_tstep)]
            psi_ga_dot = [case_data_free.structure.timestep_info[i].psi_dot[int((beam0_n_nodes - 3) / 2), 1, :] for i in
                          range(n_tstep)]

            psi_a_dot = [np.linalg.solve(ag.crv2tan(psi_a[i]), omega_a[i]) for i in range(n_tstep)]

            term1 = [-ag.crv2tan(psi_ga[i]) @ psi_ga_dot[i]
                     - ag.crv2rotation(psi_ga[i]).T @ omega_a[i]
                     + ag.crv2rotation(psi_ga[i]).T @ ag.quat2rotation(quat0[i]).T @ ag.quat2rotation(quat1[i]) @
                     omega_h[i]
                     for i in range(n_tstep)]

            psi_hg_dot = [np.linalg.inv(ag.crv2tan(psi_hg[i])) @ ag.crv2rotation(psi_hg[i]) @ term1[i] for i in
                          range(n_tstep)]

            # Controller inputs
            angle_input_file0 = case_out_folder + f'angle_input0_{hinge_dir}.npy'
            angle_input_file1 = case_out_folder + f'angle_input1_{hinge_dir}.npy'
            vel_input_file0 = case_out_folder + f'vel_input0_{hinge_dir}.npy'
            vel_input_file1 = case_out_folder + f'vel_input1_{hinge_dir}.npy'

            angle_input0 = np.array(psi_a)
            angle_input1 = np.array(psi_hg)
            vel_input0 = np.array(psi_a_dot)
            vel_input1 = np.array(psi_hg_dot)

            # Uncomment to stop prescribed pendulum tracking after a certain timestep (prove both cases aren't actually free!)
            # angle_input1[int(n_tstep/2):, :] = angle_input1[int(n_tstep/2) - 1, :]
            # vel_input1[int(n_tstep/2):, :] = 0.

            np.save(angle_input_file0, angle_input0)
            np.save(angle_input_file1, angle_input1)
            np.save(vel_input_file0, vel_input0)
            np.save(vel_input_file1, vel_input1)

            SimInfo.define_num_steps(n_tstep - 1)

            # Prescribe top joint angular velocity
            LC0_prescribed = gc.LagrangeConstraint()
            LC0_prescribed.behaviour = 'control_rot_vel_FoR'
            LC0_prescribed.controller_id = 'controller0'
            LC0_prescribed.body_FoR = 0
            LC0_prescribed.scalingFactor = dt ** -2

            # Actuated joint between beams
            LC1_prescribed = gc.LagrangeConstraint()
            LC1_prescribed.behaviour = 'control_node_FoR_rot_vel'
            LC1_prescribed.controller_id = 'controller1'
            LC1_prescribed.node_in_body = beam1_n_nodes - 1
            LC1_prescribed.body = 0
            LC1_prescribed.body_FoR = 1
            LC1_prescribed.scalingFactor = dt ** -2
            LC1_prescribed.rel_posB = np.zeros(3)

            LC_prescribed = [LC0_prescribed, LC1_prescribed]  # List of LCs

            SimInfo.solvers['SHARPy']['case'] = case_name_prescribed
            SimInfo.solvers['DynamicCoupled']['controller_id'] = {
                'controller0': 'MultibodyController',
                'controller1': 'MultibodyController',
            }
            SimInfo.solvers['DynamicCoupled']['controller_settings']['controller0'] = {
                'ang_history_input_file': angle_input_file0,
                'ang_vel_history_input_file': vel_input_file0,
                'dt': dt,
            }
            SimInfo.solvers['DynamicCoupled']['controller_settings']['controller1'] = {
                'ang_history_input_file': angle_input_file1,
                'ang_vel_history_input_file': vel_input_file1,
                'dt': dt,
            }

            # Write files
            gc.clean_test_files(case_route, case_name_prescribed)
            SimInfo.generate_solver_file()
            SimInfo.generate_dyn_file(n_tstep)
            beam0.generate_h5_files(case_route, case_name_prescribed)
            gc.generate_multibody_file(LC_prescribed, MB, case_route, case_name_prescribed, use_jax=True)

            # Run SHARPy for prescribed case
            case_data_prescribed = sharpy.sharpy_main.main(['', case_route + case_name_prescribed + '.sharpy'])

            # compare (controller framework operates a timestep behind)
            # comparing y tip displacement
            pos_free = case_data_free.structure.timestep_info[-3].pos[-1, 1]
            pos_prescribed = case_data_prescribed.structure.timestep_info[-1].pos[-1, 1]

            diff = np.abs((pos_free / pos_prescribed) - 1.)

            if diff > 1e-3:
                raise ValueError(f"Free and prescribed pendulum results no not closely match: diff={diff}")


    def test_double_prescribed_pendulum(self):
        self.run_and_assert()

    def tearDown(self):
        solver_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/'

        shutil.rmtree(solver_path + 'cases/')
        shutil.rmtree(solver_path + 'output/')

    def teardown_method(self):
        pass

if __name__ == '__main__':
    unittest.main()
