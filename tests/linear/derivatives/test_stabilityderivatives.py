import numpy as np
import os
import unittest
import cases.templates.flying_wings as wings
import sharpy.sharpy_main
import h5py
import sharpy.utils.h5utils as h5utils
import sharpy.utils.algebra as algebra
import sharpy.linear.src.libss as libss


class TestLinearDerivatives(unittest.TestCase):

    def run_sharpy(self, alpha_deg, flow, target_system, not_run=False):
        # Problem Set up
        u_inf = 10.
        rho = 1.02
        num_modes = 4

        # Lattice Discretisation
        M = 4
        N = 4
        M_star_fact = 10

        # Linear UVLM settings
        integration_order = 2
        remove_predictor = False
        use_sparse = True

        # ROM Properties
        rom_settings = dict()
        rom_settings['algorithm'] = 'mimo_rational_arnoldi'
        rom_settings['r'] = 6
        rom_settings['single_side'] = 'observability'
        frequency_continuous_k = np.array([0.])

        # Case Admin - Create results folders
        case_name = 'wing'
        if target_system == 'aerodynamic':
            case_name += '_uvlm'
        elif target_system == 'aeroelastic':
            case_name += '_coupled'
        else:
            NameError('Unrecognised system')

        case_nlin_info = '_uinf{:04g}_a{:04g}'.format(u_inf*10, alpha_deg*100)
        case_name += case_nlin_info

        self.route_test_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

        # SHARPy nonlinear reference solution
        ws = wings.QuasiInfinite(M=M,
                                 aspect_ratio=10,
                                 N=N,
                                 Mstar_fact=M_star_fact,
                                 u_inf=u_inf,
                                 alpha=alpha_deg,
                                 rho=rho,
                                 sweep=0,
                                 n_surfaces=2,
                                 route=self.route_test_dir + '/cases',
                                 case_name=case_name)

        ws.gust_intensity = 0.01
        ws.sigma = 1e-1

        ws.clean_test_files()
        ws.update_derived_params()
        ws.set_default_config_dict()

        ws.generate_aero_file()
        ws.generate_fem_file()

        frequency_continuous_w = 2 * u_inf * frequency_continuous_k / ws.c_ref
        rom_settings['frequency'] = frequency_continuous_w

        ws.config['SHARPy'] = {
            'flow': flow,
            'case': ws.case_name, 'route': ws.route,
            'write_screen': 'off', 'write_log': 'on',
            'log_folder': self.route_test_dir + '/output/',
            'log_file': ws.case_name + '.log'}

        ws.config['BeamLoader'] = {
            'unsteady': 'off',
            'orientation': ws.quat}

        ws.config['AerogridLoader'] = {
            'unsteady': 'off',
            'aligned_grid': 'on',
            'mstar': ws.Mstar_fact * ws.M,
            'freestream_dir': ws.u_inf_direction,
            'wake_shape_generator': 'StraightWake',
            'wake_shape_generator_input': {'u_inf': ws.u_inf,
                                           'u_inf_direction': ws.u_inf_direction,
                                           'dt': ws.dt}}

        ws.config['StaticUvlm'] = {
            'rho': ws.rho,
            'velocity_field_generator': 'SteadyVelocityField',
            'velocity_field_input': {
                'u_inf': ws.u_inf,
                'u_inf_direction': ws.u_inf_direction},
            'rollup_dt': ws.dt,
            'print_info': 'on',
            'horseshoe': 'off',
            'num_cores': 4,
            'n_rollup': 0,
            'rollup_aic_refresh': 0,
            'rollup_tolerance': 1e-4}

        ws.config['StaticCoupled'] = {
            'print_info': 'on',
            'max_iter': 200,
            'n_load_steps': 1,
            'tolerance': 1e-10,
            'relaxation_factor': 0.,
            'aero_solver': 'StaticUvlm',
            'aero_solver_settings': {
                'rho': ws.rho,
                'print_info': 'off',
                'horseshoe': 'off',
                'num_cores': 4,
                'n_rollup': 0,
                'rollup_dt': ws.dt,
                'rollup_aic_refresh': 1,
                'rollup_tolerance': 1e-4,
                'velocity_field_generator': 'SteadyVelocityField',
                'velocity_field_input': {
                    'u_inf': ws.u_inf,
                    'u_inf_direction': ws.u_inf_direction}},
            'structural_solver': 'NonLinearStatic',
            'structural_solver_settings': {'print_info': 'off',
                                           'max_iterations': 150,
                                           'num_load_steps': 4,
                                           'delta_curved': 1e-1,
                                           'min_delta': 1e-10,
                                           'gravity_on': 'off',
                                           'gravity': 9.81}}

        ws.config['AerogridPlot'] = {'include_rbm': 'off',
                                     'include_applied_forces': 'on',
                                     'minus_m_star': 0}

        ws.config['AeroForcesCalculator'] = {'write_text_file': 'on',
                                             # 'text_file_name': ws.case_name + '_aeroforces.csv',
                                             'screen_output': 'on',
                                             'unsteady': 'off'}

        ws.config['BeamPlot'] = {'include_rbm': 'off',
                                 'include_applied_forces': 'on'}

        ws.config['BeamCsvOutput'] = {'output_pos': 'on',
                                      'output_psi': 'on',
                                      'screen_output': 'on'}

        ws.config['Modal'] = {'print_info': 'on',
                              'use_undamped_modes': 'on',
                              'NumLambda': 20,
                              'rigid_body_modes': 'on',
                              'write_modes_vtk': 'on',
                              'print_matrices': 'off',
                              'save_data': 'on',
                              'rigid_modes_cg': 'off'}

        settings = {}
        settings['NonLinearDynamicCoupledStep'] = {'print_info': 'off',
                                                   'max_iterations': 950,
                                                   'delta_curved': 1e-1,
                                                   'min_delta': ws.tolerance,
                                                   'newmark_damp': 5e-3,
                                                   'gravity_on': 'on',
                                                   'gravity': 9.81,
                                                   'num_steps': 4,
                                                   'dt': ws.dt,
                                                   'initial_velocity': u_inf}

        relative_motion = 'off'
        settings['StepUvlm'] = {'print_info': 'off',
                                'horseshoe': 'off',
                                'num_cores': 4,
                                'n_rollup': 0,
                                'convection_scheme': 2,
                                'rollup_dt': ws.dt,
                                'rollup_aic_refresh': 1,
                                'rollup_tolerance': 1e-4,
                                'gamma_dot_filtering': 6,
                                'vortex_radius': 1e-8,
                                'velocity_field_generator': 'SteadyVelocityField',
                                'velocity_field_input': {'u_inf': 0 * u_inf,
                                                         'u_inf_direction': [1., 0, 0]},
                                'rho': rho,
                                'n_time_steps': 1,
                                'dt': ws.dt}

        solver = 'NonLinearDynamicCoupledStep'
        ws.config['DynamicCoupled'] = {'structural_solver': solver,
                                      'structural_solver_settings': settings[solver],
                                      'aero_solver': 'StepUvlm',
                                      'aero_solver_settings': settings['StepUvlm'],
                                      'fsi_substeps': 200,
                                      'fsi_tolerance': ws.tolerance,
                                      'relaxation_factor': ws.relaxation_factor,
                                      'minimum_steps': 1,
                                      'relaxation_steps': 150,
                                      'final_relaxation_factor': 0.5,
                                      'n_time_steps': 1,
                                      'dt': ws.dt,
                                      'include_unsteady_force_contribution': 'off',
                                                                  }

        ws.config['LinearAssembler'] = {'linear_system': 'LinearAeroelastic',
                                       'linear_system_settings': {
                                           'track_body': 'off',
                                           'beam_settings': {'modal_projection': 'on',
                                                             'inout_coords': 'modes',
                                                             'discrete_time': 'off',
                                                             'newmark_damp': 5e-4,
                                                             'discr_method': 'newmark',
                                                             'dt': ws.dt,
                                                             'proj_modes': 'undamped',
                                                             'use_euler': 'on',
                                                             'num_modes': 20,
                                                             'print_info': 'on',
                                                             'gravity': 'off',
                                                             'remove_dofs': [],
                                                             'remove_rigid_states': 'on'},
                                           'aero_settings': {'dt': ws.dt,
                                                             # 'ScalingDict': {'density': rho,
                                                             #                 'length': ws.c_ref * 0.5,
                                                             #                 'speed': u_inf},
                                                             'integr_order': 2,
                                                             'density': rho,
                                                             'remove_predictor': 'off',
                                                             'use_sparse': 'off',
                                                             'vortex_radius': 1e-8,
                                                             'convert_to_ct': 'on',
                                                             'remove_inputs': ['u_gust'],
                                                             # 'rom_method': ['Krylov'],
                                                             'rom_method_settings': {
                                                                 'Krylov': {'algorithm': 'mimo_rational_arnoldi',
                                                                            'frequency': [0.],
                                                                            'r': 4,
                                                                            'single_side': 'observability'}}
                                                             },
                                           'use_euler': 'on',
                                       },
                                       }

        ws.config['FrequencyResponse'] = {'quick_plot': 'off',
                                          'frequency_unit': 'k',
                                          'frequency_bounds': [0.0001, 1.0],
                                          'num_freqs': 100,
                                          'frequency_spacing': 'log',
                                          'target_system': ['aeroelastic'],
                                          }

        ws.config['StabilityDerivatives'] = {'u_inf': ws.u_inf,
                                             'c_ref': ws.main_chord,
                                             'b_ref': ws.wing_span,
                                             'S_ref': ws.wing_span * ws.main_chord,
                                             }

        ws.config['AsymptoticStability'] = {'print_info': 'on'}

        ws.config['SaveParametricCase'] = {'save_case': 'off',
                                           'parameters': {'alpha': alpha_deg}}

        ws.config['SaveData'] = {'save_aero': 'off',
                                 'save_struct': 'off',
                                 'save_linear': 'on',
                                 'save_linear_uvlm': 'on',
                                 'format': 'mat',
                                 }

        ws.config.write()

        if not not_run:
            self.data = sharpy.sharpy_main.main(['', ws.route + ws.case_name + '.sharpy'])

        return ws

    def test_derivatives(self):
        # target_system_list = ['aerodynamic']#, 'aeroelastic']
        # target_system_list = ['aeroelastic']
        target_system_list = ['aerodynamic', 'aeroelastic']
        for system in target_system_list:
            with self.subTest(target_system=system):
                self.run_case(target_system=system)

    def run_case(self, target_system):

        if target_system == 'aerodynamic':
            nonlinear_solver = ['StaticUvlm']
        elif target_system == 'aeroelastic':
            nonlinear_solver = ['StaticCoupled']
        else:
            NameError('Unrecognised system')

        case_name_db = []
        not_run = False # for debugging
        # Reference Case at 4 degrees
        # Run nonlinear simulation and save linerised ROM
        alpha_deg_ref = 0
        alpha0 = alpha_deg_ref * np.pi/180
        ref = self.run_sharpy(alpha_deg=alpha_deg_ref,
                              flow=['BeamLoader',
                                    'AerogridLoader', ] +
                                   nonlinear_solver +
                                   ['AeroForcesCalculator',
                                    'Modal',
                                    'LinearAssembler',
                                    # 'AsymptoticStability',
                                    'StabilityDerivatives',
                                    'SaveData',
                                    'SaveParametricCase'],
                              target_system=target_system,
                              not_run=False)
        u_inf = ref.u_inf
        ref_case_name = ref.case_name
        case_name_db.append(ref_case_name)
        qS = 0.5 * ref.rho * u_inf ** 2 * ref.c_ref * ref.wing_span
        print(ref.wing_span, ref.main_chord, ref.aspect_ratio)

        # Run nonlinear cases in the vicinity
        nonlinear_sim_flow = ['BeamLoader',
                              'AerogridLoader'] + nonlinear_solver + ['AeroForcesCalculator', 'SaveParametricCase']
        alpha_deg_min = alpha_deg_ref - 5e-3
        alpha_deg_max = alpha_deg_ref + 5e-3
        n_evals = 11
        alpha_vec = np.linspace(alpha_deg_min, alpha_deg_max, n_evals)
        nlin_forces_g = np.zeros((n_evals, 3))
        nlin_forces_a = np.zeros((n_evals, 3))
        for ith, alpha in enumerate(alpha_vec):
            if alpha == alpha_deg_ref:
                case_name = ref_case_name
                continue
            else:
                case = self.run_sharpy(alpha, flow=nonlinear_sim_flow, target_system=target_system, not_run=not_run)
                case_name_db.append(case.case_name)
                case_name = case.case_name
            nlin_forces = np.loadtxt(self.route_test_dir +
                                     '/output/{:s}/forces/forces_aeroforces.txt'.format(case_name),
                                     skiprows=1,
                                     delimiter=',')
            nlin_forces_g[ith, :3] = nlin_forces[1:4]
            nlin_forces_a[ith, :3] = nlin_forces[7:10]
        nlin_forces_g /= qS
        nlin_forces_a /= qS

        print('Nonlinear coefficients')
        lift_poly = np.polyfit(alpha_vec * np.pi/180, nlin_forces_g[:, 2], deg=1)
        nonlin_cla = lift_poly[0]
        print('CLa', nonlin_cla)
        drag_poly = np.polyfit(alpha_vec * np.pi/180, nlin_forces_g[:, 0], deg=2)
        nonlin_cda = 2 * drag_poly[0] * alpha0 + drag_poly[1]
        print('CDa', nonlin_cda)

        print('Nonlinear coefficients - body axes')
        lift_poly = np.polyfit(alpha_vec * np.pi/180, nlin_forces_a[:, 2], deg=1)
        nonlin_cza = lift_poly[0]
        print('CZa', nonlin_cza)
        drag_poly = np.polyfit(alpha_vec * np.pi/180, nlin_forces_a[:, 0], deg=2)
        nonlin_cxa = 2 * drag_poly[0] * alpha0 + drag_poly[1]
        print('CXa', nonlin_cxa)

        # Get Linear ROM at reference case
        import scipy.io as scio
        linuvlm_data = scio.loadmat(self.route_test_dir + '/output/{:s}/savedata/{:s}.uvlmss.mat'.format(ref_case_name,
                                                                                                     ref_case_name))
        linss_data = scio.loadmat(self.route_test_dir + '/output/{:s}/savedata/{:s}.linss.mat'.format(ref_case_name,
                                                                                                  ref_case_name))
        
        # Steady State transfer function
        if target_system == 'aerodynamic':
            A, B, C, D = linuvlm_data['A'], linuvlm_data['B'], linuvlm_data['C'], linuvlm_data['D']
            try:
                dt = linuvlm_data['dt']
            except KeyError:
                dt = None
        else:
            A, B, C, D = linss_data['A'], linss_data['B'], linss_data['C'], linss_data['D']
            try:
                dt = linss_data['dt']
            except KeyError:
                dt = None

        ss = libss.StateSpace(A, B, C, D, dt=dt)
        H0 = ss.freqresp(np.array([1e-5]))[:, :, 0].real

        cga = algebra.quat2rotation(algebra.euler2quat(np.array([0, alpha0, 0])))

        vx_ind = 20  # x_a input index
        vz_ind = 20 + 2  # z_a input index

        n_evals = 2
        forces = np.zeros((n_evals, 4))
        moments = np.zeros_like(forces)
        body_forces = np.zeros_like(forces)

        phi = linss_data['mode_shapes'].copy().real

        eps = 1e-5
        dalpha_vec = np.array([-eps, +eps])
        for i_alpha, dalpha in enumerate(dalpha_vec):
            print(dalpha)
            alpha = alpha0 + dalpha  # rad
            deuler = np.array([0, dalpha, 0])
            euler0 = np.array([0, alpha0, 0])

            u = np.zeros((B.shape[1]))  # input vector
            V0 = np.array([-1, 0, 0], dtype=float) * u_inf  # G
            Vp = u_inf * np.array([-np.cos(dalpha), 0, -np.sin(dalpha)])  # G

            dvg = Vp - V0  # G
            dva = cga.T.dot(dvg)  # A
            dvz = dva[2]
            dvx = dva[0]

            #         print(Vp)
            #         print(algebra.euler2rot(deuler).T.dot(V0))
            #         print(algebra.der_Peuler_by_v(euler0*0, V0))
            Vp2 = algebra.euler2rot(euler0 * 0).T.dot(V0) + algebra.der_Peuler_by_v(euler0 * 0, V0).dot(deuler)
            dva2 = cga.T.dot(algebra.euler2rot(deuler).T.dot(V0) - V0)
            dva3 = cga.T.dot(Vp2 - V0)
            #         print('{:.4f}\t'.format((alpha0 + dalpha) * 180 / np.pi), dva2, dva3, dva3-dva2)
            dvz = dva3[2]
            dvx = dva3[0]

            # Need to scale the mode shapes by the rigid body mode factor
            u[vx_ind] = dvx / phi[-9, 0]
            u[vz_ind] = dvz / phi[-7, 2]

            # and the same with the output forces
            flin = H0.dot(u)[:3].real / phi[-9, 0]  # A
            mlin = np.linalg.inv(phi[-6:-3, 3:6].T).dot(H0.dot(u)[3:6].real)  # A
            F0A = linss_data['forces_aero_beam_dof'][0, :3].real / phi[-9, 0]  # A - forces at the linearisation
            M0A = linss_data['forces_aero_beam_dof'][0, 3:6].real / phi[-6, 3]  # A - forces at the linearisation
            LD0 = cga.dot(F0A)  # Lift and drag at the linearisation point
            M0G = cga.dot(M0A)

            forces[i_alpha, 0] = (alpha0 + dalpha) * 180 / np.pi  # deg
            LD = LD0 + algebra.der_Ceuler_by_v(euler0, F0A).dot(deuler) + cga.dot(flin)  # stability axes
            forces[i_alpha, 1:] = LD / qS

            u_body = np.zeros_like(u)
            u_body[vz_ind] = dvz / phi[-7, 2]
            body_forces[i_alpha, 1:] = H0.dot(u_body)[:3].real / phi[-9, 0] / qS # C_Z_w in A frame
            print(H0[:3, vx_ind:vx_ind+3].real / phi[-7, 2] / phi[-9, 0] / qS)

            MD = M0G + algebra.der_Ceuler_by_v(euler0, M0A).dot(deuler) + cga.dot(mlin)
            moments[i_alpha, 0] = forces[i_alpha, 0]
            moments[i_alpha, 1:] = MD / qS / ref.main_chord

        cla = (forces[-1, -1] - forces[0, -1]) / (forces[-1, 0] - forces[0, 0]) * 180 / np.pi
        print('Lift curve slope perturbation {:.6e}'.format(cla))

        cda = (forces[-1, 1] - forces[0, 1]) / (forces[-1, 0] - forces[0, 0]) * 180 / np.pi
        print('Drag curve slope perturbation {:.6e}'.format(cda))

        cma = (moments[-1, 2] - moments[0, 2]) / (moments[-1, 0] - moments[0, 0]) * 180 / np.pi
        print('Moment curve slope perturbation {:.6e}'.format(cma))

        # body derivative
        # czwa = (body_forces[-1, -1] - body_forces[0, -1]) / (forces[-1, 0] - forces[0, 0]) * 180 / np.pi
        # print('Vertical force with vertical velocity perturbation {:.6e}'.format(czwa))

        with h5py.File(self.route_test_dir + '/output/' + ref_case_name +
                       '/derivatives/aerodynamic_stability_derivatives.h5', 'r') as f:
            sharpy_force_angle = h5utils.load_h5_in_dict(f)['force_angle_velocity']

        linsubtests = ((2, cla, 'lift'),
                       (0, cda, 'drag'),
                       (4, cma, 'moment'))
        for test in linsubtests:
            with self.subTest(test):
                try:
                    np.testing.assert_almost_equal(sharpy_force_angle[test[0], 1], test[1], decimal=3)
                except AssertionError as e:
                    print('Error Linear perturbation in {:s}'.format(test[2]))
                    print(e)
                    print('Pct error', np.abs(sharpy_force_angle[test[0], 1] - test[1]) / test[1] * 100)
                    raise AssertionError

        nonlinsubtests = ((2, nonlin_cla, 'lift'),
                          (0, nonlin_cda, 'drag'),
                          )

        for test in nonlinsubtests:
            with self.subTest(test):
                try:
                    np.testing.assert_almost_equal(sharpy_force_angle[test[0], 1], test[1], decimal=2)
                except AssertionError as e:
                    print('Error nonlinear perturbation in {:s}'.format(test[2]))
                    print(e)
                    print('Pct error', np.abs(sharpy_force_angle[test[0], 1] - test[1]) / test[1] * 100)
                    raise AssertionError
        return forces, moments

    def tearDown(self):
        import shutil
        folders = ['cases', 'output']
        for folder in folders:
            shutil.rmtree(self.route_test_dir + '/' + folder)


if __name__ == '__main__':
    unittest.main()
