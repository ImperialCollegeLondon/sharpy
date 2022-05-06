import numpy as np
import os
import unittest
import sharpy.cases.templates.flying_wings as wings
import sharpy.sharpy_main
from sharpy.linear.assembler.lineargustassembler import campbell
import pickle


class TestGolandControlSurface(unittest.TestCase):

    def setUp(self):
        self.route_test_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

    def run_sharpy(self, flow, **kwargs):
        # Problem Set up
        u_inf = 1.
        alpha_deg = 0.
        rho = 1.02
        num_modes = 4
        restart = kwargs.get('restart', False)

        # Lattice Discretisation
        M = 4
        N = 16
        M_star_fact = 1

        # Linear UVLM settings
        integration_order = 2
        remove_predictor = kwargs.get('remove_predictor', False)
        use_sparse = True

        # Case Admin - Create results folders
        case_name = 'goland_cs'
        case_nlin_info = 'M%dN%dMs%d_nmodes%d' % (M, N, M_star_fact, num_modes)

        case_name += case_nlin_info

        fig_folder = self.route_test_dir + '/figures/'
        os.makedirs(fig_folder, exist_ok=True)

        # SHARPy nonlinear reference solution
        ws = wings.GolandControlSurface(M=M,
                                        N=N,
                                        Mstar_fact=M_star_fact,
                                        u_inf=u_inf,
                                        alpha=alpha_deg,
                                        # cs_deflection=[0, 0],
                                        rho=rho,
                                        sweep=0,
                                        physical_time=2,
                                        n_surfaces=2,
                                        route=self.route_test_dir + '/cases',
                                        case_name=case_name)

        ws.gust_intensity = 0.01
        ws.sigma = 1

        ws.clean_test_files()
        ws.update_derived_params()
        ws.set_default_config_dict()

        ws.generate_aero_file()
        ws.generate_fem_file()

        x0 = np.zeros(256)
        lin_tsteps = 40
        u_vec = np.zeros((lin_tsteps, num_modes // 2 * 3 + 1 + 2))
        # elevator
        u_vec[5:, 5] = 10 * np.pi / 180
        # gust
        u_vec[:, 4] = np.sin(np.linspace(0, 2*np.pi, lin_tsteps))
        ws.create_linear_files(x0, u_vec)
        np.savetxt(self.route_test_dir + '/cases/elevator.txt', u_vec[:, 5])
        np.savetxt(self.route_test_dir + '/cases/gust.txt', u_vec[:, 4])

        ws.config['SHARPy'] = {
            'flow': flow,
            'case': ws.case_name, 'route': ws.route,
            'write_screen': 'off', 'write_log': 'on',
            'log_folder': self.route_test_dir + '/output/' + ws.case_name + '/',
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
                                           'gravity_on': 'on',
                                           'gravity': 9.81}}

        ws.config['AerogridPlot'] = {'include_rbm': 'off',
                                     'include_applied_forces': 'on',
                                     'minus_m_star': 0}

        ws.config['AeroForcesCalculator'] = {'write_text_file': 'on',
                                             'text_file_name': ws.case_name + '_aeroforces.csv',
                                             'screen_output': 'on',
                                             'unsteady': 'off'}

        ws.config['BeamPlot'] = {'include_rbm': 'off',
                                 'include_applied_forces': 'on'}

        ws.config['Modal'] = {'NumLambda': 20,
                              'rigid_body_modes': 'off',
                              'print_matrices': 'on',
                              'save_data': 'off',
                              'rigid_modes_cg': 'off',
                              'continuous_eigenvalues': 'off',
                              'dt': 0,
                              'plot_eigenvalues': False,
                              'max_rotation_deg': 15.,
                              'max_displacement': 0.15,
                              'write_modes_vtk': True,
                              'use_undamped_modes': True}

        ws.config['LinearAssembler'] = {'linear_system': 'LinearAeroelastic',
                                        'linear_system_settings': {
                                            'beam_settings': {'modal_projection': 'on',
                                                              'inout_coords': 'modes',
                                                              'discrete_time': 'on',
                                                              'newmark_damp': 0.5e-4,
                                                              'discr_method': 'newmark',
                                                              'dt': ws.dt,
                                                              'proj_modes': 'undamped',
                                                              'use_euler': 'off',
                                                              'num_modes': num_modes,
                                                              'print_info': 'off',
                                                              'gravity': 'on',
                                                              'remove_sym_modes': 'on',
                                                              'remove_dofs': []},
                                            'aero_settings': {'dt': ws.dt,
                                                              'integr_order': integration_order,
                                                              'density': ws.rho,
                                                              'remove_predictor': remove_predictor,
                                                              'use_sparse': use_sparse,
                                                              'remove_inputs': [],
                                                              'gust_assembler': 'LeadingEdge',
                                                              },
                                        }
                                        }

        ws.config['LinDynamicSim'] = {'n_tsteps': lin_tsteps,
                                      'dt': ws.dt,
                                      'input_generators': [
                                          {'name': 'control_surface_deflection',
                                           'index': 0,
                                           'file_path': self.route_test_dir + '/cases/elevator.txt'},
                                          {'name': 'u_gust',
                                           'index': 0,
                                           'file_path': self.route_test_dir + '/cases/gust.txt'}
                                      ],
                                      'postprocessors': ['AerogridPlot'],
                                      'postprocessors_settings':
                                          {'AerogridPlot': {'include_rbm': 'on',
                                                            'include_applied_forces': 'on',
                                                            'minus_m_star': 0}, }
                                      }

        ws.config['PickleData'] = {}
        ws.config.write()

        if not restart:
            data = sharpy.sharpy_main.main(['', ws.route + ws.case_name + '.sharpy'])
        else:
            with open(self.route_test_dir + '/output/{:s}.pkl'.format(ws.case_name), 'rb') as f:
                data = pickle.load(f)

        return data

    def run_linear_sharpy(self, **kwargs):
        flow = ['BeamLoader', 'AerogridLoader',
                'StaticCoupled',
                'Modal',
                'LinearAssembler',
                'LinDynamicSim',
                'PickleData']

        restart = False
        if restart:
            flow = ['LinDynamicSim']

        data = self.run_sharpy(flow, restart=restart, **kwargs)

        return data

    def extract_u_inf(self, data):
        ts = len(data.aero.timestep_info)
        _, m_chord, n_span = data.aero.timestep_info[0].u_ext[0].shape
        u_inf_ext = np.zeros((ts, m_chord, n_span))
        for i_ts in range(ts):
            u_inf_ext[i_ts, :, :] = data.aero.timestep_info[i_ts].u_ext[0][2, :, :]

        return u_inf_ext

    def test_linear_gust(self):
        test_conditions = [
            {'remove_predictor': True},
            {'remove_predictor': False},
        ]
        for test in test_conditions:
            with self.subTest(test):
                data = self.run_linear_sharpy(**test)
                u_gust_in = np.loadtxt(self.route_test_dir + '/cases/gust.txt')
                u_inf_ext = self.extract_u_inf(data)
                if test['remove_predictor']:
                    predictor_offset = 0
                else:
                    # input defined at time step n+1
                    predictor_offset = 1

                # test leading edge value is equal to input
                np.testing.assert_array_almost_equal(u_inf_ext[1 + predictor_offset:, 0, 0],
                                                     u_gust_in[:-1-predictor_offset])

                # check convection in panels downstream
                for i_chord in range(1, u_inf_ext.shape[1]):
                    np.testing.assert_array_almost_equal(u_inf_ext[predictor_offset + 1 + i_chord:-i_chord, i_chord, 0],
                                                         u_inf_ext[predictor_offset + 1 + i_chord - 1:-(i_chord + 1), i_chord - 1, 0])

    def tearDown(self):
        import shutil
        folders = ['cases', 'figures', 'output']
        for folder in folders:
            shutil.rmtree(self.route_test_dir + '/' + folder)


class TestGusts(unittest.TestCase):

    def test_campbell(self):
        """
        Test that the Campell approximation to the Von Karman filter is equivalent in continuous and
        discrete time

        """
        sigma_w = 1
        length_scale = 1
        velocity = 1
        dt = 1e-1
        omega_w = np.logspace(-3, 0, 10)

        ss_ct = campbell(sigma_w, length_scale, velocity)

        ss_dt = campbell(sigma_w, length_scale, velocity, dt=dt)

        G_ct = ss_ct.freqresp(omega_w)
        G_dt = ss_dt.freqresp(omega_w)

        np.testing.assert_array_almost_equal(G_ct[0, 0, :].real, G_dt[0, 0, :].real, decimal=3)


if __name__ == '__main__':
    unittest.main()


