import numpy as np
import os
import unittest
import cases.templates.flying_wings as wings
import sharpy.sharpy_main


class TestGolandFlutter(unittest.TestCase):

    def setup(self):
        # Problem Set up
        u_inf = 1.
        alpha_deg = 0.
        rho = 1.02
        num_modes = 4

        # Lattice Discretisation
        M = 16
        N = 32
        M_star_fact = 10

        # Linear UVLM settings
        integration_order = 2
        remove_predictor = False
        use_sparse = True

        # ROM Properties
        rom_settings = dict()
        rom_settings['algorithm'] = 'mimo_rational_arnoldi'
        rom_settings['r'] = 6
        frequency_continuous_k = np.array([0.])

        # Case Admin - Create results folders
        case_name = 'goland_cs'
        case_nlin_info = 'M%dN%dMs%d_nmodes%d' % (M, N, M_star_fact, num_modes)
        case_rom_info = 'rom_MIMORA_r%d_sig%04d_%04dj' % (rom_settings['r'], frequency_continuous_k[-1].real * 100,
                                                          frequency_continuous_k[-1].imag * 100)

        case_name += case_nlin_info + case_rom_info

        self.route_test_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
        fig_folder = self.route_test_dir + '/figures/'
        os.makedirs(fig_folder, exist_ok=True)

        # SHARPy nonlinear reference solution
        ws = wings.GolandControlSurface(M=M,
                                        N=N,
                                        Mstar_fact=M_star_fact,
                                        u_inf=u_inf,
                                        alpha=alpha_deg,
                                        cs_deflection=[0, 0],
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

        frequency_continuous_w = 2 * u_inf * frequency_continuous_k / ws.c_ref
        rom_settings['frequency'] = frequency_continuous_w
        rom_settings['tangent_input_file'] = ws.route + '/' + ws.case_name + '.rom.h5'

        ws.config['SHARPy'] = {
            'flow':
                ['BeamLoader', 'AerogridLoader',
                 'StaticCoupled',
                 'AerogridPlot',
                 'BeamPlot',
                 'Modal',
                 'LinearAssembler',
                 'FrequencyResponse',
                 'AsymptoticStability',
                 ],
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
            'freestream_dir': ws.u_inf_direction
        }

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
                                           'gravity': 9.754}}

        ws.config['AerogridPlot'] = {'folder': self.route_test_dir + '/output/',
                                     'include_rbm': 'off',
                                     'include_applied_forces': 'on',
                                     'minus_m_star': 0}

        ws.config['AeroForcesCalculator'] = {'folder': self.route_test_dir + '/output/forces',
                                             'write_text_file': 'on',
                                             'text_file_name': ws.case_name + '_aeroforces.csv',
                                             'screen_output': 'on',
                                             'unsteady': 'off'}

        ws.config['BeamPlot'] = {'folder': self.route_test_dir + '/output/',
                                 'include_rbm': 'off',
                                 'include_applied_forces': 'on'}

        ws.config['BeamCsvOutput'] = {'folder': self.route_test_dir + '/output/',
                                      'output_pos': 'on',
                                      'output_psi': 'on',
                                      'screen_output': 'on'}

        ws.config['Modal'] = {'folder': self.route_test_dir + '/output/',
                              'NumLambda': 20,
                              'rigid_body_modes': 'off',
                              'print_matrices': 'on',
                              'keep_linear_matrices': 'on',
                              'write_dat': 'off',
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
                                                              'print_info': 'on',
                                                              'gravity': 'on',
                                                              'remove_sym_modes': 'on',
                                                              'remove_dofs': []},
                                            'aero_settings': {'dt': ws.dt,
                                                              'ScalingDict': {'length': 0.5 * ws.c_ref,
                                                                              'speed': u_inf,
                                                                              'density': rho},
                                                              'integr_order': integration_order,
                                                              'density': ws.rho,
                                                              'remove_predictor': remove_predictor,
                                                              'use_sparse': use_sparse,
                                                              'rigid_body_motion': 'off',
                                                              'use_euler': 'off',
                                                              'remove_inputs': ['u_gust'],
                                                              'rom_method': ['Krylov'],
                                                              'rom_method_settings': {'Krylov': rom_settings}},
                                            'rigid_body_motion': False}}

        ws.config['AsymptoticStability'] = {'print_info': True,
                                            'folder': self.route_test_dir + '/output/',
                                            'velocity_analysis': [160, 180, 20]}

        ws.config['LinDynamicSim'] = {'dt': ws.dt,
                                      'n_tsteps': ws.n_tstep,
                                      'sys_id': 'LinearAeroelastic',
                                      'postprocessors': ['BeamPlot', 'AerogridPlot'],
                                      'postprocessors_settings': {'AerogridPlot': {
                                          'u_inf': ws.u_inf,
                                          'folder': self.route_test_dir + '/output/',
                                          'include_rbm': 'on',
                                          'include_applied_forces': 'on',
                                          'minus_m_star': 0},
                                          'BeamPlot': {'folder': ws.route + '/output/',
                                                       'include_rbm': 'on',
                                                       'include_applied_forces': 'on'}}}

        ws.config['FrequencyResponse'] = {'compute_fom': 'on',
                                          'quick_plot': 'off',
                                          'folder': self.route_test_dir + '/output/',
                                          'frequency_unit': 'k',
                                          'frequency_bounds': [0.0001, 1.0],
                                          }

        ws.config.write()

        self.data = sharpy.sharpy_main.main(['', ws.route + ws.case_name + '.sharpy'])

    def run_rom_stable(self):
        ssrom = self.data.linear.linear_system.uvlm.rom['Krylov'].ssrom
        eigs_rom = np.linalg.eigvals(ssrom.A)

        assert all(np.abs(eigs_rom) <= 1.), 'UVLM Krylov ROM is unstable - flutter speed may not be correct. Change' \
                                            'ROM settings to achieve stability'
        print('ROM is stable')

    def run_flutter(self):
        flutter_ref_speed = 166 # at current discretisation

        u_inf = self.data.linear.stability['velocity_results']['u_inf']
        eval_real = self.data.linear.stability['velocity_results']['evals_real']
        eval_imag = self.data.linear.stability['velocity_results']['evals_imag']

        # Flutter onset
        ind_zero_real = np.where(eval_real >= 0)[0][0]
        assert ind_zero_real > 0, 'Flutter speed not below 165.00 m/s'
        flutter_speed = 0.5 * (u_inf[ind_zero_real] + u_inf[ind_zero_real - 1])
        flutter_frequency = np.sqrt(eval_real[ind_zero_real] ** 2 + eval_imag[ind_zero_real] ** 2)

        print('Flutter speed = %.1f m/s' % flutter_speed)
        print('Flutter frequency = %.2f rad/s' % flutter_frequency)
        assert np.abs(
            flutter_speed - flutter_ref_speed) / flutter_ref_speed < 1e-2, ' Flutter speed error greater than ' \
                                                                           '1 percent'
        print('Test Complete')

    def test_flutter(self):
        self.setup()
        self.run_rom_stable()
        self.run_flutter()

    def tearDown(self):
        import shutil
        folders = ['cases', 'figures', 'output']
        for folder in folders:
            shutil.rmtree(self.route_test_dir + '/' + folder)


if __name__ == '__main__':
    unittest.main()
