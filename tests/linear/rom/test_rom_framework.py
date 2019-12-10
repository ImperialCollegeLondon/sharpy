import numpy as np
import os
import unittest
import cases.templates.flying_wings as wings
import sharpy.sharpy_main


class TestROMFramework(unittest.TestCase):
    """
    Test verifyies the execution of the balancing ROMs (i.e. checks that everything runs rather than checking
    against a benchmark case
    """
    
    def setUp(self):
        # Problem Set up
        u_inf = 1.
        alpha_deg = 0.
        rho = 1.02
        num_modes = 4

        # Lattice Discretisation
        M = 4
        N = 8
        M_star_fact = 1

        # Linear UVLM settings
        integration_order = 2
        remove_predictor = False
        use_sparse = False

        # Case Admin - Create results folders
        case_name = 'goland_cs'
        case_nlin_info = 'M%dN%dMs%d_nmodes%d' % (M, N, M_star_fact, num_modes)
        case_name += case_nlin_info

        self.route_test_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
        fig_folder = self.route_test_dir + '/figures/'
        os.makedirs(fig_folder, exist_ok=True)

        # SHARPy nonlinear reference solution
        ws = wings.Goland(M=M,
                          N=N,
                          Mstar_fact=M_star_fact,
                          u_inf=u_inf,
                          alpha=alpha_deg,
                          rho=rho,
                          sweep=0,
                          physical_time=2,
                          n_surfaces=2,
                          route=self.route_test_dir + '/cases',
                          case_name=case_name)

        ws.gust_intensity = 0.01
        ws.sigma = 1
        ws.horseshoe = True

        ws.clean_test_files()
        ws.update_derived_params()
        ws.set_default_config_dict()

        ws.generate_aero_file()
        ws.generate_fem_file()

        ws.config['SHARPy'] = {
            'flow':
                ['BeamLoader', 'AerogridLoader',
                 'StaticCoupled',
                 'AerogridPlot',
                 'BeamPlot',
                 'Modal',
                 'LinearAssembler',
                 'FrequencyResponse',
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
            'mstar': 1,
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
                                                              },
                                            'rigid_body_motion': False}}

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

        self.ws = ws

    def test_roms(self):
        settings = dict()
        settings['Direct'] = {'algorithm': 'Direct',
                              'print_info': 'off',
                              'algorithm_settings': {'tune': 'off',
                                                     'use_schur': 'on',
                                                     'reduction_method': 'realisation'}}

        settings['FrequencyLimited'] = {'algorithm': 'FrequencyLimited',
                                        'print_info': 'off',
                                        'algorithm_settings': {'frequency': 0.8,
                                                               'method_low': 'trapz',
                                                               'options_low': {'points': 3},
                                                               'method_high': 'gauss',
                                                               'options_high': {'partitions': 2, 'order': 1},
                                                               'check_stability': True}}

        settings['Iterative'] = {'algorithm': 'Iterative',
                                 'print_info': 'off',
                                 'algorithm_settings': {'lowrank': 'on'}}

        for rom in settings:
            with self.subTest(rom):
                self.ws.config['LinearAssembler']['linear_system_settings']['aero_settings']['rom_method'] = [
                    'Balanced']
                rom_settings = {'Balanced': settings[rom]}
                self.ws.config['LinearAssembler']['linear_system_settings']['aero_settings']['rom_method_settings'] = \
                    rom_settings
                self.ws.config.write()
                sharpy.sharpy_main.main(['', self.ws.route + self.ws.case_name + '.sharpy'])

    def tearDown(self):
        import shutil
        folders = ['cases', 'figures', 'output']
        for folder in folders:
            try:
                shutil.rmtree(self.route_test_dir + '/' + folder)
            except FileNotFoundError:
                pass


if __name__ == '__main__':
    unittest.main()
