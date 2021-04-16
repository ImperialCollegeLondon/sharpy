"""
Generates a Goland source case to be used as part of the basis interpolation test.
"""
import numpy as np
import os
import cases.templates.flying_wings as wings
import sharpy.sharpy_main

def generate_goland(u_inf, problem_type, rom_method_settings, **kwargs):
    """
    rom_method_settings is the {Krylov: krylov_settings} dictionary
    """

    # problem type:: source, interpolation or actual

    # Problem Set up
    alpha_deg = 0
    rho = 1.02
    num_modes = 4

    # Lattice Discretisation
    M = 4
    N = 8
    M_star_fact = 10

    # Linear UVLM settings
    integration_order = 2
    remove_predictor = False
    use_sparse = True

    # Case Admin - Create results folders
    case_name = 'goland_cs'
    case_nlin_info = 'M%dN%dMs%d_nmodes%d' % (M, N, M_star_fact, num_modes)

    if 'Krylov' in rom_method_settings.keys():
        rom_settings = rom_method_settings['Krylov']
        case_rom_info = 'rom_MIMORA_r%d_sig%04d_%04dj' % (rom_settings['r'], rom_settings['frequency'].real * 100,
                                                          rom_settings['frequency'].imag * 100)
    else:
        case_rom_info = ''

    case_name += 'u_%g' % u_inf
    case_name += case_nlin_info + case_rom_info

    abspath = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/' + problem_type + '/'
    if not os.path.isdir(abspath):
        os.makedirs(abspath)

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
                                    route=abspath + '/cases',
                                    case_name=case_name)


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
             'AsymptoticStability',
             'SaveParametricCase',
             ],
        'case': ws.case_name, 'route': ws.route,
        'write_screen': kwargs.get('write_screen', 'off'),
        'write_log': 'on',
        'log_folder': abspath + '/output/',
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
        'wake_shape_generator_input': {'u_inf': u_inf,
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
        'rollup_tolerance': 1e-4,
        'vortex_radius': 1e-6}

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
            'vortex_radius': 1e-6,
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

    ws.config['AerogridPlot'] = {'folder': abspath + '/output/',
                                 'include_rbm': 'off',
                                 'include_applied_forces': 'on',
                                 'minus_m_star': 0}

    ws.config['AeroForcesCalculator'] = {'folder': abspath + '/output/forces',
                                         'write_text_file': 'on',
                                         'text_file_name': ws.case_name + '_aeroforces.csv',
                                         'screen_output': 'on',
                                         'unsteady': 'off'}

    ws.config['BeamPlot'] = {'folder': abspath + '/output/',
                             'include_rbm': 'off',
                             'include_applied_forces': 'on'}

    ws.config['BeamCsvOutput'] = {'folder': abspath + '/output/',
                                  'output_pos': 'on',
                                  'output_psi': 'on',
                                  'screen_output': 'on'}

    ws.config['Modal'] = {'NumLambda': 20,
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
                                    'inout_coordinates': 'nodes',
                                    'retain_inputs': [2 * 6 * (ws.num_node_tot - 1)],
                                    'retain_outputs': [6 * (ws.num_node_tot - 1) + (ws.num_node_tot - 1) // 2 * 6],
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
                                                          # 'ScalingDict': {'length': 0.5 * ws.c_ref,
                                                          #                 'speed': u_inf,
                                                          #                 'density': rho},
                                                          'integr_order': integration_order,
                                                          'density': ws.rho,
                                                          'remove_predictor': remove_predictor,
                                                          'use_sparse': use_sparse,
                                                          'rigid_body_motion': 'off',
                                                          'vortex_radius': 1e-6,
                                                          'use_euler': 'off',
                                                          'remove_inputs': ['u_gust'],
                                                          'rom_method': list(rom_method_settings.keys()),
                                                          'rom_method_settings': rom_method_settings,
                                                          },
                                        'rigid_body_motion': False}}

    ws.config['AsymptoticStability'] = {'print_info': True,
                                        'export_eigenvalues': 'on',
                                       }

    ws.config['LinDynamicSim'] = {'dt': ws.dt,
                                  'n_tsteps': ws.n_tstep,
                                  'sys_id': 'LinearAeroelastic',
                                  'postprocessors': ['BeamPlot', 'AerogridPlot'],
                                  'postprocessors_settings': {'AerogridPlot': {
                                      'u_inf': ws.u_inf,
                                      'include_rbm': 'on',
                                      'include_applied_forces': 'on',
                                      'minus_m_star': 0},
                                      'BeamPlot': {'folder': ws.route + '/output/',
                                                   'include_rbm': 'on',
                                                   'include_applied_forces': 'on'}}}

    ws.config['FrequencyResponse'] = {'quick_plot': 'off',
                                      'frequency_unit': 'w',
                                      'frequency_bounds': [10, 200],
                                      'num_freqs': 200,
                                      }

    ws.config['SaveParametricCase'] = {'parameters': {
                                           'u_inf': u_inf,
                                       }
                                       }

    ws.config.write()

    data = sharpy.sharpy_main.main(['', ws.route + ws.case_name + '.sharpy'])

    return ws.case_name
