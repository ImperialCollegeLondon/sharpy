import numpy as np
import os
import unittest
import cases.templates.flying_wings as wings
import sharpy.sharpy_main

# Problem Set up
def generate_pazy_udp(u_inf, case_name, output_folder='/output/', cases_folder='', **kwargs):
    # u_inf = 60
    alpha_deg = kwargs.get('alpha', 0.)
    rho = 1.225
    num_modes = 16
    gravity_on = kwargs.get('gravity_on', True)
    cd = kwargs.get('cd', './')  # current directory (useful for tests)

    # Lattice Discretisation
    M = kwargs.get('M', 4)
    N = kwargs.get('N', 32)
    M_star_fact = kwargs.get('Ms', 10)

    # Linear UVLM settings
    integration_order = 2
    remove_predictor = False
    use_sparse = True

    # ROM Properties
    rom_settings = dict()
    rom_settings['algorithm'] = 'mimo_rational_arnoldi'
    rom_settings['r'] = 5
    rom_settings['single_side'] = 'observability'
    frequency_continuous_k = np.array([0.])

    # Case Admin - Create results folders
    # case_name = 'goland_cs'
#     case_nlin_info = 'M%dN%dMs%d_nmodes%d' % (M, N, M_star_fact, num_modes)
    # case_rom_info = 'rom_MIMORA_r%d_sig%04d_%04dj' % (rom_settings['r'], frequency_continuous_k[-1].real * 100,
                                                      # frequency_continuous_k[-1].imag * 100)

    # case_name += case_nlin_info #+ case_rom_info

    # os.makedirs(fig_folder, exist_ok=True)

    # SHARPy nonlinear reference solution
    ws = wings.PazyControlSurface(M=M,
                                    N=N,
                                    Mstar_fact=M_star_fact,
                                    u_inf=u_inf,
                                    alpha=alpha_deg,
                                    cs_deflection=[0, 0],
                                    rho=rho,
                                    sweep=0,
                                    physical_time=0.2,
                                    n_surfaces=2,
                                    route=cases_folder + '/' + case_name,
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
            ['BeamLoader',
            'AerogridLoader',
            # 'NonLinearStatic',
             'StaticUvlm',
             'AerogridPlot',
             'BeamPlot',
             'WriteVariablesTime',
             'DynamicCoupled',
#              'Modal',
#              'LinearAssembler',
             # 'FrequencyResponse',
#              'AsymptoticStability',
#              'SaveParametricCase',
             ],
        'case': ws.case_name, 'route': ws.route,
        'write_screen': 'on', 'write_log': 'on',
        'save_settings': 'on',
        'log_folder': output_folder + '/' + ws.case_name + '/',
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
        'horseshoe': 'on',
        'num_cores': 4,
        'n_rollup': 0,
        'rollup_aic_refresh': 0,
        'rollup_tolerance': 1e-4}

    settings = dict()
    settings['NonLinearStatic'] = {'print_info': 'off',
                                   'max_iterations': 200,
                                   'num_load_steps': 5,
                                   'delta_curved': 1e-6,
                                   'min_delta': 1e-8,
                                   'gravity_on': gravity_on,
                                   'gravity': 9.81}

    ws.config['StaticCoupled'] = {
        'print_info': 'on',
        'max_iter': 200,
        'n_load_steps': 4,
        'tolerance': 1e-5,
        'relaxation_factor': 0.1,
        'aero_solver': 'StaticUvlm',
        'aero_solver_settings': {
            'rho': ws.rho,
            'print_info': 'off',
            'horseshoe': 'on',
            'num_cores': 4,
            'n_rollup': 0,
            'rollup_dt': ws.dt,
            'rollup_aic_refresh': 1,
            'rollup_tolerance': 1e-4,
            'velocity_field_generator': 'SteadyVelocityField',
            'velocity_field_input': {
                'u_inf': ws.u_inf,
                'u_inf_direction': ws.u_inf_direction},
            'vortex_radius': 1e-9},
        'structural_solver': 'NonLinearStatic',
        'structural_solver_settings': settings['NonLinearStatic']}

    ws.config['AerogridPlot'] = {'folder': output_folder,
                                 'include_rbm': 'off',
                                 'include_applied_forces': 'on',
                                 'minus_m_star': 0}

    ws.config['AeroForcesCalculator'] = {'folder': output_folder,
                                         'write_text_file': 'on',
                                         'text_file_name': ws.case_name + '_aeroforces.csv',
                                         'screen_output': 'on',
                                         'unsteady': 'off'}

    ws.config['BeamPlot'] = {'folder': output_folder,
                             'include_rbm': 'off',
                             'include_applied_forces': 'on'}

    ws.config['BeamCsvOutput'] = {'folder': output_folder,
                                  'output_pos': 'on',
                                  'output_psi': 'on',
                                  'screen_output': 'on'}

    ws.config['WriteVariablesTime'] = {'folder': output_folder,
                                        'structure_variables': ['pos'],
                                        'structure_nodes': list(range(0, ws.num_node_surf)),
                                        'cleanup_old_solution': 'on'}

    ws.config['Modal'] = {'folder': output_folder,
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
                                    # 'modal_tstep': 0,
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
                                                          'gravity': gravity_on,
                                                          'remove_sym_modes': 'on',
                                                          'remove_dofs': []},
                                        'aero_settings': {'dt': ws.dt,
                                                          # 'ScalingDict': {'length': 0.5 * ws.c_ref,
                                                                          # 'speed': u_inf,
                                                                          # 'density': rho},
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
                                        'folder': output_folder,
                                        'export_eigenvalues': 'on',
                                        'target_system': ['aeroelastic', 'aerodynamic', 'structural'],
                                        # 'velocity_analysis': [160, 180, 20]}
                                        }

    ws.config['LinDynamicSim'] = {'dt': ws.dt,
                                  'n_tsteps': ws.n_tstep,
                                  'sys_id': 'LinearAeroelastic',
                                  'postprocessors': ['BeamPlot', 'AerogridPlot'],
                                  'postprocessors_settings': {'AerogridPlot': {
                                      'u_inf': ws.u_inf,
                                      'folder': output_folder,
                                      'include_rbm': 'on',
                                      'include_applied_forces': 'on',
                                      'minus_m_star': 0},
                                      'BeamPlot': {'folder': ws.route + output_folder,
                                                   'include_rbm': 'on',
                                                   'include_applied_forces': 'on'}}}

    ws.config['FrequencyResponse'] = {'quick_plot': 'off',
                                      'folder': output_folder,
                                      'frequency_unit': 'w',
                                      'frequency_bounds': [0.0001, 1.0],
                                      'num_freqs': 100,
                                      'frequency_spacing': 'log',
                                      'target_system': ['aeroelastic', 'aerodynamic', 'structural'],
                                      }

    ws.config['SaveParametricCase'] = {'folder': output_folder + ws.case_name + '/',
                                       'save_case':'off',
                                       'parameters': {'u_inf': u_inf}}
    settings = dict()
    settings['NonLinearDynamicPrescribedStep'] = {'print_info': 'on',
                                                  'max_iterations': 950,
                                                  'delta_curved': 1e-2,
                                                  'min_delta': 1e-5,
                                                  'newmark_damp': 5e-2,
                                                  'gravity_on': 'on',
                                                  'gravity': 9.81,
                                                  'num_steps': ws.n_tstep,
                                                  'dt': ws.dt}

    settings['StepUvlm'] = {'print_info': 'on',
                            'horseshoe': 'off',
                            'num_cores': 4,
                            'n_rollup': 100,
                            'convection_scheme': 2,
                            'rollup_dt': ws.dt,
                            'rollup_aic_refresh': 1,
                            'rollup_tolerance': 1e-4,
                            'velocity_field_generator': 'SteadyVelocityField',
                            'velocity_field_input': {'u_inf': ws.u_inf*1,
                                                     'u_inf_direction': [1., 0., 0.]},
                            'rho': ws.rho,
                            'n_time_steps': ws.n_tstep,
                            'dt': ws.dt,
                            'gamma_dot_filtering': 3}


    settings['DynamicCoupled'] = {'print_info': 'on',
                                  'structural_substeps': 10,
                                  'dynamic_relaxation': 'on',
                                  'clean_up_previous_solution': 'on',
                                  'structural_solver': 'NonLinearDynamicPrescribedStep',
                                  'structural_solver_settings': settings['NonLinearDynamicPrescribedStep'],
                                  'aero_solver': 'StepUvlm',
                                  'aero_solver_settings': settings['StepUvlm'],
                                  'fsi_substeps': 200,
                                  'fsi_tolerance': 1e-6,
                                  'relaxation_factor': ws.relaxation_factor,
                                  'minimum_steps': 1,
                                  'relaxation_steps': 150,
                                  'final_relaxation_factor': 0.0,
                                  'n_time_steps': 4, #ws.n_tstep,
                                  'dt': ws.dt,
                                  'include_unsteady_force_contribution': 'off',
                                  'network_connections': False,
                                  'postprocessors': ['UDPout'],
                                  'postprocessors_settings': {'BeamLoads': {'folder': output_folder,
                                                                            'csv_output': 'off'},
                                                              'BeamPlot': {'folder': output_folder,
                                                                           'include_rbm': 'on',
                                                                           'include_applied_forces': 'on'},
                                                              'StallCheck': {},
                                                              'AerogridPlot': {
                                                                  'u_inf': ws.u_inf,
                                                                  'folder': output_folder,
                                                                  'include_rbm': 'on',
                                                                  'include_applied_forces': 'on',
                                                                  'minus_m_star': 0},
                                                              'WriteVariablesTime': {
                                                                  'folder': output_folder,
                                                                  'structure_variables': ['pos', 'psi'],
                                                                  'structure_nodes': [ws.num_node_surf - 1,
                                                                                      ws.num_node_surf,
                                                                                      ws.num_node_surf + 1]},
                                                              'UDPout': {'variables_filename': cd + '/variables.yaml',
                                                                         'output_network_settings':
                                                                             {'destination_address': ['127.0.0.1'],
                                                                                                     'destination_ports': [65431]},
                                                                         'console_log_level': 'info',
                                                                         },
                                                              }}

    ws.config['DynamicCoupled'] = settings['DynamicCoupled']


    ws.config.write()

    sharpy.sharpy_main.main(['', ws.route + ws.case_name + '.sharpy'])

if __name__== '__main__':
    # u_inf = 1
    # generate_pazi(u_inf)
#
    from datetime import datetime

    # datetime object containing current date and time
    # u_inf_vec = np.array([46])
    # u_inf_vec = np.linspace(28, 30, 50)
    # u_inf_vec = np.linspace(5, 30, 26)
    u_inf_vec = [50]
#     u_inf_vec = np.linspace(20, 60, 41)
#     u_inf_vec = np.linspace(1, 20, 20)
#     u_inf_vec = np.linspace(1, 60, 60)


    alpha = 1
    gravity_on = True

    M = 4
    N = 16
    Ms = 1

    # For comparison with Marc
#     M = 8
#     N = 128
#     Ms = 8


    batch_log = 'batch_log_alpha{:04g}'.format(alpha*100)

    with open('./{:s}.txt'.format(batch_log), 'w') as f:
    # dd/mm/YY H:M:S
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write('SHARPy launch - START\n')
        f.write("date and time = %s\n\n" % dt_string)

    for i, u_inf in enumerate(u_inf_vec):
        print('RUNNING SHARPY %f\n' % u_inf)
        case_name = 'pazi_uinf{:04g}_alpha{:04g}'.format(u_inf*10, alpha*100)
        try:
            generate_pazy_udp(u_inf, case_name, output_folder='/output/vortex_radius_M{:g}N{:g}Ms{:g}_alpha{:04g}/'.format(M, N, Ms, alpha * 100),
                              cases_subfolder='/M{:g}N{:g}Ms{:g}/'.format(M, N, Ms),
                              M=M, N=N, Ms=Ms, alpha=alpha,
                              gravity_on=gravity_on)
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            with open('./{:s}.txt'.format(batch_log), 'a') as f:
                f.write('%s Ran case %i :::: u_inf = %f\n\n' % (dt_string, i, u_inf))
        except AssertionError:
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            with open('./{:s}.txt'.format(batch_log), 'a') as f:
                f.write('%s ERROR RUNNING case %f\n\n' % (dt_string, u_inf))