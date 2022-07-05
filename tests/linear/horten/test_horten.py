import sharpy.utils.algebra as algebra
from sharpy.cases.hangar.richards_wing import Baseline
import sharpy.sharpy_main
import numpy as np
import configobj
import unittest
import os
import shutil

route_test_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))


def run_rom_convergence(case_name, case_route='./cases/', output_folder='./output/', **kwargs):
    M = kwargs.get('M', 4)
    N = kwargs.get('N', 11)
    Msf = kwargs.get('Msf', 10)

    trim = kwargs.get('trim', True)

    rho_fact = 1.
    track_body = True
    payload = 0
    u_inf = kwargs.get('u_inf', 30)

    use_euler = True
    if use_euler:
        orient = 'euler'
    else:
        orient = 'quat'

    case_name += 'M%gN%gMsf%g_u%g' % (M, N, Msf, u_inf)

    # M4N11Msf5
    alpha_deg = 3.974
    cs_deflection = 0.3582
    thrust = 4.8062

    # ROM settings
    rom_settings = dict()
    rom_settings['algorithm'] = 'mimo_rational_arnoldi'
    rom_settings['r'] = 10
    rom_settings['frequency'] = np.array([0], dtype=float)
    rom_settings['single_side'] = 'observability'

    case_name += 'rom_%g_%s' % (rom_settings['r'], rom_settings['single_side'][:3])

    ws = Baseline(M=M,
                  N=N,
                  Mstarfactor=Msf,
                  u_inf=u_inf,
                  rho=1.02,
                  alpha_deg=alpha_deg,  # 7.7563783342984385,
                  roll_deg=0,
                  cs_deflection_deg=cs_deflection,  # -6.733360628875144,
                  thrust=thrust,  # 10.140622253017584,
                  physical_time=20,
                  case_name=case_name,
                  case_route=case_route,
                  case_name_format=2)

    ws.set_properties()
    ws.initialise()
    ws.clean_test_files()

    # ws.update_mass_stiffness(sigma=1., sigma_mass=1.5)
    ws.update_mass_stiffness(sigma=.5, sigma_mass=1.0, payload=payload)
    ws.update_fem_prop()
    ws.generate_fem_file()
    ws.update_aero_properties()
    ws.generate_aero_file()

    flow = ['BeamLoader',
            'AerogridLoader',
            'StaticTrim',
            'BeamPlot',
            'AerogridPlot',
            'AeroForcesCalculator',
            'DynamicCoupled',
            'Modal',
            'LinearAssembler',
            'AsymptoticStability',
            'SaveParametricCase'
            ]

    if not trim:
        flow[2] = 'StaticCoupled'

    settings = dict()
    settings['SHARPy'] = {'case': ws.case_name,
                          'route': ws.case_route,
                          'flow': flow,
                          'write_screen': 'off',
                          'write_log': 'on',
                          'log_folder': output_folder,
                          'log_file': ws.case_name + '.log'}

    settings['BeamLoader'] = {'unsteady': 'off',
                              'orientation': algebra.euler2quat(np.array([ws.roll,
                                                                          ws.alpha,
                                                                          ws.beta]))}

    settings['AerogridLoader'] = {'unsteady': 'off',
                                  'aligned_grid': 'on',
                                  'mstar': int(ws.M * ws.Mstarfactor),
                                  'freestream_dir': ['1', '0', '0'],
                                  'control_surface_deflection': [''],
                                  'wake_shape_generator': 'StraightWake',
                                  'wake_shape_generator_input': {'u_inf': u_inf,
                                                                 'u_inf_direction': [1., 0., 0.],
                                                                 'dt': ws.dt},
                                  }

    if ws.horseshoe is True:
        settings['AerogridLoader']['mstar'] = 1

    settings['StaticCoupled'] = {'print_info': 'on',
                                 'structural_solver': 'NonLinearStatic',
                                 'structural_solver_settings': {'print_info': 'off',
                                                                'max_iterations': 200,
                                                                'num_load_steps': 1,
                                                                'delta_curved': 1e-5,
                                                                'min_delta': ws.tolerance,
                                                                'gravity_on': 'on',
                                                                'gravity': 9.81},
                                 'aero_solver': 'StaticUvlm',
                                 'aero_solver_settings': {'print_info': 'on',
                                                          'horseshoe': ws.horseshoe,
                                                          'num_cores': 4,
                                                          'n_rollup': int(0),
                                                          'rollup_dt': ws.dt,
                                                          'rollup_aic_refresh': 1,
                                                          'rollup_tolerance': 1e-4,
                                                          'vortex_radius': 1e-6,
                                                          'velocity_field_generator': 'SteadyVelocityField',
                                                          'velocity_field_input': {'u_inf': ws.u_inf,
                                                                                   'u_inf_direction': [1., 0, 0]},
                                                          'rho': ws.rho},
                                 'max_iter': 200,
                                 'n_load_steps': 1,
                                 'tolerance': ws.tolerance,
                                 'relaxation_factor': 0.2}

    settings['StaticTrim'] = {'solver': 'StaticCoupled',
                              'solver_settings': settings['StaticCoupled'],
                              'thrust_nodes': ws.thrust_nodes,
                              'initial_alpha': ws.alpha,
                              'initial_deflection': ws.cs_deflection,
                              'initial_thrust': ws.thrust,
                              'max_iter': 200,
                              'fz_tolerance': 1e-2,
                              'fx_tolerance': 1e-2,
                              'm_tolerance': 1e-2,
                              'save_info': 'on'}

    settings['AerogridPlot'] = {'include_rbm': 'off',
                                'include_applied_forces': 'on',
                                'minus_m_star': 0,
                                'u_inf': ws.u_inf
                                }
    settings['AeroForcesCalculator'] = {'write_text_file': 'off',
                                        'text_file_name': ws.case_name + '_aeroforces.csv',
                                        'screen_output': 'on',
                                        'coefficients': True,
                                        'q_ref': 0.5 * ws.rho * ws.u_inf ** 2,
                                        'S_ref': 12.809,
                                        }

    settings['BeamPlot'] = {'include_rbm': 'on',
                            'include_applied_forces': 'on',
                            'include_FoR': 'on'}

    struct_solver_settings = {'print_info': 'off',
                              'initial_velocity_direction': [-1., 0., 0.],
                              'max_iterations': 950,
                              'delta_curved': 1e-6,
                              'min_delta': ws.tolerance,
                              'newmark_damp': 5e-3,
                              'gravity_on': True,
                              'gravity': 9.81,
                              'num_steps': ws.n_tstep,
                              'dt': ws.dt,
                              'initial_velocity': ws.u_inf * 1}

    step_uvlm_settings = {'print_info': 'on',
                          'num_cores': 4,
                          'convection_scheme': ws.wake_type,
                          'vortex_radius': 1e-6,
                          'velocity_field_generator': 'SteadyVelocityField',
                          'velocity_field_input': {'u_inf': ws.u_inf * 0,
                                                   'u_inf_direction': [1., 0., 0.]},
                          'rho': ws.rho,
                          'n_time_steps': ws.n_tstep,
                          'dt': ws.dt,
                          'gamma_dot_filtering': 3}

    settings['DynamicCoupled'] = {'print_info': 'on',
                                  'structural_solver': 'NonLinearDynamicCoupledStep',
                                  'structural_solver_settings': struct_solver_settings,
                                  'aero_solver': 'StepUvlm',
                                  'aero_solver_settings': step_uvlm_settings,
                                  'fsi_substeps': 200,
                                  'fsi_tolerance': ws.fsi_tolerance,
                                  'relaxation_factor': ws.relaxation_factor,
                                  'minimum_steps': 1,
                                  'relaxation_steps': 150,
                                  'final_relaxation_factor': 0.5,
                                  'n_time_steps': 1,  # ws.n_tstep,
                                  'dt': ws.dt,
                                  'include_unsteady_force_contribution': 'off',
                                  'postprocessors': ['BeamLoads', 'BeamPlot', 'AerogridPlot', 'WriteVariablesTime'],
                                  'postprocessors_settings': {'BeamLoads': {'csv_output': 'off'},
                                                              'BeamPlot': {'include_rbm': 'on',
                                                                           'include_applied_forces': 'on'},
                                                              'AerogridPlot': {
                                                                  'u_inf': ws.u_inf,
                                                                  'include_rbm': 'on',
                                                                  'include_applied_forces': 'on',
                                                                  'minus_m_star': 0},
                                                              'WriteVariablesTime': {
                                                                  'cleanup_old_solution': 'on',
                                                                  'delimiter': ',',
                                                                  'FoR_variables': ['total_forces',
                                                                                    'total_gravity_forces',
                                                                                    'for_pos', 'quat'],
                                                              }}}

    settings['Modal'] = {'print_info': True,
                         'use_undamped_modes': True,
                         'NumLambda': 30,
                         'rigid_body_modes': True,
                         'write_modes_vtk': 'on',
                         'print_matrices': 'on',
                         'save_data': 'on',
                         'continuous_eigenvalues': 'off',
                         'dt': ws.dt,
                         'plot_eigenvalues': False,
                         'rigid_modes_cg': False}

    settings['LinearAssembler'] = {'linear_system': 'LinearAeroelastic',
                                   'linearisation_tstep': -1,
                                   'linear_system_settings': {
                                       'beam_settings': {'modal_projection': 'on',
                                                         'inout_coords': 'modes',
                                                         'discrete_time': True,
                                                         'newmark_damp': 0.5e-2,
                                                         'discr_method': 'newmark',
                                                         'dt': ws.dt,
                                                         'proj_modes': 'undamped',
                                                         'use_euler': use_euler,
                                                         'num_modes': 20,
                                                         'print_info': 'on',
                                                         'gravity': 'on',
                                                         'remove_dofs': []},
                                       'aero_settings': {'dt': ws.dt,
                                                         'integr_order': 2,
                                                         'density': ws.rho * rho_fact,
                                                         'remove_predictor': False,
                                                         'use_sparse': False,
                                                         'vortex_radius': 1e-6,
                                                         'remove_inputs': ['u_gust'],
                                                         'rom_method': ['Krylov'],
                                                         'rom_method_settings': {'Krylov': rom_settings}},
                                       'track_body': track_body,
                                       'use_euler': use_euler,
                                   }}

    settings['AsymptoticStability'] = {
        'print_info': 'on',
        'modes_to_plot': [],
        # 'velocity_analysis': [27, 29, 3],
        'display_root_locus': 'off',
        'frequency_cutoff': 0,
        'export_eigenvalues': 'on',
        'target_system': ['aeroelastic', 'aerodynamic', 'structural'],
        'output_file_format': 'dat',
        'num_evals': 100}

    settings['FrequencyResponse'] = {'print_info': 'on',
                                     'frequency_bounds': [0.1, 100]}

    settings['LinDynamicSim'] = {'dt': ws.dt,
                                 'n_tsteps': ws.n_tstep,
                                 'sys_id': 'LinearAeroelastic',
                                 # 'reference_velocity': ws.u_inf,
                                 'write_dat': ['x', 'y', 't', 'u'],
                                 # 'write_dat': 'on',
                                 'postprocessors': ['BeamPlot', 'AerogridPlot'],
                                 'postprocessors_settings': {'AerogridPlot': {
                                     'u_inf': ws.u_inf,
                                     'include_rbm': 'on',
                                     'include_applied_forces': 'on',
                                     'minus_m_star': 0},
                                     'BeamPlot': {'include_rbm': 'on',
                                                  'include_applied_forces': 'on'}}}

    settings['StabilityDerivatives'] = {'u_inf': ws.u_inf,
                                        'S_ref': 12.809,
                                        'b_ref': ws.span,
                                        'c_ref': 0.719}

    settings['SaveData'] = {'save_aero': 'off',
                            'save_struct': 'off',
                            'save_linear': 'on',
                            'save_linear_uvlm': 'on'}

    settings['PickleData'] = {}

    settings['SaveParametricCase'] = {'parameters': {'r': rom_settings['r']}}

    config = configobj.ConfigObj()
    file_name = ws.case_route + '/' + ws.case_name + '.solver.txt'
    config.filename = file_name
    for k, v in settings.items():
        config[k] = v
    config.write()

    return sharpy.sharpy_main.main(['', ws.case_route + '/' + ws.case_name + '.solver.txt'])


class TestHortenWing(unittest.TestCase):


    def test_horten(self):

        M = 4
        N = 11
        Msf = 5

        trim = False

        case_name = 'horten_'

        case_route = route_test_dir + '/cases/'
        output_route = route_test_dir + '/output/'

        data = run_rom_convergence(case_name=case_name, case_route=case_route,
                                   output_folder=output_route,
                                   M=M, N=N, Msf=Msf, trim=trim)

        # check first 10 eigs are zero (9 integro states + yaw)
        path_to_eigs = data.output_folder + '/stability/aeroelastic_eigenvalues.dat'
        eigs = np.loadtxt(path_to_eigs)

        # check that aerodynamic and structural eigs are also written
        target_system = ['aerodynamic', 'structural']
        for sys in target_system:
            np.loadtxt(data.output_folder + f'/stability/{sys}_eigenvalues.dat')

        np.testing.assert_allclose(np.abs(eigs[:10]), 0, atol=1e-8)

        # check that the phugoid mode is there (more or less, very high tolerance)
        phugoid_eigs = eigs[11]
        wn_phugoid = np.linalg.norm(phugoid_eigs)
        period_phugoid = 2 * np.pi / wn_phugoid
        np.testing.assert_allclose(period_phugoid, 20.7, atol=0.1, rtol=0.1)

    def tearDown(self):
        folders = ['output/', 'cases/']

        for folder in folders:
            if os.path.isdir(route_test_dir + '/' + folder):
                shutil.rmtree(route_test_dir + '/' + folder)


if __name__ == '__main__':
    unittest.main()
