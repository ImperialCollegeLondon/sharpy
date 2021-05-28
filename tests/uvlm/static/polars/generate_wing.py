import numpy as np
import cases.templates.flying_wings as wings
import sharpy.utils.algebra as algebra
import sharpy.sharpy_main


def generate_infinite_wing(case_name, alpha, **kwargs):

    m = 8
    n = 4
    mstar = 500
    u_inf = 1
    alpha_deg = alpha
    alpha_rad = alpha_deg * np.pi / 180
    rho = 1.225
    gravity = False

    tolerance = 1e-5

    case_route = kwargs.get('case_route', './')
    case_route += '/' + case_name

    output_route = kwargs.get('output_route', './output/')
    # if not os.path.isdir(case_route):
    #     os.makedirs(case_route)

    polar_file = kwargs.get('polar_file', None)

    wing = wings.QuasiInfinite(M=m,
                               N=n,
                               Mstar_fact=mstar,
                               u_inf=u_inf,
                               alpha=alpha_rad,
                               aspect_ratio=kwargs.get('aspect_ratio', 8),
                               rho=rho,
                               route=case_route,
                               case_name=case_name,
                               polar_file=polar_file)

    wing.main_ea = kwargs.get('main_ea', 0.25)
    wing.clean_test_files()
    wing.update_derived_params()
    wing.sigma = 1e25
    wing.update_mass_stiff()

    wing.generate_aero_file()
    wing.generate_fem_file()

    settings = dict()
    settings['SHARPy'] = {'case': case_name,
                          'route': case_route,
                          'flow': kwargs.get('flow', []),
                          'write_screen': 'off',
                          'write_log': 'on',
                          'log_folder': output_route,
                          'log_file': case_name + '.log'}

    settings['BeamLoader'] = {'unsteady': 'on',
                              'orientation': algebra.euler2quat(np.array([0.,
                                                                          alpha_rad,
                                                                          0.]))}

    u_inf_direction = algebra.rotation3d_y(alpha_rad * 0).T.dot(np.array([1, 0, 0]))
    settings['AerogridLoader'] = {'unsteady': 'on',
                                  'aligned_grid': 'on',
                                  'mstar': int(kwargs.get('wake_length', 100) * m),
                                  'wake_shape_generator': 'StraightWake',
                                  'wake_shape_generator_input': {
                                      'u_inf': u_inf,
                                      'u_inf_direction': u_inf_direction,
                                      'dt': wing.dt,
                                  },
                                  }

    settings['NonLinearStatic'] = {'print_info': 'off',
                                   'max_iterations': 150,
                                   'num_load_steps': 1,
                                   'delta_curved': 1e-1,
                                   'min_delta': tolerance,
                                   'gravity_on': gravity,
                                   'gravity': 9.81,
                                   'initial_position': [0., 0., 0.]}

    settings['StaticUvlm'] = {'print_info': 'on',
                              'horseshoe': 'on',
                              'num_cores': 4,
                              'vortex_radius': 1e-6,
                              'velocity_field_generator': 'SteadyVelocityField',
                              'velocity_field_input': {'u_inf': u_inf,
                                                       'u_inf_direction': u_inf_direction},
                              'rho': rho}

    settings['StaticCoupled'] = {'print_info': 'off',
                                 'structural_solver': 'NonLinearStatic',
                                 'structural_solver_settings': settings['NonLinearStatic'],
                                 'aero_solver': 'StaticUvlm',
                                 'aero_solver_settings': settings['StaticUvlm'],
                                 'max_iter': 100,
                                 'n_load_steps': kwargs.get('n_load_steps', 1),
                                 'tolerance': kwargs.get('fsi_tolerance', 1e-5),
                                 'relaxation_factor': kwargs.get('relaxation_factor', 0.2)}
    if polar_file is not None:
        settings['StaticCoupled']['correct_forces_method'] = 'PolarCorrection'
        settings['StaticCoupled']['correct_forces_settings'] = {'cd_from_cl': 'off',
                                                                'correct_lift': 'on',
                                                                'moment_from_polar': 'on'}

    settings['AerogridPlot'] = {'include_incidence_angle': 'on',
                                'include_velocities': 'on'}

    settings['BeamPlot'] = {}

    settings['AeroForcesCalculator'] = {'write_text_file': 'on',
                                        'coefficients': 'on',
                                        'q_ref': 0.5 * rho * u_inf ** 2,
                                        'S_ref': wing.main_chord * wing.wing_span}

    settings['Modal'] = {'print_info': True,
                         'use_undamped_modes': True,
                         'NumLambda': 50,
                         'rigid_body_modes': True,
                         'write_modes_vtk': 'on',
                         'print_matrices': 'on',
                         'write_data': 'on',
                         'continuous_eigenvalues': 'off',
                         'plot_eigenvalues': False,
                         'rigid_modes_ppal_axes': 'on',
                         'folder': output_route}

    # ROM settings
    rom_settings = dict()
    rom_settings['algorithm'] = 'mimo_rational_arnoldi'
    rom_settings['r'] = 4
    rom_settings['frequency'] = np.array([0], dtype=float)
    rom_settings['single_side'] = 'observability'

    settings['LinearAssembler'] = {'linear_system': 'LinearAeroelastic',
                                   'linear_system_settings': {
                                       'beam_settings': {'modal_projection': 'off',
                                                         'inout_coords': 'modes',
                                                         'discrete_time': 'off',
                                                         'newmark_damp': 0.5e-3,
                                                         'discr_method': 'newmark',
                                                         'dt': wing.dt,
                                                         'proj_modes': 'undamped',
                                                         'use_euler': 'on',
                                                         'num_modes': 20,
                                                         'print_info': 'on',
                                                         'gravity': kwargs.get('gravity', 'on'),
                                                         'remove_dofs': [],
                                                         'remove_rigid_states': 'on'},
                                       'aero_settings': {'dt': wing.dt,
                                                         'integr_order': 2,
                                                         'density': rho,
                                                         'remove_predictor': 'off',
                                                         'use_sparse': False,
                                                         'rigid_body_motion': True,
                                                         'vortex_radius': 1e-7,
                                                         'remove_inputs': ['u_gust'],
                                                         'convert_to_ct': 'on',
                                                         },
                                       'track_body': 'off',
                                       'use_euler': 'on',
                                   }}


    settings['StabilityDerivatives'] = {'u_inf': u_inf,
                                        'c_ref': wing.main_chord,
                                        'b_ref': wing.wing_span,
                                        'S_ref': wing.wing_span * wing.main_chord,
                                        }

    settings['SaveParametricCase'] = {'save_case': 'off',
                                      'parameters': {'alpha': alpha_deg}}

    wing.settings_to_config(settings)

    sharpy.sharpy_main.main(['', f'{case_route}/{case_name}.sharpy'])

    return wing


def get_case_header(polar, infinite_wing, compute_uind, high_re, main_ea, use2pi):
    if polar:
        case_header = 'polar'
        if high_re:
            case_header += '_highre'
        if use2pi:
            case_header += '_2pi'
    else:
        case_header = 'uvlm'

    if not infinite_wing:
        case_header += '_ar8'
    else:
        case_header += '_ari'

    if compute_uind and polar:
        case_header += '_uind'

    case_header += '_ea{:02g}'.format(main_ea * 100)

    return case_header


def run(infinite_wing, compute_uind, main_ea, high_re, case_route_root, output_route_root, use2pi=False,
        polar_file=None):
    flow = ['BeamLoader',
            'AerogridLoader',
            'StaticCoupled',
            'AeroForcesCalculator',
            'SaveParametricCase']

    if polar_file is not None:
        polar = True
    else:
        polar = False

    if not infinite_wing:
        ar = 8
    else:
        ar = 1e7

    case_header = get_case_header(polar, infinite_wing, compute_uind, high_re, main_ea, use2pi)
    case_route = case_route_root + '/' + case_header + '/'
    output_route = output_route_root + '/' + case_header + '/'
    for alpha in np.linspace(-5, 5, 6):
        case_name = '{:s}_alpha{:04g}'.format(case_header,
                                              alpha * 100).replace('-', 'M')
        generate_infinite_wing(case_name, alpha,
                               flow=flow,
                               case_route=case_route,
                               polar_file=polar_file,
                               aspect_ratio=ar,
                               compute_uind=compute_uind,
                               main_ea=main_ea,
                               output_route=output_route,
                               actual_aoa=not use2pi,
                               )

    # pc.main(case_header, output_route)


def all_iter(case_route_root, output_route_root):
    for main_ea in [0.25, 0.5]:
        for polar in [True, False]:
            for infinite_wing in [True, False]:
                if polar:
                    for high_re in [True, False]:
                        for compute_uind in [True, False]:
                            run(polar, infinite_wing, compute_uind, main_ea, high_re, case_route_root, output_route_root)
                else:
                    pass
                    # run(polar, infinite_wing, False, main_ea, False, case_route_root, output_route_root)


if __name__ == '__main__':
    import postproc_infinite_wing as pc
    case_route_root = './cases_linear/'
    output_route_root = './output_linear/'
    polar = True
    infinite_wing = False
    compute_uind = False
    main_ea = 0.25
    high_re = False
    use2pi = True

    run(polar, infinite_wing, compute_uind, main_ea, high_re, case_route_root, output_route_root, use2pi=use2pi)
    # all_iter(case_route_root, output_route_root)
