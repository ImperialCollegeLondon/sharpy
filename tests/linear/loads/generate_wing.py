import numpy as np
import sharpy.sharpy_main
import sharpy.utils.algebra as algebra
import cases.templates.flying_wings as wings


def generate_wing(case_name, case_route, output_route, **kwargs):

    m = kwargs.get('M', 4)
    n = kwargs.get('N', 8)
    msf = kwargs.get('Ms')

    alpha_deg = kwargs.get('alpha_deg', 0)
    tolerance = 1e-5
    gravity = True
    alpha_rad = alpha_deg * np.pi / 180
    u_inf = kwargs.get('u_inf', 10)
    rho = 1

    wing = wings.PazyControlSurface(M=m,
                                    N=n,
                                    Mstar_fact=msf,
                                    u_inf=u_inf,
                                    alpha=alpha_deg,
                                    rho=rho,
                                    route=case_route,
                                    case_name=case_name)

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

    settings['AerogridLoader'] = {'unsteady': 'on',
                                  'aligned_grid': 'on',
                                  'mstar': int(kwargs.get('wake_length', 100) * m),
                                  'wake_shape_generator': 'StraightWake',
                                  'wake_shape_generator_input': {
                                      'u_inf': u_inf,
                                      'u_inf_direction': [1., 0., 0.],
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
                                                       'u_inf_direction': [1., 0, 0]},
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

    wing.settings_to_config(settings)

    data = sharpy.sharpy_main.main(['', f'{case_route}/{case_name}.sharpy'])

    import pdb; pdb.set_trace()
    print('finished')

    return wing
