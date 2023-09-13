"""
    Script to set the required SHARPy settings for  all cases within
    the UDP Control example notebook. 
"""
def get_settings_udp(model, flow, **kwargs):
    gust = kwargs.get('gust', False)
    
    horseshoe = kwargs.get('horseshoe', False)
    gravity = kwargs.get('gravity', True)
    wake_length_factor = kwargs.get('wake_length_factor', 10)

    num_cores = kwargs.get('num_cores', 2)
    num_modes = kwargs.get('num_modes', 20)
    free_flight = kwargs.get('free_flight', False)
    tolerance = kwargs.get('tolerance', 1e-6)
    n_load_steps = kwargs.get('n_load_steps', 5)
    fsi_tolerance = kwargs.get('fsi_tolerance', 1e-6)
    relaxation_factor = kwargs.get('relaxation_factor', 0.1)
    newmark_damp = kwargs.get('newmark_damp', 0.5e-4)
    output_folder = kwargs.get('output_folder', './output/')
    model.config['SHARPy'] = {'case': model.case_name,
                        'route': model.route,
                        'flow': flow,
                        'write_screen': kwargs.get('write_screen', 'on'),
                        'write_log': 'on',
                        'save_settings': 'on',
                        'log_folder':  output_folder + '/',
                        'log_file':  model.case_name + '.log'}


    model.config['BeamLoader'] = {'unsteady': 'off',
                                'orientation': model.quat}

    model.config['AerogridLoader'] = {'unsteady': 'on',
                                'aligned_grid': 'on',
                                'mstar': wake_length_factor*model.M, #int(20/tstep_factor),
                                'wake_shape_generator': 'StraightWake',
                                'wake_shape_generator_input': {
                                    'u_inf':model.u_inf,
                                    'u_inf_direction': [1., 0., 0.],
                                    'dt':model.dt,
                                },
                            }
    if horseshoe:
        model.config['AerogridLoader']['mstar'] = 1

    model.config['StaticCoupled'] = {
        'print_info': 'on',
        'max_iter': 200,
        'n_load_steps': n_load_steps,
        'tolerance': 1e-5,
        'relaxation_factor': relaxation_factor,
        'aero_solver': 'StaticUvlm',
        'aero_solver_settings': {
            'rho': model.rho,
            'print_info': 'off',
            'horseshoe': 'on',
            'num_cores': num_cores,
            'n_rollup': 0,
            'velocity_field_generator': 'SteadyVelocityField',
            'velocity_field_input': {
                'u_inf': model.u_inf,
                'u_inf_direction': model.u_inf_direction},
            'vortex_radius': 1e-9},
        'structural_solver': 'NonLinearStatic',
        'structural_solver_settings': {'print_info': 'off',
                                   'max_iterations': 200,
                                   'num_load_steps': 5,
                                   'delta_curved': 1e-6,
                                   'min_delta': 1e-8,
                                   'gravity': gravity,
                                   'gravity': 9.81},
    }


    model.config['AerogridPlot'] = {'include_rbm': 'off',
                                 'include_applied_forces': 'on',
                                 'minus_m_star': 0}
    model.config['BeamPlot'] = {'include_rbm': 'off',
                             'include_applied_forces': 'on'}
    
    model.config['WriteVariablesTime'] = {'structure_variables': ['pos'],
                                        'structure_nodes': list(range(0, model.num_node_surf)),
                                        'cleanup_old_solution': 'on',
                                        }
    
    
    
    model.config['DynamicCoupled'] = {'print_info': 'on',
                                  'structural_substeps': 10,
                                  'dynamic_relaxation': 'on',
                                  'cleanup_previous_solution': 'on',
                                  'structural_solver': 'NonLinearDynamicPrescribedStep',
                                  'structural_solver_settings':  {'print_info': 'off',
                                                                'max_iterations': 950,
                                                                'delta_curved': 1e-1,
                                                                'min_delta': tolerance,
                                                                'newmark_damp': newmark_damp,
                                                                'gravity': gravity,
                                                                'gravity': 9.81,
                                                                'num_steps': model.n_tstep,
                                                                'dt':model.dt,
                                                                },
                                  'aero_solver': 'StepUvlm',
                                  'aero_solver_settings':  {'print_info': 'on',
                                                            'num_cores': num_cores,
                                                            'convection_scheme': 2,
                                                            'velocity_field_generator': 'SteadyVelocityField',
                                                            'velocity_field_input': {'u_inf': model.u_inf,
                                                                                    'u_inf_direction': [1., 0., 0.]},
                                                            'rho': model.rho,
                                                            'n_time_steps': model.n_tstep,
                                                            'vortex_radius': 1e-9,
                                                            'dt': model.dt,
                                                            'gamma_dot_filtering': 3},
                                  'fsi_substeps': 200,
                                  'fsi_tolerance': fsi_tolerance,
                                  'relaxation_factor': model.relaxation_factor,
                                  'minimum_steps': 1,
                                  'relaxation_steps': 150,
                                  'final_relaxation_factor': 0.0,
                                  'n_time_steps': model.n_tstep,
                                  'dt': model.dt,
                                  'include_unsteady_force_contribution': kwargs.get('unsteady_force_distribution', True),
                                  'postprocessors': ['WriteVariablesTime', 'BeamPlot', 'AerogridPlot'],
                                  'postprocessors_settings': {'BeamPlot': {'include_rbm': 'on',
                                                                           'include_applied_forces': 'on'},
                                                              'StallCheck': {},
                                                              'AerogridPlot': {
                                                                  'u_inf': model.u_inf,
                                                                  'include_rbm': 'on',
                                                                  'include_applied_forces': 'on',
                                                                  'minus_m_star': 0},
                                                              'WriteVariablesTime': {
                                                                  'structure_variables': ['pos', 'psi'],
                                                                  'structure_nodes': [model.num_node_surf - 1,
                                                                                      model.num_node_surf,
                                                                                      model.num_node_surf + 1],
                                                                    },
                                                                    },
                                    'network_settings': kwargs.get('network_settings', {}),
                                    }
    
    if gust:
        gust_settings = kwargs.get('gust_settings', {'gust_shape': '1-cos',
                                                     'gust_length': 10.,
                                                     'gust_intensity': 0.01,
                                                     'gust_offset': 0.})
        model.config['DynamicCoupled']['aero_solver_settings']['velocity_field_generator'] = 'GustVelocityField'
        model.config['DynamicCoupled']['aero_solver_settings']['velocity_field_input'] =  {'u_inf': model.u_inf,
                                                                                            'u_inf_direction': [1., 0, 0],
                                                                                            'relative_motion': bool(not free_flight),
                                                                                            'offset': gust_settings['gust_offset'],
                                                                                            'gust_shape': gust_settings['gust_shape'],
                                                                                            'gust_parameters': {
                                                                                                                'gust_length': gust_settings['gust_length'],
                                                                                                                'gust_intensity': gust_settings['gust_intensity'] * model.u_inf,
                                                                                                            }                                                                
                                                                                        }

    model.config['PickleData'] = {}
    
    model.config['LinearAssembler'] = {'linear_system': 'LinearAeroelastic',
                                    'inout_coordinates': 'nodes', 
                                    # 'recover_accelerations': True,           
                                'linear_system_settings': {
                                    'beam_settings': {'modal_projection': True,
                                                        'inout_coords': 'modes',
                                                        'discrete_time': True,
                                                        'newmark_damp': newmark_damp,
                                                        'discr_method': 'newmark',
                                                        'dt': model.dt,
                                                        'proj_modes': 'undamped',
                                                        'num_modes': num_modes,
                                                        'print_info': 'on',
                                                        'gravity': gravity,
                                                        'remove_dofs': []},
                                    'aero_settings': {'dt': model.dt,
                                                        'integr_order': 2,
                                                        'density': model.rho,
                                                        'remove_predictor': True,
                                                        'use_sparse': 'off',
                                                        'gust_assembler':  'LeadingEdge',
                                                        'ScalingDict': kwargs.get('scaling_dict', {'length':1, 'speed': 1, 'density': 1}),
                                                        },
                                    'track_body': free_flight,
                                    'use_euler': free_flight,
                                    }}
    model.config['Modal'] = {'print_info': True,
                            'use_undamped_modes': True,
                            'NumLambda': num_modes,
                            'rigid_body_modes': free_flight, 
                            'write_modes_vtk': False,
                            'print_matrices': False,
                            'continuous_eigenvalues': 'off',
                            'dt': model.dt,
                            'plot_eigenvalues': False,
                            }
    model.config['SaveData']['save_linear'] = True
    if kwargs.get('remove_gust_input_in_statespace', False):
        model.config['LinearAssembler']['linear_system_settings']['aero_settings']['remove_inputs'] =  ['u_gust']

    rom_settings = kwargs.get('rom_settings', {'use': False})
    if rom_settings['use']:
        model.config['LinearAssembler']['linear_system_settings']['aero_settings']['rom_method'] = rom_settings['rom_method'],
        model.config['LinearAssembler']['linear_system_settings']['aero_settings']['rom_method_settings'] = rom_settings['rom_method_settings']
    return model.config
