import sharpy.cases.templates.flying_wings as wings
import sharpy.sharpy_main


def generate_pazy(u_inf, case_name, output_folder='/output/', cases_folder='', **kwargs):
    # u_inf = 60
    alpha_deg = kwargs.get('alpha', 0.)
    rho = 1.225
    num_modes = 16
    gravity_on = kwargs.get('gravity_on', True)
    symmetry_condition = kwargs.get('symmetry_condition', False)
    gust_vanes = kwargs.get('gust_vanes', False)
    dynamic = kwargs.get('dynamic', False)
    n_tsteps = kwargs.get('n_tsteps', 1)
    cs_deflection_file=kwargs.get('cs_deflection_file', None)
    # Lattice Discretisation
    M = kwargs.get('M', 4)
    N = kwargs.get('N', 32)
    M_star_fact = kwargs.get('Ms', 10)
    flag_multiple_mstar_input = kwargs.get('test_multiple_inputs', False)

    # SHARPy nonlinear reference solution
    ws = wings.PazyControlSurface(M=M,
                                    N=N,
                                    Mstar_fact=M_star_fact,
                                    u_inf=u_inf,
                                    alpha=alpha_deg,
                                    cs_deflection=[0, 0],
                                    rho=rho,
                                    sweep=0,
                                    physical_time=2,
                                    n_surfaces=2,
                                    route=cases_folder + '/' + case_name,
                                    case_name=case_name)

    ws.gust_intensity = 0.01
    ws.sigma = 1

    ws.clean_test_files()
    ws.update_derived_params()
    ws.set_default_config_dict()
    if symmetry_condition:
        ws.reduce_model_to_symmetric_wing()
    ws.generate_aero_file()
    ws.generate_fem_file()
    set_final_settings(ws, 
                        dynamic=dynamic,
                        surface_m=M,
                        output_folder=output_folder,
                        symmetry_condition = symmetry_condition,
                        gust_vanes = gust_vanes,
                        gravity_on = gravity_on,
                        n_tsteps=n_tsteps,
                        cs_deflection_file=cs_deflection_file)

    sharpy.sharpy_main.main(['', ws.route + ws.case_name + '.sharpy'])

def set_final_settings(ws, dynamic = False, surface_m=8, output_folder='/output/', symmetry_condition = False, gravity_on = True, n_tsteps=1,flag_multiple_mstar_input=False, gust_vanes = False, cs_deflection_file=None):
    ws.config['SHARPy'] = {
        'flow':
            ['BeamLoader',
            'AerogridLoader',
            'AerogridPlot',
            'StaticCoupled',
            'BeamPlot',
            'WriteVariablesTime',
            ],
        'case': ws.case_name, 'route': ws.route,
        'write_screen': 'off', 'write_log': 'on',
        'save_settings': 'on',
        'log_folder': output_folder,
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
    if flag_multiple_mstar_input:
        ws.config['AerogridLoader']['mstar'] = [ws.config['AerogridLoader']['mstar'], ws.config['AerogridLoader']['mstar']]
        ws.config['SHARPy']['flow'] = ['BeamLoader','AerogridLoader']
    else:
        ws.config['StaticUvlm'] = {
            'rho': ws.rho,
            'velocity_field_generator': 'SteadyVelocityField',
            'velocity_field_input': {
                'u_inf': ws.u_inf,
                'u_inf_direction': ws.u_inf_direction},
        'rollup_dt': ws.dt,
        'print_info': 'on',
        'horseshoe': 'on',
        'symmetry_condition': symmetry_condition,
        'symmetry_plane': 1,
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
            'vortex_radius': 1e-9,
            'symmetry_condition': symmetry_condition,
            'symmetry_plane': 1,},
        'structural_solver': 'NonLinearStatic',
        'structural_solver_settings': settings['NonLinearStatic']}

        ws.config['AerogridPlot'] = {'include_rbm': 'off',
                                    'include_applied_forces': 'on',
                                    'minus_m_star': 0}

        ws.config['BeamPlot'] = {'include_rbm': 'off',
                                'include_applied_forces': 'on'}

        ws.config['WriteVariablesTime'] = {'structure_variables': ['pos'],
                                    'structure_nodes': list(range(0, ws.num_node_surf)),
                                    'cleanup_old_solution': 'on'}

    if dynamic:
        ws.config['SHARPy']['flow'].insert(-2, 'DynamicCoupled')
        ws.config['SHARPy']['flow'].insert(-2, 'BeamLoads')
  
        settings['StepUvlm'] = {'num_cores': 2,
                        'convection_scheme': 3,
                        'gamma_dot_filtering': 7,
                        'cfl1': True,
                        # 'velocity_field_generator': 'SteadyVelocityField',
                        # 'velocity_field_input': {'u_inf':u_inf,
                        #                         'u_inf_direction': [1., 0, 0]},
                        'velocity_field_generator': 'GustVelocityField',
                        'velocity_field_input':{'u_inf': ws.u_inf,
                                                'u_inf_direction': [1., 0, 0],
                                                'relative_motion': True,
                                                'offset': 0,
                                                'gust_shape': 'continuous_sin',
                                                'gust_parameters': {'gust_length':  5. * ws.u_inf,
                                                                    'gust_intensity': 0.05 * ws.u_inf,
                                                                }                                                                
                                            },
                        'rho': ws.rho,
                        'n_time_steps': 20,
                        'dt': ws.dt,
                        'symmetry_condition': symmetry_condition,
                        'symmetry_plane': 1,
                        }

        settings['NonLinearDynamicPrescribedStep'] = {'print_info': 'on',
                    'max_iterations': 950,
                    'delta_curved': 1e-1,
                    'min_delta': 1e-6,
                    'newmark_damp': 1e-4,
                    'gravity_on': gravity_on,
                    'gravity': 9.81,
                    'num_steps': 10,
                    'dt': ws.dt,
                    # 'initial_velocity': u_inf,
                    }

        ws.config['DynamicCoupled'] = {'structural_solver':'NonLinearDynamicPrescribedStep',
            'structural_solver_settings': settings['NonLinearDynamicPrescribedStep'],
            'aero_solver': 'StepUvlm',
            'aero_solver_settings': settings['StepUvlm'],
            'fsi_substeps': 200,
            'fsi_tolerance': 1e-4,
            'relaxation_factor': 0.2,
            'minimum_steps': 1,
            'relaxation_steps': 150,
            'final_relaxation_factor': 0.05,
            'n_time_steps': n_tsteps,
            'dt': ws.dt,
            'include_unsteady_force_contribution': True, 
            'postprocessors': ['BeamLoads'],
            'postprocessors_settings': {
                                        'BeamLoads': {'csv_output': 'on'},
                                        },
            }
        ws.config['BeamLoads'] = {'csv_output': True}
    if gust_vanes:
        import numpy as np
        ws.config = apply_gust_vane_settings(ws.config,
                                             cs_deflection_file,
                                             ws.dt,
                                             ws.u_inf,
                                             surface_m,
                                             False,
                                             symmetry_condition)
        ws.config['DynamicCoupled']['postprocessors'].append('WriteVariablesTime')
        ws.config['DynamicCoupled']['postprocessors_settings']['WriteVariablesTime'] = {
            'vel_field_variables': ['uind'],
            'vel_field_points': np.array([-1.25, 0.0, 0.25 ]),
                                        }
    ws.config.write()

    sharpy.sharpy_main.main(['', ws.route + ws.case_name + '.sharpy'])

def apply_gust_vane_settings(settings,
                             cs_deflection_file,
                             dt,
                             u_inf,
                             surface_m,
                             vertical,
                             symmetry_condition):
    """
    Updates the SHARPy configuration with gust vane definitions.

    Args:
        settings: SHARPy configuration object.
        cs_deflection_file: Path to the gust vane deflection file.
        dt: Time step size.
        u_inf: Freestream velocity.
        surface_m: Chordwise discretisation.
        vertical: Orientation of the gust vanes (vertical or horizontal).
        symmetry_condition: Whether the problem is symmetric.
        only_gust_vanes: Whether gust vanes are the only gust source.

    Returns:
        cs_deflection_file. Modified the input settings dictionary.
    """
    # breakpoint()
    wake_length_vanes = 5
    gust_vane_parameters = {
        'M': surface_m * 3,
        'N': 40,
        'M_star': int(wake_length_vanes / (dt * u_inf)),
        'span': 5,
        'chord': 0.3,
        'control_surface_deflection_generator_settings': {
            'dt': dt,
            'deflection_file': cs_deflection_file
        }
    }

    settings['AerogridLoader']['gust_vanes'] = True
    settings['AerogridLoader']['gust_vanes_generator_settings'] = {
        'n_vanes': 2,
        'streamwise_position': [-1.5, -1.5],
        'vertical_position': [-0.25, 0.25],
        'symmetry_condition': symmetry_condition,
        'vane_parameters': [gust_vane_parameters, gust_vane_parameters],
        'vertical': vertical
    }

    # Override velocity field for gust vanes
    stepuvlm_updates = {
            'convection_scheme': 3,
            'velocity_field_generator': 'SteadyVelocityField',
            'velocity_field_input': {
                'u_inf': u_inf,
                'u_inf_direction': [1., 0., 0.],
            }
        }
    settings['DynamicCoupled']['aero_solver_settings'].update(stepuvlm_updates)

    return settings