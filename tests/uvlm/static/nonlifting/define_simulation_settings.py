import numpy as np
import sharpy.utils.algebra as algebra

def define_simulation_settings(flow, model, alpha_deg, u_inf, rho = 1.225, lifting_only=True, nonlifting_only=False, horseshoe=False, **kwargs):
    gravity = kwargs.get('gravity',True)
    nonlifting_body_interactions = not lifting_only and not nonlifting_only
    wake_length = kwargs.get('wake_length', 10)
    # Other parameters
    if horseshoe:
        dt = 1
        mstar = 1 
    else:
        dt = model.aero.chord_main / model.aero.m / u_inf
        mstar = wake_length*model.aero.m
    # numerics
    n_step = kwargs.get('n_step', 5)
    structural_relaxation_factor = kwargs.get('structural_relaxation_factor', 0.6)
    tolerance = kwargs.get('tolerance', 1e-6)
    fsi_tolerance = kwargs.get('fsi_tolerance', 1e-4)
    num_cores = kwargs.get('num_cores',2)

    if not lifting_only:
        nonlifting_body_interactions = True
    else:
        nonlifting_body_interactions = False
        flow.remove("NonliftingbodygridLoader")
    settings = {}
    settings['SHARPy'] = {'case': model.case_name,
                        'route': model.case_route,
                        'flow': flow,
                        'write_screen': 'on',
                        'write_log': 'on',
                        'log_folder': model.output_route,
                        'log_file': model.case_name + '.log'}


    settings['BeamLoader'] = {'unsteady': 'on',
                                'orientation': algebra.euler2quat(np.array([0.,
                                                                            np.deg2rad(alpha_deg),
                                                                            0.]))}


    settings['LiftDistribution'] = {'rho': rho,
                                    'coefficients': True}
    
    settings['NonLinearStatic'] = {'print_info': 'off',
                                'max_iterations': 150,
                                'num_load_steps': 1,
                                'delta_curved': 1e-1,
                                'min_delta': tolerance,
                                'gravity_on': gravity,
                                'gravity': 9.81}

    settings['StaticUvlm'] = {'print_info': 'on',
                            'horseshoe': horseshoe,
                            'num_cores': num_cores,
                            'velocity_field_generator': 'SteadyVelocityField',
                            'velocity_field_input': {'u_inf': u_inf,
                                                    'u_inf_direction': [1., 0, 0]},
                            'rho': rho,
                            'nonlifting_body_interactions': nonlifting_body_interactions,
                            'only_nonlifting': nonlifting_only,
                            }

    settings['StaticCoupled'] = {'print_info': 'off',
                                'structural_solver': 'NonLinearStatic',
                                'structural_solver_settings': settings['NonLinearStatic'],
                                'aero_solver': 'StaticUvlm',
                                'aero_solver_settings': settings['StaticUvlm'],
                                'max_iter': 100,
                                'n_load_steps': n_step,
                                'tolerance': fsi_tolerance,
                                'relaxation_factor': structural_relaxation_factor,
                                'nonlifting_body_interactions': nonlifting_body_interactions}

    settings['AerogridLoader'] = {'unsteady': 'on',
                                'aligned_grid': 'on',
                                'mstar': mstar, #int(20/tstep_factor),
                                'wake_shape_generator': 'StraightWake',
                                'wake_shape_generator_input': {
                                    'u_inf': u_inf,
                                    'u_inf_direction': [1., 0., 0.],
                                    'dt': dt,
                                },
                            }

    settings['WriteVariablesTime'] = {
                'cleanup_old_solution': True,
                'nonlifting_nodes_variables': ['pressure_coefficients'],                                                                
                'nonlifting_nodes_isurf': np.zeros((model.n_node - 1,)),
                'nonlifting_nodes_im':  np.zeros((model.n_node - 1,)),
                'nonlifting_nodes_in': list(range(model.n_node - 1)),
    }

    settings['NonliftingbodygridLoader'] = {}

    settings['AeroForcesCalculator'] = {'coefficients': False}


    return settings