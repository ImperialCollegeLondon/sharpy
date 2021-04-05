"""
Linearised Horten Wing Example

Model information can be found in the relevant class:
    - :class:`cases.hangar.richards_wing.Baseline`

Simple flying wing linearised using the LinearAssembler solver. Then the eigenvalues are calculated, a short simulation
is run and finally the stability derivatives are computed.

"""
import sharpy.utils.algebra as algebra
from cases.hangar.richards_wing import Baseline
import numpy as np
import configobj

u_inf = 28
M = 4  # chordwise panels
N = 11  # spanwise panels
Msf = 5  # wake length in chord numbers
rho_fact = 1.  # air density factor

use_euler = True  # use euler angles or quaternions as orientation parametrisation
if use_euler:
    orient = 'euler'
else:
    orient = 'quat'

case_notes = '_rho%s' % (str(rho_fact))
case_rmks = 'M%gN%gMsf%g' % (M, N, Msf)

# M4N11Msf5 - trim values
alpha_deg = 4.5135
cs_deflection = 0.1814
thrust = 5.5129

# M8N11Msf5 - trim values
# alpha_deg = 4.5162
# cs_deflection = 0.2373
# thrust = 5.5129
ws = Baseline(M=M,
              N=N,
              Mstarfactor=Msf,
              u_inf=u_inf,
              rho=1.02,
              alpha_deg=alpha_deg,
              roll_deg=0,
              cs_deflection_deg=cs_deflection,
              thrust=thrust,
              physical_time=20,
              case_name='horten_%s' % orient,
              case_name_format=4,
              case_remarks=case_rmks + case_notes)

ws.set_properties()
ws.initialise()
ws.clean_test_files()

# ws.update_mass_stiffness(sigma=1., sigma_mass=1.5)
ws.update_mass_stiffness(sigma=1., sigma_mass=2.5)
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
        'LinDynamicSim',
        'StabilityDerivatives',
        ]

settings = dict()
settings['SHARPy'] = {'case': ws.case_name,
                      'route': ws.case_route,
                      'flow': flow,
                      'write_screen': 'on',
                      'write_log': 'on',
                      'log_folder': './output/' + ws.case_name + '/',
                      'log_file': ws.case_name + '.log'}

settings['BeamLoader'] = {'unsteady': 'off',
                          'orientation': algebra.euler2quat(np.array([ws.roll,
                                                                      ws.alpha,
                                                                      ws.beta]))}

if ws.horseshoe is True:
    settings['AerogridLoader'] = {'unsteady': 'off',
                                  'aligned_grid': 'on',
                                  'mstar': 1,
                                  'freestream_dir': ['1', '0', '0'],
                                  'control_surface_deflection': ['']}
else:
    settings['AerogridLoader'] = {'unsteady': 'off',
                                  'aligned_grid': 'on',
                                  'mstar': int(ws.M * ws.Mstarfactor),
                                  'freestream_dir': ['1', '0', '0'],
                                  'control_surface_deflection': ['']}


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
                          'm_tolerance': 1e-2}

settings['AerogridPlot'] = {'include_rbm': 'off',
                            'include_applied_forces': 'on',
                            'minus_m_star': 0,
                            'u_inf': ws.u_inf
                            }
settings['AeroForcesCalculator'] = {'write_text_file': 'off',
                                    'text_file_name': ws.case_name + '_aeroforces.csv',
                                    'screen_output': 'on',
                                    'unsteady': 'off',
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
                      'horseshoe': ws.horseshoe,
                      'num_cores': 4,
                      'n_rollup': 1,
                      'convection_scheme': ws.wake_type,
                      'rollup_dt': ws.dt,
                      'rollup_aic_refresh': 1,
                      'rollup_tolerance': 1e-4,
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
                              'n_time_steps': 1,
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
                     'write_data': 'on',
                     'continuous_eigenvalues': 'off',
                     'dt': ws.dt,
                     'plot_eigenvalues': False,
                     'rigid_modes_cg': False}

settings['LinearAssembler'] = {'linear_system': 'LinearAeroelastic',
                               'linear_system_settings': {
                                   'beam_settings': {'modal_projection': 'off',
                                                     'inout_coords': 'modes',
                                                     'discrete_time': True,
                                                     'newmark_damp': 0.5e-2,
                                                     'discr_method': 'newmark',
                                                     'dt': ws.dt,
                                                     'proj_modes': 'undamped',
                                                     'use_euler': use_euler,
                                                     'num_modes': 9,
                                                     'print_info': 'on',
                                                     'gravity': 'on',
                                                     'remove_dofs': []},
                                   'aero_settings': {'dt': ws.dt,
                                                     'integr_order': 2,
                                                     'density': ws.rho * rho_fact,
                                                     'remove_predictor': 'off',
                                                     'use_sparse': 'off',
                                                     'rigid_body_motion': 'on',
                                                     'use_euler': use_euler,
                                                     'remove_inputs': ['u_gust']},
                                   'rigid_body_motion': True,
                                   'track_body': 'on',
                                   'use_euler': use_euler,
                                   'linearisation_tstep': -1
                                }}

settings['AsymptoticStability'] = {
                                    'print_info': 'on',
                                    'modes_to_plot': [],
                                    'display_root_locus': 'off',
                                    'frequency_cutoff': 0,
                                    'export_eigenvalues': 'on',
                                    'num_evals': 10000,}


settings['LinDynamicSim'] = {'dt': ws.dt,
                             'n_tsteps': ws.n_tstep,
                             'sys_id': 'LinearAeroelastic',
                             'write_dat': ['x', 'y', 't', 'u'],
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

config = configobj.ConfigObj()
np.set_printoptions(precision=16)
file_name = ws.case_route + '/' + ws.case_name + '.sharpy'
config.filename = file_name
for k, v in settings.items():
    config[k] = v
config.write()

delta = np.zeros((ws.n_tstep, 1))
delta_dot = np.zeros_like(delta)
d_elev = 1*np.pi/180 * 0.01
t_init = 1.0
t_ramp = 2.0
t_final = 5.0
delta[int(t_init//ws.dt):(int(t_ramp//ws.dt)), 0] = np.linspace(0, d_elev, (int(t_ramp//ws.dt)) - int(t_init//ws.dt))
delta[int(t_ramp//ws.dt):(int(t_final//ws.dt)), 0] = d_elev
delta[int(t_final//ws.dt):(int((t_final+1.0)//ws.dt)), 0] = np.linspace(d_elev, 0, (int((t_final+1)//ws.dt)) - int(t_final//ws.dt))
delta_dot[int(t_init//ws.dt):int(t_ramp//ws.dt), 0] = d_elev / (t_ramp - t_init) * 1
delta_dot[int(t_final//ws.dt):(int((t_final+1.0)//ws.dt)), 0] = - d_elev / 1. * 1
ws.create_linear_simulation(delta, delta_dot)

# data = sharpy.sharpy_main.main(['', ws.case_route + '/' + ws.case_name + '.sharpy'])
