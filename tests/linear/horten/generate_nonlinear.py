import numpy as np
import configobj
from cases.hangar.richards_wing import Baseline
import sharpy.utils.algebra as algebra

M = 4
N = 11
Msf = 5

case_rmks = '_M%gN%gMsf%g_20s_test' % (M, N, Msf)
# case_rmks = 'TEST_outvars'

# M4N11Msf5
alpha_deg = 4.5158
cs_deflection = 0.1726
thrust = 7.1662

# M8N11Msf5
# alpha_deg = 4.5162
# cs_deflection = 0.2373
# thrust = 5.5129

ws = Baseline(M=M,
              N=N,
              Mstarfactor=Msf,
              u_inf=28,
              rho=1.02,
              alpha_deg=alpha_deg,  # 7.7563783342984385,
              roll_deg=-5 * 0,
              cs_deflection_deg=cs_deflection,  # -6.733360628875144,
              thrust=thrust,  # 10.140622253017584,
              physical_time=20,
              case_name_format=4,
              case_remarks=case_rmks)

delta = np.ones(ws.n_tstep) * cs_deflection * np.pi / 180
d_elev = 1 * np.pi / 180 * 0.01
t_init = 0.5
t_ramp = 2.0
t_final = 5.0
delta[int(t_init // ws.dt):(int(t_ramp // ws.dt))] += np.linspace(0, d_elev,
                                                                  (int(t_ramp // ws.dt)) - int(t_init // ws.dt))
delta[int(t_ramp // ws.dt):(int(t_final // ws.dt))] += d_elev
delta[int(t_final // ws.dt):(int((t_final + 1.0) // ws.dt))] += np.linspace(d_elev, 0,
                                                                            (int((t_final + 1) // ws.dt)) - int(
                                                                                t_final // ws.dt))

ws.set_properties()
ws.initialise()
ws.clean_test_files()
ws.dynamic_control_surface(delta)
ws.update_mass_stiffness(sigma=.5, sigma_mass=1)
ws.update_fem_prop()
ws.generate_fem_file()
ws.update_aero_properties()
ws.generate_aero_file()

ws.horseshoe = False

settings = dict()
flow = ['BeamLoader',
        'AerogridLoader',
        'StaticCoupled',
        # 'StaticTrim',
        # 'Modal',
        'BeamPlot',
        'AerogridPlot',
        'DynamicCoupled',
        # 'LinearAssembler',
        # 'SaveData',
        # 'AsymptoticStability',
        ]

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

# if ws.horseshoe is True:
#     settings['AerogridLoader'] = {'unsteady': 'off',
#                                   'aligned_grid': 'on',
#                                   'mstar': 1,
#                                   'freestream_dir': ['1', '0', '0'],
#                                   'control_surface_deflection': ['']
#                                     }
# else:
settings['AerogridLoader'] = {'unsteady': 'off',
                              'aligned_grid': 'on',
                              'mstar': int(ws.M * ws.Mstarfactor),
                              'freestream_dir': ['1', '0', '0'],
                              'control_surface_deflection': ['DynamicControlSurface'],
                              'control_surface_deflection_generator_settings': {'0': {'dt': ws.dt,
                                   'deflection_file': ws.case_route + '/' + ws.case_name + '.input.txt'}},
                              }

settings['StaticCoupled'] = {'print_info': 'off',
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
                                                      'rollup_dt': ws.dt,  # self.c_root / self.M / self.u_inf,
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
                              # 'structural_substeps': 1,
                              # 'dynamic_relaxation': 'on',
                              # 'clean_up_previous_solution': 'on',
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
                              'n_time_steps': ws.n_tstep,
                              'dt': ws.dt,
                              'include_unsteady_force_contribution': 'off',
                              'postprocessors': ['BeamLoads', 'BeamPlot', 'AerogridPlot', 'WriteVariablesTime'],
                              'postprocessors_settings': {'BeamLoads': {'folder': './output/',
                                                                        'csv_output': 'off'},
                                                          'BeamPlot': {'folder': './output/',
                                                                       'include_rbm': 'on',
                                                                       'include_applied_forces': 'on'},
                                                          'AerogridPlot': {
                                                              'u_inf': ws.u_inf,
                                                              'folder': './output/',
                                                              'include_rbm': 'on',
                                                              'include_applied_forces': 'on',
                                                              'minus_m_star': 0},
                                                          'WriteVariablesTime': {
                                                              'folder': './output/',
                                                              'cleanup_old_solution': 'on',
                                                              'delimiter': ',',
                                                              'FoR_variables': ['total_forces',
                                                                                      'total_gravity_forces',
                                                                                      'for_pos', 'quat'],
                                                          }}}

settings['AerogridPlot'] = {'folder': './output/',
                            'include_rbm': 'off',
                            'include_applied_forces': 'on',
                            'minus_m_star': 0,
                            'u_inf': ws.u_inf
                            }
settings['AeroForcesCalculator'] = {'folder': './output/',
                                    'write_text_file': 'off',
                                    'text_file_name': ws.case_name + '_aeroforces.csv',
                                    'screen_output': 'on',
                                    'unsteady': 'off',
                                    'coefficients': True,
                                    'q_ref': 0.5 * ws.rho * ws.u_inf ** 2,
                                    'S_ref': 12.809
                                    }
settings['BeamPlot'] = {'folder': './output/',
                        'include_rbm': 'on',
                        'include_applied_forces': 'on',
                        'include_FoR': 'on'}

config = configobj.ConfigObj()
config.filename = ws.case_route + '/' + ws.case_name + '.sharpy'
for k, v in settings.items():
    config[k] = v
config.write()

import sharpy.sharpy_main
data = sharpy.sharpy_main.main(['', ws.case_route + '/' + ws.case_name + '.sharpy'])

