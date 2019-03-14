
import numpy as np
import tests.coupled.dynamic.cases_horten.horten_wing as horten_wing
import sharpy.sharpy_main

aero_type = 'lin'

ws = horten_wing.HortenWing(M=6,
                            N=11,
                            Mstarfactor=10,
                            u_inf=20,
                            thrust=5.202074681591291,
                            alpha_deg=6.0005198217378,
                            cs_deflection_deg=-5.866679093759233,
                            case_name_format=2,
                            physical_time=3,
                            case_remarks=aero_type,
                            case_route='cases/')
ws.horseshoe = False
ws.gust_intensity = -0.01
# ws.n_tstep = 1
# ws.dt_factor = 0.25
# ws.sweep_LE = 30 * np.pi/180

# ws.main_ea_root = 0.25
# ws.main_ea_tip = 0.25

ws.clean_test_files()
ws.update_mass_stiffness(sigma=1)
ws.update_aero_properties()
ws.update_fem_prop()
ws.generate_aero_file()
ws.generate_fem_file()
ws.set_default_config_dict()

ws.config['SHARPy']['flow'] = ['BeamLoader',
                               'AerogridLoader',
                               # 'StaticTrim',
                               'StaticCoupled',
                               # 'Modal',
                               'AeroForcesCalculator',
                               'AerogridPlot',
                               'BeamPlot',
                               'DynamicCoupled',
                               'AeroForcesCalculator',
                               'Modal',
                               # 'SaveData']
                                ]

ws.config['SHARPy']['write_screen'] = True
ws.config['DynamicCoupled']['n_time_steps'] = 1
ws.config['Modal']['rigid_body_modes'] = True
ws.config['Modal'] = {'print_info': True,
                             'use_undamped_modes': True,
                             'NumLambda': 308,
                             'rigid_body_modes': True,
                             'write_modes_vtk': 'on',
                             'print_matrices': 'on',
                             'write_data': 'on',
                             'continuous_eigenvalues': 'off',
                             'dt': ws.dt,
                             'plot_eigenvalues': False}

if aero_type == 'lin':
    ws.config['StepLinearUVLM']['solution_method'] = 'minsize'
    ws.config['DynamicCoupled']['aero_solver'] = 'StepLinearUVLM'
    ws.config['DynamicCoupled']['aero_solver_settings'] = {'dt': ws.settings['StepLinearUVLM']['dt'],
                                                           'remove_predictor': False,
                                                           'use_sparse': False,
                                                           'integr_order': 2,
                                                           'velocity_field_generator': 'GustVelocityField',
                                                           'velocity_field_input': {'u_inf': ws.u_inf,
                                                                                    'u_inf_direction': [1., 0., 0.],
                                                                                    'gust_shape': '1-cos',
                                                                                    'gust_length': 3.,
                                                                                    'gust_intensity': ws.gust_intensity
                                                                                                      * ws.u_inf,
                                                                                    'offset': 15.,
                                                                                    'span': ws.span}}

ws.config.write()

data = sharpy.sharpy_main.main(['',ws.case_route+'/'+ws.case_name+'.solver.txt'])

