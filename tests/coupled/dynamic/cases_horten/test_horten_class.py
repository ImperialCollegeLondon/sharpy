

import tests.coupled.dynamic.cases_horten.horten_wing as horten_wing
import sharpy.sharpy_main

aero_type = 'nlin'

ws = horten_wing.HortenWing(M=10,
                            N=19,
                            Mstarfactor=10,
                            u_inf=18,
                            thrust=7.338619348130693,
                            alpha_deg=7.648445762565274,
                            cs_deflection_deg=-9.085218584145167,
                            case_name_format=2,
                            physical_time=10,
                            case_remarks=aero_type+'modes',
                            case_route='cases/')
ws.horseshoe = False
ws.gust_intensity = 0.01
# ws.n_tstep = 2

ws.main_ea_root = 0.25
ws.main_ea_tip = 0.25

ws.clean_test_files()
ws.update_mass_stiffness(sigma=0.3)
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
                               # 'BeamPlot',
                               'DynamicCoupled',
                               # 'AeroForcesCalculator',
                               'Modal',
                               # 'SaveData']
                                ]

ws.config['SHARPy']['write_screen'] = True
ws.config['DynamicCoupled']['n_time_steps'] = 1
ws.config['Modal']['rigid_body_modes'] = True

if aero_type == 'lin':
    ws.config['StepLinearUVLM']['solution_method'] = 'minsize'
    ws.config['DynamicCoupled']['aero_solver'] = 'StepLinearUVLM'
    ws.config['DynamicCoupled']['aero_solver_settings'] = {'dt': ws.settings['StepLinearUVLM']['dt'],
                                                           'remove_predictor': False,
                                                           'use_sparse': False,
                                                           'integr_order': 1,
                                                           'velocity_field_generator': 'GustVelocityField',
                                                           'velocity_field_input': {'u_inf': ws.u_inf,
                                                                                    'u_inf_direction': [1., 0., 0.],
                                                                                    'gust_shape': '1-cos',
                                                                                    'gust_length': 1.,
                                                                                    'gust_intensity': ws.gust_intensity
                                                                                                      * ws.u_inf,
                                                                                    'offset': 15.,
                                                                                    'span': ws.span}}

ws.config.write()

data = sharpy.sharpy_main.main(['',ws.case_route+ws.case_name+'.solver.txt'])

