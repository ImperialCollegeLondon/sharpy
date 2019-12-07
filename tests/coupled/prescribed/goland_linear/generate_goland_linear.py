import cases.templates.flying_wings as wings
import sharpy.sharpy_main

aero_type = 'lin'
ws = wings.Goland(M=4,
                  N=12,
                  Mstar_fact=10,
                  u_inf=50,
                  alpha=1.,
                  rho=1.225,
                  sweep=0,
                  physical_time=2,
                  n_surfaces=2,
                  route='cases',
                  case_name='goland_'+aero_type+'_newsolver_pred')

ws.gust_intensity = 0.01
# ws.n_tstep = 2
ws.sigma = 1

ws.clean_test_files()
ws.update_derived_params()
ws.update_aero_prop()
ws.update_fem_prop()
ws.set_default_config_dict()

ws.generate_aero_file()
ws.generate_fem_file()

ws.config['SHARPy']['flow'] = ['BeamLoader', 'AerogridLoader',
                        #'StaticUvlm',
                        'StaticCoupled',
                        'AerogridPlot', 'BeamPlot',
                        'DynamicCoupled','Modal',
                        # 'SaveData']
                               ]
ws.config['SHARPy']['write_screen'] = 'on'

ws.config['DynamicCoupled']['aero_solver_settings']['velocity_field_input']['gust_length'] = 5
if aero_type == 'lin':
    ws.config['DynamicCoupled']['aero_solver'] = 'StepLinearUVLM'
    ws.config['DynamicCoupled']['aero_solver_settings'] = {'dt': ws.dt,
                                                           'remove_predictor': False,
                                                           'use_sparse': False,
                                                           'integr_order': 2,
                                                           'velocity_field_generator': 'GustVelocityField',
                                                           'velocity_field_input': {'u_inf': ws.u_inf,
                                                                                    'u_inf_direction': [1., 0., 0.],
                                                                                    'gust_shape': '1-cos',
                                                                                    'offset': 15.,
                                                                                    'gust_parameters': {'gust_length': 5.,
                                                                                                        'gust_intensity': ws.gust_intensity
                                                                                                                          * ws.u_inf,
                                                                                                        'span': ws.main_chord * ws.aspect_ratio}}}
                                                           #  'velocity_field_generator': 'SteadyVelocityField',
                                                           #  'velocity_field_input': {'u_inf': ws.u_inf*1,
                                                           #                          'u_inf_direction': [1., 0., 0.]}}

ws.config.write()

data = sharpy.sharpy_main.main(['',ws.route+ws.case_name+'.sharpy'])
