

import tests.coupled.dynamic.cases_horten.horten_wing as horten_wing
import sharpy.sharpy_main

aero_type = 'nlin'

ws = horten_wing.HortenWing(M=8,
                            N=19,
                            Mstarfactor=10,
                            u_inf=35,
                            thrust=2.959985072701038,
                            alpha_deg=3.8328492608310607,
                            cs_deflection_deg=-3.042203421093528,
                            case_name_format=2,
                            physical_time=10,
                            case_remarks=aero_type+'flutter',
                            case_route='cases/')
ws.horseshoe = False
ws.gust_intensity = 0.05
# ws.n_tstep = 2

ws.clean_test_files()
ws.update_mass_stiffness(sigma=0.3)
ws.update_aero_properties()
ws.update_fem_prop()
ws.generate_aero_file()
ws.generate_fem_file()
ws.set_default_config_dict()

ws.config['SHARPy']['flow'] = ['BeamLoader',
                               'AerogridLoader',
                               'StaticTrim',
                               'StaticCoupled',
                               'AeroForcesCalculator',
                               'AerogridPlot',
                               'BeamPlot',
                               'DynamicCoupled',
                               'AeroForcesCalculator',
                               # 'Modal',
                               'SaveData']

ws.config['SHARPy']['write_screen'] = True
# ws.config['DynamicCoupled']['n_time_steps'] = 2

if aero_type == 'lin':
    ws.config['StepLinearUVLM']['solution_method'] = 'minsize'
    ws.config['DynamicCoupled']['aero_solver'] = 'StepLinearUVLM'
    ws.config['DynamicCoupled']['aero_solver_settings'] = {'dt': ws.settings['StepLinearUVLM']['dt'],
                                                           'solution_method': 'minsize',
                                                           'velocity_field_generator': 'GustVelocityField',
                                                           'velocity_field_input': {'u_inf': ws.u_inf,
                                                                                    'u_inf_direction': [1., 0., 0.],
                                                                                    'gust_shape': '1-cos',
                                                                                    'gust_length': 1.,
                                                                                    'gust_intensity': ws.gust_intensity
                                                                                                      * ws.u_inf,
                                                                                    'offset': 30.,
                                                                                    'span': ws.span}}

ws.config.write()

# data = sharpy.sharpy_main.main(['',ws.case_route+ws.case_name+'.solver.txt'])
#
# import numpy as np
# max_force = np.zeros(3)
# for n in range(3):
#     max_force[n] = np.max(data.aero.timestep_info[n].forces[3])