

import tests.coupled.dynamic.cases_horten.horten_wing as horten_wing
import sharpy.sharpy_main

ws = horten_wing.HortenWing(M=6,
                            N=11,
                            Mstarfactor=10,
                            u_inf=30,
                            thrust=1.9100217951807725,
                            alpha_deg=2.897652932442695,
                            cs_deflection_deg=-2.115640253647247,
                            case_name_format=1)
ws.horseshoe = False

ws.clean_test_files()
ws.update_mass_stiffness()
ws.update_aero_properties()
ws.update_fem_prop()
ws.generate_aero_file()
ws.generate_fem_file()
ws.set_default_config_dict()

ws.config['SHARPy']['flow'] = ['BeamLoader',
                               'AerogridLoader',
                               'StaticTrim',
                               'StaticCoupled',
                               'AerogridPlot',
                               'BeamPlot',
                               'DynamicCoupled',
                               'Modal',
                               'SaveData']

ws.config['SHARPy']['write_screen'] = True
ws.config['DynamicCoupled']['aero_solver'] = 'StepLinearUVLM'


ws.config.write()

data = sharpy.sharpy_main.main(['',ws.case_route+ws.case_name+'.solver.txt'])
