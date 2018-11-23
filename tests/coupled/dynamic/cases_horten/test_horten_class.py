

import tests.coupled.dynamic.cases_horten.horten_wing as horten_wing
import sharpy.sharpy_main

ws = horten_wing.HortenWing(M=6,
                            N=11,
                            Mstarfactor=10,
                            u_inf=30)
ws.horseshoe = True

ws.clean_test_files()
ws.update_mass_stiffness()
ws.update_aero_properties()
ws.update_fem_prop()
ws.generate_aero_file()
ws.generate_fem_file()
ws.set_default_config_dict()

data = sharpy.sharpy_main.main(['',ws.case_route+ws.case_name+'.solver.txt'])
