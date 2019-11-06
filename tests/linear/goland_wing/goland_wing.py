import cases.templates.flying_wings as wings
import sharpy.sharpy_main

def create_goland_wing_modal_file(u_inf):

    ws = wings.Goland(M=12,
                      N=16,
                      Mstar_fact=20,
                      u_inf=u_inf,
                      alpha=1.,
                      rho=1.020,
                      sweep=0,
                      physical_time=2,
                      n_surfaces=2,
                      route='cases',
                      case_name='goland_u%04g' %int(u_inf))

    ws.gust_intensity = 0.01
    ws.sigma = 1

    ws.clean_test_files()
    ws.update_derived_params()
    ws.n_tstep = 1
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
                            'SaveData']
                                   # ]
    ws.config['SHARPy']['write_screen'] = 'on'

    ws.config['DynamicCoupled']['aero_solver_settings']['velocity_field_input']['gust_length'] = 5


    ws.config['Modal'] = {'folder': ws.route+'/output/',
                               'NumLambda': 1000,
                               'rigid_body_modes': False,
                               'print_matrices': 'off',
                               'keep_linear_matrices': 'on',
                               'write_modes_vtk': True,
                               'use_undamped_modes': True}

    ws.config.write()

    data = sharpy.sharpy_main.main(['',ws.route+ws.case_name+'.solver.txt'])

    return data

if __name__ == '__main__':
    create_goland_wing_modal_file(100)
