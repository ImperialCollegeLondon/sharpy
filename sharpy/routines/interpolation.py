

def sol_550(variablesx, flow=[], **settings):
    pass
    
settings = dict()
# case_name = 'pmor_flutter_weakMAC_direct'
case_name = 'pmor_flutter_weakMAC'




settings['SHARPy'] = {'case': case_name,
                      'route': './cases/',
                      'flow': ['ParametricModelInterpolation'],
                      'write_screen': 'on',
                      'write_log': 'on',
                      'log_folder': './output/' + case_name + '/',
                      'log_file': case_name + '.log'}

settings['ParametricModelInterpolation'] = {'cases_folder': './source/output_source_30_50_11_modesign_scaled_uvlm',
                                            'reference_case': 5,
                                            'interpolation_system': 'aeroelastic',
                                            'input_file': './input.yaml',
                                            'cleanup_previous_cases': 'on',
                                            'library_filepath': '',  #'./source/interp_flutter_case5.pkl',
                                            # 'projection_method': 'strongMAC',
                                            # 'independent_interpolation': 'on',
                                            'interpolation_settings': {
                                                # 'aerodynamic': {
                                                    # 'projection_method': 'weakMAC',
                                                    # 'interpolation_space': 'direct',
                                                # },
                                                'aeroelastic': {
                                                    'projection_method': 'strongMAC',
                                                    'interpolation_space': 'direct',
                                                },
                                                # 'structural': {
                                                    # 'projection_method': 'weakMAC',
                                                    # 'interpolation_space': 'direct',
                                                # },
                                            },
                                            'interpolation_scheme': 'lagrange',
                                            'interpolation_degree': 1,
                                            'postprocessors': ['AsymptoticStability', 'FrequencyResponse'],
                                            'postprocessors_settings': {'AsymptoticStability': {'print_info': 'on',
                                                                                                'export_eigenvalues': 'on',
                                                                                                },
                                                                        'FrequencyResponse': {'print_info': 'on',
                                                                                              'compute_fom': 'on',
                                                                                              'frequency_bounds': [0.1,
                                                                                                                   100],
                                                                                              'num_freqs': 200,
                                                                                              'frequency_spacing': 'log',
                                                                                              'compute_hinf': 'off',
                                                                                              }
                                                                        }}
