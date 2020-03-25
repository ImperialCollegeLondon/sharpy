import configobj
import sharpy.sharpy_main

def generate_pmor(source_path, pmor_route, input_file, pmor_output):
    settings = dict()
    case_name = 'goland_ct_pmor'
    settings['SHARPy'] = {'case': case_name,
                          'route': pmor_route,
                          'flow': ['ParametricModelInterpolation'],
                          'write_screen': 'off',
                          'write_log': 'on',
                          'log_folder': pmor_output + '/' + case_name + '/',
                          'log_file': 'pmor' + '.log'}

    settings['ParametricModelInterpolation'] = {'cases_folder': source_path,
                                                'reference_case': 0,
                                                'interpolation_system': 'aeroelastic',
                                                'input_file': input_file,
                                                'cleanup_previous_cases': 'on',
                                                'projection_method': 'weakMAC',
                                                'interpolation_space': 'direct',
                                                'interpolation_scheme': 'lagrange',
                                                'postprocessors': ['FrequencyResponse'],
                                                'postprocessors_settings': {'FrequencyResponse': {'frequency_bounds': [10,
                                                                                                                       200],
                                                                                                  'num_freqs': 200,
                                                                                                  'folder': pmor_output
                                                                                                  }
                                                                            }}


    config = configobj.ConfigObj()
    file_name = pmor_route + '/pmor.sharpy'
    config.filename = file_name
    for k, v in settings.items():
        config[k] = v
    config.write()

    sharpy.sharpy_main.main(['', file_name])

    return case_name
