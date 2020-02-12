from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.rom.interpolation.pmorlibrary as pmorlibrary
import sharpy.rom.utils.librom_interp as librominterp
import numpy as np
import itertools


@solver
class ParametricModelReduction(BaseSolver):
    """
    Warnings:
        Under development

    Standalone solver to

    * Load parametric ROM libraries

    * Create parametric ROMs

    * Interpolate ROMs

    Notes:
        This solver can be used as a standalone solver. I.e it could be the only
        solver in the SHARPy ``flow`` variable.

    See Also:
        How to create pROM libraries in :class:`~sharpy.rom.interpolation.pmorlibrary.ROMLibrary`
    """
    solver_id = 'ParametricModelReduction'
    solver_classification = 'model reduction'

    settings_types = dict()
    settings_description = dict()
    settings_default = dict()
    settings_options = dict()

    settings_types['library_filepath'] = 'str'
    settings_default['library_filepath'] = ''
    settings_description['library_filepath'] = 'Filepath to .pkl file containing pROM library.'

    settings_types['reference_case'] = 'int'
    settings_default['reference_case'] = -1
    settings_description['reference_case'] = 'Reference case for coordinate transformation. If ``-1`` the library' \
                                             'default value is chosen. If the library has no set default, it will ' \
                                             'prompt the user.'

    settings_types['projection_method'] = 'str'
    settings_default['projection_method'] = None
    settings_description['projection_method'] = 'Projection method employed in the transformation of the ' \
                                                'reduced bases to a set of generalised coordinates.'
    settings_options['projection_method'] = ['leastsq',
                                             'strongMAC',
                                             'strongMAC_BT',
                                             'maraniello_BT',
                                             'weakMAC_right_orth',
                                             'weakMAC']

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.settings = None

        self.rom_library = None

        self.pmor = None  # InterpROM

        self.param_values = None
        self.mapping = None
        self.parameters = None
        self.parameter_index = None
        self.inverse_mapping = None

    def initialise(self, data):

        self.settings = data.settings[self.solver_id]
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default,
                                 self.settings_options, no_ctype=True)

        self.rom_library = pmorlibrary.ROMLibrary()

        if self.settings['library_filepath'] is '':
            self.rom_library.interface()
        else:
            self.rom_library.load_library(path=self.settings['library_filepath'])

        if self.settings['reference_case'] != -1 or self.rom_library.reference_case is None:
            self.rom_library.set_reference_case(self.settings['reference_case'])

        # Create pMOR
        ss_list = [rom.linear.linear_system.uvlm.ss for rom, _ in self.rom_library.library]
        vv_list = [rom.linear.linear_system.uvlm.rom['Krylov'].V for rom, _ in self.rom_library.library]
        wwt_list = [rom.linear.linear_system.uvlm.rom['Krylov'].W.T for rom, _ in self.rom_library.library]

        v_ref = vv_list[self.rom_library.reference_case]
        wt_ref = wwt_list[self.rom_library.reference_case]

        # self.pmor = librominterp.InterpROM(ss_list, vv_list, wwt_list, v_ref, wt_ref,
        #                                    method_proj=self.settings['projection_method'])

        # Transform onto gen coordinates
        # self.pmor.project()

        # Generate mappings for easier interpolation
        self.sort_grid()

        # Future: save for online use?

    def run(self):
        pass
        # keep this section for the online part i.e. the interpolation

        # input is list of dicts with parameters at each point
        input_list = [{'payload': 10, 'u_inf': 30}]  # simulated input

        for case in input_list:
            weights = [0] * len(self.rom_library.library)
            # find lower limits
            lower_limit = [np.searchsorted(self.param_values[self.parameter_index[parameter]],
                                           case[parameter], side='right') - 1 for parameter in self.parameters]
            upper_limit = [np.searchsorted(self.param_values[self.parameter_index[parameter]],
                                           case[parameter],
                                           side='left') for parameter in self.parameters]
            # upper_limit = [current_param + 1 for current_param in lower_limit]

            interpolation_weights = np.empty([len(self.parameters)] * len(self.parameters), dtype=float)
            bin_table = itertools.product(*[[0, 1] for i in range(len(self.parameters))])
            for permutation in bin_table:
                print(permutation)
                current_interpolation_weight = []
                for parameter in self.parameters:
                    param_index = self.parameter_index[parameter]
                    print(parameter)
                    print(case[parameter])
                    x_min = self.param_values[param_index][lower_limit[param_index]]
                    x_max = self.param_values[param_index][upper_limit[param_index]]
                    print(x_min)
                    print(x_max)

                    try:
                        alpha = 1 - (case[parameter] - x_min) / (x_max - x_min)
                        beta = 1 - alpha
                    except ZeroDivisionError:
                        alpha = 1
                        beta = 0
                    if permutation[param_index] == 0:
                        current_interpolation_weight.append(alpha)
                    elif permutation[param_index] == 1:
                        current_interpolation_weight.append(beta)
                    else:
                        raise ValueError('Permutation incorrect')

                interpolation_weights[permutation] = np.prod(current_interpolation_weight)  # visualising corners of the interpolation

                actual_rom_index = [sum(x) for x in zip(permutation, lower_limit)]  # sum of permutation + lower index
                try:
                    real_rom_index = self.inverse_mapping[tuple(actual_rom_index)]
                    weights[real_rom_index] = np.prod(current_interpolation_weight)
                except IndexError:
                    pass  # no data available hence we don't interpolate

                if sum(weights) == 1:
                    if 1 in weights:
                        print('Hit data point')
                    break  # you have hit a data point

            print(weights)


    def sort_grid(self):

        param_library = [parameter for _, parameter in self.rom_library.library]  # list of dicts
        parameters = list(param_library[0].keys())  # all should have the same parameters
        param_values = [[case[param] for case in param_library] for param in parameters]

        parameter_index = {parameter: ith for ith, parameter in enumerate(parameters)}

        # sort parameters
        [parameter_values.sort() for parameter_values in param_values]
        parameter_values = [f7(item) for item in param_values]  # TODO: rename these variables

        inverse_mapping = np.empty([len(parameter_values[n]) for n in range(len(parameters))], dtype=int)
        mapping = []
        i_case = 0
        for _, case_parameters in self.rom_library.library:
            current_case_mapping = []
            for ith, parameter in enumerate(parameters):
                p_index = parameter_values[ith].index(case_parameters[parameter])
                current_case_mapping.append(p_index)
            mapping.append(current_case_mapping)
            inverse_mapping[tuple(current_case_mapping)] = i_case
            i_case += 1

        self.parameters = parameters
        self.param_values = parameter_values  # TODO: caution -- rename
        self.mapping = mapping
        self.parameter_index = parameter_index
        self.inverse_mapping = inverse_mapping

# def find_nearest(array,value):
#     idx = np.searchsorted(array, value, side="left")
#     if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
#         return array[idx-1]
#     else:
#         return array[idx]

def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
