"""Basis Interpolation Model Reduction"""
import numpy as np
import sharpy.utils.settings as settings
import sharpy.utils.cout_utils as cout
import sharpy.utils.rom_interface as rom_interface
import sharpy.rom.interpolation.pmorlibrary as pmorlibrary
import sharpy.rom.interpolation.interpolationspaces as interpolationspaces
import sharpy.rom.utils.librom_interp as librominterp

@rom_interface.rom
class BasisInterpolation(rom_interface.BaseRom):
    """
    Model Order Reduction by basis interpolation.
    """
    rom_id = 'BasisInterpolation'

    settings_types = dict()
    settings_description = dict()
    settings_default = dict()
    settings_options = dict()

    settings_types['cases_folder'] = 'str'
    settings_default['cases_folder'] = None
    settings_description['cases_folder'] = 'Path to folder containing cases, a new library will be generated.'

    settings_types['reference_case'] = 'int'
    settings_default['reference_case'] = -1
    settings_description['reference_case'] = "Reference case for coordinate transformation. If ``-1`` the library's " \
                                             'default value will be chosen. ' \
                                             'If the library has no set default, it will ' \
                                             'prompt the user.'

    settings_types['input_file'] = 'str'
    settings_default['input_file'] = None
    settings_description['input_file'] = 'Path to YAML file containing the input cases.'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description, settings_options)

    def __init__(self):
        self.settings = None

        self.rom_library = None
        self.pmor = None

        self.input_cases = None

    def initialise(self, in_settings):

        self.settings = in_settings

        settings.to_custom_types(self.settings, self.settings_types, self.settings_default, self.settings_options, no_ctype=True)

        self.rom_library = pmorlibrary.ROMLibrary()
        self.rom_library.create(settings={'pickle_source_path': self.settings['cases_folder']})
        self.rom_library.set_reference_case(self.settings['reference_case'])

        self.rom_library.display_library()
        self.rom_library.sort_grid()
        self.rom_library.load_data_from_library()

        ss_list, vv_list, wwt_list = self.rom_library.get_reduced_order_bases(target_system='uvlm')
        self.pmor = interpolationspaces.BasisInterpolation(v_list=vv_list, vt_list=wwt_list, ss_list=ss_list)

        self.pmor.create_tangent_space()

        # Change this: input case can only be one (the one being run)
        self.input_cases = librominterp.load_parameter_cases(self.settings['input_file'])

    def run(self, ss):

        weights = self.interpolate(self.input_cases[0], 'lagrange', interpolation_parameter=0)

        return self.pmor.interpolate(weights, ss)

    def interpolate(self, case, method, interpolation_parameter):

        x_vec = self.rom_library.param_values[interpolation_parameter]
        x0 = case[self.rom_library.parameters[interpolation_parameter]]

        if method == 'lagrange':
            weights = librominterp.lagrange_interpolation(x_vec, x0)
            order = [i[0] for i in self.rom_library.mapping]
            weights = [weights[i] for i in order]  # give weights in order in which state-spaces are stored.

        else:
            raise NotImplementedError('Interpolation method %s not yet implemented/recognised' % method)

        return weights
