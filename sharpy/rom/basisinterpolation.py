"""Basis Interpolation Model Reduction"""
import sharpy.utils.settings as settings
import sharpy.utils.rom_interface as rom_interface
import sharpy.rom.interpolation.pmorlibrary as pmorlibrary
import sharpy.rom.interpolation.interpolationspaces as interpolationspaces
import sharpy.rom.utils.librom_interp as librominterp


@rom_interface.rom
class BasisInterpolation(rom_interface.BaseRom):
    """
    Model Order Reduction by basis interpolation [1].

    This model order reduction method for the UVLM system is based on interpolating the reduced order bases generated
    for some source cases and employing an interpolated version of these for the reduction of the given high fidelity
    model.

    The user specifies a ``cases_folder`` where the parametric source cases are stored. Reference
    :class:`~sharpy.solvers.SaveParametricCase` to see how to properly save the source cases.

    The interpolation of the basis requires of a reference set of bases, specified by the ``reference_case``. An optimal
    choice for the reference is still an open problem [2].

    Finally, the ``interpolation_parameter`` is parsed as a dictionary, where the key in the dictionary is the parameter
    name, which has to be the same as that used to save the source parametric case.

    The interpolation is performed for a single parameter using a Lagrange interpolation method.

    Examples:

        We desire to obtain reduced order bases at an interpolated point, where the parameter is the free stream
        velocity. For the source cases, we include the :class:`~sharpy.solvers.SaveParametricCase` with the following
        settings:

        .. code-block::

            settings = dict()
            settings['SaveParametricCase'] = {'folder': '<path_to_source_output>',
                                              'parameters': {
                                                   'u_inf': 10}  # The name of the parameter is at the user's discretion
                                               }


        Once we have the set of source cases, we can run a new case where the reduced order bases are calculated by
        interpolation. An example set of settings could be

        .. code-block::

            interpolation_settings = dict()
            interpolation_settings['cases_folder'] = self.route_test_dir + '/source/output/'
            interpolation_settings['reference_case'] = 0
            interpolation_settings['interpolation_parameter']: {'u_inf': 15}  # The name must be the same as in the source

            # These settings can now be used to define the ROM in the linear UVLM settings.

            settings = dict()  # SHARPy case settings. Populate as desired
            settings['LinearUVLM'] = {'rom_method': ['BasisInterpolation'],
                                      'rom_method_settings': {'BasisInterpolation':interpolation_settings}}


    References:

        [1] Amsallem, D., & Farhat, C. (2008). Interpolation method for adapting reduced-order models and application
        to aeroelasticity. AIAA Journal, 46(7), 1803–1813. https://doi.org/10.2514/1.35374

        [2] Amsallem, D., & Farhat, C. (2011). An Online Method for interpolating linear parametric reduced order
        models, 33(5), 2169–2198.
    """
    rom_id = 'BasisInterpolation'

    settings_types = dict()
    settings_description = dict()
    settings_default = dict()
    settings_options = dict()

    settings_types['cases_folder'] = 'str'
    settings_default['cases_folder'] = None
    settings_description['cases_folder'] = 'Path to folder containing cases, a new library will be generated.'

    settings_types['library_filepath'] = 'str'
    settings_default['library_filepath'] = ''
    settings_description['library_filepath'] = 'Filepath to .pkl file containing pROM library. If previously created.'

    settings_types['reference_case'] = 'int'
    settings_default['reference_case'] = -1
    settings_description['reference_case'] = "Reference case for coordinate transformation. If ``-1`` the library's " \
                                             'default value will be chosen. ' \
                                             'If the library has no set default, it will ' \
                                             'prompt the user.'

    settings_types['interpolation_parameter'] = 'dict'
    settings_default['interpolation_parameter'] = None
    settings_description['interpolation_parameter'] = 'Dictionary containing the name of the interpolation parameter ' \
                                                      'as key and the corresponding value to interpolate. Ensure the ' \
                                                      'name of the parameter is the same as that used in the source ' \
                                                      'case.'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description, settings_options)

    def __init__(self):
        self.settings = None

        self.rom_library = None  #: sharpy.rom.interpolation.pmorlibrary.ROMlibrary
        self.pmor = None  #: sharpy.rom.interpolation.interpolationspaces.BasisInterpolation

        self.input_cases = None  #: dict where the key is the parameter name and value is the parameter value

    def initialise(self, in_settings):

        self.settings = in_settings

        settings.to_custom_types(self.settings, self.settings_types, self.settings_default, self.settings_options,
                                 no_ctype=True)

        self.rom_library = pmorlibrary.ROMLibrary()
        if self.settings['cases_folder'] is not '':
            # creates library from a folder containing all cases
            new_library_settings = {'pickle_source_path': self.settings['cases_folder']}
            self.rom_library.create(settings=new_library_settings)

        elif self.settings['library_filepath'] is '':
            self.rom_library.interface()

        else:
            self.rom_library.load_library(path=self.settings['library_filepath'])

        if self.settings['reference_case'] != -1 or self.rom_library.reference_case is None:
            self.rom_library.set_reference_case(self.settings['reference_case'])

        self.rom_library.display_library()
        self.rom_library.sort_grid()
        self.rom_library.load_data_from_library()

        ss_list, vv_list, wwt_list = self.rom_library.get_reduced_order_bases(target_system='uvlm')
        self.pmor = interpolationspaces.BasisInterpolation(v_list=vv_list,
                                                           reference_case=self.rom_library.reference_case)

        self.pmor.create_tangent_space()

        self.input_cases = self.settings['interpolation_parameter']

    def run(self, ss):

        weights = self.interpolate(self.input_cases, 'lagrange', interpolation_parameter=0)

        return self.pmor.interpolate(weights, ss)

    def interpolate(self, case, method, interpolation_parameter):

        x_vec = self.rom_library.param_values[interpolation_parameter]
        x0 = float(case[self.rom_library.parameters[interpolation_parameter]])

        if method == 'lagrange':
            weights = librominterp.lagrange_interpolation(x_vec, x0)
            order = [i[0] for i in self.rom_library.mapping]
            weights = [weights[i] for i in order]  # give weights in order in which state-spaces are stored.

        else:
            raise NotImplementedError('Interpolation method %s not yet implemented/recognised' % method)

        return weights
