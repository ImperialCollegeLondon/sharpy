from sharpy.utils.solver_interface import solver, BaseSolver, initialise_solver
import os
import sharpy.utils.settings as settings
import sharpy.rom.interpolation.pmorlibrary as pmorlibrary
import sharpy.rom.utils.librom_interp as librominterp
import numpy as np
import itertools
import sharpy.utils.cout_utils as cout
import shutil
import sharpy.rom.interpolation.interpolationsystem as interpolationsystem


@solver
class ParametricModelInterpolation(BaseSolver):
    """
    Warnings:
        Under development.

        Following the latest developments in the organisation of settings this documentation is outdated.

    This solver allows the user to obtain an interpolated reduced order model based on tabulated, stored
    SHARPy cases of the ``interpolation_system`` (``uvlm`` or ``aeroelastic``).

    This is a standalone solver, in other words, it can be the only solver in the SHARPy ``flow`` variable
    and loads the previously calculated SHARPy cases in the user defined ``cases_folder``.

    A reference ROM has to be chosen as the ``reference_case``, this is for the purposes of a coordinate transformation
    onto a set of generalised coordinates [1] which will be performed using the desired ``projection_method``. For more
    details on the different ways of achieving this congruent transformation
    see :class:`~sharpy.linear.rom.utils.librom_interp.InterpROM`.

    Once this transformation has been performed, we can choose whether to interpolate the ROMs directly or in the
    tangent manifold to the reference system by means of ``interpolation_space``.
    See :class:`~sharpy.linear.rom.utils.librom_interp.TangentInterpolation` for further details on the implications
    of interpolating in the tangent space.

    Finally, the ``interpolation_scheme`` sets the interpolation method (i.e. ``lagrange``, ``linear``, etc).
    Note that some methods are only available for monoparametric cases.

    The interpolation is performed at those points specified in the ``input_file``. This file is a simple ``.yaml``
    consisting of a list of cases with the appropriate parameters and values. See the example below.

    For each of the cases listed in the ``input_file``, an interpolated state-space will be produced with which the
    user may interact by means of ``postprocessors`` and their associated ``postprocessors_settings``.


    Examples:

        The input ``.yaml`` file may look something like this for a case with a single parameter:

        .. code-block:: yaml

            # Cases to run
            - alpha: 0.0  # case 1
            - alpha: 5.0  # case 2

        Or if you use multiple parameters:

        .. code-block:: yaml

            # Cases to run
            - alpha: 0.0  # case 1
              u_inf: 10
            - alpha: 5.0  # case 2
              u_inf: 15

        A possible set of settings for this solver could be given by:

        .. code-block:: python

            settings['ParametricModelInterpolation'] = {'cases_folder': './source/output/',
                                                    'reference_case': 0,
                                                    'interpolation_system': 'uvlm',
                                                    'input_file': './input_v.yaml',
                                                    'cleanup_previous_cases': 'on',
                                                    'projection_method': 'weakMAC',
                                                    'interpolation_space': 'direct',
                                                    'interpolation_scheme': 'linear',
                                                    'postprocessors': ['AsymptoticStability', 'FrequencyResponse'],
                                                    'postprocessors_settings': {'AsymptoticStability': {'print_info': 'on',
                                                                                                        'export_eigenvalues': 'on',
                                                                                                        },
                                                                                'FrequencyResponse': {'print_info': 'on',
                                                                                                      'compute_fom': 'on',
                                                                                                      'frequency_bounds': [1, 200],
                                                                                                      'num_freqs': 200,
                                                                                                      }
                                                                                }}

    Notes:
        This solver can be used as a standalone solver. In other words, it could be the only
        solver in the SHARPy ``flow`` variable.

    See Also:
        How to create pROM libraries in :class:`~sharpy.rom.interpolation.pmorlibrary.ROMLibrary`
    """
    solver_id = 'ParametricModelInterpolation'
    solver_classification = 'model reduction'

    settings_types = dict()
    settings_description = dict()
    settings_default = dict()
    settings_options = dict()

    settings_types['print_info'] = 'bool'
    settings_default['print_info'] = True
    settings_description['print_info'] = 'Display output to screen'

    settings_types['cases_folder'] = 'str'
    settings_default['cases_folder'] = ''
    settings_description['cases_folder'] = 'Path to folder containing cases, a new library will be generated.'

    settings_types['library_filepath'] = 'str'
    settings_default['library_filepath'] = ''
    settings_description['library_filepath'] = 'Filepath to .pkl file containing pROM library. If previously created.'

    settings_types['input_file'] = 'str'
    settings_default['input_file'] = None
    settings_description['input_file'] = 'Path to YAML file containing the input cases.'

    settings_types['cleanup_previous_cases'] = 'bool'
    settings_default['cleanup_previous_cases'] = False
    settings_description['cleanup_previous_cases'] = 'Reruns any previously computed interpolated ROMs.'

    settings_types['reference_case'] = 'int'
    settings_default['reference_case'] = -1
    settings_description['reference_case'] = "Reference case for coordinate transformation. If ``-1`` the library's " \
                                             'default value will be chosen. ' \
                                             'If the library has no set default, it will ' \
                                             'prompt the user.'

    settings_types['continuous_time'] = 'bool'
    settings_default['continuous_time'] = False
    settings_description['continuous_time'] = 'Convert systems to continuous time.'

    settings_types['interpolation_system'] = 'str'
    settings_default['interpolation_system'] = None
    settings_description['interpolation_system'] = 'System on which to perform the interpolation'
    settings_options['interpolation_system'] = ['aeroelastic', 'aerodynamic', 'structural']

    settings_types['interpolation_settings'] = 'dict'
    settings_default['interpolation_settings'] = dict()
    settings_description['interpolation_settings'] = 'Settings with keys ``aerodynamic`` and/or ``structural``, ' \
                                                     'depending on the choice for the ``interpolation_system``.'

    settings_types['independent_interpolation'] = 'bool'
    settings_default['independent_interpolation'] = False
    settings_description['independent_interpolation'] = 'Interpolate the aerodynamic and structural subsystems ' \
                                                        'independently.'

    interpolation_system_types = dict()
    interpolation_system_default = dict()

    interpolation_system_types['aerodynamic'] = 'dict'
    interpolation_system_default['aerodynamic'] = None

    interpolation_system_types['structural'] = 'dict'
    interpolation_system_default['structural'] = None

    interpolation_settings_types = dict()
    interpolation_settings_default = dict()
    interpolation_settings_description = dict()
    interpolation_settings_options = dict()

    interpolation_settings_types['projection_method'] = 'str'
    interpolation_settings_default['projection_method'] = None
    interpolation_settings_description['projection_method'] = 'Projection method employed in the transformation of the ' \
                                                              'reduced bases to a set of generalised coordinates.'
    interpolation_settings_options['projection_method'] = ['leastsq',
                                                           'strongMAC',
                                                           'strongMAC_BT',
                                                           'maraniello_BT',
                                                           'weakMAC_right_orth',
                                                           'weakMAC',
                                                           'amsallem',
                                                           'panzer',
                                                           ]

    interpolation_settings_types['interpolation_space'] = 'str'
    interpolation_settings_default['interpolation_space'] = 'direct'
    interpolation_settings_description[
        'interpolation_space'] = 'Perform a ``direct`` interpolation of the ROM matrices or perform ' \
                                 'the interpolation in the ``tangent`` manifold to the reference ' \
                                 'system.'
    interpolation_settings_options['interpolation_space'] = ['direct', 'tangent', 'real', 'tangentspd']

    # interpolation_settings_types['interpolation_scheme'] = 'str'
    # interpolation_settings_default['interpolation_scheme'] = None
    # interpolation_settings_description['interpolation_scheme'] = 'Desired interpolation scheme.'
    # interpolation_settings_options['interpolation_scheme'] = ['linear', 'lagrange']

    settings_types['interpolation_scheme'] = 'str'
    settings_default['interpolation_scheme'] = None
    settings_description['interpolation_scheme'] = 'Desired interpolation scheme.'
    settings_options['interpolation_scheme'] = ['linear', 'lagrange']

    settings_types['interpolation_degree'] = 'int'
    settings_default['interpolation_degree'] = 4
    settings_description['interpolation_degree'] = 'Degree of interpolation for applicable schemes, such as ' \
                                                   '``lagrange``.'

    settings_types['postprocessors'] = 'list(str)'
    settings_default['postprocessors'] = list()
    settings_description['postprocessors'] = 'List of the postprocessors to run at the end of every time step'
    settings_options['postprocessors'] = ['AsymptoticStability', 'FrequencyResponse', 'SaveStateSpace']  # supported postprocs for pMOR

    settings_types['postprocessors_settings'] = 'dict'
    settings_default['postprocessors_settings'] = dict()
    settings_description['postprocessors_settings'] = 'Dictionary with the applicable settings for every ' \
                                                      '``postprocessor``. Every ``postprocessor`` needs its entry, ' \
                                                      'even if empty.'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description, settings_options)

    interpolation_settings_table = settings.SettingsTable()
    __doc__ += interpolation_settings_table.generate(interpolation_settings_types,
                                                     interpolation_settings_default,
                                                     interpolation_settings_description,
                                                     interpolation_settings_options,
                                                     header_line='The acceptable interpolation settings for each '
                                                                 'state-space are listed below:')

    def __init__(self):
        self.settings = None
        self.interpolation_settings = None

        self.rom_library = None

        self.pmor = None  # InterpROM

        self.data = None
        self.postprocessors = dict()
        self.postproc_output_folder = dict()

        self.input_cases = list()
        self.folder = None

    def initialise(self, data):

        self.data = data

        self.settings = data.settings[self.solver_id]
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default,
                                 self.settings_options, no_ctype=True)

        self.interpolation_settings = self.settings['interpolation_settings']
        for k in self.settings['interpolation_settings'].keys():
            settings.to_custom_types(self.interpolation_settings[k], self.interpolation_settings_types,
                                     self.interpolation_settings_default, self.interpolation_settings_options,
                                     no_ctype=True)

        self.folder = self.data.output_folder

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

        for postproc in self.settings['postprocessors']:
            self.postprocessors[postproc] = initialise_solver(postproc)
            self.postprocessors[postproc].initialise(self.data, self.settings['postprocessors_settings'][postproc])
            self.postproc_output_folder[postproc] = self.postprocessors[postproc].folder
            if self.settings['cleanup_previous_cases']:
                try:
                    shutil.rmtree(self.postproc_output_folder[postproc])
                except FileNotFoundError:
                    pass

        cout.cout_wrap('Current library for ROM interpolation')
        self.rom_library.display_library()

        # load the actual data pickles from the pointers in the library
        self.rom_library.load_data_from_library()

        # Generate mappings for easier interpolation
        self.rom_library.sort_grid()

        if self.settings['interpolation_system'] == 'aeroelastic' and \
                self.settings['independent_interpolation']:
            if self.rom_library.data_library[0].linear.linear_system.uvlm.scaled:
                sys = interpolationsystem.CoupledScaledPMOR
            else:
                sys = interpolationsystem.CoupledPMOR
            self.pmor = sys(self.rom_library,
                            interpolation_settings=self.interpolation_settings)
        else:
            try:
                target_system = self.settings['interpolation_system']
            except KeyError:
                raise KeyError('No settings given for the interpolation_system: %s in '
                               'interpolation_settings' % self.settings['interpolation_settings'])
            system_settings = self.interpolation_settings[target_system]
            self.pmor = interpolationsystem.IndividualPMOR(self.rom_library,
                                                           target_system=target_system,
                                                           interpolation_settings=system_settings,
                                                           use_ct=False)

        # <<<<<<<<<<<<<<<<<<<<<<

        # >>>>> DEBUG: Save projected systems
        # self.pmor.save_projected_ss(self.settings['postprocessors_settings']['SaveStateSpace']['folder'])
        # <<<<<

        # Future: save for online use?

        # load input cases from file
        self.input_cases = librominterp.load_parameter_cases(self.settings['input_file'])

    def run(self):
        # keep this section for the online part i.e. the interpolation
        interpolated_roms = pmorlibrary.InterpolatedROMLibrary()

        if not self.settings['cleanup_previous_cases']:
            interpolated_roms.load_previous_cases(self.folder + './pmor_summary.txt')

        input_list = [case for case in self.input_cases if case not in interpolated_roms.parameter_list]
        for case_number, case in enumerate(input_list):

            cout.cout_wrap('Interpolating...')
            cout.cout_wrap('\tCase: %g of %g' % (case_number + 1, len(input_list)), 1)
            weights = self.interpolate(case,
                                       method=self.settings['interpolation_scheme'],
                                       interpolation_parameter=0,
                                       interpolation_degree=self.settings['interpolation_degree'])

            interpolated_ss = self.pmor(weights)

            interpolated_roms.append(interpolated_ss, case)

            for postproc in self.postprocessors:
                for target_system in self.settings['postprocessors_settings'][postproc]['target_system']:

                    ss = self.pmor.interpolated_systems[target_system]
                    self.postprocessors[postproc].folder = self.postproc_output_folder[postproc] + \
                                                           '/param_case%02g' % (interpolated_roms.case_number - 1) \
                                                           + '/' + target_system + '/'

                    if not os.path.exists(self.postprocessors[postproc].folder):
                        os.makedirs(self.postprocessors[postproc].folder)
                    self.postprocessors[postproc].run(ss=ss)

        interpolated_roms.write_summary(self.folder + './pmor_summary.txt')
        self.data.interp_rom = interpolated_roms

        return self.data

    def interpolate(self, case, method, interpolation_parameter, interpolation_degree=None):

        x_vec = self.rom_library.param_values[interpolation_parameter]
        x0 = case[self.rom_library.parameters[interpolation_parameter]]

        if method == 'lagrange':
            weights = librominterp.lagrange_interpolation(x_vec, x0, interpolation_degree=interpolation_degree)
            order = [i[0] for i in self.rom_library.mapping]
            weights = [weights[i] for i in order]  # give weights in order in which state-spaces are stored.

        elif method == 'linear':
            weights = self.interpolate_n_sys(case)

        else:
            raise NotImplementedError('Interpolation method %s not yet implemented/recognised' % method)

        return weights

    def interpolate_n_sys(self, case):
        """
        n-dimensional linear interpolation based on binary operations
        """
        weights = [0] * len(self.rom_library.data_library)
        # find lower limits
        lower_limit = [np.searchsorted(self.rom_library.param_values[self.rom_library.parameter_index[parameter]],
                                       case[parameter], side='right') - 1 for parameter in self.rom_library.parameters]
        upper_limit = [np.searchsorted(self.rom_library.param_values[self.rom_library.parameter_index[parameter]],
                                       case[parameter],
                                       side='left') for parameter in self.rom_library.parameters]

        # this is a table of binary combinations, i.e, if n = 2, the result is a list with [00, 01, 10, 11]
        bin_table = itertools.product(
            *[[0, 1] for i in range(len(self.rom_library.parameters))])  # the ``i`` is not used on purpose

        for permutation in bin_table:
            current_interpolation_weight = []
            for parameter in self.rom_library.parameters:
                param_index = self.rom_library.parameter_index[parameter]
                x_min = self.rom_library.param_values[param_index][lower_limit[param_index]]
                x_max = self.rom_library.param_values[param_index][upper_limit[param_index]]

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

            actual_rom_index = [sum(x) for x in zip(permutation, lower_limit)]  # sum of permutation + lower index
            try:
                real_rom_index = self.rom_library.inverse_mapping[tuple(actual_rom_index)]
                weights[real_rom_index] = np.prod(current_interpolation_weight)
            except IndexError:
                pass  # no data available hence we don't interpolate

            if sum(weights) == 1:
                if 1 in weights:
                    print('Hit data point')
                break  # you have hit a data point

        if sum(weights) < 1:
            cout.cout_wrap('Warning: Extrapolating data - You are at the edge of your data set', 4)
        return weights

