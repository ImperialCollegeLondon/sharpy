import sharpy.rom.interpolation.interpolationspaces
from sharpy.utils.solver_interface import solver, BaseSolver, initialise_solver
import os
import sharpy.utils.settings as settings
import sharpy.rom.interpolation.pmorlibrary as pmorlibrary
import sharpy.rom.utils.librom_interp as librominterp
import numpy as np
import itertools
import sharpy.utils.cout_utils as cout
import scipy.linalg as sclalg
import yaml
import sharpy.linear.src.libss as libss
import shutil


@solver
class ParametricModelInterpolation(BaseSolver):
    """
    Warnings:
        Under development

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

    settings_types['interpolation_system'] = 'str'
    settings_default['interpolation_system'] = None
    settings_description['interpolation_system'] = 'System on which to perform the interpolation'
    settings_options['interpolation_system'] = ['aeroelastic', 'uvlm']

    settings_types['projection_method'] = 'str'
    settings_default['projection_method'] = None
    settings_description['projection_method'] = 'Projection method employed in the transformation of the ' \
                                                'reduced bases to a set of generalised coordinates.'
    settings_options['projection_method'] = ['leastsq',
                                             'strongMAC',
                                             'strongMAC_BT',
                                             'maraniello_BT',
                                             'weakMAC_right_orth',
                                             'weakMAC',
                                             'amsallem',
                                             'panzer',
                                             ]

    settings_types['interpolation_space'] = 'str'
    settings_default['interpolation_space'] = 'direct'
    settings_description['interpolation_space'] = 'Perform a ``direct`` interpolation of the ROM matrices or perform ' \
                                                  'the interpolation in the ``tangent`` manifold to the reference ' \
                                                  'system.'
    settings_options['interpolation_space'] = ['direct', 'tangent', 'real']

    settings_types['interpolation_scheme'] = 'str'
    settings_default['interpolation_scheme'] = None
    settings_description['interpolation_scheme'] = 'Desired interpolation scheme.'
    settings_options['interpolation_scheme'] = ['linear', 'lagrange']

    settings_types['postprocessors'] = 'list(str)'
    settings_default['postprocessors'] = list()
    settings_description['postprocessors'] = 'List of the postprocessors to run at the end of every time step'
    settings_options['postprocessors'] = ['AsymptoticStability', 'FrequencyResponse']  # supported postprocs for pMOR

    settings_types['postprocessors_settings'] = 'dict'
    settings_default['postprocessors_settings'] = dict()
    settings_description['postprocessors_settings'] = 'Dictionary with the applicable settings for every ' \
                                                      '``postprocessor``. Every ``postprocessor`` needs its entry, ' \
                                                      'even if empty.'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description, settings_options)

    def __init__(self):
        self.settings = None

        self.rom_library = None

        self.pmor = None  # InterpROM

        self.param_values = None
        self.mapping = None
        self.parameters = None
        self.parameter_index = None
        self.inverse_mapping = None

        self.data = None
        self.postprocessors = dict()
        self.postproc_output_folder = dict()

        self.input_cases = list()

    def initialise(self, data):

        self.data = data

        self.settings = data.settings[self.solver_id]
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default,
                                 self.settings_options, no_ctype=True)

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

        if self.settings['print_info']:
            cout.cout_wrap.cout_talk()
        else:
            cout.cout_wrap.cout_quiet()

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

        ss_list, vv_list, wwt_list = self.aeroelastic_bases()  # list of ss and reduced order bases

        ## >>> testing basis interpolation
        #
        # self.pmor = librominterp.BasisInterpolation(vv_list, wwt_list, ss_list, self.rom_library.reference_case)
        # self.pmor.create_tangent_space()
        #
        ## <<< end of test - to be removed to a dedicated solver

        if self.settings['interpolation_space'] == 'direct':
            cout.cout_wrap('\tInterpolating Directly', 1)
            self.pmor = sharpy.rom.interpolation.interpolationspaces.InterpROM()
        elif self.settings['interpolation_space'] == 'tangent':
            cout.cout_wrap('\tInterpolating in the Tangent space', 1)
            self.pmor = sharpy.rom.interpolation.interpolationspaces.TangentInterpolation()
        elif self.settings['interpolation_space'] == 'real':
            cout.cout_wrap('\tInterpolating Real Matrices', 1)
            self.pmor = sharpy.rom.interpolation.interpolationspaces.InterpolationRealMatrices()
        else:
            raise NotImplementedError('Interpolation space %s is not recognised' % self.settings['interpolation_space'])

        self.pmor.initialise(ss_list, vv_list, wwt_list,
                             method_proj=self.settings['projection_method'],
                             reference_case=self.rom_library.reference_case)

        # Transform onto gen coordinates
        self.pmor.project()

        # Generate mappings for easier interpolation
        self.sort_grid()

        # Future: save for online use?

        # load input cases
        self.input_cases = librominterp.load_parameter_cases(self.settings['input_file'])

    def run(self):
        # keep this section for the online part i.e. the interpolation
        interpolated_roms = pmorlibrary.InterpolatedROMLibrary()

        if not self.settings['cleanup_previous_cases']:
            interpolated_roms.load_previous_cases(self.data.settings['SHARPy']['log_folder'] + './pmor_summary.txt')

        input_list = [case for case in self.input_cases if case not in interpolated_roms.parameter_list]
        for case_number, case in enumerate(input_list):

            cout.cout_wrap('Interpolating...')
            cout.cout_wrap('\tCase: %g of %g' % (case_number, len(input_list)), 1)
            weights = self.interpolate(case,
                                       method=self.settings['interpolation_scheme'],
                                       interpolation_parameter=0)

            interpolated_ss = self.pmor(weights)

            # >>>> Basis Interpolation
            # interpolated_ss = self.pmor.interpolate(weights, ss=self.retrieve_fom(self.rom_library.reference_case))
            # interpolated_ss = self.pmor.interpolate(weights, ss=self.retrieve_fom(1))
            # <<<< Basis Interpolation

            interpolated_roms.append(interpolated_ss, case)

            for postproc in self.postprocessors:
                self.postprocessors[postproc].folder = self.postproc_output_folder[postproc] + \
                                                       '/param_case%02g' % (interpolated_roms.case_number - 1) + '/'
                if not os.path.exists(self.postprocessors[postproc].folder):
                    os.makedirs(self.postprocessors[postproc].folder)
                self.postprocessors[postproc].run(ss=interpolated_ss)

        interpolated_roms.write_summary(self.data.settings['SHARPy']['log_folder'] + './pmor_summary.txt')
        self.data.interp_rom = interpolated_roms

        return self.data

    def interpolate(self, case, method, interpolation_parameter):

        x_vec = self.param_values[interpolation_parameter]
        x0 = case[self.parameters[interpolation_parameter]]

        if method == 'lagrange':
            weights = librominterp.lagrange_interpolation(x_vec, x0)
            order = [i[0] for i in self.mapping]
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
        lower_limit = [np.searchsorted(self.param_values[self.parameter_index[parameter]],
                                       case[parameter], side='right') - 1 for parameter in self.parameters]
        upper_limit = [np.searchsorted(self.param_values[self.parameter_index[parameter]],
                                       case[parameter],
                                       side='left') for parameter in self.parameters]

        bin_table = itertools.product(*[[0, 1] for i in range(len(self.parameters))])  # this is a table of binary
                                                                                       # combinations, i.e, if n = 2,
                                                                                       # the result is a list with
                                                                                       # [00, 01, 10, 11]
        for permutation in bin_table:
            current_interpolation_weight = []
            for parameter in self.parameters:
                param_index = self.parameter_index[parameter]
                x_min = self.param_values[param_index][lower_limit[param_index]]
                x_max = self.param_values[param_index][upper_limit[param_index]]

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
                real_rom_index = self.inverse_mapping[tuple(actual_rom_index)]
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

    def sort_grid(self):

        param_library = [case['parameters'] for case in self.rom_library.library]
        parameters = list(param_library[0].keys())  # all should have the same parameters
        param_values = [[case[param] for case in param_library] for param in parameters]

        parameter_index = {parameter: ith for ith, parameter in enumerate(parameters)}

        # sort parameters
        [parameter_values.sort() for parameter_values in param_values]
        parameter_values = [f7(item) for item in param_values]  # TODO: rename these variables

        inverse_mapping = np.empty([len(parameter_values[n]) for n in range(len(parameters))], dtype=int)
        mapping = []
        i_case = 0
        for case_parameters in param_library:
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

    def aeroelastic_bases(self):
        """
        Returns the bases and state spaces of the chosen systems.

        To Do: find system regardless of MOR method

        Returns:
            tuple: list of state spaces, list of right ROBs and list of left ROBs
        """
        if self.settings['interpolation_system'] == 'uvlm':
            ss_list = [rom.linear.linear_system.uvlm.ss for rom in self.rom_library.data_library]
            vv_list = [rom.linear.linear_system.uvlm.rom['Krylov'].V for rom in self.rom_library.data_library]
            wwt_list = [rom.linear.linear_system.uvlm.rom['Krylov'].W.T for rom in self.rom_library.data_library]
        elif self.settings['interpolation_system'] == 'aeroelastic':
            ss_list = []
            vv_list = []
            wwt_list = []
            for rom in self.rom_library.data_library:
                vv = sclalg.block_diag(rom.linear.linear_system.uvlm.rom['Krylov'].V,
                                       np.eye(rom.linear.linear_system.beam.ss.states))
                wwt = sclalg.block_diag(rom.linear.linear_system.uvlm.rom['Krylov'].W.T,
                                        np.eye(rom.linear.linear_system.beam.ss.states))
                ss_list.append(rom.linear.ss)
                vv_list.append(vv)
                wwt_list.append(wwt)
        else:
            raise NameError('Unrecognised system on which to perform interpolation')

        return ss_list, vv_list, wwt_list

    def retrieve_fom(self, rom_index):
        # Move to a dedicated solver for reduced order basis interpolation
        ss_fom_aero = self.rom_library.data_library[rom_index].linear.linear_system.uvlm.rom['Krylov'].ss
        ss_fom_beam = self.rom_library.data_library[rom_index].linear.linear_system.beam.ss

        Tas = np.eye(ss_fom_aero.inputs, ss_fom_beam.outputs)
        Tsa = np.eye(ss_fom_beam.inputs, ss_fom_aero.outputs)

        ss = libss.couple(ss_fom_aero, ss_fom_beam, Tas, Tsa)

        return ss


# def find_nearest(array,value):
#     idx = np.searchsorted(array, value, side="left")
#     if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
#         return array[idx-1]
#     else:
#         return array[idx]

def f7(seq):
    """
    Adds single occurrences of an item in a list
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
