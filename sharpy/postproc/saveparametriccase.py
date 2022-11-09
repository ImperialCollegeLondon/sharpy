from sharpy.utils.solver_interface import solver, BaseSolver, initialise_solver
import sharpy.utils.settings as settings_utils
import configobj
import os
import sharpy.utils.cout_utils as cout
import warnings


@solver
class SaveParametricCase(BaseSolver):
    """
    SaveParametricCase is a post-processor that creates a ConfigParser text file called
    ``<sharpy_case_name>.pmor.sharpy`` that contains information on certain simulation parameters. It is useful as
    a record keeper if you are doing a parametric study and for parametric model interpolation.


    If the setting ``save_case`` is selected and the post processor :class:`~sharpy.solvers.pickledata.PickleData`
    is not present in the SHARPy flow, this solver will pickle the data to the path given in the ``folder`` setting.

    The setting ``save_pmor_items`` saves to h5 the following state-spaces and gains, if present:
        * Aeroelastic state-space saved to: <output_folder> / save_pmor_data / <case_name>_statespace.h5
        * Aerodynamic ROM reduced order bases saved to: <output_folder> / save_pmor_data / <case_name>_aerorob.h5
        * Structural ROM reduced order bases saved to: <output_folder> / save_pmor_data / <case_name>_modal_structrob.h5

    The setting ``save_pmor_subsystem saves the additional state-spaces to h5 files:
        * Structural matrices saved to: <output_folder> / save_pmor_data / <case_name>_struct_matrices.h5
        * Structural state-space saved to: <output_folder> / save_pmor_data / <case_name>_beamstatespace.h5
        * Aerodynamic state-space saved to: <output_folder> / save_pmor_data / <case_name>_aerostatespace.h5

    Examples:

        In the case you are running several SHARPy cases, varying for instance the velocity, the settings would
        be something like:

        >>> parameter_value = 10  # parameter of study
        >>> input_settings = {'<name_of_your_parameter>': value  # the name of the parameter is at the user's discretion
        >>>                  }  # add more parameters as required

        The result would be the ``<sharpy_case_name>.pmor.sharpy`` file with the following content:

        .. code-block:: none

            [parameters]
            <name_of_your_parameter> = value

    """
    solver_id = 'SaveParametricCase'
    solver_classification = 'post-processor'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['save_case'] = 'bool'
    settings_default['save_case'] = False
    settings_description['save_case'] = 'DeprecationWarning - Save a .pkl of the SHARPy case. Required for PMOR.'

    settings_types['parameters'] = 'dict'
    settings_default['parameters'] = None
    settings_description['parameters'] = 'Dictionary containing the chosen simulation parameters and their values.'

    settings_types['save_pmor_items'] = 'bool'
    settings_default['save_pmor_items'] = False
    settings_description['save_pmor_items'] = 'Saves to h5 the items required for PMOR interpolation: the aerodynamic ' \
                                              'reduced order bases, the structural modal matrix and the ' \
                                              'reduced state-space'

    settings_types['save_pmor_subsystems'] = 'bool'
    settings_default['save_pmor_subsystems'] = False
    settings_description['save_pmor_subsystems'] = 'Saves to h5 the statespaces and matrices of the UVLM and beam and ' \
                                                   'the M, C, K matrices. The setting ``save_pmor_items`` ' \
                                                   'should be set to `on`'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.data = None
        self.settings = None
        self.folder = None
        self.case_name = None

    def initialise(self, data, custom_settings=None, restart=False):
        self.data = data

        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings

        settings_utils.to_custom_types(self.settings,
                           self.settings_types,
                           self.settings_default)

        self.folder = data.output_folder
        self.case_name = self.data.settings['SHARPy']['case']

    def run(self, **kwargs):
        
        online = settings_utils.set_value_or_default(kwargs, 'online', False)
        restart = settings_utils.set_value_or_default(kwargs, 'restart', False)

        config = configobj.ConfigObj()
        file_name = self.folder + '/' + self.data.settings['SHARPy']['case'] + '.pmor.sharpy'
        config.filename = file_name
        config['parameters'] = dict()
        for k, v in self.settings['parameters'].items():
            cout.cout_wrap('\tWriting parameter %s: %s' % (k, str(v)), 1)
            config['parameters'][k] = v

        sim_info = dict()
        sim_info['case'] = self.data.settings['SHARPy']['case']

        if 'PickleData' not in self.data.settings['SHARPy']['flow'] and self.settings['save_case']:
            warnings.warn('Post-proc: SaveParametricCase: Saving a pickle is not recommended - try saving required '
                          'attributes individually',
                          DeprecationWarning)
            pickle_solver = initialise_solver('PickleData')
            pickle_solver.initialise(self.data, restart=restart)
            self.data = pickle_solver.run()
            sim_info['path_to_data'] = os.path.abspath(self.folder)

        sim_info['path_to_data'] = os.path.abspath(self.folder)

        config['sim_info'] = sim_info
        config.write()

        if self.settings['save_pmor_items']:
            try:
                self.data.linear
            except AttributeError:
                pass
            else:
                if not os.path.exists(self.folder + '/save_pmor_data/'):
                    os.makedirs(self.folder + '/save_pmor_data/')
                self.save_state_space()
                self.save_aero_rom_bases()
                self.save_structural_modal_matrix()

                if self.settings['save_pmor_subsystems']:
                    self.save_structural_matrices()
                    self.save_aero_state_space()

        return self.data

    def save_aero_rom_bases(self):
        """Save the aerodynamic reduced order bases to h5 files in the output directory"""
        # check rom's exist
        rom_dict = self.data.linear.linear_system.uvlm.rom
        if rom_dict is None:
            return None

        for rom_name, rom_class in rom_dict.items():
            rom_class.save_reduced_order_bases(self.base_name + f'_{rom_name.lower()}_aerorob.h5')

    def save_structural_modal_matrix(self):
        if not self.data.linear.linear_system.beam.sys.modal:
            return None

        self.data.linear.linear_system.beam.save_reduced_order_bases(self.base_name + f'_modal_structrob.h5')

    def save_state_space(self):
        self.data.linear.linear_system.ss.save(self.base_name + '_statespace.h5')

    def save_structural_matrices(self):
        self.data.linear.linear_system.beam.ss.save(self.base_name + '_beamstatespace.h5')
        self.data.linear.linear_system.beam.save_structural_matrices(self.base_name + '_struct_matrices.h5')

    def save_aero_state_space(self):
        self.data.linear.linear_system.uvlm.ss.save(self.base_name + '_aerostatespace.h5')

    @property
    def base_name(self):
        return self.folder + '/save_pmor_data/' + self.case_name

