from sharpy.utils.solver_interface import solver, BaseSolver, initialise_solver
import sharpy.utils.settings as settings
import configobj
import os
import sharpy.utils.cout_utils as cout


@solver
class SaveParametricCase(BaseSolver):
    """
    SaveParametricCase is a post-processor that creates a ConfigParser text file called
    ``<sharpy_case_name>.pmor.sharpy`` that contains information on certain simulation parameters. It is useful as
    a record keeper if you are doing a parametric study and for parametric model interpolation.


    If the setting ``save_case`` is selected and the post processor :class:`~sharpy.solvers.pickledata.PickleData`
    is not present in the SHARPy flow, this solver will pickle the data to the path given in the ``folder`` setting.

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
    settings_default['save_case'] = True
    settings_description['save_case'] = 'Save a .pkl of the SHARPy case. Required for PMOR.'

    settings_types['parameters'] = 'dict'
    settings_default['parameters'] = None
    settings_description['parameters'] = 'Dictionary containing the chosen simulation parameters and their values.'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.data = None
        self.settings = None
        self.folder = None

    def initialise(self, data, custom_settings=None):
        self.data = data

        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings

        settings.to_custom_types(self.settings,
                                 self.settings_types,
                                 self.settings_default,
                                 no_ctype=True)

        # create folder for containing files if necessary
        self.folder = data.output_folder

    def run(self):

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
            pickle_solver = initialise_solver('PickleData')
            pickle_solver.initialise(self.data)
            self.data = pickle_solver.run()
            sim_info['path_to_data'] = os.path.abspath(self.folder)
        else:
            sim_info['path_to_data'] = os.path.abspath(self.folder)

        config['sim_info'] = sim_info
        config.write()

        return self.data
