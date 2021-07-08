import configparser
import configobj
import os
import sharpy.utils.cout_utils as cout
from sharpy.utils.solver_interface import solver, dict_of_solvers
import sharpy.utils.settings as settings
import sharpy.utils.exceptions as exceptions


@solver
class PreSharpy(object):
    """
    The PreSharpy solver is the main loader solver of SHARPy. It takes the admin-like settings for the simulation,
    including the case name, case route and the list of solvers to run and in which order to run them. This order
    of solvers is referred to, throughout SHARPy, as the ``flow`` setting.

    This is a mandatory solver for all simulations at the start so it is never included in the ``flow`` setting.

    The settings for this solver are parsed through in the configuration file under the header ``SHARPy``. I.e, when
    you are defining the config file for a simulation, the settings for PreSharpy are included as:

    .. code-block:: python

        import configobj
        filename = '<case_route>/<case_name>.sharpy'
        config = configobj.ConfigObj()
        config.filename = filename
        config['SHARPy'] = {'case': '<your SHARPy case name>',  # an example setting
                            # Rest of your settings for the PreSHARPy class
                            }

    """
    solver_id = 'PreSharpy'
    solver_classification = 'loader'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['flow'] = 'list(str)'
    settings_default['flow'] = None
    settings_description['flow'] = "List of the desired solvers' ``solver_id`` to run in sequential order."

    settings_types['case'] = 'str'
    settings_default['case'] = 'default_case_name'
    settings_description['case'] = 'Case name'

    settings_types['route'] = 'str'
    settings_default['route'] = None
    settings_description['route'] = 'Route to case files'

    settings_types['write_screen'] = 'bool'
    settings_default['write_screen'] = True
    settings_description['write_screen'] = 'Display output on terminal screen'

    settings_types['write_log'] = 'bool'
    settings_default['write_log'] = False
    settings_description['write_log'] = 'Write log file'

    settings_types['log_folder'] = 'str'
    settings_default['log_folder'] = './output/'
    settings_description['log_folder'] = 'A folder with the case name will be created at this directory ' \
                                         'containing the SHARPy log and output folders'

    settings_types['log_file'] = 'str'
    settings_default['log_file'] = 'log'
    settings_description['log_file'] = 'Name of the log file'

    settings_types['save_settings'] = 'bool'
    settings_default['save_settings'] = False
    settings_description['save_settings'] = 'Save a copy of the settings to a ``.sharpy`` file in the output ' \
                                            'directory specified in ``log_folder``'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description,
                                       header_line='The following are the settings that the PreSharpy class takes:')

    def __init__(self, in_settings=None):
        self._settings = True
        if in_settings is None:
            # call for documentation only
            self._settings = False

        self.ts = 0

        if self._settings:
            self.settings = in_settings
            self.settings['SHARPy']['flow'] = self.settings['SHARPy']['flow']

            settings.to_custom_types(self.settings['SHARPy'], self.settings_types, self.settings_default)
            self.output_folder = self.settings['SHARPy']['log_folder'] + '/' + self.settings['SHARPy']['case'] + '/'
            if not os.path.isdir(self.output_folder):
                os.makedirs(self.output_folder)

            cout.cout_wrap.initialise(self.settings['SHARPy']['write_screen'],
                                      self.settings['SHARPy']['write_log'],
                                      self.output_folder,
                                      self.settings['SHARPy']['log_file'])

            self.case_route = in_settings['SHARPy']['route'] + '/'
            self.case_name = in_settings['SHARPy']['case']
            for solver_name in in_settings['SHARPy']['flow']:
                try:
                    dict_of_solvers[solver_name]
                except KeyError:
                    exceptions.NotImplementedSolver(solver_name)

            cout.cout_wrap('SHARPy output folder set')
            cout.cout_wrap('\t' + self.output_folder, 1)

            if self.settings['SHARPy']['save_settings']:
                self.save_settings()

    def initialise(self):
        pass

    def update_settings(self, new_settings):
        self.settings = new_settings
        self.settings['SHARPy']['flow'] = self.settings['SHARPy']['flow']
        settings.to_custom_types(self.settings['SHARPy'], self.settings_types, self.settings_default)

        self.output_folder = self.settings['SHARPy']['log_folder'] + '/' + self.settings['SHARPy']['case'] + '/'
        if not os.path.isdir(self.output_folder):
            os.makedirs(self.output_folder)
            
        cout.cout_wrap.initialise(self.settings['SHARPy']['write_screen'],
                                  self.settings['SHARPy']['write_log'],
                                  self.output_folder,
                                  self.settings['SHARPy']['log_file'])

        self.case_route = self.settings['SHARPy']['route'] + '/'
        self.case_name = self.settings['SHARPy']['case']

    def save_settings(self):
        """
        Saves the settings to a ``.sharpy`` config obj file in the output directory.
        """
        out_settings = configobj.ConfigObj()
        for k, v in self.settings.items():
            out_settings[k] = v
        out_settings.filename = self.output_folder + self.settings['SHARPy']['case'] + '.sharpy'
        out_settings.write()

    @staticmethod
    def load_config_file(file_name):
        config = configparser.ConfigParser()
        config.read(file_name)
        return config
