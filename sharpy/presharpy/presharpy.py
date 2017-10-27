"""The main class for preSHARPy.
"""
import configparser

import sharpy.utils.cout_utils as cout
from sharpy.utils.solver_interface import solver, dict_of_solvers
import sharpy.utils.settings as settings
import sharpy.utils.exceptions as exceptions


@solver
class PreSharpy(object):
    solver_id = 'PreSharpy'

    def __init__(self, in_settings):
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['flow'] = 'list(str)'
        self.settings_default['flow'] = None

        self.settings_types['case'] = 'str'
        self.settings_default['case'] = 'default_case_name'

        self.settings_types['route'] = 'str'
        self.settings_default['route'] = None

        self.settings_types['write_log'] = 'bool'
        self.settings_default['write_log'] = False

        self.settings_types['log_folder'] = 'str'
        self.settings_default['log_folder'] = ''

        self.settings_types['log_file'] = 'str'
        self.settings_default['log_file'] = 'log'

        self.settings = in_settings
        self.settings['SHARPy']['flow'] = self.settings['SHARPy']['flow']
        settings.to_custom_types(self.settings['SHARPy'], self.settings_types, self.settings_default)

        cout.cout_wrap.initialise(True, self.settings['SHARPy']['write_log'],
                                  self.settings['SHARPy']['log_folder'],
                                  self.settings['SHARPy']['log_file'])

        self.case_route = in_settings['SHARPy']['route'] + '/'
        self.case_name = in_settings['SHARPy']['case']
        for solver_name in in_settings['SHARPy']['flow']:
            try:
                dict_of_solvers[solver_name]
            except KeyError:
                exceptions.NotImplementedSolver(solver_name)

    def initialise(self):
        pass

    @staticmethod
    def load_config_file(file_name):
        """This function reads the flight condition and solver input files.

        Args:
            file_name (str): contains the path and file name of the file to be read by the ``configparser``
                reader.

        Returns:
            config (dict): a ``ConfigParser`` object that behaves like a dictionary
        """
        config = configparser.ConfigParser()
        config.read(file_name)
        return config

