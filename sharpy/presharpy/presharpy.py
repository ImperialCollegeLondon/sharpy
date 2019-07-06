
import configparser

import sharpy.utils.cout_utils as cout
from sharpy.utils.solver_interface import solver, dict_of_solvers
import sharpy.utils.settings as settings
import sharpy.utils.exceptions as exceptions


@solver
class PreSharpy(object):
    """
    The main class for preSHARPy.

    PreSharpy objects contain information on the output of the problem.
    The ``PreSharpy`` ``settings`` attributes are obtained from the ``.solver.txt`` file, under ``[SHARPy]``.

    Args:
        in_settings: settings file, from which ``[SHARPy]`` values are extracted

    Attributes:
        settings (dict): name-value pairs of solution settings types.
            See Notes for acceptable combinations.
        settings_types (dict): accepted types for values in ``settings``
        settings_default (dict): default values for ``settings``, should none be provided.
        ts (int): solution time step
        case_route (str): route to case folder
        case_name (str): name of the case

    Notes:

        The following key-value pairs are acceptable ``settings`` for the ``PreSharpy`` class:

        ================  =============  ===============================================  =====================
        Key               Type           Description                                      Default
        ================  =============  ===============================================  =====================
        ``flow``          ``list(str)``  List of solvers to run in the appropriate order  ``None``
        ``case``          ``str``        Case name                                        ``default_case_name``
        ``route``         ``str``        Route to folder                                  ``None``
        ``write_screen``  ``bool``       Write output to terminal screen                  ``True``
        ``write_log``     ``bool``       Write log file to output folder                  ``False``
        ``log_folder``    ``str``        Folder to write log file within ``/output``      ``None``
        ``log_file``      ``str``        Log file name                                    ``log``
        ================  =============  ===============================================  =====================
    """
    solver_id = 'PreSharpy'


    def __init__(self, in_settings):
        self.ts = 0

        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['flow'] = 'list(str)'
        self.settings_default['flow'] = None

        self.settings_types['case'] = 'str'
        self.settings_default['case'] = 'default_case_name'

        self.settings_types['route'] = 'str'
        self.settings_default['route'] = None

        self.settings_types['write_screen'] = 'bool'
        self.settings_default['write_screen'] = True

        self.settings_types['write_log'] = 'bool'
        self.settings_default['write_log'] = False

        self.settings_types['log_folder'] = 'str'
        self.settings_default['log_folder'] = ''

        self.settings_types['log_file'] = 'str'
        self.settings_default['log_file'] = 'log'

        self.settings = in_settings
        self.settings['SHARPy']['flow'] = self.settings['SHARPy']['flow']
        settings.to_custom_types(self.settings['SHARPy'], self.settings_types, self.settings_default)

        cout.cout_wrap.initialise(self.settings['SHARPy']['write_screen'],
                                  self.settings['SHARPy']['write_log'],
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

    def update_settings(self, new_settings):
        self.settings = new_settings
        self.settings['SHARPy']['flow'] = self.settings['SHARPy']['flow']
        settings.to_custom_types(self.settings['SHARPy'], self.settings_types, self.settings_default)

        cout.cout_wrap.initialise(self.settings['SHARPy']['write_screen'],
                                  self.settings['SHARPy']['write_log'],
                                  self.settings['SHARPy']['log_folder'],
                                  self.settings['SHARPy']['log_file'])

    @staticmethod
    def load_config_file(file_name):
        """Reads the flight condition and solver input files.

        Args:
            file_name (str): contains the path and file name of the file to be read by the ``configparser``
                reader.

        Returns:
            ``ConfigParser`` object that behaves like a dictionary
        """
        config = configparser.ConfigParser()
        config.read(file_name)
        return config

