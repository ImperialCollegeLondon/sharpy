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

        self.settings = in_settings
        self.settings['SHARPy']['flow'] = self.settings['SHARPy']['flow'][0]
        settings.to_custom_types(self.settings['SHARPy'], self.settings_types, self.settings_default)
        self.case_route = in_settings['SHARPy']['route'] + '/'
        self.case_name = in_settings['SHARPy']['case']
        for solver_name in in_settings['SHARPy']['flow']:
            try:
                dict_of_solvers[solver_name]
            except KeyError:
                exceptions.NotImplementedSolver(solver_name)

        # self.initialise()

    def initialise(self):
        pass
        # fem_file_name = self.case_route + '/' + self.case_name + '.fem.h5'
        # if not self.only_structural:
        #     aero_file_name = self.case_route + '/' + self.case_name + '.aero.h5'
        #
        # if not self.steady_problem:
        #     dyn_file_name = self.case_route + '/' + self.case_name + '.dyn.h5'

        # Check if files exist
        # h5utils.check_file_exists(fem_file_name)
        # print('\tThe FEM file exists, it will be loaded.')
        # if not self.only_structural:
        #     h5utils.check_file_exists(aero_file_name)
        #     print('\tThe aero file exists, it will be loaded.')
        # if not self.steady_problem:
        #     h5utils.check_file_exists(dyn_file_name)
        #     print('\tThe dynamic forces file exists, it will be loaded.')

        # Assign handles
        # !!!
        # These handles are stored in the class.
        # That means that they will stay open until
        # we manually close them or the instance of
        # ProblemData is destroyed
        # self.fem_handle = h5.File(fem_file_name, 'r')
        # """h5py.File: .fem.h5 file handle"""
        # if not self.only_structural:
        #     self.aero_handle = h5.File(aero_file_name, 'r')
        #     """h5py.File: .aero.h5 file handle"""
        # if not self.steady_problem:
        #     self.dyn_handle = h5.File(dyn_file_name, 'r')
        #
        # # Store h5 info in dictionaries
        # self.fem_data_dict = (
        #     h5utils.load_h5_in_dict(self.fem_handle))
        # """dict: contains all the input data of the ``FEM`` file stored in a dictionary"""
        # h5utils.check_fem_dict(self.fem_data_dict)
        #
        # if not self.only_structural:
        #     self.aero_data_dict = (
        #         h5utils.load_h5_in_dict(self.aero_handle))
        #     """dict: contains all the input data of the ``aero`` file stored in a dictionary"""
        #     # h5utils.check_aero_dict(self.aero_data_dict)   #TODO
        #
        # if not self.steady_problem:
        #     self.dyn_data_dict = (
        #         h5utils.load_h5_in_dict(self.dyn_handle))
        #
        # cout.cout_wrap('\tDONE')
        # cout.cout_wrap(cout.separator)
        # cout.cout_wrap('Processing fem input and generating beam model...')
        # if not self.steady_problem:
        #     self.beam = beam.Beam(self.fem_data_dict, self.dyn_data_dict)
        # else:
        #     self.beam = beam.Beam(self.fem_data_dict)

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

