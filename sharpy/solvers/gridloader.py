import h5py as h5
import numpy as np

from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings_utils
import sharpy.utils.h5utils as h5utils


@solver
class GridLoader(BaseSolver):
    """
    ``GridLoader`` class, inherited from ``BaseSolver``

    Parent class for Aerogridloader and Nonliftingbodygridloader. Both classes
    generate aerodynamic grids based on the input data

    Args:
        data (PreSharpy): ``ProblemData`` class structure

    Attributes:
        settings (dict): Name-value pair of the settings employed by the aerodynamic solver
        settings_types (dict): Acceptable types for the values in ``settings``
        settings_default (dict): Name-value pair of default values for the aerodynamic settings
        data (ProblemData): class structure
        afile_name (str): name of the HDF5 file, e.g. ``.aero.h5``
        aero: empty attribute
        data_dict (dict): key-value pairs of aerodynamic data

    """
    solver_id = 'GridLoader'
    solver_classification = 'loader'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    def __init__(self):
        self.data = None
        self.settings = None
        self.file_name = ''
        # storage of file contents
        self.data_dict = dict()

        # aero storage
        #self.aero = None

    def initialise(self, data):
        self.data = data
        self.settings = data.settings[self.solver_id]

        # init settings
        settings_utils.to_custom_types(self.settings,
                                       self.settings_types,
                                       self.settings_default)

        # read input file
        self.read_files()

    def run(self):
        pass
    
    def read_files(self):
        #  first check that the file exists
        self.file_name = (self.data.case_route +
                          '/' +
                          self.data.case_name +
                          self.file_name)
        #  first check that the file exists
        h5utils.check_file_exists(self.file_name)

        #  read and store the hdf5 file
        with h5.File(self.file_name, 'r') as file_handle:
            # store files in dictionary
            self.data_dict = h5utils.load_h5_in_dict(file_handle)