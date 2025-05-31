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
    solver_classification = 'other'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    def __init__(self):
        self.data = None
        self.settings = None
        self.file_name = ''
        self.data_dict = dict()

    def initialise(self, data, restart=False):
        self.data = data
        self.read_input_files()
        
        self.settings = data.settings[self.solver_id]
        settings_utils.to_custom_types(self.settings,
                                       self.settings_types,
                                       self.settings_default, options=self.settings_options)


    def read_input_files(self):
        self.file_name = (self.data.case_route +
                          '/' +
                          self.data.case_name +
                          self.file_name)
                                                    
        h5utils.check_file_exists(self.file_name)

        #  read and store the hdf5 file in dictionary
        with h5.File(self.file_name, 'r') as file_handle:
            self.data_dict = h5utils.load_h5_in_dict(file_handle)