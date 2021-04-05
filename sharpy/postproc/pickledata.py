import os
import pickle

import h5py

import sharpy
import sharpy.utils.cout_utils as cout
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.utils.h5utils as h5utils


# Define basic numerical types
# BasicNumTypes=(float,float32,float64,int,int32,int64,complex)

@solver
class PickleData(BaseSolver):
    """
    This postprocessor writes the SHARPy ``data`` structure in a pickle file, such that classes and
    methods from SHARPy are retained for restarted solutions or further post-processing.

    A pickle is saved to the SHARPy output folder, specified in the settings for SHARPy as ``log_folder``.

    This solver does not have settings, yet it still needs to be included in the `.sharpy` file as an
    empty dictionary.
    """
    solver_id = 'PickleData'
    solver_classification = 'post-processor'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['stride'] = 'int'
    settings_default['stride'] = 1
    settings_description['stride'] = 'Number of steps between the execution calls when run online'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        import sharpy

        self.data = None
        self.filename = None
        self.folder = None
        self.caller = None

    def initialise(self, data, custom_settings=None, caller=None):
        self.data = data

        self.folder = data.output_folder
        self.filename = self.folder + self.data.settings['SHARPy']['case']+'.pkl'
        self.caller = caller

    def run(self, online=False):
        if ((online and (self.data.ts % self.settings['stride'] == 0)) or (not online)):
            with open(self.filename, 'wb') as f:
                pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)

        return self.data
