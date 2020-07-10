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

    """
    solver_id = 'PickleData'
    solver_classification = 'post-processor'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['folder'] = 'str'
    settings_default['folder'] = './output'
    settings_description['folder'] = 'Folder to output pickle file'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        import sharpy

        self.settings = None
        self.data = None
        self.filename = None
        self.folder = None
        self.caller = None

    def initialise(self, data, custom_settings=None, caller=None):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings.to_custom_types(self.settings,
                                 self.settings_types, self.settings_default)

        # create folder for containing files if necessary
        if not os.path.exists(self.settings['folder']):
            os.makedirs(self.settings['folder'])
        self.folder = self.settings['folder'] + '/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.filename = self.folder + self.data.settings['SHARPy']['case']+'.pkl'
        self.caller = caller

    def run(self, online=False):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)

        return self.data
