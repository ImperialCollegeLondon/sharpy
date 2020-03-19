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
class LoadPickleData(BaseSolver):
    """
    This solvers read the SHARPy ``data`` structure previoulsy saved in a pickle file by ``PickleData''
    """
    solver_id = 'LoadPickleData'
    solver_classification = 'loader'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['filename'] = 'str'
    settings_default['filename'] = None
    settings_description['filename'] = 'Pickle file'

    settings_types['num_steps'] = 'int'
    settings_default['num_steps'] = None
    settings_description['num_steps'] = 'Number of time steps for initialisation purposes'
    
    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        import sharpy

        self.settings = None
        self.data = None
        self.filename = None
        self.new_settings = None
        self.num_steps = None

    def initialise(self, data, custom_settings=None):
        self.data = data
        self.new_settings = self.data.settings.copy()
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings.to_custom_types(self.settings,
                                 self.settings_types,
                                 self.settings_default,
                                 no_ctype=True)

        self.filename = self.settings['filename']
        self.num_steps = self.settings['num_steps']

    def run(self, online=False):
        
        fid = open(self.filename, "rb")
        self.data = pickle.load(fid)
        fid.close()
        
        # Overwrite the settings of the previous simulation by the new ones
        self.data.settings = self.new_settings

        # Redefine the arrays that depend on the number of timesteps
        self.data.structure.dyn_dict = dict()
        self.data.structure.dynamic_input = []
        for it in range(self.num_steps):
            self.data.structure.dynamic_input.append(dict())
            # self.dynamic_input[it]['dynamic_forces'] = dyn_dict['dynamic_forces'][it, :, :]
            # self.dynamic_input[it]['for_pos'] = dyn_dict['for_pos'][it, :]

        return self.data
