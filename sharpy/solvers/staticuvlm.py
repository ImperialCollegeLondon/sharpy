import ctypes as ct

import sharpy.aero.models.aerogrid as aerogrid
import sharpy.aero.utils.mapping as mapping
import sharpy.aero.utils.utils as aero_utils
import sharpy.aero.utils.uvlmlib as uvlmlib
import sharpy.utils.cout_utils as cout
import sharpy.utils.settings as settings
from sharpy.utils.settings import str2bool
from sharpy.utils.solver_interface import solver, BaseSolver

@solver
class StaticUvlm(BaseSolver):
    solver_id = 'StaticUvlm'

    def __init__(self):
        # settings list
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['print_info'] = 'bool'
        self.settings_default['print_info'] = True

        self.settings_types['horseshoe'] = 'bool'
        self.settings_default['horseshoe'] = False

        self.settings_types['num_cores'] = 'int'
        self.settings_default['num_cores'] = 0

        self.settings_types['n_rollup'] = 'int'
        self.settings_default['n_rollup'] = 1

        self.settings_types['rollup_dt'] = 'float'
        self.settings_default['rollup_dt'] = 0.1

        self.settings_types['rollup_aic_refresh'] = 'int'
        self.settings_default['rollup_aic_refresh'] = 1

        self.settings_types['rollup_tolerance'] = 'float'
        self.settings_default['rollup_tolerance'] = 1e-4


        self.ts = 0
        self.data = None
        self.settings = None

        pass

    def initialise(self, data):
        self.ts = 0
        self.data = data
        self.settings = data.settings[self.solver_id]
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

    def update(self):
        pass

    def run(self):
        cout.cout_wrap('Running static UVLM solver...', 1)
        uvlmlib.vlm_solver(self.data.aero.timestep_info[self.ts],
                           self.data.flightconditions,
                           self.settings)

        cout.cout_wrap('...Finished', 1)
        return self.data
