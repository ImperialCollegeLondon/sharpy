"""
@modified   Alfonso del Carre
"""

import ctypes as ct
import numpy as np

import sharpy.structure.utils.xbeamlib as xbeamlib
from sharpy.utils.settings import str2bool
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.utils.cout_utils as cout


@solver
class NonLinearDynamic(BaseSolver):
    solver_id = 'NonLinearDynamic'

    def __init__(self):
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['print_info'] = 'bool'
        self.settings_default['print_info'] = True

        self.settings_types['max_iterations'] = 'int'
        self.settings_default['max_iterations'] = 100

        self.settings_types['num_load_steps'] = 'int'
        self.settings_default['num_load_steps'] = 5

        self.settings_types['delta_curved'] = 'float'
        self.settings_default['delta_curved'] = 1e-5

        self.settings_types['min_delta'] = 'float'
        self.settings_default['min_delta'] = 1e-5

        self.settings_types['newmark_damp'] = 'float'
        self.settings_default['newmark_damp'] = 1e-4

        self.settings_types['prescribed_motion'] = 'bool'
        self.settings_default['prescribed_motion'] = None

        self.settings_types['dt'] = 'float'
        self.settings_default['dt'] = 0.01

        self.settings_types['num_steps'] = 'int'
        self.settings_default['num_steps'] = 500

        self.settings_types['gravity_on'] = 'bool'
        self.settings_default['gravity_on'] = False

        self.settings_types['gravity'] = 'float'
        self.settings_default['gravity'] = 9.81

        self.settings_types['gravity_dir'] = 'list(float)'
        self.settings_default['gravity_dir'] = np.array([0, 0, 1])

        self.data = None
        self.settings = None

    def initialise(self, data):
        self.data = data
        self.settings = data.settings[self.solver_id]
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        # load info from dyn dictionary
        self.data.structure.add_unsteady_information(self.data.structure.dyn_dict, self.settings['num_steps'].value)

        # allocate timestep_info
        for i in range(self.settings['num_steps'].value):
            self.data.structure.add_timestep(self.data.structure.timestep_info)
            if i>0:
                self.data.structure.timestep_info[i].unsteady_applied_forces[:] = self.data.structure.dynamic_input[i - 1]['dynamic_forces']
            self.data.structure.timestep_info[i].steady_applied_forces[:] = self.data.structure.ini_info.steady_applied_forces


    def run(self):
        prescribed_motion = False
        try:
            prescribed_motion = self.settings['prescribed_motion'].value
        except KeyError:
            pass
        if prescribed_motion is True:
            cout.cout_wrap('Running non linear dynamic solver...', 2)
            # raise NotImplementedError
            xbeamlib.cbeam3_solv_nlndyn(self.data.structure, self.settings)
        else:
            cout.cout_wrap('Running non linear dynamic solver with RB...', 2)
            xbeamlib.xbeam_solv_couplednlndyn(self.data.structure, self.settings)

        self.data.ts = self.settings['num_steps'].value
        cout.cout_wrap('...Finished', 2)
        return self.data

