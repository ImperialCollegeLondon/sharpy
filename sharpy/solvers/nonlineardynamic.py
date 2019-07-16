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
    """
    Structural solver used for the dynamic simulation of free-flying structures.

    This solver provides an interface to the structural library (``xbeam``) and updates the structural parameters
    for every time step of the simulation.

    This solver is called as part of a standalone structural simulation.

    """
    solver_id = 'NonLinearDynamic'
    solver_classification = 'structural'

    settings_types = dict()
    settings_default = dict()#
    settings_description = dict()

    settings_types['print_info'] = 'bool'
    settings_default['print_info'] = True
    settings_description['print_info'] = 'Print output to screen'

    settings_types['max_iterations'] = 'int'
    settings_default['max_iterations'] = 100
    settings_description['max_iterations'] = 'Sets maximum number of iterations'

    settings_types['num_load_steps'] = 'int'
    settings_default['num_load_steps'] = 5

    settings_types['delta_curved'] = 'float'
    settings_default['delta_curved'] = 1e-5

    settings_types['min_delta'] = 'float'
    settings_default['min_delta'] = 1e-5
    settings_description['min_delta'] = 'Structural solver tolerance'

    settings_types['newmark_damp'] = 'float'
    settings_default['newmark_damp'] = 1e-4
    settings_description['newmark_damp'] = 'Sets the Newmark damping coefficient'

    settings_types['prescribed_motion'] = 'bool'
    settings_default['prescribed_motion'] = None

    settings_types['dt'] = 'float'
    settings_default['dt'] = 0.01
    settings_description['dt'] = 'Time step increment'

    settings_types['num_steps'] = 'int'
    settings_default['num_steps'] = 500

    settings_types['gravity_on'] = 'bool'
    settings_default['gravity_on'] = False
    settings_description['gravity_on'] = 'Flag to include gravitational forces'

    settings_types['gravity'] = 'float'
    settings_default['gravity'] = 9.81
    settings_description['gravity'] = 'Gravitational acceleration'

    settings_types['gravity_dir'] = 'list(float)'
    settings_default['gravity_dir'] = np.array([0, 0, 1])

    # still not implemented in the structural solver.
    settings_types['relaxation_factor'] = 'float'
    settings_default['relaxation_factor'] = 0.3

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
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

