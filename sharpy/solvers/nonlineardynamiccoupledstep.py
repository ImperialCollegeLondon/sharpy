"""
@modified   Alfonso del Carre
"""

import ctypes as ct
import numpy as np

import sharpy.structure.utils.xbeamlib as xbeamlib
from sharpy.utils.settings import str2bool
from sharpy.utils.solver_interface import solver, BaseSolver, solver_from_string
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra
import sharpy.utils.cout_utils as cout


_BaseStructural = solver_from_string('_BaseStructural')

@solver
class NonLinearDynamicCoupledStep(_BaseStructural):
    """
    Structural solver used for the dynamic simulation of free-flying structures.

    This solver provides an interface to the structural library (``xbeam``) and updates the structural parameters
    for every k-th step in the FSI iteration.

    This solver can be called as part of a standalone structural simulation or as the structural solver of a coupled
    aeroelastic simulation.

    """
    solver_id = 'NonLinearDynamicCoupledStep'
    solver_classification = 'structural'

    settings_types = _BaseStructural.settings_types.copy()
    settings_default = _BaseStructural.settings_default.copy()
    settings_description = _BaseStructural.settings_description.copy()

    settings_types['balancing'] = 'bool'
    settings_default['balancing'] = False

    # initial speed direction is given in inertial FOR!!!
    settings_types['initial_velocity_direction'] = 'list(float)'
    settings_default['initial_velocity_direction'] = [-1.0, 0.0, 0.0]
    settings_description['initial_velocity_direction'] = 'Initial velocity of the reference node given in the inertial FOR'

    settings_types['initial_velocity'] = 'float'
    settings_default['initial_velocity'] = 0
    settings_description['initial_velocity'] = 'Initial velocity magnitude of the reference node'

    settings_types['relaxation_factor'] = 'float'
    settings_default['relaxation_factor'] = 0.3
    settings_description['relaxation factor'] = 'Relaxation factor'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.data = None
        self.settings = None

    def initialise(self, data, custom_settings=None):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        # load info from dyn dictionary
        self.data.structure.add_unsteady_information(self.data.structure.dyn_dict, self.settings['num_steps'].value)

        # add initial speed to RBM
        if self.settings['initial_velocity']:
            new_direction = np.dot(self.data.structure.timestep_info[-1].cag(),
                                   self.settings['initial_velocity_direction'])
            self.data.structure.timestep_info[-1].for_vel[0:3] = new_direction*self.settings['initial_velocity']

        # generate q, dqdt and dqddt
        xbeamlib.xbeam_solv_disp2state(self.data.structure, self.data.structure.timestep_info[-1])

    def run(self, structural_step, previous_structural_step=None, dt=None):
        if dt is None:
            dt = self.settings['dt']

        xbeamlib.xbeam_step_couplednlndyn(self.data.structure,
                                          self.settings,
                                          self.data.ts,
                                          structural_step,
                                          dt=dt)
        self.extract_resultants(structural_step)
        self.data.structure.integrate_position(structural_step, dt)

        return self.data

    def add_step(self):
        self.data.structure.next_step()

    def next_step(self):
        pass

    def extract_resultants(self, tstep=None):
        if tstep is None:
            tstep = self.data.structure.timestep_info[self.data.ts]
        steady, unsteady, grav = tstep.extract_resultants(self.data.structure, force_type=['steady', 'unsteady', 'grav'])
        totals = np.sum(steady + unsteady + grav)
        return totals[0:3], totals[3:6]

