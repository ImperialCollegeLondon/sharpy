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
class NonLinearDynamicCoupledStep(BaseSolver):
    solver_id = 'NonLinearDynamicCoupledStep'

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

        self.settings_types['dt'] = 'float'
        self.settings_default['dt'] = 0.01

        self.settings_types['num_steps'] = 'int'
        self.settings_default['num_steps'] = 500

        self.settings_types['gravity_on'] = 'bool'
        self.settings_default['gravity_on'] = False

        self.settings_types['balancing'] = 'bool'
        self.settings_default['balancing'] = False

        self.settings_types['gravity'] = 'float'
        self.settings_default['gravity'] = 9.81

        self.settings_types['structural_substeps'] = 'int'
        self.settings_default['structural_substeps'] = 0

        # initial speed direction is given in inertial FOR!!!
        self.settings_types['initial_velocity_direction'] = 'list(float)'
        self.settings_default['initial_velocity_direction'] = np.array([-1.0, 0.0, 0.0])

        self.settings_types['initial_velocity'] = 'float'
        self.settings_default['initial_velocity'] = 0

        self.settings_types['relaxation_factor'] = 'float'
        self.settings_default['relaxation_factor'] = 0.3

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

    def extract_resultants(self, step=None):
        if step is None:
            step = self.data.structure.timestep_info[-1]
        applied_forces = self.data.structure.nodal_b_for_2_a_for(step.steady_applied_forces,
                                                                 step)

        applied_forces_copy = applied_forces.copy()
        gravity_forces_copy = step.gravity_forces.copy()
        for i_node in range(self.data.structure.num_node):
            applied_forces_copy[i_node, 3:6] += np.cross(step.pos[i_node, :],
                                                         applied_forces_copy[i_node, 0:3])
            gravity_forces_copy[i_node, 3:6] += np.cross(step.pos[i_node, :],
                                                         gravity_forces_copy[i_node, 0:3])

        totals = np.sum(applied_forces_copy + gravity_forces_copy, axis=0)
        return totals[0:3], totals[3:6]

