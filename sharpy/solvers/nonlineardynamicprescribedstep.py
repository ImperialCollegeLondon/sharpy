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
class NonLinearDynamicPrescribedStep(BaseSolver):
    """
    Structural solver used for the dynamic simulation of clamped structures or those subject to a prescribed motion.

    This solver provides an interface to the structural library (``xbeam``) and updates the structural parameters
    for every k-th step in the FSI iteration.

    This solver can be called as part of a standalone structural simulation or as the structural solver of a coupled
    aeroelastic simulation.
    
    """

    solver_id = 'NonLinearDynamicPrescribedStep'
    solver_classification = 'structural'

    settings_types = dict()
    settings_default = dict()
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


    def run(self, structural_step=None, dt=None):
        if self.data.ts > 0:
            try:
                structural_step.for_vel[:] = self.data.structure.dynamic_input[self.data.ts - 1]['for_vel']
                structural_step.for_acc[:] = self.data.structure.dynamic_input[self.data.ts - 1]['for_acc']
            except IndexError:
                pass

        xbeamlib.cbeam3_step_nlndyn(self.data.structure,
                                    self.settings,
                                    self.data.ts,
                                    structural_step,
                                    dt=dt)
        self.extract_resultants(structural_step)
        # if self.data.ts > 0:
        #     self.data.structure.integrate_position(self.data.ts, self.settings['dt'].value)
        return self.data

    def add_step(self):
        self.data.structure.next_step()

    def next_step(self):
        pass
        # self.data.structure.next_step()
        # ts = len(self.data.structure.timestep_info) - 1
        # if ts > 0:
        #     self.data.structure.timestep_info[ts].for_vel[:] = self.data.structure.dynamic_input[ts - 1]['for_vel']
        #     self.data.structure.timestep_info[ts].for_acc[:] = self.data.structure.dynamic_input[ts - 1]['for_acc']
        #     self.data.structure.timestep_info[ts].unsteady_applied_forces[:] = self.data.structure.dynamic_input[ts - 1]['dynamic_forces']

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

    def update(self, tstep=None):
        self.create_q_vector(tstep)

    def create_q_vector(self, tstep=None):
        import sharpy.structure.utils.xbeamlib as xb
        if tstep is None:
            tstep = self.data.structure.timestep_info[-1]

        xb.xbeam_solv_disp2state(self.data.structure, tstep)
