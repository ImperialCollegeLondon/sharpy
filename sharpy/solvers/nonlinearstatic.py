import numpy as np

import sharpy.structure.utils.xbeamlib as xbeamlib
import sharpy.utils.cout_utils as cout
import sharpy.utils.settings as settings
from sharpy.utils.solver_interface import solver, BaseSolver


@solver
class NonLinearStatic(BaseSolver):
    """
    Structural solver used for the static simulation of free-flying structures.

    This solver provides an interface to the structural library (``xbeam``) and updates the structural parameters
    for every k-th step of the FSI iteration.

    This solver can be called as part of a standalone structural simulation or as the structural solver of a coupled
    static aeroelastic simulation.

    """
    solver_id = 'NonLinearStatic'
    solver_classification = 'structural'

    # settings list
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

    settings_types['gravity_on'] = 'bool'
    settings_default['gravity_on'] = False
    settings_description['gravity_on'] = 'Flag to include gravitational forces'

    settings_types['gravity'] = 'float'
    settings_default['gravity'] = 9.81
    settings_description['gravity'] = 'Gravitational acceleration'

    settings_types['min_delta'] = 'float'
    settings_default['min_delta'] = 1e-7
    settings_description['min_delta'] = 'Structural solver tolerance'

    settings_types['initial_position'] = 'list(float)'
    settings_default['initial_position'] = np.array([0.0, 0.0, 0.0])

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

    def run(self):
        self.data.structure.timestep_info[self.data.ts].for_pos[0:3] = self.settings['initial_position']
        xbeamlib.cbeam3_solv_nlnstatic(self.data.structure, self.settings, self.data.ts)
        self.extract_resultants()
        return self.data

    def next_step(self):
        self.data.structure.next_step()

    def extract_resultants(self, tstep=None):
        if tstep is None:
            tstep = self.data.structure.timestep_info[self.data.ts]
        applied_forces = self.data.structure.nodal_b_for_2_a_for(tstep.steady_applied_forces,
                                                                 tstep)

        applied_forces_copy = applied_forces.copy()
        gravity_forces_copy = tstep.gravity_forces.copy()
        for i_node in range(self.data.structure.num_node):
            applied_forces_copy[i_node, 3:6] += np.cross(tstep.pos[i_node, :],
                                                         applied_forces_copy[i_node, 0:3])
            gravity_forces_copy[i_node, 3:6] += np.cross(tstep.pos[i_node, :],
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


