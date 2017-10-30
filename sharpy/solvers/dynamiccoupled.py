import ctypes as ct

import numpy as np

import sharpy.aero.utils.mapping as mapping
import sharpy.utils.cout_utils as cout
import sharpy.utils.solver_interface as solver_interface
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra


@solver
class DynamicCoupled(BaseSolver):
    solver_id = 'DynamicCoupled'

    def __init__(self):
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['print_info'] = 'bool'
        self.settings_default['print_info'] = True

        self.settings_types['structural_solver'] = 'str'
        self.settings_default['structural_solver'] = None

        self.settings_types['structural_solver_settings'] = 'dict'
        self.settings_default['structural_solver_settings'] = None

        self.settings_types['aero_solver'] = 'str'
        self.settings_default['aero_solver'] = None

        self.settings_types['aero_solver_settings'] = 'dict'
        self.settings_default['aero_solver_settings'] = None

        self.settings_types['n_time_steps'] = 'int'
        self.settings_default['n_time_steps'] = 100

        self.data = None
        self.settings = None
        self.structural_solver = None
        self.aero_solver = None

        self.previous_force = None

    def initialise(self, data):
        self.data = data
        self.settings = data.settings[self.solver_id]
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        self.structural_solver = solver_interface.initialise_solver(self.settings['structural_solver'])
        self.structural_solver.initialise(self.data, self.settings['structural_solver_settings'])
        self.aero_solver = solver_interface.initialise_solver(self.settings['aero_solver'])
        self.aero_solver.initialise(self.structural_solver.data, self.settings['aero_solver_settings'])
        self.data = self.aero_solver.data

        # if there's data in timestep_info[>0], copy the last one to
        # timestep_info[0] and remove the rest
        self.cleanup_timestep_info()

    def cleanup_timestep_info(self):
        if max(len(self.data.aero.timestep_info), len(self.data.structure.timestep_info)) > 1:
            # copy last info to first
            self.data.aero.timestep_info[0] = self.data.aero.timestep_info[-1]
            self.data.structure.timestep_info[0] = self.data.structure.timestep_info[-1]
            # delete all the rest
            for i in range(1, len(self.data.aero.timestep_info)):
                del self.data.aero.timestep_info[i]
            for i in range(1, len(self.data.structure.timestep_info)):
                del self.data.structure.timestep_info[i]

        self.data.ts = 1

    def increase_ts(self):
        self.structural_solver.next_step()
        self.aero_solver.next_step()

    def run(self):
        # dynamic simulations start at tstep == 1, 0 is reserved for the initial state
        for self.data.ts in range(1, self.settings['n_time_steps'].value + 1):
            self.increase_ts()

            # run aero
            self.data = self.aero_solver.run()

            # map forces
            self.map_forces()

            # run beam
            self.data = self.structural_solver.run()

            # update grid
            # todo pos_dot and psi_dot calculation
            self.aero_solver.update_step()

        cout.cout_wrap('...Finished', 1)
        return self.data

    def map_forces(self):
        struct_forces = mapping.aero2struct_force_mapping(
            self.data.aero.timestep_info[self.data.ts].forces,
            self.data.aero.struct2aero_mapping,
            self.data.aero.timestep_info[self.data.ts].zeta,
            self.data.structure.timestep_info[self.data.ts].pos,
            self.data.structure.timestep_info[self.data.ts].psi,
            self.data.structure.node_master_elem,
            self.data.structure.master,
            algebra.quat2rot(self.data.structure.timestep_info[self.data.ts].quat).T)
        dynamic_struct_forces = mapping.aero2struct_force_mapping(
            self.data.aero.timestep_info[self.data.ts].dynamic_forces,
            self.data.aero.struct2aero_mapping,
            self.data.aero.timestep_info[self.data.ts].zeta,
            self.data.structure.timestep_info[self.data.ts].pos,
            self.data.structure.timestep_info[self.data.ts].psi,
            self.data.structure.node_master_elem,
            self.data.structure.master,
            algebra.quat2rot(self.data.structure.timestep_info[self.data.ts].quat).T)

        self.data.structure.timestep_info[self.data.ts].steady_applied_forces = (
            (struct_forces + self.data.structure.ini_info.steady_applied_forces).astype(dtype=ct.c_double, order='F'))
        self.data.structure.timestep_info[self.data.ts].unsteady_applied_forces = (
            dynamic_struct_forces.astype(dtype=ct.c_double, order='F'))

    def convergence(self, i_iter, i_step):
        if i_iter == self.settings['max_iter'].value - 1:
            cout.cout_wrap('StaticCoupled did not converge!', 0)
            quit(-1)

        return_value = None
        if i_iter == 0:
            self.initial_residual = np.linalg.norm(self.data.structure.timestep_info[self.data.ts].pos)
            self.previous_residual = self.initial_residual
            self.current_residual = self.initial_residual
            return False

        self.current_residual = np.linalg.norm(self.data.structure.timestep_info[self.data.ts].pos)
        cout.cout_wrap('Res = %8e' % (np.abs(self.current_residual - self.previous_residual)/self.previous_residual), 2)

        if return_value is None:
            if np.abs(self.current_residual - self.previous_residual)/self.initial_residual < self.settings['tolerance'].value:
                return_value = True
            else:
                self.previous_residual = self.current_residual
                return_value = False

        if return_value is None:
            return_value = False

        return return_value

