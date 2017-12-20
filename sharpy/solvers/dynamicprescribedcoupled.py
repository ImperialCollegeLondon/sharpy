import ctypes as ct

import numpy as np

import sharpy.aero.utils.mapping as mapping
import sharpy.utils.cout_utils as cout
import sharpy.utils.solver_interface as solver_interface
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra


@solver
class DynamicPrescribedCoupled(BaseSolver):
    solver_id = 'DynamicPrescribedCoupled'

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

        self.settings_types['dt'] = 'float'
        self.settings_default['dt'] = 0.05

        self.settings_types['structural_substeps'] = 'int'
        self.settings_default['structural_substeps'] = 1

        self.data = None
        self.settings = None
        self.structural_solver = None
        self.aero_solver = None

        self.previous_force = None

        self.dt = 0.

    def initialise(self, data):
        self.data = data
        self.settings = data.settings[self.solver_id]
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)
        self.dt = self.settings['dt']

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
            while len(self.data.aero.timestep_info) - 1:
                del self.data.aero.timestep_info[-1]
            while len(self.data.structure.timestep_info) - 1:
                del self.data.structure.timestep_info[-1]

        self.data.ts = 1

    def increase_ts(self):
        self.structural_solver.add_step()
        self.aero_solver.add_step()

    def run(self):
        # dynamic simulations start at tstep == 1, 0 is reserved for the initial state
        for self.data.ts in range(1, self.settings['n_time_steps'].value + 1):
            cout.cout_wrap('\nit = %u' % self.data.ts)
            self.increase_ts()

            # run aero
            self.data = self.aero_solver.run()

            # map forces
            self.map_forces()
            cout.cout_wrap('Max steady force = %f' %
                           self.data.aero.timestep_info[self.data.ts].forces[0].max())
            cout.cout_wrap('Max unsteady force = %f' %
                           self.data.aero.timestep_info[self.data.ts].dynamic_forces[0].max())
            cout.cout_wrap('Tip deformation = %f, %f, %f' %
                           (self.data.structure.timestep_info[self.data.ts].pos[-1, 0],
                            self.data.structure.timestep_info[self.data.ts].pos[-1, 1],
                            self.data.structure.timestep_info[self.data.ts].pos[-1, 2]))
            for i_substep in range(self.settings['structural_substeps'].value):
                cout.cout_wrap('Substep: %u' % i_substep)
                dt = self.settings['dt'].value/self.settings['structural_substeps'].value
                # run structural solver
                self.data = self.structural_solver.run(dt=dt)

            # update orientation in beam and
            # update grid (all done with aero_solver.update_step()
            self.aero_solver.update_grid(self.data.structure)
            self.data.structure.integrate_position(self.data.ts, self.settings['dt'].value)

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
