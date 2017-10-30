import ctypes as ct

import numpy as np

import sharpy.aero.utils.mapping as mapping
import sharpy.utils.cout_utils as cout
import sharpy.utils.solver_interface as solver_interface
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra


@solver
class StaticCoupled(BaseSolver):
    solver_id = 'StaticCoupled'

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

        self.settings_types['max_iter'] = 'int'
        self.settings_default['max_iter'] = 100

        self.settings_types['n_load_steps'] = 'int'
        self.settings_default['n_load_steps'] = 5

        self.settings_types['tolerance'] = 'float'
        self.settings_default['tolerance'] = 1e-5

        self.settings_types['relaxation_factor'] = 'float'
        self.settings_default['relaxation_factor'] = 0.

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

    def increase_ts(self):
        self.data.ts += 1
        self.structural_solver.next_step()
        self.aero_solver.next_step()

    def run(self):
        cout.cout_wrap('Running static coupled solver...', 1)

        for i_step in range(self.settings['n_load_steps'].value):
            # load step coefficient
            if not self.settings['n_load_steps'].value == 0:
                load_step_multiplier = (i_step + 1.0)/self.settings['n_load_steps'].value
            else:
                load_step_multiplier = 1.0

            # new storage every load step
            if i_step > 0:
                self.increase_ts()

            for i_iter in range(self.settings['max_iter'].value):
                cout.cout_wrap('i_step: %u, i_iter: %u' % (i_step, i_iter))

                # run aero
                self.data = self.aero_solver.run()

                print('Beam last node: ')
                print(self.data.structure.timestep_info[self.data.ts].pos[20, :])
                print(self.data.structure.timestep_info[self.data.ts].pos[21, :])

                # map force
                struct_forces = mapping.aero2struct_force_mapping(
                    self.data.aero.timestep_info[self.data.ts].forces,
                    self.data.aero.struct2aero_mapping,
                    self.data.aero.timestep_info[self.data.ts].zeta,
                    self.data.structure.timestep_info[self.data.ts].pos,
                    self.data.structure.timestep_info[self.data.ts].psi,
                    self.data.structure.node_master_elem,
                    self.data.structure.master,
                    algebra.quat2rot(self.data.structure.timestep_info[self.data.ts].quat))

                if not self.settings['relaxation_factor'].value == 0.:
                    if i_iter == 0:
                        self.previous_force = struct_forces.copy()

                    temp = struct_forces.copy()
                    struct_forces = ((1.0 - self.settings['relaxation_factor'].value)*struct_forces +
                                     self.settings['relaxation_factor'].value*self.previous_force)
                    self.previous_force = temp

                # copy force in beam
                temp1 = load_step_multiplier*struct_forces
                self.data.structure.timestep_info[self.data.ts].steady_applied_forces = temp1.astype(dtype=ct.c_double,
                                                                                                     order='F')

                # update gravity direction
                # TODO

                # run beam
                self.data = self.structural_solver.run()

                # update grid
                self.aero_solver.update_step()

                # convergence
                if self.convergence(i_iter, i_step):
                    break


        # for i_step in range(self.settings['n_load_steps']):
        #     coeff = ct.c_double((i_step + 1.0)/(self.settings['n_load_steps']))
        #     for i_iter in range(self.settings['max_iter']):
        #         cout.cout_wrap('Iter: %u, step: %u' % (i_iter, i_step), 2)
        #
        #         self.aero_solver.initialise(self.data, update_flightcon=False, quiet=True)
        #         self.data = self.aero_solver.run()
        #
        #         struct_forces = mapping.aero2struct_force_mapping(
        #             self.data.grid.timestep_info[self.ts].forces,
        #             self.data.grid.struct2aero_mapping,
        #             self.data.grid.timestep_info[self.ts].zeta,
        #             self.data.beam.timestep_info[self.ts].pos_def,
        #             self.data.beam.timestep_info[self.ts].psi_def,
        #             self.data.beam.node_master_elem,
        #             self.data.beam.master,
        #             self.data.grid.inertial2aero)
        #
        #         if self.relaxation_factor > 0.0:
        #             struct_forces = ((1.0 - self.relaxation_factor)*struct_forces +
        #                             self.relaxation_factor*self.previous_forces)
        #
        #         self.previous_forces = struct_forces
        #         self.data.beam.update_forces(struct_forces)
        #         self.structural_solver.initialise(self.data)
        #         self.data = self.structural_solver.run(coeff)
        #
        #         if self.convergence(i_iter):
        #             self.data.flightconditions['FlightCon']['u_inf'] = self.original_u_inf
        #             self.aero_solver.initialise(self.data, quiet=True)
        #             self.data = self.aero_solver.run()
        #             if i_step == self.settings['n_load_steps'] - 1:
        #                 cout.cout_wrap('Converged in %u iterations' % (i_iter + 1), 2)
        #             break

        cout.cout_wrap('...Finished', 1)
        return self.data

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

