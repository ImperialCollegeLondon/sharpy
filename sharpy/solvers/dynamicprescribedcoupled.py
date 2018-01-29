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

        self.settings_types['fsi_substeps'] = 'int'
        self.settings_default['fsi_substeps'] = 70

        self.settings_types['fsi_tolerance'] = 'float'
        self.settings_default['fsi_tolerance'] = 1e-5

        self.settings_types['relaxation_factor'] = 'float'
        self.settings_default['relaxation_factor'] = 0.9

        self.settings_types['final_relaxation_factor'] = 'float'
        self.settings_default['final_relaxation_factor'] = 0.4

        self.settings_types['minimum_steps'] = 'int'
        self.settings_default['minimum_steps'] = 3

        self.settings_types['relaxation_steps'] = 'int'
        self.settings_default['relaxation_steps'] = 2

        self.settings_types['dynamic_relaxation'] = 'bool'
        self.settings_default['dynamic_relaxation'] = True

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

        self.data.ts = 0

    def increase_ts(self):
        self.structural_solver.add_step()
        self.aero_solver.add_step()

    def run(self):
        # TODO this solver could probably be included in dynamic coupled directly
        # NOTE use only for non-coupled problems (structural or aero, but not both)

        structural_kstep = self.data.structure.timestep_info[-1].copy()

        # dynamic simulations start at tstep == 1, 0 is reserved for the initial state
        for self.data.ts in range(1, self.settings['n_time_steps'].value + 1):
            cout.cout_wrap('\nit = %u' % self.data.ts)

            aero_kstep = self.data.aero.timestep_info[-1].copy()
            previous_kstep = self.data.structure.timestep_info[-1].copy()
            ts = len(self.data.structure.timestep_info) - 1
            if ts > 0:
                self.data.structure.timestep_info[ts].for_vel[:] = self.data.structure.dynamic_input[ts - 1]['for_vel']
                self.data.structure.timestep_info[ts].for_acc[:] = self.data.structure.dynamic_input[ts - 1]['for_acc']
                # self.data.structure.timestep_info[ts].unsteady_applied_forces[:] = self.data.structure.dynamic_input[ts - 1]['dynamic_forces']
            structural_kstep = self.data.structure.timestep_info[-1].copy()

            for k in range(self.settings['fsi_substeps'].value + 1):
                if k == self.settings['fsi_substeps'].value and not self.settings['fsi_substeps'] == 0:
                    cout.cout_wrap('The FSI solver did not converge!!!')
                    break
                    # TODO Raise Exception

                cout.cout_wrap(str(k))
                # # generate new grid (already rotated)
                self.aero_solver.update_custom_grid(structural_kstep, aero_kstep)

                # run the solver
                self.data = self.aero_solver.run(aero_kstep,
                                                 structural_kstep,
                                                 self.data.aero.timestep_info[-1],
                                                 convect_wake=False)

                structural_kstep = self.data.structure.timestep_info[-1].copy()

                # map forces
                self.map_forces(aero_kstep,
                                structural_kstep,
                                1.0)

                # relax_forces(structural_kstep, previous_kstep, self.relaxation_factor(k))
                # run structural solver
                self.data = self.structural_solver.run(structural_step=structural_kstep)

                # check for non-convergence
                if not all(np.isfinite(structural_kstep.q)):
                    cout.cout_wrap('***No converged!', 3)
                    break

                print(str(np.log10(np.linalg.norm(structural_kstep.pos -
                                                  previous_kstep.pos)/
                                   np.linalg.norm(previous_kstep.pos))))

                # convergence
                if (np.linalg.norm(structural_kstep.pos - previous_kstep.pos)/np.linalg.norm(previous_kstep.pos) <
                    self.settings['fsi_tolerance'].value) \
                        and \
                        k > self.settings['minimum_steps'].value - 1:
                    break

                # relaxation
                # relax(self.data.structure, structural_kstep, previous_kstep, self.relaxation_factor(k))
                # copy for next iteration
                previous_kstep = structural_kstep.copy()

            # allocate and copy previous timestep, copying steady and unsteady forces from input
            self.structural_solver.add_step()
            self.data.structure.timestep_info[-1] = structural_kstep.copy()
            self.data.structure.integrate_position(self.data.ts, self.settings['dt'].value)

            self.aero_solver.add_step()
            self.data.aero.timestep_info[-1] = aero_kstep.copy()
            self.aero_solver.update_custom_grid(self.data.structure.timestep_info[-1],
                                                self.data.aero.timestep_info[-1])
            # run the solver
            self.data = self.aero_solver.run(self.data.aero.timestep_info[-1],
                                             self.data.structure.timestep_info[-1],
                                             self.data.aero.timestep_info[-2],
                                             convect_wake=True)
        return self.data

        # dynamic simulations start at tstep == 1, 0 is reserved for the initial state
        # for self.data.ts in range(1, self.settings['n_time_steps'].value + 1):
        #     cout.cout_wrap('\nit = %u' % self.data.ts)
        #     self.increase_ts()
        #
        #     # run aero
        #     self.data = self.aero_solver.run()
        #
        #     # map forces
        #     self.map_forces()
        #     cout.cout_wrap('Max steady force = %f' %
        #                    self.data.aero.timestep_info[self.data.ts].forces[0].max())
        #     cout.cout_wrap('Max unsteady force = %f' %
        #                    self.data.aero.timestep_info[self.data.ts].dynamic_forces[0].max())
        #     cout.cout_wrap('Tip deformation = %f, %f, %f' %
        #                    (self.data.structure.timestep_info[self.data.ts].pos[-1, 0],
        #                     self.data.structure.timestep_info[self.data.ts].pos[-1, 1],
        #                     self.data.structure.timestep_info[self.data.ts].pos[-1, 2]))
        #     for i_substep in range(self.settings['structural_substeps'].value):
        #         cout.cout_wrap('Substep: %u' % i_substep)
        #         dt = self.settings['dt'].value/self.settings['structural_substeps'].value
        #         # run structural solver
        #         self.data = self.structural_solver.run(dt=dt)
        #
        #     # update orientation in beam and
        #     # update grid (all done with aero_solver.update_step()
        #     self.aero_solver.update_grid(self.data.structure)
        #     self.data.structure.integrate_position(self.data.ts, self.settings['dt'].value)
        #
        # cout.cout_wrap('...Finished', 1)
        # return self.data

    def map_forces(self, aero_kstep, structural_kstep, unsteady_forces_coeff=1.0):
        # set all forces to 0
        structural_kstep.steady_applied_forces.fill(0.0)
        structural_kstep.unsteady_applied_forces.fill(0.0)

        # aero forces to structural forces
        struct_forces = mapping.aero2struct_force_mapping(
            aero_kstep.forces,
            self.data.aero.struct2aero_mapping,
            aero_kstep.zeta,
            structural_kstep.pos,
            structural_kstep.psi,
            self.data.structure.node_master_elem,
            self.data.structure.master,
            structural_kstep.cag())
        dynamic_struct_forces = unsteady_forces_coeff*mapping.aero2struct_force_mapping(
            aero_kstep.dynamic_forces,
            self.data.aero.struct2aero_mapping,
            aero_kstep.zeta,
            structural_kstep.pos,
            structural_kstep.psi,
            self.data.structure.node_master_elem,
            self.data.structure.master,
            structural_kstep.cag())

        # prescribed forces + aero forces
        structural_kstep.steady_applied_forces = (
            (struct_forces + self.data.structure.ini_info.steady_applied_forces).
                astype(dtype=ct.c_double, order='F', copy=True))
        structural_kstep.unsteady_applied_forces = (
            (dynamic_struct_forces + self.data.structure.dynamic_input[max(self.data.ts - 1, 0)]['dynamic_forces']).
                astype(dtype=ct.c_double, order='F', copy=True))

    # def map_forces(self):
    #     struct_forces = mapping.aero2struct_force_mapping(
    #         self.data.aero.timestep_info[self.data.ts].forces,
    #         self.data.aero.struct2aero_mapping,
    #         self.data.aero.timestep_info[self.data.ts].zeta,
    #         self.data.structure.timestep_info[self.data.ts].pos,
    #         self.data.structure.timestep_info[self.data.ts].psi,
    #         self.data.structure.node_master_elem,
    #         self.data.structure.master,
    #         self.data.structure.timestep_info[self.data.ts].cag())
    #     dynamic_struct_forces = 0*mapping.aero2struct_force_mapping(
    #         self.data.aero.timestep_info[self.data.ts].dynamic_forces,
    #         self.data.aero.struct2aero_mapping,
    #         self.data.aero.timestep_info[self.data.ts].zeta,
    #         self.data.structure.timestep_info[self.data.ts].pos,
    #         self.data.structure.timestep_info[self.data.ts].psi,
    #         self.data.structure.node_master_elem,
    #         self.data.structure.master,
    #         self.data.structure.timestep_info[self.data.ts].cag())
    #
    #     self.data.structure.timestep_info[self.data.ts].steady_applied_forces = (
    #         (struct_forces + self.data.structure.ini_info.steady_applied_forces).astype(dtype=ct.c_double, order='F'))
    #     self.data.structure.timestep_info[self.data.ts].unsteady_applied_forces = (
    #         (dynamic_struct_forces + self.data.structure.dynamic_input[max(self.data.ts - 1, 0)]['dynamic_forces']).
    #             astype(dtype=ct.c_double, order='F', copy=True))
    #     # self.data.structure.timestep_info[self.data.ts].unsteady_applied_forces = (
    #     #     dynamic_struct_forces.astype(dtype=ct.c_double, order='F'))
