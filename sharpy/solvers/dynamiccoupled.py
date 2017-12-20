import ctypes as ct

import numpy as np

import sharpy.aero.utils.mapping as mapping
import sharpy.utils.cout_utils as cout
import sharpy.utils.solver_interface as solver_interface
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra
import sharpy.structure.utils.xbeamlib as xbeam


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

        self.predictor = False

    def initialise(self, data):
        self.data = data
        self.settings = data.settings[self.solver_id]
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)
        self.dt = self.settings['dt']
        # if there's data in timestep_info[>0], copy the last one to
        # timestep_info[0] and remove the rest
        self.cleanup_timestep_info()

        self.structural_solver = solver_interface.initialise_solver(self.settings['structural_solver'])
        self.structural_solver.initialise(self.data, self.settings['structural_solver_settings'])
        self.aero_solver = solver_interface.initialise_solver(self.settings['aero_solver'])
        self.aero_solver.initialise(self.structural_solver.data, self.settings['aero_solver_settings'])
        self.data = self.aero_solver.data

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

    def run(self):
        # previous_kstep = self.data.structure.timestep_info[-1].copy()
        # xbeam.cbeam3_solv_state2disp(self.data.structure, previous_kstep)

        # aero_kstep = self.data.aero.timestep_info[-1].copy()
        structural_kstep = self.data.structure.timestep_info[-1].copy()

        # dynamic simulations start at tstep == 1, 0 is reserved for the initial state
        for self.data.ts in range(1, self.settings['n_time_steps'].value + 1):
            cout.cout_wrap('\nit = %u' % self.data.ts)

            aero_kstep = self.data.aero.timestep_info[-1].copy()
            previous_kstep = self.data.structure.timestep_info[-1].copy()
            structural_kstep = self.data.structure.timestep_info[-1].copy()
            # structural_predictor(self.data.structure, structural_kstep, 1*self.settings['dt'].value)

            for k in range(0*self.settings['fsi_substeps'].value + 1):
                if k == self.settings['fsi_substeps'].value:
                    cout.cout_wrap('The FSI solver did not converge!!!')
                    break
                    # TODO Raise Exception

                cout.cout_wrap(str(k))
                # # generate new grid (already rotated)
                # self.aero_solver.update_custom_grid(structural_kstep, aero_kstep)

                # # run the solver
                # self.data = self.aero_solver.run(aero_kstep,
                                                 # structural_kstep,
                                                 # self.data.aero.timestep_info[-1],
                                                 # convect_wake=False)

                structural_kstep = self.data.structure.timestep_info[-1].copy()
                # structural_predictor(self.data.structure, structural_kstep, 1.*self.settings['dt'].value)

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

                print(str(np.log10(np.linalg.norm(structural_kstep.q -
                                                  previous_kstep.q)/
                                   np.linalg.norm(previous_kstep.q))))

                # convergence
                if (np.linalg.norm(structural_kstep.q - previous_kstep.q)/np.linalg.norm(previous_kstep.q) <
                    self.settings['fsi_tolerance'].value)\
                        and \
                        k > self.settings['minimum_steps'].value - 1:
                    break

                # relaxation
                relax(self.data.structure, structural_kstep, previous_kstep, self.relaxation_factor(k))
                # copy for next iteration
                previous_kstep = structural_kstep.copy()

            # allocate and copy previous timestep, copying steady and unsteady forces from input
            self.structural_solver.add_step()
            self.data.structure.timestep_info[-1] = structural_kstep.copy()
            # structural_predictor(self.data.structure, self.data.structure.timestep_info[-1], self.settings['dt'].value)
            # self.map_forces(aero_kstep,
            #                 self.data.structure.timestep_info[-1],
            #                 0.0)
            # self.data = self.structural_solver.run(structural_step=self.data.structure.timestep_info[-1])
            self.data.structure.integrate_position(self.data.ts, self.settings['dt'].value)

            self.aero_solver.add_step()
            # self.data.aero.timestep_info[-1] = aero_kstep.copy()
            # self.aero_solver.update_custom_grid(self.data.structure.timestep_info[-1],
                                                # self.data.aero.timestep_info[-1])
            # # run the solver
            # self.data = self.aero_solver.run(self.data.aero.timestep_info[-1],
                                             # self.data.structure.timestep_info[-1],
                                             # self.data.aero.timestep_info[-2],
                                             # convect_wake=True)

            print('Time = %f' % (self.data.ts*self.settings['dt'].value))
            print('FoR acc in inertial:')
            print(np.dot(
                self.data.structure.timestep_info[self.data.ts].cga(),
                self.data.structure.timestep_info[self.data.ts].for_acc[0:3]))
            print(np.linalg.norm(self.data.structure.timestep_info[self.data.ts].for_acc[0:3]))

        cout.cout_wrap('...Finished', 1)
        return self.data

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

        # DEBUG
        # cout.cout_wrap('NO aero dynamic forces')

        # prescribed forces + aero forces
        structural_kstep.steady_applied_forces = (
            (struct_forces + self.data.structure.ini_info.steady_applied_forces).
            astype(dtype=ct.c_double, order='F', copy=True))
        structural_kstep.unsteady_applied_forces = (
            (dynamic_struct_forces + self.data.structure.dynamic_input[max(self.data.ts - 1, 0)]['dynamic_forces']).
            astype(dtype=ct.c_double, order='F', copy=True))

    def relaxation_factor(self, k):
        initial = self.settings['relaxation_factor'].value
        if not self.settings['dynamic_relaxation'].value:
            return initial

        final = self.settings['final_relaxation_factor'].value
        if k >= self.settings['relaxation_steps'].value:
            return final

        value = initial + (final - initial)/self.settings['relaxation_steps'].value*k
        return value


def relax(beam, timestep, previous_timestep, coeff):
    from sharpy.structure.utils.xbeamlib import xbeam_solv_state2disp
    numdof = beam.num_dof.value
    if coeff > 0.0:
        # timestep.q[0:numdof] = (1.0 - coeff)*timestep.q[0:numdof] + coeff*previous_timestep.q[0:numdof]
        timestep.q = (1.0 - coeff)*timestep.q + coeff*previous_timestep.q
        # timestep.dqdt[0:numdof] = (1.0 - coeff)*timestep.dqdt[0:numdof] + coeff*previous_timestep.dqdt[0:numdof]
        # timestep.dqddt[0:numdof] = (1.0 - coeff)*timestep.dqddt[0:numdof] + coeff*previous_timestep.dqddt[0:numdof]
        timestep.dqdt = (1.0 - coeff)*timestep.dqdt + coeff*previous_timestep.dqdt
        timestep.dqddt = (1.0 - coeff)*timestep.dqddt + coeff*previous_timestep.dqddt
        # timestep.dqdt[numdof:numdof + 6] = (1.0 - coeff)*timestep.dqdt[numdof:numdof + 6] + coeff*previous_timestep.dqdt[numdof:numdof + 6]
        # timestep.dqddt[numdof:numdof + 6] = (1.0 - coeff)*timestep.dqddt[numdof:numdof + 6] + coeff*previous_timestep.dqddt[numdof:numdof + 6]

        normalise_quaternion(timestep)
        xbeam_solv_state2disp(beam, timestep)


def relax_forces(timestep, previous_timestep, coeff):
    timestep.steady_applied_forces = ((1.0 - coeff)*timestep.steady_applied_forces + coeff*previous_timestep.steady_applied_forces).astype(dtype=ct.c_double, order='F')
    timestep.unsteady_applied_forces = ((1.0 - coeff)*timestep.unsteady_applied_forces + coeff*previous_timestep.unsteady_applied_forces).astype(dtype=ct.c_double, order='F')


def structural_predictor(structure, timestep, dt):
    from sharpy.structure.utils.xbeamlib import xbeam_solv_state2disp

    dt = dt
    # Q = Q + dt*dQdt + 0.5*dt*dt*dQddt
    # dQdt = dQdt + dt*dQddt
    timestep.q += dt*timestep.dqdt + 0.25*dt*dt*timestep.dqddt
    timestep.dqdt += 0.5*dt*timestep.dqddt
    timestep.dqddt.fill(0.0)
    # timestep.dqddt[structure.num_dof.value:] = 0.0
    normalise_quaternion(timestep)
    xbeam_solv_state2disp(structure, timestep)

    timestep.for_vel = timestep.dqdt[structure.num_dof.value:structure.num_dof.value + 6].astype(dtype=ct.c_double, order='F', copy=True)
    timestep.for_acc = timestep.dqddt[structure.num_dof.value:structure.num_dof.value + 6].astype(dtype=ct.c_double, order='F', copy=True)


def normalise_quaternion(tstep):
    tstep.dqdt[-4:] = algebra.unit_vector(tstep.dqdt[-4:])
    tstep.quat = tstep.dqdt[-4:].astype(dtype=ct.c_double, order='F', copy=True)


