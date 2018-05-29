import ctypes as ct
import numpy as np

import sharpy.utils.algebra as algebra
import sharpy.utils.settings as settings
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.solver_interface as solver_interface
import sharpy.utils.cout_utils as cout


# @solver
# class PrescribedUvlm(BaseSolver):
#     solver_id = 'PrescribedUvlm'
#
#     def __init__(self):
#         # settings list
#         self.settings_types = dict()
#         self.settings_default = dict()
#
#         self.settings_types['print_info'] = 'bool'
#         self.settings_default['print_info'] = True
#
#         self.settings_types['num_cores'] = 'int'
#         self.settings_default['num_cores'] = 0
#
#         self.settings_types['n_time_steps'] = 'int'
#         self.settings_default['n_time_steps'] = 100
#
#         self.settings_types['convection_scheme'] = 'int'
#         self.settings_default['convection_scheme'] = 3
#
#         self.settings_types['steady_n_rollup'] = 'int'
#         self.settings_default['steady_n_rollup'] = 0
#
#         self.settings_types['steady_rollup_tolerance'] = 'float'
#         self.settings_default['steady_rollup_tolerance'] = 1e-4
#
#         self.settings_types['steady_rollup_aic_refresh'] = 'int'
#         self.settings_default['steady_rollup_aic_refresh'] = 1
#
#         self.settings_types['dt'] = 'float'
#         self.settings_default['dt'] = 0.1
#
#         self.settings_types['iterative_solver'] = 'bool'
#         self.settings_default['iterative_solver'] = False
#
#         self.settings_types['iterative_tol'] = 'float'
#         self.settings_default['iterative_tol'] = 1e-4
#
#         self.settings_types['iterative_precond'] = 'bool'
#         self.settings_default['iterative_precond'] = False
#
#         self.settings_types['velocity_field_generator'] = 'str'
#         self.settings_default['velocity_field_generator'] = 'SteadyVelocityField'
#
#         self.settings_types['velocity_field_input'] = 'dict'
#         self.settings_default['velocity_field_input'] = {}
#
#         self.settings_types['rho'] = 'float'
#         self.settings_default['rho'] = 1.225
#
#         self.data = None
#         self.settings = None
#         self.velocity_generator = None
#
#     def initialise(self, data, custom_settings=None):
#         self.data = data
#         if custom_settings is None:
#             self.settings = data.settings[self.solver_id]
#         else:
#             self.settings = custom_settings
#         settings.to_custom_types(self.settings, self.settings_types, self.settings_default)
#
#         self.data.structure.add_unsteady_information(self.data.structure.dyn_dict, self.settings['n_time_steps'].value)
#
#         # generates and rotates the aero grid and rotates the structure
#         self.update_step()
#
#         # init velocity generator
#         velocity_generator_type = gen_interface.generator_from_string(
#             self.settings['velocity_field_generator'])
#         self.velocity_generator = velocity_generator_type()
#         self.velocity_generator.initialise(self.settings['velocity_field_input'])
#
#         self.data.ts = 0
#         # generate uext
#         self.velocity_generator.generate({'zeta': self.data.aero.timestep_info[self.data.ts].zeta,
#                                           'override': True,
#                                           'ts': self.data.ts,
#                                           't': 0.0},
#                                          self.data.aero.timestep_info[self.data.ts].u_ext)
#
#         uvlmlib.uvlm_init(self.data.aero.timestep_info[self.data.ts], self.settings)
#
#     def run(self):
#         for self.data.ts in range(1, self.settings['n_time_steps'].value + 1):
#             cout.cout_wrap('i_iter: ' + str(self.data.ts))
#             self.next_step()
#             t = self.data.ts*self.settings['dt'].value
#             # generate uext
#             self.velocity_generator.generate({'zeta': self.data.aero.timestep_info[self.data.ts].zeta,
#                                               'override': True,
#                                               'ts': self.data.ts,
#                                               't': t},
#                                              self.data.aero.timestep_info[self.data.ts].u_ext)
#             if self.settings['convection_scheme'].value > 1:
#                 # generate uext_star
#                 self.velocity_generator.generate({'zeta': self.data.aero.timestep_info[self.data.ts].zeta_star,
#                                                   'override': True,
#                                                   'ts': self.data.ts,
#                                                   't': t},
#                                                  self.data.aero.timestep_info[self.data.ts].u_ext_star)
#
#             self.data.structure.timestep_info[self.data.ts].for_vel = self.data.structure.dynamic_input[self.data.ts - 1]['for_vel'].astype(ct.c_double)
#
#             uvlmlib.uvlm_solver(self.data.ts,
#                                 self.data.aero.timestep_info[self.data.ts],
#                                 self.data.aero.timestep_info[self.data.ts - 1],
#                                 self.data.structure.timestep_info[self.data.ts],
#                                 self.settings)
#
#             self.data.structure.timestep_info[self.data.ts].for_pos[0:3] = (
#                 self.data.structure.timestep_info[self.data.ts - 1].for_pos[0:3] +
#                 np.dot(self.data.structure.timestep_info[self.data.ts].cga().transpose(),
#                        self.settings['dt'].value*self.data.structure.timestep_info[self.data.ts - 1].for_vel[0:3]))
#             self.data.structure.timestep_info[self.data.ts].for_pos[3:6] = (
#                 self.data.structure.timestep_info[self.data.ts - 1].for_pos[3:6] +
#                 np.dot(self.data.structure.timestep_info[self.data.ts].cga().transpose(),
#                        self.settings['dt'].value*self.data.structure.timestep_info[self.data.ts - 1].for_vel[3:6]))
#
#         return self.data
#
#     def next_step(self):
#         """ Updates de aerogrid based on the info of the step, and increases
#         the self.ts counter """
#         self.data.structure.next_step()
#         self.data.aero.add_timestep()
#         self.update_step()
#
#     def update_step(self, integrate_orientation=True):
#         self.data.aero.generate_zeta(self.data.structure,
#                                      self.data.aero.aero_settings,
#                                      self.data.ts)
#
#         if integrate_orientation:
#             if self.data.ts > 0:
#                 # euler = self.data.structure.dynamic_input[self.data.ts - 1]['for_pos'][3:6]
#                 # euler_rot = algebra.euler2rot(euler)  # this is Cag
#                 # quat = algebra.mat2quat(euler_rot.T)
#                 # TODO need to update orientation
#                 # quat = self.data.structure.timestep_info[self.data.ts - 1].quat
#                 quat = algebra.rotate_quaternion(self.data.structure.timestep_info[self.data.ts].quat,
#                                                  self.data.structure.timestep_info[self.data.ts].for_vel[3:6]*
#                                                  self.settings['dt'])
#             else:
#                 quat = self.data.structure.ini_info.quat.copy()
#         else:
#             quat = self.data.structure.timestep_info[self.data.ts].quat
#
#         quat = algebra.unit_vector(quat)
#         self.data.structure.update_orientation(quat, self.data.ts)  # quat corresponding to Cga
#         self.data.aero.update_orientation(self.data.structure.timestep_info[self.data.ts].quat, self.data.ts)       # quat corresponding to Cga
#


@solver
class PrescribedUvlm(BaseSolver):
    solver_id = 'PrescribedUvlm'

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

        self.settings_types['postprocessors'] = 'list(str)'
        self.settings_default['postprocessors'] = list()

        self.settings_types['postprocessors_settings'] = 'dict'
        self.settings_default['postprocessors_settings'] = dict()

        self.data = None
        self.settings = None
        self.structural_solver = None
        self.aero_solver = None

        self.previous_force = None

        self.dt = 0.
        self.postprocessors = dict()
        self.with_postprocessors = False

    def initialise(self, data):
        self.data = data
        self.settings = data.settings[self.solver_id]
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)
        self.dt = self.settings['dt']

        self.aero_solver = solver_interface.initialise_solver(self.settings['aero_solver'])
        self.aero_solver.initialise(self.data, self.settings['aero_solver_settings'])
        self.data = self.aero_solver.data

        # if there's data in timestep_info[>0], copy the last one to
        # timestep_info[0] and remove the rest
        self.cleanup_timestep_info()

        # initialise postprocessors
        self.postprocessors = dict()
        if len(self.settings['postprocessors']) > 0:
            self.with_postprocessors = True
        for postproc in self.settings['postprocessors']:
            self.postprocessors[postproc] = solver_interface.initialise_solver(postproc)
            self.postprocessors[postproc].initialise(
                self.data, self.settings['postprocessors_settings'][postproc])

        self.residual_table = cout.TablePrinter(2, 14, ['g', 'f'])
        self.residual_table.field_length[0] = 6
        self.residual_table.field_length[1] = 6
        self.residual_table.print_header(['ts', 't'])

    def cleanup_timestep_info(self):
        if len(self.data.aero.timestep_info) > 1:
            # copy last info to first
            self.data.aero.timestep_info[0] = self.data.aero.timestep_info[-1]
            # delete all the rest
            while len(self.data.aero.timestep_info) - 1:
                del self.data.aero.timestep_info[-1]

        self.data.ts = 0

    def increase_ts(self):
        self.data.structure.next_step()
        self.aero_solver.add_step()

    def run(self):
        structural_kstep = self.data.structure.ini_info.copy()

        # dynamic simulations start at tstep == 1, 0 is reserved for the initial state
        for self.data.ts in range(1, self.settings['n_time_steps'].value + 1):
            aero_kstep = self.data.aero.timestep_info[-1].copy()
            structural_kstep = self.data.structure.timestep_info[-1].copy()
            ts = len(self.data.structure.timestep_info) - 1
            if ts > 0:
                self.data.structure.timestep_info[ts].for_vel[:] = self.data.structure.dynamic_input[ts - 1]['for_vel']
                self.data.structure.timestep_info[ts].for_acc[:] = self.data.structure.dynamic_input[ts - 1]['for_acc']


            # # # generate new grid (already rotated)
            # self.aero_solver.update_custom_grid(structural_kstep, aero_kstep)
            #
            # # run the solver
            # self.data = self.aero_solver.run(aero_kstep,
            #                                  structural_kstep,
            #                                  self.data.aero.timestep_info[-1],
            #                                  convect_wake=True)
            #
            # self.residual_table.print_line([self.data.ts,
            #                                 self.data.ts*self.dt.value])

            self.data.structure.next_step()
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
            self.residual_table.print_line([self.data.ts,
                                            self.data.ts*self.dt.value])

            # run postprocessors
            if self.with_postprocessors:
                for postproc in self.postprocessors:
                    self.data = self.postprocessors[postproc].run(online=True)

        return self.data

#
#     def map_forces(self, aero_kstep, structural_kstep, unsteady_forces_coeff=1.0):
#         # set all forces to 0
#         structural_kstep.steady_applied_forces.fill(0.0)
#         structural_kstep.unsteady_applied_forces.fill(0.0)
#
#         # aero forces to structural forces
#         struct_forces = mapping.aero2struct_force_mapping(
#             aero_kstep.forces,
#             self.data.aero.struct2aero_mapping,
#             aero_kstep.zeta,
#             structural_kstep.pos,
#             structural_kstep.psi,
#             self.data.structure.node_master_elem,
#             self.data.structure.master,
#             structural_kstep.cag())
#         dynamic_struct_forces = unsteady_forces_coeff*mapping.aero2struct_force_mapping(
#             aero_kstep.dynamic_forces,
#             self.data.aero.struct2aero_mapping,
#             aero_kstep.zeta,
#             structural_kstep.pos,
#             structural_kstep.psi,
#             self.data.structure.node_master_elem,
#             self.data.structure.master,
#             structural_kstep.cag())
#
#         # prescribed forces + aero forces
#         structural_kstep.steady_applied_forces = (
#             (struct_forces + self.data.structure.ini_info.steady_applied_forces).
#                 astype(dtype=ct.c_double, order='F', copy=True))
#         structural_kstep.unsteady_applied_forces = (
#             (dynamic_struct_forces + self.data.structure.dynamic_input[max(self.data.ts - 1, 0)]['dynamic_forces']).
#                 astype(dtype=ct.c_double, order='F', copy=True))
#
#     def relaxation_factor(self, k):
#         initial = self.settings['relaxation_factor'].value
#         if not self.settings['dynamic_relaxation'].value:
#             return initial
#
#         final = self.settings['final_relaxation_factor'].value
#         if k >= self.settings['relaxation_steps'].value:
#             return final
#
#         value = initial + (final - initial)/self.settings['relaxation_steps'].value*k
#         return value
#
#
# def relax(beam, timestep, previous_timestep, coeff):
#     if coeff > 0.0:
#         timestep.steady_applied_forces[:] = ((1.0 - coeff)*timestep.steady_applied_forces
#                                              + coeff*previous_timestep.steady_applied_forces)
#         timestep.unsteady_applied_forces[:] = ((1.0 - coeff)*timestep.unsteady_applied_forces
#                                                + coeff*previous_timestep.unsteady_applied_forces)
#         # timestep.pos_dot[:] = (1.0 - coeff)*timestep.pos_dot + coeff*previous_timestep.pos_dot
#         # timestep.psi[:] = (1.0 - coeff)*timestep.psi + coeff*previous_timestep.psi
#         # timestep.psi_dot[:] = (1.0 - coeff)*timestep.psi_dot + coeff*previous_timestep.psi_dot
#
#         # normalise_quaternion(timestep)
#         # xbeam_solv_state2disp(beam, timestep)
#
#
#
#
#

