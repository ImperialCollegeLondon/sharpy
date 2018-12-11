import ctypes as ct

import numpy as np

import sharpy.aero.utils.mapping as mapping
import sharpy.utils.cout_utils as cout
import sharpy.utils.solver_interface as solver_interface
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra
import sharpy.utils.multibody as mb


@solver
class InitializeMultibody(BaseSolver):
    solver_id = 'InitializeMultibody'

    def __init__(self):
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['dt'] = 'float'
        self.settings_default['dt'] = 0.

        self.settings_types['rot_vel'] = 'float'
        self.settings_default['rot_vel'] = 0.0

        self.settings_types['rot_axis'] = 'list(float)'
        self.settings_default['rot_axis'] = np.array([1.,0.,0.])

        self.settings_types['rot_center'] = 'list(float)'
        self.settings_default['rot_center'] = np.array([0.,0.,0.])

        self.settings_types['u_inf'] = 'float'
        self.settings_default['u_inf'] = 0.0

        self.settings_types['u_inf_direction'] = 'list(float)'
        self.settings_default['u_inf_direction'] = np.array([1.,0.,0.])

        self.data = None
        self.settings = None
        self.structural_solver = None
        self.aero_solver = None

        self.previous_force = None

    def initialise(self, data, input_dict=None):
        self.data = data
        if input_dict is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = input_dict
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        # self.data = self.aero_solver.data

    def increase_ts(self):
        self.data.ts += 1
        self.structural_solver.next_step()
        self.aero_solver.next_step()

    def cleanup_timestep_info(self):
        if max(len(self.data.aero.timestep_info), len(self.data.structure.timestep_info)) > 1:
            # copy last info to first
            self.data.aero.timestep_info[0] = self.data.aero.timestep_info[-1].copy()
            self.data.structure.timestep_info[0] = self.data.structure.timestep_info[-1].copy()
            # delete all the rest
            while len(self.data.aero.timestep_info) - 1:
                del self.data.aero.timestep_info[-1]
            while len(self.data.structure.timestep_info) - 1:
                del self.data.structure.timestep_info[-1]

        self.data.ts = 0

    def run(self):

        # Initialize solid velocities:
        MB_beam, MB_tstep = mb.split_multibody(self.data.structure, self.data.structure.timestep_info[-1], self.data.structure.mb_dict, 0)

        # for inode in self.data.structure.num_node:
        #     ibody = self.data.structure.body_number[self.data.structure.node_master_elem[inode]]

        # self.data.structure.ini_info.pos_dot *= -1.
        # self.data.structure.ini_info.psi_dot *= -1.
        # self.data.structure.timestep_info[-1].pos_dot *= -1.
        # self.data.structure.timestep_info[-1].psi_dot *= -1.


        # Initialize wake
        rot_vel = self.settings['rot_vel']
        rot_axis = self.settings['rot_axis']
        rot_center = self.settings['rot_center']
        disp_vel = np.zeros((3,),)
        wsp = self.settings['u_inf']*self.settings['u_inf_direction']

        self.define_helicoidal_wake(self.data.aero.timestep_info[-1], self.data.structure.timestep_info[-1], rot_vel.value, rot_axis, rot_center, wsp-disp_vel)

        # for i_step in range(self.settings['n_load_steps'].value + 1):
        #     if (i_step == self.settings['n_load_steps'].value and
        #             self.settings['n_load_steps'].value > 0):
        #         break
        #     # load step coefficient
        #     if not self.settings['n_load_steps'].value == 0:
        #         load_step_multiplier = (i_step + 1.0)/self.settings['n_load_steps'].value
        #     else:
        #         load_step_multiplier = 1.0
        #
        #     # new storage every load step
        #     if i_step > 0:
        #         self.increase_ts()
        #
        #     for i_iter in range(self.settings['max_iter'].value):
        #         if self.settings['print_info'].value:
        #             cout.cout_wrap('i_step: %u, i_iter: %u' % (i_step, i_iter))
        #
        #         # run aero
        #         self.data = self.aero_solver.run()
        #
        #         # map force
        #         struct_forces = mapping.aero2struct_force_mapping(
        #             self.data.aero.timestep_info[self.data.ts].forces,
        #             self.data.aero.struct2aero_mapping,
        #             self.data.aero.timestep_info[self.data.ts].zeta,
        #             self.data.structure.timestep_info[self.data.ts].pos,
        #             self.data.structure.timestep_info[self.data.ts].psi,
        #             self.data.structure.node_master_elem,
        #             self.data.structure.master,
        #             self.data.structure.timestep_info[self.data.ts].cag())
        #
        #         if not self.settings['relaxation_factor'].value == 0.:
        #             if i_iter == 0:
        #                 self.previous_force = struct_forces.copy()
        #
        #             temp = struct_forces.copy()
        #             struct_forces = ((1.0 - self.settings['relaxation_factor'].value)*struct_forces +
        #                              self.settings['relaxation_factor'].value*self.previous_force)
        #             self.previous_force = temp
        #
        #         # copy force in beam
        #         old_g = self.structural_solver.settings['gravity'].value
        #         self.structural_solver.settings['gravity'] = 0.0
        #         temp1 = load_step_multiplier*(struct_forces + self.data.structure.ini_info.steady_applied_forces)
        #         # self.data.structure.timestep_info[self.data.ts].steady_applied_forces[:] = temp1
        #         # run beam
        #         # self.data = self.structural_solver.run()
        #         self.structural_solver.settings['gravity'] = ct.c_double(old_g)
        #
        #         # update grid
        #         self.aero_solver.update_step()
        #
        #         # convergence
        #         if self.convergence(i_iter, i_step):
        #             # create q and dqdt vectors
        #             self.structural_solver.update(self.data.structure.timestep_info[self.data.ts])
        #             self.cleanup_timestep_info()
        #             break
        #
        # if self.settings['print_info']:
        #     resultants = self.extract_resultants()
        #     cout.cout_wrap('Resultant forces and moments: ' + str(resultants))
        return self.data

    def define_helicoidal_wake(self, aero_data, structure_data, rot_vel, rot_axis, rot_center, wsp):

        def rotate_vector(vector,direction,angle):
            # This function rotates a "vector" around a "direction" a certain "angle"
            # according to Rodrigues formula

            # Assure that "direction" has unit norm
            if not np.linalg.norm(direction) == 0:
                direction/=np.linalg.norm(direction)

            rot_vector=vector*np.cos(angle)+np.cross(direction,vector)*np.sin(angle)+direction*np.dot(direction,vector)*(1.0-np.cos(angle))

            return rot_vector

        for i_surf in range(aero_data.n_surf):
            for i_n in range(aero_data.dimensions_star[i_surf, 1]+1):
                for i_m in range(aero_data.dimensions_star[i_surf, 0]+1):
                    associated_t=self.settings['dt'].value*i_m
                    # wake rotates in the opposite direction to the solid
                    dphi=-1.0*rot_vel*associated_t

                    aero_data.zeta_star[i_surf][:, i_m, i_n] = rotate_vector(aero_data.zeta[i_surf][:, -1 , i_n] - rot_center,rot_axis,dphi) + rot_center
                    aero_data.zeta_star[i_surf][:, i_m, i_n] += wsp*associated_t

    # def convergence(self, i_iter, i_step):
    #     if i_iter == self.settings['max_iter'].value - 1:
    #         cout.cout_wrap('StaticCoupled did not converge!', 0)
    #         # quit(-1)
    #
    #     return_value = None
    #     if i_iter == 0:
    #         self.initial_residual = np.linalg.norm(self.data.structure.timestep_info[self.data.ts].pos)
    #         self.previous_residual = self.initial_residual
    #         self.current_residual = self.initial_residual
    #         return False
    #
    #     self.current_residual = np.linalg.norm(self.data.structure.timestep_info[self.data.ts].pos)
    #     if self.settings['print_info'].value:
    #         cout.cout_wrap('Res = %8e' % (np.abs(self.current_residual - self.previous_residual)/self.previous_residual), 2)
    #
    #     if return_value is None:
    #         if np.abs(self.current_residual - self.previous_residual)/self.initial_residual < self.settings['tolerance'].value:
    #             return_value = True
    #         else:
    #             self.previous_residual = self.current_residual
    #             return_value = False
    #
    #     if return_value is None:
    #         return_value = False
    #
    #     return return_value
    #
    # def change_trim(self, alpha, thrust, thrust_nodes, tail_deflection, tail_cs_index):
    #     # self.cleanup_timestep_info()
    #     self.data.structure.timestep_info = []
    #     self.data.structure.timestep_info.append(self.data.structure.ini_info.copy())
    #     aero_copy = self.data.aero.timestep_info[-1]
    #     self.data.aero.timestep_info = []
    #     self.data.aero.timestep_info.append(aero_copy)
    #     self.data.ts = 0
    #     # alpha
    #     orientation_quat = algebra.euler2quat(np.array([0.0, alpha, 0.0]))
    #     self.data.structure.timestep_info[0].quat[:] = orientation_quat[:]
    #
    #     try:
    #         self.force_orientation
    #     except AttributeError:
    #         self.force_orientation = np.zeros((len(thrust_nodes), 3))
    #         for i_node, node in enumerate(thrust_nodes):
    #             self.force_orientation[i_node, :] = (
    #                 algebra.unit_vector(self.data.structure.ini_info.steady_applied_forces[node, 0:3]))
    #         # print(self.force_orientation)
    #
    #     # thrust
    #     # thrust is scaled so that the direction of the forces is conserved
    #     # in all nodes.
    #     # the `thrust` parameter is the force PER node.
    #     # if there are two or more nodes in thrust_nodes, the total forces
    #     # is n_nodes_in_thrust_nodes*thrust
    #     # thrust forces have to be indicated in structure.ini_info
    #     # print(algebra.unit_vector(self.data.structure.ini_info.steady_applied_forces[0, 0:3])*thrust)
    #     for i_node, node in enumerate(thrust_nodes):
    #         # self.data.structure.ini_info.steady_applied_forces[i_node, 0:3] = (
    #         #     algebra.unit_vector(self.data.structure.ini_info.steady_applied_forces[i_node, 0:3])*thrust)
    #         self.data.structure.ini_info.steady_applied_forces[node, 0:3] = (
    #                 self.force_orientation[i_node, :]*thrust)
    #         self.data.structure.timestep_info[0].steady_applied_forces[node, 0:3] = (
    #                 self.force_orientation[i_node, :]*thrust)
    #
    #     # tail deflection
    #     try:
    #         self.data.aero.aero_dict['control_surface_deflection'][tail_cs_index] = tail_deflection
    #     except KeyError:
    #         raise Exception('This model has no control surfaces')
    #     except IndexError:
    #         raise Exception('The tail control surface index > number of surfaces')
    #
    #     # update grid
    #     self.aero_solver.update_step()
    #
    # def extract_resultants(self, tstep=None):
    #     return self.structural_solver.extract_resultants(tstep)
