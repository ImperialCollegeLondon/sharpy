import ctypes as ct

import numpy as np

import sharpy.aero.utils.mapping as mapping
import sharpy.utils.cout_utils as cout
import sharpy.utils.solver_interface as solver_interface
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra
import sharpy.structure.utils.xbeamlib as xbeam

# Needed to refer to the cpp library
from sharpy.utils.sharpydir import SharpyDir
import sharpy.utils.ctypes_utils as ct_utils
UvlmLib = ct_utils.import_ctypes_lib(SharpyDir + '/lib/', 'libuvlm')


@solver
class SteadyHelicoidalWake(BaseSolver):
    solver_id = 'SteadyHelicoidalWake'

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

        self.settings_types['fsi_vel_tolerance'] = 'float'
        self.settings_default['fsi_vel_tolerance'] = 1e-5

        self.settings_types['relaxation_factor'] = 'float'
        self.settings_default['relaxation_factor'] = 0.2

        self.settings_types['final_relaxation_factor'] = 'float'
        self.settings_default['final_relaxation_factor'] = 0.7

        self.settings_types['minimum_steps'] = 'int'
        self.settings_default['minimum_steps'] = 3

        self.settings_types['relaxation_steps'] = 'int'
        self.settings_default['relaxation_steps'] = 100

        self.settings_types['dynamic_relaxation'] = 'bool'
        self.settings_default['dynamic_relaxation'] = True

        self.settings_types['postprocessors'] = 'list(str)'
        self.settings_default['postprocessors'] = list()

        self.settings_types['postprocessors_settings'] = 'dict'
        self.settings_default['postprocessors_settings'] = dict()

        self.settings_types['cleanup_previous_solution'] = 'bool'
        self.settings_default['cleanup_previous_solution'] = True

        self.settings_types['include_unsteady_force_contribution'] = 'bool'
        self.settings_default['include_unsteady_force_contribution'] = False

        self.settings_types['rigid_structure'] = 'bool'
        self.settings_default['rigid_structure'] = False

        self.settings_types['circulation_tolerance'] = 'float'
        self.settings_default['circulation_tolerance'] = 1e-5

        self.settings_types['circulation_substeps'] = 'int'
        self.settings_default['circulation_substeps'] = 70

        self.settings_types['rot_vel'] = 'float'
        self.settings_default['rot_vel'] = 0.0

        self.settings_types['rot_axis'] = 'list(float)'
        self.settings_default['rot_axis'] = np.array([1.,0.,0.])

        self.settings_types['rot_center'] = 'list(float)'
        self.settings_default['rot_center'] = np.array([0.,0.,0.])

        self.data = None
        self.settings = None
        self.structural_solver = None
        self.aero_solver = None
        self.print_info = False

        self.res = 0.0
        self.res_dqdt = 0.0
        self.res_gamma = 0.0

        self.residual_table = None
        self.postprocessors = dict()
        self.with_postprocessors = False

    def get_g(self):
        return self.structural_solver.settings['gravity'].value

    def set_g(self, new_g):
        self.structural_solver.settings['gravity'] = ct.c_double(new_g)

    def get_rho(self):
        return self.aero_solver.settings['rho'].value

    def set_rho(self, new_rho):
        self.aero_solver.settings['rho'] = ct.c_double(new_rho)

    def initialise(self, data, custom_settings=None):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)
        self.dt = self.settings['dt']
        self.print_info = self.settings['print_info']
        if self.settings['cleanup_previous_solution']:
            # if there's data in timestep_info[>0], copy the last one to
            # timestep_info[0] and remove the rest
            self.cleanup_timestep_info()

        self.structural_solver = solver_interface.initialise_solver(self.settings['structural_solver'])
        self.structural_solver.initialise(self.data, self.settings['structural_solver_settings'])
        self.aero_solver = solver_interface.initialise_solver(self.settings['aero_solver'])
        self.aero_solver.initialise(self.structural_solver.data, self.settings['aero_solver_settings'])
        self.data = self.aero_solver.data

        if self.print_info:
            self.residual_table = cout.TablePrinter(7, 14, ['g', 'f', 'g', 'f', 'f', 'f', 'e'])
            self.residual_table.field_length[0] = 6
            self.residual_table.field_length[1] = 6
            self.residual_table.field_length[1] = 6
            self.residual_table.print_header(['ts', 't', 'iter', 'residual pos', 'residual vel', 'residual gamma', 'pos[0][0,-1]'])

        # initialise postprocessors
        self.postprocessors = dict()
        if len(self.settings['postprocessors']) > 0:
            self.with_postprocessors = True
        for postproc in self.settings['postprocessors']:
            self.postprocessors[postproc] = solver_interface.initialise_solver(postproc)
            self.postprocessors[postproc].initialise(
                self.data, self.settings['postprocessors_settings'][postproc])

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

        # Define simulation variables
        # self.data.structure.timestep_info[-1].for_vel[:] = self.data.structure.dynamic_input[0]['for_vel']
        # self.data.structure.timestep_info[-1].for_acc[:] = [0.0,0.0,0.0,0.0,0.0,0.0]
        # disp_vel = self.data.structure.timestep_info[-1].for_vel[0:3]
        # rot_vel = np.linalg.norm(self.data.structure.timestep_info[-1].for_vel[3:6])
        # if not rot_vel == 0:
        #     rot_axis = self.data.structure.timestep_info[-1].for_vel[3:6] / rot_vel
        # else:
        #     rot_axis = np.zeros((3,),)
        rot_vel = self.settings['rot_vel']
        rot_axis = self.settings['rot_axis']
        rot_center = self.settings['rot_center']
        disp_vel = np.zeros((3,),)
        self.data.structure.timestep_info[-1].for_vel[3:6] = rot_vel*rot_axis
        self.data.structure.timestep_info[-1].for_acc[:] = [0.0,0.0,0.0,0.0,0.0,0.0]
        wsp = self.aero_solver.velocity_generator.u_inf * self.aero_solver.velocity_generator.u_inf_direction

        # Definition of the blade aero discretization and the helicoidal wake
        self.aero_solver.update_custom_grid(self.data.structure.timestep_info[-1], self.data.aero.timestep_info[-1])
        self.define_helicoidal_wake(self.data.aero.timestep_info[-1], self.data.structure.timestep_info[-1], rot_vel.value, rot_axis, rot_center, wsp-disp_vel)

        # Information for the FSI iteration
        aero_kstep = self.data.aero.timestep_info[-1].copy()
        structural_kstep = self.data.structure.timestep_info[-1].copy()

        # Reference velocity to judge convergence
        ref_vel_convergence = np.zeros((len(self.data.structure.timestep_info[-1].pos[:,0])),)
        for inode in range(len(self.data.structure.timestep_info[-1].pos[:,0])):
            ref_vel_convergence[inode] = np.linalg.norm(self.data.structure.timestep_info[-1].for_vel[0:3] +
                                                        np.dot(algebra.skew(self.data.structure.timestep_info[-1].pos[inode,:]),
                                                                  self.data.structure.timestep_info[-1].for_vel[3:6]))

        k=0
        for k in range(self.settings['fsi_substeps'].value + 1):
            if k == self.settings['fsi_substeps'].value and not self.settings['fsi_substeps'] == 0:
                cout.cout_wrap('The FSI solver did not converge!!!')
                break
                # TODO Raise Exception

            # Set velocity to zero (steady-state solver)
            structural_kstep.pos_dot = np.zeros_like(structural_kstep.pos_dot)
            previous_kstep = structural_kstep.copy()
            previous_aero_kstep = aero_kstep.copy()

            # Update aero discretization according to previous structural_kstep
            self.aero_solver.update_custom_grid(structural_kstep, aero_kstep)
            self.define_helicoidal_wake(aero_kstep, structural_kstep, rot_vel.value, rot_axis, rot_center, wsp-disp_vel)

            # Iterations to converge the circuation in the aerodynamic solver
            k2 = 0
            for k2 in range(self.settings['circulation_substeps'].value + 1):

                previous_aero_kstep = aero_kstep.copy()
                self.data = self.aero_solver.run(aero_kstep,
                                                 structural_kstep,
                                                 convect_wake=False)
                self.copy_gamma_star(aero_kstep)

                # Compute the residual of gamma and gamma star
                self.res_gamma = 0.0
                for isurf in range(len(aero_kstep.gamma)):
                    self.res_gamma += (np.linalg.norm(aero_kstep.gamma[isurf]-
                                               previous_aero_kstep.gamma[isurf])/
                                               np.linalg.norm(previous_aero_kstep.gamma[isurf]))
                    self.res_gamma += (np.linalg.norm(aero_kstep.gamma_star[isurf]-
                                               previous_aero_kstep.gamma_star[isurf])/
                                               np.linalg.norm(previous_aero_kstep.gamma_star[isurf]))

                # convergence
                if self.res_gamma < self.settings['circulation_tolerance'].value:
                    break

                # END OF ITERATIONS TO CONVERGE CIRCULATION

            # map forces
            force_coeff = 0.0
            if self.settings['include_unsteady_force_contribution']:
                force_coeff = -1.0
            self.map_forces(aero_kstep,
                            structural_kstep,
                            force_coeff)

            # relaxation
            relax_factor = self.relaxation_factor(k)
            relax(self.data.structure,
                  structural_kstep,
                  previous_kstep,
                  relax_factor)


            if not self.settings['rigid_structure']:
                # run structural solver
                self.data = self.structural_solver.run(structural_step=structural_kstep)
                # Move the structure back to the original position
                structural_kstep.quat = previous_kstep.quat.copy()

            # check for non-convergence
            if not np.isfinite(structural_kstep.pos).all:
                   cout.cout_wrap('***No converged!', 3)
                   break

            #xbeam.xbeam_solv_disp2state(self.data.structure, structural_kstep)

            # self.res = (np.linalg.norm(structural_kstep.q-
            #                            previous_kstep.q)/
            #                            np.linalg.norm(previous_kstep.q))

            self.res = (np.linalg.norm(structural_kstep.pos-
                                       previous_kstep.pos)/
                                       np.linalg.norm(previous_kstep.pos))

            # check velocity convergence with respect to the velocities from RBM
            # otherwise, as it is a steady-state solver tend to zero and the residual is not accurate
            point_vel = np.zeros((len(structural_kstep.pos[:,0])),)
            for inode in range(len(self.data.structure.timestep_info[-1].pos[:,0])):
                point_vel[inode] = np.linalg.norm(structural_kstep.pos_dot[inode,:])
            self.res_dqdt = np.linalg.norm(point_vel)/np.linalg.norm(ref_vel_convergence)

            # self.res_dqdt = (np.linalg.norm(structural_kstep.dqdt-
            #                                 previous_kstep.dqdt)/
            #                                 np.linalg.norm(structural_kstep.dqdt))

            if self.print_info:
                self.residual_table.print_line([self.data.ts,
                                                self.data.ts*self.dt.value,
                                                k,
                                                np.log10(self.res),
                                                np.log10(self.res_dqdt),
                                                np.log10(self.res_gamma),
                                                structural_kstep.pos[-1, 0]])

            # convergence
            if k > self.settings['minimum_steps'].value - 1:
                if self.res < self.settings['fsi_tolerance'].value:
                    if self.res_dqdt < self.settings['fsi_vel_tolerance'].value:
                       break

            # END OF FSI ITERATIONS

        # Overwrite the first time-step solution with the new one
        self.data.structure.timestep_info[-1] = structural_kstep.copy()
        self.data.aero.timestep_info[-1] = aero_kstep.copy()

        # run postprocessors
        if self.with_postprocessors:
            for postproc in self.postprocessors:
                self.data = self.postprocessors[postproc].run(online=True)

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

    def define_helicoidal_wake(self, aero_data, structure_data, rot_vel, rot_axis, rot_center, wsp):

        def rotate_vector(vector,direction,angle):
            # This function rotates a "vector" around a "direction" a certain "angle"
            # according to Rodrigues formula

            # Assure that "direction" has unit norm
            if not np.linalg.norm(direction) == 0:
                direction/=np.linalg.norm(direction)

            rot_vector=vector*np.cos(angle)+np.dot(algebra.skew(direction),vector)*np.sin(angle)+direction*np.dot(direction,vector)*(1.0-np.cos(angle))

            return rot_vector

        for i_surf in range(aero_data.n_surf):
            for i_n in range(aero_data.dimensions_star[i_surf, 1]+1):
                for i_m in range(aero_data.dimensions_star[i_surf, 0]+1):
                    associated_t=self.dt.value*i_m
                    # wake rotates in the opposite direction to the solid
                    dphi=-1.0*rot_vel*associated_t

                    aero_data.zeta_star[i_surf][:, i_m, i_n] = rotate_vector(aero_data.zeta[i_surf][:, -1 , i_n] - rot_center,rot_axis,dphi) + rot_center
                    aero_data.zeta_star[i_surf][:, i_m, i_n] += wsp*associated_t

    def copy_gamma_star(self, aero_data):

        # This function copies the circulation in the TE to all the points in the wake
        for i_surf in range(aero_data.n_surf):
            for i_n in range(aero_data.dimensions_star[i_surf, 1]):
                for i_m in range(aero_data.dimensions_star[i_surf, 0]):
                        aero_data.gamma_star[i_surf][i_m, i_n] = aero_data.gamma[i_surf][-1, i_n]

def relax(beam, timestep, previous_timestep, coeff):
    # from sharpy.structure.utils.xbeamlib import xbeam_solv_state2disp
    # numdof = beam.num_dof.value
    # timestep.q[:] = (1.0 - coeff)*timestep.q + coeff*previous_timestep.q
    # timestep.dqdt[:] = (1.0 - coeff)*timestep.dqdt + coeff*previous_timestep.dqdt
    # timestep.dqddt[:] = (1.0 - coeff)*timestep.dqddt + coeff*previous_timestep.dqddt

    # normalise_quaternion(timestep)
    # xbeam_solv_state2disp(beam, timestep)

    timestep.steady_applied_forces[:] = ((1.0 - coeff)*timestep.steady_applied_forces +
            coeff*previous_timestep.steady_applied_forces)
    timestep.unsteady_applied_forces[:] = ((1.0 - coeff)*timestep.unsteady_applied_forces +
            coeff*previous_timestep.unsteady_applied_forces)

def normalise_quaternion(tstep):
    tstep.dqdt[-4:] = algebra.unit_vector(tstep.dqdt[-4:])
    tstep.quat = tstep.dqdt[-4:].astype(dtype=ct.c_double, order='F', copy=True)
