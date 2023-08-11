import sys
import numpy as np

import sharpy.aero.utils.mapping as mapping
import sharpy.utils.cout_utils as cout
from sharpy.utils.solver_interface import solver, BaseSolver, initialise_solver

import sharpy.utils.settings as settings_utils
import sharpy.utils.algebra as algebra
import sharpy.utils.generator_interface as gen_interface

@solver
class StaticCoupled(BaseSolver):
    """
    This class is the main FSI driver for static simulations.
    It requires a ``structural_solver`` and a ``aero_solver`` to be defined.
    """
    solver_id = 'StaticCoupled'
    solver_classification = 'Coupled'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    settings_types['print_info'] = 'bool'
    settings_default['print_info'] = True
    settings_description['print_info'] = 'Write status to screen'

    settings_types['structural_solver'] = 'str'
    settings_default['structural_solver'] = None
    settings_description['structural_solver'] = 'Structural solver to use in the coupled simulation'

    settings_types['structural_solver_settings'] = 'dict'
    settings_default['structural_solver_settings'] = None
    settings_description['structural_solver_settings'] = 'Dictionary of settings for the structural solver'

    settings_types['aero_solver'] = 'str'
    settings_default['aero_solver'] = None
    settings_description['aero_solver'] = 'Aerodynamic solver to use in the coupled simulation'

    settings_types['aero_solver_settings'] = 'dict'
    settings_default['aero_solver_settings'] = None
    settings_description['aero_solver_settings'] = 'Dictionary of settings for the aerodynamic solver'

    settings_types['max_iter'] = 'int'
    settings_default['max_iter'] = 100
    settings_description['max_iter'] = 'Max iterations in the FSI loop'

    settings_types['n_load_steps'] = 'int'
    settings_default['n_load_steps'] = 0
    settings_description['n_load_steps'] = 'Length of ramp for forces and gravity during FSI iteration'

    settings_types['tolerance'] = 'float'
    settings_default['tolerance'] = 1e-5
    settings_description['tolerance'] = 'Convergence threshold for the FSI loop'

    settings_types['relaxation_factor'] = 'float'
    settings_default['relaxation_factor'] = 0.
    settings_description['relaxation_factor'] = 'Relaxation parameter in the FSI iteration. 0 is no relaxation and -> 1 is very relaxed'

    settings_types['correct_forces_method'] = 'str'
    settings_default['correct_forces_method'] = ''
    settings_description['correct_forces_method'] = 'Function used to correct aerodynamic forces. ' \
                                                    'See :py:mod:`sharpy.generators.polaraeroforces`'
    settings_options['correct_forces_method'] = ['EfficiencyCorrection', 'PolarCorrection']

    settings_types['correct_forces_settings'] = 'dict'
    settings_default['correct_forces_settings'] = {}
    settings_description['correct_forces_settings'] = 'Settings for corrected forces evaluation'

    settings_types['runtime_generators'] = 'dict'
    settings_default['runtime_generators'] = dict()
    settings_description['runtime_generators'] = 'The dictionary keys are the runtime generators to be used. ' \
                                                 'The dictionary values are dictionaries with the settings ' \
                                                 'needed by each generator.'
    
    settings_types['nonlifting_body_interactions'] = 'bool'
    settings_default['nonlifting_body_interactions'] = False
    settings_description['nonlifting_body_interactions'] = 'Consider forces induced by nonlifting bodies'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description, settings_options)

    def __init__(self):

        self.data = None
        self.settings = None
        self.structural_solver = None
        self.aero_solver = None

        self.previous_force = None

        self.residual_table = None

        self.correct_forces = False
        self.correct_forces_generator = None

        self.runtime_generators = dict()
        self.with_runtime_generators = False

    def initialise(self, data, input_dict=None, restart=False):
        self.data = data
        if input_dict is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = input_dict
        settings_utils.to_custom_types(self.settings,
                           self.settings_types,
                           self.settings_default,
                           options=self.settings_options,
                           no_ctype=True)

        self.print_info = self.settings['print_info']

        self.structural_solver = initialise_solver(self.settings['structural_solver'])
        self.structural_solver.initialise(self.data, self.settings['structural_solver_settings'], restart=restart)
        self.aero_solver = initialise_solver(self.settings['aero_solver'])
        self.aero_solver.initialise(self.structural_solver.data, self.settings['aero_solver_settings'], restart=restart)
        self.data = self.aero_solver.data

        if self.print_info:
            self.residual_table = cout.TablePrinter(9, 8, ['g', 'g', 'f', 'f', 'f', 'f', 'f', 'f', 'f'])
            self.residual_table.field_length[0] = 3
            self.residual_table.field_length[1] = 3
            self.residual_table.field_length[2] = 10
            self.residual_table.print_header(['iter', 'step', 'log10(res)', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'])

        # Define the function to correct aerodynamic forces
        if self.settings['correct_forces_method'] != '':
            self.correct_forces = True
            self.correct_forces_generator = gen_interface.generator_from_string(self.settings['correct_forces_method'])()
            self.correct_forces_generator.initialise(in_dict=self.settings['correct_forces_settings'],
                                                     aero=self.data.aero,
                                                     structure=self.data.structure,
                                                     rho=self.settings['aero_solver_settings']['rho'],
                                                     vortex_radius=self.settings['aero_solver_settings']['vortex_radius'],
                                                     output_folder = self.data.output_folder)

        # initialise runtime generators
        self.runtime_generators = dict()
        if self.settings['runtime_generators']:
            self.with_runtime_generators = True
            for rg_id, param in self.settings['runtime_generators'].items():
                gen = gen_interface.generator_from_string(rg_id)
                self.runtime_generators[rg_id] = gen()
                self.runtime_generators[rg_id].initialise(param, data=self.data, restart=restart)

    def increase_ts(self):
        self.data.ts += 1
        self.structural_solver.next_step()
        self.aero_solver.next_step()

    def cleanup_timestep_info(self):
        if max(len(self.data.aero.timestep_info), len(self.data.structure.timestep_info)) > 1:
            self.remove_old_timestep_info(self.data.structure.timestep_info)
            self.remove_old_timestep_info(self.data.aero.timestep_info)            
            if self.settings['nonlifting_body_interactions']:
                self.remove_old_timestep_info(self.data.nonlifting_body.timestep_info)

        self.data.ts = 0

    def remove_old_timestep_info(self, tstep_info):
        # copy last info to first
        tstep_info[0] = tstep_info[-1].copy()
        # delete all the rest
        while len(tstep_info) - 1:
            del tstep_info[-1]

    def run(self, **kwargs):
        for i_step in range(self.settings['n_load_steps'] + 1):
            if (i_step == self.settings['n_load_steps'] and
                    self.settings['n_load_steps'] > 0):
                break
            # load step coefficient
            if not self.settings['n_load_steps'] == 0:
                load_step_multiplier = (i_step + 1.0)/self.settings['n_load_steps']
            else:
                load_step_multiplier = 1.0

            # new storage every load step
            if i_step > 0:
                self.increase_ts()

            for i_iter in range(self.settings['max_iter']):
                # run aero
                self.data = self.aero_solver.run()

                # map force
                struct_forces = mapping.aero2struct_force_mapping(
                    self.data.aero.timestep_info[self.data.ts].forces,
                    self.data.aero.struct2aero_mapping,
                    self.data.aero.timestep_info[self.data.ts].zeta,
                    self.data.structure.timestep_info[self.data.ts].pos,
                    self.data.structure.timestep_info[self.data.ts].psi,
                    self.data.structure.node_master_elem,
                    self.data.structure.connectivities,
                    self.data.structure.timestep_info[self.data.ts].cag(),
                    self.data.aero.data_dict)
                        
                if self.correct_forces:
                    struct_forces = \
                        self.correct_forces_generator.generate(aero_kstep=self.data.aero.timestep_info[self.data.ts],
                                                               structural_kstep=self.data.structure.timestep_info[self.data.ts],
                                                               struct_forces=struct_forces,
                                                               ts=0)

                # map nonlifting forces to structural nodes
                if self.settings['nonlifting_body_interactions']:
                    struct_forces += mapping.aero2struct_force_mapping(
                        self.data.nonlifting_body.timestep_info[self.data.ts].forces,
                        self.data.nonlifting_body.struct2aero_mapping,
                        self.data.nonlifting_body.timestep_info[self.data.ts].zeta,
                        self.data.structure.timestep_info[self.data.ts].pos,
                        self.data.structure.timestep_info[self.data.ts].psi,
                        self.data.structure.node_master_elem,
                        self.data.structure.connectivities,
                        self.data.structure.timestep_info[self.data.ts].cag(),
                        self.data.nonlifting_body.data_dict,
                        skip_moments_generated_by_forces = True)

                self.data.aero.timestep_info[self.data.ts].aero_steady_forces_beam_dof = struct_forces
                self.data.structure.timestep_info[self.data.ts].postproc_node['aero_steady_forces'] = struct_forces  # B
                
                # Add external forces
                if self.with_runtime_generators:
                    self.data.structure.timestep_info[self.data.ts].runtime_steady_forces.fill(0.)
                    self.data.structure.timestep_info[self.data.ts].runtime_unsteady_forces.fill(0.)
                    params = dict()
                    params['data'] = self.data
                    params['struct_tstep'] = self.data.structure.timestep_info[self.data.ts]
                    params['aero_tstep'] = self.data.aero.timestep_info[self.data.ts]
                    params['fsi_substep'] = -i_iter
                    for id, runtime_generator in self.runtime_generators.items():
                        runtime_generator.generate(params)

                    struct_forces += self.data.structure.timestep_info[self.data.ts].runtime_steady_forces
                    struct_forces += self.data.structure.timestep_info[self.data.ts].runtime_unsteady_forces

                if not self.settings['relaxation_factor'] == 0.:
                    if i_iter == 0:
                        self.previous_force = struct_forces.copy()

                    temp = struct_forces.copy()
                    struct_forces = ((1.0 - self.settings['relaxation_factor'])*struct_forces +
                                     self.settings['relaxation_factor']*self.previous_force)
                    self.previous_force = temp

                # copy force in beam
                old_g = self.structural_solver.settings['gravity']
                self.structural_solver.settings['gravity'] = old_g*load_step_multiplier
                temp1 = load_step_multiplier*(struct_forces + self.data.structure.ini_info.steady_applied_forces)
                self.data.structure.timestep_info[self.data.ts].steady_applied_forces[:] = temp1
                # run beam
                self.data = self.structural_solver.run()
                self.structural_solver.settings['gravity'] = old_g
                (self.data.structure.timestep_info[self.data.ts].total_forces[0:3],
                 self.data.structure.timestep_info[self.data.ts].total_forces[3:6]) = (
                        self.extract_resultants(self.data.structure.timestep_info[self.data.ts]))

                # update grid
                self.aero_solver.update_step()

                self.structural_solver.update(self.data.structure.timestep_info[self.data.ts])
                # convergence
                if self.convergence(i_iter, i_step):
                    # create q and dqdt vectors
                    self.structural_solver.update(self.data.structure.timestep_info[self.data.ts])
                    self.cleanup_timestep_info()
                    break

        return self.data

    def convergence(self, i_iter, i_step):
        if i_iter == self.settings['max_iter'] - 1:
            cout.cout_wrap('StaticCoupled did not converge!', 0)
            # quit(-1)

        return_value = None
        if i_iter == 0:
            self.initial_residual = np.linalg.norm(self.data.structure.timestep_info[self.data.ts].pos)
            self.previous_residual = self.initial_residual
            self.current_residual = self.initial_residual
            if self.print_info:
                forces = self.data.structure.timestep_info[self.data.ts].total_forces
                self.residual_table.print_line([i_iter,
                        i_step,
                        0.0,
                        forces[0],
                        forces[1],
                        forces[2],
                        forces[3],
                        forces[4],
                        forces[5],
                        ])
            return False

        self.current_residual = np.linalg.norm(self.data.structure.timestep_info[self.data.ts].pos)
        if self.print_info:
            forces = self.data.structure.timestep_info[self.data.ts].total_forces
            res_print = np.NINF
            if (np.abs(self.current_residual - self.previous_residual) >
                sys.float_info.epsilon*10):
                res_print = np.log10(np.abs(self.current_residual - self.previous_residual)/self.initial_residual)

            self.residual_table.print_line([i_iter,
                    i_step,
                    res_print,
                    forces[0],
                    forces[1],
                    forces[2],
                    forces[3],
                    forces[4],
                    forces[5],
                    ])

        if return_value is None:
            if np.abs(self.current_residual - self.previous_residual)/self.initial_residual < self.settings['tolerance']:
                return_value = True
            else:
                self.previous_residual = self.current_residual
                return_value = False

        if return_value is None:
            return_value = False

        return return_value

    def change_trim(self, alpha, thrust, thrust_nodes, tail_deflection, tail_cs_index):
        # self.cleanup_timestep_info()
        self.data.structure.timestep_info = []
        self.data.structure.timestep_info.append(self.data.structure.ini_info.copy())
        aero_copy = self.data.aero.timestep_info[-1]
        self.data.aero.timestep_info = []
        self.data.aero.timestep_info.append(aero_copy)
        self.data.ts = 0
        # alpha
        orientation_quat = algebra.euler2quat(np.array([0.0, alpha, 0.0]))
        self.data.structure.timestep_info[0].quat[:] = orientation_quat[:]

        try:
            self.force_orientation
        except AttributeError:
            self.force_orientation = np.zeros((len(thrust_nodes), 3))
            for i_node, node in enumerate(thrust_nodes):
                self.force_orientation[i_node, :] = (
                    algebra.unit_vector(self.data.structure.ini_info.steady_applied_forces[node, 0:3]))
            # print(self.force_orientation)

        # thrust
        # thrust is scaled so that the direction of the forces is conserved
        # in all nodes.
        # the `thrust` parameter is the force PER node.
        # if there are two or more nodes in thrust_nodes, the total forces
        # is n_nodes_in_thrust_nodes*thrust
        # thrust forces have to be indicated in structure.ini_info
        # print(algebra.unit_vector(self.data.structure.ini_info.steady_applied_forces[0, 0:3])*thrust)
        for i_node, node in enumerate(thrust_nodes):
            # self.data.structure.ini_info.steady_applied_forces[i_node, 0:3] = (
            #     algebra.unit_vector(self.data.structure.ini_info.steady_applied_forces[i_node, 0:3])*thrust)
            self.data.structure.ini_info.steady_applied_forces[node, 0:3] = (
                    self.force_orientation[i_node, :]*thrust)
            self.data.structure.timestep_info[0].steady_applied_forces[node, 0:3] = (
                    self.force_orientation[i_node, :]*thrust)

        # tail deflection
        try:
            self.data.aero.data_dict['control_surface_deflection'][tail_cs_index] = tail_deflection
        except KeyError:
            raise Exception('This model has no control surfaces')
        except IndexError:
            raise Exception('The tail control surface index > number of surfaces')

        # update grid
        self.aero_solver.update_step()

    def extract_resultants(self, tstep=None):
        return self.structural_solver.extract_resultants(tstep)
    

    def teardown(self):
        
        self.structural_solver.teardown()
        self.aero_solver.teardown()
        if self.with_runtime_generators:
            for rg in self.runtime_generators.values():
                rg.teardown()


