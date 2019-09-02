import numpy as np
import os

import sharpy.utils.cout_utils as cout
import sharpy.utils.solver_interface as solver_interface
import sharpy.utils.generator_interface as gen_interface
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra


@solver
class LagrangeMultipliersTrajectoryControl(BaseSolver):
    solver_id = 'LagrangeMultipliersTrajectoryControl'

    def __init__(self):
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['print_info'] = 'bool'
        self.settings_default['print_info'] = True

        self.settings_types['trajectory_solver'] = 'str'
        self.settings_default['trajectory_solver'] = None

        self.settings_types['trajectory_solver_settings'] = 'dict'
        self.settings_default['trajectory_solver_settings'] = None

        self.settings_types['n_time_steps'] = 'int'
        self.settings_default['n_time_steps'] = 100

        self.settings_types['dt'] = 'float'
        self.settings_default['dt'] = None

        self.settings_types['postprocessors'] = 'list(str)'
        self.settings_default['postprocessors'] = list()

        self.settings_types['postprocessors_settings'] = 'dict'
        self.settings_default['postprocessors_settings'] = dict()

        self.settings_types['nodes_trajectory'] = 'list(int)'
        self.settings_default['nodes_trajectory'] = None

        self.settings_types['node_constraints'] = 'list(str)'
        self.settings_default['node_constraints'] = None

        self.settings_types['trajectory_generator'] = 'str'
        self.settings_default['trajectory_generator'] = None

        self.settings_types['trajectory_generator_input'] = 'dict'
        self.settings_default['trajectory_generator_input'] = None

        self.settings_types['transient_nsteps'] = 'int'
        self.settings_default['transient_nsteps'] = 0

        self.settings_types['write_trajectory_data'] = 'bool'
        self.settings_default['write_trajectory_data'] = False

        self.data = None
        self.settings = None
        self.solver = None

        self.previous_force = None

        self.dt = 0.

        self.controllers = list()
        self.n_controlled_points = 0

        self.trajectory = None
        self.input_trajectory = None
        self.force_history = None

        self.predictor = False
        self.residual_table = None
        self.postprocessors = dict()
        self.with_postprocessors = False

        self.trajectory_generator = None
        self.trajectory_steps = None

        self.print_info = None
        self.trajectory_data_folder = None
        self.trajectory_data_filename = []

    def initialise(self, data):
        self.data = data
        self.settings = data.settings[self.solver_id]
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)
        self.dt = self.settings['dt'].value

        self.solver = solver_interface.initialise_solver(self.settings['trajectory_solver'])
        self.settings['trajectory_solver_settings']['n_time_steps'] = 1
        self.solver.initialise(self.data, self.settings['trajectory_solver_settings'])

        self.n_controlled_points = len(self.settings['nodes_trajectory'])

        self.trajectory = np.zeros((self.settings['n_time_steps'].value, len(self.settings['nodes_trajectory']), 3))
        self.input_trajectory = np.zeros((self.settings['n_time_steps'].value, len(self.settings['nodes_trajectory']), 3))
        self.force_history = np.zeros((self.settings['n_time_steps'].value, len(self.settings['nodes_trajectory']), 3))

        # initialise trayectory generator
        trajectory_generator_type = gen_interface.generator_from_string(
            self.settings['trajectory_generator'])
        self.trajectory_generator = trajectory_generator_type()
        self.trajectory_generator.initialise(self.settings['trajectory_generator_input'])
        self.trajectory_steps = self.trajectory_generator.get_n_steps()

        if self.settings['write_trajectory_data']:
            self.trajectory_data_folder = os.path.abspath(self.data.settings['SHARPy']['route'] +
                                                          '/output/' +
                                                          self.data.settings['SHARPy']['case'] +
                                                          '/trajectory_data/')
            if not os.path.exists(self.trajectory_data_folder):
                os.makedirs(self.trajectory_data_folder)

            self.trajectory_data_filename = []
            for i, i_node in enumerate(self.settings['nodes_trajectory']):
                self.trajectory_data_filename.append(self.trajectory_data_folder + '/node_%u' % i_node + '.csv')
                if os.path.exists(self.trajectory_data_filename[i]):
                    os.remove(self.trajectory_data_filename[i])

            self.format = ('%i', )
            for i in range(3 + 3 + 3 + 1):
                self.format = self.format + ('%10.5f', )

        self.print_info = self.settings['print_info']
        if self.print_info:
            self.residual_table = cout.TablePrinter(3, 14, ['g', 'f', 'f'])
            self.residual_table.field_length[0] = 6
            self.residual_table.field_length[1] = 6
            self.residual_table.field_length[1] = 6
            self.residual_table.print_header(['ts', 't', 'traj. offset'])

    def run(self):
        local_it = -1
        MB_dict = self.data.structure.mb_dict
        for self.data.ts in range(len(self.data.structure.timestep_info) - 1,
                                  self.settings['n_time_steps'].value + len(self.data.structure.timestep_info) - 1):
            # print(self.data.ts)
            local_it += 1
            if local_it < self.settings['transient_nsteps'].value:
                coeff = local_it/self.settings['transient_nsteps'].value

                old_g = self.solver.get_g()
                new_g = coeff*old_g
                self.solver.set_g(new_g)

                old_rho = self.solver.get_rho()
                new_rho = coeff*old_rho
                self.solver.set_rho(new_rho)

            if local_it < self.trajectory_steps:
                # get location of points
                self.trajectory[local_it, :, :] = self.extract_trajectory(self.settings['nodes_trajectory'])
                # get desired location of points
                # add generator
                parameters = {'it': local_it}
                temp_trajectory = self.trajectory_generator(parameters)
                print('local_it in lmtrajectory: ', local_it)
                print('self.data.ts in lmtrajectory: ', self.data.ts)
                print('vel = ', temp_trajectory)

                for inode in range(len(self.settings['nodes_trajectory'])):
                    i_global_node = self.settings['nodes_trajectory'][inode]
                    self.input_trajectory[local_it, inode, :] = temp_trajectory

                    constraint = self.settings['node_constraints'][inode]

                    # add the velocity information
                    MB_dict[constraint]['velocity'] = self.input_trajectory[local_it, inode, :]
            else:
                print('it>trajectory_steps')
                # import pdb; pdb.set_trace()
                self.solver.structural_solver.remove_constraints()

            self.data = self.solver.run()

            if local_it < self.settings['transient_nsteps'].value:
                self.solver.set_g(old_g)
                self.solver.set_rho(old_rho)

            if self.settings['write_trajectory_data']:
                for i, i_node in enumerate(self.settings['nodes_trajectory']):
                    with open(self.trajectory_data_filename[i], 'ba') as f:
                        line = np.array([self.data.ts,
                                         self.data.ts*self.settings['dt'].value,
                                         self.input_trajectory[local_it, i, 0],
                                         self.input_trajectory[local_it, i, 1],
                                         self.input_trajectory[local_it, i, 2],
                                         self.trajectory[local_it, i, 0],
                                         self.trajectory[local_it, i, 1],
                                         self.trajectory[local_it, i, 2]])
                        np.savetxt(f, np.atleast_2d(line), delimiter=',', fmt=self.format, newline="\n")


            if self.print_info:
                res = np.linalg.norm(self.input_trajectory[local_it, :, :] - self.trajectory[local_it, :, :])
                force = np.linalg.norm(self.force_history[local_it, ...])
                self.residual_table.print_line([self.data.ts,
                                                self.data.ts*self.settings['dt'].value,
                                                res,
                                                force])

            if local_it > self.trajectory_steps:
                break

        return self.data

    def extract_trajectory(self, nodes, it=None):
        if it is None:
            it = self.data.ts

        trajectory = np.zeros((len(nodes), 3))
        coordinates = self.data.structure.timestep_info[it - 1].glob_pos(include_rbm=True)
        for inode in range(len(nodes)):
            trajectory[inode] = coordinates[nodes[inode], :]

        return trajectory

