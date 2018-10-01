import ctypes as ct
import numpy as np

import sharpy.utils.cout_utils as cout
import sharpy.utils.solver_interface as solver_interface
import sharpy.utils.generator_interface as gen_interface
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra


@solver
class PIDTrajectoryControl(BaseSolver):
    solver_id = 'PIDTrajectoryControl'

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
        self.settings_default['dt'] = 0.05

        self.settings_types['postprocessors'] = 'list(str)'
        self.settings_default['postprocessors'] = list()

        self.settings_types['postprocessors_settings'] = 'dict'
        self.settings_default['postprocessors_settings'] = dict()

        self.settings_types['PID_P_gain'] = 'float'
        self.settings_default['PID_P_gain'] = 1.2

        self.settings_types['PID_I_gain'] = 'float'
        self.settings_default['PID_I_gain'] = 1.

        self.settings_types['PID_D_gain'] = 'float'
        self.settings_default['PID_D_gain'] = 0.01

        self.settings_types['nodes_trajectory'] = 'list(int)'
        self.settings_default['nodes_trajectory'] = None

        self.settings_types['trajectory_generator'] = 'str'
        self.settings_default['trajectory_generator'] = None

        self.settings_types['trajectory_generator_input'] = 'dict'
        self.settings_default['trajectory_generator_input'] = dict()

        self.settings_types['transient_nsteps'] = 'int'
        self.settings_default['transient_nsteps'] = 0

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
        self.ini_coord_a = None
        self.trajectory_steps = None

        self.print_info = None

    def initialise(self, data):
        self.data = data
        self.settings = data.settings[self.solver_id]
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)
        self.dt = self.settings['dt'].value

        self.solver = solver_interface.initialise_solver(self.settings['trajectory_solver'])
        self.settings['trajectory_solver_settings']['n_time_steps'] = 1
        self.solver.initialise(self.data, self.settings['trajectory_solver_settings'])

        self.n_controlled_points = len(self.settings['nodes_trajectory'])
        # initialized controllers for the trayectory nodes
        # there will be 3 controllers per node (one per dimension)
        for i_trajec in range(self.n_controlled_points):
            self.controllers.append(list())
            for i_dim in range(3):
                self.controllers[i_trajec].append(PID(self.settings['PID_P_gain'].value,
                                                      self.settings['PID_I_gain'].value,
                                                      self.settings['PID_D_gain'].value,
                                                      self.dt))

        self.trajectory = np.zeros((self.settings['n_time_steps'].value, len(self.settings['nodes_trajectory']), 3))
        self.input_trajectory = np.zeros((self.settings['n_time_steps'].value, len(self.settings['nodes_trajectory']), 3))
        self.force_history = np.zeros((self.settings['n_time_steps'].value, len(self.settings['nodes_trajectory']), 3))

        # initialise trayectory generator
        trajectory_generator_type = gen_interface.generator_from_string(
            self.settings['trajectory_generator'])
        self.trajectory_generator = trajectory_generator_type()
        self.trajectory_generator.initialise(self.settings['trajectory_generator_input'])
        self.trajectory_steps = self.trajectory_generator.get_n_steps()

        # generate coordinates offset in order to be able to use only one
        # generator
        self.ini_coord_a = self.data.structure.ini_info.glob_pos(include_rbm=False)

        self.print_info = self.settings['print_info']
        if self.print_info:
            self.residual_table = cout.TablePrinter(4, 14, ['g', 'f', 'f', 'e'])
            self.residual_table.field_length[0] = 6
            self.residual_table.field_length[1] = 6
            self.residual_table.field_length[1] = 6
            self.residual_table.print_header(['ts', 't', 'traj. offset', 'norm(force)'])

    def run(self):
        local_it = -1
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
                # if self.data.structure.dynamic_input[self.data.ts]['enforce_trajectory'][:, :].any():
                # TODO change behaviour of enforce trajectory so that the full trajectory time is included
                # get location of points
                self.trajectory[local_it, :, :] = self.extract_trajectory(self.settings['nodes_trajectory'])
                # get desired location of points
                # add generator
                parameters = {'it': local_it}

                temp_trajectory = self.trajectory_generator(parameters)

                # self.input_trayectory[local_it, :, :] = (self.data.structure.dynamic_input[self.data.ts]
                #                                                                           ['trayectories']
                #                                                                           [self.settings['nodes_trayectory'], :])

                for inode in range(len(self.settings['nodes_trajectory'])):
                    i_global_node = self.settings['nodes_trajectory'][inode]
                    self.input_trajectory[local_it, inode, :] = self.ini_coord_a[i_global_node, :] + temp_trajectory
                    # print('node:' + str(i_global_node))
                    self.force_history[local_it, inode, :] = self.calculate_forces(self.trajectory[local_it, inode, :],
                                                                                   self.input_trajectory[local_it, inode, :],
                                                                                   i_global_node,
                                                                                   inode)

                    # debug fix:
                    if inode > 0:
                        self.force_history[local_it, inode, 1] = 0
                    # apply the forces now
                    (self.data.structure.dynamic_input[self.data.ts]
                        ['dynamic_forces']
                        [i_global_node, 0:3]) = (
                            self.data.structure.dynamic_input[self.data.ts]['dynamic_forces']
                            [i_global_node, 0:3] + self.force_history[local_it, inode, :])

                # print(self.force_history[local_it, :, :])
            # print(self.input_trajectory[local_it, :, :])
            self.data = self.solver.run()

            if local_it < self.settings['transient_nsteps'].value:
                self.solver.set_g(old_g)
                self.solver.set_rho(old_rho)

            if self.print_info:
                res = np.linalg.norm(self.input_trajectory[local_it, :, :] - self.trajectory[local_it, :, :])
                force = np.linalg.norm(self.force_history[local_it, ...])
                self.residual_table.print_line([self.data.ts,
                                                self.data.ts*self.settings['dt'].value,
                                                res,
                                                force])

        return self.data

    # def generate_input_trajectory(self, local_it):
    #     nodes = self.settings['nodes_trajectory']
    #     for i in nodes

    def extract_trajectory(self, nodes, it=None):
        if it is None:
            it = self.data.ts

        trajectory = np.zeros((len(nodes), 3))
        coordinates = self.data.structure.timestep_info[it - 1].glob_pos(include_rbm=True)
        for inode in range(len(nodes)):
            trajectory[inode] = coordinates[nodes[inode], :]

        return trajectory

    def calculate_forces(self, trajectory, input_trajectory, i_global_node, i_trajectory_node):
        """
        This calculates the necessary response based on the current trajectory (trajectoy),
        the desired trayectory (input_trajectory) and the given node id.
        The output will be already converted to the material (b) FoR.
        """
        # for i in range(n_steps):
        #     state[i] = np.sum(feedback[:i])
        #     controller.set_point(set_point[i])
        #     feedback[i] = controller(state[i])
        force = np.zeros((3,))
        for idim in range(3):
            # print('dim = ' + str(idim))
            self.controllers[i_trajectory_node][idim].set_point(input_trajectory[idim])
            force[idim] = self.controllers[i_trajectory_node][idim](trajectory[idim])

        master_elem, i_local_node_master_elem = self.data.structure.node_master_elem[i_global_node, :]
        cba = algebra.crv2rotation(
            self.data.structure.timestep_info[self.data.ts].psi[master_elem, i_local_node_master_elem, :]).T
        cag = self.data.structure.timestep_info[self.data.ts].cag()

        force = np.dot(cba, np.dot(cag, force))
        return force


def second_order_fd(history, n_calls, dt):
    # history is ordered such that the last element is the most recent one (t), and thus
    # it goes [(t - 2), (t - 1), (t)]
    coefficients = np.zeros((3,))
    if n_calls <= 1:
        # no derivative, return 0
        pass

    # elif n_calls == 2:
    else:
        # first order derivative
        coefficients[1:3] = [-1.0, 1.0]

    # else:
        # second order derivative
        # coefficients[:] = [1.0, -4.0, 3.0]

    derivative = np.dot(coefficients, history)/dt
    return derivative


class PID(object):
    # for i in range(n_steps):
    #     state[i] = np.sum(feedback[:i])
    #     controller.set_point(set_point[i])
    #     feedback[i] = controller(state[i])
    def __init__(self, gain_p, gain_i, gain_d, dt):
        self._kp = gain_p
        self._ki = gain_i
        self._kd = gain_d

        self._point = 0.0

        self._dt = dt

        self._accumulated_integral = 0.0
        self._integral_limits = np.array([-1., 1.])*1
        self._error_history = np.zeros((3,))

        self._derivator = second_order_fd
        self._derivative_limits = np.array([-1, 1])*0.05

        self._n_calls = 0

    def set_point(self, point):
        self._point = point

    def reset_integrator(self):
        self._accumulated_integral = 0.0

    def __call__(self, state):
        self._n_calls += 1
        actuation = 0.0
        error = self._point - state
        # displace previous errors one position to the left
        self._error_history = np.roll(self._error_history, -1)
        self._error_history[-1] = error

        # print(id(self))
        # Proportional gain
        actuation += error*self._kp
        # print('Error = ' + str(error*self._kp))

        # Derivative gain
        derivative = self._derivator(self._error_history, self._n_calls, self._dt)
        derivative = max(derivative, self._derivative_limits[0])
        derivative = min(derivative, self._derivative_limits[1])
        actuation += derivative*self._kd
        # print('derivaive = ' + str(derivative*self._kd))

        # Integral gain
        self._accumulated_integral += error*self._dt
        if self._accumulated_integral < self._integral_limits[0]:
            self._accumulated_integral = self._integral_limits[0]
            # cout.cout_wrap('Integrator in PID controller reached lower limit', 3)
        elif self._accumulated_integral > self._integral_limits[1]:
            self._accumulated_integral = self._integral_limits[1]
            # cout.cout_wrap('Integrator in PID controller reached upper limit', 3)
        # print('accumulated integral = ' + str(self._accumulated_integral*self._ki))

        actuation += self._accumulated_integral*self._ki

        return actuation


