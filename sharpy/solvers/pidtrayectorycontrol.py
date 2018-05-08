import ctypes as ct

import numpy as np

import sharpy.utils.cout_utils as cout
import sharpy.utils.solver_interface as solver_interface
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra
# import sharpy.structure.utils.xbeamlib as xbeam


@solver
class PIDTrayectoryControl(BaseSolver):
    solver_id = 'PIDTrayectoryControl'

    def __init__(self):
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['print_info'] = 'bool'
        self.settings_default['print_info'] = True

        self.settings_types['trayectory_solver'] = 'str'
        self.settings_default['trayectory_solver'] = None

        self.settings_types['trayectory_solver_settings'] = 'dict'
        self.settings_default['trayectory_solver_settings'] = None

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

        self.settings_types['nodes_trayectory'] = 'list(int)'
        self.settings_default['nodes_trayectory'] = None

        self.data = None
        self.settings = None
        self.solver = None

        self.previous_force = None

        self.dt = 0.

        self.controllers = list()
        self.n_controlled_points = 0

        self.trayectory = None
        self.input_trayectory = None
        self.force_history = None

        self.predictor = False
        self.residual_table = None
        self.postprocessors = dict()
        self.with_postprocessors = False

    def initialise(self, data):
        self.data = data
        self.settings = data.settings[self.solver_id]
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)
        self.dt = self.settings['dt'].value

        self.solver = solver_interface.initialise_solver(self.settings['trayectory_solver'])
        self.settings['trayectory_solver_settings']['n_time_steps'] = 1
        self.solver.initialise(self.data, self.settings['trayectory_solver_settings'])

        self.n_controlled_points = len(self.settings['nodes_trayectory'])
        # initialized controllers for the trayectory nodes
        # there will be 3 controllers per node (one per dimension)
        for i_trayec in range(self.n_controlled_points):
            self.controllers.append(list())
            for i_dim in range(3):
                self.controllers[i_trayec].append(PID(self.settings['PID_P_gain'].value,
                                                      self.settings['PID_I_gain'].value,
                                                      self.settings['PID_D_gain'].value,
                                                      self.dt))

        self.trayectory = np.zeros((self.settings['n_time_steps'].value, len(self.settings['nodes_trayectory']), 3))
        self.input_trayectory = np.zeros((self.settings['n_time_steps'].value, len(self.settings['nodes_trayectory']), 3))
        self.force_history = np.zeros((self.settings['n_time_steps'].value, len(self.settings['nodes_trayectory']), 3))

    def run(self):
        local_it = -1
        for self.data.ts in range(len(self.data.structure.timestep_info) - 1,
                                  self.settings['n_time_steps'].value + len(self.data.structure.timestep_info) - 1):
            print(self.data.ts)
            local_it += 1
            if self.data.structure.dynamic_input[self.data.ts]['enforce_trayectory']:
                # get location of points
                self.trayectory[local_it, :, :] = self.extract_trayectory(self.settings['nodes_trayectory'])
                # get desired location of points
                self.input_trayectory[local_it, :, :] = (self.data.structure.dynamic_input[self.data.ts]
                                                                                          ['trayectories']
                                                                                          [self.settings['nodes_trayectory'], :])
                for inode in range(len(self.settings['nodes_trayectory'])):
                    self.force_history[local_it, inode, :] = self.calculate_forces(self.trayectory[local_it, inode, :],
                                                                                   self.input_trayectory[local_it, inode, :],
                                                                                   self.settings['nodes_trayectory'][inode],
                                                                                   inode)
                    # apply the forces now
                    (self.data.structure.dynamic_input[self.data.ts]
                        ['dynamic_forces']
                        [self.settings['nodes_trayectory'][inode], 0:3]) = self.force_history[local_it, inode, :]

            self.data = self.solver.run()

        return self.data

    def extract_trayectory(self, nodes, it=None):
        if it is None:
            it = self.data.ts

        trayectory = np.zeros((len(nodes), 3))
        coordinates = self.data.structure.timestep_info[it - 1].glob_pos()
        for inode in range(len(nodes)):
            trayectory[inode] = coordinates[nodes[inode], :]

        return trayectory

    def calculate_forces(self, trayectory, input_trayectory, i_global_node, i_trayectory_node):
        """
        This calculates the necessary response based on the current trayectory (trayectoy),
        the desired trayectory (input_trayectory) and the given node id.
        The output will be already converted to the material (b) FoR.
        """
        # for i in range(n_steps):
        #     state[i] = np.sum(feedback[:i])
        #     controller.set_point(set_point[i])
        #     feedback[i] = controller(state[i])
        force = np.zeros((3,))
        for idim in range(3):
            self.controllers[i_trayectory_node][idim].set_point(input_trayectory[idim])
            force[idim] = self.controllers[i_trayectory_node][idim](trayectory[idim])

        master_elem, i_local_node_master_elem = self.data.structure.node_master_elem[i_global_node, :]
        cba = algebra.crv2rot(
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

    elif n_calls == 2:
        # first order derivative
        coefficients[1:3] = [-1.0, 1.0]

    else:
        # second order derivative
        coefficients[:] = [1.0, -4.0, 3.0]

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
        self._integral_limits = np.array([-100.0, 100.0])
        self._error_history = np.zeros((3,))

        self.derivator = second_order_fd

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

        # Proportional gain
        actuation += error*self._kp

        # Derivative gain
        actuation += self.derivator(self._error_history, self._n_calls, self._dt)*self._kd

        # Integral gain
        self._accumulated_integral += error*self._dt
        if self._accumulated_integral < self._integral_limits[0]:
            self._accumulated_integral = self._integral_limits[0]
            cout.cout_wrap('Integrator in PID controller reached lower limit', 3)
        elif self._accumulated_integral > self._integral_limits[1]:
            self._accumulated_integral = self._integral_limits[1]
            cout.cout_wrap('Integrator in PID controller reached upper limit', 3)

        actuation += self._accumulated_integral*self._ki

        return actuation


