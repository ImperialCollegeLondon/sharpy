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

        self.settings_types['coupled_solver'] = 'str'
        self.settings_default['coupled_solver'] = None

        self.settings_types['coupled_solver_settings'] = 'dict'
        self.settings_default['coupled_solver_settings'] = None

        self.settings_types['n_time_steps'] = 'int'
        self.settings_default['n_time_steps'] = 100

        self.settings_types['dt'] = 'float'
        self.settings_default['dt'] = 0.05

        self.settings_types['postprocessors'] = 'list(str)'
        self.settings_default['postprocessors'] = list()

        self.settings_types['postprocessors_settings'] = 'dict'
        self.settings_default['postprocessors_settings'] = dict()

        self.settings_types['PID_P_gain'] = 'float'
        self.settings_default['PID_P_gain'] = 1000.

        self.settings_types['PID_I_gain'] = 'float'
        self.settings_default['PID_I_gain'] = 1000.

        self.settings_types['PID_D_gain'] = 'float'
        self.settings_default['PID_D_gain'] = 10.

        self.data = None
        self.settings = None
        self.structural_solver = None
        self.aero_solver = None

        self.previous_force = None

        self.dt = 0.

        self.predictor = False
        self.residual_table = None
        self.postprocessors = dict()
        self.with_postprocessors = False

    def initialise(self, data):
        self.data = data
        self.settings = data.settings[self.solver_id]
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

    def run(self):
        return self.data


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


