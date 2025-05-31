"""Controller Utilities
"""
import numpy as np

def second_order_fd(history, n_calls, dt):
    # history is ordered such that the last element is the most recent one (t), and thus
    # it goes [(t - 2), (t - 1), (t)]
    coefficients = np.zeros((3,))
    if n_calls <= 1:
        # no derivative, return 0
        pass

    elif n_calls == 2:
    # else:
        # first order derivative
        coefficients[1:3] = [-1.0, 1.0]

    else:
        # # second order derivative
        coefficients[:] = [1.0, -4.0, 3.0]
        coefficients *= 0.5

    derivative = np.dot(coefficients, history)/dt
    return derivative

class PID(object):
    """
    Class implementing a classic PID controller

    Instance attributes:
    :param `gain_p`: Proportional gain.
    :param `gain_i`: Integral gain.
    :param `gain_d`: Derivative gain.
    :param `dt`: Simulation time step.

    The class should be used as:

        pid = PID(100, 10, 0.1, 0.1)
        pid.set_point(target_point)
        control = pid(current_point)
    """
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
        self._integral_limits = np.array([-1., 1.])*10000
        self._error_history = np.zeros((3,))

        self._derivator = second_order_fd
        self._derivative_limits = np.array([-1, 1])*10000

        self._n_calls = 0

        self._anti_windup_lim = None

    def set_point(self, point):
        self._point = point

    def set_anti_windup_lim(self, lim):
        self._anti_windup_lim = lim

    def reset_integrator(self):
        self._accumulated_integral = 0.0

    def __call__(self, state):
        self._n_calls += 1
        actuation = 0.0
        error = self._point - state
        # displace previous errors one position to the left
        self._error_history = np.roll(self._error_history, -1)
        self._error_history[-1] = error

        detailed = np.zeros((3,))
        # Proportional gain
        actuation += error*self._kp
        detailed[0] = error*self._kp

        # Derivative gain
        derivative = self._derivator(self._error_history, self._n_calls, self._dt)
        derivative = max(derivative, self._derivative_limits[0])
        derivative = min(derivative, self._derivative_limits[1])
        actuation += derivative*self._kd
        detailed[2] = derivative*self._kd

        # Integral gain
        aux_acc_int = self._accumulated_integral + error*self._dt
        if aux_acc_int < self._integral_limits[0]:
            aux_acc_int = self._integral_limits[0]
        elif aux_acc_int > self._integral_limits[1]:
            aux_acc_int = self._integral_limits[1]

        if self._anti_windup_lim is not None:
            # Apply anti wind up
            aux_actuation = actuation + aux_acc_int*self._ki
            if ((aux_actuation > self._anti_windup_lim[0]) and
                (aux_actuation < self._anti_windup_lim[1])):
                # Within limits
                self._accumulated_integral = aux_acc_int
                # If the system exceeds the limits, this 
                # will not be added to the self._accumulated_integral
        else:
            self._accumulated_integral = aux_acc_int

        actuation += self._accumulated_integral*self._ki
        detailed[1] = self._accumulated_integral*self._ki

        return actuation, detailed
