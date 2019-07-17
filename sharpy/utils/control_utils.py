import numpy as np

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
        derivative = self._derivator(self._error_history, self._n_calls, self._dt)
        derivative = max(derivative, self._derivative_limits[0])
        derivative = min(derivative, self._derivative_limits[1])
        actuation += derivative*self._kd

        # Integral gain
        self._accumulated_integral += error*self._dt
        if self._accumulated_integral < self._integral_limits[0]:
            self._accumulated_integral = self._integral_limits[0]
        elif self._accumulated_integral > self._integral_limits[1]:
            self._accumulated_integral = self._integral_limits[1]

        actuation += self._accumulated_integral*self._ki

        return actuation
