class PID_Controller(object):
    def __init__(self, P_gain, I_gain, D_gain, target=0):
        self.P_gain = P_gain
        self.I_gain = I_gain
        self.D_gain = D_gain
        self.target = target

        self.error = 0.
        self.integral_error = 0.
        self.derivative_error = 0.
        self.previous_error = 0.

    def generate_control_input(self, sensor_measurement, dt):
        self.error = self.target - sensor_measurement
        self.integral_error += self.error * dt        
        self.derivative_error = (self.error - self.previous_error) * dt
        self.previous_error = self.error
        
        return self.P_gain * self.error + self.I_gain * self.integral_error + self.derivative_error * self.D_gain