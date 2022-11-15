import numpy as np
from numpy.polynomial import polynomial as P
import scipy as sc
from scipy import interpolate

import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings
import sharpy.utils.cout_utils as cout


@generator_interface.generator
class TrajectoryGenerator(generator_interface.BaseGenerator):
    """
    ``TrajectoryGenerator`` is used to generate nodal positions or velocities
    for trajectory constraints such as the ones included in the multibody
    solver.

    It is usually called from a ``Controller`` module.
    """
    generator_id = 'TrajectoryGenerator'
    generator_classification = 'utils'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['angle_end'] = 'float'
    settings_default['angle_end'] = 0.0
    settings_description['angle_end'] = 'Trajectory angle wrt horizontal at release'

    settings_types['veloc_end'] = 'float'
    settings_default['veloc_end'] = None
    settings_description['veloc_end'] = 'Release velocity at release'

    settings_types['shape'] = 'str'
    settings_default['shape'] = 'quadratic'
    settings_description['shape'] = 'Shape of the ``z`` vs ``x`` function. ``quadratic`` or ``linear`` are supported'

    settings_types['acceleration'] = 'str'
    settings_default['acceleration'] = 'linear'
    settings_description['acceleration'] = 'Acceleration law, possible values are ``linear`` or ``constant``'

    settings_types['dt'] = 'float'
    settings_default['dt'] = None
    settings_description['dt'] = 'Time step of the simulation'

    settings_types['coords_end'] = 'list(float)'
    settings_default['coords_end'] = None
    settings_description['coords_end'] = 'Coordinates of the final ramp point'

    settings_types['plot'] = 'bool'
    settings_default['plot'] = False
    settings_description['plot'] = 'Plot the ramp shape. Requires matplotlib installed'

    settings_types['print_info'] = 'bool'
    settings_default['print_info'] = False
    settings_description['print_info'] = 'Print information on runtime'

    settings_types['time_offset'] = 'float'
    settings_default['time_offset'] = 0.0
    settings_description['time_offset'] = 'Time interval before the start of the ramp acceleration'

    settings_types['offset'] = 'list(float)'
    settings_default['offset'] = np.zeros((3,))
    settings_description['offset'] = 'Coordinates of the starting point of the simulation'

    settings_types['return_velocity'] = 'bool'
    settings_default['return_velocity'] = False
    settings_description['return_velocity'] = 'If ``True``, nodal velocities are given, if ``False``, coordinates are the output'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.in_dict = dict()

        self.x_vec = None
        self.y_vec = None
        self.x_dot_vec = None
        self.y_dot_vec = None
        self.time = None
        self.n_steps = None
        self.travel_time = None
        self.curve_length = None

        self.implemented_shapes = ["linear", "quadratic"]
        self.implemented_accelerations = ["constant", "linear"]


    def initialise(self, in_dict, restart=False):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict, self.settings_types, self.settings_default)

        # input validation
        self.in_dict['shape'] = self.in_dict['shape'].lower()
        self.in_dict['acceleration'] = self.in_dict['acceleration'].lower()

        self.calculate_trajectory()
        if self.in_dict['print_info']:
            self.print_info()

        if self.in_dict['plot']:
            self.plot()

        if self.in_dict['return_velocity']:
            self.diff_trajectory()

    def print_info(self):
        cout.cout_wrap('Trajectory information:', 2)
        cout.cout_wrap('\t-travel time: {}'.format(self.travel_time), 2)
        cout.cout_wrap('\t-curve length: {}'.format(self.curve_length), 2)
        cout.cout_wrap('-------------------------------', 2)

    def plot(self):
        try:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.scatter(self.x_vec, self.y_vec, c=self.time_vec)
            plt.colorbar(orientation='horizontal')
            plt.axis('equal')
            plt.show()
        except ModuleNotFoundError:
            import warnings
            warnings.warn('Unable to import matplotlib, skipping plot')

    def get_n_steps(self):
        return self.n_steps

    def __call__(self, params):
        return self.generate(params)

    def generate(self, params):
        it = params['it']
        if self.n_steps is not None:
            if it >= self.n_steps:
                return [None]*3

        if self.in_dict['return_velocity']:
            return np.array([self.x_dot_vec[it], 0.0, self.y_dot_vec[it]]) + self.in_dict['offset']
        else:
            return np.array([self.x_vec[it], 0.0, self.y_vec[it]]) + self.in_dict['offset']

    def calculate_trajectory(self):
        in_dict = self.in_dict
        # shape variables
        shape_polynomial = np.zeros((3,))  # [c, b, a] results in y = c + b*x + a*x^2
        curve_length = 0.0

        # calculate coefficients and curve length
        if in_dict['shape'] == "linear":
            shape_polynomial[0] = 0.0
            shape_polynomial[1] = in_dict['coords_end'][1]/in_dict['coords_end'][0]
            if np.isnan(shape_polynomial[1]):
                shape_polynomial[1] = 0.0
            shape_polynomial[2] = 0.0
            curve_length = linear_curve_length(shape_polynomial, in_dict['coords_end'])

        elif in_dict['shape'] == "quadratic":
            shape_polynomial[2] = (np.arctan(in_dict['angle_end'].value) - in_dict['coords_end'][1]/in_dict['coords_end'][0])/in_dict['coords_end'][0]
            shape_polynomial[1] = np.arctan(in_dict['angle_end'].value) - 2.0*shape_polynomial[2]*in_dict['coords_end'][0]
            shape_polynomial[0] = 0.0
            curve_length = quadratic_curve_length(shape_polynomial, in_dict['coords_end'])

        curve_length = np.abs(curve_length)

        # acceleration
        acceleration_position_coefficients = None
        travel_time = 0.0
        if in_dict['acceleration'] == 'constant':
            acceleration_position_coefficients = constant_acceleration_position_coeffs(curve_length, np.abs(in_dict['veloc_end'].value))
            travel_time = constant_acceleration_travel_time(curve_length,
                                                            np.abs(in_dict['veloc_end'].value))
        elif in_dict['acceleration'] == 'linear':
            acceleration_position_coefficients = linear_acceleration_position_coeffs(curve_length, np.abs(in_dict['veloc_end'].value))
            travel_time = linear_acceleration_travel_time(curve_length,
                                                          np.abs(in_dict['veloc_end'].value))

        # time
        n_steps_travel = int(round(travel_time/in_dict['dt'].value))
        n_steps_offset = int(round(self.in_dict['time_offset'].value/in_dict['dt'].value))
        self.n_steps = n_steps_offset + n_steps_travel
        self.time_vec = np.zeros((self.n_steps, ))
        self.time_vec[n_steps_offset:] = np.linspace(0.0, n_steps_travel*in_dict['dt'].value, n_steps_travel)

        # with t I get s
        s_vec = P.polyval(self.time_vec, acceleration_position_coefficients)
        # need to get a function for x(s)
        # sample s(x) and create an interpolator
        n_samples = 1000
        sampled_x_vec = np.linspace(0.0, in_dict['coords_end'][0], n_samples)
        sampled_s_vec = np.zeros((n_samples, ))
        for i_sample in range(n_samples):
            if in_dict['shape'] == "linear":
                sampled_s_vec[i_sample] = np.abs(linear_curve_length(shape_polynomial, np.array([sampled_x_vec[i_sample], 0.0])))
            elif in_dict['shape'] == 'quadratic':
                sampled_s_vec[i_sample] = np.abs(quadratic_curve_length(shape_polynomial, np.array([sampled_x_vec[i_sample], 0.0])))
        x_of_s_interp = sc.interpolate.interp1d(sampled_s_vec, sampled_x_vec, kind='quadratic', fill_value='extrapolate')
        self.x_vec = x_of_s_interp(s_vec)

        # with x, I obtain y and done
        self.y_vec = P.polyval(self.x_vec, shape_polynomial)
        self.travel_time = travel_time
        self.curve_length = curve_length

    def diff_trajectory(self):
        dt = self.in_dict['dt'].value

        self.x_dot_vec = np.gradient(self.x_vec, dt)
        self.y_dot_vec = np.gradient(self.y_vec, dt)


def linear_curve_length(shape_polynomial, coords_end):
    dzdx_end = shape_polynomial[1]

    length = coords_end[0]*np.sqrt(dzdx_end**2 + 1)
    return length


def quadratic_curve_length(shape_polynomial, coords_end):
    dzdx_end = 2.0*shape_polynomial[2]*coords_end[0] + shape_polynomial[1]
    a = shape_polynomial[2]
    b = shape_polynomial[1]
    xe = coords_end[0]
    length = (2.0*a*xe + b)*np.sqrt((2.0*a*xe + b)**2 + 1.0)
    length += np.arcsinh(2.0*a*xe + b)
    length -= b*np.sqrt(b**2 + 1.0)
    length -= np.arcsinh(b)
    length /= 4.0*a

    return length


def constant_acceleration_position_coeffs(s_e, s_dot_e):
    coeffs = (0.0, 0.0, (4.0*s_dot_e**2)/s_e, 0.0)
    return coeffs


def linear_acceleration_position_coeffs(s_e, s_dot_e):
    coeff = ((6.0*s_e)**(2./3.)/(2.0*s_dot_e))**(-3.)
    coeffs = (0.0, 0.0, 0.0, (1./6.)*coeff)
    return coeffs


def constant_acceleration_travel_time(s_e, s_dot_e):
    return 2.0*s_e/s_dot_e


def linear_acceleration_travel_time(s_e, s_dot_e):
    return 3.0*s_e/s_dot_e
