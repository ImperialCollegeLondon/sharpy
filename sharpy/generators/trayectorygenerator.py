import numpy as np
from numpy.polynomial import polynomial as P
import scipy as sc
from scipy import interpolate
import matplotlib.pyplot as plt

import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings


@generator_interface.generator
class TrayectoryGenerator(generator_interface.BaseGenerator):
    generator_id = 'TrayectoryGenerator'

    def __init__(self):
        self.in_dict = dict()
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['slope_end'] = 'float'
        self.settings_default['slope_end'] = 0.0

        self.settings_types['veloc_end'] = 'float'
        self.settings_default['veloc_end'] = None

        self.settings_types['shape'] = 'str'
        self.settings_default['shape'] = 'quadratic'

        self.settings_types['acceleration'] = 'str'
        self.settings_default['acceleration'] = 'linear'

        self.settings_types['dt'] = 'float'
        self.settings_default['dt'] = None

        self.settings_types['coords_end'] = 'list(float)'
        self.settings_default['coords_end'] = None

        self.settings_types['plot'] = 'bool'
        self.settings_default['plot'] = False

        self.implemented_shapes = ["linear", "quadratic"]
        self.implemented_accelerations = ["constant", "linear"]

    def initialise(self, in_dict):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict, self.settings_types, self.settings_default)

        # input validation
        self.in_dict['shape'] = self.in_dict['shape'].lower()
        self.in_dict['acceleration'] = self.in_dict['acceleration'].lower()

    def generate(self, params, trayectory):
        in_dict = self.in_dict
        # shape variables
        shape_polynomial = np.zeros((3,))  # [c, b, a] results in y = c + b*x + a*x^2
        curve_length = 0.0

        # calculate coefficients and curve length
        if in_dict['shape'] == "linear":
            shape_polynomial[0] = 0.0
            shape_polynomial[1] = in_dict['coords_end'][1]/in_dict['coords_end'][0]
            shape_polynomial[2] = 0.0
            curve_length = linear_curve_length(shape_polynomial, in_dict['coords_end'])

        elif in_dict['shape'] == "quadratic":
            shape_polynomial[2] = (np.arctan(in_dict['slope_end']) - in_dict['coords_end'][1]/in_dict['coords_end'][0])/in_dict['coords_end'][0]
            shape_polynomial[1] = np.arctan(in_dict['slope_end']) - 2.0*shape_polynomial[2]*in_dict['coords_end'][0]
            shape_polynomial[0] = 0.0
            curve_length = quadratic_curve_length(shape_polynomial, in_dict['coords_end'])


        # acceleration
        acceleration_position_coefficients = None
        travel_time = 0.0
        if in_dict['acceleration'] == 'constant':
            acceleration_position_coefficients = constant_acceleration_position_coeffs(curve_length, in_dict['veloc_end'])
            travel_time = constant_acceleration_travel_time(acceleration_position_coefficients,
                                                            curve_length)
        elif in_dict['acceleration'] == 'linear':
            acceleration_position_coefficients = linear_acceleration_position_coeffs(curve_length, in_dict['veloc_end'])
            travel_time = linear_acceleration_travel_time(acceleration_position_coefficients,
                                                          curve_length)

        # we need to map x vs s (arc length parameter)


        # time
        n_steps = round(travel_time/in_dict['dt'])
        # time_vec = np.linspace(0.0, travel_time, n_steps)
        time_vec = np.linspace(0.0, n_steps*in_dict['dt'], n_steps)
        print(time_vec[-1], travel_time)

        # with t I get s
        s_vec = P.polyval(time_vec, acceleration_position_coefficients)
        # need to get a function for x(s)
        # sample s(x) and create an interpolator
        n_samples = 1000
        sampled_x_vec = np.linspace(0.0, in_dict['coords_end'][0], n_samples)
        sampled_s_vec = np.zeros((n_samples, ))
        for i_sample in range(n_samples):
            if in_dict['shape'] == "linear":
                sampled_s_vec[i_sample] = linear_curve_length(shape_polynomial, np.array([sampled_x_vec[i_sample], 0.0]))
            elif in_dict['shape'] == 'quadratic':
                sampled_s_vec[i_sample] = quadratic_curve_length(shape_polynomial, np.array([sampled_x_vec[i_sample], 0.0]))
        x_of_s_interp = sc.interpolate.interp1d(sampled_s_vec, sampled_x_vec, kind='quadratic', fill_value='extrapolate')
        x_vec = x_of_s_interp(s_vec)

        # with x, I obtain y and done
        y_vec = P.polyval(x_vec, shape_polynomial)

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
    coeffs = (0.0, 0.0, 0.5*(0.5*4.0*s_dot_e**2)/s_e, 0.0)
    return coeffs


def linear_acceleration_position_coeffs(s_e, s_dot_e):
    # coeffs = (0.0, 0.0, 0.0, ((6.0*s_e)**1.5/s_dot_e)**2/6.)
    # coeffs = (0.0, 0.0, 0.0, ((6.0*s_e)**(2./3.)/s_dot_e)**2/6.)
    coeff = ((2.*s_dot_e)**1.5 * 6.*s_e)**(2./5.)
    coeffs = (0.0, 0.0, 0.0, (1./6.)**2*coeff)
    return coeffs


def constant_acceleration_travel_time(pos_coeffs, s_e):
    # return np.sqrt(2.0*s_e/pos_coeffs[2])
    return np.sqrt(1.0*s_e/pos_coeffs[2])


def linear_acceleration_travel_time(pos_coeffs, s_e):
    # return np.cbrt(6.0*s_e/pos_coeffs[3])
    return np.cbrt(s_e/pos_coeffs[3])

