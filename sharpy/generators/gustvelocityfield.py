import numpy as np

import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings
import sharpy.utils.exceptions as exc


@generator_interface.generator
class GustVelocityField(generator_interface.BaseGenerator):
    generator_id = 'GustVelocityField'

    def __init__(self):
        self.in_dict = dict()
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['u_inf'] = 'float'
        self.settings_default['u_inf'] = None

        self.settings_types['u_inf_direction'] = 'list(float)'
        self.settings_default['u_inf_direction'] = np.array([1.0, 0, 0])

        self.settings_types['gust_shape'] = 'str'
        self.settings_default['gust_shape'] = None

        self.settings_types['gust_length'] = 'float'
        self.settings_default['gust_length'] = None

        self.settings_types['gust_intensity'] = 'float'
        self.settings_default['gust_intensity'] = None

        self.settings_types['offset'] = 'float'
        self.settings_default['offset'] = 0.0

        self.settings_types['span'] = 'float'
        self.settings_default['span'] = 0.

        self.u_inf = 0.
        self.u_inf_direction = None

        self.implemented_gusts = []
        self.implemented_gusts.append('1-cos')
        self.implemented_gusts.append('DARPA')

        self.settings = dict()

    def initialise(self, in_dict):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict, self.settings_types, self.settings_default)
        self.settings = self.in_dict

        # check that the gust type is valid
        if not (self.settings['gust_shape'] in self.implemented_gusts):
            raise AttributeError('The gust shape ' + self.settings['gust_shape'] + ' is not implemented')

        self.u_inf = self.in_dict['u_inf'].value
        self.u_inf_direction = self.in_dict['u_inf_direction']

    def generate(self, params, uext):
        zeta = params['zeta']
        override = params['override']
        ts = params['ts']
        dt = params['dt']
        t = params['t']
        gust_shape = None
        if self.settings['gust_shape'] == '1-cos':
            def gust_shape(x, y, z, gust_length, gust_intensity, span=0):
                vel = np.zeros((3,))
                if (x > 0.0 or x < -gust_length):
                    return vel

                vel[2] = (1.0 - np.cos(2.0*np.pi*x/gust_length))*gust_intensity*0.5
                return vel
        elif self.settings['gust_shape'] == 'DARPA':
            def gust_shape(x, y, z, gust_length, gust_intensity, span=0):
                vel = np.zeros((3,))
                if (x > 0.0 or x < -gust_length):
                    return vel

                vel[2] = (1.0 - np.cos(2.0*np.pi*x/gust_length))*gust_intensity*0.5
                vel[2] *= -np.cos(y/span*np.pi)
                return vel

        for i_surf in range(len(zeta)):
            if override:
                uext[i_surf].fill(0.0)

            for i in range(zeta[i_surf].shape[1]):
                for j in range(zeta[i_surf].shape[2]):
                    uext[i_surf][:, i, j] += self.u_inf*self.u_inf_direction
                    uext[i_surf][:, i, j] += gust_shape(zeta[i_surf][0, i, j] - self.u_inf*t + self.settings['offset'],
                                                        zeta[i_surf][1, i, j],
                                                        zeta[i_surf][2, i, j],
                                                        self.settings['gust_length'].value,
                                                        self.settings['gust_intensity'].value,
                                                        self.settings['span'].value
                                                        )
