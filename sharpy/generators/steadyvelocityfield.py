import numpy as np

import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings


@generator_interface.generator
class SteadyVelocityField(generator_interface.BaseGenerator):
    generator_id = 'SteadyVelocityField'

    def __init__(self):
        self.in_dict = dict()
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['u_inf'] = 'float'
        self.settings_default['u_inf'] = None

        self.settings_types['u_inf_direction'] = 'list(float)'
        self.settings_default['u_inf_direction'] = np.array([1.0, 0, 0])

        self.u_inf = 0.
        self.u_inf_direction = None

    def initialise(self, in_dict):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict, self.settings_types, self.settings_default)

        self.u_inf = self.in_dict['u_inf']
        self.u_inf_direction = self.in_dict['u_inf_direction']

    def generate(self, params, uext):
        zeta = params['zeta']
        override = params['override']
        for i_surf in range(len(zeta)):
            if override:
                uext[i_surf].fill(0.0)
            for i in range(zeta[i_surf].shape[1]):
                for j in range(zeta[i_surf].shape[2]):
                    uext[i_surf][:, i, j] += self.u_inf*self.u_inf_direction
