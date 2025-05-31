import numpy as np

import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings


@generator_interface.generator
class ShearVelocityField(generator_interface.BaseGenerator):
    r"""
    Shear Velocity Field Generator

    ``ShearVelocityField`` class inherited from ``BaseGenerator``

    The object creates a steady velocity field with shear

    .. math:: \hat{u} = \hat{u}\_\infty \left( \frac{h - h\_\mathrm{corr}}{h\_\mathrm{ref}} \right)^{\mathrm{shear}\_\mathrm{exp}}
    .. math:: h = \zeta \cdot \mathrm{shear}\_\mathrm{direction}

    """
    generator_id = 'ShearVelocityField'
    generator_classification = 'velocity-field'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['u_inf'] = 'float'
    settings_default['u_inf'] = None
    settings_description['u_inf'] = 'Free stream velocity magnitude'

    settings_types['u_inf_direction'] = 'list(float)'
    settings_default['u_inf_direction'] = np.array([1.0, 0, 0])
    settings_description['u_inf_direction'] = '``x``, ``y`` and ``z`` relative components of the free stream velocity'

    settings_types['shear_direction'] = 'list(float)'
    settings_default['shear_direction'] = np.array([.0, 0, 1.0])
    settings_description['shear_direction'] = '``x``, ``y`` and ``z`` relative components of the direction along which shear applies'

    settings_types['shear_exp'] = 'float'
    settings_default['shear_exp'] = 0.
    settings_description['shear_exp'] = 'Exponent of the shear law'

    settings_types['h_ref'] = 'float'
    settings_default['h_ref'] = 1.
    settings_description['h_ref'] = 'Reference height at which ``u_inf`` is defined'

    settings_types['h_corr'] = 'float'
    settings_default['h_corr'] = 0.
    settings_description['h_corr'] = 'Height to correct shear law'

    setting_table = settings.SettingsTable()
    __doc__ += setting_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.in_dict = dict()

        self.u_inf = 0.
        self.u_inf_direction = None
        self.shear_direction = None
        self.shear_exp = None
        self.h_ref = None
        self.h_corr = None

    def initialise(self, in_dict, restart=False):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict, self.settings_types, self.settings_default)

        self.u_inf = self.in_dict['u_inf']
        self.u_inf_direction = self.in_dict['u_inf_direction']
        self.shear_direction = self.in_dict['shear_direction']
        self.shear_exp = self.in_dict['shear_exp']
        self.h_ref = self.in_dict['h_ref']
        self.h_corr = self.in_dict['h_corr']

    def generate(self, params, uext):
        zeta = params['zeta']
        override = params['override']
        for i_surf in range(len(zeta)):
            if override:
                uext[i_surf].fill(0.0)
            for i in range(zeta[i_surf].shape[1]):
                for j in range(zeta[i_surf].shape[2]):
                    h = np.dot(zeta[i_surf][:, i, j], self.shear_direction) + self.h_corr
                    uext[i_surf][:, i, j] += self.u_inf*self.u_inf_direction*(h/self.h_ref)**self.shear_exp
