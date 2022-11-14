import numpy as np

import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings
import sharpy.utils.exceptions as exc


@generator_interface.generator
class BumpVelocityField(generator_interface.BaseGenerator):
    r"""
    Bump Velocity Field Generator

    ``BumpVelocityField`` is a class inherited from ``BaseGenerator``

    The ``BumpVelocityField`` class generates a bump-shaped gust profile velocity field, and the profile has the characteristics
    specified by the user.

    To call this generator, the ``generator_id = BumpVelocityField`` shall be used.
    This is parsed as the value for the ``velocity_field_generator`` key in the desired aerodynamic solver's settings.

    The resultant velocity, $w_g$, is calculated as follows:

    .. math::

        w_g = \frac{w_0}{4}\left( 1 + \cos(\frac{(x - x_0)}{H_x} \right)\left( 1 + \cos(\frac{(y - y_0)}{H_y} \right)


    Notes:
        For now, only simulations where the inertial FoR is fixed are supported.

    """
    generator_id = 'BumpVelocityField'
    generator_classification = 'velocity-field'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['gust_intensity'] = 'float'
    settings_default['gust_intensity'] = None
    settings_description['gust_intensity'] = 'Intensity of the gust'

    settings_types['x0'] = 'float'
    settings_default['x0'] = 0.0
    settings_description['x0'] = 'x location of the centre of the bump'

    settings_types['y0'] = 'float'
    settings_default['y0'] = 0.0
    settings_description['y0'] = 'y location of the centre of the bump'

    settings_types['hx'] = 'float'
    settings_default['hx'] = 1.
    settings_description['hx'] = 'Gust gradient in the x direction'

    settings_types['hy'] = 'float'
    settings_default['hy'] = 1.
    settings_description['hy'] = 'Gust gradient in the y direction'

    settings_types['relative_motion'] = 'bool'
    settings_default['relative_motion'] = False
    settings_description['relative_motion'] = 'When true the gust will move at the prescribed velocity'

    settings_types['u_inf'] = 'float'
    settings_default['u_inf'] = None
    settings_description['u_inf'] = 'Free stream velocity'

    settings_types['u_inf_direction'] = 'list(float)'
    settings_default['u_inf_direction'] = np.array([1.0, 0, 0])
    settings_description['u_inf_direction'] = 'Free stream velocity direction'

    table = settings.SettingsTable()
    __doc__ += table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.in_dict = dict()
        self.settings = dict()

        self.u_inf = 0.
        self.u_inf_direction = None

    def initialise(self, in_dict, restart=False):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict, BumpVelocityField.settings_types, BumpVelocityField.settings_default)
        self.settings = self.in_dict

        self.u_inf = self.settings['u_inf']
        self.u_inf_direction = self.in_dict['u_inf_direction']

    def generate(self, params, uext):
        zeta = params['zeta']
        override = params['override']
        for_pos = params['for_pos']
        t = params['t']

        def gust_shape(x, y, z, hx, hy, x0, y0, w0):
            vel = np.zeros((3,))
            if np.abs(x - x0) > hx or np.abs(y - y0) > hy:
                return vel

            vel[2] = 0.25*w0*(1 + np.cos((x - x0)/hx * np.pi))*(1 + np.cos((y - y0)/hy * np.pi))
            return vel

        for i_surf in range(len(zeta)):
            if override:
                uext[i_surf].fill(0.0)

            for i in range(zeta[i_surf].shape[1]):
                for j in range(zeta[i_surf].shape[2]):
                    uext[i_surf][:, i, j] += gust_shape(zeta[i_surf][0, i, j] + for_pos[0],
                                                        zeta[i_surf][1, i, j] + for_pos[1],
                                                        zeta[i_surf][2, i, j] + for_pos[2],
                                                        self.settings['hx'],
                                                        self.settings['hy'],
                                                        self.settings['x0'],
                                                        self.settings['y0'],
                                                        self.settings['gust_intensity'])

                    if self.settings['relative_motion']:
                        uext[i_surf][:, i, j] += self.u_inf*t
