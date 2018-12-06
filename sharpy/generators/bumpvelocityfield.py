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
    $$
    w_g = \frac{w_0}{4}\left( 1 + \cos(\frac{(x - x_0)}{H_x} \right)\left( 1 + \cos(\frac{(y - y_0)}{H_y} \right)
    $$

    Notes:
        For now, only simulations where the inertial FoR is fixed are supported.

    Args:
        in_dict (dict): Input data in the form of dictionary. See acceptable entries below:

        ===================  ===============  ==================================================================  ===================
        Name                 Type             Description                                                         Default
        ===================  ===============  ==================================================================  ===================
        ``gust_intensity``   ``float``        Intensity of the gust                                               ``None``
        ``x0``               ``float``        x location of the center of the bump                                ``0.0``
        ``y0``               ``float``        x location of the center of the bump                                ``0.0``
        ``hx``               ``float``        Gust gradient in the X direction                                    ``1.0``
        ``hy``               ``float``        Gust gradient in the Y direction                                    ``1.0``
        ===================  ===============  ==================================================================  ===================

    Attributes:

    See Also:
        .. py:class:: sharpy.utils.generator_interface.BaseGenerator

    """
    generator_id = 'BumpVelocityField'

    def __init__(self):
        self.in_dict = dict()
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['gust_intensity'] = 'float'
        self.settings_default['gust_intensity'] = None

        self.settings_types['x0'] = 'float'
        self.settings_default['x0'] = 0.0

        self.settings_types['y0'] = 'float'
        self.settings_default['y0'] = 0.0

        self.settings_types['hx'] = 'float'
        self.settings_default['hx'] = 0.

        self.settings_types['hy'] = 'float'
        self.settings_default['hy'] = 0.

        self.settings = dict()

    def initialise(self, in_dict):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict, self.settings_types, self.settings_default)
        self.settings = self.in_dict

    def generate(self, params, uext):
        zeta = params['zeta']
        override = params['override']
        for_pos = params['for_pos']
        gust_shape = None
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
                                                        self.settings['hx'].value,
                                                        self.settings['hy'].value,
                                                        self.settings['x0'].value,
                                                        self.settings['y0'].value,
                                                        self.settings['gust_intensity'].value)
