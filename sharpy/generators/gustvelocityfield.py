import numpy as np

import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings
import sharpy.utils.exceptions as exc


@generator_interface.generator
class GustVelocityField(generator_interface.BaseGenerator):
    r"""
    Gust Velocity Field Generator

    ``GustVelocityField`` is a class inherited from ``BaseGenerator``

    The ``GustVelocityField`` class generates a gust profile velocity field, and the profile has the characteristics
    specified by the user.

    To call this generator, the ``generator_id = GustVelocityField`` shall be used.
    This is parsed as the value for the ``velocity_field_generator`` key in the desired aerodynamic solver's settings.

    Supported gusts:
        - 1-cos: Discrte gust model
            .. math:: U_z = \frac{u_{de}}{2}\left[1-\cos\left(\frac{2\pi x}{S}\right)\right]

        - DARPA: Discrete, non-uniform span model
            .. math:: U_z = \frac{u_{de}}{2}\left[1-\cos\left(\frac{2\pi x}{S}\right)\right]\cos\left(\frac{\pi y}{b}\right)

        - continuous_sin: Continuous sinusoidal gust model
            .. math:: U_z = \frac{u_{de}}{2}\sin\left(\frac{2\pi x}{S}\right)

        - time varying: The inflow velocity changes with time but it is uniform in space. It is read from a 4 column file:
            .. math:: time[s] \Delta U_x \Delta U_y \Delta U_z

        - time varying global: Similar to the previous one but the velocity changes instanteneously in the whole flow field. It is not fed into the solid

    where, :math:`u_{de}` is the gust intensity, :math:`S` is the gust length and :math:`b` is the wing span.
    :math:`x` and :math:`y` refer to the chordwise and spanwise distance penetrated into the gust, respectively.

    Args:
        in_dict (dict): Input data in the form of dictionary. See acceptable entries below:

            ===================  ===============  =================================================  ===================
            Name                 Type             Description                                        Default
            ===================  ===============  =================================================  ===================
            ``u_inf``            ``float``        Free stream velocity                               ``0.0``
            ``u_inf_direction``  ``list(float)``  Free stream velocity relative component            ``[1.0, 0.0, 0.0]``
            ``gust_shape``       ``str``          Gust profile shape.                                ``None``
            ``gust_length``      ``float``        Length of gust                                     ``0.0``
            ``gust_intensity``   ``float``        Intensity of the gust                              ``0.0``
            ``offset``           ``float``        Spatial offset of the gust with respect to origin  ``0.0``
            ``span``             ``float``        Wing span                                          ``0.0``
            ``file``             ``str``          File with the information (only for time varying)  ``Empty string``
            ``relative_motion``  ``bool``         If true, the gust is convected with u_inf          ``False``
            ===================  ===============  =================================================  ===================

    Attributes:
        settings_types (dict): Acceptable data types of the input data
        settings_default (dict): Default values for input data should the user not provide them
        u_inf (float): Free stream velocity
        u_inf_direction (list(float)): Free stream velocity relative components in ``x`, ``y`` and ``z``
        gust_shape (str): Gust profile shape
        gust_length (float): Length of gust, usually noted as $2H$
        gust_intensity (float): Intensity of the gust, in m/s
        offset (float): Spatial offset of the gust position with respect to origin
        span (float): Wing span
        implemented_gusts (list(str)): Currently supported gust profiles
        file_info (np.array): gust information read from the text file
        gust_shape (function): interpolates the velocity on a point according to the gust type

    See Also:
        .. py:class:: sharpy.utils.generator_interface.BaseGenerator

    """
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
        self.settings_default['gust_length'] = 0.0

        self.settings_types['gust_intensity'] = 'float'
        self.settings_default['gust_intensity'] = 0.0

        self.settings_types['offset'] = 'float'
        self.settings_default['offset'] = 0.0

        self.settings_types['span'] = 'float'
        self.settings_default['span'] = 0.

        self.settings_types['file'] = 'str'
        self.settings_default['file'] = ''

        self.settings_types['relative_motion'] = 'bool'
        self.settings_default['relative_motion'] = False

        self.u_inf = 0.
        self.u_inf_direction = None

        self.file_info = None

        self.implemented_gusts = []
        self.implemented_gusts.append('1-cos')
        self.implemented_gusts.append('DARPA')
        self.implemented_gusts.append('continuous_sin')
        self.implemented_gusts.append('lateral 1-cos')
        self.implemented_gusts.append('time varying')
        self.implemented_gusts.append('time varying global')

        self.settings = dict()

        self.gust_shape = None

    def initialise(self, in_dict):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict, self.settings_types, self.settings_default)
        self.settings = self.in_dict

        # check that the gust type is valid
        if not (self.settings['gust_shape'] in self.implemented_gusts):
            raise AttributeError('The gust shape ' + self.settings['gust_shape'] + ' is not implemented')

        self.u_inf = self.in_dict['u_inf'].value
        self.u_inf_direction = self.in_dict['u_inf_direction']

        if self.settings['gust_shape'] == 'time varying':
            self.file_info = np.loadtxt(self.settings['file'])
        elif self.settings['gust_shape'] == 'time varying global':
            self.file_info = np.loadtxt(self.settings['file'])

        # print(self.file_info)

        # Create the adequate gust interpolator
        # gust_shape = None
        if self.settings['gust_shape'] == '1-cos':
            def gust_shape(x, y, z, time=0):
                gust_length = self.settings['gust_length'].value
                gust_intensity = self.settings['gust_intensity'].value

                vel = np.zeros((3,))
                if x > 0.0 or x < -gust_length:
                    return vel

                vel[2] = (1.0 - np.cos(2.0*np.pi*x/gust_length))*gust_intensity*0.5
                return vel
        elif self.settings['gust_shape'] == 'DARPA':
            def gust_shape(x, y, z, time=0):
                gust_length = self.settings['gust_length'].value
                gust_intensity = self.settings['gust_intensity'].value
                span = self.settings['span'].value

                vel = np.zeros((3,))
                if x > 0.0 or x < -gust_length:
                    return vel

                vel[2] = (1.0 - np.cos(2.0*np.pi*x/gust_length))*gust_intensity*0.5
                vel[2] *= -np.cos(y/span*np.pi)
                return vel
        elif self.settings['gust_shape'] == 'continuous_sin':
            def gust_shape(x, y, z, time=0):
                gust_length = self.settings['gust_length'].value
                gust_intensity = self.settings['gust_intensity'].value

                vel = np.zeros((3,))
                if x > 0.0:
                    return vel

                vel[2] = 0.5 * gust_intensity * np.sin(2 * np.pi * x / gust_length)
                return vel
        elif self.settings['gust_shape'] == 'lateral 1-cos':
            def gust_shape(x, y, z, time=0):
                gust_length = self.settings['gust_length'].value
                gust_intensity = self.settings['gust_intensity'].value

                vel = np.zeros((3,))
                if x > 0.0 or x < -gust_length:
                    return vel

                vel[1] = (1.0 - np.cos(2.0*np.pi*x/gust_length))*gust_intensity*0.5
                return vel
        elif self.settings['gust_shape'] == 'time varying':
            def gust_shape(x, y, z, time=0):
                vel = np.zeros((3,))
                d = np.dot(np.array([x, y, z]), self.settings['u_inf_direction'])
                if d > 0.0:
                    return vel

                vel[0] = np.interp(d, -self.file_info[::-1,0]*self.u_inf, self.file_info[::-1,1])
                vel[1] = np.interp(d, -self.file_info[::-1,0]*self.u_inf, self.file_info[::-1,2])
                vel[2] = np.interp(d, -self.file_info[::-1,0]*self.u_inf, self.file_info[::-1,3])
                return vel

        elif self.settings['gust_shape'] == 'time varying global':
            def gust_shape(x, y, z, time=0):
                vel = np.zeros((3,))

                vel[0] = np.interp(time, self.file_info[:,0], self.file_info[:,1])
                vel[1] = np.interp(time, self.file_info[:,0], self.file_info[:,2])
                vel[2] = np.interp(time, self.file_info[:,0], self.file_info[:,3])
                return vel

        self.gust_shape = gust_shape

    def generate(self, params, uext):
        zeta = params['zeta']
        override = params['override']
        ts = params['ts']
        dt = params['dt']
        t = params['t']
        for_pos = params['for_pos'][0:3]

        for i_surf in range(len(zeta)):
            if override:
                uext[i_surf].fill(0.0)

            for i in range(zeta[i_surf].shape[1]):
                for j in range(zeta[i_surf].shape[2]):
                    total_offset_val = self.settings['offset'].value
                    if self.settings['relative_motion']:
                        uext[i_surf][:, i, j] += self.u_inf*self.u_inf_direction
                        total_offset_val -= self.u_inf*t

                    total_offset = total_offset_val*self.settings['u_inf_direction'] + for_pos
                    uext[i_surf][:, i, j] += self.gust_shape(zeta[i_surf][0, i, j] + total_offset[0],
                                                        zeta[i_surf][1, i, j] + total_offset[1],
                                                        zeta[i_surf][2, i, j] + total_offset[2],
                                                        t)
