"""Gust Velocity Field Generators

These generators are used to create a gust velocity field. :class:`.GustVelocityField` is the main class that should be
parsed as the ``velocity_field_input`` to the desired aerodynamic solver.

The remaining classes are the specific gust profiles and parsed as ``gust_shape``.

Examples:
    The typical input to the aerodynamic solver settings would therefore read similar to:

    >>> aero_settings = {'<some_aero_settings>': '<some_aero_settings>',
    >>>                  'velocity_field_generator': 'GustVelocityField',
    >>>                  'velocity_field_input': {'u_inf': 1,
    >>>                                           'gust_shape': '<desired_gust>',
    >>>                                           'gust_parameters': '<gust_settings>'}}

"""
import numpy as np
from abc import ABCMeta
import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings

dict_of_gusts = {}

doc_settings_description = 'The ``GustVelocityField`` generator takes the following settings as a dictionary assigned' \
                           'to ``gust_parameters``.'


def gust(arg):
    global dict_of_gusts
    try:
        arg.gust_id
    except AttributeError:
        raise AttributeError('Class defined as gust has no gust_id attribute')
    dict_of_gusts[arg.gust_id] = arg
    return arg


# @gust
class BaseGust(metaclass=ABCMeta):

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    def __init__(self):
        self.settings = dict()
        self._u_inf = None
        self._u_inf_direction = None

    @property
    def u_inf(self):
        return self._u_inf

    @u_inf.setter
    def u_inf(self, value):
        self._u_inf = value

    @property
    def u_inf_direction(self):
        return self._u_inf_direction

    @u_inf_direction.setter
    def u_inf_direction(self, value):
        self._u_inf_direction = value


@gust
class one_minus_cos(BaseGust):
    r"""
    One minus cos gust (single bump)

        .. math:: U_z = \frac{u_{de}}{2}\left[1-\cos\left(\frac{2\pi x}{S}\right)\right]

    This gust can be used by using the setting ``gust_shape = '1-cos'`` in :class:`.GustVelocityField`.
    """
    gust_id = '1-cos'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['gust_length'] = 'float'
    settings_default['gust_length'] = 0.0
    settings_description['gust_length'] = 'Length of gust, :math:`S`.'

    settings_types['gust_intensity'] = 'float'
    settings_default['gust_intensity'] = 0.0
    settings_description['gust_intensity'] = 'Intensity of the gust :math:`u_{de}`.'

    setting_table = settings.SettingsTable()
    __doc__ += setting_table.generate(settings_types, settings_default, settings_description,
                                      header_line=doc_settings_description)

    def initialise(self, in_dict):
        self.settings = in_dict
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

    def gust_shape(self, x, y, z, time=0):
        gust_length = self.settings['gust_length']
        gust_intensity = self.settings['gust_intensity']

        vel = np.zeros((3,))
        if x > 0.0 or x < -gust_length:
            return vel

        vel[2] = (1.0 - np.cos(2.0 * np.pi * x / gust_length)) * gust_intensity * 0.5
        return vel

@gust
class one_minus_cos_aircraft_certification(one_minus_cos):
    r"""
    One minus cos gust (single bump) as defined in EASA's certification requirements (CS 25.341)

        .. math:: U_z = \frac{u_{de}}{2}\left[1-\cos\left(\frac{2\pi x}{S}\right)\right]

    This gust can be used by using the setting ``gust_shape = '1-cos_aircraft_certification'`` in :class:`.GustVelocityField`.
    """
    gust_id = '1-cos_aircraft_certification'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['gust_length'] = 'float'
    settings_default['gust_length'] = 0.0
    settings_description['gust_length'] = 'Length of gust, :math:`S`.'

    settings_types['altitude'] = 'float'
    settings_default['altitude'] = 0.0
    settings_description['altitude'] = 'Flight altitude.'

    settings_types['aircraft_design_speed'] = 'bool'
    settings_default['aircraft_design_speed'] = True
    settings_description['aircraft_design_speed'] = 'Flight speed equal to aircraft design speed.'

    settings_types['Z_mo'] = 'float'
    settings_default['Z_mo'] = 800.
    settings_description['Z_mo'] = 'Maximum operating altitude, , :math:`Z_{mo}`, in meter.'

    settings_types['R1'] = 'float'
    settings_default['R1'] = 1.
    settings_description['R1'] = 'Ratio of maximum landing weight to maximum take-off weight,  :math:`R1`.'

    settings_types['R2'] = 'float'
    settings_default['R2'] = 0.96
    settings_description['R2'] = 'Ratio of zero fuel weight to maximum take-off weight,  :math:`R2`.'
 
    setting_table = settings.SettingsTable()
    __doc__ += setting_table.generate(settings_types, settings_default, settings_description,
                                      header_line=doc_settings_description)

    def initialise(self, in_dict):
        self.settings = in_dict
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)
    
    def get_flight_profile_alleviation_factor(self):
        r"""
        Computes the flight profile alleviation factor :math:`F_g` according to CS25.341(a)(6).
        """
        Fgz = 1. - self.settings['Z_mo'] / 76200
        Fgm = np.sqrt( self.settings['R2'] * np.tan(np.pi *  self.settings['R1'] / 4.))
        return 0.5 * (Fgz + Fgm)
    
    def get_reference_velocity(self):
        r"""
        Computes the reference velocity :math:`U_{ref}` according to CS25.341(a)(5).
        """
        if self.settings['altitude'] <= 0:
            U_ref = 17.07
        elif self.settings['altitude'] <= 13.41:
            U_ref = 17.07 - self.settings['altitude'] * (17.07 - 13.41) / 4572
        elif self.settings['altitude'] >= 18288:
            U_ref = 6.36
        else:
            U_ref = 13.41 - self.settings['altitude'] * (13.41 - 6.36) / (18288 - 4572)
        return U_ref

    def get_gust_design_speed(self):        
        r"""
        Computes the design gust velocity which is equivalent to the intensity of the gust :math:`u_{de}` 
        as specified in CS25.341(a)(4).
        """
        return self.get_reference_velocity() * self.get_flight_profile_alleviation_factor() * \
            (self.settings['gust_length'] / 107.)**(1/6)

    def gust_shape(self, x, y, z, time=0):
        
        self.settings['gust_intensity'] = self.get_gust_design_speed()
        return super.gust_shape(x, y, z, time)
 
@gust
class DARPA(BaseGust):
    r"""
    Discrete, non-uniform span model

    .. math:: U_z = \frac{u_{de}}{2}\left[1-\cos\left(\frac{2\pi x}{S}\right)\right]\cos\left(\frac{\pi y}{b}\right)

    This gust can be used by using the setting ``gust_shape = 'DARPA'`` in :class:`.GustVelocityField`.
    """
    gust_id = 'DARPA'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['gust_length'] = 'float'
    settings_default['gust_length'] = 0.0
    settings_description['gust_length'] = 'Length of gust'

    settings_types['gust_intensity'] = 'float'
    settings_default['gust_intensity'] = 0.0
    settings_description['gust_intensity'] = 'Intensity of the gust'

    settings_types['span'] = 'float'
    settings_default['span'] = 0.
    settings_description['span'] = 'Wing span'

    setting_table = settings.SettingsTable()
    __doc__ += setting_table.generate(settings_types, settings_default, settings_description,
                                      header_line=doc_settings_description)

    def initialise(self, in_dict):
        self.settings = in_dict
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

    def gust_shape(self, x, y, z, time=0):
        gust_length = self.settings['gust_length']
        gust_intensity = self.settings['gust_intensity']
        span = self.settings['span']

        vel = np.zeros((3,))
        if x > 0.0 or x < -gust_length:
            return vel

        vel[2] = (1.0 - np.cos(2.0 * np.pi * x / gust_length)) * gust_intensity * 0.5
        vel[2] *= -np.cos(y / span * np.pi)
        return vel


@gust
class continuous_sin(BaseGust):
    r"""
    Continuous sinusoidal gust model

    .. math:: U_z = \frac{u_{de}}{2}\sin\left(\frac{2\pi x}{S}\right)

    This gust can be used by using the setting ``gust_shape = 'continuous_sin'`` in :class:`GustVelocityField`.
    """
    gust_id = 'continuous_sin'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['gust_length'] = 'float'
    settings_default['gust_length'] = 0.0
    settings_description['gust_length'] = 'Length of gust'

    settings_types['gust_intensity'] = 'float'
    settings_default['gust_intensity'] = 0.0
    settings_description['gust_intensity'] = 'Intensity of the gust'

    setting_table = settings.SettingsTable()
    __doc__ += setting_table.generate(settings_types, settings_default, settings_description,
                                      header_line=doc_settings_description)

    def initialise(self, in_dict):
        self.settings = in_dict
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

    def gust_shape(self, x, y, z, time=0):
        gust_length = self.settings['gust_length']
        gust_intensity = self.settings['gust_intensity']

        vel = np.zeros((3,))
        if x > 0.0:
            return vel

        vel[2] = 0.5 * gust_intensity * np.sin(2 * np.pi * x / gust_length)
        return vel


@gust
class lateral_one_minus_cos(BaseGust):
    r"""
    This gust can be used by using the setting ``gust_shape = 'lateral 1-cos'`` in :class:`GustVelocityField`.
    """
    gust_id = 'lateral 1-cos'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['gust_length'] = 'float'
    settings_default['gust_length'] = 0.0
    settings_description['gust_length'] = 'Length of gust'

    settings_types['gust_intensity'] = 'float'
    settings_default['gust_intensity'] = 0.0
    settings_description['gust_intensity'] = 'Intensity of the gust'

    setting_table = settings.SettingsTable()
    __doc__ += setting_table.generate(settings_types, settings_default, settings_description,
                                      header_line=doc_settings_description)

    def initialise(self, in_dict):
        self.settings = in_dict
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

    def gust_shape(self, x, y, z, time=0):
        gust_length = self.settings['gust_length']
        gust_intensity = self.settings['gust_intensity']

        vel = np.zeros((3,))
        if x > 0.0 or x < -gust_length:
            return vel

        vel[1] = (1.0 - np.cos(2.0 * np.pi * x / gust_length)) * gust_intensity * 0.5
        return vel


@gust
class time_varying(BaseGust):
    r"""
    The inflow velocity changes with time but it is uniform in space. It is read from a 4 column file:

    .. math:: time[s] \Delta U_x \Delta U_y \Delta U_z

    This gust can be used by using the setting ``gust_shape = 'time varying'`` in :class:.`GustVelocityField`.
    """
    gust_id = 'time varying'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['file'] = 'str'
    settings_default['file'] = ''
    settings_description['file'] = 'File with the information'

    setting_table = settings.SettingsTable()
    __doc__ += setting_table.generate(settings_types, settings_default, settings_description,
                                      header_line=doc_settings_description)

    def __init__(self):
        super().__init__()

        self.file_info = None

    def initialise(self, in_dict):
        self.settings = in_dict
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        self.file_info = np.loadtxt(self.settings['file'])

    def gust_shape(self, x, y, z, time=0):
        vel = np.zeros((3,))
        d = np.dot(np.array([x, y, z]), self.u_inf_direction)
        if d > 0.0:
            return vel

        vel[0] = np.interp(d, -self.file_info[::-1, 0] * self.u_inf, self.file_info[::-1, 1])
        vel[1] = np.interp(d, -self.file_info[::-1, 0] * self.u_inf, self.file_info[::-1, 2])
        vel[2] = np.interp(d, -self.file_info[::-1, 0] * self.u_inf, self.file_info[::-1, 3])
        return vel


@gust
class time_varying_global(BaseGust):
    r"""
    Similar to the previous one but the velocity changes instanteneously in the whole flow field. It is not fed
    into the solid.

    This gust can be used by using the setting ``gust_shape = 'time varying global'`` in :class:`GustVelocityField`.
    """
    gust_id = 'time varying global'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['file'] = 'str'
    settings_default['file'] = ''
    settings_description['file'] = 'File with the information (only for time varying)'

    setting_table = settings.SettingsTable()
    __doc__ += setting_table.generate(settings_types, settings_default, settings_description,
                                      header_line=doc_settings_description)

    def __init__(self):
        super().__init__()

        self.file_info = None

    def initialise(self, in_dict):
        self.settings = in_dict
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        self.file_info = np.loadtxt(self.settings['file'])

    def gust_shape(self, x, y, z, time=0):
        vel = np.zeros((3,))

        vel[0] = np.interp(time, self.file_info[:, 0], self.file_info[:, 1])
        vel[1] = np.interp(time, self.file_info[:, 0], self.file_info[:, 2])
        vel[2] = np.interp(time, self.file_info[:, 0], self.file_info[:, 3])
        return vel


@gust
class span_sine(BaseGust):
    r"""
    This gust can be used by using the setting ``gust_shape = 'span sine'`` in :class:`GustVelocityField`.
    """
    gust_id = 'span sine'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['gust_intensity'] = 'float'
    settings_default['gust_intensity'] = 0.0
    settings_description['gust_intensity'] = 'Intensity of the gust'

    settings_types['span'] = 'float'
    settings_default['span'] = 0.
    settings_description['span'] = 'Wing span'

    settings_types['periods_per_span'] = 'int'
    settings_default['periods_per_span'] = 1
    settings_description['periods_per_span'] = 'Number of times that the sine is repeated in the span of the wing'

    settings_types['perturbation_dir'] = 'list(float)'
    settings_default['perturbation_dir'] = np.array([0, 0, 1.])
    settings_description['perturbation_dir'] = 'Direction in which the perturbation will be applied in A FoR'

    settings_types['span_dir'] = 'list(float)'
    settings_default['span_dir'] = np.array([0, 1., 0])
    settings_description['span_dir'] = 'Direction of the span of the wing'

    settings_types['span_with_gust'] = 'float'
    settings_default['span_with_gust'] = 0.
    settings_description['span_with_gust'] = 'Extension of the span to which the gust will be applied'

    setting_table = settings.SettingsTable()
    __doc__ += setting_table.generate(settings_types, settings_default, settings_description,
                                      header_line=doc_settings_description)

    def initialise(self, in_dict):
        self.settings = in_dict
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        if self.settings['span_with_gust'] == 0:
            self.settings['span_with_gust'] = self.settings['span']

    def gust_shape(self, x, y, z, time=0):
        d = np.dot(np.array([x, y, z]), self.settings['span_dir'])
        if np.abs(d) <= self.settings['span_with_gust'] / 2:
            vel = 0.5 * self.settings['gust_intensity'] * np.sin(
                d * 2. * np.pi / (self.settings['span'] / self.settings['periods_per_span']))
        else:
            vel = np.zeros((3,))

        return vel * self.settings['perturbation_dir']


@generator_interface.generator
class GustVelocityField(generator_interface.BaseGenerator):
    r"""
    Gust Velocity Field Generator

    ``GustVelocityField`` is a class inherited from ``BaseGenerator``

    The ``GustVelocityField`` class generates a gust profile velocity field, and the profile has the characteristics
    specified by the user.

    To call this generator, the ``generator_id = GustVelocityField`` shall be used.
    This is parsed as the value for the ``velocity_field_generator`` key in the desired aerodynamic solver's settings.

    Notation: :math:`u_{de}` is the gust intensity, :math:`S` is the gust length and :math:`b` is the wing span.
    :math:`x` and :math:`y` refer to the chordwise and spanwise distance penetrated into the gust, respectively.

    Several gust profiles are available. Your chosen gust profile should be parsed to ``gust_shape`` and the
    corresponding settings as a dictionary to ``gust_parameters``.

    """
    generator_id = 'GustVelocityField'
    generator_classification = 'velocity-field'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['u_inf'] = 'float'
    settings_default['u_inf'] = None
    settings_description['u_inf'] = 'Free stream velocity'

    settings_types['u_inf_direction'] = 'list(float)'
    settings_default['u_inf_direction'] = [1., 0., 0.]
    settings_description['u_inf_direction'] = 'Free stream velocity relative component'

    settings_types['offset'] = 'float'
    settings_default['offset'] = 0.0
    settings_description['offset'] = 'Spatial offset of the gust with respect to origin'

    settings_types['relative_motion'] = 'bool'
    settings_default['relative_motion'] = False
    settings_description['relative_motion'] = 'If true, the gust is convected with u_inf'

    settings_types['gust_shape'] = 'str'
    settings_default['gust_shape'] = None
    settings_description['gust_shape'] = 'Gust profile shape'

    settings_types['gust_parameters'] = 'dict'
    settings_default['gust_parameters'] = dict()
    settings_description['gust_parameters'] = 'Dictionary of parameters specific of the gust_shape selected'

    setting_table = settings.SettingsTable()
    __doc__ += setting_table.generate(settings_types, settings_default, settings_description,
                                      header_line='This generator takes the following settings')

    def __init__(self):

        self.settings = dict()
        self.gust = None
        self.u_inf = None
        self.u_inf_direction = None

        self.implemented_gusts = dict_of_gusts

    def initialise(self, in_dict):
        self.settings = in_dict
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        # check that the gust type is valid
        if not (self.settings['gust_shape'] in self.implemented_gusts):
            raise AttributeError('The gust shape ' + self.settings['gust_shape'] + ' is not implemented')

        self.gust = dict_of_gusts[self.settings['gust_shape']]()

        self.u_inf = self.settings['u_inf']
        self.u_inf_direction = self.settings['u_inf_direction']

        # set gust properties
        self.gust.u_inf = self.u_inf
        self.gust.u_inf_direction = self.u_inf_direction

        self.gust.initialise(self.settings['gust_parameters'])

    def generate(self, params, uext):
        zeta = params['zeta']
        override = params['override']
        if self.settings['gust_shape'] == 'span sine':
            ts = 0
            t = 0
            dt = 0
        else:
            ts = params['ts']
            t = params['t']
            dt = params['dt']

        for_pos = params['for_pos'][0:3]

        for i_surf in range(len(zeta)):
            if override:
                uext[i_surf].fill(0.0)

            for i in range(zeta[i_surf].shape[1]):
                for j in range(zeta[i_surf].shape[2]):
                    total_offset_val = self.settings['offset']
                    if self.settings['relative_motion']:
                        uext[i_surf][:, i, j] += self.settings['u_inf'] * self.settings['u_inf_direction']
                        total_offset_val -= self.settings['u_inf'] * t

                    total_offset = total_offset_val * self.settings['u_inf_direction'] + for_pos
                    uext[i_surf][:, i, j] += self.gust.gust_shape(
                        zeta[i_surf][0, i, j] + total_offset[0],
                        zeta[i_surf][1, i, j] + total_offset[1],
                        zeta[i_surf][2, i, j] + total_offset[2],
                        t
                    )
