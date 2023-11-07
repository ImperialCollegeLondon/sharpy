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
from scipy.interpolate import interp1d

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

    settings_types['gust_component'] = 'int'
    settings_default['gust_component'] = 2
    settings_description['gust_component'] = 'Component of the gust velocity in the G-frame (x,y,z)->(0,1,2).'

    setting_table = settings.SettingsTable()
    __doc__ += setting_table.generate(settings_types, settings_default, settings_description,
                                      header_line=doc_settings_description)

    def initialise(self, in_dict, restart=False):
        self.settings = in_dict
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

    def gust_shape(self, x, y, z, time=0):
        gust_length = self.settings['gust_length']
        gust_intensity = self.settings['gust_intensity']

        vel = np.zeros((3,))
        if x > 0.0 or x < -gust_length:
            return vel

        vel[self.settings['gust_component']] = (1.0 - np.cos(2.0 * np.pi * x / gust_length)) * gust_intensity * 0.5
        return vel


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

    settings_types['gust_component'] = 'int'
    settings_default['gust_component'] = 2
    settings_description['gust_component'] = 'Component of the gust velocity in the G-frame (x,y,z)->(0,1,2).'

    setting_table = settings.SettingsTable()
    __doc__ += setting_table.generate(settings_types, settings_default, settings_description,
                                      header_line=doc_settings_description)

    def initialise(self, in_dict, restart=False):
        self.settings = in_dict
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

    def gust_shape(self, x, y, z, time=0):
        gust_length = self.settings['gust_length']
        gust_intensity = self.settings['gust_intensity']
        span = self.settings['span']

        vel = np.zeros((3,))
        if x > 0.0 or x < -gust_length:
            return vel

        vel[self.settings['gust_component']] = (1.0 - np.cos(2.0 * np.pi * x / gust_length)) * gust_intensity * 0.5
        vel[self.settings['gust_component']] *= -np.cos(y / span * np.pi)
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

    settings_types['gust_component'] = 'int'
    settings_default['gust_component'] = 2
    settings_description['gust_component'] = 'Component of the gust velocity in the G-frame (x,y,z)->(0,1,2).'

    setting_table = settings.SettingsTable()
    __doc__ += setting_table.generate(settings_types, settings_default, settings_description,
                                      header_line=doc_settings_description)

    def initialise(self, in_dict, restart=False):
        self.settings = in_dict
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

    def gust_shape(self, x, y, z, time=0):
        gust_length = self.settings['gust_length']
        gust_intensity = self.settings['gust_intensity']

        vel = np.zeros((3,))
        if x > 0.0:
            return vel

        vel[self.settings['gust_component']] = 0.5 * gust_intensity * np.sin(2 * np.pi * x / gust_length)
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

    settings_types['gust_component'] = ['list(int)', 'int']
    settings_default['gust_component'] = [0, 1, 2]
    settings_description['gust_component'] = 'Component of the gust velocity in the G-frame to be considered (x,y,z)->(0,1,2).'

    setting_table = settings.SettingsTable()
    __doc__ += setting_table.generate(settings_types, settings_default, settings_description,
                                      header_line=doc_settings_description)

    def __init__(self):
        super().__init__()
        self.file_info = None
        self.list_interpolated_velocity_field_functions = []

    def initialise(self, in_dict, restart=False):
        self.settings = in_dict
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        self.file_info = np.loadtxt(self.settings['file'])
        self.initialise_interpolation_functions()
    
    def initialise_interpolation_functions(self):
        for idim in self.settings['gust_component']:
            self.list_interpolated_velocity_field_functions.append(interp1d(self.file_info[:, 0], self.file_info[:, idim+1], 
                                                                            bounds_error=False,fill_value="extrapolate"))

    def gust_shape(self, x, y, z, time=0):
        vel = np.zeros((3,))
        for counter, idim in enumerate(self.settings['gust_component']):
            vel[idim] = self.list_interpolated_velocity_field_functions[counter](time)
        return vel


@gust
class time_varying(time_varying_global):
    r"""
    The inflow velocity changes with time but it is uniform in space. It is read from a 4 column file:

    .. math:: time[s] \Delta U_x \Delta U_y \Delta U_z

    This gust can be used by using the setting ``gust_shape = 'time varying'`` in :class:.`GustVelocityField`.
    """
    gust_id = 'time varying'
    
    def initialise_interpolation_functions(self):
        for idim in self.settings['gust_component']:
            self.list_interpolated_velocity_field_functions.append(interp1d(-self.file_info[::-1, 0] * self.u_inf, self.file_info[::-1, idim+1], 
                                                                            bounds_error=False,fill_value="extrapolate"))

    def gust_shape(self, x, y, z, time=0):
        vel = np.zeros((3,))
        d = np.dot(np.array([x, y, z]), self.u_inf_direction)
        if d <= 0.0:       
            for counter, idim in enumerate(self.settings['gust_component']):
                vel[idim] = self.list_interpolated_velocity_field_functions[counter](d)
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

    def initialise(self, in_dict, restart=False):
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

    def initialise(self, in_dict, restart=False):
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

        self.gust.initialise(self.settings['gust_parameters'], restart=restart)

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
