import numpy as np

import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings
import sharpy.utils.cout_utils as cout_utils


@generator_interface.generator
class DynamicControlSurface(generator_interface.BaseGenerator):
    """
    Dynamic Control Surface deflection Generator

    The object generates a deflection in radians based on the time series given as a single vector in the input data.
    A first order finite-differences scheme is used to calculate the deflection rate based on the provided time step
    increment.

    To call this generator, the ``generator_id = DynamicControlSurface`` key shall be used for the setting
    `control_surface_deflection` in the ``AerogridLoader`` solver.

   One instance of this generator will be created for each control surface, thus, a group of settings should be defined
   for each control surface (``cs0_settings``, ``cs1_settings`` ... in the example below).
   All of these groups of settings should be collected as values in a dictionary which keys are the associated control surface number in string format.
   This dictionary should be parsed to the variable
    ``control_surface_deflection_generator_settings`` in ``AerogridLoader``. This is shown better
    in the example below:

    Examples:

        .. code-block:

            cs0_settings = {}  # these are the settings for control surface number 0
            cs1_settings = {}  # these are the settings for control surface number 1
            dict_of_cs = {'0': cs0_settings,
                                  '1': cs1_settings} # This dictionary groups all the settings for all the control surfaces
            settings = {}
            settings['AerogridLoader] = {'control_surface_deflection' : ['DynamicControlSurface],
                                                         'control_surface_deflection_generator_settings: dict_of_cs}



    Attributes:
        deflection (np.array): Array of deflection of the control surface
        deflection_dot (np.array): Array of the time derivative of the cs deflection. Calculated using 1st order finite differences.

    """
    generator_id = 'DynamicControlSurface'
    generator_classification = 'utils'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['dt'] = 'float'
    settings_default['dt'] = None
    settings_description['dt'] = 'Time step increment'

    settings_types['deflection_file'] = 'str'
    settings_default['deflection_file'] = None
    settings_description['deflection_file'] = 'Path to the file with the deflection information'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description,
                                       header_line='This generator takes the following inputs:')

    def __init__(self):
        self.in_dict = dict()

        self.deflection = None
        self.deflection_dot = None

    def initialise(self, in_dict, restart=False):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict, self.settings_types, self.settings_default, no_ctype=True)

        # load file
        try:
            self.deflection = np.loadtxt(self.in_dict['deflection_file'])
        except OSError:
            cout_utils.cout_wrap('Unable to find control surface deflection file input', 4)
            raise FileNotFoundError('Could not locate deflection file: '
                                    '{:s}'.format(self.in_dict['deflection_file']))
        else:
            cout_utils.cout_wrap('\tSuccess loading file {:s}'.format(self.in_dict['deflection_file']), 2)

        # deflection velocity
        self.deflection_dot = np.zeros_like(self.deflection)
        self.deflection_dot[0:-1] = np.diff(self.deflection)/self.in_dict['dt']
        self.deflection_dot[-1] = 0

    def generate(self, params):
        it = params['it']
        return self.deflection[it], self.deflection_dot[it]

    def __call__(self, params):
        return self.generate(params)
