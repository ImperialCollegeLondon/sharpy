import numpy as np

import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings


@generator_interface.generator
class DynamicControlSurface(generator_interface.BaseGenerator):
    """
    Dynamic Control Surface deflection Generator

    ``DynamicControlSurface`` class inherited from ``BaseGenerator``

    The object generates a deflection in radians based on the time series given as a vector in the input data

    To call this generator, the ``generator_id = DynamicControlSurface`` shall be used.
    This is parsed as the value for the ``control_surface_deflection_generator`` key in the aerogridloader solver's settings.

    Args:
        in_dict (dict): Input data in the form of dictionary. See acceptable entries below.

            ======================   ===============  ======================================================================  ===================
            Name                     Type             Description                                                             Default
            ======================   ===============  ======================================================================  ===================
            ``dt``                   ``float``        Timestep for the simulation                                             ``None``
            ``deflection_file``      ``str``          Relative path to the file with the deflection information.              ``None``
            ======================   ===============  ======================================================================  ===================

    Attributes:
        settings_types (dict): Acceptable data types of the input data
        settings_default (dict): Default values for input data should the user not provide them
        deflection (np.array): Array of deflection of the control surface
        deflection_dot (np.array): Array of the time derivative of the cs deflection. Calculated using 1st order finite differences.

    See Also:
        .. py:class:: sharpy.utils.generator_interface.BaseGenerator

    """
    generator_id = 'DynamicControlSurface'

    def __init__(self):
        self.in_dict = dict()
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['dt'] = 'float'
        self.settings_default['dt'] = None

        self.settings_types['deflection_file'] = 'str'
        self.settings_default['deflection_file'] = None

        self.deflection = None
        self.deflection_dot = None

    def initialise(self, in_dict):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict, self.settings_types, self.settings_default)

        # load file
        self.deflection = np.loadtxt(self.in_dict['deflection_file'])

        # deflection velocity
        self.deflection_dot = np.zeros_like(self.deflection)
        self.deflection_dot[0:-1] = np.diff(self.deflection)/self.in_dict['dt'].value
        self.deflection_dot[-1] = 0
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(self.deflection)
        # plt.plot(self.deflection_dot)
        # plt.show()

    def generate(self, params):
        it = params['it']
        return self.deflection[it], self.deflection_dot[it]

    def __call__(self, params):
        return self.generate(params)
