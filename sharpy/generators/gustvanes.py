import numpy as np
import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings

@generator_interface.generator
class GustVanes(generator_interface.BaseGenerator):
    """
    ``GustVanes`` generator.

    This generator sets up a specified number of gust vanes that are oscillated by a previous defined signal. These oscillating
    gust vanes generate a gust as done in wind tunnel experiments. For each vane a new aerogrid surface including a wake is 
    generated in the ``Aerogrid`` solver. 
 
    """
    generator_id = 'GustVanes'
    generator_classification = 'runtime'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    settings_types['n_vanes'] = 'int'
    settings_default['n_vanes'] = 1
    settings_description['n_vanes'] = 'Number of gust vanes to be generated'

    settings_types['streamwise_position'] = 'list(float)'
    settings_default['streamwise_position'] = [-2.]
    settings_description['streamwise_position'] = 'List of streamwise coordinaten of each gust vane leading edge'

    settings_types['vertical_position'] = 'list(float)'
    settings_default['vertical_position'] = [-2.]
    settings_description['vertical_position'] = 'List of vertical coordinaten of each gust vane leading edge'

    settings_types['vane_parameters'] = 'list(dict)'
    settings_default['vane_parameters'] = [{'M': 4,'N': 20, 'M_star': 40, 'span': 10., 'chord': 0.1,}]
    settings_description['vane_parameters'] = 'Dictionary of parameters to specify the gust vane geometry and its position '


    def __init__(self):
        self.settings = None
        self.vane_info = None
        self.aero_dimensions = None
        self.aero_dimensions_star = None

    def initialise(self, in_dict, **kwargs):
        self.settings = in_dict
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default, no_ctype=True,
                                 options=self.settings_options)
        self.n_vanes = self.settings['n_vanes']
        self.vane_info = self.settings['vane_parameters']     
        
