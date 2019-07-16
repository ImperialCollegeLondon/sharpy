import numpy as np

import sharpy.utils.controller_interface as controller_interface
import sharpy.utils.settings as settings
import sharpy.utils.exceptions as exc

@controller_interface.controller
class ControlSurfacePidController(controller_interface.BaseController):
    r"""


    """
    controller_id = 'ControlSurfacePidController'

    settings_types = dict()
    settings_default = dict()

    settings_types['time_history_input_file'] = 'str'
    settings_default['time_history_input_file'] = None

    settings_types['P'] = 'float'
    settings_default['P'] = None
    settings_description['P'] = 'Proportional gain of the controller'

    settings_types['I'] = 'float'
    settings_default['I'] = 0.0
    settings_description['I'] = 'Integral gain of the controller'

    settings_types['D'] = 'float'
    settings_default['D'] = 0.0
    settings_description['D'] = 'Differential gain of the controller'

    settings_types['input_type'] = 'str'
    settings_default['input_type'] = None
    settings_description['input_type'] = ('Quantity used to define the' +
        ' reference state. Supported: `pitch`')

    supported_input_types = ['pitch']

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types,
                                       settings_default,
                                       settings_description)

    def __init__(self):
        self.in_dict()

        self.data = None
        self.settings = None

    def initialise(self, in_dict):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict,
                                 self.settings_types,
                                 self.settings_default)

        self.settings = self.in_dict()

        # validate that the input_type is in the supported ones
        if self.settings['input_type'] not in self.supported_input_types:
            cout.cout_wrap('The input_type {} is not supported by {}'.format(self.settings['input_type'], self.controller_id), 3)
            cout.cout_wrap('The supported ones are:', 3)
            for i in self.supported_input_types:
                cout.cout_wrap('    {}'.format(i), 3)
            raise NotImplementedError()

    def control(self, data, current_state):
        r"""
        Main routine of the controller.
        Input is `data` (the self.data in the solver), and
        `currrent_state` which is a dictionary with ['structural', 'aero']
        time steps for the current iteration.

        These states can be included in data.[aero, structural].timestep_info
        and the controller will make sure it is not duplicating the data
        by checking that
        `current_state['structural'] is not data.structural.timestep_info[-1]`
        and the same for aero.
        """

        # get desired time history input
        
        # get current state input
        
        # calculate output of controller
        # (input is history, state, required state)
        
        # apply it where needed.













