import os

import numpy as np
from tvtk.api import tvtk, write_data

import sharpy.utils.cout_utils as cout
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra
import sharpy.structure.utils.xbeamlib as xbeamlib


@solver
class Cleanup(BaseSolver):
    """
    Clean-up old timesteps to save RAM memory
    """
    solver_id = 'Cleanup'
    solver_classification = 'post-processor'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['clean_structure'] = 'bool'
    settings_default['clean_structure'] = True
    settings_description['clean_structure'] = 'Clean-up :class:`~sharpy.utils.datastructures.StructTimeStepInfo`'

    settings_types['clean_aero'] = 'bool'
    settings_default['clean_aero'] = True
    settings_description['clean_aero'] = 'Clean-up :class:`~sharpy.utils.datastructures.AeroTimeStepInfo`'

    settings_types['remaining_steps'] = 'int'
    settings_default['remaining_steps'] = 10
    settings_description['remaining_steps'] = 'The last `remaining_steps` are not cleaned up'

    table = settings.SettingsTable()
    __doc__ += table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.settings = None
        self.data = None
        self.caller = None

    def initialise(self, data, custom_settings=None, caller=None):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings.to_custom_types(self.settings,
                                 self.settings_types,
                                 self.settings_default)
        self.caller = caller

    def run(self, online=False):
        if self.settings['clean_structure']:
            self.clean(self.data.structure.timestep_info, self.settings['remaining_steps'])
        if self.settings['clean_aero']:
            self.clean(self.data.aero.timestep_info, self.settings['remaining_steps'])

        return self.data

    def clean(self, series, n_steps):
        for i in range(len(series[:-n_steps])):
            series[i] = None
