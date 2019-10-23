import ctypes as ct
import numpy as np
import scipy.optimize
import scipy.signal

import sharpy.utils.algebra as algebra
import sharpy.aero.utils.uvlmlib as uvlmlib
import sharpy.utils.settings as settings
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.generator_interface as gen_interface
import sharpy.utils.cout_utils as cout


@solver
class StepUvlm(BaseSolver):
    """
    Solver to be used with DynamicCoupled when aerodynamics are not of interest
    """
    solver_id = 'NoAero'
    solver_classification = 'Aerodynamic'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.data = None
        self.settings = None

    def initialise(self, data, custom_settings=None):
        """
        To be called just once per simulation.
        """
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings.to_custom_types(self.settings,
                                 self.settings_types,
                                 self.settings_default)

    def run(self,
            aero_tstep=None,
            structure_tstep=None,
            convect_wake=True,
            dt=None,
            t=None,
            unsteady_contribution=False):
        
        return self.data

    def add_step(self):
        self.data.aero.add_timestep()

    def update_grid(self, beam):
        pass

    def update_custom_grid(self, structure_tstep, aero_tstep):
        pass
    
