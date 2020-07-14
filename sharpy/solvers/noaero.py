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
class NoAero(BaseSolver):
    """
    Solver to be used with DynamicCoupled when aerodynamics are not of interest
    """
    solver_id = 'NoAero'
    solver_classification = 'Aero'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    
    settings_types['dt'] = 'float'
    settings_default['dt'] = 0.1
    settings_description['dt'] = 'Time step'

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
        
        # generate the wake because the solid shape might change
        if aero_tstep is None: 
            aero_tstep = self.data.aero.timestep_info[self.data.ts]
        self.data.aero.wake_shape_generator.generate({'zeta': aero_tstep.zeta,
                                            'zeta_star': aero_tstep.zeta_star,
                                            'gamma': aero_tstep.gamma,
                                            'gamma_star': aero_tstep.gamma_star,
                                            'dist_to_orig': aero_tstep.dist_to_orig,
                                            'wake_conv_vel': aero_tstep.wake_conv_vel})

        return self.data

    def add_step(self):
        self.data.aero.add_timestep()

    def update_grid(self, beam):
        self.data.aero.generate_zeta(beam,
                                     self.data.aero.aero_settings,
                                     -1,
                                     beam_ts=-1)

    def update_custom_grid(self, structure_tstep, aero_tstep):
        self.data.aero.generate_zeta_timestep_info(structure_tstep,
                                                   aero_tstep,
                                                   self.data.structure,
                                                   self.data.aero.aero_settings,
                                                   dt=self.settings['dt'].value)
    
