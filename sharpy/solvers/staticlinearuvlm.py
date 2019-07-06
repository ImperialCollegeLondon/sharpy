"""
Static Solver using the linearised UVLM
N Goizueta
Nov 2018
"""

import sharpy.utils.settings as settings
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.generator_interface as gen_interface

@solver
class StaticLinearUVLM(BaseSolver):
    """
    Static Linear UVLM solver derived from the linearisation of the UVLM
    """
    solver_id = 'StaticLinearUvlm'

    def __init__(self):

        # TODO: add full settings
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['rho'] = 'float'
        self.settings_default['rho'] = 1.225

        self.data = None
        self.settings = None
        self.velocity_generator = None

    def initialise(self, data, custom_settings=None):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        self.update_step()

        # init velocity generator
        velocity_generator_type = gen_interface.generator_from_string(
            self.settings['velocity_field_generator'])
        self.velocity_generator = velocity_generator_type()
        self.velocity_generator.initialise(self.settings['velocity_field_input'])

        # TODO: check that timestep exists (ie reference about which to linearise)
        # TODO: linearise UVLM system. Assemble state
        # TODO: check reshape method and compare with StepLinearUVLM

    def run(self):
        # generate uext
        self.velocity_generator.generate({'zeta': self.data.aero.timestep_info[self.data.ts].zeta,
                                          'override': True},
                                         self.data.aero.timestep_info[self.data.ts].u_ext)

        # TODO: replace with Linearised UVLM and solve
        # grid orientation
        #uvlmlib.vlm_solver(self.data.aero.timestep_info[self.data.ts],
        #                   self.settings)

        return self.data

    def next_step(self):
        """ Updates de aerogrid based on the info of the step, and increases
        the self.ts counter """
        self.data.aero.add_timestep()
        self.update_step()

    def update_step(self):
        self.data.aero.generate_zeta(self.data.structure,
                                     self.data.aero.aero_settings,
                                     self.data.ts)
