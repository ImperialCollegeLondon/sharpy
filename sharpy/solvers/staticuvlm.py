import ctypes as ct
import numpy as np

# import sharpy.aero.models.aerogrid as aerogrid
# import sharpy.aero.utils.mapping as mapping
import sharpy.utils.algebra as algebra
import sharpy.aero.utils.uvlmlib as uvlmlib
import sharpy.utils.cout_utils as cout
import sharpy.utils.settings as settings
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.generator_interface as gen_interface

@solver
class StaticUvlm(BaseSolver):
    solver_id = 'StaticUvlm'

    def __init__(self):
        # settings list
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['print_info'] = 'bool'
        self.settings_default['print_info'] = True

        self.settings_types['horseshoe'] = 'bool'
        self.settings_default['horseshoe'] = False

        self.settings_types['num_cores'] = 'int'
        self.settings_default['num_cores'] = 0

        self.settings_types['n_rollup'] = 'int'
        self.settings_default['n_rollup'] = 1

        self.settings_types['rollup_dt'] = 'float'
        self.settings_default['rollup_dt'] = 0.1

        self.settings_types['rollup_aic_refresh'] = 'int'
        self.settings_default['rollup_aic_refresh'] = 1

        self.settings_types['rollup_tolerance'] = 'float'
        self.settings_default['rollup_tolerance'] = 1e-4

        self.settings_types['iterative_solver'] = 'bool'
        self.settings_default['iterative_solver'] = False

        self.settings_types['iterative_tol'] = 'float'
        self.settings_default['iterative_tol'] = 1e-4

        self.settings_types['iterative_precond'] = 'bool'
        self.settings_default['iterative_precond'] = False

        self.settings_types['velocity_field_generator'] = 'str'
        self.settings_default['velocity_field_generator'] = 'SteadyVelocityField'

        self.settings_types['velocity_field_input'] = 'dict'
        self.settings_default['velocity_field_input'] = {}

        # self.settings_types['alpha'] = 'float'
        # self.settings_default['alpha'] = 0.0
        #
        # self.settings_types['beta'] = 'float'
        # self.settings_default['beta'] = 0.0
        #
        # self.settings_types['roll'] = 'float'
        # self.settings_default['roll'] = 0.0

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

    def run(self):
        # generate uext
        self.velocity_generator.generate({'zeta': self.data.aero.timestep_info[self.data.ts].zeta,
                                          'override': True},
                                         self.data.aero.timestep_info[self.data.ts].u_ext)
        # grid orientation
        uvlmlib.vlm_solver(self.data.aero.timestep_info[self.data.ts],
                           self.settings)

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
        for i_surf in range(self.data.aero.timestep_info[self.data.ts].n_surf):
            self.data.aero.timestep_info[self.data.ts].forces[i_surf].fill(0.0)
            self.data.aero.timestep_info[self.data.ts].dynamic_forces[i_surf].fill(0.0)













