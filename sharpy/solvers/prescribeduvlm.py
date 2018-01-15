import ctypes as ct
import numpy as np

import sharpy.utils.algebra as algebra
import sharpy.aero.utils.uvlmlib as uvlmlib
import sharpy.utils.settings as settings
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.generator_interface as gen_interface
import sharpy.utils.cout_utils as cout


@solver
class PrescribedUvlm(BaseSolver):
    solver_id = 'PrescribedUvlm'

    def __init__(self):
        # settings list
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['print_info'] = 'bool'
        self.settings_default['print_info'] = True

        self.settings_types['num_cores'] = 'int'
        self.settings_default['num_cores'] = 0

        self.settings_types['n_time_steps'] = 'int'
        self.settings_default['n_time_steps'] = 100

        self.settings_types['convection_scheme'] = 'int'
        self.settings_default['convection_scheme'] = 3

        self.settings_types['steady_n_rollup'] = 'int'
        self.settings_default['steady_n_rollup'] = 0

        self.settings_types['steady_rollup_tolerance'] = 'float'
        self.settings_default['steady_rollup_tolerance'] = 1e-4

        self.settings_types['steady_rollup_aic_refresh'] = 'int'
        self.settings_default['steady_rollup_aic_refresh'] = 1

        self.settings_types['dt'] = 'float'
        self.settings_default['dt'] = 0.1

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

        self.data.structure.add_unsteady_information(self.data.structure.dyn_dict, self.settings['n_time_steps'].value)

        # generates and rotates the aero grid and rotates the structure
        self.update_step()

        # init velocity generator
        velocity_generator_type = gen_interface.generator_from_string(
            self.settings['velocity_field_generator'])
        self.velocity_generator = velocity_generator_type()
        self.velocity_generator.initialise(self.settings['velocity_field_input'])

        self.data.ts = 0
        # generate uext
        self.velocity_generator.generate({'zeta': self.data.aero.timestep_info[self.data.ts].zeta,
                                          'override': True,
                                          'ts': self.data.ts,
                                          't': 0.0},
                                         self.data.aero.timestep_info[self.data.ts].u_ext)

        uvlmlib.uvlm_init(self.data.aero.timestep_info[self.data.ts], self.settings)

    def run(self):
        for self.data.ts in range(1, self.settings['n_time_steps'].value + 1):
            cout.cout_wrap('i_iter: ' + str(self.data.ts))
            self.next_step()
            t = self.data.ts*self.settings['dt'].value
            # generate uext
            self.velocity_generator.generate({'zeta': self.data.aero.timestep_info[self.data.ts].zeta,
                                              'override': True,
                                              'ts': self.data.ts,
                                              't': t},
                                             self.data.aero.timestep_info[self.data.ts].u_ext)
            if self.settings['convection_scheme'].value > 1:
                # generate uext_star
                self.velocity_generator.generate({'zeta': self.data.aero.timestep_info[self.data.ts].zeta_star,
                                                  'override': True,
                                                  'ts': self.data.ts,
                                                  't': t},
                                                 self.data.aero.timestep_info[self.data.ts].u_ext_star)

            self.data.structure.timestep_info[self.data.ts].for_vel = self.data.structure.dynamic_input[self.data.ts - 1]['for_vel'].astype(ct.c_double)

            uvlmlib.uvlm_solver(self.data.ts,
                                self.data.aero.timestep_info[self.data.ts],
                                self.data.aero.timestep_info[self.data.ts - 1],
                                self.data.structure.timestep_info[self.data.ts],
                                self.settings)

            self.data.structure.timestep_info[self.data.ts].for_pos[0:3] = (
                self.data.structure.timestep_info[self.data.ts - 1].for_pos[0:3] +
                np.dot(self.data.structure.timestep_info[self.data.ts].cga().transpose(),
                       self.settings['dt'].value*self.data.structure.timestep_info[self.data.ts - 1].for_vel[0:3]))
            self.data.structure.timestep_info[self.data.ts].for_pos[3:6] = (
                self.data.structure.timestep_info[self.data.ts - 1].for_pos[3:6] +
                np.dot(self.data.structure.timestep_info[self.data.ts].cga().transpose(),
                       self.settings['dt'].value*self.data.structure.timestep_info[self.data.ts - 1].for_vel[3:6]))

        return self.data

    def next_step(self):
        """ Updates de aerogrid based on the info of the step, and increases
        the self.ts counter """
        self.data.structure.next_step()
        self.data.aero.add_timestep()
        self.update_step()

    def update_step(self, integrate_orientation=True):
        self.data.aero.generate_zeta(self.data.structure,
                                     self.data.aero.aero_settings,
                                     self.data.ts)

        if integrate_orientation:
            if self.data.ts > 0:
                # euler = self.data.structure.dynamic_input[self.data.ts - 1]['for_pos'][3:6]
                # euler_rot = algebra.euler2rot(euler)  # this is Cag
                # quat = algebra.mat2quat(euler_rot.T)
                # TODO need to update orientation
                # quat = self.data.structure.timestep_info[self.data.ts - 1].quat
                quat = algebra.rotate_quaternion(self.data.structure.timestep_info[self.data.ts].quat,
                                                 self.data.structure.timestep_info[self.data.ts].for_vel[3:6]*
                                                 self.settings['dt'])
            else:
                quat = self.data.structure.ini_info.quat.copy()
        else:
            quat = self.data.structure.timestep_info[self.data.ts].quat

        quat = algebra.unit_vector(quat)
        self.data.structure.update_orientation(quat, self.data.ts)  # quat corresponding to Cga
        self.data.aero.update_orientation(self.data.structure.timestep_info[self.data.ts].quat, self.data.ts)       # quat corresponding to Cga











