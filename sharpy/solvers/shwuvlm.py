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
import sys


@solver
class SHWUvlm(BaseSolver):
    """
    Steady vortex method assuming helicoidal wake shape
    """

    solver_id = 'SHWUvlm'
    solver_classification = 'aero'

    # settings list
    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['print_info'] = 'bool'
    settings_default['print_info'] = True
    settings_description['print_info'] = 'Output run-time information'

    settings_types['num_cores'] = 'int'
    settings_default['num_cores'] = 0
    settings_description['num_cores'] = 'Number of cores to used in parallelisation'

    settings_types['convection_scheme'] = 'int'
    settings_default['convection_scheme'] = 2
    settings_description['convection_scheme'] = 'Convection scheme for the wake (only 2 tested for this solver)'

    settings_types['dt'] = 'float'
    settings_default['dt'] = 0.1
    settings_description['dt'] = 'time step used to discretise the wake'

    settings_types['iterative_solver'] = 'bool'
    settings_default['iterative_solver'] = False
    settings_description['iterative_solver'] = ''

    settings_types['iterative_tol'] = 'float'
    settings_default['iterative_tol'] = 1e-4
    settings_description['iterative_tol'] = ''

    settings_types['iterative_precond'] = 'bool'
    settings_default['iterative_precond'] = False
    settings_description['iterative_precond'] = ''

    settings_types['velocity_field_generator'] = 'str'
    settings_default['velocity_field_generator'] = 'SteadyVelocityField'
    settings_description['velocity_field_generator'] = 'Name of the velocity field generator'

    settings_types['velocity_field_input'] = 'dict'
    settings_default['velocity_field_input'] = {}
    settings_description['velocity_field_input'] = 'Dictionary of inputs needed by the velocity field generator'

    settings_types['gamma_dot_filtering'] = 'int'
    settings_default['gamma_dot_filtering'] = 0
    settings_description['gamma_dot_filtering'] = 'Parameter used to filter gamma dot (only odd numbers bigger than one allowed)'

    settings_types['rho'] = 'float'
    settings_default['rho'] = 1.225
    settings_description['rho'] = 'Density'

    settings_types['rot_vel'] = 'float'
    settings_default['rot_vel'] = 0.0
    settings_description['rot_vel'] = 'Rotation velocity in rad/s'

    settings_types['rot_axis'] = 'list(float)'
    settings_default['rot_axis'] = [1., 0., 0.]
    settings_description['rot_axis'] = 'Axis of rotation of the wake'

    settings_types['rot_center'] = 'list(float)'
    settings_default['rot_center'] = [0., 0., 0.]
    settings_description['rot_center'] = 'Center of rotation of the wake'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):

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

        # self.data.structure.add_unsteady_information(self.data.structure.dyn_dict, self.settings['n_time_steps'].value)

        # init velocity generator
        velocity_generator_type = gen_interface.generator_from_string(
            self.settings['velocity_field_generator'])
        self.velocity_generator = velocity_generator_type()
        self.velocity_generator.initialise(self.settings['velocity_field_input'])

        # Checks
        if not self.settings['convection_scheme'].value == 2:
            sys.exit("ERROR: convection_scheme: %u. Only 2 supported" % self.settings['convection_scheme'].value)

    def run(self,
            aero_tstep=None,
            structure_tstep=None,
            convect_wake=False,
            dt=None,
            t=None,
            unsteady_contribution=False):

        # Checks
        if convect_wake:
            sys.exit("ERROR: convect_wake should be set to False")

        if aero_tstep is None:
            aero_tstep = self.data.aero.timestep_info[-1]
        if structure_tstep is None:
            structure_tstep = self.data.structure.timestep_info[-1]
        if dt is None:
            dt = self.settings['dt'].value
        if t is None:
            t = self.data.ts*dt

        # generate the wake because the solid shape might change
        aero_tstep = self.data.aero.timestep_info[self.data.ts]
        self.data.aero.wake_shape_generator.generate({'zeta': aero_tstep.zeta,
                                            'zeta_star': aero_tstep.zeta_star,
                                            'gamma': aero_tstep.gamma,
                                            'gamma_star': aero_tstep.gamma_star})

        # generate uext
        self.velocity_generator.generate({'zeta': aero_tstep.zeta,
                                          'override': True,
                                          't': t,
                                          'ts': self.data.ts,
                                          'dt': dt,
                                          'for_pos': structure_tstep.for_pos},
                                         aero_tstep.u_ext)
        # if self.settings['convection_scheme'].value > 1 and convect_wake:
        #     # generate uext_star
        #     self.velocity_generator.generate({'zeta': aero_tstep.zeta_star,
        #                                       'override': True,
        #                                       'ts': self.data.ts,
        #                                       'dt': dt,
        #                                       't': t,
        #                                       'for_pos': structure_tstep.for_pos},
        #                                      aero_tstep.u_ext_star)

        # previous_ts = max(len(self.data.aero.timestep_info) - 1, 0) - 1
        # previous_ts = -1
        # print('previous_step max circulation: %f' % previous_aero_tstep.gamma[0].min())
        # print('current step max circulation: %f' % aero_tstep.gamma[0].min())
        uvlmlib.shw_solver(self.data.ts,
                            aero_tstep,
                            structure_tstep,
                            self.settings,
                            convect_wake=False,
                            dt=dt)
        # print('current step max unsforce: %f' % aero_tstep.dynamic_forces[0].max())

        if unsteady_contribution:
            # calculate unsteady (added mass) forces:
            self.data.aero.compute_gamma_dot(dt, aero_tstep, self.data.aero.timestep_info[-3:])
            if self.settings['gamma_dot_filtering'].value > 0:
                self.filter_gamma_dot(aero_tstep, self.data.aero.timestep_info, self.settings['gamma_dot_filtering'].value)
            uvlmlib.uvlm_calculate_unsteady_forces(aero_tstep,
                                                   structure_tstep,
                                                   self.settings,
                                                   convect_wake=convect_wake,
                                                   dt=dt)
        else:
            for i_surf in range(len(aero_tstep.gamma)):
                aero_tstep.gamma_dot[i_surf][:] = 0.0

        return self.data

    def add_step(self):
        self.data.aero.add_timestep()

    def update_grid(self, beam):
        self.data.aero.generate_zeta(beam, self.data.aero.aero_settings, -1, beam_ts=-1)

    def update_custom_grid(self, structure_tstep, aero_tstep):
        self.data.aero.generate_zeta_timestep_info(structure_tstep, aero_tstep, self.data.structure, self.data.aero.aero_settings)

    def update_step(self):
        self.data.aero.generate_zeta(self.data.structure,
                                     self.data.aero.aero_settings,
                                     self.data.ts)

    @staticmethod
    def filter_gamma_dot(tstep, history, filter_param):
        series_length = len(history) + 1
        for i_surf in range(len(tstep.zeta)):
            n_rows, n_cols = tstep.gamma[i_surf].shape
            for i in range(n_rows):
                for j in range(n_cols):
                    series = np.zeros((series_length,))
                    for it in range(series_length - 1):
                        series[it] = history[it].gamma_dot[i_surf][i, j]
                    series[-1] = tstep.gamma_dot[i_surf][i, j]

                    # filter
                    tstep.gamma_dot[i_surf][i, j] = scipy.signal.wiener(series, filter_param)[-1]
