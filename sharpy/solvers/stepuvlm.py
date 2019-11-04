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
    StepUVLM is the main solver to use for unsteady aerodynamics.
    """
    solver_id = 'StepUvlm'
    solver_classification = 'Aerodynamic'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['print_info'] = 'bool'
    settings_default['print_info'] = True
    settings_description['print_info'] = 'Print info to screen'

    settings_types['num_cores'] = 'int'
    settings_default['num_cores'] = 0
    settings_description['num_cores'] = 'Number of cores to use in the VLM lib'

    settings_types['n_time_steps'] = 'int'
    settings_default['n_time_steps'] = 100
    settings_description['n_time_steps'] = 'Number of time steps to be run'

    settings_types['convection_scheme'] = 'int'
    settings_default['convection_scheme'] = 3
    settings_description['convection_scheme'] = '0 for fixed wake, 2 for convected with background flow and 3 for full force-free wake'

    settings_types['dt'] = 'float'
    settings_default['dt'] = 0.1
    settings_description['dt'] = 'Time step'

    settings_types['iterative_solver'] = 'bool'
    settings_default['iterative_solver'] = False
    settings_description['iterative_solver'] = 'Not in use'

    settings_types['iterative_tol'] = 'float'
    settings_default['iterative_tol'] = 1e-4
    settings_description['iterative_tol'] = 'Not in use'

    settings_types['iterative_precond'] = 'bool'
    settings_default['iterative_precond'] = False
    settings_description['iterative_precond'] = 'Not in use'

    settings_types['velocity_field_generator'] = 'str'
    settings_default['velocity_field_generator'] = 'SteadyVelocityField'
    settings_description['velocity_field_generator'] = 'Name of the velocity field generator to be used in the simulation'

    settings_types['velocity_field_input'] = 'dict'
    settings_default['velocity_field_input'] = {}
    settings_description['velocity_field_input'] = 'Dictionary of settings for the velocity field generator'

    settings_types['gamma_dot_filtering'] = 'int'
    settings_default['gamma_dot_filtering'] = 0
    settings_description['gamma_dot_filtering'] = 'Filtering parameter for the Welch filter for the Gamma_dot estimation. Used when ``unsteady_force_contribution`` is ``on``.'

    settings_types['rho'] = 'float'
    settings_default['rho'] = 1.225
    settings_description['rho'] = 'Air density'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.data = None
        self.settings = None
        self.velocity_generator = None

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

        self.data.structure.add_unsteady_information(
            self.data.structure.dyn_dict,
            self.settings['n_time_steps'].value)

        # Filtering
        if self.settings['gamma_dot_filtering'].value == 1:
            cout.cout_wrap(
                "gamma_dot_filtering cannot be one. Changing it to None", 2)
            self.settings['gamma_dot_filtering'] = None
        if self.settings['gamma_dot_filtering'] is not None:
            if self.settings['gamma_dot_filtering'].value:
                if not self.settings['gamma_dot_filtering'].value % 2:
                    cout.cout_wrap(
                        "gamma_dot_filtering does not support even numbers." +
                        "Changing " +
                        str(self.settings['gamma_dot_filtering'].value) +
                        " to " +
                        str(self.settings['gamma_dot_filtering'].value + 1),
                        2)
                    self.settings['gamma_dot_filtering'] = (
                        ct.c_int(self.settings['gamma_dot_filtering'].value + 1))

        # init velocity generator
        velocity_generator_type = gen_interface.generator_from_string(
            self.settings['velocity_field_generator'])
        self.velocity_generator = velocity_generator_type()
        self.velocity_generator.initialise(
            self.settings['velocity_field_input'])

    def run(self,
            aero_tstep=None,
            structure_tstep=None,
            convect_wake=True,
            dt=None,
            t=None,
            unsteady_contribution=False):
        """
        Runs a step of the aerodynamics as implemented in UVLM.
        """

        if aero_tstep is None:
            aero_tstep = self.data.aero.timestep_info[-1]
        if structure_tstep is None:
            structure_tstep = self.data.structure.timestep_info[-1]
        if dt is None:
            dt = self.settings['dt'].value
        if t is None:
            t = self.data.ts*dt

        if not aero_tstep.zeta:
            return self.data

        # generate uext
        self.velocity_generator.generate({'zeta': aero_tstep.zeta,
                                          'override': True,
                                          't': t,
                                          'ts': self.data.ts,
                                          'dt': dt,
                                          'for_pos': structure_tstep.for_pos},
                                         aero_tstep.u_ext)
        if self.settings['convection_scheme'].value > 1 and convect_wake:
            # generate uext_star
            self.velocity_generator.generate({'zeta': aero_tstep.zeta_star,
                                              'override': True,
                                              'ts': self.data.ts,
                                              'dt': dt,
                                              't': t,
                                              'for_pos': structure_tstep.for_pos},
                                             aero_tstep.u_ext_star)

        uvlmlib.uvlm_solver(self.data.ts,
                            aero_tstep,
                            structure_tstep,
                            self.settings,
                            convect_wake=convect_wake,
                            dt=dt)

        if unsteady_contribution:
            # calculate unsteady (added mass) forces:
            self.data.aero.compute_gamma_dot(dt,
                                             aero_tstep,
                                             self.data.aero.timestep_info[-3:])
            if self.settings['gamma_dot_filtering'] is None:
                self.filter_gamma_dot(aero_tstep,
                                      self.data.aero.timestep_info,
                                      None)
            elif self.settings['gamma_dot_filtering'].value > 0:
                self.filter_gamma_dot(
                    aero_tstep,
                    self.data.aero.timestep_info,
                    self.settings['gamma_dot_filtering'].value)
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

    @staticmethod
    def filter_gamma_dot(tstep, history, filter_param):
        clean_history = [x for x in history if x is not None]
        series_length = len(clean_history) + 1
        for i_surf in range(len(tstep.zeta)):
            n_rows, n_cols = tstep.gamma[i_surf].shape
            for i in range(n_rows):
                for j in range(n_cols):
                    series = np.zeros((series_length,))
                    for it in range(series_length - 1):
                        series[it] = clean_history[it].gamma_dot[i_surf][i, j]
                    series[-1] = tstep.gamma_dot[i_surf][i, j]

                    # filter
                    tstep.gamma_dot[i_surf][i, j] = scipy.signal.wiener(
                        series, filter_param)[-1]
