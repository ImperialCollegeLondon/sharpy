import numpy as np
import scipy.optimize
import scipy.signal

import sharpy.aero.utils.uvlmlib as uvlmlib
import sharpy.utils.settings as settings_utils
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.generator_interface as gen_interface
import sharpy.utils.cout_utils as cout
from sharpy.utils.constants import vortex_radius_def


@solver
class StepUvlm(BaseSolver):
    """
    StepUVLM is the main solver to use for unsteady aerodynamics.

    The desired flow field is injected into the simulation by means of a ``generator``. For a list of available
    velocity field generators see the documentation page on generators which can be found under SHARPy Source Code.

    Typical generators could be:

    * :class:`~.generators.steadyvelocityfield.SteadyVelocityField`

    * :class:`~.generators.gustvelocityfield.GustVelocityField`

    * :class:`~.generators.turbvelocityfield.TurbVelocityField`

    amongst others.

    """
    solver_id = 'StepUvlm'
    solver_classification = 'aero'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

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
    settings_description['convection_scheme'] = '``0``: fixed wake, ' \
                                                '``2``: convected with background flow;' \
                                                '``3``: full force-free wake'
    settings_options['convection_scheme'] = [0, 2, 3]

    settings_types['dt'] = 'float'
    settings_default['dt'] = 0.1
    settings_description['dt'] = 'Time step'

    # the following settings are not in used but are required in place since they are called in uvlmlib
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
    settings_description['velocity_field_generator'] = 'Name of the velocity field generator to be used in the ' \
                                                       'simulation'

    settings_types['velocity_field_input'] = 'dict'
    settings_default['velocity_field_input'] = {}
    settings_description['velocity_field_input'] = 'Dictionary of settings for the velocity field generator'

    settings_types['gamma_dot_filtering'] = 'int'
    settings_default['gamma_dot_filtering'] = 0
    settings_description['gamma_dot_filtering'] = 'Filtering parameter for the Welch filter for the Gamma_dot ' \
                                                  'estimation. Used when ``unsteady_force_contribution`` is ``on``.'

    settings_types['rho'] = 'float'
    settings_default['rho'] = 1.225
    settings_description['rho'] = 'Air density'

    settings_types['cfl1'] = 'bool'
    settings_default['cfl1'] = True
    settings_description['cfl1'] = 'If it is ``True``, it assumes that the discretisation complies with CFL=1'

    settings_types['vortex_radius'] = 'float'
    settings_default['vortex_radius'] = vortex_radius_def
    settings_description['vortex_radius'] = 'Distance between points below which induction is not computed'

    settings_types['vortex_radius_wake_ind'] = 'float'
    settings_default['vortex_radius_wake_ind'] = vortex_radius_def
    settings_description['vortex_radius_wake_ind'] = 'Distance between points below which induction is not computed in the wake convection'

    settings_types['interp_coords'] = 'int'
    settings_default['interp_coords'] = 0
    settings_description['interp_coords'] = 'Coordinates to use for wake description: cartesian(0) or cylindrical_z(1)'
    settings_options['interp_coords'] = [0, 1]

    settings_types['filter_method'] = 'int'
    settings_default['filter_method'] = 0
    settings_description['filter_method'] = 'Method to filter the points: no filter (0) moving average(2)'
    settings_options['filter_method'] = [0, 2]
    # filter_method = 1 was dedicated to a splines filter. If it is needed in the future chack dev_alglib branch

    settings_types['interp_method'] = 'int'
    settings_default['interp_method'] = 0
    settings_description['interp_method'] = 'Method of interpolation: linear(0), parabolic(1), splines(2), slerp around z(3), slerp around yaw_slerp(4)'
    settings_options['interp_method'] = [0, 1, 2, 3, 4]

    settings_types['yaw_slerp'] = 'float'
    settings_default['yaw_slerp'] = 0
    settings_description['yaw_slerp'] = 'Yaw angle in radians to be used when interp_metod == 4'

    settings_types['centre_rot'] = 'list(float)'
    settings_default['centre_rot'] = [0., 0., 0.]
    settings_description['centre_rot'] = 'Coordinates of the centre of rotation to perform slerp interpolation or cylindrical coordinates'

    settings_types['quasi_steady'] = 'bool'
    settings_default['quasi_steady'] = False
    settings_description['quasi_steady'] = 'Use quasi-steady approximation in UVLM'
    
    settings_types['only_nonlifting'] = 'bool'
    settings_default['only_nonlifting'] = False
    settings_description['only_nonlifting'] = 'Consider nonlifting body interactions'

    settings_types['nonlifting_body_interactions'] = 'bool'
    settings_default['nonlifting_body_interactions'] = False
    settings_description['nonlifting_body_interactions'] = 'Consider nonlifting body interactions'

    settings_types['phantom_wing_test'] = 'bool'
    settings_default['phantom_wing_test'] = False
    settings_description['phantom_wing_test'] = 'Debug option'

    settings_types['centre_rot_g'] = 'list(float)'
    settings_default['centre_rot_g'] = [0., 0., 0.]
    settings_description['centre_rot_g'] = 'Centre of rotation in G FoR around which ``rbm_vel_g`` is applied'

    settings_types['ignore_first_x_nodes_in_force_calculation'] = 'int'
    settings_default['ignore_first_x_nodes_in_force_calculation'] = 0
    settings_description['ignore_first_x_nodes_in_force_calculation'] = 'Ignores the forces on the first user-specified number of nodes of all surfaces.'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description, settings_options)

    def __init__(self):
        self.data = None
        self.settings = None
        self.velocity_generator = None

    def initialise(self, data, custom_settings=None, restart=False):
        """
        To be called just once per simulation.
        """
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings_utils.to_custom_types(self.settings,
                           self.settings_types,
                           self.settings_default,
                           self.settings_options)

        self.data.structure.add_unsteady_information(
            self.data.structure.dyn_dict,
            self.settings['n_time_steps'])

        # Filtering
        if self.settings['gamma_dot_filtering'] == 1:
            cout.cout_wrap(
                "gamma_dot_filtering cannot be one. Changing it to None", 2)
            self.settings['gamma_dot_filtering'] = None
        if self.settings['gamma_dot_filtering'] is not None:
            if self.settings['gamma_dot_filtering']:
                if not self.settings['gamma_dot_filtering'] % 2:
                    cout.cout_wrap(
                        "gamma_dot_filtering does not support even numbers." +
                        "Changing " +
                        str(self.settings['gamma_dot_filtering']) +
                        " to " +
                        str(self.settings['gamma_dot_filtering'] + 1),
                        2)
                    self.settings['gamma_dot_filtering'] += 1

        # init velocity generator
        velocity_generator_type = gen_interface.generator_from_string(
            self.settings['velocity_field_generator'])
        self.velocity_generator = velocity_generator_type()
        self.velocity_generator.initialise(
            self.settings['velocity_field_input'],
            restart=restart)

    def run(self, **kwargs):
        """
        Runs a step of the aerodynamics as implemented in UVLM.
        """

        # Default values
        aero_tstep = settings_utils.set_value_or_default(kwargs, 'aero_step', self.data.aero.timestep_info[-1])
        structure_tstep = settings_utils.set_value_or_default(kwargs, 'structural_step', self.data.structure.timestep_info[-1])
        convect_wake = settings_utils.set_value_or_default(kwargs, 'convect_wake', True)
        dt= settings_utils.set_value_or_default(kwargs, 'dt', self.settings['dt'])
        t = settings_utils.set_value_or_default(kwargs, 't', self.data.ts*dt)
        unsteady_contribution = settings_utils.set_value_or_default(kwargs, 'unsteady_contribution', False)

        if not aero_tstep.zeta:
            return self.data

        # generate uext
        self.velocity_generator.generate({'zeta': aero_tstep.zeta,
                                          'override': True,
                                          't': t,
                                          'ts': self.data.ts,
                                          'dt': dt,
                                          'for_pos': structure_tstep.for_pos,
                                          'is_wake': False},
                                         aero_tstep.u_ext)
        if ((self.settings['convection_scheme'] > 1 and convect_wake) or
           (not self.settings['cfl1'])):
            # generate uext_star
            self.velocity_generator.generate({'zeta': aero_tstep.zeta_star,
                                              'override': True,
                                              'ts': self.data.ts,
                                              'dt': dt,
                                              't': t,
                                              'for_pos': structure_tstep.for_pos,
                                              'is_wake': True},
                                             aero_tstep.u_ext_star)
        if self.settings['nonlifting_body_interactions']:

            nl_body_tstep = settings_utils.set_value_or_default(kwargs, 'nl_body_tstep', self.data.nonlifting_body.timestep_info[-1])
            self.velocity_generator.generate({'zeta': nl_body_tstep.zeta,
                                              'override': True,
                                              'ts': self.data.ts,
                                              'dt': dt,
                                              't': t,
                                              'for_pos': structure_tstep.for_pos,
                                              'is_wake': False},
                                             nl_body_tstep.u_ext)
            
            uvlmlib.uvlm_solver_lifting_and_nonlifting(self.data.ts,
                                                       aero_tstep, 
                                                       nl_body_tstep,
                                                       structure_tstep, 
                                                       self.settings,
                                                       convect_wake=convect_wake, 
                                                       dt=dt)
        else:            
            uvlmlib.uvlm_solver(self.data.ts,
                                aero_tstep,
                                structure_tstep,
                                self.settings,
                                convect_wake=convect_wake,
                                dt=dt)

        if unsteady_contribution and not self.settings['quasi_steady']:
            # calculate unsteady (added mass) forces:
            self.data.aero.compute_gamma_dot(dt,
                                             aero_tstep,
                                             self.data.aero.timestep_info[-3:])
            if self.settings['gamma_dot_filtering'] is None:
                self.filter_gamma_dot(aero_tstep,
                                      self.data.aero.timestep_info,
                                      None)
            elif self.settings['gamma_dot_filtering'] > 0:
                self.filter_gamma_dot(
                    aero_tstep,
                    self.data.aero.timestep_info,
                    self.settings['gamma_dot_filtering'])
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
        if self.settings['nonlifting_body_interactions']:              
            self.data.nonlifting_body.add_timestep()

    def update_grid(self, beam):
        self.data.aero.generate_zeta(beam,
                                     self.data.aero.aero_settings,
                                     -1,
                                     beam_ts=-1)
        if self.settings['nonlifting_body_interactions']:                            
            self.data.nonlifting_body.generate_zeta(beam,
                                                    self.data.aero.aero_settings,
                                                    -1,
                                                    beam_ts=-1)

    def update_custom_grid(self, structure_tstep, aero_tstep, nl_body_tstep = None):
        self.data.aero.generate_zeta_timestep_info(structure_tstep,
                                                   aero_tstep,
                                                   self.data.structure,
                                                   self.data.aero.aero_settings,
                                                   dt=self.settings['dt'])
        if self.settings['nonlifting_body_interactions']:
            if nl_body_tstep is None:
                nl_body_tstep  =  self.data.nonlifting_body.timestep_info[-1] 
            self.data.nonlifting_body.generate_zeta_timestep_info(structure_tstep,
                                                                  nl_body_tstep,
                                                                  self.data.structure,
                                                                  self.data.nonlifting_body.aero_settings,
                                                                  dt = self.settings['dt'])

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
