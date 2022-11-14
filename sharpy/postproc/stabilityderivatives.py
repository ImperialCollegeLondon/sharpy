import sharpy.utils.solver_interface as solver_interface
import os
import numpy as np
import sharpy.utils.cout_utils as cout
import sharpy.utils.settings as settings_utils
from sharpy.linear.utils.derivatives import Derivatives


@solver_interface.solver
class StabilityDerivatives(solver_interface.BaseSolver):
    """
    Outputs the stability derivatives of a free-flying aircraft

    """
    solver_id = 'StabilityDerivatives'
    solver_classification = 'post-processor'

    settings_default = dict()
    settings_description = dict()
    settings_types = dict()
    settings_options = dict()

    settings_types['print_info'] = 'bool'
    settings_default['print_info'] = True
    settings_description['print_info'] = 'Display info to screen'

    settings_types['u_inf'] = 'float'
    settings_default['u_inf'] = 1.
    settings_description['u_inf'] = 'Free stream reference velocity'

    settings_types['S_ref'] = 'float'
    settings_default['S_ref'] = 1.
    settings_description['S_ref'] = 'Reference planform area'

    settings_types['b_ref'] = 'float'
    settings_default['b_ref'] = 1.
    settings_description['b_ref'] = 'Reference span'

    settings_types['c_ref'] = 'float'
    settings_default['c_ref'] = 1.
    settings_description['c_ref'] = 'Reference chord'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.data = None
        self.settings = dict()

        self.u_inf = 1
        self.inputs = 0
        self.caller = None
        self.folder = None

        self.ppal_axes = None
        self.n_control_surfaces = None

        self.coefficients = dict  # type: dict # name: scaling coefficient

    def initialise(self, data, custom_settings=None, caller=None, restart=False):
        self.data = data

        if custom_settings:
            self.settings = custom_settings
        else:
            self.settings = self.data.settings[self.solver_id]

        settings_utils.to_custom_types(self.settings,
                           self.settings_types,
                           self.settings_default,
                           options=self.settings_options,
                           no_ctype=True)
        self.caller = caller
        self.folder = data.output_folder + '/derivatives/'
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder, exist_ok=True)

        u_inf = self.settings['u_inf']
        s_ref = self.settings['S_ref']
        b_ref = self.settings['b_ref']
        c_ref = self.settings['c_ref']
        rho = self.data.linear.tsaero0.rho
        self.ppal_axes = self.data.settings['Modal']['rigid_modes_ppal_axes']

        # need to decide whether coefficients stay here or goes just in Derivatives class
        self.coefficients = {'force': 0.5 * rho * u_inf ** 2 * s_ref,
                             'moment_lon': 0.5 * rho * u_inf ** 2 * s_ref * c_ref,
                             'moment_lat': 0.5 * rho * u_inf ** 2 * s_ref * b_ref,
                             'force_angular_vel': 0.5 * rho * u_inf ** 2 * s_ref * c_ref / u_inf,
                             'moment_lon_angular_vel': 0.5 * rho * u_inf ** 2 * s_ref * c_ref * c_ref / u_inf}  # missing rates

        reference_dimensions = {}
        for k in ['S_ref', 'b_ref', 'c_ref', 'u_inf']:
            reference_dimensions[k] = self.settings[k]
        reference_dimensions['rho'] = rho
        reference_dimensions['quat'] = self.data.linear.tsstruct0.quat

        self.data.linear.derivatives = dict()  # {str:Derivatives()} (sharpy.linear.utils.derivatives.Derivatives)
        self.data.linear.derivatives['aerodynamic'] = Derivatives(reference_dimensions,
                                                                  static_state=self.steady_aero_forces(),
                                                                  target_system='aerodynamic')
        self.data.linear.derivatives['aeroelastic'] = Derivatives(reference_dimensions,
                                                                  static_state=self.steady_aero_forces(),
                                                                  target_system='aeroelastic')


    def run(self, **kwargs):
        
        online = settings_utils.set_value_or_default(kwargs, 'online', False)

        # TODO: consider running all required solvers inside this one to keep the correct settings
        # i.e: run Modal, Linear Assembly

        derivatives = self.data.linear.derivatives  # {str:Derivatives()} (sharpy.linear.utils.derivatives.Derivatives)
        if self.data.linear.linear_system.beam.sys.modal:
            phi = self.data.linear.linear_system.linearisation_vectors['mode_shapes'].real
        else:
            phi = None

        steady_forces = self.data.linear.linear_system.linearisation_vectors['forces_aero_beam_dof']
        v0 = self.get_freestream_velocity()
        quat = self.data.linear.tsstruct0.quat

        try:
            tpa = self.data.linear.tsstruct0.modal['t_pa']
        except KeyError:
            tpa = None

        if self.data.linear.linear_system.uvlm.scaled:
            raise NotImplementedError('Stability Derivatives not yet implented for scaled system')
            self.data.linear.linear_system.update(self.settings['u_inf'])

        for target_system in ['aerodynamic', 'aeroelastic']:
            state_space = self.get_state_space(target_system)
            target_system_derivatives = derivatives[target_system]

            target_system_derivatives.initialise_derivatives(state_space,
                                                             steady_forces,
                                                             quat,
                                                             v0,
                                                             phi,
                                                             self.data.linear.tsstruct0.modal['cg'],
                                                             tpa=tpa)
            target_system_derivatives.dict_of_derivatives[
                'force_angle_velocity'] = target_system_derivatives.new_derivative(
                'stability',
                'angle_derivatives',
                'Force/Angle via velocity')

            # useful to double check the effect of the ``track_body`` == 'on' setting
            # current_derivative.dict_of_derivatives['force_angle_angle'] = current_derivative.new_derivative(
            #     'stability',
            #     'angle_derivatives_tb',
            #     'Force/Angle via Track Body')

            target_system_derivatives.dict_of_derivatives['force_velocity'] = target_system_derivatives.new_derivative(
                'body',
                'body_derivatives')

            target_system_derivatives.dict_of_derivatives['force_cs'] = target_system_derivatives.new_derivative(
                'body',
                'control_surface_derivatives')
            target_system_derivatives.save(self.folder)
            target_system_derivatives.savetxt(self.folder)

        return self.data

    def get_freestream_velocity(self):
        try:
            u_inf = self.data.settings['StaticUvlm']['aero_solver_settings']['u_inf']
            u_inf_direction = self.data.settings['StaticCoupled']['aero_solver_settings']['u_inf_direction']
        except KeyError:
            try:
                u_inf = self.data.settings['StaticCoupled']['aero_solver_settings']['velocity_field_input']['u_inf']
                u_inf_direction = self.data.settings['StaticCoupled']['aero_solver_settings']['velocity_field_input']['u_inf_direction']
            except KeyError:
                cout.cout_wrap('Unable to find free stream velocity settings in StaticUvlm or StaticCoupled,'
                               'please ensure these settings are provided in the config .sharpy file. If'
                               'you are running a restart simulation make sure they are included too, regardless'
                               'of these solvers being present in the SHARPy flow', 4)
                raise KeyError

        try:
            v0 = u_inf * u_inf_direction * -1
        except TypeError:
            # For restart solutions, where the settings may have not been processed and thus may
            # exist but in string format
            try:
                u_inf_direction = np.array(u_inf_direction, dtype=float)
            except ValueError:
                if u_inf_direction.find(',') < 0:
                    u_inf_direction = np.fromstring(u_inf_direction.strip('[]'), sep=' ', dtype=float)
                else:
                    u_inf_direction = np.fromstring(u_inf_direction.strip('[]'), sep=',', dtype=float)
            finally:
                v0 = np.array(u_inf_direction, dtype=float) * float(u_inf) * -1

        return v0

    def get_state_space(self, target_system):
        if target_system == 'aerodynamic':
            ss = self.data.linear.linear_system.uvlm.ss
        elif target_system == 'aeroelastic':
            ss = self.data.linear.ss
        else:
            raise NameError('Unknown target system {:s}'.format(target_system))

        return ss

    def steady_aero_forces(self):
        fx = np.sum(self.data.aero.timestep_info[0].inertial_steady_forces[:, 0], 0) + \
             np.sum(self.data.aero.timestep_info[0].inertial_unsteady_forces[:, 0], 0)

        fy = np.sum(self.data.aero.timestep_info[0].inertial_steady_forces[:, 1], 0) + \
             np.sum(self.data.aero.timestep_info[0].inertial_unsteady_forces[:, 1], 0)

        fz = np.sum(self.data.aero.timestep_info[0].inertial_steady_forces[:, 2], 0) + \
             np.sum(self.data.aero.timestep_info[0].inertial_unsteady_forces[:, 2], 0)

        return fx, fy, fz
