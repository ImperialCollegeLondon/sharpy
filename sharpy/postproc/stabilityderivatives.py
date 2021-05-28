import sharpy.utils.solver_interface as solver_interface
import os
import numpy as np
import sharpy.utils.cout_utils as cout
import sharpy.utils.algebra as algebra
import sharpy.utils.settings as settings
from sharpy.linear.utils.derivatives import Derivatives, DerivativeSet
import sharpy.linear.utils.derivatives as derivatives_utils


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

    settings_table = settings.SettingsTable()
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

    def initialise(self, data, custom_settings=None, caller=None):
        self.data = data

        if custom_settings:
            self.settings = custom_settings
        else:
            self.settings = self.data.settings[self.solver_id]

        settings.to_custom_types(self.settings, self.settings_types, self.settings_default,
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

        # need to decide whether coefficients stays here or goes just in Derivatives class
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

        self.data.linear.derivatives = dict()
        self.data.linear.derivatives['aerodynamic'] = Derivatives(reference_dimensions,
                                                                  static_state=self.steady_aero_forces(),
                                                                  target_system='aerodynamic')
        self.data.linear.derivatives['aeroelastic'] = Derivatives(reference_dimensions,
                                                                  static_state=self.steady_aero_forces(),
                                                                  target_system='aeroelastic')


    def run(self, online=False):

        # TODO: consider running all required solvers inside this one to keep the correct settings
        # i.e: run Modal, Linear Ass... COMING SOON with SHARPy routines

        derivatives = self.data.linear.derivatives
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
            cout.cout_wrap('-------- {:s} SYSTEM DERIVATIVES ---------'.format(target_system.upper()))
            state_space = self.get_state_space(target_system)
            current_derivative = derivatives[target_system]

            current_derivative.initialise_derivatives(state_space,
                                                      steady_forces,
                                                      quat,
                                                      v0,
                                                      phi,
                                                      self.data.linear.tsstruct0.modal['cg'],
                                                      tpa=tpa)
            current_derivative.dict_of_derivatives['force_angle_velocity'] = current_derivative.new_derivative(
                'stability',
                'angle_derivatives',
                'Force/Angle via velocity')

            # current_derivative.dict_of_derivatives['force_angle_angle'] = current_derivative.new_derivative(
            #     'stability',
            #     'angle_derivatives_tb',
            #     'Force/Angle via Track Body')

            current_derivative.dict_of_derivatives['force_velocity'] = current_derivative.new_derivative(
                'body',
                'body_derivatives')
            
            current_derivative.dict_of_derivatives['force_cs'] = current_derivative.new_derivative(
                'body',
                'control_surface_derivatives')
            current_derivative.save(self.folder)
            current_derivative.savetxt(self.folder)

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

    def uvlm_steady_state_transfer_function(self):
        """
        Stability derivatives calculated using the transfer function of the UVLM projected onto the structural
        degrees of freedom at zero frequency (steady state).

        Returns:
            np.array: matrix containing the steady state values of the transfer function between the force output
              (columns) and the velocity / control surface inputs (rows).
        """
        if self.settings['target_system'] == 'aerodynamic':
            ss = self.data.linear.linear_system.uvlm.ss
        elif self.settings['target_system'] == 'aeroelastic':
            ss = self.data.linear.ss
        else:
            raise NameError('Unknown target system {:s}'.format(self.settings['target_system']))
        modal = self.data.linear.linear_system.beam.sys.modal
        use_euler = self.data.linear.linear_system.beam.sys.use_euler

        nout = 6
        if use_euler:
            rig_dof = 9
        else:
            rig_dof = 10

        A, B, C, D = ss.get_mats()
        H0 = ss.freqresp(np.array([1e-5]))[:, :, 0]
        # if type(A) == libsp.csc_matrix:
        #     H0 = C.dot(scsp.linalg.inv(scsp.eye(ss.states, format='csc') - A).dot(B)) + D
        # else:
        #     H0 = C.dot(np.linalg.inv(np.eye(ss.states) - A).dot(B)) + D

        if modal:
            vel_inputs_variables = ss.input_variables.get_variable_from_name('q_dot')
            rbm_indices = vel_inputs_variables.cols_loc[:9]

            # look for control surfaces
            try:
                cs_input_variables = ss.input_variables.get_variable_from_name('control_surface_deflection')
                dot_cs_input_variables = ss.input_variables.get_variable_from_name('dot_control_surface_deflection')
            except ValueError:
                cs_indices = np.array([], dtype=int)
                dot_cs_indices = np.array([], dtype=int)
            else:
                cs_indices = cs_input_variables.cols_loc
                dot_cs_indices = dot_cs_input_variables.cols_loc
                self.n_control_surfaces = cs_input_variables.size
            finally:
                input_indices = np.concatenate((rbm_indices, cs_indices, dot_cs_indices))

            output_indices = ss.output_variables.get_variable_from_name('Q').rows_loc[:6]

            H0 = H0[np.ix_(output_indices, input_indices)].real

            return H0

    def steady_aero_forces(self):
        # Find ref forces in G
        fx, fy, fz = self.data.linear.tsaero0.total_steady_inertial_forces[:3]
        return fx, fy, fz

    def static_state(self):
        fx, fy, fz = self.steady_aero_forces()
        force_coeff = 0.5 * self.data.linear.tsaero0.rho * self.settings['u_inf'].value ** 2 * self.settings['S_ref'].value
        Cfx = fx / force_coeff
        Cfy = fy / force_coeff
        Cfz = fz / force_coeff

        return Cfx, Cfy, Cfz


