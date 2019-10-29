import sharpy.utils.solver_interface as solver_interface
import os
import numpy as np
import scipy.sparse as scsp
import sharpy.linear.src.libsparse as libsp
import sharpy.utils.cout_utils as cout
import sharpy.utils.algebra as algebra
import sharpy.utils.settings as settings


@solver_interface.solver
class StabilityDerivatives(solver_interface.BaseSolver):
    """
    Outputs the stability derivatives of a free-flying aircraft

    Warnings:
        Under Development

    To Do:
        * Coefficient of stability derivatives
        * Option to output in NED frame

    """
    solver_id = 'StabilityDerivatives'

    settings_default = dict()
    settings_description = dict()
    settings_types = dict()

    settings_types['print_info'] = 'bool'
    settings_default['print_info'] = True
    settings_description['print_info'] = 'Display info to screen'

    settings_types['folder'] = 'str'
    settings_default['folder'] = './output/'
    settings_description['folder'] = 'Output directory'

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

    def initialise(self, data, custom_settings=None):
        self.data = data

        if custom_settings:
            self.settings = custom_settings
        else:
            self.settings = self.data.settings[self.solver_id]

        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

    def run(self):

        Y_freq = self.uvlm_steady_state_transfer_function()
        derivatives_dimensional, derivatives_coeff = self.derivatives(Y_freq)

        self.export_derivatives(np.hstack((derivatives_coeff[:, :6], derivatives_coeff[:, -2:])))

        return self.data

    def uvlm_steady_state_transfer_function(self):
        """
        Stability derivatives calculated using the transfer function of the UVLM projected onto the structural
        degrees of freedom at zero frequency (steady state).

        Returns:
            np.array: matrix containing the steady state values of the transfer function between the force output
              (columns) and the velocity / control surface inputs (rows).
        """
        ssuvlm = self.data.linear.linear_system.uvlm.ss
        modal = self.data.linear.linear_system.beam.sys.modal
        use_euler = self.data.linear.linear_system.beam.sys.use_euler

        nout = 6
        if use_euler:
            rig_dof = 9
        else:
            rig_dof = 10

        # Get rigid body + control surface inputs
        try:
            n_ctrl_sfc = self.data.linear.linear_system.uvlm.control_surface.n_control_surfaces
        except AttributeError:
            n_ctrl_sfc = 0

        self.inputs = rig_dof + n_ctrl_sfc

        in_matrix = np.zeros((ssuvlm.inputs, self.inputs))
        out_matrix = np.zeros((nout, ssuvlm.outputs))

        if modal:
            # Modal scaling
            raise NotImplementedError('Not yet implemented in modal space')
        else:
            in_matrix[-self.inputs:, :] = np.eye(self.inputs)
            out_matrix[:, -rig_dof:-rig_dof+6] = np.eye(nout)

        ssuvlm.addGain(in_matrix, where='in')
        ssuvlm.addGain(out_matrix, where='out')

        A, B, C, D = ssuvlm.get_mats()
        if type(A) == libsp.csc_matrix:
            Y_freq = C.dot(scsp.linalg.inv(scsp.eye(ssuvlm.states, format='csc') - A).dot(B)) + D
        else:
            Y_freq = C.dot(np.linalg.inv(np.eye(ssuvlm.states) - A).dot(B)) + D
        Yf = ssuvlm.freqresp(np.array([0]))

        return Y_freq

    def derivatives(self, Y_freq):

        Cng = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])  # Project SEU on NED - TODO implementation
        u_inf = self.settings['u_inf'].value
        s_ref = self.settings['S_ref'].value
        b_ref = self.settings['b_ref'].value
        c_ref = self.settings['c_ref'].value
        rho = self.data.linear.tsaero0.rho

        # Inertial frame
        try:
            euler = self.data.linear.tsstruct0.euler
            Pga = algebra.euler2rot(euler)
            rig_dof = 9
        except AttributeError:
            quat = self.data.linear.tsstruct0.quat
            Pga = algebra.quat2rotation(quat)
            rig_dof = 10

        derivatives_g = np.zeros((6, Y_freq.shape[1] + 2))
        coefficients = {'force': 0.5*rho*u_inf**2*s_ref,
                        'moment_lon': 0.5*rho*u_inf**2*s_ref*c_ref,
                        'moment_lat': 0.5*rho*u_inf**2*s_ref*b_ref,
                        'force_angular_vel': 0.5*rho*u_inf**2*s_ref*c_ref/u_inf,
                        'moment_lon_angular_vel': 0.5*rho*u_inf**2*s_ref*c_ref*c_ref/u_inf} # missing rates

        for in_channel in range(Y_freq.shape[1]):
            derivatives_g[:3, in_channel] = Pga.dot(Y_freq[:3, in_channel])
            derivatives_g[3:, in_channel] = Pga.dot(Y_freq[3:, in_channel])

        derivatives_g[:3, :3] /= coefficients['force']
        derivatives_g[:3, 3:6] /= coefficients['force_angular_vel']
        derivatives_g[4, :3] /= coefficients['moment_lon']
        derivatives_g[4, 3:6] /= coefficients['moment_lon_angular_vel']
        derivatives_g[[3, 5], :] /= coefficients['moment_lat']

        derivatives_g[:, -2] = derivatives_g[:, 2] * u_inf  # ders wrt alpha
        derivatives_g[:, -1] = derivatives_g[:, 1] * u_inf  # ders wrt beta

        der_matrix = np.zeros((6, self.inputs - (rig_dof - 6)))
        der_col = 0
        for i in list(range(6))+list(range(rig_dof, self.inputs)):
            der_matrix[:3, der_col] = Y_freq[:3, i]
            der_matrix[3:6, der_col] = Y_freq[3:6, i]
            der_col += 1

        labels_force = {0: 'X',
                        1: 'Y',
                        2: 'Z',
                        3: 'L',
                        4: 'M',
                        5: 'N'}

        labels_velocity = {0: 'u',
                           1: 'v',
                           2: 'w',
                           3: 'p',
                           4: 'q',
                           5: 'r',
                           6: 'flap1',
                           7: 'flap2',
                           8: 'flap3'}

        table = cout.TablePrinter(n_fields=7, field_length=12, field_types=['s', 'f', 'f', 'f', 'f', 'f', 'f'])
        table.print_header(['der'] + list(labels_force.values()))
        for i in range(der_matrix.shape[1]):
            table.print_line([labels_velocity[i]] + list(der_matrix[:, i]))

        table_coeff = cout.TablePrinter(n_fields=7, field_length=12, field_types=['s']+6*['f'])
        labels_out = {0: 'C_D',
                      1: 'C_Y',
                      2: 'C_L',
                      3: 'C_l',
                      4: 'C_m',
                      5: 'C_n'}
        labels_der = {0: 'u',
                           1: 'v',
                           2: 'w',
                           3: 'p',
                           4: 'q',
                           5: 'r',
                      6: 'alpha',
                      7: 'beta'}
        table_coeff.print_header(['der'] + list(labels_out.values()))
        for i in range(6):
            table_coeff.print_line([labels_der[i]] + list(derivatives_g[:, i]))
        table_coeff.print_line([labels_der[6]] + list(derivatives_g[:, -2]))
        table_coeff.print_line([labels_der[7]] + list(derivatives_g[:, -1]))

        return der_matrix, derivatives_g

    def export_derivatives(self, der_matrix_g):

        folder = self.settings['folder'] + '/' + self.data.settings['SHARPy']['case'] + '/stability/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = 'stability_derivatives.txt'

        u_inf = self.settings['u_inf'].value
        s_ref = self.settings['S_ref'].value
        b_ref = self.settings['b_ref'].value
        c_ref = self.settings['c_ref'].value
        rho = self.data.linear.tsaero0.rho
        euler_orient = algebra.quat2euler(self.data.settings['BeamLoader']['orientation']) * 180/np.pi

        labels_der = {0: 'u',
                           1: 'v',
                           2: 'w',
                           3: 'p',
                           4: 'q',
                           5: 'r',
                      6: 'alpha',
                      7: 'beta'}

        labels_out = {0: 'C_D',
                      1: 'C_Y',
                      2: 'C_L',
                      3: 'C_l',
                      4: 'C_m',
                      5: 'C_n'}

        separator = '\n' + 80*'#' + '\n'

        with open(folder + '/' + filename, mode='w') as outfile:
            outfile.write('SHARPy Stability Derivatives Analysis\n')

            outfile.write('State:\n')
            outfile.write('\t%.4f\t\t\t # Free stream velocity\n' % u_inf)
            outfile.write('\t%.4f\t\t\t # Free stream density\n' % rho)
            outfile.write('\t%.4f\t\t\t # Alpha [deg]\n' % euler_orient[1])
            outfile.write('\t%.4f\t\t\t # Beta [deg]\n' % euler_orient[2])

            outfile.write(separator)
            outfile.write('\nReference Dimensions:\n')
            outfile.write('\t%.4f\t\t\t # Reference planform area\n' % s_ref)
            outfile.write('\t%.4f\t\t\t # Reference chord\n' % c_ref)
            outfile.write('\t%.4f\t\t\t # Reference span\n' % b_ref)

            outfile.write(separator)
            outfile.write('\nCoefficients:\n')
            coeffs = self.static_state()
            for i in range(3):
                outfile.write('\t%.4f\t\t\t # %s\n' % (coeffs[i], labels_out[i]))

            outfile.write(separator)
            for k, v in labels_der.items():
                outfile.write('%s derivatives:\n' % v)
                for i in range(6):
                    outfile.write('\t%.4f\t\t\t # %s_%s derivative\n' % (der_matrix_g[i, k], labels_out[i], labels_der[k]))
                outfile.write(separator)

    def static_state(self):
        fx = np.sum(self.data.aero.timestep_info[0].inertial_steady_forces[:, 0], 0) + \
                     np.sum(self.data.aero.timestep_info[0].inertial_unsteady_forces[:, 0], 0)

        fy = np.sum(self.data.aero.timestep_info[0].inertial_steady_forces[:, 1], 0) + \
             np.sum(self.data.aero.timestep_info[0].inertial_unsteady_forces[:, 1], 0)

        fz = np.sum(self.data.aero.timestep_info[0].inertial_steady_forces[:, 2], 0) + \
             np.sum(self.data.aero.timestep_info[0].inertial_unsteady_forces[:, 2], 0)

        force_coeff = 0.5 * self.data.linear.tsaero0.rho * self.settings['u_inf'].value ** 2 * self.settings['S_ref'].value
        Cfx = fx / force_coeff
        Cfy = fy / force_coeff
        Cfz = fz / force_coeff

        return Cfx, Cfy, Cfz
