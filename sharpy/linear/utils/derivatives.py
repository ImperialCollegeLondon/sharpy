import h5py
import numpy as np

from sharpy.utils import algebra as algebra, cout_utils as cout


class Derivatives:
    """
    Class containing the derivatives set for a given state-space system (i.e. aeroelastic or aerodynamic)
    """
    def __init__(self, reference_dimensions, static_state, target_system=None):

        self.target_system = target_system  # type: str # name of target system (aerodynamic/aeroelastic)
        self.transfer_function = None  # type: np.array # matrix of steady-state TF for target system

        self.static_state = static_state  # type: tuple # [fx, fy, fz] at ref state
        self.reference_dimensions = reference_dimensions  # type: dict # name: ref_dimension_value dictionary

        self.separator = '\n' + 80 * '#' + '\n'

        self.dict_of_derivatives = {}  # type: dict # {name:DerivativeSet} Each of the derivative sets DerivativeSet

        s_ref = self.reference_dimensions['S_ref']
        b_ref = self.reference_dimensions['b_ref']
        c_ref = self.reference_dimensions['c_ref']
        u_inf = self.reference_dimensions['u_inf']
        rho = self.reference_dimensions['rho']
        self.dynamic_pressure = 0.5 * rho * u_inf ** 2

        self.coefficients = {'force': self.dynamic_pressure * s_ref,
                             'moment_lon': self.dynamic_pressure * s_ref * c_ref,
                             'moment_lat': self.dynamic_pressure * s_ref * b_ref,
                             'force_angular_vel': self.dynamic_pressure * s_ref * c_ref / u_inf,
                             'moment_lon_angular_vel': self.dynamic_pressure * s_ref * c_ref * c_ref / u_inf}  # missing rates

        self.steady_coefficients = np.array(self.static_state) / self.coefficients['force']

        self.filename = 'stability_derivatives.txt'
        if target_system is not None:
            self.filename = target_system + '_' + self.filename

        self.cg = None

    def initialise_derivatives(self, state_space, steady_forces, quat, v0, phi=None, cg=None, tpa=None):
        """
        Initialises the required class attributes for all derivative calculations/

        Args:
            state_space (sharpy.linear.src.libss.StateSpace): State-space object for the target system
            steady_forces (np.array): Array of steady forces (at the linearisation) expressed in the beam degrees of
              freedom and with size equal to the number of structural degrees of freedom
            quat (np.array): Quaternion at the linearisation reference state
            v0 (np.array): Free stream velocity vector at the linearisation condition
            phi (np.array (optional)): Mode shape matrix for modal systems
            tpa (np.array (optional)): Transformation matrix onto principal axes

        """
        cls = DerivativeSet  # explain what is the DerivativeSet class
        if cls.quat is None:
            cls.quat = quat
            cls.cga = algebra.quat2rotation(cls.quat)
            cls.v0 = v0
            cls.coefficients = self.coefficients

            if phi is not None:
                cls.modal = True
                cls.phi = phi[-9:-3, :6]
                cls.inv_phi_forces = np.linalg.inv(phi[-9:-3, :6].T)
                cls.inv_phi_vel = np.linalg.inv(phi[-9:-3, :6])
            else:
                cls.modal = False
        cls.steady_forces = steady_forces

        H0 = state_space.freqresp(np.array([1e-5]))[:, :, 0].real
        if cls.modal:
            vel_inputs_variables = state_space.input_variables.get_variable_from_name('q_dot')
            output_indices = state_space.output_variables.get_variable_from_name('Q').rows_loc[:6]
            cls.steady_forces = cls.inv_phi_forces.dot(cls.steady_forces[output_indices])
        else:
            vel_inputs_variables = state_space.input_variables.get_variable_from_name('beta')
            output_indices = state_space.output_variables.get_variable_from_name('forces_n').rows_loc[-9:-3]
            cls.steady_forces = cls.steady_forces[output_indices]
        rbm_indices = vel_inputs_variables.cols_loc[:9]

        # look for control surfaces
        try:
            cs_input_variables = state_space.input_variables.get_variable_from_name('control_surface_deflection')
            dot_cs_input_variables = state_space.input_variables.get_variable_from_name('dot_control_surface_deflection')
        except ValueError:
            cs_indices = np.array([], dtype=int)
            dot_cs_indices = np.array([], dtype=int)
            cls.n_control_surfaces = 0
        else:
            cs_indices = cs_input_variables.cols_loc
            dot_cs_indices = dot_cs_input_variables.cols_loc
            cls.n_control_surfaces = cs_input_variables.size
        finally:
            input_indices = np.concatenate((rbm_indices, cs_indices, dot_cs_indices))

        self.transfer_function = H0[np.ix_(output_indices, input_indices)].real

        self.cg = cg
        self.tpa = tpa

    def save(self, output_route):
        with h5py.File(output_route + '/' + self.filename.replace('.txt', '.h5'), 'w') as f:
            for k, v in self.dict_of_derivatives.items():
                if v.matrix is None:
                    continue
                f.create_dataset(name=k, data=v.matrix)

    def savetxt(self, folder):

        filename = self.filename

        u_inf = self.reference_dimensions['u_inf']
        s_ref = self.reference_dimensions['S_ref']
        b_ref = self.reference_dimensions['b_ref']
        c_ref = self.reference_dimensions['c_ref']
        rho = self.reference_dimensions['rho']
        quat = self.reference_dimensions['quat']
        euler_orient = algebra.quat2euler(quat) * 180/np.pi

        labels_out = ['CD', 'CY', 'CL', 'Cl', 'Cm', 'Cn']

        separator = '\n' + 80*'#' + '\n'

        with open(folder + '/' + filename, mode='w') as outfile:
            outfile.write('SHARPy Stability Derivatives Analysis\n')

            outfile.write('State:\n')
            outfile.write('\t{:4f}\t\t\t # Free stream velocity\n'.format(u_inf))
            outfile.write('\t{:4f}\t\t\t # Free stream density\n'.format(rho))
            outfile.write('\t{:4f}\t\t\t # Alpha [deg]\n'.format(euler_orient[1]))
            outfile.write('\t{:4f}\t\t\t # Beta [deg]\n'.format(euler_orient[2]))

            if self.cg is not None:
                outfile.write(separator)
                outfile.write('Centre of Gravity:\n')
                lab = ('x', 'y', 'z')
                for i in range(3):
                    outfile.write('\t{:s}_A = {:.4f}\t\t\t # [m]\n'.format(lab[i], self.cg[i]))

                if self.tpa is not None:
                    outfile.write('Principal Axes Directions (expressed in the A frame):\n')
                    for i in range(3):
                        outfile.write('\t{:s}_ppal in A = [{:.4f}, {:.4f}, {:.4f}]\t\t\t\n'.format(
                            lab[i], *self.tpa.dot(np.eye(3)[:, i])))

            outfile.write(separator)
            outfile.write('\nReference Dimensions:\n')
            outfile.write('\t{:4f}\t\t\t # Reference planform area\n'.format(s_ref))
            outfile.write('\t{:4f}\t\t\t # Reference chord\n'.format(c_ref))
            outfile.write('\t{:4f}\t\t\t # Reference span\n'.format(b_ref))

            outfile.write(separator)
            outfile.write('\nCoefficients:\n')
            for ith, coeff in enumerate(self.steady_coefficients):
                outfile.write('\t{:4e}\t\t\t # {:s}\n'.format(coeff,  labels_out[ith]))

            outfile.write(separator)

        # this needs to be out of with open as it is done in each of the Derivatives objects
        for derivative_set in self.dict_of_derivatives.values():
            if derivative_set.matrix is None:
                continue
            derivative_set.print(derivative_filename=folder + '/' + filename)

    def new_derivative(self, frame_of_reference, derivative_calculation=None, name=None):
        """
        Returns a DerivativeSet() instance with the appropriate transfer function included
        for the relevant target system.

        Args:
            frame_of_reference (str): Output frame of reference. Body or Stability. (Not Yet Implemented)
            derivative_calculation (str): Name of function used to create derivative set
            name (str (optional)): Optional custom name to use as title in output.

        Returns:
            DerivativeSet: Instance of class with the relevant transfer function for the current
              target system.
        """
        new_derivative = DerivativeSet(frame_of_reference, derivative_calculation,
                                       name,
                                       transfer_function=self.transfer_function)

        return new_derivative


class DerivativeSet:
    """
    Class containing the stability derivative set for each of the input/output combinations. A derivative set may be
    force/angle or force/control_surface, for example.

    The class attributes contain the parameters common across all derivative sets. The instance attributes those
    pertaining to the specific set.
    """
    steady_forces = None
    coefficients = None
    quat = None

    cga = None
    n_control_surfaces = None

    v0 = None

    # Modal cases
    modal = None
    phi = None
    inv_phi_forces = None
    inv_phi_vel = None

    def __init__(self, frame_of_reference, derivative_calculation=None, name=None,
                 transfer_function=None):
        """

        Args:
            frame_of_reference (str): Name of the frame of reference (stability or body axes)
            derivative_calculation (str): Name of the method to compute derivatives
            name (str): Name of the derivative set
            transfer_function (np.array): steady state transfer function for the desired input output channels
        """

        self.transfer_function = transfer_function  # type: np.array # steady-state TF for the specific out/in
        self.matrix = None  # type: np.array # matrix of stability derivatives
        self.labels_in = []  # type: list(str) # strings describing the input channels
        self.labels_out = []  # type list(str) # strings describing the output channels
        self.frame_of_reference = frame_of_reference  # type: str # name of the FoR (stability or body axes)

        self.table = None  # type: cout.TablePrinter
        self.name = name  # type: str # name of the set

        # TODO: remove in clean up and make derivative_calculation a position argument
        if derivative_calculation is not None:
            self.__getattribute__(derivative_calculation)()

    def print(self, derivative_filename=None):
        if self.name is not None:
            cout.cout_wrap(self.name)
            with open(derivative_filename, 'a') as f:
                f.write('Derivative set: {:s}\n'.format(self.name))
                f.write('Axes {:s}\n'.format(self.frame_of_reference))
        self.table = cout.TablePrinter(n_fields=len(self.labels_in)+1,
                                       field_types=['s']+len(self.labels_in) * ['e'], filename=derivative_filename)
        self.table.print_header(field_names=list(['der'] + self.labels_in))
        for i in range(len(self.labels_out)):
            out_list = [self.labels_out[i]] + list(self.matrix[i, :])
            self.table.print_line(out_list)
        self.table.print_divider_line()
        self.table.character_return(n_lines=2)

    def save(self, derivative_name, output_name):
        with h5py.File(output_name + '.stability.h5', 'w') as f:
            f.create_dataset(derivative_name, data=self.matrix)

    def angle_derivatives(self):
        r"""
        Stability derivatives against aerodynamic angles (angle of attack and sideslip) expressed in stability axes, i.e
        forces are lift, drag...

        Linearised forces in stability axes are expressed as

        .. math::
            F^S = F_0^S + \frac{\partial}{\partial \alpha}\left(C^{GA}(\alpha)F_0^A\right)\delta\alpha + C_0^{GA}\delta F^A

        Therefore, the stability derivative becomes

        .. math:: \frac{\partial\F^S}{\partial\alpha} =\frac{\partial}{\partial \alpha}\left(C^{GA}(\alpha)F_0^A\right) +
           C_0^{GA}\frac{\partial F^A}{\partial\alpha}

        where

        .. math:: \frac{\partial F^A}{\partial\alpha} = \frac{\partial F^A}{\partial v^A}\frac{\partial v^A}{\partial\alpha}

        and

        .. math:: \frac{\partial v^A}{\partial\alpha} = C^{AG}\frac{\partial}{\partial\alpha}\left(C(0)V_0^G\right).

        The term :math:`\frac{\partial F^A}{\partial v^A}` is obtained directly from the steady state transfer
        function of the linear UVLM expressed in the beam degrees of freedoms.

        """
        self.labels_in = ['phi', 'alpha', 'beta']
        self.labels_out = ['CD', 'CY', 'CL', 'Cl', 'Cm', 'Cn']
        self.matrix = np.zeros((6, 3))

        # Get free stream velocity direction
        v0 = self.v0

        f0a = self.steady_forces[:3]
        m0a = self.steady_forces[-3:]

        euler0 = algebra.quat2euler(self.quat)
        cga = self.cga

        # first term in the stability derivative expression
        stab_der_trans = algebra.der_Ceuler_by_v(euler0, f0a)
        stab_der_mom = algebra.der_Ceuler_by_v(euler0, m0a)

        # second term in the stability derivative expression
        if self.modal:
            delta_nodal_vel = np.linalg.inv(self.phi[:3, :3]).dot(cga.T.dot(algebra.der_Peuler_by_v(euler0 * 0, v0)))
            delta_nodal_forces = self.inv_phi_forces.dot(self.transfer_function[:6, :3].real.dot(delta_nodal_vel))
        else:
            delta_nodal_vel = cga.T.dot(algebra.der_Peuler_by_v(euler0 * 0, v0))
            delta_nodal_forces = self.transfer_function[:6, :3].real.dot(delta_nodal_vel)

        stab_der_trans2 = cga.dot(delta_nodal_forces[:3, :])
        stab_der_mom2 = cga.dot(delta_nodal_forces[3:, :])

        self.matrix[:3, :] = stab_der_trans + stab_der_trans2
        self.matrix[3:6, :] = stab_der_mom + stab_der_mom2

        self.apply_coefficients()

    def angle_derivatives_tb(self):
        self.name = 'Force/Angle derivatives via angle input'
        self.labels_in = ['phi', 'alpha', 'beta']
        self.labels_out = ['CD', 'CY', 'CL', 'Cl', 'Cm', 'Cn']
        self.matrix = np.zeros((6, 3))

        # Get free stream velocity direction
        v0 = self.v0

        f0a = self.steady_forces[:3]
        m0a = self.steady_forces[-3:]

        euler0 = algebra.quat2euler(self.quat)
        cga = self.cga

        # first term in the stability derivative expression
        stab_der_trans = algebra.der_Ceuler_by_v(euler0, f0a)
        stab_der_mom = algebra.der_Ceuler_by_v(euler0, m0a)

        # second term in the stability derivative expression
        if self.modal:
            delta_nodal_forces = self.inv_phi_forces.dot(self.transfer_function[:6, 6:9].real)
        else:
            delta_nodal_forces = self.transfer_function[:6, 6:9].real

        stab_der_trans2 = cga.dot(delta_nodal_forces[:3, :])
        stab_der_mom2 = cga.dot(delta_nodal_forces[3:, :])

        self.matrix[:3, :] = stab_der_trans + stab_der_trans2
        self.matrix[3:6, :] = stab_der_mom + stab_der_mom2

        self.apply_coefficients()

    def body_derivatives(self):
        self.name = 'Force derivatives to rigid body velocities - Body derivatives'
        self.labels_in = ['uA', 'vA', 'wA', 'pA', 'qA', 'rA']
        self.labels_out = ['C_XA', 'C_YA', 'C_ZA', 'C_LA', 'C_MA', 'C_NA']
        self.matrix = np.zeros((6, 6))

        body_derivatives = self.transfer_function[:6, :6]

        if self.modal:
            body_derivatives = self.inv_phi_forces.dot(body_derivatives).dot(self.inv_phi_vel)

        self.matrix = body_derivatives
        self.apply_coefficients()

    def control_surface_derivatives(self):
        n_control_surfaces = self.n_control_surfaces
        if n_control_surfaces == 0:
            return None

        self.name = 'Force derivatives wrt control surface inputs - Body axes'
        self.labels_out = ['C_XA', 'C_YA', 'C_ZA', 'C_LA', 'C_MA', 'C_NA']
        labels_in_deflection = []
        labels_in_rate = []
        for i in range(n_control_surfaces):
            labels_in_deflection.append('delta_{:g}'.format(i))
            labels_in_rate.append('dot(delta)_{:g}'.format(i))
        self.labels_in = labels_in_deflection + labels_in_rate

        body_derivatives = self.transfer_function[:6, 9:]
        assert body_derivatives.shape == (6, 2 * self.n_control_surfaces), 'Incorrect TF shape'

        if self.modal:
            self.matrix = self.inv_phi_forces.dot(body_derivatives)
        else:
            self.matrix = body_derivatives

        self.apply_coefficients()

    def apply_coefficients(self):
        self.matrix[:3, :] /= self.coefficients['force']
        self.matrix[np.ix_([3, 5]), :] /= self.coefficients['moment_lat']
        self.matrix[4, :] /= self.coefficients['moment_lon']
