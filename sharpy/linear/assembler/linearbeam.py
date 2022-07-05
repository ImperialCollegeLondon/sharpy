"""
Linear State Beam Element Class

"""
import h5py

from sharpy.linear.utils.ss_interface import BaseElement, linear_system, LinearVector
import sharpy.linear.src.lingebm as lingebm
import numpy as np
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra
import sharpy.structure.utils.modalutils as modalutils
from sharpy.linear.utils.ss_interface import VectorVariable, OutputVariable, InputVariable
import sharpy.linear.src.libss as libss


@linear_system
class LinearBeam(BaseElement):
    r"""
    State space member

    Define class for linear state-space realisation of GEBM flexible-body
    equations from SHARPy ``timestep_info`` class and with the nonlinear structural information.

    State-space models can be defined in continuous or discrete time (dt
    required). Modal projection, either on the damped or undamped modal shapes,
    is also available.

    The beam state space has information on the states which will depend on whether the system is modal or expressed
    in physical coordinates.

    If ``modal`` the state variables will be ``q`` and ``q_dot`` representing the modal displacements and the
    time derivatives.

    If ``nodal`` and free-flying, the state variables will be ``eta`` for the flexible degrees of freedom (displacements
    and CRVs for each node (dim6)), ``V`` representing the linear velocities at the A frame (dim3), ``W`` representing
    the angular velocities at the ``A`` frame (dim3), and ``orient`` representing the orientation variable of the
    ``A`` frame with respect to ``G``

    Notes on the settings:

        a. ``modal_projection={True,False}``: determines whether to project the states
            onto modal coordinates. Projection over damped or undamped modal
            shapes can be obtained selecting:

                - ``proj_modes = {'damped','undamped'}``

            while

                 - ``inout_coords={'modes','nodal'}``

             determines whether the modal state-space inputs/outputs are modal
             coords or nodal degrees-of-freedom. If ``modes`` is selected, the
             ``Kin`` and ``Kout`` gain matrices are generated to transform nodal to modal
             dofs

        b. ``dlti={True,False}``: if true, generates discrete-time system.
            The continuous to discrete transformation method is determined by::

                discr_method={ 'newmark',  # Newmark-beta
                                    'zoh',		# Zero-order hold
                                    'bilinear'} # Bilinear (Tustin) transformation

            DLTIs can be obtained directly using the Newmark-:math:`\beta` method

                ``discr_method='newmark'``
                ``newmark_damp=xx`` with ``xx<<1.0``

            for full-states descriptions (``modal_projection=False``) and modal projection
            over the undamped structural modes (``modal_projection=True`` and ``proj_modes``).
            The Zero-order holder and bilinear methods, instead, work in all
            descriptions, but require the continuous state-space equations.


    """
    sys_id = "LinearBeam"

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    settings_default['modal_projection'] = True
    settings_types['modal_projection'] = 'bool'
    settings_description['modal_projection'] = 'Use modal projection'

    settings_default['inout_coords'] = 'nodes'
    settings_types['inout_coords'] = 'str'
    settings_description['inout_coords'] = 'Beam state space input/output coordinates'
    settings_options['inout_coords'] = ['nodes', 'modes']

    settings_types['num_modes'] = 'int'
    settings_default['num_modes'] = 10
    settings_description['num_modes'] = 'Number of modes to retain'

    settings_default['discrete_time'] = True
    settings_types['discrete_time'] = 'bool'
    settings_description['discrete_time'] = 'Assemble beam in discrete time'

    settings_default['dt'] = 0.001
    settings_types['dt'] = 'float'
    settings_description['dt'] = 'Discrete time system integration time step'

    settings_default['proj_modes'] = 'undamped'
    settings_types['proj_modes'] = 'str'
    settings_description['proj_modes'] = 'Use ``undamped`` or ``damped`` modes'
    settings_options['proj_modes'] = ['damped', 'undamped']

    settings_default['discr_method'] = 'newmark'
    settings_types['discr_method'] = 'str'
    settings_description['discr_method'] = 'Discrete time assembly system method:'
    settings_options['discr_method'] = ['newmark', 'zoh', 'bilinear']

    settings_default['newmark_damp'] = 1e-4
    settings_types['newmark_damp'] = 'float'
    settings_description['newmark_damp'] = 'Newmark damping value. For systems assembled using ``newmark``'

    settings_default['use_euler'] = True
    settings_types['use_euler'] = 'bool'
    settings_description['use_euler'] = 'Use euler angles for rigid body parametrisation'

    settings_default['print_info'] = True
    settings_types['print_info'] = 'bool'
    settings_description['print_info'] = 'Display information on screen'

    settings_default['gravity'] = False
    settings_types['gravity'] = 'bool'
    settings_description['gravity'] = 'Linearise gravitational forces'

    settings_types['remove_dofs'] = 'list(str)'
    settings_default['remove_dofs'] = []
    settings_description['remove_dofs'] = 'Remove desired degrees of freedom (flexible DOFs, ' \
                                          'linear velocities, rotational velocities, orientation)'
    settings_options['remove_dofs'] = ['eta', 'V', 'W', 'orient']

    settings_types['remove_sym_modes'] = 'bool'
    settings_default['remove_sym_modes'] = False
    settings_description['remove_sym_modes'] = 'Remove symmetric modes if wing is clamped'

    settings_types['remove_rigid_states'] = 'bool'
    settings_default['remove_rigid_states'] = False
    settings_description['remove_rigid_states'] = '(For Stability Derivatives) - Remove RIGID STATES from SS leaving' \
                                                  ' input/output channels unchanged'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description, settings_options)

    def __init__(self):
        self.sys = None  # The actual object
        self.ss = None  # The state space object
        self.clamped = None
        self.tsstruct0 = None

        self.settings = dict()
        self.state_variables = None
        self.linearisation_vectors = dict()

    def initialise(self, data, custom_settings=None):

        if custom_settings:
            self.settings = custom_settings
        else:
            try:
                self.settings = data.settings['LinearAssembler']['linear_system_settings']
            except KeyError:
                pass
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default,
                                 self.settings_options, no_ctype=True)

        self.settings['rigid_modes_ppal_axes'] = data.settings['Modal']['rigid_modes_ppal_axes']  # use the same value as in Modal solver
        beam = lingebm.FlexDynamic(data.linear.tsstruct0, data.structure, self.settings)
        self.sys = beam
        self.tsstruct0 = data.linear.tsstruct0

        # State variables - for the purposes of dof removal PRIOR to first order system assembly
        num_dof_flex = self.sys.structure.num_dof.value
        num_dof_rig = self.sys.Mstr.shape[0] - num_dof_flex

        # Variables described in class docstring. If modified, remember to change docstring
        if num_dof_rig == 0:
            state_variable_list = [
                VectorVariable('eta', size=num_dof_flex, index=0),
            ]
        else:
            state_variable_list = [
                VectorVariable('eta', size=num_dof_flex, index=0),  # flexible dofs
                VectorVariable('V', size=3, index=1),  # translational velocities
                VectorVariable('W', size=3, index=2),  # angular velocities
                VectorVariable('orient', size=num_dof_rig - 6, index=3),  # orientation
            ]

        self.state_variables = LinearVector(state_variable_list)

        if num_dof_rig == 0:
            self.clamped = True

        self.linearisation_vectors['eta'] = self.tsstruct0.q
        self.linearisation_vectors['eta_dot'] = self.tsstruct0.dqdt
        self.linearisation_vectors['forces_struct'] = self.tsstruct0.steady_applied_forces.reshape(-1, order='C') # B frame

    def assemble(self, t_ref=None):
        """
        Assemble the beam state-space system.

        Args:
            t_ref (float): Scaling factor to non-dimensionalise the beam's time step.

        Returns:

        """
        if self.settings['gravity']:
            self.sys.linearise_gravity_forces()

        # follower force effect
        self.sys.linearise_applied_forces()

        if self.settings['remove_dofs']:
            self.trim_nodes(self.settings['remove_dofs'])

        if self.settings['modal_projection'] and self.settings['remove_sym_modes'] and self.clamped:
            self.remove_symmetric_modes()

        if t_ref is not None:
            self.sys.scale_system_normalised_time(t_ref)

        self.sys.assemble()

        # TODO: remove integrals of the rigid body modes (and change mode shapes to account for this in the coupling matrices)
        # Option to remove certain dofs via dict: i.e. dofs to remove
        # Map dofs to equations
        # Boundary conditions
        # Same with modal, remove certain modes. Need to specify that modes to keep refer to flexible ones only

        if self.sys.SSdisc:
            self.ss = self.sys.SSdisc
        elif self.sys.SScont:
            self.ss = self.sys.SScont

        if self.settings['remove_rigid_states']:
            # Temporary - should be incorporated into StabilityDerivatives and the coupled system reassembled
            self.remove_rigid_states()

        return self.ss

    def remove_rigid_states(self):
        if self.sys.clamped:
            return
        if self.settings['use_euler']:
            num_rig_dof = 9
        else:
            num_rig_dof = 10
        if self.sys.modal:

            self.ss.A = self.ss.A[num_rig_dof:, num_rig_dof:]
            self.ss.B = self.ss.B[num_rig_dof:, :]
            self.ss.C = self.ss.C[:, num_rig_dof:]
            self.ss.state_variables.modify('q', size=self.ss.state_variables[0].size - num_rig_dof)
            self.ss.state_variables.update_locations()

            retain_state = np.zeros((self.ss.states - 9, self.ss.states))
            retain_state[:self.ss.state_variables[0].size, :self.ss.state_variables[0].size] = \
                np.eye(self.ss.state_variables[0].size)
            retain_state[self.ss.state_variables[0].size:, self.ss.state_variables[0].size + 9:] = \
                np.eye(self.ss.state_variables[0].size)
            self.ss.A = retain_state.dot(self.ss.A.dot(retain_state.T))
            self.ss.B = retain_state.dot(self.ss.B)
            self.ss.C = self.ss.C.dot(retain_state.T)
            self.ss.state_variables.modify('q_dot', size=self.ss.state_variables[1].size - 9)
            self.ss.state_variables.update_locations()

        else:
            retain_state = np.zeros((self.ss.states - 2 * num_rig_dof, self.ss.states))
            eta_size = self.ss.state_variables[0].size
            retain_state[:eta_size, :eta_size] = np.eye(eta_size)
            retain_state[eta_size: 2 * eta_size, eta_size + 9: 2 * eta_size + 9] = np.eye(eta_size)

            self.ss.A = retain_state.dot(self.ss.A.dot(retain_state.T))
            self.ss.B = retain_state.dot(self.ss.B)
            self.ss.C = self.ss.C.dot(retain_state.T)
            self.ss.state_variables.remove('beta_bar')
            self.ss.state_variables.remove('beta')
            self.ss.state_variables.update_locations()

    def x0(self):
        x = np.concatenate((self.tsstruct0.q, self.tsstruct0.dqdt))
        return x

    def trim_nodes(self, trim_list=list):
        """
        Removes degrees of freedom from the second order system.

        Args:
            trim_list (list): List of degrees of freedom to remove ``eta``, ``V``, ``W`` or ``orient``

        """

        num_dof_flex = self.sys.structure.num_dof.value

        n_dofs = self.state_variables.size
        self.state_variables.remove(*trim_list)
        removed_dofs = n_dofs - self.state_variables.size
        trim_matrix = np.zeros((n_dofs, n_dofs - removed_dofs))

        for variable in self.state_variables:
            trim_matrix[variable.rows_loc, variable.cols_loc] = 1

        # Update matrices
        self.sys.Mstr = trim_matrix.T.dot(self.sys.Mstr.dot(trim_matrix))
        self.sys.Cstr = trim_matrix.T.dot(self.sys.Cstr.dot(trim_matrix))
        self.sys.Kstr = trim_matrix.T.dot(self.sys.Kstr.dot(trim_matrix))

    def remove_symmetric_modes(self):
        """
        Removes symmetric modes when the wing is clamped at the midpoint.

        It will force the wing tip displacements in ``z`` to be postive for all modes.

        Updates the mode shapes matrix, the natural frequencies and the number of modes.
        """

        # Group modes into symmetric and anti-symmetric modes
        modes_sym = np.zeros_like(self.sys.U)  # grouped modes
        total_modes = self.sys.num_modes

        for i in range(total_modes//2):
            je = 2*i
            jo = 2*i + 1
            modes_sym[:, je] = 1./np.sqrt(2)*(self.sys.U[:, je] + self.sys.U[:, jo])
            modes_sym[:, jo] = 1./np.sqrt(2)*(self.sys.U[:, je] - self.sys.U[:, jo])

        self.sys.U = modes_sym

        # Remove anti-symmetric modes
        # Wing 1 and 2 nodes
        # z-displacement index
        ind_w1 = [6*i + 2 for i in range(self.sys.structure.num_node // 2)]  # Wing 1 nodes are in the first half rows
        ind_w1_x = [6*i for i in range(self.sys.structure.num_node // 2)]  # Wing 1 nodes are in the first half rows
        ind_w1_y = [6*i + 1 for i in range(self.sys.structure.num_node // 2)]  # Wing 1 nodes are in the first half rows
        ind_w2 = [6*i + 2 for i in range(self.sys.structure.num_node // 2, self.sys.structure.num_node - 1)]  # Wing 2 nodes are in the second half rows

        sym_mode_index = []
        for i in range(self.sys.num_modes//2):
            found_symmetric = False

            for j in range(2):
                ind = 2*i + j

                # Maximum z displacement for wings 1 and 2
                ind_max_w1 = np.argmax(np.abs(modes_sym[ind_w1, ind]))
                ind_max_w2 = np.argmax(np.abs(modes_sym[ind_w2, ind]))
                z_max_w1 = modes_sym[ind_w1, ind][ind_max_w1]
                z_max_w2 = modes_sym[ind_w2, ind][ind_max_w2]

                z_max_diff = np.abs(z_max_w1 - z_max_w2)
                if z_max_diff < np.abs(z_max_w1 + z_max_w2):
                    sym_mode_index.append(ind)
                    if found_symmetric:
                        raise NameError('Symmetric Mode previously found')
                    found_symmetric = True

        self.sys.U = modes_sym[:, sym_mode_index]

        # make all elastic modes have a positive z component at the wingtip
        self.sys.U = modalutils.mode_sign_convention(self.sys.structure.boundary_conditions,
                                                     self.sys.U,
                                                     rigid_body_motion=not self.clamped,
                                                     use_euler=self.settings['use_euler'])

        self.sys.freq_natural = self.sys.freq_natural[sym_mode_index]
        self.sys.num_modes = len(self.sys.freq_natural)

    def unpack_ss_vector(self, x_n, u_n, struct_tstep):
        """
        Warnings:
            Under development. Missing:
                * Accelerations
                * Double check the cartesian rotation vector
                * Tangential operator for the moments

        Takes the state :math:`x = [\eta, \dot{\eta}]` and input vectors :math:`u = N` of a linearised beam and returns
        a SHARPy timestep instance, including the reference values.

        Args:
            x_n (np.ndarray): Structural beam state vector in nodal space
            y_n (np.ndarray): Beam input vector (nodal forces)
            struct_tstep (utils.datastructures.StructTimeStepInfo): Reference timestep used for linearisation

        Returns:
            utils.datastructures.StructTimeStepInfo: new timestep with linearised values added to the reference value
        """

        # check if clamped
        vdof = self.sys.structure.vdof
        num_node = struct_tstep.num_node
        num_dof = 6*sum(vdof >= 0)
        if self.sys.clamped:
            clamped = True
            rig_dof = 0
        else:
            clamped = False
            if self.settings['use_euler']:
                rig_dof = 9
            else:
                rig_dof = 10

        q = np.zeros_like(struct_tstep.q)
        q = np.zeros((num_dof + rig_dof))
        dqdt = np.zeros_like(q)
        dqddt = np.zeros_like(q)

        pos = np.zeros_like(struct_tstep.pos)
        pos_dot = np.zeros_like(struct_tstep.pos_dot)
        psi = np.zeros_like(struct_tstep.psi)
        psi_dot = np.zeros_like(struct_tstep.psi_dot)

        for_pos = np.zeros_like(struct_tstep.for_pos)
        for_vel = np.zeros_like(struct_tstep.for_vel)
        for_acc = np.zeros_like(struct_tstep.for_acc)
        quat = np.zeros_like(struct_tstep.quat)

        gravity_forces = np.zeros_like(struct_tstep.gravity_forces)
        total_gravity_forces = np.zeros_like(struct_tstep.total_gravity_forces)
        steady_applied_forces = np.zeros_like(struct_tstep.steady_applied_forces)
        unsteady_applied_forces = np.zeros_like(struct_tstep.unsteady_applied_forces)

        q[:num_dof + rig_dof] = x_n[:num_dof + rig_dof]
        dqdt[:num_dof + rig_dof] = x_n[num_dof + rig_dof:]
        # Missing the forces
        # dqddt = self.sys.Minv.dot(-self.sys.Cstr.dot(dqdt) - self.sys.Kstr.dot(q))

        # for i_node in vdof[vdof >= 0]:
        #     pos[i_node + 1, :] = q[6*i_node: 6*i_node + 3]
        #     pos_dot[i_node + 1, :] = dqdt[6*i_node + 0: 6*i_node + 3]
        #
        # TODO: CRV of clamped node and double check that the CRV takes this form
        # for i_elem in range(struct_tstep.num_elem):
        #     for i_node in range(struct_tstep.num_node_elem):
        #         psi[i_elem, i_node, :] = np.linalg.inv(algebra.crv2tan(struct_tstep.psi[i_elem, i_node]).T).dot(q[i_node + 3: i_node + 6])
        #         psi_dot[i_elem, i_node, :] = dqdt[i_node + 3: i_node + 6]

        pos, psi, pos_dot, psi_dot = self.unpack_flex_dof(q, dqdt)

        if not clamped:
            for_vel = dqdt[-rig_dof: -rig_dof + 6]
            if self.settings['use_euler']:
                euler = dqdt[-4:-1]
                quat = algebra.euler2quat(euler)
            else:
                quat = dqdt[-4:]
            for_pos = q[-rig_dof:-rig_dof + 6]
            for_acc = dqddt[-rig_dof:-rig_dof + 6]

        if u_n is not None:
            cba_m = algebra.get_transformation_matrix('ba')
            for i_node in vdof[vdof >= 0]:
                i_elem = self.sys.structure.node_master_elem[i_node, 0]
                i_local_node = self.sys.structure.node_master_elem[i_node, 1]
                cba = cba_m(struct_tstep.psi[i_elem, i_local_node])
                steady_applied_forces[i_node+1, :3] = cba.dot(u_n[6*i_node: 6*i_node + 3])
                steady_applied_forces[i_node+1, 3:] = cba.dot(u_n[6*i_node + 3: 6*i_node + 6])
            i_elem = self.sys.structure.node_master_elem[0, 0]
            i_local_node = self.sys.structure.node_master_elem[0, 1]
            cba = cba_m(struct_tstep.psi[i_elem, i_local_node])
            steady_applied_forces[0, :3] = cba.dot(u_n[-10:-7]) - np.sum(steady_applied_forces[1:, :3], 0)
            steady_applied_forces[0, 3:] = cba.dot(u_n[-7:-4]) - np.sum(steady_applied_forces[1:, 3:], 0)

        # gravity forces - careful - debug
        C_grav = np.zeros((q.shape[0], q.shape[0]))
        K_grav = np.zeros_like(C_grav)
        try:
            Crr = self.sys.Crr_grav
            Csr = self.sys.Csr_grav
            C_grav[:-rig_dof, -rig_dof:] = Csr # TODO: sort out changing q vector with euler
            C_grav[-rig_dof:, -rig_dof:] = Crr
            K_grav[-rig_dof:, :-rig_dof] = self.sys.Krs_grav
            K_grav[:-rig_dof, :-rig_dof] = self.sys.Kss_grav
            fgrav = -C_grav.dot(dqdt) - K_grav.dot(q)
            for i in range(gravity_forces.shape[0]-1):
                #add bc at node - doing it manually here
                gravity_forces[i+1, :] = fgrav[6*i:6*(i+1)]
            gravity_forces[0, :] = fgrav[-rig_dof:-rig_dof+6] - np.sum(gravity_forces[1:], 0)
        except AttributeError:
            pass

        current_time_step = struct_tstep.copy()
        current_time_step.q[:len(q)] = q + struct_tstep.q[:len(q)]
        current_time_step.dqdt[:len(q)] = dqdt + struct_tstep.dqdt[:len(q)]
        current_time_step.dqddt[:len(q)] = dqddt + struct_tstep.dqddt[:len(q)]
        current_time_step.pos = pos + struct_tstep.pos
        current_time_step.pos_dot = pos + struct_tstep.pos_dot
        current_time_step.psi = psi + struct_tstep.psi
        current_time_step.psi_dot = psi_dot + struct_tstep.psi_dot
        current_time_step.for_vel = for_vel + struct_tstep.for_vel
        current_time_step.for_acc = for_acc + struct_tstep.for_acc
        current_time_step.for_pos = for_pos + struct_tstep.for_pos
        current_time_step.gravity_forces = gravity_forces + struct_tstep.gravity_forces
        current_time_step.total_gravity_forces = total_gravity_forces + struct_tstep.total_gravity_forces
        current_time_step.unsteady_applied_forces = unsteady_applied_forces + struct_tstep.unsteady_applied_forces
        new_quat = quat + struct_tstep.quat
        current_time_step.quat = new_quat/np.linalg.norm(new_quat)
        current_time_step.steady_applied_forces = steady_applied_forces + struct_tstep.steady_applied_forces

        return current_time_step

    def unpack_flex_dof(self, eta, eta_dot=None):
        """
        Unpacks a vector of structural displacements and velocities into a SHARPy familiar
        form of pos, psi and their time derivatives

        Args:
            eta (np.array): Vector of structural displacements
            eta_dot (np.array (Optional): Vector of structural velocities

        Returns:
            tuple: Containing ``pos``, ``psi``, ``pos_dot``, ``psi_dot`` if ``eta_dot`` is provided, else
              only the displacements are returned
        """
        vdof = self.sys.structure.vdof
        if np.max(np.abs(eta.imag)) > 0:
            dtype=complex
        else:
            dtype=float
        pos = np.zeros_like(self.tsstruct0.pos, dtype=dtype)
        psi = np.zeros_like(self.tsstruct0.psi, dtype=dtype)
        pos_dot = np.zeros_like(self.tsstruct0.pos_dot, dtype=dtype)
        psi_dot = np.zeros_like(self.tsstruct0.psi_dot, dtype=dtype)

        return_vels = True
        if eta_dot is None:
            return_vels = False
            eta_dot = np.zeros_like(eta)

        for i_node in vdof[vdof >= 0]:
            pos[i_node + 1, :] = eta[6*i_node: 6*i_node + 3]
            pos_dot[i_node + 1, :] = eta_dot[6*i_node + 0: 6*i_node + 3]

        # TODO: CRV of clamped node and double check that the CRV takes this form
        for i_elem in range(self.tsstruct0.num_elem):
            for i_node in range(self.tsstruct0.num_node_elem):
                psi[i_elem, i_node, :] = np.linalg.inv(
                    algebra.crv2tan(self.tsstruct0.psi[i_elem, i_node]).T).dot(eta[i_node + 3: i_node + 6])
                psi_dot[i_elem, i_node, :] = eta_dot[i_node + 3: i_node + 6]

        if return_vels:
            return pos, psi, pos_dot, psi_dot
        else:
            return pos, psi

    def recover_accelerations(self, full_ss):
        """
        For a system with displacement and velocity outputs (``full_ss``), recover the accelerations and append them
        as new output channels.

        This function produces an output gain that should then be connected in series to the desired system

        Args:
            full_ss (libss.StateSpace): State space for which to provide output gain to recover accelerations

        Returns:
            libss.Gain: Gain adding the accelerations as new output channels
        """
        n_in = full_ss.outputs
        n_out = full_ss.outputs + self.ss.states // 2

        acc_gain = np.zeros((n_out, n_in))

        input_variables = LinearVector.transform(full_ss.output_variables, to_type=InputVariable)

        output_variables = full_ss.output_variables.copy()
        acceleration_variables = []
        for var in self.ss.output_variables[self.ss.output_variables.num_variables//2:]:
            new_var = var.copy()
            new_var.name += '_dot'
            output_variables.append(new_var)
            acceleration_variables.append(new_var)

        acc_gain[:n_in, :n_in] = np.eye(n_in)
        acc_gain[-self.ss.states//2:, :self.ss.inputs] = self.ss.B[self.ss.states//2:]
        acc_gain[-self.ss.states//2:, self.ss.inputs:] = self.ss.A[self.ss.states//2:, :]

        acceleration_recovery = libss.Gain(acc_gain,
                                           input_vars=input_variables,
                                           output_vars=output_variables)

        if self.sys.modal:
            output_variables = []
            for var in self.sys.Kout.output_variables[self.sys.Kout.output_variables.num_variables//2:]:
                new_var = var.copy()
                new_var.name += '_dot'
                output_variables.append(new_var)
            modal_gain = libss.Gain(self.sys.U,
                                    input_vars=LinearVector.transform(LinearVector(acceleration_variables),
                                                                      to_type=InputVariable),
                                    output_vars=LinearVector(output_variables))
            self.sys.acceleration_modal_gain = modal_gain

        return acceleration_recovery

    def save_reduced_order_bases(self, file_name):
        gain = libss.Gain(self.sys.U)
        gain.save(file_name)

    def save_structural_matrices(self, file_name):
        with h5py.File(file_name, 'w') as f:
            f.create_dataset('m', data=self.sys.Mstr)
            f.create_dataset('c', data=self.sys.Cstr)
            f.create_dataset('k', data=self.sys.Kstr)
