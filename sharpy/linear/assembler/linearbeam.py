"""
Linear State Beam Element Class

"""

from sharpy.linear.utils.ss_interface import BaseElement, linear_system, LinearVector
import sharpy.linear.src.lingebm as lingebm
import numpy as np
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra


@linear_system
class LinearBeam(BaseElement):
    r"""
    State space member

    Define class for linear state-space realisation of GEBM flexible-body
    equations from SHARPy``timestep_info`` class and with the nonlinear structural information.

    State-space models can be defined in continuous or discrete time (dt
    required). Modal projection, either on the damped or undamped modal shapes,
    is also avaiable.

    To produce the state-space equations:

    Notes on the settings:

        a. ``modal_projection={True,False}``: determines whether to project the states
            onto modal coordinates. Projection over damped or undamped modal
            shapes can be obtained selecting:

                - ``proj_modes={'damped','undamped'}``

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

    settings_default['modal_projection'] = True
    settings_types['modal_projection'] = 'bool'
    settings_description['modal_projection'] = 'Use modal projection'

    settings_default['inout_coords'] = 'nodes'
    settings_types['inout_coords'] = 'str'
    settings_description['inout_coords'] = 'Beam state space input/output coordinates. ``modes`` or ``nodes``'

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

    settings_default['discr_method'] = 'newmark'
    settings_types['discr_method'] = 'str'
    settings_description['discr_method'] = 'Discrete time assembly system method: ``newmark`` or ``zoh``'

    settings_default['newmark_damp'] = 1e-4
    settings_types['newmark_damp'] = 'float'
    settings_description['newmark_damp'] = 'Newmark damping value. For systems assembled using ``newmark``'

    settings_default['use_euler'] = False
    settings_types['use_euler'] = 'bool'
    settings_description['use_euler'] = 'Use euler angles for rigid body parametrisation'

    settings_default['print_info'] = True
    settings_types['print_info'] = 'bool'
    settings_description['print_info'] = 'Display information on screen'

    settings_default['gravity'] = False
    settings_types['gravity'] = 'bool'
    settings_description['gravity'] = 'Linearise gravitational forces'

    settings_types['remove_dofs'] = 'list'
    settings_default['remove_dofs'] = []
    settings_description['remove_dofs'] = 'Remove desired degrees of freedom: ``eta``, ``V``, ``W`` or ``orient``'

    settings_types['remove_sym_modes'] = 'bool'
    settings_default['remove_sym_modes'] = False
    settings_description['remove_sym_modes'] = 'Remove symmetric modes if wing is clamped'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

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
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default, no_ctype=True)

        beam = lingebm.FlexDynamic(data.linear.tsstruct0, data.structure, self.settings)
        self.sys = beam
        self.tsstruct0 = data.linear.tsstruct0

        # State variables
        num_dof_flex = self.sys.structure.num_dof.value
        num_dof_rig = self.sys.Mstr.shape[0] - num_dof_flex
        state_db = {'eta': [0, num_dof_flex],
                  'V_bar': [num_dof_flex, num_dof_flex + 3],
                  'W_bar': [num_dof_flex + 3, num_dof_flex + 6],
                  'orient_bar': [num_dof_flex + 6, num_dof_flex + num_dof_rig],
                  'dot_eta': [num_dof_flex + num_dof_rig, 2 * num_dof_flex + num_dof_rig],
                  'V': [2 * num_dof_flex + num_dof_rig, 2 * num_dof_flex + num_dof_rig + 3],
                  'W': [2 * num_dof_flex + num_dof_rig + 3, 2 * num_dof_flex + num_dof_rig + 6],
                  'orient': [2 * num_dof_flex + num_dof_rig + 6, 2 * num_dof_flex + 2 * num_dof_rig]}
        self.state_variables = LinearVector(state_db, self.sys_id)

        if num_dof_rig == 0:
            self.clamped = True

        self.linearisation_vectors['eta'] = self.tsstruct0.q
        self.linearisation_vectors['eta_dot'] = self.tsstruct0.dqdt
        self.linearisation_vectors['forces_struct'] = self.tsstruct0.steady_applied_forces.reshape(-1, order='C')

    def assemble(self, t_ref=None):
        """
        Assemble the beam state-space system.

        Args:
            t_ref (float): Scaling factor to non-dimensionalise the beam's time step.

        Returns:

        """
        if self.settings['gravity'].value:
            self.sys.linearise_gravity_forces()

        if self.settings['remove_dofs']:
            self.trim_nodes(self.settings['remove_dofs'])

        if self.settings['modal_projection'].value and self.settings['remove_sym_modes'].value and self.clamped:
            self.remove_symmetric_modes()

        if t_ref is not None:
            self.sys.scale_system_normalised_time(t_ref)

        # import sharpy.linear.assembler.linearthrust as linearthrust
        # engine = linearthrust.LinearThrust()
        # engine.initialise()

        # K_thrust = engine.generate(self.tsstruct0, self.sys)
        #
        # self.sys.Kstr += K_thrust

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

        return self.ss

    def x0(self):
        x = np.concatenate((self.tsstruct0.q, self.tsstruct0.dqdt))
        return x

    def trim_nodes(self, trim_list=list):

        num_dof_flex = self.sys.structure.num_dof.value
        num_dof_rig = self.sys.Mstr.shape[0] - num_dof_flex

        # Dictionary containing DOFs and corresponding equations
        dof_db = {'eta': [0, num_dof_flex, 1],
                  'V': [num_dof_flex, num_dof_flex + 3, 2],
                  'W': [num_dof_flex + 3, num_dof_flex + 6, 3],
                  'orient': [num_dof_flex + 6, num_dof_flex + num_dof_rig, 4],
                  'yaw': [num_dof_flex + 8, num_dof_flex + num_dof_rig, 1]}

        # -----------------------------------------------------------------------
        # Better to place in a function available to all elements since it will equally apply
        # Therefore, the dof_db should be a class attribute
        # Take away alongside the vector variable class

        # All variables
        vec_db = dict()
        for item in dof_db:
            vector_var = VectorVariable(item, dof_db[item], 'LinearBeam')
            vec_db[item] = vector_var

        used_vars_db = vec_db.copy()

        # Variables to remove
        removed_dofs = 0
        removed_db = dict()
        for item in trim_list:
            removed_db[item] = vec_db[item]
            removed_dofs += vec_db[item].size
            del used_vars_db[item]

        # Update variables position
        for rem_item in removed_db:
            for item in used_vars_db:
                if used_vars_db[item].rows_loc[0] < removed_db[rem_item].first_pos:
                    continue
                else:
                    # Update order and position
                    used_vars_db[item].first_pos -= removed_db[rem_item].size
                    used_vars_db[item].end_pos -= removed_db[rem_item].size

        self.state_variables = used_vars_db
        # TODO: input and output variables
        ### ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        # Map dofs to equations
        trim_matrix = np.zeros((num_dof_rig+num_dof_flex, num_dof_flex+num_dof_rig-removed_dofs))
        for item in used_vars_db:
            trim_matrix[used_vars_db[item].rows_loc, used_vars_db[item].cols_loc] = 1

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
        for i in range(self.sys.U.shape[1]):
            if np.abs(self.sys.U[ind_w1[-1], i]) > 1e-10:
                self.sys.U[:, i] = np.sign(self.sys.U[ind_w1[-1], i]) * self.sys.U[:, i]
            elif np.abs(self.sys.U[ind_w1_y, i][-1]) > 1e-4:
                self.sys.U[:, i] = np.sign(self.sys.U[ind_w1_y[-1], i]) * self.sys.U[:, i]

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

        for i_node in vdof[vdof >= 0]:
            pos[i_node + 1, :] = q[6*i_node: 6*i_node + 3]
            pos_dot[i_node + 1, :] = dqdt[6*i_node + 0: 6*i_node + 3]

        # TODO: CRV of clamped node and double check that the CRV takes this form
        for i_elem in range(struct_tstep.num_elem):
            for i_node in range(struct_tstep.num_node_elem):
                psi[i_elem, i_node, :] = np.linalg.inv(algebra.crv2tan(struct_tstep.psi[i_elem, i_node]).T).dot(q[i_node + 3: i_node + 6])
                psi_dot[i_elem, i_node, :] = dqdt[i_node + 3: i_node + 6]

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
            for i_node in vdof[vdof >= 0]:
                steady_applied_forces[i_node+1] = u_n[6*i_node: 6*i_node + 6]
            steady_applied_forces[0] = u_n[-10:-4] - np.sum(steady_applied_forces[1:, :], 0)

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

    def rigid_aero_forces(self):

        # Debug adding rigid forces from tornado
        derivatives_alpha = np.zeros((6, 5))
        derivatives_alpha[0, :] = np.array([0.0511, 0, 0, 0.08758, 0])  # drag derivatives
        derivatives_alpha[1, :] = np.array([0, 0, -0.05569, 0, 0])  # Y derivatives
        derivatives_alpha[2, :] = np.array([5.53, 0, 0, 11.35, 0])  # lift derivatives
        derivatives_alpha[3, :] = np.array([0, 0, -0.609, 0, 0])  # roll derivatives
        derivatives_alpha[4, :] = np.array([-9.9988, 0, 0, -37.61, 0]) # pitch derivatives
        derivatives_alpha[5, :] = np.array([0, 0, -0.047, 0, 0])  # yaw derivatives

        Cx0 = -0.0324
        Cz0 = 0.436
        Cm0 = -0.78966


        quat = self.tsstruct0.quat
        Cga = algebra.quat2rotation(quat)



class VectorVariable(object):

    def __init__(self, name, pos_list, var_system):

        self.name = name
        self.var_system = var_system

        self.first_pos = pos_list[0]
        self.end_pos = pos_list[1]
        self.rows_loc = np.arange(self.first_pos, self.end_pos, dtype=int) # Original location, should not update


    # add methods to reorganise into SHARPy method?

    @property
    def cols_loc(self):
        return np.arange(self.first_pos, self.end_pos, dtype=int)

    @property
    def size(self):
        return self.end_pos - self.first_pos
