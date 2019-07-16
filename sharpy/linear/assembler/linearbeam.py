"""
Linear State Beam Element Class

"""

from sharpy.linear.utils.ss_interface import BaseElement, linear_system, LinearVector
import sharpy.linear.src.lingebm as lingebm
import numpy as np
import sharpy.utils.settings as settings


@linear_system
class LinearBeam(BaseElement):
    """
    State space member
    """
    sys_id = "LinearBeam"


    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['gravity'] = 'bool'
    settings_default['gravity'] = False

    settings_types['remove_dofs'] = 'list'
    settings_default['remove_dofs'] = []

    settings_types['remove_sym_modes'] = 'bool'
    settings_default['remove_sym_modes'] = False
    settings_description['remove_sym_modes'] = 'Remove symmetric modes if wing is clamped'

    def __init__(self):
        self.sys = None  # The actual object
        self.ss = None  # The state space object
        self.clamped = None
        self.tsstruct0 = None

        self.settings = dict()
        self.state_variables = None

    def initialise(self, data, custom_settings=None):

        if custom_settings:
            self.settings = custom_settings
        else:
            try:
                self.settings = data.settings['LinearAssembler'][self.sys_id]
            except KeyError:
                pass
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

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

    def assemble(self):
        # Would assemble the system as per the settings
        # Here we would add further options such as discarding DOFs etc

        # linearise then trim
        if self.settings['gravity'].value:
            self.sys.linearise_gravity_forces()

        if self.settings['remove_dofs']:
            self.trim_nodes(self.settings['remove_dofs'])

        if self.settings['modal_projection'].value and self.settings['remove_sym_modes'].value and self.clamped:
            self.remove_symmetric_modes()

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

    def x0(self):
        x = np.concatenate((self.tsstruct0.q, self.tsstruct0.dqdt))
        return x

    def trim_nodes(self, trim_list=list):

        num_dof_flex = self.sys.structure.num_dof
        num_dof_rig = self.sys.Mstr.shape[0] - num_dof_flex

        # Dictionary containing DOFs and corresponding equations
        dof_db = {'eta': [0, num_dof_flex, 1],
                  'V': [num_dof_flex, num_dof_flex + 3, 2],
                  'W': [num_dof_flex + 3, num_dof_flex + 6, 3],
                  'orient': [num_dof_flex + 6, num_dof_flex + num_dof_rig, 4]}

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
                if used_vars_db[item].first_pos < removed_db[rem_item].first_pos:
                    pass
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
        if len(x_n) == 2 * num_dof:
            clamped = True
            rig_dof = 0
        else:
            clamped = False
            rig_dof = 10

        q = np.zeros_like(struct_tstep.q)
        dqdt = np.zeros_like(struct_tstep.dqdt)
        dqddt = np.zeros_like(struct_tstep.dqddt)

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
                psi[i_elem, i_node, :] = q[i_node + 3: i_node + 6]
                psi_dot[i_elem, i_node, :] = dqdt[i_node + 3: i_node + 6]

        if not clamped:
            for_vel = dqdt[-rig_dof: -rig_dof + 6]
            quat = dqdt[-4:]
            for_pos = q[-rig_dof:-rig_dof + 6]
            for_acc = dqddt[-rig_dof:-rig_dof + 6]

        if u_n is not None:
            for i_node in vdof[vdof >= 0]:
                steady_applied_forces[i_node+1] = u_n[6*i_node: 6*i_node + 6]

        current_time_step = struct_tstep.copy()
        current_time_step.q = q + struct_tstep.q
        current_time_step.dqdt = dqdt + struct_tstep.dqdt
        current_time_step.dqddt = dqddt + struct_tstep.dqddt
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
