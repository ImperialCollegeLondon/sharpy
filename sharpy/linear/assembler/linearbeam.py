"""
Linear State Beam Element Class

"""

from sharpy.linear.utils.ss_interface import BaseElement, linear_system
import sharpy.linear.src.lingebm as lingebm
import numpy as np
import sharpy.utils.settings as settings

@linear_system
class LinearBeam(BaseElement):
    """
    State space member
    """
    sys_id = "LinearBeam"

    def __init__(self):

        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['gravity'] = 'bool'
        self.settings_default['gravity'] = False

        self.settings_types['remove_dofs'] = 'list'
        self.settings_default['remove_dofs'] = []

        self.sys = None  # The actual object
        self.ss = None  # The state space object
        self.tstruct0 = None

        self.data = None
        self.settings = dict()
        self.state_variables = None

    def initialise(self, data, custom_settings=None):

        self.data = data
        if custom_settings:
            self.settings = custom_settings
        else:
            try:
                self.settings = data.settings['LinearAssembler'][self.sys_id]  # Load settings, the settings should be stored in data.linear.settings
            except KeyError:
                pass
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        beam = lingebm.FlexDynamic(data.linear.tsstruct0, data.structure, self.settings)
        self.sys = beam
        self.tstruct0 = data.linear.tsstruct0


    def assemble(self):
        # Would assemble the system as per the settings
        # Here we would add further options such as discarding DOFs etc

        # linearise then trim
        if self.settings['gravity'].value:
            self.sys.linearise_gravity_forces()

        if self.settings['remove_dofs']:
            self.trim_nodes(self.settings['remove_dofs'])

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
        x = np.concatenate((self.tstruct0.q, self.tstruct0.dqdt))
        return x


    def trim_nodes(self, trim_list=list):

        num_dof_flex = self.data.structure.num_dof
        num_dof_rig = self.sys.Mstr.shape[0] - num_dof_flex

        # Dictionary containing DOFs and corresponding equations
        dof_db = {'eta': [0, num_dof_flex, 1],
                  'V': [num_dof_flex, num_dof_flex + 3, 2],
                  'W': [num_dof_flex + 3, num_dof_flex + 6, 3],
                  'orient': [num_dof_flex + 6, num_dof_flex + num_dof_rig, 4]}

        # -----------------------------------------------------------------------
        # Better to place in a function avaibale to all elements since it will equally apply
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


        print('End')

    def unpack_ss_vector(self, x_n, y_n, struct_tstep):

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

        for_vel = np.zeros_like(struct_tstep.for_vel)
        for_acc = np.zeros_like(struct_tstep.for_acc)
        quat = np.zeros_like(struct_tstep.quat)

        gravity_forces = np.zeros_like(struct_tstep.gravity_forces)
        total_gravity_forces = np.zeros_like(struct_tstep.total_gravity_forces)
        steady_applied_forces = np.zeros_like(struct_tstep.steady_applied_forces)
        unsteady_applied_forces = np.zeros_like(struct_tstep.unsteady_applied_forces)

        q[:num_dof + rig_dof] = x_n[:num_dof + rig_dof]
        dqdt[:num_dof + rig_dof] = x_n[num_dof + rig_dof:]

        for i_node in vdof[vdof >= 0]:
            pos[i_node, :] = q[i_node + 0: i_node + 3]
            pos_dot[i_node, :] = dqdt[i_node + 0: i_node + 3]

        # TODO: CRV
        # for i_elem in range(struct_tstep.num_elem):
        #     for i_node in range(struct_tstep.num_node_elem):
        #         psi[i_elem, i_node, :] = q[i_node + 3: i_node + 6]
        #         psi_dot[i_elem, i_node, :] = dqdt[i_node + 3: i_node + 6]

        if rig_dof > 0:
            for_vel = dqdt[-rig_dof: -rig_dof + 6]
            quat = dqdt[-4:]
        else:
            quat = struct_tstep.quat.copy()

        if y_n is not None:
            for i_node in vdof[vdof >= 0]:
                steady_applied_forces[i_node+1] = y_n[6*i_node: 6*i_node + 6]

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
