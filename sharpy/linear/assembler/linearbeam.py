"""
Linear State Beam Element Class

"""

from sharpy.linear.utils.ss_interface import BaseElement, linear_system
import sharpy.linear.src.lingebm as lingebm
import numpy as np

@linear_system
class LinearBeam(BaseElement):
    """
    State space member
    """
    sys_id = "LinearBeam"

    def __init__(self):

        self.data = None
        self.sys = None  # The actual object
        self.ss = None  # The state space object

        self.settings = dict()
        self.state_variables = None

    def initialise(self, data):

        self.data = data
        # The initialise method would get everything ready for the beam, instantiate the class etc

        self.settings = self.data.settings['LinearSpace'][self.sys_id]  # Load settings, the settings should be stored in data.linear.settings
        # data.linear.settings should be created in the class above containing the entire set up

        beam = lingebm.FlexDynamic(self.data.linear.tsstruct0, self.data.structure, self.settings)
        self.sys = beam


    def assemble(self):
        # Would assemble the system as per the settings
        # Here we would add further options such as discarding DOFs etc

        # linearise then trim
        if self.settings['gravity'].value:
            self.sys.linearise_gravity_forces()

        if self.settings['remove_dofs']:
            self.trim_nodes(self.settings['remove_dofs'])

        self.sys.assemble()

        # Option to remove certain dofs via dict: i.e. dofs to remove
        # Map dofs to equations
        # Boundary conditions
        # Same with modal, remove certain modes. Need to specify that modes to keep refer to flexible ones only

        if self.sys.SSdisc:
            self.ss = self.sys.SSdisc
        elif self.sys.SScont:
            self.ss = self.sys.SScont


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
                if used_vars_db[item].order < removed_db[rem_item].order:
                    pass
                else:
                    # Update order and position
                    used_vars_db[item].first_pos -= removed_db[rem_item].size
                    used_vars_db[item].end_pos -= removed_db[rem_item].size
                    used_vars_db[item].order -= 1

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



class VectorVariable(object):

    def __init__(self, name, pos_list, var_system):

        self.name = name
        self.var_system = var_system

        self.first_pos = pos_list[0]
        self.end_pos = pos_list[1]
        self.order = pos_list[2]
        self.rows_loc = np.arange(self.first_pos, self.end_pos, dtype=int) # Original location, should not update


    # add methods to reorganise into SHARPy method?

    @property
    def cols_loc(self):
        return np.arange(self.first_pos, self.end_pos, dtype=int)

    @property
    def size(self):
        return self.end_pos - self.first_pos
