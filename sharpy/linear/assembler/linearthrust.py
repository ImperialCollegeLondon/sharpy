import sharpy.linear.utils.ss_interface as ss_interface
import numpy as np
import sharpy.linear.src.lin_aeroelastic as lin_aeroelastic
import sharpy.linear.src.libss as libss
import scipy.linalg as sclalg
import warnings
import sharpy.utils.settings as settings
import sharpy.utils.cout_utils as cout
import sharpy.utils.algebra as algebra

@ss_interface.linear_system
class LinearThrust(object):
    sys_id = 'LinearThrust'

    def __init__(self):
        self.n_thrust_nodes = None
        self.thrust_nodes = None
        self.thrust0 = None

    def initialise(self):

        self.n_thrust_nodes = 2
        self.thrust_nodes = [2, 26]
        self.thrust0 = 5.5085

    def generate(self, tsstruct0, sys):

        structure = sys.structure
        K_thrust = np.zeros_like(sys.Kstr)
        thrust0B = np.zeros((structure.num_dof.value, 3))
        thrust0B[2, :] = np.array([0, self.thrust0, 0])
        thrust0B[26, :] = np.array([0, -self.thrust0, 0])

        for i_node in self.thrust_nodes:
            ee, node_loc = structure.node_master_elem[i_node, :]
            psi = tsstruct0.psi[ee, node_loc, :]
            Cab = algebra.crv2rotation(psi)

            jj = 0  # nodal dof index
            bc_at_node = structure.boundary_conditions[i_node]  # Boundary conditions at the node

            if bc_at_node == 1:  # clamp (only rigid-body)
                dofs_at_node = 0
                jj_tra, jj_rot = [], []

            elif bc_at_node == -1 or bc_at_node == 0:  # (rigid+flex body)
                dofs_at_node = 6
                jj_tra = 6 * structure.vdof[i_node] + np.array([0, 1, 2], dtype=int)  # Translations
                jj_rot = 6 * structure.vdof[i_node] + np.array([3, 4, 5], dtype=int)  # Rotations
            else:
                raise NameError('Invalid boundary condition (%d) at node %d!' \
                                % (bc_at_node, i_node))

            jj += dofs_at_node

            if bc_at_node != 1:
                K_thrust[np.ix_(jj_tra, jj_rot)] -= algebra.der_Ccrv_by_v(psi, thrust0B[i_node])

                K_thrust[-10:-7, jj_rot] -= algebra.der_Ccrv_by_v(psi, thrust0B[i_node])

        return K_thrust
