"""
LagrangeConstraints library

Library used to create the matrices associated to boundary conditions through
the method of Lagrange Multipliers. The source code includes four different sections.

* Basic structures: basic functions and variables needed to organise the library with different Lagrange Constraints to enhance the interaction with this library.

* Auxiliar functions: basic queries that are performed repeatedly.

* Equations: functions that generate the equations associated to the constraint of basic degrees of freedom.

* Lagrange Constraints: different available Lagrange Constraints. They tipically use the basic functions in "Equations" to assembly the required set of equations.

Attributes:
    dict_of_lc (dict): Dictionary including the available Lagrange Contraint identifier
    (``_lc_id``) and the associated ``BaseLagrangeConstraint`` class

Notes:
    To use this library: import sharpy.structure.utils.lagrangeconstraints as lagrangeconstraints

Args:
    lc_list (list): list of all the defined contraints
    MBdict (dict): dictionary with the MultiBody and LagrangeMultipliers information
    MB_beam (list): list of :class:`~sharpy.structure.models.beam.Beam` of each of the bodies that form the system
    MB_tstep (list): list of :class:`~sharpy.utils.datastructures.StructTimeStepInfo` of each of the bodies that form the system
    num_LM_eq (int): number of new equations needed to define the boundary boundary conditions
    sys_size (int): total number of degrees of freedom of the multibody system
    dt (float): time step
    Lambda (np.ndarray): list of Lagrange multipliers values
    Lambda_dot (np.ndarray): list of the first derivative of the Lagrange multipliers values
    dynamic_or_static (str): string defining if the computation is dynamic or static
    LM_C (np.ndarray): Damping matrix associated to the Lagrange Multipliers equations
    LM_K (np.ndarray): Stiffness matrix associated to the Lagrange Multipliers equations
    LM_Q (np.ndarray): Vector of independent terms associated to the Lagrange Multipliers equations
"""
from abc import ABCMeta, abstractmethod
import sharpy.utils.cout_utils as cout
import os
import ctypes as ct
import numpy as np
import sharpy.utils.algebra as ag
from sharpy.utils.settings import set_value_or_default

###############################################################################
# Basic structures
###############################################################################

dict_of_lc = {}
lc = {}  # for internal working


# decorator
def lagrangeconstraint(arg):
    """
    Decorator used to create the dictionary (``dict_of_lc``) that links constraints id (``_lc_id``) to the associated ``BaseLagrangeConstraint`` class
    """
    global dict_of_lc
    try:
        arg._lc_id
    except AttributeError:
        raise AttributeError('Class defined as lagrange constraint has no _lc_id attribute')
    dict_of_lc[arg._lc_id] = arg
    return arg

def print_available_lc():
    """
    Prints the available Lagrange Constraints
    """
    cout.cout_wrap('The available lagrange constraints on this session are:', 2)
    for name, i_lc in dict_of_lc.items():
        cout.cout_wrap('%s ' % i_lc._lc_id, 2)

def lc_from_string(string):
    """
    Returns the ``BaseLagrangeConstraint`` class associated to a constraint id (``_lc_id``)
    """
    return dict_of_lc[string]

def lc_list_from_path(cwd):
    onlyfiles = [f for f in os.listdir(cwd) if os.path.isfile(os.path.join(cwd, f))]

    for i_file in range(len(onlyfiles)):
        if ".py" in onlyfiles[i_file]:
            if onlyfiles[i_file] == "__init__.py":
                onlyfiles[i_file] = ""
                continue
            onlyfiles[i_file] = onlyfiles[i_file].replace('.py', '')
        else:
            onlyfiles[i_file] = ""

    files = [file for file in onlyfiles if not file == ""]
    return files


def initialise_lc(lc_name, print_info=True):
    """
    Initialises the Lagrange Constraints
    """
    if print_info:
        cout.cout_wrap('Generating an instance of %s' % lc_name, 2)
    cls_type = lc_from_string(lc_name)
    lc = cls_type()
    return lc


class BaseLagrangeConstraint(metaclass=ABCMeta):
    __doc__ = """
    BaseLagrangeConstraint

    Base class for LagrangeConstraints showing the methods required. They will
    be inherited by all the Lagrange Constraints

    Attributes:
        _n_eq (int): Number of equations required by a LagrangeConstraint
        _ieq (int): Number of the first equation associated to the Lagrange Constraint in the whole set of Lagrange equations
    """
    _lc_id = 'BaseLagrangeConstraint'

    def __init__(self):
        """
        Initialisation
        """
        self._n_eq = None
        self._ieq = None

    @abstractmethod
    def get_n_eq(self):
        """
        Returns the number of equations required by the Lagrange Constraint
        """
        return self._n_eq

    @abstractmethod
    #  def initialise(self, **kwargs):
    def initialise(self, MBdict_entry, ieq):
        """
        Initialisation
        """
        self._ieq = ieq
        return self._ieq + self._n_eq

    @abstractmethod
    # def staticmat(self, **kwargs):
    def staticmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                  sys_size, dt, Lambda, Lambda_dot):
        """
        Generates the structural matrices (damping, stiffness) and the independent vector
        associated to the LagrangeConstraint in a static simulation
        """
        return np.zeros((6, 6))

    @abstractmethod
    # def dynamicmat(self, **kwargs):
    def dynamicmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                   sys_size, dt, Lambda, Lambda_dot):
        """
        Generates the structural matrices (damping, stiffness) and the independent vector
        associated to the LagrangeConstraint in a dynamic simulation
        """
        return np.zeros((10, 10))

    @abstractmethod
    # def staticpost(self, **kwargs):
    def staticpost(self, lc_list, MB_beam, MB_tstep):
        """
        Postprocess operations needed by the LagrangeConstraint in a static simulation
        """
        return

    @abstractmethod
    # def dynamicpost(self, **kwargs):
    def dynamicpost(self, lc_list, MB_beam, MB_tstep):
        """
        Postprocess operations needed by the LagrangeConstraint in a dynamic simulation
        """
        return


################################################################################
# Auxiliar functions
################################################################################
def define_node_dof(MB_beam, node_body, num_node):
    """
    define_node_dof

    Define the position of the first degree of freedom associated to a certain node

    Args:
        MB_beam(list): list of :class:`~sharpy.structure.models.beam.Beam`
        node_body(int): body to which the node belongs
        num_node(int): number os the node within the body

    Returns:
        node_dof(int): first degree of freedom associated to the node
    """
    node_dof = 0
    for ibody in range(node_body):
        node_dof += MB_beam[ibody].num_dof.value
        if MB_beam[ibody].FoR_movement == 'free':
            node_dof += 10
    node_dof += 6*MB_beam[node_body].vdof[num_node]
    return node_dof

def define_FoR_dof(MB_beam, FoR_body):
    """
    define_FoR_dof

    Define the position of the first degree of freedom associated to a certain frame of reference

    Args:
        MB_beam(list): list of :class:`~sharpy.structure.models.beam.Beam`
        node_body(int): body to which the node belongs
        num_node(int): number os the node within the body

    Returns:
        node_dof(int): first degree of freedom associated to the node
    """
    FoR_dof = 0
    for ibody in range(FoR_body):
        FoR_dof += MB_beam[ibody].num_dof.value
        if MB_beam[ibody].FoR_movement == 'free':
            FoR_dof += 10
    FoR_dof += MB_beam[FoR_body].num_dof.value
    return FoR_dof


################################################################################
# Equations
################################################################################
def equal_pos_node_FoR(MB_tstep, MB_beam, FoR_body, node_body, inode_in_body, node_FoR_dof, node_dof, FoR_dof, sys_size, Lambda, scalingFactor, penaltyFactor, ieq, LM_K, LM_C, LM_Q):
    """
    This function generates the stiffness and damping matrices and the independent vector associated to a constraint that
    imposes equal positions between a node and a frame of reference

    See ``LagrangeConstraints`` for the description of variables

    Args:
        node_FoR_dof (int): position of the first degree of freedom of the FoR to which the "node" belongs
        node_dof (int): position of the first degree of freedom associated to the "node"
        FoR_body (int): body number of the "FoR"
        FoR_dof (int): position of the first degree of freedom associated to the "FoR"

    Note: this equation constitutes a holonomic constraint which is not currently supported. Check ``equal_lin_vel_node_FoR``
    """
    cout.cout_wrap("WARNING: this equation constitutes a holonomic constraint which is not currently supported. Check ``equal_lin_vel_node_FoR``", 3)

    num_LM_eq_specific = 3
    Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')
    B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')

    # Simplify notation
    node_cga = MB_tstep[node_body].cga()
    node_pos = MB_tstep[node_body].pos[inode_in_body, :]
    node_FoR_pos = MB_tstep[node_body].for_pos[0:3]
    FoR_pos = MB_tstep[FoR_body].for_pos[0:3]

    # if MB_beam[node_body].FoR_movement == 'free':
    B[:, node_FoR_dof:node_FoR_dof+3] = np.eye(3)
    B[:, node_dof:node_dof+3] = node_cga
    B[:, FoR_dof:FoR_dof+3] = -np.eye(3)

    LM_K[sys_size + ieq : sys_size + ieq + num_LM_eq_specific, :sys_size] += scalingFactor*B
    LM_K[:sys_size, sys_size + ieq : sys_size + ieq + num_LM_eq_specific] += scalingFactor*np.transpose(B)

    LM_Q[:sys_size] += scalingFactor*np.dot(np.transpose(B), Lambda[ieq:ieq+num_LM_eq_specific])
    LM_Q[sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += scalingFactor*(node_FoR_pos +
                                                                         np.dot(node_cga, node_pos) -
                                                                         FoR_pos)

    LM_C[node_dof:node_dof+3, node_FoR_dof+6:node_FoR_dof+10] += scalingFactor*ag.der_CquatT_by_v(MB_tstep[node_body].quat, Lambda[ieq : ieq + num_LM_eq_specific])

    if penaltyFactor:
        q = np.zeros((sys_size, ))
        q[node_FoR_dof:node_FoR_dof+3] = node_FoR_pos
        q[node_dof:node_dof+3] = node_pos
        q[FoR_dof:FoR_dof+3] = FoR_pos

        LM_Q[:sys_size] += penaltyFactor*np.dot(B.T, np.dot(B, q))

        LM_K[node_FoR_dof:node_FoR_dof+3, node_FoR_dof:node_FoR_dof+3] += penaltyFactor*np.eye(3)
        LM_K[node_FoR_dof:node_FoR_dof+3, node_dof:node_dof+3] += penaltyFactor*node_cga
        LM_K[node_FoR_dof:node_FoR_dof+3, FoR_dof:FoR_dof+3] += -penaltyFactor*np.eye(3)
        LM_C[node_FoR_dof:node_FoR_dof+3, node_FoR_dof+6:node_FoR_dof+10] += penaltyFactor*ag.der_Cquat_by_v(MB_tstep[node_body].quat, node_pos)

        LM_K[node_dof:node_dof+3, node_FoR_dof:node_FoR_dof+3] += penaltyFactor*node_cga.T
        LM_K[node_dof:node_dof+3, node_dof:node_dof+3] += penaltyFactor*np.eye(3)
        LM_K[node_dof:node_dof+3, FoR_dof:FoR_dof+3] += -penaltyFactor*node_cga.T
        LM_C[node_dof:node_dof+3, node_FoR_dof+6:node_FoR_dof+10] += penaltyFactor*(ag.der_CquatT_by_v(MB_tstep[node_body].quat, node_FoR_pos - FoR_pos))

        LM_K[FoR_dof:FoR_dof+3, node_FoR_dof:node_FoR_dof+3] += -penaltyFactor*np.eye(3)
        LM_K[FoR_dof:FoR_dof+3, node_dof:node_dof+3] += -penaltyFactor*node_cga.T
        LM_K[FoR_dof:FoR_dof+3, FoR_dof:FoR_dof+3] += penaltyFactor*np.eye(3)
        LM_C[FoR_dof:FoR_dof+3, node_FoR_dof+6:node_FoR_dof+10] += -penaltyFactor*ag.der_Cquat_by_v(MB_tstep[node_body].quat, node_pos)

    ieq += 3
    return ieq


def equal_lin_vel_node_FoR(MB_tstep, MB_beam, FoR_body, node_body, node_number, node_FoR_dof, node_dof, FoR_dof, sys_size, Lambda_dot, scalingFactor, penaltyFactor, ieq, LM_K, LM_C, LM_Q, rel_posB = np.zeros((3))):
    """
    This function generates the stiffness and damping matrices and the independent vector associated to a constraint that
    imposes equal linear velocities between a node and a frame of reference

    See ``LagrangeConstraints`` for the description of variables

    Args:
        node_number (int): number of the "node" within its own body
        node_body (int): body number of the "node"
        node_FoR_dof (int): position of the first degree of freedom of the FoR to which the "node" belongs
        node_dof (int): position of the first degree of freedom associated to the "node"
        FoR_body (int): body number of the "FoR"
        FoR_dof (int): position of the first degree of freedom associated to the "FoR"
        rel_posB (float np.array): relative position between the node and the FoR (in the node B FoR)
    """

    num_LM_eq_specific = 3
    Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')
    B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')

    # Simplify notation
    node_cga = MB_tstep[node_body].cga()
    node_FoR_va = MB_tstep[node_body].for_vel[0:3]
    node_FoR_wa = MB_tstep[node_body].for_vel[3:6]

    ielem, inode_in_elem = MB_beam[node_body].node_master_elem[node_number]
    psi = MB_tstep[node_body].psi[ielem, inode_in_elem, :]
    node_cab = ag.crv2rotation(psi)
    node_Ra = MB_tstep[node_body].pos[node_number,:] + np.dot(node_cab, rel_posB)

    node_dot_Ra = MB_tstep[node_body].pos_dot[node_number,:]
    FoR_cga = MB_tstep[FoR_body].cga()
    FoR_va = MB_tstep[FoR_body].for_vel[0:3]

    Bnh[:, FoR_dof:FoR_dof+3] = FoR_cga
    Bnh[:, node_dof:node_dof+3] = -1.0*node_cga
    if MB_beam[node_body].FoR_movement == 'free':
        Bnh[:, node_FoR_dof:node_FoR_dof+3] = -1.0*node_cga
        Bnh[:, node_FoR_dof+3:node_FoR_dof+6] = np.dot(node_cga,ag.skew(node_Ra))

    LM_C[sys_size+ieq:sys_size+ieq+num_LM_eq_specific,:sys_size] += scalingFactor*Bnh
    LM_C[:sys_size,sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += scalingFactor*np.transpose(Bnh)

    LM_Q[:sys_size] += scalingFactor*np.dot(np.transpose(Bnh), Lambda_dot[ieq:ieq+num_LM_eq_specific])
    LM_Q[sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += scalingFactor*(np.dot(FoR_cga, FoR_va) +
                                                          -1.0*np.dot(node_cga,
                                                                      node_dot_Ra +
                                                                      node_FoR_va +
                                                                      -1.0*np.dot(ag.skew(node_Ra), node_FoR_wa)))

    LM_C[FoR_dof:FoR_dof+3, FoR_dof+6:FoR_dof+10] += scalingFactor*ag.der_CquatT_by_v(MB_tstep[FoR_body].quat, Lambda_dot[ieq:ieq+num_LM_eq_specific])

    if MB_beam[node_body].FoR_movement == 'free':
        LM_C[node_dof:node_dof+3,node_FoR_dof+6:node_FoR_dof+10] -= scalingFactor*ag.der_CquatT_by_v(MB_tstep[node_body].quat, Lambda_dot[ieq:ieq+num_LM_eq_specific])

        LM_C[node_FoR_dof:node_FoR_dof+3,node_FoR_dof+6:node_FoR_dof+10] -= scalingFactor*ag.der_CquatT_by_v(MB_tstep[node_body].quat,Lambda_dot[ieq:ieq+num_LM_eq_specific])

        LM_C[node_FoR_dof+3:node_FoR_dof+6,node_FoR_dof+6:node_FoR_dof+10] += scalingFactor*np.dot(ag.skew(node_Ra).T,
                                                                                     ag.der_CquatT_by_v(MB_tstep[node_body].quat,
                                                                                                             Lambda_dot[ieq:ieq+num_LM_eq_specific]))

        LM_K[node_FoR_dof+3:node_FoR_dof+6,node_dof:node_dof+3] += scalingFactor*ag.skew(np.dot(node_cga.T,Lambda_dot[ieq:ieq+num_LM_eq_specific]))

    if penaltyFactor:
        q = np.zeros((sys_size))
        q[FoR_dof:FoR_dof+3] = FoR_va
        q[node_dof:node_dof+3] = node_dot_Ra
        if MB_beam[node_body].FoR_movement == 'free':
            q[node_FoR_dof:node_FoR_dof+3] = node_FoR_va
            q[node_FoR_dof+3:node_FoR_dof+6] = node_FoR_wa

        LM_Q[:sys_size] += penaltyFactor*np.dot(np.dot(Bnh.T, Bnh), q)

        LM_C[:sys_size, :sys_size] += penaltyFactor*np.dot(Bnh.T, Bnh)

        # Derivatives wrt the FoR quaterion
        LM_C[FoR_dof:FoR_dof+3, FoR_dof+6:FoR_dof+10] -= penaltyFactor*ag.der_CquatT_by_v(MB_tstep[FoR_body].quat,
                                                                                                 np.dot(node_cga, node_dot_Ra + node_FoR_va +
                                                                                                         np.dot(ag.skew(node_Ra), node_FoR_wa)))

        LM_C[node_dof:node_dof+3, FoR_dof+6:FoR_dof+10] -= penaltyFactor*np.dot(node_cga.T, ag.der_CquatT_by_v(MB_tstep[FoR_body].quat,
                                                                                                               FoR_va))

        if MB_beam[node_body].FoR_movement == 'free':
            LM_C[node_FoR_dof:node_FoR_dof+3, FoR_dof+6:FoR_dof+10] -= penaltyFactor*np.dot(node_cga.T, ag.der_CquatT_by_v(MB_tstep[FoR_body].quat,
                                                                                                               FoR_va))

            mat = ag.multiply_matrices(ag.skew(node_Ra).T, node_cga.T)
            LM_C[node_FoR_dof+3:node_FoR_dof+6, FoR_dof+6:FoR_dof+10] += penaltyFactor*np.dot(mat, ag.der_CquatT_by_v(MB_tstep[FoR_body].quat,
                                                                                                               FoR_va))

        # Derivatives wrt the node quaternion
        if MB_beam[node_body].FoR_movement == 'free':
            vec = -node_dot_Ra - node_FoR_va + np.dot(ag.skew(node_Ra), node_FoR_wa)
            LM_C[FoR_dof:FoR_dof+3, node_FoR_dof+6:node_FoR_dof+10] += penaltyFactor*np.dot(FoR_cga.T, ag.der_Cquat_by_v(MB_tstep[node_body].quat, vec))

            derivative = -ag.der_CquatT_by_v(MB_tstep[node_body].quat, np.dot(FoR_cga, FoR_va))
            LM_C[node_dof:node_dof+3, node_FoR_dof+6:node_FoR_dof+10] += penaltyFactor*derivative
            LM_C[node_FoR_dof:node_FoR_dof+3, node_FoR_dof+6:node_FoR_dof+10] += penaltyFactor*derivative
            LM_C[node_FoR_dof+3:node_FoR_dof+6, node_FoR_dof+6:node_FoR_dof+10] -= penaltyFactor*np.dot(ag.skew(node_Ra), derivative)

        # Derivatives wrt the node Ra
        LM_K[FoR_dof:FoR_dof+3, node_dof:node_dof+3] -= penaltyFactor*ag.multiply_matrices(FoR_cga.T, node_cga, ag.skew(node_FoR_wa))
        LM_K[node_dof:node_dof+3, node_dof:node_dof+3] += penaltyFactor*ag.skew(node_FoR_wa)
        if MB_beam[node_body].FoR_movement == 'free':
            LM_K[node_FoR_dof:node_FoR_dof+3, node_dof:node_dof+3] += penaltyFactor*ag.skew(node_FoR_wa)
            vec = ag.multiply_matrices(node_cga.T, FoR_cga, FoR_va) - node_dot_Ra - node_FoR_va
            LM_K[node_FoR_dof+3:node_FoR_dof+6, node_dof:node_dof+3] += penaltyFactor*ag.skew(vec)
            LM_K[node_FoR_dof+3:node_FoR_dof+6, node_dof:node_dof+3] -= penaltyFactor*ag.der_skewp_skewp_v(node_Ra, node_FoR_wa)

    ieq += 3
    return ieq


def rel_rot_vel_node_FoR(MB_tstep, MB_beam, FoR_body, node_body, node_number, node_FoR_dof, node_dof, FoR_dof, sys_size, Lambda_dot, scalingFactor, penaltyFactor, ieq, LM_K, LM_C, LM_Q, rel_vel=np.zeros((3))):
    """
    This function generates the stiffness and damping matrices and the independent vector associated to a constraint that
    imposes equal rotation velocities between a node and a frame of reference

    See ``LagrangeConstraints`` for the description of variables

    Args:
        node_number (int): number of the "node" within its own body
        node_body (int): body number of the "node"
        node_FoR_dof (int): position of the first degree of freedom of the FoR to which the "node" belongs
        node_dof (int): position of the first degree of freedom associated to the "node"
        FoR_body (int): body number of the "FoR"
        FoR_dof (int): position of the first degree of freedom associated to the "FoR"
        rel_vel (np.array): relative velocity FoR-node
    """

    num_LM_eq_specific = 3
    Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')

    # Simplify notation
    ielem, inode_in_elem = MB_beam[node_body].node_master_elem[node_number]
    node_cga = MB_tstep[node_body].cga()
    node_FoR_wa = MB_tstep[node_body].for_vel[3:6]
    psi = MB_tstep[node_body].psi[ielem,inode_in_elem,:]
    cab = ag.crv2rotation(psi)
    tan = ag.crv2tan(psi)

    FoR_cga = MB_tstep[FoR_body].cga()
    FoR_wa = MB_tstep[FoR_body].for_vel[3:6]

    Bnh[:, node_dof+3:node_dof+6] += tan.copy()
    Bnh[:, FoR_dof+3:FoR_dof+6] -= ag.multiply_matrices(cab.T, node_cga.T, FoR_cga)
    if MB_beam[node_body].FoR_movement == 'free':
        Bnh[:, node_FoR_dof+3:node_FoR_dof+6] += cab.T

    LM_C[sys_size+ieq:sys_size+ieq+num_LM_eq_specific,:sys_size] += scalingFactor*Bnh
    LM_C[:sys_size,sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += scalingFactor*np.transpose(Bnh)

    LM_Q[:sys_size] += scalingFactor*np.dot(np.transpose(Bnh), Lambda_dot[ieq:ieq+num_LM_eq_specific])
    LM_Q[sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += scalingFactor*(np.dot(tan, MB_tstep[node_body].psi_dot[ielem, inode_in_elem, :]) +
                                                                         np.dot(cab.T, node_FoR_wa) -
                                                                         ag.multiply_matrices(cab.T, node_cga.T, FoR_cga, FoR_wa) + rel_vel)

    LM_K[node_dof+3:node_dof+6, node_dof+3:node_dof+6] += scalingFactor*ag.der_TanT_by_xv(psi, Lambda_dot[ieq:ieq+num_LM_eq_specific])
    if MB_beam[node_body].FoR_movement == 'free':
        LM_K[node_FoR_dof+3:node_FoR_dof+6, node_dof+3:node_dof+6] += scalingFactor*ag.der_Ccrv_by_v(psi, Lambda_dot[ieq:ieq+num_LM_eq_specific])

    LM_K[FoR_dof+3:FoR_dof+6, node_dof+3:node_dof+6] -= scalingFactor*ag.der_Ccrv_by_v(psi,
                                                                                       ag.multiply_matrices(node_cga, FoR_cga.T, Lambda_dot[ieq:ieq+num_LM_eq_specific]))

    LM_C[FoR_dof+3:FoR_dof+6, node_FoR_dof+6:node_FoR_dof+10] -= scalingFactor*np.dot(cab, ag.der_Cquat_by_v(MB_tstep[node_body].quat,
                                                                                                 np.dot(FoR_cga.T, Lambda_dot[ieq:ieq+num_LM_eq_specific])))
    LM_C[FoR_dof+3:FoR_dof+6, FoR_dof+6:FoR_dof+10] -= scalingFactor*ag.multiply_matrices(cab, node_cga, ag.der_CquatT_by_v(MB_tstep[FoR_body].quat,
                                                                                                               Lambda_dot[ieq:ieq+num_LM_eq_specific]))

    ieq += 3
    return ieq

def def_rot_axis_FoR_wrt_node_general(MB_tstep, MB_beam, FoR_body, node_body, node_number, node_FoR_dof, node_dof, FoR_dof, sys_size, Lambda_dot, rot_axisB, scalingFactor, penaltyFactor, ieq, LM_K, LM_C, LM_Q, indep):
    """
    This function generates the stiffness and damping matrices and the independent vector associated to a joint that
    forces the rotation axis of a FoR to be parallel to a certain direction. This direction is defined in the
    B FoR of a node and, thus, might change along the simulation.

    See ``LagrangeConstraints`` for the description of variables

    Args:
        rot_axisB (np.ndarray): Rotation axis with respect to the node B FoR
        indep (np.ndarray): Number of the equations that are used as independent
        node_number (int): number of the "node" within its own body
        node_body (int): body number of the "node"
        node_FoR_dof (int): position of the first degree of freedom of the FoR to which the "node" belongs
        node_dof (int): position of the first degree of freedom associated to the "node"
        FoR_body (int): body number of the "FoR"
        FoR_dof (int): position of the first degree of freedom associated to the "FoR"

    Notes: this function is missing the contribution of the rotation velocity of the reference node. See ``def_rot_axis_FoR_wrt_node_general``
    """
    cout.cout_wrap("WARNING: this function is missing the contribution of the rotation velocity of the reference node. See ``def_rot_axis_FoR_wrt_node_general``", 3)

    ielem, inode_in_elem = MB_beam[node_body].node_master_elem[node_number]

    # Simplify notation
    cab = ag.crv2rotation(MB_tstep[node_body].psi[ielem,inode_in_elem,:])
    node_cga = MB_tstep[node_body].cga()
    FoR_cga = MB_tstep[FoR_body].cga()
    FoR_wa = MB_tstep[FoR_body].for_vel[3:6]

    if not indep:
        aux_Bnh = ag.multiply_matrices(ag.skew(rot_axisB),
                                  cab.T,
                                  node_cga.T,
                                  FoR_cga)

        # indep = None
        n0 = np.linalg.norm(aux_Bnh[0,:])
        n1 = np.linalg.norm(aux_Bnh[1,:])
        n2 = np.linalg.norm(aux_Bnh[2,:])
        if ((n0 < n1) and (n0 < n2)):
            indep[:] = [1, 2]
        elif ((n1 < n0) and (n1 < n2)):
            indep[:] = [0, 2]
        elif ((n2 < n0) and (n2 < n1)):
            indep[:] = [0, 1]

    new_Lambda_dot = np.zeros(3)
    new_Lambda_dot[indep[0]] = Lambda_dot[ieq]
    new_Lambda_dot[indep[1]] = Lambda_dot[ieq+1]

    num_LM_eq_specific = 2
    Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')
    B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')

    Bnh[:, FoR_dof+3:FoR_dof+6] = ag.multiply_matrices(ag.skew(rot_axisB),
                                                       cab.T,
                                                       node_cga.T,
                                                       FoR_cga)[indep,:]

    # Constrain angular velocities
    LM_Q[:sys_size] += scalingFactor*np.dot(np.transpose(Bnh), Lambda_dot[ieq:ieq+num_LM_eq_specific])
    LM_Q[sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += scalingFactor*ag.multiply_matrices(ag.skew(rot_axisB),
                                                  cab.T,
                                                  node_cga.T,
                                                  FoR_cga,
                                                  FoR_wa)[indep]

    LM_C[sys_size+ieq:sys_size+ieq+num_LM_eq_specific,:sys_size] += scalingFactor*Bnh
    LM_C[:sys_size,sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += scalingFactor*np.transpose(Bnh)

    if MB_beam[node_body].FoR_movement == 'free':
        LM_C[FoR_dof+3:FoR_dof+6,node_FoR_dof+6:node_FoR_dof+10] += scalingFactor*np.dot(FoR_cga.T,
                                                                           ag.der_Cquat_by_v(MB_tstep[node_body].quat,
                                                                                                  ag.multiply_matrices(cab,
                                                                                                                    ag.skew(rot_axisB).T,
                                                                                                                    new_Lambda_dot)))

    LM_C[FoR_dof+3:FoR_dof+6,FoR_dof+6:FoR_dof+10] += scalingFactor*ag.der_CquatT_by_v(MB_tstep[FoR_body].quat,
                                                                              ag.multiply_matrices(node_cga,
                                                                                                cab,
                                                                                                ag.skew(rot_axisB).T,
                                                                                                new_Lambda_dot))

    LM_K[FoR_dof+3:FoR_dof+6,node_dof+3:node_dof+6] += scalingFactor*ag.multiply_matrices(FoR_cga.T,
                                                                         node_cga,
                                                                         cab,
                                                                         ag.skew(rot_axisB).T,
                                                                         new_Lambda_dot)

    if penaltyFactor:
        q = np.zeros((sys_size,))
        q[FoR_dof+3:FoR_dof+6] = MB_tstep[FoR_body].for_vel[3:6]

        LM_Q[:sys_size] += penaltyFactor*np.dot(Bnh.T, np.dot(Bnh, q))

        LM_C[:sys_size, :sys_size] += penaltyFactor*np.dot(Bnh.T, Bnh)

        sq_rot_axisB = np.dot(ag.skew(rot_axisB).T, ag.skew(rot_axisB))

        # Derivatives with the quaternion of the FoR
        vec = ag.multiply_matrices(node_cga,
                                   cab,
                                   sq_rot_axisB,
                                   cab.T,
                                   node_cga.T,
                                   FoR_cga,
                                   FoR_wa)
        LM_C[FoR_dof+3:FoR_dof+6, FoR_dof+6:FoR_dof+10] += penaltyFactor*ag.der_CquatT_by_v(MB_tstep[FoR_body].quat, vec)

        mat = ag.multiply_matrices(FoR_cga.T,
                                    node_cga,
                                    cab,
                                    sq_rot_axisB,
                                    cab.T,
                                    node_cga.T)
        LM_C[FoR_dof+3:FoR_dof+6, FoR_dof+6:FoR_dof+10] += penaltyFactor*np.dot(mat, ag.der_Cquat_by_v(MB_tstep[FoR_body].quat, FoR_wa))

        if MB_beam[node_body].FoR_movement == 'free':
            # Derivatives with the quaternion of the FoR of the node
            vec = ag.multiply_matrices(cab,
                                        sq_rot_axisB,
                                        cab.T,
                                        node_cga.T,
                                        FoR_cga,
                                        FoR_wa)
            LM_C[FoR_dof+3:FoR_dof+6, node_FoR_dof+6:node_FoR_dof+10] += penaltyFactor*np.dot(FoR_cga.T,
                                                                                          ag.der_Cquat_by_v(MB_tstep[node_body].quat, vec))
            mat = ag.multiply_matrices(FoR_cga.T,
                                        node_cga,
                                        cab,
                                        sq_rot_axisB,
                                        cab.T)
            vec = np.dot(FoR_cga, FoR_wa)
            LM_C[FoR_dof+3:FoR_dof+6, node_FoR_dof+6:node_FoR_dof+10] += penaltyFactor*np.dot(mat, ag.der_CquatT_by_v(MB_tstep[node_body].quat, vec))

        # Derivatives with the CRV
        mat = np.dot(FoR_cga.T, node_cga)
        vec = ag.multiply_matrices(sq_rot_axisB,
                                    cab.T,
                                    node_cga.T,
                                    FoR_cga,
                                    FoR_wa)
        LM_K[FoR_dof+3:FoR_dof+6, node_dof+3:node_dof+6] += penaltyFactor*np.dot(mat, ag.der_Ccrv_by_v(MB_tstep[node_body].psi[ielem,inode_in_elem,:], vec))

        mat = ag.multiply_matrices(FoR_cga.T,
                                    node_cga,
                                    cab,
                                    sq_rot_axisB)
        vec = ag.multiply_matrices(node_cga.T,
                                    FoR_cga,
                                    FoR_wa)
        LM_K[FoR_dof+3:FoR_dof+6, node_dof+3:node_dof+6] += penaltyFactor*np.dot(mat, ag.der_CcrvT_by_v(MB_tstep[node_body].psi[ielem,inode_in_elem,:], vec))

    ieq += 2
    return ieq


def def_rot_axis_FoR_wrt_node_xyz(MB_tstep, MB_beam, FoR_body, node_body, node_number, node_FoR_dof, node_dof, FoR_dof, sys_size, Lambda_dot, rot_axisB, scalingFactor, penaltyFactor, ieq, LM_K, LM_C, LM_Q, zero_comp):
    """
    This function generates the stiffness and damping matrices and the independent vector associated to a joint that
    forces the rotation axis of a FoR to be parallel to a certain direction. This direction is defined in the
    B FoR of a node and parallel to x, y or z

    See ``LagrangeConstraints`` for the description of variables

    Args:
        rot_axisB (np.ndarray): Rotation axis with respect to the node B FoR
        indep (np.ndarray): Number of the equations that are used as independent
        node_number (int): number of the "node" within its own body
        node_body (int): body number of the "node"
        node_FoR_dof (int): position of the first degree of freedom of the FoR to which the "node" belongs
        node_dof (int): position of the first degree of freedom associated to the "node"
        FoR_body (int): body number of the "FoR"
        FoR_dof (int): position of the first degree of freedom associated to the "FoR"
    """

    ielem, inode_in_elem = MB_beam[node_body].node_master_elem[node_number]

    num_LM_eq_specific = 2
    Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')
    B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')

    # Simplify notation
    cab = ag.crv2rotation(MB_tstep[node_body].psi[ielem,inode_in_elem,:])
    node_cga = MB_tstep[node_body].cga()
    FoR_cga = MB_tstep[FoR_body].cga()
    FoR_wa = MB_tstep[FoR_body].for_vel[3:6]
    psi = MB_tstep[node_body].psi[ielem,inode_in_elem,:]
    psi_dot = MB_tstep[node_body].psi_dot[ielem,inode_in_elem,:]

    # Components to be zero
    Z = np.zeros((2,3))
    Z[:, zero_comp] = np.eye(2)

    Bnh[:, FoR_dof+3:FoR_dof+6] += ag.multiply_matrices(Z, cab.T, node_cga.T, FoR_cga)
    Bnh[:, node_dof+3:node_dof+6] -= ag.multiply_matrices(Z, ag.crv2tan(psi))
    Bnh[:, node_FoR_dof+3:node_FoR_dof+6] -= ag.multiply_matrices(Z, cab.T)

    # Constrain angular velocities
    LM_Q[:sys_size] += scalingFactor*np.dot(np.transpose(Bnh), Lambda_dot[ieq:ieq+num_LM_eq_specific])
    LM_Q[sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += scalingFactor*ag.multiply_matrices(Z, cab.T, node_cga.T, FoR_cga, FoR_wa)
    LM_Q[sys_size+ieq:sys_size+ieq+num_LM_eq_specific] -= scalingFactor*ag.multiply_matrices(Z, ag.crv2tan(psi), psi_dot)
    LM_Q[sys_size+ieq:sys_size+ieq+num_LM_eq_specific] -= scalingFactor*ag.multiply_matrices(Z, cab.T, MB_tstep[node_body].for_vel[3:6])

    LM_C[sys_size+ieq:sys_size+ieq+num_LM_eq_specific,:sys_size] += scalingFactor*Bnh
    LM_C[:sys_size,sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += scalingFactor*np.transpose(Bnh)

    vec = ag.multiply_matrices(node_cga, cab, Z.T, Lambda_dot[ieq:ieq+num_LM_eq_specific])
    LM_C[FoR_dof+3:FoR_dof+6, FoR_dof+6:FoR_dof+10] += scalingFactor*ag.der_CquatT_by_v(MB_tstep[FoR_body].quat, vec)

    if MB_beam[node_body].FoR_movement == 'free':
        vec = ag.multiply_matrices(cab, Z.T, Lambda_dot[ieq:ieq+num_LM_eq_specific])
        LM_C[FoR_dof+3:FoR_dof+6, node_FoR_dof+6:node_FoR_dof+10] += scalingFactor*ag.multiply_matrices(FoR_cga.T, ag.der_Cquat_by_v(MB_tstep[node_body].quat, vec))

    LM_K[FoR_dof+3:FoR_dof+6, node_dof+3:node_dof+6] += scalingFactor*ag.multiply_matrices(FoR_cga.T, node_cga, ag.der_Ccrv_by_v(MB_tstep[node_body].psi[ielem,inode_in_elem,:],
                                                                                                                                  np.dot(Z.T, Lambda_dot[ieq:ieq+num_LM_eq_specific])))

    LM_K[node_dof+3:node_dof+6, node_dof+3:node_dof+6] -= scalingFactor*ag.der_TanT_by_xv(psi, ag.multiply_matrices(Z.T, Lambda_dot[ieq:ieq+num_LM_eq_specific]))
    LM_K[node_FoR_dof+3:node_FoR_dof+6, node_dof+3:node_dof+6] -= scalingFactor*ag.der_Ccrv_by_v(psi, ag.multiply_matrices(Z.T, Lambda_dot[ieq:ieq+num_LM_eq_specific]))

    if penaltyFactor:
        q = np.zeros((sys_size,))
        q[FoR_dof+3:FoR_dof+6] = FoR_wa

        LM_Q[:sys_size] += penaltyFactor*np.dot(Bnh.T, np.dot(Bnh, q))

        LM_C[:sys_size, :sys_size] += penaltyFactor*np.dot(Bnh.T, Bnh)

        ZTZ = np.dot(Z.T, Z)

        # Derivatives with the quaternion of the FoR
        vec = ag.multiply_matrices(node_cga,
                                   cab,
                                   ZTZ,
                                   cab.T,
                                   node_cga.T,
                                   FoR_cga,
                                   FoR_wa)
        LM_C[FoR_dof+3:FoR_dof+6, FoR_dof+6:FoR_dof+10] += penaltyFactor*ag.der_CquatT_by_v(MB_tstep[FoR_body].quat, vec)

        mat = ag.multiply_matrices(FoR_cga.T,
                                   node_cga,
                                   cab,
                                   ZTZ,
                                   cab.T,
                                   node_cga.T)
        LM_C[FoR_dof+3:FoR_dof+6, FoR_dof+6:FoR_dof+10] += penaltyFactor*np.dot(mat, ag.der_Cquat_by_v(MB_tstep[FoR_body].quat, FoR_wa))

        if MB_beam[node_body].FoR_movement == 'free':
            # Derivatives with the quaternion of the FoR of the node
            vec = ag.multiply_matrices(cab,
                                       ZTZ,
                                       cab.T,
                                       node_cga.T,
                                       FoR_cga,
                                       FoR_wa)
            LM_C[FoR_dof+3:FoR_dof+6, node_FoR_dof+6:node_FoR_dof+10] += penaltyFactor*np.dot(FoR_cga.T,
                                                                                          ag.der_Cquat_by_v(MB_tstep[node_body].quat, vec))

            mat = ag.multiply_matrices(FoR_cga.T,
                                       node_cga,
                                       cab,
                                       ZTZ,
                                       cab.T)
            vec = np.dot(FoR_cga, FoR_wa)
            LM_C[FoR_dof+3:FoR_dof+6, node_FoR_dof+6:node_FoR_dof+10] += penaltyFactor*np.dot(mat, ag.der_CquatT_by_v(MB_tstep[node_body].quat, vec))

        # Derivatives with the CRV
        mat = np.dot(FoR_cga.T, node_cga)
        vec = ag.multiply_matrices(ZTZ,
                                   cab.T,
                                   node_cga.T,
                                   FoR_cga,
                                   FoR_wa)
        LM_K[FoR_dof+3:FoR_dof+6, node_dof+3:node_dof+6] += penaltyFactor*np.dot(mat, ag.der_Ccrv_by_v(MB_tstep[node_body].psi[ielem,inode_in_elem,:], vec))

        mat = ag.multiply_matrices(FoR_cga.T,
                                   node_cga,
                                   cab,
                                   ZTZ)
        vec = ag.multiply_matrices(node_cga.T,
                                   FoR_cga,
                                   FoR_wa)
        LM_K[FoR_dof+3:FoR_dof+6, node_dof+3:node_dof+6] += penaltyFactor*np.dot(mat, ag.der_CcrvT_by_v(MB_tstep[node_body].psi[ielem,inode_in_elem,:], vec))

    ieq += 2
    return ieq


def def_rot_vel_mod_FoR_wrt_node(MB_tstep, MB_beam, FoR_body, node_body, node_number, node_FoR_dof, node_dof, FoR_dof, sys_size, Lambda_dot, nonzero_comp, rot_vel, scalingFactor, penaltyFactor, ieq, LM_K, LM_C, LM_Q):
    """
    This function generates the stiffness and damping matrices and the independent vector associated to a joint that
    forces the rotation velocity of a FoR with respect to a node

    See ``LagrangeConstraints`` for the description of variables

    Args:
        nonzero_comp (int): Component of the rotation axis with respect to the node B FoR which is non-zero
        rot_vel (float): Rotation velocity
        node_number (int): number of the "node" within its own body
        node_body (int): body number of the "node"
        node_FoR_dof (int): position of the first degree of freedom of the FoR to which the "node" belongs
        node_dof (int): position of the first degree of freedom associated to the "node"
        FoR_body (int): body number of the "FoR"
        FoR_dof (int): position of the first degree of freedom associated to the "FoR"
    """

    ielem, inode_in_elem = MB_beam[node_body].node_master_elem[node_number]
    num_LM_eq_specific = 1
    Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')
    B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')

    # Simplify notation
    cab = ag.crv2rotation(MB_tstep[node_body].psi[ielem,inode_in_elem,:])
    node_cga = MB_tstep[node_body].cga()
    FoR_cga = MB_tstep[FoR_body].cga()
    FoR_wa = MB_tstep[FoR_body].for_vel[3:6]

    # Components to be zero
    Znon = np.zeros((1,3))
    Znon[:, nonzero_comp] = 1

    Bnh[:, FoR_dof+3:FoR_dof+6] += ag.multiply_matrices(Znon, cab.T, node_cga.T, FoR_cga)

    # Constrain angular velocities
    LM_Q[:sys_size] += scalingFactor*np.dot(np.transpose(Bnh), Lambda_dot[ieq:ieq+num_LM_eq_specific])
    LM_Q[sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += scalingFactor*ag.multiply_matrices(Znon, cab.T, node_cga.T, FoR_cga, FoR_wa)
    LM_Q[sys_size+ieq:sys_size+ieq+num_LM_eq_specific] -= scalingFactor*rot_vel

    LM_C[sys_size+ieq:sys_size+ieq+num_LM_eq_specific,:sys_size] += scalingFactor*Bnh
    LM_C[:sys_size,sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += scalingFactor*np.transpose(Bnh)

    vec = ag.multiply_matrices(node_cga, cab, Znon.T, Lambda_dot[ieq:ieq+num_LM_eq_specific])
    LM_C[FoR_dof+3:FoR_dof+6, FoR_dof+6:FoR_dof+10] += scalingFactor*ag.der_CquatT_by_v(MB_tstep[FoR_body].quat, vec)

    if MB_beam[node_body].FoR_movement == 'free':
        vec = ag.multiply_matrices(cab, Znon.T, Lambda_dot[ieq:ieq+num_LM_eq_specific])
        LM_C[FoR_dof+3:FoR_dof+6, node_FoR_dof+6:node_FoR_dof+10] += scalingFactor*ag.multiply_matrices(FoR_cga.T, ag.der_Cquat_by_v(MB_tstep[node_body].quat, vec))

    LM_K[FoR_dof+3:FoR_dof+6, node_dof+3:node_dof+6] += scalingFactor*ag.multiply_matrices(FoR_cga.T, node_cga, ag.der_Ccrv_by_v(MB_tstep[node_body].psi[ielem,inode_in_elem,:],
                                                                                                                                  np.dot(Znon.T, Lambda_dot[ieq:ieq+num_LM_eq_specific])))

    ieq += 1
    return ieq

def def_rot_vect_FoR_wrt_node(MB_tstep, MB_beam, FoR_body, node_body, node_number, node_FoR_dof, node_dof, FoR_dof, sys_size, Lambda_dot, rot_vect, scalingFactor, penaltyFactor, ieq, LM_K, LM_C, LM_Q):
    """
        This function fixes the rotation velocity VECTOR of a FOR equal to a velocity vector defined in the B FoR of a node
        This function is a new implementation that combines and simplifies the use of 'def_rot_vel_mod_FoR_wrt_node' and 'def_rot_axis_FoR_wrt_node' together
    """

    num_LM_eq_specific = 3
    Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')

    # Simplify notation
    ielem, inode_in_elem = MB_beam[node_body].node_master_elem[node_number]
    node_cga = MB_tstep[node_body].cga()
    cab = ag.crv2rotation(MB_tstep[node_body].psi[ielem,inode_in_elem,:])
    FoR_cga = MB_tstep[FoR_body].cga()
    FoR_wa = MB_tstep[FoR_body].for_vel[3:6]

    Bnh[:, FoR_dof+3:FoR_dof+6] = ag.multiply_matrices(cab.T,
                                                       node_cga.T,
                                                       FoR_cga)

    # Constrain angular velocities
    LM_Q[:sys_size] += scalingFactor*np.dot(np.transpose(Bnh), Lambda_dot[ieq:ieq+num_LM_eq_specific])
    LM_Q[sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += scalingFactor*(np.dot(Bnh[:, FoR_dof+3:FoR_dof+6], FoR_wa) -
                                                                         rot_vect)

    LM_C[sys_size+ieq:sys_size+ieq+num_LM_eq_specific,:sys_size] += scalingFactor*Bnh
    LM_C[:sys_size,sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += scalingFactor*np.transpose(Bnh)

    if MB_beam[node_body].FoR_movement == 'free':
        LM_C[FoR_dof+3:FoR_dof+6,node_FoR_dof+6:node_FoR_dof+10] += scalingFactor*np.dot(FoR_cga.T,
                                                                           ag.der_Cquat_by_v(MB_tstep[node_body].quat,
                                                                                                  np.dot(cab, Lambda_dot[ieq:ieq+num_LM_eq_specific])))

    LM_C[FoR_dof+3:FoR_dof+6,FoR_dof+6:FoR_dof+10] += scalingFactor*ag.der_CquatT_by_v(MB_tstep[FoR_body].quat,
                                                                              ag.multiply_matrices(node_cga, cab, Lambda_dot[ieq:ieq+num_LM_eq_specific]))

    LM_K[FoR_dof+3:FoR_dof+6,node_dof+3:node_dof+6] += scalingFactor*ag.multiply_matrices(FoR_cga.T,
                                                                         node_cga,
                                                                         ag.der_Ccrv_by_v(MB_tstep[node_body].psi[ielem,inode_in_elem,:],
                                                                                                Lambda_dot[ieq:ieq+num_LM_eq_specific]))

    if penaltyFactor:
        LM_C[FoR_dof+3:FoR_dof+6, FoR_dof+3:FoR_dof+6] += penaltyFactor*np.eye(3)

        q = np.zeros((sys_size))
        q[FoR_dof+3:FoR_dof+6] = FoR_wa
        LM_Q[:sys_size] += penaltyFactor*np.dot(np.dot(Bnh.T, Bnh), q)

    ieq += 3
    return ieq


################################################################################
# Lagrange constraints
################################################################################
@lagrangeconstraint
class hinge_node_FoR(BaseLagrangeConstraint):
    __doc__ = """
    hinge_node_FoR

    This constraint forces a hinge behaviour between a node and a FoR

    See ``LagrangeConstraints`` for the description of variables

    Attributes:
        node_number (int): number of the "node" within its own body
        node_body (int): body number of the "node"
        FoR_body (int): body number of the "FoR"
        rot_axisB (np.ndarray): Rotation axis with respect to the node B FoR
    """
    _lc_id = 'hinge_node_FoR'

    def __init__(self):
        self.required_parameters = ['node_in_body', 'body', 'body_FoR', 'rot_axisB']
        self._n_eq = 5

    def get_n_eq(self):
        return self._n_eq

    def initialise(self, MBdict_entry, ieq, print_info=True):
        self.node_number = MBdict_entry['node_in_body']
        self.node_body = MBdict_entry['body']
        self.FoR_body = MBdict_entry['body_FoR']
        self.rot_axisB = MBdict_entry['rot_axisB']
        self._ieq = ieq
        self.scalingFactor = set_value_or_default(MBdict_entry, "scalingFactor", 1.)
        self.penaltyFactor = set_value_or_default(MBdict_entry, "penaltyFactor", 0.)

        if (self.rot_axisB[[1, 2]]  == 0).all():
            self.rot_dir = 'x'
            self.zero_comp = np.array([1, 2], dtype=int)
        elif (self.rot_axisB[[0, 2]]  == 0).all():
            self.rot_dir = 'y'
            self.zero_comp = np.array([0, 2], dtype=int)
        elif (self.rot_axisB[[0, 1]]  == 0).all():
            self.rot_dir = 'z'
            self.zero_comp = np.array([0, 1], dtype=int)
        else:
            raise NotImplementedError("Hinges should be parallel to the xB, yB or zB of the reference node")
            #self.rot_dir = 'general'
            #self.indep = []

        return self._ieq + self._n_eq

    def staticmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot):
        return

    def dynamicmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot):

        # Define the position of the first degree of freedom associated to the node
        node_dof = define_node_dof(MB_beam, self.node_body, self.node_number)
        node_FoR_dof = define_FoR_dof(MB_beam, self.node_body)
        FoR_dof = define_FoR_dof(MB_beam, self.FoR_body)
        ieq = self._ieq

        # Define the equations
        # ieq =  equal_pos_node_FoR(MB_tstep, MB_beam, self.FoR_body, self.node_body, self.node_number, node_FoR_dof, node_dof, FoR_dof, sys_size, Lambda, self.scalingFactor, self.penaltyFactor, ieq, LM_K, LM_C, LM_Q)
        ieq = equal_lin_vel_node_FoR(MB_tstep, MB_beam, self.FoR_body, self.node_body, self.node_number, node_FoR_dof, node_dof, FoR_dof, sys_size, Lambda_dot, self.scalingFactor, self.penaltyFactor, ieq, LM_K, LM_C, LM_Q)
        if self.rot_dir == 'general':
            ieq = def_rot_axis_FoR_wrt_node_general(MB_tstep, MB_beam, self.FoR_body, self.node_body, self.node_number, node_FoR_dof, node_dof, FoR_dof, sys_size, Lambda_dot, self.rot_axisB, self.scalingFactor, self.penaltyFactor, ieq, LM_K, LM_C, LM_Q, self.indep)
        else:
            ieq = def_rot_axis_FoR_wrt_node_xyz(MB_tstep, MB_beam, self.FoR_body, self.node_body, self.node_number, node_FoR_dof, node_dof, FoR_dof, sys_size, Lambda_dot, self.rot_axisB, self.scalingFactor, self.penaltyFactor, ieq, LM_K, LM_C, LM_Q, self.zero_comp)

        return

    def staticpost(self, lc_list, MB_beam, MB_tstep):
        return

    def dynamicpost(self, lc_list, MB_beam, MB_tstep):
        MB_tstep[self.FoR_body].for_pos[0:3] = np.dot(MB_tstep[self.node_body].cga(), MB_tstep[self.node_body].pos[self.node_number,:]) + MB_tstep[self.node_body].for_pos[0:3]
        return


@lagrangeconstraint
class hinge_node_FoR_constant_vel(BaseLagrangeConstraint):
    __doc__ = """
    hinge_node_FoR_constant_vel

    This constraint forces a hinge behaviour between a node and a FoR and
    a constant rotation velocity at the join

    See ``LagrangeConstraints`` for the description of variables

    Attributes:
        node_number (int): number of the "node" within its own body
        node_body (int): body number of the "node"
        FoR_body (int): body number of the "FoR"
        rot_vect (np.ndarray): Rotation velocity vector in the node B FoR
        rel_posB (np.ndarray): Relative position between the node and the frame of reference in the node B FoR
    """
    _lc_id = 'hinge_node_FoR_constant_vel'

    def __init__(self):
        self.required_parameters = ['node_in_body', 'body', 'body_FoR', 'rot_vect', 'rel_posB']
        self._n_eq = 6

    def get_n_eq(self):
        return self._n_eq

    def initialise(self, MBdict_entry, ieq, print_info=True):

        self.node_number = MBdict_entry['node_in_body']
        self.node_body = MBdict_entry['body']
        self.FoR_body = MBdict_entry['body_FoR']
        self.rel_posB = MBdict_entry['rel_posB']
        self._ieq = ieq
        self.indep = []
        self.scalingFactor = set_value_or_default(MBdict_entry, "scalingFactor", 1.)
        self.penaltyFactor = set_value_or_default(MBdict_entry, "penaltyFactor", 0.)

        self.rot_axisB = ag.unit_vector(MBdict_entry['rot_vect'])
        if (self.rot_axisB[[1, 2]]  == 0).all():
            self.rot_dir = 'x'
            self.zero_comp = np.array([1, 2], dtype=int)
            self.nonzero_comp = 0
        elif (self.rot_axisB[[0, 2]]  == 0).all():
            self.rot_dir = 'y'
            self.zero_comp = np.array([0, 2], dtype=int)
            self.nonzero_comp = 1
        elif (self.rot_axisB[[0, 1]]  == 0).all():
            self.rot_dir = 'z'
            self.zero_comp = np.array([0, 1], dtype=int)
            self.nonzero_comp = 2
        else:
            raise NotImplementedError("Hinges should be parallel to the xB, yB or zB of the reference node")
        self.set_rot_vel(MBdict_entry['rot_vect'][self.nonzero_comp])

        return self._ieq + self._n_eq


    def set_rot_vel(self, rot_vel):
        self.rot_vel = rot_vel


    def staticmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot):
        return

    def dynamicmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot):

        # Define the position of the first degree of freedom associated to the node
        node_dof = define_node_dof(MB_beam, self.node_body, self.node_number)
        node_FoR_dof = define_FoR_dof(MB_beam, self.node_body)
        FoR_dof = define_FoR_dof(MB_beam, self.FoR_body)
        ieq = self._ieq

        # Define the equations
        # ieq =  equal_pos_node_FoR(MB_tstep, MB_beam, self.FoR_body, self.node_body, self.node_number, node_FoR_dof, node_dof, FoR_dof, sys_size, Lambda, self.scalingFactor, self.penaltyFactor, ieq, LM_K, LM_C, LM_Q)
        ieq = equal_lin_vel_node_FoR(MB_tstep, MB_beam, self.FoR_body, self.node_body, self.node_number, node_FoR_dof, node_dof, FoR_dof, sys_size, Lambda_dot, self.scalingFactor, self.penaltyFactor, ieq, LM_K, LM_C, LM_Q, rel_posB=self.rel_posB)
        # ieq = def_rot_vect_FoR_wrt_node(MB_tstep, MB_beam, self.FoR_body, self.node_body, self.node_number, node_FoR_dof, node_dof, FoR_dof, sys_size, Lambda_dot, self.rot_vect, self.scalingFactor, self.penaltyFactor, ieq, LM_K, LM_C, LM_Q)
        ieq = def_rot_axis_FoR_wrt_node_xyz(MB_tstep, MB_beam, self.FoR_body, self.node_body, self.node_number, node_FoR_dof, node_dof, FoR_dof, sys_size, Lambda_dot, self.rot_axisB, self.scalingFactor, self.penaltyFactor, ieq, LM_K, LM_C, LM_Q, self.zero_comp)
        ieq = def_rot_vel_mod_FoR_wrt_node(MB_tstep, MB_beam, self.FoR_body, self.node_body, self.node_number, node_FoR_dof, node_dof, FoR_dof, sys_size, Lambda_dot, self.nonzero_comp, self.rot_vel, self.scalingFactor, self.penaltyFactor, ieq, LM_K, LM_C, LM_Q)
        return

    def staticpost(self, lc_list, MB_beam, MB_tstep):
        return

    def dynamicpost(self, lc_list, MB_beam, MB_tstep):

        ielem, inode_in_elem = MB_beam[self.node_body].node_master_elem[self.node_number]
        node_cga = MB_tstep[self.node_body].cga()
        cab = ag.crv2rotation(MB_tstep[self.node_body].psi[ielem, inode_in_elem, :])

        MB_tstep[self.FoR_body].for_pos[0:3] = (np.dot(node_cga,
                                                       MB_tstep[self.node_body].pos[self.node_number,:] + np.dot(cab, self.rel_posB)) +
                                                MB_tstep[self.node_body].for_pos[0:3])
        return


@lagrangeconstraint
class hinge_node_FoR_pitch(BaseLagrangeConstraint):
    __doc__ = """
    hinge_node_FoR_pitch

    This constraint forces a hinge behaviour between a node and a FoR and
    a rotation velocity at the joint

    See ``LagrangeConstraints`` for the description of variables

    Attributes:
        node_number (int): number of the "node" within its own body
        node_body (int): body number of the "node"
        FoR_body (int): body number of the "FoR"
        rot_vect (np.ndarray): Rotation velocity vector in the node B FoR
        rel_posB (np.ndarray): Relative position between the node and the frame of reference in the node B FoR
    """
    _lc_id = 'hinge_node_FoR_pitch'

    def __init__(self):
        self.required_parameters = ['node_in_body', 'body', 'body_FoR', 'rotor_vel', 'rel_posB']
        self._n_eq = 6

    def get_n_eq(self):
        return self._n_eq

    def initialise(self, MBdict_entry, ieq, print_info=True):

        self.node_number = MBdict_entry['node_in_body']
        self.node_body = MBdict_entry['body']
        self.FoR_body = MBdict_entry['body_FoR']
        self.rel_posB = MBdict_entry['rel_posB']
        self._ieq = ieq
        self.indep = []
        self.scalingFactor = set_value_or_default(MBdict_entry, "scalingFactor", 1.)
        self.penaltyFactor = set_value_or_default(MBdict_entry, "penaltyFactor", 0.)

        self.set_rotor_vel(MBdict_entry['rotor_vel'])
        pitch_vel = set_value_or_default(MBdict_entry, "pitch_vel", 0.)
        self.set_pitch_vel(pitch_vel)

        return self._ieq + self._n_eq


    def set_rotor_vel(self, rotor_vel):
        self.rotor_vel = rotor_vel


    def set_pitch_vel(self, pitch_vel):
        self.pitch_vel = pitch_vel


    def staticmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot):
        return

    def dynamicmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot):

        # Define the position of the first degree of freedom associated to the node
        node_dof = define_node_dof(MB_beam, self.node_body, self.node_number)
        node_FoR_dof = define_FoR_dof(MB_beam, self.node_body)
        FoR_dof = define_FoR_dof(MB_beam, self.FoR_body)
        ieq = self._ieq

        # Compute relative velocity
        ielem, inode_in_elem = MB_beam[self.node_body].node_master_elem[self.node_number]
        node_cga = MB_tstep[self.node_body].cga()
        cab = ag.crv2rotation(MB_tstep[self.node_body].psi[ielem, inode_in_elem, :])
        FoR_cga = MB_tstep[self.FoR_body].cga()

        # rel_vel in B FoR
        rel_vel = np.array([0., 0., self.rotor_vel])
        rel_vel += ag.multiply_matrices(cab.T, node_cga.T, FoR_cga,
                                        np.array([self.pitch_vel, 0., 0.]))

        # Define the equations
        ieq = equal_lin_vel_node_FoR(MB_tstep, MB_beam, self.FoR_body, self.node_body, self.node_number, node_FoR_dof, node_dof, FoR_dof, sys_size, Lambda_dot, self.scalingFactor, self.penaltyFactor, ieq, LM_K, LM_C, LM_Q, rel_posB=self.rel_posB)
        ieq = rel_rot_vel_node_FoR(MB_tstep, MB_beam, self.FoR_body, self.node_body, self.node_number, node_FoR_dof, node_dof, FoR_dof, sys_size, Lambda_dot, self.scalingFactor, self.penaltyFactor, ieq, LM_K, LM_C, LM_Q, rel_vel=rel_vel)
        return

    def staticpost(self, lc_list, MB_beam, MB_tstep):
        return

    def dynamicpost(self, lc_list, MB_beam, MB_tstep):

        ielem, inode_in_elem = MB_beam[self.node_body].node_master_elem[self.node_number]
        node_cga = MB_tstep[self.node_body].cga()
        cab = ag.crv2rotation(MB_tstep[self.node_body].psi[ielem, inode_in_elem, :])

        MB_tstep[self.FoR_body].for_pos[0:3] = (np.dot(node_cga,
                                                       MB_tstep[self.node_body].pos[self.node_number,:] + np.dot(cab, self.rel_posB)) +
                                                MB_tstep[self.node_body].for_pos[0:3])

        return


@lagrangeconstraint
class spherical_node_FoR(BaseLagrangeConstraint):
    __doc__ = """
    spherical_node_FoR

    This constraint forces a spherical join between a node and a FoR

    See ``LagrangeConstraints`` for the description of variables

    Attributes:
        node_number (int): number of the "node" within its own body
        node_body (int): body number of the "node"
        FoR_body (int): body number of the "FoR"
    """
    _lc_id = 'spherical_node_FoR'

    def __init__(self):
        self.required_parameters = ['node_in_body', 'body', 'body_FoR']
        self._n_eq = 3

    def get_n_eq(self):
        return self._n_eq

    def initialise(self, MBdict_entry, ieq, print_info=True):

        self.node_number = MBdict_entry['node_in_body']
        self.node_body = MBdict_entry['body']
        self.FoR_body = MBdict_entry['body_FoR']
        self._ieq = ieq
        self.scalingFactor = set_value_or_default(MBdict_entry, "scalingFactor", 1.)
        self.penaltyFactor = set_value_or_default(MBdict_entry, "penaltyFactor", 0.)

        return self._ieq + self._n_eq

    def staticmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot):
        return

    def dynamicmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot):

        # Define the position of the first degree of freedom associated to the node
        node_dof = define_node_dof(MB_beam, self.node_body, self.node_number)
        node_FoR_dof = define_FoR_dof(MB_beam, self.node_body)
        FoR_dof = define_FoR_dof(MB_beam, self.FoR_body)
        ieq = self._ieq

        # Define the equations
        ieq = equal_lin_vel_node_FoR(MB_tstep, MB_beam, self.FoR_body, self.node_body, self.node_number, node_FoR_dof, node_dof, FoR_dof, sys_size, Lambda_dot, self.scalingFactor, self.penaltyFactor, ieq, LM_K, LM_C, LM_Q)

        return

    def staticpost(self, lc_list, MB_beam, MB_tstep):
        return

    def dynamicpost(self, lc_list, MB_beam, MB_tstep):
        MB_tstep[self.FoR_body].for_pos[0:3] = np.dot(MB_tstep[self.node_body].cga(), MB_tstep[self.node_body].pos[self.node_number,:]) + MB_tstep[self.node_body].for_pos[0:3]
        return


@lagrangeconstraint
class free(BaseLagrangeConstraint):
    _lc_id = 'free'
    __doc__ = _lc_id

    def __init__(self):
        self.required_parameters = []
        self._n_eq = 0

    def get_n_eq(self):
        return self._n_eq

    def initialise(self, MBdict_entry, ieq, print_info=True):

        self._ieq = ieq
        return self._ieq + self._n_eq

    def staticmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot):
        return

    def dynamicmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot):
        return

    def staticpost(self, lc_list, MB_beam, MB_tstep):
        return

    def dynamicpost(self, lc_list, MB_beam, MB_tstep):
        return


@lagrangeconstraint
class spherical_FoR(BaseLagrangeConstraint):
    __doc__ = """
    spherical_FoR

    This constraint forces a spherical join at a FoR

    See ``LagrangeConstraints`` for the description of variables

    Attributes:
        body_FoR (int): body number of the "FoR"
    """
    _lc_id = 'spherical_FoR'

    def __init__(self):
        self.required_parameters = ['body_FoR']
        self._n_eq = 3

    def get_n_eq(self):
        return self._n_eq

    def initialise(self, MBdict_entry, ieq, print_info=True):

        self.body_FoR = MBdict_entry['body_FoR']
        self._ieq = ieq
        self.scalingFactor = set_value_or_default(MBdict_entry, "scalingFactor", 1.)
        self.penaltyFactor = set_value_or_default(MBdict_entry, "penaltyFactor", 0.)

        return self._ieq + self._n_eq

    def staticmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot):
        return

    def dynamicmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot):
        num_LM_eq_specific = self._n_eq
        Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')
        B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')

        # Define the position of the first degree of freedom associated to the FoR
        FoR_dof = define_FoR_dof(MB_beam, self.body_FoR)
        ieq = self._ieq

        Bnh[:3, FoR_dof:FoR_dof+3] = 1.0*np.eye(3)

        LM_C[sys_size+ieq:sys_size+ieq+num_LM_eq_specific,:sys_size] += self.scalingFactor*Bnh
        LM_C[:sys_size,sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += self.scalingFactor*np.transpose(Bnh)

        LM_Q[:sys_size] += self.scalingFactor*np.dot(np.transpose(Bnh),Lambda_dot[ieq:ieq+num_LM_eq_specific])

        LM_Q[sys_size+ieq:sys_size+ieq+3] += self.scalingFactor*MB_tstep[self.body_FoR].for_vel[0:3].astype(dtype=ct.c_double, copy=True, order='F')

        ieq += 3
        return

    def staticpost(self, lc_list, MB_beam, MB_tstep):
        return

    def dynamicpost(self, lc_list, MB_beam, MB_tstep):
        return


@lagrangeconstraint
class hinge_FoR(BaseLagrangeConstraint):
    __doc__ = """
    hinge_FoR

    This constraint forces a hinge at a FoR

    See ``LagrangeConstraints`` for the description of variables

    Attributes:
        body_FoR (int): body number of the "FoR"
        rot_axis_AFoR (np.ndarray): Rotation axis with respect to the node A FoR
    """
    _lc_id = 'hinge_FoR'

    def __init__(self):
        self.required_parameters = ['body_FoR', 'rot_axis_AFoR']
        self._n_eq = 5

    def get_n_eq(self):
        return self._n_eq

    def initialise(self, MBdict_entry, ieq, print_info=True):

        self.body_FoR = MBdict_entry['body_FoR']
        self.rot_axis = MBdict_entry['rot_axis_AFoR']
        self._ieq = ieq
        self.scalingFactor = set_value_or_default(MBdict_entry, "scalingFactor", 1.)
        self.penaltyFactor = set_value_or_default(MBdict_entry, "penaltyFactor", 0.)

        if (self.rot_axis[[1, 2]]  == 0).all():
            self.rot_dir = 'x'
            self.zero_comp = np.array([1, 2], dtype=int)
        elif (self.rot_axis[[0, 2]]  == 0).all():
            self.rot_dir = 'y'
            self.zero_comp = np.array([0, 2], dtype=int)
        elif (self.rot_axis[[0, 1]]  == 0).all():
            self.rot_dir = 'z'
            self.zero_comp = np.array([0, 1], dtype=int)
        else:
            self.rot_dir = 'general'

        return self._ieq + self._n_eq

    def staticmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot):
        return

    def dynamicmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot):
        num_LM_eq_specific = self._n_eq
        Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')
        B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')

        # Define the position of the first degree of freedom associated to the FoR
        FoR_dof = define_FoR_dof(MB_beam, self.body_FoR)
        ieq = self._ieq

        Bnh[:3, FoR_dof:FoR_dof+3] = 1.0*np.eye(3)

        if self.rot_dir == 'general':
            # Only two of these equations are linearly independent
            skew_rot_axis = ag.skew(self.rot_axis)
            n0 = np.linalg.norm(skew_rot_axis[0,:])
            n1 = np.linalg.norm(skew_rot_axis[1,:])
            n2 = np.linalg.norm(skew_rot_axis[2,:])
            if ((n0 < n1) and (n0 < n2)):
                row0 = 1
                row1 = 2
            elif ((n1 < n0) and (n1 < n2)):
                row0 = 0
                row1 = 2
            elif ((n2 < n0) and (n2 < n1)):
                row0 = 0
                row1 = 1
            Bnh[3:5, FoR_dof+3:FoR_dof+6] = skew_rot_axis[[row0,row1],:]
        else:
            Bnh[3:5, FoR_dof+3+self.zero_comp] = np.eye(2)

        LM_C[sys_size+ieq:sys_size+ieq+num_LM_eq_specific,:sys_size] += self.scalingFactor*Bnh
        LM_C[:sys_size,sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += self.scalingFactor*np.transpose(Bnh)

        LM_Q[:sys_size] += self.scalingFactor*np.dot(np.transpose(Bnh),Lambda_dot[ieq:ieq+num_LM_eq_specific])

        LM_Q[sys_size+ieq:sys_size+ieq+3] += self.scalingFactor*MB_tstep[self.body_FoR].for_vel[0:3].astype(dtype=ct.c_double, copy=True, order='F')
        if self.rot_dir == 'general':
            LM_Q[sys_size+ieq+3:sys_size+ieq+5] += self.scalingFactor*np.dot(skew_rot_axis[[row0,row1],:], MB_tstep[self.body_FoR].for_vel[3:6])
        else:
            LM_Q[sys_size+ieq+3:sys_size+ieq+5] += self.scalingFactor*MB_tstep[self.body_FoR].for_vel[3 + self.zero_comp]

        if self.penaltyFactor:
            LM_Q[FoR_dof:FoR_dof+3] += self.penaltyFactor*MB_tstep[self.body_FoR].for_vel[0:3]
            LM_C[FoR_dof:FoR_dof+3, FoR_dof:FoR_dof+3] += self.penaltyFactor*np.eye(3)

            if self.rot_dir == 'general':
                sq_rot_axis = np.dot(ag.skew(self.rot_axis).T, ag.skew(self.rot_axis))
                LM_Q[FoR_dof+3:FoR_dof+6] += self.penaltyFactor*np.dot(sq_rot_axis, MB_tstep[self.body_FoR].for_vel[3:6])
                LM_C[FoR_dof+3:FoR_dof+6, FoR_dof+3:FoR_dof+6] += self.penaltyFactor*sq_rot_axis
            else:
                LM_Q[FoR_dof+3:FoR_dof+6] += self.penaltyFactor*MB_tstep[self.body_FoR].for_vel[3:6]
                LM_C[FoR_dof+3:FoR_dof+6, FoR_dof+3:FoR_dof+6] += self.penaltyFactor*np.eye(3)

        ieq += 5
        return

    def staticpost(self, lc_list, MB_beam, MB_tstep):
        return

    def dynamicpost(self, lc_list, MB_beam, MB_tstep):
        return


@lagrangeconstraint
class hinge_FoR_wrtG(BaseLagrangeConstraint):
    __doc__ = """
    hinge_FoR_wrtG

    This constraint forces a hinge at a FoR

    See ``LagrangeConstraints`` for the description of variables

    Attributes:
        body_FoR (int): body number of the "FoR"
        rot_axis_AFoR (np.ndarray): Rotation axis with respect to the node G FoR
    """
    _lc_id = 'hinge_FoR_wrtG'

    def __init__(self):
        self.required_parameters = ['body_FoR', 'rot_axis_AFoR']
        self._n_eq = 5

    def get_n_eq(self):
        return self._n_eq

    def initialise(self, MBdict_entry, ieq, print_info=True):

        self.body_FoR = MBdict_entry['body_FoR']
        self.rot_axis = MBdict_entry['rot_axis_AFoR']
        self._ieq = ieq
        self.scalingFactor = set_value_or_default(MBdict_entry, "scalingFactor", 1.)
        self.penaltyFactor = set_value_or_default(MBdict_entry, "penaltyFactor", 0.)

        return self._ieq + self._n_eq

    def staticmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot):
        return

    def dynamicmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot):
        num_LM_eq_specific = self._n_eq
        Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')
        B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')

        # Define the position of the first degree of freedom associated to the FoR
        FoR_dof = define_FoR_dof(MB_beam, self.body_FoR)
        ieq = self._ieq

        Bnh[:3, FoR_dof:FoR_dof+3] = MB_tstep[self.body_FoR].cga()

        # Only two of these equations are linearly independent
        skew_rot_axis = ag.skew(self.rot_axis)
        n0 = np.linalg.norm(skew_rot_axis[0,:])
        n1 = np.linalg.norm(skew_rot_axis[1,:])
        n2 = np.linalg.norm(skew_rot_axis[2,:])
        if ((n0 < n1) and (n0 < n2)):
            row0 = 1
            row1 = 2
        elif ((n1 < n0) and (n1 < n2)):
            row0 = 0
            row1 = 2
        elif ((n2 < n0) and (n2 < n1)):
            row0 = 0
            row1 = 1

        Bnh[3:5, FoR_dof+3:FoR_dof+6] = skew_rot_axis[[row0,row1],:]

        LM_C[sys_size+ieq:sys_size+ieq+num_LM_eq_specific,:sys_size] += self.scalingFactor*Bnh
        LM_C[:sys_size,sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += self.scalingFactor*np.transpose(Bnh)

        LM_C[FoR_dof:FoR_dof+3,FoR_dof+6:FoR_dof+10] += self.scalingFactor*ag.der_CquatT_by_v(MB_tstep[self.body_FoR].quat,Lambda_dot[ieq:ieq+3])

        LM_Q[:sys_size] += self.scalingFactor*np.dot(np.transpose(Bnh),Lambda_dot[ieq:ieq+num_LM_eq_specific])

        LM_Q[sys_size+ieq:sys_size+ieq+3] += self.scalingFactor*np.dot(MB_tstep[self.body_FoR].cga(),MB_tstep[self.body_FoR].for_vel[0:3])
        LM_Q[sys_size+ieq+3:sys_size+ieq+5] += self.scalingFactor*np.dot(skew_rot_axis[[row0,row1],:], MB_tstep[self.body_FoR].for_vel[3:6])

        ieq += 5
        return

    def staticpost(self, lc_list, MB_beam, MB_tstep):
        return

    def dynamicpost(self, lc_list, MB_beam, MB_tstep):
        return


@lagrangeconstraint
class fully_constrained_node_FoR(BaseLagrangeConstraint):
    __doc__ = """
    fully_constrained_node_FoR

    This constraint forces linear and angular displacements between a node
    and a FoR to be the same

    See ``LagrangeConstraints`` for the description of variables

    Attributes:
        node_number (int): number of the "node" within its own body
        node_body (int): body number of the "node"
        FoR_body (int): body number of the "FoR"
    """
    _lc_id = 'fully_constrained_node_FoR'

    def __init__(self):
        self.required_parameters = ['node_in_body', 'body', 'body_FoR', 'rel_posB']
        self._n_eq = 6

    def get_n_eq(self):
        return self._n_eq

    def initialise(self, MBdict_entry, ieq, print_info=True):

        # cout.cout_wrap("WARNING: do not use fully_constrained_node_FoR. It is outdated. Definetly not working if 'body' has velocity", 3)
        self.node_number = MBdict_entry['node_in_body']
        self.node_body = MBdict_entry['body']
        self.FoR_body = MBdict_entry['body_FoR']
        self.rel_posB = MBdict_entry['rel_posB']
        self._ieq = ieq
        self.scalingFactor = set_value_or_default(MBdict_entry, "scalingFactor", 1.)
        self.penaltyFactor = set_value_or_default(MBdict_entry, "penaltyFactor", 0.)

        return self._ieq + self._n_eq

    def staticmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot):
        return

    def dynamicmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot):

        # Define the position of the first degree of freedom associated to the node
        node_dof = define_node_dof(MB_beam, self.node_body, self.node_number)
        node_FoR_dof = define_FoR_dof(MB_beam, self.node_body)
        FoR_dof = define_FoR_dof(MB_beam, self.FoR_body)
        ieq = self._ieq

        # Define the equations
        ieq = equal_lin_vel_node_FoR(MB_tstep, MB_beam, self.FoR_body, self.node_body, self.node_number, node_FoR_dof, node_dof, FoR_dof, sys_size, Lambda_dot, self.scalingFactor, self.penaltyFactor, ieq, LM_K, LM_C, LM_Q, rel_posB=self.rel_posB)
        ieq = rel_rot_vel_node_FoR(MB_tstep, MB_beam, self.FoR_body, self.node_body, self.node_number, node_FoR_dof, node_dof, FoR_dof, sys_size, Lambda_dot, self.scalingFactor, self.penaltyFactor, ieq, LM_K, LM_C, LM_Q, rel_vel=np.zeros((3)))

        return

    def staticpost(self, lc_list, MB_beam, MB_tstep):
        return

    def dynamicpost(self, lc_list, MB_beam, MB_tstep):
        ielem, inode_in_elem = MB_beam[self.node_body].node_master_elem[self.node_number]
        node_cga = MB_tstep[self.node_body].cga()
        cab = ag.crv2rotation(MB_tstep[self.node_body].psi[ielem, inode_in_elem, :])

        MB_tstep[self.FoR_body].for_pos[0:3] = (np.dot(node_cga,
                                                       MB_tstep[self.node_body].pos[self.node_number,:] + np.dot(cab, self.rel_posB)) +
                                                MB_tstep[self.node_body].for_pos[0:3])
        return


@lagrangeconstraint
class constant_rot_vel_FoR(BaseLagrangeConstraint):
    __doc__ = """
    constant_rot_vel_FoR

    This constraint forces a constant rotation velocity of a FoR

    See ``LagrangeConstraints`` for the description of variables

    Attributes:
        FoR_body (int): body number of the "FoR"
    """
    _lc_id = 'constant_rot_vel_FoR'

    def __init__(self):
        self.required_parameters = ['FoR_body', 'rot_vel']
        self._n_eq = 3

    def get_n_eq(self):
        return self._n_eq

    def initialise(self, MBdict_entry, ieq, print_info=True):

        self.rot_vel = MBdict_entry['rot_vel']
        self.FoR_body = MBdict_entry['FoR_body']
        self._ieq = ieq
        self.scalingFactor = set_value_or_default(MBdict_entry, "scalingFactor", 1.)
        self.penaltyFactor = set_value_or_default(MBdict_entry, "penaltyFactor", 0.)

        return self._ieq + self._n_eq

    def staticmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot):
        return

    def dynamicmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot):
        num_LM_eq_specific = self._n_eq
        Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')
        B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')

        # Define the position of the first degree of freedom associated to the FoR
        FoR_dof = define_FoR_dof(MB_beam, self.FoR_body)
        ieq = self._ieq

        Bnh[:3,FoR_dof+3:FoR_dof+6] = np.eye(3)

        LM_C[sys_size+ieq:sys_size+ieq+num_LM_eq_specific,:sys_size] += self.scalingFactor*Bnh
        LM_C[:sys_size,sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += self.scalingFactor*np.transpose(Bnh)

        LM_Q[:sys_size] += self.scalingFactor*np.dot(np.transpose(Bnh),Lambda_dot[ieq:ieq+num_LM_eq_specific])
        LM_Q[sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += self.scalingFactor*(MB_tstep[self.FoR_body].for_vel[3:6] - self.rot_vel)

        ieq += 3
        return

    def staticpost(self, lc_list, MB_beam, MB_tstep):
        return

    def dynamicpost(self, lc_list, MB_beam, MB_tstep):
        return

@lagrangeconstraint
class constant_vel_FoR(BaseLagrangeConstraint):
    __doc__ = """
    constant_vel_FoR

    This constraint forces a constant velocity of a FoR

    See ``LagrangeConstraints`` for the description of variables

    Attributes:
        FoR_body (int): body number of the "FoR"
        vel (np.ndarray): 6 components of the desired velocity
    """
    _lc_id = 'constant_vel_FoR'

    def __init__(self):
        self.required_parameters = ['FoR_body', 'vel']
        self._n_eq = 6

    def get_n_eq(self):
        return self._n_eq

    def initialise(self, MBdict_entry, ieq, print_info=True):

        self.vel = MBdict_entry['vel']
        self.FoR_body = MBdict_entry['FoR_body']
        self._ieq = ieq
        self.scalingFactor = set_value_or_default(MBdict_entry, "scalingFactor", 1.)
        self.penaltyFactor = set_value_or_default(MBdict_entry, "penaltyFactor", 0.)

        return self._ieq + self._n_eq

    def staticmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot):
        return

    def dynamicmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot):
        num_LM_eq_specific = self._n_eq
        Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order='F')
        B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order='F')

        # Define the position of the first degree of freedom associated to the FoR
        FoR_dof = define_FoR_dof(MB_beam, self.FoR_body)
        ieq = self._ieq

        Bnh[:num_LM_eq_specific, FoR_dof:FoR_dof+6] = np.eye(6)

        LM_C[sys_size + ieq:sys_size + ieq + num_LM_eq_specific, :sys_size] += self.scalingFactor * Bnh
        LM_C[:sys_size, sys_size + ieq:sys_size + ieq + num_LM_eq_specific] += self.scalingFactor * np.transpose(Bnh)

        LM_Q[:sys_size] += self.scalingFactor * np.dot(np.transpose(Bnh), Lambda_dot[ieq:ieq + num_LM_eq_specific])
        LM_Q[sys_size + ieq:sys_size + ieq + num_LM_eq_specific] += self.scalingFactor*(MB_tstep[self.FoR_body].for_vel - self.vel)

        ieq += 6
        return

    def staticpost(self, lc_list, MB_beam, MB_tstep):
        return

    def dynamicpost(self, lc_list, MB_beam, MB_tstep):
        return


@lagrangeconstraint
class zero_lin_vel_sine_rot_vel_FoR(BaseLagrangeConstraint):
    __doc__ = """
    zero_lin_vel_sine_rot_vel_FoR

    Zero linear velocity and sinusoidal rotation velocity FoR

    See ``LagrangeConstraints`` for the description of variables

    Attributes:
        FoR_body (int): body number of the "FoR"
        vel_amp (float): Rotation velocity amplitude
        omega (float): Frequency of the sinusoidally-varying rotation velocity
        xyz (string): Axis with the sine velocity
    """
    _lc_id = 'zero_lin_vel_sine_rot_vel_FoR'

    def __init__(self):
        self.required_parameters = ['FoR_body', 'vel_amp', 'omega', 'xyz']
        self._n_eq = 6

    def get_n_eq(self):
        return self._n_eq

    def initialise(self, MBdict_entry, ieq, print_info=True):

        self.FoR_body = MBdict_entry['FoR_body']
        self.vel_amp = MBdict_entry['vel_amp']
        self.omega = MBdict_entry['omega']
        if MBdict_entry['xyz'] == 'x':
            self.xyz_index = 0
        elif MBdict_entry['xyz'] == 'y':
            self.xyz_index = 1
        elif MBdict_entry['xyz'] == 'z':
            self.xyz_index = 2
        else:
            raise NotImplementedError("FoR rotation velocity shouldd be parallel to x, y or z")
        self._ieq = ieq
        self.scalingFactor = set_value_or_default(MBdict_entry, "scalingFactor", 1.)
        self.penaltyFactor = set_value_or_default(MBdict_entry, "penaltyFactor", 0.)

        return self._ieq + self._n_eq

    def staticmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot):
        return

    def dynamicmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot):
        num_LM_eq_specific = self._n_eq
        Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order='F')
        B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order='F')

        # Define the position of the first degree of freedom associated to the FoR
        FoR_dof = define_FoR_dof(MB_beam, self.FoR_body)
        ieq = self._ieq

        vel = np.zeros((6))
        vel[3 + self.xyz_index] = self.vel_amp*np.sin(self.omega*ts*dt)

        Bnh[:num_LM_eq_specific, FoR_dof:FoR_dof+6] = np.eye(6)

        LM_C[sys_size + ieq:sys_size + ieq + num_LM_eq_specific, :sys_size] += self.scalingFactor * Bnh
        LM_C[:sys_size, sys_size + ieq:sys_size + ieq + num_LM_eq_specific] += self.scalingFactor * np.transpose(Bnh)

        LM_Q[:sys_size] += self.scalingFactor * np.dot(np.transpose(Bnh), Lambda_dot[ieq:ieq + num_LM_eq_specific])
        LM_Q[sys_size + ieq:sys_size + ieq + num_LM_eq_specific] += self.scalingFactor*(MB_tstep[self.FoR_body].for_vel - vel)

        ieq += 6
        return

    def staticpost(self, lc_list, MB_beam, MB_tstep):
        return

    def dynamicpost(self, lc_list, MB_beam, MB_tstep):
        return


@lagrangeconstraint
class lin_vel_node_wrtA(BaseLagrangeConstraint):
    __doc__ = """
    lin_vel_node_wrtA

    This constraint forces the linear velocity of a node to have a
    certain value with respect to the A FoR

    See ``LagrangeConstraints`` for the description of variables

    Attributes:
        node_number (int): number of the "node" within its own body
        body_number (int): body number of the "node"
        vel (np.ndarray): 6 components of the desired velocity with respect to the A FoR
    """
    _lc_id = 'lin_vel_node_wrtA'

    def __init__(self):
        self.required_parameters = ['velocity', 'body_number', 'node_number']
        self._n_eq = 3

    def get_n_eq(self):
        return self._n_eq

    def initialise(self, MBdict_entry, ieq, print_info=True):

        self.vel = MBdict_entry['velocity']
        self.body_number = MBdict_entry['body_number']
        self.node_number = MBdict_entry['node_number']
        self._ieq = ieq
        self.scalingFactor = set_value_or_default(MBdict_entry, "scalingFactor", 1.)
        self.penaltyFactor = set_value_or_default(MBdict_entry, "penaltyFactor", 0.)

        return self._ieq + self._n_eq

    def staticmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot):

        num_LM_eq_specific = self._n_eq
        B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order='F')

        # Define the position of the first degree of freedom associated to the FoR
        node_dof = define_node_dof(MB_beam, self.body_number, self.node_number)
        ieq = self._ieq

        B[:num_LM_eq_specific, node_dof:node_dof+3] = np.eye(3)

        LM_K[sys_size + ieq:sys_size + ieq + num_LM_eq_specific, :sys_size] += self.scalingFactor * B
        LM_K[:sys_size, sys_size + ieq:sys_size + ieq + num_LM_eq_specific] += self.scalingFactor * np.transpose(B)

        LM_Q[:sys_size] += self.scalingFactor * np.dot(np.transpose(B), Lambda[ieq:ieq + num_LM_eq_specific])
        LM_Q[sys_size + ieq:sys_size + ieq + num_LM_eq_specific] += self.scalingFactor*(MB_tstep[self.body_number].pos[self.node_number,:] -
                                                                                        MB_beam[self.body_number].ini_info.pos[self.node_number,:])

        ieq += 3

        return

    def dynamicmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot):

        if len(self.vel.shape) > 1:
            current_vel = self.vel[ts-1, :]
        else:
            current_vel = self.vel

        num_LM_eq_specific = self._n_eq
        Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order='F')
        B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order='F')

        # Define the position of the first degree of freedom associated to the FoR
        node_dof = define_node_dof(MB_beam, self.body_number, self.node_number)
        ieq = self._ieq

        Bnh[:num_LM_eq_specific, node_dof:node_dof+3] = np.eye(3)

        LM_C[sys_size + ieq:sys_size + ieq + num_LM_eq_specific, :sys_size] += self.scalingFactor * Bnh
        LM_C[:sys_size, sys_size + ieq:sys_size + ieq + num_LM_eq_specific] += self.scalingFactor * np.transpose(Bnh)

        LM_Q[:sys_size] += self.scalingFactor * np.dot(np.transpose(Bnh), Lambda_dot[ieq:ieq + num_LM_eq_specific])
        LM_Q[sys_size + ieq:sys_size + ieq + num_LM_eq_specific] += self.scalingFactor*(MB_tstep[self.body_number].pos_dot[self.node_number,:] - current_vel)

        ieq += 3
        return

    def staticpost(self, lc_list, MB_beam, MB_tstep):
        return

    def dynamicpost(self, lc_list, MB_beam, MB_tstep):
        return

@lagrangeconstraint
class lin_vel_node_wrtG(BaseLagrangeConstraint):
    __doc__ = """
    lin_vel_node_wrtG

    This constraint forces the linear velocity of a node to have a
    certain value with respect to the G FoR

    See ``LagrangeConstraints`` for the description of variables

    Attributes:
        node_number (int): number of the "node" within its own body
        body_number (int): body number of the "node"
        vel (np.ndarray): 6 components of the desired velocity with respect to the G FoR
    """
    _lc_id = 'lin_vel_node_wrtG'

    def __init__(self):
        self.required_parameters = ['velocity', 'body_number', 'node_number']
        self._n_eq = 3

    def get_n_eq(self):
        return self._n_eq

    def initialise(self, MBdict_entry, ieq, print_info=True):

        self.vel = MBdict_entry['velocity']
        self.body_number = MBdict_entry['body_number']
        self.node_number = MBdict_entry['node_number']
        self._ieq = ieq
        self.scalingFactor = set_value_or_default(MBdict_entry, "scalingFactor", 1.)
        self.penaltyFactor = set_value_or_default(MBdict_entry, "penaltyFactor", 0.)

        return self._ieq + self._n_eq

    def staticmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot):

        num_LM_eq_specific = self._n_eq
        B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order='F')

        # Define the position of the first degree of freedom associated to the FoR
        node_dof = define_node_dof(MB_beam, self.body_number, self.node_number)
        ieq = self._ieq

        B[:num_LM_eq_specific, node_dof:node_dof+3] = MB_tstep[self.body_number].cga()

        LM_K[sys_size + ieq:sys_size + ieq + num_LM_eq_specific, :sys_size] += self.scalingFactor * B
        LM_K[:sys_size, sys_size + ieq:sys_size + ieq + num_LM_eq_specific] += self.scalingFactor * np.transpose(B)

        LM_Q[:sys_size] += self.scalingFactor * np.dot(np.transpose(B), Lambda[ieq:ieq + num_LM_eq_specific])
        LM_Q[sys_size + ieq:sys_size + ieq + num_LM_eq_specific] += self.scalingFactor*(np.dot(MB_tstep[self.body_number].cga(), MB_tstep[self.body_number].pos[self.node_number,:]) +
                                                                     MB_tstep[self.body_number].for_pos)
        LM_Q[sys_size + ieq:sys_size + ieq + num_LM_eq_specific] -= self.scalingFactor*(np.dot(MB_beam[self.body_number].ini_info.cga(), MB_beam[self.body_number].ini_info.pos[self.node_number,:]) +
                                                                     MB_beam[self.body_number].ini_info.for_pos)

        ieq += 3

        return

    def dynamicmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot):
        if len(self.vel.shape) > 1:
            current_vel = self.vel[ts-1, :]
        else:
            current_vel = self.vel

        num_LM_eq_specific = self._n_eq
        Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order='F')
        B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order='F')

        # Define the position of the first degree of freedom associated to the FoR
        FoR_dof = define_FoR_dof(MB_beam, self.body_number)
        node_dof = define_node_dof(MB_beam, self.body_number, self.node_number)
        ieq = self._ieq

        if MB_beam[self.body_number].FoR_movement == 'free':
            Bnh[:num_LM_eq_specific, FoR_dof:FoR_dof+3] = MB_tstep[self.body_number].cga()
            Bnh[:num_LM_eq_specific, FoR_dof+3:FoR_dof+6] = -np.dot(MB_tstep[self.body_number].cga(), ag.skew(MB_tstep[self.body_number].pos[self.node_number,:]))
        Bnh[:num_LM_eq_specific, node_dof:node_dof+3] = MB_tstep[self.body_number].cga()

        LM_C[sys_size + ieq:sys_size + ieq + num_LM_eq_specific, :sys_size] += self.scalingFactor * Bnh
        LM_C[:sys_size, sys_size + ieq:sys_size + ieq + num_LM_eq_specific] += self.scalingFactor * np.transpose(Bnh)

        if MB_beam[self.body_number].FoR_movement == 'free':
            LM_C[FoR_dof:FoR_dof+3, FoR_dof+6:FoR_dof+10] += self.scalingFactor*ag.der_CquatT_by_v(MB_tstep[self.body_number].quat,Lambda_dot[ieq:ieq + num_LM_eq_specific])
            LM_C[node_dof:node_dof+3, FoR_dof+6:FoR_dof+10] += self.scalingFactor*ag.der_CquatT_by_v(MB_tstep[self.body_number].quat,Lambda_dot[ieq:ieq + num_LM_eq_specific])
            LM_C[FoR_dof+3:FoR_dof+6, FoR_dof+6:FoR_dof+10] += self.scalingFactor*np.dot(ag.skew(MB_tstep[self.body_number].pos[self.node_number,:]), ag.der_CquatT_by_v(MB_tstep[self.body_number].quat,Lambda_dot[ieq:ieq + num_LM_eq_specific]))

            LM_K[FoR_dof+3:FoR_dof+6, node_dof:node_dof+3] -= self.scalingFactor*ag.skew(np.dot(MB_tstep[self.body_number].cga().T, Lambda_dot[ieq:ieq + num_LM_eq_specific]))

        LM_Q[:sys_size] += self.scalingFactor * np.dot(np.transpose(Bnh), Lambda_dot[ieq:ieq + num_LM_eq_specific])
        LM_Q[sys_size + ieq:sys_size + ieq + num_LM_eq_specific] += self.scalingFactor*(np.dot( MB_tstep[self.body_number].cga(), (
                MB_tstep[self.body_number].for_vel[0:3] +
                np.dot(ag.skew(MB_tstep[self.body_number].for_vel[3:6]), MB_tstep[self.body_number].pos[self.node_number,:]) +
                MB_tstep[self.body_number].pos_dot[self.node_number,:])) -
                current_vel)

        ieq += 3
        return

    def staticpost(self, lc_list, MB_beam, MB_tstep):
        return

    def dynamicpost(self, lc_list, MB_beam, MB_tstep):
        return


################################################################################
# Funtions to interact with this Library
################################################################################
def initialize_constraints(MBdict):

    index_eq = 0
    num_constraints = MBdict['num_constraints']
    lc_list = list()

    # Read the dictionary and create the constraints
    for iconstraint in range(num_constraints):
        lc_list.append(lc_from_string(MBdict["constraint_%02d" % iconstraint]['behaviour'])())
        MBdict_entry = MBdict["constraint_%02d" % iconstraint]
        if "penaltyFactor" in MBdict_entry.keys():
            if not MBdict_entry['penaltyFactor'] == 0.:
                raise NotImplementedError("Penalty method not completely implemented for Lagrange Constraints")
        index_eq = lc_list[-1].initialise(MBdict_entry, index_eq)

    return lc_list

def define_num_LM_eq(lc_list):
    """
    define_num_LM_eq

    Define the number of equations needed to define the boundary boundary conditions

    Args:
        lc_list(): list of all the defined contraints
    Returns:
        num_LM_eq(int): number of new equations needed to define the boundary boundary conditions

    Examples:
        num_LM_eq = lagrangeconstraints.define_num_LM_eq(lc_list)

    Notes:

    """

    num_LM_eq = 0

    # Compute the number of equations
    for lc in lc_list:
        num_LM_eq += lc.get_n_eq()

    return num_LM_eq


def generate_lagrange_matrix(lc_list, MB_beam, MB_tstep, ts, num_LM_eq, sys_size, dt, Lambda, Lambda_dot, dynamic_or_static):
    """
    generate_lagrange_matrix

    Generates the matrices associated to the Lagrange multipliers boundary conditions

    Args:
        lc_list(): list of all the defined contraints
        MBdict(dict): dictionary with the MultiBody and LagrangeMultipliers information
        MB_beam(list): list of 'beams' of each of the bodies that form the system
        MB_tstep(list): list of 'StructTimeStepInfo' of each of the bodies that form the system
        num_LM_eq(int): number of new equations needed to define the boundary boundary conditions
        sys_size(int): total number of degrees of freedom of the multibody system
        dt(float): time step
        Lambda(np.ndarray): list of Lagrange multipliers values
        Lambda_dot(np.ndarray): list of the first derivative of the Lagrange multipliers values
        dynamic_or_static (str): string defining if the computation is dynamic or static

    Returns:
        LM_C (np.ndarray): Damping matrix associated to the Lagrange Multipliers equations
        LM_K (np.ndarray): Stiffness matrix associated to the Lagrange Multipliers equations
        LM_Q (np.ndarray): Vector of independent terms associated to the Lagrange Multipliers equations
    """
    # Initialize matrices
    LM_C = np.zeros((sys_size + num_LM_eq,sys_size + num_LM_eq), dtype=ct.c_double, order = 'F')
    LM_K = np.zeros((sys_size + num_LM_eq,sys_size + num_LM_eq), dtype=ct.c_double, order = 'F')
    LM_Q = np.zeros((sys_size + num_LM_eq,),dtype=ct.c_double, order = 'F')

    # Define the matrices associated to the constratints
    # TODO: Is there a better way to deal with ieq?
    # ieq = 0
    for lc in lc_list:
        if dynamic_or_static.lower() == "static":
            lc.staticmat(LM_C=LM_C,
                        LM_K=LM_K,
                        LM_Q=LM_Q,
                        MB_beam=MB_beam,
                        MB_tstep=MB_tstep,
                        ts=ts,
                        num_LM_eq=num_LM_eq,
                        sys_size=sys_size,
                        dt=dt,
                        Lambda=Lambda,
                        Lambda_dot=Lambda_dot)

        elif dynamic_or_static.lower() == "dynamic":
            lc.dynamicmat(LM_C=LM_C,
                        LM_K=LM_K,
                        LM_Q=LM_Q,
                        MB_beam=MB_beam,
                        MB_tstep=MB_tstep,
                        ts=ts,
                        num_LM_eq=num_LM_eq,
                        sys_size=sys_size,
                        dt=dt,
                        Lambda=Lambda,
                        Lambda_dot=Lambda_dot)

    return LM_C, LM_K, LM_Q


def postprocess(lc_list, MB_beam, MB_tstep, dynamic_or_static):
    """
    Run the postprocess of all the Lagrange Constraints in the system
    """
    for lc in lc_list:
        if dynamic_or_static.lower() == "static":
            lc.staticpost(lc_list = lc_list,
                           MB_beam = MB_beam,
                           MB_tstep = MB_tstep)

        elif dynamic_or_static.lower() == "dynamic":
            lc.dynamicpost(lc_list = lc_list,
                           MB_beam = MB_beam,
                           MB_tstep = MB_tstep)

    return


def remove_constraint(MBdict, constraint):
    """
    Removes a constraint from the list.
    This function is thought to release constraints at some point during
    a dynamic simulation
    """
    try:
        del(MBdict[constraint])
        MBdict['num_constraints'] -= 1
    except KeyError:
        # The entry did not exist in the dict, pass without substracting 1 to
        # num_constraints
        pass


################################################################################
################################################################################
################################################################################
print_available_lc()
