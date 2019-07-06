"""
LagrangeConstraints library

Library used to create the matrices associate to boundary conditions through
the method of Lagrange Multipliers

Args:

Returns:

Examples:
    To use this library: import sharpy.structure.utils.lagrangeconstraints as lagrangeconstraints

Notes:

"""
from abc import ABCMeta, abstractmethod
import sharpy.utils.cout_utils as cout
import os
import ctypes as ct
import numpy as np
import sharpy.utils.algebra as algebra

dict_of_lc = {}
lc = {}  # for internal working


# decorator
def lagrangeconstraint(arg):
    # global available_solvers
    global dict_of_lc
    try:
        arg._lc_id
    except AttributeError:
        raise AttributeError('Class defined as lagrange constraint has no _lc_id attribute')
    dict_of_lc[arg._lc_id] = arg
    return arg

def print_available_lc():
    cout.cout_wrap('The available lagrange constraints on this session are:', 2)
    for name, i_lc in dict_of_lc.items():
        cout.cout_wrap('%s ' % i_lc._lc_id, 2)

def lc_from_string(string):
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
    if print_info:
        cout.cout_wrap('Generating an instance of %s' % lc_name, 2)
    cls_type = lc_from_string(lc_name)
    lc = cls_type()
    return lc


class BaseLagrangeConstraint(metaclass=ABCMeta):
    def __init__(self):
        self._n_eq = None
        self._ieq = None

    @abstractmethod
    def get_n_eq(self):
        pass

    @abstractmethod
    #  def initialise(self, **kwargs):
    def initialise(self, MBdict_entry, ieq):
        pass

    @abstractmethod
    # def staticmat(self, **kwargs):
    def staticmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                  sys_size, dt, Lambda, Lambda_dot,
                  scalingFactor, penaltyFactor):
        pass

    @abstractmethod
    # def dynamicmat(self, **kwargs):
    def dynamicmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                   sys_size, dt, Lambda, Lambda_dot,
                  scalingFactor, penaltyFactor):
        pass

    @abstractmethod
    # def staticpost(self, **kwargs):
    def staticpost(self, lc_list, MB_beam, MB_tstep):
        pass

    @abstractmethod
    # def dynamicpost(self, **kwargs):
    def dynamicpost(self, lc_list, MB_beam, MB_tstep):
        pass


################################################################################
# Auxiliar functions
################################################################################
def define_node_dof(MB_beam, node_body, num_node):
    """
    define_node_dof

    Define the position of the first degree of freedom associated to a certain node

    Args:
        MB_beam(list): list of 'Beam'
        node_body(int): body to which the node belongs
        num_node(int): number os the node within the body

    Returns:
        node_dof(int): first degree of freedom associated to the node

    Examples:

    Notes:

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
        MB_beam(list): list of 'Beam'
        node_body(int): body to which the node belongs
        num_node(int): number os the node within the body

    Returns:
        node_dof(int): first degree of freedom associated to the node

    Examples:

    Notes:

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
def equal_lin_vel_node_FoR(MB_tstep, MB_beam, FoR_body, node_body, node_number, node_FoR_dof, node_dof, FoR_dof, sys_size, Lambda_dot, scalingFactor, penaltyFactor, ieq, LM_K, LM_C, LM_Q):

    # Variables names. The naming of the variables can be quite confusing. The reader should think that
    # the BC relates one "node" and one "FoR" (writen between quotes in these lines).
    # If a variable is related to one of them starts with "node_" or "FoR_" respectively
    # node_number: number of the "node" within its own body
    # node_body: body number of the "node"
    # node_FoR_dof: position of the first degree of freedom of the FoR to which the "node" belongs
    # node_dof: position of the first degree of freedom associated to the "node"
    # FoR_body: body number of the "FoR"
    # FoR_dof: position of the first degree of freedom associated to the "FoR"

    num_LM_eq_specific = 3
    Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')
    B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')

    Bnh[:, FoR_dof:FoR_dof+3] = algebra.quat2rotation(MB_tstep[FoR_body].quat)

    Bnh[:, node_dof:node_dof+3] = -1.0*algebra.quat2rotation(MB_tstep[node_body].quat)
    if MB_beam[node_body].FoR_movement == 'free':
        Bnh[:, node_FoR_dof:node_FoR_dof+3] = -1.0*algebra.quat2rotation(MB_tstep[node_body].quat)
        Bnh[:, node_FoR_dof+3:node_FoR_dof+6] = 1.0*np.dot(algebra.quat2rotation(MB_tstep[node_body].quat),algebra.skew(MB_tstep[node_body].pos[node_number,:]))

    LM_C[sys_size+ieq:sys_size+ieq+num_LM_eq_specific,:sys_size] += scalingFactor*Bnh
    LM_C[:sys_size,sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += scalingFactor*np.transpose(Bnh)

    LM_Q[:sys_size] += scalingFactor*np.dot(np.transpose(Bnh),Lambda_dot[ieq:ieq+num_LM_eq_specific])
    LM_Q[sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += (np.dot(algebra.quat2rotation(MB_tstep[FoR_body].quat),MB_tstep[FoR_body].for_vel[0:3]) +
                                                          -1.0*np.dot(algebra.quat2rotation(MB_tstep[node_body].quat),
                                                                      MB_tstep[node_body].pos_dot[node_number,:] +
                                                                      MB_tstep[node_body].for_vel[0:3] +
                                                                      -1.0*np.dot(algebra.skew(MB_tstep[node_body].pos[node_number,:]),MB_tstep[node_body].for_vel[3:6])))

    LM_C[FoR_dof:FoR_dof+3,FoR_dof+6:FoR_dof+10] += algebra.der_CquatT_by_v(MB_tstep[FoR_body].quat,scalingFactor*Lambda_dot[ieq:ieq+num_LM_eq_specific])

    if MB_beam[node_body].FoR_movement == 'free':
        LM_C[node_dof:node_dof+3,node_FoR_dof+6:node_FoR_dof+10] -= algebra.der_CquatT_by_v(MB_tstep[node_body].quat,scalingFactor*Lambda_dot[ieq:ieq+num_LM_eq_specific])

        LM_C[node_FoR_dof:node_FoR_dof+3,node_FoR_dof+6:node_FoR_dof+10] -= algebra.der_CquatT_by_v(MB_tstep[node_body].quat,scalingFactor*Lambda_dot[ieq:ieq+num_LM_eq_specific])

        LM_C[node_FoR_dof+3:node_FoR_dof+6,node_FoR_dof+6:node_FoR_dof+10] -= np.dot(algebra.skew(MB_tstep[node_body].pos[node_number,:]),
                                                                                     algebra.der_CquatT_by_v(MB_tstep[node_body].quat,
                                                                                                             scalingFactor*Lambda_dot[ieq:ieq+num_LM_eq_specific]))

        LM_K[node_FoR_dof+3:node_FoR_dof+6,node_dof:node_dof+3] += algebra.skew(np.dot(algebra.quat2rotation(MB_tstep[node_body].quat).T,Lambda_dot[ieq:ieq+num_LM_eq_specific]))

    ieq += 3
    return ieq


def def_rot_axis_FoR_wrt_node(MB_tstep, MB_beam, FoR_body, node_body, node_number, node_FoR_dof, node_dof, FoR_dof, sys_size, Lambda_dot, rot_axisB, scalingFactor, penaltyFactor, ieq, LM_K, LM_C, LM_Q, indep):

    # Variables names. The naming of the variables can be quite confusing. The reader should think that
    # the BC relates one "node" and one "FoR" (writen between quotes in these lines).
    # If a variable is related to one of them starts with "node_" or "FoR_" respectively
    # node_number: number of the "node" within its own body
    # node_body: body number of the "node"
    # node_FoR_dof: position of the first degree of freedom of the FoR to which the "node" belongs
    # node_dof: position of the first degree of freedom associated to the "node"
    # FoR_body: body number of the "FoR"
    # FoR_dof: position of the first degree of freedom associated to the "FoR"

    ielem, inode_in_elem = MB_beam[node_body].node_master_elem[node_number]

    if not indep:
        aux_Bnh = algebra.multiply_matrices(algebra.skew(rot_axisB),
                                  algebra.crv2rotation(MB_tstep[node_body].psi[ielem,inode_in_elem,:]).T,
                                  algebra.quat2rotation(MB_tstep[node_body].quat).T,
                                  algebra.quat2rotation(MB_tstep[FoR_body].quat))

        # indep = None
        n0 = np.linalg.norm(aux_Bnh[0,:])
        n1 = np.linalg.norm(aux_Bnh[1,:])
        n2 = np.linalg.norm(aux_Bnh[2,:])
        if ((n0 < n1) and (n0 < n2)):
            # indep = np.array([1,2], dtype = int)
            indep[:] = [1, 2]
            # new_Lambda_dot = np.array([0., Lambda_dot[ieq], Lambda_dot[ieq+1]])
        elif ((n1 < n0) and (n1 < n2)):
            # indep = np.array([0,2], dtype = int)
            indep[:] = [0, 2]
            # new_Lambda_dot = np.array([Lambda_dot[ieq], 0.0, Lambda_dot[ieq+1]])
        elif ((n2 < n0) and (n2 < n1)):
            # indep = np.array([0,1], dtype = int)
            indep[:] = [0, 1]
            # new_Lambda_dot = np.array([Lambda_dot[ieq], Lambda_dot[ieq+1], 0.0])

    new_Lambda_dot = np.zeros(3)
    new_Lambda_dot[indep[0]] = Lambda_dot[ieq]
    new_Lambda_dot[indep[1]] = Lambda_dot[ieq+1]

    num_LM_eq_specific = 2
    Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')
    B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')

    # Lambda_dot[ieq:ieq+num_LM_eq_specific]
    # np.concatenate((Lambda_dot[ieq:ieq+num_LM_eq_specific], np.array([0.])))

    # print(indep)
    Bnh[:, FoR_dof+3:FoR_dof+6] = algebra.multiply_matrices(algebra.skew(rot_axisB),
                                                  algebra.crv2rotation(MB_tstep[node_body].psi[ielem,inode_in_elem,:]).T,
                                                  algebra.quat2rotation(MB_tstep[node_body].quat).T,
                                                  algebra.quat2rotation(MB_tstep[FoR_body].quat))[indep,:]

    # Constrain angular velocities
    LM_Q[:sys_size] += scalingFactor*np.dot(np.transpose(Bnh),Lambda_dot[ieq:ieq+num_LM_eq_specific])
    LM_Q[sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += algebra.multiply_matrices(algebra.skew(rot_axisB),
                                                  algebra.crv2rotation(MB_tstep[node_body].psi[ielem,inode_in_elem,:]).T,
                                                  algebra.quat2rotation(MB_tstep[node_body].quat).T,
                                                  algebra.quat2rotation(MB_tstep[FoR_body].quat),
                                                  MB_tstep[FoR_body].for_vel[3:6])[indep]

    LM_C[sys_size+ieq:sys_size+ieq+num_LM_eq_specific,:sys_size] += scalingFactor*Bnh
    LM_C[:sys_size,sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += scalingFactor*np.transpose(Bnh)

    if MB_beam[node_body].FoR_movement == 'free':
        LM_C[FoR_dof+3:FoR_dof+6,node_FoR_dof+6:node_FoR_dof+10] += np.dot(algebra.quat2rotation(MB_tstep[FoR_body].quat).T,
                                                                           algebra.der_Cquat_by_v(MB_tstep[node_body].quat,
                                                                                                  algebra.multiply_matrices(algebra.crv2rotation(MB_tstep[node_body].psi[ielem,inode_in_elem,:]),
                                                                                                                    algebra.skew(rot_axisB).T,
                                                                                                                    new_Lambda_dot)))

    LM_C[FoR_dof+3:FoR_dof+6,FoR_dof+6:FoR_dof+10] += algebra.der_CquatT_by_v(MB_tstep[FoR_body].quat,
                                                                              algebra.multiply_matrices(algebra.quat2rotation(MB_tstep[node_body].quat),
                                                                                                algebra.crv2rotation(MB_tstep[node_body].psi[ielem,inode_in_elem,:]).T,
                                                                                                algebra.skew(rot_axisB).T,
                                                                                                new_Lambda_dot))

    LM_K[FoR_dof+3:FoR_dof+6,node_dof+3:node_dof+6] += algebra.multiply_matrices(algebra.quat2rotation(MB_tstep[FoR_body].quat).T,
                                                                         algebra.quat2rotation(MB_tstep[node_body].quat),
                                                                         algebra.der_Ccrv_by_v(MB_tstep[node_body].psi[ielem,inode_in_elem,:],
                                                                                                np.dot(algebra.skew(rot_axisB).T,
                                                                                                       new_Lambda_dot)))

    ieq += 2
    return ieq


def def_rot_vel_FoR_wrt_node(MB_tstep, MB_beam, FoR_body, node_body, node_number, node_FoR_dof, node_dof, FoR_dof, sys_size, Lambda_dot, rot_axisB, rot_vel, scalingFactor, penaltyFactor, ieq, LM_K, LM_C, LM_Q):

    # Variables names. The naming of the variables can be quite confusing. The reader should think that
    # the BC relates one "node" and one "FoR" (writen between quotes in these lines).
    # If a variable is related to one of them starts with "node_" or "FoR_" respectively
    # node_number: number of the "node" within its own body
    # node_body: body number of the "node"
    # node_FoR_dof: position of the first degree of freedom of the FoR to which the "node" belongs
    # node_dof: position of the first degree of freedom associated to the "node"
    # FoR_body: body number of the "FoR"
    # FoR_dof: position of the first degree of freedom associated to the "FoR"

    num_LM_eq_specific = 1
    Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')
    B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')

    # Lambda_dot[ieq:ieq+num_LM_eq_specific]
    # np.concatenate((Lambda_dot[ieq:ieq+num_LM_eq_specific], np.array([0.])))

    ielem, inode_in_elem = MB_beam[node_body].node_master_elem[node_number]
    Bnh[:, FoR_dof+3:FoR_dof+6] = algebra.multiply_matrices(rot_axisB,
                                                  algebra.crv2rotation(MB_tstep[node_body].psi[ielem,inode_in_elem,:]).T,
                                                  algebra.quat2rotation(MB_tstep[node_body].quat).T,
                                                  algebra.quat2rotation(MB_tstep[FoR_body].quat))

    # Constrain angular velocities
    LM_Q[:sys_size] += scalingFactor*np.dot(np.transpose(Bnh),Lambda_dot[ieq:ieq+num_LM_eq_specific])
    LM_Q[sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += algebra.multiply_matrices(rot_axisB,
                                                  algebra.crv2rotation(MB_tstep[node_body].psi[ielem,inode_in_elem,:]).T,
                                                  algebra.quat2rotation(MB_tstep[node_body].quat).T,
                                                  algebra.quat2rotation(MB_tstep[FoR_body].quat),
                                                  MB_tstep[FoR_body].for_vel[3:6]) - rot_vel

    LM_C[sys_size+ieq:sys_size+ieq+num_LM_eq_specific,:sys_size] += scalingFactor*Bnh
    LM_C[:sys_size,sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += scalingFactor*np.transpose(Bnh)

    if MB_beam[node_body].FoR_movement == 'free':
        LM_C[FoR_dof+3:FoR_dof+6,node_FoR_dof+6:node_FoR_dof+10] += np.dot(algebra.quat2rotation(MB_tstep[FoR_body].quat).T,
                                                                           algebra.der_Cquat_by_v(MB_tstep[node_body].quat,
                                                                                                  algebra.multiply_matrices(algebra.crv2rotation(MB_tstep[node_body].psi[ielem,inode_in_elem,:]),
                                                                                                                    # rot_axisB.T,
                                                                                                                    rot_axisB.T*Lambda_dot[ieq:ieq+num_LM_eq_specific])))

    LM_C[FoR_dof+3:FoR_dof+6,FoR_dof+6:FoR_dof+10] += algebra.der_CquatT_by_v(MB_tstep[FoR_body].quat,
                                                                              algebra.multiply_matrices(algebra.quat2rotation(MB_tstep[node_body].quat),
                                                                                                algebra.crv2rotation(MB_tstep[node_body].psi[ielem,inode_in_elem,:]).T,
                                                                                                rot_axisB.T*Lambda_dot[ieq:ieq+num_LM_eq_specific]))

    LM_K[FoR_dof+3:FoR_dof+6,node_dof+3:node_dof+6] += algebra.multiply_matrices(algebra.quat2rotation(MB_tstep[FoR_body].quat).T,
                                                                         algebra.quat2rotation(MB_tstep[node_body].quat),
                                                                         algebra.der_Ccrv_by_v(MB_tstep[node_body].psi[ielem,inode_in_elem,:],
                                                                                                rot_axisB.T*Lambda_dot[ieq:ieq+num_LM_eq_specific]))

    ieq += 1
    return ieq

################################################################################
# Lagrange constraints
################################################################################
@lagrangeconstraint
class SampleLagrange(BaseLagrangeConstraint):
    _lc_id = 'SampleLagrange'

    def __init__(self):
        self._n_eq = 3

    def get_n_eq(self):

        return self._n_eq

    def initialise(self, MBdict_entry, ieq):
        print('Type of LC: ', self._lc_id)
        print('Arguments and values:')
        for k, v in MBdict_entry.items():
            print(k, v)
        self._ieq = ieq

        return self._ieq + self._n_eq

    def staticmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot,
                scalingFactor, penaltyFactor):
        return np.zeros((6, 6))

    def dynamicmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot,
                scalingFactor, penaltyFactor):
        return np.zeros((10, 10))

    def staticpost(self, lc_list, MB_beam, MB_tstep):
        return

    def dynamicpost(self, lc_list, MB_beam, MB_tstep):
        return


@lagrangeconstraint
class hinge_node_FoR(BaseLagrangeConstraint):
    _lc_id = 'hinge_node_FoR'

    def __init__(self):
        self._n_eq = 5

    def get_n_eq(self):
        return self._n_eq

    def initialise(self, MBdict_entry, ieq):
        print('Type of LC: ', self._lc_id)
        print('Arguments and values:')
        for k, v in MBdict_entry.items():
            print(k, v)

        self.node_number = MBdict_entry['node_in_body']
        self.node_body = MBdict_entry['body']
        self.FoR_body = MBdict_entry['body_FoR']
        self.rot_axisB = MBdict_entry['rot_axisB']
        self._ieq = ieq
        self.indep = []

        return self._ieq + self._n_eq

    def staticmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot,
                scalingFactor, penaltyFactor):
        return

    def dynamicmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot,
                scalingFactor, penaltyFactor):

        # Define the position of the first degree of freedom associated to the node
        node_dof = define_node_dof(MB_beam, self.node_body, self.node_number)
        node_FoR_dof = define_FoR_dof(MB_beam, self.node_body)
        FoR_dof = define_FoR_dof(MB_beam, self.FoR_body)
        ieq = self._ieq

        # Define the equations
        ieq = equal_lin_vel_node_FoR(MB_tstep, MB_beam, self.FoR_body, self.node_body, self.node_number, node_FoR_dof, node_dof, FoR_dof, sys_size, Lambda_dot, scalingFactor, penaltyFactor, ieq, LM_K, LM_C, LM_Q)
        ieq = def_rot_axis_FoR_wrt_node(MB_tstep, MB_beam, self.FoR_body, self.node_body, self.node_number, node_FoR_dof, node_dof, FoR_dof, sys_size, Lambda_dot, self.rot_axisB, scalingFactor, penaltyFactor, ieq, LM_K, LM_C, LM_Q, self.indep)

        return

    def staticpost(self, lc_list, MB_beam, MB_tstep):
        return

    def dynamicpost(self, lc_list, MB_beam, MB_tstep):
        MB_tstep[self.FoR_body].for_pos[0:3] = np.dot(algebra.quat2rotation(MB_tstep[self.node_body].quat), MB_tstep[self.node_body].pos[self.node_number,:]) + MB_tstep[self.node_body].for_pos[0:3]
        return


@lagrangeconstraint
class hinge_node_FoR_constant_vel(BaseLagrangeConstraint):
    _lc_id = 'hinge_node_FoR_constant_vel'

    def __init__(self):
        self._n_eq = 6

    def get_n_eq(self):
        return self._n_eq

    def initialise(self, MBdict_entry, ieq):
        print('Type of LC: ', self._lc_id)
        print('Arguments and values:')
        for k, v in MBdict_entry.items():
            print(k, v)

        self.node_number = MBdict_entry['node_in_body']
        self.node_body = MBdict_entry['body']
        self.FoR_body = MBdict_entry['body_FoR']
        self.rot_axisB = MBdict_entry['rot_axisB']
        self.rot_vel = MBdict_entry['rot_vel']
        self._ieq = ieq
        self.indep = []

        return self._ieq + self._n_eq

    def staticmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot,
                scalingFactor, penaltyFactor):
        return

    def dynamicmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot,
                scalingFactor, penaltyFactor):

        # Define the position of the first degree of freedom associated to the node
        node_dof = define_node_dof(MB_beam, self.node_body, self.node_number)
        node_FoR_dof = define_FoR_dof(MB_beam, self.node_body)
        FoR_dof = define_FoR_dof(MB_beam, self.FoR_body)
        ieq = self._ieq

        # Define the equations
        ieq = equal_lin_vel_node_FoR(MB_tstep, MB_beam, self.FoR_body, self.node_body, self.node_number, node_FoR_dof, node_dof, FoR_dof, sys_size, Lambda_dot, scalingFactor, penaltyFactor, ieq, LM_K, LM_C, LM_Q)
        ieq = def_rot_axis_FoR_wrt_node(MB_tstep, MB_beam, self.FoR_body, self.node_body, self.node_number, node_FoR_dof, node_dof, FoR_dof, sys_size, Lambda_dot, self.rot_axisB, scalingFactor, penaltyFactor, ieq, LM_K, LM_C, LM_Q, self.indep)
        ieq = def_rot_vel_FoR_wrt_node(MB_tstep, MB_beam, self.FoR_body, self.node_body, self.node_number, node_FoR_dof, node_dof, FoR_dof, sys_size, Lambda_dot, self.rot_axisB, self.rot_vel, scalingFactor, penaltyFactor, ieq, LM_K, LM_C, LM_Q)
        return

    def staticpost(self, lc_list, MB_beam, MB_tstep):
        return

    def dynamicpost(self, lc_list, MB_beam, MB_tstep):
        MB_tstep[self.FoR_body].for_pos[0:3] = np.dot(algebra.quat2rotation(MB_tstep[self.node_body].quat), MB_tstep[self.node_body].pos[self.node_number,:]) + MB_tstep[self.node_body].for_pos[0:3]
        return


@lagrangeconstraint
class spherical_node_FoR(BaseLagrangeConstraint):
    _lc_id = 'spherical_node_FoR'

    def __init__(self):
        self._n_eq = 3

    def get_n_eq(self):
        return self._n_eq

    def initialise(self, MBdict_entry, ieq):
        print('Type of LC: ', self._lc_id)
        print('Arguments and values:')
        for k, v in MBdict_entry.items():
            print(k, v)

        self.node_number = MBdict_entry['node_in_body']
        self.node_body = MBdict_entry['body']
        self.FoR_body = MBdict_entry['body_FoR']
        self._ieq = ieq

        return self._ieq + self._n_eq

    def staticmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot,
                scalingFactor, penaltyFactor):
        return

    def dynamicmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot,
                scalingFactor, penaltyFactor):

        # Define the position of the first degree of freedom associated to the node
        node_dof = define_node_dof(MB_beam, self.node_body, self.node_number)
        node_FoR_dof = define_FoR_dof(MB_beam, self.node_body)
        FoR_dof = define_FoR_dof(MB_beam, self.FoR_body)
        ieq = self._ieq

        # Define the equations
        ieq = equal_lin_vel_node_FoR(MB_tstep, MB_beam, self.FoR_body, self.node_body, self.node_number, node_FoR_dof, node_dof, FoR_dof, sys_size, Lambda_dot, scalingFactor, penaltyFactor, ieq, LM_K, LM_C, LM_Q)

        return

    def staticpost(self, lc_list, MB_beam, MB_tstep):
        return

    def dynamicpost(self, lc_list, MB_beam, MB_tstep):
        MB_tstep[self.FoR_body].for_pos[0:3] = np.dot(algebra.quat2rotation(MB_tstep[self.node_body].quat), MB_tstep[self.node_body].pos[self.node_number,:]) + MB_tstep[self.node_body].for_pos[0:3]
        return


@lagrangeconstraint
class free(BaseLagrangeConstraint):
    _lc_id = 'free'

    def __init__(self):
        self._n_eq = 0

    def get_n_eq(self):
        return self._n_eq

    def initialise(self, MBdict_entry, ieq):
        print('Type of LC: ', self._lc_id)
        print('Arguments and values:')
        for k, v in MBdict_entry.items():
            print(k, v)

        self._ieq = ieq
        return self._ieq + self._n_eq

    def staticmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot,
                scalingFactor, penaltyFactor):
        return

    def dynamicmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot,
                scalingFactor, penaltyFactor):
        return

    def staticpost(self, lc_list, MB_beam, MB_tstep):
        return

    def dynamicpost(self, lc_list, MB_beam, MB_tstep):
        return


@lagrangeconstraint
class spherical_FoR(BaseLagrangeConstraint):
    _lc_id = 'spherical_FoR'

    def __init__(self):
        self._n_eq = 3

    def get_n_eq(self):
        return self._n_eq

    def initialise(self, MBdict_entry, ieq):
        print('Type of LC: ', self._lc_id)
        print('Arguments and values:')
        for k, v in MBdict_entry.items():
            print(k, v)

        self.body_FoR = MBdict_entry['body_FoR']
        self._ieq = ieq

        return self._ieq + self._n_eq

    def staticmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot,
                scalingFactor, penaltyFactor):
        return

    def dynamicmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot,
                scalingFactor, penaltyFactor):
        num_LM_eq_specific = self._n_eq
        Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')
        B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')

        # Define the position of the first degree of freedom associated to the FoR
        FoR_dof = define_FoR_dof(MB_beam, self.body_FoR)
        ieq = self._ieq

        Bnh[:3, FoR_dof:FoR_dof+3] = 1.0*np.eye(3)

        LM_C[sys_size+ieq:sys_size+ieq+num_LM_eq_specific,:sys_size] += scalingFactor*Bnh
        LM_C[:sys_size,sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += scalingFactor*np.transpose(Bnh)

        LM_Q[:sys_size] += scalingFactor*np.dot(np.transpose(Bnh),Lambda_dot[ieq:ieq+num_LM_eq_specific])

        LM_Q[sys_size+ieq:sys_size+ieq+3] += MB_tstep[self.body_FoR].for_vel[0:3].astype(dtype=ct.c_double, copy=True, order='F')

        ieq += 3
        return

    def staticpost(self, lc_list, MB_beam, MB_tstep):
        return

    def dynamicpost(self, lc_list, MB_beam, MB_tstep):
        return


@lagrangeconstraint
class hinge_FoR(BaseLagrangeConstraint):
    _lc_id = 'hinge_FoR'

    def __init__(self):
        self._n_eq = 5

    def get_n_eq(self):
        return self._n_eq

    def initialise(self, MBdict_entry, ieq):
        print('Type of LC: ', self._lc_id)
        print('Arguments and values:')
        for k, v in MBdict_entry.items():
            print(k, v)

        self.body_FoR = MBdict_entry['body_FoR']
        self.rot_axis = MBdict_entry['rot_axis_AFoR']
        self._ieq = ieq

        return self._ieq + self._n_eq

    def staticmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot,
                scalingFactor, penaltyFactor):
        return

    def dynamicmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot,
                scalingFactor, penaltyFactor):
        num_LM_eq_specific = self._n_eq
        Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')
        B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')

        # Define the position of the first degree of freedom associated to the FoR
        FoR_dof = define_FoR_dof(MB_beam, self.body_FoR)
        ieq = self._ieq

        Bnh[:3, FoR_dof:FoR_dof+3] = 1.0*np.eye(3)

        # Only two of these equations are linearly independent
        skew_rot_axis = algebra.skew(self.rot_axis)
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

        LM_C[sys_size+ieq:sys_size+ieq+num_LM_eq_specific,:sys_size] += scalingFactor*Bnh
        LM_C[:sys_size,sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += scalingFactor*np.transpose(Bnh)

        LM_Q[:sys_size] += scalingFactor*np.dot(np.transpose(Bnh),Lambda_dot[ieq:ieq+num_LM_eq_specific])

        LM_Q[sys_size+ieq:sys_size+ieq+3] += MB_tstep[self.body_FoR].for_vel[0:3].astype(dtype=ct.c_double, copy=True, order='F')
        LM_Q[sys_size+ieq+3:sys_size+ieq+5] += np.dot(skew_rot_axis[[row0,row1],:], MB_tstep[self.body_FoR].for_vel[3:6])

        ieq += 5
        return

    def staticpost(self, lc_list, MB_beam, MB_tstep):
        return

    def dynamicpost(self, lc_list, MB_beam, MB_tstep):
        return


@lagrangeconstraint
class hinge_FoR_wrtG(BaseLagrangeConstraint):
    _lc_id = 'hinge_FoR_wrtG'

    def __init__(self):
        self._n_eq = 5

    def get_n_eq(self):
        return self._n_eq

    def initialise(self, MBdict_entry, ieq):
        print('Type of LC: ', self._lc_id)
        print('Arguments and values:')
        for k, v in MBdict_entry.items():
            print(k, v)

        self.body_FoR = MBdict_entry['body_FoR']
        self.rot_axis = MBdict_entry['rot_axis_AFoR']
        self._ieq = ieq

        return self._ieq + self._n_eq

    def staticmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot,
                scalingFactor, penaltyFactor):
        return

    def dynamicmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot,
                scalingFactor, penaltyFactor):
        num_LM_eq_specific = self._n_eq
        Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')
        B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')

        # Define the position of the first degree of freedom associated to the FoR
        FoR_dof = define_FoR_dof(MB_beam, self.body_FoR)
        ieq = self._ieq

        Bnh[:3, FoR_dof:FoR_dof+3] = algebra.quat2rotation(MB_tstep[self.body_FoR].quat)

        # Only two of these equations are linearly independent
        skew_rot_axis = algebra.skew(self.rot_axis)
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

        LM_C[sys_size+ieq:sys_size+ieq+num_LM_eq_specific,:sys_size] += scalingFactor*Bnh
        LM_C[:sys_size,sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += scalingFactor*np.transpose(Bnh)

        LM_C[FoR_dof:FoR_dof+3,FoR_dof+6:FoR_dof+10] += algebra.der_CquatT_by_v(MB_tstep[self.body_FoR].quat,Lambda_dot[ieq:ieq+3])

        LM_Q[:sys_size] += scalingFactor*np.dot(np.transpose(Bnh),Lambda_dot[ieq:ieq+num_LM_eq_specific])

        LM_Q[sys_size+ieq:sys_size+ieq+3] += np.dot(algebra.quat2rotation(MB_tstep[self.body_FoR].quat),MB_tstep[self.body_FoR].for_vel[0:3])
        LM_Q[sys_size+ieq+3:sys_size+ieq+5] += np.dot(skew_rot_axis[[row0,row1],:], MB_tstep[self.body_FoR].for_vel[3:6])

        ieq += 5
        return

    def staticpost(self, lc_list, MB_beam, MB_tstep):
        return

    def dynamicpost(self, lc_list, MB_beam, MB_tstep):
        return


@lagrangeconstraint
class fully_constrained_node_FoR(BaseLagrangeConstraint):
    _lc_id = 'fully_constrained_node_FoR'

    def __init__(self):
        self._n_eq = 6

    def get_n_eq(self):
        return self._n_eq

    def initialise(self, MBdict_entry, ieq):
        print('Type of LC: ', self._lc_id)
        print('Arguments and values:')
        for k, v in MBdict_entry.items():
            print(k, v)

        print("WARNING: do not use fully_constrained_node_FoR. It is outdated")
        self.node_in_body = MBdict_entry['node_in_body']
        self.node_body = MBdict_entry['body']
        self.body_FoR = MBdict_entry['body_FoR']
        self._ieq = ieq
        return self._ieq + self._n_eq

    def staticmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot,
                scalingFactor, penaltyFactor):
        return

    def dynamicmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot,
                scalingFactor, penaltyFactor):
        num_LM_eq_specific = self._n_eq
        Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')
        B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')

        node_dof = define_node_dof(MB_beam, self.node_body, self.node_in_body)
        FoR_dof = define_FoR_dof(MB_beam, self.body_FoR)
        ieq = self._ieq

        # Option with non holonomic constraints
        # BC for linear velocities
        Bnh[:3, node_dof:node_dof+3] = -1.0*np.eye(3)
        quat = algebra.quat_bound(MB_tstep[self.body_FoR].quat)
        Bnh[:3, FoR_dof:FoR_dof+3] = algebra.quat2rotation(quat)

        # BC for angular velocities
        Bnh[3:6,FoR_dof+3:FoR_dof+6] = -1.0*algebra.quat2rotation(quat)
        ielem, inode_in_elem = MB_beam[0].node_master_elem[self.node_in_body]
        Bnh[3:6,node_dof+3:node_dof+6] = algebra.crv2tan(MB_tstep[0].psi[ielem, inode_in_elem, :])

        LM_C[sys_size+ieq:sys_size+ieq+num_LM_eq_specific,:sys_size] += scalingFactor*Bnh
        LM_C[:sys_size,sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += scalingFactor*np.transpose(Bnh)

        LM_Q[:sys_size] += scalingFactor*np.dot(np.transpose(Bnh),Lambda_dot[ieq:ieq+num_LM_eq_specific])
        LM_Q[sys_size+ieq:sys_size+ieq+3] += -MB_tstep[0].pos_dot[-1,:] + np.dot(algebra.quat2rotation(quat),MB_tstep[1].for_vel[0:3])
        LM_Q[sys_size+ieq+3:sys_size+ieq+6] += (np.dot(algebra.crv2tan(MB_tstep[0].psi[ielem, inode_in_elem, :]),MB_tstep[0].psi_dot[ielem, inode_in_elem, :]) -
                                      np.dot(algebra.quat2rotation(quat), MB_tstep[self.body_FoR].for_vel[3:6]))

        #LM_K[FoR_dof:FoR_dof+3,FoR_dof+6:FoR_dof+10] = algebra.der_CquatT_by_v(MB_tstep[body_FoR].quat,Lambda_dot)
        LM_C[FoR_dof:FoR_dof+3,FoR_dof+6:FoR_dof+10] += algebra.der_CquatT_by_v(quat,scalingFactor*Lambda_dot[ieq:ieq+3])
        LM_C[FoR_dof+3:FoR_dof+6,FoR_dof+6:FoR_dof+10] -= algebra.der_CquatT_by_v(quat,scalingFactor*Lambda_dot[ieq+3:ieq+6])

        LM_K[node_dof+3:node_dof+6,node_dof+3:node_dof+6] += algebra.der_TanT_by_xv(MB_tstep[0].psi[ielem, inode_in_elem, :],scalingFactor*Lambda_dot[ieq+3:ieq+6])

        ieq += 6
        return

    def staticpost(self, lc_list, MB_beam, MB_tstep):
        return

    def dynamicpost(self, lc_list, MB_beam, MB_tstep):
        return


# @lagrangeconstraint
# class hinge_node_FoR_constant_rotation(BaseLagrangeConstraint):
#     _lc_id = 'hinge_node_FoR_constant_rotation'
#
#     def __init__(self):
#         self._n_eq = 4
#
#     def get_n_eq(self):
#         return self._n_eq
#
#     def initialise(self, MBdict_entry, ieq):
#         print('Type of LC: ', self._lc_id)
#         print('Arguments and values:')
#         for k, v in MBdict_entry.items():
#             print(k, v)
#
#         self._ieq = ieq
#         return self._ieq + self._n_eq
#
#     def staticmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
#                 sys_size, dt, Lambda, Lambda_dot,
#                 scalingFactor, penaltyFactor):
#         return
#
#     def dynamicmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
#                 sys_size, dt, Lambda, Lambda_dot,
#                 scalingFactor, penaltyFactor):
#         return
#
#     def staticpost(self, lc_list, MB_beam, MB_tstep):
#         return
#
#     def dynamicpost(self, lc_list, MB_beam, MB_tstep):
#         return

@lagrangeconstraint
class constant_rot_vel_FoR(BaseLagrangeConstraint):
    _lc_id = 'constant_rot_vel_FoR'

    def __init__(self):
        self._n_eq = 3

    def get_n_eq(self):
        return self._n_eq

    def initialise(self, MBdict_entry, ieq):
        print('Type of LC: ', self._lc_id)
        print('Arguments and values:')
        for k, v in MBdict_entry.items():
            print(k, v)

        self.rot_vel = MBdict_entry['rot_vel']
        self.FoR_body = MBdict_entry['FoR_body']
        self._ieq = ieq

        return self._ieq + self._n_eq

    def staticmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot,
                scalingFactor, penaltyFactor):
        return

    def dynamicmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot,
                scalingFactor, penaltyFactor):
        num_LM_eq_specific = self._n_eq
        Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')
        B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')

        # Define the position of the first degree of freedom associated to the FoR
        FoR_dof = define_FoR_dof(MB_beam, self.FoR_body)
        ieq = self._ieq

        Bnh[:3,FoR_dof+3:FoR_dof+6] = np.eye(3)

        LM_C[sys_size+ieq:sys_size+ieq+num_LM_eq_specific,:sys_size] += scalingFactor*Bnh
        LM_C[:sys_size,sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += scalingFactor*np.transpose(Bnh)

        LM_Q[:sys_size] += scalingFactor*np.dot(np.transpose(Bnh),Lambda_dot[ieq:ieq+num_LM_eq_specific])
        LM_Q[sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += MB_tstep[self.FoR_body].for_vel[3:6] - self.rot_vel

        ieq += 3
        return

    def staticpost(self, lc_list, MB_beam, MB_tstep):
        return

    def dynamicpost(self, lc_list, MB_beam, MB_tstep):
        return

@lagrangeconstraint
class constant_vel_FoR(BaseLagrangeConstraint):
    _lc_id = 'constant_vel_FoR'

    def __init__(self):
        self._n_eq = 6

    def get_n_eq(self):
        return self._n_eq

    def initialise(self, MBdict_entry, ieq):
        print('Type of LC: ', self._lc_id)
        print('Arguments and values:')
        for k, v in MBdict_entry.items():
            print(k, v)

        self.vel = MBdict_entry['vel']
        self.FoR_body = MBdict_entry['FoR_body']
        self._ieq = ieq

        return self._ieq + self._n_eq

    def staticmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot,
                scalingFactor, penaltyFactor):
        return

    def dynamicmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot,
                scalingFactor, penaltyFactor):
        num_LM_eq_specific = self._n_eq
        Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order='F')
        B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order='F')

        # Define the position of the first degree of freedom associated to the FoR
        FoR_dof = define_FoR_dof(MB_beam, self.FoR_body)
        ieq = self._ieq

        Bnh[:num_LM_eq_specific, FoR_dof:FoR_dof+6] = np.eye(6)

        LM_C[sys_size + ieq:sys_size + ieq + num_LM_eq_specific, :sys_size] += scalingFactor * Bnh
        LM_C[:sys_size, sys_size + ieq:sys_size + ieq + num_LM_eq_specific] += scalingFactor * np.transpose(Bnh)

        LM_Q[:sys_size] += scalingFactor * np.dot(np.transpose(Bnh), Lambda_dot[ieq:ieq + num_LM_eq_specific])
        LM_Q[sys_size + ieq:sys_size + ieq + num_LM_eq_specific] += MB_tstep[self.FoR_body].for_vel - self.vel

        ieq += 6
        return

    def staticpost(self, lc_list, MB_beam, MB_tstep):
        return

    def dynamicpost(self, lc_list, MB_beam, MB_tstep):
        return

@lagrangeconstraint
class lin_vel_node_wrtA(BaseLagrangeConstraint):
    _lc_id = 'lin_vel_node_wrtA'

    def __init__(self):
        self._n_eq = 3

    def get_n_eq(self):
        return self._n_eq

    def initialise(self, MBdict_entry, ieq):
        print('Type of LC: ', self._lc_id)
        print('Arguments and values:')
        for k, v in MBdict_entry.items():
            print(k, v)

        self.vel = MBdict_entry['velocity']
        self.body_number = MBdict_entry['body_number']
        self.node_number = MBdict_entry['node_number']
        self._ieq = ieq

        return self._ieq + self._n_eq

    def staticmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot,
                scalingFactor, penaltyFactor):

        num_LM_eq_specific = self._n_eq
        B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order='F')

        # Define the position of the first degree of freedom associated to the FoR
        # FoR_dof = define_FoR_dof(MB_beam, self.body_number)
        node_dof = define_node_dof(MB_beam, self.body_number, self.node_number)
        ieq = self._ieq

        B[:num_LM_eq_specific, node_dof:node_dof+3] = np.eye(3)

        LM_K[sys_size + ieq:sys_size + ieq + num_LM_eq_specific, :sys_size] += scalingFactor * B
        LM_K[:sys_size, sys_size + ieq:sys_size + ieq + num_LM_eq_specific] += scalingFactor * np.transpose(B)

        LM_Q[:sys_size] += scalingFactor * np.dot(np.transpose(B), Lambda[ieq:ieq + num_LM_eq_specific])
        LM_Q[sys_size + ieq:sys_size + ieq + num_LM_eq_specific] += MB_tstep[self.body_number].pos[self.node_number,:] - MB_beam[self.body_number].ini_info.pos[self.node_number,:]

        ieq += 3

        return

    def dynamicmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot,
                scalingFactor, penaltyFactor):

        if len(self.vel.shape) > 1:
            current_vel = self.vel[ts-1, :]
        else:
            current_vel = self.vel

        num_LM_eq_specific = self._n_eq
        Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order='F')
        B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order='F')

        # Define the position of the first degree of freedom associated to the FoR
        # FoR_dof = define_FoR_dof(MB_beam, self.body_number)
        node_dof = define_node_dof(MB_beam, self.body_number, self.node_number)
        ieq = self._ieq

        Bnh[:num_LM_eq_specific, node_dof:node_dof+3] = np.eye(3)

        LM_C[sys_size + ieq:sys_size + ieq + num_LM_eq_specific, :sys_size] += scalingFactor * Bnh
        LM_C[:sys_size, sys_size + ieq:sys_size + ieq + num_LM_eq_specific] += scalingFactor * np.transpose(Bnh)

        LM_Q[:sys_size] += scalingFactor * np.dot(np.transpose(Bnh), Lambda_dot[ieq:ieq + num_LM_eq_specific])
        LM_Q[sys_size + ieq:sys_size + ieq + num_LM_eq_specific] += MB_tstep[self.body_number].pos_dot[self.node_number,:] - current_vel

        ieq += 3
        return

    def staticpost(self, lc_list, MB_beam, MB_tstep):
        return

    def dynamicpost(self, lc_list, MB_beam, MB_tstep):
        return

@lagrangeconstraint
class lin_vel_node_wrtG(BaseLagrangeConstraint):
    _lc_id = 'lin_vel_node_wrtG'

    def __init__(self):
        self._n_eq = 3

    def get_n_eq(self):
        return self._n_eq

    def initialise(self, MBdict_entry, ieq):
        print('Type of LC: ', self._lc_id)
        print('Arguments and values:')
        for k, v in MBdict_entry.items():
            print(k, v)

        self.vel = MBdict_entry['velocity']
        self.body_number = MBdict_entry['body_number']
        self.node_number = MBdict_entry['node_number']
        self._ieq = ieq
        return self._ieq + self._n_eq

    def staticmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot,
                scalingFactor, penaltyFactor):

        num_LM_eq_specific = self._n_eq
        B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order='F')

        # Define the position of the first degree of freedom associated to the FoR
        # FoR_dof = define_FoR_dof(MB_beam, self.body_number)
        node_dof = define_node_dof(MB_beam, self.body_number, self.node_number)
        ieq = self._ieq

        B[:num_LM_eq_specific, node_dof:node_dof+3] = algebra.quat2rotation(MB_tstep[self.body_number].quat)

        LM_K[sys_size + ieq:sys_size + ieq + num_LM_eq_specific, :sys_size] += scalingFactor * B
        LM_K[:sys_size, sys_size + ieq:sys_size + ieq + num_LM_eq_specific] += scalingFactor * np.transpose(B)

        LM_Q[:sys_size] += scalingFactor * np.dot(np.transpose(B), Lambda[ieq:ieq + num_LM_eq_specific])
        LM_Q[sys_size + ieq:sys_size + ieq + num_LM_eq_specific] += (np.dot(algebra.quat2rotation(MB_tstep[self.body_number].quat), MB_tstep[self.body_number].pos[self.node_number,:]) +
                                                                     MB_tstep[self.body_number].for_pos)
        LM_Q[sys_size + ieq:sys_size + ieq + num_LM_eq_specific] -= (np.dot(algebra.quat2rotation(MB_beam[self.body_number].ini_info.quat), MB_beam[self.body_number].ini_info.pos[self.node_number,:]) +
                                                                     MB_beam[self.body_number].ini_info.for_pos)

        ieq += 3

        return

    def dynamicmat(self, LM_C, LM_K, LM_Q, MB_beam, MB_tstep, ts, num_LM_eq,
                sys_size, dt, Lambda, Lambda_dot,
                scalingFactor, penaltyFactor):
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
            Bnh[:num_LM_eq_specific, FoR_dof:FoR_dof+3] = algebra.quat2rotation(MB_tstep[self.body_number].quat)
            Bnh[:num_LM_eq_specific, FoR_dof+3:FoR_dof+6] = -np.dot(algebra.quat2rotation(MB_tstep[self.body_number].quat), algebra.skew(MB_tstep[self.body_number].pos[self.node_number,:]))
        Bnh[:num_LM_eq_specific, node_dof:node_dof+3] = algebra.quat2rotation(MB_tstep[self.body_number].quat)

        LM_C[sys_size + ieq:sys_size + ieq + num_LM_eq_specific, :sys_size] += scalingFactor * Bnh
        LM_C[:sys_size, sys_size + ieq:sys_size + ieq + num_LM_eq_specific] += scalingFactor * np.transpose(Bnh)

        if MB_beam[self.body_number].FoR_movement == 'free':
            LM_C[FoR_dof:FoR_dof+3, FoR_dof+6:FoR_dof+10] += algebra.der_CquatT_by_v(MB_tstep[self.body_number].quat,Lambda_dot[ieq:ieq + num_LM_eq_specific])
            LM_C[node_dof:node_dof+3, FoR_dof+6:FoR_dof+10] += algebra.der_CquatT_by_v(MB_tstep[self.body_number].quat,Lambda_dot[ieq:ieq + num_LM_eq_specific])
            LM_C[FoR_dof+3:FoR_dof+6, FoR_dof+6:FoR_dof+10] += np.dot(algebra.skew(MB_tstep[self.body_number].pos[self.node_number,:]), algebra.der_CquatT_by_v(MB_tstep[self.body_number].quat,Lambda_dot[ieq:ieq + num_LM_eq_specific]))

            LM_K[FoR_dof+3:FoR_dof+6, node_dof:node_dof+3] -= algebra.skew(np.dot(algebra.quat2rotation(MB_tstep[self.body_number].quat).T, Lambda_dot[ieq:ieq + num_LM_eq_specific]))

        LM_Q[:sys_size] += scalingFactor * np.dot(np.transpose(Bnh), Lambda_dot[ieq:ieq + num_LM_eq_specific])
        LM_Q[sys_size + ieq:sys_size + ieq + num_LM_eq_specific] += (np.dot( algebra.quat2rotation(MB_tstep[self.body_number].quat), (
                MB_tstep[self.body_number].for_vel[0:3] +
                np.dot(algebra.skew(MB_tstep[self.body_number].for_vel[3:6]), MB_tstep[self.body_number].pos[self.node_number,:]) +
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
        index_eq = lc_list[-1].initialise(MBdict["constraint_%02d" % iconstraint], index_eq)

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
        MBdict(MBdict): dictionary with the MultiBody and LagrangeMultipliers information
        MB_beam(list): list of 'beams' of each of the bodies that form the system
        MB_tstep(list): list of 'StructTimeStepInfo' of each of the bodies that form the system
        num_LM_eq(int): number of new equations needed to define the boundary boundary conditions
        sys_size(int): total number of degrees of freedom of the multibody system
        dt(float): time step
        Lambda(numpy array): list of Lagrange multipliers values
        Lambda_dot(numpy array): list of the first derivative of the Lagrange multipliers values
        dynamic_or_static (str): string defining if the computation is dynamic or static

    Returns:
        LM_C (numpy array): Damping matrix associated to the Lagrange Multipliers equations
        LM_K (numpy array): Stiffness matrix associated to the Lagrange Multipliers equations
        LM_Q (numpy array): Vector of independent terms associated to the Lagrange Multipliers equations

    Examples:

    Notes:

    """
    # Lagrange multipliers parameters
    # TODO: set them as an input variable (at this point they should not be changed)
    penaltyFactor = 0.0
    scalingFactor = 1.0

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
                        # MBdict=MBdict,
                        MB_beam=MB_beam,
                        MB_tstep=MB_tstep,
                        ts=ts,
                        num_LM_eq=num_LM_eq,
                        sys_size=sys_size,
                        dt=dt,
                        Lambda=Lambda,
                        Lambda_dot=Lambda_dot,
                        # ieq=ieq,
                        scalingFactor=scalingFactor,
                        penaltyFactor=penaltyFactor)

        elif dynamic_or_static.lower() == "dynamic":
            lc.dynamicmat(LM_C=LM_C,
                        LM_K=LM_K,
                        LM_Q=LM_Q,
                        # MBdict=MBdict,
                        MB_beam=MB_beam,
                        MB_tstep=MB_tstep,
                        ts=ts,
                        num_LM_eq=num_LM_eq,
                        sys_size=sys_size,
                        dt=dt,
                        Lambda=Lambda,
                        Lambda_dot=Lambda_dot,
                        # ieq=ieq,
                        scalingFactor=scalingFactor,
                        penaltyFactor=penaltyFactor)

    return LM_C, LM_K, LM_Q

def postprocess(lc_list, MB_beam, MB_tstep, dynamic_or_static):

    for lc in lc_list:
        if dynamic_or_static.lower() == "static":
            lc.staticpost(lc_list = lc_list,
                           MB_beam = MB_beam,
                           MB_tstep = MB_tstep)
                           # MBdict = MBdict)

        elif dynamic_or_static.lower() == "dynamic":
            lc.dynamicpost(lc_list = lc_list,
                           MB_beam = MB_beam,
                           MB_tstep = MB_tstep)
                           # MBdict = MBdict)

    return


################################################################################
################################################################################
################################################################################
# this at the end of the file
print_available_lc()

# test
# if __name__ == '__main__':
    # lc_list = list()
    # lc_list.append(lc_from_string('SampleLagrange')())
    # lc_list.append(lc_from_string('SampleLagrange')())

    # counter = -1
    # for lc in lc_list:
        # counter += 1
        # lc.initialise(counter=counter)
