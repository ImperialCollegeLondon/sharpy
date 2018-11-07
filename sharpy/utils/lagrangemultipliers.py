"""
LagrangeMultipliers library

Library used to create the matrices associate to boundary conditions through
the method of Lagrange Multipliers

Args:

Returns:

Examples:
    To use this library: import sharpy.utils.lagrangemultipliers as lagrangemultipliers

Notes:

"""
import ctypes as ct
import numpy as np
import sharpy.utils.algebra as algebra
from IPython import embed
import ipdb


def multiply_matrices(*argv):

    size = np.shape(argv[0])
    nrow = size[0]

    # nrow, ncol = np.shape(argv[0])
    sol = np.eye(nrow)
    for M in argv:
        sol = np.dot(sol, M)
    return sol


def define_num_LM_eq(MBdict):
    """
    define_num_LM_eq

    Define the number of equations needed to define the boundary boundary conditions

    Args:
        MBdict(MBdict): dictionary with the MultiBody and LagrangeMultipliers information
    Returns:
        num_LM_eq(int): number of new equations needed to define the boundary boundary conditions

    Examples:
        num_LM_eq = lagrangemultipliers.define_num_LM_eq(MBdict)

    Notes:

    """

    num_constraints = MBdict['num_constraints']
    num_LM_eq = 0

    # Define the number of equations that we need
    for iconstraint in range(num_constraints):

        if MBdict["constraint_%02d" % iconstraint]['behaviour'] == 'hinge_node_FoR':
            num_LM_eq += 5
        elif MBdict["constraint_%02d" % iconstraint]['behaviour'] == 'hinge_node_FoR_constant_vel':
            num_LM_eq += 6
        elif MBdict["constraint_%02d" % iconstraint]['behaviour'] == 'spherical_node_FoR':
            num_LM_eq += 3
        elif MBdict["constraint_%02d" % iconstraint]['behaviour'] == 'free':
            num_LM_eq += 0
        elif MBdict["constraint_%02d" % iconstraint]['behaviour'] == 'hinge_FoR':
            num_LM_eq += 5
        elif MBdict["constraint_%02d" % iconstraint]['behaviour'] == 'fully_constrained_node_FoR':
            num_LM_eq += 6
        elif MBdict["constraint_%02d" % iconstraint]['behaviour'] == 'hinge_node_FoR_constant_rotation':
            num_LM_eq += 4
        elif MBdict["constraint_%02d" % iconstraint]['behaviour'] == 'constant_rot_vel_FoR':
            num_LM_eq += 3
        else:
            print("ERROR: not recognized constraint type")

    return num_LM_eq


def generate_lagrange_matrix(MBdict, MB_beam, MB_tstep, num_LM_eq, sys_size, dt, Lambda, Lambda_dot):
    """
    generate_lagrange_matrix

    Generates the matrices associated to the Lagrange multipliers boundary conditions

    Args:
        MBdict(MBdict): dictionary with the MultiBody and LagrangeMultipliers information
        MB_beam(list): list of 'beams' of each of the bodies that form the system
        MB_tstep(list): list of 'StructTimeStepInfo' of each of the bodies that form the system
        num_LM_eq(int): number of new equations needed to define the boundary boundary conditions
        sys_size(int): total number of degrees of freedom of the multibody system
        dt(float): time step
        Lambda(numpy array): list of Lagrange multipliers values
        Lambda_dot(numpy array): list of the first derivative of the Lagrange multipliers values

    Returns:
        LM_C (numpy array): Damping matrix associated to the Lagrange Multipliers equations
        LM_K (numpy array): Stiffness matrix associated to the Lagrange Multipliers equations
        LM_Q (numpy array): Vector of independent terms associated to the Lagrange Multipliers equations

    Examples:

    Notes:

    """
    # ipdb.set_trace()
    # Lagrange multipliers parameters
    # TODO: set them as an input variable (at this point they should not be changed)
    penaltyFactor = 0.0
    scalingFactor = 1.0

    # Rename variables
    num_constraints = MBdict['num_constraints']

    # Initialize matrices
    LM_C = np.zeros((sys_size + num_LM_eq,sys_size + num_LM_eq), dtype=ct.c_double, order = 'F')
    LM_K = np.zeros((sys_size + num_LM_eq,sys_size + num_LM_eq), dtype=ct.c_double, order = 'F')
    LM_Q = np.zeros((sys_size + num_LM_eq,),dtype=ct.c_double, order = 'F')

    # Bnh = np.zeros((num_LM_eq, sys_size), dtype=ct.c_double, order = 'F')
    # B = np.zeros((num_LM_eq, sys_size), dtype=ct.c_double, order = 'F')

    # Define the matrices associated to the constratints
    ieq = 0
    for iconstraint in range(num_constraints):

        # Rename variables from dictionary
        behaviour = MBdict["constraint_%02d" % iconstraint]['behaviour']

        ###################################################################
        ###################  SPHERICAL BETWEEN NODE AND FOR  ##################
        ###################################################################
        if behaviour == 'spherical_node_FoR':

            # Variables names. The naming of the variables can be quite confusing. The reader should think that
            # the BC relates one "node" and one "FoR" (writen between quotes in these lines).
            # If a variable is related to one of them starts with "node_" or "FoR_" respectively
            # node_number: number of the "node" within its own body
            # node_body: body number of the "node"
            # node_FoR_dof: position of the first degree of freedom of the FoR to which the "node" belongs
            # node_dof: position of the first degree of freedom associated to the "node"
            # FoR_body: body number of the "FoR"
            # FoR_dof: position of the first degree of freedom associated to the "FoR"

            # Rename variables from dictionary
            node_number = MBdict["constraint_%02d" % iconstraint]['node_in_body']
            node_body = MBdict["constraint_%02d" % iconstraint]['body']
            FoR_body = MBdict["constraint_%02d" % iconstraint]['body_FoR']

            num_LM_eq_specific = 3
            Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')
            B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')

            # Define the position of the first degree of freedom associated to the node
            node_dof = 0
            for ibody in range(node_body):
                node_dof += MB_beam[ibody].num_dof.value
                if MB_beam[ibody].FoR_movement == 'free':
                    node_dof += 10
            node_FoR_dof = node_dof + MB_beam[node_body].num_dof.value
            node_dof += 6*(node_number-1)
            if not (MB_beam[node_body].num_dof == 6*(MB_beam[node_body].num_node - 1)):
                # TODO: the previous statement will NOT work for more than one clamped node
                print("WARNING: hinge_node_FoR does not work for more than one clamped node")

            # Define the position of the first degree of freedom associated to the FoR
            FoR_dof = 0
            for ibody in range(FoR_body):
                FoR_dof += MB_beam[ibody].num_dof.value
                if MB_beam[ibody].FoR_movement == 'free':
                    FoR_dof += 10
            FoR_dof += MB_beam[FoR_body].num_dof.value
            # ipdb.set_trace()
            # Option with non holonomic constraints
            #if True:
            # Bnh[ieq:ieq+3, node_dof:node_dof+3] = -1.0*np.eye(3)
            Bnh[:, FoR_dof:FoR_dof+3] = algebra.quat2rotation(MB_tstep[FoR_body].quat)

            Bnh[:, node_dof:node_dof+3] = -1.0*algebra.quat2rotation(MB_tstep[node_body].quat)
            Bnh[:, node_FoR_dof:node_FoR_dof+3] = -1.0*algebra.quat2rotation(MB_tstep[node_body].quat)
            Bnh[:, node_FoR_dof+3:node_FoR_dof+6] = 1.0*np.dot(algebra.quat2rotation(MB_tstep[node_body].quat),algebra.skew(MB_tstep[node_body].pos[node_number,:]))


            # Bnh[ieq+3,ieq+FoR_dof+3] = 1.0

            LM_C[sys_size+ieq:sys_size+ieq+num_LM_eq_specific,:sys_size] += scalingFactor*Bnh
            LM_C[:sys_size,sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += scalingFactor*np.transpose(Bnh)

            LM_Q[:sys_size] += scalingFactor*np.dot(np.transpose(Bnh),Lambda_dot[ieq:ieq+num_LM_eq_specific])
            LM_Q[sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += (np.dot(algebra.quat2rotation(MB_tstep[FoR_body].quat),MB_tstep[FoR_body].for_vel[0:3]) +
                                                                  -1.0*np.dot(algebra.quat2rotation(MB_tstep[node_body].quat),
                                                                    MB_tstep[node_body].pos_dot[node_number,:] +
                                                                    MB_tstep[node_body].for_vel[0:3] +
                                                                    -1.0*np.dot(algebra.skew(MB_tstep[node_body].pos[node_number,:]),MB_tstep[node_body].for_vel[3:6])))

            # LM_Q[sys_size:sys_size+3] = -MB_tstep[0].pos_dot[-1,:] + np.dot(algebra.quat2rotation(quat),MB_tstep[1].for_vel[0:3])
            # LM_Q[sys_size+3] = MB_tstep[1].for_vel[3]

            #LM_K[FoR_dof:FoR_dof+3,FoR_dof+6:FoR_dof+10] = algebra.der_CquatT_by_v(MB_tstep[FoR_body].quat,Lambda_dot)
            LM_C[FoR_dof:FoR_dof+3,FoR_dof+6:FoR_dof+10] += algebra.der_CquatT_by_v(MB_tstep[FoR_body].quat,scalingFactor*Lambda_dot[ieq:ieq+num_LM_eq_specific])

            LM_C[node_dof:node_dof+3,node_FoR_dof+6:node_FoR_dof+10] -= algebra.der_CquatT_by_v(MB_tstep[node_body].quat,scalingFactor*Lambda_dot[ieq:ieq+num_LM_eq_specific])
            LM_C[node_FoR_dof:node_FoR_dof+3,node_FoR_dof+6:node_FoR_dof+10] -= algebra.der_CquatT_by_v(MB_tstep[node_body].quat,scalingFactor*Lambda_dot[ieq:ieq+num_LM_eq_specific])

            LM_C[node_FoR_dof+3:node_FoR_dof+6,node_FoR_dof+6:node_FoR_dof+10] -= np.dot(algebra.skew(MB_tstep[node_body].pos[node_number,:]), algebra.der_CquatT_by_v(MB_tstep[node_body].quat,scalingFactor*Lambda_dot[ieq:ieq+num_LM_eq_specific]))

            LM_K[node_FoR_dof+3:node_FoR_dof+6,node_dof:node_dof+3] += algebra.skew(np.dot(algebra.quat2rotation(MB_tstep[node_body].quat).T,Lambda_dot[ieq:ieq+num_LM_eq_specific]))

            # ieq += 4
            ieq += 3

        ###################################################################
        ###################  HINGE BETWEEN NODE AND FOR  ##################
        ###################################################################
        elif behaviour == 'hinge_node_FoR':

            # Variables names. The naming of the variables can be quite confusing. The reader should think that
            # the BC relates one "node" and one "FoR" (writen between quotes in these lines).
            # If a variable is related to one of them starts with "node_" or "FoR_" respectively
            # node_number: number of the "node" within its own body
            # node_body: body number of the "node"
            # node_FoR_dof: position of the first degree of freedom of the FoR to which the "node" belongs
            # node_dof: position of the first degree of freedom associated to the "node"
            # FoR_body: body number of the "FoR"
            # FoR_dof: position of the first degree of freedom associated to the "FoR"

            # Rename variables from dictionary
            node_number = MBdict["constraint_%02d" % iconstraint]['node_in_body']
            node_body = MBdict["constraint_%02d" % iconstraint]['body']
            FoR_body = MBdict["constraint_%02d" % iconstraint]['body_FoR']
            rot_axisB = MBdict["constraint_%02d" % iconstraint]['rot_axisB']

            # Define the position of the first degree of freedom associated to the node
            node_dof = 0
            for ibody in range(node_body):
                node_dof += MB_beam[ibody].num_dof.value
                if MB_beam[ibody].FoR_movement == 'free':
                    node_dof += 10
            node_FoR_dof = node_dof + MB_beam[node_body].num_dof.value
            node_dof += 6*(node_number-1)
            if not (MB_beam[node_body].num_dof == 6*(MB_beam[node_body].num_node - 1)):
                # TODO: the previous statement will NOT work for more than one clamped node
                print("WARNING: hinge_node_FoR does not work for more than one clamped node")

            # Define the position of the first degree of freedom associated to the FoR
            FoR_dof = 0
            for ibody in range(FoR_body):
                FoR_dof += MB_beam[ibody].num_dof.value
                if MB_beam[ibody].FoR_movement == 'free':
                    FoR_dof += 10
            FoR_dof += MB_beam[FoR_body].num_dof.value

            # Constrain linear velocities
            num_LM_eq_specific = 3
            Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')
            B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')

            Bnh[:, FoR_dof:FoR_dof+3] = algebra.quat2rotation(MB_tstep[FoR_body].quat)

            Bnh[:, node_dof:node_dof+3] = -1.0*algebra.quat2rotation(MB_tstep[node_body].quat)
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

            LM_C[node_dof:node_dof+3,node_FoR_dof+6:node_FoR_dof+10] -= algebra.der_CquatT_by_v(MB_tstep[node_body].quat,scalingFactor*Lambda_dot[ieq:ieq+num_LM_eq_specific])
            LM_C[node_FoR_dof:node_FoR_dof+3,node_FoR_dof+6:node_FoR_dof+10] -= algebra.der_CquatT_by_v(MB_tstep[node_body].quat,scalingFactor*Lambda_dot[ieq:ieq+num_LM_eq_specific])

            LM_C[node_FoR_dof+3:node_FoR_dof+6,node_FoR_dof+6:node_FoR_dof+10] -= np.dot(algebra.skew(MB_tstep[node_body].pos[node_number,:]), algebra.der_CquatT_by_v(MB_tstep[node_body].quat,scalingFactor*Lambda_dot[ieq:ieq+num_LM_eq_specific]))

            LM_K[node_FoR_dof+3:node_FoR_dof+6,node_dof:node_dof+3] += algebra.skew(np.dot(algebra.quat2rotation(MB_tstep[node_body].quat).T,Lambda_dot[ieq:ieq+num_LM_eq_specific]))

            ieq += 3


            # The three equations to constrain angular velocitie swill be linearly dependent, remove one
            ielem, inode_in_elem = MB_beam[node_body].node_master_elem[node_number]
            aux_Bnh = multiply_matrices(algebra.skew(rot_axisB),
                                      algebra.crv2rotation(MB_tstep[node_body].psi[ielem,inode_in_elem,:]),
                                      algebra.quat2rotation(MB_tstep[node_body].quat).T,
                                      algebra.quat2rotation(MB_tstep[FoR_body].quat))

            n0 = np.linalg.norm(aux_Bnh[0,:])
            n1 = np.linalg.norm(aux_Bnh[1,:])
            n2 = np.linalg.norm(aux_Bnh[2,:])
            if ((n0 < n1) and (n0 < n2)):
                indep = np.array([1,2], dtype = int)
                new_Lambda_dot = np.array([0., Lambda_dot[ieq], Lambda_dot[ieq+1]])
            elif ((n1 < n0) and (n1 < n2)):
                indep = np.array([0,2], dtype = int)
                new_Lambda_dot = np.array([Lambda_dot[ieq], 0.0, Lambda_dot[ieq+1]])
            elif ((n2 < n0) and (n2 < n1)):
                indep = np.array([0,1], dtype = int)
                new_Lambda_dot = np.array([Lambda_dot[ieq], Lambda_dot[ieq+1], 0.0])

            num_LM_eq_specific = 2
            Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')
            B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')

            # Lambda_dot[ieq:ieq+num_LM_eq_specific]
            # np.concatenate((Lambda_dot[ieq:ieq+num_LM_eq_specific], np.array([0.])))


            Bnh[:, FoR_dof+3:FoR_dof+6] = multiply_matrices(algebra.skew(rot_axisB),
                                                          algebra.crv2rotation(MB_tstep[node_body].psi[ielem,inode_in_elem,:]),
                                                          algebra.quat2rotation(MB_tstep[node_body].quat).T,
                                                          algebra.quat2rotation(MB_tstep[FoR_body].quat))[indep,:]

            # Constrain angular velocities
            LM_Q[:sys_size] += scalingFactor*np.dot(np.transpose(Bnh),Lambda_dot[ieq:ieq+num_LM_eq_specific])
            LM_Q[sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += multiply_matrices(algebra.skew(rot_axisB),
                                                          algebra.crv2rotation(MB_tstep[node_body].psi[ielem,inode_in_elem,:]),
                                                          algebra.quat2rotation(MB_tstep[node_body].quat).T,
                                                          algebra.quat2rotation(MB_tstep[FoR_body].quat),
                                                          MB_tstep[FoR_body].for_vel[3:6])[indep]

            LM_C[sys_size+ieq:sys_size+ieq+num_LM_eq_specific,:sys_size] += scalingFactor*Bnh
            LM_C[:sys_size,sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += scalingFactor*np.transpose(Bnh)

            LM_C[FoR_dof+3+indep,node_FoR_dof+6:node_FoR_dof+10] += np.dot(algebra.quat2rotation(MB_tstep[FoR_body].quat).T,
                                                                               algebra.der_Cquat_by_v(MB_tstep[node_body].quat,
                                                                                                      multiply_matrices(algebra.crv2rotation(MB_tstep[node_body].psi[ielem,inode_in_elem,:]).T,
                                                                                                                        algebra.skew(rot_axisB).T,
                                                                                                                        new_Lambda_dot)))[indep,:]

            LM_C[FoR_dof+3+indep,FoR_dof+6:FoR_dof+10] += algebra.der_CquatT_by_v(MB_tstep[FoR_body].quat,
                                                                                      multiply_matrices(algebra.quat2rotation(MB_tstep[node_body].quat),
                                                                                                        algebra.crv2rotation(MB_tstep[node_body].psi[ielem,inode_in_elem,:]).T,
                                                                                                        algebra.skew(rot_axisB).T,
                                                                                                        new_Lambda_dot))[indep,:]

            LM_K[FoR_dof+3+indep,node_dof+3:node_dof+6] += multiply_matrices(algebra.quat2rotation(MB_tstep[FoR_body].quat).T,
                                                                                 algebra.quat2rotation(MB_tstep[node_body].quat),
                                                                                 algebra.der_CcrvT_by_v(MB_tstep[node_body].psi[ielem,inode_in_elem,:],
                                                                                                        np.dot(algebra.skew(rot_axisB).T,
                                                                                                               new_Lambda_dot)))[indep,:]

            ieq += 2

        ###################################################################
        ##############  HINGE BETWEEN NODE AND FOR  CONSTANT VEL  #########
        ###################################################################
        elif behaviour == 'hinge_node_FoR_constant_vel':

            # Variables names. The naming of the variables can be quite confusing. The reader should think that
            # the BC relates one "node" and one "FoR" (writen between quotes in these lines).
            # If a variable is related to one of them starts with "node_" or "FoR_" respectively
            # node_number: number of the "node" within its own body
            # node_body: body number of the "node"
            # node_FoR_dof: position of the first degree of freedom of the FoR to which the "node" belongs
            # node_dof: position of the first degree of freedom associated to the "node"
            # FoR_body: body number of the "FoR"
            # FoR_dof: position of the first degree of freedom associated to the "FoR"

            # Rename variables from dictionary
            node_number = MBdict["constraint_%02d" % iconstraint]['node_in_body']
            node_body = MBdict["constraint_%02d" % iconstraint]['body']
            FoR_body = MBdict["constraint_%02d" % iconstraint]['body_FoR']
            rot_axisB = MBdict["constraint_%02d" % iconstraint]['rot_axisB']
            rot_vel = MBdict["constraint_%02d" % iconstraint]['rot_vel']

            # Define the position of the first degree of freedom associated to the node
            node_dof = 0
            for ibody in range(node_body):
                node_dof += MB_beam[ibody].num_dof.value
                if MB_beam[ibody].FoR_movement == 'free':
                    node_dof += 10
            node_FoR_dof = node_dof + MB_beam[node_body].num_dof.value
            node_dof += 6*(node_number-1)
            if not (MB_beam[node_body].num_dof == 6*(MB_beam[node_body].num_node - 1)):
                # TODO: the previous statement will NOT work for more than one clamped node
                print("WARNING: hinge_node_FoR does not work for more than one clamped node")

            # Define the position of the first degree of freedom associated to the FoR
            FoR_dof = 0
            for ibody in range(FoR_body):
                FoR_dof += MB_beam[ibody].num_dof.value
                if MB_beam[ibody].FoR_movement == 'free':
                    FoR_dof += 10
            FoR_dof += MB_beam[FoR_body].num_dof.value

            # Constrain linear velocities
            num_LM_eq_specific = 3
            Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')
            B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')

            Bnh[:, FoR_dof:FoR_dof+3] = algebra.quat2rotation(MB_tstep[FoR_body].quat)

            Bnh[:, node_dof:node_dof+3] = -1.0*algebra.quat2rotation(MB_tstep[node_body].quat)
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

            LM_C[node_dof:node_dof+3,node_FoR_dof+6:node_FoR_dof+10] -= algebra.der_CquatT_by_v(MB_tstep[node_body].quat,scalingFactor*Lambda_dot[ieq:ieq+num_LM_eq_specific])
            LM_C[node_FoR_dof:node_FoR_dof+3,node_FoR_dof+6:node_FoR_dof+10] -= algebra.der_CquatT_by_v(MB_tstep[node_body].quat,scalingFactor*Lambda_dot[ieq:ieq+num_LM_eq_specific])

            LM_C[node_FoR_dof+3:node_FoR_dof+6,node_FoR_dof+6:node_FoR_dof+10] -= np.dot(algebra.skew(MB_tstep[node_body].pos[node_number,:]), algebra.der_CquatT_by_v(MB_tstep[node_body].quat,scalingFactor*Lambda_dot[ieq:ieq+num_LM_eq_specific]))

            LM_K[node_FoR_dof+3:node_FoR_dof+6,node_dof:node_dof+3] += algebra.skew(np.dot(algebra.quat2rotation(MB_tstep[node_body].quat).T,Lambda_dot[ieq:ieq+num_LM_eq_specific]))

            ieq += 3


            # The three equations to constrain angular velocitie swill be linearly dependent, remove one
            ielem, inode_in_elem = MB_beam[node_body].node_master_elem[node_number]
            aux_Bnh = multiply_matrices(algebra.skew(rot_axisB),
                                      algebra.crv2rotation(MB_tstep[node_body].psi[ielem,inode_in_elem,:]),
                                      algebra.quat2rotation(MB_tstep[node_body].quat).T,
                                      algebra.quat2rotation(MB_tstep[FoR_body].quat))

            n0 = np.linalg.norm(aux_Bnh[0,:])
            n1 = np.linalg.norm(aux_Bnh[1,:])
            n2 = np.linalg.norm(aux_Bnh[2,:])
            if ((n0 < n1) and (n0 < n2)):
                indep = np.array([1,2], dtype = int)
                new_Lambda_dot = np.array([0., Lambda_dot[ieq], Lambda_dot[ieq+1]])
            elif ((n1 < n0) and (n1 < n2)):
                indep = np.array([0,2], dtype = int)
                new_Lambda_dot = np.array([Lambda_dot[ieq], 0.0, Lambda_dot[ieq+1]])
            elif ((n2 < n0) and (n2 < n1)):
                indep = np.array([0,1], dtype = int)
                new_Lambda_dot = np.array([Lambda_dot[ieq], Lambda_dot[ieq+1], 0.0])

            num_LM_eq_specific = 2
            Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')
            B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')

            # Lambda_dot[ieq:ieq+num_LM_eq_specific]
            # np.concatenate((Lambda_dot[ieq:ieq+num_LM_eq_specific], np.array([0.])))


            Bnh[:, FoR_dof+3:FoR_dof+6] = multiply_matrices(algebra.skew(rot_axisB),
                                                          algebra.crv2rotation(MB_tstep[node_body].psi[ielem,inode_in_elem,:]),
                                                          algebra.quat2rotation(MB_tstep[node_body].quat).T,
                                                          algebra.quat2rotation(MB_tstep[FoR_body].quat))[indep,:]

            # Constrain angular velocities
            LM_Q[:sys_size] += scalingFactor*np.dot(np.transpose(Bnh),Lambda_dot[ieq:ieq+num_LM_eq_specific])
            LM_Q[sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += multiply_matrices(algebra.skew(rot_axisB),
                                                          algebra.crv2rotation(MB_tstep[node_body].psi[ielem,inode_in_elem,:]),
                                                          algebra.quat2rotation(MB_tstep[node_body].quat).T,
                                                          algebra.quat2rotation(MB_tstep[FoR_body].quat),
                                                          MB_tstep[FoR_body].for_vel[3:6])[indep]

            LM_C[sys_size+ieq:sys_size+ieq+num_LM_eq_specific,:sys_size] += scalingFactor*Bnh
            LM_C[:sys_size,sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += scalingFactor*np.transpose(Bnh)

            LM_C[FoR_dof+3+indep,node_FoR_dof+6:node_FoR_dof+10] += np.dot(algebra.quat2rotation(MB_tstep[FoR_body].quat).T,
                                                                               algebra.der_Cquat_by_v(MB_tstep[node_body].quat,
                                                                                                      multiply_matrices(algebra.crv2rotation(MB_tstep[node_body].psi[ielem,inode_in_elem,:]).T,
                                                                                                                        algebra.skew(rot_axisB).T,
                                                                                                                        new_Lambda_dot)))[indep,:]

            LM_C[FoR_dof+3+indep,FoR_dof+6:FoR_dof+10] += algebra.der_CquatT_by_v(MB_tstep[FoR_body].quat,
                                                                                      multiply_matrices(algebra.quat2rotation(MB_tstep[node_body].quat),
                                                                                                        algebra.crv2rotation(MB_tstep[node_body].psi[ielem,inode_in_elem,:]).T,
                                                                                                        algebra.skew(rot_axisB).T,
                                                                                                        new_Lambda_dot))[indep,:]

            LM_K[FoR_dof+3+indep,node_dof+3:node_dof+6] += multiply_matrices(algebra.quat2rotation(MB_tstep[FoR_body].quat).T,
                                                                                 algebra.quat2rotation(MB_tstep[node_body].quat),
                                                                                 algebra.der_CcrvT_by_v(MB_tstep[node_body].psi[ielem,inode_in_elem,:],
                                                                                                        np.dot(algebra.skew(rot_axisB).T,
                                                                                                               new_Lambda_dot)))[indep,:]

            ieq += 2

            # fix velocity value
            num_LM_eq_specific = 1
            Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')
            B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')

            # Lambda_dot[ieq:ieq+num_LM_eq_specific]
            # np.concatenate((Lambda_dot[ieq:ieq+num_LM_eq_specific], np.array([0.])))


            Bnh[:, FoR_dof+3:FoR_dof+6] = multiply_matrices(rot_axisB,
                                                          algebra.crv2rotation(MB_tstep[node_body].psi[ielem,inode_in_elem,:]),
                                                          algebra.quat2rotation(MB_tstep[node_body].quat).T,
                                                          algebra.quat2rotation(MB_tstep[FoR_body].quat))

            # Constrain angular velocities
            LM_Q[:sys_size] += scalingFactor*np.dot(np.transpose(Bnh),Lambda_dot[ieq:ieq+num_LM_eq_specific])
            LM_Q[sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += multiply_matrices(rot_axisB,
                                                          algebra.crv2rotation(MB_tstep[node_body].psi[ielem,inode_in_elem,:]),
                                                          algebra.quat2rotation(MB_tstep[node_body].quat).T,
                                                          algebra.quat2rotation(MB_tstep[FoR_body].quat),
                                                          MB_tstep[FoR_body].for_vel[3:6]) - rot_vel

            LM_C[sys_size+ieq:sys_size+ieq+num_LM_eq_specific,:sys_size] += scalingFactor*Bnh
            LM_C[:sys_size,sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += scalingFactor*np.transpose(Bnh)

            LM_C[FoR_dof+3:FoR_dof+6,node_FoR_dof+6:node_FoR_dof+10] += np.dot(algebra.quat2rotation(MB_tstep[FoR_body].quat).T,
                                                                               algebra.der_Cquat_by_v(MB_tstep[node_body].quat,
                                                                                                      multiply_matrices(algebra.crv2rotation(MB_tstep[node_body].psi[ielem,inode_in_elem,:]).T,
                                                                                                                        # rot_axisB.T,
                                                                                                                        rot_axisB.T*Lambda_dot[ieq:ieq+num_LM_eq_specific])))

            LM_C[FoR_dof+3:FoR_dof+6,FoR_dof+6:FoR_dof+10] += algebra.der_CquatT_by_v(MB_tstep[FoR_body].quat,
                                                                                      multiply_matrices(algebra.quat2rotation(MB_tstep[node_body].quat),
                                                                                                        algebra.crv2rotation(MB_tstep[node_body].psi[ielem,inode_in_elem,:]).T,
                                                                                                        rot_axisB.T*Lambda_dot[ieq:ieq+num_LM_eq_specific]))

            LM_K[FoR_dof+3:FoR_dof+6,node_dof+3:node_dof+6] += multiply_matrices(algebra.quat2rotation(MB_tstep[FoR_body].quat).T,
                                                                                 algebra.quat2rotation(MB_tstep[node_body].quat),
                                                                                 algebra.der_CcrvT_by_v(MB_tstep[node_body].psi[ielem,inode_in_elem,:],
                                                                                                        rot_axisB.T*Lambda_dot[ieq:ieq+num_LM_eq_specific]))

            ieq += 1


        ###################################################################
        ###############################  HINGE FOR  #######################
        ###################################################################
        elif behaviour == 'hinge_FoR':

            # Rename variables from dictionary
            body_FoR = MBdict["constraint_%02d" % iconstraint]['body_FoR']
            rot_axis = MBdict["constraint_%02d" % iconstraint]['rot_axis_AFoR']

            num_LM_eq_specific = 5
            Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')
            B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')

            # Define the position of the first degree of freedom associated to the FoR
            FoR_dof = 0
            for ibody in range(body_FoR):
                FoR_dof += MB_beam[ibody].num_dof.value
                if MB_beam[ibody].FoR_movement == 'free':
                    FoR_dof += 10
            FoR_dof += MB_beam[body_FoR].num_dof.value

            Bnh[:3, FoR_dof:FoR_dof+3] = 1.0*np.eye(3)

            # Only two of these equations are linearly independent
            skew_rot_axis = algebra.skew(rot_axis)
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

            LM_Q[sys_size+ieq:sys_size+ieq+3] += MB_tstep[body_FoR].for_vel[0:3].astype(dtype=ct.c_double, copy=True, order='F')
            LM_Q[sys_size+ieq+3:sys_size+ieq+5] += np.dot(skew_rot_axis[[row0,row1],:], MB_tstep[body_FoR].for_vel[3:6])

            ieq += 5

        ###################################################################
        #############  FULL CONSTRAINT BETWEEN NODE AND FOR  ##############
        ###################################################################
        elif MBdict["constraint_%02d" % iconstraint]['behaviour'] == 'fully_constrained_node_FoR':

            print("WARNING: do not use fully_constrained_node_FoR. It is outdated")

            # Rename variables from dictionary
            node_in_body = MBdict["constraint_%02d" % iconstraint]['node_in_body']
            node_body = MBdict["constraint_%02d" % iconstraint]['body']
            body_FoR = MBdict["constraint_%02d" % iconstraint]['body_FoR']

            num_LM_eq_specific = 6
            Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')
            B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')

            # Define the position of the first degree of freedom associated to the node
            node_dof = 0
            for ibody in range(node_body):
                node_dof += MB_beam[ibody].num_dof.value
                if MB_beam[ibody].FoR_movement == 'free':
                    node_dof += 10
            # TODO: this will NOT work for more than one clamped node
            node_dof += 6*(node_in_body-1)

            # Define the position of the first degree of freedom associated to the FoR
            FoR_dof = 0
            for ibody in range(body_FoR):
                FoR_dof += MB_beam[ibody].num_dof.value
                if MB_beam[ibody].FoR_movement == 'free':
                    FoR_dof += 10
            FoR_dof += MB_beam[body_FoR].num_dof.value

            # Option with non holonomic constraints
            # BC for linear velocities
            Bnh[:3, node_dof:node_dof+3] = -1.0*np.eye(3)
            #TODO: change this when the master AFoR is able to move
            quat = algebra.quat_bound(MB_tstep[body_FoR].quat)
            Bnh[:3, FoR_dof:FoR_dof+3] = algebra.quat2rotation(quat)

            # BC for angular velocities
            Bnh[3:6,FoR_dof+3:FoR_dof+6] = -1.0*algebra.quat2rotation(quat)
            ielem, inode_in_elem = MB_beam[0].node_master_elem[node_in_body]
            Bnh[3:6,node_dof+3:node_dof+6] = algebra.crv2tan(MB_tstep[0].psi[ielem, inode_in_elem, :])

            LM_C[sys_size+ieq:sys_size+ieq+num_LM_eq_specific,:sys_size] += scalingFactor*Bnh
            LM_C[:sys_size,sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += scalingFactor*np.transpose(Bnh)

            LM_Q[:sys_size] += scalingFactor*np.dot(np.transpose(Bnh),Lambda_dot[ieq:ieq+num_LM_eq_specific])
            LM_Q[sys_size+ieq:sys_size+ieq+3] += -MB_tstep[0].pos_dot[-1,:] + np.dot(algebra.quat2rotation(quat),MB_tstep[1].for_vel[0:3])
            LM_Q[sys_size+ieq+3:sys_size+ieq+6] += (np.dot(algebra.crv2tan(MB_tstep[0].psi[ielem, inode_in_elem, :]),MB_tstep[0].psi_dot[ielem, inode_in_elem, :]) -
                                          np.dot(algebra.quat2rotation(quat), MB_tstep[body_FoR].for_vel[3:6]))

            #LM_K[FoR_dof:FoR_dof+3,FoR_dof+6:FoR_dof+10] = algebra.der_CquatT_by_v(MB_tstep[body_FoR].quat,Lambda_dot)
            LM_C[FoR_dof:FoR_dof+3,FoR_dof+6:FoR_dof+10] += algebra.der_CquatT_by_v(quat,scalingFactor*Lambda_dot[ieq:ieq+3])
            LM_C[FoR_dof+3:FoR_dof+6,FoR_dof+6:FoR_dof+10] -= algebra.der_CquatT_by_v(quat,scalingFactor*Lambda_dot[ieq+3:ieq+6])

            LM_K[node_dof+3:node_dof+6,node_dof+3:node_dof+6] += algebra.der_TanT_by_xv(MB_tstep[0].psi[ielem, inode_in_elem, :],scalingFactor*Lambda_dot[ieq+3:ieq+6])

            ieq += 6

        ###################################################################
        ###################  HINGE BETWEEN NODE AND FOR  ##################
        ###################################################################
        # elif behaviour == 'hinge_node_FoR_constant_rotation':
        #
        #     print("WARNING: do not use hinge_node_FoR_constant_rotation. It is outdated")
        #
        #     # Rename variables from dictionary
        #     node_in_body = MBdict["constraint_%02d" % iconstraint]['node_in_body']
        #     node_body = MBdict["constraint_%02d" % iconstraint]['body']
        #     body_FoR = MBdict["constraint_%02d" % iconstraint]['body_FoR']
        #     rot_vel = MBdict["constraint_%02d" % iconstraint]['rot_vel']
        #
        #     num_LM_eq_specific = 4
        #     Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')
        #     B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')
        #
        #     # Define the position of the first degree of freedom associated to the node
        #     node_dof = 0
        #     for ibody in range(node_body):
        #         node_dof += MB_beam[ibody].num_dof.value
        #         if MB_beam[ibody].FoR_movement == 'free':
        #             node_dof += 10
        #     # TODO: this will NOT work for more than one clamped node
        #     node_dof += 6*(node_in_body-1)
        #
        #     # Define the position of the first degree of freedom associated to the FoR
        #     FoR_dof = 0
        #     for ibody in range(body_FoR):
        #         FoR_dof += MB_beam[ibody].num_dof.value
        #         if MB_beam[ibody].FoR_movement == 'free':
        #             FoR_dof += 10
        #     FoR_dof += MB_beam[body_FoR].num_dof.value
        #
        #     # Option with non holonomic constraints
        #     #if True:
        #     Bnh[:3, node_dof:node_dof+3] = -1.0*np.eye(3)
        #     #TODO: change this when the master AFoR is able to move
        #     quat = algebra.quat_bound(MB_tstep[body_FoR].quat)
        #     Bnh[:3, FoR_dof:FoR_dof+3] = algebra.quat2rotation(quat)
        #     # Bnh[ieq:ieq+3, FoR_dof:FoR_dof+3] = np.eye(3)
        #
        #     Bnh[3,ieq+FoR_dof+5] = 1.0
        #
        #     LM_C[sys_size+ieq:sys_size+ieq+num_LM_eq_specific,:sys_size] += scalingFactor*Bnh
        #     LM_C[:sys_size,sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += scalingFactor*np.transpose(Bnh)
        #
        #     LM_Q[:sys_size] += scalingFactor*np.dot(np.transpose(Bnh),Lambda_dot[ieq:ieq+4])
        #     LM_Q[sys_size+ieq:sys_size+ieq+3] += -MB_tstep[0].pos_dot[-1,:] + np.dot(algebra.quat2rotation(quat),MB_tstep[1].for_vel[0:3])
        #     LM_Q[sys_size+ieq+3] += MB_tstep[1].for_vel[5] - rot_vel
        #
        #     LM_C[FoR_dof:FoR_dof+3,FoR_dof+6:FoR_dof+10] += algebra.der_CquatT_by_v(quat,scalingFactor*Lambda_dot[ieq:ieq+3])
        #
        #     ieq += 4

        ###################################################################
        ###################  CONSTANT ANGULAR VEL FOR  ####################
        ###################################################################
        elif behaviour == 'constant_rot_vel_FoR':

                # Rename variables from dictionary
                rot_vel = MBdict["constraint_%02d" % iconstraint]['rot_vel']
                FoR_body = MBdict["constraint_%02d" % iconstraint]['FoR_body']

                num_LM_eq_specific = 3
                Bnh = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')
                B = np.zeros((num_LM_eq_specific, sys_size), dtype=ct.c_double, order = 'F')

                # Define the position of the first degree of freedom associated to the FoR
                FoR_dof = 0
                for ibody in range(FoR_body):
                    FoR_dof += MB_beam[ibody].num_dof.value
                    if MB_beam[ibody].FoR_movement == 'free':
                        FoR_dof += 10
                FoR_dof += MB_beam[FoR_body].num_dof.value

                Bnh[:3,FoR_dof+3:FoR_dof+6] = np.eye(3)

                LM_C[sys_size+ieq:sys_size+ieq+num_LM_eq_specific,:sys_size] += scalingFactor*Bnh
                LM_C[:sys_size,sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += scalingFactor*np.transpose(Bnh)

                LM_Q[:sys_size] += scalingFactor*np.dot(np.transpose(Bnh),Lambda_dot[ieq:ieq+num_LM_eq_specific])
                LM_Q[sys_size+ieq:sys_size+ieq+num_LM_eq_specific] += MB_tstep[FoR_body].for_vel[3:6] - rot_vel

                ieq += 3

    return LM_C, LM_K, LM_Q
