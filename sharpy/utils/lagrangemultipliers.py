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
            num_LM_eq += 4
        elif MBdict["constraint_%02d" % iconstraint]['behaviour'] == 'free':
            num_LM_eq += 0
        elif MBdict["constraint_%02d" % iconstraint]['behaviour'] == 'hinge_FoR':
            num_LM_eq += 5
        elif MBdict["constraint_%02d" % iconstraint]['behaviour'] == 'fully_constrained_node_FoR':
            num_LM_eq += 6
        elif MBdict["constraint_%02d" % iconstraint]['behaviour'] == 'hinge_node_FoR_constant_rotation':
            num_LM_eq += 4
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

    Bnh = np.zeros((num_LM_eq, sys_size), dtype=ct.c_double, order = 'F')
    B = np.zeros((num_LM_eq, sys_size), dtype=ct.c_double, order = 'F')

    # Define the matrices associated to the constratints
    ieq = 0
    for iconstraint in range(num_constraints):

        # Rename variables from dictionary
        behaviour = MBdict["constraint_%02d" % iconstraint]['behaviour']

        ###################################################################
        ###################  HINGE BETWEEN NODE AND FOR  ##################
        ###################################################################
        if behaviour == 'hinge_node_FoR':

            # Rename variables from dictionary
            node_in_body = MBdict["constraint_%02d" % iconstraint]['node_in_body']
            node_body = MBdict["constraint_%02d" % iconstraint]['body']
            body_FoR = MBdict["constraint_%02d" % iconstraint]['body_FoR']

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
            #if True:
            Bnh[ieq:ieq+3, node_dof:node_dof+3] = -1.0*np.eye(3)
            #TODO: change this when the master AFoR is able to move
            quat = algebra.quat_bound(MB_tstep[body_FoR].quat)
            Bnh[ieq:ieq+3, FoR_dof:FoR_dof+3] = algebra.quat2rotation(quat)

            Bnh[3,FoR_dof+3] = 1.0

            LM_C[sys_size:,:sys_size] = scalingFactor*Bnh
            LM_C[:sys_size,sys_size:] = scalingFactor*np.transpose(Bnh)

            LM_Q[:sys_size] = scalingFactor*np.dot(np.transpose(Bnh),Lambda_dot)
            LM_Q[sys_size:sys_size+3] = -MB_tstep[0].pos_dot[-1,:] + np.dot(algebra.quat2rotation(quat),MB_tstep[1].for_vel[0:3])
            LM_Q[sys_size+3] = MB_tstep[1].for_vel[3]

            #LM_K[FoR_dof:FoR_dof+3,FoR_dof+6:FoR_dof+10] = algebra.der_CquatT_by_v(MB_tstep[body_FoR].quat,Lambda_dot)
            LM_C[FoR_dof:FoR_dof+3,FoR_dof+6:FoR_dof+10] += algebra.der_CquatT_by_v(quat,scalingFactor*Lambda_dot[0:3])

            ieq += 4

        ###################################################################
        ###############################  HINGE FOR  #######################
        ###################################################################
        elif behaviour == 'hinge_FoR':

            # Rename variables from dictionary
            body_FoR = MBdict["constraint_%02d" % iconstraint]['body_FoR']
            rot_axis = MBdict["constraint_%02d" % iconstraint]['rot_axis_AFoR']

            # Define the position of the first degree of freedom associated to the FoR
            FoR_dof = 0
            for ibody in range(body_FoR):
                FoR_dof += MB_beam[ibody].num_dof.value
                if MB_beam[ibody].FoR_movement == 'free':
                    FoR_dof += 10
            FoR_dof += MB_beam[body_FoR].num_dof.value

            Bnh[ieq:ieq+3, FoR_dof:FoR_dof+3] = 1.0*np.eye(3)

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

            Bnh[ieq+3:ieq+5, FoR_dof+3:FoR_dof+6] = skew_rot_axis[[row0,row1],:]

            LM_C[sys_size:,:sys_size] = scalingFactor*Bnh
            LM_C[:sys_size,sys_size:] = scalingFactor*np.transpose(Bnh)

            LM_Q[:sys_size] = scalingFactor*np.dot(np.transpose(Bnh),Lambda_dot)

            LM_Q[sys_size:sys_size+3] = MB_tstep[body_FoR].for_vel[0:3].astype(dtype=ct.c_double, copy=True, order='F')
            LM_Q[sys_size+3:sys_size+5] += np.dot(skew_rot_axis[[row0,row1],:], MB_tstep[body_FoR].for_vel[3:6])

            ieq += 5

        ###################################################################
        #############  FULL CONSTRAINT BETWEEN NODE AND FOR  ##############
        ###################################################################
        elif MBdict["constraint_%02d" % iconstraint]['behaviour'] == 'fully_constrained_node_FoR':

            # Rename variables from dictionary
            node_in_body = MBdict["constraint_%02d" % iconstraint]['node_in_body']
            node_body = MBdict["constraint_%02d" % iconstraint]['body']
            body_FoR = MBdict["constraint_%02d" % iconstraint]['body_FoR']

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
            Bnh[ieq:ieq+3, node_dof:node_dof+3] = -1.0*np.eye(3)
            #TODO: change this when the master AFoR is able to move
            quat = algebra.quat_bound(MB_tstep[body_FoR].quat)
            Bnh[ieq:ieq+3, FoR_dof:FoR_dof+3] = algebra.quat2rotation(quat)

            # BC for angular velocities
            Bnh[ieq+3:ieq+6,FoR_dof+3:FoR_dof+6] = -1.0*algebra.quat2rotation(quat)
            ielem, inode_in_elem = MB_beam[0].node_master_elem[node_in_body]
            Bnh[ieq+3:ieq+6,node_dof+3:node_dof+6] = algebra.crv2tan(MB_tstep[0].psi[ielem, inode_in_elem, :])

            LM_C[sys_size:,:sys_size] = scalingFactor*Bnh
            LM_C[:sys_size,sys_size:] = scalingFactor*np.transpose(Bnh)

            LM_Q[:sys_size] = scalingFactor*np.dot(np.transpose(Bnh),Lambda_dot)
            LM_Q[sys_size:sys_size+3] = -MB_tstep[0].pos_dot[-1,:] + np.dot(algebra.quat2rotation(quat),MB_tstep[1].for_vel[0:3])
            LM_Q[sys_size+3:sys_size+6] = (np.dot(algebra.crv2tan(MB_tstep[0].psi[ielem, inode_in_elem, :]),MB_tstep[0].psi_dot[ielem, inode_in_elem, :]) -
                                          np.dot(algebra.quat2rotation(quat), MB_tstep[body_FoR].for_vel[3:6]))

            #LM_K[FoR_dof:FoR_dof+3,FoR_dof+6:FoR_dof+10] = algebra.der_CquatT_by_v(MB_tstep[body_FoR].quat,Lambda_dot)
            LM_C[FoR_dof:FoR_dof+3,FoR_dof+6:FoR_dof+10] += algebra.der_CquatT_by_v(quat,scalingFactor*Lambda_dot[0:3])
            LM_C[FoR_dof+3:FoR_dof+6,FoR_dof+6:FoR_dof+10] -= algebra.der_CquatT_by_v(quat,scalingFactor*Lambda_dot[3:6])

            LM_K[node_dof+3:node_dof+6,node_dof+3:node_dof+6] += algebra.der_TanT_by_xv(MB_tstep[0].psi[ielem, inode_in_elem, :],scalingFactor*Lambda_dot[3:6])

            ieq += 6

        ###################################################################
        ###################  HINGE BETWEEN NODE AND FOR  ##################
        ###################################################################
        elif behaviour == 'hinge_node_FoR_constant_rotation':

            # Rename variables from dictionary
            node_in_body = MBdict["constraint_%02d" % iconstraint]['node_in_body']
            node_body = MBdict["constraint_%02d" % iconstraint]['body']
            body_FoR = MBdict["constraint_%02d" % iconstraint]['body_FoR']
            rot_vel = MBdict["constraint_%02d" % iconstraint]['rot_vel']

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
            #if True:
            Bnh[ieq:ieq+3, node_dof:node_dof+3] = -1.0*np.eye(3)
            #TODO: change this when the master AFoR is able to move
            quat = algebra.quat_bound(MB_tstep[body_FoR].quat)
            Bnh[ieq:ieq+3, FoR_dof:FoR_dof+3] = algebra.quat2rotation(quat)
            # Bnh[ieq:ieq+3, FoR_dof:FoR_dof+3] = np.eye(3)

            Bnh[3,FoR_dof+5] = 1.0

            LM_C[sys_size:,:sys_size] = scalingFactor*Bnh
            LM_C[:sys_size,sys_size:] = scalingFactor*np.transpose(Bnh)

            LM_Q[:sys_size] = scalingFactor*np.dot(np.transpose(Bnh),Lambda_dot)
            LM_Q[sys_size:sys_size+3] = -MB_tstep[0].pos_dot[-1,:] + np.dot(algebra.quat2rotation(quat),MB_tstep[1].for_vel[0:3])
            LM_Q[sys_size+3] = MB_tstep[1].for_vel[5] - rot_vel

            LM_C[FoR_dof:FoR_dof+3,FoR_dof+6:FoR_dof+10] += algebra.der_CquatT_by_v(quat,scalingFactor*Lambda_dot[0:3])

            ieq += 4

    return LM_C, LM_K, LM_Q
