'''
To use this library: import sharpy.utils.lagrangemultipliers as lagrangemultipliers
'''

import ctypes as ct
import numpy as np
import sharpy.utils.algebra as algebra
from IPython import embed


def define_num_LM_eq(MBdict):
    '''
    This function defines the number of equations needed to impose the constraints
    defined in the dictionary MBdict

    num_LM_eq = lagrangemultipliers.define_num_LM_eq(MBdict)
    '''

    num_constraints = MBdict['num_constraints']
    num_LM_eq = 0

    # Define the number of equations that I need
    for iconstraint in range(num_constraints):

        if MBdict["constraint_%02d" % iconstraint]['behaviour'] == 'hinge_node_FoR':
            num_LM_eq += 4
        elif MBdict["constraint_%02d" % iconstraint]['behaviour'] == 'free':
            num_LM_eq += 0
        elif MBdict["constraint_%02d" % iconstraint]['behaviour'] == 'hinge_FoR':
            num_LM_eq += 3
        elif MBdict["constraint_%02d" % iconstraint]['behaviour'] == 'fully_constrained_node_FoR':
            num_LM_eq += 6
        elif MBdict["constraint_%02d" % iconstraint]['behaviour'] == 'hinge_node_FoR_constant_rotation':
            num_LM_eq += 4
        else:
            print("ERROR: not recognized constraint type")

    return num_LM_eq


def generate_lagrange_matrix(MBdict, MB_beam, MB_tstep, num_LM_eq, sys_size, dt, Lambda, Lambda_dot):
    '''
    Generates the matrices associated to the Lagrange multipliers of a dictionary of "constraints"

    LM_K: matrix associate to the terms in K. Usually holonomic constraints
    LM_C: matrix associate to the terms in C. Usually non-holonomic constraints
    LM_Q: vector associated to the independent terms

    LM_C, LM_K, LM_Q = lagrangemultipliers.generate_lagrange_matrix(MBdict, MB_beam, MB_tstep, num_LM_eq, sys_size, dt, Lambda, Lambda_dot)
    '''
    # Lagrange multipliers parameters
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
            # Bnh[ieq:ieq+3, FoR_dof:FoR_dof+3] = np.eye(3)

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

            # Define the position of the first degree of freedom associated to the FoR
            FoR_dof = 0
            for ibody in range(body_FoR):
                FoR_dof += MB_beam[ibody].num_dof.value
                if MB_beam[ibody].FoR_movement == 'free':
                    FoR_dof += 10
            FoR_dof += MB_beam[body_FoR].num_dof.value

            Bnh[ieq:ieq+3, FoR_dof:FoR_dof+3] = 1.0*np.eye(3)

            LM_C[sys_size:,:sys_size] = scalingFactor*Bnh
            LM_C[:sys_size,sys_size:] = scalingFactor*np.transpose(Bnh)

            LM_Q[:sys_size] = scalingFactor*np.dot(np.transpose(Bnh),Lambda_dot)
            LM_Q[FoR_dof:FoR_dof+3] = MB_tstep[body_FoR].for_vel[0:3].astype(dtype=ct.c_double, copy=True, order='F')

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
