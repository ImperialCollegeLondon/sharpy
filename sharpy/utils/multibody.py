"""
Multibody library

Library used to manipulate multibody systems

Args:

Returns:

Examples:
    To use this library: import sharpy.utils.multibody as mb

Notes:

"""
import numpy as np
import sharpy.structure.utils.xbeamlib as xbeamlib
import sharpy.utils.algebra as algebra
from IPython import embed
import ctypes as ct
import traceback


def split_multibody(beam, tstep, mb_data_dict):
    """
    split_multibody

    This functions splits a structure at a certain time step in its different bodies

    Longer description

    Args:
    	beam (beam): structural information of the multibody system
    	tstep (StructTimeStepInfo): timestep information of the multibody system
        mb_data_dict (): Dictionary including the multibody information

    Returns:
        MB_beam (list of beam): each entry represents a body
        MB_tstep (list of StructTimeStepInfo): each entry represents a body

    Examples:

    Notes:

    """

    update_mb_db_before_split(tstep)

    MB_beam = []
    MB_tstep = []

    for ibody in range(beam.num_bodies):
        ibody_beam = None
        ibody_tstep = None
        ibody_beam = beam.get_body(ibody = ibody)
        ibody_beam.ini_info.change_to_local_AFoR(ibody)
        ibody_beam.timestep_info.change_to_local_AFoR(ibody)
        ibody_tstep = tstep.get_body(beam, ibody = ibody)
        ibody_tstep.change_to_local_AFoR(ibody)

        ibody_beam.FoR_movement = mb_data_dict['body_%02d' % ibody]['FoR_movement']

        MB_beam.append(ibody_beam)
        MB_tstep.append(ibody_tstep)

    return MB_beam, MB_tstep

def merge_multibody(MB_tstep, MB_beam, beam, tstep, mb_data_dict, dt):
    """
    merge_multibody

    This functions merges a series of bodies into a multibody system at a certain time step

    Longer description

    Args:
        MB_beam (list of beam): each entry represents a body
        MB_tstep (list of StructTimeStepInfo): each entry represents a body
    	beam (beam): structural information of the multibody system
    	tstep (StructTimeStepInfo): timestep information of the multibody system
        mb_data_dict (): Dictionary including the multibody information
        dt(int): time step

    Returns:
        beam (beam): structural information of the multibody system
    	tstep (StructTimeStepInfo): timestep information of the multibody system

    Examples:

    Notes:

    """

    update_mb_dB_before_merge(tstep, MB_tstep)

    first_node = 0
    first_elem = 0
    first_dof = 0

    for ibody in range(beam.num_bodies):
        last_node = first_node + MB_beam[ibody].num_node
        last_elem = first_elem + MB_beam[ibody].num_elem

        # Merge tstep
        MB_tstep[ibody].change_to_global_AFoR(ibody)
        tstep.pos[first_node:last_node,:] = MB_tstep[ibody].pos.astype(dtype=ct.c_double, order='F', copy=True)
        tstep.pos_dot[first_node:last_node,:] = MB_tstep[ibody].pos_dot.astype(dtype=ct.c_double, order='F', copy=True)
        tstep.psi[first_elem:last_elem,:,:] = MB_tstep[ibody].psi.astype(dtype=ct.c_double, order='F', copy=True)
        tstep.psi_dot[first_elem:last_elem,:,:] = MB_tstep[ibody].psi_dot.astype(dtype=ct.c_double, order='F', copy=True)

        # Merge states
        ibody_num_dof = MB_beam[ibody].num_dof.value
        tstep.q[first_dof:first_dof+ibody_num_dof] = MB_tstep[ibody].q[:-10].astype(dtype=ct.c_double, order='F', copy=True)
        tstep.dqdt[first_dof:first_dof+ibody_num_dof] = MB_tstep[ibody].dqdt[:-10].astype(dtype=ct.c_double, order='F', copy=True)
        tstep.dqddt[first_dof:first_dof+ibody_num_dof] = MB_tstep[ibody].dqddt[:-10].astype(dtype=ct.c_double, order='F', copy=True)
        first_dof += ibody_num_dof

        first_node += MB_beam[ibody].num_node
        first_elem += MB_beam[ibody].num_elem

    tstep.q[-10:] = MB_tstep[0].q[-10:].astype(dtype=ct.c_double, order='F', copy=True)
    tstep.dqdt[-10:] = MB_tstep[0].dqdt[-10:].astype(dtype=ct.c_double, order='F', copy=True)
    tstep.dqddt[-10:] = MB_tstep[0].dqddt[-10:].astype(dtype=ct.c_double, order='F', copy=True)

    # Define the new FoR information
    CAG = algebra.quat2rot(tstep.quat)
    tstep.for_pos = tstep.mb_FoR_pos[0,:].astype(dtype=ct.c_double, order='F', copy=True)
    tstep.for_vel[0:3] = np.dot(CAG,tstep.mb_FoR_vel[0,0:3])
    tstep.for_vel[3:6] = np.dot(CAG,tstep.mb_FoR_vel[0,3:6])
    tstep.for_acc[0:3] = np.dot(CAG,tstep.mb_FoR_acc[0,0:3])
    tstep.for_acc[3:6] = np.dot(CAG,tstep.mb_FoR_acc[0,3:6])
    tstep.quat = tstep.mb_quat[0,:].astype(dtype=ct.c_double, order='F', copy=True)

def update_mb_db_before_split(tstep):
    """
    update_mb_db_before_split

    Updates the FoR information database before split the system

    Longer description

    Args:
    	tstep (StructTimeStepInfo): timestep information of the multibody system

    Returns:

    Examples:

    Notes:
        At this point, this function does nothing, but we might need it at some point

    """

    # TODO: Right now, the Amaster FoR is not expected to move
    # when it does, the rest of FoR positions should be updated accordingly
    # right now, this function should be useless (I check it below)

    if False:
        CGAmaster = np.transpose(algebra.quat2rot(tstep.quat))

        tstep.mb_FoR_pos[0,:] = tstep.for_pos.astype(dtype=ct.c_double, order='F', copy=True)
        tstep.mb_FoR_vel[0,0:3] = np.dot(CGAmaster,tstep.for_vel[0:3])
        tstep.mb_FoR_vel[0,3:6] = np.dot(CGAmaster,tstep.for_vel[3:6])
        tstep.mb_FoR_acc[0,0:3] = np.dot(CGAmaster,tstep.for_acc[0:3])
        tstep.mb_FoR_acc[0,3:6] = np.dot(CGAmaster,tstep.for_acc[3:6])
        tstep.mb_quat[0,:] = tstep.quat.astype(dtype=ct.c_double, order='F', copy=True)
    else:
        pass

    # if not (tstep.mb_FoR_pos[0,:] == tstep.for_pos).all():
    #     print("multibody.py, mismatch in A FoR postion")
    # if not (tstep.mb_FoR_vel[0,:] == tstep.for_vel).all():
    #     print("multibody.py, mismatch in A FoR velocity")
    # if not (tstep.mb_FoR_acc[0,:] == tstep.for_acc).all():
    #     print("multibody.py, mismatch in A FoR acceleration")
    # if not (tstep.mb_quat[0,:] == tstep.quat).all():
    #     print("multibody.py, mismatch in A FoR quaternion")
        #traceback.print_stack()
        #

    # self.mb_FoR_pos[0,:] = self.for_pos.astype(dtype=ct.c_double, order='F', copy=True)
    # self.mb_FoR_vel[0,:] = self.for_vel.astype(dtype=ct.c_double, order='F', copy=True)
    # self.mb_FoR_acc[0,:] = self.for_acc.astype(dtype=ct.c_double, order='F', copy=True)
    # self.mb_quat[0,:] = self.quat.astype(dtype=ct.c_double, order='F', copy=True)

def update_mb_dB_before_merge(tstep, MB_tstep):
    """
    update_mb_db_before_merge

    Updates the FoR information database before merge the bodies

    Longer description

    Args:
    	tstep (StructTimeStepInfo): timestep information of the multibody system
        MB_tstep (list of StructTimeStepInfo): each entry represents a body

    Returns:

    Examples:

    Notes:

    """

    for ibody in range(len(MB_tstep)):

        CAslaveG = algebra.quat2rot(MB_tstep[ibody].quat)

        tstep.mb_FoR_pos[ibody,:] = MB_tstep[ibody].for_pos
        tstep.mb_FoR_vel[ibody,0:3] = np.dot(np.transpose(CAslaveG), MB_tstep[ibody].for_vel[0:3])
        tstep.mb_FoR_vel[ibody,3:6] = np.dot(np.transpose(CAslaveG), MB_tstep[ibody].for_vel[3:6])
        tstep.mb_FoR_acc[ibody,0:3] = np.dot(np.transpose(CAslaveG), MB_tstep[ibody].for_acc[0:3])
        tstep.mb_FoR_acc[ibody,3:6] = np.dot(np.transpose(CAslaveG), MB_tstep[ibody].for_acc[3:6])
        tstep.mb_quat[ibody,:] =  MB_tstep[ibody].quat.astype(dtype=ct.c_double, order='F', copy=True)


    # TODO: Is it convenient to do this?
    for ibody in range(len(MB_tstep)):
        MB_tstep[ibody].mb_FoR_pos = tstep.mb_FoR_pos.astype(dtype=ct.c_double, order='F', copy=True)
        MB_tstep[ibody].mb_FoR_vel = tstep.mb_FoR_vel.astype(dtype=ct.c_double, order='F', copy=True)
        MB_tstep[ibody].mb_FoR_acc = tstep.mb_FoR_acc.astype(dtype=ct.c_double, order='F', copy=True)
        MB_tstep[ibody].mb_quat = tstep.mb_quat.astype(dtype=ct.c_double, order='F', copy=True)

def disp2state(MB_beam, MB_tstep, q, dqdt, dqddt):
    """
    disp2state

    Fills the vector of states according to the displacements information

    Longer description

    Args:
        MB_beam (list of beam): each entry represents a body
        MB_tstep (list of StructTimeStepInfo): each entry represents a body
        q(numpy array): Vector of states
    	dqdt(numpy array): Time derivatives of states
        dqddt(numpy array): Second time derivatives of states

    Returns:

    Examples:

    Notes:

    """

    first_dof = 0
    for ibody in range(len(MB_beam)):

        ibody_num_dof = MB_beam[ibody].num_dof.value
        if (MB_beam[ibody].FoR_movement == 'prescribed'):
            xbeamlib.cbeam3_solv_disp2state(MB_beam[ibody], MB_tstep[ibody])
            q[first_dof:first_dof+ibody_num_dof]=MB_tstep[ibody].q[:-10].astype(dtype=ct.c_double, order='F', copy=True)
            dqdt[first_dof:first_dof+ibody_num_dof]=MB_tstep[ibody].dqdt[:-10].astype(dtype=ct.c_double, order='F', copy=True)
            dqddt[first_dof:first_dof+ibody_num_dof]=MB_tstep[ibody].dqddt[:-10].astype(dtype=ct.c_double, order='F', copy=True)
            first_dof += ibody_num_dof

        elif (MB_beam[ibody].FoR_movement == 'free'):
            xbeamlib.xbeam_solv_disp2state(MB_beam[ibody], MB_tstep[ibody])
            q[first_dof:first_dof+ibody_num_dof+10]=MB_tstep[ibody].q.astype(dtype=ct.c_double, order='F', copy=True)
            dqdt[first_dof:first_dof+ibody_num_dof+10]=MB_tstep[ibody].dqdt.astype(dtype=ct.c_double, order='F', copy=True)
            dqddt[first_dof:first_dof+ibody_num_dof+10]=MB_tstep[ibody].dqddt.astype(dtype=ct.c_double, order='F', copy=True)
            first_dof += ibody_num_dof + 10

        # MB_beam[ibody].timestep_info = MB_tstep[ibody].copy()

def state2disp(q, dqdt, dqddt, MB_beam, MB_tstep):
    """
    state2disp

    Recovers the displacements from the states

    Longer description

    Args:
        MB_beam (list of beam): each entry represents a body
        MB_tstep (list of StructTimeStepInfo): each entry represents a body
        q(numpy array): Vector of states
    	dqdt(numpy array): Time derivatives of states
        dqddt(numpy array): Second time derivatives of states

    Returns:

    Examples:

    Notes:

    """

    first_dof = 0
    for ibody in range(len(MB_beam)):

        ibody_num_dof = MB_beam[ibody].num_dof.value
        if (MB_beam[ibody].FoR_movement == 'prescribed'):
            MB_tstep[ibody].q[:-10] = q[first_dof:first_dof+ibody_num_dof].astype(dtype=ct.c_double, order='F', copy=True)
            MB_tstep[ibody].dqdt[:-10] = dqdt[first_dof:first_dof+ibody_num_dof].astype(dtype=ct.c_double, order='F', copy=True)
            MB_tstep[ibody].dqddt[:-10] = dqddt[first_dof:first_dof+ibody_num_dof].astype(dtype=ct.c_double, order='F', copy=True)
            xbeamlib.cbeam3_solv_state2disp(MB_beam[ibody], MB_tstep[ibody])
            first_dof += ibody_num_dof

        elif (MB_beam[ibody].FoR_movement == 'free'):
            MB_tstep[ibody].q = q[first_dof:first_dof+ibody_num_dof+10].astype(dtype=ct.c_double, order='F', copy=True)
            # dqdt[first_dof+ibody_num_dof+6:first_dof+ibody_num_dof+10] = algebra.unit_quat(dqdt[first_dof+ibody_num_dof+6:first_dof+ibody_num_dof+10])
            MB_tstep[ibody].dqdt = dqdt[first_dof:first_dof+ibody_num_dof+10].astype(dtype=ct.c_double, order='F', copy=True)
            MB_tstep[ibody].dqddt = dqddt[first_dof:first_dof+ibody_num_dof+10].astype(dtype=ct.c_double, order='F', copy=True)
            xbeamlib.xbeam_solv_state2disp(MB_beam[ibody], MB_tstep[ibody])
            # if onlyFlex:
            #     xbeamlib.cbeam3_solv_state2disp(MB_beam[ibody], MB_tstep[ibody])
            # else:
            #     xbeamlib.xbeam_solv_state2disp(MB_beam[ibody], MB_tstep[ibody])
            first_dof += ibody_num_dof + 10


    for ibody in range(len(MB_beam)):
        CAslaveG = algebra.quat2rot(MB_tstep[ibody].quat)
        # MB_tstep[0].mb_FoR_pos[ibody,:] = MB_tstep[ibody].for_pos.astype(dtype=ct.c_double, order='F', copy=True)
        MB_tstep[0].mb_FoR_vel[ibody,0:3] = np.dot(CAslaveG.T, MB_tstep[ibody].for_vel[0:3])
        MB_tstep[0].mb_FoR_vel[ibody,3:6] = np.dot(CAslaveG.T, MB_tstep[ibody].for_vel[3:6])
        MB_tstep[0].mb_FoR_acc[ibody,0:3] = np.dot(CAslaveG.T, MB_tstep[ibody].for_acc[0:3])
        MB_tstep[0].mb_FoR_acc[ibody,3:6] = np.dot(CAslaveG.T, MB_tstep[ibody].for_acc[3:6])
        MB_tstep[0].mb_quat[ibody,:] = MB_tstep[ibody].quat.astype(dtype=ct.c_double, order='F', copy=True)

    for ibody in range(len(MB_beam)):
        # MB_tstep[ibody].mb_FoR_pos = MB_tstep[0].mb_FoR_pos.astype(dtype=ct.c_double, order='F', copy=True)
        MB_tstep[ibody].mb_FoR_vel = MB_tstep[0].mb_FoR_vel.astype(dtype=ct.c_double, order='F', copy=True)
        MB_tstep[ibody].mb_FoR_acc = MB_tstep[0].mb_FoR_acc.astype(dtype=ct.c_double, order='F', copy=True)
        MB_tstep[ibody].mb_quat = MB_tstep[0].mb_quat.astype(dtype=ct.c_double, order='F', copy=True)
