"""
Multibody library

Library used to manipulate multibody systems

To use this library: import sharpy.utils.multibody as mb

"""
import numpy as np
import sharpy.structure.utils.xbeamlib as xbeamlib
import sharpy.utils.algebra as algebra
import ctypes as ct
import traceback


def split_multibody(beam, tstep, mb_data_dict, ts):
    """
    split_multibody

    This functions splits a structure at a certain time step in its different bodies

    Args:
    	beam (:class:`~sharpy.structure.models.beam.Beam`): structural information of the multibody system
    	tstep (:class:`~sharpy.utils.datastructures.StructTimeStepInfo`): timestep information of the multibody system
        mb_data_dict (dict): Dictionary including the multibody information
        ts (int): time step number

    Returns:
        MB_beam (list(:class:`~sharpy.structure.models.beam.Beam`)): each entry represents a body
        MB_tstep (list(:class:`~sharpy.utils.datastructures.StructTimeStepInfo`)): each entry represents a body
    """

    MB_beam = []
    MB_tstep = []

    quat0 = tstep.quat.astype(dtype=ct.c_double, order='F', copy=True)
    for0_pos = tstep.for_pos.astype(dtype=ct.c_double, order='F', copy=True)
    for0_vel = tstep.for_vel.astype(dtype=ct.c_double, order='F', copy=True)

    ini_quat0 = beam.ini_info.quat.astype(dtype=ct.c_double, order='F', copy=True)
    ini_for0_pos = beam.ini_info.for_pos.astype(dtype=ct.c_double, order='F', copy=True)
    ini_for0_vel = beam.ini_info.for_vel.astype(dtype=ct.c_double, order='F', copy=True)

    for ibody in range(beam.num_bodies):
        ibody_beam = None
        ibody_tstep = None
        ibody_beam = beam.get_body(ibody = ibody)
        ibody_tstep = tstep.get_body(beam, ibody_beam.num_dof, ibody = ibody)

        ibody_beam.FoR_movement = mb_data_dict['body_%02d' % ibody]['FoR_movement']

        ibody_beam.ini_info.compute_psi_local_AFoR(ini_for0_pos, ini_for0_vel, ini_quat0)
        ibody_beam.ini_info.change_to_local_AFoR(ini_for0_pos, ini_for0_vel, ini_quat0)
        if ts == 1:
            ibody_tstep.compute_psi_local_AFoR(for0_pos, for0_vel, quat0)
        ibody_tstep.change_to_local_AFoR(for0_pos, for0_vel, quat0)

        MB_beam.append(ibody_beam)
        MB_tstep.append(ibody_tstep)

    return MB_beam, MB_tstep

def merge_multibody(MB_tstep, MB_beam, beam, tstep, mb_data_dict, dt):
    """
    merge_multibody

    This functions merges a series of bodies into a multibody system at a certain time step

    Longer description

    Args:
        MB_beam (list(:class:`~sharpy.structure.models.beam.Beam`)): each entry represents a body
        MB_tstep (list(:class:`~sharpy.utils.datastructures.StructTimeStepInfo`)): each entry represents a body
    	beam (:class:`~sharpy.structure.models.beam.Beam`): structural information of the multibody system
    	tstep (:class:`~sharpy.utils.datastructures.StructTimeStepInfo`): timestep information of the multibody system
        mb_data_dict (dict): Dictionary including the multibody information
        dt(int): time step

    Returns:
        beam (:class:`~sharpy.structure.models.beam.Beam`): structural information of the multibody system
    	tstep (:class:`~sharpy.utils.datastructures.StructTimeStepInfo`): timestep information of the multibody system
    """

    update_mb_dB_before_merge(tstep, MB_tstep)

    quat0 = MB_tstep[0].quat.astype(dtype=ct.c_double, order='F', copy=True)
    for0_pos = MB_tstep[0].for_pos.astype(dtype=ct.c_double, order='F', copy=True)
    for0_vel = MB_tstep[0].for_vel.astype(dtype=ct.c_double, order='F', copy=True)

    for ibody in range(beam.num_bodies):
        MB_tstep[ibody].change_to_global_AFoR(for0_pos, for0_vel, quat0)

    first_dof = 0
    for ibody in range(beam.num_bodies):
        # Renaming for clarity
        ibody_elems = MB_beam[ibody].global_elems_num
        ibody_nodes = MB_beam[ibody].global_nodes_num

        # Merge tstep
        tstep.pos[ibody_nodes,:] = MB_tstep[ibody].pos.astype(dtype=ct.c_double, order='F', copy=True)
        tstep.pos_dot[ibody_nodes,:] = MB_tstep[ibody].pos_dot.astype(dtype=ct.c_double, order='F', copy=True)
        tstep.pos_ddot[ibody_nodes,:] = MB_tstep[ibody].pos_ddot.astype(dtype=ct.c_double, order='F', copy=True)
        tstep.psi[ibody_elems,:,:] = MB_tstep[ibody].psi.astype(dtype=ct.c_double, order='F', copy=True)
        tstep.psi_local[ibody_elems,:,:] = MB_tstep[ibody].psi_local.astype(dtype=ct.c_double, order='F', copy=True)
        tstep.psi_dot[ibody_elems,:,:] = MB_tstep[ibody].psi_dot.astype(dtype=ct.c_double, order='F', copy=True)
        tstep.psi_dot_local[ibody_elems,:,:] = MB_tstep[ibody].psi_dot_local.astype(dtype=ct.c_double, order='F', copy=True)
        tstep.psi_ddot[ibody_elems,:,:] = MB_tstep[ibody].psi_ddot.astype(dtype=ct.c_double, order='F', copy=True)
        tstep.gravity_forces[ibody_nodes,:] = MB_tstep[ibody].gravity_forces.astype(dtype=ct.c_double, order='F', copy=True)
        tstep.steady_applied_forces[ibody_nodes,:] = MB_tstep[ibody].steady_applied_forces.astype(dtype=ct.c_double, order='F', copy=True)
        tstep.unsteady_applied_forces[ibody_nodes,:] = MB_tstep[ibody].unsteady_applied_forces.astype(dtype=ct.c_double, order='F', copy=True)
        tstep.runtime_steady_forces[ibody_nodes,:] = MB_tstep[ibody].runtime_steady_forces.astype(dtype=ct.c_double, order='F', copy=True)
        tstep.runtime_unsteady_forces[ibody_nodes,:] = MB_tstep[ibody].runtime_unsteady_forces.astype(dtype=ct.c_double, order='F', copy=True)
        # TODO: Do I need a change in FoR for the following variables? Maybe for the FoR ones.
        tstep.forces_constraints_nodes[ibody_nodes,:] = MB_tstep[ibody].forces_constraints_nodes.astype(dtype=ct.c_double, order='F', copy=True)
        tstep.forces_constraints_FoR[ibody, :] = MB_tstep[ibody].forces_constraints_FoR[ibody, :].astype(dtype=ct.c_double, order='F', copy=True)

        # Merge states
        ibody_num_dof = MB_beam[ibody].num_dof.value
        tstep.q[first_dof:first_dof+ibody_num_dof] = MB_tstep[ibody].q[:-10].astype(dtype=ct.c_double, order='F', copy=True)
        tstep.dqdt[first_dof:first_dof+ibody_num_dof] = MB_tstep[ibody].dqdt[:-10].astype(dtype=ct.c_double, order='F', copy=True)
        tstep.dqddt[first_dof:first_dof+ibody_num_dof] = MB_tstep[ibody].dqddt[:-10].astype(dtype=ct.c_double, order='F', copy=True)

        tstep.mb_dquatdt[ibody, :] = MB_tstep[ibody].dqddt[-4:].astype(dtype=ct.c_double, order='F', copy=True)

        first_dof += ibody_num_dof

    tstep.q[-10:] = MB_tstep[0].q[-10:].astype(dtype=ct.c_double, order='F', copy=True)
    tstep.dqdt[-10:] = MB_tstep[0].dqdt[-10:].astype(dtype=ct.c_double, order='F', copy=True)
    tstep.dqddt[-10:] = MB_tstep[0].dqddt[-10:].astype(dtype=ct.c_double, order='F', copy=True)

    # Define the new FoR information
    tstep.for_pos = MB_tstep[0].for_pos.astype(dtype=ct.c_double, order='F', copy=True)
    tstep.for_vel = MB_tstep[0].for_vel.astype(dtype=ct.c_double, order='F', copy=True)
    tstep.for_acc = MB_tstep[0].for_acc.astype(dtype=ct.c_double, order='F', copy=True)
    tstep.quat = MB_tstep[0].quat.astype(dtype=ct.c_double, order='F', copy=True)


def update_mb_dB_before_merge(tstep, MB_tstep):
    """
    update_mb_db_before_merge

    Updates the FoR information database before merging bodies

    Args:
    	tstep (:class:`~sharpy.utils.datastructures.StructTimeStepInfo`): timestep information of the multibody system
        MB_tstep (list(:class:`~sharpy.utils.datastructures.StructTimeStepInfo`)): each entry represents a body
    """

    for ibody in range(len(MB_tstep)):

        tstep.mb_FoR_pos[ibody,:] = MB_tstep[ibody].for_pos.astype(dtype=ct.c_double, order='F', copy=True)
        tstep.mb_FoR_vel[ibody,:] = MB_tstep[ibody].for_vel.astype(dtype=ct.c_double, order='F', copy=True)
        tstep.mb_FoR_acc[ibody,:] = MB_tstep[ibody].for_acc.astype(dtype=ct.c_double, order='F', copy=True)
        tstep.mb_quat[ibody,:] =  MB_tstep[ibody].quat.astype(dtype=ct.c_double, order='F', copy=True)
        assert (MB_tstep[ibody].mb_dquatdt[ibody, :] == MB_tstep[ibody].dqddt[-4:]).all(), "Error in multibody storage"
        tstep.mb_dquatdt[ibody, :] = MB_tstep[ibody].dqddt[-4:].astype(dtype=ct.c_double, order='F', copy=True)


def disp_and_accel2state(MB_beam, MB_tstep, Lambda, Lambda_dot, sys_size, num_LM_eq):
    """
    disp2state

    Fills the vector of states according to the displacements information

    Args:
        MB_beam (list(:class:`~sharpy.structure.models.beam.Beam`)): each entry represents a body
        MB_tstep (list(:class:`~sharpy.utils.datastructures.StructTimeStepInfo`)): each entry represents a body
        Lambda(np.ndarray): Lagrange multipliers of holonomic constraints
        Lambda_dot(np.ndarray): Lagrange multipliers of non-holonomic constraints

        sys_size(int): number of degrees of freedom of the system of equations not accounting for lagrange multipliers
        num_LM_eq(int): Number of equations associated to the Lagrange Multipliers

        q(np.ndarray): Vector of states
    	dqdt(np.ndarray): Time derivatives of states
        dqddt(np.ndarray): Second time derivatives of states
    """
    q = np.zeros((sys_size + num_LM_eq, ), dtype=ct.c_double, order='F')
    dqdt = np.zeros((sys_size + num_LM_eq, ), dtype=ct.c_double, order='F')
    dqddt = np.zeros((sys_size + num_LM_eq, ), dtype=ct.c_double, order='F')

    first_dof = 0
    for ibody in range(len(MB_beam)):

        ibody_num_dof = MB_beam[ibody].num_dof.value
        if (MB_beam[ibody].FoR_movement == 'prescribed'):
            xbeamlib.cbeam3_solv_disp2state(MB_beam[ibody], MB_tstep[ibody])
            xbeamlib.cbeam3_solv_accel2state(MB_beam[ibody], MB_tstep[ibody])
            q[first_dof:first_dof+ibody_num_dof]=MB_tstep[ibody].q[:-10].astype(dtype=ct.c_double, order='F', copy=True)
            dqdt[first_dof:first_dof+ibody_num_dof]=MB_tstep[ibody].dqdt[:-10].astype(dtype=ct.c_double, order='F', copy=True)
            dqddt[first_dof:first_dof+ibody_num_dof]=MB_tstep[ibody].dqddt[:-10].astype(dtype=ct.c_double, order='F', copy=True)
            first_dof += ibody_num_dof

        elif (MB_beam[ibody].FoR_movement == 'free'):
            dquatdt = MB_tstep[ibody].mb_dquatdt[ibody, :].astype(dtype=ct.c_double, order='F', copy=True)
            xbeamlib.xbeam_solv_disp2state(MB_beam[ibody], MB_tstep[ibody])
            xbeamlib.xbeam_solv_accel2state(MB_beam[ibody], MB_tstep[ibody])
            q[first_dof:first_dof+ibody_num_dof+10]=MB_tstep[ibody].q.astype(dtype=ct.c_double, order='F', copy=True)
            dqdt[first_dof:first_dof+ibody_num_dof+10]=MB_tstep[ibody].dqdt.astype(dtype=ct.c_double, order='F', copy=True)
            dqddt[first_dof:first_dof+ibody_num_dof+10]=MB_tstep[ibody].dqddt.astype(dtype=ct.c_double, order='F', copy=True)
            dqddt[first_dof+ibody_num_dof+6:first_dof+ibody_num_dof+10]=dquatdt.astype(dtype=ct.c_double, order='F', copy=True)
            first_dof += ibody_num_dof + 10

    if num_LM_eq > 0:
        q[first_dof:] = Lambda.astype(dtype=ct.c_double, order='F', copy=True)
        dqdt[first_dof:] = Lambda_dot.astype(dtype=ct.c_double, order='F', copy=True)

    return q, dqdt, dqddt

def state2disp_and_accel(q, dqdt, dqddt, MB_beam, MB_tstep, num_LM_eq):
    """
    state2disp

    Recovers the displacements from the states

    Longer description

    Args:
        MB_beam (list(:class:`~sharpy.structure.models.beam.Beam`)): each entry represents a body
        MB_tstep (list(:class:`~sharpy.utils.datastructures.StructTimeStepInfo`)): each entry represents a body
        q(np.ndarray): Vector of states
    	dqdt(np.ndarray): Time derivatives of states
        dqddt(np.ndarray): Second time derivatives of states

        num_LM_eq(int): Number of equations associated to the Lagrange Multipliers

        Lambda(np.ndarray): Lagrange multipliers of holonomic constraints
        Lambda_dot(np.ndarray): Lagrange multipliers of non-holonomic constraints

    """

    Lambda = np.zeros((num_LM_eq, ), dtype=ct.c_double, order='F')
    Lambda_dot = np.zeros((num_LM_eq, ), dtype=ct.c_double, order='F')

    first_dof = 0
    for ibody in range(len(MB_beam)):

        ibody_num_dof = MB_beam[ibody].num_dof.value
        if (MB_beam[ibody].FoR_movement == 'prescribed'):
            MB_tstep[ibody].q[:-10] = q[first_dof:first_dof+ibody_num_dof].astype(dtype=ct.c_double, order='F', copy=True)
            MB_tstep[ibody].dqdt[:-10] = dqdt[first_dof:first_dof+ibody_num_dof].astype(dtype=ct.c_double, order='F', copy=True)
            MB_tstep[ibody].dqddt[:-10] = dqddt[first_dof:first_dof+ibody_num_dof].astype(dtype=ct.c_double, order='F', copy=True)
            xbeamlib.cbeam3_solv_state2disp(MB_beam[ibody], MB_tstep[ibody])
            xbeamlib.cbeam3_solv_state2accel(MB_beam[ibody], MB_tstep[ibody])
            first_dof += ibody_num_dof

        elif (MB_beam[ibody].FoR_movement == 'free'):
            MB_tstep[ibody].q = q[first_dof:first_dof+ibody_num_dof+10].astype(dtype=ct.c_double, order='F', copy=True)
            MB_tstep[ibody].dqdt = dqdt[first_dof:first_dof+ibody_num_dof+10].astype(dtype=ct.c_double, order='F', copy=True)
            MB_tstep[ibody].dqddt = dqddt[first_dof:first_dof+ibody_num_dof+10].astype(dtype=ct.c_double, order='F', copy=True)
            MB_tstep[ibody].mb_dquatdt[ibody, :] = MB_tstep[ibody].dqddt[-4:]
            xbeamlib.xbeam_solv_state2disp(MB_beam[ibody], MB_tstep[ibody])
            xbeamlib.xbeam_solv_state2accel(MB_beam[ibody], MB_tstep[ibody])
            first_dof += ibody_num_dof + 10

    Lambda = q[first_dof:].astype(dtype=ct.c_double, order='F', copy=True)
    Lambda_dot = dqdt[first_dof:].astype(dtype=ct.c_double, order='F', copy=True)

    return Lambda, Lambda_dot

def get_elems_nodes_list(beam, ibody):
    """
        get_elems_nodes_list

        This function returns the elements (``ibody_elements``) and the nodes
        (``ibody_nodes``) that belong to the body number ``ibody``

        Args:
    	   beam (:class:`~sharpy.structure.models.beam.Beam`): structural information of the multibody system
           ibody (int): Body number about which the information is required

        Returns:
            ibody_elements (list): List of elements that belong the ``ibody``
            ibody_nodes (list): List of nodes that belong the ``ibody``

    """
    int_list = np.arange(beam.num_elem)
    ibody_elements = int_list[beam.body_number == ibody]
    ibody_nodes = np.sort(np.unique(beam.connectivities[ibody_elements, :].reshape(-1)))

    return ibody_elements, ibody_nodes
