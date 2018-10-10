import numpy as np
import sharpy.structure.utils.xbeamlib as xbeamlib
import sharpy.utils.algebra as algebra
from IPython import embed
import ctypes as ct
# import sharpy.utils.datastructures

# To use this library:
# import sharpy.utils.multibody as mb
import traceback

####################  ALGEBRA  ####################
def rotate_crv(crv_in, quat_in):

    quat_in_copy = algebra.quat_bound(quat_in)
    quat_from_crv_in = algebra.quat_bound(algebra.crv2quat(crv_in))
    quat_out = algebra.quat_bound(algebra.quaternion_product(quat_from_crv_in,quat_in_copy))
    crv_out = algebra.quat2crv(quat_out)

    return crv_out

def opposed_quat(quat_in):

    theta_from_quat = -2.0*np.arccos(quat_in[0])
    nv_from_quat = quat_in[1:4].copy()
    if np.linalg.norm(nv_from_quat) > tol_norm:
        nv_from_quat /= np.linalg.norm(nv_from_quat)
    opposed_quat = np.zeros((4,),)
    opposed_quat[0] = np.cos(theta_from_quat/2.0)
    opposed_quat[1:4] = np.sin(theta_from_quat/2.0)*nv_from_quat
    opposed_quat = algebra.quat_bound(opposed_quat)

    return opposed_quat

####################  MULTIBODY  ####################
def split_multibody(beam, tstep, mb_data_dict):
    """
    split_multibody

    This functions splits a structure at a certain time step in its different bodies

    Longer description

    Args:
    	beam (BaseStructure): arg1 description
    	tstep (StructTimeStepInfo): arg2 description
        mb_data_dict (): Dictionary including the multibody information

    Returns:
        MB_beam (list of BaseStructure): each entry represents a body
        MB_tstep (list of StructTimeStepInfo): each entry represents a body

    Examples:

    Notes:

    	Here you can have specifics and math descriptions.
    	To enter math mode simply write
    	.. math:: e^{i\\pi} + 1 = 0

    	Math mode supports TeX sintax but note that you should use two backslashes \\ instead of one.
    	Sphinx supports reStructuredText should you want to format text...

    """

    # call: MB_beam, MB_tstep = split_multibody(self, beam, tstep)

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

        # if ibody_tstep.quat is None:
        #     ibody_tstep.for_pos = mb_data_dict['body_%02d' % ibody]['FoR_position']
        #     ibody_tstep.for_vel = mb_data_dict['body_%02d' % ibody]['FoR_velocity']
        #     ibody_tstep.for_acc = mb_data_dict['body_%02d' % ibody]['FoR_acceleration']
        #     ibody_tstep.for_quat = mb_data_dict['body_%02d' % ibody]['FoR_quat']

        MB_beam.append(ibody_beam)
        MB_tstep.append(ibody_tstep)

    return MB_beam, MB_tstep



def update_mb_db_before_split(tstep):

    # TODO: Right now, the Amaster FoR is not expected to move
    # when it does, the rest of FoR positions should be updated accordingly
    # right now, this function should be useless (I check it below)

    CGAmaster = np.transpose(algebra.quat2rot(tstep.quat))

    tstep.mb_FoR_pos[0,:] = tstep.for_pos.astype(dtype=ct.c_double, order='F', copy=True)
    tstep.mb_FoR_vel[0,0:3] = np.dot(CGAmaster,tstep.for_vel[0:3])
    tstep.mb_FoR_vel[0,3:6] = np.dot(CGAmaster,tstep.for_vel[3:6])
    tstep.mb_FoR_acc[0,0:3] = np.dot(CGAmaster,tstep.for_acc[0:3])
    tstep.mb_FoR_acc[0,3:6] = np.dot(CGAmaster,tstep.for_acc[3:6])
    tstep.mb_quat[0,:] = tstep.quat.astype(dtype=ct.c_double, order='F', copy=True)

    # if not (tstep.mb_FoR_pos[0,:] == tstep.for_pos).all():
    #     print("multibody.py, mismatch in A FoR postion")
    # if not (tstep.mb_FoR_vel[0,:] == tstep.for_vel).all():
    #     print("multibody.py, mismatch in A FoR velocity")
    # if not (tstep.mb_FoR_acc[0,:] == tstep.for_acc).all():
    #     print("multibody.py, mismatch in A FoR acceleration")
    # if not (tstep.mb_quat[0,:] == tstep.quat).all():
    #     print("multibody.py, mismatch in A FoR quaternion")
        #traceback.print_stack()
        #embed()

    # self.mb_FoR_pos[0,:] = self.for_pos.astype(dtype=ct.c_double, order='F', copy=True)
    # self.mb_FoR_vel[0,:] = self.for_vel.astype(dtype=ct.c_double, order='F', copy=True)
    # self.mb_FoR_acc[0,:] = self.for_acc.astype(dtype=ct.c_double, order='F', copy=True)
    # self.mb_quat[0,:] = self.quat.astype(dtype=ct.c_double, order='F', copy=True)



def disp2state(MB_beam, MB_tstep, q, dqdt, dqddt):

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

    first_dof = 0
    for ibody in range(len(MB_beam)):

        #print("quat before state2disp: ", MB_tstep[ibody].quat)

        ibody_num_dof = MB_beam[ibody].num_dof.value
        if (MB_beam[ibody].FoR_movement == 'prescribed'):
            MB_tstep[ibody].q[:-10] = q[first_dof:first_dof+ibody_num_dof].astype(dtype=ct.c_double, order='F', copy=True)
            MB_tstep[ibody].dqdt[:-10] = dqdt[first_dof:first_dof+ibody_num_dof].astype(dtype=ct.c_double, order='F', copy=True)
            MB_tstep[ibody].dqddt[:-10] = dqddt[first_dof:first_dof+ibody_num_dof].astype(dtype=ct.c_double, order='F', copy=True)
            xbeamlib.cbeam3_solv_state2disp(MB_beam[ibody], MB_tstep[ibody])
            first_dof += ibody_num_dof

        elif (MB_beam[ibody].FoR_movement == 'free'):
            MB_tstep[ibody].q = q[first_dof:first_dof+ibody_num_dof+10].astype(dtype=ct.c_double, order='F', copy=True)
            MB_tstep[ibody].dqdt = dqdt[first_dof:first_dof+ibody_num_dof+10].astype(dtype=ct.c_double, order='F', copy=True)
            MB_tstep[ibody].dqddt = dqddt[first_dof:first_dof+ibody_num_dof+10].astype(dtype=ct.c_double, order='F', copy=True)
            xbeamlib.xbeam_solv_state2disp(MB_beam[ibody], MB_tstep[ibody])
            #print("quat in dqdt: ",MB_tstep[1].dqdt[-4:], "quat: ", MB_tstep[1].quat, "quat in db: ", MB_tstep[1].mb_quat[1])
            first_dof += ibody_num_dof + 10

        # print("state2disp")
        # embed()
        # MB_beam[ibody].timestep_info = MB_tstep[ibody].copy()
