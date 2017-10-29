import ctypes as ct
import numpy as np
import scipy as sc
import scipy.integrate

import sharpy.utils.algebra as algebra
import sharpy.utils.ctypes_utils as ct_utils
from sharpy.utils.sharpydir import SharpyDir
from sharpy.utils.datastructures import StructTimeStepInfo
import sharpy.utils.cout_utils as cout
import sharpy.utils.exceptions as exc


class Xbopts(ct.Structure):
    """Structure skeleton for options input in xbeam

    """
    _fields_ = [("FollowerForce", ct.c_bool),
                ("FollowerForceRig", ct.c_bool),
                ("PrintInfo", ct.c_bool),
                ("OutInBframe", ct.c_bool),
                ("OutInaframe", ct.c_bool),
                ("ElemProj", ct.c_int),
                ("MaxIterations", ct.c_int),
                ("NumLoadSteps", ct.c_int),
                ("NumGauss", ct.c_int),
                ("Solution", ct.c_int),
                ("DeltaCurved", ct.c_double),
                ("MinDelta", ct.c_double),
                ("NewmarkDamp", ct.c_double),
                ("gravity_on", ct.c_bool),
                ("gravity", ct.c_double),
                ("gravity_dir_x", ct.c_double),
                ("gravity_dir_y", ct.c_double),
                ("gravity_dir_z", ct.c_double)
                ]

    def __init__(self):
        ct.Structure.__init__(self)
        self.FollowerForce = ct.c_bool(True)
        self.FollowerForceRig = ct.c_bool(True)
        self.PrintInfo = ct.c_bool(True)
        self.OutInBframe = ct.c_bool(True)
        self.OutInaframe = ct.c_bool(False)
        self.ElemProj = ct.c_int(0)
        self.MaxIterations = ct.c_int(99)
        self.NumLoadSteps = ct.c_int(5)
        self.Solution = ct.c_int(111)
        self.DeltaCurved = ct.c_double(1.0e-5)
        self.MinDelta = ct.c_double(1.0e-8)
        self.NewmarkDamp = ct.c_double(0.0)
        self.gravity_on = ct.c_bool(False)
        self.gravity = ct.c_double(0.0)
        self.gravity_dir_x = ct.c_double(0.0)
        self.gravity_dir_y = ct.c_double(0.0)
        self.gravity_dir_z = ct.c_double(1.0)


xbeamlib = ct_utils.import_ctypes_lib(SharpyDir + '/lib/', 'libxbeam')

# ctypes pointer types
doubleP = ct.POINTER(ct.c_double)
intP = ct.POINTER(ct.c_int)
charP = ct.POINTER(ct.c_char_p)


f_cbeam3_solv_nlnstatic = xbeamlib.cbeam3_solv_nlnstatic_python
f_cbeam3_solv_nlnstatic.restype = None


def cbeam3_solv_nlnstatic(beam, settings, ts):
    """@brief Python wrapper for f_cbeam3_solv_nlnstatic
     Alfonso del Carre
    """
    n_elem = ct.c_int(beam.num_elem)
    n_nodes = ct.c_int(beam.num_node)
    n_mass = ct.c_int(beam.n_mass)
    n_stiff = ct.c_int(beam.n_stiff)

    xbopts = Xbopts()
    xbopts.PrintInfo = ct.c_bool(settings['print_info'])
    xbopts.Solution = ct.c_int(112)
    # xbopts.OutInaframe = ct.c_bool(settings['out_a_frame'])
    # xbopts.OutInBframe = ct.c_bool(settings['out_b_frame'])
    # xbopts.ElemProj = settings['elem_proj']
    xbopts.MaxIterations = settings['max_iterations']
    xbopts.NumLoadSteps = settings['num_load_steps']
    # xbopts.NumGauss = ct.c_int(0)
    xbopts.DeltaCurved = settings['delta_curved']
    # xbopts.MinDelta = settings['min_delta']
    # xbopts.NewmarkDamp = settings['newmark_damp']
    xbopts.gravity_on = settings['gravity_on']
    xbopts.gravity = settings['gravity']
    xbopts.gravity_dir_x = ct.c_double(settings['gravity_dir'][0])
    xbopts.gravity_dir_y = ct.c_double(settings['gravity_dir'][1])
    xbopts.gravity_dir_z = ct.c_double(settings['gravity_dir'][2])

    # here we only need to set the flags at True, all the forces are follower
    xbopts.FollowerForce = ct.c_bool(True)
    xbopts.FollowerForceRig = ct.c_bool(True)

    f_cbeam3_solv_nlnstatic(ct.byref(n_elem),
                            ct.byref(n_nodes),
                            beam.fortran['num_nodes'].ctypes.data_as(intP),
                            beam.fortran['num_mem'].ctypes.data_as(intP),
                            beam.fortran['connectivities'].ctypes.data_as(intP),
                            beam.fortran['master'].ctypes.data_as(intP),
                            ct.byref(n_mass),
                            beam.fortran['mass'].ctypes.data_as(doubleP),
                            beam.fortran['mass_indices'].ctypes.data_as(intP),
                            ct.byref(n_stiff),
                            beam.fortran['stiffness'].ctypes.data_as(doubleP),
                            beam.fortran['inv_stiffness'].ctypes.data_as(doubleP),
                            beam.fortran['stiffness_indices'].ctypes.data_as(intP),
                            beam.fortran['frame_of_reference_delta'].ctypes.data_as(doubleP),
                            beam.fortran['rbmass'].ctypes.data_as(doubleP),
                            beam.fortran['node_master_elem'].ctypes.data_as(intP),
                            beam.fortran['vdof'].ctypes.data_as(intP),
                            beam.fortran['fdof'].ctypes.data_as(intP),
                            ct.byref(xbopts),
                            beam.ini_info.pos.ctypes.data_as(doubleP),
                            beam.ini_info.psi.ctypes.data_as(doubleP),
                            beam.timestep_info[ts].pos.ctypes.data_as(doubleP),
                            beam.timestep_info[ts].psi.ctypes.data_as(doubleP),
                            beam.timestep_info[ts].steady_applied_forces.ctypes.data_as(doubleP)
                            )


f_cbeam3_solv_modal = xbeamlib.cbeam3_solv_modal_python
f_cbeam3_solv_modal.restype = None


def cbeam3_solv_modal(beam, settings):
    """@brief Python wrapper for cbeam3_solv_modal

     Alfonso del Carre"""
    n_elem = ct.c_int(beam.num_elem)
    n_nodes = ct.c_int(beam.num_node)
    n_mass = ct.c_int(beam.n_mass)
    n_stiff = ct.c_int(beam.n_stiff)

    xbopts = Xbopts()
    xbopts.PrintInfo = ct.c_bool(settings['print_info'])
    xbopts.Solution = ct.c_int(103)
    xbopts.OutInaframe = ct.c_bool(settings['out_a_frame'])
    xbopts.OutInBframe = ct.c_bool(settings['out_b_frame'])
    xbopts.ElemProj = settings['elem_proj']
    xbopts.MaxIterations = settings['max_iterations']
    xbopts.NumLoadSteps = settings['num_load_steps']
    xbopts.NumGauss = ct.c_int(0)
    xbopts.DeltaCurved = settings['delta_curved']
    xbopts.MinDelta = settings['min_delta']
    xbopts.NewmarkDamp = settings['newmark_damp']

    # applied forces as 0=G, 1=a, 2=b
    # here we only need to set the flags at True, all the forces are follower
    xbopts.FollowerForce = ct.c_bool(True)
    xbopts.FollowerForceRig = ct.c_bool(True)

    num_dof = sum(beam.vdof > 0)*6
    fullM = np.zeros((num_dof, num_dof), order='F')
    fullK = np.zeros((num_dof, num_dof), order='F')
    num_dof = ct.c_int(num_dof)

    f_cbeam3_solv_modal    (ct.byref(n_elem),
                            ct.byref(n_nodes),
                            ct.byref(num_dof),
                            beam.num_nodes_matrix.ctypes.data_as(intP),
                            beam.num_mem_matrix.ctypes.data_as(intP),
                            beam.connectivities_fortran.ctypes.data_as(intP),
                            beam.master_fortran.ctypes.data_as(intP),
                            ct.byref(n_mass),
                            beam.mass_matrix.ctypes.data_as(doubleP),
                            beam.mass_indices.ctypes.data_as(intP),
                            ct.byref(n_stiff),
                            beam.stiffness_matrix.ctypes.data_as(doubleP),
                            beam.inv_stiffness_db.ctypes.data_as(doubleP),
                            beam.stiffness_indices.ctypes.data_as(intP),
                            beam.frame_of_reference_delta.ctypes.data_as(doubleP),
                            beam.rbmass_fortran.ctypes.data_as(doubleP),
                            beam.node_master_elem_fortran.ctypes.data_as(intP),
                            beam.vdof.ctypes.data_as(intP),
                            beam.fdof.ctypes.data_as(intP),
                            ct.byref(xbopts),
                            beam.pos_ini.ctypes.data_as(doubleP),
                            beam.psi_ini.ctypes.data_as(doubleP),
                            beam.pos_def.ctypes.data_as(doubleP),
                            beam.psi_def.ctypes.data_as(doubleP),
                            fullM.ctypes.data_as(doubleP),
                            fullK.ctypes.data_as(doubleP)
                            )

    import scipy.linalg as la
    import sharpy.utils.num_utils as num_utils
    if not num_utils.check_symmetric(fullM):
        raise ArithmeticError
    if not num_utils.check_symmetric(fullK):
        raise ArithmeticError

    beam.w, beam.v = la.eigh(fullK, fullM)


f_cbeam3_solv_nlndyn = xbeamlib.cbeam3_solv_nlndyn_python
f_cbeam3_solv_nlndyn.restype = None


def cbeam3_solv_nlndyn(beam, settings):

    n_elem = ct.c_int(beam.num_elem)
    n_nodes = ct.c_int(beam.num_node)
    n_mass = ct.c_int(beam.n_mass)
    n_stiff = ct.c_int(beam.n_stiff)

    it = beam.it

    dt = settings['dt'].value
    n_tsteps = settings['num_steps'].value
    time = np.zeros((n_tsteps,), dtype=ct.c_double, order='F')
    for i in range(n_tsteps):
        time[i] = i*dt

    # deformation history matrices
    pos_def_history = np.zeros((n_tsteps, beam.num_node, 3), order='F')
    pos_dot_def_history = np.zeros((n_tsteps, beam.num_node, 3), order='F')
    psi_def_history = np.zeros((n_tsteps, beam.num_elem, 3, 3), order='F')
    psi_dot_def_history = np.zeros((n_tsteps, beam.num_elem, 3, 3), order='F')

    dt = ct.c_double(dt)
    n_tsteps = ct.c_int(n_tsteps)

    xbopts = Xbopts()
    xbopts.PrintInfo = ct.c_bool(settings['print_info'])
    xbopts.Solution = ct.c_int(312)
    xbopts.OutInaframe = ct.c_bool(settings['out_a_frame'])
    xbopts.OutInBframe = ct.c_bool(settings['out_b_frame'])
    xbopts.ElemProj = settings['elem_proj']
    xbopts.MaxIterations = settings['max_iterations']
    xbopts.NumLoadSteps = settings['num_load_steps']
    xbopts.NumGauss = ct.c_int(0)
    xbopts.DeltaCurved = settings['delta_curved']
    xbopts.MinDelta = settings['min_delta']
    xbopts.NewmarkDamp = settings['newmark_damp']
    xbopts.gravity_on = settings['gravity_on']
    xbopts.gravity = settings['gravity']
    xbopts.gravity_dir_x = ct.c_double(settings['gravity_dir'][0])
    xbopts.gravity_dir_y = ct.c_double(settings['gravity_dir'][1])
    xbopts.gravity_dir_z = ct.c_double(settings['gravity_dir'][2])

    # here we only need to set the flags at True, all the forces are follower
    xbopts.FollowerForce = ct.c_bool(True)
    xbopts.FollowerForceRig = ct.c_bool(True)

    f_cbeam3_solv_nlndyn(ct.byref(n_elem),
                         ct.byref(n_nodes),
                         ct.byref(n_tsteps),
                         time.ctypes.data_as(doubleP),
                         beam.num_nodes_matrix.ctypes.data_as(intP),
                         beam.num_mem_matrix.ctypes.data_as(intP),
                         beam.connectivities_fortran.ctypes.data_as(intP),
                         beam.master_fortran.ctypes.data_as(intP),
                         ct.byref(n_mass),
                         beam.mass_matrix.ctypes.data_as(doubleP),
                         beam.mass_indices.ctypes.data_as(intP),
                         ct.byref(n_stiff),
                         beam.stiffness_matrix.ctypes.data_as(doubleP),
                         beam.inv_stiffness_db.ctypes.data_as(doubleP),
                         beam.stiffness_indices.ctypes.data_as(intP),
                         beam.frame_of_reference_delta.ctypes.data_as(doubleP),
                         beam.rbmass_fortran.ctypes.data_as(doubleP),
                         beam.node_master_elem_fortran.ctypes.data_as(intP),
                         beam.vdof.ctypes.data_as(intP),
                         beam.fdof.ctypes.data_as(intP),
                         ct.byref(xbopts),
                         beam.pos_ini.ctypes.data_as(doubleP),
                         beam.psi_ini.ctypes.data_as(doubleP),
                         beam.timestep_info[it].pos_def.ctypes.data_as(doubleP),
                         beam.timestep_info[it].psi_def.ctypes.data_as(doubleP),
                         beam.app_forces_fortran.ctypes.data_as(doubleP),
                         beam.dynamic_forces_amplitude_fortran.ctypes.data_as(doubleP),
                         beam.dynamic_forces_time_fortran.ctypes.data_as(doubleP),
                         beam.forced_vel_fortran.ctypes.data_as(doubleP),
                         beam.forced_acc_fortran.ctypes.data_as(doubleP),
                         pos_def_history.ctypes.data_as(doubleP),
                         psi_def_history.ctypes.data_as(doubleP),
                         pos_dot_def_history.ctypes.data_as(doubleP),
                         psi_dot_def_history.ctypes.data_as(doubleP)
                         )

    for i in range(1, n_tsteps.value):
        beam.timestep_info.append(StructTimeStepInfo(beam.num_node,
                                                     beam.num_elem,
                                                     beam.num_node_elem))
        beam.timestep_info[i].pos_def[:] = pos_def_history[i, :]
        beam.timestep_info[i].psi_def[:] = psi_def_history[i, :]
        beam.timestep_info[i].pos_dot_def[:] = pos_dot_def_history[i, :]
        beam.timestep_info[i].psi_dot_def[:] = psi_dot_def_history[i, :]


f_xbeam_solv_couplednlndyn = xbeamlib.xbeam_solv_couplednlndyn_python
f_xbeam_solv_couplednlndyn.restype = None


def xbeam_solv_couplednlndyn(beam, settings):
    n_elem = ct.c_int(beam.num_elem)
    n_nodes = ct.c_int(beam.num_node)
    n_mass = ct.c_int(beam.n_mass)
    n_stiff = ct.c_int(beam.n_stiff)

    dt = settings['dt'].value
    n_tsteps = settings['num_steps'].value
    time = np.zeros((n_tsteps,), dtype=ct.c_double, order='F')
    for i in range(n_tsteps):
        time[i] = i*dt

    # deformation history matrices
    for_vel = np.zeros((n_tsteps + 1, 6), order='F')
    for_acc = np.zeros((n_tsteps + 1, 6), order='F')
    quat_history = np.zeros((n_tsteps, 4), order='F')

    dt = ct.c_double(dt)
    n_tsteps = ct.c_int(n_tsteps)

    xbopts = Xbopts()
    xbopts.PrintInfo = ct.c_bool(settings['print_info'])
    xbopts.Solution = ct.c_int(910)
    xbopts.OutInaframe = ct.c_bool(False)
    xbopts.MaxIterations = settings['max_iterations']
    xbopts.NumLoadSteps = settings['num_load_steps']
    # xbopts.NumGauss = ct.c_int(0)
    xbopts.DeltaCurved = settings['delta_curved']
    xbopts.MinDelta = settings['min_delta']
    xbopts.NewmarkDamp = settings['newmark_damp']
    xbopts.gravity_on = settings['gravity_on']
    xbopts.gravity = settings['gravity']
    xbopts.gravity_dir_x = ct.c_double(settings['gravity_dir'][0])
    xbopts.gravity_dir_y = ct.c_double(settings['gravity_dir'][1])
    xbopts.gravity_dir_z = ct.c_double(settings['gravity_dir'][2])

    pos_def_history = np.zeros((n_tsteps.value, beam.num_node, 3), order='F', dtype=ct.c_double)
    pos_dot_def_history = np.zeros((n_tsteps.value, beam.num_node, 3), order='F', dtype=ct.c_double)
    psi_def_history = np.zeros((n_tsteps.value, beam.num_elem, 3, 3), order='F', dtype=ct.c_double)
    psi_dot_def_history = np.zeros((n_tsteps.value, beam.num_elem, 3, 3), order='F', dtype=ct.c_double)

    dynamic_force = np.zeros((n_nodes.value, 6, n_tsteps.value), dtype=ct.c_double, order='F')
    for it in range(n_tsteps.value):
        dynamic_force[:, :, it] = beam.dynamic_input[it]['dynamic_forces']

    # status flag
    success = ct.c_bool(True)

    # here we only need to set the flags at True, all the forces are follower
    xbopts.FollowerForce = ct.c_bool(True)
    xbopts.FollowerForceRig = ct.c_bool(True)
    import time as ti
    start_time = ti.time()
    f_xbeam_solv_couplednlndyn(ct.byref(n_elem),
                               ct.byref(n_nodes),
                               ct.byref(n_tsteps),
                               time.ctypes.data_as(doubleP),
                               beam.fortran['num_nodes'].ctypes.data_as(intP),
                               beam.fortran['num_mem'].ctypes.data_as(intP),
                               beam.fortran['connectivities'].ctypes.data_as(intP),
                               beam.fortran['master'].ctypes.data_as(intP),
                               ct.byref(n_mass),
                               beam.fortran['mass'].ctypes.data_as(doubleP),
                               beam.fortran['mass_indices'].ctypes.data_as(intP),
                               ct.byref(n_stiff),
                               beam.fortran['stiffness'].ctypes.data_as(doubleP),
                               beam.fortran['inv_stiffness'].ctypes.data_as(doubleP),
                               beam.fortran['stiffness_indices'].ctypes.data_as(intP),
                               beam.fortran['frame_of_reference_delta'].ctypes.data_as(doubleP),
                               beam.fortran['rbmass'].ctypes.data_as(doubleP),
                               beam.fortran['node_master_elem'].ctypes.data_as(intP),
                               beam.fortran['vdof'].ctypes.data_as(intP),
                               beam.fortran['fdof'].ctypes.data_as(intP),
                               ct.byref(xbopts),
                               beam.ini_info.pos.ctypes.data_as(doubleP),
                               beam.ini_info.psi.ctypes.data_as(doubleP),
                               beam.timestep_info[0].steady_applied_forces.ctypes.data_as(doubleP),
                               dynamic_force.ctypes.data_as(doubleP),
                               for_vel.ctypes.data_as(doubleP),
                               for_acc.ctypes.data_as(doubleP),
                               pos_def_history.ctypes.data_as(doubleP),
                               psi_def_history.ctypes.data_as(doubleP),
                               pos_dot_def_history.ctypes.data_as(doubleP),
                               psi_dot_def_history.ctypes.data_as(doubleP),
                               quat_history.ctypes.data_as(doubleP),
                               ct.byref(success))
    cout.cout_wrap("\n--- %s seconds ---" % (ti.time() - start_time), 1)
    if not success:
        raise Exception('couplednlndyn did not converge')

    for_pos = np.zeros_like(for_vel)
    for_pos[:, 0] = sc.integrate.cumtrapz(for_vel[:, 0], dx=dt.value, initial=0)
    for_pos[:, 1] = sc.integrate.cumtrapz(for_vel[:, 1], dx=dt.value, initial=0)
    for_pos[:, 2] = sc.integrate.cumtrapz(for_vel[:, 2], dx=dt.value, initial=0)

    glob_pos_def = np.zeros_like(pos_def_history)
    for it in range(n_tsteps.value):
        rot = algebra.quat2rot(quat_history[it, :]).transpose()
        for inode in range(beam.num_node):
            glob_pos_def[it, inode, :] = np.dot(rot.T, pos_def_history[it, inode, :])

    for i in range(n_tsteps.value - 1):
        beam.timestep_info[i + 1].pos[:] = pos_def_history[i+1, :]
        beam.timestep_info[i + 1].psi[:] = psi_def_history[i+1, :]
        beam.timestep_info[i + 1].pos_dot[:] = pos_dot_def_history[i+1, :]
        beam.timestep_info[i + 1].psi_dot[:] = psi_dot_def_history[i+1, :]

        beam.timestep_info[i + 1].quat[:] = quat_history[i+1, :]
        beam.timestep_info[i + 1].for_pos[:] = for_pos[i+1, :]
        beam.timestep_info[i + 1].for_vel[:] = for_vel[i+1, :]


f_cbeam3_solv_update_static = xbeamlib.cbeam3_solv_update_static_python
f_cbeam3_solv_update_static.restype = None


def xbeam_step_couplednlndyn(beam, settings):
    n_elem = ct.c_int(beam.num_elem)
    n_nodes = ct.c_int(beam.num_node)
    n_mass = ct.c_int(beam.n_mass)
    n_stiff = ct.c_int(beam.n_stiff)

    dt = settings['dt'].value
    # n_tsteps = settings['num_steps'].value
    time = ct.c_double(beam.t)
    i_iter = ct.c_int(beam.it)

    # deformation history matrices
    beam.for_vel = np.zeros((6,), order='F')
    beam.for_acc = np.zeros((6,), order='F')
    # beam.pos_def_history = np.zeros((n_tsteps, beam.num_node, 3), order='F')
    # beam.pos_dot_def_history = np.zeros((n_tsteps, beam.num_node, 3), order='F')
    # beam.psi_def_history = np.zeros((n_tsteps, beam.num_elem, 3, 3), order='F')
    # beam.psi_dot_def_history = np.zeros((n_tsteps, beam.num_elem, 3, 3), order='F')
    beam.quat = np.zeros((4,), order='F')

    dt = ct.c_double(dt)

    xbopts = Xbopts()
    xbopts.PrintInfo = ct.c_bool(settings['print_info'])
    xbopts.Solution = ct.c_int(0)
    xbopts.OutInaframe = ct.c_bool(settings['out_a_frame'])
    xbopts.OutInBframe = ct.c_bool(settings['out_b_frame'])
    xbopts.ElemProj = settings['elem_proj']
    xbopts.MaxIterations = settings['max_iterations']
    xbopts.NumLoadSteps = settings['num_load_steps']
    xbopts.NumGauss = ct.c_int(0)
    xbopts.DeltaCurved = settings['delta_curved']
    xbopts.MinDelta = settings['min_delta']
    xbopts.NewmarkDamp = settings['newmark_damp']
    xbopts.gravity_on = settings['gravity_on']
    xbopts.gravity = settings['gravity']
    # TODO add gravity orientation update
    xbopts.gravity_dir_x = ct.c_double(settings['gravity_dir'][0])
    xbopts.gravity_dir_y = ct.c_double(settings['gravity_dir'][1])
    xbopts.gravity_dir_z = ct.c_double(settings['gravity_dir'][2])

    # pos_def_history = np.zeros((n_tsteps.value, beam.num_node, 3), order='F', dtype=ct.c_double)
    # pos_dot_def_history = np.zeros((n_tsteps.value, beam.num_node, 3), order='F', dtype=ct.c_double)
    # psi_def_history = np.zeros((n_tsteps.value, beam.num_elem, 3, 3), order='F', dtype=ct.c_double)
    # psi_dot_def_history = np.zeros((n_tsteps.value, beam.num_elem, 3, 3), order='F', dtype=ct.c_double)

    # status flag
    success = ct.c_bool(True)

    # here we only need to set the flags at True, all the forces are follower
    xbopts.FollowerForce = ct.c_bool(True)
    xbopts.FollowerForceRig = ct.c_bool(True)
    import time as ti
    start_time = ti.time()
    if beam.it < 1:
        prev_pos_def = beam.pos_ini.ctypes.data_as(doubleP)
        prev_psi_def = beam.psi_ini.ctypes.data_as(doubleP)
        prev_pos_def_dot = np.zeros_like(prev_pos_def)
        prev_pos_def_dot = prev_pos_def_dot.ctypes.data_as(doubleP)
        prev_psi_def_dot = np.zeros_like(prev_psi_def)
        prev_psi_def_dot = prev_psi_def_dot.ctypes.data_as(doubleP)
        prev_quat = beam.timestep_info[beam.it].quat.copy()
        prev_quat = prev_quat.ctypes.data_as(doubleP)
        prev_for_vel = np.zeros((6,)).ctypes.data_as(doubleP),
        prev_for_acc = np.zeros((6,)).ctypes.data_as(doubleP),
    else:
        prev_pos_def = beam.timestep_info[beam.it - 1].pos_def.ctypes.data_as(doubleP),
        prev_psi_def = beam.timestep_info[beam.it - 1].psi_def.ctypes.data_as(doubleP),
        prev_pos_def_dot = beam.timestep_info[beam.it - 1].pos_def_dot.ctypes.data_as(doubleP),
        prev_psi_def_dot = beam.timestep_info[beam.it - 1].psi_def_dot.ctypes.data_as(doubleP),
        prev_quat = beam.timestep_info[beam.it - 1].quat.ctypes.data_as(doubleP),
        prev_for_vel = beam.timestep_info[beam.it - 1].for_vel.ctypes.data_as(doubleP),
        prev_for_acc = beam.timestep_info[beam.it - 1].for_acc.ctypes.data_as(doubleP),
    f_xbeam_solv_couplednlndyn(ct.byref(dt),
                               ct.byref(i_iter),
                               ct.byref(time),
                               ct.byref(n_elem),
                               ct.byref(n_nodes),
                               beam.num_nodes_matrix.ctypes.data_as(intP),
                               beam.num_mem_matrix.ctypes.data_as(intP),
                               beam.connectivities_fortran.ctypes.data_as(intP),
                               beam.master_fortran.ctypes.data_as(intP),
                               ct.byref(n_mass),
                               beam.mass_matrix.ctypes.data_as(doubleP),
                               beam.mass_indices.ctypes.data_as(intP),
                               ct.byref(n_stiff),
                               beam.stiffness_matrix.ctypes.data_as(doubleP),
                               beam.inv_stiffness_db.ctypes.data_as(doubleP),
                               beam.stiffness_indices.ctypes.data_as(intP),
                               beam.frame_of_reference_delta.ctypes.data_as(doubleP),
                               beam.rbmass_fortran.ctypes.data_as(doubleP),
                               beam.node_master_elem_fortran.ctypes.data_as(intP),
                               beam.vdof.ctypes.data_as(intP),
                               beam.fdof.ctypes.data_as(intP),
                               ct.byref(xbopts),
                               beam.pos_ini.ctypes.data_as(doubleP),
                               beam.psi_ini.ctypes.data_as(doubleP),
                               beam.timestep_info[beam.it].pos_def.ctypes.data_as(doubleP),
                               beam.timestep_info[beam.it].psi_def.ctypes.data_as(doubleP),
                               beam.timestep_info[beam.it].pos_def_dot.ctypes.data_as(doubleP),
                               beam.timestep_info[beam.it].psi_def_dot.ctypes.data_as(doubleP),
                               prev_pos_def,
                               prev_psi_def,
                               prev_pos_def_dot,
                               prev_psi_def_dot,
                               prev_quat,
                               beam.app_forces_fortran.ctypes.data_as(doubleP),
                               beam.dynamic_forces_fortran.ctypes.data_as(doubleP),
                               beam.timestep_info[beam.it].quat.ctypes.data_as(doubleP),
                               prev_for_vel,
                               prev_for_acc,
                               beam.timestep_info[beam.it].for_vel.ctypes.data_as(doubleP),
                               beam.timestep_info[beam.it].for_acc.ctypes.data_as(doubleP),
                               ct.byref(success)
                               )
    print("\n--- %s seconds ---" % (ti.time() - start_time))
    if not success:
        raise Exception('couplednlndyn did not converge')

    beam.timestep_info[beam.it].for_pos[0] = sc.integrate.cumtrapz(beam.timestep_info[beam.it].for_vel[:, 0], dx=dt.value, initial=0)
    beam.timestep_info[beam.it].for_pos[1] = sc.integrate.cumtrapz(beam.timestep_info[beam.it].for_vel[:, 1], dx=dt.value, initial=0)
    beam.timestep_info[beam.it].for_pos[2] = sc.integrate.cumtrapz(beam.timestep_info[beam.it].for_vel[:, 2], dx=dt.value, initial=0)

    # glob_pos_def = np.zeros_like(pos_def_history)
    # for it in range(n_tsteps.value):
    #     rot = algebra.quat2rot(beam.quat_history[it, :])
    #     for inode in range(beam.num_node):
    #         glob_pos_def[it, inode, :] = beam.for_pos[it, 0:3] + np.dot(rot.T, pos_def_history[it, inode, :])

    # beam.timestep_info[0] = (StructTimeStepInfo(beam.num_node,
    #                                             beam.num_elem,
    #                                             beam.num_node_elem,
    #                                             rb=True))
    # for i in range(1, n_tsteps.value):
    #     beam.timestep_info.append(StructTimeStepInfo(beam.num_node,
    #                                                  beam.num_elem,
    #                                                  beam.num_node_elem,
    #                                                  rb=True))
    # for i in range(n_tsteps.value):
    #     beam.timestep_info[i].pos_def[:] = pos_def_history[i, :]
    #     beam.timestep_info[i].psi_def[:] = psi_def_history[i, :]
    #     beam.timestep_info[i].pos_dot_def[:] = pos_dot_def_history[i, :]
    #     beam.timestep_info[i].psi_dot_def[:] = psi_dot_def_history[i, :]
    #
    #     beam.timestep_info[i].quat[:] = beam.quat_history[i, :]
    #     # beam.timestep_info[i].for_pos[:] = beam.for_pos[i, 0:3]
    #     # beam.timestep_info[i].for_vel[:] = beam.for_vel[i, 0:3]
    #     beam.timestep_info[i].for_pos[:] = beam.for_pos[i, :]
    #     beam.timestep_info[i].for_vel[:] = beam.for_vel[i, :]
    #     beam.timestep_info[i].glob_pos_def[:] = glob_pos_def[i, :]
    #
    # beam.for_pos = []
    # beam.for_vel = []

def cbeam3_solv_update_static_python(beam, deltax, pos_def, psi_def):
    n_node = ct.c_int(beam.num_node)
    n_elem = ct.c_int(beam.num_elem)

    max_pos = np.max(pos_def)
    max_delta = np.max(deltax)
    coeff = 0.2*max_pos/max_delta
    deltax *= coeff

    if not np.isfortran(deltax):
        deltax = np.asfortranarray(deltax)

    f_cbeam3_solv_update_static(ct.byref(beam.num_dof),
                                ct.byref(n_node),
                                beam.node_master_elem_fortran.ctypes.data_as(intP),
                                beam.vdof.ctypes.data_as(intP),
                                ct.byref(n_elem),
                                beam.master_fortran.ctypes.data_as(intP),
                                beam.num_nodes_matrix.ctypes.data_as(intP),
                                beam.psi_ini.ctypes.data_as(doubleP),
                                pos_def.ctypes.data_as(doubleP),
                                psi_def.ctypes.data_as(doubleP),
                                deltax.ctypes.data_as(doubleP))















