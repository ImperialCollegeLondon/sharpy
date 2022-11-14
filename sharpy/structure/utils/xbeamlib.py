import ctypes as ct
import numpy as np
import scipy as sc
import scipy.integrate

import sharpy.utils.algebra as algebra
import sharpy.utils.ctypes_utils as ct_utils
from sharpy.utils.sharpydir import SharpyDir
# from sharpy.utils.datastructures import StructTimeStepInfo
import sharpy.utils.cout_utils as cout


try:
    xbeamlib = ct_utils.import_ctypes_lib(SharpyDir + '/xbeam', 'libxbeam')
except OSError:
    xbeamlib = ct_utils.import_ctypes_lib(SharpyDir + '/lib/xbeam/lib',
    'libxbeam')

# ctypes pointer types
doubleP = ct.POINTER(ct.c_double)
intP = ct.POINTER(ct.c_int)
charP = ct.POINTER(ct.c_char_p)


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
                ("abs_threshold", ct.c_double),
                ("NewmarkDamp", ct.c_double),
                ("gravity_on", ct.c_bool),
                ("gravity", ct.c_double),
                ("gravity_dir_x", ct.c_double),
                ("gravity_dir_y", ct.c_double),
                ("gravity_dir_z", ct.c_double),
                ("balancing", ct.c_bool),
                ("relaxation_factor", ct.c_double),
                ]

    def __init__(self):
        ct.Structure.__init__(self)
        self.FollowerForce = ct.c_bool(True)
        self.FollowerForceRig = ct.c_bool(True)
        self.PrintInfo = ct.c_bool(True)
        self.OutInBframe = ct.c_bool(False)
        self.OutInaframe = ct.c_bool(True)
        self.ElemProj = ct.c_int(0)
        self.MaxIterations = ct.c_int(99)
        self.NumLoadSteps = ct.c_int(5)
        self.NumGauss = ct.c_int(0)
        self.Solution = ct.c_int(111)
        self.DeltaCurved = ct.c_double(1.0e-2)
        self.MinDelta = ct.c_double(1.0e-8)
        self.abs_threshold = ct.c_double(1.0e-13)
        self.NewmarkDamp = ct.c_double(0.0)
        self.gravity_on = ct.c_bool(False)
        self.gravity = ct.c_double(0.0)
        self.gravity_dir_x = ct.c_double(0.0)
        self.gravity_dir_y = ct.c_double(0.0)
        self.gravity_dir_z = ct.c_double(1.0)
        self.balancing = ct.c_bool(False)
        self.relaxation_factor = ct.c_double(0.3)


def cbeam3_solv_nlnstatic(beam, settings, ts):
    """@brief Python wrapper for f_cbeam3_solv_nlnstatic
     Alfonso del Carre
    """
    f_cbeam3_solv_nlnstatic = xbeamlib.cbeam3_solv_nlnstatic_python
    f_cbeam3_solv_nlnstatic.restype = None

    n_elem = ct.c_int(beam.num_elem)
    n_nodes = ct.c_int(beam.num_node)
    n_mass = ct.c_int(beam.n_mass)
    n_stiff = ct.c_int(beam.n_stiff)

    xbopts = Xbopts()
    xbopts.PrintInfo = ct.c_bool(settings['print_info'])
    xbopts.Solution = ct.c_int(112)
    xbopts.MaxIterations = ct.c_int(settings['max_iterations'])
    xbopts.NumLoadSteps = ct.c_int(settings['num_load_steps'])
    xbopts.DeltaCurved = ct.c_double(settings['delta_curved'])
    xbopts.MinDelta = ct.c_double(settings['min_delta'])
    xbopts.abs_threshold = ct.c_double(settings['abs_threshold'])
    xbopts.gravity_on = ct.c_bool(settings['gravity_on'])
    xbopts.gravity = ct.c_double(settings['gravity'])
    gravity_vector = np.dot(beam.timestep_info[ts].cag(), settings['gravity_dir'])
    xbopts.gravity_dir_x = ct.c_double(gravity_vector[0])
    xbopts.gravity_dir_y = ct.c_double(gravity_vector[1])
    xbopts.gravity_dir_z = ct.c_double(gravity_vector[2])
    xbopts.relaxation_factor = ct.c_double(settings['relaxation_factor'])

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
                            beam.timestep_info[ts].steady_applied_forces.ctypes.data_as(doubleP),
                            beam.timestep_info[ts].gravity_forces.ctypes.data_as(doubleP)
                            )


def cbeam3_loads(beam, timestep):
    """Python wrapper for f_cbeam3_loads
    
    Args:
        beam (sharpy.structure.models.beam.Beam): Structural info class
        timestep (sharpy.utils.datastructures.StructTimeStepInfo): Structural time step class

    Returns:
        tuple: Tuple containing the ``strains`` and ``loads``.
    """
    f_cbeam3_loads = xbeamlib.cbeam3_loads
    f_cbeam3_loads.restype = None

    n_elem = ct.c_int(beam.num_elem)
    n_nodes = ct.c_int(beam.num_node)
    n_stiff = ct.c_int(beam.n_stiff)

    strain = np.zeros((n_elem.value, 6), dtype=ct.c_double, order='F')
    loads = np.zeros((n_elem.value, 6), dtype=ct.c_double, order='F')

    f_cbeam3_loads(ct.byref(n_elem),
                   ct.byref(n_nodes),
                   beam.fortran['connectivities'].ctypes.data_as(intP),
                   beam.ini_info.pos.ctypes.data_as(doubleP),
                   timestep.pos.ctypes.data_as(doubleP),
                   beam.ini_info.psi.ctypes.data_as(doubleP),
                   timestep.psi.ctypes.data_as(doubleP),
                   beam.fortran['stiffness_indices'].ctypes.data_as(intP),
                   ct.byref(n_stiff),
                   beam.fortran['stiffness'].ctypes.data_as(doubleP),
                   strain.ctypes.data_as(doubleP),
                   loads.ctypes.data_as(doubleP))

    return strain, loads


def cbeam3_solv_nlndyn(beam, settings):
    f_cbeam3_solv_nlndyn = xbeamlib.cbeam3_solv_nlndyn_python
    f_cbeam3_solv_nlndyn.restype = None

    n_elem = ct.c_int(beam.num_elem)
    n_nodes = ct.c_int(beam.num_node)
    n_mass = ct.c_int(beam.n_mass)
    n_stiff = ct.c_int(beam.n_stiff)

    dt = settings['dt']
    n_tsteps = settings['num_steps']
    time = np.zeros((n_tsteps,), dtype=ct.c_double, order='F')
    for i in range(n_tsteps):
        time[i] = i*dt

    # deformation history matrices
    pos_def_history = np.zeros((n_tsteps, beam.num_node, 3), order='F')
    pos_dot_def_history = np.zeros((n_tsteps, beam.num_node, 3), order='F')
    psi_def_history = np.zeros((n_tsteps, beam.num_elem, 3, 3), order='F')
    psi_dot_def_history = np.zeros((n_tsteps, beam.num_elem, 3, 3), order='F')

    n_tsteps = ct.c_int(n_tsteps)

    xbopts = Xbopts()
    xbopts.PrintInfo = ct.c_bool(settings['print_info'])
    xbopts.Solution = ct.c_int(312)
    # xbopts.OutInaframe = ct.c_bool(settings['out_a_frame'])
    # xbopts.OutInBframe = ct.c_bool(settings['out_b_frame'])
    # xbopts.ElemProj = settings['elem_proj']
    xbopts.MaxIterations = ct.c_int(settings['max_iterations'])
    xbopts.NumLoadSteps = ct.c_int(settings['num_load_steps'])
    xbopts.NumGauss = ct.c_int(0)
    xbopts.DeltaCurved = ct.c_double(settings['delta_curved'])
    xbopts.MinDelta = ct.c_double(settings['min_delta'])
    xbopts.abs_threshold = ct.c_double(settings['abs_threshold'])
    xbopts.NewmarkDamp = ct.c_double(settings['newmark_damp'])
    xbopts.gravity_on = ct.c_bool(settings['gravity_on'])
    xbopts.gravity = ct.c_double(settings['gravity'])
    xbopts.gravity_dir_x = ct.c_double(settings['gravity_dir'][0])
    xbopts.gravity_dir_y = ct.c_double(settings['gravity_dir'][1])
    xbopts.gravity_dir_z = ct.c_double(settings['gravity_dir'][2])
    xbopts.relaxation_factor = ct.c_double(settings['relaxation_factor'])

    # here we only need to set the flags at True, all the forces are follower
    xbopts.FollowerForce = ct.c_bool(True)
    xbopts.FollowerForceRig = ct.c_bool(True)


    f_cbeam3_solv_nlndyn(ct.byref(n_elem),
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
                         beam.timestep_info[0].pos.ctypes.data_as(doubleP),
                         beam.timestep_info[0].psi.ctypes.data_as(doubleP),
                         beam.timestep_info[0].steady_applied_forces.ctypes.data_as(doubleP),
                         dynamic_forces.ctypes.data_as(doubleP),
                         beam.forced_vel_fortran.ctypes.data_as(doubleP),
                         beam.forced_acc_fortran.ctypes.data_as(doubleP),
                         pos_def_history.ctypes.data_as(doubleP),
                         psi_def_history.ctypes.data_as(doubleP),
                         pos_dot_def_history.ctypes.data_as(doubleP),
                         psi_dot_def_history.ctypes.data_as(doubleP)
                         )

    for i in range(1, n_tsteps.value):
        beam.add_timestep()
        beam.timestep_info[i].pos[:] = pos_def_history[i, :]
        beam.timestep_info[i].psi[:] = psi_def_history[i, :]
        beam.timestep_info[i].pos_dot[:] = pos_dot_def_history[i, :]
        beam.timestep_info[i].psi_dot[:] = psi_dot_def_history[i, :]


def cbeam3_step_nlndyn(beam, settings, ts, tstep=None, dt=None):
    f_cbeam3_solv_nlndyn_step = xbeamlib.cbeam3_solv_nlndyn_step_python
    f_cbeam3_solv_nlndyn_step.restype = None

    if tstep is None:
        tstep = beam.timestep_info[-1]

    n_elem = ct.c_int(beam.num_elem)
    n_nodes = ct.c_int(beam.num_node)
    n_mass = ct.c_int(beam.n_mass)
    n_stiff = ct.c_int(beam.n_stiff)
    num_dof = ct.c_int(len(tstep.q) - 10)

    xbopts = Xbopts()
    xbopts.PrintInfo = ct.c_bool(settings['print_info'])
    xbopts.Solution = ct.c_int(312)
    xbopts.MaxIterations = ct.c_int(settings['max_iterations'])
    xbopts.NumLoadSteps = ct.c_int(settings['num_load_steps'])
    xbopts.NumGauss = ct.c_int(0)
    xbopts.DeltaCurved = ct.c_double(settings['delta_curved'])
    xbopts.MinDelta = ct.c_double(settings['min_delta'])
    xbopts.abs_threshold = ct.c_double(settings['abs_threshold'])
    xbopts.NewmarkDamp = ct.c_double(settings['newmark_damp'])
    xbopts.gravity_on = ct.c_bool(settings['gravity_on'])
    xbopts.gravity = ct.c_double(settings['gravity'])
    xbopts.gravity_dir_x = ct.c_double(settings['gravity_dir'][0])
    xbopts.gravity_dir_y = ct.c_double(settings['gravity_dir'][1])
    xbopts.gravity_dir_z = ct.c_double(settings['gravity_dir'][2])
    xbopts.relaxation_factor = ct.c_double(settings['relaxation_factor'])

    # here we only need to set the flags at True, all the forces are follower
    xbopts.FollowerForce = ct.c_bool(True)
    xbopts.FollowerForceRig = ct.c_bool(True)

    if dt is None:
        in_dt = ct.c_double(settings['dt'])
    else:
        in_dt = ct.c_double(dt)

    f_cbeam3_solv_nlndyn_step(ct.byref(num_dof),
                              ct.byref(n_elem),
                              ct.byref(n_nodes),
                              ct.byref(in_dt),
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
                              tstep.pos.ctypes.data_as(doubleP),
                              tstep.pos_dot.ctypes.data_as(doubleP),
                              tstep.pos_ddot.ctypes.data_as(doubleP),
                              tstep.psi.ctypes.data_as(doubleP),
                              tstep.psi_dot.ctypes.data_as(doubleP),
                              tstep.psi_ddot.ctypes.data_as(doubleP),
                              tstep.steady_applied_forces.ctypes.data_as(doubleP),
                              tstep.unsteady_applied_forces.ctypes.data_as(doubleP),
                              tstep.gravity_forces.ctypes.data_as(doubleP),
                              tstep.quat.ctypes.data_as(doubleP),
                              tstep.for_vel.ctypes.data_as(doubleP),
                              tstep.for_acc.ctypes.data_as(doubleP),
                              tstep.q.ctypes.data_as(doubleP),
                              tstep.dqdt.ctypes.data_as(doubleP)
                              )

    xbeam_solv_state2disp(beam, tstep, cbeam3=True)


f_xbeam_solv_couplednlndyn = xbeamlib.xbeam_solv_couplednlndyn_python
f_xbeam_solv_couplednlndyn.restype = None


def xbeam_solv_couplednlndyn(beam, settings):
    n_elem = ct.c_int(beam.num_elem)
    n_nodes = ct.c_int(beam.num_node)
    n_mass = ct.c_int(beam.n_mass)
    n_stiff = ct.c_int(beam.n_stiff)

    dt = settings['dt']
    n_tsteps = settings['num_steps']
    time = np.zeros((n_tsteps,), dtype=ct.c_double, order='F')
    for i in range(n_tsteps):
        time[i] = i*dt

    # deformation history matrices
    for_vel = np.zeros((n_tsteps + 1, 6), order='F')
    for_acc = np.zeros((n_tsteps + 1, 6), order='F')
    quat_history = np.zeros((n_tsteps, 4), order='F')
    quat_history[0, 0] = 1.0
    quat_history[0, :] = beam.timestep_info[0].quat[:]

    dt = ct.c_double(dt)
    n_tsteps = ct.c_int(n_tsteps)

    xbopts = Xbopts()
    xbopts.PrintInfo = ct.c_bool(settings['print_info'])
    xbopts.Solution = ct.c_int(910)
    xbopts.OutInaframe = ct.c_bool(True)
    xbopts.MaxIterations = ct.c_int(settings['max_iterations'])
    xbopts.NumLoadSteps = ct.c_int(settings['num_load_steps'])
    # xbopts.NumGauss = ct.c_int(0)
    xbopts.DeltaCurved = ct.c_double(settings['delta_curved'])
    xbopts.MinDelta = ct.c_double(settings['min_delta'])
    xbopts.abs_threshold = ct.c_double(settings['abs_threshold'])
    xbopts.NewmarkDamp = ct.c_double(settings['newmark_damp'])
    xbopts.gravity_on = ct.c_bool(settings['gravity_on'])
    xbopts.gravity = ct.c_double(settings['gravity'])
    xbopts.gravity_dir_x = ct.c_double(settings['gravity_dir'][0])
    xbopts.gravity_dir_y = ct.c_double(settings['gravity_dir'][1])
    xbopts.gravity_dir_z = ct.c_double(settings['gravity_dir'][2])
    xbopts.relaxation_factor = ct.c_double(settings['relaxation_factor'])
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
                               beam.ini_info.steady_applied_forces.ctypes.data_as(doubleP),
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
        rot = algebra.quat2rotation(quat_history[it, :])
        for inode in range(beam.num_node):
            glob_pos_def[it, inode, :] = np.dot(rot.T, pos_def_history[it, inode, :])

    for i in range(n_tsteps.value - 1):
        beam.timestep_info[i + 1].pos[:] = pos_def_history[i+1, :]
        beam.timestep_info[i + 1].psi[:] = psi_def_history[i+1, :]
        beam.timestep_info[i + 1].pos_dot[:] = pos_dot_def_history[i+1, :]
        beam.timestep_info[i + 1].psi_dot[:] = psi_dot_def_history[i+1, :]

        beam.timestep_info[i + 1].quat[:] = quat_history[i+1, :]
        # beam.timestep_info[i + 1].for_pos[:] = for_pos[i+1, :]
        beam.timestep_info[i + 1].for_vel[:] = for_vel[i+1, :]

    for it in range(n_tsteps.value - 1):
        beam.integrate_position(it + 1, dt.value)


def xbeam_step_couplednlndyn(beam, settings, ts, tstep=None, dt=None):
    # library load
    f_xbeam_solv_nlndyn_step_python = xbeamlib.xbeam_solv_nlndyn_step_python
    f_xbeam_solv_nlndyn_step_python.restype = None

    if tstep is None:
        tstep = beam.timestep_info[-1]

    # initialisation
    n_elem = ct.c_int(beam.num_elem)
    n_nodes = ct.c_int(beam.num_node)
    n_mass = ct.c_int(beam.n_mass)
    n_stiff = ct.c_int(beam.n_stiff)

    xbopts = Xbopts()
    xbopts.PrintInfo = ct.c_bool(settings['print_info'])
    xbopts.MaxIterations = ct.c_int(settings['max_iterations'])
    xbopts.NumLoadSteps = ct.c_int(settings['num_load_steps'])
    xbopts.DeltaCurved = ct.c_double(settings['delta_curved'])
    xbopts.MinDelta = ct.c_double(settings['min_delta'])
    xbopts.abs_threshold = ct.c_double(settings['abs_threshold'])
    xbopts.NewmarkDamp = ct.c_double(settings['newmark_damp'])
    xbopts.gravity_on = ct.c_bool(settings['gravity_on'])
    xbopts.gravity = ct.c_double(settings['gravity'])
    xbopts.balancing = ct.c_bool(settings['balancing'])
    xbopts.gravity_dir_x = ct.c_double(settings['gravity_dir'][0])
    xbopts.gravity_dir_y = ct.c_double(settings['gravity_dir'][1])
    xbopts.gravity_dir_z = ct.c_double(settings['gravity_dir'][2])
    xbopts.relaxation_factor = ct.c_double(settings['relaxation_factor'])

    if dt is None:
        in_dt = ct.c_double(settings['dt'])
    else:
        in_dt = ct.c_double(dt)

    ctypes_ts = ct.c_int(ts)
    numdof = ct.c_int(beam.num_dof.value)

    f_xbeam_solv_nlndyn_step_python(ct.byref(numdof),
                                    ct.byref(ctypes_ts),
                                    ct.byref(n_elem),
                                    ct.byref(n_nodes),
                                    ct.byref(in_dt),
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
                                    tstep.pos.ctypes.data_as(doubleP),
                                    tstep.pos_dot.ctypes.data_as(doubleP),
                                    tstep.pos_ddot.ctypes.data_as(doubleP),
                                    tstep.psi.ctypes.data_as(doubleP),
                                    tstep.psi_dot.ctypes.data_as(doubleP),
                                    tstep.psi_ddot.ctypes.data_as(doubleP),
                                    tstep.steady_applied_forces.ctypes.data_as(doubleP),
                                    tstep.unsteady_applied_forces.ctypes.data_as(doubleP),
                                    tstep.gravity_forces.ctypes.data_as(doubleP),
                                    tstep.quat.ctypes.data_as(doubleP),
                                    tstep.for_vel.ctypes.data_as(doubleP),
                                    tstep.for_acc.ctypes.data_as(doubleP),
                                    tstep.q.ctypes.data_as(doubleP),
                                    tstep.dqdt.ctypes.data_as(doubleP),
                                    tstep.dqddt.ctypes.data_as(doubleP))


def xbeam_init_couplednlndyn(beam, settings, ts, dt=None):
    # library load
    f_xbeam_solv_nlndyn_init_python = xbeamlib.xbeam_solv_nlndyn_init_python
    f_xbeam_solv_nlndyn_init_python.restype = None

    # initialisation
    n_elem = ct.c_int(beam.num_elem)
    n_nodes = ct.c_int(beam.num_node)
    n_mass = ct.c_int(beam.n_mass)
    n_stiff = ct.c_int(beam.n_stiff)

    xbopts = Xbopts()
    xbopts.PrintInfo = ct.c_bool(settings['print_info'])
    xbopts.MaxIterations = ct.c_int(settings['max_iterations'])
    xbopts.NumLoadSteps = ct.c_int(settings['num_load_steps'])
    xbopts.DeltaCurved = ct.c_double(settings['delta_curved'])
    xbopts.MinDelta = ct.c_double(settings['min_delta'])
    xbopts.abs_threshold = ct.c_double(settings['abs_threshold'])
    xbopts.NewmarkDamp = ct.c_double(settings['newmark_damp'])
    xbopts.gravity_on = ct.c_bool(settings['gravity_on'])
    xbopts.gravity = ct.c_double(settings['gravity'])
    xbopts.gravity_dir_x = ct.c_double(settings['gravity_dir'][0])
    xbopts.gravity_dir_y = ct.c_double(settings['gravity_dir'][1])
    xbopts.gravity_dir_z = ct.c_double(settings['gravity_dir'][2])

    in_dt = ct.c_double(dt)

    if ts < 0:
        ctypes_ts = ct.c_int(0)
    else:
        ctypes_ts = ct.c_int(ts)
    numdof = ct.c_int(beam.num_dof.value)

    f_xbeam_solv_nlndyn_init_python(ct.byref(numdof),
                                    ct.byref(ctypes_ts),
                                    ct.byref(n_elem),
                                    ct.byref(n_nodes),
                                    ct.byref(ct.c_double(settings['dt'])),
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
                                    beam.timestep_info[ts].pos_dot.ctypes.data_as(doubleP),
                                    beam.timestep_info[ts].psi.ctypes.data_as(doubleP),
                                    beam.timestep_info[ts].psi_dot.ctypes.data_as(doubleP),
                                    beam.timestep_info[ts].steady_applied_forces.ctypes.data_as(doubleP),
                                    beam.timestep_info[ts].unsteady_applied_forces.ctypes.data_as(doubleP),
                                    beam.timestep_info[ts].quat.ctypes.data_as(doubleP),
                                    beam.timestep_info[ts].for_vel.ctypes.data_as(doubleP),
                                    beam.timestep_info[ts].for_acc.ctypes.data_as(doubleP),
                                    beam.timestep_info[ts].q.ctypes.data_as(doubleP),
                                    beam.timestep_info[ts].dqdt.ctypes.data_as(doubleP),
                                    beam.timestep_info[ts].dqddt.ctypes.data_as(doubleP))


def xbeam_solv_state2disp(beam, tstep, cbeam3=False):
    numdof = beam.num_dof.value
    cbeam3_solv_state2disp(beam, tstep)
    if not cbeam3:
        tstep.for_pos[0:3] = tstep.q[numdof:numdof+3].astype(dtype=ct.c_double, order='F', copy=True)
        tstep.for_vel = tstep.dqdt[numdof:numdof+6].astype(dtype=ct.c_double, order='F', copy=True)
        # tstep.for_acc = tstep.dqddt[numdof:numdof+6].astype(dtype=ct.c_double, order='F', copy=True)
        tstep.quat = algebra.unit_vector(tstep.dqdt[numdof+6:]).astype(dtype=ct.c_double, order='F', copy=True)

def xbeam_solv_state2accel(beam, tstep, cbeam3=False):
    numdof = beam.num_dof.value
    cbeam3_solv_state2accel(beam, tstep)
    if not cbeam3:
        tstep.for_acc = tstep.dqddt[numdof:numdof+6].astype(dtype=ct.c_double, order='F', copy=True)

def cbeam3_solv_state2disp(beam, tstep):
    # library load
    f_cbeam3_solv_state2disp = xbeamlib.cbeam3_solv_state2disp_python
    f_cbeam3_solv_state2disp.restype = None

    # initialisation
    n_elem = ct.c_int(beam.num_elem)
    n_nodes = ct.c_int(beam.num_node)
    numdof = ct.c_int(beam.num_dof.value)

    f_cbeam3_solv_state2disp(
        ct.byref(n_elem),
        ct.byref(n_nodes),
        ct.byref(numdof),
        beam.ini_info.pos.ctypes.data_as(doubleP),
        beam.ini_info.psi.ctypes.data_as(doubleP),
        tstep.pos.ctypes.data_as(doubleP),
        tstep.psi.ctypes.data_as(doubleP),
        tstep.pos_dot.ctypes.data_as(doubleP),
        tstep.psi_dot.ctypes.data_as(doubleP),
        beam.fortran['node_master_elem'].ctypes.data_as(intP),
        beam.fortran['vdof'].ctypes.data_as(intP),
        beam.fortran['num_nodes'].ctypes.data_as(intP),
        beam.fortran['master'].ctypes.data_as(intP),
        tstep.q.ctypes.data_as(doubleP),
        tstep.dqdt.ctypes.data_as(doubleP))

def cbeam3_solv_state2accel(beam, tstep):
    # library load
    f_cbeam3_solv_state2disp = xbeamlib.cbeam3_solv_state2disp_python
    f_cbeam3_solv_state2disp.restype = None

    # initialisation
    n_elem = ct.c_int(beam.num_elem)
    n_nodes = ct.c_int(beam.num_node)
    numdof = ct.c_int(beam.num_dof.value)

    dummy_pos = np.zeros_like(tstep.pos)
    dummy_psi = np.zeros_like(tstep.psi)
    dummy_ini_pos = np.zeros_like(tstep.pos)
    dummy_ini_psi = np.zeros_like(tstep.psi)
    dummy_q = np.zeros_like(tstep.q)

    f_cbeam3_solv_state2disp(
        ct.byref(n_elem),
        ct.byref(n_nodes),
        ct.byref(numdof),
        dummy_ini_pos.ctypes.data_as(doubleP),
        dummy_ini_psi.ctypes.data_as(doubleP),
        dummy_pos.ctypes.data_as(doubleP),
        dummy_psi.ctypes.data_as(doubleP),
        tstep.pos_ddot.ctypes.data_as(doubleP),
        tstep.psi_ddot.ctypes.data_as(doubleP),
        beam.fortran['node_master_elem'].ctypes.data_as(intP),
        beam.fortran['vdof'].ctypes.data_as(intP),
        beam.fortran['num_nodes'].ctypes.data_as(intP),
        beam.fortran['master'].ctypes.data_as(intP),
        dummy_q.ctypes.data_as(doubleP),
        tstep.dqddt.ctypes.data_as(doubleP))

def xbeam_solv_disp2state(beam, tstep):
    numdof = beam.num_dof.value
    cbeam3_solv_disp2state(beam, tstep)
    tstep.q[numdof:numdof+3] = tstep.for_pos[0:3]
    tstep.dqdt[numdof:numdof+6] = tstep.for_vel
    # tstep.dqddt[numdof:numdof+6] = tstep.for_acc
    tstep.dqdt[numdof+6:] = algebra.unit_vector(tstep.quat)
    # tstep.dqdt[numdof+6:] = tstep.quat
    # tstep.dqdt[numdof+6:] = algebra.unit_quat(tstep.quat)

def xbeam_solv_accel2state(beam, tstep):
    numdof = beam.num_dof.value
    cbeam3_solv_accel2state(beam, tstep)
    tstep.dqddt[numdof:numdof+6] = tstep.for_acc

def cbeam3_solv_disp2state(beam, tstep):
    # library load
    f_cbeam3_solv_disp2state = xbeamlib.cbeam3_solv_disp2state_python
    f_cbeam3_solv_disp2state.restype = None

    # initialisation
    n_elem = ct.c_int(beam.num_elem)
    n_nodes = ct.c_int(beam.num_node)
    numdof = ct.c_int(beam.num_dof.value)

    f_cbeam3_solv_disp2state(
        ct.byref(n_elem),
        ct.byref(n_nodes),
        ct.byref(numdof),
        tstep.pos.ctypes.data_as(doubleP),
        tstep.psi.ctypes.data_as(doubleP),
        tstep.pos_dot.ctypes.data_as(doubleP),
        tstep.psi_dot.ctypes.data_as(doubleP),
        beam.fortran['vdof'].ctypes.data_as(intP),
        beam.fortran['node_master_elem'].ctypes.data_as(intP),
        tstep.q.ctypes.data_as(doubleP),
        tstep.dqdt.ctypes.data_as(doubleP))

def cbeam3_solv_accel2state(beam, tstep):
    # library load
    f_cbeam3_solv_disp2state = xbeamlib.cbeam3_solv_disp2state_python
    f_cbeam3_solv_disp2state.restype = None

    # initialisation
    n_elem = ct.c_int(beam.num_elem)
    n_nodes = ct.c_int(beam.num_node)
    numdof = ct.c_int(beam.num_dof.value)

    dummy_pos = np.zeros_like(tstep.pos)
    dummy_psi = np.zeros_like(tstep.psi)
    dummy_q = np.zeros_like(tstep.q)

    # Reused
    f_cbeam3_solv_disp2state(
        ct.byref(n_elem),
        ct.byref(n_nodes),
        ct.byref(numdof),
        dummy_pos.ctypes.data_as(doubleP),
        dummy_psi.ctypes.data_as(doubleP),
        tstep.pos_ddot.ctypes.data_as(doubleP),
        tstep.psi_ddot.ctypes.data_as(doubleP),
        beam.fortran['vdof'].ctypes.data_as(intP),
        beam.fortran['node_master_elem'].ctypes.data_as(intP),
        dummy_q.ctypes.data_as(doubleP),
        tstep.dqddt.ctypes.data_as(doubleP))

def cbeam3_solv_modal(beam, settings, ts, FullMglobal, FullCglobal, FullKglobal):
    """
    cbeam3_solv_modal

    Generates the system matrices for modal analysis

    Args:
        beam(Beam): beam information
        settings(settings):
        ts(int): timestep
        FullMglobal(numpy array): Mass matrix
        FullCglobal(numpy array): Damping matrix
        FullKglobal(numpy array): Stiffness matrix

    Returns:
        FullMglobal(numpy array): Mass matrix
        FullCglobal(numpy array): Damping matrix
        FullKglobal(numpy array): Stiffness matrix

    Examples:

    Notes:

    """

    f_cbeam3_solv_modal = xbeamlib.cbeam3_solv_modal_python
    f_cbeam3_solv_modal.restype = None

    n_elem = ct.c_int(beam.num_elem)
    n_nodes = ct.c_int(beam.num_node)
    n_mass = ct.c_int(beam.n_mass)
    n_stiff = ct.c_int(beam.n_stiff)
    num_dof = ct.c_int(beam.num_dof.value)

    xbopts = Xbopts()
    xbopts.PrintInfo = ct.c_bool(settings['print_info'])
    xbopts.Solution = ct.c_int(312)
    # xbopts.OutInaframe = ct.c_bool(settings['out_a_frame'])
    # xbopts.OutInBframe = ct.c_bool(settings['out_b_frame'])
    # xbopts.ElemProj = settings['elem_proj']
    # xbopts.MaxIterations = settings['max_iterations']
    # xbopts.NumLoadSteps = settings['num_load_steps']
    xbopts.NumGauss = ct.c_int(0)
    xbopts.DeltaCurved = ct.c_double(settings['delta_curved'])
    # xbopts.MinDelta = settings['min_delta']
    # xbopts.NewmarkDamp = settings['newmark_damp']
    # xbopts.gravity_on = settings['gravity_on']
    # xbopts.gravity = settings['gravity']
    # xbopts.gravity_dir_x = ct.c_double(settings['gravity_dir'][0])
    # xbopts.gravity_dir_y = ct.c_double(settings['gravity_dir'][1])
    # xbopts.gravity_dir_z = ct.c_double(settings['gravity_dir'][2])


    # print("ts: ",ts)
    # print("FoR vel: ", beam.timestep_info[ts].for_vel)
    # print("initial position: ", beam.ini_info.pos)
    # print("initial rotation: ", beam.ini_info.psi)
    # print("position: ", beam.timestep_info[ts].pos)
    # print("rotation: ", beam.timestep_info[ts].psi)
    # print("mass: ", beam.fortran['mass'])
    # print("mass: ", beam.fortran['stiffness'])
    # print("mass: ", beam.fortran['inv_stiffness'])


    f_cbeam3_solv_modal(ct.byref(num_dof),
                        ct.byref(n_elem),
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
                        beam.timestep_info[ts].for_vel.ctypes.data_as(doubleP),
                        FullMglobal.ctypes.data_as(doubleP),
                        FullCglobal.ctypes.data_as(doubleP),
                        FullKglobal.ctypes.data_as(doubleP))


def cbeam3_asbly_dynamic(beam, tstep, settings):
    """
    cbeam3_asbly_dynamic

    Generates the system matrices for a nonlinear dynamic structure with
    prescribed FoR motions

    Args:
        beam(Beam): beam information
        tstep(StructTimeStepInfo): time step information
        settings(settings):

    Returns:
    	Mglobal(numpy array): Mass matrix
        Cglobal(numpy array): Damping matrix
        Kglobal(numpy array): Stiffness matrix
        Qglobal(numpy array): Vector of independent terms

    Examples:

    Notes:

    """

    # library load
    f_cbeam3_asbly_dynamic_python = xbeamlib.cbeam3_asbly_dynamic_python
    f_cbeam3_asbly_dynamic_python.restype = None

    # initialisation
    n_elem = ct.c_int(beam.num_elem)
    n_nodes = ct.c_int(beam.num_node)
    num_dof = beam.num_dof.value
    n_mass = ct.c_int(beam.n_mass)
    n_stiff = ct.c_int(beam.n_stiff)
    dt = ct.c_double(settings['dt'])

    # Options
    xbopts = Xbopts()
    xbopts.PrintInfo = ct.c_bool(settings['print_info'])
    xbopts.Solution = ct.c_int(312)
    xbopts.MaxIterations = ct.c_int(settings['max_iterations'])
    xbopts.NumLoadSteps = ct.c_int(settings['num_load_steps'])
    xbopts.NumGauss = ct.c_int(0)
    xbopts.DeltaCurved = ct.c_double(settings['delta_curved'])
    xbopts.MinDelta = ct.c_double(settings['min_delta'])
    xbopts.abs_threshold = ct.c_double(settings['abs_threshold'])
    xbopts.NewmarkDamp = ct.c_double(settings['newmark_damp'])
    xbopts.gravity_on = ct.c_bool(settings['gravity_on'])
    xbopts.gravity = ct.c_double(settings['gravity'])
    xbopts.gravity_dir_x = ct.c_double(settings['gravity_dir'][0])
    xbopts.gravity_dir_y = ct.c_double(settings['gravity_dir'][1])
    xbopts.gravity_dir_z = ct.c_double(settings['gravity_dir'][2])

    # Initialize matrices
    Mglobal = np.zeros((num_dof, num_dof), dtype=ct.c_double, order='F')
    Cglobal = np.zeros((num_dof, num_dof), dtype=ct.c_double, order='F')
    Kglobal = np.zeros((num_dof, num_dof), dtype=ct.c_double, order='F')
    Qglobal = np.zeros((num_dof, ), dtype=ct.c_double, order='F')

    f_cbeam3_asbly_dynamic_python(ct.byref(ct.c_int(num_dof)),
                                  ct.byref(n_nodes),
                                  ct.byref(n_elem),
                                  ct.byref(dt),
                                  beam.ini_info.pos.ctypes.data_as(doubleP),
                                  beam.ini_info.psi.ctypes.data_as(doubleP),
                                  tstep.pos.ctypes.data_as(doubleP),
                                  tstep.pos_dot.ctypes.data_as(doubleP),
                                  tstep.pos_ddot.ctypes.data_as(doubleP),
                                  tstep.psi.ctypes.data_as(doubleP),
                                  tstep.psi_dot.ctypes.data_as(doubleP),
                                  tstep.psi_ddot.ctypes.data_as(doubleP),
                                  tstep.steady_applied_forces.ctypes.data_as(doubleP),
                                  tstep.unsteady_applied_forces.ctypes.data_as(doubleP),
                                  tstep.for_vel.ctypes.data_as(doubleP),
                                  tstep.for_acc.ctypes.data_as(doubleP),
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
                                  # CAREFUL, this is dXddt, with num_dof elements,
                                  # not num_dof + 10
                                  tstep.dqddt.ctypes.data_as(doubleP),
                                  tstep.quat.ctypes.data_as(doubleP),
                                  tstep.gravity_forces.ctypes.data_as(doubleP),
                                  Mglobal.ctypes.data_as(doubleP),
                                  Cglobal.ctypes.data_as(doubleP),
                                  Kglobal.ctypes.data_as(doubleP),
                                  Qglobal.ctypes.data_as(doubleP))

    return Mglobal, Cglobal, Kglobal, Qglobal

def xbeam3_asbly_dynamic(beam, tstep, settings):
    """
    xbeam3_asbly_dynamic

    Generates the system matrices for a nonlinear dynamic structure with
    free FoR motions

    Args:
        beam(Beam): beam information
        tstep(StructTimeStepInfo): time step information
        settings(settings):

    Returns:
    	Mglobal(numpy array): Mass matrix
        Cglobal(numpy array): Damping matrix
        Kglobal(numpy array): Stiffness matrix
        Qglobal(numpy array): Vector of independent terms

    Examples:

    Notes:

    """

    # library load
    f_xbeam3_asbly_dynamic_python = xbeamlib.xbeam3_asbly_dynamic_python
    f_xbeam3_asbly_dynamic_python.restype = None

    # initialisation
    n_elem = ct.c_int(beam.num_elem)
    n_nodes = ct.c_int(beam.num_node)
    num_dof = beam.num_dof.value
    n_mass = ct.c_int(beam.n_mass)
    n_stiff = ct.c_int(beam.n_stiff)
    dt = ct.c_double(settings['dt'])

    # Options
    xbopts = Xbopts()
    xbopts.PrintInfo = ct.c_bool(settings['print_info'])
    xbopts.Solution = ct.c_int(312)
    xbopts.MaxIterations = ct.c_int(settings['max_iterations'])
    xbopts.NumLoadSteps = ct.c_int(settings['num_load_steps'])
    xbopts.NumGauss = ct.c_int(0)
    xbopts.DeltaCurved = ct.c_double(settings['delta_curved'])
    xbopts.MinDelta = ct.c_double(settings['min_delta'])
    xbopts.abs_threshold = ct.c_double(settings['abs_threshold'])
    xbopts.NewmarkDamp = ct.c_double(settings['newmark_damp'])
    xbopts.gravity_on = ct.c_bool(settings['gravity_on'])
    xbopts.gravity = ct.c_double(settings['gravity'])
    xbopts.gravity_dir_x = ct.c_double(settings['gravity_dir'][0])
    xbopts.gravity_dir_y = ct.c_double(settings['gravity_dir'][1])
    xbopts.gravity_dir_z = ct.c_double(settings['gravity_dir'][2])

    # Initialize matrices
    Mtotal = np.zeros((num_dof+10, num_dof+10), dtype=ct.c_double, order='F')
    Ctotal = np.zeros((num_dof+10, num_dof+10), dtype=ct.c_double, order='F')
    Ktotal = np.zeros((num_dof+10, num_dof+10), dtype=ct.c_double, order='F')
    Qtotal = np.zeros((num_dof+10, ), dtype=ct.c_double, order='F')

    f_xbeam3_asbly_dynamic_python(ct.byref(ct.c_int(num_dof)),
                            ct.byref(n_nodes),
                            ct.byref(n_elem),
                            ct.byref(dt),
                            beam.ini_info.pos.ctypes.data_as(doubleP),
                            beam.ini_info.psi.ctypes.data_as(doubleP),
                            tstep.pos.ctypes.data_as(doubleP),
                            tstep.pos_dot.ctypes.data_as(doubleP),
                            tstep.pos_ddot.ctypes.data_as(doubleP),
                            tstep.psi.ctypes.data_as(doubleP),
                            tstep.psi_dot.ctypes.data_as(doubleP),
                            tstep.psi_ddot.ctypes.data_as(doubleP),
                            tstep.steady_applied_forces.ctypes.data_as(doubleP),
                            tstep.unsteady_applied_forces.ctypes.data_as(doubleP),
                            tstep.for_vel.ctypes.data_as(doubleP),
                            tstep.for_acc.ctypes.data_as(doubleP),
                            # ct.byref(in_dt),
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
                            tstep.quat.ctypes.data_as(doubleP),
                            tstep.q.ctypes.data_as(doubleP),
                            tstep.dqdt.ctypes.data_as(doubleP),
                            tstep.dqddt.ctypes.data_as(doubleP),
                            tstep.gravity_forces.ctypes.data_as(doubleP),
                            Mtotal.ctypes.data_as(doubleP),
                            Ctotal.ctypes.data_as(doubleP),
                            Ktotal.ctypes.data_as(doubleP),
                            Qtotal.ctypes.data_as(doubleP))

    return Mtotal, Ctotal, Ktotal, Qtotal

def cbeam3_correct_gravity_forces(beam, tstep, settings):
    """
    cbeam3_correct_gravity_forces

    Corrects the gravity forces orientation after a time step

    Args:
        beam(Beam): beam information
        tstep(StructTimeStepInfo): time step information
        settings(settings):
    """

    # library load
    f_cbeam3_correct_gravity_forces_python = xbeamlib.cbeam3_correct_gravity_forces_python
    f_cbeam3_correct_gravity_forces_python.restype = None

    # initialisation
    n_elem = ct.c_int(beam.num_elem)
    n_nodes = ct.c_int(beam.num_node)
    n_mass = ct.c_int(beam.n_mass)
    n_stiff = ct.c_int(beam.n_stiff)

    f_cbeam3_correct_gravity_forces_python(ct.byref(n_nodes),
                            ct.byref(n_elem),
                            beam.ini_info.psi.ctypes.data_as(doubleP),
                            tstep.psi.ctypes.data_as(doubleP),
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
                            tstep.gravity_forces.ctypes.data_as(doubleP))

def cbeam3_asbly_static(beam, tstep, settings, iLoadStep):
    """
    cbeam3_asbly_static

    Generates the system matrices for a nonlinear static structure

    Args:
        beam(Beam): beam information
        tstep(StructTimeStepInfo): time step information
        settings(settings):

    Returns:
        Kglobal(numpy array): Stiffness matrix
        Qglobal(numpy array): Vector of independent terms

    Examples:

    Notes:

    """

    # library load
    f_cbeam3_asbly_static_python = xbeamlib.cbeam3_asbly_static_python
    f_cbeam3_asbly_static_python.restype = None

    # initialisation
    n_elem = ct.c_int(beam.num_elem)
    n_nodes = ct.c_int(beam.num_node)
    num_dof = beam.num_dof.value
    n_mass = ct.c_int(beam.n_mass)
    n_stiff = ct.c_int(beam.n_stiff)
    # dt = settings['dt']

    # Options
    xbopts = Xbopts()
    xbopts.PrintInfo = ct.c_bool(settings['print_info'])
    # xbopts.Solution = ct.c_int(312)
    # xbopts.MaxIterations = settings['max_iterations']
    xbopts.NumLoadSteps = ct.c_int(settings['num_load_steps'] + 1)
    # xbopts.NumGauss = ct.c_int(0)
    # xbopts.DeltaCurved = settings['delta_curved']
    # xbopts.MinDelta = settings['min_delta']
    # xbopts.NewmarkDamp = settings['newmark_damp']
    xbopts.gravity_on = ct.c_bool(settings['gravity_on'])
    xbopts.gravity = ct.c_double(settings['gravity'])
    gravity_vector = np.dot(beam.timestep_info[ts].cag(), settings['gravity_dir'])
    xbopts.gravity_dir_x = ct.c_double(gravity_vector[0])
    xbopts.gravity_dir_y = ct.c_double(gravity_vector[1])
    xbopts.gravity_dir_z = ct.c_double(gravity_vector[2])

    # Initialize matrices
    # Mglobal = np.zeros((num_dof, num_dof), dtype=ct.c_double, order='F')
    # Cglobal = np.zeros((num_dof, num_dof), dtype=ct.c_double, order='F')
    Kglobal = np.zeros((num_dof, num_dof), dtype=ct.c_double, order='F')
    Qglobal = np.zeros((num_dof, ), dtype=ct.c_double, order='F')

    f_cbeam3_asbly_static_python(ct.byref(ct.c_int(num_dof)),
                            ct.byref(n_nodes),
                            ct.byref(n_elem),
                            beam.ini_info.pos.ctypes.data_as(doubleP),
                            beam.ini_info.psi.ctypes.data_as(doubleP),
                            tstep.pos.ctypes.data_as(doubleP),
                            tstep.psi.ctypes.data_as(doubleP),
                            tstep.steady_applied_forces.ctypes.data_as(doubleP),
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
                            tstep.gravity_forces.ctypes.data_as(doubleP),
                            Kglobal.ctypes.data_as(doubleP),
                            Qglobal.ctypes.data_as(doubleP),
                            ct.byref(ct.c_int(iLoadStep)))

    return Kglobal, Qglobal


def xbeam_step_coupledrigid(beam, settings, ts, tstep=None, dt=None):
    # library load
    f_xbeam_solv_rigid_step_python = xbeamlib.xbeam_solv_coupledrigid_step_python
    f_xbeam_solv_rigid_step_python.restype = None

    if tstep is None:
        tstep = beam.timestep_info[-1]

    # initialisation
    n_elem = ct.c_int(beam.num_elem)
    n_nodes = ct.c_int(beam.num_node)
    n_mass = ct.c_int(beam.n_mass)
    n_stiff = ct.c_int(beam.n_stiff)

    xbopts = Xbopts()
    xbopts.PrintInfo = ct.c_bool(settings['print_info'])
    xbopts.MaxIterations = ct.c_int(settings['max_iterations'])
    xbopts.NumLoadSteps = ct.c_int(settings['num_load_steps'])
    xbopts.DeltaCurved = ct.c_double(settings['delta_curved'])
    xbopts.MinDelta = ct.c_double(settings['min_delta'])
    xbopts.abs_threshold = ct.c_double(settings['abs_threshold'])
    xbopts.NewmarkDamp = ct.c_double(settings['newmark_damp'])
    xbopts.gravity_on = ct.c_bool(settings['gravity_on'])
    xbopts.gravity = ct.c_double(settings['gravity'])
    xbopts.balancing = ct.c_bool(settings['balancing'])
    xbopts.gravity_dir_x = ct.c_double(settings['gravity_dir'][0])
    xbopts.gravity_dir_y = ct.c_double(settings['gravity_dir'][1])
    xbopts.gravity_dir_z = ct.c_double(settings['gravity_dir'][2])
    xbopts.relaxation_factor = ct.c_double(settings['relaxation_factor'])

    if dt is None:
        in_dt = ct.c_double(settings['dt'])
    else:
        in_dt = ct.c_double(dt)

    ctypes_ts = ct.c_int(ts)
    numdof = ct.c_int(beam.num_dof.value)

    f_xbeam_solv_rigid_step_python(ct.byref(numdof),
                                    ct.byref(ctypes_ts),
                                    ct.byref(n_elem),
                                    ct.byref(n_nodes),
                                    ct.byref(in_dt),
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
                                    tstep.pos.ctypes.data_as(doubleP),
                                    tstep.psi.ctypes.data_as(doubleP),
                                    tstep.steady_applied_forces.ctypes.data_as(doubleP),
                                    tstep.unsteady_applied_forces.ctypes.data_as(doubleP),
                                    tstep.gravity_forces.ctypes.data_as(doubleP),
                                    tstep.quat.ctypes.data_as(doubleP),
                                    tstep.for_vel.ctypes.data_as(doubleP),
                                    tstep.for_acc.ctypes.data_as(doubleP),
                                    tstep.q[-10:].ctypes.data_as(doubleP),
                                    tstep.dqdt[-10:].ctypes.data_as(doubleP),
                                    tstep.dqddt[-10:].ctypes.data_as(doubleP))
