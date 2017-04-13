import ctypes as ct
import numpy as np
import scipy as sc
import scipy.integrate

import sharpy.presharpy.utils.algebra as algebra
import sharpy.utils.ctypes_utils as ct_utils
from sharpy.utils.sharpydir import SharpyDir


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
        self.NewmarkDamp = ct.c_double(1.0e-4)
        self.gravity_on = ct.c_bool(False)
        self.gravity = ct.c_double(0.0)
        self.gravity_dir_x = ct.c_double(0.0)
        self.gravity_dir_y = ct.c_double(0.0)
        self.gravity_dir_z = ct.c_double(1.0)


BeamLib = ct_utils.import_ctypes_lib(SharpyDir + '/lib/', 'libxbeam')

# ctypes pointer types
doubleP = ct.POINTER(ct.c_double)
intP = ct.POINTER(ct.c_int)
charP = ct.POINTER(ct.c_char_p)


f_cbeam3_solv_nlnstatic = BeamLib.cbeam3_solv_nlnstatic_python
f_cbeam3_solv_nlnstatic.restype = None


def cbeam3_solv_nlnstatic(beam, settings):
    """@brief Python wrapper for f_cbeam3_solv_nlnstatic

     Modified by Alfonso del Carre"""
    n_elem = ct.c_int(beam.num_elem)
    n_nodes = ct.c_int(beam.num_node)
    n_mass = ct.c_int(beam.n_mass)
    n_stiff = ct.c_int(beam.n_stiff)

    xbopts = Xbopts()
    xbopts.PrintInfo = ct.c_bool(settings['print_info'])
    xbopts.Solution = ct.c_int(112)
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

    # applied forces as 0=G, 1=a, 2=b
    # here we only need to set the flags at True, all the forces are follower
    xbopts.FollowerForce = ct.c_bool(True)
    xbopts.FollowerForceRig = ct.c_bool(True)

    f_cbeam3_solv_nlnstatic(ct.byref(n_elem),
                            ct.byref(n_nodes),
                            beam.num_nodes_matrix.ctypes.data_as(intP),
                            beam.num_mem_matrix.ctypes.data_as(intP),
                            beam.connectivities_fortran.ctypes.data_as(intP),
                            beam.master_nodes.ctypes.data_as(intP),
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
                            ct.byref(beam.n_app_forces),
                            beam.app_forces_fortran.ctypes.data_as(doubleP),
                            beam.node_app_forces_fortran.ctypes.data_as(intP)
                            )
    # angle = 0*np.pi/180
    # import presharpy.utils.algebra as algebra
    # rot = np.zeros((3, 3))
    # rot[0, :] = [np.cos(angle), -np.sin(angle), 0.0]
    # rot[1, :] = [np.sin(angle), np.cos(angle), 0.0]
    # rot[2, :] = [0, 0, 1.0]
    #
    # psi = beam.psi_def[-1, 2, :]
    # total = algebra.crv2rot(psi)
    #
    # def_rot = np.dot(rot.T, total)
    # psi_proj = algebra.rot2crv(def_rot)
    # print(psi_proj)
    # print(beam.pos_def[-1, :])
    # print(beam.psi_def[-1, 2, :])


f_cbeam3_solv_modal = BeamLib.cbeam3_solv_modal_python
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
                            beam.master_nodes.ctypes.data_as(intP),
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


f_cbeam3_solv_nlndyn = BeamLib.cbeam3_solv_nlndyn_python
f_cbeam3_solv_nlndyn.restype = None


def cbeam3_solv_nlndyn(beam, settings):

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
                         beam.master_nodes.ctypes.data_as(intP),
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
                         ct.byref(beam.n_app_forces),
                         beam.app_forces_fortran.ctypes.data_as(doubleP),
                         beam.node_app_forces_fortran.ctypes.data_as(intP),
                         beam.dynamic_forces_amplitude_fortran.ctypes.data_as(doubleP),
                         beam.dynamic_forces_time_fortran.ctypes.data_as(doubleP),
                         beam.forced_vel_fortran.ctypes.data_as(doubleP),
                         beam.forced_acc_fortran.ctypes.data_as(doubleP),
                         pos_def_history.ctypes.data_as(doubleP),
                         psi_def_history.ctypes.data_as(doubleP),
                         pos_dot_def_history.ctypes.data_as(doubleP),
                         psi_dot_def_history.ctypes.data_as(doubleP)
                         )

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    n_tsteps = int(3*n_tsteps.value/4)
    n_plots = 20
    delta = 5
    cm_subs = np.linspace(0, 1, n_plots)
    colors = [cm.jet(x) for x in cm_subs]
    plt.figure()
    plt.title('Time history of tip z displacements and y rotations')
    plt.xlabel('time (s)')
    plt.ylabel('z displacement (m), y rotation (rad)')
    plt.plot(time, pos_def_history[:, -1, 2], 'r')
    plt.plot(time, psi_def_history[:, -1, 2, 1], 'k')
    plt.grid(True)
    plt.show()
    plt.figure()
    for i in range(n_plots):
        plt.plot(pos_def_history[n_tsteps + i*delta, :, 0], pos_def_history[n_tsteps + i*delta, :, 2], color=colors[i])
    plt.grid(True)
    plt.show()


f_xbeam_solv_couplednlndyn = BeamLib.xbeam_solv_couplednlndyn_python
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
    beam.for_vel = np.zeros((n_tsteps + 1, 6), order='F')
    beam.for_acc = np.zeros((n_tsteps + 1, 6), order='F')
    beam.pos_def_history = np.zeros((n_tsteps, beam.num_node, 3), order='F')
    beam.pos_dot_def_history = np.zeros((n_tsteps, beam.num_node, 3), order='F')
    beam.psi_def_history = np.zeros((n_tsteps, beam.num_elem, 3, 3), order='F')
    beam.psi_dot_def_history = np.zeros((n_tsteps, beam.num_elem, 3, 3), order='F')
    beam.quat_history = np.zeros((n_tsteps, 4), order='F')

    dt = ct.c_double(dt)
    n_tsteps = ct.c_int(n_tsteps)

    xbopts = Xbopts()
    xbopts.PrintInfo = ct.c_bool(settings['print_info'])
    xbopts.Solution = ct.c_int(910)
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
                         beam.num_nodes_matrix.ctypes.data_as(intP),
                         beam.num_mem_matrix.ctypes.data_as(intP),
                         beam.connectivities_fortran.ctypes.data_as(intP),
                         beam.master_nodes.ctypes.data_as(intP),
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
                         ct.byref(beam.n_app_forces),
                         beam.app_forces_fortran.ctypes.data_as(doubleP),
                         beam.node_app_forces_fortran.ctypes.data_as(intP),
                         beam.dynamic_forces_amplitude_fortran.ctypes.data_as(doubleP),
                         beam.dynamic_forces_time_fortran.ctypes.data_as(doubleP),
                         beam.for_vel.ctypes.data_as(doubleP),
                         beam.for_acc.ctypes.data_as(doubleP),
                         beam.pos_def_history.ctypes.data_as(doubleP),
                         beam.psi_def_history.ctypes.data_as(doubleP),
                         beam.pos_dot_def_history.ctypes.data_as(doubleP),
                         beam.psi_dot_def_history.ctypes.data_as(doubleP),
                         beam.quat_history.ctypes.data_as(doubleP),
                         ct.byref(success)
                         )
    print("--- %s seconds ---" % (ti.time() - start_time))
    if not success:
        raise Exception('couplednlndyn did not converge')

    beam.for_pos = np.zeros_like(beam.for_vel)
    beam.for_pos[:, 0] = sc.integrate.cumtrapz(beam.for_vel[:, 0], dx=dt.value, initial=0)
    beam.for_pos[:, 1] = sc.integrate.cumtrapz(beam.for_vel[:, 1], dx=dt.value, initial=0)
    beam.for_pos[:, 2] = sc.integrate.cumtrapz(beam.for_vel[:, 2], dx=dt.value, initial=0)

    glob_pos_def = np.zeros_like(beam.pos_def_history)
    for it in range(n_tsteps.value):
        # rot = algebra.crv2rot(algebra.quat2crv(beam.quat_history[it, :]))
        rot = algebra.quat2rot(beam.quat_history[it, :])
        for inode in range(beam.num_node):
            glob_pos_def[it, inode, :] = beam.for_pos[it, 0:3] + np.dot(rot.T, beam.pos_def_history[it, inode, :])


    import matplotlib.pyplot as plt
    plt.figure()
    length = np.zeros((n_tsteps.value,))
    for it in range(n_tsteps.value):
        length[it] = np.linalg.norm(beam.pos_def_history[it, 0, :] - beam.pos_def_history[it, -1, :])

    plt.plot(time, length)
    plt.grid('on')
    plt.show()

    # gravity value
    g = -9.81
    z = 0.5*time**2*g + glob_pos_def[0, beam.num_node/2, 2]
    plt.figure()
    plt.plot(time, z, 'k--')
    plt.plot(time, glob_pos_def[:, beam.num_node/2, 2])
    plt.grid('on')
    plt.show()
    # plt.figure()
    # plt.hold('on')
    # plt.plot(time[:-1], pos_def_history[:-2, -1, 2], 'r')
    # plt.plot(time[:-1], pos_def_history[:-2, -1, 1], 'b')
    # plt.grid(True)
    # plt.show()
    # n_tsteps = settings['num_steps'].value/2
    # n_plots = 5
    # delta = 199
    # plt.figure()
    # for i in range(n_plots):
    #     plt.hold('on')
    #     plt.plot(beam.pos_def_history[n_tsteps + i*delta, :, 0], beam.pos_def_history[n_tsteps + i*delta, :, 2])
    # plt.grid(True)
    # plt.show()

    from matplotlib import cm

    delta = int(0.1/dt.value)
    itime = range(1, int(n_tsteps.value)-1, delta)
    cm_subs = np.linspace(0, 1, n_tsteps.value)
    colors = [cm.jet(x) for x in cm_subs]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plt.hold('on')
    for i in itime:
        ax.plot(glob_pos_def[i, :, 0], glob_pos_def[i, :, 1], glob_pos_def[i, :, 2], color=colors[i])
    ax.plot(beam.for_pos[:, 0], beam.for_pos[:, 1], beam.for_pos[:, 2], 'r', linewidth=1)
    if n_nodes.value % 2 == 1:
        half1 = int(n_nodes.value/2)
        half2 = half1 + 1
    else:
        half1 = half2 = n_nodes.value/2
    ax.plot(0.5*(glob_pos_def[:, half1, 0] + glob_pos_def[:, half2, 0]),
            0.5*(glob_pos_def[:, half1, 1] + glob_pos_def[:, half2, 1]),
            0.5*(glob_pos_def[:, half1, 2] + glob_pos_def[:, half2, 2]),
            'k', linewidth=1)


    ax.plot(glob_pos_def[:, -1, 0], glob_pos_def[:, -1, 1], glob_pos_def[:, -1, 2], 'b', linewidth=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis('equal')
    plt.show()

    a = 1


f_cbeam3_solv_update_static = BeamLib.cbeam3_solv_update_static_python
f_cbeam3_solv_update_static.restype = None


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
                                beam.master_nodes.ctypes.data_as(intP),
                                beam.num_nodes_matrix.ctypes.data_as(intP),
                                beam.psi_ini.ctypes.data_as(doubleP),
                                pos_def.ctypes.data_as(doubleP),
                                psi_def.ctypes.data_as(doubleP),
                                deltax.ctypes.data_as(doubleP))















