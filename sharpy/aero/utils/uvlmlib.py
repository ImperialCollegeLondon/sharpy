from sharpy.utils.sharpydir import SharpyDir
import sharpy.utils.ctypes_utils as ct_utils

import ctypes as ct
import numpy as np
import platform
import os
from sharpy.utils.constants import NDIM

UvlmLib = ct_utils.import_ctypes_lib(SharpyDir + '/lib/UVLM/lib/', 'libuvlm')


class VMopts(ct.Structure):
    """ctypes definition for VMopts class
        struct VMopts {
            bool ImageMethod;
            unsigned int Mstar;
            bool Steady;
            bool horseshoe;
            bool KJMeth;
            bool NewAIC;
            double DelTime;
            bool Rollup;
            unsigned int NumCores;
            unsigned int NumSurfaces;
            bool cfl1;
        };
    """
    _fields_ = [("ImageMethod", ct.c_bool),
                ("Steady", ct.c_bool),
                ("horseshoe", ct.c_bool),
                ("KJMeth", ct.c_bool),
                ("NewAIC", ct.c_bool),
                ("DelTime", ct.c_double),
                ("Rollup", ct.c_bool),
                ("NumCores", ct.c_uint),
                ("NumSurfaces", ct.c_uint),
                ("dt", ct.c_double),
                ("n_rollup", ct.c_uint),
                ("rollup_tolerance", ct.c_double),
                ("rollup_aic_refresh", ct.c_uint),
                ("iterative_solver", ct.c_bool),
                ("iterative_tol", ct.c_double),
                ("iterative_precond", ct.c_bool),
                ("cfl1", ct.c_bool)]

    def __init__(self):
        ct.Structure.__init__(self)
        self.ImageMethod = ct.c_bool(False)
        self.Steady = ct.c_bool(True)
        self.horseshoe = ct.c_bool(True)
        self.KJMeth = ct.c_bool(False)  # legacy var
        self.NewAIC = ct.c_bool(False)  # legacy var
        self.DelTime = ct.c_double(1.0)
        self.Rollup = ct.c_bool(False)
        self.NumCores = ct.c_uint(4)
        self.NumSurfaces = ct.c_uint(1)
        self.dt = ct.c_double(0.01)
        self.n_rollup = ct.c_uint(0)
        self.rollup_tolerance = ct.c_double(1e-5)
        self.rollup_aic_refresh = ct.c_uint(1)
        self.iterative_solver = ct.c_bool(False)
        self.iterative_tol = ct.c_double(0)
        self.iterative_precond = ct.c_bool(False)
        self.cfl1 = ct.c_bool(True)
        self.rbm_vel_g = np.ctypeslib.as_ctypes(np.zeros((6)))


class UVMopts(ct.Structure):
    _fields_ = [("dt", ct.c_double),
                ("NumCores", ct.c_uint),
                ("NumSurfaces", ct.c_uint),
                # ("steady_n_rollup", ct.c_uint),
                # ("steady_rollup_tolerance", ct.c_double),
                # ("steady_rollup_aic_refresh", ct.c_uint),
                ("convection_scheme", ct.c_uint),
                # ("Mstar", ct.c_uint),
                ("ImageMethod", ct.c_bool),
                ("iterative_solver", ct.c_bool),
                ("iterative_tol", ct.c_double),
                ("iterative_precond", ct.c_bool),
                ("convect_wake", ct.c_bool),
                ("cfl1", ct.c_bool)]

    def __init__(self):
        ct.Structure.__init__(self)
        self.dt = ct.c_double(0.01)
        self.NumCores = ct.c_uint(4)
        self.NumSurfaces = ct.c_uint(1)
        self.convection_scheme = ct.c_uint(2)
        # self.Mstar = ct.c_uint(10)
        self.ImageMethod = ct.c_bool(False)
        self.iterative_solver = ct.c_bool(False)
        self.iterative_tol = ct.c_double(0)
        self.iterative_precond = ct.c_bool(False)
        self.convect_wake = ct.c_bool(True)
        self.cfl1 = ct.c_bool(True)


class FlightConditions(ct.Structure):
    _fields_ = [("uinf", ct.c_double),
                ("uinf_direction", ct.c_double*3),
                ("rho", ct.c_double),
                ("c_ref", ct.c_double)]

    def __init__(self):
        ct.Structure.__init__(self)

    # def __init__(self, fc_dict):
    #     ct.Structure.__init__(self)
    #     self.uinf = fc_dict['FlightCon']['u_inf']
    #     alpha = fc_dict['FlightCon']['alpha']
    #     beta = fc_dict['FlightCon']['beta']
    #     uinf_direction_temp = np.array([1, 0, 0], dtype=ct.c_double)
    #     self.uinf_direction = np.ctypeslib.as_ctypes(uinf_direction_temp)
    #     self.rho = fc_dict['FlightCon']['rho_inf']
    #     self.c_ref = fc_dict['FlightCon']['c_ref']


class SHWOptions(ct.Structure):
    _fields_ = [("dt", ct.c_double),
                ("rot_center", ct.c_double*3),
                ("rot_vel", ct.c_double),
                ("rot_axis", ct.c_double*3)]

    def __init__(self):
        ct.Structure.__init__(self)

# type for 2d integer matrix
t_2int = ct.POINTER(ct.c_int)*2


def vlm_solver(ts_info, options):
    run_VLM = UvlmLib.run_VLM
    run_VLM.restype = None

    vmopts = VMopts()
    vmopts.Steady = ct.c_bool(True)
    vmopts.NumSurfaces = ct.c_uint(ts_info.n_surf)
    vmopts.horseshoe = ct.c_bool(options['horseshoe'].value)
    vmopts.dt = ct.c_double(options["rollup_dt"].value)
    vmopts.n_rollup = ct.c_uint(options["n_rollup"].value)
    vmopts.rollup_tolerance = ct.c_double(options["rollup_tolerance"].value)
    vmopts.rollup_aic_refresh = ct.c_uint(options['rollup_aic_refresh'].value)
    vmopts.NumCores = ct.c_uint(options['num_cores'].value)
    vmopts.iterative_solver = ct.c_bool(options['iterative_solver'].value)
    vmopts.iterative_tol = ct.c_double(options['iterative_tol'].value)
    vmopts.iterative_precond = ct.c_bool(options['iterative_precond'].value)
    vmopts.cfl1 = ct.c_bool(options['cfl1'])
    vmopts.rbm_vel_g = np.ctypeslib.as_ctypes(options['rbm_vel_g'])

    flightconditions = FlightConditions()
    flightconditions.rho = options['rho']
    flightconditions.uinf = np.ctypeslib.as_ctypes(np.linalg.norm(ts_info.u_ext[0][:, 0, 0]))
    flightconditions.uinf_direction = np.ctypeslib.as_ctypes(ts_info.u_ext[0][:, 0, 0]/flightconditions.uinf)

    ts_info.generate_ctypes_pointers()
    run_VLM(ct.byref(vmopts),
            ct.byref(flightconditions),
            ts_info.ct_p_dimensions,
            ts_info.ct_p_dimensions_star,
            ts_info.ct_p_zeta,
            ts_info.ct_p_zeta_star,
            ts_info.ct_p_zeta_dot,
            ts_info.ct_p_u_ext,
            ts_info.ct_p_gamma,
            ts_info.ct_p_gamma_star,
            ts_info.ct_p_forces)
    ts_info.remove_ctypes_pointers()


def uvlm_init(ts_info, options):
    init_UVLM = UvlmLib.init_UVLM
    init_UVLM.restype = None

    vmopts = VMopts()
    vmopts.Steady = ct.c_bool(True)
    # vmopts.Mstar = ct.c_uint(options['mstar'])
    vmopts.NumSurfaces = ct.c_uint(ts_info.n_surf)
    vmopts.horseshoe = ct.c_bool(False)
    vmopts.dt = options["dt"]
    try:
        vmopts.n_rollup = ct.c_uint(options["steady_n_rollup"].value)
        vmopts.rollup_tolerance = ct.c_double(options["steady_rollup_tolerance"].value)
        vmopts.rollup_aic_refresh = ct.c_uint(options['steady_rollup_aic_refresh'].value)
    except KeyError:
        pass
    vmopts.NumCores = ct.c_uint(options['num_cores'].value)

    flightconditions = FlightConditions()
    flightconditions.rho = options['rho']
    flightconditions.uinf = np.ctypeslib.as_ctypes(np.linalg.norm(ts_info.u_ext[0][:, 0, 0]))
    flightconditions.uinf_direction = np.ctypeslib.as_ctypes(ts_info.u_ext[0][:, 0, 0]/flightconditions.uinf)

    # rbm_vel[0:3] = np.dot(inertial2aero.transpose(), rbm_vel[0:3])
    # rbm_vel[3:6] = np.dot(inertial2aero.transpose(), rbm_vel[3:6])
    p_rbm_vel = np.zeros((6,)).ctypes.data_as(ct.POINTER(ct.c_double))

    ts_info.generate_ctypes_pointers()
    init_UVLM(ct.byref(vmopts),
              ct.byref(flightconditions),
              ts_info.ct_p_dimensions,
              ts_info.ct_p_dimensions_star,
              ts_info.ct_p_u_ext,
              ts_info.ct_p_zeta,
              ts_info.ct_p_zeta_star,
              ts_info.ct_p_zeta_dot,
              ts_info.ct_p_zeta_star_dot,
              p_rbm_vel,
              ts_info.ct_p_gamma,
              ts_info.ct_p_gamma_star,
              ts_info.ct_p_normals,
              ts_info.ct_p_forces)
    ts_info.remove_ctypes_pointers()


def uvlm_solver(i_iter, ts_info, struct_ts_info, options, convect_wake=True, dt=None):
    run_UVLM = UvlmLib.run_UVLM
    run_UVLM.restype = None

    uvmopts = UVMopts()
    if dt is None:
        uvmopts.dt = ct.c_double(options["dt"].value)
    else:
        uvmopts.dt = ct.c_double(dt)
    uvmopts.NumCores = ct.c_uint(options["num_cores"].value)
    uvmopts.NumSurfaces = ct.c_uint(ts_info.n_surf)
    uvmopts.ImageMethod = ct.c_bool(False)
    uvmopts.convection_scheme = ct.c_uint(options["convection_scheme"].value)
    uvmopts.iterative_solver = ct.c_bool(options['iterative_solver'].value)
    uvmopts.iterative_tol = ct.c_double(options['iterative_tol'].value)
    uvmopts.iterative_precond = ct.c_bool(options['iterative_precond'].value)
    uvmopts.convect_wake = ct.c_bool(convect_wake)
    uvmopts.cfl1 = ct.c_bool(options['cfl1'])

    flightconditions = FlightConditions()
    flightconditions.rho = options['rho']
    flightconditions.uinf = np.ctypeslib.as_ctypes(np.linalg.norm(ts_info.u_ext[0][:, 0, 0]))
    # direction = np.array([1.0, 0, 0])
    flightconditions.uinf_direction = np.ctypeslib.as_ctypes(ts_info.u_ext[0][:, 0, 0]/flightconditions.uinf)
    # flightconditions.uinf_direction = np.ctypeslib.as_ctypes(direction)

    rbm_vel = struct_ts_info.for_vel.copy()
    rbm_vel[0:3] = np.dot(struct_ts_info.cga(), rbm_vel[0:3])
    rbm_vel[3:6] = np.dot(struct_ts_info.cga(), rbm_vel[3:6])
    p_rbm_vel = rbm_vel.ctypes.data_as(ct.POINTER(ct.c_double))

    i = ct.c_uint(i_iter)
    ts_info.generate_ctypes_pointers()
    # previous_ts_info.generate_ctypes_pointers()
    run_UVLM(ct.byref(uvmopts),
             ct.byref(flightconditions),
             ts_info.ct_p_dimensions,
             ts_info.ct_p_dimensions_star,
             ct.byref(i),
             ts_info.ct_p_u_ext,
             ts_info.ct_p_u_ext_star,
             ts_info.ct_p_zeta,
             ts_info.ct_p_zeta_star,
             ts_info.ct_p_zeta_dot,
             p_rbm_vel,
             ts_info.ct_p_gamma,
             ts_info.ct_p_gamma_star,
             # previous_ts_info.ct_p_gamma,
             ts_info.ct_p_normals,
             ts_info.ct_p_forces,
             ts_info.ct_p_dynamic_forces)
    ts_info.remove_ctypes_pointers()
    # previous_ts_info.remove_ctypes_pointers()


def shw_solver(i_iter, ts_info, struct_ts_info, options, convect_wake=True, dt=None):
    run_SHW = UvlmLib.run_SHW
    run_SHW.restype = None

    uvmopts = UVMopts()
    if dt is None:
        uvmopts.dt = ct.c_double(options["dt"].value)
    else:
        uvmopts.dt = ct.c_double(dt)
    uvmopts.NumCores = ct.c_uint(options["num_cores"].value)
    uvmopts.NumSurfaces = ct.c_uint(ts_info.n_surf)
    uvmopts.ImageMethod = ct.c_bool(False)
    uvmopts.convection_scheme = ct.c_uint(options["convection_scheme"].value)
    uvmopts.iterative_solver = ct.c_bool(options['iterative_solver'].value)
    uvmopts.iterative_tol = ct.c_double(options['iterative_tol'].value)
    uvmopts.iterative_precond = ct.c_bool(options['iterative_precond'].value)
    uvmopts.convect_wake = ct.c_bool(convect_wake)

    shwopts = SHWOptions()
    shwopts.dt = uvmopts.dt
    shwopts.rot_center = np.ctypeslib.as_ctypes(options['rot_center'])
    shwopts.rot_vel = options['rot_vel']
    shwopts.rot_axis = np.ctypeslib.as_ctypes(options['rot_axis'])

    flightconditions = FlightConditions()
    flightconditions.rho = options['rho']
    flightconditions.uinf = np.ctypeslib.as_ctypes(np.linalg.norm(ts_info.u_ext[0][:, 0, 0]))
    # direction = np.array([1.0, 0, 0])
    flightconditions.uinf_direction = np.ctypeslib.as_ctypes(ts_info.u_ext[0][:, 0, 0]/flightconditions.uinf)
    # flightconditions.uinf_direction = np.ctypeslib.as_ctypes(direction)

    rbm_vel = struct_ts_info.for_vel.copy()
    rbm_vel[0:3] = np.dot(struct_ts_info.cga(), rbm_vel[0:3])
    rbm_vel[3:6] = np.dot(struct_ts_info.cga(), rbm_vel[3:6])
    p_rbm_vel = rbm_vel.ctypes.data_as(ct.POINTER(ct.c_double))

    i = ct.c_uint(i_iter)
    ts_info.generate_ctypes_pointers()
    # previous_ts_info.generate_ctypes_pointers()
    run_SHW(ct.byref(uvmopts),
             ct.byref(flightconditions),
             ct.byref(shwopts),
             ts_info.ct_p_dimensions,
             ts_info.ct_p_dimensions_star,
             ct.byref(i),
             ts_info.ct_p_u_ext,
             ts_info.ct_p_u_ext_star,
             ts_info.ct_p_zeta,
             ts_info.ct_p_zeta_star,
             ts_info.ct_p_zeta_dot,
             p_rbm_vel,
             ts_info.ct_p_gamma,
             ts_info.ct_p_gamma_star,
             # previous_ts_info.ct_p_gamma,
             ts_info.ct_p_normals,
             ts_info.ct_p_forces,
             ts_info.ct_p_dynamic_forces)
    ts_info.remove_ctypes_pointers()


def uvlm_calculate_unsteady_forces(ts_info,
                                   struct_ts_info,
                                   options,
                                   convect_wake=True,
                                   dt=None):
    calculate_unsteady_forces = UvlmLib.calculate_unsteady_forces
    calculate_unsteady_forces.restype = None

    uvmopts = UVMopts()
    if dt is None:
        uvmopts.dt = ct.c_double(options["dt"].value)
    else:
        uvmopts.dt = ct.c_double(dt)
    uvmopts.NumCores = ct.c_uint(options["num_cores"].value)
    uvmopts.NumSurfaces = ct.c_uint(ts_info.n_surf)
    uvmopts.ImageMethod = ct.c_bool(False)
    uvmopts.convection_scheme = ct.c_uint(options["convection_scheme"].value)
    uvmopts.iterative_solver = ct.c_bool(options['iterative_solver'].value)
    uvmopts.iterative_tol = ct.c_double(options['iterative_tol'].value)
    uvmopts.iterative_precond = ct.c_bool(options['iterative_precond'].value)
    uvmopts.convect_wake = ct.c_bool(convect_wake)

    flightconditions = FlightConditions()
    flightconditions.rho = options['rho']
    flightconditions.uinf = np.ctypeslib.as_ctypes(np.linalg.norm(ts_info.u_ext[0][:, 0, 0]))
    flightconditions.uinf_direction = np.ctypeslib.as_ctypes(ts_info.u_ext[0][:, 0, 0]/flightconditions.uinf)

    rbm_vel = struct_ts_info.for_vel.copy()
    rbm_vel[0:3] = np.dot(struct_ts_info.cga(), rbm_vel[0:3])
    rbm_vel[3:6] = np.dot(struct_ts_info.cga(), rbm_vel[3:6])
    p_rbm_vel = rbm_vel.ctypes.data_as(ct.POINTER(ct.c_double))

    for i_surf in range(ts_info.n_surf):
        ts_info.dynamic_forces[i_surf].fill(0.0)

    ts_info.generate_ctypes_pointers()
    calculate_unsteady_forces(ct.byref(uvmopts),
                              ct.byref(flightconditions),
                              ts_info.ct_p_dimensions,
                              ts_info.ct_p_dimensions_star,
                              ts_info.ct_p_zeta,
                              ts_info.ct_p_zeta_star,
                              p_rbm_vel,
                              ts_info.ct_p_gamma,
                              ts_info.ct_p_gamma_star,
                              ts_info.ct_p_gamma_dot,
                              ts_info.ct_p_normals,
                              ts_info.ct_p_dynamic_forces)
    ts_info.remove_ctypes_pointers()


def uvlm_calculate_incidence_angle(ts_info,
                                   struct_ts_info):
    calculate_incidence_angle = UvlmLib.UVLM_check_incidence_angle
    calculate_incidence_angle.restype = None

    rbm_vel = struct_ts_info.for_vel.copy()
    rbm_vel[0:3] = np.dot(struct_ts_info.cga(), rbm_vel[0:3])
    rbm_vel[3:6] = np.dot(struct_ts_info.cga(), rbm_vel[3:6])
    p_rbm_vel = rbm_vel.ctypes.data_as(ct.POINTER(ct.c_double))

    n_surf = ct.c_uint(ts_info.n_surf)

    ts_info.generate_ctypes_pointers()
    calculate_incidence_angle(ct.byref(n_surf),
                              ts_info.ct_p_dimensions,
                              ts_info.ct_p_u_ext,
                              ts_info.ct_p_zeta,
                              ts_info.ct_p_zeta_dot,
                              ts_info.ct_p_normals,
                              p_rbm_vel,
                              ts_info.postproc_cell['incidence_angle_ct_pointer'])
    ts_info.remove_ctypes_pointers()

def uvlm_calculate_total_induced_velocity_at_points(ts_info,
                                                   target_triads,
                                                   for_pos=np.zeros((6)),
                                                   ncores=ct.c_uint(1)):
    """
    uvlm_calculate_total_induced_velocity_at_points

    Caller to the UVLM library to compute the induced velocity of all the
    surfaces and wakes at a list of points

    Args:
        ts_info (AeroTimeStepInfo): Time step information
        target_triads (np.array): Point coordinates, size=(npoints, 3)
        uind (np.array): Induced velocity

    Returns:
    	uind (np.array): Induced velocity, size=(npoints, 3)

    """
    calculate_uind_at_points = UvlmLib.total_induced_velocity_at_points
    calculate_uind_at_points.restype = None

    uvmopts = UVMopts()
    uvmopts.NumSurfaces = ct.c_uint(ts_info.n_surf)
    uvmopts.ImageMethod = ct.c_bool(False)
    uvmopts.NumCores = ct.c_uint(ncores.value)

    npoints = target_triads.shape[0]
    uind = np.zeros((npoints, 3), dtype=ct.c_double)

    if type(target_triads[0,0]) == ct.c_double:
        aux_target_triads = target_triads
    else:
        aux_target_triads = target_triads.astype(dtype=ct.c_double)

    p_target_triads = ((ct.POINTER(ct.c_double))(* [np.ctypeslib.as_ctypes(aux_target_triads.reshape(-1))]))
    p_uind = ((ct.POINTER(ct.c_double))(* [np.ctypeslib.as_ctypes(uind.reshape(-1))]))

    # make a copy of ts info and add for_pos to zeta and zeta_star
    ts_info_copy = ts_info.copy()
    for i_surf in range(ts_info_copy.n_surf):
        # zeta
        for iM in range(ts_info_copy.zeta[i_surf].shape[1]):
            for iN in range(ts_info_copy.zeta[i_surf].shape[2]):
                ts_info_copy.zeta[i_surf][:, iM, iN] += for_pos[0:3]
        # zeta_star
        for iM in range(ts_info_copy.zeta_star[i_surf].shape[1]):
            for iN in range(ts_info_copy.zeta_star[i_surf].shape[2]):
                ts_info_copy.zeta_star[i_surf][:, iM, iN] += for_pos[0:3]

    ts_info_copy.generate_ctypes_pointers()
    calculate_uind_at_points(ct.byref(uvmopts),
                              ts_info_copy.ct_p_dimensions,
                              ts_info_copy.ct_p_dimensions_star,
                              ts_info_copy.ct_p_zeta,
                              ts_info_copy.ct_p_zeta_star,
                              ts_info_copy.ct_p_gamma,
                              ts_info_copy.ct_p_gamma_star,
                              p_target_triads,
                              p_uind,
                              ct.c_uint(npoints))
    ts_info_copy.remove_ctypes_pointers()
    del p_uind
    del p_target_triads

    return uind


def biot_panel_cpp(zeta_point, zeta_panel, gamma=1.0):
    """
    Linear UVLM function

    Returns the induced velocity at a point ``zeta_point`` due to a panel located at ``zeta_panel`` with circulation
    ``gamma``.

    Args:
        zeta_point (np.ndarray): Coordinates of the point with size ``(3,)``.
        zeta_panel (np.ndarray): Panel coordinates with size ``(4, 3)``.
        gamma (float): Panel circulation.

    Returns:
        np.ndarray: Induced velocity at point

    """

    assert zeta_point.flags['C_CONTIGUOUS'] and zeta_panel.flags['C_CONTIGUOUS'], \
        'Input not C contiguous'

    velP = np.zeros((3,), order='C')
    UvlmLib.call_biot_panel(
        velP.ctypes.data_as(ct.POINTER(ct.c_double)),
        zeta_point.ctypes.data_as(ct.POINTER(ct.c_double)),
        zeta_panel.ctypes.data_as(ct.POINTER(ct.c_double)),
        ct.byref(ct.c_double(gamma)))

    return velP


def eval_panel_cpp(zeta_point, zeta_panel, gamma_pan=1.0):
    """
    Linear UVLM function

    Returns
        tuple: The derivative of the induced velocity with respect to point ``P`` and panel vertices ``ZetaP``.

    Warnings:
        Function may fail if zeta_point is not stored contiguously.

        Eg:

        The following will fail

            zeta_point=Mat[:,2,5]
            eval_panel_cpp(zeta_point,zeta_panel,gamma_pan=1.0)

        but

            zeta_point=Mat[:,2,5].copy()
            eval_panel_cpp(zeta_point,zeta_panel,gamma_pan=1.0)

        will not.
    """

    assert zeta_point.flags['C_CONTIGUOUS'] and zeta_panel.flags['C_CONTIGUOUS'], \
        'Input not C contiguous'

    der_point = np.zeros((3, 3), order='C')
    der_vertices = np.zeros((4, 3, 3), order='C')

    UvlmLib.call_der_biot_panel(
        der_point.ctypes.data_as(ct.POINTER(ct.c_double)),
        der_vertices.ctypes.data_as(ct.POINTER(ct.c_double)),
        zeta_point.ctypes.data_as(ct.POINTER(ct.c_double)),
        zeta_panel.ctypes.data_as(ct.POINTER(ct.c_double)),
        ct.byref(ct.c_double(gamma_pan)))

    return der_point, der_vertices


def get_induced_velocity_cpp(maps, zeta, gamma, zeta_target):
    """
    Linear UVLM function used in bound surfaces

    Computes induced velocity at a point zeta_target.

    Args:
        maps (sharpy.linear.src.surface.AeroGridSurface): instance of bound surface
        zeta (np.ndarray): Coordinates of panel
        gamma (float): Panel circulation strength
        zeta_target (np.ndarray): Coordinates of target point

    Returns:
        np.ndarray: Induced velocity by panel at target point

    """
    call_ind_vel = UvlmLib.call_ind_vel
    call_ind_vel.restype = None

    assert zeta_target.flags['C_CONTIGUOUS'], "Input not C contiguous"

    M, N = maps.M, maps.N
    uind_target = np.zeros((3,), order='C')

    call_ind_vel(
        uind_target.ctypes.data_as(ct.POINTER(ct.c_double)),
        zeta_target.ctypes.data_as(ct.POINTER(ct.c_double)),
        zeta.ctypes.data_as(ct.POINTER(ct.c_double)),
        gamma.ctypes.data_as(ct.POINTER(ct.c_double)),
        ct.byref(ct.c_int(M)),
        ct.byref(ct.c_int(N)))

    return uind_target


def get_aic3_cpp(maps, zeta, zeta_target):
    """
    Linear UVLM function used in bound surfaces

    Produces influence coefficient matrix to calculate the induced velocity
    at a target point. The aic3 matrix has shape (3,K)

    Args:
        maps (sharpy.linear.src.surface.AeroGridSurface): instance of linear bound surface
        zeta (np.ndarray): Coordinates of panel
        zeta_target (np.ndarray): Coordinates of target point

    Returns:
        np.ndarray: Aerodynamic influence coefficient
    """

    assert zeta_target.flags['C_CONTIGUOUS'], "Input not C contiguous"

    K = maps.K
    aic3 = np.zeros((3, K), order='C')

    UvlmLib.call_aic3(
        aic3.ctypes.data_as(ct.POINTER(ct.c_double)),
        zeta_target.ctypes.data_as(ct.POINTER(ct.c_double)),
        zeta.ctypes.data_as(ct.POINTER(ct.c_double)),
        ct.byref(ct.c_int(maps.M)),
        ct.byref(ct.c_int(maps.N)))

    return aic3


def dvinddzeta_cpp(zetac, surf_in, is_bound, M_in_bound=None):
    """
    Linear UVLM function used in the assembly of the linear system

    Produces derivatives of induced velocity by surf_in w.r.t. the zetac point.
    Derivatives are divided into those associated to the movement of zetac, and
    to the movement of the surf_in vertices (DerVert).

    If surf_in is bound (is_bound==True), the circulation over the TE due to the
    wake is not included in the input.

    If surf_in is a wake (is_bound==False), derivatives w.r.t. collocation
    points are computed ad the TE contribution on ``der_vert``. In this case, the
    chordwise paneling Min_bound of the associated input is required so as to
    calculate Kzeta and correctly allocate the derivative matrix.

    Returns:
         tuple: output derivatives are:
                - der_coll: 3 x 3 matrix
                - der_vert: 3 x 3*Kzeta (if surf_in is a wake, Kzeta is that of the bound)

    Warning:
        zetac must be contiguously stored!
    """

    M_in, N_in = surf_in.maps.M, surf_in.maps.N
    Kzeta_in = surf_in.maps.Kzeta
    shape_zeta_in = (3, M_in + 1, N_in + 1)

    # allocate matrices
    der_coll = np.zeros((3, 3), order='C')

    if is_bound:
        M_in_bound = M_in
    Kzeta_in_bound = (M_in_bound + 1) * (N_in + 1)
    der_vert = np.zeros((3, 3 * Kzeta_in_bound))

    UvlmLib.call_dvinddzeta(
        der_coll.ctypes.data_as(ct.POINTER(ct.c_double)),
        der_vert.ctypes.data_as(ct.POINTER(ct.c_double)),
        zetac.ctypes.data_as(ct.POINTER(ct.c_double)),
        surf_in.zeta.ctypes.data_as(ct.POINTER(ct.c_double)),
        surf_in.gamma.ctypes.data_as(ct.POINTER(ct.c_double)),
        ct.byref(ct.c_int(M_in)),
        ct.byref(ct.c_int(N_in)),
        ct.byref(ct.c_bool(is_bound)),
        ct.byref(ct.c_int(M_in_bound)),
    )

    return der_coll, der_vert
