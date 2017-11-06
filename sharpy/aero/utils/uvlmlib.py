from sharpy.utils.sharpydir import SharpyDir
import sharpy.utils.ctypes_utils as ct_utils

import ctypes as ct
import numpy as np
import platform
import os

UvlmLib = ct_utils.import_ctypes_lib(SharpyDir + '/lib/', 'libuvlm')


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
                ("iterative_precond", ct.c_bool)]

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


class UVMopts(ct.Structure):
    _fields_ = [("dt", ct.c_double),
                ("NumCores", ct.c_uint),
                ("NumSurfaces", ct.c_uint),
                # ("steady_n_rollup", ct.c_uint),
                # ("steady_rollup_tolerance", ct.c_double),
                # ("steady_rollup_aic_refresh", ct.c_uint),
                ("convection_scheme", ct.c_uint),
                ("Mstar", ct.c_uint),
                ("ImageMethod", ct.c_bool),
                ("iterative_solver", ct.c_bool),
                ("iterative_tol", ct.c_double),
                ("iterative_precond", ct.c_bool)]

    def __init__(self):
        ct.Structure.__init__(self)
        self.dt = ct.c_double(0.01)
        self.NumCores = ct.c_uint(4)
        self.NumSurfaces = ct.c_uint(1)
        self.convection_scheme = ct.c_uint(2)
        self.Mstar = ct.c_uint(10)
        self.ImageMethod = ct.c_bool(False)
        self.iterative_solver = ct.c_bool(False)
        self.iterative_tol = ct.c_double(0)
        self.iterative_precond = ct.c_bool(False)


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
            ts_info.ct_p_u_ext,
            ts_info.ct_p_gamma,
            ts_info.ct_p_gamma_star,
            ts_info.ct_p_normals,
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


def uvlm_solver(i_iter, ts_info, previous_ts_info, struct_ts_info, options):
    run_UVLM = UvlmLib.run_UVLM
    run_UVLM.restype = None

    uvmopts = UVMopts()
    uvmopts.dt = ct.c_double(options["dt"].value)
    uvmopts.NumCores = ct.c_uint(options["num_cores"].value)
    uvmopts.NumSurfaces = ct.c_uint(ts_info.n_surf)
    uvmopts.ImageMethod = ct.c_bool(False)
    uvmopts.convection_scheme = ct.c_uint(options["convection_scheme"].value)
    uvmopts.iterative_solver = ct.c_bool(options['iterative_solver'].value)
    uvmopts.iterative_tol = ct.c_double(options['iterative_tol'].value)
    uvmopts.iterative_precond = ct.c_bool(options['iterative_precond'].value)

    flightconditions = FlightConditions()
    flightconditions.rho = options['rho']
    flightconditions.uinf = np.ctypeslib.as_ctypes(np.linalg.norm(ts_info.u_ext[0][:, 0, 0]))
    flightconditions.uinf_direction = np.ctypeslib.as_ctypes(ts_info.u_ext[0][:, 0, 0]/flightconditions.uinf)

    # for u_ext in ts_info.u_ext:
    #     u_ext[0, :, :] = flightconditions.uinf
    #     u_ext[1, :, :] = 0.0
    #     u_ext[2, :, :] = 0.0
    # if options['convection_scheme'] > 1:
    #     for u_ext in ts_info.u_ext_star:
    #         u_ext[0, :, :] = flightconditions.uinf
    #         u_ext[1, :, :] = 0.0
    #         u_ext[2, :, :] = 0.0

    rbm_vel = struct_ts_info.for_vel

    rbm_vel[0:3] = np.dot(struct_ts_info.cga().transpose(), rbm_vel[0:3])
    rbm_vel[3:6] = np.dot(struct_ts_info.cga().transpose(), rbm_vel[3:6])
    p_rbm_vel = rbm_vel.ctypes.data_as(ct.POINTER(ct.c_double))

    i = ct.c_uint(i_iter)
    ts_info.generate_ctypes_pointers()
    previous_ts_info.generate_ctypes_pointers()
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
             ts_info.ct_p_zeta_star_dot,
             p_rbm_vel,
             ts_info.ct_p_gamma,
             ts_info.ct_p_gamma_star,
             previous_ts_info.ct_p_gamma,
             ts_info.ct_p_normals,
             ts_info.ct_p_forces,
             ts_info.ct_p_dynamic_forces)
    ts_info.remove_ctypes_pointers()
    previous_ts_info.remove_ctypes_pointers()

