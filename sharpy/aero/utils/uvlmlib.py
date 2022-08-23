from sharpy.utils.sharpydir import SharpyDir
import sharpy.utils.ctypes_utils as ct_utils

import ctypes as ct
from ctypes import *
import numpy as np
import platform
import os
from sharpy.utils.constants import NDIM, vortex_radius_def

try:
    UvlmLib = ct_utils.import_ctypes_lib(SharpyDir + '/UVLM', 'libuvlm')
except OSError:
    UvlmLib = ct_utils.import_ctypes_lib(SharpyDir + '/lib/UVLM/lib', 'libuvlm')



# TODO: Combine VMOpts and UVMOpts (Class + inheritance)?
# TODO: Combine solver functions (e.g. vlm solver is able to start nonlifting only, lifitng only, nonlifting and lifting coupled) 

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
            bool only_lifting;
            bool only_nonlifting;
            unsigned int NumCores;
            unsigned int NumSurfaces;
            unsigned int NumSurfacesNonlifting;
            bool cfl1;
            double vortex_radius;
            double vortex_radius_wake_ind;
            double* centre_rot_g[3];
            double* rbm_vel_g[6];
        };
    """
    _fields_ = [("ImageMethod", ct.c_bool),
                ("Steady", ct.c_bool),
                ("horseshoe", ct.c_bool),
                ("KJMeth", ct.c_bool),
                ("NewAIC", ct.c_bool),
                ("DelTime", ct.c_double),
                ("Rollup", ct.c_bool),
                ("only_lifting", ct.c_bool),
                ("only_nonlifting", ct.c_bool),
                ("NumCores", ct.c_uint),
                ("NumSurfaces", ct.c_uint),
                ("NumSurfacesNonlifting", ct.c_uint),
                ("dt", ct.c_double),
                ("n_rollup", ct.c_uint),
                ("rollup_tolerance", ct.c_double),
                ("rollup_aic_refresh", ct.c_uint),
                ("iterative_solver", ct.c_bool),
                ("iterative_tol", ct.c_double),
                ("iterative_precond", ct.c_bool),
                ("cfl1", ct.c_bool),
                ("vortex_radius", ct.c_double),
                ("vortex_radius_wake_ind", ct.c_double),
                ("centre_rot_g", ct.c_double * 3),
                ("rbm_vel_g", ct.c_double * 6)]
    


    def __init__(self):
        ct.Structure.__init__(self)
        self.ImageMethod = ct.c_bool(False)
        self.Steady = ct.c_bool(True)
        self.horseshoe = ct.c_bool(True)
        self.KJMeth = ct.c_bool(False)  # legacy var
        self.NewAIC = ct.c_bool(False)  # legacy var
        self.DelTime = ct.c_double(1.0)
        self.Rollup = ct.c_bool(False)
        self.only_lifting = ct.c_bool(False)
        self.only_nonlifting = ct.c_bool(False)
        self.NumCores = ct.c_uint(4)
        self.NumSurfaces = ct.c_uint(1)
        self.NumSurfacesNonlifting = ct.c_uint(0)
        self.dt = ct.c_double(0.01)
        self.n_rollup = ct.c_uint(0)
        self.rollup_tolerance = ct.c_double(1e-5)
        self.rollup_aic_refresh = ct.c_uint(1)
        self.iterative_solver = ct.c_bool(False)
        self.iterative_tol = ct.c_double(0)
        self.iterative_precond = ct.c_bool(False)
        self.cfl1 = ct.c_bool(True)
        self.vortex_radius = ct.c_double(vortex_radius_def)
        self.vortex_radius_wake_ind = ct.c_double(vortex_radius_def)
        self.centre_rot_g = np.ctypeslib.as_ctypes(np.zeros((3)))
        self.rbm_vel_g = np.ctypeslib.as_ctypes(np.zeros((6)))


    def set_options(self, options, n_surfaces = 0, n_surfaces_nonlifting = 0, rbm_vel_g = np.zeros(6)):
        self.Steady = ct.c_bool(True)
        self.NumSurfaces = ct.c_uint(n_surfaces)
        self.NumSurfacesNonlifting = ct.c_uint(n_surfaces_nonlifting)
        self.horseshoe = ct.c_bool(options['horseshoe'])
        self.dt = ct.c_double(options["rollup_dt"])
        self.n_rollup = ct.c_uint(options["n_rollup"])
        self.rollup_tolerance = ct.c_double(options["rollup_tolerance"])
        self.rollup_aic_refresh = ct.c_uint(options['rollup_aic_refresh'])
        self.NumCores = ct.c_uint(options['num_cores'])
        self.iterative_solver = ct.c_bool(options['iterative_solver'])
        self.iterative_tol = ct.c_double(options['iterative_tol'])
        self.iterative_precond = ct.c_bool(options['iterative_precond'])
        self.cfl1 = ct.c_bool(options['cfl1'])
        self.vortex_radius = ct.c_double(options['vortex_radius'])
        self.vortex_radius_wake_ind = ct.c_double(options['vortex_radius_wake_ind'])

        self.only_nonlifting = ct.c_bool(options["only_nonlifting"])
        self.only_lifting = ct.c_bool(not options["nonlifting_body_interactions"])

        for i in range(len(options["centre_rot_g"])):
            self.centre_rot_g[i] = ct.c_double(options["centre_rot_g"][i])
        for i in range(len(rbm_vel_g)):            
            self.rbm_vel_g[i] = ct.c_double(rbm_vel_g[i])

class UVMopts(ct.Structure):
    # TODO: add set_options function
    # TODO: possible to combine with VMopts?
    _fields_ = [("dt", ct.c_double),
                ("NumCores", ct.c_uint),
                ("NumSurfaces", ct.c_uint),
                ("NumSurfacesNonlifting", ct.c_uint),            
                ("only_lifting", ct.c_bool),
                ("only_nonlifting", ct.c_bool),  
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
                ("cfl1", ct.c_bool),
                ("vortex_radius", ct.c_double),
                ("vortex_radius_wake_ind", ct.c_double),
                ("interp_coords", ct.c_uint),
                ("filter_method", ct.c_uint),
                ("interp_method", ct.c_uint),
                ("yaw_slerp", ct.c_double),
                ("quasi_steady", ct.c_bool),
                ("centre_rot_g", ct.c_double * 3),
                ("rbm_vel_g", ct.c_double * 6),
                ("num_spanwise_panels_wo_induced_velocity", ct.c_uint)]

    def __init__(self):
        ct.Structure.__init__(self)
        self.dt = ct.c_double(0.01)
        self.NumCores = ct.c_uint(4)
        self.NumSurfaces = ct.c_uint(1)
        self.NumSurfacesNonlifting = ct.c_uint(1)
        self.convection_scheme = ct.c_uint(2)
        # self.Mstar = ct.c_uint(10)
        self.ImageMethod = ct.c_bool(False)
        self.iterative_solver = ct.c_bool(False)
        self.iterative_tol = ct.c_double(0)
        self.iterative_precond = ct.c_bool(False)
        self.convect_wake = ct.c_bool(True)
        self.cfl1 = ct.c_bool(True)
        self.vortex_radius = ct.c_double(vortex_radius_def)
        self.vortex_radius_wake_ind = ct.c_double(vortex_radius_def)
        self.yaw_slerp = ct.c_double(0.)
        self.quasi_steady = ct.c_bool(False)
        self.centre_rot_g = np.ctypeslib.as_ctypes(np.zeros((3)))
        self.rbm_vel_g = np.ctypeslib.as_ctypes(np.zeros((6)))
        self.num_spanwise_panels_wo_induced_velocity = ct.c_uint(0)

    def set_options(self, 
                    options, 
                    n_surfaces = 0, 
                    n_surfaces_nonlifting = 0, 
                    dt = None, 
                    convect_wake = False, 
                    rbm_vel_g = np.zeros(6),
                    image_method = False,
                     n_span_panels_wo_u_ind = 0):
        if dt is None:
            self.dt = ct.c_double(options["dt"])
        else:
            self.dt = ct.c_double(dt)
        self.NumCores = ct.c_uint(options["num_cores"])
        self.NumSurfaces = ct.c_uint(n_surfaces)
        self.NumSurfacesNonlifting = ct.c_uint(n_surfaces_nonlifting)
        self.ImageMethod = ct.c_bool(False)
        self.convection_scheme = ct.c_uint(options["convection_scheme"])
        self.iterative_solver = ct.c_bool(options['iterative_solver'])
        self.iterative_tol = ct.c_double(options['iterative_tol'])
        self.iterative_precond = ct.c_bool(options['iterative_precond'])
        self.convect_wake = ct.c_bool(convect_wake)
        self.cfl1 = ct.c_bool(options['cfl1'])
        self.vortex_radius = ct.c_double(options['vortex_radius'])
        self.vortex_radius_wake_ind = ct.c_double(options['vortex_radius_wake_ind'])
        self.interp_coords = ct.c_uint(options["interp_coords"])
        self.filter_method = ct.c_uint(options["filter_method"])
        self.interp_method = ct.c_uint(options["interp_method"])
        self.yaw_slerp = ct.c_double(options["yaw_slerp"])
        self.quasi_steady = ct.c_bool(options['quasi_steady'])
 
        self.only_nonlifting = ct.c_bool(options["only_nonlifting"])
        self.only_lifting = ct.c_bool(not options["nonlifting_body_interactions"])
        self.num_spanwise_panels_wo_induced_velocity = n_span_panels_wo_u_ind

        for i in range(len(options["centre_rot_g"])):
            self.centre_rot_g[i] = ct.c_double(options["centre_rot_g"][i])
        for i in range(len(rbm_vel_g)):     
            self.rbm_vel_g[i] = ct.c_double(rbm_vel_g[i])       

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
    vmopts.set_options(options, n_surfaces = ts_info.n_surf)

    flightconditions = FlightConditions()
    flightconditions.rho = options['rho']
    flightconditions.uinf = np.ctypeslib.as_ctypes(np.linalg.norm(ts_info.u_ext[0][:, 0, 0]))
    flightconditions.uinf_direction = np.ctypeslib.as_ctypes(ts_info.u_ext[0][:, 0, 0]/flightconditions.uinf)

    p_rbm_vel_g = options['rbm_vel_g'].ctypes.data_as(ct.POINTER(ct.c_double))
    p_centre_rot_g = options['centre_rot_g'].ctypes.data_as(ct.POINTER(ct.c_double))

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
            ts_info.ct_p_forces,
            p_rbm_vel_g,
            p_centre_rot_g)
    ts_info.remove_ctypes_pointers()

def vlm_solver_nonlifting_body(ts_info, options):
    run_VLM_nonlifting = UvlmLib.run_VLM_nonlifting_body

    vmopts = VMopts()
    vmopts.set_options(options, n_surfaces_nonlifting = ts_info.n_surf)

    flightconditions = FlightConditions()
    flightconditions.rho = options['rho']
    flightconditions.uinf = np.ctypeslib.as_ctypes(np.linalg.norm(ts_info.u_ext[0][:, 0, 0]))
    flightconditions.uinf_direction = np.ctypeslib.as_ctypes(ts_info.u_ext[0][:, 0, 0]/flightconditions.uinf)

    ts_info.generate_ctypes_pointers()
    run_VLM_nonlifting(ct.byref(vmopts),
                       ct.byref(flightconditions),
                       ts_info.ct_p_dimensions,
                       ts_info.ct_p_zeta,
                       ts_info.ct_p_u_ext,
                       ts_info.ct_p_sigma,
                       ts_info.ct_p_forces)
    ts_info.remove_ctypes_pointers()

def vlm_solver_lifting_and_nonlifting_bodies(ts_info_lifting, ts_info_nonlifting, options):
    run_VLM_lifting_and_nonlifting = UvlmLib.run_VLM_lifting_and_nonlifting_bodies
    run_VLM_lifting_and_nonlifting.restype = None

    vmopts = VMopts()    
    vmopts.set_options(options, n_surfaces = ts_info_lifting.n_surf, n_surfaces_nonlifting = ts_info_nonlifting.n_surf)

    flightconditions = FlightConditions()
    flightconditions.rho = options['rho']
    flightconditions.uinf = np.ctypeslib.as_ctypes(np.linalg.norm(ts_info_lifting.u_ext[0][:, 0, 0]))
    flightconditions.uinf_direction = np.ctypeslib.as_ctypes(ts_info_lifting.u_ext[0][:, 0, 0]/flightconditions.uinf)

    p_rbm_vel_g = options['rbm_vel_g'].ctypes.data_as(ct.POINTER(ct.c_double))
    ts_info_lifting.generate_ctypes_pointers()
    ts_info_nonlifting.generate_ctypes_pointers()
    run_VLM_lifting_and_nonlifting(ct.byref(vmopts),
            ct.byref(flightconditions),
            ts_info_lifting.ct_p_dimensions,
            ts_info_lifting.ct_p_dimensions_star,
            ts_info_lifting.ct_p_zeta,
            ts_info_lifting.ct_p_zeta_star,
            ts_info_lifting.ct_p_zeta_dot,
            ts_info_lifting.ct_p_u_ext,
            ts_info_lifting.ct_p_gamma,
            ts_info_lifting.ct_p_gamma_star,
            ts_info_lifting.ct_p_forces,
            ts_info_lifting.ct_p_flag_zeta_phantom,
            ts_info_nonlifting.ct_p_dimensions,
            ts_info_nonlifting.ct_p_zeta,
            ts_info_nonlifting.ct_p_u_ext,
            ts_info_nonlifting.ct_p_sigma,
            ts_info_nonlifting.ct_p_forces)

    ts_info_lifting.remove_ctypes_pointers()
    ts_info_nonlifting.remove_ctypes_pointers()


def uvlm_solver(i_iter, ts_info, struct_ts_info, options, convect_wake=True, dt=None):
    rbm_vel = struct_ts_info.for_vel.copy()
    rbm_vel[0:3] = np.dot(struct_ts_info.cga(), rbm_vel[0:3])
    rbm_vel[3:6] = np.dot(struct_ts_info.cga(), rbm_vel[3:6])
    p_rbm_vel = rbm_vel.ctypes.data_as(ct.POINTER(ct.c_double))
    p_centre_rot = options['centre_rot'].ctypes.data_as(ct.POINTER(ct.c_double))

    
    run_UVLM = UvlmLib.run_UVLM
    run_UVLM.restype = None

    uvmopts = UVMopts()
    uvmopts.set_options(options,
                        n_surfaces = ts_info.n_surf,
                        n_surfaces_nonlifting = 0, 
                        dt = dt, 
                        convect_wake = convect_wake, 
                        rbm_vel_g = rbm_vel,
                        image_method = False,
                        n_span_panels_wo_u_ind=0)


    flightconditions = FlightConditions()
    flightconditions.rho = options['rho']
    flightconditions.uinf = np.ctypeslib.as_ctypes(np.linalg.norm(ts_info.u_ext[0][:, 0, 0]))
    # direction = np.array([1.0, 0, 0])
    flightconditions.uinf_direction = np.ctypeslib.as_ctypes(ts_info.u_ext[0][:, 0, 0]/flightconditions.uinf)
    # flightconditions.uinf_direction = np.ctypeslib.as_ctypes(direction)

    
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
             ts_info.ct_p_gamma,
             ts_info.ct_p_gamma_star,
             ts_info.ct_p_dist_to_orig,
             # previous_ts_info.ct_p_gamma,
             ts_info.ct_p_normals,
             ts_info.ct_p_forces,
             ts_info.ct_p_dynamic_forces)
    ts_info.remove_ctypes_pointers()
    # previous_ts_info.remove_ctypes_pointers()

def uvlm_solver_lifting_and_nonlifting(i_iter, ts_info, ts_info_nonlifting, struct_ts_info, options, convect_wake=True, dt=None):
    rbm_vel = struct_ts_info.for_vel.copy()
    rbm_vel[0:3] = np.dot(struct_ts_info.cga(), rbm_vel[0:3])
    rbm_vel[3:6] = np.dot(struct_ts_info.cga(), rbm_vel[3:6])
    p_rbm_vel = rbm_vel.ctypes.data_as(ct.POINTER(ct.c_double))
    p_centre_rot = options['centre_rot'].ctypes.data_as(ct.POINTER(ct.c_double))

    uvmopts = UVMopts()
    uvmopts.set_options(options,
                        n_surfaces = ts_info.n_surf,
                        n_surfaces_nonlifting = ts_info_nonlifting.n_surf, 
                        dt = dt, 
                        convect_wake = convect_wake, 
                        rbm_vel_g = rbm_vel,
                        image_method = False,
                        n_span_panels_wo_u_ind=4)
    uvmopts.only_lifting = ct.c_bool(False)
    run_UVLM = UvlmLib.run_UVLM_lifting_and_nonlifting
    run_UVLM.restype = None

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
    p_centre_rot = options['centre_rot'].ctypes.data_as(ct.POINTER(ct.c_double))

    i = ct.c_uint(i_iter)
    ts_info.generate_ctypes_pointers()
    ts_info_nonlifting.generate_ctypes_pointers()
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
             ts_info.ct_p_gamma,
             ts_info.ct_p_gamma_star,
             ts_info.ct_p_dist_to_orig,
             ts_info.ct_p_normals,
             ts_info.ct_p_forces,
             ts_info.ct_p_dynamic_forces,
             ts_info.ct_p_flag_zeta_phantom,
             ts_info_nonlifting.ct_p_dimensions,
             ts_info_nonlifting.ct_p_zeta,
             ts_info_nonlifting.ct_p_u_ext,
             ts_info_nonlifting.ct_p_sigma,
             ts_info_nonlifting.ct_p_forces)

    ts_info.remove_ctypes_pointers()
    ts_info_nonlifting.remove_ctypes_pointers()

def uvlm_calculate_unsteady_forces(ts_info,
                                   struct_ts_info,
                                   options,
                                   convect_wake=True,
                                   dt=None):
    calculate_unsteady_forces = UvlmLib.calculate_unsteady_forces
    calculate_unsteady_forces.restype = None

    uvmopts = UVMopts()
    if dt is None:
        uvmopts.dt = ct.c_double(options["dt"])
    else:
        uvmopts.dt = ct.c_double(dt)
    uvmopts.NumCores = ct.c_uint(options["num_cores"])
    uvmopts.NumSurfaces = ct.c_uint(ts_info.n_surf)
    uvmopts.ImageMethod = ct.c_bool(False)
    uvmopts.convection_scheme = ct.c_uint(options["convection_scheme"])
    uvmopts.iterative_solver = ct.c_bool(options['iterative_solver'])
    uvmopts.iterative_tol = ct.c_double(options['iterative_tol'])
    uvmopts.iterative_precond = ct.c_bool(options['iterative_precond'])
    uvmopts.convect_wake = ct.c_bool(convect_wake)
    uvmopts.vortex_radius = ct.c_double(options['vortex_radius'])

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
                                                    vortex_radius,
                                                    for_pos=np.zeros((6)),
                                                    ncores=ct.c_uint(1)):
    """
    uvlm_calculate_total_induced_velocity_at_points

    Caller to the UVLM library to compute the induced velocity of all the
    surfaces and wakes at a list of points

    Args:
        ts_info (AeroTimeStepInfo): Time step information
        target_triads (np.array): Point coordinates, size=(npoints, 3)
        vortex_radius (float): Vortex radius threshold below which do not compute induced velocity
        uind (np.array): Induced velocity

    Returns:
    	uind (np.array): Induced velocity, size=(npoints, 3)

    """
    calculate_uind_at_points = UvlmLib.total_induced_velocity_at_points
    calculate_uind_at_points.restype = None

    uvmopts = UVMopts()
    uvmopts.NumSurfaces = ct.c_uint(ts_info.n_surf)
    uvmopts.ImageMethod = ct.c_bool(False)
    uvmopts.NumCores = ct.c_uint(ncores)
    uvmopts.vortex_radius = ct.c_double(vortex_radius)

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


def biot_panel_cpp(zeta_point, zeta_panel, vortex_radius, gamma=1.0):
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

    if type(vortex_radius) is ct.c_double:
        vortex_radius_float = vortex_radius.value
    else:
        vortex_radius_float = vortex_radius
    velP = np.zeros((3,), order='C')
    UvlmLib.call_biot_panel(
        velP.ctypes.data_as(ct.POINTER(ct.c_double)),
        zeta_point.ctypes.data_as(ct.POINTER(ct.c_double)),
        zeta_panel.ctypes.data_as(ct.POINTER(ct.c_double)),
        ct.byref(ct.c_double(gamma)),
        ct.byref(ct.c_double(vortex_radius_float)))

    return velP


def eval_panel_cpp(zeta_point, zeta_panel,
                   vortex_radius, gamma_pan=1.0):
    """
    Linear UVLM function

    Returns
        tuple: The derivative of the induced velocity with respect to point ``P`` and panel vertices ``ZetaP``.

    Warnings:
        Function may fail if zeta_point is not stored contiguously.

        Eg:

        The following will fail

            zeta_point=Mat[:,2,5]
            eval_panel_cpp(zeta_point,zeta_panel, vortex_radius, gamma_pan=1.0)

        but

            zeta_point=Mat[:,2,5].copy()
            eval_panel_cpp(zeta_point,zeta_panel, vortex_radius, gamma_pan=1.0)

        will not.
    """

    assert zeta_point.flags['C_CONTIGUOUS'] and zeta_panel.flags['C_CONTIGUOUS'], \
        'Input not C contiguous'

    der_point = np.zeros((3, 3), order='C')
    der_vertices = np.zeros((4, 3, 3), order='C')

    if type(vortex_radius) is ct.c_double:
        vortex_radius_float = vortex_radius.value
    else:
        vortex_radius_float = vortex_radius
    UvlmLib.call_der_biot_panel(
        der_point.ctypes.data_as(ct.POINTER(ct.c_double)),
        der_vertices.ctypes.data_as(ct.POINTER(ct.c_double)),
        zeta_point.ctypes.data_as(ct.POINTER(ct.c_double)),
        zeta_panel.ctypes.data_as(ct.POINTER(ct.c_double)),
        ct.byref(ct.c_double(gamma_pan)),
        ct.byref(ct.c_double(vortex_radius_float)))

    return der_point, der_vertices


def get_induced_velocity_cpp(maps, zeta, gamma, zeta_target,
                             vortex_radius):
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

    if type(vortex_radius) is ct.c_double:
        vortex_radius_float = vortex_radius.value
    else:
        vortex_radius_float = vortex_radius
    call_ind_vel(
        uind_target.ctypes.data_as(ct.POINTER(ct.c_double)),
        zeta_target.ctypes.data_as(ct.POINTER(ct.c_double)),
        zeta.ctypes.data_as(ct.POINTER(ct.c_double)),
        gamma.ctypes.data_as(ct.POINTER(ct.c_double)),
        ct.byref(ct.c_int(M)),
        ct.byref(ct.c_int(N)),
        ct.byref(ct.c_double(vortex_radius_float)))

    return uind_target


def get_aic3_cpp(maps, zeta, zeta_target, vortex_radius):
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

    if type(vortex_radius) is ct.c_double:
        vortex_radius_float = vortex_radius.value
    else:
        vortex_radius_float = vortex_radius
    UvlmLib.call_aic3(
        aic3.ctypes.data_as(ct.POINTER(ct.c_double)),
        zeta_target.ctypes.data_as(ct.POINTER(ct.c_double)),
        zeta.ctypes.data_as(ct.POINTER(ct.c_double)),
        ct.byref(ct.c_int(maps.M)),
        ct.byref(ct.c_int(maps.N)),
        ct.byref(ct.c_double(vortex_radius_float)))

    return aic3


def dvinddzeta_cpp(zetac, surf_in, is_bound,
                   vortex_radius, M_in_bound=None):
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

    if type(vortex_radius) is ct.c_double:
        vortex_radius_float = vortex_radius.value
    else:
        vortex_radius_float = vortex_radius
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
        ct.byref(ct.c_double(vortex_radius_float))
    )

    return der_coll, der_vert
