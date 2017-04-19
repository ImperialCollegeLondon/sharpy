from sharpy.utils.sharpydir import SharpyDir
import sharpy.utils.ctypes_utils as ct_utils

import ctypes as ct
import numpy as np
import platform
import os


UvlmLib = ct_utils.import_ctypes_lib(SharpyDir + '/lib/', 'libuvlm')

class VMopts(ct.Structure):
    """ctypes definition for VMopts class
    """
    _fields_ = [("ImageMethod", ct.c_bool),
                ("Mstar", ct.c_uint),
                ("Steady", ct.c_bool),
                ("KJMeth", ct.c_bool),
                ("NewAIC", ct.c_bool),
                ("DelTime", ct.c_double),
                ("Rollup", ct.c_bool),
                ("NumCores", ct.c_uint),
                ("NumSurfaces", ct.c_uint)]

    def __init__(self):
        ct.Structure.__init__(self)
        self.ImageMethod = ct.c_bool(False)
        self.Mstar = ct.c_uint(1)
        self.Steady = ct.c_bool(True)
        self.KJMeth = ct.c_bool(False)  # legacy var
        self.NewAIC = ct.c_bool(False)  # legacy var
        self.DelTime = ct.c_double(1.0)
        self.Rollup = ct.c_bool(False)
        self.NumCores = ct.c_uint(1)
        self.NumSurfaces = ct.c_uint(1)


# type for 2d integer matrix
t_2int = ct.POINTER(ct.c_int)*2


def vlm_solver(grid, settings):
    run_VLM = UvlmLib.run_VLM
    run_VLM.restype = None

    vmopts = VMopts()
    vmopts.Steady = ct.c_bool(True)
    vmopts.Mstar = ct.c_uint(1)
    vmopts.NumSurfaces = ct.c_uint(grid.n_surf)

    n_surf = grid.n_surf
    n_dim = ct.c_int(3)
    dimensions = grid.aero_dimensions.astype(dtype=ct.c_int)
    dimensions_star = grid.aero_dimensions.astype(dtype=ct.c_int)
    dimensions_star[0, :] = vmopts.Mstar

    zeta_list = []
    for i_surf in range(n_surf):
        for i_dim in range(n_dim.value):
            zeta_list.append(grid.zeta[i_surf][i_dim, :, :].reshape(-1))

    normals_list = []
    for i_surf in range(n_surf):
        for i_dim in range(n_dim.value):
            normals_list.append(grid.normals[i_surf][i_dim, :, :].reshape(-1))

    zeta_star_list = []
    for i_surf in range(n_surf):
        for i_dim in range(n_dim.value):
            zeta_star_list.append(grid.zeta_star[i_surf][i_dim, :, :].reshape(-1))

    p_zeta = (ct.POINTER(ct.c_double)*len(zeta_list))(* [np.ctypeslib.as_ctypes(array) for array in zeta_list])
    p_normals = (ct.POINTER(ct.c_double)*len(normals_list))(* [np.ctypeslib.as_ctypes(array) for array in normals_list])
    p_zeta_star = (ct.POINTER(ct.c_double)*len(zeta_star_list))(* [np.ctypeslib.as_ctypes(array) for array in zeta_star_list])
    p_dimensions = (t_2int)(* np.ctypeslib.as_ctypes(dimensions))
    p_dimensions_star = (t_2int)(* np.ctypeslib.as_ctypes(dimensions_star))

    run_VLM(ct.byref(vmopts), ct.byref(n_dim), p_dimensions, p_dimensions_star, p_zeta, p_zeta_star, p_normals)

    del p_zeta, zeta_list
    del p_normals, normals_list
    del p_dimensions

