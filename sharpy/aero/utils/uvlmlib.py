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


def vlm_solver(ts_info):
    run_VLM = UvlmLib.run_VLM
    run_VLM.restype = None

    vmopts = VMopts()
    vmopts.Steady = ct.c_bool(True)
    vmopts.Mstar = ct.c_uint(1)
    vmopts.NumSurfaces = ct.c_uint(ts_info.n_surf)

    # n_surf = ts_info.n_surf
    # from sharpy.utils.constants import NDIM
    # n_dim = ct.c_int(NDIM)

    ts_info.generate_ctypes_pointers()
    run_VLM(ct.byref(vmopts),
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

