from sharpy.utils.sharpydir import SharpyDir
import sharpy.utils.ctypes_utils as ct_utils

import ctypes as ct
import numpy as np
import platform
import os


UvlmLib = ct_utils.import_ctypes_lib(SharpyDir + '/lib/', 'libuvlm')

t_2double = ct.POINTER(ct.c_double)*3
t_2int = ct.POINTER(ct.c_int)*2


def VLM_solver():
    run_VLM = UvlmLib.run_VLM
    run_VLM.restype = None
    n_surf = ct.c_int(1)
    dimensions = np.zeros((1, 2), dtype=ct.c_int)
    dimensions[0, 0] = 3
    dimensions[0, 1] = 2

    zeta1 = np.arange(6).reshape((3, 2)).astype(dtype=ct.c_double) + 0.1
    zeta2 = np.arange(6).reshape((3, 2)).astype(dtype=ct.c_double) + 10.1
    zeta3 = np.arange(6).reshape((3, 2)).astype(dtype=ct.c_double) + 100.1

    zeta = [zeta1.reshape(-1), zeta2.reshape(-1), zeta3.reshape(-1)]

    p_zeta = (ct.POINTER(ct.c_double)*len(zeta))(* [np.ctypeslib.as_ctypes(array) for array in zeta])
    p_dimensions = (t_2int)(* np.ctypeslib.as_ctypes(dimensions))

    run_VLM(ct.byref(n_surf), p_dimensions, p_zeta)

