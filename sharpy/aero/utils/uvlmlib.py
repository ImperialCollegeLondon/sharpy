from sharpy.utils.sharpydir import SharpyDir
import sharpy.utils.ctypes_utils as ct_utils

import ctypes as ct
import numpy as np
import platform
import os


UvlmLib = ct_utils.import_ctypes_lib(SharpyDir + '/lib/', 'libuvlm')

t_2double = ct.POINTER(ct.c_double)*3
t_2int = ct.POINTER(ct.c_int)*2


def VLM_solver(grid, settings):
    run_VLM = UvlmLib.run_VLM
    run_VLM.restype = None

    n_surf = ct.c_int(grid.n_surf)
    n_dim = ct.c_int(3)
    dimensions = grid.aero_dimensions.astype(dtype=ct.c_int)


    print(grid.zeta[0][0,:,:].shape)
    print('---')
    print(grid.zeta[0][1,:,:])





    zeta_list = []
    for i_surf in range(n_surf.value):
        for i_dim in range(n_dim.value):
            zeta_list.append(grid.zeta[i_surf][i_dim, :, :].reshape(-1))

    normals_list = []
    for i_surf in range(n_surf.value):
        for i_dim in range(n_dim.value):
            normals_list.append(grid.normals[i_surf][i_dim, :, :].reshape(-1))


    p_zeta = (ct.POINTER(ct.c_double)*len(zeta_list))(* [np.ctypeslib.as_ctypes(array) for array in zeta_list])
    p_normals = (ct.POINTER(ct.c_double)*len(normals_list))(* [np.ctypeslib.as_ctypes(array) for array in normals_list])
    p_dimensions = (t_2int)(* np.ctypeslib.as_ctypes(dimensions))

    run_VLM(ct.byref(n_surf), ct.byref(n_dim), p_dimensions, p_zeta, p_normals)

    del p_zeta, zeta_list
    del p_normals, normals_list
    del p_dimensions
    a = 1

    # n_surf = ct.c_int(1)
    # dimensions = np.zeros((1, 2), dtype=ct.c_int)
    # dimensions[0, 0] = 3
    # dimensions[0, 1] = 2
    #
    # zeta1 = np.arange(6).reshape((3, 2)).astype(dtype=ct.c_double) + 0.1
    # zeta2 = np.arange(6).reshape((3, 2)).astype(dtype=ct.c_double) + 10.1
    # zeta3 = np.arange(6).reshape((3, 2)).astype(dtype=ct.c_double) + 100.1
    #
    # zeta = [zeta1.reshape(-1), zeta2.reshape(-1), zeta3.reshape(-1)]
    #
    # p_zeta = (ct.POINTER(ct.c_double)*len(zeta))(* [np.ctypeslib.as_ctypes(array) for array in zeta])
    # p_dimensions = (t_2int)(* np.ctypeslib.as_ctypes(dimensions))
    #
    # run_VLM(ct.byref(n_surf), p_dimensions, p_zeta)

