import ctypes as ct
import numpy as np
import copy

import sharpy.utils.algebra as algebra


class AeroTimeStepInfo(object):
    def __init__(self, dimensions, dimensions_star):
        self.ct_dimensions = None
        self.ct_dimensions_star = None

        self.dimensions = dimensions
        self.dimensions_star = dimensions_star
        self.n_surf = dimensions.shape[0]
        # generate placeholder for aero grid zeta coordinates
        self.zeta = []
        for i_surf in range(self.n_surf):
            self.zeta.append(np.zeros((3,
                                       dimensions[i_surf, 0] + 1,
                                       dimensions[i_surf, 1] + 1),
                                      dtype=ct.c_double))
        self.zeta_dot = []
        for i_surf in range(self.n_surf):
            self.zeta_dot.append(np.zeros((3,
                                           dimensions[i_surf, 0] + 1,
                                           dimensions[i_surf, 1] + 1),
                                          dtype=ct.c_double))

        # panel normals
        self.normals = []
        for i_surf in range(self.n_surf):
            self.normals.append(np.zeros((3,
                                          dimensions[i_surf, 0],
                                          dimensions[i_surf, 1]),
                                         dtype=ct.c_double))

        # panel forces
        self.forces = []
        for i_surf in range(self.n_surf):
            self.forces.append(np.zeros((6,
                                         dimensions[i_surf, 0] + 1,
                                         dimensions[i_surf, 1] + 1),
                                        dtype=ct.c_double))
        # panel forces
        self.dynamic_forces = []
        for i_surf in range(self.n_surf):
            self.dynamic_forces.append(np.zeros((6,
                                                 dimensions[i_surf, 0] + 1,
                                                 dimensions[i_surf, 1] + 1),
                                                dtype=ct.c_double))

        # generate placeholder for aero grid zeta_star coordinates
        self.zeta_star = []
        for i_surf in range(self.n_surf):
            self.zeta_star.append(np.zeros((3,
                                            dimensions_star[i_surf, 0] + 1,
                                            dimensions_star[i_surf, 1] + 1),
                                           dtype=ct.c_double))

        self.zeta_star_dot = []
        for i_surf in range(self.n_surf):
            self.zeta_star_dot.append(np.zeros((3,
                                                dimensions_star[i_surf, 0] + 1,
                                                dimensions_star[i_surf, 1] + 1),
                                               dtype=ct.c_double))
        # placeholder for external velocity
        self.u_ext = []
        for i_surf in range(self.n_surf):
            self.u_ext.append(np.zeros((3,
                                        dimensions[i_surf, 0] + 1,
                                        dimensions[i_surf, 1] + 1),
                                       dtype=ct.c_double))

        self.u_ext_star = []
        for i_surf in range(self.n_surf):
            self.u_ext_star.append(np.zeros((3,
                                             dimensions_star[i_surf, 0] + 1,
                                             dimensions_star[i_surf, 1] + 1),
                                            dtype=ct.c_double))

        # allocate gamma and gamma star matrices
        self.gamma = []
        for i_surf in range(self.n_surf):
            self.gamma.append(np.zeros((dimensions[i_surf, 0],
                                        dimensions[i_surf, 1]),
                                       dtype=ct.c_double))

        self.gamma_star = []
        for i_surf in range(self.n_surf):
            self.gamma_star.append(np.zeros((dimensions_star[i_surf, 0],
                                             dimensions_star[i_surf, 1]),
                                            dtype=ct.c_double))

    def generate_ctypes_pointers(self):
        self.ct_dimensions = self.dimensions.astype(dtype=ct.c_uint)
        self.ct_dimensions_star = self.dimensions_star.astype(dtype=ct.c_uint)

        n_surf = len(self.dimensions)

        from sharpy.utils.constants import NDIM

        self.ct_zeta_list = []
        for i_surf in range(self.n_surf):
            for i_dim in range(NDIM):
                self.ct_zeta_list.append(self.zeta[i_surf][i_dim, :, :].reshape(-1))

        self.ct_zeta_dot_list = []
        for i_surf in range(self.n_surf):
            for i_dim in range(NDIM):
                self.ct_zeta_dot_list.append(self.zeta_dot[i_surf][i_dim, :, :].reshape(-1))

        self.ct_zeta_star_list = []
        for i_surf in range(self.n_surf):
            for i_dim in range(NDIM):
                self.ct_zeta_star_list.append(self.zeta_star[i_surf][i_dim, :, :].reshape(-1))

        self.ct_zeta_star_dot_list = []
        for i_surf in range(self.n_surf):
            for i_dim in range(NDIM):
                self.ct_zeta_star_dot_list.append(self.zeta_star_dot[i_surf][i_dim, :, :].reshape(-1))

        self.ct_u_ext_list = []
        for i_surf in range(self.n_surf):
            for i_dim in range(NDIM):
                self.ct_u_ext_list.append(self.u_ext[i_surf][i_dim, :, :].reshape(-1))

        self.ct_u_ext_star_list = []
        for i_surf in range(self.n_surf):
            for i_dim in range(NDIM):
                self.ct_u_ext_star_list.append(self.u_ext_star[i_surf][i_dim, :, :].reshape(-1))

        self.ct_gamma_list = []
        for i_surf in range(self.n_surf):
            self.ct_gamma_list.append(self.gamma[i_surf][:, :].reshape(-1))

        self.ct_gamma_star_list = []
        for i_surf in range(self.n_surf):
            self.ct_gamma_star_list.append(self.gamma_star[i_surf][:, :].reshape(-1))

        self.ct_normals_list = []
        for i_surf in range(self.n_surf):
            for i_dim in range(NDIM):
                self.ct_normals_list.append(self.normals[i_surf][i_dim, :, :].reshape(-1))

        self.ct_forces_list = []
        for i_surf in range(self.n_surf):
            for i_dim in range(NDIM*2):
                self.ct_forces_list.append(self.forces[i_surf][i_dim, :, :].reshape(-1))

        self.ct_dynamic_forces_list = []
        for i_surf in range(self.n_surf):
            for i_dim in range(NDIM*2):
                self.ct_dynamic_forces_list.append(self.dynamic_forces[i_surf][i_dim, :, :].reshape(-1))

        self.ct_p_dimensions = ((ct.POINTER(ct.c_uint)*n_surf)
                                (* np.ctypeslib.as_ctypes(self.ct_dimensions)))
        self.ct_p_dimensions_star = ((ct.POINTER(ct.c_uint)*n_surf)
                                     (* np.ctypeslib.as_ctypes(self.ct_dimensions_star)))
        self.ct_p_zeta = ((ct.POINTER(ct.c_double)*len(self.ct_zeta_list))
                          (* [np.ctypeslib.as_ctypes(array) for array in self.ct_zeta_list]))
        self.ct_p_zeta_dot = ((ct.POINTER(ct.c_double)*len(self.ct_zeta_dot_list))
                          (* [np.ctypeslib.as_ctypes(array) for array in self.ct_zeta_dot_list]))
        self.ct_p_zeta_star = ((ct.POINTER(ct.c_double)*len(self.ct_zeta_star_list))
                               (* [np.ctypeslib.as_ctypes(array) for array in self.ct_zeta_star_list]))
        self.ct_p_zeta_star_dot = ((ct.POINTER(ct.c_double)*len(self.ct_zeta_star_dot_list))
                               (* [np.ctypeslib.as_ctypes(array) for array in self.ct_zeta_star_dot_list]))
        self.ct_p_u_ext = ((ct.POINTER(ct.c_double)*len(self.ct_u_ext_list))
                           (* [np.ctypeslib.as_ctypes(array) for array in self.ct_u_ext_list]))
        self.ct_p_u_ext_star = ((ct.POINTER(ct.c_double)*len(self.ct_u_ext_star_list))
                           (* [np.ctypeslib.as_ctypes(array) for array in self.ct_u_ext_star_list]))
        self.ct_p_gamma = ((ct.POINTER(ct.c_double)*len(self.ct_gamma_list))
                           (* [np.ctypeslib.as_ctypes(array) for array in self.ct_gamma_list]))
        self.ct_p_gamma_star = ((ct.POINTER(ct.c_double)*len(self.ct_gamma_star_list))
                                (* [np.ctypeslib.as_ctypes(array) for array in self.ct_gamma_star_list]))
        self.ct_p_normals = ((ct.POINTER(ct.c_double)*len(self.ct_normals_list))
                             (* [np.ctypeslib.as_ctypes(array) for array in self.ct_normals_list]))
        self.ct_p_forces = ((ct.POINTER(ct.c_double)*len(self.ct_forces_list))
                            (* [np.ctypeslib.as_ctypes(array) for array in self.ct_forces_list]))
        self.ct_p_dynamic_forces = ((ct.POINTER(ct.c_double)*len(self.ct_dynamic_forces_list))
                            (* [np.ctypeslib.as_ctypes(array) for array in self.ct_dynamic_forces_list]))

    def remove_ctypes_pointers(self):
        try:
            del self.ct_p_zeta, self.ct_zeta_list
        except AttributeError:
            pass
        try:
            del self.ct_p_zeta_star, self.ct_zeta_star_list
        except AttributeError:
            pass
        try:
            del self.ct_p_u_ext, self.ct_u_ext_list
        except AttributeError:
            pass
        try:
            del self.ct_p_u_ext_star, self.ct_u_ext_star_list
        except AttributeError:
            pass
        try:
            del self.ct_p_gamma, self.ct_gamma_list
        except AttributeError:
            pass
        try:
            del self.ct_p_gamma_star, self.ct_gamma_star_list
        except AttributeError:
            pass
        try:
            del self.ct_p_normals, self.ct_normals_list
        except AttributeError:
            pass
        try:
            del self.ct_p_forces, self.ct_forces_list
        except AttributeError:
            pass
        try:
            del self.ct_p_dynamic_forces, self.ct_dynamic_forces_list
        except AttributeError:
            pass
        try:
            del self.ct_p_dimensions
        except AttributeError:
            pass

    def copy(self):
        out = AeroTimeStepInfo(self.dimensions, self.dimensions_star)
        out.zeta = copy.deepcopy(self.zeta)
        out.zeta_star = copy.deepcopy(self.zeta_star)
        out.u_ext = copy.deepcopy(self.u_ext)
        out.u_ext_star = copy.deepcopy(self.u_ext_star)
        out.gamma = copy.deepcopy(self.gamma)
        out.gamma_star = copy.deepcopy(self.gamma_star)
        out.normals = copy.deepcopy(self.normals)
        out.forces = copy.deepcopy(self.forces)
        return out

    def update_orientation(self, rot):
        for i_surf in range(self.n_surf):
            for i_row in range(self.zeta[i_surf].shape[1]):
                for i_col in range(self.zeta[i_surf].shape[2]):
                    self.zeta[i_surf][:, i_row, i_col] = np.dot(rot,
                                                                self.zeta[i_surf][:, i_row, i_col])


class StructTimeStepInfo(object):
    def __init__(self, num_node, num_elem, num_node_elem=3):
        self.num_node = num_node
        self.num_elem = num_elem
        self.num_node_elem = num_node_elem
        # generate placeholder for node coordinates
        self.pos = np.zeros((self.num_node, 3), dtype=ct.c_double, order='F')
        self.pos_dot = np.zeros((self.num_node, 3), dtype=ct.c_double, order='F')

        # placeholder for CRV
        self.psi = np.zeros((self.num_elem, num_node_elem, 3), dtype=ct.c_double, order='F')
        self.psi_dot = np.zeros((self.num_elem, num_node_elem, 3), dtype=ct.c_double, order='F')

        # FoR data
        self.quat = np.array([1, 0, 0, 0], dtype=float)
        self.for_pos = np.zeros((6,))
        self.for_vel = np.zeros((6,))

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def glob_pos(self, include_rbm=True):
        coords = self.pos.copy()
        c = algebra.quat2rot(self.quat).transpose()
        for i_node in range(self.num_node):
            coords[i_node, :] = np.dot(c, coords[i_node, :])
            if include_rbm:
                coords[i_node, :] += self.for_pos[0:3]
        return coords

    def update_orientation(self, quat):
        self.quat = quat.copy()



