import ctypes as ct
import numpy as np
import copy

import sharpy.utils.algebra as algebra
import copy


class AeroTimeStepInfo(object):
    def __init__(self, dimensions, dimensions_star):
        self.ct_dimensions = None
        self.ct_dimensions_star = None

        self.dimensions = dimensions.copy()
        self.dimensions_star = dimensions_star.copy()
        self.n_surf = self.dimensions.shape[0]
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

        self.gamma_dot = []
        for i_surf in range(self.n_surf):
            self.gamma_dot.append(np.zeros((dimensions[i_surf, 0],
                                            dimensions[i_surf, 1]),
                                           dtype=ct.c_double))

        # total forces
        self.inertial_total_forces = np.zeros((self.n_surf, 6))
        self.body_total_forces = np.zeros((self.n_surf, 6))
        self.inertial_steady_forces = np.zeros((self.n_surf, 6))
        self.body_steady_forces = np.zeros((self.n_surf, 6))
        self.inertial_unsteady_forces = np.zeros((self.n_surf, 6))
        self.body_unsteady_forces = np.zeros((self.n_surf, 6))

        self.postproc_cell = dict()
        self.postproc_node = dict()

    def copy(self):
        copied = AeroTimeStepInfo(self.dimensions, self.dimensions_star)
        # generate placeholder for aero grid zeta coordinates
        for i_surf in range(copied.n_surf):
            copied.zeta[i_surf] = self.zeta[i_surf].astype(dtype=ct.c_double, copy=True, order='C')

        for i_surf in range(copied.n_surf):
            copied.zeta_dot[i_surf] = self.zeta_dot[i_surf].astype(dtype=ct.c_double, copy=True, order='C')

        # panel normals
        for i_surf in range(copied.n_surf):
            copied.normals[i_surf] = self.normals[i_surf].astype(dtype=ct.c_double, copy=True, order='C')

        # panel forces
        for i_surf in range(copied.n_surf):
            copied.forces[i_surf] = self.forces[i_surf].astype(dtype=ct.c_double, copy=True, order='C')

        # panel forces
        for i_surf in range(copied.n_surf):
            copied.dynamic_forces[i_surf] = self.dynamic_forces[i_surf].astype(dtype=ct.c_double, copy=True, order='C')

        # generate placeholder for aero grid zeta_star coordinates
        for i_surf in range(copied.n_surf):
            copied.zeta_star[i_surf] = self.zeta_star[i_surf].astype(dtype=ct.c_double, copy=True, order='C')

        # placeholder for external velocity
        for i_surf in range(copied.n_surf):
            copied.u_ext[i_surf] = self.u_ext[i_surf].astype(dtype=ct.c_double, copy=True, order='C')

        for i_surf in range(copied.n_surf):
            copied.u_ext_star[i_surf] = self.u_ext_star[i_surf].astype(dtype=ct.c_double, copy=True, order='C')

        # allocate gamma and gamma star matrices
        for i_surf in range(copied.n_surf):
            copied.gamma[i_surf] = self.gamma[i_surf].astype(dtype=ct.c_double, copy=True, order='C')

        for i_surf in range(copied.n_surf):
            copied.gamma_dot[i_surf] = self.gamma_dot[i_surf].astype(dtype=ct.c_double, copy=True, order='C')

        for i_surf in range(copied.n_surf):
            copied.gamma_star[i_surf] = self.gamma_star[i_surf].astype(dtype=ct.c_double, copy=True, order='C')

        # total forces
        copied.inertial_total_forces = self.inertial_total_forces.astype(dtype=ct.c_double, copy=True, order='C')
        copied.body_total_forces = self.body_total_forces.astype(dtype=ct.c_double, copy=True, order='C')
        copied.inertial_steady_forces = self.inertial_steady_forces.astype(dtype=ct.c_double, copy=True, order='C')
        copied.body_steady_forces = self.body_steady_forces.astype(dtype=ct.c_double, copy=True, order='C')
        copied.inertial_unsteady_forces = self.inertial_unsteady_forces.astype(dtype=ct.c_double, copy=True, order='C')
        copied.body_unsteady_forces = self.body_unsteady_forces.astype(dtype=ct.c_double, copy=True, order='C')

        copied.postproc_cell = copy.deepcopy(self.postproc_cell)
        copied.postproc_node = copy.deepcopy(self.postproc_node)

        return copied

    def generate_ctypes_pointers(self):
        self.ct_dimensions = self.dimensions.astype(dtype=ct.c_uint, copy=True)
        self.ct_dimensions_star = self.dimensions_star.astype(dtype=ct.c_uint, copy=True)

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

        self.ct_gamma_dot_list = []
        for i_surf in range(self.n_surf):
            self.ct_gamma_dot_list.append(self.gamma_dot[i_surf][:, :].reshape(-1))

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
        self.ct_p_u_ext = ((ct.POINTER(ct.c_double)*len(self.ct_u_ext_list))
                           (* [np.ctypeslib.as_ctypes(array) for array in self.ct_u_ext_list]))
        self.ct_p_u_ext_star = ((ct.POINTER(ct.c_double)*len(self.ct_u_ext_star_list))
                           (* [np.ctypeslib.as_ctypes(array) for array in self.ct_u_ext_star_list]))
        self.ct_p_gamma = ((ct.POINTER(ct.c_double)*len(self.ct_gamma_list))
                           (* [np.ctypeslib.as_ctypes(array) for array in self.ct_gamma_list]))
        self.ct_p_gamma_dot = ((ct.POINTER(ct.c_double)*len(self.ct_gamma_dot_list))
                               (* [np.ctypeslib.as_ctypes(array) for array in self.ct_gamma_dot_list]))
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
            del self.ct_p_dimensions
        except AttributeError:
            pass

        try:
            del self.ct_p_dimensions_star
        except AttributeError:
            pass

        try:
            del self.ct_p_zeta
        except AttributeError:
            pass

        try:
            del self.ct_p_zeta_star
        except AttributeError:
            pass

        try:
            del self.ct_p_zeta_dot
        except AttributeError:
            pass

        try:
            del self.ct_p_u_ext
        except AttributeError:
            pass

        try:
            del self.ct_p_u_ext_star
        except AttributeError:
            pass

        try:
            del self.ct_p_gamma
        except AttributeError:
            pass

        try:
            del self.ct_p_gamma_dot
        except AttributeError:
            pass

        try:
            del self.ct_p_gamma_star
        except AttributeError:
            pass

        try:
            del self.ct_p_normals
        except AttributeError:
            pass

        try:
            del self.ct_p_forces
        except AttributeError:
            pass

        try:
            del self.ct_p_dynamic_forces
        except AttributeError:
            pass

        for k in list(self.postproc_cell.keys()):
            if 'ct_list' in k:
                del self.postproc_cell[k]
            elif 'ct_pointer' in k:
                del self.postproc_cell[k]


def init_matrix_structure(dimensions, with_dim_dimension, added_size=0):
    matrix = []
    for i_surf in range(len(dimensions)):
        if with_dim_dimension:
            matrix.append(np.zeros((3,
                                    dimensions[i_surf, 0] + added_size,
                                    dimensions[i_surf, 1] + added_size),
                                   dtype=ct.c_double))
        else:
            matrix.append(np.zeros((dimensions[i_surf, 0] + added_size,
                                    dimensions[i_surf, 1] + added_size),
                                   dtype=ct.c_double))
    return matrix


def standalone_ctypes_pointer(matrix):
    ct_list = []
    n_surf = len(matrix)

    if len(matrix[0].shape) == 2:
        # [i_surf][m, n], like gamma
        for i_surf in range(n_surf):
            ct_list.append(matrix[i_surf][:, :].reshape(-1))

    elif len(matrix[0].shape) == 3:
        # [i_surf][i_dim, m, n], like zeta
        for i_surf in range(n_surf):
            n_dim = matrix[i_surf].shape[0]
            for i_dim in range(n_dim):
                ct_list.append(matrix[i_surf][i_dim, :, :].reshape(-1))

    ct_pointer = ((ct.POINTER(ct.c_double)*len(ct_list))
                  (* [np.ctypeslib.as_ctypes(array) for array in ct_list]))

    return ct_list, ct_pointer


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
        self.quat = np.array([1., 0, 0, 0], dtype=ct.c_double, order='F')
        self.for_pos = np.zeros((6,), dtype=ct.c_double, order='F')
        self.for_vel = np.zeros((6,), dtype=ct.c_double, order='F')
        self.for_acc = np.zeros((6,), dtype=ct.c_double, order='F')

        self.gravity_vector_inertial = np.array([0.0, 0.0, 1.0], dtype=ct.c_double, order='F')
        self.gravity_vector_body = np.array([0.0, 0.0, 1.0], dtype=ct.c_double, order='F')

        self.steady_applied_forces = np.zeros((self.num_node, 6), dtype=ct.c_double, order='F')
        self.unsteady_applied_forces = np.zeros((self.num_node, 6), dtype=ct.c_double, order='F')
        self.gravity_forces = np.zeros((self.num_node, 6), dtype=ct.c_double, order='F')
        self.total_gravity_forces = np.zeros((6,), dtype=ct.c_double, order='F')

        self.q = np.zeros(((self.num_node - 1)*6 + 6 + 4,), dtype=ct.c_double, order='F')
        self.dqdt = np.zeros(((self.num_node - 1)*6 + 6 + 4,), dtype=ct.c_double, order='F')
        self.dqddt = np.zeros(((self.num_node - 1)*6 + 6 + 4,), dtype=ct.c_double, order='F')

        self.postproc_cell = dict()
        self.postproc_node = dict()

    def copy(self):
        copied = StructTimeStepInfo(self.num_node, self.num_elem)

        copied.num_node = self.num_node
        copied.num_elem = self.num_elem
        copied.num_node_elem = self.num_node_elem

        # generate placeholder for node coordinates
        copied.pos = self.pos.astype(dtype=ct.c_double, order='F', copy=True)
        copied.pos_dot = self.pos_dot.astype(dtype=ct.c_double, order='F', copy=True)
        # self.pos_dot = np.zeros((self.num_node, 3), dtype=ct.c_double, order='F')

        # placeholder for CRV
        copied.psi = self.psi.astype(dtype=ct.c_double, order='F', copy=True)
        copied.psi_dot = self.psi_dot.astype(dtype=ct.c_double, order='F', copy=True)

        # FoR data
        copied.quat = self.quat.astype(dtype=ct.c_double, order='F', copy=True)
        copied.for_pos = self.for_pos.astype(dtype=ct.c_double, order='F', copy=True)
        copied.for_vel = self.for_vel.astype(dtype=ct.c_double, order='F', copy=True)
        copied.for_acc = self.for_acc.astype(dtype=ct.c_double, order='F', copy=True)

        copied.gravity_vector_inertial = self.gravity_vector_inertial.astype(dtype=ct.c_double, order='F', copy=True)
        copied.gravity_vector_body = self.gravity_vector_body.astype(dtype=ct.c_double, order='F', copy=True)

        copied.steady_applied_forces = self.steady_applied_forces.astype(dtype=ct.c_double, order='F', copy=True)
        copied.unsteady_applied_forces = self.unsteady_applied_forces.astype(dtype=ct.c_double, order='F', copy=True)
        copied.gravity_forces = self.gravity_forces.astype(dtype=ct.c_double, order='F', copy=True)
        copied.total_gravity_forces = self.total_gravity_forces.astype(dtype=ct.c_double, order='F', copy=True)

        copied.q = self.q.astype(dtype=ct.c_double, order='F', copy=True)
        copied.dqdt = self.dqdt.astype(dtype=ct.c_double, order='F', copy=True)
        copied.dqddt = self.dqddt.astype(dtype=ct.c_double, order='F', copy=True)

        copied.postproc_cell = copy.deepcopy(self.postproc_cell)
        copied.postproc_node = copy.deepcopy(self.postproc_node)

        return copied

    def glob_pos(self, include_rbm=True):
        coords = self.pos.copy()
        c = self.cga()
        for i_node in range(self.num_node):
            coords[i_node, :] = np.dot(c, coords[i_node, :])
            if include_rbm:
                coords[i_node, :] += self.for_pos[0:3]
        return coords

    def cag(self):
        return algebra.quat2rot(self.quat)

    def cga(self):
        return self.cag().T






