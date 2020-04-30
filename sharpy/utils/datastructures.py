"""Data Management Structures

Classes for the Aerotimestep and Structuraltimestep, amongst others
"""
import copy
import ctypes as ct
import numpy as np

import sharpy.utils.algebra as algebra
import sharpy.utils.multibody as mb


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

        # Distance from the trailing edge of the wake vertices
        self.dist_to_orig = []
        for i_surf in range(self.n_surf):
            self.dist_to_orig.append(np.zeros((dimensions_star[i_surf, 0] + 1,
                                               dimensions_star[i_surf, 1] + 1),
                                               dtype=ct.c_double))

        self.wake_conv_vel = []
        for i_surf in range(self.n_surf):
            self.wake_conv_vel.append(np.zeros((dimensions_star[i_surf, 0],
                                            dimensions_star[i_surf, 1]),
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

        # Multibody variables
        self.in_global_AFoR = True

        self.control_surface_deflection = np.array([])

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

        for i_surf in range(copied.n_surf):
            copied.dist_to_orig[i_surf] = self.dist_to_orig[i_surf].astype(dtype=ct.c_double, copy=True, order='C')

        for i_surf in range(copied.n_surf):
            copied.wake_conv_vel[i_surf] = self.wake_conv_vel[i_surf].astype(dtype=ct.c_double, copy=True, order='C')

        # total forces
        copied.inertial_total_forces = self.inertial_total_forces.astype(dtype=ct.c_double, copy=True, order='C')
        copied.body_total_forces = self.body_total_forces.astype(dtype=ct.c_double, copy=True, order='C')
        copied.inertial_steady_forces = self.inertial_steady_forces.astype(dtype=ct.c_double, copy=True, order='C')
        copied.body_steady_forces = self.body_steady_forces.astype(dtype=ct.c_double, copy=True, order='C')
        copied.inertial_unsteady_forces = self.inertial_unsteady_forces.astype(dtype=ct.c_double, copy=True, order='C')
        copied.body_unsteady_forces = self.body_unsteady_forces.astype(dtype=ct.c_double, copy=True, order='C')

        copied.postproc_cell = copy.deepcopy(self.postproc_cell)
        copied.postproc_node = copy.deepcopy(self.postproc_node)

        copied.control_surface_deflection = self.control_surface_deflection.astype(dtype=ct.c_double, copy=True)

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

        self.ct_dist_to_orig_list = []
        for i_surf in range(self.n_surf):
            self.ct_dist_to_orig_list.append(self.dist_to_orig[i_surf][:, :].reshape(-1))

        self.ct_wake_conv_vel_list = []
        for i_surf in range(self.n_surf):
            self.ct_wake_conv_vel_list.append(self.wake_conv_vel[i_surf][:, :].reshape(-1))

        try:
            self.postproc_cell['incidence_angle']
        except KeyError:
            with_incidence_angle = False
        else:
            with_incidence_angle = True

        if with_incidence_angle:
            self.ct_incidence_list = []
            for i_surf in range(self.n_surf):
                self.ct_incidence_list.append(self.postproc_cell['incidence_angle'][i_surf][:, :].reshape(-1))

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
        self.ct_p_dist_to_orig = ((ct.POINTER(ct.c_double)*len(self.ct_dist_to_orig_list))
                           (* [np.ctypeslib.as_ctypes(array) for array in self.ct_dist_to_orig_list]))
        self.ct_p_wake_conv_vel = ((ct.POINTER(ct.c_double)*len(self.ct_wake_conv_vel_list))
                           (* [np.ctypeslib.as_ctypes(array) for array in self.ct_wake_conv_vel_list]))

        if with_incidence_angle:
            self.postproc_cell['incidence_angle_ct_pointer'] = ((ct.POINTER(ct.c_double)*len(self.ct_incidence_list))
                            (* [np.ctypeslib.as_ctypes(array) for array in self.ct_incidence_list]))

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
    def __init__(self, num_node, num_elem, num_node_elem=3, num_dof=None, num_bodies=1):
        self.num_node = num_node
        self.num_elem = num_elem
        self.num_node_elem = num_node_elem
        # generate placeholder for node coordinates
        self.pos = np.zeros((self.num_node, 3), dtype=ct.c_double, order='F')
        self.pos_dot = np.zeros((self.num_node, 3), dtype=ct.c_double, order='F')
        self.pos_ddot = np.zeros((self.num_node, 3), dtype=ct.c_double, order='F')

        # placeholder for CRV
        self.psi = np.zeros((self.num_elem, num_node_elem, 3), dtype=ct.c_double, order='F')
        self.psi_dot = np.zeros((self.num_elem, num_node_elem, 3), dtype=ct.c_double, order='F')
        self.psi_ddot = np.zeros((self.num_elem, num_node_elem, 3), dtype=ct.c_double, order='F')

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
        self.total_forces = np.zeros((6,), dtype=ct.c_double, order='F')

        if num_dof is None:
            # For backwards compatibility
            num_dof = (self.num_node.value - 1)*6
        self.q = np.zeros((num_dof.value + 6 + 4,), dtype=ct.c_double, order='F')
        self.dqdt = np.zeros((num_dof.value + 6 + 4,), dtype=ct.c_double, order='F')
        self.dqddt = np.zeros((num_dof.value + 6 + 4,), dtype=ct.c_double, order='F')

        self.postproc_cell = dict()
        self.postproc_node = dict()

        # Multibody
        self.mb_FoR_pos = np.zeros((num_bodies,6), dtype=ct.c_double, order='F')
        self.mb_FoR_vel = np.zeros((num_bodies,6), dtype=ct.c_double, order='F')
        self.mb_FoR_acc = np.zeros((num_bodies,6), dtype=ct.c_double, order='F')
        self.mb_quat = np.zeros((num_bodies,4), dtype=ct.c_double, order='F')
        self.mb_quat[:,0] = np.ones((num_bodies), dtype=ct.c_double, order='F')
        self.mb_dqddt_quat = np.zeros((num_bodies,4), dtype=ct.c_double, order='F')
        self.forces_constraints_nodes = np.zeros((self.num_node, 6), dtype=ct.c_double, order='F')
        self.forces_constraints_FoR = np.zeros((num_bodies, 10), dtype=ct.c_double, order='F')
        self.mb_dict = None

    def copy(self):
        copied = StructTimeStepInfo(self.num_node, self.num_elem, self.num_node_elem, ct.c_int(len(self.q)-10), self.mb_quat.shape[0])

        copied.num_node = self.num_node
        copied.num_elem = self.num_elem
        copied.num_node_elem = self.num_node_elem

        # generate placeholder for node coordinates
        copied.pos = self.pos.astype(dtype=ct.c_double, order='F', copy=True)
        copied.pos_dot = self.pos_dot.astype(dtype=ct.c_double, order='F', copy=True)
        copied.pos_ddot = self.pos_ddot.astype(dtype=ct.c_double, order='F', copy=True)
        # self.pos_dot = np.zeros((self.num_node, 3), dtype=ct.c_double, order='F')

        # placeholder for CRV
        copied.psi = self.psi.astype(dtype=ct.c_double, order='F', copy=True)
        copied.psi_dot = self.psi_dot.astype(dtype=ct.c_double, order='F', copy=True)
        copied.psi_ddot = self.psi_ddot.astype(dtype=ct.c_double, order='F', copy=True)

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
        copied.total_forces = self.total_forces.astype(dtype=ct.c_double, order='F', copy=True)

        copied.q = self.q.astype(dtype=ct.c_double, order='F', copy=True)
        copied.dqdt = self.dqdt.astype(dtype=ct.c_double, order='F', copy=True)
        copied.dqddt = self.dqddt.astype(dtype=ct.c_double, order='F', copy=True)

        copied.postproc_cell = copy.deepcopy(self.postproc_cell)
        copied.postproc_node = copy.deepcopy(self.postproc_node)

        #if not self.mb_quat is None:
        copied.mb_FoR_pos = self.mb_FoR_pos.astype(dtype=ct.c_double, order='F', copy=True)
        copied.mb_FoR_vel = self.mb_FoR_vel.astype(dtype=ct.c_double, order='F', copy=True)
        copied.mb_FoR_acc = self.mb_FoR_acc.astype(dtype=ct.c_double, order='F', copy=True)
        copied.mb_quat = self.mb_quat.astype(dtype=ct.c_double, order='F', copy=True)
        copied.mb_dqddt_quat = self.mb_dqddt_quat.astype(dtype=ct.c_double, order='F', copy=True)
        copied.forces_constraints_nodes = self.forces_constraints_nodes.astype(dtype=ct.c_double, order='F', copy=True)
        copied.forces_constraints_FoR = self.forces_constraints_FoR.astype(dtype=ct.c_double, order='F', copy=True)

        copied.mb_dict = copy.deepcopy(self.mb_dict)

        return copied

    def glob_pos(self, include_rbm=True):
        coords = self.pos.copy()
        c = self.cga()
        for i_node in range(self.num_node):
            coords[i_node, :] = np.dot(c, coords[i_node, :])
            if include_rbm:
                coords[i_node, :] += self.for_pos[0:3]
        return coords

    def cga(self):
        return algebra.quat2rotation(self.quat)

    def cag(self):
        return self.cga().T

    def euler_angles(self):
        """
        Returns the 3 Euler angles (roll, pitch, yaw) for a given time step.

        :returns: `np.array` (roll, pitch, yaw) in radians.
        """

        return algebra.quat2euler(self.quat)


    def get_body(self, beam, num_dof_ibody, ibody):
        """
        get_body

        Extract the body number 'ibody' from a multibody system

        Given 'self' as a StructTimeStepInfo class of a multibody system, this
        function returns another StructTimeStepInfo class (ibody_StructTimeStepInfo)
        that only includes the body number 'ibody' of the original system

        Args:
            self(StructTimeStepInfo): timestep information of the multibody system
            beam(Beam): beam information of the multibody system
            ibody(int): body number to be extracted

        Returns:
        	ibody_StructTimeStepInfo(StructTimeStepInfo): timestep information of the isolated body

        Examples:

        Notes:

        """

        # Define the nodes and elements belonging to the body
        ibody_elems, ibody_nodes = mb.get_elems_nodes_list(beam, ibody)

        ibody_num_node = len(ibody_nodes)
        ibody_num_elem = len(ibody_elems)

        ibody_first_dof = 0
        for index_body in range(ibody - 1):
            aux_elems, aux_nodes = mb.get_elems_nodes_list(beam, index_body)
            ibody_first_dof += np.sum(beam.vdof[aux_nodes] > -1)*6

        # Initialize the new StructTimeStepInfo
        ibody_StructTimeStepInfo = StructTimeStepInfo(ibody_num_node, ibody_num_elem, self.num_node_elem, num_dof = num_dof_ibody, num_bodies = beam.num_bodies)

        # Assign all the variables
        # ibody_StructTimeStepInfo.quat = self.quat.astype(dtype=ct.c_double, order='F', copy=True)
        # ibody_StructTimeStepInfo.for_pos = self.for_pos.astype(dtype=ct.c_double, order='F', copy=True)
        # ibody_StructTimeStepInfo.for_vel = self.for_vel.astype(dtype=ct.c_double, order='F', copy=True)
        # ibody_StructTimeStepInfo.for_acc = self.for_acc.astype(dtype=ct.c_double, order='F', copy=True)

        CAslaveG = algebra.quat2rotation(self.mb_quat[ibody, :]).T
        ibody_StructTimeStepInfo.quat = self.mb_quat[ibody, :].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.for_pos = self.mb_FoR_pos[ibody, :].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.for_vel[0:3] = np.dot(CAslaveG, self.mb_FoR_vel[ibody, 0:3])
        ibody_StructTimeStepInfo.for_vel[3:6] = np.dot(CAslaveG, self.mb_FoR_vel[ibody, 3:6])
        ibody_StructTimeStepInfo.for_acc[0:3] = np.dot(CAslaveG, self.mb_FoR_acc[ibody, 0:3])
        ibody_StructTimeStepInfo.for_acc[3:6] = np.dot(CAslaveG, self.mb_FoR_acc[ibody, 3:6])

        ibody_StructTimeStepInfo.pos = self.pos[ibody_nodes,:].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.pos_dot = self.pos_dot[ibody_nodes,:].astype(dtype=ct.c_double, order='F', copy=True)

        ibody_StructTimeStepInfo.psi = self.psi[ibody_elems,:,:].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.psi_dot = self.psi_dot[ibody_elems,:,:].astype(dtype=ct.c_double, order='F', copy=True)

        ibody_StructTimeStepInfo.gravity_vector_inertial = self.gravity_vector_inertial.astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.gravity_vector_body = self.gravity_vector_body.astype(dtype=ct.c_double, order='F', copy=True)

        ibody_StructTimeStepInfo.steady_applied_forces = self.steady_applied_forces[ibody_nodes,:].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.unsteady_applied_forces = self.unsteady_applied_forces[ibody_nodes,:].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.gravity_forces = self.gravity_forces[ibody_nodes,:].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.total_gravity_forces = self.total_gravity_forces.astype(dtype=ct.c_double, order='F', copy=True)

        ibody_StructTimeStepInfo.q[0:num_dof_ibody.value] = self.q[ibody_first_dof:ibody_first_dof+num_dof_ibody.value].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.dqdt[0:num_dof_ibody.value] = self.dqdt[ibody_first_dof:ibody_first_dof+num_dof_ibody.value].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.dqddt[0:num_dof_ibody.value] = self.dqddt[ibody_first_dof:ibody_first_dof+num_dof_ibody.value].astype(dtype=ct.c_double, order='F', copy=True)

        # ibody_StructTimeStepInfo.q[-10:] = self.q[-10:].astype(dtype=ct.c_double, order='F', copy=True)
        # ibody_StructTimeStepInfo.dqdt[-10:] = self.dqdt[-10:].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.dqdt[-4:] = ibody_StructTimeStepInfo.quat.astype(dtype=ct.c_double, order='F', copy=True)
        # ibody_StructTimeStepInfo.dqddt[-10:] = self.dqddt[-10:].astype(dtype=ct.c_double, order='F', copy=True)

        ibody_StructTimeStepInfo.mb_quat = self.mb_quat.astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.mb_FoR_pos = self.mb_FoR_pos.astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.mb_FoR_vel = self.mb_FoR_vel.astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.mb_FoR_acc = self.mb_FoR_acc.astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.mb_dqddt_quat = self.mb_dqddt_quat.astype(dtype=ct.c_double, order='F', copy=True)

        return ibody_StructTimeStepInfo

    def change_to_local_AFoR(self, global_ibody):
        """
        change_to_local_AFoR

        Reference a StructTimeStepInfo to the local A frame of reference

        Given 'self' as a StructTimeStepInfo class, this function references
        it to the local A frame of reference

        Args:
            self(StructTimeStepInfo): timestep information
            global_ibody(int): body number (as defined in the mutibody system) to be modified

        Returns:

        Examples:

        Notes:

        """

        # Define the rotation matrices between the different FoR
        CAslaveG = algebra.quat2rotation(self.mb_quat[global_ibody,:]).T
        CGAmaster = algebra.quat2rotation(self.mb_quat[0,:])
        Csm = np.dot(CAslaveG, CGAmaster)

        delta_pos_ms = self.mb_FoR_pos[global_ibody,:] - self.mb_FoR_pos[0,:]
        delta_vel_ms = self.mb_FoR_vel[global_ibody,:] - self.mb_FoR_vel[0,:]

        # Modify position
        for inode in range(self.pos.shape[0]):
            pos_previous = self.pos[inode,:] + np.zeros((3,),)
            self.pos[inode,:] = np.dot(Csm,self.pos[inode,:]) - np.dot(CAslaveG,delta_pos_ms[0:3])
            # self.pos_dot[inode,:] = np.dot(Csm,self.pos_dot[inode,:]) - np.dot(CAslaveG,delta_vel_ms[0:3])
            self.pos_dot[inode,:] = (np.dot(Csm, self.pos_dot[inode,:]) -
                                    np.dot(CAslaveG, delta_vel_ms[0:3]) -
                                    np.dot(algebra.skew(np.dot( CAslaveG, self.mb_FoR_vel[global_ibody,3:6])), self.pos[inode,:]) +
                                    np.dot(Csm, np.dot(algebra.skew(np.dot(CGAmaster.T, self.mb_FoR_vel[0,3:6])), pos_previous)))

            self.gravity_forces[inode,0:3] = np.dot(Csm, self.gravity_forces[inode,0:3])
            self.gravity_forces[inode,3:6] = np.dot(Csm, self.gravity_forces[inode,3:6])

            # self.pos_dot[inode,:] = (np.dot(np.transpose(Csm),self.pos_dot[inode,:]) +
            #                         np.dot(np.transpose(CGAmaster),delta_vel_ms[0:3]) +
            #                         np.dot(Csm.T, np.cross( np.dot(CAslaveG, self.mb_FoR_vel[global_ibody,3:6]), pos_previous)) -
            #                         np.cross(np.dot(CGAmaster.T, self.mb_FoR_vel[0,3:6]), self.pos[inode,:]))


        # Modify local rotations
        for ielem in range(self.psi.shape[0]):
            for inode in range(3):
                psi_previous = self.psi[ielem,inode,:] + np.zeros((3,),)
                self.psi[ielem,inode,:] = algebra.rotation2crv(np.dot(Csm,algebra.crv2rotation(self.psi[ielem,inode,:])))
                self.psi_dot[ielem, inode, :] = np.dot(np.dot(algebra.crv2tan(self.psi[ielem,inode,:]),Csm),
                                                    (np.dot(algebra.crv2tan(psi_previous).T,self.psi_dot[ielem,inode,:]) - np.dot(CGAmaster.T,delta_vel_ms[3:6])))

        # Set the output FoR variables
        self.for_pos = self.mb_FoR_pos[global_ibody,:].astype(dtype=ct.c_double, order='F', copy=True)
        self.for_vel[0:3] = np.dot(CAslaveG,self.mb_FoR_vel[global_ibody,0:3])
        self.for_vel[3:6] = np.dot(CAslaveG,self.mb_FoR_vel[global_ibody,3:6])
        self.for_acc[0:3] = np.dot(CAslaveG,self.mb_FoR_acc[global_ibody,0:3])
        self.for_acc[3:6] = np.dot(CAslaveG,self.mb_FoR_acc[global_ibody,3:6])
        self.quat = self.mb_quat[global_ibody,:].astype(dtype=ct.c_double, order='F', copy=True)
        self.dqdt[-4:] = self.quat.astype(dtype=ct.c_double, order='F', copy=True)

    def change_to_global_AFoR(self, global_ibody):
        """
        change_to_global_AFoR

        Reference a StructTimeStepInfo to the global A frame of reference

        Given 'self' as a StructTimeStepInfo class, this function references
        it to the global A frame of reference

        Args:
            self(StructTimeStepInfo): timestep information
            global_ibody(int): body number (as defined in the mutibody system) to be modified

        Returns:

        Examples:

        Notes:

        """

        # Define the rotation matrices between the different FoR
        CAslaveG = algebra.quat2rotation(self.mb_quat[global_ibody,:]).T
        CGAmaster = algebra.quat2rotation(self.mb_quat[0,:])
        Csm = np.dot(CAslaveG, CGAmaster)

        delta_pos_ms = self.mb_FoR_pos[global_ibody,:] - self.mb_FoR_pos[0,:]
        delta_vel_ms = self.mb_FoR_vel[global_ibody,:] - self.mb_FoR_vel[0,:]

        for inode in range(self.pos.shape[0]):
            pos_previous = self.pos[inode,:] + np.zeros((3,),)
            self.pos[inode,:] = (np.dot(np.transpose(Csm),self.pos[inode,:]) +
                                np.dot(np.transpose(CGAmaster),delta_pos_ms[0:3]))
            self.pos_dot[inode,:] = (np.dot(np.transpose(Csm),self.pos_dot[inode,:]) +
                                    np.dot(np.transpose(CGAmaster),delta_vel_ms[0:3]) +
                                    np.dot(Csm.T, np.dot(algebra.skew(np.dot(CAslaveG, self.mb_FoR_vel[global_ibody,3:6])), pos_previous)) -
                                    np.dot(algebra.skew(np.dot(CGAmaster.T, self.mb_FoR_vel[0,3:6])), self.pos[inode,:]))
            self.gravity_forces[inode,0:3] = np.dot(Csm.T, self.gravity_forces[inode,0:3])
            self.gravity_forces[inode,3:6] = np.dot(Csm.T, self.gravity_forces[inode,3:6])
                                    # np.cross(np.dot(CGAmaster.T, delta_vel_ms[3:6]), pos_previous))

        for ielem in range(self.psi.shape[0]):
            for inode in range(3):
                psi_previous = self.psi[ielem,inode,:] + np.zeros((3,),)
                self.psi[ielem,inode,:] = algebra.rotation2crv(np.dot(Csm.T, algebra.crv2rotation(self.psi[ielem,inode,:])))
                self.psi_dot[ielem, inode, :] = np.dot(algebra.crv2tan(self.psi[ielem,inode,:]),
                                                (np.dot(Csm.T, np.dot(algebra.crv2tan(psi_previous).T, self.psi_dot[ielem, inode, :])) +
                                                np.dot(algebra.quat2rotation(self.mb_quat[0,:]).T, delta_vel_ms[3:6])))

        # Set the output FoR variables
        self.for_pos = self.mb_FoR_pos[0,:].astype(dtype=ct.c_double, order='F', copy=True)
        self.for_vel[0:3] = np.dot(np.transpose(CGAmaster),self.mb_FoR_vel[0,0:3])
        self.for_vel[3:6] = np.dot(np.transpose(CGAmaster),self.mb_FoR_vel[0,3:6])
        self.for_acc[0:3] = np.dot(np.transpose(CGAmaster),self.mb_FoR_acc[0,0:3])
        self.for_acc[3:6] = np.dot(np.transpose(CGAmaster),self.mb_FoR_acc[0,3:6])
        self.quat = self.mb_quat[0,:].astype(dtype=ct.c_double, order='F', copy=True)


class LinearTimeStepInfo(object):
    """
    Linear timestep info containing the state, input and output variables for a given timestep

    """
    def __init__(self):
        self.x = None
        self.y = None
        self.u = None
        self.t = None

    def copy(self):
        copied = LinearTimeStepInfo()
        copied.x = self.x.copy()
        copied.y = self.y.copy()
        copied.u = self.u.copy()
        copied.t = self.t.copy()
