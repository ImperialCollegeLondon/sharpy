"""Data Management Structures

These classes are responsible for storing the aerodynamic and structural time step information and relevant variables.

"""
import copy
import ctypes as ct
import numpy as np

import sharpy.utils.algebra as algebra
import sharpy.utils.multibody as mb


class AeroTimeStepInfo(object):
    """
    Aerodynamic Time step class.

    Contains the relevant aerodynamic attributes for a single time step. All variables should be expressed in ``G``
    FoR unless otherwise stated.

    Attributes:
        ct_dimensions: Pointer to ``dimensions`` to interface the C++ library `uvlmlib``
        ct_dimensions_star: Pointer to ``dimensions_star`` to interface the C++ library `uvlmlib``

        dimensions (np.ndarray): Matrix defining the dimensions of the vortex grid on solid surfaces
          ``[num_surf x chordwise panels x spanwise panels]``
        dimensions_star (np.ndarray): Matrix defining the dimensions of the vortex grid on wakes
          ``[num_surf x streamwise panels x spanwise panels]``

        n_surf (int): Number of aerodynamic surfaces on solid bodies. Each aerodynamic surface on solid bodies will
          have an associted wake.

        zeta (list(np.ndarray): Location of solid grid vertices
          ``[n_surf][3 x (chordwise nodes + 1) x (spanwise nodes + 1)]``
        zeta_dot (list(np.ndarray)): Time derivative of ``zeta``
        normals (list(np.ndarray)): Normal direction to panels at the panel center
          ``[n_surf][3 x chordwise nodes x spanwise nodes]``
        forces (list(np.ndarray)): Forces not associated to time derivatives on grid vertices
          ``[n_surf][3 x (chordwise nodes + 1) x (spanwise nodes + 1)]``
        dynamic_forces (list(np.ndarray)): Forces associated to time derivatives on grid vertices
          ``[n_surf][3 x (chordwise nodes + 1) x (spanwise nodes + 1)]``
        zeta_star (list(np.ndarray): Location of wake grid vertices
          ``[n_surf][3 x (streamwise nodes + 1) x (spanwise nodes + 1)]``
        u_ext (list(np.ndarray)): Background flow velocity on solid grid nodes
          ``[n_surf][3 x (chordwise nodes + 1) x (spanwise nodes + 1)]``
        u_ext_star (list(np.ndarray)): Background flow velocity on wake grid nodes
          ``[n_surf][3 x (streamwise nodes + 1) x (spanwise nodes + 1)]``
        gamma (list(np.ndarray)): Circulation associated to solid panels
          ``[n_surf][3 x chordwise nodes x spanwise nodes]``
        gamma_star (list(np.ndarray)): Circulation associated to wake panels
          ``[n_surf][3 x streamwise nodes x spanwise nodes]``
        gamma_dot (list(np.ndarray)): Time derivative of ``gamma``

        inertial_total_forces (list(np.ndarray)): Total aerodynamic forces in ``G`` FoR ``[n_surf x 6]``, written by ``AeroForcesCalculator``.
        body_total_forces (list(np.ndarray)): Total aerodynamic forces in ``A`` FoR ``[n_surf x 6]``, written by ``AeroForcesCalculator``.
        inertial_steady_forces (list(np.ndarray)): Total aerodynamic steady forces in ``G`` FoR ``[n_surf x 6]``, written by ``AeroForcesCalculator``.
        body_steady_forces (list(np.ndarray)): Total aerodynamic steady forces in ``A`` FoR ``[n_surf x 6]``, written by ``AeroForcesCalculator``.
        inertial_unsteady_forces (list(np.ndarray)): Total aerodynamic unsteady forces in ``G`` FoR ``[n_surf x 6]``, written by ``AeroForcesCalculator``.
        body_unsteady_forces (list(np.ndarray)): Total aerodynamic unsteady forces in ``A`` FoR ``[n_surf x 6]``, written by ``AeroForcesCalculator``.

        postproc_cell (dict): Variables associated to cells to be postprocessed
        postproc_node (dict): Variables associated to nodes to be postprocessed

        in_global_AFoR (bool): ``True`` if the variables are stored in the global A FoR. ``False`` if they are stored
          in the local A FoR of each body. Always ``True`` for single-body simulations. Currently not used.

        control_surface_deflection (np.ndarray): Deflection of the control surfaces, in `rad` and if fitted.

    Args:
        dimensions (np.ndarray): Matrix defining the dimensions of the vortex grid on solid surfaces
          ``[num_surf x chordwise panels x spanwise panels]``
        dimensions_star (np.ndarray): Matrix defining the dimensions of the vortex grid on wakes
          ``[num_surf x streamwise panels x spanwise panels]``
    """
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

        # total forces - written by AeroForcesCalculator
        self.inertial_total_forces = np.zeros((self.n_surf, 6))
        self.body_total_forces = np.zeros((self.n_surf, 6))
        self.inertial_steady_forces = np.zeros((self.n_surf, 6))
        self.body_steady_forces = np.zeros((self.n_surf, 6))
        self.inertial_unsteady_forces = np.zeros((self.n_surf, 6))
        self.body_unsteady_forces = np.zeros((self.n_surf, 6))
        self.total_inertial_forces = np.zeros((6,))  # G Frame
        self.total_body_forces = np.zeros((6,))  # A Frame

        self.postproc_cell = dict()
        self.postproc_node = dict()

        # Multibody variables
        self.in_global_AFoR = True

        self.control_surface_deflection = np.array([])

    def copy(self):
        """
        Returns a copy of a deepcopy of a :class:`~sharpy.utils.datastructures.AeroTimeStepInfo`
        """
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

        # total forces
        copied.inertial_total_forces = self.inertial_total_forces.astype(dtype=ct.c_double, copy=True, order='C')
        copied.body_total_forces = self.body_total_forces.astype(dtype=ct.c_double, copy=True, order='C')
        copied.inertial_steady_forces = self.inertial_steady_forces.astype(dtype=ct.c_double, copy=True, order='C')
        copied.body_steady_forces = self.body_steady_forces.astype(dtype=ct.c_double, copy=True, order='C')
        copied.inertial_unsteady_forces = self.inertial_unsteady_forces.astype(dtype=ct.c_double, copy=True, order='C')
        copied.body_unsteady_forces = self.body_unsteady_forces.astype(dtype=ct.c_double, copy=True, order='C')
        copied.total_inertial_forces = self.total_inertial_forces.astype(dtype=ct.c_double, copy=True, order='C')
        copied.total_body_forces = self.total_body_forces.astype(dtype=ct.c_double, copy=True, order='C')

        copied.postproc_cell = copy.deepcopy(self.postproc_cell)
        copied.postproc_node = copy.deepcopy(self.postproc_node)

        copied.control_surface_deflection = self.control_surface_deflection.astype(dtype=ct.c_double, copy=True)

        return copied

    def generate_ctypes_pointers(self):
        """
        Generates the pointers to aerodynamic variables used to interface the C++ library ``uvlmlib``
        """
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

        if with_incidence_angle:
            self.postproc_cell['incidence_angle_ct_pointer'] = ((ct.POINTER(ct.c_double)*len(self.ct_incidence_list))
                            (* [np.ctypeslib.as_ctypes(array) for array in self.ct_incidence_list]))

    def remove_ctypes_pointers(self):
        """
        Removes the pointers to aerodynamic variables used to interface the C++ library ``uvlmlib``
        """
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

        try:
            del self.ct_p_dist_to_orig
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
    """
    Structural Time Step Class.

    Contains the relevant attributes for the structural description of a single time step.

    Attributes:
        in_global_AFoR (bool): ``True`` if the variables are stored in the global A FoR. ``False'' if they are stored
          in the local A FoR of each body. Always ``True`` for single-body simulations

        num_node (int): Number of nodes
        num_elem (int): Number of elements
        num_node_elem (int): Number of nodes per element

        pos (np.ndarray): Displacements. ``[num_node x 3]`` containing the vector of ``x``, ``y`` and ``z``
          coordinates (in ``A`` frame) of the beam nodes.
        pos_dot (np.ndarray): Velocities. Time derivative of ``pos``.
        pos_ddot (np.ndarray): Accelerations. Time derivative of ``pos_dot``

        psi (np.ndarray): Cartesian Rotation Vector. ``[num_elem x num_node_elem x 3]`` CRV for each node in each
          element.
        psi_dot (np.ndarray): Time derivative of ``psi``.
        psi_ddot (np.ndarray): Time derivative of ``psi_dot``.

        quat (np.ndarray): Quaternion expressing the transformation between the ``A`` and ``G`` frames.
        for_pos (np.ndarray): ``A`` frame of reference position (with respect to the `G`` frame of reference).
        for_vel (np.ndarray): ``A`` frame of reference velocity. Expressed in A FoR
        for_acc (np.ndarray): ``A`` frame of reference acceleration. Expressed in A FoR

        steady_applied_forces (np.ndarray): Forces applied to the structure not associated to time derivatives
          ``[num_nodes x 6]``. Expressed in B FoR
        unsteady_applied_forces (np.ndarray): Forces applied to the structure associated to time derivatives
          ``[num_node x 6]``. Expressed in B FoR
        runtime_generated_forces (np.ndarray): Forces generated at runtime through runtime generators
          ``[num_node x 6]``. Expressed in B FoR
        gravity_forces (np.ndarray): Gravity forces at nodes ``[num_node x 6]``. Expressed in A FoR

        total_gravity_forces (np.ndarray): Total gravity forces on the structure ``[6]``. Expressed in A FoR
        total_forces (np.ndarray): Total forces applied to the structure ``[6]``. Expressed in A FoR

        q (np.ndarray): State vector associated to the structural system of equations ``[num_dof + 10]``
        dqdt (np.ndarray): Time derivative of ``q``
        dqddt (np.ndarray): Time derivative of ``dqdt``

        postproc_cell (dict): Variables associated to cells to be postprocessed
        postproc_node (dict): Variables associated to nodes to be postprocessed

        mb_FoR_pos (np.ndarray): Position of the local A FoR of each body ``[num_bodies x 6]``
        mb_FoR_vel (np.ndarray): Velocity of the local A FoR of each body ``[num_bodies x 6]``
        mb_FoR_acc (np.ndarray): Acceleration of the local A FoR of each body ``[num_bodies x 6]``
        mb_quat (np.ndarray): Quaternion of the local A FoR of each body ``[num_bodies x 4]``
        mb_dquatdt (np.ndarray): Time derivative of ``mb_quat``

        forces_constraints_nodes (np.ndarray): Forces associated to Lagrange Constraints on nodes ``[num_node x 6]``
        forces_constraints_FoR (np.ndarray): Forces associated to Lagrange Contraints on frames of reference
          ``[num_bodies x 10]``

        mb_dict (np.ndarray): Dictionary with the multibody information. It comes from the file ``case.mb.h5``
    """
    def __init__(self, num_node, num_elem, num_node_elem=3, num_dof=None, num_bodies=1):
        self.in_global_AFoR = True
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

        self.steady_applied_forces = np.zeros((self.num_node, 6), dtype=ct.c_double, order='F')
        self.unsteady_applied_forces = np.zeros((self.num_node, 6), dtype=ct.c_double, order='F')
        self.runtime_generated_forces = np.zeros((self.num_node, 6), dtype=ct.c_double, order='F')
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
        self.mb_dquatdt = np.zeros((num_bodies, 4), dtype=ct.c_double, order='F')
        self.forces_constraints_nodes = np.zeros((self.num_node, 6), dtype=ct.c_double, order='F')
        self.forces_constraints_FoR = np.zeros((num_bodies, 10), dtype=ct.c_double, order='F')
        self.mb_dict = None

    def copy(self):
        """
        Returns a copy of a deepcopy of a :class:`~sharpy.utils.datastructures.StructTimeStepInfo`
        """
        copied = StructTimeStepInfo(self.num_node, self.num_elem, self.num_node_elem, ct.c_int(len(self.q)-10),
                                    self.mb_quat.shape[0])

        copied.in_global_AFoR = self.in_global_AFoR
        copied.num_node = self.num_node
        copied.num_elem = self.num_elem
        copied.num_node_elem = self.num_node_elem

        # generate placeholder for node coordinates
        copied.pos = self.pos.astype(dtype=ct.c_double, order='F', copy=True)
        copied.pos_dot = self.pos_dot.astype(dtype=ct.c_double, order='F', copy=True)
        copied.pos_ddot = self.pos_ddot.astype(dtype=ct.c_double, order='F', copy=True)

        # placeholder for CRV
        copied.psi = self.psi.astype(dtype=ct.c_double, order='F', copy=True)
        copied.psi_dot = self.psi_dot.astype(dtype=ct.c_double, order='F', copy=True)
        copied.psi_ddot = self.psi_ddot.astype(dtype=ct.c_double, order='F', copy=True)

        # FoR data
        copied.quat = self.quat.astype(dtype=ct.c_double, order='F', copy=True)
        copied.for_pos = self.for_pos.astype(dtype=ct.c_double, order='F', copy=True)
        copied.for_vel = self.for_vel.astype(dtype=ct.c_double, order='F', copy=True)
        copied.for_acc = self.for_acc.astype(dtype=ct.c_double, order='F', copy=True)

        copied.steady_applied_forces = self.steady_applied_forces.astype(dtype=ct.c_double, order='F', copy=True)
        copied.unsteady_applied_forces = self.unsteady_applied_forces.astype(dtype=ct.c_double, order='F', copy=True)
        copied.runtime_generated_forces = self.runtime_generated_forces.astype(dtype=ct.c_double, order='F', copy=True)
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
        copied.mb_dquatdt = self.mb_dquatdt.astype(dtype=ct.c_double, order='F', copy=True)
        copied.forces_constraints_nodes = self.forces_constraints_nodes.astype(dtype=ct.c_double, order='F', copy=True)
        copied.forces_constraints_FoR = self.forces_constraints_FoR.astype(dtype=ct.c_double, order='F', copy=True)

        copied.mb_dict = copy.deepcopy(self.mb_dict)

        return copied

    def glob_pos(self, include_rbm=True):
        """
        Returns the position of the nodes in ``G`` FoR
        """
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

        Extract the body number ``ibody`` from a multibody system

        This function returns a :class:`~sharpy.utils.datastructures.StructTimeStepInfo` class (``ibody_StructTimeStepInfo``)
        that only includes the body number ``ibody`` of the original multibody system ``self``

        Args:
            beam(:class:`~sharpy.structure.models.beam.Beam`): beam information of the multibody system
            num_dof_ibody (int): Number of degrees of freedom associated to the ``ibody``
            ibody(int): body number to be extracted

        Returns:
        	StructTimeStepInfo: timestep information of the isolated body
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
        ibody_StructTimeStepInfo.quat = self.mb_quat[ibody, :].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.for_pos = self.mb_FoR_pos[ibody, :].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.for_vel = self.mb_FoR_vel[ibody, :]
        ibody_StructTimeStepInfo.for_acc = self.mb_FoR_acc[ibody, :]

        ibody_StructTimeStepInfo.pos = self.pos[ibody_nodes,:].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.pos_dot = self.pos_dot[ibody_nodes,:].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.pos_ddot = self.pos_ddot[ibody_nodes,:].astype(dtype=ct.c_double, order='F', copy=True)

        ibody_StructTimeStepInfo.psi = self.psi[ibody_elems,:,:].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.psi_dot = self.psi_dot[ibody_elems,:,:].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.psi_ddot = self.psi_ddot[ibody_elems,:,:].astype(dtype=ct.c_double, order='F', copy=True)

        ibody_StructTimeStepInfo.steady_applied_forces = self.steady_applied_forces[ibody_nodes,:].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.unsteady_applied_forces = self.unsteady_applied_forces[ibody_nodes,:].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.runtime_generated_forces = self.runtime_generated_forces[ibody_nodes,:].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.gravity_forces = self.gravity_forces[ibody_nodes,:].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.total_gravity_forces = self.total_gravity_forces.astype(dtype=ct.c_double, order='F', copy=True)

        ibody_StructTimeStepInfo.q[0:num_dof_ibody.value] = self.q[ibody_first_dof:ibody_first_dof+num_dof_ibody.value].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.dqdt[0:num_dof_ibody.value] = self.dqdt[ibody_first_dof:ibody_first_dof+num_dof_ibody.value].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.dqddt[0:num_dof_ibody.value] = self.dqddt[ibody_first_dof:ibody_first_dof+num_dof_ibody.value].astype(dtype=ct.c_double, order='F', copy=True)

        ibody_StructTimeStepInfo.dqdt[-10:-4] = ibody_StructTimeStepInfo.for_vel.astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.dqddt[-10:-4] = ibody_StructTimeStepInfo.for_acc.astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.dqdt[-4:] = self.quat.astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.dqddt[-4:] = self.mb_dquatdt[ibody, :].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.mb_dquatdt[ibody, :] = self.mb_dquatdt[ibody, :].astype(dtype=ct.c_double, order='F', copy=True)

        ibody_StructTimeStepInfo.mb_quat = None
        ibody_StructTimeStepInfo.mb_FoR_pos = None
        ibody_StructTimeStepInfo.mb_FoR_vel = None
        ibody_StructTimeStepInfo.mb_FoR_acc = None

        return ibody_StructTimeStepInfo

    def change_to_local_AFoR(self, for0_pos, for0_vel, quat0):
        """
        change_to_local_AFoR

        Reference a :class:`~sharpy.utils.datastructures.StructTimeStepInfo` to the local A frame of reference

        Args:
            for0_pos (np.ndarray): Position of the global A FoR
            for0_vel (np.ndarray): Velocity of the global A FoR
            quat0 (np.ndarray): Quaternion of the global A FoR
        """

        # Define the rotation matrices between the different FoR
        CAslaveG = algebra.quat2rotation(self.quat).T
        CGAmaster = algebra.quat2rotation(quat0)
        Csm = np.dot(CAslaveG, CGAmaster)

        delta_vel_ms = np.zeros((6,))
        delta_pos_ms = self.for_pos[0:3] - for0_pos[0:3]
        delta_vel_ms[0:3] = np.dot(CAslaveG.T, self.for_vel[0:3]) - np.dot(CGAmaster, for0_vel[0:3])
        delta_vel_ms[3:6] = np.dot(CAslaveG.T, self.for_vel[3:6]) - np.dot(CGAmaster, for0_vel[3:6])

        # Modify position
        for inode in range(self.pos.shape[0]):
            pos_previous = self.pos[inode,:] + np.zeros((3,),)
            self.pos[inode,:] = np.dot(Csm,self.pos[inode,:]) - np.dot(CAslaveG,delta_pos_ms[0:3])
            # self.pos_dot[inode,:] = np.dot(Csm,self.pos_dot[inode,:]) - np.dot(CAslaveG,delta_vel_ms[0:3])
            self.pos_dot[inode,:] = (np.dot(Csm, self.pos_dot[inode,:]) -
                                    np.dot(CAslaveG, delta_vel_ms[0:3]) -
                                    np.dot(algebra.skew(np.dot( CAslaveG, self.for_vel[3:6])), self.pos[inode,:]) +
                                    np.dot(Csm, np.dot(algebra.skew(np.dot(CGAmaster.T, for0_vel[3:6])), pos_previous)))

            self.gravity_forces[inode,0:3] = np.dot(Csm, self.gravity_forces[inode,0:3])
            self.gravity_forces[inode,3:6] = np.dot(Csm, self.gravity_forces[inode,3:6])

        # Modify local rotations
        for ielem in range(self.psi.shape[0]):
            for inode in range(3):
                psi_previous = self.psi[ielem,inode,:] + np.zeros((3,),)
                self.psi[ielem,inode,:] = algebra.rotation2crv(np.dot(Csm,algebra.crv2rotation(self.psi[ielem,inode,:])))
                self.psi_dot[ielem, inode, :] = np.dot(np.dot(algebra.crv2tan(self.psi[ielem,inode,:]),Csm),
                                                    (np.dot(algebra.crv2tan(psi_previous).T,self.psi_dot[ielem,inode,:]) - np.dot(CGAmaster.T,delta_vel_ms[3:6])))


    def change_to_global_AFoR(self, for0_pos, for0_vel, quat0):
        """
        Reference a :class:`~sharpy.utils.datastructures.StructTimeStepInfo` to the global A frame of reference

        Args:
            for0_pos (np.ndarray): Position of the global A FoR
            for0_vel (np.ndarray): Velocity of the global A FoR
            quat0 (np.ndarray): Quaternion of the global A FoR
        """

        # Define the rotation matrices between the different FoR
        CAslaveG = algebra.quat2rotation(self.quat).T
        CGAmaster = algebra.quat2rotation(quat0)
        Csm = np.dot(CAslaveG, CGAmaster)

        delta_vel_ms = np.zeros((6,))
        delta_pos_ms = self.for_pos[0:3] - for0_pos[0:3]
        delta_vel_ms[0:3] = np.dot(CAslaveG.T, self.for_vel[0:3]) - np.dot(CGAmaster, for0_vel[0:3])
        delta_vel_ms[3:6] = np.dot(CAslaveG.T, self.for_vel[3:6]) - np.dot(CGAmaster, for0_vel[3:6])

        for inode in range(self.pos.shape[0]):
            pos_previous = self.pos[inode,:] + np.zeros((3,),)
            self.pos[inode,:] = (np.dot(np.transpose(Csm),self.pos[inode,:]) +
                                np.dot(np.transpose(CGAmaster),delta_pos_ms[0:3]))
            self.pos_dot[inode,:] = (np.dot(np.transpose(Csm),self.pos_dot[inode,:]) +
                                    np.dot(np.transpose(CGAmaster),delta_vel_ms[0:3]) +
                                    np.dot(Csm.T, np.dot(algebra.skew(np.dot(CAslaveG, self.for_vel[3:6])), pos_previous)) -
                                    np.dot(algebra.skew(np.dot(CGAmaster.T, for0_vel[3:6])), self.pos[inode,:]))
            self.gravity_forces[inode,0:3] = np.dot(Csm.T, self.gravity_forces[inode,0:3])
            self.gravity_forces[inode,3:6] = np.dot(Csm.T, self.gravity_forces[inode,3:6])
                                    # np.cross(np.dot(CGAmaster.T, delta_vel_ms[3:6]), pos_previous))

        for ielem in range(self.psi.shape[0]):
            for inode in range(3):
                psi_previous = self.psi[ielem,inode,:] + np.zeros((3,),)
                self.psi[ielem,inode,:] = algebra.rotation2crv(np.dot(Csm.T, algebra.crv2rotation(self.psi[ielem,inode,:])))
                self.psi_dot[ielem, inode, :] = np.dot(algebra.crv2tan(self.psi[ielem,inode,:]),
                                                (np.dot(Csm.T, np.dot(algebra.crv2tan(psi_previous).T, self.psi_dot[ielem, inode, :])) +
                                                np.dot(algebra.quat2rotation(quat0).T, delta_vel_ms[3:6])))


    def whole_structure_to_local_AFoR(self, beam):
        """
        Same as change_to_local_AFoR but for a multibody structure

        Args:
            beam(sharpy.structure.models.beam.Beam): Beam structure of ``PreSharpy``
        """
        if not self.in_global_AFoR:
            raise NotImplementedError("Wrong managing of FoR")

        self.in_global_AFoR = False
        quat0 = self.quat.astype(dtype=ct.c_double, order='F', copy=True)
        for0_pos = self.for_pos.astype(dtype=ct.c_double, order='F', copy=True)
        for0_vel = self.for_vel.astype(dtype=ct.c_double, order='F', copy=True)

        MB_beam = [None]*beam.num_bodies
        MB_tstep = [None]*beam.num_bodies

        for ibody in range(beam.num_bodies):
            MB_beam[ibody] = beam.get_body(ibody = ibody)
            MB_tstep[ibody] = self.get_body(beam, MB_beam[ibody].num_dof, ibody = ibody)
            MB_tstep[ibody].change_to_local_AFoR(for0_pos, for0_vel, quat0)

        first_dof = 0
        for ibody in range(beam.num_bodies):
            # Renaming for clarity
            ibody_elems = MB_beam[ibody].global_elems_num
            ibody_nodes = MB_beam[ibody].global_nodes_num

            # Merge tstep
            self.pos[ibody_nodes,:] = MB_tstep[ibody].pos.astype(dtype=ct.c_double, order='F', copy=True)
            self.psi[ibody_elems,:,:] = MB_tstep[ibody].psi.astype(dtype=ct.c_double, order='F', copy=True)
            self.gravity_forces[ibody_nodes,:] = MB_tstep[ibody].gravity_forces.astype(dtype=ct.c_double, order='F', copy=True)

            self.pos_dot[ibody_nodes,:] = MB_tstep[ibody].pos_dot.astype(dtype=ct.c_double, order='F', copy=True)
            self.psi_dot[ibody_elems,:,:] = MB_tstep[ibody].psi_dot.astype(dtype=ct.c_double, order='F', copy=True)

            # TODO: Do I need a change in FoR for the following variables? Maybe for the FoR ones.
            # tstep.forces_constraints_nodes[ibody_nodes,:] = MB_tstep[ibody].forces_constraints_nodes.astype(dtype=ct.c_double, order='F', copy=True)
            # tstep.forces_constraints_FoR[ibody, :] = MB_tstep[ibody].forces_constraints_FoR[ibody, :].astype(dtype=ct.c_double, order='F', copy=True)

    def whole_structure_to_global_AFoR(self, beam):
        """
        Same as change_to_global_AFoR but for a multibody structure

        Args:
            beam(sharpy.structure.models.beam.Beam): Beam structure of ``PreSharpy``
        """
        if self.in_global_AFoR:
            raise NotImplementedError("Wrong managing of FoR")

        self.in_global_AFoR = True

        MB_beam = [None]*beam.num_bodies
        MB_tstep = [None]*beam.num_bodies
        quat0 = self.quat.astype(dtype=ct.c_double, order='F', copy=True)
        for0_pos = self.for_pos.astype(dtype=ct.c_double, order='F', copy=True)
        for0_vel = self.for_vel.astype(dtype=ct.c_double, order='F', copy=True)

        for ibody in range(beam.num_bodies):
            MB_beam[ibody] = beam.get_body(ibody = ibody)
            MB_tstep[ibody] = self.get_body(beam, MB_beam[ibody].num_dof, ibody = ibody)
            MB_tstep[ibody].change_to_global_AFoR(for0_pos, for0_vel, quat0)


        first_dof = 0
        for ibody in range(beam.num_bodies):
            # Renaming for clarity
            ibody_elems = MB_beam[ibody].global_elems_num
            ibody_nodes = MB_beam[ibody].global_nodes_num

            # Merge tstep
            self.pos[ibody_nodes,:] = MB_tstep[ibody].pos.astype(dtype=ct.c_double, order='F', copy=True)
            # tstep.pos_dot[ibody_nodes,:] = MB_tstep[ibody].pos_dot.astype(dtype=ct.c_double, order='F', copy=True)
            self.psi[ibody_elems,:,:] = MB_tstep[ibody].psi.astype(dtype=ct.c_double, order='F', copy=True)
            self.gravity_forces[ibody_nodes,:] = MB_tstep[ibody].gravity_forces.astype(dtype=ct.c_double, order='F',
                                                                                       copy=True)
            
            self.pos_dot[ibody_nodes,:] = MB_tstep[ibody].pos_dot.astype(dtype=ct.c_double, order='F', copy=True)
            self.psi_dot[ibody_elems,:,:] = MB_tstep[ibody].psi_dot.astype(dtype=ct.c_double, order='F', copy=True)


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


class Linear(object):
    """
    This is the class responsible for the transfer of information between linear systems
    and can be accessed as ``data.linear``. It stores
    as class attributes the following classes that describe the linearised problem.

    Attributes:
        ss (sharpy.linear.src.libss.StateSpace): State-space system
        linear_system (sharpy.linear.utils.ss_interface.BaseElement): Assemble system properties
        tsaero0 (sharpy.utils.datastructures.AeroTimeStepInfo): Linearisation aerodynamic timestep
        tsstruct0 (sharpy.utils.datastructures.StructTimeStepInfo): Linearisation structural timestep
        timestep_info (list): Linear time steps
    """

    def __init__(self, tsaero0, tsstruct0):
        self.linear_system = None
        self.ss = None
        self.tsaero0 = tsaero0
        self.tsstruct0 = tsstruct0
        self.timestep_info = []
        self.uvlm = None
        self.beam = None
