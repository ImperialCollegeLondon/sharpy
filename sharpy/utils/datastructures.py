"""Data Management Structures

These classes are responsible for storing the aerodynamic and structural time step information and relevant variables.

"""
import copy
import ctypes as ct
import numpy as np

import sharpy.utils.algebra as algebra
import sharpy.utils.multibody as mb


class TimeStepInfo(object):
    """
    Time step class.

    It is the parent class of the AeroTimeStepInfo and NonliftingBodyTimeStepInfo, which contain the relevant
    aerodynamic attributes for a single time step. All variables should be expressed in ``G`` FoR unless
    otherwise stated.

    Attributes:
        ct_dimensions: Pointer to ``dimensions`` to interface the C++ library `uvlmlib``

        dimensions (np.ndarray): Matrix defining the dimensions of the vortex grid on solid surfaces
          ``[num_surf x chordwise panels x spanwise panels]``

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
        u_ext (list(np.ndarray)): Background flow velocity on solid grid nodes
          ``[n_surf][3 x (chordwise nodes + 1) x (spanwise nodes + 1)]``

        inertial_steady_forces (list(np.ndarray)): Total aerodynamic steady forces in ``G`` FoR ``[n_surf x 6]``
        body_steady_forces (list(np.ndarray)): Total aerodynamic steady forces in ``A`` FoR ``[n_surf x 6]``
        inertial_unsteady_forces (list(np.ndarray)): Total aerodynamic unsteady forces in ``G`` FoR ``[n_surf x 6]``
        body_unsteady_forces (list(np.ndarray)): Total aerodynamic unsteady forces in ``A`` FoR ``[n_surf x 6]``

        postproc_cell (dict): Variables associated to cells to be postprocessed
        postproc_node (dict): Variables associated to nodes to be postprocessed

        in_global_AFoR (bool): ``True`` if the variables are stored in the global A FoR. ``False`` if they are stored
          in the local A FoR of each body. Always ``True`` for single-body simulations. Currently not used.


    Args:
        dimensions (np.ndarray): Matrix defining the dimensions of the vortex grid on solid surfaces
          ``[num_surf x chordwise panels x spanwise panels]``
    """
    def __init__(self, dimensions):
        self.ct_dimensions = None

        self.dimensions = dimensions.copy()
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

        # placeholder for external velocity
        self.u_ext = []
        for i_surf in range(self.n_surf):
            self.u_ext.append(np.zeros((3,
                                        dimensions[i_surf, 0] + 1,
                                        dimensions[i_surf, 1] + 1),
                                       dtype=ct.c_double))

        # total forces - written by AeroForcesCalculator
        self.inertial_steady_forces = np.zeros((self.n_surf, 6))
        self.body_steady_forces = np.zeros((self.n_surf, 6))
        self.inertial_unsteady_forces = np.zeros((self.n_surf, 6))
        self.body_unsteady_forces = np.zeros((self.n_surf, 6))

        self.postproc_cell = dict()
        self.postproc_node = dict()

        # Multibody variables
        self.in_global_AFoR = True


    def copy(self):
        """
        Returns a copy of a deepcopy of a :class:`~sharpy.utils.datastructures.TimeStepInfo`
        """
        return self.create_placeholder(TimeStepInfo(self.dimensions))

    def create_placeholder(self, copied):
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

        # placeholder for external velocity
        for i_surf in range(copied.n_surf):
            copied.u_ext[i_surf] = self.u_ext[i_surf].astype(dtype=ct.c_double, copy=True, order='C')

        # total forces
        copied.inertial_steady_forces = self.inertial_steady_forces.astype(dtype=ct.c_double, copy=True, order='C')
        copied.body_steady_forces = self.body_steady_forces.astype(dtype=ct.c_double, copy=True, order='C')
        copied.inertial_unsteady_forces = self.inertial_unsteady_forces.astype(dtype=ct.c_double, copy=True, order='C')
        copied.body_unsteady_forces = self.body_unsteady_forces.astype(dtype=ct.c_double, copy=True, order='C')
        
        copied.postproc_cell = copy.deepcopy(self.postproc_cell)
        copied.postproc_node = copy.deepcopy(self.postproc_node)

        return copied

    def generate_ctypes_pointers(self):
        """
        Generates the pointers to aerodynamic variables used to interface the C++ library ``uvlmlib``
        """
        self.ct_dimensions = self.dimensions.astype(dtype=ct.c_uint, copy=True)

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

        self.ct_u_ext_list = []
        for i_surf in range(self.n_surf):
            for i_dim in range(NDIM):
                self.ct_u_ext_list.append(self.u_ext[i_surf][i_dim, :, :].reshape(-1))

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
        self.ct_p_zeta = ((ct.POINTER(ct.c_double)*len(self.ct_zeta_list))
                          (* [np.ctypeslib.as_ctypes(array) for array in self.ct_zeta_list]))
        self.ct_p_zeta_dot = ((ct.POINTER(ct.c_double)*len(self.ct_zeta_dot_list))
                          (* [np.ctypeslib.as_ctypes(array) for array in self.ct_zeta_dot_list]))
        self.ct_p_u_ext = ((ct.POINTER(ct.c_double)*len(self.ct_u_ext_list))
                           (* [np.ctypeslib.as_ctypes(array) for array in self.ct_u_ext_list]))
        self.ct_p_normals = ((ct.POINTER(ct.c_double)*len(self.ct_normals_list))
                             (* [np.ctypeslib.as_ctypes(array) for array in self.ct_normals_list]))
        self.ct_p_forces = ((ct.POINTER(ct.c_double)*len(self.ct_forces_list))
                            (* [np.ctypeslib.as_ctypes(array) for array in self.ct_forces_list]))
        self.ct_p_dynamic_forces = ((ct.POINTER(ct.c_double)*len(self.ct_dynamic_forces_list))
                            (* [np.ctypeslib.as_ctypes(array) for array in self.ct_dynamic_forces_list]))

        if with_incidence_angle:
            self.postproc_cell['incidence_angle_ct_pointer'] = ((ct.POINTER(ct.c_double)*len(self.ct_incidence_list))
                            (* [np.ctypeslib.as_ctypes(array) for array in self.ct_incidence_list]))

    def remove_ctypes_pointers(self):
        """
            Removes the pointers to aerodynamic variables used to interface the C++ library ``uvlmlib``
        """

        list_class_attributes = list(self.__dict__.keys()).copy()
        for name_attribute in list_class_attributes:
            if "ct_p_" in name_attribute:
                self.__delattr__(name_attribute)

        for k in list(self.postproc_cell.keys()):
            if 'ct_list' in k:
                del self.postproc_cell[k]
            elif 'ct_pointer' in k:
                del self.postproc_cell[k]


class NonliftingBodyTimeStepInfo(TimeStepInfo):
    """
    Nonlifting Body Time step class.

    It is the inheritance from ``TimeStepInfo`` and contains the relevant aerodynamic attributes for
    a single time step of a nonlifting body. All variables should be expressed in ``G``
    FoR unless otherwise stated.

    Attributes:
        ct_dimensions: Pointer to ``dimensions`` to interface the C++ library `uvlmlib``

        dimensions (np.ndarray): Matrix defining the dimensions of the vortex grid on solid surfaces
        ``[num_surf x radial panels x spanwise panels]``

        n_surf (int): Number of aerodynamic surfaces on nonlifting bodies.

        zeta (list(np.ndarray): Location of solid grid vertices
          ``[n_surf][3 x (radial panel) x (spanwise panel)]``
        zeta_dot (list(np.ndarray)): Time derivative of ``zeta``
        normals (list(np.ndarray)): Normal direction to panels at the panel center
          ``[n_surf][3 x radial panels x spanwise panels]``
        forces (list(np.ndarray)): Forces not associated to time derivatives on grid vertices
          ``[n_surf][3 x (radial panels) x (spanwise panels)]``
        dynamic_forces (list(np.ndarray)): Forces associated to time derivatives on grid vertices
          ``[n_surf][3 x (radial panels) x (spanwise panels)]``

        u_ext (list(np.ndarray)): Background flow velocity on solid grid panel
          ``[n_surf][3 x (radial panels) x (spanwise panel + 1)]``
        sigma (list(np.ndarray)): Source strength associated to solid panels
          ``[n_surf][3 x radial panel x spanwise panel]``
        sigma_dot (list(np.ndarray)): Time derivative of ``sigma``
        pressure_coefficients (list(np.ndarray)): Pressure coefficient associated to solid panels
          ``[n_surf][radial panel x spanwise panel]``

        inertial_total_forces (list(np.ndarray)): Total aerodynamic forces in ``G`` FoR ``[n_surf x 6]``
        body_total_forces (list(np.ndarray)): Total aerodynamic forces in ``A`` FoR ``[n_surf x 6]``
        inertial_steady_forces (list(np.ndarray)): Total aerodynamic steady forces in ``G`` FoR ``[n_surf x 6]``
        body_steady_forces (list(np.ndarray)): Total aerodynamic steady forces in ``A`` FoR ``[n_surf x 6]``
        inertial_unsteady_forces (list(np.ndarray)): Total aerodynamic unsteady forces in ``G`` FoR ``[n_surf x 6]``
        body_unsteady_forces (list(np.ndarray)): Total aerodynamic unsteady forces in ``A`` FoR ``[n_surf x 6]``

        postproc_cell (dict): Variables associated to cells to be postprocessed
        postproc_node (dict): Variables associated to panel to be postprocessed

        in_global_AFoR (bool): ``True`` if the variables are stored in the global A FoR. ``False`` if they are stored
          in the local A FoR of each body. Always ``True`` for single-body simulations. Currently not used.

    Args:
        dimensions (np.ndarray): Matrix defining the dimensions of the vortex grid on solid surfaces
          ``[num_surf x radial panels x spanwise panels]``
    """
    def __init__(self, dimensions): #remove dimensions_star as input
        super().__init__(dimensions)

        # allocate sigma matrices
        self.sigma = []
        for i_surf in range(self.n_surf):
            self.sigma.append(np.zeros((dimensions[i_surf, 0],
                                        dimensions[i_surf, 1]),
                                       dtype=ct.c_double))

        self.sigma_dot = []
        for i_surf in range(self.n_surf):
            self.sigma_dot.append(np.zeros((dimensions[i_surf, 0],
                                            dimensions[i_surf, 1]),
                                           dtype=ct.c_double))

        self.pressure_coefficients = []
        for i_surf in range(self.n_surf):
            self.pressure_coefficients.append(np.zeros((dimensions[i_surf, 0],
                                        dimensions[i_surf, 1]),
                                        dtype=ct.c_double))
    def copy(self):
        """
        Returns a copy of a deepcopy of a :class:`~sharpy.utils.datastructures.AeroTimeStepInfo`
        """
        return self.create_placeholder(NonliftingBodyTimeStepInfo(self.dimensions))

    def create_placeholder(self, copied):
        super().create_placeholder(copied)

        # allocate sigma matrices
        for i_surf in range(copied.n_surf):
            copied.sigma[i_surf] = self.sigma[i_surf].astype(dtype=ct.c_double, copy=True, order='C')

        for i_surf in range(copied.n_surf):
            copied.sigma_dot[i_surf] = self.sigma_dot[i_surf].astype(dtype=ct.c_double, copy=True, order='C')

        for i_surf in range(copied.n_surf):
            copied.pressure_coefficients[i_surf] = self.pressure_coefficients[i_surf].astype(dtype=ct.c_double, copy=True, order='C')


        return copied


    def generate_ctypes_pointers(self):
        """
        Generates the pointers to aerodynamic variables used to interface the C++ library ``uvlmlib``
        """
        super().generate_ctypes_pointers()

        from sharpy.utils.constants import NDIM

        self.ct_sigma_list = []
        for i_surf in range(self.n_surf):
            self.ct_sigma_list.append(self.sigma[i_surf][:, :].reshape(-1))

        self.ct_sigma_dot_list = []
        for i_surf in range(self.n_surf):
            self.ct_sigma_dot_list.append(self.sigma_dot[i_surf][:, :].reshape(-1))

        self.ct_pressure_coefficients_list = []
        for i_surf in range(self.n_surf):
            self.ct_pressure_coefficients_list.append(self.pressure_coefficients[i_surf][:, :].reshape(-1))

        self.ct_p_sigma = ((ct.POINTER(ct.c_double)*len(self.ct_sigma_list))
                            (* [np.ctypeslib.as_ctypes(array) for array in self.ct_sigma_list]))
        self.ct_p_sigma_dot = ((ct.POINTER(ct.c_double)*len(self.ct_sigma_dot_list))
                            (* [np.ctypeslib.as_ctypes(array) for array in self.ct_sigma_list]))
        self.ct_p_pressure_coefficients = ((ct.POINTER(ct.c_double)*len(self.ct_pressure_coefficients_list))
                                (* [np.ctypeslib.as_ctypes(array) for array in self.ct_pressure_coefficients_list]))



class AeroTimeStepInfo(TimeStepInfo):
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

        inertial_total_forces (list(np.ndarray)): Total aerodynamic forces in ``G`` FoR ``[n_surf x 6]``
        body_total_forces (list(np.ndarray)): Total aerodynamic forces in ``A`` FoR ``[n_surf x 6]``
        inertial_steady_forces (list(np.ndarray)): Total aerodynamic steady forces in ``G`` FoR ``[n_surf x 6]``
        body_steady_forces (list(np.ndarray)): Total aerodynamic steady forces in ``A`` FoR ``[n_surf x 6]``
        inertial_unsteady_forces (list(np.ndarray)): Total aerodynamic unsteady forces in ``G`` FoR ``[n_surf x 6]``
        body_unsteady_forces (list(np.ndarray)): Total aerodynamic unsteady forces in ``A`` FoR ``[n_surf x 6]``

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
        super().__init__(dimensions)
        self.ct_dimensions_star = None

        self.dimensions_star = dimensions_star.copy()

        # generate placeholder for aero grid zeta_star coordinates
        self.zeta_star = []
        for i_surf in range(self.n_surf):
            self.zeta_star.append(np.zeros((3,
                                            dimensions_star[i_surf, 0] + 1,
                                            dimensions_star[i_surf, 1] + 1),
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

        # Junction handling
        self.flag_zeta_phantom = np.zeros((1, self.n_surf),
                                            dtype=ct.c_int)
        self.control_surface_deflection = np.array([])

    def copy(self):
        return self.create_placeholder(AeroTimeStepInfo(self.dimensions, self.dimensions_star))

    def create_placeholder(self, copied):
        super().create_placeholder(copied)
        # generate placeholder for aero grid zeta_star coordinates
        for i_surf in range(copied.n_surf):
            copied.zeta_star[i_surf] = self.zeta_star[i_surf].astype(dtype=ct.c_double, copy=True, order='C')

        # placeholder for external velocity
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

        copied.control_surface_deflection = self.control_surface_deflection.astype(dtype=ct.c_double, copy=True)

        # phantom panel flags
        copied.flag_zeta_phantom = self.flag_zeta_phantom.astype(dtype=ct.c_int, copy=True, order='C')
        
        return copied

    def generate_ctypes_pointers(self):
        from sharpy.utils.constants import NDIM
        n_surf = len(self.dimensions)
        super().generate_ctypes_pointers()
        self.ct_dimensions_star = self.dimensions_star.astype(dtype=ct.c_uint, copy=True)

        self.ct_zeta_star_list = []
        for i_surf in range(self.n_surf):
            for i_dim in range(NDIM):
                self.ct_zeta_star_list.append(self.zeta_star[i_surf][i_dim, :, :].reshape(-1))

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

        self.ct_dist_to_orig_list = []
        for i_surf in range(self.n_surf):
            self.ct_dist_to_orig_list.append(self.dist_to_orig[i_surf][:, :].reshape(-1))

        self.ct_wake_conv_vel_list = []
        for i_surf in range(self.n_surf):
            self.ct_wake_conv_vel_list.append(self.wake_conv_vel[i_surf][:, :].reshape(-1))
        self.ct_flag_zeta_phantom_list = self.flag_zeta_phantom[:].reshape(-1)
 


        self.ct_p_dimensions_star = ((ct.POINTER(ct.c_uint)*n_surf)
                                     (* np.ctypeslib.as_ctypes(self.ct_dimensions_star)))
        self.ct_p_zeta_star = ((ct.POINTER(ct.c_double)*len(self.ct_zeta_star_list))
                               (* [np.ctypeslib.as_ctypes(array) for array in self.ct_zeta_star_list]))
        self.ct_p_u_ext_star = ((ct.POINTER(ct.c_double)*len(self.ct_u_ext_star_list))
                           (* [np.ctypeslib.as_ctypes(array) for array in self.ct_u_ext_star_list]))
        self.ct_p_gamma = ((ct.POINTER(ct.c_double)*len(self.ct_gamma_list))
                           (* [np.ctypeslib.as_ctypes(array) for array in self.ct_gamma_list]))
        self.ct_p_gamma_dot = ((ct.POINTER(ct.c_double)*len(self.ct_gamma_dot_list))
                               (* [np.ctypeslib.as_ctypes(array) for array in self.ct_gamma_dot_list]))
        self.ct_p_gamma_star = ((ct.POINTER(ct.c_double)*len(self.ct_gamma_star_list))
                                (* [np.ctypeslib.as_ctypes(array) for array in self.ct_gamma_star_list]))
        self.ct_p_dist_to_orig = ((ct.POINTER(ct.c_double)*len(self.ct_dist_to_orig_list))
                           (* [np.ctypeslib.as_ctypes(array) for array in self.ct_dist_to_orig_list]))
        self.ct_p_wake_conv_vel = ((ct.POINTER(ct.c_double)*len(self.ct_wake_conv_vel_list))
                           (* [np.ctypeslib.as_ctypes(array) for array in self.ct_wake_conv_vel_list]))
        self.ct_p_flag_zeta_phantom = np.ctypeslib.as_ctypes(self.ct_flag_zeta_phantom_list)
 


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
        runtime_steady_forces (np.ndarray): Steady forces generated at runtime through runtime generators
          ``[num_node x 6]``. Expressed in B FoR
        runtime_unsteady_forces (np.ndarray): Unsteady forces generated at runtime through runtime generators
          ``[num_node x 6]``. Expressed in B FoR
        gravity_forces (np.ndarray): Gravity forces at nodes ``[num_node x 6]``. Expressed in A FoR

        total_gravity_forces (np.ndarray): Total gravity forces on the structure ``[6]``. Expressed in A FoR
        total_forces (np.ndarray): Total forces applied to the structure ``[6]``. Expressed in A FoR

        q (np.ndarray): State vector associated to the structural system of equations ``[num_dof + 10]``
        dqdt (np.ndarray): Time derivative of ``q``
        dqddt (np.ndarray): Time derivative of ``dqdt``

        postproc_cell (dict): Variables associated to cells to be postprocessed
        postproc_node (dict): Variables associated to nodes to be postprocessed

        psi_local (np.ndarray): Cartesian Rotation Vector for each node in each element in local FoR
        psi_dot_local (np.ndarray): Time derivative of ``psi`` in the local FoR
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
        self.runtime_steady_forces = np.zeros((self.num_node, 6), dtype=ct.c_double, order='F')
        self.runtime_unsteady_forces = np.zeros((self.num_node, 6), dtype=ct.c_double, order='F')
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
        self.psi_local = np.zeros((self.num_elem, num_node_elem, 3), dtype=ct.c_double, order='F')
        self.psi_dot_local = np.zeros((self.num_elem, num_node_elem, 3), dtype=ct.c_double, order='F')
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
        copied.runtime_steady_forces = self.runtime_steady_forces.astype(dtype=ct.c_double, order='F', copy=True)
        copied.runtime_unsteady_forces = self.runtime_unsteady_forces.astype(dtype=ct.c_double, order='F', copy=True)
        copied.gravity_forces = self.gravity_forces.astype(dtype=ct.c_double, order='F', copy=True)
        copied.total_gravity_forces = self.total_gravity_forces.astype(dtype=ct.c_double, order='F', copy=True)
        copied.total_forces = self.total_forces.astype(dtype=ct.c_double, order='F', copy=True)

        copied.q = self.q.astype(dtype=ct.c_double, order='F', copy=True)
        copied.dqdt = self.dqdt.astype(dtype=ct.c_double, order='F', copy=True)
        copied.dqddt = self.dqddt.astype(dtype=ct.c_double, order='F', copy=True)

        copied.postproc_cell = copy.deepcopy(self.postproc_cell)
        copied.postproc_node = copy.deepcopy(self.postproc_node)

        copied.psi_local = self.psi_local.astype(dtype=ct.c_double, order='F', copy=True)
        copied.psi_dot_local = self.psi_dot_local.astype(dtype=ct.c_double, order='F', copy=True)
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
        ibody_StructTimeStepInfo.psi_local = self.psi_local[ibody_elems,:,:].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.psi_dot = self.psi_dot[ibody_elems,:,:].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.psi_dot_local = self.psi_dot_local[ibody_elems,:,:].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.psi_ddot = self.psi_ddot[ibody_elems,:,:].astype(dtype=ct.c_double, order='F', copy=True)

        ibody_StructTimeStepInfo.steady_applied_forces = self.steady_applied_forces[ibody_nodes,:].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.unsteady_applied_forces = self.unsteady_applied_forces[ibody_nodes,:].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.runtime_steady_forces = self.runtime_steady_forces[ibody_nodes,:].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_StructTimeStepInfo.runtime_unsteady_forces = self.runtime_unsteady_forces[ibody_nodes,:].astype(dtype=ct.c_double, order='F', copy=True)
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

    def compute_psi_local_AFoR(self, for0_pos, for0_vel, quat0):
        """
        compute_psi_local_AFoR

        Compute psi and psi_dot in the local A frame of reference

        Args:
            for0_pos (np.ndarray): Position of the global A FoR
            for0_vel (np.ndarray): Velocity of the global A FoR
            quat0 (np.ndarray): Quaternion of the global A FoR
        """

        # Define the rotation matrices between the different FoR
        CAslaveG = algebra.quat2rotation(self.quat).T
        CGAmaster = algebra.quat2rotation(quat0)
        Csm = np.dot(CAslaveG, CGAmaster)

        for ielem in range(self.psi.shape[0]):
            for inode in range(3):
                self.psi_local[ielem, inode, :] = algebra.rotation2crv(np.dot(Csm,algebra.crv2rotation(self.psi[ielem,inode,:])))
                self.psi_dot_local[ielem, inode, :] = np.zeros((3))

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

        # Modify position
        for inode in range(self.pos.shape[0]):
            vel_master = (self.pos_dot[inode,:] +
                          for0_vel[0:3] +
                          algebra.cross3(for0_vel[3:6], self.pos[inode, :]))
            self.pos[inode, :] = np.dot(Csm,self.pos[inode,:]) + np.dot(CAslaveG, for0_pos[0:3] - self.for_pos[0:3])
            self.pos_dot[inode, :] = (np.dot(Csm, vel_master) -
                                     self.for_vel[0:3] -
                                     algebra.cross3(self.for_vel[3:6], self.pos[inode,:]))

            self.gravity_forces[inode, 0:3] = np.dot(Csm, self.gravity_forces[inode, 0:3])
            self.gravity_forces[inode, 3:6] = np.dot(Csm, self.gravity_forces[inode, 3:6])

        # Modify local rotations
        for ielem in range(self.psi.shape[0]):
            for inode in range(3):
                # psi_master = self.psi[ielem, inode, :] + np.zeros((3,),)
                # self.psi[ielem, inode, :] = np.dot(Csm, self.psi[ielem, inode, :])
                # self.psi_dot[ielem, inode, :] = (np.dot(Csm, self.psi_dot[ielem, inode, :] + algebra.cross3(for0_vel[3:6], psi_master)) -
                #                                  algebra.multiply_matrices(CAslaveG, algebra.skew(self.for_vel[3:6]), CGAmaster, psi_master))

                # psi_master = self.psi[ielem,inode,:] + np.zeros((3,),)
                # self.psi[ielem, inode, :] = algebra.rotation2crv(np.dot(Csm,algebra.crv2rotation(self.psi[ielem,inode,:])))
                # psi_slave = self.psi[ielem, inode, :] + np.zeros((3,),)
                # cbam = algebra.crv2rotation(psi_master).T
                # cbas = algebra.crv2rotation(psi_slave).T
                # tm = algebra.crv2tan(psi_master)
                # inv_ts = np.linalg.inv(algebra.crv2tan(psi_slave))

                # self.psi_dot[ielem, inode, :] = np.dot(inv_ts, (np.dot(tm, self.psi_dot[ielem, inode, :]) +
                #                                                 np.dot(cbam, for0_vel[3:6]) -
                #                                                 np.dot(cbas, self.for_vel[3:6])))
                self.psi[ielem, inode, :] = self.psi_local[ielem,inode,:].copy()
                self.psi_dot[ielem, inode, :] = self.psi_dot_local[ielem, inode, :].copy()


    def change_to_global_AFoR(self, for0_pos, for0_vel, quat0):
        """
        Reference a :class:`~sharpy.utils.datastructures.StructTimeStepInfo` to the global A frame of reference

        Args:
            for0_pos (np.ndarray): Position of the global A FoR
            for0_vel (np.ndarray): Velocity of the global A FoR
            quat0 (np.ndarray): Quaternion of the global A FoR
        """

        # Define the rotation matrices between the different FoR
        CGAslave = algebra.quat2rotation(self.quat)
        CAmasterG = algebra.quat2rotation(quat0).T
        Cms = np.dot(CAmasterG, CGAslave)

        for inode in range(self.pos.shape[0]):
            vel_slave = (self.pos_dot[inode, :] +
                         self.for_vel[0:3] +
                         algebra.cross3(self.for_vel[3:6], self.pos[inode, :]))
            self.pos[inode, :] = np.dot(Cms, self.pos[inode,:]) + np.dot(CAmasterG, self.for_pos[0:3] - for0_pos[0:3])
            self.pos_dot[inode, :] = (np.dot(Cms, vel_slave) -
                                      for0_vel[0:3] -
                                      algebra.cross3(for0_vel[3:6], self.pos[inode, :]))

            self.gravity_forces[inode, 0:3] = np.dot(Cms, self.gravity_forces[inode, 0:3])
            self.gravity_forces[inode, 3:6] = np.dot(Cms, self.gravity_forces[inode, 3:6])

        for ielem in range(self.psi.shape[0]):
            for inode in range(3):
                # psi_slave = self.psi[ielem,inode,:] + np.zeros((3,),)
                # self.psi[ielem, inode, :] = np.dot(Cms, self.psi[ielem, inode, :])
                # self.psi_dot[ielem, inode, :] = (np.dot(Cms, self.psi_dot[ielem, inode, :] + algebra.cross3(self.for_vel[3:6], psi_slave)) -
                #                                  algebra.multiply_matrices(CAmasterG, algebra.skew(self.for0_vel[3:6]), CGAslave, psi_slave))


                self.psi_local[ielem, inode, :] = self.psi[ielem, inode, :].copy() # Copy here the result from the structural computation
                self.psi_dot_local[ielem, inode, :] = self.psi_dot[ielem, inode, :].copy() # Copy here the result from the structural computation

                # psi_slave = self.psi[ielem, inode, :] + np.zeros((3,),)
                self.psi[ielem, inode, :] = algebra.rotation2crv(np.dot(Cms,algebra.crv2rotation(self.psi[ielem,inode,:])))

                # Convert psi_dot_local to psi_dot to be used by the rest of the code
                psi_master = self.psi[ielem,inode,:] + np.zeros((3,),)
                cbam = algebra.crv2rotation(psi_master).T
                cbas = algebra.crv2rotation(self.psi_local[ielem, inode, :]).T
                ts = algebra.crv2tan(self.psi_local[ielem, inode, :])
                inv_tm = np.linalg.inv(algebra.crv2tan(psi_master))

                self.psi_dot[ielem, inode, :] = np.dot(inv_tm, (np.dot(ts, self.psi_dot_local[ielem, inode, :]) +
                                                                np.dot(cbas, self.for_vel[3:6]) -
                                                                np.dot(cbam, for0_vel[3:6])))

    # def whole_structure_to_local_AFoR(self, beam, compute_psi_local=False):
    #     """
    #     Same as change_to_local_AFoR but for a multibody structure
    #
    #     Args:
    #         beam(sharpy.structure.models.beam.Beam): Beam structure of ``PreSharpy``
    #     """
    #     if not self.in_global_AFoR:
    #         raise NotImplementedError("Wrong managing of FoR")
    #
    #     self.in_global_AFoR = False
    #     quat0 = self.quat.astype(dtype=ct.c_double, order='F', copy=True)
    #     for0_pos = self.for_pos.astype(dtype=ct.c_double, order='F', copy=True)
    #     for0_vel = self.for_vel.astype(dtype=ct.c_double, order='F', copy=True)
    #
    #     MB_beam = [None]*beam.num_bodies
    #     MB_tstep = [None]*beam.num_bodies
    #
    #     for ibody in range(beam.num_bodies):
    #         MB_beam[ibody] = beam.get_body(ibody = ibody)
    #         MB_tstep[ibody] = self.get_body(beam, MB_beam[ibody].num_dof, ibody = ibody)
    #         if compute_psi_local:
    #             MB_tstep[ibody].compute_psi_local_AFoR(for0_pos, for0_vel, quat0)
    #         MB_tstep[ibody].change_to_local_AFoR(for0_pos, for0_vel, quat0)
    #
    #     first_dof = 0
    #     for ibody in range(beam.num_bodies):
    #         # Renaming for clarity
    #         ibody_elems = MB_beam[ibody].global_elems_num
    #         ibody_nodes = MB_beam[ibody].global_nodes_num
    #
    #         # Merge tstep
    #         self.pos[ibody_nodes,:] = MB_tstep[ibody].pos.astype(dtype=ct.c_double, order='F', copy=True)
    #         self.psi[ibody_elems,:,:] = MB_tstep[ibody].psi.astype(dtype=ct.c_double, order='F', copy=True)
    #         self.psi_local[ibody_elems,:,:] = MB_tstep[ibody].psi_local.astype(dtype=ct.c_double, order='F', copy=True)
    #         self.gravity_forces[ibody_nodes,:] = MB_tstep[ibody].gravity_forces.astype(dtype=ct.c_double, order='F', copy=True)
    #
    #         self.pos_dot[ibody_nodes,:] = MB_tstep[ibody].pos_dot.astype(dtype=ct.c_double, order='F', copy=True)
    #         self.psi_dot[ibody_elems,:,:] = MB_tstep[ibody].psi_dot.astype(dtype=ct.c_double, order='F', copy=True)
    #         self.psi_dot_local[ibody_elems,:,:] = MB_tstep[ibody].psi_dot_local.astype(dtype=ct.c_double, order='F', copy=True)
    #
    #         # TODO: Do I need a change in FoR for the following variables? Maybe for the FoR ones.
    #         # tstep.forces_constraints_nodes[ibody_nodes,:] = MB_tstep[ibody].forces_constraints_nodes.astype(dtype=ct.c_double, order='F', copy=True)
    #         # tstep.forces_constraints_FoR[ibody, :] = MB_tstep[ibody].forces_constraints_FoR[ibody, :].astype(dtype=ct.c_double, order='F', copy=True)

    # def whole_structure_to_global_AFoR(self, beam):
    #     """
    #     Same as change_to_global_AFoR but for a multibody structure
    #
    #     Args:
    #         beam(sharpy.structure.models.beam.Beam): Beam structure of ``PreSharpy``
    #     """
    #     if self.in_global_AFoR:
    #         raise NotImplementedError("Wrong managing of FoR")
    #
    #     self.in_global_AFoR = True
    #
    #     MB_beam = [None]*beam.num_bodies
    #     MB_tstep = [None]*beam.num_bodies
    #     quat0 = self.quat.astype(dtype=ct.c_double, order='F', copy=True)
    #     for0_pos = self.for_pos.astype(dtype=ct.c_double, order='F', copy=True)
    #     for0_vel = self.for_vel.astype(dtype=ct.c_double, order='F', copy=True)
    #
    #     for ibody in range(beam.num_bodies):
    #         MB_beam[ibody] = beam.get_body(ibody = ibody)
    #         MB_tstep[ibody] = self.get_body(beam, MB_beam[ibody].num_dof, ibody = ibody)
    #         MB_tstep[ibody].change_to_global_AFoR(for0_pos, for0_vel, quat0)
    #
    #
    #     first_dof = 0
    #     for ibody in range(beam.num_bodies):
    #         # Renaming for clarity
    #         ibody_elems = MB_beam[ibody].global_elems_num
    #         ibody_nodes = MB_beam[ibody].global_nodes_num
    #
    #         # Merge tstep
    #         self.pos[ibody_nodes,:] = MB_tstep[ibody].pos.astype(dtype=ct.c_double, order='F', copy=True)
    #         self.psi[ibody_elems,:,:] = MB_tstep[ibody].psi.astype(dtype=ct.c_double, order='F', copy=True)
    #         self.psi_local[ibody_elems,:,:] = MB_tstep[ibody].psi_local.astype(dtype=ct.c_double, order='F', copy=True)
    #         self.gravity_forces[ibody_nodes,:] = MB_tstep[ibody].gravity_forces.astype(dtype=ct.c_double, order='F',
    #                                                                                    copy=True)
    #
    #         self.pos_dot[ibody_nodes,:] = MB_tstep[ibody].pos_dot.astype(dtype=ct.c_double, order='F', copy=True)
    #         self.psi_dot[ibody_elems,:,:] = MB_tstep[ibody].psi_dot.astype(dtype=ct.c_double, order='F', copy=True)
    #         self.psi_dot_local[ibody_elems,:,:] = MB_tstep[ibody].psi_dot_local.astype(dtype=ct.c_double, order='F', copy=True)


    def nodal_b_for_2_a_for(self, nodal, beam, filter=np.array([True]*6), ibody=None):
        """
        Projects a nodal variable from the local, body-attached frame (B) to the reference A frame.

        Args:
            nodal (np.array): Nodal variable of size ``(num_node, 6)``
            beam (sharpy.datastructures.StructTimeStepInfo): beam info.
            filter (np.array): optional argument that filters and does not convert a specific degree of
              freedom. Defaults to ``np.array([True, True, True, True, True, True])``.

        Returns:
            np.array: the ``nodal`` argument projected onto the reference ``A`` frame.
        """
        nodal_a = np.zeros_like(nodal)
        for i_node in range(self.num_node):
            # get master elem and i_local_node
            i_master_elem, i_local_node = beam.node_master_elem[i_node, :]
            if ((ibody is None) or (beam.body_number[i_master_elem] == ibody)):
                crv = self.psi[i_master_elem, i_local_node, :]
                cab = algebra.crv2rotation(crv)
                nodal_a[i_node, 0:3] = np.dot(cab, nodal[i_node, 0:3])
                nodal_a[i_node, 3:6] = np.dot(cab, nodal[i_node, 3:6])
                nodal_a *= filter

        return nodal_a

    def nodal_type_b_for_2_a_for(self, beam,
                            force_type=['steady', 'unsteady'],
                            filter=np.array([True]*6),
                            ibody=None):
        forces_output = []
        for ft in force_type:
            if ft == 'steady':
                fb = self.steady_applied_forces
            elif ft == 'unsteady':
                fb = self.unsteady_applied_forces

            forces_output.append(self.nodal_b_for_2_a_for(fb, beam, filter=filter, ibody=ibody))

        return forces_output


    def extract_resultants(self, beam, force_type=['steady', 'unsteady', 'grav'], ibody=None):

        forces_output = []
        for ft in force_type:
            totals = np.zeros((6))
            if ft == 'steady':
                fa = self.nodal_type_b_for_2_a_for(beam, force_type=['steady'], ibody=ibody)[0]
            elif ft == 'grav':
                fa = self.gravity_forces.copy()
            elif ft == 'unsteady':
                fa = self.nodal_type_b_for_2_a_for(beam, force_type=['unsteady'], ibody=ibody)[0]

            for i_node in range(beam.num_node):
                totals += fa[i_node, :]
                totals[3:6] += algebra.cross3(self.pos[i_node, :],
                                              fa[i_node, 0:3])

            forces_output.append(totals)

        return forces_output

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
