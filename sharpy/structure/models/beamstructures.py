# Alfonso del Carre
import numpy as np

import sharpy.utils.algebra as algebra


class Element(object):
    """
    This class stores all the required data for the definition of
    a linear or quadratic beam element.
    """
    ordering = [0, 2, 1]
    max_nodes_elem = 3

    def __init__(self,
                 ielem,
                 n_nodes,
                 global_connectivities,
                 coordinates,
                 frame_of_reference_delta,
                 structural_twist,
                 num_mem,
                 stiff_index,
                 mass_index):
        # store info in instance
        # global element number
        self.ielem = ielem
        # number of nodes per elem
        self.n_nodes = n_nodes
        if self.max_nodes_elem < self.n_nodes:
            raise AttributeError('Elements with more than 3 nodes are not allowed')
        # global connectivities (global node numbers)
        self.global_connectivities = global_connectivities
        self.reordered_global_connectivities = global_connectivities[self.ordering]
        # coordinates of the nodes in a-frame (body-fixed frame)
        self.coordinates_def = coordinates.copy()
        # frame of reference points
        self.frame_of_reference_delta = frame_of_reference_delta
        # structural twist
        self.structural_twist = structural_twist
        # number in memory (for fortran routines)
        self.num_mem = num_mem
        # stiffness and mass matrices indices (stored in parent beam class)
        self.stiff_index = stiff_index
        self.mass_index = mass_index

        # placeholder for RBMass
        self.rbmass = None  # np.zeros((self.max_nodes_elem, 6, 6))

        self.update(self.coordinates_def)

    def update(self, coordinates_def, psi_def=None):
        self.coordinates_def = coordinates_def.copy()

        if psi_def is not None:
            # element orientation
            self.psi_def = psi_def.copy()

        # element length
        self.calculate_length()

        if psi_def is None:  # ini conditions, initial crv has to be calculated
            # we need to define the FoR z direction for every beam element
            v1, v2, v3 = self.get_triad()
            self.psi_ini = algebra.triad2crv_vec(v1, v2, v3)
            self.psi_def = self.psi_ini.copy()

            # copy all the info to _ini fields
            self.coordinates_ini = self.coordinates_def.copy()

    def calculate_length(self):
        # TODO implement length based on integration
        self.length = np.linalg.norm(self.coordinates_def[0, :] - self.coordinates_def[1, :])

    def add_attributes(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

    def generate_curve(self, n_elem_curve, defor=False):
        curve = np.zeros((n_elem_curve, 3))
        t_vec = np.linspace(0, 2, n_elem_curve)
        for i in range(n_elem_curve):
            t = t_vec[i]
            for idim in range(3):
                if defor:
                    polyfit, _, _ = algebra.get_polyfit(self.coordinates_def, self.ordering)
                else:
                    polyfit, _, _ = algebra.get_polyfit(self.coordinates_ini, self.ordering)
                polyf = np.poly1d(polyfit[idim])
                curve[i, idim] = (polyf(t))
        return curve

    def get_triad(self):
        """
        Generates two unit vectors in body FoR that define the local FoR for
        a beam element. These vectors are calculated using `frame_of_reference_delta`
        :return:
        """
        # now, calculate tangent vector (and coefficients of the polynomial
        # fit just in case)
        tangent, polyfit = algebra.tangent_vector(
            self.coordinates_def,
            Element.ordering)
        normal = np.zeros_like(tangent)
        binormal = np.zeros_like(tangent)

        # v_vector is the vector with origin the FoR node and delta
        # equals frame_of_reference_delta
        for inode in range(self.n_nodes):
            v_vector = self.frame_of_reference_delta[inode, :]
            normal[inode, :] = algebra.unit_vector(np.cross(
                                                        tangent[inode, :],
                                                        v_vector
                                                        )
                                                   )
            binormal[inode, :] = -algebra.unit_vector(np.cross(
                                                        tangent[inode, :],
                                                        normal[inode, :]
                                                                )
                                                      )

        # we apply twist now
        for inode in range(self.n_nodes):
            if not self.structural_twist[inode] == 0.0:
                rotation_mat = algebra.rotation_matrix_around_axis(tangent[inode, :],
                                                                   self.structural_twist[inode])
                normal[inode, :] = np.dot(rotation_mat, normal[inode, :])
                binormal[inode, :] = np.dot(rotation_mat, binormal[inode, :])

        return tangent, binormal, normal

    def deformed_triad(self, psi=None):
        if psi is None:
            return algebra.crv2triad_vec(self.psi_def)
        else:
            return algebra.crv2triad_vec(psi)
