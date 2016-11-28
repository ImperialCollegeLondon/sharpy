# Alfonso del Carre
import numpy as np
import copy
import presharpy.utils.algebra as algebra


class Element(object):
    """
    This class stores all the required data for the definition of
    a linear or quadratic beam element.
    """
    ordering = [0, 2, 1]

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
        # global connectivities (global node numbers)
        self.global_connectivities = global_connectivities
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

        self.update(self.coordinates_def, ini=True)


    def update(self, coordinates_def, ini=False):
        self.coordinates_def = coordinates_def.copy()

        # now, calculate tangent vector (and coefficients of the polynomial
        # fit just in case)
        self.tangent_vector_def, self.polyfit_vec_def = algebra.tangent_vector(
            self.coordinates_def,
            Element.ordering)

        # we need to define the FoR z direction for every beam element
        self.get_triad()

        # element length
        self.calculate_length()

        if ini:
            # copy all the info to _ini fields
            self.coordinates_ini = self.coordinates_def.copy()

            self.tangent_vector_ini = self.tangent_vector_def.copy()
            self.normal_vector_ini = self.normal_vector_def.copy()
            self.binormal_vector_ini = self.binormal_vector_def.copy()

            self.polyfit_vec_ini = copy.deepcopy(self.polyfit_vec_def)

    def calculate_length(self):
        # TODO implement length based on integration
        self.length = np.linalg.norm(self.coordinates_def[0, :] - self.coordinates_def[1, :])

    def preferent_direction(self):
        index = np.argmax(np.abs(self.tangent_vector_def[0, :]))
        direction = np.zeros((3,))
        direction[index] = 1
        return direction

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
                    polyf = np.poly1d(self.polyfit_vec_def[idim])
                else:
                    polyf = np.poly1d(self.polyfit_vec_ini[idim])
                curve[i, idim] = (polyf(t))
        return curve

    def get_triad(self):
        """
        Generates two unit vectors in body FoR that define the local FoR for
        a beam element. These vectors are calculated using `frame_of_reference_delta`
        :return:
        """

        self.normal_vector_def = np.zeros_like(self.tangent_vector_def)
        self.binormal_vector_def = np.zeros_like(self.tangent_vector_def)

        # v_vector is the vector with origin the FoR node and delta
        # equals frame_of_reference_delta
        for inode in range(self.n_nodes):
            v_vector = self.frame_of_reference_delta[inode, :]
            self.normal_vector_def[inode, :] = algebra.unit_vector(np.cross(
                                                                        self.tangent_vector_def[inode, :],
                                                                        v_vector
                                                                        )
                                                               )
            self.binormal_vector_def[inode, :] = -algebra.unit_vector(np.cross(
                                                                        self.tangent_vector_def[inode, :],
                                                                        self.normal_vector_def[inode, :]
                                                                            )
                                                                  )

        # we apply twist now
        for inode in range(self.n_nodes):
            rotation_mat = algebra.rotation_matrix_around_axis(self.tangent_vector_def[inode, :],
                                                               self.structural_twist)
            self.normal_vector_def[inode, :] = np.dot(rotation_mat, self.normal_vector_def[inode, :])
            self.binormal_vector_def[inode, :] = np.dot(rotation_mat, self.binormal_vector_def[inode, :])

    def plot(self, fig=None, ax=None, plot_triad=False, n_elem_plot=10, defor=False):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D, proj3d
        if fig is None or ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            plt.title('Structure plot')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_zlabel('z (m)')

        plt.hold('on')
        # generate line for plotting element
        curve = self.generate_curve(n_elem_plot, defor)
        if defor:
            colour = 'b'
        else:
            colour = 'k'
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], colour+'-')
        if plot_triad:
            scale_val = 1
            length = 0.06
            if not defor:
                coords = self.coordinates_ini
                tangent_vec = self.tangent_vector_ini
                normal_vec = self.normal_vector_ini
                binormal_vec = self.binormal_vector_ini
            else:
                coords = self.coordinates_def
                tangent_vec = self.tangent_vector_def
                normal_vec = self.normal_vector_def
                binormal_vec = self.binormal_vector_def

            ax.quiver(coords[:, 0], coords[:, 1], coords[:, 2],
                      tangent_vec[:, 0]*scale_val,
                      tangent_vec[:, 1]*scale_val,
                      tangent_vec[:, 2]*scale_val,
                      length=length,
                      pivot='tail', colors=[1, 0, 0])
            ax.quiver(coords[:, 0], coords[:, 1], coords[:, 2],
                      binormal_vec[:, 0]*scale_val,
                      binormal_vec[:, 1]*scale_val,
                      binormal_vec[:, 2]*scale_val,
                      length=length,
                      pivot='tail', colors=[0, 1, 0])
            ax.quiver(coords[:, 0], coords[:, 1], coords[:, 2],
                      normal_vec[:, 0]*scale_val,
                      normal_vec[:, 1]*scale_val,
                      normal_vec[:, 2]*scale_val,
                      length=length,
                      pivot='tail', colors=[0, 0, 1])
        plt.hold('off')

