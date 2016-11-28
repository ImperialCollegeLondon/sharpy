# Alfonso del Carre
import numpy as np
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
        self.coordinates = coordinates
        # element length
        # TODO implement length based on integration
        self.length = np.linalg.norm(self.coordinates[0, :] - self.coordinates[1, :])
        # frame of reference points
        self.frame_of_reference_delta = frame_of_reference_delta
        # structural twist
        self.structural_twist = structural_twist
        # number in memory (for fortran routines)
        self.num_mem = num_mem
        # stiffness and mass matrices indices (stored in parent beam class)
        self.stiff_index = stiff_index
        self.mass_index = mass_index

        # now, calculate tangent vector (and coefficients of the polynomial
        # fit just in case)
        self.tangent_vector, self.polyfit_vec = algebra.tangent_vector(
                                                    self.coordinates,
                                                    Element.ordering)

        # we need to define the FoR z direction for every beam element
        self.get_triad()

    def preferent_direction(self):
        index = np.argmax(np.abs(self.tangent_vector[0, :]))
        direction = np.zeros((3,))
        direction[index] = 1
        return direction

    def add_attributes(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

    def generate_curve(self, n_elem_curve):
        curve = np.zeros((n_elem_curve, 3))
        t_vec = np.linspace(0, 2, n_elem_curve)
        for i in range(n_elem_curve):
            t = t_vec[i]
            for idim in range(3):
                polyf = np.poly1d(self.polyfit_vec[idim])
                curve[i, idim] = (polyf(t))
        return curve

    def get_triad(self):
        '''
        Generates two unit vectors in body FoR that define the local FoR for
        a beam element. These vectors are calculated using `frame_of_reference_delta`
        :return:
        '''

        self.normal_vector = np.zeros_like(self.tangent_vector)
        self.binormal_vector = np.zeros_like(self.tangent_vector)

        # v_vector is the vector with origin the FoR node and delta
        # equals frame_of_reference_delta
        for inode in range(self.n_nodes):
            v_vector = self.frame_of_reference_delta[inode, :]
            self.normal_vector[inode, :] = algebra.unit_vector(np.cross(
                                                                        self.tangent_vector[inode, :],
                                                                        v_vector
                                                                        )
                                                               )
            self.binormal_vector[inode, :] = -algebra.unit_vector(np.cross(
                                                                        self.tangent_vector[inode, :],
                                                                        self.normal_vector[inode, :]
                                                                            )
                                                                  )

        # we apply twist now
        for inode in range(self.n_nodes):
            rotation_mat = algebra.rotation_matrix_around_axis(self.tangent_vector[inode, :],
                                                               self.structural_twist)
            self.normal_vector[inode, :] = np.dot(rotation_mat, self.normal_vector[inode, :])
            self.binormal_vector[inode, :] = np.dot(rotation_mat, self.binormal_vector[inode, :])

    def plot(self, fig=None, ax=None, plot_triad=False, n_elem_plot=10):
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
        curve = self.generate_curve(n_elem_plot)
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], 'k-')
        if plot_triad:
            scale_val = 1
            length = 0.06
            ax.quiver(self.coordinates[:, 0], self.coordinates[:, 1], self.coordinates[:, 2],
                      self.tangent_vector[:, 0]*scale_val,
                      self.tangent_vector[:, 1]*scale_val,
                      self.tangent_vector[:, 2]*scale_val,
                      length=length,
                      pivot='tail', colors=[1, 0, 0])
            ax.quiver(self.coordinates[:, 0], self.coordinates[:, 1], self.coordinates[:, 2],
                      self.binormal_vector[:, 0]*scale_val,
                      self.binormal_vector[:, 1]*scale_val,
                      self.binormal_vector[:, 2]*scale_val,
                      length=length,
                      pivot='tail', colors=[0, 1, 0])
            ax.quiver(self.coordinates[:, 0], self.coordinates[:, 1], self.coordinates[:, 2],
                      self.normal_vector[:, 0]*scale_val,
                      self.normal_vector[:, 1]*scale_val,
                      self.normal_vector[:, 2]*scale_val,
                      length=length,
                      pivot='tail', colors=[0, 0, 1])
        plt.hold('off')

