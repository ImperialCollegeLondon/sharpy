# Alfonso del Carre
import numpy as np
import presharpy.beam.beamutils as beamutils


class Element(object):
    '''
    This class stores all the required data for the definition of
    a linear or quadratic beam element.
    '''
    def __init__(self,
                 ielem,
                 n_nodes,
                 global_connectivities,
                 coordinates,
                 frame_of_reference_delta):
        # store info in instance
        # global element number
        self.ielem = ielem
        # number of nodes per elem
        self.n_nodes = n_nodes
        # global connectivities (global node numbers)
        self.global_connectivities = global_connectivities
        # coordinates of the nodes in a (body-fixed frame)
        self.coordinates = coordinates
        # element length
        self.length = np.linalg.norm(self.coordinates[0, :] - self.coordinates[n_nodes-1, :])
        # frame of reference points
        self.frame_of_reference_delta = frame_of_reference_delta

        # now, calculate tangent vector (and coefficients of the polynomial
        # fit just in case)
        self.tangent_vector, self.polyfit_vec = beamutils.tangent_vector(
                                                    self.coordinates,
                                                    self.n_nodes)

        # we need to define the FoR z direction for every beam element
        # self.normal_vector = beamutils.normal_vector(
        #                                             self.tangent_vector)
        self.get_triad()

        # import pdb; pdb.set_trace()
        # y direction of the beam elem
        # self.binormal_vector = np.zeros_like(self.normal_vector)
        # for inode in range(self.n_nodes):
        #     self.binormal_vector[inode, :] = (
        #             np.cross(self.normal_vector[inode, :],
        #                      self.tangent_vector[inode, :]))

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
        #TODO
        pass

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

