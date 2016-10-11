# Alfonso del Carre
# alfonso.del-carre14@imperial.ac.uk
# Imperial College London
# LoCA lab
# 29 Sept 2016

# AeroGrid contains all the necessary routines to generate an aerodynamic
# grid based on the input dictionaries.

import numpy as np
import scipy as sc
import scipy.interpolate as interpolate
import sympy.geometry as geo
import presharpy.aerogrid.gridutils as gridutils


class AeroGrid(object):

    def __init__(self, aero_dict, solver_config, flightcon_config, beam):
        print('Generating aerodynamic grid...')
        # get dimensions and settings
        self.num_node = beam.num_node
        self.num_elem = beam.num_elem
        self.num_node_elem = beam.num_node_elem

        self.M = solver_config['GRID'].getint('M')
        self.M_distribution = solver_config['GRID']['M_distribution']
        if not self.M_distribution == 'uniform':
            raise NotImplementedError('M distribution only uniform, modify place_airfoil')
        self.wake_length = solver_config['GRID'].getfloat('wake_length')
        self.rollup = solver_config['GRID'].getboolean('rollup')
        self.aligned_grid = solver_config['GRID'].getboolean('aligned_grid')
        if not self.aligned_grid:
            raise NotImplementedError('Not aligned grids are not supported (yet)')
        self.beta = (flightcon_config['FLIGHT_CONDITIONS'].getfloat('beta')*np.pi/180)
        self.alpha = (flightcon_config['FLIGHT_CONDITIONS'].getfloat('alpha')*np.pi/180)
        self.delta = (flightcon_config['FLIGHT_CONDITIONS'].getfloat('delta')*np.pi/180)
        self.Q = flightcon_config['FLIGHT_CONDITIONS'].getfloat('Q')
        self.freestream_direction = [np.cos(self.alpha)*np.cos(self.beta),
                                     np.sin(self.beta),
                                     np.sin(self.alpha)]
        # store airfoil db and other aero data for the model
        self.aero_node = aero_dict['aero_node']
        self.airfoil_distribution = aero_dict['airfoil_distribution']
        self.airfoils = aero_dict['airfoils']
        self.twist = aero_dict['twist']
        self.chord = aero_dict['chord']
        self.elastic_axis = aero_dict['elastic_axis']

        self.beam = beam

        # generating zeta
        self.generate_grid()

    def generate_grid(self):
        # for every node, align the given airfoil with the [1, 0, 0] vector in body frame
        # and then apply the other transformations (twist...)
        x_direction = np.array([1, 0, 0])
        # we calculate the interpolation class for the airfoils just once:
        self.airfoil_interpolators = dict()
        for k, v in self.airfoils.items():
            try:
                float(k)
            except ValueError:
                continue

            x = v[:, 0]
            z = v[:, 1]
            self.airfoil_interpolators[k] = interpolate.interp1d(
                                                                 x,
                                                                 z,
                                                                 kind=2)

        self.airfoil_coords = np.zeros((self.num_node, self.M, 3))
        for inode in range(self.num_node):
            # just beam, no lifting surface
            if not self.aero_node[inode]:
                continue

            self.airfoil_coords[inode, :, :] = self.place_airfoil(inode, x_direction)


    def place_airfoil(self, inode, direction=np.array([1, 0, 0])):
        # finds the orientation and location of the airfoil based on
        # aero data and the structure class
        # returns the airfoil coordinates in body (a) FoR
        airfoil_coords = np.zeros((self.M, 3))
        kairfoil = str(self.airfoil_distribution[inode])
        if self.M_distribution == 'uniform':
            local_x = np.linspace(0, 1, self.M)
        else:
            raise NotImplemented('Non-uniform chordwise distribution not yet supported')

        local_z = (self.airfoil_interpolators[kairfoil])(local_x)
        # elastic axis
        local_x -= self.elastic_axis[inode]
        local_x *= self.chord[inode]
        local_z *= self.chord[inode]

        # we apply the twist rotation now
        rotation = gridutils.rot_matrix_2d(self.twist[inode])
        for iM in range(self.M):
            temp = np.dot(rotation, np.array([local_x[iM], local_z[iM]]))
            local_x[iM] = temp[0]
            local_z[iM] = temp[1]
        # now dihedral rotation (given by angle between [0 1 0] and t vector)
        local_y = np.zeros_like(local_x)
        ielem = self.beam.node_master_elem[inode, 0]  # elem to which the node belongs
        i_local_node = self.beam.node_master_elem[inode, 1]  # node of the element which is our node
        elem = self.beam.elements[ielem]
        tangent_vec = elem.tangent_vector
        dihedral_angle = gridutils.angle_between_vectors([0, 1, 0], tangent_vec[i_local_node, :])
        dihedral_rotation = gridutils.x_rotation_matrix_3d(dihedral_angle)
        for iM in range(self.M):
            temp = np.dot(dihedral_rotation,
                          np.array([local_x[iM], local_y[iM], local_z[iM]]))
            local_x[iM] = temp[0]
            local_y[iM] = temp[1]
            local_z[iM] = temp[2]

        # this could be merged in the previous loop, but I like it simple
        for iM in range(self.M):
            airfoil_coords[iM, :] = ([local_x[iM], local_y[iM], local_z[iM]] +
                                     self.beam.node_coordinates[inode])

        return airfoil_coords

    def plot(self, fig=None, ax=None):
        #TODO preliminary plotting routine
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
        # airfoils by nodes
        for inode in range(self.num_node):
            airfoil = ax.plot(self.airfoil_coords[inode, :, 0],
                              self.airfoil_coords[inode, :, 1],
                              self.airfoil_coords[inode, :, 2],
                              'k')
        plt.hold('off')






