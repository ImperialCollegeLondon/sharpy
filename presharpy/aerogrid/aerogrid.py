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

        for inode in range(self.num_node):
            # just beam, no lifting surface
            if not self.aero_node[inode]:
                continue

            airfoil_coords = self.place_airfoil(inode, x_direction)


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



        return airfoil_coords








