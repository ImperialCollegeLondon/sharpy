# Alfonso del Carre
# alfonso.del-carre14@imperial.ac.uk
# Imperial College London
# LoCA lab
# 29 Sept 2016

# AeroGrid contains all the necessary routines to generate an aerodynamic
# grid based on the input dictionaries.

import numpy as np
import scipy as sc
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
        self.wake_length = solver_config['GRID'].getfloat('wake_length')
        self.rollup = solver_config['GRID'].getboolean('rollup')
        self.aligned_grid = solver_config['GRID'].getboolean('aligned_grid')
        if not self.aligned_grid:
            raise NotImplemented('Not aligned grids are not supported (yet)')

        self.beta = (flightcon_config['FLIGHT_CONDITIONS'].
                getfloat('beta')*np.pi/180)
        self.alpha = (flightcon_config['FLIGHT_CONDITIONS'].
                getfloat('alpha')*np.pi/180)
        self.delta = (flightcon_config['FLIGHT_CONDITIONS'].
                getfloat('delta')*np.pi/180)
        self.Q = flightcon_config['FLIGHT_CONDITIONS'].getfloat('Q')

        # import pdb; pdb.set_trace()









