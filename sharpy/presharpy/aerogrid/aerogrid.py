# Alfonso del Carre
# alfonso.del-carre14@imperial.ac.uk
# Imperial College London
# LoCA lab
# 29 Sept 2016

# AeroGrid contains all the necessary routines to generate an aerodynamic
# grid based on the input dictionaries.

import numpy as np
import scipy.interpolate as interpolate

import sharpy.presharpy.utils.algebra as algebra
import sharpy.utils.cout_utils as cout


class AeroGrid(object):
    def __init__(self, beam, aero_dict, settings):
        # number of surfaces
        self.n_surf = len(set(aero_dict['surface_distribution']))
        # number of chordwise panels
        self.surface_m = aero_dict['surface_m']
        # number of total nodes (structural + aero&struc)
        self.total_nodes = len(aero_dict['aero_node'])
        # number of aero nodes
        self.n_aero_nodes = sum(aero_dict['aero_node'])

        # get N per surface
        self.aero_dimensions = np.zeros((self.n_surf, 2), dtype=int)
        for i in range(self.n_surf):
            # adding M values
            self.aero_dimensions[i, 0] = self.surface_m[0]

        # count N values (actually, the count result
        # will be N+1)
        for i_node in range(self.n_aero_nodes):
            self.aero_dimensions[(aero_dict['surface_distribution'][i_node]), 1] +=\
                int(aero_dict['aero_node'][i_node])
        # accounting for N+1 nodes -> N panels
        self.aero_dimensions[:, 1] -= 1

        if settings['print_info']:
            cout.cout_wrap('The aerodynamic grid contains %u surfaces' % self.n_surf, 1)
            for i_surf in range(self.n_surf):
                cout.cout_wrap('  Surface %u, M=%u, N=%u' % (i_surf,
                                                             self.aero_dimensions[i_surf, 0],
                                                             self.aero_dimensions[i_surf, 1]), 1)
            cout.cout_wrap('  In total: %u bound panels' % sum(self.aero_dimensions[:, 0]*
                                                               self.aero_dimensions[:, 1]), 1)
        # generate placeholder for aero grid zeta coordinates
