# Alfonso del Carre
# alfonso.del-carre14@imperial.ac.uk
# Imperial College London
# LoCA lab
# 29 Sept 2016

# AeroGrid contains all the necessary routines to generate an aerodynamic
# grid based on the input dictionaries.

import ctypes as ct

import numpy as np
import scipy.interpolate

import sharpy.utils.algebra as algebra
import sharpy.utils.cout_utils as cout


class AeroGrid(object):
    def __init__(self, beam, aero_dict):
        self.aero_dict = aero_dict
        # self.settings = settings

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

        # if settings['print_info']:
        cout.cout_wrap('The aerodynamic grid contains %u surfaces' % self.n_surf, 1)
        for i_surf in range(self.n_surf):
            cout.cout_wrap('  Surface %u, M=%u, N=%u' % (i_surf,
                                                         self.aero_dimensions[i_surf, 0],
                                                         self.aero_dimensions[i_surf, 1]), 1)
        cout.cout_wrap('  In total: %u bound panels' % sum(self.aero_dimensions[:, 0]*
                                                           self.aero_dimensions[:, 1]), 1)

        # generate placeholder for aero grid zeta coordinates
        self.zeta = []
        for i_surf in range(self.n_surf):
            self.zeta.append(np.zeros((3,
                                       self.aero_dimensions[i_surf, 0] + 1,
                                       self.aero_dimensions[i_surf, 1] + 1),
                                      dtype=ct.c_double))

        # panel normals
        self.normals = []
        for i_surf in range(self.n_surf):
            self.normals.append(np.zeros((3,
                                       self.aero_dimensions[i_surf, 0],
                                       self.aero_dimensions[i_surf, 1]),
                                      dtype=ct.c_double))
        # generate placeholder for aero grid zeta_star coordinates
        self.zeta_star = []
        for i_surf in range(self.n_surf):
            self.zeta_star.append(np.zeros((3,
                                            self.aero_dimensions[i_surf, 0] + 1,
                                            self.aero_dimensions[i_surf, 1] + 1),
                                           dtype=ct.c_double))

        # placeholder for external velocity (includes gust)
        self.u_ext = []
        for i_surf in range(self.n_surf):
            self.u_ext.append(np.zeros((3,
                                        self.aero_dimensions[i_surf, 0] + 1,
                                        self.aero_dimensions[i_surf, 1] + 1),
                                       dtype=ct.c_double))

        # airfoils db
        self.airfoil_db = {}
        for i_node in range(self.total_nodes):
            try:
                self.airfoil_db[self.aero_dict['airfoil_distribution'][i_node]]
            except KeyError:
                airfoil_coords = self.aero_dict['airfoils'][str(self.aero_dict['airfoil_distribution'][i_node])]
                self.airfoil_db[self.aero_dict['airfoil_distribution'][i_node]] = (
                    scipy.interpolate.interp1d(airfoil_coords[:, 0],
                                               airfoil_coords[:, 1],
                                               kind='quadratic',
                                               copy=False,
                                               assume_sorted=True))

        # info from aero.h5 file and mapping with beam elements
        surface_counter = np.zeros((self.n_surf,), dtype=int) - 1
        for i_node in range(self.total_nodes):
            if not self.aero_dict['aero_node'][i_node]: continue

            i_surf = self.aero_dict['surface_distribution'][i_node]
            surface_counter[i_surf] += 1
            node_info = dict()
            node_info['i_node'] = i_node
            node_info['i_surf'] = i_surf
            node_info['chord'] = self.aero_dict['chord'][i_node]
            node_info['eaxis'] = self.aero_dict['elastic_axis'][i_node]
            node_info['twist'] = self.aero_dict['twist'][i_node]
            node_info['M'] = self.aero_dimensions[i_surf, 0]
            node_info['M_distribution'] = self.aero_dict['m_distribution'].decode('ascii')
            node_info['airfoil'] = self.aero_dict['airfoil_distribution'][i_node]
            i_beam_elem = beam.node_master_elem[i_node, 0]
            i_beam_local_node = beam.node_master_elem[i_node, 1]
            node_info['beam_coord'] = beam.pos_ini[i_node, :]
            node_info['beam_psi'] = beam.psi_ini[i_beam_elem, i_beam_local_node, :]

            self.zeta[i_surf][:, :, surface_counter[i_surf]] = generate_strip(node_info, self.airfoil_db)


def generate_strip(node_info, airfoil_db):
    strip_coordinates_a_frame = np.zeros((3, node_info['M'] + 1), dtype=ct.c_double)
    strip_coordinates_b_frame = np.zeros_like(strip_coordinates_a_frame, dtype=ct.c_double)

    # airfoil coordinates
    if node_info['M_distribution'] == 'uniform':
        strip_coordinates_b_frame[1, :] = np.linspace(0.0, 1.0, node_info['M'] + 1)
    else:
        raise NotImplemented('M_distribution is ' + node_info['M_distribution'] +
                             ' and it is not yet supported')
    strip_coordinates_b_frame[2, :] = airfoil_db[node_info['airfoil']](
                                            strip_coordinates_b_frame[1, :])

    # elastic axis correction
    strip_coordinates_b_frame[1, :] -= node_info['eaxis']
    strip_coordinates_b_frame[1, :] *= -1
    # chord
    strip_coordinates_b_frame *= node_info['chord']

    # twist rotation
    if not node_info['twist'] == 0:
        twist_mat = algebra.rotation3d_x(node_info['twist'])
        for i_m in range(node_info['M'] + 1):
            strip_coordinates_b_frame[:, i_m] = np.dot(twist_mat,
                                                       strip_coordinates_b_frame[:, i_m])

    # CRV to rotation matrix
    crv = node_info['beam_psi']
    rotation_mat = algebra.crv2rot(crv)
    for i_m in range(node_info['M'] + 1):
        strip_coordinates_a_frame[:, i_m] = np.dot(rotation_mat,
                                                   strip_coordinates_b_frame[:, i_m])

    # node coordinates addition
    for i_m in range(node_info['M'] + 1):
        strip_coordinates_a_frame[:, i_m] += node_info['beam_coord']

    return strip_coordinates_a_frame









