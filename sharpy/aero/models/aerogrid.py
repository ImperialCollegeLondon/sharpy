# Alfonso del Carre
# alfonso.del-carre14@imperial.ac.uk
# Imperial College London
# LoCA lab
# 29 Sept 2016

# Aerogrid contains all the necessary routines to generate an aerodynamic
# grid based on the input dictionaries.

import ctypes as ct

import numpy as np
import scipy.interpolate

import sharpy.utils.algebra as algebra
import sharpy.utils.cout_utils as cout
from sharpy.utils.datastructures import AeroTimeStepInfo


class Aerogrid(object):
    def __init__(self):
        self.aero_dict = None
        self.beam = None
        self.aero_settings = None

        self.timestep_info = []
        self.ini_info = None

        self.surface_distribution = None
        self.surface_m = None
        self.aero_dimensions = None
        self.aero_dimensions_star = None
        self.airfoil_db = dict()
        self.struct2aero_mapping = None
        self.aero2struct_mapping = []

        self.n_node = 0
        self.n_elem = 0
        self.n_surf = 0
        self.n_aero_node = 0

    def generate(self, aero_dict, beam, aero_settings):
        self.aero_dict = aero_dict
        self.beam = beam
        self.aero_settings = aero_settings

        # number of total nodes (structural + aero&struc)
        self.n_node = len(aero_dict['aero_node'])
        # number of elements
        self.n_elem = len(aero_dict['surface_distribution'])
        # surface distribution
        self.surface_distribution = aero_dict['surface_distribution']
        # number of surfaces
        temp = set(aero_dict['surface_distribution'])
        self.n_surf = sum(1 for i in temp if i >= 0)
        # number of chordwise panels
        self.surface_m = aero_dict['surface_m']
        # number of aero nodes
        self.n_aero_node = sum(aero_dict['aero_node'])

        # get N per surface
        self.calculate_dimensions()

        # write grid info to screen
        self.output_info()

        # allocating initial grid storage
        self.ini_info = AeroTimeStepInfo(self.aero_dimensions,
                                         self.aero_dimensions_star)

        # load airfoils db
        for i_node in range(self.n_node):
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
        self.add_timestep()
        self.generate_zeta(self.beam, self.aero_settings, ts=0)

    def output_info(self):
        cout.cout_wrap('The aerodynamic grid contains %u surfaces' % self.n_surf, 1)
        for i_surf in range(self.n_surf):
            cout.cout_wrap('  Surface %u, M=%u, N=%u' % (i_surf,
                                                         self.aero_dimensions[i_surf, 0],
                                                         self.aero_dimensions[i_surf, 1]), 1)
            cout.cout_wrap('     Wake %u, M=%u, N=%u' % (i_surf,
                                                         self.aero_dimensions_star[i_surf, 0],
                                                         self.aero_dimensions_star[i_surf, 1]), 1)
        cout.cout_wrap('  In total: %u bound panels' % (sum(self.aero_dimensions[:, 0]*
                                                            self.aero_dimensions[:, 1], 1)))
        cout.cout_wrap('  In total: %u wake panels' % (sum(self.aero_dimensions_star[:, 0]*
                                                            self.aero_dimensions_star[:, 1], 1)))
        cout.cout_wrap('  Total number of panels = %u' % (sum(self.aero_dimensions_star[:, 0]*
                                                              self.aero_dimensions_star[:, 1], 1) +
                                                          sum(self.aero_dimensions_star[:, 0]*
                                                              self.aero_dimensions_star[:, 1], 1)))

    def calculate_dimensions(self):
        self.aero_dimensions = np.zeros((self.n_surf, 2), dtype=int)
        for i in range(self.n_surf):
            # adding M values
            self.aero_dimensions[i, 0] = self.surface_m[i]
        # count N values (actually, the count result
        # will be N+1)
        nodes_in_surface = []
        for i_surf in range(self.n_surf):
            nodes_in_surface.append([])
        for i_elem in range(self.beam.num_elem):
            nodes = self.beam.elements[i_elem].global_connectivities
            i_surf = self.aero_dict['surface_distribution'][i_elem]
            if i_surf < 0:
                continue
            for i_global_node in nodes:
                if i_global_node in nodes_in_surface[i_surf]:
                    continue
                else:
                    nodes_in_surface[i_surf].append(i_global_node)
                if self.aero_dict['aero_node'][i_global_node]:
                    self.aero_dimensions[i_surf, 1] += 1

        # accounting for N+1 nodes -> N panels
        self.aero_dimensions[:, 1] -= 1

        self.aero_dimensions_star = self.aero_dimensions.copy()
        for i_surf in range(self.n_surf):
            self.aero_dimensions_star[i_surf, 0] = self.aero_dict['surface_m'][i_surf]

    def add_timestep(self):
        self.timestep_info.append(AeroTimeStepInfo(self.aero_dimensions,
                                                   self.aero_dimensions_star))
        if len(self.timestep_info) > 1:
            self.timestep_info[-1] = self.timestep_info[-2].copy()

    def generate_zeta(self, beam, aero_settings, ts=0):
        self.generate_mapping()
        nodes_in_surface = []
        for i_surf in range(self.n_surf):
            nodes_in_surface.append([])
        for i_elem in range(self.n_elem):
            for i_local_node in self.beam.elements[i_elem].ordering:
                i_global_node = self.beam.elements[i_elem].global_connectivities[i_local_node]
                if not self.aero_dict['aero_node'][i_global_node]:
                    continue
                for i in range(len(self.struct2aero_mapping[i_global_node])):
                    i_n = self.struct2aero_mapping[i_global_node][i]['i_n']
                    i_surf = self.struct2aero_mapping[i_global_node][i]['i_surf']
                    if i_n in nodes_in_surface[i_surf]:
                        continue
                    else:
                        nodes_in_surface[i_surf].append(i_n)

                    master_elem, master_elem_node = beam.master[i_elem, i_local_node, :]
                    if master_elem < 0:
                        master_elem = i_elem
                        master_elem_node = i_local_node

                    node_info = dict()
                    node_info['i_node'] = i_global_node
                    node_info['i_local_node'] = i_local_node
                    node_info['chord'] = self.aero_dict['chord'][i_global_node]
                    node_info['eaxis'] = self.aero_dict['elastic_axis'][i_global_node]
                    node_info['twist'] = self.aero_dict['twist'][i_global_node]
                    node_info['M'] = self.aero_dimensions[i_surf, 0]
                    node_info['M_distribution'] = self.aero_dict['m_distribution'].decode('ascii')
                    node_info['airfoil'] = self.aero_dict['airfoil_distribution'][i_global_node]
                    node_info['beam_coord'] = beam.timestep_info[beam.it].pos[i_global_node, :]
                    node_info['beam_psi'] = beam.timestep_info[beam.it].psi[master_elem, master_elem_node, :]
                    node_info['for_delta'] = beam.frame_of_reference_delta[master_elem, master_elem_node, :]
                    node_info['elem'] = beam.elements[master_elem]
                    self.timestep_info[ts].zeta[i_surf][:, :, i_n] = (
                        generate_strip(node_info,
                                       self.airfoil_db,
                                       aero_settings['aligned_grid'],
                                       orientation_in=aero_settings['freestream_dir']))

    def generate_mapping(self):
        self.struct2aero_mapping = [[]]*self.n_node
        surf_n_counter = np.zeros((self.n_surf,), dtype=int)
        nodes_in_surface = []
        for i_surf in range(self.n_surf):
            nodes_in_surface.append([])

        for i_elem in range(self.n_elem):
            i_surf = self.aero_dict['surface_distribution'][i_elem]
            if i_surf == -1:
                continue
            for i_global_node in self.beam.elements[i_elem].reordered_global_connectivities:
                if not self.aero_dict['aero_node'][i_global_node]:
                    continue

                if i_global_node in nodes_in_surface[i_surf]:
                    continue
                else:
                    nodes_in_surface[i_surf].append(i_global_node)
                    surf_n_counter[i_surf] += 1
                    try:
                        self.struct2aero_mapping[i_global_node][0]
                    except IndexError:
                        self.struct2aero_mapping[i_global_node] = []

                i_n = surf_n_counter[i_surf] - 1
                self.struct2aero_mapping[i_global_node].append({'i_surf': i_surf,
                                                                'i_n': i_n})

        nodes_in_surface = []
        for i_surf in range(self.n_surf):
            nodes_in_surface.append([])

        for i_surf in range(self.n_surf):
            self.aero2struct_mapping.append([-1]*(surf_n_counter[i_surf]))

        for i_elem in range(self.n_elem):
            for i_global_node in self.beam.elements[i_elem].global_connectivities:
                for i in range(len(self.struct2aero_mapping[i_global_node])):
                    try:
                        i_surf = self.struct2aero_mapping[i_global_node][i]['i_surf']
                        i_n = self.struct2aero_mapping[i_global_node][i]['i_n']
                        if i_global_node in nodes_in_surface[i_surf]:
                            continue
                        else:
                            nodes_in_surface[i_surf].append(i_global_node)
                    except KeyError:
                        continue
                    self.aero2struct_mapping[i_surf][i_n] = i_global_node


def generate_strip(node_info, airfoil_db, aligned_grid, orientation_in=np.array([1, 0, 0])):
    """
    Returns a strip in "a" frame of reference, it has to be then rotated to
    simulate angles of attack, etc
    :param node_info:
    :param airfoil_db:
    :param aligned_grid:
    :param orientation_in:
    :return:
    """
    strip_coordinates_a_frame = np.zeros((3, node_info['M'] + 1), dtype=ct.c_double)
    strip_coordinates_b_frame = np.zeros((3, node_info['M'] + 1), dtype=ct.c_double)

    # airfoil coordinates
    # we are going to store everything in the x-z plane of the b
    # FoR, so that the transformation Cab rotates everything in place.
    if node_info['M_distribution'] == 'uniform':
        strip_coordinates_b_frame[1, :] = np.linspace(0.0, 1.0, node_info['M'] + 1)
    elif node_info['M_distribution'] == '1-cos':
        domain = np.linspace(0, 1.0, node_info['M'] + 1)
        strip_coordinates_b_frame[1, :] = 0.5*(1.0 - np.cos(domain*np.pi))
    else:
        raise NotImplemented('M_distribution is ' + node_info['M_distribution'] +
                             ' and it is not yet supported')
    strip_coordinates_b_frame[2, :] = airfoil_db[node_info['airfoil']](
                                            strip_coordinates_b_frame[1, :])

    # elastic axis correction
    for i_M in range(node_info['M'] + 1):
        strip_coordinates_b_frame[1, i_M] -= node_info['eaxis']

    # chord scaling
    strip_coordinates_b_frame *= node_info['chord']

    # twist transformation (rotation around x_b axis)
    if np.abs(node_info['twist']) > 1e-6:
        Ctwist = algebra.rotation3d_x(node_info['twist'])
    else:
        Ctwist = np.eye(3)

    # Cab transformation
    Cab = algebra.crv2rot(node_info['beam_psi'])

    # sweep angle correction
    # angle between orientation_in and chord line
    chord_line_b_frame = strip_coordinates_b_frame[:, -1] - strip_coordinates_b_frame[:, 0]
    chord_line_a_frame = np.dot(Cab, chord_line_b_frame)
    sweep_angle = algebra.angle_between_vectors_sign(orientation_in, chord_line_a_frame, np.array([0, 0, 1]))
    # rotation matrix
    Csweep = algebra.rotation3d_z(-sweep_angle)

    # transformation from beam to aero
    for i_M in range(node_info['M'] + 1):
        strip_coordinates_a_frame[:, i_M] = np.dot(Cab, np.dot(Csweep, np.dot(Ctwist, strip_coordinates_b_frame[:, i_M])))

    # add node coords
    for i_M in range(node_info['M'] + 1):
        strip_coordinates_a_frame[:, i_M] += node_info['beam_coord']

    return strip_coordinates_a_frame


