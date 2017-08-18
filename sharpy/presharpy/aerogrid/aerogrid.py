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
import sharpy.presharpy.aerogrid.utils as aero_utils

from sharpy.utils.datastructures import AeroTimeStepInfo
from sharpy.presharpy.beam.beamstructures import Element



class AeroGrid(object):
    def __init__(self, beam, aero_dict, aero_settings, inertial2aero=None, quiet=False, ts=0, t=0.0):
        self.aero_dict = aero_dict
        self.ts = ts
        self.t = t
        self.beam = beam
        self.quiet = quiet
        self.inertial2aero = inertial2aero

        # number of total nodes (structural + aero&struc)
        self.total_nodes = len(aero_dict['aero_node'])
        # number of elements
        self.n_elem = len(aero_dict['surface_distribution'])
        # surface distribution
        self.surface_distribution = aero_dict['surface_distribution']
        # number of surfaces
        temp = set(aero_dict['surface_distribution'])
        self.n_surf = sum(1 for i in temp if i >= 0)
        # self.n_surf = len(set(aero_dict['surface_distribution']))
        # number of chordwise panels
        self.surface_m = aero_dict['surface_m']
        # number of aero nodes
        self.n_aero_nodes = sum(aero_dict['aero_node'])

        self.node_master_elem = beam.node_master_elem

        # get N per surface
        self.aero_dimensions = np.zeros((self.n_surf, 2), dtype=int)
        for i in range(self.n_surf):
            # adding M values
            self.aero_dimensions[i, 0] = self.surface_m[i]

        # count N values (actually, the count result
        # will be N+1)
        nodes_in_surface = []
        for i_surf in range(self.n_surf):
            nodes_in_surface.append([])
        for i_elem in range(beam.num_elem):
            nodes = beam.elements[i_elem].global_connectivities
            i_surf = aero_dict['surface_distribution'][i_elem]
            if i_surf < 0:
                continue
            for i_global_node in nodes:
                if i_global_node in nodes_in_surface[i_surf]:
                    continue
                else:
                    nodes_in_surface[i_surf].append(i_global_node)
                if aero_dict['aero_node'][i_global_node]:
                    self.aero_dimensions[i_surf, 1] += 1

        # accounting for N+1 nodes -> N panels
        self.aero_dimensions[:, 1] -= 1

        self.aero_dimensions_star = self.aero_dimensions.copy()
        self.aero_dimensions_star[:, 0] = aero_settings['mstar']

        if not self.quiet:
            cout.cout_wrap('The aerodynamic grid contains %u surfaces' % self.n_surf, 1)
            for i_surf in range(self.n_surf):
                cout.cout_wrap('  Surface %u, M=%u, N=%u' % (i_surf,
                                                             self.aero_dimensions[i_surf, 0],
                                                             self.aero_dimensions[i_surf, 1]), 1)
            cout.cout_wrap('  In total: %u bound panels' % sum(self.aero_dimensions[:, 0]*
                                                               self.aero_dimensions[:, 1]), 1)

        self.timestep_info = []
        self.timestep_info.append(AeroTimeStepInfo(self.aero_dimensions,
                                                   self.aero_dimensions_star,
                                                   self.ts,
                                                   self.t))

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
        self.generate_zeta(beam, aero_settings)

    def generate_zeta(self, beam, aero_settings):
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
                    node_info['beam_coord'] = beam.timestep_info[beam.it].pos_def[i_global_node, :]
                    node_info['beam_psi'] = beam.timestep_info[beam.it].psi_def[master_elem, master_elem_node, :]
                    node_info['for_delta'] = beam.frame_of_reference_delta[master_elem, master_elem_node, :]
                    node_info['elem'] = beam.elements[master_elem]
                    self.timestep_info[self.ts].zeta[i_surf][:, :, i_n] = (
                        generate_strip(node_info,
                                       self.airfoil_db,
                                       aero_settings['aligned_grid'],
                                       inertial2aero=self.inertial2aero))

    def generate_mapping(self):
        self.struct2aero_mapping = [[]]*self.total_nodes
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
                self.struct2aero_mapping[i_global_node].append(({'i_surf': i_surf,
                                                                 'i_n': i_n}))

        self.aero2struct_mapping = []
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

    def initialise_unsteady_wake(self, flightconditions, delta_t):
        ts = self.ts
        orientation = aero_utils.alpha_beta_to_direction(flightconditions['FlightCon']['alpha'],
                                                         flightconditions['FlightCon']['beta'])
        delta_x = flightconditions['FlightCon']['u_inf']*delta_t
        delta = algebra.unit_vector(orientation)*delta_x
        for i_surf in range(self.n_surf):
            for i_N in range(self.aero_dimensions[i_surf, 1]):
                base_pos = self.timestep_info[ts].zeta[i_surf][:, -1, i_N]
                for i_M in range(self.aero_dimensions_star[i_surf, 0]):
                    self.timestep_info[ts].zeta_star[i_surf][:, i_M, i_N] = base_pos + i_M*delta

    def initialise_steady_wake(self, flightconditions):
        ts = 0
        orientation = aero_utils.alpha_beta_to_direction(flightconditions['FlightCon']['alpha'],
                                                         flightconditions['FlightCon']['beta'])
        delta_x = 1.0
        delta = algebra.unit_vector(orientation)*delta_x
        for i_surf in range(self.n_surf):
            for i_N in range(self.aero_dimensions[i_surf, 1]):
                base_pos = self.timestep_info[ts].zeta[i_surf][:, -1, i_N]
                for i_M in range(self.aero_dimensions_star[i_surf, 0]):
                    self.timestep_info[ts].zeta_star[i_surf][:, i_M, i_N] = base_pos + i_M*delta


def generate_strip(node_info, airfoil_db, aligned_grid=True, inertial2aero=None, orientation_in=np.array([1, 0, 0])):
    strip_coordinates_a_frame = np.zeros((3, node_info['M'] + 1), dtype=ct.c_double)
    strip_coordinates_b_frame = np.zeros_like(strip_coordinates_a_frame, dtype=ct.c_double)

    # airfoil coordinates
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
    strip_coordinates_b_frame[1, :] -= node_info['eaxis']
    # chord
    strip_coordinates_b_frame *= node_info['chord']

    psi = node_info['beam_psi']
    Cab = algebra.crv2rot(psi)
    if aligned_grid:
        axis = np.dot(Cab, np.array([0, 1, 0]))
        angle = algebra.angle_between_vectors_sign(axis, orientation_in)
    else:
        angle = 0
    # Cab = np.dot(algebra.rotation3d_z(angle), Cab)

    # used to work except for sweep
    delta = node_info['for_delta']
    psi = node_info['beam_psi']
    Cab = algebra.crv2rot(psi)

    angle = algebra.angle_between_vectors(delta, orientation_in)
    Cab = np.dot(Cab, algebra.rotation3d_z(-angle))

    #TODO delta z on definiton of elastic axis
    # twist rotation
    if not node_info['twist'] == 0:
        twist_mat = algebra.rotation3d_x(node_info['twist'])
        for i_m in range(node_info['M'] + 1):
            strip_coordinates_b_frame[:, i_m] = np.dot(twist_mat,
                                                       strip_coordinates_b_frame[:, i_m])

    # CRV to rotation matrix
    for i_m in range(node_info['M'] + 1):
        strip_coordinates_a_frame[:, i_m] = np.dot(Cab,
                                                   strip_coordinates_b_frame[:, i_m])

    # node coordinates addition
    for i_m in range(node_info['M'] + 1):
        strip_coordinates_a_frame[:, i_m] += node_info['beam_coord']

    # inertial2aero application
    # strip_coordinates_a_frame is now in inertial FoR
    if inertial2aero is not None:
        for i_m in range(node_info['M'] + 1):
            strip_coordinates_a_frame[:, i_m] = np.dot(inertial2aero.T,
                                                       strip_coordinates_a_frame[:, i_m])

    return strip_coordinates_a_frame



