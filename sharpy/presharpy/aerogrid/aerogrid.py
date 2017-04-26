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


class TimeStepInfo(object):
    def __init__(self, dimensions, dimensions_star, i_ts=0, t=0.0):
        self.dimensions = dimensions
        self.dimensions_star = dimensions_star
        self.n_surf = dimensions.shape[0]
        self.i_ts = i_ts
        self.t = t
        # generate placeholder for aero grid zeta coordinates
        self.zeta = []
        for i_surf in range(self.n_surf):
            self.zeta.append(np.zeros((3,
                                       dimensions[i_surf, 0] + 1,
                                       dimensions[i_surf, 1] + 1),
                                      dtype=ct.c_double))

        # panel normals
        self.normals = []
        for i_surf in range(self.n_surf):
            self.normals.append(np.zeros((3,
                                          dimensions[i_surf, 0],
                                          dimensions[i_surf, 1]),
                                         dtype=ct.c_double))

        # panel normals
        self.forces= []
        for i_surf in range(self.n_surf):
            self.forces.append(np.zeros((3,
                                         dimensions[i_surf, 0],
                                         dimensions[i_surf, 1]),
                                        dtype=ct.c_double))

        # generate placeholder for aero grid zeta_star coordinates
        self.zeta_star = []
        for i_surf in range(self.n_surf):
            self.zeta_star.append(np.zeros((3,
                                            dimensions_star[i_surf, 0] + 1,
                                            dimensions_star[i_surf, 1] + 1),
                                           dtype=ct.c_double))

        # placeholder for external velocity
        self.u_ext = []
        for i_surf in range(self.n_surf):
            self.u_ext.append(np.zeros((3,
                                        dimensions[i_surf, 0] + 1,
                                        dimensions[i_surf, 1] + 1),
                                       dtype=ct.c_double))

        # allocate gamma and gamma star matrices
        self.gamma = []
        for i_surf in range(self.n_surf):
            self.gamma.append(np.zeros((dimensions[i_surf, 0],
                                        dimensions[i_surf, 1]),
                                       dtype=ct.c_double))

        self.gamma_star = []
        for i_surf in range(self.n_surf):
            self.gamma_star.append(np.zeros((dimensions_star[i_surf, 0],
                                             dimensions_star[i_surf, 1]),
                                            dtype=ct.c_double))

    def generate_ctypes_pointers(self):
        self.ct_dimensions = self.dimensions.astype(dtype=ct.c_uint)
        self.ct_dimensions_star = self.dimensions_star.astype(dtype=ct.c_uint)

        from sharpy.utils.constants import NDIM

        self.ct_zeta_list = []
        for i_surf in range(self.n_surf):
            for i_dim in range(NDIM):
                self.ct_zeta_list.append(self.zeta[i_surf][i_dim, :, :].reshape(-1))

        self.ct_zeta_star_list = []
        for i_surf in range(self.n_surf):
            for i_dim in range(NDIM):
                self.ct_zeta_star_list.append(self.zeta_star[i_surf][i_dim, :, :].reshape(-1))

        self.ct_u_ext_list = []
        for i_surf in range(self.n_surf):
            for i_dim in range(NDIM):
                self.ct_u_ext_list.append(self.u_ext[i_surf][i_dim, :, :].reshape(-1))

        self.ct_gamma_list = []
        for i_surf in range(self.n_surf):
            self.ct_gamma_list.append(self.gamma[i_surf][:, :].reshape(-1))

        self.ct_gamma_star_list = []
        for i_surf in range(self.n_surf):
            self.ct_gamma_star_list.append(self.gamma_star[i_surf][:, :].reshape(-1))

        self.ct_normals_list = []
        for i_surf in range(self.n_surf):
            for i_dim in range(NDIM):
                self.ct_normals_list.append(self.normals[i_surf][i_dim, :, :].reshape(-1))

        self.ct_forces_list = []
        for i_surf in range(self.n_surf):
            for i_dim in range(NDIM):
                self.ct_forces_list.append(self.forces[i_surf][i_dim, :, :].reshape(-1))

        self.ct_p_dimensions = (ct.POINTER(ct.c_uint)*2)(* np.ctypeslib.as_ctypes(self.ct_dimensions))
        self.ct_p_dimensions_star = (ct.POINTER(ct.c_uint)*2)(* np.ctypeslib.as_ctypes(self.ct_dimensions_star))
        self.ct_p_zeta = (ct.POINTER(ct.c_double)*len(self.ct_zeta_list))(* [np.ctypeslib.as_ctypes(array) for array in self.ct_zeta_list])
        self.ct_p_zeta_star = (ct.POINTER(ct.c_double)*len(self.ct_zeta_star_list))(* [np.ctypeslib.as_ctypes(array) for array in self.ct_zeta_star_list])
        self.ct_p_u_ext = (ct.POINTER(ct.c_double)*len(self.ct_u_ext_list))(* [np.ctypeslib.as_ctypes(array) for array in self.ct_u_ext_list])
        self.ct_p_gamma = (ct.POINTER(ct.c_double)*len(self.ct_gamma_list))(* [np.ctypeslib.as_ctypes(array) for array in self.ct_gamma_list])
        self.ct_p_gamma_star = (ct.POINTER(ct.c_double)*len(self.ct_gamma_star_list))(* [np.ctypeslib.as_ctypes(array) for array in self.ct_gamma_star_list])
        self.ct_p_normals = (ct.POINTER(ct.c_double)*len(self.ct_normals_list))(* [np.ctypeslib.as_ctypes(array) for array in self.ct_normals_list])
        self.ct_p_forces = (ct.POINTER(ct.c_double)*len(self.ct_forces_list))(* [np.ctypeslib.as_ctypes(array) for array in self.ct_forces_list])

    def remove_ctypes_pointers(self):
        del self.ct_p_zeta, self.ct_zeta_list
        del self.ct_p_zeta_star, self.ct_zeta_star_list
        del self.ct_p_u_ext, self.ct_u_ext_list
        del self.ct_p_gamma, self.ct_gamma_list
        del self.ct_p_gamma_star, self.ct_gamma_star_list
        del self.ct_p_normals, self.ct_normals_list
        del self.ct_p_dimensions


class AeroGrid(object):
    def __init__(self, beam, aero_dict, aero_settings, ts=0, t=0.0):
        self.aero_dict = aero_dict
        self.ts = ts
        self.t = t

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

        self.aero_dimensions_star = self.aero_dimensions.copy()
        self.aero_dimensions_star[:, 0] = aero_settings['mstar']

        cout.cout_wrap('The aerodynamic grid contains %u surfaces' % self.n_surf, 1)
        for i_surf in range(self.n_surf):
            cout.cout_wrap('  Surface %u, M=%u, N=%u' % (i_surf,
                                                         self.aero_dimensions[i_surf, 0],
                                                         self.aero_dimensions[i_surf, 1]), 1)
        cout.cout_wrap('  In total: %u bound panels' % sum(self.aero_dimensions[:, 0]*
                                                           self.aero_dimensions[:, 1]), 1)

        self.timestep_info = []
        self.timestep_info.append(TimeStepInfo(self.aero_dimensions,
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

            self.timestep_info[self.ts].zeta[i_surf][:, :, surface_counter[i_surf]] = (
                generate_strip(node_info,
                               self.airfoil_db,
                               aero_settings['aligned_grid']))

        self.generate_mapping()

    def generate_mapping(self):
        self.struct2aero_mapping = []
        max_i_N = np.zeros((self.n_surf,), dtype=int)
        surf_n_counter = np.zeros((self.n_surf,), dtype=int)
        n_counter = 0
        for i_node in range(self.total_nodes):
            if self.aero_dict['aero_node'][i_node]:
                self.struct2aero_mapping.append({'i_surf': self.aero_dict['surface_distribution'][i_node],
                                                 'i_N': surf_n_counter[self.aero_dict['surface_distribution'][i_node]]})
                surf_n_counter[self.aero_dict['surface_distribution'][i_node]] += 1
                # if n_counter > max_i_N[self.aero_dict['surface_distribution'][i_node]]:
                #     max_i_N[self.aero_dict['surface_distribution'][i_node]] += 1
                # n_counter += 1
            else:
                self.struct2aero_mapping.append({})

        self.aero2struct_mapping = []
        for i_surf in range(self.n_surf):
            self.aero2struct_mapping.append([-1]*(surf_n_counter[i_surf]))

        for i_node in range(self.total_nodes):
            try:
                i_surf = self.struct2aero_mapping[i_node]['i_surf']
                i_N = self.struct2aero_mapping[i_node]['i_N']
            except KeyError:
                continue

            print(i_N)
            self.aero2struct_mapping[i_surf][i_N] = i_node



def generate_strip(node_info, airfoil_db, aligned_grid=True, orientation_in=np.array([1, 0, 0])):
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

    # aligned grid correction
    crv = node_info['beam_psi']
    rotation_mat = algebra.crv2rot(crv)
    if aligned_grid:
        orientation = orientation_in.copy()
        orientation = orientation/np.linalg.norm(orientation)

        local_orientation = np.dot(algebra.rotation3d_z(90*np.pi/180.0), orientation)

        # rotation wrt local z:
        # angle of chord line before correction
        old_x = orientation.copy()
        new_x = np.dot(rotation_mat.T, local_orientation)

        old_x[-1] = 0
        new_x[-1] = 0
        angle = algebra.angle_between_vectors(old_x, new_x)

        z_rotation_mat = algebra.rotation3d_z(angle)
        for i_m in range(node_info['M'] + 1):
            strip_coordinates_b_frame[:, i_m] = np.dot(z_rotation_mat,
                                                       strip_coordinates_b_frame[:, i_m])

    # twist rotation
    if not node_info['twist'] == 0:
        twist_mat = algebra.rotation3d_x(node_info['twist'])
        for i_m in range(node_info['M'] + 1):
            strip_coordinates_b_frame[:, i_m] = np.dot(twist_mat,
                                                       strip_coordinates_b_frame[:, i_m])

    # CRV to rotation matrix
    for i_m in range(node_info['M'] + 1):
        strip_coordinates_a_frame[:, i_m] = np.dot(rotation_mat,
                                                   strip_coordinates_b_frame[:, i_m])

    # node coordinates addition
    for i_m in range(node_info['M'] + 1):
        strip_coordinates_a_frame[:, i_m] += node_info['beam_coord']

    return strip_coordinates_a_frame


