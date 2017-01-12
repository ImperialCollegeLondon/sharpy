import numpy as np
import ctypes as ct
import copy

import presharpy.beam.beamstructures as beamstructures
import presharpy.utils.algebra as algebra


class Beam(object):
    def __init__(self, fem_dictionary):
        # read and store data
        # type of node
        self.num_node_elem = fem_dictionary['num_node_elem']
        # node coordinates
        self.num_node = fem_dictionary['num_node']
        self.pos_ini = fem_dictionary['coordinates']
        self.pos_def = self.pos_ini.copy()
        # element connectivity
        self.num_elem = fem_dictionary['num_elem']
        self.connectivities = fem_dictionary['connectivities']
        # stiffness index of the elems
        self.elem_stiffness = fem_dictionary['elem_stiffness']
        # mass per unit length of elems
        self.elem_mass = fem_dictionary['elem_mass']
        # frame of reference delta
        self.frame_of_reference_delta = fem_dictionary['frame_of_reference_delta']
        # structural twist
        self.structural_twist = fem_dictionary['structural_twist']
        # boundary conditions
        self.boundary_conditions = fem_dictionary['boundary_conditions']
        # beam number for every elem
        try:
            self.beam_number = fem_dictionary['beam_number']
        except KeyError:
            self.beam_number = np.zeros((self.num_elem, ), dtype=int)

        # applied forces
        try:
            self.app_forces = fem_dictionary['app_forces']
            self.node_app_forces = fem_dictionary['node_app_forces']
        except KeyError:
            print('*** No applied forces indicated (or in a wrong format)')
            self.app_forces = None
            self.node_app_forces = None

        # lumped masses
        try:
            self.lumped_mass = fem_dictionary['lumped_mass']
        except KeyError:
            self.lumped_mass = None
        else:
            self.lumped_mass_nodes = fem_dictionary['lumped_mass_nodes']
            self.lumped_mass_inertia = fem_dictionary['lumped_mass_inertia']
            self.lumped_mass_position = fem_dictionary['lumped_mass_position']
            self.n_lumped_mass, _ = self.lumped_mass_position.shape

        # now, we are going to import the mass and stiffness
        # databases
        self.mass_db = fem_dictionary['mass_db']
        (self.n_mass, _, _) = self.mass_db.shape
        self.stiffness_db = fem_dictionary['stiffness_db']
        (self.n_stiff, _, _) = self.stiffness_db.shape
        self.inv_stiffness_db = np.zeros_like(self.stiffness_db, dtype=ct.c_double, order='F')
        for i in range(self.n_stiff):
            self.inv_stiffness_db[i, :, :] = np.linalg.inv(self.stiffness_db[i, :, :])

        # generate the Element array
        self.elements = []
        for ielem in range(self.num_elem):
            self.elements.append(
                beamstructures.Element(
                       ielem,
                       self.num_node_elem,
                       self.connectivities[ielem, :],
                       self.pos_ini[self.connectivities[ielem, :], :],
                       self.frame_of_reference_delta[self.connectivities[ielem, :], :],
                       self.structural_twist[self.connectivities[ielem, :]],
                       self.beam_number[ielem],
                       self.elem_stiffness[ielem],
                       self.elem_mass[ielem]))

        # now we need to add the attributes like mass and stiffness index
        for ielem in range(self.num_elem):
            dictionary = dict()
            dictionary['stiffness_index'] = self.elem_stiffness[ielem]
            dictionary['mass_index'] = self.elem_mass[ielem]
            self.elements[ielem].add_attributes(dictionary)

        # master-slave structure
        self.generate_master_structure()

        # lumped masses to element mass
        if self.lumped_mass is not None:
            self.lump_masses()

        # psi calculation
        self.generate_psi()

    def generate_dof_arrays(self, indexing='C'):
        self.vdof = np.zeros((self.num_node,), dtype=ct.c_int, order='F') - 1
        self.fdof = np.zeros((self.num_node,), dtype=ct.c_int, order='F') - 1

        vcounter = -1
        fcounter = -1
        for inode in range(self.num_node):
            if self.boundary_conditions[inode] == 0:
                vcounter += 1
                fcounter += 1
                self.vdof[inode] = vcounter
                self.fdof[inode] = fcounter
            elif self.boundary_conditions[inode] == -1:
                vcounter += 1
                self.vdof[inode] = vcounter
            elif self.boundary_conditions[inode] == 1:
                fcounter += 1
                self.fdof[inode] = fcounter

        self.num_dof = ct.c_int(vcounter*6)

        if indexing == 'F':
            self.vdof += 1
            self.fdof += 1

    def lump_masses(self):
        for i_lumped in range(self.n_lumped_mass):
            r = self.lumped_mass_position[i_lumped, :]
            m = self.lumped_mass[i_lumped]
            j = self.lumped_mass_inertia[i_lumped, :, :]

            i_lumped_node = self.lumped_mass_nodes[i_lumped]
            i_lumped_master_elem, i_lumped_master_node_local = self.node_master_elem[i_lumped_node]

            inertia_tensor = np.zeros((6, 6))
            r_skew = algebra.rot_skew(r)
            inertia_tensor[0:3, 0:3] = m*np.eye(3)
            inertia_tensor[0:3, 3:6] = m*np.transpose(r_skew)
            inertia_tensor[3:6, 0:3] = m*r_skew
            inertia_tensor[3:6, 3:6] = j + m*(np.dot(np.transpose(r_skew), r_skew))

            if self.elements[i_lumped_master_elem].RBMass is None:
                # allocate memory
                self.elements[i_lumped_master_elem].RBMass = np.zeros((
                    self.elements[i_lumped_master_elem].max_nodes_elem, 6, 6))

            self.elements[i_lumped_master_elem].RBMass[i_lumped_master_node_local, :, :] += (
                inertia_tensor)


    def generate_master_structure(self):
        for elem in self.elements:
            elem.master = np.zeros((elem.n_nodes, 2), dtype=ct.c_int, order='F') - 1
            ielem = elem.ielem
            for inode_local in range(elem.n_nodes - 1, -1, -1):
                inode_global = self.connectivities[ielem, inode_local]

                if inode_global == 0 and ielem == 0:
                    # this is the master node in the master elem
                    # has to stay [-1, -1]
                    continue

                found_previous = False
                for i_prev_elem in range(0, ielem):
                    for i_prev_node in range(self.elements[i_prev_elem].n_nodes):
                        if found_previous:
                            continue
                        i_prev_node_global = self.connectivities[i_prev_elem, i_prev_node]
                        if inode_global == i_prev_node_global:
                            # found node in previous elements in list
                            # the master is the first element to own the node
                            # if elem.master[inode_local, 1] == -1:
                            elem.master[inode_local, :] = [i_prev_elem, i_prev_node]
                            found_previous = True
                            continue
                if not found_previous:
                    # next case: nodes belonging to their element only
                    elem.master[inode_local, :] = [ielem, inode_local - 1]

        self.generate_node_master_elem()

    def generate_node_master_elem(self):
        """
        Returns a matrix indicating the master element for a given node
        :return:
        """
        self.node_master_elem = np.zeros((self.num_node, 2), dtype=ct.c_int, order='F') - 1
        # for ielem in range(self.num_elem):
        #     for inode in range(self.num_node_elem):
        #         inode_global = self.connectivities[ielem, inode]
        #
        #         if self.node_master_elem[inode_global, 0] == -1:
        #             self.node_master_elem[inode_global, :] = self.elements[ielem].master[inode, :]
        for ielem in range(self.num_elem):
            elem = self.elements[ielem]
            for inode in range(elem.n_nodes):
                inode_global = self.connectivities[ielem, inode]
                if self.node_master_elem[inode_global, 0] == -1:
                    self.node_master_elem[inode_global, 0] = ielem
                    self.node_master_elem[inode_global, 1] = inode

    def generate_aux_information(self):

        self.num_nodes_matrix = np.zeros((self.num_elem,), dtype=ct.c_int, order='F')
        for elem in self.elements:
            self.num_nodes_matrix[elem.ielem] = elem.n_nodes

        self.num_mem_matrix = np.zeros_like(self.num_nodes_matrix, dtype=ct.c_int)
        for elem in self.elements:
            self.num_mem_matrix[elem.ielem] = elem.num_mem

        self.connectivities_fortran = self.connectivities.astype(ct.c_int, order='F') + 1

        # correction of the indices
        self.master_nodes = np.zeros((self.num_elem, self.num_node_elem, 2), dtype=ct.c_int, order='F')
        for elem in self.elements:
            ielem = elem.ielem
            self.master_nodes[ielem, :, :] = elem.master + 1

        # ADC: test CAREFUL
        for i in range(1,3):
            self.master_nodes[:, i, :] = 0

        self.node_master_elem_fortran = self.node_master_elem.astype(dtype=ct.c_int, order='F') + 1

        self.length_matrix = np.zeros_like(self.num_nodes_matrix)
        for elem in self.elements:
            self.length_matrix[elem.ielem] = elem.length

        self.mass_matrix = self.mass_db.astype(ct.c_double, order='F')
        self.stiffness_matrix = self.stiffness_db.astype(ct.c_double, order='F')
        self.mass_indices = self.elem_mass.astype(ct.c_int, order='F') + 1
        self.stiffness_indices = self.elem_stiffness.astype(ct.c_int, order='F') + 1

        self.frame_of_reference_delta = self.frame_of_reference_delta.astype(ct.c_double, order='F')

        # Vdof and Fdof vector calculation
        self.generate_dof_arrays('F')

        self.n_app_forces, _ = self.app_forces.shape
        self.n_app_forces = ct.c_int(self.n_app_forces)
        self.app_forces_fortran = self.app_forces.astype(dtype=ct.c_double, order='F')
        self.node_app_forces_fortran = self.node_app_forces.astype(dtype=ct.c_int, order='F')

        # Psi matrix
        self.psi_def = self.psi_ini.astype(dtype=ct.c_double, order='F')

        # deformed structure matrices
        self.pos_ini = self.pos_ini.astype(dtype=ct.c_double, order='F')
        self.pos_def = self.pos_ini.astype(dtype=ct.c_double, order='F')

        # lumped masses to new organisation
        # self.n_lumped_mass_fortran = 0  # will need to convert that to c_int
        #                                 # !!! not the same value as n_lumped_mass
        #                                 # this one counts the number of nodes with applied RBMass
        # lumped_mass_elem = []
        # lumped_mass_local_node = []
        # for elem in self.elements:
        #     if elem.RBMass is None:
        #         continue
        #     for inode in range(elem.max_nodes_elem):
        #         if np.linalg.norm(elem.RBMass[inode, :, :]) < 1e-6:
        #             continue
        #         self.n_lumped_mass_fortran += 1
        #         lumped_mass_elem.append(elem.ielem)
        #         lumped_mass_local_node.append(inode)
        #
        # # we know the nodes which have RBMass, now we copy it into a temp matrix
        # rbmass_temp = np.zeros((self.n_lumped_mass_fortran, 6, 6))
        # for i_lumped_mass in range(self.n_lumped_mass_fortran):
        #     rbmass_temp[i_lumped_mass, :, :] = (self.elements[lumped_mass_elem[i_lumped_mass]].
        #                                         RBMass[lumped_mass_local_node[i_lumped_mass], :, :])
        #
        # # conversion to ctypes matrices
        # self.lumped_mass_elem_fortran = np.array(lumped_mass_elem, dtype=ct.c_int, order='F')
        # self.lumped_mass_local_node_fortran = np.array(lumped_mass_local_node, dtype=ct.c_int, order='F')
        # self.lumped_mass_fortran = np.array(rbmass_temp, dtype=ct.c_double, order='F')
        # self.n_lumped_mass_fortran = ct.c_int(self.n_lumped_mass_fortran)
        max_nodes_elem = self.elements[0].max_nodes_elem
        rbmass_temp = np.zeros((self.num_elem, max_nodes_elem, 6, 6))
        for elem in self.elements:
            for inode in range(elem.n_nodes):
                if elem.RBMass is not None:
                    rbmass_temp[elem.ielem, inode, :, :] = elem.RBMass[inode, :, :]

        self.rbmass_fortran = rbmass_temp.astype(dtype=ct.c_double, order='F')

    def generate_psi(self):
        #     # it will just generate the CRV for all the nodes of the element
        self.psi_ini = np.zeros((self.num_elem, 3, 3), dtype=ct.c_double, order='F')
        for elem in self.elements:
            self.psi_ini[elem.ielem, :, :] = elem.psi_ini

    def update(self):
        for elem in self.elements:
            elem.update(self.pos_def[self.connectivities[elem.ielem, :], :],
                        self.psi_def[elem.ielem, :, :])

    def plot(self, fig=None, ax=None, plot_triad=True, defor=False, ini=True):
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
        if ini:
            for elem in self.elements:
                elem.plot(fig, ax, plot_triad=plot_triad, defor=False)
            # nodes
            nodes = ax.scatter(self.pos_ini[:, 0],
                               self.pos_ini[:, 1],
                               self.pos_ini[:, 2], color='k')
        if defor:
            for elem in self.elements:
                elem.plot(fig, ax, plot_triad=plot_triad, defor=True)
            # nodes
            nodes = ax.scatter(self.pos_def[:, 0],
                               self.pos_def[:, 1],
                               self.pos_def[:, 2], color='b')

        plt.hold('off')



























