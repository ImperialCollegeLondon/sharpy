import numpy as np
import ctypes as ct

import presharpy.beam.beamstructures as beamstructures


class Beam(object):
    def __init__(self, fem_dictionary):
        # read and store data
        # type of node
        self.num_node_elem = fem_dictionary['num_node_elem']
        # node coordinates
        self.num_node = fem_dictionary['num_node']
        self.node_coordinates = fem_dictionary['coordinates']
        self.pos = self.node_coordinates.copy()
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
        self.beam_number = fem_dictionary['beam_number']

        # now, we are going to import the mass and stiffness
        # databases
        self.mass_db = fem_dictionary['mass_db']
        self.stiffness_db = fem_dictionary['stiffness_db']

        # generate the Element array
        self.elements = []
        for ielem in range(self.num_elem):
            self.elements.append(
                beamstructures.Element(
                       ielem,
                       self.num_node_elem,
                       self.connectivities[ielem, :],
                       self.node_coordinates[self.connectivities[ielem, :], :],
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

        # number of degrees of freedom calculation
        self.num_dof = ct.c_int(6*sum(self.boundary_conditions < 1))

    def generate_master_structure(self):
        '''
        Master-slave relationships are necessary for
        later stages, as nodes belonging to two different
        elements have two different values of their rotation.
        '''
        # let's just keep the outer nodes of the element
        temp_connectivities = np.zeros((self.num_elem, 2),
                                       dtype=int)
        temp_connectivities[:, 0] = self.connectivities[:, 0]
        temp_connectivities[:, -1] = self.connectivities[:, -1]

        # master_elems contains the index of the master
        # element for every element
        # master_nodes contains the index of the master
        # node belongin to the master element
        # the numbering of the nodes is based on the
        # local one (0, 1 or 2) for a 3-noded element
        self.master_elems = np.zeros(self.num_elem,
                dtype=int) - 1
        self.master_nodes = np.zeros_like(self.master_elems,
                dtype=int) - 1
        for ielem in range(self.num_elem):
            # import pdb; pdb.set_trace()
            if ielem == 0:
                continue

            temp = temp_connectivities[0:ielem,:]
            elem_nodes = temp_connectivities[ielem,:]
            # case: global master elem
            # (none of the extreme nodes appear in previous
            #  connectivities)
            if not (elem_nodes[0] in temp or
                    elem_nodes[1] in temp):
                continue

            # nodes in elem ielem
            for inode in range(1, -1, -1):
                # previous elements in the list
                for iielem in range(ielem):
                    # nodes of the previous elements in the list
                    for iinode in range(1, -1, -1):
                        # connectivity already found
                        if not self.master_elems[ielem] == -1:
                            continue
                        if elem_nodes[inode] == temp_connectivities[iielem, iinode]:
                            # found a repeated connectivity
                            self.master_elems[ielem] = iielem
                            if iinode == 0:
                                self.master_nodes[ielem] = iinode
                            elif iinode == 1:
                                self.master_nodes[ielem] = self.num_node_elem

        self.generate_node_master_elem()

    def generate_node_master_elem(self):
        """
        Returns a matrix indicating the master element for a given node
        :return:
        """
        self.node_master_elem = np.zeros((self.num_node, 2), dtype=int)
        for ielem in range(self.num_elem):
            for inode in range(self.num_node_elem):
                iinode = self.connectivities[ielem, inode]
                if self.node_master_elem[iinode, 0] == 0:
                    self.node_master_elem[iinode, 0] = ielem
                    self.node_master_elem[iinode, 1] = inode

    def generate_aux_information(self):
        self.num_nodes_matrix = np.ones((self.num_elem,), dtype=int)

        self.num_mem_matrix = np.zeros_like(self.num_nodes_matrix, dtype=int)
        for elem in self.elements:
            self.num_mem_matrix[elem.ielem] = elem.num_mem

        # correction of the indices
        self.master_nodes_fortran = self.master_nodes + 1
        self.master_nodes_fortran.flatten('F')

        self.length_matrix = np.zeros_like(self.num_nodes_matrix)
        for elem in self.elements:
            self.length_matrix[elem.ielem] = elem.length

        self.precurv = np.zeros((self.num_elem, 3))
        self.precurv = self.precurv.flatten('F')

        self.psi = np.zeros((self.num_elem, 3))
        self.psi = self.psi.flatten('F')

        self.local_vec = np.zeros((self.num_elem, 3), order='F')
        for elem in self.elements:
            self.local_vec[elem.ielem, :] = elem.frame_of_reference_delta[0, :]

        self.mass_matrix = np.zeros((self.num_elem*6, 6), order='F')
        self.stiffness_matrix = np.zeros((self.num_elem*6, 6), order='F')
        self.inv_stiffness_matrix = np.zeros((self.num_elem*6, 6), order='F')
        for elem in self.elements:
            self.mass_matrix[6*elem.ielem:6*elem.ielem + 6, :] = self.mass_db[elem.mass_index, :, :]
            self.stiffness_matrix[6*elem.ielem:6*elem.ielem + 6, :] = self.stiffness_db[elem.stiff_index, :, :]
            self.inv_stiffness_matrix[6*elem.ielem:6*elem.ielem + 6, :] = np.linalg.inv(self.stiffness_db[elem.stiff_index, :, :])

        # TODO RBMass support
        self.rbmass_matrix = np.zeros((self.num_elem*
                                       self.num_node_elem*
                                       6*6))

        self.node_master_elem_fortran = self.node_master_elem + 1



    def plot(self, fig=None, ax=None, plot_triad=True):
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
        # nodes
        nodes = ax.scatter(self.node_coordinates[:, 0],
                           self.node_coordinates[:, 1],
                           self.node_coordinates[:, 2])
        for elem in self.elements:
            elem.plot(fig, ax, plot_triad=plot_triad)
        plt.hold('off')



























