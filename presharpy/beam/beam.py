import numpy as np
import ctypes as ct

import presharpy.beam.beamstructures as beamstructures
import presharpy.utils.algebra as algebra


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
        try:
            self.beam_number = fem_dictionary['beam_number']
        except KeyError:
            self.beam_number = np.zeros((self.num_elem, ), dtype=int)

        # applied forces
        try:
            self.app_forces = fem_dictionary['app_forces']
        except KeyError:
            print('*** No applied forces indicated')
            self.app_forces = None

        try:
            self.app_forces_type = fem_dictionary['app_forces_type']
        except KeyError:
            self.app_forces_type = np.zeros((self.num_node, ), dtype=int)

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

        # psi calculation
        self.generate_psi()

    def generate_dof_arrays(self, indexing='C'):
        self.vdof = np.zeros((self.num_node,), dtype=ct.c_int, order='F') - 1
        self.fdof = np.zeros((self.num_node,), dtype=ct.c_int, order='F') - 1

        counter = -1
        vcounter = -1
        fcounter = -1
        for bc in self.boundary_conditions:
            counter += 1
            fcounter += 1
            self.fdof[counter] = fcounter
            if bc == 1:
                pass
            else:
                vcounter += 1
                self.vdof[counter] = vcounter

        if indexing == 'F':
            self.vdof = self.vdof + 1
            self.fdof = self.fdof + 1

    def generate_master_structure(self):
        for elem in self.elements:
            elem.master = np.zeros((elem.n_nodes, 2), dtype=ct.c_int, order='F') - 1
            ielem = elem.ielem
            previous_connectivities = self.connectivities[0:ielem, :]
            for inode_local in range(elem.n_nodes):
                inode_global = self.connectivities[ielem, inode_local]

                if inode_global == 0 and ielem == 0:
                    # this is the master node in the master elem
                    # has to stay [-1, -1]
                    continue

                for i_prev_elem in range(0, ielem):
                    for i_prev_node in range(self.elements[i_prev_elem].n_nodes):
                        i_prev_node_global = self.connectivities[i_prev_elem, i_prev_node]
                        if inode_global == i_prev_node_global:
                            # found node in previous elements in list
                            # the master is the first element to own the node
                            if elem.master[inode_local, 1] == -1:
                                elem.master[inode_local, :] = [i_prev_elem, i_prev_node]
                            continue

                # next case: nodes belonging to their element only
                elem.master[inode_local, :] = [ielem, inode_local - 1]

        self.generate_node_master_elem()
        1

    def generate_node_master_elem(self):
        """
        Returns a matrix indicating the master element for a given node
        :return:
        """
        self.node_master_elem = np.zeros((self.num_node, 2), dtype=ct.c_int, order='F')
        for ielem in range(self.num_elem):
            for inode in range(self.num_node_elem):
                iinode = self.connectivities[ielem, inode]
                if self.node_master_elem[iinode, 0] == 0:
                    self.node_master_elem[iinode, 0] = ielem
                    self.node_master_elem[iinode, 1] = inode

    def generate_aux_information(self):
        self.num_nodes_matrix = np.zeros((self.num_elem,), dtype=int)
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

        self.node_master_elem_fortran = self.node_master_elem.astype(dtype=ct.c_int, order='F') + 1

        self.length_matrix = np.zeros_like(self.num_nodes_matrix)
        for elem in self.elements:
            self.length_matrix[elem.ielem] = elem.length

        # TODO precurv support
        # self.precurv = np.zeros((self.num_elem, 3))
        # self.precurv = self.precurv.flatten('F')
        #
        # self.local_vec = np.zeros((self.num_elem, 3), order='F', dtype=ct.c_double)
        # for elem in self.elements:
        #     self.local_vec[elem.ielem, :] = elem.frame_of_reference_delta[1, :]

        self.mass_matrix = self.mass_db.astype(ct.c_double, order='F')
        self.stiffness_matrix = self.stiffness_db.astype(ct.c_double, order='F')
        self.mass_indices = self.elem_mass.astype(ct.c_int, order='F') + 1
        self.stiffness_indices = self.elem_stiffness.astype(ct.c_int, order='F') + 1

        # TODO RBMass support
        self.rbmass_matrix = np.zeros((self.num_elem,
                                       3,
                                       6,
                                       6), dtype=ct.c_double, order='F')

        # Vdof and Fdof vector calculation
        self.generate_dof_arrays('F')

        self.app_forces_fortran = self.app_forces.astype(dtype=ct.c_double, order='F')

        # Psi matrix
        self.generate_psi()
        self.psi_def = self.psi_ini.copy()

        # deformed structure matrices
        self.node_coordinates = self.node_coordinates.astype(dtype=ct.c_double, order='F')
        self.node_coordinates_defor = self.node_coordinates.astype(dtype=ct.c_double, order='F')

    def generate_psi(self):
        # it will just generate the CRV for all the nodes of the element
        self.psi_ini = np.zeros((self.num_elem, self.num_node_elem, 3), dtype=ct.c_double, order='F')
        for elem in self.elements:
            for inode in range(elem.n_nodes):
                self.psi_ini[elem.ielem, inode, :] = algebra.triad2crv(elem.tangent_vector[inode, :],
                                                                       elem.normal_vector[inode, :],
                                                                       elem.binormal_vector[inode, :])

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



























