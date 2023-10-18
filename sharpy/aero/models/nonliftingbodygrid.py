"""Nonlifting Body grid

Description
"""

from sharpy.aero.models.grid import Grid
from sharpy.utils.datastructures import NonliftingBodyTimeStepInfo
import numpy as np
import sharpy.utils.algebra as algebra


class NonliftingBodyGrid(Grid):
    """
    ``Nonlifting Body Grid`` is the main object containing information of the
        nonlifting bodygrid, consisting of triangular and quadrilateral panels.
    It is created by the solver :class:`sharpy.solvers.aerogridloader.AerogridLoader`

    """
    def __init__(self):
        super().__init__()
        self.grid_type = 'nonlifting_body'


    def generate(self, data_dict, beam, nonlifting_body_settings, ts):
        super().generate(data_dict, beam, nonlifting_body_settings, ts)

        # allocating initial grid storage
        self.ini_info = NonliftingBodyTimeStepInfo(self.dimensions)

        self.add_timestep()
        self.generate_mapping()
        self.generate_zeta(self.beam, self.aero_settings, ts)

    def generate_zeta_timestep_info(self, structure_tstep, nonlifting_body_tstep, beam, aero_settings, it=None, dt=None):
        super().generate_zeta_timestep_info(structure_tstep, nonlifting_body_tstep, beam, aero_settings, it, dt)

        for i_surf in range(self.n_surf):
            # Get Zeta and Zeta_dot (Panel node positions in G frame)
            nonlifting_body_tstep.zeta[i_surf], nonlifting_body_tstep.zeta_dot[i_surf] = self.get_zeta_and_zeta_dot(i_surf, structure_tstep)

    def get_zeta_and_zeta_dot(self,i_surf, structure_tstep):
        numb_radial_nodes = self.dimensions[i_surf][0] +1
        matrix_nodes = np.zeros((3, numb_radial_nodes,
                                 self.dimensions[i_surf][1]+1))
        matrix_nodes_dot = matrix_nodes.copy()
        array_phi_coordinates = np.linspace(0, 2*np.pi, numb_radial_nodes)
        # cache sin and cos values
        array_sin_phi = np.sin(array_phi_coordinates)
        array_cos_phi = np.cos(array_phi_coordinates)

        cga_rotation_matrix = structure_tstep.cga()

        for node_counter, i_global_node in enumerate(self.aero2struct_mapping[i_surf]):
            # 1) Set B-Frame position
            # 1a) get cross-sectional fuselage geometry at node
            if self.data_dict["shape"].decode() == 'specific':
                a_ellipse = self.data_dict["a_ellipse"][i_global_node]
                b_ellipse = self.data_dict["b_ellipse"][i_global_node]
                z_0 =self.data_dict["z_0_ellipse"][i_global_node]
                if a_ellipse == 0. or b_ellipse == 0.:
                    radius = 0
                else:
                    radius = a_ellipse*b_ellipse/np.sqrt(
                        (b_ellipse*array_cos_phi)**2
                        +(a_ellipse*array_sin_phi)**2)
            else:
                radius = self.data_dict["radius"][i_global_node]
                z_0 = 0
            
            # 1b) Get nodes position in B frame
            matrix_nodes[1, :, node_counter] = radius*array_cos_phi
            matrix_nodes[2, :, node_counter] = radius*array_sin_phi + z_0
            
            # 2) A frame
            # 2a) Convert structural position from B to A frame 
            i_elem, i_local_node = self.get_elment_and_local_node_id(i_surf, i_global_node)
            psi_node = structure_tstep.psi[i_elem, i_local_node,:]
            if not (psi_node == [0, 0, 0]).all():
                # just perform roation from B to A if psi not 0
                Cab = algebra.crv2rotation(psi_node)
                for idx in range(numb_radial_nodes):
                    matrix_nodes[:, idx, node_counter] = np.dot(Cab, matrix_nodes[:, idx, node_counter])
            # 2b) Add beam displacements (expressed in A-frame)
            for dim in range(3):
                matrix_nodes[dim, :, node_counter] += structure_tstep.pos[i_global_node,dim]

            # 2c) Add structural beam velocities (expressed in A-frame)
            for dim in range(3):
                # velocity due to pos_dot
                matrix_nodes_dot[dim, :, node_counter] += structure_tstep.pos_dot[i_global_node, dim]
            # 2d) Add effect of structural beam rotations an node velocity (expressed in A-frame)
            psi_dot_node = structure_tstep.psi_dot[i_elem, i_local_node,:]
            omega_a = algebra.crv_dot2omega(psi_node, psi_dot_node)     
            for idx in range(numb_radial_nodes):
                matrix_nodes_dot[:, idx, node_counter] += (np.dot(algebra.skew(omega_a), matrix_nodes[:, idx, node_counter]))
            
            # 3) Convert position and velocities from A to G frame
            for idx in range(numb_radial_nodes):
                matrix_nodes[:, idx, node_counter] = np.dot(cga_rotation_matrix,
                                          matrix_nodes[:, idx, node_counter])
                matrix_nodes_dot[:, idx, node_counter] = np.dot(cga_rotation_matrix,
                                          matrix_nodes_dot[:, idx, node_counter])
        return matrix_nodes, matrix_nodes_dot


    def get_elment_and_local_node_id(self, i_surf, i_global_node):
        # get beam elements of surface
        idx_beam_elements_surface = np.where(self.surface_distribution == i_surf)[0]
        # find element and local node of the global node and return psi
        for i_elem in idx_beam_elements_surface:
            if i_global_node in self.beam.elements[i_elem].reordered_global_connectivities:
                i_local_node = np.where(self.beam.elements[i_elem].reordered_global_connectivities == i_global_node)[0][0]
                return i_elem, i_local_node
        raise Exception("The global node %u could not be assigned to any element of surface %u." % (i_global_node, i_surf))







