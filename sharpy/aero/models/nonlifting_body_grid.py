"""Nonlifting Body grid

Description
"""

from sharpy.aero.models.grid import Grid
from sharpy.utils.datastructures import NonliftingBodyTimeStepInfo
import numpy as np
from scipy.optimize import fsolve
import sharpy.utils.algebra as algebra


class Nonlifting_body_grid(Grid):
    """
    ``Nonlifting Body Grid`` is the main object containing information of the
        nonlifting bodygrid, consisting of triangular and quadrilateral panels.
    It is created by the solver :class:`sharpy.solvers.aerogridloader.AerogridLoader`

    """
    def __init__(self):
        super().__init__()
        self.grid_type = 'nonlifting_body'


    def generate(self, data_dict, beam, nonlifting_body_settings, ts): ##input?
        super().generate(data_dict, beam, nonlifting_body_settings, ts)

        # allocating initial grid storage
        self.ini_info = NonliftingBodyTimeStepInfo(self.dimensions)

        self.add_timestep()
        self.generate_mapping()
        self.generate_zeta(self.beam, self.aero_settings, ts)

    def generate_zeta_timestep_info(self, structure_tstep, nonlifting_body_tstep, beam, aero_settings, it=None, dt=None):
        super().generate_zeta_timestep_info(structure_tstep, nonlifting_body_tstep, beam, aero_settings, it, dt)

        for i_surf in range(self.n_surf):
            # Get Zeta (Collocation point positions in A? frame)
            # TO-DO: Consider fuselage deformation for node position calculations
            nonlifting_body_tstep.zeta[i_surf] = self.get_collocation_point_pos(i_surf, self.dimensions[i_surf][-1], structure_tstep)
            # TO-DO: Add Zeta Dot Calculation
            # aero_tstep.zeta_dot[i_surf]

    def get_triangle_center(self, p1, p2, p3):
        return (p1+p2+p3)/3

    def get_quadrilateral_center(self, list_points):
        # based on http://jwilson.coe.uga.edu/EMT668/EMT668.Folders.F97/Patterson/EMT%20669/centroid%20of%20quad/Centroid.html
        array_triangle_centroids = np.zeros((4,3))
        counter = 0
        triangle_combinations = [[0, 1, 2], [0, 2, 3], [0, 1, 3], [1, 2, 3]]
        for triangle in triangle_combinations: #list(itertools.combinations([0, 1, 2, 3], 3)):
            array_triangle_centroids[counter] = self.get_triangle_center(
                list_points[triangle[0]],
                list_points[triangle[1]],
                list_points[triangle[2]])
            counter+=1
        centroid_quadrilateral = self.find_intersection_points(array_triangle_centroids)
        return centroid_quadrilateral

    def get_nodes_position(self,i_surf, structure_tstep, numb_radial_nodes):
        matrix_nodes = np.zeros(((self.dimensions[i_surf][1]-1)
                                    *(numb_radial_nodes)+2, 3))
        array_phi_coordinates = np.linspace(0, 2*np.pi, numb_radial_nodes)
        # cache sin and cos values
        array_sin_phi = np.sin(array_phi_coordinates)
        array_cos_phi = np.cos(array_phi_coordinates)

        phi_counter = 0
        row_idx_start, row_idx_end = 0, 0
        for i_global_node in self.aero2struct_mapping[i_surf]:
            radius = self.data_dict["radius"][i_global_node]
            if radius == 0.0:
                # axissymmetric body has only one node at nose and tail (r = 0)
                # position is already in A frame
                matrix_nodes[row_idx_start, :] = structure_tstep.pos[i_global_node, :]
                row_idx_start += 1
            else:
                row_idx_end = row_idx_start + numb_radial_nodes
                # get nodes position in B frame
                matrix_nodes[row_idx_start:row_idx_end, 0] = 0
                matrix_nodes[row_idx_start:row_idx_end, 1] = radius*array_cos_phi
                matrix_nodes[row_idx_start:row_idx_end, 2] = radius*array_sin_phi
                # convert position from B to A frame
                psi_node = self.get_psi(i_surf, i_global_node, structure_tstep)
                if not (psi_node == [0, 0, 0]).all():
                    # just perform roation from B to A if psi not 0
                    Cab = algebra.crv2rotation(psi_node)
                    for idx in range(row_idx_start, row_idx_end):
                        matrix_nodes[idx,:] = np.dot(Cab, matrix_nodes[idx,:])
                matrix_nodes[row_idx_start:row_idx_end,:] += structure_tstep.pos[i_global_node,:]
                row_idx_start += numb_radial_nodes
            phi_counter += 1

        return matrix_nodes


    def get_collocation_point_pos(self, i_surf, numb_spanwise_elements, structure_tstep):
        numb_radial_nodes = self.surface_m[i_surf]+1
        matrix_nodes = self.get_nodes_position(i_surf, structure_tstep, numb_radial_nodes)
        counter, i_row = 0, 0
        matrix_collocation = np.zeros(((numb_spanwise_elements)*(numb_radial_nodes-1),3))
        for i in range(0, numb_spanwise_elements):
            if i == 0:
                for i_rad_node in range(0, numb_radial_nodes-1):
                    matrix_collocation[i_row] = self.get_triangle_center(
                        matrix_nodes[0],
                        matrix_nodes[i_rad_node+1],
                        matrix_nodes[i_rad_node+2])
                    i_row += 1
                counter+=1
            elif i == numb_spanwise_elements-1:
                for i_rad_node in range(0, numb_radial_nodes-1):
                    matrix_collocation[i_row] = self.get_triangle_center(
                        matrix_nodes[-1],
                        matrix_nodes[counter],
                        matrix_nodes[counter+1])
                    i_row+=1
                    counter+=1
            else:
                for i_rad_node in range(0, numb_radial_nodes-1):
                    matrix_collocation[i_row] = self.get_quadrilateral_center(
                        matrix_nodes[[counter, counter +1,
                                      counter+numb_radial_nodes+1,
                                      counter+numb_radial_nodes]])
                    counter+=1
                    i_row+=1
                counter+=1
        return matrix_collocation

    def get_psi(self, i_surf, i_global_node, structure_tstep):
        # get beam elements of surface
        idx_beam_elements_surface = np.where(self.surface_distribution == i_surf)[0]
        # find element and local node of the global node and return psi
        for i_elem in idx_beam_elements_surface:
            if i_global_node in self.beam.elements[i_elem].reordered_global_connectivities:
                i_local_node = np.where(self.beam.elements[i_elem].reordered_global_connectivities == i_global_node)[0][0]
                return structure_tstep.psi[i_elem, i_local_node, :]
        raise Exception("The global node %u could not be assigned to any element of surface %u." % (i_global_node, i_surf))

    def find_intersection_points(self, array_points):
        sol_parameter = fsolve(self.intersection_point_equation,
                               [0.3, 0.3, 0.0], args = tuple(array_points))
        return array_points[0]+sol_parameter[0]*(array_points[1]-array_points[0])

    def intersection_point_equation(self,z, *data):
        """
        Function to determine intersection point between 4 points for fsolve.
        Math: A+t*(B-A) -C - u*(D-C) = 0
        TO-DO: Define function in utils as helper function (algebra??)
        """
        point_A, point_B, point_C, point_D = data
        t = z[0]
        u = z[1]
        F = point_A+t*(point_B-point_A) - point_C - u*(point_D-point_C)
        return F






