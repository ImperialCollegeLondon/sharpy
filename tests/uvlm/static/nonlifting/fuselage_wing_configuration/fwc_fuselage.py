#! /usr/bin/env python3
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
   

class FWC_Fuselage:
    """
        FWC_Fuselage contains all attributes to define the nonlifting body grid 
        representing the fuselage shape and makes it accessible for SHARPy.
    """
    def __init__(self, structure, case_name, case_route, **kwargs):
        """        
        Key-Word Arguments:
            - structure: structural object of BFF model
        """
        self.num_radial_panels = kwargs.get('num_radial_panels', 12)
        self.max_radius = kwargs.get('max_radius')
        self.fuselage_shape = kwargs.get('fuselage_shape','cylindrical')
        self.flag_plot_radius = kwargs.get('flag_plot_radius', False)
        self.structure = structure

        self.route = case_route
        self.case_name = case_name
        self.n_nonlifting_bodies = 1

    def generate(self): 
        """
            Function to set up all necessary parameter inputs to define the geometry and discretisation
            of the nonlifting surfaces. Finally, informations are saved to an .h5 file serving as an input
            file for SHARPy.
        """

        self.initialize_parameters()

        self.nonlifting_body_node[self.structure.n_node_wing_total:self.structure.n_node_fuselage_tail] = True
        if self.structure.vertical_wing_position == 0:
            self.nonlifting_body_node[0] = True
        self.nonlifting_body_distribution[self.structure.n_elem_per_wing*2:self.structure.n_elem_per_wing*2+self.structure.n_elem_fuselage] = 0
        self.nonlifting_body_m[0] = self.num_radial_panels

        self.get_fuselage_geometry()

        self.write_fuselage_input_file()
        

    def initialize_parameters(self):  
        """
            Initilializes all necessary attributes for the nonlifting.h5 input file based on the number of 
            nodes, elements, and surfaces of the nonlifting model.
        """      
        self.nonlifting_body_node = np.zeros((self.structure.n_node,), dtype=bool)
        self.nonlifting_body_distribution = np.zeros((self.structure.n_elem,), dtype=int) - 1
        self.nonlifting_body_m = np.zeros((self.n_nonlifting_bodies, ), dtype=int)
        self.radius = np.zeros((self.structure.n_node,))

    def get_fuselage_geometry(self):
        x_coord_fuselage_sorted = np.sort(self.get_values_at_fuselage_nodes(self.structure.x))
        if self.fuselage_shape == 'cylindrical':
            radius_fuselage = self.create_fuselage_geometry(x_coord_fuselage_sorted.copy(), 0.2*self.structure.fuselage_length, 0.8*self.structure.fuselage_length) 
        elif self.fuselage_shape == 'ellipsoid':
            radius_fuselage = self.get_radius_ellipsoid(x_coord_fuselage_sorted.copy(), self.structure.fuselage_length/2, self.max_radius)
        else:
            raise "ERROR Fuselage shape {} unknown.".format(self.fuselage_shape)
        if self.structure.vertical_wing_position == 0:
            self.radius[0] = radius_fuselage[self.structure.idx_junction]
            self.radius[self.structure.n_node_wing_total:] =np.delete(radius_fuselage, self.structure.idx_junction)
        else:
            self.radius[self.structure.n_node_wing_total:self.structure.n_node_fuselage_tail] = radius_fuselage
        if self.flag_plot_radius:
            self.plot_fuselage_radius(self.get_values_at_fuselage_nodes(self.structure.x), self.get_values_at_fuselage_nodes(self.radius))
            self.plot_fuselage_radius(x_coord_fuselage_sorted, radius_fuselage)

    def get_values_at_fuselage_nodes(self, array):
        return array[self.nonlifting_body_node]
    
    def plot_fuselage_radius(self, x, radius):
        plt.scatter(x,
                 radius)
        plt.grid()
        plt.xlabel("x, m")
        plt.ylabel("r, m")
        plt.gca().set_aspect('equal')
        plt.show()

    def write_fuselage_input_file(self):
        with h5.File(self.route + '/' + self.case_name + '.nonlifting_body.h5', 'a') as h5file:
            h5file.create_dataset('shape', data='cylindrical')
            h5file.create_dataset('surface_m', data=self.nonlifting_body_m)
            h5file.create_dataset('nonlifting_body_node', data=self.nonlifting_body_node)
            h5file.create_dataset('surface_distribution', data=self.nonlifting_body_distribution)
            h5file.create_dataset('radius', data=self.radius)

    def get_radius_ellipsoid(self, x_coordinates, a, b):
        x_coordinates[:] -=  x_coordinates[-1] - a
        y_coordinates = b*np.sqrt(1-(x_coordinates/a)**2)
        return y_coordinates

    def find_index_of_closest_entry(self, array_values, target_value):
        return np.argmin(np.abs(array_values - target_value))


    def create_fuselage_geometry(self, x_coord_fuselage, x_nose_end, x_tail_start):
        array_radius = np.zeros_like(x_coord_fuselage)
        # start with nose at zero to get indices of nose and tail correct
        x_coord_fuselage[:] -=  x_coord_fuselage[0]
        idx_cylinder_start = self.find_index_of_closest_entry(x_coord_fuselage, x_nose_end)
        idx_cylinder_end = self.find_index_of_closest_entry(x_coord_fuselage, x_tail_start)

        # set constant radius of cylinder
        array_radius[idx_cylinder_start:idx_cylinder_end] = self.max_radius
        # get ellipsoidal nose and tail shape
        array_radius[:idx_cylinder_start+1] = self.get_nose_shape(x_coord_fuselage[:idx_cylinder_start+1],
                                                                  self.max_radius)
        
        array_radius[idx_cylinder_end-1:] = np.flip(self.get_nose_shape(x_coord_fuselage[idx_cylinder_end-1:],
                                                                  self.max_radius))
        
        return array_radius
    
    def get_nose_shape(self, x_coord, radius):
        n_nodes = len(x_coord)
        x_coord[:] -= x_coord[-1]
        x_coord = np.concatenate((x_coord, np.flip(x_coord[:-1])))        
        radius = self.get_radius_ellipsoid(x_coord, x_coord[0], radius)
        return radius[:n_nodes]
