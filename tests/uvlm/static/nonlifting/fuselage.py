import os
import numpy as np
import h5py as h5
import configobj
import sharpy.sharpy_main


class Fuselage():
    def __init__(self, case_name, case_route, output_route):
        self.case_name = case_name
        self.case_route = case_route
        self.output_route = output_route

        self.settings = None

        self.n_node_elem = 3

    def clean(self):
        list_files = ['.fem.h5', '.aero.h5', '.nonlifting_body.h5', '.dyn.h5', '.mb.h5', '.sharpy', '.flightcon.txt']
        for file in list_files:
            path_file = self.case_route + '/' + self.case_name + file
            if os.path.isfile(path_file):
                os.remove(path_file)
                
    def generate_structure(self, **kwargs):
        self.length = kwargs.get('length', 10)
        self.n_elem = kwargs.get('n_elem', 11)
        self.sigma_fuselage = kwargs.get('sigma_fuselage', 10.)

        self.set_beam_properties()
        self.set_stiffness_and_mass_propoerties()

        self.write_structural_input_file()

    def generate(self, **kwargs):
        self.clean()
        self.generate_structure(**kwargs)
        self.generate_fuselage(**kwargs)

    def generate_fuselage(self, **kwargs):
        self.num_radial_panels = kwargs.get('num_radial_panels', 24)
        self.max_radius = kwargs.get('max_radius', 2) # rename
        self.n_nonlifting_bodies = 1
        self.fuselage_shape = kwargs.get('fuselage_shape', 'ellipsoid')


        self.set_fuselage_properties()
        self.write_fuselage_input_file()

    def set_fuselage_properties(self):
        self.nonlifting_body_node = np.ones((self.n_node,), dtype=bool)
        self.nonlifting_body_distribution = np.zeros((self.n_elem,), dtype=int)
        self.nonlifting_body_m = np.zeros((self.n_nonlifting_bodies, ), dtype=int) + self.num_radial_panels
        self.radius = np.zeros((self.n_node,))

        if self.fuselage_shape == 'ellipsoid':
            self.radius = self.get_radius_ellipsoid(self.x, self.length/2, self.max_radius)

    def get_radius_ellipsoid(self, x_coordinates, a, b):
        x_coordinates -= np.mean(x_coordinates) # move origin to center
        y_coordinates = b*np.sqrt(1-(x_coordinates/a)**2)
        return y_coordinates

    def set_beam_properties(self):
        # number of nodes
        self.n_node = self.n_elem*2+1
        # coordinates
        self.x = np.linspace(0, self.length, self.n_node)
        self.y = np.zeros((self.n_node,))
        self.z = np.zeros((self.n_node,))

        self.frame_of_reference_delta = np.zeros((self.n_elem, self.n_node_elem, 3))
        self.conn = np.zeros((self.n_elem, self.n_node_elem), dtype=int)
        for ielem in range(self.n_elem):
            self.conn[ielem, :] = ((np.ones((3, )) * ielem * (self.n_node_elem - 1)) +
                                [0, 2, 1])               
            for ilocalnode in range(self.n_node_elem):
                self.frame_of_reference_delta[ielem, ilocalnode, :] = [0.0, 1.0, 0.0]  
   
        
        self.beam_number = np.zeros((self.n_elem, ), dtype=int)
        self.boundary_conditions = np.zeros((self.n_node, ), dtype=int)
        self.boundary_conditions[0] = -1
        self.boundary_conditions[-1] = -1
        self.boundary_conditions[self.n_node//2-1] = 1

        self.structural_twist = np.zeros((self.n_elem, self.n_node_elem))
    
    def set_stiffness_and_mass_propoerties(self):
        n_material = 1 # body
        self.stiffness = np.zeros((n_material, 6, 6))
        self.mass = np.zeros((n_material, 6, 6))
        self.elem_stiffness = np.zeros((self.n_elem, ), dtype=int)
        self.mass = np.zeros((n_material, 6, 6))
        self.elem_mass = np.zeros((self.n_elem, ), dtype=int)

        # Define aeroelastic properties (negligible here)
        ea = 1e7
        ga = 1e5
        gj = 1e4
        eiy = 2e4
        eiz = 4e6
        m_bar_main = 0.75
        j_bar_main = 0.075
        base_stiffness_main = np.diag([ea, ga, ga, gj, eiy, eiz])*self.sigma_fuselage
        base_stiffness_main[4, 4] = base_stiffness_main[5, 5]

        self.stiffness[0, ...] = base_stiffness_main
        self.mass[0, ...] = np.diag([m_bar_main, m_bar_main, m_bar_main, j_bar_main, 0.1 * j_bar_main, 1.0 * j_bar_main])

    def write_fuselage_input_file(self):
        """
            Writes previously defined parameters to an .h5 file which serves later as an 
            input file for SHARPy.
        """                
        with h5.File(self.case_route + '/' + self.case_name + '.nonlifting_body.h5', 'a') as h5file:
            h5file.create_dataset('shape', data='cylindrical')
            h5file.create_dataset('surface_m', data=self.nonlifting_body_m)
            h5file.create_dataset('nonlifting_body_node', data=self.nonlifting_body_node)

            h5file.create_dataset('surface_distribution', data=self.nonlifting_body_distribution)
            h5file.create_dataset('radius', data=self.radius)
  
    def write_structural_input_file(self):
        """
            Writes previously defined parameters to an .h5 file which serves later as an 
            input file for SHARPy.
        """                
        with h5.File(self.case_route + '/' + self.case_name + '.fem.h5', 'a') as h5file:
            h5file.create_dataset('coordinates', data=np.column_stack((self.x, self.y, self.z)))
            h5file.create_dataset('connectivities', data=self.conn)
            h5file.create_dataset('num_node_elem', data=self.n_node_elem)
            h5file.create_dataset('num_node', data=self.n_node)
            h5file.create_dataset('num_elem', data=self.n_elem)
            h5file.create_dataset('stiffness_db', data=self.stiffness)
            h5file.create_dataset('elem_stiffness', data=self.elem_stiffness)
            h5file.create_dataset('mass_db', data=self.mass)
            h5file.create_dataset('elem_mass', data=self.elem_mass)
            h5file.create_dataset('frame_of_reference_delta', data=self.frame_of_reference_delta)
            h5file.create_dataset('boundary_conditions', data=self.boundary_conditions)
            h5file.create_dataset('beam_number', data=self.beam_number)

            h5file.create_dataset('structural_twist', data=self.structural_twist)
            h5file.create_dataset('app_forces', data=np.zeros((self.n_node, 6)))

    def create_settings(self, settings):
        file_name = self.case_route + '/' + self.case_name + '.sharpy'
        config = configobj.ConfigObj()
        config.filename = file_name
        for k, v in settings.items():
            config[k] = v
        config.write()
        self.settings = settings

    def run(self):
        sharpy.sharpy_main.main(['', self.case_route + '/' + self.case_name + '.sharpy'])
