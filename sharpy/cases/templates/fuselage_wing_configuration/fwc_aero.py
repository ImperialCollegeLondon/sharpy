import h5py as h5
import numpy as np


class FWC_Aero:
    """
        FWC_Aero contains all attributes to define the aerodynamic grid and makes it accessible for SHARPy.
    """
    def __init__(self, structure, case_name, case_route, **kwargs):
        """        
        Key-Word Arguments:
            - structure: structural object of BFF model
        """
        self.structure = structure

        self.route = case_route
        self.case_name = case_name

        self.ea_wing = kwargs.get('elastic_axis',0.5)
        self.num_chordwise_panels = kwargs.get('num_chordwise_panels', 4)
        self.chord_wing = kwargs.get('chord', 1.)
        self.alpha_zero_deg = kwargs.get('alpha_zero_deg', 0.)
        self.n_surfaces = 2
        self.radius_fuselage = kwargs.get('max_radius', 0.5)
        self.lifting_only = kwargs.get('lifting_only', True)
        

    def generate(self):
        """
            Function to set up all necessary parameter inputs to define the geometry and discretisation
            of the lifting surfaces. Finally, informations are saved to an .h5 file serving as an input
            file for SHARPy.
        """
        self.initialize_attributes()
        self.set_wing_properties()
        self.set_junction_boundary_conditions()
        self.write_input_file()

    def initialize_attributes(self):
        """
            Initilializes all necessary attributes for the aero.h5 input file based on the number of 
            nodes, elements, and surfaces of the model.
        """
        self.airfoil_distribution = np.zeros((self.structure.n_elem, self.structure.n_node_elem), dtype=int)
        self.surface_distribution = np.zeros((self.structure.n_elem,), dtype=int)
        self.surface_m = np.zeros((self.n_surfaces, ), dtype=int) + self.num_chordwise_panels
        self.aero_node = np.zeros((self.structure.n_node,), dtype=bool)

        self.twist = np.zeros((self.structure.n_elem, self.structure.n_node_elem))
        self.sweep = np.zeros_like(self.twist)
        self.chord = np.zeros_like(self.twist)
        self.elastic_axis = np.zeros_like(self.twist)

        self.junction_boundary_condition_aero = np.zeros((1, self.n_surfaces), dtype=int) - 1
    
    def get_y_junction(self):
        if self.structure.vertical_wing_position == 0:
            return self.radius_fuselage
        else:
            return np.sqrt(self.radius_fuselage**2-self.structure.vertical_wing_position**2)
    def set_wing_properties(self):
        """
            Sets necessary parameters to define the lifting surfaces of one wing (right).
        """
        
        if not self.lifting_only:
            self.aero_node[:self.structure.n_node_right_wing] = abs(self.structure.y[:self.structure.n_node_right_wing]) > self.get_y_junction() - 0.05
            self.aero_node[self.structure.n_node_right_wing:self.structure.n_node_wing_total] = self.aero_node[1:self.structure.n_node_right_wing]
        else:
            self.aero_node[:self.structure.n_node_wing_total] = True
        self.chord[:2*self.structure.n_elem_per_wing, :] = self.chord_wing
        self.elastic_axis[:2*self.structure.n_elem_per_wing, :] = self.ea_wing
        self.twist[:2*self.structure.n_elem_per_wing, :] = -np.deg2rad(self.alpha_zero_deg)
        # surf distribution 0 for right and 1 for left wing
        self.surface_distribution[self.structure.n_elem_per_wing:2*self.structure.n_elem_per_wing] = 1

    def set_junction_boundary_conditions(self):
        """
            Sets the boundary conditions for the fuselage-wing junction. These BCs
            define the partner surface.
        """
        # Right wing (surface 0) has the left wing (surface 1) as a partner surface.
        self.junction_boundary_condition_aero[0, 0] = 1
        # Left wing (surface 1) has the left wing (surface 0) as a partner surface.
        self.junction_boundary_condition_aero[0, 1] = 0

    def write_input_file(self):
        """
            Writes previously defined parameters to an .h5 file which serves later as an 
            input file for SHARPy.
        """
            
        with h5.File(self.route + '/' + self.case_name + '.aero.h5', 'a') as h5file:
            airfoils_group = h5file.create_group('airfoils')
            # add one airfoil
            airfoils_group.create_dataset('0', 
                                          data=np.column_stack(
                                            self.generate_naca_camber(P=0, M=0)
                                          ))

            h5file.create_dataset('chord', data=self.chord)
            h5file.create_dataset('twist', data=self.twist)
            h5file.create_dataset('sweep', data=self.sweep)

            # airfoil distribution
            h5file.create_dataset('airfoil_distribution', data=self.airfoil_distribution)
            h5file.create_dataset('surface_distribution', data=self.surface_distribution)
            h5file.create_dataset('surface_m', data=self.surface_m)
            h5file.create_dataset('m_distribution', data='uniform')
            h5file.create_dataset('aero_node', data=self.aero_node)
            h5file.create_dataset('elastic_axis', data=self.elastic_axis)
            h5file.create_dataset('junction_boundary_condition', data=self.junction_boundary_condition_aero)
   

    def generate_naca_camber(self,M=0, P=0):
        """
            Generates the camber line coordinates of a specified NACA airfoil.
            # TODO: needed?
        """
        mm = M*1e-2
        p = P*1e-1

        def naca(x, mm, p):
            if x < 1e-6:
                return 0.0
            elif x < p:
                return mm/(p*p)*(2*p*x - x*x)
            elif x > p and x < 1+1e-6:
                return mm/((1-p)*(1-p))*(1 - 2*p + 2*p*x - x*x)

        x_vec = np.linspace(0, 1, 1000)
        y_vec = np.array([naca(x, mm, p) for x in x_vec])
        
        return x_vec, y_vec
