import numpy as np
import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings
import sharpy.utils.cout_utils as cout
import sharpy.utils.algebra as algebra
from sharpy.aero.models.aerogrid import generate_strip


@generator_interface.generator
class GustVanes(generator_interface.BaseGenerator):
    """
    ``GustVanes`` generator.

    This generator sets up a specified number of gust vanes that are oscillated by a previous defined signal. These oscillating
    gust vanes generate a gust as done in wind tunnel experiments. For each vane a new aerogrid surface including a wake is 
    generated in the ``Aerogrid`` solver. 
 
    """
    generator_id = 'GustVanes'
    generator_classification = 'runtime'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    settings_types['n_vanes'] = 'int'
    settings_default['n_vanes'] = 1
    settings_description['n_vanes'] = 'Number of gust vanes to be generated'

    settings_types['streamwise_position'] = 'list(float)'
    settings_default['streamwise_position'] = [-2.]
    settings_description['streamwise_position'] = 'List of streamwise coordinaten of each gust vane leading edge'

    settings_types['vertical_position'] = 'list(float)'
    settings_default['vertical_position'] = [-2.]
    settings_description['vertical_position'] = 'List of vertical coordinaten of each gust vane leading edge'

    settings_types['vane_parameters'] = 'list(dict)'
    settings_default['vane_parameters'] = [{'M': 4,'N': 20, 'M_star': 40, 'span': 10., 'chord': 0.1,}]
    settings_description['vane_parameters'] = 'Dictionary of parameters to specify the gust vane geometry and its position '

    settings_types['symmetry_condition'] = 'bool'
    settings_default['symmetry_condition'] = False
    settings_description['symmetry_condition'] = 'If ``True``, symmetry is enforced at global x-z-plane at y = 0'
        
    settings_types['vertical'] = 'bool'
    settings_default['vertical'] = False
    settings_description['vertical'] = 'If ``True``, gust vanes are oriented vertically.'

    settings_types['1-cosine_discretisation'] = 'bool'
    settings_default['1-cosine_discretisation'] = False
    settings_description['1-cosine_discretisation'] = 'If ``True``, gust vanes are discretised in a 1-cosine shape in spanwise direction.'

    settings_types['consider_zeta_dot'] = 'bool'
    settings_default['consider_zeta_dot'] = False
    settings_description['consider_zeta_dot'] = 'If ``True``, gust vanes consider the velocity of the lifting surfaces in the UVLM.'

    settings_types['wingtip_refinement_panels'] = 'int'
    settings_default['wingtip_refinement_panels'] = 3
    settings_description['wingtip_refinement_panels'] = 'Number of panels to refine wingtip discretisation for roll-up improvements'


    def __init__(self):
        self.settings = None
        self.vane_info = None
        self.aero_dimensions = None
        self.aero_dimensions_star = None
        self.cs_generators = [] 
        self.y_coord = []

    def initialise(self, in_dict, **kwargs):
        self.settings = in_dict
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default, no_ctype=True,
                                 options=self.settings_options)
        self.n_vanes = self.settings['n_vanes']
        self.vane_info = self.settings['vane_parameters']     
        
        self.set_default_vane_settings()
        self.init_control_surfaces()      
        self.get_y_coordinates()       
        self.get_dimensions()

    def get_y_coordinates(self):

        for ivane in range(self.n_vanes):
            if self.settings['1-cosine_discretisation']:
                # TODO: discretisation in case for symmetry condition
                domain = np.linspace(0, 1, self.vane_info[ivane]['N']+1)
                y_coord = 0.5*(1.0 - np.cos(domain*np.pi))
                y_coord = (y_coord * self.vane_info[ivane]['span']) \
                    - self.vane_info[ivane]['span']/2*int(not self.settings["symmetry_condition"])
                self.y_coord.append(y_coord)
            else:
                # Starts at y = 0 if symmetry condition is enforced
                self.y_coord.append(np.linspace(self.vane_info[ivane]['span']/2,
                    -self.vane_info[ivane]['span']/2*int(not self.settings["symmetry_condition"]),
                    self.vane_info[ivane]['N']+1))

            if self.settings['wingtip_refinement_panels'] > 0:
                self.refine_wingtip_discretisation(ivane)
            

    def refine_wingtip_discretisation(self, ivane):
        """
            Refines the spanwise grid discretisation at the wingtip. The wingtip panel is split 
            into a specified number of uniform panels to avoid instabilities in the wake roll-up
            process of the gust vanes.
        """
        delta_y_refined = (self.y_coord[ivane][1] - self.y_coord[ivane][0])\
             / (self.settings['wingtip_refinement_panels'] + 1)

        y_coord_refinement_panels = np.array([self.y_coord[ivane][0] + (i + 1) * delta_y_refined \
            for i in range(self.settings['wingtip_refinement_panels'])])

        self.y_coord[ivane] = np.insert(self.y_coord[ivane], 1,y_coord_refinement_panels)
        
        if not self.settings["symmetry_condition"]:
            self.y_coord[ivane] = np.insert(self.y_coord[ivane], -1,  -np.flip(y_coord_refinement_panels))


    def set_default_vane_settings(self):
        # TODO: Find better solution, especially for beam_psi
        for ivane in range(self.n_vanes):  
            self.vane_info[ivane]['sweep'] = 0    
            self.vane_info[ivane]['twist'] = 0   
            self.vane_info[ivane]['eaxis'] = 0    
            self.vane_info[ivane]['airfoil'] = 0  
            self.vane_info[ivane]['beam_coord'] = [self.settings['streamwise_position'][ivane],
                                                    0.,
                                                    self.settings['vertical_position'][ivane],]    
            
            if not self.settings['vertical']:
                beam_cab = np.zeros((3,3))
                beam_cab[0,1] = 1.  
                beam_cab[1,0] = 1.
                beam_cab[2,2] = 1.  
                self.vane_info[ivane]['beam_psi'] = algebra.rotation2crv(beam_cab)
            else:
                self.vane_info[ivane]['beam_psi'] = np.array([1.209199576156145373, -1.20919957615615, 1.20919957615615])

            self.vane_info[ivane]['psi_dot'] = [0.,0.,0.]
            self.vane_info[ivane]['pos_dot'] = [0.,0.,0.]
            self.vane_info[ivane]['cga'] = np.eye(3)
            self.vane_info[ivane]['M_distribution'] = 'uniform'


    def get_dimensions(self):
        self.aero_dimensions = np.zeros((self.n_vanes, 2), dtype=int)
        self.aero_dimensions_star = self.aero_dimensions.copy()
        for ivane in range(self.n_vanes):
            self.aero_dimensions[ivane, :] = [self.vane_info[ivane]['M'], len(self.y_coord[ivane])-1]
            self.aero_dimensions_star[ivane, :] = [self.vane_info[ivane]['M_star'],  self.aero_dimensions[ivane, 1]]

    def init_control_surfaces(self):
        for ivane in range(self.n_vanes):    
            self.vane_info[ivane]['control_surface'] ={'control_surface_type': 'dynamic',
                                 'hinge_coords': 0.25,                # TODO: specify by input
                                 'chord': self.vane_info[ivane]['M'], # whole wing is control surface
                                 'deflection':0,
                                 'deflection_dot': 0}
            self.cs_generators.append(generator_interface.generator_from_string('DynamicControlSurface')())
            try:
                self.cs_generators[ivane].initialise(
                    self.vane_info[ivane]['control_surface_deflection_generator_settings']
                    )
            except KeyError:
                cout.cout_wrap('Error, unable to locate a settings dictionary for gust vane '
                                '{:g}'.format(ivane), 4)

    def generate_zeta(self, iteration, aero_tstep, airfoil_db, freestream_dir = np.array([1., 0., 0.])):
        if iteration > 0:
            self.update_cs_deflection_and_rate(iteration)
        for ivane in range(self.n_vanes):
            for inode in range(self.aero_dimensions[ivane, 1] + 1):
                if not self.settings['vertical']:
                    self.vane_info[ivane]['beam_coord'][1] = self.y_coord[ivane][inode]
                else:
                    self.vane_info[ivane]['beam_coord'][2] = self.y_coord[ivane][inode]
                    self.vane_info[ivane]['beam_coord'][1] = self.settings['vertical_position'][ivane]
                # print(self.vane_info[ivane]['beam_coord'])

                (aero_tstep.zeta[aero_tstep.n_surf - self.n_vanes + ivane][:, :, inode],
                        aero_tstep.zeta_dot[aero_tstep.n_surf - self.n_vanes + ivane][:, :, inode]) = (
                            generate_strip(self.vane_info[ivane],
                                            airfoil_db,
                                            orientation_in=freestream_dir, #TODO: Check if this changes
                                            calculate_zeta_dot=self.settings['consider_zeta_dot']))

    def update_cs_deflection_and_rate(self, iteration):
        for ivane in range(self.n_vanes):
            self.vane_info[ivane]['control_surface']['deflection'], self.vane_info[ivane]['control_surface']['deflection_dot'] = \
                                    self.cs_generators[ivane]({'it': iteration})
