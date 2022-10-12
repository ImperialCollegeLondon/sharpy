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
        self.get_dimensions()
        self.get_y_coordinates()

    def get_y_coordinates(self):
        for ivane in range(self.n_vanes):
            # Starts at y = 0 if symmetry condition is enforced
            self.y_coord.append(np.linspace(self.vane_info[ivane]['span']/2,
                -self.vane_info[ivane]['span']/2*int(not self.settings["symmetry_condition"]),
                self.vane_info[ivane]['N']+1))
            print(self.y_coord[-1])
            print(np.shape(self.y_coord[-1]))

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
            beam_cab = np.zeros((3,3))
            beam_cab[0,1] = 1.  
            beam_cab[1,0] = 1.
            beam_cab[2,2] = 1.  
            self.vane_info[ivane]['beam_psi'] =algebra.rotation2crv(beam_cab)
            self.vane_info[ivane]['psi_dot'] = [0.,0.,0.]
            self.vane_info[ivane]['pos_dot'] = [0.,0.,0.]
            self.vane_info[ivane]['cga'] = np.eye(3)
            self.vane_info[ivane]['M_distribution'] = 'uniform'


    def get_dimensions(self):
        self.aero_dimensions = np.zeros((self.n_vanes, 2), dtype=int)
        self.aero_dimensions_star = self.aero_dimensions.copy()
        for ivane in range(self.n_vanes):
            self.aero_dimensions[ivane, :] = [self.vane_info[ivane]['M'], self.vane_info[ivane]['N']]
            self.aero_dimensions_star[ivane, :] = [self.vane_info[ivane]['M_star'], self.vane_info[ivane]['N']] # Check if correct

    def init_control_surfaces(self):
        for ivane in range(self.n_vanes):    
            self.vane_info[ivane]['control_surface'] ={'control_surface_type': 'dynamic',
                                 'hinge_coords': 0.25,
                                 'chord': self.vane_info[ivane]['M'], # whole wing is control surface
                                 'deflection':0,
                                 'deflection_dot': 0}
            self.cs_generators.append(generator_interface.generator_from_string('DynamicControlSurface')())
            try:
                self.cs_generators[ivane].initialise(
                    self.vane_info[ivane]['control_surface_deflection_generator_settings']) #self.settings['control_surface_deflection_generator_settings'][str(ivane)])
            except KeyError:
                with_error_initialising_cs = True
                cout.cout_wrap('Error, unable to locate a settings dictionary for gust vane '
                                '{:g}'.format(ivane), 4)

    def generate_zeta(self, iteration, aero_tstep, airfoil_db, freestream_dir = np.array([1., 0., 0.])):
        if iteration > 0:
            self.update_cs_deflection_and_rate(iteration)
        for ivane in range(self.n_vanes):
            for inode in range(self.vane_info[ivane]['N']+1):
                self.vane_info[ivane]['beam_coord'][1] = self.y_coord[ivane][inode]
                if self.settings['vertical']:
                    self.vane_info[ivane]['beam_coord'][1:] = np.flip(self.vane_info[ivane]['beam_coord'][1:])
       
                (aero_tstep.zeta[aero_tstep.n_surf - self.n_vanes + ivane][:, :, inode],
                        aero_tstep.zeta_dot[aero_tstep.n_surf - self.n_vanes + ivane][:, :, inode]) = (
                            generate_strip(self.vane_info[ivane],
                                            airfoil_db,
                                            orientation_in=freestream_dir, #TODO: Check if this changes
                                            calculate_zeta_dot=True)) #TODO: Check effect of zeta dot on wake later
        
    def update_cs_deflection_and_rate(self, iteration):
        for ivane in range(self.n_vanes):
            self.vane_info[ivane]['control_surface']['deflection'], self.vane_info[ivane]['control_surface']['deflection_dot'] = \
                                    self.cs_generators[ivane]({'it': iteration})
