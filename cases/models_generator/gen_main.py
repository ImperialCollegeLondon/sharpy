import numpy as np
import copy
import os
import configobj
import h5py as h5
from functools import partial
import importlib
import sharpy.utils.generate_cases as gc
import sharpy.utils.h5utils as h5utils
import sharpy.utils.solver_interface as solver_interface
import sharpy.sharpy_main
import cases.models_generator.gen_utils as gu

class Sharpy_data:

    def __init__(self, workflow):

        self.fem = dict()
        self.aero = dict()
        self.workflow = workflow
        
        self.structure_create = self.structure_input = self.structure_read = 0
        if 'create_structure' in workflow:
            self.structure_create =1
        elif 'input_structure' in workflow:
            self.structure_input = 1
        elif 'read_structure' in workflow:
            self.structure_read = 1
            
        self.aero_create =  self.aero_create0 = self.aero_input = self.aero_read = 0
        if 'create_aero' in workflow:
            self.aero_create =1
        elif 'create_aero0' in workflow:
            self.aero_create0 =1
        elif 'input_aero' in workflow:
            self.aero_input = 1
        elif 'read_aero' in workflow:
            self.aero_read = 1
            
        if 'simulation' in workflow:
            self.sim = Simulation('sharpy')
            
    def read_structure(self, file_name):
        
        h5utils.check_file_exists(file_name)
        with h5.File(file_name, 'r') as fem_file_handle:
            self.fem = h5utils.load_h5_in_dict(fem_file_handle)
            
    def input_structure(self, **kwargs):
        self.fem = kwargs

    def write_structure(self, file_name):
        
        if os.path.exists(file_name):
            os.remove(file_name)
        elif not os.path.exists('/'.join(file_name.split('/')[:-1])):
            os.makedirs('/'.join(file_name.split('/')[:-1]))
        with h5.File(file_name, 'a') as h5file:
            for k, v in self.fem.items():
                h5file.create_dataset(k, data=v)
        
    def read_aero(self,file_name):
        
        h5utils.check_file_exists(file_name)
        with h5.File(file_name, 'r') as aero_file_handle:
            self.aero = h5utils.load_h5_in_dict(aero_file_handle)

    def input_aero(self, **kwargs):

        self.aero = kwargs

    def write_aero(self, file_name):

        if os.path.exists(file_name):
            os.remove(file_name)
        elif not os.path.exists('/'.join(file_name.split('/')[:-1])):
            os.makedirs('/'.join(file_name.split('/')[:-1]))
        if os.path.exists(file_name):
            os.remove(file_name)

        with h5.File(file_name, 'a') as h5file:
            for k, v in self.aero.items():
                if k == 'airfoils':
                    airfoils_group = h5file.create_group('airfoils')
                    for iairfoil in range(len(v)):
                        airfoils_group.create_dataset("%d" % iairfoil,
                                                      data=v[iairfoil])
                elif k =='m_distribution':
                    h5file.create_dataset('m_distribution', data=v.encode('ascii', 'ignore'))
                else:
                    h5file.create_dataset(k, data=v)
    def read_sim(self,file_name):
        self.sim.read_sharpy(file_name)
        
    def input_sim(self, **kwargs):
        self.sim.settings = kwargs
        
    def write_sim(self, file_name):
        
        if os.path.exists(file_name):
            os.remove(file_name)
        elif not os.path.exists('/'.join(file_name.split('/')[:-1])):
            os.makedirs('/'.join(file_name.split('/')[:-1]))
        if os.path.exists(file_name):
            os.remove(file_name)
            
        self.sim.write_sharpy(file_name)

class Cpacs_data:
    pass

class Nastran_data:
    pass

class Components:

    def __init__(self, key, in_put, out_put, settings):

        self.key = key
        if 'sharpy' in out_put:
            self.sharpy = Sharpy_data(settings['workflow'])
            #import pdb; pdb.set_trace()
            if self.sharpy.structure_create:
                if 'geometry' in settings.keys():
                    settings['fem']['coordinates'] = self.generate_geometry(
                                                       **settings['geometry'])
                else:
                    self.generate_geometry(coordinates=settings['fem']['coordinates'])
                self.sharpy_fem(**settings['fem'])
            elif self.sharpy.structure_read:
                self.sharpy.read_structure(settings['read_structure_file'])
            elif self.sharpy.structure_input:
                self.sharpy.input_structure(**settings['fem'])
            if self.sharpy.aero_create:
                self.sharpy_aero(**settings['aero'])
            elif self.sharpy.aero_create0:
                self.sharpy_aero(chord=0, elastic_axis=0, surface_m=2,
                                 aero_node=False, surface_distribution=-1)
            elif self.sharpy.aero_read:
                self.sharpy.read_aero(settings['read_aero_file'])
            elif self.sharpy.aero_input:
                self.sharpy.input_aero(**settings['aero'])

        if 'cpacs' in out_put:
            self.cpacs = Cpacs_data(settings['workflow'])
        if 'nastran' in out_put:
            self.nastran = Nastran_data(settings['workflow'])

    def generate_geometry(self,
                          length=1.,
                          num_node=None,
                          ds=None,
                          direction=None,
                          node0=[0., 0., 0.],
                          sweep=0.,
                          dihedral=0.,
                          coordinates=None):

        try:
            self.subcomponents = len(length)
        except TypeError:
            self.subcomponents = 1

        if coordinates is None:
            if (num_node is not None) and (ds is None):
                self.num_node = num_node
                self.ds = [length/(num_node-1)]
                self.ds_string = 0
            elif ds is not None:
                if ds is float or ds is int:
                    self.num_node = length/ds+1
                    self.ds = [ds]
                    self.ds_string = 0
                else:
                    self.num_node = len(ds)+1
                    self.ds_string = 1
                    self.ds = ds
            if isinstance(node0, list):
                node0 = np.array(node0)
            self.coordinates = [node0]
            if direction is not None:
                if isinstance(direction,list):
                    self.direction = np.array(direction)
                else:
                    self.direction = direction
                self.direction = self.direction/np.linalg.norm(self.direction)    
            else:
                Rotation =  np.array([[1., 0., 0.],
                                     [0.,np.cos(dihedral), np.sin(dihedral)],
                                     [0.,-np.sin(dihedral), np.cos(dihedral)]]).dot(
                            np.array([[np.cos(sweep), np.sin(sweep), 1.],
                                     [-np.sin(sweep), np.cos(sweep), 0.],
                                     [0., 0., 1.]]))
                self.direction = Rotation.dot(np.array([0., 1., 0.]))
            for ni in range(1,self.num_node):
                if self.ds_string:
                    self.coordinates.append(node0+sum(ds[:ni])*self.direction)
                    self.ds_list = self.ds
                else:
                    self.coordinates.append(node0+self.ds[0]*ni*self.direction)
                    self.ds_list = [self.ds[0] for i in range(self.num_node-1)]

            self.coordinates = np.array(self.coordinates)
            return self.coordinates
        else:
            self.coordinates = coordinates
            self.num_node = len(coordinates)
            self.ds_string = 1
            self.ds = [np.linalg.norm(coordinates[i+1]-coordinates[i]) for i in \
                       range(self.num_node-1)]
            self.ds = np.array(self.ds)

    def sharpy_fem(self,
                   coordinates,
                   stiffness_db,
                   mass_db,
                   sigma=1.,
                   sigma_m=1.,
                   num_node=None,
                   num_elem=None,
                   conn0=0,
                   connectivities=None,
                   num_node_elem=3,
                   frame_of_reference_delta=[-1, 0, 0],
                   boundary_conditions=None,
                   beam_number=None,
                   structural_twist=None,
                   app_forces=None,
                   elem_stiffness=None,
                   elem_mass=None,
                   lumped_mass=None,
                   lumped_mass_nodes=None,
                   lumped_mass_inertia=None,
                   lumped_mass_position=None):

        self.sharpy.fem['coordinates'] = coordinates
        if num_node is None:
            self.sharpy.fem['num_node'] = len(coordinates)
        else:
            assert len(coordinates)==num_node, 'num_node not equal to len(coordinates)'
            self.sharpy.fem['num_node'] = num_node
        if num_elem is None:
            self.sharpy.fem['num_elem'] = int((self.sharpy.fem['num_node']-1)/(num_node_elem-1))
        else:
            self.sharpy.fem['num_elem'] = num_elem
        self.sharpy.fem['num_node_elem'] = num_node_elem        
        if connectivities is None:
            self.sharpy.fem['connectivities'] = gu.do_connectivities(conn0,
                                                self.sharpy.fem['num_elem'], lista=[])
        else:
            self.sharpy.fem['connectivities'] = connectivities
            
        self.sharpy.fem['frame_of_reference_delta'] = np.zeros((self.sharpy.fem['num_elem'],
                                                                self.sharpy.fem['num_node_elem'], 3))
        if len(np.shape(frame_of_reference_delta)) == 1:
            self.sharpy.fem['frame_of_reference_delta'][:,:] = frame_of_reference_delta
        elif len(np.shape(frame_of_reference_delta))==2:
            for ne in range(self.sharpy.fem['num_elem']):
                self.sharpy.fem['frame_of_reference_delta'][ne,:] = frame_of_reference_delta[ne]
        elif len(np.shape(frame_of_reference_delta))==3:
                self.sharpy.fem['frame_of_reference_delta'] = frame_of_reference_delta
        if boundary_conditions is None:
            self.sharpy.fem['boundary_conditions'] = np.zeros(self.sharpy.fem['num_node'])
            self.sharpy.fem['boundary_conditions'][0] = 1
            self.sharpy.fem['boundary_conditions'][-1] = -1
        else:
            self.sharpy.fem['boundary_conditions'] = boundary_conditions
        if beam_number is None:
            self.sharpy.fem['beam_number'] = np.zeros(self.sharpy.fem['num_elem'])
        else:
            self.sharpy.fem['beam_number'] = beam_number
        if structural_twist is not None:
            self.sharpy.fem['structural_twist'] = structural_twist
        else:
            self.sharpy.fem['structural_twist'] = np.zeros((self.sharpy.fem['num_elem'],
                                                                self.sharpy.fem['num_node_elem']))
        if app_forces is not None:
            self.sharpy.fem['app_forces'] = app_forces
        else:
            self.sharpy.fem['app_forces'] = np.zeros((self.sharpy.fem['num_node'], 6))
        if len(np.shape(stiffness_db)) == 3:
            self.sharpy.fem['stiffness_db'] = sigma*stiffness_db
        elif len(np.shape(stiffness_db)) == 2:
            self.sharpy.fem['stiffness_db'] = sigma*self.create_stiff_db(stiffness_db[0],
                                                                         stiffness_db[1],
                                                                         stiffness_db[2],
                                                                         stiffness_db[3],
                                                                         stiffness_db[4],
                                                                         stiffness_db[5],
                                                                         stiffness_db[6])
        if elem_stiffness is None:
            self.sharpy.fem['elem_stiffness'] = np.zeros(self.sharpy.fem['num_elem'], dtype=int)
        else:
            self.sharpy.fem['elem_stiffness'] = elem_stiffness
        if len(np.shape(mass_db)) == 3:
            self.sharpy.fem['mass_db'] = sigma_m*mass_db
        elif len(np.shape(mass_db)) == 2:
            self.sharpy.fem['mass_db'] = sigma_m*self.create_mass_db(mass_db[0],
                                                                     mass_db[1],
                                                                     mass_db[2],
                                                                     mass_db[3],
                                                                     mass_db[4],
                                                                     mass_db[5],
                                                                     mass_db[6])
        if elem_mass is None:
            self.sharpy.fem['elem_mass'] = np.zeros(self.sharpy.fem['num_elem'], dtype=int)
        else:
            self.sharpy.fem['elem_mass'] = elem_mass
        if lumped_mass is not None:
            self.sharpy.fem['lumped_mass'] = lumped_mass
        if lumped_mass_nodes is not None:
            self.sharpy.fem['lumped_mass_nodes'] = lumped_mass_nodes
        if lumped_mass_inertia is not None:
            self.sharpy.fem['lumped_mass_inertia'] = lumped_mass_inertia
        if lumped_mass_position is not None:
            self.sharpy.fem['lumped_mass_position'] = lumped_mass_position
        
    def create_mass(self,
                    vec_mass_per_unit_length,
                    vec_mass_iner_x,
                    vec_mass_iner_y,
                    vec_mass_iner_z,
                    vec_pos_cg_B,
                    vec_mass_iner_yz=None):
        """
        Create the mass matrices from the vectors of properties

        Args:
            vec_mass_per_unit_length (np.array): masses per unit length
            vec_mass_iner_x (np.array): inertias around the x axis
            vec_mass_iner_y (np.array): inertias around the y axis
            vec_mass_iner_z (np.array): inertias around the z axis
            vec_pos_cg_B (np.array): position of the masses
            vec_mass_iner_yz (np.array): inertias around the yz axis
        """

        if vec_mass_iner_yz is None:
            vec_mass_iner_yz = np.zeros_like(vec_mass_per_unit_length)

        mass_db = np.zeros((len(vec_mass_per_unit_length), 6, 6), dtype=float)
        mass = np.zeros((6, 6),)
        for i in range(len(vec_mass_per_unit_length)):
            mass[0:3, 0:3] = np.eye(3)*vec_mass_per_unit_length[i]
            mass[0:3, 3:6] = -1.0*vec_mass_per_unit_length[i]*algebra.skew(vec_pos_cg_B[i])
            mass[3:6, 0:3] = -1.0*mass[0:3, 3:6]
            mass[3:6, 3:6] = np.diag([vec_mass_iner_x[i],
                                      vec_mass_iner_y[i],
                                      vec_mass_iner_z[i]])
            mass[4, 5] = vec_mass_iner_yz[i]
            mass[5, 4] = vec_mass_iner_yz[i]
            mass_db[i] = mass
        return mass_db
                
    def create_stiff_db(self,
                        vec_EA,
                        vec_GAy,
                        vec_GAz,
                        vec_GJ,
                        vec_EIy,
                        vec_EIz,
                        vec_EIyz=None):
        """
        Create the stiffness matrices from the vectors of properties

        Args:
            vec_EA (np.array): Axial stiffness
            vec_GAy (np.array): Shear stiffness in the y direction
            vec_GAz (np.array): Shear stiffness in the z direction
            vec_GJ (np.array): Torsional stiffness
            vec_EIy (np.array): Bending stiffness in the y direction
            vec_EIz (np.array): Bending stiffness in the z direction
            vec_EIyz (np.array): Bending stiffness in the yz direction
        """

        if vec_EIyz is None:
            vec_EIyz = np.zeros_like(vec_EA)

        stiffness_db = np.zeros((len(vec_EA), 6, 6),)
        for i in range(len(vec_EA)):
            stiffness_db[i] = np.diag([vec_EA[i],
                                            vec_GAy[i],
                                            vec_GAz[i],
                                            vec_GJ[i],
                                            vec_EIy[i],
                                            vec_EIz[i]])
            stiffness_db[i][4, 5] = vec_EIyz[i]
            stiffness_db[i][5, 4] = vec_EIyz[i]
        return stiffness_db
                
    def cpacs_fem(self):
        pass
    
    def nastran_fem(self):
        pass
        
    def sharpy_aero(self,
                    chord,
                    elastic_axis,
                    surface_m,
                    twist=0.,
                    sweep=0.,
                    surface_distribution=None,
                    m_distribution='uniform',
                    aero_node=None,
                    airfoils=None,
                    airfoil_distribution=None,
                    airfoil_efficiency=None):

        for v in ['chord', 'twist', 'sweep', 'elastic_axis']:
            self.sharpy_aero_input(v, locals()[v])
        if isinstance(surface_m, int):
            self.sharpy.aero['surface_m'] = [surface_m]
        elif isinstance(surface_m, float):
            self.sharpy.aero['surface_m'] = [int(surface_m)]
        else:
            self.sharpy.aero['surface_m'] = surface_m
        self.sharpy.aero['m_distribution'] = m_distribution
        if surface_distribution is not None:
            if isinstance(surface_distribution,int):
                self.sharpy.aero['surface_distribution'] = \
                surface_distribution*np.ones((self.sharpy.fem['num_elem'],), dtype=int)
            else:
                self.sharpy.aero['surface_distribution'] = surface_distribution
        else:
            self.sharpy.aero['surface_distribution'] = np.zeros((self.sharpy.fem['num_elem'],), dtype=int)
        if aero_node is None:
            self.sharpy.aero['aero_node'] = np.ones((self.num_node,), dtype = bool)
        elif aero_node is False:
            self.sharpy.aero['aero_node'] = np.zeros((self.num_node,), dtype = bool)
        else:
            self.sharpy.aero['aero_node'] = aero_node
        if airfoils is not None:
            self.sharpy.aero['airfoils'] = airfoils
        else:
            camber_points = self.sharpy.aero['surface_m'][0]+1
            self.sharpy.aero['airfoils'] = np.zeros((1,
                                                     camber_points,
                                                     2), dtype=float)
            self.sharpy.aero['airfoils'][0, :, 0] = np.linspace(0.0, 1.0, camber_points)
        if airfoil_distribution is not None:
            self.sharpy.aero['airfoil_distribution'] = airfoil_distribution
        else:
            self.sharpy.aero['airfoil_distribution'] = np.zeros((self.sharpy.fem['num_elem'],
                                                                 self.sharpy.fem['num_node_elem']), dtype=int)
        if airfoil_efficiency is not None:
            self.sharpy.aero['airfoil_efficiency'] = airfoil_efficiency

    def sharpy_aero_input(self, key, var):

        right_size=(self.sharpy.fem['num_elem'], self.sharpy.fem['num_node_elem'])
        if isinstance(var, float) or isinstance(var, int):
            self.sharpy.aero[key] = var*np.ones(right_size, dtype=float)
        elif (isinstance(var, np.ndarray) or isinstance(var,list)) and len(var) == 2:
            self.sharpy.aero[key] = np.zeros(right_size)
            self.sharpy.aero[key][0, 0] = var[0]
            dyds=(var[1]-var[0])/sum(self.ds_list)
            #import pdb;pdb.set_trace()
            for k in range(self.num_node):
                i=int(k/(right_size[1]-1))
                j=k%(right_size[1]-1)
                if i==0 and j==0:
                    self.sharpy.aero[key][i,j] = var[0]
                    var_p = var[0]
                    continue
                elif i==int((self.num_node-1)/(right_size[1]-1)):
                    self.sharpy.aero[key][i-1,1] = var[1]
                    assert np.isclose(var[1],var_p + self.ds_list[k-1]*dyds)
                    continue
                elif j==0:
                    self.sharpy.aero[key][i,j] = var_p + self.ds_list[k-1]*dyds
                    self.sharpy.aero[key][i-1,1] = var_p + self.ds_list[k-1]*dyds
                else:
                    self.sharpy.aero[key][i,2] = var_p + self.ds_list[k-1]*dyds

                var_p += self.ds_list[k-1]*dyds
                    
        elif (isinstance(var,np.ndarray) or isinstance(var,list)) and len(var)>2:
            self.sharpy.aero[key] = var
            assert np.shape(self.sharpy.aero[key]) == right_size, \
                "{} not complying with syze {}".format(key, right_size)
    def generate_controls(self):
        pass


class Model:

    
    def __init__(self, m_input, m_output, model_dict, components_dict, simulation_dict,
                 model_name=None):

        self.m_input = m_input
        self.m_output = m_output
        self.model_dict = model_dict
        self.components_dict = components_dict
        self.simulation_dict = simulation_dict
        self.components = [*components_dict]
        self.num_components = len(self.components)
        if model_name is None:
            self.model_name = self.components[0]
        else:
            self.model_name = model_name
        try:
            if len(model_dict['iterate_vars']) == 0:
                self.num_models = 1
                self.model_labels = ['']
            else:
                
                self.iteration = Iterations(self.model_dict['iterate_vars'],
                                            self.model_dict['iterate_type'])
                self.num_models = self.iteration.num_combinations
                self.model_labels = self.iteration.labels(**self.model_dict['iterate_labels'])
                self.dict2iterate = self.iteration.get_combinations_dict()
                for mi in range(self.num_models):
                    #import pdb;pdb.set_trace()
                    for k in self.dict2iterate[mi].keys():
                        if len(k.split('*'))==1:
                            self.dict2iterate[mi][self.components[0]+'*'+k] = self.dict2iterate[mi].pop(k)
                
        except KeyError:
            self.num_models = 1
            self.model_labels = ['']
        self.models = []

    def assemble_models(self, models, settings):

        self.built_models = [type('model%s'%i, (object,), {})() for i in range(self.num_models)]
        for mi in range(self.num_models):
            self.built_models[mi].sharpy = Sharpy_data(workflow=['structure_input',
                                                                 'aero_input'])
            dic_structure, dic_aero = self.assemble_components(models[mi], settings)
            self.built_models[mi].sharpy.input_structure(**dic_structure)
            if settings['include_aero']:
                self.built_models[mi].sharpy.input_aero(**dic_aero)
    
    def assemble_components(self, model, settings):

        dic_struc = dict()      # Dictionary to concatenate structural components
        dic_aero = dict()       # Dictionary to concatenate aero components
        self.dict_comp = dict() # Dictionary to save new components variables
        num_elem = 0
        node2add = 0
        stiffness_dbi = 0
        mass_dbi = 0
        for ci in range(self.num_components):
            self.dict_comp[self.components[ci]] = dict()
            if ci == 0:
                dic_struc['num_node_elem'] = model[ci].sharpy.fem['num_node_elem']
                coordinates = model[ci].sharpy.fem['coordinates']
                boundary_conditions = model[ci].sharpy.fem['boundary_conditions']
                connectivities = gu.do_connectivities(0,
                                                   model[ci].sharpy.fem['num_elem'], lista=[])
                self.dict_comp[self.components[ci]]['nodes'] = np.arange(model[ci].sharpy.fem['num_node']) \
                                                               + node2add
                node2add += len(coordinates)
                elem0 = model[ci].sharpy.fem['num_elem']
                dic_struc['num_node'] = node2add
                dic_struc['num_elem'] = elem0
                dic_struc['coordinates'] = coordinates
                self.dict_comp[self.components[ci]]['coordinates'] = coordinates
                dic_struc['connectivities'] = connectivities
                dic_struc['boundary_conditions'] = boundary_conditions
                dic_struc['elem_stiffness'] = model[ci].sharpy.fem['elem_stiffness']
                dic_struc['elem_mass'] = model[ci].sharpy.fem['elem_mass']
                dic_struc['stiffness_db'] = model[ci].sharpy.fem['stiffness_db']
                dic_struc['mass_db'] = model[ci].sharpy.fem['mass_db']
                dic_struc['frame_of_reference_delta'] = model[ci].sharpy.fem['frame_of_reference_delta']
                if settings['default_settings']:
                    beam_number = ci*np.ones(model[ci].sharpy.fem['num_elem'])
                else:
                    beam_number = model[ci].sharpy.fem['beam_number']
                dic_struc['beam_number'] = beam_number
                dic_struc['structural_twist'] = model[ci].sharpy.fem['structural_twist']
                dic_struc['app_forces'] = model[ci].sharpy.fem['app_forces']
                
                aero_vars = ['chord', 'elastic_axis', 'twist', 'sweep',
                             'airfoil_efficiency']
                if model[ci].sharpy.aero_create or model[ci].sharpy.aero_create0:             
                    for k in aero_vars:
                        try:
                            dic_aero[k] = model[ci].sharpy.aero[k]
                        except KeyError:
                            print('No variable %s defined in component %s'%(k,self.components[ci]))
                    dic_aero['aero_node'] = model[ci].sharpy.aero['aero_node']
                    dic_aero['surface_distribution'] = model[ci].sharpy.aero['surface_distribution']
                    if np.max(model[ci].sharpy.aero['surface_distribution'])<0:
                        dic_aero['surface_m'] = []
                    else:
                        dic_aero['surface_m'] = model[ci].sharpy.aero['surface_m']
                    dic_aero['m_distribution'] = model[ci].sharpy.aero['m_distribution']
                    dic_aero['airfoils'] = [model[ci].sharpy.aero['airfoils'][i] for i \
                                            in range(len(model[ci].sharpy.aero['airfoils']))]
                    dic_aero['airfoil_distribution'] = model[ci].sharpy.aero['airfoil_distribution']

            else:
                #import pdb; pdb.set_trace()
                coordinates0 = self.dict_comp[settings[self.components[ci]]['upstream_component']]\
                              ['coordinates'][settings[self.components[ci]]['node_in_upstream']]
                coordinates = model[ci].sharpy.fem['coordinates'][1:] + coordinates0 \
                    - model[ci].sharpy.fem['coordinates'][0]
                self.dict_comp[self.components[ci]]['coordinates'] = model[ci].sharpy.fem['coordinates'] + \
                coordinates0 - model[ci].sharpy.fem['coordinates'][0]
                boundary_conditions = model[ci].sharpy.fem['boundary_conditions'][1:]
                connectivities = gu.do_connectivities(node2add-1,
                                                   model[ci].sharpy.fem['num_elem'], lista=[])
                node_connection = self.dict_comp[settings[self.components[ci]]['upstream_component']]['nodes']\
                                  [settings[self.components[ci]]['node_in_upstream']]
                connectivities[0][0] = node_connection
                self.dict_comp[self.components[ci]]['nodes'] = np.arange(model[ci].sharpy.fem['num_node']) \
                        + node2add-1
                self.dict_comp[self.components[ci]]['nodes'][0] = node_connection
                #settings[self.components[ci]]['node2add'] = node2add
                node2add += len(coordinates)
                elem0 += model[ci].sharpy.fem['num_elem']
                dic_struc['num_node'] = node2add
                dic_struc['num_elem'] = elem0
                dic_struc['coordinates'] = np.concatenate((dic_struc['coordinates'],
                                                            coordinates), axis=0)
                dic_struc['connectivities'] = np.concatenate((dic_struc['connectivities'],
                                                            connectivities), axis=0)
                if 'boundary_connection' in settings[self.components[ci]]:
                    dic_struc['boundary_conditions'][node_connection] = settings[self.components[ci]]\
                                                                        ['boundary_connection']
                dic_struc['boundary_conditions'] = np.concatenate((dic_struc['boundary_conditions'],
                                                                 boundary_conditions), axis=0)
                elem_stiffness = model[ci].sharpy.fem['elem_stiffness'] + len(dic_struc['stiffness_db'])
                elem_mass = model[ci].sharpy.fem['elem_mass'] + len(dic_struc['mass_db'])
                dic_struc['elem_stiffness'] = np.concatenate((dic_struc['elem_stiffness'],
                                                                 elem_stiffness), axis=0)
                dic_struc['elem_mass'] = np.concatenate((dic_struc['elem_mass'],
                                                                 elem_mass), axis=0)
                dic_struc['stiffness_db'] = np.concatenate((dic_struc['stiffness_db'],
                                                        model[ci].sharpy.fem['stiffness_db']),axis=0)
                dic_struc['mass_db'] = np.concatenate((dic_struc['mass_db'],
                                                        model[ci].sharpy.fem['mass_db']),axis=0)
                dic_struc['frame_of_reference_delta'] = np.concatenate((dic_struc['frame_of_reference_delta'],
                                                                      model[ci].sharpy.fem['frame_of_reference_delta']),axis=0)
                if settings['default_settings']:
                    beam_number = ci*np.ones(model[ci].sharpy.fem['num_elem'])
                else:
                    beam_number = model[ci].sharpy.fem['beam_number']
                dic_struc['beam_number'] = np.concatenate((dic_struc['beam_number'],
                                                         beam_number), axis=0)
                dic_struc['structural_twist'] = np.concatenate((dic_struc['structural_twist'],
                                                                      model[ci].sharpy.fem['structural_twist']),axis=0)
                app_forces = model[ci].sharpy.fem['app_forces'][1:]
                dic_struc['app_forces'] = np.concatenate((dic_struc['app_forces'],
                                                          app_forces),axis=0)
                
                if model[ci].sharpy.aero_create or model[ci].sharpy.aero_create0:
                    #import pdb;pdb.set_trace()
                    for k in aero_vars:
                        try:
                            dic_aero[k] = np.concatenate((dic_aero[k],
                                                              model[ci].sharpy.aero[k]),axis=0)
                        except KeyError:
                            print('No variable %s defined in component %s'%(k,self.components[ci]))
                    #import pdb;pdb.set_trace()
                    aero_node = model[ci].sharpy.aero['aero_node'][1:]
                    if 'keep_aero_node' in settings[self.components[ci]]:
                        if settings[self.components[ci]]['keep_aero_node']:
                            dic_aero['aero_node'][node_connection] = model[ci].sharpy.aero['aero_node'][0]
                    dic_aero['aero_node'] = np.concatenate((dic_aero['aero_node'], aero_node),axis=0)
                    if settings['default_settings']:
                        if np.max(model[ci].sharpy.aero['surface_distribution'])>-1:
                            surface_distributionx = np.max(dic_aero['surface_distribution'])+1
                        else:
                            surface_distributionx = np.max(model[ci].sharpy.aero['surface_distribution'])
                        surface_distribution = surface_distributionx*np.ones(model[ci].sharpy.fem['num_elem'],
                                                                         dtype=int)
                    else:
                        surface_distribution = model[ci].sharpy.aero['surface_distribution']
                    dic_aero['surface_distribution'] = np.concatenate((dic_aero['surface_distribution'],
                                                                       surface_distribution),axis=0)
                    if np.max(model[ci].sharpy.aero['surface_distribution'])<0:
                        surface_m = dic_aero['surface_m']
                    else:
                        surface_m = dic_aero['surface_m'] + model[ci].sharpy.aero['surface_m']
                    dic_aero['surface_m'] = gu.flatten2(surface_m)
                    airfoil_distribution = model[ci].sharpy.aero['airfoil_distribution']+ \
                                                       len(dic_aero['airfoils'])
                    dic_aero['airfoil_distribution'] = np.concatenate((dic_aero['airfoil_distribution'],
                                                                       airfoil_distribution),axis=0)
                    airfoils = [model[ci].sharpy.aero['airfoils'][i] for i \
                                            in range(len(model[ci].sharpy.aero['airfoils']))]
                    dic_aero['airfoils'] +=airfoils 

                if model[ci].sharpy.aero_create:
                    dic_aero['m_distribution'] = model[ci].sharpy.aero['m_distribution']

            if 'lumped_mass' in model[ci].sharpy.fem.keys():
                dic_struc['lumped_mass'] = np.concatenate((dic_struc['lumped_mass'],
                                                        model[ci].sharpy.fem['lumped_mass']),axis=0)
                dic_struc['lumped_mass_inertia'] = np.concatenate((dic_struc['lumped_mass_inertia'],
                                                        model[ci].sharpy.fem['lumped_mass_inertia']),axis=0)
                dic_struc['lumped_mass_position'] = np.concatenate((dic_struc['lumped_mass_position'],
                                                        model[ci].sharpy.fem['lumped_mass_position']),axis=0)
                lumped_mass_nodes = model[ci].sharpy.fem['lumped_mass_nodes'] + settings[self.components[ci]]['node2add']
                dic_struc['lumped_mass_nodes'] = np.concatenate((dic_struc['lumped_mass_nodes'],
                                                               lumped_mass_nodes),axis=0)

        return dic_struc, dic_aero
        
    def build(self):
        
        if self.num_models > 1:
            for mi in range(self.num_models):
                for k, i in self.dict2iterate[mi].items():
                    #k01 -> compont name ;
                    #k1 -> model_type (fem,aero,generate..) ; k2 -> variable name
                    k01, k02 = k.split('*')
                    k1, k2 = k02.split('-')
                    self.components_dict[k01][k1][k2] = i
                compX = []
                for ci in self.components:
                    if 'symmetric' in self.components_dict[ci]:
                        ci2 = self.components_dict[ci]['symmetric']['component']
                        components_dictx = copy.deepcopy(self.components_dict[ci2])
                        if 'geometry' not in self.components_dict[ci].keys():
                            components_dictx['fem']['coordinates'][:,1] = \
                                -self.components_dict[ci2]['fem']['coordinates'][:,1]
                        if 'geometry' in self.components_dict[ci].keys() and \
                           'node0' in self.components_dict[ci]['geometry'].keys():
                            components_dictx['geometry']['node0'] = self.components_dict[ci]['geometry']['node0']
                        if 'geometry' in components_dictx.keys() and \
                        'sweep' in components_dictx['geometry'].keys():    
                            components_dictx['geometry']['sweep'] = np.pi-components_dictx['geometry']['sweep']
                        elif 'geometry' in components_dictx.keys() and \
                        'direction' in components_dictx['geometry'].keys():    
                            components_dictx['geometry']['direction'][1] = -components_dictx['geometry']['direction'][1]
                        try:    
                            components_dictx['fem']['frame_of_reference_delta'] = -components_dictx['fem']['frame_of_reference_delta']
                        except TypeError:
                            components_dictx['fem']['frame_of_reference_delta'] = -np.array(components_dictx['fem']['frame_of_reference_delta'])
                        compX.append(Components(ci, in_put=self.m_input, out_put=self.m_output, settings=components_dictx))
                    else:
                        compX.append(Components(ci, in_put=self.m_input, out_put=self.m_output, settings=self.components_dict[ci]))
                self.models.append(compX)
        elif self.num_models ==1:
            compX = []    
            for ci in self.components:    
                compX.append(Components(ci, in_put=self.m_input, out_put=self.m_output, settings=self.components_dict[ci]))
            self.models.append(compX)
        if self.num_components > 1:
            self.assemble_models(self.models, self.model_dict['assembly'])
        elif self.num_components == 1:
            self.built_models = [self.models[i][0] for i in range(self.num_models)]
        
    def run(self):
        self.build()
        if 'sharpy' in self.m_output:

            for mi in range(self.num_models):
                case_route = self.model_dict['model_route']
                case_name = self.model_dict['model_name'] + self.model_labels[mi]
                folder2write = case_route + '/' + case_name
                file2write = folder2write + '/' + self.model_dict['model_name']
                gu.change_dic(self.simulation_dict['sharpy'], 'folder', folder2write)
                self.built_models[mi].sharpy.sim = Simulation(sim_type='sharpy',
                                                              settings_sim=self.simulation_dict['sharpy'],
                                                              case_route=folder2write,
                                                              case_name=self.model_dict['model_name'])
                self.built_models[mi].sharpy.sim.get_sharpy(inp=self.simulation_dict['sharpy']['simulation_input'])
                self.built_models[mi].sharpy.write_structure(file2write+'.fem.h5')
                self.built_models[mi].sharpy.write_aero(file2write+'.aero.h5')
                self.built_models[mi].sharpy.write_sim(file2write+'.sharpy')
                sharpy.sharpy_main.main(['', file2write+'.sharpy'])


#folder_replace = partial(change_dic, oldstr='folder')
#'run_in_loop','generate_sharpy','run_sharpy','generate_nastran','run_nastran','generate_cpacs'
#fem-num_node
#components_dict['wing']['fem'/'aero'/'generate']['a']
#components_dict -> input_program
#iterate_vars={'wing*fem-num_node:10'} dict2iterate=[{wing*fem-num_node:10}]
#iterate_sim{''}
class Simulation:

    def __init__(self, sim_type, settings_sim={}, case_route='', case_name=''):

        self.sim_type = sim_type
        self.settings_sim = settings_sim
        self.case_route = case_route
        self.case_name = case_name
        self.settings = dict()

    @staticmethod
    def sharpy_defaults(default_module, default_solution, default_solution_vars):
        module = importlib.import_module(default_module)
        solution = getattr(module, default_solution)
        flow, settings = solution(**default_solution_vars)
        return flow, settings
    
    def gen_sharpy_main(self, flow, **kwargs):

        self.settings['SHARPy'] = {'case': self.case_name,
                      'route': self.case_route,
                      'flow': flow,
                      'write_screen': 'on',
                      'write_log': 'on',
                      'log_folder': self.case_route,
                      'log_file': self.case_name + '.log',
                      'save_settings': 'on'}

        for k, v in kwargs.items():
            self.settings['SHARPy'][k] = v

    def read_sharpy(self, file_name):
        self.settings = configobj.ConfigObj(file_name)

    def input_simulation(self, **kwargs):
        self.settings = kwargs

    def get_sharpy(self, inp=None):

        if inp is None:
            self.flow, self.settings = self.sharpy_defaults(
                self.settings_sim['default_module'],
                self.settings_sim['default_solution'],
                self.settings_sim['default_solution_vars'])
            self.gen_sharpy_main(self.flow, **self.settings_sim['default_sharpy'])
        elif isinstance(inp, str):
            self.read_sharpy(inp)
        elif isinstance(inp, dict):
            self.input_simulation()

    def write_sharpy(self, file_name=''):
        
        config = configobj.ConfigObj()
        np.set_printoptions(precision=16)
        if not file_name:
            file_name = self.case_route + '/' + self.case_name + '.sharpy'
        config.filename = file_name
        for k, v in self.settings.items():
            config[k] = v
        config.write()

class Iterations:

    """
    Class that manages iterations from the variables to iterate, vars2iterate,
    the type of iteration, full factorial and design of experiments implemented so far,
    to the labels and values corresponding to each iteration
    """
    
    def __init__(self,var2iterate,iter_type='Full_Factorial'):

        self.iter_type = iter_type
        if isinstance(var2iterate,list):
            self.varl = var2iterate
        elif isinstance(var2iterate,dict):
            self.vard = var2iterate
            self.varl = [var2iterate[k] for k in var2iterate.keys()]
            self.var_names = [k for k in var2iterate.keys()]
        self.num_var = len(self.varl)
        self.shape_var = [len(i) for i in self.varl]
        if self.iter_type == 'Full_Factorial':
            self.num_combinations = np.prod(self.shape_var)
        elif self.iter_type == 'DoE':
            self.num_combinations = self.shape_var[0]
            
    def get_combinations_FF(self):
        
        """
        List with all combinations from full factorial experiment
        """
        
        self.varl_combinations = []
        self.varl_index = []
        for i in range(self.num_combinations):
            comb_i = []
            for j in range(self.num_var):
                if j==0:
                    n=1
                else:
                    n = np.prod(self.shape_var[:j])
                l = int(i/n)
                comb_i.append(l%self.shape_var[j])    # indices of the combinations

            self.varl_combinations.append([self.varl[k][comb_i[k]] for k in range(self.num_var)])
            self.varl_index.append(comb_i)
        #self.num_combinations = len(self.varl_combinations)
        return self.varl_combinations

    def get_combinations_DoE(self):
        
        """
        List with all combinations from full factorial experiment
        """
        
        self.varl_combinations = []
        self.varl_index = []
        for k in range(self.num_combinations):
            self.varl_combinations.append([self.varl[j][k] for j in range(self.num_var)])
            self.varl_index.append([k for j in range(self.num_var)])
        #self.num_combinations = len(self.varl_combinations)
        return self.varl_combinations

    def get_combinations_dict(self):

        """
        List of all the combinations in the iteration and
        formed of dictionaries with keys the name of the variables
        """

        if self.iter_type == 'Full_Factorial':
            varl_combinations = self.get_combinations_FF()
        elif self.iter_type == 'DoE':
            varl_combinations = self.get_combinations_DoE()
        self.vard_combinations = []
        for i in range(self.num_combinations):
            dic = dict()
            for j in range(self.num_var):
                dic[self.var_names[j]] = varl_combinations[i][j]
            self.vard_combinations.append(dic)
        return self.vard_combinations
    
    def label_name(self, num2keep=3, space='_', print_name_var=1):

        """
        Labels for the iteration with the name of the variable followed its value or index
        """
        
        dic1 = self.get_combinations_dict()
        label = []
        for ni in range(self.num_combinations):
            name = ''
            for k, i in dic1[ni].items():
                if not print_name_var:
                    k = ''
                name += k +str(i)[0:num2keep]
                name+= space
            if space:
                label.append(name[:-1])
            else:
                label.append(name)
        return label

    def label_number(self, num2keep=3, space='_', print_name_var=1):

        dic1 = self.get_combinations_dict()
        #import pdb;pdb.set_trace()
        label = []
        for ni in range(self.num_combinations):
            name = ''
            j=0
            for k,i in dic1[ni].items():
                if not print_name_var:
                    k = ''
                #for j in 
                name += k +str(list(self.varl[j]).index(i))
                name+= space
                j+=1
            if space:
                label.append(name[:-1])
            else:
                label.append(name)
        return label

    def labels(self,label_type='name', **kwargs):

        if label_type == 'name':
            return self.label_name(**kwargs)
        elif label_type == 'number':
            return self.label_number(**kwargs)

    def write_vars2text():
        pass
