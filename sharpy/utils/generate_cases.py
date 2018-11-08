"""
Generate cases

This library provides functions and classes to help in the definition of SHARPy cases

# Longer description
#
# Args:
#
# 	arg1 (float): arg1 description
#
# 	arg2 (float): arg2 description
#
# Returns:
#
# 	float: function return descritpion

Examples:

    tests in: tests/utils/generate_cases

Notes:

    To use this library: import sharpy.utils.generate_cases as generate_cases
	# Here you can have specifics and math descriptions.
    #
	# To enter math mode simply write
    #
	# .. math:: e^{i\\pi} + 1 = 0
    #
	# Math mode supports TeX sintax but note that you should use two backslashes \\ instead of one.
    #
	# Sphinx supports reStructuredText should you want to format text...

"""

######################################################################
##################  PYTHON PACKAGES  #################################
######################################################################
# Usual SHARPy
import h5py as h5
import numpy as np
import os
import sharpy.utils.algebra as algebra
import sharpy.utils.cout_utils as cout  # to use color output
# Read excel files
import pandas as pd
from pandas import ExcelFile
# Generate errors during execution
import sys
# Use .isnan
import math

from copy import deepcopy

######################################################################
##################  MATHEMATICAL FUNCTIONS  ##########################
######################################################################
def rotate_vector(vector, direction, angle):
    # This function rotates a "vector" around a "direction" a certain "angle"
    # according to Rodrigues formula

    # Create numpy arrays from vectors
    vector = np.array(vector)
    direction = np.array(direction)

    # Assure that "direction" has unit norm
    direction /= np.linalg.norm(direction)

    rot_vector = (vector*np.cos(angle) +
                  np.cross(direction, vector)*np.sin(angle) +
                  direction*np.dot(direction, vector)*(1.0-np.cos(angle)))

    return rot_vector


def get_airfoil_camber(x, y, n_points_camber):
    # Returns the airfoil camber for a given set of coordinates (XFOIL format expected)

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)
    imin_x = 0

    # Look for the minimum x (it will be assumed as the LE position
    for i in range(n):
        if(x[i] < x[imin_x]):
            imin_x = i

    x_suction = np.zeros((imin_x+1, ))
    y_suction = np.zeros((imin_x+1, ))
    x_pressure = np.zeros((n-imin_x, ))
    y_pressure = np.zeros((n-imin_x, ))

    for i in range(0, imin_x+1):
        x_suction[i] = x[imin_x-i]
        y_suction[i] = y[imin_x-i]

    for i in range(imin_x, n):
        x_pressure[i-imin_x] = x[i]
        y_pressure[i-imin_x] = y[i]

    # Compute the camber coordinates
    camber_y = np.zeros((n_points_camber, ))
    camber_x = np.linspace(0.0, 1.0, n_points_camber)

    # camber_y=0.5*(np.interp(camber_x,x[imin_x::-1],y[imin_x::-1])+np.interp(camber_x,x[imin_x:],y[imin_x:]))
    camber_y = 0.5*(np.interp(camber_x, x_suction, y_suction) +
                    np.interp(camber_x, x_pressure, y_pressure))

    # The function should be called as: camber_x, camber_y = get_airfoil_camber(x,y)
    return camber_x, camber_y

def from_node_list_to_elem_matrix(node_list, connectivities):

    num_elem = len(connectivities)
    # TODO: change the "3" for self.num_node_elem
    elem_matrix = np.zeros((num_elem,3), dtype=node_list.dtype)
    for ielem in range(num_elem):
        elem_matrix[ielem, :] = node_list[connectivities[ielem, :]]

    return elem_matrix

######################################################################
##################  DEFINE CONSTANTS  ################################
######################################################################
deg2rad = 2.0*np.pi/360.0

######################################################################
###############  STRUCTURAL INFORMATION  #############################
######################################################################
class StructuralInformation():

    def __init__(self):

        # Basic variables
        self.num_node_elem = 3

        # Variables to write in the h5 file
        self.num_elem = None
        self.num_node = None
        self.coordinates = None
        self.connectivities = None
        # self.num_node_elem = None
        self.stiffness_db = None
        self.elem_stiffness = None
        self.mass_db = None
        self.elem_mass = None
        self.frame_of_reference_delta = None
        self.structural_twist = None
        self.boundary_conditions = None
        self.beam_number = None
        self.app_forces = None

        # Other variabes
        self.defi_FoR_wrt_AFoR = None # Coordinates of the FoR in which the structure is being defined with respect to the A FoR
        # TODO: include rotations between "defi" and AFoR
        self.elem_r = None
        self.node_r = None
        self.in_global_AFoR = False

        # Variables related to wind turbine blades
        self.cone = None
        self.tilt = None
        self.node_prebending = None
        self.node_presweept = None
        self.node_structural_twist = None
        self.elem_EA = None
        self.elem_EIz = None
        self.elem_EIy = None
        self.elem_GJ = None
        self.elem_GAy = None
        self.elem_GAz = None
        self.elem_mass_per_unit_length = None
        self.elem_mass_iner_x = None
        self.elem_mass_iner_y = None
        self.elem_mass_iner_z = None
        self.elem_pos_cg_B = None

        self.body_number = None


    def create_node_radial_pos_from_elem_centres(self, elem_centres):
        '''
        elem_centres contains the radial position of the following relevant points
        measured from the hub centre:
        - First value: Beginning of the blade
        - Last value: Tip of the blade
        - Rest of the values: The rest of the strucutral element centres
        '''

        self.elem_r = elem_centres[1:-1]
        self.node_r = np.zeros((self.num_node, ), )
        self.node_r[0] = elem_centres[0]
        self.node_r[-2] = elem_centres[-2]
        self.node_r[-1] = elem_centres[-1]

        for ielem in range(self.num_elem-1):
            self.node_r[ielem*(self.num_node_elem-1)+1] = self.elem_r[ielem]
            self.node_r[ielem*(self.num_node_elem-1)+2] = 0.5*(self.elem_r[ielem]+self.elem_r[ielem+1])

    def create_frame_of_reference_delta(self, y_BFoR = 'z_AFoR'):

        if y_BFoR == 'x_AFoR':
            yB = np.array([1.0, 0.0, 0.0])
        elif y_BFoR == 'y_AFoR':
            yB = np.array([0.0, 1.0, 0.0])
        elif y_BFoR == 'z_AFoR':
            yB = np.array([0.0, 0.0, 1.0])
        else:
            print("WARNING: y_BFoR not recognized, using the default value: y_BFoR = z_AFoR")

        # y vector of the B frame of reference
        self.frame_of_reference_delta = np.zeros((self.num_elem,self.num_node_elem,3),)
        for ielem in range(self.num_elem):
            for inode in range(self.num_node_elem):
                # TODO: do i need to use the connectivities?
                self.frame_of_reference_delta[ielem,inode,:] = yB


    def create_mass_db(self, vec_mass_per_unit_length, vec_mass_iner_x, vec_mass_iner_y, vec_mass_iner_z, vec_pos_cg_B):

        self.mass_db = np.zeros((self.num_elem,6,6),)
        mass = np.zeros((6,6),)
        for i in range(len(vec_mass_per_unit_length)):
            mass[0:3, 0:3] = np.eye(3)*vec_mass_per_unit_length[i]
            mass[0:3, 3:6] = -1.0*vec_mass_per_unit_length[i]*algebra.skew(vec_pos_cg_B[i])
            mass[3:6, 0:3] = -1.0*mass[0:3, 3:6]
            mass[3:6, 3:6] = np.diag([vec_mass_iner_x[i],
                                      vec_mass_iner_y[i],
                                      vec_mass_iner_z[i]])

            self.mass_db[i] = mass

    def create_stiff_db(self, vec_EA, vec_GAy, vec_GAz, vec_GJ, vec_EIy, vec_EIz):

        self.stiffness_db = np.zeros((self.num_elem,6,6),)
        for i in range(len(vec_EA)):
            self.stiffness_db[i] = np.diag([vec_EA[i], vec_GAy[i], vec_GAz[i], vec_GJ[i], vec_EIy[i], vec_EIz[i]])

    def rotate_beam(self, axis, angle):

        # axis = [np.cos(tilt),np.sin(tilt),0.0]
        for inode in range(len(self.coordinates)):
            self.coordinates[inode,:] = rotate_vector(self.coordinates[inode,:], axis, angle)

        for ielem in range(self.num_elem):
            for inode in range(self.num_node_elem):
                # TODO: do i need to use the connectivities?
                self.frame_of_reference_delta[ielem,inode,:] = rotate_vector(self.frame_of_reference_delta[ielem,inode, :], axis, angle)

    def create_cantilever_BC(self):

        self.boundary_conditions = np.zeros((self.num_node), dtype=int)
        self.boundary_conditions[0] = 1
        self.boundary_conditions[-1] = -1

    def create_simple_connectivities(self):

        self.connectivities = np.zeros((self.num_elem,self.num_node_elem), dtype=int)
        for ielem in range(self.num_elem):
            self.connectivities[ielem, :] = np.array([0, 2, 1], dtype=int) + ielem*(self.num_node_elem - 1)

    def assembly_structures(self, *args):
        '''
        This function concatenates structures to be writen in the same h5 File
        The structures does NOT share any node (even if nodes are defined at the same coordinates)
        '''
        total_num_beam = max(self.beam_number)+1
        total_num_node = self.num_node
        total_num_elem = self.num_elem

        for structure_to_add in args:
            self.coordinates = np.concatenate((self.coordinates, structure_to_add.coordinates ), axis=0)
            self.connectivities = np.concatenate((self.connectivities, structure_to_add.connectivities + total_num_node), axis=0)
            assert self.num_node_elem == structure_to_add.num_node_elem, "num_node_elem does NOT match"
            self.stiffness_db = np.concatenate((self.stiffness_db, structure_to_add.stiffness_db), axis=0)
            self.elem_stiffness = np.concatenate((self.elem_stiffness, structure_to_add.elem_stiffness + total_num_elem), axis=0)
            self.mass_db = np.concatenate((self.mass_db, structure_to_add.mass_db), axis=0)
            self.elem_mass = np.concatenate((self.elem_mass, structure_to_add.elem_mass + total_num_elem), axis=0)
            self.frame_of_reference_delta = np.concatenate((self.frame_of_reference_delta, structure_to_add.frame_of_reference_delta), axis=0)
            self.node_structural_twist = np.concatenate((self.node_structural_twist, structure_to_add.node_structural_twist), axis=0)
            self.boundary_conditions = np.concatenate((self.boundary_conditions, structure_to_add.boundary_conditions), axis=0)
            self.beam_number = np.concatenate((self.beam_number, structure_to_add.beam_number + total_num_beam), axis=0)
            self.app_forces = np.concatenate((self.app_forces, structure_to_add.app_forces), axis=0)
            self.body_number = np.concatenate((self.body_number, structure_to_add.body_number), axis=0)

            total_num_beam += max(structure_to_add.beam_number)
            total_num_node += structure_to_add.num_node
            total_num_elem += structure_to_add.num_elem

        self.num_node = total_num_node
        self.num_elem = total_num_elem

    def generate_uniform_sym_beam(self, node_pos, mass_per_unit_length, mass_iner, EA, GJ, EI):

        # Check the number of nodes
        self.num_node = len(node_pos)
        if ((self.num_node-1) % (self.num_node_elem-1)) == 0:
            self.num_elem = int((self.num_node-1)/(self.num_node_elem-1))
        else:
            print("ERROR: number of nodes cannot be converted into 3-noded elements")

        mass_iner_y = mass_iner*np.ones((self.num_elem),)
        mass_iner_z = mass_iner*np.ones((self.num_elem),)
        mass_iner_x = 2*mass_iner*np.ones((self.num_elem),)
        pos_cg_B = np.zeros((self.num_elem,3),)

        EA = EA*np.ones((self.num_elem),)
        GJ = GJ*np.ones((self.num_elem),)
        EI = EI*np.ones((self.num_elem),)
        mass_per_unit_length = mass_per_unit_length*np.ones((self.num_elem),)

        poisson_coef = 0.3
        GAy = EA/2.0/(1.0+poisson_coef)
        GAz = EA/2.0/(1.0+poisson_coef)

        EIy = EI
        EIz = EI

        self.generate_uniform_beam(node_pos, mass_per_unit_length, mass_iner_x, mass_iner_y, mass_iner_z, pos_cg_B, EA, GAy, GAz, GJ, EIy, EIz)

    def generate_uniform_beam(self, node_pos, mass_per_unit_length, mass_iner_x, mass_iner_y, mass_iner_z, pos_cg_B, EA, GAy, GAz, GJ, EIy, EIz):

        # Define the properties
        self.cone = 0.0
        self.tilt = 0.0
        self.coordinates = node_pos
        self.create_simple_connectivities()
        self.create_cantilever_BC()
        self.create_frame_of_reference_delta(y_BFoR = 'y_AFoR')
        self.beam_number = np.zeros((self.num_elem), dtype=int)
        self.elem_stiffness = np.linspace(0,self.num_elem-1,self.num_elem, dtype=int)
        self.create_stiff_db(EA, GAy, GAz, GJ, EIy, EIz)
        self.elem_mass = np.linspace(0,self.num_elem-1,self.num_elem, dtype=int)
        self.create_mass_db(mass_per_unit_length, mass_iner_x, mass_iner_y, mass_iner_z, pos_cg_B)
        self.node_structural_twist = np.zeros((self.num_node),)
        self.app_forces = np.zeros((self.num_node,6),)
        self.body_number = np.zeros((self.num_elem), dtype = int)

    def check_StructuralInformation(self):
        # CHECKING
        if(self.elem_stiffness.shape[0]!=self.num_elem):
            sys.exit("ERROR: Element stiffness must be defined for each element")
        if(self.elem_mass.shape[0]!=self.num_elem):
            sys.exit("ERROR: Element mass must be defined for each element")
        if(self.frame_of_reference_delta.shape[0]!=self.num_elem):
            sys.exit("ERROR: The first dimension of FoR does not match the number of elements")
        if(self.frame_of_reference_delta.shape[1]!=self.num_node_elem):
            sys.exit("ERROR: The second dimension of FoR does not match the number of nodes element")
        if(self.frame_of_reference_delta.shape[2]!=3):
            sys.exit("ERROR: The third dimension of FoR must be 3")
        if(self.structural_twist.shape[0]!=self.num_node):
            sys.exit("ERROR: The structural twist must be defined for each node")
        if(self.boundary_conditions.shape[0]!=self.num_node):
            sys.exit("ERROR: The boundary conditions must be defined for each node")
        if(self.beam_number.shape[0]!=self.num_node):
            sys.exit("ERROR: The beam number must be defined for each node")
        if(self.app_forces.shape[0]!=self.num_node):
            sys.exit("ERROR: The first dimension of the applied forces matrix does not match the number of nodes")
        if(self.app_forces.shape[1]!=6):
            sys.exit("ERROR: The second dimension of the applied forces matrix must be 6")

    def generate_fem_file(self):

    	# TODO: check variables that are not defined

        # Writting the file
        with h5.File(route + '/' + case_name + '.fem.h5', 'a') as h5file:
    		# TODO: include something to write only exsisting variables
            h5file.create_dataset('coordinates', data=self.coordinates)
            h5file.create_dataset('connectivities', data=self.connectivities)
            h5file.create_dataset('num_node_elem', data=self.num_node_elem)
            h5file.create_dataset('num_node', data=self.num_node)
            h5file.create_dataset('num_elem', data=self.num_elem)
            h5file.create_dataset('stiffness_db', data=self.stiffness_db)
            h5file.create_dataset('elem_stiffness', data=self.elem_stiffness)
            h5file.create_dataset('mass_db', data=self.mass_db)
            h5file.create_dataset('elem_mass', data=self.elem_mass)
            h5file.create_dataset('frame_of_reference_delta', data=self.frame_of_reference_delta)
            h5file.create_dataset('structural_twist', data=self.node_structural_twist)
            h5file.create_dataset('boundary_conditions', data=self.boundary_conditions)
            h5file.create_dataset('beam_number', data=self.beam_number)
            h5file.create_dataset('app_forces', data=self.app_forces)
            h5file.create_dataset('body_number', data=self.body_number)
            # lumped_mass_nodes_handle = h5file.create_dataset(
            #     'lumped_mass_nodes', data=lumped_mass_nodes)
            # lumped_mass_handle = h5file.create_dataset(
            #     'lumped_mass', data=lumped_mass)
            # lumped_mass_inertia_handle = h5file.create_dataset(
            #     'lumped_mass_inertia', data=lumped_mass_inertia)
            # lumped_mass_position_handle = h5file.create_dataset(
            #     'lumped_mass_position', data=lumped_mass_position)

######################################################################
###############  BLADE AERODYNAMIC INFORMATION  ######################
######################################################################
class AerodynamicInformation():

    def __init__(self):

        # Basic parameters
        self.n_points_camber = 1000
        #self.m_distribution = []
        #self.m_distribution.append('uniform')
        self.m_distribution = 'uniform'

        # Properties to be writen in the h5 file
        self.chord = None
        self.aerodynamic_twist = None
        self.surface_m = np.array([m])
        self.surface_distribution = None
        # self.m_distribution = 'uniform'
        self.aero_node = None
        self.elastic_axis = None
        self.airfoil_distribution = None
        self.airfoils = None

        # Other properties
        self.num_airfoils = None
        self.num_surfaces = None
        self.node_chord = None
        self.node_aero_twist = None
        self.node_elastic_axis = None

    def assembly_aerodynamics(self, *args):
        '''
        This function concatenates aerodynamics to be writen in the same h5 File
        '''

        total_num_airfoils = self.num_airfoils
        total_num_surfaces = self.num_surfaces
        # TODO: check why I only need one definition of m and not one per surface

        for aerodynamics_to_add in args:
            self.chord = np.concatenate((self.chord, aerodynamics_to_add.chord), axis=0)
            self.aerodynamic_twist = np.concatenate((self.aerodynamic_twist, aerodynamics_to_add.aerodynamic_twist), axis=0)
            self.surface_m = np.concatenate((self.surface_m, aerodynamics_to_add.surface_m), axis=0)
            #self.m_distribution.append(aerodynamics_to_add.m_distribution)
            assert self.m_distribution == aerodynamics_to_add.m_distribution, "m_distribution does not match"
            self.surface_distribution = np.concatenate((self.surface_distribution, aerodynamics_to_add.surface_distribution + total_num_surfaces), axis=0)
            self.aero_node = np.concatenate((self.aero_node, aerodynamics_to_add.aero_node), axis=0)
            self.elastic_axis = np.concatenate((self.elastic_axis, aerodynamics_to_add.elastic_axis), axis=0)
            # np.concatenate((self.airfoil_distribution, aerodynamics_to_add.airfoil_distribution), axis=0)
            self.airfoil_distribution = np.concatenate((self.airfoil_distribution, aerodynamics_to_add.airfoil_distribution + total_num_airfoils), axis=0)
            self.airfoils = np.concatenate((self.airfoils, aerodynamics_to_add.airfoils), axis=0)

            total_num_airfoils += aerodynamics_to_add.num_airfoils
            total_num_surfaces += aerodynamics_to_add.num_surfaces

        self.num_airfoils = total_num_airfoils
        self.num_surfaces = total_num_surfaces

    def interpolate_airfoils_camber(self, pure_airfoils_camber, r_pure_airfoils, r):

        num_node = len(r)
        airfoils_camber = np.zeros((num_node,self.n_points_camber,2),)

        for inode in range(num_node):
            # camber_x, camber_y = get_airfoil_camber(x,y)

            iairfoil=0
            while(r[inode]<r_pure_airfoils[iairfoil]):
                iairfoil+=1
                if(iairfoil==len(r_pure_airfoils)):
                    iairfoil-=1
                    break

            beta=min((r[inode]-r_pure_airfoils[iairfoil-1])/(r_pure_airfoils[iairfoil]-r_pure_airfoils[iairfoil-1]),1.0)
            beta=max(0.0,beta)

            airfoils_camber[inode,:,0]=(1-beta)*pure_airfoils_camber[iairfoil-1,:,0]+beta*pure_airfoils_camber[iairfoil,:,0]
            airfoils_camber[inode,:,1]=(1-beta)*pure_airfoils_camber[iairfoil-1,:,1]+beta*pure_airfoils_camber[iairfoil,:,1]

        return airfoils_camber

    def check_AerodynamicInformation(self, StructuralInformation):

        # CHECKING
        if(self.aero_node.shape[0] != StructuralInformation.num_node):
            sys.exit("ERROR: Aero node must be defined for each node")
        if(self.airfoil_distribution.shape[0] != StructuralInformation.num_elem or self.airfoil_distribution.shape[1]!=StructuralInformation.num_node_elem):
            sys.exit("ERROR: Airfoil distribution must be defined for each element/local node")
        if(self.chord.shape[0] != StructuralInformation.num_elem):
            sys.exit("ERROR: The first dimension of the chord matrix does not match the number of elements")
        if(self.chord.shape[1] != StructuralInformation.num_node_elem):
            sys.exit("ERROR: The second dimension of the chord matrix does not match the number of nodes per element")
        if(self.elastic_axis.shape[0] != StructuralInformation.num_elem):
            sys.exit("ERROR: The first dimension of the elastic axis matrix does not match the number of elements")
        if(self.elastic_axis.shape[1] != StructuralInformation.num_node_elem):
            sys.exit("ERROR: The second dimension of the elastic axis matrix does not match the number of nodes per element")
        if(self.surface_distribution.shape[0] != StructuralInformation.num_elem):
            sys.exit("ERROR: The surface distribution must be defined for each element")
        if(self.aerodynamic_twist.shape[0] != StructuralInformation.num_elem):
            sys.exit("ERROR: The first dimension of the aerodynamic twist does not match the number of elements")
        if(self.aerodynamic_twist.shape[1] != StructuralInformation.num_node_elem):
            sys.exit("ERROR: The second dimension of the aerodynamic twist does not match the number nodes per element")

    def generate_aero_file(self):

        with h5.File(route + '/' + case_name + '.aero.h5', 'a') as h5file:

            h5file.create_dataset('aero_node', data=self.aero_node)
            chord_input = h5file.create_dataset('chord', data=self.chord)
            chord_input .attrs['units'] = 'm'
            twist_input = h5file.create_dataset('twist', data=self.aerodynamic_twist)
            twist_input.attrs['units'] = 'rad'
            h5file.create_dataset('surface_m', data=self.surface_m)
            h5file.create_dataset('surface_distribution', data=self.surface_distribution)
            h5file.create_dataset('m_distribution', data=self.m_distribution.encode('ascii', 'ignore'))
            h5file.create_dataset('elastic_axis', data=self.elastic_axis)
            h5file.create_dataset('airfoil_distribution', data=self.airfoil_distribution)

            airfoils_group = h5file.create_group('airfoils')
            for iairfoil in range(len(self.airfoils)):
                airfoils_group.create_dataset("%d" % iairfoil, data=self.airfoils[iairfoil,:,:])

            #control_surface_input = h5file.create_dataset('control_surface', data=control_surface)
            #control_surface_deflection_input = h5file.create_dataset('control_surface_deflection', data=control_surface_deflection)
            #control_surface_chord_input = h5file.create_dataset('control_surface_chord', data=control_surface_chord)
            #control_surface_types_input = h5file.create_dataset('control_surface_type', data=control_surface_type)

######################################################################
###############  BLADE AEROELASTIC INFORMATION  ######################
######################################################################
class AeroelasticInformation():

    def __init__(self):

        self.StructuralInformation = StructuralInformation()
        self.AerodynamicInformation = AerodynamicInformation()

    def generate(self, StructuralInformation, AerodynamicInformation):

        self.StructuralInformation = StructuralInformation.copy()
        self.AerodynamicInformation = AerodynamicInformation.copy()

    def assembly(self, *args):

        list_of_SI = []
        list_of_AI = []

        for AEI in args:
            list_of_SI.append(AEI.StructuralInformation)
            list_of_AI.append(AEI.AerodynamicInformation)

        self.StructuralInformation.assembly_structures(*list_of_SI)
        self.AerodynamicInformation.assembly_aerodynamics(*list_of_AI)

    def copy(self):

        copied = AeroelasticInformation()

        copied.StructuralInformation = self.StructuralInformation.copy()
        copied.AerodynamicInformation = self.AerodynamicInformation.copy()

        return copied

    def write_h5_files(self):

        self.StructuralInformation.generate_fem_file()
        self.AerodynamicInformation.generate_aero_file()

    def define_no_aerodynamics(self, StructuralInformation):

        self.AerodynamicInformation.num_airfoils = 1
        self.AerodynamicInformation.num_surfaces = 0

        self.AerodynamicInformation.aero_node = np.zeros((StructuralInformation.num_node,), dtype = bool)
        self.AerodynamicInformation.chord = np.zeros((StructuralInformation.num_elem,StructuralInformation.num_node_elem),)
        self.AerodynamicInformation.aerodynamic_twist = np.zeros((StructuralInformation.num_elem,StructuralInformation.num_node_elem),)
        # TODO: SHARPy does not ignore the surface_m when the surface is not aerodynamic
        #self.AerodynamicInformation.surface_m = np.array([0], dtype = int)
        self.AerodynamicInformation.surface_m = np.array([], dtype=int)
        self.AerodynamicInformation.surface_distribution = np.zeros((StructuralInformation.num_elem,), dtype=int) - 1
        self.AerodynamicInformation.m_distribution = 'uniform'
        self.AerodynamicInformation.elastic_axis = np.zeros((StructuralInformation.num_elem,StructuralInformation.num_node_elem),)
        self.AerodynamicInformation.airfoil_distribution = np.zeros((StructuralInformation.num_elem,StructuralInformation.num_node_elem), dtype=int)
        self.AerodynamicInformation.airfoils = np.zeros((1,self.AerodynamicInformation.n_points_camber,2),)
        self.AerodynamicInformation.airfoils[0,:,0] = np.linspace(0.0, 1.0, self.AerodynamicInformation.n_points_camber)

    def define_basic_aerodynamics(self, StructuralInformation):

        self.AerodynamicInformation.num_airfoils = 1
        self.AerodynamicInformation.num_surfaces = 1

        self.AerodynamicInformation.aero_node = np.ones((StructuralInformation.num_node,), dtype = bool)
        self.AerodynamicInformation.chord = np.ones((StructuralInformation.num_elem,StructuralInformation.num_node_elem),)
        self.AerodynamicInformation.aerodynamic_twist = 0.0*deg2rad*np.ones((StructuralInformation.num_elem,StructuralInformation.num_node_elem),)
        # TODO: SHARPy does not ignore the surface_m when the surface is not aerodynamic
        #self.AerodynamicInformation.surface_m = np.array([0], dtype = int)
        self.AerodynamicInformation.surface_m = np.array([1], dtype=int)
        self.AerodynamicInformation.surface_distribution = np.zeros((StructuralInformation.num_elem,), dtype=int)
        self.AerodynamicInformation.m_distribution = 'uniform'
        self.AerodynamicInformation.elastic_axis = 0.25*np.ones((StructuralInformation.num_elem,StructuralInformation.num_node_elem),)
        self.AerodynamicInformation.airfoil_distribution = np.zeros((StructuralInformation.num_elem,StructuralInformation.num_node_elem), dtype=int)
        self.AerodynamicInformation.airfoils = np.zeros((1,self.AerodynamicInformation.n_points_camber,2),)
        self.AerodynamicInformation.airfoils[0,:,0] = np.linspace(0.0, 1.0, self.AerodynamicInformation.n_points_camber)

    def remove_clamping(self, keep=None):

        if keep == None:
            # I keep the first clamped node and remove the rest
            is_first_clamped = True
            for inode in range(self.StructuralInformation.num_node):
                if((self.StructuralInformation.boundary_conditions[inode] == 1) and (is_first_clamped)):
                    is_first_clamped = False
                elif(self.StructuralInformation.boundary_conditions[inode] == 1):
                    self.StructuralInformation.boundary_conditions[inode] = -1

        else:
            for inode in self.StructuralInformation.num_node:
                if((self.StructuralInformation.boundary_conditions[inode] == 1) and (not inode in keep)):
                    self.StructuralInformation.boundary_conditions[inode] = -1

######################################################################
###################  SIMULATION INFORMATION  #########################
######################################################################
class SimulationInformation():

    def __init__(self):

        self.WSP = None
        self.airDensity = None
        self.omega = None

        self.dt = None
        self.n_tstep = None
        self.alpha = None
        self.beta = None

        self.angular_velocity = None

        self.dynamic_forces_time = None
        self.forced_for_vel = None
        self.with_dynamic_forces = False
        self.with_forced_vel = False

    def generate_basic(self, WSP=10.0, airDensity=0.0, dt=0.05, n_tstep=10, alpha=0.0, beta=0.0):

        self.WSP = WSP
        self.airDensity = airDensity

        self.dt = dt
        self.n_tstep = n_tstep
        self.alpha = alpha
        self.beta = beta

    def generate_dyn_file(self):

        if self.with_forced_vel:
            self.forced_for_vel = np.zeros((self.n_tstep, 6))
            self.forced_for_acc = np.zeros((self.n_tstep, 6))
            for it in range(self.n_tstep):
                # forced_for_vel[it, 3:6] = it/n_tstep*angular_velocity
                # self.forced_for_vel[it, 3:6] = self.angular_velocity
                # The omega is not applied to the AFoR, it will be included through Lagrange multipliers
                self.forced_for_vel[it, 3:6] = np.zeros((1,3),)

        with h5.File(route + '/' + case_name + '.dyn.h5', 'a') as h5file:
            if self.with_dynamic_forces:
                h5file.create_dataset(
                    'dynamic_forces', data=self.dynamic_forces_time)
            if self.with_forced_vel:
                # TODO: check coherence velocity-acceleration
                h5file.create_dataset(
                    'for_vel', data=self.forced_for_vel)
                h5file.create_dataset(
                    'for_acc', data=self.forced_for_acc)
            h5file.create_dataset(
                'num_steps', data=self.n_tstep)

    def generate_solver_file(self):
        file_name = route + '/' + case_name + '.solver.txt'
        settings = dict()
        aux_settings = dict()
        settings['SHARPy'] = {'case': case_name,
                              'route': route,
                              'flow': flow,
                              'write_screen': 'on',
                              'write_log': 'on',
                              'log_folder': route + '/output/',
                              'log_file': case_name + '.log'}

        # AUX DICTIONARIES
        aux_settings['velocity_field_input'] = {'u_inf': self.WSP,
                                            'u_inf_direction': [0.0, 1.0, 0.0]}

        # LOADERS

        settings['BeamLoader'] = {'unsteady': 'on',
                                  'orientation': algebra.euler2quat(np.array([0.0,
                                                                              self.alpha,
                                                                              self.beta]))}

        settings['AerogridLoader'] = {'unsteady': 'on',
                                      'aligned_grid': 'on',
                                      'mstar': mstar,
                                      'freestream_dir': ['0.0', '1.0', '0.0']}

        # POSTPROCESS

        settings['AerogridPlot'] = {'folder': route + '/output/',
                                    'include_rbm': 'on',
                                    'include_forward_motion': 'off',
                                    'include_applied_forces': 'on',
                                    'minus_m_star': 0,
                                    'u_inf': self.WSP,
                                    'dt': self.dt}

        #settings['AeroForcesCalculator'] = {'folder': route + '/output/forces',
                                            #'write_text_file': 'on',
                                            #'text_file_name': case_name + '_aeroforces.csv',
                                            #'screen_output': 'on',
                                            #'unsteady': 'off'}

        settings['BeamPlot'] = {'folder': route + '/output/',
                                'include_rbm': 'on',
                                'include_applied_forces': 'on',
                                'include_forward_motion': 'on'}

        #settings['BeamCsvOutput'] = {'folder': route + '/output/',
                                     #'output_pos': 'on',
                                     #'output_psi': 'on',
                                     #'screen_output': 'off'}

        aux_settings['BeamLoads'] = {}

        aux_settings['WriteVariablesTime'] = {'delimiter': ' ',
                                          'FoR_variables': ['GFoR_pos', 'GFoR_vel', 'GFoR_acc', 'GFoR_quat'],
                                          'FoR_number': [0,1],
                                          'structure_variables': ['AFoR_position'],
                                          'structure_nodes': [10],
                                          'aero_panels_variables': [],
                                          'aero_panels_isurf': [],
                                          'aero_panels_im': [],
                                          'aero_panels_in': [],
                                          'aero_nodes_variables': [],
                                          'aero_nodes_isurf': [],
                                          'aero_nodes_im': [],
                                          'aero_nodes_in': []}

        # STATIC COUPLED

        aux_settings['NonLinearStatic'] = {'print_info': 'on',
                                       'max_iterations': 150,
                                       'num_load_steps': 1,
                                       'delta_curved': 1e-15,
                                       'min_delta': 1e-8,
                                       'gravity_on': gravity,
                                       'gravity': 9.81}

        aux_settings['StaticUvlm'] = {'print_info': 'on',
                                  'horseshoe': 'off',
                                  'num_cores': 4,
                                  'n_rollup': 0,
                                  'rollup_dt': self.dt,
                                  'rollup_aic_refresh': 1,
                                  'rollup_tolerance': 1e-4,
                                  'velocity_field_generator': 'SteadyVelocityField',
                                  'velocity_field_input': aux_settings['velocity_field_input'],
                                  'rho': 0.0}

        settings['StaticCoupled'] = {'print_info': 'on',
                                     'structural_solver': 'NonLinearStatic',
                                     'structural_solver_settings': aux_settings['NonLinearStatic'],
                                     'aero_solver': 'StaticUvlm',
                                     'aero_solver_settings': aux_settings['StaticUvlm'],
                                     'max_iter': 100,
                                     'n_load_steps': 1,
                                     'tolerance': 1e-8,
                                     'relaxation_factor': 0}

        # DYNAMIC PRESCRIBED COUPLED

        aux_settings['NonLinearDynamicCoupledStep'] = {'print_info': 'on',
                                                   'max_iterations': 95000,
                                                   'delta_curved': 1e-9,
                                                   'min_delta': 1e-6,
                                                   'newmark_damp': 1e-2,
                                                   'gravity_on': gravity,
                                                   'gravity': 9.81,
                                                   'num_steps': self.n_tstep,
                                                   'dt': self.dt}

        aux_settings['NonLinearDynamicMultibody'] = {'print_info': 'on',
                                                   'max_iterations': 95000,
                                                   'delta_curved': 1e-9,
                                                   'min_delta': 1e-8,
                                                   'newmark_damp': 1e-2,
                                                   'gravity_on': gravity,
                                                   'gravity': 9.81,
                                                   'num_steps': self.n_tstep,
                                                   'dt': self.dt,
                                                   'postprocessors': ['BeamPlot', 'AerogridPlot', 'WriteVariablesTime'],
                                                   'postprocessors_settings': {'BeamPlot': settings['BeamPlot'],
                                                                               'AerogridPlot': settings['AerogridPlot'],
                                                                               'WriteVariablesTime': aux_settings['WriteVariablesTime']}}

        aux_settings['StepUvlm'] = {'print_info': 'on',
                                'horseshoe': 'off',
                                'num_cores': 4,
                                'n_rollup': 0,
                                'convection_scheme': 2,
                                'rollup_dt': self.dt,
                                'rollup_aic_refresh': 1,
                                'rollup_tolerance': 1e-4,
                                'velocity_field_generator': 'SteadyVelocityField',
                                'velocity_field_input': aux_settings['velocity_field_input'],
                                'rho': 0.0,
                                'n_time_steps': self.n_tstep,
                                'dt': self.dt}

        settings['DynamicPrescribedCoupled'] = {'structural_solver': 'NonLinearDynamicMultibody',
                                                'structural_solver_settings': aux_settings['NonLinearDynamicMultibody'],
                                                'aero_solver': 'StepUvlm',
                                                'aero_solver_settings': aux_settings['StepUvlm'],
                                                'fsi_substeps': 20000,
                                                'fsi_tolerance': 1e-5,
                                                'fsi_vel_tolerance': 1e-3,
                                                'relaxation_factor': 0,
                                                'minimum_steps': 1,
                                                'relaxation_steps': 150,
                                                'final_relaxation_factor': 0.0,
                                                'n_time_steps': self.n_tstep,
                                                'dt': self.dt,
                                                'include_unsteady_force_contribution': 'on',
                                                'print_info': 'on',
                                                # 'postprocessors': ['BeamPlot', 'AerogridPlot'],
                                                # 'postprocessors_settings': {'BeamPlot': settings['BeamPlot'],
                                                #                             'AerogridPlot': settings['AerogridPlot']}}
                                                'postprocessors': ['BeamPlot', 'AerogridPlot', 'WriteVariablesTime'],
                                                'postprocessors_settings': {'BeamPlot': settings['BeamPlot'],
                                                                            'AerogridPlot': settings['AerogridPlot'],
                                                                            'WriteVariablesTime': aux_settings['WriteVariablesTime']}}


        # STEADY HELICOIDAL WAKE
        # settings['SteadyHelicoidalWake'] = settings['DynamicCoupled']




        import configobj
        config = configobj.ConfigObj()
        config.filename = file_name
        # for k, v in settings.items():
        #     config[k] = v
        for k in settings.['SHARPy']['flow']:
            config[k] = settings.[k]
        config.write()

######################################################################
#########################  CLEAN FILES  ##############################
######################################################################

def clean_test_files():
    fem_file_name = route + '/' + case_name + '.fem.h5'
    if os.path.isfile(fem_file_name):
        os.remove(fem_file_name)

    dyn_file_name = route + '/' + case_name + '.dyn.h5'
    if os.path.isfile(dyn_file_name):
        os.remove(dyn_file_name)

    aero_file_name = route + '/' + case_name + '.aero.h5'
    if os.path.isfile(aero_file_name):
        os.remove(aero_file_name)

    lagrange_file_name = route + '/' + case_name + '.mb.h5'
    if os.path.isfile(lagrange_file_name):
        os.remove(lagrange_file_name)

    solver_file_name = route + '/' + case_name + '.solver.txt'
    if os.path.isfile(solver_file_name):
        os.remove(solver_file_name)

    flightcon_file_name = route + '/' + case_name + '.flightcon.txt'
    if os.path.isfile(flightcon_file_name):
        os.remove(flightcon_file_name)
