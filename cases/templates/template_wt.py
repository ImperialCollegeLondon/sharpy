"""
template_wt

Functions needed to generate a wind turbines

Notes:
    To load this library: import cases.templates.template_wt as template_wt
"""

import sharpy.utils.generate_cases as gc
import pandas as pd
import numpy as np
import scipy.interpolate as scint
import math
import os
import sharpy.utils.algebra as algebra
import sharpy.utils.h5utils as h5


deg2rad = np.pi/180.

######################################################################
# AUX FUNCTIONS
######################################################################

def create_node_radial_pos_from_elem_centres(root_elem_centres_tip, num_node, num_elem, num_node_elem):
    """
    create_node_radial_pos_from_elem_centres

    Define the position of the nodes adn the elements in the blade from the list of element centres

    Args:
        root_elem_centres_tip (np.array):
            - First value: Radial position of the beginning of the blade
            - Last value: Radial position of the tip of the blade
            - Rest of the values: Radial position the rest of the strucutral element centres
        num_node (int): number of nodes
        num_elem (int): number of elements
        num_node_elem (int): number of nodes in each element

    Returns:
        node_r (np.array): Radial position of the nodes
        elem_r (np.array): Radial position of the elements

    Notes:
        Radial positions are measured from the hub centre and measured in the rotation plane
    """

    elem_r = root_elem_centres_tip[1:-1]
    node_r = np.zeros((num_node, ), )
    node_r[0] = root_elem_centres_tip[0]
    node_r[-2] = root_elem_centres_tip[-2]
    node_r[-1] = root_elem_centres_tip[-1]

    for ielem in range(num_elem-1):
        node_r[ielem*(num_node_elem-1)+1] = elem_r[ielem]
        node_r[ielem*(num_node_elem-1)+2] = 0.5*(elem_r[ielem]+elem_r[ielem+1])

    return node_r, elem_r


def create_blade_coordinates(num_node, node_r, node_y, node_z):
    """
    create_blade_coordinates

    Creates SHARPy format of the nodes coordinates and
    applies prebending and presweept to node radial position

    Args:
        num_node (int): number of nodes
        node_r (np.array): Radial position of the nodes
        node_y (np.array): Displacement of each point IN the rotation plane
        node_z (np.array): Displacement of each point OUT OF the rotation plane

    Returns:
        coordinates (np.array): nodes coordinates
    """
    coordinates = np.zeros((num_node,3),)
    coordinates[:,0] = node_r
    coordinates[:,1] = node_y
    coordinates[:,2] = node_z
    return coordinates

######################################################################
# FROM EXCEL TYPE 01
######################################################################
def generate_from_excel_type01(chord_panels,
                                  rotation_velocity,
                                  pitch,
                                  excel_file_name= 'database_type01.xlsx',
                                  excel_sheet_structural_blade = 'structural_blade',
                                  excel_sheet_aero_blade = 'aero_blade',
                                  excel_sheet_airfoil_coord = 'airfoil_coord',
                                  excel_sheet_rotor = 'rotor_parameters',
                                  excel_sheet_structural_tower = 'structural_tower',
                                  excel_sheet_nacelle = 'structural_nacelle',
                                  m_distribution = 'uniform',
                                  n_points_camber = 100,
                                  tol_remove_points = 1e-3):

    """
    generate_wt_from_excel_type01

    Function needed to generate a wind turbine from an excel database of type 01 (FAST format)

    Args:
    	  chord_panels (int): Number of panels on the blade surface in the chord direction
          rotation_velocity (float): Rotation velocity of the rotor
          pitch (float): pitch angle in degrees
          excel_file_name (str):
          excel_sheet_structural_blade (str):
          excel_sheet_aero_blade (str):
          excel_sheet_airfoil_coord (str):
          excel_sheet_rotor (str):
          excel_sheet_structural_tower (str):
          excel_sheet_nacelle (str):
          m_distribution (str):
          n_points_camber (int): number of points to define the camber of the airfoil,
          tol_remove_points (float): maximum distance to remove adjacent points

    Returns:
        wt (sharpy.utils.generate_cases.AeroelasticInfromation): Aeroelastic infrmation of the wind turbine
        LC (list): list of all the Lagrange constraints needed in the cases (sharpy.utils.generate_cases.LagrangeConstraint)
        MB (list): list of the multibody information of each body (sharpy.utils.generate_cases.BodyInfrmation)
    """

    ######################################################################
    ## BLADE
    ######################################################################

    blade = gc.AeroelasticInformation()

    ######################################################################
    ### STRUCTURE
    ######################################################################
    # Read blade structural information from excel file
    Radius = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'Radius')
    BlFract = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'BlFract')
    AeroCent= gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'AeroCent')
    # TODO: implement aerocent
    print("WARNING: AeroCent not implemented")
    StrcTwst= (gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'StrcTwst') + pitch)*deg2rad
    BMassDen= gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'BMassDen')
    FlpStff= gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'FlpStff')
    EdgStff = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'EdgStff')
    GJStff = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'GJStff')
    EAStff = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'EAStff')
    Alpha = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'Alpha')
    FlpIner= gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'FlpIner')
    EdgIner= gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'EdgIner')
    PrecrvRef = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'PrecrvRef')
    PreswpRef = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'PreswpRef')
    FlpcgOf = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'FlpcgOf')
    EdgcgOf = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'EdgcgOf')
    FlpEAOf = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'FlpEAOf')
    EdgEAOf = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'EdgEAOf')

    # Base parameters
    blade.StructuralInformation.num_elem = len(Radius) - 2
    blade.StructuralInformation.num_node_elem = 3
    blade.StructuralInformation.compute_basic_num_node()

    # Interpolate excel variables into the correct locations
    # Geometry
    node_r, elem_r = create_node_radial_pos_from_elem_centres(Radius,
                                        blade.StructuralInformation.num_node,
                                        blade.StructuralInformation.num_elem,
                                        blade.StructuralInformation.num_node_elem)
    node_prebending = np.interp(node_r,Radius,PrecrvRef)
    node_presweept = np.interp(node_r,Radius,PreswpRef)
    node_structural_twist = -1.0*np.interp(node_r,Radius,StrcTwst)
    # Stiffness
    elem_EA = np.interp(elem_r,Radius,EAStff)
    elem_EIy = np.interp(elem_r,Radius,FlpStff)
    elem_EIz = np.interp(elem_r,Radius,EdgStff)
    elem_GJ = np.interp(elem_r,Radius,GJStff)
    # Stiffness: estimate unknown properties
    print('WARNING: The poisson cofficient is assumed equal to 0.3')
    print('WARNING: Cross-section area is used as shear area')
    poisson_coef = 0.3
    elem_GAy = elem_EA/2.0/(1.0+poisson_coef)
    elem_GAz = elem_EA/2.0/(1.0+poisson_coef)
    # Inertia
    # TODO: check yz axis and Flap-edge
    elem_pos_cg_B = np.zeros((blade.StructuralInformation.num_elem,3),)
    elem_pos_cg_B[:,2]=np.interp(elem_r,Radius,FlpcgOf)
    elem_pos_cg_B[:,1]=np.interp(elem_r,Radius,EdgcgOf)

    elem_mass_per_unit_length = np.interp(elem_r,Radius,BMassDen)
    elem_mass_iner_y = np.interp(elem_r,Radius,FlpIner)
    elem_mass_iner_z = np.interp(elem_r,Radius,EdgIner)

    # Inertia: estimate unknown properties
    print('WARNING: Using perpendicular axis theorem to compute the inertia around xB')
    elem_mass_iner_x = elem_mass_iner_y + elem_mass_iner_z

    # Generate blade structural properties
    blade.StructuralInformation.create_mass_db_from_vector(elem_mass_per_unit_length, elem_mass_iner_x, elem_mass_iner_y, elem_mass_iner_z, elem_pos_cg_B)
    blade.StructuralInformation.create_stiff_db_from_vector(elem_EA, elem_GAy, elem_GAz, elem_GJ, elem_EIy, elem_EIz)

    coordinates = create_blade_coordinates(blade.StructuralInformation.num_node, node_r, node_prebending, node_presweept)

    blade.StructuralInformation.generate_1to1_from_vectors(
            num_node_elem = blade.StructuralInformation.num_node_elem,
            num_node = blade.StructuralInformation.num_node,
            num_elem = blade.StructuralInformation.num_elem,
            coordinates = coordinates,
            stiffness_db = blade.StructuralInformation.stiffness_db,
            mass_db = blade.StructuralInformation.mass_db,
            frame_of_reference_delta = 'y_AFoR',
            vec_node_structural_twist = node_structural_twist,
            num_lumped_mass = 0)

    # Boundary conditions
    blade.StructuralInformation.boundary_conditions = np.zeros((blade.StructuralInformation.num_node), dtype = int)
    blade.StructuralInformation.boundary_conditions[0] = 1
    blade.StructuralInformation.boundary_conditions[-1] = -1

    ######################################################################
    ### AERODYNAMICS
    ######################################################################
    # Read blade aerodynamic information from excel file
    excel_aero_r = gc.read_column_sheet_type01(excel_file_name, excel_sheet_aero_blade, 'Rnodes')
    excel_aerodynamic_twist = gc.read_column_sheet_type01(excel_file_name, excel_sheet_aero_blade, 'AeroTwst')*deg2rad
    excel_chord = gc.read_column_sheet_type01(excel_file_name, excel_sheet_aero_blade, 'Chord')
    pure_airfoils_names = gc.read_column_sheet_type01(excel_file_name, excel_sheet_aero_blade, 'Airfoil_Table')

    # Read coordinates of the pure airfoils
    n_elem_aero = len(excel_aero_r)
    # TODO: change this with a list of thickness and pure airfoils
    pure_airfoils_camber=np.zeros((n_elem_aero,n_points_camber,2),)
    xls = pd.ExcelFile(excel_file_name)
    excel_db = pd.read_excel(xls, sheet_name=excel_sheet_airfoil_coord)
    for iairfoil in range(len(pure_airfoils_names)):
        # Look for the NaN
        icoord=2
        while(not(math.isnan(excel_db["%s_x" % pure_airfoils_names[iairfoil]][icoord]))):
            icoord+=1
            if(icoord==len(excel_db["%s_x" % pure_airfoils_names[iairfoil]])):
                break

        # Compute the camber of the airfoil
        pure_airfoils_camber[iairfoil,:,0], pure_airfoils_camber[iairfoil,:,1] = gc.get_airfoil_camber(excel_db["%s_x" % pure_airfoils_names[iairfoil]][2:icoord] , excel_db["%s_y" % pure_airfoils_names[iairfoil]][2:icoord], n_points_camber)

    # Basic variables
    n_elem_aero = len(excel_aero_r)
    num_airfoils = blade.StructuralInformation.num_node
    surface_distribution = np.zeros((blade.StructuralInformation.num_elem), dtype=int)

    # Interpolate in the correct positions
    node_chord=np.interp(node_r, excel_aero_r, excel_chord)
    node_aero_twist = -1.0*(np.interp(node_r, excel_aero_r, excel_aerodynamic_twist) + node_structural_twist)
    node_sweep = np.ones((blade.StructuralInformation.num_node), )*np.pi
    node_elastic_axis=np.ones((blade.StructuralInformation.num_node,))*0.25

    # Define the nodes with aerodynamic properties
    # Look for the first element that is goint to be aerodynamic
    first_aero_elem=0
    while (elem_r[first_aero_elem]<=excel_aero_r[0]):
        first_aero_elem+=1
    first_aero_node=first_aero_elem*(blade.StructuralInformation.num_node_elem-1)
    aero_node = np.zeros((blade.StructuralInformation.num_node,), dtype=bool)
    aero_node[first_aero_node:]=np.ones((blade.StructuralInformation.num_node-first_aero_node,),dtype=bool)

    airfoils = blade.AerodynamicInformation.interpolate_airfoils_camber(pure_airfoils_camber,excel_aero_r, node_r, n_points_camber)

    # Write SHARPy format
    airfoil_distribution = np.linspace(0,blade.StructuralInformation.num_node-1,blade.StructuralInformation.num_node, dtype=int)

    blade.AerodynamicInformation.create_aerodynamics_from_vec(blade.StructuralInformation,
                                                            aero_node,
                                                            node_chord,
                                                            node_aero_twist,
                                                            node_sweep,
                                                            chord_panels,
                                                            surface_distribution,
                                                            m_distribution,
                                                            node_elastic_axis,
                                                            airfoil_distribution,
                                                            airfoils)

    ######################################################################
    ## ROTOR
    ######################################################################

    # Read from excel file
    numberOfBlades = gc.read_column_sheet_type01(excel_file_name, excel_sheet_rotor, 'NumberOfBlades')
    tilt = gc.read_column_sheet_type01(excel_file_name, excel_sheet_rotor, 'Tilt')*deg2rad
    cone = gc.read_column_sheet_type01(excel_file_name, excel_sheet_rotor, 'Cone')*deg2rad
    # pitch = gc.read_column_sheet_type01(excel_file_name, excel_sheet_rotor, 'Pitch')*deg2rad

    # Apply coning
    blade.StructuralInformation.rotate_around_origin(np.array([0.,1.,0.]), cone)

    # Build the whole rotor
    rotor = blade.copy()
    for iblade in range(numberOfBlades-1):
        blade2 = blade.copy()
        blade2.StructuralInformation.rotate_around_origin(np.array([0.,0.,1.]), (iblade+1)*(360.0/numberOfBlades)*deg2rad)
        rotor.assembly(blade2)
        blade2 = None

    rotor.remove_duplicated_points(tol_remove_points)

    # Apply tilt
    rotor.StructuralInformation.rotate_around_origin(np.array([0.,1.,0.]), -tilt)

    ######################################################################
    ## TOWER
    ######################################################################

    # Read from excel file
    Elevation = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_tower, 'Elevation')
    TMassDen = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_tower, 'TMassDen')
    TwFAStif = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_tower, 'TwFAStif')
    TwSSStif = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_tower, 'TwSSStif')
    TwGJStif = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_tower, 'TwGJStif')
    TwEAStif = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_tower, 'TwEAStif')
    TwFAIner = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_tower, 'TwFAIner')
    TwSSIner = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_tower, 'TwSSIner')
    TwFAcgOf = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_tower, 'TwFAcgOf')
    TwSScgOf = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_tower, 'TwSScgOf')

    # Define the TOWER
    tower = gc.AeroelasticInformation()

    tower.StructuralInformation.num_elem = len(Elevation) - 2
    tower.StructuralInformation.num_node_elem = 3
    tower.StructuralInformation.compute_basic_num_node()

    # Interpolate excel variables into the correct locations
    node_r, elem_r = create_node_radial_pos_from_elem_centres(Elevation,
                                        tower.StructuralInformation.num_node,
                                        tower.StructuralInformation.num_elem,
                                        tower.StructuralInformation.num_node_elem)

    # Stiffness
    elem_EA = np.interp(elem_r,Elevation,TwEAStif)
    elem_EIz = np.interp(elem_r,Elevation,TwSSStif)
    elem_EIy = np.interp(elem_r,Elevation,TwFAStif)
    elem_GJ = np.interp(elem_r,Elevation,TwGJStif)
    # Stiffness: estimate unknown properties
    print('WARNING: The poisson cofficient is assumed equal to 0.3')
    print('WARNING: Cross-section area is used as shear area')
    poisson_coef = 0.3
    elem_GAy = elem_EA/2.0/(1.0+poisson_coef)
    elem_GAz = elem_EA/2.0/(1.0+poisson_coef)

    # Inertia
    elem_mass_per_unit_length = np.interp(elem_r,Elevation,TMassDen)
    elem_mass_iner_y = np.interp(elem_r,Elevation,TwFAIner)
    elem_mass_iner_z = np.interp(elem_r,Elevation,TwSSIner)
    # TODO: check yz axis and Flap-edge
    elem_pos_cg_B = np.zeros((tower.StructuralInformation.num_elem,3),)
    elem_pos_cg_B[:,1]=np.interp(elem_r,Elevation,TwSScgOf)
    elem_pos_cg_B[:,2]=np.interp(elem_r,Elevation,TwFAcgOf)

    # Stiffness: estimate unknown properties
    print('WARNING: Using perpendicular axis theorem to compute the inertia around xB')
    elem_mass_iner_x = elem_mass_iner_y + elem_mass_iner_z

    # Create the tower
    tower.StructuralInformation.create_mass_db_from_vector(elem_mass_per_unit_length, elem_mass_iner_x, elem_mass_iner_y, elem_mass_iner_z, elem_pos_cg_B)
    tower.StructuralInformation.create_stiff_db_from_vector(elem_EA, elem_GAy, elem_GAz, elem_GJ, elem_EIy, elem_EIz)

    coordinates = np.zeros((tower.StructuralInformation.num_node,3),)
    coordinates[:,0] = node_r

    tower.StructuralInformation.generate_1to1_from_vectors(
            num_node_elem = tower.StructuralInformation.num_node_elem,
            num_node = tower.StructuralInformation.num_node,
            num_elem = tower.StructuralInformation.num_elem,
            coordinates = coordinates,
            stiffness_db = tower.StructuralInformation.stiffness_db,
            mass_db = tower.StructuralInformation.mass_db,
            frame_of_reference_delta = 'y_AFoR',
            vec_node_structural_twist = np.zeros((tower.StructuralInformation.num_node,),),
            num_lumped_mass = 1)

    tower.StructuralInformation.boundary_conditions = np.zeros((tower.StructuralInformation.num_node), dtype = int)
    tower.StructuralInformation.boundary_conditions[0] = 1

    # Read overhang and nacelle properties from excel file
    overhang_len = gc.read_column_sheet_type01(excel_file_name, excel_sheet_nacelle, 'overhang')
    HubMass = gc.read_column_sheet_type01(excel_file_name, excel_sheet_nacelle, 'HubMass')
    NacelleMass = gc.read_column_sheet_type01(excel_file_name, excel_sheet_nacelle, 'NacelleMass')
    NacelleYawIner = gc.read_column_sheet_type01(excel_file_name, excel_sheet_nacelle, 'NacelleYawIner')

    # Include nacelle mass
    tower.StructuralInformation.lumped_mass_nodes = np.array([tower.StructuralInformation.num_node-1], dtype=int)
    tower.StructuralInformation.lumped_mass = np.array([NacelleMass], dtype=float)

    tower.AerodynamicInformation.set_to_zero(tower.StructuralInformation.num_node_elem,
                                            tower.StructuralInformation.num_node,
                                            tower.StructuralInformation.num_elem)

    # Assembly overhang with the tower
    overhang = gc.AeroelasticInformation()
    overhang.StructuralInformation.num_node = 3
    overhang.StructuralInformation.num_node_elem = 3
    overhang.StructuralInformation.compute_basic_num_elem()
    node_pos = np.zeros((overhang.StructuralInformation.num_node,3), )
    node_pos[:,0] += tower.StructuralInformation.coordinates[-1,0]
    node_pos[:,0] += np.linspace(0.,overhang_len*np.sin(tilt*deg2rad), overhang.StructuralInformation.num_node)
    node_pos[:,2] = np.linspace(0.,-overhang_len*np.cos(tilt*deg2rad), overhang.StructuralInformation.num_node)
    # TODO: change the following by real values
    # Same properties as the last element of the tower
    print("WARNING: Using the structural properties of the last tower section for the overhang")
    oh_mass_per_unit_length = tower.StructuralInformation.mass_db[-1,0,0]
    oh_mass_iner = tower.StructuralInformation.mass_db[-1,3,3]
    oh_EA = tower.StructuralInformation.stiffness_db[-1,0,0]
    oh_GA = tower.StructuralInformation.stiffness_db[-1,1,1]
    oh_GJ = tower.StructuralInformation.stiffness_db[-1,3,3]
    oh_EI = tower.StructuralInformation.stiffness_db[-1,4,4]
    overhang.StructuralInformation.generate_uniform_sym_beam(node_pos,
                                                            oh_mass_per_unit_length,
                                                            oh_mass_iner,
                                                            oh_EA,
                                                            oh_GA,
                                                            oh_GJ,
                                                            oh_EI,
                                                            num_node_elem = 3,
                                                            y_BFoR = 'y_AFoR',
                                                            num_lumped_mass=0)

    overhang.StructuralInformation.boundary_conditions = np.zeros((overhang.StructuralInformation.num_node), dtype = int)
    overhang.StructuralInformation.boundary_conditions[-1] = -1

    overhang.AerodynamicInformation.set_to_zero(overhang.StructuralInformation.num_node_elem,
                                                overhang.StructuralInformation.num_node,
                                                overhang.StructuralInformation.num_elem)

    tower.assembly(overhang)
    tower.remove_duplicated_points(tol_remove_points)

    ######################################################################
    ##  WIND TURBINE
    ######################################################################
    # Assembly the whole case
    wt = tower.copy()
    hub_position = tower.StructuralInformation.coordinates[-1,:]
    rotor.StructuralInformation.coordinates += hub_position
    wt.assembly(rotor)

    # Redefine the body numbers
    wt.StructuralInformation.body_number *= 0
    wt.StructuralInformation.body_number[tower.StructuralInformation.num_elem:wt.StructuralInformation.num_elem] += 1

    ######################################################################
    ## MULTIBODY
    ######################################################################
    # Define the boundary condition between the rotor and the tower tip
    LC1 = gc.LagrangeConstraint()
    LC1.behaviour = 'hinge_node_FoR_constant_vel'
    LC1.node_in_body = tower.StructuralInformation.num_node-1
    LC1.body = 0
    LC1.body_FoR = 1
    LC1.rot_axisB = np.array([1.,0.,0.0])
    LC1.rot_vel = -rotation_velocity

    LC = []
    LC.append(LC1)

    # Define the multibody infromation for the tower and the rotor
    MB1 = gc.BodyInformation()
    MB1.body_number = 0
    MB1.FoR_position = np.zeros((6,),)
    MB1.FoR_velocity = np.zeros((6,),)
    MB1.FoR_acceleration = np.zeros((6,),)
    MB1.FoR_movement = 'prescribed'
    MB1.quat = np.array([1.0,0.0,0.0,0.0])

    MB2 = gc.BodyInformation()
    MB2.body_number = 1
    MB2.FoR_position = np.array([rotor.StructuralInformation.coordinates[0, 0], rotor.StructuralInformation.coordinates[0, 1], rotor.StructuralInformation.coordinates[0, 2], 0.0, 0.0, 0.0])
    MB2.FoR_velocity = np.array([0.,0.,0.,0.,0.,rotation_velocity])
    MB2.FoR_acceleration = np.zeros((6,),)
    MB2.FoR_movement = 'free'
    MB2.quat = algebra.euler2quat(np.array([0.0,tilt,0.0]))

    MB = []
    MB.append(MB1)
    MB.append(MB2)

    ######################################################################
    ## RETURN
    ######################################################################
    return wt, LC, MB



######################################################################
# FROM OpenFAST database
######################################################################
def rotor_from_OpenFAST_db(chord_panels,
                                  rotation_velocity,
                                  pitch_deg,
                                  excel_file_name= 'database_OpenFAST.xlsx',
                                  excel_sheet_parameters = 'parameters',
                                  excel_sheet_structural_blade = 'structural_blade',
                                  excel_sheet_aero_blade = 'aero_blade',
                                  excel_sheet_airfoil_coord = 'airfoil_coord',
                                  m_distribution = 'uniform',
                                  h5_cross_sec_prop = None,
                                  n_points_camber = 100,
                                  tol_remove_points = 1e-3):

    """
    generate_from_OpenFAST_db

    Function needed to generate a wind turbine from an excel database according to OpenFAST inputs

    Args:
    	  chord_panels (int): Number of panels on the blade surface in the chord direction
          rotation_velocity (float): Rotation velocity of the rotor
          pitch_deg (float): pitch angle in degrees
          excel_file_name (str):
          excel_sheet_structural_blade (str):
          excel_sheet_aero_blade (str):
          excel_sheet_airfoil_coord (str):
          excel_sheet_parameters (str):
          h5_cross_sec_prop (str): h5 containing mass and stiffness matrices along the blade.
          m_distribution (str):
          n_points_camber (int): number of points to define the camber of the airfoil,
          tol_remove_points (float): maximum distance to remove adjacent points

    Returns:
        rotor (sharpy.utils.generate_cases.AeroelasticInfromation): Aeroelastic infrmation of the rotor

    Note:
        - h5_cross_sec_prop is a path to a h5 containing the following groups:
            - str_prop: with:
                - K: list of 6x6 stiffness matrices
                - M: list of 6x6 mass matrices
                - radius: radial location (including hub) of K and M matrices
        - when h5_cross_sec_prop is not None, mass and stiffness properties are
        interpolated at BlFract location specified in "excel_sheet_structural_blade"
    """
    ######################################################################
    ## BLADE
    ######################################################################

    blade = gc.AeroelasticInformation()

    ######################################################################
    ### STRUCTURE
    ######################################################################

    # Read blade structural information from excel file
    BlFract = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'BlFract')
    PitchAxis = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'PitchAxis')
    # TODO: implement pitch axsi
    # print("WARNING: PitchAxis not implemented")
    # StrcTwst= gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'StrcTwst')*deg2rad
    BMassDen= gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'BMassDen')
    FlpStff= gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'FlpStff')
    EdgStff = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'EdgStff')
    # Missing the following variables
    GJStff = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'GJStff')
    EAStff = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'EAStff')
    Alpha = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'Alpha')
    FlpIner= gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'FlpIner')
    EdgIner= gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'EdgIner')
    #PrecrvRef = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'PrecrvRef')
    #PreswpRef = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'PreswpRef')
    FlpcgOf = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'FlpcgOf')
    EdgcgOf = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'EdgcgOf')
    FlpEAOf = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'FlpEAOf')
    EdgEAOf = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'EdgEAOf')

    # From the aerodynamic sheet
    excel_aero_r = gc.read_column_sheet_type01(excel_file_name, excel_sheet_aero_blade, 'BlSpn')
    BlCrvAC = gc.read_column_sheet_type01(excel_file_name, excel_sheet_aero_blade, 'BlCrvAC')
    BlSwpAC = gc.read_column_sheet_type01(excel_file_name, excel_sheet_aero_blade, 'BlSwpAC')
    BlCrvAng = gc.read_column_sheet_type01(excel_file_name, excel_sheet_aero_blade, 'BlCrvAng')
    if not (BlCrvAng == 0.).all():
        # TODO: implement this angle
        print("ERROR: BlCrvAng not implemented, assumed to be zero")
    BlTwist = gc.read_column_sheet_type01(excel_file_name, excel_sheet_aero_blade, 'BlTwist')*deg2rad

    # Blade parameters
    TipRad = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'TipRad')
    HubRad = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'HubRad')



    # Interpolate excel variables into the correct locations
    # Geometry
    Radius = HubRad + BlFract*(TipRad - HubRad)
    excel_aero_r +=  HubRad

    include_hub_node = True
    if include_hub_node:
        Radius = np.concatenate((np.array([0.]), Radius),)
        PitchAxis = np.concatenate((np.array([PitchAxis[0]]), PitchAxis),)
        BMassDen  = np.concatenate((np.array([BMassDen[0]]), BMassDen),)
        FlpStff = np.concatenate((np.array([FlpStff[0]]), FlpStff),)
        EdgStff = np.concatenate((np.array([EdgStff[0]]), EdgStff),)
        GJStff = np.concatenate((np.array([GJStff[0]]), GJStff),)
        EAStff = np.concatenate((np.array([EAStff[0]]), EAStff),)
        Alpha = np.concatenate((np.array([Alpha[0]]), Alpha),)
        FlpIner = np.concatenate((np.array([FlpIner[0]]), FlpIner),)
        EdgIner = np.concatenate((np.array([EdgIner[0]]), EdgIner),)
        FlpcgOf = np.concatenate((np.array([FlpcgOf[0]]), FlpcgOf),)
        EdgcgOf = np.concatenate((np.array([EdgcgOf[0]]), EdgcgOf),)
        FlpEAOf = np.concatenate((np.array([FlpEAOf[0]]), FlpEAOf),)
        EdgEAOf = np.concatenate((np.array([EdgEAOf[0]]), EdgEAOf),)


    # Base parameters
    use_excel_struct_as_elem = False
    if use_excel_struct_as_elem:
        blade.StructuralInformation.num_node_elem = 3
        blade.StructuralInformation.num_elem = len(Radius) - 2
        blade.StructuralInformation.compute_basic_num_node()

        node_r, elem_r = create_node_radial_pos_from_elem_centres(Radius,
                                            blade.StructuralInformation.num_node,
                                            blade.StructuralInformation.num_elem,
                                            blade.StructuralInformation.num_node_elem)
    else:
        # Use excel struct as nodes
        # Check the number of nodes
        blade.StructuralInformation.num_node_elem = 3
        blade.StructuralInformation.num_node = len(Radius)
        if ((len(Radius) - 1) % (blade.StructuralInformation.num_node_elem - 1)) == 0:
            blade.StructuralInformation.num_elem = int((len(Radius) - 1)/(blade.StructuralInformation.num_node_elem - 1))
            node_r = Radius
            elem_r = Radius[1::2] + 0.
        else:
            print("ERROR: Cannot build ", blade.StructuralInformation.num_node_elem, "-noded elements from ", blade.StructuralInformation.num_node, "nodes")

    # TODO: how is this defined now?
    node_prebending = np.interp(node_r,excel_aero_r,BlCrvAC)
    # node_presweept = np.interp(node_r,excel_aero_r,BlSwpAC)
    print("WARNING: Check the implementation for presweept blades")
    node_presweept = np.zeros_like(node_r)

    # node_structural_twist = -1.0*np.interp(node_r,Radius,StrcTwst)
    node_structural_twist = -1.0*np.interp(node_r,excel_aero_r,BlTwist)
    node_pitch_axis = np.interp(node_r,Radius,PitchAxis)

    coordinates = create_blade_coordinates(blade.StructuralInformation.num_node, node_r, node_prebending, node_presweept)


    if h5_cross_sec_prop is None:
        # Stiffness
        elem_EA = np.interp(elem_r,Radius,EAStff)
        elem_EIy = np.interp(elem_r,Radius,FlpStff)
        elem_EIz = np.interp(elem_r,Radius,EdgStff)
        elem_GJ = np.interp(elem_r,Radius,GJStff)
        # Stiffness: estimate unknown properties
        print('WARNING: The poisson cofficient is assumed equal to 0.3')
        print('WARNING: Cross-section area is used as shear area')
        poisson_coef = 0.3
        elem_GAy = elem_EA/2.0/(1.0+poisson_coef)
        elem_GAz = elem_EA/2.0/(1.0+poisson_coef)
        # Inertia
        # TODO: check yz axis and Flap-edge
        elem_pos_cg_B = np.zeros((blade.StructuralInformation.num_elem,3),)
        elem_pos_cg_B[:,2]=np.interp(elem_r,Radius,FlpcgOf)
        elem_pos_cg_B[:,1]=np.interp(elem_r,Radius,EdgcgOf)

        elem_mass_per_unit_length = np.interp(elem_r,Radius,BMassDen)
        elem_mass_iner_y = np.interp(elem_r,Radius,FlpIner)
        elem_mass_iner_z = np.interp(elem_r,Radius,EdgIner)

        # Inertia: estimate unknown properties
        print('WARNING: Using perpendicular axis theorem to compute the inertia around xB')
        elem_mass_iner_x = elem_mass_iner_y + elem_mass_iner_z

        # Generate blade structural properties
        blade.StructuralInformation.create_mass_db_from_vector(elem_mass_per_unit_length, elem_mass_iner_x, elem_mass_iner_y, elem_mass_iner_z, elem_pos_cg_B)
        blade.StructuralInformation.create_stiff_db_from_vector(elem_EA, elem_GAy, elem_GAz, elem_GJ, elem_EIy, elem_EIz)

    else: # read Mass/Stiffness from database
        cross_prop=h5.readh5(h5_cross_sec_prop).str_prop

        # create mass_db/stiffness_db (interpolate at mid-node of each element)
        blade.StructuralInformation.mass_db = scint.interp1d(
                    cross_prop.radius, cross_prop.M, kind='cubic', copy=False, assume_sorted=True, axis=0)(node_r[1::2])
        blade.StructuralInformation.stiffness_db = scint.interp1d(
                    cross_prop.radius, cross_prop.K, kind='cubic', copy=False, assume_sorted=True, axis=0)(node_r[1::2])

    blade.StructuralInformation.generate_1to1_from_vectors(
            num_node_elem = blade.StructuralInformation.num_node_elem,
            num_node = blade.StructuralInformation.num_node,
            num_elem = blade.StructuralInformation.num_elem,
            coordinates = coordinates,
            stiffness_db = blade.StructuralInformation.stiffness_db,
            mass_db = blade.StructuralInformation.mass_db,
            frame_of_reference_delta = 'y_AFoR',
            vec_node_structural_twist = node_structural_twist,
            num_lumped_mass = 0)

    # Boundary conditions
    blade.StructuralInformation.boundary_conditions = np.zeros((blade.StructuralInformation.num_node), dtype = int)
    blade.StructuralInformation.boundary_conditions[0] = 1
    blade.StructuralInformation.boundary_conditions[-1] = -1

    ######################################################################
    ### AERODYNAMICS
    ######################################################################
    # Read blade aerodynamic information from excel file
    # excel_aerodynamic_twist = gc.read_column_sheet_type01(excel_file_name, excel_sheet_aero_blade, 'BlTwist')*deg2rad
    excel_chord = gc.read_column_sheet_type01(excel_file_name, excel_sheet_aero_blade, 'BlChord')
    pure_airfoils_names = gc.read_column_sheet_type01(excel_file_name, excel_sheet_aero_blade, 'BlAFID')

    # Read coordinates of the pure airfoils
    n_elem_aero = len(excel_aero_r)
    # TODO: change this with a list of thickness and pure airfoils
    pure_airfoils_camber=np.zeros((n_elem_aero,n_points_camber,2),)
    xls = pd.ExcelFile(excel_file_name)
    excel_db = pd.read_excel(xls, sheet_name=excel_sheet_airfoil_coord)
    for iairfoil in range(len(pure_airfoils_names)):
        # Look for the NaN
        icoord=2
        while(not(math.isnan(excel_db["%s_x" % pure_airfoils_names[iairfoil]][icoord]))):
            icoord+=1
            if(icoord==len(excel_db["%s_x" % pure_airfoils_names[iairfoil]])):
                break

        # Compute the camber of the airfoil
        pure_airfoils_camber[iairfoil,:,0], pure_airfoils_camber[iairfoil,:,1] = gc.get_airfoil_camber(excel_db["%s_x" % pure_airfoils_names[iairfoil]][2:icoord] , excel_db["%s_y" % pure_airfoils_names[iairfoil]][2:icoord], n_points_camber)

    # Basic variables
    n_elem_aero = len(excel_aero_r)
    num_airfoils = blade.StructuralInformation.num_node
    surface_distribution = np.zeros((blade.StructuralInformation.num_elem), dtype=int)

    # Interpolate in the correct positions
    node_chord=np.interp(node_r, excel_aero_r, excel_chord)
    # node_aero_twist = -1.0*(np.interp(node_r, excel_aero_r, excel_aerodynamic_twist) + node_structural_twist)
    node_sweep = np.ones((blade.StructuralInformation.num_node), )*np.pi
    # node_elastic_axis=np.ones((blade.StructuralInformation.num_node,))*0.25

    # Define the nodes with aerodynamic properties
    # Look for the first element that is goint to be aerodynamic
    first_aero_elem=0
    while (elem_r[first_aero_elem]<=excel_aero_r[0]):
        first_aero_elem+=1
    first_aero_node=first_aero_elem*(blade.StructuralInformation.num_node_elem-1)
    aero_node = np.zeros((blade.StructuralInformation.num_node,), dtype=bool)
    aero_node[first_aero_node:]=np.ones((blade.StructuralInformation.num_node-first_aero_node,),dtype=bool)

    airfoils = blade.AerodynamicInformation.interpolate_airfoils_camber(pure_airfoils_camber,excel_aero_r, node_r, n_points_camber)

    # Write SHARPy format
    airfoil_distribution = np.linspace(0,blade.StructuralInformation.num_node-1,blade.StructuralInformation.num_node, dtype=int)

    blade.AerodynamicInformation.create_aerodynamics_from_vec(blade.StructuralInformation,
                                                            aero_node,
                                                            node_chord,
                                                            np.zeros_like(node_chord),
                                                            node_sweep,
                                                            chord_panels,
                                                            surface_distribution,
                                                            m_distribution,
                                                            node_pitch_axis,
                                                            airfoil_distribution,
                                                            airfoils)

    ######################################################################
    ## ROTOR
    ######################################################################

    # Read from excel file
    numberOfBlades = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'NumBl')
    tilt = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'ShftTilt')*deg2rad
    cone = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'Cone')*deg2rad
    # pitch = gc.read_column_sheet_type01(excel_file_name, excel_sheet_rotor, 'Pitch')*deg2rad

    # Apply pitch
    blade.StructuralInformation.rotate_around_origin(np.array([1.,0.,0.]), -pitch_deg*deg2rad)

    # Apply coning
    blade.StructuralInformation.rotate_around_origin(np.array([0.,1.,0.]), -cone)

    # Build the whole rotor
    rotor = blade.copy()
    for iblade in range(numberOfBlades-1):
        blade2 = blade.copy()
        blade2.StructuralInformation.rotate_around_origin(np.array([0.,0.,1.]), (iblade+1)*(360.0/numberOfBlades)*deg2rad)
        rotor.assembly(blade2)
        blade2 = None

    rotor.remove_duplicated_points(tol_remove_points)

    # Apply tilt
    rotor.StructuralInformation.rotate_around_origin(np.array([0.,1.,0.]), tilt)

    return rotor


def generate_from_OpenFAST_db(chord_panels,
                                  rotation_velocity,
                                  pitch_deg,
                                  excel_file_name= 'database_OpenFAST.xlsx',
                                  excel_sheet_parameters = 'parameters',
                                  excel_sheet_structural_blade = 'structural_blade',
                                  excel_sheet_aero_blade = 'aero_blade',
                                  excel_sheet_airfoil_coord = 'airfoil_coord',
                                  excel_sheet_structural_tower = 'structural_tower',
                                  m_distribution = 'uniform',
                                  n_points_camber = 100,
                                  tol_remove_points = 1e-3):

    """
    generate_from_OpenFAST_db

    Function needed to generate a wind turbine from an excel database according to OpenFAST inputs

    Args:
    	  chord_panels (int): Number of panels on the blade surface in the chord direction
          rotation_velocity (float): Rotation velocity of the rotor
          pitch_deg (float): pitch angle in degrees
          excel_file_name (str):
          excel_sheet_structural_blade (str):
          excel_sheet_aero_blade (str):
          excel_sheet_airfoil_coord (str):
          excel_sheet_parameters (str):
          excel_sheet_structural_tower (str):
          m_distribution (str):
          n_points_camber (int): number of points to define the camber of the airfoil,
          tol_remove_points (float): maximum distance to remove adjacent points

    Returns:
        wt (sharpy.utils.generate_cases.AeroelasticInfromation): Aeroelastic infrmation of the wind turbine
        LC (list): list of all the Lagrange constraints needed in the cases (sharpy.utils.generate_cases.LagrangeConstraint)
        MB (list): list of the multibody information of each body (sharpy.utils.generate_cases.BodyInfrmation)
    """

    rotor = rotor_from_OpenFAST_db(chord_panels,
                                  rotation_velocity,
                                  pitch_deg,
                                  excel_file_name= excel_file_name,
                                  excel_sheet_parameters = excel_sheet_parameters,
                                  excel_sheet_structural_blade = excel_sheet_structural_blade,
                                  excel_sheet_aero_blade = excel_sheet_aero_blade,
                                  excel_sheet_airfoil_coord = excel_sheet_airfoil_coord,
                                  m_distribution = m_distribution,
                                  n_points_camber = n_points_camber,
                                  tol_remove_points = tol_remove_points)

    ######################################################################
    ## TOWER
    ######################################################################

    # Read from excel file
    HtFract = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_tower, 'HtFract')
    TMassDen = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_tower, 'TMassDen')
    TwFAStif = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_tower, 'TwFAStif')
    TwSSStif = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_tower, 'TwSSStif')
    # TODO> variables to be defined
    TwGJStif = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_tower, 'TwGJStif')
    TwEAStif = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_tower, 'TwEAStif')
    TwFAIner = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_tower, 'TwFAIner')
    TwSSIner = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_tower, 'TwSSIner')
    TwFAcgOf = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_tower, 'TwFAcgOf')
    TwSScgOf = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_tower, 'TwSScgOf')

    # Define the TOWER
    TowerHt = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'TowerHt')
    Elevation = TowerHt*HtFract

    tower = gc.AeroelasticInformation()
    tower.StructuralInformation.num_elem = len(Elevation) - 2
    tower.StructuralInformation.num_node_elem = 3
    tower.StructuralInformation.compute_basic_num_node()

    # Interpolate excel variables into the correct locations
    node_r, elem_r = create_node_radial_pos_from_elem_centres(Elevation,
                                        tower.StructuralInformation.num_node,
                                        tower.StructuralInformation.num_elem,
                                        tower.StructuralInformation.num_node_elem)

    # Stiffness
    elem_EA = np.interp(elem_r,Elevation,TwEAStif)
    elem_EIz = np.interp(elem_r,Elevation,TwSSStif)
    elem_EIy = np.interp(elem_r,Elevation,TwFAStif)
    elem_GJ = np.interp(elem_r,Elevation,TwGJStif)
    # Stiffness: estimate unknown properties
    print('WARNING: The poisson cofficient is assumed equal to 0.3')
    print('WARNING: Cross-section area is used as shear area')
    poisson_coef = 0.3
    elem_GAy = elem_EA/2.0/(1.0+poisson_coef)
    elem_GAz = elem_EA/2.0/(1.0+poisson_coef)

    # Inertia
    elem_mass_per_unit_length = np.interp(elem_r,Elevation,TMassDen)
    elem_mass_iner_y = np.interp(elem_r,Elevation,TwFAIner)
    elem_mass_iner_z = np.interp(elem_r,Elevation,TwSSIner)
    # TODO: check yz axis and Flap-edge
    elem_pos_cg_B = np.zeros((tower.StructuralInformation.num_elem,3),)
    elem_pos_cg_B[:,1]=np.interp(elem_r,Elevation,TwSScgOf)
    elem_pos_cg_B[:,2]=np.interp(elem_r,Elevation,TwFAcgOf)

    # Stiffness: estimate unknown properties
    print('WARNING: Using perpendicular axis theorem to compute the inertia around xB')
    elem_mass_iner_x = elem_mass_iner_y + elem_mass_iner_z

    # Create the tower
    tower.StructuralInformation.create_mass_db_from_vector(elem_mass_per_unit_length, elem_mass_iner_x, elem_mass_iner_y, elem_mass_iner_z, elem_pos_cg_B)
    tower.StructuralInformation.create_stiff_db_from_vector(elem_EA, elem_GAy, elem_GAz, elem_GJ, elem_EIy, elem_EIz)

    coordinates = np.zeros((tower.StructuralInformation.num_node,3),)
    coordinates[:,0] = node_r

    tower.StructuralInformation.generate_1to1_from_vectors(
            num_node_elem = tower.StructuralInformation.num_node_elem,
            num_node = tower.StructuralInformation.num_node,
            num_elem = tower.StructuralInformation.num_elem,
            coordinates = coordinates,
            stiffness_db = tower.StructuralInformation.stiffness_db,
            mass_db = tower.StructuralInformation.mass_db,
            frame_of_reference_delta = 'y_AFoR',
            vec_node_structural_twist = np.zeros((tower.StructuralInformation.num_node,),),
            num_lumped_mass = 1)

    tower.StructuralInformation.boundary_conditions = np.zeros((tower.StructuralInformation.num_node), dtype = int)
    tower.StructuralInformation.boundary_conditions[0] = 1

    # Read overhang and nacelle properties from excel file
    overhang_len = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'overhang')
    # HubMass = gc.read_column_sheet_type01(excel_file_name, excel_sheet_nacelle, 'HubMass')
    NacelleMass = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'NacMass')
    # NacelleYawIner = gc.read_column_sheet_type01(excel_file_name, excel_sheet_nacelle, 'NacelleYawIner')

    # Include nacelle mass
    tower.StructuralInformation.lumped_mass_nodes = np.array([tower.StructuralInformation.num_node-1], dtype=int)
    tower.StructuralInformation.lumped_mass = np.array([NacelleMass], dtype=float)

    tower.AerodynamicInformation.set_to_zero(tower.StructuralInformation.num_node_elem,
                                            tower.StructuralInformation.num_node,
                                            tower.StructuralInformation.num_elem)

    # Assembly overhang with the tower
    # numberOfBlades = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'NumBl')
    tilt = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'ShftTilt')*deg2rad
    # cone = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'Cone')*deg2rad

    overhang = gc.AeroelasticInformation()
    overhang.StructuralInformation.num_node = 3
    overhang.StructuralInformation.num_node_elem = 3
    overhang.StructuralInformation.compute_basic_num_elem()
    node_pos = np.zeros((overhang.StructuralInformation.num_node,3), )
    node_pos[:,0] += tower.StructuralInformation.coordinates[-1,0]
    node_pos[:,0] += np.linspace(0.,overhang_len*np.sin(tilt*deg2rad), overhang.StructuralInformation.num_node)
    node_pos[:,2] = np.linspace(0.,-overhang_len*np.cos(tilt*deg2rad), overhang.StructuralInformation.num_node)
    # TODO: change the following by real values
    # Same properties as the last element of the tower
    print("WARNING: Using the structural properties of the last tower section for the overhang")
    oh_mass_per_unit_length = tower.StructuralInformation.mass_db[-1,0,0]
    oh_mass_iner = tower.StructuralInformation.mass_db[-1,3,3]
    oh_EA = tower.StructuralInformation.stiffness_db[-1,0,0]
    oh_GA = tower.StructuralInformation.stiffness_db[-1,1,1]
    oh_GJ = tower.StructuralInformation.stiffness_db[-1,3,3]
    oh_EI = tower.StructuralInformation.stiffness_db[-1,4,4]
    overhang.StructuralInformation.generate_uniform_sym_beam(node_pos,
                                                            oh_mass_per_unit_length,
                                                            oh_mass_iner,
                                                            oh_EA,
                                                            oh_GA,
                                                            oh_GJ,
                                                            oh_EI,
                                                            num_node_elem = 3,
                                                            y_BFoR = 'y_AFoR',
                                                            num_lumped_mass=0)

    overhang.StructuralInformation.boundary_conditions = np.zeros((overhang.StructuralInformation.num_node), dtype = int)
    overhang.StructuralInformation.boundary_conditions[-1] = -1

    overhang.AerodynamicInformation.set_to_zero(overhang.StructuralInformation.num_node_elem,
                                                overhang.StructuralInformation.num_node,
                                                overhang.StructuralInformation.num_elem)

    tower.assembly(overhang)
    tower.remove_duplicated_points(tol_remove_points)

    ######################################################################
    ##  WIND TURBINE
    ######################################################################
    # Assembly the whole case
    wt = tower.copy()
    hub_position = tower.StructuralInformation.coordinates[-1,:]
    rotor.StructuralInformation.coordinates += hub_position
    wt.assembly(rotor)

    # Redefine the body numbers
    wt.StructuralInformation.body_number *= 0
    wt.StructuralInformation.body_number[tower.StructuralInformation.num_elem:wt.StructuralInformation.num_elem] += 1

    ######################################################################
    ## MULTIBODY
    ######################################################################
    # Define the boundary condition between the rotor and the tower tip
    LC1 = gc.LagrangeConstraint()
    LC1.behaviour = 'hinge_node_FoR_constant_vel'
    LC1.node_in_body = tower.StructuralInformation.num_node-1
    LC1.body = 0
    LC1.body_FoR = 1
    LC1.rot_axisB = np.array([1.,0.,0.0])
    LC1.rot_vel = -rotation_velocity

    LC = []
    LC.append(LC1)

    # Define the multibody infromation for the tower and the rotor
    MB1 = gc.BodyInformation()
    MB1.body_number = 0
    MB1.FoR_position = np.zeros((6,),)
    MB1.FoR_velocity = np.zeros((6,),)
    MB1.FoR_acceleration = np.zeros((6,),)
    MB1.FoR_movement = 'prescribed'
    MB1.quat = np.array([1.0,0.0,0.0,0.0])

    MB2 = gc.BodyInformation()
    MB2.body_number = 1
    MB2.FoR_position = np.array([rotor.StructuralInformation.coordinates[0, 0], rotor.StructuralInformation.coordinates[0, 1], rotor.StructuralInformation.coordinates[0, 2], 0.0, 0.0, 0.0])
    MB2.FoR_velocity = np.array([0.,0.,0.,0.,0.,rotation_velocity])
    MB2.FoR_acceleration = np.zeros((6,),)
    MB2.FoR_movement = 'free'
    MB2.quat = algebra.euler2quat(np.array([0.0,tilt,0.0]))

    MB = []
    MB.append(MB1)
    MB.append(MB2)

    ######################################################################
    ## RETURN
    ######################################################################
    return wt, LC, MB

######################################################################
# FROM excel type02
######################################################################
def rotor_from_excel_type02(chord_panels,
                                  rotation_velocity,
                                  pitch_deg,
                                  excel_file_name= 'database_excel_type02.xlsx',
                                  excel_sheet_parameters = 'parameters',
                                  excel_sheet_structural_blade = 'structural_blade',
                                  excel_sheet_discretization_blade = 'discretization_blade',
                                  excel_sheet_aero_blade = 'aero_blade',
                                  excel_sheet_airfoil_info = 'airfoil_info',
                                  excel_sheet_airfoil_coord = 'airfoil_coord',
                                  m_distribution = 'uniform',
                                  h5_cross_sec_prop = None,
                                  n_points_camber = 100,
                                  tol_remove_points = 1e-3,
                                  user_defined_m_distribution_type = None,
                                  camber_effect_on_twist = False,
                                  wsp = 0.,
                                  dt = 0.):

    """
    generate_from_excel_type02_db

    Function needed to generate a wind turbine from an excel database type02

    Args:
    	  chord_panels (int): Number of panels on the blade surface in the chord direction
          rotation_velocity (float): Rotation velocity of the rotor
          pitch_deg (float): pitch angle in degrees
          excel_file_name (str):
          excel_sheet_structural_blade (str):
          excel_sheet_discretization_blade (str):
          excel_sheet_aero_blade (str):
          excel_sheet_airfoil_info (str):
          excel_sheet_airfoil_coord (str):
          excel_sheet_parameters (str):
          h5_cross_sec_prop (str): h5 containing mass and stiffness matrices along the blade.
          m_distribution (str):
          n_points_camber (int): number of points to define the camber of the airfoil,
          tol_remove_points (float): maximum distance to remove adjacent points

    Returns:
        rotor (sharpy.utils.generate_cases.AeroelasticInfromation): Aeroelastic information of the rotor

    Note:
        - h5_cross_sec_prop is a path to a h5 containing the following groups:
            - str_prop: with:
                - K: list of 6x6 stiffness matrices
                - M: list of 6x6 mass matrices
                - radius: radial location (including hub) of K and M matrices
        - when h5_cross_sec_prop is not None, mass and stiffness properties are
        interpolated at BlFract location specified in "excel_sheet_structural_blade"
    """
    ######################################################################
    ## BLADE
    ######################################################################

    blade = gc.AeroelasticInformation()

    ######################################################################
    ### STRUCTURE
    ######################################################################
    # Read blade structural information from excel file
    rR_structural = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'rR')
    OutPElAxis = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'OutPElAxis')
    InPElAxis = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'InPElAxis')
    ElAxisAftLEc = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'ElAxisAftLEc')
    StrcTwst = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'StrcTwst')*deg2rad
    BMassDen = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'BMassDen')
    FlpStff = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'FlpStff')
    EdgStff = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'EdgStff')
    FlapEdgeStiff = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'FlapEdgeStiff')
    GJStff = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'GJStff')
    EAStff = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'EAStff')
    FlpIner = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'FlpIner')
    EdgIner = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'EdgIner')
    FlapEdgeIner = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'FlapEdgeIner')
    PrebendRef = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'PrebendRef')
    PreswpRef = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'PreswpRef')
    OutPcg = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'OutPcg')
    InPcg = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_blade, 'InPcg')

    # Blade parameters
    TipRad = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'TipRad')
    # HubRad = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'HubRad')

    # Discretization points
    rR = gc.read_column_sheet_type01(excel_file_name, excel_sheet_discretization_blade, 'rR')

    # Interpolate excel variables into the correct locations
    # Geometry
    if rR[0] < rR_structural[0]:
        rR_structural = np.concatenate((np.array([0.]), rR_structural),)
        OutPElAxis = np.concatenate((np.array([OutPElAxis[0]]), OutPElAxis),)
        InPElAxis  = np.concatenate((np.array([InPElAxis[0]]), InPElAxis),)
        ElAxisAftLEc = np.concatenate((np.array([ElAxisAftLEc[0]]), ElAxisAftLEc),)
        StrcTwst = np.concatenate((np.array([StrcTwst[0]]), StrcTwst),)
        BMassDen = np.concatenate((np.array([BMassDen[0]]), BMassDen),)
        FlpStff = np.concatenate((np.array([FlpStff[0]]), FlpStff),)
        EdgStff = np.concatenate((np.array([EdgStff[0]]), EdgStff),)
        FlapEdgeStiff = np.concatenate((np.array([FlapEdgeStiff[0]]), FlapEdgeStiff),)
        GJStff = np.concatenate((np.array([GJStff[0]]), GJStff),)
        EAStff = np.concatenate((np.array([EAStff[0]]), EAStff),)
        FlpIner = np.concatenate((np.array([FlpIner[0]]), FlpIner),)
        EdgIner = np.concatenate((np.array([EdgIner[0]]), EdgIner),)
        FlapEdgeIner = np.concatenate((np.array([FlapEdgeIner[0]]), FlapEdgeIner),)
        PrebendRef = np.concatenate((np.array([PrebendRef[0]]), PrebendRef),)
        PreswpRef = np.concatenate((np.array([PreswpRef[0]]), PreswpRef),)
        OutPcg = np.concatenate((np.array([OutPcg[0]]), OutPcg),)
        InPcg = np.concatenate((np.array([InPcg[0]]), InPcg),)


    # Base parameters
    use_excel_struct_as_elem = False
    if use_excel_struct_as_elem:
        blade.StructuralInformation.num_node_elem = 3
        blade.StructuralInformation.num_elem = len(rR) - 2
        blade.StructuralInformation.compute_basic_num_node()

        node_r, elem_r = create_node_radial_pos_from_elem_centres(rR*TipRad,
                                            blade.StructuralInformation.num_node,
                                            blade.StructuralInformation.num_elem,
                                            blade.StructuralInformation.num_node_elem)
    else:
        # Use excel struct as nodes
        # Check the number of nodes
        blade.StructuralInformation.num_node_elem = 3
        blade.StructuralInformation.num_node = len(rR)
        if ((len(rR) - 1) % (blade.StructuralInformation.num_node_elem - 1)) == 0:
            blade.StructuralInformation.num_elem = int((len(rR) - 1)/(blade.StructuralInformation.num_node_elem - 1))
            node_r = rR*TipRad
            elem_rR = rR[1::2] + 0.
            elem_r = rR[1::2]*TipRad + 0.
        else:
            print("ERROR: Cannot build ", blade.StructuralInformation.num_node_elem, "-noded elements from ", blade.StructuralInformation.num_node, "nodes")

    node_y = np.interp(rR,rR_structural,InPElAxis) + np.interp(rR,rR_structural,PreswpRef)
    node_z = -np.interp(rR,rR_structural,OutPElAxis) - np.interp(rR,rR_structural,PrebendRef)
    node_twist = -1.0*np.interp(rR,rR_structural,StrcTwst)

    coordinates = create_blade_coordinates(blade.StructuralInformation.num_node, node_r, node_y, node_z)

    if h5_cross_sec_prop is None:
        # Stiffness
        elem_EA = np.interp(elem_rR,rR_structural,EAStff)
        elem_EIy = np.interp(elem_rR,rR_structural,FlpStff)
        elem_EIz = np.interp(elem_rR,rR_structural,EdgStff)
        elem_EIyz = np.interp(elem_rR,rR_structural,FlapEdgeStiff)
        elem_GJ = np.interp(elem_rR,rR_structural,GJStff)

        # Stiffness: estimate unknown properties
        print('WARNING: The poisson cofficient is assumed equal to 0.3')
        print('WARNING: Cross-section area is used as shear area')
        poisson_coef = 0.3
        elem_GAy = elem_EA/2.0/(1.0+poisson_coef)
        elem_GAz = elem_EA/2.0/(1.0+poisson_coef)
        # Inertia
        elem_pos_cg_B = np.zeros((blade.StructuralInformation.num_elem,3),)
        elem_pos_cg_B[:,1] = np.interp(elem_rR,rR_structural,InPcg)
        elem_pos_cg_B[:,2] = -np.interp(elem_rR,rR_structural,OutPcg)

        elem_mass_per_unit_length = np.interp(elem_rR,rR_structural,BMassDen)
        elem_mass_iner_y = np.interp(elem_rR,rR_structural,FlpIner)
        elem_mass_iner_z = np.interp(elem_rR,rR_structural,EdgIner)
        elem_mass_iner_yz = np.interp(elem_rR,rR_structural,FlapEdgeIner)

        # Inertia: estimate unknown properties
        print('WARNING: Using perpendicular axis theorem to compute the inertia around xB')
        elem_mass_iner_x = elem_mass_iner_y + elem_mass_iner_z

        # Generate blade structural properties
        blade.StructuralInformation.create_mass_db_from_vector(elem_mass_per_unit_length, elem_mass_iner_x, elem_mass_iner_y, elem_mass_iner_z, elem_pos_cg_B, elem_mass_iner_yz)
        blade.StructuralInformation.create_stiff_db_from_vector(elem_EA, elem_GAy, elem_GAz, elem_GJ, elem_EIy, elem_EIz, elem_EIyz)

    else: # read Mass/Stiffness from database
        cross_prop=h5.readh5(h5_cross_sec_prop).str_prop

        # create mass_db/stiffness_db (interpolate at mid-node of each element)
        blade.StructuralInformation.mass_db = scint.interp1d(
                    cross_prop.radius, cross_prop.M, kind='cubic', copy=False, assume_sorted=True, axis=0,
                                                    bounds_error = False, fill_value='extrapolate')(node_r[1::2])
        blade.StructuralInformation.stiffness_db = scint.interp1d(
                    cross_prop.radius, cross_prop.K, kind='cubic', copy=False, assume_sorted=True, axis=0,
                                                    bounds_error = False, fill_value='extrapolate')(node_r[1::2])

    blade.StructuralInformation.generate_1to1_from_vectors(
            num_node_elem = blade.StructuralInformation.num_node_elem,
            num_node = blade.StructuralInformation.num_node,
            num_elem = blade.StructuralInformation.num_elem,
            coordinates = coordinates,
            stiffness_db = blade.StructuralInformation.stiffness_db,
            mass_db = blade.StructuralInformation.mass_db,
            frame_of_reference_delta = 'y_AFoR',
            vec_node_structural_twist = node_twist,
            num_lumped_mass = 0)

    # Boundary conditions
    blade.StructuralInformation.boundary_conditions = np.zeros((blade.StructuralInformation.num_node), dtype = int)
    blade.StructuralInformation.boundary_conditions[0] = 1
    blade.StructuralInformation.boundary_conditions[-1] = -1

    ######################################################################
    ### AERODYNAMICS
    ######################################################################
    # Read blade aerodynamic information from excel file
    rR_aero = gc.read_column_sheet_type01(excel_file_name, excel_sheet_aero_blade, 'rR')
    chord_aero = gc.read_column_sheet_type01(excel_file_name, excel_sheet_aero_blade, 'BlChord')
    thickness_aero = gc.read_column_sheet_type01(excel_file_name, excel_sheet_aero_blade, 'BlThickness')

    pure_airfoils_names = gc.read_column_sheet_type01(excel_file_name, excel_sheet_airfoil_info, 'Name')
    pure_airfoils_thickness = gc.read_column_sheet_type01(excel_file_name, excel_sheet_airfoil_info, 'Thickness')

    node_ElAxisAftLEc = np.interp(node_r,rR_structural*TipRad,ElAxisAftLEc)

    # Read coordinates of the pure airfoils
    n_pure_airfoils = len(pure_airfoils_names)

    pure_airfoils_camber=np.zeros((n_pure_airfoils,n_points_camber,2),)
    xls = pd.ExcelFile(excel_file_name)
    excel_db = pd.read_excel(xls, sheet_name=excel_sheet_airfoil_coord)
    for iairfoil in range(n_pure_airfoils):
        # Look for the NaN
        icoord=2
        while(not(math.isnan(excel_db["%s_x" % pure_airfoils_names[iairfoil]][icoord]))):
            icoord+=1
            if(icoord==len(excel_db["%s_x" % pure_airfoils_names[iairfoil]])):
                break

        # Compute the camber of the airfoils at the defined chord points
        pure_airfoils_camber[iairfoil,:,0], pure_airfoils_camber[iairfoil,:,1] = gc.get_airfoil_camber(excel_db["%s_x" % pure_airfoils_names[iairfoil]][2:icoord] , excel_db["%s_y" % pure_airfoils_names[iairfoil]][2:icoord], n_points_camber)

    # Basic variables
    n_elem_aero = len(rR_aero)
    num_airfoils = blade.StructuralInformation.num_node
    surface_distribution = np.zeros((blade.StructuralInformation.num_elem), dtype=int)

    # Interpolate in the correct positions
    node_chord = np.interp(node_r, rR_aero*TipRad, chord_aero)

    # Define the nodes with aerodynamic properties
    # Look for the first element that is goint to be aerodynamic
    first_aero_elem=0
    while (elem_r[first_aero_elem]<=rR_aero[0]*TipRad):
        first_aero_elem+=1
    first_aero_node=first_aero_elem*(blade.StructuralInformation.num_node_elem-1)
    aero_node = np.zeros((blade.StructuralInformation.num_node,), dtype=bool)
    aero_node[first_aero_node:]=np.ones((blade.StructuralInformation.num_node-first_aero_node,),dtype=bool)

    # Define the airfoil at each stage
    # airfoils = blade.AerodynamicInformation.interpolate_airfoils_camber(pure_airfoils_camber,excel_aero_r, node_r, n_points_camber)

    node_thickness = np.interp(node_r, rR_aero*TipRad, thickness_aero)

    airfoils = blade.AerodynamicInformation.interpolate_airfoils_camber_thickness(pure_airfoils_camber, pure_airfoils_thickness, node_thickness, n_points_camber)
    airfoil_distribution = np.linspace(0,blade.StructuralInformation.num_node-1,blade.StructuralInformation.num_node, dtype=int)

    # User defined m distribution
    if (m_distribution == 'user_defined') and (user_defined_m_distribution_type == 'last_geometric'):
        # WSP =10.5
        # dt = 0.01846909261369661/2
        blade_nodes = blade.StructuralInformation.num_node
        udmd_by_nodes = np.zeros((blade_nodes, chord_panels[0] + 1))
        for inode in range(blade_nodes):
            r = np.linalg.norm(blade.StructuralInformation.coordinates[inode, :])
            vrel = np.sqrt(rotation_velocity**2*r**2 + wsp**2)
            # ielem, inode_in_elem = gc.get_ielem_inode(blade.StructuralInformation.connectivities, inode)
            last_length = vrel*dt/node_chord[inode]
            last_length = np.minimum(last_length, 0.5)
            if last_length <= 0.5:
                ratio = gc.get_factor_geometric_progression(last_length, 1., chord_panels)
                udmd_by_nodes[inode, -1] = 1.
                udmd_by_nodes[inode, 0] = 0.
                for im in range(chord_panels[0] -1, 0, -1):
                    udmd_by_nodes[inode, im] = udmd_by_nodes[inode, im + 1] - last_length
                    last_length *= ratio
                # Check
                if (np.diff(udmd_by_nodes[inode, :]) < 0.).any():
                    sys.error("ERROR in the panel discretization of the blade in node %d" % (inode))
            else:
                print("ERROR: cannot match the last panel size for node:", inode)
                udmd_by_nodes[inode,:] = np.linspace(0, 1, chord_panels + 1)

    else:
        udmd_by_nodes = None
    # udmd_by_elements = gc.from_node_array_to_elem_matrix(udmd_by_nodes, rotor.StructuralInformation.connectivities[0:int((blade_nodes-1)/2), :])
    # rotor.user_defined_m_distribution = (udmd_by_elements, udmd_by_elements, udmd_by_elements)

    node_twist = np.zeros_like(node_chord)
    if camber_effect_on_twist:
        print("WARNING: The steady applied Mx should be manually multiplied by the density")
        for inode in range(blade.StructuralInformation.num_node):
            node_twist[inode] = gc.get_aoacl0_from_camber(airfoils[inode, :, 0], airfoils[inode, :, 1])
            mu0 = gc.get_mu0_from_camber(airfoils[inode, :, 0], airfoils[inode, :, 1])
            r = np.linalg.norm(blade.StructuralInformation.coordinates[inode, :])
            vrel = np.sqrt(rotation_velocity**2*r**2 + wsp**2)
            if inode == 0:
                dr = 0.5*np.linalg.norm(blade.StructuralInformation.coordinates[1,:] - blade.StructuralInformation.coordinates[0,:])
            elif inode == len(blade.StructuralInformation.coordinates[:,0]) - 1:
                dr = 0.5*np.linalg.norm(blade.StructuralInformation.coordinates[-1,:] - blade.StructuralInformation.coordinates[-2,:])
            else:
                dr = 0.5*np.linalg.norm(blade.StructuralInformation.coordinates[inode + 1,:] - blade.StructuralInformation.coordinates[inode - 1,:])
            moment_factor = 0.5*vrel**2*node_chord[inode]**2*dr
            # print("node", inode, "mu0", mu0, "CMc/4", 2.*mu0 + np.pi/2*node_twist[inode])
            blade.StructuralInformation.app_forces[inode, 3] = (2.*mu0 + np.pi/2*node_twist[inode])*moment_factor
            airfoils[inode, :, 1] *= 0.

    # Write SHARPy format
    blade.AerodynamicInformation.create_aerodynamics_from_vec(blade.StructuralInformation,
                                                            aero_node,
                                                            node_chord,
                                                            node_twist,
                                                            np.pi*np.ones_like(node_chord),
                                                            chord_panels,
                                                            surface_distribution,
                                                            m_distribution,
                                                            node_ElAxisAftLEc,
                                                            airfoil_distribution,
                                                            airfoils,
                                                            udmd_by_nodes)

    ######################################################################
    ## ROTOR
    ######################################################################

    # Read from excel file
    numberOfBlades = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'NumBl')
    tilt = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'ShftTilt')*deg2rad
    cone = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'Cone')*deg2rad
    # pitch = gc.read_column_sheet_type01(excel_file_name, excel_sheet_rotor, 'Pitch')*deg2rad

    # Apply pitch
    blade.StructuralInformation.rotate_around_origin(np.array([1.,0.,0.]), -pitch_deg*deg2rad)

    # Apply coning
    blade.StructuralInformation.rotate_around_origin(np.array([0.,1.,0.]), -cone)

    # Build the whole rotor
    rotor = blade.copy()
    for iblade in range(numberOfBlades-1):
        blade2 = blade.copy()
        blade2.StructuralInformation.rotate_around_origin(np.array([0.,0.,1.]), (iblade+1)*(360.0/numberOfBlades)*deg2rad)
        rotor.assembly(blade2)
        blade2 = None

    rotor.remove_duplicated_points(tol_remove_points)

    # Apply tilt
    rotor.StructuralInformation.rotate_around_origin(np.array([0.,1.,0.]), tilt)

    return rotor


def generate_from_excel_type02(chord_panels,
                                  rotation_velocity,
                                  pitch_deg,
                                  excel_file_name= 'database_excel_type02.xlsx',
                                  excel_sheet_parameters = 'parameters',
                                  excel_sheet_structural_blade = 'structural_blade',
                                  excel_sheet_discretization_blade = 'discretization_blade',
                                  excel_sheet_aero_blade = 'aero_blade',
                                  excel_sheet_airfoil_info = 'airfoil_info',
                                  excel_sheet_airfoil_coord = 'airfoil_coord',
                                  excel_sheet_structural_tower = 'structural_tower',
                                  m_distribution = 'uniform',
                                  h5_cross_sec_prop = None,
                                  n_points_camber = 100,
                                  tol_remove_points = 1e-3,
                                  user_defined_m_distribution_type = None,
                                  wsp = 0.,
                                  dt = 0.):

    """
    generate_from_excel_type02

    Function needed to generate a wind turbine from an excel database according to OpenFAST inputs

    Args:
    	  chord_panels (int): Number of panels on the blade surface in the chord direction
          rotation_velocity (float): Rotation velocity of the rotor
          pitch_deg (float): pitch angle in degrees
          excel_file_name (str):
          excel_sheet_structural_blade (str):
          excel_sheet_aero_blade (str):
          excel_sheet_airfoil_coord (str):
          excel_sheet_parameters (str):
          excel_sheet_structural_tower (str):
          m_distribution (str):
          n_points_camber (int): number of points to define the camber of the airfoil,
          tol_remove_points (float): maximum distance to remove adjacent points

    Returns:
        wt (sharpy.utils.generate_cases.AeroelasticInfromation): Aeroelastic infrmation of the wind turbine
        LC (list): list of all the Lagrange constraints needed in the cases (sharpy.utils.generate_cases.LagrangeConstraint)
        MB (list): list of the multibody information of each body (sharpy.utils.generate_cases.BodyInfrmation)
    """

    rotor = rotor_from_excel_type02(chord_panels,
                                  rotation_velocity,
                                  pitch_deg,
                                  excel_file_name= excel_file_name,
                                  excel_sheet_parameters = excel_sheet_parameters,
                                  excel_sheet_structural_blade = excel_sheet_structural_blade,
                                  excel_sheet_discretization_blade = excel_sheet_discretization_blade,
                                  excel_sheet_aero_blade = excel_sheet_aero_blade,
                                  excel_sheet_airfoil_info = excel_sheet_airfoil_info,
                                  excel_sheet_airfoil_coord = excel_sheet_airfoil_coord,
                                  m_distribution = m_distribution,
                                  h5_cross_sec_prop = h5_cross_sec_prop,
                                  n_points_camber = n_points_camber,
                                  tol_remove_points = tol_remove_points,
                                  user_defined_m_distribution = user_defined_m_distribution,
                                  wsp = 0.,
                                  dt = 0.)

    ######################################################################
    ## TOWER
    ######################################################################

    # Read from excel file
    HtFract = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_tower, 'HtFract')
    TMassDen = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_tower, 'TMassDen')
    TwFAStif = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_tower, 'TwFAStif')
    TwSSStif = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_tower, 'TwSSStif')
    # TODO> variables to be defined
    TwGJStif = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_tower, 'TwGJStif')
    TwEAStif = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_tower, 'TwEAStif')
    TwFAIner = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_tower, 'TwFAIner')
    TwSSIner = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_tower, 'TwSSIner')
    TwFAcgOf = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_tower, 'TwFAcgOf')
    TwSScgOf = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_tower, 'TwSScgOf')

    # Define the TOWER
    TowerHt = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'TowerHt')
    Elevation = TowerHt*HtFract

    tower = gc.AeroelasticInformation()
    tower.StructuralInformation.num_elem = len(Elevation) - 2
    tower.StructuralInformation.num_node_elem = 3
    tower.StructuralInformation.compute_basic_num_node()

    # Interpolate excel variables into the correct locations
    node_r, elem_r = create_node_radial_pos_from_elem_centres(Elevation,
                                        tower.StructuralInformation.num_node,
                                        tower.StructuralInformation.num_elem,
                                        tower.StructuralInformation.num_node_elem)

    # Stiffness
    elem_EA = np.interp(elem_r,Elevation,TwEAStif)
    elem_EIz = np.interp(elem_r,Elevation,TwSSStif)
    elem_EIy = np.interp(elem_r,Elevation,TwFAStif)
    elem_GJ = np.interp(elem_r,Elevation,TwGJStif)
    # Stiffness: estimate unknown properties
    print('WARNING: The poisson cofficient is assumed equal to 0.3')
    print('WARNING: Cross-section area is used as shear area')
    poisson_coef = 0.3
    elem_GAy = elem_EA/2.0/(1.0+poisson_coef)
    elem_GAz = elem_EA/2.0/(1.0+poisson_coef)

    # Inertia
    elem_mass_per_unit_length = np.interp(elem_r,Elevation,TMassDen)
    elem_mass_iner_y = np.interp(elem_r,Elevation,TwFAIner)
    elem_mass_iner_z = np.interp(elem_r,Elevation,TwSSIner)
    # TODO: check yz axis and Flap-edge
    elem_pos_cg_B = np.zeros((tower.StructuralInformation.num_elem,3),)
    elem_pos_cg_B[:,1]=np.interp(elem_r,Elevation,TwSScgOf)
    elem_pos_cg_B[:,2]=np.interp(elem_r,Elevation,TwFAcgOf)

    # Stiffness: estimate unknown properties
    print('WARNING: Using perpendicular axis theorem to compute the inertia around xB')
    elem_mass_iner_x = elem_mass_iner_y + elem_mass_iner_z

    # Create the tower
    tower.StructuralInformation.create_mass_db_from_vector(elem_mass_per_unit_length, elem_mass_iner_x, elem_mass_iner_y, elem_mass_iner_z, elem_pos_cg_B)
    tower.StructuralInformation.create_stiff_db_from_vector(elem_EA, elem_GAy, elem_GAz, elem_GJ, elem_EIy, elem_EIz)

    coordinates = np.zeros((tower.StructuralInformation.num_node,3),)
    coordinates[:,0] = node_r

    tower.StructuralInformation.generate_1to1_from_vectors(
            num_node_elem = tower.StructuralInformation.num_node_elem,
            num_node = tower.StructuralInformation.num_node,
            num_elem = tower.StructuralInformation.num_elem,
            coordinates = coordinates,
            stiffness_db = tower.StructuralInformation.stiffness_db,
            mass_db = tower.StructuralInformation.mass_db,
            frame_of_reference_delta = 'y_AFoR',
            vec_node_structural_twist = np.zeros((tower.StructuralInformation.num_node,),),
            num_lumped_mass = 1)

    tower.StructuralInformation.boundary_conditions = np.zeros((tower.StructuralInformation.num_node), dtype = int)
    tower.StructuralInformation.boundary_conditions[0] = 1

    # Read overhang and nacelle properties from excel file
    overhang_len = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'overhang')
    # HubMass = gc.read_column_sheet_type01(excel_file_name, excel_sheet_nacelle, 'HubMass')
    NacelleMass = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'NacMass')
    # NacelleYawIner = gc.read_column_sheet_type01(excel_file_name, excel_sheet_nacelle, 'NacelleYawIner')

    # Include nacelle mass
    tower.StructuralInformation.lumped_mass_nodes = np.array([tower.StructuralInformation.num_node-1], dtype=int)
    tower.StructuralInformation.lumped_mass = np.array([NacelleMass], dtype=float)

    tower.AerodynamicInformation.set_to_zero(tower.StructuralInformation.num_node_elem,
                                            tower.StructuralInformation.num_node,
                                            tower.StructuralInformation.num_elem)

    # Assembly overhang with the tower
    # numberOfBlades = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'NumBl')
    tilt = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'ShftTilt')*deg2rad
    # cone = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'Cone')*deg2rad

    overhang = gc.AeroelasticInformation()
    overhang.StructuralInformation.num_node = 3
    overhang.StructuralInformation.num_node_elem = 3
    overhang.StructuralInformation.compute_basic_num_elem()
    node_pos = np.zeros((overhang.StructuralInformation.num_node,3), )
    node_pos[:,0] += tower.StructuralInformation.coordinates[-1,0]
    node_pos[:,0] += np.linspace(0.,overhang_len*np.sin(tilt*deg2rad), overhang.StructuralInformation.num_node)
    node_pos[:,2] = np.linspace(0.,-overhang_len*np.cos(tilt*deg2rad), overhang.StructuralInformation.num_node)
    # TODO: change the following by real values
    # Same properties as the last element of the tower
    print("WARNING: Using the structural properties of the last tower section for the overhang")
    oh_mass_per_unit_length = tower.StructuralInformation.mass_db[-1,0,0]
    oh_mass_iner = tower.StructuralInformation.mass_db[-1,3,3]
    oh_EA = tower.StructuralInformation.stiffness_db[-1,0,0]
    oh_GA = tower.StructuralInformation.stiffness_db[-1,1,1]
    oh_GJ = tower.StructuralInformation.stiffness_db[-1,3,3]
    oh_EI = tower.StructuralInformation.stiffness_db[-1,4,4]
    overhang.StructuralInformation.generate_uniform_sym_beam(node_pos,
                                                            oh_mass_per_unit_length,
                                                            oh_mass_iner,
                                                            oh_EA,
                                                            oh_GA,
                                                            oh_GJ,
                                                            oh_EI,
                                                            num_node_elem = 3,
                                                            y_BFoR = 'y_AFoR',
                                                            num_lumped_mass=0)

    overhang.StructuralInformation.boundary_conditions = np.zeros((overhang.StructuralInformation.num_node), dtype = int)
    overhang.StructuralInformation.boundary_conditions[-1] = -1

    overhang.AerodynamicInformation.set_to_zero(overhang.StructuralInformation.num_node_elem,
                                                overhang.StructuralInformation.num_node,
                                                overhang.StructuralInformation.num_elem)

    tower.assembly(overhang)
    tower.remove_duplicated_points(tol_remove_points)

    ######################################################################
    ##  WIND TURBINE
    ######################################################################
    # Assembly the whole case
    wt = tower.copy()
    hub_position = tower.StructuralInformation.coordinates[-1,:]
    rotor.StructuralInformation.coordinates += hub_position
    wt.assembly(rotor)

    # Redefine the body numbers
    wt.StructuralInformation.body_number *= 0
    wt.StructuralInformation.body_number[tower.StructuralInformation.num_elem:wt.StructuralInformation.num_elem] += 1

    ######################################################################
    ## MULTIBODY
    ######################################################################
    # Define the boundary condition between the rotor and the tower tip
    LC1 = gc.LagrangeConstraint()
    LC1.behaviour = 'hinge_node_FoR_constant_vel'
    LC1.node_in_body = tower.StructuralInformation.num_node-1
    LC1.body = 0
    LC1.body_FoR = 1
    LC1.rot_axisB = np.array([1.,0.,0.0])
    LC1.rot_vel = -rotation_velocity

    LC = []
    LC.append(LC1)

    # Define the multibody infromation for the tower and the rotor
    MB1 = gc.BodyInformation()
    MB1.body_number = 0
    MB1.FoR_position = np.zeros((6,),)
    MB1.FoR_velocity = np.zeros((6,),)
    MB1.FoR_acceleration = np.zeros((6,),)
    MB1.FoR_movement = 'prescribed'
    MB1.quat = np.array([1.0,0.0,0.0,0.0])

    MB2 = gc.BodyInformation()
    MB2.body_number = 1
    MB2.FoR_position = np.array([rotor.StructuralInformation.coordinates[0, 0], rotor.StructuralInformation.coordinates[0, 1], rotor.StructuralInformation.coordinates[0, 2], 0.0, 0.0, 0.0])
    MB2.FoR_velocity = np.array([0.,0.,0.,0.,0.,rotation_velocity])
    MB2.FoR_acceleration = np.zeros((6,),)
    MB2.FoR_movement = 'free'
    MB2.quat = algebra.euler2quat(np.array([0.0,tilt,0.0]))

    MB = []
    MB.append(MB1)
    MB.append(MB2)

    ######################################################################
    ## RETURN
    ######################################################################
    return wt, LC, MB
