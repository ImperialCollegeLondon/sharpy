"""
template_wt

Functions needed to generate a wind turbines

Notes:
    To load this library: import cases.templates.template_wt as template_wt
"""

import sharpy.utils.generate_cases as gc
import pandas as pd
import numpy as np
import math
import os
import sharpy.utils.algebra as algebra

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


def create_blade_coordinates(num_node, node_r, node_prebending, node_presweept):
    """
    create_blade_coordinates

    Creates SHARPy format of the nodes coordinates and
    applies prebending and presweept to node radial position

    Args:
        num_node (int): number of nodes
        node_r (np.array): Radial position of the nodes
        node_prebending (np.array): Displacement of each point OUT OF the rotation plane
        node_presweept (np.array): Displacement of each point IN the rotation plane

    Returns:
        coordinates (np.array): nodes coordinates
    """
    coordinates = np.zeros((num_node,3),)
    coordinates[:,0] = node_r
    coordinates[:,1] = -1.0*node_presweept
    coordinates[:,2] = node_prebending
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
    print('WARNING: The poisson cofficient is supossed equal to 0.3')
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
    excel_db = pd.read_excel(xls, sheetname=excel_sheet_airfoil_coord)
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
    print('WARNING: The poisson cofficient is supossed equal to 0.3')
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
    blade.StructuralInformation.num_elem = len(Radius) - 2
    blade.StructuralInformation.num_node_elem = 3
    blade.StructuralInformation.compute_basic_num_node()

    node_r, elem_r = create_node_radial_pos_from_elem_centres(Radius,
                                        blade.StructuralInformation.num_node,
                                        blade.StructuralInformation.num_elem,
                                        blade.StructuralInformation.num_node_elem)
    # TODO: how is this defined now?
    node_prebending = np.interp(node_r,excel_aero_r,BlCrvAC)
    node_presweept = np.interp(node_r,excel_aero_r,BlSwpAC)

    # node_structural_twist = -1.0*np.interp(node_r,Radius,StrcTwst)
    node_structural_twist = -1.0*np.interp(node_r,excel_aero_r,BlTwist)
    node_pitch_axis = np.interp(node_r,Radius,PitchAxis)
    # Stiffness
    elem_EA = np.interp(elem_r,Radius,EAStff)
    elem_EIy = np.interp(elem_r,Radius,FlpStff)
    elem_EIz = np.interp(elem_r,Radius,EdgStff)
    elem_GJ = np.interp(elem_r,Radius,GJStff)
    # Stiffness: estimate unknown properties
    print('WARNING: The poisson cofficient is supossed equal to 0.3')
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
    # excel_aerodynamic_twist = gc.read_column_sheet_type01(excel_file_name, excel_sheet_aero_blade, 'BlTwist')*deg2rad
    excel_chord = gc.read_column_sheet_type01(excel_file_name, excel_sheet_aero_blade, 'BlChord')
    pure_airfoils_names = gc.read_column_sheet_type01(excel_file_name, excel_sheet_aero_blade, 'BlAFID')

    # Read coordinates of the pure airfoils
    n_elem_aero = len(excel_aero_r)
    # TODO: change this with a list of thickness and pure airfoils
    pure_airfoils_camber=np.zeros((n_elem_aero,n_points_camber,2),)
    xls = pd.ExcelFile(excel_file_name)
    excel_db = pd.read_excel(xls, sheetname=excel_sheet_airfoil_coord)
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
    print('WARNING: The poisson cofficient is supossed equal to 0.3')
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
