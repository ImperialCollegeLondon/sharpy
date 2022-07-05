"""
template_wt

Functions needed to generate a wind turbines

Notes:
    To load this library: import sharpy.cases.templates.template_wt as template_wt
"""


import pandas as pd
import numpy as np
import scipy.interpolate as scint
import math
import sys

import sharpy.utils.generate_cases as gc
import sharpy.utils.algebra as algebra
import sharpy.utils.h5utils as h5
from sharpy.utils.constants import deg2rad
import sharpy.aero.utils.airfoilpolars as ap
import sharpy.utils.cout_utils as cout

if not cout.check_running_unittest():
    cout.cout_wrap.print_screen = True
    cout.cout_wrap.print_file = False


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
    coordinates = np.zeros((num_node, 3),)
    coordinates[:, 0] = node_r
    coordinates[:, 1] = node_y
    coordinates[:, 2] = node_z
    return coordinates


######################################################################
# FROM excel type04
######################################################################
def spar_from_excel_type04(op_params,
                            geom_params,
                            excel_description,
                            options):
    """
    spar_from_excel_type04

    Function needed to generate a spar floating wind turbine rotor from an excel database type04.
    Rotor + tower + floating spar

    Args:
        op_param (dict): Dictionary with operating parameters
        geom_param (dict): Dictionray with geometical parameters
        excel_description (dict): Dictionary describing the sheets of the excel file
        option (dict): Dictionary with the different options for the wind turbine generation

    Returns:
        floating (sharpy.utils.generate_cases.AeroelasticInfromation): Aeroelastic information of the spar floating wind turbine
    """

    # Generate the tower + rotor
    wt, LC, MB, hub_nodes = generate_from_excel_type03(op_params,
                                    geom_params,
                                    excel_description,
                                    options)
    # Remove base clam
    wt.StructuralInformation.boundary_conditions[0] = 0

    excel_file_name = excel_description['excel_file_name']
    excel_sheet_parameters = excel_description['excel_sheet_parameters']
    excel_sheet_structural_spar = excel_description['excel_sheet_structural_spar']
    tol_remove_points = geom_params['tol_remove_points']

    TowerBaseHeight = gc.read_column_sheet_type01(excel_file_name,
                                                  excel_sheet_parameters,
                                                  'TowerBaseHeight')

    # Generate the spar
    if options['concentrate_spar']:
        mtower = wt.StructuralInformation.mass_db[0, 0, 0]

        PlatformTotalMass = gc.read_column_sheet_type01(excel_file_name,
                                                 excel_sheet_parameters,
                                                 'PlatformTotalMass')
        PlatformIrollCM = gc.read_column_sheet_type01(excel_file_name,
                                                 excel_sheet_parameters,
                                                 'PlatformIrollCM')
        PlatformIpitchCM = gc.read_column_sheet_type01(excel_file_name,
                                                 excel_sheet_parameters,
                                                 'PlatformIpitchCM')
        PlatformIyawCM = gc.read_column_sheet_type01(excel_file_name,
                                                 excel_sheet_parameters,
                                                 'PlatformIyawCM')
        PlatformCMbelowSWL = gc.read_column_sheet_type01(excel_file_name,
                                                 excel_sheet_parameters,
                                                 'PlatformCMbelowSWL')

        mpoint = PlatformTotalMass - mtower*TowerBaseHeight
        IyawPoint = PlatformIyawCM - (1./6)*mtower*TowerBaseHeight**3
        IpitchPoint = PlatformIpitchCM + PlatformTotalMass*PlatformCMbelowSWL**2 - (1./3)*mtower*TowerBaseHeight**3
        IrollPoint = PlatformIrollCM + PlatformTotalMass*PlatformCMbelowSWL**2 - (1./3)*mtower*TowerBaseHeight**3
        xpoint = (PlatformTotalMass*PlatformCMbelowSWL + 0.5*mtower*TowerBaseHeight**2)/mpoint

        iner_mat = np.zeros((6,6))
        iner_mat[0:3, 0:3] = mpoint*np.eye(3)
        iner_mat[0:3, 3:6] = -mpoint*algebra.skew(np.array([-xpoint, 0, 0]))
        iner_mat[3:6, 0:3] = -iner_mat[0:3, 3:6]
        iner_mat[3, 3] = IyawPoint
        iner_mat[4, 4] = IpitchPoint
        iner_mat[5, 5] = IrollPoint

        base_stiffness_top = wt.StructuralInformation.stiffness_db[0, :, :]
        base_mass_top = wt.StructuralInformation.mass_db[0, :, :]
        base_stiffness_bot = 100*base_stiffness_top
        base_mass_bot = base_mass_top

        num_lumped_mass_mat = 1

    else:
        SparHeight = gc.read_column_sheet_type01(excel_file_name,
                                                 excel_sheet_parameters,
                                                 'SparHeight')
        SparBallastMass = gc.read_column_sheet_type01(excel_file_name,
                                                      excel_sheet_parameters,
                                                      'SparBallastMass')
        SparBallastDepth = gc.read_column_sheet_type01(excel_file_name,
                                                       excel_sheet_parameters,
                                                       'SparBallastDepth')

        # Read from excel file
        SparHtFract = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_spar, 'SparHtFract')
        SparMassDen = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_spar, 'SparMassDen')
        SparFAStif = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_spar, 'SparFAStif')
        SparSSStif = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_spar, 'SparSSStif')
        SparGJStif = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_spar, 'SparGJStif')
        SparEAStif = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_spar, 'SparEAStif')
        SparFAIner = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_spar, 'SparFAIner')
        SparSSIner = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_spar, 'SparSSIner')
        SparFAcgOf = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_spar, 'SparFAcgOf')
        SparSScgOf = gc.read_column_sheet_type01(excel_file_name, excel_sheet_structural_spar, 'SparSScgOf')

        ElevationSpar = SparHtFract*SparHeight
        spar = gc.AeroelasticInformation()
        spar.StructuralInformation.num_elem = len(ElevationSpar) - 2
        spar.StructuralInformation.num_node_elem = 3
        spar.StructuralInformation.compute_basic_num_node()

        # Interpolate excel variables into the correct locations
        node_r, elem_r = create_node_radial_pos_from_elem_centres(ElevationSpar,
                                        spar.StructuralInformation.num_node,
                                        spar.StructuralInformation.num_elem,
                                        spar.StructuralInformation.num_node_elem)

        # Stiffness
        elem_EA = np.interp(elem_r, ElevationSpar, SparEAStif)
        elem_EIz = np.interp(elem_r, ElevationSpar, SparSSStif)
        elem_EIy = np.interp(elem_r, ElevationSpar, SparFAStif)
        elem_GJ = np.interp(elem_r, ElevationSpar, SparGJStif)
        # Stiffness: estimate unknown properties
        cout.cout_wrap.print_file = False
        cout.cout_wrap('WARNING: The poisson cofficient is assumed equal to 0.3', 3)
        cout.cout_wrap('WARNING: Cross-section area is used as shear area', 3)
        poisson_coef = 0.3
        elem_GAy = elem_EA/2.0/(1.0+poisson_coef)
        elem_GAz = elem_EA/2.0/(1.0+poisson_coef)

        # Inertia
        elem_mass_per_unit_length = np.interp(elem_r, ElevationSpar, SparMassDen)
        elem_mass_iner_y = np.interp(elem_r, ElevationSpar, SparFAIner)
        elem_mass_iner_z = np.interp(elem_r, ElevationSpar, SparSSIner)
        # TODO: check yz axis and Flap-edge
        elem_pos_cg_B = np.zeros((spar.StructuralInformation.num_elem, 3),)
        elem_pos_cg_B[:, 1] = np.interp(elem_r, ElevationSpar, SparSScgOf)
        elem_pos_cg_B[:, 2] = np.interp(elem_r, ElevationSpar, SparFAcgOf)

        # Stiffness: estimate unknown properties
        cout.cout_wrap('WARNING: Using perpendicular axis theorem to compute the inertia around xB', 3)
        elem_mass_iner_x = elem_mass_iner_y + elem_mass_iner_z

        # Create the tower
        spar.StructuralInformation.create_mass_db_from_vector(elem_mass_per_unit_length, elem_mass_iner_x, elem_mass_iner_y, elem_mass_iner_z, elem_pos_cg_B)
        spar.StructuralInformation.create_stiff_db_from_vector(elem_EA, elem_GAy, elem_GAz, elem_GJ, elem_EIy, elem_EIz)

        coordinates = np.zeros((spar.StructuralInformation.num_node, 3),)
        coordinates[:, 0] = node_r

        spar.StructuralInformation.generate_1to1_from_vectors(
            num_node_elem=spar.StructuralInformation.num_node_elem,
            num_node=spar.StructuralInformation.num_node,
            num_elem=spar.StructuralInformation.num_elem,
            coordinates=coordinates,
            stiffness_db=spar.StructuralInformation.stiffness_db,
            mass_db=spar.StructuralInformation.mass_db,
            frame_of_reference_delta='y_AFoR',
            vec_node_structural_twist=np.zeros((spar.StructuralInformation.num_node,),),
            num_lumped_mass=1)

        spar.StructuralInformation.boundary_conditions[0] = -1
        spar.StructuralInformation.boundary_conditions[-1] = 1

        # Include ballast mass
        dist = np.abs(coordinates[:, 0] + SparBallastDepth)
        min_dist = np.amin(dist)
        loc_min = np.where(dist == min_dist)[0]

        cout.cout_wrap("Ballast at node %d at position %f" % (loc_min, coordinates[loc_min, 0]), 2)

        spar.StructuralInformation.lumped_mass_nodes[0] = loc_min
        spar.StructuralInformation.lumped_mass[0] = SparBallastMass
        spar.StructuralInformation.lumped_mass_inertia[0] = np.zeros((3,3))
        spar.StructuralInformation.lumped_mass_position[0] = np.zeros((3,))

        spar.AerodynamicInformation.set_to_zero(spar.StructuralInformation.num_node_elem,
                                            spar.StructuralInformation.num_node,
                                            spar.StructuralInformation.num_elem)

        base_stiffness_bot = spar.StructuralInformation.stiffness_db[-1, :, :]
        base_mass_bot = spar.StructuralInformation.mass_db[-1, :, :]
        base_stiffness_top = base_stiffness_bot
        base_mass_top = base_mass_bot

        num_lumped_mass_mat = 0

    # Generate tower base
    NodesBase = gc.read_column_sheet_type01(excel_file_name,
                                                  excel_sheet_parameters,
                                                  'NodesBase')

    base = gc.AeroelasticInformation()
    base.StructuralInformation.num_node = NodesBase
    base.StructuralInformation.num_node_elem = 3
    base.StructuralInformation.compute_basic_num_elem()

    node_coord = np.zeros((base.StructuralInformation.num_node, 3))
    node_coord[:, 0] = np.linspace(0., TowerBaseHeight, base.StructuralInformation.num_node)

    base_stiffness_db = np.zeros((base.StructuralInformation.num_elem, 6, 6))
    base_mass_db = np.zeros((base.StructuralInformation.num_elem, 6, 6))
    vec_node_structural_twist = np.zeros((base.StructuralInformation.num_node,))

    # for ielem in range(base.StructuralInformation.num_elem):
    #     base_stiffness_db[ielem, :, :] = base_stiffness.copy()
    #     base_mass_db[ielem, :, :] = base_mass.copy()
    for ielem in range(base.StructuralInformation.num_elem):
        inode_cent = 2*ielem + 1
        rel_dist_to_bot = node_coord[inode_cent, 0]/TowerBaseHeight
        rel_dist_to_top = (node_coord[-1, 0] - node_coord[inode_cent, 0])/TowerBaseHeight
        base_stiffness_db[ielem, :, :] = (base_stiffness_bot*rel_dist_to_top +
                                      base_stiffness_top*rel_dist_to_bot)
        base_mass_db[ielem, :, :] = (base_mass_bot*rel_dist_to_top +
                                     base_mass_top*rel_dist_to_bot)

    base.StructuralInformation.generate_1to1_from_vectors(
                            base.StructuralInformation.num_node_elem,
                            base.StructuralInformation.num_node,
                            base.StructuralInformation.num_elem,
                            node_coord,
                            base_stiffness_db,
                            base_mass_db,
                            'y_AFoR',
                            vec_node_structural_twist,
                            num_lumped_mass=0,
                            num_lumped_mass_mat=num_lumped_mass_mat)

    base.StructuralInformation.boundary_conditions[0] = 1
    base.StructuralInformation.body_number *= 0

    base.AerodynamicInformation.set_to_zero(base.StructuralInformation.num_node_elem,
                                        base.StructuralInformation.num_node,
                                        base.StructuralInformation.num_elem)

    if options['concentrate_spar']:
        base.StructuralInformation.lumped_mass_mat_nodes = np.array([0], dtype=int)
        base.StructuralInformation.lumped_mass_mat = np.zeros((1, 6, 6))
        base.StructuralInformation.lumped_mass_mat[0, :, :] = iner_mat

        nodes_base = base.StructuralInformation.num_node + 0
        for inode in range(len(hub_nodes)):
            hub_nodes[inode] += nodes_base
        wt.StructuralInformation.coordinates[:, 0] += TowerBaseHeight
        base.assembly(wt)
        base.remove_duplicated_points(1e-6, skip=hub_nodes)
        for ielem in range(base.StructuralInformation.num_elem):
            if not base.StructuralInformation.body_number[ielem] == 0:
                base.StructuralInformation.body_number[ielem] -= 1
        for ilc in range(len(LC)):
            LC[ilc].node_in_body += nodes_base - 1
        spar = base # Just a rename for the return
    else:
        # Assembly
        spar.assembly(base)
        spar.remove_duplicated_points(1e-6)
        spar.StructuralInformation.body_number *= 0
        nodes_spar = spar.StructuralInformation.num_node + 0
        for inode in range(len(hub_nodes)):
            hub_nodes[inode] += nodes_spar

        wt.StructuralInformation.coordinates[:, 0] += TowerBaseHeight
        spar.assembly(wt)
        spar.remove_duplicated_points(1e-6, skip=hub_nodes)
        for ielem in range(spar.StructuralInformation.num_elem):
            if not spar.StructuralInformation.body_number[ielem] == 0:
                spar.StructuralInformation.body_number[ielem] -= 1

        # Update Lagrange Constraints and Multibody information
        for ilc in range(len(LC)):
            LC[ilc].node_in_body += nodes_spar - 1

    MB[0].FoR_movement = 'free'
    for ibody in range(1, len(MB)):
        MB[ibody].FoR_position[0] += TowerBaseHeight

    return spar, LC, MB

######################################################################
# FROM excel type03
######################################################################
def rotor_from_excel_type03(in_op_params,
                            in_geom_params,
                            in_excel_description,
                            in_options):
    """
    rotor_from_excel_type03

    Function needed to generate a wind turbine rotor from an excel database type03

    Args:
        op_param (dict): Dictionary with operating parameters
        geom_param (dict): Dictionray with geometical parameters
        excel_description (dict): Dictionary describing the sheets of the excel file
        option (dict): Dictionary with the different options for the wind turbine generation

    Returns:
        rotor (sharpy.utils.generate_cases.AeroelasticInfromation): Aeroelastic information of the rotor
    """
    # Default values
    op_params = {}
    op_params['rotation_velocity'] = None # Rotation velocity of the rotor
    op_params['pitch_deg'] = None # pitch angle in degrees
    op_params['wsp'] = 0. # wind speed (It may be needed for discretisation purposes)
    op_params['dt'] = 0. # time step (It may be needed for discretisation purposes)

    geom_params = {}
    geom_params['chord_panels'] = None # Number of panels on the blade surface in the chord direction
    geom_params['tol_remove_points'] = 1e-3 # maximum distance to remove adjacent points
    geom_params['n_points_camber'] = 100 # number of points to define the camber of the airfoil
    geom_params['h5_cross_sec_prop'] = None # h5 containing mass and stiffness matrices along the blade
    geom_params['m_distribution'] = 'uniform' #

    options = {}
    options['camber_effect_on_twist'] = False # When true plain airfoils are used and the blade is twisted and preloaded based on thin airfoil theory
    options['user_defined_m_distribution_type'] = None # type of distribution of the chordwise panels when 'm_distribution' == 'user_defined'
    options['include_polars'] = False #
    options['separate_blades'] = False # Keep blades as different bodies
    options['twist_in_aero'] = False # Twist the aerodynamic surface instead of the structure

    excel_description = {}
    excel_description['excel_file_name'] = 'database_excel_type02.xlsx'
    excel_description['excel_sheet_parameters'] = 'parameters'
    excel_description['excel_sheet_structural_blade'] = 'structural_blade'
    excel_description['excel_sheet_discretization_blade'] = 'discretization_blade'
    excel_description['excel_sheet_aero_blade'] = 'aero_blade'
    excel_description['excel_sheet_airfoil_info'] = 'airfoil_info'
    excel_description['excel_sheet_airfoil_chord'] = 'airfoil_coord'

    # Overwrite the default values with the values of the input arguments
    for key in in_op_params:
        op_params[key] = in_op_params[key]
    for key in in_geom_params:
        geom_params[key] = in_geom_params[key]
    for key in in_options:
        options[key] = in_options[key]
    for key in in_excel_description:
        excel_description[key] = in_excel_description[key]

    # Put the dictionaries information into variables (to avoid changing the function)
    rotation_velocity = op_params['rotation_velocity']
    pitch_deg = op_params['pitch_deg']
    wsp = op_params['wsp']
    dt = op_params['dt']

    chord_panels = geom_params['chord_panels']
    tol_remove_points = geom_params['tol_remove_points']
    n_points_camber = geom_params['n_points_camber']
    h5_cross_sec_prop = geom_params['h5_cross_sec_prop']
    m_distribution = geom_params['m_distribution']

    camber_effect_on_twist = options['camber_effect_on_twist']
    user_defined_m_distribution_type = options['user_defined_m_distribution_type']
    include_polars = options['include_polars']

    excel_file_name = excel_description['excel_file_name']
    excel_sheet_parameters = excel_description['excel_sheet_parameters']
    excel_sheet_structural_blade = excel_description['excel_sheet_structural_blade']
    excel_sheet_discretization_blade = excel_description['excel_sheet_discretization_blade']
    excel_sheet_aero_blade = excel_description['excel_sheet_aero_blade']
    excel_sheet_airfoil_info = excel_description['excel_sheet_airfoil_info']
    excel_sheet_airfoil_coord = excel_description['excel_sheet_airfoil_chord']

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
        InPElAxis = np.concatenate((np.array([InPElAxis[0]]), InPElAxis),)
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
            raise RuntimeError(("ERROR: Cannot build %d-noded elements from %d nodes" % (blade.StructuralInformation.num_node_elem, blade.StructuralInformation.num_node)))

    node_y = np.interp(rR, rR_structural, InPElAxis) + np.interp(rR, rR_structural, PreswpRef)
    node_z = -np.interp(rR, rR_structural, OutPElAxis) - np.interp(rR, rR_structural, PrebendRef)
    node_twist = -1.0*np.interp(rR, rR_structural, StrcTwst)

    coordinates = create_blade_coordinates(blade.StructuralInformation.num_node, node_r, node_y, node_z)

    if h5_cross_sec_prop is None:
        # Stiffness
        elem_EA = np.interp(elem_rR, rR_structural, EAStff)
        elem_EIy = np.interp(elem_rR, rR_structural, FlpStff)
        elem_EIz = np.interp(elem_rR, rR_structural, EdgStff)
        elem_EIyz = np.interp(elem_rR, rR_structural, FlapEdgeStiff)
        elem_GJ = np.interp(elem_rR, rR_structural, GJStff)

        # Stiffness: estimate unknown properties
        cout.cout_wrap.print_file = False
        cout.cout_wrap('WARNING: The poisson cofficient is assumed equal to 0.3', 3)
        cout.cout_wrap('WARNING: Cross-section area is used as shear area', 3)
        poisson_coef = 0.3
        elem_GAy = elem_EA/2.0/(1.0+poisson_coef)
        elem_GAz = elem_EA/2.0/(1.0+poisson_coef)
        # Inertia
        elem_pos_cg_B = np.zeros((blade.StructuralInformation.num_elem, 3),)
        elem_pos_cg_B[:, 1] = np.interp(elem_rR, rR_structural, InPcg)
        elem_pos_cg_B[:, 2] = -np.interp(elem_rR, rR_structural, OutPcg)

        elem_mass_per_unit_length = np.interp(elem_rR, rR_structural, BMassDen)
        elem_mass_iner_y = np.interp(elem_rR, rR_structural, FlpIner)
        elem_mass_iner_z = np.interp(elem_rR, rR_structural, EdgIner)
        elem_mass_iner_yz = np.interp(elem_rR, rR_structural, FlapEdgeIner)

        # Inertia: estimate unknown properties
        cout.cout_wrap('WARNING: Using perpendicular axis theorem to compute the inertia around xB', 3)
        elem_mass_iner_x = elem_mass_iner_y + elem_mass_iner_z

        # Generate blade structural properties
        blade.StructuralInformation.create_mass_db_from_vector(elem_mass_per_unit_length, elem_mass_iner_x, elem_mass_iner_y, elem_mass_iner_z, elem_pos_cg_B, elem_mass_iner_yz)
        blade.StructuralInformation.create_stiff_db_from_vector(elem_EA, elem_GAy, elem_GAz, elem_GJ, elem_EIy, elem_EIz, elem_EIyz)

    else:
        # read Mass/Stiffness from database
        cross_prop = h5.readh5(h5_cross_sec_prop).str_prop

        # create mass_db/stiffness_db (interpolate at mid-node of each element)
        blade.StructuralInformation.mass_db = scint.interp1d(
                    cross_prop.radius, cross_prop.M, kind='cubic', copy=False, assume_sorted=True, axis=0,
                                                    bounds_error=False, fill_value='extrapolate')(node_r[1::2])
        blade.StructuralInformation.stiffness_db = scint.interp1d(
                    cross_prop.radius, cross_prop.K, kind='cubic', copy=False, assume_sorted=True, axis=0,
                                                    bounds_error=False, fill_value='extrapolate')(node_r[1::2])

    blade.StructuralInformation.generate_1to1_from_vectors(
            num_node_elem=blade.StructuralInformation.num_node_elem,
            num_node=blade.StructuralInformation.num_node,
            num_elem=blade.StructuralInformation.num_elem,
            coordinates=coordinates,
            stiffness_db=blade.StructuralInformation.stiffness_db,
            mass_db=blade.StructuralInformation.mass_db,
            frame_of_reference_delta='y_AFoR',
            vec_node_structural_twist=np.zeros_like(node_twist) if options['twist_in_aero'] else node_twist,
            num_lumped_mass=0)

    # Boundary conditions
    blade.StructuralInformation.boundary_conditions = np.zeros((blade.StructuralInformation.num_node), dtype=int)
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

    node_ElAxisAftLEc = np.interp(node_r, rR_structural*TipRad, ElAxisAftLEc)

    # Read coordinates of the pure airfoils
    n_pure_airfoils = len(pure_airfoils_names)

    pure_airfoils_camber = np.zeros((n_pure_airfoils, n_points_camber, 2),)
    xls = pd.ExcelFile(excel_file_name)
    excel_db = pd.read_excel(xls, sheet_name=excel_sheet_airfoil_coord)
    for iairfoil in range(n_pure_airfoils):
        # Look for the NaN
        icoord = 2
        while(not(math.isnan(excel_db["%s_x" % pure_airfoils_names[iairfoil]][icoord]))):
            icoord += 1
            if(icoord == len(excel_db["%s_x" % pure_airfoils_names[iairfoil]])):
                break

        # Compute the camber of the airfoils at the defined chord points
        pure_airfoils_camber[iairfoil, :, 0], pure_airfoils_camber[iairfoil, :, 1] = gc.get_airfoil_camber(excel_db["%s_x" % pure_airfoils_names[iairfoil]][2:icoord],
                                                                                                           excel_db["%s_y" % pure_airfoils_names[iairfoil]][2:icoord],
                                                                                                           n_points_camber)

    # Basic variables
    surface_distribution = np.zeros((blade.StructuralInformation.num_elem), dtype=int)


    # Interpolate in the correct positions
    node_chord = np.interp(node_r, rR_aero*TipRad, chord_aero)

    # Define the nodes with aerodynamic properties
    # Look for the first element that is goint to be aerodynamic
    first_aero_elem = 0
    while (elem_r[first_aero_elem] <= rR_aero[0]*TipRad):
        first_aero_elem += 1
    first_aero_node = first_aero_elem*(blade.StructuralInformation.num_node_elem - 1)
    aero_node = np.zeros((blade.StructuralInformation.num_node,), dtype=bool)
    aero_node[first_aero_node:] = np.ones((blade.StructuralInformation.num_node-first_aero_node,), dtype=bool)

    # Define the airfoil at each stage
    # airfoils = blade.AerodynamicInformation.interpolate_airfoils_camber(pure_airfoils_camber,excel_aero_r, node_r, n_points_camber)

    node_thickness = np.interp(node_r, rR_aero*TipRad, thickness_aero)

    airfoils = blade.AerodynamicInformation.interpolate_airfoils_camber_thickness(pure_airfoils_camber, pure_airfoils_thickness, node_thickness, n_points_camber)
    airfoil_distribution = np.linspace(0, blade.StructuralInformation.num_node - 1, blade.StructuralInformation.num_node, dtype=int)

    # User defined m distribution
    if (m_distribution == 'user_defined') and (user_defined_m_distribution_type == 'last_geometric'):
        blade_nodes = blade.StructuralInformation.num_node
        udmd_by_nodes = np.zeros((blade_nodes, chord_panels[0] + 1))
        for inode in range(blade_nodes):
            r = np.linalg.norm(blade.StructuralInformation.coordinates[inode, :])
            vrel = np.sqrt(rotation_velocity**2*r**2 + wsp**2)
            last_length = vrel*dt/node_chord[inode]
            last_length = np.minimum(last_length, 0.5)
            if last_length <= 0.5:
                ratio = gc.get_factor_geometric_progression(last_length, 1., chord_panels)
                udmd_by_nodes[inode, -1] = 1.
                udmd_by_nodes[inode, 0] = 0.
                for im in range(chord_panels[0] - 1, 0, -1):
                    udmd_by_nodes[inode, im] = udmd_by_nodes[inode, im + 1] - last_length
                    last_length *= ratio
                # Check
                if (np.diff(udmd_by_nodes[inode, :]) < 0.).any():
                    sys.error("ERROR in the panel discretization of the blade in node %d" % (inode))
            else:
                raise RuntimeError(("ERROR: cannot match the last panel size for node: %d" % inode))
                udmd_by_nodes[inode, :] = np.linspace(0, 1, chord_panels + 1)
    else:
        udmd_by_nodes = None

    node_twist_aero = np.zeros_like(node_chord)
    if camber_effect_on_twist:
        cout.cout_wrap("WARNING: The steady applied Mx should be manually multiplied by the density", 3)
        for inode in range(blade.StructuralInformation.num_node):
            node_twist_aero[inode] = gc.get_aoacl0_from_camber(airfoils[inode, :, 0], airfoils[inode, :, 1])
            mu0 = gc.get_mu0_from_camber(airfoils[inode, :, 0], airfoils[inode, :, 1])
            r = np.linalg.norm(blade.StructuralInformation.coordinates[inode, :])
            vrel = np.sqrt(rotation_velocity**2*r**2 + wsp**2)
            if inode == 0:
                dr = 0.5*np.linalg.norm(blade.StructuralInformation.coordinates[1, :] - blade.StructuralInformation.coordinates[0, :])
            elif inode == len(blade.StructuralInformation.coordinates[:, 0]) - 1:
                dr = 0.5*np.linalg.norm(blade.StructuralInformation.coordinates[-1, :] - blade.StructuralInformation.coordinates[-2, :])
            else:
                dr = 0.5*np.linalg.norm(blade.StructuralInformation.coordinates[inode + 1, :] - blade.StructuralInformation.coordinates[inode - 1, :])
            moment_factor = 0.5*vrel**2*node_chord[inode]**2*dr
            # print("node", inode, "mu0", mu0, "CMc/4", 2.*mu0 + np.pi/2*node_twist_struct[inode])
            blade.StructuralInformation.app_forces[inode, 3] = (2.*mu0 + np.pi/2*node_twist_aero[inode])*moment_factor
            airfoils[inode, :, 1] *= 0.

    # Write SHARPy format
    blade.AerodynamicInformation.create_aerodynamics_from_vec(blade.StructuralInformation,
                                                            aero_node,
                                                            node_chord,
                                                            (node_twist + node_twist_aero) if options['twist_in_aero'] else node_twist_aero,
                                                            np.pi*np.ones_like(node_chord),
                                                            chord_panels,
                                                            surface_distribution,
                                                            m_distribution,
                                                            node_ElAxisAftLEc,
                                                            airfoil_distribution,
                                                            airfoils,
                                                            udmd_by_nodes,
                                                            first_twist=False)

    # Read the polars of the pure airfoils
    if include_polars:
        pure_polars = [None]*n_pure_airfoils
        for iairfoil in range(n_pure_airfoils):
            excel_sheet_polar = pure_airfoils_names[iairfoil]
            aoa = gc.read_column_sheet_type01(excel_file_name, excel_sheet_polar, 'AoA')
            cl = gc.read_column_sheet_type01(excel_file_name, excel_sheet_polar, 'CL')
            cd = gc.read_column_sheet_type01(excel_file_name, excel_sheet_polar, 'CD')
            cm = gc.read_column_sheet_type01(excel_file_name, excel_sheet_polar, 'CM')
            polar = ap.Polar()
            polar.initialise(np.column_stack((aoa, cl, cd, cm)))
            pure_polars[iairfoil] = polar

        # Generate the polars for each airfoil
        blade.AerodynamicInformation.polars = [None]*blade.StructuralInformation.num_node
        for inode in range(blade.StructuralInformation.num_node):
            # Find the airfoils between which the node is;
            ipure = 0
            while pure_airfoils_thickness[ipure] > node_thickness[inode]:
                ipure += 1
                if(ipure == n_pure_airfoils):
                    ipure -= 1
                    break

            coef = (node_thickness[inode] - pure_airfoils_thickness[ipure - 1])/(pure_airfoils_thickness[ipure] - pure_airfoils_thickness[ipure - 1])
            polar = ap.interpolate(pure_polars[ipure - 1], pure_polars[ipure], coef)
            blade.AerodynamicInformation.polars[inode] = polar.table

    ######################################################################
    ## ROTOR
    ######################################################################

    # Read from excel file
    numberOfBlades = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'NumBl')
    tilt = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'ShftTilt')*deg2rad
    cone = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'Cone')*deg2rad
    # pitch = gc.read_column_sheet_type01(excel_file_name, excel_sheet_rotor, 'Pitch')*deg2rad

    # Apply pitch
    blade.StructuralInformation.rotate_around_origin(np.array([1., 0., 0.]), -pitch_deg*deg2rad)

    # Apply coning
    blade.StructuralInformation.rotate_around_origin(np.array([0., 1., 0.]), -cone)

    # Build the whole rotor
    rotor = blade.copy()
    hub_nodes = [0]
    for iblade in range(numberOfBlades-1):
        hub_nodes.append((iblade + 1)*blade.StructuralInformation.num_node)
        blade2 = blade.copy()
        blade2.StructuralInformation.rotate_around_origin(np.array([0., 0., 1.]), (iblade + 1)*(360.0/numberOfBlades)*deg2rad)
        rotor.assembly(blade2)
        blade2 = None

    if not options['separate_blades']:
        rotor.remove_duplicated_points(tol_remove_points)
        hub_nodes = [0]
        rotor.StructuralInformation.body_number *= 0

    # Apply tilt
    rotor.StructuralInformation.rotate_around_origin(np.array([0., 1., 0.]), tilt)

    return rotor, hub_nodes


def generate_from_excel_type03(op_params,
                                   geom_params,
                                   excel_description,
                                   options):

    """
    generate_from_excel_type03

    Function needed to generate a wind turbine (tower + rotor) from an excel database type03.
    See ``rotor_from_excel_type03'' for more information.

    Args:
        op_param (dict): Dictionary with operating parameters
        geom_param (dict): Dictionray with geometical parameters
        excel_description (dict): Dictionary describing the sheets of the excel file
        option (dict): Dictionary with the different options for the wind turbine generation

    Returns:
        wt (sharpy.utils.generate_cases.AeroelasticInfromation): Aeroelastic information of the wind turbine (tower + rotor)
    """
    rotor, hub_nodes = rotor_from_excel_type03(op_params,
                                               geom_params,
                                               excel_description,
                                               options)


    excel_file_name = excel_description['excel_file_name']
    excel_sheet_parameters = excel_description['excel_sheet_parameters']
    excel_sheet_structural_tower = excel_description['excel_sheet_structural_tower']
    tol_remove_points = geom_params['tol_remove_points']
    rotation_velocity = op_params['rotation_velocity']

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
    elem_EA = np.interp(elem_r, Elevation, TwEAStif)
    elem_EIz = np.interp(elem_r, Elevation, TwSSStif)
    elem_EIy = np.interp(elem_r, Elevation, TwFAStif)
    elem_GJ = np.interp(elem_r, Elevation, TwGJStif)
    # Stiffness: estimate unknown properties
    cout.cout_wrap.print_file = False
    cout.cout_wrap('WARNING: The poisson cofficient is assumed equal to 0.3', 3)
    cout.cout_wrap('WARNING: Cross-section area is used as shear area', 3)
    poisson_coef = 0.3
    elem_GAy = elem_EA/2.0/(1.0+poisson_coef)
    elem_GAz = elem_EA/2.0/(1.0+poisson_coef)

    # Inertia
    elem_mass_per_unit_length = np.interp(elem_r, Elevation, TMassDen)
    elem_mass_iner_y = np.interp(elem_r, Elevation, TwFAIner)
    elem_mass_iner_z = np.interp(elem_r, Elevation, TwSSIner)
    # TODO: check yz axis and Flap-edge
    elem_pos_cg_B = np.zeros((tower.StructuralInformation.num_elem, 3),)
    elem_pos_cg_B[:, 1] = np.interp(elem_r, Elevation, TwSScgOf)
    elem_pos_cg_B[:, 2] = np.interp(elem_r, Elevation, TwFAcgOf)

    # Stiffness: estimate unknown properties
    cout.cout_wrap('WARNING: Using perpendicular axis theorem to compute the inertia around xB', 3)
    elem_mass_iner_x = elem_mass_iner_y + elem_mass_iner_z

    # Create the tower
    tower.StructuralInformation.create_mass_db_from_vector(elem_mass_per_unit_length, elem_mass_iner_x, elem_mass_iner_y, elem_mass_iner_z, elem_pos_cg_B)
    tower.StructuralInformation.create_stiff_db_from_vector(elem_EA, elem_GAy, elem_GAz, elem_GJ, elem_EIy, elem_EIz)

    coordinates = np.zeros((tower.StructuralInformation.num_node, 3),)
    coordinates[:, 0] = node_r

    tower.StructuralInformation.generate_1to1_from_vectors(
            num_node_elem=tower.StructuralInformation.num_node_elem,
            num_node=tower.StructuralInformation.num_node,
            num_elem=tower.StructuralInformation.num_elem,
            coordinates=coordinates,
            stiffness_db=tower.StructuralInformation.stiffness_db,
            mass_db=tower.StructuralInformation.mass_db,
            frame_of_reference_delta='y_AFoR',
            vec_node_structural_twist=np.zeros((tower.StructuralInformation.num_node,),),
            num_lumped_mass=1)

    tower.StructuralInformation.boundary_conditions = np.zeros((tower.StructuralInformation.num_node), dtype = int)
    tower.StructuralInformation.boundary_conditions[0] = 1

    # Nacelle properties from excel file
    NacelleMass = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'NacMass')
    NacelleMass_x = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'NacMass_x')
    NacelleMass_z = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'NacMass_z')
    # NacelleYawIner = gc.read_column_sheet_type01(excel_file_name, excel_sheet_nacelle, 'NacelleYawIner')

    # Include nacelle mass
    tower.StructuralInformation.lumped_mass_nodes = np.array([tower.StructuralInformation.num_node - 1], dtype=int)
    tower.StructuralInformation.lumped_mass = np.array([NacelleMass], dtype=float)
    if not NacelleMass_x is None and not NacelleMass_z is None:
        tower.StructuralInformation.lumped_mass_position = np.array([np.array([NacelleMass_z, 0, NacelleMass_x])], dtype=float)
    else:
        cout.cout_wrap('WARNING: Nacelle mass placed at tower top', 3)

    tower.AerodynamicInformation.set_to_zero(tower.StructuralInformation.num_node_elem,
                                            tower.StructuralInformation.num_node,
                                            tower.StructuralInformation.num_elem)

    # Overhang
    tilt = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'ShftTilt')*deg2rad
    if not tilt == 0.:
        raise NonImplementedError("Non-zero tilt not supported")
    NodesOverhang = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'NodesOverhang')
    overhang_len = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'overhang')
    if NodesOverhang == 0:
        with_overhang = False
    else:
        with_overhang = True

        overhang = gc.AeroelasticInformation()
        overhang.StructuralInformation.num_node = NodesOverhang
        overhang.StructuralInformation.num_node_elem = 3
        overhang.StructuralInformation.compute_basic_num_elem()
        node_pos = np.zeros((overhang.StructuralInformation.num_node, 3), )
        node_pos[:, 0] += tower.StructuralInformation.coordinates[-1, 0]
        node_pos[:, 0] += np.linspace(0., -overhang_len*np.sin(tilt*deg2rad), overhang.StructuralInformation.num_node)
        node_pos[:, 2] = np.linspace(0., overhang_len*np.cos(tilt*deg2rad), overhang.StructuralInformation.num_node)
        # TODO: change the following by real values
        # Same properties as the last element of the tower
        # cout.cout_wrap("WARNING: Using the structural properties of the last tower section for the overhang", 3)
        # oh_mass_per_unit_length = tower.StructuralInformation.mass_db[-1, 0, 0]
        # oh_mass_iner = tower.StructuralInformation.mass_db[-1, 3, 3]
        cout.cout_wrap("WARNING: Using the structural properties (*0.1) of the last tower section for the overhang", 3)
        oh_mass_per_unit_length = tower.StructuralInformation.mass_db[-1, 0, 0]/10.
        oh_mass_iner = tower.StructuralInformation.mass_db[-1, 3, 3]/10.
        oh_EA = tower.StructuralInformation.stiffness_db[-1, 0, 0]
        oh_GA = tower.StructuralInformation.stiffness_db[-1, 1, 1]
        oh_GJ = tower.StructuralInformation.stiffness_db[-1, 3, 3]
        oh_EI = tower.StructuralInformation.stiffness_db[-1, 4, 4]

        overhang.StructuralInformation.generate_uniform_sym_beam(node_pos,
                                                            oh_mass_per_unit_length,
                                                            oh_mass_iner,
                                                            oh_EA,
                                                            oh_GA,
                                                            oh_GJ,
                                                            oh_EI,
                                                            num_node_elem=3,
                                                            y_BFoR='y_AFoR',
                                                            num_lumped_mass=0)

        overhang.StructuralInformation.boundary_conditions[-1] = -1

        overhang.AerodynamicInformation.set_to_zero(overhang.StructuralInformation.num_node_elem,
                                                overhang.StructuralInformation.num_node,
                                                overhang.StructuralInformation.num_elem)

        tower.assembly(overhang)
        tower.remove_duplicated_points(tol_remove_points)
        tower.StructuralInformation.body_number *= 0

    # Hub mass
    HubMass = gc.read_column_sheet_type01(excel_file_name, excel_sheet_parameters, 'HubMass')
    if HubMass is not None:
        if with_overhang:
            tower.StructuralInformation.add_lumped_mass(tower.StructuralInformation.num_node -1,
                                  HubMass,
                                  inertia=np.zeros((3, 3)),
                                  pos=np.zeros((3)))
        else:
            n_hub_nodes = len(hub_nodes)
            for inode_hub in range(n_hub_nodes):
                rotor.StructuralInformation.add_lumped_mass(hub_nodes[inode_hub],
                                      HubMass/n_hub_nodes,
                                      inertia=np.zeros((3, 3)),
                                      pos=np.zeros((3)))
    else:
        cout.cout_wrap('WARNING: HubMass not found', 3)

    for inode in range(len(hub_nodes)):
        hub_nodes[inode] += tower.StructuralInformation.num_node

    ######################################################################
    ##  WIND TURBINE
    ######################################################################
    # Assembly the whole case
    wt = tower.copy()
    hub_position = tower.StructuralInformation.coordinates[-1, :]
    if not with_overhang:
        hub_position += np.array([0., 0., overhang_len])
    rotor.StructuralInformation.coordinates += hub_position
    wt.assembly(rotor)

    ######################################################################
    ## MULTIBODY
    ######################################################################
    LC = []
    for iblade in range(len(hub_nodes)):
        # Define the boundary condition between the rotor and the tower tip
        LC1 = gc.LagrangeConstraint()
        LC1.behaviour = 'hinge_node_FoR_constant_vel'
        LC1.node_in_body = tower.StructuralInformation.num_node - 1
        LC1.body = 0
        LC1.body_FoR = iblade + 1
        if with_overhang:
            LC1.rot_vect = np.array([-1., 0., 0.])*rotation_velocity
            LC1.rel_posB = np.zeros((3))
        else:
            LC1.rot_vect = np.array([0., 0., 1.])*rotation_velocity
            LC1.rel_posB = np.array([0., 0., overhang_len])
        LC.append(LC1)

    # Define the multibody infromation for the tower and the rotor
    MB = []
    MB1 = gc.BodyInformation()
    MB1.body_number = 0
    MB1.FoR_position = np.zeros((6,),)
    MB1.FoR_velocity = np.zeros((6,),)
    MB1.FoR_acceleration = np.zeros((6,),)
    MB1.FoR_movement = 'prescribed'
    MB1.quat = np.array([1.0, 0.0, 0.0, 0.0])
    MB.append(MB1)

    numberOfBlades = len(hub_nodes)
    for iblade in range(numberOfBlades):
        MB2 = gc.BodyInformation()
        MB2.body_number = iblade + 1
        MB2.FoR_position = np.concatenate((hub_position, np.zeros((3))))
        MB2.FoR_velocity = np.array([0., 0., 0., 0., 0., rotation_velocity])
        MB2.FoR_acceleration = np.zeros((6,),)
        MB2.FoR_movement = 'free'
        blade_azimuth = (iblade*(360.0/numberOfBlades)*deg2rad)
        MB2.quat = algebra.euler2quat(np.array([0.0, tilt, blade_azimuth]))
        MB.append(MB2)

    ######################################################################
    ## RETURN
    ######################################################################
    return wt, LC, MB, hub_nodes


######################################################################
# FROM excel type02
######################################################################
def rotor_from_excel_type02(chord_panels,
                            rotation_velocity,
                            pitch_deg,
                            excel_file_name='database_excel_type02.xlsx',
                            excel_sheet_parameters='parameters',
                            excel_sheet_structural_blade='structural_blade',
                            excel_sheet_discretization_blade='discretization_blade',
                            excel_sheet_aero_blade='aero_blade',
                            excel_sheet_airfoil_info='airfoil_info',
                            excel_sheet_airfoil_coord='airfoil_coord',
                            excel_sheet_structural_tower='structural_tower',
                            m_distribution='uniform',
                            h5_cross_sec_prop=None,
                            n_points_camber=100,
                            tol_remove_points=1e-3,
                            user_defined_m_distribution_type=None,
                            camber_effect_on_twist=False,
                            wsp=0.,
                            dt=0.):

    # Warning for back compatibility
    cout.cout_wrap('rotor_from_excel_type02 is obsolete! rotor_from_excel_type03 instead!', 3)

    # Assign values to dictionaries
    op_params = {}
    op_params['rotation_velocity'] = rotation_velocity
    op_params['pitch_deg'] = pitch_deg
    op_params['wsp'] = wsp
    op_params['dt'] = dt

    geom_params = {}
    geom_params['chord_panels'] = chord_panels
    geom_params['tol_remove_points'] = tol_remove_points
    geom_params['n_points_camber'] = n_points_camber
    geom_params['h5_cross_sec_prop'] = h5_cross_sec_prop
    geom_params['m_distribution'] = m_distribution

    options = {}
    options['camber_effect_on_twist'] = camber_effect_on_twist
    options['user_defined_m_distribution_type'] = user_defined_m_distribution_type
    options['include_polars'] = False
    options['twist_in_aero'] = False

    excel_description = {}
    excel_description['excel_file_name'] = excel_file_name
    excel_description['excel_sheet_parameters'] = excel_sheet_parameters
    excel_description['excel_sheet_structural_blade'] = excel_sheet_structural_blade
    excel_description['excel_sheet_discretization_blade'] = excel_sheet_discretization_blade
    excel_description['excel_sheet_aero_blade'] = excel_sheet_aero_blade
    excel_description['excel_sheet_airfoil_info'] = excel_sheet_airfoil_info
    excel_description['excel_sheet_airfoil_chord'] = excel_sheet_airfoil_coord

    rotor = rotor_from_excel_type03(op_params,
                                   geom_params,
                                   excel_description,
                                   options)

    return rotor


def generate_from_excel_type02(chord_panels,
                                rotation_velocity,
                                pitch_deg,
                                excel_file_name='database_excel_type02.xlsx',
                                excel_sheet_parameters='parameters',
                                excel_sheet_structural_blade='structural_blade',
                                excel_sheet_discretization_blade='discretization_blade',
                                excel_sheet_aero_blade='aero_blade',
                                excel_sheet_airfoil_info='airfoil_info',
                                excel_sheet_airfoil_coord='airfoil_coord',
                                excel_sheet_structural_tower='structural_tower',
                                m_distribution='uniform',
                                h5_cross_sec_prop=None,
                                n_points_camber=100,
                                tol_remove_points=1e-3,
                                user_defined_m_distribution_type=None,
                                camber_effect_on_twist=False,
                                wsp=0.,
                                dt=0.):

    # Warning for back compatibility
    cout.cout_wrap('generate_from_excel_type02 is obsolete! rotor_from_excel_type03 instead!', 3)

    # Assign values to dictionaries
    op_params = {}
    op_params['rotation_velocity'] = rotation_velocity
    op_params['pitch_deg'] = pitch_deg
    op_params['wsp'] = wsp
    op_params['dt'] = dt

    geom_params = {}
    geom_params['chord_panels'] = chord_panels
    geom_params['tol_remove_points'] = tol_remove_points
    geom_params['n_points_camber'] = n_points_camber
    geom_params['h5_cross_sec_prop'] = h5_cross_sec_prop
    geom_params['m_distribution'] = m_distribution

    options = {}
    options['camber_effect_on_twist'] = camber_effect_on_twist
    options['user_defined_m_distribution_type'] = user_defined_m_distribution_type
    options['include_polars'] = False

    excel_description = {}
    excel_description['excel_file_name'] = excel_file_name
    excel_description['excel_sheet_parameters'] = excel_sheet_parameters
    excel_description['excel_sheet_structural_tower'] = excel_sheet_structural_tower
    excel_description['excel_sheet_structural_blade'] = excel_sheet_structural_blade
    excel_description['excel_sheet_discretization_blade'] = excel_sheet_discretization_blade
    excel_description['excel_sheet_aero_blade'] = excel_sheet_aero_blade
    excel_description['excel_sheet_airfoil_info'] = excel_sheet_airfoil_info
    excel_description['excel_sheet_airfoil_chord'] = excel_sheet_airfoil_coord

    wt = generate_from_excel_type03(op_params,
                                   geom_params,
                                   excel_description,
                                   options)

    return wt
