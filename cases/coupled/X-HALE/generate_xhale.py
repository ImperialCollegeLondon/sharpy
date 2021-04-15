#!/usr/env python
# X-HALE model for SHARPy.
# Version 1.0
# See IFASD 2019 paper by: del Carre, A and Teixeira, P and Palacios, R and Cesnik, C. E. S.
#
# Alfonso del Carre
# June 2019
# ============================================================================

import h5py as h5
import numpy as np
import os
import sharpy.utils.algebra as algebra

route = os.path.dirname(os.path.realpath(__file__)) + '/'

cases = [
    '0',    # 15%, 15 chords, lateral
    ]

data = dict()
data['0'] = {
    'gust_length': 15,              # in chords
    'gust_intensity': 0.15,         # in % of uinf
    'gust_shape': 'lateral 1-cos',  # '1-cos' for vertical gust
}

horseshoe = 'off'

for case in cases:
    vertical_tail = True
    if vertical_tail:
        case_name = 'xhale_ifasd' + '_' + case
        print('Generating xhale with vertical Ctail')
    else:
        case_name = 'xhale_ifasd' + '_' + case
        print('Generating xhale with horizontal Ctail')


    flow = [
            'BeamLoader',
            'AerogridLoader',
            # 'StaticTrim',     # uncomment for longitudinal static trim
            'StaticCoupled',    # static coupled solver with GECB and UVLM
            'BeamLoads',        # beam loads and strains computation for static
            'BeamPlot',         # beam structure and data output for static
            'AerogridPlot',     # aero grid output for static
            'DynamicCoupled',   # dynamic coupled solver. See end of file:
                                # settings['DynamicCoupled']['postprocessors']
                                # for the corresponding 'BeamLoads', 'BeamPlot'
                                # and 'AerogridPlot' settings and calls.
            ]

    u_inf = 14                  # free stream vel [SI]
    rho = 1.225                 # density [SI]
    chord = 0.2                 # main wing chord length for gust dimensioning.
                                # the value used for the geometry generation is
                                # given in the input/*.xslx files

    if vertical_tail:
        alpha = 2.4744791522743887*np.pi/180        # angle of attack [rad]
        beta = 0.*np.pi/180                         # sideslip angle  [rad]
        cs_deflection = 1.186515244051093*np.pi/180 # elevators deflection [rad]
        aileron_deflection = 0*0.039011*np.pi/180   # aileron deflection [rad]
        thrustC = 0.21033486522175712               # baseline thrust [N]
        differential = 0                            # thrust of right side: T_R = thrustC*(1 + differential)
                                                    # thrust of left side: T_L = thrustC*(1 - differential)
        roll = 0                                    # initial/static roll angle [rad]
        in_structural_twist = 5*np.pi/180


    else:
        # NOTE: These values are not the correct trim. Run StaticTrim with your
        # current discretisation
        alpha = 2.4744791522743887*np.pi/180        # angle of attack [rad]
        beta = 0.*np.pi/180                         # sideslip angle  [rad]
        cs_deflection = 1.186515244051093*np.pi/180 # elevators deflection [rad]
        aileron_deflection = 0*0.039011*np.pi/180   # aileron deflection [rad]
        thrustC = 0.21033486522175712               # baseline thrust [N]
        differential = 0                            # thrust of right side: T_R = thrustC*(1 + differential)
                                                    # thrust of left side: T_L = thrustC*(1 - differential)
        roll = 0                                    # initial/static roll angle [rad]
        in_structural_twist = 5*np.pi/180

    gravity = 'on'
    gravity_value = 9.807

    # stiffness multiplier
    sigma = 1
    # shear stiffness multipliers
    ga_mult = 0.1

    # spatial offset [m] for the gust. (if == 1, gust 1 m in front of reference
    # point [0, 0, 0].
    space_offset = 1.

    try:
        gust_intensity = data[case]['gust_intensity']
        gust_length = data[case]['gust_length']*chord
        gust_shape = data[case]['gust_shape']
    except KeyError:
        gust_intensity = 0
        gust_length = 0
        gust_shape = '1-cos'

    # number of load substeps in the static coupled solver.
    n_step = 1

    # relaxation factor for the static coupled solver
    # static relaxation factor \in [0, 1). 0 == no relaxation
    static_relaxation_factor = 0.5
    # Dynamic relaxation parameters. Relaxation is linearly varied between
    # initial and final relaxation factor in relaxation_steps
    initial_relaxation_factor = 0.4
    final_relaxation_factor = 0.9
    relaxation_steps = 15

    # nonlinear beam tolerance.
    tolerance = 1e-6
    # FSI iteration tolerance
    fsi_tolerance = 1e-6
    # wake length when not running horseshoe
    wake_length = 4 # meters

    # geometrical data
    span_section = 1.0
    dihedral_outer = 10*np.pi/180

    length_centre_tail = 1.106
    length_outer_tail = 0.65
    span_tail = 0.24
    span_ctail_L = 0.145
    span_ctail_R = 0.24
    span_fin = 0.184
    span_vfin = 0.15

    n_sections = 3

    # DISCRETISATION
    # spatial discretisation
    # chordwise discretisation
    m = 8               # main wing
    m_tail = 3          # tails
    m_fin = 4           # fins and pods

    # number of structural elements in inner and outer sections.
    # note that you will have 2*n_elem spanwise aero panels
    n_elem_section = 4
    # structural elements in dihedral section
    n_elem_section_dihedral = 8
    # elements in central tail boom
    n_elem_centre_tail = 1
    # elements in outer tail booms
    n_elem_outer_tail = 1
    # elements in tails (per semi span)
    n_elem_tail = 1
    # elements in fins and pods
    n_elem_fin = 1
    n_elem_main = int((n_sections-1)*n_elem_section + n_elem_section_dihedral)
    # number of aero surfaces. To understand the logic of this, check SHARPy's
    # documentation
    n_surfaces = 20

    # temporal discretisation
    # seconds of simulation
    physical_time = 10
    # factor multiplying the theoretical timestep
    # (dt = chord/m/u_inf*tstep_factor)
    tstep_factor = 1
    dt = chord/m/u_inf*tstep_factor
    n_tstep = round(physical_time/dt)
    print('n_tstep: ', n_tstep)

    # if horseshoe wake (only for static) is 'on'
    # we only need one chordwise wake panel
    if horseshoe == 'on':
        mstar = 1
    else:
    # else, we put as many as we need to reach wake_length in steady conditions
        mstar = int(wake_length/(u_inf*dt))
    print('mstar = ', mstar)

    # beam processing
    # don't modify this
    n_node_elem = 3
    span_main = n_sections*span_section

    # total elements, nodes... calculation
    # total number of elements
    n_elem = 0
    n_elem += n_elem_main
    n_elem += n_elem_main
    n_elem += n_elem_centre_tail
    n_elem += n_elem_tail
    n_elem += n_elem_tail
    n_elem += n_elem_fin
    n_elem += n_elem_outer_tail
    n_elem += n_elem_tail
    n_elem += n_elem_tail
    n_elem += n_elem_fin
    n_elem += n_elem_outer_tail
    n_elem += n_elem_tail
    n_elem += n_elem_tail
    n_elem += n_elem_fin
    n_elem += n_elem_outer_tail
    n_elem += n_elem_tail
    n_elem += n_elem_tail
    n_elem += n_elem_outer_tail
    n_elem += n_elem_tail
    n_elem += n_elem_tail
    n_elem += n_elem_fin
    n_elem += n_elem_fin
    n_elem += n_elem_fin
    n_elem += n_elem_fin
    n_elem += n_elem_fin

# number of nodes per part
    n_node_section = n_elem_section*(n_node_elem - 1) + 1
    n_node_section_dihedral = n_elem_section_dihedral*(n_node_elem - 1) + 1
    n_node_main = n_elem_main*(n_node_elem - 1) + 1
    n_node_centre_tail = n_elem_centre_tail*(n_node_elem - 1) + 1
    n_node_tail = n_elem_tail*(n_node_elem - 1) + 1
    n_node_outer_tail = n_elem_outer_tail*(n_node_elem - 1) + 1
    n_node_fin = n_elem_fin*(n_node_elem - 1) + 1

# total number of nodes
    n_node = 0
    n_node += n_node_main + n_node_main - 1
    n_node += n_node_centre_tail - 1
    n_node += n_node_tail - 1
    n_node += n_node_tail - 1
    n_node += n_node_fin - 1
    n_node += n_node_outer_tail - 1
    n_node += n_node_tail - 1
    n_node += n_node_tail - 1
    n_node += n_node_fin - 1
    n_node += n_node_outer_tail - 1
    n_node += n_node_tail - 1
    n_node += n_node_tail - 1
    n_node += n_node_fin - 1
    n_node += n_node_outer_tail - 1
    n_node += n_node_tail - 1
    n_node += n_node_tail - 1
    n_node += n_node_outer_tail - 1
    n_node += n_node_tail - 1
    n_node += n_node_tail - 1
    n_node += n_node_fin - 1
    n_node += n_node_fin - 1
    n_node += n_node_fin - 1
    n_node += n_node_fin - 1
    n_node += n_node_fin - 1

    # stiffness and mass matrices
    # if you add custom stiffness/mass matrices, make sure to update this
    # number
    n_stiffness = 12
    n_mass = 17

    # PLACEHOLDERS
    # beam
    x = np.zeros((n_node, ))
    y = np.zeros((n_node, ))
    z = np.zeros((n_node, ))
    stiffness_db = np.zeros((n_stiffness, 6, 6))
    mass_db = np.zeros((n_mass, 6, 6))
    beam_number = np.zeros((n_elem, ), dtype=int)
    num_node_elements = np.zeros((n_elem, ), dtype=int) + 3
    frame_of_reference_delta = np.zeros((n_elem, n_node_elem, 3))
    structural_twist = np.zeros((n_elem, n_node_elem))
    conn = np.zeros((n_elem, n_node_elem), dtype=int)
    elem_stiffness = np.zeros((n_elem, ), dtype=int)
    elem_mass = np.zeros((n_elem, ), dtype=int)
    boundary_conditions = np.zeros((n_node, ), dtype=int)
    app_forces = np.zeros((n_node, 6))
    n_lumped_mass = 0
    lumped_mass_nodes = None
    lumped_mass = None
    lumped_mass_inertia = None
    lumped_mass_position = None

    end_nodesL = np.zeros((n_sections,), dtype=int)
    end_nodesR = np.zeros((n_sections,), dtype=int)
    end_elementsL = np.zeros((n_sections,), dtype=int)
    end_elementsR = np.zeros((n_sections,), dtype=int)
    end_tails_nodesL = np.zeros((2, ), dtype=int)
    end_tails_elementsL = np.zeros((2, ), dtype=int)
    end_tails_nodesR = np.zeros((2, ), dtype=int)
    end_tails_elementsR = np.zeros((2, ), dtype=int)
    end_of_centre_tail_node  = 0
    end_of_centre_tail_elem  = 0

    end_tip_tail_nodeC = np.zeros((2, ), dtype=int)
    end_tip_tail_elemC = np.zeros((2, ), dtype=int)

    tail_beam_numbersR = np.zeros((2, 3)) # 0=centre spar, 1=R tail, 2=L tail
    tail_beam_numbersL = np.zeros((2, 3)) # 0=centre spar, 1=R tail, 2=L tail
    tail_beam_numbersC = np.zeros((3, ))

    fin_beam_numberC = 0
    fin_beam_numberL = 0
    fin_beam_numberR = 0
    vfin_beam_numberC = 0
    vfin_beam_numberL = 0
    vfin_beam_numberR = 0
    fin_beam_numberLL = 0
    fin_beam_numberRR = 0

    # aero
    airfoil_distribution = np.zeros((n_elem, n_node_elem), dtype=int)
    surface_distribution = np.zeros((n_elem,), dtype=int) - 1
    surface_m = np.zeros((n_surfaces, ), dtype=int)
    # chordwise panel distribution. I'd leave there, but if you want,
    # you can try '1-cos'
    m_distribution = 'uniform'
    aero_node = np.zeros((n_node,), dtype=bool)
    twist = np.zeros((n_elem, n_node_elem))
    chord = np.zeros((n_elem, n_node_elem,))
    elastic_axis = np.zeros((n_elem, n_node_elem,))

    thrust_nodes = np.zeros((5,), dtype=int)

# FUNCTIONS-------------------------------------------------------------
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

        solver_file_name = route + '/' + case_name + '.sharpy'
        if os.path.isfile(solver_file_name):
            os.remove(solver_file_name)

        flightcon_file_name = route + '/' + case_name + '.flightcon.txt'
        if os.path.isfile(flightcon_file_name):
            os.remove(flightcon_file_name)

    def read_beam_data(filename='inputs/beam_properties.xlsx'):
        """
        Function reading the xlsx file with beam properties, including
        stiffness and distributed mass.

        Data is then processed to obtain the 6x6 mass matrices.
        """
        import pandas as pd
        import sharpy.utils.model_utils as model_utils
        # mass
        mass_sheet = pd.read_excel(filename, sheet_name='mass', header=1, skip_rows=1, index_col=0)
        # remove units
        mass_sheet = mass_sheet.drop(['[-]'])
        mass_data = dict()
        for index, row in mass_sheet.iterrows():
            mass_data[index] = dict()
            mass_data[index]['mass'] = mass_sheet['mass'][index]
            mass_data[index]['inertia'] = np.zeros((3, 3))
            mass_data[index]['inertia'][0, 0] = mass_sheet['ixx'][index]
            mass_data[index]['inertia'][1, 1] = mass_sheet['iyy'][index]
            mass_data[index]['inertia'][2, 2] = mass_sheet['izz'][index]
            mass_data[index]['inertia'][1, 2] = mass_sheet['iyz'][index]
            mass_data[index]['inertia'][2, 1] = mass_sheet['iyz'][index]
            mass_data[index]['xcg'] = np.zeros((3,))
            mass_data[index]['xcg'][0] = mass_sheet['xcg'][index]
            mass_data[index]['xcg'][1] = mass_sheet['ycg'][index]
            mass_data[index]['xcg'][2] = mass_sheet['zcg'][index]

            mass_data[index]['full_matrix'] = (
                model_utils.mass_matrix_generator(mass_data[index]['mass'],
                                                  mass_data[index]['xcg'],
                                                  mass_data[index]['inertia']))
        # stiffness
        stiff_sheet = pd.read_excel(filename, sheet_name='stiffness', header=1, skip_rows=1, index_col=0)
        # remove units
        stiff_sheet = stiff_sheet.drop(['[-]'])
        stiff_data = dict()
        for index, row in stiff_sheet.iterrows():
            stiff_data[index] = np.zeros((6, 6))
            stiff_data[index][0, 0] = stiff_sheet['ea'][index]
            stiff_data[index][1, 1] = stiff_sheet['gay'][index]*ga_mult
            stiff_data[index][2, 2] = stiff_sheet['gaz'][index]*ga_mult
            stiff_data[index][3, 3] = stiff_sheet['gj'][index]
            stiff_data[index][4, 4] = stiff_sheet['eiy'][index]
            stiff_data[index][5, 5] = stiff_sheet['eiz'][index]
            stiff_data[index][0, 4] = stiff_sheet['k13'][index]     # cross-
            stiff_data[index][4, 0] = stiff_sheet['k13'][index]     # stiffness
            stiff_data[index][0, 5] = stiff_sheet['k14'][index]
            stiff_data[index][5, 0] = stiff_sheet['k14'][index]     # 1-axial
            stiff_data[index][4, 5] = stiff_sheet['k34'][index]     # 3-OOP
            stiff_data[index][5, 4] = stiff_sheet['k34'][index]     # 4-IP

        return mass_data, stiff_data

    def read_lumped_mass_data(filename='inputs/lumped_mass.xlsx'):
        import pandas as pd
        xl = pd.ExcelFile(filename)
        sheets = {sheet_name: xl.parse(sheet_name, header=0, skiprows=(0, 2)) for sheet_name in xl.sheet_names}

        lumped_mass_data = dict()
        n_lumped = 0
        for sheet, val in sheets.items():
            if sheet == 'Notes':
                continue
            lumped_mass_data[sheet] = list()
            for row in range(len(val)):
                n_lumped += 1
                lumped_mass_data[sheet].append(dict())
                lumped_mass_data[sheet][row]['mass'] = 0.0
                lumped_mass_data[sheet][row]['inertia'] = np.zeros((3, 3))
                lumped_mass_data[sheet][row]['xcg'] = np.zeros((3,))

                lumped_mass_data[sheet][row]['mass'] = val['Mass'][row]
                lumped_mass_data[sheet][row]['inertia'][0, 0] = val['Ixx'][row]
                lumped_mass_data[sheet][row]['inertia'][1, 1] = val['Iyy'][row]
                lumped_mass_data[sheet][row]['inertia'][2, 2] = val['Izz'][row]
                lumped_mass_data[sheet][row]['inertia'][0, 1] = val['Ixy'][row]
                lumped_mass_data[sheet][row]['inertia'][1, 0] = val['Ixy'][row]
                lumped_mass_data[sheet][row]['inertia'][0, 2] = val['Ixz'][row]
                lumped_mass_data[sheet][row]['inertia'][2, 0] = val['Ixz'][row]
                lumped_mass_data[sheet][row]['inertia'][1, 2] = val['Iyz'][row]
                lumped_mass_data[sheet][row]['inertia'][2, 1] = val['Iyz'][row]

                # location of centre of gravity for lumped mass is given in
                # material frame of reference, which depends on the element and
                # node you are clamping it to. Check the documentation for
                # a full explanation, but as a rule of thumb: check the
                # frame_of_reference_delta vector parameter in the beam
                # definition. It usually points towards "y" in material FoR, and
                # "x" is always tangent to the beam, pointing towards increasing
                # node numbers.
                # This is valid as long as structural_twist is 0. If
                # it is not, you need to take into account that the material
                # FoR will be rotated a structrual_twist angle around
                # the material "x" axis.
                lumped_mass_data[sheet][row]['xcg'][0] = val['xcg'][row]
                lumped_mass_data[sheet][row]['xcg'][1] = val['ycg'][row]
                lumped_mass_data[sheet][row]['xcg'][2] = val['zcg'][row]

        lumped_mass_data['n_lumped_mass'] = n_lumped
        return lumped_mass_data

    def generate_fem():
        global end_of_centre_tail_node, end_of_centre_tail_elem
        global fin_beam_numberC, fin_beam_numberL, fin_beam_numberR
        global vfin_beam_numberC, vfin_beam_numberL, vfin_beam_numberR
        global fin_beam_numberLL, fin_beam_numberRR
        global residual_forces, residual_moments

        mass_data, stiff_data = read_beam_data()
        lumped_mass_data = read_lumped_mass_data()

        # process lumped mass
        n_lumped_mass = lumped_mass_data['n_lumped_mass']
        lumped_mass_nodes = np.zeros((n_lumped_mass, ), dtype=int)
        lumped_mass = np.zeros((n_lumped_mass, ))
        lumped_mass_inertia = np.zeros((n_lumped_mass, 3, 3))
        lumped_mass_position = np.zeros((n_lumped_mass, 3))
        lumped_mass_indices = dict()
        i_lm = 0
        for i, k in enumerate(lumped_mass_data.keys()):
            if k == 'n_lumped_mass':
                continue
            lumped_mass_indices[k] = list()
            for j in range(len(lumped_mass_data[k])):
                lumped_mass[i_lm] = lumped_mass_data[k][j]['mass']
                lumped_mass_inertia[i_lm] = lumped_mass_data[k][j]['inertia']
                lumped_mass_position[i_lm] = lumped_mass_data[k][j]['xcg']
                lumped_mass_indices[k].append(i_lm)
                i_lm += 1

        # process distributed mass
        mass_db[0, ...] = mass_data['Linboard']['full_matrix']
        mass_db[1, ...] = mass_data['Loutboard']['full_matrix']
        mass_db[2, ...] = mass_data['Ldihedral']['full_matrix']
        mass_db[3, ...] = mass_data['Rinboard']['full_matrix']
        mass_db[4, ...] = mass_data['Routboard']['full_matrix']
        mass_db[5, ...] = mass_data['Rdihedral']['full_matrix']
        mass_db[6, ...] = mass_data['boom']['full_matrix']
        mass_db[7, ...] = mass_data['tailL']['full_matrix']
        mass_db[8, ...] = mass_data['tailR']['full_matrix']
        mass_db[9, ...] = mass_data['Cfin']['full_matrix']
        mass_db[10, ...] = mass_data['Lfin']['full_matrix']
        mass_db[11, ...] = mass_data['Rfin']['full_matrix']
        mass_db[12, ...] = mass_data['LLfin']['full_matrix']
        mass_db[13, ...] = mass_data['RRfin']['full_matrix']
        mass_db[14, ...] = mass_data['Cvfin']['full_matrix']
        mass_db[15, ...] = mass_data['Lvfin']['full_matrix']
        mass_db[16, ...] = mass_data['Rvfin']['full_matrix']

        # process stiffness data
        stiffness_db[0, ...] = sigma*stiff_data['Linboard']
        stiffness_db[1, ...] = sigma*stiff_data['Loutboard']
        stiffness_db[2, ...] = sigma*stiff_data['Ldihedral']
        stiffness_db[3, ...] = sigma*stiff_data['Rinboard']
        stiffness_db[4, ...] = sigma*stiff_data['Routboard']
        stiffness_db[5, ...] = sigma*stiff_data['Rdihedral']
        stiffness_db[6, ...] = sigma*stiff_data['boom']
        stiffness_db[7, ...] = sigma*stiff_data['tailL']
        stiffness_db[8, ...] = sigma*stiff_data['tailR']
        stiffness_db[9, ...] = sigma*stiff_data['Cfin']
        stiffness_db[10, ...] = sigma*stiff_data['Lfin']
        stiffness_db[11, ...] = sigma*stiff_data['Rfin']

        # this matrix is useful when using symmetrical data for the mass, while
        # your local frames of reference are not.
        # For example, the right wing has a material "y" axis pointing fore,
        # while the left points aft due to "x" pointing in opposite directions.
        rotation_mat = algebra.rotation3d_z(np.pi)
        # I use this matrix to convert from the canonical material FoR to the one with
        # structural twist considered
        rotation_mat_x = algebra.rotation3d_x(-in_structural_twist)

        we = 0
        wn = 0

        # SECTION 0R
        # add the lumped mass of the pods
        lumped_mass_id = 'centre_pod'
        for i in range(len(lumped_mass_indices[lumped_mass_id])):
            lumped_mass_nodes[lumped_mass_indices[lumped_mass_id][i]] = 0
            lumped_mass_position[lumped_mass_indices[lumped_mass_id][i]] = np.dot(rotation_mat_x,
                                                                                  np.dot(np.eye(3),
                                                                                         lumped_mass_position[lumped_mass_indices[lumped_mass_id][i]])
                                                                            )

        # add thrust as applied force
        # it is necessary to rotate it in order to have a horizontal force,
        # as "y_materal" points 5 deg above.
        app_forces[0, 0:3] = thrustC*np.array([0.0, np.cos(in_structural_twist), -np.sin(in_structural_twist)])
        thrust_nodes[0] = 0

        beam_number[we:we + n_elem_section] = 0
        y[wn:wn + n_node_section] = np.linspace(0.0, span_section, n_node_section)
        structural_twist[we:we + n_elem_section, :] = in_structural_twist
        for ielem in range(n_elem_section):
            # connectivities. Order is [0, 2, 1]
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
        elem_stiffness[we:we + n_elem_section] = 3
        elem_mass[we:we + n_elem_section] = 3
        boundary_conditions[0] = 1
        # we: working_element
        # wn: working_node
        we += n_elem_section
        wn += n_node_section
        # keeping track of the node and element of very end of section to then
        # map the aero more easily.
        end_nodesR[0] = wn - 1
        end_elementsR[0] = we - 1

        # SECTION 1R
        # add the lumped mass of the pods
        lumped_mass_id = 'R_inboard_pod'
        for i in range(len(lumped_mass_indices[lumped_mass_id])):
            lumped_mass_nodes[lumped_mass_indices[lumped_mass_id][i]] = wn - 1
            lumped_mass_position[lumped_mass_indices[lumped_mass_id][i]] = np.dot(rotation_mat_x,
                                                                                  np.dot(np.eye(3),
                                                                                         lumped_mass_position[lumped_mass_indices[lumped_mass_id][i]])
                                                                            )

        app_forces[wn-1, 0:3] = thrustC*np.array([0.0, np.cos(in_structural_twist), -np.sin(in_structural_twist)])
        thrust_nodes[1] = wn - 1

        beam_number[we:we + n_elem_section] = 1
        y[wn:wn + n_node_section - 1] = y[wn - 1] + np.linspace(0.0, span_section, n_node_section)[1:]
        structural_twist[we:we + n_elem_section, :] = in_structural_twist
        for ielem in range(n_elem_section):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
        elem_stiffness[we:we + n_elem_section] = 4
        elem_mass[we:we + n_elem_section] = 4
        we += n_elem_section
        wn += n_node_section - 1
        end_nodesR[1] = wn - 1
        end_elementsR[1] = we - 1

        # SECTION 2R
        # add the lumped mass of the pods
        lumped_mass_id = 'R_outboard_pod'
        for i in range(len(lumped_mass_indices[lumped_mass_id])):
            lumped_mass_nodes[lumped_mass_indices[lumped_mass_id][i]] = wn - 1
            lumped_mass_position[lumped_mass_indices[lumped_mass_id][i]] = np.dot(rotation_mat_x,
                                                                                  np.dot(np.eye(3),
                                                                                         lumped_mass_position[lumped_mass_indices[lumped_mass_id][i]])
                                                                            )

        app_forces[wn-1, 0:3] = thrustC*(1 + differential)*np.array([0.0, np.cos(in_structural_twist), -np.sin(in_structural_twist)])
        thrust_nodes[1] = wn - 1

        beam_number[we:we + n_elem_section_dihedral] = 2
        structural_twist[we:we + n_elem_section_dihedral, :] = in_structural_twist
        y[wn:wn + n_node_section_dihedral - 1] = y[wn - 1] + np.linspace(0.0,
                                                                np.cos(dihedral_outer)*span_section,
                                                                n_node_section_dihedral)[1:]
        z[wn:wn + n_node_section_dihedral - 1] = z[wn - 1] + np.linspace(0.0,
                                                                np.sin(dihedral_outer)*span_section,
                                                                n_node_section_dihedral)[1:]
        for ielem in range(n_elem_section_dihedral):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
        elem_stiffness[we:we + n_elem_section_dihedral] = 5
        elem_mass[we:we + n_elem_section_dihedral] = 5
        boundary_conditions[wn + n_node_section_dihedral - 1 - 1] = -1
        we += n_elem_section_dihedral
        wn += n_node_section_dihedral - 1
        end_nodesR[2] = wn - 1
        end_elementsR[2] = we - 1

        # SECTION 0L
        beam_number[we:we + n_elem_section] = 3
        structural_twist[we:we + n_elem_section, :] = -in_structural_twist
        y[wn:wn + n_node_section - 1] = np.linspace(0.0, -span_section, n_node_section)[1:]
        for ielem in range(n_elem_section):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
        conn[we, 0] = 0
        elem_stiffness[we:we + n_elem_section] = 0
        elem_mass[we:we + n_elem_section] = 0
        we += n_elem_section
        wn += n_node_section - 1
        end_nodesL[0] = wn - 1
        end_elementsL[0] = we - 1

        # SECTION 1L
        # add the lumped mass of the pods
        lumped_mass_id = 'L_inboard_pod'
        for i in range(len(lumped_mass_indices[lumped_mass_id])):
            lumped_mass_nodes[lumped_mass_indices[lumped_mass_id][i]] = wn - 1
            lumped_mass_position[lumped_mass_indices[lumped_mass_id][i]] = np.dot(rotation_mat_x.T,
                                                                                  np.dot(rotation_mat,
                                                                                         lumped_mass_position[lumped_mass_indices[lumped_mass_id][i]])
                                                                            )
            lumped_mass_position[lumped_mass_indices[lumped_mass_id][i]][0] *= -1
        app_forces[wn-1, 0:3] = thrustC*np.array([0.0, -np.cos(in_structural_twist), -np.sin(in_structural_twist)])
        thrust_nodes[3] = wn - 1


        beam_number[we:we + n_elem_section] = 4
        structural_twist[we:we + n_elem_section, :] = -in_structural_twist
        y[wn:wn + n_node_section - 1] = y[wn - 1] + np.linspace(0.0, -span_section, n_node_section)[1:]
        for ielem in range(n_elem_section):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
        elem_stiffness[we:we + n_elem_section] = 1
        elem_mass[we:we + n_elem_section] = 1
        we += n_elem_section
        wn += n_node_section - 1
        end_nodesL[1] = wn - 1
        end_elementsL[1] = we - 1

        # SECTION 2L
        # add the lumped mass of the pods
        lumped_mass_id = 'L_outboard_pod'
        for i in range(len(lumped_mass_indices[lumped_mass_id])):
            lumped_mass_nodes[lumped_mass_indices[lumped_mass_id][i]] = wn - 1
            lumped_mass_position[lumped_mass_indices[lumped_mass_id][i]] = np.dot(rotation_mat_x.T,
                                                                                  np.dot(rotation_mat,
                                                                                         lumped_mass_position[lumped_mass_indices[lumped_mass_id][i]]))
            lumped_mass_position[lumped_mass_indices[lumped_mass_id][i]][0] *= -1
        app_forces[wn-1, 0:3] = thrustC*(1 - differential)*np.array([0.0, -np.cos(in_structural_twist), -np.sin(in_structural_twist)])
        thrust_nodes[4] = wn - 1

        beam_number[we:we + n_elem_section_dihedral] = 5
        structural_twist[we:we + n_elem_section_dihedral, :] = -in_structural_twist
        y[wn:wn + n_node_section_dihedral - 1] = y[wn - 1] + np.linspace(0.0, -np.cos(dihedral_outer)*span_section, n_node_section_dihedral)[1:]
        z[wn:wn + n_node_section_dihedral - 1] = z[wn - 1] + np.linspace(0.0, np.sin(dihedral_outer)*span_section, n_node_section_dihedral)[1:]
        for ielem in range(n_elem_section_dihedral):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
        elem_stiffness[we:we + n_elem_section_dihedral] = 2
        elem_mass[we:we + n_elem_section_dihedral] = 2
        boundary_conditions[wn + n_node_section_dihedral - 1 - 1] = -1
        we += n_elem_section_dihedral
        wn += n_node_section_dihedral - 1
        end_nodesL[2] = wn - 1
        end_elementsL[2] = we - 1


        # centre tail
        beam_number[we:we + n_elem_centre_tail] = 6
        tail_beam_numbersC[0] = 6
        x[wn:wn + n_node_centre_tail - 1] = np.linspace(0.0, length_centre_tail, n_node_centre_tail)[1:]
        for ielem in range(n_elem_centre_tail):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [0.0, 1.0, 0.0]
        conn[we, 0] = 0
        elem_stiffness[we:we + n_elem_centre_tail] = 6
        elem_mass[we:we + n_elem_centre_tail] = 6
        we += n_elem_centre_tail
        wn += n_node_centre_tail - 1
        end_of_centre_tail_node = wn - 1
        end_of_centre_tail_elem = we

        if vertical_tail:
            beam_number[we:we + n_elem_tail] = 7
            tail_beam_numbersC[1] = 7
            x[wn:wn + n_node_tail - 1] = x[wn - 1]
            y[wn:wn + n_node_tail - 1] = y[wn - 1]
            z[wn:wn + n_node_tail - 1] = z[wn - 1] + np.linspace(0.0, span_ctail_R, n_node_tail)[1:]
            for ielem in range(n_elem_tail):
                conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
                for inode in range(n_node_elem):
                    frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
            elem_stiffness[we:we + n_elem_tail] = 8
            elem_mass[we:we + n_elem_tail] = 8
            boundary_conditions[wn + n_node_tail - 1 - 1] = -1
            end_tip_tail_nodeC[0] = wn + n_node_tail - 1 - 1
            end_tip_tail_elemC[0] = we + n_elem_tail - 1
            we += n_elem_tail
            wn += n_node_tail - 1

            beam_number[we:we + n_elem_tail] = 8
            tail_beam_numbersC[2] = 8
            x[wn:wn + n_node_tail - 1] = x[end_of_centre_tail_node]
            y[wn:wn + n_node_tail - 1] = y[wn - 1]
            z[wn:wn + n_node_tail - 1] = z[end_of_centre_tail_node] + np.linspace(0.0, -span_ctail_L, n_node_tail)[1:]
            for ielem in range(n_elem_tail):
                conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
                for inode in range(n_node_elem):
                    frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
            conn[we, 0] = end_of_centre_tail_node
            elem_stiffness[we:we + n_elem_tail] = 7
            elem_mass[we:we + n_elem_tail] = 7
            boundary_conditions[wn + n_node_tail - 1 -1] = -1
            end_tip_tail_nodeC[1] = wn + n_node_tail - 1 - 1
            end_tip_tail_elemC[1] = we + n_elem_tail - 1
            we += n_elem_tail
            wn += n_node_tail - 1
        else:
            beam_number[we:we + n_elem_tail] = 7
            tail_beam_numbersC[1] = 7
            x[wn:wn + n_node_tail - 1] = x[wn - 1]
            y[wn:wn + n_node_tail - 1] = y[wn - 1] + np.linspace(0.0, span_ctail_R, n_node_tail)[1:]
            for ielem in range(n_elem_tail):
                conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
                for inode in range(n_node_elem):
                    frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
            elem_stiffness[we:we + n_elem_tail] = 8
            elem_mass[we:we + n_elem_tail] = 8
            boundary_conditions[wn + n_node_tail - 1 - 1] = -1
            end_tip_tail_nodeC[0] = wn + n_node_tail - 1 - 1
            end_tip_tail_elemC[0] = we + n_elem_tail - 1
            we += n_elem_tail
            wn += n_node_tail - 1

            beam_number[we:we + n_elem_tail] = 8
            tail_beam_numbersC[2] = 8
            x[wn:wn + n_node_tail - 1] = x[end_of_centre_tail_node]
            y[wn:wn + n_node_tail - 1] = y[end_of_centre_tail_node] + np.linspace(0.0, -span_ctail_L, n_node_tail)[1:]
            for ielem in range(n_elem_tail):
                conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
                for inode in range(n_node_elem):
                    frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
            conn[we, 0] = end_of_centre_tail_node
            elem_stiffness[we:we + n_elem_tail] = 7
            elem_mass[we:we + n_elem_tail] = 7
            boundary_conditions[wn + n_node_tail - 1 -1] = -1
            end_tip_tail_nodeC[1] = wn + n_node_tail - 1 - 1
            end_tip_tail_elemC[1] = we + n_elem_tail - 1
            we += n_elem_tail
            wn += n_node_tail - 1

        # outer tail 0R
        beam_number[we:we + n_elem_outer_tail] = 9
        tail_beam_numbersR[0,0] = 9
        x[wn:wn + n_node_outer_tail - 1] = np.linspace(0.0, length_outer_tail, n_node_outer_tail)[1:]
        y[wn:wn + n_node_outer_tail - 1] = y[end_nodesR[0]]
        for ielem in range(n_elem_outer_tail):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [0.0, 1.0, 0.0]
        conn[we, 0] = end_nodesR[0]
        elem_stiffness[we:we + n_elem_outer_tail] = 6
        elem_mass[we:we + n_elem_outer_tail] = 6
        we += n_elem_outer_tail
        wn += n_node_outer_tail - 1
        end_tails_nodesR[0] = wn - 1
        end_tails_elementsR[0] = we - 1

        beam_number[we:we + n_elem_tail] = 10
        tail_beam_numbersR[0,1] = 10
        x[wn:wn + n_node_tail - 1] = x[end_tails_nodesR[0]]
        y[wn:wn + n_node_tail - 1] = y[end_tails_nodesR[0]] + np.linspace(0.0, span_tail, n_node_tail)[1:]
        for ielem in range(n_elem_tail):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
        elem_stiffness[we:we + n_elem_tail] = 8
        elem_mass[we:we + n_elem_tail] = 8
        boundary_conditions[wn + n_node_tail - 1 -1] = -1
        we += n_elem_tail
        wn += n_node_tail - 1

        beam_number[we:we + n_elem_tail] = 11
        tail_beam_numbersR[0,2] = 11
        x[wn:wn + n_node_tail - 1] = x[end_tails_nodesR[0]]
        y[wn:wn + n_node_tail - 1] = y[end_tails_nodesR[0]] + np.linspace(0.0, -span_tail, n_node_tail)[1:]
        for ielem in range(n_elem_tail):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
        conn[we, 0] = end_tails_nodesR[0]
        elem_stiffness[we:we + n_elem_tail] = 7
        elem_mass[we:we + n_elem_tail] = 7
        boundary_conditions[wn + n_node_tail - 1 -1] = -1
        we += n_elem_tail
        wn += n_node_tail - 1

        # outer tail 1R
        beam_number[we:we + n_elem_outer_tail] = 12
        tail_beam_numbersR[1,0] = 12
        x[wn:wn + n_node_outer_tail - 1] = np.linspace(0.0, length_outer_tail, n_node_outer_tail)[1:]
        y[wn:wn + n_node_outer_tail - 1] = y[end_nodesR[1]]
        for ielem in range(n_elem_outer_tail):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [0.0, 1.0, 0.0]
        conn[we, 0] = end_nodesR[1]
        elem_stiffness[we:we + n_elem_outer_tail] = 6
        elem_mass[we:we + n_elem_outer_tail] = 6
        we += n_elem_outer_tail
        wn += n_node_outer_tail - 1
        end_tails_nodesR[1] = wn - 1
        end_tails_elementsR[1] = we - 1

        beam_number[we:we + n_elem_tail] = 13
        tail_beam_numbersR[1,1] = 13
        x[wn:wn + n_node_tail - 1] = x[end_tails_nodesR[1]]
        y[wn:wn + n_node_tail - 1] = y[end_tails_nodesR[1]] + np.linspace(0.0, span_tail, n_node_tail)[1:]
        for ielem in range(n_elem_tail):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
        elem_stiffness[we:we + n_elem_tail] = 8
        elem_mass[we:we + n_elem_tail] = 8
        boundary_conditions[wn + n_node_tail - 1 -1] = -1
        we += n_elem_tail
        wn += n_node_tail - 1

        beam_number[we:we + n_elem_tail] = 14
        tail_beam_numbersR[1,2] = 14
        x[wn:wn + n_node_tail - 1] = x[end_tails_nodesR[1]]
        y[wn:wn + n_node_tail - 1] = y[end_tails_nodesR[1]] + np.linspace(0.0, -span_tail, n_node_tail)[1:]
        for ielem in range(n_elem_tail):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
        conn[we, 0] = end_tails_nodesR[1]
        elem_stiffness[we:we + n_elem_tail] = 7
        elem_mass[we:we + n_elem_tail] = 7
        boundary_conditions[wn + n_node_tail - 1 -1] = -1
        we += n_elem_tail
        wn += n_node_tail - 1

        # outer tail 0L
        beam_number[we:we + n_elem_outer_tail] = 15
        tail_beam_numbersL[0,0] = 15
        x[wn:wn + n_node_outer_tail - 1] = np.linspace(0.0, length_outer_tail, n_node_outer_tail)[1:]
        y[wn:wn + n_node_outer_tail - 1] = y[end_nodesL[0]]
        for ielem in range(n_elem_outer_tail):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [0.0, 1.0, 0.0]
        conn[we, 0] = end_nodesL[0]
        elem_stiffness[we:we + n_elem_outer_tail] = 6
        elem_mass[we:we + n_elem_outer_tail] = 6
        we += n_elem_outer_tail
        wn += n_node_outer_tail - 1
        end_tails_nodesL[0] = wn - 1
        end_tails_elementsL[0] = we - 1

        beam_number[we:we + n_elem_tail] = 16
        tail_beam_numbersL[0,1] = 16
        x[wn:wn + n_node_tail - 1] = x[end_tails_nodesL[0]]
        y[wn:wn + n_node_tail - 1] = y[end_tails_nodesL[0]] + np.linspace(0.0, span_tail, n_node_tail)[1:]
        for ielem in range(n_elem_tail):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
        conn[we, 0] = end_tails_nodesL[0]
        elem_stiffness[we:we + n_elem_tail] = 8
        elem_mass[we:we + n_elem_tail] = 8
        boundary_conditions[wn + n_node_tail - 1 - 1] = -1
        we += n_elem_tail
        wn += n_node_tail - 1

        beam_number[we:we + n_elem_tail] = 17
        tail_beam_numbersL[0,2] = 17
        x[wn:wn + n_node_tail - 1] = x[end_tails_nodesL[0]]
        y[wn:wn + n_node_tail - 1] = y[end_tails_nodesL[0]] + np.linspace(0.0, -span_tail, n_node_tail)[1:]
        for ielem in range(n_elem_tail):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
        conn[we, 0] = end_tails_nodesL[0]
        elem_stiffness[we:we + n_elem_tail] = 7
        elem_mass[we:we + n_elem_tail] = 7
        boundary_conditions[wn + n_node_tail - 1 - 1] = -1
        we += n_elem_tail
        wn += n_node_tail - 1

        # outer tail 1L
        beam_number[we:we + n_elem_outer_tail] = 18
        tail_beam_numbersL[1,0] = 18
        x[wn:wn + n_node_outer_tail - 1] = np.linspace(0.0, length_outer_tail, n_node_outer_tail)[1:]
        y[wn:wn + n_node_outer_tail - 1] = y[end_nodesL[1]]
        for ielem in range(n_elem_outer_tail):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [0.0, 1.0, 0.0]
        conn[we, 0] = end_nodesL[1]
        elem_stiffness[we:we + n_elem_outer_tail] = 6
        elem_mass[we:we + n_elem_outer_tail] = 6
        we += n_elem_outer_tail
        wn += n_node_outer_tail - 1
        end_tails_nodesL[1] = wn - 1
        end_tails_elementsL[1] = we - 1

        beam_number[we:we + n_elem_tail] = 19
        tail_beam_numbersL[1,1] = 19
        x[wn:wn + n_node_tail - 1] = x[end_tails_nodesL[1]]
        y[wn:wn + n_node_tail - 1] = y[end_tails_nodesL[1]] + np.linspace(0.0, span_tail, n_node_tail)[1:]
        for ielem in range(n_elem_tail):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
        elem_stiffness[we:we + n_elem_tail] = 8
        elem_mass[we:we + n_elem_tail] = 8
        boundary_conditions[wn + n_node_tail - 1 -1] = -1
        we += n_elem_tail
        wn += n_node_tail - 1

        beam_number[we:we + n_elem_tail] = 20
        tail_beam_numbersL[1,2] = 20
        x[wn:wn + n_node_tail - 1] = x[end_tails_nodesL[1]]
        y[wn:wn + n_node_tail - 1] = y[end_tails_nodesL[1]] + np.linspace(0.0, -span_tail, n_node_tail)[1:]
        for ielem in range(n_elem_tail):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
        conn[we, 0] = end_tails_nodesL[1]
        elem_stiffness[we:we + n_elem_tail] = 7
        elem_mass[we:we + n_elem_tail] = 7
        boundary_conditions[wn + n_node_tail - 1 -1] = -1
        we += n_elem_tail
        wn += n_node_tail - 1

        # vertical fins (pods)
        # centre one
        beam_number[we:we + n_elem_fin] = 21
        fin_beam_numberC = 21
        x[wn:wn + n_node_fin - 1] = x[0]
        y[wn:wn + n_node_fin - 1] = y[0]
        z[wn:wn + n_node_fin - 1] = z[0] + np.linspace(0.0, -span_fin, n_node_fin)[1:]
        for ielem in range(n_elem_fin):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
        conn[we, 0] = 0
        elem_stiffness[we:we + n_elem_fin] = 9
        elem_mass[we:we + n_elem_fin] = 9
        boundary_conditions[wn + n_node_fin - 1 - 1] = -1
        we += n_elem_fin
        wn += n_node_fin - 1

        # left one
        beam_number[we:we + n_elem_fin] = 22
        fin_beam_numberL = 22
        x[wn:wn + n_node_fin - 1] = x[end_nodesL[0]]
        y[wn:wn + n_node_fin - 1] = y[end_nodesL[0]]
        z[wn:wn + n_node_fin - 1] = z[end_nodesL[0]] + np.linspace(0.0, -span_fin, n_node_fin)[1:]
        for ielem in range(n_elem_fin):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
        conn[we, 0] = end_nodesL[0]
        elem_stiffness[we:we + n_elem_fin] = 10
        elem_mass[we:we + n_elem_fin] = 10
        boundary_conditions[wn + n_node_fin - 1 - 1] = -1
        we += n_elem_fin
        wn += n_node_fin - 1

        # right one
        beam_number[we:we + n_elem_fin] = 23
        fin_beam_numberR = 23
        x[wn:wn + n_node_fin - 1] = x[end_nodesR[0]]
        y[wn:wn + n_node_fin - 1] = y[end_nodesR[0]]
        z[wn:wn + n_node_fin - 1] = z[end_nodesR[0]] + np.linspace(0.0, -span_fin, n_node_fin)[1:]
        for ielem in range(n_elem_fin):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
        conn[we, 0] = end_nodesR[0]
        elem_stiffness[we:we + n_elem_fin] = 11
        elem_mass[we:we + n_elem_fin] = 11
        boundary_conditions[wn + n_node_fin - 1 - 1] = -1
        we += n_elem_fin
        wn += n_node_fin - 1

        # left outer one
        beam_number[we:we + n_elem_fin] = 24
        fin_beam_numberLL = 24
        x[wn:wn + n_node_fin - 1] = x[end_nodesL[1]]
        y[wn:wn + n_node_fin - 1] = y[end_nodesL[1]]
        z[wn:wn + n_node_fin - 1] = z[end_nodesL[1]] + np.linspace(0.0, -span_fin, n_node_fin)[1:]
        for ielem in range(n_elem_fin):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
        conn[we, 0] = end_nodesL[1]
        elem_stiffness[we:we + n_elem_fin] = 10
        elem_mass[we:we + n_elem_fin] = 12
        boundary_conditions[wn + n_node_fin - 1 - 1] = -1
        we += n_elem_fin
        wn += n_node_fin - 1

        # right outer one
        beam_number[we:we + n_elem_fin] = 25
        fin_beam_numberRR = 25
        x[wn:wn + n_node_fin - 1] = x[end_nodesR[1]]
        y[wn:wn + n_node_fin - 1] = y[end_nodesR[1]]
        z[wn:wn + n_node_fin - 1] = z[end_nodesR[1]] + np.linspace(0.0, -span_fin, n_node_fin)[1:]
        for ielem in range(n_elem_fin):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
        conn[we, 0] = end_nodesR[1]
        elem_stiffness[we:we + n_elem_fin] = 11
        elem_mass[we:we + n_elem_fin] = 13
        boundary_conditions[wn + n_node_fin - 1 - 1] = -1
        we += n_elem_fin
        wn += n_node_fin - 1

        # vertical fins
        # centre one
        beam_number[we:we + n_elem_fin] = 26
        vfin_beam_numberC = 26
        x[wn:wn + n_node_fin - 1] = x[end_of_centre_tail_node]
        y[wn:wn + n_node_fin - 1] = y[end_of_centre_tail_node]
        z[wn:wn + n_node_fin - 1] = z[end_of_centre_tail_node] + np.linspace(0.0, -span_vfin, n_node_fin)[1:]
        for ielem in range(n_elem_fin):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
        conn[we, 0] = end_of_centre_tail_node
        elem_stiffness[we:we + n_elem_fin] = 9
        elem_mass[we:we + n_elem_fin] = 14
        boundary_conditions[wn + n_node_fin - 1 - 1] = -1
        we += n_elem_fin
        wn += n_node_fin - 1

        # left one
        beam_number[we:we + n_elem_fin] = 27
        vfin_beam_numberL = 27
        x[wn:wn + n_node_fin - 1] = x[end_tails_nodesL[0]]
        y[wn:wn + n_node_fin - 1] = y[end_tails_nodesL[0]]
        z[wn:wn + n_node_fin - 1] = z[end_tails_nodesL[0]] + np.linspace(0.0, -span_vfin, n_node_fin)[1:]
        for ielem in range(n_elem_fin):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
        conn[we, 0] = end_tails_nodesL[0]
        elem_stiffness[we:we + n_elem_fin] = 10
        elem_mass[we:we + n_elem_fin] = 15
        boundary_conditions[wn + n_node_fin - 1 - 1] = -1
        we += n_elem_fin
        wn += n_node_fin - 1

        # right one
        beam_number[we:we + n_elem_fin] = 28
        vfin_beam_numberR = 28
        x[wn:wn + n_node_fin - 1] = x[end_tails_nodesR[0]]
        y[wn:wn + n_node_fin - 1] = y[end_tails_nodesR[0]]
        z[wn:wn + n_node_fin - 1] = z[end_tails_nodesR[0]] + np.linspace(0.0, -span_vfin, n_node_fin)[1:]
        for ielem in range(n_elem_fin):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
        conn[we, 0] = end_tails_nodesR[0]
        elem_stiffness[we:we + n_elem_fin] = 11
        elem_mass[we:we + n_elem_fin] = 16
        boundary_conditions[wn + n_node_fin - 1 - 1] = -1
        we += n_elem_fin
        wn += n_node_fin - 1


        # this output to the HDF5 file that is read by 'BeamLoader'.
        # if you want to see what's inside, check HDFView
        with h5.File(route + '/' + case_name + '.fem.h5', 'a') as h5file:
            coordinates = h5file.create_dataset('coordinates', data=np.column_stack((x, y, z)))
            conectivities = h5file.create_dataset('connectivities', data=conn)
            num_nodes_elem_handle = h5file.create_dataset(
                'num_node_elem', data=n_node_elem)
            num_nodes_handle = h5file.create_dataset(
                'num_node', data=n_node)
            num_elem_handle = h5file.create_dataset(
                'num_elem', data=n_elem)
            stiffness_db_handle = h5file.create_dataset(
                'stiffness_db', data=stiffness_db)
            stiffness_handle = h5file.create_dataset(
                'elem_stiffness', data=elem_stiffness)
            mass_db_handle = h5file.create_dataset(
                'mass_db', data=mass_db)
            mass_handle = h5file.create_dataset(
                'elem_mass', data=elem_mass)
            frame_of_reference_delta_handle = h5file.create_dataset(
                'frame_of_reference_delta', data=frame_of_reference_delta)
            structural_twist_handle = h5file.create_dataset(
                'structural_twist', data=structural_twist)
            bocos_handle = h5file.create_dataset(
                'boundary_conditions', data=boundary_conditions)
            beam_handle = h5file.create_dataset(
                'beam_number', data=beam_number)
            app_forces_handle = h5file.create_dataset(
                'app_forces', data=app_forces)
            lumped_mass_nodes_handle = h5file.create_dataset(
                'lumped_mass_nodes', data=lumped_mass_nodes)
            lumped_mass_handle = h5file.create_dataset(
                'lumped_mass', data=lumped_mass)
            lumped_mass_inertia_handle = h5file.create_dataset(
                'lumped_mass_inertia', data=lumped_mass_inertia)
            lumped_mass_position_handle = h5file.create_dataset(
                'lumped_mass_position', data=lumped_mass_position)

    def read_aero_data(filename='inputs/aero_properties.xlsx'):
        import pandas as pd

        xl = pd.ExcelFile(filename)
        sheets = {sheet_name: xl.parse(sheet_name, header=0, index_col=0) for sheet_name in xl.sheet_names}

        aero_data = dict()
        for sheet, val in sheets.items():
            aero_data[sheet] = dict()
            for item in val['value'].items():
                aero_data[sheet][item[0]] = item[1]

        return aero_data


    def generate_aero_file():
        global x, y, z
        global fin_beam_numberC, fin_beam_numberL, fin_beam_numberR
        global vfin_beam_numberC, vfin_beam_numberL, vfin_beam_numberR
        global fin_beam_numberLL, fin_beam_numberRR

        aero_data = read_aero_data()
        # control surfaces
        n_control_surfaces = 3
        control_surface = np.zeros((n_elem, n_node_elem), dtype=int) - 1
        control_surface_type = np.zeros((n_control_surfaces, ), dtype=int)
        control_surface_deflection = np.zeros((n_control_surfaces, ))
        control_surface_chord = np.zeros((n_control_surfaces, ), dtype=int)
        control_surface_hinge_coord = np.zeros((n_control_surfaces, ), dtype=float)

        # control surface type is 0 if static, 1 if dynamic
        control_surface_type[0] = 0
        control_surface_deflection[0] = cs_deflection
        # chord is given as number of panels, not length, so it has to be integer
        control_surface_chord[0] = m_tail
        # location of the hinge wrt e.axis when surface_chord == m_surface
        control_surface_hinge_coord[0] = 0

        control_surface_type[1] = 0
        control_surface_deflection[1] = aileron_deflection
        control_surface_chord[1] = int(m*0.25)
        control_surface_hinge_coord[1] = 0

        control_surface_type[2] = 0
        control_surface_deflection[2] = -aileron_deflection
        control_surface_chord[2] = int(m*0.25)
        control_surface_hinge_coord[2] = 0

        # right wing (surface 0, beams 0, 1, 2, 3)
        type = 'inboard'
        main_chord = aero_data[type]['chord']
        initial_node = 0
        final_node = end_nodesR[-1]
        initial_elem = 0
        final_elem = end_elementsR[-1]
        i_surf = 0
        airfoil_distribution[initial_elem: final_elem + 1, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
        surface_distribution[initial_elem: final_elem + 1] = i_surf
        surface_m[i_surf] = m
        aero_node[initial_node:final_node + 1] = True
        node_counter = 0
        for i_elem in range(initial_elem, final_elem + 1):
            for i_local_node in range(n_node_elem):
                if not i_local_node == 0:
                    node_counter += 1
                chord[i_elem, i_local_node] = aero_data[type]['chord']
                elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
                twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180
        for i_elem in range(end_elementsR[1] + 1, end_elementsR[2] + 1):
            if i_elem == 0 or i_elem == end_elementsR[2]:
                continue
            control_surface[i_elem, :] = 1

        # left wing (surface 1, beams 4, 5, 6, 7)
        initial_node = end_nodesR[-1] + 1
        final_node = end_nodesL[-1]
        initial_elem = end_elementsR[-1] + 1
        final_elem = end_elementsL[-1]
        i_surf += 1
        airfoil_distribution[initial_elem: final_elem + 1, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
        surface_distribution[initial_elem: final_elem + 1] = i_surf
        surface_m[i_surf] = m
        aero_node[initial_node:final_node + 1] = True
        node_counter = 0
        for i_elem in range(initial_elem, final_elem + 1):
            for i_local_node in range(n_node_elem):
                if not i_local_node == 0:
                    node_counter += 1
                chord[i_elem, i_local_node] = aero_data[type]['chord']
                elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
                twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180
        for i_elem in range(end_elementsL[1] + 1, end_elementsL[2] + 1):
            if i_elem == 0 or i_elem == end_elementsL[2]:
                continue
            control_surface[i_elem, :] = 2

        # centre tail
        type = 'Ctail'
        ctail_chord = aero_data[type]['chord']
        ctail_m = m_tail
        elements = np.linspace(0, n_elem - 1, n_elem, dtype=int)[beam_number == tail_beam_numbersC[1]]
        i_surf += 1
        for i_elem in elements:
            for i_node in range(n_node_elem):
                airfoil_distribution[i_elem, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
                aero_node[conn[i_elem, i_node]] = True
        surface_distribution[elements] = i_surf
        surface_m[i_surf] = ctail_m
        node_counter = 0
        for i_elem in elements:
            for i_local_node in [0, 1, 2]:
                if not i_local_node == 0:
                    node_counter += 1
                chord[i_elem, i_local_node] = aero_data[type]['chord']
                elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
                twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180

        elements = np.linspace(0, n_elem - 1, n_elem, dtype=int)[beam_number == tail_beam_numbersC[2]]
        i_surf += 1
        for i_elem in elements:
            for i_node in range(n_node_elem):
                airfoil_distribution[i_elem, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
                aero_node[conn[i_elem, i_node]] = True
        surface_distribution[elements] = i_surf
        surface_m[i_surf] = ctail_m
        node_counter = 0
        for i_elem in elements:
            for i_local_node in [0, 1, 2]:
                if not i_local_node == 0:
                    node_counter += 1
                chord[i_elem, i_local_node] = aero_data[type]['chord']
                elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
                twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180

        # 0R tail
        type = '0Rtail'
        rtail_chord = aero_data[type]['chord']
        tail_m = max(3, int(m*rtail_chord/main_chord))
        tail_m = m_tail
        elements = np.linspace(0, n_elem - 1, n_elem, dtype=int)[beam_number == tail_beam_numbersR[0,1]]
        i_surf += 1
        for i_elem in elements:
            for i_node in range(n_node_elem):
                airfoil_distribution[i_elem, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
                aero_node[conn[i_elem, i_node]] = True
        surface_distribution[elements] = i_surf
        surface_m[i_surf] = tail_m
        node_counter = 0
        for i_elem in elements:
            for i_local_node in [0, 1, 2]:
                if not i_local_node == 0:
                    node_counter += 1
                chord[i_elem, i_local_node] = aero_data[type]['chord']
                elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
                twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180
                control_surface[i_elem, i_local_node] = 0

        elements = np.linspace(0, n_elem - 1, n_elem, dtype=int)[beam_number == tail_beam_numbersR[0,2]]
        i_surf += 1
        for i_elem in elements:
            for i_node in range(n_node_elem):
                airfoil_distribution[i_elem, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
                aero_node[conn[i_elem, i_node]] = True
        surface_distribution[elements] = i_surf
        surface_m[i_surf] = tail_m
        node_counter = 0
        for i_elem in elements:
            for i_local_node in [0, 1, 2]:
                if not i_local_node == 0:
                    node_counter += 1
                chord[i_elem, i_local_node] = aero_data[type]['chord']
                elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
                twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180
                control_surface[i_elem, i_local_node] = 0

        # 1R tail
        type = '0Rtail'
        elements = np.linspace(0, n_elem - 1, n_elem, dtype=int)[beam_number == tail_beam_numbersR[1,1]]
        i_surf += 1
        for i_elem in elements:
            for i_node in range(n_node_elem):
                airfoil_distribution[i_elem, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
                aero_node[conn[i_elem, i_node]] = True
        surface_distribution[elements] = i_surf
        surface_m[i_surf] = tail_m
        node_counter = 0
        for i_elem in elements:
            for i_local_node in [0, 1, 2]:
                if not i_local_node == 0:
                    node_counter += 1
                chord[i_elem, i_local_node] = aero_data[type]['chord']
                elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
                twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180
                control_surface[i_elem, i_local_node] = 0

        elements = np.linspace(0, n_elem - 1, n_elem, dtype=int)[beam_number == tail_beam_numbersR[1,2]]
        i_surf += 1
        for i_elem in elements:
            for i_node in range(n_node_elem):
                airfoil_distribution[i_elem, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
                aero_node[conn[i_elem, i_node]] = True
        surface_distribution[elements] = i_surf
        surface_m[i_surf] = tail_m
        node_counter = 0
        for i_elem in elements:
            for i_local_node in [0, 1, 2]:
                if not i_local_node == 0:
                    node_counter += 1
                chord[i_elem, i_local_node] = aero_data[type]['chord']
                elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
                twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180
                control_surface[i_elem, i_local_node] = 0

        # 0L tail
        type = '0Ltail'
        elements = np.linspace(0, n_elem - 1, n_elem, dtype=int)[beam_number == tail_beam_numbersL[0,1]]
        i_surf += 1
        for i_elem in elements:
            for i_node in range(n_node_elem):
                airfoil_distribution[i_elem, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
                aero_node[conn[i_elem, i_node]] = True
        surface_distribution[elements] = i_surf
        surface_m[i_surf] = tail_m
        node_counter = 0
        for i_elem in elements:
            for i_local_node in [0, 1, 2]:
                if not i_local_node == 0:
                    node_counter += 1
                chord[i_elem, i_local_node] = aero_data[type]['chord']
                elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
                twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180
                control_surface[i_elem, i_local_node] = 0

        elements = np.linspace(0, n_elem - 1, n_elem, dtype=int)[beam_number == tail_beam_numbersL[0,2]]
        i_surf += 1
        for i_elem in elements:
            for i_node in range(n_node_elem):
                airfoil_distribution[i_elem, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
                aero_node[conn[i_elem, i_node]] = True
        surface_distribution[elements] = i_surf
        surface_m[i_surf] = tail_m
        node_counter = 0
        for i_elem in elements:
            for i_local_node in [0, 1, 2]:
                if not i_local_node == 0:
                    node_counter += 1
                chord[i_elem, i_local_node] = aero_data[type]['chord']
                elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
                twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180
                control_surface[i_elem, i_local_node] = 0

        # 1L tail
        type = '0Ltail'
        elements = np.linspace(0, n_elem - 1, n_elem, dtype=int)[beam_number == tail_beam_numbersL[1,1]]
        i_surf += 1
        for i_elem in elements:
            for i_node in range(n_node_elem):
                airfoil_distribution[i_elem, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
                aero_node[conn[i_elem, i_node]] = True
        surface_distribution[elements] = i_surf
        surface_m[i_surf] = tail_m
        node_counter = 0
        for i_elem in elements:
            for i_local_node in [0, 1, 2]:
                if not i_local_node == 0:
                    node_counter += 1
                chord[i_elem, i_local_node] = aero_data[type]['chord']
                elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
                twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180
                control_surface[i_elem, i_local_node] = 0

        elements = np.linspace(0, n_elem - 1, n_elem, dtype=int)[beam_number == tail_beam_numbersL[1,2]]
        i_surf += 1
        for i_elem in elements:
            for i_node in range(n_node_elem):
                airfoil_distribution[i_elem, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
                aero_node[conn[i_elem, i_node]] = True
        surface_distribution[elements] = i_surf
        surface_m[i_surf] = tail_m
        node_counter = 0
        for i_elem in elements:
            for i_local_node in [0, 1, 2]:
                if not i_local_node == 0:
                    node_counter += 1
                chord[i_elem, i_local_node] = aero_data[type]['chord']
                elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
                twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180
                control_surface[i_elem, i_local_node] = 0

        type = 'Cfin'
        cfin_chord = aero_data[type]['chord']
        cfin_m = m_fin
        elements = np.linspace(0, n_elem - 1, n_elem, dtype=int)[beam_number == fin_beam_numberC]
        i_surf += 1
        for i_elem in elements:
            for i_node in range(n_node_elem):
                airfoil_distribution[i_elem, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
                aero_node[conn[i_elem, i_node]] = True
        surface_distribution[elements] = i_surf
        surface_m[i_surf] = cfin_m
        node_counter = 0
        for i_elem in elements:
            for i_local_node in [0, 1, 2]:
                if not i_local_node == 0:
                    node_counter += 1
                chord[i_elem, i_local_node] = aero_data[type]['chord']
                elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
                twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180

        type = 'Lfin'
        lfin_chord = aero_data[type]['chord']
        fin_m = m_fin
        elements = np.linspace(0, n_elem - 1, n_elem, dtype=int)[beam_number == fin_beam_numberL]
        i_surf += 1
        for i_elem in elements:
            for i_node in range(n_node_elem):
                airfoil_distribution[i_elem, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
                aero_node[conn[i_elem, i_node]] = True
        surface_distribution[elements] = i_surf
        surface_m[i_surf] = fin_m
        node_counter = 0
        for i_elem in elements:
            for i_local_node in [0, 1, 2]:
                if not i_local_node == 0:
                    node_counter += 1
                chord[i_elem, i_local_node] = aero_data[type]['chord']
                elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
                twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180

        type = 'Rfin'
        elements = np.linspace(0, n_elem - 1, n_elem, dtype=int)[beam_number == fin_beam_numberR]
        i_surf += 1
        for i_elem in elements:
            for i_node in range(n_node_elem):
                airfoil_distribution[i_elem, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
                aero_node[conn[i_elem, i_node]] = True
        surface_distribution[elements] = i_surf
        surface_m[i_surf] = fin_m
        node_counter = 0
        for i_elem in elements:
            for i_local_node in [0, 1, 2]:
                if not i_local_node == 0:
                    node_counter += 1
                chord[i_elem, i_local_node] = aero_data[type]['chord']
                elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
                twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180

        type = 'LLfin'
        lfin_chord = aero_data[type]['chord']
        fin_m = m_fin
        elements = np.linspace(0, n_elem - 1, n_elem, dtype=int)[beam_number == fin_beam_numberLL]
        i_surf += 1
        for i_elem in elements:
            for i_node in range(n_node_elem):
                airfoil_distribution[i_elem, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
                aero_node[conn[i_elem, i_node]] = True
        surface_distribution[elements] = i_surf
        surface_m[i_surf] = fin_m
        node_counter = 0
        for i_elem in elements:
            for i_local_node in [0, 1, 2]:
                if not i_local_node == 0:
                    node_counter += 1
                chord[i_elem, i_local_node] = aero_data[type]['chord']
                elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
                twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180

        type = 'RRfin'
        elements = np.linspace(0, n_elem - 1, n_elem, dtype=int)[beam_number == fin_beam_numberRR]
        i_surf += 1
        for i_elem in elements:
            for i_node in range(n_node_elem):
                airfoil_distribution[i_elem, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
                aero_node[conn[i_elem, i_node]] = True
        surface_distribution[elements] = i_surf
        surface_m[i_surf] = fin_m
        node_counter = 0
        for i_elem in elements:
            for i_local_node in [0, 1, 2]:
                if not i_local_node == 0:
                    node_counter += 1
                chord[i_elem, i_local_node] = aero_data[type]['chord']
                elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
                twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180

        type = 'Cvfin'
        cfin_chord = aero_data[type]['chord']
        cfin_m = m_fin
        elements = np.linspace(0, n_elem - 1, n_elem, dtype=int)[beam_number == vfin_beam_numberC]
        i_surf += 1
        for i_elem in elements:
            for i_node in range(n_node_elem):
                airfoil_distribution[i_elem, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
                aero_node[conn[i_elem, i_node]] = True
        surface_distribution[elements] = i_surf
        surface_m[i_surf] = cfin_m
        node_counter = 0
        for i_elem in elements:
            for i_local_node in [0, 1, 2]:
                if not i_local_node == 0:
                    node_counter += 1
                chord[i_elem, i_local_node] = aero_data[type]['chord']
                elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
                twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180

        type = 'Lvfin'
        lfin_chord = aero_data[type]['chord']
        fin_m = m_fin
        elements = np.linspace(0, n_elem - 1, n_elem, dtype=int)[beam_number == vfin_beam_numberL]
        i_surf += 1
        for i_elem in elements:
            for i_node in range(n_node_elem):
                airfoil_distribution[i_elem, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
                aero_node[conn[i_elem, i_node]] = True
        surface_distribution[elements] = i_surf
        surface_m[i_surf] = fin_m
        node_counter = 0
        for i_elem in elements:
            for i_local_node in [0, 1, 2]:
                if not i_local_node == 0:
                    node_counter += 1
                chord[i_elem, i_local_node] = aero_data[type]['chord']
                elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
                twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180

        type = 'Rvfin'
        elements = np.linspace(0, n_elem - 1, n_elem, dtype=int)[beam_number == vfin_beam_numberR]
        i_surf += 1
        for i_elem in elements:
            for i_node in range(n_node_elem):
                airfoil_distribution[i_elem, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
                aero_node[conn[i_elem, i_node]] = True
        surface_distribution[elements] = i_surf
        surface_m[i_surf] = fin_m
        node_counter = 0
        for i_elem in elements:
            for i_local_node in [0, 1, 2]:
                if not i_local_node == 0:
                    node_counter += 1
                chord[i_elem, i_local_node] = aero_data[type]['chord']
                elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
                twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180

        with h5.File(route + '/' + case_name + '.aero.h5', 'a') as h5file:
            airfoils_group = h5file.create_group('airfoils')
            # add one airfoil
            emx07_main = airfoils_group.create_dataset('0', data=load_airfoil('inputs/EMX-07_camber.txt'))
            flat_tail = airfoils_group.create_dataset('1', data=np.column_stack(
                generate_naca_camber(P=0, M=0)))

            # chord
            chord_input = h5file.create_dataset('chord', data=chord)
            dim_attr = chord_input .attrs['units'] = 'm'

            # twist
            twist_input = h5file.create_dataset('twist', data=twist)
            dim_attr = twist_input.attrs['units'] = 'rad'

            # airfoil distribution
            airfoil_distribution_input = h5file.create_dataset('airfoil_distribution', data=airfoil_distribution)

            surface_distribution_input = h5file.create_dataset('surface_distribution', data=surface_distribution)
            surface_m_input = h5file.create_dataset('surface_m', data=surface_m)
            m_distribution_input = h5file.create_dataset('m_distribution', data=m_distribution.encode('ascii', 'ignore'))

            aero_node_input = h5file.create_dataset('aero_node', data=aero_node)
            elastic_axis_input = h5file.create_dataset('elastic_axis', data=elastic_axis)

            control_surface_input = h5file.create_dataset('control_surface', data=control_surface)
            control_surface_deflection_input = h5file.create_dataset('control_surface_deflection', data=control_surface_deflection)
            control_surface_chord_input = h5file.create_dataset('control_surface_chord', data=control_surface_chord)
            control_surface_types_input = h5file.create_dataset('control_surface_type', data=control_surface_type)
            control_surface_hinge_coord_input = h5file.create_dataset('control_surface_hinge_coord', data=control_surface_hinge_coord)

    def load_airfoil(filename):
        data = np.loadtxt(filename, skiprows=1)
        return data


    def generate_naca_camber(M=0, P=0):
        """
        https://en.wikipedia.org/wiki/NACA_airfoil
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


    def generate_solver_file():
        """
        Main file generator. This text file contains the location of the
        case files, the case name and all the solvers to be run together
        with their settings.
        """
        file_name = route + '/' + case_name + '.sharpy'     # the extension
                                                            # of the file is
                                                            # not important
        settings = dict()
        settings['SHARPy'] = {'case': case_name,
                              'route': route,
                              'flow': flow,
                              'write_screen': 'on',
                              'write_log': 'on',
                              'log_folder': route + '/output/',
                              'log_file': case_name + '.log'}

        settings['BeamLoader'] = {'unsteady': 'on',     # it's ok to leave it
                                                        # 'on' for static
                                  # initial orientation of the aircraft,
                                  # this is where the trim conditions come
                                  'orientation': algebra.euler2quat(np.array([roll,
                                                                              alpha,
                                                                              beta]))}

        settings['AerogridLoader'] = {'unsteady': 'on',
                                      'aligned_grid': 'on',
                                      'mstar': mstar,
                                      # freestream_dir only used for wake init
                                      'freestream_dir': ['1', '0', '0']}

        settings['NonLinearStatic'] = {'print_info': 'off',
                                       'max_iterations': 350,
                                       'num_load_steps': 1,
                                       'delta_curved': 1e-1,
                                       'min_delta': tolerance,
                                       'balancing': 'off',
                                       'gravity_on': gravity,
                                       'gravity': gravity_value}

        settings['StaticUvlm2'] = {'print_info': 'on',
                                   'horseshoe': horseshoe,
                                   'num_cores': 1,
                                   'n_rollup': 0,
                                   'rollup_dt': dt,
                                   'rollup_aic_refresh': 1,
                                   'rollup_tolerance': 1e-4,
                                   'velocity_field_generator': 'SteadyVelocityField',
                                   'velocity_field_input': {'u_inf': u_inf,
                                                            'u_inf_direction': [1., 0, 0]},
                                   'rho': rho}

        settings['StaticCoupled'] = {'print_info': 'on',
                                     'structural_solver': 'NonLinearStatic',
                                     'structural_solver_settings': settings['NonLinearStatic'],
                                     'aero_solver': 'StaticUvlm',
                                     'aero_solver_settings': settings['StaticUvlm2'],
                                     'max_iter': 100,
                                     'n_load_steps': n_step,
                                     'tolerance': fsi_tolerance,
                                     'relaxation_factor': static_relaxation_factor}

        settings['StaticUvlm'] = {'print_info': 'on',
                                  'horseshoe': horseshoe,
                                  'num_cores': 1,
                                  'n_rollup': 0,
                                  'rollup_dt': dt,
                                  'rollup_aic_refresh': 1,
                                  'rollup_tolerance': 1e-4,
                                  'velocity_field_generator': 'SteadyVelocityField',
                                  'velocity_field_input': {'u_inf': u_inf,
                                                           'u_inf_direction': [1., 0, 0]},
                                  'rho': rho}

        # this trim solver is much slower, but supports lateral trim.
        # it is based on scipy.optimize
        settings['Trim'] = {'solver': 'StaticCoupled',
                            'solver_settings': settings['StaticCoupled'],
                            'initial_alpha': alpha,
                            'initial_beta': beta,
                            'initial_roll': roll,
                            'cs_indices': 0,
                            'initial_cs_deflection': cs_deflection,
                            'initial_thrust': [],
                            'thrust_nodes': [],
                            'refine_solution': 'on',
                            'special_case': {'case_name': 'differential_thrust',
                                             'initial_base_thrust': thrustC,
                                             'initial_differential_parameter': differential,
                                             'base_thrust_nodes': [thrust_nodes[0]],
                                             'negative_thrust_nodes': [thrust_nodes[2]],
                                             'positive_thrust_nodes': [thrust_nodes[1]]}}

        settings['NonLinearDynamicCoupledStep'] = {'print_info': 'off',
                                                   'max_iterations': 500,
                                                   'delta_curved': 1e-1,
                                                   'min_delta': tolerance,
                                                   # newmark damping parameter.
                                                   # it affects higher freq
                                                   # mostly.
                                                   'newmark_damp': 1e-3,
                                                   'gravity_on': gravity,
                                                   'gravity': gravity_value,
                                                   'num_steps': n_tstep,
                                                   'balancing': 'off',
                                                   'dt': dt,
                                                   'initial_velocity_direction': np.array([-1.0, 0.0, 0.0]),
                                                   'initial_velocity': u_inf}

        settings['StaticTrim'] = {'solver': 'StaticCoupled',
                                  'solver_settings': settings['StaticCoupled'],
                                  'initial_alpha': alpha,
                                  'initial_deflection': cs_deflection,
                                  'initial_thrust': thrustC,
                                  'thrust_nodes': thrust_nodes}

        settings['StepUvlm'] = {'print_info': 'off',
                                'horseshoe': horseshoe,
                                'num_cores': 6,
                                'n_rollup': 0,
                                'convection_scheme': 2,
                                'rollup_dt': dt,
                                'rollup_aic_refresh': 1,
                                'rollup_tolerance': 1e-4,
                                'gamma_dot_filtering': 6,
                                # this is where the gust is input.
                                'velocity_field_generator': 'GustVelocityField',
                                'velocity_field_input': {'u_inf': u_inf,
                                                         'u_inf_direction': [1., 0, 0],
                                                         'gust_shape': gust_shape,
                                                         'gust_length': gust_length,
                                                         'gust_intensity': gust_intensity*u_inf,
                                                         'offset': -space_offset,
                                                         'span': span_main},
                                'rho': rho,
                                'n_time_steps': n_tstep,
                                'dt': dt}

        settings['DynamicCoupled'] = {'structural_solver': 'NonLinearDynamicCoupledStep',
                                      'structural_solver_settings': settings['NonLinearDynamicCoupledStep'],
                                      'aero_solver': 'StepUvlm',
                                      'aero_solver_settings': settings['StepUvlm'],
                                      'fsi_substeps': 100,
                                      'fsi_tolerance': fsi_tolerance,
                                      'relaxation_factor': initial_relaxation_factor,
                                      'final_relaxation_factor': final_relaxation_factor,
                                      'minimum_steps': 1,
                                      'relaxation_steps': relaxation_steps,
                                      'n_time_steps': n_tstep,
                                      'dt': dt,
                                      'include_unsteady_force_contribution': 'on',
                                      'cleanup_previous_solution': 'on',
                                      # the postprocessors in this list are
                                      # called every time step. Some of them
                                      # are the same you call in static simulations
                                      # and you wrote at the beginning of the
                                      # file
                                      'postprocessors': [
                                            # output some variables by text
                                            'WriteVariablesTime',
                                            # remove old timesteps
                                            # already output to save RAM
                                            'Cleanup',
                                            # calculate internal
                                            # beam loads and strains
                                            'BeamLoads',
                                            'BeamPlot',
                                            'AerogridPlot',
                                            # saves a copy of self.data
                                            # (the state of the simulation)
                                            # to restart later if needed.
                                            # careful, because if you don't
                                            # add Cleanup and don't pay attention
                                            # to the 'frequency' and
                                            # 'keep' parameters, you might
                                            # fill you HD for long simulations
                                            'CreateSnapshot',
                                            ],
                                      'postprocessors_settings': {'BeamLoads': {}, # you need to specify the dict even if empty
                                                                  'BeamPlot': {'include_rbm': 'on',
                                                                               'include_applied_forces': 'on'},
                                                                  'AerogridPlot': {
                                                                      'include_rbm': 'on',
                                                                      'include_applied_forces': 'on',
                                                                      'minus_m_star': 0},
                                                                  'CreateSnapshot': {# every how many tsteps if should create a snapshot
                                                                                     'frequency': 100,
                                                                                     # how many old snapshots should it keep?
                                                                                     # the rest will be discarded.
                                                                                     'keep': 5},
                                                                  'Cleanup': {# keep only the last 2000 tsteps.
                                                                              # it is good to keep a relatively large number
                                                                              # in order to have a good noise filter
                                                                              # for gamma_dot
                                                                              'remaining_steps': 2000
                                                                              },
                                                                  'WriteVariablesTime': {
                                                                      # output a text file with
                                                                      # quat: orientation of the aircraft in quaternion
                                                                      # for_pos: FoR position in inertial FoR
                                                                      # for_vel: the derivative of the previous
                                                                      # for_acc: the derivative of the previous
                                                                      'FoR_variables': ['quat', 'for_pos', 'for_vel', 'for_acc'],
                                                                      'cleanup_old_solution': 'on',
                                                                      'delimiter': ', ',
                                                                  }
                                                              }
                                                            }

        settings['Modal'] = {'print_info': 'on',
            'use_undamped_modes': 'on',
            'NumLambda': 100,
            'write_modes_vtk': 'on',
            'print_matrices': 'on',
            'write_data': 'on',
            'continuous_eigenvalues': 'off',
            'dt': dt,
            'plot_eigenvalues': 'on'}
        settings['BeamPlot'] = {'include_rbm': 'on',
            'include_applied_forces': 'on',
            'include_forward_motion': 'off'}

        settings['AerogridPlot'] = {'include_rbm': 'on',
                                    'include_forward_motion': 'off',
                                    'include_applied_forces': 'on',
                                    'minus_m_star': 0,
                                    'u_inf': u_inf,
                                    'dt': dt}
        settings['BeamLoads'] = dict()

        import configobj
        config = configobj.ConfigObj()
        config.filename = file_name
        for k, v in settings.items():
            config[k] = v
        config.write()



    clean_test_files()
    generate_fem()
    generate_aero_file()
    generate_solver_file()
