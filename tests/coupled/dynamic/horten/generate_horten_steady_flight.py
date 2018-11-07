# GENERATE FLEXIBLE SWEPT FLYING WING GEOMETRY
# NORBERTO GOIZUETA 26 SEPT 2018
#
# LOG:
# 180927: Successful first run of SFW

import numpy as np
import h5py as h5
import os
import sharpy.utils.algebra as algebra

case_name = 'horten'
route = os.path.dirname(os.path.realpath(__file__)) + '/'

# EXECUTION
# NG_180926 To be confirmed
flow = ['BeamLoader',
        'AerogridLoader',
        # 'NonLinearStatic',
        #'StaticUvlm',
        # 'Trim',
        #'StaticTrim',
        'StaticCoupled',
        'BeamLoads',
        'AerogridPlot',
        'BeamPlot',
        'DynamicCoupled',
        'AeroForcesCalculator',
        # 'Modal'
        ]

# CONSTANTS
# inches to meters m/in
in2m = 0.0254
# pounds to Newtons N/lbf
lbf2n = 4.448

# FLIGHT CONDITIONS
u_inf = 25.0  # [m/s] Flight velocity
rho = 1.225   # [kg/m3] Air density

# TRIM CONDITIONS
alpha_rad = 4.011184119715089*np.pi/180
beta = 0.0
roll = 0.0
gravity = 'on'
cs_deflection = -3.8625342948311894*np.pi/180
thrust = 3.368379648272879
sigma = 1
dihedral = 0*np.pi/180

# GUST CONDITIONS
gust_intensity = 0.0
n_step = 3
relaxation_factor = 0.3
tolerance = 1e-5
fsi_tolerance = 1e-7

# WING GEOMETRY
span = 20.0   # [m]
sweep_LE = 20*np.pi/180 # [rad] Leading Edge Sweep
c_root = 1.0 # [m] Root chord
taper_ratio = 0.25
main_ea = 0.25

# FUSELAGE GEOMETRY
fuselage_width = 1.65

# WASH OUT
washout_root = -0.2*np.pi/180
washout_tip = -2*np.pi/180

# STIFFNESS PROPERTIES
ea = 1e6
ga = 1e6
gj = 4.24e5
eiy = 3.84e5
eiz = 2.46e7


# WING PROPERTIES Ref Richards et al 2016
# mass per unit span
# mu = kg/m
# mu = mu_ref*c_bar**2
mu_0 = 9.761



# polar moment of inertia per unit span linear variation
# From Richards, GJ_root = 4.24e5 N m^2 using VABS
# G = 27 GPa for typical aluminium alloy
# Estimate polar mass moment of inertia: [kg m]
j_root = 4.24e5/27e9*2700


# LUMPED MASSES
n_lumped_mass = 3
lumped_mass_nodes = np.zeros((n_lumped_mass), dtype=int)
lumped_mass = np.zeros((n_lumped_mass))
lumped_mass_inertia = np.zeros((n_lumped_mass, 3, 3))
lumped_mass_position = np.zeros((n_lumped_mass, 3))

# engines
lumped_mass[0:2] = 51.445/9.81
lumped_mass_inertia[0] = [0.29547, 0.29322, 0.29547]
lumped_mass_inertia[1] = [0.29547, 0.29322, 0.29547]

# lumped_mass_position[0] = [0, fuselage_width/2, 0]
# lumped_mass_position[1] = [0, -fuselage_width/2, 0]

# fuselage
lumped_mass[2] = 150/9.81
lumped_mass_inertia[2] = np.array([0.5, 1.0, 1.0])*lumped_mass[2]
lumped_mass_nodes[2] = 0
# lumped_mass_position[2] = [0, 0, 0]


# NUMERICAL DISCRETISATION
# SPATIAL
wake_length = 8
horseshoe_on = False
m = 4
n_elem_fuselage = 1
n_elem_wing = 11
n_surfaces = 4

# TIME
physical_time = 50
tstep_factor = 1.
dt = 1.0/m/u_inf*tstep_factor
n_tstep = round(physical_time/dt)

###
# END OF INPUT SECTION
###
#------------------------------------------------------------------------------
###
# SHARPY PRE-PROCESSING
###

# BEAM ELEMENT DEFINITION
n_node_elem = 3 # Quadratic Element Definition

# total number of elements
n_elem = 0
n_elem += 2*(n_elem_wing + n_elem_fuselage)

# total number of nodes
n_node = 0
n_node_wing = n_elem_wing*(n_node_elem-1)
n_node_fuselage = n_elem_fuselage*n_node_elem
n_node += 2*n_node_fuselage - 1 + 2*n_node_wing

# THRUST NODES
thrust_nodes = [n_node_fuselage-1,
                n_node_fuselage + n_node_wing + 1]


# MASS VARIES LINEARLY ALONG SPAN
# mass matrix database
c_bar_temp = np.linspace(c_root, taper_ratio*c_root,n_elem_wing)
mu_temp = mu_0*c_bar_temp**2
j_temp = j_root*c_bar_temp**2


n_mass = n_elem_wing
base_mass_db = np.ndarray((n_mass, 6, 6))

for mass_entry in range(n_mass):
    base_mass_db[mass_entry, :, :] = np.diag([mu_temp[mass_entry],
                                        mu_temp[mass_entry],
                                        mu_temp[mass_entry],
                                        j_temp[mass_entry],
                                        j_temp[mass_entry],
                                        j_temp[mass_entry]
                                              ])

# STIFFNESS DATA PREPARATION
# stiffness matrix and number of different stiffnesses employed
n_stiffness = n_elem_wing + n_elem_fuselage
#base_stiffness = sigma*np.diag([ea, ga, ga, gj, eiy, eiz])
base_stiffness = np.ndarray((n_stiffness, 6, 6))
sigma_fuselage = 1000
stiffness_root = sigma*np.diag([ea, ga, ga, gj, eiy, eiz])

base_stiffness[0, :, :] = sigma_fuselage*stiffness_root
for stiff_entry in range(1, n_stiffness):
    base_stiffness[stiff_entry, :, :] = stiffness_root*c_bar_temp[stiff_entry-1]**3


# Lumped mass nodal positions
lumped_mass_nodes[0] = 2
lumped_mass_nodes[1] = n_node_fuselage + n_node_wing + 1


# H5 FEM FILE VARIABLES INITIALISATION
# coordinates
x = np.zeros((n_node, ))
y = np.zeros((n_node, ))
z = np.zeros((n_node, ))
# twist
structural_twist = np.zeros_like(x)
# beam number
beam_number = np.zeros(n_elem, dtype=int)
# frame of reference delta
frame_of_reference_delta = np.zeros((n_elem, n_node_elem, 3))
# connectivity of beams
conn = np.zeros((n_elem, n_node_elem), dtype=int)
# stiffness
stiffness = np.zeros((n_stiffness, 6, 6))
elem_stiffness = np.zeros((n_elem, ), dtype=int)
# mass
mass = np.zeros((n_mass, 6, 6))
elem_mass = np.zeros((n_elem), dtype=int)
# boundary conditions
boundary_conditions = np.zeros((n_node, ), dtype=int)
# applied forces
app_forces = np.zeros((n_node, 6))

# H5 AERO FILE VARIABLES INITIALISATION
# airfoil distribution
airfoil_distribution = np.zeros((n_elem, n_node_elem), dtype=int)
# surface distribution
surface_distribution = np.zeros((n_elem, ), dtype=int) - 1
surface_m = np.zeros((n_surfaces, ), dtype=int)
m_distribution = 'uniform'
# aerodynamic nodes boolean
aero_nodes = np.zeros((n_node, ), dtype=bool)
# aero twist
twist = np.zeros((n_elem, n_node_elem))
# chord
chord = np.zeros((n_elem, n_node_elem))
# elastic axis
elastic_axis = np.zeros((n_elem, n_node_elem))

# -----------------------------------------------------------------------
# FUNCTION DEFINITIONS

def clean_test_files():
    """
    Clears previously generated files
    """

    # FEM
    fem_file_name = route + '/' + case_name + '.fem.h5'
    if os.path.isfile(fem_file_name):
        os.remove(fem_file_name)

    # Dynamics File
    dyn_file_name = route + '/' + case_name + '.dyn.h5'
    if os.path.isfile(dyn_file_name):
        os.remove(dyn_file_name)

    # Aerodynamics File
    aero_file_name = route + '/' + case_name + '.aero.h5'
    if os.path.isfile(aero_file_name):
        os.remove(aero_file_name)

    # Solver file
    solver_file_name = route + '/' + case_name + '.solver.txt'
    if os.path.isfile(solver_file_name):
        os.remove(solver_file_name)

    # Flight conditions file
    flightcon_file_name = route + '/' + case_name + '.flightcon.txt'
    if os.path.isfile(flightcon_file_name):
        os.remove(flightcon_file_name)

# Generate FEM H5 file
def generate_fem():
    """
    Generates H5 FEM file
    :return:
    """

    # mass and stiffness matrices
    stiffness = base_stiffness
    mass = base_mass_db

    # assemble connectivites
    # worked elements
    we = 0
    # worked nodes
    wn = 0

    # RIGHT RIGID FUSELAGE
    beam_number[we:we+1] = 0
    # coordinates
    x[wn:wn+n_node_fuselage] = 0
    y[wn:wn+n_node_fuselage] = np.linspace(0, fuselage_width/2, n_node_fuselage)

    # connectivities
    elem_mass[0] = 0
    conn[we, :] = [0, 2, 1]

    # frame of reference change
    frame_of_reference_delta[0, 0, :] = [-1.0, 0.0, 0.0]
    frame_of_reference_delta[0, 1, :] = [-1.0, 0.0, 0.0]
    frame_of_reference_delta[0, 2, :] = [-1.0, 0.0, 0.0]

    # element stiffness
    elem_stiffness[0] = 0
    elem_mass[0] = 0

    # boundary conditions
    boundary_conditions[0] = 1

    # applied forces - engine 1
    app_forces[thrust_nodes[0]] = [0, thrust, 0,
                     0, 0, 0]

    # updated worked nodes and elements
    we += n_elem_fuselage
    wn += n_node_fuselage


    # RIGHT WING
    beam_number[we:we + n_elem_wing] = 1
    # y coordinate (positive right)
    y[wn:wn+n_node_wing] = np.linspace(fuselage_width/2,
                                       span/2,
                                       n_node_wing+1)[1:]
    x[wn:wn+n_node_wing] = 0 + (y[wn:wn+n_node_wing]-fuselage_width/2)*np.tan(sweep_LE)

    # connectivities
    for ielem in range(n_elem_wing):
        conn[we + ielem, :] = (np.ones(n_node_elem)*(we+ielem)*(n_node_elem - 1) +
                              [0, 2, 1])
        for inode in range(n_node_elem):
            frame_of_reference_delta[we+ielem, inode, :] = [-1.0, 0.0, 0.0]

        elem_mass[we+ielem] = ielem
        elem_stiffness[we+ielem] = ielem + 1

    # element stiffness and mass
    #elem_stiffness[we:we+n_elem_wing] = 0
    #elem_mass[we:we+n_elem_wing] = 0

    # boundary conditions of free end
    boundary_conditions[wn+n_node_wing-1] = -1

    # update worked elements and nodes
    we += n_elem_wing
    wn += n_node_wing

    # LEFT FUSELAGE
    beam_number[we:we+n_elem_fuselage] = 2
    # coordinates
    y[wn:wn+n_node_fuselage-1] = np.linspace(0,
                                             -fuselage_width/2,
                                             n_node_fuselage)[1:]
    x[wn:wn+n_node_fuselage-1] = 0

    # connectivity
    conn[we, :] = [0, wn + 1, wn]

    # frame of reference delta
    for ielem in range(n_elem_fuselage):
        for inode in range(n_node_elem):
            frame_of_reference_delta[we+ielem, inode, :] = [1.0, 0.0, 0.0]

    # element stiffness and mass
    elem_stiffness[we:we+n_elem_fuselage] = 0
    elem_mass[we:we+n_elem_fuselage] = 0

    # applied forces - engine 2
    app_forces[thrust_nodes[1]] = [0, -thrust, 0,
                        0, 0, 0]

    # update worked elements and nodes
    we += n_elem_fuselage
    wn += n_node_fuselage - 1

    # LEFT WING
    # coordinates
    beam_number[we:we+n_elem_wing] = 3
    y[wn:wn+n_node_wing] = np.linspace(-fuselage_width/2,
                                       -span/2,
                                       n_node_wing+1)[1:]
    x[wn:wn + n_node_wing] = 0 + -1*(y[wn:wn+n_node_wing]+fuselage_width/2)*np.tan(sweep_LE)

    # left wing connectivities
    for ielem in range(n_elem_wing):
        conn[we + ielem, :] = np.ones(n_node_elem)*(we+ielem)*(n_node_elem-1) + [0, 2, 1]

        for inode in range(n_node_elem):
            frame_of_reference_delta[we+ielem, inode, :] = [1.0, 0.0, 0.0]

        elem_mass[we+ielem] = ielem
        elem_stiffness[we+ielem] = ielem + 1


    # element stiffness and mass
    #elem_stiffness[we:we+n_node_wing] = 0

    # boundary conditions at the free end
    boundary_conditions[wn+n_node_wing-1] = -1

    # update worked elements and nodes
    we += n_elem_wing
    wn += n_node_wing

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
                'stiffness_db', data=stiffness)
            stiffness_handle = h5file.create_dataset(
                'elem_stiffness', data=elem_stiffness)
            mass_db_handle = h5file.create_dataset(
                'mass_db', data=mass)
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

    if False:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(x, y)
        plt.scatter(x[boundary_conditions == -1], y[boundary_conditions == -1], s=None)
        plt.scatter(x[boundary_conditions == 1], y[boundary_conditions == 1], s=None)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
        plt.figure()
        plt.scatter(y, z)
        plt.scatter(y[boundary_conditions == -1], z[boundary_conditions == -1], s=None)
        plt.scatter(y[boundary_conditions == 1], z[boundary_conditions == 1], s=None)
        plt.xlabel('y')
        plt.ylabel('z')
        plt.show(block = True)



def generate_aero_file():
    global x, y, z

    # control surfaces
    n_control_surfaces = 1
    control_surface = np.zeros((n_elem, n_node_elem), dtype=int) - 1
    control_surface_type = np.zeros((n_control_surfaces, ), dtype=int)
    control_surface_deflection = np.zeros((n_control_surfaces, ))
    control_surface_chord = np.zeros((n_control_surfaces, ), dtype=int)
    control_surface_hinge_coord = np.zeros((n_control_surfaces, ), dtype=float)

    # control surface type: 0 = static
    # control surface type: 1 = dynamic
    control_surface_type[0] = 0
    control_surface_deflection[0] = cs_deflection
    control_surface_chord[0] = 1 #m
    control_surface_hinge_coord[0] = 0.25

    # RIGHT FUSELAGE (Surface 0, Beam 0)
    we = 0
    wn = 0

    i_surf = 0
    airfoil_distribution[we:we+n_elem_fuselage] = 0
    surface_distribution[we:we+n_elem_fuselage] = i_surf
    surface_m[i_surf] = m

    aero_nodes[wn:wn+n_node_fuselage] = True

    temp_chord = c_root
    temp_washout = 0

    # apply chord and elastic axis at each node
    node_counter = 0
    for ielem in range(we, we + n_elem_fuselage):
        for i_local_node in range(n_node_elem):
            if not i_local_node == 0:
                node_counter += 1
            chord[ielem, i_local_node] = temp_chord
            elastic_axis[ielem, i_local_node] = main_ea
            twist[ielem, i_local_node] = -temp_washout

    we += n_elem_fuselage
    wn += n_node_fuselage


    # RIGHT WING (Surface 1, Beam 1)
    # surface_id
    i_surf = 1
    airfoil_distribution[we:we+n_elem_wing, :] = 0
    surface_distribution[we:we+n_elem_wing] = i_surf
    surface_m[i_surf] = m

    # specify aerodynamic characteristics of wing nodes
    aero_nodes[wn:wn+n_node_wing-1] = True

    # linear taper initialisation
    temp_chord = np.linspace(c_root, taper_ratio*c_root, n_node_wing+1)

    # linear wash out initialisation
    temp_washout = np.linspace(washout_root, washout_tip, n_node_wing+1)

    # apply chord and elastic axis at each node
    node_counter = 0
    for ielem in range(we, we + n_elem_wing):
        for i_local_node in range(n_node_elem):
            if not i_local_node == 0:
                node_counter += 1
            chord[ielem, i_local_node] = temp_chord[node_counter]
            elastic_axis[ielem, i_local_node] = main_ea
            twist[ielem, i_local_node] = -temp_washout[node_counter]
            if ielem >= round(((we+n_elem_wing)/2)):
                control_surface[ielem, i_local_node] = 0

    # update working element and node
    we += n_elem_wing
    wn += n_node_wing - 1

    # LEFT FUSELAGE (Surface 2, Beam 2)
    i_surf = 2
    airfoil_distribution[we:we+n_elem_fuselage] = 0
    surface_distribution[we:we+n_elem_fuselage] = i_surf
    surface_m[i_surf] = m

    aero_nodes[wn:wn+n_node_fuselage] = True

    temp_chord = c_root
    temp_washout = 0

    # apply chord and elastic axis at each node
    node_counter = 0
    for ielem in range(we, we + n_elem_fuselage):
        for i_local_node in range(n_node_elem):
            if not i_local_node == 0:
                node_counter += 1
            chord[ielem, i_local_node] = temp_chord
            elastic_axis[ielem, i_local_node] = main_ea
            twist[ielem, i_local_node] = -temp_washout

    we += n_elem_fuselage
    wn += n_node_fuselage

    # LEFT WING (Surface 3, Beam 3)
    i_surf = 3
    airfoil_distribution[we:we + n_elem_wing, :] = 0
    surface_distribution[we: we + n_elem_wing] = i_surf
    surface_m[i_surf] = m

    # linear taper initialisation
    temp_chord = np.linspace(c_root, taper_ratio*c_root, n_node_wing+1)

    # linear wash out initialisation
    temp_washout = np.linspace(washout_root, washout_tip, n_node_wing+1)

    # specify aerodynamic characterisics of wing nodes
    aero_nodes[wn:wn + n_node_wing] = True

    # linear taper initialisation
    # apply chord and elastic axis at each node
    node_counter = 0
    for ielem in range(we, we + n_elem_wing):
        for i_local_node in range(n_node_elem):
            if not i_local_node == 0:
                node_counter += 1
            chord[ielem, i_local_node] = temp_chord[node_counter]
            elastic_axis[ielem, i_local_node] = main_ea
            twist[ielem, i_local_node] = -temp_washout[node_counter]
            if ielem >= round((we+n_elem_wing/2)):
                control_surface[ielem, i_local_node] = 0

    # update working element and node
    we += n_elem_wing
    wn += n_node_wing

    # end node is the middle node
    mid_chord = np.array(chord[:,1],copy=True)
    chord[:,1] = chord[:,2]
    chord[:,2] = mid_chord

    with h5.File(route + '/' + case_name + '.aero.h5', 'a') as h5file:
        airfoils_group = h5file.create_group('airfoils')
        # add one airfoil
        naca_airfoil_main = airfoils_group.create_dataset('0', data=np.column_stack(
            generate_naca_camber(P=0, M=0)))
        naca_airfoil_tail = airfoils_group.create_dataset('1', data=np.column_stack(
            generate_naca_camber(P=0, M=0)))
        naca_airfoil_fin = airfoils_group.create_dataset('2', data=np.column_stack(
            generate_naca_camber(P=0, M=0)))

        # chord
        chord_input = h5file.create_dataset('chord', data=chord)
        dim_attr = chord_input.attrs['units'] = 'm'

        # twist
        twist_input = h5file.create_dataset('twist', data=twist)
        dim_attr = twist_input.attrs['units'] = 'rad'

        # airfoil distribution
        airfoil_distribution_input = h5file.create_dataset('airfoil_distribution', data=airfoil_distribution)

        surface_distribution_input = h5file.create_dataset('surface_distribution', data=surface_distribution)
        surface_m_input = h5file.create_dataset('surface_m', data=surface_m)
        m_distribution_input = h5file.create_dataset('m_distribution', data=m_distribution.encode('ascii', 'ignore'))

        aero_node_input = h5file.create_dataset('aero_node', data=aero_nodes)
        elastic_axis_input = h5file.create_dataset('elastic_axis', data=elastic_axis)

        control_surface_input = h5file.create_dataset('control_surface', data=control_surface)
        control_surface_deflection_input = h5file.create_dataset('control_surface_deflection', data=control_surface_deflection)
        control_surface_chord_input = h5file.create_dataset('control_surface_chord', data=control_surface_chord)
        control_surface_hinge_coord_input = h5file.create_dataset('control_surface_hinge_coord', data=control_surface_hinge_coord)
        control_surface_types_input = h5file.create_dataset('control_surface_type', data=control_surface_type)


def generate_naca_camber(M = 0, P = 0):
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

def generate_solver_file(horseshoe=False):
    file_name = route + '/' + case_name + '.solver.txt'
    # config = configparser.ConfigParser()

    settings = dict()
    settings['SHARPy'] = {'case': case_name,
                        'route': route,
                        'flow': flow,
                        'write_screen': 'on',
                        'write_log': 'on',
                        'log_folder': route + '/output/',
                        'log_file': case_name + '.log'}
    settings['BeamLoader'] = {'unsteady': 'on',
                            'orientation': algebra.euler2quat(np.array([0.0,
                                                                        alpha_rad,
                                                                        beta*np.pi/180]))}

    settings['StaticUvlm'] = {'print_info': 'on',
                                'horseshoe': horseshoe_on,
                                'num_cores': 4,
                                'n_rollup': 1,
                                'rollup_dt': dt,
                                'rollup_aic_refresh': 1,
                                'rollup_tolerance': 1e-4,
                                'velocity_field_generator': 'SteadyVelocityField',
                                'velocity_field_input': {'u_inf': u_inf,
                                                         'u_inf_direction': [1., 0, 0]},
                                'rho': rho}


    settings['StaticCoupled'] = {'print_info': 'on',
                               'structural_solver': 'NonLinearStatic',
                               'structural_solver_settings': {'print_info': 'off',
                                                              'max_iterations': 150,
                                                              'num_load_steps': 1,
                                                              'delta_curved': 1e-5,
                                                              'min_delta': 1e-5,
                                                              'gravity_on': 'on',
                                                              'gravity': 9.81},
                               'aero_solver': 'StaticUvlm',
                               'aero_solver_settings': {'print_info': 'off',
                                                        'horseshoe': horseshoe_on,
                                                        'num_cores': 4,
                                                        'n_rollup': int(0*100),
                                                        'rollup_dt': c_root/m/u_inf,
                                                        'rollup_aic_refresh': 1,
                                                        'rollup_tolerance': 1e-4,
                                                        'velocity_field_generator': 'SteadyVelocityField',
                                                        'velocity_field_input': {'u_inf': u_inf,
                                                                                 'u_inf_direction': [1., 0, 0]},
                                                        'rho': rho,
                                                        'alpha': alpha_rad,
                                                        'beta': beta},
                               'max_iter': 100,
                               'n_load_steps': 1,
                               'tolerance': 1e-6,
                               'relaxation_factor': 0.}

    if horseshoe is True:
        settings['AerogridLoader'] = {'unsteady': 'off',
                                    'aligned_grid': 'on',
                                    'mstar': 1,
                                    'freestream_dir': ['1', '0', '0']}
    else:
        settings['AerogridLoader'] = {'unsteady': 'off',
                                    'aligned_grid': 'on',
                                    'mstar': int(wake_length/dt/u_inf),
                                    'freestream_dir': ['1', '0', '0']}

    settings['NonLinearStatic'] = {'print_info': 'off',
                                   'max_iterations': 150,
                                   'num_load_steps': 1,
                                   'delta_curved': 1e-8,
                                   'min_delta': tolerance,
                                   'gravity_on': gravity,
                                   'gravity': 9.81}



    # settings['StaticCoupled'] = {'print_info': 'off',
    #                              'structural_solver': 'NonLinearStatic',
    #                              'structural_solver_settings': settings['NonLinearStatic'],
    #                              'aero_solver': 'StaticUvlm',
    #                              'aero_solver_settings': settings['StaticUvlm'],
    #                              'max_iter': 100,
    #                              'n_load_steps': n_step,
    #                              'tolerance': fsi_tolerance,
    #                              'relaxation_factor': relaxation_factor}

    settings['StaticTrim'] = {'solver': 'StaticCoupled',
                              'solver_settings': settings['StaticCoupled'],
                              'thrust_nodes': thrust_nodes,
                              'initial_alpha': alpha_rad,
                              'initial_deflection': cs_deflection,
                              'initial_thrust': thrust}

    settings['Trim'] = {'solver': 'StaticCoupled',
                        'solver_settings': settings['StaticCoupled'],
                        'initial_alpha': alpha_rad,
                        'initial_beta': beta,
                        'cs_indices': [0],
                        'initial_cs_deflection': [cs_deflection],
                        'initial_thrust': [thrust]}

    settings['NonLinearDynamicCoupledStep'] = {'print_info': 'off',
                                               'initial_velocity_direction': [-1., 0., 0.],
                                               'max_iterations': 950,
                                               'delta_curved': 1e-6,
                                               'min_delta': tolerance,
                                               'newmark_damp': 5e-3,
                                               'gravity_on': gravity,
                                               'gravity': 9.81,
                                               'num_steps': n_tstep,
                                               'dt': dt,
                                               'initial_velocity': u_inf}

    settings['StepUvlm'] = {'print_info': 'off',
                            'horseshoe': horseshoe_on,
                            'num_cores': 4,
                            'n_rollup': 100,
                            'convection_scheme': 2,
                            'rollup_dt': dt,
                            'rollup_aic_refresh': 1,
                            'rollup_tolerance': 1e-4,
                            # 'velocity_field_generator': 'TurbSimVelocityField',
                            # 'velocity_field_input': {'turbulent_field': '/2TB/turbsim_fields/TurbSim_wide_long_A_low.h5',
                            #                          'offset': [30., 0., -10],
                            #                          'u_inf': 0.},
                            # 'velocity_field_generator': 'GustVelocityField',
                            # 'velocity_field_input': {'u_inf': 0*u_inf,
                            #                          'u_inf_direction': [1., 0, 0],
                            #                          'gust_shape': '1-cos',
                            #                          'gust_length': 1,
                            #                          'gust_intensity': gust_intensity*u_inf,
                            #                          'offset': 5.0,
                            #                          'span': span},
                            'velocity_field_generator': 'SteadyVelocityField',
                            'velocity_field_input': {'u_inf': u_inf*0,
                                                        'u_inf_direction': [1., 0., 0.]},
                            'rho': rho,
                            'n_time_steps': n_tstep,
                            'dt': dt}

    settings['DynamicCoupled'] = {'print_info': 'on',
                                  'structural_substeps': 1,
                                  'dynamic_relaxation': 'on',
                                  'clean_up_previous_solution': 'on',
                                  'structural_solver': 'NonLinearDynamicCoupledStep',
                                  'structural_solver_settings': settings['NonLinearDynamicCoupledStep'],
                                  'aero_solver': 'StepUvlm',
                                  'aero_solver_settings': settings['StepUvlm'],
                                  'fsi_substeps': 200,
                                  'fsi_tolerance': fsi_tolerance,
                                  'relaxation_factor': relaxation_factor,
                                  'minimum_steps': 1,
                                  'relaxation_steps': 150,
                                  'final_relaxation_factor': 0.0,
                                  'n_time_steps': n_tstep,
                                  'dt': dt,
                                  'include_unsteady_force_contribution': 'off',
                                  'postprocessors': ['BeamLoads', 'StallCheck', 'BeamPlot', 'AerogridPlot'],
                                  'postprocessors_settings': {'BeamLoads': {'folder': route + '/output/',
                                                                            'csv_output': 'off'},
                                                              'StallCheck': {'output_degrees': True,
                                                                             'stall_angles': {'0': [-12*np.pi/180, 6*np.pi/180],
                                                                                              '1': [-12*np.pi/180, 6*np.pi/180],
                                                                                              '2': [-12*np.pi/180, 6*np.pi/180]}},
                                                              'BeamPlot': {'folder': route + '/output/',
                                                                           'include_rbm': 'on',
                                                                           'include_applied_forces': 'on'},
                                                              'AerogridPlot': {
                                                                  'u_inf':u_inf,
                                                                  'folder': route + '/output/',
                                                                  'include_rbm': 'on',
                                                                  'include_applied_forces': 'on',
                                                                  'minus_m_star': 0}}}

    settings['Modal'] = {'print_info': True,
                         'use_undamped_modes': False,
                         'NumLambda': 20,
                         'write_modes_vtk': 'on',
                         'print_matrices': 'on',
                         'write_data': 'on',
                         'continuous_eigenvalues': 'off',
                         'dt': dt,
                         'plot_eigenvalues': False}

    settings['AerogridPlot'] = {'folder': route + '/output/',
                              'include_rbm': 'off',
                              'include_applied_forces': 'on',
                              'minus_m_star': 0,
                              'u_inf': u_inf
                              }
    settings['AeroForcesCalculator'] = {'folder': route + '/output/forces',
                                      'write_text_file': 'on',
                                      'text_file_name': case_name + '_aeroforces.csv',
                                      'screen_output': 'on',
                                      'unsteady': 'off'
                                      }
    settings['BeamPlot'] = {'folder': route + '/output/',
                          'include_rbm': 'off',
                          'include_applied_forces': 'on'}

    settings['BeamLoads'] = {'folder': route + '/output/',
                        'csv_output': 'off'}

    import configobj
    config = configobj.ConfigObj()
    config.filename = file_name
    for k, v in settings.items():
        config[k] = v
    config.write()



clean_test_files()
generate_fem()
generate_aero_file()
generate_solver_file(horseshoe=horseshoe_on)
