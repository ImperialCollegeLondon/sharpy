#! /usr/bin/env python3
import h5py as h5
import numpy as np
import os
import pandas as pd
import sharpy.utils.algebra as algebra
import matplotlib.pyplot as plt


case_name = 'simple_wing_fuselage'
route = os.path.dirname(os.path.realpath(__file__)) + '/'

# EXECUTION
flow = ['BeamLoader',
        'AerogridLoader',
        # 'NonLinearStatic',
        'StaticUvlm',
        #'StaticTrim',
         # 'StaticCoupled',
        'BeamLoads',
        'AerogridPlot',
        'BeamPlot',
        #'DynamicCoupled',
        # 'Modal',
        # 'LinearAssember',
        # 'AsymptoticStability',
        ]


# FLIGHT CONDITIONS
# the simulation is set such that the aircraft flies at a u_inf velocity while
# the air is calm.
u_inf = 10
rho = 1.225

free_flight = True
if not free_flight:
    case_name += '_prescribed'
    amplitude = 0*np.pi/180
    period = 3
    case_name += '_amp_' + str(amplitude).replace('.', '') + '_period_' + str(period)

alpha = 6.0*np.pi/180
beta = 0
roll = 0
gravity = 'on'
thrust = 0
sigma = 1.5

# gust settings
gust_intensity = 0.20
gust_length = 1*u_inf
gust_offset = 0.5*u_inf

# numerics
n_step = 5
structural_relaxation_factor = 0.6
relaxation_factor = 0.35
tolerance = 1e-6
fsi_tolerance = 1e-4

num_cores = 2

# MODEL GEOMETRY
# beam
span_main = 3.0
ea_main = 0.3

ea = 1e7
ga = 1e5
gj = 1e4
eiy = 2e4
eiz = 4e6
m_bar_main = 0.75
j_bar_main = 0.075

length_fuselage = 10
offset_fuselage_vertical = 0
offset_fuselage_wing = 4
radius_fuselage = 1.3333333333333333/2
list_cylinder_position_fuselage = [0.3, 0.7] # percent where fuselage has cylinder shape
sigma_fuselage = 10
m_bar_fuselage = 0.2
j_bar_fuselage = 0.08

# lumped masses
n_lumped_mass = 1
lumped_mass_nodes = np.zeros((n_lumped_mass, ), dtype=int)
lumped_mass = np.zeros((n_lumped_mass, ))
lumped_mass[0] = 50
lumped_mass_inertia = np.zeros((n_lumped_mass, 3, 3))
lumped_mass_position = np.zeros((n_lumped_mass, 3))
lumped_mass_position[0] = offset_fuselage_wing

# aero
chord_main = 1.0

# DISCRETISATION
# spatial discretisation
# chordiwse panels
m = 4
# spanwise elements
n_elem_multiplier = 2
n_elem_main = int(2*n_elem_multiplier) #int(4*n_elem_multiplier)
n_elem_fuselage = 21
n_surfaces = 2

# temporal discretisation
physical_time = 1
tstep_factor = 1.
dt = 1.0/m/u_inf*tstep_factor
n_tstep = round(physical_time/dt)

# END OF INPUT-----------------------------------------------------------------

# beam processing
n_node_elem = 3

# total number of elements
n_elem = 0
n_elem += n_elem_main + n_elem_main
n_elem += n_elem_fuselage

# number of nodes per part
n_node_main = n_elem_main*(n_node_elem - 1) + 1
n_node_fuselage = n_elem_fuselage*(n_node_elem - 1) + 1

# total number of nodes
n_node = 0
n_node += n_node_main + n_node_main - 1
n_node += n_node_fuselage - 1

# stiffness and mass matrices
n_stiffness = 2
base_stiffness_main = sigma*np.diag([ea, ga, ga, gj, eiy, eiz])
base_stiffness_fuselage = base_stiffness_main.copy()*sigma_fuselage
base_stiffness_fuselage[4, 4] = base_stiffness_fuselage[5, 5]

n_mass = 2
base_mass_main = np.diag([m_bar_main, m_bar_main, m_bar_main, j_bar_main, 0.5*j_bar_main, 0.5*j_bar_main])
base_mass_fuselage = np.diag([m_bar_fuselage,
                              m_bar_fuselage,
                              m_bar_fuselage,
                              j_bar_fuselage,
                              j_bar_fuselage*0.5,
                              j_bar_fuselage*0.5])

# PLACEHOLDERS
# beam
x = np.zeros((n_node, ))
y = np.zeros((n_node, ))
z = np.zeros((n_node, ))
beam_number = np.zeros((n_elem, ), dtype=int)
frame_of_reference_delta = np.zeros((n_elem, n_node_elem, 3))
structural_twist = np.zeros((n_elem, 3))
conn = np.zeros((n_elem, n_node_elem), dtype=int)
stiffness = np.zeros((n_stiffness, 6, 6))
elem_stiffness = np.zeros((n_elem, ), dtype=int)
mass = np.zeros((n_mass, 6, 6))
elem_mass = np.zeros((n_elem, ), dtype=int)
boundary_conditions = np.zeros((n_node, ), dtype=int)
app_forces = np.zeros((n_node, 6))


# aero
airfoil_distribution = np.zeros((n_elem, n_node_elem), dtype=int)
surface_distribution = np.zeros((n_elem,), dtype=int) - 1
surface_m = np.zeros((n_surfaces, ), dtype=int)
m_distribution = 'uniform'
aero_node = np.zeros((n_node,), dtype=bool)
nonlifting_body_node = np.zeros((n_node,), dtype=bool)
twist = np.zeros((n_elem, n_node_elem))
sweep = np.zeros((n_elem, n_node_elem))
chord = np.zeros((n_elem, n_node_elem,))
elastic_axis = np.zeros((n_elem, n_node_elem,))
boundary_conditions_aero = np.zeros((n_node, ), dtype=int)

# nonlifting body
nonlifting_body_distribution = np.zeros((n_elem,), dtype=int) - 1
nonlifting_body_m = np.zeros((n_nonlifting_bodies, ), dtype=int)
radius = np.zeros((n_node,))


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


def find_index_of_closest_entry(array_values, target_value):
    return (np.abs(array_values - target_value)).argmin()



def generate_fem():
    stiffness[0, ...] = base_stiffness_main
    stiffness[1, ...] = base_stiffness_fuselage

    mass[0, ...] = base_mass_main
    mass[1, ...] = base_mass_fuselage

    we = 0
    wn = 0

    # inner right wing
    beam_number[we:we + n_elem_main] = 0
    x[wn:wn + n_node_main] = offset_fuselage_wing
    y[wn:wn + n_node_main] = np.linspace(0, span_main, n_node_main)
    y[wn:wn + n_node_main] += diameter_fuselage
    for ielem in range(n_elem_main):
        conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) +
                               [0, 2, 1])
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]

    elem_stiffness[we:we + n_elem_main] = 0
    elem_mass[we:we + n_elem_main] = 0
    boundary_conditions[wn] = 1
    boundary_conditions[wn + n_elem_main] = -1
    we += n_elem_main
    wn += n_node_main

    # inner left wing
    beam_number[we:we + n_elem_main] = 1
    x[wn:wn + n_node_main] = offset_fuselage_wing
    y[wn:wn + n_node_main] = np.linspace(0, -span_main, n_node_main)
    y[wn:wn + n_node_main] -= radius_fuselage
    for ielem in range(n_elem_main):
        conn[we + ielem, :] = ((np.ones((3, ))*(we+ielem)*(n_node_elem - 1)) +
                               1 + [0, 2, 1])
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
    elem_stiffness[we:we + n_elem_main] = 0
    elem_mass[we:we + n_elem_main] = 0
    boundary_conditions[wn] = 1
    boundary_conditions[wn + n_elem_main] = -1

    we += n_elem_main
    wn += n_node_main

    # fuselage
    beam_number[we:we + n_elem_fuselage] = 2
    x[wn:wn + n_node_fuselage] = np.linspace(0.0, length_fuselage, n_node_fuselage-2)
    z[wn:wn + n_node_fuselage] = np.linspace(0.0, offset_fuselage_vertical, n_node_fuselage-2)

    # adjust node closes to fuselage wing junction to be in the same z-y-plane than wing nodes
    idx_fuselage_wing_junction = find_index_of_closest_entry(x[wn:wn + n_node_fuselage - 3], offset_fuselage_wing)
    z[idx_fuselage_wing_junction] = np.interp(offset_fuselage_wing, x[wn:wn + n_node_fuselage - 3], z[wn:wn + n_node_fuselage - 3])
    x[idx_fuselage_wing_junction] = offset_fuselage_wing

    for ielem in range(n_elem_fuselage-1):
        conn[we + ielem, :] = ((np.ones((3,))*(we + ielem)*(n_node_elem - 1)) +
                               2 + [0, 2, 1])
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [0.0, 1.0, 0.0]
    conn[we+n_elem_fuselage-1,:] = np.array([conn[0,0],conn[n_elem_main,0],idx_fuselage_wing_junction])
    for inode in range(n_node_elem):
        # TO-DO: Correct reference frame for wing junction beam
        frame_of_reference_delta[we+n_elem_fuselage-1, inode, :] = [0.0, 1.0, 0.0]
    elem_stiffness[we:we + n_elem_fuselage] = 1
    elem_mass[we:we + n_elem_fuselage] = 1

    boundary_conditions[wn] = -1
    boundary_conditions[idx_fuselage_wing_junction] = 1
    boundary_conditions[wn + n_elem_main] = -1

    we += n_elem_fuselage
    wn += n_node_fuselage

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

def generate_aero_file():
    global x, y, z

    we = 0
    wn = 0
    # right wing (surface 0, beam 0)
    i_surf = 0
    airfoil_distribution[we:we + n_elem_main, :] = 0
    surface_distribution[we:we + n_elem_main] = i_surf
    surface_m[i_surf] = m
    aero_node[wn:wn + n_node_main] = True
    temp_chord = np.linspace(chord_main, chord_main, n_node_main)
    temp_sweep = np.linspace(0.0, 0*np.pi/180, n_node_main)
    node_counter = 0
    for i_elem in range(we, we + n_elem_main):
        for i_local_node in range(n_node_elem):
            if not i_local_node == 0:
                node_counter += 1
            chord[i_elem, i_local_node] = temp_chord[node_counter]
            elastic_axis[i_elem, i_local_node] = ea_main
            sweep[i_elem, i_local_node] = temp_sweep[node_counter]

    we += n_elem_main
    wn += n_node_main

    # left wing (surface 1, beam 1)
    i_surf = 1
    airfoil_distribution[we:we + n_elem_main, :] = 0
    surface_distribution[we:we + n_elem_main] = i_surf
    surface_m[i_surf] = m
    aero_node[wn:wn + n_node_main] = True
    temp_chord = np.linspace(chord_main, chord_main, n_node_main)
    node_counter = 0
    for i_elem in range(we, we + n_elem_main):
        for i_local_node in range(n_node_elem):
            if not i_local_node == 0:
                node_counter += 1
            chord[i_elem, i_local_node] = temp_chord[node_counter]
            elastic_axis[i_elem, i_local_node] = ea_main
            sweep[i_elem, i_local_node] = -temp_sweep[node_counter]

    we += n_elem_main
    wn += n_node_main - 1

    # fuselage
    we += n_elem_fuselage
    wn += n_node_fuselage - 1

    with h5.File(route + '/' + case_name + '.aero.h5', 'a') as h5file:
        airfoils_group = h5file.create_group('airfoils')
        # add one airfoil
        naca_airfoil_main = airfoils_group.create_dataset('0', data=np.column_stack(
            generate_naca_camber(P=0, M=0)))

        # chord
        chord_input = h5file.create_dataset('chord', data=chord)
        dim_attr = chord_input .attrs['units'] = 'm'

        # twist
        twist_input = h5file.create_dataset('twist', data=twist)
        dim_attr = twist_input.attrs['units'] = 'rad'

        # sweep
        sweep_input = h5file.create_dataset('sweep', data=sweep)
        dim_attr = sweep_input.attrs['units'] = 'rad'

        # airfoil distribution
        airfoil_distribution_input = h5file.create_dataset('airfoil_distribution', data=airfoil_distribution)

        surface_distribution_input = h5file.create_dataset('surface_distribution', data=surface_distribution)
        surface_m_input = h5file.create_dataset('surface_m', data=surface_m)
        m_distribution_input = h5file.create_dataset('m_distribution', data=m_distribution.encode('ascii', 'ignore'))

        aero_node_input = h5file.create_dataset('aero_node', data=aero_node)
        elastic_axis_input = h5file.create_dataset('elastic_axis', data=elastic_axis)


def generate_nonlifting_body_file():
    we = 0
    wn = 0

    # right wing
    nonlifting_body_node[wn:wn + n_node_main] = False
    we += n_elem_main
    wn += n_node_main

    # left wing
    nonlifting_body_node[wn:wn + n_node_main] = False
    we += n_elem_main
    wn += n_node_main

    #fuselage (beam?, body ID = 0)
    i_body = 0
    nonlifting_body_node[wn:wn + n_node_fuselage] = True
    nonlifting_body_distribution[wn:wn + n_node_fuselage] = i_body
    nonlifting_body_m[i_body] = m_radial_elem_fuselage
    radius[wn:wn + n_node_fuselage] = create_fuselage_geometry()

    with h5.File(route + '/' + case_name + '.nonlifting_body.h5', 'a') as h5file:
        nonlifting_body_m_input = h5file.create_dataset('nonlifting_body_m', data=nonlifting_body_m)
        nonlifting_body_node_input = h5file.create_dataset('nonlifting_body_node', data=nonlifting_body_node)

        nonlifting_body_distribution_input = h5file.create_dataset('nonlifting_body_distribution', data=nonlifting_body_distribution)

        # radius
        radius_input = h5file.create_dataset('radius', data=radius)
        dim_attr = radius_input.attrs['units'] = 'm'

    # right wing (surface 0, beam 0)
def generate_naca_camber(M=0, P=0):
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

def find_index_of_closest_entry(array_values, target_value):
    return (np.abs(array_values - target_value)).argmin()

def create_ellipsoid(x_geom, a, b, flip):
    x_geom -= x_geom.min()
    y = b*np.sqrt(1-(x_geom/a)**2)
    if flip:
        y = np.flip(y.tolist())
    return y

def add_nose_or_tail_shape(idx, array_x, nose = True):
    if nose:
        shape = create_ellipsoid(array_x[:idx], array_x[idx] - array_x[0], radius_fuselage, True)
    if not nose:
        #TO-DO: Add paraboloid shaped tail
        shape = create_ellipsoid(array_x[idx:], array_x[-1]-array_x[idx], radius_fuselage, False)
    return shape

def create_fuselage_geometry():
    array_radius = np.zeros((sum(nonlifting_body_node)))
    x_fuselage = x[nonlifting_body_node]
    fuselage_length = max(x_fuselage)-min(x_fuselage) # useful??
    idx_cylinder_start = find_index_of_closest_entry(x_fuselage, list_cylinder_position_fuselage[0]*fuselage_length)
    idx_cylinder_end = find_index_of_closest_entry(x_fuselage,list_cylinder_position_fuselage[1]*fuselage_length)
    # set constant radius of cylinder
    array_radius[idx_cylinder_start:idx_cylinder_end] = radius_fuselage
    # set r(x) for nose and tail region
    array_radius[:idx_cylinder_start] = add_nose_or_tail_shape(idx_cylinder_start, x_fuselage, nose = True)
    array_radius[idx_cylinder_end:] = add_nose_or_tail_shape(idx_cylinder_end, x_fuselage, nose = False)
     # ensure radius = 0 at nose/tail
    array_radius[0] = 0
    array_radius[-1] = 0
    return array_radius


clean_test_files()
generate_fem()
generate_aero_file()
generate_nonlifting_body_file()