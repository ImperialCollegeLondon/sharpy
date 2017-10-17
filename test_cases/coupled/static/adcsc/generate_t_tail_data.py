import h5py as h5
import numpy as np
import configparser
import os

def skew(vector):
    matrix = np.zeros((3, 3))
    matrix[1, 2] = -vector[0]
    matrix[2, 0] = -vector[1]
    matrix[0, 1] = -vector[2]
    matrix[2, 1] = vector[0]
    matrix[0, 2] = vector[1]
    matrix[1, 0] = vector[2]
    return matrix

def rot_matrix_around_axis(axis, angle):
    axis = axis/np.linalg.norm(axis)
    rot = np.cos(angle)*np.eye(3)
    rot += np.sin(angle)*skew(axis)
    rot += (1 - np.cos(angle))*np.outer(axis, axis)
    return rot

case_name = 'adcsc'
route = os.path.dirname(os.path.realpath(__file__)) + '/'
n_time_steps = 1000
dt = 0.01
A = 30
period = 5
# flight conditions
u_inf = 10 # m/s
rho = 1.225 
alpha = 5
beta = 0
c_ref = 1 # not used
b_ref = 16 # not used
bay_l = 0.37 # bay lenght in m
nbay = 22 # number of bays
dihedral = 20*np.pi/180.

alpha_rad = alpha*np.pi/180

# main geometry data
main_span = nbay*bay_l # half
main_chord = 1.8
main_ea = 0.3 # elastic axys position in chord units
main_sigma = 1 # stiffness mulitplier
main_airfoil_P = 0 # NACA parameter
main_airfoil_M = 0 # NACA parameter

fuselage_length = 6.6 
fuselage_sigma = 100 # stiffness mulitplier
fuselage_mass_sigma = 0.1 # multiplier of mass repect the wing

tail_span = bay_l*4 # half
tail_chord = 0.75
tail_ea = 0.25 # elastic axys position in chord units 
tail_sigma = 1 # stiffness mulitplier 
tail_mass_sigma = 0.1 # multiplier of mass repect the wing
tail_airfoil_P = 5 # NACA parameter
tail_airfoil_M = 5 # NACA parameter
tail_twist = 0*np.pi/180 # aoa of tail at plane aoa=0 (to be implemented)

fin_span = bay_l*2 # vertical lenght of stabiliser
fin_chord = 0.75 # chord of stabiliser
fin_ea = 0.25 # elastic axys position in chord units
fin_sigma = 1 # stiffness mulitplier 
fin_mass_sigma = 0.1 # multiplier of mass repect the wing 
fin_airfoil_P = 0 # NACA parameter
fin_airfoil_M = 0 # NACA parameter

n_surfaces = 5
force = 0 # not used
momenty = 0 # not used
momentx = 0 # not used

# discretisation data
num_elem_main = 11 # main wing
num_elem_tail = 2
num_elem_fin = 2
num_elem_fuselage = 2

# data for different wing sections
section1_nelem = 3
section1_span = bay_l*(section1_nelem*2)
section1_chord = 1.8
section2_nelem = 2
section2_span = bay_l*(section2_nelem*2)
section3_nelem = 4
section3_span = bay_l*(section3_nelem*2)
section3_chord = 1.0
ogive1_nelem = 1
ogive1_span = 0.5
ogive1_chord = 1.0
ogive1_dihedral = -10*np.pi/180
ogive1_sweep = -5*np.pi/180
ogive2_nelem = 1
ogive2_span = 0.25
ogive2_chord = 0.5
ogive2_dihedral = -30*np.pi/180
ogive2_sweep = -10*np.pi/180
ogive3_nelem = 1
ogive3_span = 0.125
ogive3_chord = 0.25
ogive3_dihedral = -40*np.pi/180
ogive3_sweep = -20*np.pi/180

num_node_elem = 3
num_elem = section1_nelem + section2_nelem + section3_nelem + ogive1_nelem + ogive2_nelem + ogive3_nelem + ogive1_nelem + ogive2_nelem + ogive3_nelem  + section3_nelem + section2_nelem + section1_nelem + num_elem_fuselage + num_elem_fin + num_elem_tail + num_elem_tail
num_node_main = num_elem_main*(num_node_elem - 1) + 1
num_node_fuselage = num_elem_fuselage*(num_node_elem - 1) + 1
num_node_tail = num_elem_tail*(num_node_elem - 1) + 1
num_node_fin = num_elem_fin*(num_node_elem - 1) + 1

num_node = section1_nelem*2 + 1 + section2_nelem*2 + section3_nelem*2 + ogive1_nelem*2 + ogive2_nelem*2 + ogive3_nelem*2 + ogive1_nelem*2 + ogive2_nelem*2 + ogive3_nelem*2 + section3_nelem*2 + 1 + section2_nelem*2 + section1_nelem*2 - 1 + num_elem_fuselage*2 + num_elem_fin*2 + num_elem_tail*2 + num_elem_tail*2

m_main = 4 # number of pannel chordwise direction
m_tail = 4
m_fin = 4


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

    solver_file_name = route + '/' + case_name + '.solver.txt'
    if os.path.isfile(solver_file_name):
        os.remove(solver_file_name)

    flightcon_file_name = route + '/' + case_name + '.flightcon.txt'
    if os.path.isfile(flightcon_file_name):
        os.remove(flightcon_file_name)


def generate_fem_file():
    # placeholders
    # coordinates
    global x, y, z
    x = np.zeros((num_node, ))
    y = np.zeros((num_node, ))
    z = np.zeros((num_node, ))
    # struct twist
    structural_twist = np.zeros_like(x)
    # beam number
    beam_number = np.zeros((num_elem, ), dtype=int)
    # frame of reference delta
    frame_of_reference_delta = np.zeros((num_elem, num_node_elem, 3))
    # connectivities
    conn = np.zeros((num_elem, num_node_elem), dtype=int)
    # stiffness
    num_stiffness = 4
    ea = 1e6  
    ga = 1e6 
    gj = 1e4 # torsion SI
    eiy = 2e4 # bending y-dir
    eiz = 5e6 # bending z-dir
    sigma = 0.3 # multiplier for stiffness of all aricraft
    base_stiffness = sigma*np.diag([ea, ga, ga, gj, eiy, eiz])
    stiffness = np.zeros((num_stiffness, 6, 6))
    stiffness[0, :, :] = main_sigma*base_stiffness
    stiffness[1, :, :] = fuselage_sigma*base_stiffness
    stiffness[2, :, :] = tail_sigma*base_stiffness
    stiffness[3, :, :] = fin_sigma*base_stiffness
    elem_stiffness = np.zeros((num_elem,), dtype=int)
    # mass
    num_mass = 4 # number of different masses for beams
    m_base = 0.75 # mass per unit lenght
    j_base = 0.1 # inertia
    base_mass = np.diag([m_base, m_base, m_base, j_base, j_base, j_base]) # mass matrix
    mass = np.zeros((num_mass, 6, 6))
    mass[0, :, :] = main_sigma*base_mass
    mass[1, :, :] = fuselage_mass_sigma*base_mass
    mass[2, :, :] = tail_mass_sigma*base_mass
    mass[3, :, :] = fin_mass_sigma*base_mass
    elem_mass = np.zeros((num_elem,), dtype=int)
    # boundary conditions
    boundary_conditions = np.zeros((num_node, ), dtype=int)
    boundary_conditions[0] = 1
    # applied forces (in case you want to add thrust)
    n_app_forces = 0
    node_app_forces = np.zeros((n_app_forces,), dtype=int)
    app_forces = np.zeros((n_app_forces, 6))

    # lumped masses (given in beam FOR)
    n_lumped_mass = 0
    lumped_mass_nodes = np.array([], dtype=int)
    lumped_mass = np.zeros((n_lumped_mass, ))
    lumped_mass_inertia = np.zeros((n_lumped_mass, 3, 3))
    lumped_mass_position = np.zeros((n_lumped_mass, 3))

    # right wing (beam 0) --------------------------------------------------------------
    working_elem = 0
    working_node = 0
    beam_number[working_elem:working_elem + section1_nelem] = 0
    y[working_node:working_node + section1_nelem*2+1] = np.linspace(0.0, section1_span, section1_nelem*2+1)
    for ielem in range(section1_nelem):
        for inode in range(num_node_elem):
            frame_of_reference_delta[working_elem + ielem, inode, :] = [-1, 0, 0]
    # connectivity
    for ielem in range(section1_nelem):
        conn[working_elem + ielem, :] = ((np.ones((3,))*(working_elem + ielem)*(num_node_elem - 1)) +
                                         [0, 2, 1])
    elem_stiffness[working_elem:working_elem + section1_nelem] = 0
    elem_mass[working_elem:working_elem + section1_nelem] = 0
    boundary_conditions[0] = 1 # clamped node
    working_elem += section1_nelem
    working_node += section1_nelem*2+1

    # right wing (beam 1) --------------------------------------------------------------
    beam_number[working_elem:working_elem + section2_nelem] = 1
    y[working_node:working_node + section2_nelem*2] = y[working_node-1] + np.linspace(0.0, section2_span, section2_nelem*2+1)[1:]
    for ielem in range(section2_nelem):
        for inode in range(num_node_elem):
            frame_of_reference_delta[working_elem + ielem, inode, :] = [-1, 0, 0]
    # connectivity
    for ielem in range(section2_nelem):
        conn[working_elem + ielem, :] = ((np.ones((3,))*(working_elem + ielem)*(num_node_elem - 1)) +
                                         [0, 2, 1])
    elem_stiffness[working_elem:working_elem + section2_nelem] = 0
    elem_mass[working_elem:working_elem + section2_nelem] = 0
    working_elem += section2_nelem
    working_node += section2_nelem*2

    # right wing (beam 2) --------------------------------------------------------------
    beam_number[working_elem:working_elem + section3_nelem] = 2
    y[working_node:working_node + section3_nelem*2] = y[working_node-1] + np.linspace(0.0, section3_span*np.cos(dihedral), section3_nelem*2+1)[1:]
    z[working_node:working_node + section3_nelem*2] = z[working_node-1] + np.linspace(0.0, section3_span*np.sin(dihedral), section3_nelem*2+1)[1:]
    no_ogive_coords = np.array([x[working_node + section3_nelem*2 - 1], y[working_node + section3_nelem*2 - 1], z[working_node + section3_nelem*2 - 1]])
    for ielem in range(section3_nelem):
        for inode in range(num_node_elem):
            frame_of_reference_delta[working_elem + ielem, inode, :] = [-1, 0, 0]
    # connectivity
    for ielem in range(section3_nelem):
        conn[working_elem + ielem, :] = ((np.ones((3,))*(working_elem + ielem)*(num_node_elem - 1)) +
                                         [0, 2, 1])
    elem_stiffness[working_elem:working_elem + section3_nelem] = 0
    elem_mass[working_elem:working_elem + section3_nelem] = 0
    working_elem += section3_nelem
    working_node += section3_nelem*2

    # right ogive (beam 3) --------------------------------------------------------------
    beam_number[working_elem:working_elem + ogive1_nelem] = 3
    tmp = np.zeros((ogive1_nelem*2,3))
    tmp[:, 1] = np.linspace(0.0, ogive1_span, ogive1_nelem*2+1)[1:]
    rot = rot_matrix_around_axis(np.array([0, 0, 1]), ogive1_sweep)
    rot = np.dot(rot_matrix_around_axis(np.array([1, 0, 0]), ogive1_dihedral), rot)
    for inode in range(ogive1_nelem*2):
        tmp_vec = np.dot(rot, tmp[inode, :])
        x[working_node + inode] = x[working_node-1] + tmp_vec[0]                
        y[working_node + inode] = y[working_node-1] + tmp_vec[1]                
        z[working_node + inode] = z[working_node-1] + tmp_vec[2]                
    for ielem in range(ogive1_nelem):
        rot = rot_matrix_around_axis(np.array([0, 0, 1]), ogive1_sweep)
        rot = rot_matrix_around_axis(np.array([0, 0, 1]), 0)
        for inode in range(num_node_elem):
            frame_of_reference_delta[working_elem + ielem, inode, :] = np.dot(rot, np.array([-1, 0, 0]))
    # connectivity
    for ielem in range(ogive1_nelem):
        conn[working_elem + ielem, :] = ((np.ones((3,))*(working_elem + ielem)*(num_node_elem - 1)) +
                                         [0, 2, 1])
    elem_stiffness[working_elem:working_elem + ogive1_nelem] = 0
    elem_mass[working_elem:working_elem + ogive1_nelem] = 0
    working_elem += ogive1_nelem
    working_node += ogive1_nelem*2

    # right ogive (beam 4) --------------------------------------------------------------
    beam_number[working_elem:working_elem + ogive2_nelem] = 4
    tmp = np.zeros((ogive2_nelem*2,3))
    tmp[:, 1] = np.linspace(0.0, ogive2_span, ogive2_nelem*2+1)[1:]
    rot = rot_matrix_around_axis(np.array([0, 0, 1]), ogive2_sweep)
    rot = np.dot(rot_matrix_around_axis(np.array([1, 0, 0]), ogive2_dihedral), rot)
    for inode in range(ogive2_nelem*2):
        tmp_vec = np.dot(rot, tmp[inode, :])
        x[working_node + inode] = x[working_node-1] + tmp_vec[0]                
        y[working_node + inode] = y[working_node-1] + tmp_vec[1]                
        z[working_node + inode] = z[working_node-1] + tmp_vec[2]                
    for ielem in range(ogive2_nelem):
        rot = rot_matrix_around_axis(np.array([0, 0, 1]), ogive2_sweep)
        rot = rot_matrix_around_axis(np.array([0, 0, 1]), 0)
        for inode in range(num_node_elem):
            frame_of_reference_delta[working_elem + ielem, inode, :] = np.dot(rot, np.array([-1, 0, 0]))
    # connectivity
    for ielem in range(ogive2_nelem):
        conn[working_elem + ielem, :] = ((np.ones((3,))*(working_elem + ielem)*(num_node_elem - 1)) +
                                         [0, 2, 1])
    elem_stiffness[working_elem:working_elem + ogive2_nelem] = 0
    elem_mass[working_elem:working_elem + ogive2_nelem] = 0
    working_elem += ogive2_nelem
    working_node += ogive2_nelem*2

    # right ogive (beam 5) --------------------------------------------------------------
    beam_number[working_elem:working_elem + ogive3_nelem] = 5
    tmp = np.zeros((ogive3_nelem*2,3))
    tmp[:, 1] = np.linspace(0.0, ogive3_span, ogive3_nelem*2+1)[1:]
    rot = rot_matrix_around_axis(np.array([0, 0, 1]), ogive3_sweep)
    rot = np.dot(rot_matrix_around_axis(np.array([1, 0, 0]), ogive3_dihedral), rot)
    for inode in range(ogive3_nelem*2):
        tmp_vec = np.dot(rot, tmp[inode, :])
        x[working_node + inode] = x[working_node-1] + tmp_vec[0]                
        y[working_node + inode] = y[working_node-1] + tmp_vec[1]                
        z[working_node + inode] = z[working_node-1] + tmp_vec[2]                
    for ielem in range(ogive3_nelem):
        rot = rot_matrix_around_axis(np.array([0, 0, 1]), ogive2_sweep)
        rot = rot_matrix_around_axis(np.array([0, 0, 1]), 0)
        for inode in range(num_node_elem):
            frame_of_reference_delta[working_elem + ielem, inode, :] = np.dot(rot, np.array([-1, 0, 0]))
    # connectivity
    for ielem in range(ogive2_nelem):
        conn[working_elem + ielem, :] = ((np.ones((3,))*(working_elem + ielem)*(num_node_elem - 1)) +
                                         [0, 2, 1])
    elem_stiffness[working_elem:working_elem + ogive3_nelem] = 0
    elem_mass[working_elem:working_elem + ogive3_nelem] = 0
    boundary_conditions[working_node + ogive3_nelem*2 - 1] = -1 # free node (-1 in the index because you start from 0 ofc)
    working_elem += ogive3_nelem
    working_node += ogive3_nelem*2

    # left ogive (beam 6) --------------------------------------------------------------
    beam_number[working_elem:working_elem + ogive3_nelem] = 6
    tmp = np.zeros((ogive3_nelem*2+1,3))
    tmp[:, 1] = np.linspace(0.0, ogive3_span, ogive3_nelem*2+1)
    rot = rot_matrix_around_axis(np.array([0, 0, 1]), ogive3_sweep)
    rot = np.dot(rot_matrix_around_axis(np.array([1, 0, 0]), ogive3_dihedral), rot).transpose()
    for inode in range(ogive3_nelem*2+1):
        tmp_vec = np.dot(rot, tmp[inode, :])
        x[working_node + inode] = x[working_node-1] + tmp_vec[0]                
        y[working_node + inode] = -y[working_node-1] + tmp_vec[1]                
        z[working_node + inode] = z[working_node-1] + tmp_vec[2]                
    for ielem in range(ogive3_nelem):
        rot = rot_matrix_around_axis(np.array([0, 0, 1]), -ogive3_sweep)
        rot = rot_matrix_around_axis(np.array([0, 0, 1]), 0)
        for inode in range(num_node_elem):
            frame_of_reference_delta[working_elem + ielem, inode, :] = np.dot(rot, np.array([-1, 0, 0]))
    # connectivity
    for ielem in range(ogive3_nelem):
        conn[working_elem + ielem, :] = ((np.ones((3,))*(working_elem + ielem)*(num_node_elem - 1)) +
                                         [0, 2, 1] + 1)
    elem_stiffness[working_elem:working_elem + ogive3_nelem] = 0
    elem_mass[working_elem:working_elem + ogive3_nelem] = 0
    boundary_conditions[working_node] = -1 # free node 
    working_elem += ogive3_nelem
    working_node += ogive3_nelem*2 + 1

    # left ogive (beam 7) --------------------------------------------------------------
    beam_number[working_elem:working_elem + ogive2_nelem] = 7
    tmp = np.zeros((ogive2_nelem*2,3))
    tmp[:, 1] = np.linspace(0.0, ogive2_span, ogive2_nelem*2+1)[1:]
    rot = rot_matrix_around_axis(np.array([0, 0, 1]), ogive2_sweep)
    rot = np.dot(rot_matrix_around_axis(np.array([1, 0, 0]), ogive2_dihedral), rot).transpose()
    for inode in range(ogive2_nelem*2):
        tmp_vec = np.dot(rot, tmp[inode, :])
        x[working_node + inode] = x[working_node-1] + tmp_vec[0]                
        y[working_node + inode] = y[working_node-1] + tmp_vec[1]                
        z[working_node + inode] = z[working_node-1] + tmp_vec[2]                
    for ielem in range(ogive2_nelem):
        rot = rot_matrix_around_axis(np.array([0, 0, 1]), -ogive2_sweep)
        rot = rot_matrix_around_axis(np.array([0, 0, 1]), 0)
        for inode in range(num_node_elem):
            frame_of_reference_delta[working_elem + ielem, inode, :] = np.dot(rot, np.array([-1, 0, 0]))
    # connectivity
    for ielem in range(ogive2_nelem):
        conn[working_elem + ielem, :] = ((np.ones((3,))*(working_elem + ielem)*(num_node_elem - 1)) +
                                         [0, 2, 1] + 1)
    elem_stiffness[working_elem:working_elem + ogive2_nelem] = 0
    elem_mass[working_elem:working_elem + ogive2_nelem] = 0
    working_elem += ogive2_nelem
    working_node += ogive2_nelem*2

    # left ogive (beam 8) --------------------------------------------------------------
    beam_number[working_elem:working_elem + ogive1_nelem] = 8
    tmp = np.zeros((ogive1_nelem*2,3))
    tmp[:, 1] = np.linspace(0.0, ogive1_span, ogive1_nelem*2+1)[1:]
    rot = rot_matrix_around_axis(np.array([0, 0, 1]), ogive1_sweep)
    rot = np.dot(rot_matrix_around_axis(np.array([1, 0, 0]), ogive1_dihedral), rot).transpose()
    for inode in range(ogive1_nelem*2):
        tmp_vec = np.dot(rot, tmp[inode, :])
        x[working_node + inode] = x[working_node-1] + tmp_vec[0]                
        y[working_node + inode] = y[working_node-1] + tmp_vec[1]                
        z[working_node + inode] = z[working_node-1] + tmp_vec[2]                
    for ielem in range(ogive1_nelem):
        rot = rot_matrix_around_axis(np.array([0, 0, 1]), -ogive1_sweep)
        rot = rot_matrix_around_axis(np.array([0, 0, 1]), 0)
        for inode in range(num_node_elem):
            frame_of_reference_delta[working_elem + ielem, inode, :] = np.dot(rot, np.array([-1, 0, 0]))
    # connectivity
    for ielem in range(ogive1_nelem):
        conn[working_elem + ielem, :] = ((np.ones((3,))*(working_elem + ielem)*(num_node_elem - 1)) +
                                         [0, 2, 1] + 1)
    elem_stiffness[working_elem:working_elem + ogive1_nelem] = 0
    elem_mass[working_elem:working_elem + ogive1_nelem] = 0
    working_elem += ogive1_nelem
    working_node += ogive1_nelem*2

    # left wing (beam 9) --------------------------------------------------------------
    beam_number[working_elem:working_elem + section3_nelem] = 9
    y[working_node:working_node + section3_nelem*2] = -no_ogive_coords[1] + np.linspace(0.0, section3_span*np.cos(dihedral), section3_nelem*2+1)[1:]
    z[working_node:working_node + section3_nelem*2] =  no_ogive_coords[2] - np.linspace(0.0, section3_span*np.sin(dihedral), section3_nelem*2+1)[1:]
    for ielem in range(section3_nelem):
        for inode in range(num_node_elem):
            frame_of_reference_delta[working_elem + ielem, inode, :] = [-1, 0, 0]
    # connectivity
    for ielem in range(section3_nelem):
        conn[working_elem + ielem, :] = ((np.ones((3,))*(working_elem + ielem)*(num_node_elem - 1)) +
                                         [0, 2, 1] + 1)
    elem_stiffness[working_elem:working_elem + section3_nelem] = 0
    elem_mass[working_elem:working_elem + section3_nelem] = 0
    working_elem += section3_nelem
    working_node += section3_nelem*2

    # left wing (beam 10) --------------------------------------------------------------
    beam_number[working_elem:working_elem + section2_nelem] = 10
    y[working_node:working_node + section2_nelem*2] = y[working_node-1] + np.linspace(0.0, section2_span, section2_nelem*2+1)[1:]
    for ielem in range(section2_nelem):
        for inode in range(num_node_elem):
            frame_of_reference_delta[working_elem + ielem, inode, :] = [-1, 0, 0]
    # connectivity
    for ielem in range(section2_nelem):
        conn[working_elem + ielem, :] = ((np.ones((3,))*(working_elem + ielem)*(num_node_elem - 1)) +
                                         [0, 2, 1] + 1)
    elem_stiffness[working_elem:working_elem + section2_nelem] = 0
    elem_mass[working_elem:working_elem + section2_nelem] = 0
    working_elem += section2_nelem
    working_node += section2_nelem*2

    # left wing (beam 11) --------------------------------------------------------------
    beam_number[working_elem:working_elem + section1_nelem] = 11
    y[working_node:working_node + section1_nelem*2-1] = y[working_node-1] + np.linspace(0.0, section1_span, section1_nelem*2+1)[1:-1]
    for ielem in range(section1_nelem):
        for inode in range(num_node_elem):
            frame_of_reference_delta[working_elem + ielem, inode, :] = [-1, 0, 0]
    # connectivity
    for ielem in range(section1_nelem):
        conn[working_elem + ielem, :] = ((np.ones((3,))*(working_elem + ielem)*(num_node_elem - 1)) +
                                         [0, 2, 1] + 1)
    conn[working_elem + section1_nelem - 1, 1] = 0 # because last element node is node 0
    elem_stiffness[working_elem:working_elem + section1_nelem] = 0
    elem_mass[working_elem:working_elem + section1_nelem] = 0
    boundary_conditions[0] = 1 # clamped node
    working_elem += section1_nelem
    working_node += section1_nelem*2-1

    # fuselage (beam 12) --------------------------------------------------------------
    beam_number[working_elem:working_elem + num_elem_fuselage] = 12
    tempx = np.linspace(0.0, fuselage_length, num_node_fuselage)
    x[working_node:working_node + num_node_fuselage - 1] = tempx[1:]
    for ielem in range(num_elem_fuselage):
        for inode in range(num_node_elem):
            frame_of_reference_delta[working_elem + ielem, inode, :] = [0, 1, 0]
    # connectivity
    for ielem in range(num_elem_fuselage):
        conn[working_elem + ielem, :] = ((np.ones((3,))*(working_elem + ielem)*(num_node_elem - 1)) +
                                         [0, 2, 1])
    conn[working_elem, 0] = 0
    elem_stiffness[working_elem:working_elem + num_elem_fuselage] = 1
    elem_mass[working_elem:working_elem + num_elem_fuselage] = 1
    working_elem += num_elem_fuselage
    working_node += num_node_fuselage - 1
    global end_of_fuselage_node
    end_of_fuselage_node = working_node - 1

    # fin (beam 13) --------------------------------------------------------------
    beam_number[working_elem:working_elem + num_elem_fin] = 13
    tempz = np.linspace(0.0, fin_span, num_node_fin)
    x[working_node:working_node + num_node_fin - 1] = x[end_of_fuselage_node]
    z[working_node:working_node + num_node_fin - 1] = tempz[1:]
    for ielem in range(num_elem_fin):
        for inode in range(num_node_elem):
            frame_of_reference_delta[working_elem + ielem, inode, :] = [-1, 0, 0]
    # connectivity
    for ielem in range(num_elem_fin):
        conn[working_elem + ielem, :] = ((np.ones((3,))*(working_elem + ielem)*(num_node_elem - 1)) +
                                         [0, 2, 1])
    conn[working_elem, 0] = end_of_fuselage_node
    elem_stiffness[working_elem:working_elem + num_elem_fin] = 3
    elem_mass[working_elem:working_elem + num_elem_fin] = 3
    end_of_fin_node = working_node + num_node_fin - 2
    working_elem += num_elem_fin
    working_node += num_node_fin - 1

    # right tail (beam 14) --------------------------------------------------------------
    beam_number[working_elem:working_elem + num_elem_tail] = 14
    tempy = np.linspace(0.0, tail_span, num_node_tail)
    y[working_node:working_node + num_node_tail - 1] = tempy[1:]
    x[working_node:working_node + num_node_tail - 1] = x[end_of_fin_node]
    z[working_node:working_node + num_node_tail - 1] = z[end_of_fin_node]
    for ielem in range(num_elem_tail):
        for inode in range(num_node_elem):
            frame_of_reference_delta[working_elem + ielem, inode, :] = [-1, 0, 0]
    # connectivity
    for ielem in range(num_elem_tail):
        conn[working_elem + ielem, :] = ((np.ones((3,))*(working_elem + ielem)*(num_node_elem - 1)) +
                                         [0, 2, 1])
    conn[working_elem, 0] = conn[working_elem - 1, 1]
    elem_stiffness[working_elem:working_elem + num_elem_tail] = 2
    elem_mass[working_elem:working_elem + num_elem_fuselage] = 2
    boundary_conditions[working_node + num_node_tail - 1 - 1] = -1
    working_elem += num_elem_tail
    working_node += num_node_tail - 1

    # left tail (beam 15) --------------------------------------------------------------
    beam_number[working_elem:working_elem + num_elem_tail] = 15
    tempy = np.linspace(-tail_span, 0, num_node_tail)
    y[working_node:working_node + num_node_tail - 1] = tempy[:-1]
    x[working_node:working_node + num_node_tail - 1] = x[end_of_fin_node]
    z[working_node:working_node + num_node_tail - 1] = z[end_of_fin_node]
    for ielem in range(num_elem_tail):
        for inode in range(num_node_elem):
            frame_of_reference_delta[working_elem + ielem, inode, :] = [-1, 0, 0]
    # connectivity
    for ielem in range(num_elem_tail):
        conn[working_elem + ielem, :] = ((np.ones((3,))*(working_elem + ielem)*(num_node_elem - 1)) +
                                         [0, 2, 1]) + 1
    conn[working_elem + num_elem_tail - 1, 1] = end_of_fin_node
    elem_stiffness[working_elem:working_elem + num_elem_tail] = 2
    elem_mass[working_elem:working_elem + num_elem_fuselage] = 2
    boundary_conditions[working_node] = -1
    working_elem += num_elem_tail
    working_node += num_node_tail - 1

    with h5.File(route + '/' + case_name + '.fem.h5', 'a') as h5file:
        coordinates = h5file.create_dataset('coordinates', data=np.column_stack((x, y, z)))
        conectivities = h5file.create_dataset('connectivities', data=conn)
        num_nodes_elem_handle = h5file.create_dataset(
            'num_node_elem', data=num_node_elem)
        num_nodes_handle = h5file.create_dataset(
            'num_node', data=num_node)
        num_elem_handle = h5file.create_dataset(
            'num_elem', data=num_elem)
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
        node_app_forces_handle = h5file.create_dataset(
            'node_app_forces', data=node_app_forces)
        lumped_mass_nodes_handle = h5file.create_dataset(
            'lumped_mass_nodes', data=lumped_mass_nodes)
        lumped_mass_handle = h5file.create_dataset(
            'lumped_mass', data=lumped_mass)
        lumped_mass_inertia_handle = h5file.create_dataset(
            'lumped_mass_inertia', data=lumped_mass_inertia)
        lumped_mass_position_handle = h5file.create_dataset(
            'lumped_mass_position', data=lumped_mass_position)
        # orientation_handle = h5file.create_dataset(
        #     'orientation', data=inertial2aero)


def generate_aero_file():
    global x, y, z
    airfoil_distribution = np.zeros((num_node,), dtype=int)
    surface_distribution = np.zeros((num_elem,), dtype=int) - 1
    surface_m = np.zeros((n_surfaces, ), dtype=int)
    m_distribution = 'uniform' # distribution in chord direction
    aero_node = np.zeros((num_node,), dtype=bool)
    twist = np.zeros((num_node,))
    chord = np.zeros((num_node,))
    elastic_axis = np.zeros((num_node,))

    working_elem = 0
    working_node = 0

    # right wing (surface 0, beam 0)
    i_surf = 0
    airfoil_distribution[working_node:working_node + section1_nelem*2+1] = 0
    surface_distribution[working_elem:working_elem + section1_nelem] = i_surf
    surface_m[i_surf] = m_main
    aero_node[working_node:working_node + section1_nelem*2+1] = True
    chord[working_node:working_node + section1_nelem*2+1] = section1_chord
    elastic_axis[working_node:working_node + section1_nelem*2+1] = main_ea
    working_elem += section1_nelem
    working_node += section1_nelem*2+1

    # right wing (surface 1, beam 1)
    i_surf = 0
    airfoil_distribution[working_node:working_node + section2_nelem*2] = 0
    surface_distribution[working_elem:working_elem + section2_nelem] = i_surf
    surface_m[i_surf] = m_main
    aero_node[working_node:working_node + section2_nelem*2] = True
    chord[working_node:working_node + section2_nelem*2] = np.linspace(section1_chord, section3_chord, section2_nelem*2+1)[1:]
    elastic_axis[working_node:working_node + section2_nelem*2] = main_ea
    working_elem += section2_nelem
    working_node += section2_nelem*2

    # right wing (surface 2, beam 2)
    i_surf = 0
    airfoil_distribution[working_node:working_node + section3_nelem*2] = 0
    surface_distribution[working_elem:working_elem + section3_nelem] = i_surf
    surface_m[i_surf] = m_main
    aero_node[working_node:working_node + section3_nelem*2] = True
    chord[working_node:working_node + section3_nelem*2] = section3_chord
    elastic_axis[working_node:working_node + section3_nelem*2] = main_ea
    working_elem += section3_nelem
    working_node += section3_nelem*2

    # right ogive (surface 3, beam 3)
    i_surf = 0
    airfoil_distribution[working_node:working_node + ogive1_nelem*2] = 0
    surface_distribution[working_elem:working_elem + ogive1_nelem] = i_surf
    surface_m[i_surf] = m_main
    aero_node[working_node:working_node + ogive1_nelem*2] = True
    chord[working_node:working_node + ogive1_nelem*2] = np.linspace(section3_chord, ogive1_chord, ogive1_nelem*2+1)[1:]
    elastic_axis[working_node:working_node + ogive1_nelem*2] = main_ea
    working_elem += ogive1_nelem
    working_node += ogive1_nelem*2

    # right ogive (surface 4, beam 4)
    i_surf = 0
    airfoil_distribution[working_node:working_node + ogive2_nelem*2] = 0
    surface_distribution[working_elem:working_elem + ogive2_nelem] = i_surf
    surface_m[i_surf] = m_main
    aero_node[working_node:working_node + ogive2_nelem*2] = True
    chord[working_node:working_node + ogive2_nelem*2] = np.linspace(ogive1_chord, ogive2_chord, ogive2_nelem*2+1)[1:]
    elastic_axis[working_node:working_node + ogive2_nelem*2] = main_ea
    working_elem += ogive2_nelem
    working_node += ogive2_nelem*2

    # right ogive (surface 5, beam 5)
    i_surf = 0
    airfoil_distribution[working_node:working_node + ogive3_nelem*2] = 0
    surface_distribution[working_elem:working_elem + ogive3_nelem] = i_surf
    surface_m[i_surf] = m_main
    aero_node[working_node:working_node + ogive3_nelem*2] = True
    chord[working_node:working_node + ogive3_nelem*2] = np.linspace(ogive2_chord, ogive3_chord, ogive3_nelem*2+1)[1:]
    elastic_axis[working_node:working_node + ogive3_nelem*2] = main_ea
    working_elem += ogive3_nelem
    working_node += ogive3_nelem*2

    # left ogive (surface 6, beam 6)
    i_surf = 1
    airfoil_distribution[working_node:working_node + ogive3_nelem*2+1] = 0
    surface_distribution[working_elem:working_elem + ogive3_nelem] = i_surf
    surface_m[i_surf] = m_main
    aero_node[working_node:working_node + ogive3_nelem*2+1] = True
    chord[working_node:working_node + ogive3_nelem*2+1] = np.linspace(ogive3_chord, ogive2_chord, ogive3_nelem*2+1)
    elastic_axis[working_node:working_node + ogive3_nelem*2+1] = main_ea
    working_elem += ogive3_nelem
    working_node += ogive3_nelem*2+1

    # left ogive (surface 7, beam 7)
    i_surf = 1
    airfoil_distribution[working_node:working_node + ogive2_nelem*2] = 0
    surface_distribution[working_elem:working_elem + ogive2_nelem] = i_surf
    surface_m[i_surf] = m_main
    aero_node[working_node:working_node + ogive2_nelem*2] = True
    chord[working_node:working_node + ogive2_nelem*2] = np.linspace(ogive2_chord, ogive1_chord, ogive2_nelem*2+1)[1:]
    elastic_axis[working_node:working_node + ogive2_nelem*2] = main_ea
    working_elem += ogive2_nelem
    working_node += ogive2_nelem*2

    # left ogive (surface 8, beam 8)
    i_surf = 1
    airfoil_distribution[working_node:working_node + ogive1_nelem*2] = 0
    surface_distribution[working_elem:working_elem + ogive1_nelem] = i_surf
    surface_m[i_surf] = m_main
    aero_node[working_node:working_node + ogive1_nelem*2] = True
    chord[working_node:working_node + ogive1_nelem*2] = np.linspace(ogive1_chord, section3_chord, ogive1_nelem*2+1)[1:]
    elastic_axis[working_node:working_node + ogive1_nelem*2] = main_ea
    working_elem += ogive1_nelem
    working_node += ogive1_nelem*2

    # left wing (surface 9, beam 9)
    i_surf = 1
    airfoil_distribution[working_node:working_node + section3_nelem*2] = 0
    surface_distribution[working_elem:working_elem + section3_nelem] = i_surf
    surface_m[i_surf] = m_main
    aero_node[working_node:working_node + section3_nelem*2] = True
    chord[working_node:working_node + section3_nelem*2] = section3_chord
    elastic_axis[working_node:working_node + section3_nelem*2] = main_ea
    working_elem += section3_nelem
    working_node += section3_nelem*2

    # left wing (surface 10, beam 10)
    i_surf = 1
    airfoil_distribution[working_node:working_node + section2_nelem*2] = 0
    surface_distribution[working_elem:working_elem + section2_nelem] = i_surf
    surface_m[i_surf] = m_main
    aero_node[working_node:working_node + section2_nelem*2] = True
    chord[working_node:working_node + section2_nelem*2] = np.linspace(section3_chord, section1_chord, section2_nelem*2+1)[1:]
    elastic_axis[working_node:working_node + section2_nelem*2] = main_ea
    working_elem += section2_nelem
    working_node += section2_nelem*2

    # left wing (surface 11, beam 11)
    i_surf = 1
    airfoil_distribution[working_node:working_node + section1_nelem*2] = 0
    surface_distribution[working_elem:working_elem + section1_nelem] = i_surf
    surface_m[i_surf] = m_main
    aero_node[working_node:working_node + section1_nelem*2] = True
    chord[working_node:working_node + section1_nelem*2] = section1_chord
    elastic_axis[working_node:working_node + section1_nelem*2] = main_ea
    working_elem += section1_nelem
    working_node += section1_nelem*2-1

    working_elem += num_elem_fuselage
    working_node += num_node_fuselage - 1 - 1

    # fin (surface 12, beam 13)
    i_surf = 2
    airfoil_distribution[working_node:working_node + num_node_fin] = 0
    surface_distribution[working_elem:working_elem + num_elem_fin] = i_surf
    surface_m[i_surf] = m_fin
    aero_node[working_node:working_node + num_node_fin] = True
    chord[working_node:working_node + num_node_fin] = fin_chord
    twist[end_of_fuselage_node] = 0
    twist[working_node:] = 0
    elastic_axis[working_node:working_node + num_node_fin] = fin_ea
    working_elem += num_elem_fin
    working_node += num_node_fin

    # right tail (surface 13, beam 14)
    i_surf = 3
    airfoil_distribution[working_node:working_node + num_node_tail] = 0
    surface_distribution[working_elem:working_elem + num_elem_tail] = i_surf
    surface_m[i_surf] = m_tail
    aero_node[working_node:] = True
    chord[working_node:working_node + num_node_tail] = tail_chord
    elastic_axis[working_node:working_node + num_node_main] = tail_ea
    twist[working_node:working_node + num_node_tail] = -tail_twist
    working_elem += num_elem_tail
    working_node += num_node_tail

    # left tail (surface 14, beam 15)
    i_surf = 4
    airfoil_distribution[working_node:working_node + num_node_tail-1] = 0
    surface_distribution[working_elem:working_elem + num_elem_tail] = i_surf
    surface_m[i_surf] = m_tail
    aero_node[working_node:working_node + num_node_tail - 1] = True
    chord[working_node:working_node + num_node_tail] = tail_chord
    elastic_axis[working_node:working_node + num_node_main] = tail_ea
    twist[working_node:working_node + num_node_tail-1] = -tail_twist
    working_elem += num_elem_tail
    working_node += num_node_tail

    with h5.File(route + '/' + case_name + '.aero.h5', 'a') as h5file:
        airfoils_group = h5file.create_group('airfoils')
        # add one airfoil
        naca_airfoil_main = airfoils_group.create_dataset('0', data=np.column_stack(
                                generate_naca_camber(P=main_airfoil_P, M=main_airfoil_M)))
        naca_airfoil_tail = airfoils_group.create_dataset('1', data=np.column_stack(
            generate_naca_camber(P=tail_airfoil_P, M=tail_airfoil_M)))
        naca_airfoil_fin = airfoils_group.create_dataset('2', data=np.column_stack(
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


def generate_naca_camber(M=0, P=0):
    m = M*1e-2
    p = P*1e-1
    def naca(x, m, p):
        if x < 1e-6:
            return 0.0
        elif x < p:
            return m/(p*p)*(2*p*x - x*x)
        elif x > p and x < 1+1e-6:
            return m/((1-p)*(1-p))*(1 - 2*p + 2*p*x - x*x)

    x_vec = np.linspace(0, 1, 1000)
    y_vec = np.array([naca(x, m, p) for x in x_vec])
    return x_vec, y_vec


def generate_dyn_file():
    rbm_pos = np.zeros((n_time_steps, 6))
    rbm_vel = np.zeros_like(rbm_pos)

    for it in range(n_time_steps):
        # rbm_pos[it, 2] = A*np.sin(2*np.pi/period*it*dt)
        # rbm_vel[it, 2] = 2*np.pi/period*A*np.cos(2*np.pi*it*dt/period)
        rbm_vel[it, 3] = 2*np.pi/period*A*np.cos(2*np.pi*it*dt/period)*np.pi/180
        # rbm_pos[it, 4] = -np.pi/180*A*np.sin(2*np.pi/period * it*dt)
        # rbm_vel[it, 4] = -np.pi/180*2*np.pi/period*A*np.cos(2*np.pi*it*dt/period)

    with h5.File(route + '/' + case_name + '.dyn.h5', 'a') as h5file:
        rbm_pos_handle = h5file.create_dataset(
            'rbm_pos', data=rbm_pos)
        rbm_vel_handle = h5file.create_dataset(
            'rbm_vel', data=rbm_vel)
        n_tsteps_handle = h5file.create_dataset(
            'num_steps', data=n_time_steps)

def generate_solver_file():
    file_name = route + '/' + case_name + '.solver.txt'
    config = configparser.ConfigParser()
    config['SHARPy'] = {'case': case_name,
                        'route': route,
                        # 'flow': 'StaticCoupled, BeamLoadsCalculator, BeamPlot, AeroGridPlot, AeroForcesSteadyCalculator',
                        # 'flow': 'NonLinearStatic, BeamPlot',
                        # 'flow': 'StaticUvlm, AeroForcesSteadyCalculator, BeamPlot, AeroGridPlot',
                        'flow': 'PrescribedUvlm, AeroForcesSteadyCalculator, BeamPlot, AeroGridPlot',
                        'plot': 'on'}
    config['StaticUvlm'] = {'print_info': 'on',
                            'Mstar': 50,
                            'rollup': 'off',
                            'aligned_grid': 'on',
                            # 'prescribed_wake': 'on',
                            'num_cores': 4,
                            'horseshoe': 'off'}
    config['PrescribedUvlm'] = {'print_info': 'off',
                                'Mstar': 80,
                                'aligned_grid': 'on',
                                'num_cores': 4,
                                'steady_n_rollup': 0,
                                'steady_rollup_dt': main_chord/m_main/u_inf,
                                'steady_rollup_aic_refresh': 1,
                                'steady_rollup_tolerance': 1e-5,
                                'convection_scheme': 3,
                                'n_time_steps': n_time_steps,
                                'dt': dt,
                                'iterative_solver': 'off',
                                'iterative_tol': 1e-3,
                                'iterative_precond': 'off'}

    config['NonLinearStatic'] = {'print_info': 'on',
                                 'out_b_frame': 'off',
                                 'out_a_frame': 'off',
                                 'elem_proj': 0,
                                 'max_iterations': 99,
                                 'num_load_steps': 10,
                                 'delta_curved': 1e-5,
                                 'min_delta': 1e-4,
                                 'newmark_damp': 0.000,
                                 'gravity_on': 'on',
                                 'gravity': 9.754,
                                 'gravity_dir': (str(-np.sin(alpha_rad)) +
                                                 ', ' +
                                                 str(0.0) +
                                                 ', ' +
                                                 str(np.cos(alpha_rad)))
                                 }
    config['BeamPlot'] = {'route': './output',
                          'frame': 'inertial',
                          'applied_forces': 'on',
                          'print_pos_def': 'on',
                          'name_prefix': ''}
    config['AeroGridPlot'] = {'route': './output',
                              'include_rbm': 'on'}
    config['AeroForcesSteadyCalculator'] = {'beams': '0, 1'}
    config['BeamLoadsCalculator'] = {}

    with open(file_name, 'w') as configfile:
        config.write(configfile)


def generate_flightcon_file():
    file_name = route + '/' + case_name + '.flightcon.txt'
    config = configparser.ConfigParser()
    config['FlightCon'] = {'u_inf': u_inf,
                           'alpha': alpha,
                           'beta': beta,
                           'rho_inf': rho,
                           'c_ref': c_ref,
                           'b_ref': b_ref}

    with open(file_name, 'w') as configfile:
        config.write(configfile)


if __name__ == '__main__':
    clean_test_files()
    generate_fem_file()
    generate_dyn_file()
    generate_solver_file()
    generate_aero_file()
    generate_flightcon_file()








