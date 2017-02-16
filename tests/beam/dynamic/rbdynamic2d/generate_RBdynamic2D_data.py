import h5py as h5
import numpy as np
import os
import configparser

"""
2D rigid body motion beam test case
see Hesse thesis (chap. 5.1.1)
Introduced by Simo and Vu-Quoc, modified because
xbeam will not support dead loads.
This case has the same loads than Simo, but
the forces are follower forces

UPDATE: modified case, with gravity loads and
different loads. Still very easy to replicate
Simo if necessary"""

dt = 0.005
simulation_time = 1
num_steps = int(simulation_time/dt)
route = './'
case_name = 'rbdynamic2d'
num_elem = 10
num_node_elem = 3

# dont touch this
num_node = 0


def clean_test_files():
    global dt
    global num_steps
    global route
    global case_name
    global num_elem
    global num_node_elem
    global num_node
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


def generate_files():
    global dt
    global num_steps
    global route
    global case_name
    global num_elem
    global num_node_elem
    global num_node
    clean_test_files()
    generate_fem_file()
    generate_solver_file()
    generate_dyn_file()


def generate_dyn_file():
    global dt
    global num_steps
    global route
    global case_name
    global num_elem
    global num_node_elem
    global num_node

    with_dynamic_forces = True
    if with_dynamic_forces:
        angle = 0*np.arctan(8.0/6.0)
        dynamic_forces = np.zeros((num_node, 6))
        # dynamic_forces[0, 1] = 10
        # dynamic_forces[-1, 1] = -10
        # dynamic_forces[-1, 4] = -10
        force_time = np.zeros((num_steps, ))
        force_time[0:50] = np.linspace(0, 1, 50)
        # i_top = int(2.5/dt)
        # force_time[:i_top] = np.linspace(0, 1, i_top)

    with h5.File(route + '/' + case_name + '.dyn.h5', 'a') as h5file:
        if with_dynamic_forces:
            dynamic_forces_handle = h5file.create_dataset(
                'dynamic_forces_amplitude', data=dynamic_forces)
            dynamic_forces_handle = h5file.create_dataset(
                'dynamic_forces_time', data=force_time)
        num_steps_handle = h5file.create_dataset(
            'num_steps', data=num_steps)


def generate_fem_file():
    global dt
    global num_steps
    global route
    global case_name
    global num_elem
    global num_node_elem
    global num_node
    length = 1  # np.linalg.norm(np.array([6, 8, 0]))

    num_node = (num_node_elem - 1)*num_elem + 1
    # import pdb; pdb.set_trace()
    angle = 45*np.pi/180
    x = (np.linspace(0, length, num_node))*np.cos(angle)
    y = np.zeros((num_node,))
    z = (np.linspace(0, length, num_node))*np.sin(angle)

    structural_twist = np.zeros_like(x)

    frame_of_reference_delta = np.zeros((num_node, 3))
    for inode in range(num_node):
        # frame_of_reference_delta[inode, :] = [0, 1, 0]
        frame_of_reference_delta[inode, :] = [0, 1, 0]

    scale = 1

    x *= scale
    y *= scale
    z *= scale

    conn = np.zeros((num_elem, num_node_elem), dtype=int)
    for ielem in range(num_elem):
        conn[ielem, :] = (np.ones((3,)) * ielem * (num_node_elem - 1)
                          + [0, 2, 1])

    # stiffness array
    # import pdb; pdb.set_trace()
    num_stiffness = 1
    ea = 1e4
    ga = 1e4
    gj = 50
    ei = 500
    base_stiffness = np.diag([ea, ga, ga, gj, ei, ei])
    stiffness = np.zeros((num_stiffness, 6, 6))
    # print('Sigma, careful')
    sigma = 1
    # import pdb; pdb.set_trace()
    for i in range(num_stiffness):
        stiffness[i, :, :] = sigma*base_stiffness

    # element stiffness
    elem_stiffness = np.zeros((num_elem,), dtype=int)

    # mass array
    num_mass = 1
    m_bar = 0.1
    j = 1
    base_mass = np.diag([m_bar, m_bar, m_bar, 2*j, j, j])
    mass = np.zeros((num_mass, 6, 6))
    for i in range(num_mass):
        mass[i, :, :] = base_mass
    # element masses
    elem_mass = np.zeros((num_elem,), dtype=int)

    # bocos
    boundary_conditions = np.zeros((num_node, 1), dtype=int)
    boundary_conditions[0] = 1
    boundary_conditions[-1] = -1

    # beam number
    beam_number = np.zeros((num_elem, 1), dtype=int)

    # applied static forces
    n_app_forces = 0
    node_app_forces = np.array([])
    app_forces = np.zeros((n_app_forces, 6))
    # app_forces[0, :] = [0, 0, 0, 0, 0, 0]

    # lumped masses input
    n_lumped_mass = 1
    lumped_mass_nodes = np.array([num_node - 1], dtype=int)
    lumped_mass = np.zeros((n_lumped_mass, ))
    lumped_mass[0] = 20
    lumped_mass_inertia = np.zeros((n_lumped_mass, 3, 3))
    lumped_mass_position = np.zeros((n_lumped_mass, 3))

    with h5.File(route + '/' + case_name + '.fem.h5', 'a') as h5file:
        coordinates = h5file.create_dataset('coordinates', data = np.column_stack((x, y, z)))
        conectivities = h5file.create_dataset('connectivities', data = conn)
        num_nodes_elem_handle = h5file.create_dataset(
            'num_node_elem', data = num_node_elem)
        num_nodes_handle = h5file.create_dataset(
            'num_node', data = num_node)
        num_elem_handle = h5file.create_dataset(
            'num_elem', data = num_elem)
        stiffness_db_handle = h5file.create_dataset(
            'stiffness_db', data = stiffness)
        stiffness_handle = h5file.create_dataset(
            'elem_stiffness', data = elem_stiffness)
        mass_db_handle = h5file.create_dataset(
            'mass_db', data = mass)
        mass_handle = h5file.create_dataset(
            'elem_mass', data = elem_mass)
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
    return num_node, coordinates


def generate_solver_file():
    global dt
    global num_steps
    global route
    global case_name
    global num_elem
    global num_node_elem
    global num_node
    file_name = route + '/' + case_name + '.solver.txt'
    config = configparser.ConfigParser()
    config['SHARPy'] = {'case': case_name,
                        'route': './tests/beam/dynamic/rbdynamic2d',
                        'flow': 'NonLinearDynamic',
                        'plot': 'on'}
    config['NonLinearDynamic'] = {'print_info': 'on',
                                  'out_b_frame': 'off',
                                  'out_a_frame': 'off',
                                  'elem_proj': 2,
                                  'max_iterations': 150,
                                  'num_load_steps': 10,
                                  'num_gauss': 2,
                                  'delta_curved': 1e-5,
                                  'min_delta': 1e-5,
                                  'newmark_damp': 0.001,
                                  'dt': dt,
                                  'num_steps': num_steps,
                                  'prescribed_motion': 'off',
                                  'gravity_on': 'on',
                                  'gravity': 9.81,
                                  'gravity_dir': '0, 0, 1'}

    with open(file_name, 'w') as configfile:
        config.write(configfile)

if __name__ == '__main__':
    generate_files()
