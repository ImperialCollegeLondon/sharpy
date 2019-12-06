import h5py as h5
import numpy as np
import configparser
import os

case_name = 'geradin'
route = os.path.dirname(os.path.realpath(__file__)) + '/'


def clean_test_files():
    fem_file_name = route + '/' + case_name + '.fem.h5'
    if os.path.isfile(fem_file_name):
        os.remove(fem_file_name)

    solver_file_name = route + '/' + case_name + '.sharpy'
    if os.path.isfile(solver_file_name):
        os.remove(solver_file_name)


def generate_fem_file(route, case_name, num_elem, num_node_elem=3):
    length = 5

    num_node = (num_node_elem - 1)*num_elem + 1
    # import pdb; pdb.set_trace()
    angle = 0*np.pi/180.0
    x = (np.linspace(0, length, num_node))*np.cos(angle)
    y = (np.linspace(0, length, num_node))*np.sin(angle)
    z = np.zeros((num_node,))

    structural_twist = np.zeros((num_elem, num_node_elem))

    frame_of_reference_delta = np.zeros((num_elem, num_node_elem, 3))
    for ielem in range(num_elem):
        for inode in range(num_node_elem):
            # frame_of_reference_delta[inode, :] = [0, 1, 0]
            frame_of_reference_delta[ielem, inode, :] = [-np.sin(angle), np.cos(angle), 0]

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
    ea = 4.8e8
    ga = 3.231e8
    gj = 1.0e6
    ei = 9.346e6
    base_stiffness = np.diag([ea, ga, ga, gj, ei, ei])
    stiffness = np.zeros((num_stiffness, 6, 6))
    # import pdb; pdb.set_trace()
    for i in range(num_stiffness):
        stiffness[i, :, :] = base_stiffness

    # element stiffness
    elem_stiffness = np.zeros((num_elem,), dtype=int)

    # mass array
    num_mass = 1
    m_bar = 0.
    j = 10
    base_mass = np.diag([m_bar, m_bar, m_bar, j, j, j])
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

    # new app forces scheme (only follower)
    app_forces = np.zeros((num_node, 6))
    # app_forces[0, :] = [0, 0, 3000000, 0, 0, 0]

    # lumped masses input
    n_lumped_mass = 1
    lumped_mass_nodes = np.array([num_node - 1], dtype=int)
    lumped_mass = np.zeros((n_lumped_mass, ))
    lumped_mass[0] = 600e3/9.81
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
    file_name = route + '/' + case_name + '.sharpy'
    # config = configparser.ConfigParser()
    import configobj
    config = configobj.ConfigObj()
    config.filename = file_name
    config['SHARPy'] = {'case': case_name,
                        'route': route,
                        'flow': ['BeamLoader', 'NonLinearStatic', 'WriteVariablesTime'],
                        'write_screen': 'off',
                        'write_log': 'on',
                        'log_folder': route + '/output/',
                        'log_file': case_name + '.log'}

    config['BeamLoader'] = {'unsteady': 'off'}
    config['NonLinearStatic'] = {'print_info': 'off',
                                 'max_iterations': 99,
                                 'num_load_steps': 10,
                                 'delta_curved': 1e-5,
                                 'min_delta': 1e-8,
                                 'gravity_on': 'on',
                                 'gravity': 9.81,
                                 'gravity_dir': ['0', '0', '1']}
    config['WriteVariablesTime'] = {'structure_variables': ['pos', 'psi'],
                                    'cleanup_old_solution': 'on',
                                    'folder': route + '/output'}
    config['BeamPlot'] = {'folder': route + '/output',
                          'include_rbm': 'off',
                          'include_applied_forces': 'on'}

    config.write()
    # with open(file_name, 'w') as configfile:
    #     config.write(configfile)


# run everything
clean_test_files()
generate_fem_file(route, case_name, 20)
generate_solver_file()







