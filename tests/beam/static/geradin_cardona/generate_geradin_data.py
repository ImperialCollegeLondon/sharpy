import h5py as h5
import numpy as np
import configparser


def clean_test_files(route, case_name):
    fem_file_name = route + '/' + case_name + '.fem.h5'
    if os.path.isfile(fem_file_name):
        os.remove(fem_file_name)

    aero_file_name = route + '/' + case_name + '.aero.h5'
    if os.path.isfile(aero_file_name):
        os.remove(aero_file_name)

    solver_file_name = route + '/' + case_name + '.solver.txt'
    if os.path.isfile(solver_file_name):
        os.remove(solver_file_name)

    flightcon_file_name = route + '/' + case_name + '.flightcon.txt'
    if os.path.isfile(flightcon_file_name):
        os.remove(flightcon_file_name)


def generate_files(route, case_name, num_elem=10, num_node_elem=3):
    clean_test_files(route, case_name)
    num_node, coordinates = generate_fem_file(route,
                                              case_name,
                                              num_elem,
                                              num_node_elem
                                              )
    # generate_aero_file(route, case_name, num_elem, num_node, coordinates)
    generate_solver_file(route, case_name)
    # generate_flightcon_file(route, case_name)


def generate_fem_file(route, case_name, num_elem, num_node_elem=3):
    length = 5

    num_node = (num_node_elem - 1)*num_elem + 1
    # import pdb; pdb.set_trace()
    angle = 0*np.pi/180.0
    x = (np.linspace(0, length, num_node))*np.cos(angle)
    y = (np.linspace(0, length, num_node))*np.sin(angle)
    z = np.zeros((num_node,))

    structural_twist = np.zeros_like(x)

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
    m_bar = 0*100
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
    n_app_forces = 0
    node_app_forces = np.array([])
    app_forces = np.zeros((n_app_forces, 6))
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


def generate_aero_file(route, case_name, num_elem, num_node, coordinates, n_vertical_node=5):
    # example airfoil
    naca_file, naca_x, naca_y, naca_description = generate_naca_camber(route, 9)
    # airfoil distribution
    airfoil_distribution = []
    for i in range(num_node):
        if i < n_vertical_node:
            airfoil_distribution.append(0)
        else:
            airfoil_distribution.append(0)

    aero_node = np.zeros(num_node, dtype=bool)
    aero_node[:] = True

    # twist distribution
    twist = np.linspace(0, 5, num_node)*np.pi/180

    # chord distribution
    chord = np.linspace(0.1, 0.05, num_node)

    # elastic axis distribution
    elastic_axis = 0.35*np.ones((num_node,))

    # import pdb; pdb.set_trace()
    with h5.File(route + '/' + case_name + '.aero.h5', 'a') as h5file:
        airfoils_group = h5file.create_group('airfoils')
        # add one airfoil
        naca_airfoil = airfoils_group.create_dataset('0',
                                    data = np.column_stack((naca_x, naca_y)))
        naca_airfoil.attrs['airfoil'] = naca_description

        # add another flat airfoil (or symmetric)
        flat_airfoil = airfoils_group.create_dataset('1',
                                data = np.column_stack((np.linspace(0, 1, 100),
                                                        np.zeros(100,))))
        flat_airfoil.attrs['airfoil'] = 'NACA00xx'

        # chord
        chord_input = h5file.create_dataset('chord', data = chord)
        dim_attr = chord_input .attrs['units'] = 'm'

        # twist
        twist_input = h5file.create_dataset('twist', data = twist)
        dim_attr = twist_input.attrs['units'] = 'rad'

        # airfoil distribution
        airfoil_distribution_input = h5file.create_dataset(
                        'airfoil_distribution',
                        data =airfoil_distribution)

        aero_node_input = h5file.create_dataset('aero_node', data=aero_node)
        elastic_axis_input = h5file.create_dataset('elastic_axis', data=elastic_axis)


def generate_naca_camber(route, M=2, P=4):
    m = M/100
    p = P/10

    def naca(x, m, p):
        if x < p:
            return m/(p*p)*(2*p*x - x*x)
        elif x > p and x < 1+1e-6:
            return m/((1-p)*(1-p))*(1 - 2*p + 2*p*x - x*x)

    # import pdb; pdb.set_trace()
    x_vec = np.linspace(0, 1, 1000)
    y_vec = [naca(x, m, p) for x in x_vec]

    mat = np.column_stack((x_vec, y_vec))
    np.savetxt(route + '/naca.csv', mat, delimiter=',')

    return route + '/naca.csv', x_vec, y_vec, 'NACA%i%ixx'%(M, P)


def generate_solver_file(route, case_name):
    file_name = route + '/' + case_name + '.solver.txt'
    config = configparser.ConfigParser()
    config['SHARPy'] = {'case': 'geradin_cardona',
                        'route': './tests/beam/static/geradin_cardona',
                        'flow': 'NonLinearStatic, BeamLoadsCalculator, BeamPlot',
                        'plot': 'on'}
    config['NonLinearStatic'] = {'print_info': 'on',
                                 'out_b_frame': 'off',
                                 'out_a_frame': 'off',
                                 'elem_proj': 2,
                                 'max_iterations': 99,
                                 'num_load_steps': 5,
                                 'delta_curved': 1e-5,
                                 'min_delta': 1e-8,
                                 'newmark_damp': 0.000,
                                 'gravity_on': 'on',
                                 'gravity': 9.81,
                                 'gravity_dir': '0, 0, 1'
                                 }
    config['BeamPlot'] = {'plot_shape': 'on',
                          'print_info': 'on',
                          'print_pos_def': 'on',
                          'name_prefix': '_2_'}

    config['BeamLoadsCalculator'] = {}

    with open(file_name, 'w') as configfile:
        config.write(configfile)


if __name__ == '__main__':
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    generate_files(dir_path + '/', 'geradin_cardona', 50, 3)
    print('The test case has been successfully generated!!!')
