import h5py as h5
import numpy as np
import os
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
                                              num_node_elem)
    generate_aero_file(route, case_name, num_elem, num_node, coordinates)
    generate_solver_file(route, case_name)
    generate_flightcon_file(route, case_name)


def generate_fem_file(route, case_name, num_elem, num_node_elem=3):
    # generate dummy set
    num_node = (num_node_elem - 1)*num_elem+2
    n_vertical_elem = 2
    n_vertical_node = n_vertical_elem*num_node_elem - 1
    # import pdb; pdb.set_trace()
    x = np.zeros((num_node,))
    y = np.zeros((num_node,))
    z = np.zeros((num_node,))

    x[n_vertical_node:] = np.linspace(0, 1, num_node-n_vertical_node)  #np.zeros((num_node,))
    y[n_vertical_node:] = np.power(np.linspace(0, 1, num_node-n_vertical_node), 1.5)
    z[n_vertical_node:] = np.power(y[:num_node - n_vertical_node], 2)

    x[:n_vertical_node] = 0
    y[:n_vertical_node] = 0
    z[:n_vertical_node] = np.linspace(0.5, 1, n_vertical_node)

    scale = 10

    x *= scale
    y *= scale
    z *= scale

    conn = np.zeros((num_elem, num_node_elem), dtype=int)
    for ielem in range(num_elem):
        if ielem < 2:
            conn[ielem, :] = (np.ones((3,)) * ielem * (num_node_elem - 1)
                              + [0, 1, 2])
        else:
            conn[ielem,:] = (np.ones((3,))*ielem*(num_node_elem - 1)
                            + [0, 1, 2] + 1)

    # stiffness array
    # import pdb; pdb.set_trace()
    num_stiffness = 2
    base_stiffness = 1e4*np.diag([10, 10, 10, 1, 1, 1])
    stiffness = np.zeros((num_stiffness, 6, 6))
    # import pdb; pdb.set_trace()
    for i in range(num_stiffness):
        stiffness[i,:,:] = 1/(i+1)*base_stiffness

    # element stiffness
    elem_stiffness = np.zeros((num_elem,), dtype=int)
    elem_stiffness[int(num_elem/2):] = 1

    # mass array
    num_mass = 3;
    base_mass = 0.4*np.diag([1, 1, 1, 0.1, 0.5, 0.5])
    mass = np.zeros((num_mass, 6, 6))
    for i in range(num_mass):
        mass[i,:,:] = 1/(i+1)*base_mass
    #element masses
    elem_mass = np.zeros((num_elem,), dtype=int)
    elem_mass[int(num_elem/3+1):int(2*num_elem/3)] = 1
    elem_mass[int(2*num_elem/3+1):] = 2

    # import pdb; pdb.set_trace()
    with h5.File(route + '/' + case_name + '.fem.h5', 'a') as h5file:
        coordinates = h5file.create_dataset('coordinates', data = np.column_stack((x,y,z)))
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

    return num_node, coordinates

def generate_aero_file(route, case_name, num_elem, num_node, coordinates):
    # example airfoil
    naca_file, naca_x, naca_y, naca_description = generate_naca_camber(route)
    # airfoil distribution
    airfoil_distribution = []
    for i in range(num_node):
        if (i < num_node/2):
            airfoil_distribution.append(0)
        else:
            airfoil_distribution.append(1)

    # twist distribution
    twist = np.linspace(0, 5, num_node)*np.pi/180

    # chord distribution
    chord = np.linspace(0.1, 0.05, num_node)

    # import pdb; pdb.set_trace()
    with h5.File(route + '/' + case_name + '.aero.h5', 'a') as h5file:
        airfoils_group = h5file.create_group('airfoils')
        #add one airfoil
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
    config['GRID'] = {'M': 5,
                      'M_distribution': 'linear',
                      'wake_length': 10,
                      'rollup': 'on',
                      'aligned_grid': 'on'}

    with open(file_name, 'w') as configfile:
        config.write(configfile)


def generate_flightcon_file(route, case_name):
    file_name = route + '/' + case_name + '.flightcon.txt'
    config = configparser.ConfigParser()
    config['FLIGHT_CONDITIONS'] = {'Q': 5,
                                   'rho': 1.225,
                                   'alpha': 3,
                                   'beta': 0,
                                   'delta': 0}

    with open(file_name, 'w') as configfile:
        config.write(configfile)

















if __name__ == '__main__':
    generate_files('./', 'test', 10, 3)
