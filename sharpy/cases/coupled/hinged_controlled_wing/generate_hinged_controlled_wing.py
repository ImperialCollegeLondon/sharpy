import h5py as h5
import numpy as np
import configparser
import os

import sharpy.utils.algebra as algebra
import sharpy.utils.generate_cases as gc

case_name = 'hinged_controlled_wing'
route = os.path.dirname(os.path.realpath(__file__)) + '/'

# m = 16 gives flutter with 165
m_main = 4
amplitude = 5*np.pi/180
period = 3
dt_factor = 1.
n_structural_substeps = 1

# flight conditions
u_inf = 10
rho = 1.225
alpha = 0.5
beta = 0
c_ref = 1
b_ref = 16
sweep = 0*np.pi/180.
sigma = 1
wake_length = 10 # chords

alpha_rad = alpha*np.pi/180

pitch_file = route + 'pitch.csv'

gains = -np.array([0.9, 6.0, 0.75])

# main geometry data
main_span = 10
main_chord = 2.
main_ea = 0.
main_cg = 0.3
main_sigma = 1
main_airfoil_P = 0
main_airfoil_M = 0

n_surfaces = 1

dt = main_chord/m_main/u_inf*dt_factor
num_steps = int(10./dt)

alpha_hist = np.linspace(0, num_steps*dt, num_steps)
alpha_hist = amplitude*np.sin(2.0*np.pi*alpha_hist/period)
np.savetxt(pitch_file, alpha_hist)

# discretisation data
num_elem_main = 5

num_node_elem = 3
num_elem = num_elem_main
num_node_main = num_elem_main*(num_node_elem - 1) + 1
num_node = num_node_main


def clean_test_files():
    fem_file_name = route + '/' + case_name + '.fem.h5'
    if os.path.isfile(fem_file_name):
        os.remove(fem_file_name)

    aero_file_name = route + '/' + case_name + '.aero.h5'
    if os.path.isfile(aero_file_name):
        os.remove(aero_file_name)

    dyn_file_name = route + '/' + case_name + '.dyn.h5'
    if os.path.isfile(dyn_file_name):
        os.remove(dyn_file_name)

    solver_file_name = route + '/' + case_name + '.sharpy'
    if os.path.isfile(solver_file_name):
        os.remove(solver_file_name)

    flightcon_file_name = route + '/' + case_name + '.flightcon.txt'
    if os.path.isfile(flightcon_file_name):
        os.remove(flightcon_file_name)


def generate_dyn_file():
    global dt
    global num_steps
    global route
    global case_name
    global num_elem
    global num_node_elem
    global num_node
    global amplitude
    global period

    dynamic_forces_time = None
    with_dynamic_forces = False
    with_forced_vel = False
    if with_dynamic_forces:
        f1 = 100
        dynamic_forces = np.zeros((num_node, 6))
        app_node = [int(num_node_main - 1), int(num_node_main)]
        dynamic_forces[app_node, 2] = f1
        force_time = np.zeros((num_steps, ))
        limit = round(0.05/dt)
        force_time[50:61] = 1

        dynamic_forces_time = np.zeros((num_steps, num_node, 6))
        for it in range(num_steps):
            dynamic_forces_time[it, :, :] = force_time[it]*dynamic_forces

    forced_for_vel = None
    if with_forced_vel:
        forced_for_vel = np.zeros((num_steps, 6))
        forced_for_acc = np.zeros((num_steps, 6))
        for it in range(num_steps):
            # if dt*it < period:
            forced_for_vel[it, 2] = 2*np.pi/period*amplitude*np.sin(2*np.pi*dt*it/period)
            forced_for_acc[it, 2] = (2*np.pi/period)**2*amplitude*np.cos(2*np.pi*dt*it/period)
            # forced_for_vel[it, 2] = 2*np.pi/period*np.pi/180*amplitude*np.cos(2*np.pi*dt*it/period)

    with h5.File(route + '/' + case_name + '.dyn.h5', 'a') as h5file:
        if with_dynamic_forces:
            h5file.create_dataset(
                'dynamic_forces', data=dynamic_forces_time)
        if with_forced_vel:
            h5file.create_dataset(
                'for_vel', data=forced_for_vel)
            h5file.create_dataset(
                'for_acc', data=forced_for_acc)
        h5file.create_dataset(
            'num_steps', data=num_steps)


def generate_fem_file():
    # placeholders
    # coordinates
    global x, y, z
    global sigma
    x = np.zeros((num_node, ))
    y = np.zeros((num_node, ))
    z = np.zeros((num_node, ))
    # struct twist
    structural_twist = np.zeros((num_elem, 3))
    # beam number
    beam_number = np.zeros((num_elem, ), dtype=int)
    # frame of reference delta
    frame_of_reference_delta = np.zeros((num_elem, num_node_elem, 3))
    # connectivities
    conn = np.zeros((num_elem, num_node_elem), dtype=int)
    # stiffness
    num_stiffness = 1
    ea = 1e5
    ga = 1e5
    gj = 0.987581e6
    eiy = 9.77221e6
    eiz = 9.77221e8
    base_stiffness = sigma*np.diag([ea, ga, ga, gj, eiy, eiz])
    stiffness = np.zeros((num_stiffness, 6, 6))
    stiffness[0, :, :] = main_sigma*base_stiffness
    elem_stiffness = np.zeros((num_elem,), dtype=int)
    # mass
    num_mass = 1
    m_base = 35.71
    j_base = 8.64
    import sharpy.utils.algebra as algebra
    # m_chi_cg = algebra.skew(m_base*np.array([0., -(main_ea - main_cg), 0.]))
    m_chi_cg = algebra.skew(m_base*np.array([0., (main_ea - main_cg), 0.]))
    base_mass = np.diag([m_base, m_base, m_base, j_base, 0.1*j_base, 0.1*j_base])
    base_mass[0:3, 3:6] = -m_chi_cg
    base_mass[3:6, 0:3] = m_chi_cg
    mass = np.zeros((num_mass, 6, 6))
    mass[0, :, :] = base_mass
    elem_mass = np.zeros((num_elem,), dtype=int)
    # boundary conditions
    boundary_conditions = np.zeros((num_node, ), dtype=int)
    boundary_conditions[0] = 1
    # applied forces
    # n_app_forces = 2
    # node_app_forces = np.zeros((n_app_forces,), dtype=int)
    app_forces = np.zeros((num_node, 6))

    spacing_param = 4

    # right wing (beam 0) --------------------------------------------------------------
    working_elem = 0
    working_node = 0
    beam_number[working_elem:working_elem + num_elem_main] = 0
    domain = np.linspace(0, 1.0, num_node_main)
    # 16 - (np.geomspace(20, 4, 10) - 4)
    x[working_node:working_node + num_node_main] = np.sin(sweep)*(main_span - (np.geomspace(main_span + spacing_param,
                                                                                            0 + spacing_param,
                                                                                            num_node_main)
                                                                               - spacing_param))
    y[working_node:working_node + num_node_main] = np.abs(np.cos(sweep)*(main_span - (np.geomspace(main_span + spacing_param,
                                                                                            0 + spacing_param,
                                                                                            num_node_main)
                                                                               - spacing_param)))
    y[0] = 0
    # y[working_node:working_node + num_node_main] = np.cos(sweep)*np.linspace(0.0, main_span, num_node_main)
    # x[working_node:working_node + num_node_main] = np.sin(sweep)*np.linspace(0.0, main_span, num_node_main)
    for ielem in range(num_elem_main):
        for inode in range(num_node_elem):
            frame_of_reference_delta[working_elem + ielem, inode, :] = [-1, 0, 0]
    # connectivity
    for ielem in range(num_elem_main):
        conn[working_elem + ielem, :] = ((np.ones((3,))*(working_elem + ielem)*(num_node_elem - 1)) +
                                         [0, 2, 1])
    elem_stiffness[working_elem:working_elem + num_elem_main] = 0
    elem_mass[working_elem:working_elem + num_elem_main] = 0
    boundary_conditions[0] = 1
    boundary_conditions[working_node + num_node_main - 1] = -1
    working_elem += num_elem_main
    working_node += num_node_main

    # # left wing (beam 1) --------------------------------------------------------------
    # beam_number[working_elem:working_elem + num_elem_main] = 1
    # domain = np.linspace(0.0, 1.0, num_node_main)
    # tempy = np.linspace(0.0, main_span, num_node_main)
    # # x[working_node:working_node + num_node_main - 1] = -np.sin(sweep)*tempy[0:-1]
    # # y[working_node:working_node + num_node_main - 1] = np.cos(sweep)*tempy[0:-1]
    # x[working_node:working_node + num_node_main - 1] = -np.sin(sweep)*(main_span - (np.geomspace(0 + spacing_param,
                                                                                            # main_span + spacing_param,
                                                                                            # num_node_main)[:-1]
                                                                               # - spacing_param))
    # y[working_node:working_node + num_node_main - 1] = -np.abs(np.cos(sweep)*(main_span - (np.geomspace(0 + spacing_param,
                                                                                                   # main_span + spacing_param,
                                                                                                   # num_node_main)[:-1]
                                                                                      # - spacing_param)))
    # for ielem in range(num_elem_main):
        # for inode in range(num_node_elem):
            # frame_of_reference_delta[working_elem + ielem, inode, :] = [1, 0, 0]
    # # connectivity
    # for ielem in range(num_elem_main):
        # conn[working_elem + ielem, :] = ((np.ones((3,))*(working_elem + ielem)*(num_node_elem - 1)) +
                                         # [0, 2, 1])
    # conn[working_elem , 0] = 0
    # elem_stiffness[working_elem:working_elem + num_elem_main] = 0
    # elem_mass[working_elem:working_elem + num_elem_main] = 0
    # boundary_conditions[working_node + num_node_main - 1 - 1] = -1
    # working_elem += num_elem_main
    # working_node += num_node_main - 1

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
        body_number_handle = h5file.create_dataset(
            'body_number', data=np.zeros((num_elem, ), dtype=int))
        # node_app_forces_handle = h5file.create_dataset(
        #     'node_app_forces', data=node_app_forces)


def generate_aero_file():
    global x, y, z

    n_control_surfaces = 1
    control_surface = np.zeros((num_elem, num_node_elem), dtype=int)
    control_surface_type = np.zeros((n_control_surfaces,), dtype=int)
    control_surface_deflection = np.zeros((n_control_surfaces,))
    control_surface_chord = np.zeros((n_control_surfaces, ), dtype=int)
    control_surface_hinge_coord = np.zeros((n_control_surfaces, ))

    control_surface_type[0] = 2
    control_surface_deflection[0] = 0.0
    control_surface_chord[0] = 1

    airfoil_distribution = np.zeros((num_elem, num_node_elem), dtype=int)
    surface_distribution = np.zeros((num_elem,), dtype=int) - 1
    surface_m = np.zeros((n_surfaces, ), dtype=int)
    m_distribution = 'uniform'
    aero_node = np.zeros((num_node,), dtype=bool)
    twist = np.zeros((num_elem, 3))
    chord = np.zeros((num_elem, 3))
    elastic_axis = np.zeros((num_elem, 3,))

    working_elem = 0
    working_node = 0
    # right wing (surface 0, beam 0)
    i_surf = 0
    chord[:] = main_chord
    elastic_axis[:] = main_ea
    airfoil_distribution[working_elem:working_elem + num_elem_main, :] = 0
    surface_distribution[working_elem:working_elem + num_elem_main] = i_surf
    surface_m[i_surf] = m_main
    aero_node[working_node:working_node + num_node_main] = True
    # chord[working_node:working_node + num_node_main] = main_chord
    # elastic_axis[working_node:working_node + num_node_main] = main_ea
    working_elem += num_elem_main
    working_node += num_node_main

    # left wing (surface 1, beam 1)
    # i_surf = 1
    # airfoil_distribution[working_elem:working_elem + num_elem_main, :] = 0
    # # airfoil_distribution[working_node:working_node + num_node_main - 1] = 0
    # surface_distribution[working_elem:working_elem + num_elem_main] = i_surf
    # surface_m[i_surf] = m_main
    # aero_node[working_node:working_node + num_node_main - 1] = True
    # # chord[working_node:working_node + num_node_main - 1] = main_chord
    # # elastic_axis[working_node:working_node + num_node_main - 1] = main_ea
    # working_elem += num_elem_main
    # working_node += num_node_main - 1

    with h5.File(route + '/' + case_name + '.aero.h5', 'a') as h5file:
        airfoils_group = h5file.create_group('airfoils')
        # add one airfoil
        naca_airfoil_main = airfoils_group.create_dataset('0', data=np.column_stack(
                                generate_naca_camber(P=main_airfoil_P, M=main_airfoil_M)))
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

        control_surface_input = h5file.create_dataset('control_surface', data = control_surface)
        control_surface_input = h5file.create_dataset('control_surface_deflection', data = control_surface_deflection)
        control_surface_input = h5file.create_dataset('control_surface_chord', data = control_surface_chord)
        control_surface_input = h5file.create_dataset('control_surface_hinge_coord', data = control_surface_hinge_coord)
        control_surface_input = h5file.create_dataset('control_surface_type', data = control_surface_type)


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

def generate_multibody_file():
    LC2 = gc.LagrangeConstraint()
    LC2.behaviour = 'hinge_FoR'
    LC2.rot_axis_AFoR = np.array([0.0, 1.0, 0.0])
    LC2.body_FoR = 0
    # LC2.node_number = int(0)

    LC = []
    LC.append(LC2)

    MB1 = gc.BodyInformation()
    MB1.body_number = 0
    MB1.FoR_position = np.zeros((6,))
    MB1.FoR_velocity = np.zeros((6,))
    MB1.FoR_acceleration = np.zeros((6,))
    MB1.FoR_movement = 'free'
    MB1.quat = np.array([1.0, 0.0, 0.0, 0.0])

    MB = []
    MB.append(MB1)
    gc.clean_test_files(route, case_name)
    gc.generate_multibody_file(LC, MB, route, case_name)

def generate_solver_file(horseshoe=False):
    file_name = route + '/' + case_name + '.sharpy'
    # config = configparser.ConfigParser()
    import configobj
    config = configobj.ConfigObj()
    config.filename = file_name
    config['SHARPy'] = {'case': case_name,
                        'route': route,
                        'flow': ['BeamLoader',
                                 'AerogridLoader',
                                 'StaticCoupled',
                                 'DynamicCoupled',
                                 # 'AerogridPlot',
                                 # 'BeamPlot',
                                 ],
                        'write_screen': 'on',
                        'write_log': 'on',
                        'log_folder': route + '/output/',
                        'log_file': case_name + '.log'}
    config['BeamLoader'] = {'unsteady': 'on',
                            'orientation': algebra.euler2quat(np.array([0.0,
                                                                        alpha_rad,
                                                                        beta*np.pi/180]))}

    config['StaticCoupled'] = {'print_info': 'on',
                               'structural_solver': 'NonLinearStatic',
                               'structural_solver_settings': {'print_info': 'off',
                                                              'max_iterations': 150,
                                                              'num_load_steps': 1,
                                                              'delta_curved': 1e-5,
                                                              'min_delta': 1e-13,
                                                              'gravity_on': 'on',
                                                              'gravity': 9.754},
                               'aero_solver': 'StaticUvlm',
                               'aero_solver_settings': {'print_info': 'off',
                                                        'horseshoe': 'off',
                                                        'num_cores': 4,
                                                        'n_rollup': 0,
                                                        'rollup_dt': main_chord/m_main/u_inf,
                                                        'rollup_aic_refresh': 1,
                                                        'rollup_tolerance': 1e-4,
                                                        'velocity_field_generator': 'SteadyVelocityField',
                                                        'velocity_field_input': {'u_inf': u_inf,
                                                                                 'u_inf_direction': [1., 0, 0]},
                                                        'rho': rho},
                               'max_iter': 80,
                               'n_load_steps': 1,
                               'tolerance': 1e-5,
                               'relaxation_factor': 0.0}

    config['DynamicCoupled'] = {'print_info': 'on',
                                'structural_solver': 'NonLinearDynamicMultibody',
                                'structural_solver_settings': {'print_info': 'off',
                                                               'max_iterations': 550,
                                                               'num_load_steps': 1,
                                                               'delta_curved': 1e-1,
                                                               'min_delta': 1e-4,
                                                               'newmark_damp': 5e-3,
                                                               'gravity_on': 'on',
                                                               'gravity': 9.754,
                                                               'num_steps': num_steps,
                                                               'dt': dt,
                                                               'relax_factor_lm': 0.2,
                                                               'time_integrator': 'NewmarkBeta',
                                                               'time_integrator_settings':
                                                                   {'dt': dt,
                                                                    'newmark_damp': 5e-3,
                                                                    'num_LM_eq': 5}},
                                'aero_solver': 'StepUvlm',
                                'aero_solver_settings':
                                    {'print_info': 'off',
                                     'num_cores': 4,
                                     'convection_scheme': 2,
                                     'velocity_field_generator': 'GustVelocityField',
                                     'velocity_field_input': {'u_inf': u_inf,
                                                              'u_inf_direction': [1., 0, 0],
                                                              'gust_shape': '1-cos',
                                                              'gust_parameters': {
                                                                  'gust_length': 4.0 * main_chord,
                                                                  'gust_intensity': 0.1 * u_inf},
                                                              'offset': 1.0,
                                                              'relative_motion': 'on'},
                                     'rho': rho,
                                     'n_time_steps': num_steps,
                                     'dt': dt},
                                'controller_id': {'controller_tip': 'ControlSurfacePidController'},
                                'controller_settings': {'controller_tip': {'P': gains[0],
                                                                           'I': gains[1],
                                                                           'D': gains[2],
                                                                           'dt': dt,
                                                                           'input_type': 'pitch',
                                                                           'controller_log_route': './output/' + case_name + '/',
                                                                           'controlled_surfaces': 0,
                                                                           'time_history_input_file': 'pitch.csv'}},
                                'postprocessors': ['BeamPlot', 'AerogridPlot'],
                                'postprocessors_settings': {'BeamPlot': {'include_rbm': 'on',
                                                                         'include_applied_forces': 'on'},
                                                            'AerogridPlot': {
                                                                'include_rbm': 'on',
                                                                'include_applied_forces': 'on',
                                                                'minus_m_star': 0}},
                                'fsi_substeps': 100,
                                # 'fsi_substeps': 1,
                                'fsi_tolerance': 1e-5,
                                'relaxation_factor': 0.3,
                                'dynamic_relaxation': 'off',
                                'minimum_steps': 1,
                                'relaxation_steps': 25,
                                'structural_substeps': n_structural_substeps,
                                'n_time_steps': num_steps,
                                'dt': dt}

    config['AerogridLoader'] = {'unsteady': 'on',
                                'aligned_grid': 'on',
                                'mstar': 1,
                                'freestream_dir': ['1', '0', '0'],
                                'wake_shape_generator': 'StraightWake',
                                'wake_shape_generator_input': {'u_inf': u_inf,
                                                               'u_inf_direction': ['1', '0', '0'],
                                                               'dt': dt}}
    if not horseshoe:
        config['AerogridLoader']['mstar'] = int(m_main*wake_length)

    config['AerogridPlot'] = {'include_rbm': 'on',
                              'include_applied_forces': 'on',
                              'minus_m_star': 0
                              }
    config['AeroForcesCalculator'] = {'write_text_file': 'on',
                                      'text_file_name': case_name + '_aeroforces.csv',
                                      'screen_output': 'on',
                                      'unsteady': 'off'
                                      }
    config['BeamPlot'] = {'include_rbm': 'on',
                          'include_applied_forces': 'on'}
    config.write()


clean_test_files()
generate_multibody_file()
generate_fem_file()
generate_dyn_file()
generate_solver_file(horseshoe=False)
generate_aero_file()
