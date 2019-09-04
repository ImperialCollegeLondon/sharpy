#! /usr/bin/env python3
import os
import math
import h5py as h5
import numpy as np
import sharpy.utils.algebra as algebra
import sharpy.utils.generate_cases as gc


def generate(x_dict={}, case_name=None):
    """

    """
    if case_name is None:
        case_name = 'base'
    route = os.path.dirname(os.path.realpath(__file__)) + '/'

    case_notes = str(x_dict)

# EXECUTION
    flow = ['BeamLoader',
            'AerogridLoader',
            # 'StaticTrim',
            'StaticCoupled',
            'BeamLoads',
            'DynamicCoupled',
            # 'PickleData'
            ]

# FLIGHT CONDITIONS
# the simulation is set such that the aircraft flies at a u_inf velocity while
# the air is calm.
    try:
        u_inf = x_dict['release_vel']
    except KeyError:
        u_inf = 10

    u_inf_cruise = 10
    original_u_inf = u_inf
    u_background = 0.5
    rho = 1.225

# trim sigma = 1.5
    try:
        alpha_cato_delta = x_dict['dAoA']*np.pi/180
    except KeyError:
        alpha_cato_delta = 0*np.pi/180

    try:
        ramp_angle = x_dict['ramp_angle']*np.pi/180
    except KeyError:
        ramp_angle = 0.0

    alpha = 4.0782*np.pi/180 + alpha_cato_delta
    beta = 0
    roll = 0
    gravity = 'on'
    cs_deflection = -1.2703*np.pi/180
    rudder_static_deflection = 0.0
    # rudder_step = 0.0*np.pi/180
    thrust = 3.8682
    sigma = 1.5
    lambda_dihedral = 20*np.pi/180


# trajectory
    t_start = 1.5

    try:
        acceleration = x_dict['acceleration']
    except KeyError:
        acceleration = 3.

    t_ramp = u_inf/acceleration
    t_finish = t_start + t_ramp
    t_free = 8
    controller_ramp = -1

# numerics
    n_step = 1
    relaxation_factor = 0.4
    tolerance = 1e-6
    fsi_tolerance = 1e-5
    structural_substeps = 1

    num_cores = 4

# MODEL GEOMETRY
# beam
    span_main = 16.0
    lambda_main = 0.25
    ea_main = 0.3

    ea = 1e7
    ga = 1e5
    gj = 1e4
    eiy = 2e4
    eiz = 4e5
    m_bar_main = 0.75
    j_bar_main = 0.075

    length_fuselage = 10
    offset_fuselage = 0
    sigma_fuselage = 10
    m_bar_fuselage = 0.2
    j_bar_fuselage = 0.08

    span_tail = 2.5
    ea_tail = 0.5
    fin_height = 2.5
    ea_fin = 0.5
    sigma_tail = 10
    m_bar_tail = 0.3
    j_bar_tail = 0.08

# lumped masses
    n_lumped_mass = 1
    lumped_mass_nodes = np.zeros((n_lumped_mass, ), dtype=int)
    lumped_mass = np.zeros((n_lumped_mass, ))
    lumped_mass[0] = 50
    lumped_mass_inertia = np.zeros((n_lumped_mass, 3, 3))
    lumped_mass_position = np.zeros((n_lumped_mass, 3))

# aero
    chord_main = 1.0
    chord_tail = 0.5
    chord_fin = 0.5

# DISCRETISATION
# spatial discretisation
# chordiwse panels
    m = 4
# spanwise elements
    n_elem_multiplier = 2
    n_elem_main = int(4*n_elem_multiplier)
    n_elem_tail = int(2*n_elem_multiplier)
    n_elem_fin = int(2*n_elem_multiplier)
    n_elem_fuselage = int(2*n_elem_multiplier)
    n_surfaces = 5

# temporal discretisation
    physical_time = t_finish + t_free
    tstep_factor = 1.
    dt = 1.0/m/u_inf_cruise*tstep_factor
    n_tstep = int(round(physical_time/dt))

# END OF INPUT-----------------------------------------------------------------

    end_of_fuselage_node = 0
    flat_end_node = np.zeros((2, ), dtype=int) - 1

# beam processing
    n_node_elem = 3
    span_main1 = (1.0 - lambda_main)*span_main
    span_main2 = lambda_main*span_main

    n_elem_main1 = round(n_elem_main*(1 - lambda_main))
    n_elem_main2 = n_elem_main - n_elem_main1

# total number of elements
    n_elem = 0
    n_elem += n_elem_main1 + n_elem_main1
    n_elem += n_elem_main2 + n_elem_main2
    n_elem += n_elem_fuselage
    n_elem += n_elem_fin
    n_elem += n_elem_tail + n_elem_tail

# number of nodes per part
    n_node_main1 = n_elem_main1*(n_node_elem - 1) + 1
    n_node_main2 = n_elem_main2*(n_node_elem - 1) + 1
    n_node_main = n_node_main1 + n_node_main2 - 1
    n_node_fuselage = n_elem_fuselage*(n_node_elem - 1) + 1
    n_node_fin = n_elem_fin*(n_node_elem - 1) + 1
    n_node_tail = n_elem_tail*(n_node_elem - 1) + 1

# total number of nodes
    n_node = 0
    n_node += n_node_main1 + n_node_main1 - 1
    n_node += n_node_main2 - 1 + n_node_main2 - 1
    n_node += n_node_fuselage - 1
    n_node += n_node_fin - 1
    n_node += n_node_tail - 1
    n_node += n_node_tail - 1

# stiffness and mass matrices
    n_stiffness = 3
    base_stiffness_main = sigma*np.diag([ea, ga, ga, gj, eiy, eiz])
    base_stiffness_fuselage = base_stiffness_main.copy()*sigma_fuselage
    base_stiffness_fuselage[4, 4] = base_stiffness_fuselage[5, 5]
    base_stiffness_tail = base_stiffness_main.copy()*sigma_tail
    base_stiffness_tail[4, 4] = base_stiffness_tail[5, 5]

    n_mass = 3
    base_mass_main = np.diag([m_bar_main, m_bar_main, m_bar_main, j_bar_main, 0.5*j_bar_main, 0.5*j_bar_main])
    base_mass_fuselage = np.diag([m_bar_fuselage,
                                  m_bar_fuselage,
                                  m_bar_fuselage,
                                  j_bar_fuselage,
                                  j_bar_fuselage*0.5,
                                  j_bar_fuselage*0.5])
    base_mass_tail = np.diag([m_bar_tail,
                              m_bar_tail,
                              m_bar_tail,
                              j_bar_tail,
                              j_bar_tail*0.5,
                              j_bar_tail*0.5])


# PLACEHOLDERS
# beam
    x = np.zeros((n_node, ))
    y = np.zeros((n_node, ))
    z = np.zeros((n_node, ))
    beam_number = np.zeros((n_elem, ), dtype=int)
    frame_of_reference_delta = np.zeros((n_elem, n_node_elem, 3))
    structural_twist = np.zeros((n_elem, n_node_elem))
    conn = np.zeros((n_elem, n_node_elem), dtype=int)
    stiffness = np.zeros((n_stiffness, 6, 6))
    elem_stiffness = np.zeros((n_elem, ), dtype=int)
    mass = np.zeros((n_mass, 6, 6))
    elem_mass = np.zeros((n_elem, ), dtype=int)
    boundary_conditions = np.zeros((n_node, ), dtype=int)
    app_forces = np.zeros((n_node, 6))

    trajectory_file = route + '/' + case_name + '.traj.csv'
    LC = None
    first_node_centre = np.zeros((2, ), dtype=int)

# aero
    airfoil_distribution = np.zeros((n_elem, n_node_elem), dtype=int)
    surface_distribution = np.zeros((n_elem,), dtype=int) - 1
    surface_m = np.zeros((n_surfaces, ), dtype=int)
    m_distribution = 'uniform'
    aero_node = np.zeros((n_node,), dtype=bool)
    twist = np.zeros((n_elem, n_node_elem))
    sweep = np.zeros((n_elem, n_node_elem))
    chord = np.zeros((n_elem, n_node_elem,))
    elastic_axis = np.zeros((n_elem, n_node_elem,))


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

        traj_file_name = route + '/' + case_name + '.traj.csv'
        if os.path.isfile(traj_file_name):
            os.remove(traj_file_name)

        flightcon_file_name = route + '/' + case_name + '.flightcon.txt'
        if os.path.isfile(flightcon_file_name):
            os.remove(flightcon_file_name)


    def generate_trajectory_file():
        it_start = math.ceil(t_start/dt)
        it_ramp = math.ceil(t_ramp/dt)
        it_total = it_start + it_ramp
        out_t = np.linspace(0, it_total*dt, it_total)
        out_x = np.zeros((it_total, ))
        out_y = np.zeros((it_total, ))
        out_z = np.zeros((it_total, ))

        t_hist_ramp = np.linspace(0, dt*it_ramp, it_ramp)
        x_ramp = -0.5*np.cos(ramp_angle)*acceleration*t_hist_ramp**2
        z_ramp = 0.5*np.sin(ramp_angle)*acceleration*t_hist_ramp**2

        out_x[:it_start] = 0.0
        out_x[it_start:] = x_ramp
        out_z[it_start:] = z_ramp

        out = np.zeros((it_total, 4))
        out[:, 0] = out_t
        out[:, 1] = out_x
        out[:, 2] = out_y
        out[:, 3] = out_z
        np.savetxt(trajectory_file, out, delimiter=',')

    def generate_dyn_file():
        global dt
        global n_tstep
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
            force_time = np.zeros((n_tstep, ))
            limit = round(0.05/dt)
            force_time[50:61] = 1

            dynamic_forces_time = np.zeros((n_tstep, num_node, 6))
            for it in range(n_tstep):
                dynamic_forces_time[it, :, :] = force_time[it]*dynamic_forces

        forced_for_vel = None
        if with_forced_vel:
            forced_for_vel = np.zeros((n_tstep, 6))
            forced_for_acc = np.zeros((n_tstep, 6))
            for it in range(n_tstep):
                # if dt*it < period:
                # forced_for_vel[it, 2] = 2*np.pi/period*amplitude*np.sin(2*np.pi*dt*it/period)
                # forced_for_acc[it, 2] = (2*np.pi/period)**2*amplitude*np.cos(2*np.pi*dt*it/period)

                forced_for_vel[it, 3] = 2*np.pi/period*amplitude*np.sin(2*np.pi*dt*it/period)
                forced_for_acc[it, 3] = (2*np.pi/period)**2*amplitude*np.cos(2*np.pi*dt*it/period)

        if with_dynamic_forces or with_forced_vel:
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
                    'num_steps', data=n_tstep)

    def generate_fem():
        stiffness[0, ...] = base_stiffness_main
        stiffness[1, ...] = base_stiffness_fuselage
        stiffness[2, ...] = base_stiffness_tail

        mass[0, ...] = base_mass_main
        mass[1, ...] = base_mass_fuselage
        mass[2, ...] = base_mass_tail

        we = 0
        wn = 0
        # inner right wing
        beam_number[we:we + n_elem_main1] = 0
        y[wn:wn + n_node_main1] = np.linspace(0.0, span_main1, n_node_main1)

        for ielem in range(n_elem_main1):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) +
                                   [0, 2, 1])
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]

        elem_stiffness[we:we + n_elem_main1] = 0
        elem_mass[we:we + n_elem_main1] = 0
        flat_end_node[0] = n_node_main1 - 1
        first_node_centre[0] = wn + 1 + 1
        boundary_conditions[0] = 1
        # remember this is in B FoR
        app_forces[0] = [0, thrust, 0, 0, 0, 0]
        we += n_elem_main1
        wn += n_node_main1

        # outer right wing
        beam_number[we:we + n_elem_main1] = 0
        y[wn:wn + n_node_main2 - 1] = y[wn - 1] + np.linspace(0.0, np.cos(lambda_dihedral)*span_main2, n_node_main2)[1:]
        z[wn:wn + n_node_main2 - 1] = z[wn - 1] + np.linspace(0.0, np.sin(lambda_dihedral)*span_main2, n_node_main2)[1:]
        for ielem in range(n_elem_main2):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) +
                                   [0, 2, 1])
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
        elem_stiffness[we:we + n_elem_main2] = 0
        elem_mass[we:we + n_elem_main2] = 0
        boundary_conditions[wn + n_node_main2 - 2] = -1
        we += n_elem_main2
        wn += n_node_main2 - 1

        # inner left wing
        beam_number[we:we + n_elem_main1 - 1] = 1
        y[wn:wn + n_node_main1 - 1] = np.linspace(0.0, -span_main1, n_node_main1)[1:]
        for ielem in range(n_elem_main1):
            conn[we + ielem, :] = ((np.ones((3, ))*(we+ielem)*(n_node_elem - 1)) +
                                   [0, 2, 1])
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
        conn[we, 0] = 0
        elem_stiffness[we:we + n_elem_main1] = 0
        elem_mass[we:we + n_elem_main1] = 0
        flat_end_node[1] = wn + n_node_main1 - 1 - 1
        first_node_centre[1] = wn + 1
        we += n_elem_main1
        wn += n_node_main1 - 1

        # outer left wing
        beam_number[we:we + n_elem_main2] = 1
        y[wn:wn + n_node_main2 - 1] = y[wn - 1] + np.linspace(0.0, -np.cos(lambda_dihedral)*span_main2, n_node_main2)[1:]
        z[wn:wn + n_node_main2 - 1] = z[wn - 1] + np.linspace(0.0, np.sin(lambda_dihedral)*span_main2, n_node_main2)[1:]
        for ielem in range(n_elem_main2):
            conn[we + ielem, :] = ((np.ones((3, ))*(we+ielem)*(n_node_elem - 1)) +
                                   [0, 2, 1])
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
        elem_stiffness[we:we + n_elem_main2] = 0
        elem_mass[we:we + n_elem_main2] = 0
        boundary_conditions[wn + n_node_main2 - 2] = -1
        we += n_elem_main2
        wn += n_node_main2 - 1

        # fuselage
        beam_number[we:we + n_elem_fuselage] = 2
        x[wn:wn + n_node_fuselage - 1] = np.linspace(0.0, length_fuselage, n_node_fuselage)[1:]
        z[wn:wn + n_node_fuselage - 1] = np.linspace(0.0, offset_fuselage, n_node_fuselage)[1:]
        for ielem in range(n_elem_fuselage):
            conn[we + ielem, :] = ((np.ones((3,))*(we + ielem)*(n_node_elem - 1)) +
                                   [0, 2, 1])
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [0.0, 1.0, 0.0]
        conn[we, 0] = 0
        elem_stiffness[we:we + n_elem_fuselage] = 1
        elem_mass[we:we + n_elem_fuselage] = 1
        we += n_elem_fuselage
        wn += n_node_fuselage - 1
        global end_of_fuselage_node
        end_of_fuselage_node = wn - 1

        # fin
        beam_number[we:we + n_elem_fin] = 3
        x[wn:wn + n_node_fin - 1] = x[end_of_fuselage_node]
        z[wn:wn + n_node_fin - 1] = z[end_of_fuselage_node] + np.linspace(0.0, fin_height, n_node_fin)[1:]
        for ielem in range(n_elem_fin):
            conn[we + ielem, :] = ((np.ones((3,))*(we + ielem)*(n_node_elem - 1)) +
                                   [0, 2, 1])
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
        conn[we, 0] = end_of_fuselage_node
        elem_stiffness[we:we + n_elem_fin] = 2
        elem_mass[we:we + n_elem_fin] = 2
        we += n_elem_fin
        wn += n_node_fin - 1
        end_of_fin_node = wn - 1

        # right tail
        beam_number[we:we + n_elem_tail] = 4
        x[wn:wn + n_node_tail - 1] = x[end_of_fin_node]
        y[wn:wn + n_node_tail - 1] = np.linspace(0.0, span_tail, n_node_tail)[1:]
        z[wn:wn + n_node_tail - 1] = z[end_of_fin_node]
        for ielem in range(n_elem_tail):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) +
                                   [0, 2, 1])
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
        conn[we, 0] = end_of_fin_node
        elem_stiffness[we:we + n_elem_tail] = 2
        elem_mass[we:we + n_elem_tail] = 2
        boundary_conditions[wn + n_node_tail - 2] = -1
        we += n_elem_tail
        wn += n_node_tail - 1

        # left tail
        beam_number[we:we + n_elem_tail] = 5
        x[wn:wn + n_node_tail - 1] = x[end_of_fin_node]
        y[wn:wn + n_node_tail - 1] = np.linspace(0.0, -span_tail, n_node_tail)[1:]
        z[wn:wn + n_node_tail - 1] = z[end_of_fin_node]
        for ielem in range(n_elem_tail):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) +
                                   [0, 2, 1])
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
        conn[we, 0] = end_of_fin_node
        elem_stiffness[we:we + n_elem_tail] = 2
        elem_mass[we:we + n_elem_tail] = 2
        boundary_conditions[wn + n_node_tail - 2] = -1
        we += n_elem_tail
        wn += n_node_tail - 1

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
        # control surfaces
        n_control_surfaces = 2
        control_surface = np.zeros((n_elem, n_node_elem), dtype=int) - 1
        control_surface_type = np.zeros((n_control_surfaces, ), dtype=int)
        control_surface_deflection = np.zeros((n_control_surfaces, ))
        control_surface_chord = np.zeros((n_control_surfaces, ), dtype=int)
        control_surface_hinge_coord = np.zeros((n_control_surfaces, ), dtype=float)

        # control surface type 0 = static
        # control surface type 1 = dynamic
        control_surface_type[0] = 0
        control_surface_deflection[0] = cs_deflection
        control_surface_chord[0] = m
        control_surface_hinge_coord[0] = -0.25 # nondimensional wrt elastic axis (+ towards the trailing edge)

        control_surface_type[1] = 0
        control_surface_deflection[1] = rudder_static_deflection
        control_surface_chord[1] = 1
        control_surface_hinge_coord[1] = -0. # nondimensional wrt elastic axis (+ towards the trailing edge)

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
        # airfoil_distribution[wn:wn + n_node_main - 1] = 0
        surface_distribution[we:we + n_elem_main] = i_surf
        surface_m[i_surf] = m
        aero_node[wn:wn + n_node_main - 1] = True
        # chord[wn:wn + num_node_main - 1] = np.linspace(main_chord, main_tip_chord, num_node_main)[1:]
        # chord[wn:wn + num_node_main - 1] = main_chord
        # elastic_axis[wn:wn + num_node_main - 1] = main_ea
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

        we += n_elem_fuselage
        wn += n_node_fuselage - 1 - 1
        
        # fin (surface 2, beam 3)
        i_surf = 2
        airfoil_distribution[we:we + n_elem_fin, :] = 1
        # airfoil_distribution[wn:wn + n_node_fin] = 0
        surface_distribution[we:we + n_elem_fin] = i_surf
        surface_m[i_surf] = m
        aero_node[wn:wn + n_node_fin] = True
        # chord[wn:wn + num_node_fin] = fin_chord
        for i_elem in range(we, we + n_elem_fin):
            for i_local_node in range(n_node_elem):
                chord[i_elem, i_local_node] = chord_fin
                elastic_axis[i_elem, i_local_node] = ea_fin
                control_surface[i_elem, i_local_node] = 1
                # twist[end_of_fuselage_node] = 0
                # twist[wn:] = 0
                # elastic_axis[wn:wn + num_node_main] = fin_ea
        we += n_elem_fin
        wn += n_node_fin - 1
        #
        # # # right tail (surface 3, beam 4)
        i_surf = 3
        airfoil_distribution[we:we + n_elem_tail, :] = 2
        # airfoil_distribution[wn:wn + n_node_tail] = 0
        surface_distribution[we:we + n_elem_tail] = i_surf
        surface_m[i_surf] = m
        # XXX not very elegant
        aero_node[wn:] = True
        # chord[wn:wn + num_node_tail] = tail_chord
        # elastic_axis[wn:wn + num_node_main] = tail_ea
        for i_elem in range(we, we + n_elem_tail):
            for i_local_node in range(n_node_elem):
                twist[i_elem, i_local_node] = -0
        for i_elem in range(we, we + n_elem_tail):
            for i_local_node in range(n_node_elem):
                chord[i_elem, i_local_node] = chord_tail
                elastic_axis[i_elem, i_local_node] = ea_tail
                control_surface[i_elem, i_local_node] = 0

        we += n_elem_tail
        wn += n_node_tail
        #
        # # left tail (surface 4, beam 5)
        i_surf = 4
        airfoil_distribution[we:we + n_elem_tail, :] = 2
        # airfoil_distribution[wn:wn + n_node_tail - 1] = 0
        surface_distribution[we:we + n_elem_tail] = i_surf
        surface_m[i_surf] = m
        aero_node[wn:wn + n_node_tail - 1] = True
        # chord[wn:wn + num_node_tail] = tail_chord
        # elastic_axis[wn:wn + num_node_main] = tail_ea
        # twist[we:we + num_elem_tail] = -tail_twist
        for i_elem in range(we, we + n_elem_tail):
            for i_local_node in range(n_node_elem):
                twist[i_elem, i_local_node] = -0
        for i_elem in range(we, we + n_elem_tail):
            for i_local_node in range(n_node_elem):
                chord[i_elem, i_local_node] = chord_tail
                elastic_axis[i_elem, i_local_node] = ea_tail
                control_surface[i_elem, i_local_node] = 0
        we += n_elem_tail
        wn += n_node_tail


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

            control_surface_input = h5file.create_dataset('control_surface', data=control_surface)
            control_surface_deflection_input = h5file.create_dataset('control_surface_deflection', data=control_surface_deflection)
            control_surface_chord_input = h5file.create_dataset('control_surface_chord', data=control_surface_chord)
            control_surface_hinge_coord_input = h5file.create_dataset('control_surface_hinge_coord', data=control_surface_hinge_coord)
            control_surface_types_input = h5file.create_dataset('control_surface_type', data=control_surface_type)


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

    def generate_multibody_file():
        global end_of_fuselage_node
        global LC
        # LCR = gc.LagrangeConstraint()
        # LCR.behaviour = 'lin_vel_node_wrtG'
        # LCR.velocity = np.zeros((3,))
        # LCR.body_number = 0
        # LCR.node_number = flat_end_node[0]

        # LCL = gc.LagrangeConstraint()
        # LCL.behaviour = 'lin_vel_node_wrtG'
        # LCL.velocity = np.zeros((3,))
        # LCL.body_number = 0
        # LCL.node_number = flat_end_node[1]

        LCF = gc.LagrangeConstraint()
        LCF.behaviour = 'lin_vel_node_wrtG'
        LCF.velocity = np.zeros((3,))
        LCF.body_number = 0
        LCF.node_number = end_of_fuselage_node

        LCCR = gc.LagrangeConstraint()
        LCCR.behaviour = 'lin_vel_node_wrtG'
        LCCR.velocity = np.zeros((3,))
        LCCR.body_number = 0
        LCCR.node_number = first_node_centre[0]

        LCCL = gc.LagrangeConstraint()
        LCCL.behaviour = 'lin_vel_node_wrtG'
        LCCL.velocity = np.zeros((3,))
        LCCL.body_number = 0
        LCCL.node_number = first_node_centre[1]

        # LC = [LCR, LCL, LCF, LCCR, LCCL]
        LC = [LCF, LCCR, LCCL]

        MB1 = gc.BodyInformation()
        MB1.body_number = 0
        MB1.FoR_position = np.zeros((6,))
        MB1.FoR_velocity = np.zeros((6,))
        MB1.FoR_acceleration = np.zeros((6,))
        MB1.FoR_movement = 'free'
        MB1.quat = np.array([1.0, 0.0, 0.0, 0.0])

        MB = [MB1]
        gc.generate_multibody_file(LC, MB, route, case_name)


    def generate_solver_file():
        file_name = route + '/' + case_name + '.sharpy'
        settings = dict()
        settings['SHARPy'] = {'case': case_name,
                              'route': route,
                              'flow': flow,
                              'write_screen': 'off',
                              'write_log': 'on',
                              'log_folder': route + '/output/',
                              'log_file': case_name + '.log'}

        settings['BeamLoader'] = {'unsteady': 'on',
                                  'orientation':
                                      algebra.euler2quat(np.array([roll,
                                                                   alpha,
                                                                   beta]))}
        settings['AerogridLoader'] = {'unsteady': 'on',
                                      'aligned_grid': 'on',
                                      # 'mstar': int(160/tstep_factor),
                                      'mstar': int(100/tstep_factor),
                                      'freestream_dir': ['1', '0', '0'],
                                      'control_surface_deflection': ['', ''],
                                      'control_surface_deflection_generator':
                                      {'0': {},
                                       '1': {}}}

        settings['NonLinearStatic'] = {'print_info': 'off',
                                       'max_iterations': 150,
                                       'num_load_steps': 1,
                                       'delta_curved': 1e-1,
                                       'min_delta': tolerance,
                                       'gravity_on': gravity,
                                       'gravity': 0*9.81}

        settings['StaticUvlm'] = {'print_info': 'off',
                                  'horseshoe': 'off',
                                  'num_cores': num_cores,
                                  'n_rollup': 0,
                                  'rollup_dt': dt,
                                  'rollup_aic_refresh': 1,
                                  'rollup_tolerance': 1e-4,
                                  'velocity_field_generator': 'SteadyVelocityField',
                                  'velocity_field_input': {'u_inf': u_background,
                                                           'u_inf_direction': [1., 0, 0]},
                                  'rho': 0*rho}

        settings['StaticCoupled'] = {'print_info': 'off',
                                     'structural_solver': 'NonLinearStatic',
                                     'structural_solver_settings':
                                          settings['NonLinearStatic'],
                                     'aero_solver': 'StaticUvlm',
                                     'aero_solver_settings': settings['StaticUvlm'],
                                     'max_iter': 100,
                                     'n_load_steps': n_step,
                                     'tolerance': fsi_tolerance,
                                     'relaxation_factor': relaxation_factor}

        settings['StaticTrim'] = {'solver': 'StaticCoupled',
                                  'solver_settings': settings['StaticCoupled'],
                                  'initial_alpha': alpha,
                                  'initial_deflection': cs_deflection,
                                  'initial_thrust': thrust}

        settings['NonLinearDynamicCoupledStep'] = {'print_info': 'off',
                                                   'max_iterations': 950,
                                                   'delta_curved': 1e-1,
                                                   'min_delta': tolerance,
                                                   'newmark_damp': 1e-2,
                                                   'gravity_on': gravity,
                                                   'gravity': 9.81,
                                                   'num_steps': n_tstep,
                                                   'dt': dt,
                                                   'initial_velocity': 0}

        settings['NonLinearDynamicMultibody'] = {'print_info': 'off',
                                                 'max_iterations': 950,
                                                 'delta_curved': 1e-1,
                                                 'min_delta': tolerance,
                                                 'newmark_damp': 1e-2,
                                                 'gravity_on': gravity,
                                                 'gravity': 9.81,
                                                 'num_steps': n_tstep,
                                                 'dt': dt}

        relative_motion = 'off'
        settings['StepUvlm'] = {'print_info': 'off',
                                'horseshoe': 'off',
                                'num_cores': num_cores,
                                'n_rollup': 0,
                                'convection_scheme': 2,
                                'rollup_dt': dt,
                                'rollup_aic_refresh': 1,
                                'rollup_tolerance': 1e-4,
                                'gamma_dot_filtering': 6,
                                'velocity_field_generator': 'SteadyVelocityField',
                                'velocity_field_input': {'u_inf': u_background,
                                                         'u_inf_direction': [1., 0, 0]},
                                'rho': rho,
                                'n_time_steps': n_tstep,
                                'dt': dt}

        solver = 'NonLinearDynamicMultibody'
        settings['PickleData'] = {'folder': './' + case_name + '/'}
        settings['DynamicCoupled'] = {'structural_solver': solver,
                                      'structural_solver_settings': settings[solver],
                                      'aero_solver': 'StepUvlm',
                                      'aero_solver_settings': settings['StepUvlm'],
                                      'fsi_substeps': 200,
                                      'fsi_tolerance': fsi_tolerance,
                                      'relaxation_factor': relaxation_factor,
                                      'minimum_steps': 2,
                                      'relaxation_steps': 150,
                                      'dynamic_relaxation': 'off',
                                      'final_relaxation_factor': 0.5,
                                      'n_time_steps': n_tstep,
                                      'dt': dt,
                                      'structural_substeps': structural_substeps,
                                      'include_unsteady_force_contribution': 'on',
                                      'steps_without_unsteady_force': 9,
                                      'controller_id': {#'controller_right': 'TakeOffTrajectoryController',
                                                        # 'controller_left': 'TakeOffTrajectoryController',
                                                        'controller_tail': 'TakeOffTrajectoryController',
                                                        'controller_cright': 'TakeOffTrajectoryController',
                                                        'controller_cleft': 'TakeOffTrajectoryController',
                                                        },
                                      'controller_settings': {
                                          # 'controller_right':
                                      # {
                                          # 'trajectory_input_file': trajectory_file,
                                          # 'dt': dt,
                                          # 'trajectory_method': 'lagrange',
                                          # 'controlled_constraint': 'constraint_00',
                                          # 'initial_ramp_length_structural_substeps': controller_ramp,
                                          # 'write_controller_log': 'off',
                                      # },
                                          # 'controller_left':
                                      # {
                                          # 'trajectory_input_file': trajectory_file,
                                          # 'dt': dt,
                                          # 'trajectory_method': 'lagrange',
                                          # 'controlled_constraint': 'constraint_01',
                                          # 'initial_ramp_length_structural_substeps': controller_ramp,
                                          # 'write_controller_log': 'off',
                                      # },
                                      'controller_tail':
                                      {
                                          'trajectory_input_file': trajectory_file,
                                          'dt': dt,
                                          'trajectory_method': 'lagrange',
                                          'controlled_constraint': 'constraint_00',
                                          'initial_ramp_length_structural_substeps': controller_ramp,
                                          'write_controller_log': 'off',
                                      }, 'controller_cright':
                                      {
                                          'trajectory_input_file': trajectory_file,
                                          'dt': dt,
                                          'trajectory_method': 'lagrange',
                                          'controlled_constraint': 'constraint_01',
                                          'initial_ramp_length_structural_substeps': controller_ramp,
                                          'write_controller_log': 'off',
                                      }, 'controller_cleft':
                                      {
                                          'trajectory_input_file': trajectory_file,
                                          'dt': dt,
                                          'trajectory_method': 'lagrange',
                                          'controlled_constraint': 'constraint_02',
                                          'initial_ramp_length_structural_substeps': controller_ramp,
                                          'write_controller_log': 'off',
                                      }},
                                      'postprocessors': ['BeamLoads'],
                                      'postprocessors_settings': {'BeamLoads': {'folder': route + '/output/',
                                                                                'csv_output': 'off'},
                                                                  'BeamPlot': {'folder': route + '/output/',
                                                                               'include_rbm': 'on',
                                                                               'include_applied_forces': 'on'},
                                                                  'AerogridPlot': {
                                                                      'folder': route + '/output/',
                                                                      'include_rbm': 'on',
                                                                      'include_applied_forces': 'on',
                                                                      'minus_m_star': 0},
                                                                 }
                                      }

        settings['BeamLoads'] = {'folder': route + '/output/',
                                 'csv_output': 'off'}

        settings['BeamPlot'] = {'folder': route + '/output/',
                                'include_rbm': 'on',
                                'include_applied_forces': 'on',
                                'include_forward_motion': 'on'}

        settings['AerogridPlot'] = {'folder': route + '/output/',
                                    'include_rbm': 'on',
                                    'include_forward_motion': 'off',
                                    'include_applied_forces': 'on',
                                    'minus_m_star': 0,
                                    'u_inf': 0,
                                    'dt': dt}

        settings['Notes'] = {'note': case_notes}

        import configobj
        config = configobj.ConfigObj()
        config.filename = file_name
        for k, v in settings.items():
            config[k] = v
        config.write()

    gc.clean_test_files(route, case_name)
    generate_fem()
    generate_multibody_file()
    generate_aero_file()
    generate_solver_file()
    generate_dyn_file()
    if 'StaticTrim' not in flow:
        generate_trajectory_file()

    return {'sharpy': route + '/' + case_name + '.sharpy'}


if __name__ == "__main__":
    generate()
