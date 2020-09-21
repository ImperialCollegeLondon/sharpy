#! /usr/bin/env python3
import h5py as h5
import numpy as np
import os
import sharpy.utils.algebra as algebra

# alpha_deg = 0

def execute_sharpy_simulations(test_dir):
    for alpha_deg in np.linspace(0, 5, 6):
        case_name = 'simple_HALE_uvlm_alpha{:04g}'.format(alpha_deg * 100)

        route = os.path.dirname(os.path.realpath(__file__)) + '/'


        # EXECUTION
        flow = ['BeamLoader',
                'AerogridLoader',
                'StaticUvlm',
                'AeroForcesCalculator',
                'SaveParametricCase'
                ]

        # if free_flight is False, the motion of the centre of the wing is prescribed.
        free_flight = True
        if not free_flight:
            case_name += '_prescribed'
            amplitude = 0*np.pi/180
            period = 3
            case_name += '_amp_' + str(amplitude).replace('.', '') + '_period_' + str(period)

        lumped_mass_factor = 1
        # case_name += '_lm{:g}'.format(lumped_mass_factor)

        ## ROM
        rom = True
        # linear settings
        num_modes = 20
        # case_name += '_rom{:g}_nmodes{:g}'.format(rom, num_modes)

        # FLIGHT CONDITIONS
        # the simulation is set such that the aircraft flies at a u_inf velocity while
        # the air is calm.
        u_inf = 10
        rho = 1.225

        # trim sigma = 1.5
        alpha = alpha_deg * np.pi/180
        beta = 0
        roll = 0
        gravity = 'on'
        cs_deflection = 0 #-2.0249*np.pi/180
        rudder_static_deflection = 0.0
        rudder_step = 0.0*np.pi/180
        thrust = 5.9443
        sigma = 1.5
        lambda_dihedral = 20*np.pi/180

        # 8 | 4.1515 | -1.1249 | 4.7300 | 0.0088 | -0.0000 | 0.0005 | 0.0000 | -0.0005 | 0.0000
        # m = 16
        # alpha = 4.1515 * np.pi / 180
        # cs_deflection = -1.1249 * np.pi / 180
        # thrust  = 4.7300


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
        span_main = 16.0
        lambda_main = 0.25
        ea_main = 0.3

        ea = 1e7
        ga = 1e5
        gj = 1e4
        eiy = 2e4
        eiz = 4e6
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
        sigma_tail = 100
        m_bar_tail = 0.3
        j_bar_tail = 0.08

        # lumped masses
        n_lumped_mass = 1
        lumped_mass_nodes = np.zeros((n_lumped_mass, ), dtype=int)
        lumped_mass = np.zeros((n_lumped_mass, ))
        lumped_mass[0] = 50 * lumped_mass_factor
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
        m_star_factor = 10
        # spanwise elements
        n_elem_multiplier = 2
        n_elem_main = int(4*n_elem_multiplier)
        n_elem_tail = int(2*n_elem_multiplier)
        n_elem_fin = int(2*n_elem_multiplier)
        n_elem_fuselage = int(2*n_elem_multiplier)
        n_surfaces = 5

        # temporal discretisation
        physical_time = 30
        tstep_factor = 1.
        dt = 1.0/m/u_inf*tstep_factor
        n_tstep = round(physical_time/dt)


        # linear files
        elevator = 30 * np.pi / 180
        rudder = 25 * np.pi / 180

        # END OF INPUT-----------------------------------------------------------------

        # case files folder
        cases_folder = route + '/cases/' + case_name + '/'
        if not os.path.isdir(cases_folder):
            os.makedirs(cases_folder, exist_ok=True)

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
        twist = np.zeros((n_elem, n_node_elem))
        sweep = np.zeros((n_elem, n_node_elem))
        chord = np.zeros((n_elem, n_node_elem,))
        elastic_axis = np.zeros((n_elem, n_node_elem,))

        # linear time domain vectors
        # for m = 16 only
        input_vec = np.zeros((10, num_modes * 3 + 4))
        x0 = np.zeros(10)
        input_vec[5:, 2 * num_modes] = elevator
        input_vec[5:, 2 * num_modes + 1] = rudder


        # FUNCTIONS-------------------------------------------------------------
        def clean_test_files():
            fem_file_name = cases_folder + case_name + '.fem.h5'
            if os.path.isfile(fem_file_name):
                os.remove(fem_file_name)

            dyn_file_name = cases_folder + case_name + '.dyn.h5'
            if os.path.isfile(dyn_file_name):
                os.remove(dyn_file_name)

            aero_file_name = cases_folder + case_name + '.aero.h5'
            if os.path.isfile(aero_file_name):
                os.remove(aero_file_name)

            solver_file_name = cases_folder + case_name + '.sharpy'
            if os.path.isfile(solver_file_name):
                os.remove(solver_file_name)

            flightcon_file_name = cases_folder + case_name + '.flightcon.txt'
            if os.path.isfile(flightcon_file_name):
                os.remove(flightcon_file_name)

            linear_files = cases_folder + case_name + '.lininput.h5'
            if os.path.isfile(linear_files):
                os.remove(linear_files)

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
            global free_flight

            dynamic_forces_time = None
            with_dynamic_forces = False
            with_forced_vel = False
            if not free_flight:
                with_forced_vel = True

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
                with h5.File(cases_folder + case_name + '.dyn.h5', 'a') as h5file:
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

            with h5.File(cases_folder + case_name + '.fem.h5', 'a') as h5file:
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
            control_surface_chord[0] = m # m
            control_surface_hinge_coord[0] = -0.25 * 1 # nondimensional wrt elastic axis (+ towards the trailing edge)

            control_surface_type[1] = 0
            control_surface_deflection[1] = rudder_static_deflection
            control_surface_chord[1] = m // 2 #1
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
            #
            # # fin (surface 2, beam 3)
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

            #remove last elem of the control surface
            control_surface[i_elem, :] = -1

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


            with h5.File(cases_folder + case_name + '.aero.h5', 'a') as h5file:
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


        def generate_linear_sim_files(x0, input_vec):

            with h5.File(cases_folder + '/' + case_name + '.lininput.h5', 'a') as h5file:
                x0 = h5file.create_dataset(
                    'x0', data=x0)
                u = h5file.create_dataset(
                    'u', data=input_vec)


        def generate_solver_file():
            file_name = cases_folder + case_name + '.sharpy'
            settings = dict()
            settings['SHARPy'] = {'case': case_name,
                                  'route': cases_folder,
                                  'flow': flow,
                                  'write_screen': 'on',
                                  'write_log': 'on',
                                  'log_folder': route + '/output/' + case_name,
                                  'log_file': case_name + '.log'}

            settings['BeamLoader'] = {'unsteady': 'on',
                                      'orientation': algebra.euler2quat(np.array([roll,
                                                                                  alpha,
                                                                                  beta]))}
            settings['AerogridLoader'] = {'unsteady': 'on',
                                          'aligned_grid': 'on',
                                          'mstar': m_star_factor * m,
                                          'freestream_dir': ['1', '0', '0']}

            settings['NonLinearStatic'] = {'print_info': 'off',
                                           'max_iterations': 150,
                                           'num_load_steps': 1,
                                           'delta_curved': 1e-1,
                                           'min_delta': tolerance,
                                           'gravity_on': gravity,
                                           'gravity': 9.81}

            settings['StaticUvlm'] = {'print_info': 'on',
                                      'horseshoe': 'off',
                                      'num_cores': num_cores,
                                      'n_rollup': 0,
                                      'rollup_dt': dt,
                                      'rollup_aic_refresh': 1,
                                      'vortex_radius': 1e-8,
                                      'rollup_tolerance': 1e-4,
                                      'velocity_field_generator': 'SteadyVelocityField',
                                      'velocity_field_input': {'u_inf': u_inf,
                                                               'u_inf_direction': [1., 0, 0]},
                                      'rho': rho}

            settings['StaticCoupled'] = {'print_info': 'off',
                                         'structural_solver': 'NonLinearStatic',
                                         'structural_solver_settings': settings['NonLinearStatic'],
                                         'aero_solver': 'StaticUvlm',
                                         'aero_solver_settings': settings['StaticUvlm'],
                                         'max_iter': 100,
                                         'n_load_steps': n_step,
                                         'tolerance': fsi_tolerance,
                                         'relaxation_factor': structural_relaxation_factor}

            settings['StaticTrim'] = {'solver': 'StaticCoupled',
                                      'solver_settings': settings['StaticCoupled'],
                                      'initial_alpha': alpha,
                                      'initial_deflection': cs_deflection,
                                      'initial_thrust': thrust}

            settings['NonLinearDynamicCoupledStep'] = {'print_info': 'off',
                                                       'max_iterations': 950,
                                                       'delta_curved': 1e-1,
                                                       'min_delta': tolerance,
                                                       'newmark_damp': 5e-3,
                                                       'gravity_on': gravity,
                                                       'gravity': 9.81,
                                                       'num_steps': n_tstep,
                                                       'dt': dt,
                                                       'initial_velocity': u_inf}

            settings['NonLinearDynamicPrescribedStep'] = {'print_info': 'off',
                                                   'max_iterations': 950,
                                                   'delta_curved': 1e-1,
                                                   'min_delta': tolerance,
                                                   'newmark_damp': 5e-3,
                                                   'gravity_on': gravity,
                                                   'gravity': 9.81,
                                                   'num_steps': n_tstep,
                                                   'dt': dt,
                                                   'initial_velocity': u_inf*int(free_flight)}

            relative_motion = 'off'
            if not free_flight:
                relative_motion = 'on'
            settings['StepUvlm'] = {'print_info': 'off',
                                    'horseshoe': 'off',
                                    'num_cores': num_cores,
                                    'n_rollup': 0,
                                    'convection_scheme': 2,
                                    'rollup_dt': dt,
                                    'rollup_aic_refresh': 1,
                                    'rollup_tolerance': 1e-4,
                                    'gamma_dot_filtering': 6,
                                    'vortex_radius': 1e-8,
                                    'velocity_field_generator': 'GustVelocityField',
                                    'velocity_field_input': {'u_inf': int(not free_flight)*u_inf,
                                                             'u_inf_direction': [1., 0, 0],
                                                             'gust_shape': '1-cos',
                                                             'gust_length': gust_length,
                                                             'gust_intensity': gust_intensity*u_inf,
                                                             'offset': gust_offset,
                                                             'span': span_main,
                                                             'relative_motion': relative_motion},
                                    'rho': rho,
                                    'n_time_steps': n_tstep,
                                    'dt': dt}

            if free_flight:
                solver = 'NonLinearDynamicCoupledStep'
            else:
                solver = 'NonLinearDynamicPrescribedStep'
            settings['DynamicCoupled'] = {'structural_solver': solver,
                                          'structural_solver_settings': settings[solver],
                                          'aero_solver': 'StepUvlm',
                                          'aero_solver_settings': settings['StepUvlm'],
                                          'fsi_substeps': 200,
                                          'fsi_tolerance': fsi_tolerance,
                                          'relaxation_factor': relaxation_factor,
                                          'minimum_steps': 1,
                                          'relaxation_steps': 150,
                                          'final_relaxation_factor': 0.5,
                                          'n_time_steps': 1,
                                          'dt': dt,
                                          'include_unsteady_force_contribution': 'off',
                                          'postprocessors': ['BeamLoads', 'BeamPlot', 'AerogridPlot'],
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
                                                                      }}

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
                                        'u_inf': u_inf,
                                        'dt': dt}

            settings['AeroForcesCalculator'] = {'folder': route + '/output/',
                                                'write_text_file': 'on',
                                                'screen_output': 'on'}

            settings['Modal'] = {'print_info': 'on',
                                 'use_undamped_modes': 'on',
                                 'NumLambda': num_modes,
                                 'rigid_body_modes': free_flight,
                                 'write_modes_vtk': 'on',
                                 'print_matrices': 'off',
                                 'write_data': 'on',
                                 'rigid_modes_cg': 'on'}

            settings['LinearAssembler'] = {'linear_system': 'LinearAeroelastic',
                                           'linear_system_settings': {
                                               'beam_settings': {'modal_projection': 'on',
                                                                 'inout_coords': 'modes',
                                                                 'discrete_time': 'on',
                                                                 'newmark_damp': 5e-4,
                                                                 'discr_method': 'newmark',
                                                                 'dt': dt,
                                                                 'proj_modes': 'undamped',
                                                                 'use_euler': 'on',
                                                                 'num_modes': num_modes,
                                                                 'print_info': 'on',
                                                                 'gravity': 'on',
                                                                 'remove_dofs': []},
                                               'aero_settings': {'dt': dt,
                                                                 'ScalingDict': {'density': rho,
                                                                                 'length': chord_main * 0.5,
                                                                                 'speed': u_inf},
                                                                 'integr_order': 2,
                                                                 'density': rho,
                                                                 'remove_predictor': 'off',
                                                                 'use_sparse': 'on',
                                                                 'vortex_radius': 1e-8,
                                                                 'remove_inputs': ['u_gust']},
                                               'rigid_body_motion': free_flight,
                                               'use_euler': 'on',
                                           }
                                           }

            if rom:
                settings['LinearAssembler']['linear_system_settings']['aero_settings']['rom_method'] = ['Krylov']
                settings['LinearAssembler']['linear_system_settings']['aero_settings']['rom_method_settings'] = {
                    'Krylov': {'algorithm': 'mimo_rational_arnoldi',
                               'frequency': [0.],
                               'r': 4,
                               'single_side': 'observability'}}

            settings['AsymptoticStability'] = {'sys_id': 'LinearAeroelastic',
                                               'print_info': 'on',
                                               'modes_to_plot': [],
                                               'display_root_locus': 'off',
                                               'frequency_cutoff': 0,
                                               'export_eigenvalues': 'off',
                                               'num_evals': 40,
                                               'folder': route + '/output/'}

            settings['SaveData'] = {'folder': route + '/output/' + case_name + '/',
                                    'save_aero': 'off',
                                    'save_struct': 'off',
                                    'save_linear': 'on',
                                    'save_linear_uvlm': 'on',
                                    'format': 'mat'
                                    }

            settings['LinDynamicSim'] = {'folder': route + '/output/',
                                         'n_tsteps': 10,
                                         'dt': dt,
                                         'postprocessors': ['AerogridPlot'],
                                         'postprocessors_settings':
                                             {'AerogridPlot': {'folder': route + '/output/',
                                                               'include_rbm': 'on',
                                                               'include_applied_forces': 'on',
                                                               'minus_m_star': 0}, }
                                         }

            settings['SaveParametricCase'] = {'folder': route + '/output/' + case_name + '/',
                                          'save_case': 'off',
                                          'parameters': {'alpha': alpha_deg}}

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
        # generate_dyn_file()
        generate_linear_sim_files(x0, input_vec)

        import sharpy.sharpy_main
        sharpy.sharpy_main.main(['', cases_folder + case_name + '.sharpy'])

        forces_g = np.loadtxt(test_dir + '/src/output/{:s}/forces/aeroforces.txt'.format(case_name), skiprows=1,
                              delimiter=',')
        # save forces to common file
        with open(test_dir + '/nonlinear_results.txt', 'a') as f:
            f.write('{:3f},\t{:3f},\t{:3f},\t{:3f}\n'.format(alpha_deg, *forces_g[1:4]))
