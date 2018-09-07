import h5py as h5
import numpy as np
import configparser
import os
import sharpy.utils.algebra as algebra
# Generate errors during execution
import sys

case_name = 'bielawa_baromega2_1e3'
route = os.path.dirname(os.path.realpath(__file__)) + '/'

num_elem = 40
num_node_elem = 3

length = 1

# linear_factor: scaling factor to make the non linear solver behave as a linear one
linear_factor=1
E=1e6*linear_factor
A=1e4
I=1e-4
ei = E*I
m_bar = 1*linear_factor

rot_speed=np.sqrt(1e3*ei/m_bar/length**4)

steps_per_revolution = 180
dt = 2.0*np.pi/rot_speed/steps_per_revolution
n_tstep = 1*steps_per_revolution+1

n_tstep = 90

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

def generate_fem_file(route, case_name, num_elem, num_node_elem=3):

    global num_node
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

    ea = E*A
    # APPROXIMATION!!!
    print("WARNING: Assuming isotropic material")
    G = E / 2.0 / (1.0+0.3)
    print("WARNING: Using total cross-section area as shear area")
    ga = G*A
    print("WARNING: Assuming planar cross-sections")
    J = 2.0* I
    gj = G*J

    base_stiffness = np.diag([ea, ga, ga, gj, ei, ei])
    stiffness = np.zeros((num_stiffness, 6, 6))
    # import pdb; pdb.set_trace()
    for i in range(num_stiffness):
        stiffness[i, :, :] = base_stiffness
    # element stiffness
    elem_stiffness = np.zeros((num_elem,), dtype=int)

    # mass array
    num_mass = 1
    base_mass = m_bar*np.diag([1.0, 1.0, 1.0, J/A, I/A, I/A])
    # base_mass = m_bar*np.diag([1.0, 1.0, 1.0, 1.0,1.0,1.0])
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
    beam_number = np.zeros((num_node, 1), dtype=int)

    # new app forces scheme (only follower)
    app_forces = np.zeros((num_node, 6))
    # app_forces[0, :] = [0, 0, 3000000, 0, 0, 0]

    # lumped masses input
    n_lumped_mass = 1
    lumped_mass_nodes = np.array([num_node - 1], dtype=int)
    lumped_mass = np.zeros((n_lumped_mass, ))
    lumped_mass[0] = 0.0
    lumped_mass_inertia = np.zeros((n_lumped_mass, 3, 3))
    lumped_mass_position = np.zeros((n_lumped_mass, 3))

    #n_lumped_mass = 1
    #lumped_mass_nodes = np.ones((num_node,), dtype=int)
    #lumped_mass = np.zeros((n_lumped_mass, ))
    #lumped_mass[0] = m_bar*length/num_elem/(num_node_elem-1)
    #lumped_mass_inertia[0,:,:] = np.diag([J, I, I])
    #lumped_mass_position = np.zeros((n_lumped_mass, 3))

    with h5.File(route + '/' + case_name + '.fem.h5', 'a') as h5file:

        # CHECKING
        if(elem_stiffness.shape[0]!=num_elem):
            sys.exit("ERROR: Element stiffness must be defined for each element")
        if(elem_mass.shape[0]!=num_elem):
            sys.exit("ERROR: Element mass must be defined for each element")
        if(frame_of_reference_delta.shape[0]!=num_elem):
            sys.exit("ERROR: The first dimension of FoR does not match the number of elements")
        if(frame_of_reference_delta.shape[1]!=num_node_elem):
            sys.exit("ERROR: The second dimension of FoR does not match the number of nodes element")
        if(frame_of_reference_delta.shape[2]!=3):
            sys.exit("ERROR: The third dimension of FoR must be 3")
        if(structural_twist.shape[0]!=num_node):
            sys.exit("ERROR: The structural twist must be defined for each node")
        if(boundary_conditions.shape[0]!=num_node):
            sys.exit("ERROR: The boundary conditions must be defined for each node")
        if(beam_number.shape[0]!=num_node):
            sys.exit("ERROR: The beam number must be defined for each node")
        if(app_forces.shape[0]!=num_node):
            sys.exit("ERROR: The first dimension of the applied forces matrix does not match the number of nodes")
        if(app_forces.shape[1]!=6):
            sys.exit("ERROR: The second dimension of the applied forces matrix must be 6")

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

def generate_dyn_file():
    global num_node

    forced_for_vel = np.zeros((n_tstep, 6))
    dynamic_forces_time = np.zeros((n_tstep, num_node,6))
    for it in range(n_tstep):
        # forced_for_vel[it, 3:6] = it/n_tstep*angular_velocity
        forced_for_vel[it, 5] = rot_speed
        # dynamic_forces_time[it,-1,2] = 100



    with h5.File(route + '/' + case_name + '.dyn.h5', 'a') as h5file:
        h5file.create_dataset(
            'dynamic_forces', data=dynamic_forces_time)
        h5file.create_dataset(
            'for_vel', data=forced_for_vel)
        h5file.create_dataset(
            'num_steps', data=n_tstep)

def generate_aero_file():
    global num_node
    with h5.File(route + '/' + case_name + '.aero.h5', 'a') as h5file:
        airfoils_group = h5file.create_group('airfoils')
        # add the airfoils
        airfoils_group.create_dataset("0", data = np.column_stack( (np.linspace( 0.0, 1.0, 10), np.zeros(10) )) )

        # chord
        chord_input = h5file.create_dataset('chord', data= np.ones((num_elem,num_node_elem),))
        dim_attr = chord_input .attrs['units'] = 'm'

        # twist
        twist_input = h5file.create_dataset('twist', data=np.zeros((num_elem,num_node_elem),))
        dim_attr = twist_input.attrs['units'] = 'rad'

        # airfoil distribution
        airfoil_distribution_input = h5file.create_dataset('airfoil_distribution', data=np.zeros((num_elem,num_node_elem),dtype=int))

        surface_distribution_input = h5file.create_dataset('surface_distribution', data=np.zeros((num_elem,),dtype=int))
        surface_m_input = h5file.create_dataset('surface_m', data = np.ones((1,),dtype=int))
        m_distribution = 'uniform'
        m_distribution_input = h5file.create_dataset('m_distribution', data=m_distribution.encode('ascii', 'ignore'))

        aero_node = np.zeros((num_node,),dtype=bool)
        aero_node[-3:] = np.ones((3,),dtype=bool)
        aero_node_input = h5file.create_dataset('aero_node', data=aero_node)
        elastic_axis_input = h5file.create_dataset('elastic_axis', data=0.5*np.ones((num_elem,num_node_elem),))

def generate_solver_file():
    file_name = route + '/' + case_name + '.solver.txt'

    aux_settings = dict()
    settings = dict()

    settings['SHARPy'] = {'case': case_name,
                        'route': route,
                        'flow': ['BeamLoader', 'AerogridLoader', 'StaticCoupled', 'BeamPlot', 'AerogridPlot',  'DynamicPrescribedCoupled', 'Modal'],
                        'write_screen': 'off',
                        'write_log': 'on',
                        'log_folder': route + '/output/',
                        'log_file': case_name + '.log'}

    # AUX DICTIONARIES
    aux_settings['velocity_field_input'] = {'u_inf': 100.0,
                                            'u_inf_direction': [0.0, -1.0, 0.0]}

    # LOADERS
    settings['BeamLoader'] = {'unsteady': 'on',
                              'orientation': algebra.euler2quat(np.array([0.0,0.0,0.0]))}

    settings['AerogridLoader'] = {'unsteady': 'on',
                                  'aligned_grid': 'on',
                                  'mstar': 1,
                                  'freestream_dir': ['0', '-1', '0']}

    # POSTPROCESS
    settings['AerogridPlot'] = {'folder': route + '/output/',
                                'include_rbm': 'on',
                                'include_forward_motion': 'off',
                                'include_applied_forces': 'on',
                                'minus_m_star': 0,
                                'u_inf': 100.0,
                                'dt': dt}

    #settings['AeroForcesCalculator'] = {'folder': route + '/output/forces',
                                        #'write_text_file': 'on',
                                        #'text_file_name': case_name + '_aeroforces.csv',
                                        #'screen_output': 'on',
                                        #'unsteady': 'off'}

    settings['BeamPlot'] = {'folder': route + '/output/',
                            'include_rbm': 'on',
                            'include_applied_forces': 'on',
                            'include_forward_motion': 'on'}

    #settings['BeamCsvOutput'] = {'folder': route + '/output/',
                                 #'output_pos': 'on',
                                 #'output_psi': 'on',
                                 #'screen_output': 'off'}

    settings['BeamLoads'] = {}


    # STATIC COUPLED

    settings['NonLinearStatic'] = {'print_info': 'on',
                                   'max_iterations': 150,
                                   'num_load_steps': 1,
                                   'delta_curved': 1e-15,
                                   'min_delta': 1e-8,
                                   'gravity_on': 'off',
                                   'gravity': 9.81}

    settings['StaticUvlm'] = {'print_info': 'on',
                              'horseshoe': 'off',
                              'num_cores': 4,
                              'n_rollup': 0,
                              'rollup_dt': dt,
                              'rollup_aic_refresh': 1,
                              'rollup_tolerance': 1e-4,
                              'velocity_field_generator': 'SteadyVelocityField',
                              'velocity_field_input': aux_settings['velocity_field_input'],
                              'rho': 0.0}

    settings['StaticCoupled'] = {'print_info': 'on',
                                 'structural_solver': 'NonLinearStatic',
                                 'structural_solver_settings': settings['NonLinearStatic'],
                                 'aero_solver': 'StaticUvlm',
                                 'aero_solver_settings': settings['StaticUvlm'],
                                 'max_iter': 100,
                                 'n_load_steps': 4,
                                 'tolerance': 1e-8,
                                 'relaxation_factor': 0}

    # DYNAMIC PRESCRIBED COUPLED

    settings['NonLinearDynamicPrescribedStep'] = {'print_info': 'on',
                                               'max_iterations': 95000,
                                               'delta_curved': 1e-9,
                                               'min_delta': 1e-6,
                                               'newmark_damp': 1e-3,
                                               'gravity_on': 'off',
                                               'gravity': 9.81,
                                               'num_steps': n_tstep,
                                               'dt': dt}

    settings['StepUvlm'] = {'print_info': 'on',
                            'horseshoe': 'off',
                            'num_cores': 4,
                            'n_rollup': 0,
                            'convection_scheme': 2,
                            'rollup_dt': dt,
                            'rollup_aic_refresh': 1,
                            'rollup_tolerance': 1e-4,
                            'velocity_field_generator': 'SteadyVelocityField',
                            'velocity_field_input': aux_settings['velocity_field_input'],
                            'rho': 0.0,
                            'n_time_steps': n_tstep,
                            'dt': dt}

    settings['DynamicPrescribedCoupled'] = {'structural_solver': 'NonLinearDynamicPrescribedStep',
                                            'structural_solver_settings': settings['NonLinearDynamicPrescribedStep'],
                                            'aero_solver': 'StepUvlm',
                                            'aero_solver_settings': settings['StepUvlm'],
                                            'fsi_substeps': 20000,
                                            'fsi_tolerance': 1e-9,
                                            'relaxation_factor': 0,
                                            'minimum_steps': 1,
                                            'relaxation_steps': 150,
                                            'final_relaxation_factor': 0.0,
                                            'n_time_steps': n_tstep,
                                            'dt': dt,
                                            'postprocessors': ['BeamPlot', 'AerogridPlot'],
                                            'postprocessors_settings': {'BeamPlot': settings['BeamPlot'],
                                                                        'AerogridPlot': settings['AerogridPlot']}}

    settings['Modal'] = {'folder': route + '/output',
                          'include_rbm': 'on',
                          'NumLambda': 10000,
                          'num_steps': 1,
                          'print_matrices': 'on'}

    import configobj
    config = configobj.ConfigObj()
    config.filename = file_name
    for k, v in settings.items():
        config[k] = v
    config.write()

# run everything
clean_test_files()
generate_fem_file(route, case_name, num_elem, num_node_elem)
generate_aero_file()
generate_dyn_file()
generate_solver_file()

print('Reference for validation: "Rotary wing structural dynamics and aeroelasticity", R.L. Bielawa. AIAA education series. Second edition')
