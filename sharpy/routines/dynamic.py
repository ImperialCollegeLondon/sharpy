import numpy as np
from sharpy.routines.basic import Basic, update_dic
from sharpy.routines.static import Static
import sharpy.utils.algebra as algebra

class Dynamic(Static, Basic):
    
    predefined_flows = dict()
    #############
    predefined_flows['400'] = ['BeamLoader', 'AerogridLoader',
                               'StaticCoupled',
                               'Modal',
                               'LinearAssembler',
                               'DynamicLoads']
    #############
    predefined_flows['146'] = ['BeamLoader', 'AerogridLoader',
                               'StaticCoupled',
                               'Modal',
                               'LinearAssembler',
                               'DynamicLoads']
    #############
    predefined_flows['147'] = ['BeamLoader', 'AerogridLoader',
                               'StaticTrim',
                               'Modal',
                               'LinearAssembler',
                               'DynamicLoads']
    
    def __init__(self):
        super().__init__()

def sol_401(num_modes,
            forA=[0.,0.,0.],
            beam_orientation=[1., 0, 0, 0],
            rigid_body_modes=False,
            rigid_modes_cg=False,
            use_undamped_modes=True,
            flow=[], **settings):
    """
    Modal solution (stiffness and mass matrices, and natural frequencies)
    in the reference configuration
    """

    settings_new = dict()
    if flow == []:
        flow = ['BeamLoader','NoAero','Modal']
    for k in flow:
        settings_new[k] = {}

    settings_new['BeamLoader']['for_pos'] = forA
    settings_new['BeamLoader']['orientation'] = beam_orientation
    settings_new['BeamLoader']['unsteady'] = False
    settings_new['Modal']['NumLambda'] = num_modes
    settings_new['Modal']['rigid_body_modes'] = rigid_body_modes
    settings_new['Modal']['rigid_modes_cg'] = rigid_modes_cg
    settings_new['Modal']['use_undamped_modes'] = use_undamped_modes
    settings_new['Modal']['print_matrices'] = True
    settings_new['Modal']['write_modes_vtk'] = True
    
    settings_new = basic.update_dic(settings_new, settings)        
    return flow, settings_new

def sol_402(flow=[], **settings):
    """
    Modal solution (stiffness and mass matrices, and natural frequencies)
    in a deformed structural or aeroelastic configuration
    """

    settings_new = dict()
    if flow == []:
        flow = []
        for k in flow:
            settings_new[k] = {}
    else:
        for k in flow:
            settings_new[k] = {}      

    settings_new = basic.update_dic(settings_new, settings)        
    return flow, settings_new

def sol_404(flow=[], **settings):
    """
    Modal solution (stiffness and mass matrices, and natural frequencies)
    in a deformed  configuration after trim analysis
    """

    settings_new = dict()
    if flow == []:
        flow = []
        for k in flow:
            settings_new[k] = {}
    else:
        for k in flow:
            settings_new[k] = {}      

    settings_new = basic.update_dic(settings_new, settings)        
    return flow, settings_new

def sol_410(num_modes,                  # Num modes in the solution
            alpha,                      # Static AoA       
            beta,                       # Static sideslip anum_modes,
            roll,                       # Static roll angle
            panels_wake,                # Number of wake panels
            monitoring_stations,
            components,
            remove_dof,
            father_components,
            flight_conditions,
            white_noise_covariance=[],
            rho=1.22,                   # Air density
            u_inf=1.,                   # Free stream velocity  
            c_ref=1.,                   # Reference chord
            horseshoe=False,            # Use horseshoe approx. in the aerodynamics
            fsi_maxiter=100,            # Max iterations in fsi loop
            fsi_tolerance=1e-5,         # Tolerance in fsi
            fsi_relaxation=0.1,         # Relaxation in fsi
            n_load_steps= 1,            # Apply loads in ramp in the StaticCoupled solver
            gravity_on=True,            # consider gravity forces
            nz=1.,                      # Gravity factor
            velocity_range=[],          # Velocity vector at which stability of the linear system \
                                        # is computed in the AsymptoticStability solver
            root_method='secant',       # Method to compute the eigenvalaues crossing the x-axis          
            flutter_reference=None,     # Starting velocity for the algorithm in DynamicLoads \
                                        # that finds the flutter speed
            velocity_increment=10.,     # Increment of the velocity to find an interval on which \
                                        # the damping of the aeroelastic system is positive and \
                                        # negative at the (thus bounds are found for
                                        # the flutter velocity).
            flutter_error=0.1,          # Interval of the velocity bounds on which to stop iterations
            damping_tolerance=5e-3,     # Fringe above and below 0 where damping criteria is
                                        # considered (flutter speed taken when damping is above
                                        # this value; it helps a lot convergence and filtering of
                                        # modes to have an epsilon rather that 0.)
            frequency_cutoff=0,         # Frequency above which eigenvalues are cut. 0 means no
                                        # no frequency cut 
            forA=[0.,0.,0.],            # Reference node A coordinates  
            dt=0.1,                     # dt for the linear system
            linear_coordinates='modes', # Coordinates in the linearised system
            rigid_body_modes = False,   # Include rigid body modes
            rom_method=['Balanced'],    # Name of reduced order model (empty string calculates \
                                        #the linear system without reducing the model; it might \
                                        #be too expensive for large models)
            rom_settings={},
            flow=[],
            **settings):
    """ 
    Dynamic loads
    """

    settings_new = dict()
    if flow == []:
        flow = ['BeamLoader', 'AerogridLoader',
                'StaticCoupled',
                'Modal',
                'LinearAssembler',
                'LinStateSpace',
                'DynamicLoads']
    for k in flow:
        settings_new[k] = {}
            
    orientation = algebra.euler2quat(np.array([roll,alpha,beta]))
    #orientation = algebra.euler2quat(np.array([0., 0, 0.]))
    u_inf_direction = np.array([1., 0., 0.])
    #u_inf_direction = np.array([np.cos(alpha), 0, np.sin(alpha)])
    settings_new['BeamLoader']['for_pos'] = forA
    settings_new['BeamLoader']['orientation'] = orientation
    settings_new['BeamLoader']['unsteady'] = 'off'
    settings_new['AerogridLoader'] = {'unsteady': 'off',
                                      'aligned_grid': 'on',
                                      'freestream_dir': ['1', '0', '0'],
                                      'wake_shape_generator': 'StraightWake',
                                      'wake_shape_generator_input': {'u_inf': u_inf,
                                                                     'u_inf_direction': u_inf_direction,
                                                                     'dt': dt}}
    if horseshoe:
        settings_new['AerogridLoader']['mstar'] = 1
    else:
        settings_new['AerogridLoader']['mstar'] = panels_wake

    #settings_new['AerogridLoader']['unsteady'] = False   
    settings_new['StaticCoupled']['n_load_steps'] = n_load_steps
    settings_new['StaticCoupled']['max_iter'] = fsi_maxiter
    settings_new['StaticCoupled']['tolerance'] = fsi_tolerance
    settings_new['StaticCoupled']['relaxation_factor'] = fsi_relaxation
    settings_new['StaticCoupled']['aero_solver'] = 'StaticUvlm'
    settings_new['StaticCoupled']['aero_solver_settings'] = {'rho':rho,
                                                             'horseshoe': horseshoe,
                                                             'num_cores': 1,
                                                             'n_rollup': int(1.15*panels_wake),
                                                             'rollup_dt': dt,
                                                             'velocity_field_generator': \
                                                             'SteadyVelocityField',
                                                             'velocity_field_input': \
                                                             {'u_inf': u_inf,
                                                              'u_inf_direction':u_inf_direction}
                                                             }
    settings_new['StaticCoupled']['structural_solver'] = 'NonLinearStatic'
    settings_new['StaticCoupled']['structural_solver_settings'] = {'initial_position':forA,
                                                                   'dt': dt,
                                                                   'gravity_on':gravity_on,
                                                                   'gravity':nz*9.807,
                                                                   }
    settings_new['Modal']['NumLambda'] = num_modes
    settings_new['Modal']['write_modes_vtk'] = 'off'
    settings_new['Modal']['rigid_body_modes'] = rigid_body_modes
    settings_new['Modal']['rigid_modes_cg'] = False
    settings_new['Modal']['use_undamped_modes'] = True
    
    struct_solver_settings = {'print_info': 'off',
                              'initial_velocity_direction': [-1., 0., 0.],
                              'max_iterations': 950,
                              'delta_curved': 1e-6,
                              'min_delta': 1e-6,
                              'newmark_damp': 5e-3,
                              'gravity_on': gravity_on,
                              'gravity': 9.807,
                              'num_steps': 1,
                              'dt': dt,
                              'initial_velocity': u_inf * 1}

    step_uvlm_settings = {'print_info': 'on',
                          # 'horseshoe': 'off',
                          'num_cores': 4,
                          # 'n_rollup': 1,
                          'convection_scheme': 2,
                          # 'rollup_dt': dt,
                          # 'rollup_aic_refresh': 1,
                          # 'rollup_tolerance': 1e-4,
                          'vortex_radius': 1e-6,
                          'velocity_field_generator': 'SteadyVelocityField',
                          'velocity_field_input': {'u_inf': u_inf * 0,
                                                   'u_inf_direction': [1., 0., 0.]},
                          'rho': rho,
                          'n_time_steps': 1,
                          'dt': dt,
                          'gamma_dot_filtering': 3}
    
    settings_new['DynamicCoupled'] = {'print_info': 'on',
                                  # 'structural_substeps': 1,
                                  # 'dynamic_relaxation': 'on',
                                  # 'clean_up_previous_solution': 'on',
                                  'structural_solver': 'NonLinearDynamicCoupledStep',
                                  'structural_solver_settings': struct_solver_settings,
                                  'aero_solver': 'StepUvlm',
                                  'aero_solver_settings': step_uvlm_settings,
                                  'fsi_substeps': 200,
                                  'fsi_tolerance': 1e-10,
                                  'relaxation_factor': 0.2,
                                  'minimum_steps': 1,
                                  'relaxation_steps': 150,
                                  'final_relaxation_factor': 0.5,
                                  'n_time_steps': 1,  # ws.n_tstep,
                                  'dt': dt}

    if  rom_method == '': # Flutter using the full assembled matrices
        rom_settings = {}
    elif 'Balanced' in rom_method and (not rom_settings): # Default settings
        rom_algorithm = 'FrequencyLimited'
        rom_settings = {'frequency':1.,
                        'method_low': 'trapz',
                        'options_low': {'points': 12},
                        'method_high': 'gauss',
                        'options_high': {'partitions': 2, 'order': 8},
                        'check_stability': True}
        rom_settings = {'Balanced':{"algorithm":rom_algorithm,
                                    "algorithm_settings":rom_settings}}
    
    if linear_coordinates == 'modes': #Flutter after projection on the modes
        modal_projection = True
    elif linear_coordinates == 'nodes':
        modal_projection = False
        
    settings_new['LinearAssembler']['inout_coordinates'] = linear_coordinates # 'nodes', 'modes'
    settings_new['LinearAssembler']['linear_system'] = 'LinearAeroelastic'
    settings_new['LinearAssembler']['linear_system_settings'] = {
                                    'uvlm_filename':'',
                                    'track_body':True,
                                    'use_euler':False,
                                    'beam_settings': {'modal_projection': modal_projection,
                                                      'inout_coords': linear_coordinates,
                                                      'discrete_time': 'on',
                                                      'newmark_damp': 0.5e-4,
                                                      'discr_method': 'newmark',
                                                      'dt': dt,
                                                      'proj_modes': 'undamped',
                                                      'use_euler': 'off',
                                                      'num_modes': num_modes,
                                                      'gravity': gravity_on,
                                                      },
                                    'aero_settings': {'dt': dt,
                                                      'ScalingDict': {'length': c_ref,
                                                                      'speed': u_inf,
                                                                      'density': rho},
                                                      'density': rho,
                                                      'remove_inputs': [],
                                                      'rom_method': rom_method,
                                                      'rom_method_settings': rom_settings
                                                      }
                                                              }
    
    settings_new['LinStateSpace']['ss_workflow'] = ['ss_gust',
                                                    'ss_aeroelastic',
                                                    'gain_loads',
                                                    'gain_internalloads']
    settings_new['LinStateSpace']['ss_gust'] = [{'gust_assembler':'LeadingEdge',
                                                'gust_assembler_in':{},
                                                'turbulence_filter':'campbell',
                                                'turbulence_filter_in':{'velocity':u_inf,
                                                                        'dt':dt*u_inf/c_ref}
                                                }]
    settings_new['LinStateSpace']['ss_aeroelastic'] = [{'u_inf':u_inf}]
    settings_new['LinStateSpace']['gain_loads'] = [{}]
    settings_new['LinStateSpace']['gain_internalloads'] = [
        {'monitoring_stations':monitoring_stations,
         'components':components,
         'remove_dof':remove_dof,
         'father_components':father_components}]

    flight_conditions.update({'U_inf':u_inf})
    
    settings_new['DynamicLoads'] = {'print_info': True,
                                    'reference_velocity': flutter_reference,
                                    'velocity_increment': velocity_increment,
                                    'flutter_error': flutter_error,
                                    'root_method': root_method,
                                    'damping_tolerance': damping_tolerance,
                                    'calculate_flutter': True,
                                    'frequency_cutoff': frequency_cutoff,
                                    'save_eigenvalues': True,
                                    'save_flutter': 'txt',
                                    #########
                                    'calculate_rootloads':True,
                                    'flight_conditions':flight_conditions,
                                    'gust_regulation':'Continuous_gust',
                                    'white_noise_covariance':white_noise_covariance
                                    }

    settings_new = basic.update_dic(settings_new, settings)

    
    return flow, settings_new
