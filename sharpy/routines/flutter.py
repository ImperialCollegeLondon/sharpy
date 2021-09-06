import numpy as np
import sharpy.routines.basic as basic
import sharpy.utils.algebra as algebra


def sol_145(num_modes,                  # Num modes in the solution
            panels_wake,                # Number of wake panels
            rho,                        # Air density
            u_inf,                      # Free stream velocity  
            c_ref=1.,                   # Reference chord
            horseshoe=False,            # Horseshoe aerodynamic approximation
            velocity_range=[],          # Velocity vector at which stability of the linear system \
                                        # is computed in the AsymptoticStability solver
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
            frequency_cutoff=0,         # Frequency above which eigenvalues are cut. 0 means no \
                                        # no frequency cut 
            forA=[0.,0.,0.],            # Reference node A coordinates  
            dt=0.1,                     # dt for the linear system
            gravity_linearise=True,     # Take gravity forces in the linear system
            linear_coordinates='modes', # Coordinates in the linearised system
            rigid_body_modes = False,   # Include rigid body modes
            rom_method=['Balanced'],    # Name of reduced order model (empty string calculates \
                                        #the linear system without reducing the model; it might \
                                        #be too expensive for large models)
            rom_settings={},
            flow=[],
            **settings):
    """ 
    Flutter predifined solution in the reference configuration
    """

    settings_new = dict()
    if flow == []:
        flow = ['BeamLoader', 'AerogridLoader',
                'StaticCoupled',
                'Modal',
                'LinearAssembler']
        if velocity_range:                      # Run asymptotic analysis in the free-stream velocity specified
            flow += ['AsymptoticStability']
        if flutter_reference:                   # Compute the flutter speed starting from velocity flutter_reference 
            flow += ['DynamicLoads']
        for k in flow:
            settings_new[k] = {}
    else:
        for k in flow:
            settings_new[k] = {}        

    settings_new['BeamLoader']['for_pos'] = forA
    settings_new['BeamLoader']['orientation'] = [1.0, 0, 0, 0]
    settings_new['BeamLoader']['unsteady'] = 'off'
    settings_new['AerogridLoader'] = {'unsteady': 'off',
                                      'aligned_grid': 'on',
                                      'freestream_dir': ['1', '0', '0'],
                                      'wake_shape_generator': 'StraightWake',
                                      'wake_shape_generator_input': {'u_inf': u_inf,
                                                                     'u_inf_direction': np.array([1., 0., 0.]),
                                                                     'dt': dt}}
    if horseshoe:
        settings_new['AerogridLoader']['mstar'] = 1
    else:
        settings_new['AerogridLoader']['mstar'] = panels_wake

    settings_new['StaticCoupled']['n_load_steps'] = 1
    settings_new['StaticCoupled']['aero_solver'] = 'StaticUvlm'
    settings_new['StaticCoupled']['aero_solver_settings'] = {'rho': 0.001,            # So there is no deformations
                                                             'horseshoe':horseshoe,
                                                             'num_cores': 1,
                                                             'n_rollup': 0,
                                                             'rollup_dt': dt,
                                                             'rollup_aic_refresh': 1,
                                                             'rollup_tolerance': 1e-4,
                                                             'velocity_field_generator':'SteadyVelocityField',
                                                             'velocity_field_input':{'u_inf': u_inf,
                                                                                     'u_inf_direction':[1.,0.,0.]}
                                                            }
    settings_new['StaticCoupled']['structural_solver'] = 'NonLinearStatic'
    settings_new['StaticCoupled']['structural_solver_settings'] = {'initial_position':forA,
                                                                   'dt': dt,
                                                                   'gravity_on':False
                                                                   }
    settings_new['Modal']['NumLambda'] = num_modes
    settings_new['Modal']['write_modes_vtk'] = 'off'
    settings_new['Modal']['rigid_body_modes'] = rigid_body_modes
    settings_new['Modal']['rigid_modes_cg'] = False
    settings_new['Modal']['use_undamped_modes'] = True
    
    if  rom_method == '': # Flutter using the full assembled matrices
        rom_settings = {}
    elif 'Balanced' in rom_method and (not rom_settings):
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
                                                      'gravity': gravity_linearise,
                                                      },
                                    'aero_settings': {'dt': dt,
                                                      'ScalingDict': {'length': c_ref,
                                                                      'speed': u_inf,
                                                                      'density': rho},
                                                      'density': rho,
                                                      'remove_inputs': ['u_gust'],
                                                      'rom_method': rom_method,
                                                      'rom_method_settings': rom_settings
                                                      }
                                                              }
    settings_new['AsymptoticStability'] = {'print_info': True,
                                           'velocity_analysis': velocity_range,
                                           }

    settings_new['DynamicLoads'] = {'print_info': True,
                                    'reference_velocity': flutter_reference,
                                    'velocity_increment': velocity_increment,
                                    'flutter_error': flutter_error,
                                    'damping_tolerance': damping_tolerance,
                                    'calculate_flutter': True,
                                    'frequency_cutoff': frequency_cutoff,
                                    'save_eigenvalues': True
                                   }
    
    settings_new = basic.update_dic(settings_new, settings)        

    return flow, settings_new

def sol_146(num_modes,                  # Num modes in the solution
            alpha,                      # Static AoA       
            beta,                       # Static sideslip anum_modes,
            roll,                       # Static roll angle
            panels_wake,                # Number of wake panels
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
    Flutter predifined solution after aeroelastic equilibrium
    """

    settings_new = dict()
    if flow == []:
        flow = ['BeamLoader', 'AerogridLoader',
                'StaticCoupled',
                'Modal',
                'LinearAssembler']
        if velocity_range:                      # Run asymptotic analysis in the free-stream velocity specified
            flow += ['AsymptoticStability']
        if flutter_reference:                   # Compute the flutter speed starting from velocity flutter_reference 
            flow += ['DynamicLoads']
        for k in flow:
            settings_new[k] = {}
    else:
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
                                                      'remove_inputs': ['u_gust'],
                                                      'rom_method': rom_method,
                                                      'rom_method_settings': rom_settings
                                                      }
                                                              }
    settings_new['AsymptoticStability'] = {'print_info': True,
                                           'velocity_analysis': velocity_range,
                                           }

    settings_new['DynamicLoads'] = {'print_info': True,
                                    'reference_velocity': flutter_reference,
                                    'velocity_increment': velocity_increment,
                                    'flutter_error': flutter_error,
                                    'damping_tolerance': damping_tolerance,
                                    'calculate_flutter': True,
                                    'frequency_cutoff': frequency_cutoff,
                                    'save_eigenvalues': True
                                    }

    settings_new = basic.update_dic(settings_new, settings)

    
    return flow, settings_new


def sol_147(num_modes,                    # Num modes in the solution
            u_inf,                        # Free stream velocity
            rho,                          # Air density 
            panels_wake,                  # Number of wake panels 
            alpha0,                       # Initial angle of attack
            thrust0,                      # Number of     
            cs0,                          # Number of wake panels 
            thrust_nodes,                 # Nodes where thrust is applied
            cs_i,                         # Indices of control surfaces to be trimmed
            nz=1.,                        # Gravity factor for manoeuvres          
            Dcs0=0.01,                    # Initial control surface deflection
            Dthrust0=0.1,                 # Initial thrust variation 
            fx_tolerance=0.01,            # Tolerance in horizontal  force
            fz_tolerance=0.01,            # Tolerance in vertical force
            pitching_tolerance=0.01,      # Tolerance in pitching 
            forA=[0., 0., 0.],            # Reference node A coordinates 
            horseshoe=False,              # Use horseshoe approx. in the aerodynamics
            dt=0.05,                      # dt for uvlm 
            trim_max_iter=100,            # Mximum number of trim iterations
            trim_relaxation_factor=0.2,   # Relaxation factor 
            struct_tol=1e-5,              # Tolerance of structural solver
            fsi_tolerance=1e-5,           # FSI loop tolerance
            fsi_relaxation=0.1,           # FSI relaxation_factor
            fsi_maxiter=100,              # FSI maximum number of iterations
            n_load_steps= 1,              # Apply loads in ramp in the StaticCoupled solver
            c_ref=1.,                     # Reference chord
            velocity_range=[],            # Velocity vector at which stability of the linear system \
                                        # is computed in the AsymptoticStability solver
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
            gravity_linearise=True,     # Take gravity forces in the linear system
            linear_coordinates='modes', # Coordinates in the linearised system
            rigid_body_modes = False,   # Include rigid body modes
            rom_method=['Balanced'],    # Name of reduced order model (empty string calculates \
                                        # the linear system without reducing the model; it might \
                                        # be too expensive for large models)
            rom_settings={},            # Setting for the ROM method
            flow=[],
            **settings):
    """ 
    Flutter predifined solution after trimmed flight
    """

    settings_new = dict()
    if flow == []:
        flow = ['BeamLoader', 'AerogridLoader',
                'StaticTrim',
                'Modal',
                'LinearAssembler']
        if velocity_range:                      # Run asymptotic analysis in the free-stream velocity specified
            flow += ['AsymptoticStability']
        if flutter_reference:                   # Compute the flutter speed starting from velocity flutter_reference 
            flow += ['DynamicLoads']
        for k in flow:
            settings_new[k] = {}
    else:
        for k in flow:
            settings_new[k] = {}
            
    orientation = algebra.euler2quat(np.array([0., alpha0, 0.]))
    settings_new['BeamLoader']['for_pos'] = forA
    settings_new['BeamLoader']['orientation'] = orientation
    settings_new['BeamLoader']['unsteady'] = 'off'
    settings_new['AerogridLoader']['mstar'] = panels_wake
    settings_new['AerogridLoader']['unsteady'] = False   
    settings_new['StaticTrim']['initial_alpha'] = alpha0
    settings_new['StaticTrim']['initial_deflection'] = cs0
    settings_new['StaticTrim']['initial_angle_eps'] = Dcs0
    settings_new['StaticTrim']['initial_thrust'] = thrust0
    settings_new['StaticTrim']['initial_thrust_eps'] = Dthrust0
    settings_new['StaticTrim']['thrust_nodes'] = thrust_nodes
    settings_new['StaticTrim']['tail_cs_index'] = cs_i
    settings_new['StaticTrim']['fx_tolerance'] = fx_tolerance
    settings_new['StaticTrim']['fz_tolerance'] = fz_tolerance
    settings_new['StaticTrim']['m_tolerance'] = pitching_tolerance
    settings_new['StaticTrim']['max_iter'] = trim_max_iter
    settings_new['StaticTrim']['relaxation_factor'] = trim_relaxation_factor
    settings_new['StaticTrim']['save_info'] = 'on'
    settings_new['StaticTrim']['solver'] = 'StaticCoupled'
    settings_new['StaticTrim']['solver_settings'] = {'print_info': 'on',
                                 'structural_solver': 'NonLinearStatic',
                                 'structural_solver_settings': {'print_info': 'off',
                                                                'max_iterations': 200,
                                                                'num_load_steps': 1,
                                                                'delta_curved': 1e-5,
                                                                'min_delta': struct_tol,
                                                                'gravity_on': 'on',
                                                                'gravity':nz*9.807,
                                                                'initial_position':forA,
                                                                'dt':dt
                                                                },
                                 'aero_solver': 'StaticUvlm',
                                 'aero_solver_settings': {'print_info': 'off',
                                                          'horseshoe': horseshoe,
                                                          'num_cores': 1,
                                                          'n_rollup': int(0),
                                                          'rollup_dt': dt,
                                                          'rollup_aic_refresh': 1,
                                                          'rollup_tolerance': 1e-4,
                                                          'velocity_field_generator': \
                                                          'SteadyVelocityField',
                                                          'velocity_field_input': \
                                                          {'u_inf':u_inf,
                                                           'u_inf_direction': [1., 0, 0]
                                                          },
                                                           'rho':rho
                                                          },
                                 'max_iter': fsi_maxiter,
                                 'n_load_steps': n_load_steps,
                                 'tolerance': fsi_tolerance,
                                 'relaxation_factor': fsi_relaxation
                                                }
    settings_new['Modal']['NumLambda'] = num_modes
    settings_new['Modal']['write_modes_vtk'] = 'off'
    settings_new['Modal']['rigid_body_modes'] = rigid_body_modes
    settings_new['Modal']['rigid_modes_cg'] = False
    settings_new['Modal']['use_undamped_modes'] = True

    if not rom_method == '': # Flutter using the full assembled matrices
        rom_settings = {}
    elif 'Balanced' in rom_method and (not rom_settings):
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
                                                      'gravity': 'on',
                                                      },
                                    'aero_settings': {'dt': dt,
                                                      'ScalingDict': {'length': c_ref,
                                                                      'speed': u_inf,
                                                                      'density': rho},
                                                      'density': rho,
                                                      'remove_inputs': ['u_gust'],
                                                      'rom_method': rom_method,
                                                      'rom_method_settings': rom_settings
                                                      }
                                                              }
    settings_new['AsymptoticStability'] = {'print_info': True,
                                           'velocity_analysis': velocity_range,
                                           }

    settings_new['DynamicLoads'] = {'print_info': True,
                                    'reference_velocity': flutter_reference,
                                    'velocity_increment': velocity_increment,
                                    'flutter_error': flutter_error,
                                    'damping_tolerance': damping_tolerance,
                                    'calculate_flutter': True,
                                    'frequency_cutoff': frequency_cutoff,
                                    'save_eigenvalues': True
                                   }

    settings_new = basic.update_dic(settings_new, settings)        
    return flow, settings_new
