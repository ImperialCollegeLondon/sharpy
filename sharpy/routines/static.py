import numpy as np
import sharpy.routines.basic as basic
import sharpy.utils.algebra as algebra


def sol_101(forA=[0.,0.,0.],
            gravity_on=0,
            max_iterations=200,
            tolerance=1e-5,
            num_load_steps=5,
            newmark_damp=1e-4,
            flow=[], **settings):
    
    """Structural equilibrium"""

    settings_new = dict()
    if flow == []:
        flow = ['BeamLoader','NonLinearStatic']
    for k in flow:
        settings_new[k] = {}

    settings_new['BeamLoader']['for_pos'] = forA
    settings_new['BeamLoader']['orientation'] = [1.0, 0, 0, 0]
    settings_new['BeamLoader']['usteady'] = 'off'
    settings_new['NonLinearStatic']['gravity_on'] = gravity_on
    settings_new['NonLinearStatic']['initial_position'] = forA
    settings_new['NonLinearStatic']['max_iterations'] = max_iterations
    settings_new['NonLinearStatic']['min_delta'] = tolerance
    settings_new['NonLinearStatic']['num_load_steps'] = num_load_steps
    settings_new['NonLinearStatic']['newmark_damp'] = newmark_damp
        
    settings_new = basic.update_dic(settings_new, settings)        
    return flow, settings_new


def sol_102(alpha,                    # Static AoA 
            beta,                     # Static sideslip angle    
            roll,                     # Static roll angle
            u_inf,
            rho,
            panels_wake,
            horseshoe=False,
            dt=0.05,
            gravity_on=0,
            forA=[0.,0.,0.],
            fsi_maxiter=100,
            fsi_tolerance=1e-5,
            fsi_relaxation=0.1,
            flow=[], **settings):

    
    """ Aeroelastic equilibrium"""

    settings_new = dict()
    if flow == []:
        flow = ['BeamLoader', 'AerogridLoader',
                'StaticCoupled']
    for k in flow:
        settings_new[k] = {}      

    orientation = algebra.euler2quat(np.array([roll,alpha,beta]))
    settings_new['BeamLoader']['for_pos'] = forA
    settings_new['BeamLoader']['orientation'] = orientation
    settings_new['BeamLoader']['usteady'] = 'off'
    if horseshoe:
        settings_new['AerogridLoader']['mstar'] = 1
    else:
        settings_new['AerogridLoader']['mstar'] = panels_wake
    settings_new['AerogridLoader']['unsteady'] = False
    settings_new['AerogridLoader']['freestream_dir'] = [1.,0.,0.]
    settings_new['AerogridLoader']['wake_shape_generator'] = 'StraightWake'
    settings_new['AerogridLoader']['wake_shape_generator_input'] = {'u_inf': u_inf,
                                                                    'u_inf_direction':['1', '0', '0'],
                                                                    'dt': dt}
    settings_new['StaticCoupled']['n_load_steps'] = 1
    settings_new['StaticCoupled']['max_iter'] = fsi_maxiter
    settings_new['StaticCoupled']['tolerance'] = fsi_tolerance
    settings_new['StaticCoupled']['relaxation_factor'] = fsi_relaxation
    settings_new['StaticCoupled']['aero_solver'] = 'StaticUvlm'
    settings_new['StaticCoupled']['aero_solver_settings'] = {'rho':rho,
                                                             'horseshoe': horseshoe,
                                                             'num_cores': 1,
                                                             'n_rollup': 1.15*panels_wake,
                                                             'rollup_dt': dt,
                                                             'velocity_field_generator': \
                                                             'SteadyVelocityField',
                                                             'velocity_field_input': \
                                                             {'u_inf': u_inf,
                                                              'u_inf_direction':[1.,0.,0.]}
                                                             },
    settings_new['StaticCoupled']['structural_solver'] = 'NonLinearStatic'
    settings_new['StaticCoupled']['structural_solver_settings'] = {'initial_position':forA,
                                                                   'dt': dt,
                                                                   'gravity_on':gravity_on
                                                                   }

    settings_new = basic.update_dic(settings_new, settings)        
    return flow, settings_new


def sol_144(u_inf,                        # Free stream velocity
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
            horseshoe='off',              # Horseshoe aerodynamic approximation 
            dt=0.05,                      # dt for uvlm 
            trim_max_iter=100,            # Mximum number of trim iterations
            trim_relaxation_factor=0.2,   # Relaxation factor 
            struct_tol=1e-5,              # Tolerance of structural solver
            fsi_tolerance=1e-5,           # FSI loop tolerance
            fsi_relaxation=0.1,           # FSI relaxation_factor
            fsi_maxiter=100,              # FSI maximum number of iterations
            flow=[], **settings):         # Flow and settings to modify the predifined solution
    """ Longitudinal aircraft trim"""

    settings_new = dict()
    if flow == []:
        flow = ['BeamLoader', 'AerogridLoader', 'StaticTrim']
    for k in flow:
        settings_new[k] = {}
            
    orientation = algebra.euler2quat(np.array([0.,alpha0,0.]))
    settings_new['BeamLoader']['for_pos'] = forA
    settings_new['BeamLoader']['orientation'] = orientation
    settings_new['BeamLoader']['usteady'] = 'on'
    if horseshoe:
        settings_new['AerogridLoader']['mstar'] = 1
    else:
        settings_new['AerogridLoader']['mstar'] = panels_wake
    settings_new['AerogridLoader']['unsteady'] = False
    settings_new['AerogridLoader']['freestream_dir'] = [1.,0.,0.]
    settings_new['AerogridLoader']['wake_shape_generator'] = 'StraightWake'
    settings_new['AerogridLoader']['wake_shape_generator_input'] = {'u_inf': u_inf,
                                                                    'u_inf_direction':['1', '0', '0'],
                                                                    'dt': dt}
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
                                 'n_load_steps': 1,
                                 'tolerance': fsi_tolerance,
                                 'relaxation_factor': fsi_relaxation
                                                }
            
    settings_new = basic.update_dic(settings_new, settings)        
    return flow, settings_new

def sol_143(flow=[], **settings):
    """ Aircraft general trim"""

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

