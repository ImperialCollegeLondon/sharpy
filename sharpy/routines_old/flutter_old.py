import numpy as np
import copy
from cases.models_generator.gen_utils import update_dic
import sharpy.routines_old.basic

def sol_145(panels_wake,
            num_modes,
            rho,
            u_inf,
            c_ref,
            velocity_analysis,
            rom_algorithm,
            rom_size,
            folder,
            dt=0.1,
            u_inf_direction=[1., 0., 0.],
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
                'LinearAssembler',
                'AsymptoticStability']
        for k in flow:
            settings_new[k] = {}
    else:
        for k in flow:
            settings_new[k] = {}        
    rom_settings = dict()
    rom_settings['algorithm'] = rom_algorithm # 'mimo_rational_arnoldi'  # reduction algorithm
    rom_settings['r'] = rom_size  # Krylov subspace order
    frequency_continuous_k = np.array([0.])  # Interpolation point in the complex plane with reduced frequency units
    frequency_continuous_w = 2 * u_inf * frequency_continuous_k / c_ref
    rom_settings['frequency'] = frequency_continuous_w
    settings_new['BeamLoader']['usteady'] = 'off'
    settings_new['AerogridLoader']['mstar'] = panels_wake
    settings_new['StaticCoupled'] = {'print_info': 'on',
                                     'max_iter': 200,
                                     'n_load_steps': 1,
                                     'tolerance': 1e-10,
                                     'relaxation_factor': 0.,
                                     'aero_solver': 'StaticUvlm',
                                     'aero_solver_settings': {'rho': rho,
                                                              'print_info': 'off',
                                                              'horseshoe': 'off',
                                                              'num_cores': 4,
                                                              'n_rollup': 0,
                                                              'rollup_dt': dt,
                                                              'rollup_aic_refresh': 1,
                                                              'rollup_tolerance': 1e-4,
                                                              'velocity_field_generator': 'SteadyVelocityField',
                                                              'velocity_field_input': {'u_inf': u_inf*1e-6,
                                                                                       'u_inf_direction': u_inf_direction}},
                                     'structural_solver': 'NonLinearStatic',
                                     'structural_solver_settings': {'print_info': 'off',
                                                                    'max_iterations': 150,
                                                                    'num_load_steps': 4,
                                                                    'delta_curved': 1e-1,
                                                                    'min_delta': 1e-10,
                                                                    'gravity_on': 'on',
                                                                    'gravity': 9.807}}
    settings_new['Modal']['NumLambda'] = num_modes
    settings_new['Modal']['write_modes_vtk'] = 'off'
    settings_new['Modal']['folder'] = folder

    settings_new['LinearAssembler']['linear_system'] = 'LinearAeroelastic'
    settings_new['LinearAssembler']['linear_system_settings'] = {
                                    'beam_settings': {'modal_projection': 'on',
                                                      'inout_coords': 'modes',
                                                      'discrete_time': 'on',
                                                      'newmark_damp': 0.5e-4,
                                                      'discr_method': 'newmark',
                                                      'dt': dt,
                                                      'proj_modes': 'undamped',
                                                      'use_euler': 'off',
                                                      'num_modes': num_modes,
                                                      'print_info': 'on',
                                                      'gravity': 'on',
                                                      'remove_sym_modes': 'off',
                                                      'remove_dofs': []},
                                    'aero_settings': {'dt': dt,
                                                      'ScalingDict': {'length': 0.5 * c_ref,
                                                                      'speed': 0.01,
                                                                      'density': rho},
                                                      'integr_order': 2,
                                                      'density': rho,
                                                      'remove_predictor': 'on',
                                                      'use_sparse': 'on',
                                                      'rigid_body_motion': 'off',
                                                      'use_euler': 'off',
                                                      'remove_inputs': ['u_gust'],
                                                      'rom_method': ['Krylov'],
                                                      'rom_method_settings': {'Krylov': rom_settings}},
                                    'rigid_body_motion': False}
    settings_new['AsymptoticStability'] = {'print_info': True,
                                           'folder': folder,
                                           'velocity_analysis': velocity_analysis,
                                           'modes_to_plot': []}
    
    settings_new = update_dic(settings_new, settings)        
    return flow, settings_new

def sol_146(varx, flow=[], settings={}):
    """ 
    Flutter predifined solution after aeroelastic equilibrium
    """

    pass

def sol_147(varx, flow=[], settings={}):
    """ 
    Flutter predifined solution after trimmed flight
    """

    pass
