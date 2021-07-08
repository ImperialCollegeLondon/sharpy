import numpy as np
import copy
from cases.models_generator.gen_utils import update_dic
#import sharpy.routines.basic as basic

def sol_145(num_modes,
            panels_wake,
            velocity_range,
            forA=[0.,0.,0.],
            rho=1.22,
            u_inf=1.,
            c_ref=1.,
            dt=0.1,
            gravity='on',
            linear_coordinates='modes',
            rom_method='Balanced',
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
                'LinearAssembler',
                'AsymptoticStability']
        for k in flow:
            settings_new[k] = {}
    else:
        for k in flow:
            settings_new[k] = {}        

    settings_new['BeamLoader']['for_pos'] = forA
    settings_new['BeamLoader']['orientation'] = [1.0, 0, 0, 0]
    settings_new['BeamLoader']['usteady'] = 'off'
    settings_new['AerogridLoader']['mstar'] = panels_wake
    settings_new['AerogridLoader']['unsteady'] = False   
    settings_new['StaticCoupled']['n_load_steps'] = 1
    settings_new['StaticCoupled']['aero_solver'] = 'StaticUvlm'
    settings_new['StaticCoupled']['aero_solver_settings'] = {'rho': 0.,
                                                             'horseshoe': 'off',
                                                             'num_cores': 1,
                                                             'n_rollup': int(1.15*panels_wake),
                                                             'rollup_dt': dt,
                                                             'velocity_field_generator':'SteadyVelocityField',
                                                             'velocity_field_input':{'u_inf': u_inf,
                                                                                     'u_inf_direction':[1.,0.,0.]}}
    settings_new['StaticCoupled']['structural_solver'] = 'NonLinearStatic'
    settings_new['StaticCoupled']['structural_solver_settings'] = {'initial_position':forA,
                                                                   'dt': dt,
                                                                   'gravity_on':False
                                                                   }
    settings_new['Modal']['NumLambda'] = num_modes
    settings_new['Modal']['write_modes_vtk'] = 'off'
    settings_new['Modal']['rigid_body_modes'] = False
    settings_new['Modal']['rigid_modes_cg'] = False
    settings_new['Modal']['use_undamped_modes'] = True

    if not rom_method: # Flutter using the full assembled matrices
        rom_settings = {}
    elif rom_method=='Balanced' and (not rom_settings):
        rom_settings = {"algorithm":'Iterative',
                        "algorithm_settings":{}}
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
                                                      'gravity': gravity,
                                                      },
                                    'aero_settings': {'dt': dt,
                                                      'ScalingDict': {'length': c_ref,
                                                                      'speed': u_inf,
                                                                      'density': rho},
                                                      'density': rho,
                                                      'rigid_body_motion': 'off',
                                                      'use_euler': 'off',
                                                      'remove_inputs': ['u_gust'],
                                                      'rom_method': rom_method,
                                                      'rom_method_settings': rom_settings
                                                      }
                                                              }
    settings_new['AsymptoticStability'] = {'print_info': True,
                                           'velocity_analysis': velocity_range,
                                           }
    
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
