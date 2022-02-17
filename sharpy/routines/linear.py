import numpy as np
from sharpy.routines.basic import Basic, update_dic
from sharpy.routines.static import Static
from sharpy.routines.modal import Modal
import sharpy.utils.algebra as algebra

class Linear(Modal, Static, Basic):
    
    predefined_flows = dict()
    #############
    predefined_flows['501'] = ['BeamLoader', 'AerogridLoader',
                               'StaticCoupled',
                               'Modal',
                               'LinearAssembler']
    predefined_flows['502'] = ['BeamLoader', 'AerogridLoader',
                               'StaticCoupled',
                               'Modal',
                               'LinearAssembler']
    predefined_flows['504'] = ['BeamLoader', 'AerogridLoader',
                               'StaticTrim',
                               'Modal',
                               'LinearAssembler']
    
    def __init__(self):
        super().__init__()

    def set_linear(self):

        self.settings_new['LinearAssembler']['linear_system'] = 'LinearAeroelastic'
        self.settings_new['LinearAssembler']['linear_system_settings'] = {
                                        'beam_settings': {'modal_projection': 'on',
                                                          'inout_coords': 'modes',
                                                          'discrete_time': 'on',
                                                          'newmark_damp': 0.5e-4,
                                                          'discr_method': 'newmark',
                                                          'dt': dt,
                                                          'proj_modes': 'undamped',
                                                          'use_euler': 'off',
                                                          'num_modes': num_modes,
                                                          'print_info': 'off',
                                                          'gravity': gravity_on,
                                                          'remove_sym_modes': 'off',
                                                          'remove_dofs': []},
                                        'aero_settings': {'dt': dt,
                                                          'ScalingDict': {'length': 0.5 * c_ref,
                                                                          'speed': u_inf,
                                                                          'density': rho},
                                                          'integr_order': 2,
                                                          'density': rho,
                                                          'remove_predictor': 'off',
                                                          'use_sparse': 'on',
                                                          'rigid_body_motion': rigid_body,
                                                          'use_euler': 'off',
                                                          'remove_inputs': ['u_gust'],
                                                          'rom_method': [rom_method],
                                                          'rom_method_settings': {rom_method: rom_settings}},
                                        'rigid_body_motion': rigid_body}

        
    def sol_501(panels_wake,
                num_modes,
                rho,
                u_inf,
                c_ref,
                dt=0.1,
                gravity_on=1,
                u_inf_direction=[1., 0., 0.],
                rigid_body='off',
                rom_method='Default',
                rom_settings=None,
                flow=[],
                **settings):
        """
        Predefined solution to obtain an aeroelastic Reduced-Order-Model in the 
        reference configuration
        """

        settings_new = dict()
        if flow == []:
            flow = ['BeamLoader', 'AerogridLoader', 'DynamicCoupled',
                    'Modal', 'LinearAssembler', 'SaveData']
            for k in flow:
                settings_new[k] = {}
        else:
            for k in flow:
                settings_new[k] = {}
        if rom_method == 'Default':
            rom_method = 'Krylov'
            rom_settings = {'algorithm':'mimo_rational_arnoldi',
                            'r':6}
            frequency_continuous_k = np.array([0.])  
            frequency_continuous_w = 2 * u_inf * frequency_continuous_k / c_ref
            rom_settings['frequency'] = frequency_continuous_w

        settings_new['BeamLoader']['unsteady'] = 'off'
        settings_new['AerogridLoader'] = {
            'unsteady': 'off',
            'aligned_grid': 'on',
            'mstar': panels_wake,
            'freestream_dir': u_inf_direction,
            'wake_shape_generator': 'StraightWake',
            'wake_shape_generator_input': {'u_inf': u_inf,
                                           'u_inf_direction': u_inf_direction,
                                           'dt': dt}}

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

        settings_new['Modal'] = {'NumLambda': num_modes,
                                 'rigid_body_modes': rigid_body,
                                 'print_matrices': 'on',
                                 'keep_linear_matrices': 'on',
                                 'write_dat': 'on',
                                 'continuous_eigenvalues': 'off',
                                 'write_modes_vtk': 'off',
                                 'use_undamped_modes': 'on'}

        if primary:
            self.modify_settings(self.flow, **kwargs)
            return self.flow, self.settings_new
    

    def sol_501(panels_wake,
                num_modes,
                rho,
                u_inf,
                c_ref,
                folder,
                dt=0.1,
                gravity_on=1,
                u_inf_direction=[1., 0., 0.],
                rigid_body='off',
                rom_method='Default',
                rom_settings=None,
                flow=[],
                **settings):
        """
        Predefined solution to obtain an aeroelastic Reduced-Order-Model 
        on a static coupled equilibrium
        """

        settings_new = dict()
        if flow == []:
            flow = ['BeamLoader', 'AerogridLoader', 'StaticCoupled',
                    'Modal', 'LinearAssembler', 'SaveData']
            for k in flow:
                settings_new[k] = {}
        else:
            for k in flow:
                settings_new[k] = {}
        if rom_method == 'Default':
            rom_method = 'Krylov'
            rom_settings = {'algorithm':'mimo_rational_arnoldi',
                            'r':6}
            frequency_continuous_k = np.array([0.])  # Interpolation point in the complex plane with reduced frequency units
            frequency_continuous_w = 2 * u_inf * frequency_continuous_k / c_ref
            rom_settings['frequency'] = frequency_continuous_w

        settings_new['BeamLoader']['usteady'] = 'off'
        settings_new['AerogridLoader'] = {
            'unsteady': 'off',
            'aligned_grid': 'on',
            'mstar': panels_wake,
            'freestream_dir': u_inf_direction,
            'wake_shape_generator': 'StraightWake',
            'wake_shape_generator_input': {'u_inf': u_inf,
                                           'u_inf_direction': u_inf_direction,
                                           'dt': dt}}

        settings_new['StaticCoupled'] = {'print_info': 'on',
                                         'max_iter': 200,
                                         'n_load_steps': 4,
                                         'tolerance': 1e-5,
                                         'relaxation_factor': 0.05,
                                         'aero_solver': 'StaticUvlm',
                                         'aero_solver_settings': {'rho': rho,
                                                                  'print_info': 'off',
                                                                  'horseshoe': 'on',
                                                                  'num_cores': 4,
                                                                  'n_rollup': 0,
                                                                  'rollup_dt': dt,
                                                                  'rollup_aic_refresh': 1,
                                                                  'rollup_tolerance': 1e-4,
                                                                  'velocity_field_generator': 'SteadyVelocityField',
                                                                  'velocity_field_input': {'u_inf': u_inf,
                                                                                           'u_inf_direction': u_inf_direction},
                                                                  'vortex_radius': 1e-9 },
                                         'structural_solver': 'NonLinearStatic',
                                         'structural_solver_settings': {'print_info': 'off',
                                                                        'max_iterations': 150,
                                                                        'num_load_steps': 4,
                                                                        'delta_curved': 1e-2,
                                                                        'min_delta': 1e-7,
                                                                        'gravity_on': gravity_on,
                                                                        'gravity': 9.807}}

        settings_new['Modal'] = {'folder': folder,
                                'NumLambda': num_modes,
                                'rigid_body_modes': rigid_body,
                                'print_matrices': 'on',
                                'keep_linear_matrices': 'on',
                                'write_dat': 'on',
                                'continuous_eigenvalues': 'off',
                                'write_modes_vtk': 'off',
                                'use_undamped_modes': 'on'}

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
                                                          'print_info': 'off',
                                                          'gravity': gravity_on,
                                                          'remove_sym_modes': 'off',
                                                          'remove_dofs': []},
                                        'aero_settings': {'dt': dt,
                                                          'ScalingDict': {'length': 0.5 * c_ref,
                                                                          'speed': u_inf,
                                                                          'density': rho},
                                                          'integr_order': 2,
                                                          'density': rho,
                                                          'remove_predictor': 'off',
                                                          'use_sparse': 'on',
                                                          'rigid_body_motion': rigid_body,
                                                          'use_euler': 'off',
                                                          'remove_inputs': ['u_gust'],
                                                          'rom_method': [rom_method],
                                                          'rom_method_settings': {rom_method: rom_settings}},
                                        'rigid_body_motion': rigid_body}

        settings_new['SaveData']['folder'] = folder
        settings_new['SaveData']['save_aero'] = False
        settings_new['SaveData']['save_struct'] = False
        settings_new['SaveData']['save_linear'] = True
        settings_new['SaveData']['save_linear_uvlm'] = True
        settings_new['SaveData']['save_rom'] = True

        if primary:
            self.modify_settings(self.flow, **kwargs)
            return self.flow, self.settings_new

    def sol_502(panels_wake,
                num_modes,
                rho,
                u_inf,
                c_ref,
                folder,
                dt=0.1,
                gravity_on=1,
                u_inf_direction=[1., 0., 0.],
                rigid_body='off',
                rom_method='Default',
                rom_settings=None,
                flow=[],
                **settings):
        """
        Predefined solution to obtain an aeroelastic Reduced-Order-Model 
        after trimming the aircraft
        """

        if primary:
            self.modify_settings(self.flow, **kwargs)
            return self.flow, self.settings_new
        
