import numpy as np
from sharpy.routines.modal import Modal

class Linear(Modal):
    
    predefined_flows = dict()
    #############

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
        
    def sol_502(self,
                u_inf,
                dt,
                rho,
                rho_static=1e-5,
                panels_wake=1,
                num_modes=10,
                inout_coordinates='modal',
                newmark_damp=1e-4,
                use_euler=False,
                track_body=True,
                mach_number=0.,
                c_ref=1.,
                gravity_on=1,
                horseshoe=False,
                rotationA=[0., 0., 0.],
                rom_method='',
                rom_settings=None,
                rigid_body_modes=False,
                rigid_modes_cg=False,
                use_undamped_modes=True,
                write_modal_data=True,
                write_modes_vtk=True,
                max_modal_disp=0.15,
                max_modal_rot_deg=15.,
                print_modal_matrices=False,
                fsi_maxiter=100,
                fsi_tolerance=1e-5,
                fsi_relaxation=0.05,
                fsi_load_steps=1,
                s_maxiter=100,
                s_tolerance=1e-5,
                s_relaxation=1e-3,
                s_load_steps=1,
                s_delta_curved=1e-4,
                correct_forces_method=None,
                correct_forces_settings=None,
                primary=True,
                **kwargs):
        """
        Predefined solution to obtain an aeroelastic Reduced-Order-Model in the 
        reference configuration
        """
        if horseshoe:
            panels_wake = 1
        if primary:
            predefined_flow = list(self.predefined_flows['501'])
            self.set_constants(**kwargs)            
            self.set_flow(predefined_flow, **kwargs)
            self.set_loaders(panels_wake,
                             u_inf,
                             dt,
                             rotationA,
                             unsteady=False,                            
                             **kwargs)
            self.set_plot(u_inf,
                          dt,
                          **kwargs)

        if  (rom_method == '') or (rom_method == []):
            # Flutter using the full assembled matrices
            rom_settings = {}
            rom_methodX = ''
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
            rom_methodX = ['Balanced']
        elif 'Krylov' in rom_method and (not rom_settings):
            rom_settings = {'algorithm':'mimo_rational_arnoldi',
                            'r':6}
            frequency_continuous_k = np.array([0.])
            frequency_continuous_w = 2 * u_inf * frequency_continuous_k / c_ref
            rom_settings['frequency'] = frequency_continuous_w
            rom_methodX = ['Krylov']
        if inout_coordinates == 'modes': #Flutter after projection on the modes
            modal_projection = True
        elif inout_coordinates == 'nodes':
            modal_projection = False

        beam_settings = self.get_linear_sett('LinearBeam',
                                             modal_projection=modal_projection, 
                                             inout_coords=inout_coordinates,  
                                             discrete_time=True,    
                                             newmark_damp=newmark_damp,   
                                             discr_method='newmark',
                                             dt= dt,                 
                                             proj_modes='undamped', 
                                             use_euler=use_euler,       
                                             num_modes=num_modes,   
                                             gravity=gravity_on,    
                                             remove_sym_modes='off')

        aero_settings = self.get_linear_sett('LinearUVLM',
                                             dt=dt,                                         
                                             ScalingDict= {'length': c_ref,            
                                                            'speed': u_inf,                   
                                                            'density': rho},
                                             density=rho,                                   
                                             remove_predictor=True,                        
                                             use_sparse='on',                               
                                             remove_inputs=['u_gust'],                      
                                             rom_method=rom_methodX,                      
                                             rom_method_settings=rom_settings)

        linear_system_settings = self.get_linear_sett('LinearAeroelastic',
                                                      beam_settings=beam_settings,
                                                      aero_settings=aero_settings,
                                                      track_body=track_body,
                                                      use_euler=use_euler,
                                                      mach_number=mach_number)
        
        self.settings_new['LinearAssembler'] = self.get_solver_sett('LinearAssembler',
                                                                    linear_system='LinearAeroelastic',
                                                                    inout_coordinates=inout_coordinates,
                                                                    linear_system_settings=\
                                                                    linear_system_settings)

        self.sol_132(num_modes,
                     u_inf,
                     rho_static,
                     dt,
                     rotationA,
                     panels_wake,
                     horseshoe,
                     gravity_on,                   
                     rigid_body_modes,
                     rigid_modes_cg,
                     use_undamped_modes,
                     write_modal_data,
                     write_modes_vtk,
                     print_modal_matrices,
                     max_modal_disp,
                     max_modal_rot_deg,                     
                     fsi_maxiter,
                     fsi_tolerance,
                     fsi_relaxation,
                     fsi_load_steps,
                     s_maxiter,
                     s_tolerance,
                     s_relaxation,
                     s_load_steps,
                     s_delta_curved,
                     correct_forces_method,
                     correct_forces_settings,
                     primary=False,
                     **kwargs)

        if primary:
            self.modify_settings(self.flow, **kwargs)
            return self.flow, self.settings_new

    def sol_504(self):
        """
        Predefined solution to obtain an aeroelastic Reduced-Order-Model 
        after trimming the aircraft
        """

        pass
