from sharpy.routines.linear import Linear

class Flutter(Linear):

    predefined_flows = dict()
    #############
    predefined_flows['145'] = ('BeamLoader', 'AerogridLoader',
                               'StaticCoupled',
                               'Modal',
                               'LinearAssembler',
                               'DynamicLoads')
    #############
    predefined_flows['152'] = ('BeamLoader', 'AerogridLoader',
                               'StaticCoupled',
                               'Modal',
                               'LinearAssembler',
                               'DynamicLoads')
    #############
    predefined_flows['152s'] = ('BeamLoader', 'AerogridLoader',
                                'StaticCoupled',
                                'Modal',
                                'LinearAssembler',
                                'AsymptoticStability')

    #############
    predefined_flows['154'] = ('BeamLoader', 'AerogridLoader',
                               'StaticTrim',
                               'Modal',
                               'LinearAssembler',
                               'DynamicLoads')

    def __init__(self):
        super().__init__()

    def sol_145(self,
                flutter_reference,
                velocity_increment,
                flutter_error,
                root_method,
                damping_tolerance,
                u_inf,
                dt,
                rho,
                flutter_upperbound=0,
                flutter_lowerbound=0,
                secant_max_calls=0,
                frequency_cutoff=0,
                save_eigenvalues=True,                
                panels_wake=1,
                num_modes=10,
                inout_coordinates='modes',
                mach_number=0.,                
                newmark_damp=1e-4,
                use_euler=False,
                track_body=True,
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
                primary=True,
                **kwargs
                ):

        """
        Flutter predifined solution in the reference configuration

        Args:
            num_modes: Num modes in the solution
            panels_wake: Number of wake panels
            rho: Air density
            u_inf: Free stream velocity
            c_ref: Reference chord
            dt: for the linear system
            horseshoe: Horseshoe aerodynamic approximation
            root_method: Method to compute the eigenvalaues crossing the x-axis
            flutter_reference: Starting velocity for the algorithm in DynamicLoads \
                               that finds the flutter speed
            velocity_increment: Increment of the velocity to find an interval on which \
                                the damping of the aeroelastic system is positive and \
                                negative at the (thus bounds are found for
                                the flutter velocity).
            flutter_error: Interval of the velocity bounds on which to stop iterations
            damping_tolerance: Fringe above and below 0 where damping criteria is
                               considered (flutter speed taken when damping is above
                               this value; it helps a lot convergence and filtering of
                               modes to have an epsilon rather that 0.)
            frequency_cutoff: Frequency above which eigenvalues are cut. 0 means no \
                              no frequency cut
            gravity_linearise: Take gravity forces in the linear system
            linear_coordinates: Coordinates in the linearised system
            rigid_body_modes:  Include rigid body modes
            rom_method: Name of reduced order model (empty string calculates \
                        the linear system without reducing the model; it might \
                        be too expensive for large models)

        """
        if horseshoe:
            panels_wake = 1        
        if primary:
            predefined_flow = list(self.predefined_flows['145'])
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

        self.settings_new['DynamicLoads'] = {'reference_velocity': flutter_reference,
                                             'velocity_increment': velocity_increment,
                                             'flutter_error':      flutter_error,
                                             'root_method':        root_method,
                                             'flutter_upperbound': flutter_upperbound,
                                             'flutter_lowerbound': flutter_lowerbound,
                                             'secant_max_calls':   secant_max_calls,
                                             'damping_tolerance':  damping_tolerance,
                                             'calculate_flutter':  True,
                                             'frequency_cutoff':   frequency_cutoff,
                                             'save_eigenvalues':   save_eigenvalues
                                            }

        self.sol_502(u_inf,
                     dt,
                     rho,
                     1e-6,
                     panels_wake,
                     num_modes,
                     inout_coordinates,
                     newmark_damp,
                     use_euler,
                     track_body,
                     mach_number,
                     c_ref,
                     gravity_on,
                     horseshoe,
                     rotationA,
                     rom_method,
                     rom_settings,
                     rigid_body_modes,
                     rigid_modes_cg,
                     use_undamped_modes,
                     write_modal_data,
                     write_modes_vtk,
                     max_modal_disp,
                     max_modal_rot_deg,
                     print_modal_matrices,
                     primary=False,
                     **kwargs)

        if primary:
            self.modify_settings(self.flow, **kwargs)
            return self.flow, self.settings_new

    def sol_152(self,
                flutter_reference,
                velocity_increment,
                flutter_error,
                root_method,
                damping_tolerance,
                u_inf,
                dt,
                rho,
                flutter_upperbound=0,
                flutter_lowerbound=0,
                secant_max_calls=0,
                frequency_cutoff=0,
                save_eigenvalues=True,                
                panels_wake=1,
                num_modes=10,
                inout_coordinates='modes',
                mach_number=0.,
                newmark_damp=1e-4,
                use_euler=False,
                track_body=True,
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
                **kwargs
                ):
        """
        Flutter predifined solution after aeroelastic equilibrium
        """
        if horseshoe:
            panels_wake = 1        
        if primary:
            predefined_flow = list(self.predefined_flows['152'])
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

        self.settings_new['DynamicLoads'] = {'reference_velocity': flutter_reference,
                                             'velocity_increment': velocity_increment,
                                             'flutter_error':      flutter_error,
                                             'root_method':        root_method,
                                             'flutter_upperbound': flutter_upperbound,
                                             'flutter_lowerbound': flutter_lowerbound,
                                             'secant_max_calls':   secant_max_calls,
                                             'damping_tolerance':  damping_tolerance,
                                             'calculate_flutter':  True,
                                             'frequency_cutoff':   frequency_cutoff,
                                             'save_eigenvalues':   save_eigenvalues
                                            }

        self.sol_502(u_inf,
                     dt,
                     rho,
                     rho,
                     panels_wake,
                     num_modes,
                     inout_coordinates,
                     newmark_damp,
                     use_euler,
                     track_body,
                     mach_number,
                     c_ref,
                     gravity_on,
                     horseshoe,
                     rotationA,
                     rom_method,
                     rom_settings,
                     rigid_body_modes,
                     rigid_modes_cg,
                     use_undamped_modes,
                     write_modal_data,
                     write_modes_vtk,
                     max_modal_disp,
                     max_modal_rot_deg,
                     print_modal_matrices,
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

    def sol_154(self,
                primary=False,
                **kwargs):
        """
        Flutter predifined solution after trimmed flight
        """
        # if horseshoe:
        #     panels_wake = 1
        if primary:
            self.modify_settings(self.flow, **kwargs)
            return self.flow, self.settings_new


    def sol_152s(self,
                 velocity_analysis,
                 u_inf,
                 dt,
                 rho,
                 num_eigs=300,
                 frequency_cutoff=0,
                 save_eigenvalues=True,                
                 panels_wake=1,
                 num_modes=10,
                 inout_coordinates='modes',
                 mach_number=0.,
                 newmark_damp=1e-4,
                 use_euler=False,
                 track_body=True,
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
                 **kwargs
                 ):
        """
        Flutter predifined solution after aeroelastic equilibrium
        """
        if horseshoe:
            panels_wake = 1        
        if primary:
            predefined_flow = list(self.predefined_flows['152s'])
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

        self.settings_new['AsymptoticStability'] = self.get_solver_sett('AsymptoticStability',
                                                                        velocity_analysis=velocity_analysis,
                                                                        reference_velocity=u_inf,
                                                                        export_eigenvalues=True,
                                                                        num_evals=num_eigs)
        self.sol_502(u_inf,
                     dt,
                     rho,
                     rho,
                     panels_wake,
                     num_modes,
                     inout_coordinates,
                     newmark_damp,
                     use_euler,
                     track_body,
                     mach_number,
                     c_ref,
                     gravity_on,
                     horseshoe,
                     rotationA,
                     rom_method,
                     rom_settings,
                     rigid_body_modes,
                     rigid_modes_cg,
                     use_undamped_modes,
                     write_modal_data,
                     write_modes_vtk,
                     max_modal_disp,
                     max_modal_rot_deg,
                     print_modal_matrices,
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

    #update!!settings_new = basic.update_dic(settings_new, settings)

    
    return flow, settings_new
