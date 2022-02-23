from sharpy.routines.linear import Linear

class Flutter(Linear):

    predefined_flows = dict()
    #############
    predefined_flows['145'] = ['BeamLoader', 'AerogridLoader',
                               'StaticCoupled',
                               'Modal',
                               'LinearAssembler',
                               'DynamicLoads']
    #############
    predefined_flows['152'] = ['BeamLoader', 'AerogridLoader',
                               'StaticCoupled',
                               'Modal',
                               'LinearAssembler',
                               'DynamicLoads']
    #############
    predefined_flows['154'] = ['BeamLoader', 'AerogridLoader',
                               'StaticTrim',
                               'Modal',
                               'LinearAssembler',
                               'DynamicLoads']

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
                frequency_cutoff=0,
                save_eigenvalues=True,                
                panels_wake=1,
                num_modes=10,
                inout_coordinates='modal',
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
                frequency_cutoff=0,
                save_eigenvalues=True,                
                panels_wake=1,
                num_modes=10,
                inout_coordinates='modal',
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
