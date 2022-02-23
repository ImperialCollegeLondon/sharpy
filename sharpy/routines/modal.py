import numpy as np
from sharpy.routines.basic import Basic
from sharpy.routines.static import Static
import sharpy.utils.algebra as algebra

class Modal(Static):

    #global predefined_flows
    predefined_flows = dict()
    predefined_flows['103'] = ('BeamLoader', 'Modal')
    predefined_flows['132'] = ('BeamLoader', 'AerogridLoader',
                               'StaticCoupled',
                               'Modal')
    predefined_flows['134'] = ('BeamLoader', 'AerogridLoader',
                               'StaticTrim',
                               'Modal')

    def __init__(self):
        super().__init__()

    def sol_103(self,
                num_modes,
                rigid_body_modes=False,
                rigid_modes_cg=False,
                use_undamped_modes=True,
                write_modal_data=True,
                write_modes_vtk=True,
                max_modal_disp=0.15,
                max_modal_rot_deg=15.,                
                print_modal_matrices=False,
                primary=True,
                **kwargs):

        """
        Modal solution (stiffness and mass matrices, and natural frequencies)
        in the reference configuration
        """

        if primary:
            predefined_flow = list(self.predefined_flows['103'])
            self.set_constants(**kwargs)
            self.set_flow(predefined_flow, **kwargs)
            self.set_struct_loader(**kwargs)
            self.set_plot(**kwargs)

        self.settings_new['Modal'] = self.get_solver_sett('Modal',
                                    NumLambda=num_modes,
                                    rigid_body_modes=rigid_body_modes,
                                    rigid_modes_cg=rigid_modes_cg,
                                    use_undamped_modes=use_undamped_modes,
                                    max_displacement=max_modal_disp,
                                    max_rotation_deg=max_modal_rot_deg,
                                    print_matrices=print_modal_matrices,
                                    write_dat=write_modal_data,
                                    write_modes_vtk=write_modes_vtk)

        if primary:
            self.modify_settings(self.flow, **kwargs)
            return self.flow, self.settings_new

    def sol_132(self,
                num_modes,
                u_inf,
                rho,
                dt,
                rotationA=[0., 0., 0.],
                panels_wake=1,
                horseshoe=False,
                gravity_on=0,                
                rigid_body_modes=False,
                rigid_modes_cg=False,
                use_undamped_modes=True,
                write_modal_data=True,
                write_modes_vtk=True,
                print_modal_matrices=False,
                max_modal_disp=0.15,
                max_modal_rot_deg=15.,
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
        Modal solution (stiffness and mass matrices, and natural frequencies)
        in a deformed structural or aeroelastic configuration
        """
        if horseshoe:
            panels_wake = 1
        if primary:
            predefined_flow = list(self.predefined_flows['132'])
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

        self.settings_new['Modal'] = self.get_solver_sett('Modal',
                                    NumLambda=num_modes,
                                    rigid_body_modes=rigid_body_modes,
                                    rigid_modes_cg=rigid_modes_cg,
                                    use_undamped_modes=use_undamped_modes,
                                    max_displacement=max_modal_disp,
                                    max_rotation_deg=max_modal_rot_deg,
                                    print_matrices=print_modal_matrices,
                                    write_dat=write_modal_data,
                                    write_modes_vtk=write_modes_vtk)

        self.sol_112(u_inf,
                     rho,
                     dt,
                     rotationA,
                     panels_wake,
                     horseshoe,
                     gravity_on,
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
                     False,
                     **kwargs)

        if primary:
            self.modify_settings(self.flow, **kwargs)
            return self.flow, self.settings_new

    def sol_134(self,
                num_modes,
                u_inf,
                rho,
                panels_wake,
                dt,
                alpha0,
                thrust0,
                cs0,
                thrust_nodes,
                cs_i,
                horseshoe='off',
                rigid_body_modes=False,
                rigid_modes_cg=False,
                use_undamped_modes=True,
                write_modal_data=True,
                write_modes_vtk=True,
                print_modal_matrices=False,
                nz=1.,
                Dcs0=0.01,
                Dthrust0=0.1,
                fx_tolerance=0.01,
                fz_tolerance=0.01,
                pitching_tolerance=0.01,
                trim_max_iter=100,
                trim_relaxation_factor=0.2,
                fsi_tolerance=1e-5,
                fsi_relaxation=0.1,
                fsi_maxiter=100,
                fsi_load_steps=1,
                s_maxiter=100,
                s_tolerance=1e-5,
                s_relaxation=1e-3,
                s_load_steps=1,
                s_delta_curved=1e-4,
                primary=True,
                **kwargs):

        """
        Modal solution (stiffness and mass matrices, and natural frequencies)
        in a deformed  configuration after trim analysis
        """
        if horseshoe:
            panels_wake = 1
        if primary:
            predefined_flow = list(self.predefined_flows['134'])
            self.set_constants(**kwargs)
            self.set_flow(predefined_flow, **kwargs)
            self.set_loaders(panels_wake,
                             u_inf,
                             dt,
                             unsteady=False,
                             **kwargs)
            self.set_plot(u_inf,
                          dt,
                          **kwargs)

        self.settings_new['Modal'] = self.get_solver_sett('Modal',
                                    NumLambda=num_modes,
                                    rigid_body_modes=rigid_body_modes,
                                    rigid_modes_cg=rigid_modes_cg,
                                    use_undamped_modes=use_undamped_modes,
                                    print_matrices=print_modal_matrices,
                                    write_dat=write_modal_data,
                                    write_modes_vtk=write_modes_vtk)

        self.sol_144(u_inf,
                     rho,
                     panels_wake,
                     dt,
                     alpha0,
                     thrust0,
                     cs0,
                     thrust_nodes,
                     cs_i,
                     horseshoe,
                     nz,
                     Dcs0,
                     Dthrust0,
                     fx_tolerance,
                     fz_tolerance,
                     pitching_tolerance,
                     trim_max_iter,
                     trim_relaxation_factor,
                     fsi_tolerance,
                     fsi_relaxation,
                     fsi_maxiter,
                     fsi_load_steps,
                     s_maxiter,
                     s_tolerance,
                     s_relaxation,
                     s_load_steps,
                     s_delta_curved,
                     primary=False)

        if primary:
            self.modify_settings(self.flow, **kwargs)
            return self.flow, self.settings_new
