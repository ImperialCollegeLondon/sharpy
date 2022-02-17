import numpy as np
from sharpy.routines.basic import Basic, update_dic
from sharpy.routines.static import Static
import sharpy.utils.algebra as algebra

class Modal(Static, Basic):
    
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
                primary=1,
                forA=[0., 0., 0.],
                rigid_body_modes=False,
                rigid_modes_cg=False,
                use_undamped_modes=True,
                modify_settings=None,              
                **kwargs):

        """
        Modal solution (stiffness and mass matrices, and natural frequencies)
        in the reference configuration
        """

        predefined_flow = list(self.predefined_flows['103'])
        if primary:

            self.set_constants(**kwargs)
            self.set_flow(predefined_flow, **kwargs)
            self.set_struct_loader(**kwargs)
            self.set_plot(**kwargs)

        self.settings_new['Modal'] = self.get_solver_sett('Modal',
                                    NumLambda=num_modes,
                                    rigid_body_modes=rigid_body_modes,
                                    rigid_modes_cg=rigid_modes_cg,
                                    use_undamped_modes=use_undamped_modes,
                                    print_matrices=True,
                                    write_modes_vtk=True)

        if primary:
            self.modify_settings(self.flow, **kwargs)
            return self.flow, self.settings_new

    def sol_132(self,
                num_modes,
                u_inf,
                rho,
                dt,
                primary=1,
                forA=[0., 0., 0.],
                rigid_body_modes=False,
                rigid_modes_cg=False,
                use_undamped_modes=True,                                  
                rotationA=[1.0, 0., 0., 0.],
                panels_wake=1,              
                horseshoe=False,            
                gravity_on=0,               
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
                modify_settings=None,       
                **kwargs):

        """
        Modal solution (stiffness and mass matrices, and natural frequencies)
        in a deformed structural or aeroelastic configuration
        """

        predefined_flow = list(self.predefined_flows['132'])
        if primary:

            self.set_constants(**kwargs)
            self.set_flow(predefined_flow, **kwargs)
            self.set_struct_loader(**kwargs)
            self.set_plot(**kwargs)

        self.settings_new['Modal'] = self.get_solver_sett('Modal',
                                    NumLambda=num_modes,
                                    rigid_body_modes=rigid_body_modes,
                                    rigid_modes_cg=rigid_modes_cg,
                                    use_undamped_modes=use_undamped_modes,
                                    print_matrices=True,
                                    write_modes_vtk=True)

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
                     primary=False,
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
                primary=1,                 
                horseshoe='off',
                rigid_body_modes=False,
                rigid_modes_cg=False,
                use_undamped_modes=True,                                                  
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
                modify_settings = None,
                **kwargs):
                        
        """
        Modal solution (stiffness and mass matrices, and natural frequencies)
        in a deformed  configuration after trim analysis
        """
                
        predefined_flow = list(self.predefined_flows['134'])
        if primary:

            self.set_constants(**kwargs)
            self.set_flow(predefined_flow, **kwargs)
            self.set_loaders(**kwargs)
            self.set_plot(**kwargs)

        self.settings_new['Modal'] = self.get_solver_sett('Modal',
                                    NumLambda=num_modes,
                                    rigid_body_modes=rigid_body_modes,
                                    rigid_modes_cg=rigid_modes_cg,
                                    use_undamped_modes=use_undamped_modes,
                                    print_matrices=True,
                                    write_modes_vtk=True)

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
