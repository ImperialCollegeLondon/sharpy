import numpy as np
import sharpy.routines.basic as basic
import sharpy.utils.algebra as algebra


class Static(basic.Basic):

    predefined_flows = dict()
    predefined_flows['101'] = ('BeamLoader','NonLinearStatic')
    predefined_flows['112'] = ('BeamLoader', 'AerogridLoader', 'StaticCoupled')
    predefined_flows['144'] = ('BeamLoader', 'AerogridLoader', 'StaticTrim')
    
    def __init__(self):
        super().__init__()
        
    def sol_101(self,
                primary=1,
                gravity_on=0,
                s_maxiter=300,
                s_tolerance=1e-8,
                s_delta_curved=1e-3,
                s_load_steps=1,
                s_relaxation=0.01,
                l_ramp=False,
                modify_settings=None,              
                **kwargs):

        """Structural equilibrium"""

        predefined_flow = list(self.predefined_flows['101'])
        if primary:
            
            self.set_constants(**kwargs)
            self.set_flow(predefined_flow, **kwargs)
            self.set_struct_loader(**kwargs)
            self.set_plot(**kwargs)
            
        self.settings_new['NonLinearStatic'] = self.get_solver_sett('NonLinearStatic',
                                             gravity_on=gravity_on,
                                             gravity_dir=self.constants['gravity_dir'],
                                             gravity=self.constants['gravity'],
                                             initial_position=self.constants['forA'],
                                             max_iterations=s_maxiter,
                                             min_delta=s_tolerance,
                                             delta_curved=s_delta_curved,
                                             num_load_steps=s_load_steps,
                                             load_ramping=l_ramp,
                                             relaxation_factor=s_relaxation)
        if primary:
            if modify_settings is not None:
                self.settings_new = basic.update_dic(self.settings_new, modify_settings)
            return self.flow, self.settings_new

    def sol_112(self,
                u_inf,
                rho,
                dt,
                primary=1,
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

        """ Aeroelastic equilibrium"""

        predefined_flow = list(self.predefined_flows['102'])
        if horseshoe:
            panels_wake = 1
        if primary:
            self.set_constants(**kwargs)            
            self.set_flow(predefined_flow, **kwargs)
            self.set_loaders(**kwargs)
            self.set_plot(u_inf,
                          dt,
                          **kwargs)

        self.settings_new['StaticCoupled']['n_load_steps'] = fsi_load_steps
        self.settings_new['StaticCoupled']['max_iter'] = fsi_maxiter
        self.settings_new['StaticCoupled']['tolerance'] = fsi_tolerance
        self.settings_new['StaticCoupled']['relaxation_factor'] = fsi_relaxation
        self.settings_new['StaticCoupled']['aero_solver'] = 'StaticUvlm'
        self.settings_new['StaticCoupled']['aero_solver_settings'] = self.get_solver_sett('StaticUvlm',
                                                                     rho=rho,
                                                                     horseshoe=horseshoe,
                                                                     num_cores=self.constants['num_cores'],
                                                                     n_rollup=int(1.2*panels_wake),
                                                                     rollup_dt=dt,
                                                                     velocity_field_generator= \
                                                                     'SteadyVelocityField',
                                                                     velocity_field_input= \
                                                                     {'u_inf': u_inf,              
                                                                     'u_inf_direction':\
                                                                     self.constants['u_inf_direction']})
                       
        self.settings_new['StaticCoupled']['structural_solver'] = 'NonLinearStatic'
        self.settings_new['StaticCoupled']['structural_solver_settings'] = self.get_solver_sett('NonLinearStatic', 
                                                                           gravity_on=gravity_on,                    
                                                                           gravity_dir=self.constants['gravity_dir'],
                                                                           gravity=self.constants['gravity'],        
                                                                           initial_position=self.constants['forA'],  
                                                                           max_iterations=s_maxiter,                 
                                                                           min_delta=s_tolerance,                    
                                                                           delta_curved=s_delta_curved,              
                                                                           num_load_steps=s_load_steps,              
                                                                           relaxation_factor=s_relaxation)           

        if correct_forces_method is not None:
            self.settings_new['StaticCoupled']['correct_forces_method'] = correct_forces_method
            self.settings_new['StaticCoupled']['correct_forces_settings'] = correct_forces_settings

        if primary:
            if modify_settings is not None:
                self.settings_new = basic.update_dic(self.settings_new,
                                               modify_settings)
            return self.flow, self.settings_new
        
    def sol_144(self,
                u_inf,                        # Free stream velocity
                rho,                          # Air density 
                panels_wake,                  # Number of wake panels
                dt,                           # dt for uvlm                 
                alpha0,                       # Initial angle of attack
                thrust0,                      # Number of     
                cs0,                          # Number of wake panels 
                thrust_nodes,                 # Nodes where thrust is applied
                cs_i,                         # Indices of control surfaces to be trimmed
                primary=1,
                horseshoe='off',              # Horseshoe aerodynamic approximation
                nz=1.,                        # Gravity factor for manoeuvres          
                Dcs0=0.01,                    # Initial control surface deflection
                Dthrust0=0.1,                 # Initial thrust variation 
                fx_tolerance=0.01,            # Tolerance in horizontal  force
                fz_tolerance=0.01,            # Tolerance in vertical force
                pitching_tolerance=0.01,      # Tolerance in pitching 
                trim_max_iter=100,            # Mximum number of trim iterations
                trim_relaxation_factor=0.2,   # Relaxation factor 
                fsi_tolerance=1e-5,           # FSI loop tolerance
                fsi_relaxation=0.1,           # FSI relaxation_factor
                fsi_maxiter=100,              # FSI maximum number of iterations
                fsi_load_steps=1,
                s_maxiter=100,
                s_tolerance=1e-5,             # Tolerance of structural solver
                s_relaxation=1e-3,
                s_load_steps=1,
                s_delta_curved=1e-4,
                modify_settings = None,
                **kwargs):
        
        """ Longitudinal aircraft trim"""

        predefined_flow = list(self.predefined_flows['144'])
        if horseshoe:
            panels_wake = 1        
        if primary:
            self.set_constants(**kwargs)
            self.set_flow(predefined_flow, **kwargs)
            self.set_loaders(**kwargs)
            self.set_plot(**kwargs)

        gravity_on = True
        self.settings_new['StaticTrim']['initial_alpha'] = alpha0
        self.settings_new['StaticTrim']['initial_deflection'] = cs0
        self.settings_new['StaticTrim']['initial_angle_eps'] = Dcs0
        self.settings_new['StaticTrim']['initial_thrust'] = thrust0
        self.settings_new['StaticTrim']['initial_thrust_eps'] = Dthrust0
        self.settings_new['StaticTrim']['thrust_nodes'] = thrust_nodes
        self.settings_new['StaticTrim']['tail_cs_index'] = cs_i
        self.settings_new['StaticTrim']['fx_tolerance'] = fx_tolerance
        self.settings_new['StaticTrim']['fz_tolerance'] = fz_tolerance
        self.settings_new['StaticTrim']['m_tolerance'] = pitching_tolerance
        self.settings_new['StaticTrim']['max_iter'] = trim_max_iter
        self.settings_new['StaticTrim']['relaxation_factor'] = trim_relaxation_factor
        self.settings_new['StaticTrim']['save_info'] = 'on'
        self.settings_new['StaticTrim']['solver'] = 'StaticCoupled'
        self.settings_new['StaticTrim']['solver_settings'] = \
            {'print_info': 'on',
             'structural_solver': 'NonLinearStatic',
             'structural_solver_settings': self.get_solver_sett('NonLinearStatic',   
                                           gravity_on=gravity_on,                    
                                           gravity_dir=self.constants['gravity_dir'],
                                           gravity=self.constants['gravity'],        
                                           initial_position=self.constants['forA'],  
                                           max_iterations=s_maxiter,                 
                                           min_delta=s_tolerance,                    
                                           delta_curved=s_delta_curved,              
                                           num_load_steps=s_load_steps,              
                                           relaxation_factor=s_relaxation),
             'aero_solver': 'StaticUvlm',
             'aero_solver_settings':  self.get_solver_sett('StaticUvlm',   
                                      rho=rho,                              
                                      horseshoe=horseshoe,                  
                                      num_cores=self.constants['num_cores'],
                                      n_rollup=int(1.2*panels_wake),        
                                      rollup_dt=dt,                         
                                      velocity_field_generator= \
                                      'SteadyVelocityField',                
                                      velocity_field_input= \
                                      {'u_inf': u_inf,                      
                                       'u_inf_direction':\
                                       self.constants['u_inf_direction']}),             
             'max_iter': fsi_maxiter,
             'n_load_steps': fsi_load_steps,
             'tolerance': fsi_tolerance,
             'relaxation_factor': fsi_relaxation
            }
        if primary:
            if modify_settings is not None:
                self.settings_new = basic.update_dic(self.settings_new, modify_settings)
            return self.flow, self.settings_new

    def sol_148(self):
        """ Aircraft general trim"""

        return self.flow, self.settings_new

# def sol_101(
#             forA=[0.,0.,0.],
#             gravity_on=0,
#             max_iterations=300,
#             tolerance=1e-8,
#             num_load_steps=200,
#             newmark_damp=1e-10,
#             add_to_flow = None,                
#             flow=[], **settings):

#     """Structural equilibrium"""

#     settings_new = dict()
#     if flow == []:
#         flow = ['BeamLoader','NonLinearStatic']
#     for k in flow:
#         settings_new[k] = {}

#     settings_new['BeamLoader']['for_pos'] = forA
#     settings_new['BeamLoader']['orientation'] = [1.0, 0, 0, 0]
#     settings_new['BeamLoader']['unsteady'] = 'off'
#     settings_new['NonLinearStatic']['gravity_on'] = gravity_on
#     settings_new['NonLinearStatic']['initial_position'] = forA
#     settings_new['NonLinearStatic']['max_iterations'] = max_iterations
#     settings_new['NonLinearStatic']['min_delta'] = tolerance
#     settings_new['NonLinearStatic']['delta_curved'] = 1e-5
#     settings_new['NonLinearStatic']['num_load_steps'] = num_load_steps
#     settings_new['NonLinearStatic']['newmark_damp'] = newmark_damp
#     settings_new['NonLinearStatic']['relaxation_factor'] = 0.1
#     settings_new['NonLinearStatic']['dt'] = 0.001
#     settings_new['NonLinearStatic']['num_steps'] = 1000
#     settings_new = basic.update_dic(settings_new, settings)        
#     return flow, settings_new
