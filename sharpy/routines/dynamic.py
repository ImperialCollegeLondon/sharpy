import numpy as np
from sharpy.routines.basic import Basic
from sharpy.routines.static import Static
import sharpy.utils.algebra as algebra

class Dynamic(Static):
    
    predefined_flows = dict()
    predefined_flows['400'] = ('BeamLoader', 'AerogridLoader', 'NonLinearDynamic')
    predefined_flows['401'] = ('BeamLoader', 'AerogridLoader',
                               'DynamicCoupled')
    predefined_flows['402'] = ('BeamLoader', 'AerogridLoader',
                               'StaticCoupled',
                               'DynamicCoupled')
    predefined_flows['404'] = ('BeamLoader', 'AerogridLoader',
                               'StaticTrim',
                               'DynamicCoupled')

    def __init__(self):
        super().__init__()

def sol_400(self):
    """
    """

    pass

def sol_401(self,
            u_inf,
            rho,
            dt,
            total_time,
            initial_velocity=0.,
            initial_velocity_direction=[-1., 0., 0.],
            panels_wake=1,
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
            s_newmark_damp=1e-3,
            a_convection_scheme=0,
            a_gamma_dot_filtering=0,
            a_num_cores=0,
            velocity_field_generator='SteadyVelocityField',
            velocity_field_input=None,
            primary=True,
            **kwargs):

    """
    """
    self.settings_new['DynamicCoupled'] = self.get_solver_sett('DynamicCoupled')
    self.settings_new['DynamicCoupled']['structural_solver'] = 'NonLinearDynamicCoupledStep'
    self.settings_new['DynamicCoupled']['structural_solver_settings'] = \
        self.get_solver_sett('NonLinearDynamicCoupledStep',
                             gravity_on=gravity_on,             
                             gravity_dir=self.constants['gravity_dir'],
                             gravity=self.constants['gravity'],  
                             initial_velocity=initial_velocity,
                             initial_velocity_direction=initial_velocity_direction,
                             dt=dt,
                             num_steps=round(total_time/dt),
                             newmark_damp=s_newmark_damp,
                             max_iterations=s_maxiter,          
                             min_delta=s_tolerance,                    
                             delta_curved=s_delta_curved,              
                             num_load_steps=s_load_steps,              
                             relaxation_factor=s_relaxation)
    if velocity_field_input is None:
        velocity_field_input= {'u_inf': u_inf,
                               'u_inf_direction': self.constants['u_inf_direction']}
    self.settings_new['DynamicCoupled']['aero_solver'] = 'StepUvlm'
    self.settings_new['DynamicCoupled']['aero_solver_settings'] = \
        self.get_solver_sett('StepUvlm',
                             dt=dt,
                             n_time_steps=round(total_time/dt),
                             rho=rho,
                             print_info=self.constants['print_info'],
                             convection_scheme=a_convection_scheme,
                             gamma_dot_filtering=a_gamma_dot_filtering,
                             num_cores=a_num_cores,
                             
                             
                             
                             
                             

    self.settings_new['DynamicCoupled'][] =
    self.settings_new['DynamicCoupled'][] =
    self.settings_new['DynamicCoupled'][] =
    self.settings_new['DynamicCoupled'][] = 





    
                                      'structural_solver_settings': settings['NonLinearDynamicCoupledStep'],
                                      'aero_solver': 'StepUvlm',
                                      'aero_solver_settings': settings['StepUvlm'],
                                      'fsi_substeps': 100,
                                      'fsi_tolerance': fsi_tolerance,
                                      'relaxation_factor': initial_relaxation_factor,
                                      'final_relaxation_factor': final_relaxation_factor,
                                      'minimum_steps': 1,
                                      'relaxation_steps': relaxation_steps,
                                      'n_time_steps': n_tstep,
                                      'dt': dt,
                                      'include_unsteady_force_contribution': 'on',
                                      'cleanup_previous_solution': 'on',
                                      # the postprocessors in this list are
                                      # called every time step. Some of them
                                      # are the same you call in static simulations
                                      # and you wrote at the beginning of the
                                      # file
                                      'postprocessors': [
                                            # output some variables by text
                                            'WriteVariablesTime',
                                            # remove old timesteps
                                            # already output to save RAM
                                            'Cleanup',
                                            # calculate internal
                                            # beam loads and strains
                                            'BeamLoads',
                                            'BeamPlot',
                                            'AerogridPlot',
                                            # saves a copy of self.data
                                            # (the state of the simulation)
                                            # to restart later if needed.
                                            # careful, because if you don't
                                            # add Cleanup and don't pay attention
                                            # to the 'frequency' and
                                            # 'keep' parameters, you might
                                            # fill you HD for long simulations
                                            'CreateSnapshot',
                                            ],
                                      'postprocessors_settings': {'BeamLoads': {}, # you need to specify the dict even if empty    

def sol_402():
    """
    """


def sol_405():
    """
    """
    

