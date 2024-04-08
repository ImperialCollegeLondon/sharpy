import numpy as np

import sharpy.utils.cout_utils as cout
import sharpy.utils.solver_interface as solver_interface
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings_utils
import os


@solver
class DynamicTrim(BaseSolver):
    """
    The ``StaticTrim`` solver determines the longitudinal state of trim (equilibrium) for an aeroelastic system in
    static conditions. It wraps around the desired solver to yield the state of trim of the system, in most cases
    the :class:`~sharpy.solvers.staticcoupled.StaticCoupled` solver.

    It calculates the required angle of attack, elevator deflection and thrust required to achieve longitudinal
    equilibrium. The output angles are shown in degrees.

    The results from the trimming iteration can be saved to a text file by using the `save_info` option.
    """
    solver_id = 'DynamicTrim'
    solver_classification = 'Flight Dynamics'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['print_info'] = 'bool'
    settings_default['print_info'] = True
    settings_description['print_info'] = 'Print info to screen'

    settings_types['solver'] = 'str'
    settings_default['solver'] = ''
    settings_description['solver'] = 'Solver to run in trim routine'

    settings_types['solver_settings'] = 'dict'
    settings_default['solver_settings'] = dict()
    settings_description['solver_settings'] = 'Solver settings dictionary'

    settings_types['max_iter'] = 'int'
    settings_default['max_iter'] = 40000
    settings_description['max_iter'] = 'Maximum number of iterations of trim routine'

    settings_types['fz_tolerance'] = 'float'
    settings_default['fz_tolerance'] = 0.01
    settings_description['fz_tolerance'] = 'Tolerance in vertical force'

    settings_types['fx_tolerance'] = 'float'
    settings_default['fx_tolerance'] = 0.01
    settings_description['fx_tolerance'] = 'Tolerance in horizontal force'

    settings_types['m_tolerance'] = 'float'
    settings_default['m_tolerance'] = 0.01
    settings_description['m_tolerance'] = 'Tolerance in pitching moment'

    settings_types['tail_cs_index'] = ['int', 'list(int)']
    settings_default['tail_cs_index'] = 0
    settings_description['tail_cs_index'] = 'Index of control surfaces that move to achieve trim'

    settings_types['thrust_nodes'] = 'list(int)'
    settings_default['thrust_nodes'] = [0]
    settings_description['thrust_nodes'] = 'Nodes at which thrust is applied'

    settings_types['initial_alpha'] = 'float'
    settings_default['initial_alpha'] = 0.
    settings_description['initial_alpha'] = 'Initial angle of attack'

    settings_types['initial_deflection'] = 'float'
    settings_default['initial_deflection'] = 0.
    settings_description['initial_deflection'] = 'Initial control surface deflection'

    settings_types['initial_thrust'] = 'float'
    settings_default['initial_thrust'] = 0.0
    settings_description['initial_thrust'] = 'Initial thrust setting'

    settings_types['initial_angle_eps'] = 'float'
    settings_default['initial_angle_eps'] = 0.05
    settings_description['initial_angle_eps'] = 'Initial change of control surface deflection'

    settings_types['initial_thrust_eps'] = 'float'
    settings_default['initial_thrust_eps'] = 2.
    settings_description['initial_thrust_eps'] = 'Initial thrust setting change'

    settings_types['relaxation_factor'] = 'float'
    settings_default['relaxation_factor'] = 0.2
    settings_description['relaxation_factor'] = 'Relaxation factor'

    settings_types['notrim_relax'] = 'bool'
    settings_default['notrim_relax'] = False
    settings_description['notrim_relax'] = 'Disable gains for trim - releases internal loads at initial values'

    settings_types['notrim_relax_iter'] = 'int'
    settings_default['notrim_relax_iter'] = 10000000
    settings_description['notrim_relax_iter'] = 'Terminate notrim_relax at defined number of steps'


    settings_types['speed_up_factor'] = 'float'
    settings_default['speed_up_factor'] = 1.0
    settings_description['speed_up_factor'] = 'increase dt in trim iterations'

    settings_types['save_info'] = 'bool'
    settings_default['save_info'] = False
    settings_description['save_info'] = 'Save trim results to text file'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.data = None
        self.settings = None
        self.solver = None

        # The order is
        # [0]: alpha/fz
        # [1]: alpha + delta (gamma)/moment
        # [2]: thrust/fx

        self.n_input = 3
        self.i_iter = 0

        self.input_history = []
        self.output_history = []
        self.gradient_history = []
        self.trimmed_values = np.zeros((3,))

        self.table = None
        self.folder = None

    def initialise(self, data, restart=False):
        self.data = data
        self.settings = data.settings[self.solver_id]
        settings_utils.to_custom_types(self.settings, self.settings_types, self.settings_default)

        self.solver = solver_interface.initialise_solver(self.settings['solver'])

        if self.settings['solver_settings']['structural_solver'] == "NonLinearDynamicCoupledStep":
            #replace free flying with clamped
            oldsettings = self.settings['solver_settings']
            self.settings['solver_settings']['structural_solver'] = 'NonLinearDynamicPrescribedStep'
            # self.settings['solver_settings']['structural_solver_settings'] = {'print_info': 'off',
            #                                                                 'max_iterations': 950,
            #                                                                 'delta_curved': 1e-1,
            #                                                                 'min_delta': tolerance,
            #                                                                 'newmark_damp': 5e-3,
            #                                                                 'gravity_on': gravity,
            #                                                                 'gravity': 9.81,
            #                                                                 'num_steps': n_tstep,
            #                                                                 'dt': dt,
            #                                                                 'initial_velocity': u_inf * int(free_flight)}  commented since both solvers take (almost) same inputs
            u_inf = self.settings['solver_settings']['structural_solver_settings']['initial_velocity']
            self.settings['solver_settings']['structural_solver_settings'].pop('initial_velocity')
            self.settings['solver_settings']['structural_solver_settings']['newmark_damp'] = 1.0
            # settings['StepUvlm'] = {'print_info': 'off',
            #                     'num_cores': num_cores,
            #                     'convection_scheme': 2,
            #                     'gamma_dot_filtering': 6,
            #                     'velocity_field_generator': 'GustVelocityField',
            #                     'velocity_field_input': {'u_inf': int(not free_flight) * u_inf,
            #                                              'u_inf_direction': [1., 0, 0],
            #                                              'gust_shape': '1-cos',
            #                                              'gust_parameters': {'gust_length': gust_length,
            #                                                                  'gust_intensity': gust_intensity * u_inf},
            #                                              'offset': gust_offset,
            #                                              'relative_motion': relative_motion},
            #                     'rho': rho,
            #                     'n_time_steps': n_tstep,
            #                     'dt': dt}
            self.settings['solver_settings']['aero_solver_settings']['velocity_field_generator'] = 'SteadyVelocityField'
            u_inf_direction = self.settings['solver_settings']['aero_solver_settings']['velocity_field_input']['u_inf_direction']
            self.settings['solver_settings']['aero_solver_settings']['velocity_field_input'].clear()
            self.settings['solver_settings']['aero_solver_settings']['velocity_field_input']['u_inf'] = u_inf
            self.settings['solver_settings']['aero_solver_settings']['velocity_field_input']['u_inf_direction'] = u_inf_direction

            self.settings['solver_settings']
            # TODO: plus dynamic coupled to add controller

            # import pdb
            # pdb.set_trace()

            gains_elev = -np.array([0.00015, 0.00015, 0.0])   #we can pick them! dimensional force/moment versus radians
            gains_alpha = np.array([0.00015, 0.00015, 0.0])
            gains_thrust = -np.array([0.1, 0.05, 0.0])        #we can pick them too! this is 1-1 almost
            route = data.settings['SHARPy']['route']
            n_tstep = int(self.settings['solver_settings']['n_time_steps'])
            dt = float(self.settings['solver_settings']['dt'])
            elev_file = route + '/elev.csv'
            alpha_file = route + '/alpha.csv'
            thrust_file = route + '/thrust.csv'

            elev_hist = np.linspace(0, n_tstep*dt, n_tstep)
            elev_hist = 0.0/180.0*np.pi*elev_hist

            alpha_hist = np.linspace(1, 1, n_tstep)
            alpha_hist = 0.0/180.0*np.pi*alpha_hist

            thrust_hist = np.linspace(1, 1, n_tstep)
            thrust_hist = 0.0/180.0*np.pi*thrust_hist

            np.savetxt(elev_file, elev_hist)
            np.savetxt(alpha_file, alpha_hist)
            np.savetxt(thrust_file, thrust_hist)


            try:
                self.settings['solver_settings']['controller_id'].clear()
                self.settings['solver_settings']['controller_settings'].clear()
            except:
                print("original no controller")    

            self.settings['solver_settings']['controller_id']= {'controller_elevator': 'ControlSurfacePidController',
                                                                'controller_alpha': 'AlphaController',
                                                                'controller_thrust': 'ThrustController'}
            self.settings['solver_settings']['controller_settings']= {'controller_elevator': {'P': gains_elev[0],
                                                                                'I': gains_elev[1],
                                                                                'D': gains_elev[2],
                                                                                'dt': dt,
                                                                                'input_type': 'm_y',
                                                                                'write_controller_log': True,
                                                                                'controlled_surfaces': [0],
                                                                                'time_history_input_file': route + '/elev.csv',
                                                                                'use_initial_value': True,
                                                                                'initial_value': self.settings['initial_deflection']
                                                                                }
                                                                                ,
                                                                    'controller_alpha': {'P': gains_alpha[0],
                                                                                'I': gains_alpha[1],
                                                                                'D': gains_alpha[2],
                                                                                'dt': dt,
                                                                                'input_type': 'f_z',
                                                                                'write_controller_log': True,
                                                                                'time_history_input_file': route + '/alpha.csv',
                                                                                'use_initial_value': True,
                                                                                'initial_value': self.settings['initial_alpha']}
                                                                                ,
                                                                    'controller_thrust': {'thrust_nodes': self.settings['thrust_nodes'],
                                                                                'P': gains_thrust[0],
                                                                                'I': gains_thrust[1],
                                                                                'D': gains_thrust[2],
                                                                                'dt': dt,
                                                                                'input_type': 'f_x',
                                                                                'write_controller_log': True,
                                                                                'time_history_input_file': route + '/thrust.csv',
                                                                                'use_initial_value': True,
                                                                                'initial_value': self.settings['initial_thrust']}
                                                                                }
            # import pdb
            # pdb.set_trace()
            # #end

            self.solver.initialise(self.data, self.settings['solver_settings'], restart=restart)

            self.folder = data.output_folder + '/statictrim/'
            if not os.path.exists(self.folder):
                os.makedirs(self.folder)

            self.table = cout.TablePrinter(10, 8, ['g', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f'],
                                        filename=self.folder+'trim_iterations.txt')
            self.table.print_header(['iter', 'alpha[deg]', 'elev[deg]', 'thrust', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'])
        
        
        elif self.settings['solver_settings']['structural_solver'] == "NonLinearDynamicMultibody": 
            # import pdb
            # pdb.set_trace()
            self.data.structure.ini_mb_dict['body_00']['FoR_movement'] = 'prescribed_trim'
            self.data.structure.timestep_info[0].mb_dict['body_00']['FoR_movement'] = 'prescribed_trim'
            self.data.structure.ini_info.mb_dict['body_00']['FoR_movement'] = 'prescribed_trim'  
            # self.data.structure.ini_mb_dict['body_00']['FoR_movement'] = 'prescribed_trim'
            # self.data.structure.timestep_info[0].mb_dict['body_00']['FoR_movement'] = 'prescribed_trim'
            # self.data.structure.ini_info.mb_dict['body_00']['FoR_movement'] = 'prescribed_trim'                
            #replace free flying with clamped
            oldsettings = self.settings['solver_settings']
            self.settings['solver_settings']['structural_solver'] = 'NonLinearDynamicMultibody'
            # self.settings['solver_settings']['structural_solver_settings'] = {'print_info': 'off',
            #                                                                 'max_iterations': 950,
            #                                                                 'delta_curved': 1e-1,
            #                                                                 'min_delta': tolerance,
            #                                                                 'newmark_damp': 5e-3,
            #                                                                 'gravity_on': gravity,
            #                                                                 'gravity': 9.81,
            #                                                                 'num_steps': n_tstep,
            #                                                                 'dt': dt,
            #                                                                 'initial_velocity': u_inf * int(free_flight)}  commented since both solvers take (almost) same inputs
            self.settings['solver_settings']['structural_solver_settings'].pop('dyn_trim')
            u_inf = self.settings['solver_settings']['structural_solver_settings']['initial_velocity']
            self.settings['solver_settings']['structural_solver_settings'].pop('initial_velocity')
            # import pdb
            # pdb.set_trace()
            dt = self.settings['solver_settings']['structural_solver_settings']['time_integrator_settings']['dt']
            # self.settings['solver_settings']['structural_solver_settings']['time_integrator_settings']['dt'] = float(dt)/5.
            self.settings['solver_settings']['structural_solver_settings']['time_integrator_settings']['newmark_damp'] = 0.1
            # settings['StepUvlm'] = {'print_info': 'off',
            #                     'num_cores': num_cores,
            #                     'convection_scheme': 2,
            #                     'gamma_dot_filtering': 6,
            #                     'velocity_field_generator': 'GustVelocityField',
            #                     'velocity_field_input': {'u_inf': int(not free_flight) * u_inf,
            #                                              'u_inf_direction': [1., 0, 0],
            #                                              'gust_shape': '1-cos',
            #                                              'gust_parameters': {'gust_length': gust_length,
            #                                                                  'gust_intensity': gust_intensity * u_inf},
            #                                              'offset': gust_offset,
            #                                              'relative_motion': relative_motion},
            #                     'rho': rho,
            #                     'n_time_steps': n_tstep,
            #                     'dt': dt}
            # self.settings['solver_settings']['aero_solver_settings']['convection_scheme'] = 2 

            self.settings['solver_settings']['aero_solver_settings']['velocity_field_generator'] = 'SteadyVelocityField'
            u_inf_direction = self.settings['solver_settings']['aero_solver_settings']['velocity_field_input']['u_inf_direction']
            self.settings['solver_settings']['aero_solver_settings']['velocity_field_input'].clear()
            self.settings['solver_settings']['aero_solver_settings']['velocity_field_input']['u_inf'] = u_inf
            self.settings['solver_settings']['aero_solver_settings']['velocity_field_input']['u_inf_direction'] = u_inf_direction

            self.settings['solver_settings']
            # TODO: plus dynamic coupled to add controller

            # import pdb
            # pdb.set_trace()

            # gains_elev = -np.array([0.000015, 0.000015, 0.0])   #we can pick them! dimensional force/moment versus radians
            # gains_alpha = np.array([0.000010, 0.000010, 0.0])
            # # gains_elev = -np.array([0.00015, 0.00015, 0.0])   #we can pick them! dimensional force/moment versus radians
            # # gains_alpha = np.array([0.00010, 0.00010, 0.0])            
            # gains_thrust = -np.array([0.1, 0.05, 0.0])        #we can pick them too! this is 1-1 almost
            gains_elev = (not self.settings['notrim_relax'])*-np.array([0.000015, 0.000010, 0.0])   #we can pick them! dimensional force/moment versus radians
            gains_alpha = (not self.settings['notrim_relax'])*np.array([0.000015, 0.000010, 0.0])
            # gains_elev = -np.array([0.00015, 0.00015, 0.0])   #we can pick them! dimensional force/moment versus radians
            # gains_alpha = np.array([0.00010, 0.00010, 0.0])            
            gains_thrust = (not self.settings['notrim_relax'])*-np.array([0.1, 0.05, 0.0])  
            
            

            #we can pick them too! this is 1-1 almost            
            route = data.settings['SHARPy']['route']
            n_tstep = int(self.settings['solver_settings']['n_time_steps'])
            n_tstep *=40
            self.settings['solver_settings']['n_time_steps'] = n_tstep
            self.settings['solver_settings']['structural_solver_settings']['num_steps'] = n_tstep
            self.settings['solver_settings']['aero_solver_settings']['n_time_steps'] = n_tstep
            # import pdb
            # pdb.set_trace()
            
            dt = float(self.settings['solver_settings']['dt'])*self.settings['speed_up_factor']
            # import pdb
            # pdb.set_trace()
            self.settings['solver_settings']['dt'] = dt
            self.settings['solver_settings']['structural_solver_settings']['time_integrator_settings']['dt'] = dt
            self.settings['solver_settings']['aero_solver_settings']['dt'] = dt
            elev_file = route + '/elev.csv'
            alpha_file = route + '/alpha.csv'
            thrust_file = route + '/thrust.csv'

            elev_hist = np.linspace(0, n_tstep*dt, n_tstep)
            elev_hist = 0.0/180.0*np.pi*elev_hist

            alpha_hist = np.linspace(1, 1, n_tstep)
            alpha_hist = 0.0/180.0*np.pi*alpha_hist

            thrust_hist = np.linspace(1, 1, n_tstep)
            thrust_hist = 0.0/180.0*np.pi*thrust_hist

            np.savetxt(elev_file, elev_hist)
            np.savetxt(alpha_file, alpha_hist)
            np.savetxt(thrust_file, thrust_hist)

            try:
                self.settings['solver_settings']['controller_id'].clear()
                self.settings['solver_settings']['controller_settings'].clear()
            except:
                print("original no controller")    

            self.settings['solver_settings']['controller_id']= {'controller_elevator': 'ControlSurfacePidController'
                                                                ,
                                                                'controller_alpha': 'AlphaController'
                                                                ,
                                                                'controller_thrust': 'ThrustController'
                                                                }
            self.settings['solver_settings']['controller_settings']= {'controller_elevator': {'P': gains_elev[0],
                                                                                'I': gains_elev[1],
                                                                                'D': gains_elev[2],
                                                                                'dt': dt,
                                                                                'input_type': 'm_y',
                                                                                'write_controller_log': True,
                                                                                'controlled_surfaces': [0],
                                                                                'time_history_input_file': route + '/elev.csv',
                                                                                'use_initial_value': True,
                                                                                'initial_value': self.settings['initial_deflection']
                                                                                }
                                                                                ,
                                                                    'controller_alpha': {'P': gains_alpha[0],
                                                                                'I': gains_alpha[1],
                                                                                'D': gains_alpha[2],
                                                                                'dt': dt,
                                                                                'input_type': 'f_z',
                                                                                'write_controller_log': True,
                                                                                'time_history_input_file': route + '/alpha.csv',
                                                                                'use_initial_value': True,
                                                                                'initial_value': self.settings['initial_alpha']}
                                                                                ,
                                                                    'controller_thrust': {'thrust_nodes': self.settings['thrust_nodes'],
                                                                                'P': gains_thrust[0],
                                                                                'I': gains_thrust[1],
                                                                                'D': gains_thrust[2],
                                                                                'dt': dt,
                                                                                'input_type': 'f_x',
                                                                                'write_controller_log': True,
                                                                                'time_history_input_file': route + '/thrust.csv',
                                                                                'use_initial_value': True,
                                                                                'initial_value': self.settings['initial_thrust']}
                                                                                }
            # import pdb
            # pdb.set_trace()
            # #end

            self.solver.initialise(self.data, self.settings['solver_settings'], restart=restart)

            self.folder = data.output_folder + '/statictrim/'
            if not os.path.exists(self.folder):
                os.makedirs(self.folder)

            self.table = cout.TablePrinter(10, 8, ['g', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f'],
                                        filename=self.folder+'trim_iterations.txt')
            self.table.print_header(['iter', 'alpha[deg]', 'elev[deg]', 'thrust', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'])
        else:
            raise NotImplementedError('Dynamic trim is only working with nonlinearcoupled or multibody!')  


    def undo_changes(self, data):

        # import pdb
        # pdb.set_trace()
        if self.settings['solver_settings']['structural_solver'] == "NonLinearDynamicMultibody": 
            data.structure.ini_mb_dict['body_00']['FoR_movement'] = 'free'
            data.structure.timestep_info[-1].mb_dict['body_00']['FoR_movement'] = 'free'
            data.structure.ini_info.mb_dict['body_00']['FoR_movement'] = 'free'
            print("HARDCODED!! 16723")

            # data.structure.ini_info.pos_dot *= 0.
            # data.structure.ini_info.pos_ddot *= 0.
            # data.structure.ini_info.psi_dot *= 0.
            # data.structure.ini_info.psi_dot_local *= 0.
            # data.structure.ini_info.psi_ddot *= 0.
            # data.structure.timestep_info[-1].pos_dot *= 0.
            # data.structure.timestep_info[-1].pos_ddot *= 0.
            # data.structure.timestep_info[-1].psi_dot *= 0.
            # data.structure.timestep_info[-1].psi_dot_local *= 0.
            # data.structure.timestep_info[-1].psi_ddot *= 0.
            # data.structure.timestep_info[-1].mb_FoR_vel *= 0.
            # data.structure.timestep_info[-1].mb_FoR_acc *= 0.
            # data.structure.timestep_info[-1].mb_dquatdt *= 0.
            # data.structure.timestep_info[-1].dqddt *= 0.
            # data.structure.timestep_info[-1].forces_constraints_FoR *= 0.
            # data.structure.timestep_info[-1].forces_constraints_nodes *= 0.

            # data.structure.timestep_info[-1].dqdt *= 0.
            # data.structure.timestep_info[-1].q *= 0.
            # data.structure.timestep_info[-1].steady_applied_forces *= 0.
            # data.structure.timestep_info[-1].unsteady_applied_forces *= 0.

            # import pdb
            # pdb.set_trace()
            # data.structure.timestep_info[0].q = np.append(data.structure.timestep_info[0].q, np.zeros(10))
            # data.structure.timestep_info[0].dqdt = np.append(data.structure.timestep_info[0].dqdt, np.zeros(10))
            # data.structure.timestep_info[0].dqddt = np.append(data.structure.timestep_info[0].dqddt, np.zeros(10))
            

    def increase_ts(self):
        self.data.ts += 1
        self.structural_solver.next_step()
        self.aero_solver.next_step()

    def run(self, **kwargs):

        # In the event the modal solver has been run prior to StaticCoupled (i.e. to get undeformed modes), copy
        # results and then attach to the resulting timestep
        try:
            modal = self.data.structure.timestep_info[-1].modal.copy()
            modal_exists = True
        except AttributeError:
            modal_exists = False

        self.trim_algorithm()

        if modal_exists:
            self.data.structure.timestep_info[-1].modal = modal

        if self.settings['save_info']:
            np.savetxt(self.folder + '/trim_values.txt', self.trimmed_values)

        # save trimmed values for dynamic coupled multibody access if needed
        self.data.trimmed_values = self.trimmed_values
        self.undo_changes(self.data)

        return self.data

    def convergence(self, fz, m, fx, thrust):
        return_value = np.array([False, False, False])

        if np.abs(fz) < self.settings['fz_tolerance']:
            return_value[0] = True

        if np.abs(m) < self.settings['m_tolerance']:
            return_value[1] = True

        # print(fx)
        # print(thrust)    
        if np.abs(fx) < self.settings['fx_tolerance']:
            return_value[2] = True

        return return_value

    def trim_algorithm(self):
        """
        Trim algorithm method

        The trim condition is found iteratively.

        Returns:
            np.array: array of trim values for angle of attack, control surface deflection and thrust.
        """
        for self.i_iter in range(self.settings['max_iter'] + 1):
            if self.i_iter == self.settings['max_iter']:
                raise Exception('The Trim routine reached max iterations without convergence!')

            self.input_history.append([])
            self.output_history.append([])
            # self.gradient_history.append([])
            for i in range(self.n_input):
                self.input_history[self.i_iter].append(0)
                self.output_history[self.i_iter].append(0)
                # self.gradient_history[self.i_iter].append(0)

            # the first iteration requires computing gradients
            if not self.i_iter:
                # add to input history the initial estimation
                self.input_history[self.i_iter][0] = self.settings['initial_alpha']
                self.input_history[self.i_iter][1] = (self.settings['initial_deflection'] +
                                                      self.settings['initial_alpha'])
                self.input_history[self.i_iter][2] = self.settings['initial_thrust']

                # compute output
                (self.output_history[self.i_iter][0],
                 self.output_history[self.i_iter][1],
                 self.output_history[self.i_iter][2]) = self.evaluate(self.input_history[self.i_iter][0],
                                                                      self.input_history[self.i_iter][1],
                                                                      self.input_history[self.i_iter][2])
                
                # # do not check for convergence in first step to let transient take effect!
                # # check for convergence (in case initial values are ok)
                # if all(self.convergence(self.output_history[self.i_iter][0],
                #                         self.output_history[self.i_iter][1],
                #                         self.output_history[self.i_iter][2],
                #                         self.input_history[self.i_iter][2])):
                #     self.trimmed_values = self.input_history[self.i_iter]
                #     return

                # # compute gradients
                # # dfz/dalpha
                # (l, m, d) = self.evaluate(self.input_history[self.i_iter][0] + self.settings['initial_angle_eps'],
                #                           self.input_history[self.i_iter][1],
                #                           self.input_history[self.i_iter][2])

                # self.gradient_history[self.i_iter][0] = ((l - self.output_history[self.i_iter][0]) /
                #                                          self.settings['initial_angle_eps'])

                # # dm/dgamma
                # (l, m, d) = self.evaluate(self.input_history[self.i_iter][0],
                #                           self.input_history[self.i_iter][1] + self.settings['initial_angle_eps'],
                #                           self.input_history[self.i_iter][2])

                # self.gradient_history[self.i_iter][1] = ((m - self.output_history[self.i_iter][1]) /
                #                                          self.settings['initial_angle_eps'])

                # # dfx/dthrust
                # (l, m, d) = self.evaluate(self.input_history[self.i_iter][0],
                #                           self.input_history[self.i_iter][1],
                #                           self.input_history[self.i_iter][2] +
                #                           self.settings['initial_thrust_eps'])

                # self.gradient_history[self.i_iter][2] = ((d - self.output_history[self.i_iter][2]) /
                #                                          self.settings['initial_thrust_eps'])

                continue

            # if not all(np.isfinite(self.gradient_history[self.i_iter - 1]))
            # now back to normal evaluation (not only the i_iter == 0 case)
            # compute next alpha with the previous gradient
            # convergence = self.convergence(self.output_history[self.i_iter - 1][0],
            #                                self.output_history[self.i_iter - 1][1],
            #                                self.output_history[self.i_iter - 1][2])
            convergence = np.full((3, ), False)
            if convergence[0]:
                # fz is converged, don't change it
                self.input_history[self.i_iter][0] = self.input_history[self.i_iter - 1][0]
                # self.gradient_history[self.i_iter][0] = self.gradient_history[self.i_iter - 1][0]
            else:
                self.input_history[self.i_iter][0] = self.input_history[self.i_iter - 1][0]

            if convergence[1]:
                # m is converged, don't change it
                self.input_history[self.i_iter][1] = self.input_history[self.i_iter - 1][1]
                # self.gradient_history[self.i_iter][1] = self.gradient_history[self.i_iter - 1][1]
            else:
                # compute next gamma with the previous gradient
                self.input_history[self.i_iter][1] = self.input_history[self.i_iter - 1][1]

            if convergence[2]:
                # fx is converged, don't change it
                self.input_history[self.i_iter][2] = self.input_history[self.i_iter - 1][2]
                # self.gradient_history[self.i_iter][2] = self.gradient_history[self.i_iter - 1][2]
            else:
                # compute next gamma with the previous gradient
                self.input_history[self.i_iter][2] = self.input_history[self.i_iter - 1][2]    
            # else:
            #     if convergence[0] and convergence[1]:
                
            #         # compute next gamma with the previous gradient
            #         self.input_history[self.i_iter][2] = self.balance_thrust(self.output_history[self.i_iter - 1][2])
            #     else:
            #         self.input_history[self.i_iter][2] = self.input_history[self.i_iter - 1][2]

            if self.settings['relaxation_factor']:
                for i_dim in range(3):
                    self.input_history[self.i_iter][i_dim] = (self.input_history[self.i_iter][i_dim]*(1 - self.settings['relaxation_factor']) +
                                                              self.input_history[self.i_iter][i_dim]*self.settings['relaxation_factor'])

            # evaluate
            (self.output_history[self.i_iter][0],
             self.output_history[self.i_iter][1],
             self.output_history[self.i_iter][2]) = self.evaluate(self.input_history[self.i_iter][0],
                                                                  self.input_history[self.i_iter][1],
                                                                  self.input_history[self.i_iter][2])

            if not convergence[0]:
                # self.gradient_history[self.i_iter][0] = ((self.output_history[self.i_iter][0] -
                #                                           self.output_history[self.i_iter - 1][0]) /
                #                                          (self.input_history[self.i_iter][0] -
                #                                           self.input_history[self.i_iter - 1][0]))
                pass

            if not convergence[1]:
                # self.gradient_history[self.i_iter][1] = ((self.output_history[self.i_iter][1] -
                #                                           self.output_history[self.i_iter - 1][1]) /
                #                                          (self.input_history[self.i_iter][1] -
                #                                           self.input_history[self.i_iter - 1][1]))
                pass

            if not convergence[2]:
                # self.gradient_history[self.i_iter][2] = ((self.output_history[self.i_iter][2] -
                #                                           self.output_history[self.i_iter - 1][2]) /
                #                                          (self.input_history[self.i_iter][2] -
                #                                           self.input_history[self.i_iter - 1][2]))
                pass

            # check convergence
            convergence = self.convergence(self.output_history[self.i_iter][0],
                                           self.output_history[self.i_iter][1],
                                           self.output_history[self.i_iter][2],
                                           self.input_history[self.i_iter][2])
            # print(convergence)
            if all(convergence) or ((self.settings['notrim_relax']) or (self.i_iter > self.settings['notrim_relax_iter'])):
                self.trimmed_values = self.input_history[self.i_iter]
                self.table.close_file()
                return

    # def balance_thrust(self, drag):
    #     thrust_nodes = self.settings['thrust_nodes']
    #     thrust_nodes_num = len(thrust_nodes)
    #     orientation = self.solver.get_direction(thrust_nodes)
    #     thrust = -drag/np.sum(orientation, axis=0)[0]
    #     return thrust


    def evaluate(self, alpha, deflection_gamma, thrust):
        if not np.isfinite(alpha):
            import pdb; pdb.set_trace()
        if not np.isfinite(deflection_gamma):
            import pdb; pdb.set_trace()
        if not np.isfinite(thrust):
            import pdb; pdb.set_trace()

        # cout.cout_wrap('--', 2)
        # cout.cout_wrap('Trying trim: ', 2)
        # cout.cout_wrap('Alpha: ' + str(alpha*180/np.pi), 2)
        # cout.cout_wrap('CS deflection: ' + str((deflection_gamma - alpha)*180/np.pi), 2)
        # cout.cout_wrap('Thrust: ' + str(thrust), 2)
        # modify the trim in the static_coupled solver
        # self.solver.change_trim(thrust,
        #                         self.settings['thrust_nodes'],
        #                         deflection_gamma - alpha,
        #                         self.settings['tail_cs_index'])
        # run the solver
        # import pdb
        # pdb.set_trace()
        control = self.solver.time_step()
        # extract resultants
        forces, moments = self.solver.extract_resultants()
        # extract controller inputs
        # control = self.solver.extract_controlcommand()
        alpha = control[1]
        self.input_history[self.i_iter][0] = alpha

        deflection_gamma = control[0]
        self.input_history[self.i_iter][1] = deflection_gamma

        thrust = control[2]
        self.input_history[self.i_iter][2] = thrust

        # deflection_gamma = control[0]
        # thrust = control[1]

        forcez = forces[2]
        forcex = forces[0]
        moment = moments[1]
        # cout.cout_wrap('Forces and moments:', 2)
        # cout.cout_wrap('fx = ' + str(forces[0]) + ' mx = ' + str(moments[0]), 2)
        # cout.cout_wrap('fy = ' + str(forces[1]) + ' my = ' + str(moments[1]), 2)
        # cout.cout_wrap('fz = ' + str(forces[2]) + ' mz = ' + str(moments[2]), 2)

        self.table.print_line([self.i_iter,
                               alpha*180/np.pi,
                               (deflection_gamma)*180/np.pi,
                               thrust,
                               forces[0],
                               forces[1],
                               forces[2],
                               moments[0],
                               moments[1],
                               moments[2]])

        return forcez, moment, forcex
