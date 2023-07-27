import numpy as np
import scipy.optimize

import sharpy.utils.cout_utils as cout
import sharpy.utils.solver_interface as solver_interface
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings_utils
import sharpy.utils.algebra as algebra


@solver
class Trim(BaseSolver):
    """
    Trim routine with support for lateral dynamics. It usually struggles much more
    than the ``StaticTrim`` (only longitudinal) solver.

    We advise to start with ``StaticTrim`` even if you configuration is not totally symmetric.
    """
    solver_id = 'Trim'
    solver_classification = 'Flight dynamics'

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
    settings_default['max_iter'] = 100
    settings_description['max_iter'] = 'Maximum number of iterations of trim routine'

    settings_types['tolerance'] = 'float'
    settings_default['tolerance'] = 1e-4
    settings_description['tolerance'] = 'Threshold for convergence of trim'

    settings_types['initial_alpha'] = 'float'
    settings_default['initial_alpha'] = 0.
    settings_description['initial_alpha'] = 'Initial angle of attack'

    settings_types['initial_beta'] = 'float'
    settings_default['initial_beta'] = 0.
    settings_description['initial_beta'] = 'Initial sideslip angle'

    settings_types['initial_roll'] = 'float'
    settings_default['initial_roll'] = 0
    settings_description['initial_roll'] = 'Initial roll angle'

    settings_types['cs_indices'] = 'list(int)'
    settings_default['cs_indices'] = []
    settings_description['cs_indices'] = 'Indices of control surfaces to be trimmed'

    settings_types['initial_cs_deflection'] = 'list(float)'
    settings_default['initial_cs_deflection'] = []
    settings_description['initial_cs_deflection'] = 'Initial deflection of the control surfaces in order.'

    settings_types['thrust_nodes'] = 'list(int)'
    settings_default['thrust_nodes'] = [0]
    settings_description['thrust_nodes'] = 'Nodes at which thrust is applied'

    settings_types['initial_thrust'] = 'list(float)'
    settings_default['initial_thrust'] = [1.]
    settings_description['initial_thrust'] = 'Initial thrust setting'

    settings_types['thrust_direction'] = 'list(float)'
    settings_default['thrust_direction'] = [.0, 1.0, 0.0]
    settings_description['thrust_direction'] = 'Thrust direction setting'

    settings_types['special_case'] = 'dict'
    settings_default['special_case'] = dict()
    settings_description['special_case'] = 'Extra settings for specific cases such as differential thrust control'

    settings_types['refine_solution'] = 'bool'
    settings_default['refine_solution'] = False
    settings_description['refine_solution'] = 'If ``True`` and the optimiser routine allows for it, the optimiser will try to improve the solution with hybrid methods'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.data = None
        self.settings = None
        self.solver = None

        self.x_info = dict()
        self.initial_state = None

        self.with_special_case = False

    def initialise(self, data, restart=False):
        self.data = data
        self.settings = data.settings[self.solver_id]
        settings_utils.to_custom_types(self.settings, self.settings_types, self.settings_default)

        self.solver = solver_interface.initialise_solver(self.settings['solver'])
        self.solver.initialise(self.data, self.settings['solver_settings'], restart=restart)

        # generate x_info (which elements of the x array are what)
        counter = 0
        self.x_info['n_variables'] = 0
        # alpha
        self.x_info['i_alpha'] = counter
        counter += 1
        # beta
        self.x_info['i_beta'] = counter
        counter += 1
        # roll
        self.x_info['i_roll'] = counter
        counter += 1
        # control surfaces
        n_control_surfaces = len(self.settings['cs_indices'])
        self.x_info['i_control_surfaces'] = []  # indices in the state vector
        self.x_info['control_surfaces_id'] = []  # indices of the trimmed control surfaces
        for i_cs in range(n_control_surfaces):
            self.x_info['i_control_surfaces'].append(counter)
            self.x_info['control_surfaces_id'].append(self.settings['cs_indices'][i_cs])
            counter += 1
        # thrust
        n_thrust_nodes = len(self.settings['thrust_nodes'])
        self.x_info['i_thrust'] = []
        self.x_info['thrust_nodes'] = []
        self.x_info['thrust_direction'] = []
        for i_thrust in range(n_thrust_nodes):
            self.x_info['i_thrust'].append(counter)
            self.x_info['thrust_nodes'].append(self.settings['thrust_nodes'][i_thrust])
            self.x_info['thrust_direction'].append(self.settings['thrust_direction'])
            counter += 1
        self.x_info['n_variables'] = counter

        # special cases
        self.with_special_case = self.settings['special_case']
        if self.with_special_case:
            if self.settings['special_case']['case_name'] == 'differential_thrust':
                self.x_info['special_case'] = 'differential_thrust'
                self.x_info['i_base_thrust'] = counter
                counter += 1
                self.x_info['i_differential_parameter'] = counter
                counter += 1
                self.x_info['initial_base_thrust'] = self.settings['special_case']['initial_base_thrust']
                self.x_info['initial_differential_parameter'] = self.settings['special_case']['initial_differential_parameter']
                self.x_info['base_thrust_nodes'] = [int(e) for e in self.settings['special_case']['base_thrust_nodes']]
                self.x_info['negative_thrust_nodes'] = [int(e) for e in self.settings['special_case']['negative_thrust_nodes']]
                self.x_info['positive_thrust_nodes'] = [int(e) for e in self.settings['special_case']['positive_thrust_nodes']]

            self.x_info['n_variables'] = counter


        # initial state vector
        self.initial_state = np.zeros(self.x_info['n_variables'])
        self.initial_state[self.x_info['i_alpha']] = self.settings['initial_alpha']
        self.initial_state[self.x_info['i_beta']] = self.settings['initial_beta']
        self.initial_state[self.x_info['i_roll']] = self.settings['initial_roll']
        for i_cs in range(n_control_surfaces):
            self.initial_state[self.x_info['i_control_surfaces'][i_cs]] = self.settings['initial_cs_deflection'][i_cs]
        for i_thrust in range(n_thrust_nodes):
            self.initial_state[self.x_info['i_thrust'][i_thrust]] = self.settings['initial_thrust'][i_thrust]
        if self.with_special_case:
            if self.settings['special_case']['case_name'] == 'differential_thrust':
                self.initial_state[self.x_info['i_base_thrust']] = self.x_info['initial_base_thrust']
                self.initial_state[self.x_info['i_differential_parameter']] = self.x_info['initial_differential_parameter']


        # bounds
        # NOTE probably not necessary anymore, as Nelder-Mead method doesn't use them
        self.bounds = self.x_info['n_variables']*[None]
        for k, v in self.x_info.items():
            if k == 'i_alpha':
                self.bounds[v] = (self.initial_state[self.x_info['i_alpha']] - 3*np.pi/180,
                                  self.initial_state[self.x_info['i_alpha']] + 3*np.pi/180)
            elif k == 'i_beta':
                self.bounds[v] = (self.initial_state[self.x_info['i_beta']] - 2*np.pi/180,
                                  self.initial_state[self.x_info['i_beta']] + 2*np.pi/180)
            elif k == 'i_roll':
                self.bounds[v] = (self.initial_state[self.x_info['i_roll']] - 2*np.pi/180,
                                  self.initial_state[self.x_info['i_roll']] + 2*np.pi/180)
            elif k == 'i_thrust':
                for ii, i in enumerate(v):
                    self.bounds[i] = (self.initial_state[self.x_info['i_thrust'][ii]] - 2,
                                      self.initial_state[self.x_info['i_thrust'][ii]] + 2)
            elif k == 'i_control_surfaces':
                for ii, i in enumerate(v):
                    self.bounds[i] = (self.initial_state[self.x_info['i_control_surfaces'][ii]] - 4*np.pi/180,
                                      self.initial_state[self.x_info['i_control_surfaces'][ii]] + 4*np.pi/180)
            elif k == 'i_base_thrust':
                if self.with_special_case:
                    if self.settings['special_case']['case_name'] == 'differential_thrust':
                        self.bounds[v] = (float(self.x_info['initial_base_thrust'])*0.5,
                                          float(self.x_info['initial_base_thrust'])*1.5)
            elif k == 'i_differential_parameter':
                if self.with_special_case:
                    if self.settings['special_case']['case_name'] == 'differential_thrust':
                        self.bounds[v] = (-0.5, 0.5)

    def increase_ts(self):
        self.data.ts += 1
        self.structural_solver.next_step()
        self.aero_solver.next_step()

    def cleanup_timestep_info(self):
        if max(len(self.data.aero.timestep_info), len(self.data.structure.timestep_info)) > 1:
            # copy last info to first
            self.data.aero.timestep_info[0] = self.data.aero.timestep_info[-1]
            self.data.structure.timestep_info[0] = self.data.structure.timestep_info[-1]
            # delete all the rest
            while len(self.data.aero.timestep_info) - 1:
                del self.data.aero.timestep_info[-1]
            while len(self.data.structure.timestep_info) - 1:
                del self.data.structure.timestep_info[-1]

        self.data.ts = 0

    def run(self):
        self.trim_algorithm()
        return self.data

    def trim_algorithm(self):
        # create bounds

        # call optimiser
        self.optimise(solver_wrapper,
                      tolerance=self.settings['tolerance'],
                      print_info=True,
                      # method='BFGS')
                      method='Nelder-Mead',
                      # method='SLSQP',
                      refine=self.settings['refine_solution'])

        # self.optimise(self.solver_wrapper, )
        pass

    def optimise(self, func, tolerance, print_info, method, refine):
        args = (self.x_info, self, -2)

        solution = scipy.optimize.minimize(func,
                                           self.initial_state,
                                           args=args,
                                           method=method,
                                           options={'disp': print_info,
                                                    'maxfev': 1000,
                                                    'xatol': tolerance,
                                                    'fatol': 1e-4})
        if refine:
            cout.cout_wrap('Refining results with a gradient-based method', 1)
            solution = scipy.optimize.minimize(func,
                                               solution.x,
                                               args=args,
                                               method='BFGS',
                                               options={'disp': print_info,
                                                        'eps': 0.05,
                                                        'maxfev': 5000,
                                                        'fatol': 1e-4})

        cout.cout_wrap('Solution = ')
        cout.cout_wrap(solution.x)
        # pretty_print_x(x, x_info)
        return solution


# def pretty_print_x(x, x_info):
#     cout.cout_wrap('X vector:', 1)
#     for k, v in x_info:
#         if k.startswith('i_'):
#             if isinstance(v, list):
#                 for i, vv in v:
#                     cout.cout_wrap(k + ' ' + str(i) + ': ', vv)
#             else:
#                 cout.cout_wrap(k + ': ', v)


def solver_wrapper(x, x_info, solver_data, i_dim=-1):
    if solver_data.settings['print_info']:
        cout.cout_wrap('x = ' + str(x), 1)
    # print('x = ', x)
    alpha = x[x_info['i_alpha']]
    beta = x[x_info['i_beta']]
    roll = x[x_info['i_roll']]
    # change input data
    solver_data.data.structure.timestep_info[solver_data.data.ts] = solver_data.data.structure.ini_info.copy()
    tstep = solver_data.data.structure.timestep_info[solver_data.data.ts]
    aero_tstep = solver_data.data.aero.timestep_info[solver_data.data.ts]
    orientation_quat = algebra.euler2quat(np.array([roll, alpha, beta]))
    tstep.quat[:] = orientation_quat
    # control surface deflection
    for i_cs in range(len(x_info['i_control_surfaces'])):
        solver_data.data.aero.data_dict['control_surface_deflection'][x_info['control_surfaces_id'][i_cs]] = x[x_info['i_control_surfaces'][i_cs]]
    # thrust input
    tstep.steady_applied_forces[:] = 0.0
    try:
        x_info['special_case']
    except KeyError:
        for i_thrust in range(len(x_info['i_thrust'])):
            thrust = x[x_info['i_thrust'][i_thrust]]
            i_node = x_info['thrust_nodes'][i_thrust]
            solver_data.data.structure.ini_info.steady_applied_forces[i_node, 0:3] = thrust*x_info['thrust_direction'][i_thrust]
    else:
        if x_info['special_case'] == 'differential_thrust':
            base_thrust = x[x_info['i_base_thrust']]
            pos_thrust = base_thrust*(1.0 + x[x_info['i_differential_parameter']])
            neg_thrust = -base_thrust*(1.0 - x[x_info['i_differential_parameter']])
            for i_base_node in x_info['base_thrust_nodes']:
                solver_data.data.structure.ini_info.steady_applied_forces[i_base_node, 1] = base_thrust
            for i_pos_diff_node in x_info['positive_thrust_nodes']:
                solver_data.data.structure.ini_info.steady_applied_forces[i_pos_diff_node, 1] = pos_thrust
            for i_neg_diff_node in x_info['negative_thrust_nodes']:
                solver_data.data.structure.ini_info.steady_applied_forces[i_neg_diff_node, 1] = neg_thrust

    # run the solver
    solver_data.solver.run()
    # extract resultants
    forces, moments = solver_data.solver.extract_resultants()

    totals = np.zeros((6,))
    totals[0:3] = forces
    totals[3:6] = moments
    if solver_data.settings['print_info']:
        cout.cout_wrap(' forces = ' + str(totals), 1)
    # print('total forces = ', totals)
    # try:
    #     totals += x[x_info['i_none']]
    # except KeyError:
    #     pass
    # return resultant forces and moments
    # return np.linalg.norm(totals)
    if i_dim >= 0:
        return totals[i_dim]
    elif i_dim == -1:
        # return [np.sum(totals[0:3]**2), np.sum(totals[4:6]**2)]
        return totals
    elif i_dim == -2:
        coeffs = np.array([1.0, 1.0, 1.0, 2, 2, 2])
        # print('return = ', np.dot(coeffs*totals, coeffs*totals))
        if solver_data.settings['print_info']:
            cout.cout_wrap(' val = ' + str(np.dot(coeffs*totals, coeffs*totals)), 1)
        return np.dot(coeffs*totals, coeffs*totals)
