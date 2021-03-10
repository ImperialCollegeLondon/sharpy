import ctypes as ct
import numpy as np
import scipy.optimize
import warnings

import sharpy.aero.utils.mapping as mapping
import sharpy.utils.cout_utils as cout
import sharpy.utils.solver_interface as solver_interface
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra


@solver
class StaticEquilibriumWT(BaseSolver):
    """
    Compute equilibrium equilibrium of a multibody wind turbine
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

    settings_types['initial_pos'] = 'list(float)'
    settings_default['initial_pos'] = [0., 0., 0.]
    settings_description['initial_pos'] = 'xyz initial location of the frame of reference'

    settings_types['initial_orientation'] = 'list(float)'
    settings_default['initial_orientation'] = [0., 0., 0.]
    settings_description['initial_orientation'] = 'xyz initial rotations of the frame of reference'

    settings_types['refine_solution'] = 'bool'
    settings_default['refine_solution'] = False
    settings_description['refine_solution'] = 'If ``True`` and the optimiser routine allows for it, the optimiser will try to improve the solution with hybrid methods'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.data = None
        self.settings = None
        self.solver = None

        self.x_info = dict()
        self.initial_state = None

        self.with_special_case = False

    def initialise(self, data):
        self.data = data
        self.settings = data.settings[self.solver_id]
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        self.solver = solver_interface.initialise_solver(self.settings['solver'])
        self.solver.initialise(self.data, self.settings['solver_settings'])

        # generate x_info (which elements of the x array are what)
        counter = 0
        self.x_info['i_pos_x'] = counter
        counter += 1
        self.x_info['i_pos_y'] = counter
        counter += 1
        self.x_info['i_pos_z'] = counter
        counter += 1

        self.x_info['i_euler_x'] = counter
        counter += 1
        self.x_info['i_euler_y'] = counter
        counter += 1
        self.x_info['i_euler_z'] = counter
        counter += 1

        self.x_info['n_variables'] = counter

        # initial state vector
        self.initial_state = np.zeros(self.x_info['n_variables'])
        self.initial_state[self.x_info['i_pos_x']] = self.settings['initial_pos'][0]
        self.initial_state[self.x_info['i_pos_y']] = self.settings['initial_pos'][1]
        self.initial_state[self.x_info['i_pos_z']] = self.settings['initial_pos'][2]

        self.initial_state[self.x_info['i_euler_x']] = self.settings['initial_orientation'][0]
        self.initial_state[self.x_info['i_euler_y']] = self.settings['initial_orientation'][1]
        self.initial_state[self.x_info['i_euler_z']] = self.settings['initial_orientation'][2]

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
        # call optimiser
        self.optimise(solver_wrapper,
                      tolerance=self.settings['tolerance'].value,
                      print_info=True,
                      method='Nelder-Mead',
                      refine=self.settings['refine_solution'])
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
        return solution


def solver_wrapper(x, x_info, solver_data, i_dim=-1):
    if solver_data.settings['print_info']:
        cout.cout_wrap('x = ' + str(x), 1)

    for_pos = np.array([x[x_info['i_pos_x']],
                        x[x_info['i_pos_y']],
                        x[x_info['i_pos_z']],])
    euler = np.array([x[x_info['i_euler_x']],
                      x[x_info['i_euler_y']],
                      x[x_info['i_euler_z']],])
    # change input data
    solver_data.data.structure.timestep_info[solver_data.data.ts] = solver_data.data.structure.ini_info.copy()
    tstep = solver_data.data.structure.timestep_info[solver_data.data.ts]
    aero_tstep = solver_data.data.aero.timestep_info[solver_data.data.ts]
    tstep.for_pos[0:3] = for_pos.copy()
    tstep.quat[:] = algebra.euler2quat(euler).copy()

    # Update multibody information
    tstep.mb_FoR_pos[0, :] = tstep.for_pos.copy()
    tstep.mb_quat[0, :] = tstep.quat.copy()

    for ibody in range(1, solver_data.data.structure.num_bodies):
        tstep.mb_FoR_pos[ibody, :] += tstep.for_pos
        tstep.mb_quat[ibody, :] = algebra.quaternion_product(tstep.mb_quat[ibody, :],
                                                             tstep.mb_quat[0, :])

    # run the solver
    solver_data.solver.run()
    # extract resultants
    forces, moments = solver_data.solver.extract_resultants()

    # Measure of the error for the optimisation
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
        # coeffs = np.array([1.0, 1.0, 1.0, 2, 2, 2])
        coeffs = np.ones((6))
        # print('return = ', np.dot(coeffs*totals, coeffs*totals))
        if solver_data.settings['print_info']:
            cout.cout_wrap(' val = ' + str(np.dot(coeffs*totals, coeffs*totals)), 1)
        return np.dot(coeffs*totals, coeffs*totals)
