import numpy as np

import sharpy.utils.cout_utils as cout
import sharpy.utils.solver_interface as solver_interface
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings_utils
import os


@solver
class StaticTrim(BaseSolver):
    """
    The ``StaticTrim`` solver determines the longitudinal state of trim (equilibrium) for an aeroelastic system in
    static conditions. It wraps around the desired solver to yield the state of trim of the system, in most cases
    the :class:`~sharpy.solvers.staticcoupled.StaticCoupled` solver.

    It calculates the required angle of attack, elevator deflection and thrust required to achieve longitudinal
    equilibrium. The output angles are shown in degrees.

    The results from the trimming iteration can be saved to a text file by using the `save_info` option.
    """
    solver_id = 'StaticTrim'
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
    settings_default['max_iter'] = 100
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
        self.solver.initialise(self.data, self.settings['solver_settings'], restart=restart)

        self.folder = data.output_folder + '/statictrim/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.table = cout.TablePrinter(10, 8, ['g', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f'],
                                       filename=self.folder+'trim_iterations.txt')
        self.table.print_header(['iter', 'alpha[deg]', 'elev[deg]', 'thrust', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'])

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

        return self.data

    def convergence(self, fz, m, fx):
        return_value = np.array([False, False, False])

        if np.abs(fz) < self.settings['fz_tolerance']:
            return_value[0] = True

        if np.abs(m) < self.settings['m_tolerance']:
            return_value[1] = True

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
            self.gradient_history.append([])
            for i in range(self.n_input):
                self.input_history[self.i_iter].append(0)
                self.output_history[self.i_iter].append(0)
                self.gradient_history[self.i_iter].append(0)

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

                # check for convergence (in case initial values are ok)
                if all(self.convergence(self.output_history[self.i_iter][0],
                                        self.output_history[self.i_iter][1],
                                        self.output_history[self.i_iter][2])):
                    self.trimmed_values = self.input_history[self.i_iter]
                    return

                # compute gradients
                # dfz/dalpha
                (l, m, d) = self.evaluate(self.input_history[self.i_iter][0] + self.settings['initial_angle_eps'],
                                          self.input_history[self.i_iter][1],
                                          self.input_history[self.i_iter][2])

                self.gradient_history[self.i_iter][0] = ((l - self.output_history[self.i_iter][0]) /
                                                         self.settings['initial_angle_eps'])

                # dm/dgamma
                (l, m, d) = self.evaluate(self.input_history[self.i_iter][0],
                                          self.input_history[self.i_iter][1] + self.settings['initial_angle_eps'],
                                          self.input_history[self.i_iter][2])

                self.gradient_history[self.i_iter][1] = ((m - self.output_history[self.i_iter][1]) /
                                                         self.settings['initial_angle_eps'])

                # dfx/dthrust
                (l, m, d) = self.evaluate(self.input_history[self.i_iter][0],
                                          self.input_history[self.i_iter][1],
                                          self.input_history[self.i_iter][2] +
                                          self.settings['initial_thrust_eps'])

                self.gradient_history[self.i_iter][2] = ((d - self.output_history[self.i_iter][2]) /
                                                         self.settings['initial_thrust_eps'])

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
                self.gradient_history[self.i_iter][0] = self.gradient_history[self.i_iter - 1][0]
            else:
                self.input_history[self.i_iter][0] = (self.input_history[self.i_iter - 1][0] -
                                                      (self.output_history[self.i_iter - 1][0] /
                                                       self.gradient_history[self.i_iter - 1][0]))

            if convergence[1]:
                # m is converged, don't change it
                self.input_history[self.i_iter][1] = self.input_history[self.i_iter - 1][1]
                self.gradient_history[self.i_iter][1] = self.gradient_history[self.i_iter - 1][1]
            else:
                # compute next gamma with the previous gradient
                self.input_history[self.i_iter][1] = (self.input_history[self.i_iter - 1][1] -
                                                      (self.output_history[self.i_iter - 1][1] /
                                                       self.gradient_history[self.i_iter - 1][1]))

            if convergence[2]:
                # fx is converged, don't change it
                self.input_history[self.i_iter][2] = self.input_history[self.i_iter - 1][2]
                self.gradient_history[self.i_iter][2] = self.gradient_history[self.i_iter - 1][2]
            else:
                # compute next gamma with the previous gradient
                self.input_history[self.i_iter][2] = (self.input_history[self.i_iter - 1][2] -
                                                      (self.output_history[self.i_iter - 1][2] /
                                                       self.gradient_history[self.i_iter - 1][2]))

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
                self.gradient_history[self.i_iter][0] = ((self.output_history[self.i_iter][0] -
                                                          self.output_history[self.i_iter - 1][0]) /
                                                         (self.input_history[self.i_iter][0] -
                                                          self.input_history[self.i_iter - 1][0]))

            if not convergence[1]:
                self.gradient_history[self.i_iter][1] = ((self.output_history[self.i_iter][1] -
                                                          self.output_history[self.i_iter - 1][1]) /
                                                         (self.input_history[self.i_iter][1] -
                                                          self.input_history[self.i_iter - 1][1]))

            if not convergence[2]:
                self.gradient_history[self.i_iter][2] = ((self.output_history[self.i_iter][2] -
                                                          self.output_history[self.i_iter - 1][2]) /
                                                         (self.input_history[self.i_iter][2] -
                                                          self.input_history[self.i_iter - 1][2]))

            # check convergence
            convergence = self.convergence(self.output_history[self.i_iter][0],
                                           self.output_history[self.i_iter][1],
                                           self.output_history[self.i_iter][2])
            if all(convergence):
                self.trimmed_values = self.input_history[self.i_iter]
                self.table.close_file()
                return

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
        self.solver.change_trim(alpha,
                                thrust,
                                self.settings['thrust_nodes'],
                                deflection_gamma - alpha,
                                self.settings['tail_cs_index'])
        # run the solver
        self.solver.run()
        # extract resultants
        forces, moments = self.solver.extract_resultants()

        forcez = forces[2]
        forcex = forces[0]
        moment = moments[1]
        # cout.cout_wrap('Forces and moments:', 2)
        # cout.cout_wrap('fx = ' + str(forces[0]) + ' mx = ' + str(moments[0]), 2)
        # cout.cout_wrap('fy = ' + str(forces[1]) + ' my = ' + str(moments[1]), 2)
        # cout.cout_wrap('fz = ' + str(forces[2]) + ' mz = ' + str(moments[2]), 2)

        self.table.print_line([self.i_iter,
                               alpha*180/np.pi,
                               (deflection_gamma - alpha)*180/np.pi,
                               thrust,
                               forces[0],
                               forces[1],
                               forces[2],
                               moments[0],
                               moments[1],
                               moments[2]])

        return forcez, moment, forcex
