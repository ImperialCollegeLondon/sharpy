import ctypes as ct
import numpy as np
import warnings

import sharpy.aero.utils.mapping as mapping
import sharpy.utils.cout_utils as cout
import sharpy.utils.solver_interface as solver_interface
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra



@solver
class StaticTrim(BaseSolver):
    solver_id = 'StaticTrim'

    def __init__(self):
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['print_info'] = 'bool'
        self.settings_default['print_info'] = True

        self.settings_types['solver'] = 'str'
        self.settings_default['solver'] = None

        self.settings_types['solver_settings'] = 'dict'
        self.settings_default['solver_settings'] = None

        self.settings_types['max_iter'] = 'int'
        self.settings_default['max_iter'] = 100

        self.settings_types['fz_tolerance'] = 'float'
        self.settings_default['fz_tolerance'] = 0.1

        self.settings_types['fx_tolerance'] = 'float'
        self.settings_default['fx_tolerance'] = 0.1

        self.settings_types['m_tolerance'] = 'float'
        self.settings_default['m_tolerance'] = 0.1

        self.settings_types['tail_cs_index'] = 'int'
        self.settings_default['tail_cs_index'] = 0

        self.settings_types['thrust_nodes'] = 'list(int)'
        self.settings_default['thrust_nodes'] = np.array([0])

        self.settings_types['initial_alpha'] = 'float'
        self.settings_default['initial_alpha'] = 4*np.pi/180.

        self.settings_types['initial_deflection'] = 'float'
        self.settings_default['initial_deflection'] = 1*np.pi/180.

        self.settings_types['initial_thrust'] = 'float'
        self.settings_default['initial_thrust'] = 0.0

        self.settings_types['initial_angle_eps'] = 'float'
        self.settings_default['initial_angle_eps'] = 0.5*np.pi/180.

        self.settings_types['initial_thrust_eps'] = 'float'
        self.settings_default['initial_thrust_eps'] = 2.

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


    def initialise(self, data):
        self.data = data
        self.settings = data.settings[self.solver_id]
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        self.solver = solver_interface.initialise_solver(self.settings['solver'])
        self.solver.initialise(self.data, self.settings['solver_settings'])

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
        # TODO modify trimmed values for next solver
        return self.data

    def convergence(self, fz, m, fx):
        return_value = np.array([False, False, False])

        if np.abs(fz) < self.settings['fz_tolerance']:
            return_value[0] = True

        if np.abs(m) < self.settings['m_tolerance']:
            return_value[1] = True

        if np.abs(fx) < self.settings['fx_tolerance']:
            return_value[2] = True

        # TODO temp for not adjusting thrust

        # if (np.abs(self.forcex_history[-1]) < 1000000*self.settings['fx_tolerance'].value
        #     and
        #     np.abs(self.forcez_history[-1]) < self.settings['fz_tolerance'].value
        #     and
        #     np.abs(self.moment_history[-1]) < self.settings['m_tolerance'].value):
        #     return_value = True

        return return_value

    def trim_algorithm(self):
        for self.i_iter in range(self.settings['max_iter'].value + 1):
            if self.i_iter == self.settings['max_iter'].value:
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
                self.input_history[self.i_iter][0] = self.settings['initial_alpha'].value
                self.input_history[self.i_iter][1] = (self.settings['initial_deflection'].value +
                                                      self.settings['initial_alpha'].value)
                self.input_history[self.i_iter][2] = self.settings['initial_thrust'].value

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
                (l, m, d) = self.evaluate(self.input_history[self.i_iter][0] + self.settings['initial_angle_eps'].value,
                                          self.input_history[self.i_iter][1],
                                          self.input_history[self.i_iter][2])

                self.gradient_history[self.i_iter][0] = ((l - self.output_history[self.i_iter][0]) /
                                                          self.settings['initial_angle_eps'].value)

                # dm/dgamma
                (l, m, d) = self.evaluate(self.input_history[self.i_iter][0],
                                          self.input_history[self.i_iter][1] + self.settings['initial_angle_eps'].value,
                                          self.input_history[self.i_iter][2])

                self.gradient_history[self.i_iter][1] = ((m - self.output_history[self.i_iter][1]) /
                                                          self.settings['initial_angle_eps'].value)

                # dfx/dthrust
                (l, m, d) = self.evaluate(self.input_history[self.i_iter][0],
                                          self.input_history[self.i_iter][1],
                                          self.input_history[self.i_iter][2] +
                                          self.settings['initial_thrust_eps'].value)

                self.gradient_history[self.i_iter][2] = ((d - self.output_history[self.i_iter][2]) /
                                                         self.settings['initial_thrust_eps'].value)

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
                # self.output_history[self.i_iter][0] = self.output_history[self.i_iter - 1][0]
                # self.output_history[self.i_iter][1] = self.output_history[self.i_iter - 1][1]
                # self.output_history[self.i_iter][2] = self.output_history[self.i_iter - 1][2]
                self.gradient_history[self.i_iter][0] = self.gradient_history[self.i_iter - 1][0]
            else:
                self.input_history[self.i_iter][0] = (self.input_history[self.i_iter - 1][0] -
                                                      (self.output_history[self.i_iter - 1][0] /
                                                       self.gradient_history[self.i_iter - 1][0]))

            if convergence[1]:
                # m is converged, don't change it
                self.input_history[self.i_iter][1] = self.input_history[self.i_iter - 1][1]
                # self.output_history[self.i_iter][0] = self.output_history[self.i_iter - 1][0]
                # self.output_history[self.i_iter][1] = self.output_history[self.i_iter - 1][1]
                # self.output_history[self.i_iter][2] = self.output_history[self.i_iter - 1][2]
                self.gradient_history[self.i_iter][1] = self.gradient_history[self.i_iter - 1][1]
            else:
                # compute next gamma with the previous gradient
                self.input_history[self.i_iter][1] = (self.input_history[self.i_iter - 1][1] -
                                                      (self.output_history[self.i_iter - 1][1] /
                                                       self.gradient_history[self.i_iter - 1][1]))

            if convergence[2]:
                # fx is converged, don't change it
                self.input_history[self.i_iter][2] = self.input_history[self.i_iter - 1][2]
                # self.output_history[self.i_iter][0] = self.output_history[self.i_iter - 1][0]
                # self.output_history[self.i_iter][1] = self.output_history[self.i_iter - 1][1]
                # self.output_history[self.i_iter][2] = self.output_history[self.i_iter - 1][2]
                self.gradient_history[self.i_iter][2] = self.gradient_history[self.i_iter - 1][2]
            else:
                # compute next gamma with the previous gradient
                self.input_history[self.i_iter][2] = (self.input_history[self.i_iter - 1][2] -
                                                      (self.output_history[self.i_iter - 1][2] /
                                                       self.gradient_history[self.i_iter - 1][2]))

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
                return







                # # evaluate
                # (self.output_history[self.i_iter][0],
                #  self.output_history[self.i_iter][1],
                #  self.output_history[self.i_iter][2]) = self.evaluate(self.input_history[self.i_iter][0],
                #                                                       self.input_history[self.i_iter - 1][1],
                #                                                       self.input_history[self.i_iter - 1][2])
                # self.gradient_history[self.i_iter][0] = ((self.output_history[self.i_iter][0] -
                #                                            self.output_history[self.i_iter - 1][0]) /
                #                                           (self.input_history[self.i_iter][0] -
                #                                            self.input_history[self.i_iter - 1][0]))

            # # check convergence
            # convergence = self.convergence(self.output_history[self.i_iter][0],
            #                                self.output_history[self.i_iter][1],
            #                                self.output_history[self.i_iter][2])
            # if all(convergence):
            #     self.trimmed_values = self.input_history[self.i_iter]
            #     return
            #
            # if convergence[1]:
            #     # m is converged, don't change it
            #     self.input_history[self.i_iter][1] = self.input_history[self.i_iter - 1][1]
            #     self.output_history[self.i_iter][0] = self.output_history[self.i_iter - 1][0]
            #     self.output_history[self.i_iter][1] = self.output_history[self.i_iter - 1][1]
            #     self.output_history[self.i_iter][2] = self.output_history[self.i_iter - 1][2]
            #     self.gradient_history[self.i_iter][1] = self.gradient_history[self.i_iter - 1][1]
            # else:
            #     # compute next gamma with the previous gradient
            #     self.input_history[self.i_iter][1] = (self.input_history[self.i_iter - 1][1] -
            #                                           (self.output_history[self.i_iter - 1][1] /
            #                                            self.gradient_history[self.i_iter - 1][1]))
            #     # evaluate
            #     (self.output_history[self.i_iter][0],
            #      self.output_history[self.i_iter][1],
            #      self.output_history[self.i_iter][2]) = self.evaluate(self.input_history[self.i_iter][0],
            #                                                           self.input_history[self.i_iter][1],
            #                                                           self.input_history[self.i_iter - 1][2])
            #     self.gradient_history[self.i_iter][1] = ((self.output_history[self.i_iter][1] -
            #                                                self.output_history[self.i_iter - 1][1]) /
            #                                               (self.input_history[self.i_iter][1] -
            #                                                self.input_history[self.i_iter - 1][1]))

            # check convergence
            # convergence = self.convergence(self.output_history[self.i_iter][0],
            #                                self.output_history[self.i_iter][1],
            #                                self.output_history[self.i_iter][2])
            # if all(convergence):
            #     self.trimmed_values = self.input_history[self.i_iter]
            #     return
            #
            # if convergence[2]:
            #     # fx is converged, don't change it
            #     self.input_history[self.i_iter][2] = self.input_history[self.i_iter - 1][2]
            #     self.output_history[self.i_iter][0] = self.output_history[self.i_iter - 1][0]
            #     self.output_history[self.i_iter][1] = self.output_history[self.i_iter - 1][1]
            #     self.output_history[self.i_iter][2] = self.output_history[self.i_iter - 1][2]
            #     self.gradient_history[self.i_iter][2] = self.gradient_history[self.i_iter - 1][2]
            # else:
            #     # compute next gamma with the previous gradient
            #     self.input_history[self.i_iter][2] = (self.input_history[self.i_iter - 1][2] -
            #                                           (self.output_history[self.i_iter - 1][2] /
            #                                            self.gradient_history[self.i_iter - 1][2]))
            #     # evaluate
            #     (self.output_history[self.i_iter][0],
            #      self.output_history[self.i_iter][1],
            #      self.output_history[self.i_iter][2]) = self.evaluate(self.input_history[self.i_iter][0],
            #                                                           self.input_history[self.i_iter][1],
            #                                                           self.input_history[self.i_iter][2])
            #     self.gradient_history[self.i_iter][2] = ((self.output_history[self.i_iter][2] -
            #                                                self.output_history[self.i_iter - 1][2]) /
            #                                               (self.input_history[self.i_iter][2] -
            #                                                self.input_history[self.i_iter - 1][2]))




    def evaluate(self, alpha, deflection_gamma, thrust):
        if not np.isfinite(alpha):
            1
        if not np.isfinite(deflection_gamma):
            1
        if not np.isfinite(thrust):
            1

        print('--')
        print(alpha*180/np.pi, (deflection_gamma - alpha)*180/np.pi, thrust)
        # modify the trim in the static_coupled solver
        self.solver.change_trim(alpha,
                                thrust,
                                self.settings['thrust_nodes'],
                                deflection_gamma - alpha,
                                self.settings['tail_cs_index'].value)
        # run the solver
        self.solver.run()
        # extract resultants
        forces, moments = self.solver.extract_resultants()

        forcez = forces[2]
        forcex = forces[0]
        moment = moments[1]
        print(forcez, moment, forcex)
        print(forces, moments)

        return forcez, moment, forcex


























