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
    """
    ``StaticTrim`` class solver, inherited from ``BaseSolver``

    The ``StaticTrim`` solver determines the state of trim (equilibrium) for an aeroelastic system in static conditions.
    It wraps around the desired solver to yield the state of trim of the system.

    Args:
        data (PreSharpy): object with problem data

    Attributes:
        settings (dict): Name-value pair of settings employed by solver.

            ======================  =============  ===============================================  ==========
            Name                    Type           Description                                      Default
            ======================  =============  ===============================================  ==========
            ``print_info``          ``bool``       Print solver information to terminal             ``True``
            ``solver``              ``str``        Underlying solver for aeroelastic system choice  ``''``
            ``solver_settings``     ``str``        Settings for the desired ``solver``              ``{}``
            ``max_iter``            ``int``        Maximum number of iterations                     ``100``
            ``fz_tolerance``        ``float``      Force tolerance in the ``z`` direction           ``0.01``
            ``fx_tolerance``        ``float``      Force tolerance in the ``x`` direction           ``0.01``
            ``m_tolerance``         ``float``      Moment tolerance                                 ``0.01``
            ``tail_cs_index``       ``int``        Control surface index                            ``0``
            ``thrust_nodes``        ``list(int)``  Index of nodes that provide thrust               ``[0]``
            ``initial_alpha``       ``float``      Initial angle of attack (radians)                ``0.0698``
            ``initial_deflection``  ``float``      Initial control surface deflection (radians)     ``0.0174``
            ``initial_thrust``      ``float``      Initial thrust per engine (N)                    ``0.0``
            ``initial_angle_eps``   ``float``      Initial angular variation for algorithm          ``0.0034``
            ``initial_thrust_eps``  ``float``      Initial thrust variation for algorithm           ``2.0``
            ======================  =============  ===============================================  ==========

        settings_types (dict): Acceptable data types for entries in ``settings``
        settings_default (dict): Default values for the available ``settings``
        data (ProblemData): object containing the information of the problem
        solver (BaseSolver): solver object employed for the solution of the problem
        n_input (int): number of inputs to vary to achieve trim
        i_iter (int): iteration number
        input_history (list): list of input history during iteration
        output_history (list): list of output history during iteration
        gradient_history (list): history of gradients during iteration
        trimmed_values (np.array): trim configuration values

    Methods:
        trim_algorithm: algorithm to find equilibrium conditions

    """
    solver_id = 'StaticTrim'

    def __init__(self):
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['print_info'] = 'bool'
        self.settings_default['print_info'] = True

        self.settings_types['solver'] = 'str'
        self.settings_default['solver'] = ''

        self.settings_types['solver_settings'] = 'dict'
        self.settings_default['solver_settings'] = dict()

        self.settings_types['max_iter'] = 'int'
        self.settings_default['max_iter'] = 100

        self.settings_types['fz_tolerance'] = 'float'
        self.settings_default['fz_tolerance'] = 0.01

        self.settings_types['fx_tolerance'] = 'float'
        self.settings_default['fx_tolerance'] = 0.01

        self.settings_types['m_tolerance'] = 'float'
        self.settings_default['m_tolerance'] = 0.01

        self.settings_types['tail_cs_index'] = 'int'
        self.settings_default['tail_cs_index'] = 0

        self.settings_types['thrust_nodes'] = 'list(int)'
        self.settings_default['thrust_nodes'] = np.array([0])

        self.settings_types['initial_alpha'] = 'float'
        self.settings_default['initial_alpha'] = 4.*np.pi/180.

        self.settings_types['initial_deflection'] = 'float'
        self.settings_default['initial_deflection'] = 1.*np.pi/180.

        self.settings_types['initial_thrust'] = 'float'
        self.settings_default['initial_thrust'] = 0.0

        self.settings_types['initial_angle_eps'] = 'float'
        self.settings_default['initial_angle_eps'] = 0.2*np.pi/180.

        self.settings_types['initial_thrust_eps'] = 'float'
        self.settings_default['initial_thrust_eps'] = 2.

        self.settings_types['relaxation_factor'] = 'float'
        self.settings_default['relaxation_factor'] = 0.2

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

        return return_value

    def trim_algorithm(self):
        """
        Trim algorithm method

        The trim condition is found iteratively.

        Returns:
            np.array: array of trim values for angle of attack, control surface deflection and thrust.
        """
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

            if self.settings['relaxation_factor'].value:
                for i_dim in range(3):
                    self.input_history[self.i_iter][i_dim] = (self.input_history[self.i_iter][i_dim]*(1 - self.settings['relaxation_factor'].value) +
                                                              self.input_history[self.i_iter][i_dim]*self.settings['relaxation_factor'].value)

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

    def evaluate(self, alpha, deflection_gamma, thrust):
        if not np.isfinite(alpha):
            1
        if not np.isfinite(deflection_gamma):
            1
        if not np.isfinite(thrust):
            1

        cout.cout_wrap('--', 2)
        cout.cout_wrap('Trying trim: ', 2)
        cout.cout_wrap('Alpha: ' + str(alpha*180/np.pi), 2)
        cout.cout_wrap('CS deflection: ' + str((deflection_gamma - alpha)*180/np.pi), 2)
        cout.cout_wrap('Thrust: ' + str(thrust), 2)
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
        cout.cout_wrap('Forces and moments:', 2)
        cout.cout_wrap('fx = ' + str(forces[0]) + ' mx = ' + str(moments[0]), 2)
        cout.cout_wrap('fy = ' + str(forces[1]) + ' my = ' + str(moments[1]), 2)
        cout.cout_wrap('fz = ' + str(forces[2]) + ' mz = ' + str(moments[2]), 2)

        return forcez, moment, forcex


























