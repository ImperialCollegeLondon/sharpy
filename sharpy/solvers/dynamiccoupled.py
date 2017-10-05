import ctypes as ct
import numpy as np

from sharpy.presharpy.utils.settings import str2bool
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.cout_utils as cout
import sharpy.presharpy.aerogrid.aerogrid as aerogrid
import sharpy.aero.utils.uvlmlib as uvlmlib
import sharpy.presharpy.utils.settings as settings
import sharpy.presharpy.aerogrid.utils as aero_utils
import sharpy.aero.utils.mapping as mapping

import sharpy.utils.solver_interface as solver_interface


@solver
class StaticCoupled(BaseSolver):
    solver_id = 'StaticCoupled'
    solver_type = 'coupled'

    def __init__(self):
        pass

    def initialise(self, data):
        self.ts = 0
        self.data = data
        self.settings = data.settings[self.solver_id]
        self.convert_settings()

        self.structural_solver = solver_interface.initialise_solver(self.structural_solver_id)
        self.structural_solver.initialise(self.data)
        self.aero_solver = solver_interface.initialise_solver(self.aero_solver_id)
        self.aero_solver.initialise(self.structural_solver.data, quiet=True)
        self.data = self.aero_solver.data

        self.original_u_inf = self.data.flightconditions['FlightCon']['u_inf']

        self.previous_forces = np.zeros((self.data.beam.num_node, 6))

    def run(self):
        cout.cout_wrap('Running static coupled solver...', 1)

        for i_step in range(self.settings['n_load_steps']):
            coeff = ct.c_double((i_step + 1.0)/(self.settings['n_load_steps']))
            for i_iter in range(self.settings['max_iter']):
                cout.cout_wrap('Iter: %u, step: %u' % (i_iter, i_step), 2)

                self.aero_solver.initialise(self.data, update_flightcon=False, quiet=True)
                self.data = self.aero_solver.run()

                struct_forces = mapping.aero2struct_force_mapping(
                    self.data.grid.timestep_info[self.ts].forces,
                    self.data.grid.struct2aero_mapping,
                    self.data.grid.timestep_info[self.ts].zeta,
                    self.data.beam.timestep_info[self.ts].pos_def,
                    self.data.beam.timestep_info[self.ts].psi_def,
                    self.data.beam.node_master_elem,
                    self.data.beam.master,
                    self.data.grid.inertial2aero)

                if self.relaxation_factor > 0.0:
                    struct_forces = ((1.0 - self.relaxation_factor)*struct_forces +
                                    self.relaxation_factor*self.previous_forces)

                self.previous_forces = struct_forces
                self.data.beam.update_forces(struct_forces)
                self.structural_solver.initialise(self.data)
                self.data = self.structural_solver.run(coeff)

                if self.convergence(i_iter):
                    self.data.flightconditions['FlightCon']['u_inf'] = self.original_u_inf
                    self.aero_solver.initialise(self.data, quiet=True)
                    self.data = self.aero_solver.run()
                    if i_step == self.settings['n_load_steps'] - 1:
                        cout.cout_wrap('Converged in %u iterations' % (i_iter + 1), 2)
                    break

        cout.cout_wrap('...Finished', 1)
        return self.data

    def convergence(self, i_iter):

        if i_iter == self.settings['max_iter'] - 1:
            cout.cout_wrap('StaticCoupled did not converge!', 0)
            quit(-1)

        return_value = None
        if i_iter == 0:
            self.initial_residual = np.linalg.norm(self.data.beam.timestep_info[0].pos_def)
            self.previous_residual = self.initial_residual
            self.current_residual = self.initial_residual
            return False

        self.current_residual = np.linalg.norm(self.data.beam.timestep_info[0].pos_def)
        cout.cout_wrap('Res = %8e' % (np.abs(self.current_residual - self.previous_residual)/self.previous_residual), 2)

        if return_value is None:
            if np.abs(self.current_residual - self.previous_residual)/self.initial_residual < self.settings['tolerance']:
                return_value = True
            else:
                self.previous_residual = self.current_residual
                return_value = False

        if return_value is None:
            return_value = False

        return return_value

    def convert_settings(self):
        self.settings['print_info'] = str2bool(self.settings['print_info'])
        self.aero_solver_id = str(self.settings['aero_solver'])
        self.structural_solver_id = str(self.settings['structural_solver'])
        self.settings['max_iter'] = int(self.settings['max_iter'])
        self.settings['n_load_steps'] = int(self.settings['n_load_steps'])
        self.settings['tolerance'] = float(self.settings['tolerance'])
        try:
            self.settings['residual_plot'] = str2bool(self.settings['residual_plot'])
        except KeyError:
            self.settings['residual_plot'] = True
        try:
            self.settings['relaxation_factor'] = float(self.settings['relaxation_factor'])
            self.relaxation_factor = self.settings['relaxation_factor']
            if self.settings['relaxation_factor'] < 0 or self.settings['relaxation_factor'] >= 1:
                raise ValueError('Relaxation factor cannot be <0 or >=1')
        except KeyError:
            self.settings['relaxation_factor'] = 0.0
