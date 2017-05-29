import ctypes as ct

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
        self.structural_solver.initialise(data)
        self.aero_solver = solver_interface.initialise_solver(self.aero_solver_id)
        self.aero_solver.initialise(data)

    def run(self):
        cout.cout_wrap('Running static coupled solver...', 1)

        for i_iter in range(self.settings['max_iter']):
            self.aero_solver.initialise(self.data)
            self.data = self.aero_solver.run()
            for i_load_step in range(self.settings['n_load_steps']):
                coeff = float(i_load_step)/self.settings['n_load_steps']
                struct_forces = coeff*mapping.aero2struct_force_mapping(
                    self.data.grid.timestep_info[self.ts].forces,
                    self.data.grid.aero2struct_mapping,
                    self.data.grid.timestep_info[self.ts].zeta,
                    self.data.beam.pos_def,
                    self.data.beam.psi_def,
                    self.data.beam.node_master_elem
                )
                self.structural_solver.initialise(self.data)
                self.data.beam.update_forces(struct_forces)
                self.data = self.structural_solver.run()


        cout.cout_wrap('...Finished', 1)
        return self.data

    def convert_settings(self):
        self.settings['print_info'] = str2bool(self.settings['print_info'])
        self.aero_solver_id = str(self.settings['aero_solver'])
        self.structural_solver_id = str(self.settings['structural_solver'])
        self.settings['max_iter'] = int(self.settings['max_iter'])
        self.settings['n_load_steps'] = int(self.settings['n_load_steps'])
        self.settings['tolerance'] = float(self.settings['tolerance'])
