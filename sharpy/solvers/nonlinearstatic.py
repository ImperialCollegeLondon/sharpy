import ctypes as ct
import numpy as np

import sharpy.beam.utils.beamlib as beamlib
from sharpy.presharpy.utils.settings import str2bool
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.cout_utils as cout


@solver
class NonLinearStatic(BaseSolver):
    solver_id = 'NonLinearStatic'
    solver_type = 'structural'

    def __init__(self):
        pass

    def initialise(self, data, quiet=False):
        self.data = data
        self.settings = data.settings[self.solver_id]
        self.convert_settings()
        self.quiet = quiet
        # data.beam.timestep_info[0].pos_def[:] = data.beam.pos_ini[:]
        # data.beam.timestep_info[0].psi_def[:] = data.beam.psi_ini[:]
        data.beam.generate_aux_information()

    def run(self, coeff=ct.c_double(1.0)):
        if self.quiet:
            cout.cout_wrap('Running non linear static solver...', 2)
        beamlib.cbeam3_solv_nlnstatic(self.data.beam, self.settings, coeff)
        self.data.beam.update()
        if self.quiet:
            cout.cout_wrap('...Finished', 2)
        return self.data

    def convert_settings(self):
        try:
            self.settings['print_info'] = ct.c_bool(str2bool(self.settings['print_info']))
        except ValueError:
            pass
        try:
            self.settings['out_b_frame'] = ct.c_bool(str2bool(self.settings['out_b_frame']))
        except ValueError:
            pass
        try:
            self.settings['out_a_frame'] = ct.c_bool(str2bool(self.settings['out_a_frame']))
        except ValueError:
            pass
        try:
            self.settings['elem_proj'] = ct.c_int(int(self.settings['elem_proj']))
        except ValueError:
            pass
        try:
            self.settings['max_iterations'] = ct.c_int(int(self.settings['max_iterations']))
        except ValueError:
            pass
        try:
            self.settings['num_load_steps'] = ct.c_int(int(self.settings['num_load_steps']))
        except ValueError:
            pass
        try:
            self.settings['delta_curved'] = ct.c_double(float(self.settings['delta_curved']))
        except ValueError:
            pass
        try:
            self.settings['min_delta'] = ct.c_double(float(self.settings['min_delta']))
        except ValueError:
            pass
        try:
            self.settings['newmark_damp'] = ct.c_double(float(self.settings['newmark_damp']))
        except ValueError:
            pass
        try:
            self.settings['gravity_on'] = ct.c_bool(str2bool(self.settings['gravity_on']))
        except ValueError:
            pass
        try:
            self.settings['gravity'] = ct.c_double(float(self.settings['gravity']))
        except ValueError:
            pass
        try:
            if isinstance(self.settings['gravity_dir'], np.ndarray):
                pass
            else:
                self.settings['gravity_dir'] = np.fromstring(self.settings['gravity_dir'], sep=',', dtype=ct.c_double)
        except ValueError:
            pass

