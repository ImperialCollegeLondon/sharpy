"""@package PyBeam.Solver.NonlinearStatic
@brief      Nonlinear static solvers.
@author     Rob Simpson
@contact    r.simpson11@imperial.ac.uk
@version    0.0
@date       25/10/2012
@pre        None
@warning    None

@modified   Alfonso del Carre
"""

# import DerivedTypes
# import BeamIO
# import beam.utils.derivedtypes as derivedtypes
# import XbeamLib
# import BeamInit
import ctypes as ct

import numpy as np

import sharpy.beam.utils.beamlib as beamlib
from sharpy.presharpy.utils.settings import str2bool
from sharpy.utils.solver_interface import solver, BaseSolver


@solver
class NonLinearStatic(BaseSolver):
    solver_id = 'NonLinearStatic'
    solver_type = 'structural'

    def __init__(self):
        pass

    def initialise(self, data):
        self.data = data
        self.settings = data.settings[self.solver_id]
        self.convert_settings()
        data.beam.generate_aux_information()

    def run(self):
        print('Running non linear static solver...')
        beamlib.cbeam3_solv_nlnstatic(self.data.beam, self.settings)
        self.data.beam.update()
        print('...Finished')
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

