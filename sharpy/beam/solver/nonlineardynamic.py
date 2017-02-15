"""
@modified   Alfonso del Carre
"""

import ctypes as ct
import numpy as np

import sharpy.beam.utils.beamlib as beamlib
from presharpy.utils.settings import str2bool
from sharpy.utils.solver_interface import solver, BaseSolver


@solver
class NonLinearDynamic(BaseSolver):
    solver_id = 'NonLinearDynamic'
    solver_type = 'structural'
    solver_unsteady = True

    def __init__(self):
        pass

    def initialise(self, data):
        self.data = data
        self.settings = data.settings[self.solver_id]
        self.convert_settings()
        data.beam.read_dynamic_data()
        data.beam.generate_aux_information(dynamic=True)

    def run(self):
        prescribed_motion = True
        try:
            prescribed_motion = self.settings['prescribed_motion'].value
        except KeyError:
            pass
        if prescribed_motion is True:
            print('Running non linear dynamic solver...')
            beamlib.cbeam3_solv_nlndyn(self.data.beam, self.settings)
        else:
            print('Running non linear dynamic solver with RB...')
            beamlib.xbeam_solv_couplednlndyn(self.data.beam, self.settings)
        self.data.beam.update()
        print('...Finished')
        return self.data

    def convert_settings(self):
        try:
            self.settings['print_info'] = ct.c_bool(str2bool(self.settings['print_info']))
        except KeyError:
            pass
        try:
            self.settings['out_b_frame'] = ct.c_bool(str2bool(self.settings['out_b_frame']))
        except KeyError:
            pass

        try:
            self.settings['out_a_frame'] = ct.c_bool(str2bool(self.settings['out_a_frame']))
        except KeyError:
            pass

        try:
            self.settings['elem_proj'] = ct.c_int(int(self.settings['elem_proj']))
        except KeyError:
            pass

        try:
            self.settings['max_iterations'] = ct.c_int(int(self.settings['max_iterations']))
        except KeyError:
            pass

        try:
            self.settings['num_load_steps'] = ct.c_int(int(self.settings['num_load_steps']))
        except KeyError:
            pass

        try:
            self.settings['delta_curved'] = ct.c_double(float(self.settings['delta_curved']))
        except KeyError:
            pass

        try:
            self.settings['min_delta'] = ct.c_double(float(self.settings['min_delta']))
        except KeyError:
            pass

        try:
            self.settings['newmark_damp'] = ct.c_double(float(self.settings['newmark_damp']))
        except KeyError:
            pass

        try:
            self.settings['prescribed_motion'] = ct.c_bool(str2bool(self.settings['prescribed_motion']))
        except KeyError:
            pass

        try:
            self.settings['dt'] = ct.c_double(float(self.settings['dt']))
        except KeyError:
            pass

        try:
            self.settings['num_steps'] = ct.c_int(int(self.settings['num_steps']))
        except KeyError:
            pass

        try:
            self.settings['gravity_on'] = ct.c_bool(str2bool(self.settings['gravity_on']))
        except KeyError:
            pass

        try:
            self.settings['gravity'] = ct.c_double(float(self.settings['gravity']))
        except KeyError:
            pass

        try:
            self.settings['gravity_dir'] = np.fromstring(self.settings['gravity_dir'], sep=',', dtype=ct.c_double)
        except KeyError:
            pass
