import ctypes as ct

import sharpy.beam.utils.beamlib as beamlib
from sharpy.presharpy.utils.settings import str2bool
from sharpy.utils.solver_interface import solver, BaseSolver


@solver
class VibrationModes(BaseSolver):
    solver_id = 'VibrationModes'
    solver_type = 'structural'
    solver_unsteady = False

    def __init__(self):
        pass

    def initialise(self, data):
        self.data = data
        self.settings = data.settings[self.solver_id]
        self.convert_settings()
        data.beam.generate_aux_information()

    def run(self):
        print('Running modal solver...')
        beamlib.cbeam3_solv_modal(self.data.beam, self.settings)
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

