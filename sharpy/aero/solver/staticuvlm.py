import ctypes as ct

from sharpy.presharpy.utils.settings import str2bool
from sharpy.utils.solver_interface import solver, BaseSolver


@solver
class StaticUvlm(BaseSolver):
    solver_id = 'StaticUvlm'
    solver_type = 'aero'

    def __init__(self):
        pass

    def initialise(self, data):
        self.data = data
        self.settings = data.settings[self.solver_id]
        self.convert_settings()

    def run(self):
        print('Running static UVLM solver...')
        print('...Finished')
        return self.data

    def convert_settings(self):
        self.settings['print_info'] = ct.c_bool(str2bool(self.settings['print_info']))
