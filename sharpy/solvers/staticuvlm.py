import ctypes as ct

from sharpy.presharpy.utils.settings import str2bool
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.cout_utils as cout
import sharpy.presharpy.aerogrid.aerogrid as aerogrid
import sharpy.aero.utils.uvlmlib as uvlmlib
import sharpy.presharpy.utils.settings as settings
import sharpy.presharpy.aerogrid.utils as aero_utils


@solver
class StaticUvlm(BaseSolver):
    solver_id = 'StaticUvlm'
    solver_type = 'aero'

    def __init__(self):
        pass

    def initialise(self, data):
        self.ts = 0
        self.data = data
        self.settings = data.settings[self.solver_id]
        self.convert_settings()
        cout.cout_wrap('Generating aero grid...', 1)
        self.data.grid = aerogrid.AeroGrid(self.data.beam,
                                           self.data.aero_data_dict,
                                           self.settings)
        self.data.flightconditions = settings.load_config_file(self.data.case_route +
                                                               '/' +
                                                               self.data.case_name +
                                                               '.flightcon.txt')
        aero_utils.flightcon_file_parser(self.data.flightconditions)

        cout.cout_wrap('...Finished', 1)

    def update(self):
        pass

    def run(self):
        cout.cout_wrap('Running static UVLM solver...', 1)
        uvlmlib.vlm_solver(self.data.grid.timestep_info[self.ts],
                           self.data.flightconditions)

        cout.cout_wrap('...Finished', 1)
        return self.data


    def convert_settings(self):
        self.settings['print_info'] = str2bool(self.settings['print_info'])
        self.settings['rollup'] = str2bool(self.settings['rollup'])
        self.settings['aligned_grid'] = str2bool(self.settings['aligned_grid'])
        self.settings['prescribed_wake'] = str2bool(self.settings['prescribed_wake'])
        self.settings['mstar'] = int(self.settings['mstar'])

