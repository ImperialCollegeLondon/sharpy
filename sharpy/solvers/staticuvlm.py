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


@solver
class StaticUvlm(BaseSolver):
    solver_id = 'StaticUvlm'
    solver_type = 'aero'

    def __init__(self):
        pass

    def initialise(self, data, update_flightcon=True, quiet=False):
        self.ts = 0
        self.data = data
        self.settings = data.settings[self.solver_id]
        self.quiet = quiet
        self.convert_settings()
        if not self.quiet:
            cout.cout_wrap('Generating aero grid...', 1)
        if update_flightcon:
            self.data.flightconditions = settings.load_config_file(self.data.case_route +
                                                                   '/' +
                                                                   self.data.case_name +
                                                                   '.flightcon.txt')
            aero_utils.flightcon_file_parser(self.data.flightconditions)

        try:
            self.inertial2aero = self.data.beam.orientation
        except AttributeError:
            self.inertial2aero = mapping.inertial2aero_rotation(self.data.flightconditions['FlightCon']['alpha'],
                                                                self.data.flightconditions['FlightCon']['beta'])
        if self.inertial2aero is None:
            self.inertial2aero = mapping.inertial2aero_rotation(self.data.flightconditions['FlightCon']['alpha'],
                                                                self.data.flightconditions['FlightCon']['beta'])

        self.data.grid = aerogrid.AeroGrid(self.data.beam,
                                           self.data.aero_data_dict,
                                           self.settings,
                                           inertial2aero=self.inertial2aero,
                                           quiet=self.quiet)
        # self.data.grid.initialise_steady_wake(self.data.flightconditions)
        if not self.quiet:
            cout.cout_wrap('...Finished', 1)

    def update(self):
        pass

    def run(self):
        if not self.quiet:
            cout.cout_wrap('Running static UVLM solver...', 1)
        uvlmlib.vlm_solver(self.data.grid.timestep_info[self.ts],
                           self.data.flightconditions,
                           self.settings)

        if not self.quiet:
            cout.cout_wrap('...Finished', 1)
        return self.data

    def convert_settings(self):
        self.settings['print_info'] = str2bool(self.settings['print_info'])
        self.settings['rollup'] = str2bool(self.settings['rollup'])
        self.settings['aligned_grid'] = str2bool(self.settings['aligned_grid'])
        self.settings['prescribed_wake'] = str2bool(self.settings['prescribed_wake'])
        self.settings['horseshoe'] = str2bool(self.settings['horseshoe'])
        self.settings['mstar'] = int(self.settings['mstar'])
        try:
            self.settings['num_cores'] = int(self.settings['num_cores'])
        except KeyError:
            self.settings['num_cores'] = 4
        try:
            self.settings['n_rollup'] = int(self.settings['n_rollup'])
        except KeyError:
            self.settings['n_rollup'] = 0
        try:
            self.settings['rollup_tolerance'] = float(self.settings['rollup_tolerance'])
        except KeyError:
            self.settings['rollup_tolerance'] = 1e-5
        try:
            self.settings['rollup_aic_refresh'] = int(self.settings['rollup_aic_refresh'])
        except KeyError:
            self.settings['rollup_aic_refresh'] = 1
        try:
            self.settings['rollup_dt'] = float(self.settings['rollup_dt'])
        except KeyError:
            self.settings['rollup_dt'] = 0.01

