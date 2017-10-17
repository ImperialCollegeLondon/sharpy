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
class UnsteadyVlm(BaseSolver):
    solver_id = 'UnsteadyVlm'
    solver_type = 'aero'
    solver_unsteady = True

    def __init__(self):
        pass

    def initialise(self, data, quiet=False):
        self.ts = 0
        self.data = data
        self.settings = data.settings[self.solver_id]
        self.quiet = quiet
        self.convert_settings()
        # if self.with_rb:
        self.generate_rbm()
        if not self.quiet:
            cout.cout_wrap('Generating aero grid...', 1)

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

        uvlmlib.uvlm_init(self.data.grid.timestep_info[self.ts],
                          self.data.beam.timestep_info[self.ts],
                          self.data.flightconditions,
                          self.settings,
                          self.inertial2aero)
        if not self.quiet:
            cout.cout_wrap('...Finished', 1)

    def update(self):
        pass

    def run(self):
        # if not self.quiet:
        #     cout.cout_wrap('Running static UVLM solver...', 1)
        for i in range(self.settings['n_time_steps']):
            self.ts = i
            self.t = i*self.settings['dt']
            if i > 0:
                self.data.grid.add_timestep()
                self.data.beam.add_timestep(with_rb=True)
            self.data.grid.ts = i
            self.data.grid.t = i*self.settings['dt']
            self.data.beam.ts = i
            self.data.beam.t = i*self.settings['dt']

            self.data.beam.timestep_info[self.ts].with_rb = True
            self.data.beam.timestep_info[self.ts].for_vel = \
                self.rbm_vel[i, :]
            self.data.beam.timestep_info[self.ts].for_pos = \
                self.data.beam.timestep_info[self.ts - 1].for_pos + self.rbm_vel[i, :]*self.settings['dt']
            #     self.data.vector_generator['rbm_pos'][self.ts, :]
            #     self.data.vector_generator['rbm_vel'][self.ts, :]
            print('it = %i' % i)
            if i > 0:
                uvlmlib.uvlm_solver(i,
                                    self.data.grid.timestep_info[self.ts],
                                    self.data.grid.timestep_info[self.ts - 1],
                                    self.data.beam.timestep_info[self.ts],
                                    self.data.flightconditions,
                                    self.settings,
                                    self.inertial2aero)
            else:
                uvlmlib.uvlm_solver(i,
                                    self.data.grid.timestep_info[self.ts],
                                    self.data.grid.timestep_info[self.ts],
                                    self.data.beam.timestep_info[self.ts],
                                    self.data.flightconditions,
                                    self.settings,
                                    self.inertial2aero)
        return self.data

    def generate_rbm(self):
        try:
            self.rbm_pos = self.data.dyn_data_dict['rbm_pos']
            self.rbm_vel = self.data.dyn_data_dict['rbm_vel']
        except:
            self.rbm_pos = np.zeros((self.settings['n_time_steps'], 6))
            self.rbm_vel = np.zeros((self.settings['n_time_steps'], 6))

        # class Empty(object):
        #     def __init__(self):
        #         pass
        #
        # self.temp_data = Empty()
        # self.temp_data.dummy_settings = dict()
        # self.temp_data.dummy_settings['n_time_steps'] = self.settings['n_time_steps']
        # self.temp_data.dummy_settings['size'] = 6
        # self.temp_data.dummy_settings['values'] = self.settings['rbm_pos']
        # self.temp_data.dummy_settings['vector_name'] = 'rbm_pos'
        # self.temp_data.dummy_settings['include_derivative'] = 'on'
        # self.temp_data.dummy_settings['derivative_name'] = 'rbm_vel'
        # self.temp_data.dummy_settings['dt'] = self.settings['dt']
        #
        # gen = solver_interface.initialise_solver('VectorGenerator')
        # gen.initialise(self.temp_data.dummy_settings)
        # self.temp_data = gen.run()
        #



    def convert_settings(self):
        self.settings['print_info'] = str2bool(self.settings['print_info'])
        self.settings['aligned_grid'] = str2bool(self.settings['aligned_grid'])
        self.settings['mstar'] = int(self.settings['mstar'])
        try:
            self.settings['num_cores'] = int(self.settings['num_cores'])
        except KeyError:
            self.settings['num_cores'] = 4
        try:
            self.settings['steady_n_rollup'] = int(self.settings['steady_n_rollup'])
        except KeyError:
            self.settings['steady_n_rollup'] = 0
        try:
            self.settings['steady_rollup_tolerance'] = float(self.settings['steady_rollup_tolerance'])
        except KeyError:
            self.settings['steady_rollup_tolerance'] = 1e-5
        try:
            self.settings['steady_rollup_aic_refresh'] = int(self.settings['steady_rollup_aic_refresh'])
        except KeyError:
            self.settings['steady_rollup_aic_refresh'] = 1

        try:
            self.settings['n_time_steps'] = int(self.settings['n_time_steps'])
        except KeyError:
            cout.cout_wrap('No n_time_steps defined in PrescribedUvlm. Using the default: 1000', 2)
            self.settings['n_time_steps'] = 1000
        try:
            self.settings['dt'] = float(self.settings['dt'])
        except KeyError:
            self.settings['dt'] = 0.01
        try:
            self.settings['convection_scheme'] = int(self.settings['convection_scheme'])
        except KeyError:
            self.settings['convection_scheme'] = 0
        try:
            self.settings['iterative_solver'] = str2bool(self.settings['iterative_solver'])
        except KeyError:
            self.settings['iterative_solver'] = False
        try:
            self.settings['iterative_tol'] = float(self.settings['iterative_tol'])
        except KeyError:
            self.settings['iterative_tol'] = 0.0
        try:
            self.settings['iterative_precond'] = str2bool(self.settings['iterative_precond'])
        except KeyError:
            self.settings['iterative_precond'] = False

        self.with_rb = True

