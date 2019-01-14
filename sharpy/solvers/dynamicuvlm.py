"""
Time Domain Aerodynamic Solver

N Goizueta Jan 19
"""

import ctypes as ct
import numpy as np

import sharpy.utils.solver_interface as solver_interface
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.utils.cout_utils as cout

@solver
class DynamicUVLM(BaseSolver):
    """
    Dynamic Aerodynamic Time Domain Simulation

    Provides an aerodynamic only simulation in time by time stepping the solution. The type of aerodynamic solver is
    parsed as a setting.

    Warnings:
        Under development

    """
    solver_id = 'DynamicUVLM'

    def __init__(self):
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['print_info'] = 'bool'
        self.settings_default['print_info'] = False

        self.settings_types['aero_solver'] = 'str'
        self.settings_default['aero_solver'] = None

        self.settings_types['aero_solver_settings'] = 'dict'
        self.settings_default['aero_solver_settings'] = None

        self.settings_types['n_time_steps'] = 'int'
        self.settings_default['n_time_steps'] = 100

        self.settings_types['dt'] = 'float'
        self.settings_default['dt'] = 0.05

        self.settings_types['postprocessors'] = 'list(str)'
        self.settings_default['postprocessors'] = list()

        self.settings_types['postprocessors_settings'] = 'dict'
        self.settings_default['postprocessors_settings'] = dict()

        self.data = None
        self.settings = None
        self.aero_solver = None
        self.print_info = False
        self.dt = None
        self.residual_table = None

        self.postprocessors = dict()
        self.with_postprocessors = False

    def initialise(self, data, custom_settings=None):
        self.data = data

        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings

        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)
        self.dt = self.settings['dt']
        self.print_info = self.settings['print_info'].value

        self.aero_solver = solver_interface.initialise_solver(self.settings['aero_solver'])
        self.aero_solver.initialise(self.data, self.settings['aero_solver_settings'])
        self.data = self.aero_solver.data

        # initialise postprocessors
        self.postprocessors = dict()
        if len(self.settings['postprocessors']) > 0:
            self.with_postprocessors = True
        for postproc in self.settings['postprocessors']:
            self.postprocessors[postproc] = solver_interface.initialise_solver(postproc)
            self.postprocessors[postproc].initialise(
                self.data, self.settings['postprocessors_settings'][postproc])

        if self.print_info:
            self.residual_table = cout.TablePrinter(2, 14, ['g', 'f'])
            self.residual_table.print_header(['ts', 't'])

    def run(self):

        # struct info - only for orientation, no structural solution is performed
        struct_ini_step = self.data.structure.timestep_info[-1]


        for self.data.ts in range(len(self.data.aero.timestep_info),
                                  len(self.data.aero.timestep_info) + self.settings['n_time_steps'].value):

            aero_tstep = self.data.aero.timestep_info[-1]

            self.data = self.aero_solver.run(aero_tstep,
                                             struct_ini_step,
                                             convect_wake=True,
                                             unsteady_contribution=True)

            self.aero_solver.add_step()
            self.data.aero.timestep_info[-1] = aero_tstep.copy()

            if self.print_info:
                self.residual_table.print_line([self.data.ts,
                                                self.data.ts * self.dt.value])

            if self.with_postprocessors:
                for postproc in self.postprocessors:
                    self.data = self.postprocessors[postproc].run(online=True)

        if self.print_info:
            cout.cout_wrap('...Finished', 1)
        return self.data
