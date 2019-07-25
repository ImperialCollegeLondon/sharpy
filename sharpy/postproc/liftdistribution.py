import os

import numpy as np
from tvtk.api import tvtk, write_data

import sharpy.utils.algebra as algebra
import sharpy.utils.cout_utils as cout
from sharpy.utils.settings import str2bool
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
from sharpy.utils.datastructures import init_matrix_structure, standalone_ctypes_pointer
import sharpy.aero.utils.uvlmlib as uvlmlib


@solver
class LiftDistribution(BaseSolver):
    solver_id = 'LiftDistribution'

    def __init__(self):
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['print_info'] = 'bool'
        self.settings_default['print_info'] = True

        self.settings_types['normalise'] = 'bool'
        self.settings_default['normalise'] = True

        self.settings = None
        self.data = None

        self.ts_max = None
        self.ts = None

    def initialise(self, data, custom_settings=None):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)
        self.ts_max = len(self.data.structure.timestep_info)

    def run(self, online=False):
        if not online:
            for self.ts in range(self.ts_max):
                self.lift_distribution()
            cout.cout_wrap('...Finished', 1)
        else:
            self.ts = len(self.data.structure.timestep_info) - 1
            self.lift_distribution()
        return self.data

    def lift_distribution(self):
        # add entry to dictionary for postproc
        tstep = self.data.aero.timestep_info[self.ts]
        tstep.postproc_cell['lift_distribution'] = init_matrix_structure(dimensions=tstep.dimensions,
                                                                           with_dim_dimension=False)

        norm = 1.0

        for i_surf in range(tstep.n_surf):
            n_dim, n_m, n_n = tstep.zeta[i_surf].shape

            forces = tstep.forces[i_surf] + tstep.unsteady_forces[i_surf]
            abs_forces = np.zeros((n_m, n_n))
            chord = np.zeros((n_n,))
            for i_n in range(n_n):
                chord[i_n] = np.abs(tstep[i_surf].zeta[:, -1, i_n] - tstep[i_surf].zeta[:, 0, i_n])
                if not i_surf and not i_n:
                    if self.settings['normalise']:
                        norm = chord[0]

                for i_m in range(n_m):
                    abs_forces[i_n, i_m] = np.abs(forces[:, i_m, i_n])

                tstep.postproc_cell['lift_distribution'][i_surf][i_n, i_m] = np.sum(abs_forces[i_n, :], axis = 2)/norm

