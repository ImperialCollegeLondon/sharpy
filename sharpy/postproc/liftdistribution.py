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
        
        self.settings_types['folder'] = 'str'
        self.settings_default['folder'] = './output'

        self.settings = None
        self.data = None

        self.ts_max = None
        self.ts = None
        self.caller = None

    def initialise(self, data, custom_settings=None, caller=None):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)
        self.ts_max = len(self.data.structure.timestep_info)
        self.caller = caller

    def run(self, online=False):
        if not online:
            for self.ts in range(self.ts_max):
                self.lift_distribution()
            cout.cout_wrap('...Finished', 1)
        else:
            self.ts = len(self.data.aero.timestep_info) - 1
            self.lift_distribution()
        return self.data

    def lift_distribution(self):
        tstep = self.data.aero.timestep_info[self.data.ts]
        k_total = 0
        # get dimensions
        for i_surf in range(tstep.n_surf):
            n_dim, n_m, n_n = tstep.zeta[i_surf].shape
            k_total += n_n
            
        # get aero forces
        lift_distribution = np.zeros((k_total, 5))
        node_counter = 0
        for i_surf in range(tstep.n_surf):
            n_dim, n_m, n_n = tstep.zeta[i_surf].shape
            forces = tstep.forces[i_surf] + tstep.dynamic_forces[i_surf]
            abs_forces = np.zeros((n_m, n_n))
            for i_n in range(n_n):
                for i_m in range(n_m):
                    abs_forces[i_m, i_n] = np.abs(forces[2, i_m, i_n])
                lift_distribution[node_counter,4]=np.sum(abs_forces[:, i_n])  # forces
                lift_distribution[node_counter,3]=tstep.zeta[i_surf][2, 0, i_n]  #z 
                lift_distribution[node_counter,2]=tstep.zeta[i_surf][1, 0, i_n]  #y 
                lift_distribution[node_counter,1]=tstep.zeta[i_surf][0, 0, i_n]  #x 
                lift_distribution[node_counter,0]=i_surf
                node_counter += 1

        # Correct lift at nodes shared from different lifting surfaces
        for i_row in range(lift_distribution.shape[0]):
            for j_row in range(i_row+1,lift_distribution.shape[0]):
                if (np.round(lift_distribution[i_row,1:4],decimals=4) == np.round(lift_distribution[j_row,1:4],decimals=4)).all():
                    lift_distribution[i_row,4] += lift_distribution[j_row,4]
                    lift_distribution[j_row,4] = lift_distribution[i_row,4]
        # Export lift distribution data
        np.savetxt(self.settings["folder"]+'/lift_distribution.txt', lift_distribution, fmt='%i' + ', %10e'*4, delimiter = ", ", header= "i_surf, x,y,z, fz")

