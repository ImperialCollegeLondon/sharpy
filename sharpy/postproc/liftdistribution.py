import os

import numpy as np

import sharpy.utils.cout_utils as cout
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.aero.utils.mapping as mapping


@solver
class LiftDistribution(BaseSolver):
    """
    Exports lift distribution to txt file
    """
    solver_id = 'LiftDistribution'

    def __init__(self):
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['print_info'] = 'bool'
        self.settings_default['print_info'] = True

        # TO-DO: implement option for normalization
        self.settings_types['normalise'] = 'bool'
        self.settings_default['normalise'] = True

        self.settings_types['folder'] = 'str'
        self.settings_default['folder'] = './output'

        self.settings = None
        self.data = None

        self.caller = None

    def initialise(self, data, custom_settings=None, caller=None):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)
        self.caller = caller

    def run(self, online=False):
        if not online:
            for it in range(len(self.data.structure.timestep_info)):
                self.lift_distribution(self.data.structure.timestep_info[it],
                                       self.data.aero.timestep_info[it])
            cout.cout_wrap('...Finished', 1)
        else:
            self.lift_distribution(self.data.structure.timestep_info[self.data.ts],
                                   self.data.aero.timestep_info[self.data.ts])
        return self.data

    def lift_distribution(self, struct_tstep, aero_tstep):        
        # Force mapping
        forces = mapping.aero2struct_force_mapping(
            aero_tstep.forces + aero_tstep.dynamic_forces,
            self.data.aero.struct2aero_mapping,
            aero_tstep.zeta,
            struct_tstep.pos,
            struct_tstep.psi,
            self.data.structure.node_master_elem,
            self.data.structure.connectivities,
            struct_tstep.cag(),
            self.data.aero.aero_dict)
        # Prepare output matrix and file 
        N_nodes = self.data.structure.num_node
        numb_col = 4
        header= "x,y,z,fz"
        # get aero forces
        lift_distribution = np.zeros((N_nodes, numb_col))
        for inode in range(self.data.structure.num_node):
            lift_distribution[inode,4]=np.linalg.norm(forces[inode, 0:3])  # forces
            lift_distribution[inode,3]=struct_tstep.pos[inode, 2]  #z
            lift_distribution[inode,2]=struct_tstep.pos[inode, 1]  #y
            lift_distribution[inode,1]=struct_tstep.pos[inode, 0]  #x
            
        # Export lift distribution data
        np.savetxt(self.settings["folder"]+'/lift_distribution.txt', lift_distribution, fmt='%10e,'*(numb_col-1)+'%10e', delimiter = ", ", header= header)
