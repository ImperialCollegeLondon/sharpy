import os

import numpy as np

import sharpy.utils.cout_utils as cout
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.aero.utils.mapping as mapping
import sharpy.utils.algebra as algebra


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
        rot = algebra.quat2rotation(struct_tstep.quat)
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
        for inode in range(N_nodes):
            if self.data.aero.aero_dict['aero_node'][inode]:
                lift_distribution[inode,3]=np.dot(rot.T, forces[inode, :3])[2]  # lift force (z direction in A frame or B frame)
                lift_distribution[inode,2]=struct_tstep.pos[inode, 2]  #z
                lift_distribution[inode,1]=struct_tstep.pos[inode, 1]  #y
                lift_distribution[inode,0]=struct_tstep.pos[inode, 0]  #x

        if self.settings["normalise"]: 
            # get lift coefficient           
            strip_area = self.calculate_panel_area(aero_tstep)  
            # TODO: add nondimensional spanwise column y/s
            header += ",cl"
            numb_col += 1
            lift_distribution = np.concatenate((lift_distribution, np.zeros((N_nodes,1))), axis=1)
            for inode in range(N_nodes):                
                if self.data.aero.aero_dict['aero_node'][inode]:               
                    local_node = self.data.aero.struct2aero_mapping[inode][0]["i_n"]             
                    ielem, _ = self.data.structure.node_master_elem[inode]                    
                    i_surf = int(self.data.aero.surface_distribution[ielem])
                    lift_distribution[inode,4] = lift_distribution[inode,3]/(self.settings['q_ref']*\
                            strip_area[i_surf][local_node]) # cl                           
                            
                    # Check if shared nodes from different surfaces exist (e.g. two wings joining at symmetry plane)
                    # Leads to error since panel area just donates for half the panel size while lift forces is summed up
                    lift_distribution[inode,4] /= len(self.data.aero.struct2aero_mapping[inode])

        # Export lift distribution data
        np.savetxt(self.settings["folder"]+'/lift_distribution.txt', lift_distribution, fmt='%10e,'*(numb_col-1)+'%10e', delimiter = ", ", header= header)
    
    def calculate_panel_area(self, aero_tstep):
        # Function to get the panel area (TODO: Better description)
        strip_area = []
        for i_surf in range(self.data.aero.n_surf):
            N_panel = self.data.aero.aero_dimensions[i_surf][1]
            array_panel_area = np.zeros((N_panel))
            # the area is calculated for all chordwise panels together
            for i_panel in range(N_panel):
                array_panel_area[i_panel] = algebra.panel_area(
                    aero_tstep.zeta[i_surf][:, -1, i_panel],
                    aero_tstep.zeta[i_surf][:, 0, i_panel], 
                    aero_tstep.zeta[i_surf][:, 0, i_panel+1], 
                    aero_tstep.zeta[i_surf][:, -1, i_panel+1])
            # assume each strip shares half of each adjacent panel
            strip_area.append(np.zeros((N_panel+1)))
            strip_area[i_surf][:-1] = abs(np.roll(array_panel_area[:],1)+array_panel_area[:]).reshape(-1)
            strip_area[i_surf][0] = abs(array_panel_area[0])            
            strip_area[i_surf][-1] = abs(array_panel_area[-1])         
            strip_area[i_surf][:] /= 2
        
        return strip_area