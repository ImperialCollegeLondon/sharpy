import os

import numpy as np
from tvtk.api import tvtk, write_data

import sharpy.utils.cout_utils as cout
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra
import sharpy.structure.utils.xbeamlib as xbeamlib
#from IPython import embed


@solver
class WriteVariablesTime(BaseSolver):
    solver_id = 'WriteVariablesTime'

    def __init__(self):
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['delimiter'] = 'str'
        self.settings_default['delimiter'] = ' '

        self.settings_types['structure_variables'] = 'list(str)'
        self.settings_default['structure_variables'] = ''

        self.settings_types['structure_nodes'] = 'list(int)'
        self.settings_default['structure_nodes'] = np.array([-1])

        self.settings_types['aero_panels_variables'] = 'list(str)'
        self.settings_default['aero_panels_variables'] = ''

        self.settings_types['aero_panels_isurf'] = 'list(int)'
        self.settings_default['aero_panels_isurf'] = np.array([0])
        self.settings_types['aero_panels_im'] = 'list(int)'
        self.settings_default['aero_panels_im'] = np.array([0])
        self.settings_types['aero_panels_in'] = 'list(int)'
        self.settings_default['aero_panels_in'] = np.array([0])

        self.settings_types['aero_nodes_variables'] = 'list(str)'
        self.settings_default['aero_nodes_variables'] = ''

        self.settings_types['aero_nodes_isurf'] = 'list(int)'
        self.settings_default['aero_nodes_isurf'] = np.array([0])
        self.settings_types['aero_nodes_im'] = 'list(int)'
        self.settings_default['aero_nodes_im'] = np.array([0])
        self.settings_types['aero_nodes_in'] = 'list(int)'
        self.settings_default['aero_nodes_in'] = np.array([0])

        self.settings = None
        self.data = None
        self.dir = 'output/'

    def initialise(self, data, custom_settings=None):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        self.dir =   self.data.case_route + 'output/' + self.data.case_name + '/' + 'WriteVariablesTime/'
        if not os.path.isdir(self.dir):
            os.makedirs(self.dir)

        # Check inputs
        if not ((len(self.settings['aero_panels_isurf']) == len(self.settings['aero_panels_im'])) and (len(self.settings['aero_panels_isurf']) == len(self.settings['aero_panels_in']))):
            print("ERROR: aero_panels should be defined as [i_surf,i_m,i_n]")
        if not ((len(self.settings['aero_nodes_isurf']) == len(self.settings['aero_nodes_im'])) and (len(self.settings['aero_nodes_isurf']) == len(self.settings['aero_nodes_in']))):
            print("ERROR: aero_nodes should be defined as [i_surf,i_m,i_n]")

    def run(self, online=False):

    # settings['WriteVariablesTime'] = {'delimiter': ' ',
    #                                   'FoR_varibles': ['GFoR_pos', 'GFoR_vel', 'GFoR_acc'],
    #                                   'FoR_number': [0,1],
    #                                   'structure_variables': ['AFoR_steady_forces', 'AFoR_unsteady_forces','AFoR_position'],
    #                                   'structure_nodes': [0,-1],
    #                                   'aero_panels_variables': ['gamma', 'norm_gamma', 'norm_gamma_star'],
    #                                   'aero_panels_isurf': [0,1,2],
    #                                   'aero_panels_im': [1,1,1],
    #                                   'aero_panels_in': [-2,-2,-2],
    #                                   'aero_nodes_variables': ['GFoR_steady_force', 'GFoR_unsteady_force'],
    #                                   'aero_nodes_isurf': [0,1,2],
    #                                   'aero_nodes_im': [1,1,1],
    #                                   'aero_nodes_in': [-2,-2,-2]}

        # FoR variables
        if 'FoR_number' in self.settings:
            pass
        else:
            self.settings['FoR_number'] = np.array([0], dtype=int)

        for ivariable in range(len(self.settings['FoR_variables'])):
            for ifor in range(len(self.settings['FoR_number'])):
                filename = self.dir + "FoR_" + self.settings['FoR_number'][ifor] + "_" + self.settings['FoR_variables'][ivariable] + ".dat"
                fid = open(filename,"a")

                if (self.settings['FoR_variables'][ivariable] == 'GFoR_pos'):
                    self.write_nparray_to_file(fid, self.data.ts, self.data.structure.timestep_info[-1].mb_FoR_pos[ifor,:], self.settings['delimiter'])
                elif (self.settings['FoR_variables'][ivariable] == 'GFoR_vel'):
                    self.write_nparray_to_file(fid, self.data.ts, self.data.structure.timestep_info[-1].mb_FoR_vel[ifor,:], self.settings['delimiter'])
                elif (self.settings['FoR_variables'][ivariable] == 'GFoR_acc'):
                    self.write_nparray_to_file(fid, self.data.ts, self.data.structure.timestep_info[-1].mb_FoR_acc[ifor,:], self.settings['delimiter'])
                elif (self.settings['FoR_variables'][ivariable] == 'GFoR_quat'):
                    self.write_nparray_to_file(fid, self.data.ts, self.data.structure.timestep_info[-1].mb_quat[ifor,:], self.settings['delimiter'])
                elif (self.settings['FoR_variables'][ivariable] == 'AFoR_pos'):
                    self.write_nparray_to_file(fid, self.data.ts, self.data.structure.timestep_info[-1].for_pos, self.settings['delimiter'])
                elif (self.settings['FoR_variables'][ivariable] == 'AFoR_vel'):
                    self.write_nparray_to_file(fid, self.data.ts, self.data.structure.timestep_info[-1].for_vel, self.settings['delimiter'])
                elif (self.settings['FoR_variables'][ivariable] == 'AFoR_acc'):
                    self.write_nparray_to_file(fid, self.data.ts, self.data.structure.timestep_info[-1].for_acc, self.settings['delimiter'])
                elif (self.settings['FoR_variables'][ivariable] == 'AFoR_quat'):
                    self.write_nparray_to_file(fid, self.data.ts, self.data.structure.timestep_info[-1].quat, self.settings['delimiter'])
                else:
                    print("Unrecognized " + self.settings['FoR_variables'][ivariable] + " variable")

                fid.close()

        # Structure variables at nodes
        for ivariable in range(len(self.settings['structure_variables'])):
            for inode in range(len(self.settings['structure_nodes'])):
                node = self.settings['structure_nodes'][inode]
                filename = self.dir + "struct_" + self.settings['structure_variables'][ivariable] + "_node" + str(node) + ".dat"
                fid = open(filename,"a")

                if (self.settings['structure_variables'][ivariable] == 'AFoR_steady_forces'):
                    self.write_nparray_to_file(fid, self.data.ts, self.data.structure.timestep_info[-1].steady_applied_forces[node,:], self.settings['delimiter'])
                elif (self.settings['structure_variables'][ivariable] == 'AFoR_unsteady_forces'):
                    self.write_nparray_to_file(fid, self.data.ts, self.data.structure.timestep_info[-1].unsteady_applied_forces[node,:], self.settings['delimiter'])
                elif (self.settings['structure_variables'][ivariable] == 'AFoR_position'):
                    self.write_nparray_to_file(fid, self.data.ts, self.data.structure.timestep_info[-1].pos[node,:], self.settings['delimiter'])
                elif (self.settings['structure_variables'][ivariable] == 'AFoR_velocity'):
                    self.write_nparray_to_file(fid, self.data.ts, self.data.structure.timestep_info[-1].pos_dot[node,:], self.settings['delimiter'])
                else:
                    print("Unrecognized " + self.settings['structure_variables'][ivariable] + " variable")

                fid.close()

        # Aerodynamic variables at panels
        for ivariable in range(len(self.settings['aero_panels_variables'])):
            for ipanel in range(len(self.settings['aero_panels_isurf'])):
                i_surf = self.settings['aero_panels_isurf'][ipanel]
                i_m = self.settings['aero_panels_im'][ipanel]
                i_n = self.settings['aero_panels_in'][ipanel]

                filename = self.dir + "aero_" + self.settings['aero_panels_variables'][ivariable] + "_panel" + "_isurf" + str(i_surf) + "_im"+ str(i_m) + "_in"+ str(i_n) + ".dat"
                fid = open(filename,"a")

                if (self.settings['aero_panels_variables'][ivariable] == 'gamma'):
                    self.write_value_to_file(fid, self.data.ts, self.data.aero.timestep_info[-1].gamma[i_surf][i_m,i_n], self.settings['delimiter'])
                elif (self.settings['aero_panels_variables'][ivariable] == 'norm_gamma'):
                    self.write_value_to_file(fid, self.data.ts, np.linalg.norm(self.data.aero.timestep_info[-1].gamma), self.settings['delimiter'])
                elif (self.settings['aero_panels_variables'][ivariable] == 'norm_gamma_star'):
                    self.write_value_to_file(fid, self.data.ts, np.linalg.norm(self.data.aero.timestep_info[-1].gamma_star), self.settings['delimiter'])
                else:
                    print("Unrecognized " + self.settings['aero_panels_variables'][ivariable] + " variable")

                fid.close()

        # Aerodynamic variables at nodes
        for ivariable in range(len(self.settings['aero_nodes_variables'])):
            for inode in range(len(self.settings['aero_nodes_isurf'])):
                i_surf = self.settings['aero_nodes_isurf'][inode]
                i_m = self.settings['aero_nodes_im'][inode]
                i_n = self.settings['aero_nodes_in'][inode]

                filename = self.dir + "aero_" + self.settings['aero_nodes_variables'][ivariable] + "_node" + "_isurf" + str(i_surf) + "_im"+ str(i_m) + "_in"+ str(i_n) + ".dat"
                fid = open(filename,"a")

                if (self.settings['aero_nodes_variables'][ivariable] == 'GFoR_steady_force'):
                    self.write_nparray_to_file(fid, self.data.ts, self.data.aero.timestep_info[-1].forces[i_surf][:,i_m,i_n], self.settings['delimiter'])
                elif (self.settings['aero_nodes_variables'][ivariable] == 'GFoR_unsteady_force'):
                    self.write_nparray_to_file(fid, self.data.ts, self.data.aero.timestep_info[-1].dynamic_forces[i_surf][:,i_m,i_n], self.settings['delimiter'])
                else:
                    print("Unrecognized " + self.settings['aero_nodes_variables'][ivariable] + " variable")

                fid.close()

        return self.data

    def write_nparray_to_file(self, fid, ts, nparray, delimiter):

        fid.write("%d%s" % (ts,delimiter))
        for idim in range(np.shape(nparray)[0]):
            fid.write("%e%s" % (nparray[idim],delimiter))

        fid.write("\n")

    def write_value_to_file(self, fid, ts, value, delimiter):

        fid.write("%d%s%e\n" % (ts,delimiter,value))
