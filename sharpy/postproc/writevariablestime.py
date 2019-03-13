import os

import numpy as np
from tvtk.api import tvtk, write_data

import sharpy.utils.cout_utils as cout
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra
import sharpy.structure.utils.xbeamlib as xbeamlib


@solver
class WriteVariablesTime(BaseSolver):
    solver_id = 'WriteVariablesTime'

    def __init__(self):
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['delimiter'] = 'str'
        self.settings_default['delimiter'] = ' '

        self.settings_types['FoR_variables'] = 'list(str)'
        self.settings_default['FoR_variables'] = ['']

        self.settings_types['FoR_number'] = 'list(int)'
        self.settings_default['FoR_number'] = np.array([0], dtype=int)

        self.settings_types['structure_variables'] = 'list(str)'
        self.settings_default['structure_variables'] = ['']

        self.settings_types['structure_nodes'] = 'list(int)'
        self.settings_default['structure_nodes'] = np.array([-1])

        self.settings_types['aero_panels_variables'] = 'list(str)'
        self.settings_default['aero_panels_variables'] = ['']

        self.settings_types['aero_panels_isurf'] = 'list(int)'
        self.settings_default['aero_panels_isurf'] = np.array([0])
        self.settings_types['aero_panels_im'] = 'list(int)'
        self.settings_default['aero_panels_im'] = np.array([0])
        self.settings_types['aero_panels_in'] = 'list(int)'
        self.settings_default['aero_panels_in'] = np.array([0])

        self.settings_types['aero_nodes_variables'] = 'list(str)'
        self.settings_default['aero_nodes_variables'] = ['']

        self.settings_types['aero_nodes_isurf'] = 'list(int)'
        self.settings_default['aero_nodes_isurf'] = np.array([0])
        self.settings_types['aero_nodes_im'] = 'list(int)'
        self.settings_default['aero_nodes_im'] = np.array([0])
        self.settings_types['aero_nodes_in'] = 'list(int)'
        self.settings_default['aero_nodes_in'] = np.array([0])

        self.settings_types['cleanup_old_solution'] = 'bool'
        self.settings_default['cleanup_old_solution'] = 'false'


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

        if self.settings['cleanup_old_solution']:
            for ivariable in range(len(self.settings['FoR_variables'])):
                if self.settings['FoR_variables'][ivariable] == '':
                    continue
                for ifor in range(len(self.settings['FoR_number'])):
                    filename = self.dir + "FoR_" + '%02d' % self.settings['FoR_number'][ifor] + "_" + self.settings['FoR_variables'][ivariable] + ".dat"
                    try:
                        os.remove(filename)
                    except FileNotFoundError:
                        pass

            # Structure variables at nodes
            for ivariable in range(len(self.settings['structure_variables'])):
                if self.settings['structure_variables'][ivariable] == '':
                    continue
                for inode in range(len(self.settings['structure_nodes'])):
                    node = self.settings['structure_nodes'][inode]
                    filename = self.dir + "struct_" + self.settings['structure_variables'][ivariable] + "_node" + str(node) + ".dat"
                    try:
                        os.remove(filename)
                    except FileNotFoundError:
                        pass

            # Aerodynamic variables at panels
            for ivariable in range(len(self.settings['aero_panels_variables'])):
                if self.settings['aero_panels_variables'][ivariable] == '':
                    continue
                for ipanel in range(len(self.settings['aero_panels_isurf'])):
                    i_surf = self.settings['aero_panels_isurf'][ipanel]
                    i_m = self.settings['aero_panels_im'][ipanel]
                    i_n = self.settings['aero_panels_in'][ipanel]

                    filename = self.dir + "aero_" + self.settings['aero_panels_variables'][ivariable] + "_panel" + "_isurf" + str(i_surf) + "_im"+ str(i_m) + "_in"+ str(i_n) + ".dat"
                    try:
                        os.remove(filename)
                    except FileNotFoundError:
                        pass

            # Aerodynamic variables at nodes
            for ivariable in range(len(self.settings['aero_nodes_variables'])):
                if self.settings['aero_nodes_variables'][ivariable] == '':
                    continue
                for inode in range(len(self.settings['aero_nodes_isurf'])):
                    i_surf = self.settings['aero_nodes_isurf'][inode]
                    i_m = self.settings['aero_nodes_im'][inode]
                    i_n = self.settings['aero_nodes_in'][inode]

                    filename = self.dir + "aero_" + self.settings['aero_nodes_variables'][ivariable] + "_node" + "_isurf" + str(i_surf) + "_im"+ str(i_m) + "_in"+ str(i_n) + ".dat"
                    try:
                        os.remove(filename)
                    except FileNotFoundError:
                        pass

    def run(self, online=False):

    # settings['WriteVariablesTime'] = {'delimiter': ' ',
    #                                   'FoR_variables': ['GFoR_pos', 'GFoR_vel', 'GFoR_acc'],
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
                filename = self.dir + "FoR_" + '%02d' % self.settings['FoR_number'][ifor] + "_" + self.settings['FoR_variables'][ivariable] + ".dat"
                fid = open(filename,"a")

                var = getattr(self.data.structure.timestep_info[-1], self.settings['FoR_variables'][ivariable])
                rows, cols = var.shape
                if ((cols == 1) and (rows == 1)):
                    self.write_value_to_file(fid, self.data.ts, var, self.settings['delimiter'])
                elif ((cols > 1) and (rows == 1)):
                    self.write_nparray_to_file(fid, self.data.ts, var, self.settings['delimiter'])
                elif ((cols == 1) and (rows >= 1)):
                    self.write_value_to_file(fid, self.data.ts, var[ifor], self.settings['delimiter'])
                else:
                    self.write_nparray_to_file(fid, self.data.ts, var[ifor,:], self.settings['delimiter'])
                fid.close()

        # Structure variables at nodes
        for ivariable in range(len(self.settings['structure_variables'])):
            for inode in range(len(self.settings['structure_nodes'])):
                node = self.settings['structure_nodes'][inode]
                filename = self.dir + "struct_" + self.settings['structure_variables'][ivariable] + "_node" + str(node) + ".dat"
                fid = open(filename,"a")

                var = getattr(self.data.structure.timestep_info[-1], self.settings['structure_variables'][ivariable])
                num_indices = len(var.shape)
                if num_indices == 2:
                    self.write_nparray_to_file(fid, self.data.ts, var[node,:], self.settings['delimiter'])
                elif num_indices == 3:
                    ielem, inode_in_elem = self.data.structure.node_master_elem[node]
                    self.write_nparray_to_file(fid, self.data.ts, var[ielem,inode_in_elem,:], self.settings['delimiter'])

                fid.close()

        # Aerodynamic variables at panels
        for ivariable in range(len(self.settings['aero_panels_variables'])):
            for ipanel in range(len(self.settings['aero_panels_isurf'])):
                i_surf = self.settings['aero_panels_isurf'][ipanel]
                i_m = self.settings['aero_panels_im'][ipanel]
                i_n = self.settings['aero_panels_in'][ipanel]

                filename = self.dir + "aero_" + self.settings['aero_panels_variables'][ivariable] + "_panel" + "_isurf" + str(i_surf) + "_im"+ str(i_m) + "_in"+ str(i_n) + ".dat"
                fid = open(filename,"a")

                var = getattr(self.data.aero.timestep_info[-1], self.settings['aero_panels_variables'][ivariable])
                self.write_value_to_file(fid, self.data.ts, var.gamma[i_surf][i_m,i_n], self.settings['delimiter'])

                fid.close()

        # Aerodynamic variables at nodes
        for ivariable in range(len(self.settings['aero_nodes_variables'])):
            for inode in range(len(self.settings['aero_nodes_isurf'])):
                i_surf = self.settings['aero_nodes_isurf'][inode]
                i_m = self.settings['aero_nodes_im'][inode]
                i_n = self.settings['aero_nodes_in'][inode]

                filename = self.dir + "aero_" + self.settings['aero_nodes_variables'][ivariable] + "_node" + "_isurf" + str(i_surf) + "_im"+ str(i_m) + "_in"+ str(i_n) + ".dat"
                fid = open(filename,"a")

                var = getattr(self.data.aero.timestep_info[-1], self.settings['aero_nodes_variables'][ivariable])
                self.write_nparray_to_file(fid, self.data.ts, var[i_surf][:,i_m,i_n], self.settings['delimiter'])

                fid.close()

        return self.data

    def write_nparray_to_file(self, fid, ts, nparray, delimiter):

        fid.write("%d%s" % (ts,delimiter))
        for idim in range(np.shape(nparray)[0]):
            fid.write("%e%s" % (nparray[idim],delimiter))

        fid.write("\n")

    def write_value_to_file(self, fid, ts, value, delimiter):

        fid.write("%d%s%e\n" % (ts,delimiter,value))
