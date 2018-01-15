import os
import numpy as np

import sharpy.utils.cout_utils as cout
from sharpy.utils.settings import str2bool
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings


@solver
class BeamCsvOutput(BaseSolver):
    solver_id = 'BeamCsvOutput'

    def __init__(self):
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['folder'] = 'str'
        self.settings_default['folder'] = './output'

        self.settings_types['output_pos'] = 'bool'
        self.settings_default['output_pos'] = True

        self.settings_types['output_psi'] = 'bool'
        self.settings_default['output_psi'] = False

        self.settings_types['output_for_pos'] = 'bool'
        self.settings_default['output_for_pos'] = False

        self.settings_types['output_glob_pos'] = 'bool'
        self.settings_default['output_glob_pos'] = False

        self.settings_types['screen_output'] = 'bool'
        self.settings_default['screen_output'] = False

        self.settings_types['name_prefix'] = 'str'
        self.settings_default['name_prefix'] = ''

        self.data = None
        self.it = 0
        self.settings = None
        self.folder = ''
        self.filename = ''

    def initialise(self, data):
        self.data = data
        self.it = self.data.ts
        self.settings = data.settings[self.solver_id]
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

    def run(self):
        # create folder for containing files if necessary
        if not os.path.exists(self.settings['folder']):
            os.makedirs(self.settings['folder'])
        self.folder = self.settings['folder'] + '/' + self.data.settings['SHARPy']['case'] + '/beam/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.filename = (self.folder +
                         'beam_' +
                         self.data.settings['SHARPy']['case'] +
                         '_' +
                         self.settings['name_prefix'])
        # for self.it in range(self.data.ts):
        self.text()
        self.print_info()
        cout.cout_wrap('...Finished', 1)
        return self.data

    def text(self):
        for it in range(self.data.ts + 1):
            if self.settings['output_pos']:
                it_filename = (self.filename +
                               '%06u' % it +
                               '.csv')
                # write file
                np.savetxt(it_filename, self.data.structure.timestep_info[it].pos)

            if self.settings['output_for_pos']:
                it_filename = (self.filename +
                               '%06u' % it +
                               '.for_pos.csv')
                # write file
                np.savetxt(it_filename, self.data.structure.timestep_info[it].for_pos)

            if self.settings['output_psi']:
                it_filename = (self.filename + 'crv_' +
                               '%06u' % it +
                               '.csv')
                crv_matrix = np.zeros((self.data.structure.num_node, 3))
                for i_node in range(self.data.structure.num_node):
                    i_elem, i_local_node = self.data.structure.node_master_elem[i_node, :]
                    crv_matrix[i_node, :] = self.data.structure.timestep_info[it].psi[i_elem, i_local_node, :]
                # write file
                np.savetxt(it_filename, crv_matrix)

            if self.settings['output_glob_pos']:
                it_filename = (self.filename + 'glob_' +
                               '%06u' % it +
                               '.csv')
                # write file
                np.savetxt(it_filename, self.data.structure.timestep_info[it].glob_pos(include_rbm=True))

    def print_info(self):
        if self.settings['screen_output']:
            cout.cout_wrap('Pos_def', 1)
            cout.cout_wrap(str(self.data.structure.timestep_info[-1].pos), 1)
            cout.cout_wrap('', 1)

            cout.cout_wrap('Psi_def', 1)
            crv_matrix = np.zeros((self.data.structure.num_node, 3))
            for i_node in range(self.data.structure.num_node):
                i_elem, i_local_node = self.data.structure.node_master_elem[i_node, :]
                crv_matrix[i_node, :] = self.data.structure.timestep_info[-1].psi[i_elem, i_local_node, :]
            cout.cout_wrap(str(crv_matrix), 1)

