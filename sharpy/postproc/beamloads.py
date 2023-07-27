import os
import numpy as np
import os
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings_utils
import sharpy.structure.utils.xbeamlib as xbeamlib


@solver
class BeamLoads(BaseSolver):
    """
    Writes to file the total loads acting on the beam elements

    """
    solver_id = 'BeamLoads'
    solver_classification = 'post-processor'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['csv_output'] = 'bool'
    settings_default['csv_output'] = False
    settings_description['csv_output'] = 'Write ``csv`` file with results'

    settings_types['output_file_name'] = 'str'
    settings_default['output_file_name'] = 'beam_loads'
    settings_description['output_file_name'] = 'Output file name'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):

        self.settings = None
        self.data = None

        self.folder = None
        self.caller = None

    def initialise(self, data, custom_settings=None, caller=None, restart=False):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings_utils.to_custom_types(self.settings, self.settings_types, self.settings_default)
        self.caller = caller

        self.folder = data.output_folder + '/beam/'
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)

    def run(self, **kwargs):

        online = settings_utils.set_value_or_default(kwargs, 'online', False)

        self.calculate_loads(online)
        if self.settings['csv_output']:
            self.print_loads(online)
        return self.data

    def print_loads(self, online):
        if online:
            it = len(self.data.structure.timestep_info) - 1
            n_elem = self.data.structure.timestep_info[it].psi.shape[0]
            data = np.zeros((n_elem, 10))
            # coords
            data[:, 0:3] = self.data.structure.timestep_info[it].postproc_cell['coords_a']
            header = 'x_a, y_a, z_a, '
            # beam number
            data[:, 3] = self.data.structure.beam_number
            header += 'beam_number, '
            # loads_0
            data[:, 4:10] = self.data.structure.timestep_info[it].postproc_cell['loads'][:, :]
            header += 'Fx, Fy, Fz, Mx, My, Mz'

            filename = self.folder
            filename += self.settings['output_file_name'] + '_' + '{0}'.format(it)
            filename += '.csv'
            np.savetxt(filename, data, delimiter=',', header=header)
        else:
            for it in range(len(self.data.structure.timestep_info)):
                it = len(self.data.structure.timestep_info) - 1
                n_elem = self.data.structure.timestep_info[it].num_elem
                data = np.zeros((n_elem, 10))
                # coords
                data[:, 0:3] = self.data.structure.timestep_info[it].postproc_cell['coords_a']
                header = 'x_a, y_a, z_a, '
                # beam number
                data[:, 3] = self.data.structure.beam_number
                header += 'beam_number, '
                # loads_0
                data[:, 4:10] = self.data.structure.timestep_info[it].postproc_cell['loads'][:, :]
                header += 'Fx, Fy, Fz, Mx, My, Mz'

                filename = self.folder
                filename += self.settings['output_file_name'] + '_' + '{0}'.format(it)
                filename += '.csv'
                np.savetxt(filename, data, delimiter=',', header=header)

    def calculate_loads(self, online):
        if online:
            it = -1
            timestep_add_loads(self.data.structure, self.data.structure.timestep_info[it])
            self.calculate_coords_a(self.data.structure.timestep_info[it])
        else:
            for it in range(len(self.data.structure.timestep_info)):
                timestep_add_loads(self.data.structure, self.data.structure.timestep_info[it])
                self.calculate_coords_a(self.data.structure.timestep_info[it])

    def calculate_coords_a(self, timestep_info):
        timestep_info.postproc_cell['coords_a'] = np.zeros((timestep_info.num_elem, 3))
        for ielem in range(timestep_info.num_elem):
            iglobal_node = self.data.structure.connectivities[ielem, 2]
            timestep_info.postproc_cell['coords_a'][ielem, :] = timestep_info.pos[iglobal_node, :]


def timestep_add_loads(structure, timestep):
    timestep.postproc_cell['strain'], timestep.postproc_cell['loads'] = \
        xbeamlib.cbeam3_loads(structure, timestep)

    # def calculate_loads(self):
    #     # initial (ini) loads
    #     tstep = self.data.structure.ini_info
    #     pos = tstep.pos
    #     psi = tstep.psi
    #
    #     tstep.postproc_cell['gamma'] = np.zeros((self.data.structure.num_elem, 3))
    #     tstep.postproc_cell['kappa'] = np.zeros((self.data.structure.num_elem, 3))
    #
    #     for ielem in range(self.data.structure.num_elem):
    #         crv = 0.5*(psi[ielem, 1, :] + psi[ielem, 0, :])
    #         cba = algebra.crv2rotation(crv).T
    #         tan = algebra.crv2tan(crv)
    #
    #         inode0 = self.data.structure.elements[ielem].global_connectivities[0]
    #         inode1 = self.data.structure.elements[ielem].global_connectivities[1]
    #         tstep.postproc_cell['gamma'][ielem, :] = (
    #                 np.dot(cba,
    #                        pos[inode1, :] - pos[inode0, :]) /
    #                 np.linalg.norm(pos[inode1, :] - pos[inode0, :]))
    #         tstep.postproc_cell['kappa'][ielem, :] = (
    #                 np.dot(tan,
    #                        psi[ielem, 1, :] - psi[ielem, 0, :]) /
    #                 np.linalg.norm(pos[inode1, :] - pos[inode0, :]))
    #
    #     # time-dependant loads
    #     for it in range(len(self.data.structure.timestep_info)):
    #         tstep = self.data.structure.timestep_info[it]
    #         pos = tstep.pos
    #         psi = tstep.psi
    #
    #         tstep.postproc_cell['gamma'] = np.zeros((self.data.structure.num_elem, 3))
    #         tstep.postproc_cell['kappa'] = np.zeros((self.data.structure.num_elem, 3))
    #         tstep.postproc_cell['strain'] = np.zeros((self.data.structure.num_elem, 6))
    #         tstep.postproc_cell['loads'] = np.zeros((self.data.structure.num_elem, 6))
    #
    #         for ielem in range(self.data.structure.num_elem):
    #             crv = 0.5*(psi[ielem, 2, :] + psi[ielem, 0, :])
    #             cba = algebra.crv2rotation(crv).T
    #             tan = algebra.crv2tan(crv)
    #
    #             inode0 = self.data.structure.elements[ielem].global_connectivities[0]
    #             inode1 = self.data.structure.elements[ielem].global_connectivities[1]
    #             tstep.postproc_cell['gamma'][ielem, :] = (
    #                     np.dot(cba,
    #                            pos[inode1, :] - pos[inode0, :]) /
    #                     np.linalg.norm(pos[inode1, :] - pos[inode0, :]))
    #             tstep.postproc_cell['kappa'][ielem, :] = (
    #                     np.dot(tan,
    #                            psi[ielem, 1, :] - psi[ielem, 0, :]) /
    #                     np.linalg.norm(pos[inode1, :] - pos[inode0, :]))
    #
    #             tstep.postproc_cell['strain'][ielem, 0:3] = (
    #                     tstep.postproc_cell['gamma'][ielem, :]
    #                     -
    #                     self.data.structure.ini_info.postproc_cell['gamma'][ielem, :])
    #             tstep.postproc_cell['strain'][ielem, 3:6] = (
    #                     tstep.postproc_cell['kappa'][ielem, :]
    #                     -
    #                     self.data.structure.ini_info.postproc_cell['kappa'][ielem, :])
    #             tstep.postproc_cell['loads'][ielem, :] = np.dot(
    #                 self.data.structure.stiffness_db[self.data.structure.elements[ielem].stiff_index, :, :],
    #                 tstep.postproc_cell['strain'][ielem, :])

    # def calculate_loads(self):
    #     order = [0, 2, 1]
    #     # initial (ini) loads
    #     tstep = self.data.structure.ini_info
    #     pos = tstep.pos
    #     psi = tstep.psi
    #
    #     gamma0 = np.zeros((self.data.structure.num_elem, 2, 3))
    #     kappa0 = np.zeros((self.data.structure.num_elem, 2, 3))
    #     counter = np.zeros((self.data.structure.num_node,), dtype=int)
    #
    #     for ielem in range(self.data.structure.num_elem):
    #         for isegment in range(self.data.structure.elements[ielem].n_nodes - 1):
    #             i_local_node0 = order[isegment]
    #             i_local_node1 = order[isegment + 1]
    #             crv = 0.5*(psi[ielem, i_local_node1, :] + psi[ielem, i_local_node0, :])
    #
    #             cba = algebra.crv2rotation(crv).T
    #             tan = algebra.crv2tan(crv)
    #
    #             inode0 = self.data.structure.elements[ielem].global_connectivities[i_local_node0]
    #             inode1 = self.data.structure.elements[ielem].global_connectivities[i_local_node1]
    #
    #             counter[inode0] += 1
    #             counter[inode1] += 1
    #
    #             print('----')
    #             print(ielem)
    #             print(isegment)
    #             print(inode0, inode1)
    #             print(counter[inode0], counter[inode1])
    #             print('----')
    #
    #             gamma0[ielem, isegment, :] = (
    #                     np.dot(cba,
    #                            pos[inode1, :] - pos[inode0, :]) /
    #                     np.linalg.norm(pos[inode1, :] - pos[inode0, :]))
    #             kappa0[ielem, isegment, :] = (
    #                     np.dot(tan,
    #                            psi[ielem, i_local_node1, :] - psi[ielem, i_local_node0, :]) /
    #                     np.linalg.norm(pos[inode1, :] - pos[inode0, :]))
    #
    #     # time-dependant loads
    #     for it in range(len(self.data.structure.timestep_info)):
    #         tstep = self.data.structure.timestep_info[it]
    #         pos = tstep.pos
    #         psi = tstep.psi
    #
    #         gamma = np.zeros((self.data.structure.num_elem, 2, 3))
    #         kappa = np.zeros((self.data.structure.num_elem, 2, 3))
    #         # tstep.postproc_cell['strain'] = np.zeros((self.data.structure.num_elem, 6))
    #         tstep.postproc_node['loads'] = np.zeros((self.data.structure.num_node, 6))
    #         strain = np.zeros((self.data.structure.num_elem, 2, 6))
    #         for ielem in range(self.data.structure.num_elem):
    #             for isegment in range(self.data.structure.elements[ielem].n_nodes - 1):
    #                 i_local_node0 = order[isegment]
    #                 i_local_node1 = order[isegment + 1]
    #                 crv = 0.5*(psi[ielem, i_local_node1, :] + psi[ielem, i_local_node0, :])
    #
    #                 cba = algebra.crv2rotation(crv).T
    #                 tan = algebra.crv2tan(crv)
    #
    #                 inode0 = self.data.structure.elements[ielem].global_connectivities[i_local_node0]
    #                 inode1 = self.data.structure.elements[ielem].global_connectivities[i_local_node1]
    #
    #                 gamma[ielem, isegment, :] = (
    #                         np.dot(cba,
    #                                pos[inode1, :] - pos[inode0, :]) /
    #                         np.linalg.norm(pos[inode1, :] - pos[inode0, :]))
    #                 kappa[ielem, isegment, :] = (
    #                         np.dot(tan,
    #                                psi[ielem, i_local_node1, :] - psi[ielem, i_local_node0, :]) /
    #                         np.linalg.norm(pos[inode1, :] - pos[inode0, :]))
    #                 strain[ielem, isegment, 0:3] += (
    #                         gamma[ielem, isegment, :]
    #                         -
    #                         gamma0[ielem, isegment, :])
    #                 strain[ielem, isegment, 3:6] += (
    #                         kappa[ielem, isegment, :]
    #                         -
    #                         kappa0[ielem, isegment, :])
    #                 # it might be necessary to rotate the results so that the B frame is the
    #                 # Master FoR (so that all the loads -- intrinsically in material FoR -- can be
    #                 # added together at the nodes).
    #                 prerotate = np.eye(6)
    #                 posrotate = np.eye(6)
    #                 if not self.data.structure.node_master_elem[inode0, 0] == ielem:
    #                     cab2 = algebra.crv2rotation(psi[ielem, self.data.structure.node_master_elem[inode0, 1], :])
    #                     prerotate[0:3, 0:3] = np.dot(cab2.T, cba.T)
    #                     prerotate[3:6, 3:6] = np.dot(cab2.T, cba.T)
    #
    #                 tstep.postproc_node['loads'][inode0, :] = np.dot(prerotate,
    #                     np.dot(np.dot(
    #                     self.data.structure.stiffness_db[self.data.structure.elements[ielem].stiff_index, :, :],
    #                     strain[ielem, isegment, :])/counter[inode0], posrotate))
    #                 prerotate = np.eye(6)
    #                 posrotate = np.eye(6)
    #                 if not self.data.structure.node_master_elem[inode1, 0] == ielem:
    #                     cab2 = algebra.crv2rotation(psi[ielem, self.data.structure.node_master_elem[inode1, 1], :])
    #                     prerotate[0:3, 0:3] = np.dot(cab2.T, cba.T)
    #                     prerotate[3:6, 3:6] = np.dot(cab2.T, cba.T)
    #
    #                 tstep.postproc_node['loads'][inode1, :] = np.dot(prerotate,
    #                                                                  np.dot(np.dot(
    #                                                                      self.data.structure.stiffness_db[self.data.structure.elements[ielem].stiff_index, :, :],
    #                                                                      strain[ielem, isegment, :])/counter[inode1], posrotate))
    #                 # tstep.postproc_node['loads'][inode1, :] = np.dot(
    #                 #     self.data.structure.stiffness_db[self.data.structure.elements[ielem].stiff_index, :, :],
    #                 #     strain[ielem, isegment, :])/counter[inode1]
