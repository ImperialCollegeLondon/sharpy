import os

import numpy as np
from tvtk.api import tvtk, write_data

import sharpy.utils.cout_utils as cout
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra


@solver
class BeamPlot(BaseSolver):
    solver_id = 'BeamPlot'

    def __init__(self):
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['folder'] = 'str'
        self.settings_default['folder'] = './output'

        self.settings_types['include_rbm'] = 'bool'
        self.settings_default['include_rbm'] = True

        self.settings_types['include_applied_forces'] = 'bool'
        self.settings_default['include_applied_forces'] = True

        self.settings_types['include_applied_moments'] = 'bool'
        self.settings_default['include_applied_moments'] = True

        self.settings_types['name_prefix'] = 'str'
        self.settings_default['name_prefix'] = ''

        self.settings_types['output_rbm'] = 'bool'
        self.settings_default['output_rbm'] = True

        self.settings = None
        self.data = None

        self.folder = ''
        self.filename = ''

    def initialise(self, data):
        self.data = data
        self.settings = data.settings[self.solver_id]
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)
        # create folder for containing files if necessary
        if not os.path.exists(self.settings['folder']):
            os.makedirs(self.settings['folder'])
        self.folder = self.settings['folder'] + '/' + self.data.settings['SHARPy']['case'] + '/beam/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.filename = (self.folder +
                         self.settings['name_prefix'] +
                         'beam_' +
                         self.data.settings['SHARPy']['case'])

    def run(self):
        self.plot()
        self.write()
        cout.cout_wrap('...Finished', 1)
        return self.data

    def write(self):
        if self.settings['output_rbm']:
            filename = self.filename + '_rbm_acc.csv'
            timesteps = len(self.data.structure.timestep_info)
            temp_matrix = np.zeros((timesteps, 6))
            for it in range(timesteps):
                temp_matrix[it, :] = self.data.structure.timestep_info[it].for_acc

            np.savetxt(filename, temp_matrix, delimiter=',')




    def plot(self):
        for it in range(len(self.data.structure.timestep_info)):
            it_filename = (self.filename +
                           '%06u' % it)
            num_nodes = self.data.structure.num_node
            num_elem = self.data.structure.num_elem

            coords = np.zeros((num_nodes, 3))
            conn = np.zeros((num_elem, 3), dtype=int)
            node_id = np.zeros((num_nodes,), dtype=int)
            elem_id = np.zeros((num_elem,), dtype=int)
            local_x = np.zeros((num_nodes, 3))
            local_y = np.zeros((num_nodes, 3))
            local_z = np.zeros((num_nodes, 3))
            # output_loads = True
            # try:
            #     self.data.beam.timestep_info[it].loads
            # except AttributeError:
            #     output_loads = False

            # if output_loads:
            #     gamma = np.zeros((num_elem, 3))
            #     kappa = np.zeros((num_elem, 3))

            # if self.settings['applied_forces']:

            # if self.settings['frame'] == 'inertial':
            #     try:
            #         self.aero2inertial = self.data.grid.inertial2aero.T
            #     except AttributeError:
            #         self.aero2inertial = np.eye(3)
            #         cout.cout_wrap('BeamPlot: No inertial2aero information, output will be in body FoR', 0)
            #
            # else:
            #     self.aero2inertial = np.eye(3)

            app_forces = np.zeros((num_nodes, 3))
            app_moment = np.zeros((num_nodes, 3))

            # aero2inertial rotation
            aero2inertial = self.data.structure.timestep_info[it].cga()

            # coordinates of corners
            coords = self.data.structure.timestep_info[it].glob_pos(include_rbm=self.settings['include_rbm'])

            for i_node in range(num_nodes):
                i_elem = self.data.structure.node_master_elem[i_node, 0]
                i_local_node = self.data.structure.node_master_elem[i_node, 1]
                node_id[i_node] = i_node

                v1 = np.array([1., 0, 0])
                v2 = np.array([0., 1, 0])
                v3 = np.array([0., 0, 1])
                cab = algebra.crv2rot(
                    self.data.structure.timestep_info[it].psi[i_elem, i_local_node, :])
                local_x[i_node, :] = np.dot(aero2inertial, np.dot(cab, v1))
                local_y[i_node, :] = np.dot(aero2inertial, np.dot(cab, v2))
                local_z[i_node, :] = np.dot(aero2inertial, np.dot(cab, v3))

                # applied forces
                cab = algebra.crv2rot(self.data.structure.timestep_info[it].psi[i_elem, i_local_node, :])
                app_forces[i_node, :] = np.dot(aero2inertial,
                                               np.dot(cab,
                                                      self.data.structure.timestep_info[it].steady_applied_forces[i_node, 0:3]+
                                                      self.data.structure.timestep_info[it].unsteady_applied_forces[i_node, 0:3]))
                app_moment[i_node, :] = np.dot(aero2inertial,
                                               np.dot(cab,
                                                      self.data.structure.timestep_info[it].steady_applied_forces[i_node, 3:6]+
                                                      self.data.structure.timestep_info[it].unsteady_applied_forces[i_node, 3:6]))
                # if not it == 0:
                #     try:
                #         unsteady_app_forces[i_node, :] = np.dot(aero2inertial,
                #                                                 np.dot(cab,
                #                                                        self.data.structure.dynamic_input[it - 1]['dynamic_forces'][i_node, 0:3]))
                #     except IndexError:
                #         pass

            for i_elem in range(num_elem):
                conn[i_elem, :] = self.data.structure.elements[i_elem].reordered_global_connectivities
                elem_id[i_elem] = i_elem
                # if output_loads:
                #     gamma[i_elem, :] = self.data.structure.timestep_info[it].loads[i_elem, 0:3]
                #     kappa[i_elem, :] = self.data.structure.timestep_info[it].loads[i_elem, 3:6]

            ug = tvtk.UnstructuredGrid(points=coords)
            ug.set_cells(tvtk.Line().cell_type, conn)
            ug.cell_data.scalars = elem_id
            ug.cell_data.scalars.name = 'elem_id'
            # if output_loads:
            #     ug.cell_data.add_array(gamma, 'vector')
            #     ug.cell_data.get_array(1).name = 'gamma'
            #     ug.cell_data.add_array(kappa, 'vector')
            #     ug.cell_data.get_array(2).name = 'kappa'
            ug.point_data.scalars = node_id
            ug.point_data.scalars.name = 'node_id'
            point_vector_counter = 1
            ug.point_data.add_array(local_x, 'vector')
            ug.point_data.get_array(point_vector_counter).name = 'local_x'
            point_vector_counter += 1
            ug.point_data.add_array(local_y, 'vector')
            ug.point_data.get_array(point_vector_counter).name = 'local_y'
            point_vector_counter += 1
            ug.point_data.add_array(local_z, 'vector')
            ug.point_data.get_array(point_vector_counter).name = 'local_z'
            if self.settings['include_applied_forces']:
                point_vector_counter += 1
                ug.point_data.add_array(app_forces, 'vector')
                ug.point_data.get_array(point_vector_counter).name = 'app_forces'
            if self.settings['include_applied_moments']:
                point_vector_counter += 1
                ug.point_data.add_array(app_moment, 'vector')
                ug.point_data.get_array(point_vector_counter).name = 'app_moments'

            write_data(ug, it_filename)


