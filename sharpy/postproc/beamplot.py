import sharpy.utils.cout_utils as cout
from sharpy.presharpy.utils.settings import str2bool
from sharpy.utils.solver_interface import solver, BaseSolver

from tvtk.api import tvtk, write_data
import numpy as np
import os


@solver
class BeamPlot(BaseSolver):
    solver_id = 'BeamPlot'
    solver_type = 'postproc'
    solver_unsteady = False

    def __init__(self):
        self.ts = 0  # steady solver
        pass

    def initialise(self, data):
        self.data = data
        self.it = data.beam.it
        self.settings = data.settings[self.solver_id]
        self.convert_settings()

    def run(self):
        # create folder for containing files if necessary
        if not os.path.exists(self.settings['route']):
            os.makedirs(self.settings['route'])
        self.plot()
        cout.cout_wrap('...Finished', 1)
        return self.data

    def convert_settings(self):
        try:
            self.settings['route'] = (self.settings['route'])
        except KeyError:
            cout.cout_wrap(self.solver_id + ': no location for figures defined, defaulting to ./output', 3)
            self.settings['route'] = './output'

        try:
            self.settings['applied_forces'] = str2bool(self.settings['applied_forces'])
        except KeyError:
            self.settings['applied_forces'] = False

    def plot(self):
        folder = self.settings['route'] + '/' + self.data.settings['SHARPy']['case'] + '/beam/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        for it in range(len(self.data.beam.timestep_info)):
            it_filename = (folder +
                           'beam_' +
                           self.data.settings['SHARPy']['case'] +
                           '_' +
                           '%06u' % it)
            num_nodes = self.data.beam.num_node
            num_elem = self.data.beam.num_elem

            coords = np.zeros((num_nodes, 3))
            conn = np.zeros((num_elem, 3), dtype=int)
            node_id = np.zeros((num_nodes,), dtype=int)
            elem_id = np.zeros((num_elem,), dtype=int)
            local_x = np.zeros((num_nodes, 3))
            local_y = np.zeros((num_nodes, 3))
            local_z = np.zeros((num_nodes, 3))
            if self.settings['applied_forces']:
                app_forces = np.zeros((num_nodes, 3))
            # coordinates of corners
            for i_node in range(num_nodes):
                if self.data.beam.timestep_info[it].with_rb:
                    coords[i_node, :] = self.data.beam.timestep_info[it].glob_pos_def[i_node, :]
                else:
                    if self.settings['frame'] == 'inertial':
                        coords[i_node, :] = np.dot(self.data.grid.inertial2aero.T, self.data.beam.timestep_info[it].pos_def[i_node, :])
                    else:
                        coords[i_node, :] = self.data.beam.timestep_info[it].pos_def[i_node, :]

            for i_node in range(num_nodes):
                i_elem = self.data.beam.node_master_elem[i_node, 0]
                i_local_node = self.data.beam.node_master_elem[i_node, 1]
                node_id[i_node] = i_node

                self.data.beam.update(it)

                v1, v2, v3 = self.data.beam.elements[i_elem].deformed_triad()
                if self.settings['frame'] == 'inertial':
                    local_x[i_node, :] = np.dot(self.data.grid.inertial2aero.T, v1[i_local_node, :])
                    local_y[i_node, :] = np.dot(self.data.grid.inertial2aero.T, v2[i_local_node, :])
                    local_z[i_node, :] = np.dot(self.data.grid.inertial2aero.T, v3[i_local_node, :])
                else:
                    local_x[i_node, :] = v1[i_local_node, :]
                    local_y[i_node, :] = v2[i_local_node, :]
                    local_z[i_node, :] = v3[i_local_node, :]

                # applied forces
                if self.settings['applied_forces']:
                    import sharpy.utils.algebra as algebra
                    Cab = algebra.crv2rot(self.data.beam.timestep_info[it].psi_def[i_elem, i_local_node, :])
                    app_forces[i_node, :] = np.dot(self.data.grid.inertial2aero.T,
                                                   np.dot(Cab,
                                                          self.data.beam.app_forces[i_node, 0:3]))

            for i_elem in range(num_elem):
                conn[i_elem, :] = self.data.beam.elements[i_elem].reordered_global_connectivities
                elem_id[i_elem] = i_elem

            ug = tvtk.UnstructuredGrid(points=coords)
            ug.set_cells(tvtk.Line().cell_type, conn)
            ug.cell_data.scalars = elem_id
            ug.cell_data.scalars.name = 'elem_id'
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
            if self.settings['applied_forces']:
                point_vector_counter += 1
                ug.point_data.add_array(app_forces, 'vector')
                ug.point_data.get_array(point_vector_counter).name = 'app_forces'

            write_data(ug, it_filename)


