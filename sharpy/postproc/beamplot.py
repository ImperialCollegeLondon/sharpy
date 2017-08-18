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
        self.folder = self.settings['route'] + '/' + self.data.settings['SHARPy']['case'] + '/beam/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.filename = (self.folder +
                         'beam_' +
                         self.data.settings['SHARPy']['case'] +
                         '_' +
                         self.settings['name_prefix'])
        self.print_info()
        self.text()
        self.plot()
        cout.cout_wrap('...Finished', 1)
        return self.data

    def text(self):
        if self.settings['print_pos_def']:
            for it in range(len(self.data.beam.timestep_info)):
                it_filename = (self.filename +
                               '%06u' % it +
                               '.csv')
                # write file
                np.savetxt(it_filename, self.data.beam.timestep_info[it].pos_def)

                it_filename = (self.filename +
                               '%06u' % it + '_crv_' +
                               '.csv')
                crv_matrix = np.zeros((self.data.beam.num_node, 3))
                for i_node in range(self.data.beam.num_node):
                    i_elem, i_local_node = self.data.beam.node_master_elem[i_node, :]
                    crv_matrix[i_node, :] = self.data.beam.timestep_info[it].psi_def[i_elem, i_local_node, :]
                # write file
                np.savetxt(it_filename, crv_matrix)


    def print_info(self):
        # find first bc == -1
        inode_tip = None
        for inode in range(self.data.beam.num_node):
            if self.data.beam.boundary_conditions[inode] == -1:
                inode_tip = inode
                ielem_tip, i_local_node = self.data.beam.node_master_elem[inode_tip, :]
                break

        if inode_tip is not None:
            cout.cout_wrap('Node %3u position:' % inode_tip, 2)
            cout.cout_wrap('\t%6f, %6f, %6f' % (
                self.data.beam.timestep_info[self.ts].pos_def[inode_tip, 0],
                self.data.beam.timestep_info[self.ts].pos_def[inode_tip, 1],
                self.data.beam.timestep_info[self.ts].pos_def[inode_tip, 2]), 2)
            cout.cout_wrap('Node %3u CRV:' % inode_tip, 2)
            cout.cout_wrap('\t%8e, %8e, %8e' % (
                self.data.beam.timestep_info[self.ts].psi_def[ielem_tip, i_local_node, 0],
                self.data.beam.timestep_info[self.ts].psi_def[ielem_tip, i_local_node, 1],
                self.data.beam.timestep_info[self.ts].psi_def[ielem_tip, i_local_node, 2]), 2)

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

        try:
            self.settings['frame']
        except KeyError:
            self.settings['frame'] = ''

        try:
            self.settings['print_pos_def'] = str2bool(self.settings['print_pos_def'])
        except KeyError:
            self.settings['print_pos_def'] = False

        try:
            self.settings['name_prefix']
        except KeyError:
            self.settings['name_prefix'] = ''

    def plot(self):
        for it in range(len(self.data.beam.timestep_info)):
            it_filename = (self.filename +
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
            output_loads = True
            try:
                self.data.beam.timestep_info[it].loads
            except AttributeError:
                output_loads = False

            if output_loads:
                gamma = np.zeros((num_elem, 3))
                kappa = np.zeros((num_elem, 3))

            if self.settings['applied_forces']:
                app_forces = np.zeros((num_nodes, 3))

            if self.settings['frame'] == 'inertial':
                try:
                    self.aero2inertial = self.data.grid.inertial2aero.T
                except AttributeError:
                    self.aero2inertial = np.eye(3)
                    cout.cout_wrap('BeamPlot: No inertial2aero information, output will be in body FoR', 0)
            else:
                self.aero2inertial = np.eye(3)
            # coordinates of corners
            for i_node in range(num_nodes):
                if self.data.beam.timestep_info[it].with_rb:
                    coords[i_node, :] = self.data.beam.timestep_info[it].glob_pos_def[i_node, :]
                else:
                    coords[i_node, :] = np.dot(self.aero2inertial, self.data.beam.timestep_info[it].pos_def[i_node, :])

            for i_node in range(num_nodes):
                i_elem = self.data.beam.node_master_elem[i_node, 0]
                i_local_node = self.data.beam.node_master_elem[i_node, 1]
                node_id[i_node] = i_node

                self.data.beam.update(it)

                v1, v2, v3 = self.data.beam.elements[i_elem].deformed_triad()
                local_x[i_node, :] = np.dot(self.aero2inertial, v1[i_local_node, :])
                local_y[i_node, :] = np.dot(self.aero2inertial, v2[i_local_node, :])
                local_z[i_node, :] = np.dot(self.aero2inertial, v3[i_local_node, :])

                # applied forces
                if self.settings['applied_forces']:
                    import sharpy.utils.algebra as algebra
                    Cab = algebra.crv2rot(self.data.beam.timestep_info[it].psi_def[i_elem, i_local_node, :])
                    app_forces[i_node, :] = np.dot(self.aero2inertial,
                                                   np.dot(Cab,
                                                          self.data.beam.app_forces[i_node, 0:3]))
                    # app_forces[i_node, :] = self.data.beam.app_forces[i_node, 0:3]

            for i_elem in range(num_elem):
                conn[i_elem, :] = self.data.beam.elements[i_elem].reordered_global_connectivities
                elem_id[i_elem] = i_elem
                if output_loads:
                    gamma[i_elem, :] = self.data.beam.timestep_info[it].loads[i_elem, 0:3]
                    kappa[i_elem, :] = self.data.beam.timestep_info[it].loads[i_elem, 3:6]

            ug = tvtk.UnstructuredGrid(points=coords)
            ug.set_cells(tvtk.Line().cell_type, conn)
            ug.cell_data.scalars = elem_id
            ug.cell_data.scalars.name = 'elem_id'
            if output_loads:
                ug.cell_data.add_array(gamma, 'vector')
                ug.cell_data.get_array(1).name = 'gamma'
                ug.cell_data.add_array(kappa, 'vector')
                ug.cell_data.get_array(2).name = 'kappa'
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


