import os

import numpy as np

import sharpy.utils.cout_utils as cout
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings_utils
import sharpy.utils.algebra as algebra


import vtk
from vtk.numpy_interface import algorithms as algs
from vtk.numpy_interface import dataset_adapter as dsa


@solver
class BeamPlot(BaseSolver):
    """
    Plots beam to Paraview format
    """
    solver_id = 'BeamPlot'
    solver_classification = 'post-processor'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['include_rbm'] = 'bool'
    settings_default['include_rbm'] = True
    settings_description['include_rbm'] = 'Include frame of reference rigid body motion'

    settings_types['include_FoR'] = 'bool'
    settings_default['include_FoR'] = False
    settings_description['include_FoR'] = 'Include frame of reference variables'

    settings_types['include_applied_forces'] = 'bool'
    settings_default['include_applied_forces'] = True
    settings_description['include_applied_forces'] = 'Write beam applied forces'

    settings_types['include_applied_moments'] = 'bool'
    settings_default['include_applied_moments'] = True
    settings_description['include_applied_moments'] = 'Write beam applied moments'

    settings_types['name_prefix'] = 'str'
    settings_default['name_prefix'] = ''
    settings_description['name_prefix'] = 'Name prefix for files'

    settings_types['output_rbm'] = 'bool'
    settings_default['output_rbm'] = True
    settings_description['output_rbm'] = 'Write ``csv`` file with rigid body motion data'

    settings_types['stride'] = 'int'
    settings_default['stride'] = 1
    settings_description['stride'] = 'Number of steps between the execution calls when run online'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):

        self.settings = None
        self.data = None

        self.folder = ''
        self.filename = ''
        self.filename_for = ''
        self.caller = None

    def initialise(self, data, custom_settings=None, caller=None, restart=False):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings_utils.to_custom_types(self.settings, self.settings_types, self.settings_default)
        # create folder for containing files if necessary
        self.folder = data.output_folder + '/beam/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.filename = (self.folder +
                         self.settings['name_prefix'] +
                         'beam_' +
                         self.data.settings['SHARPy']['case'])
        self.filename_for = (self.folder +
                             self.settings['name_prefix'] +
                             'for_' +
                             self.data.settings['SHARPy']['case'])
        self.caller = caller

    def run(self, **kwargs):

        online = settings_utils.set_value_or_default(kwargs, 'online', False)

        self.plot(online)
        if not online:
            self.write()
            cout.cout_wrap('...Finished', 1)
        return self.data

    def write(self):
        if self.settings['output_rbm']:
            filename = self.filename + '_rbm_acc.csv'
            timesteps = len(self.data.structure.timestep_info)
            temp_matrix = np.zeros((timesteps, 6))
            for it in range(timesteps):
                if self.data.structure.timestep_info[it] is not None:
                    temp_matrix[it, :] = self.data.structure.timestep_info[it].for_acc

            np.savetxt(filename, temp_matrix, delimiter=',')

    def plot(self, online):
        if not online:
            for it in range(len(self.data.structure.timestep_info)):
                if self.data.structure.timestep_info[it] is not None:
                    self.write_beam(it)
                    if self.settings['include_FoR']:
                        self.write_for(it)
        elif ((len(self.data.structure.timestep_info) - 1) % self.settings['stride'] == 0):
            it = len(self.data.structure.timestep_info) - 1
            self.write_beam(it)
            if self.settings['include_FoR']:
                self.write_for(it)

    def write_beam(self, it):
        it_filename = f"{self.filename}{it:06d}.vtp"
        num_nodes = self.data.structure.num_node
        num_elem = self.data.structure.num_elem

        conn = np.zeros((num_elem, 3), dtype=int)
        node_id = np.zeros((num_nodes,), dtype=int)
        elem_id = np.zeros((num_elem,), dtype=int)
        coords_a_cell = np.zeros((num_elem, 3), dtype=int)
        local_x = np.zeros((num_nodes, 3))
        local_y = np.zeros((num_nodes, 3))
        local_z = np.zeros((num_nodes, 3))
        coords_a = np.zeros((num_nodes, 3))
        app_forces = np.zeros((num_nodes, 3))
        app_moment = np.zeros((num_nodes, 3))

        forces_constraints_nodes = np.zeros((num_nodes, 3))
        moments_constraints_nodes = np.zeros((num_nodes, 3))

        tstep = self.data.structure.timestep_info[it]

        # aero2inertial rotation
        aero2inertial = tstep.cga()

        # coordinates of corners
        coords = tstep.glob_pos(include_rbm=self.settings['include_rbm'])

        if tstep.mb_dict is None:
            pass
        else:
            for i_node in range(tstep.num_node):
                c = self.data.structure.timestep_info[0].cga()
                coords[i_node, :] += c @ tstep.for_pos[:3]

        # check if I can output gravity forces
        with_gravity = False
        try:
            gravity_forces = tstep.gravity_forces[:]
            gravity_forces_g = np.zeros_like(gravity_forces)
            with_gravity = True
        except AttributeError:
            pass

        # check if postproc dicts are present and count/prepare
        with_postproc_cell = False
        try:
            tstep.postproc_cell
            with_postproc_cell = True
        except AttributeError:
            pass
        with_postproc_node = False
        try:
            tstep.postproc_node
            with_postproc_node = True
        except AttributeError:
            pass

        # count number of arguments
        postproc_cell_vector = []
        postproc_cell_6vector = []
        for k, v in tstep.postproc_cell.items():
            _, cols = v.shape
            if cols == 1:
                raise NotImplementedError('scalar cell types not supported in beamplot (Easy to implement)')
                # postproc_cell_scalar.append(k)
            elif cols == 3:
                postproc_cell_vector.append(k)
            elif cols == 6:
                postproc_cell_6vector.append(k)
            else:
                raise AttributeError('Only scalar and 3-vector types supported in beamplot')
        # count number of arguments
        postproc_node_scalar = []
        postproc_node_vector = []
        postproc_node_6vector = []
        for k, v in tstep.postproc_node.items():
            try:
                _, cols = v.shape
            except ValueError:
                # for np.arrays with shape (x,)
                cols = 1
            if cols == 1:
                postproc_node_scalar.append(k)
            elif cols == 3:
                postproc_node_vector.append(k)
            elif cols == 6:
                postproc_node_6vector.append(k)
            else:
                raise AttributeError('Only scalar and 3-vector types supported in beamplot')

        for i_node in range(num_nodes):
            i_elem = self.data.structure.node_master_elem[i_node, 0]
            i_local_node = self.data.structure.node_master_elem[i_node, 1]
            node_id[i_node] = i_node

            v1 = np.array([1., 0, 0])
            v2 = np.array([0., 1, 0])
            v3 = np.array([0., 0, 1])
            cab = algebra.crv2rotation(
                tstep.psi[i_elem, i_local_node, :])
            local_x[i_node, :] = np.dot(aero2inertial, np.dot(cab, v1))
            local_y[i_node, :] = np.dot(aero2inertial, np.dot(cab, v2))
            local_z[i_node, :] = np.dot(aero2inertial, np.dot(cab, v3))

            if i_local_node == 2:
                coords_a_cell[i_elem, :] = tstep.pos[i_node, :]
            coords_a[i_node, :] = tstep.pos[i_node, :]

            # applied forces
            cab = algebra.crv2rotation(tstep.psi[i_elem, i_local_node, :])
            app_forces[i_node, :] = np.dot(aero2inertial,
                                           np.dot(cab,
                                                  tstep.steady_applied_forces[i_node, 0:3]+
                                                  tstep.unsteady_applied_forces[i_node, 0:3]))
            app_moment[i_node, :] = np.dot(aero2inertial,
                                           np.dot(cab,
                                                  tstep.steady_applied_forces[i_node, 3:6]+
                                                  tstep.unsteady_applied_forces[i_node, 3:6]))
            forces_constraints_nodes[i_node, :] = np.dot(aero2inertial,
                                                         np.dot(cab,
                                                                tstep.forces_constraints_nodes[i_node, 0:3]))
            moments_constraints_nodes[i_node, :] = np.dot(aero2inertial,
                                                          np.dot(cab,
                                                                 tstep.forces_constraints_nodes[i_node, 3:6]))

            if with_gravity:
                gravity_forces_g[i_node, 0:3] = np.dot(aero2inertial,
                                                     gravity_forces[i_node, 0:3])
                gravity_forces_g[i_node, 3:6] = np.dot(aero2inertial,
                                                     gravity_forces[i_node, 3:6])

        for i_elem in range(num_elem):
            conn[i_elem, :] = self.data.structure.elements[i_elem].reordered_global_connectivities
            elem_id[i_elem] = i_elem

        ug = vtk.vtkPolyData(points=coords)
        cells = vtk.vtkCellArray()
        for _conn in conn:
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(3)
            line.GetPointIds().SetId(0, _conn[0])
            line.GetPointIds().SetId(1, _conn[1])
            line.GetPointIds().SetId(2, _conn[2])
            cells.InsertNextCell(line)
        ug.SetLines(cells)

        ug.GetCellData().AddArray(dsa.numpyTovtkDataArray(np.array(elem_id), name='elem_id'))
        ug.GetCellData().AddArray(dsa.numpyTovtkDataArray(coords_a_cell, name="coords_a_elem"))
        ug.GetPointData().AddArray(
            dsa.numpyTovtkDataArray(node_id, name="node_id")
        )
        ug.GetPointData().AddArray(dsa.numpyTovtkDataArray(local_x, name="local_x"))
        ug.GetPointData().AddArray(dsa.numpyTovtkDataArray(local_y, name="local_y"))
        ug.GetPointData().AddArray(dsa.numpyTovtkDataArray(local_z, name="local_z"))
        ug.GetPointData().AddArray(dsa.numpyTovtkDataArray(coords_a, name="coords_a"))

        if with_postproc_cell:
            for k in postproc_cell_vector:
                ug.GetCellData().AddArray(
                    dsa.numpyTovtkDataArray(tstep.postproc_cell[k], name=f"{k}_cell")
                )
            for k in postproc_cell_6vector:
                for i in range(2):
                    ug.GetCellData().AddArray(
                        dsa.numpyTovtkDataArray(
                            tstep.postproc_cell[k][:, 3*i:3*(i+1)], name=f"{k}_{i}_cell"
                        )
                    )

        if self.settings['include_applied_forces']:
            ug.GetPointData().AddArray(dsa.numpyTovtkDataArray(app_forces, name="app_forces"))
            ug.GetPointData().AddArray(
                dsa.numpyTovtkDataArray(forces_constraints_nodes, name="forces_constraints_node")
            )
            if with_gravity:
                ug.GetPointData().AddArray(
                    dsa.numpyTovtkDataArray(
                        gravity_forces_g[:, :3], name="gravity_forces"
                    )
                )

        if self.settings['include_applied_moments']:
            ug.GetPointData().AddArray(
                dsa.numpyTovtkDataArray(app_moment, name="app_moments")
            )
            ug.GetPointData().AddArray(
                dsa.numpyTovtkDataArray(moments_constraints_nodes, name="moments_constraints_nodes")
            )
            if with_gravity:
                ug.GetPointData().AddArray(dsa.numpyTovtkDataArray(gravity_forces_g[:, 3:6], name="gravity_moments"))

        if with_postproc_node:
            for k in postproc_node_vector:
                ug.GetPointData().AddArray(
                    dsa.numpyTovtkDataArray(
                        tstep.postproc_node[k], name=f"{k}_point"
                    )
                )
            for k in postproc_node_6vector:
                for i in range(2):
                    ug.GetPointData().AddArray(
                        dsa.numpyTovtkDataArray(
                            tstep.postproc_node[k][:, 3*i:3*(i+1)], name=f"{k}_{i}_point"
                        )
                    )

            for k in postproc_node_scalar:
                ug.GetPointData().AddArray(
                    dsa.numpyTovtkDataArray(
                        tstep.postproc_node[k],
                        name=str(k),
                    )
                )

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(it_filename)
        writer.SetInputData(ug)
        writer.Write()

    def write_for(self, it):
        it_filename = f"{self.filename_for}{it:06d}.vtp"

        forces_constraints_for = np.zeros((self.data.structure.num_bodies, 3))
        moments_constraints_for = np.zeros((self.data.structure.num_bodies, 3))

        # aero2inertial rotation
        aero2inertial = self.data.structure.timestep_info[it].cga()

        # coordinates of corners
        for_coords = np.zeros((self.data.structure.num_bodies, 3))
        if self.settings['include_rbm']:
            offset = np.zeros((3,))
        else:
            offset = self.data.structure.timestep_info[it].mb_FoR_pos[0, 0:3]

        for ibody in range(self.data.structure.num_bodies):
            for_coords[ibody, :] = self.data.structure.timestep_info[it].mb_FoR_pos[ibody, 0:3] - offset
            forces_constraints_for[ibody, :] = aero2inertial @ self.data.structure.timestep_info[it].forces_constraints_FoR[ibody, :3]
            moments_constraints_for[ibody, :] = aero2inertial @ self.data.structure.timestep_info[it].forces_constraints_FoR[ibody, 3:6]

        for_mesh = vtk.vtkPolyData(points=for_coords)

        for_mesh.GetPointData().AddArray(
            dsa.numpyTovtkDataArray(
                forces_constraints_for,
                'forces_constraints_FoR',
            )
        )

        for_mesh.GetPointData().AddArray(
            dsa.numpyTovtkDataArray(
                moments_constraints_for,
                "moments_constraints_FoR",
            )
        )

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(it_filename)
        writer.SetInputData(for_mesh)
        writer.Write()
