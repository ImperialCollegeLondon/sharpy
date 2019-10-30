import os

import numpy as np
from tvtk.api import tvtk, write_data

import sharpy.utils.algebra as algebra
import sharpy.utils.cout_utils as cout
from sharpy.utils.settings import str2bool
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.aero.utils.uvlmlib as uvlmlib


@solver
class AerogridPlot(BaseSolver):
    """
    Aerodynamic Grid Plotter

    """
    solver_id = 'AerogridPlot'
    solver_classification = 'post-processor'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['folder'] = 'str'
    settings_default['folder'] = './output'
    settings_description['folder'] = 'Output folder'

    settings_types['include_rbm'] = 'bool'
    settings_default['include_rbm'] = True

    settings_types['include_forward_motion'] = 'bool'
    settings_default['include_forward_motion'] = False

    settings_types['include_applied_forces'] = 'bool'
    settings_default['include_applied_forces'] = True

    settings_types['include_unsteady_applied_forces'] = 'bool'
    settings_default['include_unsteady_applied_forces'] = False

    settings_types['minus_m_star'] = 'int'
    settings_default['minus_m_star'] = 0

    settings_types['name_prefix'] = 'str'
    settings_default['name_prefix'] = ''
    settings_description['name_prefix'] = 'Prefix to add to file name'

    settings_types['u_inf'] = 'float'
    settings_default['u_inf'] = 0.

    settings_types['dt'] = 'float'
    settings_default['dt'] = 0.

    settings_types['include_velocities'] = 'bool'
    settings_default['include_velocities'] = False

    settings_types['num_cores'] = 'int'
    settings_default['num_cores'] = 1

    table = settings.SettingsTable()
    __doc__ += table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.settings = None
        self.data = None

        self.folder = ''
        self.body_filename = ''
        self.wake_filename = ''
        self.ts_max = 0

    def initialise(self, data, custom_settings=None):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)
        self.ts_max = self.data.ts + 1
        # create folder for containing files if necessary
        if not os.path.exists(self.settings['folder']):
            os.makedirs(self.settings['folder'])
        self.folder = self.settings['folder'] + '/' + self.data.settings['SHARPy']['case'] + '/aero/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.body_filename = (self.folder +
                              self.settings['name_prefix'] +
                              'body_' +
                              self.data.settings['SHARPy']['case'])
        self.wake_filename = (self.folder +
                              self.settings['name_prefix'] +
                              'wake_' +
                              self.data.settings['SHARPy']['case'])

    def run(self, online=False):
        # TODO: Create a dictionary to plot any variable as in beamplot
        if not online:
            for self.ts in range(self.ts_max):
                if self.data.aero.timestep_info[self.ts] is not None:
                    self.plot_body()
                    self.plot_wake()
            cout.cout_wrap('...Finished', 1)
        else:
            aero_tsteps = len(self.data.aero.timestep_info) - 1
            struct_tsteps = len(self.data.structure.timestep_info) - 1
            self.ts = np.max((aero_tsteps, struct_tsteps))
            self.plot_body()
            self.plot_wake()
        return self.data

    def plot_body(self):
        for i_surf in range(self.data.aero.timestep_info[self.ts].n_surf):
            filename = (self.body_filename +
                        '_' +
                        '%02u_' % i_surf +
                        '%06u' % self.ts)

            dims = self.data.aero.timestep_info[self.ts].dimensions[i_surf, :]
            point_data_dim = (dims[0]+1)*(dims[1]+1)  # + (dims_star[0]+1)*(dims_star[1]+1)
            panel_data_dim = (dims[0])*(dims[1])  # + (dims_star[0])*(dims_star[1])

            coords = np.zeros((point_data_dim, 3))
            conn = []
            panel_id = np.zeros((panel_data_dim,), dtype=int)
            panel_surf_id = np.zeros((panel_data_dim,), dtype=int)
            panel_gamma = np.zeros((panel_data_dim,))
            panel_gamma_dot = np.zeros((panel_data_dim,))
            normal = np.zeros((panel_data_dim, 3))
            point_struct_id = np.zeros((point_data_dim,), dtype=int)
            point_cf = np.zeros((point_data_dim, 3))
            point_unsteady_cf = np.zeros((point_data_dim, 3))
            zeta_dot = np.zeros((point_data_dim, 3))
            u_inf = np.zeros((point_data_dim, 3))
            if self.settings['include_velocities']:
                vel = np.zeros((point_data_dim, 3))
            counter = -1

            # coordinates of corners
            for i_n in range(dims[1]+1):
                for i_m in range(dims[0]+1):
                    counter += 1
                    coords[counter, :] = self.data.aero.timestep_info[self.ts].zeta[i_surf][:, i_m, i_n]
                    if self.settings['include_rbm']:
                        coords[counter, :] += self.data.structure.timestep_info[self.ts].for_pos[0:3]
                    if self.settings['include_forward_motion']:
                        coords[counter, 0] -= self.settings['dt'].value*self.ts*self.settings['u_inf'].value

            with_incidence_angle = True
            try:
                self.data.aero.timestep_info[self.ts].postproc_cell['incidence_angle']
            except KeyError:
                with_incidence_angle = False
            else:
                incidence_angle = np.zeros_like(panel_gamma)

            counter = -1
            node_counter = -1
            for i_n in range(dims[1] + 1):
                global_counter = self.data.aero.aero2struct_mapping[i_surf][i_n]
                for i_m in range(dims[0] + 1):
                    node_counter += 1
                    # point data
                    point_struct_id[node_counter] = global_counter
                    point_cf[node_counter, :] = self.data.aero.timestep_info[self.ts].forces[i_surf][0:3, i_m, i_n]
                    try:
                        point_unsteady_cf[node_counter, :] = self.data.aero.timestep_info[self.ts].dynamic_forces[i_surf][0:3, i_m, i_n]
                    except AttributeError:
                        pass
                    try:
                        zeta_dot[node_counter, :] = self.data.aero.timestep_info[self.ts].zeta_dot[i_surf][0:3, i_m, i_n]
                    except AttributeError:
                        pass
                    try:
                        u_inf[node_counter, :] = self.data.aero.timestep_info[self.ts].u_ext[i_surf][0:3, i_m, i_n]
                    except AttributeError:
                        pass
                    if i_n < dims[1] and i_m < dims[0]:
                        counter += 1
                    else:
                        continue

                    conn.append([node_counter + 0,
                                 node_counter + 1,
                                 node_counter + dims[0]+2,
                                 node_counter + dims[0]+1])
                    # cell data
                    normal[counter, :] = self.data.aero.timestep_info[self.ts].normals[i_surf][:, i_m, i_n]
                    panel_id[counter] = counter
                    panel_surf_id[counter] = i_surf
                    panel_gamma[counter] = self.data.aero.timestep_info[self.ts].gamma[i_surf][i_m, i_n]
                    panel_gamma_dot[counter] = self.data.aero.timestep_info[self.ts].gamma_dot[i_surf][i_m, i_n]

                    if with_incidence_angle:
                        incidence_angle[counter] = \
                            self.data.aero.timestep_info[self.ts].postproc_cell['incidence_angle'][i_surf][i_m, i_n]

            if self.settings['include_velocities']:
                vel = uvlmlib.uvlm_calculate_total_induced_velocity_at_points(self.data.aero.timestep_info[self.ts],
                                                                                              coords,
                                                                                              self.data.aero.timestep_info[self.ts].for_pos,
                                                                                              self.settings['numcores'])

            ug = tvtk.UnstructuredGrid(points=coords)
            ug.set_cells(tvtk.Quad().cell_type, conn)
            ug.cell_data.scalars = panel_id
            ug.cell_data.scalars.name = 'panel_n_id'
            ug.cell_data.add_array(panel_surf_id)
            ug.cell_data.get_array(1).name = 'panel_surface_id'
            ug.cell_data.add_array(panel_gamma)
            ug.cell_data.get_array(2).name = 'panel_gamma'
            ug.cell_data.add_array(panel_gamma_dot)
            ug.cell_data.get_array(3).name = 'panel_gamma_dot'
            if with_incidence_angle:
                ug.cell_data.add_array(incidence_angle)
                ug.cell_data.get_array(4).name = 'incidence_angle'
            ug.cell_data.vectors = normal
            ug.cell_data.vectors.name = 'panel_normal'
            ug.point_data.scalars = np.arange(0, coords.shape[0])
            ug.point_data.scalars.name = 'n_id'
            ug.point_data.add_array(point_struct_id)
            ug.point_data.get_array(1).name = 'point_struct_id'
            ug.point_data.add_array(point_cf)
            ug.point_data.get_array(2).name = 'point_steady_force'
            ug.point_data.add_array(point_unsteady_cf)
            ug.point_data.get_array(3).name = 'point_unsteady_force'
            ug.point_data.add_array(zeta_dot)
            ug.point_data.get_array(4).name = 'zeta_dot'
            ug.point_data.add_array(u_inf)
            ug.point_data.get_array(5).name = 'u_inf'
            if self.settings['include_velocities']:
                ug.point_data.add_array(vel)
                ug.point_data.get_array(6).name = 'velocity'
            write_data(ug, filename)

    def plot_wake(self):
        for i_surf in range(self.data.aero.timestep_info[self.ts].n_surf):
            filename = (self.wake_filename +
                        '_' +
                        '%02u_' % i_surf +
                        '%06u' % self.ts)

            dims_star = self.data.aero.timestep_info[self.ts].dimensions_star[i_surf, :]
            dims_star[0] -= self.settings['minus_m_star']

            point_data_dim = (dims_star[0]+1)*(dims_star[1]+1)
            panel_data_dim = (dims_star[0])*(dims_star[1])

            coords = np.zeros((point_data_dim, 3))
            conn = []
            panel_id = np.zeros((panel_data_dim,), dtype=int)
            panel_surf_id = np.zeros((panel_data_dim,), dtype=int)
            panel_gamma = np.zeros((panel_data_dim,))
            counter = -1
            # rotation_mat = self.data.structure.timestep_info[self.ts].cga().T
            # coordinates of corners
            for i_n in range(dims_star[1]+1):
                for i_m in range(dims_star[0]+1):
                    counter += 1
                    coords[counter, :] = self.data.aero.timestep_info[self.ts].zeta_star[i_surf][:, i_m, i_n]
                    if self.settings['include_rbm']:
                        coords[counter, :] += self.data.structure.timestep_info[self.ts].for_pos[0:3]
                    if self.settings['include_forward_motion']:
                        coords[counter, 0] -= self.settings['dt'].value*self.ts*self.settings['u_inf'].value

            counter = -1
            node_counter = -1
            # wake
            for i_n in range(dims_star[1]+1):
                for i_m in range(dims_star[0]+1):
                    node_counter += 1
                    # cell data
                    if i_n < dims_star[1] and i_m < dims_star[0]:
                        counter += 1
                    else:
                        continue

                    conn.append([node_counter + 0,
                                 node_counter + 1,
                                 node_counter + dims_star[0]+2,
                                 node_counter + dims_star[0]+1])
                    panel_id[counter] = counter
                    panel_surf_id[counter] = i_surf
                    panel_gamma[counter] = self.data.aero.timestep_info[self.ts].gamma_star[i_surf][i_m, i_n]

            ug = tvtk.UnstructuredGrid(points=coords)
            ug.set_cells(tvtk.Quad().cell_type, conn)
            ug.cell_data.scalars = panel_id
            ug.cell_data.scalars.name = 'panel_n_id'
            ug.cell_data.add_array(panel_surf_id)
            ug.cell_data.get_array(1).name = 'panel_surface_id'
            ug.cell_data.add_array(panel_gamma)
            ug.cell_data.get_array(2).name = 'panel_gamma'
            ug.point_data.scalars = np.arange(0, coords.shape[0])
            ug.point_data.scalars.name = 'n_id'
            write_data(ug, filename)
