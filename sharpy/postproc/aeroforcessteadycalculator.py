import sharpy.utils.cout_utils as cout
from sharpy.presharpy.utils.settings import str2bool
from sharpy.utils.solver_interface import solver, BaseSolver

from tvtk.api import tvtk, write_data
import numpy as np
import os


@solver
class AeroForcesSteadyCalculator(BaseSolver):
    solver_id = 'AeroForcesCalculator'
    solver_type = 'postproc'
    solver_unsteady = False

    def __init__(self):
        self.ts = 0  # steady solver
        pass

    def initialise(self, data):
        self.data = data
        self.settings = data.settings[self.solver_id]
        self.convert_settings()

    def run(self):
        # create folder for containing files if necessary
        if not os.path.exists(self.settings['route']):
            os.makedirs(self.settings['route'])
        self.calculate_forces()
        self.output_forces()
        cout.cout_wrap('...Finished', 1)
        return self.data

    def convert_settings(self):
        try:
            self.settings['route'] = (str2bool(self.settings['route']))
        except KeyError:
            cout.cout_wrap('AeroForcesSteadyCalculator: no location for figures defined, defaulting to ./output', 3)
            self.settings['route'] = './output'
        pass

    def calculate_forces(self):
        dynamic_pressure = 0.5*self.data.grid
        # CL

        # CD

        # CM


    def output_forces(self):
        pass

    # def plot_grid(self):
    #     for i_surf in range(self.data.grid.timestep_info[self.ts].n_surf):
    #         filename = 'grid_%s_%03u' % (self.data.settings['SHARPy']['case'], i_surf)
    #
    #         dims = self.data.grid.timestep_info[self.ts].dimensions[i_surf, :]
    #         dims_star = self.data.grid.timestep_info[self.ts].dimensions_star[i_surf, :]
    #         point_data_dim = (dims[0]+1)*(dims[1]+1) + (dims_star[0]+1)*(dims_star[1]+1)
    #         panel_data_dim = (dims[0])*(dims[1]) + (dims_star[0])*(dims_star[1])
    #
    #         coords = np.zeros((point_data_dim, 3))
    #         conn = []
    #         panel_id = np.zeros((panel_data_dim,), dtype=int)
    #         panel_surf_id = np.zeros((panel_data_dim,), dtype=int)
    #         panel_gamma = np.zeros((panel_data_dim,))
    #         normal = np.zeros((panel_data_dim, 3))
    #         point_struct_id = np.zeros((point_data_dim,), dtype=int)
    #         point_cf = np.zeros((point_data_dim, 3))
    #         counter = -1
    #         # coordinates of corners
    #         for i_n in range(dims[1]+1):
    #             for i_m in range(dims[0]+1):
    #                 counter += 1
    #                 coords[counter, :] = self.data.grid.timestep_info[self.ts].zeta[i_surf][:, i_m, i_n]
    #         for i_n in range(dims_star[1]+1):
    #             for i_m in range(dims_star[0]+1):
    #                 counter += 1
    #                 coords[counter, :] = self.data.grid.timestep_info[self.ts].zeta_star[i_surf][:, i_m, i_n]
    #
    #         counter = -1
    #         node_counter = -1
    #         for i_n in range(dims[1] + 1):
    #             global_counter = self.data.grid.aero2struct_mapping[i_surf][i_n]
    #             for i_m in range(dims[0] + 1):
    #                 node_counter += 1
    #                 # point data
    #                 point_struct_id[node_counter] = global_counter
    #                 point_cf[node_counter, :] = self.data.grid.timestep_info[self.ts].forces[i_surf][0:3, i_m, i_n]
    #                 if i_n < dims[1] and i_m < dims[0]:
    #                     counter += 1
    #                 else:
    #                     continue
    #
    #                 conn.append([node_counter + 0,
    #                              node_counter + 1,
    #                              node_counter + dims[0]+2,
    #                              node_counter + dims[0]+1])
    #                 # cell data
    #                 normal[counter, :] = self.data.grid.timestep_info[self.ts].normals[i_surf][:, i_m, i_n]
    #                 panel_id[counter] = counter
    #                 panel_surf_id[counter] = i_surf
    #                 panel_gamma[counter] = self.data.grid.timestep_info[self.ts].gamma[i_surf][i_m, i_n]
    #
    #         # wake
    #         for i_n in range(dims_star[1]+1):
    #             for i_m in range(dims_star[0]+1):
    #                 node_counter += 1
    #                 # point data
    #
    #                 # cell data
    #                 if i_n < dims_star[1] and i_m < dims_star[0]:
    #                     counter += 1
    #                 else:
    #                     continue
    #
    #                 conn.append([node_counter + 0,
    #                              node_counter + 1,
    #                              node_counter + dims_star[0]+2,
    #                              node_counter + dims_star[0]+1])
    #                 panel_id[counter] = counter
    #                 panel_surf_id[counter] = i_surf
    #                 panel_gamma[counter] = self.data.grid.timestep_info[self.ts].gamma_star[i_surf][i_m, i_n]
    #
    #
    #         ug = tvtk.UnstructuredGrid(points=coords)
    #         ug.set_cells(tvtk.Quad().cell_type, conn)
    #         ug.cell_data.scalars = panel_id
    #         ug.cell_data.scalars.name = 'panel_n_id'
    #         ug.cell_data.add_array(panel_surf_id)
    #         ug.cell_data.get_array(1).name = 'panel_surface_id'
    #         ug.cell_data.add_array(panel_gamma)
    #         ug.cell_data.get_array(2).name = 'panel_gamma'
    #         ug.cell_data.vectors = normal
    #         ug.cell_data.vectors.name = 'panel_normal'
    #         ug.point_data.scalars = np.arange(0, coords.shape[0])
    #         ug.point_data.scalars.name = 'n_id'
    #         ug.point_data.add_array(point_struct_id)
    #         ug.point_data.get_array(1).name = 'point_struct_id'
    #         ug.point_data.vectors = point_cf
    #         ug.point_data.vectors.name = 'point_cf'
    #         write_data(ug, filename)
    #     pass


