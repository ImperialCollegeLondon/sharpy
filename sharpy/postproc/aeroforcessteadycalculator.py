import sharpy.utils.cout_utils as cout
from sharpy.presharpy.utils.settings import str2bool
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.presharpy.aerogrid.utils as aero_utils

from tvtk.api import tvtk, write_data
import numpy as np
import os
import ctypes as ct


class ForcesContainer(object):
    def __init__(self):
        self.ts = 0
        self.t = 0.0
        self.forces = []
        self.coords = []


@solver
class AeroForcesSteadyCalculator(BaseSolver):
    solver_id = 'AeroForcesSteadyCalculator'
    solver_type = 'postproc'
    solver_unsteady = False

    def __init__(self):
        self.ts = 0  # steady solver
        pass

    def initialise(self, data):
        self.data = data
        self.settings = data.settings[self.solver_id]
        self.convert_settings()

        # nodes per beam
        # self.n_nodes_beam = max(self.settings['beams'])*[]
        # for i_beam in self.settings['beams']:
        #     self.n_nodes_beam[i_beam] = sum([1 for i in self.data.beam.beam_number if i == i_beam])
        #     print('Beam %u, %u nodes' % (i_beam, self.n_nodes_beam[i_beam]))
        #
        # # initialise forces container
        # self.data.beam.forces_container = []
        # self.data.beam.forces_container.append(ForcesContainer())
        # self.data.beam.forces_container[0].ts = self.ts
        # self.data.beam.forces_container[0].t = self.t
        # self.data.beam.forces_container[0].forces = max(self.settings['beams'])*[]
        # for i_beam in self.settings['beams']:
        #     self.data.beam.forces_container[0].forces.append(np.zeros())

    def run(self):
        self.ts = 0


        # create folder for containing files if necessary
        if not os.path.exists(self.settings['route']):
            os.makedirs(self.settings['route'])
        self.calculate_forces()
        # self.output_forces()
        cout.cout_wrap('...Finished', 1)
        return self.data

    def convert_settings(self):
        try:
            self.settings['route'] = (str2bool(self.settings['route']))
        except KeyError:
            # cout.cout_wrap('AeroForcesSteadyCalculator: no location for figures defined, defaulting to ./output', 3)
            self.settings['route'] = './output'
        try:
            self.settings['beams'] = np.fromstring(self.settings['beams'], sep=',', dtype=ct.c_double)
        except KeyError:
            self.settings['beams'] = []

    def calculate_forces(self):
        # dynamic_pressure = (0.5*self.data.flightconditions['FlightCon']['rho_inf']*
        #                     self.data.flightconditions['FlightCon']['u_inf']**2*
        #                     self.data.flightconditions['FlightCon']['c_ref']*
        #                     self.data.flightconditions['FlightCon']['b_ref'])
        # rot = aero_utils.wind2body_rot(self.data.flightconditions['FlightCon']['alpha'],
        #                                self.data.flightconditions['FlightCon']['beta'])
        rot = np.eye(3)

        force = self.data.grid.timestep_info[self.ts].forces
        total_force = np.zeros((3,))
        n_surf = len(force)
        for i_surf in range(n_surf):
            if self.settings['beams'] is not []:
                if i_surf not in self.settings['beams']:
                    continue
            _, n_rows, n_cols = force[i_surf].shape
            for i_m in range(n_rows):
                for i_n in range(n_cols):
                    total_force += np.dot(rot.T, force[i_surf][0:3, i_m, i_n])

        lift = total_force[2]
        cout.cout_wrap('Lift = %f (N)' % lift)
        drag = total_force[0]
        cout.cout_wrap('Drag= %f (N)' % drag)
        side_force = total_force[1]
        cout.cout_wrap('Side force = %f (N)' % side_force)

        # CL
        # cl = total_force[2]/dynamic_pressure
        # # CD
        # cd = total_force[0]/dynamic_pressure
        # cout.cout_wrap('CD = %f6 ' % cd)
        # # C side force
        # cs = total_force[1]/dynamic_pressure
        # cout.cout_wrap('CLateral = %f6 ' % cs)
        # CM

    def output_forces(self):
        self.coor = []
        self.forces = []
        for i_beam in self.settings['beams']:
            n_nodes_beam = sum([1 for i in self.data.beam.beam_number if i == i_beam])
            print('Beam %u, %u nodes' % (i_beam, n_nodes_beam))
        # i_surf = 0
        # in_force = self.data.grid.timestep_info[self.ts].forces
        # y_coor = np.zeros((self.data.grid.aero_dimensions[i_surf][1],))
        # spacing = np.zeros((self.data.grid.aero_dimensions[i_surf][1],))
        # self.forces = np.zeros((self.data.grid.aero_dimensions[i_surf][1], 6))
        # beam_global_node = np.zeros((self.data.grid.aero_dimensions[i_surf][1],), dtype=int)
        # for i_n in range(self.data.grid.aero_dimensions[i_surf][1]):
        #     beam_global_node[i_n] = self.data.grid.aero2struct_mapping[i_surf][i_n]
        #     y_coor[i_n] = self.data.beam.timestep_info[self.ts].pos_def[beam_global_node[i_n], 1]
        #     if i_n == 0:
        #         spacing[i_n] =
        #     elif i_n == self.data.grid.aero_dimensions[i_surf][1] - 1:
        #         spacing[i_n] =
        #     else:
        #         i_1 = i_n - 1
        #         i_2 = i_n + 1
        #         spacing[i_n] =
        #
        #     for i_m in range(self.data.grid.aero_dimensions[i_surf][0]):
        #         # TODO moments
        #         for i_dim in range(3):
        #             self.forces[i_n, i_dim] += in_force[i_surf][i_dim, i_m, i_n]


        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(y_coor, self.forces[:, 0], 'r')
        plt.plot(y_coor, self.forces[:, 1], 'k')
        plt.plot(y_coor, self.forces[:, 2], 'b')
        plt.grid('on')
        plt.show()

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


