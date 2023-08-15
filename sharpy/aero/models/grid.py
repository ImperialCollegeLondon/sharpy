"""Grid

Grid contains
"""
import ctypes as ct
import warnings

import numpy as np

import sharpy.utils.algebra as algebra
import sharpy.utils.generator_interface as gen_interface


class Grid(object):
    """
    ``Grid``is the parent class for the lifting surface grid and nonlifting
    body grids.

    It is created by the solver :class:`sharpy.solvers.aerogridloader.AerogridLoader`

    """
    def __init__(self):
        self.data_dict = None
        self.beam = None
        self.aero_settings = None
        self.timestep_info = []
        self.ini_info = None

        self.surface_distribution = None
        self.surface_m = None
        self.dimensions = None
        self.grid_type = None

        self.n_node = 0
        self.n_elem = 0
        self.n_surf = 0
        self.n_aero_node = 0
        self.grid_type = None

        self.struct2aero_mapping = None
        self.aero2struct_mapping = []


    def generate(self, data_dict, beam, aero_settings, ts):


        self.data_dict = data_dict
        self.beam = beam
        self.aero_settings = aero_settings
        # key words = safe in aero_settings? --> grid_type
        # number of total nodes (structural + aero&struc)
        self.n_node = len(data_dict[self.grid_type + '_node']) # gridtype + '_node'
        # number of elements
        self.n_elem = len(data_dict['surface_distribution'])
        # surface distribution
        self.surface_distribution = data_dict['surface_distribution']
        # number of surfaces
        temp = set(data_dict['surface_distribution'])
        #TO-DO: improve: avoid for loops
        self.n_surf = sum(1 for i in temp if i >= 0)
        # number of chordwise panels
        self.surface_m = data_dict['surface_m']
        # number of aero nodes
        self.n_aero_node = sum(data_dict[self.grid_type + '_node'])


        # get N per surface
        self.calculate_dimensions()

        # write grid info to screen
        # self.output_info()

    def calculate_dimensions(self):
        self.dimensions = np.zeros((self.n_surf, 2), dtype=int)
        for i in range(self.n_surf):
            # adding M values
            self.dimensions[i, 0] = self.surface_m[i]
        # Improvement:
        # self.aero.dimensions[:, 0] = self.surface_m[:]
        # count N values (actually, the count result
        # will be N+1)
        nodes_in_surface = []

        #IMPROVEMENT
        for i_surf in range(self.n_surf):
            nodes_in_surface.append([])

        # Improvement!
        for i_elem in range(self.beam.num_elem):
            nodes = self.beam.elements[i_elem].global_connectivities
            i_surf = self.surface_distribution[i_elem]
            if i_surf < 0:
                continue
            for i_global_node in nodes:
                if i_global_node in nodes_in_surface[i_surf]:
                    continue
                else:
                    nodes_in_surface[i_surf].append(i_global_node)
                if self.data_dict[self.grid_type + '_node'][i_global_node]:
                    self.dimensions[i_surf, 1] += 1

        # accounting for N+1 nodes -> N panels
        self.dimensions[:, 1] -= 1


    def add_timestep(self):
        try:
            self.timestep_info.append(self.timestep_info[-1].copy())
        except IndexError:
            self.timestep_info.append(self.ini_info.copy())

    def generate_zeta_timestep_info(self, structure_tstep, aero_tstep, beam, aero_settings, it=None, dt=None):
        if it is None:
            it = len(beam.timestep_info) - 1


    def generate_zeta(self, beam, aero_settings, ts=-1, beam_ts=-1):
        self.generate_zeta_timestep_info(beam.timestep_info[beam_ts],
                                         self.timestep_info[ts],
                                         beam,
                                         aero_settings)

    def generate_mapping(self):
        self.struct2aero_mapping = [[]]*self.n_node
        surf_n_counter = np.zeros((self.n_surf,), dtype=int)
        nodes_in_surface = []
        for i_surf in range(self.n_surf):
            nodes_in_surface.append([])
        for i_elem in range(self.n_elem):
            i_surf = self.surface_distribution[i_elem]
            if i_surf == -1:
                continue
            for i_global_node in self.beam.elements[i_elem].reordered_global_connectivities:
                if not self.data_dict[self.grid_type + '_node'][i_global_node]:
                    continue

                if i_global_node in nodes_in_surface[i_surf]:
                    continue
                else:
                    nodes_in_surface[i_surf].append(i_global_node)
                    surf_n_counter[i_surf] += 1
                    try:
                        self.struct2aero_mapping[i_global_node][0]
                    except IndexError:
                        self.struct2aero_mapping[i_global_node] = []

                i_n = surf_n_counter[i_surf] - 1
                self.struct2aero_mapping[i_global_node].append({'i_surf': i_surf,
                                                                'i_n': i_n})

        nodes_in_surface = []
        for i_surf in range(self.n_surf):
            nodes_in_surface.append([])

        for i_surf in range(self.n_surf):
            self.aero2struct_mapping.append([-1]*(surf_n_counter[i_surf]))

        for i_elem in range(self.n_elem):
            for i_global_node in self.beam.elements[i_elem].global_connectivities:
                for i in range(len(self.struct2aero_mapping[i_global_node])):
                    try:
                        i_surf = self.struct2aero_mapping[i_global_node][i]['i_surf']
                        i_n = self.struct2aero_mapping[i_global_node][i]['i_n']
                        if i_global_node in nodes_in_surface[i_surf]:
                            continue
                        else:
                            nodes_in_surface[i_surf].append(i_global_node)
                    except KeyError:
                        continue
                    self.aero2struct_mapping[i_surf][i_n] = i_global_node

    def update_orientation(self, quat, ts=-1):
        rot = algebra.quat2rotation(quat)
        self.timestep_info[ts].update_orientation(rot.T)
