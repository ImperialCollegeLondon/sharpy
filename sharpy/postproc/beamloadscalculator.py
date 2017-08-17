import sharpy.utils.cout_utils as cout
from sharpy.presharpy.utils.settings import str2bool
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.algebra as algebra
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
class BeamLoadsCalculator(BaseSolver):
    solver_id = 'BeamLoadsCalculator'
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
        self.calculate_loads()
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

    def calculate_loads(self):
        self.data.beam.timestep_info[self.ts].loads = np.zeros((self.data.beam.num_elem, 6))
        # shortcut to loads
        loads = self.data.beam.timestep_info[self.ts].loads
        pos_def = self.data.beam.timestep_info[self.ts].pos_def
        psi_def = self.data.beam.timestep_info[self.ts].psi_def

        pos_ini = self.data.beam.pos_ini
        psi_ini = self.data.beam.psi_ini

        num_elem = self.data.beam.num_elem
        for i_elem in range(num_elem):
            conn = self.data.beam.connectivities[i_elem, :]
            elem_pos_def = pos_def[conn, :]
            elem_pos_ini = pos_ini[conn, :]
            elem_psi_def = psi_def[i_elem, :, :]
            elem_psi_ini = psi_ini[i_elem, :, :]

            # first approximation to element length
            elem_length_ini = (np.linalg.norm(elem_pos_ini[2, :] - elem_pos_ini[0, :]) +
                               np.linalg.norm(elem_pos_ini[1, :] - elem_pos_ini[2, :]))
            elem_length_def = (np.linalg.norm(elem_pos_def[2, :] - elem_pos_def[0, :]) +
                               np.linalg.norm(elem_pos_def[1, :] - elem_pos_def[2, :]))

            # rotation at center of element
            # psi = psi_ini[i_elem, 2, :]
            # rot_ini = algebra.crv2rot(psi)
            # psi = psi_def[i_elem, 2, :]
            # rot_def = algebra.crv2rot(psi)

            loads[i_elem, 0:3] = ((elem_pos_def[1, :] - elem_pos_def[0, :])/elem_length_def -
                                  (elem_pos_ini[1, :] - elem_pos_ini[0, :])/elem_length_ini)

            psi = psi_ini[i_elem, 2, :]
            tan_ini = algebra.crv2tan(psi)
            psi = psi_def[i_elem, 2, :]
            tan_def = algebra.crv2tan(psi)

            loads[i_elem, 3:6] = (np.dot(tan_def, elem_psi_def[1, :] - elem_psi_def[0, :])/elem_length_def -
                                  np.dot(tan_ini, elem_psi_ini[1, :] - elem_psi_ini[0, :])/elem_length_ini)

            elem_stiff = self.data.beam.stiffness_db[self.data.beam.elements[i_elem].stiff_index, :, :]
            loads[i_elem, :] = np.dot(elem_stiff, loads[i_elem, :])







