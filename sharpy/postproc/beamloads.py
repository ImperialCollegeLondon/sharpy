import os

import numpy as np
from tvtk.api import tvtk, write_data

import sharpy.utils.cout_utils as cout
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra


@solver
class BeamLoads(BaseSolver):
    solver_id = 'BeamLoads'

    def __init__(self):
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings = None
        self.data = None

        self.folder = ''
        self.filename = ''

    def initialise(self, data):
        self.data = data
        self.settings = data.settings[self.solver_id]
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

    def run(self):
        self.calculate_loads()
        return self.data

    def calculate_loads(self):
        # initial (ini) loads
        tstep = self.data.structure.ini_info
        pos = tstep.pos
        psi = tstep.psi

        tstep.postproc_cell['gamma'] = np.zeros((self.data.structure.num_elem, 3))
        tstep.postproc_cell['kappa'] = np.zeros((self.data.structure.num_elem, 3))

        for ielem in range(self.data.structure.num_elem):
            crv = 0.5*(psi[ielem, 1, :] + psi[ielem, 0, :])
            cba = algebra.crv2rot(crv)
            tan = algebra.crv2tan(crv)

            inode0 = self.data.structure.elements[ielem].global_connectivities[0]
            inode1 = self.data.structure.elements[ielem].global_connectivities[1]
            tstep.postproc_cell['gamma'][ielem, :] = (
                    np.dot(cba,
                           pos[inode1, :] - pos[inode0, :]) /
                    np.linalg.norm(pos[inode1, :] - pos[inode0, :]))
            tstep.postproc_cell['kappa'][ielem, :] = (
                    np.dot(tan,
                           psi[ielem, 1, :] - psi[ielem, 0, :]) /
                    np.linalg.norm(pos[inode1, :] - pos[inode0, :]))

        # time loads
        for it in range(len(self.data.structure.timestep_info)):
            tstep = self.data.structure.timestep_info[it]
            pos = tstep.pos
            psi = tstep.psi

            tstep.postproc_cell['gamma'] = np.zeros((self.data.structure.num_elem, 3))
            tstep.postproc_cell['kappa'] = np.zeros((self.data.structure.num_elem, 3))
            tstep.postproc_cell['strain'] = np.zeros((self.data.structure.num_elem, 6))
            tstep.postproc_cell['loads'] = np.zeros((self.data.structure.num_elem, 6))

            for ielem in range(self.data.structure.num_elem):
                crv = 0.5*(psi[ielem, 1, :] + psi[ielem, 0, :])
                cba = algebra.crv2rot(crv)
                tan = algebra.crv2tan(crv)

                inode0 = self.data.structure.elements[ielem].global_connectivities[0]
                inode1 = self.data.structure.elements[ielem].global_connectivities[1]
                tstep.postproc_cell['gamma'][ielem, :] = (
                        np.dot(cba,
                               pos[inode1, :] - pos[inode0, :]) /
                        np.linalg.norm(pos[inode1, :] - pos[inode0, :]))
                tstep.postproc_cell['kappa'][ielem, :] = (
                        np.dot(tan,
                               psi[ielem, 1, :] - psi[ielem, 0, :]) /
                        np.linalg.norm(pos[inode1, :] - pos[inode0, :]))

                tstep.postproc_cell['strain'][ielem, 0:3] = (
                        tstep.postproc_cell['gamma'][ielem, :]
                        -
                        self.data.structure.ini_info.postproc_cell['gamma'][ielem, :])
                tstep.postproc_cell['strain'][ielem, 3:6] = (
                        tstep.postproc_cell['kappa'][ielem, :]
                        -
                        self.data.structure.ini_info.postproc_cell['kappa'][ielem, :])
                tstep.postproc_cell['loads'][ielem, :] = np.dot(
                        self.data.structure.stiffness_db[self.data.structure.elements[ielem].stiff_index, :, :],
                        tstep.postproc_cell['strain'][ielem, :])
