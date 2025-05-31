import numpy as np

import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings
import numpy as np
from tvtk.api import tvtk, write_data
import ctypes as ct
import copy


@generator_interface.generator
class GridBox(generator_interface.BaseGenerator):
    """
    GridBox

    Generatex a grid within a box to be used to generate the flow field during the postprocessing

    """
    generator_id = 'GridBox'
    generator_classification = 'utils'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['coords_0'] = 'list(float)'
    settings_default['coords_0'] = [0., 0., 0.]
    settings_description['coords_0'] = 'First bounding box corner'

    settings_types['coords_1'] = 'list(float)'
    settings_default['coords_1'] = [10., 0., 10.]
    settings_description['coords_1'] = 'Second bounding box corner'

    settings_types['spacing'] = 'list(float)'
    settings_default['spacing'] = [1., 1., 1.]
    settings_description['spacing'] = 'Spacing parameters of the bbox'

    settings_types['moving'] = 'bool'
    settings_default['moving'] = False
    settings_description['moving'] = 'If ``True``, the box moves with the body frame of reference. It does not rotate with it, though'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.in_dict = dict()
        self.settings = None

    def initialise(self, in_dict, restart=False):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict, self.settings_types, self.settings_default)
        self.settings = copy.deepcopy(self.in_dict)

        self.x0 = self.in_dict['coords_0'][0]
        self.y0 = self.in_dict['coords_0'][1]
        self.z0 = self.in_dict['coords_0'][2]
        self.x1 = self.in_dict['coords_1'][0]
        self.y1 = self.in_dict['coords_1'][1]
        self.z1 = self.in_dict['coords_1'][2]
        self.dx = self.in_dict['spacing'][0]
        self.dy = self.in_dict['spacing'][1]
        self.dz = self.in_dict['spacing'][2]

    def generate(self, params):
        if self.settings['moving']:
            for_pos = params['for_pos']
        else:
            for_pos = np.zeros((3,))
        nx = np.abs(int((self.x1-self.x0)/self.dx + 1))
        ny = np.abs(int((self.y1-self.y0)/self.dy + 1))
        nz = np.abs(int((self.z1-self.z0)/self.dz + 1))

        xarray = np.linspace(self.x0, self.x1, nx) + for_pos[0]
        yarray = np.linspace(self.y0, self.y1, ny) + for_pos[1]
        zarray = np.linspace(self.z0, self.z1, nz) + for_pos[2]
        grid = []
        for iz in range(nz):
            grid.append(np.zeros((3, nx, ny), dtype=ct.c_double))
            for ix in range(nx):
                for iy in range(ny):
                    grid[iz][0, ix, iy] = xarray[ix]
                    grid[iz][1, ix, iy] = yarray[iy]
                    grid[iz][2, ix, iy] = zarray[iz]


        vtk_info = tvtk.RectilinearGrid()
        vtk_info.dimensions = np.array([nx, ny, nz], dtype=int)
        vtk_info.x_coordinates = xarray
        vtk_info.y_coordinates = yarray
        vtk_info.z_coordinates = zarray

        return vtk_info, grid
