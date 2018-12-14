import numpy as np

import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings
import numpy as np
from tvtk.api import tvtk, write_data
import ctypes as ct


@generator_interface.generator
class MovingGridBox(generator_interface.BaseGenerator):
    """
    MovingGridBox

    Generate a grid within a box to be used to generate the flow field during the postprocessing
    MovingGridBox follows the for_pos of the aircraft

    """
    generator_id = 'MovingGridBox'

    def __init__(self):
        self.in_dict = dict()
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['x0'] = 'float'
        self.settings_default['x0'] = 0.

        self.settings_types['y0'] = 'float'
        self.settings_default['y0'] = 0.

        self.settings_types['z0'] = 'float'
        self.settings_default['z0'] = 0.

        self.settings_types['x1'] = 'float'
        self.settings_default['x1'] = 10.

        self.settings_types['y1'] = 'float'
        self.settings_default['y1'] = 0.

        self.settings_types['z1'] = 'float'
        self.settings_default['z1'] = 10.

        self.settings_types['dx'] = 'float'
        self.settings_default['dx'] = 1.

        self.settings_types['dy'] = 'float'
        self.settings_default['dy'] = 1.

        self.settings_types['dz'] = 'float'
        self.settings_default['dz'] = 1.


    def initialise(self, in_dict):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict, self.settings_types, self.settings_default)

        self.x0 = self.in_dict['x0']
        self.y0 = self.in_dict['y0']
        self.z0 = self.in_dict['z0']
        self.x1 = self.in_dict['x1']
        self.y1 = self.in_dict['y1']
        self.z1 = self.in_dict['z1']
        self.dx = self.in_dict['dx']
        self.dy = self.in_dict['dy']
        self.dz = self.in_dict['dz']

    def generate(self, params):
        for_pos = params['for_pos']
        nx = np.abs(int((self.x1.value-self.x0.value)/self.dx.value + 1))
        ny = np.abs(int((self.y1.value-self.y0.value)/self.dy.value + 1))
        nz = np.abs(int((self.z1.value-self.z0.value)/self.dz.value + 1))

        xarray = np.linspace(self.x0.value, self.x1.value, nx) + for_pos[0]
        yarray = np.linspace(self.y0.value, self.y1.value, ny) + for_pos[1]
        zarray = np.linspace(self.z0.value, self.z1.value, nz) + for_pos[2]
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
