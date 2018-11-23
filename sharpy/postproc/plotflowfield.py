"""
PlotFlowField

Computes the flow velocity at a set of points (grid)

Args:

Returns:

Examples:

Notes:

"""
import os
import numpy as np
from tvtk.api import tvtk, write_data
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.aero.utils.uvlmlib as uvlmlib
import ctypes as ct


@solver
class PlotFlowField(BaseSolver):
    solver_id = 'PlotFlowField'

    def __init__(self):
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['grid_generation_method'] = 'str'
        self.settings_default['grid_generation_method'] = 'box'

        self.settings_types['options'] = dict()

        self.settings_types['u_inf'] = 'float'
        self.settings_default['u_inf'] = None

        self.settings_types['u_inf_direction'] = 'list(float)'
        self.settings_default['u_inf_direction'] = np.array([1.0, 0, 0])

        self.settings = None
        self.data = None
        self.dir = 'output/'

    def initialise(self, data, custom_settings=None):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        self.dir =   self.data.case_route + 'output/' + self.data.case_name + '/' + 'GenerateFlowField/'
        if not os.path.isdir(self.dir):
            os.makedirs(self.dir)

    def run(self, online=False):

        # Generate the grid
        if self.settings['grid_generation_method'] == 'box':

            x0 = float(self.settings['options']['x0'])
            y0 = float(self.settings['options']['y0'])
            z0 = float(self.settings['options']['z0'])
            x1 = float(self.settings['options']['x1'])
            y1 = float(self.settings['options']['y1'])
            z1 = float(self.settings['options']['z1'])
            dx = float(self.settings['options']['dx'])
            dy = float(self.settings['options']['dy'])
            dz = float(self.settings['options']['dz'])

            nx = int((x1-x0)/dx + 1)
            ny = int((y1-y0)/dy + 1)
            nz = int((z1-z0)/dz + 1)

            xarray = np.linspace(x0,x1,nx)
            yarray = np.linspace(y0,y1,ny)
            zarray = np.linspace(z0,z1,nz)
            grid = np.zeros((nx,ny,nz), dtype=(ct.c_double,3))
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        grid[ix, iy, iz, 0] = xarray[ix]
                        grid[ix, iy, iz, 1] = yarray[iy]
                        grid[ix, iy, iz, 2] = zarray[iz]


            vtk_info = tvtk.RectilinearGrid()
            vtk_info.dimensions = np.array([nx, ny, nz], dtype=int)
            vtk_info.x_coordinates = xarray
            vtk_info.y_coordinates = yarray
            vtk_info.z_coordinates = zarray

        # Compute the induced velocities
        u = np.zeros_like(grid, dtype=ct.c_double)
        for ix in range(grid.shape[0]):
            for iy in range(grid.shape[1]):
                for iz in range(grid.shape[2]):
                    u[ix, iy, iz, :] = uvlmlib.uvlm_calculate_total_induced_velocity_at_point(self.data.aero.timestep_info[-1],
                                                           grid[ix, iy, iz, :])

        # Add the external velocities
        # TODO: broad this to any kind of flow velocity generation
        for ix in range(grid.shape[0]):
            for iy in range(grid.shape[1]):
                for iz in range(grid.shape[2]):
                    u[ix, iy, iz, :] += self.settings['u_inf']*self.settings['u_inf_direction']

        # Write the data
        vtk_info.point_data.add_array(u.reshape((-1, u.shape[-1]), order='F')) # Reshape the array except from the last dimension
        vtk_info.point_data.get_array(0).name = 'Velocity'
        vtk_info.point_data.update()

        it = len(self.data.structure.timestep_info) - 1
        filename = self.dir + "VelocityFied_" + '%06u' % it + ".vtk"
        write_data(vtk_info, filename)
