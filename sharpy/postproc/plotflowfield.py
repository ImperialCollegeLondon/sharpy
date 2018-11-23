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
import sharpy.utils.generator_interface as gen_interface
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

        self.settings_types['velocity_field_generator'] = 'str'
        self.settings_default['velocity_field_generator'] = 'SteadyVelocityField'

        self.settings_types['velocity_field_input'] = 'dict'
        self.settings_default['velocity_field_input'] = dict()

        self.settings_types['dt'] = 'float'
        self.settings_default['dt'] = 0.1

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

        # init velocity generator
        velocity_generator_type = gen_interface.generator_from_string(
            self.settings['velocity_field_generator'])
        self.velocity_generator = velocity_generator_type()
        self.velocity_generator.initialise(self.settings['velocity_field_input'])

    def run(self, online=False):

        # Notice that SHARPy utilities deal with several two-dimensional surfaces
        # To be able to build 3D volumes, I will make use of the surface index as
        # the third index in space

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
        zeta = []
        u_ext = []
        for iz in range(grid.shape[2]):
            zeta.append(np.zeros((3,grid.shape[0],grid.shape[1]), dtype=float))
            u_ext.append(np.zeros((3,grid.shape[0],grid.shape[1]), dtype=float))
            for ix in range(grid.shape[0]):
                for iy in range(grid.shape[1]):
                    zeta[iz][:,ix,iy] = grid[ix, iy, iz, :]
                    u_ext[iz][:,ix,iy] = 0.0

        self.velocity_generator.generate({'zeta': zeta,
                                      'override': True,
                                      't': self.data.ts*self.settings['dt'].value,
                                      'ts': self.data.ts,
                                      'dt': self.settings['dt'].value,
                                      'for_pos': self.data.structure.timestep_info[-1].for_pos},
                                      u_ext)

        # Add both velocities
        for ix in range(grid.shape[0]):
            for iy in range(grid.shape[1]):
                for iz in range(grid.shape[2]):
                    u[ix, iy, iz, :] += u_ext[iz][:,ix,iy]

        # Write the data
        vtk_info.point_data.add_array(u.reshape((-1, u.shape[-1]), order='F')) # Reshape the array except from the last dimension
        vtk_info.point_data.get_array(0).name = 'Velocity'
        vtk_info.point_data.update()

        it = len(self.data.structure.timestep_info) - 1
        filename = self.dir + "VelocityFied_" + '%06u' % it + ".vtk"
        write_data(vtk_info, filename)
