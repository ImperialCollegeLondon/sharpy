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
from sharpy.utils.constants import vortex_radius_def


@solver
class PlotFlowField(BaseSolver):
    solver_id = 'PlotFlowField'

    def __init__(self):
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['postproc_grid_generator'] = 'str'
        self.settings_default['postproc_grid_generator'] = 'box'

        self.settings_types['postproc_grid_input'] = 'dict'
        self.settings_default['postproc_grid_input'] = dict()

        self.settings_types['velocity_field_generator'] = 'str'
        self.settings_default['velocity_field_generator'] = 'SteadyVelocityField'

        self.settings_types['velocity_field_input'] = 'dict'
        self.settings_default['velocity_field_input'] = dict()

        self.settings_types['dt'] = 'float'
        self.settings_default['dt'] = 0.1

        self.settings_types['include_external'] = 'bool'
        self.settings_default['include_external'] = True

        self.settings_types['include_induced'] = 'bool'
        self.settings_default['include_induced'] = True

        self.settings_types['stride'] = 'int'
        self.settings_default['stride'] = 1

        self.settings_types['num_cores'] = 'int'
        self.settings_default['num_cores'] = 1

        self.settings_types['vortex_radius'] = 'float'
        self.settings_default['vortex_radius'] = vortex_radius_def
        # settings_description['vortex_radius'] = 'Distance below which inductions are not computed'

        self.settings = None
        self.data = None
        self.dir = 'output/'
        self.caller = None

    def initialise(self, data, custom_settings=None, caller=None):
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

        # init postproc grid generator
        postproc_grid_generator_type = gen_interface.generator_from_string(
            self.settings['postproc_grid_generator'])
        self.postproc_grid_generator = postproc_grid_generator_type()
        self.postproc_grid_generator.initialise(self.settings['postproc_grid_input'])
        self.caller = caller

    def output_velocity_field(self, ts):
        # Notice that SHARPy utilities deal with several two-dimensional surfaces
        # To be able to build 3D volumes, I will make use of the surface index as
        # the third index in space
        # It does not apply to the 'u' array because this way it is easier to
        # write it in paraview

        # Generate the grid
        vtk_info, grid = self.postproc_grid_generator.generate({
                'for_pos': self.data.structure.timestep_info[ts].for_pos[0:3]})

        # Compute the induced velocities
        nx = grid[0].shape[1]
        ny = grid[0].shape[2]
        nz = len(grid)

        array_counter = 0
        u_ind = np.zeros((nx, ny, nz, 3), dtype=float)
        if self.settings['include_induced']:
            target_triads = np.zeros((nx*ny*nz, 3))
            ipoint = -1
            for iz in range(nz):
                for ix in range(nx):
                    for iy in range(ny):
                        ipoint += 1
                        target_triads[ipoint, :] = grid[iz][:, ix, iy].astype(dtype=ct.c_double, order='F', copy=True)

            u_ind_points = uvlmlib.uvlm_calculate_total_induced_velocity_at_points(self.data.aero.timestep_info[ts],
                                                                                                      target_triads,
                                                                                                      self.settings['vortex_radius'],
                                                                                                      self.data.structure.timestep_info[ts].for_pos[0:3],
                                                                                                      self.settings['num_cores'])
            ipoint = -1
            for iz in range(nz):
                for ix in range(nx):
                    for iy in range(ny):
                        ipoint += 1
                        u_ind[ix, iy, iz, :] = u_ind_points[ipoint, :]

            # Write the data
            vtk_info.point_data.add_array(u_ind.reshape((-1, u_ind.shape[-1]), order='F')) # Reshape the array except from the last dimension
            vtk_info.point_data.get_array(array_counter).name = 'induced_velocity'
            vtk_info.point_data.update()
            array_counter += 1

        # Add the external velocities
        u_ext_out = np.zeros((nx, ny, nz, 3), dtype=float)

        if self.settings['include_external']:
            u_ext = []
            for iz in range(nz):
                u_ext.append(np.zeros((3, nx, ny), dtype=ct.c_double))
            self.velocity_generator.generate({'zeta': grid,
                                              'override': True,
                                              't': ts*self.settings['dt'].value,
                                              'ts': ts,
                                              'dt': self.settings['dt'].value,
                                              'for_pos': 0*self.data.structure.timestep_info[ts].for_pos},
                                             u_ext)
            for iz in range(nz):
                for ix in range(nx):
                    for iy in range(ny):
                        u_ext_out[ix, iy, iz, :] += u_ext[iz][:, ix, iy]

            # Write the data
            vtk_info.point_data.add_array(u_ext_out.reshape((-1, u_ext_out.shape[-1]), order='F')) # Reshape the array except from the last dimension
            vtk_info.point_data.get_array(array_counter).name = 'external_velocity'
            vtk_info.point_data.update()
            array_counter += 1

        # add the data
        u = u_ind + u_ext_out

        # Write the data
        vtk_info.point_data.add_array(u.reshape((-1, u.shape[-1]), order='F')) # Reshape the array except from the last dimension
        vtk_info.point_data.get_array(array_counter).name = 'velocity'
        vtk_info.point_data.update()
        array_counter += 1

        filename = self.dir + "VelocityField_" + '%06u' % ts + ".vtk"
        write_data(vtk_info, filename)

    def run(self, online=False):
        if online:
            if divmod(self.data.ts, self.settings['stride'].value)[1] == 0:
                self.output_velocity_field(len(self.data.structure.timestep_info) - 1)
        else:
            for ts in range(0, len(self.data.structure.timestep_info)):
                if not self.data.structure.timestep_info[ts] is None:
                    self.output_velocity_field(ts)
        return self.data
