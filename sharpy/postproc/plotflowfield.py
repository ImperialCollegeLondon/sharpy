import os
import numpy as np
from tvtk.api import tvtk, write_data
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.generator_interface as gen_interface
import sharpy.utils.settings as settings_utils
import sharpy.aero.utils.uvlmlib as uvlmlib
import ctypes as ct
from sharpy.utils.constants import vortex_radius_def


@solver
class PlotFlowField(BaseSolver):
    """
    Plots the flow field in Paraview and computes the velocity at a set of points in a grid.
    """
    solver_id = 'PlotFlowField'
    solver_classification = 'post-processor'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    settings_types['postproc_grid_generator'] = 'str'
    settings_default['postproc_grid_generator'] = 'GridBox'
    settings_description['postproc_grid_generator'] = 'Generator used to create grid and plot flow field'
    settings_options['postproc_grid_generator'] = ['GridBox']

    settings_types['postproc_grid_input'] = 'dict'
    settings_default['postproc_grid_input'] = dict()
    settings_description['postproc_grid_input'] = 'Dictionary containing settings for ``postproc_grid_generator``.'

    settings_types['velocity_field_generator'] = 'str'
    settings_default['velocity_field_generator'] = 'SteadyVelocityField'
    settings_description['velocity_field_generator'] = 'Chosen velocity field generator'

    settings_types['velocity_field_input'] = 'dict'
    settings_default['velocity_field_input'] = dict()
    settings_description['velocity_field_input'] = 'Dictionary containing settings for the selected ``velocity_field_generator``.'

    settings_types['dt'] = 'float'
    settings_default['dt'] = 0.1
    settings_description['dt'] = 'Time step.'

    settings_types['include_external'] = 'bool'
    settings_default['include_external'] = True
    settings_description['include_external'] = 'Include external velocities.'

    settings_types['include_induced'] = 'bool'
    settings_default['include_induced'] = True
    settings_description['include_induced'] = 'Include induced velocities.'

    settings_types['stride'] = 'int'
    settings_default['stride'] = 1
    settings_description['stride'] = 'Number of time steps between plots.'

    settings_types['num_cores'] = 'int'
    settings_default['num_cores'] = 1
    settings_description['num_cores'] = 'Number of cores to use.'

    settings_types['vortex_radius'] = 'float'
    settings_default['vortex_radius'] = vortex_radius_def
    settings_description['vortex_radius'] = 'Distance below which inductions are not computed.'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description, settings_options)

    def __init__(self):
        self.settings = None
        self.data = None
        self.folder = None
        self.caller = None

    def initialise(self, data, custom_settings=None, caller=None, restart=False):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings_utils.to_custom_types(self.settings,
                           self.settings_types,
                           self.settings_default,
                           self.settings_options)

        self.folder = data.output_folder + '/' + 'GenerateFlowField/'
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)

        # init velocity generator
        velocity_generator_type = gen_interface.generator_from_string(
            self.settings['velocity_field_generator'])
        self.velocity_generator = velocity_generator_type()
        self.velocity_generator.initialise(self.settings['velocity_field_input'], restart=restart)

        # init postproc grid generator
        postproc_grid_generator_type = gen_interface.generator_from_string(
            self.settings['postproc_grid_generator'])
        self.postproc_grid_generator = postproc_grid_generator_type()
        self.postproc_grid_generator.initialise(self.settings['postproc_grid_input'], restart=restart)
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
                                              't': ts*self.settings['dt'],
                                              'ts': ts,
                                              'dt': self.settings['dt'],
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

        filename = self.folder + "VelocityField_" + '%06u' % ts + ".vtk"
        write_data(vtk_info, filename)

    def run(self, **kwargs):

        online = settings_utils.set_value_or_default(kwargs, 'online', False)

        if online:
            if divmod(self.data.ts, self.settings['stride'])[1] == 0:
                self.output_velocity_field(len(self.data.structure.timestep_info) - 1)
        else:
            for ts in range(0, len(self.data.structure.timestep_info)):
                if not self.data.structure.timestep_info[ts] is None:
                    self.output_velocity_field(ts)
        return self.data
