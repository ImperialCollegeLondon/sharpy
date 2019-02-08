import numpy as np
import scipy.interpolate as interpolate
import h5py as h5
import os
from xml.dom import minidom
from lxml import objectify

import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings
import sharpy.utils.cout_utils as cout


@generator_interface.generator
class TurbVelocityField(generator_interface.BaseGenerator):
    r"""
    Turbulent Velocity Field Generator

    ``TurbVelocitityField`` is a class inherited from ``BaseGenerator``

    The ``TurbVelocitityField`` class generates a velocity field based on the input from an [XDMF](http://www.xdmf.org) file.
    It supports time-dependant fields as well as frozen turbulence.

    To call this generator, the ``generator_id = TurbVelocityField`` shall be used.
    This is parsed as the value for the ``velocity_field_generator`` key in the desired aerodynamic solver's settings.

    Supported files:
        - `field_id.xdmf`: Steady or Unsteady XDMF file

    This generator also performs time interpolation between two different time steps. For now, only linear interpolation is possible.

    Space interpolation is done through `scipy.interpolate` trilinear interpolation. However, turbulent fields are
    read directly from the binary file and not copied into memory. This is performed using `np.memmap`.
    The overhead of this procedure is ~18% for the interpolation stage, however, initially reading the binary velocity field
    (which will be much more common with time-domain simulations) is faster by a factor of 1e4.
    Also, memory savings are quite substantial: from 6Gb for a typical field to a handful of megabytes for the whole program.

    Args:
        in_dict (dict): Input data in the form of dictionary. See acceptable entries below:

            ===================  ===============  ===============================================================  ===================
            Name                 Type             Description                                                      Default
            ===================  ===============  ===============================================================  ===================
            ``print_info``       ``bool``         Output solver-specific information in runtime.                   ``True``
            ``turbulent_field``  ``str``          XDMF file path of the velocity file.                             ``None``
            ``offset``           ``list(float)``  Spatial offset in the 3 dimensions                               ``[0.0, 0.0, 0.0]``
            ``centre_y``         ``bool``         Flag for changing the domain to [-y_max/2, y_max/2].             ``True``
            ``periodicity``      ``str``          Axes in which periodicity is enforced                            ``xy``
            ``frozen``           ``bool``         If True, the turbulent field will not be updated in time.        ``True``
            ===================  ===============  ===============================================================  ===================

    Attributes:

    See Also:
        .. py:class:: sharpy.utils.generator_interface.BaseGenerator

    """
    generator_id = 'TurbVelocityField'

    def __init__(self):
        self.in_dict = dict()
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['print_info'] = 'bool'
        self.settings_default['print_info'] = True

        self.settings_types['turbulent_field'] = 'str'
        self.settings_default['turbulent_field'] = None

        self.settings_types['offset'] = 'list(float)'
        self.settings_default['offset'] = np.zeros((3,))

        self.settings_types['centre_y'] = 'bool'
        self.settings_default['centre_y'] = True

        self.settings_types['periodicity'] = 'str'
        self.settings_default['periodicity'] = 'xy'

        self.settings_types['frozen'] = 'bool'
        self.settings_default['frozen'] = True

        self.settings = dict()

        self.file = None
        self.extension = None

        self.bbox = None
        self.interpolator = 3*[None]
        self.x_periodicity = False
        self.y_periodicity = False

        self.grid_data = dict()

    def initialise(self, in_dict):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict, self.settings_types, self.settings_default)
        self.settings = self.in_dict

        _, self.extension = os.path.splitext(self.settings['turbulent_field'])

        if self.extension is '.h5':
            self.read_btl(self.settings['turbulent_field'])
        if self.extension in '.xdmf':
            self.read_xdmf(self.settings['turbulent_field'])

        if 'z' in self.settings['periodicity']:
            raise ValueError('Periodicitiy setting in TurbVelocityField cannot be z.\n A turbulent boundary layer is not periodic in the z direction!')

        if 'x' in self.settings['periodicity']:
            self.x_periodicity = True
        if 'y' in self.settings['periodicity']:
            self.y_periodicity = True

    # these functions need to define the interpolators
    def read_btl(self, in_file):
        raise NotImplementedError('The BTL reader is not up to date!')

    def read_xdmf(self, in_file):
        # store route of file for the other files
        route = os.path.dirname(os.path.abspath(in_file))

        # file to string
        with open(in_file, 'r') as self.file:
            data = self.file.read().replace('\n', '')

        # parse data
        tree = objectify.fromstring(data)

        # mesh dimensions
        self.grid_data['dimensions'] = np.fromstring(tree.Domain.Topology.attrib['Dimensions'],
                                                     sep=' ',
                                                     count=3,
                                                     dtype=int)

        # origin
        self.grid_data['origin'] = np.fromstring(tree.Domain.Geometry.DataItem[0].text,
                                                 sep=' ',
                                                 count=int(tree.Domain.Geometry.DataItem[0].attrib['Dimensions']),
                                                 dtype=float)

        # dxdydz
        # because of how XDMF does it, it is actually dzdydx
        self.grid_data['dxdydz'] = np.fromstring(tree.Domain.Geometry.DataItem[1].text,
                                                 sep=' ',
                                                 count=int(tree.Domain.Geometry.DataItem[1].attrib['Dimensions']),
                                                 dtype=float)
        # now onto the grid
        self.grid_data['n_grid'] = len(tree.Domain.Grid.Grid)
        self.grid_data['grid'] = [dict()]*self.grid_data['n_grid']
# TODO make sure this read all the tsteps
        for i, i_grid in enumerate(tree.Domain.Grid.Grid):
            # cycle through attributes
            for k_attrib, v_attrib in i_grid.attrib.items():
                self.grid_data['grid'][i][k_attrib] = v_attrib
            # get Attributes (upper case A is not a mistake)
            for i_attrib, attrib in enumerate(i_grid.Attribute):
                self.grid_data['grid'][i_grid][attrib.attrib['Name']] = dict()
                self.grid_data['grid'][i_grid][attrib.attrib['Name']]['file'] = attrib.DataItem.text.replace(' ', '')

        # now we have the file names and the dimensions
        self.grid_data['initial_x_grid'] = np.array(np.arange(0, self.grid_data['dimensions'][2]))*self.grid_data['dxdydz'][2]
        # z in the file is -y for us in sharpy (y_sharpy = right)
        self.initial_y_grid = -np.array(np.arange(0, self.grid_data['dimensions'][0]))*self.grid_data['dxdydz'][0]
        # y in the file is z for us in sharpy (up)
        self.initial_z_grid = np.array(np.arange(0, self.grid_data['dimensions'][1]))*self.grid_data['dxdydz'][1]

        # the domain now goes:
        # x \in [0, dimensions[0]*dx]
        # y \in [-dimensions[2]*dz, 0]
        # z \in [0, dimensions[1]*dy]
        centre_y_offset = 0.
        if self.settings['centre_y']:
            centre_y_offset = -0.5*(self.initial_y_grid[-1] - self.initial_y_grid[0])

        self.initial_x_grid += self.settings['offset'][0] + origin[0]
        self.initial_x_grid -= np.max(self.initial_x_grid)
        self.initial_y_grid += self.settings['offset'][1] + origin[1] + centre_y_offset
        self.initial_y_grid = self.initial_y_grid[::-1]
        self.initial_z_grid += self.settings['offset'][2] + origin[2]

        self.bbox = self.get_field_bbox(self.initial_x_grid,
                                        self.initial_y_grid,
                                        self.initial_z_grid)
        if self.settings['print_info']:
            cout.cout_wrap('The domain bbox is:', 1)
            cout.cout_wrap(' x = [' + str(self.bbox[0, 0]) + ', ' + str(self.bbox[0, 1]) + ']', 1)
            cout.cout_wrap(' y = [' + str(self.bbox[1, 0]) + ', ' + str(self.bbox[1, 1]) + ']', 1)
            cout.cout_wrap(' z = [' + str(self.bbox[2, 0]) + ', ' + str(self.bbox[2, 1]) + ']', 1)

    def read_grid(i_grid):
        if n_grid > 1:
            cout.cout_wrap('CAREFUL: n_grid > 1, but we don\'t support time series yet')



        # now we load the velocities (one by one, so we don't store all the
        # info more than once at the same time)
        velocities = ['ux', 'uz', 'uy']
        velocities_mult = np.array([1.0, -1.0, 1.0])
        for i_dim in range(3):
            file_name = grid[0][velocities[i_dim]]['file']

            # load file
            with open(route + '/' + file_name, "rb") as ufile:
                vel = np.fromfile(ufile, dtype=np.float64)

            vel = np.swapaxes(vel.reshape((dimensions[2], dimensions[1], dimensions[0]),
                                          order='F')*velocities_mult[i_dim], 1, 2)

            self.interpolator[i_dim] = self.init_interpolator(vel,
                                                              self.initial_x_grid,
                                                              self.initial_y_grid,
                                                              self.initial_z_grid,
                                                              i_dim=i_dim)

    def generate(self, params, uext):
        zeta = params['zeta']
        for_pos = params['for_pos']
        t = params['t']

        self.interpolate_zeta(zeta,
                              for_pos,
                              uext)

    @staticmethod
    def get_field_bbox(x_grid, y_grid, z_grid):
        bbox = np.zeros((3, 2))
        bbox[0, :] = [np.min(x_grid), np.max(x_grid)]
        bbox[1, :] = [np.min(y_grid), np.max(y_grid)]
        bbox[2, :] = [np.min(z_grid), np.max(z_grid)]
        return bbox

    def init_interpolator(self, data, x_grid, y_grid, z_grid, i_dim=None):
        if i_dim is None:
            interpolator = list()
            for i_dim in range(3):
                interpolator.append(interpolate.RegularGridInterpolator((z_grid, y_grid, x_grid),
                                                                        data[i_dim, :, :, :],
                                                                        bounds_error=False,
                                                                        fill_value=0.0))
        else:
            interpolator = interpolate.RegularGridInterpolator((x_grid, y_grid, z_grid),
                                                               data,
                                                               bounds_error=False,
                                                               fill_value=0.0)
        return interpolator

    def interpolate_zeta(self, zeta, for_pos, u_ext):
        for i_dim in range(3):
            for isurf in range(len(zeta)):
                _, n_m, n_n = zeta[isurf].shape
                for i_m in range(n_m):
                    for i_n in range(n_n):
                        coord = self.apply_periodicity(zeta[isurf][:, i_m, i_n] + for_pos[0:3])
                        try:
                            u_ext[isurf][i_dim, i_m, i_n] = self.interpolator[i_dim](coord)
                        except ValueError:
                            print(coord)
                            raise ValueError()

    def apply_periodicity(self, coord):
        new_coord = coord.copy()
        if self.x_periodicity:
            i = 0
            # x in interval:
            if self.bbox[i, 0] <= new_coord[i] <= self.bbox[i, 1]:
                pass
            # lower than min bbox
            elif new_coord[i] < self.bbox[i, 0]:
                temp = divmod(new_coord[i], self.bbox[i, 0])[1]
                if np.isnan(temp):
                    pass
                else:
                    new_coord[i] = temp

            # greater than max bbox
            elif new_coord[i] > self.bbox[i, 1]:
                temp = divmod(new_coord[i], self.bbox[i, 1])[1]
                if np.isnan(temp):
                    pass
                else:
                    new_coord[i] = temp

        if self.y_periodicity:
            i = 1
            # y in interval:
            if self.bbox[i, 0] <= new_coord[i] <= self.bbox[i, 1]:
                pass
            # lower than min bbox
            elif new_coord[i] < self.bbox[i, 0]:
                temp = divmod(new_coord[i], self.bbox[i, 0])[1]
                if np.isnan(temp):
                    pass
                else:
                    new_coord[i] = temp
            # greater than max bbox
            elif new_coord[i] > self.bbox[i, 1]:
                temp = divmod(new_coord[i], self.bbox[i, 1])[1]
                if np.isnan(temp):
                    pass
                else:
                    new_coord[i] = temp

        return new_coord



