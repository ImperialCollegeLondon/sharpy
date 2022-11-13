import numpy as np
import scipy.interpolate as interpolate
import h5py as h5
import os
from lxml import objectify, etree

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

    Attributes:

    See Also:
        .. py:class:: sharpy.utils.generator_interface.BaseGenerator

    """
    generator_id = 'TurbVelocityField'
    generator_classification = 'velocity-field'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['print_info'] = 'bool'
    settings_default['print_info'] = True
    settings_description['print_info'] = 'Output solver-specific information in runtime.'

    settings_types['turbulent_field'] = 'str'
    settings_default['turbulent_field'] = None
    settings_description['turbulent_field'] = 'XDMF file path of the velocity field'

    settings_types['offset'] = 'list(float)'
    settings_default['offset'] = np.zeros((3,))
    settings_description['offset'] = 'Spatial offset in the 3 dimensions'

    settings_types['centre_y'] = 'bool'
    settings_default['centre_y'] = True
    settings_description['centre_y'] = 'Flat for changing the domain to [``-y_max/2``, ``y_max/2``]'

    settings_types['periodicity'] = 'str'
    settings_default['periodicity'] = 'xy'
    settings_description['periodicity'] = 'Axes in which periodicity is enforced'

    settings_types['frozen'] = 'bool'
    settings_default['frozen'] = True
    settings_description['frozen'] = 'If ``True``, the turbulent field will not be updated in time'

    settings_types['store_field'] = 'bool'
    settings_default['store_field'] = False
    settings_description['store_field'] = 'If ``True``, the xdmf snapshots are stored in memory. Only two at a time for the linear interpolation'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.in_dict = dict()

        self.settings = dict()

        self.file = None
        self.extension = None

        self.grid_data = dict()

        self.interpolator = 3*[None]
        self.x_periodicity = False
        self.y_periodicity = False

        # variables for interpolator wrapper
        self._t0 = -1
        self._t1 = -1
        self._it0 = -1
        self._it1 = -1
        self._interpolator0 = None
        self._interpolator1 = None
        self.coeff = 0.
        self.double_initialisation = True
        self.vel_holder0 = 3*[None]
        self.vel_holder1 = 3*[None]

    def initialise(self, in_dict, restart=False):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict, self.settings_types, self.settings_default)
        self.settings = self.in_dict

        _, self.extension = os.path.splitext(self.settings['turbulent_field'])

        if self.extension == '.h5':
            self.read_btl(self.settings['turbulent_field'])
        if self.extension == '.xdmf':
            self.read_xdmf(self.settings['turbulent_field'])

        if 'z' in self.settings['periodicity']:
            raise ValueError('Periodicitiy setting in TurbVelocityField cannot be z.\n A turbulent boundary layer is not periodic in the z direction!')

        if 'x' in self.settings['periodicity']:
            self.x_periodicity = True
        if 'y' in self.settings['periodicity']:
            self.y_periodicity = True

    # ADC: VERY VERY UGLY. NEED A BETTER WAY
    def interpolator_wrapper0(self, coords, i_dim=0):
        coeff = self.get_coeff()
        return (1.0 - self.coeff)*self._interpolator0[i_dim](coords) + self.coeff*self._interpolator1[i_dim](coords)
    def interpolator_wrapper1(self, coords, i_dim=1):
        coeff = self.get_coeff()
        return (1.0 - self.coeff)*self._interpolator0[i_dim](coords) + self.coeff*self._interpolator1[i_dim](coords)
    def interpolator_wrapper2(self, coords, i_dim=2):
        coeff = self.get_coeff()
        return (1.0 - self.coeff)*self._interpolator0[i_dim](coords) + self.coeff*self._interpolator1[i_dim](coords)

    def get_coeff(self):
        return self.coeff

    def init_interpolator(self):
        if self.settings['frozen']:
            self.interpolator = self._interpolator0
            return

        # continuing the ugliness
        self.interpolator[0] = self.interpolator_wrapper0
        self.interpolator[1] = self.interpolator_wrapper1
        self.interpolator[2] = self.interpolator_wrapper2

    # these functions need to define the interpolators
    def read_btl(self, in_file):
        """
        Legacy function, not using the custom format based on HDF5 anymore.
        """
        raise NotImplementedError('The BTL reader is not up to date!')

    def read_xdmf(self, in_file):
        """
        Reads the xml file `<case_name>.xdmf`. Writes the self.grid_data data structure
        with all the information necessary.

        Note: this function does not load any turbulence data (such as ux000, ...),
        it only reads the header information contained in the xdmf file.
        """
        # store route of file for the other files
        self.route = os.path.dirname(os.path.abspath(in_file))

        # file to string
        with open(in_file, 'r') as self.file:
            data = self.file.read().replace('\n', '')

        # parse data
        # this next line is necessary to avoid problems with parsing in the Time part:
        # <!--Start....
        # 0.0, 1.0 ...
        # see https://stackoverflow.com/a/18313932
        parser = objectify.makeparser(remove_comments=True)
        tree = objectify.fromstring(data, parser=parser)

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
        self.grid_data['dxdydz'] = (
            np.fromstring(tree.Domain.Geometry.DataItem[1].text,
                          sep=' ',
                          count=int(tree.Domain.Geometry.DataItem[1].attrib['Dimensions']),
                          dtype=float))

        # now onto the grid
        # time information
        # [0] is start, [1] is stride
        self.grid_data['time'] = np.fromstring(tree.Domain.Grid.Time.DataItem.text,
                                               sep=' ',
                                               count=2,
                                               dtype=float)
        self.grid_data['n_grid'] = len(tree.Domain.Grid.Grid)
        # self.grid_data['grid'] = [dict()]*self.grid_data['n_grid']
        self.grid_data['grid'] = []
        for i, i_grid in enumerate(tree.Domain.Grid.Grid):
            self.grid_data['grid'].append(dict())
            # cycle through attributes
            for k_attrib, v_attrib in i_grid.attrib.items():
                self.grid_data['grid'][i][k_attrib] = v_attrib
            # get Attributes (upper case A is not a mistake)
            for i_attrib, attrib in enumerate(i_grid.Attribute):
                self.grid_data['grid'][i][attrib.attrib['Name']] = dict()
                self.grid_data['grid'][i][attrib.attrib['Name']]['file'] = (
                    attrib.DataItem.text.replace(' ', ''))
                if attrib.DataItem.attrib['Precision'].strip() == '4':
                    self.grid_data['grid'][i][attrib.attrib['Name']]['Precision'] = np.float32
                elif attrib.DataItem.attrib['Precision'].strip() == '8':
                    self.grid_data['grid'][i][attrib.attrib['Name']]['Precision'] = np.float64

        # now we have the file names and the dimensions
        self.grid_data['initial_x_grid'] = np.array(np.arange(0,
            self.grid_data['dimensions'][2]))*self.grid_data['dxdydz'][2]
        # z in the file is -y for us in sharpy (y_sharpy = right)
        self.grid_data['initial_y_grid'] = np.array(np.arange(0,
            self.grid_data['dimensions'][1]))*self.grid_data['dxdydz'][1]
        # y in the file is z for us in sharpy (up)
        self.grid_data['initial_z_grid'] = np.array(np.arange(0,
            self.grid_data['dimensions'][0]))*self.grid_data['dxdydz'][0]

        # the domain now goes:
        # x \in [0, dimensions[0]*dx]
        # y \in [-dimensions[2]*dz, 0]
        # z \in [0, dimensions[1]*dy]
        centre_z_offset = 0.
        if self.settings['centre_y']:
            centre_z_offset = -0.5*(self.grid_data['initial_z_grid'][-1] - self.grid_data['initial_z_grid'][0])

        self.grid_data['initial_x_grid'] += self.settings['offset'][0] + self.grid_data['origin'][0]
        self.grid_data['initial_x_grid'] -= np.max(self.grid_data['initial_x_grid'])
        self.grid_data['initial_y_grid'] += self.settings['offset'][1] + self.grid_data['origin'][1]
        self.grid_data['initial_z_grid'] += self.settings['offset'][2] + self.grid_data['origin'][2] + centre_z_offset

        self.bbox = self.get_field_bbox(self.grid_data['initial_x_grid'],
                                        self.grid_data['initial_y_grid'],
                                        self.grid_data['initial_z_grid'],
                                        frame='G')
        if self.settings['print_info']:
            cout.cout_wrap('The domain bbox is:', 1)
            cout.cout_wrap(' x = [' + str(self.bbox[0, 0]) + ', ' + str(self.bbox[0, 1]) + ']', 1)
            cout.cout_wrap(' y = [' + str(self.bbox[1, 0]) + ', ' + str(self.bbox[1, 1]) + ']', 1)
            cout.cout_wrap(' z = [' + str(self.bbox[2, 0]) + ', ' + str(self.bbox[2, 1]) + ']', 1)

    def generate(self, params, uext):
        zeta = params['zeta']
        for_pos = params['for_pos']
        t = params['t']

        self.update_cache(t)

        self.update_coeff(t)

        self.init_interpolator()
        self.interpolate_zeta(zeta,
                              for_pos,
                              uext)

    def update_cache(self, t):
        self.double_initialisation = False
        if self.settings['frozen']:
            if self._interpolator0 is None:
                self._t0 = self.timestep_2_time(0)
                self._it0 = 0
                self._interpolator0 = self.read_grid(self._it0, i_cache=0)
            return
        # most common case: t already in the [t0, t1] interval
        if self._t0 <= t <= self._t1:
            return

        # t < t0, something weird (time going backwards)
        if t < self._t0:
            raise ValueError('Please make sure everything is ok. Your time is going backwards.')

        # t > t1, need initialisation
        if t > self._t1:
            new_it = self.time_2_timestep(t)
            # new timestep requires initialising the two of them (not likely at all)
            # this means that the simulation timestep > LES timestep
            if new_it > self._it1:
                self.double_initialisation = True
            else:
                # t1 goes to t0
                self._t0 = self._t1
                self._it0 = self._it1
                self._interpolator0 = self._interpolator1.copy()

                # t1 updates to the next (new_it + 1)
                self._it1 = new_it + 1
                self._t1 = self.timestep_2_time(self._it1)
                self._interpolator1 = self.read_grid(self._it1, i_cache=1)
                return

        # last case, both interp need to be initialised
        if (self._t0 is None or self.double_initialisation):
            self._t0 = self.timestep_2_time(new_it)
            self._it0 = new_it
            self._interpolator0 = self.read_grid(self._it0, i_cache=0)

            self._it1 = new_it + 1
            self._t1 = self.timestep_2_time(self._it1)
            self._interpolator1 = self.read_grid(self._it1, i_cache=1)

    def update_coeff(self, t):
        if self.settings['frozen']:
            self.coeff = 0.0
            return

        self.coeff = self.linear_coeff([self._t0, self._t1], t)
        return

    def time_2_timestep(self, t):
        return int(max(0, np.floor((t - self.grid_data['time'][0])/self.grid_data['time'][1])))

    def timestep_2_time(self, it):
        return it*self.grid_data['time'][1] + self.grid_data['time'][0]

    def get_field_bbox(self, x_grid, y_grid, z_grid, frame='G'):
        bbox = np.zeros((3, 2))
        bbox[0, :] = [np.min(x_grid), np.max(x_grid)]
        bbox[1, :] = [np.min(y_grid), np.max(y_grid)]
        bbox[2, :] = [np.min(z_grid), np.max(z_grid)]
        if frame == 'G':
            bbox[:, 0] = self.gstar_2_g(bbox[:, 0])
            bbox[:, 1] = self.gstar_2_g(bbox[:, 1])
        return bbox

    def create_interpolator(self, data, x_grid, y_grid, z_grid, i_dim):
        interpolator = interpolate.RegularGridInterpolator((x_grid, y_grid, z_grid),
                                                            data,
                                                            bounds_error=False,
                                                            fill_value=0.0)
        return interpolator


    def interpolate_zeta(self, zeta, for_pos, u_ext, interpolator=None, offset=np.zeros((3))):
        if interpolator is None:
            interpolator = self.interpolator

        for isurf in range(len(zeta)):
            _, n_m, n_n = zeta[isurf].shape
            for i_m in range(n_m):
                for i_n in range(n_n):
                    coord = self.g_2_gstar(self.apply_periodicity(zeta[isurf][:, i_m, i_n] + for_pos[0:3] + offset))
                    for i_dim in range(3):
                        try:
                            u_ext[isurf][i_dim, i_m, i_n] = self.interpolator[i_dim](coord)
                        except ValueError:
                            print(coord)
                            raise ValueError()
                    u_ext[isurf][:, i_m, i_n] = self.gstar_2_g(u_ext[isurf][:, i_m, i_n])


    @staticmethod
    def periodicity(x, bbox):
        try:
            new_x = bbox[0] + divmod(x - bbox[0], bbox[1] - bbox[0])[1]
        except ZeroDivisionError:
            new_x = x
        return new_x


    def apply_periodicity(self, coord):
        new_coord = coord.copy()
        if self.x_periodicity:
            i = 0
            new_coord[i] = self.periodicity(new_coord[i], self.bbox[i, :])
        if self.y_periodicity:
            i = 1
            new_coord[i] = self.periodicity(new_coord[i], self.bbox[i, :])

        # if self.x_periodicity:
        #TODO I think this does not work when bbox is not ordered (bbox[i, 0] is not < bbox[i, 1])
            # i = 0
            # # x in interval:
            # if self.bbox[i, 0] <= new_coord[i] <= self.bbox[i, 1]:
                # pass
            # # lower than min bbox
            # elif new_coord[i] < self.bbox[i, 0]:
                # temp = divmod(new_coord[i], self.bbox[i, 0])[1]
                # if np.isnan(temp):
                    # pass
                # else:
                    # new_coord[i] = temp

            # # greater than max bbox
            # elif new_coord[i] > self.bbox[i, 1]:
                # temp = divmod(new_coord[i], self.bbox[i, 1])[1]
                # if np.isnan(temp):
                    # pass
                # else:
                    # new_coord[i] = temp

        # if self.y_periodicity:
            # i = 1
            # # y in interval:
            # if self.bbox[i, 0] <= new_coord[i] <= self.bbox[i, 1]:
                # pass
            # # lower than min bbox
            # elif new_coord[i] < self.bbox[i, 0]:
                # try:
                    # temp = divmod(new_coord[i], self.bbox[i, 0])[1]
                # except ZeroDivisionError:
                    # temp = new_coord[i]
                # if np.isnan(temp):
                    # pass
                # else:
                    # new_coord[i] = temp
                    # if new_coord[i] < 0.0:
                        # new_coord[i] = self.bbox[i, 1] + new_coord[i]
            # # greater than max bbox
            # elif new_coord[i] > self.bbox[i, 1]:
                # temp = divmod(new_coord[i], self.bbox[i, 1])[1]
                # if np.isnan(temp):
                    # pass
                # else:
                    # new_coord[i] = temp
                    # if new_coord[i] < 0.0:
                        # new_coord[i] = self.bbox[i, 1] + new_coord[i]
        return new_coord


    @staticmethod
    def linear_coeff(t_vec, t):
        # this is 0 when t == t_vec[0]
        # 1 when t == t_vec[1]
        return (t - t_vec[0])/(t_vec[1] - t_vec[0])

    def read_grid(self, i_grid, i_cache=0):
        """
        This function returns an interpolator list of size 3 made of `scipy.interpolate.RegularGridInterpolator`
        objects.
        """
        velocities = ['ux', 'uy', 'uz']
        interpolator = list()
        for i_dim in range(3):
            file_name = self.grid_data['grid'][i_grid][velocities[i_dim]]['file']
            if i_cache == 0:
                if not self.settings['store_field']:
                    # load file, but dont copy it
                    self.vel_holder0[i_dim] = np.memmap(self.route + '/' + file_name,
                                                   # dtype='float64',
                                                   dtype=self.grid_data['grid'][i_grid][velocities[i_dim]]['Precision'],
                                                   shape=(self.grid_data['dimensions'][2],
                                                          self.grid_data['dimensions'][1],
                                                          self.grid_data['dimensions'][0]),
                                                   order='F')
                else:
                    # load and store file
                    self.vel_holder0[i_dim] = (np.fromfile(open(self.route + '/' + file_name, 'rb'),
                                                          dtype=self.grid_data['grid'][i_grid][velocities[i_dim]]['Precision']).\
                                                          reshape((self.grid_data['dimensions'][2],
                                                                   self.grid_data['dimensions'][1],
                                                                   self.grid_data['dimensions'][0]),
                                                                   order='F'))

                interpolator.append(self.create_interpolator(self.vel_holder0[i_dim],
                                                        self.grid_data['initial_x_grid'],
                                                        self.grid_data['initial_y_grid'],
                                                        self.grid_data['initial_z_grid'],
                                                        i_dim=i_dim))
            elif i_cache == 1:
                if not self.settings['store_field']:
                    # load file, but dont copy it
                    self.vel_holder1[i_dim] = np.memmap(self.route + '/' + file_name,
                                                   # dtype='float64',
                                                   dtype=self.grid_data['grid'][i_grid][velocities[i_dim]]['Precision'],
                                                   shape=(self.grid_data['dimensions'][2],
                                                          self.grid_data['dimensions'][1],
                                                          self.grid_data['dimensions'][0]),
                                                   order='F')
                else:
                    # load and store file
                    self.vel_holder1[i_dim] = (np.fromfile(open(self.route + '/' + file_name, 'rb'),
                                                          dtype=self.grid_data['grid'][i_grid][velocities[i_dim]]['Precision']).\
                                                          reshape((self.grid_data['dimensions'][2],
                                                                   self.grid_data['dimensions'][1],
                                                                   self.grid_data['dimensions'][0]),
                                                                   order='F'))

                interpolator.append(self.create_interpolator(self.vel_holder1[i_dim],
                                                        self.grid_data['initial_x_grid'],
                                                        self.grid_data['initial_y_grid'],
                                                        self.grid_data['initial_z_grid'],
                                                        i_dim=i_dim))
            else:
                raise Error('i_cache has to be 0 or 1')
        return interpolator

    @staticmethod
    def g_2_gstar(coord_g):
        return np.array([coord_g[0], coord_g[2], -coord_g[1]])

    @staticmethod
    def gstar_2_g(coord_star):
        return np.array([coord_star[0], -coord_star[2], coord_star[1]])
