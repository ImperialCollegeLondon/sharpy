import numpy as np
import scipy.interpolate as interpolate
import h5py as h5
import os
from xml.dom import minidom

import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings
import sharpy.utils.cout_utils as cout


@generator_interface.generator
class NoCopyTurbVelocityField(generator_interface.BaseGenerator):
    generator_id = 'NoCopyTurbVelocityField'

    def __init__(self):
        self.in_dict = dict()
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['print_info'] = 'bool'
        self.settings_default['print_info'] = True

        self.settings_types['turbulent_field'] = 'str'
        self.settings_default['turbulent_field'] = None

        self.settings_types['u_inf'] = 'float'
        self.settings_default['u_inf'] = 0.

        self.settings_types['offset'] = 'list(float)'
        self.settings_default['offset'] = np.zeros((3,))

        self.settings_types['centre_y'] = 'bool'
        self.settings_default['centre_y'] = True

        self.settings_types['periodicity'] = 'str'
        self.settings_default['periodicity'] = 'xy'

        self.settings = dict()

        self.file = None
        self.extension = None
        self.turb_time = None
        self.turb_x_initial = None
        self.turb_y_initial = None
        self.turb_z_initial = None
        self.turb_u_ref = None
        self.turb_data = None

        self.bbox = None
        self.interpolator = 3*[None]
        self.x_periodicity = False
        self.y_periodicity = False
        self.vel_holder = 3*[None]

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
        # load the turbulent field HDF5
        # with h5.File(self.settings['turbulent_field']) as self.file:
        #     # make time to increase from -t to 0 instead of 0 to t
        #     try:
        #         self.turb_time = self.file['time'].value
        #         self.turb_time = self.turb_time - np.max(self.turb_time)
        #         self.turb_u_ref = self.file['u_inf'].value
        #         self.turb_x_initial = self.turb_time*self.turb_u_ref + self.settings['offset'][0]
        #     except KeyError:
        #         self.turb_x_initial = self.file['x_grid'].value - np.max(self.file['x_grid'].value) + self.settings['offset'][0]
        #     self.turb_y_initial = self.file['y_grid'].value + self.settings['offset'][1]
        #     self.turb_z_initial = self.file['z_grid'].value + self.settings['offset'][2]
        #
        #     self.turb_data = self.h5file['data/velocity'].value
        #
        #     self.init_interpolator(self.turb_data, self.turb_x_initial, self.turb_y_initial, self.turb_z_initial)

    def read_xdmf(self, in_file):
        # store route of file for the other files
        route = os.path.dirname(os.path.abspath(in_file))

        # file to string
        with open(in_file, 'r') as self.file:
            data = self.file.read().replace('\n', '')

        # parse data
        from lxml import objectify
        tree = objectify.fromstring(data)

        # mesh dimensions
        dimensions = np.fromstring(tree.Domain.Topology.attrib['Dimensions'],
                                   sep=' ',
                                   count=3,
                                   dtype=int)

        # origin
        # NOTE: we can count here the offset?
        origin = np.fromstring(tree.Domain.Geometry.DataItem[0].text,
                               sep=' ',
                               count=int(tree.Domain.Geometry.DataItem[0].attrib['Dimensions']),
                               dtype=float)

        # dxdydz
        # because of how XDMF does it, it is actually dzdydx
        dxdydz = np.fromstring(tree.Domain.Geometry.DataItem[1].text,
                               sep=' ',
                               count=int(tree.Domain.Geometry.DataItem[1].attrib['Dimensions']),
                               dtype=float)
        # now onto the grid
        n_grid = len(tree.Domain.Grid.Grid)
        grid = [dict()]*n_grid
        for i, i_grid in enumerate(tree.Domain.Grid.Grid):
            # cycle through attributes
            for k_attrib, v_attrib in i_grid.attrib.items():
                grid[i][k_attrib] = v_attrib

        if n_grid > 1:
            cout.cout_wrap('CAREFUL: n_grid > 1, but we don\' support time series yet')

        # get Attributes (upper case A is not a mistake)
        for i_attrib, attrib in enumerate(i_grid.Attribute):
            grid[0][attrib.attrib['Name']] = dict()
            grid[0][attrib.attrib['Name']]['file'] = attrib.DataItem.text.replace(' ', '')

        # now we have the file names and the dimensions
        self.initial_x_grid = np.array(np.arange(0, dimensions[2]))*dxdydz[2]
        self.initial_y_grid = np.array(np.arange(0, dimensions[1]))*dxdydz[1]
        self.initial_z_grid = np.array(np.arange(0, dimensions[0]))*dxdydz[0]

        # the domain now goes:
        centre_z_offset = 0.
        if self.settings['centre_y']:
            centre_z_offset = -0.5*(self.initial_z_grid[-1] - self.initial_z_grid[0])

        # this is all in G* frame (Y is up)
        self.initial_x_grid += self.settings['offset'][0] + origin[0]
        self.initial_x_grid -= np.max(self.initial_x_grid)
        self.initial_y_grid += self.settings['offset'][1] + origin[1]
        # self.initial_y_grid = self.initial_y_grid[::-1]
        self.initial_z_grid += self.settings['offset'][2] + origin[2] + centre_z_offset

        self.bbox = self.get_field_bbox(self.initial_x_grid,
                                        self.initial_y_grid,
                                        self.initial_z_grid,
                                        frame='G')
        if self.settings['print_info']:
            cout.cout_wrap('The domain bbox in G FoR is:', 1)
            cout.cout_wrap(' x = [' + str(self.bbox[0, 0]) + ', ' + str(self.bbox[0, 1]) + ']', 1)
            cout.cout_wrap(' y = [' + str(self.bbox[1, 0]) + ', ' + str(self.bbox[1, 1]) + ']', 1)
            cout.cout_wrap(' z = [' + str(self.bbox[2, 0]) + ', ' + str(self.bbox[2, 1]) + ']', 1)

        # now we load the velocities (one by one, so we don't store all the
        # info more than once at the same time)
        velocities = ['ux', 'uy', 'uz']
        velocities_mult = np.array([1.0, 1.0, 1.0])
        for i_dim in range(3):
            file_name = grid[0][velocities[i_dim]]['file']

            # load file, but dont copy it
            self.vel_holder[i] = np.memmap(route + '/' + file_name,
                                           dtype='float64',
                                           shape=(dimensions[2], dimensions[1], dimensions[0]),
                                           order='F')

            self.init_interpolator(self.vel_holder[i],
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
    def get_field_bbox(x_grid, y_grid, z_grid, frame='G'):
        bbox = np.zeros((3, 2))
        bbox[0, :] = [np.min(x_grid), np.max(x_grid)]
        bbox[1, :] = [np.min(y_grid), np.max(y_grid)]
        bbox[2, :] = [np.min(z_grid), np.max(z_grid)]
        if frame == 'G':
            bbox[:, 0] = self.gstar_2_g(bbox[:, 0])
            bbox[:, 1] = self.gstar_2_g(bbox[:, 1])
        return bbox

    def init_interpolator(self, data, x_grid, y_grid, z_grid, i_dim=None):
        if i_dim is None:
            raise NotImplementedError('Don\'t use this option without going over the code and updating it. Right now it is wrong')
            for i_dim in range(3):
                self.interpolator[i_dim] = interpolate.RegularGridInterpolator((z_grid, y_grid, x_grid),
                                                                               data[i_dim, :, :, :],
                                                                               bounds_error=False,
                                                                               fill_value=0.0)
        else:
            bounds_error = False
            if 'x' in self.settings['periodicity'] and i_dim == 0:
                bounds_error = True
            elif 'y' in self.settings['periodicity'] and i_dim == 2:
                bounds_error = True
            self.interpolator[i_dim] = interpolate.RegularGridInterpolator((x_grid, y_grid, z_grid),
                                                                           data,
                                                                           bounds_error=bounds_error,
                                                                           fill_value=0.0)

    def interpolate_zeta(self, zeta, for_pos, u_ext):
        for i_dim in range(3):
            for isurf in range(len(zeta)):
                _, n_m, n_n = zeta[isurf].shape
                for i_m in range(n_m):
                    for i_n in range(n_n):
                        coord_star = self.g_2_gstar(self.apply_periodicity(zeta[isurf][:, i_m, i_n] + for_pos[0:3]))
                        try:
                            u_ext[isurf][i_dim, i_m, i_n] = self.gstar_2_g(self.interpolator[i_dim](coord))
                        except ValueError:
                            print(coord)
                            raise ValueError()

    @staticmethod
    def g_2_gstar(coord_g):
        return np.array([coord_g[0], coord_g[2], -coord_g[1]])

    @staticmethod
    def gstar_2_g(coord_star):
        return np.array([coord_star[0], -coord_star[2], coord_star[1]])

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



