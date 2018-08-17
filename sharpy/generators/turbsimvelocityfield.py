import numpy as np
import scipy.interpolate as interpolate

import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings
import sharpy.utils.cout_utils as cout
import h5py as h5


@generator_interface.generator
class TurbSimVelocityField(generator_interface.BaseGenerator):
    generator_id = 'TurbSimVelocityField'

    def __init__(self):
        self.in_dict = dict()
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['turbulent_field'] = 'str'
        self.settings_default['turbulent_field'] = None

        self.settings_types['u_inf'] = 'float'
        self.settings_default['u_inf'] = None

        self.settings_types['offset'] = 'list(float)'
        self.settings_default['offset'] = np.zeros((3,))

        self.settings = dict()

        self.h5file = None
        self.turb_time = None
        self.turb_x_initial = None
        self.turb_y_initial = None
        self.turb_z_initial = None
        self.turb_u_ref = None
        self.turb_data = None

        self.bbox = None
        self.interpolator = 3*[None]

    def initialise(self, in_dict):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict, self.settings_types, self.settings_default)
        self.settings = self.in_dict

        # load the turbulent field
        self.h5file = h5.File(self.settings['turbulent_field'])
        # make time to increase from -t to 0 instead of 0 to t
        self.turb_time = self.h5file['time'].value
        self.turb_time = self.turb_time - np.max(self.turb_time)
        self.turb_u_ref = self.h5file['u_inf'].value
        # self.turb_x_initial = self.turb_u_ref*(self.turb_time[1] - self.turb_time[0]) - self.settings['offset'][0]
        self.turb_x_initial = self.turb_time*self.turb_u_ref + self.settings['offset'][0]
        self.turb_y_initial = self.h5file['y_grid'].value + self.settings['offset'][1]
        self.turb_z_initial = self.h5file['z_grid'].value + self.settings['offset'][2]

        self.turb_data = self.h5file['data/velocity'].value

        self.init_interpolator(self.turb_data, self.turb_x_initial, self.turb_y_initial, self.turb_z_initial)

    def generate(self, params, uext):
        zeta = params['zeta']
        for_pos = params['for_pos']
        # ts = params['ts']
        # dt = params['dt']
        # t = params['t']

        # check that u_inf in the solver == u_inf in the generator
        # if np.abs(self.settings['u_inf'].value - self.turb_u_ref) > 1e-3:
        #     cout.cout_wrap('The freestream velocity in the solver does \
        #         not match the mean velocity in the turbulent field', 3)

        # get boundary box
        # self.bbox = self.get_bbox(zeta)

        # x_coordinates = self.turb_x_initial.copy()
        # y_coordinates = self.turb_y_initial.copy()
        # z_coordinates = self.turb_z_initial.copy()

        # calculate relevant slices
        # slices = (self.bbox[0, 0] + for_pos[0]) <= x_coordinates
        # slices = np.logical_and(slices, x_coordinates <= (self.bbox[0, 1] + for_pos[0]))

        # remember to remove the reference speed u_inf (turb) from the velocity field
        self.interpolate_zeta(zeta,
                              for_pos,
                              uext)
        # a = 1

    @staticmethod
    def get_bbox(zeta):
        """
        Calculates the bounding box of a given set of point coordinates
        :param zeta:
        :return:
        """
        bbox = np.zeros((3, 2))
        for i_surf in range(len(zeta)):
            ndim, nn, nm = zeta[i_surf].shape
            for idim in range(ndim):
                bbox[idim, :] = (min(bbox[idim, 0], np.min(zeta[i_surf][idim, :, :])),
                                 max(bbox[idim, 1], np.max(zeta[i_surf][idim, :, :])))
        return bbox

    def init_interpolator(self, data, x_grid, y_grid, z_grid):
        for i_dim in range(3):
            self.interpolator[i_dim] = interpolate.RegularGridInterpolator((y_grid, z_grid, x_grid),
                                                                           data[i_dim, :, :, :],
                                                                           bounds_error=False,
                                                                           fill_value=0.0)

    def interpolate_zeta(self, zeta, for_pos, u_ext):
        for i_dim in range(3):
            for isurf in range(len(zeta)):
                _, n_m, n_n = zeta[isurf].shape
                for i_m in range(n_m):
                    for i_n in range(n_n):
                        coord = zeta[isurf][:, i_m, i_n] + for_pos[0:3]
                        coord = np.roll(coord, -1)
                        try:
                            u_ext[isurf][i_dim, i_m, i_n] = self.interpolator[i_dim](coord)
                        except ValueError:
                            print(coord)
                            raise ValueError()
                        # next = u_ext[isurf][i_dim, i_m, i_n]


