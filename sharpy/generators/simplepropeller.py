import numpy as np
import sympy as sym
import scipy.special as sp
import sharpy.utils.algebra as algebra

import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings
import sharpy.utils.exceptions as exc

@generator_interface.generator
class SimplePropeller(generator_interface.BaseGenerator):
    r"""
    Simple propeller velocity field.

    ``SimplePropeller`` is a class inherited from ``BaseGenerator``

    Args:

            ===================  ===============  =================================================  ===================
            Name                 Type             Description                                        Default
            ===================  ===============  =================================================  ===================
            ===================  ===============  =================================================  ===================

    Attributes:
        settings_types (dict): Acceptable data types of the input data
        settings_default (dict): Default values for input data should the user not provide them

    See Also:
        .. py:class:: sharpy.utils.generator_interface.BaseGenerator

    """
    generator_id = 'SimplePropeller'

    def __init__(self):
        self.in_dict = dict()
        self.settings_types = dict()
        self.settings_default = dict()

        # INPUTS
        # Radius of propeller
        # Omega = rotational velocity in rpm

        self.settings_types['u_inf'] = 'float'
        self.settings_default['u_inf'] = None

        self.settings_types['u_inf_direction'] = 'list(float)'
        self.settings_default['u_inf_direction'] = np.array([1.0, 0, 0])

        self.settings_types['radius'] = 'float'
        self.settings_default['radius'] = None

        self.settings_types['omega'] = 'float'
        self.settings_default['omega'] = None

        self.settings_types['gust_shape'] = 'str'
        self.settings_default['gust_shape'] = None

        self.settings_types['gust_length'] = 'float'
        self.settings_default['gust_length'] = 0.0

        self.settings_types['gust_intensity'] = 'float'
        self.settings_default['gust_intensity'] = 0.0

        self.settings_types['offset'] = 'float'
        self.settings_default['offset'] = 0.0

        self.settings_types['span'] = 'float'
        self.settings_default['span'] = 0.

        self.settings_types['file'] = 'str'
        self.settings_default['file'] = ''

        self.settings_types['relative_motion'] = 'bool'
        self.settings_default['relative_motion'] = False

        self.u_inf = 0.
        self.u_inf_direction = None

        self.file_info = None

        self.implemented_gusts = []
        self.implemented_gusts.append('1-cos')

        self.settings = dict()

    def initialise(self, in_dict):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict, self.settings_types, self.settings_default)
        self.settings = self.in_dict

        # check that the gust type is valid
        if not (self.settings['gust_shape'] in self.implemented_gusts):
            raise AttributeError('The gust shape ' + self.settings['gust_shape'] + ' is not implemented')

        self.u_inf = self.in_dict['u_inf'].value
        self.u_inf_direction = self.in_dict['u_inf_direction']

        if self.settings['gust_shape'] == 'time varying':
            self.file_info = np.loadtxt(self.settings['file'])

        self.lambd = self.settings['omega'].value*self.settings['radius'].value/self.settings['u_inf'].value

    # PROPELLER WAKE MODEL
    # IN: GAMMA_T, GAMMA_TOT, GAMMA_L, ZETA, POS (G FoR), ORIENTATION(G FoR)
    # OUT: U_EXT AT THE GRID POINTS ZETA
    # def propeller_wake(self, gamma_t, gamma_tot, gamma_l, zeta, pos_g, orientation_g, u_ext_g):
    def generate(self, params, uext):
        zeta = params['zeta']
        for_pos = params['for_pos']
        override = params['override']
        ts = params['ts']
        dt = params['dt']
        t = params['t']
        gust_shape = None
        if self.settings['gust_shape'] == '1-cos':
            def gust_shape(x, y, z, gust_length, gust_intensity, span=0):
                vel = np.zeros((3,))
                if x > 0.0 or x < -gust_length:
                    return vel

                vel[2] = (1.0 - np.cos(2.0*np.pi*x/gust_length))*gust_intensity*0.5
                return vel

        for i_surf in range(len(zeta)):
            if override:
                uext[i_surf].fill(0.0)

            for i in range(zeta[i_surf].shape[1]):
                for j in range(zeta[i_surf].shape[2]):
                    if self.settings['relative_motion']:
                        uext[i_surf][:, i, j] += self.u_inf*self.u_inf_direction
                        uext[i_surf][:, i, j] += gust_shape(zeta[i_surf][0, i, j] - self.u_inf*t + self.settings['offset'],
                                                            zeta[i_surf][1, i, j],
                                                            zeta[i_surf][2, i, j],
                                                            self.settings['gust_length'].value,
                                                            self.settings['gust_intensity'].value,
                                                            self.settings['span'].value
                                                            )
                    else:
                        uext[i_surf][:, i, j] += gust_shape(for_pos[0] + zeta[i_surf][0, i, j] + self.settings['offset'],
                                                            for_pos[1] + zeta[i_surf][1, i, j],
                                                            for_pos[2] + zeta[i_surf][2, i, j],
                                                            self.settings['gust_length'].value,
                                                            self.settings['gust_intensity'].value,
                                                            self.settings['span'].value
                                                            )
        R = 4
        gamma_t = 1
        gamma_tot = 2
        gamma_l = gamma_tot*0.5/(np.pi*R)
        pos_g = np.array([[0], [1], [0]])
        orientation_g = -np.array([[0.0], [1.0], [0.0]])
        # the propeller (P) FoR is given by x_p, y_p, z_p
        # y_p = orientation_g
        # z_p = [0, 0, 1]_g projected in the plane with normal y_p
        z_g = np.array([[0.0], [0.0], [1.0]])
        # z_g_projected = algebra.project_vect_on_plane(z_g, orientation_g)
        z_g_projected = -np.array([[0.0], [0.0], [1.0]])

        # x_g = np.cross(orientation_g, z_g_projected)
        # x_g = np.cross(orientation_g, z_g)
        x_g = np.array([[1.0], [0.0], [0.0]])

        # vectors are packed by rows
        # triad_g = [x, y, z]^T
        # triad_g = np.array([x_g, orientation_g, z_g_projected])
        triad_g = np.hstack([x_g, orientation_g, z_g_projected])

        # rotation matrix from p to G
        rot_pg = triad_g

        ex = np.array([[1.0], [0.0], [0.0]])
        ey = np.array([[0.0], [1.0], [0.0]])
        ez = np.array([[0.0], [0.0], [1.0]])

        for i_surf in range(len(zeta)):
            for i_M in range(zeta[i_surf].shape[1]):
                for i_N in range(zeta[i_surf].shape[2]):
                    # coordinates of the grid point in G for
                    grid_g = np.zeros((3, 1))
                    grid_g[0, 0] = zeta[i_surf][0, i_M, i_N]
                    grid_g[1, 0] = zeta[i_surf][1, i_M, i_N]
                    grid_g[2, 0] = zeta[i_surf][2, i_M, i_N]

                    u_vec_g = self.calculate_induced_velocities(grid_g,
                        pos_g,
                        rot_pg,
                        R,
                        gamma_t,
                        gamma_tot,
                        gamma_l,
                        ex,
                        ey,
                        ez)

                    # uext[i_surf][:, i_M, i_N] += u_vec_g
                    uext[i_surf][0, i_M, i_N] += u_vec_g[0]
                    uext[i_surf][1, i_M, i_N] += u_vec_g[1]
                    uext[i_surf][2, i_M, i_N] += u_vec_g[2]

    # @staticmethod
    def calculate_induced_velocities(self, grid_g, pos_g, rot_pg, R, gamma_t, gamma_tot, gamma_l, ex, ey, ez):
    # def calculate_induced_velocities(grid_g, pos_g, rot_pg, R, gamma_t, gamma_tot, gamma_l, ex, ey, ez):
        # print(grid_g)
        # print(pos_g)
        # print(rot_pg)

        # coords of the grid point in G wrt to propeller
        x_g = grid_g - pos_g

        # coords of the grid point in P wrt to propeller
        x_l = np.dot(rot_pg.T, x_g)

        # cyl coordinates of grid point wrt to propeller
        x = x_l[0]
        y = x_l[1]
        z = x_l[2]
        r = np.sqrt(x ** 2 + z ** 2)
        if x == 0 and z == 0:
            psi = 0
        elif z >= 0:
            psi = np.arccos(x / r)
        else:
            psi = 2.0 * np.pi - np.arccos(x / r)

        er = + np.cos(psi)*ex + np.sin(psi)*ez
        ep = - np.sin(psi)*ex + np.cos(psi)*ez

        # tangential:
        uz_t, ur_t = self.propeller_wake_tangential(r, R, gamma_t, y)
        # bound
        up_b = self.propeller_wake_bound(r, R, gamma_tot, y)
        # wake root
        up_r = self.propeller_wake_root(r, R, gamma_tot, y)
        # longitudinal
        up_l = self.propeller_wake_longitudinal(r, R, gamma_l, y)

        # # tangential:
        # uz_t, ur_t = propeller_wake_tangential(r, R, gamma_t, y)
        # # bound
        # up_b = propeller_wake_bound(r, R, gamma_tot, y)
        # # wake root
        # up_r = propeller_wake_root(r, R, gamma_tot, y)
        # # longitudinal
        # up_l = propeller_wake_longitudinal(r, R, gamma_l, y)

        ur = ur_t
        uz = uz_t
        up = up_b + up_r + up_l

        u_vec_l = ur*er + uz*ey + up*ep
        u_vec_g = np.dot(rot_pg, u_vec_l)
        # u_vec_g = u_vec_g.T
        # print(u_vec_g)

        return u_vec_g
        # return u_vec_g, ur, uz, up

    # @staticmethod
    def propeller_wake_k2_y(self, r, R, y):
    # def propeller_wake_k2_y(r, R, y):
        return 4.0*r*R/((R + r)**2 + y**2)

    # @staticmethod
    def propeller_wake_tangential(self, r, R, gamma_t, y, vortex_cutoff=0.01):
    # def propeller_wake_tangential(r, R, gamma_t, y, vortex_cutoff=0.01):
        k2_y = self.propeller_wake_k2_y(r, R, y)
        k2_0 = self.propeller_wake_k2_y(r, R, 0)
        # k2_y = propeller_wake_k2_y(r, R, y)
        # k2_0 = propeller_wake_k2_y(r, R, 0)
        k_y = np.sqrt(k2_y)
        ur_t = 0.0
        uz_t1 = 0.0
        uz_t2 = 0.0
        uz_t3 = 0.0
        uy_t = 0.0

        if r < vortex_cutoff * R:
            ur_t = - 0.25 * gamma_t * (r * R ** 2) / (R ** 2 + y ** 2) ** 1.5
            uz_t1 = 0.0
            uz_t2 = 0.0
            uz_t3 = 0.0
            uz_t = 0.5 * gamma_t * (1 + y / np.sqrt(R ** 2 + r ** 2))
        elif np.abs(r - R) < 1e-6:
            K = sp.ellipk(k2_y)
            E = sp.ellipe(k2_y)
            if np.abs(y) < 1e-6:
                ur_t = -gamma_t / (2 * np.pi) * np.sqrt(R / r) * ((2.0 - k2_y) / k_y * K - 2. / k_y * E)
                uz_t1 = gamma_t / 4
                uz_t2 = 0
                uz_t3 = 0
            else:
                ur_t = -gamma_t / (2 * np.pi) * np.sqrt(R / r) * ((2.0 - k2_y) / k_y * K - 2. / k_y * E)
                uz_t1 = gamma_t / 4
                uz_t2 = gamma_t / 2 * (y * k_y) / (2 * np.pi * np.sqrt(r * R)) * K
                uz_t3 = 0
        else:
            K = sp.ellipk(k2_y)
            E = sp.ellipe(k2_y)
            PI = float(sym.elliptic_pi(float(k2_0), float(k2_y)))
            ur_t = -gamma_t / (2 * np.pi) * np.sqrt(R / r) * ((2.0 - k2_y) / k_y * K - 2. / k_y * E)
            uz_t1 = gamma_t / 2. * (R - r + np.abs(R - r)) / (2 * np.abs(R - r))
            uz_t2 = gamma_t / 2. * (y * k_y) / (2 * np.pi * np.sqrt(r * R)) * K
            uz_t3 = gamma_t / 2. * (y * k_y) / (2 * np.pi * np.sqrt(r * R)) * (R - r) / (R + r) * PI

        uz_t = uz_t1 + uz_t2 + uz_t3

        # if r < 0.01:
        #     ur_t = - 0.25 * gamma_t * (r * R ** 2) / (R ** 2 + y ** 2) ** 1.5
        #     uz_t1 = 0.0
        #     uz_t2 = 0.0
        #     uz_t3 = 0.0
        #     uz_t = 0.5 * gamma_t * (1 + y / np.sqrt(R ** 2 + r ** 2))
        # else:
        #     K = sp.ellipk(k2_y)
        #     E = sp.ellipe(k2_y)
        #     PI = float(sym.elliptic_pi(float(k2_0), float(k2_y)))
        #     ur_t = -gamma_t / (2 * np.pi) * np.sqrt(R / r) * ((2.0 - k2_y) / k_y * K - 2. / k_y * E)
        #     uz_t1 = gamma_t / 2. * (R - r + np.abs(R - r)) / (2 * np.abs(R - r))
        #     uz_t2 = gamma_t / 2. * (y * k_y) / (2 * np.pi * np.sqrt(r * R)) * K
        #     uz_t3 = gamma_t / 2. * (y * k_y) / (2 * np.pi * np.sqrt(r * R)) * (R - r) / (R + r) * PI
        #     uz_t = uz_t1 + uz_t2 + uz_t3
        # if r == R:
        #     K = sp.ellipk(k2_y)
        #     uz_t1 = gamma_t / 4
        #     uz_t2 = gamma_t / 2 * (y * k_y) / (2 * np.pi * np.sqrt(r * R)) * K
        #     uz_t3 = 0
        #     uz_t = uz_t1 + uz_t2 + uz_t3
        # if (r == R) and (y == 0):
        #     ur_t = 0
        #     uz_t1 = gamma_t / 4
        #     uz_t2 = 0
        #     uz_t3 = 0
        #     uz_t = uz_t1 + uz_t2 + uz_t3

        return uz_t, ur_t

    # @staticmethod
    def propeller_wake_bound(self, r, R, gamma_tot, y, vortex_cutoff=0.01):
    # def propeller_wake_bound(r, R, gamma_tot, y, vortex_cutoff=0.01):
        up_b = 0.0
        # if r < vortex_cutoff * R or np.abs(y) < 1e-6:
        if r < 0.01 or y == 0:
            up_b = 0.0
        else:
            T1 = ((np.sqrt(r ** 2 + y ** 2) - r) * (r + R) - y ** 2) / (2 * y ** 2)
            T2 = ((np.sqrt(r ** 2 + y ** 2) + r) * (np.sqrt(r ** 2 + y ** 2) + R)) / (2 * y ** 2)
            # k2_y = propeller_wake_k2_y(r, R, y)
            k2_y = self.propeller_wake_k2_y(r, R, y)
            k_y = np.sqrt(k2_y)
            K = sp.ellipk(k2_y)
            n1 = 2 * r / (r + np.sqrt(r ** 2 + y ** 2))
            n2 = 2 * r / (r - np.sqrt(r ** 2 + y ** 2))
            PI1 = float(sym.elliptic_pi(float(n1), float(k2_y)))
            PI2 = float(sym.elliptic_pi(float(n2), float(k2_y)))
            up_b = gamma_tot / (4 * np.pi) * (
                        1 / r * (y / np.sqrt(r ** 2 + y ** 2) - np.sign(y)) - 1 / (np.pi * y) * np.sqrt(
                    r / R) * y ** 2 / r ** 2 * k_y * (K + T1 * PI1 - T2 * PI2))
        return up_b

    # @staticmethod
    def propeller_wake_root(self, r, R, gamma_tot, y, vortex_cutoff=0.01):
    # def propeller_wake_root(r, R, gamma_tot, y, vortex_cutoff=0.01):
        up_r = 0.0
        # if r < vortex_cutoff * R:
        if r < 0.01:
            pass
        else:
            up_r = -0.25 * gamma_tot / (np.pi * r) * (1.0 + y / np.sqrt(r ** 2 + y ** 2))
        return up_r

    # @staticmethod
    def propeller_wake_longitudinal(self, r, R, gamma_l, y, vortex_cutoff=0.01):
    # def propeller_wake_longitudinal(r, R, gamma_l, y, vortex_cutoff=0.01):
        up_l = 0.0
        up_l1 = 0.0
        up_l2 = 0.0
        up_l3 = 0.0
        # if r < R * vortex_cutoff:
        if r < 0.01:
            up_l = 0
        # elif np.abs(r - R) < 1e-6:
        elif r == R:
            # k2_y = propeller_wake_k2_y(r, R, y)
            k2_y = self.propeller_wake_k2_y(r, R, y)
            k_y = np.sqrt(k2_y)
            K = sp.ellipk(k2_y)
            up_l = gamma_l / 4 * (R / r) + gamma_l / 2 * (R / r) * (y * k_y) / (2 * np.pi * np.sqrt(r * R)) * K
        else:
            k2_y = self.propeller_wake_k2_y(r, R, y)
            k2_0 = self.propeller_wake_k2_y(r, R, 0)
            # k2_y = propeller_wake_k2_y(r, R, y)
            # k2_0 = propeller_wake_k2_y(r, R, 0)
            k_y = np.sqrt(k2_y)
            K = sp.ellipk(k2_y)
            PI = float(sym.elliptic_pi(float(k2_0), float(k2_y)))
            up_l1 = gamma_l / 2 * (R / r) * (r - R + np.abs(R - r)) / (2 * np.abs(R - r))
            up_l2 = gamma_l / 2 * (R / r) * (y * k_y) / (2 * np.pi * np.sqrt(r * R)) * K
            up_l3 = -gamma_l / 2 * (R / r) * (y * k_y) / (2 * np.pi * np.sqrt(r * R)) * (R - r) / (R + r) * PI
            up_l = up_l1 + up_l2 + up_l3
        if (r == R) and (y == 0):
            up_l = 0
        return up_l
