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

        # TODO this u_inf needs to be an input in real time

        self.settings_types['radius'] = 'float'
        self.settings_default['radius'] = None

        self.settings_types['omega'] = 'float'
        self.settings_default['omega'] = None

        self.settings_types['node_global'] = 'int'
        self.settings_default['node_global'] = None

        self.settings_types['element_node'] = 'list(int)' # 2 element vector.
        self.settings_default['element_node'] = None  # [i_elem, i_node_local]

        self.settings_types['offset'] = 'list(float)'
        self.settings_default['offset'] = np.zeros((3,))


        self.settings_types['direction_type'] = 'str'
        self.settings_default['direction_type'] = 'material'
        # Supported directions:
        # "u_inf"
        # fixed in "material" FoR
        # fixed in "body-fixed" FoR

        # TODO write documentation
        self.settings_types['direction_input'] = 'list(float)'
        self.settings_default['direction_input'] = np.array([0.0, -1.0, 0.0])

        self.settings_types['u_inf'] = 'float'
        self.settings_default['u_inf'] = None

        self.settings_types['u_inf_direction'] = 'list(float)'
        self.settings_default['u_inf_direction'] = np.array([1.0, 0.0, 0.0])

        self.settings_types['sigma_cutoff'] = 'float'
        self.settings_default['sigma_cutoff'] = 2.


        self.settings = dict()

    def initialise(self, in_dict):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict, self.settings_types, self.settings_default)
        self.settings = self.in_dict

        self.lambd = self.settings['omega'].value*self.settings['radius'].value/self.settings['u_inf'].value

    # PROPELLER WAKE MODEL
    # IN: GAMMA_T, GAMMA_TOT, GAMMA_L, ZETA, POS (G FoR), ORIENTATION(G FoR)
    # OUT: U_EXT AT THE GRID POINTS ZETA
    # def propeller_wake(self, gamma_t, gamma_tot, gamma_l, zeta, pos_g, orientation_g, u_ext_g):
    def generate(self, params, uext):
        zeta = params['zeta']
        for_pos = params['for_pos']
        override = params['override']
        try:
            ts = params['ts']
            dt = params['dt']
            t = params['t']
        except KeyError:
            pass
        pos = params['pos'] # nodal position in A FoR
        psi = params['psi'] # CRV
        self.quat = params['quat']
        # TODO write documentation of params

        R = self.settings['radius'].value
        #TODO implement propeller model for populating gamma_t, gamma_tot and gamma_l
        gamma_t = 1
        gamma_tot = 2
        gamma_l = gamma_tot*0.5/(np.pi*R)

        # checks that the input_type is supported and returns rotation matrix and offset in g FoR
        rotation_to_G = np.zeros((3, 3))
        if self.settings['direction_type'] == 'material':
            i_node_global = self.settings['node_global'].value
            i_elem = self.settings['element_node'][0]
            i_node_local = self.settings['element_node'][1]

            self.node_pos = pos[i_node_global, :]
            self.node_crv = psi[i_elem, i_node_local, :]

            # rotation from B to A
            self.rotation_AB = algebra.crv2rotation(self.node_crv)

            # rotation from A to G
            self.rotation_GA = algebra.quat2rotation(self.quat)

            # composed rotation
            # this makes rotation_to_G == rotation_GB
            self.rotation_to_G = np.dot(self.rotation_GA, self.rotation_AB)

            self.direction_B_FoR = self.settings['direction_input']

            # propeller position in B FoR
            self.propeller_position_B_wrt_A_origin = np.dot(self.rotation_AB.T, self.node_pos) + self.settings['offset']

        elif self.settings['direction_type'] == 'body-fixed':
            raise NotImplementedError('Body-fixed propeller direction is not yet supported!')
        elif self.settings['direction_type'] == 'u_inf':
            raise NotImplementedError('u_inf propeller direction is not yet supported!')
        else:
            raise NotImplementedError(self.settings['direction_type'] + ' propeller direction is not yet supported!')

        # define the rotation matrix from P to B
        self.define_coordinates()



        # ====================================================================================================== #
        # Material frame of reference (B frame) that has Propeller
        # These are imported from structural solver at each time step
        # Alfonso's help is necessary. How can the B frame be imported?
        # Can the time histories propeller coordinate [rp, rx, ry, rz] be exported as paraview data file?
        rp0 = np.array([[0], [0], [0]])  # B-frame origin vector defined in G-frame
        rx0 = np.array([[1], [0], [0]])  # B-frame axis-direction vector defined in G-frame
        ry0 = np.array([[0], [1], [0]])  # B-frame chord-direction vector defined in G-frame
        rz0 = np.array([[0], [0], [1]])  # B-frame surface-normal vector defined in G-frame
        # ====================================================================================================== #

        yp = -1  # Propeller-coordinate origin defined in material frame of reference
        zp = 0  # Propeller-coordinate origin defined in material frame of reference
        angle_rx = 180/180*np.pi - 0/180*np.pi  # Pitch angle wrt material frame of reference
        angle_rz = 0/180*np.pi  # Yaw angle wrt propeller coordinate considering pitch

        # rp, rx, ry, rz = self.propeller_coordinate(rp0, rx0, ry0, rz0, yp, zp, angle_rx, angle_rz)
        # # rotation matrix from p to G
        # rot_pg = np.hstack([rx, ry, rz])

        self.ex = np.array([[1.0], [0.0], [0.0]])
        self.ey = np.array([[0.0], [1.0], [0.0]])
        self.ez = np.array([[0.0], [0.0], [1.0]])

        for i_surf in range(len(zeta)):
            for i_M in range(zeta[i_surf].shape[1]):
                for i_N in range(zeta[i_surf].shape[2]):
                    # TODO only for now
                    uext[i_surf][:, i_M, i_N] = 0.0
                    # coordinates of the grid point in G for
                    grid_g = np.zeros((3, 1))
                    grid_g[0, 0] = zeta[i_surf][0, i_M, i_N]
                    grid_g[1, 0] = zeta[i_surf][1, i_M, i_N]
                    grid_g[2, 0] = zeta[i_surf][2, i_M, i_N]

                    grid_p = self.inertial_2_propeller(grid_g)

                    # grid_p is the relative vector between grid and propeller
                    u_vec_p = self.calculate_induced_velocities(grid_p,
                                                                R,
                                                                gamma_t,
                                                                gamma_tot,
                                                                gamma_l)
                    u_vec_g = self.propeller_2_inertial(u_vec_p)
                    uext[i_surf][:, i_M, i_N] += u_vec_g
                    uext[i_surf][:, i_M, i_N] += self.settings['u_inf'].value*self.settings['u_inf_direction']

    def inertial_2_propeller(self, inertial_coord):
        # zeta in G to zeta in A
        zeta_A = np.dot(self.rotation_GA.T, inertial_coord)
        # zeta in A to zeta in B
        zeta_B = np.squeeze(np.dot(self.rotation_AB.T, zeta_A))
        # offset
        zeta_P = np.dot(self.rotation_BP.T, zeta_B - self.settings['offset'])
        return zeta_P

    def propeller_2_inertial(self, propeller):
        vel_B = np.dot(self.rotation_BP, propeller)
        vel_A = np.dot(self.rotation_AB, vel_B)
        vel_G = np.dot(self.rotation_GA, vel_A)
        return vel_G

    def define_coordinates(self):
        # z_p_B = means vector z of propeller FoR in B FoR.
        x_p_P = np.array([1.0, 0.0, 0.0])
        y_p_P = np.array([0.0, 1.0, 0.0])
        z_p_P = np.array([0.0, 0.0, 1.0])

        z_p_B = self.direction_B_FoR
        x_p_B = algebra.project_vect_on_plane(v=x_p_P,
                                              n=z_p_B)
        y_p_B = np.cross(z_p_B, x_p_B)

        # TODO transpose or not?
        self.rotation_BP = np.column_stack((x_p_B, y_p_B, z_p_B))


    def propeller_coordinate(self, rp0, rx0, ry0, rz0, yp, zp, angle_rx, angle_rz):
        # # Propeller-coordinate origin on yz plane of B frame
        # rp = rp0 + yp * ry0 + zp * rz0

        # Propeller coordinate considering pitch
        rx1 = rx0
        ry1 = + np.cos(angle_rx) * ry0 + np.sin(angle_rx) * rz0
        rz1 = - np.sin(angle_rx) * ry0 + np.cos(angle_rx) * rz0

        # Propeller coordinate considering yaw
        rx = + np.cos(angle_rz) * rx1 + np.sin(angle_rz) * ry1
        ry = - np.sin(angle_rz) * rx1 + np.cos(angle_rz) * ry1
        rz = rz1

        # Propeller-coordinate origin on yz plane of propeller frame
        rp = rp0 + yp * ry + zp * rz

        return rp, rx, ry, rz

    # @staticmethod
    def calculate_induced_velocities(self, grid_p, R, gamma_t, gamma_tot, gamma_l):
        # coordinates of the grid point in P wrt to propeller
        x_l = grid_p

        # cyl coordinates of grid point wrt to propeller
        x = x_l[0]
        y = x_l[1]
        z = x_l[2]
        r = np.sqrt(x ** 2 + z ** 2)
        # if r > self.settings['sigma_cutoff'].value*R:
        #     return np.zeros((3,))

        if x == 0 and z == 0:
            psi = 0
        elif z >= 0:
            psi = np.arccos(x / r)
        else:
            psi = 2.0 * np.pi - np.arccos(x / r)

        er = + np.cos(psi)*self.ex + np.sin(psi)*self.ez
        ep = - np.sin(psi)*self.ex + np.cos(psi)*self.ez

        # tangential:
        uz_t, ur_t = self.propeller_wake_tangential(r, R, gamma_t, y)
        # bound
        up_b = self.propeller_wake_bound(r, R, gamma_tot, y)
        # wake root
        up_r = self.propeller_wake_root(r, R, gamma_tot, y)
        # longitudinal
        up_l = self.propeller_wake_longitudinal(r, R, gamma_l, y)

        ur = ur_t
        uz = uz_t
        up = up_b + up_r + up_l

        u_vec_p = ur*er + uz*self.ey + up*ep

        return np.squeeze(u_vec_p)

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

        # if r < vortex_cutoff * R:
        #     ur_t = - 0.25 * gamma_t * (r * R ** 2) / (R ** 2 + y ** 2) ** 1.5
        #     uz_t1 = 0.0
        #     uz_t2 = 0.0
        #     uz_t3 = 0.0
        #     uz_t = 0.5 * gamma_t * (1 + y / np.sqrt(R ** 2 + r ** 2))
        # elif np.abs(r - R) < 1e-6:
        #     K = sp.ellipk(k2_y)
        #     E = sp.ellipe(k2_y)
        #     if np.abs(y) < 1e-6:
        #         ur_t = -gamma_t / (2 * np.pi) * np.sqrt(R / r) * ((2.0 - k2_y) / k_y * K - 2. / k_y * E)
        #         uz_t1 = gamma_t / 4
        #         uz_t2 = 0
        #         uz_t3 = 0
        #     else:
        #         ur_t = -gamma_t / (2 * np.pi) * np.sqrt(R / r) * ((2.0 - k2_y) / k_y * K - 2. / k_y * E)
        #         uz_t1 = gamma_t / 4
        #         uz_t2 = gamma_t / 2 * (y * k_y) / (2 * np.pi * np.sqrt(r * R)) * K
        #         uz_t3 = 0
        # else:
        #     K = sp.ellipk(k2_y)
        #     E = sp.ellipe(k2_y)
        #     PI = float(sym.elliptic_pi(float(k2_0), float(k2_y)))
        #     ur_t = -gamma_t / (2 * np.pi) * np.sqrt(R / r) * ((2.0 - k2_y) / k_y * K - 2. / k_y * E)
        #     uz_t1 = gamma_t / 2. * (R - r + np.abs(R - r)) / (2 * np.abs(R - r))
        #     uz_t2 = gamma_t / 2. * (y * k_y) / (2 * np.pi * np.sqrt(r * R)) * K
        #     uz_t3 = gamma_t / 2. * (y * k_y) / (2 * np.pi * np.sqrt(r * R)) * (R - r) / (R + r) * PI
        #
        # uz_t = uz_t1 + uz_t2 + uz_t3

        # vortex cutoff in action
        if r < vortex_cutoff * R:
            ur_t = - 0.25 * gamma_t * (r * R ** 2) / (R ** 2 + y ** 2) ** 1.5

            uz_t1 = 0.0
            uz_t2 = 0.0
            uz_t3 = 0.0
            uz_t = 0.5 * gamma_t * (1 + y / np.sqrt(R ** 2 + r ** 2))
        # edge of wake cylinder
        elif np.abs(r - R) < 1e-6:
            # edge of propeller
            if np.abs(y) < 1e-6:
                ur_t = 0
                uz_t1 = gamma_t / 4
                uz_t2 = 0
                uz_t3 = 0
                uz_t = uz_t1 + uz_t2 + uz_t3
            else:
                K = sp.ellipk(k2_y)

                uz_t1 = gamma_t / 4
                uz_t2 = gamma_t / 2 * (y * k_y) / (2 * np.pi * np.sqrt(r * R)) * K
                uz_t3 = 0
                uz_t = uz_t1 + uz_t2 + uz_t3
        # standard case
        else:
            K = sp.ellipk(k2_y)
            E = sp.ellipe(k2_y)
            PI = float(sym.elliptic_pi(float(k2_0), float(k2_y)))

            ur_t = -gamma_t / (2 * np.pi) * np.sqrt(R / r) * ((2.0 - k2_y) / k_y * K - 2. / k_y * E)

            uz_t1 = gamma_t / 2. * (R - r + np.abs(R - r)) / (2 * np.abs(R - r))
            uz_t2 = gamma_t / 2. * (y * k_y) / (2 * np.pi * np.sqrt(r * R)) * K
            uz_t3 = gamma_t / 2. * (y * k_y) / (2 * np.pi * np.sqrt(r * R)) * (R - r) / (R + r) * PI
            uz_t = uz_t1 + uz_t2 + uz_t3


        return uz_t, ur_t

    # @staticmethod
    def propeller_wake_bound(self, r, R, gamma_tot, y, vortex_cutoff=0.01):
    # def propeller_wake_bound(r, R, gamma_tot, y, vortex_cutoff=0.01):
        up_b = 0.0
        # if r < vortex_cutoff * R or np.abs(y) < 1e-6:
        if r < vortex_cutoff * R or np.abs(y) < 1e-6:
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
        if r < vortex_cutoff * R:
        # if r < 0.01:
            pass
        else:
            up_r = -0.25 * gamma_tot / (np.pi * r) * (1.0 + y / np.sqrt(r ** 2 + y ** 2))
        return up_r

    def propeller_wake_longitudinal(self, r, R, gamma_l, y, vortex_cutoff=0.01):
        up_l = 0.0
        up_l1 = 0.0
        up_l2 = 0.0
        up_l3 = 0.0
        if r < R * vortex_cutoff:
        # if r < 0.01:
            up_l = 0
        elif np.abs(r - R) < 1e-6:
            if np.abs(y) < 1e-6:
                up_l = 0
            else:
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
        return up_l
