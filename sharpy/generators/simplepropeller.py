import numpy as np

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
        #

        self.settings_types['radius'] = 'float'
        self.settings_default['radius'] = None

        self.settings_types['omega'] = 'float'
        self.settings_default['omega'] = None

        self.settings_types['u_inf'] = 'float'
        self.settings_default['u_inf'] = None

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


    # PROPELLER MODEL ONLY
    # IN: U0
    # OUT: GAMMA_1, GAMMA_TOT, GAMMA_L
    def propeller_model(self, u0):
        r = self.settings['radius'].value
        omega = self.settings['omega'].value
        a = self.propeller_a(r, u0)
        a_prime = self.propeller_a(r, u0)

        gamma_1 = -2.0*a*u0
        gamma_tot = 4.0*np.pi*a_prime*omega*r*r
        gamma_l = gamma_tot*0.5/(np.pi*r)

        return (gamma_1, gamma_tot, gamma_l)

    def propeller_a(self, r, u0=None):
        if u0 is not None:
            raise NotImplementedError('u0 dependency of a not coded yet')

        mu = r/self.settings['radius'].value
        a = 1.0/3.0 - 2.0/(81.*self.lambd**2*mu**2) + 10./(729.0*self.lambd**4*mu**4)
        return a

    def propeller_a_prime(self, r, u0=None):
        if u0 is not None:
            raise NotImplementedError('u0 dependency of a not coded yet')
        mu = r/self.settings['radius'].value
        a_prime = 2.0/(9.*self.lambd**2*mu**2) + 14./(243.0*self.lambd**4*mu**4)
        return a_prime

    # PROPELLER WAKE MODEL
    # IN: GAMMA_1, GAMMA_TOT, GAMMA_L, ZETA, POS (G FoR), ORIENTATION(G FoR)
    # OUT: U_EXT AT THE GRID POINTS ZETA
    def propeller_wake(self, gamma_1, gamma_tot, gamma_l, zeta, pos_g, orientation_g):
        # the propeller (P) FoR is given by x_p, y_p, z_p
        # y_p = orientation_g
        # z_p = [0, 0, 1]_g projected in the plane with normal y_p
        z_g = np.array([0.0, 0.0, 1.0])
        z_g_projected = algebra.project_vect_on_plane(z_g, orientation_g)

        x_g = np.cross(orientation_g, z_g_projected)

        # vectors are packed by rows
        # triad_g = [x, y, z]^T
        triad_g = np.array([x_g, orientation_g, z_g_projected])

        # rotation matrix from p to G
        rot_pg = triad_g.T

        def p2cyl(coords_p):
            r = np.sqrt(coords_p[0]**2 + coords_p[2]**2)
            y = coords_p[1]
            try:
                psi = np.arccos(coords_p[0]/r)
            except ZeroDivisionEror:
                psi = 0.0
            if coords_p[2] < 0:
                psi = 2.0*np.pi - psi
            return np.array([r, y, psi])






        for i_surf in range(len(zeta)):
            for i_M in range(zeta[i_surf].shape[0]):
                for i_N in range(zeta[i_surf].shape[1]):
                    # coordinates of the grid point in G for
                    grid_g = zeta[i_surf][:, i_M, i_N]

                    x_g = grid_g - pos_g

                    

















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
        elif self.settings['gust_shape'] == 'DARPA':
            def gust_shape(x, y, z, gust_length, gust_intensity, span=0):
                vel = np.zeros((3,))
                if x > 0.0 or x < -gust_length:
                    return vel

                vel[2] = (1.0 - np.cos(2.0*np.pi*x/gust_length))*gust_intensity*0.5
                vel[2] *= -np.cos(y/span*np.pi)
                return vel

        elif self.settings['gust_shape'] == 'continuous_sin':
            def gust_shape(x, y, z, gust_length, gust_intensity, span=0):
                vel = np.zeros((3,))
                if x > 0.0:
                    return vel

                vel[2] = 0.5 * gust_intensity * np.sin(2 * np.pi * x / gust_length)
                return vel
        elif self.settings['gust_shape'] == 'lateral 1-cos':
            def gust_shape(x, y, z, gust_length, gust_intensity, span=0):
                vel = np.zeros((3,))
                if x > 0.0 or x < -gust_length:
                    return vel

                vel[1] = (1.0 - np.cos(2.0*np.pi*x/gust_length))*gust_intensity*0.5
                return vel
        elif self.settings['gust_shape'] == 'time varying':
            def gust_shape(x, y, z, gust_length, gust_intensity, span=0):
                vel = np.zeros((3,))
                if x > 0.0:
                    return vel

                vel[0] = np.interp(x, -self.file_info[:,0]*self.u_inf, self.file_info[:,1])
                vel[1] = np.interp(x, -self.file_info[:,0]*self.u_inf, self.file_info[:,2])
                vel[2] = np.interp(x, -self.file_info[:,0]*self.u_inf, self.file_info[:,3])
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
