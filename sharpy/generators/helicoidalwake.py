import numpy as np

import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra


@generator_interface.generator
class HelicoidalWake(generator_interface.BaseGenerator):
    r"""
    Helicoidal wake shape generator

    ``HelicoidalWake`` class inherited from ``BaseGenerator``

    The object creates a helicoidal wake shedding from the trailing edge based on
    the time step ``dt``, the incoming velocity magnitude ``u_inf``,
    direction ``u_inf_direction``, the rotation velocity ``rotation_velocity`` and
    the shear parameters
    """
    generator_id = 'HelicoidalWake'
    generator_classification = 'wake'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['u_inf'] = 'float'
    settings_default['u_inf'] = None
    settings_description['u_inf'] = 'Free stream velocity magnitude'

    settings_types['u_inf_direction'] = 'list(float)'
    settings_default['u_inf_direction'] = None
    settings_description['u_inf_direction'] = '``x``, ``y`` and ``z`` relative components of the free stream velocity'

    settings_types['dt'] = 'float'
    settings_default['dt'] = None
    settings_description['dt'] = 'Time step'

    settings_types['dphi1'] = 'float'
    settings_default['dphi1'] = -1.0
    settings_description['dphi1'] = 'Size of the first wake panel in radians'

    settings_types['ndphi1'] = 'int'
    settings_default['ndphi1'] = 1
    settings_description['ndphi1'] = 'Number of panels with size ``dphi1``'

    settings_types['r'] = 'float'
    settings_default['r'] = 1.
    settings_description['r'] = 'Growth rate after ``ndphi1`` panels'

    settings_types['dphimax'] = 'float'
    settings_default['dphimax'] = -1.0
    settings_description['dphimax'] = 'Maximum panel size in radians'

    # Shear parameters
    settings_types['shear_direction'] = 'list(float)'
    settings_default['shear_direction'] = np.array([1.0, 0, 0])
    settings_description['shear_direction'] = '``x``, ``y`` and ``z`` relative components of the direction along which shear applies'

    settings_types['shear_exp'] = 'float'
    settings_default['shear_exp'] = 0.
    settings_description['shear_exp'] = 'Exponent of the shear law'

    settings_types['h_ref'] = 'float'
    settings_default['h_ref'] = 1.
    settings_description['h_ref'] = 'Reference height at which ``u_inf`` is defined'

    settings_types['h_corr'] = 'float'
    settings_default['h_corr'] = 1.
    settings_description['h_corr'] = 'Height to correct the shear law'

    # Rotation
    settings_types['rotation_velocity'] = 'list(float)'
    settings_default['rotation_velocity'] = None
    settings_description['rotation_velocity'] = 'Rotation velocity'

    setting_table = settings.SettingsTable()
    __doc__ += setting_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.in_dict = dict()

        self.u_inf = 0.
        self.u_inf_direction = None
        self.rotation_velocity = None
        self.dt = None

        self.dphi1 = None
        self.ndphi1 = None
        self.r = None
        self.dphimax = None

        self.shear_direction = None
        self.shear_exp = None
        self.h_ref = None
        self.h_corr = None

    def initialise(self, data, in_dict, restart=False):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict, self.settings_types, self.settings_default,  no_ctype=True)

        self.u_inf = self.in_dict['u_inf']
        self.u_inf_direction = self.in_dict['u_inf_direction']
        self.rotation_velocity = self.in_dict['rotation_velocity']
        self.dt = self.in_dict['dt']

        if self.in_dict['dphi1'] == -1:
            self.dphi1 = np.linalg.norm(self.rotation_velocity)*self.dt
        else:
            self.dphi1 = self.in_dict['dphi1']

        self.ndphi1 = self.in_dict['ndphi1']
        self.r = self.in_dict['r']

        if self.in_dict['dphimax'] == -1:
            self.dphimax = self.dphi1
        else:
            self.dphimax = self.in_dict['dphimax']

        self.shear_direction = self.in_dict['shear_direction']
        self.shear_exp = self.in_dict['shear_exp']
        self.h_ref = self.in_dict['h_ref']
        self.h_corr = self.in_dict['h_corr']


    def generate(self, params):
        # Renaming for convenience
        zeta = params['zeta']
        zeta_star = params['zeta_star']
        gamma = params['gamma']
        gamma_star = params['gamma_star']
        dist_to_orig = params['dist_to_orig']

        nsurf = len(zeta)
        for isurf in range(nsurf):
            M, N = zeta_star[isurf][0, :, :].shape
            angle = 0.
            for i in range(M):
                # Compute the step in azimuthal angle
                angle -= self.get_dphi(i, self.dphi1, self.ndphi1, self.r, self.dphimax)

                delta_t = -angle/np.linalg.norm(self.rotation_velocity)
                rot = algebra.rotation_matrix_around_axis(algebra.unit_vector(self.rotation_velocity), angle)
                for j in range(N):
                    # Define the helicoidal
                    aux_zeta_TE = zeta[isurf][:, -1, j] - (self.h_ref - self.h_corr)*self.shear_direction
                    aux_zeta_TE = np.dot(rot, aux_zeta_TE) + (self.h_ref - self.h_corr)*self.shear_direction

                    # Translate according to u_inf depending on the height
                    h = np.dot(aux_zeta_TE, self.shear_direction) + self.h_corr
                    zeta_star[isurf][:, i, j] = aux_zeta_TE + self.u_inf*self.u_inf_direction*(h/self.h_ref)**self.shear_exp*delta_t
                    # zeta_star[isurf][:, i, j] = zeta[isurf][:, -1, j] + self.u_inf*self.u_inf_direction*self.dt*i
                    # print(zeta_star[isurf][:, i, j])

            gamma[isurf] *= 0.
            gamma_star[isurf] *= 0.

        for isurf in range(nsurf):
            M, N = zeta_star[isurf][0, :, :].shape
            dist_to_orig[isurf][0, :] = 0.
            for j in range(0, N):
                for i in range(1, M):
                    dist_to_orig[isurf][i, j] = (dist_to_orig[isurf][i - 1, j] +
                                          np.linalg.norm(zeta_star[isurf][:, i, j] -
                                                         zeta_star[isurf][:, i - 1, j]))
                dist_to_orig[isurf][:, j] /= dist_to_orig[isurf][-1, j]

    @staticmethod
    def get_dphi(i, dphi1, ndphi1, r, dphimax):
        if i == 0:
            dphi = 0.
        elif i <= ndphi1:
            dphi = dphi1
        else:
            dphi = dphi1*r**(i - ndphi1)
        dphi = min(dphi, dphimax)

        return dphi
