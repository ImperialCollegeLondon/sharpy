import numpy as np

import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings


@generator_interface.generator
class StraightWake(generator_interface.BaseGenerator):
    r"""
    Straight wake shape generator

    ``StraightWake`` class inherited from ``BaseGenerator``

    The object creates a straight wake shedding from the trailing edge based on
    the time step ``dt``, the incoming velocity magnitude ``u_inf`` and
    direction ``u_inf_direction``
    """
    generator_id = 'StraightWake'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['u_inf'] = 'float'
    settings_default['u_inf'] = None
    settings_description['u_inf'] = 'Free stream velocity magnitude'

    settings_types['u_inf_direction'] = 'list(float)'
    settings_default['u_inf_direction'] = np.array([1.0, 0, 0])
    settings_description['u_inf_direction'] = '``x``, ``y`` and ``z`` relative components of the free stream velocity'

    settings_types['dt'] = 'float'
    settings_default['dt'] = None
    settings_description['dt'] = 'Time step'

    settings_types['dx1'] = 'float'
    settings_default['dx1'] = -1.0
    settings_description['dx1'] = 'Size of the first wake panel'

    settings_types['ndx1'] = 'int'
    settings_default['ndx1'] = -1
    settings_description['ndx1'] = 'Number of panels with size ``dx1``'

    settings_types['r'] = 'float'
    settings_default['r'] = 1.
    settings_description['r'] = 'Growth rate after ``ndx1`` panels'

    settings_types['dxmax'] = 'float'
    settings_default['dxmax'] = -1.0
    settings_description['dxmax'] = 'Maximum panel size'

    setting_table = settings.SettingsTable()
    __doc__ += setting_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.in_dict = dict()

        self.u_inf = 0.
        self.u_inf_direction = None
        self.dt = None

    def initialise(self, in_dict):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict, self.settings_types, self.settings_default)

        self.u_inf = self.in_dict['u_inf']
        self.u_inf_direction = self.in_dict['u_inf_direction']
        self.dt = self.in_dict['dt']

        try:
            self.dx1 = self.in_dict['dx1']
        except KeyError:
            self.dx1 = self.u_inf*self.dt

        try:
            self.ndx1 = self.in_dict['ndx1']
        except KeyError:
            self.ndx1 = -1

        try:
            self.r = self.in_dict['r']
        except KeyError:
            self.r = 1.

        try:
            self.dxmax = self.in_dict['dxmax']
        except KeyError:
            self.dxmax = self.dx1

    def generate(self, params):
        # Renaming for convenience
        zeta = params['zeta']
        zeta_star = params['zeta_star']
        gamma = params['gamma']
        gamma_star = params['gamma_star']

        nsurf = len(zeta)
        for isurf in range(nsurf):
            M, N = zeta_star[isurf][0, :, :].shape
            for j in range(N):
                zeta_star[isurf][:, 0, j] = zeta[isurf][:, -1, j]
                zeta_star[isurf][:, 1, j] = zeta_star[isurf][:, 0, j] + self.dx1*self.u_inf_direction
                for i in range(2, self.ndx1 + 1):
                    zeta_star[isurf][:, i, j] = zeta_star[isurf][:, i-1, j] + self.dx1*self.u_inf_direction
                for i in range(self.ndx1 + 1, M):
                    deltax = self.dx1*self.r**(i - self.ndx1)
                    if deltax > self.dxmax:
                        deltax = self.dxmax
                    zeta_star[isurf][:, i, j] = zeta_star[isurf][:, i-1, j] + deltax*self.u_inf_direction
            gamma[isurf] *= 0.
            gamma_star[isurf] *= 0.
