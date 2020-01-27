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

    def generate(self, params):
        # Renaming for convenience
        zeta = params['zeta']
        zeta_star = params['zeta_star']
        gamma = params['gamma']
        gamma_star = params['gamma_star']

        nsurf = len(zeta)
        deltax = self.u_inf*self.u_inf_direction*self.dt
        for isurf in range(nsurf):
            M, N = zeta_star[isurf][0, :, :].shape
            for i in range(M):
                for j in range(N):
                    zeta_star[isurf][:, i, j] = zeta[isurf][:, -1, j] + deltax*i
            gamma[isurf] *= 0.
            gamma_star[isurf] *= 0.
