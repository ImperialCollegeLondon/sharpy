import numpy as np
import ctypes as ct

import sharpy.utils.settings as settings
from sharpy.utils.solver_interface import solver


@solver
class _BaseTimeIntegrator():
    """
    Base structure for time integrators
    """

    solver_id = '_BaseTimeIntegrator'
    solver_classification = 'time_integrator'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    def __init__(self):
        pass


    def initialise(self, data, custom_settings=None):
        pass


    def predictor(self, q, dqdt, dqddt):
        pass


    def build_matrix(self, M, C, K):
        pass


    def corrector(self, q, dqdt, dqddt, Dq):
        pass


@solver
class NewmarkBeta(_BaseTimeIntegrator):
    """
    Time integration according to the Newmark-beta scheme
    """

    solver_id = 'NewmarkBeta'
    solver_classification = 'time_integrator'

    settings_types = _BaseTimeIntegrator.settings_types.copy()
    settings_default = _BaseTimeIntegrator.settings_default.copy()
    settings_description = _BaseTimeIntegrator.settings_description.copy()
    settings_options = _BaseTimeIntegrator.settings_options.copy()

    settings_types['dt'] = 'float'
    settings_default['dt'] = None
    settings_description['dt'] = 'Time step'

    settings_types['newmark_damp'] = 'float'
    settings_default['newmark_damp'] = 1e-4
    settings_description['newmark_damp'] = 'Newmark damping coefficient'

    settings_types['sys_size'] = 'int'
    settings_default['sys_size'] = 0
    settings_description['sys_size'] = 'Size of the system without constraints'

    settings_types['num_LM_eq'] = 'int'
    settings_default['num_LM_eq'] = 0
    settings_description['num_LM_eq'] = 'Number of contraint equations'

    def __init__(self):

        self.dt = None
        self.beta = None
        self.gamma = None

    def initialise(self, data, custom_settings=None):

        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings.to_custom_types(self.settings,
                                 self.settings_types,
                                 self.settings_default,
                                 no_ctype=True)

        self.dt = self.settings['dt']
        self.gamma = 0.5 + self.settings['newmark_damp']
        self.beta = 0.25*(self.gamma + 0.5)*(self.gamma + 0.5)

        self.sys_size = self.settings['sys_size']
        self.num_LM_eq = self.settings['num_LM_eq']

    def predictor(self, q, dqdt, dqddt):

        sys_size = self.sys_size
        q[:sys_size] += self.dt*dqdt[:sys_size] + (0.5 - self.beta)*self.dt*self.dt*dqddt[:sys_size]
        dqdt[:sys_size] += (1.0 - self.gamma)*self.dt*dqddt[:sys_size]
        dqddt *= 0.

        # q[sys_size:] = q[sys_size:]
        dqdt[sys_size:] = dqdt[sys_size:]

    def build_matrix(self, M, C, K, Q, kBnh, LM_Q):

        sys_size = self.sys_size
        num_LM_eq = self.num_LM_eq

        Asys = np.zeros((sys_size + num_LM_eq, sys_size + num_LM_eq),
                         dtype=ct.c_double, order='F')
        Qout = np.zeros((sys_size + num_LM_eq), dtype=ct.c_double, order='F')

        Asys[:sys_size, :sys_size] = K + C*self.gamma/(self.beta*self.dt) + M/(self.beta*self.dt*self.dt)
        Qout[:sys_size] = Q.copy()

        Asys[sys_size:, :sys_size] = (self.gamma/self.beta/self.dt)*kBnh
        Asys[:sys_size, sys_size:] = kBnh.T
        Qout[sys_size:] = LM_Q.copy()

        return Asys, Qout

    def corrector(self, q, dqdt, dqddt, Dq):

        sys_size = self.sys_size
        num_LM_eq = self.num_LM_eq

        q[:sys_size] += Dq[:sys_size]
        dqdt[:sys_size] += self.gamma/(self.beta*self.dt)*Dq[:sys_size]
        dqddt[:sys_size] += 1.0/(self.beta*self.dt*self.dt)*Dq[:sys_size]

        dqdt[sys_size:] += Dq[sys_size:]


@solver
class GeneralisedAlpha(_BaseTimeIntegrator):
    """
    Time integration according to the Generalised-Alpha scheme
    """

    solver_id = 'GeneralisedAlpha'
    solver_classification = 'time_integrator'

    settings_types = _BaseTimeIntegrator.settings_types.copy()
    settings_default = _BaseTimeIntegrator.settings_default.copy()
    settings_description = _BaseTimeIntegrator.settings_description.copy()
    settings_options = _BaseTimeIntegrator.settings_options.copy()

    settings_types['dt'] = 'float'
    settings_default['dt'] = None
    settings_description['dt'] = 'Time step'

    settings_types['am'] = 'float'
    settings_default['am'] = 0.
    settings_description['am'] = 'alpha_M coefficient'

    settings_types['af'] = 'float'
    settings_default['af'] = 0.1
    settings_description['af'] = 'alpha_F coefficient'


    def __init__(self):

        self.dt = None
        self.am = None
        self.af = None
        self.gamma = None
        self.beta = None

    def initialise(self, data, custom_settings=None):

        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings.to_custom_types(self.settings,
                                 self.settings_types,
                                 self.settings_default,
                                 no_ctype=True)

        self.dt = self.settings['dt']
        self.am = self.settings['am']
        self.af = self.settings['af']

        self.om_am = 1. - self.am
        self.om_af = 1. - self.af

        self.gamma = 0.5 - self.am + self.af
        self.beta = 0.25*(1. - self.am + self.af)**2


    def predictor(self, q, dqdt, dqddt):

        q += self.dt*dqdt + (0.5 - self.beta)*self.dt*self.dt*dqddt
        dqdt += (1.0 - self.gamma)*self.dt*dqddt
        dqddt *= 0.


    def build_matrix(self, M, C, K, Q):

        Asys = (self.om_af*K +
                self.gamma*self.om_af/self.beta/self.dt*C +
                self.om_am/(self.beta*self.dt*self.dt)*M)

        Qout = Q.copy() # + np.dot(M, dqddt)
                #np.dot(C, self.dt*self.om_af*dqddt + dqdt) +
                #np.dot(K, 0.5*self.om_af*self.dt**2*dqddt + self.om_af*self.dt*dqdt + q))

        return Asys, Qout

    def corrector(self, q, dqdt, dqddt, Dq):

        # Values at the beginning of the time step
        # q_i = q.copy()
        # dqdt_i = dqdt.copy()
        # dqddt_i = dqddt.copy()

        # dqddt = dqddt_i + 1./(self.beta*self.dt*self.dt)*Dq
        # dqdt = dqdt_i + self.dt*dqddt_i + self.gamma/self.beta/self.dt*Dq
        # q = q_i + self.dt*dqdt_i + self.dt**2/2*dqddt_i + Dq

        q += self.om_af*Dq
        dqdt += self.gamma*self.om_af/self.beta/self.dt*Dq
        dqddt += self.om_am/self.beta/self.dt**2*Dq
