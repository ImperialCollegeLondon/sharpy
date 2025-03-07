import numpy as np
from abc import abstractmethod
import typing

import sharpy.utils.settings as settings_utils
from sharpy.utils.solver_interface import solver

arr: typing.Type = np.ndarray


@solver
class _BaseTimeIntegrator:
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

    @abstractmethod
    def initialise(self, data, custom_settings=None, restart=False):
        pass

    @abstractmethod
    def predictor(self, q: arr, dqdt: arr, dqddt: arr):
        pass

    @abstractmethod
    def build_matrix(self, m: arr, c: arr, k: arr):
        pass

    @abstractmethod
    def corrector(self, q: arr, dqdt: arr, dqddt: arr, dq: arr):
        pass


@solver
class NewmarkBetaJAX(_BaseTimeIntegrator):
    """
    Time integration according to the Newmark-beta scheme
    """

    solver_id = 'NewmarkBetaJAX'
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
    settings_description['num_LM_eq'] = 'Number of constraint equations'

    def __init__(self):
        super().__init__()  # I know the base class has no function here, but this makes Pycharm leave me alone
        self.dt = None
        self.beta = None
        self.gamma = None
        self.sys_size = None
        self.num_lm_eq = None
        self.settings = None

    def initialise(self, data, custom_settings=None, restart=False) -> None:

        if custom_settings is None:
            self.settings = data.input_settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings_utils.to_custom_types(self.settings,
                                       self.settings_types,
                                       self.settings_default,
                                       no_ctype=True)

        self.dt = self.settings['dt']
        self.gamma = 0.5 + self.settings['newmark_damp']
        self.beta = 0.25 * (self.gamma + 0.5) * (self.gamma + 0.5)

        self.sys_size = self.settings['sys_size']
        self.num_lm_eq = self.settings['num_LM_eq']

    def predictor(self, q, dqdt, dqddt):
        q[:self.sys_size] += (self.dt * dqdt[:self.sys_size]
                              + (0.5 - self.beta) * self.dt * self.dt * dqddt[:self.sys_size])
        dqdt[:self.sys_size] += (1. - self.gamma) * self.dt * dqddt[:self.sys_size]
        dqddt.fill(0.)

    def build_matrix(self, m: arr, c: arr, k: arr) -> arr:
        a_sys = np.zeros((self.sys_size + self.num_lm_eq, self.sys_size + self.num_lm_eq))
        a_sys[:, :self.sys_size] = (k[:, :self.sys_size]
                                    + c[:, :self.sys_size] * self.gamma / (self.beta * self.dt)
                                    + m[:, :self.sys_size] / (self.beta * self.dt * self.dt))
        a_sys[:, self.sys_size:] = k[:, self.sys_size:] + c[:, self.sys_size:]
        return a_sys

    def corrector(self, q: arr, dqdt: arr, dqddt: arr, dq: arr) -> None:
        q[:self.sys_size] += dq[:self.sys_size]
        dqdt[:self.sys_size] += self.gamma / (self.beta * self.dt) * dq[:self.sys_size]
        dqddt[:self.sys_size] += 1. / (self.beta * self.dt * self.dt) * dq[:self.sys_size]
        dqdt[self.sys_size:] += dq[self.sys_size:]


@solver
class GeneralisedAlphaJAX(_BaseTimeIntegrator):
    """
    Time integration according to the Generalised-Alpha scheme
    """

    solver_id = 'GeneralisedAlphaJAX'
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

    settings_types['sys_size'] = 'int'
    settings_default['sys_size'] = 0
    settings_description['sys_size'] = 'Size of the system without constraints'

    settings_types['num_LM_eq'] = 'int'
    settings_default['num_LM_eq'] = 0
    settings_description['num_LM_eq'] = 'Number of contraint equations'

    def __init__(self):
        super().__init__()
        self.dt = None
        self.am = None
        self.af = None
        self.gamma = None
        self.beta = None
        self.om_am = None
        self.om_af = None
        self.sys_size = None
        self.num_lm_eq = None

    def initialise(self, data, custom_settings=None, restart=False) -> None:

        if custom_settings is None:
            self.settings = data.input_settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings_utils.to_custom_types(self.settings,
                                       self.settings_types,
                                       self.settings_default,
                                       no_ctype=True)

        self.dt = self.settings['dt']
        self.am = self.settings['am']
        self.af = self.settings['af']
        self.om_am = 1. - self.am
        self.om_af = 1. - self.af
        self.gamma = 0.5 - self.am + self.af
        self.beta = 0.25 * (1. - self.am + self.af) ** 2
        self.sys_size = self.settings['sys_size']
        self.num_lm_eq = self.settings['num_LM_eq']

    def predictor(self, q: arr, dqdt: arr, dqddt: arr):
        q[:self.sys_size] += (self.dt * dqdt[:self.sys_size]
                              + (0.5 - self.beta) * self.dt * self.dt * dqddt[:self.sys_size])
        dqdt[:self.sys_size] += (1. - self.gamma) * self.dt * dqddt[:self.sys_size]
        dqddt.fill(0.)

    def build_matrix(self, m: arr, c: arr, k: arr) -> arr:
        a_sys = np.zeros((self.sys_size + self.num_lm_eq, self.sys_size + self.num_lm_eq))
        a_sys[:, :self.sys_size] = (self.om_af * k
                                    + self.gamma * self.om_af / (self.beta * self.dt) * c
                                    + self.om_am / (self.beta * self.dt * self.dt) * m)

        a_sys[:, self.sys_size:] = k[:, self.sys_size:] + c[:, self.sys_size:]
        return a_sys

    def corrector(self, q: arr, dqdt: arr, dqddt: arr, dq: arr) -> None:
        q[:self.sys_size] += self.om_af * dq[:self.sys_size]
        dqdt[:self.sys_size] += self.gamma * self.om_af / self.beta / self.dt * dq[:self.sys_size]
        dqddt[:self.sys_size] += self.om_am / self.beta / self.dt ** 2 * dq[:self.sys_size]
        dqdt[self.sys_size:] += dq[self.sys_size:]
