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


    def predictor(self, q, dqdt, dqddt):

        q += self.dt*dqdt + (0.5 - self.beta)*self.dt*self.dt*dqddt
        dqdt += (1.0 - self.gamma)*self.dt*dqddt
        dqddt *= 0.

    def build_matrix(self, M, C, K):

        Asys = K + C*self.gamma/(self.beta*self.dt) + M/(self.beta*self.dt*self.dt)

        return Asys

    def corrector(self, q, dqdt, dqddt, Dq):

        q += Dq
        dqdt += self.gamma/(self.beta*self.dt)*Dq
        dqddt += 1.0/(self.beta*self.dt*self.dt)*Dq
