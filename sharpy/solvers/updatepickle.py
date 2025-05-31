import numpy as np
import sharpy.utils.settings as settings_utils
from sharpy.utils.solver_interface import solver, BaseSolver


@solver
class UpdatePickle(BaseSolver):
    """

    """
    solver_id = 'UpdatePickle'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.data = None
        self.settings = None

    def initialise(self, data, custom_settings=None, restart=False):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings_utils.to_custom_types(self.settings,
                           self.settings_types,
                           self.settings_default,
                           no_ctype=True)

    def run(self, **kwargs):

        for sts in self.data.structure.timestep_info:
            if sts is not None:
                sts.in_global_AFoR = True
                sts.runtime_steady_forces = np.zeros_like(sts.steady_applied_forces)
                sts.runtime_unsteady_forces = np.zeros_like(sts.steady_applied_forces)
                sts.psi_local = sts.psi.copy()
                sts.psi_dot_local = sts.psi_dot.copy()
                sts.mb_dquatdt = np.zeros_like(sts.mb_quat)

        return self.data

