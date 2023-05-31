import numpy as np

import sharpy.utils.settings as settings_utils
from sharpy.utils.solver_interface import solver, BaseSolver, solver_from_string

_BaseStructural = solver_from_string('_BaseStructural')

@solver
class NoStructural(_BaseStructural):
    """
    Structural solver used for the static simulation of free-flying structures.

    This solver provides an interface to the structural library (``xbeam``) and updates the structural parameters
    for every k-th step of the FSI iteration.

    This solver can be called as part of a standalone structural simulation or as the structural solver of a coupled
    static aeroelastic simulation.

    """
    solver_id = 'NoStructural'
    solver_classification = 'structural'

    # settings list
    settings_types = _BaseStructural.settings_types.copy()
    settings_default = _BaseStructural.settings_default.copy()
    settings_description = _BaseStructural.settings_description.copy()

    settings_types['initial_position'] = 'list(float)'
    settings_default['initial_position'] = np.array([0.0, 0.0, 0.0])

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
        settings_utils.to_custom_types(self.settings, self.settings_types, self.settings_default)

    def run(self, **kwargs):
        self.data.structure.timestep_info[self.data.ts].for_pos[0:3] = self.settings['initial_position']
        self.extract_resultants()
        return self.data

    def next_step(self):
        self.data.structure.next_step()

    def extract_resultants(self, tstep=None):
        if tstep is None:
            tstep = self.data.structure.timestep_info[self.data.ts]
        steady, grav = tstep.extract_resultants(self.data.structure, force_type=['steady', 'grav'])
        totals = steady + grav
        return totals[0:3], totals[3:6]

    def update(self, tstep=None):
        self.create_q_vector(tstep)

    def create_q_vector(self, tstep=None):
        import sharpy.structure.utils.xbeamlib as xb
        if tstep is None:
            tstep = self.data.structure.timestep_info[-1]

        xb.xbeam_solv_disp2state(self.data.structure, tstep)


