"""
@modified   Alfonso del Carre
"""

import sharpy.structure.utils.xbeamlib as xbeamlib
from sharpy.utils.settings import str2bool
from sharpy.utils.solver_interface import solver, solver_from_string
import sharpy.utils.settings as settings_utils
import sharpy.utils.cout_utils as cout

_BaseStructural = solver_from_string('_BaseStructural')


@solver
class NonLinearDynamic(_BaseStructural):
    """
    Structural solver used for the dynamic simulation of free-flying structures.

    This solver provides an interface to the structural library (``xbeam``) and updates the structural parameters
    for every time step of the simulation.

    This solver is called as part of a standalone structural simulation.

    """
    solver_id = 'NonLinearDynamic'
    solver_classification = 'structural'

    settings_types = _BaseStructural.settings_types.copy()
    settings_default = _BaseStructural.settings_default.copy()
    settings_description = _BaseStructural.settings_description.copy()

    settings_types['prescribed_motion'] = 'bool'
    settings_default['prescribed_motion'] = None

    settings_types['gravity_dir'] = 'list(float)'
    settings_default['gravity_dir'] = [0, 0, 1]

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.data = None
        self.settings = None

    def initialise(self, data, restart=False):
        self.data = data
        self.settings = data.settings[self.solver_id]
        settings_utils.to_custom_types(self.settings, self.settings_types, self.settings_default)

        # load info from dyn dictionary
        self.data.structure.add_unsteady_information(self.data.structure.dyn_dict, self.settings['num_steps'])

        # allocate timestep_info
        for i in range(self.settings['num_steps']):
            self.data.structure.add_timestep(self.data.structure.timestep_info)
            if i>0:
                self.data.structure.timestep_info[i].unsteady_applied_forces[:] = self.data.structure.dynamic_input[i - 1]['dynamic_forces']
            self.data.structure.timestep_info[i].steady_applied_forces[:] = self.data.structure.ini_info.steady_applied_forces


    def run(self, **kwargs):
        prescribed_motion = False
        try:
            prescribed_motion = self.settings['prescribed_motion']
        except KeyError:
            pass
        if prescribed_motion is True:
            cout.cout_wrap('Running non linear dynamic solver...', 2)
            # raise NotImplementedError
            xbeamlib.cbeam3_solv_nlndyn(self.data.structure, self.settings)
        else:
            cout.cout_wrap('Running non linear dynamic solver with RB...', 2)
            xbeamlib.xbeam_solv_couplednlndyn(self.data.structure, self.settings)

        self.data.ts = self.settings['num_steps']
        cout.cout_wrap('...Finished', 2)
        return self.data
