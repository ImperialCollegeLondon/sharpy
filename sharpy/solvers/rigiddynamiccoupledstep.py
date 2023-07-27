import sharpy.structure.utils.xbeamlib as xbeamlib
from sharpy.utils.solver_interface import solver, BaseSolver, solver_from_string
import sharpy.utils.settings as settings_utils


_BaseStructural = solver_from_string('_BaseStructural')

@solver
class RigidDynamicCoupledStep(_BaseStructural):
    """
    Structural solver used for the dynamic simulation of free-flying rigid structures.

    This solver provides an interface to the structural library (``xbeam``) and updates the structural parameters
    for every k-th step in the FSI iteration.

    This solver can be called as part of a standalone structural simulation or as the structural solver of a coupled
    aeroelastic simulation.

    """
    solver_id = 'RigidDynamicCoupledStep'
    solver_classification = 'structural'

    settings_types = _BaseStructural.settings_types.copy()
    settings_default = _BaseStructural.settings_default.copy()
    settings_description = _BaseStructural.settings_description.copy()

    settings_types['balancing'] = 'bool'
    settings_default['balancing'] = False

    settings_types['relaxation_factor'] = 'float'
    settings_default['relaxation_factor'] = 0.3
    settings_description['relaxation factor'] = 'Relaxation factor'

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

        # load info from dyn dictionary
        self.data.structure.add_unsteady_information(self.data.structure.dyn_dict, self.settings['num_steps'])

        # generate q, dqdt and dqddt
        xbeamlib.xbeam_solv_disp2state(self.data.structure, self.data.structure.timestep_info[-1])

    def run(self, **kwargs):

        structural_step = settings_utils.set_value_or_default(kwargs, 'structural_step', self.data.structure.timestep_info[-1])
        # TODO: previous_structural_step never used
        previous_structural_step = settings_utils.set_value_or_default(kwargs, 'previous_structural_step', self.data.structure.timestep_info[-1])
        dt= settings_utils.set_value_or_default(kwargs, 'dt', self.settings['dt'])  

        xbeamlib.xbeam_step_coupledrigid(self.data.structure,
                                         self.settings,
                                         self.data.ts,
                                         structural_step,
                                         dt=dt)
        self.extract_resultants(structural_step)
        self.data.structure.integrate_position(structural_step, dt)

        return self.data

    def add_step(self):
        self.data.structure.next_step()

    def next_step(self):
        pass

    def extract_resultants(self, tstep=None):
        if tstep is None:
            tstep = self.data.structure.timestep_info[self.data.ts]
        steady, unsteady, grav = tstep.extract_resultants(self.data.structure, force_type=['steady', 'unsteady', 'grav'])
        totals = steady + unsteady + grav
        return totals[0:3], totals[3:6]

