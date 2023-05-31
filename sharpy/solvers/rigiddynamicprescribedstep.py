"""
@modified   Alfonso del Carre
"""

import numpy as np

import sharpy.structure.utils.xbeamlib as xbeamlib
from sharpy.utils.solver_interface import solver, BaseSolver, solver_from_string
import sharpy.utils.settings as settings_utils
import sharpy.utils.algebra as algebra

_BaseStructural = solver_from_string('_BaseStructural')

@solver
class RigidDynamicPrescribedStep(BaseSolver):
    solver_id = 'RigidDynamicPrescribedStep'
    solver_classification = 'structural'

    settings_types = _BaseStructural.settings_types.copy()
    settings_default = _BaseStructural.settings_default.copy()
    settings_description = _BaseStructural.settings_description.copy()

    settings_types['dt'] = 'float'
    settings_default['dt'] = 0.01
    settings_description['dt'] = 'Time step of simulation'

    settings_types['num_steps'] = 'int'
    settings_default['num_steps'] = 500
    settings_description['num_steps'] = 'Number of timesteps to be run'

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

    def run(self, **kwargs):
        structural_step = settings_utils.set_value_or_default(kwargs, 'structural_step', self.data.structure.timestep_info[-1])
        dt= settings_utils.set_value_or_default(kwargs, 'dt', self.settings['dt'])  

        if self.data.ts > 0:
            try:
                structural_step.for_vel[:] = self.data.structure.dynamic_input[self.data.ts - 1]['for_vel']
                structural_step.for_acc[:] = self.data.structure.dynamic_input[self.data.ts - 1]['for_acc']
            except IndexError:
                pass

        Temp = np.linalg.inv(np.eye(4) + 0.25*algebra.quadskew(structural_step.for_vel[3:6])*dt)
        structural_step.quat = np.dot(Temp, np.dot(np.eye(4) - 0.25*algebra.quadskew(structural_step.for_vel[3:6])*dt, structural_step.quat))

        xbeamlib.cbeam3_solv_disp2state(self.data.structure, structural_step)

        # xbeamlib.cbeam3_step_nlndyn(self.data.structure,
        #                             self.settings,
        #                             self.data.ts,
        #                             structural_step,
        #                             dt=dt)
        self.extract_resultants(structural_step)
        if self.data.ts > 0:
            self.data.structure.integrate_position(structural_step, self.settings['dt'])
        return self.data

    def add_step(self):
        self.data.structure.next_step()

    def next_step(self):
        pass
        # self.data.structure.next_step()
        # ts = len(self.data.structure.timestep_info) - 1
        # if ts > 0:
        #     self.data.structure.timestep_info[ts].for_vel[:] = self.data.structure.dynamic_input[ts - 1]['for_vel']
        #     self.data.structure.timestep_info[ts].for_acc[:] = self.data.structure.dynamic_input[ts - 1]['for_acc']
        #     self.data.structure.timestep_info[ts].unsteady_applied_forces[:] = self.data.structure.dynamic_input[ts - 1]['dynamic_forces']

    def extract_resultants(self, tstep=None):
        if tstep is None:
            tstep = self.data.structure.timestep_info[self.data.ts]
        steady, unsteady, grav = tstep.extract_resultants(self.data.structure, force_type=['steady', 'unsteady', 'grav'])
        totals = steady + unsteady + grav
        return totals[0:3], totals[3:6]


    def update(self, tstep=None):
        self.create_q_vector(tstep)

    def create_q_vector(self, tstep=None):
        import sharpy.structure.utils.xbeamlib as xb
        if tstep is None:
            tstep = self.data.structure.timestep_info[-1]

        xb.xbeam_solv_disp2state(self.data.structure, tstep)
