import numpy as np
import h5py as h5

from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra
import sharpy.utils.h5utils as h5utils

@solver
class PrescribedStructure(BaseSolver):
    r"""
        This solver prescribes pos, pos_dot, psi, psi_dot and for_vel
        at each time step from a .h5 file
    """
    solver_id = 'PrescribedStructure'
    solver_classification = 'structural'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['input_file'] = 'str'
    settings_default['input_file'] = None
    settings_description['input_file'] = 'Input file containing the simulation data'

    settings_types['dt'] = 'float'
    settings_default['dt'] = 0.01
    settings_description['dt'] = 'Time step of simulation'

    settings_types['num_steps'] = 'int'
    settings_default['num_steps'] = 500
    settings_description['num_steps'] = 'Number of timesteps to be run'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.data = None
        self.settings = None
        self.sim_info = None

    def initialise(self, data, custom_settings=None):

        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings.to_custom_types(self.settings,
                                 self.settings_types,
                                 self.settings_default,
                                 no_ctype=True)

        # Load simulation data
        file_info = h5utils.readh5(self.settings['input_file'])
        self.sim_info = [None]*(self.settings['num_steps'] + 1)
        for it in range(self.settings['num_steps'] + 1):
            self.sim_info[it] = getattr(file_info, '%0.5d' % it)
    

    def run(self, structural_step=None, dt=None):

        if structural_step is None:
            structural_step = self.data.structure.timestep_info[-1]
        if dt is None:
            dt = self.settings['dt']

        # Prescribe the information from file
        structural_step.pos = self.sim_info[self.data.ts].pos
        structural_step.pos_dot = self.sim_info[self.data.ts].pos_dot
        structural_step.psi = self.sim_info[self.data.ts].psi
        structural_step.psi_dot = self.sim_info[self.data.ts].psi_dot
        structural_step.for_vel = self.sim_info[self.data.ts - 1].for_vel

        Temp = np.linalg.inv(np.eye(4) + 0.25*algebra.quadskew(structural_step.for_vel[3:6])*dt)
        structural_step.quat = np.dot(Temp, np.dot(np.eye(4) - 0.25*algebra.quadskew(structural_step.for_vel[3:6])*dt, structural_step.quat))

        self.extract_resultants(structural_step)
        if self.data.ts > 0:
            self.data.structure.integrate_position(structural_step, self.settings['dt'])
        return self.data

    def add_step(self):
        self.data.structure.next_step()

    def next_step(self):
        pass

    def extract_resultants(self, step=None):
        if step is None:
            step = self.data.structure.timestep_info[-1]
        applied_forces = self.data.structure.nodal_b_for_2_a_for(step.steady_applied_forces + step.unsteady_applied_forces,
                                                                 step)

        applied_forces_copy = applied_forces.copy()
        gravity_forces_copy = step.gravity_forces.copy()
        for i_node in range(self.data.structure.num_node):
            applied_forces_copy[i_node, 3:6] += algebra.cross3(step.pos[i_node, :],
                                                               applied_forces_copy[i_node, 0:3])
            gravity_forces_copy[i_node, 3:6] += algebra.cross3(step.pos[i_node, :],
                                                               gravity_forces_copy[i_node, 0:3])

        totals = np.sum(applied_forces_copy + gravity_forces_copy, axis=0)
        step.total_forces = np.sum(applied_forces_copy, axis=0)
        step.total_gravity_forces = np.sum(gravity_forces_copy, axis=0)
        return totals[0:3], totals[3:6]

    def update(self, tstep=None):
        self.create_q_vector(tstep)

    def create_q_vector(self, tstep=None):
        import sharpy.structure.utils.xbeamlib as xb
        if tstep is None:
            tstep = self.data.structure.timestep_info[-1]

        xb.xbeam_solv_disp2state(self.data.structure, tstep)
