import numpy as np
import os

import sharpy.utils.cout_utils as cout
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra


class ForcesContainer(object):
    def __init__(self):
        self.ts = 0
        self.t = 0.0
        self.forces = []
        self.coords = []


@solver
class AeroForcesCalculator(BaseSolver):
    """AeroForcesCalculator

    Calculates the total aerodynamic forces on the frame of reference ``A``.

    """
    solver_id = 'AeroForcesCalculator'
    solver_classification = 'post-processor'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['folder'] = 'str'
    settings_default['folder'] = './output'
    settings_description['folder'] = 'Output folder location'

    settings_types['write_text_file'] = 'bool'
    settings_default['write_text_file'] = False
    settings_description['write_text_file'] = 'Write ``txt`` file with results'

    settings_types['text_file_name'] = 'str'
    settings_default['text_file_name'] = ''
    settings_description['text_file_name'] = 'Text file name'

    settings_types['screen_output'] = 'bool'
    settings_default['screen_output'] = True
    settings_description['screen_output'] = 'Show results on screen'

    settings_types['unsteady'] = 'bool'
    settings_default['unsteady'] = False
    settings_description['unsteady'] = 'Include unsteady contributions'

    settings_default['coefficients'] = False
    settings_types['coefficients'] = 'bool'
    settings_description['coefficients'] = 'Calculate aerodynamic coefficients'

    settings_types['q_ref'] = 'float'
    settings_default['q_ref'] = 1
    settings_description['q_ref'] = 'Reference dynamic pressure'

    settings_types['S_ref'] = 'float'
    settings_default['S_ref'] = 1
    settings_description['S_ref'] = 'Reference area'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):

        self.settings = None
        self.data = None
        self.ts_max = 0
        self.ts = 0

        self.folder = ''

    def initialise(self, data):
        self.data = data
        self.settings = data.settings[self.solver_id]
        if self.data.structure.settings['unsteady']:
            self.ts_max = self.data.ts + 1
        else:
            self.ts_max = 1
            self.ts_max = len(self.data.structure.timestep_info)
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

    def run(self, online=False):
        self.ts = 0

        self.calculate_forces()
        if self.settings['write_text_file']:
            self.folder = (self.settings['folder'] + '/' +
                           self.data.settings['SHARPy']['case'] + '/' +
                           'forces/')
            # create folder for containing files if necessary
            if not os.path.exists(self.folder):
                os.makedirs(self.folder)
            self.folder += self.settings['text_file_name']
            self.file_output()
        if self.settings['screen_output']:
            self.screen_output()
        cout.cout_wrap('...Finished', 1)
        return self.data

    def calculate_forces(self):
        for self.ts in range(self.ts_max):
            rot = algebra.quat2rotation(self.data.structure.timestep_info[self.ts].quat)

            force = self.data.aero.timestep_info[self.ts].forces
            unsteady_force = self.data.aero.timestep_info[self.ts].dynamic_forces
            n_surf = len(force)
            for i_surf in range(n_surf):
                total_steady_force = np.zeros((3,))
                total_unsteady_force = np.zeros((3,))
                _, n_rows, n_cols = force[i_surf].shape
                for i_m in range(n_rows):
                    for i_n in range(n_cols):
                        total_steady_force += force[i_surf][0:3, i_m, i_n]
                        total_unsteady_force += unsteady_force[i_surf][0:3, i_m, i_n]
                self.data.aero.timestep_info[self.ts].inertial_steady_forces[i_surf, 0:3] = total_steady_force
                self.data.aero.timestep_info[self.ts].inertial_unsteady_forces[i_surf, 0:3] = total_unsteady_force
                self.data.aero.timestep_info[self.ts].body_steady_forces[i_surf, 0:3] = np.dot(rot.T, total_steady_force)
                self.data.aero.timestep_info[self.ts].body_unsteady_forces[i_surf, 0:3] = np.dot(rot.T, total_unsteady_force)

    def calculate_coefficients(self, fx, fy, fz):
        qS = self.settings['q_ref'].value * self.settings['S_ref'].value
        return fx/qS, fy/qS, fz/qS

    def screen_output(self):
        line = ''
        cout.cout_wrap.print_separator()
        # output header
        if self.settings['coefficients']:
            line = "{0:5s} | {1:10s} | {2:10s} | {3:10s} | {4:10s} | {5:10s} | {6:10s}".format(
                'tstep', '  fx_g', '  fy_g', '  fz_g', '  Cfx_g', '  Cfy_g', '  Cfz_g')
            cout.cout_wrap(line, 1)
            for self.ts in range(self.ts_max):
                fx = np.sum(self.data.aero.timestep_info[self.ts].inertial_steady_forces[:, 0], 0) + \
                     np.sum(self.data.aero.timestep_info[self.ts].inertial_unsteady_forces[:, 0], 0)

                fy = np.sum(self.data.aero.timestep_info[self.ts].inertial_steady_forces[:, 1], 0) + \
                     np.sum(self.data.aero.timestep_info[self.ts].inertial_unsteady_forces[:, 1], 0)

                fz = np.sum(self.data.aero.timestep_info[self.ts].inertial_steady_forces[:, 2], 0) + \
                     np.sum(self.data.aero.timestep_info[self.ts].inertial_unsteady_forces[:, 2], 0)

                Cfx, Cfy, Cfz = self.calculate_coefficients(fx, fy, fz)

                line = "{0:5d} | {1: 8.3e} | {2: 8.3e} | {3: 8.3e} | {4: 8.3e} | {5: 8.3e} | {6: 8.3e}".format(
                    self.ts, fx, fy, fz, Cfx, Cfy, Cfz)
                cout.cout_wrap(line, 1)
        else:
            line = "{0:5s} | {1:10s} | {2:10s} | {3:10s}".format(
                'tstep', '  fx_g', '  fy_g', '  fz_g')
            cout.cout_wrap(line, 1)
            for self.ts in range(self.ts_max):
                fx = np.sum(self.data.aero.timestep_info[self.ts].inertial_steady_forces[:, 0], 0) + \
                     np.sum(self.data.aero.timestep_info[self.ts].inertial_unsteady_forces[:, 0], 0)

                fy = np.sum(self.data.aero.timestep_info[self.ts].inertial_steady_forces[:, 1], 0) + \
                     np.sum(self.data.aero.timestep_info[self.ts].inertial_unsteady_forces[:, 1], 0)

                fz = np.sum(self.data.aero.timestep_info[self.ts].inertial_steady_forces[:, 2], 0) + \
                     np.sum(self.data.aero.timestep_info[self.ts].inertial_unsteady_forces[:, 2], 0)

                line = "{0:5d} | {1: 8.3e} | {2: 8.3e} | {3: 8.3e}".format(
                    self.ts, fx, fy, fz)
                cout.cout_wrap(line, 1)

    def file_output(self):
        # assemble forces matrix
        # (1 timestep) + (3+3 inertial steady+unsteady) + (3+3 body steady+unsteady)
        force_matrix = np.zeros((self.ts_max, 1 + 3 + 3 + 3 + 3))
        for self.ts in range(self.ts_max):
            i = 0
            force_matrix[self.ts, i] = self.ts
            i += 1
            force_matrix[self.ts, i:i+3] = np.sum(self.data.aero.timestep_info[self.ts].inertial_steady_forces[:, 0:3], 0)
            i += 3
            force_matrix[self.ts, i:i+3] = np.sum(self.data.aero.timestep_info[self.ts].inertial_unsteady_forces[:, 0:3], 0)
            i += 3
            force_matrix[self.ts, i:i+3] = np.sum(self.data.aero.timestep_info[self.ts].body_steady_forces[:, 0:3], 0)
            i += 3
            force_matrix[self.ts, i:i+3] = np.sum(self.data.aero.timestep_info[self.ts].body_unsteady_forces[:, 0:3], 0)

        header = ''
        header += 'tstep, '
        header += 'fx_steady_G, fy_steady_G, fz_steady_G, '
        header += 'fx_unsteady_G, fy_unsteady_G, fz_unsteady_G, '
        header += 'fx_steady_a, fy_steady_a, fz_steady_a, '
        header += 'fx_unsteady_a, fy_unsteady_a, fz_unsteady_a'

        np.savetxt(self.folder,
                   force_matrix,
                   fmt='%i' + ', %10e'*12,
                   delimiter=',',
                   header=header,
                   comments='#')
