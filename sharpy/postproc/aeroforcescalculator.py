import numpy as np
import os

import sharpy.utils.cout_utils as cout
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings_utils
import sharpy.utils.algebra as algebra
import sharpy.aero.utils.mapping as mapping
import warnings


@solver
class AeroForcesCalculator(BaseSolver):
    """AeroForcesCalculator

    Calculates the total aerodynamic forces and moments on the frame of reference ``A``.

    """
    solver_id = 'AeroForcesCalculator'
    solver_classification = 'post-processor'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['write_text_file'] = 'bool'
    settings_default['write_text_file'] = False
    settings_description['write_text_file'] = 'Write ``txt`` file with results'

    settings_types['text_file_name'] = 'str'
    settings_default['text_file_name'] = 'aeroforces.txt'
    settings_description['text_file_name'] = 'Text file name'

    settings_types['screen_output'] = 'bool'
    settings_default['screen_output'] = True
    settings_description['screen_output'] = 'Show results on screen'

    settings_default['coefficients'] = False
    settings_types['coefficients'] = 'bool'
    settings_description['coefficients'] = 'Calculate aerodynamic coefficients'

    settings_default['lifting_surfaces'] = True
    settings_types['lifting_surfaces'] = 'bool'
    settings_description['lifting_surfaces'] = 'Includes aerodynamic forces from lifting surfaces'

    settings_default['nonlifting_body'] = False
    settings_types['nonlifting_body'] = 'bool'
    settings_description['nonlifting_body'] = 'Includes aerodynamic forces from nonlifting bodies'

    settings_types['q_ref'] = 'float'
    settings_default['q_ref'] = 1
    settings_description['q_ref'] = 'Reference dynamic pressure'

    settings_types['S_ref'] = 'float'
    settings_default['S_ref'] = 1
    settings_description['S_ref'] = 'Reference area'

    settings_types['b_ref'] = 'float'
    settings_default['b_ref'] = 1
    settings_description['b_ref'] = 'Reference span'

    settings_types['c_ref'] = 'float'
    settings_default['c_ref'] = 1
    settings_description['c_ref'] = 'Reference chord'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):

        self.settings = None
        self.data = None
        self.ts_max = 0

        self.folder = None
        self.caller = None

        self.table = None
        self.rot = None
        self.moment_reference_location = np.array([0., 0., 0.])

    def initialise(self, data, custom_settings=None, caller=None, restart=False):
        self.data = data
        self.settings = data.settings[self.solver_id]
        self.ts_max = len(self.data.structure.timestep_info)
        settings_utils.to_custom_types(self.settings, self.settings_types, self.settings_default)
        self.caller = caller

        self.folder = data.output_folder + '/forces/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        if self.settings['screen_output']:
            if self.settings['coefficients']:
                self.table = cout.TablePrinter(7, field_length=12, field_types=['g'] + 6 * ['f'])
                self.table.print_header(['tstep', 'Cfx_g', 'Cfy_g', 'Cfz_g', 'Cmx_g', 'Cmy_g', 'Cmz_g'])
            else:
                self.table = cout.TablePrinter(7, field_length=12, field_types=['g'] + 6 * ['e'])
                self.table.print_header(['tstep', 'fx_g', 'fy_g', 'fz_g', 'mx_g', 'my_g', 'mz_g'])

    def run(self, **kwargs):

        online = settings_utils.set_value_or_default(kwargs, 'online', False)

        if online:
            self.ts_max = len(self.data.structure.timestep_info)
            self.calculate_forces(-1)

            if self.settings['screen_output']:
                self.screen_output(-1)
        else:
            for ts in range(self.ts_max):
                self.calculate_forces(ts)
                if self.settings['screen_output']:
                    self.screen_output(ts)
            cout.cout_wrap('...Finished', 1)

        if self.settings['write_text_file']:
            self.file_output(self.settings['text_file_name'])
        return self.data
 
    def calculate_forces(self, ts):
        # Forces per surface in G frame
        self.rot = algebra.quat2rotation(self.data.structure.timestep_info[ts].quat)
        force = self.data.aero.timestep_info[ts].forces
        unsteady_force = self.data.aero.timestep_info[ts].dynamic_forces
        for i_surf in range(self.data.aero.n_surf):
            (
            self.data.aero.timestep_info[ts].inertial_steady_forces[i_surf, 0:3], 
            self.data.aero.timestep_info[ts].inertial_unsteady_forces[i_surf, 0:3],
            self.data.aero.timestep_info[ts].body_steady_forces[i_surf, 0:3],
            self.data.aero.timestep_info[ts].body_unsteady_forces[i_surf, 0:3]
            ) = self.calculate_forces_for_isurf_in_g_frame(force[i_surf], unsteady_force=unsteady_force[i_surf])
        
        if self.settings["nonlifting_body"]:
            for i_surf in range(self.data.nonlifting_body.n_surf):
                (
                self.data.nonlifting_body.timestep_info[ts].inertial_steady_forces[i_surf, 0:3], 
                self.data.nonlifting_body.timestep_info[ts].body_steady_forces[i_surf, 0:3],
                ) = self.calculate_forces_for_isurf_in_g_frame(self.data.nonlifting_body.timestep_info[ts].forces[i_surf], nonlifting=True)
        
        # Convert to forces in B frame
        try:
            steady_forces_b = self.data.structure.timestep_info[ts].postproc_node['aero_steady_forces']
        except KeyError:
            if self.settings["nonlifting_body"]:
                warnings.warn('Nonlifting forces are not considered in aero forces calculation since forces cannot not be retrieved from postproc node.')
            steady_forces_b = self.map_forces_beam_dof(self.data.aero, ts, force)

        try:
            unsteady_forces_b = self.data.structure.timestep_info[ts].postproc_node['aero_unsteady_forces']
        except KeyError:
            unsteady_forces_b = self.map_forces_beam_dof(self.data.aero, ts, unsteady_force)
        # Convert to forces in A frame
        steady_forces_a = self.data.structure.timestep_info[ts].nodal_b_for_2_a_for(steady_forces_b,
                                                                                    self.data.structure)

        unsteady_forces_a = self.data.structure.timestep_info[ts].nodal_b_for_2_a_for(unsteady_forces_b,
                                                                                      self.data.structure)

        # Express total forces in A frame
        self.data.aero.timestep_info[ts].total_steady_body_forces = \
            mapping.total_forces_moments(steady_forces_a,
                                         self.data.structure.timestep_info[ts].pos,
                                         ref_pos=self.moment_reference_location)
        self.data.aero.timestep_info[ts].total_unsteady_body_forces = \
            mapping.total_forces_moments(unsteady_forces_a,
                                         self.data.structure.timestep_info[ts].pos,
                                         ref_pos=self.moment_reference_location)

        # Express total forces in G frame
        self.data.aero.timestep_info[ts].total_steady_inertial_forces = \
            np.block([[self.rot, np.zeros((3, 3))],
                      [np.zeros((3, 3)), self.rot]]).dot(
                self.data.aero.timestep_info[ts].total_steady_body_forces)

        self.data.aero.timestep_info[ts].total_unsteady_inertial_forces = \
            np.block([[self.rot, np.zeros((3, 3))],
                      [np.zeros((3, 3)), self.rot]]).dot(
                self.data.aero.timestep_info[ts].total_unsteady_body_forces)

    def calculate_forces_for_isurf_in_g_frame(self, force, unsteady_force = None, nonlifting = False):
        """
            Forces for a surface in G frame
        """
        # Forces per surface in G frame
        total_steady_force = np.zeros((3,))
        total_unsteady_force = np.zeros((3,))
        _, n_rows, n_cols = force.shape
        for i_m in range(n_rows):
            for i_n in range(n_cols):
                total_steady_force += force[0:3, i_m, i_n]
                if not nonlifting:
                    total_unsteady_force += unsteady_force[0:3, i_m, i_n]
        if not nonlifting:
            return total_steady_force, total_unsteady_force, np.dot(self.rot.T, total_steady_force), np.dot(self.rot.T, total_unsteady_force)
        else:
            return total_steady_force, np.dot(self.rot.T, total_steady_force)


    def map_forces_beam_dof(self, aero_data, ts, force):
        struct_tstep = self.data.structure.timestep_info[ts]
        aero_forces_beam_dof = mapping.aero2struct_force_mapping(force,
                                                                 aero_data.struct2aero_mapping,
                                                                 aero_data.timestep_info[ts].zeta,
                                                                 struct_tstep.pos,
                                                                 struct_tstep.psi,
                                                                 None,
                                                                 self.data.structure.connectivities,
                                                                 struct_tstep.cag())
        return aero_forces_beam_dof

    def calculate_coefficients(self, fx, fy, fz, mx, my, mz):
        qS = self.settings['q_ref'] * self.settings['S_ref']
        return fx/qS, fy/qS, fz/qS, mx/qS/self.settings['b_ref'], my/qS/self.settings['c_ref'], \
               mz/qS/self.settings['b_ref']

    def screen_output(self, ts):
        # print time step total aero forces
        forces  = np.zeros(3)
        moments  = np.zeros(3)

        aero_tstep = self.data.aero.timestep_info[ts]
        forces += aero_tstep.total_steady_inertial_forces[:3] + aero_tstep.total_unsteady_inertial_forces[:3]
        moments += aero_tstep.total_steady_inertial_forces[3:] + aero_tstep.total_unsteady_inertial_forces[3:]

        if self.settings['coefficients']: # TODO: Check if coefficients have to be computed differently for fuselages
            Cfx, Cfy, Cfz, Cmx, Cmy, Cmz = self.calculate_coefficients(*forces, *moments)
            self.table.print_line([ts, Cfx, Cfy, Cfz, Cmx, Cmy, Cmz])
        else:
            self.table.print_line([ts, *forces, *moments])

    def file_output(self, filename):
        # assemble forces/moments matrix
        # (1 timestep) + (3+3 inertial steady+unsteady) + (3+3 body steady+unsteady)
        force_matrix = np.zeros((self.ts_max, 1 + 3 + 3 + 3 + 3 ))
        moment_matrix = np.zeros((self.ts_max, 1 + 3 + 3 + 3 + 3))
        for ts in range(self.ts_max):
            aero_tstep = self.data.aero.timestep_info[ts]
            i = 0
            force_matrix[ts, i] = ts
            moment_matrix[ts, i] = ts
            i += 1

            # Steady forces/moments G
            force_matrix[ts, i:i+3] = aero_tstep.total_steady_inertial_forces[:3]
            moment_matrix[ts, i:i+3] = aero_tstep.total_steady_inertial_forces[3:]
            i += 3

            # Unsteady forces/moments G
            force_matrix[ts, i:i+3] = aero_tstep.total_unsteady_inertial_forces[:3]
            moment_matrix[ts, i:i+3] = aero_tstep.total_unsteady_inertial_forces[3:]
            i += 3

            # Steady forces/moments A
            force_matrix[ts, i:i+3] = aero_tstep.total_steady_body_forces[:3]
            moment_matrix[ts, i:i+3] = aero_tstep.total_steady_body_forces[3:]
            i += 3

            # Unsteady forces/moments A
            force_matrix[ts, i:i+3] = aero_tstep.total_unsteady_body_forces[:3]
            moment_matrix[ts, i:i+3] = aero_tstep.total_unsteady_body_forces[3:]


        header = ''
        header += 'tstep, '
        header += 'fx_steady_G, fy_steady_G, fz_steady_G, '
        header += 'fx_unsteady_G, fy_unsteady_G, fz_unsteady_G, '
        header += 'fx_steady_a, fy_steady_a, fz_steady_a, '
        header += 'fx_unsteady_a, fy_unsteady_a, fz_unsteady_a'

        np.savetxt(self.folder + 'forces_' + filename,
                   force_matrix,
                   fmt='%i' + ', %10e' * (np.shape(force_matrix)[1] - 1),
                   delimiter=',',
                   header=header,
                   comments='#')

        header = ''
        header += 'tstep, '
        header += 'mx_steady_G, my_steady_G, mz_steady_G, '
        header += 'mx_unsteady_G, my_unsteady_G, mz_unsteady_G, '
        header += 'mx_steady_a, my_steady_a, mz_steady_a, '
        header += 'mx_unsteady_a, my_unsteady_a, mz_unsteady_a'

        np.savetxt(self.folder + 'moments_' + filename,
                   moment_matrix,
                   fmt='%i' + ', %10e' * (np.shape(moment_matrix)[1] - 1),
                   delimiter=',',
                   header=header,
                   comments='#')
