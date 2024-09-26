import numpy as np
import os

import sharpy.utils.cout_utils as cout
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings_utils
import sharpy.utils.algebra as algebra


@solver
class AeroForcesCalculator(BaseSolver):
    """AeroForcesCalculator

    Calculates the total aerodynamic forces and moments on the body ``A`` and inertial  ``G`` frame of reference .

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

    settings_types['u_inf_dir'] = 'list(float)'
    settings_default['u_inf_dir'] = [1., 0., 0.]
    settings_description['u_inf_dir'] = 'Flow direction'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):

        self.settings = None
        self.data = None
        self.ts_max = 0

        self.folder = None
        self.caller = None

        self.table = None
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
        if kwargs.get('online', False):
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

    # generate forcing data for a given timestep
    def calculate_forces(self, ts):
        # body orientation rotation
        rot_ag = algebra.quat2rotation(self.data.structure.timestep_info[ts].quat).T  # R_AG

        # flow rotation angle from x and z components of flow direction
        # WARNING: this will give incorrect results when there is sideslip (u_inf_dir[1] != 0.)
        rot_xy = np.arctan2(-self.settings['u_inf_dir'][2], self.settings['u_inf_dir'][0])
        rmat_xy = algebra.euler2rot((0., rot_xy, 0.))

        # number of aero surfaces
        n_surf = self.data.aero.n_surf

        # forces per node per wing surface
        # naming convention: s=steady, u=unsteady, a=body for, g=inertial for
        force_s_a = [np.squeeze(rmat_xy.T @ np.expand_dims(np.moveaxis(
            self.data.aero.timestep_info[ts].forces[i], 0, -1)[..., :3], -1)) for i in range(n_surf)]
        force_u_a = [np.squeeze(rmat_xy.T @ np.expand_dims(np.moveaxis(
            self.data.aero.timestep_info[ts].dynamic_forces[i], 0, -1)[..., :3], -1)) for i in range(n_surf)]

        # add nonlifting: TODO: check FoR of nonlifting forces
        if self.settings["nonlifting_body"]:
            n_surf_nonlift = self.data.nonlifting_body.n_surf
            force_s_a.extend([np.squeeze(rmat_xy.T @ np.expand_dims(np.moveaxis(
                self.data.nonlifting_body.timestep_info[ts].inertial_steady_forces[i], 0, -1)[..., :3], -1))
                              for i in range(n_surf_nonlift)])

        # list of steady and unsteady body and inertial forces per surface
        f_s_a_surf = [np.sum(force_s_a[i], axis=(0, 1)) for i in range(n_surf)]
        f_s_g_surf = [rot_ag.T @ f_s_a_surf[i] for i in range(n_surf)]
        f_u_a_surf = [np.sum(force_u_a[i], axis=(0, 1)) for i in range(n_surf)]
        f_u_g_surf = [rot_ag.T @ f_u_a_surf[i] for i in range(n_surf)]

        # calculate moments
        # surface dimensions
        m = [surf.shape[0] for surf in force_s_a]
        n = [surf.shape[1] for surf in force_s_a]

        # aerogrid node body position
        zeta_a = [np.squeeze(rot_ag @ np.expand_dims(np.moveaxis(
            self.data.aero.timestep_info[ts].zeta[i], 0, -1), axis=-1)) for i in range(n_surf)]

        # add nonlifting body node positions
        if self.settings["nonlifting_body"]:
            zeta_a.extend([np.squeeze(rot_ag @ np.expand_dims(np.moveaxis(
                self.data.nonlifting_body.timestep_info[ts].zeta[i], 0, -1), axis=-1)) for i in range(n_surf)])

        # node body position relative to reference position and its skew
        r_a = [zeta_a[i] - np.tile(self.moment_reference_location.reshape(1, 1, 3), (m[i], n[i], 1)) for i in
               range(n_surf)]
        r_a_tilde = [np.apply_along_axis(algebra.skew, 2, r_a[i]) for i in range(n_surf)]

        # steady moments per node per surface
        mom_s_a = [np.squeeze(r_a_tilde[i] @ np.expand_dims(force_s_a[i], axis=-1)) for i in range(n_surf)]

        # list of steady body and inertial forces per surface
        mom_s_a_surf = [np.sum(mom_s_a[i], axis=(0, 1)) for i in range(n_surf)]
        mom_s_g_surf = [rot_ag.T @ mom_s_a_surf[i] for i in range(n_surf)]

        # unsteady moments per node per surface
        mom_u_a = [np.squeeze(r_a_tilde[i] @ np.expand_dims(force_u_a[i], axis=-1)) for i in range(n_surf)]

        # list of unsteady body and inertial forces per surface
        mom_u_a_surf = [np.sum(mom_u_a[i], axis=(0, 1)) for i in range(n_surf)]
        mom_u_g_surf = [rot_ag.T @ mom_u_a_surf[i] for i in range(n_surf)]

        # save steady forcing to timestep data
        self.data.aero.timestep_info[ts].body_steady_forces = np.hstack((f_s_a_surf, mom_s_a_surf))
        self.data.aero.timestep_info[ts].inertial_steady_forces = np.hstack((f_s_g_surf, mom_s_g_surf))
        self.data.aero.timestep_info[ts].total_steady_body_forces \
            = np.sum(np.hstack((f_s_a_surf, mom_s_a_surf)), axis=0)
        self.data.aero.timestep_info[ts].total_steady_inertial_forces \
            = np.sum(np.hstack((f_s_g_surf, mom_s_g_surf)), axis=0)

        # save unsteady forcing to timestep data
        self.data.aero.timestep_info[ts].body_unsteady_forces = np.hstack((f_u_a_surf, mom_u_a_surf))
        self.data.aero.timestep_info[ts].inertial_unsteady_forces = np.hstack((f_u_g_surf, mom_u_g_surf))
        self.data.aero.timestep_info[ts].total_unsteady_body_forces \
            = np.sum(np.hstack((f_u_a_surf, mom_u_a_surf)), axis=0)
        self.data.aero.timestep_info[ts].total_unsteady_inertial_forces \
            = np.sum(np.hstack((f_u_g_surf, mom_u_g_surf)), axis=0)

    def calculate_coefficients(self, fx, fy, fz, mx, my, mz):
        q_s = self.settings['q_ref'] * self.settings['S_ref']
        return (fx / q_s,
                fy / q_s,
                fz / q_s,
                mx / (q_s * self.settings['b_ref']),
                my / (q_s * self.settings['c_ref']),
                mz / (q_s * self.settings['b_ref']))

    def screen_output(self, ts):
        # print time step total aero forces
        aero_tstep = self.data.aero.timestep_info[ts]
        forces = aero_tstep.total_steady_inertial_forces[:3] + aero_tstep.total_unsteady_inertial_forces[:3]
        moments = aero_tstep.total_steady_inertial_forces[3:] + aero_tstep.total_unsteady_inertial_forces[3:]

        if self.settings['coefficients']:  # TODO: Check if coefficients have to be computed differently for fuselages
            coeffs = self.calculate_coefficients(*forces, *moments)
            self.table.print_line([ts, *coeffs])
        else:
            self.table.print_line([ts, *forces, *moments])

    def file_output(self, filename):
        # assemble forces/moments matrix: (1 timestep) + (3+3 inertial steady+unsteady) + (3+3 body steady+unsteady)
        force_matrix = np.zeros((self.ts_max, 13))
        moment_matrix = np.zeros((self.ts_max, 13))
        for ts in range(self.ts_max):
            aero_tstep = self.data.aero.timestep_info[ts]
            # timestep column
            force_matrix[ts, 0] = moment_matrix[ts, 0] = ts
            # Steady forces/moments G
            force_matrix[ts, 1:4], moment_matrix[ts, 1:4] = np.split(aero_tstep.total_steady_inertial_forces, 2)
            # Unsteady forces/moments G
            force_matrix[ts, 4:7], moment_matrix[ts, 4:7] = np.split(aero_tstep.total_unsteady_inertial_forces, 2)
            # Steady forces/moments A
            force_matrix[ts, 7:10], moment_matrix[ts, 7:10] = np.split(aero_tstep.total_steady_body_forces, 2)
            # Unsteady forces/moments A
            force_matrix[ts, 10:13], moment_matrix[ts, 10:13] = np.split(aero_tstep.total_unsteady_body_forces, 2)

        # save linear force data
        header_f = ('tstep, fx_steady_G, fy_steady_G, fz_steady_G, fx_unsteady_G, fy_unsteady_G, fz_unsteady_G, '
                    'fx_steady_a, fy_steady_a, fz_steady_a, fx_unsteady_a, fy_unsteady_a, fz_unsteady_a')

        np.savetxt(self.folder + 'forces_' + filename,
                   force_matrix,
                   fmt='%i' + ', %10e' * (np.shape(force_matrix)[1] - 1),
                   delimiter=',',
                   header=header_f,
                   comments='#')

        # save moment data
        header_m = ('tstep, mx_steady_G, my_steady_G, mz_steady_G, mx_unsteady_G, my_unsteady_G, mz_unsteady_G, '
                    'mx_steady_a, my_steady_a, mz_steady_a, mx_unsteady_a, rot_agmy_unsteady_a, mz_unsteady_a')

        np.savetxt(self.folder + 'moments_' + filename,
                   moment_matrix,
                   fmt='%i' + ', %10e' * (np.shape(moment_matrix)[1] - 1),
                   delimiter=',',
                   header=header_m,
                   comments='#')
