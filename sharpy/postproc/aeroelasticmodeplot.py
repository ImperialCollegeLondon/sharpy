import os
import numpy as np
from itertools import chain
from tvtk.api import tvtk, write_data

from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings_utils
from sharpy.utils.algebra import crv2rotation, quat2rotation
from sharpy.utils.cout_utils import cout_wrap


@solver
class AeroelasticModal(BaseSolver):
    """
    This class performs modal analysis on a linearised aeroelastic system. The modes are then plotted in phase steps,
    and the stability eigenvalues printed to the console.

    Note: this supports model order reduction through the use of a Krylov projection. For this postprocessor to work,
    the linearised model must use a modal projection of the structure.
    """

    solver_id = 'AeroelasticModal'
    solver_classification = 'Linear'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['num_modes'] = 'int'
    settings_default['num_modes'] = 5
    settings_description['num_modes'] = 'Number of modes to retain'

    settings_types['use_custom_timestep'] = 'int'
    settings_default['use_custom_timestep'] = -1
    settings_description['use_custom_timestep'] = 'Timestep of solution to use for analysis'

    settings_types['step_count'] = 'int'
    settings_default['step_count'] = 30
    settings_description['step_count'] = 'Number of phase steps to take along the mode shape'

    settings_types['remove_conjugate'] = 'bool'
    settings_default['remove_conjugate'] = True
    settings_description['remove_conjugate'] = 'Remove conjugate modes'

    settings_types['max_rotation_deg'] = 'float'
    settings_default['max_rotation_deg'] = 25.0
    settings_description['max_rotation_deg'] = 'Scale mode shape to have specified maximum rotation'

    settings_types['max_displacement'] = 'float'
    settings_default['max_displacement'] = 0.05
    settings_description['max_displacement'] = 'Scale mode shape to have specified maximum displacement'

    settings_types['max_gamma'] = 'float'
    settings_default['max_gamma'] = 1.0
    settings_description['max_gamma'] = 'Scale mode shape to have specified maximum circulation'

    settings_types['save_data'] = 'bool'
    settings_default['save_data'] = True
    settings_description['save_data'] = 'Write mode shapes, frequencies and damping to file'

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
                                       self.settings_default)

    def run(self, **kwargs):
        if not hasattr(self.data, 'linear'):
            raise AttributeError('Linear data not found')

        # extract the A (system) matrix from the linearised aeroelastic system
        a_mat = self.data.linear.ss.A

        # obtain vectors for the indices for the states
        state_index = {}
        for var in self.data.linear.ss._state_variables.vector_variables:
            if var.name in ('gamma', 'gamma_w', 'q', 'krylov'):
                state_index[var.name] = var.cols_loc

        # perform the eigenvalue decomposition of the system matrix
        evals_d_ae, evecs_ae = np.linalg.eig(a_mat)

        # convert eigenvalues from discrete to continuous time
        evals_c_ae = np.log(evals_d_ae) / self.data.linear.ss.dt

        # remove conjugate eigenvalues by making the imaginary part of the eigenvalues positive and adding to a set
        if self.settings['remove_conjugate']:
            # dictionary contains {rounded eigenvalue: eigenvalue index} pairs
            eig_set = dict()

            for eig_index, eig in enumerate(evals_c_ae):
                # we here round the eigenvalues to 5 decimal places to prevent a pair not matching to machine precision
                eig_rounded = np.round(eig.real + 1j * np.abs(eig.imag), 5)
                if eig_rounded not in eig_set.keys():
                    eig_set.update({eig_rounded: eig_index})

            first_eig_index = np.fromiter(eig_set.values(), dtype=int)
            order = first_eig_index[np.argsort(-evals_c_ae.real[first_eig_index])][:self.settings['num_modes']]
        else:
            order = np.argsort(-evals_c_ae.real)[:self.settings['num_modes']]

        # order and truncate set of aeroelastic modes and eigenvalues
        evals_c_ae_trunc = evals_c_ae[order]
        evecs_ae_trunc = evecs_ae[:, order]

        # print eigenvalues to console
        cout_wrap('Aeroelastic eigenvalues:', 0)
        for i, eig in enumerate(evals_c_ae_trunc):
            cout_wrap(f'{i}: {eig:.3f}', 1)

        # return if no data is to be saved
        if not self.settings['save_data']:
            return self.data

        # vector of complex phase values, evenly spaced around a unit circle
        phase = np.exp(1j * np.linspace(0, 2 * np.pi, self.settings['step_count'], endpoint=False))

        # mode shapes of the structure, as a projection of the free modes
        num_node = self.data.structure.num_node
        num_modes_struct = \
            self.data.structure.timestep_info[self.settings['use_custom_timestep']].modal['eigenvectors'].shape[1]

        # add zeros to account for the clamped nodes which are not included in the structural eigenvectors
        struct_evects = np.zeros((6 * num_node, num_modes_struct))
        struct_evects[6:, :] = self.data.structure.timestep_info[self.settings['use_custom_timestep']].modal[
            'eigenvectors']
        node_disp_base = struct_evects @ evecs_ae_trunc[state_index['q'], :]

        # mode shapes of the structural nodes, bound gamma and wake gamma
        node_disp = np.einsum('ij,k->ijk', node_disp_base, phase).real  # [num_dof, num_lambda, num_phase]

        if 'gamma' in state_index and 'gamma_w' in state_index:
            gamma = np.vstack((np.einsum('ij,k->ijk', evecs_ae_trunc[state_index['gamma'], :], phase),
                               np.einsum('ij,k->ijk', evecs_ae_trunc[state_index['gamma_w'], :], phase))).real
        elif 'krylov' in state_index:
            # project the krylov states onto the full gamma states
            v = self.data.linear.linear_system.uvlm.ss.v.value

            full_state_index = {}
            for var in self.data.linear.linear_system.uvlm.ss.ss_full.state_variables.vector_variables:
                if var.name in ('gamma', 'gamma_w'):
                    full_state_index[var.name] = var.cols_loc

            aero_evects = v @ evecs_ae_trunc[state_index['krylov'], :]

            gamma = np.vstack((np.einsum('ij,k->ijk', aero_evects[full_state_index['gamma'], :], phase),
                               np.einsum('ij,k->ijk', aero_evects[full_state_index['gamma_w'], :], phase))).real
            pass
        else:
            raise KeyError("No aerodynamic states found in linear model")

        # split structure into displacements and rotations
        num_dof = node_disp.shape[0]
        num_node = int(num_dof // 6)

        i_pos = np.array(list(chain([6 * i, 6 * i + 1, 6 * i + 2] for i in range(num_node))))
        i_rot = np.array(list(chain([6 * i + 3, 6 * i + 4, 6 * i + 5] for i in range(num_node))))

        node_pos = node_disp[i_pos, :, :]
        node_rot = node_disp[i_rot, :, :]

        # scale mode shapes
        max_pos = np.max(np.linalg.norm(node_pos, axis=1), axis=(0, 2))
        node_pos_scaled = np.einsum('ijkl,k->ijkl', node_pos, 1.0 / max_pos) * self.settings['max_displacement']

        max_rot = np.max(np.linalg.norm(node_rot, axis=1), axis=(0, 2))
        node_rot_scaled = np.einsum('ijkl,k->ijkl', node_rot, 1.0 / max_rot) * np.deg2rad(
            self.settings['max_rotation_deg'])

        gamma_scale_fact = 1.0 / np.max(np.abs(gamma), axis=(0, 2))
        gamma_scaled = np.einsum('ijk,j->ijk', gamma, gamma_scale_fact) * self.settings['max_gamma']

        # add structural deflections to linearisation point
        base_pos = self.data.structure.timestep_info[self.settings['use_custom_timestep']].pos

        # split to give node positions for each surface
        base_pos_surf = [base_pos[index, :] for index in self.data.aero.aero2struct_mapping]

        # create psi vector for nodes in order
        psi0 = np.expand_dims(self.data.structure.timestep_info[self.settings['use_custom_timestep']].psi[0, 0, :], 0)
        psi1 = self.data.structure.timestep_info[self.settings['use_custom_timestep']].psi[:, 1:, :].reshape(-1, 3)
        base_rot = np.vstack((psi0, psi1))

        # node position in modes, combining deformation at linearisation point with mode shapes
        combined_beam_pos = (np.broadcast_to(np.expand_dims(base_pos, axis=(2, 3)), shape=node_pos_scaled.shape)
                             + node_pos_scaled)

        # node rotation matrices in modes, combining deformation at linearisation point with mode shapes
        rmat_node_rot = np.apply_along_axis(crv2rotation, 1, node_rot_scaled)
        rmat_base_rot = np.apply_along_axis(crv2rotation, 1, base_rot)
        combined_rmat = np.einsum('ijk,iklmn->ijlmn', rmat_base_rot, rmat_node_rot)

        # initial aerodynamic grid and wake shapein the inertial frame
        zeta_base_g = self.data.aero.timestep_info[self.settings['use_custom_timestep']].zeta
        zeta_w = self.data.aero.timestep_info[self.settings['use_custom_timestep']].zeta_star

        # rotation matrix to A frame
        orient_rmat = quat2rotation(self.data.structure.timestep_info[self.settings['use_custom_timestep']].quat)

        # split base positions and rotations to per surface
        rmat_base_rot_surf = [rmat_base_rot[index, ...] for index in self.data.aero.aero2struct_mapping]
        combined_rmat_surf = [combined_rmat[index, ...] for index in self.data.aero.aero2struct_mapping]
        combined_beam_pos_surf = [combined_beam_pos[index, ...] for index in self.data.aero.aero2struct_mapping]

        zeta = []
        zeta_delta = []
        is_bound = []
        surf_number = []

        # loop through each surface
        for i_surf, zeta_base_g in enumerate(zeta_base_g):
            # rotate aero grid to body frame
            zeta_base_a = np.einsum('ij,jkl->ikl', orient_rmat.T, zeta_base_g)

            # rotate aero grid to material frame
            zeta_base_b = np.einsum('ijk,kli->jli', rmat_base_rot_surf[i_surf], zeta_base_a
                                    - np.broadcast_to(np.expand_dims(base_pos_surf[i_surf].T, 1), zeta_base_a.shape))

            # rotate to deformed state [n+1, m+1, 3, num_lambda, num_phase]
            zeta_a = np.einsum('ijklm,jni->inklm', combined_rmat_surf[i_surf], zeta_base_b) + np.expand_dims(
                combined_beam_pos_surf[i_surf], 1)

            # rotate back to inertial frame
            zeta.append(np.einsum('ij,kljmn->klimn', orient_rmat, zeta_a))

            # calculate displacement of each node relative to reference
            zeta_delta.append(np.linalg.norm(zeta[-1] - np.expand_dims(np.transpose(zeta_base_g, (2, 1, 0)),
                                                                       axis=(3, 4)), axis=2))

            is_bound.append(True)
            surf_number.append(i_surf)

        # add wake data
        for i_wake, wake in enumerate(zeta_w):
            zeta.append(np.tile(np.expand_dims(np.transpose(wake, (2, 1, 0)), axis=(3, 4)),
                                (1, 1, 1, self.settings['num_modes'], self.settings['step_count'])))
            zeta_delta.append(np.zeros_like(zeta[-1][:, :, 0, ...]))
            is_bound.append(False)
            surf_number.append(i_wake)

        # create folder to save modes to
        folder = self.data.output_folder + '/aeroelastic_modes/'
        if not os.path.exists(folder):
            os.makedirs(folder)

        # write data to file
        # this has a loop per mode, phase step and surface. Wake panels are here classed as surfaces.
        for i_mode in range(self.settings['num_modes']):
            for i_phase in range(self.settings['step_count']):
                i_state = 0
                for i_surf in range(len(zeta)):
                    if is_bound[i_surf]:
                        filename = folder + f"mode{i_mode:02d}_body_surf{surf_number[i_surf]}_phase{i_phase}.vtu"
                    else:
                        filename = folder + f"mode{i_mode:02d}_wake_surf{surf_number[i_surf]}_phase{i_phase}.vtu"

                    n, m = zeta[i_surf].shape[:2]

                    panel_data_dim = (m - 1) * (n - 1)

                    coords = zeta[i_surf][:, :, :, i_mode, i_phase].reshape(-1, 3)
                    panel_id = np.arange(panel_data_dim)
                    panel_surf_id = np.full_like(panel_id, i_surf)

                    # create node connectivity to create panels
                    base_range = np.arange(m - 1)
                    base_conn = np.concatenate([base_range + i * m for i in range(n - 1)])
                    conn = np.stack([base_conn, base_conn + 1, base_conn + m + 1, base_conn + m]).T

                    ug = tvtk.UnstructuredGrid(points=coords)
                    ug.set_cells(tvtk.Quad().cell_type, conn)

                    ug.cell_data.scalars = panel_id
                    ug.cell_data.scalars.name = 'panel_n_id'

                    ug.cell_data.add_array(
                        gamma_scaled[i_state:i_state + panel_data_dim, i_mode, i_phase].reshape(m - 1, -1).T.flatten())
                    ug.cell_data.get_array(1).name = 'gamma'

                    ug.cell_data.add_array(panel_surf_id)
                    ug.cell_data.get_array(2).name = 'panel_surface_id'

                    ug.point_data.scalars = np.arange(coords.shape[0])
                    ug.point_data.scalars.name = 'n_id'

                    ug.point_data.add_array(zeta_delta[i_surf][:, :, i_mode, i_phase].ravel())
                    ug.point_data.get_array(1).name = 'point_displacement_magnitude'

                    write_data(ug, filename)

                    i_state += panel_data_dim

        return self.data
