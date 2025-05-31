import os
import warnings
import numpy as np
import scipy.linalg as sclalg
import sharpy.utils.settings as settings_utils
from sharpy.utils.solver_interface import solver, BaseSolver, initialise_solver
import sharpy.utils.cout_utils as cout
import sharpy.utils.algebra as algebra
import sharpy.solvers.lindynamicsim as lindynamicsim
import sharpy.structure.utils.modalutils as modalutils
import sharpy.utils.frequencyutils as frequencyutils
import sharpy.linear.src.libss as libss
import h5py


@solver
class AsymptoticStability(BaseSolver):
    """
    Calculates the asymptotic stability properties of the linearised system by computing
    the corresponding eigenvalues and eigenvectors.

    The output of this solver is written to the ``stability`` directory in the case output.

    The stability of the systems specified in ``target_systems`` is performed. If the system has been previously
    scaled, a ``reference_velocity`` should be provided to compute the stability at such point.

    The eigenvalues can be truncated, keeping a minimum ``num_evals`` (sorted by decreasing real part) or by limiting
    the higher frequency modes through ``frequency_cutoff``.

    Results can be saved to file using ``export_eigenvalues``. The setting ``display_root_locus`` shows a simple
    Argand diagram where the continuous time eigenvalues are displayed.

    The eigenvectors can be displayed (in .vtu format for use in Paraview) with the ``modes_to_plot`` setting, whereby
    the user specifies the mode indices that are to be plotted (sorted with in the same way as the eigenvalue table,
    i.e. by decreasing real part). A snapshot of the eigenvector is produced for the 0, 45, 90 and 135 degree phases.
    This feature currently supports the flexible structural modes and does not show the rigid body contribution.
    The output is written to the ``stability/modes`` folder and includes the structure at the
    reference linearisation state.
    """
    solver_id = 'AsymptoticStability'
    solver_classification = 'post-processor'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    settings_types['print_info'] = 'bool'
    settings_default['print_info'] = False
    settings_description['print_info'] = 'Print information and table of eigenvalues'

    settings_types['reference_velocity'] = 'float'
    settings_default['reference_velocity'] = 1.
    settings_description['reference_velocity'] = 'Reference velocity at which to compute eigenvalues for scaled systems'

    settings_types['frequency_cutoff'] = 'float'
    settings_default['frequency_cutoff'] = 0
    settings_description['frequency_cutoff'] = 'Truncate higher frequency modes. If zero none are truncated'

    settings_types['export_eigenvalues'] = 'bool'
    settings_default['export_eigenvalues'] = False
    settings_description['export_eigenvalues'] = 'Save eigenvalues and eigenvectors to file.'

    settings_types['output_file_format'] = 'str'
    settings_default['output_file_format'] = 'dat'
    settings_description['output_file_format'] = 'Eigenvalue/eigenvector output file format. HDF5 or text (.dat) files.'
    settings_options['output_file_format'] = ['h5', 'dat']

    settings_types['display_root_locus'] = 'bool'
    settings_default['display_root_locus'] = False
    settings_description['display_root_locus'] = 'Show plot with eigenvalues on Argand diagram'

    settings_types['velocity_analysis'] = 'list(float)'
    settings_default['velocity_analysis'] = []
    settings_description['velocity_analysis'] = 'List containing min, max and number ' \
                                                'of velocities to analyse the system'

    settings_types['target_system'] = 'list(str)'
    settings_default['target_system'] = ['aeroelastic']
    settings_description['target_system'] = 'System or systems for which to find frequency response.'
    settings_options['target_system'] = ['aeroelastic', 'aerodynamic', 'structural']

    settings_types['num_evals'] = 'int'
    settings_default['num_evals'] = 200
    settings_description['num_evals'] = 'Number of eigenvalues to retain.'

    settings_types['modes_to_plot'] = 'list(int)'
    settings_default['modes_to_plot'] = []
    settings_description['modes_to_plot'] = 'List of mode numbers to plot. Plots the 0, 45, 90 and 135' \
                                            'degree phases.'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.settings = None
        self.data = None
        self.folder = None
        self.print_info = False

        self.frequency_cutoff = np.inf
        self.num_evals = None

        self.postprocessors = dict()
        self.with_postprocessors = False
        self.caller = None

    def initialise(self, data, custom_settings=None, caller=None, restart=False):
        self.data = data

        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings

        settings_utils.to_custom_types(self.settings, self.settings_types, self.settings_default, 
                           options=self.settings_options, no_ctype=True)

        self.num_evals = self.settings['num_evals']

        self.folder = data.output_folder + '/stability/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        if self.settings['print_info']:
            self.print_info = True
        else:
            self.print_info = False

        try:
            self.frequency_cutoff = self.settings['frequency_cutoff']
        except AttributeError:
            self.frequency_cutoff = float(self.settings['frequency_cutoff'])

        if self.frequency_cutoff == 0:
            self.frequency_cutoff = np.inf

        self.caller = caller

    def run(self, **kwargs):
        """
        Computes the eigenvalues and eigenvectors

        Returns:
            eigenvalues (np.ndarray): Eigenvalues sorted and frequency truncated
            eigenvectors (np.ndarray): Corresponding mode shapes

        """
        online = settings_utils.set_value_or_default(kwargs, 'online', False)

        # if the system is scaled, only one system can be analysed
        if self.settings['reference_velocity'] != 1. and self.data.linear.linear_system.uvlm.scaled:
            ss_list = [self.data.linear.linear_system.update(self.settings['reference_velocity'])]
            not_scaled = False
            system_name_list = ['aeroelastic']
            if len(self.settings['target_system']) > 1:
                cout.cout_wrap('Warning: the system is scaled thus the only analysis currently supported is'
                               ' for the aeroelastic system', 3)
        else:
            ss_list = [frequencyutils.find_target_system(self.data, system_name) for system_name in
                       self.settings['target_system']]
            not_scaled = True
            system_name_list = self.settings['target_system']

        self.compute_eigenvalues(ss_list, system_name_list, not_scaled)

        return self.data

    def compute_eigenvalues(self, ss, system_name_list=None, not_scaled=True):
        """
        Computes the eigenvalues and eigenvectors of the state-space

        Args:
            ss (libss.StateSpace or list([libss.StateSpace]): State-space or list of state-spaces
            system_name_list (list([str]): Names of systems in the case multiple systems are required
            not_scaled (bool): Flag to indicate whether the systems are assembled in non-dimensional time
        """

        if type(ss) is libss.StateSpace:
            ss_list = [ss]
            if system_name_list is None:
                system_name_list = ['']
        elif type(ss) is list:
            ss_list = ss

            if system_name_list is None:
                system_name_list = []
                for sys, sys_number in enumerate(ss_list):
                    system_name_list.append(f'system{sys_number:g}')
                    if type(sys) is not libss.StateSpace:
                        raise TypeError(f'State-space {sys_number} is not type libss.StateSpace')
        else:
            raise TypeError('ss input must be either a libss.StateSpace instance or a list[libss.StateSpace]')

        for ith, system in enumerate(ss_list):
            system_name = system_name_list[ith]

            if self.print_info:
                cout.cout_wrap('Calculating %s eigenvalues using direct method' % system_name)

            eigenvalues, eigenvectors = sclalg.eig(system.A)

            # Convert DT eigenvalues into CT
            if system.dt:
                eigenvalues = self.convert_to_continuoustime(system.dt, eigenvalues, not_scaled)

            num_evals = min(self.num_evals, len(eigenvalues))

            eigenvalues, eigenvectors = self.sort_eigenvalues(eigenvalues, eigenvectors, self.frequency_cutoff,
                                                              number_of_eigenvalues=num_evals)

            if self.settings['export_eigenvalues']:
                self.export_eigenvalues(num_evals, eigenvalues, eigenvectors, filename=system_name)

            if self.settings['print_info']:
                cout.cout_wrap(f'Dynamical System Eigenvalues - {system_name} system')
                if system_name != '':
                    eig_table_filename = system_name + '_'
                else:
                    eig_table_filename = ''
                eigenvalue_description_file = self.folder + '/{:s}eigenvaluetable.txt'.format(eig_table_filename)
                eigenvalue_table = modalutils.EigenvalueTable(filename=eigenvalue_description_file)
                eigenvalue_table.print_header(eigenvalue_table.headers)
                eigenvalue_table.print_evals(eigenvalues[:self.num_evals])
                eigenvalue_table.close_file()

            if self.settings['display_root_locus']:
                self.display_root_locus(eigenvalues)

            if len(self.settings['velocity_analysis']) == 3 and system_name == 'aeroelastic':
                assert self.data.linear.linear_system.uvlm.scaled, \
                    'The UVLM system is unscaled, unable to rescale the structural equations only. Rerun with a ' \
                    'normalised UVLM system.'
                self.velocity_analysis()

            # Under development
            if len(self.settings['modes_to_plot']) != 0 and system_name == 'aeroelastic':
                self.plot_modes(eigenvectors)

        return eigenvalues, eigenvectors

    def convert_to_continuoustime(self, dt, discrete_time_eigenvalues, not_scaled=False):
        r"""
        Convert eigenvalues to discrete time. The ``not_scaled`` argument can be used to bypass the search from
        within SHARPy of scaling factors. For instance, when the state-space of choice is not part of a standard
        SHARPy case but rather an interpolated ROM etc.

        The eigenvalues are converted to continuous time using

        .. math:: \lambda_{ct} = \frac{\log (\lambda_{dt})}{\Delta t}

        If the system is scaled, the dimensional time step is retrieved as

        .. math:: \Delta t_{dim} = \bar{\Delta t} \frac{l_{ref}}{U_{\infty, actual}}

        where :math:`l_{ref}` is the reference length and :math:`U_{\infty, actual}` is the free stream velocity at
        which to calculate the eigenvalues.

        Args:
            dt (float): Discrete time increment.
            discrete_time_eigenvalues (np.ndarray): Array of discrete time eigenvalues.
            not_scaled (bool): Treat the system as not scaled. No Scaling Factors will be searched in SHARPy.
        """
        if not_scaled:
            dt = dt
        else:
            try:
                ScalingFacts = self.data.linear.linear_system.uvlm.sys.ScalingFacts
                if ScalingFacts['length'] != 1.0 and ScalingFacts['time'] != 1.0:
                    dt *= ScalingFacts['length'] / self.settings['reference_velocity']
                else:
                    dt = dt
            except AttributeError:
                dt = dt

        return np.log(discrete_time_eigenvalues) / dt

    def export_eigenvalues(self, num_evals, eigenvalues, eigenvectors, filename=None):
        """
        Saves a ``num_evals`` number of eigenvalues and eigenvectors to file.

        The files are saved in the output directory and include:

            * ``{system_name}_eigenvalues.dat``: Array of eigenvalues of shape ``(num_evals, 2)`` where the first column corresponds
              to the real part and the second column to the imaginary part.

            * ``{system_name}_stability.h5``: An ``.h5`` file containing the desired number of eigenvalues and eigenvectors of the
              chosen systems.

        The units of the eigenvalues are ``rad/s``.

        Args:
            num_evals (int): Number of eigenvalues to save.
            eigenvalues (np.ndarray): Eigenvalue array.
            eigenvectors (np.ndarray): Matrix of eigenvectors.
            filename (str (optional)): Optional prefix of the output filenames.

        See Also:
            Loading and saving complex arrays:
            https://stackoverflow.com/questions/6494102/how-to-save-and-load-an-array-of-complex-numbers-using-numpy-savetxt/6522396
        """

        stability_folder_path = self.folder

        if filename is None or filename == '':
            filename = ''
        else:
            filename += '_'

        if self.settings['output_file_format'] == 'dat':
            np.savetxt(self.folder + f'/{filename:s}eigenvalues.dat', eigenvalues.view(float).reshape(-1, 2))
            np.savetxt(self.folder + f'/{filename:s}eigenvectors_r.dat', eigenvectors.real)
            np.savetxt(self.folder + f'/{filename:s}eigenvectors_i.dat', eigenvectors.imag)
        elif self.settings['output_file_format'] == 'h5':
            with h5py.File(stability_folder_path + f'/{filename:s}stability.h5', 'w') as f:
                f.create_dataset('eigenvalues', data=eigenvalues, dtype=complex)
                f.create_dataset('eigenvectors', data=eigenvectors, dtype=complex)
                f.create_dataset('num_eigenvalues', data=num_evals, dtype=int)
        else:
            raise TypeError(f'Unrecognised file type saving option {self.settings["output_file_format"]}')
            # this shouldn't happen as the settings_options should check the validity of the setting

    def velocity_analysis(self):
        """
        Velocity analysis for scaled systems.

        Runs the stability analysis for different velocities for aeroelastic systems that have been previously scaled.

        For every velocity, the linear system is updated. This involves updating the structural matrices and the
        coupling matrix. The eigenvalues saved are in continuous time.

        It saves the results to a ``.dat`` file where the first column corresponds to the free stream velocity and the
        second and third columns to the real and imaginary parts of the eigenvalues.
        """

        ulb, uub, num_u = self.settings['velocity_analysis']

        if self.settings['print_info']:
            cout.cout_wrap('Velocity Asymptotic Stability Analysis', 1)
            cout.cout_wrap('Initial velocity: {:02f} m/s'.format(ulb), 1)
            cout.cout_wrap('Final velocity: {:02f} m/s'.format(uub), 1)
            cout.cout_wrap('Number of evaluations: {:g}'.format(num_u), 1)

        u_inf_vec = np.linspace(ulb, uub, int(num_u))

        real_part_plot = []
        imag_part_plot = []
        uinf_part_plot = []

        for i in range(len(u_inf_vec)):
            ss_aeroelastic = self.data.linear.linear_system.update(u_inf_vec[i])

            eigs, eigenvectors = sclalg.eig(ss_aeroelastic.A)

            eigs, eigenvectors = self.sort_eigenvalues(eigs, eigenvectors)

            # Obtain dimensional time
            dt_dimensional = self.data.linear.linear_system.uvlm.sys.ScalingFacts['length'] / u_inf_vec[i] \
                             * ss_aeroelastic.dt

            eigs_cont = np.log(eigs) / dt_dimensional
            Nunst = np.sum(eigs_cont.real > 0)
            fn = np.abs(eigs_cont)

            if self.settings['print_info']:
                cout.cout_wrap('LTI\tu: %.2f m/2\tmax. CT eig. real: %.6f\t' \
                               % (u_inf_vec[i], np.max(eigs_cont.real)))
                cout.cout_wrap('\tN unstab.: %.3d' % (Nunst,))
                cout.cout_wrap(
                    '\tUnstable aeroelastic natural frequency CT(rad/s):' + Nunst * '\t%.2f' % tuple(fn[:Nunst]))

            # Store eigenvalues for plot
            real_part_plot.append(eigs_cont.real)
            imag_part_plot.append(eigs_cont.imag)
            uinf_part_plot.append(np.ones_like(eigs_cont.real) * u_inf_vec[i])

        real_part_plot = np.hstack(real_part_plot)
        imag_part_plot = np.hstack(imag_part_plot)
        uinf_part_plot = np.hstack(uinf_part_plot)

        velocity_file_name = self.folder + '/velocity_analysis_min{:04g}_max{:04g}_nvel{:04g}.dat'.format(
            ulb * 10,
            uub * 10,
            num_u)

        np.savetxt(velocity_file_name,
                   np.concatenate((uinf_part_plot, real_part_plot, imag_part_plot)).reshape((-1, 3), order='F'))

        if self.print_info:
            cout.cout_wrap('\t\tSuccessfully saved velocity analysis to {:s}'.format(velocity_file_name), 2)

    @staticmethod
    def display_root_locus(eigenvalues):
        """
        Displays root locus diagrams.

        Returns the ``fig`` and ``ax`` handles for further editing.

        Returns:
            fig:
            ax:
        """

        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            cout.cout_wrap('Could not plot in asymptoticstability beacuse there is no Matplotlib', 4)
            return
        fig, ax = plt.subplots()

        ax.scatter(np.real(eigenvalues), np.imag(eigenvalues),
                   s=6,
                   color='k',
                   marker='s')
        ax.set_xlabel(r'Real, $\mathbb{R}(\lambda_i)$ [rad/s]')
        ax.set_ylabel(r'Imag, $\mathbb{I}(\lambda_i)$ [rad/s]')
        ax.grid(True)
        fig.show()

        return fig, ax

    def plot_modes(self, eigenvectors):
        r"""
        Plot the aeroelastic mode shapes for the first ``n_modes_to_plot``

        Plots the 0, 45, 90 and 135 degrees phase of the mode. Also plots the reference at the linearisation state.

        .. math:: x_{out} = Re(\Phi_i e^{i\theta})

        for :math:`\theta \in \{0, \pi/4, \pi/2, 3\pi/4}`.

        """
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            cout.cout_wrap('Could not plot in asymptoticstability because there is no Matplotlib', 4)
            return

        mode_shape_list = self.settings['modes_to_plot']

        route = self.folder + '/modes/'
        if not os.path.isdir(route):
            os.makedirs(route, exist_ok=True)

        for mode in mode_shape_list:
            # Scale mode

            beam = self.data.linear.linear_system.beam
            displacement_states = beam.ss.states // 2
            structural_states = beam.ss.states
            structural_modal_coords = beam.sys.modal

            for phase in [0, 1, 2, 3]:
                v = eigenvectors[-structural_states:-structural_states//2, mode]
                v_dot = eigenvectors[-structural_states//2:, mode]

                if beam.sys.clamped:
                    num_dof_rig = 1  # to get the full vector (if 0 line356 returns empty array)
                else:
                    warnings.warn('The rigid body motion contribution to the mode will not be shown, '
                                  'just the flexible contributions.')
                    num_dof_rig = beam.sys.num_dof_rig

                if structural_modal_coords:
                    # project aeroelastic mode back to modal coordinates
                    phi = beam.sys.U
                    eta = phi.dot(v)
                    eta_dot = phi.dot(v_dot)[:-num_dof_rig]
                else:
                    eta = v[:-num_dof_rig]
                    eta_dot = v_dot[:-num_dof_rig]

                eta_phase = np.real(np.exp(1j * phase * np.pi / 4) * eta)
                amplitude_factor = modalutils.scale_mode(self.data,
                                                         eta_phase,
                                                         rot_max_deg=10, perc_max=0.15)
                eta_phase *= amplitude_factor
                zeta_mode = modalutils.get_mode_zeta(self.data, eta_phase)
                modalutils.write_zeta_vtk(zeta_mode, self.data.linear.tsaero0.zeta,
                                          filename_root=route + f'mode_{mode:06g}_phase{phase:04g}')

        # Reference - linearisation state
        eta *= 0
        zeta_mode = modalutils.get_mode_zeta(self.data, eta.real)
        modalutils.write_zeta_vtk(zeta_mode, self.data.linear.tsaero0.zeta, filename_root=route + 'mode_ref')

    @staticmethod
    def sort_eigenvalues(eigenvalues, eigenvectors, frequency_cutoff=0, number_of_eigenvalues=None):
        """
        Sort continuous-time eigenvalues by order of magnitude.

        The conjugate of complex eigenvalues is removed, then if specified, high frequency modes are truncated.
        Finally, the eigenvalues are sorted by largest to smallest real part.

        Args:
            eigenvalues (np.ndarray): Continuous-time eigenvalues
            eigenvectors (np.ndarray): Corresponding right eigenvectors
            frequency_cutoff (float): Cutoff frequency for truncation ``[rad/s]``
            number_of_eigenvalues (int (optional)): Number of eigenvalues to retain

        Returns:
            tuple(np.array, np.array): eigenvalues and eigenvectors
        """

        if frequency_cutoff == 0:
            frequency_cutoff = np.inf

        # Remove poles in the negative imaginary plane (Im(\lambda)<0)
        criteria_a = np.abs(np.imag(eigenvalues)) <= frequency_cutoff
        # criteria_b = np.imag(eigenvalues) > -1e-2
        eigenvalues_truncated = eigenvalues[criteria_a].copy()
        eigenvectors_truncated = eigenvectors[:, criteria_a].copy()

        order = np.argsort(eigenvalues_truncated.real)[::-1]

        if number_of_eigenvalues is not None:
            if number_of_eigenvalues > len(order):
                cout.cout_wrap(f'Desired number of eigenvalues ({number_of_eigenvalues}) exceeds system size '
                               f'({len(order)}) after frequency truncation.', 3)
            else:
                order = order[:number_of_eigenvalues]
        return eigenvalues_truncated[order], eigenvectors_truncated[:, order]
