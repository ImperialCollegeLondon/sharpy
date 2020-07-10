import os
import warnings as warn
import numpy as np
import scipy.linalg as sclalg
import sharpy.utils.settings as settings
from sharpy.utils.solver_interface import solver, BaseSolver, initialise_solver
import sharpy.utils.cout_utils as cout
import sharpy.utils.algebra as algebra
import sharpy.solvers.lindynamicsim as lindynamicsim
import sharpy.structure.utils.modalutils as modalutils
import scipy.sparse as scsp


@solver
class AsymptoticStability(BaseSolver):
    """
    Calculates the asymptotic stability properties of the linearised aeroelastic system by computing
    the corresponding eigenvalues.

    To use an iterative eigenvalue solver, the setting ``iterative_eigvals`` should be set to ``on``. This
    will be beneficial when deailing with very large systems. However, the direct method is
    preferred and more efficient when the system is of a relatively small size (typically around 5000 states).

    Warnings:
        The setting ``modes_to_plot`` to plot the eigenvectors in Paraview is currently under development.

    """
    solver_id = 'AsymptoticStability'
    solver_classification = 'post-processor'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['folder'] = 'str'
    settings_default['folder'] = './output'
    settings_description['folder'] = 'Output folder'

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
    settings_description['export_eigenvalues'] = 'Save eigenvalues and eigenvectors to file. '

    settings_types['display_root_locus'] = 'bool'
    settings_default['display_root_locus'] = False
    settings_description['display_root_locus'] = 'Show plot with eigenvalues on Argand diagram'

    settings_types['velocity_analysis'] = 'list(float)'
    settings_default['velocity_analysis'] = []
    settings_description['velocity_analysis'] = 'List containing min, max and number ' \
                                                'of velocities to analyse the system'

    settings_types['iterative_eigvals'] = 'bool'
    settings_default['iterative_eigvals'] = False
    settings_description['iterative_eigvals'] = 'Calculate the first ``num_evals`` using an iterative solver.'

    settings_types['num_evals'] = 'int'
    settings_default['num_evals'] = 200
    settings_description['num_evals'] = 'Number of eigenvalues to retain.'

    settings_types['modes_to_plot'] = 'list(int)'
    settings_default['modes_to_plot'] = []
    settings_description['modes_to_plot'] = 'List of mode numbers to simulate and plot'

    settings_types['postprocessors'] = 'list(str)'
    settings_default['postprocessors'] = list()
    settings_description['postprocessors'] = 'To be used with ``modes_to_plot``. Under development.'

    settings_types['postprocessors_settings'] = 'dict'
    settings_default['postprocessors_settings'] = dict()
    settings_description['postprocessors_settings'] = 'To be used with ``modes_to_plot``. Under development.'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.settings = None
        self.data = None
        self.folder = None

        self.eigenvalues = None
        self.eigenvectors = None
        self.frequency_cutoff = np.inf
        self.eigenvalue_table = None
        self.num_evals = None

        self.postprocessors = dict()
        self.with_postprocessors = False

    def initialise(self, data, custom_settings=None):
        self.data = data

        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings

        settings.to_custom_types(self.settings, self.settings_types, self.settings_default, no_ctype=True)

        self.num_evals = self.settings['num_evals']

        stability_folder_path = self.settings['folder'] + '/' + self.data.settings['SHARPy']['case'] + '/stability'
        if not os.path.exists(stability_folder_path):
            os.makedirs(stability_folder_path)
        self.folder = stability_folder_path

        if not os.path.exists(stability_folder_path):
            os.makedirs(stability_folder_path)

        if self.settings['print_info']:
            cout.cout_wrap('Dynamical System Eigenvalues')
            eigenvalue_description_file = stability_folder_path + '/eigenvaluetable.txt'
            self.eigenvalue_table = modalutils.EigenvalueTable(filename=eigenvalue_description_file)

        # Output dict
        self.data.linear.stability = dict()

    def run(self, online=False):
        """
        Computes the eigenvalues and eigenvectors

        Returns:
            eigenvalues (np.ndarray): Eigenvalues sorted and frequency truncated
            eigenvectors (np.ndarray): Corresponding mode shapes

        """
        try:
            self.frequency_cutoff = self.settings['frequency_cutoff']
        except AttributeError:
            self.frequency_cutoff = float(self.settings['frequency_cutoff'])

        if self.frequency_cutoff == 0:
            self.frequency_cutoff = np.inf

        if self.settings['reference_velocity'] != 1. and self.data.linear.linear_system.uvlm.scaled:
            ss = self.data.linear.linear_system.update(self.settings['reference_velocity'])
        else:
            ss = self.data.linear.ss

        if self.settings['print_info']:
            cout.cout_wrap('Calculating eigenvalues using direct method')
        eigenvalues, eigenvectors = sclalg.eig(ss.A)

        # Convert DT eigenvalues into CT
        if ss.dt:
            # Obtain dimensional time step
            try:
                ScalingFacts = self.data.linear.linear_system.uvlm.sys.ScalingFacts
                if ScalingFacts['length'] != 1.0 and ScalingFacts['time'] != 1.0:
                    dt = ScalingFacts['length'] / self.settings['reference_velocity'] * ss.dt
                else:
                    dt = ss.dt
            except AttributeError:
                dt = ss.dt
            eigenvalues = np.log(eigenvalues) / dt

        self.num_evals = min(self.num_evals, len(eigenvalues))

        self.eigenvalues, self.eigenvectors = self.sort_eigenvalues(eigenvalues, eigenvectors, self.frequency_cutoff)

        if self.settings['export_eigenvalues']:
            self.export_eigenvalues(self.num_evals)

        if self.settings['print_info']:
            self.eigenvalue_table.print_header(self.eigenvalue_table.headers)
            self.eigenvalue_table.print_evals(self.eigenvalues[:self.num_evals])
            self.eigenvalue_table.close_file()

        if self.settings['display_root_locus']:
            self.display_root_locus()

        # Under development
        if len(self.settings['modes_to_plot']) != 0:
            warn.warn('Plotting modes is under development')
            self.plot_modes()

        if len(self.settings['velocity_analysis']) == 3:
            assert self.data.linear.linear_system.uvlm.scaled, 'The UVLM system is unscaled, unable to rescale the ' \
                                                               'structural equations only. Rerun with a normalised ' \
                                                               'UVLM system.'
            self.velocity_analysis()

        self.data.linear.stability['eigenvectors'] = self.eigenvectors
        self.data.linear.stability['eigenvalues'] = self.eigenvalues

        return self.data

    def export_eigenvalues(self, num_evals):
        """
        Saves a ``num_evals`` number of eigenvalues and eigenvectors to file. The files are saved in the output directoy
        and include:

            * ``eigenvectors.dat``: ``(num_dof, num_evals)`` array of eigenvectors

            * ``eigenvalues_r.dat``: ``(num_evals, 1)`` array of the real part of the eigenvalues

            * ``eigenvalues_i.dat``: ``(num_evals, 1)`` array of the imaginary part of the eigenvalues.

        The units of the eigenvalues are ``rad/s``

        References:
            Loading and saving complex arrays:
            https://stackoverflow.com/questions/6494102/how-to-save-and-load-an-array-of-complex-numbers-using-numpy-savetxt/6522396

        Args:
            num_evals: Number of eigenvalues to save
        """

        stability_folder_path = self.folder

        num_evals = min(num_evals, self.eigenvalues.shape[0])

        np.savetxt(stability_folder_path + '/eigenvalues.dat', self.eigenvalues[:num_evals].view(float).reshape(-1, 2))
        np.savetxt(stability_folder_path + '/eigenvectors_r.dat', self.eigenvectors.real[:, :num_evals])
        np.savetxt(stability_folder_path + '/eigenvectors_i.dat', self.eigenvectors.imag[:, :num_evals])

    def print_eigenvalues(self):
        """
        Prints the eigenvalues to a table with the corresponding natural frequency, period and damping ratios

        """

        self.eigenvalue_table.print_evals(self.eigenvalues[:self.settings['num_evals']])

    def velocity_analysis(self):

        ulb, uub, num_u = self.settings['velocity_analysis']

        if self.settings['print_info']:
            cout.cout_wrap('Velocity Asymptotic Stability Analysis', 1)
            cout.cout_wrap('Initial velocity: %.2f m/s' % ulb, 1)
            cout.cout_wrap('Final velocity: %.2f m/s' % uub, 1)
            cout.cout_wrap('Number of evaluations: %g' % num_u, 1)

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

            cout.cout_wrap('LTI\tu: %.2f m/2\tmax. CT eig. real: %.6f\t' \
                           % (u_inf_vec[i], np.max(eigs_cont.real)))
            cout.cout_wrap('\tN unstab.: %.3d' % (Nunst,))
            cout.cout_wrap('\tUnstable aeroelastic natural frequency CT(rad/s):' + Nunst * '\t%.2f' % tuple(fn[:Nunst]))

            # Store eigenvalues for plot
            real_part_plot.append(eigs_cont.real)
            imag_part_plot.append(eigs_cont.imag)
            uinf_part_plot.append(np.ones_like(eigs_cont.real)*u_inf_vec[i])

        real_part_plot = np.hstack(real_part_plot)
        imag_part_plot = np.hstack(imag_part_plot)
        uinf_part_plot = np.hstack(uinf_part_plot)

        cout.cout_wrap('Saving velocity analysis results...')
        np.savetxt(self.folder + '/velocity_analysis_min%04d_max%04d_nvel%04d.dat' %(ulb*10, uub*10, num_u),
                   np.concatenate((uinf_part_plot, real_part_plot, imag_part_plot)).reshape((-1, 3), order='F'))
        cout.cout_wrap('\tSuccessful', 1)

        self.velocity_results = dict()
        self.data.linear.stability['velocity_results'] = dict()
        self.data.linear.stability['velocity_results']['u_inf'] = uinf_part_plot
        self.data.linear.stability['velocity_results']['evals_real'] = real_part_plot
        self.data.linear.stability['velocity_results']['evals_imag'] = imag_part_plot

    def display_root_locus(self):
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

        ax.scatter(np.real(self.eigenvalues), np.imag(self.eigenvalues),
                   s=6,
                   color='k',
                   marker='s')
        ax.set_xlabel(r'Real, $\mathbb{R}(\lambda_i)$ [rad/s]')
        ax.set_ylabel(r'Imag, $\mathbb{I}(\lambda_i)$ [rad/s]')
        ax.grid(True)
        fig.show()

        return fig, ax

    def plot_modes(self):
        """
        Warnings:
            Under development

        Plot the aeroelastic mode shapes for the first ``n_modes_to_plot``

        """
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            cout.cout_wrap('Could not plot in asymptoticstability beacuse there is no Matplotlib', 4)
            return
        mode_shape_list = self.settings['modes_to_plot']
        for mode in mode_shape_list:
            # Scale mode
            aero_states = self.data.linear.linear_system.uvlm.ss.states
            displacement_states = self.data.linear.linear_system.beam.ss.states // 2
            amplitude_factor = modalutils.scale_mode(self.data,
                                                self.eigenvectors[aero_states:aero_states + displacement_states-9,
                                                mode], rot_max_deg=10, perc_max=0.1)

            fact_rbm = self.scale_rigid_body_mode(self.eigenvectors[:, mode], self.eigenvalues[mode].imag)* 100
            print(fact_rbm)

            t, x = self.mode_time_domain(amplitude_factor, fact_rbm, mode)

            # Initialise postprocessors - new folder for each mode
            # initialise postprocessors
            route = self.settings['folder'] + '/stability/mode_%06d/' % mode
            postprocessors = dict()
            postprocessor_list = ['AerogridPlot', 'BeamPlot']
            postprocessors_settings = dict()
            postprocessors_settings['AerogridPlot'] = {'folder': route,
                                    'include_rbm': 'on',
                                    'include_applied_forces': 'on',
                                    'minus_m_star': 0,
                                    'u_inf': 1
                                    }
            postprocessors_settings['BeamPlot'] = {'folder': route + '/',
                                    'include_rbm': 'on',
                                    'include_applied_forces': 'on'}

            for postproc in postprocessor_list:
                postprocessors[postproc] = initialise_solver(postproc)
                postprocessors[postproc].initialise(
                    self.data, postprocessors_settings[postproc])

            # Plot reference
            for postproc in postprocessor_list:
                self.data = postprocessors[postproc].run(online=True)
            for n in range(t.shape[1]):
                aero_tstep, struct_tstep = lindynamicsim.state_to_timestep(self.data, x[:, n])
                self.data.aero.timestep_info.append(aero_tstep)
                self.data.structure.timestep_info.append(struct_tstep)

                for postproc in postprocessor_list:
                    self.data = postprocessors[postproc].run(online=True)

            # Delete 'modal' timesteps ready for next mode
            del self.data.structure.timestep_info[1:]
            del self.data.aero.timestep_info[1:]

    def mode_time_domain(self, fact, fact_rbm, mode_num, cycles=2):
        """
        Returns a single, scaled mode shape in time domain.

        Args:
            fact: Structural deformation scaling
            fact_rbm: Rigid body motion scaling
            mode_num: Number of mode to plot
            cycles: Number of periods/cycles to plot

        Returns:
            tuple: Time domain array and scaled eigenvector in time.
        """


        # Time domain representation of the mode
        eigenvalue = self.eigenvalues[mode_num]
        natural_freq = np.abs(eigenvalue)
        damping = eigenvalue.real / natural_freq
        period = 2*np.pi / natural_freq
        dt = period/100
        t_dom = np.linspace(0, 2 * period, int(np.ceil(2 * cycles * period/dt)))
        t_dom.shape = (1, len(t_dom))
        eigenvector = self.eigenvectors[:, mode_num]
        eigenvector.shape = (len(eigenvector), 1)

        # eigenvector[-10:] *= fact_rbm
        # eigenvector[-self.data.linear.linear_system.beam.ss.states // 2 - 10: -self.data.linear.linear_system.beam.ss.states] *= fact_rbm

        # State simulation
        # x_sim = np.real(fact_rbm * eigenvector.dot(np.exp(1j*eigenvalue*t_dom)))
        x_sim = fact_rbm * eigenvector.real.dot(np.cos(natural_freq * t_dom) * np.exp(damping * t_dom))

        return t_dom, x_sim

    def reconstruct_mode(self, eig):
        uvlm = self.data.linear.linear_system.uvlm
        # beam = self.data.linear.lsys[sys_id].lsys['LinearBeam']

        # for eig in range(10):
        x_aero = self.eigenvectors[:uvlm.ss.states, eig]
        forces, gamma, gamma_dot, gamma_star = uvlm.unpack_ss_vector(self.data, x_aero, self.data.linear.tsaero0)

        x_struct = self.eigenvectors[uvlm.ss.states:, eig]
        return gamma, gamma_dot, gamma_star, x_struct

    @staticmethod
    def sort_eigenvalues(eigenvalues, eigenvectors, frequency_cutoff=0):
        """
        Sort continuous-time eigenvalues by order of magnitude.

        The conjugate of complex eigenvalues is removed, then if specified, high frequency modes are truncated.
        Finally, the eigenvalues are sorted by largest to smallest real part.

        Args:
            eigenvalues (np.ndarray): Continuous-time eigenvalues
            eigenvectors (np.ndarray): Corresponding right eigenvectors
            frequency_cutoff (float): Cutoff frequency for truncation ``[rad/s]``

        Returns:

        """

        if frequency_cutoff == 0:
            frequency_cutoff = np.inf

        # Remove poles in the negative imaginary plane (Im(\lambda)<0)
        criteria_a = np.abs(np.imag(eigenvalues)) <= frequency_cutoff
        # criteria_b = np.imag(eigenvalues) > -1e-2
        eigenvalues_truncated = eigenvalues[criteria_a].copy()
        eigenvectors_truncated = eigenvectors[:, criteria_a].copy()

        order = np.argsort(eigenvalues_truncated.real)[::-1]

        return eigenvalues_truncated[order], eigenvectors_truncated[:, order]

    @staticmethod
    def scale_rigid_body_mode(eigenvector, freq_d):
        rigid_body_mode = eigenvector[-10:]

        max_angle = 10 * np.pi/180

        v = rigid_body_mode[0:3].real
        omega = rigid_body_mode[3:6].real
        dquat = rigid_body_mode[-4:]
        euler = algebra.quat2euler(dquat)
        max_euler = np.max(np.abs(euler))

        if max_euler >= max_angle:
            fact = max_euler / max_angle
        else:
            fact = 1

        if np.abs(freq_d) < 1e-3:
            fact = 1 / np.max(np.abs(v))
        else:
            max_omega = max_angle * freq_d
            fact = np.max(np.abs(omega)) / max_omega

        return fact
