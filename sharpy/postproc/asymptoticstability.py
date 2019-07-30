import matplotlib.pyplot as plt
import numpy as np
import sharpy.utils.settings as settings
from sharpy.utils.solver_interface import solver, BaseSolver, initialise_solver
import sharpy.utils.h5utils as h5
import sharpy.solvers.modal as modal
import sharpy.utils.cout_utils as cout
import sharpy.utils.algebra as algebra
import sharpy.solvers.lindynamicsim as lindynamicsim
import pandas as pd
import os
import sharpy.structure.utils.modalutils as modalutils
import scipy.linalg as sclalg

@solver
class AsymptoticStability(BaseSolver):
    """
    Calculates the asymptotic stability properties of aeroelastic systems by creating linearised systems and computing
    the corresponding eigenvalues

    Todo:
        Better integration of the linear system settings (create a loader and check that the system has not been
        previously assembled.

    Warnings:
        Currently under development.

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

    settings_types['frequency_cutoff'] = 'float'
    settings_default['frequency_cutoff'] = 0
    settings_description['frequency_cutoff'] = 'Truncate higher frequency modes. If zero none are truncated'

    settings_types['export_eigenvalues'] = 'bool'
    settings_default['export_eigenvalues'] = False
    settings_description['export_eigenvalues'] = 'Save eigenvalues and eigenvectors to file'

    settings_types['display_root_locus'] = 'bool'
    settings_default['display_root_locus'] = False
    settings_description['display_root_locus'] = 'Show plot with eigenvalues on Argand diagram'

    settings_types['velocity_analysis'] = 'list(float)'
    settings_default['velocity_analysis'] = []
    settings_description['velocity_analysis'] = 'List containing min, max and number ' \
                                                'of velocities to analyse the system'

    settings_types['modes_to_plot'] = 'list(int)'
    settings_default['modes_to_plot'] = []
    settings_description['modes_to_plot'] = 'List of mode numbers to simulate and plot'

    settings_types['num_evals'] = 'int'
    settings_default['num_evals'] = 200
    settings_description['num_evals'] = 'Number of eigenvalues to retain.'

    settings_types['postprocessors'] = 'list(str)'
    settings_default['postprocessors'] = list()

    settings_types['postprocessors_settings'] = 'dict'
    settings_default['postprocessors_settings'] = dict()


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

        self.postprocessors = dict()
        self.with_postprocessors = False

    def initialise(self, data, custom_settings=None):
        self.data = data

        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings

        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        # # Initialise postproc
        # # initialise postprocessors
        # # self.postprocessors = dict()
        # if len(self.settings['postprocessors']) > 0:
        #     self.with_postprocessors = True
        # for postproc in self.settings['postprocessors']:
        #     self.postprocessors[postproc] = initialise_solver(postproc)
        #     self.postprocessors[postproc].initialise(
        #         self.data, self.settings['postprocessors_settings'][postproc])

        stability_folder_path = self.settings['folder'] + '/' + self.data.settings['SHARPy']['case'] + '/stability'
        if not os.path.exists(stability_folder_path):
            os.makedirs(stability_folder_path)
        self.folder = stability_folder_path

        if not os.path.exists(stability_folder_path):
            os.makedirs(stability_folder_path)

        if self.settings['print_info']:
            cout.cout_wrap('Dynamical System Eigenvalues')
            self.eigenvalue_table = modalutils.EigenvalueTable()
            self.eigenvalue_table.print_header(self.eigenvalue_table.headers)

    def run(self):
        """
        Computes the eigenvalues and eigenvectors

        Returns:
            eigenvalues (np.ndarray): Eigenvalues sorted and frequency truncated
            eigenvectors (np.ndarray): Corresponding mode shapes

        """
        try:
            self.frequency_cutoff = self.settings['frequency_cutoff'].value
        except AttributeError:
            self.frequency_cutoff = float(self.settings['frequency_cutoff'])

        if self.frequency_cutoff == 0:
            self.frequency_cutoff = np.inf

        ss = self.data.linear.ss

        # Calculate eigenvectors and eigenvalues of the full system
        eigenvalues, eigenvectors = np.linalg.eig(ss.A)

        # Convert DT eigenvalues into CT
        if ss.dt:
            # Obtain dimensional time step
            ScalingFacts = self.data.linear.linear_system.uvlm.sys.ScalingFacts
            if ScalingFacts['length'] != 1.0 and ScalingFacts['time'] != 1.0:
                dt = ScalingFacts['length'] * 2 / self.data.aero.surface_m[0] / ScalingFacts['speed']
                assert np.abs(dt - ScalingFacts['time'] * ss.dt) < 1e-14, 'dimensional time-scaling not correct!'
            else:
                dt = ss.dt
            eigenvalues = np.log(eigenvalues) / dt

        self.eigenvalues, self.eigenvectors = self.sort_eigenvalues(eigenvalues, eigenvectors, self.frequency_cutoff)

        if self.settings['export_eigenvalues'].value:
            self.export_eigenvalues(self.settings['num_evals'].value)

        if self.settings['print_info'].value:
            self.eigenvalue_table.print_evals(eigenvalues[:self.settings['num_evals'].value])

        if self.settings['display_root_locus']:
            self.display_root_locus()

        # Under development
        if self.settings['modes_to_plot']:
            self.plot_modes()

        if self.settings['velocity_analysis']:
            self.velocity_analysis()

        self.data.linear.stability = dict()
        self.data.linear.stability['eigenvectors'] = self.eigenvectors
        self.data.linear.stability['eigenvalues'] = self.eigenvalues
        # self.data.linear.stability.mode_shapes = mode_shape_list

        return self.data

    def export_eigenvalues(self, num_evals):
        """
        Saves a certain number of eigenvalues and eigenvectors to file

        References:
            Loading and saving complex arrays:
            https://stackoverflow.com/questions/6494102/how-to-save-and-load-an-array-of-complex-numbers-using-numpy-savetxt/6522396

        Args:
            num_evals: Number of eigenvalues to save
        """

        stability_folder_path = self.folder

        evec_pd = pd.DataFrame(data=self.eigenvectors[:, :num_evals])
        # eval_pd = pd.DataFrame(data=[self.eigenvalues.real, self.eigenvalues.imag]).T
        evec_pd.to_csv(stability_folder_path + '/eigenvectors.csv')
        # eval_pd.to_csv(stability_folder_path + '/eigenvalues.csv')
        np.savetxt(stability_folder_path + '/eigenvalues.dat', self.eigenvalues[:num_evals].view(float).reshape(-1, 2))

    def print_eigenvalues(self):
        """
        Prints the eigenvalues to a table with the corresponding natural frequency, period and damping ratios

        """
        # for eval in range(self.settings['num_evals'].value):
        #     eigenvalue = self.eigenvalues[eval]
        #     omega_n = np.abs(eigenvalue)
        #     omega_d = np.abs(eigenvalue.imag)
        #     damping_ratio = -eigenvalue.real / omega_n
        #     f_n = omega_n / 2 / np.pi
        #     f_d = omega_d / 2 / np.pi
        #     period = 1 / f_d
        #     self.eigenvalue_table.print_line([eval, eigenvalue.real, eigenvalue.imag, f_n, f_d, damping_ratio, period])

        self.eigenvalue_table.print_evals(self.eigenvalues[:self.settings['num_evals'].value])

    def velocity_analysis(self):

        cout.cout_wrap('Velocity Asymptotic Stability Analysis')

        ulb, uub, num_u = self.settings['velocity_analysis']

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

            cout.cout_wrap('CLTI\tu: %.2f m/2\tmax.eig. real: %.6f\t' \
                           % (u_inf_vec[i], np.max(eigs_cont.real)))
            cout.cout_wrap('\tN unstab.: %.3d' % (Nunst,))
            print('\tUnstable aeroelastic natural frequency CT(rad/s):' + Nunst * '\t%.2f' % tuple(fn[:Nunst]))

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

    def display_root_locus(self):
        """
        Displays root locus diagrams.

        Returns the ``fig`` and ``ax`` handles for further editing.

        Returns:
            fig:
            ax:
        """

        # Title
        fig, ax = plt.subplots()

        ax.scatter(np.real(self.eigenvalues), np.imag(self.eigenvalues),
                   s=6,
                   color='k',
                   marker='s')
        ax.set_xlabel('Real, $\mathbb{R}(\lambda_i)$ [rad/s]')
        ax.set_ylabel('Imag, $\mathbb{I}(\lambda_i)$ [rad/s]')
        # ax.set_ylim([0, self.frequency_cutoff])
        ax.grid(True)
        # ax.set_ylim(-90,90)
        # ax.set_xlim(-10,1)
        fig.show()

        return fig, ax

    def plot_modes(self):
        """
        Warnings:
            under development

        Plot the aeroelastic mode shapes for the first n_modes_to_plot

        Todo:
            Export to paraview format

        Returns:

        """
        mode_shape_list = self.settings['modes_to_plot']
        for mode in mode_shape_list:
            # Scale mode
            aero_states = self.data.linear.linear_system.uvlm.ss.states
            displacement_states = self.data.linear.linear_system.beam.ss.states // 2
            amplitude_factor = modal.scale_mode(self.data,
                                                self.eigenvectors[aero_states:aero_states + displacement_states-10,
                                                mode], rot_max_deg=10, perc_max=0.1)

            fact_rbm = self.scale_rigid_body_mode(self.eigenvectors[:, mode], self.eigenvalues[mode].imag)
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
                aero_tstep, struct_tstep = lindynamicsim.state_to_timestep(self.data, None, x[:, n])
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

        eigenvector[-10:] *= fact_rbm
        eigenvector[-self.data.linear.linear_system.beam.ss.states // 2 - 10: -self.data.linear.linear_system.beam.ss.states] *= fact_rbm

        # State simulation
        x_sim = np.real(fact * eigenvector.dot(np.exp(1j*eigenvalue*t_dom)))
        # x_sim = fact * eigenvector.real.dot(np.sin(natural_freq * t_dom) * np.exp(damping * t_dom))

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

if __name__ == '__main__':
    u_inf = 140
    try:
        data = h5.readh5('/home/ng213/code/sharpy/tests/linear/goland_wing/cases/output/goland_u%04g.data.h5' %u_inf).data
    except FileNotFoundError:
        raise FileNotFoundError('Unable to find test case')


    integr_order = 2
    predictor = True
    sparse = False

    aeroelastic_settings = {'LinearUvlm':{
        'dt': data.settings['LinearUvlm']['dt'],
        'integr_order': integr_order,
        'density': 1.020,
        'remove_predictor': predictor,
        'use_sparse': sparse,
        'ScalingDict': {'length': 1,
                        'speed': 1,
                        'density': 1},
        'rigid_body_motion': False},
        'frequency_cutoff': 0,
        'export_eigenvalues': True,
    }

    analysis = AsymptoticStability()

    analysis.initialise(data, aeroelastic_settings)
    eigenvalues, eigenvectors = analysis.run()

    fig, ax = analysis.display_root_locus()

    # analysis.plot_modes(10)