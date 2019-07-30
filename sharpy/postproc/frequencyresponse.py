import numpy as np
import time
import os
import matplotlib.pyplot as plt
import sharpy.utils.solver_interface as solver_interface
import sharpy.utils.settings as settings_utils
import sharpy.utils.cout_utils as cout


@solver_interface.solver
class FrequencyResponse(solver_interface.BaseSolver):
    """
    Frequency Response Calculator

    """
    solver_id = 'FrequencyResponse'
    solver_classification = 'post-processor'
    
    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['folder'] = 'str'
    settings_default['folder'] = './output'
    settings_description['folder'] = 'Output folder'

    settings_types['compute_fom'] = 'bool'
    settings_default['compute_fom'] = False
    settings_description['compute_fom'] = 'Compute frequency response of full order model (use caution if large)'

    settings_types['frequency_unit'] = 'str'
    settings_default['frequency_unit'] = 'k'
    settings_description['frequency_unit'] = 'Units of frequency, "w" for rad/s, "k" reduced'

    settings_types['frequency_bounds'] = 'list(float)'
    settings_default['frequency_bounds'] = [1e-3, 1]
    settings_description['frequency_bounds'] = 'Lower and upper frequency bounds in the corresponding unit'

    settings_types['num_freqs'] = 'int'
    settings_default['num_freqs'] = 50
    settings_description['num_freqs'] = 'Number of frequencies to evaluate'

    settings_types['quick_plot'] = 'bool'
    settings_default['quick_plot'] = False
    settings_description['quick_plot'] = 'Produce array of plots showing response'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    # settings_types['plot_type'] = 'str'
    # settings_default['plot_type'] = 'bode'
    #
    # settings_types['inputs'] = 'list(int)'
    # settings_default['inputs'] = []
    #
    # settings_types['outputs'] = 'list(int)'
    # settings_default['outputs'] = []

    def __init__(self):

        self.settings = None
        self.data = None
        self.folder = None

        self.ss = None
        self.ssrom = None

        self.nfreqs = None
        self.w_to_k = 1
        self.wv = None

    def initialise(self, data, custom_settings=None):

        self.data = data
        try:
            self.ss = data.linear.linear_system.uvlm.rom.ss
            self.ssrom = data.linear.linear_system.uvlm.ss
        except AttributeError:
            self.ss = data.linear.linear_system.uvlm.ss

        if not custom_settings:
            self.settings = self.data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings_utils.to_custom_types(self.settings, self.settings_types, self.settings_default)

        # Number of interpolation points
        try:
            self.nfreqs = self.rom.frequency.shape[0]
        except AttributeError:
            self.nfreqs = 1

        scaling = self.data.linear.linear_system.uvlm.sys.ScalingFacts
        if self.settings['frequency_unit'] == 'k':
            self.w_to_k = scaling['length'] / scaling['speed']
        else:
            self.w_to_k = 1.

        lb = self.settings['frequency_bounds'][0] / self.w_to_k
        ub = self.settings['frequency_bounds'][1] / self.w_to_k

        nfreqs = self.settings['num_freqs'].value
        self.wv = np.linspace(lb, ub, nfreqs)

    def run(self):
        """
        Get the frequency response of the linear state-space
        Returns:

        """
        Y_freq_rom = None
        Y_freq_fom = None

        cout.cout_wrap('Computing frequency response...')
        if self.settings['compute_fom'].value:
            cout.cout_wrap('Full order system:', 1)
            t0fom = time.time()
            Y_freq_fom = self.ss.freqresp(self.wv)
            tfom = time.time() - t0fom
            self.save_freq_resp(self.wv, Y_freq_fom, 'fom')
            cout.cout_wrap('\tComputed the frequency response of the full order system in %f s' %tfom, 2)
        if self.ssrom is not None:
            cout.cout_wrap('Reduced order system:', 1)
            t0rom = time.time()
            Y_freq_rom = self.ssrom.freqresp(self.wv)
            trom = time.time() - t0rom
            cout.cout_wrap('\tComputed the frequency response of the reduced order system in %f s' %trom, 2)
            self.save_freq_resp(self.wv, Y_freq_rom, 'rom')

        if self.settings['quick_plot'].value:
            self.quick_plot(Y_freq_fom, Y_freq_rom)

        return self.data

    def save_freq_resp(self, wv, Yfreq, filename):

        if not os.path.exists(self.settings['folder']):
            os.makedirs(self.settings['folder'])
        self.folder = self.settings['folder'] + '/' + self.data.settings['SHARPy']['case'] + '/frequencyresponse/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        with open(self.folder + '/freqdata_readme.txt', 'w') as outfile:
            outfile.write('Frequency Response Data Output\n\n')
            outfile.write('Frequency range found in _wv.txt file in rad/s\n')
            outfile.write('Response data from input m to output p in complex form. Column 1 corresponds'
                          ' to the real value and column 2 to the imaginary part.')

        np.savetxt(self.folder + '/' + filename + '_wv.dat', wv)

        for mj in range(self.ss.inputs):
            for pj in range(self.ss.outputs):
                freq_2_cols = Yfreq[pj, mj, :].view(float).reshape(-1, 2)
                np.savetxt(self.folder + '/' + 'Y_freq_' + filename + '_m%02d_p%02d.dat' % (mj, pj),
                           freq_2_cols)

    def quick_plot(self, Y_freq_fom=None, Y_freq_rom=None):
        fig1, ax1 = plt.subplots(nrows=self.ss.inputs, ncols=self.ss.outputs, sharex=True)
        fig2, ax2 = plt.subplots(nrows=self.ss.inputs, ncols=self.ss.outputs, sharex=True)
        for mj in range(self.ss.inputs):
            for pj in range(self.ss.outputs):
                if Y_freq_fom is not None:
                    ax1[mj, pj].plot(self.wv * self.w_to_k, Y_freq_fom[pj, mj, :].real, color='C0')
                    ax2[mj, pj].plot(self.wv * self.w_to_k, Y_freq_fom[pj, mj, :].imag, '--', color='C0')
                if Y_freq_rom is not None:
                    ax1[mj, pj].plot(self.wv * self.w_to_k, Y_freq_rom[pj, mj, :].real, color='C1')
                    ax2[mj, pj].plot(self.wv * self.w_to_k, Y_freq_rom[pj, mj, :].imag, '--', color='C1')

        if self.settings['frequency_unit'] == 'k':
            [ax1[-1, p].set_xlabel('Reduced Frequency, k [-]') for p in range(self.ss.outputs)]
            [ax2[-1, p].set_xlabel('Reduced Frequency, k [-]') for p in range(self.ss.outputs)]
        else:
            [ax1[-1, p].set_xlabel('Reduced Frequency, $\omega$ [rad/s]') for p in range(self.ss.outputs)]
            [ax2[-1, p].set_xlabel('Reduced Frequency, $\omega$ [rad/s]') for p in range(self.ss.outputs)]

        [ax1[m, 0].set_ylabel('Real Y') for m in range(self.ss.inputs)]
        [ax2[m, 0].set_ylabel('Imag Y') for m in range(self.ss.inputs)]

        fig1.show()
        fig2.show()

    # def plot_frequency_response(self, kv, Y_freq_ss, Y_freq_rom, interp_frequencies):
    #
    #     nstates = self.ss.states
    #     if self.rom is not None and Y_freq_rom is not None:
    #         rstates = self.rom.ssrom.states
    #         freqresp_title = 'ROM - %s' % self.rom.algorithm
    #     else:
    #         freqresp_title = ''
    #     if self.settings['frequency_type'] == 'k':
    #         freq_label = 'Reduced Frequency, k'
    #     else:
    #         freq_label = 'Angular Frequency, $\omega$ [rad/s]'
    #
    #
    #     if self.settings['plot_type'] == 'bode':
    #         fig, ax = plt.subplots(nrows=2, sharex=True)
    #
    #         phase_ss = np.angle((Y_freq_ss[0, 0, :]))  # - (np.angle((Y_full_system[0, 0, :])) // np.pi) * 2 * np.pi
    #         if Y_freq_rom is not None:
    #             phase_ssrom = np.angle((Y_freq_rom[0, 0, :]))  #- (np.angle((Y_freq_rom[0, 0, :])) // np.pi) * 2 * np.pi
    #
    #         ax[0].semilogx(kv, np.abs(Y_freq_ss[0, 0, :]),
    #                        lw=4,
    #                        alpha=0.5,
    #                        color='b',
    #                        label='Full - %g states' % nstates)
    #         ax[1].semilogx(kv, phase_ss, ls='-',
    #                        lw=4,
    #                        alpha=0.5,
    #                        color='b')
    #
    #         if Y_freq_rom is not None:
    #             ax[0].semilogx(kv, np.abs(Y_freq_rom[0, 0, :]), ls='-.',
    #                            lw=1.5,
    #                            color='k',
    #                            label='ROM - %g states' % rstates)
    #             ax[1].semilogx(kv, phase_ssrom, ls='-.',
    #                            lw=1.5,
    #                            color='k')
    #
    #         ax[1].set_xlim(0, kv[-1])
    #
    #         ax[0].grid()
    #         ax[1].grid()
    #
    #         if self.settings['frequency_type'] == 'k':
    #             ax[1].set_xlabel('Reduced Frequency, k')
    #         else:
    #             ax[1].set_xlabel('Angular Frequency, $\omega$ [rad/s]')
    #
    #         ax[0].set_ylabel('Gain, M [-]')
    #         ax[1].set_ylabel('Phase, $\Phi$ [rad]')
    #
    #         ax[1].set_ylim([-3.3, 3.3])
    #         ax[1].set_yticks(np.linspace(-np.pi, np.pi, 5))
    #         ax[1].set_yticklabels(['-$\pi$','-$\pi/2$', '0', '$\pi/2$', '$\pi$'])
    #
    #
    #
    #         ax[0].set_title(freqresp_title)
    #         ax[0].legend()
    #
    #
    #         # Plot interpolation regions
    #         nfreqs = self.nfreqs
    #         if nfreqs > 1:
    #             for i in range(nfreqs):
    #                 if interp_frequencies[i] != 0 and interp_frequencies[i] != np.inf:
    #                     index_of_frequency = np.argwhere(kv >= interp_frequencies[i].imag)[0]
    #                     ax[0].plot(interp_frequencies[i].imag,
    #                                np.max(np.abs(Y_freq_ss[0, 0, index_of_frequency])),
    #                                lw=1,
    #                                marker='o',
    #                                color='r')
    #                     ax[1].plot(interp_frequencies[i],
    #                                np.max(np.angle(Y_freq_ss[0, 0, index_of_frequency])),
    #                                lw=1,
    #                                marker='o',
    #                                color='r')
    #         else:
    #             if interp_frequencies != 0 and interp_frequencies != np.inf:
    #                 index_of_frequency = np.argwhere(kv >= interp_frequencies.imag)[0]
    #                 ax[0].plot(interp_frequencies,
    #                            np.max(np.abs(Y_freq_ss[0, 0, index_of_frequency])),
    #                            lw=1,
    #                            marker='o',
    #                            color='r')
    #                 ax[1].plot(interp_frequencies,
    #                            np.max(np.angle(Y_freq_ss[0, 0, index_of_frequency])),
    #                            lw=1,
    #                            marker='o',
    #                            color='r')
    #         fig.show()
    #
    #         self.fig = fig
    #         self.ax = ax
    #
    #     elif self.settings['plot_type'] == 'nyquist':
    #
    #         fig, ax = plt.subplots()
    #
    #         ax.plot(Y_freq_ss[0, 0, :].real, Y_freq_ss[0, 0, :].imag,
    #                 lw=4,
    #                 alpha=0.5,
    #                 color='b',
    #                 label='Full - %g states' % self.ss.states)
    #
    #         ax.plot(Y_freq_rom[0, 0, :].real, Y_freq_rom[0, 0, :].imag,
    #                 ls='-.',
    #                 lw=1.5,
    #                 color='k',
    #                 label='ROM - %g states' % rstates)
    #
    #         fig.show()
    #
    #     elif self.settings['plot_type'] == 'real_and_imaginary_mimo':
    #
    #         nu = Y_freq_rom.shape[0]
    #         ny = Y_freq_rom.shape[1]
    #
    #         fig, ax = plt.subplots(nrows=nu, ncols=ny, sharex=True, squeeze=True, constrained_layout=True)
    #         # fig.suptitle(freqresp_title)
    #
    #         for i in range(nu):
    #             for j in range(ny):
    #                 ax[i, j].semilogx(kv, Y_freq_ss[i, j, :].real,
    #                               lw=4,
    #                               alpha=0.5,
    #                               color='b',
    #                               label='Real - %g states' % nstates)
    #                 ax[i, j].semilogx(kv, Y_freq_ss[i, j, :].imag,
    #                               lw=4,
    #                               alpha=0.5,
    #                               color='b',
    #                               ls='-.',
    #                               label='Imag - %g states' % nstates)
    #                 ax[i, j].semilogx(kv, Y_freq_rom[i, j, :].real, ls='-',
    #                               lw=1.5,
    #                               color='k',
    #                               label='Real - %g states' % rstates)
    #                 ax[i, j].semilogx(kv, Y_freq_rom[i, j, :].imag, ls='-.',
    #                               lw=1.5,
    #                               color='k',
    #                               label='Imag - %g states' % rstates)
    #
    #                 if j == 0:
    #                     ax[i, 0].set_ylabel('To Output [%d]' % (i+1))
    #
    #                 if i == 0:
    #                     ax[0, j].set_title('From Input [%d]' % (j+1))
    #
    #                 if i == ny - 1:
    #                     ax[ny-1, j].set_xlabel(freq_label)
    #
    #         ax[0, 0].legend()
    #
    #         fig.show()
    #         self.fig = fig
    #         self.ax = ax
    #
    #     elif self.settings['plot_type'] == 'real_and_imaginary_siso':
    #         fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, squeeze=True, constrained_layout=True)
    #         ax.plot(kv, Y_freq_ss[0, 0, :].real,
    #                       lw=4,
    #                       alpha=0.5,
    #                       color='b',
    #                       label='Real - %g states' % nstates)
    #         ax.plot(kv, Y_freq_ss[0, 0, :].imag,
    #                       lw=4,
    #                       alpha=0.5,
    #                       color='b',
    #                       ls='-.',
    #                       label='Imag - %g states' % nstates)
    #         ax.plot(kv, Y_freq_rom[0, 0, :].real, ls='-',
    #                       lw=1.5,
    #                       color='k',
    #                       label='Real - %g states' % rstates)
    #         ax.plot(kv, Y_freq_rom[0, 0, :].imag, ls='-.',
    #                       lw=1.5,
    #                       color='k',
    #                       label='Imag - %g states' % rstates)
    #
    #         ax.set_ylabel('Normalised Response')
    #         ax.set_xlabel(freq_label)
    #
    #         ax.legend()
    #
    #         fig.show()
    #         self.fig = fig
    #         self.ax = ax
    #
    #     else:
    #         raise NotImplementedError('%s - Plot type not yet implemented')
    #
    #
    # def savefig(self, filename):
    #     # Incorporate folder paths to save to output folder.
    #     self.fig.savefig(filename)
    #
    #     # if self.data is not None:
    #     #     Uinf0 = self.data.aero.timestep_info[0].u_ext[0][0, 0, 0]
    #     #     c_ref = self.data.aero.timestep_info[0].zeta[0][0, -1, 0] - self.data.aero.timestep_info[0].zeta[0][0, 0, 0]
    #     #     ds = 2. / self.data.aero.aero_dimensions[0][0]  # Spatial discretisation
    #     #     fs = 1. / ds
    #     #     fn = fs / 2.
    #     #     ks = 2. * np.pi * fs
    #     #     kn = 2. * np.pi * fn  # Nyquist frequency
    #     #     Nk = 151  # Number of frequencies to evaluate
    #     #     kv = np.linspace(1e-3, kn, Nk)  # Reduced frequency range
    #     #     wv = 2. * Uinf0 / c_ref * kv  # Angular frequency range
    #     # else:
    #     #     kv = wv
    #     #     c_ref = 2
    #     #     Uinf0 = 1
    #     #
    #     # frequency = self.frequency
    #     # # TODO to be modified for plotting purposes when using multi rational interpolation
    #     # try:
    #     #     nfreqs = frequency.shape[0]
    #     # except AttributeError:
    #     #     nfreqs = 1
    #     #
    #     # if frequency is None:
    #     #     k_rom = np.inf
    #     # else:
    #     #     if self.ss.dt is not None:
    #     #         ct_frequency = np.log(frequency)/self.ss.dt
    #     #         k_rom = c_ref * ct_frequency * 0.5 / Uinf0
    #     #     else:
    #     #         k_rom = c_ref * frequency * 0.5 / Uinf0
    #     #
    #     # display_frequency = '$\sigma$ ='
    #     # if nfreqs > 1:
    #     #     display_frequency += ' ['
    #     #     for i in range(nfreqs):
    #     #         if type(k_rom[i]) == complex:
    #     #             display_frequency += ' %.1f + %.1fj' % (k_rom[i].real, k_rom[i].imag)
    #     #         else:
    #     #             display_frequency += ' %.1f' % k_rom[i]
    #     #         display_frequency += ','
    #     #     display_frequency += ']'
    #     # else:
    #     #     if type(k_rom) == complex:
    #     #         display_frequency += ', %.1f + %.1fj' % (k_rom.real, k_rom.imag)
    #     #     else:
    #     #         display_frequency += ', %.1f' % k_rom
    #     #
    #     # nstates = self.ss.states
    #     # rstates = self.ssrom.states
    #     #
    #     # # Compute the frequency response
    #     # Y_full_system = self.ss.freqresp(wv)
    #     # Y_freq_rom = self.ssrom.freqresp(wv)
    #     #
    #     # rel_error = (Y_freq_rom[0, 0, :] - Y_full_system[0, 0, :]) / Y_full_system[0, 0, :]
    #     #
    #     # fig, ax = plt.subplots(nrows=2)
    #     #
    #     # if plot_figures:
    #     #
    #     #     phase_ss = np.angle((Y_full_system[0, 0, :])) # - (np.angle((Y_full_system[0, 0, :])) // np.pi) * 2 * np.pi
    #     #     phase_ssrom = np.angle((Y_freq_rom[0, 0, :])) #- (np.angle((Y_freq_rom[0, 0, :])) // np.pi) * 2 * np.pi
    #     #
    #     #     ax[0].semilogx(kv, np.abs(Y_full_system[0, 0, :]),
    #     #                    lw=4,
    #     #                    alpha=0.5,
    #     #                    color='b',
    #     #                    label='UVLM - %g states' % nstates)
    #     #     ax[1].semilogx(kv, phase_ss, ls='-',
    #     #                    lw=4,
    #     #                    alpha=0.5,
    #     #                    color='b')
    #     #
    #     #     ax[1].set_xlim(0, kv[-1])
    #     #     ax[0].grid()
    #     #     ax[1].grid()
    #     #     ax[0].semilogx(kv, np.abs(Y_freq_rom[0, 0, :]), ls='-.',
    #     #                    lw=1.5,
    #     #                    color='k',
    #     #                    label='ROM - %g states' % rstates)
    #     #     ax[1].semilogx(kv, phase_ssrom, ls='-.',
    #     #                    lw=1.5,
    #     #                    color='k')
    #     #
    #     #     # axins0 = inset_axes(ax[0], 1, 1, loc=1)
    #     #     # axins0.semilogx(kv, np.abs(Y_full_system[0, 0, :]),
    #     #     #             lw=4,
    #     #     #             alpha=0.5,
    #     #     #             color='b')
    #     #     # axins0.semilogx(kv, np.abs(Y_freq_rom[0, 0, :]), ls='-.',
    #     #     #             lw=1.5,
    #     #     #             color='k')
    #     #     # axins0.set_xlim([0, 1])
    #     #     # axins0.set_ylim([0, 0.1])
    #     #     #
    #     #     # axins1 = inset_axes(ax[1], 1, 1.25, loc=1)
    #     #     # axins1.semilogx(kv, np.angle((Y_full_system[0, 0, :])), ls='-',
    #     #     #             lw=4,
    #     #     #             alpha=0.5,
    #     #     #             color='b')
    #     #     # axins1.semilogx(kv, np.angle((Y_freq_rom[0, 0, :])), ls='-.',
    #     #     #             lw=1.5,
    #     #     #             color='k')
    #     #     # axins1.set_xlim([0, 1])
    #     #     # axins1.set_ylim([-3.5, 3.5])
    #     #
    #     #     ax[1].set_xlabel('Reduced Frequency, k')
    #     #     ax[1].set_ylim([-3.3, 3.3])
    #     #     ax[1].set_yticks(np.linspace(-np.pi, np.pi, 5))
    #     #     ax[1].set_yticklabels(['-$\pi$','-$\pi/2$', '0', '$\pi/2$', '$\pi$'])
    #     #     # ax.set_ylabel('Normalised Response')
    #     #     freqresp_title = 'ROM - %s, r = %g, %s' % (self.algorithm, rstates, display_frequency)
    #     #     ax[0].set_title(freqresp_title)
    #     #     ax[0].legend()
    #     #
    #     #
    #     #
    #     #     # Plot interpolation regions
    #     #     if nfreqs > 1:
    #     #         for i in range(nfreqs):
    #     #             if k_rom[i] != 0 and k_rom[i] != np.inf:
    #     #                 index_of_frequency = np.argwhere(kv >= k_rom[i])[0]
    #     #                 ax[0].plot(k_rom[i],
    #     #                            np.max(np.abs(Y_full_system[0, 0, index_of_frequency])),
    #     #                            lw=1,
    #     #                            marker='o',
    #     #                            color='r')
    #     #                 ax[1].plot(k_rom[i],
    #     #                            np.max(np.angle(Y_full_system[0, 0, index_of_frequency])),
    #     #                            lw=1,
    #     #                            marker='o',
    #     #                            color='r')
    #     #     else:
    #     #         if k_rom != 0 and k_rom != np.inf:
    #     #             index_of_frequency = np.argwhere(kv >= k_rom)[0]
    #     #             ax[0].plot(k_rom,
    #     #                        np.max(np.abs(Y_full_system[0, 0, index_of_frequency])),
    #     #                        lw=1,
    #     #                        marker='o',
    #     #                        color='r')
    #     #             ax[1].plot(k_rom,
    #     #                        np.max(np.angle(Y_full_system[0, 0, index_of_frequency])),
    #     #                        lw=1,
    #     #                        marker='o',
    #     #                        color='r')
    #     #     fig.show()
    #     #     # fig.savefig('./figs/theo_rolled/Freq_resp%s.eps' % freqresp_title)
    #     #     # fig.savefig('./figs/theo_rolled/Freq_resp%s.png' % freqresp_title)
    #     #
    #     #     # Relative error
    #     #     fig, ax = plt.subplots()
    #     #
    #     #     real_rel_error = np.abs(rel_error.real)
    #     #     imag_rel_error = np.abs(rel_error.imag)
    #     #
    #     #     ax.loglog(kv, real_rel_error,
    #     #               color='k',
    #     #               lw=1.5,
    #     #               label='Real')
    #     #
    #     #     ax.loglog(kv, imag_rel_error,
    #     #               ls='--',
    #     #               color='k',
    #     #               lw=1.5,
    #     #               label='Imag')
    #     #
    #     #     errresp_title = 'ROM - %s, r = %g, %s' % (self.algorithm, rstates, display_frequency)
    #     #     ax.set_title(errresp_title)
    #     #     ax.set_xlabel('Reduced Frequency, k')
    #     #     ax.set_ylabel('Relative Error')
    #     #     ax.set_ylim([1e-5, 1])
    #     #     ax.legend()
    #     #     fig.show()
    #
    #         # fig.savefig('./figs/theo_rolled/Err_resp%s.eps' % errresp_title)
    #         # fig.savefig('./figs/theo_rolled/Err_resp%s.png' % errresp_title)
    #
    # # def display_frequency(self):
    # #
    # #     frequency = self.ssrom.frequency
    # #
    # #     display_frequency = '$\sigma$ ='
    # #     if self.nfreqs > 1:
    # #         for i in range(self.nfreqs):
    # #         pass
    # #
    # #
    # #
    # #
    # #     if frequency is None:
    # #         k_rom = np.inf
    # #     else:
    # #         if self.ss.dt is not None:
    # #             ct_frequency = np.log(frequency)/self.ss.dt
    # #             k_rom = c_ref * ct_frequency * 0.5 / Uinf0
    # #         else:
    # #             k_rom = c_ref * frequency * 0.5 / Uinf0
    # #
    # #     display_frequency = '$\sigma$ ='
    # #     if nfreqs > 1:
    # #         display_frequency += ' ['
    # #         for i in range(nfreqs):
    # #             if type(k_rom[i]) == complex:
    # #                 display_frequency += ' %.1f + %.1fj' % (k_rom[i].real, k_rom[i].imag)
    # #             else:
    # #                 display_frequency += ' %.1f' % k_rom[i]
    # #             display_frequency += ','
    # #         display_frequency += ']'
    # #     else:
    # #         if type(k_rom) == complex:
    # #             display_frequency += ', %.1f + %.1fj' % (k_rom.real, k_rom.imag)
    # #         else:
    # #             display_frequency += ', %.1f' % k_rom
