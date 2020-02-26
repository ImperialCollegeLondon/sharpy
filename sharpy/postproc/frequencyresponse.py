import numpy as np
import time
import os
import sharpy.utils.solver_interface as solver_interface
import sharpy.utils.settings as settings_utils
import sharpy.utils.cout_utils as cout
import warnings


@solver_interface.solver
class FrequencyResponse(solver_interface.BaseSolver):
    """
    Frequency Response Calculator

    Computes the frequency response of a linear system. If a reduced order model has been created, a comparison is
    made between the two responses.

    """
    solver_id = 'FrequencyResponse'
    solver_classification = 'post-processor'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    settings_types['folder'] = 'str'
    settings_default['folder'] = './output'
    settings_description['folder'] = 'Output folder'

    settings_types['print_info'] = 'bool'
    settings_default['print_info'] = False
    settings_description['print_info'] = 'Write output to screen'

    settings_types['compute_fom'] = 'bool'
    settings_default['compute_fom'] = False
    settings_description['compute_fom'] = 'Compute frequency response of full order model (use caution if large)'

    settings_types['load_fom'] = 'str'
    settings_default['load_fom'] = ''
    settings_description['load_fom'] = 'Folder to locate full order model frequency response data'

    settings_types['frequency_unit'] = 'str'
    settings_default['frequency_unit'] = 'k'
    settings_description['frequency_unit'] = 'Units of frequency, "w" for rad/s, "k" reduced'
    settings_options['frequency_unit'] = ['w', 'k']

    settings_types['frequency_bounds'] = 'list(float)'
    settings_default['frequency_bounds'] = [1e-3, 1]
    settings_description['frequency_bounds'] = 'Lower and upper frequency bounds in the corresponding unit'

    settings_types['num_freqs'] = 'int'
    settings_default['num_freqs'] = 50
    settings_description['num_freqs'] = 'Number of frequencies to evaluate'

    settings_types['quick_plot'] = 'bool'
    settings_default['quick_plot'] = False
    settings_description['quick_plot'] = 'Produce array of ``.png`` plots showing response. Requires matplotlib'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description, settings_options)

    def __init__(self):

        self.settings = None
        self.data = None
        self.folder = None

        self.ss = None
        self.ssrom = None

        self.w_to_k = 1
        self.wv = None

    def initialise(self, data, custom_settings=None):

        self.data = data
        try:
            rom_method = data.linear.linear_system.uvlm.settings['rom_method'][0]
            self.ss = data.linear.linear_system.uvlm.rom[rom_method].ss
            self.ssrom = data.linear.linear_system.uvlm.ss
        except IndexError:
            self.ss = data.linear.linear_system.uvlm.ss

        if not custom_settings:
            self.settings = self.data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings_utils.to_custom_types(self.settings, self.settings_types, self.settings_default, self.settings_options)

        scaling = self.data.linear.linear_system.uvlm.sys.ScalingFacts
        if self.settings['frequency_unit'] == 'k':
            self.w_to_k = scaling['length'] / scaling['speed']
        else:
            self.w_to_k = 1.

        lb = self.settings['frequency_bounds'][0] / self.w_to_k
        ub = self.settings['frequency_bounds'][1] / self.w_to_k

        nfreqs = self.settings['num_freqs'].value
        self.wv = np.linspace(lb, ub, nfreqs)

        if not os.path.exists(self.settings['folder']):
            os.makedirs(self.settings['folder'])
        self.folder = self.settings['folder'] + '/' + self.data.settings['SHARPy']['case'] + '/frequencyresponse/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def run(self):
        """
        Get the frequency response of the linear state-space
        Returns:

        """
        Y_freq_rom = None
        Y_freq_fom = None

        compute_fom = False

        if self.settings['load_fom'] != '':
            if os.path.exists(self.settings['load_fom']):
                try:
                    Y_freq_fom = self.load_frequency_data()
                except OSError:
                    compute_fom = True
            else:
                compute_fom = True

        if (self.settings['compute_fom'].value and self.settings['load_fom'] == '') or compute_fom:
            if self.settings['print_info']:
                cout.cout_wrap('Computing frequency response...')
                cout.cout_wrap('Full order system:', 1)
            t0fom = time.time()
            Y_freq_fom = self.ss.freqresp(self.wv)
            tfom = time.time() - t0fom
            self.save_freq_resp(self.wv, Y_freq_fom, 'fom')
            if self.settings['print_info']:
                cout.cout_wrap('\tComputed the frequency response of the full order system in %f s' % tfom, 2)

        if self.ssrom is not None:
            if self.settings['print_info']:
                cout.cout_wrap('Computing frequency response...')
                cout.cout_wrap('Reduced order system:', 1)
            t0rom = time.time()
            Y_freq_rom = self.ssrom.freqresp(self.wv)
            trom = time.time() - t0rom
            if self.settings['print_info']:
                cout.cout_wrap('\tComputed the frequency response of the reduced order system in %f s' % trom, 2)
            self.save_freq_resp(self.wv, Y_freq_rom, 'rom')

            if Y_freq_fom is not None:
                frequency_error(Y_freq_fom, Y_freq_rom, self.wv)

        if self.settings['quick_plot'].value:
            self.quick_plot(Y_freq_fom, Y_freq_rom)

        return self.data

    def save_freq_resp(self, wv, Yfreq, filename):

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
        try:
            cout.cout_wrap('Creating Quick plots of the frequency response')
            import matplotlib.pyplot as plt
            for mj in range(self.ss.inputs):
                for pj in range(self.ss.outputs):
                    fig1, ax1 = plt.subplots()
                    fig_title = 'in%02g_out%02g' % (mj, pj)
                    ax1.set_title(fig_title)
                    if Y_freq_fom is not None:
                        ax1.plot(self.wv * self.w_to_k, Y_freq_fom[pj, mj, :].real, color='C0', label='Real FOM')
                        ax1.plot(self.wv * self.w_to_k, Y_freq_fom[pj, mj, :].imag, '--', color='C0', label='Imag FOM')
                    if Y_freq_rom is not None:
                        ax1.plot(self.wv * self.w_to_k, Y_freq_rom[pj, mj, :].real, color='C1', label='Real ROM')
                        ax1.plot(self.wv * self.w_to_k, Y_freq_rom[pj, mj, :].imag, '--', color='C1', label='Imag FOM')
                    ax1.legend()
                    if self.settings['frequency_unit'] == 'k':
                        ax1.set_xlabel('Reduced Frequency, k [-]')
                    else:
                        ax1.set_xlabel(r'Frequency, $\omega$ [rad/s]')

                    ax1.set_ylabel('Y')
                    fig1.savefig(self.folder + '/' + fig_title + '.png')
                    plt.close()

            cout.cout_wrap('\tPlots saved to %s' % self.folder, 1)
        except ModuleNotFoundError:
            warnings.warn('Matplotlib not found - skipping plot')

    def load_frequency_data(self):
        if self.settings['print_info']:
            cout.cout_wrap('Loading frequency response from:')
            cout.cout_wrap('\t%s' % self.settings['load_fom'], 1)
        Y_freq_fom = np.zeros((self.ss.outputs, self.ss.inputs, len(self.wv)), dtype=complex)
        for m in range(self.ss.inputs):
            for p in range(self.ss.outputs):
                y_load = np.loadtxt(self.settings['load_fom'] +
                                    '/Y_freq_fom_m%02g_p%02g.dat' %(m,p)).view(complex)
                y_load.shape = (y_load.shape[0], )
                Y_freq_fom[p, m, :] = y_load

        return Y_freq_fom


def frequency_error(Y_fom, Y_rom, wv):
    n_in = Y_fom.shape[1]
    n_out = Y_fom.shape[0]
    cout.cout_wrap('Computing error in frequency response')
    max_error = np.zeros((n_out, n_in, 2))
    for m in range(n_in):
        for p in range(n_out):
            cout.cout_wrap('m = %g, p = %g' %(m, p))
            max_error[p, m, 0] = error_between_signals(Y_fom[p, m, :].real,
                                                            Y_rom[p, m, :].real,
                                                            wv, 'real')
            max_error[p, m, 1] = error_between_signals(Y_fom[p, m, :].imag,
                                                            Y_rom[p, m, :].imag,
                                                            wv, 'imag')

    if np.max(np.log10(max_error)) >= 0:
        warnings.warn('Significant mismatch in the frequency response of the ROM and FOM')

    return np.max(max_error)


def error_between_signals(sig1, sig2, wv, sig_title=''):
    abs_error = np.abs(sig1 - sig2)
    max_error = np.max(abs_error)
    max_error_index = np.argmax(abs_error)
    pct_error = max_error/sig1[max_error_index]

    max_err_freq = wv[max_error_index]
    if 1e-1 > max_error > 1e-3:
        c = 3
    elif max_error >= 1e-1:
        c = 4
    else:
        c = 1
    cout.cout_wrap('\tError Magnitude -%s-: log10(error) = %.2f (%.2f pct) at %.2f rad/s'
                   % (sig_title, np.log10(max_error), pct_error, max_err_freq), c)

    return max_error

