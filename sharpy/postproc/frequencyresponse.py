import numpy as np
import time
import os
import sharpy.utils.solver_interface as solver_interface
import sharpy.utils.settings as settings_utils
import sharpy.utils.cout_utils as cout
import warnings
import sharpy.linear.src.libss as libss
import h5py as h5


@solver_interface.solver
class FrequencyResponse(solver_interface.BaseSolver):
    """
    Frequency Response Calculator.

    Computes the frequency response of a built linear system. The frequency will be calculated for the systems
    specified in the ``target_system`` list. The desired ``frequency_unit`` will be either ``w`` for radians/s or ``k``
    for reduced frequency (if the system is scaled). The ``frequency_bounds`` setting will set the lower and upper
    bounds of the response, while ``num_freqs`` will specify the number of evaluations. The option ``frequency_spacing``
    allows you to space the evaluations point following a ``log`` or ``linear`` spacing.

    This will be saved to a binary ``.h5`` file as detailed in :func:`save_freq_resp`.

    Finally, the ``quick_plot`` option will plot some quick and dirty bode plots of the response. This requires
    access to ``matplotlib``.

    """
    solver_id = 'FrequencyResponse'
    solver_classification = 'post-processor'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    settings_types['folder'] = 'str'
    settings_default['folder'] = './output'
    settings_description['folder'] = 'Output folder.'

    # settings_types['print_info'] = 'bool'
    # settings_default['print_info'] = False
    # settings_description['print_info'] = 'Write output to screen.'

    settings_types['target_system'] = 'list(str)'
    settings_default['target_system'] = ['aeroelastic']
    settings_description['target_system'] = 'System or systems for which to find frequency response.'
    settings_options['target_system'] = ['aeroelastic', 'aerodynamic', 'structural']

    # >>>>>>>>>>>>>>>>>> MOVE TO ROM COMPARISON MODULE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # settings_types['compute_fom'] = 'bool'
    # settings_default['compute_fom'] = False
    # settings_description['compute_fom'] = 'Compute frequency response of full order model (use caution if large).'

    # settings_types['load_fom'] = 'str'
    # settings_default['load_fom'] = ''
    # settings_description['load_fom'] = 'Folder to locate full order model frequency response data.'
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    settings_types['frequency_unit'] = 'str'
    settings_default['frequency_unit'] = 'k'
    settings_description['frequency_unit'] = 'Units of frequency, ``w`` for rad/s, ``k`` reduced.'
    settings_options['frequency_unit'] = ['w', 'k']

    settings_types['frequency_bounds'] = 'list(float)'
    settings_default['frequency_bounds'] = [1e-3, 1]
    settings_description['frequency_bounds'] = 'Lower and upper frequency bounds in the corresponding unit.'

    settings_types['frequency_spacing'] = 'str'
    settings_default['frequency_spacing'] = 'linear'
    settings_description['frequency_spacing'] = 'Compute the frequency response in a ``linear`` or ``log`` grid.'
    settings_options['frequency_spacing'] = ['linear', 'log']

    settings_types['num_freqs'] = 'int'
    settings_default['num_freqs'] = 50
    settings_description['num_freqs'] = 'Number of frequencies to evaluate.'

    settings_types['quick_plot'] = 'bool'
    settings_default['quick_plot'] = False
    settings_description['quick_plot'] = 'Produce array of ``.png`` plots showing response. Requires matplotlib.'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description, settings_options)

    def __init__(self):

        self.settings = None
        self.data = None
        self.folder = None

        # self.ss = None
        # self.ssrom = None

        self.w_to_k = 1
        self.wv = None

    def initialise(self, data, custom_settings=None):

        self.data = data
        # self.ss = data.linear.linear_system.uvlm.ss

        if not custom_settings:
            self.settings = self.data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings_utils.to_custom_types(self.settings, self.settings_types, self.settings_default,
                                       self.settings_options,
                                       no_cytpe=True)

        try:
            scaling = self.data.linear.linear_system.uvlm.sys.ScalingFacts
            if self.settings['frequency_unit'] == 'k':
                self.w_to_k = scaling['length'] / scaling['speed']
            else:
                self.w_to_k = 1.
        except AttributeError:
            self.w_to_k = 1.

        lb = self.settings['frequency_bounds'][0] / self.w_to_k
        ub = self.settings['frequency_bounds'][1] / self.w_to_k

        nfreqs = self.settings['num_freqs']
        if self.settings['frequency_spacing'] == 'linear':
            self.wv = np.linspace(lb, ub, nfreqs)
        elif self.settings['frequency_spacing'] == 'log':
            self.wv = np.logspace(np.log10(lb), np.log10(ub), nfreqs)
        else:
            raise NotImplementedError('Unrecognised frequency spacing setting %s' % self.settings['frequency_spacing'])

        if not os.path.exists(self.settings['folder']):
            os.makedirs(self.settings['folder'])
        self.folder = self.settings['folder'] + '/' + self.data.settings['SHARPy']['case'] + '/frequencyresponse/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        # if self.settings['print_info']:
        #     cout.cout_wrap.cout_talk()
        # else:
        #     cout.cout_wrap.cout_quiet()

    def run(self, ss=None):
        """
        Get the frequency response of the linear state-space
        Returns:

        """

        if ss is None:
            ss_list = [self.find_target_system(system_name) for system_name in self.settings['target_system']]
        elif type(ss) is libss.ss:
            ss_list = [ss]
        elif type(ss) is list:
            ss_list = ss
        else:
            raise TypeError('ss input must be either a libss.ss instance or a list of libss.ss')

        for ith, system in enumerate(ss_list):
            cout.cout_wrap('Computing frequency response...')
            if ss is None:
                try:
                    system_name = self.settings['target_system'][ith]
                    cout.cout_wrap('\tComputing frequency response for %s system' % system_name, 1)
                except IndexError:
                    system_name = None
            else:
                system = None

            t0fom = time.time()
            y_freq_fom = system.freqresp(self.wv)
            tfom = time.time() - t0fom

            self.save_freq_resp(self.wv, y_freq_fom, system_name=system_name)

            cout.cout_wrap('\tComputed the frequency response in %f s' % tfom, 2)

            if self.settings['quick_plot']:
                self.quick_plot(y_freq_fom, subfolder=system_name)

            # >>>>>>>>>>>>>>>> Move to a dedicated ROM metric module
            # Y_freq_rom = None
            # compute_fom = False

            # if self.settings['load_fom'] != '':
            #     if os.path.exists(self.settings['load_fom']):
            #         try:
            #             Y_freq_fom = self.load_frequency_data()
            #         except OSError:
            #             compute_fom = True
            #     else:
            #         compute_fom = True
            #
            # if (self.settings['compute_fom'] and self.settings['load_fom'] == '') or compute_fom:
            #     if self.settings['print_info']:
            #         cout.cout_wrap('Computing frequency response...')
            #         cout.cout_wrap('Full order system:', 1)
            #     t0fom = time.time()
            #     Y_freq_fom = system.freqresp(self.wv)
            #     tfom = time.time() - t0fom
            #     self.save_freq_resp(self.wv, Y_freq_fom, 'fom')
            #     if self.settings['print_info']:
            #         cout.cout_wrap('\tComputed the frequency response of the full order system in %f s' % tfom, 2)
            #
            # if self.ssrom is not None:
            #     if self.settings['print_info']:
            #         cout.cout_wrap('Computing frequency response...')
            #         cout.cout_wrap('Reduced order system:', 1)
            #     t0rom = time.time()
            #     Y_freq_rom = self.ssrom.freqresp(self.wv)
            #     trom = time.time() - t0rom
            #     if self.settings['print_info']:
            #         cout.cout_wrap('\tComputed the frequency response of the reduced order system in %f s' % trom, 2)
            #     self.save_freq_resp(self.wv, Y_freq_rom, 'rom')
            #
            #     if Y_freq_fom is not None:
            #         frequency_error(Y_freq_fom, Y_freq_rom, self.wv)
            # <<<<<<<<<<<<<<<<<< Things to do with ROM comparison need to go elsewhere

        return self.data

    def find_target_system(self, target_system):

        if target_system == 'aeroelastic':
            ss = self.data.linear.ss

        elif target_system == 'structural':
            ss = self.data.linear.linear_system.beam.ss

        elif target_system == 'aerodynamic':
            ss = self.data.linear.linear_system.uvlm.ss  # this could be a ROM

        else:
            raise NameError('Unrecognised system')

        return ss

    def save_freq_resp(self, wv, Yfreq, filename, system_name=None):
        """
        Saves the frequency response to a binary ``.h5`` file.

        If the system has not been scaled, the units of frequency are ``rad/s`` and the response is given in complex
        form. The response is saved in a ``[p, m, n_freq_eval]`` format, where ``p`` corresponds to the system's
        outputs, ``n`` to the number of inputs and ``n_freq_eval`` to the number of frequency evaluations.

        Args:
            wv (np.ndarray): Frequency array.
            Y_freq (np.ndarray): Frequency response data ``[p, m, n_freq_eval]`` matrix.
            filename (str)
        """

        with open(self.folder + '/freqdata_readme.txt', 'w') as outfile:
            outfile.write('Frequency Response Data Output\n\n')
            outfile.write('Frequency data found in the relevant .h5 file\n')
            outfile.write('If the system has not been scaled, the units of frequency are rad/s\nThe frequency' \
                          'response is given in complex form.')

        out_folder = self.folder
        if system_name is not None:
            out_folder += '/' + system_name

        p, m, _ = Yfreq.shape

        h5filename = out_folder + '.freqresp.h5'
        with h5.File(h5filename, 'r') as f:
            f.create_dataset('frequency', data=wv)
            f.create_dataset('response', data=Yfreq, dtype=complex)
            f.create_dataset('inputs', data=m)
            f.create_dataset('outputs', data=p)
        cout.cout_wrap('Saved .h5 file to %s with frequency response data' % h5filename)

    def quick_plot(self, Y_freq_fom=None, subfolder=None):
        p, m, _ = Y_freq_fom.shape
        try:
            cout.cout_wrap('\tCreating Quick plots of the frequency response', 1)

            out_folder = self.folder
            if subfolder:
                out_folder += '/' + subfolder

            if not os.path.isdir(out_folder):
                os.makedirs(out_folder, exist_ok=True)

            import matplotlib.pyplot as plt
            for mj in range(m):
                for pj in range(p):
                    fig1, ax1 = plt.subplots(nrows=2)
                    fig_title = 'in%02g_out%02g' % (mj, pj)
                    ax1[0].set_title(fig_title)
                    if Y_freq_fom is not None:
                        ax1[0].plot(self.wv * self.w_to_k, 20 * np.log10(np.abs(Y_freq_fom[pj, mj, :])), color='C0')
                        ax1[1].plot(self.wv * self.w_to_k, np.angle(Y_freq_fom[pj, mj, :]), '-', color='C0')
                    if self.settings['frequency_unit'] == 'k':
                        ax1[1].set_xlabel('Reduced Frequency, k [-]')
                    else:
                        ax1[1].set_xlabel(r'Frequency, $\omega$ [rad/s]')

                    ax1[0].set_ylabel('Amplitude [dB]')
                    ax1[1].set_ylabel('Phase [rad]')
                    fig1.savefig(out_folder + '/' + fig_title + '.png')
                    plt.close()

            cout.cout_wrap('\tPlots saved to %s' % out_folder, 1)
        except ModuleNotFoundError:
            warnings.warn('Matplotlib not found - skipping plot')

    # >>>>>> Needs to go with ROM >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # def load_frequency_data(self):
    #     # TODO: need to change so that it doesn't require self.ss
    #     if self.settings['print_info']:
    #         cout.cout_wrap('Loading frequency response from:')
    #         cout.cout_wrap('\t%s' % self.settings['load_fom'], 1)
    #     Y_freq_fom = np.zeros((self.ss.outputs, self.ss.inputs, len(self.wv)), dtype=complex)
    #     for m in range(self.ss.inputs):
    #         for p in range(self.ss.outputs):
    #             y_load = np.loadtxt(self.settings['load_fom'] +
    #                                 '/Y_freq_fom_m%02g_p%02g.dat' %(m,p)).view(complex)
    #             y_load.shape = (y_load.shape[0], )
    #             Y_freq_fom[p, m, :] = y_load
    #
    #     return Y_freq_fom
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
