"""Balancing Methods

The following classes are available to reduce a linear system employing balancing methods.

The main class is :class:`.Balanced` and the other available classes:

* :class:`.Direct`

* :class:`.Iterative`

* :class:`.FrequencyLimited`

correspond to the reduction algorithm.

"""
import sharpy.utils.settings as settings
import numpy as np
from abc import ABCMeta
import sharpy.utils.cout_utils as cout
import sharpy.utils.rom_interface as rom_interface
import sharpy.rom.utils.librom as librom
import sharpy.linear.src.libss as libss
import time
from sharpy.linear.utils.ss_interface import LinearVector, StateVariable

dict_of_balancing_roms = dict()

def bal_rom(arg):
    global dict_of_balancing_roms
    try:
        arg._bal_rom_id
    except AttributeError:
        raise AttributeError('Class defined as balanced rom has no _bal_rom_id attribute')
    dict_of_balancing_roms[arg._bal_rom_id] = arg
    return arg


class BaseBalancedRom(metaclass=ABCMeta):

    print_info = False

    def initialise(self, in_settings=None):
        pass

    def run(self, ss):
        pass


@bal_rom
class Direct(BaseBalancedRom):
    __doc__ = librom.balreal_direct_py.__doc__
    _bal_rom_id = 'Direct'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    settings_types['tune'] = 'bool'
    settings_default['tune'] = True
    settings_description['tune'] = 'Tune ROM to specified tolerance'

    settings_types['use_schur'] = 'bool'
    settings_default['use_schur'] = False
    settings_description['use_schur'] = 'Use Schur decomposition during build'

    settings_types['rom_tolerance'] = 'float'
    settings_default['rom_tolerance'] = 1e-2
    settings_description['rom_tolerance'] = 'Absolute accuracy with respect to full order frequency response'

    settings_types['rom_tune_freq_range'] = 'list(float)'
    settings_default['rom_tune_freq_range'] = [0, 1]
    settings_description['rom_tune_freq_range'] = 'Beginning and end of frequency range where to tune ROM'

    settings_types['convergence'] = 'str'
    settings_default['convergence'] = 'min'
    settings_description['convergence'] = 'ROM tuning convergence. If ``min`` attempts to find minimal number of states.' \
                                          'If ``all`` it starts from larger size ROM until convergence to ' \
                                          'specified tolerance is found.'

    settings_types['reduction_method'] = 'str'
    settings_default['reduction_method'] = 'realisation'
    settings_description['reduction_method'] = 'Desired reduction method'
    settings_options['reduction_method'] = ['realisation', 'truncation']

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description, settings_options)

    def __init__(self):
        self.settings = dict()

    def initialise(self, in_settings=None):
        if in_settings is not None:
            self.settings = in_settings

        settings.to_custom_types(self.settings, self.settings_types, self.settings_default, self.settings_options,
                                 no_ctype=True)

    def run(self, ss):
        if self.print_info:
            cout.cout_wrap('Reducing system using a Direct balancing method...')
        t0 = time.time()
        A, B, C, D = ss.get_mats()

        try:
            if ss.dt is not None:
                dtsystem = True
            else:
                dtsystem = False
        except AttributeError:
            dtsystem = False

        S, T, Tinv = librom.balreal_direct_py(A, B, C, DLTI=dtsystem, Schur=self.settings['use_schur'])

        Ar = T.dot(A.dot(Tinv))
        Br = T.dot(B)
        Cr = C.dot(Tinv)

        if dtsystem:
            ss_bal = libss.StateSpace(Ar, Br, Cr, D, dt=ss.dt)
        else:
            ss_bal = libss.StateSpace(Ar, Br, Cr, D)

        t1 = time.time()
        if self.print_info:
            cout.cout_wrap('\t...completed balancing in %.2fs' % (t1-t0), 1)

        if self.settings['tune']:
            cout.cout_wrap('Tuning ROM to specified tolerance...', 1)
            kv = np.linspace(self.settings['rom_tune_freq_range'][0],
                             self.settings['rom_tune_freq_range'][1])
            ssrom = librom.tune_rom(ss_bal,
                                    kv=kv,
                                    tol=self.settings['rom_tolerance'],
                                    gv=S,
                                    convergence=self.settings['convergence'],
                                    method=self.settings['reduction_method'])
            if librom.check_stability(ssrom.A, dt=True):
                if self.print_info:
                    cout.cout_wrap('ROM by direct balancing is stable')
            t2 = time.time()
            cout.cout_wrap('\t...completed reduction in %.2fs' % (t2-t0), 1)
            return ssrom
        else:
            return ss_bal


@bal_rom
class FrequencyLimited(BaseBalancedRom):
    __doc__ = librom.balfreq.__doc__

    _bal_rom_id = 'FrequencyLimited'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    settings_types['frequency'] = 'float'
    settings_default['frequency'] = 1.
    settings_description['frequency'] = 'defines limit frequencies for balancing. The balanced model will be accurate ' \
                                        'in the range ``[0,F]``, where ``F`` is the value of this key. Note that ``F`` ' \
                                        'units must be consistent with the units specified in the in ' \
                                        'the ``self.ScalingFacts`` dictionary.'

    settings_types['method_low'] = 'str'
    settings_default['method_low'] = 'trapz'
    settings_description['method_low'] = 'Specifies whether to use gauss quadrature or ' \
                                         'trapezoidal rule in the low-frequency range ``[0,F]``'
    settings_options['method_low'] = ['gauss', 'trapz']

    settings_types['options_low'] = 'dict'
    settings_default['options_low'] = dict()
    settings_description['options_low'] = 'Settings for the low frequency integration. See Notes.'

    settings_types['method_high'] = 'str'
    settings_default['method_high'] = 'trapz'
    settings_description['method_high'] = 'Specifies whether to use gauss quadrature or ' \
                                          'trapezoidal rule in the high-frequency range ``[F,FN]``'
    settings_options['method_high'] = ['gauss', 'trapz']

    settings_types['options_high'] = 'dict'
    settings_default['options_high'] = dict()
    settings_description['options_high'] = 'Settings for the high frequency integration. See Notes.'

    settings_types['check_stability'] = 'bool'
    settings_default['check_stability'] = True
    settings_description['check_stability'] = 'if True, the balanced model is truncated to eliminate ' \
                                              'unstable modes - if any is found. Note that very accurate ' \
                                              'balanced model can still be obtained, even if high order ' \
                                              'modes are unstable.'

    settings_types['get_frequency_response'] = 'bool'
    settings_default['get_frequency_response'] = False
    settings_description['get_frequency_response'] = 'if True, the function also returns the frequency ' \
                                                     'response evaluated at the low-frequency range integration' \
                                                     ' points. If True, this option also allows to automatically' \
                                                     ' tune the balanced model.'

    # Integrator options
    settings_options_types = dict()
    settings_options_default = dict()
    settings_options_description = dict()

    settings_options_types['points'] = 'int'
    settings_options_default['points'] = 12
    settings_options_description['points'] = 'Trapezoidal points of integration'

    settings_options_types['partitions'] = 'int'
    settings_options_default['partitions'] = 2
    settings_options_description['partitions'] = 'Number of Gauss-Lobotto quadratures'

    settings_options_types['order'] = 'int'
    settings_options_default['order'] = 2
    settings_options_description['order'] = 'Order of Gauss-Lobotto quadratures'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description, settings_options)

    options_table = settings.SettingsTable()
    __doc__ += options_table.generate(settings_options_types, settings_options_default, settings_options_description,
                                      header_line='The parameters of integration take the following options:\n')

    def __init__(self):
        self.settings = dict()

    def initialise(self, in_settings=None):

        if in_settings is not None:
            self.settings = in_settings

        settings.to_custom_types(self.settings, self.settings_types, self.settings_default,
                                 self.settings_options, no_ctype=True)
        settings.to_custom_types(self.settings['options_low'], self.settings_options_types,
                                 self.settings_options_default, no_ctype=True)
        settings.to_custom_types(self.settings['options_high'], self.settings_options_types,
                                 self.settings_options_default, no_ctype=True)

    def run(self, ss):

        output_results = librom.balfreq(ss, self.settings)

        return output_results[0]


@bal_rom
class Iterative(BaseBalancedRom):
    __doc__ = librom.balreal_iter.__doc__
    _bal_rom_id = 'Iterative'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['lowrank'] = 'bool'
    settings_default['lowrank'] = True
    settings_description['lowrank'] = 'Use low rank methods'

    settings_types['smith_tol'] = 'float'
    settings_default['smith_tol'] = 1e-10
    settings_description['smith_tol'] = 'Smith tolerance'

    settings_types['tolSVD'] = 'float'
    settings_default['tolSVD'] = 1e-6
    settings_description['tolSVD'] = 'SVD threshold'

    settings_types['tolSVD'] = 'float'
    settings_default['tolSVD'] = 1e-6
    settings_description['tolSVD'] = 'SVD threshold'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.settings = dict()

    def initialise(self, in_settings=None):
        if in_settings is not None:
            self.settings = in_settings

        settings.to_custom_types(self.settings, self.settings_types, self.settings_default,
                                 no_ctype=True)

    def run(self, ss):

        A, B, C, D = ss.get_mats()

        s, T, Tinv, rcmax, romax = librom.balreal_iter(A, B, C,
                                                       lowrank=self.settings['lowrank'],
                                                       tolSmith=self.settings['smith_tol'],
                                                       tolSVD=self.settings['tolSVD'])

        Ar = Tinv.dot(A.dot(T))
        Br = Tinv.dot(B)
        Cr = C.dot(T)

        ssrom = libss.StateSpace(Ar, Br, Cr, D, dt=ss.dt)
        return ssrom


@rom_interface.rom
class Balanced(rom_interface.BaseRom):
    """Balancing ROM methods

    Main class to load a balancing ROM. See below for the appropriate settings to be parsed in
    the ``algorithm_settings`` based on your selection.

    Supported algorithms:
        * Direct balancing :class:`.Direct`

        * Iterative balancing :class:`.Iterative`

        * Frequency limited balancing :class:`.FrequencyLimited`

    """
    rom_id = 'Balanced'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    settings_types['print_info'] = 'bool'
    settings_default['print_info'] = True
    settings_description['print_info'] = 'Write output to screen'

    settings_types['algorithm'] = 'str'
    settings_default['algorithm'] = ''
    settings_description['algorithm'] = 'Balanced realisation method'
    settings_options['algorithm'] = ['Direct', 'Iterative', 'FrequencyLimited']

    settings_types['algorithm_settings'] = 'dict'
    settings_default['algorithm_settings'] = dict()
    settings_description['algorithm_settings'] = 'Settings for the desired algorithm'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description, settings_options)

    def __init__(self):
        self.settings = dict()
        self.algorithm = None
        self.ssrom = None
        self.ss = None
        self.dtsystem = None

    def initialise(self, in_settings=None):

        if in_settings is not None:
            self.settings = in_settings

        settings.to_custom_types(self.settings, self.settings_types, self.settings_default, self.settings_options)

        if not (self.settings['algorithm'] in dict_of_balancing_roms):
            raise AttributeError('Balancing algorithm %s is not yet implemented' % self.settings['algorithm'])

        self.algorithm = dict_of_balancing_roms[self.settings['algorithm']]()
        self.algorithm.initialise(self.settings['algorithm_settings'])
        self.algorithm.print_info = self.settings['print_info']

    def run(self, ss):

        self.ss = ss

        A, B, C, D = self.ss.get_mats()

        if self.ss.dt:
            self.dtsystem = True
        else:
            self.dtsystem = False

        out = self.algorithm.run(ss)

        if type(out) == libss.StateSpace:
            self.ssrom = out

        else:
            Ar, Br, Cr = out
            if self.dtsystem:
                self.ssrom = libss.StateSpace(Ar, Br, Cr, D, dt=self.ss.dt)
            else:
                self.ssrom = libss.StateSpace(Ar, Br, Cr, D)

        try:
            self.ssrom.input_variables = self.ss.input_variables.copy()
            self.ssrom.output_variables = self.ss.output_variables.copy()
            self.ssrom.state_variables = LinearVector(
                [StateVariable('balanced_{:s}'.format(self.settings['algorithm'].lower()),
                               size=self.ssrom.states, index=0)])
        except AttributeError:
            pass

        return self.ssrom
