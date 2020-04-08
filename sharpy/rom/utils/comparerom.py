# Work In Progress
# This is intended as a kind of postproc for the different state space objects that can call the rom classes in them
# to compare them
#
# Ideally it would allow the comparison of the ROM and FOM of the system for which the ROM was calculated but also
# systems modified downstream.
#
# For instance: a ROM is calculated for the UVLM, it should be capable of comparing the UVLM FOMs and ROMs using the
# already implemented SHARPy tools. In addition, the aeroelastic system of ROM vs FOM could be compared (where possible).
#
# Should also work for the structural system to compare modal vs nonmodal.
#
# Therefore, could include a method in the relevant linear ss class to retrieve FOMs and ROMs
#
# The class for this "linear_postproc" should be an attribute of the linear ss classes and called from each of the
# linear systems' settings.


# Stuff previously in the FrequencyResponse postprocessor

# >>>>>>>>>>>>>>>>>> MOVE TO ROM COMPARISON MODULE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# settings_types['compute_fom'] = 'bool'
# settings_default['compute_fom'] = False
# settings_description['compute_fom'] = 'Compute frequency response of full order model (use caution if large).'

# settings_types['load_fom'] = 'str'
# settings_default['load_fom'] = ''
# settings_description['load_fom'] = 'Folder to locate full order model frequency response data.'
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

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

class ROMAnalysis:
    pass
