import os
import h5py
import copy
import sharpy
import sharpy.utils.cout_utils as cout
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings_utils
import sharpy.utils.h5utils as h5utils
from sharpy.presharpy.presharpy import PreSharpy


@solver
class SaveData(BaseSolver):
    """
    The ``SaveData`` postprocessor writes the SHARPy variables into ``hdf5`` files. The linear state space files
    may be saved to ``.mat`` if desired instead.

    It has options to save the following classes:

        * :class:`~sharpy.sharpy.aero.models.Aerogrid` including :class:`sharpy.sharpy.utils.datastructures.AeroTimeStepInfo`

        * :class:`~sharpy.sharpy.structure.beam.Beam` including :class:`sharpy.sharpy.utils.datastructures.StructTimeStepInfo`

        * :class:`sharpy.solvers.linearassembler.Linear` including classes in :exc:`sharpy.linear.assembler`

    Notes:
        This method saves simply the data. If you would like to preserve the SHARPy methods of the relevant classes
        see also :class:`sharpy.solvers.pickledata.PickleData`.

    """
    solver_id = 'SaveData'
    solver_classification = 'post-processor'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    settings_types['save_aero'] = 'bool'
    settings_default['save_aero'] = True
    settings_description['save_aero'] = 'Save aerodynamic classes.'

    settings_types['save_nonlifting'] = 'bool'
    settings_default['save_nonlifting'] = False
    settings_description['save_nonlifting'] = 'Save aerodynamic classes.'

    settings_types['save_struct'] = 'bool'
    settings_default['save_struct'] = True
    settings_description['save_struct'] = 'Save structural classes.'

    settings_types['save_linear'] = 'bool'
    settings_default['save_linear'] = False
    settings_description['save_linear'] = 'Save linear state space system.'

    settings_types['save_linear_uvlm'] = 'bool'
    settings_default['save_linear_uvlm'] = False
    settings_description['save_linear_uvlm'] = 'Save linear UVLM state space system. Use with caution when dealing with ' \
                                               'large systems.'

    settings_types['save_wake'] = 'bool'
    settings_default['save_wake'] = True
    settings_description['save_wake'] = 'Save aero wake classes.'

    settings_types['save_rom'] = 'bool'
    settings_default['save_rom'] = False
    settings_description['save_rom'] = 'Saves the ROM matrices and the reduced order model'

    settings_types['skip_attr'] = 'list(str)'
    settings_default['skip_attr'] = ['fortran',
                                     'airfoils',
                                     'airfoil_db',
                                     'settings_types',
                                     # 'beam',
                                     'ct_dynamic_forces_list',
                                     # 'ct_forces_list',
                                     'ct_gamma_dot_list',
                                     'ct_gamma_list',
                                     'ct_gamma_star_list',
                                     'ct_normals_list',
                                     'ct_u_ext_list',
                                     'ct_u_ext_star_list',
                                     'ct_zeta_dot_list',
                                     'ct_zeta_list',
                                     'ct_zeta_star_list',
                                     'dynamic_input']
    settings_description['skip_attr'] = 'List of attributes to skip when writing file'

    settings_types['compress_float'] = 'bool'
    settings_default['compress_float'] = False
    settings_description['compress_float'] = 'Compress float'

    settings_types['format'] = 'str'
    settings_default['format'] = 'h5'
    settings_description['format'] = 'Save linear state space to hdf5 ``.h5`` or Matlab ``.mat`` format.'
    settings_options['format'] = ['h5', 'mat']

    settings_types['stride'] = 'int'
    settings_default['stride'] = 1
    settings_description['stride'] = 'Number of steps between the execution calls when run online'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description,
                                       settings_options=settings_options)

    def __init__(self):

        self.settings = None
        self.data = None

        self.folder = ''
        self.filename = ''
        self.filename_linear = ''
        self.caller = None

        ### specify which classes are saved as hdf5 group
        # see initialise and add_as_grp
        self.ClassesToSave = (PreSharpy,)

    def initialise(self, data, custom_settings=None, caller=None, restart=False):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings

        settings_utils.to_custom_types(self.settings,
                           self.settings_types,
                           self.settings_default,
                           options=self.settings_options)

        # Add these anyway - therefore if you add your own skip_attr you don't have to retype all of these
        self.settings['skip_attr'].extend(['fortran',
                                            'airfoils',
                                            'airfoil_db',
                                            'settings_types',
                                            'ct_dynamic_forces_list',
                                            'ct_forces_list',
                                            'ct_gamma_dot_list',
                                            'ct_gamma_list',
                                            'ct_gamma_star_list',
                                            'ct_normals_list',
                                            'ct_u_ext_list',
                                            'ct_u_ext_star_list',
                                            'ct_zeta_dot_list',
                                            'ct_zeta_list',
                                            'ct_zeta_star_list',
                                            'dynamic_input'])


        # create folder for containing files if necessary
        self.folder = data.output_folder + '/savedata/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.filename = self.folder + self.data.settings['SHARPy']['case'] + '.data.h5'
        self.filename_linear = self.folder + self.data.settings['SHARPy']['case'] + '.linss.h5'

        # remove old file if it exists
        self.remove_file_if_exist(self.filename)

        # check that there is a linear system - else return setting to false
        if self.settings['save_linear'] or self.settings['save_linear_uvlm']:
            try:
                self.data.linear
            except AttributeError:
                cout.cout_wrap('SaveData variables ``save_linear`` and/or ``save_linear`` are True but no linear system'
                               'been found', 3)
                self.settings['save_linear'] = False
                self.settings['save_linear_uvlm'] = False

        if self.settings['format'] == 'h5':

            # allocate list of classes to be saved
            if self.settings['save_aero']:
                self.ClassesToSave += (sharpy.aero.models.aerogrid.Aerogrid,
                                       sharpy.utils.datastructures.AeroTimeStepInfo,)
                if not self.settings['save_wake']:
                    self.settings['skip_attr'].append('zeta_star')
                    self.settings['skip_attr'].append('u_ext_star')
                    self.settings['skip_attr'].append('gamma_star')
                    self.settings['skip_attr'].append('dist_to_orig')
                    self.settings['skip_attr'].append('wake_conv_vel')


            if self.settings['save_nonlifting']:
                self.ClassesToSave += (sharpy.aero.models.nonliftingbodygrid.NonliftingBodyGrid,
                                       sharpy.utils.datastructures.NonliftingBodyTimeStepInfo,)

            if self.settings['save_struct']:
                self.ClassesToSave += (
                    sharpy.structure.models.beam.Beam,
                    sharpy.utils.datastructures.StructTimeStepInfo,)

            if self.settings['save_linear']:
                self.ClassesToSave += (sharpy.solvers.linearassembler.Linear,
                                       sharpy.linear.assembler.linearaeroelastic.LinearAeroelastic,
                                       sharpy.linear.assembler.linearbeam.LinearBeam,
                                       sharpy.linear.assembler.linearuvlm.LinearUVLM,
                                       sharpy.linear.src.libss.StateSpace,
                                       sharpy.linear.src.lingebm.FlexDynamic,)

            if self.settings['save_linear_uvlm']:
                self.ClassesToSave += (sharpy.solvers.linearassembler.Linear, sharpy.linear.src.libss.ss_block)
        self.caller = caller

    def remove_file_if_exist(self, filepath):
        if os.path.isfile(filepath):
            os.remove(filepath)

    def run(self, **kwargs):

        online = settings_utils.set_value_or_default(kwargs, 'online', False)

        # Use the following statement in case the ct types are not defined and
        # you need them on uvlm3d
        # self.data.aero.timestep_info[-1].generate_ctypes_pointers()

        if ((online and (self.data.ts % self.settings['stride'] == 0)) or (not online)):
            if self.settings['format'] == 'h5':
                file_exists = os.path.isfile(self.filename)
                hdfile = h5py.File(self.filename, 'a')

                if (online and file_exists):
                    self.save_timestep(self.data, self.settings, self.data.ts, hdfile)
                else:
                    skip_attr_init = copy.deepcopy(self.settings['skip_attr'])
                    skip_attr_init.append('timestep_info')

                    h5utils.add_as_grp(self.data, hdfile, grpname='data',
                                       ClassesToSave=self.ClassesToSave, SkipAttr=skip_attr_init,
                                       compress_float=self.settings['compress_float'])

                    if self.settings['save_struct']:
                        h5utils.add_as_grp(list(),
                                hdfile['data']['structure'],
                                grpname='timestep_info')
                    if self.settings['save_aero']:
                        h5utils.add_as_grp(list(),
                                hdfile['data']['aero'],
                                grpname='timestep_info')
                    if self.settings['save_nonlifting']:
                        h5utils.add_as_grp(list(),
                                hdfile['data']['nonlifting_body'],
                                grpname='timestep_info')

                    for it in range(len(self.data.structure.timestep_info)):
                        tstep_p = self.data.structure.timestep_info[it]
                        if tstep_p is not None:
                            self.save_timestep(self.data, self.settings, it, hdfile)

                    hdfile.close()

                if self.settings['save_linear_uvlm']:
                    linhdffile = h5py.File(self.filename.replace('.data.h5', '.uvlmss.h5'), 'a')
                    self.remove_file_if_exist(linhdffile)
                    h5utils.add_as_grp(self.data.linear.linear_system.uvlm.ss, linhdffile, grpname='ss',
                                    ClassesToSave=self.ClassesToSave, SkipAttr=self.settings['skip_attr'],
                                    compress_float=self.settings['compress_float'])
                    h5utils.add_as_grp(self.data.linear.linear_system.linearisation_vectors, linhdffile,
                                    grpname='linearisation_vectors',
                                    ClassesToSave=self.ClassesToSave, SkipAttr=self.settings['skip_attr'],
                                    compress_float=self.settings['compress_float'])
                    linhdffile.close()

                if self.settings['save_linear']:
                    self.remove_file_if_exist(self.filename_linear)
                    with h5py.File(self.filename_linear, 'a') as linfile:
                        h5utils.add_as_grp(self.data.linear.linear_system.linearisation_vectors, linfile,
                                        grpname='linearisation_vectors',
                                        ClassesToSave=self.ClassesToSave, SkipAttr=self.settings['skip_attr'],
                                        compress_float=self.settings['compress_float'])
                        h5utils.add_as_grp(self.data.linear.ss, linfile, grpname='ss',
                                        ClassesToSave=self.ClassesToSave, SkipAttr=self.settings['skip_attr'],
                                        compress_float=self.settings['compress_float'])

                if self.settings['save_rom']:
                    try:
                        for k, rom in self.data.linear.linear_system.uvlm.rom.items():
                            romhdffile = self.filename.replace('.data.h5', '_{:s}.rom.h5'.format(k.lower()))
                            self.remove_file_if_exist(romhdffile)
                            rom.save(romhdffile)
                    except AttributeError:
                        cout.cout_wrap('Could not locate a reduced order model to save')

            elif self.settings['format'] == 'mat':
                from scipy.io import savemat
                if self.settings['save_linear']:
                    # reference-forces
                    linearisation_vectors = self.data.linear.linear_system.linearisation_vectors

                    matfilename = self.filename.replace('.data.h5', '.linss.mat')
                    A, B, C, D = self.data.linear.ss.get_mats()
                    savedict = {'A': A,
                                'B': B,
                                'C': C,
                                'D': D}
                    for k, v in linearisation_vectors.items():
                        savedict[k] = v
                    dt = self.data.linear.ss.dt
                    if dt is not None:
                        savedict['dt'] = dt
                    savemat(matfilename, savedict)

                if self.settings['save_linear_uvlm']:
                    matfilename = self.filename.replace('.data.h5', '.uvlmss.mat')
                    linearisation_vectors = self.data.linear.linear_system.uvlm.linearisation_vectors
                    A, B, C, D = self.data.linear.linear_system.uvlm.ss.get_mats()
                    savedict = {'A': A,
                                'B': B,
                                'C': C,
                                'D': D}
                    for k, v in linearisation_vectors.items():
                        savedict[k] = v
                    dt = self.data.linear.linear_system.uvlm.ss.dt
                    if dt is not None:
                        savedict['dt'] = dt
                    savemat(matfilename, savedict)

        return self.data

    @staticmethod
    def save_timestep(data, settings, ts, hdfile):
        if settings['save_aero']:
            h5utils.add_as_grp(data.aero.timestep_info[ts],
                               hdfile['data']['aero']['timestep_info'],
                               grpname=("%05d" % ts),
                               ClassesToSave=(sharpy.utils.datastructures.AeroTimeStepInfo,),
                               SkipAttr=settings['skip_attr'],
                               compress_float=settings['compress_float'])
        if settings['save_nonlifting']:
            h5utils.add_as_grp(data.nonlifting_body.timestep_info[ts],
                               hdfile['data']['nonlifting_body']['timestep_info'],
                               grpname=("%05d" % ts),
                               ClassesToSave=(sharpy.utils.datastructures.NonliftingBodyTimeStepInfo,),
                               SkipAttr=settings['skip_attr'],
                               compress_float=settings['compress_float'])
        if settings['save_struct']:
            tstep = data.structure.timestep_info[ts]

            h5utils.add_as_grp(tstep,
                               hdfile['data']['structure']['timestep_info'],
                               grpname=("%05d" % ts),
                               ClassesToSave=(sharpy.utils.datastructures.StructTimeStepInfo,),
                               SkipAttr=settings['skip_attr'],
                               compress_float=settings['compress_float'])
