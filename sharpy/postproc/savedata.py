import os
import h5py

import sharpy
import sharpy.utils.cout_utils as cout
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.utils.h5utils as h5utils


# Define basic numerical types
# BasicNumTypes=(float,float32,float64,int,int32,int64,complex)
SkipAttr=[  'fortran',
            'airfoils',
            'airfoil_db',
            'settings_types',
            'beam',
            'ct_dynamic_forces_list',
            #'ct_forces_list',
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


@solver
class SaveData(BaseSolver):
    solver_id = 'SaveData'


    def __init__(self):
        import sharpy

        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['folder'] = 'str'
        self.settings_default['folder'] = './output'

        self.settings_types['save_aero'] = 'bool'
        self.settings_default['save_aero'] = True

        self.settings_types['save_struct'] = 'bool'
        self.settings_default['save_struct'] = True

        self.settings_types['skip_attr'] = 'list(str)'
        self.settings_default['skip_attr'] = SkipAttr

        self.settings_types['compress_float'] = 'bool'
        self.settings_default['compress_float'] = False

        self.settings = None
        self.data = None

        self.folder = ''
        self.filename = ''
        self.ts_max = 0

        ### specify which classes are saved as hdf5 group
        # see initialise and add_as_grp
        self.ClassesToSave=(sharpy.presharpy.presharpy.PreSharpy,)


    def initialise(self, data, custom_settings=None):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings.to_custom_types(self.settings,
                                     self.settings_types, self.settings_default)
        self.ts_max = self.data.ts + 1

        # create folder for containing files if necessary
        if not os.path.exists(self.settings['folder']):
            os.makedirs(self.settings['folder'])
        self.folder = self.settings['folder'] + '/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.filename=self.folder+self.data.settings['SHARPy']['case']+'.data.h5'

        # allocate list of classes to be saved
        if self.settings['save_aero']:
            self.ClassesToSave+=(sharpy.aero.models.aerogrid.Aerogrid,
                                 sharpy.utils.datastructures.AeroTimeStepInfo,)

        if self.settings['save_struct']:
            self.ClassesToSave+=(
                                sharpy.structure.models.beam.Beam,
                                sharpy.utils.datastructures.StructTimeStepInfo,)


    def run(self, online=False):

        # Use the following statement in case the ct types are not defined and
        # you need them on uvlm3d
        # self.data.aero.timestep_info[-1].generate_ctypes_pointers()

        hdfile=h5py.File(self.filename,'a')

        #from IPython import embed;embed()

        if online:
            raise NameError('online not implemented!')
            # self.ts=len(self.data.structure.timestep_info)-1
            # add_as_grp(self.data,hdfile,grpname='data',
            #                         ClassesToSave=self.ClassesToSave,ts=self.ts)
        else:
            h5utils.add_as_grp(self.data,hdfile,grpname='data',
                               ClassesToSave=self.ClassesToSave,SkipAttr=self.settings['skip_attr'],
                               compress_float=self.settings['compress_float'])
        hdfile.close()

        return self.data


