import os
import sharpy
import sharpy.utils.cout_utils as cout
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings

import h5py
from numpy import ndarray, float64, float32, array, int32, int64
import ctypes as ct
from IPython import embed


# Define basic numerical types
BasicNumTypes=(float,float32,float64,int,int32,int64,complex)
SkipAttr=[  'airfoil_db',
            'settings_types',
            'beam',
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
            'ct_zeta_star_list',]


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

        hdfile=h5py.File(self.filename,'a')

        #from IPython import embed;embed()

        if online:
            raise NameError('online not implemented!')
            # self.ts=len(self.data.structure.timestep_info)-1
            # add_as_grp(self.data,hdfile,grpname='data',
            #                         ClassesToSave=self.ClassesToSave,ts=self.ts)
        else:
            add_as_grp(self.data,hdfile,grpname='data',
                                ClassesToSave=self.ClassesToSave,
                                            compress_float=self.settings['compress_float'] )
        hdfile.close()

        return self.data



def add_as_grp(obj,grpParent,
                    grpname=None, ClassesToSave=(), SkipAttr=SkipAttr,
                                                compress_float=False,overwrite=False):

    '''
    Given a class, dictionary, list or tuples instance 'obj', the routine adds 
    it as a sub-group of name grpname to the parent group grpParent. An attribute
    _read_as, specifying the type of obj, is added to the group so as to allow
    reading correctly the h5 file.

    Usage and Remarks:
    - if obj contains dictionaries, listes or tuples, these are automatically 
    saved

    - if list only contains scalars or arrays of the same dimension, this will
    be saved as a numpy array

    - if obj contains classes, only those that are instances of the classes
    specified in ClassesToSave will be saved
    - If grpParent already contains a sub-group with name grpname, this will not
    be overwritten. However, pre-existing attributes of the sub-group will be 
    overwritten if obj contains attrributes with the same names.

    - attributes belonging to SkipAttr will not be saved - This functionality 
    needs improving  
    - if compress_float is True, numpy arrays will be saved in single precisions.
    '''


    ### determine if dict, list, tuple or class
    if isinstance(obj,list):
        ObjType='list'
    elif isinstance(obj,tuple):
        ObjType='tuple'
    elif isinstance(obj,dict):
        ObjType='dict'
    elif hasattr(obj,'__class__'):
        ObjType='class'
    else:
        raise NameError('object type not supported')


    ### determine sub-group name (only classes)
    if grpname is None:
        if ObjType=='class':
            grpname=obj.__class__.__name__ 
        else:
            raise NameError('grpname must be specified for dict,list and tuples')


    ### Create group (if necessary)
    if not(grpname in grpParent):
        grp=grpParent.create_group(grpname)
        grp['_read_as']=ObjType
    else:
        if overwrite:
            del grpParent[grpname]
            grp=grpParent.create_group(grpname) 
            grp['_read_as']=ObjType
        else:
            grp=grpParent[grpname]
            assert grp['_read_as'].value==ObjType,\
                                     'Can not overwrite group of different type'


    ### lists/tuples only
    # if possible, save as arrays
    if ObjType in ('list','tuple'):
        N=len(obj)
        if N>0:
            type0=type(obj[0])
            if type0 in (BasicNumTypes+(str,)):
                SaveAsArray=True
                for nn in range(N):
                    if type(obj[nn])!=type0:
                        SaveAsArray=False
                        break
                if SaveAsArray:
                    if '_as_array' in grp: 
                        del grp['_as_array']
                    if type0 in BasicNumTypes:
                        if type0==float and compress_float:
                            grp['_as_array']=float32(obj)
                        else:
                            grp['_as_array']=obj
                    else:
                        string_dt = h5py.special_dtype(vlen=str)
                        grp.create_dataset('_as_array',
                                   data=array(obj,dtype=object),dtype=string_dt)
                    return grpParent


    ### create/retrieve dictionary of attributes/elements to be saved
    if ObjType=='dict':
        dictname=obj
    elif ObjType=='class':
        dictname=obj.__dict__
    else: 
        N=len(obj)
        dictname={}
        for nn in range(N):
            dictname['%.5d'%nn ]=obj[nn]


    ### loop attributes and save
    SaveAsGroups=ClassesToSave+(list,dict,tuple, )


    for attr in dictname:
        if attr in SkipAttr: continue

        # ----- extract value & type
        value=dictname[attr]
        vtype=type(value)

        # ----- classes/dict/lists
        # ps: no need to delete if overwrite is True
        if isinstance(value,SaveAsGroups):
            add_as_grp(value,grp,attr,
                                ClassesToSave,SkipAttr,compress_float,overwrite)
            continue

        # ----- if attr already in grp always overwrite
        if attr in grp:
            del grp[attr]

        # ----- Basic types
        if isinstance(value, BasicNumTypes+(str,bytes) ):
            grp[attr]=value 
            continue           

        # c_types
        if isinstance(value,(ct.c_bool,ct.c_double,ct.c_int)):
            value=value.value
            grp[attr]=value 
            continue

        # ndarrays
        if isinstance(value,ndarray):
            add_array_to_grp(value,attr,grp,compress_float)
            continue

        # ----- Special
        if value==None:
            grp[attr]='NoneType'
            continue

        grp[attr]='not saved'

    return grpParent



def add_array_to_grp(data,name,grp,compress_float=False):
    ''' Add numpy array (data) as dataset 'name' to the group grp. If 
    compress is True, 64-bit float arrays are converted to 32-bit '''


    if compress_float and data.dtype==float64:
        #embed()
        grp.create_dataset(name,data=data,dtype='f4')
    else:
        grp[name]=data

    return grp
