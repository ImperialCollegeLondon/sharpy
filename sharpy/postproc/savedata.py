import os
import sharpy.utils.cout_utils as cout
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings

import h5py
from numpy import ndarray, float32, array, int32, int64
import ctypes as ct
from IPython import embed




@solver
class SaveData(BaseSolver):
    solver_id = 'SaveData'


    def __init__(self):
        import sharpy

        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['folder'] = 'str'
        self.settings_default['folder'] = './output'

        self.settings = None
        self.data = None

        self.folder = ''
        self.filename = ''
        self.ts_max = 0

        # specify which classes are saved as hdf5 group
        self.ClassesToSave=(sharpy.presharpy.presharpy.PreSharpy,
                            sharpy.aero.models.aerogrid.Aerogrid,
                            sharpy.structure.models.beam.Beam   )


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



    def run(self, online=False):

        hdfile=h5py.File(self.filename,'a')
        if online:
            self.ts=len(self.data.structure.timestep_info)-1
            add_as_grp(self.data,hdfile,grpname='data',
                                    ClassesToSave=self.ClassesToSave,ts=self.ts)
        else:
            add_as_grp(self.data,hdfile,grpname='data',
                                               ClassesToSave=self.ClassesToSave)
        hdfile.close()

        return self.data



def add_as_grp(obj,grpParent,grpname=None,ClassesToSave=(),ts=None,
                              compress=False,overwrite=False,save_ctypes=False):
    '''
    Given a class or dictionary instance 'obj', the routine adds it as a group 
    to a hdf5 file with name grpname. 
    Classes belonging to 'ClassesToSave' and *TimeStepInfo are also saved as 
    sub-groups. If ts is not None, only the current time-step is saved.
    
    Remarks: 
        - the previous content of the file is not deleted or modified. 
        - If group with the obj class name already exists:
            - the group will be fully overwritten if overwrite is True
            - new attributes of obj will be added to the grp but any 
            pre-existing attributes will not be overwritten.
    
    Warning: 
        - if compress is True, numpy arrays will be saved in single precisions.
    '''
  

    IsObjDict=isinstance(obj,dict)
    if grpname is None and (not IsObjDict):
        grpname=obj.__class__.__name__ 

    # check whether group with same name already exists
    if not(grpname in grpParent):
        grp=grpParent.create_group(grpname) 
    else:
        if overwrite:
            del grpParent[grpname]
            grp=grpParent.create_group(grpname) 
        else:
            grp=grpParent[grpname]

    # loop attributes
    if IsObjDict:
        dictname=obj
    else:
        dictname=obj.__dict__


    for attr in dictname:

        # ----- extract value & type
        if IsObjDict:
            value=obj[attr]
        else:
            value=getattr(obj,attr)
        vtype=type(value)
        if value is None:
            continue


        # ----- classes:
        # ps: no need to delete if overwrite is True
        if isinstance(value,ClassesToSave):
            add_as_grp(value,grp,attr,ClassesToSave,compress,overwrite)
            continue

        if attr=='timestep_info':
            if ts is None:
                for tt in range(len(value)):
                    add_as_grp(value[tt],grp,'tsinfo%.5d'%tt,
                                          ClassesToSave,None,compress,overwrite)
            else:
                add_as_grp(value[ts],grp,'tsinfo%.5d'%ts,
                                          ClassesToSave,None,compress,overwrite)               


        # ----- dictionaries
        if isinstance(value,dict):
            if attr=='airfoil_db':
                continue
            else:
                add_as_grp(value,grp,attr,ClassesToSave,compress,overwrite)
            continue


        # ----- if attr already in grp...
        if attr in grp:
            if overwrite:
                del grp[attr]
            else:
                continue


        # ----- Basic types
        if isinstance(value,(float,int,int32,int64,str,complex) ):
            grp[attr]=value 
            continue           

        # c_types
        if isinstance(value,(ct.c_bool,ct.c_double,ct.c_int)):
            value=value.value
            grp[attr]=value 
            continue

        # ndarrays
        if isinstance(value,ndarray):
            if compress is True:
                grp[attr]=float32(value)
            else:
                grp[attr]=value
            continue

        # ------ lists
        if vtype is list:
            #if any(isinstance(x, str) for x in value):
            if check_in_list(value,(str,)):
                value=array(value,dtype=object)
                string_dt = h5py.special_dtype(vlen=str)
                grp.create_dataset(attr, data=value, dtype=string_dt)         
            else:
                try:
                    # if all floats/integers, convert into float array
                    grp[attr]=value
                except TypeError:                    
                    grp[attr]='TypeError'
                except ValueError:
                    grp[attr]='ValueError'
                except:
                    grp[attr]='UnknownError' 
            continue
        grp[attr]='Type not identified!'

    return grpParent   



def check_in_list(List,TList):
    '''
    Given a tuple of types, TList, the function checks whether any object in 
    List, or any of its sub-lists, is a string.
    '''

    if isinstance(TList,list): 
        TList=tuple(TList)

    Found=False
    for x in List:
        # if x is a list subiterate
        if isinstance(x,list):
            Found=check_in_list(x,TList)
            if Found: 
                break
        # otherwise, check if x belongs to TList types
        if isinstance(x,TList):
            Found=True
            break

    return Found