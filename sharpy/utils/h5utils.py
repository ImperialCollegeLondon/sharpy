# Alfonso del Carre
# alfonso.del-carre14@imperial.ac.uk
# Imperial College London
# LoCA lab
# 28 Sept 2016
"""H5 File Management Utilities
Set of utilities for opening/reading files
"""
import h5py as h5
import os
import errno

import numpy as np
import warnings
from numpy import ndarray, float64, float32, array, int32, int64
import ctypes as ct

BasicNumTypes = (float, float32, float64, int, int32, int64, complex)


def check_file_exists(file_name):
    """
    Checks if the file exists and throws a FileNotFoundError exception
    that includes the route to the non-existing file.

    Args:
        file_name (str): path to the HDF5 file

    Returns:
        FileNotFoundError : if the file does not exist, an error is raised with path to the non-existent file
    """
    if not os.path.isfile(file_name):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT),
            file_name)


def load_h5_in_dict(handle, path='/'):
    dictionary = {}
    for k, i in handle[path].items():
        if isinstance(i, h5._hl.dataset.Dataset):
            dictionary[k] = i[()]
        elif isinstance(i, h5._hl.group.Group):
            dictionary[k] = load_h5_in_dict(handle, path + k + '/')

        if len(i.attrs.items()):
            dictionary['Attributes'] = load_attributes(handle, path + k + '/')

    return dictionary


def load_attributes(handle, path):
    attributes = []
    for k, i in handle[path].attrs.items():
        attributes.append((k, i))

    return attributes


def check_fem_dict(fem_dict):
    print('\tRunning tests for the FEM input file...', end='')
    # import pdb; pdb.set_trace()
    (num_elem_dict, num_node_elem_dict) = np.shape(fem_dict['connectivities'])
    num_elem = fem_dict['num_elem']
    num_node_elem = fem_dict['num_node_elem']
    if not ((num_elem == num_elem_dict) or (num_node_elem == num_node_elem_dict)):
        raise Exception('ERROR: FEM input file is not consistent')
    else:
        print(' PASSED')


def check_data_dict(data_dict):
    pass


# --------------------------------------------------------------- Reading tools


def readh5(filename, GroupName=None):
    """
    Read the HDF5 file 'filename' into a class. Groups within the hdf5 file are
    by default loaded as sub classes, unless they include a _read_as attribute
    (see sharpy.postproc.savedata). In this case, group can be loaded as classes,
    dictionaries, lists or tuples.

    filename: string to file location

    GroupName = string or list of strings. Default is None: if given, allows
    reading a specific group h5 file.

    Warning:
        Groups that need to be read as lists and tuples are assumed to conform to
        the format used in sharpy.postproc.savedata
    """

    Hinst = ReadInto()

    ### read and scan file
    hdfile = h5.File(filename, 'r')

    NamesList = []  # dataset names
    hdfile.visit(NamesList.append)

    ### Identify higher level groups / attributes
    if GroupName is None:
        MainLev = []
        for name in NamesList:
            if '/' not in name: MainLev.append(name)
    else:
        if type(GroupName) is list:
            MainLev = GroupName
        else:
            MainLev = [GroupName]

    ### Loop through higher level
    for name in MainLev:
        # sub-group
        if type(hdfile[name]) is h5._hl.group.Group:
            Ginst = read_group(hdfile[name])
            try:
                Ginst.name = name
            except:
                pass
            setattr(Hinst, name, Ginst)
        else:
            setattr(Hinst, name, hdfile[name][()])

    # close and return
    hdfile.close()

    return Hinst


def read_group(Grp):
    """ Read an hdf5 group """

    NamesList = []
    Grp.visit(NamesList.append)

    ### identify higher level
    MainLev = []
    for name in NamesList:
        if '/' not in name: MainLev.append(name)

    ### determine output format
    read_as = 'class'
    if '_read_as' in MainLev:
        read_as = Grp['_read_as'][()]

    ### initialise output
    if read_as == 'class':
        Hinst = ReadInto()
    elif read_as == 'dict':
        Hinst = {}
    elif read_as == 'list' or read_as == 'tuple':
        Hinst = []

    ### Loop through higher level
    if read_as == 'list' or read_as == 'tuple':
        if '_as_array' in MainLev:
            Hinst = list(Grp['_as_array'][()])
        else:
            N = len(MainLev) - 1
            list_ts = MainLev.copy()
            list_ts.remove('_read_as')
            list_ts = np.sort(np.unique(np.array(list_ts, dtype=np.int)))
            if len(list_ts > 0):
                for nn in range(list_ts[0] - 1):
                    Hinst.append('NoneType')
            for nn in list_ts:
                name = '%.5d' % nn
                ### extract value
                if type(Grp[name]) is h5._hl.group.Group:
                    value = read_group(Grp[name])
                else:
                    value = Grp[name][()]
                Hinst.append(value)
        if read_as == 'tuple': tuple(Hinst)
    else:
        for name in MainLev:
            if name == '_read_as': continue

            ### extract value
            if type(Grp[name]) is h5._hl.group.Group:
                value = read_group(Grp[name])
            else:
                value = Grp[name][()]

            ### allocate
            if read_as == 'class':
                setattr(Hinst, name, value)
            else:
                Hinst[name] = value

    return Hinst


class ReadInto:
    def __init__(self, name='ReadInto'):
        self._name = name

    pass


# ---------------------------------------------------------------- Saving tools


def saveh5(savedir, h5filename, *class_inst, permission='a', ClassesToSave=()):
    """
    Creates h5filename and saves all the classes specified in class_inst

    Args
        savedir: target directory
        h5filename: file name
        class_inst: a number of classes to save
        permission=['a','w']: append or overwrite, according to h5py.File
        ClassesToSave: if the classes in class_inst contain sub-classes, these will be saved only if instances of the classes in this list
    """

    h5filename = os.path.join(savedir, h5filename)
    hdfile = h5.File(h5filename, permission)

    for cc in class_inst:
        add_as_grp(cc, hdfile, ClassesToSave=ClassesToSave)

    hdfile.close()
    return None


def add_as_grp(obj, grpParent,
               grpname=None, ClassesToSave=(), SkipAttr=[],
               compress_float=False, overwrite=False):
    """
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
    """

    ### determine if dict, list, tuple or class
    if isinstance(obj, list):
        ObjType = 'list'
    elif isinstance(obj, tuple):
        ObjType = 'tuple'
    elif isinstance(obj, dict):
        ObjType = 'dict'
    elif hasattr(obj, '__class__'):
        ObjType = 'class'
    else:
        raise NameError('object type not supported')

    ### determine sub-group name (only classes)
    if grpname is None:
        if ObjType == 'class':
            if hasattr(obj, '_name'):
                grpname = obj._name
            else:
                grpname = obj.__class__.__name__
        else:
            raise NameError('grpname must be specified for dict,list and tuples')

    ### Create group (if necessary)
    if not (grpname in grpParent):
        grp = grpParent.create_group(grpname)
        grp['_read_as'] = ObjType
    else:
        if overwrite:
            del grpParent[grpname]
            grp = grpParent.create_group(grpname)
            grp['_read_as'] = ObjType
        else:
            grp = grpParent[grpname]
            assert grp['_read_as'][()] == ObjType, \
                'Can not overwrite group of different type'

    ### lists/tuples only: try to save as arrays
    if ObjType in ('list', 'tuple'):
        Success = save_list_as_array(
            list_obj=obj, grp_target=grp, compress_float=compress_float)
        if Success:
            return grpParent

    ### create/retrieve dictionary of attributes/elements to be saved
    if ObjType == 'dict':
        dictname = obj
    elif ObjType == 'class':
        dictname = obj.__dict__
    else:
        N = len(obj)
        dictname = {}
        for nn in range(N):
            dictname['%.5d' % nn] = obj[nn]

    ### loop attributes and save
    SaveAsGroups = ClassesToSave + (list, dict, tuple,)

    for attr in dictname:
        if attr in SkipAttr: continue

        # ----- extract value & type
        value = dictname[attr]
        vtype = type(value)

        # ----- classes/dict/lists
        # ps: no need to delete if overwrite is True
        if isinstance(value, SaveAsGroups):
            add_as_grp(value, grp, attr,
                       ClassesToSave, SkipAttr, compress_float, overwrite)
            continue

        # ----- if attr already in grp always overwrite
        if attr in grp:
            del grp[attr]

        # ----- Basic types
        if isinstance(value, BasicNumTypes + (str, bytes)):
            grp[attr] = value
            continue

        # c_types
        if isinstance(value, (ct.c_bool, ct.c_double, ct.c_int)):
            value = value.value
            grp[attr] = value
            continue

        # ndarrays
        if isinstance(value, ndarray):
            add_array_to_grp(value, attr, grp, compress_float)
            continue

        # ----- Special
        if value == None:
            grp[attr] = 'NoneType'
            continue

        grp[attr] = 'not saved'

    return grpParent


def add_array_to_grp(data, name, grp, compress_float=False):
    """ Add numpy array (data) as dataset 'name' to the group grp. If
    compress is True, 64-bit float arrays are converted to 32-bit """

    if compress_float and data.dtype == float64:
        # embed()
        grp.create_dataset(name, data=data, dtype='f4')
    else:
        grp[name] = data

    return grp


def save_list_as_array(list_obj, grp_target, compress_float=False):
    """
    Works for both lists and tuples. Returns True if the saving was successful.
    """

    N = len(list_obj)

    if N > 0:
        type0 = type(list_obj[0])
        if type0 in (BasicNumTypes + (str, ndarray)):
            SaveAsArray = True
            for nn in range(N):
                if type(list_obj[nn]) != type0:
                    SaveAsArray = False
                    break
                if type(list_obj[nn]) == ndarray:
                    if list_obj[0].shape != list_obj[nn].shape:
                        SaveAsArray = False
                        break
            if SaveAsArray:
                if '_as_array' in grp_target:
                    del grp_target['_as_array']
                if type0 in BasicNumTypes:  # list of scalars
                    if type0 == float and compress_float:
                        grp_target['_as_array'] = float32(list_obj)
                    else:
                        grp_target['_as_array'] = list_obj
                elif type0 == str:  # list of strings
                    string_dt = h5.special_dtype(vlen=str)
                    grp_target.create_dataset('_as_array',
                                              data=array(list_obj, dtype=object), dtype=string_dt)

                elif type0 == ndarray:  # list of arrays
                    if list_obj[0].dtype in BasicNumTypes:
                        if list_obj[0].dtype == float64 and compress_float:
                            grp_target['_as_array'] = float32(list_obj)
                        else:
                            grp_target['_as_array'] = list_obj
                    else:
                        string_dt = h5.special_dtype(vlen=str)
                        grp_target.create_dataset('_as_array',
                                                  data=array(list_obj, dtype=object), dtype=string_dt)
                else:
                    warnings.warn(
                        '%s could not be saved as an array' % (grp_target.name,))
                    return False

                return True
    return False
