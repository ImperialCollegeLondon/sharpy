# Alfonso del Carre
# alfonso.del-carre14@imperial.ac.uk
# Imperial College London
# LoCA lab
# 28 Sept 2016 

# Set of utilities for opening/reading files

import h5py as h5
import os
import errno
import numpy as np


def check_file_exists(file_name):
    '''
    Checks if the file exists and throws a FileNotFoundError exception
    that includes the route to the non-existing file.
    '''
    if not os.path.isfile(file_name):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT),
            file_name)


def load_h5_in_dict(handle, path='/'):
    dictionary = {}
    for k,i in handle[path].items():
        if isinstance(i, h5._hl.dataset.Dataset):
            dictionary[k] = i.value
        elif isinstance(i, h5._hl.group.Group):
            dictionary[k] = load_h5_in_dict(handle, path + k + '/')

        if len(i.attrs.items()):
            dictionary['Attributes'] = load_attributes(handle, path + k + '/')

    return dictionary


def load_attributes(handle, path):
    attributes = []
    for k, i in handle[path].attrs.items():
        attributes.append((k,i))

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


def check_aero_dict(aero_dict):
    pass
