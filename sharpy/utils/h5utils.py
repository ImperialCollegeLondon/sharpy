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


def readh5(filename):
    '''
    Read the HDF5 file 'filename' into a class. Groups within the hdf5 file are 
    by default loaded as sub classes, unless they include a _read_as attribute 
    (see sharpy.postproc.savedata). In this case, group can be loaded as classes, 
    dictionaries, lists or tuples. 

    Warning:
    - groups that need to be read as lists and tuples are assumed to conform to 
    the format used in sharpy.postproc.savedata
    '''
    
    Hinst=ReadInto()
     
    ### read and scan file
    hdfile=h5.File(filename,'r')

    NamesList=[]                   # dataset names
    hdfile.visit(NamesList.append)
    
    ### Identify higher level groups / attributes
    MainLev=[]
    for name in NamesList:
        if '/' not in name: MainLev.append(name)

    ### Loop through higher level
    for name in MainLev:
        # sub-group
        if type(hdfile[name]) is h5._hl.group.Group:
            Ginst=read_group(hdfile[name])
            Ginst.name=name
            setattr(Hinst,name,Ginst)
        else:
            setattr(Hinst,name,hdfile[name].value)

    # close and return
    hdfile.close()  
        
    return Hinst  


def read_group(Grp):
    ''' Read an hdf5 group '''

    NamesList=[]
    Grp.visit(NamesList.append)

    ### identify higher level
    MainLev=[]
    for name in NamesList:
        if '/' not in name: MainLev.append(name)

    ### determine output format
    read_as='class'
    if '_read_as' in MainLev:
        read_as=Grp['_read_as'].value

    ### initialise output
    if read_as=='class':
        Hinst=ReadInto()
    elif read_as=='dict':
        Hinst={}
    elif read_as=='list' or read_as=='tuple':
        Hinst=[]

    ### Loop through higher level
    if read_as=='list' or read_as=='tuple':
        if '_as_array' in MainLev:       
            Hinst=list(Grp['_as_array'].value)
        else:
            N=len(MainLev)-1
            for nn in range(N):
                name='%.5d'%nn 
                ### extract value
                if type(Grp[name]) is h5._hl.group.Group:
                    value=read_group(Grp[name])
                else:
                    value=Grp[name].value   
                Hinst.append(value)           
        if read_as=='tuple': tuple(Hinst)
    else:
        for name in MainLev:
            if name=='_read_as': continue

            ### extract value
            if type(Grp[name]) is h5._hl.group.Group:
                value=read_group(Grp[name])
            else:
                value=Grp[name].value

            ### allocate
            if read_as=='class':
                setattr(Hinst,name,value)
            else:
                Hinst[name]=value

    return Hinst


class ReadInto:
    pass



if __name__=='__main__':

    from IPython import embed
    filename='../../tests/coupled/dynamic/coupled_configuration/output/coupled_configuration.data.h5'
    filename='/home/sm6110/git/sharpy/dev/test_savingh5/smith_Nsurf01M04N12wk10_a020.data.h5'
    filename='/home/sm6110/git/uvlm3d_studies/00_coupled_static/00_smith_g/test_of_saving_ss_matrices.h5'
    filename='/home/sm6110/git/uvlm3d_studies/00_coupled_static/00_smith_g/cases/a0080/smith_Nsurf02M04N20wk15_a0080.data.h5'  


    h=readh5(filename)
    embed()