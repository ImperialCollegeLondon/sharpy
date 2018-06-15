'''
Created on 21 Sep 2015

@author: sm6110
'''

import h5py
import os
from numpy import ndarray, float32, array, int32, int64
from IPython import embed



def save_aero(savedir,h5filename,data):
    '''
    Saves state of UVLM steady solution to h5 file.
    '''

    ts_max=len(data.aero.timestep_info)
    # if ts_max>1:
    #   raise NameError('Not saving a steady solution!')

    ### other params
    rho=data.settings['StaticUvlm']['rho'].value

    for tt in range(ts_max):
        #tsinfo.name=
        tsinfo=data.aero.timestep_info[0]
        tsinfo.name='ts%.5d'%tt
        tsinfo.rho=rho

    h5file(savedir,h5filename,*tuple(data.aero.timestep_info))



def h5file(savedir,h5filename, *class_inst):
    '''
    Creates h5filename and saves all the classes specified after the
    first input argument
    
    @param savedir: target directory
    @param h5filename: file name
    @param *class_inst: a number of classes to save
    '''

    os.system('mkdir -p %s'%savedir)
    h5filename=os.path.join(savedir,h5filename)
    
    hdfile=h5py.File(h5filename,'a')

    for cc in class_inst:
        add_class_as_grp(cc,hdfile)

    hdfile.close()
    return None



def add_class_as_grp(obj,grpParent,compress=False,overwrite=False):
    '''
    Given a class instance 'obj', the routine adds it as a group to a hdf5 file
    
    Remarks: 
        - the previous content of the file is not deleted or modified. 
        - If group with the obj class name already exists:
            - the group will be fully overwritten if overwrite is True
            - otherwise, the new attributes of obj will be added to the grp
              any pre-existing attributes will not be overwritten.
    
    Warning: 
        - multiple calls of this function may lead to a file increase
        - if compress is True, numpy arrays will be saved in single precisions.

    '''
    
    # look for a name, otherwise use class name
    if hasattr(obj,'name'): 
        grpname=obj.name   
    else: 
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

    # identify items to save
    dictname=obj.__dict__

    for attr in dictname:

        # check if attr already in grp and decide what to do
        if attr in grp:
            continue
            ### full group already deleted if overwrite
            # if overwrite:
            #     del grp[attr]
            # else:
            #     continue

        value=getattr(obj,attr)
        vtype=type(value)

        if value is None:
            continue

        if isinstance(value,(float,int,int32,int64,str,complex) ):
            grp[attr]=value 
            continue           

        # Add Output class as subgroup
        if isinstance(value,Output):
            #print('Detected class %s' %value.name)
            add_class_as_grp(value,grp,compress=compress)
            continue

        # Add c_types
        if vtype.__name__[:2]=='c_': 
            value=value.value
            continue

        # ndarrays
        #if vtype is ndarray:
        if isinstance(value,ndarray):
            #print('%s Array detected!!!!'%attr)
            if compress is True:
                grp[attr]=float32(value)
            else:
                grp[attr]=value
            continue

        # lists
        if vtype is list:
            #if any(isinstance(x, str) for x in value):
            if check_in_list(value,(str,)):
                #print('%s is a list with at least 1 string!!!' %attr)
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
                    #print ('unknown error occurred when saving %s' %attr)

                    #print ('TypeError occurred when saving %s' %attr)
                    #value=array(value,dtype=object)
                    # embed()
                    # var_len_dt=h5py.special_dtype(vlen=np.dtype('float64'))
                    # value=np.array(value,dtype=object)
                    # grp.create_dataset(attr,data=value,dtype=var_len_dt)   
                    grp[attr]='ValueError'
                except:
                    grp[attr]='UnknownError' 
            continue

        grp[attr]='Type not identified!'
        #embed()


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



class Output:
    '''
    Class to store output
    '''
    
    def __init__(self,name=None):
        self.name=name
        
    def drop(self, **kwargs):
        '''Attach random variables to this class'''
        for ww in kwargs:
            setattr(self, ww, kwargs[ww])
        
        return self



if __name__=='__main__':
    

    import numpy as np

    Set=Output('mainset')
    Set.Sub=Output('subset')

    Set.drop(
        float=3.4, integer=1, string='something', complex=2.+3.j,
        array_real=np.random.rand(3,2,4),
        array_complex=np.random.rand(2,3,4)+1.j*np.random.rand(2,3,4),
        liststr=['a','list','of','strings'],
        listnums=[1,2,3,4.],
        listcomplex=[1.,3,2.+3.j],
        listmix=[2.,4,'lala',3.+1.j],
        listmix_sub=[2.,3,['lala',2],3.j],
        listmat=[[1,200],[1.j,3]],
        listnotmat=[1,[1,3]] # not supported
              )  

    Set.Sub.drop(float=3.4, integer=1, string='something', complex=2.+3.j,
        liststr=['a','list','of','strings'],
             ndarray=np.random.rand(3,2,4))  

    print('saving...')
    h5file('.','testfile.h5',*(Set,))   


    # adding to h5 file
    Set=Output('mainset')
    Set.drop(newitem=3,newitem2=[2,4,6])
    h5file('.','testfile.h5',*(Set,)) 


    import read
    h=read.h5file('./testfile.h5')
    embed()             