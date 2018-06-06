'''
Created on 21 Sep 2015
Modified: 9 aug 2017
@author: sm6110
'''

import h5py
from warnings import warn
from IPython import embed


def h5file(filename,ReadList=None):
    '''
    Read entries of a HDF5 file and attaches them to class.
    Groups are saved as sub-classes, while dataset values are saved as class 
    attributes
    
    Important: though this method allows to read all the input/output classes 
    required to run the aeroelastic solutions, ctypes variable are not saved as
    such!
    
    If a ReadList=None all groups/attributes are read. 
    If ReadList is given, the following format is used:
    
    ReadList = ['name']
    
    if name is an attribute, this is read as such.
    if name refers to a group, all entries of the group are read
    '''
    
    class H: pass
    Hinst=H()
     
    ### read and scan file
    hdfile=h5py.File(filename,'r')

    NamesList=[]                   # dataset names
    hdfile.visit(NamesList.append)
    
    ### Identify higher level groups / attributes
    if ReadList is None: 
        MainLev=[]
        for name in NamesList:
            if '/' not in name: MainLev.append(name)
    else:
        MainLev=ReadList

    ### Loop through higher level
    for name in MainLev:
        # sub-group
        if type(hdfile[name]) is h5py._hl.group.Group:
            #print('adding group %s as subclass' %name)
            Ginst=read_group_as_class(hdfile[name])
            Ginst.name=name
            setattr(Hinst,name,Ginst)

        else:
            #print('adding attribute %s' %name)
            setattr(Hinst,name,hdfile[name].value)

    # close and return
    hdfile.close()  
        
    return Hinst  



def read_group_as_class(Grp):
    '''
    Read an hdf5 group
    '''

    class H: pass
    Hinst=H()

    NamesList=[]
    Grp.visit(NamesList.append)

    ### identify higher level
    MainLev=[]
    for name in NamesList:
        if '/' not in name: MainLev.append(name)

    ### Loop through higher level
    for name in MainLev:
        # sub-group
        if type(Grp[name]) is h5py._hl.group.Group:
            #print('adding subclass %s' %name)
            Ginst=read_group_as_class(Grp[name])
            setattr(Hinst,name,Ginst)
        else:
            #print('adding attribute %s' %name)
            setattr(Hinst,name,Grp[name].value)

    return Hinst


  
def h5series(rootname,ReadList=None,N0=0):
    ''' 
    Given a list of datasets, creates a list of lists for all the solutions
    run for a DOE or optimisation. the output is in a list of classes, each
    containing all the attributes/groups read.
    
    See h5file for ReadList format
    '''
    
    outlist=[]
    
    cc=N0
    go_on=True
    
    while go_on is True:
        cc_str =  '%.3d' % (cc)
        filename = rootname + cc_str + '.h5'
        try:
            print( 'Reading: %s' %(filename) )
            Hinst = h5file(filename,ReadList)
            outlist.append(Hinst)
            cc=cc+1
        except OSError:
            print( '%s not found. %s files read in total!' %(filename,cc_str) )
            go_on=False
    return outlist
    
     


def h5list(fileslist,ReadList=None):
    ''' 
    Equivalent to h5series but reads files from an user defined list (fileslist)
    
    All the attributes in 'attrlist' are read and stored in outlist.
    '''
    
    outlist=[]
    
    for filename in fileslist:
        
        print( 'Reading: %s' %(filename) )
        Hinst = h5file(filename,ReadList)
        outlist.append(Hinst)

    return outlist





def conditional_reading(hdfile,obj,fieldname): 
    ''' 
    Given a hdf5 file 'hdfile' and the object obj, the routine:
        a. if the field field-name is found and has a value, assigns it to the
           attribute obj.field-name.  
        b. does nothing otherwise    
    '''
    
    try:
        val = hdfile[fieldname].value
        if val!='not found' and val!='no value':
            setattr(obj,fieldname,val)
    except:
        warn('Attribute "%s" not found!!!' %fieldname)
            
    return obj 




def collect(Hlist,ReadList):
    '''
    Given a list of classes Hlist given in output by h5series or h5list methods,
    the function extracts specific attributes from each element of Hlist and
    assigns the related values to a bunch of output variables.
    
    If ReadList contains subclasses, these will be saved as such.
    
    @warning: method not working with nested subclasses
    '''
    
    
    Nlist = len(Hlist) 
    Nattr = len(ReadList)
    
    ValList = []
    
    for ii in range(Nattr):
        
        name = ReadList[ii]
        if '/' in name:
            subnames=name.split('/')
            Nlevels = len(subnames)
            subclass_list = [getattr(Hlist[hh],subnames[0]) for hh in range(Nlist)]   
            for cc in range(1,Nlevels):
                subclass_list = [ getattr(subclass_list[hh],subnames[cc]) for hh in range(Nlist) ]  
            ValList.append( subclass_list )  
        else:
            ValList.append( getattr(Hlist[hh],name) for hh in range(Nlist) )  
            
                
    return tuple(ValList)
    




if __name__=='__main__':
    pass
       



