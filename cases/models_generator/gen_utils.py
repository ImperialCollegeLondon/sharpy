import numpy as np

def change_dic(dic, variable_name, new_value):
    """ 
    Inside a dictionary, dic, change the variable_name with the new_value.
    dic can have an arbitrary number of  levels with other dictionaries but
    note the variable_name needs to be defined at some level
    """

    for k, v in dic.items():
        if not isinstance(v,dict):
            if variable_name in k:
                dic[k] = new_value
        else:
            change_dic(v, variable_name, new_value)
            
    return dic

def update_dic(dic, dic_new):
    """ 
    Updates dic with dic_new in a recursive manner 
    """
    
    for k, v in dic_new.items():
        if not isinstance(v,dict):
                dic.update({k:v})
        else:
            if k in dic.keys():
                if isinstance(dic[k],dict):
                    update_dic(dic[k],v)
                else:
                    dic[k] = v
            else:
                dic[k] = v
    return dic


def flatten2(lis):
    """
    Returns the flatten list from lis, 
    which may have an arbitrary number of sublists inside:
    flatten2([[1,2,3],[4],[5,6,7,8]]) -> [1,2,3,4,5,6,7,8]
    flatten2([1,[2,3],4,[[5,6],[7,[[8]]]]]) -> [1,2,3,4,5,6,7,8]
    """
    
    l=list(lis)
    i=0

    while i < len(l):
         if type(l[i]) is tuple:
          l[i] = list(l[i])
         if type(l[i]) is list:
          if len(l[i])==0:
           del l[i]
           continue
          for j in range(len(l[i])):
            l.insert(i+j,l[i+j][j])

          del l[i+j+1]
         else:
          i=i+1

    return l

def do_connectivities(p0,num_elem,lista=[]):
    """
    Builds SHARPy connectivities for elements in increasing order
    """
    if num_elem < 1:
        return np.array(lista)
    lista.append([p0, p0+2, p0+1])
    return do_connectivities(p0+2, num_elem-1, lista)


def get_chord(ledge, tedge, beam):
    """
    Returns vectors for the chord and elastic_axis from leading-edge, trailing edge
    and beam lines
    """
    
    num_node = len(beam)
    assert len(ledge)==num_node
    assert len(tedge)==num_node
    if isinstance(ledge,list):
        ledge = np.array(ledge)
    if isinstance(tedge,list):
        tedge = np.array(tedge)
    if isinstance(beam,list):
        beam = np.array(beam)
        
    chord = np.zeros(num_node)
    ea = np.zeros(num_node)
    for i in range(num_node):
        chord[i] = np.linalg.norm(ledge[i]-tedge[i])
        ea[i] = np.linalg.norm(ledge[i]-beam[i])/chord[i]
    return ea, chord

def node2aero(*qlist):
    """ 
    Aero quantities (size=(num_node)) in qlist are converted into 
    the appropriate SHARPy vector (size=(num_elem,3)
    """
    num_list = len(qlist)
    qx = []
    for li in range(num_list):
        q = qlist[li]
        num_node = len(q)
        assert num_node%2 ==1, 'num_node is an even number'
        num_element = int((num_node-1)/2)
        qaero = np.zeros((num_element,3))
        for k in range(num_node):
            i=int(k/2)
            j=k%2
            if i==0 and j==0:
                qaero[i][j] = q[0]
                continue
            if i==num_element:
                qaero[i-1][1] = q[-1]
                break
            if j==0:
                qaero[i,0] = q[k]
                qaero[i-1,1] = q[k]
            else:
                qaero[i,2] = q[k]
        qx.append(qaero)
    if num_list > 1:
        return qx
    else:
        return qx[0]
