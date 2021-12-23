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
    return chord, ea

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


def get_edge(x, v_p, edge1, edge2):
    #import pdb; pdb.set_trace();

    v_e = (edge2 - edge1)/np.linalg.norm(edge2 - edge1)
    v_e1x = (x - edge1)/np.linalg.norm(x - edge1)
    alpha_e12x = np.arccos(v_e.dot(v_e1x))
    alpha_e1xp = np.arccos(-v_e1x.dot(v_p))
    distance_e1x = np.linalg.norm(edge1-x)
    distance2p = np.sin(alpha_e1xp)*distance_e1x/np.sin(np.pi-alpha_e12x - alpha_e1xp)
    intersercting_p = edge1 + v_e*distance2p

    return intersercting_p

def get_tl_edge(beam, leading_edge1, leading_edge2, trailing_edge1, trailing_edge2,
                rtol=1e-5, atol=1e-8, **kwargs):

    num_nodes = len(beam)
    plane_verctor = np.cross(leading_edge1 - leading_edge2,
                             leading_edge1 - trailing_edge1)
    assert np.allclose(np.dot(trailing_edge2-leading_edge1, plane_verctor), 0., rtol, atol), \
                 'Leading and trailing edge points not defined in the same plane'
    
    for i in range(num_nodes):
        assert np.allclose(np.dot(beam[i]-leading_edge1, plane_verctor), 0., rtol, atol), \
               'beam nodes not defined in aerodynamic plane'

    l_edge = [leading_edge1]
    t_edge = [trailing_edge1]
    v_pl = (leading_edge1-beam[0])/np.linalg.norm(leading_edge1-beam[0])
    v_pt = (trailing_edge1-beam[0])/np.linalg.norm(trailing_edge1-beam[0])
    for i in range(1, num_nodes):
        l_edge.append(get_edge(beam[i], v_pl, leading_edge1, leading_edge2))
        t_edge.append(get_edge(beam[i], v_pt, trailing_edge1, trailing_edge2))
    assert np.allclose(l_edge[-1], leading_edge2, rtol, atol), \
        'leading edge end point not coincident with given one' 
    assert np.allclose(t_edge[-1], trailing_edge2, rtol, atol), \
        'trailing edge end point not coincident with given one' 

    return l_edge, t_edge

def from4points2chord(beam, leading_edge1, leading_edge2,
                      trailing_edge1, trailing_edge2, out_consecutive=False, **kwargs):
    """
    This function gets the chord and elastic_axis variables in SHARPy from the beam
    nodes and 4 corners defining the aerodynamic surface
    """
    
    if isinstance(leading_edge1, list):
        leading_edge1 = np.array(leading_edge1)
    if isinstance(leading_edge2, list):
        leading_edge2 = np.array(leading_edge2)
    if isinstance(trailing_edge1, list):
        trailing_edge1 = np.array(trailing_edge1)
    if isinstance(trailing_edge2, list):
        trailing_edge2 = np.array(trailing_edge2)
        
    le, te = get_tl_edge(beam, leading_edge1, leading_edge2, trailing_edge1, trailing_edge2, **kwargs)
    chord1, ea1 = get_chord(le, te, beam)
    if out_consecutive:
        return chord1, ea1
    else:
        chord, ea = node2aero(chord1, ea1)
        return chord, ea
    

if __name__=='__main__':
    chord01 = 6.5
    ea01 = 0.4
    sweep1 = np.pi/180*20
    num_node1 = 7
    lenght1 = 6.
    dl1 = lenght1/num_node1
    ledge01 = np.array([-chord01*ea01,0.,0.])
    ledge1 = [np.array([dl1*i*np.sin(sweep1), dl1*i*np.cos(sweep1), 1.]) for i in range(num_node1)]
    ledge1 = np.array(ledge1) + ledge01
    beam1 = [np.array([dl1*i*np.sin(sweep1), dl1*i*np.cos(sweep1), 1.]) for i in range(num_node1)]
    beam1 = np.array(beam1)
    tedge1 = [np.array([(1.-ea01)*chord01, dl1*i*np.cos(sweep1), 1.]) for i in range(num_node1)]
    tedge1 = np.array(tedge1)

    le,te = get_tl_edge(beam1, ledge1[0], ledge1[-1], tedge1[0], tedge1[-1])


    ea1, chord1 = get_chord(ledge1, tedge1, beam1)
    ea1, chord1 = node2aero(ea1,chord1)
    ea,c = from4points2chord(beam1, ledge1[0], ledge1[-1], tedge1[0], tedge1[-1],True)
