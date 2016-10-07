#
import numpy as np
import scipy.linalg

def unit_vector(a):
    return a/np.linalg.norm(a)

def skew(a):
    if not a.size == 3:
        raise Exception('The input vector is not 3D')

    matrix = np.zeros((3,3))
    matrix[0,1] = -a[2]
    matrix[1,0] =  a[2]
    matrix[2,0] = -a[1]
    matrix[0,2] =  a[1]
    matrix[1,2] = -a[0]
    matrix[2,1] =  a[0]

    return matrix

def crv_from_vectors(a, b):
    # import pdb;pdb.set_trace()
    if np.linalg.norm(np.cross(a,b)) < 1e-8:
        axis = unit_vector(a)
    else:
        axis = np.cross(a,b)/np.linalg.norm(np.cross(a, b))
    angle = np.arccos(np.dot(unit_vector(a), unit_vector(b)))
    return axis*angle

def crv2rot(crv):
    rot = np.eye(3)
    if np.linalg.norm(crv) < 1e-8:
        return rot

    crv_norm = np.linalg.norm(crv)
    rot += (np.sin(crv_norm)/crv_norm)*skew(crv)
    rot += (1 - np.cos(crv_norm))/(crv_norm)*np.dot(skew(crv), skew(crv))
    return rot



if __name__ == '__main__':
    a = np.array([1, 0, 0])
    b = np.array([1., 0.05, 0])
    crv = crv_from_vectors(a,b)
    print(crv)
    rot = crv2rot(crv)
    # check using exponential mapping
    rot_2 = scipy.linalg.expm(skew(crv))
    if np.linalg.norm(rot - rot_2) > 1e-3:
        print('Check crv2rot routine')
        print(rot)
        print('----')
        print(rot_2)
    print(rot)
    print(np.dot(rot, a))
    import pdb; pdb.set_trace()
