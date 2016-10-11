#
import numpy as np


def unit_vector(vector):
    return vector/np.linalg.norm(vector)


def skew(vector):
    if not vector.size == 3:
        raise Exception('The input vector is not 3D')

    matrix = np.zeros((3, 3))
    matrix[1, 2] = -vector[0]
    matrix[2, 0] = -vector[1]
    matrix[0, 1] = -vector[2]
    matrix[2, 1] = vector[0]
    matrix[0, 2] = vector[1]
    matrix[1, 0] = vector[2]
    return matrix

#
# def crv_from_vectors(veca, vecb):
#     # import pdb;pdb.set_trace()
#     if np.linalg.norm(np.cross(veca,vecb)) < 1e-8:
#         axis = unit_vector(veca)
#     else:
#         axis = np.cross(veca,vecb)/np.linalg.norm(np.cross(veca, vecb))
#     angle = np.arccos(np.dot(unit_vector(veca), unit_vector(vecb)))
#     return axis*angle
#
#
# def crv2rot(crv):
#     rot = np.eye(3)
#     if np.linalg.norm(crv) < 1e-8:
#         return rot
#
#     crv_norm = np.linalg.norm(crv)
#     rot += (np.sin(crv_norm)/crv_norm)*skew(crv)
#     rot += (1 - np.cos(crv_norm))/(crv_norm)*np.dot(skew(crv), skew(crv))
#     return rot


def triad2rot(xb, yb, zb):
    '''
    If the input triad is the "b" coord system given in "a" frame,
    this function returns Rab
    :param xb:
    :param yb:
    :param zb:
    :return: rotation matrix Rab
    '''
    rot = np.column_stack((xb, yb, zb))
    return rot


def rot_matrix_2d(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def angle_between_vectors(vec_a, vec_b):
    return np.arctan2(np.linalg.norm(np.cross(vec_a, vec_b)), np.dot(vec_a, vec_b))


if __name__ == '__main__':
    pass
