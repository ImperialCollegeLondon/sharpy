import numpy as np


def tangent_vector(coord):
    """ Tangent vector calculation for 2+ noded elements.

    Calculates the tangent vector interpolating every dimension
    separately. It uses a (n_nodes - 1) degree polynomial, and the
    differentiation is analytical.

    Args:
        coord (np.ndarray): array of coordinates of the nodes. Dimensions = ``[n_nodes, ndim]``

    Notes:
        Dimensions are treated independent from each other, interpolating polynomials are computed
        individually.

    """
    n_nodes, ndim = coord.shape

    polynomial_degree = n_nodes - 1
    # first, the polynomial fit.
    # we are going to differentiate wrt the indices ([0, 1, 2] for a 3-node)
    polyfit_vec = []  # we are going to store here the coefficients of the polyfit
    for idim in range(ndim):
        polyfit_vec.append(np.polyfit(range(n_nodes), coord[:, idim],
                                      polynomial_degree))

    # differentiation
    polyfit_der_vec = []
    for idim in range(ndim):
        polyfit_der_vec.append(np.poly1d(np.polyder(polyfit_vec[idim])))

    # tangent vector calculation
    # \vec{t} = \frac{fx'i + fy'j + fz'k}/mod(...)
    tangent = np.zeros_like(coord)
    for inode in range(n_nodes):
        vector = []
        for idim in range(ndim):
            vector.append((polyfit_der_vec[idim])(inode))
        # vector = np.array([polyfit_der_vec[0](inode),
        vector = np.array(vector)
        vector /= np.linalg.norm(vector)
        tangent[inode, :] = vector

    # check orientation of tangent vector
    fake_tangent = np.zeros_like(tangent)
    for inode in range(n_nodes):
        if inode == n_nodes - 1:
            # use previous vector
            fake_tangent[inode, :] = fake_tangent[inode - 1, :]
            continue
        fake_tangent[inode, :] = coord[inode+1, :] - coord[inode, :]

    inverted_tangent = False
    for inode in range(n_nodes):
        if np.dot(tangent[inode, :], fake_tangent[inode, :]) < 0:
            inverted_tangent = True
            break

    if inverted_tangent:
        tangent *= -1

    return tangent, polyfit_vec


def unit_vector(vector):
    return vector/np.linalg.norm(vector)


def rotation_matrix_around_axis(axis, angle):
    axis = unit_vector(axis)
    rot = np.cos(angle)*np.eye(3)
    rot += np.sin(angle)*skew(axis)
    rot += (1 - np.cos(angle))*np.outer(axis, axis)
    return rot


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


def triad2rot(xb, yb, zb):
    '''
    If the input triad is the "b" coord system given in "a" frame,
    (the vectors of the triad are xb, yb, zb)
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


def angle_between_vector_and_plane(vector, plane_normal):
    angle = np.arcsin((np.linalg.norm(np.dot(vector, plane_normal)))/
                      (np.linalg.norm(vector)*np.linalg.norm(plane_normal)))
    return angle

