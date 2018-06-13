import numpy as np
import scipy.linalg


def tangent_vector(in_coord, ordering=None):
    """ Tangent vector calculation for 2+ noded elements.

    Calculates the tangent vector interpolating every dimension
    separately. It uses a (n_nodes - 1) degree polynomial, and the
    differentiation is analytical.

    Args:
        in_coord (np.ndarray): array of coordinates of the nodes. Dimensions = ``[n_nodes, ndim]``

    Notes:
        Dimensions are treated independent from each other, interpolating polynomials are computed
        individually.

    """
    n_nodes, ndim = in_coord.shape

    if ordering is None:
        if n_nodes == 2:
            ordering = [0, n_nodes - 1]
        elif n_nodes == 3:
            ordering = [0, 2, 1]
        else:
            raise NotImplementedError('Elements with more than 3 nodes are not supported')

    polyfit_vec, polyfit_der_vec, coord = get_polyfit(in_coord, ordering)

    # tangent vector calculation
    # \vec{t} = \frac{fx'i + fy'j + fz'k}/mod(...)
    tangent = np.zeros_like(coord)
    for inode in range(n_nodes):
        vector = []
        for idim in range(ndim):
            vector.append((polyfit_der_vec[idim])(inode))
        # vector = np.array([polyfit_der_vec[0](inode),
        # DEBUG
        vector = np.array(vector)
        if (np.linalg.norm(vector)) == 0.0:
            print(vector)
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


def get_polyfit(in_coord, ordering):
    coord = in_coord.copy()
    n_nodes, ndim = coord.shape
    for index in range(n_nodes):
        order = ordering[index]
        coord[index, :] = in_coord[order, :]

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

    return polyfit_vec, polyfit_der_vec, coord


def unit_vector(vector):
    """
    Tested
    :param vector:
    :return:
    """
    if np.linalg.norm(vector) < 1e-6:
        return np.zeros_like(vector)
    return vector/np.linalg.norm(vector)


def rotation_matrix_around_axis(axis, angle):
    axis = unit_vector(axis)
    rot = np.cos(angle)*np.eye(3)
    rot += np.sin(angle)*skew(axis)
    rot += (1 - np.cos(angle))*np.outer(axis, axis)
    return rot


def skew(vector):
    if not vector.size == 3:
        raise ValueError('The input vector is not 3D')

    matrix = np.zeros((3, 3))
    matrix[1, 2] = -vector[0]
    matrix[2, 0] = -vector[1]
    matrix[0, 1] = -vector[2]
    matrix[2, 1] = vector[0]
    matrix[0, 2] = vector[1]
    matrix[1, 0] = vector[2]
    return matrix


def triad2rot(xb, yb, zb):
    """
    If the input triad is the "b" coord system given in "a" frame,
    (the vectors of the triad are xb, yb, zb)
    this function returns Rab
    :param xb:
    :param yb:
    :param zb:
    :return: rotation matrix Rab
    """
    rot = np.row_stack((xb, yb, zb))
    return rot


def rot_matrix_2d(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def angle_between_vectors(vec_a, vec_b):
    angle = np.arctan2(np.linalg.norm(np.cross(vec_a, vec_b)), np.dot(vec_a, vec_b))
    return angle


def angle_between_vectors_sign(vec_a, vec_b, plane_normal=np.array([0, 0, 1])):
    angle = np.arctan2(np.linalg.norm(np.cross(vec_a, vec_b)), np.dot(vec_a, vec_b))
    if np.dot(plane_normal, np.cross(vec_a, vec_b)) < 0:
        angle *= -1
    return angle


def angle_between_vector_and_plane(vector, plane_normal):
    angle = np.arcsin((np.linalg.norm(np.dot(vector, plane_normal)))/
                      (np.linalg.norm(vector)*np.linalg.norm(plane_normal)))
    return angle


def rot2crv(rot):
    if np.linalg.norm(rot) < 1e-6:
        raise AttributeError('Element Vector V is not orthogonal to reference line (51105)')

    quat = mat2quat(rot)
    crv = quat2crv(quat)

    if np.linalg.norm(crv) < 1.0e-15:
        crv[0] = rot[1, 2]
        crv[1] = rot[2, 0]
        crv[2] = rot[0, 1]

    crv = crv_bounds(crv)
    return crv


def mat2quat(mat):
    matT = mat.T

    s = np.zeros((4, 4))

    s[0, 0] = 1.0 + np.trace(matT)
    s[0, 1:] = matrix2skewvec(matT)

    s[1, 0] = matT[2, 1] - matT[1, 2]
    s[1, 1] = 1.0 + matT[0, 0] - matT[1, 1] - matT[2, 2]
    s[1, 2] = matT[0, 1] + matT[1, 0]
    s[1, 3] = matT[0, 2] + matT[2, 0]

    s[2, 0] = matT[0, 2] - matT[2, 0]
    s[2, 1] = matT[1, 0] + matT[0, 1]
    s[2, 2] = 1.0 - matT[0, 0] + matT[1, 1] - matT[2, 2]
    s[2, 3] = matT[1, 2] + matT[2, 1]

    s[3, 0] = matT[1, 0] - matT[0, 1]
    s[3, 1] = matT[0, 2] + matT[2, 0]
    s[3, 2] = matT[1, 2] + matT[2, 1]
    s[3, 3] = 1.0 - matT[0, 0] - matT[1, 1] + matT[2, 2]

    smax = np.max(np.diag(s))
    ismax = np.argmax(np.diag(s))

    # compute quaternion angles
    quat = np.zeros((4,))
    quat[ismax] = 0.5*np.sqrt(smax)
    for i in range(4):
        if i == ismax:
            continue
        quat[i] = 0.25*s[ismax, i]/quat[ismax]

    return quat


def matrix2skewvec(matrix):
    vector = np.array([matrix[2, 1] - matrix[1, 2],
                       matrix[0, 2] - matrix[2, 0],
                       matrix[1, 0] - matrix[0, 1]])
    return vector


def quat2crv(quat):
    crv_norm = 2.0*np.arccos(max(-1.0, min(quat[0], 1.0)))

    # normal vector
    if abs(crv_norm) < 1e-15:
        psi = np.zeros((3,))
    else:
        psi = crv_norm*quat[1:4]/np.sin(crv_norm*0.5)

    return psi

def crv_bounds(crv_ini):
    crv = crv_ini.copy()
    # original norm
    norm_ini = np.linalg.norm(crv_ini)

    # force the norm to be in [-pi, pi]
    norm = norm_ini - 2.0*np.pi*int(norm_ini/(2*np.pi))

    if norm == 0.0:
        crv *= 0.0
    else:
        if norm > np.pi:
            norm -= 2.0*np.pi
        elif norm < -np.pi:
            norm += 2.0*np.pi
        crv *= (norm/norm_ini)

    return crv


def triad2crv(xb, yb, zb):
    return rot2crv(triad2rot(xb, yb, zb))


def crv2triad(psi):
    rot_matrix = crv2rot(psi)
    return rot_matrix[:, 0], rot_matrix[:, 1], rot_matrix[:, 2]


def crv2rot(psi):
    # this returns Cab

    # this is psi2mat in the matlab version
    norm_psi = np.linalg.norm(psi)

    if norm_psi < 1e-15:
        skew_psi = rot_skew(psi)
        rot_matrix = np.eye(3) + skew_psi + 0.5*np.dot(skew_psi, skew_psi)
    else:
        normal = psi/norm_psi
        skew_normal = rot_skew(normal)

        rot_matrix = np.eye(3)
        rot_matrix += np.sin(norm_psi)*skew_normal
        rot_matrix += (1.0 - np.cos(norm_psi))*np.dot(skew_normal, skew_normal)

    return rot_matrix


def crv2tan(psi):
    norm_psi = np.linalg.norm(psi)
    psi_skew = skew(psi)

    eps = 1e-8
    # if norm_psi < eps:
    #     k1 = 1.0
    #     k2 = 1.0/6.0
    # else:
    #     k1 = np.sin(norm_psi*0.5)/(norm_psi*0.5)
    #     k2 = (1.0 - np.sin(norm_psi)/norm_psi)/(norm_psi*norm_psi)
    #
    # T = np.eye(3) - (0.5*k1*k1)*psi_skew + k2*np.dot(psi_skew, psi_skew)

    # new expression for tangent operator (equation 4.11 in Geradin and Cardona)
    if norm_psi < eps:
        return np.eye(3) - 0.5*psi_skew + 1.0/6.0*np.dot(psi_skew, psi_skew)
    else:
        k1 = (1.0 - np.cos(norm_psi))/(norm_psi*norm_psi)
        k2 = (1.0 - np.sin(norm_psi)/norm_psi)/(norm_psi*norm_psi)
        return np.eye(3) + k1*psi_skew + k2*np.dot(psi_skew, psi_skew)


def crv2invtant(psi):
    tan = crv2tan(psi).T
    return np.linalg.inv(tan)


def triad2crv_vec(v1, v2, v3):
    n_nodes, _ = v1.shape
    crv_vec = np.zeros((n_nodes, 3))
    for inode in range(n_nodes):
        crv_vec[inode, :] = triad2crv(v1[inode, :], v2[inode, :], v3[inode, :])

    return crv_vec


def crv2triad_vec(crv_vec):
    n_nodes, _ = crv_vec.shape
    v1 = np.zeros((n_nodes, 3))
    v2 = np.zeros((n_nodes, 3))
    v3 = np.zeros((n_nodes, 3))
    for inode in range(n_nodes):
        v1[inode, :], v2[inode, :], v3[inode, :] = crv2triad(crv_vec[inode, :])
    return v1, v2, v3


def quat2rot(q1):
    """@brief Calculate rotation matrix based on quaternions.
    See Aircraft Control and Simulation, pag. 31, by Stevens, Lewis.
    Copied from S. Maraniello's SHARPy

    Remark: if A is a FoR obtained rotating a FoR G of angle fi about an axis n (remind n will be
    invariant during the rotation), and q is the related quaternion q(fi,n), the function will
    return the matrix Cag such that:
        - Cag rotates A to G
        - Cag transforms the coordinates of a vector defined in G component to A components.
    """

    q = q1.copy(order='F')
    q /= np.linalg.norm(q)

    rot_mat = np.zeros((3, 3), order='F')

    rot_mat[0, 0] = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
    rot_mat[1, 1] = q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2
    rot_mat[2, 2] = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2

    rot_mat[0, 1] = 2.*(q[1]*q[2] + q[0]*q[3])
    rot_mat[1, 0] = 2.*(q[1]*q[2] - q[0]*q[3])

    rot_mat[0, 2] = 2.*(q[1]*q[3] - q[0]*q[2])
    rot_mat[2, 0] = 2.*(q[1]*q[3] + q[0]*q[2])

    rot_mat[1, 2] = 2.*(q[2]*q[3] + q[0]*q[1])
    rot_mat[2, 1] = 2.*(q[2]*q[3] - q[0]*q[1])

    return rot_mat


def rot_skew(vec):
    matrix = np.zeros((3, 3))
    matrix[0, 1] = -vec[2]
    matrix[0, 2] = vec[1]
    matrix[1, 0] = vec[2]
    matrix[1, 2] = -vec[0]
    matrix[2, 0] = -vec[1]
    matrix[2, 1] = vec[0]
    return matrix


def rotation3d_x(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    mat = np.zeros((3, 3))
    mat[0, :] = [1.0, 0.0, 0.0]
    mat[1, :] = [0.0,   c,  -s]
    mat[2, :] = [0.0,   s,   c]
    return mat


def rotation3d_y(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    mat = np.zeros((3, 3))
    mat[0, :] = [c, 0.0, -s]
    mat[1, :] = [0.0, 1.0, 0.0]
    mat[2, :] = [s, 0.0,  c]
    return mat


def rotation3d_z(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    mat = np.zeros((3, 3))
    mat[0, :] = [  c,  -s, 0.0]
    mat[1, :] = [  s,   c, 0.0]
    mat[2, :] = [0.0, 0.0, 1.0]
    return mat


def rotate_crv(crv_in, axis, angle):
    crv = np.zeros_like(crv_in)
    C = crv2rot(crv_in).T
    rot = rotation_matrix_around_axis(axis, angle)
    C = np.dot(C, rot)
    crv = rot2crv(C)
    return crv


def euler2rot(euler):
    """
    :param euler: [roll, pitch, yaw]
    :return:
    """
    rot = rotation3d_z(euler[2])
    rot = np.dot(rotation3d_y(euler[1]), rot)
    rot = np.dot(rotation3d_x(euler[0]), rot)
    return rot


def euler2quat(euler):
    euler_rot = euler2rot(euler)  # this is Cag
    quat = mat2quat(euler_rot)
    return quat


def crv_dot2omega(crv, crv_dot):
    return np.dot(crv2tan(crv).T, crv_dot)


def crv_dot2Omega(crv, crv_dot):
    return np.dot(crv2tan(crv), crv_dot)


def quaternion_product(q, r):
    result = np.zeros((4,))
    result[0] = q[0]*r[0] - q[1]*r[1] - q[2]*r[2] - q[3]*r[3]
    result[1] = q[0]*r[1] + q[1]*r[0] + q[2]*r[3] - q[3]*r[2]
    result[2] = q[0]*r[2] - q[1]*r[3] + q[2]*r[0] + q[3]*r[1]
    result[3] = q[0]*r[3] + q[1]*r[2] - q[2]*r[1] + q[3]*r[0]
    return result


def omegadt2quat(omegadt):
    quat = np.zeros((4,))

    omegadt_norm = np.linalg.norm(omegadt)
    quat[0] = np.cos(0.5*omegadt_norm)
    quat[1:4] = unit_vector(omegadt)*np.sin(0.5*omegadt_norm)
    return quat


def rotate_quaternion(quat, omegadt):
    return quaternion_product(omegadt2quat(omegadt), quat)

