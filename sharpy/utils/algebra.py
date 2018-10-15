'''
Rotation algebra library

Note: testing in tests/utils/algebra_test
'''

import numpy as np
import scipy.linalg
from warnings import warn 

#######
# functions for back compatibility
def quat2rot(quat):
    warn('quat2rot(quat) is obsolite! Use quat2rotation(quat).T instead!')
    return quat2rotation(quat).T
def crv2rot(psi):
    warn('crv2rot(psi) is obsolite! Use crv2rotation(psi) instead!')    
    return crv2rotation(psi)
def rot2crv(rot):
    warn('rot2crv(rot) is obsolite! Use rotation2crv(rot.T) instead!')    
    return rotation2crv(rot.T)
def triad2rot(xb,yb,zb):
    warn('triad2rot(xb,yb,zb) is obsolite! Use triad2rotation(xb,yb,zb).T instead!') 
    return triad2rotation(xb,yb,zb).T
def mat2quat(rot):
     warn('mat2quat(rot) is obsolite! Use rotation2quat(rot.T) instead!')
     return rotation2quat(rot.T)
#######

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


def triad2rotation(xb, yb, zb):
    """
    If the input triad is the "b" coord system given in "a" frame,
    (the vectors of the triad are xb, yb, zb), this function returns Rab, ie the
    rotation matrix required to rotate the FoR A onto B.
    :param xb:
    :param yb:
    :param zb:
    :return: rotation matrix Rab
    """  
    return np.column_stack((xb, yb, zb))




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


# def mat2quat(mat):
#     matT = mat.T

#     s = np.zeros((4, 4))

#     s[0, 0] = 1.0 + np.trace(matT)
#     s[0, 1:] = matrix2skewvec(matT)

#     s[1, 0] = matT[2, 1] - matT[1, 2]
#     s[1, 1] = 1.0 + matT[0, 0] - matT[1, 1] - matT[2, 2]
#     s[1, 2] = matT[0, 1] + matT[1, 0]
#     s[1, 3] = matT[0, 2] + matT[2, 0]

#     s[2, 0] = matT[0, 2] - matT[2, 0]
#     s[2, 1] = matT[1, 0] + matT[0, 1]
#     s[2, 2] = 1.0 - matT[0, 0] + matT[1, 1] - matT[2, 2]
#     s[2, 3] = matT[1, 2] + matT[2, 1]

#     s[3, 0] = matT[1, 0] - matT[0, 1]
#     s[3, 1] = matT[0, 2] + matT[2, 0]
#     s[3, 2] = matT[1, 2] + matT[2, 1]
#     s[3, 3] = 1.0 - matT[0, 0] - matT[1, 1] + matT[2, 2]

#     smax = np.max(np.diag(s))
#     ismax = np.argmax(np.diag(s))

#     # compute quaternion angles
#     quat = np.zeros((4,))
#     quat[ismax] = 0.5*np.sqrt(smax)
#     for i in range(4):
#         if i == ismax:
#             continue
#         quat[i] = 0.25*s[ismax, i]/quat[ismax]

#     return quat


def rotation2quat(Cab):
    '''
    Given a rotation matrix Cab rotating the frame a onto b, the function returns
    the minimal "positive angle" quaternion representing this rotation. 

    Note: this is the inverse of quat2rotation for Cartesian rotation vectors 
    associated to rotations in the range [-pi,pi], i.e.:
        fv == algebra.rotation2crv(algebra.crv2rotation(fv))  
    for each fv=a*nv such that nv is a unit vector and the scalar a in [-pi,pi].
    '''

    s = np.zeros((4, 4))

    s[0, 0] = 1.0 + np.trace(Cab)
    s[0, 1:] = matrix2skewvec(Cab)

    s[1, 0] = Cab[2, 1] - Cab[1, 2]
    s[1, 1] = 1.0 + Cab[0, 0] - Cab[1, 1] - Cab[2, 2]
    s[1, 2] = Cab[0, 1] + Cab[1, 0]
    s[1, 3] = Cab[0, 2] + Cab[2, 0]

    s[2, 0] = Cab[0, 2] - Cab[2, 0]
    s[2, 1] = Cab[1, 0] + Cab[0, 1]
    s[2, 2] = 1.0 - Cab[0, 0] + Cab[1, 1] - Cab[2, 2]
    s[2, 3] = Cab[1, 2] + Cab[2, 1]

    s[3, 0] = Cab[1, 0] - Cab[0, 1]
    s[3, 1] = Cab[0, 2] + Cab[2, 0]
    s[3, 2] = Cab[1, 2] + Cab[2, 1]
    s[3, 3] = 1.0 - Cab[0, 0] - Cab[1, 1] + Cab[2, 2]

    smax = np.max(np.diag(s))
    ismax = np.argmax(np.diag(s))

    # compute quaternion angles
    quat = np.zeros((4,))
    quat[ismax] = 0.5*np.sqrt(smax)
    for i in range(4):
        if i == ismax:
            continue
        quat[i] = 0.25*s[ismax, i]/quat[ismax]

    return quat_bound(quat)


def quat_bound(quat):
    '''
    Given a quaternion associated to a rotation of angle a about an axis nv, the 
    function "bounds" the quaternion, i.e. sets the rotation axis nv such that
    a in [-pi,pi]. 

    Note: as quaternions are defined as qv=[cos(a/2); sin(a/2)*nv], this is
    equivalent to enforce qv[0]>=0.
    '''
    if quat[0]<0:
        quat*=-1.
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


def crv2quat(psi):
    '''
    Converts a Cartesian rotation vector into a "minimal rotation" quaternion,
    ie, being the quaternion defined as:
        qv= [cos(a/2); sin(a/2)*nv ]
    the rotation axis is such that the rotation angle a is in [-pi,pi] or,
    equivalently, qv[0]>=0.    
    '''

    # minimise crv rotation
    psi_new=crv_bounds(psi)    

    fi=np.linalg.norm(psi_new)
    if fi > 1e-15:
        nv=psi_new/fi
    else:
        nv = psi_new

    quat=np.zeros((4,))
    quat[0]=np.cos(.5*fi)
    quat[1:]=np.sin(.5*fi)*nv 

    return quat


def crv_bounds(crv_ini):
    '''
    Forces the Cartesian rotation vector norm to be in [-pi,pi], i.e. determines
    the rotation axis orientation so as to ensure "minimal rotation".
    '''

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
    return rotation2crv(triad2rotation(xb, yb, zb))


def crv2triad(psi):
    rot_matrix = crv2rotation(psi)
    return rot_matrix[:, 0], rot_matrix[:, 1], rot_matrix[:, 2]


def crv2rotation(psi):
    '''
    Given a Cartesian rotation vector psi, the function produces the rotation
    matrix required to rotate a vector according to psi.

    Note: this is psi2mat in the matlab version
    '''

    norm_psi = np.linalg.norm(psi)

    if norm_psi < 1e-15:
        skew_psi = skew(psi)
        rot_matrix = np.eye(3) + skew_psi + 0.5*np.dot(skew_psi, skew_psi)
    else:
        normal = psi/norm_psi
        skew_normal = skew(normal)

        rot_matrix = np.eye(3)
        rot_matrix += np.sin(norm_psi)*skew_normal
        rot_matrix += (1.0 - np.cos(norm_psi))*np.dot(skew_normal, skew_normal)

    return rot_matrix


def rotation2crv(Cab):
    '''
    Given a rotation matrix Cab rotating the frame a onto b, the function returns
    the minimal size Cartesian rotation vector representing this rotation. 

    Note: this is the inverse of crv2rotation for Cartesian rotation vectors 
    associated to rotations in the range [-pi,pi], i.e.:
        fv == algebra.rotation2crv(algebra.crv2rotation(fv))  
    for each fv=a*nv such that nv is a unit vector and the scalar a in [-pi,pi].
    '''
    
    if np.linalg.norm(Cab) < 1e-6:
        raise AttributeError(\
                 'Element Vector V is not orthogonal to reference line (51105)')

    quat = rotation2quat(Cab)
    psi = quat2crv(quat)

    if np.linalg.norm(Cab) < 1.0e-15:
        psi[0] = Cab[1, 2]
        psi[1] = Cab[2, 0]
        psi[2] = Cab[0, 1]

    return crv_bounds(psi)



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
        k1 = (np.cos(norm_psi) - 1.0)/(norm_psi*norm_psi)
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


def quat2rotation(q1):
    """@brief Calculate rotation matrix based on quaternions.
    See Aircraft Control and Simulation, pag. 31, by Stevens, Lewis.
    Copied from S. Maraniello's SHARPy

    Remark: if B is a FoR obtained rotating a FoR A of angle fi about an axis n 
    (remind n will be invariant during the rotation), and q is the related 
    quaternion q(fi,n), the function will return the matrix Cab such that:
        - Cab rotates A onto B
        - Cab transforms the coordinates of a vector defined in B component to 
        A components.
    """

    q = q1.copy(order='F')
    q /= np.linalg.norm(q)

    rot_mat = np.zeros((3, 3), order='F')

    rot_mat[0, 0] = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
    rot_mat[1, 1] = q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2
    rot_mat[2, 2] = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2

    rot_mat[1, 0] = 2.*(q[1]*q[2] + q[0]*q[3])
    rot_mat[0, 1] = 2.*(q[1]*q[2] - q[0]*q[3])

    rot_mat[2, 0] = 2.*(q[1]*q[3] - q[0]*q[2])
    rot_mat[0, 2] = 2.*(q[1]*q[3] + q[0]*q[2])

    rot_mat[2, 1] = 2.*(q[2]*q[3] + q[0]*q[1])
    rot_mat[1, 2] = 2.*(q[2]*q[3] - q[0]*q[1])

    return rot_mat


def rot_skew(vec):
    from warnings import warn
    warn("use 'skew' function instead of 'rot_skew'")
    return skew(vec)


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
    C = crv2rotation(crv_in).T
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


def get_triad(coordinates_def, frame_of_reference_delta, twist=None, n_nodes=3, ordering=np.array([0, 2, 1])):
    """
    Generates two unit vectors in body FoR that define the local FoR for
    a beam element. These vectors are calculated using `frame_of_reference_delta`
    :return:
    """
    # now, calculate tangent vector (and coefficients of the polynomial
    # fit just in case)
    tangent, polyfit = tangent_vector(
        coordinates_def,
        ordering)
    normal = np.zeros_like(tangent)
    binormal = np.zeros_like(tangent)

    # v_vector is the vector with origin the FoR node and delta
    # equals frame_of_reference_delta
    for inode in range(n_nodes):
        v_vector = frame_of_reference_delta[inode, :]
        normal[inode, :] = unit_vector(np.cross(
                                                tangent[inode, :],
                                                v_vector
                                                )
                                           )
        binormal[inode, :] = -unit_vector(np.cross(
                                                tangent[inode, :],
                                                normal[inode, :]
                                                        )
                                              )

    if twist is not None:
        raise NotImplementedError('Structural twist is not yet supported in algebra.get_triad, but it is in beamstructures.py')
    # # we apply twist now
    # for inode in range(self.n_nodes):
    #     if not self.structural_twist[inode] == 0.0:
    #         rotation_mat = algebra.rotation_matrix_around_axis(tangent[inode, :],
    #                                                            self.structural_twist[inode])
    #         normal[inode, :] = np.dot(rotation_mat, normal[inode, :])
    #         binormal[inode, :] = np.dot(rotation_mat, binormal[inode, :])

    return tangent, binormal, normal


def der_Cquat_by_v(q,v):
def get_triad(coordinates_def, frame_of_reference_delta, twist=None, n_nodes=3, ordering=np.array([0, 2, 1])):
    """
    Generates two unit vectors in body FoR that define the local FoR for
    a beam element. These vectors are calculated using `frame_of_reference_delta`
    :return:
    """
    # now, calculate tangent vector (and coefficients of the polynomial
    # fit just in case)
    tangent, polyfit = tangent_vector(
        coordinates_def,
        ordering)
    normal = np.zeros_like(tangent)
    binormal = np.zeros_like(tangent)

    # v_vector is the vector with origin the FoR node and delta
    # equals frame_of_reference_delta
    for inode in range(n_nodes):
        v_vector = frame_of_reference_delta[inode, :]
        normal[inode, :] = unit_vector(np.cross(
                                                tangent[inode, :],
                                                v_vector
                                                )
                                           )
        binormal[inode, :] = -unit_vector(np.cross(
                                                tangent[inode, :],
                                                normal[inode, :]
                                                        )
                                              )

    if twist is not None:
        raise NotImplementedError('Structural twist is not yet supported in algebra.get_triad, but it is in beamstructures.py')
    # # we apply twist now
    # for inode in range(self.n_nodes):
    #     if not self.structural_twist[inode] == 0.0:
    #         rotation_mat = algebra.rotation_matrix_around_axis(tangent[inode, :],
    #                                                            self.structural_twist[inode])
    #         normal[inode, :] = np.dot(rotation_mat, normal[inode, :])
    #         binormal[inode, :] = np.dot(rotation_mat, binormal[inode, :])

    return tangent, binormal, normal


    '''
    Being C=C(quat) the rotational matrix depending on the quaternion q and 
    defined as C=quat2rotation(q), the function returns the derivative, w.r.t. the
    quanternion components, of the vector dot(C,v), where v is a constant 
    vector.
    The elements of the resulting derivative matrix D are ordered such that:
        d(C*v) = D*d(q)
    where d(.) is a delta operator.
    '''

    vx,vy,vz=v
    q0,q1,q2,q3=q

    return 2.*np.array( [[ q0*vx + q2*vz - q3*vy, q1*vx + q2*vy + q3*vz, 
                                 q0*vz + q1*vy - q2*vx, -q0*vy + q1*vz - q3*vx], 
                         [ q0*vy - q1*vz + q3*vx, -q0*vz - q1*vy + q2*vx, 
                                 q1*vx + q2*vy + q3*vz,  q0*vx + q2*vz - q3*vy], 
                         [ q0*vz + q1*vy - q2*vx, q0*vy - q1*vz + q3*vx, 
                                -q0*vx - q2*vz + q3*vy, q1*vx + q2*vy + q3*vz]])



def der_CquatT_by_v(q,v):
    '''
    Being C=C(quat).T the projection matrix depending on the quaternion q and 
    defined as C=quat2rotation(q).T, the function returns the derivative, w.r.t. the
    quanternion components, of the vector dot(C,v), where v is a constant 
    vector.
    The elements of the resulting derivative matrix D are ordered such that:
        d(C*v) = D*d(q)
    where d(.) is a delta operator.
    '''

    vx,vy,vz=v
    q0,q1,q2,q3=q

    return 2.*np.array( [[ q0*vx + q2*vz - q3*vy, q1*vx + q2*vy + q3*vz, 
                                 q0*vz + q1*vy - q2*vx, -q0*vy + q1*vz - q3*vx], 
                         [q0*vy - q1*vz + q3*vx, -q0*vz - q1*vy + q2*vx, 
                                   q1*vx + q2*vy + q3*vz,q0*vx + q2*vz - q3*vy], 
                         [q0*vz + q1*vy - q2*vx, q0*vy - q1*vz + q3*vx,
                                -q0*vx - q2*vz + q3*vy, q1*vx + q2*vy + q3*vz]])



def der_Tan_by_xv(fv0,xv):
    '''
    Being fv0 a cartesian rotation vector and Tan the corresponding tangential 
    operator (computed through crv2tan(fv)), the function returns the derivative
    of dot(Tan,xv), where xv is a constant vector.

    The elements of the resulting derivative matrix D are ordered such that:
        d(Tan*xv) = D*d(fv)
    where d(.) is a delta operator.

    Note: the derivative expression has been derived symbolically and verified 
    by FDs. A more compact expression may be possible.
    '''

    f0=np.linalg.norm(fv0)
    sf0,cf0=np.sin(f0),np.cos(f0)

    fv0_x,fv0_y,fv0_z=fv0 
    xv_x,xv_y,xv_z=xv

    f0p2=f0**2
    f0p3=f0**3
    f0p4=f0**4  

    rs01=sf0/f0  
    rs03=sf0/f0p3 
    rc02=(cf0 - 1)/f0p2   
    rc04=(cf0 - 1)/f0p4

    Ts02=(1 - rs01)/f0p2
    Ts04=(1 - rs01)/f0p4

    # if f0<1e-8: rs01=1.0 # no need
    return np.array(
        [[xv_x*((-fv0_y**2 - fv0_z**2)*(-cf0*fv0_x/f0p2 + fv0_x*rs03)/f0p2 - 
            2*fv0_x*(1 - rs01)*(-fv0_y**2 - fv0_z**2)/f0p4) + xv_y*(fv0_x*fv0_y*(
                -cf0*fv0_x/f0p2 + fv0_x*rs03)/f0p2 + fv0_y*Ts02 + 
            fv0_x*fv0_z*rs03 - 2*fv0_x**2*fv0_y*Ts04 + 2*fv0_x*fv0_z*
            rc04) + xv_z*(fv0_x*fv0_z*(-cf0*fv0_x/f0p2 + fv0_x*rs03)/f0p2 + 
            fv0_z*Ts02 - fv0_x*fv0_y*rs03 - 2*fv0_x**2*fv0_z*Ts04 
            - 2*fv0_x*fv0_y*rc04),
            #
          xv_x*(-2*fv0_y*Ts02 + (-fv0_y**2 - fv0_z**2)*(-cf0*fv0_y/f0p2 + 
            fv0_y*rs03)/f0p2 - 2*fv0_y*(1 - rs01)*(-fv0_y**2 - fv0_z**2)/f0p4) + 
          xv_y*(fv0_x*fv0_y*(-cf0*fv0_y/f0p2 + fv0_y*rs03)/f0p2 + fv0_x*Ts02 + 
            fv0_y*fv0_z*rs03 - 2*fv0_x*fv0_y**2*Ts04 + 2*fv0_y*fv0_z*rc04) 
          + xv_z*(fv0_x*fv0_z*(-cf0*fv0_y/f0p2 + fv0_y*rs03)/f0p2 + rc02 - 
            fv0_y**2*rs03 - 2*fv0_x*fv0_y*fv0_z*Ts04 - 2*fv0_y**2*rc04),
          #
          xv_x*(-2*fv0_z*Ts02 + (-fv0_y**2 - fv0_z**2)*(-cf0*fv0_z/f0p2 
            + fv0_z*rs03)/f0p2 - 2*fv0_z*(1 - rs01)*(-fv0_y**2 - fv0_z**2)/f0p4) + 
          xv_y*(fv0_x*fv0_y*(-cf0*fv0_z/f0p2 + fv0_z*rs03)/f0p2 - rc02 
            + fv0_z**2*rs03 - 2*fv0_x*fv0_y*fv0_z*Ts04 + 2*fv0_z**2*rc04) 
          + xv_z*(fv0_x*fv0_z*(-cf0*fv0_z/f0p2 + fv0_z*rs03)/f0p2 + fv0_x*Ts02 
            - fv0_y*fv0_z*rs03 - 2*fv0_x*fv0_z**2*Ts04 - 2*fv0_y*fv0_z*rc04)],
         [xv_x*(fv0_x*fv0_y*(-cf0*fv0_x/f0p2 + fv0_x*rs03)/f0p2 + 
            fv0_y*Ts02 - fv0_x*fv0_z*rs03 - 2*fv0_x**2*fv0_y*Ts04 - 
            2*fv0_x*fv0_z*rc04) + xv_y*(-2*fv0_x*Ts02 + 
            (-fv0_x**2 - fv0_z**2)*(-cf0*fv0_x/f0p2 + fv0_x*rs03)/f0p2 
            - 2*fv0_x*(1 - rs01)*(-fv0_x**2 - fv0_z**2)/f0p4) + 
            xv_z*(fv0_y*fv0_z*(-cf0*fv0_x/f0p2 + fv0_x*rs03)/f0p2 - rc02 
                + fv0_x**2*rs03 + 2*fv0_x**2*rc04 - 2*fv0_x*fv0_y*fv0_z*Ts04),
          xv_x*(fv0_x*fv0_y*(-cf0*fv0_y/f0p2 + fv0_y*rs03)/f0p2 + 
            fv0_x*Ts02 - fv0_y*fv0_z*rs03 - 2*fv0_x*fv0_y**2*Ts04 
            - 2*fv0_y*fv0_z*rc04) + xv_y*((-fv0_x**2 - fv0_z**2)*(-cf0*fv0_y/f0p2 
                + fv0_y*rs03)/f0p2 - 2*fv0_y*(1 - rs01)*(-fv0_x**2 - fv0_z**2)/f0p4) 
            + xv_z*(fv0_y*fv0_z*(-cf0*fv0_y/f0p2 + fv0_y*rs03)/f0p2 + fv0_z*Ts02 
                + fv0_x*fv0_y*rs03 + 2*fv0_x*fv0_y*rc04 - 2*fv0_y**2*fv0_z*Ts04),
          xv_x*(fv0_x*fv0_y*(-cf0*fv0_z/f0p2 + fv0_z*rs03)/f0p2 + rc02 - fv0_z**2*rs03 
            - 2*fv0_x*fv0_y*fv0_z*Ts04 - 2*fv0_z**2*rc04) + xv_y*(-2*fv0_z*Ts02 
            + (-fv0_x**2 - fv0_z**2)*(-cf0*fv0_z/f0p2 + fv0_z*rs03)/f0p2 - 
            2*fv0_z*(1 - rs01)*(-fv0_x**2 - fv0_z**2)/f0p4) + xv_z*(fv0_y*fv0_z*(-cf0*fv0_z/f0p2 
                + fv0_z*rs03)/f0p2 + fv0_y*Ts02 + fv0_x*fv0_z*rs03 + 2*fv0_x*fv0_z*rc04 
            - 2*fv0_y*fv0_z**2*Ts04)],
         [xv_x*(fv0_x*fv0_z*(-cf0*fv0_x/f0p2 + fv0_x*rs03)/f0p2 + fv0_z*Ts02 
            + fv0_x*fv0_y*rs03 - 2*fv0_x**2*fv0_z*Ts04 + 2*fv0_x*fv0_y*rc04) 
         + xv_y*(fv0_y*fv0_z*(-cf0*fv0_x/f0p2 + fv0_x*rs03)/f0p2 + rc02 - fv0_x**2*rs03 
            - 2*fv0_x**2*rc04 - 2*fv0_x*fv0_y*fv0_z*Ts04) + xv_z*(-2*fv0_x*Ts02 
            + (-fv0_x**2 - fv0_y**2)*(-cf0*fv0_x/f0p2 + fv0_x*rs03)/f0p2 - 
            2*fv0_x*(1 - rs01)*(-fv0_x**2 - fv0_y**2)/f0p4),
          xv_x*(fv0_x*fv0_z*(-cf0*fv0_y/f0p2 + fv0_y*rs03)/f0p2 - rc02 + fv0_y**2*rs03 - 
            2*fv0_x*fv0_y*fv0_z*Ts04 + 2*fv0_y**2*rc04) + xv_y*(fv0_y*fv0_z*(-cf0*fv0_y/f0p2 
                + fv0_y*rs03)/f0p2 + fv0_z*Ts02 - fv0_x*fv0_y*rs03 - 2*fv0_x*fv0_y*rc04 
            - 2*fv0_y**2*fv0_z*Ts04) + xv_z*(-2*fv0_y*Ts02 + (-fv0_x**2 
                - fv0_y**2)*(-cf0*fv0_y/f0p2 + fv0_y*rs03)/f0p2 - 2*fv0_y*(1 - rs01)*(-fv0_x**2 
                - fv0_y**2)/f0p4),
          xv_x*(fv0_x*fv0_z*(-cf0*fv0_z/f0p2 + fv0_z*rs03)/f0p2 + fv0_x*Ts02 + 
            fv0_y*fv0_z*rs03 - 2*fv0_x*fv0_z**2*Ts04 + 2*fv0_y*fv0_z*rc04) + 
          xv_y*(fv0_y*fv0_z*(-cf0*fv0_z/f0p2 + fv0_z*rs03)/f0p2 + fv0_y*Ts02 
            - fv0_x*fv0_z*rs03 - 2*fv0_x*fv0_z*rc04 - 2*fv0_y*fv0_z**2*Ts04) + 
          xv_z*((-fv0_x**2 - fv0_y**2)*(-cf0*fv0_z/f0p2 + fv0_z*rs03)/f0p2 - 
            2*fv0_z*(1 - rs01)*(-fv0_x**2 - fv0_y**2)/f0p4)]])
    # end der_Tan_by_xv

def der_TanT_by_xv(fv0,xv):
    '''
    Being fv0 a cartesian rotation vector and Tan the corresponding tangential
    operator (computed through crv2tan(fv)), the function returns the derivative
    of dot(Tan^T,xv), where xv is a constant vector.

    The elements of the resulting derivative matrix D are ordered such that:
        d(Tan^T*xv) = D*d(fv)
    where d(.) is a delta operator.

    Note: the derivative expression has been derived symbolically and verified
    by FDs. A more compact expression may be possible.
    '''

    # Renaming variabes for clarity
    px = fv0[0]
    py = fv0[1]
    pz = fv0[2]

    vx = xv[0]
    vy = xv[1]
    vz = xv[2]

    # Defining useful functions
    eps = 1e-15
    f0=np.linalg.norm(fv0)
    if f0 < eps:
        f1 = -1.0/2.0
        f2 = 1.0/6.0
        g1 = -1.0/12.0
        g2 = 0.0 # TODO: check this
    else:
        f1 = (np.cos(f0)-1.0)/f0**2.0
        f2 = (1.0-np.sin(f0)/f0)/f0**2.0
        g1 = (f0*np.sin(f0)+2.0*(np.cos(f0)-1.0))/f0**4.0
        g2 = (2.0/f0**4 + np.cos(f0)/f0**4 - 3.0*np.sin(f0)/f0**5)

    # Computing the derivatives of the functions
    df1dpx = -1.0*px*g1
    df1dpy = -1.0*py*g1
    df1dpz = -1.0*pz*g1

    df2dpx = -1.0*px*g2
    df2dpy = -1.0*py*g2
    df2dpz = -1.0*pz*g2

    # Compute the output matrix
    der_TanT_by_xv = np.zeros((3,3),)

    # First column (derivatives with psi_x)
    der_TanT_by_xv[0,0] = -1.0*df2dpx*(py**2+pz**2)*vx + df1dpx*pz*vy + df2dpx*px*py*vy + f2*py*vy - df1dpx*py*vz + df2dpx*px*pz*vz + f2*pz*vz
    der_TanT_by_xv[1,0] = -1.0*df1dpx*pz*vx + df2dpx*px*py*vx + f2*py*vx - df2dpx*px**2*vy - 2.0*f2*px*vy - df2dpx*pz**2*vy + df1dpx*px*vz+f1*vz + df2dpx*py*pz*vz
    der_TanT_by_xv[2,0] = df1dpx*py*vx + df2dpx*px*pz*vx + f2*pz*vx - df1dpx*px*vy -f1*vy + df2dpx*py*pz*vy - df2dpx*px**2*vz - 2.0*f2*px*vz - df2dpx*py**2*vz

    # Second column (derivatives with psi_y)
    der_TanT_by_xv[0,1] = -df2dpy*py**2*vx -f2*2*py*vx - df2dpy*pz**2*vx + df1dpy*pz*vy + df2dpy*px*py*vy +f2*px*vy - df1dpy*py*vz - f1*vz + df2dpy*px*pz*vz
    der_TanT_by_xv[1,1] = -df1dpy*pz*vx + df2dpy*px*py*vx + f2*px*vx - df2dpy*px**2*vy - df2dpy*pz**2*vy + df1dpy*px*vz + df2dpy*py*pz*vz + f2*pz*vz
    der_TanT_by_xv[2,1] = df1dpy*py*vx + f1*vx + df2dpy*px*pz*vx - df1dpy*px*vy + df2dpy*py*pz*vy + f2*pz*vy - df2dpy*px**2*vz - df2dpy*py**2*vz - 2.0*f2*py*vz

    # Second column (derivatives with psi_z)
    der_TanT_by_xv[0,2] = -df2dpz*py**2*vx - df2dpz*pz**2*vx - 2.0*f2*pz*vx + df1dpz*pz*vy + f1*vy + df2dpz*px*py*vy - df1dpz*py*vz + df2dpz*px*pz*vz + f2*px*vz
    der_TanT_by_xv[1,2] = -df1dpz*pz*vx - f1*vx + df2dpz*px*py*vx - df2dpz*px**2*vy - df2dpz*pz**2*vy - 2.0*f2*pz*vy + df1dpz*px*vz + df2dpz*py*pz*vz + f2*py*vz
    der_TanT_by_xv[2,2] = df1dpz*py*vx + df2dpz*px*pz*vx + f2*px*vx - df1dpz*px*vy + df2dpz*py*pz*vy + f2*py*vy - df2dpz*px**2*vz - df2dpz*py**2*vz

    return der_TanT_by_xv
