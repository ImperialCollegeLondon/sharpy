"""
Algebra package

Extensive library with geometrical and algebraic operations

Note:
    Tests can be found in ``tests/utils/algebra_test``
"""

import numpy as np
import scipy.linalg
from warnings import warn

#######
# functions for back compatibility
def quat2rot(quat):
    warn('quat2rot(quat) is obsolete! Use quat2rotation(quat).T instead!', stacklevel=2)
    return quat2rotation(quat).T


def crv2rot(psi):
    warn('crv2rot(psi) is obsolete! Use crv2rotation(psi) instead!', stacklevel=2)
    return crv2rotation(psi)


def rot2crv(rot):
    warn('rot2crv(rot) is obsolete! Use rotation2crv(rot.T) instead!', stacklevel=2)
    return rotation2crv(rot.T)


def triad2rot(xb,yb,zb):
    warn('triad2rot(xb,yb,zb) is obsolete! Use triad2rotation(xb,yb,zb).T instead!', stacklevel=2)
    return triad2rotation(xb,yb,zb).T


def mat2quat(rot):
    """
    Rotation matrix to quaternion function.

    Warnings:
        This function is deprecated and now longer supported. Please use ``algebra.rotation2quat(rot.T)`` instead.

    Args:
        rot: Rotation matrix

    Returns:
        np.array: equivalent quaternion
    """
    warn('mat2quat(rot) is obsolete! Use rotation2quat(rot.T) instead!', stacklevel=2)

    return rotation2quat(rot.T)
#######


def tangent_vector(in_coord, ordering=None):
    r"""
    Tangent vector calculation for 2+ noded elements.

    Calculates the tangent vector interpolating every dimension
    separately. It uses a (n_nodes - 1) degree polynomial, and the
    differentiation is analytical.

    Calculation method:

        1. A n_nodes-1 polynomial is fitted through the nodes per dimension.
        2. Those polynomials are analytically differentiated with respect to the node index
        3. The tangent vector is given by:

        .. math::

            \vec{t} = \frac{s_x'\vec{i} + s_y'\vec{j} + s_z'\vec{k}}{\left| s_x'\vec{i} + s_y'\vec{j} + s_z'\vec{k}\right|}


        where :math:`'` notes the differentiation with respect to the index number


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

    for inode in range(n_nodes):
        if np.dot(tangent[inode, :], fake_tangent[inode, :]) < 0:
            tangent[inode, :] *= -1

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
    r"""
    Transforms the input vector into a unit vector

    .. math:: \mathbf{\hat{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}

    Args:
        vector (np.array): vector to normalise

    Returns:
        np.array: unit vector

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
    r"""
    Returns a skew symmetric matrix such that

    .. math:: \boldsymbol{v} \times \boldsymbol{u} = \tilde{\boldsymbol{v}}{\boldsymbol{u}

    where

    .. math:: \tilde{\boldsymbol{v}} = \begin{bmatrix}
        0 & -v_z & v_y \\
        v_z & 0 & -v_x \\
        -v_y & v_x & 0 \end{bmatrix}.

    Args:
        vector (np.ndarray): 3-dimensional vector

    Returns:
        np.array: Skew-symmetric matrix.

    """
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


def quadskew(vector):
    """
    Generates the matrix needed to obtain the quaternion in the following time step
    through integration of the FoR angular velocity.


    Args:
        vector (np.array): FoR angular velocity

    Notes:
        The angular velocity is assumed to be constant in the time interval
        Equivalent to lib_xbeam function
        Quaternion ODE to compute orientation of body-fixed frame a
        See Shearer and Cesnik (2007) for definition

    Returns:
        np.array: matrix
    """
    if not vector.size == 3:
        raise ValueError('The input vector is not 3D')

    matrix = np.zeros((4, 4))
    matrix[0,1:4] = vector
    matrix[1:4,0] = -vector
    matrix[1:4,1:4] = skew(vector)
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

def panel_area(A, B, C, D):
    """
    Calculates the area of a quadrilateral panel from the corner 
    points A,B,C, and D using Bertschneider's formula
    
    Args:
        A (np.ndarray): Coordinates of point 1
        B (np.ndarray): Coordinates of point 2
        C (np.ndarray): Coordinates of point 3
        D (np.ndarray): Coordinates of point 4

    Returns:
        float: Area of quadrilateral panel
    """      
    Theta_1 = angle_between_vectors(A-B, A-D)
    Theta_2 = angle_between_vectors(B-C, B-D)
    a = np.linalg.norm(D-A)
    b = np.linalg.norm(A-B)
    c = np.linalg.norm(B-C)
    d = np.linalg.norm(C-D)
    s = (a+b+c+d)/2
    area = np.sqrt((s-a)*(s-b)*(s-c)*(s-d)-a*b*c*d*np.cos(0.5*(Theta_1+Theta_2))**2)
    return area
    
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
    r"""
    Given a rotation matrix :math:`C^{AB}` rotating the frame A onto B, the function returns
    the minimal "positive angle" quaternion representing this rotation, where the quaternion, :math:`\vec{\chi}` is
    defined as:

        .. math:: \vec{\chi}=
            \left[\cos\left(\frac{\psi}{2}\right),\,
            \sin\left(\frac{\psi}{2}\right)\mathbf{\hat{n}}\right]

    Args:
        Cab (np.array): rotation matrix :math:`C^{AB}` from frame A to B

    Returns:
        np.array: equivalent quaternion :math:`\vec{\chi}`

    Notes:
        This is the inverse of ``algebra.quat2rotation`` for Cartesian rotation vectors
        associated to rotations in the range :math:`[-\pi,\pi]`, i.e.:

            ``fv == algebra.rotation2crv(algebra.crv2rotation(fv))``

        where ``fv`` represents the Cartesian Rotation Vector, :math:`\vec{\psi}` defined as:

            .. math:: \vec{\psi} = \psi\,\mathbf{\hat{n}}

        such that :math:`\mathbf{\hat{n}}` is a unit vector and the scalar :math:`\psi` is in the range
        :math:`[-\pi,\,\pi]`.

    """

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
    r"""
    Given a quaternion, :math:`\vec{\chi}`, associated to a rotation of angle :math:`\psi`
    about an axis :math:`\mathbf{\hat{n}}`, the function "bounds" the quaternion,
    i.e. sets the rotation axis :math:`\mathbf{\hat{n}}` such that
    :math:`\psi` in :math:`[-\pi,\pi]`.

    Notes:
        As quaternions are defined as:

            .. math:: \vec{\chi}=
                \left[\cos\left(\frac{\psi}{2}\right),\,
                \sin\left(\frac{\psi}{2}\right)\mathbf{\hat{n}}\right]

        this is equivalent to enforcing :math:`\chi_0\ge0`.

    Args:
        quat (np.array): quaternion to bound

    Returns:
        np.array: bounded quaternion

    """
    if quat[0] < 0:
        quat *= -1.
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
    r"""
    Converts a Cartesian rotation vector,

        .. math:: \vec{\psi} = \psi\,\mathbf{\hat{n}}

    into a "minimal rotation" quaternion, i.e. being the quaternion, :math:`\vec{\chi}`, defined as:

        .. math:: \vec{\chi}=
            \left[\cos\left(\frac{\psi}{2}\right),\,
            \sin\left(\frac{\psi}{2}\right)\mathbf{\hat{n}}\right]

    the rotation axis, :math:`\mathbf{\hat{n}}` is such that the
    rotation angle, :math:`\psi`, is in :math:`[-\pi,\,\pi]` or,
    equivalently, :math:`\chi_0\ge0`.

    Args:
        psi (np.array): Cartesian Rotation Vector, CRV: :math:`\vec{\psi} = \psi\,\mathbf{\hat{n}}`.

    Returns:
        np.array: equivalent quaternion :math:`\vec{\chi}`

    """

    # minimise crv rotation
    psi_new = crv_bounds(psi)

    fi = np.linalg.norm(psi_new)
    if fi > 1e-15:
        nv = psi_new / fi
    else:
        nv = psi_new

    quat = np.zeros((4,))
    quat[0] = np.cos(.5 * fi)
    quat[1:] = np.sin(.5 * fi) * nv

    return quat


def crv_bounds(crv_ini):
    r"""
    Forces the Cartesian rotation vector norm, :math:`\|\vec{\psi}\|`, to be in the range
    :math:`[-\pi,\pi]`, i.e. determines the rotation axis orientation, :math:`\mathbf{\hat{n}}`,
    so as to ensure "minimal rotation".

    Args:
        crv_ini (np.array): Cartesian rotation vector, :math:`\vec{\psi}`

    Returns:
        np.array: modified and bounded, equivalent Cartesian rotation vector

    """

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
    # return crv_ini


def triad2crv(xb, yb, zb):
    return rotation2crv(triad2rotation(xb, yb, zb))


def crv2triad(psi):
    rot_matrix = crv2rotation(psi)
    return rot_matrix[:, 0], rot_matrix[:, 1], rot_matrix[:, 2]


def crv2rotation(psi):
    r"""
    Given a Cartesian rotation vector, :math:`\boldsymbol{\Psi}`, the function produces the rotation
    matrix required to rotate a vector according to :math:`\boldsymbol{\Psi}`.

    The rotation matrix is given by

    .. math::
        \mathbf{R} = \mathbf{I} + \frac{\sin||\boldsymbol{\Psi}||}{||\boldsymbol{\Psi}||} \tilde{\boldsymbol{\Psi}} +
        \frac{1-\cos{||\boldsymbol{\Psi}||}}{||\boldsymbol{\Psi}||^2}\tilde{\boldsymbol{\Psi}} \tilde{\boldsymbol{\Psi}}

    To avoid the singularity when :math:`||\boldsymbol{\Psi}||=0`, the series expansion is used

    .. math:: \mathbf{R} = \mathbf{I} + \tilde{\boldsymbol{\Psi}} + \frac{1}{2!}\tilde{\boldsymbol{\Psi}}^2.


    Args:
        psi (np.array): Cartesian rotation vector :math:`\boldsymbol{\Psi}`.

    Returns:
        np.array: equivalent rotation matrix

    References:
        Geradin and Cardona, Flexible Multibody Dynamics: A finite element approach. Chapter 4

    """

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
    r"""
    Given a rotation matrix :math:`C^{AB}` rotating the frame A onto B, the function returns
    the minimal size Cartesian rotation vector, :math:`\vec{\psi}` representing this rotation.

    Args:
        Cab (np.array): rotation matrix :math:`C^{AB}`

    Returns:
        np.array: equivalent Cartesian rotation vector, :math:`\vec{\psi}`.

    Notes:
        this is the inverse of ``algebra.crv2rotation`` for Cartesian rotation vectors
        associated to rotations in the range :math:`[-\pi,\,\pi]`, i.e.:

            ``fv == algebra.rotation2crv(algebra.crv2rotation(fv))``

        for each Cartesian rotation vector of the form :math:`\vec{\psi} = \psi\,\mathbf{\hat{n}}`
        represented as ``fv=a*nv`` such that ``nv`` is a unit vector and the scalar ``a`` is in the
        range :math:`[-\pi,\,\pi]`.

    """

    if np.linalg.norm(Cab) < 1e-6:
        raise AttributeError('Element Vector V is not orthogonal to reference line (51105)')

    quat = rotation2quat(Cab)
    psi = quat2crv(quat)

    if np.linalg.norm(Cab) < 1.0e-15:
        psi[0] = Cab[1, 2]
        psi[1] = Cab[2, 0]
        psi[2] = Cab[0, 1]

    return crv_bounds(psi)


def crv2tan(psi):
    r"""
    Returns the tangential operator, :math:`\mathbf{T}(\boldsymbol{\Psi})`, that is a function of
    the Cartesian Rotation Vector, :math:`\boldsymbol{\Psi}`.

    .. math::

        \boldsymbol{T}(\boldsymbol{\Psi}) =
        \mathbf{I} +
        \left(\frac{\cos ||\boldsymbol{\Psi}|| - 1}{||\boldsymbol{\Psi}||^2}\right)\tilde{\boldsymbol{\Psi}}
        + \left(1 - \frac{\sin||\boldsymbol{\Psi}||}{||\boldsymbol{\Psi}||}\right)
        \frac{\tilde{\boldsymbol{\Psi}}\tilde{\boldsymbol{\Psi}}}{||\boldsymbol{\Psi}||^2}

    When the norm of the CRV approaches 0, the series expansion expression is used in-lieu of the above expression

    .. math::

        \boldsymbol{T}(\boldsymbol{\Psi}) =
        \mathbf{I}
        -\frac{1}{2!}\tilde{\boldsymbol{\Psi}} + \frac{1}{3!}\tilde{\boldsymbol{\Psi}}^2

    Args:
        psi (np.array): Cartesian Rotation Vector, :math:`\boldsymbol{\Psi}`.

    Returns:
        np.array: Tangential operator

    References:
        Geradin and Cardona. Flexible Multibody Dynamics: A Finite Element Approach. Chapter 4.
    """

    norm_psi = np.linalg.norm(psi)
    psi_skew = skew(psi)

    eps = 1e-8
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
    r"""Calculate rotation matrix based on quaternions.

    If B is a FoR obtained rotating a FoR A by an angle :math:`\phi` about an axis :math:`\mathbf{n}`
    (recall :math:`\mathbf{n}` will be invariant during the rotation), and :math:`\mathbf{q}` is the related
    quaternion, :math:`\mathbf{q}(\phi,\mathbf{n})`, the function will return the matrix :math:`C^{AB}` such that:

        - :math:`C^{AB}` rotates FoR A onto FoR B.

        - :math:`C^{AB}` transforms the coordinates of a vector defined in B component to
          A components i.e. :math:`\mathbf{v}^A = C^{AB}(\mathbf{q})\mathbf{v}^B`.

    .. math::
        C^{AB}(\mathbf{q}) = \begin{pmatrix}
            q_0^2 + q_1^2 - q_2^2 -q_3^2 & 2(q_1 q_2 - q_0 q_3) & 2(q_1 q_3 + q_0 q_2) \\
            2(q_1 q_2 + q_0 q_3) & q_0^2 - q_1^2 + q_2^2 - q_3^2 & 2(q_2 q_3 - q_0 q_1) \\
            2(q_1 q_3 - q_0 q_2) & 2(q_2 q_3 + q_0 q_1) & q_0^2 -q_1^2 -q_2^2 +q_3^2
            \end{pmatrix}

    Notes:
        The inverse rotation is defined as the transpose of the matrix :math:`C^{BA} = C^{{AB}^T}`.

        In typical SHARPy applications, the quaternion relation between the A and G frames is expressed
        as :math:`C^{GA}(\mathbf{q})`, and in the context of this function it corresponds to:

        >>> C_ga = quat2rotation(q1)
        >>> C_ag = quat2rotation.T(q1)

    Args:
        q (np.ndarray): Quaternion :math:`\mathbf{q}(\phi, \mathbf{n})`.

    Returns:
        np.ndarray: :math:`C^{AB}` rotation matrix from FoR B to FoR A.

    References:
        Stevens, L. Aircraft Control and Simulation. 1985. pg 41
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
    r"""

    Rotation matrix about the x axis by the input angle :math:`\Phi`

    .. math::

        \mathbf{\tau}_x = \begin{bmatrix}
            1 & 0 & 0 \\
            0 & \cos(\Phi) & -\sin(\Phi) \\
            0 & \sin(\Phi) & \cos(\Phi)
        \end{bmatrix}


    Args:
        angle (float): angle of rotation in radians about the x axis

    Returns:
        np.array: 3x3 rotation matrix about the x axis

    """
    c = np.cos(angle)
    s = np.sin(angle)
    mat = np.zeros((3, 3))
    mat[0, :] = [1.0, 0.0, 0.0]
    mat[1, :] = [0.0,   c,  -s]
    mat[2, :] = [0.0,   s,   c]
    return mat


def rotation3d_y(angle):
    r"""
    Rotation matrix about the y axis by the input angle :math:`\Theta`

    .. math::

        \mathbf{\tau}_y = \begin{bmatrix}
            \cos(\Theta) & 0 & -\sin(\Theta) \\
            0 & 1 & 0 \\
            \sin(\Theta) & 0 & \cos(\Theta)
        \end{bmatrix}


    Args:
        angle (float): angle of rotation in radians about the y axis

    Returns:
        np.array: 3x3 rotation matrix about the y axis

    """

    c = np.cos(angle)
    s = np.sin(angle)
    mat = np.zeros((3, 3))
    mat[0, :] = [c, 0.0, s]
    mat[1, :] = [0.0, 1.0, 0.0]
    mat[2, :] = [-s, 0.0,  c]
    return mat


def rotation3d_z(angle):
    r"""
    Rotation matrix about the z axis by the input angle :math:`\Psi`

    .. math::
        \mathbf{\tau}_z = \begin{bmatrix}
            \cos(\Psi) & -\sin(\Psi) & 0 \\
            \sin(\Psi) & \cos(\Psi) & 0 \\
            0 & 0 & 1
        \end{bmatrix}

    Args:
        angle (float): angle of rotation in radians about the z axis

    Returns:
        np.array: 3x3 rotation matrix about the z axis

    """

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
    r"""

    Transforms Euler angles (roll, pitch and yaw :math:`\Phi, \Theta, \Psi`) into a 3x3 rotation matrix describing
    that rotates a vector in yaw pitch, and roll.

    The rotations are performed successively, first in yaw, then in pitch and finally in roll.

    .. math::

        \mathbf{T}_{AG} = \mathbf{\tau}_x(\Phi) \mathbf{\tau}_y(\Theta) \mathbf{\tau}_z(\Psi)


    where :math:`\mathbf{\tau}` represents the rotation about the subscripted axis.

    Args:
        euler (np.array): 1x3 array with the Euler angles in the form ``[roll, pitch, yaw]`` in radians

    Returns:
        np.array: 3x3 transformation matrix describing the rotation by the input Euler angles.

    """

    # rot = rotation3d_z(euler[2])
    # rot = np.dot(rotation3d_y(euler[1]), rot)
    # rot = np.dot(rotation3d_x(euler[0]), rot)
    rot = rotation3d_z(euler[2]).dot(rotation3d_y(euler[1]).dot(rotation3d_x(euler[0])))
    return rot


def euler2quat(euler):
    """

    Args:
        euler: Euler angles

    Returns:
        np.ndarray: Equivalent quaternion.
    """
    euler_rot = euler2rot(euler)  # this is Cag
    quat = rotation2quat(euler_rot)
    return quat


def quat2euler(quat):
    r"""
    Quaternion to Euler angles transformation.

    Transforms a normalised quaternion :math:`\chi\longrightarrow[\phi, \theta, \psi]` to roll, pitch and yaw angles
    respectively.

    The transformation is valid away from the singularity present at:

    .. math:: \Delta = \frac{1}{2}

    where :math:`\Delta = q_0 q_2 - q_1 q_3`.

    The transformation is carried out as follows:

    .. math::
        \psi &= \arctan{\left(2\frac{q_0q_3+q_1q_2}{1-2(q_2^2+q_3^2)}\right)} \\
        \theta &= \arcsin(2\Delta) \\
        \phi &= \arctan\left(2\frac{q_0q_1 + q_2q_3}{1-2(q_1^2+q_2^2)}\right)

    Args:
        quat (np.ndarray): Normalised quaternion.

    Returns:
        np.ndarray: Array containing the Euler angles :math:`[\phi, \theta, \psi]` for roll, pitch and yaw, respectively.

    References:
        Blanco, J.L. - A tutorial on SE(3) transformation parameterizations and on-manifold optimization. Technical
        Report 012010. ETS Ingenieria Informatica. Universidad de Malaga. 2013.
    """

    assert np.abs(np.linalg.norm(quat)-1.0) < 1.e6, 'Input quaternion is not normalised'

    q0 = quat[0]
    q1 = quat[1]
    q2 = quat[2]
    q3 = quat[3]

    delta = quat[0]*quat[2] - quat[1]*quat[3]

    if np.abs(delta) > 0.9 * 0.5:
        warn('Warning, approaching singularity. Delta {:.3f} for singularity at Delta=0.5'.format(np.abs(delta)))

    yaw = np.arctan2(2*(q0*q3+q1*q2), (1-2*(q2**2+q3**2)))
    pitch = np.arcsin(2*delta)
    roll = np.arctan2(2*(q0*q1+q2*q3), (1-2*(q1**2+q2**2)))

    return np.array([roll, pitch, yaw])


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
    """
    Being C=C(quat) the rotational matrix depending on the quaternion q and
    defined as C=quat2rotation(q), the function returns the derivative, w.r.t. the
    quanternion components, of the vector dot(C,v), where v is a constant
    vector.

    The elements of the resulting derivative matrix D are ordered such that:

    .. math::   d(C*v) = D*d(q)

    where :math:`d(.)` is a delta operator.
    """

    vx,vy,vz=v
    q0,q1,q2,q3=q

    return 2.*np.array( [[ q0*vx + q2*vz - q3*vy, q1*vx + q2*vy + q3*vz,
                                 q0*vz + q1*vy - q2*vx, -q0*vy + q1*vz - q3*vx],
                         [ q0*vy - q1*vz + q3*vx, -q0*vz - q1*vy + q2*vx,
                                 q1*vx + q2*vy + q3*vz,  q0*vx + q2*vz - q3*vy],
                         [ q0*vz + q1*vy - q2*vx, q0*vy - q1*vz + q3*vx,
                                -q0*vx - q2*vz + q3*vy, q1*vx + q2*vy + q3*vz]])



def der_CquatT_by_v(q,v):
    r"""
    Returns the derivative with respect to quaternion components of a projection matrix times a constant vector.

    Being :math:`\mathbf{C}=\mathbf{R}(\boldsymbol{\chi})^\top` the projection matrix depending on the quaternion
    :math:`\boldsymbol{\chi}` and obtained through the function
    defined as ``C=quat2rotation(q).T``, this function returns the derivative with respect to the
    quaternion components, of the vector :math:`(\mathbf{C\cdot v})`, where :math:`\mathbf{v}` is a constant
    vector.

    The derivative operation is defined as:

    .. math::  \delta(\mathbf{C}\cdot \mathbf{v}) =
        \frac{\partial}{\partial\boldsymbol{\chi}}\left(\mathbf{C\cdot v}\right)\delta\boldsymbol{\chi}

    where, for simplicity, we define

    .. math:: \mathbf{D} =
        \frac{\partial}{\partial\boldsymbol{\chi}}\left(\mathbf{C\cdot v}\right) \in \mathbb{R}^{3\times4}

    and :math:`\delta(\bullet)` is a delta operator.

    The members of :math:`\mathbf{D}` are the following:

    .. math::
        \mathbf{D}_{11} &= 2 (q_0 v_x - q_2 v_z + q_3 v_y)\\
        \mathbf{D}_{12} &= 2 (q_1 v_x - q_2 v_y + q_3 v_z)\\
        \mathbf{D}_{13} &= 2 (-q_0 v_z + q_1 v_y - q_2 v_x)\\
        \mathbf{D}_{14} &= 2 (q_0 v_y + q_1 v_z - q_3 v_x)

    .. math::
        \mathbf{D}_{21} &= 2 (q_0 v_y + q_1 v_z - q_3 v_x)\\
        \mathbf{D}_{22} &= 2 (q_0 v_z - q_1 v_y + q_2 v_x)\\
        \mathbf{D}_{23} &= 2 (q_1 v_x + q_2 v_y + q_3 v_z)\\
        \mathbf{D}_{24} &= 2 (-q_0 v_x + q_2 v_z - q_3 v_y)

    .. math::
        \mathbf{D}_{31} &= 2 (q_0 v_z - q_1 v_y + q_2 v_x)\\
        \mathbf{D}_{32} &= 2 (-q_0 v_y - q_1 v_z + q_3 v_x)\\
        \mathbf{D}_{33} &= 2 (q_0 v_x - q_2 v_z + q_3 v_y)\\
        \mathbf{D}_{34} &= 2 (q_1 v_x + q_2 v_y + q_3 v_z)\\

    Returns:
        np.array: :math:`\mathbf{D}` matrix.
    """

    vx,vy,vz=v
    q0,q1,q2,q3=q

    return 2.*np.array( [[ q0*vx - q2*vz + q3*vy, q1*vx + q2*vy + q3*vz,
                                 - q0*vz + q1*vy - q2*vx, q0*vy + q1*vz - q3*vx],
                         [q0*vy + q1*vz - q3*vx, q0*vz - q1*vy + q2*vx,
                                   q1*vx + q2*vy + q3*vz,-q0*vx + q2*vz - q3*vy],
                         [q0*vz - q1*vy + q2*vx, -q0*vy - q1*vz + q3*vx,
                                q0*vx - q2*vz + q3*vy, q1*vx + q2*vy + q3*vz]])


def der_Tan_by_xv(fv0,xv):
    """
    Being fv0 a cartesian rotation vector and Tan the corresponding tangential
    operator (computed through crv2tan(fv)), the function returns the derivative
    of dot(Tan,xv), where xv is a constant vector.

    The elements of the resulting derivative matrix D are ordered such that:

    .. math::    d(Tan*xv) = D*d(fv)

    where :math:`d(.)` is a delta operator.

    Note:
        The derivative expression has been derived symbolically and verified
        by FDs. A more compact expression may be possible.
    """

    f0 = np.linalg.norm(fv0)
    sf0, cf0 = np.sin(f0), np.cos(f0)

    fv0_x, fv0_y, fv0_z = fv0
    xv_x, xv_y, xv_z = xv

    f0p2 = f0 ** 2
    f0p3 = f0 ** 3
    f0p4 = f0 ** 4

    rs01 = sf0 / f0
    rs03 = sf0 / f0p3
    rc02 = (cf0 - 1) / f0p2
    rc04 = (cf0 - 1) / f0p4

    Ts02 = (1 - rs01) / f0p2
    Ts04 = (1 - rs01) / f0p4

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
    """
    Being fv0 a cartesian rotation vector and Tan the corresponding tangential
    operator (computed through crv2tan(fv)), the function returns the derivative
    of dot(Tan^T,xv), where xv is a constant vector.

    The elements of the resulting derivative matrix D are ordered such that:

    .. math::     d(Tan^T*xv) = D*d(fv)

    where :math:`d(.)` is a delta operator.

    Note:
        The derivative expression has been derived symbolically and verified
        by FDs. A more compact expression may be possible.
    """

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


def der_Ccrv_by_v(fv0,v):
    r"""
    Being C=C(fv0) the rotational matrix depending on the Cartesian rotation
    vector fv0 and defined as C=crv2rotation(fv0), the function returns the
    derivative, w.r.t. the CRV components, of the vector dot(C,v), where v is a
    constant vector.

    The elements of the resulting derivative matrix D are ordered such that:

    .. math:: d(C*v) = D*d(fv0)

    where :math:`d(.)` is a delta operator.
    """

    Cab0=crv2rotation(fv0)
    T0=crv2tan(fv0)
    vskew=skew(v)

    return -np.dot(Cab0,np.dot(vskew,T0))


def der_CcrvT_by_v(fv0,v):
    """
    Being C=C(fv0) the rotation matrix depending on the Cartesian rotation
    vector fv0 and defined as C=crv2rotation(fv0), the function returns the
    derivative, w.r.t. the CRV components, of the vector dot(C.T,v), where v is
    a constant vector.

    The elements of the resulting derivative matrix D are ordered such that:

    .. math::    d(C.T*v) = D*d(fv0)

    where :math:`d(.)` is a delta operator.
    """

    Cba0=crv2rotation(fv0).T
    T0=crv2tan(fv0)

    return np.dot( skew( np.dot(Cba0,v) ),T0)


def der_quat_wrt_crv(quat0):
    """
    Provides change of quaternion, dquat, due to elementary rotation, dcrv,
    expressed as a 3 components Cartesian rotation vector such that

    .. math::    C(quat + dquat) = C(quat0)C(dw)

    where C are rotation matrices.

    Examples:
        Assume 3 FoRs, G, A and B where:
            - G is the initial FoR
            - quat0 defines te rotation required to obtain A from G, namely:
              Cga=quat2rotation(quat0)
            - dcrv is an inifinitesimal Cartesian rotation vector, defined in A
              components, which describes an infinitesimal rotation A -> B, namely:

              ..math ::      Cab=crv2rotation(dcrv)

            - The total rotation G -> B is:
                Cga = Cga * Cab
            - As dcrv -> 0, Cga is equal to:

              .. math::  algebra.quat2rotation(quat0 + dquat),

              where dquat is the output of this function.
    """

    Der = np.zeros((4, 3))
    Der[0, :] = -0.5 * quat0[1:]
    Der[1:, :] = -0.5 * (-quat0[0] * np.eye(3) - skew(quat0[1:]))
    return Der


def der_Ceuler_by_v(euler, v):
    r"""
    Provides the derivative of the product between the rotation matrix :math:`C^{AG}(\mathbf{\Theta})` and a constant
    vector, :math:`\mathbf{v}`, with respect to the Euler angles, :math:`\mathbf{\Theta}=[\phi,\theta,\psi]^T`:

    .. math::
        \frac{\partial}{\partial\Theta}(C^{AG}(\Theta)\mathbf{v}^G) = \frac{\partial \mathbf{f}}{\partial\mathbf{\Theta}}

    where :math:`\frac{\partial \mathbf{f}}{\partial\mathbf{\Theta}}` is the resulting 3 by 3 matrix.

    Being :math:`C^{AG}(\Theta)` the rotation matrix from the G frame to the A frame in terms of the Euler angles
    :math:`\Theta` as:

    .. math::
        C^{AG}(\Theta) = \begin{bmatrix}
        \cos\theta\cos\psi & -\cos\theta\sin\psi & \sin\theta \\
        \cos\phi\sin\psi + \sin\phi\sin\theta\cos\psi & \cos\phi\cos\psi - \sin\phi\sin\theta\sin\psi & -\sin\phi\cos\theta \\
        \sin\phi\sin\psi - \cos\phi\sin\theta\cos\psi & \sin\phi\cos\psi + \cos\phi\sin\theta\sin\psi & \cos\phi\cos\theta
        \end{bmatrix}

    the components of the derivative at hand are the following, where
    :math:`f_{1\theta} = \frac{\partial \mathbf{f}_1}{\partial\theta}`.

    .. math::
        f_{1\phi} =&0 \\
        f_{1\theta} = &-v_1\sin\theta\cos\psi \\
        &+v_2\sin\theta\sin\psi \\
        &+v_3\cos\theta \\
        f_{1\psi} = &-v_1\cos\theta\sin\psi \\
        &- v_2\cos\theta\cos\psi

    .. math::
        f_{2\phi} = &+v_1(-\sin\phi\sin\psi + \cos\phi\sin\theta\cos\psi) + \\
        &+v_2(-\sin\phi\cos\psi - \cos\phi\sin\theta\sin\psi) + \\
        &+v_3(-\cos\phi\cos\theta)\\
        f_{2\theta} = &+v_1(\sin\phi\cos\theta\cos\psi) + \\
        &+v_2(-\sin\phi\cos\theta\sin\psi) +\\
        &+v_3(\sin\phi\sin\theta) \\
        f_{2\psi} = &+v_1(\cos\phi\cos\psi - \sin\phi\sin\theta\sin\psi) + \\
        &+v_2(-\cos\phi\sin\psi - \sin\phi\sin\theta\cos\psi)

    .. math::
        f_{3\phi} = &+v_1(\cos\phi\sin\psi+\sin\phi\sin\theta\cos\psi) + \\
        &+v_2(\cos\phi\cos\psi - \sin\phi\sin\theta\sin\psi) + \\
        &+v_3(-\sin\phi\cos\theta)\\
        f_{3\theta} = &+v_1(-\cos\phi\cos\theta\cos\psi)+\\
        &+v_2(\cos\phi\cos\theta\sin\psi) + \\
        &+v_3(-\cos\phi\sin\theta)\\
        f_{3\psi} = &+v_1(\sin\phi\cos\psi+\cos\phi\sin\theta\sin\psi)  + \\
        &+v_2(-\sin\phi\sin\psi + \cos\phi\sin\theta\cos\psi)

    Args:
        euler (np.ndarray): Vector of Euler angles, :math:`\mathbf{\Theta} = [\phi, \theta, \psi]`, in radians.
        v (np.ndarray): 3 dimensional vector in G frame.

    Returns:
        np.ndarray: Resulting 3 by 3 matrix :math:`\frac{\partial \mathbf{f}}{\partial\mathbf{\Theta}}`.

    """
    res = np.zeros((3, 3))

    # Notation shorthand. sin and cos of psi (roll)
    sp = np.sin(euler[0])
    cp = np.cos(euler[0])

    # Notation shorthand. sin and cos of theta (pitch)
    st = np.sin(euler[1])
    ct = np.cos(euler[1])

    # Notation shorthand. sin and cos of psi (yaw)
    ss = np.sin(euler[2])
    cs = np.cos(euler[2])

    v1 = v[0]
    v2 = v[1]
    v3 = v[2]

    res[0, 0] = v2*(sp*ss + cp*st*cp) + v3*(cp*ss - sp*st*cs)
    res[0, 1] = v1*(-st*cs) + v2*(sp*ct*cs) + v3*(cp*ct*cs)
    res[0, 2] = v1*(ct*ss) + v2*(-cp*cs - sp*st*ss) + v3*(sp*cs-cp*st*ss)

    res[1, 0] = v2*(-sp*cs+cp*st*ss) + v3*(-cp*cs + sp*st*ss)
    res[1, 1] = v1*(-st*ss) + v2*(sp*ct*ss) + v3*(-cp*ct*ss)
    res[1, 2] = v1*(ct*cs) + v2*(-cp*ss + sp*st*cs) + v3*(sp*ss + cp*st*cs)

    res[2, 0] = v2*(cp*ct) + v3*(-sp*ct)
    res[2, 1] = v1*(-ct) + v2*(-sp*st) + v3*(-cp*st)

    return res


def der_Peuler_by_v(euler, v):
    r"""
    Provides the derivative of the product between the projection matrix :math:`P^{AG}(\mathbf{\Theta})` (that projects
    a vector in G frame onto A frame) and a constant vector expressed in G frame of reference, :math:`\mathbf{v}_G`,
    with respect to the Euler angles, :math:`\mathbf{\Theta}=[\phi,\theta,\psi]^T`:

    .. math::
        \frac{\partial}{\partial\Theta}(P^{AG}(\Theta)\mathbf{v}^G) = \frac{\partial \mathbf{f}}{\partial\mathbf{\Theta}}

    where :math:`\frac{\partial \mathbf{f}}{\partial\mathbf{\Theta}}` is the resulting 3 by 3 matrix.

    Being :math:`P^{AG}(\Theta)` the projection matrix from the G frame to the A frame in terms of the Euler angles
    :math:`\Theta` as :math:`P^{AG}(\Theta) = \tau_x(-\Phi)\tau_y(-\Theta)\tau_z(-\Psi)`, where
    the rotation matrix is expressed as:

    .. math::
        C^{AG}(\Theta) = \begin{bmatrix}
        \cos\theta\cos\psi & -\cos\theta\sin\psi & \sin\theta \\
        \cos\phi\sin\psi + \sin\phi\sin\theta\cos\psi & \cos\phi\cos\psi - \sin\phi\sin\theta\sin\psi & -\sin\phi\cos\theta \\
        \sin\phi\sin\psi - \cos\phi\sin\theta\cos\psi & \sin\phi\cos\psi + \cos\phi\sin\theta\sin\psi & \cos\phi\cos\theta
        \end{bmatrix}

    and the projection matrix as:

    .. math::
        P^{AG}(\Theta) = \begin{bmatrix}
        \cos\theta\cos\psi & \cos\theta\sin\psi & -\sin\theta \\
        -\cos\phi\sin\psi + \sin\phi\sin\theta\cos\psi & \cos\phi\cos\psi + \sin\phi\sin\theta\sin\psi & \sin\phi\cos\theta \\
        \sin\phi\sin\psi + \cos\phi\sin\theta\cos\psi & -\sin\phi\cos\psi + \cos\phi\sin\theta\sin\psi & \cos\phi\cos\theta
        \end{bmatrix}

    the components of the derivative at hand are the following, where
    :math:`f_{1\theta} = \frac{\partial \mathbf{f}_1}{\partial\theta}`.

    .. math::
        f_{1\phi} =&0 \\
        f_{1\theta} = &-v_1\sin\theta\cos\psi \\
        &+v_2\sin\theta\sin\psi \\
        &+v_3\cos\theta \\
        f_{1\psi} = &-v_1\cos\theta\sin\psi \\
        &- v_2\cos\theta\cos\psi

    .. math::
        f_{2\phi} = &+v_1(-\sin\phi\sin\psi + \cos\phi\sin\theta\cos\psi) + \\
        &+v_2(-\sin\phi\cos\psi - \cos\phi\sin\theta\sin\psi) + \\
        &+v_3(-\cos\phi\cos\theta)\\
        f_{2\theta} = &+v_1(\sin\phi\cos\theta\cos\psi) + \\
        &+v_2(-\sin\phi\cos\theta\sin\psi) +\\
        &+v_3(\sin\phi\sin\theta)\\
        f_{2\psi} = &+v_1(\cos\phi\cos\psi - \sin\phi\sin\theta\sin\psi) + \\
        &+v_2(-\cos\phi\sin\psi - \sin\phi\sin\theta\cos\psi)

    .. math::
        f_{3\phi} = &+v_1(\cos\phi\sin\psi+\sin\phi\sin\theta\cos\psi) + \\
        &+v_2(\cos\phi\cos\psi - \sin\phi\sin\theta\sin\psi) + \\
        &+v_3(-\sin\phi\cos\theta)\\
        f_{3\theta} = &+v_1(-\cos\phi\cos\theta\cos\psi)+\\
        &+v_2(\cos\phi\cos\theta\sin\psi) + \\
        &+v_3(-\cos\phi\sin\theta)\\
        f_{3\psi} = &+v_1(\sin\phi\cos\psi+\cos\phi\sin\theta\sin\psi)  + \\
        &+v_2(-\sin\phi\sin\psi + \cos\phi\sin\theta\cos\psi)

    Args:
        euler (np.ndarray): Vector of Euler angles, :math:`\mathbf{\Theta} = [\phi, \theta, \psi]`, in radians.
        v (np.ndarray): 3 dimensional vector in G frame.

    Returns:
        np.ndarray: Resulting 3 by 3 matrix :math:`\frac{\partial \mathbf{f}}{\partial\mathbf{\Theta}}`.

    """
    res = np.zeros((3, 3))

    # Notation shorthand. sin and cos of psi (roll)
    sp = np.sin(euler[0])
    cp = np.cos(euler[0])

    # Notation shorthand. sin and cos of theta (pitch)
    st = np.sin(euler[1])
    ct = np.cos(euler[1])

    # Notation shorthand. sin and cos of psi (yaw)
    ss = np.sin(euler[2])
    cs = np.cos(euler[2])

    v1 = v[0]
    v2 = v[1]
    v3 = v[2]

    res[0, 1] = v1*(-st*cs) + v2*(-st*ss) - v3*ct
    res[0, 2] = -v1*(ct*ss) + v2*ct*cs

    res[1, 0] = v1*(sp*ss + cp*st*cs) + v2*(-sp*cs + cp*st*ss) + v3 * (cp*ct)
    res[1, 1] = v1*(sp*ct*cs) + v2*(sp*ct*ss) + v3*(-sp*st)
    res[1, 2] = v1*(-cp*cs - sp*st*ss) + v2*(-cp*ss + sp*st*cs)

    res[2, 0] = v1*(cp*ss-sp*st*cs) + v2*(-cp*cs-sp*st*ss) + v3*(-sp*ct)
    res[2, 1] = v1*(cp*ct*cs) + v2*(cp*ct*ss) + v3*(-cp*st)
    res[2, 2] = v1*(sp*ss + -cp*st*ss) + v2*(sp*ss+cp*st*cs)

    return res


def der_Ceuler_by_v_NED(euler, v):
    r"""
    Provides the derivative of the product between the rotation matrix :math:`C^{AG}(\mathbf{\Theta})` and a constant
    vector, :math:`\mathbf{v}`, with respect to the Euler angles, :math:`\mathbf{\Theta}=[\phi,\theta,\psi]^T`:

    .. math::
        \frac{\partial}{\partial\Theta}(C^{AG}(\Theta)\mathbf{v}^G) = \frac{\partial \mathbf{f}}{\partial\mathbf{\Theta}}

    where :math:`\frac{\partial \mathbf{f}}{\partial\mathbf{\Theta}}` is the resulting 3 by 3 matrix.

    Being :math:`C^{AG}(\Theta)` the rotation matrix from the G frame to the A frame in terms of the Euler angles
    :math:`\Theta` as:

    .. math::
        C^{AG}(\Theta) = \begin{bmatrix}
        \cos\theta\cos\psi & \cos\theta\sin\psi & -\sin\theta \\
        -\cos\phi\sin\psi + \sin\phi\sin\theta\cos\psi & \cos\phi\cos\psi + \sin\phi\sin\theta\sin\psi & \sin\phi\cos\theta \\
        \sin\phi\sin\psi + \cos\phi\sin\theta\cos\psi & -\sin\phi\cos\psi + \cos\psi\sin\theta\sin\psi & \cos\phi\cos\theta
        \end{bmatrix}

    the components of the derivative at hand are the following, where
    :math:`f_{1\theta} = \frac{\partial \mathbf{f}_1}{\partial\theta}`.

    .. math::
        f_{1\phi} =&0 \\
        f_{1\theta} = &-v_1\sin\theta\cos\psi \\
        &-v_2\sin\theta\sin\psi \\
        &-v_3\cos\theta \\
        f_{1\psi} = &-v_1\cos\theta\sin\psi + v_2\cos\theta\cos\psi

    .. math::
        f_{2\phi} = &+v_1(\sin\phi\sin\psi + \cos\phi\sin\theta\cos\psi) + \\
        &+v_2(-\sin\phi\cos\psi + \cos\phi\sin\theta\sin\psi) + \\
        &+v_3(\cos\phi\cos\theta) \\
        f_{2\theta} = &+v_1(\sin\phi\cos\theta\cos\psi) + \\
        &+v_2(\sin\phi\cos\theta\sin\psi) +\\
        &-v_3(\sin\phi\sin\theta) \\
        f_{2\psi} = &+v_1(-\cos\phi\cos\psi - \sin\phi\sin\theta\sin\psi) + \\
        &+v_2(-\cos\phi\sin\psi + \sin\phi\sin\theta\cos\psi)

    .. math::
        f_{3\phi} = &+v_1(\cos\phi\sin\psi-\sin\phi\sin\theta\cos\psi) + \\
        &+v_2(-\cos\phi\cos\psi - \sin\phi\sin\theta\sin\psi) + \\
        &+v_3(-\sin\phi\cos\theta) \\
        f_{3\theta} = &+v_1(\cos\phi\cos\theta\cos\psi)+\\
        &+v_2(\cos\phi\cos\theta\sin\psi) + \\
        &+v_3(-\cos\phi\sin\theta) \\
        f_{3\psi} = &+v_1(\sin\phi\cos\psi-\cos\phi\sin\theta\sin\psi)  + \\
        &+v_2(\sin\phi\sin\psi + \cos\phi\sin\theta\cos\psi)

    Note:
        This function is defined in a North East Down frame which is not the typically used one in SHARPy.

    Args:
        euler (np.ndarray): Vector of Euler angles, :math:`\mathbf{\Theta} = [\phi, \theta, \psi]`, in radians.
        v (np.ndarray): 3 dimensional vector in G frame.

    Returns:
        np.ndarray: Resulting 3 by 3 matrix :math:`\frac{\partial \mathbf{f}}{\partial\mathbf{\Theta}}`.

    """

    # TODO: Verify with new euler rotation matrices

    res = np.zeros((3, 3))

    # Notation shorthand. sin and cos of psi (roll)
    sp = np.sin(euler[0])
    cp = np.cos(euler[0])

    # Notation shorthand. sin and cos of theta (pitch)
    st = np.sin(euler[1])
    ct = np.cos(euler[1])

    # Notation shorthand. sin and cos of psi (yaw)
    ss = np.sin(euler[2])
    cs = np.cos(euler[2])

    v1 = v[0]
    v2 = v[1]
    v3 = v[2]

    res[0, 0] = 0
    res[0, 1] = - v1*st*cs - v2*st*ss - v3*ct
    res[0, 2] = -v1*ct*ss + v2*ct*cs

    res[1, 0] = v1*(sp*ss + cp*st*cs) + v2*(-sp*cs + cp*st*ss) + v3*cp*ct
    res[1, 1] = v1*(sp*ct*cs) + v2*sp*ct*ss - v3*sp*st
    res[1, 2] = v1*(-cp*cs - sp*st*ss) + v2*(-cp*ss + sp*st*cs)

    res[2, 0] = v1*(cp*ss - sp*st*cs) + v2*(-cp*cs-sp*st*ss) + v3*(-sp*ct)
    res[2, 1] = v1*cp*ct*cs + v2*cp*ct*ss - v3*cp*st
    res[2, 2] = v1*(sp*cs - cp*st*ss) + v2*(sp*ss + cp*st*cs)

    return res


def cross3(v,w):
    """
    Computes the cross product of two vectors (v and w) with size 3
    """

    res = np.zeros((3,),)
    res[0] = v[1]*w[2] - v[2]*w[1]
    res[1] = -v[0]*w[2] + v[2]*w[0]
    res[2] = v[0]*w[1] - v[1]*w[0]

    return res


def deuler_dt(euler):
    r"""
    Rate of change of the Euler angles in time for a given angular velocity in A frame :math:`\omega^A=[p, q, r]`.

    .. math::
        \begin{bmatrix}\dot{\phi} \\ \dot{\theta} \\ \dot{\psi}\end{bmatrix} =
        \begin{bmatrix}
        1 & \sin\phi\tan\theta & -\cos\phi\tan\theta \\
        0 & \cos\phi & \sin\phi \\
        0 & -\frac{\sin\phi}{\cos\theta} & \frac{\cos\phi}{\cos\theta}
        \end{bmatrix}
        \begin{bmatrix}
        p \\ q \\ r
        \end{bmatrix}

    Args:
        euler (np.ndarray): Euler angles :math:`[\phi, \theta, \psi]` for roll, pitch and yaw, respectively.

    Returns:
        np.ndarray: Propagation matrix relating the rotational velocities to the euler angles.
    """

    phi = euler[0]  # roll
    theta = euler[1]  # pitch

    A = np.zeros((3, 3))
    A[0, 0] = 1
    A[0, 1] = np.tan(theta) * np.sin(phi)
    A[0, 2] = -np.tan(theta) * np.cos(phi)

    A[1, 1] = np.cos(phi)
    A[1, 2] = np.sin(phi)

    A[2, 1] = -np.sin(phi) / np.cos(theta)
    A[2, 2] = np.cos(phi) / np.cos(theta)

    return A


def deuler_dt_NED(euler):
    r"""

    Warnings:
        Based on a NED frame

    Rate of change of the Euler angles in time for a given angular velocity in A frame :math:`\omega^A=[p, q, r]`.

    .. math::
        \begin{bmatrix}\dot{\phi} \\ \dot{\theta} \\ \dot{\psi}\end{bmatrix} =
        \begin{bmatrix}
        1 & \sin\phi\tan\theta & \cos\phi\tan\theta \\
        0 & \cos\phi & -\sin\phi \\
        0 & \frac{\sin\phi}{\cos\theta} & \frac{\cos\phi}{\cos\theta}
        \end{bmatrix}
        \begin{bmatrix}
        p \\ q \\ r
        \end{bmatrix}

    Note:
        This function is defined in a North East Down frame which is not the typically used one in SHARPy.

    Args:
        euler (np.ndarray): Euler angles :math:`[\phi, \theta, \psi]` for roll, pitch and yaw, respectively.

    Returns:
        np.ndarray: Propagation matrix relating the rotational velocities to the euler angles.
    """

    # TODO: Verify with the new euler rotation matrices
    phi = euler[0]  # roll
    theta = euler[1]  # pitch

    A = np.zeros((3, 3))
    A[0, 0] = 1
    A[0, 1] = np.tan(theta) * np.sin(phi)
    A[0, 2] = np.tan(theta) * np.cos(phi)

    A[1, 1] = np.cos(phi)
    A[1, 2] = -np.sin(phi)

    A[2, 1] = np.sin(phi) / np.cos(theta)
    A[2, 2] = np.cos(phi) / np.cos(theta)

    return A


def der_Teuler_by_w(euler, w):
    r"""
    Calculates the matrix

    .. math::
        \frac{\partial}{\partial\Theta}\left.\left(T^{GA}(\mathbf{\Theta})
        \mathbf{\omega}^A\right)\right|_{\Theta_0,\omega^A_0}

    from the linearised euler propagation equations

    .. math::
       \delta\mathbf{\dot{\Theta}} = \frac{\partial}{\partial\Theta}\left.\left(T^{GA}(\mathbf{\Theta})
       \mathbf{\omega}^A\right)\right|_{\Theta_0,\omega^A_0}\delta\mathbf{\Theta} +
       T^{GA}(\mathbf{\Theta_0}) \delta\mathbf{\omega}^A

    where :math:`T^{GA}` is the nonlinear relation between the euler angle rates and the rotational velocities and is
    provided by :func:`deuler_dt`.

    The concerned matrix is calculated as follows:

    .. math::
        \frac{\partial}{\partial\Theta}\left.\left(T^{GA}(\mathbf{\Theta})
        \mathbf{\omega}^A\right)\right|_{\Theta_0,\omega^A_0} = \\
        \begin{bmatrix}
        q\cos\phi\tan\theta-r\sin\phi\tan\theta & q\sin\phi\sec^2\theta + r\cos\phi\sec^2\theta & 0 \\
        -q\sin\phi - r\cos\phi & 0 & 0 \\
        q\frac{\cos\phi}{\cos\theta}-r\frac{\sin\phi}{\cos\theta} & q\sin\phi\tan\theta\sec\theta +
        r\cos\phi\tan\theta\sec\theta & 0
        \end{bmatrix}_{\Theta_0, \omega^A_0}

    Note:
        This function is defined in a North East Down frame which is not the typically used one in SHARPy.

    Args:
        euler (np.ndarray): Euler angles at the linearisation point :math:`\mathbf{\Theta}_0 = [\phi,\theta,\psi]` or
            roll, pitch and yaw angles, respectively.
        w (np.ndarray): Rotational velocities at the linearisation point in A frame :math:`\omega^A_0`.

    Returns:
        np.ndarray: Computed :math:`\frac{\partial}{\partial\Theta}\left.\left(T^{GA}(\mathbf{\Theta})\mathbf{\omega}^A\right)\right|_{\Theta_0,\omega^A_0}`
    """


    p = w[0]
    q = w[1]
    r = w[2]

    cp = np.cos(euler[0])
    sp = np.sin(euler[0])

    st = np.sin(euler[1])
    ct = np.cos(euler[1])
    tt = np.tan(euler[1])
    tsec = ct ** -1

    derT = np.zeros((3, 3))

    derT[0, 0] = q * cp * tt + r * sp * tt
    derT[0, 1] = q * sp * tsec ** 2 - r * cp * tsec ** 2

    derT[1, 0] = -q * sp + r * cp

    derT[2, 0] = - q * cp / ct - r * sp / ct
    derT[2, 1] = - q * sp * tt * tsec + r * cp * tt * tsec

    return derT


def der_Teuler_by_w_NED(euler, w):
    r"""

    Warnings:
        Based on a NED G frame


    Calculates the matrix

    .. math::
        \frac{\partial}{\partial\Theta}\left.\left(T^{GA}(\mathbf{\Theta})
        \mathbf{\omega}^A\right)\right|_{\Theta_0,\omega^A_0}

    from the linearised euler propagation equations

    .. math::
       \delta\mathbf{\dot{\Theta}} = \frac{\partial}{\partial\Theta}\left.\left(T^{GA}(\mathbf{\Theta})
       \mathbf{\omega}^A\right)\right|_{\Theta_0,\omega^A_0}\delta\mathbf{\Theta} +
       T^{GA}(\mathbf{\Theta_0}) \delta\mathbf{\omega}^A

    where :math:`T^{GA}` is the nonlinear relation between the euler angle rates and the rotational velocities and is
    provided by :func:`deuler_dt`.

    The concerned matrix is calculated as follows:

    .. math::
        \frac{\partial}{\partial\Theta}\left.\left(T^{GA}(\mathbf{\Theta})
        \mathbf{\omega}^A\right)\right|_{\Theta_0,\omega^A_0} = \\
        \begin{bmatrix}
        q\cos\phi\tan\theta-r\sin\phi\tan\theta & q\sin\phi\sec^2\theta + r\cos\phi\sec^2\theta & 0 \\
        -q\sin\phi - r\cos\phi & 0 & 0 \\
        q\frac{\cos\phi}{\cos\theta}-r\frac{\sin\phi}{\cos\theta} & q\sin\phi\tan\theta\sec\theta +
        r\cos\phi\tan\theta\sec\theta & 0
        \end{bmatrix}_{\Theta_0, \omega^A_0}

    Args:
        euler (np.ndarray): Euler angles at the linearisation point :math:`\mathbf{\Theta}_0 = [\phi,\theta,\psi]` or
            roll, pitch and yaw angles, respectively.
        w (np.ndarray): Rotational velocities at the linearisation point in A frame :math:`\omega^A_0`.

    Returns:
        np.ndarray: Computed :math:`\frac{\partial}{\partial\Theta}\left.\left(T^{GA}(\mathbf{\Theta})\mathbf{\omega}^A\right)\right|_{\Theta_0,\omega^A_0}`
    """

    # TODO: Verify with new Euler rotation matrices

    p = w[0]
    q = w[1]
    r = w[2]

    cp = np.cos(euler[0])
    sp = np.sin(euler[0])

    st = np.sin(euler[1])
    ct = np.cos(euler[1])
    tt = np.tan(euler[1])
    tsec = ct ** -1

    derT = np.zeros((3, 3))

    derT[0, 0] = q * cp * tt - r * sp * tt
    derT[0, 1] = q * sp * tsec ** 2 + r * cp * tsec ** 2

    derT[1, 0] = -q * sp - r * cp

    derT[2, 0] = q * cp / ct - r * sp / ct
    derT[2, 1] = q * sp * tt * tsec + r * cp * tt * tsec

    return derT


def multiply_matrices(*argv):
    """
    multiply_matrices

    Multiply a series of matrices from left to right

    Args:
        *argv: series of numpy arrays
    Returns:
        sol(numpy array): product of all the given matrices

    Examples:
        solution = multiply_matrices(A, B, C)
    """

    size = np.shape(argv[0])
    nrow = size[0]

    sol = np.eye(nrow)
    for M in argv:
        sol = np.dot(sol, M)
    return sol


def norm3d(v):
    """
    Norm of a 3D vector

    Notes:
        Faster than np.linalg.norm

    Args:
        v (np.ndarray): 3D vector

    Returns:
        np.ndarray: Norm of the vector
    """
    return np.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])


def normsq3d(v):
    """
    Square of the norm of a 3D vector

    Args:
        v (np.ndarray): 3D vector

    Returns:
        np.ndarray: Square of the norm of the vector
    """
    return v[0]*v[0]+v[1]*v[1]+v[2]*v[2]


def get_transformation_matrix(transformation):
    r"""
    Returns a projection matrix function between the desired frames of reference.

    Examples:

        The projection matrix :math:`C^GA(\chi)` expresses a vector in the body-attached
        reference frame ``A`` in the inertial frame ``G``, which is a function of the quaternion.

        .. code-block::

            cga_function = get_transformation_matrix('ga')
            cga = cga_function(quat)  # The actual projection matrix between A and G for a known quaternion


        If the projection involves the ``G`` and ``B`` frames, the output function will take both the quaternion
        and the CRV as arguments.

        .. code-block::

            cgb_function = get_transformation_matrix('gb')
            cgb = cgb_function(psi, quat)  # The actual projection matrix between B and G for a known CRV and quaternion

    Args:
        transformation (str): Desired projection matrix function.

    Returns:
        function: Function to obtain the desired projection matrix. The function will either take the CRV, the
          quaternion, or both as arguments.

    Note:
        If a rotation is desired, it can be achieved by transposing the resulting projection matrix.
    """

    if transformation == 'ab':
        cab = crv2rotation
        return cab
    elif transformation == 'ba':
        def cba(psi):
            return crv2rotation(psi).T

        return cba
    elif transformation == 'ga':
        cga = quat2rotation
        return cga
    elif transformation == 'ag':
        def cag(quat):
            return quat2rotation(quat).T

        return cag
    elif transformation == 'bg':
        def cbg(psi, quat):
            cag = get_transformation_matrix('ag')
            cba = get_transformation_matrix('ba')
            return cba(psi).dot(cag(quat))

        return cbg
    elif transformation == 'gb':
        def cgb(psi, quat):
            cab = get_transformation_matrix('ba')
            cga = get_transformation_matrix('ga')
            return cga(quat).dot(cab(psi))

        return cgb
    else:
        raise NameError('Unknown transformation.')


def der_skewp_skewp_v(p, v):
    """
    This function computes:

        .. math:: \frac{d}{d\boldsymbol{p}} (\tilde{\boldsymbol{p}} \tilde{\boldsymbol{p}} v)
    """
    der = np.zeros((3,3))

    der[0, 0] = v[1]*p[1] + v[2]*p[2]
    der[0, 1] = -2*v[0]*p[1] + v[1]*p[0]
    der[0, 2] = -2*v[0]*p[2] + v[2]*p[0]

    der[1, 0] = v[0]*p[1] - 2*v[1]*p[0]
    der[1, 1] = v[0]*p[0] + v[2]*p[2]
    der[1, 2] = -2*v[1]*p[2] + v[2]*p[1]

    der[2, 0] = v[0]*p[2] - 2*v[2]*p[0]
    der[2, 1] = v[1]*p[2] - 2*v[2]*p[1]
    der[2, 2] = v[0]*p[0] + v[1]*p[1]

    return der
