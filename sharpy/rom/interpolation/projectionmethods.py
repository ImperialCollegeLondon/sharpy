"""Projection Methods

Set of transformations onto generalised coordinates.

The methods in this module have a common signature and all return a tuple. The first element is the right projector
and the second the left one. This transformations are then applied to the local reduced order systems to transform them
to a generalised set of coordinates on which to perform the interpolation.

References:
    [1] D. Amsallem and C. Farhat, An online method for interpolating linear
    parametric reduced-order models, SIAM J. Sci. Comput., 33 (2011), pp. 2169–2198.

    [2] Geuss, M., Panzer, H. & Lohmann, B., 2013. On parametric model order
    reduction by matrix interpolation. 2013 European Control Conference (ECC),
    pp.3433–3438.

    [3] Panzer, J. Mohring, R. Eid, and B. Lohmann, Parametric model order
    reduction by matrix interpolation, at–Automatisierungstechnik, 58 (2010),
    pp. 475–484.

    [4] Mahony, R., Sepulchre, R. & Absil, P. -a., 2004. Riemannian Geometry of
    Grassmann Manifolds with a View on Algorithmic Computation. Acta Applicandae
    Mathematicae, 80(2), pp.199–220.
"""
import numpy as np
import scipy.linalg as sclalg
import warnings

import sharpy.utils.cout_utils as cout


def amsallem(vv_list, wwt_list, **kwargs):
    r"""
    Congruent Transformation following the methods of Amsallem & Farhat [1].


    Args:
        vv_list (list(np.ndarray)): List of right reduced order bases.
        wwt_list (list(np.ndarray)): List of left reduced order bases transposed.
        **kwargs: Key word arguments

    Keyword Args:
        ref_case (int): Reference case index.
        vref (np.ndarray): Reference right reduced order basis (if ``ref_case`` not provided)
        wtref (np.ndarray): Reference left reduced order basis (if ``ref_case`` not provided)

    Returns:
        tuple: (Q, Q^{-1}): Tuple containing lists of :math:`Q` and :math:`Q^{-1}`.

    References:

        [1] D. Amsallem and C. Farhat, An online method for interpolating linear
        parametric reduced-order models, SIAM J. Sci. Comput., 33 (2011), pp. 2169–2198.
    """
    warnings.warn('Method untested!')

    vref, wtref = get_reference_bases(vv_list, wwt_list, **kwargs)

    q_list = []
    qinv_list = []

    for ii in range(len(vv_list)):
        U, sv, Z = sclalg.svd(np.dot(vv_list[ii].T, vref),
                              full_matrices=False,
                              overwrite_a=False,
                              lapack_driver='gesdd')
        Q = np.dot(U, Z.T)
        U, sv, Z = sclalg.svd(np.dot(wwt_list[ii], wtref.T),
                              full_matrices=False, overwrite_a=False,
                              lapack_driver='gesdd')
        Qinv = np.dot(U, Z.T).T

        q_list.append(Q)
        qinv_list.append(Qinv)

    return q_list, qinv_list


def panzer(vv_list, wwt_list, **kwargs):
    r"""
    Congruent Transformation following the methods of Panzer for orthonormal bases.

    The right reduced order bases are collected into

    .. math:: V_{all} = [V_1, V_2, \dots, V_k] \in \mathbb{R}^{n\times kq}

    The first :math:`q` most important directions are then chosen using an SVD

    .. math:: V_{all} = U\Sigma N^T,\,U\in\mathbb{R}^{n\times n},\,\Sigma\in\mathbb{R}^{n\times kq}

    and the first :math:`q` columns of :math:`U` form the basis for the subspace spanned by :math:`V_{all}`. Choosing
    the first :math:`q` columns guarantees to capture the most important directions in :math:`V_{all}`.

    Following the notation from [3], the transformation matrix :math:`R` is such that

    .. math:: R^\top x_1 = R\top x_2 \in \mathbb{R}^{q}.

    :math:`R` is chosen as :math:`R=U[:, q]`.

    The local reduced state vectors are transformed to the generalised set of coordinates through the transformation
    :math:`x_{r, i} = T^{-1}_i x^*_{r, i}` where :math:`x^*_{r, i}` is the reduced state space vector in generalised
    coordinates.

    .. math:: T_i = R^\top V_i

    and :math:`R` is an orthogonal matrix. The remaining transformation for the left projector is given analogously
    by :math:`M_i = (W_i^\top R)^{-1}` (see note 1).

    Therefore, the local reduced order systems need to be projected onto the congruent set of coordinates by using
    the right projector :math:`T_i^{-1}` and the left projector :math:`M_i`.

    Args:
        vv_list (list(np.ndarray)): List of right reduced order bases.
        wwt_list (list(np.ndarray)): List of left reduced order bases transposed.
        **kwargs: Key word arguments. Unused. Needed to keep the same signature as the other projection methods.

    Returns:

        tuple: :math:`(T^{-1}, M)` tuple containing lists of :math:`T^{-1}` and :math:`M`, where :math:`T^{-1}` is the right
        projector and :math:`M` is the left projector.

    References:

        [3] Panzer, J. Mohring, R. Eid, and B. Lohmann, Parametric model order
        reduction by matrix interpolation, at–Automatisierungstechnik, 58 (2010),
        pp. 475–484.

    Notes:
        1. For the moment, this method is only implemented for orthogonal projections (i.e. :math:`V=W^\top`) in which
           case :math:`M_i = T_i`

        2. If a warning message appears saying that :math:`R` is not orthogonal, check Remark 1 in [3].
    """

    # kwargs are required (despite them not being used) to keep the same signature as the remaining methods

    # orthogonal projection support only
    if not np.allclose(wwt_list[0].T, vv_list[0]):
        raise NotImplementedError('Panzer projection method does not currently support oblique projections. See Docs.')

    m_list = []
    tinv_list = []

    # generate basis
    U, sv = sclalg.svd(np.concatenate(vv_list, axis=1),
                       full_matrices=False, overwrite_a=False,
                       lapack_driver='gesdd')[:2]
    # chop U
    r = U[:, :vv_list[0].shape[1]]

    if not np.allclose(r.T.dot(r), np.eye(r.shape[1])):
        raise NotImplementedError('Panzer projection. Method not supported for non-orthogonal R. See Docs.')

    for ii in range(len(vv_list)):
        t = r.T.dot(vv_list[ii])
        # m = np.linalg.inv(wwt_list[ii].dot(r))  # for the case where W != V (not tested)
        m = t
        tinv_list.append(np.linalg.inv(t))
        m_list.append(m)

    return tinv_list, m_list


def leastsq(vv_list, wwt_list, **kwargs):
    r"""
    Congruent Transformation using a least squares transformation.

    Suitable for all basis.

    Args:
        vv_list (list(np.ndarray)): List of right reduced order bases.
        wwt_list (list(np.ndarray)): List of left reduced order bases transposed.
        **kwargs: Key word arguments

    Keyword Args:
        ref_case (int): Reference case index.
        vref (np.ndarray): Reference right reduced order basis (if ``ref_case`` not provided)
        wtref (np.ndarray): Reference left reduced order basis (if ``ref_case`` not provided)

    Returns:
        tuple: (Q, Q^{-1}): Tuple containing lists of :math:`Q` and :math:`Q^{-1}`.

    """
    vref, wtref = get_reference_bases(vv_list, wwt_list, **kwargs)

    q_list = []
    qinv_list = []

    for ii in range(len(vv_list)):
        Q, _, _, _ = sclalg.lstsq(vv_list[ii], vref)
        try:
            cout.cout_wrap('\tLeast Squares Congruent Transformation', 1)
            cout.cout_wrap('\t\tdet(Q): %.3e\tcond(Q): %.3e' \
                           % (np.linalg.det(Q), np.linalg.cond(Q)), 1)
        except ValueError:
            print('det(Q): %.3e\tcond(Q): %.3e' \
                  % (np.linalg.det(Q), np.linalg.cond(Q)))
        # if cond(Q) is small...
        # Qinv = np.linalg.inv(Q)
        P, _, _, _ = sclalg.lstsq(wwt_list[ii].T, wtref.T)
        q_list.append(Q)
        qinv_list.append(P.T)

    return q_list, qinv_list


def strongMAC(vv_list, wwt_list, **kwargs):
    r"""
    Congruent Transformation using a strong Modal Assurance Criterion [2] for general bases.

    See [4], Eq. (7).

    Args:
        vv_list (list(np.ndarray)): List of right reduced order bases.
        wwt_list (list(np.ndarray)): List of left reduced order bases transposed.
        **kwargs: Key word arguments

    Keyword Args:
        ref_case (int): Reference case index.
        vref (np.ndarray): Reference right reduced order basis (if ``ref_case`` not provided)
        wtref (np.ndarray): Reference left reduced order basis (if ``ref_case`` not provided)

    Returns:
        tuple: (Q, Q^{-1}): Tuple containing lists of :math:`Q` and :math:`Q^{-1}`.

    References:

        [2] Geuss, M., Panzer, H. & Lohmann, B., 2013. On parametric model order
        reduction by matrix interpolation. 2013 European Control Conference (ECC),
        pp.3433–3438.

        [4] Mahony, R., Sepulchre, R. & Absil, P. -a., 2004. Riemannian Geometry of
        Grassmann Manifolds with a View on Algorithmic Computation. Acta Applicandae
        Mathematicae, 80(2), pp.199–220.
    """
    vref, wtref = get_reference_bases(vv_list, wwt_list, **kwargs)

    q_list = []
    qinv_list = []

    VTVref = np.dot(vref.T, vref)

    for ii in range(len(vv_list)):
        Q = np.linalg.solve(np.dot(vref.T, vv_list[ii]), VTVref)
        Qinv = np.linalg.inv(Q)
        try:
            cout.cout_wrap('\tStrong MAC Congruent Transformation', 1)
            cout.cout_wrap('\t\tdet(Q): %.3e\tcond(Q): %.3e' \
                           % (np.linalg.det(Q), np.linalg.cond(Q)), 1)
        except ValueError:
            print('det(Q): %.3e\tcond(Q): %.3e' \
                  % (np.linalg.det(Q), np.linalg.cond(Q)))
        q_list.append(Q)
        qinv_list.append(Qinv)

    return q_list, qinv_list


def strongMAC_BT(vv_list, wwt_list, **kwargs):
    r"""
    Congruent Transformation using a strong Modal Assurance Criterion [2].
    This is equivalent to Mahony 2004 [4], Eq. 7, for the case of basis
    obtained by balanced truncation. In general, it will fail if VV[ii] and Vref
    do not describe the same subspace


    Args:
        vv_list (list(np.ndarray)): List of right reduced order bases.
        wwt_list (list(np.ndarray)): List of left reduced order bases transposed.
        **kwargs: Key word arguments

    Keyword Args:
        ref_case (int): Reference case index.
        vref (np.ndarray): Reference right reduced order basis (if ``ref_case`` not provided)
        wtref (np.ndarray): Reference left reduced order basis (if ``ref_case`` not provided)

    Returns:
        tuple: (Q, Q^{-1}): Tuple containing lists of :math:`Q` and :math:`Q^{-1}`.

    References:

        [2] Geuss, M., Panzer, H. & Lohmann, B., 2013. On parametric model order
        reduction by matrix interpolation. 2013 European Control Conference (ECC),
        pp.3433–3438.

        [4] Mahony, R., Sepulchre, R. & Absil, P. -a., 2004. Riemannian Geometry of
        Grassmann Manifolds with a View on Algorithmic Computation. Acta Applicandae
        Mathematicae, 80(2), pp.199–220.
    """
    vref, wtref = get_reference_bases(vv_list, wwt_list, **kwargs)

    q_list = []
    qinv_list = []

    for ii in range(len(vv_list)):
        Q = np.linalg.inv(np.dot(wtref, vv_list[ii]))
        Qinv = np.dot(wtref, vv_list[ii])
        try:
            cout.cout_wrap('\tSrong MAC BT Congruent Transformation')
            cout.cout_wrap('\t\tdet(Q): %.3e\tcond(Q): %.3e' \
                           % (np.linalg.det(Q), np.linalg.cond(Q)), 1)
        except ValueError:
            print('det(Q): %.3e\tcond(Q): %.3e' \
                  % (np.linalg.det(Q), np.linalg.cond(Q)))
        q_list.append(Q)
        qinv_list.append(Qinv)

    return q_list, qinv_list


def maraniello_BT(vv_list, wwt_list, **kwargs):
    r"""
    Equivalent to strongMAC and strongMAC_BT but
    avoids inversions. However, performance are the same as other strongMAC
    approaches - it works only when basis map the same subspaces.

    Projection over ii. This is a sort of weak enforcement

    Args:
        vv_list (list(np.ndarray)): List of right reduced order bases.
        wwt_list (list(np.ndarray)): List of left reduced order bases transposed.
        **kwargs: Key word arguments

    Keyword Args:
        ref_case (int): Reference case index.
        vref (np.ndarray): Reference right reduced order basis (if ``ref_case`` not provided)
        wtref (np.ndarray): Reference left reduced order basis (if ``ref_case`` not provided)

    Returns:
        tuple: (Q, Q^{-1}): Tuple containing lists of :math:`Q` and :math:`Q^{-1}`.

    """
    vref, wtref = get_reference_bases(vv_list, wwt_list, **kwargs)

    q_list = []
    qinv_list = []

    for ii in range(len(vv_list)):
        Q = np.dot(wwt_list[ii], vref)
        Qinv = np.dot(wtref, vv_list[ii])
        try:
            cout.cout_wrap('\tMaraniello BT Congruence Transformation', 1)
            cout.cout_wrap('\t\tdet(Q): %.3e\tcond(Q): %.3e' \
                           % (np.linalg.det(Q), np.linalg.cond(Q)), 1)
        except ValueError:
            print('det(Q): %.3e\tcond(Q): %.3e' \
                  % (np.linalg.det(Q), np.linalg.cond(Q)))
        q_list.append(Q)
        qinv_list.append(Qinv)

    return q_list, qinv_list


def weakMAC_right_orth(vv_list, wwt_list, **kwargs):
    """
    This is like Amsallem [1], but only for state-space models with right
    orthogonal basis [4].

    Args:
        vv_list (list(np.ndarray)): List of right reduced order bases.
        wwt_list (list(np.ndarray)): List of left reduced order bases transposed.
        **kwargs: Key word arguments

    Keyword Args:
        ref_case (int): Reference case index.
        vref (np.ndarray): Reference right reduced order basis (if ``ref_case`` not provided)
        wtref (np.ndarray): Reference left reduced order basis (if ``ref_case`` not provided)

    Returns:
        tuple: (Q, Q^{-1}): Tuple containing lists of :math:`Q` and :math:`Q^{-1}`.

    References:

        [1] D. Amsallem and C. Farhat, An online method for interpolating linear
        parametric reduced-order models, SIAM J. Sci. Comput., 33 (2011), pp. 2169–2198.

        [4] Mahony, R., Sepulchre, R. & Absil, P. -a., 2004. Riemannian Geometry of
        Grassmann Manifolds with a View on Algorithmic Computation. Acta Applicandae
        Mathematicae, 80(2), pp.199–220.
    """
    vref, wtref = get_reference_bases(vv_list, wwt_list, **kwargs)

    q_list = []
    qinv_list = []

    for ii in range(len(vv_list)):
        Q, sc = sclalg.orthogonal_procrustes(vv_list[ii], vref)
        Qinv = Q.T
        try:
            cout.cout_wrap('Weak MAC Right Orthogonal Basis Congruent Transformation', 1)
            cout.cout_wrap('\t\tdet(Q): %.3e\tcond(Q): %.3e' \
                           % (np.linalg.det(Q), np.linalg.cond(Q)), 1)
        except ValueError:
            print('det(Q): %.3e\tcond(Q): %.3e' \
                  % (np.linalg.det(Q), np.linalg.cond(Q)))
        q_list.append(Q)
        qinv_list.append(Qinv)

    return q_list, qinv_list


def weakMAC(vv_list, wwt_list, **kwargs):
    """
    WeakMAC enforcement on the right hand side basis, V.

    Implementation of weak MAC enforcement for a general system.
    The method orthonormalises the right basis (V) and then solves the
    orthogonal Procrustes problem.

    Args:
        vv_list (list(np.ndarray)): List of right reduced order bases.
        wwt_list (list(np.ndarray)): List of left reduced order bases transposed.
        **kwargs: Key word arguments

    Keyword Args:
        ref_case (int): Reference case index.
        vref (np.ndarray): Reference right reduced order basis (if ``ref_case`` not provided)
        wtref (np.ndarray): Reference left reduced order basis (if ``ref_case`` not provided)

    Returns:
        tuple: (Q, Q^{-1}): Tuple containing lists of :math:`Q` and :math:`Q^{-1}`.

    """
    vref, wtref = get_reference_bases(vv_list, wwt_list, **kwargs)

    q_list = []
    qinv_list = []

    # svd of reference
    Uref, svref, Zhref = sclalg.svd(vref, full_matrices=False)

    for ii in range(len(vv_list)):
        # svd of basis
        Uhere, svhere, Zhhere = sclalg.svd(vv_list[ii], full_matrices=False)

        R, sc = sclalg.orthogonal_procrustes(Uhere, Uref)
        Q = np.dot(np.dot(Zhhere.T, np.diag(svhere ** (-1))), R)
        Qinv = np.dot(R.T, np.dot(np.diag(svhere), Zhhere))
        try:
            cout.cout_wrap('\tWeak MAC Congruence Transformation', 1)
            cout.cout_wrap('\t\tdet(Q): %.3e\tcond(Q): %.3e' \
                           % (np.linalg.det(Q), np.linalg.cond(Q)), 1)
        except ValueError:
            print('det(Q): %.3e\tcond(Q): %.3e' \
                  % (np.linalg.det(Q), np.linalg.cond(Q)))
        q_list.append(Q)
        qinv_list.append(Qinv)

    return q_list, qinv_list


def get_reference_bases(vv_list, wwt_list, **kwargs):
    vref = kwargs.get('vref', None)
    wtref = kwargs.get('wtref', None)

    if vref is None and wtref is None:
        try:
            ref_case = kwargs['ref_case']
            vref = vv_list[ref_case]
            wtref = wwt_list[ref_case]
        except KeyError:
            raise KeyError('If vref and wtref are not provided you should specify a ref_case to fetch.')

    return vref, wtref
