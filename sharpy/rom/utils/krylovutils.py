import scipy.sparse as scsp
import numpy as np
import scipy.linalg as sclalg
import sharpy.linear.src.libsparse as libsp

def block_arnoldi_krylov(r, F, G, approx_type='Pade', side='controllability'):

    n = G.shape[0]
    m = G.shape[1]

    Q0, R0, P0 = sclalg.qr(G, pivoting=True)
    Q0 = Q0[:, :m]

    Q = np.zeros((n,m*r), dtype=complex)
    V = np.zeros((n,m*r), dtype=complex)

    for k in range(r):

        if k == 0:
            Q[:, 0:m] = F.dot(Q0)
        else:
            Q[:, k*m: k*m + m] = F.dot(Q[:, (k-1)*m:(k-1)*m + m])

        Q[:, :k*m + m] = mgs_ortho(Q[:, :k*m+m])

        Qf, R, P = sclalg.qr(Q[:, k*m: k*m + m], pivoting=True)
        Q[:, k*m: k*m + m ] = Qf[:, :m]
        if R[0,0] >= 1e-6:
            V[:, k*m:k*m + m] = Q[:, k*m: k*m + m ]
        else:
            print('Deflating')
            k -= 1

    V = mgs_ortho(V)

    return V

def mgs_ortho(X):
    r"""
    Modified Gram-Schmidt Orthogonalisation

    Orthogonalises input matrix :math:`\mathbf{X}` column by column.

    Args:
        X (np.ndarray): Input matrix of dimensions :math:`n` by :math:`m`.

    Returns:
        np.ndarray: Orthogonalised matrix of dimensions :math:`n` by :math:`m`.

    Notes:
        This method is faster than scipy's :func:`scipy.linalg.qr` method that returns an orthogonal matrix as part of
        the QR decomposition, albeit at a higher number of function calls.
    """

    # Q, R = sclalg.qr(X)
    n = X.shape[1]
    m = X.shape[0]

    if type(X) == scsp.csc_matrix:
        Q = scsp.csc_matrix((m, n), dtype=complex)

    else:
        Q = np.zeros((m, n), dtype=complex)

    for i in range(n):
        w = X[:, i]
        for j in range(i):
            h = Q[:, j].T.dot(w)
            w = w - h * Q[:, j]
        if type(X) == scsp.csc_matrix:
            Q[:, i] = w / scsp.linalg.norm(w)
        else:
            Q[:, i] = w / sclalg.norm(w)

    return Q


def construct_krylov(r, lu_A, B, approx_type='Pade', side='b'):
    r"""
    Contructs a Krylov subspace in an iterative manner following the methods of Gugercin [1].

    The construction of the Krylov space is focused on Pade and partial realisation cases for the purposes of model
    reduction. I.e. the partial realisation form of the Krylov space is used if
    ``approx_type = 'partial_realisation'``

        .. math::
            \text{range}(\textbf{V}) = \mathcal{K}_r(\mathbf{A}, \mathbf{b})

    Else, it is replaced by the Pade approximation form:

        .. math::
            \text{range}(\textbf{V}) = \mathcal{K}_r((\sigma\mathbf{I}_n - \mathbf{A})^{-1},
            (\sigma\mathbf{I}_n - \mathbf{A})^{-1}\mathbf{b})

    Note that no inverses are actually computed but rather a single LU decomposition is performed at the beginning
    of the algorithm. Forward and backward substitution is used thereinafter to calculate the required vectors.

    The algorithm also builds the Krylov space for the :math:`\mathbf{C}^T` matrix. It should simply replace ``B``
    and ``side`` should be ``side = 'c'``.

    Examples:
        Partial Realisation:

        >>> V = construct_krylov(r, A, B, 'partial_realisation', 'b')
        >>> W = construct_krylov(r, A, C.T, 'partial_realisation', 'c')

        Pade Approximation:

        >>> V = construct_krylov(r, (sigma * np.eye(nx) - A), B, 'Pade', 'b')
        >>> W = construct_krylov(r, (sigma * np.eye(nx) - A), C.T, 'Pade', 'c')


    References:
        [1]. Gugercin, S. - Projection Methods for Model Reduction of Large-Scale Dynamical Systems. PhD Thesis.
        Rice University. 2003.

    Args:
        r (int): Krylov space order
        lu_A (np.ndarray): For Pade approximations it should be the LU decomposition of :math:`(\sigma I - \mathbf{A})`
            in tuple form, as output from the :func:`scipy.linalg.lu_factor`. For partial realisations it is
            simply :math:`\mathbf{A}`.
        B (np.ndarray): If doing the B side it should be :math:`\mathbf{B}`, else :math:`\mathbf{C}^T`.
        approx_type (str): Type of approximation: ``partial_realisation`` or ``Pade``.
        side: Side of the projection ``b`` or ``c``.

    Returns:
        np.ndarray: Projection matrix

    """

    nx = B.shape[0]

    # Side indicates projection side. if using C then it needs to be transposed
    if side=='c':
        transpose_mode = 1
        B.shape = (nx, 1)
    else:
        transpose_mode = 0
        B.shape = (nx, 1)

    # Output projection matrices
    V = np.zeros((nx, r),
                 dtype=complex)
    H = np.zeros((r, r),
                 dtype=complex)

    # Declare iterative variables
    f = np.zeros((nx, r),
                 dtype=complex)

    if approx_type == 'partial_realisation':
        A = lu_A
        v_arb = B
        v = v_arb / np.linalg.norm(v_arb)
        w = A.dot(v)
    else:
        # LU decomposition
        v = lu_solve(lu_A, B, trans=transpose_mode)
        v = v / np.linalg.norm(v)
        w = lu_solve(lu_A, v)

    alpha = v.T.dot(w)

    # Initial assembly
    f[:, :1] = w - v.dot(alpha)
    V[:, :1] = v
    H[0, 0] = alpha

    for j in range(0, r-1):

        beta = np.linalg.norm(f[:, j])
        v = 1 / beta * f[:, j]

        V[:, j+1] = v
        H_hat = np.block([[H[:j+1, :j+1]],
                          [beta * evec(j)]])

        if approx_type == 'partial_realisation':
            w = A.dot(v)
        else:
            w = lu_solve(lu_A, v, trans=transpose_mode)

        h = V[:, :j+2].T.dot(w)
        f[:, j+1] = w - V[:, :j+2].dot(h)

        # Finite precision
        s = V[:, :j+2].T.dot(f[:, j+1])
        f[:, j+1] = f[:, j+1] - V[:, :j+2].dot(s)
        h += s

        h.shape = (j+2, 1)  # Enforce shape for concatenation
        H[:j+2, :j+2] = np.block([H_hat, h])

    return V


def lu_factor(sigma, A):
    """
    LU Factorisation wrapper of:

    .. math:: LU = (\sigma \mathbf{I} - \mathbf{A})

    In the case of ``A`` being a sparse matrix, the sparse methods in scipy are employed

    Args:
        sigma (float): Expansion frequency
        A (csc_matrix or np.ndarray): Dynamics matrix

    Returns:
        tuple or SuperLU: tuple (dense) or SuperLU (sparse) objects containing the LU factorisation
    """
    n = A.shape[0]
    if type(A) == libsp.csc_matrix:
        return scsp.linalg.splu(sigma * scsp.identity(n, dtype=complex, format='csc') - A)
    else:
        return sclalg.lu_factor(sigma * np.eye(n) - A)


def lu_solve(lu_A, b, trans=0):
    """
    LU solve wrapper.

    Computes the solution to

    .. math:: \mathbf{Ax} = \mathbf{b}

    or

    .. math:: \mathbf{A}^T\mathbf{x} = \mathbf{b}

    if ``trans=1``.

    It uses the ``SuperLU.solve()`` method if the input is a ``SuperLU`` or else will revert to the dense methods
    in scipy.

    Args:
        lu_A (SuperLU or tuple): object or tuple containing the information of the LU factorisation
        b (np.ndarray): Right hand side vector to solve
        trans (int): ``0`` or ``1`` for either solution option.

    Returns:
        np.ndarray: Solution to the system.

    """
    transpose_mode_dict = {0: 'N', 1: 'T'}
    if type(lu_A) == scsp.linalg.SuperLU:
        return lu_A.solve(b, trans=transpose_mode_dict[trans])
    else:
        return sclalg.lu_solve(lu_A, b, trans=trans)



def construct_mimo_krylov(r, lu_A_input, B, approx_type='Pade',side='controllability'):

    if side=='controllability':
        transpose_mode = 0
    else:
        transpose_mode = 1

    m = B.shape[1]  # Full system number of inputs/outputs
    n = B.shape[0]  # Full system number of states

    deflation_tolerance = 1e-10  # Inexact deflation tolerance to approximate norm(V)=0 in machine precision

    # Preallocated size may be too large in case columns are deflated
    last_column = 0

    # Pre-allocate w, V
    V = np.zeros((n, m * r), dtype=complex)
    w = np.zeros((n, m * r), dtype=complex)  # Initialise w, may be smaller than this due to deflation

    if approx_type == 'partial_realisation':
        G = B
        F = lu_A_input
    else:
        G = lu_solve(lu_A_input, B, transpose_mode)

    for k in range(m):
        w[:, k] = G[:, k]

        ## Orthogonalise w_k to preceding w_j for j < k
        if k >= 1:
            w[:, :k+1] = mgs_ortho(w[:, :k+1])[:, :k+1]

    V[:, :m+1] = w[:, :m+1]
    last_column += m

    mu = m  # Initialise controllability index
    mu_c = m  # Guess at controllability index with no deflation
    t = m   # worked column index

    for k in range(1, r):
        for j in range(mu_c):
            if approx_type == 'partial_realisation':
                w[:, t] = F.dot(w[:, t-mu])
            else:
                w[:, t] = lu_solve(lu_A_input, w[:, t-mu], transpose_mode)

            # Orthogonalise w[:,t] against V_i -
            w[:, :t+1] = mgs_ortho(w[:, :t+1])[:, :t+1]

            if np.linalg.norm(w[:, t]) < deflation_tolerance:
                # Deflate w_k
                print('Vector deflated')
                w = [w[:, 0:t], w[:, t+1:]]
                last_column -= 1
                mu -= 1
            else:
                V[:, t] = w[:, t]
                last_column += 1
                t += 1
        mu_c = mu

    return V[:, :t]

def evec(j):
    """j-th unit vector (in row format)

    Args:
        j: Unit vector dimension

    Returns:
        np.ndarray: j-th unit vector

    Examples:
        >>> evec(2)
        np.array([0, 1])
        >>> evec(3)
        np.array([0, 0, 1])

    """
    e = np.zeros(j+1)
    e[j] = 1
    return e


def schur_ordered(A):
    r"""Returns block ordered real Schur form of matrix :math:`\mathbf{A}`

    .. math:: \mathbf{TAT}^\top = \mathbf{A}_s = \begin{bmatrix} A_{11} & A_{12} \\ 0 & A_{22} \end{bmatrix}

    where :math:`A_{11}` contains the stable eigenvalues of :math:`\mathbf{A}`

    Args:
        A (np.ndarray): Matrix to decompose.

    Returns:
        tuple: Tuple containing the Schur decomposition of :math:`\mathbf{A}`, :math:`\mathbf{A}_s`; the transformation
        :math:`\mathbf{T}`; and the number of stable eigenvalues of :math:`\mathbf{A}`.

    Notes:
        This function is a wrapper of ``scipy.linalg.schur`` imposing the settings required for this application.

    """
    As, Tt, n_stable = sclalg.schur(A, output='real', sort='iuc')

    return As, Tt.T, n_stable


def remove_a12(As, n_stable):
    r"""Basis change to remove the (1, 2) block of the block-ordered real Schur matrix :math:`\mathbf{A}`

    Being :math:`\mathbf{A}_s` a matrix of the form

    .. math:: \mathbf{A}_s = \begin{bmatrix} A_{11} & A_{12} \\ 0 & A_{22} \end{bmatrix}

    the (1,2) block is removed by solving the Sylvester equation

    .. math:: \mathbf{A}_{11}\mathbf{X} - \mathbf{X}\mathbf{A}_{22} + \mathbf{A}_{12} = 0

    used to build the change of basis

    .. math:: \mathbf{T} = \begin{bmatrix} \mathbf{I} & -\mathbf{X} \\ \mathbf{0} & \mathbf{I} \end{bmatrix}

    such that

    .. math:: \mathbf{TA}_s\mathbf{T}^\top = \begin{bmatrix} A_{11} & \mathbf{0} \\ 0 & A_{22} \end{bmatrix}.

    Args:
        As (np.ndarray): Block-ordered real Schur matrix (can be built using :func:`krylovutils.schur_ordered`).
        n_stable (int): Number of stable eigenvalues in ``As``.

    Returns:
        np.ndarray: Basis transformation :math:`\mathbf{T}`.

    References:
        Jaimoukha, I. M., Kasenally, E. D.. Implicitly Restarted Krylov Subspace Methods for Stable Partial Realizations
        SIAM Journal of Matrix Analysis and Applications, 1997.
    """
    A11 = As[:n_stable, :n_stable]
    A12 = As[:n_stable, n_stable:]
    A22 = As[n_stable:, n_stable:]
    n = As.shape[0]

    X = sclalg.solve_sylvester(A11, -A22, -A12)

    T = np.block([[np.eye(n_stable), -X], [np.zeros((n-n_stable, n_stable)), np.eye(n-n_stable)]])

    return T
