"""General ROM utilities

S. Maraniello, 14 Feb 2018
"""

import warnings
import numpy as np
import scipy.linalg as scalg

import sharpy.linear.src.libsparse as libsp
import sharpy.linear.src.libss as libss


def balreal_direct_py(A, B, C, DLTI=True, Schur=False, full_outputs=False):
    r"""
    Find balanced realisation of continuous (``DLTI = False``) and discrete (``DLTI = True``)
    time of LTI systems using  scipy libraries.

    The function proceeds to achieve balanced realisation of the state-space system by first solving
    the Lyapunov equations. They are solved using Barlets-Stewart algorithm for
    Sylvester equation, which is based on A matrix Schur decomposition.

    .. math::
        \mathbf{A\,W_c + W_c\,A^T + B\,B^T} &= 0  \\
        \mathbf{A^T\,W_o + W_o\,A + C^T\,C} &= 0

    to obtain the reachability and observability gramians, which are positive definite matrices.

    Then, the gramians are decomposed into their Cholesky factors such that:

    .. math::
        \mathbf{W_c} &= \mathbf{Q_c\,Q_c^T} \\
        \mathbf{W_o} &= \mathbf{Q_o\,Q_o^T}

    A singular value decomposition (SVD) of the product of the Cholesky factors is performed

    .. math:: (\mathbf{Q_o^T\,Q_c}) = \mathbf{U\,\Sigma\,V^*}

    The singular values are then used to build the transformation matrix :math:`\mathbf{T}`

    .. math::
        \mathbf{T} &= \mathbf{Q_c\,V\,\Sigma}^{-1/2} \\
        \mathbf{T}^{-1} &= \mathbf{\Sigma}^{-1/2}\,\mathbf{U^T\,Q_o^T}

    The balanced system is therefore of the form:

    .. math::
        \mathbf{A_b} &= \mathbf{T^{-1}\,A\,T} \\
        \mathbf{B_b} &= \mathbf{T^{-1}\,B} \\
        \mathbf{C_b} &= \mathbf{C\,T} \\
        \mathbf{D_b} &= \mathbf{D}

    Warnings:
        This function may be less computationally efficient than the ``balreal``
        Matlab implementation and does not offer the option to bound the realisation
        in frequency and time.

    Notes:
        - Lyapunov equations are solved using Barlets-Stewart algorithm for
          Sylvester equation, which is based on A matrix Schur decomposition.

        - Notation above is consistent with Gawronski [2].

    Args:
        A (np.ndarray): Plant Matrix
        B (np.ndarray): Input Matrix
        C (np.ndarray): Output Matrix
        DLTI (bool): Discrete time state-space flag
        Schur (bool): Use Schur decomposition to solve the Lyapunov equations

    Returns:
        tuple of np.ndarrays: Tuple of the form ``(S, T, Tinv)`` containing:
            - Singular values in diagonal matrix (``S``)
            - Transformation matrix (``T``).
            - Inverse transformation matrix(``Tinv``).

    References:
        [1] Anthoulas, A.C.. Approximation of Large Scale Dynamical Systems. Chapter 7. Advances in Design and Control.
        SIAM. 2005.

        [2] Gawronski, W.. Dynamics and control of structures. New York: Springer. 1998
    """

    ### select solver for Lyapunov equation
    # Notation reminder:
    # scipy: A X A.T - X = -Q
    # contr: A W A.T - W = - B B.T
    # obser: A.T W A - W = - C.T C
    if DLTI:
        sollyap = scalg.solve_discrete_lyapunov
    else:
        sollyap = scalg.solve_lyapunov

    # A is a sparse matrix in csr_matrix(sparse) format, can not be directly passed into functions used in scipy _solver.py
    # Sparse matrices do not work well with Scipy (Version 1.7.3) in the following code, so A is transformed into a dense matrix here first.
    if type(A) is not np.ndarray:
        try:
            A = A.todense()
        except AttributeError:
            raise TypeError(f'Matrix needs to be in dense form. Unable to convert A matrix of type {type(A)} to '
                            f'dense using method .todense()')

    # solve Lyapunov
    if Schur:
        # decompose A
        Atri, U = scalg.schur(A)

        # solve Lyapunov
        BBtri = np.dot(U.T, np.dot(B, np.dot(B.T, U)))
        CCtri = np.dot(U.T, np.dot(C.T, np.dot(C, U)))
        Wctri = sollyap(Atri, BBtri)
        Wotri = sollyap(Atri.T, CCtri)
        # reconstruct Wo,Wc
        Wc = np.dot(U, np.dot(Wctri, U.T))
        Wo = np.dot(U, np.dot(Wotri, U.T))
    else:
        Wc = sollyap(A, np.dot(B, B.T))
        Wo = sollyap(A.T, np.dot(C.T, C))

    # Choleski factorisation: W=Q Q.T
    # Qc = scalg.cholesky(Wc).T
    # Qo = scalg.cholesky(Wo).T

    # build M matrix and SVD
    # M = np.dot(Qo.T, Qc)
    # U, s, Vh = scalg.svd(M)
    # S = np.diag(s)
    # Sinv = np.diag(1. / s)
    # V = Vh.T

    # Build transformation matrices
    # T = np.dot(Qc, np.dot(V, np.sqrt(Sinv)))
    # Tinv = np.dot(np.sqrt(Sinv), np.dot(U.T, Qo.T))

    # return S, T, Tinv

    ### Find transformation matrices
    # avoid Cholevski - unstable

    # Building T and Tinv using SVD:
    Uc, Sc, Vc = scalg.svd(Wc)
    Uo, So, Vo = scalg.svd(Wo)

    # Perform decomposition:
    Sc = np.sqrt(np.diag(Sc))
    So = np.sqrt(np.diag(So))
    Qc = Uc @ Sc
    Qot = So @ Vo

    # Build Hankel matrix:
    H = Qot @ Qc

    # Find SVD of Hankel matrix:
    U, hsv, Vt = scalg.svd(H)
    # hsv = np.diag(hsv)

    # Find T and Tinv:
    S = np.sqrt(np.diag(hsv))

    # Please note, the notation below is swapped as compared to regular notation in the literature.
    # This is a known feature of SHARPy, hence it is maintained throughout (including documentation).
    Tinv = scalg.inv(S) @ U.T @ Qot
    T = Qc @ Vt.T @ scalg.inv(S)

    if full_outputs is False:
        return hsv, T, Tinv

    else:
        # get square-root factors
        # UT, QoT = scalg.qr(np.dot(np.diag(np.sqrt(hsv)), Tinv), pivoting=False)
        # Vh, QcT = scalg.qr(np.dot(T, np.diag(np.sqrt(hsv))).T, pivoting=False)

        # return hsv, UT.T, Vh, QcT.T, QoT.T
        return hsv, U, Vt, Qc, Qot.T


def balreal_iter(A, B, C, lowrank=True, tolSmith=1e-10, tolSVD=1e-6, kmin=None,
                 tolAbs=False, Print=False, outFacts=False):
    """
    Find balanced realisation of DLTI system.

    Notes:

        Lyapunov equations are solved using iterative squared Smith
        algorithm, in its low or full rank version. These implementations are
        as per the low_rank_smith and smith_iter functions respectively but,
        for computational efficiency, the iterations are rewritten here so as to
        solve for the observability and controllability Gramians contemporary.


    - Exploiting sparsity:

        This algorithm is not ideal to exploit sparsity. However, the following
        strategies are implemented:

            - if the A matrix is provided in sparse format, the powers of A will be
              calculated exploiting sparsity UNTIL the number of non-zero elements
              is below 15% the size of A. Upon this threshold, the cost of the matrix
              multiplication rises dramatically, and A is hence converted to a dense
              numpy array.
    """

    ### Solve Lyapunov equations
    # Notation reminder:
    # scipy: A X A.T - X = -Q
    # contr: A W A.T - W = - B B.T
    # obser: A.T W A - W = - C.T C
    # low-rank smith: A.T X A - X = -Q Q.T

    # matrices size
    N = A.shape[0]
    rC = B.shape[1]
    rO = C.shape[0]

    if lowrank:  # low-rank square-Smith iteration (with SVD)

        # initialise smith iteration
        DeltaNorm = 1e6  # error
        DeltaNormNext = DeltaNorm ** 2  # error expected at next iter

        kk = 0
        Qck = B
        Qok = C.T

        if Print:
            print('Iter\tMaxZ\t|\trank_c\trank_o\tA size')
        while DeltaNorm > tolSmith and DeltaNormNext > 1e-3 * tolSmith:

            ###### controllability
            ### compute Ak^2 * Qck
            # (future: use block Arnoldi)
            Qcright = libsp.dot(A, Qck)
            MaxZhere = np.max(np.abs(Qcright))

            ### enlarge Z matrices
            Qck = np.concatenate((Qck, Qcright), axis=1)
            Qcright = None
            rC = Qck.shape[1]

            if kmin == None or kmin < rC:
                ### "cheap" SVD truncation
                Uc, svc = scalg.svd(Qck, full_matrices=False, overwrite_a=True,
                                    lapack_driver='gesdd')[:2]
                # import scipy.linalg.interpolative as sli
                # Ucnew,svcnew,temp=sli.svd(Qck,tolSVD)
                if tolAbs:
                    rcmax = np.sum(svc > tolSVD)
                else:
                    rcmax = np.sum(svc > tolSVD * svc[0])
                if kmin != None:
                    rC = max(rcmax, kmin)
                else:
                    rC = rcmax
                Qck = Uc[:, :rC] * svc[:rC]
                # free memory
                Uc = None
                Qcright = None

            ###### observability
            ### compute Ak^2 * Qok
            # (future: use block Arnoldi)
            Qoright = np.transpose(libsp.dot(Qok.T, A))
            DeltaNorm = max(MaxZhere, np.max(np.abs(Qoright)))

            ### enlarge Z matrices
            Qok = np.concatenate((Qok, Qoright), axis=1)
            Qoright = None
            rO = Qok.shape[1]

            if kmin == None or kmin < rO:
                ### "cheap" SVD truncation
                Uo, svo = scalg.svd(Qok, full_matrices=False)[:2]

                if tolAbs:
                    romax = np.sum(svo > tolSVD)
                else:
                    romax = np.sum(svo > tolSVD * svo[0])
                if kmin != None:
                    rO = max(romax, kmin)
                else:
                    rO = romax
                Qok = Uo[:, :rO] * svo[:rO]
                Uo = None

            ##### Prepare next time step
            if Print:
                print('%.3d\t%.2e\t%.5d\t%.5d\t%.5d' % (kk, DeltaNorm, rC, rO, N))
            DeltaNormNext = DeltaNorm ** 2

            if DeltaNorm > tolSmith and DeltaNormNext > 1e-3 * tolSmith:

                # compute power
                if type(A) is libsp.csc_matrix:
                    A = A.dot(A)
                    # check sparsity
                    if A.size > 0.15 * N ** 2:
                        A = A.toarray()
                elif type(A) is np.ndarray:
                    A = np.linalg.matrix_power(A, 2)
                else:
                    raise NameError('Type of A not supported')

            ### update
            kk = kk + 1

        A = None


    else:  # full-rank squared smith iteration (with Cholevsky)

        raise NameError('Use balreal_iter_old instead!')

    # find min size (only if iter used)
    cc, co = Qck.shape[1], Qok.shape[1]
    if Print:
        print('cc=%.2d, co=%.2d' % (cc, co))
        print('rank(Zc)=%.4d\trank(Zo)=%.4d' % (rcmax, romax))

    # build M matrix and SVD
    M = libsp.dot(Qok.T, Qck)
    U, s, Vh = scalg.svd(M, full_matrices=False)

    if outFacts:
        return s, Qck, Qok

    else:
        sinv = s ** (-0.5)
        T = libsp.dot(Qck, Vh.T * sinv)
        Tinv = np.dot((U * sinv).T, Qok.T)

    if Print:
        print('rank(Zc)=%.4d\trank(Zo)=%.4d' % (rcmax, romax))

    return s, T, Tinv, rcmax, romax


def balreal_iter_old(A, B, C, lowrank=True, tolSmith=1e-10, tolSVD=1e-6, kmax=None,
                     tolAbs=False):
    """
    Find balanced realisation of DLTI system.

    Notes: Lyapunov equations are solved using iterative squared Smith
    algorithm, in its low or full rank version. These implementations are
    as per the low_rank_smith and smith_iter functions respectively but,
    for computational efficiency,, the iterations are rewritten here so as to
    solve for the observability and controllability Gramians contemporary.
    """

    ### Solve Lyapunov equations
    # Notation reminder:
    # scipy: A X A.T - X = -Q
    # contr: A W A.T - W = - B B.T
    # obser: A.T W A - W = - C.T C
    # low-rank smith: A.T X A - X = -Q Q.T

    if lowrank:  # low-rank square-Smith iteration (with SVD)

        # matrices size
        N = A.shape[0]
        rB = B.shape[1]
        rC = C.shape[0]

        # initialise smith iteration
        DeltaNorm = 1e6
        print('Iter\tMaxZhere')
        kk = 0
        Apow = A
        Qck = B
        Qok = C.T

        while DeltaNorm > tolSmith:
            ### compute products Ak^2 * Zk
            ### (use block Arnoldi)
            Qcright = np.dot(Apow, Qck)
            Qoright = np.dot(Apow.T, Qok)
            Apow = np.dot(Apow, Apow)

            ### enlarge Z matrices
            Qck = np.concatenate((Qck, Qcright), axis=1)
            Qok = np.concatenate((Qok, Qoright), axis=1)

            ### check convergence without reconstructing the added term
            MaxZhere = max(np.max(np.abs(Qoright)), np.max(np.abs(Qcright)))
            print('%.4d\t%.3e' % (kk, MaxZhere))
            DeltaNorm = MaxZhere

            # fixed columns chopping
            if kmax is None:
                # cheap SVD truncation
                if Qck.shape[1] > .4 * N or Qok.shape[1] > .4 * N:
                    Uc, svc = scalg.svd(Qck, full_matrices=False)[:2]
                    Uo, svo = scalg.svd(Qok, full_matrices=False)[:2]
                    if tolAbs:
                        rcmax = np.sum(svc > tolSVD)
                        romax = np.sum(svo > tolSVD)
                    else:
                        rcmax = np.sum(svc > tolSVD * svc[0])
                        romax = np.sum(svo > tolSVD * svo[0])
                    pmax = max(rcmax, romax)
                    Qck = Uc[:, :pmax] * svc[:pmax]
                    Qok = Uo[:, :pmax] * svo[:pmax]
                # Qck_old=np.dot(Uc[:,:pmax],np.diag(svc[:pmax]))
                # Qok_old=np.dot(Uo[:,:pmax],np.diag(svo[:pmax]))
                # Qck=np.dot(Uc[:,:rcmax],np.diag(svc[:rcmax]))
                # Qok=np.dot(Uo[:,:romax],np.diag(svo[:romax]))
            else:
                if Qck.shape[1] > kmax:
                    Uc, svc = scalg.svd(Qck, full_matrices=False)[:2]
                    Qck = Uc[:, :kmax] * svc[:kmax]
                if Qok.shape[1] > kmax:
                    Uo, svo = scalg.svd(Qok, full_matrices=False)[:2]
                    Qok = Uo[:, :kmax] * svo[:kmax]

            ### update
            kk = kk + 1

        del Apow
        Qc, Qo = Qck, Qok

    else:  # full-rank squared smith iteration (with Cholevsky)

        # first iteration
        Wc = np.dot(B, B.T)
        Wo = np.dot(C.T, C)
        Apow = A
        AXAobs = np.dot(np.dot(A.T, Wo), A)
        AXActrl = np.dot(np.dot(A, Wc), A.T)
        DeltaNorm = max(np.max(np.abs(AXAobs)), np.max(np.abs(AXActrl)))

        kk = 1
        print('Iter\tRes')
        while DeltaNorm > tolSmith:
            kk = kk + 1

            # update
            Wo = Wo + AXAobs
            Wc = Wc + AXActrl

            # incremental
            Apow = np.dot(Apow, Apow)
            AXAobs = np.dot(np.dot(Apow.T, Wo), Apow)
            AXActrl = np.dot(np.dot(Apow, Wc), Apow.T)
            DeltaNorm = max(np.max(np.abs(AXAobs)), np.max(np.abs(AXActrl)))
            print('%.4d\t%.3e' % (kk, DeltaNorm))
        # final update (useless in very low tolerance)
        Wo = Wo + AXAobs
        Wc = Wc + AXActrl

        # Choleski factorisation: W=Q Q.T. If unsuccessful, directly solve
        # eigenvalue problem
        Qc = scalg.cholesky(Wc).T
        Qo = scalg.cholesky(Wo).T
    # # eigenvalues are normalised by one, hence Tinv and T matrices
    # # here are not scaled
    # ssq,Tinv,T=scalg.eig(np.dot(Wc,Wo),left=True,right=True)
    # Tinv=Tinv.T
    # #Tinv02=Tinv02.T
    # S=np.diag(np.sqrt(ssq))
    # return S,T,Tinv

    # find min size (only if iter used)
    cc, co = Qc.shape[1], Qo.shape[1]
    cmin = min(cc, co)
    print('cc=%.2d, co=%.2d' % (cc, co))

    # build M matrix and SVD
    M = np.dot(Qo.T, Qc)

    # ### not optimised
    # U,s,Vh=scalg.svd(M,full_matrices=True)
    # U,Vh,s=U[:,:cmin],Vh[:cmin,:],s[:cmin]
    # S=np.diag(s)
    # Sinv=np.diag(1./s)
    # V=Vh.T
    # # Build transformation matrices
    # T=np.dot(Qc,np.dot(V,np.sqrt(Sinv)))
    # Tinv=np.dot(np.sqrt(Sinv),np.dot(U.T,Qo.T))

    ### optimised
    U, s, Vh = scalg.svd(M, full_matrices=True)  # as M is square, full_matrices has no effect
    sinv = s ** (-0.5)
    T = np.dot(Qc, Vh.T * sinv)
    Tinv = np.dot((U * sinv).T, Qo.T)

    return s, T, Tinv


def smith_iter(S, T, tol=1e-8, Square=True):
    """
    Solves the Stein equation
        S.T X S - X = -T
    by mean of Smith or squared-Smith algorithm. Note that a solution X exists
    only if the eigenvalues of S are stricktly smaller than one, and the
    algorithm will not converge otherwise. The algorithm can not exploit
    sparsity, hence, while convergence can be improved for very large matrices,
    it can not be employed if matrices are too large to be stored in memory.

    Ref. Penzt, "A cyclic low-rank Smith method for large sparse Lyapunov
    equations", 2000.
    """

    N = S.shape[0]

    if Square:

        # first iteration
        X = T
        Spow = S
        STXS = np.dot(np.dot(S.T, X), S)
        DeltaNorm = np.max(np.abs(STXS))

        # # second iteration:
        # # can be removed using Spow=np.dot(Spow,Spow)
        # X=X+STXS
        # S=np.dot(S,S)
        # Spow=S
        # STXS=np.dot(np.dot(Spow.T,X),Spow)
        # DeltaNorm=np.max(np.abs(STXS))

        counter = 1
        print('Iter\tRes')
        while DeltaNorm > tol:
            counter = counter + 1

            # update
            X = X + STXS

            # incremental
            # Spow=np.dot(Spow,S) # use this if uncomment second iter
            Spow = np.dot(Spow, Spow)
            STXS = np.dot(np.dot(Spow.T, X), Spow)
            DeltaNorm = np.max(np.abs(STXS))

            print('%.4d\t%.3e' % (counter, DeltaNorm))

    else:
        # first iteration
        X = T
        Spow = S
        STTS = np.dot(np.dot(Spow.T, T), Spow)
        DeltaNorm = np.max(np.abs(STTS))

        counter = 1
        print('Iter\tRes')
        while DeltaNorm > tol:
            counter = counter + 1

            # update
            X = X + STTS

            # incremental
            Spow = np.dot(Spow, S)
            STTS = np.dot(np.dot(Spow.T, T), Spow)
            DeltaNorm = np.max(np.abs(STTS))

            print('%.4d\t%.3e' % (counter, DeltaNorm))

    print('Error %.2e achieved after %.4d iteration!' % (DeltaNorm, counter))

    return X


def res_discrete_lyap(A, Q, Z, Factorised=True):
    """
    Provides residual of discrete Lyapunov equation:
        A.T X A - X = -Q Q.T
    If Factorised option is true,
        X=Z*Z.T
    otherwise X=Z is chosen.

    Reminder:
    contr: A W A.T - W = - B B.T
    obser: A.T W A - W = - C.T C
    """

    if Factorised:
        X = np.dot(Z, Z.T)
    else:
        X = Z
    R = np.dot(A.T, np.dot(X, A)) - X + np.dot(Q, Q.T)
    resinf = np.max(np.abs(R))

    return resinf


def low_rank_smith(A, Q, tol=1e-10, Square=True, tolSVD=1e-12, tolAbs=False,
                   kmax=None, fullOut=True, Convergence='Zk'):
    """
    Low-rank smith algorithm for Stein equation
        A.T X A - X = -Q Q.T
    The algorithm can only be used if T is symmetric positive-definite, but this
    is not checked in this routine for computational performance. The solution X
    is provided in its factorised form:
        X=Z Z.T
    As in the most general case,  a solution X exists only if the eigenvalues of
    S are stricktly smaller than one, and the algorithm will not converge
    otherwise. The algorithm can not exploits parsity, hence, while convergence
    can be improved for very large matrices, it can not be employed if matrices
    are too large to be stored in memory.

    Parameters:
    - tol: tolerance for stopping convergence of Smith algorithm
    - Square: if true the squared-Smith algorithm is used
    - tolSVD: tolerance for reduce Z matrix based on singular values
    - kmax: if given, the Z matrix is forced to have size kmax
    - tolAbs: if True, the tolerance
    - fullOut: not implemented
    - Convergence: 'Zk','res'.
        - If 'Zk' the iteration is stopped when the inf norm of the incremental
        matrix goes below tol.
        - If 'res' the residual of the Lyapunov equation is computed. This
        strategy may fail to converge if kmax is too low or tolSVD too large!

    Ref. P. Benner, G.E. Khoury and M. Sadkane, "On the squared Smith method for
    large-scale Stein equations", 2014.
    """

    N = A.shape[0]
    ncol = Q.shape[1]
    AT = A.T

    DeltaNorm = 1e6
    print('Iter\tMaxZhere')

    kk = 0
    SvList = []
    ZcColList = []

    if Square:  # ------------------------------------------------- squared iter
        Zk = Q
        while DeltaNorm > tol:

            ### compute product Ak^2 * Zk
            ###  use block Arnoldi

            ## too expensive!!
            # Zright=Zk
            # for ii in range(2**kk):
            # 	Zright=np.dot(AT,Zright)
            Zright = np.dot(AT, Zk)
            AT = np.dot(AT, AT)

            ### enlarge Z matrix
            Zk = np.concatenate((Zk, Zright), axis=1)

            ### check convergence
            if Convergence == 'Zk':
                ### check convergence without reconstructing the added term
                MaxZhere = np.max(np.abs(Zright))
                print('%.4d\t%.3e' % (kk, MaxZhere))
                DeltaNorm = MaxZhere
            elif Convergence == 'res':
                ### check convergence through residual
                resinf = res_discrete_lyap(A, Q, Zk, Factorised=True)
                print('%.4d\t%.3e\t%.3e' % (kk, MaxZhere, resinf))
                DeltaNorm = resinf

            # cheap SVD truncation
            U, sv, Vh = scalg.svd(Zk, full_matrices=False)
            # embed()

            if kmax == None:
                if tolAbs:
                    pmax = np.sum(sv > tolSVD)
                else:
                    pmax = np.sum(sv > tolSVD * sv[0])
            else:
                pmax = kmax

            Ut = U[:, :pmax]
            svt = sv[:pmax]
            # Vht=Vh[:pmax,:]
            # Zkrec=np.dot(Ut,np.dot(np.diag(svt),Vht))
            Zk = np.dot(Ut, np.diag(svt))

            ### update
            kk = kk + 1


    else:  # -------------------------------------------------------- smith iter
        raise NameError(
            'Smith method without SVD will lead to extremely large matrices')

        Zk = []
        Zk.append(Q)
        while DeltaNorm > tol:
            Zk.append(np.dot(AT, Zk[-1]))
            kk = kk + 1
            # check convergence without reconstructing Z*Z.T
            MaxZhere = np.max(np.abs(Zk[-1]))
            print('%.4d\t%.3e' % (kk, MaxZhere))
            DeltaNorm = MaxZhere
        Zk = np.concatenate(tuple(Zk), axis=1)

    return Zk


### utilities for balfreq

def get_trapz_weights(k0, kend, Nk, knyq=False):
    """
    Returns uniform frequency grid (kv of length Nk) and weights (wv) for
    Gramians integration using trapezoidal rule. If knyq is True, it is assumed
    that kend is also the Nyquist frequency.
    """

    assert k0 >= 0. and kend >= 0., 'Frequencies must be positive!'

    dk = (kend - k0) / (Nk - 1.)
    kv = np.linspace(k0, kend, Nk)
    wv = np.ones((Nk,)) * dk * np.sqrt(2)

    if k0 / (kend - k0) < 1e-10:
        wv[0] = .5 * dk
    else:
        wv[0] = dk / np.sqrt(2)

    if knyq:
        wv[-1] = .5 * dk
    else:
        wv[-1] = dk / np.sqrt(2)

    return kv, wv


def get_gauss_weights(k0, kend, Npart, order):
    """
    Returns gauss-legendre frequency grid (kv of length Npart*order) and
    weights (wv) for Gramians integration.

    The integration grid is divided into Npart partitions, and in each of
    them integration is performed using a Gauss-Legendre quadrature of
    order order.

    Note: integration points are never located at k0 or kend, hence there
    is no need for special treatment as in (for e.g.) a uniform grid case
    (see get_unif_weights)
    """

    if Npart == 1:
        # get gauss normalised coords and weights
        xad, wad = np.polynomial.legendre.leggauss(order)
        krange = kend - k0
        kv = .5 * (k0 + kend) + .5 * krange * xad
        wv = wad * (.5 * krange) * np.sqrt(2)
        print('partitioning: %.3f to %.3f' % (k0, kend))

    else:
        kv = np.zeros((Npart * order,))
        wv = np.zeros((Npart * order,))

        dk_part = (kend - k0) / Npart

        for ii in range(Npart):
            k0_part = k0 + ii * dk_part
            kend_part = k0_part + dk_part
            iivec = range(order * ii, order * (ii + 1))
            kv[iivec], wv[iivec] = get_gauss_weights(k0_part, kend_part, Npart=1, order=order)

    return kv, wv


def balfreq(SS, DictBalFreq):
    """
    Method for frequency limited balancing.

    The Observability and controllability Gramians over the frequencies kv
    are solved in factorised form. Balanced modes are then obtained with a
    square-root method.

    Details:

        * Observability and controllability Gramians are solved in factorised form
          through explicit integration. The number of integration points determines
          both the accuracy and the maximum size of the balanced model.

        * Stability over all (Nb) balanced states is achieved if:

            a. one of the Gramian is integrated through the full Nyquist range
            b. the integration points are enough.


    Input:

    - DictBalFreq: dictionary specifying integration method with keys:

        - ``frequency``: defines limit frequencies for balancing. The balanced
           model will be accurate in the range ``[0,F]``, where ``F`` is the value of
           this key. Note that ``F`` units must be consistent with the units specified
           in the ``self.ScalingFacts`` dictionary.

        - ``method_low``: ``['gauss','trapz']`` specifies whether to use gauss
          quadrature or trapezoidal rule in the low-frequency range ``[0,F]``.

        - ``options_low``: options to use for integration in the low-frequencies.
          These depend on the integration scheme (See below).

        - ``method_high``: method to use for integration in the range [F,F_N],
          where F_N is the Nyquist frequency. See 'method_low'.

        - ``options_high``: options to use for integration in the high-frequencies.

        - ``check_stability``: if True, the balanced model is truncated to
          eliminate unstable modes - if any is found. Note that very accurate
          balanced model can still be obtained, even if high order modes are
          unstable. Note that this option is overridden if ""

        - ``get_frequency_response``: if True, the function also returns the
          frequency response evaluated at the low-frequency range integration
          points. If True, this option also allows to automatically tune the
          balanced model.


    Future options:
        - Ncpu: for parallel run


    The following integration schemes are available:
        - ``trapz``: performs integration over equally spaced points using
          trapezoidal rule. It accepts options dictionaries with keys:

             - ``points``: number of integration points to use (including
               domain boundary)

        - ``gauss`` performs gauss-lobotto quadrature. The domain can be
          partitioned in Npart sub-domain in which the gauss-lobotto quadrature
          of order Ord can be applied. A total number of Npart*Ord points is
          required. It accepts options dictionaries of the form:

             - ``partitions``: number of partitions

             - ``order``: quadrature order.


    Examples:

        The following dictionary

        >>>   DictBalFreq={'frequency': 1.2,
        >>>                'method_low': 'trapz',
        >>>                'options_low': {'points': 12},
        >>>                'method_high': 'gauss',
        >>>                'options_high': {'partitions': 2, 'order': 8},
        >>>                'check_stability': True }


        balances the state-space model in the frequency range [0, 1.2]
        using:

            a. 12 equally-spaced points integration of the Gramians in
               the low-frequency range [0,1.2] and

            b. A 2 Gauss-Lobotto 8-th order quadratures of the controllability
               Gramian in the high-frequency range.


        A total number of 28 integration points will be required, which will
        result into a balanced model with number of states

        >>>    min{ 2*28* number_inputs, 2*28* number_outputs }


        The model is finally truncated so as to retain only the first Ns stable
        modes.
    """

    ### check input dictionary
    if 'frequency' not in DictBalFreq:
        raise NameError('Solution dictionary must include the "frequency" key')

    if 'method_low' not in DictBalFreq:
        warnings.warn('Setting default options for low-frequency integration')
        DictBalFreq['method_low'] = 'trapz'
        DictBalFreq['options_low'] = {'points': 12}

    if 'method_high' not in DictBalFreq:
        warnings.warn('Setting default options for high-frequency integration')
        DictBalFreq['method_high'] = 'gauss'
        DictBalFreq['options_high'] = {'partitions': 2, 'order': 8}

    if 'check_stability' not in DictBalFreq:
        DictBalFreq['check_stability'] = True

    if 'output_modes' not in DictBalFreq:
        DictBalFreq['output_modes'] = True

    if 'get_frequency_response' not in DictBalFreq:
        DictBalFreq['get_frequency_response'] = False

    ### get integration points and weights

    # Nyquist frequency
    kn = np.pi / SS.dt

    Opt = DictBalFreq['options_low']
    if DictBalFreq['method_low'] == 'trapz':
        kv_low, wv_low = get_trapz_weights(0., DictBalFreq['frequency'],
                                           Opt['points'], False)
    elif DictBalFreq['method_low'] == 'gauss':
        kv_low, wv_low = get_gauss_weights(0., DictBalFreq['frequency'],
                                           Opt['partitions'], Opt['order'])
    else:
        raise NameError(
            'Invalid value %s for key "method_low"' % DictBalFreq['method_low'])

    Opt = DictBalFreq['options_high']
    if DictBalFreq['method_high'] == 'trapz':
        if Opt['points'] == 0:
            warnings.warn('You have chosen no points in high frequency range!')
            kv_high, wv_high = [], []
        else:
            kv_high, wv_high = get_trapz_weights(DictBalFreq['frequency'], kn,
                                                 Opt['points'], True)
    elif DictBalFreq['method_high'] == 'gauss':
        if Opt['order'] * Opt['partitions'] == 0:
            warnings.warn('You have chosen no points in high frequency range!')
            kv_high, wv_high = [], []
        else:
            kv_high, wv_high = get_gauss_weights(DictBalFreq['frequency'], kn,
                                                 Opt['partitions'], Opt['order'])
    else:
        raise NameError(
            'Invalid value %s for key "method_high"' % DictBalFreq['method_high'])

    ### -------------------------------------------------- loop frequencies

    ### merge vectors
    Nk_low = len(kv_low)
    kvdt = np.concatenate((kv_low, kv_high)) * SS.dt
    wv = np.concatenate((wv_low, wv_high)) * SS.dt
    zv = np.cos(kvdt) + 1.j * np.sin(kvdt)

    Eye = libsp.eye_as(SS.A)
    Zc = np.zeros((SS.states, 2 * SS.inputs * len(kvdt)), )
    Zo = np.zeros((SS.states, 2 * SS.outputs * Nk_low), )

    if DictBalFreq['get_frequency_response']:
        Yfreq = np.empty((SS.outputs, SS.inputs, Nk_low,), dtype=np.complex_)
        kv = kv_low

    for kk in range(len(kvdt)):

        zval = zv[kk]
        Intfact = wv[kk]  # integration factor

        Qctrl = Intfact * libsp.solve(zval * Eye - SS.A, SS.B)
        kkvec = range(2 * kk * SS.inputs, 2 * (kk + 1) * SS.inputs)
        Zc[:, kkvec[:SS.inputs]] = Qctrl.real
        Zc[:, kkvec[SS.inputs:]] = Qctrl.imag

        ### ----- frequency response
        if DictBalFreq['get_frequency_response'] and kk < Nk_low:
            Yfreq[:, :, kk] = (1. / Intfact) * \
                              libsp.dot(SS.C, Qctrl, type_out=np.ndarray) + SS.D

        ### ----- observability
        if kk >= Nk_low:
            continue

        Qobs = Intfact * libsp.solve(np.conj(zval) * Eye - SS.A.T, SS.C.T)

        kkvec = range(2 * kk * SS.outputs, 2 * (kk + 1) * SS.outputs)
        Zo[:, kkvec[:SS.outputs]] = Intfact * Qobs.real
        Zo[:, kkvec[SS.outputs:]] = Intfact * Qobs.imag

    # delete full matrices
    Kernel = None
    Qctrl = None
    Qobs = None

    # LRSQM (optimised)
    U, hsv, Vh = scalg.svd(np.dot(Zo.T, Zc), full_matrices=False)
    sinv = hsv ** (-0.5)
    T = np.dot(Zc, Vh.T * sinv)
    Ti = np.dot((U * sinv).T, Zo.T)
    # Zc,Zo=None,None

    ### build frequency balanced model
    Ab = libsp.dot(Ti, libsp.dot(SS.A, T))
    Bb = libsp.dot(Ti, SS.B)
    Cb = libsp.dot(SS.C, T)
    SSb = libss.StateSpace(Ab, Bb, Cb, SS.D, dt=SS.dt)

    ### Eliminate unstable modes - if any:
    if DictBalFreq['check_stability']:
        for nn in range(1, len(hsv) + 1):
            eigs_trunc = scalg.eigvals(SSb.A[:nn, :nn])
            eigs_trunc_max = np.max(np.abs(eigs_trunc))
            if eigs_trunc_max > 1. - 1e-16:
                SSb.truncate(nn - 1)
                hsv = hsv[:nn - 1]
                T = T[:, :nn - 1]
                Ti = Ti[:nn - 1, :]
                break

    outs = (SSb, hsv)
    if DictBalFreq['output_modes']:
        outs += (T, Ti, Zc, Zo, U, Vh)
    return outs


def modred(SSb, N, method='residualisation'):
    """
    Produces a reduced order model with N states from balanced or modal system
    SSb.
    Both "truncation" and "residualisation" methods are employed.

    Note:
    - this method is designed for small size systems, i.e. a deep copy of SSb is
    produced by default.
    """

    assert method in ['residualisation', 'realisation', 'truncation'], \
        "method must be equal to 'residualisation' or 'truncation'!"
    assert SSb.dt is not None, 'SSb is not a DLTI!'

    Nb = SSb.A.shape[0]
    if Nb == N:
        SSrom = libss.StateSpace(SSb.A, SSb.B, SSb.C, SSb.D, dt=SSb.dt)
        return SSrom

    A11 = SSb.A[:N, :N]
    B11 = SSb.B[:N, :]
    C11 = SSb.C[:, :N]
    D = SSb.D

    if method == 'truncation':
        SSrom = libss.StateSpace(A11, B11, C11, D, dt=SSb.dt)
    else:
        Nb = SSb.A.shape[0]
        IA22inv = -SSb.A[N:, N:].copy()
        eevec = range(Nb - N)
        IA22inv[eevec, eevec] += 1.
        IA22inv = scalg.inv(IA22inv, overwrite_a=True)

        SSrom = libss.StateSpace(
            A11 + np.dot(SSb.A[:N, N:], np.dot(IA22inv, SSb.A[N:, :N])),
            B11 + np.dot(SSb.A[:N, N:], np.dot(IA22inv, SSb.B[N:, :])),
            C11 + np.dot(SSb.C[:, N:], np.dot(IA22inv, SSb.A[N:, :N])),
            D + np.dot(SSb.C[:, N:], np.dot(IA22inv, SSb.B[N:, :])),
            dt=SSb.dt)

    return SSrom


def tune_rom(SSb, kv, tol, gv, method='realisation', convergence='all', Print=False):
    """
    Starting from a balanced DLTI, this function determines the number of states
    N required in a ROM (obtained either through 'residualisation' or
    'truncation' as specified in method - see also librom.modred) to match the
    frequency response of SSb over the frequency array, kv, with absolute
    accuracy tol. gv contains the balanced system Hankel singular value, and is
    used to determine the upper bound for the ROM order N.

    Unless kv does not conver the full Nyquist frequency range, the ROM accuracy
    is not guaranteed to increase monothonically with the number of states. To
    account for this, two criteria can be used to determine the ROM convergence:

        - convergence='all': in this case, the number of ROM states N is chosen
        such that any ROM of order greater than N produces an error smaller than
        tol. To guarantee this the ROM frequency response is computed for all
        N<=Nb, where Nb is the number of balanced states. This method is
        numerically inefficient.

        - convergence='min': atempts to find the minimal number of states to
        achieve the accuracy tol.

    Note:
    - the input state-space model, SSb, must be balanced.
    - the routine in not implemented for numerical efficiency and assumes that
    SSb is small.
    """

    # reference frequency response
    Nb = SSb.A.shape[0]
    Yb = libss.freqresp(SSb, kv, dlti=True)
    if gv is None:
        Nmax = Nb
    else:
        Nmax = min(np.sum(gv > tol) + 1, Nb)

    if convergence == 'all':
        # start from larger size and decrease untill the ROm accuracy is over tol
        Found = False
        N = Nmax
        while not Found:
            SSrom = modred(SSb, N, method)
            Yrom = libss.freqresp(SSrom, kv, dlti=True)
            er = np.max(np.abs(Yrom - Yb))
            if Print:
                print('N=%.3d, er:%.2e (tol=%.2e)' % (N, er, tol))

            if N == Nmax and er > tol:
                warnings.warn(
                    'librom.tune_rom: error %.2e above tolerance %.2e and HSV bound %.2e' \
                    % (er, tol, gv[N - 1]))
            # raise NameError('Hankel singluar values do not '\
            # 				'provide a bound for error! '\
            # 				'The balanced system may not be accurate')
            if er < tol:
                N -= 1
            else:
                N += 1
                Found = True
                SSrom = modred(SSb, N, method)

    elif convergence == 'min':
        Found = False
        N = 1
        while not Found:
            SSrom = modred(SSb, N, method)
            Yrom = libss.freqresp(SSrom, kv, dlti=True)
            er = np.max(np.abs(Yrom - Yb))
            if Print:
                print('N=%.3d, er:%.2e (tol=%.2e)' % (N, er, tol))
            if er < tol:
                Found = True

            else:
                N += 1

    else:
        raise NameError("'convergence' method not implemented")

    return SSrom


def eigen_dec(A, B, C, dlti=True, N=None, eigs=None, UR=None, URinv=None,
              order_by='damp', tol=1e-10, complex=False):
    """
    Eigen decomposition of state-space model (either discrete or continuous time)
    defined by the A,B,C matrices. Eigen-states are organised in decreasing
    damping order or increased frequency order such that the truncation

        ``A[:N,:N], B[:N,:], C[:,:N]``

    will retain the least N damped (or lower frequency) modes.

    If the eigenvalues of A, eigs, are complex, the state-space is automatically
    convert into real by separating its real and imaginary part. This procedure
    retains the minimal number of states as only 2 equations are added for each
    pair of complex conj eigenvalues. Extra care is however required when
    truncating the system, so as to ensure that the chosen value of N does not
    retain the real part, but not the imaginary part, of a complex pair.

    For this reason, the function also returns an optional output, ``Nlist``, such
    that, for each N in Nlist, the truncation
        A[:N,:N], B[:N,:], C[:,:N]
    does guarantee that both the real and imaginary part of a complex conj pair
    is included in the truncated model. Note that if ```order_by == None``, the eigs
    and UR must be given in input and must be such that complex pairs are stored
    consecutively.


    Args:
        A: state-space matrix
        B: state-space matrix
        C: matrices of state-space model
        dlti: specifies whether discrete (True) or continuous-time. This information
            is only required to order the eigenvalues in decreasing dmaping order
        N: number of states to retain. If None, all states are retained
        eigs,Ur: eigenvalues and right eigenvector of A matrix as given by:
            eigs,Ur=scipy.linalg.eig(A,b=None,left=False,right=True)
        Urinv: inverse of Ur
        order_by={'damp','freq','stab'}: order according to increasing damping (damp)
        or decreasing frequency (freq) or decreasing damping (stab).
            If None, the same order as eigs/UR is followed.
        tol: absolute tolerance used to identify complex conj pair of eigenvalues
        complex: if true, the system is left in complex form


    Returns:
    (Aproj,Bproj,Cproj): state-space matrices projected over the first N (or N+1
        if N removes the imaginary part equations of a complex conj pair of
        eigenvalues) related to the least damped modes
    Nlist: list of acceptable truncation values
    """

    if N == None:
        N = A.shape[0]

    if order_by is None:
        assert ((eigs is not None) and (UR is not None)), \
            'Specify criterion to order eigenvalues or provide both eigs and UR'

    ### compute eigevalues/eigenvectors
    if eigs is None:
        eigs, UR = scalg.eig(A, b=None, left=False, right=True)
    if URinv is None:
        try:
            URinv = np.linalg.inv(UR)
        except LinAlgError:
            print('The A matrix can not be diagonalised as does not admit ' \
                  'linearly independent eigenvectors')

    ### order eigenvalues/eigenvectors (or verify format)
    if order_by is None:
        # verify format
        nn = 0
        while nn < N:
            if np.abs(eigs[nn].imag) > tol:
                if nn < N - 1:
                    assert np.abs(eigs[nn].imag + eigs[nn + 1].imag) < tol, \
                        'When order_by is None, eigs and UR much be organised such ' \
                        'that complex conj pairs are consecutives'
                else:
                    assert np.abs(eigs[nn].imag + eigs[nn - 1].imag) < tol, \
                        'When order_by is None, eigs and UR much be organised such ' \
                        'that complex conj pairs are consecutives'
                nn += 2
            else:
                nn += 1
    else:
        if order_by == 'damp':
            if dlti:
                order = np.argsort(np.abs(eigs))[::-1]
            else:
                order = np.argsort(eigs.real)[::-1]
        elif order_by == 'freq':
            if dlti:
                order = np.argsort(np.abs(np.angle(eigs)))
            else:
                order = np.argsort(np.abs(eigs.imag))
        elif order_by == 'stab':
            if dlti:
                order = np.argsort(np.abs(eigs))
            else:
                order = np.argsort(eigs.real)
        else:
            raise NameError("order_by must be equal to 'damp' or 'freq'")
        eigs = eigs[order]
        UR = UR[:, order]
        URinv = URinv[order, :]

    ### compute list of available truncation size, Nlist
    Nlist = []
    nn = 0
    while nn < N:
        # check if eig are complex conj
        if nn < N - 1 and np.abs(eigs[nn] - eigs[nn + 1].conjugate()) < tol:
            nn += 2
        else:
            nn += 1
        Nlist.append(nn)
    assert Nlist[-1] >= N, \
        'Something failed when identifying the admissible truncation sizes'
    if Nlist[-1] > N:
        warnings.warn(
            'Resizing the eigendecomposition from %.3d to %.3d states' \
            % (N, Nlist[-1]))
        N = Nlist[-1]

    ### build complex form
    if complex:
        Aproj = np.diag(eigs[:N])
        Bproj = np.dot(URinv[:N, :], B)
        Cproj = np.dot(C, UR[:, :N])
        return Aproj, Bproj, Cproj, Nlist

    ### build real values form
    Aproj = np.zeros((N, N))
    Bproj = np.zeros((N, B.shape[1]))
    Cproj = np.zeros((C.shape[0], N))
    nn = 0
    while nn < N:
        # redundant check
        if (nn + 1 in Nlist) and np.abs(eigs[nn].imag) < tol:
            Aproj[nn, nn] = eigs[nn].real
            Bproj[nn, :] = np.dot(URinv[nn, :].real, B)
            Cproj[:, nn] = np.dot(C, UR[:, nn].real)
            nn += 1
        else:
            Aproj[nn, nn] = eigs[nn].real
            Aproj[nn, nn + 1] = -eigs[nn].imag
            Aproj[nn + 1, nn] = eigs[nn].imag
            Aproj[nn + 1, nn + 1] = eigs[nn].real
            #
            Bproj[nn, :] = np.dot(URinv[nn, :].real, B)
            Bproj[nn + 1, :] = np.dot(URinv[nn, :].imag, B)
            #
            Cproj[:, nn] = 2. * np.dot(C, UR[:, nn].real)
            Cproj[:, nn + 1] = -2. * np.dot(C, UR[:, nn].imag)
            nn += 2

    return Aproj, Bproj, Cproj, Nlist


def check_stability(A, dt=True):
    """
    Checks the stability of the system.

    Args:
        A (np.ndarray): System plant matrix
        dt (bool): Discrete time system

    Returns:
        bool: True if the system is stable
    """
    eigvals = scalg.eigvals(A)
    if dt:
        criteria = np.abs(eigvals) > 1.
    else:
        criteria = np.real(eigvals) > 0.0

    if np.sum(criteria) >= 1.0:
        return True
    else:
        return False
