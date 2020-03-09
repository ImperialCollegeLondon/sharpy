import warnings

import numpy as np
from scipy import linalg as sclalg

from sharpy.linear.src import libss as libss
from sharpy.utils import algebra as algebra


class InterpROM:
    r"""
    State-space 1D interpolation class.

    This class allows interpolating from a list of state-space models, SS.

    State-space models are required to have the same number of inputs and outputs
    and need to have the same number of states.

    For state-space interpolation, state-space models also need to be defined
    over the same set of generalised coordinates. If this is not the case, the
    projection matrices W and V used to produce the ROMs, ie

    .. math:: \mathbf{A}_{proj} = \mathbf{W}^\top \mathbf{A V}

    where A is the full-states matrix, also need to be provided. This will allow
    projecting the state-space models onto a common set of generalised
    coordinates before interpoling.

    For development purposes, the method currently creates a hard copy of the
    projected matrices into the self.AA, self.BB, self.CC lists


    Inputs:

    - SS: list of state-space models (instances of libss.ss class)

    - VV: list of V matrices used to produce SS. If None, it is assumed that
      ROMs are defined over the same basis

    - WWT: list of W^T matrices used to derive the ROMs.

    - Vref, WTref: reference subspaces for projection. Some methods neglect this
      input (e.g. panzer)

    - method_proj: method for projection of state-space models over common
      coordinates. Available options are:

        - leastsq: find left/right projectors using least squares approx. Suitable
          for all basis.

        - strongMAC: strong Modal Assurance Criterion [4] enforcement for general
          basis. See Ref. [3], Eq. (7)

        - strongMAC_BT: strong Modal Assurance Criterion [4] enforcement for
          basis obtained by Balanced Truncation. Equivalent to strongMAC

        - maraniello_BT: this is equivalent to strongMAC and strongMAC_BT but
          avoids inversions. However, performance are the same as other strongMAC
          approaches - it works only when basis map the same subspaces

        - weakMAC_right_orth: weak MAC enforcement [1,3] for state-space models
          with right orthonoraml basis, i.e. V.T V = I. This is like Ref. [1], but
          implemented only on one side.

        - weakMAC: implementation of weak MAC enforcement for a general system.
          The method orthonormalises the right basis (V) and then solves the
          orthogonal Procrustes problem.

        - for orthonormal basis (V.T V = I): !!! These methods are not tested !!!

            - panzer: produces a new reference point based on svd [2]
            - amsallem: project over Vref,WTref [1]

    References:

    [1] D. Amsallem and C. Farhat, An online method for interpolating linear
    parametric reduced-order models, SIAM J. Sci. Comput., 33 (2011), pp. 2169–2198.

    [2] Panzer, J. Mohring, R. Eid, and B. Lohmann, Parametric model order
    reduction by matrix interpolation, at–Automatisierungstechnik, 58 (2010),
    pp. 475–484.

    [3] Mahony, R., Sepulchre, R. & Absil, P. -a., 2004. Riemannian Geometry of
    Grassmann Manifolds with a View on Algorithmic Computation. Acta Applicandae
    Mathematicae, 80(2), pp.199–220.

    [4] Geuss, M., Panzer, H. & Lohmann, B., 2013. On parametric model order
    reduction by matrix interpolation. 2013 European Control Conference (ECC),
    pp.3433–3438.


    """

    def __init__(self):
        self.ss_list = None
        self.VV = None
        self.WWT = None

        self.Vref = None
        self.WTref = None
        self.reference_case = None

        self.method_proj = None
        self.projected = False

        self.AA = None
        self.BB = None
        self.CC = None
        self.DD = None

    def initialise(self,
                 ss_list,
                 vv_list=None,
                 wwt_list=None,
                 method_proj=None,
                 reference_case=0):

        self.ss_list = ss_list

        self.check_discrete_timestep()

        self.VV = vv_list
        self.WWT = wwt_list

        self.Vref = self.VV[reference_case]
        self.WTref = self.WWT[reference_case]

        self.method_proj = method_proj

        self.reference_case = reference_case

        self.projected = False
        if self.VV is None or self.WWT is None:
            self.projected = True
            self.AA = [ss_here.A for ss_here in self.ss_list]
            self.BB = [ss_here.B for ss_here in self.ss_list]
            self.CC = [ss_here.C for ss_here in self.ss_list]

        # projection required for D
        self.DD = [ss_here.D for ss_here in self.ss_list]

        ### check state-space models
        Nx, Nu, Ny = self.ss_list[0].states, self.ss_list[0].inputs, self.ss_list[0].outputs
        for ss_here in self.ss_list:
            assert ss_here.states == Nx, \
                'State-space models do not have the same number of states'
            assert ss_here.inputs == Nu, \
                'State-space models do not have the same number of inputs'
            assert ss_here.outputs == Ny, \
                'State-space models do not have the same number of outputs'

    def check_discrete_timestep(self):
        """
        Checks that the systems have the same timestep. If they don't, it converts them to continuous time using
        :func:`sharpy.linear.src.libss.disc2cont()`.
        """
        mismatch_dt = False
        for ss in self.ss_list:
            if ss.dt != self.ss_list[0].dt:
                mismatch_dt = True
                break

        if mismatch_dt:
            for i, ss in enumerate(self.ss_list):
                self.ss_list[i] = libss.disc2cont(ss)

    def __call__(self, wv):
        """
        Evaluate interpolated model using weights wv.
        """

        assert self.projected, ('You must project the state-space models over' +
                                ' a common basis before interpolating.')

        Aint = np.zeros_like(self.AA[0])
        Bint = np.zeros_like(self.BB[0])
        Cint = np.zeros_like(self.CC[0])
        Dint = np.zeros_like(self.DD[0])

        for ii in range(len(self.AA)):
            Aint += wv[ii] * self.AA[ii]
            Bint += wv[ii] * self.BB[ii]
            Cint += wv[ii] * self.CC[ii]
            Dint += wv[ii] * self.DD[ii]

        return libss.ss(Aint, Bint, Cint, Dint, self.ss_list[0].dt)

    def project(self):
        """
        Project the state-space models onto the generalised coordinates of
        state-space model IImap
        """

        self.AA = []
        self.BB = []
        self.CC = []

        self.QQ = []
        self.QQinv = []

        if self.method_proj == 'amsallem':
            warnings.warn('Method untested!')

            for ii in range(len(self.ss_list)):
                U, sv, Z = sclalg.svd(np.dot(self.VV[ii].T, self.Vref),
                                      full_matrices=False,
                                      overwrite_a=False,
                                      lapack_driver='gesdd')
                Q = np.dot(U, Z.T)
                U, sv, Z = sclalg.svd(np.dot(self.WWT[ii], self.WTref.T),
                                      full_matrices=False, overwrite_a=False,
                                      lapack_driver='gesdd')
                Qinv = np.dot(U, Z.T).T
                # Qinv = sclalg.inv(Q)
                self.QQ.append(Q)
                self.QQinv.append(Qinv)

        elif self.method_proj == 'panzer':
            warnings.warn('Method untested!')

            # generate basis
            U, sv = sclalg.svd(np.concatenate(self.VV, axis=1),
                               full_matrices=False, overwrite_a=False,
                               lapack_driver='gesdd')[:2]
            # chop U
            U = U[:, :self.ss_list[0].states]
            for ii in range(len(self.ss_list)):
                Qinv = np.linalg.inv(np.dot(self.WWT[ii], U))
                Q = np.linalg.inv(np.dot(self.VV[ii].T, U))
                self.QQ.append(Q)
                self.QQinv.append(Qinv)

        elif self.method_proj == 'leastsq':

            for ii in range(len(self.ss_list)):
                Q, _, _, _ = sclalg.lstsq(self.VV[ii], self.Vref)
                print('det(Q): %.3e\tcond(Q): %.3e' \
                      % (np.linalg.det(Q), np.linalg.cond(Q)))
                # if cond(Q) is small...
                # Qinv = np.linalg.inv(Q)
                P, _, _, _ = sclalg.lstsq(self.WWT[ii].T, self.WTref.T)
                self.QQ.append(Q)
                self.QQinv.append(P.T)

        elif self.method_proj == 'strongMAC':
            """
            Strong MAC enforcements as per Ref.[4]
            """

            VTVref = np.dot(self.Vref.T, self.Vref)
            for ii in range(len(self.ss_list)):
                Q = np.linalg.solve(np.dot(self.Vref.T, self.VV[ii]), VTVref)
                Qinv = np.linalg.inv(Q)
                print('det(Q): %.3e\tcond(Q): %.3e' \
                      % (np.linalg.det(Q), np.linalg.cond(Q)))
                self.QQ.append(Q)
                self.QQinv.append(Qinv)

        elif self.method_proj == 'strongMAC_BT':
            """
            This is equivalent to Mahony 2004, Eq. 7, for the case of basis
            obtained by balancing. In general, it will fail if VV[ii] and Vref
            do not describe the same subspace
            """

            for ii in range(len(self.ss_list)):
                Q = np.linalg.inv(np.dot(self.WTref, self.VV[ii]))
                Qinv = np.dot(self.WTref, self.VV[ii])
                print('det(Q): %.3e\tcond(Q): %.3e' \
                      % (np.linalg.det(Q), np.linalg.cond(Q)))
                self.QQ.append(Q)
                self.QQinv.append(Qinv)

        elif self.method_proj == 'maraniello_BT':
            """
            Projection over ii. This is a sort of weak enforcement
            """

            for ii in range(len(self.ss_list)):
                Q = np.dot(self.WWT[ii], self.Vref)
                Qinv = np.dot(self.WTref, self.VV[ii])
                print('det(Q): %.3e\tcond(Q): %.3e' \
                      % (np.linalg.det(Q), np.linalg.cond(Q)))
                self.QQ.append(Q)
                self.QQinv.append(Qinv)

        elif self.method_proj == 'weakMAC_right_orth':
            """
            This is like Amsallem, but only for state-space models with right 
            orthogonal basis 
            """

            for ii in range(len(self.ss_list)):
                Q, sc = sclalg.orthogonal_procrustes(self.VV[ii], self.Vref)
                Qinv = Q.T
                print('det(Q): %.3e\tcond(Q): %.3e' \
                      % (np.linalg.det(Q), np.linalg.cond(Q)))
                self.QQ.append(Q)
                self.QQinv.append(Qinv)

        elif self.method_proj == 'weakMAC':
            """
            WeakMAC enforcement on the right hand side basis, V
            """

            # svd of reference
            Uref, svref, Zhref = sclalg.svd(self.Vref, full_matrices=False)

            for ii in range(len(self.ss_list)):
                # svd of basis
                Uhere, svhere, Zhhere = sclalg.svd(self.VV[ii], full_matrices=False)

                R, sc = sclalg.orthogonal_procrustes(Uhere, Uref)
                Q = np.dot(np.dot(Zhhere.T, np.diag(svhere ** (-1))), R)
                Qinv = np.dot(R.T, np.dot(np.diag(svhere), Zhhere))
                print('det(Q): %.3e\tcond(Q): %.3e' \
                      % (np.linalg.det(Q), np.linalg.cond(Q)))
                self.QQ.append(Q)
                self.QQinv.append(Qinv)

        else:
            raise NameError('Projection method %s not implemented!' % self.method_proj)

        ### Project
        for ii in range(len(self.ss_list)):
            self.AA.append(np.dot(self.QQinv[ii], np.dot(self.ss_list[ii].A, self.QQ[ii])))
            self.BB.append(np.dot(self.QQinv[ii], self.ss_list[ii].B))
            self.CC.append(np.dot(self.ss_list[ii].C, self.QQ[ii]))

        self.projected = True


class TangentInterpolation(InterpROM):
    """
    Performs interpolation in the tangent space. This class is inherited from :class:`~.InterpROM`
    with minor modifications to perform the actual interpolation in the tangent manifold to the reference
    system.

    Warnings:
        Interpolation in the tangent space is not fully understood. When transforming to the tangent space
        using the logarithmic mapping, complex terms may appear whose impact/meaning is unknown. For the method to work
        the matrices to interpolate must be regular matrices.

    References:
        [1] D. Amsallem and C. Farhat, An online method for interpolating linear
        parametric reduced-order models, SIAM J. Sci. Comput., 33 (2011), pp. 2169–2198.
    """

    def __init__(self):

        super().__init__()

        self.gamma = None
        self.reference_system = None

        warn_msg = 'TangentInterpolation is not well understood for many systems where complex terms' \
                   ' may appear in the transformation. See documentation.'
        warnings.warn(warn_msg)

    def __call__(self, weights):

        assert self.projected, ('You must project the state-space models over' +
                                ' a common basis before interpolating.')

        a_tan = np.zeros_like(self.AA[0], dtype=complex)
        b_tan = np.zeros_like(self.BB[0])
        c_tan = np.zeros_like(self.CC[0])
        d_tan = np.zeros_like(self.DD[0])

        for i in range(len(self.AA)):
            a_tan += weights[i] * self.gamma[0][i]
            b_tan += weights[i] * self.gamma[1][i]
            c_tan += weights[i] * self.gamma[2][i]
            d_tan += weights[i] * self.gamma[3][i]

        a = self.from_tangent_manifold(a_tan, self.reference_system[0])
        b = self.from_tangent_manifold(b_tan, self.reference_system[1])
        c = self.from_tangent_manifold(c_tan, self.reference_system[2])
        d = self.from_tangent_manifold(d_tan, self.reference_system[3])

        if self.ss_list[0].dt:
            return libss.ss(a, b, c, d, self.ss_list[0].dt)
        else:
            return libss.ss(a, b, c, d)

    def project(self):
        r"""
        Projects the system onto a set of generalised coordinates and creates the matrices in the
        tangent manifold to the reference system.

        See Also:
            Projection methods described in :func:`~sharpy.linear.rom.utils.librom_interp.InterpROM.project()`
        """
        super().project()

        gamma_a = []
        gamma_b = []
        gamma_c = []
        gamma_d = []

        self.reference_system = (self.AA[self.reference_case],
                                 self.BB[self.reference_case],
                                 self.CC[self.reference_case],
                                 self.DD[self.reference_case],
                                 )

        for i in range(len(self.AA)):
            gamma_a.append(self.to_tangent_manifold(self.AA[i], self.reference_system[0]))
            gamma_b.append(self.to_tangent_manifold(self.BB[i], self.reference_system[1]))
            gamma_c.append(self.to_tangent_manifold(self.CC[i], self.reference_system[2]))
            gamma_d.append(self.to_tangent_manifold(self.DD[i], self.reference_system[3]))

        self.gamma = (gamma_a, gamma_b, gamma_c, gamma_d)

    @staticmethod
    def to_tangent_manifold(matrix, ref_matrix):
        r"""
        Based on Table 4.1 in Amsallem and Farhat [1] performs the mapping onto the tangent manifold:

        .. math:: \mathrm{Log}_\mathbf{X}(\mathbf{Y}) = \boldsymbol{\Gamma}

        If the matrices are square, this is calculated as:

        .. math:: \boldsymbol{\Gamma} = \log(\mathbf{YX}^{-1})

        Else:

        .. math:: \boldsymbol{\Gamma} = \mathbf{Y} - \mathbf{X}

        Args:
            matrix (np.ndarray): Matrix to map, :math:`\mathbf{Y}`.
            ref_matrix: (np.ndarray): Reference matrix, :math:`\mathbf{X}`.

        Returns:
            (np.ndarray): matrix in the tangent manifold, :math:`\boldsymbol{\Gamma}`.
        """
        m, n = matrix.shape

        if m != n:
            gamma = matrix - ref_matrix

            # >>> unmodified
            # gamma = matrix
            # <<<
        else:
            # Amsallem 2011
            inv_ref_matrix = sclalg.inv(ref_matrix)
            gamma = sclalg.logm(matrix.dot(inv_ref_matrix))

            # Carmen pg 92
            # xy_inv = sclalg.inv(ref_matrix.T.dot(matrix))
            # xx_inv = sclalg.inv(ref_matrix.T.dot(ref_matrix))
            # a = np.eye(n) - ref_matrix.dot(xx_inv.dot(ref_matrix.T))
            # b = xy_inv.dot(sclalg.sqrtm(ref_matrix.T.dot(ref_matrix)))

            # u, s, z = sclalg.svd(a.dot(matrix.dot(b)), full_matrices=False)

            # gamma = u.dot(np.diag(np.arctan(s)).dot(z))

            # Amsallem 2008
            # a = np.eye(n) - ref_matrix.dot(ref_matrix.T).dot(matrix.dot(ref_matrix.T))
            # gamma = a.dot(matrix.dot(sclalg.inv(ref_matrix.T.dot(ref_matrix))))

        return gamma

    @staticmethod
    def from_tangent_manifold(matrix, ref_matrix):
        r"""
        Based on Table 4.1 from Amsallem and Farhat [1], returns a matrix from the tangent manifold using
        an exponential mapping.

        .. math:: \mathrm{Exp}_\mathbf{X}(\boldsymbol{\Gamma}) = \mathbf{Y}

        Args:
            matrix: equivalent to gamma
            ref_matrix: reference matrix

        If the matrices are square, this is calculated as:

        .. math:: \mathbf{Y} = \exp(\boldsymbol{\Gamma})\mathbf{X}

        Else:

        .. math:: \mathbf{Y} = \mathbf{X} + \boldsymbol{\Gamma}

        Args:
            matrix (np.ndarray): Matrix in the tangent manifold, :math:`\boldsymbol{\Gamma}`.
            ref_matrix: (np.ndarray): Reference matrix, :math:`\mathbf{X}`.

        Returns:
            (np.ndarray): matrix in the original manifold, :math:`\mathbf{Y}`.

        """
        m, n = matrix.shape

        if m != n:
            return matrix + ref_matrix
            # return matrix
        else:
            pass
            # amsallem 2011
            return sclalg.expm(matrix).dot(ref_matrix)

            # carmen pg 92
            # u, s, z = sclalg.svd(matrix, full_matrices=False)

            # return ref_matrix.dot(sclalg.sqrtm(ref_matrix.T.dot(ref_matrix))).dot(z.dot(np.diag(np.cos(s)))) \
            #        + u.dot(np.diag(np.sin(s)))

            # amsallem 2008
            # u, s, z = sclalg.svd(matrix, full_matrices=False)
            # return matrix.dot(z.dot(np.diag(np.cos(s)))) + u.dot(np.diag(np.sin(s)))


class InterpolationRealMatrices(TangentInterpolation):
    """
    Uses interpolation on the manifold of real matrices.

    References:
        Geuss, Panzer, Lohmann. On Parametric Model Order Reduction By Matrix Interpolation. European Control Conference
        2013.
    """
    @staticmethod
    def to_tangent_manifold(matrix, ref_matrix):
        return matrix - ref_matrix

    @staticmethod
    def from_tangent_manifold(matrix, ref_matrix):
        return matrix + ref_matrix


class BasisInterpolation:

    def __init__(self, v_list=list(), vt_list=list(), ss_list=list(), reference_case=0):
        self.v_list = v_list
        self.vt_list = vt_list
        self.reference_case = reference_case
        self.ss_list = ss_list

        self.gamma = [None] * len(self.v_list)

    def create_tangent_space(self, indices=None):

        n, r = self.v_list[0].shape
        vref = self.v_list[self.reference_case]

        if indices is None:
            indices = range(len(self.v_list))

        for vi in indices:
            v = self.v_list[vi]
            a = np.eye(n) - vref.dot(v.T)
            b = v
            c = sclalg.inv(vref.T.dot(v))

            p, s, q = sclalg.svd(algebra.multiply_matrices(a, b, c), full_matrices=False)

            self.gamma[vi] = p.dot(np.diag(np.arctan(s)).dot(q))

    def interpolate_gamma(self, weights):

        gamma = []
        for ith, gamma_i in enumerate(self.gamma):
            if gamma_i is not None:
                gamma.append(gamma_i * weights[ith])
        return sum(gamma)

    def return_from_tangent_space(self, gamma):
        p, s, q = sclalg.svd(gamma, full_matrices=False)
        v_ref = self.v_list[self.reference_case]

        v = v_ref.dot(q.T.dot(np.diag(np.cos(s)))) + p.dot(np.diag(np.sin(s)))

        return v

    def interpolate(self, weights, ss):

        gamma = self.interpolate_gamma(weights)

        v_interp = self.return_from_tangent_space(gamma)

        ss.project(v_interp.T, v_interp)

        return ss
