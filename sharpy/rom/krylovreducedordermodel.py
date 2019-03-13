import numpy as np
import scipy.linalg as sclalg
import sharpy.linear.src.libss as libss
import sharpy.utils.h5utils as h5
import time


class KrylovReducedOrderModel(object):
    """
    Model Order Reduction Methods for Single Input Single Output (SISO) and MIMO
    Linear Time-Invariant (LTI) Systems using
    moment matching (Krylov Methods).

    Examples:
        General calling sequences for different systems

        SISO single point interpolation:
            >>>algorithm = 'one_sided_arnoldi'
            >>>interpolation_point = np.array([0.0])
            >>>krylov_r = 4
            >>>
            >>>rom = KrylovReducedOrderModel()
            >>>rom.initialise(sharpy_data, FullOrderModelSS)
            >>>rom.run(algorithm, krylov_r, interpolation_point)

        2 by 2 MIMO with tangential, multipoint interpolation:
            >>>algorithm = 'dual_rational_arnoldi'
            >>>interpolation_point = np.array([0.0, 1.0j])
            >>>krylov_r = 4
            >>>right_vector = np.block([[1, 0], [0, 1]])
            >>>left_vector = right_vector
            >>>
            >>>rom = KrylovReducedOrderModel()
            >>>rom.initialise(sharpy_data, FullOrderModelSS)
            >>>rom.run(algorithm, krylov_r, interpolation_point, right_vector, left_vector)

        2 by 2 MIMO multipoint interpolation:
            >>>algorithm = 'mimo_rational_arnoldi'
            >>>interpolation_point = np.array([0.0])
            >>>krylov_r = 4
            >>>
            >>>rom = KrylovReducedOrderModel()
            >>>rom.initialise(sharpy_data, FullOrderModelSS)
            >>>rom.run(algorithm, krylov_r, interpolation_point)
    """

    def __init__(self):

        # self.settings_types = dict()
        # self.settings_default = dict()
        #
        # self.settings_types['algorithm'] = 'str'
        # self.settings_default['algorithm'] = None
        #
        # self.settings_types['frequencies'] = 'list(complex)'
        # self.settings_default['frequencies'] = None

        self.frequency = None
        self.algorithm = None
        self.ss = None
        self.r = 100
        self.V = None
        self.H = None
        self.W = None
        self.ssrom = None
        self.data = None
        self.sstype = None
        self.nfreq = None
        self.restart_arnoldi = False

    def initialise(self, data, ss):

        self.data = data  # Optional
        self.ss = ss

        if self.ss.dt is None:
            self.sstype = 'ct'
        else:
            self.sstype = 'dt'



    def run(self, algorithm, r, frequency=None, right_tangent=None, left_tangent=None):
        """
        Performs Model Order Reduction employing Krylov space projection methods.

        Supported methods include:

        =========================  ====================  ==========================================================
        Algorithm                  Interpolation Points  Systems
        =========================  ====================  ==========================================================
        ``one_sided_arnoldi``      1                     SISO Systems
        ``two_sided_arnoldi``      1                     SISO Systems
        ``dual_rational_arnoldi``  K                     SISO systems and Tangential interpolation for MIMO systems
        ``mimo_rational_arnoldi``  K                     MIMO systems. Uses vector-wise construction (stable)
        ``mimo_block_arnoldi``     K                     MIMO systems. Uses block Arnoldi methods (more efficient)
        =========================  ====================  ==========================================================

        Args:
            algorithm (str): Selected algorithm
            r (int): Desired Krylov space order. See the relevant algorithm for details.
            frequency (np.ndarray): Array containing the interpolation points
            right_tangent (np.ndarray): Right tangential direction vector assembled in matrix form.
            left_tangent (np.ndarray): Left tangential direction vector assembled in matrix form.

        Returns:

        """
        self.algorithm = algorithm
        self.frequency = frequency
        self.r = r
        try:
            self.nfreq = frequency.shape[0]
        except AttributeError:
            self.nfreq = 1

        print('Model Order Reduction in progress...')
        t0 = time.time()

        if algorithm == 'one_sided_arnoldi':
            Ar, Br, Cr = self.one_sided_arnoldi(frequency, r)

        elif algorithm == 'two_sided_arnoldi':
            Ar, Br, Cr = self.two_sided_arnoldi(frequency, r)

        elif algorithm == 'dual_rational_arnoldi':
            Ar, Br, Cr = self.dual_rational_arnoldi(frequency, r, right_tangent, left_tangent)

        elif algorithm == 'real_rational_arnoldi':
            Ar, Br, Cr = self.real_rational_arnoldi(frequency, r)

        elif algorithm == 'mimo_rational_arnoldi':
            Ar, Br, Cr = self.mimo_rational_arnoldi(frequency, r)

        elif algorithm == 'mimo_block_arnoldi':
            Ar, Br, Cr = self.mimo_block_arnoldi(frequency, r)

        else:
            raise NotImplementedError('Algorithm %s not recognised, check for spelling or it may not be implemented'
                                      %algorithm)

        self.ssrom = libss.ss(Ar, Br, Cr, self.ss.D, self.ss.dt)

        self.check_stability(restart_arnoldi=self.restart_arnoldi)

        t_rom = time.time() - t0
        print('\t\t...Completed Model Order Reduction in %.2f s' % t_rom)


    def one_sided_arnoldi(self, frequency, r):
        r"""
        One-sided Arnoldi method expansion about a single interpolation point, :math:`\sigma`.
        The projection matrix :math:`\mathbf{V}` is constructed using an order :math:`r` Krylov space. The space for
        a single finite interpolation point known as a Pade approximation is described by:

            .. math::
                    \text{range}(\textbf{V}) = \mathcal{K}_r((\sigma\mathbf{I}_n - \mathbf{A})^{-1},
                    (\sigma\mathbf{I}_n - \mathbf{A})^{-1}\mathbf{b})

        In the case of an interpolation about infinity, the problem is known as partial realisation and the Krylov
        space is

            .. math::
                    \text{range}(\textbf{V}) = \mathcal{K}_r(\mathbf{A}, \mathbf{b})

        The resulting orthogonal projection leads to the following reduced order system:

            .. math::
                \hat{\Sigma} : \left(\begin{array}{c|c} \hat{A} & \hat{B} \\
                \hline \hat{C} & {D}\end{array}\right)
                \text{with } \begin{cases}\hat{A}=V^TAV\in\mathbb{R}^{k\times k},\,\\
                \hat{B}=V^TB\in\mathbb{R}^{k\times m},\,\\
                \hat{C}=CV\in\mathbb{R}^{p\times k},\,\\
                \hat{D}=D\in\mathbb{R}^{p\times m}\end{cases}


        Args:
            frequency (complex): Interpolation point :math:`\sigma \in \mathbb{C}`
            r (int): Number of moments to match. Equivalent to Krylov space order and order of the ROM.

        Returns:
            tuple: The reduced order model matrices: :math:`\mathbf{A}_r`, :math:`\mathbf{B}_r` and :math:`\mathbf{C}_r`

        """
        A = self.ss.A
        B = self.ss.B
        C = self.ss.C

        nx = A.shape[0]

        if frequency != np.inf and frequency is not None:
            lu_A = sclalg.lu_factor(frequency * np.eye(nx) - A)
            V = construct_krylov(r, lu_A, B, 'Pade', 'b')
        else:
            V = construct_krylov(r, A, B, 'partial_realisation', 'b')

        # Reduced state space model
        Ar = V.T.dot(A.dot(V))
        Br = V.T.dot(B)
        Cr = C.dot(V)

        return Ar, Br, Cr

    def two_sided_arnoldi(self, frequency, r):
        r"""
        Two-sided projection with a single interpolation point following the Arnoldi procedure. Very similar to the
        one-sided method available, but it adds the projection :math:`\mathbf{W}` built using the Krylov space for the
        :math:`\mathbf{c}` vector:

            .. math::
                    \mathcal{K}_r((\sigma\mathbf{I}_n - \mathbf{A})^{-T},
                    (\sigma\mathbf{I}_n - \mathbf{A})^{-T}\mathbf{c}^T)\subseteq\mathcal{W}=\text{range}(\mathbf{W})

        The oblique projection :math:`\mathbf{VW}^T` matches twice as many moments as the single sided projection.

        The resulting system takes the form:

            .. math::
                \hat{\Sigma} : \left(\begin{array}{c|c} \hat{A} & \hat{B} \\
                \hline \hat{C} & {D}\end{array}\right)
                \text{with } \begin{cases}\hat{A}=W^TAV\in\mathbb{R}^{k\times k},\,\\
                \hat{B}=W^TB\in\mathbb{R}^{k\times m},\,\\
                \hat{C}=CV\in\mathbb{R}^{p\times k},\,\\
                \hat{D}=D\in\mathbb{R}^{p\times m}\end{cases}

        Args:
            frequency (complex): Interpolation point :math:`\sigma \in \mathbb{C}`
            r (int): Number of moments to match on each side. The resulting ROM will be of order :math:`2r`.

        Returns:
            tuple: The reduced order model matrices: :math:`\mathbf{A}_r`, :math:`\mathbf{B}_r` and :math:`\mathbf{C}_r`.

        """
        A = self.ss.A
        B = self.ss.B
        C = self.ss.C

        nx = A.shape[0]

        if frequency != np.inf and frequency is not None:
            lu_A = sclalg.lu_factor(frequency * np.eye(nx) - A)
            V = construct_krylov(r, lu_A, B, 'Pade', 'b')
            W = construct_krylov(r, lu_A, C.T, 'Pade', 'c')
        else:
            V = construct_krylov(r, A, B, 'partial_realisation', 'b')
            W = construct_krylov(r, A, C.T, 'partial_realisation', 'c')

        # Ensure oblique projection to ensure W^T V = I
        # lu_WW = sclalg.lu_factor(W.T.dot(V))
        # W1 = sclalg.lu_solve(lu_WW, W.T, trans=1).T # Verify
        W = W.dot(sclalg.inv(W.T.dot(V)).T)

        # Reduced state space model
        Ar = W.T.dot(A.dot(V))
        Br = W.T.dot(B)
        Cr = C.dot(V)

        return Ar, Br, Cr

    def real_rational_arnoldi(self, frequency, r):
        """
        When employing complex frequencies, the projection matrix can be normalised to be real
        Following Algorithm 1b in Lee(2006)
        Args:
            frequency:
            r:

        Returns:

        """

        raise NotImplementedError('Real valued rational Arnoldi Method in progress')

        ### Not working, having trouble with the last column of H. need to investigate the background behind the creation of H and see hwat can be done

        A = self.ss.A
        B = self.ss.B
        C = self.ss.C

        nx = A.shape[0]
        nfreq = frequency.shape[0]

        # Columns of matrix v
        v_ncols = 2 * np.sum(r)

        # Output projection matrices
        V = np.zeros((nx, v_ncols),
                     dtype=float)
        H = np.zeros((v_ncols, v_ncols),
                     dtype=float)
        res = np.zeros((nx,v_ncols+2),
                       dtype=float)

        # lu_A = sclalg.lu_factor(frequency[0] * np.eye(nx) - A)
        v_res = sclalg.lu_solve(lu_A, B)

        H[0, 0] = np.linalg.norm(v_res)
        V[:, 0] = v_res.real / H[0, 0]

        k = 0
        for i in range(nfreq):
            for j in range(r[i]):
                # k = 2*(i*r[i] + j)
                print("i = %g\t j = %g\t k = %g" % (i, j, k))

                # res[:, k] = np.imag(v_res)
                # if k > 0:
                #     res[:, k-1] = np.real(v_res)
                #
                # # Working on the last finished column i.e. k-1 only when k>0
                # if k > 0:
                #     for t in range(k):
                #         H[t, k-1] = V[:, t].T.dot(res[:, k-1])
                #         res[:, k-1] -= res[:, k-1] - H[t, k-1] * V[:, t]
                #
                #     H[k, k-1] = np.linalg.norm(res[:, k-1])
                #     V[:, k] = res[:, k-1] / H[k, k-1]
                #
                # # Normalise working column k
                # for t in range(k+1):
                #     H[t, k] = V[:, t].T.dot(res[:, k])
                #     res[:, k] -= H[t, k] * V[:, t]
                #
                # # Subdiagonal term
                # H[k+1, k] = np.linalg.norm(res[:, k])
                # V[:, k + 1] = res[:, k] / np.linalg.norm(res[:, k])
                #
                # if j == r[i] - 1 and i < nfreq - 1:
                #     lu_A = sclalg.lu_factor(frequency[i+1] * np.eye(nx) - A)
                #     v_res = sclalg.lu_solve(lu_A, B)
                # else:
                #     v_res = - sclalg.lu_solve(lu_A, V[:, k+1])

                if k == 0:
                    V[:, 0] = v_res.real / np.linalg.norm(v_res.real)
                else:
                    res[:, k] = np.imag(v_res)
                    res[:, k-1] = np.real(v_res)

                    for t in range(k):
                        H[t, k-1] = np.linalg.norm(res[:, k-1])
                        res[:, k-1] -= H[t, k-1]*V[:, t]

                    H[k, k-1] = np.linalg.norm(res[:, k-1])
                    V[:, k] = res[:, k-1] / H[k, k-1]

                if k == 0:
                    H[0, 0] = V[:, 0].T.dot(v_res.imag)
                    res[:, 0] -= H[0, 0] * V[:, 0]

                else:
                    for t in range(k+1):
                        H[t, k] = V[:, t].T.dot(res[:, k])
                        res[:, k] -= H[t, k] * V[:, t]
                H[k+1, k] = np.linalg.norm(res[:, k])
                V[:, k+1] = res[:, k] / H[k+1, k]

                if j == r[i] - 1 and i < nfreq - 1:
                    lu_A = sclalg.lu_factor(frequency[i+1]*np.eye(nx) - A)
                    v_res = sclalg.lu_solve(lu_A, B)
                else:
                    v_res = - sclalg.lu_solve(lu_A, V[:, k+1])

                k += 2

        # Add last column of H
        print(k)
        res[:, k-1] = - sclalg.lu_solve(lu_A, V[:, k-1])
        for t in range(k-1):
            H[t, k-1] = V[:, t].T.dot(res[:, k-1])
            res[:, k-1] -= H[t, k-1]*V[:, t]

        self.V = V
        self.H = H

        Ar = V.T.dot(A.dot(V))
        Br = V.T.dot(B)
        Cr = C.dot(V)

        return Ar, Br, Cr

    def dual_rational_arnoldi(self, frequency, r, right_tangent=None, left_tangent=None):
        r"""
        Dual Rational Arnoli Interpolation for SISO sytems [1] and MIMO systems through tangential interpolation [2].

        Effectively the same as the two_sided_arnoldi and the resulting V matrices for each interpolation point are
        concatenated

        .. math::
            \bigcup\limits_{k = 1}^K\mathcal{K}_{b_k}((\sigma_i\mathbf{I}_n - \mathbf{A})^{-1}, (\sigma_i\mathbf{I}_n
            - \mathbf{A})^{-1}\mathbf{b})\subseteq\mathcal{V}&=\text{range}(\mathbf{V}) \\
            \bigcup\limits_{k = 1}^K\mathcal{K}_{c_k}((\sigma_i\mathbf{I}_n - \mathbf{A})^{-T}, (\sigma_i\mathbf{I}_n
            - \mathbf{A})^{-T}\mathbf{c}^T)\subseteq\mathcal{Z}&=\text{range}(\mathbf{Z})

        For MIMO systems, tangential interpolation is used through the right and left tangential direction vectors
        :math:`\mathbf{r}_i` and :math:`\mathbf{l}_i`.

        .. math::
            \bigcup\limits_{k = 1}^K\mathcal{K}_{b_k}((\sigma_i\mathbf{I}_n - \mathbf{A})^{-1}, (\sigma_i\mathbf{I}_n
            - \mathbf{A})^{-1}\mathbf{Br}_i)\subseteq\mathcal{V}&=\text{range}(\mathbf{V}) \\
            \bigcup\limits_{k = 1}^K\mathcal{K}_{c_k}((\sigma_i\mathbf{I}_n - \mathbf{A})^{-T}, (\sigma_i\mathbf{I}_n
            - \mathbf{A})^{-T}\mathbf{C}^T\mathbf{l}_i)\subseteq\mathcal{Z}&=\text{range}(\mathbf{Z})

        Args:
            frequency (np.ndarray): Array containing the interpolation points
                :math:`\sigma = \{\sigma_1, \dots, \sigma_K\}\in\mathbb{C}`
            r (int): Krylov space order :math:`b_k` and :math:`c_k`. At the moment, different orders for the
                controllability and observability constructions are not supported.
            right_tangent (np.ndarray): Matrix containing the right tangential direction interpolation vector for
                each interpolation point in column form, i.e. :math:`\mathbf{r}\in\mathbb{R}^{m \times K}`.
            left_tangent (np.ndarray): Matrix containing the left tangential direction interpolation vector for
                each interpolation point in column form, i.e. :math:`\mathbf{l}\in\mathbb{R}^{p \times K}`.

        Returns:
            tuple: The reduced order model matrices: :math:`\mathbf{A}_r`, :math:`\mathbf{B}_r` and :math:`\mathbf{C}_r`.

        References:
            [1] Grimme
            [2] Gallivan
        """
        A = self.ss.A
        B = self.ss.B
        C = self.ss.C

        nx = self.ss.states
        nu = self.ss.inputs
        ny = self.ss.outputs

        B.shape = (nx, nu)

        try:
            nfreq = frequency.shape[0]
        except AttributeError:
            nfreq = 1

        if nu != 1 or ny != 1:
            assert right_tangent is not None and left_tangent is not None, 'Missing interpolation vectors for MIMO case'


        # Tangential interpolation for MIMO systems
        if right_tangent is None:
            right_tangent = np.ones((nu, nfreq))
        else:
            assert right_tangent.shape == (nu, nfreq), 'Right Tangential Direction vector not the correct shape'

        if left_tangent is None:
            left_tangent = np.ones((ny, nfreq))
        else:
            assert left_tangent.shape == (ny, nfreq), 'Left Tangential Direction vector not the correct shape'

        V = np.zeros((nx, r*nfreq), dtype=complex)
        W = np.zeros((nx, r*nfreq), dtype=complex)

        we = 0
        for i in range(nfreq):
            sigma = frequency[i]
            if sigma == np.inf:
                approx_type = 'partial_realisation'
                lu_A = A
            else:
                approx_type = 'Pade'
                lu_A = sclalg.lu_factor(sigma * np.eye(nx) - A)

            V[:, we:we+r] = construct_krylov(r, lu_A, B.dot(right_tangent[:, i:i+1]), approx_type, 'b')
            W[:, we:we+r] = construct_krylov(r, lu_A, C.T.dot(left_tangent[:, i:i+1]), approx_type, 'c')

            we += r

        W = W.dot(sclalg.inv(W.T.dot(V)).T)
        self.W = W
        self.V = V

        # Reduced state space model
        Ar = W.T.dot(A.dot(V))
        Br = W.T.dot(B)
        Cr = C.dot(V)

        return Ar, Br, Cr

    def mimo_rational_arnoldi(self, frequency, r):
        """
        Following the method for vector-wise construction in Gugercin

        Warnings:
            Under Development

        Returns:

        """
 
        m = self.ss.inputs  # Full system number of inputs
        n = self.ss.states  # Full system number of states
        p = self.ss.outputs  # Full system number of outputs

        # If the number of inputs is not the same as the number of outputs, they have to be
        # a factor of one another, such that the corresponding Krylov subspace can be constructed
        # to a greater order and the resulting projection matrices are the same size
        if m != p:
            assert np.mod(np.max([p, m]), np.min([p, m])) == 0, 'Number of outputs is not a factor of number' \
                                                                ' of inputs, currently not implemented'
            if m < p:
                r_o = r
                r_c = r_o * p // m
            else:
                r_c = r
                r_o = r_c * m // p
        else:
            r_c = r
            r_o = r

        B = self.ss.B
        C = self.ss.C

        for i in range(self.nfreq):

            if frequency[i] == np.inf:
                lu_a = self.ss.A
                approx_type = 'partial_realisation'
            else:
                approx_type = 'Pade'
                lu_a = sclalg.lu_factor(frequency[i] * np.eye(n) - self.ss.A)
            if i == 0:
                V = construct_mimo_krylov(r_c, lu_a, B, approx_type=approx_type, side='controllability')
                W = construct_mimo_krylov(r_o, lu_a, C.T, approx_type=approx_type, side='observability')
            else:
                Vi = construct_mimo_krylov(r_c, lu_a, B, approx_type=approx_type, side='controllability')
                Wi = construct_mimo_krylov(r_o, lu_a, C.T, approx_type=approx_type, side='observability')
                V = np.block([V, Vi])
                W = np.block([W, Wi])
                V = mgs_ortho(V)
                W = mgs_ortho(W)

        W = W.dot(sclalg.inv(W.T.dot(V)).T)
        self.W = W
        self.V = V

        # Reduced state space model
        Ar = W.T.dot(self.ss.A.dot(V))
        Br = W.T.dot(self.ss.B)
        Cr = self.ss.C.dot(V)

        return Ar, Br, Cr

    def mimo_block_arnoldi(self, frequency, r):

        n = self.ss.states

        A = self.ss.A
        B = self.ss.B
        C = self.ss.C

        for i in range(self.nfreq):

            if self.frequency[i] == np.inf:
                F = A
                G = B
            else:
                lu_a = sclalg.lu_factor(frequency[i] * np.eye(n) - A)
                F = sclalg.lu_solve(lu_a, np.eye(n))
                G = sclalg.lu_solve(lu_a, B)

            if i == 0:
                V = block_arnoldi_krylov(r, F, G)
            else:
                Vi = block_arnoldi_krylov(r, F, G)
                V = np.block([V, Vi])

        self.V = V

        Ar = V.T.dot(A.dot(V))
        Br = V.T.dot(B)
        Cr = C.dot(V)

        return Ar, Br, Cr

    def check_stability(self, restart_arnoldi=False):
        r"""
        Checks the stability of the ROM by computing its eigenvalues.

        If the resulting system is unstable, the Arnoldi procedure can be restarted to eliminate the eigenvalues
        outside the stability boundary.

        However, if this is the case, the ROM no longer matches the moments of the original system at the specific
        frequencies since now the approximation is done with respect to a system of the form:

            .. math::
                \Sigma = \left(\begin{array}{c|c} \mathbf{A} & \mathbf{\bar{B}}
                \\ \hline \mathbf{C} & \ \end{array}\right)

        where :math:`\mathbf{\bar{B}} = (\mu \mathbf{I}_n - \mathbf{A})\mathbf{B}`

        Args:
            restart_arnoldi (bool): Restart the relevant Arnoldi algorithm with the unstable eigenvalues removed.


        """
        assert self.ssrom is not None, 'ROM not calculated yet'

        eigs = sclalg.eigvals(self.ssrom.A)

        eigs_abs = np.abs(eigs)

        unstable = False
        if self.sstype == 'dt':
            if any(eigs_abs > 1.):
                unstable = True
                unstable_eigenvalues = eigs[eigs_abs > 1.]
                print('Unstable ROM - %d Eigenvalues with |r| > 1' % len(unstable_eigenvalues))
                for mu in unstable_eigenvalues:
                    print('\tmu = %f + %fj' % (mu.real, mu.imag))
            else:
                print('ROM is stable')

        else:
            if any(eigs.real > 0):
                unstable = True
                print('Unstable ROM')
                unstable_eigenvalues = eigs[eigs.real > 0]
            else:
                print('ROM is stable')

        # Restarted Arnoldi
        # Modify the B matrix in the full state system -> maybe better to have a copy
        if unstable and restart_arnoldi:
            print('Restarting the Arnoldi method - Reducing ROM order from r = %d to r = %d' % (self.r, self.r-1))
            self.ss_original = self.ss

            remove_unstable = np.eye(self.ss.states)
            for mu in unstable_eigenvalues:
                remove_unstable = np.matmul(remove_unstable, mu * np.eye(self.ss.states) - self.ss.A)

            self.ss.B = remove_unstable.dot(self.ss.B)
            # self.ss.C = self.ss.C.dot(remove_unstable.T)

            if self.r > 1:
                self.r -= 1
                self.run(self.algorithm, self.r, self.frequency)
            else:
                print('Unable to reduce ROM any further - ROM still unstable...')

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

    Q = np.zeros((m, n), dtype=complex)

    for i in range(n):
        w = X[:, i]
        for j in range(i):
            h = Q[:, j].T.dot(w)
            w = w - h * Q[:, j]
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
        v = sclalg.lu_solve(lu_A, B, trans=transpose_mode)
        v = v / np.linalg.norm(v)
        w = sclalg.lu_solve(lu_A, v)

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
            w = sclalg.lu_solve(lu_A, v, trans=transpose_mode)

        h = V[:, :j+2].T.dot(w)
        f[:, j+1] = w - V[:, :j+2].dot(h)

        # Finite precision
        s = V[:, :j+2].T.dot(f[:, j+1])
        f[:, j+1] = f[:, j+1] - V[:, :j+2].dot(s)
        h += s

        h.shape = (j+2, 1)  # Enforce shape for concatenation
        H[:j+2, :j+2] = np.block([H_hat, h])

    return V

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
        # lu_a = sclalg.lu_factor(lu_A_input)
        G = sclalg.lu_solve(lu_A_input, B, trans=transpose_mode)

    for k in range(m):
        # if approx_type == 'partial_realisation':
        #     G = B
        #     F = lu_A_input
        # else:
        #     lu_a = sclalg.lu_factor(lu_A_input)
        #     G = sclalg.lu_solve(lu_A_input, B, trans=transpose_mode)
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
                w[:, t] = sclalg.lu_solve(lu_A_input, w[:, t-mu], trans=transpose_mode)

            # Orthogonalise w[:,t] against V_i -
            w[:, :t+1] = mgs_ortho(w[:, :t+1])[:, :t+1]

            if np.linalg.norm(w[:, t]) < deflation_tolerance:
                # Deflate w_k
                print('Vector deflated')
                w = [w[:, 0:t], w[:, t+1:]]  # TODO Figure if there is a better way to slice this
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

if __name__ == "__main__":
    pass
