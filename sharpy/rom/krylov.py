"""Krylov-subspaces model order reduction techniques
"""
import numpy as np
import scipy.linalg as sclalg
import sharpy.linear.src.libss as libss
import time
import sharpy.utils.settings as settings
import sharpy.utils.cout_utils as cout
import sharpy.utils.rom_interface as rom_interface
import sharpy.utils.h5utils as h5
import sharpy.rom.utils.krylovutils as krylovutils
import warnings as warn
import h5py
from sharpy.linear.utils.ss_interface import LinearVector, StateVariable, InputVariable, OutputVariable

@rom_interface.rom
class Krylov(rom_interface.BaseRom):
    """
    Model Order Reduction Methods for Single Input Single Output (SISO) and MIMO
    Linear Time-Invariant (LTI) Systems using
    moment matching (Krylov Methods).

    Examples:
        General calling sequences for different systems

        SISO single point interpolation:
            >>> algorithm = 'one_sided_arnoldi'
            >>> interpolation_point = np.array([0.0])
            >>> krylov_r = 4
            >>>
            >>> rom = Krylov()
            >>> rom.initialise(sharpy_data, FullOrderModelSS)
            >>> rom.run(algorithm, krylov_r, interpolation_point)

        2 by 2 MIMO with tangential, multipoint interpolation:
            >>> algorithm = 'dual_rational_arnoldi'
            >>> interpolation_point = np.array([0.0, 1.0j])
            >>> krylov_r = 4
            >>> right_vector = np.block([[1, 0], [0, 1]])
            >>> left_vector = right_vector
            >>>
            >>> rom = Krylov()
            >>> rom.initialise(sharpy_data, FullOrderModelSS)
            >>> rom.run(algorithm, krylov_r, interpolation_point, right_vector, left_vector)

        2 by 2 MIMO multipoint interpolation:
            >>> algorithm = 'mimo_rational_arnoldi'
            >>> interpolation_point = np.array([0.0])
            >>> krylov_r = 4
            >>>
            >>> rom = Krylov()
            >>> rom.initialise(sharpy_data, FullOrderModelSS)
            >>> rom.run(algorithm, krylov_r, interpolation_point)
    """
    rom_id = 'Krylov'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    settings_types['print_info'] = 'bool'
    settings_default['print_info'] = True
    settings_description['print_info'] = 'Write ROM information to screen and log'

    settings_types['frequency'] = 'list(complex)'
    settings_default['frequency'] = [0]
    settings_description['frequency'] = 'Interpolation points in the continuous time complex plane [rad/s]'

    settings_types['algorithm'] = 'str'
    settings_default['algorithm'] = ''
    settings_description['algorithm'] = 'Krylov reduction method algorithm'

    settings_types['r'] = 'int'
    settings_default['r'] = 1
    settings_description['r'] = 'Moments to match at the interpolation points'

    settings_types['single_side'] = 'str'
    settings_default['single_side'] = ''
    settings_description['single_side'] = 'Construct the rom using a single side. Leave blank (or empty string) for both.'
    settings_options['single_side'] = ['controllability', 'observability']

    settings_types['tangent_input_file'] = 'str'
    settings_default['tangent_input_file'] = ''
    settings_description['tangent_input_file'] = 'Filepath to .h5 file containing tangent interpolation vectors'

    settings_types['restart_arnoldi'] = 'bool'
    settings_default['restart_arnoldi'] = False
    settings_description['restart_arnoldi'] = 'Restart Arnoldi iteration with r-=1 if ROM is unstable'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description, settings_options)

    supported_methods = ('one_sided_arnoldi',
                         'two_sided_arnoldi',
                         'dual_rational_arnoldi',
                         'mimo_rational_arnoldi',
                         'mimo_block_arnoldi')

    def __init__(self):
        self.settings = dict()

        self.frequency = None
        self.algorithm = None
        self.ss = None
        self.r = 1
        self.V = None
        self.H = None
        self.W = None
        self.ssrom = None
        self.sstype = None
        self.nfreq = None
        self.restart_arnoldi = None
        self.stable = None
        self.cpu_summary = dict()
        self.eigenvalue_table = None

    def initialise(self, in_settings=None):

        if in_settings is not None:
            self.settings = in_settings

        settings.to_custom_types(self.settings, self.settings_types, self.settings_default, self.settings_options)

        try:
            if self.settings['print_info']:
                cout.cout_wrap('Initialising Krylov Model Order Reduction')
        except ValueError:
            pass

        self.algorithm = self.settings['algorithm']
        if self.algorithm not in self.supported_methods:
            raise NotImplementedError('Algorithm %s not recognised, check for spelling or it'
                                      'could be that is not yet implemented'
                                      % self.algorithm)

        self.frequency = np.array(self.settings['frequency'])
        self.r = self.settings['r']
        self.restart_arnoldi = self.settings['restart_arnoldi']
        try:
            self.nfreq = self.frequency.shape[0]
        except AttributeError:
            self.nfreq = 1

    def save(self, filename):
        """
        Saves to an ``.h5`` file of name ``filename`` the left and right projectors and the reduced order model

        Args:
            filename (str): path and filename to which to save the data

        """
        rom_projectors = {'right_projector': self.V,
                          'left_projector': self.W,
                          }

        if '.h5' not in filename[-3:]:
            filename += '.h5'

        with h5py.File(filename, 'a') as outfile:
            h5.add_as_grp(rom_projectors, outfile, grpname='projectors',
                          compress_float=True)
            h5.add_as_grp(self.ssrom, outfile,
                          grpname='ssrom',
                          ClassesToSave=(libss.StateSpace, ),
                          compress_float=True)

    def save_reduced_order_bases(self, file_name):
        """
        Save reduced order bases to an h5 file

        Args:
            file_name (str): path to h5 file

        """
        if not isinstance(self.V, libss.Gain):
            V = libss.Gain(self.V)
            W = libss.Gain(self.W)

        else:
            V = self.V
            W = self.W

        if self.settings['single_side'] == 'observability' or self.settings['single_side'] == 'controllability':
            # if single sided, the projector is V and W = V therefore no need to duplicate
            libss.Gain.save_multiple_gains(file_name, ('V', V))
            cout.cout_wrap(f'Saved Krylov reduced order bases, V to file: {file_name}', 1)
        else:
            libss.Gain.save_multiple_gains(file_name, ('V', V), ('W', W))
            cout.cout_wrap(f'Saved Krylov reduced order bases, V and W to file: {file_name}', 1)

    def run(self, ss):
        """
        Performs Model Order Reduction employing Krylov space projection methods.

        Supported methods include:

        =========================  ====================  ==========================================================
        Algorithm                  Interpolation Points  Systems
        =========================  ====================  ==========================================================
        ``one_sided_arnoldi``      1                     SISO Systems
        ``two_sided_arnoldi``      1                     SISO Systems
        ``dual_rational_arnoldi``  K                     SISO systems and Tangential interpolation for MIMO systems
        ``mimo_rational_arnoldi``  K                     MIMO systems. Uses vector-wise construction (more robust)
        ``mimo_block_arnoldi``     K                     MIMO systems. Uses block Arnoldi methods (more efficient)
        =========================  ====================  ==========================================================

        Args:
            ss (sharpy.linear.src.libss.StateSpace): State space to reduce

        Returns:
            (libss.StateSpace): Reduced state space system
        """
        self.ss = ss

        if self.settings['print_info']:
            cout.cout_wrap('Model Order Reduction in progress...')
            self.print_header()

        if self.ss.dt is None:
            self.sstype = 'ct'
        else:
            self.sstype = 'dt'
            self.frequency = np.exp(self.frequency * ss.dt)

        t0 = time.time()

        Ar, Br, Cr = self.__getattribute__(self.algorithm)(self.frequency, self.r)

        self.ssrom = libss.StateSpace(Ar, Br, Cr, self.ss.D, self.ss.dt)
        try:
            self.ssrom.input_variables = self.ss.input_variables.copy()
            self.ssrom.output_variables = self.ss.output_variables.copy()
            self.ssrom.state_variables = LinearVector([StateVariable('krylov', size=self.ssrom.states, index=0)])
        except AttributeError:
            pass

        self.stable = self.check_stability(restart_arnoldi=self.restart_arnoldi)

        if not self.stable:
            pass
            warn.warn('Reduced Order Model Unstable')
            # Under development
            # TL, TR = self.restart()
            # Wtr, Vr = self.restart()
            # TL, TR = self.stable_realisation()
            # self.ssrom = libss.ss(TL.T.dot(Ar.dot(TR)), TL.T.dot(Br), Cr.dot(TR), self.ss.D, self.ss.dt)
            # self.ssrom = libss.ss(Wtr.dot(self.ssrom.A.dot(Vr)), Wtr.dot(self.ssrom.B), self.ssrom.C.dot(Vr), self.ss.D, self.ss.dt)
            # self.stable = self.check_stability(restart_arnoldi=self.restart_arnoldi)

        t_rom = time.time() - t0
        self.cpu_summary['run'] = t_rom
        if self.settings['print_info']:
            cout.cout_wrap('System reduced from order %d to ' % self.ss.states)
            cout.cout_wrap('\tn = %d states' % self.ssrom.states, 1)
            cout.cout_wrap('...Completed Model Order Reduction in %.2f s' % t_rom)

        return self.ssrom

    def print_header(self):
        cout.cout_wrap('Moment Matching Krylov Model Reduction')
        cout.cout_wrap('\tConstruction Algorithm:')
        cout.cout_wrap('\t\t%s' % self.algorithm, 1)
        cout.cout_wrap('\tInterpolation points:')
        if self.frequency.dtype == complex:
            cout.cout_wrap(self.nfreq * '\t\tsigma = %4f + %4fj [rad/s]\n' %tuple(self.frequency.view(float)), 1)
        else:
            cout.cout_wrap(self.nfreq * '\t\tsigma = %4f [rad/s]\n' % self.frequency, 1)
        cout.cout_wrap('\tKrylov order:')
        cout.cout_wrap('\t\tr = %d' % self.r, 1)

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
            lu_A = krylovutils.lu_factor(frequency, A)
            V = krylovutils.construct_krylov(r, lu_A, B, 'Pade', 'b')
        else:
            V = krylovutils.construct_krylov(r, A, B, 'partial_realisation', 'b')

        # Reduced state space model
        Ar = V.T.dot(A.dot(V))
        Br = V.T.dot(B)
        Cr = C.dot(V)

        self.V = V
        self.W = V

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
            lu_A = krylovutils.lu_factor(frequency, A)
            V = krylovutils.construct_krylov(r, lu_A, B, 'Pade', 'b')
            W = krylovutils.construct_krylov(r, lu_A, C.T, 'Pade', 'c')
        else:
            V = krylovutils.construct_krylov(r, A, B, 'partial_realisation', 'b')
            W = krylovutils.construct_krylov(r, A, C.T, 'partial_realisation', 'c')

        T = W.T.dot(V)
        Tinv = sclalg.inv(T)
        reduction_checks(T, Tinv)
        self.W = W
        self.V = V

        # Reduced state space model
        Ar = W.T.dot(self.ss.A.dot(V.dot(Tinv)))
        Br = W.T.dot(self.ss.B)
        Cr = self.ss.C.dot(V.dot(Tinv))

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

        raise NotImplementedError('Real valued rational Arnoldi Method Work in progress - use mimo_rational_arnoldi')

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

        # lu_A = krylovutils.lu_factor(frequency[0] * np.eye(nx) - A)
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
                #     lu_A = krylovutils.lu_factor(frequency[i+1] * np.eye(nx) - A)
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
                    lu_A = krylovutils.lu_factor(frequency[i+1], A)
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

    def dual_rational_arnoldi(self, frequency, r):
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

        if nu != 1:
            left_tangent, right_tangent, rc, ro, fc, fo = self.load_tangent_vectors()
            assert right_tangent is not None and left_tangent is not None, 'Missing interpolation vectors for MIMO case'
        else:
            fc = np.array(frequency)
            fo = np.array(frequency)
            left_tangent = np.zeros((1, len(fo)))
            right_tangent = np.zeros((1, len(fc)))
            rc = np.array([r]*len(fc))
            ro = np.array([r]*len(fc))
            right_tangent[0, :] = 1
            left_tangent[0, :] = 1

        try:
            nfreq = frequency.shape[0]
        except AttributeError:
            nfreq = 1


        t0 = time.time()
        # # Tangential interpolation for MIMO systems
        # if right_tangent is None:
        #     right_tangent = np.eye((nu, nfreq))
        # # else:
        # #     assert right_tangent.shape == (nu, nfreq), 'Right Tangential Direction vector not the correct shape'
        #
        # if left_tangent is None:
        #     left_tangent = np.eye((ny, nfreq))
        # # else:
        # #     assert left_tangent.shape == (ny, nfreq), 'Left Tangential Direction vector not the correct shape'

        rom_dim = max(np.sum(rc), np.sum(ro))
        V = np.zeros((nx, rom_dim), dtype=complex)
        W = np.zeros((nx, rom_dim), dtype=complex)

        we = 0
        dict_of_luas = dict()
        for i in range(len(fc)):
            sigma = fc[i]
            if sigma == np.inf:
                approx_type = 'partial_realisation'
                lu_A = A
            else:
                approx_type = 'Pade'
                try:
                    lu_A = dict_of_luas[sigma]
                except KeyError:
                    lu_A = krylovutils.lu_factor(sigma, A)
                    dict_of_luas[sigma] = lu_A
            V[:, we:we+rc[i]] = krylovutils.construct_krylov(rc[i], lu_A, B.dot(right_tangent[:, i:i+1]), approx_type, 'b')

            we += rc[i]

        we = 0
        for i in range(len(fo)):
            sigma = fo[i]
            if sigma == np.inf:
                approx_type = 'partial_realisation'
                lu_A = A
            else:
                approx_type = 'Pade'
                try:
                    lu_A = dict_of_luas[sigma]
                except KeyError:
                    lu_A = krylovutils.lu_factor(sigma, A)
                    dict_of_luas[sigma] = lu_A
            W[:, we:we+ro[i]] = krylovutils.construct_krylov(ro[i], lu_A, C.T.dot(left_tangent[:, i:i+1]), approx_type, 'c')

            we += ro[i]

        T = W.T.dot(V)
        Tinv = sclalg.inv(T)
        reduction_checks(T, Tinv)
        self.W = W
        self.V = V

        # Reduced state space model
        Ar = W.T.dot(self.ss.A.dot(V.dot(Tinv)))
        Br = W.T.dot(self.ss.B)
        Cr = self.ss.C.dot(V.dot(Tinv))

        del dict_of_luas

        self.cpu_summary['algorithm'] = time.time() - t0

        return Ar, Br, Cr

    def mimo_rational_arnoldi(self, frequency, r):
        r"""
        Construct full rank orthonormal projection basis :math:`\mathbf{V}` and :math:`\mathbf{W}`.

        The main issue that one normally encounters with MIMO systems is that the minimality assumption of the system
        does not guarantee the resulting Krylov space to be full rank, unlike in the SISO case. Therefore,
        the construction is performed vector by vector, where linearly dependent vectors are eliminated or deflated
        from the Krylov subspace.

        If the number of inputs differs the number of outputs, both Krylov spaces will be built such that both
        are the same size, therefore one Krylov space may be of higher order than the other one.

        Following the method for vector-wise construction in Gugercin [1].

        Args:
            frequency (np.ndarray): Array containing interpolation frequencies
            r (int): Krylov space order

        Returns:
            tuple: Tuple of reduced system matrices ``A``, ``B`` and ``C``.

        References:
            [1] Gugercin, S. Projection Methods for Model Reduction of Large-Scale Dynamical
             Systems PhD Thesis. Rice University 2003.
        """

        m = self.ss.inputs  # Full system number of inputs
        n = self.ss.states  # Full system number of states
        p = self.ss.outputs  # Full system number of outputs

        # If the number of inputs is not the same as the number of outputs, a larger than necessary projection matrix
        # is built for the one with fewer inputs/outputs. Thence, the larger matrix is truncated to have the same number
        # of columns as the smaller matrix
        if m != p:
            if m < p:
                r_o = r
                r_c = r_o * int(np.ceil(p / m))
            else:
                r_c = r
                r_o = r_c * int(np.ceil(m / p))
        else:
            r_c = r
            r_o = r

        if self.settings['single_side']:
            # if only a single side is built, use the original setting without modification
            r_c = r
            r_o = r

        V = None
        W = None

        for i in range(self.nfreq):

            if self.settings['single_side'] == 'controllability' or self.settings['single_side'] == '':
                cout.cout_wrap('\tConstructing controllability space', 1)
                if i == 0:
                    V = krylovutils.build_krylov_space(frequency[i], r_c, side='b', a=self.ss.A, b=self.ss.B)
                else:
                    Vi = krylovutils.build_krylov_space(frequency[i], r_c, side='b', a=self.ss.A, b=self.ss.B)
                    V = np.hstack((V, Vi))
                    V = krylovutils.mgs_ortho(V)

            if self.settings['single_side'] == 'observability' or self.settings['single_side'] == '':
                cout.cout_wrap('\tConstructing observability space', 1)
                if i == 0:
                    W = krylovutils.build_krylov_space(frequency[i], r_o, side='c', a=self.ss.A, b=self.ss.C.T)
                else:
                    Wi = krylovutils.build_krylov_space(frequency[i], r_o, side='c', a=self.ss.A, b=self.ss.C.T)
                    W = np.hstack((W, Wi))
                    W = krylovutils.mgs_ortho(W)

        if self.settings['single_side'] == 'controllability' or self.settings['single_side'] == 'observability':
            if self.settings['single_side'] == 'observability':
                V = W
            if self.settings['single_side'] == 'controllability':
                W = V  # needed to properly save gains
            Ar = V.T.dot(self.ss.A.dot(V))
            Br = V.T.dot(self.ss.B)
            Cr = self.ss.C.dot(V)

        else:
            # Match number of columns in each matrix
            min_cols = min(V.shape[1], W.shape[1])
            V = V[:, :min_cols]
            W = W[:, :min_cols]

            V, W = check_rank(V, W) # checks the product for rank deficiencies
            # V = krylovutils.mgs_ortho(V) # need to verify whether this is necessary
            # W = krylovutils.mgs_ortho(W)

            T = W.T.dot(V)
            Tinv = sclalg.inv(T)
            krylovutils.check_eye(T, Tinv)

            # Reduced state space model
            Ar = W.T.dot(self.ss.A.dot(V.dot(Tinv)))
            Br = W.T.dot(self.ss.B)
            Cr = self.ss.C.dot(V.dot(Tinv))

        self.W = libss.Gain(W,
                            input_vars=LinearVector([InputVariable('krylov', size=W.shape[1], index=0)]),
                            output_vars=LinearVector.transform(self.ss.state_variables, OutputVariable))
        self.V = libss.Gain(V,
                            input_vars=LinearVector([InputVariable('krylov', size=V.shape[1], index=0)]),
                            output_vars=LinearVector.transform(self.ss.state_variables, OutputVariable))

        # for state recovery purposes
        self.projection_gain = libss.Gain(V,
                                          input_vars=LinearVector([InputVariable('krylov', size=V.shape[1], index=0)]),
                                          output_vars=LinearVector.transform(self.ss.state_variables, OutputVariable))

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
                lu_a = krylovutils.lu_factor(frequency[i], A)
                F = krylovutils.lu_solve(lu_a, np.eye(n))
                G = krylovutils.lu_solve(lu_a, B)

            if i == 0:
                V = krylovutils.block_arnoldi_krylov(r, F, G)
            else:
                Vi = krylovutils.block_arnoldi_krylov(r, F, G)
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

        order = np.argsort(eigs_abs)[::-1]
        eigs = eigs[order]
        eigs_abs = eigs_abs[order]

        unstable = False
        if self.sstype == 'dt':
            if any(eigs_abs > 1.):
                unstable = True
                unstable_eigenvalues = eigs[eigs_abs > 1.]
                try:
                    cout.cout_wrap('Unstable ROM - %d Eigenvalues with |r| > 1' % len(unstable_eigenvalues))
                except ValueError:
                    pass
                for mu in unstable_eigenvalues:
                    try:
                        cout.cout_wrap('\tmu = %f + %fj' % (mu.real, mu.imag))
                    except ValueError:
                        pass
            else:
                try:
                    cout.cout_wrap('ROM is stable')
                    cout.cout_wrap('\tDT Eigenvalues:')
                    cout.cout_wrap(len(eigs_abs) * '\t\tmu = %4f + %4fj\n' % tuple(eigs.view(float)))
                except ValueError:
                    pass

        else:
            if any(eigs.real > 0):
                unstable = True
                cout.cout_wrap('Unstable ROM', 3)
                unstable_eigenvalues = eigs[eigs.real > 0]
            else:
                cout.cout_wrap('ROM is stable')

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

        return not unstable

    def load_tangent_vectors(self):

        tangent_file = self.settings['tangent_input_file']
        if tangent_file:
            tangents = h5.readh5(tangent_file)
            right_tangent = tangents.right_tangent
            left_tangent = tangents.left_tangent
            rc = tangents.rc
            ro = tangents.ro
            fc = tangents.fc
            fo = tangents.fo
        else:
            left_tangent = None
            right_tangent = None

        return left_tangent, right_tangent, rc, ro, fc, fo

    def stable_realisation(self, *args, **kwargs):
        r"""Remove unstable poles left after reduction

        Using a Schur decomposition of the reduced plant matrix :math:`\mathbf{A}_m\in\mathbb{C}^{m\times m}`,
        the method removes the unstable eigenvalues that could have appeared after the moment-matching reduction.

        The oblique projection matrices :math:`\mathbf{T}_L\in\mathbb{C}^{m \times p}` and
        :math:`\mathbf{T}_R\in\mathbb{C}^{m \times p}`` result in a stable realisation

        .. math:: \mathbf{A}_s = \mathbf{T}_L^\top\mathbf{AT}_R \in \mathbb{C}^{p\times p}.

        Args:
            A (np.ndarray): plant matrix (if not provided ``self.ssrom.A`` will be used).

        Returns:
            tuple: Left and right projection matrices :math:`\mathbf{T}_L\in\mathbb{C}^{m \times p}` and
                :math:`\mathbf{T}_R\in\mathbb{C}^{m \times p}`

        References:
            Jaimoukha, I. M., Kasenally, E. D.. Implicitly Restarted Krylov Subspace Methods for Stable Partial
            Realizations. SIAM Journal of Matrix Analysis and Applications, 1997.

        See Also:
            The method employs :func:`sharpy.rom.utils.krylovutils.schur_ordered()` and
            :func:`sharpy.rom.utils.krylovutils.remove_a12`.
        """

        cout.cout_wrap('Stabilising system by removing unstable eigenvalues using a Schur decomposition', 1)
        if self.ssrom is None:
            A = args[0]
            ct = kwargs['ct']
            assert type(ct) == bool, 'CT system flag should be a bool'
        else:
            A = self.ssrom.A

            if self.sstype == 'ct':
                ct = True
            else:
                ct = False

        m = A.shape[0]
        As, T1, n_stable = krylovutils.schur_ordered(A, ct=ct)

        # Remove the (1,2) block of the Schur ordered matrix
        T2, X = krylovutils.remove_a12(As, n_stable)

        T3 = np.eye(m, n_stable)

        TL = T3.T.dot(T2.dot(np.conj(T1)))
        TR = T1.T.dot(np.linalg.inv(T2).dot(T3))

        cout.cout_wrap('System reduced to %g states' %n_stable, 1)

        # for debugging
        # import matplotlib.pyplot as plt
        Ar = TL.dot(A.dot(TR))
        eigsA = np.linalg.eigvals(A)
        eigsAr = np.linalg.eigvals(Ar)
        #
        # plt.scatter(eigsA.real, eigsA.imag)
        # plt.scatter(eigsAr.real, eigsAr.imag)
        # plt.show()
        if ct:
            assert np.sum(np.real(eigsAr) <= 0.) == n_stable, 'Number of stable eigvals computed not equal to those in Ar'
        else:
            assert np.sum(np.abs(eigsAr) <= 1.) == n_stable, 'Number of stable eigvals computed not equal to those in Ar'

        return TL.T, TR

    def restart(self):
        """
        Implicitly Restarted Krylov Algorithm
        """

        # Run krylov here

        W = self.W
        V = self.V

        check_eye(V, V.T)
        check_eye(W, W.T)

        Tm = W.T.dot(V)
        Tminv = sclalg.inv(Tm)

        check_eye(Tm, Tminv, 'Tm = W^T.dot(V)')

        # d = Tminv.dot(W.T)
        #
        # Tmcond = np.linalg.cond(Tm)
        #
        # if Tmcond > 1e10:
        #     cout.cout_wrap('Matrix Tm = W^T.dot(V) is poorly conditioned. Condition = %e' % Tmcond, 3)

        TL, TR = self.stable_realisation()

        m, r = TR.shape

        Ar = TL.T.dot(self.ssrom.A.dot(TR))
        eigsAr = np.linalg.eigvals(Ar)
        n_stable_r = np.sum(np.abs(eigsAr)<=1.)

        assert n_stable_r == r, 'lost evals'

        # QR decomposition
        QRfull, RRfull = sclalg.qr(TR)
        QLfull, RLfull = sclalg.qr(sclalg.inv(Tm.T).dot(TL))

        QR = QRfull[:, :r]
        QRcomp = QRfull[:, r:]
        RR = RRfull[:r, :]
        RRcomp = RRfull[r:, :]

        QL = QLfull[:, :r]
        RL = RLfull[:r, :]

        Vr = V.dot(QR)
        Wr = W.dot(QL)
        Tr = Wr.T.dot(Vr)

        print('Tr cond = %e' % np.linalg.cond(Tr))

        Trinv = sclalg.inv(QL.T.dot(Tm.dot(QR)))
        Trinv2 = RR.dot(RL.T)
        #
        print('Trinv diff = %f' % (np.max(np.abs(Trinv - Trinv2))))
        Wtr = Trinv.dot(Wr.T)
        # Wtr = Wr.T
        # return QL.dot(Trinv), QR
        Wtr2 = TL.T.dot(Tminv.dot(W.T))
        # return Wtr2, V.dot(TR)

        Ar = RR.dot(TL.T).dot(self.ssrom.A.dot(QR))
        eigsAr = np.linalg.eigvals(Ar)
        n_stable_r = np.sum(np.abs(eigsAr)<=1.)

        assert n_stable_r == r, 'lost evals'



        return RR.dot(TL.T), QR
        # pass


def reduction_checks(T, Tinv):

    cout.cout_wrap('Tm condition = %e' % np.linalg.cond(T))

    check_eye(T, Tinv)


def check_eye(T, Tinv, msg=''):

    eye_approx = Tinv.dot(T)
    max_diff = np.max(np.abs(np.eye(eye_approx.shape[0]) - eye_approx))

    if max_diff != 0.0:
        log_error = np.log10(max_diff)
        assert log_error < -6, 'T.dot(Tinv) not equal to identity, %s \nlog(error) = %.e' \
                               % (msg, np.log10(max_diff))


def check_rank(V, W):

    n_cols = V.shape[1]

    for col in range(1, n_cols):
        T = W[:, :col].T.dot(V[:, :col])
        Tinv = np.linalg.inv(T)
        try:
            check_eye(T, Tinv)
        except AssertionError:
            cout.cout_wrap('\tDeflating column %g' % col, 1)
            W = np.hstack((W[:, :col-1], W[:, col:]))
            V = np.hstack((V[:, :col-1], V[:, col:]))

    try:
        check_eye(T, Tinv)
    except AssertionError:
        V, W = check_rank(V, W)

    # need to check if necessary
    # V = krylovutils.mgs_ortho(V)
    # W = krylovutils.mgs_ortho(W)
    # V, W = check_rank(V, W)
    # breakpoint()

    return V, W

if __name__=="__main__":
    import numpy as np

    A = np.random.rand(20, 20)
    eigsA = np.sort(np.abs(np.linalg.eigvals(A)))
    print(eigsA)
    print("Number of stable eigvals = %g" %np.sum(np.abs(eigsA)<=1) )
    rom = Krylov()
    TL, TR = rom.stable_realisation(A)

    Ap = TL.T.dot(A.dot(TR))

    eigsA = np.sort(np.abs(np.linalg.eigvals(Ap)))
    print("\nNew matrix size %g" % Ap.shape[0])
    print('Stable eigvals = %g' % np.sum(np.abs(eigsA)<=1))
    print(eigsA)
