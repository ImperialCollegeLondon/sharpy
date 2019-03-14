"""
Class to store linear uvlm solver
S. Maraniello, Aug 2018
"""

import copy
import numpy as np
import scipy as sc
import scipy.linalg as scalg
import scipy.signal as scsig
import sharpy.linear.src.libss as libss 


class FlexDynamic():
    r"""
    Define class for linear state-space realisation of GEBM flexible-body
    equations (without linearised rigid-body dynamics) from Sharpy
    ``timestep_info`` class.

    State-space models can be defined in continuous or discrete time (dt
    required). Modal projection, either on the damped or undamped modal shapes,
    is also aviable. The rad/s array wv can be optionally passed for freq.
    response analysis

    To produce the state-space equations:

    1. Set the attributes

        a. ``self.modal={True,False}``: determines whether to project the states
            onto modal coordinates. Projection over damped or undamped modal
            shapes can be obtained selecting:

                - ``self.proj_modes={'damped','undamped'}``

            while

                 - ``self.inout_coords={'modes','nodal'}``

             determines whether the modal state-space inputs/outputs are modal
             coords or nodal degrees-of-freedom. If 'modes' is selected, the
             Kin and Kout gain matrices are generated to transform nodal to modal
             dofs

        b. ``self.dlti={True,False}``: if true, generates discrete-time system.
            The continuous to discrete transformation method is determined by::

                self.discr_method={ 'newmark',  # Newmark-beta
                                    'zoh',		# Zero-order hold
                                    'bilinear'} # Bilinear (Tustin) transformation

            DLTIs can be obtained directly using the Newmark-:math:`\beta` method

                ``self.discr_method='newmark'``
                ``self.newmark_damp=xx`` with ``xx<<1.0``

            for full-states descriptions (``self.modal=False``) and modal projection
            over the undamped structural modes (``self.modal=True`` and ``self.proj_modes``).
            The Zero-order holder and bilinear methods, instead, work in all
            descriptions, but require the continuous state-space equations.


    2. Run ``self.assemble()``. The method accepts an additional parameter, ``Nmodes``,
    which allows using a lower number of modes than specified in ``self.Nmodes``
    
    Examples:
        >>> self.dlti = True
        >>> self.newmark_damp = 5e-3
        >>> self.modal = True
        >>> self.proj_modes = 'undamped'
        >>> self.Nmodes = num_modes
        >>> self.discr_method = 'newmark'


    Notes:

        * Modal projection will automatically select between damped/undamped modes shapes, based on the data available
          from tsinfo.

        * If the full system matrices are available, use the modal_sol methods to override mode-shapes and eigenvectors


    """

    def __init__(self, tsinfo, dt=None, wv=None):

        # Settings - WIP - maybe better to go into a new solver such as LinGEBMLoader or similar
        # ----->
        # self.settings_default['modal_projection'] = True
        # self.settings_types['modal_projection'] = 'bool'
        #
        # self.settings_default['inout_coords'] = 'nodes' # or 'modes'
        # self.settings_types['inout_coords'] = 'str'
        #
        # self.settings_default['proj_modes'] = 'damped'
        # self.settings_types['proj_modes'] = 'str'
        #
        # self.settings_default['discrete_time'] = True
        # self.settings_types['discrete_time'] = 'bool'
        #
        # self.settings_default['dt'] = 0.001
        # self.settings_types['dt'] = 'float'
        #
        # self.settings_default['proj_modes'] = 'damped'
        # self.settings_types['proj_modes'] = 'str'
        #
        # self.settings_default['discr_method'] = 'newmark'
        # self.settings_types['discr_methods'] = 'str'
        #
        # self.settings_default['newmark_damp'] = 1e-4
        # self.settings_types['newmark_damp'] = 'float'
        # <-----

        ### extract timestep_info modal results
        # unavailable attrs will be None
        self.freq_natural = tsinfo.modal.get('freq_natural')
        self.freq_damp = tsinfo.modal.get('freq_damped')

        self.damping = tsinfo.modal.get('damping')

        self.eigs = tsinfo.modal.get('eigenvalues')
        self.U = tsinfo.modal.get('eigenvectors')
        self.V = tsinfo.modal.get('eigenvectors_left')
        self.Kin_damp = tsinfo.modal.get('Kin_damp')  # for 'damped' modes only
        self.Ccut = tsinfo.modal.get('Ccut')  # for 'undamp' modes only

        self.Mstr = tsinfo.modal.get('M')
        self.Cstr = tsinfo.modal.get('C')
        self.Kstr = tsinfo.modal.get('K')

        self.Nmodes = len(self.damping)
        self.num_dof = self.U.shape[0]
        if self.V is not None:
            self.num_dof = self.num_dof // 2

        ### set other flags
        self.modal = True
        self.inout_coords = 'nodes'  # 'modes'
        self.dlti = True
        if dt is None:
            self.dlti = False
        self.proj_modes = 'damped'
        if self.V is None:
            self.proj_modes = 'undamped'
        self.discr_method = 'newmark'
        self.newmark_damp = 1e-3

        ### set state-space variables
        self.dt = dt
        self.wv = wv
        self.SScont = None
        self.SSdisc = None
        self.Kin = None
        self.Kout = None

    def assemble(self, Nmodes=None):
        r"""
        Assemble state-space model

        Several assembly options are available:

        1. Discrete-time, Newmark-:math:`\beta`:
            * Modal projection onto undamped modes. It uses the modal projection such
              that the generalised coordinates :math:`\eta` are transformed into modal space by

                    .. math:: \mathbf{\eta} = \mathbf{\Phi\,q}

                where :math:`\mathbf{\Phi}` are the first ``Nmodes`` right eigenvectors.
                Therefore, the equation of motion can be re-written such that the modes normalise the mass matrix to
                become the identity matrix.

                    .. math:: \mathbf{I_{Nmodes}}\mathbf{\ddot{q}} + \mathbf{\Lambda_{Nmodes}\,q} = 0

                The system is then assembled in Newmark-:math:`\beta` form as detailed in :func:`newmark_ss`

            * Full size system assembly. No modifications are made to the mass, damping or stiffness matrices and the
              system is directly assembled by :func:`newmark_ss`.

        2. Continuous time state-space


        Args:
            Nmodes (int): number of modes to retain
        """

        ### checks
        assert self.inout_coords in ['modes', 'nodes'], \
            'inout_coords=%s not implemented!' % self.inout_coords

        dlti = self.dlti
        modal = self.modal
        num_dof = self.num_dof
        if Nmodes is None or Nmodes >= self.Nmodes:
            Nmodes = self.Nmodes
        # else:
        #     # Modal truncation
        #     self.update_truncated_modes(Nmodes)

        if dlti:  # ---------------------------------- assemble discrete time

            if self.discr_method in ['zoh', 'bilinear']:
                # assemble continuous-time
                self.dlti = False
                self.assemble(Nmodes)
                # convert into discrete
                self.dlti = True
                self.cont2disc()

            elif self.discr_method == 'newmark':

                if modal:  # Modal projection
                    if self.proj_modes == 'undamped':
                        Phi = self.U[:, :Nmodes]

                        if self.Ccut is None:
                            Ccut = np.zeros((Nmodes, Nmodes))
                        else:
                            Ccut = np.dot(Phi.T, np.dot(self.Ccut, Phi))

                        Ass, Bss, Css, Dss = newmark_ss(
                            np.eye(Nmodes),
                            Ccut,
                            np.diag(self.freq_natural[:Nmodes] ** 2),
                            self.dt,
                            self.newmark_damp)
                        self.Kin = Phi.T
                        self.Kout = sc.linalg.block_diag(*[Phi, Phi])
                    else:
                        raise NameError(
                            'Newmark-beta discretisation not available ' \
                            'for projection on damped eigenvectors')

                    # build state-space model
                    self.SSdisc = libss.ss(Ass, Bss, Css, Dss, dt=self.dt)
                    if self.inout_coords == 'nodes':
                        self.SSdisc = libss.addGain(self.SSdisc, self.Kin, 'in')
                        self.SSdisc = libss.addGain(self.SSdisc, self.Kout, 'out')
                        self.Kin, self.Kout = None, None


                else:  # Full system
                    Ass, Bss, Css, Dss = newmark_ss(
                        np.linalg.inv(self.Mstr), self.Cstr, self.Kstr,
                        self.dt, self.newmark_damp)
                    self.Kin = None
                    self.Kout = None
                    self.SSdisc = libss.ss(Ass, Bss, Css, Dss, dt=self.dt)

            else:
                raise NameError(
                    'Discretisation method %s not available' % self.discr_method)


        else:  # -------------------------------- assemble continuous time

            if modal:  # Modal projection

                Ass = np.zeros((2 * Nmodes, 2 * Nmodes))
                Css = np.eye(2 * Nmodes)
                iivec = np.arange(Nmodes, dtype=int)

                if self.proj_modes == 'undamped':
                    Phi = self.U[:, :Nmodes]
                    Ass[iivec, Nmodes + iivec] = 1.
                    Ass[Nmodes:, :Nmodes] = -np.diag(self.freq_natural[:Nmodes] ** 2)
                    if self.Ccut is not None:
                        Ass[Nmodes:, Nmodes:] = -self.Ccut[:Nmodes, :Nmodes]
                    Bss = np.zeros((2 * Nmodes, Nmodes))
                    Dss = np.zeros((2 * Nmodes, Nmodes))
                    Bss[Nmodes + iivec, iivec] = 1.
                    self.Kin = Phi.T
                    self.Kout = sc.linalg.block_diag(*(Phi, Phi))
                else:  # damped mode shapes
                    # The algorithm assumes that for each couple of complex conj
                    # eigenvalues, only one eigenvalue (and the eigenvectors
                    # associated to it) is include in self.eigs.  
                    eigs = self.eigs[:Nmodes]
                    U = self.U[:, :Nmodes]
                    V = self.V[:, :Nmodes]
                    Ass[iivec, iivec] = eigs.real
                    Ass[iivec, Nmodes + iivec] = -eigs.imag
                    Ass[Nmodes + iivec, iivec] = eigs.imag
                    Ass[Nmodes + iivec, Nmodes + iivec] = eigs.real
                    Bss = np.eye(2 * Nmodes)
                    Dss = np.zeros((2 * Nmodes, 2 * Nmodes))
                    self.Kin = np.block(
                        [[self.Kin_damp[iivec, :].real],
                         [self.Kin_damp[iivec, :].imag]])
                    self.Kout = np.block([2. * U.real, (-2.) * U.imag])

                # build state-space model
                self.SScont = libss.ss(Ass, Bss, Css, Dss)
                if self.inout_coords == 'nodes':
                    self.SScont = libss.addGain(self.SScont, self.Kin, 'in')
                    self.SScont = libss.addGain(self.SScont, self.Kout, 'out')
                    self.Kin, self.Kout = None, None

            else:  # Full system
                if self.Mstr is None:
                    raise NameError('Full-states matrices not available')
                Mstr, Cstr, Kstr = self.Mstr, self.Cstr, self.Kstr

                Ass = np.zeros((2 * num_dof, 2 * num_dof))
                Bss = np.zeros((2 * num_dof, num_dof))
                Css = np.eye(2 * num_dof)
                Dss = np.zeros((2 * num_dof, num_dof))
                Minv_neg = -np.linalg.inv(self.Mstr)
                Ass[range(num_dof), range(num_dof, 2 * num_dof)] = 1.
                Ass[num_dof:, :num_dof] = np.dot(Minv_neg, Kstr)
                Ass[num_dof:, num_dof:] = np.dot(Minv_neg, Cstr)
                Bss[num_dof:, :] = -Minv_neg
                self.Kin = None
                self.Kout = None
                self.SScont = libss.ss(Ass, Bss, Css, Dss)

    def freqresp(self, wv=None, bode=True):
        """
        Computes the frequency response of the current state-space model. If
        ``self.modal=True``, the in/out are determined according to ``self.inout_coords``
        """

        if wv is None:
            wv = self.wv
        assert wv is not None, 'Frequency range not provided.'

        if self.dlti:
            self.Ydisc = libss.freqresp(self.SSdisc, wv, dlti=self.dlti)
            if bode:
                self.Ydisc_abs = np.abs(self.Ydisc)
                self.Ydisc_ph = np.angle(self.Ydisc, deg=True)
        else:
            self.Ycont = libss.freqresp(self.SScont, wv, dlti=self.dlti)
            if bode:
                self.Ycont_abs = np.abs(self.Ycont)
                self.Ycont_ph = np.angle(self.Ycont, deg=True)

    def converge_modal(self, wv=None, tol=None, Yref=None, Print=False):
        """ 
        Determine number of modes required to achieve a certain convergence
        of the modal solution in a prescribed frequency range ``wv``. The H-infinity
        norm of the error w.r.t. ``Yref`` is used for assessing convergence.

        .. Warning:: if a reference freq. response, Yref, is not provided, the full-
            state continuous-time frequency response is used as reference. This
            requires the full-states matrices ``Mstr``, ``Cstr``, ``Kstr`` to be available.
        """

        if wv is None:
            wv = self.wv
        assert wv is not None, 'Frequency range not provided.'
        assert tol is not None, 'Tolerance, tol, not provided'
        assert self.modal is True, 'Convergence analysis requires modal=True'

        if Yref is None:
            # use cont. time. full-states as reference
            dlti_here = self.dlti
            self.modal = False
            self.dlti = False
            self.assemble()
            self.freqresp(wv)
            Yref = self.Ycont.copy()
            self.dlti = dlti_here
            self.modal = True

        if Print:
            print('No. modes\tError\tTolerance')
        for nn in range(1, self.Nmodes + 1):
            self.assemble(Nmodes=nn)
            self.freqresp(wv, bode=False)
            Yhere = self.Ycont
            if self.dlti: Yhere = self.Ydisc
            er = np.max(np.abs(Yhere - Yref))
            if Print: print('%.3d\t%.2e\t%.2e' % (nn, er, tol))
            if er < tol:
                if Print: print('Converged!')
                self.Nmodes = nn
                break

    def tune_newmark_damp(self, amplification_factor=0.999):
        """
        Tune artifical damping to achieve a percent reduction of the lower 
        frequency (lower damped) mode
        """

        assert self.discr_method == 'newmark' and self.dlti, \
            "select self.discr_method='newmark' and self.dlti=True"

        newmark_damp = self.newmark_damp
        import scipy.optimize as scopt

        def get_res(newmark_damp_log10):
            self.newmark_damp = 10. ** (newmark_damp_log10)
            self.assemble()
            eigsabs = np.abs(np.linalg.eigvals(self.SSdisc.A))
            return np.max(eigsabs) - amplification_factor

        exp_opt = scopt.fsolve(get_res, x0=-3)[0]

        self.newmark_damp = 10. ** exp_opt
        print('artificial viscosity: %.4e' % self.newmark_damp)

    def update_modal(self):
        r"""
        Re-projects the full-states continuous-time structural dynamics equations

        .. math::
            \mathbf{M}\,\mathbf{\ddot{x}} +\mathbf{C}\,\mathbf{\dot{x}} + \mathbf{K\,x} = \mathbf{F}

        onto modal space. The modes used to project are controlled through the
        ``self.proj_modes={damped or undamped}`` attribute.

        .. Warning:: This method overrides SHARPy ``timestep_info`` results and requires
            ``Mstr``, ``Cstr``, ``Kstr`` to be available.

        """
        pass

    def update_truncated_modes(self, nmodes):
        r"""
        Updates the system to the specified number of modes

        Args:
            nmodes:

        Returns:

        """

        # Verify that the new number of modes is less than the current value
        assert nmodes < self.Nmodes, 'Unable to truncate to %g modes since only %g are available' %(nmodes, self.Nmodes)

        self.Nmodes = nmodes
        self.eigs = self.eigs[:nmodes]
        self.U = self.U[:,:nmodes]
        self.freq_natural = self.freq_natural[:nmodes]
        try:
            self.freq_damp[:nmodes]
        except TypeError:
            pass

        # Update Ccut matrix
        self.Ccut = np.dot(self.U.T, np.dot(self.Cstr, self.U))

    def cont2disc(self, dt=None):
        """Convert continuous-time SS model into """

        assert self.discr_method is not 'newmark', \
            'For Newmark-beta discretisation, use assemble method directly.'

        if dt is not None:
            self.dt = dt
        else:
            assert self.dt is not None, \
                'Provide time-step for convertion to discrete-time'

        SScont = self.SScont
        tpl = scsig.cont2discrete(
            (SScont.A, SScont.B, SScont.C, SScont.D),
            dt=self.dt, method=self.discr_method)
        self.SSdisc = libss.ss(*tpl[:-1], dt=tpl[-1])
        self.dlti = True


def newmark_ss(Minv, C, K, dt, num_damp=1e-4):
    r"""
    Produces a discrete-time state-space model of the structural equations

    .. math::

        \mathbf{\ddot{x}} &= \mathbf{M}^{-1}( -\mathbf{C}\,\mathbf{\dot{x}}-\mathbf{K}\,\mathbf{x}+\mathbf{F} ) \\
        \mathbf{y} &= \mathbf{x}


    based on the Newmark-:math:`\beta` integration scheme. The output state-space model
    has form:

    .. math::

        \mathbf{X}_{n+1} &= \mathbf{A}\,\mathbf{X}_n + \mathbf{B}\,\mathbf{F}_n \\
        \mathbf{Y} &= \mathbf{C}\,\mathbf{X} + \mathbf{D}\,\mathbf{F}


    with :math:`\mathbf{X} = [\mathbf{x}, \mathbf{\dot{x}}]^T`

    Note that as the state-space representation only requires the input force
    :math:`\mathbf{F}` to be evaluated at time-step :math:`n`,the :math:`\mathbf{C}` and :math:`\mathbf{D}` matrices
    are, in general, fully populated.

    Args:
        Minv (np.array): Inverse mass matrix :math:`\mathbf{M^{-1}}`
        C (np.array): Damping matrix :math:`\mathbf{C}`
        K (np.array): Stiffness matrix :math:`\mathbf{K}`
        dt (float): Timestep increment
        num_damp (float): Numerical damping. Default ``1e-4``

    Returns:
        tuple: the A, B, C, D matrices of the state space packed in a tuple with the predictor and delay term removed.
    """

    # weights
    th1 = 0.5 + num_damp
    # th2=0.25*(th1+.5)**2
    th2 = 0.0625 + 0.25 * (th1 + th1 ** 2)

    dt2 = dt ** 2
    a1 = th2 * dt2
    a0 = 0.5 * dt2 - a1
    b1 = th1 * dt
    b0 = dt - b1

    # relevant matrices
    N = K.shape[0]
    Imat = np.eye(N)
    MinvK = np.dot(Minv, K)
    MinvC = np.dot(Minv, C)

    # build ss
    Ass0 = np.block([[Imat - a0 * MinvK, dt * Imat - a0 * MinvC],
                     [-b0 * MinvK, Imat - b0 * MinvC]])
    Ass1 = np.block([[Imat + a1 * MinvK, a1 * MinvC],
                     [b1 * MinvK, Imat + b1 * MinvC]])
    Ass = np.linalg.solve(Ass1, Ass0)

    Bss0 = np.linalg.solve(Ass1, np.block([[a0 * Minv], [b0 * Minv]]))
    Bss1 = np.linalg.solve(Ass1, np.block([[a1 * Minv], [b1 * Minv]]))

    # eliminate predictior term Bss1
    return libss.SSconv(Ass, Bss0, Bss1, C=np.eye(2 * N), D=np.zeros((2 * N, N)))


def sort_eigvals(eigv, eigabsv, tol=1e-6):
    """ sort by magnitude (frequency) and imaginary part if complex conj """

    order = np.argsort(np.abs(eigv))
    eigv = eigv[order]

    for ii in range(len(eigv) - 1):
        # check if ii and ii+1 are the same eigenvalue
        if np.abs(eigv[ii].imag + eigv[ii + 1].imag) / eigabsv[ii] < tol:
            if np.abs(eigv[ii].real - eigv[ii + 1].real) / eigabsv[ii] < tol:

                # swap if required
                if eigv[ii].imag > eigv[ii + 1].imag:
                    temp = eigv[ii]
                    eigv[ii] = eigv[ii + 1]
                    eigv[ii + 1] = temp

                    temp = order[ii]
                    order[ii] = order[ii + 1]
                    order[ii + 1] = temp

    return order, eigv
