"""Frequency Space Tools

"""
import warnings

import numpy as np
import scipy.linalg as sclalg

from sharpy.utils import cout_utils as cout


def frequency_error(Y_fom, Y_rom, wv):
    n_in = Y_fom.shape[1]
    n_out = Y_fom.shape[0]
    cout.cout_wrap('Computing error in frequency response')
    max_error = np.zeros((n_out, n_in, 2))
    for m in range(n_in):
        for p in range(n_out):
            cout.cout_wrap('m = %g, p = %g' % (m, p))
            max_error[p, m, 0] = error_between_signals(Y_fom[p, m, :].real,
                                                            Y_rom[p, m, :].real,
                                                            wv, 'real')
            max_error[p, m, 1] = error_between_signals(Y_fom[p, m, :].imag,
                                                            Y_rom[p, m, :].imag,
                                                            wv, 'imag')

    if np.max(np.log10(max_error)) >= 0:
        warnings.warn('Significant mismatch in the frequency response of the ROM and FOM')

    return np.max(max_error)


def error_between_signals(sig1, sig2, wv, sig_title=''):
    abs_error = np.abs(sig1 - sig2)
    max_error = np.max(abs_error)
    max_error_index = np.argmax(abs_error)
    pct_error = max_error/sig1[max_error_index]

    max_err_freq = wv[max_error_index]
    if 1e-1 > max_error > 1e-3:
        c = 3
    elif max_error >= 1e-1:
        c = 4
    else:
        c = 1
    cout.cout_wrap('\tError Magnitude -%s-: log10(error) = %.2f (%.2f pct) at %.2f rad/s'
                   % (sig_title, np.log10(max_error), pct_error, max_err_freq), c)

    return max_error


def freqresp_relative_error(y1, y2, wv=None, **kwargs):
    r"""
    Relative error between a reference signal and a second signal.

    The error metric is defined as in [1] to be:

    .. math:: \varepsilon_{rel}[\mathbf{Y}_1, \mathbf{Y}_2] = \frac{\max_{i,j} (\sup_{w\in[0, \bar{w}]}[\mathbf{Y}_2 -
        \mathbf{Y}_1]_{i, j})}{\max_{i,j}(\sup_{w\in[0, \bar{w}]}[\mathbf{Y}_1]_{i,j})}.


    Args:
        y1 (np.ndarray): Reference signal frequency response.
        y2 (np.ndarray): Frequency response matrix.
        wv (Optional [np.ndarray]): Array of frequencies. Required when specifying a max and min value
        **kwargs: Key word arguments for max and min frequencies. See below.

    Keyword Args
        vmin (float): Lower bound value to find index in ``wv``.
        vmax (float): Upper bound value to find index in ``wv``.

    Returns:
        float: Maximum relative error between frequency responses.

    References:
        Maraniello, S. and Palacios, R. Parametric Reduced Order Modelling of the Unsteady Vortex Lattice Method.
        AIAA Journal. 2020
    """
    p, m, nwv = y1.shape

    assert (p, m, nwv) == y2.shape, "Frequency responses do not have the same number of inputs, " \
                                    "outputs or evaluation points."

    # freq_upper_limit = kwargs.get('vmax', None)
    # freq_lower_limit = kwargs.get('vmin', 0)
    #
    # if wv is not None and freq_upper_limit is not None:
    #     i_min, i_max = find_limits(wv, vmin=freq_lower_limit, vmax=freq_upper_limit)
    # else:
    #     i_min = 0
    #     i_max = None

    i_min, i_max = find_limits(wv, **kwargs)

    err = np.zeros((p, m))
    for pi in range(p):
        for mi in range(m):
            err[pi, mi] = np.max(y1[pi, mi, i_min:i_max] - y2[pi, mi, i_min:i_max]) / np.max(y1[pi, mi, i_min:i_max])

    return np.max(err)


def find_limits(wv, **kwargs):
    """
    Returns the indices corresponding to the ``vmax`` and ``vmin`` key-word arguments parsed found in the ordered
    array ``wv``.

    Args:
        wv (np.ndarray): Ordered range.

    Keyword Args:
        vmin (float): Lower bound value to find index in ``wv``.
        vmax (float): Upper bound value to find index in ``wv``.

    Returns:
        tuple: Index of ``vmin`` and index of ``vmax``.

    """
    freq_upper_limit = kwargs.get('vmax', None)
    freq_lower_limit = kwargs.get('vmin', 0)

    if wv is not None and freq_upper_limit is not None:
        # find indices in frequencies
        i_max = np.where(freq_upper_limit - wv >= 0)[-1][-1]
        try:
            i_min = np.where(freq_lower_limit - wv >= 0)[-1][-1]
        except IndexError:
            i_min = 0

    else:
        i_min = 0
        i_max = None

    return i_min, i_max


def frobenius_norm(a):
    r"""
    Frobenius norm. Also known as Schatten 2-norm or Hilbert-Schmidt norm.

    .. math:: ||\mathbf{A}||_F = \sqrt{\mathrm{trace}(\mathbf{A^*A})}

    Args:
        a (np.ndarray): Complex matrix.

    Returns:
        float: Frobenius norm of the matrix ``a``.

    References:
        Antoulas, A. Approximation to Large Scale Dynamical Systems. SIAM 2005. Ch 3, Eq 3.5

    """

    dims = len(a.shape)

    if dims == 1:
        a.shape = (a.shape[0], 1)

    a_star = np.conj(a).T

    return np.sqrt(np.trace(a_star.dot(a)))


def l2norm(y_freq, wv, **kwargs):
    r"""
    Computes the L-2 norm of a complex valued function.

    .. math:: \mathcal{L}_2 = \left(\int_{-\infty}^\infty ||\mathbf{F}(i\omega)||_{F2}\,d\omega\right)^{0.5}

    where :math:`||\mathbf{F}(i\omega)||_{F2}` refers to teh Frobenius norm calculated by
    :func:`sharpy.utils.frequencyutils.frobenius_norm`.

    Args:
        y_freq (np.ndarray): Complex valued function.
        wv (np.ndarray): Frequency array.
        **kwargs: Key word arguments for max and min frequencies. See below.

    Keyword Args
        vmin (float): Lower bound value to find index in ``wv``.
        vmax (float): Upper bound value to find index in ``wv``.

    Returns:
        float: L-2 norm of ``y_freq``.

    References:
        Antoulas, A. Approximation to Large Scale Dynamical Systems. SIAM 2005. Ch 5, Eq 5.10, pg 126
    """

    nwv = y_freq.shape[-1]

    assert nwv == len(wv), "Number of frequency evaluations different %g vs %g" % (nwv, len(wv))

    i_min, i_max = find_limits(wv, **kwargs)

    freq_range = wv[i_min:i_max].copy()

    h2 = np.zeros(len(freq_range))  # frobenius norm at each frequency

    for i in range(len(freq_range)):
        h2[i] = frobenius_norm(y_freq[:, :, i])
    integral_h2 = np.sqrt(np.max(np.trapz(h2, freq_range)))

    return integral_h2


def gap_metric(ss1, ss2):
    """
    Computes the gap metric between two linear systems

    Warnings:
        Under Development

    Args:
        ss1:
        ss2:

    Returns:

    References:
        NG - Mendeley Gap Metric
    """

    pass


def right_coprime_factorisation(ss, return_m=False):
    r"""
    Computes the right coprime normalised factors (RCNF) of a linear system.

    For a linear system

    .. math:: \Sigma = \left(\begin{array}{c|c} \mathbf{A} & \mathbf{B} \\ \hline \mathbf{C} & \mathbf{D} \end{array}\right)

    its transfer function :math:`\mathbf{H}(s)\in\mathbb{R}H\infty^{p\times m}` is defined by

    .. math:: \mathbf{H}(s) = \mathbf{C}(s\mathbf{I}_n - \mathbf{A})^{-1}\mathbf{B} + \mathbf{D}.

    The right coprime factorisation results in a factorisation such that

    .. math:: \mathbf{H}(s) = \mathbf{N}(s)\mathbf{M}^{-1}(s)

    where :math:`\mathbf{N}(s)\in\mathbb{R}H_\infty^{p \times m}` and
    :math:`\mathbf{M}(s)\in\mathbf{R}H_\infty^{m \times m}` are stable, rational and proper transfer functions.

    The state-space representations of the above transfer functions are given by

    .. math:: \mathbf{N}(s) = \left(\begin{array}{c|c} \mathbf{A + BF} & \mathbf{BS}^{-1/2} \\ \hline
        \mathbf{C + DF} & \mathbf{DS}^{-1/2} \end{array}\right).

    .. math:: \mathbf{M}(s) = \left(\begin{array}{c|c} \mathbf{A + BF} & \mathbf{BS}^{-1/2} \\ \hline
        \mathbf{F} & \mathbf{S}^{-1/2} \end{array}\right).

    The matrices :math:`\mathbf{S}` and :math:`\mathbf{F}` are given by:

    .. math:: \mathbf{F} = -\mathbf{S}^{-1}(\mathbf{D^\top C} + \mathbf{B^\top X}

    .. math:: \mathbf{S} = \mathbf{I}_m + \mathbf{D^\top D},


    with :math:`\mathbf{S}^{1/2\top}\mathbf{S}^{1/2} = \mathbf{S}` computed using an upper cholesky factorisation.


    The term :math:`\mathbf{X}` is solved for using the generalised continuous algebraic Riccati equation (GCARE):

     .. math:: (\mathbf{A} - \mathbf{BS}^{-1}\mathbf{D^\top C})^\top\mathbf{X} +
        \mathbf{X}(\mathbf{A} - \mathbf{BS}^{-1}\mathbf{D^\top C}) -
        \mathbf{XBS}^{-1}\mathbf{B^\top X} +
        \mathbf{C}^\top(\mathbf{I}_p + \mathbf{DD}^\top)^{-1}\mathbf{C} = \mathbf{0}.


    The inverse representation of :math:`\mathbf{M}(s)` can be computed as

    .. math:: \mathbf{M}^{-1}(s) = \left(\begin{array}{c|c} \mathbf{A} & \mathbf{B} \\ \hline
        \mathbf{S}^{1/2}\mathbf{F} & \mathbf{S}^{1/2} \end{array}\right).

    Notes:

        The Hardy space, :math:`H_\infty` consists of all complex-valued functions of a complex variable
        :math:`\mathbf{F}(s)` which are analytic and bounded in the open right half-plane $\mathrm{Re(s)}>0$.
        By bounded, there exists a limit :math:`b` such that

        .. math:: |\mathbf{F}(s)| \le b, \mathrm{Re}(s)>0

        This bound :math:`b` is referred to as the :math:`H_\infty` norm:

        .. math:: ||\mathbf{F}(s)||_\infty = \mathrm{sup} {|\mathbf{F}(s)|: \mathrm{Re}(s)>0}

        The subset of the Hardy space that consists of all the real-rational functions (i.e. functions with real
        coefficients is denoted by :math:`\mathbb{R}H_\infty`. Therefore, :math:`\mathbf{F}(s)\in\mathbb{R}H_\infty` if
        it is proper (bounded, ||\mathbf{F}(s)||_\infty exists), stable (no poles in the closed right half-plane,
        :math:`\mathrm{Re}(s)\ge 0` and has real-valued coefficients.

    Args:
        ss (sharpy.linear.src.libss.ss): Continuous time linear system.
        return_m (bool (optional)): In addition to :math:`\mathbf{N}(s)` and :math:`\mathbf{M}^{-1}(s)`, also return
          :math:`\mathbf{M}(s)`.

    Returns:
        tuple: Tuple of linear systems :math:`(\mathbf{N}(s), \mathbf{M}^{-1}(s))`. If ``return_m==True`` then the
          system for :math:`\mathbf{M}(s)` is appended to the tuple.

    References:
        Feyel, P.. Robust Control Optimization with Metaheuristics. Wiley 2017. pg 355, Appendix A.
        https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781119340959.app1
    """
    a, b, c, d = ss.get_mats()

    s = np.eye(ss.inputs) + d.T.dot(d)
    sinv = sclalg.inv(s)

    s_root_r = sclalg.cholesky(s, lower=False)  # S^{1/2}

    # Generalised Riccati equation - formatting below for gcare scipy input
    a_care = a - b.dot(sinv.dot(d.T.dot(c)))
    b_care = b
    q_care = c.T.dot(sclalg.inv(np.eye(ss.outputs) + d.dot(d.T)).dot(c))
    r_care = s

    x = sclalg.solve_continuous_are(a_care, b_care, q_care, r_care)

    f = -sinv.dot(d.T.dot(c) + b.T.dot(x))

    sroot_inv = sclalg.inv(s_root_r)  # S^{-1/2}

    n = libss.ss(a + b.dot(f), b.dot(sroot_inv), c + d.dot(f), d.dot(sroot_inv), dt=None)
    m = libss.ss(a + b.dot(f), b.dot(sroot_inv), f, sroot_inv, dt=None)

    minv = libss.ss(a, -b, s_root_r.dot(f), s_root_r, dt=None)  # M^{-1}

    if return_m:
        return n, minv, m
    else:
        return n, minv


if __name__ == '__main__':
    import unittest
    import sharpy.linear.src.libss as libss

    class TestMetrics(unittest.TestCase):

        def setUp(self):
            self.ss = libss.random_ss(10, 4, 3, dt=None)

        def test_coprime(self):
            n, minv, m = right_coprime_factorisation(self.ss, return_m=True)

            wv = np.logspace(-1, 4, 50)
            h = self.ss.freqresp(wv)
            coprime_system = libss.series(minv, n)  # N M^{-1}

            h_bar = coprime_system.freqresp(wv)

            res = h - h_bar

            np.testing.assert_array_almost_equal(np.max(np.abs(res)), 0, decimal=6, verbose=True,
                                                 err_msg='Original transfer function and coprime '
                                                         'factorisation not equal')

            # Useful debug >>>>>>
            # print('L2 %f' % l2norm(h - h_bar, wv))

            # print('Rel err %f' % freqresp_relative_error(h, h_bar, wv))


            # tf_minv = minv.freqresp(wv)
            # tf_m = m.freqresp(wv)
            #
            # h_bar2 = np.zeros((4, 4, len(wv)), dtype=complex)  # this is n^H.dot(n) + m^H.dot(m) = I
            #
            # for i in range(len(wv)):
            #     h_bar[:, :, i] = tf_n[:, :, i].dot(tf_minv[:, :, i])
            #     h_bar2[:, :, i] = np.conj(tf_n[:, :, i]).T.dot(tf_n[:, :, i]) + np.conj(tf_m[:, :, i]).T.dot(tf_m[:, :, i])
            #
            # import matplotlib.pyplot as plt
            # plt.semilogx(wv, h[0, 0, :].real, color='tab:blue', marker='x')
            # plt.semilogx(wv, h_bar[0, 0, :].real, color='tab:orange', marker='+')
            # plt.semilogx(wv, h_bar2[0, 0, :].real, color='tab:red', marker='+')
            # plt.semilogx(wv, h[0, 0, :].imag, ls='--', color='tab:blue', marker='x')
            # plt.semilogx(wv, h_bar2[0, 0, :].imag, ls='--', color='tab:green', marker='+')
            # plt.show()
            # <<<<<<<<<<<<

    unittest.main()
