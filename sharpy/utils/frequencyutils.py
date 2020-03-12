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
