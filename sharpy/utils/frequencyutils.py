"""Frequency Space Tools

"""
import warnings

import numpy as np
import scipy.linalg as sclalg

from sharpy.utils import cout_utils as cout
import sharpy.linear.src.libss as libss


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

    .. math:: \mathcal{L}_2 = \left(\int_{-\infty}^\infty ||\mathbf{F}(i\omega)||^2_{F2}\,d\omega\right)^{0.5}

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
        h2[i] = frobenius_norm(y_freq[:, :, i]) ** 2
    integral_h2 = np.sqrt(np.max(np.trapz(h2, freq_range)))

    return integral_h2


def hamiltonian(gamma, ss):
    """
    Returns the Hamiltonian of a linear system as defined in [1].


    References:

        [1] Bruinsma, N. A., & Steinbuch, M. (1990). A fast algorithm to compute the H∞-norm of a transfer function
        matrix. Systems and Control Letters, 14(4), 287–293. https://doi.org/10.1016/0167-6911(90)90049-Z

    Args:
        gamma (float): Evaluation point.
        ss (sharpy.linear.src.libss.StateSpace): Linear system.

    Returns:
        np.ndarray: Hamiltonian evaluated at ``gamma``.
    """

    a, b, c, d = ss.get_mats()

    p, m = d.shape

    r = d.T.dot(d) - gamma ** 2 * np.eye(m)
    s = d.dot(d.T) - gamma ** 2 * np.eye(p)

    rinv = sclalg.inv(r)
    sinv = sclalg.inv(s)

    ham = np.block([[a - b.dot(rinv.dot(d.T.dot(c))),  - gamma * b.dot(rinv.dot(b.T))],
                    [gamma * c.T.dot(sinv.dot(c)), - a.T + c.T.dot(d.dot(rinv.dot(b.T)))]])
    return ham


def h_infinity_norm(ss, **kwargs):
    r"""
    Returns H-infinity norm of a linear system using iterative methods.

    The H-infinity norm of a MIMO system is traditionally calculated finding the largest SVD of the
    transfer function evaluated across the entire frequency spectrum. That can prove costly for a
    large number of evaluations, hence the iterative methods of [1] are employed.

    In the case of a SISO system the H-infinity norm corresponds to the maximum frequency gain.

    A scalar value is returned if the system is stable. If the system is unstable it returns ``np.Inf``.

    References:

        [1] Bruinsma, N. A., & Steinbuch, M. (1990). A fast algorithm to compute the H∞-norm of a transfer function
        matrix. Systems and Control Letters, 14(4), 287–293. https://doi.org/10.1016/0167-6911(90)90049-Z

    Args:
        ss (sharpy.linear.src.libss.StateSpace): Multi input multi output system.
        **kwargs: Key-word arguments.

    Keyword Args:
        tol (float (optional)): Tolerance. Defaults to ``1e-7``.
        tol_imag_eigs (float (optional)): Tolerance to find purely imaginary eigenvalues. Defaults to ``1e-7``.
        iter_max (int (optional)): Maximum number of iterations.
        print_info (bool (optional)): Print status and information. Defaults to ``False``.

    Returns:
        float: H-infinity norm of the system.
    """
    tol = kwargs.get('tol', 1e-7)
    iter_max = kwargs.get('iter_max', 10)
    print_info = kwargs.get('print_info', False)

    # tolerance to find purely imaginary eigenvalues i.e those with Re(eig) < tol_imag_eigs
    tol_imag_eigs = kwargs.get('tol_imag_eigs', 1e-7)

    if ss.dt is not None:
        ss = libss.disc2cont(ss)

    # 1) Compute eigenvalues of original system
    eigs = sclalg.eigvals(ss.A)

    if any(eigs.real > tol_imag_eigs):
        if print_info:
            try:
                cout.cout_wrap('System is unstable - H-inf = np.inf')
            except ValueError:
                print('System is unstable - H-inf = np.inf')
        return np.inf

    # 2) Find eigenvalue that maximises equation. If all real pick largest eig
    if np.max(np.abs(eigs.imag) < tol_imag_eigs):
        eig_m = np.max(eigs.real)
    else:
        eig_m, _ = max_eigs(eigs)

    # 3) Choose best option for gamma_lb
    max_steady_state = np.max(sclalg.svd(ss.transfer_function_evaluation(0), compute_uv=False))
    max_eig_m = np.max(sclalg.svd(ss.transfer_function_evaluation(1j*np.abs(eig_m)), compute_uv=False))
    max_d = np.max(sclalg.svd(ss.D, compute_uv=False))

    gamma_lb = max(max_steady_state, max_eig_m, max_d)

    iter_num = 0

    if print_info:
        try:
            cout.cout_wrap('Calculating H-inf norm\n{0:>4s} ::::: {1:^8s}'.format('Iter', 'Hinf'))
        except ValueError:
            print('Calculating H_inf norm\n{0:>4s} ::::: {1:^8s}'.format('Iter', 'Hinf'))

    while iter_num < iter_max:
        if print_info:
            try:
                cout.cout_wrap('{0:>4g} ::::: {1:>8.2e}'.format(iter_num, gamma_lb))
            except ValueError:
                print('{0:>4g} ::::: {1:>8.2e}'.format(iter_num, gamma_lb))
        gamma = (1 + 2 * tol) * gamma_lb

        # 4) compute hamiltonian and eigenvalues
        ham = hamiltonian(gamma, ss)
        eigs = sclalg.eigvals(ham)

        # If eigenvalues all eigenvalues are purely imaginary
        if any(np.abs(eigs.real) < tol_imag_eigs):
            # Select imaginary eigenvalues and those with positive values
            condition_imag = (np.abs(eigs.real) < tol_imag_eigs) * eigs.imag > 0
            imag_eigs = eigs[condition_imag].imag

            # Sort them in decreasing order
            order = np.argsort(imag_eigs)[::-1]
            imag_eigs = imag_eigs[order]

            if len(imag_eigs) == 1:
                m = imag_eigs[0]
                svdmax = np.max(sclalg.svd(ss.transfer_function_evaluation(1j*m), compute_uv=False))

                gamma_lb = svdmax
            else:
                m_list = [0.5 * (imag_eigs[i] + imag_eigs[i+1]) for i in range(len(imag_eigs) - 1)]

                svdmax = [np.max(sclalg.svd(ss.transfer_function_evaluation(1j*m), compute_uv=False)) for m in m_list]

                gamma_lb = max(svdmax)

        else:
            gamma_ub = gamma
            break

        iter_num += 1

        if iter_num == iter_max:
            raise np.linalg.LinAlgError('Unconverged H-inf solution after %g iterations' % iter_num)

    hinf = 0.5 * (gamma_lb + gamma_ub)

    return hinf


def max_eigs(eigs):
    r"""
    Returns the maximum of

    .. math:: \left|\frac{Im(\lambda_i)}{Re(\lambda_i)}\frac{1}{\lambda_i}\right|

    for a given array of eigenvalues ``eigs``.

    Used as part of the computation of the H infinity norm


    References:

        [1] Bruinsma, N. A., & Steinbuch, M. (1990). A fast algorithm to compute the H∞-norm of a transfer function
        matrix. Systems and Control Letters, 14(4), 287–293. https://doi.org/10.1016/0167-6911(90)90049-Z

    Args:
        eigs (np.ndarray): Array of eigenvalues.

    Returns:
        complex: Maximum value of function.
    """
    func = np.abs(eigs.imag / eigs.real / eigs)

    i_max = np.argmax(func)

    return func[i_max], i_max


def find_target_system(data, target_system):
    """
    Finds target system ``aeroelastic``, ``aerodynamic`` or ``structural``.

    Args:
        data (sharpy.PreSharpy): Object containing problem data
        target_system (str): Desired target system.

    Returns:
        sharpy.linear.src.libss.StateSpace: State-space object of target system
    """

    if target_system == 'aeroelastic':
        ss = data.linear.ss

    elif target_system == 'structural':
        ss = data.linear.linear_system.beam.ss

    elif target_system == 'aerodynamic':
        ss = data.linear.linear_system.uvlm.ss  # this could be a ROM

    else:
        raise NameError('Unrecognised system')

    return ss
