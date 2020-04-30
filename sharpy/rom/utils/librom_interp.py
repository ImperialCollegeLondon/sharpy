"""Methods for the interpolation of DLTI ROMs

This is library for  state-space models interpolation. These routines are intended
for small size state-space models (ROMs), hence some methods may not be optimised
to exploit sparsity structures. For generality purposes, all methods require in
input interpolatory weights.


The module includes the methods:

    - :func:`~sharpy.rom.utils.librom_interp.transfer_function`: returns an interpolatory state-space model based on the
      transfer function method [1]. This method is general and is, effectively, a
      wrapper of the :func:`sharpy.linear.src.libss.join` method.

    - :func:`~sharpy.rom.utils.librom_interp.BT_transfer_function`: evolution of transfer function methods. The growth of
      the interpolated system size is avoided through balancing.


References:

    [1] Benner, P., Gugercin, S. & Willcox, K., 2015. A Survey of Projection-Based
    Model Reduction Methods for Parametric Dynamical Systems. SIAM Review, 57(4),
    pp.483–531.


Author: S. Maraniello

Date: Mar-Apr 2019


"""

import numpy as np
import scipy.linalg as sclalg
import yaml
import warnings

# dependency
import sharpy.linear.src.libss as libss
from sharpy.rom.interpolation.interpolationspaces import TangentInterpolation


def transfer_function(SS_list, wv):
    """
    Returns an interpolatory state-space model based on the transfer function
    method [1]. This method is general and is, effectively, a wrapper of the
    :func:`sharpy.linear.src.libss.join` method.

    Features:

        - stability preserved
        - system size increases with interpolatory order, but can be optimised for
          fast on-line evaluation

    Args:
        SS_list (list): List of state-space models instances of :class:`sharpy.linear.src.libss.ss` class.
        wv (list): list of interpolatory weights.

    Notes:

        For fast online evaluation, this routine can be optimised to return a
        class that handles each state-space model independently. See ref. [1] for
        more details.

    References:
        [1] Benner, P., Gugercin, S. & Willcox, K., 2015. A Survey of Projection-Based
        Model Reduction Methods for Parametric Dynamical Systems. SIAM Review, 57(4),
        pp.483–531.
    """

    return libss.join(SS_list, wv)


def FLB_transfer_function(SS_list, wv, U_list, VT_list, hsv_list=None, M_list=None):
    r"""
    Returns an interpolatory state-space model based on the transfer function
    method [1]. This method is applicable to frequency limited balanced
    state-space models only.


    Features:

        - stability preserved
        - the interpolated state-space model has the same size than the tabulated ones
        - all state-space models, need to have the same size and the same numbers of
          hankel singular values.
        - suitable for any ROM


    Args:
        SS_list (list): List of state-space models instances of :class:`sharpy.linear.src.libss.ss` class.
        wv (list): list of interpolatory weights.
        U_list (list): small size, thin SVD factors of Gramians square roots of each state space model (:math:`\mathbf{U}`).
        VT_list (list): small size, thin SVD factors of Gramians square roots of each state space model (:math:`\mathbf{V}^\top`).
        hsv_list (list): small size, thin SVD factors of Gramians square roots of each state space model. If ``None``,
          it is assumed that
                        ``U_list = [ U_i sqrt(hsv_i) ]``
                        ``VT_list = [ sqrt(hsv_i) V_i.T ]``
          where ``U_i`` and ``V_i.T`` are square matrices and hsv is an array.

        M_list (list): for fast on-line evaluation. Small size product of Gramians
          factors of each state-space model. Each element of this list is equal to:
          ``M_i = U_i hsv_i V_i.T``

    Notes:
        Message for future generations:

            - the implementation is divided into an offline and online part.

    References:

    Maraniello S. and Palacios R., Frequency-limited balanced truncation for
    parametric reduced-order modelling of the UVLM. Only in the best theaters.

    See Also:

        Frequency-Limited Balanced ROMs may be obtained from SHARPy using :class:`sharpy.rom.balanced.FrequencyLimited`.
    """

    # ----------------------------------------------------------------- offline

    ### checks sizes
    N_interp = len(SS_list)
    states = SS_list[0].states
    inputs = SS_list[0].inputs
    outputs = SS_list[0].outputs
    for ss_here in SS_list:
        assert ss_here.states == states, \
            'State-space models must have the same number of states!'
        assert ss_here.inputs == inputs, \
            'State-space models must have the same number of states!'
        assert ss_here.outputs == outputs, \
            'State-space models must have the same number of states!'

    ### case of unbalanced state-space models
    # in this case, U_list and VT_list contain the full-rank Gramians factors
    # of each ROM
    if U_list is None and VT_list is None:
        raise NameError('apply FLB before calling this routine')
        # hsv_list = None
        # M_list, U_list, VT_list = [], [], []
        # for ii in range(N_interp):

        #     # # avoid direct
        #     # hsv,U,Vh,Zc,Zo = librom.balreal_direct_py(
        #     #                         SS_list[ii].A, SS_list[ii].B, SS_list[ii].C,
        #     #                         DLTI=True,full_outputs=True)

        #     # iterative also fails
        #     hsv,Zc,Zo = librom.balreal_iter(SS_list[ii].A, SS_list[ii].B, SS_list[ii].C,
        #                     lowrank=True,tolSmith=1e-10,tolSVD=1e-10,
        #                     kmin=None, tolAbs=False, Print=True, outFacts=True)

        #     # M_list.append( np.dot( np.dot(U,np.diag(hsv)), Vh) )
        #     M_list.append( np.dot( Zo.T,Zc ) )
        #     U_list.append(Zo.T)
        #     VT_list.append(Zc)

    # calculate small size product of Gramians factors
    elif M_list is None:
        if hsv_list is None:
            M_list = [np.dot(U, VT) for U, VT in zip(U_list, VT_list)]
        else:
            M_list = [np.dot(U * hsv, VT) for U, hsv, VT in zip(U_list, hsv_list, VT_list)]

    # ------------------------------------------------------------------ online

    ### balance interpolated model
    M_int = np.zeros_like(M_list[0])
    for ii in range(N_interp):
        M_int += wv[ii] * M_list[ii]

    U_int, hsv_int, Vh_int = sclalg.svd(M_int, full_matrices=False)
    sinv_int = hsv_int ** (-0.5)

    ### build projection matrices
    sinvUT_int = (U_int * sinv_int).T
    Vsinv_int = Vh_int.T * sinv_int

    if hsv_list is None:
        Ti_int_list = [np.dot(sinvUT_int, U) for U in U_list]
        T_int_list = [np.dot(VT, Vsinv_int) for VT in VT_list]
    else:
        Ti_int_list = [np.dot(sinvUT_int, U * np.sqrt(hsv)) \
                       for U, hsv in zip(U_list, hsv_list)]
        T_int_list = [np.dot(np.dot(np.diag(np.sqrt(hsv)), VT),
                             Vsinv_int) \
                      for hsv, VT in zip(hsv_list, VT_list)]

    ### assemble interp state-space model
    A_int = np.zeros((states, states))
    B_int = np.zeros((states, inputs))
    C_int = np.zeros((outputs, states))
    D_int = np.zeros((outputs, inputs))

    for ii in range(N_interp):
        # in A and B the weigths come from Ti
        A_int += wv[ii] * np.dot(Ti_int_list[ii],
                                 np.dot(SS_list[ii].A, T_int_list[ii]))
        B_int += wv[ii] * np.dot(Ti_int_list[ii], SS_list[ii].B)
        # in C and D the weights come from the interp system expression
        C_int += wv[ii] * np.dot(SS_list[ii].C, T_int_list[ii])
        D_int += wv[ii] * SS_list[ii].D

    return libss.ss(A_int, B_int, C_int, D_int, dt=SS_list[0].dt), hsv_int


def lagrange_interpolation(x_vec, x0, interpolation_degree=None):
    """
    Performs a lagrange interpolation over the domain ``x_vec`` at the point ``x0``.

    The ``interpolation_degree`` is an optional argument that sets the maximum degree of the lagrange polynomials
    employed. If left to ``None``, all points in ``x_vec`` are used.

    It returns the lagrange interpolation weights :math:`w_i` at each source point,
    such that the interpolation value :math:`y_0` can then be calculated as

    .. math:: y_0 = \sum_{i=0}^{N}w_i y_i.

    Args:
        x_vec (np.ndarray): Array of source points
        x0 (float): Interpolation point.
        interpolation_degree (int (optional)): Interpolation degree (optional).

    Returns:
        np.ndarray: Array of the same size as ``x_vec`` containing the weights :math:`w_i` at each of the source points.
    """

    n_points = len(x_vec)

    out = [0] * n_points

    n_lagrange_points = interpolation_degree + 1  # i.e. for a quadratic fit 3 points are needed

    if n_lagrange_points is None or n_lagrange_points > n_points:
        n_lagrange_points = n_points

    if n_lagrange_points > 15:
        warnings.warn('Caution, interpolation degree larger than 15. Method may be unstable. Be cautious of overfitting'
                      ' data.')

    if x0 > x_vec[-1] or x0 < x_vec[0]:
        warnings.warn('Use Caution: Interpolation point x0 = %f outside domain [%f, %f]. Extrapolation in progress.'
                      % (x0, x_vec[0], x_vec[-1]))

    d_half_degree = n_lagrange_points // 2  # half the interpolation degree
    rem = np.mod(n_lagrange_points, 2)  # remainder in the case of odd degrees

    # Moving window
    i_interp_point = np.searchsorted(x_vec, x0)
    if i_interp_point - d_half_degree - rem < 0:
        add_right = - (i_interp_point - d_half_degree - rem)  # points to add to the right side in case the window is exceeded
        add_left = 0
    elif i_interp_point + d_half_degree > n_points:
        add_left = - (n_points - (i_interp_point + d_half_degree))  # points to add to the left side
        add_right = 0
    else:
        add_left = 0
        add_right = 0

    if add_right > 0:
        min_i = 0
    else:
        min_i = i_interp_point - d_half_degree - rem - add_left  # index of left side of window

    if add_left > 0:
        max_i = n_points
    else:
        max_i = i_interp_point + d_half_degree + add_right  # index of right side of window

    window = range(min_i, max_i)  # window of points which will serve as the source of the lagrange polynomials

    for i in window:
        curr = []
        for j in window:
            if j != i:
                lag = (x0 - x_vec[j]) / (x_vec[i] - x_vec[j])
                curr.append(lag)
        out[i] = np.prod(curr)

    return out


def load_parameter_cases(yaml_file_name):
    """

    Args:
        yaml_file_name (str): Path to YAML file containing input of parameters.

    Returns:
        list: List of dictionaries
    """
    # TODO: input validation
    with open(yaml_file_name, 'r') as yaml_file:
        out_dict = yaml.load(yaml_file, Loader=yaml.Loader)
        yaml_file.close()
    return out_dict
