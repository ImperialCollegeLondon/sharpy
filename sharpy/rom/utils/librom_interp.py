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


def lagrange_interpolation(x_vec, x0):

    # TODO: limit the lagrange degree when there are too many points! Method could be unstable
    # Would need to have x sorted and limit the degree by a number of points left and
    # right of x0

    out = [0] * len(x_vec)

    for i in range(len(x_vec)):
        curr = []
        for j in range(len(x_vec)):
            if j != i:
                curr.append((x0 - x_vec[j]) / (x_vec[i] - x_vec[j]))
        out[i] = np.prod(curr)

    return out


def load_parameter_cases(yaml_file_name):
    """

    Args:
        yaml_file_name:

    Returns:
        list: List of dictionaries
    """
    # TODO: input validation
    return yaml.load(open(yaml_file_name, 'r'), Loader=yaml.Loader)
