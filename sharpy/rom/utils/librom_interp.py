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

import warnings
import numpy as np
import scipy.linalg as scalg

# dependency
import sharpy.linear.src.libss as libss


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
        SS_list (list): List of state-space models instances of :class:`sharpy.linear.src.libss.StateSpace` class.
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
        SS_list (list): List of state-space models instances of :class:`sharpy.linear.src.libss.StateSpace` class.
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

    U_int, hsv_int, Vh_int = scalg.svd(M_int, full_matrices=False)
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

    return libss.StateSpace(A_int, B_int, C_int, D_int, dt=SS_list[0].dt), hsv_int


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

    - SS: list of state-space models (instances of libss.StateSpace class)

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

    def __init__(self, SS, VV=None, WWT=None,
                 Vref=None, WTref=None, method_proj=None):

        self.SS = SS

        self.VV = VV
        self.WWT = WWT

        self.Vref = Vref
        self.WTref = WTref

        self.method_proj = method_proj

        self.Projected = False
        if VV is None or WWT is None:
            self.Projected = True
            self.AA = [ss_here.A for ss_here in SS]
            self.BB = [ss_here.B for ss_here in SS]
            self.CC = [ss_here.C for ss_here in SS]

        # projection required for D
        self.DD = [ss_here.D for ss_here in SS]

        ### check state-space models
        Nx, Nu, Ny = SS[0].states, SS[0].inputs, SS[0].outputs
        dt = SS[0].dt
        for ss_here in SS:
            assert ss_here.states == Nx, \
                'State-space models do not have the same number of states'
            assert ss_here.inputs == Nu, \
                'State-space models do not have the same number of inputs'
            assert ss_here.outputs == Ny, \
                'State-space models do not have the same number of outputs'
            assert ss_here.dt == dt, \
                'State-space models do not have same timestep'

    def __call__(self, wv):
        """
        Evaluate interpolated model using weights wv.
        """

        assert self.Projected, ('You must project the state-space models over' +
                                ' a common basis before interpolating')

        Aint = np.zeros_like(self.AA[0])
        Bint = np.zeros_like(self.BB[0])
        Cint = np.zeros_like(self.CC[0])
        Dint = np.zeros_like(self.DD[0])

        for ii in range(len(self.AA)):
            Aint += wv[ii] * self.AA[ii]
            Bint += wv[ii] * self.BB[ii]
            Cint += wv[ii] * self.CC[ii]
            Dint += wv[ii] * self.DD[ii]

        return libss.StateSpace(Aint, Bint, Cint, Dint, self.SS[0].dt)

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

            for ii in range(len(self.SS)):
                U, sv, Z = scalg.svd(np.dot(self.VV[ii].T, self.Vref),
                                     full_matrices=False, overwrite_a=False,
                                     lapack_driver='gesdd')
                Q = np.dot(U, Z.T)
                U, sv, Z = scalg.svd(np.dot(self.WWT[ii], self.WTref),
                                     full_matrices=False, overwrite_a=False,
                                     lapack_driver='gesdd')
                Qinv = np.dot(U, Z.T).T
                self.QQ.append(Q)
                self.QQinv.append(Qinv)

        elif self.method_proj == 'panzer':
            warnings.warn('Method untested!')

            # generate basis
            U, sv = scalg.svd(np.concatenate(self.VV, axis=1),
                              full_matrices=False, overwrite_a=False,
                              lapack_driver='gesdd')[:2]
            # chop U
            U = U[:, :self.SS[0].states]
            for ii in range(len(self.SS)):
                Qinv = np.linalg.inv(np.dot(self.WWT[ii], U))
                Q = np.linalg.inv(np.dot(self.VV[ii].T, U))
                self.QQ.append(Q)
                self.QQinv.append(Qinv)

        elif self.method_proj == 'leastsq':

            for ii in range(len(self.SS)):
                Q, _, _, _ = scalg.lstsq(self.VV[ii], self.Vref)
                print('det(Q): %.3e\tcond(Q): %.3e' \
                      % (np.linalg.det(Q), np.linalg.cond(Q)))
                # if cond(Q) is small...
                # Qinv = np.linalg.inv(Q)
                P, _, _, _ = scalg.lstsq(self.WWT[ii].T, self.WTref.T)
                self.QQ.append(Q)
                self.QQinv.append(P.T)

        elif self.method_proj == 'strongMAC':
            """
            Strong MAC enforcements as per Ref.[4]
            """

            VTVref = np.dot(self.Vref.T, self.Vref)
            for ii in range(len(self.SS)):
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

            for ii in range(len(self.SS)):
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

            for ii in range(len(self.SS)):
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

            for ii in range(len(self.SS)):
                Q, sc = scalg.orthogonal_procrustes(self.VV[ii], self.Vref)
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
            Uref, svref, Zhref = scalg.svd(self.Vref, full_matrices=False)

            for ii in range(len(self.SS)):
                # svd of basis
                Uhere, svhere, Zhhere = scalg.svd(self.VV[ii], full_matrices=False)

                R, sc = scalg.orthogonal_procrustes(Uhere, Uref)
                Q = np.dot(np.dot(Zhhere.T, np.diag(svhere ** (-1))), R)
                Qinv = np.dot(R.T, np.dot(np.diag(svhere), Zhhere))
                print('det(Q): %.3e\tcond(Q): %.3e' \
                      % (np.linalg.det(Q), np.linalg.cond(Q)))
                self.QQ.append(Q)
                self.QQinv.append(Qinv)

        else:
            raise NameError('Projection method %s not implemented!' % self.method_proj)

        ### Project
        for ii in range(len(self.SS)):
            self.AA.append(np.dot(self.QQinv[ii], np.dot(self.SS[ii].A, self.QQ[ii])))
            self.BB.append(np.dot(self.QQinv[ii], self.SS[ii].B))
            self.CC.append(np.dot(self.SS[ii].C, self.QQ[ii]))

        self.Projected = True


# ------------------------------------------------------------------------------


if __name__ == '__main__':
    import unittest


    class Test_librom_inter(unittest.TestCase):
        """ Test methods for DLTI ROM interpolation """

        def setUp(self):
            # allocate some state-space model (dense and sparse)
            dt = 0.3
            Ny, Nx, Nu = 4, 3, 2
            A = np.random.rand(Nx, Nx)
            B = np.random.rand(Nx, Nu)
            C = np.random.rand(Ny, Nx)
            D = np.random.rand(Ny, Nu)
            self.SS = libss.StateSpace(A, B, C, D, dt=dt)

# class Interp1d():
#     """
#     State-space 1D interpolation class.

#     This class allows interpolating from a list of state-space models, SS, 
#     defined over the 1D parameter space zv. 

#     State-space models are required to have the same number of inputs and outputs 
#     and need to have the same number of states.

#     For state-space interpolation, state-space models also need to be defined
#     over the same set of generalised coordinates. If this is not the case, the
#     projection matrices W and T, such that

#         A_proj = W^T A V

#     also need to be provided. This will allow projecting the state-space models
#     onto a common set of generalised coordinates before interpoling.

#     Inputs:
#     - method_interp: interpolation method as per scipy.interpolate.interp1d class

#     - method_proj: method for projection of state-space models over common
#     coordinates. Available:
#         - panzer: Panzer, J. Mohring, R. Eid, and B. Lohmann, Parametric model 
#         order reduction by matrix interpolation, at–Automatisierungstechnik, 58 
#         (2010), pp. 475–484.
#         - amsallem: D. Amsallem and C. Farhat, An online method for interpolating 
#         linear parametric reduced-order models, SIAM J. Sci. Comput., 33 (2011), 
#         pp. 2169–2198.
#     Note that 'panzel' and 'amsallem' only apply to orthogonal basis WT,V.


#     - Map: map A matrices over Riemannian manifold 

#     - IImap=if given, maps A matrices over manifold derived around A matrix 
#     of ii-th state-space in SSlist. 
#     """


#     def __init__(self, zv, SS, VV=None, WW=None, method_interp='cubic', 
#                  method_proj='panzer', Map=True, IImap=None):

#         assert IImap is not None, 'Option IImap=None not developed yet'

#         self.SS=SS
#         self.zv=zv
#         self.VV=VV
#         self.WW=WW
#         self.method_interp=method_interp
#         self.method_proj=method_proj
#         self.Map=Map
#         self.IImap=IImap

#         ### check state-space models
#         Nx,Nu,Ny = SS[0].states, SS[0].inputs, SS[0].outputs
#         for ss_here in SS:
#             assert ss_here.states == Nx,\
#                       'State-space models do not have the same number of states'
#             assert ss_here.inputs == Nu,\
#                       'State-space models do not have the same number of inputs'
#             assert ss_here.outputs == Ny,\
#                       'State-space models do not have the same number of outputs'

#         self.debug=False    # debug mode flag


#     def __call__(self, zint):
#         """
#         Evaluate at interpolation point zint. Returns a list of classes StateSpace
#         """

#         Nint=len(zint)

#         # interpolate A matrices, 
#         if self.Map is True:
#             IImap=self.IImap
#             if IImap is not None:
#                 # get inverse,
#                 AIIinv=np.linalg.inv(self.SS[IImap].A)
#                 # map,
#                 TT=[]
#                 for ii in range( len(self.zv) ):
#                     TT.append(scalg.logm( np.dot(self.SS[ii].A,AIIinv) ))
#                 # interpolate
#                 TTint=self._interp_mats(TT, zint)
#                 # and map back

#                 Aint=np.zeros( (Nint,)+AIIinv.shape )
#                 for ii in range(Nint):
#                     Aint[ii,:,:]= np.dot( scalg.expm(TTint[ii,:,:]), self.SS[IImap].A)

#             else:
#                 # get index of closest A for each element in zint and define mapping
#                 pass

#         else:
#             Aint=self._interp_mats( 
#                         [getattr(ss_here,'A') for ss_here in self.SS], zint)

#         # and B, C, D...
#         Bint=self._interp_mats( 
#                         [getattr(ss_here,'B') for ss_here in self.SS], zint)
#         Cint=self._interp_mats( 
#                         [getattr(ss_here,'C') for ss_here in self.SS], zint)        
#         Dint=self._interp_mats( 
#                         [getattr(ss_here,'D') for ss_here in self.SS], zint)

#         # and pack everything
#         SSint=[]
#         for ii in range(Nint):
#             SSint.append( StateSpace( Aint[ii,:,:], Bint[ii,:,:],
#                               Cint[ii,:,:], Dint[ii,:,:], dt=self.SS[0].dt))

#         return SSint


#     def _interp_mats(self,Mats,zint):
#         """
#         Interpolate a list of equal-size arrays, Mats, defined over zv at the 
#         points zint. The Mats are assumed to be defined onto the same set of
#         generalised coordinates.
#         """

#         # define interpolator class
#         # try:
#         IntA=scint.interp1d(self.zv,Mats,kind=self.method_interp,
#                                            copy=False,assume_sorted=True,axis=0)    

#         return IntA(zint)


#     def project(self):
#         """
#         Project the state-space models onto the generalised coordinates of 
#         state-space model IImap
#         """


#         if self.method_proj=='amsallem':

#             # get reference basis
#             Vref=self.VV[self.IImap]
#             Wref=self.WW[self.IImap]

#             for ii in range(len(self.SS)):

#                 if ii == self.IImap:
#                     continue 

#                 # get rotations
#                 U,sv,Z = scalg.svd( np.dot(self.VV[ii].T, Vref) ,
#                                     full_matrices=False,overwrite_a=False,
#                                     lapack_driver='gesdd')
#                 RotV = np.dot(U,Z.T)

#                 U,sv,Z = scalg.svd( np.dot(self.WW[ii].T, Wref) ,
#                                     full_matrices=False,overwrite_a=False,
#                                     lapack_driver='gesdd')
#                 RotW = np.dot(U,Z.T)


#                 # project state-space
#                 self.SS[ii].project(RotW.T,RotV)


#         elif self.method_proj=='panzer':

#             # generate basis
#             U,sv = scalg.svd( np.concatenate(self.VV,axis=1),
#                               full_matrices=False,overwrite_a=False,
#                               lapack_driver='gesdd')[:2]
#             # chop U
#             U=U[:,:self.SS[0].states]#*sv[:self.SS[0].states]
#             print('Panzer projection: neglecting singular values below %.2e (max: %.2e)'\
#              %(sv[self.SS[0].states],sv[0]) )


#             for ii in range(len(self.SS)):

#                 # get projection matrices
#                 M = np.linalg.inv( np.dot( self.WW[ii].T, U) )
#                 N = np.linalg.inv( np.dot( self.VV[ii].T, U) )

#                 # project
#                 self.SS[ii].project(M,N)

#         else:
#             raise NameError('Projection method %s not implemented!' %self.method_proj)
