"""Fitting Tools Library

@author: Salvatore Maraniello

@date: 15 Jan 2018
"""

import warnings
import numpy as np
from scipy.signal import tf2ss
import scipy.linalg as scalg
import scipy.optimize as scopt
import multiprocessing as mpr
import itertools


def fpoly(kv, B0, B1, B2, dyv, ddyv):
    return B0 + B1 * dyv + B2 * ddyv


def fpade(kv, Cv, B0, B1a, B1b, B2, dyv, ddyv):
    # evaluate transfer function from unsteady function Cv
    return B0 * Cv + (B1a * Cv + B1b) * dyv + B2 * ddyv


def getC(kv, Yv, B0, B1a, B1b, B2, dyv, ddyv):
    # evaluate unsteady function C required to perfectly match Yv
    Den = B0 + B1a * dyv
    # kkvec=np.abs(Den)<1e-8
    C = (Yv - B1b * dyv - B2 * ddyv) / Den
    # C[kkvec]=1.0
    # if np.sum(kkvec)>0: embed()
    return C


def rfa(cnum, cden, kv, ds=None):
    """
    Evaluates over the frequency range kv.the rational function approximation:
    [cnum[-1] + cnum[-2] z + ... + cnum[0] z**Nnum ]/...
                                [cden[-1] + cden[-2] z + ... + cden[0] z**Nden]
    where the numerator and denominator polynomial orders, Nnum and Nden, are
    the length of the cnum and cden arrays and:
        - z=exp(1.j*kv*ds), with ds sampling time if ds is given (discrete-time
        system)
        - z=1.*kv, if ds is None (continuous time system)
    """

    if ds == None:
        # continuous-time LTI system
        zv = 1.j * kv
    else:
        # discrete-time LTI system
        zv = np.exp(1.j * kv * ds)

    return np.polyval(cnum, zv) / np.polyval(cden, zv)


def rfader(cnum, cden, kv, m=1, ds=None):
    """
    Evaluates over the frequency range kv.the derivative of order m of the
    rational function approximation:
    [cnum[-1] + cnum[-2] z + ... + cnum[0] z**Nnum ]/...
                                [cden[-1] + cden[-2] z + ... + cden[0] z**Nden]
    where the numerator and denominator polynomial orders, Nnum and Nden, are
    the length of the cnum and cden arrays and:
        - z=exp(1.j*kv*ds), with ds sampling time if ds is given (discrete-time
        system)
        - z=1.*kv, if ds is None (continuous time system)
    """

    if ds == None:
        # continuous-time LTI system
        zv = 1.j * kv
        dzv = 1.j
        raise NameError('Never tested for continuous systems!')
    else:
        # discrete-time LTI system
        zv = np.exp(1.j * kv * ds)
        dzv = 1.j * ds * zv

    Nv = np.polyval(cnum, zv)
    Dv = np.polyval(cden, zv)
    dNv = np.polyval(np.polyder(cnum), zv)
    dDv = np.polyval(np.polyder(cden), zv)

    return dzv * (dNv * Dv - Nv * dDv) / Dv ** 2


def fitfrd(kv, yv, N, dt=None, mag=0, eng=None):
    """
    Wrapper for fitfrd (mag=0) and fitfrdmag (mag=1) functions in continuous and
    discrete time (if ds in input).
    Input:
       kv,yv: frequency array and frequency response
       N: order for rational function approximation
       mag=1,0: Flag for determining method to use
       dt (optional): sampling time for DLTI systems
    """

    raise NameError('Please use fitfrd function in matwrapper module!')

    return None


def get_rfa_res(xv, kv, Yv, Nnum, Nden, ds=None):
    """
    Returns magnitude of the residual Yfit-Yv of a RFA approximation at each
    point kv. The coefficients of the approximations are:
    - cnum=xv[:Nnum]
    - cdem=xv[Nnum:]
    where cnum and cden are as per the 'rfa' function.
    """

    assert Nnum + Nden == len(xv), 'Nnum+Nden must be equal to len(xv)!'
    cnum = xv[:Nnum]
    cden = xv[Nnum:]

    Yfit = rfa(cnum, cden, kv, ds)

    return np.abs(Yfit - Yv)


def get_rfa_res_norm(xv, kv, Yv, Nnum, Nden, ds=None, method='mix'):
    """
    Define residual scalar norm of Pade approximation of coefficients
    cnum=xv[:Nnum] and cden[Nnum:] (see get_rfa_res and rfa function) and
    time-step ds (if discrete time).
    """

    ErvAbs = get_rfa_res(xv, kv, Yv, Nnum, Nden, ds)

    if method == 'H2':
        res = np.sum(ErvAbs ** 2)
    elif method == 'Hinf':
        res = np.max(ErvAbs)
    elif method == 'mix':
        res = np.sum(ErvAbs ** 2) + np.max(ErvAbs)

    return res


def rfa_fit_dev(kv, Yv, Nnum, Nden, TolAbs, ds=None, Stability=True,
                NtrialMax=10, Cfbound=1e2, OutFull=False, Print=False):
    """
    Find best fitting RFA approximation from frequency response Yv over the
    frequency range kv for both continuous (ds=None) and discrete (ds>0) LTI
    systems.

    The RFA approximation is found through a 2-stage strategy:
        a. an evolutionary algoryhtm is run to determine the optimal fitting
        coefficients
        b. the search is refined through a least squares algorithm.
    and is stopped as soon as:
        1. the maximum absolute error in frequency response of the RFA falls
        below ``TolAbs``
        2. the maximum number of iterations is reached.


    Input:
    - kv: frequency range for approximation
    - Yv: frequency response vector over kv
    - TolAbs: maximum admissible absolute difference in frequency response
    between RFA and original system.
    - Nnum,Ndem: number of coefficients for Pade approximation.
    - ds: sampling time for DLTI systems
    - NtrialMax: maximum number of repetition of global and least square optimisations
    - Cfbouds: maximum absolute values of coeffieicnts (only for evolutionary
    algorithm)
    - OutFull: if False, only outputs optimal coefficients of RFA. Otherwise,
     outputs cost and RFA coefficients of each trial.

    Output:
    - cnopt: optimal coefficients (numerator)
    - cdopt: optimal coefficients (denominator)

    Important:
    - this function has the same objective as fitfrd in matwrapper module. While
    generally slower, the global optimisation approach allows to verify the
    results from fitfrd.
    """

    def pen_max_eig(cdvec):
        """
        Computes maximum eigenvalues from denominator coefficients of RFA and
        evaluates a log barrier that ensures
        """
        eigs = np.roots(cdvec)
        eigmax = np.max(np.abs(eigs))

        eiglim = 0.999
        pen = 0.
        Fact = 1e3 * TolAbs / np.log(2)
        if eigmax > eiglim:
            pen = Fact * np.log((eigmax + 1. - 2. * eiglim) / (1. - eiglim))
        else:
            pen = 0.
        return pen

    if Stability:
        def fcost_dev(xv, kv, Yv, Nnum, Nden, ds, method):
            """
            xv is a vector such that the coeff:
            cnum=xv[:Nnum]
            cden=xv[Nnum:]
            """
            xvpass = np.concatenate((xv, np.array([1, ])))
            error_fit = get_rfa_res_norm(xvpass, kv, Yv, Nnum, Nden, ds, method)
            pen_stab = pen_max_eig(xvpass[Nnum:])
            return error_fit + pen_stab

        def fcost_lsq(xv, kv, Yv, Nnum, Nden, ds):

            xvpass = np.concatenate((xv, np.array([1, ])))
            res_fit = get_rfa_res(xvpass, kv, Yv, Nnum, Nden, ds)
            pen_stab = pen_max_eig(xvpass[Nnum:])
            return res_fit + pen_stab

    else:
        def fcost_dev(xv, kv, Yv, Nnum, Nden, ds, method):
            """
            xv is a vector such that the coeff:
            cnum=xv[:Nnum]
            cden=xv[Nnum:]
            """
            xvpass = np.concatenate((xv, np.array([1, ])))
            return get_rfa_res_norm(xvpass, kv, Yv, Nnum, Nden, ds, method)

        def fcost_lsq(xv, kv, Yv, Nnum, Nden, ds):
            xvpass = np.concatenate((xv, np.array([1, ])))
            return get_rfa_res(xvpass, kv, Yv, Nnum, Nden, ds)

    Nx = Nnum + Nden - 1

    # List of optimal solutions found by the evolutionary and least squares
    # algorithms
    XvOptDev, XvOptLsq = [], []
    Cdev, Clsq = [], []

    tt = 0
    cost_best = 1e32

    while cost_best > TolAbs and tt < NtrialMax:
        tt += 1

        ###  Evolutionary algorithm
        res = scopt.differential_evolution(  # popsize=100,
            strategy='best1bin',
            func=fcost_dev,
            args=(kv, Yv, Nnum, Nden, ds, 'Hinf'),
            bounds=Nx * ((-Cfbound, Cfbound),))
        xvdev = res.x
        cost_dev = fcost_dev(xvdev, kv, Yv, Nnum, Nden, ds, 'Hinf')

        # is this the best solution?
        if cost_dev < cost_best:
            cost_best = cost_dev
            xvopt = xvdev

        # store this solution
        if OutFull:
            XvOptDev.append(xvdev)
            Cdev.append(cost_dev)

        ### Least squares fitting - unbounded
        #  method only local, but do not move to the end of global search: best
        # results can be found even when starting from a "worse" solution
        xvlsq = scopt.leastsq(fcost_lsq, x0=xvdev, args=(kv, Yv, Nnum, Nden, ds))[0]
        cost_lsq = fcost_dev(xvlsq, kv, Yv, Nnum, Nden, ds, 'Hinf')

        # is this the best solution?
        if cost_lsq < cost_best:
            cost_best = cost_lsq
            xvopt = xvlsq

        # store this solution
        if OutFull:
            XvOptLsq.append(xvlsq)
            Clsq.append(cost_lsq)

        ### Print and move on
        if Print:
            print('Trial %.2d: cost dev: %.3e, cost lsq: %.3e' \
                  % (tt, cost_dev, cost_lsq))

    if cost_best > TolAbs:
        warnings.warn(
            'RFA error (%.2e) greater than specified tolerance (%.2e)!' \
            % (cost_best, TolAbs))

    ### add 1 to denominator
    cnopt = xvopt[:Nnum]
    cdopt = np.hstack([xvopt[Nnum:], 1.])
    # if np.abs(cdopt[-1])>1e-2:
    # 	cdscale=cdopt[-1]
    # else:
    # 	cdscale=1.0
    # cnopt=cnopt/cdscale
    # cdopt=cdopt/cdscale

    # determine outputs
    Outputs = (cnopt, cdopt)
    if OutFull:
        for tt in range(Ntrial):
            if np.abs(XvOptDev[tt][-1]) > 1e-2:
                XvOptDev[tt] = XvOptDev[tt] / XvOptDev[tt][-1]
            if np.abs(XvOptLsq[tt][-1]) > 1e-2:
                XvOptLsq[tt] = XvOptLsq[tt] / XvOptLsq[tt][-1]
        Outputs = Outputs + (XvOptDev, XvOptLsq, Cdev, Clsq)

    return Outputs


def poly_fit(kv, Yv, dyv, ddyv, method='leastsq', Bup=None):
    """
    Find best II order fitting polynomial from frequency response Yv over the
    frequency range kv for both continuous (ds=None) and discrete (ds>0) LTI
    systems.

    Input:
    - kv: frequency points
    - Yv: frequency response
    - dyv,ddyv: frequency responses of I and II order derivatives
    - method='leastsq','dev': algorithm for minimisation
    - Bup (only 'dev' method): bounds for bv coefficients as per
    scipy.optimize.differential_evolution. This is a length 3 array.

    Important:
    - this function attributes equal weight to each data-point!
    """

    if method == 'leastsq':
        # pointwise residual
        def funRes(bv, kv, Yv, dyv, ddyv):
            B0, B1, B2 = bv
            rv = fpoly(kv, B0, B1, B2, dyv, ddyv) - Yv
            return np.concatenate((rv.real, rv.imag))

        # solve
        bvopt, cost = scopt.leastsq(funRes, x0=[0., 0., 0.], args=(kv, Yv, dyv, ddyv))


    elif method == 'dev':
        # use genetic algorithm with objective a sum of H2 and Hinf norms of
        # residual
        def funRes(bv, kv, Yv, dyv, ddyv):

            B0, B1, B2 = bv
            rv = fpoly(kv, B0, B1, B2, dyv, ddyv) - Yv

            Nk = len(kv)
            rvsq = rv * rv.conj()
            # H2norm=np.sqrt(np.trapz(rvsq/(Nk-1.)))
            # return H2norm+np.linalg.norm(rv,np.inf)
            return np.sum(rvsq)

        # prepare bounds
        if Bup is None:
            Bounds = 3 * ((-Bup, Bup),)
        else:
            assert len(Bup) == 3, 'Bup must be a length 3 list/array'
            Bounds = ((-Bup[0], Bup[0]), (-Bup[1], Bup[1]), (-Bup[2], Bup[2]),)

        res = scopt.differential_evolution(
            func=funRes, args=(kv, Yv, dyv, ddyv), strategy='best1bin', bounds=Bounds)
        bvopt = res.x
        cost = funRes(bvopt, kv, Yv, dyv, ddyv)

    return bvopt, cost


def rfa_mimo(Yfull, kv, ds, tolAbs, Nnum, Nden, Dmatrix=None, NtrialMax=6, Ncpu=4, method='independent'):
    """
    Given the frequency response of a MIMO DLTI system, this function returns
    the A,B,C,D matrices associated to the rational function approximation of
    the original system.

    Input:
    - Yfull: frequency response (as per libss.freqresp) of full size system over
    the frequencies kv.
    - kv: array of frequencies over which the RFA approximation is evaluated.
    - tolAbs: absolute tolerance for the rfa fitting
    - Nnum: number of numerator coefficients for RFA
    - Nden: number of denominator coefficients for RFA
    - NtrialMax: maximum number of attempts
    - method=['intependent']. Method used to produce the system:
        - intependent: each input-output combination is treated separately. The
        resulting system is a collection of independent SISO DLTIs

    """

    Nout, Nin, Nk = Yfull.shape
    assert Nk == len(kv), 'Frequency response Yfull not compatible with frequency range kv'

    iivec = range(Nin)
    oovec = range(Nout)
    plist = list(itertools.product(oovec, iivec))

    args_const = (Nnum, Nden, tolAbs, ds, True, NtrialMax, 1e2, False, False)

    with mpr.Pool(Ncpu) as pool:

        if Dmatrix is None:
            P = [pool.apply_async(rfa_fit_dev,
                                  args=(kv, Yfull[oo, ii, :],) + args_const) for oo, ii in plist]
        else:
            P = [pool.apply_async(rfa_fit_dev,
                                  args=(kv, Yfull[oo, ii, :] - Dmatrix[oo, ii],) + args_const) for oo, ii in plist]
        R = [pp.get() for pp in P]

    Asub = []
    Bsub = []
    Csub = []
    Dsub = []
    for oo in range(Nout):
        Ainner = []
        Binner = []
        Cinner = []
        Dinner = []
        for ii in range(Nin):
            # get coeffs
            pp = Nin * oo + ii
            cnopt, cdopt = R[pp]
            # assert plist[pp]==(oo,ii), 'Something wrong in loop'

            A, B, C, D = tf2ss(cnopt, cdopt)
            Ainner.append(A)
            Binner.append(B)
            Cinner.append(C)
            Dinner.append(D)

        # build s-s for each output
        Asub.append(scalg.block_diag(*Ainner))
        Bsub.append(scalg.block_diag(*Binner))
        Csub.append(np.block(Cinner))
        Dsub.append(np.block(Dinner))

    Arfa = scalg.block_diag(*Asub)
    Brfa = np.vstack(Bsub)
    Crfa = scalg.block_diag(*Csub)

    if Dmatrix is not None:
        Drfa = np.vstack(Dsub) + Dmatrix
    else:
        Drfa = np.vstack(Dsub)

    return Arfa, Brfa, Crfa, Drfa


# if __name__ == '__main__':
#     import libss
#     import matplotlib.pyplot as plt
#
#     ### common params
#     ds = 2. / 40.
#     fs = 1. / ds
#     fn = fs / 2.
#     kn = 2. * np.pi * fn
#     kv = np.linspace(0, kn, 301)
#
#     # build a state-space
#     cfnum = np.array([4, 1.25, 1.5])
#     cfden = np.array([2, .5, 1])
#     A, B, C, D = tf2ss(cfnum, cfden)
#     SS = libss.StateSpace(A, B, C, D, dt=ds)
#     Cvref = libss.freqresp(SS, kv)
#     Cvref = Cvref[0, 0, :]
#
#     # Find fitting
#     Nnum, Nden = 3, 3
#     cnopt, cdopt = rfa_fit_dev(kv, Cvref, Nnum, Nden, ds, True, 3, Cfbound=1e3,
#                                OutFull=False, Print=True)
#
#     print('Error coefficients (DLTI):')
#     print('Numerator:   ' + 3 * '%.2e  ' % tuple(np.abs(cnopt - cfnum)))
#     print('Denominator: ' + 3 * '%.2e  ' % tuple(np.abs(cnopt - cfnum)))
#
#     # Visualise
#     Cfit = rfa(cnopt, cdopt, kv, ds)
#
#     fig = plt.figure('Transfer function', (10, 4))
#     ax1 = fig.add_subplot(111)
#     ax1.plot(kv, Cvref.real, color='r', lw=2, ls='-', label=r'ref - real')
#     ax1.plot(kv, Cvref.imag, color='r', lw=2, ls='--', label=r'ref - imag')
#     ax1.plot(kv, Cfit.real, color='k', lw=1, ls='-', label=r'RFA - real')
#     ax1.plot(kv, Cfit.imag, color='k', lw=1, ls='--', label=r'RFA - imag')
#     ax1.legend(ncol=1, frameon=True, columnspacing=.5, labelspacing=.4)
#     ax1.grid(color='0.85', linestyle='-')
#     plt.show()
