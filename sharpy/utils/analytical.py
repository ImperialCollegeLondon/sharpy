"""Analytical Functions

Analytical solutions for 2D aerofoil based on thin plates theory

Author: Salvatore Maraniello

Date: 23 May 2017

References:

1. Simpson, R.J.S., Palacios, R. & Murua, J., 2013. Induced-Drag Calculations in the Unsteady Vortex Lattice Method.
   AIAA Journal, 51(7), pp.1775–1779.

2. Gulcat, U., 2009. Propulsive Force of a Flexible Flapping Thin Airfoil. Journal of Aircraft, 46(2), pp.465–473.

"""

import numpy as np
import scipy.special as scsp

# imaginary variable
j = 1.0j


def theo_fun(k):
    r"""Returns the value of Theodorsen's function at a reduced frequency :math:`k`.

    .. math:: \mathcal{C}(jk) = \frac{H_1^{(2)}(k)}{H_1^{(2)}(k) + jH_0^{(2)}(k)}

    where :math:`H_0^{(2)}(k)` and :math:`H_1^{(2)}(k)` are Hankel functions of the second kind.

    Args:
        k (np.array): Reduced frequency/frequencies at which to evaluate the function.

    Returns:
        np.array: Value of Theodorsen's function evaluated at the desired reduced frequencies.

    """

    H1 = scsp.hankel2(1, k)
    H0 = scsp.hankel2(0, k)

    C = H1 / (H1 + j * H0)

    return C


def qs_derivs(x_ea_perc, x_fh_perc):
    """
    Provides quasi-steady aerodynamic lift and moment coefficients derivatives
    Ref. Palacios and Cesnik, Chap 3.

    Args:
        x_ea_perc: position of axis of rotation in percentage of chord (measured
          from LE)
        x_fc_perc: position of flap axis of rotation in percentage of chord
          (measured from LE)
    """

    # parameters
    nu_ea = 2.0 * x_ea_perc - 1.0
    nu_fh = 2.0 * x_fh_perc - 1.0
    th = np.arccos(-nu_fh)

    # pitch/pitch rate related quantities
    CLa = 2. * np.pi  # ok
    CLda = np.pi * (1. - 2. * nu_ea)
    CMda = -0.25 * np.pi

    # flap related quantities
    CLb = 2. * (np.pi - th + np.sin(th))
    CLdb = (0.5 - nu_fh) * 2. * (np.pi - th) + (2. - nu_fh) * np.sin(th)
    CMb = -0.5 * (1 + nu_fh) * np.sin(th)
    CMdb = -.25 * (np.pi - th + 2. / 3. * np.sin(th) * (0.5 - nu_fh) * (2. + nu_fh))

    return CLa, CLda, CLb, CLdb, CMda, CMb, CMdb


def nc_derivs(x_ea_perc, x_fh_perc):
    """
    Provides non-circulatory aerodynamic lift and moment coefficients derivatives
    Ref. Palacios and Cesnik, Chap 3.

    Args:
        x_ea_perc: position of axis of rotation in percentage of chord (measured
          from LE)
        x_fc_perc: position of flap axis of rotation in percentage of chord
          (measured from LE)
    """

    # parameters
    nu_ea = 2.0 * x_ea_perc - 1.0
    nu_fh = 2.0 * x_fh_perc - 1.0
    th = np.arccos(-nu_fh)

    # pitch/pitch rate related quantities
    CLda = np.pi  # ok
    CLdda = -np.pi * nu_ea
    CMda = -.25 * np.pi
    CMdda = -0.25 * np.pi * (0.25 - nu_ea)

    # flap related quantities
    CLdb = np.pi - th - nu_fh * np.sin(th)
    CLddb = -nu_fh * (np.pi - th) + 1. / 3. * (2. + nu_fh ** 2) * np.sin(th)
    CMdb = -0.25 * (np.pi - th + (2. / 3. - nu_fh - 2. / 3. * nu_fh ** 2) * np.sin(th))
    CMddb = -0.25 * ((0.25 - nu_fh) * (np.pi - th) + \
                     (2. / 3. - 5. / 12. * nu_fh + nu_fh ** 2 / 3. + nu_fh ** 3 / 6.) * np.sin(th))

    return CLda, CLdda, CLdb, CLddb, CMda, CMdda, CMdb, CMddb


def theo_CL_freq_resp(k, x_ea_perc, x_fh_perc):
    """
    Frequency response of lift coefficient according Theodorsen's theory.

    The output is a 3 elements array containing the CL frequency response w.r.t.
    to pitch, plunge and flap motion, respectively. Sign conventions are as
    follows:
    
        * plunge: positive when moving upward

        * x_ea_perc: position of axis of rotation in percentage of chord (measured
          from LE)
    
        * x_fc_perc: position of flap axis of rotation in percentage of chord
          (measured from LE)

    Warning:
        this function uses different input/output w.r.t. theo_lift
    """

    df, ddf = j * k, -k ** 2

    # get quasi-steady derivatives
    CLa_qs, CLda_qs, CLb_qs, CLdb_qs, void, void, void = qs_derivs(x_ea_perc, x_fh_perc)

    # quasi-steady lift
    CLqs = np.array([
        CLa_qs + CLda_qs * df,
        CLa_qs * df,
        CLb_qs + CLdb_qs * df,
    ])

    # get non-circulatory derivatives
    CLda_nc, CLdda_nc, CLdb_nc, CLddb_nc, void, void, void, void \
        = nc_derivs(x_ea_perc, x_fh_perc)

    # unsteady lift
    df, ddf = j * k, -k ** 2
    CLun = np.array([
        CLda_nc * df + CLdda_nc * ddf,
        CLda_nc * ddf,
        CLdb_nc * df + CLddb_nc * ddf
    ])

    ### Total response
    Y = theo_fun(k) * CLqs + CLun

    # sign convention update
    Y[1] = -Y[1]  # plunge dof positive upward

    return Y


def theo_CM_freq_resp(k, x_ea_perc, x_fh_perc):
    """
    Frequency response of moment coefficient according Theodorsen's theory.

    The output is a 3 elements array containing the CL frequency response w.r.t.
    to pitch, plunge and flap motion, respectively.
    """

    df, ddf = j * k, -k ** 2

    # get quasi-steady derivatives
    void, void, void, void, CMda_qs, CMb_qs, CMdb_qs = qs_derivs(x_ea_perc, x_fh_perc)

    # quasi-steady lift
    CMqs = np.array([
        CMda_qs * df,
        0.0 * k,
        CMb_qs + CMdb_qs * df,
    ])

    # get non-circulatory coefficients
    void, void, void, void, CMda_nc, CMdda_nc, CMdb_nc, CMddb_nc = \
        nc_derivs(x_ea_perc, x_fh_perc)

    # unsteady lift
    CMun = np.array([
        CMda_nc * df + CMdda_nc * ddf,
        CMda_nc * ddf,
        CMdb_nc * df + CMddb_nc * ddf
    ])

    ### Total response
    Y = CMqs + CMun

    # sign convention update
    Y[1] = -Y[1]

    return Y


def theo_lift(w, A, H, c, rhoinf, uinf, x12):
    r"""
    Theodorsen's solution for lift of aerofoil undergoing sinusoidal motion.

    Time histories are built assuming:

        * ``a(t)=+/- A cos(w t) ??? not verified``

        * :math:`h(t)=-H\cos(w t)`

    Args:
        w: frequency (rad/sec) of oscillation
        A: amplitude of angle of attack change
        H: amplitude of plunge motion
        c: aerofoil chord
        rhoinf: flow density
        uinf: flow speed
        x12: distance of elastic axis from mid-point of aerofoil (positive if
            the elastic axis is ahead)

    """

    # reduced frequency
    k = 0.5 * w * c / uinf

    # compute theodorsen's function
    Ctheo = theo_fun(k)

    # Lift: circulatory
    Lcirc = np.pi * rhoinf * uinf * c * Ctheo * ((uinf + w * j * (0.25 * c + x12)) * A + w * H * j)
    Lmass = 0.25 * np.pi * rhoinf * c ** 2 * ((j * w * uinf - x12 * w ** 2) * A - H * w ** 2)
    Ltot = Lcirc + Lmass

    return Ltot, Lcirc, Lmass


def garrick_drag_plunge(w, H, c, rhoinf, uinf, time):
    r"""
    Returns Garrick solution for drag coefficient at a specific time.
    Ref.[1], eq.(8) (see also eq.(1) and (2)) or Ref[2], eq.(2)

    The aerofoil vertical motion is assumed to be:

    .. math:: h(t)=-H\cos(wt)


    The :math:`C_d` is such that:

        * :math:`C_d>0`: drag

        * :math:`C_d<0`: suction
    """

    b = 0.5 * c
    k = b * w / uinf
    Hast = H / b
    s = uinf * time / b

    # compute theodorsen's function
    Ctheo = theo_fun(k)

    Cd = -2. * np.pi * k ** 2 * Hast ** 2 * (
            Ctheo.imag * np.cos(k * s) + Ctheo.real * np.sin(k * s)) ** 2

    return Cd


def garrick_drag_pitch(w, A, c, rhoinf, uinf, x12, time):
    r"""
    Returns Garrick solution for drag coefficient at a specific time.
    Ref.[1], eq.(9), (10) and (11)

    The aerofoil pitching motion is assumed to be:

        .. math:: a(t)=A\sin(\omegat)=A\sin(ks)

    The :math:`C_d` is such that:

        * :math:`C_d>0`: drag

        * :math:`C_d<0`: suction
    """

    x12 = x12 / c
    b = 0.5 * c
    k = b * w / uinf
    s = uinf * time / b

    # compute theodorsen's function
    Ctheo = theo_fun(k)
    F, G = Ctheo.real, Ctheo.imag
    sks, cks = np.sin(k * s), np.cos(k * s)

    # angle of attack
    a = A * sks

    # lift term
    Cl = np.pi * A * (k * cks
                      + x12 * k ** 2 * sks
                      + 2. * F * (sks + (0.5 - x12) * k * cks)
                      + 2. * G * (cks - (0.5 - x12) * k * sks))

    # suction force
    Y1 = 2. * (F - k * G * (0.5 - x12))
    Y2 = 2. * (G - k * F * (0.5 - x12)) - k
    Cs = 0.5 * np.pi * A ** 2 * (Y1 * sks + Y2 * cks) ** 2

    Cd = a * Cl - Cs

    return Cd


def sears_fun(kg):
    """
    Produces Sears function
    """

    S12 = 2. / np.pi / kg / (scsp.hankel1(0, kg) + 1.j * scsp.hankel1(1, kg))
    S = np.exp(-1.j * kg) * S12.conj()

    return S


def sears_lift_sin_gust(w0, L, Uinf, chord, tv):
    """
    Returns the lift coefficient for a sinusoidal gust (see set_gust.sin) as
    the imaginary part of the CL complex function defined below. The input gust
    must be the imaginary part of

    .. math::    wgust = w0*\exp(1.0j*C*(Ux*S.time[tt] - xcoord) )

    with:

    .. math:: C=2\pi/L

    and ``xcoord=0`` at the aerofoil half-chord.
    """

    # reduced frequency
    kg = np.pi * chord / L
    # Theo's funciton
    Ctheo = theo_fun(kg)
    # Sear's function
    J0, J1 = scsp.j0(kg), scsp.j1(kg)
    S = (J0 - 1.0j * J1) * Ctheo + 1.0j * J1

    phase = np.angle(S)
    CL = 2. * np.pi * w0 / Uinf * np.abs(S) * np.sin(2. * np.pi * Uinf / L * tv + phase)

    return CL


def sears_CL_freq_resp(k):
    """
    Frequency response of lift coefficient according Sear's solution.
    Ref. Palacios and Cesnik, Chap.3
    """

    # hanckel functions
    H1 = scsp.hankel1(1, k)
    H0 = scsp.hankel1(0, k)

    # Sear's function
    S12star = 2. / (np.pi * k * (H0 + 1.j * H1))
    S0 = np.exp(-1.0j * k) * S12star.conj(S12star)

    # CL frequency response
    CL = 2. * np.pi * S0

    return CL


def wagner_imp_start(aeff, Uinf, chord, tv):
    """
    Lift coefficient resulting from impulsive start solution.
    """

    sv = 2.0 * Uinf / chord * tv
    fiv = 1.0 - 0.165 * np.exp(-0.0455 * sv) - 0.335 * np.exp(-0.3 * sv)
    CLv = 2. * np.pi * aeff * fiv

    return CLv


def flat_plate_analytical(kv, x_ea_perc, x_fh_perc, input_seq, output_seq,
                          output_scal=None, plunge_deriv=True):
    r"""
    Computes the analytical frequency response of a plat plate for the input
    output sequences in ``input_seq`` and ``output_seq`` over the frequency points ``kv``,
    if available.

    The output complex values array ``Yan`` has shape ``(Nout, Nin, Nk)``; if an analytical
    solution is not available, the response is assumed to be zero.

    If ``plunge_deriv`` is ``True``, the plunge response is expressed in terms of first
    derivative dh.

    Args:
        kv (np.array): Frequency range of length ``Nk``.
        x_ea_perc (float): Elastic axis location along the chord as chord length percentage.
        x_fh_perc (float): Flap hinge location along the chord as chord length percentage.
        input_seq (list(str)): List of ``Nin`` number of inputs.
            Supported inputs include:
                * ``gust_sears``: Response to a continuous sinusoidal gust.
                * ``pitch``: Response to an oscillatory pitching motion.
                * ``plunge``: Response to an oscillatory plunging motion.
        output_seq (list(str)): List of ``Nout`` number of outputs.
            Supported outputs include:
                * ``Fy``: Vertical force.
                * ``Mz``: Pitching moment.
        output_scal (np.array): Array of factors by which to divide the desired outputs. Dimensions of ``Nout``.
        plunge_deriv (bool): If ``True`` expresses the plunge response in terms of the first derivative, i.e. the
        rate of change of plunge :math:`d\dot{h}`.

    Returns:
        np.array: A ``(Nout, Nin, Nk)`` array containing the scaled frequency response for the inputs and outputs
        specified.

    See Also:
        The lift coefficient due to pitch and plunging motions is calculated
        using :func:`sharpy.utils.analytical.theo_CL_freq_resp`. In turn, the pitching moment is found using
        :func:`sharpy.utils.analytical.theo_CM_freq_resp`.

        The response to the continuous sinusoidal gust is calculated using
        :func:`sharpy.utils.analytical.sears_CL_freq_resp`.

    """

    Nout = len(output_seq)
    Nin = len(input_seq)
    Nk = len(kv)
    Yfreq_an = np.zeros((Nout, Nin, Nk), dtype=np.complex)

    # Get Theodorsen solutions
    CLtheo = theo_CL_freq_resp(kv, x_ea_perc, x_fh_perc)
    CMtheo = theo_CM_freq_resp(kv, x_ea_perc, x_fh_perc)

    # scaling
    if output_scal is None: output_scal = np.ones((Nout,))

    for oo in range(Nout):
        for ii in range(Nin):

            ### Sears
            if input_seq[ii] == 'gust_sears':
                # Fx,Mz null
                if output_seq[oo] == 'Fy':
                    Yfreq_an[oo, ii, :] = sears_CL_freq_resp(kv)

            ### Theodorsen
            if input_seq[ii] == 'pitch':
                if output_seq[oo] == 'Fy':
                    Yfreq_an[oo, ii, :] = CLtheo[0]
                if output_seq[oo] == 'Mz':
                    Yfreq_an[oo, ii, :] = CMtheo[0]

            if input_seq[ii] == 'plunge':
                Fact = 1.0
                if plunge_deriv: Fact = -1.j / kv
                if output_seq[oo] == 'Fy':
                    Yfreq_an[oo, ii, :] = Fact * CLtheo[1]
                if output_seq[oo] == 'Mz':
                    Yfreq_an[oo, ii, :] = Fact * CMtheo[1]

        # scale output
        Yfreq_an[oo, :, :] = Yfreq_an[oo, :, :] / output_scal[oo]

    return Yfreq_an


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#
#     kv = np.linspace(0.001, 50, 1001)
#
#     ### test Sear's function
#     sv = sears_fun(kv)
#     plt.plot(kv, sv.real, '-')
#     plt.plot(kv, sv.imag, '--')
#     plt.show()
#
#     CL = theo_CL_freq_resp(kv, 0.25, 0.8)
#
#     kv = np.linspace(0.001, 2.0, 101)
#     # data for dimensional analysis
#     b = 1.3
#     U = 15.0
#     H = 0.3
#     A = 5.0 * np.pi / 180.
#     rho = 1.225
#     tref = b / U
#
#     ### plunge frequency response
#     Ltheo = -theo_lift(kv / tref, 0, H, 2. * b, rho, U, 0.0)[0]
#     Fref = b * rho * U ** 2
#     CLfreq = Ltheo / Fref / (H / b)
#
#     Yfreq = theo_CL_freq_resp(kv, x_ea_perc=1.0, x_fh_perc=0.9)
#     CLfreq02 = Yfreq[1]
#     Er = np.max(np.abs(CLfreq - CLfreq02))
#     print('Max error for CL plunge freq response: %.2e' % Er)
#
#     # moments
#     Yfreq = theo_CM_freq_resp(kv, x_ea_perc=1.0, x_fh_perc=0.9)
#     CMfreq = Yfreq[1]  # plunge motion
#     CMfreq_vel = 1.j / kv * CMfreq
#     CMmag_vel, CMph_vel = np.abs(CMfreq_vel), np.angle(CMfreq_vel, deg=True)
#
#     fig = plt.figure('Momentum coefficient frequency response')
#     ax = fig.add_subplot(121)
#     ax.plot(kv, CMmag_vel, 'k', label='magnitude')
#     ax.legend()
#     ax = fig.add_subplot(122)
#     ax.plot(kv, CMph_vel, 'r', label='phase')
#     ax.legend()
#     plt.show()
#
#     ### pitching frequency response
#     Yfreq = theo_CL_freq_resp(kv, x_ea_perc=.5, x_fh_perc=0.9)
#     CLfreq02 = Yfreq[0]
#
#     ### geometry
#     c = 3.  # m
#     b = 0.5 * c
#
#     ### motion
#     ktarget = 1.
#     H = 0.02 * b  # m Ref.[1]
#     A = 1. * np.pi / 180.  # rad - Ref.[1]
#     x12 = -0.5 * c
#     f0 = 5.  # Hz
#     w0 = 2. * np.pi * f0  # rad/s
#
#     uinf = b * w0 / ktarget
#     rhoinf = 1.225  # kg/m3
#     qinf = 0.5 * c * rhoinf * uinf ** 2
#     # C=theo_fun(k=ktarget)
#     # L=theo_lift(w0,A,H,c,rhoinf,uinf,x12)
#
#     ##### Plunge Induced drag
#     Ncicles = 5
#     tv = np.linspace(0., 2. * np.pi * Ncicles / w0, 200 * Ncicles + 1)
#     Cdv = garrick_drag_plunge(w0, H, c, rhoinf, uinf, tv)
#     hv = -H * np.cos(w0 * tv)
#     dhv = w0 * H * np.sin(w0 * tv)
#     aeffv = np.arctan(-dhv / uinf)
#     # fig = plt.figure('Induced drag - plunge motion',(10,6))
#     # ax=fig.add_subplot(111)
#     # ax.plot(tv,hv/c,'r',label=r'h/c')
#     # ax.plot(tv,Cdv,'k',label=r'Induced Drag')
#     # ax.legend()
#     # plt.show()
#     fig = plt.figure('Plunge motion - Phase vs kinematics', (10, 6))
#     ax = fig.add_subplot(111)
#     # ax.plot(aeffv,hv/c,'r',label=r'h/c')
#     ax.plot(180. / np.pi * aeffv, Cdv, 'k', label=r'Induced Drag')
#     ax.set_xlabel('deg')
#     ax.legend()
#     plt.close()
#
#     ##### Pitching Induced drag
#     Ncicles = 5
#     tv = np.linspace(0., 2. * np.pi * Ncicles / w0, 200 * Ncicles + 1)
#     Cdv = garrick_drag_pitch(w0, A, c, rhoinf, uinf, x12, tv)
#     aeffv = A * np.sin(w0 * tv)
#     fig = plt.figure('Pitch motion - Phase vs kinematics', (10, 6))
#     ax = fig.add_subplot(111)
#     # ax.plot(aeffv,hv/c,'r',label=r'h/c')
#     ax.plot(180. / np.pi * aeffv, Cdv, 'k', label=r'Induced Drag')
#     ax.set_xlabel('deg')
#     ax.legend()
#
#     ##### Sear's solution test
#     L = .5 * c
#     w0 = 0.3
#     uinf = 6.0
#
#     # gust profile at LE
#     tv = np.linspace(0., 2., 300)
#     C = 2. * np.pi / L
#     wgustLE = w0 * np.sin(C * uinf * tv)
#     CLv = sears_lift_sin_gust(w0, L, uinf, c, tv)
#
#     fig = plt.figure('Gust response', (10, 6))
#     ax = fig.add_subplot(111)
#     ax.plot(tv, wgustLE, 'k', label=r'vertical gust velocity at LE [m/s]')
#     ax.plot(tv, CLv, 'r', label=r'CL')
#     ax.set_xlabel('time')
#     ax.legend()
#     # plt.show()
#     plt.close('all')
#
#     ##### Wagner impulsive start
#     uinf = 20.0
#     chord = 3.0
#     aeff = 2.0 * np.pi / 180.
#     tv = np.linspace(0., 10., 300)
#
#     CLv = wagner_imp_start(aeff, uinf, chord, tv)
#     CLv_inf = wagner_imp_start(aeff, uinf, chord, 1e3 * tv[-1])
#
#     fig = plt.figure('Impulsive start', (10, 6))
#     ax = fig.add_subplot(111)
#     ax.plot(tv, CLv / CLv_inf, 'r', label=r'CL')
#     ax.set_xlabel('time')
#     ax.legend()
#     plt.show()
