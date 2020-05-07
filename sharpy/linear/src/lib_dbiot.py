"""Induced Velocity Derivatives

Calculate derivatives of induced velocity.

Methods:

- eval_seg_exp and eval_seg_exp_loop: profide ders in format
    [Q_{x,y,z},ZetaPoint_{x,y,z}]
  and use fully-expanded analytical formula.
- eval_panel_exp: iterates through whole panel

- eval_seg_comp and eval_seg_comp_loop: profide ders in format
    [Q_{x,y,z},ZetaPoint_{x,y,z}]
  and use compact analytical formula.
"""

import numpy as np

from sharpy.aero.utils.uvlmlib import eval_panel_cpp
import sharpy.utils.algebra as algebra
from sharpy.utils.constants import cfact_biot

### looping through panels
svec = [0, 1, 2, 3]  # seg. no.
# avec =[ 0, 1, 2, 3] # 1st vertex no.
# bvec =[ 1, 2, 3, 0] # 2nd vertex no.
LoopPanel = [(0, 1), (1, 2), (2, 3), (3, 0)]  # used in eval_panel_{exp/comp}


def eval_panel_cpp_coll(zetaP, ZetaPanel, vortex_radius, gamma_pan=1.0):
    DerP, DerVertices = eval_panel_cpp(zetaP, ZetaPanel, vortex_radius, gamma_pan)
    return DerP


def eval_seg_exp(ZetaP, ZetaA, ZetaB, vortex_radius, gamma_seg=1.0):
    """
    Derivative of induced velocity Q w.r.t. collocation and segment coordinates
    in format:
        [ (ZetaP,ZetaA,ZetaB), (x,y,z) of Zeta,  (x,y,z) of Q]

    Warning: function optimised for performance. Variables are scaled during the
    execution.
    """

    DerP = np.zeros((3, 3))
    DerA = np.zeros((3, 3))
    DerB = np.zeros((3, 3))
    eval_seg_exp_loop(DerP, DerA, DerB, ZetaP, ZetaA, ZetaB, gamma_seg,
                      vortex_radius)
    return DerP, DerA, DerB


def eval_seg_exp_loop(DerP, DerA, DerB, ZetaP, ZetaA, ZetaB, gamma_seg,
                      vortex_radius):
    """
    Derivative of induced velocity Q w.r.t. collocation (DerC) and segment
    coordinates in format.

    To optimise performance, the function requires the derivative terms to be
    pre-allocated and passed as input.

    Each Der* term returns derivatives in the format

        [ (x,y,z) of Zeta,  (x,y,z) of Q]

    Warning: to optimise performance, variables are scaled during the execution.
    """

    RA = ZetaP - ZetaA
    RB = ZetaP - ZetaB
    RAB = ZetaB - ZetaA
    Vcr = algebra.cross3(RA, RB)
    vcr2 = np.dot(Vcr, Vcr)

    # numerical radious
    vortex_radious_here = vortex_radius * algebra.norm3d(RAB)
    if vcr2 < vortex_radious_here ** 2:
        return

    # scaling
    ra1, rb1 = algebra.norm3d(RA), algebra.norm3d(RB)
    ra2, rb2 = ra1 ** 2, rb1 ** 2
    rainv = 1. / ra1
    rbinv = 1. / rb1
    ra_dir, rb_dir = RA * rainv, RB * rbinv
    ra3inv, rb3inv = rainv ** 3, rbinv ** 3
    Vcr = Vcr / vcr2

    diff_vec = ra_dir - rb_dir
    vdot_prod = np.dot(diff_vec, RAB)
    T2 = vdot_prod / vcr2

    # Extract components
    ra_x, ra_y, ra_z = RA
    rb_x, rb_y, rb_z = RB
    rab_x, rab_y, rab_z = RAB
    vcr_x, vcr_y, vcr_z = Vcr
    ra2_x, ra2_y, ra2_z = RA ** 2
    rb2_x, rb2_y, rb2_z = RB ** 2
    ra_vcr_x, ra_vcr_y, ra_vcr_z = 2. * algebra.cross3(RA, Vcr)
    rb_vcr_x, rb_vcr_y, rb_vcr_z = 2. * algebra.cross3(RB, Vcr)
    vcr_sca_x, vcr_sca_y, vcr_sca_z = Vcr * ra3inv
    vcr_scb_x, vcr_scb_y, vcr_scb_z = Vcr * rb3inv

    # # ### derivatives indices:
    # # # the 1st is the component of the vaiable w.r.t derivative are taken.
    # # # the 2nd is the component of the output
    dQ_dRA = np.array(
        [[-vdot_prod * rb_vcr_x * vcr_x + vcr_sca_x * (
                    rab_x * (ra2 - ra2_x) - ra_x * ra_y * rab_y - ra_x * ra_z * rab_z),
          -T2 * rb_z - vdot_prod * rb_vcr_x * vcr_y + vcr_sca_y * (
                      rab_x * (ra2 - ra2_x) - ra_x * ra_y * rab_y - ra_x * ra_z * rab_z),
          T2 * rb_y - vdot_prod * rb_vcr_x * vcr_z + vcr_sca_z * (
                      rab_x * (ra2 - ra2_x) - ra_x * ra_y * rab_y - ra_x * ra_z * rab_z)],
         [T2 * rb_z - vdot_prod * rb_vcr_y * vcr_x + vcr_sca_x * (
                     rab_y * (ra2 - ra2_y) - ra_x * ra_y * rab_x - ra_y * ra_z * rab_z),
          -vdot_prod * rb_vcr_y * vcr_y + vcr_sca_y * (
                      rab_y * (ra2 - ra2_y) - ra_x * ra_y * rab_x - ra_y * ra_z * rab_z),
          -T2 * rb_x - vdot_prod * rb_vcr_y * vcr_z + vcr_sca_z * (
                      rab_y * (ra2 - ra2_y) - ra_x * ra_y * rab_x - ra_y * ra_z * rab_z)],
         [-T2 * rb_y - vdot_prod * rb_vcr_z * vcr_x + vcr_sca_x * (
                     rab_z * (ra2 - ra2_z) - ra_x * ra_z * rab_x - ra_y * ra_z * rab_y),
          T2 * rb_x - vdot_prod * rb_vcr_z * vcr_y + vcr_sca_y * (
                      rab_z * (ra2 - ra2_z) - ra_x * ra_z * rab_x - ra_y * ra_z * rab_y),
          -vdot_prod * rb_vcr_z * vcr_z + vcr_sca_z * (
                      rab_z * (ra2 - ra2_z) - ra_x * ra_z * rab_x - ra_y * ra_z * rab_y)]])

    dQ_dRB = np.array(
        [[vdot_prod * ra_vcr_x * vcr_x + vcr_scb_x * (
                    rab_x * (-rb2 + rb2_x) + rab_y * rb_x * rb_y + rab_z * rb_x * rb_z),
          T2 * ra_z + vdot_prod * ra_vcr_x * vcr_y + vcr_scb_y * (
                      rab_x * (-rb2 + rb2_x) + rab_y * rb_x * rb_y + rab_z * rb_x * rb_z),
          -T2 * ra_y + vdot_prod * ra_vcr_x * vcr_z + vcr_scb_z * (
                      rab_x * (-rb2 + rb2_x) + rab_y * rb_x * rb_y + rab_z * rb_x * rb_z)],
         [-T2 * ra_z + vdot_prod * ra_vcr_y * vcr_x + vcr_scb_x * (
                     rab_x * rb_x * rb_y + rab_y * (-rb2 + rb2_y) + rab_z * rb_y * rb_z),
          vdot_prod * ra_vcr_y * vcr_y + vcr_scb_y * (
                      rab_x * rb_x * rb_y + rab_y * (-rb2 + rb2_y) + rab_z * rb_y * rb_z),
          T2 * ra_x + vdot_prod * ra_vcr_y * vcr_z + vcr_scb_z * (
                      rab_x * rb_x * rb_y + rab_y * (-rb2 + rb2_y) + rab_z * rb_y * rb_z)],
         [T2 * ra_y + vdot_prod * ra_vcr_z * vcr_x + vcr_scb_x * (
                     rab_x * rb_x * rb_z + rab_y * rb_y * rb_z + rab_z * (-rb2 + rb2_z)),
          -T2 * ra_x + vdot_prod * ra_vcr_z * vcr_y + vcr_scb_y * (
                      rab_x * rb_x * rb_z + rab_y * rb_y * rb_z + rab_z * (-rb2 + rb2_z)),
          vdot_prod * ra_vcr_z * vcr_z + vcr_scb_z * (
                      rab_x * rb_x * rb_z + rab_y * rb_y * rb_z + rab_z * (-rb2 + rb2_z))]])

    dQ_dRAB = np.array(
        [[vcr_x * diff_vec[0],
          vcr_y * diff_vec[0],
          vcr_z * diff_vec[0]],
         [vcr_x * diff_vec[1],
          vcr_y * diff_vec[1],
          vcr_z * diff_vec[1]],
         [vcr_x * diff_vec[2],
          vcr_y * diff_vec[2],
          vcr_z * diff_vec[2]]])

    DerP += (cfact_biot * gamma_seg) * (dQ_dRA + dQ_dRB).T  # w.r.t. P
    DerA += (cfact_biot * gamma_seg) * (-dQ_dRAB - dQ_dRA).T  # w.r.t. A
    DerB += (cfact_biot * gamma_seg) * (dQ_dRAB - dQ_dRB).T  # w.r.t. B


def eval_panel_exp(zetaP, ZetaPanel, vortex_radius, gamma_pan=1.0):
    """
    Computes derivatives of induced velocity w.r.t. coordinates of target point,
    zetaP, and panel coordinates. Returns two elements:
        - DerP: derivative of induced velocity w.r.t. ZetaP, with:
            DerP.shape=(3,3) : DerC[ Uind_{x,y,z}, ZetaC_{x,y,z} ]
        - DerVertices: derivative of induced velocity wrt panel vertices, with:
            DerVertices.shape=(4,3,3) :
            DerVertices[ vertex number {0,1,2,3}, Uind_{x,y,z}, ZetaC_{x,y,z} ]
    """

    DerP = np.zeros((3, 3))
    DerVertices = np.zeros((4, 3, 3))

    for aa, bb in LoopPanel:
        eval_seg_exp_loop(DerP, DerVertices[aa, :, :], DerVertices[bb, :, :],
                          zetaP, ZetaPanel[aa, :], ZetaPanel[bb, :], gamma_pan,
                          vortex_radius)

    return DerP, DerVertices


# ------------------------------------------------------------------------------
#	Compact Formula
# ------------------------------------------------------------------------------


def Dvcross_by_skew3d(Dvcross, rv):
    """
    Fast matrix multiplication of der(vcross)*skew(rv), where
        vcross = (rv x sv)/|rv x sv|^2
    The function exploits the property that the output matrix is symmetric.
    DvCross is a list containing the lower diagonal elements
    """
    P = np.empty((3, 3))
    P[0, 0] = Dvcross[1][0] * rv[2] - Dvcross[2][0] * rv[1]
    P[0, 1] = Dvcross[2][0] * rv[0] - Dvcross[0][0] * rv[2]
    P[0, 2] = Dvcross[0][0] * rv[1] - Dvcross[1][0] * rv[0]
    #
    P[1, 0] = P[0, 1]
    P[1, 1] = Dvcross[2][1] * rv[0] - Dvcross[1][0] * rv[2]
    P[1, 2] = Dvcross[1][0] * rv[1] - Dvcross[1][1] * rv[0]
    #
    P[2, 0] = P[0, 2]
    P[2, 1] = P[1, 2]
    P[2, 2] = Dvcross[2][0] * rv[1] - Dvcross[2][1] * rv[0]
    return P


# def Dvcross_by_skew3d(Dvcross,rv):
# 	"""
# 	Fast matrix multiplication of der(vcross)*skew(rv), where
# 		vcross = (rv x sv)/|rv x sv|^2
# 	The function exploits the property that the output matrix is symmetric.
# 	"""
# 	P=np.empty((3,3))
# 	P[0,0]=Dvcross[0,1]*rv[2]-Dvcross[0,2]*rv[1]
# 	P[0,1]=Dvcross[0,2]*rv[0]-Dvcross[0,0]*rv[2]
# 	P[0,2]=Dvcross[0,0]*rv[1]-Dvcross[0,1]*rv[0]
# 	#
# 	P[1,0]=P[0,1]
# 	P[1,1]=Dvcross[1,2]*rv[0]-Dvcross[0,1]*rv[2]
# 	P[1,2]=Dvcross[0,1]*rv[1]-Dvcross[1,1]*rv[0]
# 	#
# 	P[2,0]=P[0,2]
# 	P[2,1]=P[1,2]
# 	P[2,2]=Dvcross[0,2]*rv[1]-Dvcross[1,2]*rv[0]
# 	return P

def der_runit(r, rinv, minus_rinv3):
    # alloc upper diag
    Der = np.empty((3, 3))
    Der[0, 0] = rinv + minus_rinv3 * r[0] ** 2
    Der[0, 1] = minus_rinv3 * r[0] * r[1]
    Der[0, 2] = minus_rinv3 * r[0] * r[2]
    Der[1, 1] = rinv + minus_rinv3 * r[1] ** 2
    Der[1, 2] = minus_rinv3 * r[1] * r[2]
    Der[2, 2] = rinv + minus_rinv3 * r[2] ** 2
    # alloc lower
    Der[1, 0] = Der[0, 1]
    Der[2, 0] = Der[0, 2]
    Der[2, 1] = Der[1, 2]
    return Der


def eval_seg_comp(ZetaP, ZetaA, ZetaB, vortex_radius, gamma_seg=1.0):
    DerP = np.zeros((3, 3))
    DerA = np.zeros((3, 3))
    DerB = np.zeros((3, 3))
    eval_seg_comp_loop(DerP, DerA, DerB, ZetaP, ZetaA, ZetaB, gamma_seg,
                       vortex_radius)
    return DerP, DerA, DerB


def eval_seg_comp_loop(DerP, DerA, DerB, ZetaP, ZetaA, ZetaB, gamma_seg, vortex_radius):
    """
    Derivative of induced velocity Q w.r.t. collocation and segment coordinates
    in format:
        [ (x,y,z) of Q, (x,y,z) of Zeta ]
    Warning: function optimised for performance. Variables are scaled during the
    execution.
    """

    vortex_radius_sq = vortex_radius*vortex_radius
    Cfact = cfact_biot * gamma_seg

    RA = ZetaP - ZetaA
    RB = ZetaP - ZetaB
    RAB = ZetaB - ZetaA
    Vcr = algebra.cross3(RA, RB)
    vcr2 = np.dot(Vcr, Vcr)

    # numerical radious
    if vcr2 < (vortex_radius_sq * algebra.normsq3d(RAB)):
        return

    ### other constants
    ra1, rb1 = algebra.norm3d(RA), algebra.norm3d(RB)
    rainv = 1. / ra1
    rbinv = 1. / rb1
    Tv = RA * rainv - RB * rbinv
    dotprod = np.dot(RAB, Tv)

    ### --------------------------------------------- cross-product derivatives
    # lower triangular part only
    vcr2inv = 1. / vcr2
    vcr4inv = vcr2inv * vcr2inv
    diag_fact = Cfact * vcr2inv * dotprod
    off_fact = -2. * Cfact * vcr4inv * dotprod
    Dvcross = [
        [diag_fact + off_fact * Vcr[0] ** 2],
        [off_fact * Vcr[0] * Vcr[1], diag_fact + off_fact * Vcr[1] ** 2],
        [off_fact * Vcr[0] * Vcr[2], off_fact * Vcr[1] * Vcr[2], diag_fact + off_fact * Vcr[2] ** 2]]

    ### ------------------------------------------ difference terms derivatives
    Vsc = Vcr * vcr2inv * Cfact
    Ddiff = np.array([RAB * Vsc[0], RAB * Vsc[1], RAB * Vsc[2]])
    dQ_dRAB = np.array([Tv * Vsc[0], Tv * Vsc[1], Tv * Vsc[2]])

    ### ---------------------------------------------- Final assembly (crucial)
    # ps: calling Dvcross_by_skew3d does not slow down execution.

    dQ_dRA = Dvcross_by_skew3d(Dvcross, -RB) \
             + np.dot(Ddiff, der_runit(RA, rainv, -rainv ** 3))
    dQ_dRB = Dvcross_by_skew3d(Dvcross, RA) \
             - np.dot(Ddiff, der_runit(RB, rbinv, -rbinv ** 3))

    DerP += dQ_dRA + dQ_dRB  # w.r.t. P
    DerA -= dQ_dRAB + dQ_dRA  # w.r.t. A
    DerB += dQ_dRAB - dQ_dRB  # w.r.t. B


# ### collocation point only
# DerP +=Dvcross_by_skew3d(Dvcross,RA-RB)+np.dot(Ddiff,
# 		  der_runit(RA,rainv,minus_rainv3)-der_runit(RB,rbinv,minus_rbinv3))


def eval_panel_comp(zetaP, ZetaPanel, vortex_radius, gamma_pan=1.0):
    """
    Computes derivatives of induced velocity w.r.t. coordinates of target point,
    zetaP, and panel coordinates. Returns two elements:
        - DerP: derivative of induced velocity w.r.t. ZetaP, with:
            DerP.shape=(3,3) : DerC[ Uind_{x,y,z}, ZetaC_{x,y,z} ]
        - DerVertices: derivative of induced velocity wrt panel vertices, with:
            DerVertices.shape=(4,3,3) :
            DerVertices[ vertex number {0,1,2,3},  Uind_{x,y,z}, ZetaC_{x,y,z} ]
    """

    DerP = np.zeros((3, 3))
    DerVertices = np.zeros((4, 3, 3))

    for aa, bb in LoopPanel:
        eval_seg_comp_loop(DerP, DerVertices[aa, :, :], DerVertices[bb, :, :],
                           zetaP, ZetaPanel[aa, :], ZetaPanel[bb, :], gamma_pan,
                           vortex_radius)

    return DerP, DerVertices


def eval_panel_fast(zetaP, ZetaPanel, vortex_radius, gamma_pan=1.0):
    """
    Computes derivatives of induced velocity w.r.t. coordinates of target point,
    zetaP, and panel coordinates. Returns two elements:
        - DerP: derivative of induced velocity w.r.t. ZetaP, with:
            DerP.shape=(3,3) : DerC[ Uind_{x,y,z}, ZetaC_{x,y,z} ]
        - DerVertices: derivative of induced velocity wrt panel vertices, with:
            DerVertices.shape=(4,3,3) :
            DerVertices[ vertex number {0,1,2,3},  Uind_{x,y,z}, ZetaC_{x,y,z} ]

    The function is based on eval_panel_comp, but minimises operationsby
    recycling variables.
    """

    vortex_radius_sq = vortex_radius*vortex_radius
    DerP = np.zeros((3, 3))
    DerVertices = np.zeros((4, 3, 3))

    ### ---------------------------------------------- Compute common variables
    # these are constants or variables depending only on vertices and P coords
    Cfact = cfact_biot * gamma_pan

    # distance vertex ii-th from P
    R_list = zetaP - ZetaPanel
    r1_list = [algebra.norm3d(R_list[ii]) for ii in svec]
    r1inv_list = [1. / r1_list[ii] for ii in svec]
    Runit_list = [R_list[ii] * r1inv_list[ii] for ii in svec]
    Der_runit_list = [
        der_runit(R_list[ii], r1inv_list[ii], -r1inv_list[ii] ** 3) for ii in svec]

    ### ------------------------------------------------- Loop through segments
    for aa, bb in LoopPanel:

        RAB = ZetaPanel[bb, :] - ZetaPanel[aa, :]  # segment vector
        Vcr = algebra.cross3(R_list[aa], R_list[bb])
        vcr2 = np.dot(Vcr, Vcr)

        if vcr2 < (vortex_radius_sq * algebra.normsq3d(RAB)):
            continue

        Tv = Runit_list[aa] - Runit_list[bb]
        dotprod = np.dot(RAB, Tv)

        ### ----------------------------------------- cross-product derivatives
        # lower triangular part only
        vcr2inv = 1. / vcr2
        vcr4inv = vcr2inv * vcr2inv
        diag_fact = Cfact * vcr2inv * dotprod
        off_fact = -2. * Cfact * vcr4inv * dotprod
        Dvcross = [
            [diag_fact + off_fact * Vcr[0] ** 2],
            [off_fact * Vcr[0] * Vcr[1], diag_fact + off_fact * Vcr[1] ** 2],
            [off_fact * Vcr[0] * Vcr[2], off_fact * Vcr[1] * Vcr[2], diag_fact + off_fact * Vcr[2] ** 2]]

        ### ---------------------------------------- difference term derivative
        Vsc = Vcr * vcr2inv * Cfact
        Ddiff = np.array([RAB * Vsc[0], RAB * Vsc[1], RAB * Vsc[2]])

        ### ---------------------------------------------------- RAB derivative
        dQ_dRAB = np.array([Tv * Vsc[0], Tv * Vsc[1], Tv * Vsc[2]])

        ### ------------------------------------------ Final assembly (crucial)

        dQ_dRA = Dvcross_by_skew3d(Dvcross, -R_list[bb]) \
                 + np.dot(Ddiff, Der_runit_list[aa])
        dQ_dRB = Dvcross_by_skew3d(Dvcross, R_list[aa]) \
                 - np.dot(Ddiff, Der_runit_list[bb])

        DerP += dQ_dRA + dQ_dRB  # w.r.t. P
        DerVertices[aa, :, :] -= dQ_dRAB + dQ_dRA  # w.r.t. A
        DerVertices[bb, :, :] += dQ_dRAB - dQ_dRB  # w.r.t. B

    # ### collocation point only
    # DerP +=Dvcross_by_skew3d(Dvcross,RA-RB)+np.dot(Ddiff,
    # 		  der_runit(RA,rainv,minus_rainv3)-der_runit(RB,rbinv,minus_rbinv3))

    return DerP, DerVertices


def eval_panel_fast_coll(zetaP, ZetaPanel, vortex_radius, gamma_pan=1.0):
    """
    Computes derivatives of induced velocity w.r.t. coordinates of target point,
    zetaP, coordinates. Returns two elements:
        - DerP: derivative of induced velocity w.r.t. ZetaP, with:
            DerP.shape=(3,3) : DerC[ Uind_{x,y,z}, ZetaC_{x,y,z} ]

    The function is based on eval_panel_fast, but does not perform operations
    required to compute the derivatives w.r.t. the panel coordinates.
    """

    vortex_radius_sq = vortex_radius*vortex_radius
    DerP = np.zeros((3, 3))

    ### ---------------------------------------------- Compute common variables
    # these are constants or variables depending only on vertices and P coords
    Cfact = cfact_biot * gamma_pan

    # distance vertex ii-th from P
    R_list = zetaP - ZetaPanel
    r1_list = [algebra.norm3d(R_list[ii]) for ii in svec]
    r1inv_list = [1. / r1_list[ii] for ii in svec]
    Runit_list = [R_list[ii] * r1inv_list[ii] for ii in svec]
    Der_runit_list = [
        der_runit(R_list[ii], r1inv_list[ii], -r1inv_list[ii] ** 3) for ii in svec]

    ### ------------------------------------------------- Loop through segments
    for aa, bb in LoopPanel:

        RAB = ZetaPanel[bb, :] - ZetaPanel[aa, :]  # segment vector
        Vcr = algebra.cross3(R_list[aa], R_list[bb])
        vcr2 = np.dot(Vcr, Vcr)

        if vcr2 < (vortex_radius_sq * algebra.normsq3d(RAB)):
            continue

        Tv = Runit_list[aa] - Runit_list[bb]
        dotprod = np.dot(RAB, Tv)

        ### ----------------------------------------- cross-product derivatives
        # lower triangular part only
        vcr2inv = 1. / vcr2
        vcr4inv = vcr2inv * vcr2inv
        diag_fact = Cfact * vcr2inv * dotprod
        off_fact = -2. * Cfact * vcr4inv * dotprod
        Dvcross = [
            [diag_fact + off_fact * Vcr[0] ** 2],
            [off_fact * Vcr[0] * Vcr[1], diag_fact + off_fact * Vcr[1] ** 2],
            [off_fact * Vcr[0] * Vcr[2], off_fact * Vcr[1] * Vcr[2], diag_fact + off_fact * Vcr[2] ** 2]]

        ### ---------------------------------------- difference term derivative
        Vsc = Vcr * vcr2inv * Cfact
        Ddiff = np.array([RAB * Vsc[0], RAB * Vsc[1], RAB * Vsc[2]])

        ### ------------------------------------------ Final assembly (crucial)

        # dQ_dRA=Dvcross_by_skew3d(Dvcross,-R_list[bb])\
        # 									 +np.dot(Ddiff, Der_runit_list[aa] )
        # dQ_dRB=Dvcross_by_skew3d(Dvcross, R_list[aa])\
        # 									 -np.dot(Ddiff, Der_runit_list[bb] )

        DerP += Dvcross_by_skew3d(Dvcross, RAB) + \
                +np.dot(Ddiff, Der_runit_list[aa] - Der_runit_list[bb])

    return DerP


if __name__ == '__main__':

    import cProfile

    ### verify consistency amongst models
    gamma = 4.
    zetaP = np.array([3.0, 5.5, 2.0])
    zeta0 = np.array([1.0, 3.0, 0.9])
    zeta1 = np.array([5.0, 3.1, 1.9])
    zeta2 = np.array([4.8, 8.1, 2.5])
    zeta3 = np.array([0.9, 7.9, 1.7])
    ZetaPanel = np.array([zeta0, zeta1, zeta2, zeta3])
    zetap = 0.3 * zeta1 + 0.7 * zeta2

    # ZetaPanel=np.array([[ 1.221, -0.064, -0.085],
    #        				[ 1.826, -0.064, -0.141],
    #        				[ 1.933,  1.456, -0.142],
    #        				[ 1.327,  1.456, -0.087]])
    # zetaP=np.array([-0.243,  0.776,  0.037])

    ### verify model consistency
    DPcpp, DVcpp = eval_panel_cpp(zetaP, ZetaPanel, 1e-6, gamma_pan=gamma) # vortex_radius
    DPexp, DVexp = eval_panel_exp(zetaP, ZetaPanel, gamma_pan=gamma)
    DPcomp, DVcomp = eval_panel_comp(zetaP, ZetaPanel, gamma_pan=gamma)
    DPfast, DVfast = eval_panel_fast(zetaP, ZetaPanel,
                                     vortex_radius, gamma_pan=gamma)
    DPfast_coll = eval_panel_fast_coll(zetaP, ZetaPanel,
                                       vortex_radius, gamma_pan=gamma)

    ermax = max(np.max(np.abs(DPcpp - DPexp)), np.max(np.abs(DVcpp - DVexp)))
    assert ermax < 1e-15, 'eval_panel_cpp not matching with eval_panel_exp'
    ermax = max(np.max(np.abs(DPcomp - DPexp)), np.max(np.abs(DVcomp - DVexp)))
    assert ermax < 1e-15, 'eval_panel_comp not matching with eval_panel_exp'
    ermax = max(np.max(np.abs(DPfast - DPexp)), np.max(np.abs(DVfast - DVexp)))
    assert ermax < 1e-15, 'eval_panel_fast not matching with eval_panel_exp'
    ermax = np.max(np.abs(DPfast_coll - DPexp))
    assert ermax < 1e-15, 'eval_panel_fast_coll not matching with eval_panel_exp'


    ### profiling

    def run_eval_panel_cpp():
        for ii in range(10000):
            eval_panel_cpp(zetaP, ZetaPanel, 1e-6, gamma_pan=3.) # vortex_radius


    def run_eval_panel_fast(vortex_radius=vortex_radius_def):
        for ii in range(10000):
            eval_panel_fast(zetaP, ZetaPanel,
                                   vortex_radius, gamma_pan=3.)


    def run_eval_panel_fast_coll(vortex_radius=vortex_radius_def):
        for ii in range(10000):
            eval_panel_fast_coll(zetaP, ZetaPanel,
                                 vortex_radius, gamma_pan=3.)


    def run_eval_panel_comp():
        for ii in range(10000):
            eval_panel_comp(zetaP, ZetaPanel, gamma_pan=3.)


    def run_eval_panel_exp():
        for ii in range(10000):
            eval_panel_exp(zetaP, ZetaPanel, gamma_pan=3.)


    print('------------------------------------------ profiling eval_panel_cpp')
    cProfile.runctx('run_eval_panel_cpp()', globals(), locals())

    print('----------------------------------------- profiling eval_panel_fast')
    cProfile.runctx('run_eval_panel_fast()', globals(), locals())

    print('------------------------------------ profiling eval_panel_fast_coll')
    cProfile.runctx('run_eval_panel_fast_coll()', globals(), locals())

    print('----------------------------------------- profiling eval_panel_comp')
    cProfile.runctx('run_eval_panel_comp()', globals(), locals())

    print('------------------------------------------ profiling eval_panel_exp')
    cProfile.runctx('run_eval_panel_exp()', globals(), locals())
