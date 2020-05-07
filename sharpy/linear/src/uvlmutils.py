"""Methods for UVLM solution

S. Maraniello, 1 Jun 2018
"""
import numpy as np
import sharpy.aero.utils.uvlmlib as uvlmlib
import sharpy.utils.algebra as algebra
from sharpy.utils.constants import cfact_biot

# local mapping segment/vertices of a panel
svec = [0, 1, 2, 3]  # seg. number
avec = [0, 1, 2, 3]  # 1st vertex of seg.
bvec = [1, 2, 3, 0]  # 2nd vertex of seg.
LoopPanel = [(0, 1), (1, 2), (2, 3), (3, 0)]  # used in eval_panel_{exp/comp}


def joukovski_qs_segment(zetaA, zetaB, v_mid, gamma=1.0, fact=0.5):
    """
    Joukovski force over vetices A and B produced by the segment A->B.
    The factor fact allows to compute directly the contribution over the
    vertices A and B (e.g. 0.5) or include DENSITY.
    """

    rab = zetaB - zetaA
    fs = algebra.cross3(v_mid, rab)
    gfact = fact * gamma

    return gfact * fs


def biot_segment(zetaP, zetaA, zetaB, vortex_radius, gamma=1.0):
    """
    Induced velocity of segment A_>B of circulation gamma over point P.
    """

    vortex_radius_sq = vortex_radius*vortex_radius
    # differences
    ra = zetaP - zetaA
    rb = zetaP - zetaB
    rab = zetaB - zetaA
    ra_norm, rb_norm = algebra.norm3d(ra), algebra.norm3d(rb)
    vcross = algebra.cross3(ra, rb)
    vcross_sq = np.dot(vcross, vcross)

    # numerical radius
    if vcross_sq < (vortex_radius_sq * algebra.normsq3d(rab)):
        return np.zeros((3,))

    q = ((cfact_biot * gamma / vcross_sq) * \
         (np.dot(rab, ra) / ra_norm - np.dot(rab, rb) / rb_norm)) * vcross

    return q


def biot_panel(zetaC, ZetaPanel, vortex_radius, gamma=1.0):
    """
    Induced velocity over point ZetaC of a panel of vertices coordinates
    ZetaPanel and circulaiton gamma, where:
        ZetaPanel.shape=(4,3)=[vertex local number, (x,y,z) component]
    """

    q = np.zeros((3,))
    for ss, aa, bb in zip(svec, avec, bvec):
        q += biot_segment(zetaC, ZetaPanel[aa, :], ZetaPanel[bb, :],
                          vortex_radius, gamma)

    return q


def biot_panel_fast(zetaC, ZetaPanel, vortex_radius, gamma=1.0):
    """
    Induced velocity over point ZetaC of a panel of vertices coordinates
    ZetaPanel and circulaiton gamma, where:
        ZetaPanel.shape=(4,3)=[vertex local number, (x,y,z) component]
    """

    vortex_radius_sq = vortex_radius*vortex_radius
    Cfact = cfact_biot * gamma
    q = np.zeros((3,))

    R_list = zetaC - ZetaPanel
    Runit_list = [R_list[ii] / algebra.norm3d(R_list[ii]) for ii in svec]

    for aa, bb in LoopPanel:

        RAB = ZetaPanel[bb, :] - ZetaPanel[aa, :]  # segment vector
        Vcr = algebra.cross3(R_list[aa], R_list[bb])
        vcr2 = np.dot(Vcr, Vcr)
        if vcr2 < (vortex_radius * algebra.normsq3d(RAB)):
            continue

        q += ((Cfact / vcr2) * np.dot(RAB, Runit_list[aa] - Runit_list[bb])) * Vcr

    return q


def panel_normal(ZetaPanel):
    """
    return normal of panel with vertex coordinates ZetaPanel, where:
        ZetaPanel.shape=(4,3)
    """

    # build cross-vectors
    r02 = ZetaPanel[2, :] - ZetaPanel[0, :]
    r13 = ZetaPanel[3, :] - ZetaPanel[1, :]

    nvec = algebra.cross3(r02, r13)
    nvec = nvec / algebra.norm3d(nvec)

    return nvec


def panel_area(ZetaPanel):
    """
    return area of panel with vertices coordinates ZetaPanel, where:
        ZetaPanel.shape=(4,3)
    using Bretschneider formula - for cyclic or non-cyclic quadrilaters.
    """

    # build cross-vectors
    r02 = ZetaPanel[2, :] - ZetaPanel[0, :]
    r13 = ZetaPanel[3, :] - ZetaPanel[1, :]
    # build side vectors
    r01 = ZetaPanel[1, :] - ZetaPanel[0, :]
    r12 = ZetaPanel[2, :] - ZetaPanel[1, :]
    r23 = ZetaPanel[3, :] - ZetaPanel[2, :]
    r30 = ZetaPanel[0, :] - ZetaPanel[3, :]

    # compute distances
    d02 = algebra.norm3d(r02)
    d13 = algebra.norm3d(r13)
    d01 = algebra.norm3d(r01)
    d12 = algebra.norm3d(r12)
    d23 = algebra.norm3d(r23)
    d30 = algebra.norm3d(r30)

    A = 0.25 * np.sqrt((4. * d02 ** 2 * d13 ** 2) - ((d12 ** 2 + d30 ** 2) - (d01 ** 2 + d23 ** 2)) ** 2)

    return A


if __name__ == '__main__':

    import cProfile

    ### verify consistency amongst models
    gamma = 4.
    zeta0 = np.array([1.0, 3.0, 0.9])
    zeta1 = np.array([5.0, 3.1, 1.9])
    zeta2 = np.array([4.8, 8.1, 2.5])
    zeta3 = np.array([0.9, 7.9, 1.7])
    ZetaPanel = np.array([zeta0, zeta1, zeta2, zeta3])

    zetaP = np.array([3.0, 5.5, 2.0])
    zetaP = zeta2 * 0.3 + zeta3 * 0.7

    ### verify model consistency
    qref = biot_panel(zetaP, ZetaPanel, 1e-6, gamma=gamma) # vortex_radius
    qfast = biot_panel_fast(zetaP, ZetaPanel, 1e-6, gamma=gamma) # vortex_radius
    qcpp = uvlmlib.biot_panel_cpp(zetaP, ZetaPanel, 1e-6, gamma=gamma) # vortex_radius

    ermax = np.max(np.abs(qref - qfast))
    assert ermax < 1e-16, 'biot_panel_fast not matching with biot_panel'
    ermax = np.max(np.abs(qref - qcpp))
    assert ermax < 1e-16, 'biot_panel_cpp not matching with biot_panel'


    ### profiling
    def run_biot_panel_cpp():
        for ii in range(10000):
            uvlmlib.biot_panel_cpp(zetaP, ZetaPanel, 1e-6, gamma=3.) # vortex_radius


    def run_biot_panel_fast():
        for ii in range(10000):
            biot_panel_fast(zetaP, ZetaPanel, 1e-6, gamma=3.) # vortex_radius


    def run_biot_panel_ref():
        for ii in range(10000):
            biot_panel(zetaP, ZetaPanel, 1e-6, gamma=3.) # vortex_radius


    print('------------------------------------------ profiling biot_panel_cpp')
    cProfile.runctx('run_biot_panel_cpp()', globals(), locals())

    print('----------------------------------------- profiling biot_panel_fast')
    cProfile.runctx('run_biot_panel_fast()', globals(), locals())

    print('------------------------------------------ profiling biot_panel_ref')
    cProfile.runctx('run_biot_panel_ref()', globals(), locals())
