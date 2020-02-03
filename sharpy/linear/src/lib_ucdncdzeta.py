r"""Induced Velocity Derivatives with respect to Panel Normal

Calculate derivative of

    ..  math:: \boldsymbol{u}_c\frac{\partial\boldsymbol{n}_c}{\partial\boldsymbol{zeta}}

with respect to local panel coordinates.
"""

import numpy as np
import sharpy.utils.algebra as algebra

dR_dZeta = np.array(
    [[[[-1, 0, 0], [0, 0, 0]], [[0, -1, 0], [0, 0, 0]], [[0, 0, -1], [0, 0, 0]]],
     [[[0, 0, 0], [-1, 0, 0]], [[0, 0, 0], [0, -1, 0]], [[0, 0, 0], [0, 0, -1]]],
     [[[1, 0, 0], [0, 0, 0]], [[0, 1, 0], [0, 0, 0]], [[0, 0, 1], [0, 0, 0]]],
     [[[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 0, 1]]]])


def eval(Zeta00, Zeta01, Zeta02, Zeta03, Uc):
    """
    Returns a 4 x 3 array, containing the derivative of Wnc*Uc w.r.t the panel
    vertices coordinates.
    """

    R02 = Zeta02 - Zeta00
    R13 = Zeta03 - Zeta01

    crR02R13 = algebra.cross3(R02, R13)
    norm_crR02R13 = np.linalg.norm(crR02R13)
    cub_crR02R13 = norm_crR02R13 ** 3

    Acr = algebra.cross3(crR02R13, R13)
    Bcr = algebra.cross3(crR02R13, R02)
    Cdot = np.dot(crR02R13, Uc)

    uc_x, uc_y, uc_z = Uc
    r02_x, r02_y, r02_z = R02
    r13_x, r13_y, r13_z = R13
    crR13Uc_x, crR13Uc_y, crR13Uc_z = algebra.cross3(R13, Uc)
    crR02Uc_x, crR02Uc_y, crR02Uc_z = algebra.cross3(R02, Uc)
    crR02R13_x, crR02R13_y, crR02R13_z = crR02R13
    Acr_x, Acr_y, Acr_z = Acr
    Bcr_x, Bcr_y, Bcr_z = Bcr

    # dUnorm_dR.shape=(2,3)
    dUnorm_dR = np.array(
        [[Acr_x * Cdot / cub_crR02R13 + crR13Uc_x / norm_crR02R13,
          Acr_y * Cdot / cub_crR02R13 + crR13Uc_y / norm_crR02R13,
          Acr_z * Cdot / cub_crR02R13 + crR13Uc_z / norm_crR02R13],
         [-Bcr_x * Cdot / cub_crR02R13 - crR02Uc_x / norm_crR02R13,
          -Bcr_y * Cdot / cub_crR02R13 - crR02Uc_y / norm_crR02R13,
          -Bcr_z * Cdot / cub_crR02R13 - crR02Uc_z / norm_crR02R13]])

    # Allocate
    dUnorm_dZeta = np.zeros((4, 3))
    for vv in range(4):  # loop through panel vertices
        for cc_zeta in range(3):  # loop panel vertices component
            for rr in range(2):  # loop segments R02, R13
                for cc_rvec in range(3):  # loop segment component
                    dUnorm_dZeta[vv, cc_zeta] += \
                        dUnorm_dR[rr, cc_rvec] * dR_dZeta[vv, cc_zeta, rr, cc_rvec]

    return dUnorm_dZeta


if __name__ == '__main__':

    # calculate normal
    def get_panel_normal(zetav_here):
        """From Surface.AeroGridSurface"""
        r02 = zetav_here[2, :] - zetav_here[0, :]
        r13 = zetav_here[3, :] - zetav_here[1, :]
        nvec = np.cross(r02, r13)
        nvec = nvec / np.linalg.norm(nvec)
        return nvec


    # define panel vertices
    zeta00 = np.array([1.0, 0.2, 1.0])
    zeta01 = np.array([3.9, 0.1, 0.8])
    zeta02 = np.array([4.0, 3.5, 0.9])
    zeta03 = np.array([1.2, 3.2, 1.1])
    zeta_panel = np.array([zeta00, zeta01, zeta02, zeta03])

    # reference normal
    nvec0 = get_panel_normal(zeta_panel)

    # reference normal velocity
    ucoll = np.array([2, 6, 3])
    unorm0 = np.dot(nvec0, ucoll)

    # analytical derivative
    dUnorm_dZeta = eval(zeta00, zeta01, zeta02, zeta03, ucoll)

    # numerical derivative
    Dnum = np.zeros((4, 3))
    step = 1e-6
    for ii in range(4):
        for jj in range(3):
            delta = np.zeros((4, 3))
            delta[ii, jj] = step
            nvec_pert = get_panel_normal(zeta_panel + delta)
            unorm_pert = np.dot(nvec_pert, ucoll)
            Dnum[ii, jj] = (unorm_pert - unorm0) / step

    assert np.max(np.abs(dUnorm_dZeta - Dnum)) < step, 'Derivative not accurate!'
