"""
Utilities functions for linear analysis
"""

import numpy as np



# linear uvlm
import sharpy.linear.src.lin_aeroelastic as lin_aeroelastic
import sharpy.linear.src.libss as libss


def comp_tot_force(forces, zeta, zeta_pole=np.zeros((3,))):
    """ Compute total force with exact displacements """
    Ftot = np.zeros((3,))
    Mtot = np.zeros((3,))
    for ss in range(len(forces)):
        _, Mv, Nv = forces[ss].shape
        for mm in range(Mv):
            for nn in range(Nv):
                arm = zeta[ss][:, mm, nn] - zeta_pole
                Mtot += np.cross(arm, forces[ss][:3, mm, nn])
        for cc in range(3):
            Ftot[cc] += forces[ss][cc, :, :].sum()
    return Ftot, Mtot


class Info():
    """ Summarise info about a data point """

    def __init__(self, zeta, zeta_dot, u_ext, ftot, mtot, q, qdot,
                 SSaero=None, SSbeam=None,
                 Kas=None, Kftot=None, Kmtot=None, Kmtot_disp=None,
                 Asteady_inv=None):
        self.zeta = zeta
        self.zeta_dot = zeta_dot
        self.u_ext = u_ext
        self.ftot = ftot
        self.mtot = mtot
        self.q = q
        self.qdot = qdot
        #
        self.SSaero = SSaero
        self.SSbeam = SSbeam
        self.Kas = Kas
        self.Kftot = Kftot
        self.Kmtot = Kmtot
        self.Kmtot_disp = Kmtot_disp
        self.Asteady_inv = Asteady_inv


def solve_linear(Ref, Pert, solve_beam=True):
    """
    Given 2 Info() classes associated to a reference linearisation point Ref and a
    perturbed state Pert, the method produces in output the prediction at the
    Pert state of a linearised model.

    The solution is carried on using both the aero and beam input
    """

    ### define perturbations
    dq = Pert.q - Ref.q
    dqdot = Pert.qdot - Ref.qdot
    dzeta = Pert.zeta - Ref.zeta
    dzeta_dot = Pert.zeta_dot - Ref.zeta_dot
    du_ext = Pert.u_ext - Ref.u_ext

    num_dof_str = len(dq)
    dzeta_exp = np.dot(Ref.Kas[:len(dzeta), :num_dof_str], dq)

    SSaero = Ref.SSaero
    SSbeam = Ref.SSbeam

    # zeta in
    usta = np.concatenate([dzeta, dzeta_dot, du_ext])

    if hasattr(Ref, 'Asteady_inv'):
        #x_sta = A_steady^-1 dot B u_sta
        xsta = np.dot(Ref.Asteady_inv, np.dot(SSaero.B, usta))
    else:
        # x_sta = linsolve(A_steady, Bu)
        Asteady = np.eye(*SSaero.A.shape) - SSaero.A
        xsta = np.linalg.solve(Asteady, np.dot(SSaero.B, usta))

    # y_sta = C x_sta + D u_sta
    ysta = np.dot(SSaero.C, xsta) + np.dot(SSaero.D, usta)
    ftot_aero = Ref.ftot + np.dot(Ref.Kftot, ysta)
    mtot_aero = Ref.mtot + np.dot(Ref.Kmtot, ysta) + np.dot(Ref.Kmtot_disp, dzeta)

    #### beam in
    if solve_beam:
        # warning: we need to add first the contribution due to velocity change!!!
        usta_uinf = np.concatenate([0. * dzeta, 0. * dzeta_dot, du_ext])
        xsta_uinf = np.linalg.solve(Asteady, np.dot(SSaero.B, usta_uinf))
        ysta_uinf = np.dot(SSaero.C, xsta_uinf) + np.dot(SSaero.D, usta_uinf)
        usta = np.concatenate([dq, dqdot])
        if hasattr(Ref, 'Asteady_inv'):
            xsta = np.dot(Ref.Asteady_inv, np.dot(SSbeam.B, usta))
        else:
            Asteady = np.eye(*SSbeam.A.shape) - SSbeam.A
            xsta = np.linalg.solve(Asteady, np.dot(SSbeam.B, usta))
        ysta = ysta_uinf + np.dot(SSbeam.C, xsta) + np.dot(SSbeam.D, usta)
        ftot_beam = Ref.ftot + np.dot(Ref.Kftot, ysta)
        mtot_beam = Ref.mtot + np.dot(Ref.Kmtot, ysta) + np.dot(Ref.Kmtot_disp, dzeta_exp)
    else:
        ftot_beam, mtot_beam = None, None

    return ftot_aero, mtot_aero, ftot_beam, mtot_beam


def extract_from_data(data, assemble=True,
                      zeta_pole=np.zeros((3,)), build_Asteady_inv=False):
    """
    Extract relevant info from data structure. If assemble is True, it will
    also generate a linear UVLM and the displacements/velocities gain matrices
    """

    ### extract aero info - gets info from every panel in every surface
    # from an NxM matrix and reshapes it into an K by 1 column vector
    tsaero = data.aero.timestep_info[0]
    zeta = np.concatenate([tsaero.zeta[ss].reshape(-1, order='C')
                           for ss in range(tsaero.n_surf)])
    zeta_dot = np.concatenate([tsaero.zeta_dot[ss].reshape(-1, order='C')
                               for ss in range(tsaero.n_surf)])
    uext = np.concatenate([tsaero.u_ext[ss].reshape(-1, order='C')
                           for ss in range(tsaero.n_surf)])
    ftot, mtot = comp_tot_force(tsaero.forces, tsaero.zeta, zeta_pole=zeta_pole)

    ## TEST WHETHER SETTINGS RUN
    settings = dict()
    settings['LinearUvlm'] = {'dt': 0.1,
                              'integr_order': 2,
                              'density': 1.225,
                              'ScalingDict': {'length': 1.,
                                              'speed': 1.,
                                              'density': 1.}}

    ### extract structural info
    Sol = lin_aeroelastic.LinAeroEla(data, settings)
    gebm = Sol.lingebm_str
    q = Sol.q
    qdot = Sol.dq

    ### assemble
    if assemble is True:
        uvlm = Sol.linuvlm
        uvlm.assemble_ss()
        uvlm.get_total_forces_gain(zeta_pole=zeta_pole)
        Sol.get_gebm2uvlm_gains()
        Kas = np.block([[Sol.Kdisp, np.zeros((3 * uvlm.Kzeta, gebm.num_dof + 10))],
                        [Sol.Kvel_disp, Sol.Kvel_vel],
                        [np.zeros((3 * uvlm.Kzeta, 2 * gebm.num_dof + 20))]])
        SSbeam = libss.addGain(uvlm.SS, Kas, where='in')

        if build_Asteady_inv:
            Asteady_inv = np.linalg.inv(np.eye(*uvlm.SS.A.shape) - uvlm.SS.A)
        else:
            Asteady_inv = None
        Out = Info(zeta, zeta_dot, uext, ftot, mtot, q, qdot,
                   uvlm.SS, SSbeam, Kas, uvlm.Kftot, uvlm.Kmtot, uvlm.Kmtot_disp,
                   Asteady_inv)

    else:
        Out = Info(zeta, zeta_dot, uext, ftot, mtot, q, qdot)

    return Out