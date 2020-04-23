"""Linearised UVLM 2D tests


Test linear UVLM solver against analytical results for 2D wing

Author: S. Maraniello, Dec 2018
Modified: N. Goizueta, Sep 2019
"""

import sharpy.utils.sharpydir as sharpydir
import unittest
import os
# import matplotlib.pyplot as plt
import numpy as np
import shutil
import sharpy.sharpy_main
import sharpy.utils.algebra as algebra
import sharpy.utils.analytical as an
import sharpy.linear.src.linuvlm as linuvlm
import cases.templates.flying_wings as flying_wings
import sharpy.utils.sharpydir as sharpydir
import sharpy.utils.cout_utils as cout


class Test_infinite_span(unittest.TestCase):
    """
    Test infinite-span flat wing at zero incidence against analytical solutions
    """

    test_dir = sharpydir.SharpyDir + '/tests/linear/uvlm/'

    def setUp_from_params(self, Nsurf, integr_ord, RemovePred, UseSparse, RollNodes):
        """
        Builds SHARPy solution for a rolled infinite span flat wing at zero
        incidence. Rolling can be obtained both by rotating the FoR A or
        modifying the nodes of the wing.
        """

        # Flags
        self.ProducePlots = True

        # Define Parametrisation
        M, N, Mstar_fact = 8, 8, 50

        # Flying properties
        if RollNodes:
            self.Roll0Deg = 0.
        else:
            self.Roll0Deg = 0.
        self.Alpha0Deg = 0.0
        Uinf0 = 50.

        ### ----- build directories
        self.case_code = 'wagner'
        self.case_main = self.case_code + \
                         '_r%.4daeff%.2d_rnodes%s_Nsurf%.2dM%.2dN%.2dwk%.2d' \
                         % (int(np.round(100 * self.Roll0Deg)),
                            int(np.round(100 * self.Alpha0Deg)),
                            RollNodes, Nsurf, M, N, Mstar_fact)
        self.case_main += 'ord%.1d_rp%s_sp%s' % (integr_ord, RemovePred, UseSparse)
        self.route_test_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
        route_main = self.route_test_dir + '/res/'
        self.figfold = self.route_test_dir + '/figs/'

        if os.path.exists(route_main):
            shutil.rmtree(route_main)
        if os.path.exists(self.figfold):
            shutil.rmtree(self.figfold)

        os.makedirs(route_main)
        os.makedirs(self.figfold)

        ### ----- sharpy reference solution
        # Build wing model
        ws = flying_wings.QuasiInfinite(
            M=M, N=N, Mstar_fact=Mstar_fact, n_surfaces=Nsurf,
            u_inf=Uinf0, alpha=self.Alpha0Deg, roll=self.Roll0Deg,
            aspect_ratio=1e5, RollNodes=RollNodes,
            route=route_main, case_name=self.case_main)

        ws.main_ea = .4
        ws.clean_test_files()
        ws.update_derived_params()
        ws.generate_fem_file()
        ws.generate_aero_file()

        # solution flow
        ws.set_default_config_dict()
        ws.config['SHARPy']['flow'] = ['BeamLoader', 'AerogridLoader', 'StaticUvlm', 'BeamPlot', 'AerogridPlot']
        ws.config['SHARPy']['log_folder'] = self.route_test_dir + '/output/' + self.case_code + '/'
        ws.config['SHARPy']['write_screen'] = 'off'
        ws.config['SHARPy']['write_log'] = 'off'
        ws.config['LinearUvlm'] = {'dt': ws.dt,
                                   'integr_order': integr_ord,
                                   'density': ws.rho,
                                   'remove_predictor': RemovePred,
                                   'use_sparse': UseSparse,
                                   'ScalingDict': {'length': 1.,
                                                   'speed': 1.,
                                                   'density': 1.}}
        ws.config.write()

        # solve at linearistion point
        data0 = sharpy.sharpy_main.main(['...', route_main + self.case_main + '.sharpy'])
        tsaero0 = data0.aero.timestep_info[0]
        tsaero0.rho = ws.config['LinearUvlm']['density']

        ### ---- normalisation parameters
        self.start_writer()
        # verify chord
        c_ext = np.linalg.norm(tsaero0.zeta[0][:, 0, 0] - tsaero0.zeta[0][:, -1, 0])
        assert np.abs(ws.c_ref - c_ext) < 1e-8, 'Wrong reference chord'

        # reference force - total
        qinf = 0.5 * ws.rho * Uinf0 ** 2
        span = Nsurf * np.linalg.norm(tsaero0.zeta[0][:, 0, 0] - tsaero0.zeta[0][:, 0, -1])
        Stot = ws.c_ref * span
        Fref_tot = qinf * Stot

        # reference force - section
        sec_span = np.linalg.norm(tsaero0.zeta[0][:, 0, 0] - tsaero0.zeta[0][:, 0, 1])
        S = ws.c_ref * sec_span
        Fref_span = qinf * S

        # save
        self.route_main = route_main
        self.tsaero0 = tsaero0
        self.ws = ws
        self.Fref_tot = Fref_tot
        self.Fref_span = Fref_span
        self.M = M
        self.N = N
        self.Mstar_fact = Mstar_fact
        self.Uinf0 = Uinf0

    def test_wagner(self):
        """
        Step response (Wagner):
            - set linearisation point at 0 effective incidence but non-zero roll
                  attitude. This can be obtained either by rotating the FoR A or
                  by explicitely modifying the position of the wing nodes.
            - perturb. state so as to produce a small effective angle of attack.
                This is achieved combining changes of:
                        - wing lattice orientation
                        - wing lattice speed
                        - incoming flow orientation
            - compare aerodynamic force time history to Wagner's analytical solution
            - compare steady state to analytical solution and ``StaticUvlm`` solver

        Notes:
            The function uses ``subTests`` to call ``run_wagner``.
        """

        for Nsurf in [1, 2]:
            for integr_ord in [1, 2]:
                for RemovePred in [False, True]:
                    for UseSparse in [True]:
                        for RollNodes in [True, False]:
                            with self.subTest(
                                    Nsurf=Nsurf, integr_ord=integr_ord,
                                    RemovePred=RemovePred, UseSparse=UseSparse,
                                    RollNodes=RollNodes):
                                self.run_wagner(
                                    Nsurf, integr_ord, RemovePred, UseSparse, RollNodes)

    def run_wagner(self, Nsurf, integr_ord, RemovePred, UseSparse, RollNodes):
        """
        see test_wagner
        """

        ### ----- set reference solution
        self.setUp_from_params(Nsurf, integr_ord, RemovePred, UseSparse, RollNodes)
        tsaero0 = self.tsaero0
        ws = self.ws
        M = self.M
        N = self.N
        Mstar_fact = self.Mstar_fact
        Uinf0 = self.Uinf0

        ### ----- linearisation
        uvlm = linuvlm.Dynamic(tsaero0,
                               dynamic_settings=ws.config['LinearUvlm'])
        uvlm.assemble_ss()
        zeta_pole = np.array([0., 0., 0.])
        uvlm.get_total_forces_gain(zeta_pole=zeta_pole)
        uvlm.get_rigid_motion_gains(zeta_rotation=zeta_pole)
        uvlm.get_sect_forces_gain()

        ### ----- Scale gains
        Fref_tot = self.Fref_tot
        Fref_span = self.Fref_span
        uvlm.Kftot = uvlm.Kftot / Fref_tot
        uvlm.Kmtot = uvlm.Kmtot / Fref_tot / ws.c_ref
        uvlm.Kfsec /= Fref_span
        uvlm.Kmsec /= (Fref_span * ws.c_ref)

        ### ----- step input
        # rotate incoming flow, wing lattice and wing lattice speed about
        # the (rolled) wing elastic axis to create an effective angle of attack.
        # Rotation is expressed through a CRV.
        delta_AlphaEffDeg = 1e-2
        delta_AlphaEffRad = 1e-2 * np.pi / 180.

        Roll0Rad = self.Roll0Deg / 180. * np.pi
        dcrv = -delta_AlphaEffRad * np.array([0., np.cos(Roll0Rad), np.sin(Roll0Rad)])
        uvec0 = np.array([Uinf0, 0, 0])
        uvec = np.dot(algebra.crv2rotation(dcrv), uvec0)
        duvec = uvec - uvec0
        dzeta = np.zeros((Nsurf, 3, M + 1, N // Nsurf + 1))
        dzeta_dot = np.zeros((Nsurf, 3, M + 1, N // Nsurf + 1))
        du_ext = np.zeros((Nsurf, 3, M + 1, N // Nsurf + 1))

        for ss in range(Nsurf):
            for mm in range(M + 1):
                for nn in range(N // Nsurf + 1):
                    dzeta_dot[ss, :, mm, nn] = -1. / 3 * duvec
                    du_ext[ss, :, mm, nn] = +1. / 3 * duvec
                    dzeta = 1. / 3 * np.dot(uvlm.Krot, dcrv)
        Uaero = np.concatenate((dzeta.reshape(-1),
                                dzeta_dot.reshape(-1),
                                du_ext.reshape(-1)))

        ### ----- Steady state solution
        xste, yste = uvlm.solve_steady(Uaero)
        Ftot_ste = np.dot(uvlm.Kftot, yste)
        Mtot_ste = np.dot(uvlm.Kmtot, yste)
        # first check of gain matrices...
        Ftot_ste_ref = np.zeros((3,))
        Mtot_ste_ref = np.zeros((3,))
        fnodes = yste.reshape((Nsurf, 3, M + 1, N // Nsurf + 1))
        for ss in range(Nsurf):
            for nn in range(N // Nsurf + 1):
                for mm in range(M + 1):
                    Ftot_ste_ref += fnodes[ss, :, mm, nn]
                    Mtot_ste_ref += np.cross(
                        uvlm.MS.Surfs[ss].zeta[:, mm, nn], fnodes[ss, :, mm, nn])
        Ftot_ste_ref /= Fref_tot
        Mtot_ste_ref /= (Fref_tot * ws.c_ref)
        Fmag = np.linalg.norm(Ftot_ste_ref)
        er_f = np.max(np.abs(Ftot_ste - Ftot_ste_ref)) / Fmag
        er_m = np.max(np.abs(Mtot_ste - Mtot_ste_ref)) / Fmag / ws.c_ref
        assert (er_f < 1e-8 and er_m < 1e-8), \
            'Error of total forces (%.2e) and moment (%.2e) too large!' % (er_f, er_m) + \
            'Verify gains produced by linuvlm.Dynamic.get_total_forces_gain.'

        # then compare against analytical ...
        Cl_inf = delta_AlphaEffRad * np.pi * 2.
        Cfvec_inf = Cl_inf * np.array([0., -np.sin(Roll0Rad), np.cos(Roll0Rad)])
        er_f = np.abs(np.linalg.norm(Ftot_ste) / Cl_inf - 1.)
        assert (er_f < 1e-2), \
            'Error of total lift coefficient (%.2e) too large!' % (er_f,) + \
            'Verify linuvlm.Dynamic.'
        er_f = np.abs(np.linalg.norm(Ftot_ste - Cfvec_inf) / Cl_inf)
        assert (er_f < 1e-2), \
            'Error of total aero force (%.2e) too large!' % (er_f,) + \
            'Verify linuvlm.Dynamic.'

        # ... and finally compare against non-linear UVLM
        # ps: here we roll the wing and rotate the incoming flow to generate an effective
        # angle of attack
        case_pert = 'wagner_r%.4daeff%.2d_rnodes%s_Nsurf%.2dM%.2dN%.2dwk%.2d' \
                    % (int(np.round(100 * self.Roll0Deg)),
                       int(np.round(100 * delta_AlphaEffDeg)),
                       RollNodes,
                       Nsurf, M, N, Mstar_fact)
        ws_pert = flying_wings.QuasiInfinite(
            M=M, N=N, Mstar_fact=Mstar_fact, n_surfaces=Nsurf,
            u_inf=Uinf0,
            alpha=self.Alpha0Deg,
            roll=self.Roll0Deg,
            aspect_ratio=1e5,
            route=self.route_main,
            case_name=case_pert,
            RollNodes=RollNodes)
        ws_pert.u_inf_direction = uvec / Uinf0
        ws_pert.main_ea = ws.main_ea
        ws_pert.clean_test_files()
        ws_pert.update_derived_params()
        ws_pert.generate_fem_file()
        ws_pert.generate_aero_file()

        # solution flow
        ws_pert.set_default_config_dict()
        ws_pert.config['SHARPy']['flow'] = ws.config['SHARPy']['flow']
        ws_pert.config['SHARPy']['write_screen'] = 'off'
        ws_pert.config['SHARPy']['write_log'] = 'off'
        ws_pert.config['SHARPy']['log_folder'] = self.route_test_dir + '/output/' + self.case_code + '/'
        ws_pert.config.write()

        # solve at perturbed point
        data_pert = sharpy.sharpy_main.main(['...', self.route_main + case_pert + '.sharpy'])
        tsaero = data_pert.aero.timestep_info[0]

        self.start_writer()

        # get total forces
        Ftot_ste_pert = np.zeros((3,))
        Mtot_ste_pert = np.zeros((3,))
        for ss in range(Nsurf):
            for nn in range(N // Nsurf + 1):
                for mm in range(M + 1):
                    Ftot_ste_pert += tsaero.forces[ss][:3, mm, nn]
                    Mtot_ste_pert += np.cross(
                        uvlm.MS.Surfs[ss].zeta[:, mm, nn], tsaero.forces[ss][:3, mm, nn])
        Ftot_ste_pert /= Fref_tot
        Mtot_ste_pert /= (Fref_tot * ws.c_ref)
        Fmag = np.linalg.norm(Ftot_ste_pert)
        er_f = np.max(np.abs(Ftot_ste - Ftot_ste_pert)) / Fmag
        er_m = np.max(np.abs(Mtot_ste - Mtot_ste_pert)) / Fmag / ws.c_ref
        assert (er_f < 2e-4 and er_m < 2e-4), \
            'Error of total forces (%.2e) and moment (%.2e) ' % (er_f, er_m) + \
            'with respect to geometrically-exact UVLM too large!'

        # and check non-linear uvlm against analytical solution
        er_f = np.abs(np.linalg.norm(Ftot_ste_pert - Cfvec_inf) / Cl_inf)
        assert (er_f <= 1.5e-2), \
            'Error of total aero force components (%.2e) too large!' % (er_f,) + \
            'Verify StaticUvlm'

        ### ----- Analytical step response (Wagner solution)

        NT = 251
        tv = np.linspace(0., uvlm.dt * (NT - 1), NT)
        Clv_an = an.wagner_imp_start(delta_AlphaEffRad, Uinf0, ws.c_ref, tv)
        assert np.abs(Clv_an[-1] / Cl_inf - 1.) < 1e-2, \
            'Did someone modify this test case?! The time should be enough to reach ' \
            'the steady-state CL with a 1 perc. tolerance...'

        Cfvec_an = np.zeros((NT, 3))
        Cfvec_an[:, 1] = -np.sin(Roll0Rad) * Clv_an
        Cfvec_an[:, 2] = np.cos(Roll0Rad) * Clv_an

        ### ----- Dynamic step response

        Fsect = np.zeros((NT, Nsurf, 3, N // Nsurf + 1))
        # Fbeam=np.zeros((NT,6,N//Nsurf+1))
        Ftot = np.zeros((NT, 3))
        Er_f_tot = np.zeros((NT,))

        # Ybeam=[]
        gamma = np.zeros((NT, Nsurf, M, N // Nsurf))
        gamma_dot = np.zeros((NT, Nsurf, M, N // Nsurf))
        gamma_star = np.zeros((NT, Nsurf, int(M * Mstar_fact), N // Nsurf))
        xold = np.zeros((uvlm.SS.A.shape[0],))
        for tt in range(1, NT):
            xnew, ynew = uvlm.solve_step(xold, Uaero)
            change = np.linalg.norm(xnew - xold)
            xold = xnew

            # record state ?
            if uvlm.remove_predictor is False:
                gv, gvstar, gvdot = uvlm.unpack_state(xnew)
                gamma[tt, :, :, :] = gv.reshape((Nsurf, M, N // Nsurf))
                gamma_dot[tt, :, :, :] = gvdot.reshape((Nsurf, M, N // Nsurf))
                gamma_star[tt, :, :, :] = gvstar.reshape((Nsurf, int(M * Mstar_fact), N // Nsurf))

            # calculate forces (and error)
            Ftot[tt, :3] = np.dot(uvlm.Kftot, ynew)
            Er_f_tot[tt] = np.linalg.norm(Ftot[tt, :] - Cfvec_an[tt, :]) / Clv_an[tt]
            Fsect[tt, :, :, :] = np.dot(uvlm.Kfsec, ynew).reshape((Nsurf, 3, N // Nsurf + 1))

            # ### beam forces
            # Ybeam.append(np.dot(Sol.Kforces[:-10,:],ynew))
            # Fdummy=Ybeam[-1].reshape((N//Nsurf,6)).T
            # Fbeam[tt,:,:N//Nsurf]=Fdummy[:,:N//Nsurf]
            # Fbeam[tt,:,N//Nsurf+1:]=Fdummy[:,N//Nsurf:]

        if RemovePred:
            ts2perc, ts1perc = 6, 6
        else:
            ts2perc, ts1perc = 16, 36
        er_th_2perc = np.max(Er_f_tot[ts2perc:])
        er_th_1perc = np.max(Er_f_tot[ts1perc:])

        ### ----- generate plot
        if self.ProducePlots:

            # sections to plot
            if Nsurf == 1:
                Nplot = [0, N // 2, N]
                labs = [r'tip', r'root', r'tip']
            elif Nsurf == 2:
                Nplot = [0, N // 2 - 1]
                labs = [r'tip', r'near root', r'tip', r'near root']
            axtitle = [r'$C_{F_y}$', r'$C_{F_z}$']

            # non-dimensional time
            sv = 2.0 * Uinf0 * tv / ws.c_ref

            # generate figure
            clist = ['#003366', '#CC3333', '#336633', '#FF6600'] * 4
            fontlabel = 12
            std_params = {'legend.fontsize': 10,
                          'font.size': fontlabel,
                          'xtick.labelsize': fontlabel - 2,
                          'ytick.labelsize': fontlabel - 2,
                          'figure.autolayout': True,
                          'legend.numpoints': 1}
            # plt.rcParams.update(std_params)

            # fig = plt.figure('Lift time-history', (12, 6))
            # axvec = fig.subplots(1, 2)
            # for aa in [0, 1]:
                # comp = aa + 1
                # axvec[aa].set_title(axtitle[aa])
                # axvec[aa].plot(sv, Cfvec_an[:, comp] / Cl_inf, lw=4, ls='-',
                               # alpha=0.5, color='r', label=r'Wagner')
                # axvec[aa].plot(sv, Ftot[:, comp] / Cl_inf, lw=5, ls=':',
                               # alpha=0.7, color='k', label=r'Total')
                # cc = 0
                # for ss in range(Nsurf):
                    # for nn in Nplot:
                        # axvec[aa].plot(sv, Fsect[:, ss, comp, nn] / Cl_inf,
                                       # lw=4 - cc, ls='--', alpha=0.7, color=clist[cc],
                                       # label=r'Surf. %.1d, n=%.2d (%s)' % (ss, nn, labs[cc]))
                        # cc += 1
                # axvec[aa].grid(color='0.8', ls='-')
                # axvec[aa].grid(color='0.85', ls='-', which='minor')
                # axvec[aa].set_xlabel(r'normalised time $t=2 U_\infty \tilde{t}/c$')
                # axvec[aa].set_ylabel(axtitle[aa] + r'$/C_{l_\infty}$')
                # axvec[aa].set_xlim(0, sv[-1])
                # if Cfvec_inf[comp] > 0.:
                    # axvec[aa].set_ylim(0, 1.1)
                # else:
                    # axvec[aa].set_ylim(-1.1, 0)
            # plt.legend(ncol=1)
            # # plt.show()
            # fig.savefig(self.figfold + self.case_main + '.png')
            # fig.savefig(self.figfold + self.case_main + '.pdf')
            # plt.close()

        assert er_th_2perc < 2e-2 and er_th_1perc < 1e-2, \
            'Error of dynamic step response at time-steps 16 and 36 ' + \
            '(%.2e and %.2e) too large. Verify Linear UVLM.' % (er_th_2perc, er_th_1perc)

    def start_writer(self):
        # Over write writer with print_file False to avoid I/O errors
        global cout_wrap
        cout_wrap = cout.Writer()
        # cout_wrap.initialise(print_screen=False, print_file=False)
        cout_wrap.cout_quiet()
        sharpy.utils.cout_utils.cout_wrap = cout_wrap

    def tearDown(self):
        cout.finish_writer()
        try:
            shutil.rmtree(self.route_test_dir + '/res/')
            shutil.rmtree(self.route_test_dir + '/figs/')
            shutil.rmtree(self.route_test_dir + '/output/')
        except FileNotFoundError:
            pass


if __name__ == '__main__':
    if os.path.exists('./figs/infinite_span'):
        shutil.rmtree('./figs/infinite_span')
    unittest.main()
