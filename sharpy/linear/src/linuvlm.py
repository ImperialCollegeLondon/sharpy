"""
Linear UVLM solver classes

Contains classes to assemble a linear UVLM system. The three main classes are:

* :class:`~sharpy.linear.src.linuvlm.Static`: : for static VLM solutions.

* :class:`~sharpy.linear.src.linuvlm.Dynamic`: for dynamic UVLM solutions.

* :class:`~sharpy.linear.src.linuvlm.DynamicBlock`: a more efficient representation of ``Dynamic`` using lists for the
  different blocks in the UVLM equations

References:

    Maraniello, S., & Palacios, R.. State-Space Realizations and Internal Balancing in Potential-Flow
    Aerodynamics with Arbitrary Kinematics. AIAA Journal, 57(6), 1–14. 2019. https://doi.org/10.2514/1.J058153

"""

import time
import warnings
import numpy as np
import scipy.linalg as scalg
import scipy.sparse as sparse

import sharpy.linear.src.interp as interp
import sharpy.linear.src.multisurfaces as multisurfaces
import sharpy.linear.src.assembly as ass
import sharpy.linear.src.libss as libss

import sharpy.linear.src.libsparse as libsp
import sharpy.rom.utils.librom as librom
import sharpy.utils.algebra as algebra
import sharpy.utils.settings as settings
import sharpy.utils.cout_utils as cout
import sharpy.utils.exceptions as exceptions
from sharpy.utils.constants import vortex_radius_def
from sharpy.linear.utils.ss_interface import LinearVector, StateVariable, InputVariable, OutputVariable

settings_types_static = dict()
settings_default_static = dict()

settings_types_static['vortex_radius'] = 'float'
settings_default_static['vortex_radius'] = vortex_radius_def

settings_types_static['cfl1'] = 'bool'
settings_default_static['cfl1'] = True

settings_types_dynamic = dict()
settings_default_dynamic = dict()

settings_types_dynamic['dt'] = 'float'
settings_default_dynamic['dt'] = 0.1

settings_types_dynamic['integr_order'] = 'int'
settings_default_dynamic['integr_order'] = 2

settings_types_dynamic['density'] = 'float'
settings_default_dynamic['density'] = 1.225

settings_types_dynamic['ScalingDict'] = 'dict'
settings_default_dynamic['ScalingDict'] = {'length': 1.0,
                                           'speed': 1.0,
                                           'density': 1.0}

settings_types_dynamic['remove_predictor'] = 'bool'
settings_default_dynamic['remove_predictor'] = True

settings_types_dynamic['use_sparse'] = 'bool'
settings_default_dynamic['use_sparse'] = True

settings_types_dynamic['vortex_radius'] = 'float'
settings_default_dynamic['vortex_radius'] = vortex_radius_def

settings_types_dynamic['cfl1'] = 'bool'
settings_default_dynamic['cfl1'] = True


class Static():
    """	Static linear solver """

    def __init__(self, tsdata, custom_settings=None, for_vel=np.zeros((6,))):

        cout.cout_wrap('Initialising Static linear UVLM solver class...')
        t0 = time.time()

        if custom_settings is None:
            settings_here = settings_default_static
        else:
            settings_here = custom_settings

        settings.to_custom_types(settings_here,
                                 settings_types_static,
                                 settings_default_static,
                                 no_ctype=True)

        self.vortex_radius = settings_here['vortex_radius']
        self.cfl1 = settings_here['cfl1']
        MS = multisurfaces.MultiAeroGridSurfaces(tsdata,
                                                 self.vortex_radius,
                                                 for_vel=for_vel)
        MS.get_ind_velocities_at_collocation_points()
        MS.get_input_velocities_at_collocation_points()
        MS.get_ind_velocities_at_segments()
        MS.get_input_velocities_at_segments()

        # define total sizes
        self.K = sum(MS.KK)
        self.K_star = sum(MS.KK_star)
        self.Kzeta = sum(MS.KKzeta)
        self.Kzeta_star = sum(MS.KKzeta_star)
        self.MS = MS

        # define input perturbation
        self.zeta = np.zeros((3 * self.Kzeta))
        self.zeta_dot = np.zeros((3 * self.Kzeta))
        self.u_ext = np.zeros((3 * self.Kzeta))

        self.input_variables_list = [InputVariable('zeta', size=3 * self.Kzeta, index=0),
                                     InputVariable('zeta_dot', size=3 * self.Kzeta, index=1),
                                     InputVariable('u_gust', size=3 * self.Kzeta, index=2)]

        self.state_variables_list = [StateVariable('gamma', size=self.K, index=0),
                                     StateVariable('gamma_w', size=self.K_star, index=1),
                                     StateVariable('gamma_m1', size=self.K, index=2)]

        self.output_variables_list = [OutputVariable('forces_v', size=3 * self.Kzeta, index=0)]

        # profiling output
        self.prof_out = './asbly.prof'

        self.time_init_sta = time.time() - t0
        cout.cout_wrap('\t\t\t...done in %.2f sec' % self.time_init_sta)

    def assemble_profiling(self):
        """
        Generate profiling report for assembly and save it in self.prof_out.

        To read the report:
            import pstats
            p=pstats.Stats(self.prof_out)
        """

        import cProfile
        cProfile.runctx('self.assemble()', globals(), locals(), filename=self.prof_out)

    def assemble(self):
        """
        Assemble global matrices
        """
        cout.cout_wrap('\tAssembly of static linear UVLM equations started...', 1)
        MS = self.MS
        t0 = time.time()

        # ----------------------------------------------------------- state eq.
        List_uc_dncdzeta = ass.uc_dncdzeta(MS.Surfs)
        List_nc_dqcdzeta_coll, List_nc_dqcdzeta_vert = \
            ass.nc_dqcdzeta(MS.Surfs, MS.Surfs_star)
        List_AICs, List_AICs_star = ass.AICs(MS.Surfs, MS.Surfs_star,
                                             target='collocation', Project=True)
        List_Wnv = []
        for ss in range(MS.n_surf):
            List_Wnv.append(
                interp.get_Wnv_vector(MS.Surfs[ss],
                                      MS.Surfs[ss].aM, MS.Surfs[ss].aN))

        ### zeta derivatives
        self.Ducdzeta = np.block(List_nc_dqcdzeta_vert)
        del List_nc_dqcdzeta_vert
        self.Ducdzeta += scalg.block_diag(*List_nc_dqcdzeta_coll)
        del List_nc_dqcdzeta_coll
        self.Ducdzeta += scalg.block_diag(*List_uc_dncdzeta)
        del List_uc_dncdzeta
        # # omega x zeta terms
        List_nc_domegazetadzeta_vert = ass.nc_domegazetadzeta(MS.Surfs, MS.Surfs_star)
        self.Ducdzeta += scalg.block_diag(*List_nc_domegazetadzeta_vert)
        del List_nc_domegazetadzeta_vert

        ### input velocity derivatives
        self.Ducdu_ext = scalg.block_diag(*List_Wnv)
        del List_Wnv

        ### Condense Gammaw terms
        for ss_out in range(MS.n_surf):
            K = MS.KK[ss_out]
            for ss_in in range(MS.n_surf):
                N_star = MS.NN_star[ss_in]
                aic = List_AICs[ss_out][ss_in]  # bound
                aic_star = List_AICs_star[ss_out][ss_in]  # wake

                # fold aic_star: sum along chord at each span-coordinate
                aic_star_fold = np.zeros((K, N_star))
                for jj in range(N_star):
                    aic_star_fold[:, jj] += np.sum(aic_star[:, jj::N_star], axis=1)
                aic[:, -N_star:] += aic_star_fold

        self.AIC = np.block(List_AICs)

        # ---------------------------------------------------------- output eq.

        ### Zeta derivatives
        # ... at constant relative velocity
        self.Dfqsdzeta = scalg.block_diag(
            *ass.dfqsdzeta_vrel0(MS.Surfs, MS.Surfs_star))
        # ... induced velocity contrib.
        List_coll, List_vert = ass.dfqsdvind_zeta(MS.Surfs, MS.Surfs_star)
        for ss in range(MS.n_surf):
            List_vert[ss][ss] += List_coll[ss]
        self.Dfqsdzeta += np.block(List_vert)
        del List_vert, List_coll

        ### Input velocities
        self.Dfqsdu_ext = scalg.block_diag(
            *ass.dfqsduinput(MS.Surfs, MS.Surfs_star))

        ### Gamma derivatives
        # ... at constant relative velocity
        List_dfqsdgamma_vrel0, List_dfqsdgamma_star_vrel0 = \
            ass.dfqsdgamma_vrel0(MS.Surfs, MS.Surfs_star)
        self.Dfqsdgamma = scalg.block_diag(*List_dfqsdgamma_vrel0)
        self.Dfqsdgamma_star = scalg.block_diag(*List_dfqsdgamma_star_vrel0)
        del List_dfqsdgamma_vrel0, List_dfqsdgamma_star_vrel0
        # ... induced velocity contrib.
        List_dfqsdvind_gamma, List_dfqsdvind_gamma_star = \
            ass.dfqsdvind_gamma(MS.Surfs, MS.Surfs_star)
        self.Dfqsdgamma += np.block(List_dfqsdvind_gamma)
        self.Dfqsdgamma_star += np.block(List_dfqsdvind_gamma_star)
        del List_dfqsdvind_gamma, List_dfqsdvind_gamma_star

        self.time_asbly = time.time() - t0
        cout.cout_wrap('\t\t\t...done in %.2f sec' % self.time_asbly, 1)

    def solve(self):
        r"""
        Solve for bound :math:`\\Gamma` using the equation;

        .. math::
                \\mathcal{A}(\\Gamma^n) = u^n

        # ... at constant rotation speed
        ``self.Dfqsdzeta+=scalg.block_diag(*ass.dfqsdzeta_omega(MS.Surfs,MS.Surfs_star))``

        """

        MS = self.MS
        t0 = time.time()

        ### state
        bv = np.dot(self.Ducdu_ext, self.u_ext - self.zeta_dot) + \
             np.dot(self.Ducdzeta, self.zeta)
        self.gamma = np.linalg.solve(self.AIC, -bv)

        ### retrieve gamma over wake
        gamma_star = []
        Ktot = 0
        for ss in range(MS.n_surf):
            Ktot += MS.Surfs[ss].maps.K
            N = MS.Surfs[ss].maps.N
            Mstar = MS.Surfs_star[ss].maps.M
            gamma_star.append(np.concatenate(Mstar * [self.gamma[Ktot - N:Ktot]]))
        gamma_star = np.concatenate(gamma_star)

        ### compute steady force
        self.fqs = np.dot(self.Dfqsdgamma, self.gamma) + \
                   np.dot(self.Dfqsdgamma_star, gamma_star) + \
                   np.dot(self.Dfqsdzeta, self.zeta) + \
                   np.dot(self.Dfqsdu_ext, self.u_ext - self.zeta_dot)

        self.time_sol = time.time() - t0
        cout.cout_wrap('\tSolution done in %.2f sec' % self.time_sol, 1)

    def reshape(self):
        """
        Reshapes state/output according to SHARPy format
        """

        MS = self.MS
        if not hasattr(self, 'gamma') or not hasattr(self, 'fqs'):
            raise NameError('State and output not found')

        self.Gamma = []
        self.Fqs = []

        Ktot, Kzeta_tot = 0, 0
        for ss in range(MS.n_surf):
            M, N = MS.Surfs[ss].maps.M, MS.Surfs[ss].maps.N
            K, Kzeta = MS.Surfs[ss].maps.K, MS.Surfs[ss].maps.Kzeta

            iivec = range(Ktot, Ktot + K)
            self.Gamma.append(self.gamma[iivec].reshape((M, N)))

            iivec = range(Kzeta_tot, Kzeta_tot + 3 * Kzeta)
            self.Fqs.append(self.fqs[iivec].reshape((3, M + 1, N + 1)))

            Ktot += K
            Kzeta_tot += 3 * Kzeta

    def total_forces(self, zeta_pole=np.zeros((3,))):
        """
        Calculates total force (Ftot) and moment (Mtot) (about pole zeta_pole).
        """

        if not hasattr(self, 'Gamma') or not hasattr(self, 'Fqs'):
            self.reshape()

        self.Ftot = np.zeros((3,))
        self.Mtot = np.zeros((3,))

        for ss in range(self.MS.n_surf):
            M, N = self.MS.Surfs[ss].maps.M, self.MS.Surfs[ss].maps.N
            for nn in range(N + 1):
                for mm in range(M + 1):
                    zeta_node = self.MS.Surfs[ss].zeta[:, mm, nn]
                    fnode = self.Fqs[ss][:, mm, nn]
                    self.Ftot += fnode
                    self.Mtot += np.cross(zeta_node - zeta_pole, fnode)
        # for cc in range(3):
        # 	self.Ftot[cc]+=np.sum(self.Fqs[ss][cc,:,:])

    def get_total_forces_gain(self, zeta_pole=np.zeros((3,))):
        r"""
        Calculates gain matrices to calculate the total force (Kftot) and moment (Kmtot, Kmtot_disp) about the
        pole zeta_pole.

        Being :math:`f` and :math:`\zeta` the force and position at the vertex (m,n) of the lattice
        these are produced as:

            * ``ftot=sum(f) -> dftot += df``
            * ``mtot-sum((zeta-zeta_pole) x f) ->	dmtot +=  cross(zeta0-zeta_pole) df - cross(f0) dzeta``

        """

        self.Kftot = np.zeros((3, 3 * self.Kzeta))
        self.Kmtot = np.zeros((3, 3 * self.Kzeta))
        self.Kmtot_disp = np.zeros((3, 3 * self.Kzeta))

        Kzeta_start = 0
        for ss in range(self.MS.n_surf):

            M, N = self.MS.Surfs[ss].maps.M, self.MS.Surfs[ss].maps.N

            for nn in range(N + 1):
                for mm in range(M + 1):
                    jjvec = [Kzeta_start + np.ravel_multi_index((cc, mm, nn),
                                                                (3, M + 1, N + 1)) for cc in range(3)]

                    self.Kftot[[0, 1, 2], jjvec] = 1.
                    self.Kmtot[np.ix_([0, 1, 2], jjvec)] = algebra.skew(
                        self.MS.Surfs[ss].zeta[:, mm, nn] - zeta_pole)
                    self.Kmtot_disp[np.ix_([0, 1, 2], jjvec)] = algebra.skew(
                        -self.MS.Surfs[ss].fqs[:, mm, nn])

            Kzeta_start += 3 * self.MS.KKzeta[ss]

    def get_sect_forces_gain(self):
        """
        Gains to computes sectional forces. Moments are computed w.r.t.
        mid-vertex (chord-wise index M/2) of each section.
        """

        Ntot = 0
        for ss in range(self.MS.n_surf):
            Ntot += self.MS.NN[ss] + 1
        self.Kfsec = np.zeros((3 * Ntot, 3 * self.Kzeta))
        self.Kmsec = np.zeros((3 * Ntot, 3 * self.Kzeta))

        Kzeta_start = 0
        II_start = 0
        for ss in range(self.MS.n_surf):
            M, N = self.MS.MM[ss], self.MS.NN[ss]

            for nn in range(N + 1):
                zeta_sec = self.MS.Surfs[ss].zeta[:, :, nn]

                # section indices
                iivec = [II_start + np.ravel_multi_index((cc, nn),
                                                         (3, N + 1)) for cc in range(3)]
                # iivec = [II_start + cc+6*nn for cc in range(6)]
                # iivec = [II_start + cc*(N+1) + nn for cc in range(6)]

                for mm in range(M + 1):
                    # vertex indices
                    jjvec = [Kzeta_start + np.ravel_multi_index((cc, mm, nn),
                                                                (3, M + 1, N + 1)) for cc in range(3)]

                    # sectional force
                    self.Kfsec[iivec, jjvec] = 1.0

                    # sectional moment
                    dx, dy, dz = zeta_sec[:, mm] - zeta_sec[:, M // 2]
                    self.Kmsec[np.ix_(iivec, jjvec)] = np.array([[0, -dz, dy],
                                                                 [dz, 0, -dx],
                                                                 [-dy, dx, 0]])
            Kzeta_start += 3 * self.MS.KKzeta[ss]
            II_start += 3 * (N + 1)

    def get_rigid_motion_gains(self, zeta_rotation=np.zeros((3,))):
        """
        Gains to reproduce rigid-body motion such that grid displacements and
        velocities are given by:

            * ``dzeta     = Ktra*u_tra         + Krot*u_rot``
            * ``dzeta_dot = Ktra_vel*u_tra_dot + Krot*u_rot_dot``

        Rotations are assumed to happen independently with respect to the
        zeta_rotation point and about the x,y and z axes of the inertial frame.
        """

        # warnings.warn('Rigid rotation matrix not implemented!')

        Ntot = 0
        for ss in range(self.MS.n_surf):
            Ntot += self.MS.NN[ss] + 1
        self.Ktra = np.zeros((3 * self.Kzeta, 3))
        self.Ktra_dot = np.zeros((3 * self.Kzeta, 3))
        self.Krot = np.zeros((3 * self.Kzeta, 3))
        self.Krot_dot = np.zeros((3 * self.Kzeta, 3))

        Kzeta_start = 0
        for ss in range(self.MS.n_surf):
            M, N = self.MS.MM[ss], self.MS.NN[ss]
            zeta = self.MS.Surfs[ss].zeta

            for nn in range(N + 1):
                for mm in range(M + 1):
                    # vertex indices
                    iivec = [Kzeta_start + np.ravel_multi_index((cc, mm, nn),
                                                                (3, M + 1, N + 1)) for cc in range(3)]

                    self.Ktra[iivec, [0, 1, 2]] += 1.
                    self.Ktra_dot[iivec, [0, 1, 2]] += 1.

                    # sectional moment
                    dx, dy, dz = zeta[:, mm, nn] - zeta_rotation
                    Dskew = np.array([[0, -dz, dy], [dz, 0, -dx], [-dy, dx, 0]])
                    self.Krot[iivec, :] = Dskew
                    self.Krot_dot[iivec, :] = Dskew
            Kzeta_start += 3 * self.MS.KKzeta[ss]


# # ------------------------------------------------------------------------------
# # utilities for Dynamic.balfreq method

# def get_trapz_weights(k0,kend,Nk,knyq=False):
#     """
#     Returns uniform frequency grid (kv of length Nk) and weights (wv) for
#     Gramians integration using trapezoidal rule. If knyq is True, it is assumed
#     that kend is also the Nyquist frequency.
#     """

#     assert k0>=0. and kend>=0., 'Frequencies must be positive!'

#     dk=(kend-k0)/(Nk-1.)
#     kv=np.linspace(k0,kend,Nk)
#     wv=np.ones((Nk,))*dk*np.sqrt(2)

#     if k0/(kend-k0)<1e-10:
#         wv[0]=.5*dk
#     else:
#         wv[0]=dk/np.sqrt(2)

#     if knyq:
#         wv[-1]=.5*dk
#     else:
#         wv[-1]=dk/np.sqrt(2)

#     return kv,wv


# def get_gauss_weights(k0,kend,Npart,order):
#     """
#     Returns gauss-legendre frequency grid (kv of length Npart*order) and
#     weights (wv) for Gramians integration.

#     The integration grid is divided into Npart partitions, and in each of
#     them integration is performed using a Gauss-Legendre quadrature of
#     order order.

#     Note: integration points are never located at k0 or kend, hence there
#     is no need for special treatment as in (for e.g.) a uniform grid case
#     (see get_unif_weights)
#     """

#     if Npart==1:
#         # get gauss normalised coords and weights
#         xad,wad=np.polynomial.legendre.leggauss(order)
#         krange=kend-k0
#         kv=.5*(k0+kend) + .5*krange*xad
#         wv=wad*(.5*krange)*np.sqrt(2)
#         print('partitioning: %.3f to %.3f' %(k0,kend) )

#     else:
#         kv=np.zeros((Npart*order,))
#         wv=np.zeros((Npart*order,))

#         dk_part=(kend-k0)/Npart

#         for ii in range(Npart):
#             k0_part=k0+ii*dk_part
#             kend_part=k0_part+dk_part
#             iivec=range(order*ii, order*(ii+1))
#             kv[iivec],wv[iivec]=get_gauss_weights(k0_part,kend_part,Npart=1,order=order)

#     return kv,wv


# ------------------------------------------------------------------------------


class Dynamic(Static):
    r"""
    Class for dynamic linearised UVLM solution. Linearisation around steady-state
    are only supported. The class is built upon Static, and inherits all the
    methods contained there.

    Input:
        - tsdata: aero timestep data from SHARPy solution
        - dt: time-step
        - integr_order=2: integration order for UVLM unsteady aerodynamic force
        - RemovePredictor=True: if true, the state-space model is modified so as
          to accept in input perturbations, u, evaluated at time-step n rather than
          n+1.
        - ScalingDict=None: disctionary containing fundamental reference units:

            .. code-block:: python

               {'length':  reference_length,
               'speed':   reference_speed,
               'density': reference density}

          used to derive scaling quantities for the state-space model variables.
          The scaling factors are stored in ``self.ScalingFact``.

          Note that while time, circulation, angular speeds) are scaled
          accordingly, FORCES ARE NOT. These scale by :math:`q_\infty b^2`, where :math:`b` is the
          reference length and :math:`q_\infty` is the dynamic pressure.

        - UseSparse=False: builds the A and B matrices in sparse form. C and D
          are dense anyway so the sparse format cannot be applied to them.

    Methods:
        - nondimss: normalises a dimensional state-space model based on the
          scaling factors in self.ScalingFact.
        - dimss: inverse of nondimss.
        - assemble_ss: builds state-space model. See function for more details.
        - assemble_ss_profiling: generate profiling report of the assembly and
          saves it into self.prof_out. To read the report:

            .. code-block:: python

                import pstats
                p = pstats.Stats(self.prof_out)

        - solve_steady: solves for the steady state. Several methods available.
        - solve_step: solves one time-step
        - freqresp: ad-hoc method for fast frequency response (only implemented) for ``remove_predictor=False``

    Attributes:
        Nx (int): Number of states
        Nu (int): Number of inputs
        Ny (int): Number of outputs
        K (int): Number of paneles :math:`K = MN`
        K_star (int): Number of wake panels :math:`K^*=M^*N`
        Kzeta (int): Number of panel vertices :math:`K_\zeta=(M+1)(N+1)`
        Kzeta_star (int): Number of wake panel vertices :math:`K_{\zeta,w} = (M^*+1)(N+1)`

    To do:
    Upgrade to linearise around unsteady snapshot (adjoint)
    """

    def __init__(self, tsdata, dt=None, dynamic_settings=None, integr_order=2,
                 RemovePredictor=True, ScalingDict=None, UseSparse=True, for_vel=np.zeros((6,))):

        # Transform settings dictionary - in the future remove remaining inputs
        self.settings = dict()
        if dynamic_settings:
            self.settings = dynamic_settings
            settings.to_custom_types(self.settings,
                                     settings_types_dynamic,
                                     settings_default_dynamic,
                                     no_ctype=True)
        else:
            warnings.warn('No settings dictionary found. Using default. Individual parsing of settings is deprecated',
                          DeprecationWarning)
            # Future: remove deprecation warning and make settings the only argument
            settings.to_custom_types(self.settings,
                                     settings_types_dynamic,
                                     settings_default_dynamic,
                                     no_ctype=True)
            self.settings['dt'] = dt
            self.settings['integr_order'] = integr_order
            self.settings['remove_predictor'] = RemovePredictor
            self.settings['use_sparse'] = UseSparse
            self.settings['ScalingDict'] = ScalingDict

        static_dict = {'vortex_radius': self.settings['vortex_radius'],
                       'cfl1': self.settings['cfl1']}
        super().__init__(tsdata, custom_settings=static_dict, for_vel=for_vel)

        self.dt = self.settings['dt']
        self.integr_order = self.settings['integr_order']
        self.vortex_radius = self.settings['vortex_radius']
        self.cfl1 = self.settings['cfl1']

        if self.integr_order == 1:
            Nx = 2 * self.K + self.K_star
        elif self.integr_order == 2:
            Nx = 3 * self.K + self.K_star
            # b0, bm1, bp1 = -2., 0.5, 1.5
        else:
            raise NameError('Only integration orders 1 and 2 are supported')
        Ny = 3 * self.Kzeta
        Nu = 3 * Ny
        self.Nx = Nx
        self.Nu = Nu
        self.Ny = Ny

        self.remove_predictor = self.settings['remove_predictor']
        # Stores original B matrix for state recovery later
        self.B_predictor = None
        self.D_predictor = None

        self.include_added_mass = True
        self.use_sparse = self.settings['use_sparse']

        ScalingFacts = self.settings['ScalingDict']
        ScalingFacts['time'] = ScalingFacts['length'] / ScalingFacts['speed']
        ScalingFacts['circulation'] = ScalingFacts['speed'] * ScalingFacts['length']
        ScalingFacts['dyn_pressure'] = 0.5 * ScalingFacts['density'] * ScalingFacts['speed'] ** 2
        ScalingFacts['force'] = ScalingFacts['dyn_pressure'] * ScalingFacts['length'] ** 2
        self.ScalingFacts = ScalingFacts

        self.input_variables_list = [InputVariable('zeta', size=3 * self.Kzeta, index=0),
                                     InputVariable('zeta_dot', size=3 * self.Kzeta, index=1),
                                     InputVariable('u_gust', size=3 * self.Kzeta, index=2)]

        self.state_variables_list = [StateVariable('gamma', size=self.K, index=0),
                                     StateVariable('gamma_w', size=self.K_star, index=1),
                                     StateVariable('dtgamma_dot', size=self.K, index=2),
                                     StateVariable('gamma_m1', size=self.K, index=3)]

        self.output_variables_list = [OutputVariable('forces_v', size=3 * self.Kzeta, index=0)]

        if self.integr_order == 1:
            self.state_variables_list.pop(2) # remove time derivative state

        ### collect statistics
        self.cpu_summary = {'dim': 0.,
                            'nondim': 0.,
                            'assemble': 0.}

        # Initialise State Space
        self.SS = None

    @property
    def Nu(self):
        """Number of inputs :math:`m` to the system."""
        if self.SS is not None:
            if self.SS.B.shape.__len__() == 1:
                self.Nu = 1
            else:
                self.Nu = self.SS.B.shape[1]
        return self._Nu

    @Nu.setter
    def Nu(self, value):
        self._Nu = value

    @property
    def Nx(self):
        """Number of states :math:`n` of the system."""
        if self.SS is not None:
            self.Nx = self.SS.B.shape[0]
        return self._Nx

    @Nx.setter
    def Nx(self, value):
        self._Nx = value

    @property
    def Ny(self):
        """Number of outputs :math:`p` of the system."""
        if self.SS is not None:
            self.Ny = self.SS.C.shape[0]
        return self._Ny

    @Ny.setter
    def Ny(self, value):
        self._Ny = value

    def nondimss(self):
        """
        Scale state-space model based of self.ScalingFacts
        """

        cout.cout_wrap('Scaling UVLM system with reference time %fs' % self.ScalingFacts['time'])
        t0 = time.time()
        Kzeta = self.Kzeta

        self.SS.B[:, :3 * Kzeta] *= (self.ScalingFacts['length'] / self.ScalingFacts['circulation'])
        self.SS.B[:, 3 * Kzeta:] *= (self.ScalingFacts['speed'] / self.ScalingFacts['circulation'])
        if self.remove_predictor:
            self.B_predictor[:, :3 * Kzeta] *= (self.ScalingFacts['length'] / self.ScalingFacts['circulation'])
            self.B_predictor[:, 3 * Kzeta:] *= (self.ScalingFacts['speed'] / self.ScalingFacts['circulation'])

        self.SS.C *= (self.ScalingFacts['circulation'] / self.ScalingFacts['force'])

        self.SS.D[:, :3 * Kzeta] *= (self.ScalingFacts['length'] / self.ScalingFacts['force'])
        self.SS.D[:, 3 * Kzeta:] *= (self.ScalingFacts['speed'] / self.ScalingFacts['force'])
        if self.remove_predictor:
            self.D_predictor[:, :3 * Kzeta] *= (self.ScalingFacts['length'] / self.ScalingFacts['force'])
            self.D_predictor[:, 3 * Kzeta:] *= (self.ScalingFacts['speed'] / self.ScalingFacts['force'])

        self.SS.dt = self.SS.dt / self.ScalingFacts['time']

        self.cpu_summary['nondim'] = time.time() - t0
        cout.cout_wrap('Non-dimensional time step set (%f)' % self.SS.dt, 1)
        cout.cout_wrap('System scaled in %fs' % self.cpu_summary['nondim'])

    def dimss(self):

        t0 = time.time()
        Kzeta = self.Kzeta

        self.SS.B[:, :3 * Kzeta] /= (self.ScalingFacts['length'] / self.ScalingFacts['circulation'])
        self.SS.B[:, 3 * Kzeta:] /= (self.ScalingFacts['speed'] / self.ScalingFacts['circulation'])
        if self.remove_predictor:
            self.B_predictor[:, :3 * Kzeta] /= (self.ScalingFacts['length'] / self.ScalingFacts['circulation'])
            self.B_predictor[:, 3 * Kzeta:] /= (self.ScalingFacts['speed'] / self.ScalingFacts['circulation'])

        self.SS.C /= (self.ScalingFacts['circulation'] / self.ScalingFacts['force'])

        self.SS.D[:, :3 * Kzeta] /= (self.ScalingFacts['length'] / self.ScalingFacts['force'])
        self.SS.D[:, 3 * Kzeta:] /= (self.ScalingFacts['speed'] / self.ScalingFacts['force'])
        if self.remove_predictor:
            self.D_predictor[:, :3 * Kzeta] /= (self.ScalingFacts['length'] / self.ScalingFacts['force'])
            self.D_predictor[:, 3 * Kzeta:] /= (self.ScalingFacts['speed'] / self.ScalingFacts['force'])

        self.SS.dt = self.SS.dt * self.ScalingFacts['time']

        self.cpu_summary['dim'] = time.time() - t0

    def assemble_ss(self, wake_prop_settings=None):
        r"""
        Produces state-space model of the form

            .. math::

                \mathbf{x}_{n+1} &= \mathbf{A}\,\mathbf{x}_n + \mathbf{B} \mathbf{u}_{n+1} \\
                \mathbf{y}_n &= \mathbf{C}\,\mathbf{x}_n + \mathbf{D} \mathbf{u}_n

        where the state, inputs and outputs are:

            .. math:: \mathbf{x}_n = \{ \delta \mathbf{\Gamma}_n,\, \delta \mathbf{\Gamma_{w_n}},\,
                \Delta t\,\delta\mathbf{\Gamma}'_n,\, \delta\mathbf{\Gamma}_{n-1} \}

            .. math:: \mathbf{u}_n = \{ \delta\mathbf{\zeta}_n,\, \delta\mathbf{\zeta}'_n,\,
                \delta\mathbf{u}_{ext,n} \}

            .. math:: \mathbf{y} = \{\delta\mathbf{f}\}

        with :math:`\mathbf{\Gamma}\in\mathbb{R}^{MN}` being the vector of vortex circulations,
        :math:`\mathbf{\zeta}\in\mathbb{R}^{3(M+1)(N+1)}` the vector of vortex lattice coordinates and
        :math:`\mathbf{f}\in\mathbb{R}^{3(M+1)(N+1)}` the vector of aerodynamic forces and moments. Note
        that :math:`(\bullet)'` denotes a derivative with respect to time.

        Note that the input is atypically defined at time ``n+1``, therefore by default
        ``self.remove_predictor = True`` and the predictor term ``u_{n+1}`` is eliminated through
        the change of state[1]:

            .. math::
                \mathbf{h}_n &= \mathbf{x}_n - \mathbf{B}\,\mathbf{u}_n \\

        such that:

            .. math::
                \mathbf{h}_{n+1} &= \mathbf{A}\,\mathbf{h}_n + \mathbf{A\,B}\,\mathbf{u}_n \\
                \mathbf{y}_n &= \mathbf{C\,h}_n + (\mathbf{C\,B}+\mathbf{D})\,\mathbf{u}_n


        which only modifies the equivalent :math:`\mathbf{B}` and :math:`\mathbf{D}` matrices.

        References:
            [1] Franklin, GF and Powell, JD. Digital Control of Dynamic Systems, Addison-Wesley Publishing Company, 1980

        To do:
        - remove all calls to scipy.linalg.block_diag
        """

        cout.cout_wrap('State-space realisation of UVLM equations started...')
        t0 = time.time()
        MS = self.MS
        K, K_star = self.K, self.K_star
        Kzeta = self.Kzeta

        # ------------------------------------------------------ determine size

        Nx = self.Nx
        Nu = self.Nu
        Ny = self.Ny
        if self.integr_order == 2:
            # Second order differencing scheme coefficients
            b0, bm1, bp1 = -2., 0.5, 1.5

        # ----------------------------------------------------------- state eq.

        ### state terms (A matrix)
        # - choice of sparse matrices format is optimised to reduce memory load

        # Aero influence coeffs
        List_AICs, List_AICs_star = ass.AICs(MS.Surfs, MS.Surfs_star,
                                             target='collocation', Project=True)
        A0 = np.block(List_AICs)
        A0W = np.block(List_AICs_star)
        List_AICs, List_AICs_star = None, None
        LU, P = scalg.lu_factor(A0)
        AinvAW = scalg.lu_solve((LU, P), A0W)
        A0, A0W = None, None
        # self.A0,self.A0W=A0,A0W

        ### propagation of circ
        # fast and memory efficient with both dense and sparse matrices
        List_C, List_Cstar = ass.wake_prop(MS,
                                           self.use_sparse, sparse_format='csc',
                                           settings=wake_prop_settings)
        if self.use_sparse:
            Cgamma = libsp.csc_matrix(sparse.block_diag(List_C, format='csc'))
            CgammaW = libsp.csc_matrix(sparse.block_diag(List_Cstar, format='csc'))
        else:
            Cgamma = scalg.block_diag(*List_C)
            CgammaW = scalg.block_diag(*List_Cstar)
        List_C, List_Cstar = None, None

        # recurrent dense terms stored as numpy.ndarrays
        AinvAWCgamma = -libsp.dot(AinvAW, Cgamma)
        AinvAWCgammaW = -libsp.dot(AinvAW, CgammaW)

        ### A matrix assembly
        if self.use_sparse:
            # lil format allows fast assembly
            Ass = sparse.lil_matrix((Nx, Nx))
        else:
            Ass = np.zeros((Nx, Nx))
        Ass[:K, :K] = AinvAWCgamma
        Ass[:K, K:K + K_star] = AinvAWCgammaW
        Ass[K:K + K_star, :K] = Cgamma
        Ass[K:K + K_star, K:K + K_star] = CgammaW
        Cgamma, CgammaW = None, None

        # delta eq.
        iivec = range(K + K_star, 2 * K + K_star)
        ones = np.ones((K,))
        if self.integr_order == 1:
            Ass[iivec, :K] = AinvAWCgamma
            Ass[iivec, range(K)] -= ones
            Ass[iivec, K:K + K_star] = AinvAWCgammaW
        if self.integr_order == 2:
            Ass[iivec, :K] = bp1 * AinvAWCgamma
            AinvAWCgamma = None
            Ass[iivec, range(K)] += b0 * ones
            Ass[iivec, K:K + K_star] = bp1 * AinvAWCgammaW
            AinvAWCgammaW = None
            Ass[iivec, range(2 * K + K_star, 3 * K + K_star)] = bm1 * ones
            # identity eq.
            Ass[range(2 * K + K_star, 3 * K + K_star), range(K)] = ones

        if self.use_sparse:
            # conversion to csc occupies less memory and allows fast algebra
            Ass = libsp.csc_matrix(Ass)

        # zeta derivs
        List_nc_dqcdzeta = ass.nc_dqcdzeta(MS.Surfs, MS.Surfs_star, Merge=True)
        List_uc_dncdzeta = ass.uc_dncdzeta(MS.Surfs)
        List_nc_domegazetadzeta_vert = ass.nc_domegazetadzeta(MS.Surfs, MS.Surfs_star)
        for ss in range(MS.n_surf):
            List_nc_dqcdzeta[ss][ss] += \
                (List_uc_dncdzeta[ss] + List_nc_domegazetadzeta_vert[ss])
        Ducdzeta = np.block(List_nc_dqcdzeta)  # dense matrix
        List_nc_dqcdzeta = None
        List_uc_dncdzeta = None
        List_nc_domegazetadzeta_vert = None

        # ext velocity derivs (Wnv0)
        List_Wnv = []
        for ss in range(MS.n_surf):
            List_Wnv.append(
                interp.get_Wnv_vector(MS.Surfs[ss],
                                      MS.Surfs[ss].aM, MS.Surfs[ss].aN))
        AinvWnv0 = scalg.lu_solve((LU, P), scalg.block_diag(*List_Wnv))
        List_Wnv = None

        ### B matrix assembly
        if self.use_sparse:
            Bss = sparse.lil_matrix((Nx, Nu))
        else:
            Bss = np.zeros((Nx, Nu))

        Bup = np.block([-scalg.lu_solve((LU, P), Ducdzeta), AinvWnv0, -AinvWnv0])
        AinvWnv0 = None
        Bss[:K, :] = Bup
        if self.integr_order == 1:
            Bss[K + K_star:2 * K + K_star, :] = Bup
        if self.integr_order == 2:
            Bss[K + K_star:2 * K + K_star, :] = bp1 * Bup
        Bup = None

        if self.use_sparse:
            Bss = libsp.csc_matrix(Bss)
        LU, P = None, None
        # ---------------------------------------------------------- output eq.

        ### state terms (C matrix)

        # gamma (induced velocity contrib.)
        List_dfqsdvind_gamma, List_dfqsdvind_gamma_star = \
            ass.dfqsdvind_gamma(MS.Surfs, MS.Surfs_star)

        # gamma (at constant relative velocity)
        List_dfqsdgamma_vrel0, List_dfqsdgamma_star_vrel0 = \
            ass.dfqsdgamma_vrel0(MS.Surfs, MS.Surfs_star)
        for ss in range(MS.n_surf):
            List_dfqsdvind_gamma[ss][ss] += List_dfqsdgamma_vrel0[ss]
            List_dfqsdvind_gamma_star[ss][ss] += List_dfqsdgamma_star_vrel0[ss]
        Dfqsdgamma = np.block(List_dfqsdvind_gamma)
        Dfqsdgamma_star = np.block(List_dfqsdvind_gamma_star)
        List_dfqsdvind_gamma, List_dfqsdvind_gamma_star = None, None
        List_dfqsdgamma_vrel0, List_dfqsdgamma_star_vrel0 = None, None

        # gamma_dot
        Dfunstdgamma_dot = scalg.block_diag(*ass.dfunstdgamma_dot(MS.Surfs))

        # C matrix assembly
        Css = np.zeros((Ny, Nx))
        Css[:, :K] = Dfqsdgamma
        Css[:, K:K + K_star] = Dfqsdgamma_star
        if self.include_added_mass:
            Css[:, K + K_star:2 * K + K_star] = Dfunstdgamma_dot / self.dt

        ### input terms (D matrix)
        Dss = np.zeros((Ny, Nu))

        # zeta (at constant relative velocity)
        Dss[:, :3 * Kzeta] = scalg.block_diag(
            *ass.dfqsdzeta_vrel0(MS.Surfs, MS.Surfs_star))
        # zeta (induced velocity contrib)
        List_coll, List_vert = ass.dfqsdvind_zeta(MS.Surfs, MS.Surfs_star)
        for ss in range(MS.n_surf):
            List_vert[ss][ss] += List_coll[ss]
        Dss[:, :3 * Kzeta] += np.block(List_vert)
        del List_vert, List_coll

        # input velocities (external)
        Dss[:, 6 * Kzeta:9 * Kzeta] = scalg.block_diag(
            *ass.dfqsduinput(MS.Surfs, MS.Surfs_star))

        # input velocities (body movement)
        if self.include_added_mass:
            Dss[:, 3 * Kzeta:6 * Kzeta] = -Dss[:, 6 * Kzeta:9 * Kzeta]

        if self.remove_predictor:
            Ass, Bmod, Css, Dmod = \
                libss.SSconv(Ass, None, Bss, Css, Dss, Bm1=None)
            self.SS = libss.StateSpace(Ass, Bmod, Css, Dmod, dt=self.dt)

            # Store original B matrix for state unpacking
            self.B_predictor = Bss
            self.D_predictor = Dss

            cout.cout_wrap('\tstate-space model produced in form:\n\t' \
                           '\t\th_{n+1} = A h_{n} + B u_{n}\n\t' \
                           '\t\twith:\n\tx_n = h_n + Bp u_n', 1)
        else:
            self.SS = libss.StateSpace(Ass, Bss, Css, Dss, dt=self.dt)
            cout.cout_wrap('\tstate-space model produced in form:\n\t' \
                           'x_{n+1} = A x_{n} + Bp u_{n+1}', 1)

        # add variable tracker
        self.SS.input_variables = LinearVector(self.input_variables_list)
        self.SS.state_variables = LinearVector(self.state_variables_list)
        self.SS.output_variables = LinearVector(self.output_variables_list)

        self.cpu_summary['assemble'] = time.time() - t0
        cout.cout_wrap('\t\t\t...done in %.2f sec' % self.cpu_summary['assemble'])

    def freqresp(self, kv, wake_prop_settings=None):
        """
        Ad-hoc method for fast UVLM frequency response over the frequencies
        kv. The method, only requires inversion of a K x K matrix at each
        frequency as the equation for propagation of wake circulation are solved
        exactly.
        The algorithm implemented here can be used also upon projection of
        the state-space model.

        Note:
        This method is very similar to the "minsize" solution option is the
        steady_solve.
        """

        if self.remove_predictor:
            # raise NameError('Option "remove_predictor=True" not implemented yet. '+
            #     'Refer to Frequency class implementation.')

            assert self.B_predictor.shape == self.SS.B.shape, \
                ('In order to use "freqresp" with "remove_predictor=True", project ' +
                 '"self.B_predictor" as per "self.SS.B"!')
            assert self.D_predictor.shape == self.SS.D.shape, \
                ('In order to use "freqresp" with "remove_predictor=True", project ' +
                 '"self.D_predictor" as per "self.SS.D"!')

        MS = self.MS
        K = self.K
        K_star = self.K_star

        Eye = np.eye(K)
        if self.remove_predictor:
            Bup = self.B_predictor[:K, :]
        else:
            Bup = self.SS.B[:K, :]

        if self.use_sparse:
            # warning: behaviour may change in future numpy release.
            # Ensure P,Pw,Bup are np.ndarray
            P = np.array(self.SS.A[:K, :K].todense())
            Pw = np.array(self.SS.A[:K, K:K + K_star].todense())
            if type(Bup) not in [np.ndarray, libsp.csc_matrix]:
                Bup = Bup.toarray()
        else:
            P = self.SS.A[:K, :K]
            Pw = self.SS.A[:K, K:K + K_star]

        Nk = len(kv)
        kvdt = kv * self.SS.dt
        zv = np.cos(kvdt) + 1.j * np.sin(kvdt)
        Yfreq = np.empty((self.SS.outputs, self.SS.inputs, Nk,), dtype=np.complex_)

        for kk in range(Nk):

            ###  build Cw complex
            Cw_cpx = self.get_Cw_cpx(zv[kk], settings=wake_prop_settings)

            if self.remove_predictor:
                Ygamma = zv[kk] * \
                         libsp.solve(zv[kk] * Eye - P -
                                     libsp.dot(Pw, Cw_cpx, type_out=libsp.csc_matrix),
                                     Bup)
            else:
                Ygamma = libsp.solve(zv[kk] * Eye - P -
                                     libsp.dot(Pw, Cw_cpx, type_out=libsp.csc_matrix),
                                     Bup)

            Ygamma_star = Cw_cpx.dot(Ygamma)

            if self.integr_order == 1:
                dfact = (1. - 1. / zv[kk])
            elif self.integr_order == 2:
                dfact = .5 * (3. - 4. / zv[kk] + 1. / zv[kk] ** 2)
            else:
                raise NameError('Specify valid integration order')

            # calculate solution
            if self.remove_predictor:
                Yfreq[:, :, kk] = np.dot(self.SS.C[:, :K], Ygamma) + \
                                  np.dot(self.SS.C[:, K:K + K_star], Ygamma_star) + \
                                  np.dot(self.SS.C[:, K + K_star:2 * K + K_star], dfact * Ygamma) + \
                                  self.D_predictor
            else:
                Yfreq[:, :, kk] = np.dot(self.SS.C[:, :K], Ygamma) + \
                                  np.dot(self.SS.C[:, K:K + K_star], Ygamma_star) + \
                                  np.dot(self.SS.C[:, K + K_star:2 * K + K_star], dfact * Ygamma) + \
                                  self.SS.D

        return Yfreq

    def get_Cw_cpx(self, zval, settings=None):
        r"""
        Produces a sparse matrix

            .. math:: \bar{\mathbf{C}}(z)

        where

            .. math:: z = e^{k \Delta t}

        such that the wake circulation frequency response at :math:`z` is

            .. math:: \bar{\boldsymbol{\Gamma}}_w = \bar{\mathbf{C}}(z)  \bar{\mathbf{\Gamma}}

        """

        return get_Cw_cpx(self.MS, self.K, self.K_star, zval, settings=settings)


    def balfreq(self, DictBalFreq, wake_prop_settings=None):
        """
        Low-rank method for frequency limited balancing.
        The Observability ad controllability Gramians over the frequencies kv
        are solved in factorised form. Balancd modes are then obtained with a
        square-root method.

        Details:

        Observability and controllability Gramians are solved in factorised form
        through explicit integration. The number of integration points determines
        both the accuracy and the maximum size of the balanced model.

        Stability over all (Nb) balanced states is achieved if:

            a. one of the Gramian is integrated through the full Nyquist range
            b. the integration points are enough.

        Note, however, that even when stability is not achieved over the full
        balanced states, stability of the balanced truncated model with Ns<=Nb
        states is normally observed even when a low number of integration points
        is used. Two integration methods (trapezoidal rule on uniform grid and
        Gauss-Legendre quadrature) are provided.

        Input:

        - DictBalFreq: dictionary specifying integration method with keys:

            - ``frequency``: defines limit frequencies for balancing. The balanced
              model will be accurate in the range [0,F], where F is the value of
              this key. Note that F units must be consistent with the units specified
              in the self.ScalingFacts dictionary.

            - ``method_low``: ['gauss','trapz'] specifies whether to use gauss
              quadrature or trapezoidal rule in the low-frequency range [0,F]

            - ``options_low``: options to use for integration in the low-frequencies.
              These depend on the integration scheme (See below).

            - ``method_high``: method to use for integration in the range [F,F_N],
              where F_N is the Nyquist frequency. See 'method_low'.

            - ``options_high``: options to use for integration in the high-frequencies.

            - ``check_stability``: if True, the balanced model is truncated to
              eliminate unstable modes - if any is found. Note that very accurate
              balanced model can still be obtained, even if high order modes are
              unstable. Note that this option is overridden if ""

            - ``get_frequency_response``: if True, the function also returns the
              frequency response evaluated at the low-frequency range integration
              points. If True, this option also allows to automatically tune the
              balanced model.

        Future options:

            - ``truncation_tolerance``: if ``get_frequency_response`` is True, allows
              to truncate the balanced model so as to achieved a prescribed
              tolerance in the low-frequwncy range.

            - ``Ncpu``: for parallel run

        The following integration schemes are available:

            - ``trapz``: performs integration over equally spaced points using
              trapezoidal rule. It accepts options dictionaries with keys:

                - ``points``: number of integration points to use (including
                  domain boundary)

            - ``gauss`` performs gauss-lobotto quadrature. The domain can be
              partitioned in Npart sub-domain in which the gauss-lobotto quadrature
              of order Ord can be applied. A total number of Npart*Ord points is
              required. It accepts options dictionaries of the form:

                - ``partitions``: number of partitions
                - ``order``: quadrature order.

        Example:

        The following dictionary

            .. code-block:: python

                DictBalFreq={   'frequency': 1.2,
                                'method_low': 'trapz',
                                'options_low': {'points': 12},
                                'method_high': 'gauss',
                                'options_high': {'partitions': 2, 'order': 8},
                                'check_stability': True }

        balances the state-space model self.SS in the frequency range [0, 1.2]
        using

            (a) 12 equally-spaced points integration of the Gramians in the low-frequency range [0,1.2] and
            (b) a 2 Gauss-Lobotto 8-th order quadratures of the controllability
                Gramian in the high-frequency range.

        A total number of 28 integration points will be required, which will
        result into a balanced model with number of states ``min{2*28* number_inputs, 2*28* number_outputs}``

        The model is finally truncated so as to retain only the first Ns stable
        modes.
        """

        ### check input dictionary
        if 'frequency' not in DictBalFreq:
            raise NameError('Solution dictionary must include the "frequency" key')

        if 'method_low' not in DictBalFreq:
            warnings.warn('Setting default options for low-frequency integration')
            DictBalFreq['method_low'] = 'trapz'
            DictBalFreq['options_low'] = {'points': 12}

        if 'method_high' not in DictBalFreq:
            warnings.warn('Setting default options for high-frequency integration')
            DictBalFreq['method_high'] = 'gauss'
            DictBalFreq['options_high'] = {'partitions': 2, 'order': 8}

        if 'check_stability' not in DictBalFreq:
            DictBalFreq['check_stability'] = True

        if 'output_modes' not in DictBalFreq:
            DictBalFreq['output_modes'] = True

        if 'get_frequency_response' not in DictBalFreq:
            DictBalFreq['get_frequency_response'] = False

        ### get integration points and weights

        # Nyquist frequency
        kn = np.pi / self.SS.dt

        Opt = DictBalFreq['options_low']
        if DictBalFreq['method_low'] == 'trapz':
            kv_low, wv_low = librom.get_trapz_weights(0., DictBalFreq['frequency'],
                                                      Opt['points'], False)
        elif DictBalFreq['method_low'] == 'gauss':
            kv_low, wv_low = librom.get_gauss_weights(0., DictBalFreq['frequency'],
                                                      Opt['partitions'], Opt['order'])
        else:
            raise NameError(
                'Invalid value %s for key "method_low"' % DictBalFreq['method_low'])

        Opt = DictBalFreq['options_high']
        if DictBalFreq['method_high'] == 'trapz':
            kv_high, wv_high = librom.get_trapz_weights(DictBalFreq['frequency'], kn,
                                                        Opt['points'], True)
        elif DictBalFreq['method_high'] == 'gauss':
            kv_high, wv_high = librom.get_gauss_weights(DictBalFreq['frequency'], kn,
                                                        Opt['partitions'], Opt['order'])
        else:
            raise NameError(
                'Invalid value %s for key "method_high"' % DictBalFreq['method_high'])

        ### get useful terms
        K = self.K
        K_star = self.K_star

        Eye = np.eye(K)

        if self.remove_predictor:
            Bup = self.B_predictor[:K, :]
        else:
            Bup = self.SS.B[:K, :]

        if self.use_sparse:
            # warning: behaviour may change in future numpy release.
            # Ensure P,Pw,Bup are np.ndarray
            P = np.array(self.SS.A[:K, :K].todense())
            Pw = np.array(self.SS.A[:K, K:K + K_star].todense())
            if type(Bup) not in [np.ndarray, libsp.csc_matrix]:
                Bup = Bup.toarray()
        else:
            P = self.SS.A[:K, :K]
            Pw = self.SS.A[:K, K:K + K_star]

        # indices to manipulate obs solution
        ii00 = range(0, self.K)
        ii01 = range(self.K, self.K + self.K_star)
        ii02 = range(self.K + self.K_star, 2 * self.K + self.K_star)
        ii03 = range(2 * self.K + self.K_star, 3 * self.K + self.K_star)

        # integration factors
        if self.integr_order == 2:
            b0, bm1, bp1 = -2., 0.5, 1.5
        else:
            b0, bp1 = -1., 1.
            raise NameError('Method not implemented for integration order 1')

        ### -------------------------------------------------- loop frequencies

        ### merge vectors
        Nk_low = len(kv_low)
        kvdt = np.concatenate((kv_low, kv_high)) * self.SS.dt
        wv = np.concatenate((wv_low, wv_high)) * self.SS.dt
        zv = np.cos(kvdt) + 1.j * np.sin(kvdt)

        Qobs = np.zeros((self.SS.states, self.SS.outputs), dtype=np.complex_)
        Zc = np.zeros((self.SS.states, 2 * self.SS.inputs * len(kvdt)), )
        Zo = np.zeros((self.SS.states, 2 * self.SS.outputs * Nk_low), )

        if DictBalFreq['get_frequency_response']:
            self.Yfreq = np.empty((self.SS.outputs, self.SS.inputs, Nk_low,), dtype=np.complex_)
            self.kv = kv_low

        for kk in range(len(kvdt)):

            zval = zv[kk]
            Intfact = wv[kk]  # integration factor

            #  build terms that will be recycled
            Cw_cpx = self.get_Cw_cpx(zval, settings=wake_prop_settings)
            PwCw_T = Cw_cpx.T.dot(Pw.T)
            Kernel = np.linalg.inv(zval * Eye - P - PwCw_T.T)

            ### ----- controllability
            Ygamma = Intfact * np.dot(Kernel, Bup)
            if self.remove_predictor:
                Ygamma *= zval
            Ygamma_star = Cw_cpx.dot(Ygamma)

            if self.integr_order == 1:
                dfact = (bp1 + b0 / zval)
                Qctrl = np.vstack([Ygamma, Ygamma_star, dfact * Ygamma])
            elif self.integr_order == 2:
                dfact = bp1 + b0 / zval + bm1 / zval ** 2
                Qctrl = np.vstack(
                    [Ygamma, Ygamma_star, dfact * Ygamma, (1. / zval) * Ygamma])
            else:
                raise NameError('Specify valid integration order')

            kkvec = range(2 * kk * self.SS.inputs, 2 * (kk + 1) * self.SS.inputs)
            Zc[:, kkvec[:self.SS.inputs]] = Qctrl.real  # *Intfact
            Zc[:, kkvec[self.SS.inputs:]] = Qctrl.imag  # *Intfact

            ### ----- frequency response
            if DictBalFreq['get_frequency_response'] and kk < Nk_low:
                self.Yfreq[:, :, kk] = np.dot(self.SS.C, Qctrl) / Intfact + self.SS.D

            ### ----- frequency response
            if DictBalFreq['get_frequency_response'] and kk < Nk_low:
                self.Yfreq[:, :, kk] = np.dot(self.SS.C, Qctrl) / Intfact + self.SS.D

            ### ----- observability
            # solve (1./zval*I - A.T)^{-1} C^T (in low-frequency only)
            if kk >= Nk_low:
                continue

            zinv = 1. / zval
            Cw_cpx_H = Cw_cpx.conjugate().T

            Qobs[ii02, :] = zval * self.SS.C[:, ii02].T
            if self.integr_order == 2:
                Qobs[ii03, :] = bm1 * zval ** 2 * self.SS.C[:, ii02].T

            rhs = self.SS.C[:, ii00].T + \
                  Cw_cpx_H.dot(self.SS.C[:, ii01].T) + \
                  libsp.dot(
                      (bp1 * zval) * (PwCw_T.conj() + P.T) + \
                      (b0 * zval + bm1 * zval ** 2) * Eye, self.SS.C[:, ii02].T)

            Qobs[ii00, :] = np.dot(Kernel.conj().T, rhs)

            Eye_star = libsp.csc_matrix(
                (zinv * np.ones((K_star,)), (range(K_star), range(K_star))),
                shape=(K_star, K_star), dtype=np.complex_)
            Qobs[ii01, :] = libsp.solve(
                Eye_star - self.SS.A[K:K + K_star, K:K + K_star].T,
                np.dot(Pw.T, Qobs[ii00, :] + \
                       (bp1 * zval) * self.SS.C[:, ii02].T) + \
                self.SS.C[:, ii01].T)

            kkvec = range(2 * kk * self.SS.outputs, 2 * (kk + 1) * self.SS.outputs)
            Zo[:, kkvec[:self.SS.outputs]] = Intfact * Qobs.real
            Zo[:, kkvec[self.SS.outputs:]] = Intfact * Qobs.imag

        # delete full matrices
        Kernel = None
        Qctrl = None
        Qobs = None

        # self.Zc=Zc
        # self.Zo=Zo

        # LRSQM (optimised)
        U, hsv, Vh = scalg.svd(np.dot(Zo.T, Zc), full_matrices=False)
        sinv = hsv ** (-0.5)
        T = np.dot(Zc, Vh.T * sinv)
        Ti = np.dot((U * sinv).T, Zo.T)
        # Zc,Zo=None,None

        ### build frequency balanced model
        Ab = libsp.dot(Ti, libsp.dot(self.SS.A, T))
        Bb = libsp.dot(Ti, self.SS.B)
        Cb = libsp.dot(self.SS.C, T)
        SSb = libss.StateSpace(Ab, Bb, Cb, self.SS.D, dt=self.SS.dt)

        ### Eliminate unstable modes - if any:
        if DictBalFreq['check_stability']:
            for nn in range(1, len(hsv) + 1):
                eigs_trunc = scalg.eigvals(SSb.A[:nn, :nn])
                eigs_trunc_max = np.max(np.abs(eigs_trunc))
                if eigs_trunc_max > 1. - 1e-16:
                    SSb.truncate(nn - 1)
                    hsv = hsv[:nn - 1]
                    T = T[:, :nn - 1]
                    Ti = Ti[:nn - 1, :]
                    break

        self.SSb = SSb
        self.hsv = hsv
        if DictBalFreq['output_modes']:
            self.T = T
            self.Ti = Ti
            self.Zc = Zc
            self.Zo = Zo

    def balfreq_profiling(self, wake_prop_settings=None):
        """
        Generate profiling report for balfreq function and saves it into ``self.prof_out.``
        The function also returns a ``pstats.Stats`` object.

        To read the report:
            .. code-block:: python

                import pstats
                p=pstats.Stats(self.prof_out).sort_stats('cumtime')
                p.print_stats(20)
        """
        import pstats
        import cProfile
        def wrap():
            DictBalFreq = {'frequency': 0.5, 'check_stability': False}
            self.balfreq(DictBalFreq, wake_prop_settings=wake_prop_settings)

        cProfile.runctx('wrap()', globals(), locals(), filename=self.prof_out)

        cProfile.runctx('wrap()', globals(), locals(), filename=self.prof_out)

        return pstats.Stats(self.prof_out).sort_stats('cumtime')

    def assemble_ss_profiling(self):
        """
        Generate profiling report for assembly and save it in self.prof_out.

        To read the report:
            import pstats
            p=pstats.Stats(self.prof_out)
        """

        import cProfile
        cProfile.runctx('self.assemble_ss()', globals(), locals(), filename=self.prof_out)

    def solve_steady(self, usta, method='direct'):
        """
        Steady state solution from state-space model.

        Warning: these methods are less efficient than the solver in Static
        class, Static.solve, and should be used only for verification purposes.
        The "minsize" method, however, guarantees the inversion of a K x K
        matrix only, similarly to what is done in Static.solve.
        """

        if self.remove_predictor is True and method != 'direct':
            raise NameError('Only direct solution is available if predictor ' \
                            'term has been removed from the state-space equations (see assembly_ss)')

        if self.use_sparse is True and method != 'direct':
            raise NameError('Only direct solution is available if use_sparse is True. ' \
                            '(see assembly_ss)')

        Ass, Bss, Css, Dss = self.SS.A, self.SS.B, self.SS.C, self.SS.D

        if method == 'minsize':
            # as opposed to linuvlm.Static, this solves for the bound circulation
            # starting from
            #	Gamma   = [P, Pw] * {Gamma,Gamma_w} + B u_
            #   Gamma_w = [C, Cw] * {Gamma,Gamma_w}		  			(eq. 1a,1b)
            # where time indices have been dropped. Transforming into
            #	Gamma 	= [P+PwC, PwCw] * {Gamma,Gamma_w} + B u 	(eq. 2)
            # and enforcing Gamma_w = Gamma_TE, this reduces to the KxK system:
            # 	AIC Gamma = B u 									(eq. 3)
            MS = self.MS
            K = self.K
            K_star = self.K_star

            ### build eq. 2:
            P = Ass[:K, :K]
            Pw = Ass[:K, K:K + K_star]
            C = Ass[K:K + K_star, :K]
            Cw = Ass[K:K + K_star, K:K + K_star]
            PwCw = np.dot(Pw, Cw)

            ### build eq. 3
            AIC = np.eye(K) - P - np.dot(Pw, C)

            # condense Gammaw terms
            K0out, K0out_star = 0, 0
            for ss_out in range(MS.n_surf):
                Kout = MS.KK[ss_out]
                Kout_star = MS.KK_star[ss_out]

                K0in, K0in_star = 0, 0
                for ss_in in range(MS.n_surf):
                    Kin = MS.KK[ss_in]
                    Kin_star = MS.KK_star[ss_in]

                    # link to sub-matrices
                    aic = AIC[K0out:K0out + Kout, K0in:K0in + Kin]
                    aic_star = PwCw[K0out:K0out + Kout, K0in_star:K0in_star + Kin_star]

                    # fold aic_star: sum along chord at each span-coordinate
                    N_star = MS.NN_star[ss_in]
                    aic_star_fold = np.zeros((Kout, N_star))
                    for jj in range(N_star):
                        aic_star_fold[:, jj] += np.sum(aic_star[:, jj::N_star], axis=1)
                    aic[:, -N_star:] -= aic_star_fold

                    K0in += Kin
                    K0in_star += Kin_star
                K0out += Kout
                K0out_star += Kout_star

            ### solve
            # gamma
            gamma = np.linalg.solve(AIC, np.dot(Bss[:K, :], usta))
            # retrieve gamma over wake
            gamma_star = []
            Ktot = 0
            for ss in range(MS.n_surf):
                Ktot += MS.Surfs[ss].maps.K
                N = MS.Surfs[ss].maps.N
                Mstar = MS.Surfs_star[ss].maps.M
                gamma_star.append(np.concatenate(Mstar * [gamma[Ktot - N:Ktot]]))
            gamma_star = np.concatenate(gamma_star)

            if self.integr_order == 1:
                xsta = np.concatenate((gamma, gamma_star, np.zeros_like(gamma)))
            if self.integr_order == 2:
                xsta = np.concatenate((gamma, gamma_star, np.zeros_like(gamma),
                                       gamma))

            ysta = np.dot(Css, xsta) + np.dot(Dss, usta)

        elif method == 'direct':
            """ Solves (I - A) x = B u with direct method"""
            # if self.use_sparse:
            #     xsta=libsp.solve(libsp.eye_as(Ass)-Ass,Bss.dot(usta))
            # else:
            #     Ass_steady = np.eye(*Ass.shape) - Ass
            #     xsta = np.linalg.solve(Ass_steady, np.dot(Bss, usta))
            xsta = libsp.solve(libsp.eye_as(Ass) - Ass, Bss.dot(usta))
            ysta = np.dot(Css, xsta) + np.dot(Dss, usta)

        elif method == 'recursive':
            """ Provides steady-state solution solving for impulsive response """
            tol, er = 1e-4, 1.0
            Ftot0 = np.zeros((3,))
            nn = 0
            xsta = np.zeros((self.Nx))
            while er > tol and nn < 1000:
                xsta = np.dot(Ass, xsta) + np.dot(Bss, usta)
                ysta = np.dot(Css, xsta) + np.dot(Dss, usta)
                Ftot = np.array(
                    [np.sum(ysta[cc * self.Kzeta:(cc + 1) * self.Kzeta])
                     for cc in range(3)])
                er = np.linalg.norm(Ftot - Ftot0)
                Ftot0 = Ftot.copy()
                nn += 1
            if er < tol:
                pass  # print('Recursive solution found in %.3d iterations'%nn)
            else:
                raise exceptions.NotConvergedSolver('Solution not found! Max. iterations reached with error: %.3e' % er)

        elif method == 'subsystem':
            """ Solves sub-system related to Gamma, Gamma_w states only """

            Nxsub = self.K + self.K_star
            Asub_steady = np.eye(Nxsub) - Ass[:Nxsub, :Nxsub]
            xsub = np.linalg.solve(Asub_steady, np.dot(Bss[:Nxsub, :], usta))
            if self.integr_order == 1:
                xsta = np.concatenate((xsub, np.zeros((self.K,))))
            if self.integr_order == 2:
                xsta = np.concatenate((xsub, np.zeros((2 * self.K,))))
            ysta = np.dot(Css, xsta) + np.dot(Dss, usta)

        else:
            raise NameError('Method %s not recognised!' % method)

        return xsta, ysta

    def solve_step(self, x_n, u_n, u_n1=None, transform_state=False):
        r"""
        Solve step.

        If the predictor term has not been removed (``remove_predictor = False``) then the system is solved as:

            .. math::
                \mathbf{x}^{n+1} &= \mathbf{A\,x}^n + \mathbf{B\,u}^n \\
                \mathbf{y}^{n+1} &= \mathbf{C\,x}^{n+1} + \mathbf{D\,u}^n

        Else, if ``remove_predictor = True``, the state is modified as

            ..  math:: \mathbf{h}^n = \mathbf{x}^n - \mathbf{B\,u}^n

        And the system solved by:

            .. math::
                \mathbf{h}^{n+1} &= \mathbf{A\,h}^n + \mathbf{B_{mod}\,u}^{n} \\
                \mathbf{y}^{n+1} &= \mathbf{C\,h}^{n+1} + \mathbf{D_{mod}\,u}^{n+1}

        Finally, the original state is recovered using the reverse transformation:

            .. math:: \mathbf{x}^{n+1} = \mathbf{h}^{n+1} + \mathbf{B\,u}^{n+1}

        where the modifications to the :math:`\mathbf{B}_{mod}` and :math:`\mathbf{D}_{mod}` are detailed in
        :func:`Dynamic.assemble_ss`.

        Notes:
            Although the original equations include the term :math:`\mathbf{u}_{n+1}`, it is a reasonable approximation
            to take :math:`\mathbf{u}_{n+1}\approx\mathbf{u}_n` given a sufficiently small time step, hence if the input
            at time ``n+1`` is not parsed, it is estimated from :math:`u^n`.

        Args:
            x_n (np.array): State vector at the current time step :math:`\mathbf{x}^n`
            u_n (np.array): Input vector at time step :math:`\mathbf{u}^n`
            u_n1 (np.array): Input vector at time step :math:`\mathbf{u}^{n+1}`
            transform_state (bool): When the predictor term is removed, if true it will transform the state vector. If
                false it will be assumed that the state vector that is parsed is already transformed i.e. it is
                :math:`\mathbf{h}`.

        Returns:
            Tuple: Updated state and output vector packed in a tuple :math:`(\mathbf{x}^{n+1},\,\mathbf{y}^{n+1})`

        Notes:
            To speed-up the solution and use minimal memory:
                - solve for bound vorticity (and)
                - propagate the wake
                - compute the output separately.
        """

        if u_n1 is None:
            u_n1 = u_n.copy()

        if self.remove_predictor:

            # Transform state vector
            # TODO: Agree on a way to do this. Either always transform here or transform prior to using the method.
            if transform_state:
                h_n = x_n - self.B_predictor.dot(u_n)
            else:
                h_n = x_n

            h_n1 = self.SS.A.dot(h_n) + self.SS.B.dot(u_n)
            y_n1 = np.dot(self.SS.C, h_n1) + np.dot(self.SS.D, u_n1)

            # Recover state vector
            if transform_state:
                x_n1 = h_n1 + self.B_predictor.dot(u_n1)
            else:
                x_n1 = h_n1

        else:
            x_n1 = self.SS.A.dot(x_n) + self.SS.B.dot(u_n1)
            y_n1 = np.dot(self.SS.C, x_n1) + np.dot(self.SS.D, u_n1)

        return x_n1, y_n1

    def unpack_state(self, xvec):
        r"""
        Unpacks the state vector into physical constituents for full order models.

        The state vector :math:`\mathbf{x}` of the form

            .. math:: \mathbf{x}_n = \{ \delta \mathbf{\Gamma}_n,\, \delta \mathbf{\Gamma_{w_n}},\,
                \Delta t\,\delta\mathbf{\Gamma}'_n,\, \delta\mathbf{\Gamma}_{n-1} \}

        Is unpacked into:

            .. math:: {\delta \mathbf{\Gamma}_n,\, \delta \mathbf{\Gamma_{w_n}},\,
                \,\delta\mathbf{\Gamma}'_n}

        Args:
            xvec (np.ndarray): State vector

        Returns:
            tuple: Column vectors for bound circulation, wake circulation and circulation derivative packed in a tuple.
        """

        K, K_star = self.K, self.K_star
        gamma_vec = xvec[:K]
        gamma_star_vec = xvec[K:K + K_star]
        gamma_dot_vec = xvec[K + K_star:2 * K + K_star] / self.dt

        return gamma_vec, gamma_star_vec, gamma_dot_vec


################################################################################

################################################################################

class DynamicBlock(Dynamic):
    """
    Class for dynamic linearised UVLM solution. Linearisation around steady-state
    are only supported.

    The class is a low-memory implementation of Dynamic, and inherits most of
    the methods contained there. State-space models are allocated in list-block
    form (as per numpy.block) to minimise memory usage. This class provides
    lower memory / computational time assembly, frequency response and frequency
    limited balancing.

    Input:

        - tsdata: aero timestep data from SHARPy solution
        - dt: time-step
        - integr_order=2: integration order for UVLM unsteady aerodynamic force
        - RemovePredictor=True: if true, the state-space model is modified so as
          to accept in input perturbations, u, evaluated at time-step n rather than
          n+1.
        - ScalingDict=None: disctionary containing fundamental reference units

            >>> {'length':  reference_length,
                 'speed':   reference_speed,
                 'density': reference density}


          used to derive scaling quantities for the state-space model variables.
          The scaling factors are stores in ``self.ScalingFact``.

          Note that while time, circulation, angular speeds) are scaled
          accordingly, FORCES ARE NOT. These scale by qinf*b**2, where b is the
          reference length and qinf is the dinamic pressure.
        - UseSparse=False: builds the A and B matrices in sparse form. C and D
          are dense, hence the sparce format is not used.


    Methods:
        - nondimss: normalises a dimensional state-space model based on the
          scaling factors in self.ScalingFact.
        - dimss: inverse of nondimss.
        - assemble_ss: builds state-space model. See function for more details.
        - assemble_ss_profiling: generate profiling report of the assembly and
          saves it into self.prof_out. To read the report:

            >>> import pstats
                p=pstats.Stats(self.prof_out)


        - freqresp: ad-hoc method for fast frequency response (only implemented)
          for remove_predictor=False


    To do: upgrade to linearise around unsteady snapshot (adjoint)
    """

    def __init__(self, tsdata, dt=None,
                 dynamic_settings=None,
                 integr_order=2,
                 RemovePredictor=True, ScalingDict=None, UseSparse=True, for_vel=np.zeros((6), )):

        if dynamic_settings is None:
            warnings.warn('Individual parsing of settings is deprecated. Please use the settings dictionary',
                          DeprecationWarning)

        super().__init__(tsdata, dt,
                         dynamic_settings=dynamic_settings,
                         integr_order=integr_order,
                         RemovePredictor=RemovePredictor,
                         ScalingDict=ScalingDict,
                         UseSparse=UseSparse,
                         for_vel=for_vel)

        # number of blocks
        self.nblock_x = self.integr_order + 2
        self.nblock_u = 3
        self.nblock_y = 1

        # sizes in blocks
        self.S_x = [self.K, self.K_star, self.K]
        if self.integr_order == 2: self.S_x += [self.K]
        self.S_u = 3 * [3 * self.Kzeta]
        self.S_y = [3 * self.Kzeta]

    def nondimss(self):
        """
        Scale state-space model based of self.ScalingFacts.
        """

        t0 = time.time()

        B_facts = [self.ScalingFacts['length'] / self.ScalingFacts['circulation'],
                   self.ScalingFacts['speed'] / self.ScalingFacts['circulation'],
                   self.ScalingFacts['speed'] / self.ScalingFacts['circulation']]

        D_facts = [self.ScalingFacts['length'] / self.ScalingFacts['force'],
                   self.ScalingFacts['speed'] / self.ScalingFacts['force'],
                   self.ScalingFacts['speed'] / self.ScalingFacts['force']]

        C_facts = self.nblock_x * \
                  [self.ScalingFacts['circulation'] / self.ScalingFacts['force']]

        for ii in range(self.nblock_x):
            for jj in range(self.nblock_u):
                if self.SS.B[ii][jj] is not None:
                    self.SS.B[ii][jj] *= B_facts[jj]

        for ii in range(self.nblock_y):
            for jj in range(self.nblock_x):
                if self.SS.C[ii][jj] is not None:
                    self.SS.C[ii][jj] *= C_facts[jj]

        for ii in range(self.nblock_y):
            for jj in range(self.nblock_u):
                if self.SS.D[ii][jj] is not None:
                    self.SS.D[ii][jj] *= D_facts[jj]

        self.SS.dt = self.SS.dt / self.ScalingFacts['time']
        self.cpu_summary['nondim'] = time.time() - t0

    def dimss(self):

        t0 = time.time()

        B_facts = [self.ScalingFacts['length'] / self.ScalingFacts['circulation'],
                   self.ScalingFacts['speed'] / self.ScalingFacts['circulation'],
                   self.ScalingFacts['speed'] / self.ScalingFacts['circulation']]

        D_facts = [self.ScalingFacts['length'] / self.ScalingFacts['force'],
                   self.ScalingFacts['speed'] / self.ScalingFacts['force'],
                   self.ScalingFacts['speed'] / self.ScalingFacts['force']]

        D_facts = [self.ScalingFacts['length'] / self.ScalingFacts['force'],
                   self.ScalingFacts['speed'] / self.ScalingFacts['force'],
                   self.ScalingFacts['speed'] / self.ScalingFacts['force']]

        C_facts = self.nblock_x * \
                  [self.ScalingFacts['circulation'] / self.ScalingFacts['force']]

        for ii in range(self.nblock_x):
            for jj in range(self.nblock_u):
                if self.SS.B[ii][jj] is not None:
                    self.SS.B[ii][jj] /= B_facts[jj]

        for ii in range(self.nblock_y):
            for jj in range(self.nblock_x):
                if self.SS.C[ii][jj] is not None:
                    self.SS.C[ii][jj] /= C_facts[jj]

        for ii in range(self.nblock_y):
            for jj in range(self.nblock_u):
                if self.SS.D[ii][jj] is not None:
                    self.SS.D[ii][jj] /= D_facts[jj]

        self.SS.dt = self.SS.dt * self.ScalingFacts['time']
        self.cpu_summary['dim'] = time.time() - t0

    def assemble_ss(self, wake_prop_settings=None):
        r"""
        Produces block-form of state-space model

            .. math::

                \mathbf{x}_{n+1} &= \mathbf{A}\,\mathbf{x}_n + \mathbf{B} \mathbf{u}_{n+1} \\
                \mathbf{y}_n &= \mathbf{C}\,\mathbf{x}_n + \mathbf{D} \mathbf{u}_n

        where the state, inputs and outputs are:

            .. math:: \mathbf{x}_n = \{ \delta \mathbf{\Gamma}_n,\, \delta \mathbf{\Gamma_{w_n}},\,
                \Delta t\,\delta\mathbf{\Gamma}'_n,\, \delta\mathbf{\Gamma}_{n-1} \}

            .. math:: \mathbf{u}_n = \{ \delta\mathbf{\zeta}_n,\, \delta\mathbf{\zeta}'_n,\,
                \delta\mathbf{u}_{ext,n} \}

            .. math:: \mathbf{y} = \{\delta\mathbf{f}\}

        with :math:`\mathbf{\Gamma}` being the vector of vortex circulations,
        :math:`\mathbf{\zeta}` the vector of vortex lattice coordinates and
        :math:`\mathbf{f}` the vector of aerodynamic forces and moments. Note that :math:`(\bullet)'` denotes
        a derivative with respect to time.

        Note that the input is atypically defined at time ``n+1``, therefore by default
        ``self.remove_predictor = True`` and the predictor term ``u_{n+1}`` is eliminated through
        the change of state[1]:

            .. math::
                \mathbf{h}_n &= \mathbf{x}_n - \mathbf{B}\,\mathbf{u}_n \\

        such that:

            .. math::
                \mathbf{h}_{n+1} &= \mathbf{A}\,\mathbf{h}_n + \mathbf{A\,B}\,\mathbf{u}_n \\
                \mathbf{y}_n &= \mathbf{C\,h}_n + (\mathbf{C\,B}+\mathbf{D})\,\mathbf{u}_n


        which only modifies the equivalent :math:`\mathbf{B}` and :math:`\mathbf{D}` matrices.

        References:
            [1] Franklin, GF and Powell, JD. Digital Control of Dynamic Systems, Addison-Wesley Publishing Company, 1980

        To do:
        - remove all calls to scipy.linalg.block_diag
        """

        cout.cout_wrap('\tBlock form state-space realisation of UVLM equations started...', 1)
        t0 = time.time()
        MS = self.MS
        K, K_star = self.K, self.K_star
        Kzeta = self.Kzeta

        # ------------------------------------------------------ determine size

        Nx = self.Nx
        Nu = self.Nu
        Ny = self.Ny

        nblock_x = self.nblock_x
        nblock_u = self.nblock_u
        nblock_y = self.nblock_y

        if self.integr_order == 2:
            # Second order differencing scheme coefficients
            b0, bm1, bp1 = -2., 0.5, 1.5

        # ----------------------------------------------------------- state eq.

        ### state terms (A matrix)
        # - choice of sparse matrices format is optimised to reduce memory load

        # Aero influence coeffs
        List_AICs, List_AICs_star = ass.AICs(MS.Surfs, MS.Surfs_star,
                                             target='collocation', Project=True)
        A0 = np.block(List_AICs)
        A0W = np.block(List_AICs_star)
        List_AICs, List_AICs_star = None, None
        LU, P = scalg.lu_factor(A0)
        AinvAW = scalg.lu_solve((LU, P), A0W)
        A0, A0W = None, None

        ### propagation of circ
        # fast and memory efficient with both dense and sparse matrices
        List_C, List_Cstar = ass.wake_prop(MS,
                                           self.use_sparse, sparse_format='csc',
                                           settings=wake_prop_settings)
        if self.use_sparse:
            Cgamma = libsp.csc_matrix(sparse.block_diag(List_C, format='csc'))
            CgammaW = libsp.csc_matrix(sparse.block_diag(List_Cstar, format='csc'))
        else:
            Cgamma = scalg.block_diag(*List_C)
            CgammaW = scalg.block_diag(*List_Cstar)
        List_C, List_Cstar = None, None

        # recurrent dense terms stored as numpy.ndarrays
        AinvAWCgamma = -libsp.dot(AinvAW, Cgamma)
        AinvAWCgammaW = -libsp.dot(AinvAW, CgammaW)

        ### A matrix assembly
        Ass = []

        # non-penetration condition
        Ass.append([AinvAWCgamma, AinvAWCgammaW, None, ])
        if self.integr_order == 2: Ass[0].append(None)
        # circ. proparagation
        Ass.append([Cgamma, CgammaW, None, ])
        if self.integr_order == 2: Ass[1].append(None)

        Cgamma = None
        CgammaW = None

        # delta eq.
        if self.use_sparse:
            ones = libsp.csc_matrix(
                (np.ones((K,)), (range(K), range(K))), shape=(K, K))
        else:
            ones = np.eye(K)

        if self.integr_order == 1:
            Ass.append([AinvAWCgamma - ones, AinvAWCgammaW.copy(), None])

        elif self.integr_order == 2:
            Ass.append([bp1 * AinvAWCgamma + b0 * ones, bp1 * AinvAWCgammaW, None, bm1 * ones])
            # identity eq.
            Ass.append([ones, None, None, None])
        AinvAWCgamma = None
        AinvAWCgammaW = None

        # zeta derivs
        List_nc_dqcdzeta = ass.nc_dqcdzeta(MS.Surfs, MS.Surfs_star, Merge=True)
        List_uc_dncdzeta = ass.uc_dncdzeta(MS.Surfs)
        List_nc_domegazetadzeta_vert = ass.nc_domegazetadzeta(MS.Surfs, MS.Surfs_star)
        for ss in range(MS.n_surf):
            List_nc_dqcdzeta[ss][ss] += \
                (List_uc_dncdzeta[ss] + List_nc_domegazetadzeta_vert[ss])
        Ducdzeta = np.block(List_nc_dqcdzeta)  # dense matrix
        List_nc_dqcdzeta = None
        List_uc_dncdzeta = None
        List_nc_domegazetadzeta_vert = None

        # ext velocity derivs (Wnv0)
        List_Wnv = []
        for ss in range(MS.n_surf):
            List_Wnv.append(
                interp.get_Wnv_vector(MS.Surfs[ss],
                                      MS.Surfs[ss].aM, MS.Surfs[ss].aN))
        AinvWnv0 = scalg.lu_solve((LU, P), scalg.block_diag(*List_Wnv))
        List_Wnv = None

        ### B matrix assembly
        Bss = []

        # non-penetration condition
        Bss.append([-scalg.lu_solve((LU, P), Ducdzeta), AinvWnv0, -AinvWnv0])
        AinvWnv0 = None

        # circulation eq.
        Bss.append([None, None, None])

        # delta eq.
        if self.integr_order == 1:
            Bss.append([bb.copy() for bb in Bss[0]])
        if self.integr_order == 2:
            Bss.append([bp1 * bb for bb in Bss[0]])

        # indentity eq
        if self.integr_order == 2:
            Bss.append([None, None, None])

        LU, P = None, None

        # ---------------------------------------------------------- output eq.

        ### state terms (C matrix)

        # gamma (induced velocity contrib.)
        List_dfqsdvind_gamma, List_dfqsdvind_gamma_star = \
            ass.dfqsdvind_gamma(MS.Surfs, MS.Surfs_star)

        # gamma (at constant relative velocity)
        List_dfqsdgamma_vrel0, List_dfqsdgamma_star_vrel0 = \
            ass.dfqsdgamma_vrel0(MS.Surfs, MS.Surfs_star)
        for ss in range(MS.n_surf):
            List_dfqsdvind_gamma[ss][ss] += List_dfqsdgamma_vrel0[ss]
            List_dfqsdvind_gamma_star[ss][ss] += List_dfqsdgamma_star_vrel0[ss]
        Dfqsdgamma = np.block(List_dfqsdvind_gamma)
        Dfqsdgamma_star = np.block(List_dfqsdvind_gamma_star)
        List_dfqsdvind_gamma, List_dfqsdvind_gamma_star = None, None
        List_dfqsdgamma_vrel0, List_dfqsdgamma_star_vrel0 = None, None

        # gamma_dot
        Dfunstdgamma_dot = scalg.block_diag(*ass.dfunstdgamma_dot(MS.Surfs))

        ### C matrix assembly
        Css = []
        Css.append([Dfqsdgamma, Dfqsdgamma_star, Dfunstdgamma_dot / self.dt])
        if self.integr_order == 2:
            Css[0].append(None)

        ### input terms (D matrix)
        Dss = []
        Dss.append(
            [scalg.block_diag(*ass.dfqsdzeta_vrel0(MS.Surfs, MS.Surfs_star))])

        # zeta (induced velocity contrib)
        List_coll, List_vert = ass.dfqsdvind_zeta(MS.Surfs, MS.Surfs_star)
        for ss in range(MS.n_surf):
            List_vert[ss][ss] += List_coll[ss]
        Dss[0][0] += np.block(List_vert)
        del List_vert, List_coll

        Dss[0].append(-scalg.block_diag(*ass.dfqsduinput(MS.Surfs, MS.Surfs_star)))
        Dss[0].append(-Dss[0][1])

        if self.remove_predictor:
            cout.cout_wrap("\t\tPredictor not be removed! " +
                           "(Though this is accounted for in all methods)", 1)

        self.SS = libss.ss_block(Ass, Bss, Css, Dss,
                                 self.S_x, self.S_u, self.S_y, dt=self.dt)
        cout.cout_wrap('\tstate-space model produced in form:\n\t' \
                       'x_{n+1} = A x_{n} + Bp u_{n+1}', 1)

        # add variable tracker
        self.SS.input_variables = LinearVector(self.input_variables_list)
        self.SS.state_variables = LinearVector(self.state_variables_list)
        self.SS.output_variables = LinearVector(self.output_variables_list)

        self.cpu_summary['assemble'] = time.time() - t0
        cout.cout_wrap('\t\t\t...done in %.2f sec' % self.cpu_summary['assemble'], 1)

    def freqresp(self, kv, wake_prop_settings=None):
        """
        Ad-hoc method for fast UVLM frequency response over the frequencies
        kv. The method, only requires inversion of a K x K matrix at each
        frequency as the equation for propagation of wake circulation are solved
        exactly.
        The algorithm implemented here can be used also upon projection of
        the state-space model.

        Note:
        This method is very similar to the "minsize" solution option is the
        steady_solve.
        """

        MS = self.MS
        K = self.K
        K_star = self.K_star

        Eye = np.eye(K)
        Bup = np.hstack(self.SS.B[0])
        P = self.SS.A[0][0]
        Pw = self.SS.A[0][1]

        Nk = len(kv)
        kvdt = kv * self.SS.dt
        zv = np.cos(kvdt) + 1.j * np.sin(kvdt)
        Yfreq = np.empty((self.SS.outputs, self.SS.inputs, Nk,), dtype=np.complex_)

        for kk in range(Nk):

            ###  build Cw complex
            Cw_cpx = self.get_Cw_cpx(zv[kk], settings=wake_prop_settings)

            Ygamma = libsp.solve(zv[kk] * Eye - P -
                                 libsp.dot(Pw, Cw_cpx, type_out=libsp.csc_matrix),
                                 Bup)
            if self.remove_predictor:
                Ygamma *= zv[kk]

            Ygamma_star = Cw_cpx.dot(Ygamma)

            if self.integr_order == 1:
                dfact = (1. - 1. / zv[kk])
            elif self.integr_order == 2:
                dfact = .5 * (3. - 4. / zv[kk] + 1. / zv[kk] ** 2)
            else:
                raise NameError('Specify valid integration order')

            # calculate solution
            Yfreq[:, :, kk] = np.dot(self.SS.C[0][0], Ygamma) + \
                              np.dot(self.SS.C[0][1], Ygamma_star) + \
                              np.dot(self.SS.C[0][2], dfact * Ygamma) + \
                              np.hstack(self.SS.D[0])

        return Yfreq

    def balfreq(self, DictBalFreq, wake_prop_settings=None):
        """
        Low-rank method for frequency limited balancing.
        The Observability ad controllability Gramians over the frequencies kv
        are solved in factorised form. Balancd modes are then obtained with a
        square-root method.

        Details:
        Observability and controllability Gramians are solved in factorised form
        through explicit integration. The number of integration points determines
        both the accuracy and the maximum size of the balanced model.

        Stability over all (Nb) balanced states is achieved if:
            a. one of the Gramian is integrated through the full Nyquist range
            b. the integration points are enough.
        Note, however, that even when stability is not achieved over the full
        balanced states, stability of the balanced truncated model with Ns<=Nb
        states is normally observed even when a low number of integration points
        is used. Two integration methods (trapezoidal rule on uniform grid and
        Gauss-Legendre quadrature) are provided.

        Input:

        - DictBalFreq: dictionary specifying integration method with keys:

            - 'frequency': defines limit frequencies for balancing. The balanced
            model will be accurate in the range [0,F], where F is the value of
            this key. Note that F units must be consistent with the units specified
            in the self.ScalingFacts dictionary.

            - 'method_low': ['gauss','trapz'] specifies whether to use gauss
            quadrature or trapezoidal rule in the low-frequency range [0,F]

            - 'options_low': options to use for integration in the low-frequencies.
            These depend on the integration scheme (See below).

            - 'method_high': method to use for integration in the range [F,F_N],
            where F_N is the Nyquist frequency. See 'method_low'.

            - 'options_high': options to use for integration in the high-frequencies.

            - 'check_stability': if True, the balanced model is truncated to
            eliminate unstable modes - if any is found. Note that very accurate
            balanced model can still be obtained, even if high order modes are
            unstable. Note that this option is overridden if ""

            - 'get_frequency_response': if True, the function also returns the
            frequency response evaluated at the low-frequency range integration
            points. If True, this option also allows to automatically tune the
            balanced model.

        Future options:

            - 'truncation_tolerance': if 'get_frequency_response' is True, allows
            to truncatethe balanced model so as to achieved a prescribed
            tolerance in the low-frequwncy range.

            - Ncpu: for parallel run


        The following integration schemes are available:
            - 'trapz': performs integration over equally spaced points using
            trapezoidal rule. It accepts options dictionaries with keys:
                - 'points': number of integration points to use (including
                domain boundary)

            - 'gauss' performs gauss-lobotto quadrature. The domain can be
            partitioned in Npart sub-domain in which the gauss-lobotto quadrature
            of order Ord can be applied. A total number of Npart*Ord points is
            required. It accepts options dictionaries of the form:
                - 'partitions': number of partitions
                - 'order': quadrature order.

        Example:
        The following dictionary

            DictBalFreq={   'frequency': 1.2,
                            'method_low': 'trapz',
                            'options_low': {'points': 12},
                            'method_high': 'gauss',
                            'options_high': {'partitions': 2, 'order': 8},
                            'check_stability': True }

        balances the state-space model self.SS in the frequency range [0, 1.2]
        using
            (a) 12 equally-spaced points integration of the Gramians in
        the low-frequency range [0,1.2] and
            (b) a 2 Gauss-Lobotto 8-th order quadratures of the controllability
            Gramian in the high-frequency range.

        A total number of 28 integration points will be required, which will
        result into a balanced model with number of states
            min{ 2*28* number_inputs, 2*28* number_outputs }
        The model is finally truncated so as to retain only the first Ns stable
        modes.
        """

        ### check input dictionary
        if 'frequency' not in DictBalFreq:
            raise NameError('Solution dictionary must include the "frequency" key')

        if 'method_low' not in DictBalFreq:
            warnings.warn('Setting default options for low-frequency integration')
            DictBalFreq['method_low'] = 'trapz'
            DictBalFreq['options_low'] = {'points': 12}

        if 'method_high' not in DictBalFreq:
            warnings.warn('Setting default options for high-frequency integration')
            DictBalFreq['method_high'] = 'gauss'
            DictBalFreq['options_high'] = {'partitions': 2, 'order': 8}

        if 'check_stability' not in DictBalFreq:
            DictBalFreq['check_stability'] = True

        if 'output_modes' not in DictBalFreq:
            DictBalFreq['output_modes'] = True

        if 'get_frequency_response' not in DictBalFreq:
            DictBalFreq['get_frequency_response'] = False

        ### get integration points and weights

        # Nyquist frequency
        kn = np.pi / self.SS.dt

        Opt = DictBalFreq['options_low']
        if DictBalFreq['method_low'] == 'trapz':
            kv_low, wv_low = librom.get_trapz_weights(0., DictBalFreq['frequency'],
                                                      Opt['points'], False)
        elif DictBalFreq['method_low'] == 'gauss':
            kv_low, wv_low = librom.get_gauss_weights(0., DictBalFreq['frequency'],
                                                      Opt['partitions'], Opt['order'])
        else:
            raise NameError(
                'Invalid value %s for key "method_low"' % DictBalFreq['method_low'])

        Opt = DictBalFreq['options_high']
        if DictBalFreq['method_high'] == 'trapz':
            if Opt['points'] == 0:
                warnings.warn('You have chosen no points in high frequency range!')
                kv_high, wv_high = [], []
            else:
                kv_high, wv_high = librom.get_trapz_weights(DictBalFreq['frequency'], kn,
                                                            Opt['points'], True)
        elif DictBalFreq['method_high'] == 'gauss':
            if Opt['order'] * Opt['partitions'] == 0:
                warnings.warn('You have chosen no points in high frequency range!')
                kv_high, wv_high = [], []
            else:
                kv_high, wv_high = librom.get_gauss_weights(DictBalFreq['frequency'], kn,
                                                            Opt['partitions'], Opt['order'])
        else:
            raise NameError(
                'Invalid value %s for key "method_high"' % DictBalFreq['method_high'])

        ### get useful terms
        K = self.K
        K_star = self.K_star

        Eye = np.eye(K)
        Bup = np.hstack(self.SS.B[0])

        P = self.SS.A[0][0]
        Pw = self.SS.A[0][1]

        # indices to manipulate obs solution
        ii00 = range(0, self.K)
        ii01 = range(self.K, self.K + self.K_star)
        ii02 = range(self.K + self.K_star, 2 * self.K + self.K_star)
        ii03 = range(2 * self.K + self.K_star, 3 * self.K + self.K_star)

        # integration factors
        if self.integr_order == 2:
            b0, bm1, bp1 = -2., 0.5, 1.5
        else:
            b0, bp1 = -1., 1.
            raise NameError('Method not implemented for integration order 1')

        ### -------------------------------------------------- loop frequencies

        ### merge vectors
        Nk_low = len(kv_low)
        kvdt = np.concatenate((kv_low, kv_high)) * self.SS.dt
        wv = np.concatenate((wv_low, wv_high)) * self.SS.dt
        zv = np.cos(kvdt) + 1.j * np.sin(kvdt)

        Qobs = np.zeros((self.SS.states, self.SS.outputs), dtype=np.complex_)
        Zc = np.zeros((self.SS.states, 2 * self.SS.inputs * len(kvdt)), )
        Zo = np.zeros((self.SS.states, 2 * self.SS.outputs * Nk_low), )

        if DictBalFreq['get_frequency_response']:
            self.Yfreq = np.empty((self.SS.outputs, self.SS.inputs, Nk_low,), dtype=np.complex_)
            self.kv = kv_low

        for kk in range(len(kvdt)):

            zval = zv[kk]
            Intfact = wv[kk]  # integration factor

            #  build terms that will be recycled
            Cw_cpx = self.get_Cw_cpx(zval, settings=wake_prop_settings)
            P_PwCw = P + Cw_cpx.T.dot(Pw.T).T
            Kernel = np.linalg.inv(zval * Eye - P_PwCw)

            ### ----- controllability
            Ygamma = Intfact * np.dot(Kernel, Bup)
            if self.remove_predictor:
                Ygamma *= zval
            Ygamma_star = Cw_cpx.dot(Ygamma)

            if self.integr_order == 1:
                dfact = (bp1 + bp0 / zval)
                Qctrl = np.vstack([Ygamma, Ygamma_star, dfact * Ygamma])
            elif self.integr_order == 2:
                dfact = bp1 + b0 / zval + bm1 / zval ** 2
                Qctrl = np.vstack(
                    [Ygamma, Ygamma_star, dfact * Ygamma, (1. / zval) * Ygamma])
            else:
                raise NameError('Specify valid integration order')

            kkvec = range(2 * kk * self.SS.inputs, 2 * (kk + 1) * self.SS.inputs)
            Zc[:, kkvec[:self.SS.inputs]] = Qctrl.real  # *Intfact
            Zc[:, kkvec[self.SS.inputs:]] = Qctrl.imag  # *Intfact

            ### ----- frequency response
            if DictBalFreq['get_frequency_response'] and kk < Nk_low:
                self.Yfreq[:, :, kk] = (1. / Intfact) * \
                                       (np.dot(self.SS.C[0][0], Ygamma) + \
                                        np.dot(self.SS.C[0][1], Ygamma_star) + \
                                        dfact * np.dot(self.SS.C[0][2], Ygamma)) + \
                                       np.hstack(self.SS.D[0])

            ### ----- frequency response
            if DictBalFreq['get_frequency_response'] and kk < Nk_low:
                self.Yfreq[:, :, kk] = (1. / Intfact) * \
                                       (np.dot(self.SS.C[0][0], Ygamma) + \
                                        np.dot(self.SS.C[0][1], Ygamma_star) + \
                                        dfact * np.dot(self.SS.C[0][2], Ygamma)) + \
                                       np.hstack(self.SS.D[0])

            ### ----- observability
            # solve (1./zval*I - A.T)^{-1} C^T (in low-frequency only)
            if kk >= Nk_low:
                continue

            zinv = 1. / zval
            Qobs[ii02, :] = zinv * self.SS.C[0][2].T
            if self.integr_order == 1:
                raise NameError('Obs Gramian Integr not implemented')
            elif self.integr_order == 2:
                Qobs[ii03, :] = (bm1 * zinv) * Qobs[ii02, :]

            # solve bound circulation
            rhs = Cw_cpx.T.dot(self.SS.C[0][1].T) + self.SS.C[0][0].T + \
                  Qobs[ii02, :] * (b0 + zinv * bm1) + \
                  np.dot(P_PwCw.T, bp1 * Qobs[ii02, :])
            Qobs[ii00, :] = np.dot(Kernel.T, rhs)

            # solve wake
            Eye_star = libsp.csc_matrix(
                (zval * np.ones((K_star,)), (range(K_star), range(K_star))),
                shape=(K_star, K_star), dtype=np.complex_)
            Qobs[ii01, :] = libsp.solve(
                Eye_star - self.SS.A[1][1].T,
                self.SS.C[0][1].T + np.dot(Pw.T, Qobs[ii00, :] + bp1 * Qobs[ii02, :]))

            kkvec = range(2 * kk * self.SS.outputs, 2 * (kk + 1) * self.SS.outputs)
            Zo[:, kkvec[:self.SS.outputs]] = Intfact * Qobs.real
            Zo[:, kkvec[self.SS.outputs:]] = Intfact * Qobs.imag

        # delete full matrices
        Kernel = None
        Qctrl = None
        Qobs = None

        # LRSQM (optimised)
        U, hsv, Vh = scalg.svd(np.dot(Zo.T, Zc), full_matrices=False)
        sinv = hsv ** (-0.5)
        T = np.dot(Zc, Vh.T * sinv)
        Ti = np.dot((U * sinv).T, Zo.T)

        ### build frequency balanced model
        Ab, Bb, Cb = self.SS.project(Ti, T, by_arrays=True, overwrite=False)

        if self.remove_predictor:
            Ab, Bb, Cb, Db = \
                libss.SSconv(np.block(Ab), None, np.block(Bb),
                             np.block(Cb), np.block(self.SS.D), Bm1=None)
            SSb = libss.StateSpace(Ab, Bb, Cb, Db, dt=self.dt)
        else:
            SSb = libss.StateSpace(np.block(Ab), np.block(Bb),
                                   np.block(Cb), np.block(self.SS.D), dt=self.SS.dt)

        ### Eliminate unstable modes - if any:
        if DictBalFreq['check_stability']:
            for nn in range(1, len(hsv) + 1):
                eigs_trunc = scalg.eigvals(SSb.A[:nn, :nn])
                eigs_trunc_max = np.max(np.abs(eigs_trunc))
                if eigs_trunc_max > 1. - 1e-16:
                    SSb.truncate(nn - 1)
                    hsv = hsv[:nn - 1]
                    T = T[:, :nn - 1]
                    Ti = Ti[:nn - 1, :]
                    break

        self.SSb = SSb
        self.hsv = hsv
        if DictBalFreq['output_modes']:
            self.T = T
            self.Ti = Ti
            self.Zc = Zc
            self.Zo = Zo
            self.svd_res = {'U': U, 'hsv': hsv, 'Vh': Vh}

    def solve_step(self, x_n, u_n, u_n1=None, transform_state=False):
        r"""
        Solve step.

        If the predictor term has not been removed (``remove_predictor = False``) then the system is solved as:

            .. math::
                \mathbf{x}^{n+1} &= \mathbf{A\,x}^n + \mathbf{B\,u}^n \\
                \mathbf{y}^{n+1} &= \mathbf{C\,x}^{n+1} + \mathbf{D\,u}^n

        Else, if ``remove_predictor = True``, the state is modified as

            ..  math:: \mathbf{h}^n = \mathbf{x}^n - \mathbf{B\,u}^n

        And the system solved by:

            .. math::
                \mathbf{h}^{n+1} &= \mathbf{A\,h}^n + \mathbf{B_{mod}\,u}^{n} \\
                \mathbf{y}^{n+1} &= \mathbf{C\,h}^{n+1} + \mathbf{D_{mod}\,u}^{n+1}

        Finally, the original state is recovered using the reverse transformation:

            .. math:: \mathbf{x}^{n+1} = \mathbf{h}^{n+1} + \mathbf{B\,u}^{n+1}

        where the modifications to the :math:`\mathbf{B}_{mod}` and :math:`\mathbf{D}_{mod}` are detailed in
        :func:`Dynamic.assemble_ss`.

        Notes:
            Although the original equations include the term :math:`\mathbf{u}_{n+1}`, it is a reasonable approximation
            to take :math:`\mathbf{u}_{n+1}\approx\mathbf{u}_n` given a sufficiently small time step, hence if the input
            at time ``n+1`` is not parsed, it is estimated from :math:`u^n`.

        Args:
            x_n (np.array): State vector at the current time step :math:`\mathbf{x}^n`
            u_n (np.array): Input vector at time step :math:`\mathbf{u}^n`
            u_n1 (np.array): Input vector at time step :math:`\mathbf{u}^{n+1}`
            transform_state (bool): When the predictor term is removed, if true it will transform the state vector. If
                false it will be assumed that the state vector that is parsed is already transformed i.e. it is
                :math:`\mathbf{h}`.

        Returns:
            Tuple: Updated state and output vector packed in a tuple :math:`(\mathbf{x}^{n+1},\,\mathbf{y}^{n+1})`

        Notes:
            Because in BlockDynamics the predictor is never removed when building
            'self.SS', the implementation change with respect to Dynamic. However,
            formulas are consistent.
        """

        if u_n1 is None:
            u_n1 = u_n.copy()

        if self.remove_predictor and not hasattr(self, 'CBplusD'):
            self.CBplusD = libsp.block_sum(libsp.block_dot(self.SS.C, self.SS.B), self.SS.D)

        ### transform input in block matrices
        X_n = []
        II0 = 0
        for ii in range(self.SS.blocks_x):
            IIend = II0 + self.SS.S_x[ii]
            X_n.append([x_n[II0:IIend]])
            II0 = IIend

        U_n1 = []
        II0 = 0
        for ii in range(self.SS.blocks_u):
            IIend = II0 + self.SS.S_u[ii]
            U_n1.append([u_n1[II0:IIend]])
            II0 = IIend

        if u_n is not None:
            U_n = []
            for ii in range(self.SS.blocks_u):
                IIend = II0 + self.SS.S_u[ii]
                U_n.append([u_n[II0:IIend]])
                II0 = IIend

        if self.remove_predictor:

            # Transform state vector
            # TODO: Agree on a way to do this. Either always transform here or transform prior to using the method.
            if transform_state:
                H_n1 = libsp.block_dot(self.SS.A, X_n)
            else:
                H_n = X_n
                H_n1 = libsp.block_dot(self.SS.A,
                                       libsp.block_sum(H_n, libsp.block_dot(self.SS.B, U_n)))

            Y_n1 = libsp.block_sum(libsp.block_dot(self.SS.C, H_n1),
                                   libsp.block_dot(self.CBplusD, U_n1))

            Y_n1 = libsp.block_sum(libsp.block_dot(self.SS.C, H_n1),
                                   libsp.block_dot(self.CBplusD, U_n1))

            # Recover state vector
            if transform_state:
                X_n1 = libsp.block_sum(H_n1, libsp.block_dot(self.SS.B, U_n1))
            else:
                X_n1 = H_n1

        else:
            X_n1 = libsp.block_sum(libsp.block_dot(self.SS.A, X_n),
                                   libsp.block_dot(self.SS.B, U_n1))
            Y_n1 = libsp.block_sum(libsp.block_dot(self.SS.C, X_n1),
                                   libsp.block_dot(self.SS.D, U_n1))

        x_n1 = np.concatenate([X_n1[ii][0] for ii in range(self.SS.blocks_x)])

        return x_n1, np.block(Y_n1).reshape(-1)


################################################################################

################################################################################


class Frequency(Static):
    """
    Class for frequency description of linearised UVLM solution. Linearisation
    around steady-state are only supported. The class is built upon Static, and
    inherits all the methods contained there.

    The class supports most of the features of Dynamics but has lower memory
    requirements of Dynamic, and should be preferred for:

        a. producing  memory and computationally cheap frequency responses
        b. building reduced order models using RFA/polynomial fitting

    Usage:
    Upon initialisation, the assemble method produces all the matrices required
    for the frequency description of the UVLM (see assemble for details). A
    state-space model is not allocated but:

        - Time stepping is also possible (but not implemented yet) as all the
          fundamental terms describing the UVLM equations are still produced
          (except the propagation of wake circulation)
        - ad-hoc methods for scaling, unscaling and frequency response are
          provided.

    Input:

        - tsdata: aero timestep data from SHARPy solution
        - dt: time-step
        - integr_order=0,1,2: integration order for UVLM unsteady aerodynamic
          force. If 0, the derivative is computed exactly.
        - RemovePredictor=True: This flag is only used for the frequency response
          calculation. The frequency description, in fact, naturally arises
          without the predictor, but lags can be included during the frequency
          response calculation. See Dynamic documentation for more details.
        - ScalingDict=None: disctionary containing fundamental reference units

            .. code-block:: python

                {'length':  reference_length,
                 'speed':   reference_speed,
                 'density': reference density}

          used to derive scaling quantities for the state-space model variables.
          The scaling factors are stores in ``self.ScalingFact.``

          Note that while time, circulation, angular speeds) are scaled
          accordingly, FORCES ARE NOT. These scale by qinf*b**2, where b is the
          reference length and qinf is the dinamic pressure.
        - UseSparse=False: builds the A and B matrices in sparse form. C and D
          are dense, hence the sparce format is not used.

    Methods:
        - nondimss: normalises matrices produced by the assemble method  based
          on the scaling factors in self.ScalingFact.
        - dimss: inverse of nondimss.
        - assemble: builds matrices for UVLM minimal size description.
        - assemble_profiling: generate profiling report of the assembly and
          saves it into self.prof_out. To read the report:

            .. code-block:: python

                import pstats
                p=pstats.Stats(self.prof_out)

        - freqresp: fast algorithm for frequency response.

    Methods to implement:

        - solve_steady: runs freqresp at 0 frequency.
        - solve_step: solves one time-step
    """

    def __init__(self, tsdata, dt, integr_order=2,
                 RemovePredictor=True, ScalingDict=None, UseSparse=True):

        super().__init__(tsdata)

        self.dt = dt
        self.integr_order = integr_order

        assert self.integr_order in [1, 2, 0], 'integr_order must be in [0,1,2]'

        self.inputs = 9 * self.Kzeta
        self.outputs = 3 * self.Kzeta

        self.remove_predictor = RemovePredictor
        self.use_sparse = UseSparse

        # create scaling quantities
        if ScalingDict is None:
            ScalingFacts = {'length': 1.,
                            'speed': 1.,
                            'density': 1.}
        else:
            ScalingFacts = ScalingDict

        for key in ScalingFacts:
            ScalingFacts[key] = np.float(ScalingFacts[key])

        ScalingFacts['time'] = ScalingFacts['length'] / ScalingFacts['speed']
        ScalingFacts['circulation'] = ScalingFacts['speed'] * ScalingFacts['length']
        ScalingFacts['dyn_pressure'] = 0.5 * ScalingFacts['density'] * ScalingFacts['speed'] ** 2
        ScalingFacts['force'] = ScalingFacts['dyn_pressure'] * ScalingFacts['length'] ** 2
        self.ScalingFacts = ScalingFacts

        ### collect statistics
        self.cpu_summary = {'dim': 0.,
                            'nondim': 0.,
                            'assemble': 0.}

    def nondimss(self):
        """
        Scale state-space model based of self.ScalingFacts
        """

        t0 = time.time()
        Kzeta = self.Kzeta

        self.Bss[:, :3 * Kzeta] *= (self.ScalingFacts['length'] / self.ScalingFacts['circulation'])
        self.Bss[:, 3 * Kzeta:] *= (self.ScalingFacts['speed'] / self.ScalingFacts['circulation'])

        self.Css *= (self.ScalingFacts['circulation'] / self.ScalingFacts['force'])

        self.Dss[:, :3 * Kzeta] *= (self.ScalingFacts['length'] / self.ScalingFacts['force'])
        self.Dss[:, 3 * Kzeta:] *= (self.ScalingFacts['speed'] / self.ScalingFacts['force'])

        self.dt = self.dt / self.ScalingFacts['time']

        self.cpu_summary['nondim'] = time.time() - t0

    def dimss(self):

        t0 = time.time()
        Kzeta = self.Kzeta

        self.Bss[:, :3 * Kzeta] /= (self.ScalingFacts['length'] / self.ScalingFacts['circulation'])
        self.Bss[:, 3 * Kzeta:] /= (self.ScalingFacts['speed'] / self.ScalingFacts['circulation'])

        self.Css /= (self.ScalingFacts['circulation'] / self.ScalingFacts['force'])

        self.Dss[:, :3 * Kzeta] /= (self.ScalingFacts['length'] / self.ScalingFacts['force'])
        self.Dss[:, 3 * Kzeta:] /= (self.ScalingFacts['speed'] / self.ScalingFacts['force'])

        self.dt = self.dt * self.ScalingFacts['time']

        self.cpu_summary['dim'] = time.time() - t0

    def assemble(self):
        r"""
        Assembles matrices for minumal size frequency description of UVLM. The
        state equation is represented in the form:

            .. math:: \mathbf{A_0} \mathbf{\Gamma} +
                            \mathbf{A_{w_0}} \mathbf{\Gamma_w} =
                                                \mathbf{B_0} \mathbf{u}

        While the output equation is as per the Dynamic class, namely:

            .. math:: \mathbf{y} =
                            \mathbf{C} \mathbf{x} + \mathbf{D} \mathbf{u}

        where

            .. math:: \mathbf{x} =
                     [\mathbf{\Gamma}; \mathbf{\Gamma_w}; \Delta\mathbf(\Gamma)]

        The propagation of wake circulation matrices are not produced as these
        are not required for frequency response analysis.
        """

        cout.cout_wrap('\tAssembly of frequency description of UVLM started...', 1)
        t0 = time.time()
        MS = self.MS
        K, K_star = self.K, self.K_star
        Kzeta = self.Kzeta

        Nu = self.inputs
        Ny = self.outputs

        # ----------------------------------------------------------- state eq.

        # Aero influence coeffs
        List_AICs, List_AICs_star = ass.AICs(MS.Surfs, MS.Surfs_star,
                                             target='collocation', Project=True)
        A0 = np.block(List_AICs)
        A0W = np.block(List_AICs_star)
        List_AICs, List_AICs_star = None, None

        # zeta derivs
        List_nc_dqcdzeta = ass.nc_dqcdzeta(MS.Surfs, MS.Surfs_star, Merge=True)
        List_uc_dncdzeta = ass.uc_dncdzeta(MS.Surfs)
        List_nc_domegazetadzeta_vert = ass.nc_domegazetadzeta(MS.Surfs, MS.Surfs_star)
        for ss in range(MS.n_surf):
            List_nc_dqcdzeta[ss][ss] += \
                (List_uc_dncdzeta[ss] + List_nc_domegazetadzeta_vert[ss])
        Ducdzeta = np.block(List_nc_dqcdzeta)  # dense matrix
        List_nc_dqcdzeta = None
        List_uc_dncdzeta = None
        List_nc_domegazetadzeta_vert = None

        # ext velocity derivs (Wnv0)
        List_Wnv = []
        for ss in range(MS.n_surf):
            List_Wnv.append(
                interp.get_Wnv_vector(MS.Surfs[ss],
                                      MS.Surfs[ss].aM, MS.Surfs[ss].aN))
        Wnv0 = scalg.block_diag(*List_Wnv)
        List_Wnv = None

        ### B matrix assembly
        # this could be also sparse...
        Bss = np.block([-Ducdzeta, Wnv0, -Wnv0])

        # ---------------------------------------------------------- output eq.

        ### state terms (C matrix)

        # gamma (induced velocity contrib.)
        List_dfqsdvind_gamma, List_dfqsdvind_gamma_star = \
            ass.dfqsdvind_gamma(MS.Surfs, MS.Surfs_star)

        # gamma (at constant relative velocity)
        List_dfqsdgamma_vrel0, List_dfqsdgamma_star_vrel0 = \
            ass.dfqsdgamma_vrel0(MS.Surfs, MS.Surfs_star)
        for ss in range(MS.n_surf):
            List_dfqsdvind_gamma[ss][ss] += List_dfqsdgamma_vrel0[ss]
            List_dfqsdvind_gamma_star[ss][ss] += List_dfqsdgamma_star_vrel0[ss]
        Dfqsdgamma = np.block(List_dfqsdvind_gamma)
        Dfqsdgamma_star = np.block(List_dfqsdvind_gamma_star)
        List_dfqsdvind_gamma, List_dfqsdvind_gamma_star = None, None
        List_dfqsdgamma_vrel0, List_dfqsdgamma_star_vrel0 = None, None

        # gamma_dot
        Dfunstdgamma_dot = scalg.block_diag(*ass.dfunstdgamma_dot(MS.Surfs))

        # C matrix assembly
        Css = np.zeros((Ny, 2 * K + K_star))
        Css[:, :K] = Dfqsdgamma
        Css[:, K:K + K_star] = Dfqsdgamma_star
        # added mass
        Css[:, K + K_star:2 * K + K_star] = Dfunstdgamma_dot / self.dt

        ### input terms (D matrix)
        Dss = np.zeros((Ny, Nu))

        # zeta (at constant relative velocity)
        Dss[:, :3 * Kzeta] = scalg.block_diag(
            *ass.dfqsdzeta_vrel0(MS.Surfs, MS.Surfs_star))
        # zeta (induced velocity contrib)
        List_coll, List_vert = ass.dfqsdvind_zeta(MS.Surfs, MS.Surfs_star)
        for ss in range(MS.n_surf):
            List_vert[ss][ss] += List_coll[ss]
        Dss[:, :3 * Kzeta] += np.block(List_vert)
        del List_vert, List_coll

        # input velocities (external)
        Dss[:, 6 * Kzeta:9 * Kzeta] = scalg.block_diag(
            *ass.dfqsduinput(MS.Surfs, MS.Surfs_star))

        # input velocities (body movement)
        Dss[:, 3 * Kzeta:6 * Kzeta] = -Dss[:, 6 * Kzeta:9 * Kzeta]

        ### store matrices
        self.A0 = A0
        self.A0W = A0W
        self.Bss = Bss
        self.Css = Css
        self.Dss = Dss
        self.inputs = 9 * Kzeta
        self.outputs = 3 * Kzeta

        self.cpu_summary['assemble'] = time.time() - t0
        cout.cout_wrap('\t\t\t...done in %.2f sec' % self.cpu_summary['assemble'], 1)

    def addGain(self, K, where):

        assert where in ['in', 'out'], \
            'Specify whether gains are added to input or output'

        if where == 'in':
            self.Bss = libsp.dot(self.Bss, K)
            self.Dss = libsp.dot(self.Dss, K)
            self.inputs = K.shape[1]

        if where == 'out':
            self.Css = libsp.dot(K, self.Css)
            self.Dss = libsp.dot(K, self.Dss)
            self.outputs = K.shape[0]

        if where == 'out':
            self.Css = libsp.dot(K, self.Css)
            self.Dss = libsp.dot(K, self.Dss)
            self.outputs = K.shape[0]

    def freqresp(self, kv, wake_prop_settings=None):
        """
        Ad-hoc method for fast UVLM frequency response over the frequencies
        kv. The method, only requires inversion of a K x K matrix at each
        frequency as the equation for propagation of wake circulation are solved
        exactly.
        """

        MS = self.MS
        K = self.K
        K_star = self.K_star

        Nk = len(kv)
        kvdt = kv * self.dt
        zv = np.cos(kvdt) + 1.j * np.sin(kvdt)
        Yfreq = np.empty((self.outputs, self.inputs, Nk,), dtype=np.complex_)

        ### loop frequencies
        for kk in range(Nk):

            ### build Cw complex
            Cw_cpx = self.get_Cw_cpx(zv[kk], settings=wake_prop_settings)

            # get bound state freq response
            if self.remove_predictor:
                Ygamma = np.linalg.solve(
                    self.A0 + libsp.dot(
                        self.A0W, Cw_cpx, type_out=libsp.csc_matrix), self.Bss)
            else:
                Ygamma = zv[kk] ** (-1) * \
                         np.linalg.solve(
                             self.A0 + libsp.dot(
                                 self.A0W, Cw_cpx, type_out=libsp.csc_matrix), self.Bss)
            Ygamma_star = Cw_cpx.dot(Ygamma)

            # determine factor for delta of bound circulation
            if self.integr_order == 0:
                dfact = (1.j * kv[kk]) * self.dt
            elif self.integr_order == 1:
                dfact = (1. - 1. / zv[kk])
            elif self.integr_order == 2:
                dfact = .5 * (3. - 4. / zv[kk] + 1. / zv[kk] ** 2)
            else:
                raise NameError('Specify valid integration order')

            Yfreq[:, :, kk] = np.dot(self.Css[:, :K], Ygamma) + \
                              np.dot(self.Css[:, K:K + K_star], Ygamma_star) + \
                              np.dot(self.Css[:, -K:], dfact * Ygamma) + \
                              self.Dss

        return Yfreq

    def get_Cw_cpx(self, zval, settings=None):
        r"""
        Produces a sparse matrix

            .. math:: \bar{\mathbf{C}}(z)

        where

            .. math:: z = e^{k \Delta t}

        such that the wake circulation frequency response at :math:`z` is

            .. math:: \bar{\goldsymbol{\Gamma}}_w = \bar{\mathbf{C}}(z)  \bar{\boldsymbol{\Gamma}}

        """
        return get_Cw_cpx(self.MS, self.K, self.K_star, zval, settings=settings)


    def assemble_profiling(self):
        """
        Generate profiling report for assembly and save it in self.prof_out.

        To read the report:
            import pstats
            p=pstats.Stats(self.prof_out)
        """

        import cProfile
        cProfile.runctx('self.assemble()', globals(), locals(), filename=self.prof_out)


def get_Cw_cpx(MS, K, K_star, zval, settings=None):
    r"""
    Produces a sparse matrix

        .. math:: \bar{\mathbf{C}}(z)

    where

        .. math:: z = e^{k \Delta t}

    such that the wake circulation frequency response at :math:`z` is

        .. math:: \bar{\boldsymbol{\Gamma}}_w = \bar{\mathbf{C}}(z)  \bar{\mathbf{\Gamma}}

    """

    try:
        cfl1 = settings['cfl1']
    except (KeyError, TypeError):
        # In case the key does not exist or settings=None
        cfl1 = True
    cout.cout_wrap("Computing wake propagation solution matrix if frequency domain with CFL1=%s" % cfl1, 1)
    # print("Computing wake propagation solution matrix if frequency domain with CFL1=%s" % cfl1)

    if cfl1:
        jjvec = []
        iivec = []
        valvec = []

        K0tot, K0totstar = 0, 0
        for ss in range(MS.n_surf):

            M, N = MS.dimensions[ss]
            Mstar, N = MS.dimensions_star[ss]

            for mm in range(Mstar):
                jjvec += range(K0tot + N * (M - 1), K0tot + N * M)
                iivec += range(K0totstar + mm * N, K0totstar + (mm + 1) * N)
                valvec += N * [zval ** (-mm - 1)]
            K0tot += MS.KK[ss]
            K0totstar += MS.KK_star[ss]
    else:
        # sum_m_n = 0
        sum_mstar_n = 0
        for ss in range(MS.n_surf):
            # M, N = MS.dimensions[ss]
            Mstar, N = MS.dimensions_star[ss]
            # sum_m_n += M*N
            sum_mstar_n += Mstar*N

            try:
                MS.Surfs_star[ss].zetac
            except AttributeError:
                MS.Surfs_star[ss].zetac.generate_collocations()
        jjvec = [None]*sum_mstar_n
        iivec = [None]*sum_mstar_n
        valvec = [None]*sum_mstar_n

        K0tot, K0totstar = 0, 0
        ipoint = 0
        for ss in range(MS.n_surf):
            M, N = MS.dimensions[ss]
            Mstar, N = MS.dimensions_star[ss]
            Surf = MS.Surfs[ss]
            Surf_star = MS.Surfs_star[ss]
            for iin in range(N):
                for mm in range(Mstar):
                    # Value location in the sparse array
                    ipoint = K0totstar + mm * N + iin
                    # Compute CFL
                    if mm == 0:
                        conv_vec = Surf_star.zetac[:, 0, iin] - Surf.zetac[:, -1, iin]
                        dist = np.linalg.norm(conv_vec)
                        conv_dir_te = conv_vec/dist
                        vel = Surf.u_input_coll[:, -1, iin]
                        vel_value = np.dot(vel, conv_dir_te)
                        cfl = settings['dt']*vel_value/dist
                    else:
                        conv_vec = Surf_star.zetac[:, mm, iin] - Surf_star.zetac[:, mm - 1, iin]
                        dist = np.linalg.norm(conv_vec)
                        conv_dir = conv_vec/dist
                        vel = Surf.u_input_coll[:, -1, iin]
                        vel_value = np.dot(vel, conv_dir_te)
                        cfl = settings['dt']*vel_value/dist
                    # Compute coefficient
                    coef = get_Cw_cpx_coef_cfl_n1(cfl, zval)
                    # Assign values
                    jjvec[ipoint] = K0tot + N * (M - 1) + iin
                    iivec[ipoint] = K0totstar + mm * N + iin
                    if mm == 0:
                        # First row
                        valvec[ipoint] = coef
                    else:
                        ipoint_prev = K0totstar + (mm - 1) * N + iin
                        valvec[ipoint] = coef*valvec[ipoint_prev]
            K0tot += MS.KK[ss]
            K0totstar += MS.KK_star[ss]

    return libsp.csc_matrix((valvec, (iivec, jjvec)), shape=(K_star, K), dtype=np.complex_)


def get_Cw_cpx_coef_cfl_n1(cfl, zval):
    # Convergence loop end criteria
    tol = 1e-12
    rmax = 100

    # Initial values
    coef = 0.
    r = 0

    # Loop
    error = 2*tol
    while ((error > tol) and (r < rmax)):
        delta_coef = ((1 - cfl)**r)*cfl*(zval**(-r-1))
        coef += delta_coef
        error = np.abs(delta_coef/coef)
        r += 1
    coef /= (1 - (1-cfl)**rmax*(zval**(-1)))
    if (error > tol):
        cout.cout_wrap(("WARNING computation of Cw_cpx did not reach desired accuracy. r: %d. error: %d" % (r, error)), 2)

    return coef

################################################################################


if __name__ == '__main__':

    import unittest
    from sharpy.utils.sharpydir import SharpyDir
    import sharpy.utils.h5utils as h5


    class Test_linuvlm_Sta_vs_Dyn(unittest.TestCase):
        """ Test methods into this module """

        def setUp(self):
            fname = SharpyDir + '/sharpy/linear/test/h5input/' + \
                    'goland_mod_Nsurf02_M003_N004_a040.aero_state.h5'
            haero = h5.readh5(fname)
            tsdata = haero.ts00000

            # Static solver
            Sta = Static(tsdata)
            Sta.assemble_profiling()
            Sta.assemble()
            Sta.get_total_forces_gain()

            # random input
            Sta.u_ext = 1.0 + 0.30 * np.random.rand(3 * Sta.Kzeta)
            Sta.zeta_dot = 0.2 + 0.10 * np.random.rand(3 * Sta.Kzeta)
            Sta.zeta = 0.05 * (np.random.rand(3 * Sta.Kzeta) - 1.0)

            Sta.solve()
            Sta.reshape()
            Sta.total_forces()
            self.Sta = Sta
            self.tsdata = tsdata

        def test_force_gains(self):
            """
            to do: add check on moments gain
            """
            Sta = self.Sta
            Ftot02 = libsp.dot(Sta.Kftot, Sta.fqs)
            assert np.max(np.abs(Ftot02 - Sta.Ftot)) < 1e-10, 'Total force gain matrix wrong!'

        def test_Dyn_steady_state(self):
            """
            Test steady state predicted by Dynamic and Static classes are the same.
            """

            Sta = self.Sta
            Order = [2, 1]
            RemPred = [True, False]
            UseSparse = [True, False]

            for order in Order:
                for rem_pred in RemPred:
                    for use_sparse in UseSparse:

                        # Dynamic solver
                        Dyn = Dynamic(self.tsdata,
                                      dt=0.05,
                                      integr_order=order,
                                      RemovePredictor=rem_pred,
                                      UseSparse=use_sparse)
                        Dyn.assemble_ss()

                        # steady state solution
                        usta = np.concatenate((Sta.zeta, Sta.zeta_dot, Sta.u_ext))
                        xsta, ysta = Dyn.solve_steady(usta, method='direct')

                        if use_sparse is False and rem_pred is False:
                            xmin, ymin = Dyn.solve_steady(usta, method='minsize')
                            xrec, yrec = Dyn.solve_steady(usta, method='recursive')
                            xsub, ysub = Dyn.solve_steady(usta, method='subsystem')

                            # assert all solutions are matching
                            assert max(np.linalg.norm(xsta - xmin), np.linalg.norm(ysta - ymin)), \
                                'Direct and min. size solutions not matching!'
                            assert max(np.linalg.norm(xsta - xrec), np.linalg.norm(ysta - yrec)), \
                                'Direct and recursive solutions not matching!'
                            assert max(np.linalg.norm(xsta - xsub), np.linalg.norm(ysta - ysub)), \
                                'Direct and sub-system solutions not matching!'

                        # compare against Static solver solution
                        er = np.max(np.abs(ysta - Sta.fqs) / np.linalg.norm(Sta.Ftot))
                        print('Error force distribution: %.3e' % er)
                        assert er < 1e-12, \
                            'Steady-state force not matching (error: %.2e)!' % er

                        if rem_pred is False:  # compare state

                            er = np.max(np.abs(xsta[:Dyn.K] - Sta.gamma))
                            print('Error bound circulation: %.3e' % er)
                            assert er < 1e-13, \
                                'Steady-state gamma not matching (error: %.2e)!' % er

                            gammaw_ref = np.zeros((Dyn.K_star,))
                            kk = 0
                            for ss in range(Dyn.MS.n_surf):
                                Mstar = Dyn.MS.MM_star[ss]
                                Nstar = Dyn.MS.NN_star[ss]
                                for mm in range(Mstar):
                                    gammaw_ref[kk:kk + Nstar] = Sta.Gamma[ss][-1, :]
                                    kk += Nstar

                            er = np.max(np.abs(xsta[Dyn.K:Dyn.K + Dyn.K_star] - gammaw_ref))
                            print('Error wake circulation: %.3e' % er)
                            assert er < 1e-13, 'Steady-state gamma_star not matching!'

                            er = np.max(np.abs(xsta[Dyn.K + Dyn.K_star:2 * Dyn.K + Dyn.K_star]))
                            print('Error bound derivative: %.3e' % er)
                            assert er < 1e-13, 'Non-zero derivative of circulation at steady state!'

                            if Dyn.integr_order == 2:
                                er = np.max(np.abs(xsta[:Dyn.K] - xsta[-Dyn.K:]))
                                print('Error bound circulation previous vs current time-step: %.3e' % er)
                                assert er < 1e-13, \
                                    'Circulation at previous and current time-step not matching'

                        ### Verify gains
                        Dyn.get_total_forces_gain()
                        Dyn.get_sect_forces_gain()

                        # sectional forces - algorithm for surfaces with equal M
                        n_surf = Dyn.MS.n_surf
                        M, N = Dyn.MS.MM[0], Dyn.MS.NN[0]
                        fnodes = ysta.reshape((n_surf, 3, M + 1, N + 1))
                        Fsect_ref = np.zeros((n_surf, 3, N + 1))
                        Msect_ref = np.zeros((n_surf, 3, N + 1))

                        for ss in range(n_surf):
                            for nn in range(N + 1):
                                for mm in range(M + 1):
                                    Fsect_ref[ss, :, nn] += fnodes[ss, :, mm, nn]
                                    arm = Dyn.MS.Surfs[ss].zeta[:, mm, nn] - Dyn.MS.Surfs[ss].zeta[:, M // 2, nn]
                                    Msect_ref[ss, :, nn] += np.cross(arm, fnodes[ss, :, mm, nn])

                        Fsect = np.dot(Dyn.Kfsec, ysta).reshape((n_surf, 3, N + 1))
                        assert np.max(np.abs(Fsect - Fsect_ref)) < 1e-12, \
                            'Error in gains for cross-sectional forces'
                        Msect = np.dot(Dyn.Kmsec, ysta).reshape((n_surf, 3, N + 1))
                        assert np.max(np.abs(Msect - Msect_ref)) < 1e-12, \
                            'Error in gains for cross-sectional forces'

                        # total forces
                        Ftot_ref = np.zeros((3,))
                        for cc in range(3):
                            Ftot_ref[cc] = np.sum(Fsect_ref[:, cc, :])
                        Ftot = np.dot(Dyn.Kftot, ysta)
                        assert np.max(np.abs(Ftot - Ftot_ref)) < 1e-11, \
                            'Error in gains for total forces'

        def test_nondimss_dimss(self):
            """
            Test scaling and unscaling of UVLM
            """

            Sta = self.Sta

            # estimate reference quantities
            Uinf = np.linalg.norm(self.tsdata.u_ext[0][:, 0, 0])
            chord = np.linalg.norm(self.tsdata.zeta[0][:, -1, 0] - self.tsdata.zeta[0][:, 0, 0])
            rho = self.tsdata.rho

            ScalingDict = {'length': .5 * chord,
                           'speed': Uinf,
                           'density': rho}

            # reference
            Dyn0 = Dynamic(self.tsdata,
                           dt=0.05,
                           integr_order=2, RemovePredictor=True,
                           UseSparse=True)
            Dyn0.assemble_ss()

            # scale/unscale
            Dyn1 = Dynamic(self.tsdata,
                           dt=0.05,
                           integr_order=2, RemovePredictor=True,
                           UseSparse=True, ScalingDict=ScalingDict)
            Dyn1.assemble_ss()
            Dyn1.nondimss()
            Dyn1.dimss()
            libss.compare_ss(Dyn0.SS, Dyn1.SS, tol=1e-10)
            assert np.max(np.abs(Dyn0.SS.dt - Dyn1.SS.dt)) < 1e-12 * Dyn0.SS.dt, \
                'Scaling/unscaling of time-step not correct'

        def test_freqresp(self):

            Sta = self.Sta

            # estimate reference quantities
            Uinf = np.linalg.norm(self.tsdata.u_ext[0][:, 0, 0])
            chord = np.linalg.norm(
                self.tsdata.zeta[0][:, -1, 0] - self.tsdata.zeta[0][:, 0, 0])
            rho = self.tsdata.rho

            ScalingDict = {'length': .5 * chord,
                           'speed': Uinf,
                           'density': rho}

            kv = np.linspace(0, .5, 3)

            for use_sparse in [False, True]:
                for remove_predictor in [True, False]:
                    ### ----- Dynamic class
                    Dyn = Dynamic(self.tsdata,
                                  dt=0.05, ScalingDict=ScalingDict,
                                  integr_order=2, RemovePredictor=remove_predictor,
                                  UseSparse=use_sparse)
                    Dyn.assemble_ss()
                    Dyn.nondimss()
                    Yref = libss.freqresp(Dyn.SS, kv)
                    Ydyn = Dyn.freqresp(kv)
                    ermax = np.max(np.abs(Ydyn - Yref))
                    assert ermax < 1e-13, \
                        'Dynamic.freqresp produces too large error (%.2e)!' % ermax

                    ### ----- BlockDynamic class
                    BlockDyn = DynamicBlock(self.tsdata,
                                            dt=0.05,
                                            ScalingDict=ScalingDict,
                                            integr_order=2, RemovePredictor=remove_predictor,
                                            UseSparse=use_sparse)
                    BlockDyn.assemble_ss()
                    BlockDyn.nondimss()
                    Ydyn_block = BlockDyn.freqresp(kv)
                    ermax = np.max(np.abs(Ydyn_block - Yref))
                    assert ermax < 1e-13, \
                        'Dynamic.freqresp produces too large error (%.2e)!' % ermax

                    ### ----- Frequency class
                    Freq = Frequency(self.tsdata, dt=0.05, ScalingDict=ScalingDict,
                                     integr_order=2, RemovePredictor=remove_predictor,
                                     UseSparse=use_sparse)
                    Freq.assemble()
                    Freq.nondimss()
                    Yfreq = Freq.freqresp(kv)
                    ermax = np.max(np.abs(Yfreq - Yref))
                    assert ermax < 1e-13, \
                        'Frequency.freqresp produces too large error (%.2e)!' % ermax

        def test_solve_step(self):

            Sta = self.Sta

            # estimate reference quantities
            Uinf = np.linalg.norm(self.tsdata.u_ext[0][:, 0, 0])
            chord = np.linalg.norm(
                self.tsdata.zeta[0][:, -1, 0] - self.tsdata.zeta[0][:, 0, 0])
            rho = self.tsdata.rho

            ScalingDict = {'length': .5 * chord,
                           'speed': Uinf,
                           'density': rho}

            ### build an input time history
            NT = 5
            Uin = np.random.rand(9 * Sta.Kzeta, NT)

            ### get size of output
            Ny = 3 * Sta.Kzeta
            Ydyn = np.zeros((Ny, NT))
            Yblock = np.zeros((Ny, NT))

            for integr_order in [1, 2]:

                Nx = (1 + integr_order) * Sta.K + Sta.K_star
                Xdyn = np.zeros((Nx, NT))
                Xblock = np.zeros((Nx, NT))

                for use_sparse in [True, False]:
                    for remove_predictor in [True, False]:

                        Xdyn *= 0.
                        Xblock *= 0.
                        Ydyn *= 0.
                        Yblock *= 0.

                        ### ----- Dynamic class
                        Dyn = Dynamic(self.tsdata,
                                      dt=0.05,
                                      ScalingDict=ScalingDict,
                                      integr_order=integr_order, Remove=remove_predictor,
                                      UseSparse=use_sparse)
                        Dyn.assemble_ss()
                        Dyn.nondimss()

                        for tt in range(1, NT):
                            Xdyn[:, tt], Ydyn[:, tt] = \
                                Dyn.solve_step(Xdyn[:, tt - 1], Uin[:, tt - 1],
                                               transform_state=True)

                        ### ----- BlockDynamic class
                        BlockDyn = DynamicBlock(self.tsdata,
                                                dt=0.05,
                                                ScalingDict=ScalingDict,
                                                integr_order=integr_order, RemovePredictor=remove_predictor,
                                                UseSparse=use_sparse)
                        BlockDyn.assemble_ss()
                        BlockDyn.nondimss()

                        for tt in range(1, NT):
                            Xblock[:, tt], Yblock[:, tt] = \
                                BlockDyn.solve_step(Xdyn[:, tt - 1], Uin[:, tt - 1],
                                                    transform_state=True)

                        ermax = np.max(np.abs(Xdyn - Xblock)) / np.max(np.abs(Xdyn))

                        assert ermax < 1e-14, \
                            ('solve_step methods in Dynamic and BlockDynamic not matching ' +
                             ' (relative error %.2e)!' % ermax)

                        ermax = np.max(np.abs(Ydyn - Yblock)) / np.max(np.abs(Ydyn))
                        assert ermax < 1e-14, \
                            ('solve_step methods in Dynamic and BlockDynamic not matching ' +
                             ' (relative error %.2e)!' % ermax)


    unittest.main()
