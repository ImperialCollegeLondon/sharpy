"""
Linearise UVLM solver
S. Maraniello, 7 Jun 2018
"""

import numpy as np
import scipy.linalg as scalg
import scipy.signal as scsig
import scipy.sparse as sparse

# # from IPython import embed
import time
import warnings

import sharpy.linear.src.interp as interp
import sharpy.linear.src.multisurfaces as multisurfaces
import sharpy.linear.src.assembly as ass  # :D
import sharpy.linear.src.libss as libss 
import sharpy.linear.src.libsparse as libsp 


import sharpy.utils.algebra as algebra


class Static():
    """	Static linear solver """

    def __init__(self, tsdata):

        print('Initialising Static linear UVLM solver class...')
        t0 = time.time()

        MS = multisurfaces.MultiAeroGridSurfaces(tsdata)
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

        # profiling output
        self.prof_out = './asbly.prof'

        self.time_init_sta = time.time() - t0
        print('\t\t\t...done in %.2f sec' % self.time_init_sta)


    def assemble_profiling(self):
        '''
        Generate profiling report for assembly and save it in self.prof_out.

        To read the report:
            import pstats
            p=pstats.Stats(self.prof_out)
        '''

        import cProfile
        cProfile.runctx('self.assemble()', globals(), locals(), filename=self.prof_out)


    def assemble(self):
        """
        Assemble global matrices
        """
        print('Assembly of static linear UVLM equations started...')
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
        List_nc_domegazetadzeta_vert = ass.nc_domegazetadzeta(MS.Surfs,MS.Surfs_star)
        self.Ducdzeta+=scalg.block_diag(*List_nc_domegazetadzeta_vert)
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
        print('\t\t\t...done in %.2f sec' % self.time_asbly)


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
        print('Solution done in %.2f sec' % self.time_sol)


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
        """
        Calculates gain matrices to calculate the total force (Kftot) and moment
        (Kmtot, Kmtot_disp) about the pole zeta_pole.

        Being :math:`f` and :math:`\\zeta` the force and position at the vertex (m,n) of the lattice
        these are produced as:

            ftot=sum(f) 					=> dftot += df
            mtot-sum((zeta-zeta_pole) x f)	=>
                    => 	dmtot +=  cross(zeta0-zeta_pole) df - cross(f0) dzeta

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
        self.Kfsec = np.zeros((6 * Ntot, 3 * self.Kzeta))

        Kzeta_start = 0
        II_start = 0
        for ss in range(self.MS.n_surf):
            M, N = self.MS.MM[ss], self.MS.NN[ss]

            for nn in range(N + 1):
                zeta_sec = self.MS.Surfs[ss].zeta[:, :, nn]
                
                # section indices
                iivec = [II_start + np.ravel_multi_index((cc, nn),
                                                            (6, N + 1)) for cc in range(6)] 

                for mm in range(M + 1):
                    # vertex indices
                    jjvec = [Kzeta_start + np.ravel_multi_index((cc, mm, nn),
                                                                (3, M + 1, N + 1)) for cc in range(3)]

                    # sectional force
                    self.Kfsec[iivec[:3], jjvec] = 1.0

                    # sectional moment
                    dx, dy, dz = zeta_sec[:, mm] - zeta_sec[:, M // 2]
                    self.Kfsec[np.ix_(iivec[3:], jjvec)] = np.array([[0, -dz, dy],
                                                                     [dz, 0, -dx],
                                                                     [-dy, dx, 0]])
            Kzeta_start += 3 * self.MS.KKzeta[ss]
            II_start += 6*(N+1)


    def get_rigid_motion_gains(self, zeta_rotation=np.zeros((3,))):
        """
        Gains to reproduce rigid-body motion such that grid displacements and
        velocities are given by:
            dzeta     = Ktra*u_tra         + Krot*u_rot
            dzeta_dot = Ktra_vel*u_tra_dot + Krot*u_rot_dot

        Rotations are assumed to happen independently with respect to the
        zeta_rotation point and about the x,y and z axes of the inertial frame.
        """

        warnings.warn('Rigid rotation matrix not implemented!')

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


# ------------------------------------------------------------------------------


class Dynamic(Static):

    def __init__(self, tsdata, dt, integr_order=2, 
                       RemovePredictor=True, ScalingDict=None, UseSparse=False):

        super().__init__(tsdata)

        # self.settings_types = dict()
        # self.settings_default = dict()

        # # dimensional time-step 
        # self.settings_types['dt'] = 'float'
        # self.settings_default['dt'] = 0.1

        # # integration order for bound circulation first derivative (unsteady force)   
        # self.settings_types['integr_order'] = 'int'
        # self.settings_default['integr_order'] = 2

        self.dt = dt
        self.integr_order = integr_order

        if self.integr_order == 1:
            Nx = 2 * self.K + self.K_star
        elif self.integr_order == 2:
            Nx = 3 * self.K + self.K_star
            b0, bm1, bp1 = -2., 0.5, 1.5
        else:
            raise NameError('Only integration orders 1 and 2 are supported')
        Ny = 3 * self.Kzeta
        Nu = 3 * Ny
        self.Nx = Nx
        self.Nu = Nu
        self.Ny = Ny

        self.remove_predictor = RemovePredictor
        self.include_added_mass = True
        self.use_sparse=UseSparse

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

    # print('Initialising Dynamic solver class...')
    # t0=time.time()
    # self.time_init_dyn=time.time()-t0
    # print('\t\t\t...done in %.2f sec' %self.time_init_dyn)


    def nondimss(self):
        """
        Scale state-space model based of self.SalingFacts
        """

        self.state_scal = self.ScalingFacts['circulation']
        self.output_scal = self.ScalingFacts['force']

        Kzeta = self.Kzeta
        self.input_scal = np.concatenate((
            3 * Kzeta * [self.ScalingFacts['length']],
            6 * Kzeta * [self.ScalingFacts['speed']]))
        self.SS = libss.scale_SS(
            self.SS, self.input_scal, self.output_scal, self.state_scal)
        self.SS.dt = self.SS.dt / self.ScalingFacts['time']


    def dimss(self):
        pass


    def assemble_ss(self):
        r"""
        Produces state-space model of the form

            .. math::

                \mathbf{x}_{n+1} &= \mathbf{A}\,\mathbf{x}_n + \mathbf{B} \mathbf{u}_{n+1} \\
                \mathbf{y} &= \mathbf{C}\,\mathbf{x} + \mathbf{D} \mathbf{u}


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

        print('State-space realisation of UVLM equations started...')
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
        A0, A0W  = None, None

        ### propagation of circ
        # fast and memory efficient with both dense and sparse matrices
        List_C, List_Cstar = ass.wake_prop(MS.Surfs, MS.Surfs_star, 
                                            self.use_sparse,sparse_format='csc')
        if self.use_sparse:
            Cgamma = libsp.csc_matrix(sparse.block_diag(List_C,format='csc'))
            CgammaW = libsp.csc_matrix(sparse.block_diag(List_Cstar,format='csc'))  
        else:
            Cgamma = scalg.block_diag(*List_C)
            CgammaW = scalg.block_diag(*List_Cstar)
        List_C, List_Cstar  = None, None

        # recurrent dense terms stored as numpy.ndarrays
        AinvAWCgamma = -libsp.dot(AinvAW, Cgamma)
        AinvAWCgammaW= -libsp.dot(AinvAW, CgammaW)

        ### A matrix assembly
        if self.use_sparse:
            # lil format allows fast assembly
            Ass = sparse.lil_matrix((Nx,Nx))
        else:
            Ass = np.zeros((Nx, Nx))
        Ass[:K, :K] = AinvAWCgamma
        Ass[:K, K:K + K_star] = AinvAWCgammaW
        Ass[K:K + K_star, :K] = Cgamma
        Ass[K:K + K_star, K:K + K_star] = CgammaW
        Cgamma,CgammaW = None,None

        # delta eq.
        iivec=range(K + K_star,2 * K + K_star)
        ones=np.ones((K,))
        if self.integr_order == 1:
            Ass[iivec, :K] = AinvAWCgamma
            Ass[iivec, range(K)] -= ones
            Ass[iivec, K:K+K_star] = AinvAWCgammaW
        if self.integr_order == 2:
            Ass[iivec, :K] = bp1 * AinvAWCgamma
            AinvAWCgamma=None
            Ass[iivec, range(K)] += b0*ones
            Ass[iivec, K:K+K_star] = bp1 * AinvAWCgammaW
            AinvAWCgammaW=None
            Ass[iivec, range(2*K+K_star,3*K+K_star)] = bm1*ones
            # identity eq.
            Ass[range(2*K+K_star,3*K+K_star), range(K)] = ones

        if self.use_sparse:
            # conversion to csc occupies less memory and allows fast algebra
            Ass = libsp.csc_matrix(Ass)

        # zeta derivs
        List_nc_dqcdzeta=ass.nc_dqcdzeta(MS.Surfs, MS.Surfs_star,Merge=True)
        List_uc_dncdzeta = ass.uc_dncdzeta(MS.Surfs)
        List_nc_domegazetadzeta_vert = ass.nc_domegazetadzeta(MS.Surfs,MS.Surfs_star)
        for ss in range(MS.n_surf):
            List_nc_dqcdzeta[ss][ss]+=\
                     ( List_uc_dncdzeta[ss] + List_nc_domegazetadzeta_vert[ss] )
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
            Bss=sparse.lil_matrix((Nx,Nu))
        else:
            Bss = np.zeros((Nx, Nu))

        Bup=np.block([ -scalg.lu_solve((LU,P),Ducdzeta), AinvWnv0, -AinvWnv0 ])
        AinvWnv0=None
        Bss[:K,:] = Bup
        if self.integr_order == 1:
            Bss[K + K_star:2 * K + K_star, :] = Bup
        if self.integr_order == 2:
            Bss[K + K_star:2 * K + K_star, :] = bp1 * Bup
        Bup=None

        if self.use_sparse:
            Bss=libsp.csc_matrix(Bss)
        LU,P=None,None

        # ---------------------------------------------------------- output eq.

        ### state terms (C matrix)

        # gamma (induced velocity contrib.)
        List_dfqsdvind_gamma, List_dfqsdvind_gamma_star = \
            ass.dfqsdvind_gamma(MS.Surfs, MS.Surfs_star)

        # gamma (at constant relative velocity)
        List_dfqsdgamma_vrel0, List_dfqsdgamma_star_vrel0 = \
            ass.dfqsdgamma_vrel0(MS.Surfs, MS.Surfs_star)
        for ss in range(MS.n_surf):
            List_dfqsdvind_gamma[ss][ss]+=List_dfqsdgamma_vrel0[ss]
            List_dfqsdvind_gamma_star[ss][ss]+=List_dfqsdgamma_star_vrel0[ss]                
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

        # input velocities (body moviment)
        if self.include_added_mass:
            Dss[:, 3 * Kzeta:6 * Kzeta] = -Dss[:, 6 * Kzeta:9 * Kzeta]

        if self.remove_predictor:
            Ass, Bmod, Css, Dmod = \
                libss.SSconv(Ass, None, Bss, Css, Dss, Bm1=None)
            self.SS = libss.ss(Ass, Bmod, Css, Dmod, dt=self.dt)
            print('state-space model produced in form:\n\t' \
                  'h_{n+1} = A h_{n} + B u_{n}\n\t' \
                  'with:\n\tx_n = h_n + Bp u_n')
        else:
            self.SS = libss.ss(Ass, Bss, Css, Dss, dt=self.dt)
            print('state-space model produced in form:\n\t' \
                  'x_{n+1} = A x_{n} + Bp u_{n+1}')

        self.time_ss = time.time() - t0
        print('\t\t\t...done in %.2f sec' % self.time_ss)


    def assemble_ss_profiling(self):
        '''
        Generate profiling report for assembly and save it in self.prof_out.

        To read the report:
            import pstats
            p=pstats.Stats(self.prof_out)
        '''

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

        if self.use_sparse is True and method!= 'direct':
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
            xsta=libsp.solve(libsp.eye_as(Ass)-Ass,Bss.dot(usta))
            ysta = np.dot(Css, xsta) + np.dot(Dss, usta)


        elif method == 'recursive':
            """ Proovides steady-state solution solving for impulsive response """
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
                print('Solution not found! Max. iterations reached with error: %.3e' % er)


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



    def solve_step(self, xold, uvec):
        r"""
        Solve step.

        If the predictor has been removed (``remove_predictor = True``) then the system is solved as:

            .. math::
                \mathbf{x}^{n+1} &= \mathbf{A\,x}^n + \mathbf{B\,u}^n \\
                \mathbf{y}^{n+1} &= \mathbf{C\,x}^{n+1} + \mathbf{D\,u}^n

        Else, if ``remove_predictor = False``:

            .. math::
                \mathbf{x}^{n+1} &= \mathbf{A\,x}^n + \mathbf{B\,u}^{n+1} \\
                \mathbf{y}^{n+1} &= \mathbf{C\,x}^{n+1} + \mathbf{D\,u}^{n+1}

        where the modifications to the :math:`\mathbf{B}` and :math:`\mathbf{D}` are detailed in
        :func:`Dynamic.assemble_ss`

        Args:
            xold (np.array): State vector at the current timestep :math:`\mathbf{x}^n`
            uvec (np.array): Input vector at timestep :math:`\mathbf{u}^n` or :math:`\mathbf{u}^{n+1}`, depending
                on whether the predictor term is removed or not.

        Returns:
            (np.array, np.array): Updated state and output vector :math:`\{\mathbf{x}^{n+1},\,\mathbf{y}^{n+1}\}`

        Notes:
            To speed-up the solution and use minimal memory:
                - solve for bound vorticity (and)
                - propagate the wake
                - compute the output separately.
        """

        xnew = self.SS.A.dot(xold) + self.SS.B.dot(uvec)
        ynew = np.dot(self.SS.C, xnew) + np.dot(self.SS.D, uvec)

        return xnew, ynew



    def unpack_state(self, xvec):

        K, K_star = self.K, self.K_star
        gamma_vec = xvec[:K]
        gamma_star_vec = xvec[K:K + K_star]
        gamma_dot_vec = xvec[K + K_star:2 * K + K_star]

        return gamma_vec, gamma_star_vec, gamma_dot_vec




################################################################################


if __name__ == '__main__':

    import unittest
    from sharpy.utils.sharpydir import SharpyDir
    import sharpy.utils.h5utils as h5
    import matplotlib.pyplot as plt


    class Test_linuvlm_Sta_vs_Dyn(unittest.TestCase):
        ''' Test methods into this module '''

        def setUp(self):
            fname = SharpyDir+'/sharpy/linear/test/h5input/'+\
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
            self.Sta=Sta
            self.tsdata=tsdata


        def test_force_gains(self):
            '''
            to do: add check on moments gain
            '''
            Sta=self.Sta
            Ftot02 = libsp.dot(Sta.Kftot, Sta.fqs)
            assert np.max(np.abs(Ftot02-Sta.Ftot)) < 1e-10, 'Total force gain matrix wrong!'


        def test_Dyn_steady_state(self):
            '''
            Test steady state predicted by Dynamic and Static classes are the same.
            '''

            Sta=self.Sta
            Order=[2,1]
            RemPred=[True,False]
            UseSparse=[True,False]

            for order in Order:
                for rem_pred in RemPred:
                    for use_sparse in UseSparse:

                        # Dynamic solver
                        Dyn = Dynamic(  self.tsdata, 
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
                        assert er < 1e-12,\
                             'Steady-state force not matching (error: %.2e)!'%er

                        if rem_pred is False: # compare state

                            er = np.max(np.abs(xsta[:Dyn.K] - Sta.gamma))
                            print('Error bound circulation: %.3e' % er)
                            assert er < 1e-13,\
                                 'Steady-state gamma not matching (error: %.2e)!'%er

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
                        Fsect_ref = np.zeros((n_surf, 6, N + 1))
                        for ss in range(n_surf):
                            for nn in range(N + 1):
                                for mm in range(M + 1):
                                    Fsect_ref[ss, :3, nn] += fnodes[ss, :, mm, nn]
                                    arm = Dyn.MS.Surfs[ss].zeta[:, mm, nn] - Dyn.MS.Surfs[ss].zeta[:, M // 2, nn]
                                    Fsect_ref[ss, 3:, nn] += np.cross(arm, fnodes[ss, :, mm, nn])

                        Fsect = np.dot(Dyn.Kfsec, ysta).reshape((n_surf, 6, N + 1))
                        assert np.max(np.abs(Fsect - Fsect_ref)) < 1e-12, \
                                      'Error in gains for cross-sectional forces'

                        # total forces
                        Ftot_ref = np.zeros((3,))
                        for cc in range(3):
                            Ftot_ref[cc] = np.sum(Fsect_ref[:, cc, :])
                        Ftot = np.dot(Dyn.Kftot, ysta)
                        assert np.max(np.abs(Ftot - Ftot_ref)) < 1e-11,\
                                                'Error in gains for total forces'

    unittest.main()