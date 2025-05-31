"""
Linear aeroelastic model based on coupled GEBM + UVLM
S. Maraniello, Jul 2018
"""

import warnings
import numpy as np

import sharpy.utils.settings
import sharpy.linear.src.linuvlm as linuvlm
import sharpy.linear.src.lingebm as lingebm
import sharpy.linear.src.libss as libss
import sharpy.utils.algebra as algebra


class LinAeroEla():
    r"""

    Future work:
        - settings are converted from string to type in __init__ method.
        - implement all settings of LinUVLM (e.g. support for sparse matrices)

    When integrating in SHARPy:
        * define:
            - self.setting_types
            - self.setting_default
        * use settings.to_custom_types(self.in_dict, self.settings_types, self.settings_default) for conversion to type.

    Args:
        data (sharpy.presharpy.PreSharpy): main SHARPy data class
        settings_linear (dict): optional settings file if they are not included in the ``data`` structure

    Attributes:
        settings (dict): solver settings for the linearised aeroelastic solution
        lingebm (lingebm.FlexDynamic): linearised geometrically exact beam model
        num_dof_str (int): number of structural degrees of freedom
        num_dof_rig (int): number of rigid degrees of freedom
        num_dof_flex (int): number of flexible degrees of freedom (``num_dof_flex+num_dof_rigid=num_dof_str``)
        linuvl (linuvlm.Dynamic): linearised UVLM class
        tsaero (sharpy.utils.datastructures.AeroTimeStepInfo): aerodynamic state timestep info
        tsstr (sharpy.utils.datastructures.StructTimeStepInfo): structural state timestep info
        dt (float): time increment
        q (np.array): corresponding vector of displacements of dimensions ``[1, num_dof_str]``
        dq (np.array): time derivative (:math:`\dot{\mathbf{q}}`) of the corresponding vector of displacements
            with dimensions ``[1, num_dof_str]``
        SS (scipy.signal): state space formulation (discrete or continuous time), as selected by the user
    """

    def __init__(self, data, custom_settings_linear=None, uvlm_block=False, chosen_ts=None):

        self.data = data
        if custom_settings_linear is None:
            settings_here = data.settings['LinearUvlm']
        else:
            settings_here = custom_settings_linear

        sharpy.utils.settings.to_custom_types(settings_here,
                                              linuvlm.settings_types_dynamic,
                                              linuvlm.settings_default_dynamic,
                                              no_ctype=True)

        if chosen_ts is None:
            self.chosen_ts = self.data.ts
        else:
            self.chosen_ts = chosen_ts

        ## TEMPORARY - NEED TO INCLUDE PROPER INTEGRATION OF SETTINGS
        try:
            self.rigid_body_motions = settings_here['rigid_body_motion']
        except KeyError:
            self.rigid_body_motions = False

        try:
            self.use_euler = settings_here['use_euler']
        except KeyError:
            self.use_euler = False

        if self.rigid_body_motions and settings_here['track_body']:
            self.track_body = True
        else:
            self.track_body = False
        ## -------

        ### extract aeroelastic info
        self.dt = settings_here['dt']

        ### reference to timestep_info
        # aero
        aero = data.aero
        self.tsaero = aero.timestep_info[self.chosen_ts]
        # structure
        structure = data.structure
        self.tsstr = structure.timestep_info[self.chosen_ts]

        # --- backward compatibility
        try:
            rho = settings_here['density']
        except KeyError:
            warnings.warn(
                "Key 'density' not found in 'LinearUvlm' solver settings. '\
                                      'Trying to read it from 'StaticCoupled'.")
            rho = data.settings['StaticCoupled']['aero_solver_settings']['rho']
            if type(rho) == str:
                rho = np.float(rho)
 
        self.tsaero.rho = rho
        # --- backward compatibility

        ### gebm
        if self.use_euler:
            self.num_dof_rig = 9
        else:
            self.num_dof_rig = 10

        self.num_dof_flex = np.sum(self.data.structure.vdof >= 0)*6
        self.num_dof_str = self.num_dof_flex + self.num_dof_rig
        self.reshape_struct_input()

        try:
            beam_settings = settings_here['beam_settings']
        except KeyError:
            beam_settings = dict()
        self.lingebm_str = lingebm.FlexDynamic(self.tsstr, structure, beam_settings)

        cga = algebra.quat2rotation(self.tsstr.quat)
        ### uvlm
        if uvlm_block:
            self.linuvlm = linuvlm.DynamicBlock(
                self.tsaero,
                dt=settings_here['dt'],
                dynamic_settings=settings_here,
                RemovePredictor=settings_here['remove_predictor'],
                UseSparse=settings_here['use_sparse'],
                integr_order=settings_here['integr_order'],
                ScalingDict=settings_here['ScalingDict'],
                for_vel=np.hstack((cga.dot(self.tsstr.for_vel[:3]),
                                   cga.dot(self.tsstr.for_vel[3:]))))
        else:
            self.linuvlm = linuvlm.Dynamic(
                self.tsaero,
                dt=settings_here['dt'],
                dynamic_settings=settings_here,
                RemovePredictor=settings_here['remove_predictor'],
                UseSparse=settings_here['use_sparse'],
                integr_order=settings_here['integr_order'],
                ScalingDict=settings_here['ScalingDict'],
                for_vel=np.hstack((cga.dot(self.tsstr.for_vel[:3]),
                                   cga.dot(self.tsstr.for_vel[3:]))))

        # add rotational speed
        # for ii in range(self.linuvlm.MS.n_surf):
        #     self.linuvlm.MS.Surfs[ii].omega = self.tsstr.for_vel[3:]


    def reshape_struct_input(self):
        """ Reshape structural input in a column vector """

        structure = self.data.structure  # self.data.aero.beam
        tsdata = structure.timestep_info[self.chosen_ts]

        self.q = np.zeros(self.num_dof_str)
        self.dq = np.zeros(self.num_dof_str)

        jj = 0  # structural dofs index
        for node_glob in range(structure.num_node):

            ### detect bc at node (and no. of dofs)
            bc_here = structure.boundary_conditions[node_glob]
            if bc_here == 1:  # clamp
                dofs_here = 0
                continue
            elif bc_here == -1 or bc_here == 0:
                dofs_here = 6
                jj_tra = [jj, jj + 1, jj + 2]
                jj_rot = [jj + 3, jj + 4, jj + 5]

            # retrieve element and local index
            ee, node_loc = structure.node_master_elem[node_glob, :]

            # allocate
            self.q[jj_tra] = tsdata.pos[node_glob, :]
            self.q[jj_rot] = tsdata.psi[ee, node_loc]
            # update
            jj += dofs_here

        # allocate FoR A quantities
        if self.use_euler:
            self.q[-9:-3] = tsdata.for_vel
            self.q[-3:] = algebra.quat2euler(tsdata.quat)

            wa = tsdata.for_vel[3:]
            self.dq[-9:-3] = tsdata.for_acc
            T = algebra.deuler_dt(self.q[-3:])
            self.dq[-3:] = T.dot(wa)

        else:
            self.q[-10:-4] = tsdata.for_vel
            self.q[-4:] = tsdata.quat

            wa = tsdata.for_vel[3:]
            self.dq[-10:-4] = tsdata.for_acc
            self.dq[-4] = -0.5 * np.dot(wa, tsdata.quat[1:])

    # self.dq[-3:]=-0.5*(wa*tsdata.quat[0]+np.cross(wa,tsdata.quat[1:]))

    def assemble_ss(self, beam_num_modes=None, wake_prop_settings=None):
        """Assemble State Space formulation"""
        data = self.data

        aero = self.data.aero
        structure = self.data.structure  # data.aero.beam
        tsaero = self.tsaero
        tsstr = self.tsstr

        ### assemble linear uvlm
        self.linuvlm.assemble_ss(wake_prop_settings=wake_prop_settings)
        SSaero = self.linuvlm.SS

        ### assemble gains and stiffening term due to non-zero forces
        # only flexible dof accounted for
        self.get_gebm2uvlm_gains()

        ### assemble linear gebm
        # structural part only
        self.lingebm_str.assemble(beam_num_modes)
        SSstr_flex = self.lingebm_str.SSdisc
        SSstr = SSstr_flex
        # # rigid-body (fake)
        # ZeroMat=np.zeros((self.num_dof_rig,self.num_dof_rig))
        # EyeMat=np.eye(self.num_dof_rig)
        # Astr=np.zeros((2*self.num_dof_rig,2*self.num_dof_rig))
        # Bstr=np.zeros((2*self.num_dof_rig,2*self.num_dof_rig))
        # Cstr=np.eye(2*self.num_dof_rig)
        # Dstr=np.zeros((2*self.num_dof_rig,2*self.num_dof_rig))
        # Astr[:self.num_dof_flex,:self.num_dof_flex]=SSstr.A[]
        # SSstr_rig=scsig.dlti()

        # str -> aero
        Zblock = np.zeros((3 * self.linuvlm.Kzeta, SSstr.outputs // 2))
        if self.rigid_body_motions:
            Kas = np.block([[self.Kdisp, Zblock],
                            [self.Kvel_disp, self.Kvel_vel],
                            [Zblock, Zblock]])
        else:
            Kas = np.block([[self.Kdisp[:, :-self.num_dof_rig], Zblock],
                            [self.Kvel_disp[:, :-self.num_dof_rig], self.Kvel_vel[:, :-self.num_dof_rig]],
                            [Zblock, Zblock]])

        # aero -> str
        if self.rigid_body_motions:
            Ksa = self.Kforces  # aero --> str

        else:
            Ksa = self.Kforces[:-10, :]  # aero --> str

        ### feedback connection
        self.SS = libss.couple(ss01=self.linuvlm.SS, ss02=SSstr, K12=Kas, K21=Ksa)

    def get_gebm2uvlm_gains(self):
        r"""
        Provides:
            - the gain matrices required to connect the linearised GEBM and UVLM
             inputs/outputs
            - the stiffening and damping factors to be added to the linearised
            GEBM equations in order to account for non-zero aerodynamic loads at
            the linearisation point.

        The function produces the gain matrices:

            - ``Kdisp``: gains from GEBM to UVLM grid displacements
            - ``Kvel_disp``: influence of GEBM dofs displacements to UVLM grid
              velocities.
            - ``Kvel_vel``: influence of GEBM dofs displacements to UVLM grid
              displacements.
            - ``Kforces`` (UVLM->GEBM) dimensions are the transpose than the
            Kdisp and Kvel* matrices. Hence, when allocation this term, ``ii``
            and ``jj`` indices will unintuitively refer to columns and rows,
            respectively.

        And the stiffening/damping terms accounting for non-zero aerodynamic
        forces at the linearisation point:

            - ``Kss``: stiffness factor (flexible dof -> flexible dof) accounting
            for non-zero forces at the linearisation point.
            - ``Csr``: damping factor  (rigid dof -> flexible dof)
            - ``Crs``: damping factor (flexible dof -> rigid dof)
            - ``Crr``: damping factor (rigid dof -> rigid dof)


        Stiffening and damping related terms due to the non-zero aerodynamic forces at the linearisation point:

        .. math::
            \mathbf{F}_{A,n} = C^{AG}(\mathbf{\chi})\sum_j \mathbf{f}_{G,j} \rightarrow
            \delta\mathbf{F}_{A,n} = C^{AG}_0 \sum_j \delta\mathbf{f}_{G,j} + \frac{\partial}{\partial\chi}(C^{AG}\sum_j
            \mathbf{f}_{G,j}^0)\delta\chi

        The term multiplied by the variation in the quaternion, :math:`\delta\chi`, couples the forces with the rigid
        body equations and becomes part of :math:`\mathbf{C}_{sr}`.

        Similarly, the linearisation of the moments results in expression that contribute to the stiffness and
        damping matrices.

        .. math::
            \mathbf{M}_{B,n} = \sum_j \tilde{X}_B C^{BA}(\Psi)C^{AG}(\chi)\mathbf{f}_{G,j}

        .. math::
            \delta\mathbf{M}_{B,n} = \sum_j \tilde{X}_B\left(C_0^{BG}\delta\mathbf{f}_{G,j}
            + \frac{\partial}{\partial\Psi}(C^{BA}\delta\mathbf{f}^0_{A,j})\delta\Psi
            + \frac{\partial}{\partial\chi}(C^{BA}_0 C^{AG} \mathbf{f}_{G,j})\delta\chi\right)

        The linearised equations of motion for the geometrically exact beam model take the input term :math:`\delta
        \mathbf{Q}_n = \{\delta\mathbf{F}_{A,n},\, T_0^T\delta\mathbf{M}_{B,n}\}`, which means that the moments
        should be provided as :math:`T^T(\Psi)\mathbf{M}_B` instead of :math:`\mathbf{M}_A = C^{AB}\mathbf{M}_B`,
        where :math:`T(\Psi)` is the tangential operator.

        .. math::
            \delta(T^T\mathbf{M}_B) = T^T_0\delta\mathbf{M}_B
            + \frac{\partial}{\partial\Psi}(T^T\delta\mathbf{M}_B^0)\delta\Psi

        is the linearised expression for the moments, where the first term would correspond to the input terms to the
        beam equations and the second arises due to the non-zero aerodynamic moment at the linearisation point and
        must be subtracted (since it comes from the forces) to form part of :math:`\mathbf{K}_{ss}`. In addition, the
        :math:`\delta\mathbf{M}_B` term depends on both :math:`\delta\Psi` and :math:`\delta\chi`, therefore those
        terms would also contribute to :math:`\mathbf{K}_{ss}` and :math:`\mathbf{C}_{sr}`, respectively.

        The contribution from the total forces and moments will be accounted for in :math:`\mathbf{C}_{rr}` and
        :math:`\mathbf{C}_{rs}`.

        .. math::
            \delta\mathbf{F}_{tot,A} = \sum_n\left(C^{GA}_0 \sum_j \delta\mathbf{f}_{G,j}
            + \frac{\partial}{\partial\chi}(C^{AG}\sum_j
            \mathbf{f}_{G,j}^0)\delta\chi\right)

        Therefore, after running this method, the beam matrices should be updated as:

        >>> K_beam[:flex_dof, :flex_dof] += Kss
        >>> C_beam[:flex_dof, -rigid_dof:] += Csr
        >>> C_beam[-rigid_dof:, :flex_dof] += Crs
        >>> C_beam[-rigid_dof:, -rigid_dof:] += Crr

        Track body option

        The ``track_body`` setting restricts the UVLM grid to linear translation motions and therefore should be used to
        ensure that the forces are computed using the reference linearisation frame.

        The UVLM and beam are linearised about a reference equilibrium condition. The UVLM is defined in the inertial
        reference frame while the beam employs the body attached frame and therefore a projection from one frame onto
        another is required during the coupling process.

        However, the inputs to the UVLM (i.e. the lattice grid coordinates) are obtained from the beam deformation which
        is expressed in A frame and therefore the grid coordinates need to be projected onto the inertial frame ``G``.
        As the beam rotates, the projection onto the ``G`` frame of the lattice grid coordinates will result in a grid
        that is not coincident with that at the linearisation reference and therefore the grid coordinates must be
        projected onto the original frame, which will be referred to as ``U``. The transformation between the inertial
        frame ``G`` and the ``U`` frame is a function of the rotation of the ``A`` frame and the original position:

        .. math:: C^{UG}(\chi) = C^{GA}(\chi_0)C^{AG}(\chi)

        Therefore, the grid coordinates obtained in ``A`` frame and projected onto the ``G`` frame can be transformed
        to the ``U`` frame using

        .. math:: \zeta_U = C^{UG}(\chi) \zeta_G

        which allows the grid lattice coordinates to be projected onto the original linearisation frame.

        In a similar fashion, the output lattice vertex forces of the UVLM are defined in the original linearisation
        frame ``U`` and need to be transformed onto the inertial frame ``G`` prior to projecting them onto the ``A``
        frame to use them as the input forces to the beam system.

        .. math:: \boldsymbol{f}_G = C^{GU}(\chi)\boldsymbol{f}_U

        The linearisation of the above relations lead to the following expressions that have to be added to the
        coupling matrices:

            * ``Kdisp_vel`` terms:

                .. math::
                    \delta\boldsymbol{\zeta}_U= C^{GA}_0 \frac{\partial}{\partial \boldsymbol{\chi}}
                    \left(C^{AG}\boldsymbol{\zeta}_{G,0}\right)\delta\boldsymbol{\chi} + \delta\boldsymbol{\zeta}_G

            * ``Kvel_vel`` terms:

                .. math::
                    \delta\dot{\boldsymbol{\zeta}}_U= C^{GA}_0 \frac{\partial}{\partial \boldsymbol{\chi}}
                    \left(C^{AG}\dot{\boldsymbol{\zeta}}_{G,0}\right)\delta\boldsymbol{\chi}
                    + \delta\dot{\boldsymbol{\zeta}}_G

        The transformation of the forces and moments introduces terms that are functions of the orientation and
        are included as stiffening and damping terms in the beam's matrices:

            * ``Csr`` damping terms relating to translation forces:

                .. math::
                    C_{sr}^{tra} -= \frac{\partial}{\partial\boldsymbol{\chi}}
                    \left(C^{GA} C^{AG}_0 \boldsymbol{f}_{G,0}\right)\delta\boldsymbol{\chi}

            * ``Csr`` damping terms related to moments:

                .. math::
                    C_{sr}^{rot} -= T^\top\widetilde{\mathbf{X}}_B C^{BG}
                    \frac{\partial}{\partial\boldsymbol{\chi}}
                    \left(C^{GA} C^{AG}_0 \boldsymbol{f}_{G,0}\right)\delta\boldsymbol{\chi}


        The ``track_body`` setting.

        When ``track_body`` is enabled, the UVLM grid is no longer coincident with the inertial reference frame
        throughout the simulation but rather it is able to rotate as the ``A`` frame rotates. This is to simulate a free
        flying vehicle, where, for instance, the orientation does not affect the aerodynamics. The UVLM defined in this
        frame of reference, named ``U``, satisfies the following convention:

            * The ``U`` frame is coincident with the ``G`` frame at the time of linearisation.

            * The ``U`` frame rotates as the ``A`` frame rotates.

        Transformations related to the ``U`` frame of reference:

            * The angle between the ``U`` frame and the ``A`` frame is always constant and equal
              to :math:`\boldsymbol{\Theta}_0`.

            * The angle between the ``A`` frame and the ``G`` frame is :math:`\boldsymbol{\Theta}=\boldsymbol{\Theta}_0
              + \delta\boldsymbol{\Theta}`

            * The projection of a vector expressed in the ``G`` frame onto the ``U`` frame is expressed by:

                .. math:: \boldsymbol{v}^U = C^{GA}_0 C^{AG} \boldsymbol{v}^G

            * The reverse, a projection of a vector expressed in the ``U`` frame onto the ``G`` frame, is expressed by

                .. math:: \boldsymbol{v}^U = C^{GA} C^{AG}_0 \boldsymbol{v}^U

        The effect this has on the aeroelastic coupling between the UVLM and the structural dynamics is that the
        orientation and change of orientation of the vehicle has no effect on the aerodynamics. The aerodynamics are
        solely affected by the contribution of the 6-rigid body velocities (as well as the flexible DOFs velocities).

        """

        data = self.data
        aero = self.data.aero
        structure = self.data.structure  # data.aero.beam
        tsaero = self.tsaero
        tsstr = self.tsstr

        # allocate output
        Kdisp = np.zeros((3 * self.linuvlm.Kzeta, self.num_dof_str))
        Kdisp_vel = np.zeros((3 * self.linuvlm.Kzeta, self.num_dof_str))  # Orientation is in velocity DOFs
        Kvel_disp = np.zeros((3 * self.linuvlm.Kzeta, self.num_dof_str))
        Kvel_vel = np.zeros((3 * self.linuvlm.Kzeta, self.num_dof_str))
        Kforces = np.zeros((self.num_dof_str, 3 * self.linuvlm.Kzeta))
        Kforces2 = np.zeros((self.num_dof_str, 3 * self.linuvlm.Kzeta))

        Kss = np.zeros((self.num_dof_flex, self.num_dof_flex))
        Csr = np.zeros((self.num_dof_flex, self.num_dof_rig))
        Crs = np.zeros((self.num_dof_rig, self.num_dof_flex))
        Crr = np.zeros((self.num_dof_rig, self.num_dof_rig))
        Krs = np.zeros((self.num_dof_rig, self.num_dof_flex))

        # get projection matrix A->G
        # (and other quantities indep. from nodal position)
        Cga = algebra.quat2rotation(tsstr.quat)  # NG 6-8-19 removing .T
        Cag = Cga.T

        # for_pos=tsstr.for_pos
        for_vel = tsstr.for_vel[:3]
        for_rot = tsstr.for_vel[3:]
        skew_for_rot = algebra.skew(for_rot)
        Der_vel_Ra = np.dot(Cga, skew_for_rot)

        Faero = np.zeros(3)
        FaeroA = np.zeros(3)

        # GEBM degrees of freedom
        jj_for_tra = range(self.num_dof_str - self.num_dof_rig, self.num_dof_str - self.num_dof_rig + 3)
        jj_for_rot = range(self.num_dof_str - self.num_dof_rig + 3, self.num_dof_str - self.num_dof_rig + 6)

        if self.use_euler:
            jj_euler = range(self.num_dof_str - 3, self.num_dof_str)
            euler = algebra.quat2euler(tsstr.quat)
            tsstr.euler = euler
        else:
            jj_quat = range(self.num_dof_str - 4, self.num_dof_str)

        jj = 0  # nodal dof index
        for node_glob in range(structure.num_node):

            ### detect bc at node (and no. of dofs)
            bc_here = structure.boundary_conditions[node_glob]

            if bc_here == 1:  # clamp (only rigid-body)
                dofs_here = 0
                jj_tra, jj_rot = [], []
            # continue

            elif bc_here == -1 or bc_here == 0:  # (rigid+flex body)
                dofs_here = 6
                jj_tra = 6 * structure.vdof[node_glob] + np.array([0, 1, 2], dtype=int)
                jj_rot = 6 * structure.vdof[node_glob] + np.array([3, 4, 5], dtype=int)
            # jj_tra=[jj  ,jj+1,jj+2]
            # jj_rot=[jj+3,jj+4,jj+5]
            else:
                raise NameError('Invalid boundary condition (%d) at node %d!' \
                                % (bc_here, node_glob))

            jj += dofs_here

            # retrieve element and local index
            ee, node_loc = structure.node_master_elem[node_glob, :]

            # get position, crv and rotation matrix
            Ra = tsstr.pos[node_glob, :]  # in A FoR, w.r.t. origin A-G
            Rg = np.dot(Cag.T, Ra)  # in G FoR, w.r.t. origin A-G
            psi = tsstr.psi[ee, node_loc, :]
            psi_dot = tsstr.psi_dot[ee, node_loc, :]
            Cab = algebra.crv2rotation(psi)
            Cba = Cab.T
            Cbg = np.dot(Cab.T, Cag)
            Tan = algebra.crv2tan(psi)

            track_body = self.track_body

            ### str -> aero mapping
            # some nodes may be linked to multiple surfaces...
            for str2aero_here in aero.struct2aero_mapping[node_glob]:

                # detect surface/span-wise coordinate (ss,nn)
                nn, ss = str2aero_here['i_n'], str2aero_here['i_surf']
                # print('%.2d,%.2d'%(nn,ss))

                # surface panelling
                M = aero.aero_dimensions[ss][0]
                N = aero.aero_dimensions[ss][1]

                Kzeta_start = 3 * sum(self.linuvlm.MS.KKzeta[:ss])
                shape_zeta = (3, M + 1, N + 1)

                for mm in range(M + 1):
                    # get bound vertex index
                    ii_vert = [Kzeta_start + np.ravel_multi_index(
                        (cc, mm, nn), shape_zeta) for cc in range(3)]

                    # get position vectors
                    zetag = tsaero.zeta[ss][:, mm, nn]  # in G FoR, w.r.t. origin A-G
                    zetaa = np.dot(Cag, zetag)  # in A FoR, w.r.t. origin A-G
                    Xg = zetag - Rg  # in G FoR, w.r.t. origin B
                    Xb = np.dot(Cbg, Xg)  # in B FoR, w.r.t. origin B

                    # get rotation terms
                    Xbskew = algebra.skew(Xb)
                    XbskewTan = np.dot(Xbskew, Tan)

                    # get velocity terms
                    zetag_dot = tsaero.zeta_dot[ss][:, mm, nn] - Cga.dot(for_vel)  # in G FoR, w.r.t. origin A-G
                    zetaa_dot = np.dot(Cag, zetag_dot)  # in A FoR, w.r.t. origin A-G

                    # get aero force
                    faero = tsaero.forces[ss][:3, mm, nn]
                    Faero += faero
                    faero_a = np.dot(Cag, faero)
                    FaeroA += faero_a
                    maero_g = np.cross(Xg, faero)
                    maero_b = np.dot(Cbg, maero_g)

                    ### ---------------------------------------- allocate Kdisp

                    if bc_here != 1:
                        # wrt pos - Eq 25 second term
                        Kdisp[np.ix_(ii_vert, jj_tra)] += Cga

                        # wrt psi - Eq 26
                        Kdisp[np.ix_(ii_vert, jj_rot)] -= np.dot(Cbg.T, XbskewTan)

                    # w.r.t. position of FoR A (w.r.t. origin G)
                    # null as A and G have always same origin in SHARPy

                    # # ### w.r.t. quaternion (attitude changes)
                    if self.use_euler:
                        Kdisp_vel[np.ix_(ii_vert, jj_euler)] += \
                            algebra.der_Ceuler_by_v(tsstr.euler, zetaa)

                        # Track body - project inputs as for A not moving
                        if track_body:
                            Kdisp_vel[np.ix_(ii_vert, jj_euler)] += \
                                Cga.dot(algebra.der_Peuler_by_v(tsstr.euler, zetag))
                    else:
                        # Equation 25
                        # Kdisp[np.ix_(ii_vert, jj_quat)] += \
                        #     algebra.der_Cquat_by_v(tsstr.quat, zetaa)
                        Kdisp_vel[np.ix_(ii_vert, jj_quat)] += \
                            algebra.der_Cquat_by_v(tsstr.quat, zetaa)

                        # Track body - project inputs as for A not moving
                        if track_body:
                            Kdisp_vel[np.ix_(ii_vert, jj_quat)] += \
                                Cga.dot(algebra.der_CquatT_by_v(tsstr.quat, zetag))

                    ### ------------------------------------ allocate Kvel_disp

                    if bc_here != 1:
                        # # wrt pos
                        Kvel_disp[np.ix_(ii_vert, jj_tra)] += Der_vel_Ra

                        # wrt psi (at zero psi_dot)
                        Kvel_disp[np.ix_(ii_vert, jj_rot)] -= \
                            np.dot(Cga,
                                   np.dot(skew_for_rot,
                                          np.dot(Cab, XbskewTan)))

                        # # wrt psi (psi_dot contributions - verified)
                        Kvel_disp[np.ix_(ii_vert, jj_rot)] += np.dot(Cbg.T, np.dot(
                            algebra.skew(np.dot(XbskewTan, psi_dot)), Tan))

                        if np.linalg.norm(psi) >=1e-6:
                            Kvel_disp[np.ix_(ii_vert, jj_rot)] -= \
                                np.dot(Cbg.T,
                                       np.dot(Xbskew,
                                              algebra.der_Tan_by_xv(psi, psi_dot)))

                    # # w.r.t. position of FoR A (w.r.t. origin G)
                    # # null as A and G have always same origin in SHARPy

                    # # ### w.r.t. quaternion (attitude changes) - Eq 30
                    if self.use_euler:
                        Kvel_vel[np.ix_(ii_vert, jj_euler)] += \
                            algebra.der_Ceuler_by_v(tsstr.euler, zetaa_dot)

                        # Track body if ForA is rotating
                        if track_body:
                            Kvel_vel[np.ix_(ii_vert, jj_euler)] += \
                                Cga.dot(algebra.der_Peuler_by_v(tsstr.euler, zetag_dot))
                    else:
                        Kvel_vel[np.ix_(ii_vert, jj_quat)] += \
                            algebra.der_Cquat_by_v(tsstr.quat, zetaa_dot)

                        # Track body if ForA is rotating
                        if track_body:
                            Kvel_vel[np.ix_(ii_vert, jj_quat)] += \
                                Cga.dot(algebra.der_CquatT_by_v(tsstr.quat, zetag_dot))

                    ### ------------------------------------- allocate Kvel_vel

                    if bc_here != 1:
                        # wrt pos_dot
                        Kvel_vel[np.ix_(ii_vert, jj_tra)] += Cga

                        # # wrt crv_dot
                        Kvel_vel[np.ix_(ii_vert, jj_rot)] -= np.dot(Cbg.T, XbskewTan)

                    # # wrt velocity of FoR A
                    Kvel_vel[np.ix_(ii_vert, jj_for_tra)] += Cga
                    Kvel_vel[np.ix_(ii_vert, jj_for_rot)] -= \
                        np.dot(Cga, algebra.skew(zetaa))

                    # wrt rate of change of quaternion: not implemented!

                    ### -------------------------------------- allocate Kforces

                    if bc_here != 1:
                        # nodal forces
                        Kforces[np.ix_(jj_tra, ii_vert)] += Cag

                        # nodal moments
                        Kforces[np.ix_(jj_rot, ii_vert)] += \
                            np.dot(Tan.T, np.dot(Cbg, algebra.skew(Xg)))
                    # or, equivalently, np.dot( algebra.skew(Xb),Cbg)

                    # total forces
                    Kforces[np.ix_(jj_for_tra, ii_vert)] += Cag
                    Kforces2[-10:-7, ii_vert] += Cag

                    # total moments
                    Kforces[np.ix_(jj_for_rot, ii_vert)] += \
                        np.dot(Cag, algebra.skew(zetag))

                    # quaternion equation
                    # null, as not dep. on external forces

                    ### --------------------------------------- allocate Kstiff

                    ### flexible dof equations (Kss and Csr)
                    if bc_here != 1:
                        # nodal forces
                        if self.use_euler:
                            if not track_body:
                                Csr[jj_tra, -3:] -= algebra.der_Peuler_by_v(tsstr.euler, faero)
                                # Csr[jj_tra, -3:] -= algebra.der_Ceuler_by_v(tsstr.euler, Cga.T.dot(faero))

                        else:
                            if not track_body:
                                Csr[jj_tra, -4:] -= algebra.der_CquatT_by_v(tsstr.quat, faero)

                            # Track body
                            # if track_body:
                            #     Csr[jj_tra, -4:] -= algebra.der_Cquat_by_v(tsstr.quat, Cga.T.dot(faero))

                        ### moments
                        TanTXbskew = np.dot(Tan.T, Xbskew)
                        # contrib. of TanT (dpsi) - Eq 37 - Integration of UVLM and GEBM
                        Kss[np.ix_(jj_rot, jj_rot)] -= algebra.der_TanT_by_xv(psi, maero_b)
                        # contrib of delta aero moment (dpsi) - Eq 36
                        Kss[np.ix_(jj_rot, jj_rot)] -= \
                            np.dot(TanTXbskew, algebra.der_CcrvT_by_v(psi, np.dot(Cag, faero)))
                        # contribution of delta aero moment (dquat)
                        if self.use_euler:
                            if not track_body:
                                Csr[jj_rot, -3:] -= \
                                    np.dot(TanTXbskew,
                                           np.dot(Cba,
                                                  algebra.der_Peuler_by_v(tsstr.euler, faero)))

                            # if track_body:
                            #     Csr[jj_rot, -3:] -= \
                            #         np.dot(TanTXbskew,
                            #                np.dot(Cbg,
                            #                       algebra.der_Peuler_by_v(tsstr.euler, Cga.T.dot(faero))))
                        else:
                            if not track_body:
                                Csr[jj_rot, -4:] -= \
                                    np.dot(TanTXbskew,
                                           np.dot(Cba,
                                                  algebra.der_CquatT_by_v(tsstr.quat, faero)))

                            # Track body
                            # if track_body:
                            #     Csr[jj_rot, -4:] -= \
                            #         np.dot(TanTXbskew,
                            #                np.dot(Cbg,
                            #                       algebra.der_CquatT_by_v(tsstr.quat, Cga.T.dot(faero))))

                    ### rigid body eqs (Crs and Crr)

                    if bc_here != 1:
                        # Changed Crs to Krs - NG 14/5/19
                        # moments contribution due to delta_Ra (+ sign intentional)
                        Krs[3:6, jj_tra] += algebra.skew(faero_a)
                        # moment contribution due to delta_psi (+ sign intentional)
                        Krs[3:6, jj_rot] += np.dot(algebra.skew(faero_a),
                                                   algebra.der_Ccrv_by_v(psi, Xb))

                    if self.use_euler:
                        # total force
                        if not track_body:
                            Crr[:3, -3:] -= algebra.der_Peuler_by_v(tsstr.euler, faero)

                            # total moment contribution due to change in euler angles
                            Crr[3:6, -3:] -= algebra.der_Peuler_by_v(tsstr.euler, np.cross(zetag, faero))
                            Crr[3:6, -3:] += np.dot(
                                np.dot(Cag, algebra.skew(faero)),
                                algebra.der_Peuler_by_v(tsstr.euler, np.dot(Cab, Xb)))

                            # if track_body:
                            #     # NG 20/8/19 - is the Cag needed here? Verify
                            #     Crr[:3, -3:] -= Cag.dot(algebra.der_Ceuler_by_v(tsstr.euler, Cga.T.dot(faero)))
                            #
                            #     Crr[3:6, -3:] -= Cag.dot(algebra.skew(zetag).dot(algebra.der_Ceuler_by_v(tsstr.euler, Cga.T.dot(faero))))
                            #     Crr[3:6, -3:] += Cag.dot(algebra.skew(faero)).dot(algebra.der_Ceuler_by_v(tsstr.euler, Cga.T.dot(zetag)))

                    else:
                        if not track_body:
                            # total force
                            Crr[:3, -4:] -= algebra.der_CquatT_by_v(tsstr.quat, faero)

                            # total moment contribution due to quaternion
                            Crr[3:6, -4:] -= algebra.der_CquatT_by_v(tsstr.quat, np.cross(zetag, faero))
                            Crr[3:6, -4:] += np.dot(
                                np.dot(Cag, algebra.skew(faero)),
                                algebra.der_CquatT_by_v(tsstr.quat, np.dot(Cab, Xb)))

                        # Track body
                        # if track_body:
                        #     Crr[:3, -4:] -= Cag.dot(algebra.der_Cquat_by_v(tsstr.quat, Cga.T.dot(faero)))
                        #
                        #     Crr[3:6, -4:] -= Cag.dot(algebra.skew(zetag).dot(algebra.der_Cquat_by_v(tsstr.quat, Cga.T.dot(faero))))
                        #     Crr[3:6, -4:] += Cag.dot(algebra.skew(faero)).dot(algebra.der_Cquat_by_v(tsstr.quat, Cga.T.dot(zetag)))


        # transfer
        self.Kdisp = Kdisp
        self.Kvel_disp = Kvel_disp
        self.Kdisp_vel = Kdisp_vel
        self.Kvel_vel = Kvel_vel
        self.Kforces = Kforces

        # stiffening factors
        self.Kss = Kss
        self.Krs = Krs
        self.Csr = Csr
        self.Crs = Crs
        self.Crr = Crr


if __name__ == '__main__':
    import read
    import configobj

    # select test case
    fname = '/home/sm6110/git/uvlm3d/test/h5input/smith_Nsurf01M04N12wk10_a040.state.h5'
    fname = '/home/sm6110/git/uvlm3d/test/h5input/smith_Nsurf02M04N12wk10_a040.state.h5'
    hd = read.h5file(fname)

    # read some setting
    file_settings = fname[:-8] + 'solver.txt'
    dict_config = configobj.ConfigObj(file_settings)

    # add settings for linear solver
    M, cref = 4., 1.
    Uinf = 25.
    dict_config['LinearUvlm'] = {'dt': cref / M / Uinf,
                                 'integr_order': 2,
                                 'Uref': 1.}

    Sol = AeroElaDyn(tsaero=hd.tsaero00000, tsstr=hd.tsstr00000,
                     aero2str_mapping=hd.aero2str, settings=dict_config)
