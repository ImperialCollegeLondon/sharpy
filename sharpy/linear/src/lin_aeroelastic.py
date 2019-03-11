"""
Linear aeroelastic model based on coupled GEBM + UVLM
S. Maraniello, Jul 2018
"""

import warnings
import numpy as np
import scipy.signal as scsig

import sharpy.utils.settings
import sharpy.linear.src.linuvlm as linuvlm
import sharpy.linear.src.lingebm as lingebm
import sharpy.linear.src.libss as libss
import sharpy.utils.algebra as algebra


class LinAeroEla():
    """
    todo: 
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
        dq (np.array): time derivative (:math:`\\dot{\\mathbf{q}}`) of the corresponding vector of displacements
            with dimensions ``[1, num_dof_str]``
        SS (scipy.signal): state space formulation (discrete or continuous time), as selected by the user
    """

    def __init__(self, data, custom_settings_linear=None, uvlm_block=False):

        self.data = data
        if custom_settings_linear is None:
            settings_here = data.settings['LinearUvlm']
        else:
            settings_here = custom_settings_linear

        sharpy.utils.settings.to_custom_types(settings_here, 
                                              linuvlm.settings_types_dynamic, 
                                              linuvlm.settings_default_dynamic)

        ### extract aeroelastic info
        self.dt = settings_here['dt'].value

        ### reference to timestep_info
        # aero
        aero = data.aero
        self.tsaero = aero.timestep_info[data.ts]
        # structure
        structure = data.structure
        self.tsstr = structure.timestep_info[data.ts]

        # --- backward compatibility
        try:
            rho = settings_here['density'].value
        except KeyError:
            warnings.warn(
                "Key 'density' not found in 'LinearUvlm' solver settings. '\
                                      'Trying to read it from 'StaticCoupled'.")
            rho = data.settings['StaticCoupled']['aero_solver_settings']['rho']
            if type(rho) == str:
                rho = np.float(rho)
            if hasattr(rho, 'value'):
                rho = rho.value
        self.tsaero.rho = rho
        # --- backward compatibility

        ### gebm
        self.num_dof_str = len(self.tsstr.q)
        self.num_dof_rig = 10
        self.num_dof_flex = self.num_dof_str - self.num_dof_rig
        self.reshape_struct_input()
        self.lingebm_str = lingebm.FlexDynamic(self.tsstr, dt=self.dt)

        ### uvlm
        if uvlm_block:
            self.linuvlm = linuvlm.DynamicBlock(
                self.tsaero,
                dt=settings_here['dt'].value,
                RemovePredictor=settings_here['remove_predictor'].value,
                UseSparse=settings_here['use_sparse'].value,
                integr_order=settings_here['integr_order'].value,
                ScalingDict=settings_here['ScalingDict'])
        else:
            self.linuvlm = linuvlm.Dynamic(
                self.tsaero,
                dt=settings_here['dt'].value,
                RemovePredictor=settings_here['remove_predictor'].value,
                UseSparse=settings_here['use_sparse'].value,
                integr_order=settings_here['integr_order'].value,
                ScalingDict=settings_here['ScalingDict'])

        # add rotational speed
        for ii in range(self.linuvlm.MS.n_surf):
            self.linuvlm.MS.Surfs[ii].omega = self.tsstr.for_vel[3:]


    def reshape_struct_input(self):
        """ Reshape structural input in a column vector """

        structure = self.data.structure  # self.data.aero.beam
        tsdata = structure.timestep_info[self.data.ts]

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
        self.q[-10:-4] = tsdata.for_vel
        self.q[-4:] = tsdata.quat

        wa = tsdata.for_vel[3:]
        self.dq[-10:-4] = tsdata.for_acc
        self.dq[-4] = -0.5 * np.dot(wa, tsdata.quat[1:])


    # self.dq[-3:]=-0.5*(wa*tsdata.quat[0]+np.cross(wa,tsdata.quat[1:]))


    def assemble_ss(self):
        """Assemble State Space formulation"""
        data = self.data

        aero = self.data.aero
        structure = self.data.structure  # data.aero.beam
        tsaero = self.tsaero
        tsstr = self.tsstr

        ### assemble linear uvlm
        self.linuvlm.assemble_ss()
        SSaero = self.linuvlm.SS

        ### assemble gains and stiffening term due to non-zero forces
        # only flexible dof accounted for
        self.get_gebm2uvlm_gains()

        ### assemble linear gebm
        # structural part only
        self.lingebm_str.assemble()
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
        Kas = np.block([[self.Kdisp[:, :-10], Zblock],
                        [self.Kvel_disp[:, :-10], self.Kvel_vel[:, :-10]],
                        [Zblock, Zblock]])

        # aero -> str
        Ksa = self.Kforces[:-10, :]  # aero --> str

        ### feedback connection
        self.SS = libss.couple(ss01=self.linuvlm.SS, ss02=SSstr, K12=Kas, K21=Ksa)


    def get_gebm2uvlm_gains(self):
        """
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

        Note that the stiffening/damping terms need to be added to the GEBM
        stiffness and damping matrices.
        """

        data = self.data
        aero = self.data.aero
        structure = self.data.structure  # data.aero.beam
        tsaero = self.tsaero
        tsstr = self.tsstr

        # allocate output
        Kdisp = np.zeros((3 * self.linuvlm.Kzeta, self.num_dof_str))
        Kvel_disp = np.zeros((3 * self.linuvlm.Kzeta, self.num_dof_str))
        Kvel_vel = np.zeros((3 * self.linuvlm.Kzeta, self.num_dof_str))
        Kforces = np.zeros((self.num_dof_str, 3 * self.linuvlm.Kzeta))

        Kss = np.zeros((self.num_dof_flex, self.num_dof_flex))
        Csr = np.zeros((self.num_dof_flex, self.num_dof_rig))
        Crs = np.zeros((self.num_dof_rig, self.num_dof_flex))
        Crr = np.zeros((self.num_dof_rig, self.num_dof_rig))


        # get projection matrix A->G
        # (and other quantities indep. from nodal position)
        Cga = algebra.quat2rotation(tsstr.quat)
        Cag = Cga.T

        # for_pos=tsstr.for_pos
        for_tra = tsstr.for_vel[:3]
        for_rot = tsstr.for_vel[3:]
        skew_for_rot = algebra.skew(for_rot)
        Der_vel_Ra = np.dot(Cga, skew_for_rot)

        # GEBM degrees of freedom
        jj_for_tra = range(self.num_dof_str - 10, self.num_dof_str - 7)
        jj_for_rot = range(self.num_dof_str - 7, self.num_dof_str - 4)
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
                    zetag_dot = tsaero.zeta_dot[ss][:, mm, nn]  # in G FoR, w.r.t. origin A-G
                    zetaa_dot = np.dot(Cag, zetag_dot)  # in A FoR, w.r.t. origin A-G

                    # get aero force
                    faero = tsaero.forces[ss][:3, mm, nn]
                    faero_a = np.dot(Cag,faero)
                    maero_g = np.cross(Xg,faero)
                    maero_b = np.dot(Cbg,maero_g)


                    ### ---------------------------------------- allocate Kdisp

                    if bc_here != 1:
                        # wrt pos
                        Kdisp[np.ix_(ii_vert, jj_tra)] += Cga

                        # wrt psi
                        Kdisp[np.ix_(ii_vert, jj_rot)] -= np.dot(Cbg.T, XbskewTan)

                    # w.r.t. position of FoR A (w.r.t. origin G)
                    # null as A and G have always same origin in SHARPy

                    # # ### w.r.t. quaternion (attitude changes)
                    Kdisp[np.ix_(ii_vert, jj_quat)] = \
                        algebra.der_Cquat_by_v(tsstr.quat, zetaa)


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

                        Kvel_disp[np.ix_(ii_vert, jj_rot)] -= \
                            np.dot(Cbg.T,
                                   np.dot(Xbskew,
                                          algebra.der_Tan_by_xv(psi, psi_dot)))

                    # # w.r.t. position of FoR A (w.r.t. origin G)
                    # # null as A and G have always same origin in SHARPy

                    # # ### w.r.t. quaternion (attitude changes)
                    Kvel_disp[np.ix_(ii_vert, jj_quat)] = \
                        algebra.der_Cquat_by_v(tsstr.quat, zetaa_dot)


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
                                 np.dot( Tan.T, np.dot(Cbg, algebra.skew(Xg)) )
                    # or, equivalently, np.dot( algebra.skew(Xb),Cbg)

                    # total forces
                    Kforces[np.ix_(jj_for_tra, ii_vert)] += Cag

                    # total moments
                    Kforces[np.ix_(jj_for_rot, ii_vert)] += \
                        np.dot(Cag, algebra.skew(zetag))

                    # quaternion equation
                    # null, as not dep. on external forces


                    ### --------------------------------------- allocate Kstiff

                    ### flexible dof equations (Kss and Csr)
                    if bc_here != 1:
                        # forces
                        Csr[jj_tra, -4:] -= algebra.der_CquatT_by_v(tsstr.quat, faero)

                        ### moments
                        TanTXbskew=np.dot(Tan.T,Xbskew)
                        # contrib. of TanT (dpsi)
                        Kss[np.ix_(jj_rot, jj_rot)]-=algebra.der_TanT_by_xv(psi,maero_b)
                        # contrib of delta aero moment (dpsi)
                        Kss[np.ix_(jj_rot, jj_rot)]-=\
                            np.dot( TanTXbskew, algebra.der_CcrvT_by_v(psi, np.dot(Cag,faero) ))
                        # contribution of delta aero moment (dquat)
                        Csr[jj_rot, -4:] -=\
                                np.dot( TanTXbskew,
                                    np.dot( Cba, 
                                        algebra.der_CquatT_by_v(tsstr.quat,faero)))


                    ### rigid body eqs (Crs and Crr)
    
                    if bc_here != 1:
                        # moments contribution due to delta_Ra (+ sign intentional)
                        Crs[3:6,jj_tra] += algebra.skew( faero_a ) 
                        # moment contribution due to delta_psi (+ sign intentional)
                        Crs[3:6,jj_rot] += np.dot(algebra.skew(faero_a),
                                                  algebra.der_Ccrv_by_v(psi, Xb))

                    # total force
                    Crr[:3,-4:] -= algebra.der_CquatT_by_v(tsstr.quat,faero)

                    # total moment contribution due to quaternion
                    Crr[3:6,-4:] -= algebra.der_CquatT_by_v(tsstr.quat, np.cross(zetag,faero) )
                    Crr[3:6,-4:] += np.dot( 
                                        np.dot(Cag,algebra.skew(faero)), 
                                        algebra.der_Cquat_by_v(tsstr.quat,np.dot(Cab,Xb))) 


        # transfer
        self.Kdisp = Kdisp
        self.Kvel_disp = Kvel_disp
        self.Kvel_vel = Kvel_vel
        self.Kforces = Kforces

        # stiffening factors
        self.Kss = Kss
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
